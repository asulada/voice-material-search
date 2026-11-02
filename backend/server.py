"""FastAPI 应用（方案 §6.2 API 表 + §6.6 本地文件访问边界）。

启动：
    uvicorn server:app --host 127.0.0.1 --port 8000

设计要点：
  * 单进程 + 单任务并发（§6.1）
  * 音频只按数据库里的 filepath 取，且校验仍落在 voice_library/ 内（§6.6）
  * 只监听 127.0.0.1（由启动参数保证）
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voicelib import __version__
from voicelib import audio as A
from voicelib import tags as T
from voicelib.config import Config
from voicelib.db import Database
from voicelib.embedder import Encoder
from voicelib.export import ExportOptions, export_clips, list_exports
from voicelib.ingest import ImportOptions, import_into
from voicelib.jobs import JobManager, JobProgress, stream_job
from voicelib.search import USAGE_RULES, categories_for, row_to_dict, search, search_for
from voicelib.sources import guess_kind

logging.basicConfig(
    level=os.environ.get("VOICE_LIBRARY_LOG", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("voicelib.server")

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


# ----------------------------------------------------------------- 请求模型
# 必须定义在模块级：server.py 开了 `from __future__ import annotations`，
# 定义在 create_app() 内部的模型在运行时无法被 FastAPI 解析，
# 会被误当成 query 参数（表现为「body 字段缺失」的 422）。


class SearchBody(BaseModel):
    use: Literal["indextts", "seedvc", "all"] = "indextts"
    q: str | None = None
    kinds: list[str] | None = None
    categories: list[str] | None = None
    gender: str | None = None
    language: str | None = None
    min_intensity: int | None = None
    duration: list[float] | None = None
    commercial_only: bool = False
    verified_only: bool = False
    top_k: int = Field(default=20, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    sort: Literal["relevance", "quality", "duration", "random"] = "relevance"


class ClipPatch(BaseModel):
    verified: int | None = None
    intensity: int | None = Field(default=None, ge=0, le=5)
    description: str | None = None
    notes: str | None = None
    category: str | None = None
    gender: Literal["male", "female"] | None = None
    has_background: int | None = None
    studio_quality: int | None = None
    transcript: str | None = None
    target_use: Literal["indextts", "seedvc", "both"] | None = None
    license: str | None = None


class ImportBody(BaseModel):
    kind: Literal["csemotions", "emogator", "flat"] = "csemotions"
    path: str
    target_use: Literal["indextts", "seedvc"] = "indextts"
    license: str = "apache-2.0"
    language: str = ""
    limit: int | None = None
    duration: list[float] | None = None
    clip_kind: Literal["speech", "vocal_burst"] | None = None
    overwrite: bool = True
    dry_run: bool = False
    gender_map: dict[str, str] | None = None
    extra: dict[str, Any] = {}


class ExportBody(BaseModel):
    clip_ids: list[int]
    name: str = ""
    apply_gain: bool = False
    flatten: bool = True


class ReembedBody(BaseModel):
    all: bool = False            # True 表示全库重建（耗时长，仍走后台任务）
    missing_only: bool = False   # 只补没有向量的
    description: str | None = None  # 指定新描述并重建单条


# --------------------------------------------------------------- app state


class AppState:
    """应用级状态。目录/数据库/任务队列都在首次访问时创建。

    为什么延迟：`uvicorn server:app` 会在导入时读取模块级 `app`，
    所以 `app = create_app(...)` 必须在导入期执行；但真正的磁盘副作用
    （建 voice_library/、建库）不该在「仅仅 import 了 server」时发生
    —— 那会让跑一次测试就在仓库里留下运行数据。因此构造只做配置解析。
    """

    def __init__(self, root: str | None = None):
        self.cfg = Config.load(root)
        self._db: Database | None = None
        self._jobs: JobManager | None = None
        self.encoder = Encoder(self.cfg.embed_model)

    @property
    def db(self) -> Database:
        if self._db is None:
            self.cfg.ensure_dirs()
            self._db = Database(self.cfg.db_path)
        return self._db

    @property
    def jobs(self) -> JobManager:
        if self._jobs is None:
            self._jobs = JobManager(self.db)
        return self._jobs

    def close(self) -> None:
        if self._jobs is not None:
            self._jobs.shutdown()


def create_app(root: str | None = None) -> FastAPI:
    state = AppState(root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("素材库根目录: %s", state.cfg.root_path)
        log.info("数据库: %s", state.cfg.db_path)
        # 预热 embedding 模型（跑在后台线程，避免阻塞启动）
        asyncio.get_running_loop().run_in_executor(None, state.encoder.get)
        yield
        state.close()

    app = FastAPI(title="语音素材库", version=__version__, lifespan=lifespan)
    app.state.vl = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ helpers

    def _require_clip(clip_id: int):
        row = state.db.get_clip(clip_id)
        if row is None:
            raise HTTPException(404, "素材不存在")
        return row

    def _safe_audio_path(rel: str) -> Path:
        root = state.cfg.root_path
        p = (root / rel).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise HTTPException(403, "路径越界") from None
        if not p.is_file():
            raise HTTPException(404, "文件缺失")
        return p

    # ------------------------------------------------------------------ meta

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "root": str(state.cfg.root_path),
            "db": str(state.cfg.db_path),
            "clips": state.db.count(),
        }

    @app.get("/api/meta")
    def meta():
        """前端启动时一次性拉取：用途规则、模型信息、筛选选项。"""
        enc = state.encoder.get()
        return {
            "uses": [
                {"value": k, "label": v["label"], "kinds": v["kinds"],
                 "duration": list(v["duration"])}
                for k, v in USAGE_RULES.items()
            ],
            "encoder": {"backend": enc.backend, "dim": enc.dim,
                        "model": state.cfg.embed_model},
            "target_lufs": state.cfg.target_lufs,
            "root": str(state.cfg.root_path),
            "version": __version__,
        }

    @app.get("/api/stats")
    def stats():
        data = state.db.stats()
        data["categories"] = {
            use: categories_for(use, state.db) for use in USAGE_RULES
        }
        data["exports"] = len(list_exports(state.cfg))
        return data

    @app.get("/api/config")
    def get_config():
        cfg = state.cfg
        return {
            "root": str(cfg.root_path),
            "embed_model": cfg.embed_model,
            "target_lufs": cfg.target_lufs,
            "indextts_duration": list(cfg.indextts_duration),
            "seedvc_duration": list(cfg.seedvc_duration),
        }

    # ------------------------------------------------------------------ search

    @app.post("/api/search")
    def api_search(body: SearchBody):
        use = body.use
        duration = tuple(body.duration) if body.duration else None

        if use == "all":
            # 浏览全库：不套用途默认约束，靠 kinds/categories 自己筛
            total, scored = search(
                state.db,
                state.encoder,
                use="indextts",
                q=body.q,
                kinds=body.kinds,
                categories=body.categories,
                gender=body.gender,
                language=body.language,
                min_intensity=body.min_intensity,
                duration=duration,
                commercial_only=body.commercial_only,
                verified_only=body.verified_only,
                top_k=body.top_k,
                offset=body.offset,
                sort=body.sort,
            )
        else:
            kwargs: dict[str, Any] = dict(
                categories=body.categories,
                gender=body.gender,
                language=body.language,
                min_intensity=body.min_intensity,
                commercial_only=body.commercial_only,
                verified_only=body.verified_only,
                sort=body.sort,
                offset=body.offset,
            )
            if duration:
                kwargs["duration"] = duration
            if body.kinds:
                kwargs["kinds"] = body.kinds
            total, scored = search_for(
                state.db, state.encoder, use, q=body.q,
                top_k=body.top_k, **kwargs,
            )

        items = [row_to_dict(row, score=score) for score, row in scored]
        return {
            "total": total,
            "returned": len(items),
            "offset": body.offset,
            "items": items,
            "encoder_backend": state.encoder.get().backend,
        }

    @app.get("/api/categories")
    def api_categories(use: str = "indextts"):
        return {"use": use, "categories": categories_for(use, state.db)}

    # ------------------------------------------------------------------ clips

    @app.get("/api/clips/{clip_id}")
    def api_get_clip(clip_id: int):
        return row_to_dict(_require_clip(clip_id))

    @app.get("/api/clips/{clip_id}/audio")
    def api_clip_audio(clip_id: int):
        row = _require_clip(clip_id)
        path = _safe_audio_path(row["filepath"])
        return FileResponse(path)  # 自带 Range 支持

    @app.patch("/api/clips/{clip_id}")
    def api_patch_clip(clip_id: int, body: ClipPatch):
        row = _require_clip(clip_id)
        patch = body.model_dump(exclude_unset=True, exclude_none=True)
        if not patch:
            return {"updated": False, "clip": row_to_dict(row)}

        # license 改了，commercial_ok 必须跟着重算（§4.5）
        if "license" in patch:
            patch["commercial_ok"] = T.commercial_ok_of(patch["license"])

        # 类别改了，描述里的类别也过期了 → 只有用户没同时改描述时才重建
        if "category" in patch and "description" not in patch:
            new_desc = T.build_description(
                clip_kind=row["clip_kind"],
                category=patch["category"],
                transcript=patch.get("transcript", row["transcript"]),
                gender=patch.get("gender", row["gender"]),
                language=row["language"],
                intensity=patch.get("intensity", row["intensity"]),
                duration_sec=row["duration_sec"],
                has_background=row["has_background"] or 0,
                studio_quality=row["studio_quality"] or 0,
            )
            patch["description"] = new_desc

        updated = state.db.update_clip(clip_id, patch)
        row = state.db.get_clip(clip_id)
        if row is None:
            raise HTTPException(404, "素材不存在")

        need_reembed = updated and ("description" in patch or "category" in patch)
        reembedded = False
        if need_reembed and row["description"]:
            vec = state.encoder.encode_one(row["description"])
            state.db.set_embedding(clip_id, state.encoder.to_blob(vec))
            reembedded = True

        out = row_to_dict(row)
        out["reembedded"] = reembedded
        return out

    @app.post("/api/clips/{clip_id}/reebed")
    def api_reembed(clip_id: int, body: ReembedBody | None = None):
        row = _require_clip(clip_id)
        description = (body.description if body else None) or row["description"]
        if not description or not description.strip():
            raise HTTPException(400, "描述为空，无法建向量")
        vec = state.encoder.encode_one(description.strip())
        state.db.set_embedding(clip_id, state.encoder.to_blob(vec), description.strip())
        return {"id": clip_id, "reembedded": True, "description": description.strip()}

    @app.delete("/api/clips/{clip_id}")
    def api_delete_clip(clip_id: int, delete_file: bool = Query(default=False)):
        row = state.db.delete_clip(clip_id)
        if row is None:
            raise HTTPException(404, "素材不存在")
        removed_file = False
        if delete_file:
            try:
                p = _safe_audio_path(row["filepath"])
                p.unlink()
                removed_file = True
            except HTTPException:
                pass
        return {"deleted": clip_id, "file_removed": removed_file}

    @app.get("/api/clips/{clip_id}/waveform")
    def api_waveform(clip_id: int, points: int = Query(default=600, ge=50, le=4000)):
        """给前端画波形 + 应用 gain_db 的可视化。"""
        row = _require_clip(clip_id)
        path = _safe_audio_path(row["filepath"])
        try:
            y, sr = A.load_audio(path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"解码失败: {exc}") from exc
        if y.size == 0:
            return {"points": [], "duration_sec": 0, "sample_rate": sr}

        n = int(min(points, max(1, y.size)))
        # 分桶取峰值，比直接抽样更接近人眼看到的波形包络
        buckets = np.array_split(y, n)
        peaks = [float(np.max(np.abs(b))) if b.size else 0.0 for b in buckets]
        peak = max(peaks) or 1.0
        return {
            "points": [round(p / peak, 4) for p in peaks],
            "duration_sec": round(len(y) / sr, 3),
            "sample_rate": sr,
            "gain_db": row["gain_db"],
        }

    # ------------------------------------------------------------------ import

    def _run_import_job(job, body: ImportBody, opts: ImportOptions) -> dict[str, Any]:
        progress = JobProgress(job, min_interval=0.2)

        def cb(done: int, total: int) -> None:
            progress.update(done, total)

        result = import_into(
            state.cfg,
            state.db,
            state.encoder,
            opts,
            progress=cb,
            cancel=job.cancel,
            gender_map=body.gender_map,
        )
        job.message = f"导入完成：新增/更新 {result.imported} 条"
        return result.as_dict()

    @app.post("/api/import")
    def api_import(body: ImportBody):
        root = Path(body.path)
        if not root.is_dir():
            raise HTTPException(400, f"目录不存在: {body.path}")
        opts = ImportOptions(
            kind=body.kind,
            path=str(root),
            target_use=body.target_use,
            license=body.license,
            language=body.language,
            limit=body.limit,
            duration=tuple(body.duration) if body.duration else None,
            overwrite=body.overwrite,
            clip_kind=body.clip_kind,
            dry_run=body.dry_run,
            extra=body.extra or {},
        )
        job = state.jobs.submit(
            "import",
            lambda j: _run_import_job(j, body, opts),
            message="扫描目录…",
        )
        return {"job_id": job.id}

    @app.get("/api/import/preview")
    def api_import_preview(path: str, kind: str | None = None, limit: int = 20):
        """导入向导用：看看这个目录会被识别成什么、有多少条。"""
        root = Path(path)
        if not root.is_dir():
            raise HTTPException(400, f"目录不存在: {path}")
        from voicelib.sources import iter_source

        detected = kind or guess_kind(root)
        try:
            items = []
            count = 0
            for item in iter_source(detected, root):
                count += 1
                if len(items) < max(1, min(limit, 200)):
                    items.append(
                        {
                            "file": str(item.path),
                            "relpath": item.path.relative_to(root).as_posix(),
                            "category": item.category,
                            "speaker_id": item.speaker_id,
                            "gender": item.gender,
                            "transcript": item.transcript,
                        }
                    )
            return {
                "detected_kind": detected,
                "count": count,
                "samples": items,
                "suggested_target_use": "seedvc"
                if detected == "emogator"
                else "indextts",
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"无法读取该目录: {exc}") from exc

    # ------------------------------------------------------------------ jobs

    @app.get("/api/jobs")
    def api_jobs():
        return {"jobs": state.jobs.list()}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/cancel")
    def api_job_cancel(job_id: str):
        ok = state.jobs.cancel(job_id)
        if not ok:
            raise HTTPException(400, "任务不存在或已结束")
        return {"cancelled": job_id}

    @app.websocket("/api/jobs/{job_id}/ws")
    async def job_ws(ws: WebSocket, job_id: str):
        await ws.accept()
        try:
            async for frame in stream_job(lambda: state.jobs.get(job_id)):
                await ws.send_json(frame)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("WS %s 异常: %s", job_id, exc)
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ reembed

    def _run_reembed_job(job, body: ReembedBody) -> dict[str, Any]:
        rows = state.db.iter_clips()
        if body.missing_only:
            with state.db.read() as conn:
                rows = conn.execute(
                    "SELECT * FROM clips WHERE embedding IS NULL ORDER BY id"
                ).fetchall()
        job.total = len(rows)
        done = 0
        pairs = []
        texts = []
        ids = []
        for row in rows:
            if job.cancel.is_set():
                break
            desc = row["description"] or ""
            ids.append(row["id"])
            texts.append(desc)
            if len(texts) >= 32:
                vecs = state.encoder.encode(texts)
                pairs += [(i, state.encoder.to_blob(v)) for i, v in zip(ids, vecs)]
                state.db.set_embedding_many(pairs)
                done += len(ids)
                pairs, texts, ids = [], [], []
                job.done = done
        if texts:
            vecs = state.encoder.encode(texts)
            pairs += [(i, state.encoder.to_blob(v)) for i, v in zip(ids, vecs)]
            state.db.set_embedding_many(pairs)
            done += len(ids)
        job.done = done
        return {"reembedded": done, "total": len(rows)}

    @app.post("/api/reebed")
    def api_reembed_all(body: ReembedBody):
        job = state.jobs.submit(
            "reembed",
            lambda j: _run_reembed_job(j, body),
            message="重建向量…",
        )
        return {"job_id": job.id}

    # ------------------------------------------------------------------ export

    @app.post("/api/export")
    def api_export(body: ExportBody):
        if not body.clip_ids:
            raise HTTPException(400, "未选择任何素材")
        res = export_clips(
            state.cfg,
            state.db,
            ExportOptions(
                clip_ids=body.clip_ids,
                name=body.name,
                apply_gain=body.apply_gain,
                flatten=body.flatten,
            ),
        )
        return {
            "directory": res.directory,
            "exported": res.exported,
            "missing": res.missing,
            "items": res.items,
        }

    @app.get("/api/exports")
    def api_exports():
        return {"exports": list_exports(state.cfg)}

    # ------------------------------------------------------------------ voices

    @app.get("/api/voices")
    def api_voices():
        """spk_audio_prompt 候选（§5.1）。

        约定：把音色样本放进 <root>/voices/，这里按文件列出来。
        目录不存在时返回空列表，前端提示用户去放。
        """
        d = state.cfg.root_path / "voices"
        if not d.is_dir():
            return {"dir": str(d), "voices": []}
        voices = []
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
                rel = f.relative_to(state.cfg.root_path).as_posix()
                voices.append(
                    {
                        "name": f.stem,
                        "path": str(f),
                        "url": f"/api/voices/{f.stem}",
                        "relpath": rel,
                    }
                )
        return {"dir": str(d), "voices": voices}

    @app.get("/api/voices/{name}")
    def api_voice_audio(name: str):
        d = (state.cfg.root_path / "voices").resolve()
        if not d.is_dir():
            raise HTTPException(404, "voices 目录不存在")
        for f in d.rglob("*"):
            if f.is_file() and f.stem == name and f.suffix.lower() in {
                ".wav", ".flac", ".mp3", ".ogg", ".m4a"
            }:
                p = f.resolve()
                try:
                    p.relative_to(d)
                except ValueError:
                    raise HTTPException(403, "路径越界") from None
                return FileResponse(p)
        raise HTTPException(404, "音色文件不存在")

    # ------------------------------------------------------------------ 静态托管

    @app.get("/api/export/{name}/file/{filename}")
    def api_export_file(name: str, filename: str):
        d = (state.cfg.export_dir / T.safe_slug(name)).resolve()
        try:
            d.relative_to(state.cfg.export_dir.resolve())
        except ValueError:
            raise HTTPException(403, "路径越界") from None
        p = (d / filename).resolve()
        try:
            p.relative_to(d)
        except ValueError:
            raise HTTPException(403, "路径越界") from None
        if not p.is_file():
            raise HTTPException(404, "文件不存在")
        return FileResponse(p)

    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")
    else:

        @app.get("/")
        def index_placeholder():
            return JSONResponse(
                {
                    "message": "前端尚未构建。开发模式请运行 `npm run dev`（Vite 代理 /api 到 8000）。",
                    "api_docs": "/docs",
                }
            )

    return app


app: FastAPI = create_app(os.environ.get("VOICE_LIBRARY_ROOT"))
