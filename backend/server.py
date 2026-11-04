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
from voicelib.search import (
    USAGE_RULES,
    categories_for,
    ids_matching,
    languages_for,
    row_to_dict,
    search,
    search_for,
)
from voicelib.sources import (
    iter_source,
    level_choices,
    plan_import,
)

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
    """人工标注。**显式传 null = 清掉该项人工标注、回落到程序值。**

    因此处理时只能用 `exclude_unset`，不能加 `exclude_none`：
    加了的话「传 null 清空」会被当成「没传」，用户就没有撤销入口。
    """

    verified: int | None = None
    intensity: int | None = Field(default=None, ge=0, le=5)
    description: str | None = None
    notes: str | None = None
    category: str | None = None
    gender: Literal["male", "female"] | None = None
    has_background: int | None = None
    studio_quality: int | None = None
    transcript: str | None = None
    # 语种必须能改：导入时没标语种的素材，只能靠这里补
    language: str | None = None
    target_use: Literal["indextts", "seedvc", "both"] | None = None
    license: str | None = None


class ImportBody(BaseModel):
    # auto：按目录结构自动拆分（推荐，也是界面默认）
    kind: Literal["auto", "csemotions", "emogator", "flat"] = "auto"
    path: str
    # None = 自动：每个单元按自己的排布决定线路
    target_use: Literal["indextts", "seedvc"] | None = None
    license: str = "apache-2.0"
    language: str = ""
    limit: int | None = None
    duration: list[float] | None = None
    clip_kind: Literal["speech", "vocal_burst"] | None = None
    overwrite: bool = True
    # 音频落盘策略：overwrite = 重写库里的 wav；keep = 沿用已有文件，只刷新元数据
    audio_mode: Literal["overwrite", "keep"] = "overwrite"
    dry_run: bool = False
    gender_map: dict[str, str] | None = None
    # 「选哪一层做分类名」：目录名或绝对路径，必须是 path 或其祖先。
    # 非空时整层统一类别，忽略 kind 的形状识别。
    category_level: str | None = None
    extra: dict[str, Any] = {}


class AnnotationClearBody(BaseModel):
    """批量清空人工标注。

    两种方式二选一：
      * `clip_ids` —— 清空这几条的全部人工标注（界面「清空选中标注」）
      * `by_filter=True` + 筛选字段 —— 清空当前筛选下的（界面「清空筛选下标注」）

    **必须显式传 by_filter 才按条件清**：下面这些筛选字段全是宽松默认值，
    空 body 若被当作「条件全不限」，一次调用就会把整个库的人工标注清光，
    而这种操作不可逆。
    """

    clip_ids: list[int] | None = None
    by_filter: bool = False
    # 与 /api/search 同一套筛选字段（见 build_filter_clause）
    use: Literal["indextts", "seedvc", "all"] = "indextts"
    kinds: list[str] | None = None
    categories: list[str] | None = None
    gender: str | None = None
    language: str | None = None
    min_intensity: int | None = None
    duration: list[float] | None = None
    commercial_only: bool = False
    verified_only: bool = False
    # 只清这几个字段；None = 清掉这些条目的全部标注
    fields: list[str] | None = None


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
        data["languages"] = {
            use: languages_for(use, state.db) for use in USAGE_RULES
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

    @app.get("/api/languages")
    def api_languages(use: str = "indextts"):
        return {"use": use, "languages": languages_for(use, state.db)}

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
        # 只用 exclude_unset：显式传 null 表示「清掉这项人工标注」，
        # 加 exclude_none 会把它当「没传」，用户就再也撤销不了。
        patch = body.model_dump(exclude_unset=True)
        if not patch:
            return {"updated": False, "clip": row_to_dict(row)}

        # license 改了，commercial_ok 必须跟着重算（§4.5）
        if "license" in patch and patch["license"] is not None:
            patch["commercial_ok"] = T.commercial_ok_of(patch["license"])

        # 描述由这些字段拼出来。改了其中任何一个（而没同时手写描述），
        # 描述就过期了 —— 例如语种从空改成 zh，不重建的话
        # 库里会出现 language=zh 但描述里没写「语言为中文」的条目。
        #
        # 只列**核心语义**字段。intensity / has_background / studio_quality
        # 虽然模板里也用到，但点是标、勾选这类操作频率极高，
        # 每次都重建描述 + 重算向量会让界面明显卡顿，
        # 而它们对「这段音频像不像我要的情绪」的影响远小于类别和台词。
        DESC_SOURCE = ("category", "transcript", "gender", "language")
        touched = [k for k in DESC_SOURCE if k in patch]
        program_row = state.db.get_clip_program(clip_id)
        if touched and "description" not in patch and program_row is not None:
            # 三层的取值优先级：
            #   1) 本次显式传了非 null → 用传进来的值
            #   2) 本次显式传了 null   → 该标注要被清掉，回落到**程序值**
            #   3) 本次没提这个字段     → 用当前合并值（保持界面看到的样子）
            # 少了第 2 层，用户「清空语种标注」后描述里会残留原来的
            # 「语言为英文」，与清空后的状态对不上。
            merged = {}
            for k in DESC_SOURCE:
                if k in patch:
                    merged[k] = patch[k] if patch[k] is not None else program_row[k]
                else:
                    merged[k] = row[k]
            if merged.get("category"):
                patch["description"] = T.build_description(
                    clip_kind=row["clip_kind"],
                    category=merged["category"],
                    transcript=merged["transcript"],
                    gender=merged["gender"],
                    language=merged["language"],
                    # 这几个不在 DESC_SOURCE 里，直接用当前值，
                    # 不参与「改了才重建」的判定
                    intensity=row["intensity"],
                    duration_sec=row["duration_sec"],
                    has_background=row["has_background"] or 0,
                    studio_quality=row["studio_quality"] or 0,
                )

        # 重建出来的描述是**程序产物**，不能记成用户标注：
        # 记成用户标注的话，以后导入永远刷不新它，
        # 而用户只是改了类别、并没有手写描述。
        rebuilt = "description" in patch and "description" not in body.model_fields_set
        updated = state.db.update_clip(
            clip_id, patch, program_keys=("description",) if rebuilt else ()
        )
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

    def _refresh_descriptions(clip_ids: list[int]) -> list[int]:
        """按**当前合并值**重拼描述，返回描述有变化的 id。

        为什么清空标注后要调它：描述是 category/transcript/gender/language
        拼出来的。清掉 language 标注、字段回落到未标注之后，
        描述里若还留着「语言为英文」，界面就自相矛盾。

        用户手写过描述（description 本身在 annotations 里）的条目跳过：
        那是用户的文字，不能被模板覆盖。

        重建出来的描述写回 **clips**（程序值）—— 它本来就是程序产物，
        写进 annotations 的话以后重新导入就再也刷不新了。
        """
        if not clip_ids:
            return []
        marks = ",".join("?" * len(clip_ids))
        with state.db.read() as conn:
            rows = conn.execute(
                f"SELECT * FROM clips_merged WHERE id IN ({marks})", clip_ids
            ).fetchall()
            annotated: dict[int, set[str]] = {}
            for r in conn.execute(
                f"SELECT clip_id, field FROM annotations WHERE clip_id IN ({marks})",
                clip_ids,
            ):
                annotated.setdefault(r["clip_id"], set()).add(r["field"])

        payload: list[tuple[str, int]] = []
        changed: list[int] = []
        for row in rows:
            if "description" in annotated.get(row["id"], set()):
                continue
            if not row["category"]:
                continue
            desc = T.build_description(
                clip_kind=row["clip_kind"],
                category=row["category"],
                transcript=row["transcript"],
                gender=row["gender"],
                language=row["language"],
                intensity=row["intensity"],
                duration_sec=row["duration_sec"],
                has_background=row["has_background"] or 0,
                studio_quality=row["studio_quality"] or 0,
            )
            if desc != row["description"]:
                payload.append((desc, row["id"]))
                changed.append(row["id"])
        if payload:
            with state.db.write() as conn:
                conn.executemany(
                    "UPDATE clips SET description = ? WHERE id = ?", payload
                )
        return changed

    @app.post("/api/annotations/clear")
    def api_clear_annotations(body: AnnotationClearBody):
        """批量清空人工标注。

        两种入口：按 id 列表（界面「清空选中标注」），
        或按当前筛选条件（界面「清空筛选下标注」）。
        后者不受 top_k 限制 —— 界面只显示一页，但要清的是整批。
        """
        if not body.clip_ids and not body.by_filter:
            raise HTTPException(400, "需要 clip_ids，或 by_filter=true 并提供筛选条件")

        if body.clip_ids:
            ids = [int(i) for i in body.clip_ids]
        else:
            use = "indextts" if body.use == "all" else body.use
            ids = ids_matching(
                state.db,
                use=use,
                kinds=body.kinds,
                categories=body.categories,
                gender=body.gender,
                language=body.language,
                min_intensity=body.min_intensity,
                duration=tuple(body.duration) if body.duration else None,
                commercial_only=body.commercial_only,
                verified_only=body.verified_only,
            )
            # 条件一条都不匹配时直接返回，不要当成「清了个空」
            # 静默成功 —— 用户可能以为自己选错了线路或条件。
            if not ids:
                return {"cleared": 0, "clips": 0, "reembedded": 0,
                        "message": "当前筛选条件没有命中的素材"}

        cleared = state.db.clear_annotations(ids, fields=body.fields)
        changed = _refresh_descriptions(ids)

        reembedded = 0
        if changed:
            marks = ",".join("?" * len(changed))
            with state.db.read() as conn:
                descs = [
                    r["description"]
                    for r in conn.execute(
                        f"SELECT description FROM clips WHERE id IN ({marks})", changed
                    )
                ]
            vecs = state.encoder.encode(descs)
            state.db.set_embedding_many(
                [(cid, state.encoder.to_blob(v)) for cid, v in zip(changed, vecs)]
            )
            reembedded = len(changed)

        return {"cleared": cleared, "clips": len(ids), "reembedded": reembedded}

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
            audio_mode=body.audio_mode,
            clip_kind=body.clip_kind,
            dry_run=body.dry_run,
            category_level=body.category_level or "",
            extra=body.extra or {},
        )
        job = state.jobs.submit(
            "import",
            lambda j: _run_import_job(j, body, opts),
            message="扫描目录…",
        )
        return {"job_id": job.id}

    @app.get("/api/import/preview")
    def api_import_preview(
        path: str,
        kind: str | None = None,
        limit: int = 20,
        category_level: str | None = None,
    ):
        """导入向导用：拆出这个目录里的所有导入单元，并给出样本。

        返回的 `units` 是「将要做的事」清单：
          [{relpath, kind, clip_kind, target_use, count, samples[]}, ...]
        单类型目录只有 1 个单元；混合目录会有多个，各自按正确排布解读。

        `level_choices` 给出「可以用哪一层做分类名」：从导入目录开始向上
        列出祖先，并附上该层大概有多少条音频。传 `category_level` 时，
        预览就按选定层整棵子树来算（类别统一为该层目录名）。
        """
        root = Path(path)
        if not root.is_dir():
            raise HTTPException(400, f"目录不存在: {path}")

        per_unit = max(1, min(limit, 50))

        def collect(
            unit_path: Path, unit_kind: str, base: Path | None = None
        ) -> tuple[int, list[dict]]:
            items = []
            count = 0
            # 相对路径的基准：默认是导入目录；选祖先层时音频可能不在它之下，
            # 所以调用方会传入选中的那一层。
            anchor = base or root
            # with_audio=False：预览只要元信息。parquet 的音频在容器里，
            # 若走完整读取会把每条音频都解出来（数百 MB），预览变成几十秒。
            for item in iter_source(unit_kind, unit_path, with_audio=False):
                count += 1
                if len(items) < per_unit:
                    try:
                        rel = item.path.relative_to(anchor).as_posix()
                    except ValueError:
                        rel = item.path.name
                    items.append(
                        {
                            "file": str(item.path),
                            "relpath": rel,
                            "category": item.category,
                            "speaker_id": item.speaker_id,
                            "gender": item.gender,
                            "transcript": item.transcript,
                        }
                    )
            return count, items

        try:
            if category_level:
                # 选定某一层做分类名：整层统一类别，不再按形状拆
                plans = plan_import(root, category_level=category_level)
                if not plans:
                    raise HTTPException(
                        400,
                        f"「{category_level}」这一层没有音频素材，"
                        f"或它不是导入路径的上级目录",
                    )
                units = []
                for plan in plans:
                    count, samples = collect(plan.path, plan.kind, base=plan.path)
                    if not count:
                        continue
                    d = plan.as_dict(root)
                    d.update({"count": count, "samples": samples})
                    units.append(d)
            elif kind and kind != "auto":
                # 显式指定：只报这一个单元
                count, samples = collect(root, kind)
                units = [
                    {
                        "relpath": root.name,
                        "path": str(root),
                        "kind": kind,
                        "clip_kind": "vocal_burst" if kind == "emogator" else "speech",
                        "target_use": "seedvc" if kind == "emogator" else "indextts",
                        "count": count,
                        "samples": samples,
                        "reason": "手动指定类型",
                    }
                ]
            else:
                units = []
                for plan in plan_import(root):
                    count, samples = collect(plan.path, plan.kind)
                    if not count:
                        continue
                    d = plan.as_dict(root)
                    d.update(
                        {
                            "count": count,
                            "samples": samples,
                        }
                    )
                    units.append(d)

            total = sum(u["count"] for u in units)
            if not units:
                raise HTTPException(400, "该目录里没有找到任何音频素材")

            single = units[0] if len(units) == 1 else None
            return {
                "mixed": len(units) > 1,
                "unit_count": len(units),
                "count": total,
                "units": units,
                # 「可以用哪一层做分类名」：从导入目录向上列祖先。
                # 条目带 name / path / count / count_truncated / is_root，
                # 前端据此渲染成单选，选中后带 category_level 重新请求预览。
                "level_choices": level_choices(root),
                "category_level": category_level or "",
                # 兼容旧前端的字段
                "detected_kind": single["kind"] if single else "mixed",
                "samples": (single or units[0])["samples"],
                "suggested_target_use": (
                    single["target_use"] if single else ""
                ),
            }
        except HTTPException:
            raise
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
