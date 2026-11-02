"""导出（方案 §5 + §6.2 POST /api/export）。

导出做的事：
  1. 把选中 id 对应的音频**复制**到 <root>/exports/<stamp>/（不改原文件）
  2. 写 selection.json 记录来源，方便复现（§6.4 交互要点 3）
  3. 可选：把 gain_db 应用到副本上（§4.4 的"导出 16-bit 给别人"例外）
  4. 文件名带情绪/类别与序号，可直接传给下游
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import audio as A
from . import tags as T
from .config import Config
from .db import Database


@dataclass
class ExportOptions:
    clip_ids: list[int] = field(default_factory=list)
    name: str = ""              # 子目录名，缺省用时间戳
    apply_gain: bool = False    # 是否把 gain_db 烘进副本
    flatten: bool = True        # 是否放到同一层目录
    copy_notes: bool = True


@dataclass
class ExportResult:
    directory: str
    exported: int
    items: list[dict[str, Any]] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)


def _resolve(cfg: Config, rel: str) -> Path | None:
    p = (cfg.root_path / rel).resolve()
    try:
        p.relative_to(cfg.root_path)
    except ValueError:
        return None
    return p if p.is_file() else None


def export_clips(cfg: Config, db: Database, opts: ExportOptions) -> ExportResult:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sub = T.safe_slug(opts.name) if opts.name else stamp
    out_dir = cfg.export_dir / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Any] = []
    missing: list[int] = []
    for cid in opts.clip_ids:
        row = db.get_clip(int(cid))
        if row is None:
            missing.append(int(cid))
        else:
            rows.append(row)

    items: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for row in rows:
        src = _resolve(cfg, row["filepath"])
        if src is None:
            missing.append(int(row["id"]))
            continue

        kind_cn = (
            T.emotion_cn(row["category"])
            if row["clip_kind"] == "speech"
            else T.burst_kind_cn(row["category"])
        )
        base = f"{row['id']:05d}_{kind_cn}"
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            base += f"_{seen[base]}"
        target = out_dir / (base + src.suffix)
        if opts.flatten:
            target = out_dir / (base + src.suffix)
        else:
            target = out_dir / Path(row["filepath"]).parent.name / (base + src.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)

        if opts.apply_gain:
            y, sr = A.load_audio(src, sr=None)
            y = A.normalize_loudness(y, cfg.target_lufs)
            A.write_wav(target, y, sr)
        else:
            shutil.copy2(src, target)

        items.append(
            {
                "id": row["id"],
                "export_path": str(target),
                "export_relpath": target.relative_to(cfg.root_path).as_posix(),
                "filepath": row["filepath"],
                "category": row["category"],
                "clip_kind": row["clip_kind"],
                "target_use": row["target_use"],
                "duration_sec": row["duration_sec"],
                "description": row["description"],
                "speaker_id": row["speaker_id"],
                "license": row["license"],
                "commercial_ok": row["commercial_ok"],
                "gain_db": row["gain_db"],
                "gain_applied": opts.apply_gain,
                "source_file": row["source_file"],
                "notes": row["notes"],
            }
        )

    manifest = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(cfg.root_path),
        "name": sub,
        "count": len(items),
        "missing": missing,
        "apply_gain": opts.apply_gain,
        "target_lufs": cfg.target_lufs if opts.apply_gain else None,
        "clips": items,
    }
    (out_dir / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return ExportResult(
        directory=str(out_dir),
        exported=len(items),
        items=items,
        missing=missing,
    )


def list_exports(cfg: Config) -> list[dict[str, Any]]:
    if not cfg.export_dir.is_dir():
        return []
    out = []
    for d in sorted(
        (p for p in cfg.export_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        manifest = d / "selection.json"
        info: dict[str, Any] = {
            "name": d.name,
            "path": str(d),
            "mtime": d.stat().st_mtime,
        }
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                info["count"] = data.get("count", 0)
                info["exported_at"] = data.get("exported_at")
            except Exception:  # noqa: BLE001
                pass
        out.append(info)
    return out
