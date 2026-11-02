"""导入管线（方案 §4.1 / §4.2 / §4.3 / §4.5）。

流程：遍历数据源 → 读音频 → 静音裁剪 → 写 audio/ → 生成描述 → 建向量 → 入库。

关键实现约定：
  * 每 100 条提交一次事务（§6.3），不要每条都写库
  * 响度归一化是「只记 gain_db，不改音频」（§4.4）
  * commercial_ok 由 license 推导（§4.5）
  * 发声段计数只用于标记 notes，不决定去留（§4.2）
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import audio as A
from . import tags as T
from .config import Config
from .db import Database
from .embedder import Encoder
from .sources import RawItem, iter_source

log = logging.getLogger(__name__)

COMMIT_EVERY = 100
EMBED_BATCH = 32
MAX_FILE_MB = 200


@dataclass
class ImportOptions:
    kind: str = "csemotions"           # csemotions / emogator / flat
    path: str = ""                     # 数据源目录
    target_use: str = "indextts"       # indextts / seedvc
    license: str = "apache-2.0"
    language: str = ""
    limit: int | None = None           # 仅用于调试/试跑
    # 时长筛选；None 表示跟随 usage 规则
    duration: tuple[float, float] | None = None
    overwrite: bool = True
    clip_kind: str | None = None       # 覆盖自动判断
    # 试跑：只统计会导入多少条，不写盘
    dry_run: bool = False
    # speaker_id → gender（CSEMOTIONS / EmoGator 目录里没有性别信息）
    gender_map: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportResult:
    imported: int = 0
    skipped_short: int = 0
    skipped_long: int = 0
    skipped_silent: int = 0
    skipped_other: int = 0
    duplicated: int = 0
    errors: list[str] = field(default_factory=list)
    scanned: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "imported": self.imported,
            "skipped_short": self.skipped_short,
            "skipped_long": self.skipped_long,
            "skipped_silent": self.skipped_silent,
            "skipped_other": self.skipped_other,
            "duplicated": self.duplicated,
            "scanned": self.scanned,
            "errors": self.errors[:50],
            "error_count": len(self.errors),
        }


def default_clip_kind(use: str) -> str:
    return "speech" if use == "indextts" else "vocal_burst"


def default_duration(use: str) -> tuple[float, float]:
    return (3.0, 15.0) if use == "indextts" else (0.3, 5.0)


def output_subdir(use: str, kind: str) -> str:
    if use == "seedvc" or kind == "vocal_burst":
        return "audio/vocal_burst"
    return "audio/csemotions"


# --------------------------------------------------------------------- 单条


def ingest_one(
    item: RawItem,
    *,
    cfg: Config,
    db: Database,
    encoder: Encoder,
    opts: ImportOptions,
    clip_kind: str,
    duration_range: tuple[float, float],
) -> tuple[dict[str, Any] | None, str]:
    """处理一条素材。返回 (row, 原因)。

    原因：'ok' / 'short' / 'long' / 'silent' / 'error:<msg>' / 'dup'
    """
    try:
        y, sr = A.load_audio(item.path)
    except Exception as exc:  # noqa: BLE001
        return None, f"error:读取失败 {exc}"

    if y.size == 0:
        return None, "silent"

    y = A.trim_for_ingest(y, sr)
    dur = float(len(y) / sr) if sr else 0.0
    if dur <= 0.02:
        return None, "silent"
    if dur < duration_range[0]:
        return None, "short"
    if dur > duration_range[1]:
        return None, "long"

    # 发声段计数：只做异常标记（§4.2）
    notes: str | None = None
    if clip_kind == "vocal_burst":
        try:
            n = A.count_segments(y, sr=sr)
        except Exception:  # noqa: BLE001
            n = 1
        if n == 0:
            return None, "silent"
        note = T.segment_note(n)
        if note:
            notes = note

    # 落盘文件名：<source>_<category>_<hash8>.wav
    checksum = A.checksum_bytes(y.tobytes()[: 1 << 20]) + f"{len(y)}"
    import hashlib

    short_hash = hashlib.blake2b(checksum.encode(), digest_size=4).hexdigest()
    subdir = output_subdir(opts.target_use, clip_kind)
    stem = "_".join(
        [
            T.safe_slug(item.speaker_id or "x")[:16],
            T.safe_slug(item.category)[:24],
            short_hash,
        ]
    )
    rel_path = f"{subdir}/{stem}.wav"
    abs_path = cfg.root_path / rel_path

    gender = T.normalize_gender(item.gender)
    language = item.language or opts.language or None
    license_str = T.normalize_license(
        opts.extra.get("license") or opts.license
    )
    commercial = T.commercial_ok_of(license_str)

    description = T.build_description(
        clip_kind=clip_kind,
        category=item.category,
        transcript=item.transcript,
        gender=gender,
        language=language,
        intensity=item.intensity,
        duration_sec=dur,
        has_background=int(opts.extra.get("has_background", 0)),
        studio_quality=int(opts.extra.get("studio_quality", 1 if opts.kind == "csemotions" else 0)),
    )

    row: dict[str, Any] = {
        "filepath": rel_path,
        "checksum": A.checksum_file(abs_path) if abs_path.is_file() else short_hash,
        "source_file": str(item.path),
        "start_sec": 0.0,
        "target_use": opts.target_use,
        "clip_kind": clip_kind,
        "category": str(item.category),
        "category_raw": item.category_raw,
        "intensity": item.intensity,
        "speaker_id": item.speaker_id,
        "gender": gender,
        "language": language,
        "duration_sec": round(dur, 3),
        "sample_rate": sr,
        "loudness_lufs": round(A.lufs_approx(y), 2),
        "gain_db": round(A.gain_to_target(y, cfg.target_lufs), 2),
        "has_background": int(opts.extra.get("has_background", 0)),
        "studio_quality": int(
            opts.extra.get("studio_quality", 1 if opts.kind == "csemotions" else 0)
        ),
        "transcript": item.transcript,
        "description": description,
        "license": license_str,
        "commercial_ok": commercial,
        "verified": 0,
        "notes": notes,
    }

    if opts.dry_run:
        return row, "ok"

    # 已存在同路径：覆盖或跳过
    existing = db.get_by_filepath(rel_path)
    if existing is not None and not opts.overwrite:
        return None, "dup"

    A.write_wav(abs_path, y, sr)
    row["checksum"] = A.checksum_file(abs_path)
    return row, "ok"


# --------------------------------------------------------------------- 批量


ProgressCb = Callable[[int, int], None]


def run_import(
    cfg: Config,
    db: Database,
    encoder: Encoder,
    opts: ImportOptions,
    *,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
    gender_map: dict[str, str] | None = None,
) -> ImportResult:
    root = Path(opts.path)
    if not root.is_dir():
        raise FileNotFoundError(f"数据源目录不存在: {opts.path}")

    clip_kind = opts.clip_kind or default_clip_kind(opts.target_use)
    duration_range = opts.duration or default_duration(opts.target_use)
    cfg.ensure_dirs()

    gender_map = gender_map if gender_map is not None else opts.gender_map
    items = list(iter_source(opts.kind, root, gender_map))
    if opts.limit:
        items = items[: opts.limit]

    result = ImportResult()
    result.scanned = len(items)
    total = len(items)
    if progress:
        progress(0, total)

    pending: list[dict[str, Any]] = []
    pending_texts: list[str] = []
    pending_ids: list[int] = []

    def flush(conn) -> None:
        """批量建向量 + 提交事务（每 COMMIT_EVERY 条）。"""
        if not pending:
            return
        if pending_texts:
            vecs = encoder.encode(pending_texts)
            for row, vec in zip(pending, vecs):
                row["embedding"] = encoder.to_blob(np.asarray(vec, dtype=np.float32))
        for row in pending:
            db.upsert_clip(row, conn=conn)
        if conn is not None:
            conn.commit()
        pending.clear()
        pending_texts.clear()
        pending_ids.clear()

    conn = db.connect()
    try:
        for i, item in enumerate(items, start=1):
            if cancel is not None and cancel.is_set():
                log.info("导入被取消，已处理 %d/%d", i - 1, total)
                break
            row, why = ingest_one(
                item,
                cfg=cfg,
                db=db,
                encoder=encoder,
                opts=opts,
                clip_kind=clip_kind,
                duration_range=duration_range,
            )
            if row is None:
                if why.startswith("error"):
                    result.errors.append(f"{item.path}: {why}")
                elif why == "short":
                    result.skipped_short += 1
                elif why == "long":
                    result.skipped_long += 1
                elif why == "silent":
                    result.skipped_silent += 1
                elif why == "dup":
                    result.duplicated += 1
                else:
                    result.skipped_other += 1
            else:
                result.imported += 1
                if not opts.dry_run:
                    pending.append(row)
                    pending_texts.append(row["description"])
                    if len(pending) >= COMMIT_EVERY:
                        flush(conn)

            if progress and (i % 5 == 0 or i == total):
                progress(i, total)

        if not opts.dry_run:
            flush(conn)
            if conn is not None:
                conn.commit()
    except Exception:  # noqa: BLE001
        result.errors.append(traceback.format_exc(limit=3))
        raise
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if progress:
        progress(total, total)
    return result


def import_into(
    cfg: Config,
    db: Database,
    encoder: Encoder,
    opts: ImportOptions,
    *,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
    gender_map: dict[str, str] | None = None,
) -> ImportResult:
    return run_import(
        cfg,
        db,
        encoder,
        opts,
        progress=progress,
        cancel=cancel,
        gender_map=gender_map,
    )
