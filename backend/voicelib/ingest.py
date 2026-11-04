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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import audio as A
from . import tags as T
from .config import Config
from .db import Database
from .embedder import Encoder
from .sources import ImportPlan, RawItem, count_items, iter_source, plan_import

log = logging.getLogger(__name__)

COMMIT_EVERY = 100
EMBED_BATCH = 32
MAX_FILE_MB = 200


@dataclass
class ImportOptions:
    kind: str = "csemotions"           # csemotions / emogator / flat / auto
    path: str = ""                     # 数据源目录
    target_use: str | None = None      # indextts / seedvc；None = 自动推断
    license: str = "apache-2.0"
    language: str = ""
    limit: int | None = None           # 仅用于调试/试跑
    # 时长筛选；None 表示跟随 usage 规则（按每个单元自己的线路）
    duration: tuple[float, float] | None = None
    overwrite: bool = True
    # 音频落盘策略（与上面的 overwrite 是两件事）：
    #   "overwrite" —— 重写 audio/ 下的同名文件（默认，确保与源一致）
    #   "keep"      —— 库里已有该文件就不再写盘，只刷新元数据
    # 文件名由音频内容 hash 决定，所以同名 == 同内容；
    # keep 省掉的是磁盘写入，不是元数据更新。
    audio_mode: str = "overwrite"
    clip_kind: str | None = None       # 覆盖自动判断
    # 试跑：只统计会导入多少条，不写盘
    dry_run: bool = False
    # speaker_id → gender（CSEMOTIONS / EmoGator 目录里没有性别信息）
    gender_map: dict[str, str] = field(default_factory=dict)
    # 「选哪一层做分类名」：目录名或绝对路径，必须是导入路径或其祖先。
    # 指定后整层统一类别，不再按子目录名拆分类别。
    category_level: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitResult:
    """一个导入单元（= 一个子目录）的执行结果。"""

    label: str
    kind: str
    target_use: str
    imported: int = 0
    total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "target_use": self.target_use,
            "imported": self.imported,
            "total": self.total,
        }


@dataclass
class ImportResult:
    imported: int = 0
    skipped_short: int = 0
    skipped_long: int = 0
    skipped_silent: int = 0
    skipped_other: int = 0
    duplicated: int = 0
    # 过短 / 超长素材不再丢弃，而是入库 + notes；这里单独计数以便界面提示
    imported_short: int = 0
    imported_over_long: int = 0
    # audio_mode="keep" 时，文件已在库中、本次跳过写盘的条数
    audio_kept: int = 0
    errors: list[str] = field(default_factory=list)
    scanned: int = 0
    units: list[UnitResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "imported": self.imported,
            "skipped_short": self.skipped_short,
            "skipped_long": self.skipped_long,
            "skipped_silent": self.skipped_silent,
            "skipped_other": self.skipped_other,
            "duplicated": self.duplicated,
            "imported_short": self.imported_short,
            "imported_over_long": self.imported_over_long,
            "audio_kept": self.audio_kept,
            "scanned": self.scanned,
            "errors": self.errors[:50],
            "error_count": len(self.errors),
            "units": [u.as_dict() for u in self.units],
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

    原因：
      'ok'        —— 正常入库
      'over_long' —— 已入库，但时长超出该线路范围（已写进 notes）
      'short' / 'silent' / 'dup' / 'error:<msg>' —— 未入库
    """
    try:
        if item.audio_bytes is not None:
            # parquet 等容器数据源：音频在内存里，不经磁盘
            y, sr = A.load_audio_bytes(item.audio_bytes)
        else:
            y, sr = A.load_audio(item.path)
    except Exception as exc:  # noqa: BLE001
        return None, f"error:读取失败 {exc}"

    if y.size == 0:
        return None, "silent"

    y = A.trim_for_ingest(y, sr)
    dur = float(len(y) / sr) if sr else 0.0
    if dur <= 0.02:
        return None, "silent"

    # 过短与超长都不丢弃，只入库并标记（用户决定）：
    #   * 过短：实测 CSEMOTIONS 有 74 条 <3s，其中 73 条是 angry 的短促爆发
    #     （「滚！别让我再看到你！」1.75s），恰恰是该情绪最典型、最干净的样本。
    #     按 3s 硬卡会把它们全部丢掉，愤怒类只剩长篇。
    #   * 超长：静默损失可用素材（CSEMOTIONS 有 8.5% 超 15s）。
    # 检索侧 indextts 默认仍按线路时长过滤，所以它们不会污染默认结果，
    # 但放宽时长后能找回来，也可以人工截取/拼接。
    too_short = dur < duration_range[0]
    over_long = dur > duration_range[1]
    if too_short:
        ok_status = "short"
    elif over_long:
        ok_status = "over_long"
    else:
        ok_status = "ok"

    # 发声段计数：只做异常标记（§4.2）
    notes_parts: list[str] = []
    if too_short:
        notes_parts.append(
            f"时长 {dur:.1f}s 低于 {duration_range[0]:g}s 下限，"
            f"作参考偏短，建议搭配更长的样本"
        )
    if over_long:
        notes_parts.append(
            f"时长 {dur:.1f}s 超出 {duration_range[0]:g}–{duration_range[1]:g}s 范围，"
            f"使用前需人工截取"
        )
    if clip_kind == "vocal_burst":
        try:
            n = A.count_segments(y, sr=sr)
        except Exception:  # noqa: BLE001
            n = 1
        if n == 0:
            return None, "silent"
        note = T.segment_note(n)
        if note:
            notes_parts.append(note)
    notes: str | None = "；".join(notes_parts) if notes_parts else None

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
        return row, ok_status

    # 已存在同路径：覆盖或跳过
    existing = db.get_by_filepath(rel_path)
    if existing is not None and not opts.overwrite:
        return None, "dup"

    # 音频落盘。keep 模式下文件已在就不再重写，只刷新元数据。
    # 但**文件不存在时必须写**：否则记录会指向一个空路径，
    # 试听 404、导出也拿不到音频。
    if opts.audio_mode == "keep" and abs_path.is_file():
        row["checksum"] = A.checksum_file(abs_path)
        return row, "kept"
    A.write_wav(abs_path, y, sr)
    row["checksum"] = A.checksum_file(abs_path)
    return row, ok_status


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
    """执行导入。

    kind="auto"（界面的默认值）时，先把目录拆成若干同构单元
    （见 sources.plan_import），每个单元按自己的线路与时长规则处理。
    这样「一个目录里混着 CSEMOTIONS + EmoGator + 自定义」也能一次导完，
    且各自的类别不会被解读错。
    """
    root = Path(opts.path)
    if not root.is_dir():
        raise FileNotFoundError(f"数据源目录不存在: {opts.path}")

    cfg.ensure_dirs()
    gender_map = gender_map if gender_map is not None else opts.gender_map

    if opts.category_level:
        # 「选哪一层做分类名」优先于一切形状识别：用户已经明确表达了
        # 要把该层整棵子树当成一个类别，再按子目录名拆开就是违背意图。
        plans = plan_import(
            root, assume_use=opts.target_use, category_level=opts.category_level
        )
        if not plans:
            raise ValueError(
                f"在「{opts.category_level}」这一层没有找到音频素材，"
                f"或该层不是导入路径的上级目录"
            )
    elif opts.kind == "auto":
        plans = plan_import(root, assume_use=opts.target_use)
        if not plans:
            raise ValueError("该目录里没有找到任何音频素材")
    else:
        # 显式指定类型：整体作为一个单元（保留旧行为，供高级选项使用）
        explicit_use = opts.target_use or "indextts"
        kind_count = count_items(opts.kind, root)
        plan = ImportPlan(
            path=root,
            kind=opts.kind,
            label=root.name or "根目录",
            clip_kind=opts.clip_kind or default_clip_kind(explicit_use),
            target_use=explicit_use,
            count=kind_count,
            reason="手动指定类型",
        )
        plans = [plan]

    total = sum(p.count for p in plans)
    if opts.limit:
        total = min(total, opts.limit)

    result = ImportResult()
    result.scanned = total
    if progress:
        progress(0, total)

    done = 0
    for plan in plans:
        if cancel is not None and cancel.is_set():
            break
        unit = UnitResult(
            label=plan.label, kind=plan.kind, target_use=plan.target_use, total=plan.count
        )
        result.units.append(unit)

        clip_kind = opts.clip_kind or plan.clip_kind
        duration_range = opts.duration or default_duration(plan.target_use)
        # 每个单元用自己目录的迭代器，避免把整根再扫一遍
        items = list(iter_source(plan.kind, plan.path, gender_map))

        unit_opts = ImportOptions(
            kind=plan.kind,
            path=str(plan.path),
            target_use=plan.target_use,
            # 单元自带许可（聚合数据集的每个子集许可不同）优先于请求里的全局值。
            # 许可在这里是「事实」而不是「偏好」：整包用一个值会把
            # EmoGator（Apache-2.0）错标成不可商用。
            license=plan.license or opts.license,
            language=opts.language,
            duration=opts.duration,
            overwrite=opts.overwrite,
            audio_mode=opts.audio_mode,
            clip_kind=opts.clip_kind,
            dry_run=opts.dry_run,
            gender_map=opts.gender_map,
            extra=opts.extra,
        )

        done = _run_unit(
            cfg,
            db,
            encoder,
            items,
            unit_opts,
            clip_kind=clip_kind,
            duration_range=duration_range,
            result=result,
            unit=unit,
            started_at=done,
            total=total,
            progress=progress,
            cancel=cancel,
        )
        if opts.limit and result.imported >= opts.limit:
            break

    if progress:
        progress(total, total)
    return result


def _run_unit(
    cfg: Config,
    db: Database,
    encoder: Encoder,
    items: Iterable[RawItem],
    opts: ImportOptions,
    *,
    clip_kind: str,
    duration_range: tuple[float, float],
    result: ImportResult,
    unit: UnitResult,
    started_at: int,
    total: int,
    progress: ProgressCb | None,
    cancel: threading.Event | None,
) -> int:
    """处理一个单元。返回累计「已处理」条数（用于整体进度）。"""
    pending: list[dict[str, Any]] = []
    pending_texts: list[str] = []

    def flush() -> None:
        """每 COMMIT_EVERY 条落一次库。

        顺序很关键：**先建向量，再开写事务**。
        建向量是 CPU 密集的（BGE 一批几十条可能要几百毫秒），
        若在事务内做，写锁会被一直持有——期间任何 PATCH / 任务快照
        都会排队直到超时，最终抛 "database is locked"。
        """
        if not pending:
            return
        if pending_texts:
            vecs = encoder.encode(pending_texts)
            for row, vec in zip(pending, vecs):
                row["embedding"] = encoder.to_blob(np.asarray(vec, dtype=np.float32))
        # 落库是一个短事务：只做 upsert，不做计算
        with db.write() as conn:
            for row in pending:
                db.upsert_clip(row, conn=conn)
        pending.clear()
        pending_texts.clear()

    processed = 0
    for i, item in enumerate(items, start=1):
        if cancel is not None and cancel.is_set():
            log.info("导入被取消，已处理 %d/%d", started_at + i - 1, total)
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
            unit.imported += 1
            # 过短与超长都已入库，分别计数以便界面提示
            if why == "over_long":
                result.imported_over_long += 1
            elif why == "short":
                result.imported_short += 1
            elif why == "kept":
                result.audio_kept += 1
            if not opts.dry_run:
                pending.append(row)
                pending_texts.append(row["description"])
                if len(pending) >= COMMIT_EVERY:
                    flush()

        processed = i
        if progress and (i % 5 == 0 or i == unit.total):
            progress(min(total, started_at + i), total)

    if not opts.dry_run:
        flush()
    return started_at + processed


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
