"""数据源适配层：从各种「已经下载到本地」的数据集目录读出素材。

方案 §4.1 / §4.2 的入库流程需要数据源给出的字段：
  * 音频（wav / flac / mp3 ...）
  * emotion / category 标签
  * transcript（有则用）
  * speaker / gender

支持的排布：
  csemotions   <root>/<speaker>/<emotion>/*.wav         （CSEMOTIONS）
  emogator     <root>/<label>/*.wav                     （EmoGator 30 类）
  flat         <root>/**/* + manifest.json              （自定义素材）

**不依赖网络**：不在这里调用 load_dataset，HF 数据集请先 `huggingface-cli download`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aiff", ".aif"}

SourceKind = Literal["csemotions", "emogator", "flat"]


@dataclass
class RawItem:
    """数据源里的一条原始素材（尚未裁剪、尚未落盘）。"""

    path: Path
    category: str
    category_raw: str | None = None
    speaker_id: str | None = None
    gender: str | None = None
    transcript: str | None = None
    intensity: int | None = None
    language: str | None = None
    # 数据源给不了的元信息可以在这里透传
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------- CSEMOTIONS

# CSEMOTIONS 的原始目录结构有两种常见形态：
#   <root>/<speaker_id>/<Emotion>/*.wav
#   <root>/audio/<speaker_id>/<Emotion>/*.wav
_CSE_EMOTIONS = {
    "angry",
    "happy",
    "sad",
    "fearful",
    "surprise",
    "playfulness",
    "neutral",
}


def _canonical_emotion(name: str) -> str:
    """Angry / angry / ANGRY → Angry；保持与 CSEMOTIONS 标签一致。"""
    mapping = {e.lower(): e.capitalize() for e in _CSE_EMOTIONS}
    return mapping.get(name.strip().lower(), name.strip())


def _looks_like_emotion(name: str) -> bool:
    return name.strip().lower() in _CSE_EMOTIONS


def iter_csemotions(root: Path, gender_map: dict[str, str] | None = None) -> Iterator[RawItem]:
    """遍历 CSEMOTIONS 目录。

    支持 <speaker>/<Emotion>/ 与 <root>/audio/<speaker>/<Emotion>/ 两种层级。
    """
    gender_map = gender_map or {}
    if not root.is_dir():
        return

    def scan(base: Path) -> Iterator[RawItem]:
        for speaker_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if _looks_like_emotion(speaker_dir.name):
                continue
            speaker = speaker_dir.name
            for emo_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
                if not _looks_like_emotion(emo_dir.name):
                    continue
                category = _canonical_emotion(emo_dir.name)
                for f in sorted(emo_dir.rglob("*")):
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                        yield RawItem(
                            path=f,
                            category=category,
                            category_raw=emo_dir.name,
                            speaker_id=speaker,
                            gender=gender_map.get(speaker),
                        )

    items = list(scan(root))
    if items:
        yield from items
        return
    audio_sub = root / "audio"
    if audio_sub.is_dir():
        yield from scan(audio_sub)


# ---------------------------------------------------------------- EmoGator

def iter_emogator(root: Path, gender_map: dict[str, str] | None = None) -> Iterator[RawItem]:
    """EmoGator 排布：<root>/<label>/*.mp3，文件名形如 000123-EE-1.mp3。

    EE 是 30 类的两位编码，但没有公开的编码表；优先用目录名当标签。
    目录名不是标签时（比如按贡献者分目录），退回文件名里的 EE。
    """
    gender_map = gender_map or {}
    if not root.is_dir():
        return
    for first in sorted(p for p in root.iterdir()):
        if first.is_file() and first.suffix.lower() in AUDIO_EXTS:
            label = re.sub(r"^[0-9]+[-_]", "", first.stem)
            yield RawItem(path=first, category=label or "unknown", category_raw=label)
            continue
        if not first.is_dir():
            continue
        for f in sorted(first.rglob("*")):
            if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
                continue
            label = first.name
            m = re.match(r"^(\d+)-([A-Za-z]{2})-(\d+)$", f.stem)
            raw = m.group(2) if m else label
            yield RawItem(
                path=f,
                category=label,
                category_raw=raw,
                speaker_id=m.group(1) if m else None,
                gender=gender_map.get(m.group(1)) if m else None,
            )


# ---------------------------------------------------------------- flat + manifest

MANIFEST_NAMES = ("manifest.json", "manifest.jsonl", "index.json")


def _load_manifest(root: Path) -> dict[str, dict]:
    """返回 {相对路径: 标注}。"""
    out: dict[str, dict] = {}
    for name in MANIFEST_NAMES:
        p = root / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        if name.endswith(".jsonl"):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get("filepath") or rec.get("path") or rec.get("file")
                if key:
                    out[str(key).replace("\\", "/")] = rec
        else:
            data = json.loads(text)
            records = data.get("clips") if isinstance(data, dict) else data
            for rec in records or []:
                key = rec.get("filepath") or rec.get("path") or rec.get("file")
                if key:
                    out[str(key).replace("\\", "/")] = rec
        break
    return out


def iter_flat(root: Path) -> Iterator[RawItem]:
    """扁平目录 + manifest.json。

    manifest 每条至少要有：
        {"filepath": "a/b.wav", "category": "laugh", "clip_kind": "vocal_burst"}
    可选：speaker_id / gender / transcript / intensity / language / license / target_use
    """
    if not root.is_dir():
        return
    manifest = _load_manifest(root)
    for f in sorted(root.rglob("*")):
        if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
            continue
        rel = f.relative_to(root).as_posix()
        meta = manifest.get(rel) or manifest.get(f.name) or {}
        category = meta.get("category") or meta.get("emotion") or f.parent.name or "unknown"
        yield RawItem(
            path=f,
            category=str(category),
            category_raw=meta.get("category_raw"),
            speaker_id=meta.get("speaker_id"),
            gender=meta.get("gender"),
            transcript=meta.get("transcript"),
            intensity=meta.get("intensity"),
            language=meta.get("language"),
            extras={k: v for k, v in meta.items() if k not in {
                "filepath", "path", "file", "category", "emotion", "category_raw",
                "speaker_id", "gender", "transcript", "intensity", "language",
            }},
        )


def iter_source(kind: str, root: Path, gender_map: dict[str, str] | None = None) -> Iterator[RawItem]:
    kind = (kind or "").lower()
    if kind in ("csemotions", "cse"):
        return iter_csemotions(root, gender_map)
    if kind in ("emogator", "emogators"):
        return iter_emogator(root, gender_map)
    if kind == "flat":
        return iter_flat(root)
    raise ValueError(f"不支持的数据源类型: {kind}（可用: csemotions / emogator / flat）")


def _has_audio(d: Path, limit: int = 200) -> bool:
    for i, f in enumerate(d.rglob("*")):
        if i > limit:
            return False
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            return True
    return False


def guess_kind(root: Path) -> str:
    """界面上「导入目录」时给出一个猜测。

    顺序很关键：先看目录形状，最后才看 manifest。
    因为 EmoGator 的目录也可能带一个 manifest.json，若先判 manifest
    就会把 <label>/*.mp3 误判成 flat。
    """
    if not root.is_dir():
        return "flat"

    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]

    # <speaker>/<emotion>/*.wav
    for p in subdirs:
        for q in sorted(p.iterdir()):
            if q.is_dir() and _looks_like_emotion(q.name) and _has_audio(q):
                return "csemotions"

    # <label>/*.mp3（label 不一定是情绪词，EmoGator 的 30 类大多是情绪）
    for p in subdirs:
        if _has_audio(p):
            return "emogator"

    # 音频直接摊在根目录
    if any(f.is_file() and f.suffix.lower() in AUDIO_EXTS for f in root.iterdir()):
        return "flat" if any((root / n).is_file() for n in MANIFEST_NAMES) else "emogator"

    if any((root / n).is_file() for n in MANIFEST_NAMES):
        return "flat"
    return "flat"
