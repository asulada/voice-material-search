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
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from . import audio as A

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aiff", ".aif"}
# 音频内嵌在容器里（HF datasets 的 parquet），不是可直接播放的文件
PARQUET_EXTS = {".parquet"}

SourceKind = Literal["csemotions", "emogator", "flat", "parquet", "webdataset"]


# ---------------------------------------------------------------- WebDataset 子集
#
# gijs/speech-utterances 是 6 个数据集的聚合包，各子集的**许可互不相同**。
# 整包 README 头部写的是 cc-by-nc-sa-4.0，但那只代表最严的那个；
# 若照搬会把 EmoGator（Apache-2.0）和 VIVAE（CC-BY-4.0）里本来可商用的
# 素材永久标成不可商用。所以按子集分别落库（方案 §4.5 的意图）。
#
# key 是目录名的规范化形式（小写、下划线）。
WEB_SUBSETS: dict[str, dict] = {
    "emogator": {
        "license": "apache-2.0",
        "clip_kind": "vocal_burst",
        "target_use": "seedvc",
        # 副语言（情绪性发声）不承载语种，留空是对的。
        # 填 "en" 会造成一种误解：以为能按语言检索出「英语的喘气」。
        "language": None,
        "note": "30 类情绪发声，Apache-2.0 可商用",
    },
    "vivae": {
        "license": "cc-by-4.0",
        "clip_kind": "vocal_burst",
        "target_use": "seedvc",
        "language": None,
        "note": "6 类情绪 × 4 档强度，CC-BY-4.0 可商用",
    },
    "vocalsound": {
        "license": "cc-by-sa-4.0",
        "clip_kind": "vocal_burst",
        "target_use": "seedvc",
        "language": None,
        "note": "CC-BY-SA-4.0，可商用但有 ShareAlike 义务",
    },
    "deeply_nonverbal_vocalization": {
        "license": "cc-by-nc-nd-4.0",
        "clip_kind": "vocal_burst",
        "target_use": "seedvc",
        "language": None,
        "note": "CC-BY-NC-ND-4.0，不可商用且禁止演绎",
    },
    "expresso": {
        "license": "cc-by-nc-4.0",
        "clip_kind": "speech",
        "target_use": "indextts",
        # 全库唯一有语种的子集：真人英语朗读/对话。
        "language": "en",
        # 用户听过音频后确认：text 恰为 "Expressive speech in a neutral style"
        # 的那批是整段双人对话，不要；Expresso 里其余条目正常，全都要
        # （它们的 text 是真台词，照常保留）。
        "drop_text_values": ["Expressive speech in a neutral style"],
        "note": "对话语音，CC-BY-NC-4.0 不可商用",
    },
    "nonspeech7k": {
        "license": "cc-by-nc-sa-4.0",
        "clip_kind": "vocal_burst",
        "target_use": "seedvc",
        "language": None,
        "note": "7 类发声，CC-BY-NC-SA-4.0 不可商用",
    },
}

# VIVAE 自带强度标注，直接映射到库里的星级，不用人工再打
_WEB_INTENSITY = {"low": 2, "moderate": 3, "strong": 4, "peak": 5}


def _subset_key(name: str) -> str:
    """目录名 → WEB_SUBSETS 的 key。"""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _camel_to_snake(name: str) -> str:
    """DeeplyNonverbalVocalization → deeply_nonverbal_vocalization"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())


@dataclass
class RawItem:
    """数据源里的一条原始素材（尚未裁剪、尚未落盘）。

    `path` 对普通目录数据源指向磁盘上的音频文件；
    parquet 这类「音频内嵌在容器里」的数据源没有独立文件，
    改为把字节放在 `audio_bytes`，path 只作显示用（parquet 路径 + 行号）。
    """

    path: Path
    category: str
    category_raw: str | None = None
    speaker_id: str | None = None
    gender: str | None = None
    transcript: str | None = None
    intensity: int | None = None
    language: str | None = None
    # 非空表示音频在内存里（parquet 等），不必也不能从 path 读盘
    audio_bytes: bytes | None = None
    # 数据源给不了的元信息可以在这里透传
    extras: dict = field(default_factory=dict)

    @property
    def is_in_memory(self) -> bool:
        return self.audio_bytes is not None


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


def _make_manifest_lookup(root: Path):
    """返回 f(path) -> dict 的查询函数，查不到返回 {}。

    清单不一定就在被导入的那一层：用户常填「总目录」或中间某一层
    （音频在 `wav/`、清单在根）。所以先向上定位清单目录，
    再用相对该目录的路径去查，并保留文件名兜底。
    """
    mdir = _find_manifest_dir(root)
    if mdir is None:
        return lambda f: {}
    manifest = _load_manifest(mdir)
    if not manifest:
        return lambda f: {}

    def lookup(f: Path) -> dict:
        try:
            rel = f.relative_to(mdir).as_posix()
        except ValueError:
            rel = f.name
        return manifest.get(rel) or manifest.get(f.name) or {}

    return lookup


def _manifest_transcript(meta: dict) -> str | None:
    for key in _TRANSCRIPT_FIELDS:
        v = meta.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def iter_csemotions(root: Path, gender_map: dict[str, str] | None = None) -> Iterator[RawItem]:
    """遍历 CSEMOTIONS 目录。

    支持 <speaker>/<Emotion>/ 与 <root>/audio/<speaker>/<Emotion>/ 两种层级。
    若同层或上层有台账，台词/性别优先取台账里的（用户那份 metadata.jsonl
    的台词就在里面，不读就白丢了）。
    """
    gender_map = gender_map or {}
    if not root.is_dir():
        return
    lookup = _make_manifest_lookup(root)

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
                        meta = lookup(f)
                        yield RawItem(
                            path=f,
                            category=category,
                            category_raw=emo_dir.name,
                            speaker_id=meta.get("speaker") or speaker,
                            gender=gender_map.get(speaker)
                            or meta.get("gender")
                            or _infer_gender(speaker),
                            transcript=_manifest_transcript(meta),
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
    lookup = _make_manifest_lookup(root)
    for first in sorted(p for p in root.iterdir()):
        if first.is_file() and first.suffix.lower() in AUDIO_EXTS:
            label = re.sub(r"^[0-9]+[-_]", "", first.stem)
            meta = lookup(first)
            yield RawItem(
                path=first,
                category=label or "unknown",
                category_raw=label,
                transcript=_manifest_transcript(meta),
            )
            continue
        if not first.is_dir():
            continue
        for f in sorted(first.rglob("*")):
            if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
                continue
            label = first.name
            m = re.match(r"^(\d+)-([A-Za-z]{2})-(\d+)$", f.stem)
            raw = m.group(2) if m else label
            meta = lookup(f)
            yield RawItem(
                path=f,
                category=label,
                category_raw=raw,
                speaker_id=m.group(1) if m else meta.get("speaker"),
                gender=gender_map.get(m.group(1)) if m else (meta.get("gender") or None),
                # 台账里的台词要带出来：有台词的素材会走 IndexTTS，
                # 不读台账就等于把语言内容当成了副语言。
                transcript=_manifest_transcript(meta),
            )


# ---------------------------------------------------------------- flat + manifest

# 代码认的清单文件名。前三个是通用名，后三个是导出数据集常见的元数据名
# （HF datasets 的 parquet 导出工具默认就产出 metadata.jsonl / metadata.csv）。
MANIFEST_NAMES = (
    "manifest.json",
    "manifest.jsonl",
    "index.json",
    "metadata.jsonl",
    "metadata.json",
    "metadata.csv",
)

# 清单里用来指向音频的相对路径字段。不同导出工具的命名不统一：
#   filepath/path/file  —— 本项目的 manifest 约定
#   wav_path            —— 用户这份 metadata.jsonl 用的
#   original_filename    —— 只有文件名、没有目录层级
_MANIFEST_KEY_FIELDS = (
    "filepath", "path", "file",
    "wav_path", "audio_path", "audio", "wav",
    "original_filename", "filename",
)


def _manifest_key(rec: dict) -> str | None:
    """从一条清单记录里取音频相对路径。"""
    for name in _MANIFEST_KEY_FIELDS:
        v = rec.get(name)
        if v:
            return str(v).replace("\\", "/")
    return None


def _load_manifest(root: Path) -> dict[str, dict]:
    """返回 {相对路径: 标注}，同时按文件名建一份兜底索引。

    兜底索引是必要的：用户可能把根目录填在上层（项目根），
    而清单里的路径是 `wav/<speaker>/<emotion>/x.wav`——这时按完整相对路径
    能对上；但如果填的是 `wav/` 这一层，清单键就多了一层前缀，
    只能靠文件名兜底。同名冲突时不覆盖，保留先出现的那条。
    """
    out: dict[str, dict] = {}
    for name in MANIFEST_NAMES:
        p = root / name
        if not p.is_file():
            continue
        records: list[dict] = []
        try:
            if name.endswith(".csv"):
                import csv

                # utf-8-sig：这家的 csv 带 BOM，不去掉第一列名会变成 '\ufeffwav_path'
                with p.open(encoding="utf-8-sig", newline="") as fh:
                    records = [dict(r) for r in csv.DictReader(fh)]
            elif name.endswith(".jsonl"):
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
                records = data.get("clips") if isinstance(data, dict) else data
        except (OSError, ValueError):
            continue

        for rec in records or []:
            if not isinstance(rec, dict):
                continue
            key = _manifest_key(rec)
            if not key:
                continue
            out[key] = rec
            base = key.rsplit("/", 1)[-1]
            out.setdefault(base, rec)
        break
    return out


# 台账里已经被 RawItem 消费掉的键，不该再原样塞进 extras
_CONSUMED_META_KEYS = frozenset({
    "filepath", "path", "file", "wav_path", "audio_path",
    "audio", "wav", "original_filename", "filename",
    "category", "emotion", "category_raw", "speaker",
    "speaker_id", "gender", "transcript", "text",
    "sentence", "content", "intensity", "language",
})


def _meta_extras(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k not in _CONSUMED_META_KEYS}


def iter_flat(root: Path) -> Iterator[RawItem]:
    """扁平目录 + 台账（manifest.json / metadata.jsonl / metadata.csv）。

    台账每条至少要有：
        {"filepath": "a/b.wav", "category": "laugh", "clip_kind": "vocal_burst"}
    可选：speaker_id / gender / transcript / intensity / language / license / target_use

    台账不一定在同层：用户可能只填 `wav/`，而 metadata.jsonl 在上一层，
    所以用 _make_manifest_lookup 向上找，并按文件名兜底匹配。
    """
    if not root.is_dir():
        return
    lookup = _make_manifest_lookup(root)
    for f in sorted(root.rglob("*")):
        if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
            continue
        meta = lookup(f)
        category = meta.get("category") or meta.get("emotion") or f.parent.name or "unknown"
        speaker_id = meta.get("speaker_id") or meta.get("speaker")
        yield RawItem(
            path=f,
            category=str(category),
            category_raw=meta.get("category_raw"),
            speaker_id=speaker_id,
            # 性别优先用台账的，没有就从说话人名推断（female001 → female）。
            # 不推断的话描述会写成「一位用愤怒的语气说」，检索时少一个重要维度。
            gender=meta.get("gender") or _infer_gender(speaker_id),
            # 台账里台词字段名不统一（text / transcript / sentence / content），
            # 用统一取值函数，否则用户那份 metadata.jsonl 的 text 会读不到。
            transcript=_manifest_transcript(meta),
            intensity=meta.get("intensity"),
            language=meta.get("language"),
            extras=_meta_extras(meta),
        )


def iter_by_level(
    root: Path, *, with_audio: bool = True
) -> Iterator[RawItem]:
    """把 `root` 整棵子树的音频都归成同一个类别名。

    与 iter_flat 的区别只有一点：类别名**不看直接父目录**，一律用 `root` 的
    目录名。于是 `b/c/1.wav` 与 `b/d/2.wav` 的类别都是 `b`。

    这是给「选哪一层做分类名」用的：用户在某层目录上选定名称，
    该层以下所有音频共同挂这个类别，便于把一整批素材当成一个可检索的组。

    台账仍然照常读（台词/说话人/性别/强度），查找会向上定位。
    **parquet 分片也要算进来**：它们的音频内嵌在容器里，没有独立文件，
    只扫 AUDIO_EXTS 的话 CSEMOTIONS 这类源上选层级会得到 0 条
    （实测 `count_items("by_level", data)` 返回 0，而 parquet 有 520 条）。

    `with_audio=False` 时 parquet 分支只读元信息列，供预览/计数用。
    """
    if not root.is_dir():
        return
    lookup = _make_manifest_lookup(root)
    category = root.name or "unknown"
    for f in sorted(root.rglob("*")):
        if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
            continue
        meta = lookup(f)
        speaker_id = meta.get("speaker_id") or meta.get("speaker")
        yield RawItem(
            path=f,
            category=category,
            category_raw=meta.get("category_raw"),
            speaker_id=speaker_id,
            gender=meta.get("gender") or _infer_gender(speaker_id),
            transcript=_manifest_transcript(meta),
            intensity=meta.get("intensity"),
            language=meta.get("language"),
            extras=_meta_extras(meta),
        )

    # parquet 容器里的音频：类别同样统一成该层目录名，
    # 但 category_raw 保留原始情绪标签，便于按原值回溯。
    if _has_parquet(root):
        for item in iter_parquet(root, with_audio=with_audio):
            item.category = category
            yield item


def iter_source(
    kind: str,
    root: Path,
    gender_map: dict[str, str] | None = None,
    *,
    with_audio: bool = True,
) -> Iterator[RawItem]:
    """遍历数据源。

    with_audio=False 只用于统计/预览：对 parquet 这类内嵌音频的数据源，
    可以跳过解码，避免为了数条数把几百 MB 音频读进内存。
    普通目录数据源忽略该参数（本来也不读音频内容）。
    """
    kind = (kind or "").lower()
    if kind in ("csemotions", "cse"):
        return iter_csemotions(root, gender_map)
    if kind in ("emogator", "emogators"):
        return iter_emogator(root, gender_map)
    if kind == "flat":
        return iter_flat(root)
    if kind in ("by_level", "level"):
        # 整棵子树统一用 root 的目录名做类别（「选哪一层做分类名」用）
        return iter_by_level(root, with_audio=with_audio)
    if kind == "parquet":
        return iter_parquet(root, gender_map, with_audio=with_audio)
    if kind in ("webdataset", "web", "wds"):
        return iter_webdataset(root, gender_map, with_audio=with_audio)
    raise ValueError(
        f"不支持的数据源类型: {kind}"
        f"（可用: csemotions / emogator / flat / by_level / parquet / webdataset）"
    )


# ---------------------------------------------------------------- WebDataset

# WebDataset 的分片：解压后是 <digits>/<key>.flac + <key>.json，
# 未解压是 <digits>.tar。key 位宽不统一（EmoGator 8 位、NonSpeech7k 6 位），
# 所以一律按「同名配对」而不是按位宽匹配。

_WS_SHARD_RE = re.compile(r"^\d+$")


def _is_webdataset_shard_dir(d: Path) -> bool:
    """<digits>/ 里是否放着 <key>.flac + <key>.json 这种配对。"""
    if not _WS_SHARD_RE.match(d.name):
        return False
    try:
        names = {p.name for p in d.iterdir() if p.is_file()}
    except OSError:
        return False
    if not names:
        return False
    has_json = any(n.lower().endswith(".json") for n in names)
    has_audio = any(Path(n).suffix.lower() in AUDIO_EXTS for n in names)
    return has_json and has_audio


def _is_webdataset_shape(d: Path, depth: int = 2) -> bool:
    """目录是否是 WebDataset 排布（分片目录或分片 tar）。

    允许一层包装：`NonSpeech7k/train/0000/`、`NonSpeech7k/test/0000/`
    这种 train/test 分层的也属于同一子集，必须整体识别，
    否则会被当成 flat，类别变成 "0000" 且丢失子集许可——
    实测会把 CC-BY-NC-SA 的素材错标成可商用。
    """
    if not d.is_dir() or depth < 0:
        return False
    try:
        entries = list(d.iterdir())
    except OSError:
        return False
    for p in entries:
        if p.is_dir():
            if _is_webdataset_shard_dir(p):
                return True
            if depth and _is_webdataset_shape(p, depth - 1):
                return True
        elif p.is_file() and p.suffix.lower() == ".tar" and _WS_SHARD_RE.match(p.stem):
            return True
    return False


def subset_info(name: str) -> dict:
    """按子集名取对应的形态/许可。认不出就返回空 dict。

    先按原样规范化；不中再按驼峰拆分。
    实测 `DeeplyNonverbalVocalization` 直接规范化会连成
    `deeplynonverbalvocalization`，与表里的下划线形式对不上 ——
    这样会**丢掉许可**，回退成全局值（界面默认 apache-2.0），
    把一个 CC-BY-NC-ND 的不可商用数据集标成可商用。
    """
    info = WEB_SUBSETS.get(_subset_key(name))
    if info:
        return info
    return WEB_SUBSETS.get(_subset_key(_camel_to_snake(name)), {})


def subset_info_for_path(path: Path) -> dict:
    """沿目录向上找最近一个能认出的子集名。

    子集有时多包一层（`NonSpeech7k/train/`），只看 name 会匹配不上；
    向上找能在 `train` 未命中时回退到 `NonSpeech7k`。
    """
    for p in [path, *path.parents]:
        info = subset_info(p.name)
        if info:
            return info
    return {}


def iter_webdataset(
    root: Path,
    gender_map: dict[str, str] | None = None,
    *,
    with_audio: bool = True,
) -> Iterator[RawItem]:
    """遍历 WebDataset 分片（gijs/speech-utterances 这类聚合包）。

    两种形态都支持：
      * 已解压：`<root>/0001/00000000.flac` + `00000000.json`
      * 未解压：`<root>/0001.tar`

    解压目录优先：它没有解压开销，且能直接把路径交给 ingest 流式读盘，
    不必把整段音频读成 bytes（对比 tar 必须走内存）。
    若某个分片既有目录又有 tar，只读目录。

    元信息字段各子集不同，按优先级取：
      类别   emotion → label → style
      说话人 speaker → speaker_id → contributor_id
      性别   gender → 从说话人名推断
      强度   intensity（VIVAE 专有）→ 星级
      台词   text（照常保留）

    `drop_text_values` 里的取值整条跳过（Expresso 的
    "Expressive speech in a neutral style" 是整段双人对话）。
    """
    gender_map = gender_map or {}
    if not root.is_dir():
        return

    info = subset_info_for_path(root)
    yield from _iter_ws_tree(root, info, gender_map, with_audio, depth=2)


def _iter_ws_tree(
    root: Path,
    info: dict,
    gender_map: dict[str, str],
    with_audio: bool,
    *,
    depth: int,
) -> Iterator[RawItem]:
    """递归遍历分片，允许 train/ test/ 这层包装。

    `NonSpeech7k/train/0000/` 与 `NonSpeech7k/0001/` 是同一种东西，
    只是前者多一层 split 目录。不递归就会漏掉整个子集，
    进而让它回退成 flat —— 类别变序号、**许可丢失变可商用**。
    """
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return

    for entry in entries:
        if entry.is_dir():
            # 子集名就在目录名上；认不出（如 train/、test/）就沿用上层。
            # 这样直接对整个聚合包根调用时，每个分片仍能拿到自己子集的
            # 许可，而不是继承根目录那份（根通常匹配不到）。
            sub_info = subset_info(entry.name) or info
            if _is_webdataset_shard_dir(entry):
                yield from _iter_ws_shard_dir(entry, sub_info, gender_map, with_audio)
            elif depth > 0:
                yield from _iter_ws_tree(
                    entry, sub_info, gender_map, with_audio, depth=depth - 1
                )
        elif entry.suffix.lower() == ".tar" and _WS_SHARD_RE.match(entry.stem):
            # 有对应解压目录就跳过 tar，避免同一批素材读两遍
            if entry.with_suffix("").is_dir():
                continue
            yield from _iter_ws_tar(entry, info, gender_map, with_audio)


def _ws_should_skip(meta: dict, info: dict) -> bool:
    """这条 json 是否整条跳过。

    只按子集配置的 `drop_text_values` 精确匹配 text。
    当前只有 Expresso 用到：text 恰为
    "Expressive speech in a neutral style" 的那批是整段双人对话，
    用户听过音频确认不要；同子集里其余条目是真台词，全部保留。

    注意不要用 style=="default" 代替：实测 style=default 有 825 条，
    其中大部分是 type=read 的带台词句子，会误伤。
    """
    drop = info.get("drop_text_values")
    if not drop:
        return False
    text = meta.get("text")
    if text is None:
        return False
    return str(text).strip() in drop


def _ws_meta_to_item(
    *,
    display_path: Path,
    meta: dict,
    info: dict,
    gender_map: dict[str, str],
    file_path: Path | None,
    audio_bytes: bytes | None,
    origin: str,
) -> RawItem:
    """把 WebDataset 的一条 json 变成 RawItem。"""
    category = (
        meta.get("emotion")
        or meta.get("label")
        or meta.get("style")
        or "unknown"
    )
    speaker = (
        meta.get("speaker")
        or meta.get("speaker_id")
        or meta.get("contributor_id")
    )
    speaker = str(speaker) if speaker is not None else None

    gender = meta.get("gender")
    if not gender and speaker:
        gender = gender_map.get(speaker) or _infer_gender(speaker)

    intensity = None
    raw_intensity = meta.get("intensity")
    if raw_intensity is not None:
        intensity = _WEB_INTENSITY.get(str(raw_intensity).strip().lower())

    # 台词照常保留：用户确认其余条目的 text 是真台词，要留。
    transcript = meta.get("text")
    if transcript is not None:
        transcript = str(transcript).strip() or None

    return RawItem(
        path=file_path or display_path,
        category=str(category),
        category_raw=str(category),
        speaker_id=speaker,
        gender=str(gender).lower() if gender else None,
        transcript=transcript,
        intensity=intensity,
        # 语种跟着子集走：Expresso 是英语语音，其余副语言子集留空。
        # 元信息里若自带 language 则优先用它。
        language=meta.get("language") or info.get("language"),
        audio_bytes=audio_bytes,
        extras={
            "dataset": meta.get("dataset") or _subset_key(display_path.parent.name),
            "origin": origin,
            # 许可跟着子集走，由 ingest 用它覆盖单元级 license
            **({"license": info["license"]} if info.get("license") else {}),
        },
    )


def _iter_ws_shard_dir(
    shard: Path,
    info: dict,
    gender_map: dict[str, str],
    with_audio: bool,
) -> Iterator[RawItem]:
    """读一个已解压的分片目录。音频交路径给 ingest，流式读盘。"""
    for f in sorted(shard.iterdir()):
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS:
            continue
        meta_path = f.with_suffix(".json")
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _ws_should_skip(meta, info):
            continue
        yield _ws_meta_to_item(
            display_path=f,
            meta=meta,
            info=info,
            gender_map=gender_map,
            file_path=f,
            audio_bytes=None,   # 让 ingest 从磁盘读，避免整段进内存
            origin=str(f.relative_to(shard)),
        )


def _iter_ws_tar(
    tar_path: Path,
    info: dict,
    gender_map: dict[str, str],
    with_audio: bool,
) -> Iterator[RawItem]:
    """读一个未解压的 WebDataset tar。

    WebDataset 的写入顺序是「音频在前、json 在后」，
    所以要先缓存音频，等配对的 json 到了再产出。
    同一 key 只保留一份，产出后立即释放。
    """
    import tarfile

    pending: dict[str, bytes] = {}
    try:
        tf = tarfile.open(tar_path)
    except (OSError, tarfile.TarError):
        return

    with tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if name.startswith("./"):
                name = name[2:]
            if "/" in name:          # 分片 tar 里通常是平铺，防御一下
                name = name.rsplit("/", 1)[-1]
            key = Path(name).stem
            ext = Path(name).suffix.lower()

            if ext in AUDIO_EXTS:
                if not with_audio:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                pending[key] = fh.read()
                continue

            if ext != ".json":
                continue

            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                meta = json.loads(fh.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                pending.pop(key, None)
                continue

            # 整条跳过的条目，连同已缓存的音频一起丢弃，避免占内存
            if _ws_should_skip(meta, info):
                pending.pop(key, None)
                continue

            audio = pending.pop(key, None)
            if with_audio and audio is None:
                continue

            yield _ws_meta_to_item(
                display_path=tar_path.with_name(f"{tar_path.name}#{key}"),
                meta=meta,
                info=info,
                gender_map=gender_map,
                file_path=None,
                audio_bytes=audio,
                origin=f"{tar_path.name}:{key}",
            )

        pending.clear()


def webdataset_stats(root: Path) -> dict:
    """只读 json，给出条数与类别分布（不解码音频）。"""
    total = 0
    cats: dict[str, int] = {}
    speakers: dict[str, int] = {}
    for item in iter_webdataset(root, with_audio=False):
        total += 1
        cats[item.category] = cats.get(item.category, 0) + 1
        if item.speaker_id:
            speakers[item.speaker_id] = speakers.get(item.speaker_id, 0) + 1
    return {
        "files": len(find_webdataset_shards(root)),
        "total": total,
        "categories": cats,
        "speakers": speakers,
    }


def find_webdataset_shards(root: Path) -> list[Path]:
    """有哪些分片（解压目录 + 未被目录覆盖的 tar）。"""
    out: list[Path] = []
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and _is_webdataset_shard_dir(entry):
            out.append(entry)
        elif entry.suffix.lower() == ".tar" and _WS_SHARD_RE.match(entry.stem):
            if not entry.with_suffix("").is_dir():
                out.append(entry)
    return out


# ---------------------------------------------------------------- parquet

# HF datasets 的 Audio 列常见两种形态：
#   struct<bytes: binary, path: string>   （Audio feature 的标准编码）
#   裸 binary                              （有人直接存 wav/flac 字节）
_AUDIO_COLUMNS = ("audio", "wav", "waveform", "file", "sound")
_TEXT_COLUMNS = ("text", "transcript", "sentence", "content")
_EMOTION_COLUMNS = ("emotion", "label", "category", "emotion_label")
_SPEAKER_COLUMNS = ("speaker", "speaker_id", "spk", "client_id")


def find_parquet_files(root: Path) -> list[Path]:
    """目录下（递归）有哪些 parquet 分片。"""
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in PARQUET_EXTS
    )


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """按候选名找列，大小写不敏感。"""
    low = {c.lower(): c for c in columns}
    for name in candidates:
        if name in low:
            return low[name]
    return None


def _libs() -> tuple:
    """惰性导入 pyarrow / soundfile，让没装 pyarrow 的环境照常启动。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "读取 parquet 需要 pyarrow，请先安装：pip install pyarrow"
        ) from exc
    return pq


def _infer_gender(speaker: str | None) -> str | None:
    """从说话人名推断性别。

    实测数据集用 female001..005 / male001..005 命名，前缀即性别，
    覆盖 100%，比让用户手工填 gender_map 靠谱得多。
    认不出就返回 None（不瞎猜）。
    """
    if not speaker:
        return None
    s = speaker.strip().lower()
    if s.startswith(("female", "woman", "girl", "f0", "f_")) or s.endswith("_f"):
        return "female"
    if s.startswith(("male", "man", "boy", "m0", "m_")) or s.endswith("_m"):
        return "male"
    return None


def iter_parquet(
    root: Path,
    gender_map: dict[str, str] | None = None,
    *,
    with_audio: bool = True,
) -> Iterator[RawItem]:
    """遍历 parquet 分片里的音频（HF datasets 常见格式）。

    要点：
      * **流式**：按 row group 读取，不把 323MB 一次性读进内存
      * **不落中间文件**：音频字节直接交给 ingest，避免磁盘翻倍
      * `with_audio=False` 时只读元信息列（数条数用，快得多）

    文件名/路径不可信：实测某数据集文件名是 `female003_jolliest_01.wav`，
    但 `emotion` 列是 `playfulness`（jolliest 不在情绪词表里）。
    所以类别一律取列值，不解析文件名。
    """
    pq = _libs()
    gender_map = gender_map or {}
    files = find_parquet_files(root)
    if not files:
        return

    for pf_path in files:
        pf = pq.ParquetFile(pf_path)
        schema = pf.schema_arrow
        names = [f.name for f in schema]

        audio_col = _pick(names, _AUDIO_COLUMNS)
        if audio_col is None:
            # 这个分片没有音频列，跳过而不是报错（可能有 _index / 元信息分片）
            continue
        text_col = _pick(names, _TEXT_COLUMNS)
        emo_col = _pick(names, _EMOTION_COLUMNS)
        spk_col = _pick(names, _SPEAKER_COLUMNS)

        want = [audio_col] + [c for c in (text_col, emo_col, spk_col) if c]
        if not with_audio:
            want = [c for c in want if c != audio_col]

        row_no = 0
        for batch in pf.iter_batches(batch_size=64, columns=want):
            data = batch.to_pydict()
            n = len(next(iter(data.values()))) if data else 0
            for i in range(n):
                row_no += 1
                speaker = str(data[spk_col][i]) if spk_col else None
                raw_emo = data[emo_col][i] if emo_col else None
                category = _canonical_emotion(str(raw_emo)) if raw_emo else "unknown"
                text = data[text_col][i] if text_col else None

                payload = None
                rel = ""
                if with_audio:
                    payload, rel = _extract_audio_bytes(data[audio_col][i])

                yield RawItem(
                    # 没有独立文件：用「分片路径#行号」作标识，便于回溯
                    path=pf_path.with_name(f"{pf_path.name}#{row_no}"),
                    category=category,
                    category_raw=str(raw_emo) if raw_emo else None,
                    speaker_id=speaker,
                    gender=gender_map.get(speaker or "") or _infer_gender(speaker),
                    transcript=str(text).strip() if text else None,
                    audio_bytes=payload,
                    extras={"parquet": str(pf_path), "row": row_no, "origin": rel},
                )


def _extract_audio_bytes(value) -> tuple[bytes | None, str]:
    """从 Audio 列取字节。兼容 {bytes, path} 与裸 bytes。"""
    if value is None:
        return None, ""
    if isinstance(value, dict):
        payload = value.get("bytes")
        rel = value.get("path") or ""
        return (payload if isinstance(payload, (bytes, bytearray)) else None), str(rel)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), ""
    return None, ""


def parquet_stats(root: Path) -> dict:
    """只读元信息，快速给出条数与类别分布（不解码音频）。"""
    pq = _libs()
    files = find_parquet_files(root)
    total = 0
    cats: dict[str, int] = {}
    speakers: dict[str, int] = {}
    for pf_path in files:
        pf = pq.ParquetFile(pf_path)
        names = [f.name for f in pf.schema_arrow]
        emo_col = _pick(names, _EMOTION_COLUMNS)
        spk_col = _pick(names, _SPEAKER_COLUMNS)
        cols = [c for c in (emo_col, spk_col) if c]
        if not cols:
            total += pf.metadata.num_rows
            continue
        for batch in pf.iter_batches(batch_size=1024, columns=cols):
            d = batch.to_pydict()
            n = len(next(iter(d.values()))) if d else 0
            total += n
            if emo_col:
                for v in d[emo_col]:
                    k = _canonical_emotion(str(v)) if v else "Unknown"
                    cats[k] = cats.get(k, 0) + 1
            if spk_col:
                for v in d[spk_col]:
                    k = str(v)
                    speakers[k] = speakers.get(k, 0) + 1
    return {"files": len(files), "total": total, "categories": cats, "speakers": speakers}


def _has_audio(d: Path, limit: int = 200) -> bool:
    for i, f in enumerate(d.rglob("*")):
        if i > limit:
            return False
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            return True
    return False


def _dir_has_parquet(d: Path) -> bool:
    """该目录的**直接子项**里有没有 parquet 文件（只看一层）。

    分层拆分必须要单层判断：`data/*.parquet` 与
    `root/data/*.parquet` 是两种不同情形，递归会把容器也判成 parquet。
    """
    try:
        return any(
            f.is_file() and f.suffix.lower() in PARQUET_EXTS for f in d.iterdir()
        )
    except OSError:
        return False


def _has_parquet(d: Path) -> bool:
    """目录下（递归）有没有 parquet 分片。"""
    try:
        return any(
            f.is_file() and f.suffix.lower() in PARQUET_EXTS
            for f in d.rglob("*")
        )
    except OSError:
        return False


def _is_csemotions_shape(d: Path) -> bool:
    """<speaker>/<emotion>/*.wav"""
    for p in sorted(d.iterdir()):
        if not p.is_dir():
            continue
        for q in sorted(p.iterdir()):
            if q.is_dir() and _looks_like_emotion(q.name) and _has_audio(q):
                return True
    return False


def _dir_has_direct_audio(d: Path) -> bool:
    """该目录的**直接子项**里有没有音频文件（只看一层）。"""
    try:
        return any(
            f.is_file() and f.suffix.lower() in AUDIO_EXTS for f in d.iterdir()
        )
    except OSError:
        return False


def _is_label_shape(d: Path) -> bool:
    """<label>/*.mp3（EmoGator 形态）。

    判据要同时满足两点，否则容器目录会被误判：
      1. 子目录里**直接**有音频（不能只是更深的层级）
      2. 这些子目录**内部不再嵌套**带音频的子目录

    第 2 点很关键。混合目录 `root/CSEMOTIONS/S01/Angry/*.wav` 的结构是
    `<容器>/<speaker>/<emotion>/`，容器本身也是"子目录里有音频"，
    但它其实是包装层，不是 label 层。
    """
    try:
        subdirs = [p for p in d.iterdir() if p.is_dir()]
    except OSError:
        return False
    if not any(_dir_has_direct_audio(p) for p in subdirs):
        return False
    # 有任何子目录内部还嵌套带音频的子目录 → 这是容器，不是 label 层
    for p in subdirs:
        try:
            if any(q.is_dir() and _has_audio(q) for q in p.iterdir()):
                return False
        except OSError:
            continue
    return True


def _root_has_audio(d: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() in AUDIO_EXTS for f in d.iterdir())


def guess_kind(d: Path) -> str:
    """判断**一个目录**的排布形态。

    顺序很关键：先看目录形状，最后才看 manifest。
    因为 EmoGator 的目录也可能带一个 manifest.json，若先判 manifest
    就会把 <label>/*.mp3 误判成 flat。

    注意：这个函数只看单层。混合目录请用 plan_import()，
    它会逐个识别子目录。
    """
    if not d.is_dir():
        return "flat"
    # parquet 分片直接放在本层 → 这是一个 parquet 单元。
    # 只用单层判定：若子目录里才有 parquet，说明本层是容器，
    # 应交给 plan_import 继续往下拆，而不是把整根当成一个单元。
    if _dir_has_parquet(d):
        return "parquet"
    # WebDataset 必须排在 emogator 之前：它的分片目录（0001/）里放着
    # <key>.flac + <key>.json，形状上也是「子目录里有音频」，
    # 不先判就会被当成 <label>/ 排布，类别变成 "0001"。
    if _is_webdataset_shape(d):
        return "webdataset"
    if _is_csemotions_shape(d):
        return "csemotions"
    if _is_label_shape(d):
        return "emogator"
    # 根级平铺：没有 label 子目录，类别无从推断
    if _root_has_audio(d) or any((d / n).is_file() for n in MANIFEST_NAMES):
        return "flat"
    return "flat"


def count_items(kind: str, path: Path) -> int:
    """数一个单元有多少条素材。

    两个数据源要特殊处理，因为它们的音频都在容器里：
      * parquet      —— 走 iter_source 会把每条音频解出来（320MB 级）
      * webdataset   —— tar 形态必须读完音频才能配到 json
    只为数条数不值得，这里都只读元信息。
    """
    if kind == "parquet":
        try:
            return int(parquet_stats(path)["total"])
        except Exception:  # noqa: BLE001 - 计数失败不应让整个预览挂掉
            return 0
    if kind in ("webdataset", "web", "wds"):
        try:
            return sum(1 for _ in iter_webdataset(path, with_audio=False))
        except Exception:  # noqa: BLE001
            return 0
    if kind in ("by_level", "level"):
        # 递归整棵子树，只数音频文件，不解码。
        # parquet 分片里的音频内嵌在容器里，没有独立文件，
        # 只数 AUDIO_EXTS 会漏掉它们（实测 CSEMOTIONS 会算出 0 条）。
        n = sum(
            1
            for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        )
        if _has_parquet(path):
            try:
                n += int(parquet_stats(path)["total"])
            except Exception:  # noqa: BLE001 - 计数失败不应让整个预览挂掉
                pass
        return n
    return len(list(iter_source(kind, path)))


# ---------------------------------------------------------------- 自动拆分

# 容器递归的最大深度。实测 HF 下载 CSEMOTIONS 会多出两层 `data/`：
#     CSEMOTIONS/data/CSEMOTIONS/data/*.parquet
# 按 3 层留余量到 6，再多说明是异常嵌套，宁可停下也不要扫全盘。
_MAX_PLAN_DEPTH = 6


@dataclass
class ImportPlan:
    """一个待导入的单元：目录 + 形态 + 派生出的类型/线路。"""

    path: Path
    kind: str
    label: str                 # 界面上显示的相对名
    clip_kind: str
    target_use: str
    count: int = 0
    reason: str = ""          # 为什么这样判定（给用户看）
    # 单元自带的许可。为空表示沿用导入请求里的全局 license。
    # 聚合数据集必须用它：一个根目录下的子集许可可能各不相同，
    # 统一按全局值落库会把可商用的素材标成不可商用（或反之）。
    license: str = ""

    def as_dict(self, root: Path | None = None) -> dict:
        rel = self.path.relative_to(root).as_posix() if root and root in self.path.parents else self.path.name
        return {
            "path": str(self.path),
            "relpath": rel,
            "kind": self.kind,
            "clip_kind": self.clip_kind,
            "target_use": self.target_use,
            "count": self.count,
            "reason": self.reason,
            "license": self.license,
        }


def resolve_level(root: Path, level: str | None, max_up: int = 8) -> Path | None:
    """把「分类层级」解析成一个实际目录。

    `level` 可以是：
      * 空                —— 不启用（返回 None）
      * 绝对路径           —— 必须等于 root 或 root 的某个祖先
      * 目录名（如 "b"）    —— 从 root 开始向上找第一个同名祖先

    只允许 root 及其祖先：这是「选哪一层做分类名」的语义边界，
    也避免凭一个名字就把扫描范围扩到无关目录。
    """
    if not level:
        return None
    root = root.resolve()
    chain = [root]
    cur = root
    for _ in range(max_up):
        if cur.parent == cur:
            break
        cur = cur.parent
        chain.append(cur)

    raw = str(level).strip()
    candidate = Path(raw)
    if candidate.is_absolute() or "/" in raw or "\\" in raw:
        try:
            target = candidate.resolve()
        except OSError:
            return None
        return target if target in chain else None

    for d in chain:
        if d.name == raw:
            return d
    return None


def level_choices(root: Path, max_up: int = 8, cap: int = 20000) -> list[dict]:
    """列出 root 及其祖先，供界面选「用哪层做分类名」。

    顺序由近到远：root 自己在第一项。

    每一层带一个**有上限**的条数。预算分两档：
      * root 自己 —— 用户真正关心的层，给足 0.3s
      * 祖先层     —— 只是给个量级参考，0.05s 内扫多少算多少
    祖先可能指向 `D:\\` 或用户主目录，统一给足预算会让每次预览多等
    近 1 秒（实测 4 层 × 0.15s）。
    截断时 `count_truncated=True`，界面显示「≥N」而不是精确值。
    """
    root = root.resolve()
    out: list[dict] = []
    cur = root
    for i in range(max_up + 1):
        budget = 0.30 if i == 0 else 0.05
        n, truncated = count_audio_capped(cur, cap=cap, budget_sec=budget)
        out.append(
            {
                "name": cur.name or str(cur),
                "path": str(cur),
                "count": n,
                "count_truncated": truncated,
                "is_root": cur == root,
            }
        )
        if cur.parent == cur:
            break
        cur = cur.parent
    return out


def _parquet_rows(path: Path) -> int:
    """只读 parquet footer 的行数，不解码、不遍历数据。

    用于「这一层大概有多少条」的估算：`parquet_stats` 会遍历所有 batch
    去统计类别分布，对 323MB 的分片太慢；这里只要行数，
    `ParquetFile` 只读文件尾部的元信息。
    """
    try:
        pq = _libs()
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 - 缺 pyarrow / 文件损坏都不该让预览挂掉
        return 0


def count_audio_capped(
    path: Path, cap: int = 20000, budget_sec: float = 0.15
) -> tuple[int, bool]:
    """数音频文件，同时受「最多扫 cap 个目录项」和「最多花 budget_sec 秒」约束。

    返回 (条数, 是否被截断)。用于界面展示「这一层大概有多少条」：
    祖先层可能指向整个磁盘，不设上限会把预览拖死
    （实测：对一个含几十万文件的临时目录计数，预览从 0.1s 变成 5s）。

    **parquet 分片也要算**：它们的音频内嵌在容器里，没有独立音频文件，
    只数 AUDIO_EXTS 会让 CSEMOTIONS 这类源在层级选择器上显示「0 条」，
    看起来像没数据（实测 root 层 count=0，实际有 520 条）。
    行数从 footer 读，比遍历数据便宜得多。
    """
    n = 0
    seen = 0
    deadline = time.monotonic() + budget_sec
    stack = [path]
    parquet_files: list[str] = []
    while stack:
        if time.monotonic() > deadline:
            return n, True
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    seen += 1
                    if seen > cap:
                        return n, True
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            ext = os.path.splitext(e.name)[1].lower()
                            if ext in AUDIO_EXTS:
                                n += 1
                                if n >= cap:
                                    return n, True
                            elif ext in PARQUET_EXTS:
                                parquet_files.append(e.path)
                    except OSError:
                        continue
        except OSError:
            continue

    # 遍历结束后再数 parquet 行数，同样受时间预算约束。
    # 超预算就标记截断，界面显示「≥N」而不是假装精确。
    for p in parquet_files:
        if time.monotonic() > deadline:
            return n, True
        n += _parquet_rows(Path(p))
        if n >= cap:
            return n, True
    return n, False


def plan_import(
    root: Path,
    *,
    assume_use: str | None = None,
    category_level: str | None = None,
    _depth: int = 0,
) -> list[ImportPlan]:
    """把任意目录拆成若干「同构的导入单元」。

    为什么要拆：一个根目录里可能同时有
        CSEMOTIONS/<spk>/<emotion>/*.wav   （说话片段）
        EmoGator/<label>/*.mp3             （副语言）
        custom/*.wav                       （平铺，靠 manifest 或目录名）
    之前只允许整体选一种类型，结果 CSEMOTIONS 的部分会被按 EmoGator 解读，
    类别变成说话人名 —— 数据没丢，但语义全错。

    规则：
      1. 指定了 `category_level` → 整层统一处理（见下）
      2. 根目录本身是合法排布（<spk>/<emo>/、<label>/ 或平铺带音频）→ 整个根作为一个单元
      3. 否则（根目录只是容器）→ 对每个子目录递归 plan，合并结果
      4. 单元的类型由排布决定：
            csemotions → clip_kind=speech     → target_use=indextts
            emogator   → clip_kind=vocal_burst → target_use=seedvc
            flat       → 按 assume_use（默认 indextts）

    `category_level` 指定后**不再做形状识别**：那个目录及其整棵子树的所有音频
    统一用该目录名做类别。因为这时用户已经明确表达了「我要把它们当成一组」，
    再按子目录名拆开反而违背意图。
    """
    if not root.is_dir():
        return []

    if category_level:
        target = resolve_level(root, category_level)
        if target is None or not target.is_dir():
            return []
        plan = _plan_for(target, "by_level", target.name or "根目录", assume_use)
        plan.count = count_items("by_level", target)
        return [plan] if plan.count else []

    # 情况 0：WebDataset 聚合包（gijs/speech-utterances 这类）。
    # 它的每个子目录是**独立数据集，许可各不相同**（EmoGator Apache-2.0 可商用，
    # Expresso CC-BY-NC 不可商用）。整包当一个单元会让许可全部取全局值，
    # 把 NC 素材标成可商用 —— 实测过，必须按子集拆。
    if _is_webdataset_shape(root):
        if subset_info(root.name):
            # 用户直接指到了某个子集（EmoGator/、NonSpeech7k/）：整根一个单元
            plan = _plan_for(root, "webdataset", root.name, assume_use)
            plan.count = count_items("webdataset", root)
            return [plan] if plan.count else []
        # 聚合包根：每个子目录（含 train/test 那种再嵌套的）各自成一个单元
        plans = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            if not _is_webdataset_shape(sub):
                continue
            plan = _plan_for(sub, "webdataset", sub.name, assume_use)
            plan.count = count_items("webdataset", sub)
            if plan.count:
                plans.append(plan)
        if plans:
            return plans

    # 情况 1：根本身就是一个排布（有子目录组成的 <spk>/<emo>/ 或 <label>/，
    #         或根级平铺音频、或带 manifest）。这时整根作为一个单元。
    kind = guess_kind(root)
    root_is_unit = (
        kind != "flat"
        or _root_has_audio(root)
        or any((root / n).is_file() for n in MANIFEST_NAMES)
    )
    if root_is_unit:
        plan = _plan_for(root, kind, root.name or "根目录", assume_use)
        plan.count = count_items(kind, root)
        if plan.count:
            return [plan]
        # 扫出 0 条说明判断不对（例如子目录名恰好都不是情绪词），继续往下拆

    # 情况 2：当作容器，逐个子目录识别各自的排布
    plans: list[ImportPlan] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        # parquet 分片目录里没有普通音频文件，不能只用 _has_audio 过滤
        if not (_has_audio(sub) or _has_parquet(sub)):
            continue
        sub_kind = guess_kind(sub)
        plan = _plan_for(sub, sub_kind, sub.name, assume_use)
        plan.count = count_items(sub_kind, sub)
        if plan.count:
            plans.append(plan)
        elif _depth < _MAX_PLAN_DEPTH:
            # 本层扫不出东西 → 它只是又一层容器，继续往下沉。
            # 实测踩过：HF 下载 CSEMOTIONS 会解成
            #     CSEMOTIONS/data/CSEMOTIONS/data/*.parquet
            # 中间两层 `data` 的直接子项里都没有 parquet，`guess_kind`
            # 一律返回 flat，扫 wav 得 0 条。只拆一层就返回空列表，
            # 前端只会看到「该目录里没有找到任何音频素材」，
            # 而真正的 parquet 在第三层，永远走不到。
            plans.extend(
                plan_import(sub, assume_use=assume_use, _depth=_depth + 1)
            )

    return plans


def _plan_for(path: Path, kind: str, label: str, assume_use: str | None) -> ImportPlan:
    if kind == "csemotions":
        return ImportPlan(
            path=path,
            kind=kind,
            label=label,
            clip_kind="speech",
            target_use="indextts",
            reason="<说话人>/<情绪>/ 排布，按说话片段入库",
        )
    if kind == "emogator":
        # 有台词就是语言内容，不能当副语言。
        # 判据优先于目录形状：用户的目录是 `<说话人>/<情绪>/`，
        # 形状上恰好符合 `<类别>/`，不按内容判就会把整批台词塞进 SeedVC。
        if _probe_has_transcript(path):
            return ImportPlan(
                path=path,
                kind=kind,
                label=label,
                clip_kind="speech",
                target_use="indextts",
                reason="目录按类别分层，但素材带台词，按说话片段入库",
            )
        return ImportPlan(
            path=path,
            kind=kind,
            label=label,
            clip_kind="vocal_burst",
            target_use="seedvc",
            reason="<类别>/ 排布，按副语言入库",
        )
    if kind == "parquet":
        # parquet 里 emotion/speaker/text 都是列，天然就是「说话片段」形态。
        # 它与 CSEMOTIONS 是同一份数据的两种打包方式，所以按同样方式入库。
        use = assume_use or "indextts"
        return ImportPlan(
            path=path,
            kind=kind,
            label=label,
            clip_kind="speech" if use == "indextts" else "vocal_burst",
            target_use=use,
            reason="parquet 数据集（音频内嵌），按列取 情绪/说话人/台词",
        )
    if kind == "webdataset":
        # 聚合包里的每个子集有自己的许可与线路（EmoGator 可商用、
        # Expresso 是对话语音且不可商用），所以许可跟着单元走。
        # 用向上查找：NonSpeech7k/train/ 这种多包一层的也要认得出。
        info = subset_info_for_path(path)
        use = info.get("target_use") or assume_use or "seedvc"
        return ImportPlan(
            path=path,
            kind=kind,
            label=label,
            clip_kind=info.get("clip_kind")
            or ("speech" if use == "indextts" else "vocal_burst"),
            target_use=use,
            license=info.get("license", ""),
            reason=info.get("note") or "WebDataset 分片（<key>.flac + <key>.json）",
        )
    # flat / by_level：有台词就是语言内容，优先于时长判断。
    # 时长只在「没有台词」时才用来区分副语言与短说话片段。
    if assume_use is None and _probe_has_transcript(path):
        use = "indextts"
        reason = "素材带台词，按说话片段入库"
    else:
        use = assume_use or _infer_use_for_flat(path)
        reason = (
            "靠 manifest 或父目录名取类别，按说话片段入库"
            if use == "indextts"
            else "无台词，按素材时长推断为副语言"
        )
    if kind in ("by_level", "level"):
        reason = f"整层「{label}」统一作为类别名；" + reason
    return ImportPlan(
        path=path,
        kind=kind,
        label=label,
        clip_kind="speech" if use == "indextts" else "vocal_burst",
        target_use=use,
        reason=reason,
    )


def _find_manifest_dir(path: Path, max_up: int = 4) -> Path | None:
    """从 path 向上找最近一个带清单文件的目录。

    用户常把「总目录」填进来（音频在 `wav/`，清单在根），
    所以清单不一定就在被导入的那一层。
    """
    cur = path if path.is_dir() else path.parent
    for _ in range(max_up + 1):
        if any((cur / n).is_file() for n in MANIFEST_NAMES):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


# 台账里视为「台词」的字段名
_TRANSCRIPT_FIELDS = ("text", "transcript", "sentence", "content")


def _probe_has_transcript(path: Path) -> bool:
    """这个目录下的素材是否带台词（语言内容）。

    这是判定线路的第一依据，优先于时长：
      * 有声语言（TTS 参考音）→ IndexTTS
      * 无台词（笑声/咳嗽/叹气）→ SeedVC
    副语言素材不可能有台词，所以「有台词」比「时长短」更本质。

    只采样一部分，避免为了判定把整份清单和音频都读一遍。
    """
    root = _find_manifest_dir(path)
    if root is None:
        return False

    manifest = _load_manifest(root)
    if not manifest:
        return False

    checked = 0
    for i, f in enumerate(path.rglob("*")):
        if i > 200 or checked >= 24:
            break
        if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
            continue
        checked += 1
        try:
            rel = f.relative_to(root).as_posix()
        except ValueError:
            rel = f.name
        rec = manifest.get(rel) or manifest.get(f.name)
        if not rec:
            continue
        for key in _TRANSCRIPT_FIELDS:
            v = rec.get(key)
            if v is not None and str(v).strip():
                return True
    return False


def _infer_use_for_flat(path: Path) -> str:
    """自定义（flat）目录该走哪条线：看素材实际时长。

    两条线的时长窗口是互斥的：
        IndexTTS 3–15 秒，SeedVC 0.3–5 秒
    若 flat 单元一律按 indextts 处理，1 秒级的素材会被当成"太短"
    静默丢掉（实测混合目录里 custom/*.wav 就这样消失了）。
    这里按中位时长推断：多数落在 SeedVC 窗口内就走 seedvc。
    """
    durs: list[float] = []
    for i, f in enumerate(path.rglob("*")):
        if i > 60 or len(durs) >= 12:
            break
        if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
            continue
        try:
            y, sr = A.load_audio(f)
            durs.append(len(y) / sr)
        except Exception:  # noqa: BLE001 - 读不了就跳过，让后续流程报错
            continue
    if not durs:
        return "indextts"
    durs.sort()
    median = durs[len(durs) // 2]
    return "seedvc" if median <= 5.0 else "indextts"
