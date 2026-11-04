"""原地重分类：把被误判成副语言的说话素材改回 IndexTTS。

**背景**
早期版本不读台账（metadata.jsonl / metadata.csv），线路只看目录形状，
于是 `<说话人>/<情绪>/*.wav` 这种形状被当成 `<类别>/*.wav` 判成 emogator，
带台词的中文语音全被塞进 SeedVC 的 0.3–5 秒窗口。

**为什么不能靠重导修正**
filepath 由线路决定（`audio/vocal_burst/` vs `audio/csemotions/`），
去重靠 `ON CONFLICT(filepath)`。线路一变路径就变，旧的匹配不上新的，
重导只会多出一份，不会修正。

**这个脚本做什么**
逐条按 `source_file` 回溯原始音频 → 重新读台账拿台词 →
有台词且当前判成副语言的，原地改成 speech / indextts，
并重建 description、notes、embedding。人工字段（verified 等）不动。

用法（在 backend 目录下）：

    py -m tools.reclassify --root <库根>              # 只看，列出会改哪些
    py -m tools.reclassify --root <库根> --apply      # 真正写入
    py -m tools.reclassify --root <库根> --limit 20   # 先小范围试

`--root` 省略时取 `VOICE_LIBRARY_ROOT` 或默认 `voice_library/`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 `py tools/reclassify.py` 运行（不要求 -m）
BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from voicelib import sources as S  # noqa: E402
from voicelib import tags as T  # noqa: E402
from voicelib.config import Config  # noqa: E402
from voicelib.db import Database  # noqa: E402
from voicelib.embedder import Encoder  # noqa: E402


def _duration_note(sec: float, use: str) -> str | None:
    """按目标线路的时长窗口，给出（或清掉）时长提示。"""
    lo, hi = (3.0, 15.0) if use == "indextts" else (0.3, 5.0)
    if sec > hi:
        return f"时长 {sec:.1f}s 超出 {lo:g}–{hi:g}s 范围，使用前需人工截取"
    if sec < lo:
        return f"时长 {sec:.1f}s 不足 {lo:g}s，如需用于该线路需人工加长"
    return None


def _rebuild_notes(old: str | None, dur: float, use: str) -> str | None:
    """重算 notes：时长类提示按新线路重算，其它（如段数）原样保留。

    不直接清空：notes 里可能有「检出 N 段发声」这类与线路无关的音频属性，
    那些是 audio 层面的结论，换线路也成立。
    """
    kept: list[str] = []
    for part in (old or "").split("；"):
        part = part.strip()
        if not part:
            continue
        if "超出" in part and "范围" in part:
            continue
        if "不足" in part and "如需用于该线路" in part:
            continue
        kept.append(part)
    note = _duration_note(dur, use)
    if note:
        kept.insert(0, note)
    return "；".join(kept) or None


class _LookupCache:
    """按目录缓存台账查询，避免 4000+ 条各读一遍 jsonl。

    **返回 None 表示「无法判定」，不是「没有台词」。**
    这两种情况必须分开，否则会把 webdataset 这类源全部误判：
    它们的元信息在同名 `.json` 里（如 `00000000.flac` 配 `00000000.json`），
    根本没有 manifest.json，_make_manifest_lookup 一律查不到。
    早期版本把「查不到」当成「台词丢了」，导致 66901 条副语言素材
    被判成"需要改"，且会把它们的描述性 text 清成 NULL。
    """

    def __init__(self) -> None:
        self._by_dir: dict[Path, object] = {}

    def get(self, f: Path) -> dict | None:
        d = f.parent
        fn = self._by_dir.get(d)
        if fn is None:
            fn = S._make_manifest_lookup(d)
            self._by_dir[d] = fn
        meta = fn(f)  # type: ignore[operator]
        if meta:
            return meta
        # 台账查不到时，再试 WebDataset 的同名 .json
        # （`00000000.flac` 配 `00000000.json`）。这类源没有 manifest，
        # 不补这一条的话 66901 条 webdataset 素材全都无从判定，
        # 它们的 license 与 Expresso 的线路就永远修不好。
        return _ws_meta(f)


def _ws_meta(f: Path) -> dict | None:
    """读 WebDataset 的同名 .json；读不到返回 None。"""
    meta_path = f.with_suffix(".json")
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_ws_item(meta: dict) -> bool:
    """是不是 WebDataset 的元信息。

    判据用 `dataset` 字段，**不能用 `__key__` / `__url__`**。
    实测六个子集解压出来的 json：

        Expresso          → 有 __key__/__url__（tar 内遗留）+ dataset
        EmoGator          → 只有 dataset 等普通字段
        VocalSound        → 只有 dataset
        VIVAE             → 只有 dataset
        DeeplyNonverbal   → 只有 dataset
        NonSpeech7k       → 只有 dataset

    `__key__` 只存在于 tar 包内的条目，解压后除 Expresso 外都没有。
    早期用 `__key__` 判断，导致 5/6 的子集落回「有台词 → IndexTTS」，
    把 `"A female panting"` 当成台词，副语言被误改成 speech。
    """
    return bool(meta.get("dataset")) or "__key__" in meta


def _ws_subset_info(meta: dict, src: Path) -> dict:
    """按元信息的 `dataset` 找子集配置，找不到再按路径推断。

    优先用 `dataset`：它是数据自带的，不依赖解压后的目录名怎么摆。
    路径兜底是为了兼容老数据（早期导出的 json 可能没这个字段）。
    """
    name = meta.get("dataset")
    if name:
        info = S.subset_info(str(name))
        if info:
            return info
    return S.subset_info_for_path(src)


def _resolve_level_dir(rows, arg: str) -> Path | None:
    """把用户给的层级（绝对路径或目录名）解析成一个实际目录。

    优先按绝对路径；否则从库里的 `source_file` 反推：
    取第一个「路径中出现该目录名」的位置。这样用户不必记得完整路径。
    """
    raw = (arg or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p if p.is_dir() else None

    roots: list[Path] = []
    for row in rows:
        src = Path(row["source_file"] or "")
        for parent in src.parents:
            if parent.name == raw:
                roots.append(parent)
    if not roots:
        return None
    # 取最短的那个：同名目录可能存在多份，最短路径通常是用户指的那一层
    roots.sort(key=lambda x: len(x.parts))
    return roots[0]


def _decide_ws(
    row, meta: dict, src: Path, level_dir: Path | None = None
) -> tuple[dict | None, str]:
    """WebDataset 来源的判定。

    与目录来源的根本区别：它的 `text` 是**描述性文字**
    （`"A female panting"`），不是台词，所以绝不能套用
    「有台词 → IndexTTS」。线路一律由子集配置决定：

        expresso  → indextts / speech      （它的 text 才是真台词）
        其余子集   → seedvc  / vocal_burst

    线上 66901 条的实际问题是**许可全是 unknown**：导入时没取到子集许可，
    于是 EmoGator（Apache-2.0）、VIVAE（CC-BY-4.0）这些可商用素材被
    一律标成不可商用。这里按子集表补回来。
    """
    info = _ws_subset_info(meta, src)
    if not info:
        return None, "认不出是哪个子集，保持原样"

    # 用户明确不要的条目（Expresso 的 neutral style 模板句）不该在库里
    if S._ws_should_skip(meta, info):
        return None, "属于用户明确不要的条目，应删除"

    use = info.get("target_use") or row["target_use"]
    kind = info.get("clip_kind") or row["clip_kind"]
    lic = info.get("license") or row["license"]
    commercial = T.commercial_ok_of(lic)

    category = meta.get("emotion") or meta.get("label") or meta.get("style")
    category = str(category) if category else row["category"]
    if level_dir is not None:
        category = level_dir.name

    # transcript 保留元信息里的 text（webdataset 的既有存法：
    # Expresso 是真台词，其余是描述文字），不清空。
    raw_text = meta.get("text")
    transcript = str(raw_text).strip() or None if raw_text is not None else None

    gender = row["gender"] or (str(meta["gender"]).lower() if meta.get("gender") else None)
    if not gender:
        speaker = (
            meta.get("speaker")
            or meta.get("speaker_id")
            or meta.get("contributor_id")
        )
        gender = S._infer_gender(str(speaker)) if speaker is not None else None

    intensity = row["intensity"]
    raw_int = meta.get("intensity")
    if raw_int is not None:
        mapped = S._WEB_INTENSITY.get(str(raw_int).strip().lower())
        if mapped:
            intensity = mapped

    # 语种跟着子集走：只有 Expresso 是真人英语语音，
    # 其余子集是副语言（喘气/咳嗽），本身不承载语种，留空。
    # 元信息自带 language 时优先用它。
    language = meta.get("language") or info.get("language") or row["language"]

    dur = float(row["duration_sec"] or 0.0)
    unchanged = (
        category == row["category"]
        and use == row["target_use"]
        and kind == row["clip_kind"]
        and lic == (row["license"] or "")
        and commercial == row["commercial_ok"]
        and (transcript or None) == (row["transcript"] or None)
        and (gender or None) == (row["gender"] or None)
        and intensity == row["intensity"]
        and (language or None) == (row["language"] or None)
    )
    if unchanged:
        return None, "无需改动"

    description = T.build_description(
        clip_kind=kind,
        category=category,
        # 只有 speech 才把文字写进描述；副语言的 text 是描述本身，
        # 拼进模板会变成「一段中等的A female panting」这种废话。
        transcript=transcript if kind == "speech" else None,
        gender=gender,
        language=language,
        intensity=intensity,
        duration_sec=dur,
        has_background=row["has_background"] or 0,
        studio_quality=row["studio_quality"] or 0,
    )

    why: list[str] = []
    if (use, kind) != (row["target_use"], row["clip_kind"]):
        why.append(f"线路 {row['target_use']}/{row['clip_kind']} → {use}/{kind}")
    if lic != (row["license"] or ""):
        why.append(f"许可 {row['license'] or '空'} → {lic}")
    if category != row["category"]:
        why.append(f"类别 → {category}")
    if not why:
        why.append("描述/元信息刷新")

    return {
        "target_use": use,
        "clip_kind": kind,
        "category": category,
        "transcript": transcript,
        "gender": gender,
        "intensity": intensity,
        "language": language,
        "license": lic,
        "commercial_ok": commercial,
        "description": description,
        "notes": _rebuild_notes(row["notes"], dur, use),
    }, "；".join(why)


def _decide(
    row, cache: _LookupCache, level_dir: Path | None = None
) -> tuple[dict | None, str]:
    """判断这一条该怎么改。返回 (patch, 原因)。patch 为 None 表示不动。

    `level_dir` 非空时，同时把类别名统一成该目录名（「选哪一层做分类名」）。
    """
    src = Path(row["source_file"] or "")
    if not src.is_file():
        return None, "源文件已不在"

    if level_dir is not None:
        try:
            src.relative_to(level_dir)
        except ValueError:
            return None, "不在选定层级之下"

    meta = cache.get(src)
    if meta is None:
        # 查不到台账 = 无法判定，不是「没有台词」。
        # 这里若继续往下走，会把「A female panting」这类描述当成
        # 「当前台词与台账不符」，进而把 transcript 清成 NULL。
        return None, "查不到台账，保持原样"

    # WebDataset 必须走子集配置，不能走「有台词 → IndexTTS」。
    # 它的 text 是描述性文字（"A female panting"），不是台词；
    # 按台词处理会把喘气、咳嗽这类副语言全判成语言内容。
    if _is_ws_item(meta):
        return _decide_ws(row, meta, src, level_dir)

    transcript = S._manifest_transcript(meta)
    has_text = bool(transcript)

    # 有台词就是语言内容；没台词保持原判（副语言/未知不动）
    if has_text:
        use, kind = "indextts", "speech"
    else:
        use, kind = row["target_use"], row["clip_kind"]

    category = level_dir.name if level_dir is not None else row["category"]

    unchanged = (
        category == row["category"]
        and use == row["target_use"]
        and kind == row["clip_kind"]
        and (transcript or None) == (row["transcript"] or None)
    )
    if unchanged:
        return None, "无需改动"

    dur = float(row["duration_sec"] or 0.0)
    gender = row["gender"] or S._infer_gender(meta.get("speaker") or row["speaker_id"])
    description = T.build_description(
        clip_kind=kind,
        category=category,
        transcript=transcript,
        gender=gender,
        language=row["language"],
        intensity=row["intensity"],
        duration_sec=dur,
        has_background=row["has_background"] or 0,
        studio_quality=row["studio_quality"] or 0,
    )

    if level_dir is not None and not has_text:
        reason = f"类别改为「{category}」"
    elif level_dir is not None:
        reason = f"类别改为「{category}」+ 有台词 → IndexTTS"
    else:
        reason = "有台词 → IndexTTS"

    return {
        "target_use": use,
        "clip_kind": kind,
        "category": category,
        "transcript": transcript,
        "gender": gender,
        "description": description,
        "notes": _rebuild_notes(row["notes"], dur, use),
    }, reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把误判成副语言的说话素材原地改回 IndexTTS；也可按目录层级重定类别"
    )
    ap.add_argument("--root", help="库根目录（默认 VOICE_LIBRARY_ROOT 或 voice_library/）")
    ap.add_argument("--apply", action="store_true", help="真正写入；默认只预览")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 条（0 = 全部）")
    ap.add_argument("--no-embed", action="store_true", help="不重建向量（快，但检索会失效）")
    ap.add_argument(
        "--level",
        help="按目录层级重定类别：给目录名或绝对路径，"
             "该层以下所有记录的类别统一改成该层目录名（上级目录会包含更多子目录）",
    )
    args = ap.parse_args(argv)

    cfg = Config.load(args.root)
    if not cfg.db_path.is_file():
        print(f"找不到数据库：{cfg.db_path}", file=sys.stderr)
        return 2
    db = Database(cfg.db_path)

    rows = db.iter_clips()
    if args.limit:
        rows = rows[: args.limit]
    print(f"库根：{cfg.root_path}")
    print(f"记录：{len(rows)} 条")

    level_dir = None
    if args.level:
        # 用全量 rows 反推目录，避免 --limit 把对应记录截掉后解析失败
        level_dir = _resolve_level_dir(db.iter_clips(), args.level)
        if level_dir is None:
            print(f"无法定位「{args.level}」这一层：既不是存在的绝对路径，"
                  f"也没有记录位于名为该名字的目录下。", file=sys.stderr)
            return 2
        print(f"层级：{level_dir}  （以下音频的类别统一改成「{level_dir.name}」）")
    print()

    cache = _LookupCache()
    todo: list[tuple[object, dict, str]] = []
    reasons: dict[str, int] = {}
    for row in rows:
        patch, why = _decide(row, cache, level_dir)
        if patch is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        todo.append((row, patch, why))

    print("判定统计：")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5}  {why}")
    print(f"  {len(todo):5}  需要改")
    print()

    if not todo:
        print("无需改动。")
        return 0

    print("将要修改的条目（最多显示 20 条）：")
    for row, patch, _ in todo[:20]:
        print(f"  #{row['id']} {row['category']:12} {row['duration_sec']:5.2f}s")
        # 必须打印真实的 patch，不能写死箭头。
        # 曾经这里硬编码成 `→ indextts/speech`，而 patch 实际上可能
        # 只改类别、线路不变，于是预览结果与将要写入的内容完全对不上，
        # 用户看到"66901 条都要改成 IndexTTS"这种不存在的结论。
        old_route = f"{row['target_use']}/{row['clip_kind']}"
        new_route = f"{patch['target_use']}/{patch['clip_kind']}"
        if old_route == new_route:
            print(f"     线路不变：{new_route}   （只改类别/描述）")
        else:
            print(f"     {old_route} → {new_route}")
        print(f"     类别：{row['category']} → {patch['category']}")
        print(f"     描述：{patch['description'][:58]}")
    if len(todo) > 20:
        print(f"  …还有 {len(todo) - 20} 条")
    print()

    if not args.apply:
        print("以上为预览。确认无误后加 --apply 真正写入。")
        return 0

    encoder = None
    if not args.no_embed:
        encoder = Encoder(cfg.embed_model)
        print(f"嵌入后端：{encoder.backend}（{encoder.dim} 维）")

    changed = 0
    with db.write() as conn:
        for row, patch, _ in todo:
            blob = None
            if encoder is not None:
                blob = encoder.to_blob(encoder.encode_one(patch["description"]))
            # 按 patch 实际包含的键生成 UPDATE。
            # 不能写死列名：webdataset 分支会额外产出 license / commercial_ok /
            # intensity，写死的话这些值算出来也落不了库，许可就永远修不好。
            # 列名来自本文件内的常量 patch，不用拼接外部输入。
            sets = [f"{k} = ?" for k in patch]
            params: list = [patch[k] for k in patch]
            if blob is not None:
                sets.append("embedding = ?")
                params.append(blob)
            params.append(row["id"])
            conn.execute(f"UPDATE clips SET {', '.join(sets)} WHERE id = ?", params)
            changed += 1

    print(f"\n完成：{changed} 条已更新。")
    if encoder is None:
        print("注意：跳过了向量重建，检索将找不到这些素材。"
              "需要时执行 `POST /api/reebed` 或重跑本脚本去掉 --no-embed。")
    print("\n提示：文件仍留在原目录（audio/vocal_burst/）。"
          "若之后从同一目录重新导入，会在 audio/csemotions/ 生成同名副本，"
          "届时用预览确认后再决定是否清理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
