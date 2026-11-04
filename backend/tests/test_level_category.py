"""「选哪一层做分类名」的测试。

需求（用户原话）：
    a/b/c/1.wav  a/b/c/2.wav  a/b/d/2.wav  a/b/d/4.wav  a/5.wav
    为 c 目录下的音频选择分类名，就列出 c、b、a 三个选项
    1. 选 c，就只归类 c 下的音频
    2. 选 b，就归类 b 下所有音频（含 b、c）
    3. 同名音频加编号（现阶段的命名已用内容 hash，不会冲突）

这个功能的本质是「扫描范围」+「类别名来源」两件事同时被用户接管：
默认按目录形状推断，选定层级后一律以该层目录名为类别。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicelib import audio as A
from voicelib.config import Config
from voicelib.db import Database
from voicelib.embedder import HashingEncoder
from voicelib.ingest import ImportOptions, import_into
from voicelib.sources import (
    count_audio_capped,
    count_items,
    iter_by_level,
    iter_flat,
    level_choices,
    plan_import,
    resolve_level,
)

from conftest import synth_speech


def build_tree(root: Path) -> Path:
    """按用户给的例子搭树：a/b/c/{1,2}.wav、a/b/d/{2,4}.wav、a/5.wav"""
    a = root / "a"
    for rel, spk, emo in [
        ("b/c/1.wav", "female001", "angry"),
        ("b/c/2.wav", "female001", "happy"),
        ("b/d/2.wav", "male001", "sad"),
        ("b/d/4.wav", "male001", "fearful"),
        ("5.wav", "female001", "neutral"),
    ]:
        p = a / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        A.write_wav(p, synth_speech(8.0, seed=1), 22050)

    # 台账放根：让内容判定能拿到台词，走 IndexTTS
    rows = []
    for rel, spk, emo in [
        ("b/c/1.wav", "female001", "angry"),
        ("b/c/2.wav", "female001", "happy"),
        ("b/d/2.wav", "male001", "sad"),
        ("b/d/4.wav", "male001", "fearful"),
        ("5.wav", "female001", "neutral"),
    ]:
        rows.append({
            "wav_path": rel, "speaker": spk, "emotion": emo,
            "text": f"{spk} 说了一句中文台词（{rel}）",
        })
    (a / "metadata.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return a


# ---------------------------------------------------------------- 层级解析


def test_resolve_level_by_name(tmp_path):
    a = build_tree(tmp_path)
    start = a / "b" / "c"
    assert resolve_level(start, "c") == (a / "b" / "c").resolve()
    assert resolve_level(start, "b") == (a / "b").resolve()
    assert resolve_level(start, "a") == a.resolve()


def test_resolve_level_by_absolute_path(tmp_path):
    a = build_tree(tmp_path)
    assert resolve_level(a / "b" / "c", str(a / "b")) == (a / "b").resolve()


def test_resolve_level_rejects_unrelated(tmp_path):
    """陌生名字不能解析成目录 —— 否则会把扫描范围扩到无关位置。"""
    a = build_tree(tmp_path)
    assert resolve_level(a / "b" / "c", "zzz") is None
    assert resolve_level(a / "b" / "c", str(tmp_path / "elsewhere")) is None


def test_resolve_level_empty(tmp_path):
    a = build_tree(tmp_path)
    assert resolve_level(a / "b" / "c", "") is None
    assert resolve_level(a / "b" / "c", None) is None


# ---------------------------------------------------------------- 层级列表


def test_level_choices_from_near_to_far(tmp_path):
    a = build_tree(tmp_path)
    names = [c["name"] for c in level_choices(a / "b" / "c", max_up=2)]
    assert names[:3] == ["c", "b", "a"]
    assert [c["is_root"] for c in level_choices(a / "b" / "c", max_up=2)][:3] == [
        True, False, False,
    ]


def test_level_choices_counts(tmp_path):
    a = build_tree(tmp_path)
    got = {c["name"]: c["count"] for c in level_choices(a / "b" / "c", max_up=3)}
    assert got["c"] == 2
    assert got["b"] == 4
    assert got["a"] == 5


def test_level_choices_is_bounded(tmp_path):
    """祖先可能是整个磁盘，计数必须有上限且及时收手。"""
    import time

    a = build_tree(tmp_path)
    t0 = time.monotonic()
    level_choices(a / "b" / "c", max_up=8)
    elapsed = time.monotonic() - t0
    # 8 层祖先 × 各 0.3/0.05s 预算，正常应远小于 3 秒
    assert elapsed < 3.0


def test_count_audio_capped_stops_early(tmp_path):
    d = tmp_path / "many"
    d.mkdir()
    for i in range(10):
        A.write_wav(d / f"{i}.wav", synth_speech(0.5, seed=i), 22050)
    n, truncated = count_audio_capped(d, cap=3)
    assert n == 3
    assert truncated is True


# ---------------------------------------------------------------- 扫描范围


def test_select_c_only_c(tmp_path):
    """选 c：只归类 c 下的 2 条。"""
    a = build_tree(tmp_path)
    items = list(iter_by_level(a / "b" / "c"))
    assert len(items) == 2
    assert {i.category for i in items} == {"c"}


def test_select_b_covers_subdirs(tmp_path):
    """选 b：归类 b 下所有音频（含 c 与 d），类别统一为 b。"""
    a = build_tree(tmp_path)
    items = list(iter_by_level(a / "b"))
    assert len(items) == 4
    assert {i.category for i in items} == {"b"}
    # 确实覆盖了两个子目录
    rels = {i.path.relative_to(a / "b").as_posix() for i in items}
    assert rels == {"c/1.wav", "c/2.wav", "d/2.wav", "d/4.wav"}


def test_select_a_covers_everything(tmp_path):
    a = build_tree(tmp_path)
    items = list(iter_by_level(a))
    assert len(items) == 5
    assert {i.category for i in items} == {"a"}


def test_default_flat_still_uses_parent_dir(tmp_path):
    """对照：不选层级时，类别按台账/父目录取，行为不变。

    注意 `iter_flat` 的优先级是 台账 category → 台账 emotion → 直接父目录。
    本树的台账带 emotion，所以这里拿到的是情绪名；要验证「父目录兜底」
    得用没有台账的树（见下一个用例）。
    """
    a = build_tree(tmp_path)
    cats = {i.path.name: i.category for i in iter_flat(a / "b")}
    assert cats["1.wav"] == "angry"      # 台账的 emotion
    assert cats["4.wav"] == "fearful"
    # 关键区别：不选层级时类别**不会**被统一成 "b"
    assert "b" not in cats.values()


def test_flat_without_ledger_falls_back_to_parent_dir(tmp_path):
    """无台账时，类别取直接父目录 —— 这正是不选层级时的默认行为。"""
    a = tmp_path / "a"
    for rel in ["b/c/1.wav", "b/d/4.wav"]:
        p = a / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        A.write_wav(p, synth_speech(8.0, seed=1), 22050)

    cats = {i.path.name: i.category for i in iter_flat(a / "b")}
    assert cats["1.wav"] == "c"
    assert cats["4.wav"] == "d"

    # 选 b 之后才统一成 "b"
    leveled = {i.path.name: i.category for i in iter_by_level(a / "b")}
    assert leveled == {"1.wav": "b", "4.wav": "b"}


def test_by_level_keeps_transcript(tmp_path):
    """整层统一类别时，台账里的台词仍要读出来（否则会退化成副语言描述）。"""
    a = build_tree(tmp_path)
    items = list(iter_by_level(a / "b"))
    assert all(i.transcript for i in items)
    assert all("中文台词" in i.transcript for i in items)


# ---------------------------------------------------------------- 计划与计数


def test_plan_import_with_level(tmp_path):
    a = build_tree(tmp_path)
    plans = plan_import(a / "b" / "c", category_level="b")
    assert len(plans) == 1
    p = plans[0]
    assert p.kind == "by_level"
    assert p.label == "b"
    assert p.count == 4
    assert p.target_use == "indextts"   # 有台词
    assert "b" in p.reason


def test_plan_import_without_level_unchanged(tmp_path):
    a = build_tree(tmp_path)
    plans = plan_import(a / "b" / "c")
    assert len(plans) == 1
    assert plans[0].kind != "by_level"
    assert plans[0].count == 2


def test_count_items_by_level(tmp_path):
    a = build_tree(tmp_path)
    assert count_items("by_level", a / "b") == 4
    assert count_items("by_level", a) == 5


def test_plan_import_bad_level_returns_empty(tmp_path):
    a = build_tree(tmp_path)
    assert plan_import(a / "b" / "c", category_level="zzz") == []


# ---------------------------------------------------------------- 端到端


def _import(tmp_path, path: Path, level: str | None):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    cfg = Config(root=str(lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(
            kind="auto", path=str(path), license="apache-2.0",
            category_level=level or "",
        ),
    )
    return res, db


def test_import_level_c(tmp_path):
    a = build_tree(tmp_path)
    res, db = _import(tmp_path, a / "b" / "c", "c")
    assert res.imported == 2
    assert {r["category"] for r in db.iter_clips()} == {"c"}


def test_import_level_b(tmp_path):
    a = build_tree(tmp_path)
    res, db = _import(tmp_path, a / "b" / "c", "b")
    assert res.imported == 4
    rows = db.iter_clips()
    assert {r["category"] for r in rows} == {"b"}
    # 有台词 → 走 IndexTTS，不被当成副语言
    assert all(r["target_use"] == "indextts" for r in rows)
    assert all(r["clip_kind"] == "speech" for r in rows)


def test_import_level_a(tmp_path):
    a = build_tree(tmp_path)
    res, db = _import(tmp_path, a / "b" / "c", "a")
    assert res.imported == 5
    assert {r["category"] for r in db.iter_clips()} == {"a"}


def test_import_level_invalid_raises(tmp_path):
    a = build_tree(tmp_path)
    with pytest.raises(ValueError, match="zzz"):
        _import(tmp_path, a / "b" / "c", "zzz")


def test_import_same_name_files_do_not_collide(tmp_path):
    """用户提到「同名音频加编号」。实测在同层统一类别时不会撞名：

    c/2.wav 与 d/2.wav 同名，但落盘名是 <说话人>_<类别>_<内容hash>.wav，
    内容不同 → hash 不同 → 不会覆盖。
    """
    a = build_tree(tmp_path)
    res, db = _import(tmp_path, a / "b", "b")
    assert res.imported == 4
    paths = [r["filepath"] for r in db.iter_clips()]
    assert len(paths) == len(set(paths)), "不应出现同名覆盖"
