"""台账（metadata.jsonl / metadata.csv）与「有台词定线路」的测试。

这批测试锁的是一个真实事故：用户把总目录填进来后，
他的素材是 `<speaker>/<emotion>/*.wav` 且**每条都有中文台词**，
但形状恰好符合 `<类别>/*.wav`，于是被当成副语言塞进 SeedVC，
8.5 秒的台词全部标成「超出 0.3–5s 范围」。

根因有两层，两层都要防：
  1. 台账文件没被识别（只认 manifest.json / manifest.jsonl / index.json，
     且只认 filepath/path/file 作为键）→ 台词一条都没读到
  2. 线路只看目录形状和时长，从不看有没有台词
"""

from __future__ import annotations

import json
from pathlib import Path

from voicelib import audio as A
from voicelib.config import Config
from voicelib.db import Database
from voicelib.embedder import HashingEncoder
from voicelib.ingest import ImportOptions, import_into
from voicelib.sources import (
    MANIFEST_NAMES,
    _load_manifest,
    _probe_has_transcript,
    guess_kind,
    iter_csemotions,
    iter_emogator,
    iter_flat,
    plan_import,
)

from conftest import synth_burst, synth_speech

EMOS = ("angry", "fearful", "happy", "neutral", "sad", "surprise", "playfulness")


def build_ledger(root: Path, rows: list[dict], *, fmt: str = "jsonl") -> Path:
    """在 root 下写一份台账，返回文件路径。"""
    root.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        p = root / "metadata.jsonl"
        p.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    else:
        p = root / "metadata.csv"
        import csv

        cols = list(rows[0].keys())
        with p.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return p


def make_speech_tree(root: Path) -> Path:
    """搭一个和用户数据同构的树：wav/<speaker>/<emotion>/*.wav + 根级台账。"""
    wav = root / "wav"
    rows = []
    for spk in ("female001", "male001"):
        for emo in ("angry", "sad"):
            d = wav / spk / emo
            d.mkdir(parents=True, exist_ok=True)
            name = f"{spk}_{emo}_01.wav"
            A.write_wav(d / name, synth_speech(8.49, seed=1), 22050)
            rows.append({
                "wav_path": f"wav/{spk}/{emo}/{name}",
                "speaker": spk,
                "emotion": emo,
                "text": f"{spk} 用{emo}的语气说了一句中文台词",
                "duration_sec": "8.49",
            })
    build_ledger(root, rows, fmt="jsonl")
    build_ledger(root, rows, fmt="csv")
    return wav


# ---------------------------------------------------------------- 台账识别


def test_metadata_names_are_recognised():
    for name in ("metadata.jsonl", "metadata.json", "metadata.csv"):
        assert name in MANIFEST_NAMES


def test_load_manifest_reads_wav_path_key(tmp_path):
    """用户的键名是 wav_path，不是 filepath。"""
    build_ledger(tmp_path, [
        {"wav_path": "wav/a/x.wav", "speaker": "a", "text": "台词一"},
        {"wav_path": "wav/b/y.wav", "speaker": "b", "text": "台词二"},
    ])
    m = _load_manifest(tmp_path)
    assert "wav/a/x.wav" in m
    assert m["wav/a/x.wav"]["text"] == "台词一"
    # 文件名兜底键也要有，方便填的是上层目录
    assert "x.wav" in m


def test_load_manifest_reads_csv_with_bom(tmp_path):
    """导出工具产出的 csv 带 BOM，不去掉第一列名会变成 \\ufeffwav_path。"""
    build_ledger(tmp_path, [
        {"wav_path": "wav/a/x.wav", "speaker": "a", "text": "台词"},
    ], fmt="csv")
    m = _load_manifest(tmp_path)
    assert "wav/a/x.wav" in m
    assert m["wav/a/x.wav"]["text"] == "台词"


# ---------------------------------------------------------------- 台词探测


def test_probe_detects_transcript(tmp_path):
    wav = make_speech_tree(tmp_path)
    assert _probe_has_transcript(tmp_path) is True
    # 填中间层也要能向上找到台账
    assert _probe_has_transcript(wav) is True
    assert _probe_has_transcript(wav / "female001") is True


def test_probe_false_when_text_empty(tmp_path):
    """真副语言：台账里 text 为空，不能被当成有台词。"""
    d = tmp_path / "laugh"
    d.mkdir()
    A.write_wav(d / "a.wav", synth_burst(1.5, seed=2), 22050)
    build_ledger(tmp_path, [
        {"wav_path": "laugh/a.wav", "text": "", "category": "laugh"},
    ])
    assert _probe_has_transcript(tmp_path) is False


def test_probe_false_without_ledger(tmp_path):
    d = tmp_path / "laugh"
    d.mkdir()
    A.write_wav(d / "a.wav", synth_burst(1.5, seed=2), 22050)
    assert _probe_has_transcript(tmp_path) is False


# ---------------------------------------------------------------- 线路判定


def test_speaker_layer_with_transcript_goes_to_indextts(tmp_path):
    """核心回归：<说话人>/<情绪>/ 形状 + 有台词 → 必须 IndexTTS。

    形状上它是 `<类别>/`（emogator），只看形状会判成 SeedVC。
    """
    wav = make_speech_tree(tmp_path)
    speaker_dir = wav / "female001"

    assert guess_kind(speaker_dir) == "emogator", "前提：形状仍会被认成 emogator"

    plans = plan_import(speaker_dir)
    assert len(plans) == 1
    assert plans[0].target_use == "indextts"
    assert plans[0].clip_kind == "speech"


def test_whole_root_with_transcript_goes_to_indextts(tmp_path):
    """用户填的是总目录，也要判对。"""
    make_speech_tree(tmp_path)
    plans = plan_import(tmp_path)
    assert plans, "应该至少识别出一个单元"
    assert all(p.target_use == "indextts" for p in plans)


def test_burst_without_transcript_stays_seedvc(tmp_path):
    """防误伤：真副语言（无台词）必须仍然走 SeedVC。"""
    d = tmp_path / "laugh"
    d.mkdir()
    A.write_wav(d / "a.wav", synth_burst(1.5, seed=2), 22050)
    build_ledger(tmp_path, [
        {"wav_path": "laugh/a.wav", "text": "", "category": "laugh"},
    ])
    plans = plan_import(tmp_path)
    assert len(plans) == 1
    assert plans[0].target_use == "seedvc"
    assert plans[0].clip_kind == "vocal_burst"


# ---------------------------------------------------------------- 台词回填


def test_csemotions_reads_transcript(tmp_path):
    wav = make_speech_tree(tmp_path)
    items = list(iter_csemotions(wav))
    assert len(items) == 4
    assert all(i.transcript for i in items)
    assert all("中文台词" in i.transcript for i in items)


def test_emogator_reads_transcript_from_upper_ledger(tmp_path):
    """台账在更上一层时也要能匹配（用户可能只填说话人层）。"""
    wav = make_speech_tree(tmp_path)
    items = list(iter_emogator(wav / "female001"))
    assert items
    assert all(i.transcript for i in items)
    assert items[0].speaker_id == "female001"


def test_flat_reads_transcript(tmp_path):
    make_speech_tree(tmp_path)
    items = list(iter_flat(tmp_path))
    assert items
    assert all(i.transcript for i in items)


def test_flat_infers_gender_from_speaker(tmp_path):
    """性别要能从说话人名推断，否则描述会缺「女性/男性」这一维。"""
    make_speech_tree(tmp_path)
    items = list(iter_flat(tmp_path))
    got = {i.speaker_id: i.gender for i in items}
    assert got["female001"] == "female"
    assert got["male001"] == "male"


# ---------------------------------------------------------------- 端到端


def test_import_speech_lands_on_indextts(tmp_path, tmp_lib):
    """事故现场复现：8.5s 的中文台词不能再进 SeedVC。"""
    make_speech_tree(tmp_path)
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)

    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(tmp_path), license="apache-2.0"),
    )
    assert res.errors == []
    assert res.imported == 4

    rows = db.iter_clips()
    assert all(r["target_use"] == "indextts" for r in rows)
    assert all(r["clip_kind"] == "speech" for r in rows)
    # 台词进库，描述里也有真实内容而不是「（无台词）」
    assert all(r["transcript"] for r in rows)
    assert all("无台词" not in (r["description"] or "") for r in rows)
    # 8.49s 落在 3–15s 内，不该被标超长
    assert all("超出" not in (r["notes"] or "") for r in rows)


def test_import_burst_still_seedvc(tmp_path, tmp_lib):
    d = tmp_path / "laugh"
    d.mkdir()
    A.write_wav(d / "a.wav", synth_burst(1.5, seed=2), 22050)
    build_ledger(tmp_path, [
        {"wav_path": "laugh/a.wav", "text": "", "category": "laugh"},
    ])
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(tmp_path), license="apache-2.0"),
    )
    assert res.imported == 1
    row = db.iter_clips()[0]
    assert row["target_use"] == "seedvc"
    assert row["transcript"] is None
