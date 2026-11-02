"""导入管线测试：CSEMOTIONS 目录 → 落盘 → 描述 → 向量 → 入库。

用合成音频搭一个假 CSEMOTIONS 目录，验证 §4.1 / §4.3 / §4.5 的关键行为。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voicelib import audio as A
from voicelib.config import Config
from voicelib.db import Database
from voicelib.embedder import HashingEncoder
from voicelib.ingest import ImportOptions, import_into
from voicelib.sources import guess_kind, iter_csemotions

from conftest import synth_burst, synth_decaying_sigh, synth_speech


def _write(path: Path, y: np.ndarray, sr: int = 22050) -> None:
    A.write_wav(path, y, sr)


@pytest.fixture
def fake_csemotions(tmp_path: Path) -> Path:
    root = tmp_path / "CSEMOTIONS"
    for speaker, gender_seed in (("S03", 0), ("S07", 5)):
        for emo in ("Angry", "Happy", "Sad"):
            d = root / speaker / emo
            d.mkdir(parents=True)
            for i in range(2):
                _write(d / f"{i}.wav", synth_speech(6.0, seed=gender_seed + i))
    # 一条 2 秒的（应被时长下限筛掉）
    short = root / "S03" / "Angry" / "short.wav"
    _write(short, synth_speech(2.0, seed=9))
    # 一条纯静音（应丢弃）
    _write(root / "S03" / "Angry" / "silence.wav", np.zeros(22050 * 5, np.float32))
    return root


def _cfg(tmp_lib) -> Config:
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    return cfg


def test_guess_kind_detects_csemotions(fake_csemotions):
    assert guess_kind(fake_csemotions) == "csemotions"


def test_iter_csemotions_reads_speaker_and_emotion(fake_csemotions):
    items = list(iter_csemotions(fake_csemotions, {"S03": "male", "S07": "female"}))
    assert len(items) == 12 + 2
    cats = {i.category for i in items}
    assert cats == {"Angry", "Happy", "Sad"}
    s03 = [i for i in items if i.speaker_id == "S03"]
    assert s03 and all(i.gender == "male" for i in s03)


def test_import_csemotions_end_to_end(fake_csemotions, tmp_lib):
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions",
        path=str(fake_csemotions),
        target_use="indextts",
        license="apache-2.0",
        gender_map={"S03": "male", "S07": "female"},
    )
    progress_calls: list[tuple[int, int]] = []
    res = import_into(cfg, db, enc, opts, progress=lambda d, t: progress_calls.append((d, t)))

    assert res.imported == 12
    assert res.skipped_short == 1
    assert res.skipped_silent == 1
    assert db.count() == 12
    assert progress_calls and progress_calls[-1][0] == progress_calls[-1][1]

    rows = db.iter_clips()
    for row in rows:
        # 描述非空（§2.3 模板的职责之一）
        assert row["description"].strip()
        # 向量已建立
        assert row["embedding"]
        # 音频真的落盘了，且在 audio/csemotions 下
        p = cfg.root_path / row["filepath"]
        assert p.is_file(), row["filepath"]
        assert row["filepath"].startswith("audio/csemotions/")
        assert row["clip_kind"] == "speech"
        assert row["target_use"] == "indextts"
        # 许可推导
        assert row["commercial_ok"] == 1
        assert 3.0 <= row["duration_sec"] <= 15.0
        # 只记增益，不改音频电平
        assert row["gain_db"] is not None
        assert row["loudness_lufs"] is not None

    angry = [r for r in rows if r["category"] == "Angry"]
    assert angry and "愤怒" in angry[0]["description"]


def test_reimport_is_idempotent(fake_csemotions, tmp_lib):
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts)
    n1 = db.count()
    import_into(cfg, db, enc, opts)
    assert db.count() == n1, "重复导入不应产生重复行"


def test_reimport_preserves_human_annotations(fake_csemotions, tmp_lib):
    """重复导入不得覆盖人工精修的描述 / 标注（方案 §2.3、§7 阶段四）。"""
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts)
    clip_id = db.iter_clips()[0]["id"]

    db.update_clip(
        clip_id,
        {
            "description": "一段低沉压抑的男性愤怒，像在忍笑。",
            "verified": 1,
            "intensity": 5,
            "notes": "人工标注",
        },
    )

    import_into(cfg, db, enc, opts)  # 再导入一次
    row = db.get_clip(clip_id)
    assert row["description"] == "一段低沉压抑的男性愤怒，像在忍笑。"
    assert row["verified"] == 1
    assert row["intensity"] == 5
    assert row["notes"] == "人工标注"
    # 非人工字段仍然刷新
    assert row["duration_sec"] > 0


def test_nc_license_sets_commercial_ok_zero(fake_csemotions, tmp_lib):
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        license="cc-by-nc-sa-4.0",
    )
    import_into(cfg, db, enc, opts)
    rows = db.iter_clips()
    assert rows and all(r["commercial_ok"] == 0 for r in rows)


def test_dry_run_writes_nothing(fake_csemotions, tmp_lib):
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts", dry_run=True
    )
    res = import_into(cfg, db, enc, opts)
    assert res.imported > 0
    assert db.count() == 0
    assert not list((cfg.audio_dir).rglob("*.wav"))


@pytest.fixture
def fake_bursts(tmp_path: Path) -> Path:
    root = tmp_path / "bursts"
    root.mkdir()
    for cat, seed in (("laugh", 1), ("cry", 2), ("cough", 3)):
        d = root / cat
        d.mkdir()
        for i in range(2):
            _write(d / f"{i}.wav", synth_burst(1.6, seed=seed + i))
    _write(root / "laugh" / "silent.wav", np.zeros(22050, np.float32))
    return root


def test_import_seedvc_bursts(fake_bursts, tmp_lib):
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="emogator", path=str(fake_bursts), target_use="seedvc", license="apache-2.0"
    )
    res = import_into(cfg, db, enc, opts)
    assert res.imported == 6
    assert res.skipped_silent == 1
    rows = db.iter_clips()
    assert all(r["clip_kind"] == "vocal_burst" for r in rows)
    assert all(r["filepath"].startswith("audio/vocal_burst/") for r in rows)
    assert all("笑声" in r["description"] or "哭声" in r["description"]
               or "咳嗽" in r["description"] for r in rows)
    assert all(0.3 <= r["duration_sec"] <= 5.0 for r in rows)


def test_flat_manifest_import(tmp_path, tmp_lib):
    src = tmp_path / "custom"
    src.mkdir()
    _write(src / "a.wav", synth_burst(1.2, seed=4))
    _write(src / "b.wav", synth_burst(0.8, seed=5))
    (src / "manifest.json").write_text(
        '[{"filepath": "a.wav", "category": "laugh", "gender": "female", "intensity": 4},'
        ' {"filepath": "b.wav", "category": "sigh", "gender": "male", "intensity": 2}]',
        encoding="utf-8",
    )
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(kind="flat", path=str(src), target_use="seedvc", license="apache-2.0")
    res = import_into(cfg, db, enc, opts)
    assert res.imported == 2
    rows = {r["category"]: r for r in db.iter_clips()}
    assert rows["laugh"]["gender"] == "female"
    assert rows["laugh"]["intensity"] == 4
    assert "强烈" in rows["laugh"]["description"]


def test_long_burst_is_skipped(tmp_lib, tmp_path):
    src = tmp_path / "long"
    src.mkdir()
    sr = 22050
    t = np.arange(int(9.0 * sr)) / sr
    # 连续 9 秒的发声，裁剪后仍远超 seedvc 的 5 秒上限
    _write(src / "long.wav", (np.sin(2 * np.pi * 200 * t) * 0.2).astype(np.float32))
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    opts = ImportOptions(kind="flat", path=str(src), target_use="seedvc")
    res = import_into(cfg, db, HashingEncoder(), opts)
    assert res.skipped_long == 1
    assert res.imported == 0


def test_multi_burst_flagged_in_notes(tmp_lib, tmp_path):
    """多段发声不丢弃，只写 notes（§4.2）。"""
    src = tmp_path / "multi"
    src.mkdir()
    sr = 22050
    y = np.zeros(int(3.0 * sr), dtype=np.float32)
    rng = np.random.default_rng(0)
    for i, start in enumerate((0.1, 0.8, 1.5, 2.2)):
        a = int(start * sr)
        b = a + int(0.1 * sr)
        y[a:b] = (rng.normal(0, 0.2, b - a) * np.hanning(b - a)).astype(np.float32)
    _write(src / "four.wav", y)
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    opts = ImportOptions(kind="flat", path=str(src), target_use="seedvc")
    res = import_into(cfg, db, HashingEncoder(), opts)
    assert res.imported == 1
    row = db.iter_clips()[0]
    assert row["notes"] and "段发声" in row["notes"]


def test_trim_preserves_low_level_tail_through_ingest(tmp_lib, tmp_path):
    """入库后的时长应当贴近原素材，而不是被 trim(30) 削掉尾部。"""
    src = tmp_path / "sigh_src"
    src.mkdir()
    y = synth_decaying_sigh(5.0, 22050)
    _write(src / "sigh.wav", y)
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    opts = ImportOptions(kind="flat", path=str(src), target_use="indextts")
    res = import_into(cfg, db, HashingEncoder(), opts)
    assert res.imported == 1

    import librosa

    row = db.iter_clips()[0]
    # 同一素材若用 librosa.effects.trim(top_db=30)，只剩约 2.1 秒
    trim30 = len(librosa.effects.trim(y, top_db=30)[0]) / 22050
    assert trim30 < 2.5
    assert row["duration_sec"] > 2.8, (
        f"入库后只剩 {row['duration_sec']}s（trim30 为 {trim30:.2f}s），低电平尾部被误切"
    )
