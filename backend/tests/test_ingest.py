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


def test_guess_kind_flat_root_audio_is_not_emogator(tmp_path):
    """平铺在根目录的音频应识别为 flat，而不是 emogator。

    EmoGator 的形态是 `<label>/*.mp3`；根级平铺没有 label 子目录，
    类别无从推断，强行当 emogator 会让类别变成文件名。
    """
    root = tmp_path / "flat"
    root.mkdir()
    for i in range(2):
        _write(root / f"{i}.wav", synth_burst(1.2, seed=i))
    assert guess_kind(root) == "flat"


def test_guess_kind_label_subdirs_are_emogator(tmp_path):
    root = tmp_path / "bursts"
    for label in ("laugh", "cry"):
        d = root / label
        d.mkdir(parents=True)
        _write(d / "0.wav", synth_burst(1.2))
    assert guess_kind(root) == "emogator"


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

    assert res.imported == 13            # 12 条正常 + 1 条 2s（不再丢弃）
    assert res.imported_short == 1       # 那条 2s 的：入库但标了 notes
    assert res.skipped_short == 0        # 过短不再算「跳过」
    assert res.skipped_silent == 1
    assert db.count() == 13
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
        # 只记增益，不改音频电平
        assert row["gain_db"] is not None
        assert row["loudness_lufs"] is not None
        # 时长范围：过短的也入库（标 notes），所以这里只校验上限；
        # 下限由那些没有被标「低于」的条目各自保证。
        assert row["duration_sec"] <= 15.0
        if row["duration_sec"] < 3.0:
            assert row["notes"] and "低于" in row["notes"], row["filepath"]

    # 那条 2 秒的必须保留下来，并带上下限提示
    short_rows = [r for r in rows if r["duration_sec"] < 3.0]
    assert len(short_rows) == 1
    assert "低于 3s 下限" in (short_rows[0]["notes"] or "")

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


def test_long_burst_is_imported_with_note(tmp_lib, tmp_path):
    """超长素材入库而不是丢弃，并在 notes 里标明需人工截取。

    旧行为是直接丢弃（skipped_long += 1）。实测 CSEMOTIONS 有 8.5% 的素材
    超过 IndexTTS 的 15 秒上限，静默丢掉等于白白损失可用素材。
    """
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

    assert res.imported == 1
    assert res.imported_over_long == 1
    assert res.skipped_long == 0

    row = db.iter_clips()[0]
    assert row["duration_sec"] > 5.0
    assert row["notes"], "超长素材必须写 notes"
    assert "超出" in row["notes"]
    assert "截取" in row["notes"]


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


# ---------------------------------------------------------------- 混合目录自动拆分


@pytest.fixture
def mixed_root(tmp_path: Path) -> Path:
    """一个根目录里同时有三种排布。"""
    root = tmp_path / "mixed"
    for spk in ("S01", "S02"):
        for emo in ("Angry", "Sad"):
            d = root / "CSEMOTIONS" / spk / emo
            d.mkdir(parents=True)
            for i in range(2):
                _write(d / f"{i}.wav", synth_speech(6.0, seed=i))
    for lab in ("laugh", "cry"):
        d = root / "EmoGator" / lab
        d.mkdir(parents=True)
        for i in range(2):
            _write(d / f"{i}.wav", synth_burst(1.5, seed=i))
    d = root / "custom"
    d.mkdir()
    for i in range(2):
        _write(d / f"{i}.wav", synth_burst(1.2, seed=i))
    return root


def test_plan_import_splits_mixed_root(mixed_root):
    """混合根目录必须拆成多个单元，各按正确排布解读。

    回归点：旧实现把整个根判成 emogator，于是 CSEMOTIONS 那部分
    的类别会变成说话人名（S01/S02），语义全错。
    """
    from voicelib.sources import plan_import

    plans = {p.label: p for p in plan_import(mixed_root)}
    assert set(plans) == {"CSEMOTIONS", "EmoGator", "custom"}

    assert plans["CSEMOTIONS"].kind == "csemotions"
    assert plans["CSEMOTIONS"].clip_kind == "speech"
    assert plans["CSEMOTIONS"].target_use == "indextts"
    assert plans["CSEMOTIONS"].count == 8

    assert plans["EmoGator"].kind == "emogator"
    assert plans["EmoGator"].clip_kind == "vocal_burst"
    assert plans["EmoGator"].target_use == "seedvc"
    assert plans["EmoGator"].count == 4

    assert plans["custom"].kind == "flat"


def test_import_auto_handles_mixed_root(mixed_root, tmp_lib):
    """auto 模式一次导完混合目录，两种线路各自正确。"""
    from voicelib.ingest import import_into

    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(mixed_root), license="apache-2.0"),
    )
    assert res.imported == 14
    assert res.units, "应当有单元明细"

    rows = db.iter_clips()
    speech = [r for r in rows if r["clip_kind"] == "speech"]
    burst = [r for r in rows if r["clip_kind"] == "vocal_burst"]
    # CSEMOTIONS 8 条 → speech；EmoGator 4 条 + custom 2 条（1.2 秒，推断为 SeedVC）→ vocal_burst
    assert len(speech) == 8
    assert len(burst) == 6

    # 关键：说话片段的类别必须是情绪，不能是说话人名
    assert {r["category"] for r in speech} == {"Angry", "Sad"}
    assert "S01" not in {r["category"] for r in rows}
    assert "S02" not in {r["category"] for r in rows}

    # 各自落到对应目录、对应线路
    for r in speech:
        assert r["target_use"] == "indextts"
        assert r["filepath"].startswith("audio/csemotions/")
    for r in burst:
        assert r["target_use"] == "seedvc"
        assert r["filepath"].startswith("audio/vocal_burst/")


def test_import_auto_on_single_type_dir(mixed_root, tmp_lib):
    """单类型目录在 auto 下仍只有一个单元，行为与显式指定一致。"""
    from voicelib.ingest import import_into

    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(mixed_root / "CSEMOTIONS")),
    )
    assert len(res.units) == 1
    assert res.imported == 8
    assert all(r["clip_kind"] == "speech" for r in db.iter_clips())


def test_import_auto_empty_dir_raises(tmp_path, tmp_lib):
    from voicelib.ingest import import_into

    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    with pytest.raises(ValueError):
        import_into(cfg, Database(cfg.db_path), HashingEncoder(),
                    ImportOptions(kind="auto", path=str(empty)))


def test_reimport_overwrites_program_values_but_keeps_user_annotations(fake_csemotions, tmp_lib):
    """用户要求：重复导入覆盖**程序标注**，但不动**用户标注**。

    旧实现靠 HUMAN_FIELDS 白名单保住 clips 里的四个字段，
    结果那些字段的程序值永远刷不新。现在人工值在 annotations 表：
      * clips 的每一列都应被新导入刷新
      * annotations 一行都不该动
    """
    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts)
    clip_id = db.iter_clips()[0]["id"]
    program_before = db.get_clip(clip_id)["description"]

    db.update_clip(clip_id, {"description": "我写的描述", "verified": 1})

    # 再导一次，但用不同的 license，让 clips 那一列必然变化
    opts2 = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        license="cc-by-4.0",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts2)

    with db.read() as conn:
        raw = conn.execute(
            "SELECT description, verified, license FROM clips WHERE id = ?", (clip_id,)
        ).fetchone()
    # 程序标注被覆盖：clips 里回到模板生成的描述，license 也换新了
    assert raw["description"] == program_before
    assert raw["description"] != "我写的描述"
    assert raw["license"] == "cc-by-4.0"
    assert raw["verified"] == 0        # 程序值，不是人工的 1

    # 用户标注完好：读取层仍给人看人工值
    row = db.get_clip(clip_id)
    assert row["description"] == "我写的描述"
    assert row["verified"] == 1
    assert db.annotations_of(clip_id) == {"description": "我写的描述", "verified": "1"}


# ---------------------------------------------------------------- 音频落盘策略
#
# 用户要求：设置项支持「覆盖导入音频」或「继续用库里已有的音频文件」。
# 注意与 overwrite 参数区分：那个管的是「遇到同 filepath 的记录要不要跳过」，
# 这个管的是「磁盘上的 wav 要不要重写」。


def _first_audio(cfg) -> Path:
    files = sorted((cfg.root_path / "audio").rglob("*.wav"))
    assert files, "导入后应当有音频文件"
    return files[0]


def test_audio_mode_overwrite_rewrites_file(fake_csemotions, tmp_lib):
    """默认 overwrite：同名文件被重写（mtime 变化）。"""
    import os, time

    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts)
    target = _first_audio(cfg)
    stamp = os.stat(target).st_mtime_ns

    time.sleep(0.01)
    import_into(cfg, db, enc, opts)
    assert os.stat(target).st_mtime_ns != stamp, "overwrite 模式应重写文件"


def test_audio_mode_keep_leaves_existing_file(fake_csemotions, tmp_lib):
    """keep：文件已在就不写盘，元数据照常刷新。"""
    import os, time

    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts)
    target = _first_audio(cfg)
    stamp = os.stat(target).st_mtime_ns
    before = db.count()

    time.sleep(0.01)
    opts2 = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        audio_mode="keep",
        gender_map={"S03": "male", "S07": "female"},
    )
    result = import_into(cfg, db, enc, opts2)

    assert os.stat(target).st_mtime_ns == stamp, "keep 模式不应重写文件"
    assert result.audio_kept > 0, "结果里应能看到「保留了多少条音频」"
    assert db.count() == before, "记录数不变"


def test_audio_mode_keep_still_writes_missing_file(fake_csemotions, tmp_lib):
    """keep 但文件不存在时必须补写：否则记录指向空路径，试听会 404。"""
    import os

    cfg = _cfg(tmp_lib)
    db = Database(cfg.db_path)
    enc = HashingEncoder()
    opts = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts)
    target = _first_audio(cfg)
    os.remove(target)
    assert not target.exists()

    opts2 = ImportOptions(
        kind="csemotions", path=str(fake_csemotions), target_use="indextts",
        audio_mode="keep",
        gender_map={"S03": "male", "S07": "female"},
    )
    import_into(cfg, db, enc, opts2)
    assert target.is_file(), "文件缺失时 keep 模式必须补写"
