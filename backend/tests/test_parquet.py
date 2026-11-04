"""parquet 数据源测试（HF datasets 的 Audio 列格式）。

覆盖几个容易踩的点：
  1. 音频内嵌在容器里，没有独立文件 → 不能当普通目录扫描
  2. 文件名不可信（female003_jolliest_01.wav 的 emotion 列是 playfulness）
  3. 只统计不解码（with_audio=False），否则预览要等几十秒
  4. 超长素材入库并写 notes，而不是丢弃
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from voicelib import audio as A
from voicelib.config import Config
from voicelib.db import Database
from voicelib.embedder import HashingEncoder
from voicelib.ingest import ImportOptions, import_into
from voicelib.sources import (
    count_items,
    guess_kind,
    iter_by_level,
    iter_parquet,
    parquet_stats,
    plan_import,
)

from conftest import synth_burst, synth_speech

pa = pytest.importorskip("pyarrow", reason="parquet 支持需要 pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _wav_bytes(y: np.ndarray, sr: int = 22050) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def write_parquet(
    path: Path,
    rows: list[dict],
    *,
    audio_key: str = "audio",
) -> Path:
    """写一个 HF Audio 形态的 parquet。

    rows 每条：{wav: bytes|np.ndarray, name: str, text, emotion, speaker}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    audios = []
    for r in rows:
        data = r["wav"]
        if not isinstance(data, (bytes, bytearray)):
            data = _wav_bytes(np.asarray(data), r.get("sr", 22050))
        audios.append({"bytes": bytes(data), "path": r["name"]})

    table = pa.table(
        {
            audio_key: pa.array(audios, type=AUDIO_TYPE),
            "text": pa.array([r.get("text", "") for r in rows], type=pa.string()),
            "emotion": pa.array([r["emotion"] for r in rows], type=pa.string()),
            "speaker": pa.array([r["speaker"] for r in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path)
    return path


@pytest.fixture
def fake_parquet(tmp_path: Path) -> Path:
    """一个 data/ 目录，里面有一个 parquet 分片。"""
    rows = [
        {
            "wav": synth_speech(6.0, seed=1),
            # 文件名里的词不是情绪词，真正类别在 emotion 列
            "name": "female001_jolliest_01.wav",
            "text": "听到那些虚假的承诺，我的愤怒瞬间爆发。",
            "emotion": "angry",
            "speaker": "female001",
        },
        {
            "wav": synth_speech(6.5, seed=2),
            "name": "female001_sad_02.wav",
            "text": "每当我努力后却得到无情的打击。",
            "emotion": "sad",
            "speaker": "female001",
        },
        {
            "wav": synth_speech(7.0, seed=3),
            "name": "male002_happy_03.wav",
            "text": "今天真是个好日子。",
            "emotion": "happy",
            "speaker": "male002",
        },
    ]
    d = tmp_path / "data"
    write_parquet(d / "train-00000-of-00002.parquet", rows)
    return d


# ---------------------------------------------------------------- 识别


def test_guess_kind_detects_parquet(fake_parquet):
    assert guess_kind(fake_parquet) == "parquet"


def test_parquet_dir_is_not_mistaken_for_flat_or_emogator(fake_parquet):
    kind = guess_kind(fake_parquet)
    assert kind not in ("flat", "emogator", "csemotions")


def test_plan_import_recognises_parquet_unit(fake_parquet):
    plans = plan_import(fake_parquet)
    assert len(plans) == 1
    p = plans[0]
    assert p.kind == "parquet"
    assert p.clip_kind == "speech"
    assert p.target_use == "indextts"
    assert p.count == 3


def test_parquet_nested_in_container_is_split_out(tmp_path):
    """root/dataset/*.parquet：根是容器，应拆出 dataset 单元。"""
    rows = [{
        "wav": synth_speech(6.0, seed=1), "name": "a.wav",
        "text": "你好", "emotion": "happy", "speaker": "female001",
    }]
    root = tmp_path / "root"
    write_parquet(root / "dataset" / "train-00000.parquet", rows)

    plans = plan_import(root)
    assert len(plans) == 1
    assert plans[0].label == "dataset"
    assert plans[0].kind == "parquet"


# ---------------------------------------------------------------- 读取


def test_iter_parquet_reads_columns_not_filenames(fake_parquet):
    """类别取 emotion 列；文件名里的 jolliest 不能当类别。"""
    items = list(iter_parquet(fake_parquet))
    assert len(items) == 3
    cats = {i.category for i in items}
    # angry → Angry（与 CSEMOTIONS 标签一致的大小写）
    assert cats == {"Angry", "Sad", "Happy"}
    assert "Jolliest" not in cats


def test_iter_parquet_infers_gender_from_speaker(fake_parquet):
    by_spk = {i.speaker_id: i.gender for i in iter_parquet(fake_parquet)}
    assert by_spk["female001"] == "female"
    assert by_spk["male002"] == "male"


def test_iter_parquet_keeps_transcript(fake_parquet):
    items = list(iter_parquet(fake_parquet))
    assert all(i.transcript for i in items), "台词必须保留，否则描述会退化成（无台词）"
    assert any("愤怒" in i.transcript for i in items)


def test_iter_parquet_audio_bytes_are_decodable(fake_parquet):
    item = next(iter(iter_parquet(fake_parquet)))
    assert item.audio_bytes is not None
    assert item.audio_bytes[:4] == b"RIFF"
    y, sr = A.load_audio_bytes(item.audio_bytes)
    assert sr == 22050
    assert len(y) / sr > 5.0


def test_with_audio_false_skips_decoding(fake_parquet):
    """统计路径不能把音频读出来。"""
    items = list(iter_parquet(fake_parquet, with_audio=False))
    assert len(items) == 3
    assert all(i.audio_bytes is None for i in items)
    # 元信息仍然齐全
    assert all(i.category and i.speaker_id for i in items)


def test_parquet_stats_counts_without_audio(fake_parquet):
    st = parquet_stats(fake_parquet)
    assert st["files"] == 1
    assert st["total"] == 3
    assert st["categories"] == {"Angry": 1, "Sad": 1, "Happy": 1}
    assert st["speakers"] == {"female001": 2, "male002": 1}


def test_count_items_uses_metadata_only(fake_parquet):
    assert count_items("parquet", fake_parquet) == 3


def test_bare_binary_audio_column_is_supported(tmp_path):
    """有些数据集直接存裸 bytes，而不是 struct<bytes,path>。"""
    d = tmp_path / "bare"
    d.mkdir()
    table = pa.table(
        {
            "audio": pa.array(
                [_wav_bytes(synth_speech(6.0, seed=1))], type=pa.binary()
            ),
            "text": pa.array(["你好"], type=pa.string()),
            "emotion": pa.array(["happy"], type=pa.string()),
            "speaker": pa.array(["male001"], type=pa.string()),
        }
    )
    pq.write_table(table, d / "t.parquet")

    items = list(iter_parquet(d))
    assert len(items) == 1
    assert items[0].audio_bytes is not None
    assert items[0].category == "Happy"


# ---------------------------------------------------------------- 导入


def test_import_parquet_end_to_end(fake_parquet, tmp_lib):
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)

    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(fake_parquet), license="apache-2.0"),
    )

    assert res.imported == 3
    assert res.errors == []
    rows = db.iter_clips()
    assert len(rows) == 3

    # 台词进库，描述里带着它（检索质量的关键）
    assert all(r["transcript"] for r in rows)
    assert any("愤怒" in (r["description"] or "") for r in rows)

    # 性别靠 speaker 名推断，不用手工填
    assert {r["gender"] for r in rows} == {"female", "male"}

    # 类别来自列而不是文件名
    assert {r["category"] for r in rows} == {"Angry", "Sad", "Happy"}

    # 音频真的落盘了，并且是有效 wav
    for r in rows:
        p = cfg.root_path / r["filepath"]
        assert p.is_file()
        y, sr = A.load_audio(p)
        assert len(y) > 0


def test_import_parquet_over_long_goes_to_notes(tmp_path, tmp_lib):
    """超长素材入库并标 notes（用户选的策略：不丢数据）。"""
    d = tmp_path / "data"
    write_parquet(
        d / "t.parquet",
        [{
            "wav": synth_speech(18.0, seed=7),
            "name": "female001_angry_99.wav",
            "text": "这段话很长。",
            "emotion": "angry",
            "speaker": "female001",
        }],
    )
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)

    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(d), target_use="indextts"),
    )

    assert res.imported == 1
    assert res.imported_over_long == 1
    assert res.skipped_long == 0

    row = db.iter_clips()[0]
    assert row["duration_sec"] > 15.0
    assert "超出" in row["notes"]
    assert "截取" in row["notes"]


def test_import_parquet_short_is_kept_with_note(tmp_path, tmp_lib):
    """太短也入库，但写 notes（用户决定）。

    原先是直接丢弃。实测 CSEMOTIONS 有 74 条 <3s，其中 73 条是 angry 的
    短促爆发（「滚！别让我再看到你！」1.75s），恰恰是该情绪最典型的样本，
    按硬阈值丢弃等于把愤怒类的高质量短句全部排掉。
    """
    d = tmp_path / "data"
    write_parquet(
        d / "t.parquet",
        [{
            "wav": synth_burst(1.0, seed=2),
            "name": "x.wav",
            "text": "短",
            "emotion": "happy",
            "speaker": "male001",
        }],
    )
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    res = import_into(
        cfg, Database(cfg.db_path), HashingEncoder(),
        ImportOptions(kind="auto", path=str(d), target_use="indextts"),
    )
    assert res.imported == 1
    assert res.imported_short == 1
    assert res.skipped_short == 0

    row = next(iter(Database(cfg.db_path).iter_clips()))
    assert row["notes"] and "低于" in row["notes"]


# ---------------------------------------------------------------- 深层嵌套
#
# 用户实测报错：导入 CSEMOTIONS 提示
#     「无法识别该目录，导入已取消：HTTP 400: 该目录里没有找到任何音频素材」
#
# 根因是 plan_import 注释写着「对每个子目录**递归** plan」，代码却只拆一层。
# HF 下载 CSEMOTIONS 解出来的真实结构是：
#     CSEMOTIONS/data/CSEMOTIONS/data/train-0000*-of-00008.parquet
# 中间两层 `data` 的直接子项里都没有 parquet，`guess_kind` 一律返回 flat，
# 扫 wav 得 0 条；第 3 层才是真正的 parquet 目录，却永远走不到。


def test_hf_download_nested_parquet_is_found(tmp_path):
    """CSEMOTIONS 的真实三层结构：必须能一路下探到 parquet 层。"""
    rows = [
        {
            "wav": synth_speech(6.0, seed=1),
            "name": "female001_angry_01.wav",
            "text": "听到那些虚假的承诺，我的愤怒瞬间爆发。",
            "emotion": "angry",
            "speaker": "female001",
        },
        {
            "wav": synth_speech(7.0, seed=2),
            "name": "female002_sad_01.wav",
            "text": "每当我努力后却得到无情的打击。",
            "emotion": "sad",
            "speaker": "female002",
        },
    ]
    # 复刻 HF 下载后的目录：root/data/CSEMOTIONS/data/*.parquet
    root = tmp_path / "CSEMOTIONS"
    write_parquet(root / "data" / "CSEMOTIONS" / "data" / "train-00000-of-00008.parquet", rows)
    (root / "README.md").write_text("x", encoding="utf-8")
    (root / "data" / "CSEMOTIONS.zip").write_bytes(b"PK")
    (root / "data" / "CSEMOTIONS" / "NOTICE").write_text("x", encoding="utf-8")

    plans = plan_import(root)

    assert plans, "深层嵌套的 parquet 没被找到（会报 400 没有音频素材）"
    assert len(plans) == 1
    p = plans[0]
    assert p.kind == "parquet"
    assert p.count == 2
    assert p.target_use == "indextts", "中文台词素材应走 IndexTTS"
    # 路径应指向真正含 parquet 的那层
    assert p.path.name == "data"
    assert any(p.path.glob("*.parquet"))


def test_container_without_audio_returns_empty(tmp_path):
    """真正没有音频的目录仍然返回空 —— 递归不能把 400 变成误判。"""
    root = tmp_path / "空目录"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "readme.txt").write_text("x", encoding="utf-8")

    assert plan_import(root) == []


def test_recursion_is_bounded(tmp_path, monkeypatch):
    """递归必须有深度上限，否则异常嵌套会扫穿整棵盘。"""
    from voicelib import sources as S

    # 造一棵比上限更深的树，最底层才放 parquet
    deep = tmp_path
    for i in range(S._MAX_PLAN_DEPTH + 4):
        deep = deep / f"lvl{i}"
    deep.mkdir(parents=True)
    (deep / "x.parquet").write_bytes(b"PAR1")

    # 不应抛异常，也不应无限递归
    plans = plan_import(tmp_path)
    assert isinstance(plans, list)


# ---------------------------------------------------------------- parquet 上的层级选择
#
# 用户反馈：「导入的时候没让我选择上级目录作为分类」。
# 查明是两个独立问题：
#   1. 线上 server.py 是旧版，不返回 level_choices（部署问题，不是代码问题）
#   2. 代码本身：parquet 的音频内嵌在容器里，没有独立音频文件，
#      `iter_by_level` / `count_items` / `count_audio_capped` 都只认 AUDIO_EXTS，
#      于是 CSEMOTIONS 这类源在层级选择器上：
#        - count 显示 0 条（看起来像没数据）
#        - 真选了任何一层都返回 0 个计划 → 400「这一层没有音频素材」
# 下面三条钉住第 2 点。


def test_iter_by_level_includes_parquet(tmp_path):
    """选层级时 parquet 里的音频必须算进来，类别统一成该层目录名。"""
    d = tmp_path / "data"
    write_parquet(
        d / "train-00000.parquet",
        [
            {"wav": synth_speech(6.0, seed=1), "name": "a.wav",
             "text": "听到那些虚假的承诺", "emotion": "angry", "speaker": "female001"},
            {"wav": synth_speech(7.0, seed=2), "name": "b.wav",
             "text": "每当我努力后却得到无情的打击", "emotion": "sad", "speaker": "female002"},
        ],
    )

    items = list(iter_by_level(d, with_audio=False))

    assert len(items) == 2, "parquet 里的音频没被 iter_by_level 读到"
    # 类别统一成目录名
    assert {i.category for i in items} == {"data"}
    # 但原始情绪标签保留，否则无法回溯
    assert {i.category_raw for i in items} == {"angry", "sad"}
    # 台词照常带出来
    assert all(i.transcript for i in items)


def test_count_items_by_level_counts_parquet(tmp_path):
    """层级条数不能算成 0 —— 界面会显示「0 条」，看起来像没数据。"""
    d = tmp_path / "data"
    write_parquet(
        d / "train-00000.parquet",
        [
            {"wav": synth_speech(6.0, seed=1), "name": "a.wav",
             "text": "一", "emotion": "angry", "speaker": "s1"},
            {"wav": synth_speech(7.0, seed=2), "name": "b.wav",
             "text": "二", "emotion": "sad", "speaker": "s2"},
            {"wav": synth_speech(8.0, seed=3), "name": "c.wav",
             "text": "三", "emotion": "happy", "speaker": "s3"},
        ],
    )

    assert count_items("by_level", d) == 3
    # 与 parquet 自身的计数一致
    assert count_items("by_level", d) == count_items("parquet", d)


def test_count_audio_capped_counts_parquet_rows(tmp_path):
    """层级选择器上的条数取自 count_audio_capped，它也要认 parquet。"""
    from voicelib.sources import count_audio_capped

    d = tmp_path / "data"
    write_parquet(
        d / "train-00000.parquet",
        [
            {"wav": synth_speech(6.0, seed=1), "name": "a.wav",
             "text": "一", "emotion": "angry", "speaker": "s1"},
            {"wav": synth_speech(7.0, seed=2), "name": "b.wav",
             "text": "二", "emotion": "sad", "speaker": "s2"},
        ],
    )

    n, truncated = count_audio_capped(d, cap=100, budget_sec=5.0)
    assert n == 2, f"parquet 行数没算进来（得到 {n}）"
    assert truncated is False


def test_plan_import_with_category_level_on_parquet(tmp_path):
    """在 parquet 目录上选层级，必须产出计划而不是空列表（否则接口 400）。"""
    d = tmp_path / "data"
    write_parquet(
        d / "train-00000.parquet",
        [
            {"wav": synth_speech(6.0, seed=1), "name": "a.wav",
             "text": "一", "emotion": "angry", "speaker": "s1"},
        ],
    )

    plans = plan_import(d, category_level="data")

    assert plans, "选层级返回空计划 → 界面报「这一层没有音频素材」"
    assert plans[0].kind == "by_level"
    assert plans[0].count == 1
    assert plans[0].target_use == "indextts"


def test_level_choices_on_parquet_shows_nonzero(tmp_path):
    """层级按钮上不能显示「0 条」。"""
    from voicelib.sources import level_choices

    d = tmp_path / "data"
    write_parquet(
        d / "train-00000.parquet",
        [
            {"wav": synth_speech(6.0, seed=1), "name": "a.wav",
             "text": "一", "emotion": "angry", "speaker": "s1"},
            {"wav": synth_speech(7.0, seed=2), "name": "b.wav",
             "text": "二", "emotion": "sad", "speaker": "s2"},
        ],
    )

    choices = level_choices(d, max_up=1)
    root_choice = choices[0]
    assert root_choice["is_root"] is True
    assert root_choice["count"] == 2, "root 层显示 0 条会让用户以为没数据"
