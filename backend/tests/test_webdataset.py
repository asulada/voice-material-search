"""WebDataset 数据源测试（gijs/speech-utterances 这类聚合包）。

这个聚合包有几个特别容易出错的地方，测试主要就是拦这些：
  1. 根下 6 个子集**许可各不相同**，整包当一个单元会把 NC 素材标成可商用
  2. NonSpeech7k 多一层 train/ test/ 包装，不递归就整个漏掉（同样丢许可）
  3. 各子集 json 字段不同：emotion / label / style / intensity
  4. Expresso 的 text 是模板句（描述「这是对话」），必须丢弃；音频与其他字段照常入库
  5. tar 形态里音频在 json 之前，必须能配对
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from voicelib import audio as A
from voicelib.config import Config
from voicelib.db import Database
from voicelib.embedder import HashingEncoder
from voicelib.ingest import ImportOptions, import_into
from voicelib.sources import (
    count_items,
    find_webdataset_shards,
    guess_kind,
    iter_webdataset,
    plan_import,
    subset_info,
    subset_info_for_path,
    webdataset_stats,
)

from conftest import synth_burst, synth_speech


# ---------------------------------------------------------------- 构造夹具


def write_shard(
    root: Path,
    rows: list[dict],
    *,
    subdir: str = "0000",
    speech: bool = False,
    digits: int = 8,
) -> Path:
    """在 <root>/<subdir>/ 下写 <key>.wav + <key>.json（模拟解压后的分片）。"""
    d = root / subdir
    d.mkdir(parents=True, exist_ok=True)
    for i, meta in enumerate(rows):
        key = f"{i:0{digits}d}"
        y = synth_speech(6.0, seed=i) if speech else synth_burst(1.5, seed=i)
        A.write_wav(d / f"{key}.wav", y, 22050)
        (d / f"{key}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    return d


def write_tar(
    path: Path,
    rows: list[dict],
    *,
    speech: bool = False,
) -> Path:
    """写一个真正 WebDataset 顺序的 tar：音频在前、json 在后。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for i, meta in enumerate(rows):
            key = f"{i:08d}"
            y = synth_speech(6.0, seed=i) if speech else synth_burst(1.5, seed=i)
            buf = io.BytesIO()
            import soundfile as sf

            sf.write(buf, y, 22050, format="WAV", subtype="PCM_16")
            data = buf.getvalue()
            ti = tarfile.TarInfo(f"{key}.wav")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))

            payload = json.dumps(meta, ensure_ascii=False).encode()
            ti = tarfile.TarInfo(f"{key}.json")
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
    return path


@pytest.fixture
def pkg(tmp_path: Path) -> Path:
    """模拟 speech-utterances 聚合包：四个许可不同的子集。"""
    root = tmp_path / "speech-utterances"

    write_shard(root / "EmoGator", [
        {"text": "A vocal burst expressing adoration", "emotion": "Adoration",
         "contributor_id": "000001", "instance": "1", "dataset": "emogator"},
        {"text": "A vocal burst expressing fear", "emotion": "Fear",
         "contributor_id": "000002", "instance": "1", "dataset": "emogator"},
    ])
    write_shard(root / "VIVAE", [
        {"emotion": "achievement", "intensity": "low", "speaker": "S01",
         "dataset": "vivae"},
        {"emotion": "anger", "intensity": "peak", "speaker": "S02",
         "dataset": "vivae"},
    ])
    # 多一层 train/ —— 真实数据就是这样，不递归会整个漏掉
    write_shard(root / "NonSpeech7k" / "train", [
        {"label": "breath", "dataset": "nonspeech7k"},
    ])
    write_shard(root / "NonSpeech7k" / "test", [
        {"label": "cough", "dataset": "nonspeech7k"},
    ])
    write_shard(root / "Expresso", [
        # 这一条是整段双人对话：text 恰为目标句，用户听过音频确认不要
        {"text": "Expressive speech in a neutral style", "style": "default",
         "speaker": "ex01", "type": "conversational", "dataset": "expresso"},
        # 这一条是同子集里的正常条目，text 是真台词，必须保留
        {"text": "Go to hell! (spoken in a neutral style)", "style": "default",
         "speaker": "ex02", "type": "read", "dataset": "expresso"},
    ], speech=True)
    return root


# ---------------------------------------------------------------- 识别


def test_guess_kind_detects_webdataset(pkg):
    assert guess_kind(pkg / "EmoGator") == "webdataset"


def test_guess_kind_handles_train_test_wrapper(pkg):
    """NonSpeech7k/train/ 这种多一层的也要认得出。"""
    assert guess_kind(pkg / "NonSpeech7k") == "webdataset"
    assert guess_kind(pkg / "NonSpeech7k" / "train") == "webdataset"


def test_webdataset_is_not_mistaken_for_emogator(pkg):
    """分片目录（0000/）里也有音频，容易被当成 <label>/ 排布。"""
    assert guess_kind(pkg / "EmoGator") != "emogator"


def test_find_shards_skips_tar_when_dir_exists(tmp_path):
    root = tmp_path / "EmoGator"
    write_shard(root, [{"emotion": "Fear"}])
    write_tar(root / "0000.tar", [{"emotion": "Fear"}])
    shards = find_webdataset_shards(root)
    # 目录已覆盖同名 tar，不该重复读
    assert [p.name for p in shards] == ["0000"]


# ---------------------------------------------------------------- 子集许可


def test_subset_info_known():
    assert subset_info("EmoGator")["license"] == "apache-2.0"
    assert subset_info("NonSpeech7k")["license"] == "cc-by-nc-sa-4.0"


def test_subset_info_handles_camel_case_names():
    """真实数据里是 DeeplyNonverbalVocalization（连续驼峰）。

    只按下划线规范化会得到 deeplynonverbalvocalization，与表里的
    deeply_nonverbal_vocalization 对不上 → 许可丢失 → 回退全局值
    （界面默认 apache-2.0），把 CC-BY-NC-ND 标成可商用。
    """
    info = subset_info("DeeplyNonverbalVocalization")
    assert info, "驼峰名必须能认出"
    assert info["license"] == "cc-by-nc-nd-4.0"


@pytest.mark.parametrize("name,license_,commercial", [
    ("EmoGator", "apache-2.0", True),
    ("VIVAE", "cc-by-4.0", True),
    ("VocalSound", "cc-by-sa-4.0", True),
    ("DeeplyNonverbalVocalization", "cc-by-nc-nd-4.0", False),
    ("Expresso", "cc-by-nc-4.0", False),
    ("NonSpeech7k", "cc-by-nc-sa-4.0", False),
])
def test_all_real_subset_names_resolve(name, license_, commercial):
    """187 上真实存在的六个子集目录名都必须能匹配到许可。"""
    from voicelib.tags import commercial_ok_of

    info = subset_info(name)
    assert info.get("license") == license_, f"{name} 没匹配到正确许可"
    assert bool(commercial_ok_of(license_)) is commercial


def test_subset_info_lookup_walks_up(pkg):
    """train/ 自身认不出，应回退到父目录的 NonSpeech7k。"""
    info = subset_info_for_path(pkg / "NonSpeech7k" / "train")
    assert info["license"] == "cc-by-nc-sa-4.0"


def test_plan_splits_subsets_with_own_license(pkg):
    """核心回归：子集必须拆开，各自带真实许可。

    旧行为：整包一个单元 → 全部用全局 license（界面默认 apache-2.0），
    结果 NonSpeech7k / Expresso 这些 NC 素材被标成**可商用**。
    """
    plans = {p.label: p for p in plan_import(pkg)}
    assert set(plans) == {"EmoGator", "NonSpeech7k", "Expresso", "VIVAE"}

    assert plans["EmoGator"].license == "apache-2.0"
    assert plans["VIVAE"].license == "cc-by-4.0"
    assert plans["NonSpeech7k"].license == "cc-by-nc-sa-4.0"
    assert plans["Expresso"].license == "cc-by-nc-4.0"

    # 线路也按子集区分：Expresso 是对话语音
    assert plans["Expresso"].target_use == "indextts"
    assert plans["EmoGator"].target_use == "seedvc"


def test_plan_single_subset_root(pkg):
    """直接指到某个子集目录时，整根一个单元。"""
    plans = plan_import(pkg / "EmoGator")
    assert len(plans) == 1
    assert plans[0].kind == "webdataset"
    assert plans[0].license == "apache-2.0"


# ---------------------------------------------------------------- 字段映射


def test_category_falls_back_across_field_names(pkg):
    cats = {i.category for i in iter_webdataset(pkg / "EmoGator")}
    assert cats == {"Adoration", "Fear"}

    cats = {i.category for i in iter_webdataset(pkg / "NonSpeech7k")}
    assert cats == {"breath", "cough"}

    cats = {i.category for i in iter_webdataset(pkg / "Expresso")}
    assert cats == {"default"}


def test_vivae_intensity_mapped_to_stars(pkg):
    """VIVAE 自带 low/moderate/strong/peak，直接映射星级。"""
    got = {i.category: i.intensity for i in iter_webdataset(pkg / "VIVAE")}
    assert got == {"achievement": 2, "anger": 5}


@pytest.mark.parametrize("raw,stars", [
    ("low", 2), ("moderate", 3), ("strong", 4), ("peak", 5),
    ("LOW", 2), (" peak ", 5),
])
def test_intensity_mapping_table(tmp_path, raw, stars):
    root = tmp_path / "VIVAE"
    write_shard(root, [{"emotion": "x", "intensity": raw}])
    assert next(iter_webdataset(root)).intensity == stars


def test_expresso_conversational_entry_skipped(pkg):
    """text 恰为 "Expressive speech in a neutral style" 的那条整条跳过。

    用户听过音频确认：这批是整段双人对话，不要。
    """
    items = list(iter_webdataset(pkg / "Expresso"))
    texts = {i.transcript for i in items}
    assert "Expressive speech in a neutral style" not in texts
    # 只跳过这一条，同子集另一条正常条目仍在
    assert len(items) == 1


def test_expresso_other_entries_kept_with_transcript(pkg):
    """Expresso 里其余条目正常，都要，且 text 是真台词要保留。

    这条是防回归的关键：不要用 style=="default" 当判据 ——
    实测 style=default 有 825 条，大多是 type=read 的带台词句子。
    """
    items = list(iter_webdataset(pkg / "Expresso"))
    assert len(items) == 1
    kept = items[0]
    assert kept.category == "default"          # 同 style
    assert kept.transcript == "Go to hell! (spoken in a neutral style)"
    assert kept.speaker_id == "ex02"


def test_other_subsets_keep_transcript(pkg):
    """不是无差别清空：其他子集的 text 要保留。"""
    items = list(iter_webdataset(pkg / "EmoGator"))
    assert all(i.transcript for i in items)


def test_gender_inferred_from_speaker(tmp_path):
    root = tmp_path / "VocalSound"
    write_shard(root, [
        {"label": "cough", "speaker_id": "f0003", "gender": "female"},
        {"label": "sneeze", "speaker_id": "female009"},
    ])
    items = list(iter_webdataset(root))
    assert all(i.gender == "female" for i in items)


def test_license_attached_to_item(pkg):
    item = next(iter_webdataset(pkg / "NonSpeech7k"))
    assert item.extras["license"] == "cc-by-nc-sa-4.0"


# ---------------------------------------------------------------- tar 形态


def test_reads_unpacked_tar(tmp_path):
    """未解压的 tar：音频在 json 之前，靠配对缓存关联。"""
    root = tmp_path / "EmoGator"
    write_tar(root / "0000.tar", [
        {"emotion": "Adoration", "contributor_id": "000001"},
        {"emotion": "Fear", "contributor_id": "000002"},
    ])
    items = list(iter_webdataset(root))
    assert len(items) == 2
    assert {i.category for i in items} == {"Adoration", "Fear"}
    assert all(i.audio_bytes for i in items)
    assert items[0].audio_bytes[:4] == b"RIFF"


def test_tar_with_audio_false_skips_bytes(tmp_path):
    root = tmp_path / "EmoGator"
    write_tar(root / "0000.tar", [{"emotion": "Fear"}])
    items = list(iter_webdataset(root, with_audio=False))
    assert len(items) == 1
    assert items[0].category == "Fear"
    assert items[0].audio_bytes is None


def test_count_items_does_not_decode(tmp_path):
    root = tmp_path / "VIVAE"
    write_shard(root, [{"emotion": "a"}, {"emotion": "b"}, {"emotion": "c"}])
    assert count_items("webdataset", root) == 3


def test_webdataset_stats(tmp_path):
    root = tmp_path / "EmoGator"
    write_shard(root, [{"emotion": "Fear"}, {"emotion": "Fear"}, {"emotion": "Joy"}])
    st = webdataset_stats(root)
    assert st["total"] == 3
    assert st["categories"] == {"Fear": 2, "Joy": 1}


# ---------------------------------------------------------------- 端到端


def test_import_keeps_per_subset_license(pkg, tmp_lib):
    """最重要的回归：入库后各子集许可互不串味。

    请求里给的是 apache-2.0（可商用），但 NC 子集必须落成不可商用。
    """
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)

    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(pkg), license="apache-2.0"),
    )
    assert res.errors == []

    rows = db.iter_clips()
    by_lic = {}
    for r in rows:
        by_lic.setdefault(r["license"], set()).add(r["commercial_ok"])

    assert by_lic["apache-2.0"] == {1}, "EmoGator 应可商用"
    assert by_lic["cc-by-4.0"] == {1}, "VIVAE 应可商用"
    assert by_lic["cc-by-nc-sa-4.0"] == {0}, "NonSpeech7k 必须不可商用"
    assert by_lic["cc-by-nc-4.0"] == {0}, "Expresso 必须不可商用"


def test_import_records_route_and_intensity(pkg, tmp_lib):
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(pkg), license="apache-2.0"),
    )
    rows = {r["category"]: r for r in db.iter_clips()}

    # Expresso 走说话线路，其余走副语言
    assert rows["default"]["target_use"] == "indextts"
    assert rows["default"]["clip_kind"] == "speech"
    assert rows["Adoration"]["target_use"] == "seedvc"

    # VIVAE 的强度进了库
    assert rows["achievement"]["intensity"] == 2
    assert rows["anger"]["intensity"] == 5

    # Expresso 里 text 是真台词的那条保留并写入 transcript；
    # 整段对话那条（text 恰为目标句）已整条跳过
    assert rows["default"]["transcript"] == "Go to hell! (spoken in a neutral style)"
    # 其他子集保留
    assert rows["Adoration"]["transcript"]


def test_import_from_tar_end_to_end(tmp_path, tmp_lib):
    root = tmp_path / "pkg" / "EmoGator"
    write_tar(root / "0000.tar", [
        {"emotion": "Adoration", "contributor_id": "000001"},
    ])
    cfg = Config(root=str(tmp_lib))
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    res = import_into(
        cfg, db, HashingEncoder(),
        ImportOptions(kind="auto", path=str(root), license="apache-2.0"),
    )
    assert res.imported == 1
    row = db.iter_clips()[0]
    assert row["category"] == "Adoration"
    assert row["license"] == "apache-2.0"
    # 音频真的落盘了
    assert (cfg.root_path / row["filepath"]).is_file()
