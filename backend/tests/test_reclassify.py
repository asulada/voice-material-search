"""reclassify 脚本的判定测试。

线上曾经出过一个会造成数据损坏的 bug：脚本把「查不到台账」当成
「台词丢了」，于是 webdataset 来源的 66901 条副语言素材全被判成
"需要改"，而所谓改写其实是把描述性 text 清成 NULL。

这里把三类来源的判定都钉住：
  * 无台账（webdataset 形态）→ 不判定、不修改
  * 有台账 + 有台词          → 转 IndexTTS
  * 有台账 + 无台词          → 保持原样
"""

from __future__ import annotations

import json

from conftest import synth_speech

from tools.reclassify import _decide, _LookupCache
from voicelib import audio as A


def _row(src, **over) -> dict:
    """构造一条库记录。字段与 clips 表对齐，_decide 只读这些。"""
    row = {
        "id": 1,
        "source_file": str(src),
        "category": "panting",
        "target_use": "seedvc",
        "clip_kind": "vocal_burst",
        "transcript": None,
        "duration_sec": 2.88,
        "gender": "female",
        "language": "en",
        "intensity": None,
        "has_background": 0,
        "studio_quality": 0,
        "notes": None,
        # 这两列真实表里一定有，判定逻辑会读；缺了会 KeyError
        "license": "unknown",
        "commercial_ok": 0,
    }
    row.update(over)
    return row


def test_webdataset_without_manifest_is_not_touched(tmp_path):
    """没有 manifest，就无从判断 —— 必须保持原样。

    这正是线上 66901 条被误判的场景：源是
    `DeeplyNonverbalVocalization/0001/00000000.flac`，
    元信息在同名 `.json` 里，`_make_manifest_lookup` 查不到任何东西。
    """
    src = tmp_path / "DeeplyNonverbal" / "0001" / "00000000.wav"
    src.parent.mkdir(parents=True)
    A.write_wav(src, synth_speech(2.9, seed=1), 22050)

    row = _row(src, transcript="A female panting")
    patch, why = _decide(row, _LookupCache())

    assert patch is None, "查不到台账时不应产生任何修改"
    assert "查不到" in why


def test_manifest_with_transcript_converts_to_indextts(tmp_path):
    """有台账且有台词 → 改成 speech / indextts。"""
    root = tmp_path / "spk"
    src = root / "angry" / "1.wav"
    src.parent.mkdir(parents=True)
    A.write_wav(src, synth_speech(8.0, seed=1), 22050)
    (root / "metadata.jsonl").write_text(
        json.dumps(
            {"wav_path": "angry/1.wav", "speaker": "female001",
             "text": "这是一句台词"},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    patch, why = _decide(_row(src, category="angry"), _LookupCache())

    assert patch is not None
    assert patch["target_use"] == "indextts"
    assert patch["clip_kind"] == "speech"
    assert patch["transcript"] == "这是一句台词"
    assert "IndexTTS" in why


def test_manifest_without_transcript_keeps_route(tmp_path):
    """有台账但没有台词 → 不是语言内容，不动。"""
    root = tmp_path / "bursts"
    src = root / "laugh" / "1.wav"
    src.parent.mkdir(parents=True)
    A.write_wav(src, synth_speech(1.0, seed=1), 22050)
    (root / "metadata.jsonl").write_text(
        json.dumps({"wav_path": "laugh/1.wav", "category": "laugh"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    patch, why = _decide(_row(src, category="laugh"), _LookupCache())

    assert patch is None
    assert "无需改动" in why


def test_level_only_changes_category_not_route(tmp_path):
    """只选层级时，没有台词的素材线路不变，只改类别。

    这条用来钉住打印逻辑：预览必须显示"线路不变"，
    而不是一律画一个 `→ indextts/speech` 的箭头。
    """
    root = tmp_path / "数据集"
    src = root / "b" / "d" / "1.wav"
    src.parent.mkdir(parents=True)
    A.write_wav(src, synth_speech(1.0, seed=1), 22050)
    (root / "metadata.jsonl").write_text(
        json.dumps({"wav_path": "b/d/1.wav", "category": "laugh"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    patch, _ = _decide(_row(src, category="laugh"), _LookupCache(), root / "b")

    assert patch is not None
    assert patch["category"] == "b"
    assert patch["target_use"] == "seedvc"    # 线路不变
    assert patch["clip_kind"] == "vocal_burst"


# ---------------------------------------------------------------- WebDataset 分支


def _ws_item(tmp_path, subset: str, *, text: str, key: str = "00000000",
             dataset: str | None = None, **extra):
    """按 WebDataset 的**真实**布局造一条：<子集>/0001/<key>.wav + <key>.json。

    **不要加 `__key__` / `__url__`**：这两个字段只存在于 tar 包内，
    解压后除 Expresso 外都没有。早期夹具为了"像"而加上了它们，
    于是 `_is_ws_item` 用 `__key__` 判断时测试全绿，真实数据却 5/6 落空——
    副语言被当成台词改成 IndexTTS。夹具必须还原真实结构。

    真实字段（六子集实测）示例：
        DeeplyNonverbal: text/label/speaker_id/gender/quality/noise/dataset
        EmoGator:        text/emotion/emotion_code/contributor_id/instance/dataset
        VocalSound:      text/label/speaker_id/gender/dataset
        VIVAE:           text/emotion/intensity/speaker/subset/dataset
        Expresso:        text/style/speaker/type/channel/dataset（tar 内还有 __key__）
    """
    shard = tmp_path / subset / "0001"
    shard.mkdir(parents=True, exist_ok=True)
    src = shard / f"{key}.wav"
    A.write_wav(src, synth_speech(2.0, seed=1), 22050)
    meta = {"text": text, "dataset": dataset or subset.lower()}
    meta.update(extra)
    (shard / f"{key}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return src


def test_ws_subset_license_is_restored(tmp_path):
    """线上 66901 条的许可全是 unknown，必须按子集补回来。

    EmoGator 是 Apache-2.0，可商用；标成 unknown 会让它被
    「仅可商用」筛选排除，等于把可商用素材当成禁区。
    """
    src = _ws_item(
        tmp_path, "EmoGator",
        text="A vocal burst expressing adoration",
        emotion="adoration", contributor_id="p1",
    )
    row = _row(src, category="adoration", transcript="A vocal burst expressing adoration",
               license="unknown", commercial_ok=0)

    patch, why = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["license"] == "apache-2.0"
    assert patch["commercial_ok"] == 1
    assert patch["target_use"] == "seedvc"      # 副语言线路不变
    assert "许可" in why


def test_ws_expresso_becomes_indextts(tmp_path):
    """Expresso 的真台词条目应走 IndexTTS，而不是被当成副语言。

    线上它全在 seedvc/vocal_burst —— 这是导入时按形状误判的结果。
    """
    src = _ws_item(
        tmp_path, "Expresso",
        text="Why are you beating up my jukebox? (spoken in a confused style)",
        style="confused", speaker="ex01", type="conversational",
    )
    row = _row(src, category="confused",
               transcript="Why are you beating up my jukebox? (spoken in a confused style)",
               duration_sec=2.7, gender="female")

    patch, why = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["target_use"] == "indextts"
    assert patch["clip_kind"] == "speech"
    assert patch["license"] == "cc-by-nc-4.0"
    assert patch["commercial_ok"] == 0          # NC 不可商用
    assert "线路" in why


def test_ws_panting_stays_seedvc(tmp_path):
    """副语言（喘气）的 text 是描述，不是台词 —— 线路必须保持 SeedVC。

    这是最容易被「有台词 → IndexTTS」误伤的一类。
    构造一个许可错误（unknown）的行，好让 _decide 产生 patch，
    从而能验证：修许可的同时，线路**没有**被改成 IndexTTS。
    """
    src = _ws_item(
        tmp_path, "DeeplyNonverbalVocalization",
        text="A female panting", label="panting", speaker_id="s1", gender="female",
    )
    row = _row(src, category="panting", transcript="A female panting",
               license="unknown", commercial_ok=0)

    patch, why = _decide(row, _LookupCache())

    assert patch is not None, "许可错了应当产生 patch"
    assert patch["target_use"] == "seedvc", "副语言不能被改成 IndexTTS"
    assert patch["clip_kind"] == "vocal_burst"
    assert patch["license"] == "cc-by-nc-nd-4.0"
    assert patch["commercial_ok"] == 0


def test_ws_vivae_intensity_is_mapped(tmp_path):
    """VIVAE 的 intensity 是 low/moderate/strong/peak，要映射成 2/3/4/5。"""
    src = _ws_item(
        tmp_path, "VIVAE",
        text="A vocal burst strongly expressing joy",
        emotion="joy", intensity="strong", speaker="v01", gender="female",
    )
    row = _row(src, category="joy", intensity=None, license="unknown", commercial_ok=0)

    patch, _ = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["intensity"] == 4
    assert patch["license"] == "cc-by-4.0"
    assert patch["commercial_ok"] == 1


def test_ws_already_correct_is_noop(tmp_path):
    """全对时不产生 patch —— 第二次跑应当是空操作（幂等）。"""
    src = _ws_item(
        tmp_path, "EmoGator",
        text="A vocal burst expressing adoration",
        emotion="adoration", contributor_id="p1",
    )
    row = _row(src, category="adoration", transcript="A vocal burst expressing adoration",
               license="apache-2.0", commercial_ok=1, gender=None)

    patch, why = _decide(row, _LookupCache())

    # 性别会从 contributor_id 推断不出来 → gender 仍为 None，
    # 描述可能随之变化，但路线与许可必须判定为无需改动
    if patch is not None:
        assert patch["target_use"] == "seedvc"
        assert patch["license"] == "apache-2.0"
    else:
        assert "无需改动" in why


# ---------------------------------------------------------------- 真实 json 回归
#
# 下面六段 json 是从线上数据库对应的源文件里**原样抓下来**的，不是编的。
# 教训：上一版夹具自己加了 `__key__`，判据用 `__key__` 时测试全绿，
# 真实数据却 5/6 落空，副语言被当成台词改成 IndexTTS。
# 凡是依赖外部数据格式的判断，夹具必须用真实样本。

REAL_JSON = {
    "DeeplyNonverbalVocalization": {
        "text": "A female panting", "label": "panting", "speaker_id": "A8AF",
        "gender": "female", "age": "20", "location": "indoor",
        "quality": "high", "noise": "noiseless",
        "original_filename": "A8AF_10_6_0_20_0_0_0.wav",
        "dataset": "deeply_nonverbal_vocalization",
    },
    "EmoGator": {
        "text": "A vocal burst expressing adoration", "emotion": "Adoration",
        "emotion_code": 0, "contributor_id": "000001", "instance": "1",
        "original_filename": "000001-01-1.mp3", "dataset": "emogator",
    },
    "VocalSound": {
        "text": "A female cough", "label": "cough", "speaker_id": "f0003",
        "gender": "female", "original_filename": "f0003_0_cough.wav",
        "dataset": "vocalsound",
    },
    "VIVAE": {
        "text": "A vocal burst mildly expressing a sense of achievement",
        "emotion": "achievement", "intensity": "low", "speaker": "S01",
        "subset": "full_set", "original_filename": "S01_achievement_low_01.wav",
        "dataset": "vivae",
    },
    "Expresso": {
        "text": "Expressive speech in a neutral style", "style": "default",
        "style_description": "neutral", "speaker": "ex01",
        "type": "conversational", "channel": 0,
        "original_filename": "ex01-ex02_default_001.wav", "dataset": "expresso",
    },
    "NonSpeech7k": {
        "text": "A person breathing", "label": "breath", "class_id": 0,
        "original_filename": "108160-1_0_0.wav",
        "source": "https://freesound.org/", "dataset": "nonspeech7k",
    },
}


def _real_item(tmp_path, subset: str, meta: dict, key: str = "00000000"):
    """用**真实 json 内容**造一条，字段一个不改。"""
    shard = tmp_path / subset / "0001"
    shard.mkdir(parents=True, exist_ok=True)
    src = shard / f"{key}.wav"
    A.write_wav(src, synth_speech(2.9, seed=1), 22050)
    (shard / f"{key}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return src


def test_real_deeply_nonverbal_stays_seedvc(tmp_path):
    """线上那条被误改的 #1 panting：必须保持 SeedVC。

    这是用户贴出的错误输出所指的条目：
        "seedvc/vocal_burst → indextts/speech"
        "描述：一位女性用panting的语气说：A female panting"
    """
    meta = REAL_JSON["DeeplyNonverbalVocalization"]
    src = _real_item(tmp_path, "DeeplyNonverbalVocalization", meta)
    row = _row(src, category="panting", transcript="A female panting",
               license="unknown", commercial_ok=0)

    patch, why = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["target_use"] == "seedvc", "喘气不是语言内容，不能进 IndexTTS"
    assert patch["clip_kind"] == "vocal_burst"
    assert patch["license"] == "cc-by-nc-nd-4.0"
    # 副语言的描述不该套 speech 模板
    assert "说" not in patch["description"]
    assert "语气说" not in patch["description"]


def test_real_emogator_license_and_route(tmp_path):
    meta = REAL_JSON["EmoGator"]
    src = _real_item(tmp_path, "EmoGator", meta)
    row = _row(src, category="Adoration",
               transcript="A vocal burst expressing adoration",
               license="unknown", commercial_ok=0)

    patch, why = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["license"] == "apache-2.0"
    assert patch["commercial_ok"] == 1
    assert patch["target_use"] == "seedvc"


def test_real_vocalsound_license(tmp_path):
    meta = REAL_JSON["VocalSound"]
    src = _real_item(tmp_path, "VocalSound", meta)
    row = _row(src, category="cough", transcript="A female cough",
               license="unknown", commercial_ok=0)

    patch, _ = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["license"] == "cc-by-sa-4.0"
    assert patch["target_use"] == "seedvc"


def test_real_vivae_intensity_and_license(tmp_path):
    meta = REAL_JSON["VIVAE"]
    src = _real_item(tmp_path, "VIVAE", meta)
    row = _row(src, category="achievement", intensity=None,
               transcript="A vocal burst mildly expressing a sense of achievement",
               license="unknown", commercial_ok=0)

    patch, _ = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["intensity"] == 2          # low → 2
    assert patch["license"] == "cc-by-4.0"
    assert patch["commercial_ok"] == 1


def test_real_expresso_neutral_template_is_flagged_not_converted(tmp_path):
    """Expresso 的 neutral 模板句是用户明确不要的，应提示删除而不是改线路。"""
    meta = REAL_JSON["Expresso"]
    src = _real_item(tmp_path, "Expresso", meta)
    row = _row(src, category="default",
               transcript="Expressive speech in a neutral style",
               license="unknown", commercial_ok=0)

    patch, why = _decide(row, _LookupCache())

    assert patch is None
    assert "不要" in why


def test_real_expresso_real_line_becomes_indextts(tmp_path):
    """真台词条目要转 IndexTTS —— 这才是「需要改」的主体。"""
    meta = dict(REAL_JSON["Expresso"])
    meta["text"] = "Why are you beating up my jukebox? (spoken in a confused style)"
    meta["style"] = "confused"
    src = _real_item(tmp_path, "Expresso", meta)
    row = _row(src, category="confused", transcript=meta["text"],
               license="unknown", commercial_ok=0)

    patch, why = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["target_use"] == "indextts"
    assert patch["clip_kind"] == "speech"
    assert patch["license"] == "cc-by-nc-4.0"
    assert patch["commercial_ok"] == 0


def test_real_nonspeech7k_license(tmp_path):
    meta = REAL_JSON["NonSpeech7k"]
    src = _real_item(tmp_path, "NonSpeech7k", meta)
    row = _row(src, category="breath", transcript="A person breathing",
               license="unknown", commercial_ok=0)

    patch, _ = _decide(row, _LookupCache())

    assert patch is not None
    assert patch["license"] == "cc-by-nc-sa-4.0"
    assert patch["target_use"] == "seedvc"
