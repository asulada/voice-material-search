"""标签 / 许可 / 描述模板测试。"""

from __future__ import annotations

from voicelib import tags as T


def test_commercial_ok_from_license():
    assert T.commercial_ok_of("apache-2.0") == 1
    assert T.commercial_ok_of("CC-BY-NC-SA-4.0") == 0
    assert T.commercial_ok_of("cc-by-nc-4.0") == 0
    assert T.commercial_ok_of("NonCommercial") == 0
    assert T.commercial_ok_of("non-commercial") == 0
    assert T.commercial_ok_of("cc-by-4.0") == 1
    # 空许可按不可商用处理（保守）
    assert T.commercial_ok_of("") == 0
    assert T.commercial_ok_of(None) == 0


def test_unknown_license_is_not_commercial():
    """许可"未知"必须按不可商用处理。

    之前的实现只排除 NC 与空值，于是 "unknown" 会落进"其余→1"，
    把来源不明的素材标成可商用——这是会出法律问题的方向，
    不能靠"用户记得自己改"兜底。
    """
    for value in ("unknown", "Unknown", "UNSPECIFIED", "unclear", "n/a", "none", "other"):
        assert T.commercial_ok_of(value) == 0, value
    # 明确可商用的仍然放行
    assert T.commercial_ok_of("Apache-2.0") == 1
    assert T.commercial_ok_of("MIT") == 1


def test_speech_description_matches_doc_example():
    desc = T.build_speech_description(
        emotion="Angry",
        transcript="你怎么能这样对我",
        gender="male",
        studio_quality=1,
    )
    assert desc.startswith("一位男性用愤怒的语气说：")
    assert "录音干净" in desc


def test_burst_description_has_level_and_duration():
    desc = T.build_burst_description(
        category="laugh", gender="male", intensity=4, duration_sec=1.8
    )
    assert "强烈" in desc
    assert "男性" in desc
    assert "笑声" in desc
    assert "1.8秒" in desc


def test_description_always_non_empty():
    for kind, cat in [
        ("speech", ""),
        ("vocal_burst", ""),
        ("speech", "Weird-Label"),
        ("vocal_burst", "unknown"),
    ]:
        d = T.build_description(clip_kind=kind, category=cat)
        assert d and d.strip()


def test_emotion_chinese_mapping():
    assert T.emotion_cn("Angry") == "愤怒"
    assert T.emotion_cn("angry") == "愤怒"
    assert T.emotion_cn("Neutral") == "平静"
    assert T.emotion_cn(None) == ""


def test_burst_kind_mapping_handles_unknown():
    assert T.burst_kind_cn("laugh") == "笑声"
    assert T.burst_kind_cn("cough") == "咳嗽"
    assert T.burst_kind_cn("some_new_label") == "some_new_label"


def test_gender_normalization():
    assert T.normalize_gender("Male") == "male"
    assert T.normalize_gender("F") == "female"
    assert T.normalize_gender("女") == "female"
    assert T.normalize_gender("") is None
    assert T.normalize_gender(None) is None


def test_segment_note_thresholds():
    assert T.segment_note(0) is None
    assert T.segment_note(1) is None
    assert T.segment_note(2) is not None
    assert T.segment_note(5) is not None
    assert "5" in T.segment_note(5)


def test_safe_slug_keeps_chinese():
    s = T.safe_slug("S03/Angry 愤怒")
    assert "/" not in s
    assert "愤怒" in s


# ---------------------------------------------------------------- 线上六子集真实类别
#
# 这些类别是从线上库里按子集统计出来的**实际取值**，不是编的。
# 补全之前，burst 有 45 个、speech 有 24 个类别会原样显示英文，
# 描述变成「一段中等的Adoration」「一位用confused的语气说」这种中英混杂，
# 而 description 是检索的唯一依据（§2.3），中英混杂会削弱中文语义检索。

_EMOGATOR_30 = [
    "Adoration", "Anger", "Disappointment", "Guilt", "Interest", "Pride",
    "Serenity", "Surprise (Positive)", "Sympathy", "Awe", "Confusion", "Desire",
    "Disgust", "Neutral", "Realization", "Shame", "Surprise (Negative)",
    "Triumph", "Amusement", "Contentment", "Distress", "Ecstasy", "Fear",
    "Pain", "Relief", "Romantic Love", "Sadness", "Contempt", "Elation",
    "Embarrassment",
]

_VOCALSOUND_6 = ["laughter", "sneeze", "throat clearing", "sigh", "sniff", "cough"]

_VIVAE_6 = ["pleasure", "surprise", "pain", "fear", "anger", "achievement"]

_DEEPLY_16 = [
    "lip popping", "throat clearing", "sneezing", "screaming", "sighing",
    "coughing", "nose blowing", "tongue clicking", "yawning",
    "teeth chattering", "lip smacking", "moaning", "panting",
    "teeth grinding", "laughing", "crying",
]

_EXPRESSO_STYLES = [
    "laughing", "happy", "sad", "whisper", "confused", "enunciated", "default",
    "default_emphasis", "default_essentials", "disgusted", "nonverbal", "fast",
    "projected", "sympathetic", "sarcastic", "angry", "narration", "calm",
    "animal", "animaldir", "awe", "desire", "fearful", "bored", "child",
    "childdir", "singing", "sleepy", "default_longform", "narration_longform",
]


def test_all_emogator_categories_have_chinese():
    """EmoGator 30 类必须都有中文，否则 32065 条描述全带英文。"""
    missing = [c for c in _EMOGATOR_30
               if T.burst_kind_cn(c).strip().lower() == c.strip().lower()]
    assert missing == [], f"缺中文映射: {missing}"


def test_all_vocalsound_categories_have_chinese():
    missing = [c for c in _VOCALSOUND_6
               if T.burst_kind_cn(c).strip().lower() == c.strip().lower()]
    assert missing == [], f"缺中文映射: {missing}"


def test_all_vivae_categories_have_chinese():
    missing = [c for c in _VIVAE_6
               if T.burst_kind_cn(c).strip().lower() == c.strip().lower()]
    assert missing == [], f"缺中文映射: {missing}"


def test_all_deeplynoverbal_categories_have_chinese():
    """这个子集用空格分隔的多词标签，与表里旧的下划线写法对不上。"""
    missing = [c for c in _DEEPLY_16
               if T.burst_kind_cn(c).strip().lower() == c.strip().lower()]
    assert missing == [], f"缺中文映射: {missing}"


def test_all_expresso_styles_have_chinese():
    """Expresso 的 style 类走 speech 模板，同样要翻。"""
    missing = [c for c in _EXPRESSO_STYLES
               if T.emotion_cn(c).strip().lower() == c.strip().lower()]
    assert missing == [], f"缺中文映射: {missing}"


def test_ing_form_differs_from_base_form():
    """sneeze → sneezing 词尾变了，子串匹配兜不住，必须单列。

    这是实测踩到的：`"sneeze" in "sneezing"` 为 False。
    同类还有 laughing（laugh 能命中）等，只有去 e 加 ing 的要特别注意。
    """
    assert T.burst_kind_cn("sneeze") == "喷嚏"
    assert T.burst_kind_cn("sneezing") == "喷嚏"
    assert "sneeze" not in "sneezing"   # 记录这个事实，防止有人以为能靠子串


def test_burst_description_is_fully_chinese_for_emogator():
    """端到端：EmoGator 的类别进去，描述里不该再出现英文类别名。"""
    for cat in ("Adoration", "Romantic Love", "Surprise (Positive)"):
        desc = T.build_burst_description(
            category=cat, gender="female", intensity=3, duration_sec=1.5
        )
        assert cat.lower() not in desc.lower(), f"{cat} 未翻译：{desc}"
        assert "一段" in desc and "秒。" in desc


def test_speech_description_is_fully_chinese_for_expresso():
    for style in ("confused", "enunciated", "whisper"):
        desc = T.build_speech_description(
            emotion=style, transcript="测试台词", gender="female", duration_sec=2.0
        ) if "duration_sec" in T.build_speech_description.__code__.co_varnames else \
            T.build_description(
                clip_kind="speech", category=style, transcript="测试台词",
                gender="female", duration_sec=2.0,
            )
        assert style.lower() not in desc.lower(), f"{style} 未翻译：{desc}"
