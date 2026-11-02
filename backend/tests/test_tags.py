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
