"""标签、许可、描述模板。

三个关键约定：
  * commercial_ok 必须由 license 推导（§4.5），不允许手工填
  * description 是检索的唯一依据（§2.3），模板只保证“非空 + 把字段翻译成中文”
  * 描述模板的产出不会限制查询词，语义模型负责把近义词拉到一起
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------- 许可


def commercial_ok_of(license_str: str | None) -> int:
    """CC-BY-NC / NC / non-commercial → 0，其余 → 1。"""
    s = (license_str or "").lower()
    if not s:
        return 0
    if "nc" in s or "noncommercial" in s or "non-commercial" in s:
        return 0
    return 1


def normalize_license(license_str: str | None) -> str:
    s = (license_str or "").strip()
    return s or "unknown"


# ---------------------------------------------------------------- 情绪映射

# CSEMOTIONS → 中文（§2.3）
SPEECH_EMOTION_CN: dict[str, str] = {
    "Angry": "愤怒",
    "Happy": "高兴",
    "Sad": "悲伤",
    "Fearful": "恐惧",
    "Surprise": "惊讶",
    "Playfulness": "俏皮",
    "Neutral": "平静",
    # 常见小写 / 别名
    "anger": "愤怒",
    "angry": "愤怒",
    "happy": "高兴",
    "happiness": "高兴",
    "sad": "悲伤",
    "sadness": "悲伤",
    "fearful": "恐惧",
    "fear": "恐惧",
    "surprise": "惊讶",
    "surprised": "惊讶",
    "playful": "俏皮",
    "playfulness": "俏皮",
    "neutral": "平静",
    "calm": "平静",
    "disgust": "厌恶",
    "disgusted": "厌恶",
    "melancholic": "忧郁",
}

# EmoGator 30 类 → IndexTTS 8 维（§3.6 的有损映射，仅作参考/过滤别名）
EMOTION_TO_INDEXTTS: dict[str, str] = {
    "Amusement": "happy",
    "Elation": "happy",
    "Ecstasy": "happy",
    "Triumph": "happy",
    "Pride": "happy",
    "Anger": "angry",
    "Contempt": "angry",
    "Disgust": "disgusted",
    "Sadness": "sad",
    "Disappointment": "sad",
    "Distress": "sad",
    "Guilt": "sad",
    "Shame": "sad",
    "Fear": "afraid",
    "Pain": "afraid",
    "Surprise (Negative)": "surprised",
    "Surprise (Positive)": "surprised",
    "Awe": "surprised",
    "Realization": "surprised",
    "Contentment": "calm",
    "Serenity": "calm",
    "Relief": "calm",
    "Neutral": "calm",
    "Sympathy": "calm",
}

# 副语言类别 → 中文（§2.3）
BURST_KIND_CN: dict[str, str] = {
    "laugh": "笑声",
    "laughing": "笑声",
    "giggle": "轻笑",
    "cry": "哭声",
    "crying": "哭声",
    "sob": "啜泣",
    "cough": "咳嗽",
    "coughing": "咳嗽",
    "breath": "喘息",
    "breathing": "喘息",
    "sigh": "叹息",
    "scream": "尖叫",
    "screaming": "尖叫",
    "sniff": "抽鼻子",
    "sneeze": "喷嚏",
    "yawn": "哈欠",
    "groan": "呻吟",
    "moan": "呻吟",
    "gasp": "倒吸气",
    "hum": "哼唱",
    "shush": "嘘声",
    "throat_clear": "清嗓",
    "swallow": "吞咽",
    "snore": "鼾声",
    "whistle": "口哨",
    "sing": "哼歌",
    "scream_laugh": "尖笑",
    "grunt": "哼声",
    "pant": "喘气",
    "hiss": "嘶声",
    "yell": "喊叫",
}

# 词边界模糊匹配用的中文别名（正则里统一处理大小写）
_EMOTION_ALIASES = {
    "愤怒": ["angry", "anger", "rage", "furious", "愤怒", "怒", "生气"],
    "高兴": ["happy", "happiness", "joy", "高兴", "开心", "快乐", "喜悦"],
    "悲伤": ["sad", "sadness", "sorrow", "悲伤", "难过", "伤心"],
    "恐惧": ["fearful", "fear", "afraid", "恐惧", "害怕"],
    "惊讶": ["surprise", "surprised", "惊讶", "吃惊"],
    "俏皮": ["playful", "playfulness", "俏皮", "调皮"],
    "平静": ["neutral", "calm", "平静", "冷静", "中性"],
    "厌恶": ["disgust", "disgusted", "厌恶", "恶心"],
    "忧郁": ["melancholic", "melancholy", "忧郁", "抑郁"],
}


def emotion_cn(label: str | None) -> str:
    if not label:
        return ""
    if label in SPEECH_EMOTION_CN:
        return SPEECH_EMOTION_CN[label]
    key = str(label).strip().lower()
    for k, v in SPEECH_EMOTION_CN.items():
        if k.lower() == key:
            return v
    return str(label)


def is_emotion_label(label: str | None) -> bool:
    if not label:
        return False
    key = str(label).strip().lower()
    return any(key in [a.lower() for a in aliases] for aliases in _EMOTION_ALIASES.values())


def burst_kind_cn(label: str | None) -> str:
    if not label:
        return "发声"
    key = str(label).strip().lower()
    if key in BURST_KIND_CN:
        return BURST_KIND_CN[key]
    for k, v in BURST_KIND_CN.items():
        if k in key:
            return v
    return str(label)


# ---------------------------------------------------------------- 性别

GENDER_CN = {"male": "男", "female": "女", "m": "男", "f": "女", "男": "男", "女": "女"}


def normalize_gender(g: str | None) -> str | None:
    if not g:
        return None
    v = str(g).strip().lower()
    if v in ("male", "m", "man", "男"):
        return "male"
    if v in ("female", "f", "woman", "女"):
        return "female"
    return None


def gender_cn(g: str | None) -> str:
    return GENDER_CN.get((g or "").lower(), "")


# ---------------------------------------------------------------- 描述模板


def build_speech_description(
    *,
    emotion: str,
    transcript: str | None,
    gender: str | None = None,
    language: str | None = None,
    has_background: int = 0,
    studio_quality: int = 1,
    intensity: int | None = None,
) -> str:
    """IndexTTS 线路的描述（§2.3）。"""
    e = emotion_cn(emotion) or "中性"
    g = gender_cn(gender)
    g_prefix = f"{g}性" if g else ""
    text = (transcript or "").strip() or "（无台词）"
    if text and not text.endswith(("。", "！", "？", "…", ".", "!", "?")):
        text += "。"
    parts = [f"一位{g_prefix}用{e}的语气说：{text}"]

    if intensity:
        level = {1: "很轻微", 2: "偏轻", 3: "中等", 4: "强烈", 5: "非常强烈"}.get(
            int(intensity), "中等"
        )
        parts.append(f"情绪强度{level}。")
    if studio_quality:
        parts.append("录音干净，无背景音。")
    else:
        parts.append("录音有一定底噪。" if has_background else "录音可用。")
    if language:
        lang_cn = {"zh": "中文", "en": "英文", "ko": "韩语", "pt": "葡萄牙语"}.get(
            language, language
        )
        parts.append(f"语言为{lang_cn}。")
    return "".join(parts)


def build_burst_description(
    *,
    category: str,
    gender: str | None = None,
    intensity: int | None = None,
    duration_sec: float = 0.0,
    the_emotion: str | None = None,
) -> str:
    """SeedVC 线路的描述（§2.3）。"""
    kind = burst_kind_cn(category)
    g = gender_cn(gender)
    g_prefix = f"{g}性" if g else ""
    level = {1: "微弱", 2: "轻微", 3: "中等", 4: "强烈", 5: "爆发"}.get(
        int(intensity or 3), "中等"
    )
    desc = f"一段{level}的{g_prefix}{kind}，单次发声，{duration_sec:.1f}秒。"
    if the_emotion:
        desc += f"情绪上是{emotion_cn(the_emotion)}。"
    if not gender:
        desc += "性别未知。"
    return desc


def build_description(
    *,
    clip_kind: str,
    category: str,
    transcript: str | None = None,
    gender: str | None = None,
    language: str | None = None,
    intensity: int | None = None,
    duration_sec: float = 0.0,
    has_background: int = 0,
    studio_quality: int = 0,
) -> str:
    if clip_kind == "speech":
        return build_speech_description(
            emotion=category,
            transcript=transcript,
            gender=gender,
            language=language,
            has_background=has_background,
            studio_quality=studio_quality,
            intensity=intensity,
        )
    return build_burst_description(
        category=category,
        gender=gender,
        intensity=intensity,
        duration_sec=duration_sec,
    )


# ---------------------------------------------------------------- 发声段标记


def segment_note(n: int) -> str | None:
    """§4.2：n==0 丢弃 → 返回 None；n>=2 写 notes。"""
    if n >= 4:
        return f"检出 {n} 段发声，疑似录音事故，建议人工核对"
    if n >= 2:
        return f"检出 {n} 段发声，建议人工核对"
    return None


# ---------------------------------------------------------------- 命名


def safe_slug(text: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(text or ""))
    return s.strip("_")[:48] or "unknown"


def describe_row(row: dict[str, Any]) -> str:
    """给前端 / 日志用的一行摘要。"""
    return f"#{row.get('id')} {row.get('filepath')} · {row.get('description', '')[:40]}"
