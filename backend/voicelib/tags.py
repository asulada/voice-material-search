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
    """由许可推导是否可商用（方案 §4.5）。

    只有**明确的可商用许可**才返回 1。以下一律 0：
      * 空值 / None
      * 含 NC / NonCommercial 条款
      * 未知（unknown / unspecified / 无法识别）

    最后一条是刻意的保守：许可不明确时按不可商用处理，
    避免把 NC 素材误标成可商用。代价是需要用户核对后手动改。
    """
    s = (license_str or "").strip().lower()
    if not s:
        return 0
    if "nc" in s or "noncommercial" in s or "non-commercial" in s:
        return 0
    # 明确的"未知"类取值：不猜，按不可商用
    if s in ("unknown", "unspecified", "unclear", "n/a", "na", "none", "other"):
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
    # ---- Expresso 的 style 类（走 speech 模板，必须翻成中文）----
    # 这些是说话风格而非情绪，但描述模板同一套，翻译口径保持一致。
    "laughing": "大笑",
    "whisper": "低语",
    "confused": "困惑",
    "enunciated": "吐字清晰",
    "default": "自然",
    "default_emphasis": "强调语气",
    "default_essentials": "基础语气",
    "nonverbal": "非语言",
    "fast": "快速",
    "projected": "高声",
    "sympathetic": "同情",
    "sarcastic": "讽刺",
    "narration": "叙述",
    "animal": "动物声",
    "animaldir": "动物声",
    "awe": "敬畏",
    "desire": "渴望",
    "bored": "厌倦",
    "child": "童声",
    "childdir": "童声",
    "singing": "歌唱",
    "sleepy": "困倦",
    "default_longform": "长文自然",
    "narration_longform": "长文叙述",
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
    # sneeze → sneezing 去掉了词尾 e，子串匹配不到，必须单列
    "sneezing": "喷嚏",
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
    # ---- EmoGator 30 类（GoEmotions 本体，被当作发声的情绪标签）----
    # 注意 burst_kind_cn 有子串匹配兜底（`if k in key`），
    # 所以这里只放完整词，带括号的放在对应短词前面更稳妥。
    "surprise (positive)": "惊喜",
    "surprise (negative)": "惊愕",
    "romantic love": "爱意",
    "adoration": "倾慕",
    "anger": "愤怒",
    "disappointment": "失望",
    "guilt": "内疚",
    "interest": "兴趣",
    "pride": "自豪",
    "serenity": "安宁",
    "sympathy": "同情",
    "awe": "敬畏",
    "confusion": "困惑",
    "desire": "渴望",
    "disgust": "厌恶",
    "neutral": "平静",
    "realization": "领悟",
    "shame": "羞愧",
    "triumph": "得意",
    "amusement": "愉悦",
    "contentment": "满足",
    "distress": "痛苦",
    "ecstasy": "狂喜",
    "fear": "恐惧",
    "pain": "疼痛",
    "relief": "宽慰",
    "sadness": "悲伤",
    "contempt": "轻蔑",
    "elation": "兴奋",
    "embarrassment": "尴尬",
    # ---- VIVAE 6 类（与上面重叠的已覆盖，这里补差异项）----
    "pleasure": "愉悦",
    "surprise": "惊讶",
    "achievement": "成就感",
    # ---- VocalSound / DeeplyNonverbalVocalization 的多词标签 ----
    # 现有表里是 "throat_clear"（下划线），而这两个子集用空格，
    # 子串匹配对不上，必须单独加。
    "throat clearing": "清嗓",
    "lip popping": "弹唇",
    "nose blowing": "擤鼻",
    "tongue clicking": "弹舌",
    "teeth chattering": "牙齿打颤",
    "lip smacking": "咂嘴",
    "teeth grinding": "磨牙",
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

# ---------------------------------------------------------------- 语种
#
# 显示层翻译，和类别同理：库里存 BCP-47 风格的码（`en` / `zh`），
# 界面上给中文名。映射不到的码原样显示，不留空。
LANGUAGE_CN: dict[str, str] = {
    "zh": "中文",
    "en": "英文",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "ru": "俄语",
}


def language_cn(code: str | None) -> str:
    if not code:
        return "未标注"
    key = str(code).strip().lower()
    if key in LANGUAGE_CN:
        return LANGUAGE_CN[key]
    base = key.split("-")[0]  # zh-CN → zh
    if base in LANGUAGE_CN:
        return LANGUAGE_CN[base]
    return str(code)


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
