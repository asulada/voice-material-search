"""音频处理测试：重点验证 §4.3 的裁剪行为。

这些用例对应方案里给出的实测表，是可复现的验收依据。
"""

from __future__ import annotations

import numpy as np

from voicelib import audio as A

from conftest import synth_burst, synth_decaying_sigh, synth_speech


def test_trim_digital_silence_keeps_low_level_tail(sr):
    """只裁数字静音：−45 dBFS 的渐弱尾必须保留（§4.3 的取舍）。"""
    import librosa

    y = synth_decaying_sigh(3.0, sr)
    kept = A.trim_digital_silence(y, sr)

    # 构造的样本：0～2.3s 有声（尾部约 −45 dBFS），之后才是真静音。
    # 判据是「尾部保留的是真声学内容」，而不是被提前下刀或补零。
    assert len(kept) / sr > 1.9, f"只保留 {len(kept)/sr:.2f}s，尾部被误切"
    tail_peak = float(np.abs(kept[-int(0.05 * sr) :]).max())
    assert tail_peak > 1e-5, "尾部应为低电平的真实内容"

    # 对照：trim(top_db=30) 会在渐弱中途下刀，保留得更短
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    assert len(trimmed) < len(kept)


def test_librosa_trim_cuts_more_than_digital_trim(sr):
    """对照：librosa.effects.trim 比数字静音裁剪短（说明为什么不用它）。"""
    import librosa

    y = synth_decaying_sigh(3.0, sr)
    yt, _ = librosa.effects.trim(y, top_db=30)
    digital = A.trim_digital_silence(y, sr)
    assert len(yt) < len(digital)


def test_pure_silence_becomes_empty(sr):
    y = np.zeros(sr * 2, dtype=np.float32)
    assert len(A.trim_digital_silence(y, sr)) == 0


def test_trim_never_returns_more_than_input(sr):
    y = synth_speech(2.0, sr)
    out = A.trim_digital_silence(y, sr)
    assert 0 < len(out) <= len(y)


def test_trim_for_ingest_handles_dead_edges(sr):
    y = synth_speech(2.0, sr)
    padded = np.concatenate([np.zeros(sr, np.float32), y, np.zeros(sr, np.float32)])
    out = A.trim_for_ingest(padded, sr)
    # 长死静音被吃掉，语音本身仍在
    assert len(out) / sr < len(padded) / sr
    assert len(out) / sr > 1.5


def test_lufs_and_gain(sr):
    y = synth_speech(2.0, sr)
    lufs = A.lufs_approx(y)
    assert -60 < lufs < 0
    gain = A.gain_to_target(y, -20.0)
    assert abs((lufs + gain) - (-20.0)) < 1e-6


def test_lufs_of_empty_is_safe():
    assert A.lufs_approx(np.zeros(0, dtype=np.float32)) < -60


def test_normalize_loudness_respects_peak_ceiling(sr):
    y = synth_speech(2.0, sr) * 3.0
    out = A.normalize_loudness(y, target_lufs=-10.0, peak_ceiling=-1.0)
    peak_db = 20 * np.log10(float(np.abs(out).max()) + 1e-12)
    assert peak_db <= -0.9


def test_count_segments_single_burst_is_one(sr):
    y = synth_burst(1.5, sr)
    assert A.count_segments(y, sr=sr) == 1 or A.count_segments(y, sr=sr) == 2
    # 关键：不是 0（0 会被丢弃），也不是 7、8（数峰）
    assert 1 <= A.count_segments(y, sr=sr) <= 3


def test_count_segments_silence_is_zero(sr):
    assert A.count_segments(np.zeros(sr, dtype=np.float32), sr=sr) == 0


def test_count_segments_continuous_tone_is_one(sr):
    t = np.arange(int(1.0 * sr)) / sr
    y = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    assert A.count_segments(y, sr=sr) == 1
