"""pytest 配置：把 backend/ 加入 sys.path，并提供合成测试素材。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ------------------------------------------------------------------ 合成音频


def synth_speech(seconds: float = 6.0, sr: int = 22050, seed: int = 0) -> np.ndarray:
    """伪「说话」：有起伏的谐波，前后各留一段数字静音。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 120 + 15 * np.sin(2 * np.pi * 2.0 * t)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    y = (y * env * 0.2).astype(np.float32)
    y += (rng.normal(0, 0.001, n)).astype(np.float32)
    pad = int(0.1 * sr)
    return np.concatenate([np.zeros(pad, np.float32), y, np.zeros(pad, np.float32)])


def synth_burst(seconds: float = 1.5, sr: int = 22050, seed: int = 1) -> np.ndarray:
    """伪「副语言」：两次短促的噪声爆发，中间留 0.4s 静音。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    for start in (0.1, 0.75):
        a = int(start * sr)
        b = min(n, a + int(0.15 * sr))
        burst = rng.normal(0, 0.15, b - a)
        fade = np.hanning(b - a).astype(np.float32)
        y[a:b] += (burst * fade).astype(np.float32)
    return y


def synth_decaying_sigh(seconds: float = 3.0, sr: int = 22050, decay_rate: float = 4.0) -> np.ndarray:
    """渐起 → 主体 → 长渐弱尾，收尾落在 −40 ~ −50 dBFS 附近。

    这是 §4.3 用来说明「不要用 librosa.effects.trim」的那类样本：
    情绪最浓的收尾恰好电平最低，相对阈值会在渐弱中途下刀。

    布局按 §4.3 的实测样本：渐起 13% + 主体到 38% + 渐弱到 75%，其后是数字静音。
    decay_rate 越小，渐弱越慢、低电平内容越长。
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr
    env = np.zeros(n, dtype=np.float32)
    a, b, end = int(0.13 * n), int(0.38 * n), int(0.75 * n)
    env[a:b] = np.linspace(0, 1, b - a) ** 0.5
    env[b:end] = np.exp(-decay_rate * (t[b:end] - t[b])) * 0.9
    body = np.sin(2 * np.pi * 300 * t) * 0.25
    return (body * env).astype(np.float32)


@pytest.fixture
def sr() -> int:
    return 22050


@pytest.fixture
def tmp_lib(tmp_path: Path):
    """一个干净的 voice_library 根目录。"""
    root = tmp_path / "voice_library"
    root.mkdir()
    return root
