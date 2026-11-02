"""音频处理：静音裁剪、响度近似、发声段计数、读文件。

全部按方案 §4.3 / §4.2 / §4.4 实现。关键点：
  * 只用 trim_digital_silence（阈值 −90 dBFS），**不要** librosa.effects.trim
  * lufs 用 RMS 近似，不用 pyloudnorm（短素材会抛异常）
  * gain 只算不写盘
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 静音裁剪 -

TRIM_THR_DB = -90.0
TRIM_PAD_S = 0.03
# 首尾超过 1 秒死静音时才用 -60 dBFS 再裁一次（见 §4.3）
DEAD_SILENCE_THR_DB = -60.0


def trim_digital_silence(
    y: np.ndarray, sr: int, thr_db: float = TRIM_THR_DB, pad_s: float = TRIM_PAD_S
) -> np.ndarray:
    """只裁掉近似数字静音的片段，保留所有可能是情绪的低电平声音。"""
    import librosa

    hop = 256
    if y is None or len(y) < 512:
        return y if y is not None else np.zeros(0, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    act = np.flatnonzero(db > thr_db)
    if act.size == 0:
        return y[:0]
    pad = int(pad_s * sr / hop)
    a = max(0, act[0] - pad) * hop
    b = min(len(y), (act[-1] + 1 + pad) * hop)
    return y[a:b]


def trim_dead_edges(y: np.ndarray, sr: int) -> np.ndarray:
    """首/尾有 >1 秒死静音时，**只裁那一侧**。

    这一侧的判据是「1 秒内没有任何超过 −60 dBFS 的样本」。它覆盖所有真实发声，
    所以真正的渐弱尾（哪怕低到 −52 dBFS）不会被误伤 —— 这是 §4.3 的核心顾虑。

    注意：不能像最初设想那样「检测到死静音就整段按 −60 重裁」，
    那会把另一侧的低电平渐弱尾部一起切掉。
    """
    if y is None or len(y) < sr:
        return y

    def dead(seg: np.ndarray) -> bool:
        return 20 * np.log10(max(float(np.abs(seg).max()), 1e-10)) < DEAD_SILENCE_THR_DB

    head_dead = dead(y[: int(1.0 * sr)])
    tail_dead = dead(y[-int(1.0 * sr) :])
    if not (head_dead or tail_dead):
        return y

    # 用 −60 dBFS 找出真正的有声音区间，但只在一侧生效
    import librosa

    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    act = np.flatnonzero(db > DEAD_SILENCE_THR_DB)
    if act.size == 0:
        return y[:0]
    pad = int(TRIM_PAD_S * sr / hop)
    a = max(0, act[0] - pad) * hop if head_dead else 0
    b = min(len(y), (act[-1] + 1 + pad) * hop) if tail_dead else len(y)
    return y[a:b]


def trim_for_ingest(y: np.ndarray, sr: int) -> np.ndarray:
    """入库标准流程：先只裁数字静音，再处理 >1 秒死静音。"""
    y = trim_digital_silence(y, sr)
    if len(y) == 0:
        return y
    return trim_dead_edges(y, sr)


# ---------------------------------------------------------------- 发声段计数


def count_segments(
    y: np.ndarray,
    sr: int = 22050,
    merge_frames: int = 15,
    rel_thr: float = 0.15,
    min_gap_s: float = 0.25,
) -> int:
    """数发声段（不是数峰）。段内的音节起伏不另计。见 §4.2。"""
    import librosa
    from scipy.ndimage import uniform_filter1d

    if y is None or len(y) == 0:
        return 0
    env = np.abs(librosa.stft(y, n_fft=512)).mean(axis=0)
    if env.size == 0 or float(env.max()) <= 0:
        return 0

    smooth = uniform_filter1d(librosa.util.normalize(env), merge_frames)
    active = smooth > rel_thr * smooth.max()
    if not active.any():
        return 0

    hop = 512
    min_gap = max(1, int(min_gap_s * sr / hop))
    bridged = (
        np.convolve(active, np.ones(2 * min_gap + 1, dtype=bool), mode="same") > 0
    )

    edges = np.diff(np.concatenate([[0], bridged.astype(int), [0]]))
    return int((edges == 1).sum())


# ---------------------------------------------------------------- 响度近似


def lufs_approx(y: np.ndarray) -> float:
    """近似 LUFS（未做门限），对任意长度安全。"""
    if y is None or len(y) == 0:
        return -70.0
    return float(-0.691 + 10 * np.log10(float(np.mean(y**2)) + 1e-12))


def gain_to_target(y: np.ndarray, target_lufs: float = -20.0) -> float:
    """只算增益，不改动音频（§4.4）。"""
    return float(target_lufs - lufs_approx(y))


def normalize_loudness(
    y: np.ndarray, target_lufs: float = -20.0, peak_ceiling: float = -1.0
) -> np.ndarray:
    """导出 16-bit 时才用（可选）。"""
    gain_db = target_lufs - lufs_approx(y)
    y2 = y * (10 ** (gain_db / 20))
    peak = float(np.max(np.abs(y2))) if len(y2) else 0.0
    peak_db = 20 * np.log10(max(peak, 1e-12))
    if peak_db > peak_ceiling:
        y2 = y2 * (10 ** ((peak_ceiling - peak_db) / 20))
    return y2.astype(np.float32)


def apply_gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    return (y * (10 ** (gain_db / 20))).astype(np.float32)


# ---------------------------------------------------------------- IO


def load_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    import librosa

    y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), int(actual_sr)


def write_wav(path: str | Path, y: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)


def checksum_file(path: str | Path, algo: str = "blake2b") -> str:
    h = hashlib.new(algo, digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checksum_bytes(data: bytes) -> str:
    h = hashlib.new("blake2b", digest_size=16)
    h.update(data)
    return h.hexdigest()
