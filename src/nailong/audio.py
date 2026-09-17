"""音频 I/O 与帧级特征。

原先每个脚本各写一份 `load()`（ffmpeg 管道解码）与 `feat()`（分帧 + FFT +
频段占比），阈值还各不相同。这里统一成一套，参数显式传入以保留各阶段的差异。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import config


def _ffmpeg(args: list[str], stdin: bytes | None = None) -> bytes:
    p = subprocess.run(["ffmpeg", "-v", "error", *args],
                       input=stdin, capture_output=True, check=True)
    return p.stdout


def decode(path, sr: int = config.SR_MODEL, dtype: str = "float32") -> np.ndarray:
    """把任意音视频解码成单声道 [-1,1] 波形。"""
    out = _ffmpeg(["-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"])
    x = np.frombuffer(out, dtype="<i2")
    scale = 32768.0
    return (x.astype(np.float64) / scale) if dtype == "float64" else (x.astype(np.float32) / scale)


class Decoder:
    """按 src 缓存解码结果的惰性解码器；同一集会被多个阶段反复取用。"""

    def __init__(self, sr: int = config.SR_MODEL, dtype: str = "float32"):
        self.sr, self.dtype, self._cache = sr, dtype, {}

    def __call__(self, src: str) -> np.ndarray:
        if src not in self._cache:
            self._cache[src] = decode(config.vocals_path(src), self.sr, self.dtype)
        return self._cache[src]

    def clip(self, src: str, t0: float, dur: float) -> np.ndarray:
        x = self(src)
        return x[int(t0 * self.sr):int((t0 + dur) * self.sr)].copy()


def write_wav(path, x: np.ndarray, sr: int = config.SR_EXPORT, highpass: int | None = None) -> None:
    """写 16bit wav；highpass 用于压掉分离残留的低频伴奏。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    af = ["-af", f"highpass=f={highpass}"] if highpass else []
    pcm = np.clip(np.asarray(x, dtype=np.float64) * 32768.0, -32768, 32767).astype("<i2")
    _ffmpeg(["-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
             *af, "-c:a", "pcm_s16le", str(path)], stdin=pcm.tobytes())


def frame_matrix(x: np.ndarray, fl: int, hop: int) -> np.ndarray | None:
    """加汉宁窗的帧矩阵 (nfr, fl)；太短返回 None。"""
    nfr = 1 + (len(x) - fl) // hop
    if nfr < 1:
        return None
    idx = np.arange(fl)[None, :] + hop * np.arange(nfr)[:, None]
    return x[idx] * np.hanning(fl)


def frame_db(F: np.ndarray) -> np.ndarray:
    """帧 RMS 转 dBFS，下限 -180dB 避免 log(0)。"""
    rms = np.sqrt((F ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-9))


def spectrum(F: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(F, axis=1))


def freqs(fl: int, sr: int) -> np.ndarray:
    return np.fft.rfftfreq(fl, 1 / sr)


def band_ratio(S: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """[lo, hi) 频段能量占全谱比例，逐帧。"""
    tot = (S ** 2).sum(axis=1) + 1e-12
    return (S[:, (f >= lo) & (f < hi)] ** 2).sum(axis=1) / tot


def spectral_centroid(S: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f >= lo) & (f <= hi)
    return (S[:, m] * f[m]).sum(axis=1) / (S[:, m].sum(axis=1) + 1e-9)


def f0_median(x: np.ndarray, sr: int = config.SR_MODEL,
              fl: int = config.FRAME, hop: int = config.HOP,
              f_min: float = 70.0, f_max: float = 700.0, ac_min: float = 0.35) -> float:
    """自相关基频中位数；样本不足或无稳定周期返回 nan。

    注意自相关存在倍频/半频锁定，异常平稳的 F0 未必是自然语音。
    """
    lo, hi = int(sr / f_max), int(sr / f_min)
    v = []
    for i in range(0, len(x) - fl, hop):
        s = x[i:i + fl]
        if np.sqrt((s ** 2).mean()) < 1e-4:
            continue
        s = s - s.mean()
        ac = np.correlate(s, s, "full")[fl - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        j = int(np.argmax(ac[lo:hi])) + lo
        if ac[j] >= ac_min:
            v.append(sr / j)
    return float(np.median(v)) if v else float("nan")
