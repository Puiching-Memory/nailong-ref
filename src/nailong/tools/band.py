"""通用频段分析：任意输入经 ffmpeg 转单声道 24k 后统计，用于对比分离前后的 BGM 残留。

注意低频占比不能直接当 BGM 判据——导出时的 highpass=f=80 已把 0-120Hz 压到 1~2%。
要判断伴奏必须做差分：同窗口分别测 original / vocals / no_vocals 三个 stem。
"""
import sys

import numpy as np

from .. import audio, config

SR = config.SR_EXPORT
FL, HOP = 960, 480  # 40ms / 20ms @24k
BANDS = [(0, 120), (120, 300), (300, 1000), (1000, 4000), (4000, 8000)]


def frames(x):
    F = audio.frame_matrix(x, FL, HOP)
    if F is None:
        return None
    S = audio.spectrum(F)
    f = audio.freqs(FL, SR)
    return audio.frame_db(F), [audio.band_ratio(S, f, a, b) for a, b in BANDS]


for path in sys.argv[1:]:
    x = audio.decode(path, SR, dtype="float64")
    r = frames(x)
    if r is None:
        print(f"\n=== {path} ===  太短，跳过")
        continue
    db, bs = r
    print(f"\n=== {path} ===  {len(x) / SR:.2f}s  frames={len(db)}")
    print("  RMS 百分位(dBFS): " + "  ".join(
        f"p{q}={np.percentile(db, q):.1f}" for q in (1, 5, 10, 25, 50, 90, 100)))
    print("  全帧频段占比:     " + "  ".join(
        f"{a}-{b}Hz={v.mean() * 100:.1f}%" for (a, b), v in zip(BANDS, bs, strict=True)))
    idx = np.argsort(db)[:max(1, int(len(db) * 0.08))]
    print("  最静8%(底噪指纹): " + "  ".join(
        f"{a}-{b}Hz={v[idx].mean() * 100:.1f}%" for (a, b), v in zip(BANDS, bs, strict=True)))
