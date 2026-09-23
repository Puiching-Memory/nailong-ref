"""把指定目录里的片段串成试听带（段间 1.0s 静音）并输出时长清单。

用法::

    nailong reel [源目录] [输出文件名]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import config

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else config.PRODUCTION / "accepted"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else config.REELS / "production_reel.wav"
GAP = 1.0
parts, marks = [], []
pos = 0.0
for path in sorted(SRC.glob("*.wav")):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    parts += [x, np.zeros(int(GAP * sr), dtype="float32")]
    marks.append((path.stem, len(x) / sr, pos))
    pos += len(x) / sr + GAP

OUT.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUT), np.concatenate(parts), sr)
print(f"{'#':>3} {'片段':<26}{'时长':>7}{'试听带位置':>13}")
for i, (n, d, p) in enumerate(marks, 1):
    print(f"{i:>3} {n:<26}{d:>7.2f}{p:>11.2f}s")
print(f"\n{config.rel(OUT)}  总长 {pos:.2f}s  ({sr} Hz)")
