"""声纹模型自检：确认嵌入真有区分度，而不是任意输入都挤在一起。

若 语音 vs 白噪声 的余弦也很高，说明模型/输入格式有问题，
那么此前"无簇结构"的结论就不能归因于素材本身。
"""
import sys

import numpy as np

from .. import audio, config, manifests
from ..speakers import RedimNetEmbedder, l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = config.SR_MODEL

emb = RedimNetEmbedder()
dec = audio.Decoder(SR)

rows = manifests.by_idx(config.UTT_MANIFEST)
a = rows[15]
A = dec.clip(a["src"], float(a["t0"]), float(a["dur"]))
b = rows[6]
B = dec.clip(b["src"], float(b["t0"]), float(b["dur"]))

rng = np.random.default_rng(0)
N = rng.standard_normal(len(A)).astype(np.float32) * 0.05

# 同一句话自己 + 30dB 噪声：理论余弦应 ~1.0
A2 = A + rng.standard_normal(len(A)).astype(np.float32) * (np.abs(A).max() / 30)

# 同一个人、同一段，但走"原始混音"通道（应仍高）
orig = audio.decode(config.mix_path(a["src"]), SR)
A_orig = orig[int(float(a["t0"]) * SR):int((float(a["t0"]) + float(a["dur"])) * SR)].copy()

# 440Hz 正弦，长度对齐
S = np.sin(2 * np.pi * 440 * np.arange(len(A)) / SR).astype(np.float32) * 0.1

items = [("A 本句", A), ("A+噪声(30dB)", A2), ("A 原始混音", A_orig),
         ("B 另一句(#6)", B), ("白噪声", N), ("440Hz正弦", S)]

E = {k: l2norm(emb.from_wave(v)) for k, v in items}

print(f"{'':<16}" + "".join(f"{k[:10]:>12}" for k in E))
for k1 in E:
    print(f"{k1:<16}" + "".join(f"{E[k1] @ E[k2]:>12.3f}" for k2 in E))

print("\n嵌入范数（L2 归一化前）:")
for k, v in items:
    print(f"  {k:<16} ||e||={np.linalg.norm(emb.from_wave(v)):.3f}  "
          f"mean={emb.from_wave(v).mean():+.4f}  std={emb.from_wave(v).std():.4f}")
