"""用 ReDimNet 提取声纹嵌入，跳出 FunASR。

FunASR 只注册了 CAMPPlus / ERes2NetV2 两个声纹模型，精度上限受限。
ReDimNet (Interspeech 2024) 的 b6 在 Vox1-O 上 EER 0.53%，
并且官方提供 VoxBlink2+VoxCeLeb2+CN-Celeb 混合训练的权重，
是这套素材里最合适的中文强模型。

用法::

    nailong embed-redimnet [out.npy] [orig]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import audio, config, manifests
from ..speakers import RedimNetEmbedder, anchor_cosines, l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = config.SR_MODEL
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else config.EMB_REDIMNET
USE_ORIG = len(sys.argv) > 2 and sys.argv[2] == "orig"

emb = RedimNetEmbedder(device=None)
print(f"ReDimNet(M, ft_mix, vb2+vox2+cnc) 已加载 -> {emb.dev}；"
      f"声源={'原始混音' if USE_ORIG else 'demucs 人声轨'}", flush=True)

_cache: dict[str, np.ndarray] = {}


def load(src: str) -> np.ndarray:
    if src not in _cache:
        path = config.mix_path(src) if USE_ORIG else config.vocals_path(src)
        _cache[src] = audio.decode(path, SR)
    return _cache[src]


rows = manifests.read(config.UTT_MANIFEST)
E = []
for r in rows:
    x = load(r["src"])
    i0, i1 = int(float(r["t0"]) * SR), int(float(r["t1"]) * SR)
    E.append(emb.from_wave(x[i0:i1]))

E = np.array(E)
OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT, E)
print(f"嵌入 {E.shape} -> {config.rel(OUT)}", flush=True)

En = l2norm(E)
oi = {int(r["idx"]): i for i, r in enumerate(rows)}
A = config.ANCHORS
print("原锚点两两余弦: " + anchor_cosines(En, oi))
pair = [En[oi[a]] @ En[oi[b]] for i, a in enumerate(A) for b in A[i + 1:]]
print(f"锚点整体均值: {np.mean(pair):.3f}   "
      f"(cam++=0.556, ERes2NetV2=0.615)")
print(f"全体两两余弦: 中位={np.median(En @ En.T):.3f} "
      f"p10={np.percentile(En @ En.T, 10):.3f} "
      f"p90={np.percentile(En @ En.T, 90):.3f}")
