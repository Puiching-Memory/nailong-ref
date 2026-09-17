"""用任意 FunASR 声纹模型提取全部句子的嵌入，便于横向对比模型强弱。

FunASR 只注册了 CAMPPlus / ERes2NetV2；要更强的模型走 ReDimNet（见 embed-redimnet）。

用法::

    nailong embed-funasr <model_id> <out.npy> [orig]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import audio, config, manifests, speakers

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = config.SR_MODEL
PAD = int(config.SEG_PAD * SR)
TMP = config.DATA / "_tmp_seg.wav"

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
model_id, out = sys.argv[1], Path(sys.argv[2])
use_orig = len(sys.argv) > 3 and sys.argv[3] == "orig"
sv = speakers.load_funasr(model_id)
print(f"已加载 {model_id}；"
      f"声源={'原始混音' if use_orig else 'demucs 人声轨'}", flush=True)

rows = manifests.read(config.UTT_MANIFEST)
cache: dict[str, np.ndarray] = {}
E = []
for r in rows:
    src = r["src"]
    if src not in cache:
        cache[src] = audio.decode(
            config.mix_path(src) if use_orig else config.vocals_path(src),
            SR, dtype="float64")
    x = cache[src]
    i0, i1 = int(float(r["t0"]) * SR), int(float(r["t1"]) * SR)
    seg = x[max(0, i0 - PAD):min(len(x), i1 + PAD)]
    # FunASR 只接受文件路径，落到 data/ 下的临时文件，下一句覆盖即可
    audio.write_wav(TMP, seg, SR)
    E.append(speakers.embed_funasr(sv, TMP))

E = np.array(E)
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, E)
print(f"嵌入 {E.shape} -> {config.rel(out)}", flush=True)

oi = {int(r["idx"]): i for i, r in enumerate(rows)}
En = speakers.l2norm(E)
print("原锚点两两余弦: " + speakers.anchor_cosines(En, oi))
A = config.ANCHORS
pair = [En[oi[a]] @ En[oi[b]] for i, a in enumerate(A) for b in A[i + 1:]]
print(f"锚点整体均值: {np.mean(pair):.3f}   (cam++=0.556, ERes2NetV2=0.615)")
