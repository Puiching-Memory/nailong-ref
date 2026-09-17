"""滑窗分块提取嵌入，用于检测"一句里混了多个说话人"。

动机：80 句全部余弦挤在 0.45 附近、集内集外都无簇结构，指向 VAD 切句
把相邻不同说话人的台词并成了一段（export_utts.py 的 GAP_TOL=0.10 太松）。
分块后，"混合句"的内部小段会互相不像，可直接量化。
"""
from __future__ import annotations

import sys

import numpy as np

from .. import audio, config, manifests
from ..speakers import RedimNetEmbedder, l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = config.SR_MODEL
WIN, HOP = 1.5, 0.5
MIN_CHUNK = 0.8

emb = RedimNetEmbedder()
print(f"ReDimNet 已加载 -> {emb.dev}；窗长 {WIN}s 步长 {HOP}s", flush=True)

_cache: dict[str, np.ndarray] = {}


def load(src: str) -> np.ndarray:
    if src not in _cache:
        _cache[src] = audio.decode(config.vocals_path(src), SR)
    return _cache[src]


def embed(seg):
    return emb.from_wave(seg)


E, META = [], []
for r in manifests.read(config.UTT_MANIFEST):
    x = load(r["src"])
    t0, dur = float(r["t0"]), float(r["dur"])
    starts = (np.arange(0, dur - WIN + 1e-9, HOP) if dur >= WIN else np.array([0.0]))
    if dur < MIN_CHUNK:
        continue
    for ci, st in enumerate(starts):
        a = int((t0 + st) * SR)
        b = int((t0 + min(st + WIN, dur)) * SR)
        if (b - a) / SR < MIN_CHUNK:
            continue
        E.append(embed(x[a:b]))
        META.append(dict(idx=int(r["idx"]), src=r["src"], t0=t0, dur=dur,
                         ci=ci, ct0=round(t0 + st, 3),
                         cdur=round((b - a) / SR, 3), text=""))

E = np.array(E)
np.save(config.EMB_CHUNKS, E)
svg = manifests.texts()
manifests.write(config.CHUNK_META,
                ["idx", "src", "t0", "dur", "ci", "ct0", "cdur", "text"],
                [[m["idx"], m["src"], f"{m['t0']:.2f}", f"{m['dur']:.2f}", m["ci"],
                  f"{m['ct0']:.3f}", f"{m['cdur']:.3f}", svg.get(m["idx"], "")]
                 for m in META])

n_utt = len({m["idx"] for m in META})
print(f"{len(META)} 个分块，来自 {n_utt} 句  -> "
      f"{config.rel(config.EMB_CHUNKS)} / {config.rel(config.CHUNK_META)}")

En = l2norm(E)
by = {}
for i, m in enumerate(META):
    by.setdefault(m["idx"], []).append(i)

print("\n各句内部分块一致性（低=该句可能混了多个说话人）:")
rows = []
for idx, ids in by.items():
    if len(ids) < 2:
        continue
    S = En[ids] @ En[ids].T
    iu = np.triu_indices(len(ids), 1)
    rows.append((float(S[iu].mean()), float(S[iu].min()), idx, len(ids),
                 META[ids[0]]["src"], META[ids[0]]["dur"]))
rows.sort()
for mean_c, min_c, idx, k, src, dur in rows:
    flag = "  ← 疑似混合" if mean_c < 0.35 else ""
    print(f"  #{idx:>3} {src} {dur:>5.2f}s {k}块  内部均值={mean_c:.3f} "
          f"最小={min_c:.3f}{flag}   {svg.get(idx, '')[:30]}")
print(f"\n多块句共 {len(rows)} 句；内部均值 <0.35 的有 "
      f"{sum(1 for r in rows if r[0] < 0.35)} 句")
