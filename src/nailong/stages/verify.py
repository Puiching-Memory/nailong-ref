"""验证组A 是否是奶龙。

三条互相独立的证据：
1) 文本证据：出现"奶龙/小七"的句子在两组间的分布（与声学方法完全独立）
2) 声学证据：组A 内部一致性 vs 与组B 的分离度
3) 结构检查：组A 内部会不会再裂成两个角色
"""
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .. import config, manifests
from ..speakers import l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = manifests.by_idx(config.UTT_MANIFEST)
txt = manifests.texts()
res = manifests.read(config.TWO_PASS)
grp = {int(r["idx"]): r["group"] for r in res}
sa = {int(r["idx"]): float(r["simA"]) for r in res}
sb = {int(r["idx"]): float(r["simB"]) for r in res}
idxA = sorted(i for i in grp if grp[i] == "A")
idxB = sorted(i for i in grp if grp[i] == "B")

print("=" * 88)
print("证据 1 — 文本证据（与声学完全独立）")
print("=" * 88)
for kw in ("奶龙", "小七", "暴暴"):
    hit = [i for i in sorted(grp) if kw in txt[i]]
    a = [i for i in hit if grp[i] == "A"]
    b = [i for i in hit if grp[i] == "B"]
    print(f"\n出现「{kw}」共 {len(hit)} 句 —— 组A {len(a)} 句 / 组B {len(b)} 句")
    for i in hit:
        tag = "A" if grp[i] == "A" else "B"
        print(f"   [{tag}] #{i:>3} A={sa[i]:.3f} B={sb[i]:.3f}  {txt[i][:36]}")

print(f"\n{'=' * 88}")
print("证据 2 — 声学一致性")
print("=" * 88)
E = np.load(config.EMB_REDIMNET)
En = l2norm(E)
iA = [int(r["idx"]) - 1 for r in res if r["group"] == "A"]
iB = [int(r["idx"]) - 1 for r in res if r["group"] == "B"]
# utt_manifest 顺序即 idx-1，确认一下
order = manifests.read(config.UTT_MANIFEST)
pos_of = {int(r["idx"]): k for k, r in enumerate(order)}
iA = [pos_of[int(r["idx"])] for r in res if r["group"] == "A"]
iB = [pos_of[int(r["idx"])] for r in res if r["group"] == "B"]


def mean_cos(a, b):
    M = En[a] @ En[b].T
    return float(M.mean())


print(f"  组A 内部({len(iA)}×{len(iA)})  平均余弦 = {mean_cos(iA, iA):.3f}")
print(f"  组B 内部({len(iB)}×{len(iB)})  平均余弦 = {mean_cos(iB, iB):.3f}")
print(f"  A 与 B 之间        平均余弦 = {mean_cos(iA, iB):.3f}")
print(f"  → 分离度 = 组内均值 - 组间 = {0.5 * (mean_cos(iA, iA) + mean_cos(iB, iB)) - mean_cos(iA, iB):+.3f}")

print(f"\n{'=' * 88}")
print("证据 3 — 组A 内部是否还会裂成两个角色")
print("=" * 88)
XA = En[iA]
for k in (2, 3):
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                  linkage="average").fit_predict(XA)
    sil = silhouette_score(XA, lab, metric="cosine")
    sizes = sorted(np.bincount(lab), reverse=True)
    print(f"  k={k}  silhouette={sil:.3f}  簇大小={sizes}")
lab2 = AgglomerativeClustering(n_clusters=2, metric="cosine",
                               linkage="average").fit_predict(XA)
print("\n  若 k=2 分裂，看两半的台词分布（哪半更像奶龙）:")
for c in (0, 1):
    mem = [iA[j] for j in range(len(iA)) if lab2[j] == c]
    ids = [order[j]["idx"] for j in mem]
    tot = sum(float(order[j]["dur"]) for j in mem)
    print(f"\n  [A-{c}] {len(mem)} 句 {tot:.1f}s")
    for j in sorted(mem, key=lambda j: -float(order[j]["dur"]))[:5]:
        print(f"     #{order[j]['idx']:>3} {float(order[j]['dur']):>5.2f}s  "
              f"{txt[int(order[j]['idx'])][:34]}")
