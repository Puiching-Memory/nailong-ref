"""两遍迭代：用长参考初筛，再用入围句子自建参考细化，得到干净的说话人划分。

第 1 遍：以"整集 30s 嵌入"为主角参考（已验证能把 src_02 那 3 句排到末 3 位）。
第 2 遍：把高度匹配的句子聚成参考 A、高度不匹配的聚成参考 B，
        再用 A/B 重新打分并迭代——这是标准的 EM 式两说话人精化。
"""
import sys

import numpy as np

from .. import audio, config, manifests
from ..speakers import RedimNetEmbedder, l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROUNDS = config.ROUNDS
STRONG_POS, STRONG_NEG = 30, 25   # 两端各取多少句做初始参考

emb = RedimNetEmbedder()
EP = {s: emb.from_wave(audio.decode(config.vocals_path(s), config.SR_MODEL))
      for s in config.SRCS}
np.save(config.EMB_EPISODES, np.array([EP[s] for s in config.SRCS]))
Ep = {s: l2norm(v) for s, v in EP.items()}
print(f"整集嵌入 -> {config.rel(config.EMB_EPISODES)}", flush=True)

rows = manifests.read(config.UTT_MANIFEST)
txt = manifests.texts()
En = l2norm(np.load(config.EMB_REDIMNET))
srcs = [r["src"] for r in rows]
n = len(rows)


def norm(v):
    return l2norm(v)


# 第 1 遍参考：留一法，排除本集与已知的离群集
ep_pos = np.array([norm(np.mean([Ep[q] for q in config.SRCS
                                 if q != srcs[i] and q != config.OUTLIER], axis=0))
                   for i in range(n)])
sp_ep = En @ ep_pos.T
sp_ep = np.array([sp_ep[i, i] for i in range(n)])

# 迭代参考：初始用长参考判定，之后自建
order = np.argsort(-sp_ep)
pool = [i for i in order if srcs[i] != config.OUTLIER]
strong_pos = pool[:STRONG_POS]
strong_neg = pool[-STRONG_NEG:] + [i for i in range(n) if srcs[i] == config.OUTLIER]
refA, refB = norm(np.mean(En[strong_pos], axis=0)), norm(np.mean(En[strong_neg], axis=0))

print(f"\n{'=' * 88}")
print("迭代细化")
print("=" * 88)
for it in range(1, ROUNDS + 1):
    sa, sb = En @ refA, En @ refB
    mg = sa - sb
    a = np.where(mg > 0)[0]
    b = np.where(mg <= 0)[0]
    print(f"\n第 {it} 遍: A={len(a)} 句  B={len(b)} 句   "
          f"{config.OUTLIER} 落在 B 的有 {sum(1 for i in b if srcs[i] == config.OUTLIER)}/3")
    if it < ROUNDS:
        # 只取高置信样本重建参考，避免边界句污染
        hi_a = np.argsort(-mg)[:max(10, len(a) // 2)]
        hi_b = np.argsort(mg)[:max(10, len(b) // 2)]
        refA, refB = norm(np.mean(En[hi_a], axis=0)), norm(np.mean(En[hi_b], axis=0))

sa, sb = En @ refA, En @ refB
mg = sa - sb
rank = np.argsort(-mg)

print(f"\n{'=' * 88}")
print("最终划分")
print("=" * 88)
for name, idxs in (("A", np.where(mg > 0)[0]), ("B", np.where(mg <= 0)[0])):
    tot = sum(float(rows[i]["dur"]) for i in idxs)
    print(f"\n[组 {name}]  {len(idxs)} 句  {tot:.1f}s")
    for i in sorted(idxs, key=lambda i: -float(rows[i]["dur"]))[:12]:
        r = rows[i]
        print(f"   #{int(r['idx']):>3} {r['src']} {float(r['t0']):>6.2f}s "
              f"{float(r['dur']):>5.2f}s  A={sa[i]:.3f} B={sb[i]:.3f} "
              f"差={mg[i]:+.3f}  {txt[int(r['idx'])][:26]}")

print(f"\n{'=' * 88}")
print("各组来源分布（看是否与集强相关）")
print("=" * 88)
for name, idxs in (("A", np.where(mg > 0)[0]), ("B", np.where(mg <= 0)[0])):
    from collections import Counter
    c = Counter(srcs[i] for i in idxs)
    print(f"  组{name}: " + "  ".join(f"{k}×{v}" for k, v in sorted(c.items())))

manifests.write(config.TWO_PASS,
                ["idx", "src", "t0", "dur", "simA", "simB", "margin", "group"],
                [[rows[i]["idx"], rows[i]["src"], rows[i]["t0"], rows[i]["dur"],
                  manifests.f4(sa[i]), manifests.f4(sb[i]), manifests.f4(mg[i]),
                  "A" if mg[i] > 0 else "B"] for i in rank])
print(f"\n-> {config.rel(config.TWO_PASS)}")
