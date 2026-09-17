"""无监督说话人聚类：不预设锚点，直接按音色自动分组。

人工锚点已失效——4 个"奶龙锚点"两两余弦仅 0.42-0.70（同一人通常 >0.70），
说明锚点集本身就跨了多个说话人。改为让数据自己分组，再由人指定哪簇是奶龙。
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .. import config, manifests

EMB = Path(sys.argv[1]) if len(sys.argv) > 1 else config.EMB_REDIMNET
LOG = config.MANIFESTS / f"{EMB.stem}.log"
CLUSTERS = config.MANIFESTS / f"utt_clusters_{EMB.stem.removeprefix('emb_')}.csv"
config.ensure_dirs()
_log = open(LOG, "w", encoding="utf-8", buffering=1)
_orig_out = sys.stdout


class _Tee:
    """控制台是 GBK，韩文等字符会崩；报告另存 UTF-8。"""

    def write(self, s):
        try:
            _orig_out.write(s)
        except UnicodeEncodeError:
            _orig_out.write(s.encode("gbk", "replace").decode("gbk"))
        _log.write(s)

    def flush(self):
        _orig_out.flush()
        _log.flush()


sys.stdout = _Tee()
OLD_ANCHORS = set(config.ANCHORS)
order = manifests.read(config.UTT_MANIFEST)
meta = manifests.by_idx(config.SV_ALL)
E = np.load(EMB)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9


def sweep(X, tag):
    print(f"\n--- {tag}  (n={len(X)}) ---")
    best = None
    for link in ("average", "complete"):
        row = []
        for k in range(2, 11):
            lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                          linkage=link).fit_predict(X)
            s = silhouette_score(X, lab, metric="cosine")
            row.append(f"k={k}:{s:.2f}")
            if best is None or s > best[1]:
                best = (k, s, lab, link)
        print(f"  {link:>8}  " + "  ".join(row))
    print(f"  → 最佳 k={best[0]} silhouette={best[1]:.3f} linkage={best[3]}")
    return best


long_mask = np.array([float(r["dur"]) >= 1.5 for r in order])
print(f"全部 {len(E)} 句；其中时长 >=1.5s 的 {long_mask.sum()} 句")

b_all = sweep(E, "全部句子 · 原始嵌入")
b_long = sweep(E[long_mask], "仅 >=1.5s · 原始嵌入")
Ec = E - E.mean(axis=0, keepdims=True)
Ec /= np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-9
b_cen = sweep(Ec, "全部句子 · 去均值中心化")
b_long_cen = sweep(Ec[long_mask], "仅 >=1.5s · 去均值中心化")

k, s, lab, link = max([b_all, b_long, b_cen, b_long_cen], key=lambda b: b[1])
print(f"\n{'=' * 70}\n采用: k={k}  silhouette={s:.3f}  linkage={link}")
print(f"{'=' * 70}")

used = Ec if (k, s, lab, link) in (b_cen, b_long_cen) else E
if len(lab) != len(order):
    used = Ec if used is Ec else E
groups = {}
if len(lab) == len(order):
    idxs = list(range(len(order)))
    for i, c in zip(idxs, lab, strict=True):
        groups.setdefault(int(c), []).append(i)
else:
    # 只在长句上聚类，短句按最近簇心归入
    cents = np.vstack([used[long_mask][lab == c].mean(axis=0) for c in range(k)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    assign = np.argmax(used @ cents.T, axis=1)
    for i, c in enumerate(assign):
        groups.setdefault(int(c), []).append(i)

print(f"{'簇':>3}{'句数':>5}{'总时长':>8}{'内部均值余弦':>13}  成员")
rows_out = []
for c, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    M = used[members]
    cen = M.mean(axis=0)
    cen /= np.linalg.norm(cen) + 1e-9
    sims = M @ cen
    tot = sum(float(order[i]["dur"]) for i in members)
    marks = [f"#{order[i]['idx']}" + ("*" if int(order[i]["idx"]) in OLD_ANCHORS else "")
             for i in members]
    print(f"{c:>3}{len(members):>5}{tot:>8.1f}s{sims.mean():>13.3f}  " + " ".join(marks))
    for i in members:
        r = order[i]
        rows_out.append([order[i]["idx"], r["src"], r["t0"], r["t1"], r["dur"], c,
                         f"{sims[list(members).index(i)]:.4f}"])

print("\n(* = 原人工锚点)")
print("\n各簇样本（用于人工确认音色）:")
for c, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print(f"\n[簇 {c}]  {len(members)} 句")
    for i in sorted(members, key=lambda j: -float(order[j]["dur"]))[:6]:
        r = order[i]
        t = meta[int(r["idx"])]["text"]
        print(f"   #{int(r['idx']):>3} {r['src']} {float(r['t0']):>6.2f}s "
              f"{float(r['dur']):>5.2f}s  {t[:30]}")

manifests.write(CLUSTERS,
                ["idx", "src", "t0", "t1", "dur", "cluster", "sim_to_centroid"],
                rows_out)
print(f"\n-> {config.rel(CLUSTERS)}")
