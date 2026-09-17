"""决定性测试：30 秒级长片段的嵌入是否比 2 秒级可靠得多。

若整集嵌入能分出组，说明问题只在"片段太短"，方向是把素材聚合成
长片段再建模；若整集嵌入也糊成一团，则声纹路线对这批素材不成立。
结论：30s 聚合可靠（同集 0.81-0.92），2s 短句不可靠——这直接催生了
score-long-ref 里「用长参考给短句打分」的做法。
"""
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .. import audio, config
from ..speakers import RedimNetEmbedder, l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = config.SR_MODEL
SRCS = config.SRCS

emb = RedimNetEmbedder()
print(f"ReDimNet -> {emb.dev}\n", flush=True)

full, half1, half2 = {}, {}, {}
for s in SRCS:
    x = audio.decode(config.vocals_path(s), SR)
    full[s] = emb.from_wave(x)
    m = len(x) // 2
    half1[s] = emb.from_wave(x[:m])
    half2[s] = emb.from_wave(x[m:])
    print(f"  {s}  {len(x) / SR:.2f}s", flush=True)


def top(X):
    return l2norm(X)


F = top(np.array([full[s] for s in SRCS]))
H1 = top(np.array([half1[s] for s in SRCS]))
H2 = top(np.array([half2[s] for s in SRCS]))

print(f"\n{'=' * 74}")
print("A. 整集(30s)嵌入两两余弦")
print("=" * 74)
S = F @ F.T
print("      " + "".join(f"{s[-2:]:>7}" for s in SRCS))
for i, s in enumerate(SRCS):
    print(f"{s[-2:]:>5} " + "".join(f"{S[i, j]:>7.2f}" for j in range(len(SRCS))))

iu = np.triu_indices(len(SRCS), 1)
print(f"\n  整集两两余弦: 均值={S[iu].mean():.3f} 中位={np.median(S[iu]):.3f} "
      f"min={S[iu].min():.3f} max={S[iu].max():.3f}")

print(f"\n{'=' * 74}")
print("B. 同一集前后半段 余弦（同说话人的下界参考）")
print("=" * 74)
same = np.array([H1[i] @ H2[i] for i in range(len(SRCS))])
for i, s in enumerate(SRCS):
    print(f"  {s}  前半 vs 后半 = {same[i]:.3f}")
print(f"\n  同集前后半 均值={same.mean():.3f}")
print(f"  对照: 跨集 均值={S[iu].mean():.3f}   差值={same.mean() - S[iu].mean():+.3f}")

print(f"\n{'=' * 74}")
print("C. 整集嵌入聚类")
print("=" * 74)
for k in range(2, 6):
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                  linkage="average").fit_predict(F)
    sil = silhouette_score(F, lab, metric="cosine")
    print(f"  k={k}  silhouette={sil:.3f}  簇=" +
          str([[SRCS[i] for i in range(len(SRCS)) if lab[i] == c] for c in range(k)]))

print(f"\n{'=' * 74}")
print("D. 2s句 vs 30s集 的嵌入稳定性")
print("=" * 74)
print("  2s 句 → 已知: 全体两两余弦中位 0.445, 锚点内部 0.659")
print(f"  30s 集 → 跨集两两余弦中位 {np.median(S[iu]):.3f}, 同集前后半 {same.mean():.3f}")
