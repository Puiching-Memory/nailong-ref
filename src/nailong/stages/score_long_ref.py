"""用「整集长片段」作参考来给 2 秒短句打分。

关键洞察：8/9 集的 30s 嵌入余弦高达 0.81-0.92（同一主说话人），
而 2s 短句之间只有 0.445。说明短句嵌入噪声大，但 30s 聚合非常可靠。
因此改用可靠的长参考（约 240s 素材）去评判不可靠的短句。

留一法：给 src_X 的句子打分时，参考排除 src_X 本身，避免自身泄漏。
"""
import sys

import numpy as np

from .. import audio, config, manifests
from ..speakers import RedimNetEmbedder, l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
KNOWN_POS = {15}   # 用户确认是奶龙
KNOWN_NEG = {6}    # 用户在试听带上判定为错误的说话人

emb = RedimNetEmbedder()
EP = {s: emb.from_wave(audio.decode(config.vocals_path(s), config.SR_MODEL))
      for s in config.SRCS}
print("整集嵌入完成", flush=True)
Ep = {s: l2norm(v) for s, v in EP.items()}

rows = manifests.read(config.UTT_MANIFEST)
txt = manifests.texts()
En = l2norm(np.load(config.EMB_REDIMNET))

pos_s, neg_s = [], []
for i, r in enumerate(rows):
    s = r["src"]
    pos_pool = [q for q in config.SRCS if q != s and q != config.OUTLIER]
    neg_pool = ([config.OUTLIER] if s != config.OUTLIER
                else [q for q in config.SRCS if q != s and q != config.OUTLIER])
    rp = l2norm(np.mean([Ep[q] for q in pos_pool], axis=0))
    rn = l2norm(np.mean([Ep[q] for q in neg_pool], axis=0))
    pos_s.append(En[i] @ rp)
    neg_s.append(En[i] @ rn)
pos_s, neg_s = np.array(pos_s), np.array(neg_s)
margin = pos_s - neg_s

print(f"\n{'=' * 84}")
print("A. 内建验证：src_02 的句子（确认的另一人）应排在底部")
print("=" * 84)
s02 = np.array([r["src"] == config.OUTLIER for r in rows])
print(f"  src_02 的 {s02.sum()} 句   正参考得分 均值={pos_s[s02].mean():.3f} "
      f"范围 {pos_s[s02].min():.3f}~{pos_s[s02].max():.3f}")
print(f"  其余 {len(rows) - s02.sum()} 句  正参考得分 均值={pos_s[~s02].mean():.3f} "
      f"范围 {pos_s[~s02].min():.3f}~{pos_s[~s02].max():.3f}")
rank = np.argsort(-margin)
r02 = [int(np.where(rank == i)[0][0]) + 1 for i in np.where(s02)[0]]
print(f"  src_02 各句在 {len(rows)} 句中的排名: {sorted(r02)}")
print("  （若方法有效，这些排名应集中在末段）")

print(f"\n{'=' * 84}")
print("B. 已知标签检验")
print("=" * 84)
for tag, ks in (("正（确认是奶龙）", KNOWN_POS), ("负（确认非奶龙）", KNOWN_NEG)):
    for i, r in enumerate(rows):
        if int(r["idx"]) in ks:
            p = int(np.where(rank == i)[0][0]) + 1
            print(f"  {tag} #{int(r['idx']):>3}  正={pos_s[i]:.3f} 负={neg_s[i]:.3f} "
                  f"余量={margin[i]:+.3f}  排名 {p}/{len(rows)}   {txt[int(r['idx'])][:24]}")
print(f"\n  全体余量: p50={np.percentile(margin, 50):+.3f} "
      f"p75={np.percentile(margin, 75):+.3f} p90={np.percentile(margin, 90):+.3f}")

print(f"\n{'=' * 84}")
print("C. 按余量排序（前 30）")
print("=" * 84)
print(f"{'排名':>4}{'#':>5}{'源':>9}{'起点':>7}{'时长':>7}{'正':>7}{'负':>7}{'余量':>8}  台词")
for j, i in enumerate(rank[:30], 1):
    r = rows[i]
    mark = " ★" if int(r["idx"]) in KNOWN_POS else (" ✗" if int(r["idx"]) in KNOWN_NEG else "")
    print(f"{j:>4}{int(r['idx']):>5}{r['src']:>9}{float(r['t0']):>7.2f}"
          f"{float(r['dur']):>7.2f}{pos_s[i]:>7.3f}{neg_s[i]:>7.3f}{margin[i]:>+8.3f}"
          f"  {txt[int(r['idx'])][:28]}{mark}")

manifests.write(config.MARGIN_LONG,
                ["idx", "src", "t0", "dur", "sim_pos", "sim_neg", "margin"],
                [[rows[i]["idx"], rows[i]["src"], rows[i]["t0"], rows[i]["dur"],
                  manifests.f4(pos_s[i]), manifests.f4(neg_s[i]), manifests.f4(margin[i])]
                 for i in rank])
print(f"\n-> {config.rel(config.MARGIN_LONG)}")
