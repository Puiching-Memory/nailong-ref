"""人工标注 + 主动学习的奶龙识别系统。

思路：不再依赖固定阈值（你说得对，加阈值没用），改为让模型从人工标注里
学决策边界。scikit-activeml 每轮挑出**最有信息量**的样本给你听，你标一次，
模型就更准一次；已标过的样本永远不会再问你。

标签分两档：
  human —— 你亲口判定的，最高权威，永远覆盖 weak
  weak  —— 自动方法给的弱标签，只用于冷启动，会被你的标注逐步纠正

用法::

    nailong annotate seed          初始化（human 种子 + weak 弱标签）
    nailong annotate status        看当前模型状态与人机一致度
    nailong annotate query [N]     挑 N 句最该由你判定，导出试听带
    nailong annotate teach 12:1 47:0   录入判定（1=奶龙, 0=不是, 2=有音效不能用）
"""
import csv
import sys

import numpy as np
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import CoreSet, UncertaintySampling
from sklearn.linear_model import LogisticRegression

from .. import audio, config, manifests
from ..speakers import l2norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = config.SR_EXPORT
LABELS_CSV = config.LABELS
REEL = config.REELS / "q_reel.wav"
MISSING = -1
CAND = 40                          # 先按不确定性取候选池，再用 CoreSet 在其中挑多样性样本
WEAK_K = config.WEAK_K             # 自动方法在两端各取多少句作为弱标签冷启动
MIN_DUR = config.MIN_ASK_DUR       # 只提问长于此的句子：更短的人耳也分不出说话人
# 你已亲口判定的种子（#15 你确认是奶龙；#2 你判定错误说话人）
HUMAN_SEED = config.HUMAN_SEED

rows = manifests.by_idx(config.UTT_MANIFEST)
svr = manifests.by_idx(config.SV_ALL)
tp = manifests.by_idx(config.TWO_PASS)
IDS = sorted(rows)
POS = {i: k for k, i in enumerate(IDS)}
X = l2norm(np.load(config.EMB_REDIMNET))
margin = {i: float(tp[i]["simA"]) - float(tp[i]["simB"]) for i in IDS}


def load_labels():
    lab, src = {}, {}
    if LABELS_CSV.exists():
        for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")):
            i = int(r["idx"])
            if src.get(i) == "human" and r["source"] != "human":
                continue          # human 优先
            lab[i], src[i] = int(r["label"]), r["source"]
    return lab, src


def save_labels(lab, src):
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "label", "source"])
        for i in sorted(lab):
            w.writerow([i, lab[i], src[i]])


def y_vector(lab):
    # 2 = 音效污染，是独立的排除规则，不进说话人二分类
    return np.array([lab[i] if lab.get(i) in (0, 1) else MISSING for i in IDS])


def make_clf():
    return SklearnClassifier(LogisticRegression(max_iter=3000, C=1.0,
                                                class_weight="balanced"),
                             classes=np.array([0, 1]), missing_label=MISSING,
                             random_state=0)


def cmd_seed():
    lab, src = load_labels()
    for i, v in HUMAN_SEED.items():
        lab[i], src[i] = v, "human"
    order = sorted(IDS, key=lambda i: -margin[i])
    for i in order[:WEAK_K]:
        if i not in lab:
            lab[i], src[i] = 1, "weak"
    for i in order[-WEAK_K:]:
        if i not in lab:
            lab[i], src[i] = 0, "weak"
    save_labels(lab, src)
    nh = sum(1 for v in src.values() if v == "human")
    print(f"已初始化 {LABELS_CSV}: human {nh} 条, weak {len(lab) - nh} 条, "
          f"未标注 {len(IDS) - len(lab)} 条")


def cmd_status():
    lab, src = load_labels()
    y = y_vector(lab)
    nh = sum(1 for v in src.values() if v == "human")
    nsfx = sum(1 for i, v in lab.items() if v == 2)
    print(f"标注进度: human {nh} / weak {len(lab) - nh} / 未标 {len(IDS) - len(lab)}"
          f"  (共 {len(IDS)})" + (f"  音效污染 {nsfx} 句" if nsfx else ""))
    if (y != MISSING).sum() < 4:
        print("标注太少，无法训练。先跑 query 再 teach。")
        return lab
    clf = make_clf().fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    # 人机一致度：只看 human，模型不知道这些标签时是否也认同
    hs = [POS[i] for i, s in src.items() if s == "human"]
    agree = sum(1 for k in hs if (p[k] > 0.5) == (lab[IDS[k]] == 1))
    print(f"人机一致度 (仅看 human 标注): {agree}/{len(hs)}")
    pred = (p > 0.5).astype(int)
    print(f"预测为奶龙: {pred.sum()} 句")
    return lab, src, clf, p


def cmd_query(n=12):
    lab, src = load_labels()
    y = y_vector(lab)
    if (y != MISSING).sum() < 4:
        print("标注太少，无法启动主动学习。请先跑 seed。")
        return
    clf = make_clf().fit(X, y)
    unlabeled = np.array([POS[i] for i in IDS
                          if (lab.get(i) is None or src.get(i) != "human")
                          and float(rows[i]["dur"]) >= MIN_DUR])
    if len(unlabeled) == 0:
        print("没有待判定的句子了。")
        return
    nc = min(CAND, len(unlabeled))
    cand = UncertaintySampling(method="margin_sampling", missing_label=MISSING).query(
        X=X, y=y, clf=clf, candidates=unlabeled, batch_size=nc)
    pick = CoreSet(metric="euclidean", missing_label=MISSING).query(
        X=X, y=y, candidates=cand, batch_size=min(n, len(cand)))
    p = clf.predict_proba(X)[:, 1]

    config.ensure_dirs()
    for f in config.QUERIES.iterdir():
        f.unlink()
    import soundfile as sf
    parts, order = [], []
    dec = audio.Decoder(SR, dtype="float64")

    # 故意不显示模型预测 —— 避免锚定偏误，保证你的判定是独立真值
    print(f"\n{'序':>3}{'句号':>6}{'源':>9}{'时长':>7}  台词")
    for k, j in enumerate(pick, 1):
        i = IDS[j]
        r = rows[i]
        seg = dec.clip(r["src"], float(r["t0"]), float(r["dur"]))
        f = config.QUERIES / f"{k:02d}_{i:03d}.wav"
        audio.write_wav(f, seg, SR, highpass=config.HIGHPASS_HZ)
        yy, sr = sf.read(f, dtype="float32")
        parts += [yy, np.zeros(int(1.0 * sr), dtype="float32")]
        print(f"{k:>3}{i:>6}{r['src']:>9}{float(r['dur']):>7.2f}  {svr[i]['text'][:30]}")
        order.append((k, i))
    manifests.write(config.Q_BATCH,
                    ["order", "idx", "src", "dur", "p_nailong", "text"],
                    [[k, i, rows[i]["src"], rows[i]["dur"],
                      f"{p[POS[i]]:.3f}", svr[i]["text"]] for k, i in order])
    sf.write(str(REEL), np.concatenate(parts), sr)
    print(f"\n{config.rel(REEL)} 已生成（{len(pick)} 段）。按序号回报，例如: "
          f"teach " + " ".join(f"{i}:1" for _, i in order[:2]) + " ...")


def cmd_teach(args):
    lab, src = load_labels()
    n = 0
    for a in args:
        i, v = a.split(":")
        lab[int(i)], src[int(i)] = int(v), "human"
        n += 1
    save_labels(lab, src)
    print(f"已录入 {n} 条人工标注")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "seed":
        cmd_seed()
    elif cmd == "status":
        cmd_status()
    elif cmd == "query":
        cmd_query(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif cmd == "teach":
        cmd_teach(sys.argv[2:])
    else:
        print(__doc__)
