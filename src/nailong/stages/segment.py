"""VAD 细粒度切句：导出单句 wav + 试听带 + 清单，供人工判断哪句是奶龙。

判据是「帧 RMS + 语音频段占比」双阈值，前后各留 60ms 保住字头字尾。
GAP_TOL 偏松会把相邻不同说话人的台词并成一句——这是后续声纹聚类
拿不到簇结构的主因之一（用 embed-chunk 可以量化混句程度）。
"""
from __future__ import annotations

import logging

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import EncoderClassifier

from .. import audio, config, manifests
from ..speakers import l2norm

logging.getLogger("speechbrain").setLevel(logging.ERROR)

SR = config.SR_MODEL
FL, HOP = config.FRAME, config.HOP
HOP_T = HOP / SR
GAP_TOL, MIN_UTT = config.GAP_TOL, config.MIN_UTT
OUT = config.UTTERANCES
SP_FMIN, SP_FMAX = 250, 4000      # 语音能量集中区，用来和 BGM/音效区分
CEN_FMIN, CEN_FMAX = 80, 8000     # 谱质心统计带

UTT_FIELDS = ["idx", "src", "t0", "t1", "dur", "f0_med", "centroid", "cluster", "reel_t"]


def feat(x):
    """返回 (帧 dBFS, 语音频段占比, 谱质心)，逐帧。"""
    F = audio.frame_matrix(x, FL, HOP)
    S = audio.spectrum(F)
    f = audio.freqs(FL, SR)
    return (audio.frame_db(F),
            audio.band_ratio(S, f, SP_FMIN, SP_FMAX),
            audio.spectral_centroid(S, f, CEN_FMIN, CEN_FMAX))


def segs(mask):
    out, s, last = [], None, None
    for i, v in enumerate(mask):
        if v:
            if s is None:
                s = i
            last = i
        elif s is not None and (i - last) * HOP_T > GAP_TOL:
            out.append((s, last + 1))
            s = None
    if s is not None:
        out.append((s, last + 1))
    return [(a, b) for a, b in out if (b - a) * HOP_T >= MIN_UTT]


def f0_med(x):
    return audio.f0_median(x, SR, FL, HOP)


config.ensure_dirs()
spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=str(config.PRETRAINED / "ecapa"))

recs = []
for src in config.SRCS:
    x = audio.decode(config.vocals_path(src), SR, dtype="float64")
    db, sp, cen = feat(x)
    for a, b in segs((db > config.DB_THRESH) & (sp > config.SP_THRESH)):
        i0, i1 = a * HOP, b * HOP
        seg = x[i0:i1]
        if len(seg) < int(config.MIN_SEG * SR):
            continue
        pad = int(config.SEG_PAD * SR)
        s = x[max(0, i0 - pad):min(len(x), i1 + pad)]
        with torch.no_grad():
            e = spk.encode_batch(torch.from_numpy(seg).float().unsqueeze(0)).squeeze().numpy()
        recs.append(dict(src=src, a=a * HOP_T, b=b * HOP_T, dur=len(s) / SR,
                         f0=f0_med(seg), cen=float(np.median(cen[a:b])), emb=e, wav=s))

E = l2norm(np.array([r["emb"] for r in recs]))
print(f"共 {len(recs)} 句")
for k in (2, 3, 4):
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(E)
    print(f"  k={k} silhouette={silhouette_score(E, lab, metric='cosine'):.3f} 簇大小={list(np.bincount(lab))}")

lab = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(E)
for r, c in zip(recs, lab, strict=True):
    r["cluster"] = int(c)

GAP_S = 0.6                       # 试听带段间静音
REEL = config.REELS / "utt_reel.wav"

order = sorted(range(len(recs)), key=lambda i: recs[i]["f0"])
gap = np.zeros(int(GAP_S * SR))
reel, marks = [], []
pos = 0.0
print(f"\n{'#':>4}{'源':>9}{'起点':>8}{'时长':>7}{'F0':>6}{'质心':>7}{'簇':>4}{'reel':>8}")
for n, i in enumerate(order, 1):
    r = recs[i]
    r["idx"] = n
    sf.write(str(OUT / f"utt_{n:03d}.wav"), r["wav"], SR)
    marks.append((n, pos))
    reel += [r["wav"], gap]
    pos += len(r["wav"]) / SR + GAP_S
    print(f"{n:>4}{r['src']:>9}{r['a']:>8.2f}{r['dur']:>7.2f}"
          f"{r['f0']:>6.0f}{r['cen']:>7.0f}{r['cluster']:>4}{marks[-1][1]:>8.2f}")

sf.write(str(REEL), np.concatenate(reel), SR)
reel_t = dict(marks)
manifests.write(config.UTT_MANIFEST, UTT_FIELDS, [
    [recs[i]["idx"], recs[i]["src"], manifests.f2(recs[i]["a"]), manifests.f2(recs[i]["b"]),
     manifests.f2(recs[i]["dur"]), f"{recs[i]['f0']:.0f}", f"{recs[i]['cen']:.0f}",
     recs[i]["cluster"], manifests.f2(reel_t[recs[i]["idx"]])]
    for i in order])

print(f"\n单句: {config.rel(OUT)}/utt_NNN.wav   "
      f"试听带: {config.rel(REEL)} ({pos:.1f}s)")
print(f"清单: {config.rel(config.UTT_MANIFEST)}")
