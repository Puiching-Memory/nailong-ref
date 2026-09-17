"""视频美学量化分析:运动信号、音频同步、色彩、边缘密度、镜头统计"""
import numpy as np, wave, json, os, re

OUT = r"C:\workspace\nailong-ref\vid-analysis"
VIDS = {
    "A": ("7cUPc1Huj_gSatel", 160, 160, 20.48),
    "B": ("dc_zXDu0SVyq-QPG", 160, 90, 36.096),
    "C": ("KLY8de3GLjofIDdS", 160, 160, 27.0507),
}
FPS = 24

def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes(); sr = w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr

def stft_mag(x, n_fft=1024, hop=512):
    win = np.hanning(n_fft)
    frames = [x[i:i+n_fft] for i in range(0, len(x)-n_fft, hop)]
    return np.abs(np.array([np.fft.rfft(f*win) for f in frames])), hop

def onset_env(x, sr):
    S, hop = stft_mag(x)
    logS = np.log1p(1000*S)
    flux = np.maximum(0, np.diff(logS, axis=0)).sum(axis=1)
    flux = flux - flux.mean(); flux[flux < 0] = 0
    t = np.arange(len(flux)) * hop / sr
    return flux, t, hop/sr

def rms_env(x, sr, hop=512):
    n = len(x)//hop
    e = np.sqrt((x[:n*hop].reshape(n, hop)**2).mean(axis=1))
    return e, np.arange(n)*hop/sr

def tempo_beats(flux, dt, bmin=60, bmax=190):
    f = flux - flux.mean()
    ac = np.correlate(f, f, "full")[len(f)-1:]
    lags = np.arange(1, len(ac))
    ac_l = ac[1:]
    bpm = 60.0/(lags*dt)
    mask = (bpm >= bmin) & (bpm <= bmax)
    best = lags[mask][np.argmax(ac_l[mask])]
    period = best*dt
    # 梳状滤波找相位
    dur = len(flux)*dt
    best_phase, best_score = 0, -1
    for ph in np.linspace(0, period, 24, endpoint=False):
        times = np.arange(ph, dur, period)
        idx = (times/dt).astype(int); idx = idx[idx < len(flux)]
        s = flux[idx].sum()
        if s > best_score: best_score, best_phase = s, ph
    beats = np.arange(best_phase, dur, period)
    return 60.0/period, beats

def norm01(v):
    v = v.astype(np.float64)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

def kmeans(X, k=8, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :])**2).sum(-1)
        lab = d.argmin(1)
        for i in range(k):
            if (lab == i).any(): C[i] = X[lab == i].mean(0)
    cnt = np.bincount(lab, minlength=k)
    order = np.argsort(-cnt)
    return C[order], cnt[order]/len(X)

report = {}
for key, (name, w, h, dur) in VIDS.items():
    g = np.fromfile(os.path.join(OUT, f"gray_{name}.raw"), dtype=np.uint8)
    nf = len(g)//(w*h); g = g[:nf*w*h].reshape(nf, h, w).astype(np.float32)
    t_v = np.arange(nf)/FPS
    # 运动信号: 帧间平均绝对差
    motion = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    t_m = (t_v[1:] + t_v[:-1])/2
    # 边缘密度 (Sobel 平均幅值), 每 2 帧取 1
    gx = np.abs(np.diff(g[::2], axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(g[::2], axis=1)).mean(axis=(1, 2))
    edge = (gx + gy)/2; t_e = t_v[::2]
    # 近黑/近白帧占比
    fmean = g.mean(axis=(1, 2))
    black_frac = (fmean < 14).mean(); white_frac = (fmean > 242).mean()
    # 音频
    x, sr = read_wav(os.path.join(OUT, f"audio_{name}.wav"))
    flux, t_f, dt = onset_env(x, sr)
    rms, t_r = rms_env(x, sr)
    bpm, beats = tempo_beats(flux, dt)
    # 同步: 运动包络 vs onset 包络互相关 (重采样到 24Hz)
    onset24 = np.interp(t_m, t_f, norm01(flux))
    m_n = norm01(motion)
    max_lag = int(1.0*FPS)
    cors = []
    for lag in range(-max_lag, max_lag+1):
        a = m_n[max(0, lag):len(m_n)+min(0, lag)]
        b = onset24[max(0, -lag):len(onset24)+min(0, -lag)]
        if len(a) > 24: cors.append(np.corrcoef(a, b)[0, 1])
        else: cors.append(0)
    cors = np.nan_to_num(cors)
    best_lag = (np.argmax(cors)-max_lag)/FPS
    # 镜头统计
    cuts = [float(l) for l in open(os.path.join(OUT, f"scenes_{name}.txt")) if l.strip()]
    # 合并间隔 <0.12s 的重复触发
    merged = []
    for c in cuts:
        if merged and c - merged[-1] < 0.12: continue
        merged.append(c)
    bounds = [0]+merged+[dur]
    durs = np.diff(bounds)
    # 切点落在节拍上的比例 (±0.12s)
    if len(beats):
        hit = sum(1 for c in merged if np.min(np.abs(beats-c)) <= 0.12)/len(merged)
    else: hit = 0
    # 色彩 (6fps RGB)
    rgb = np.fromfile(os.path.join(OUT, f"rgb_{name}.raw"), dtype=np.uint8)
    nc = len(rgb)//(w*h*3); rgb = rgb[:nc*w*h*3].reshape(nc, h, w, 3)
    pix = rgb[::3, ::4, ::4, :].reshape(-1, 3).astype(np.float32)
    centers, weights = kmeans(pix, k=8)
    mx = pix.max(1); mn = pix.min(1)
    sat = np.where(mx > 0, (mx-mn)/np.maximum(mx, 1), 0)
    # 逐帧饱和度/亮度曲线 (时间结构)
    f_sat = ((rgb.max(3).astype(np.float32)-rgb.min(3))/np.maximum(rgb.max(3), 1)).mean((1, 2))
    f_val = rgb.mean((1, 2, 3))
    report[key] = dict(
        name=name, dur=dur, n_frames=int(nf),
        motion_mean=float(motion.mean()), motion_p95=float(np.percentile(motion, 95)),
        edge_mean=float(edge.mean()), edge_std=float(edge.std()),
        black_frac=float(black_frac), white_frac=float(white_frac),
        bpm=float(bpm), n_beats=int(len(beats)), sync_corr=float(np.max(cors)),
        sync_lag=float(best_lag), cuts=merged, n_shots=int(len(durs)),
        shot_median=float(np.median(durs)), shot_mean=float(durs.mean()),
        shot_min=float(durs.min()), shot_max=float(durs.max()),
        beat_hit_rate=float(hit),
        palette=[(c.round(1).tolist(), float(wt)) for c, wt in zip(centers, weights)],
        sat_mean=float(sat.mean()), val_mean=float(pix.mean()),
    )
    np.save(os.path.join(OUT, f"motion_{key}.npy"), m_n)
    np.save(os.path.join(OUT, f"onset_{key}.npy"), norm01(onset24))
    np.save(os.path.join(OUT, f"edge_{key}.npy"), norm01(edge))
    np.save(os.path.join(OUT, f"rms_{key}.npy"), norm01(rms))
    np.save(os.path.join(OUT, f"sat_{key}.npy"), f_sat)
    np.save(os.path.join(OUT, f"val_{key}.npy"), f_val)
    np.save(os.path.join(OUT, f"beats_{key}.npy"), beats)

with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
for k, r in report.items():
    print(f"--- {k} {r['name']} ---")
    print(f" shots={r['n_shots']} median={r['shot_median']:.2f}s mean={r['shot_mean']:.2f}s min={r['shot_min']:.2f} max={r['shot_max']:.2f}")
    print(f" bpm={r['bpm']:.1f} beats={r['n_beats']} sync_corr={r['sync_corr']:.3f} lag={r['sync_lag']:+.2f}s beat_hit={r['beat_hit_rate']:.2f}")
    print(f" motion_mean={r['motion_mean']:.2f} edge_mean={r['edge_mean']:.2f} black%={r['black_frac']:.3f} white%={r['white_frac']:.3f} sat={r['sat_mean']:.3f} val={r['val_mean']:.1f}")
    print(" palette:", " ".join(f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}({w:.0%})" for c, w in r["palette"][:8]))
    print(" cuts:", " ".join(f"{c:.2f}" for c in r["cuts"]))
