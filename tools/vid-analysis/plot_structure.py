"""绘制三条视频的时间结构图: 运动/边缘/饱和度/音频RMS + 切点与节拍"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\workspace\nailong-ref\vid-analysis"
rep = json.load(open(os.path.join(OUT, "report.json")))
fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=False)
for ax, key in zip(axes, ["A", "B", "C"]):
    r = rep[key]
    motion = np.load(os.path.join(OUT, f"motion_{key}.npy"))
    onset = np.load(os.path.join(OUT, f"onset_{key}.npy"))
    edge = np.load(os.path.join(OUT, f"edge_{key}.npy"))
    sat = np.load(os.path.join(OUT, f"sat_{key}.npy"))
    dur = r["dur"]
    t_m = np.linspace(0, dur, len(motion))
    t_e = np.linspace(0, dur, len(edge))
    t_s = np.linspace(0, dur, len(sat))
    ax.plot(t_m, motion, color="#888", lw=0.6, label="motion")
    ax.plot(t_m, onset, color="#d1495b", lw=0.8, alpha=0.7, label="audio onset")
    ax.plot(t_e, edge, color="#2e86ab", lw=0.8, alpha=0.8, label="edge density")
    ax.plot(t_s, sat/sat.max(), color="#8a6fae", lw=0.8, alpha=0.6, label="saturation")
    for c in r["cuts"]:
        ax.axvline(c, color="k", lw=0.8, alpha=0.55)
    beats = np.load(os.path.join(OUT, f"beats_{key}.npy"))
    for b in beats:
        ax.axvline(b, color="#d1495b", lw=0.3, alpha=0.18)
    ax.set_title(f"{key} ({r['name']})  shots={r['n_shots']} bpm={r['bpm']:.0f}", loc="left")
    ax.set_ylim(0, 1.05); ax.legend(loc="upper right", fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "structure.png"), dpi=95)
print("saved")
