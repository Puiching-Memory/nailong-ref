"""从「组A = 奶龙」的句子重建正式交付片段。

判据不是相似度阈值，而是 score-long-ref + score-two-pass 推出的说话人划分
（组A = 奶龙，已由 verify 的文本证据交叉验证）。只保留连续 3-10s 的区间，
并按事件标签过滤掉非纯语音。

同时产出 `final_manifest.csv`（含每段的台词），TTS 侧靠它做数据打包。

用法::

    nailong finalize                 写音频 + 清单
    nailong finalize --manifest-only 只出清单（不读 data/sep_out，用于核对切分）
"""
import sys

from .. import audio, config, manifests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MANIFEST_ONLY = "--manifest-only" in sys.argv
GAP_TOL = config.ADJ_GAP
MIN_RUN, MAX_RUN = config.MIN_RUN, config.MAX_RUN
# 组A 内部可信度下限。#2(0.464) 已由用户判为错误说话人，且它是组A 最低分，
# 低分段还混有「您真大方。」这类明显属其他角色的敬语句，故按分数截尾。
MIN_A = config.MIN_A

rows = manifests.by_idx(config.UTT_MANIFEST)
svr = manifests.by_idx(config.SV_ALL)
tp = manifests.by_idx(config.TWO_PASS)

kept, dropped = [], []
for i, r in tp.items():
    if r["group"] != "A":
        dropped.append((i, "非组A"))
    elif svr[i]["event"] != "Speech":
        dropped.append((i, f"事件={svr[i]['event']}"))
    elif any("\uac00" <= ch <= "\ud7a3" for ch in svr[i]["text"]):
        dropped.append((i, "识别为韩文"))
    elif float(r["simA"]) < MIN_A:
        dropped.append((i, f"A分{r['simA']}低于{MIN_A}"))
    else:
        kept.append((i, rows[i]["src"], float(rows[i]["t0"]), float(rows[i]["dur"]),
                     float(r["simA"]), svr[i]["text"]))

print(f"组A 可用 {len(kept)} 句，排除 {len(dropped)} 句")
low = sorted(((i, float(r["simA"])) for i, r in tp.items()
              if r["group"] == "A" and float(r["simA"]) < MIN_A), key=lambda x: x[1])
print(f"\n因 A分 < {MIN_A} 被截尾的 {len(low)} 句:")
for i, sa in low:
    print(f"  #{i:>3}  A分={sa:.3f}  {svr[i]['text'][:32]}")
runs = []
for i, src, t0, dur, sa, t in sorted(kept, key=lambda k: (k[1], k[2])):
    if runs and runs[-1]["src"] == src and t0 - runs[-1]["segs"][-1][1] <= GAP_TOL:
        runs[-1]["segs"].append((t0, t0 + dur, sa, t, i))
    else:
        runs.append(dict(src=src, segs=[(t0, t0 + dur, sa, t, i)]))

# 长区间按语句边界切成 <= MAX_RUN 的子段（整段丢弃会漏掉开头的可用素材）
groups = []
for r in runs:
    cur = []
    for s in r["segs"]:
        if cur and s[1] - cur[0][0] > MAX_RUN:
            groups.append((r["src"], cur))
            cur = []
        cur.append(s)
    if cur:
        groups.append((r["src"], cur))

config.ensure_dirs()
dec = None if MANIFEST_ONLY else audio.Decoder(config.SR_EXPORT, dtype="float64")
made = []
print(f"\n{'片段':<28}{'时长':>7}{'句':>4}{'最低A分':>9}  台词")
for src, segs_ in groups:
    dur = segs_[-1][1] - segs_[0][0]
    if not (MIN_RUN <= dur <= MAX_RUN):
        continue
    t0, t1 = segs_[0][0], segs_[-1][1]
    fname = f"{src}_{t0:.2f}-{t1:.2f}.wav"
    lo_a = min(s[2] for s in segs_)
    if dec is not None:
        x = dec(src)
        # 高通压掉分离残留的低频伴奏
        audio.write_wav(config.FINAL / fname, x[int(t0 * dec.sr):int(t1 * dec.sr)],
                        config.SR_EXPORT, highpass=config.HIGHPASS_HZ)
    made.append((fname, src, t0, t1, dur, lo_a, config.SEG_SEP.join(s[3] for s in segs_)))
    print(f"{fname:<28}{dur:>7.2f}{len(segs_):>4}{lo_a:>9.3f}  {made[-1][6]}")

made.sort(key=lambda m: -m[5])
manifests.write(config.FINAL_MANIFEST,
                ["file", "src", "t0", "t1", "dur", "min_simA", "text"],
                [[m[0], m[1], manifests.f2(m[2]), manifests.f2(m[3]), manifests.f2(m[4]),
                  f"{m[5]:.4f}", m[6]] for m in made])

print(f"\n{'（仅清单，未写音频）' if MANIFEST_ONLY else ''}"
      f"导出 {len(made)} 个片段 -> {config.rel(config.FINAL)}/")
print(f"清单 -> {config.rel(config.FINAL_MANIFEST)}")
print("\n按最低 A 分排序（最可信的在前）:")
for name, _src, _t0, _t1, dur, sa, _text in made[:8]:
    print(f"  {sa:.3f}  {dur:>5.2f}s  {name}")
