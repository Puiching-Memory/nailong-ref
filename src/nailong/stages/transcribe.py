"""对全部切出的句子跑 SenseVoice，输出事件/情绪标签与简体文本，用于按「纯语音」筛选。

SenseVoice 会在文本里带情感与事件 token（Speech / BGM / Laughter / …），
这里的 event 字段在 finalize 阶段被当作「非纯语音」的排除规则。
"""
import re
import sys
from collections import Counter

from funasr.utils.postprocess_utils import rich_transcription_postprocess

from .. import config, manifests
from ..speakers import load_funasr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAG = re.compile(r"<\|([^|]+)\|>")
EVENTS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}
EMOS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}

model = load_funasr("iic/SenseVoiceSmall")
print("SenseVoice loaded", flush=True)

rows = manifests.read(config.UTT_MANIFEST)
out = []
for r in rows:
    p = str(config.UTTERANCES / f"utt_{int(r['idx']):03d}.wav")
    res = model.generate(input=p, language="zh", use_itn=True,
                         batch_size_s=60, merge_vad=True)
    raw = res[0]["text"]
    tags = TAG.findall(raw)
    ev = "/".join(t for t in tags if t in EVENTS) or "-"
    em = "/".join(t for t in tags if t in EMOS) or "-"
    out.append((int(r["idx"]), r["src"], float(r["t0"]), float(r["dur"]), ev, em,
                rich_transcription_postprocess(raw)))

SV_FIELDS = ["idx", "src", "t0", "dur", "event", "emotion", "text"]

manifests.write(config.SV_ALL, SV_FIELDS,
                [[r[0], r[1], manifests.f2(r[2]), manifests.f2(r[3]), r[4], r[5], r[6]]
                 for r in out])

print("\n事件标签分布:", dict(Counter(r[4] for r in out)), flush=True)
print("情绪标签分布:", dict(Counter(r[5] for r in out)), flush=True)
print(f"\n{'#':>4}{'源':>9}{'起点':>8}{'时长':>7}  {'事件':<10}{'情绪':<10}文本")
for r in out:
    print(f"{r[0]:>4}{r[1]:>9}{r[2]:>8.2f}{r[3]:>7.2f}  {r[4]:<10}{r[5]:<10}{r[6]}", flush=True)
print(f"\n已写入 {config.rel(config.SV_ALL)}", flush=True)
