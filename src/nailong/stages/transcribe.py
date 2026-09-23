"""用 SenseVoice 为新增句子追加事件、情绪与文本。"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter

from funasr.utils.postprocess_utils import rich_transcription_postprocess

from .. import config, manifests
from ..device import checked_device
from ..speakers import load_funasr

TAG = re.compile(r"<\|([^|]+)\|>")
EVENTS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}
EMOS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
FIELDS = ["idx", "src", "t0", "dur", "event", "emotion", "text"]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    device = checked_device(args.device)
    rows = manifests.read(config.UTT_MANIFEST)
    previous = manifests.read(config.SV_ALL) if config.SV_ALL.exists() else []
    done = {int(row["idx"]) for row in previous}
    pending = [row for row in rows if int(row["idx"]) not in done]
    if not pending:
        print("没有待转写的新句子")
        return 0

    model = load_funasr("iic/SenseVoiceSmall", device=device)
    print(f"SenseVoice loaded on {device}；待追加 {len(pending)} 句", flush=True)
    added = []
    config.SV_ALL.parent.mkdir(parents=True, exist_ok=True)
    with config.SV_ALL.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if not previous:
            writer.writerow(FIELDS)
        for number, row in enumerate(pending, 1):
            idx = int(row["idx"])
            path = str(config.UTTERANCES / f"utt_{idx:03d}.wav")
            result = model.generate(input=path, language="zh", use_itn=True,
                                    batch_size_s=60, merge_vad=True)[0]["text"]
            tags = TAG.findall(result)
            event = "/".join(tag for tag in tags if tag in EVENTS) or "-"
            emotion = "/".join(tag for tag in tags if tag in EMOS) or "-"
            text = rich_transcription_postprocess(result)
            record = [idx, row["src"], row["t0"], row["dur"], event, emotion, text]
            writer.writerow(record)
            stream.flush()
            added.append(record)
            print(f"转写 {number}/{len(pending)} #{idx}: {event} {text}", flush=True)
    print(f"新增事件: {dict(Counter(row[4] for row in added))}")
    print(f"总计 {len(previous) + len(added)} 句 -> {config.rel(config.SV_ALL)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
