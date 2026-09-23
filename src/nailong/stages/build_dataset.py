"""Run the complete extraction chain in order, then refresh the training review queue."""

from __future__ import annotations

import argparse
import subprocess
import sys

from .. import config
from ..device import checked_device
from .ingest_candidates import DEFAULT_RANKING

STAGES = [
    "separate", "segment", "transcribe", "embed-redimnet", "embed-eres2net",
    "calibrate-visual", "prepare-production",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=0,
                        help="先从本地候选排名导入多少个新源；0 表示只处理已有源")
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING),
                        help="候选排名 CSV，使用 --batch 时生效")
    parser.add_argument("--plan", action="store_true", help="只打印将执行的阶段")
    parser.add_argument("--device", default="cuda:0", help="模型计算设备，默认 cuda:0")
    args = parser.parse_args(argv)
    if args.batch < 0:
        parser.error("--batch 不能为负")
    if not args.plan:
        checked_device(args.device)
    steps = [([sys.executable, "-m", "nailong.stages.ingest_candidates",
               "--limit", str(args.batch), "--ranking", args.ranking])] if args.batch else []
    steps += [[sys.executable, "-m", "nailong.cli", stage, "--device", args.device]
              if stage in {"separate", "transcribe", "embed-redimnet", "embed-eres2net"}
              else [sys.executable, "-m", "nailong.cli", stage] for stage in STAGES]
    steps += [[sys.executable, "-m", "nailong_tts.cli", "review"],
              [sys.executable, "-m", "nailong_tts.cli", "audit"]]
    for number, command in enumerate(steps, 1):
        print(f"[{number}/{len(steps)}] {' '.join(command)}", flush=True)
        if not args.plan:
            result = subprocess.run(command, cwd=config.ROOT, check=False)
            if result.returncode:
                print(f"阶段失败（exit {result.returncode}）；修复后不带 --batch 续跑，"
                      "以免又导入一批新源。",
                      file=sys.stderr)
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
