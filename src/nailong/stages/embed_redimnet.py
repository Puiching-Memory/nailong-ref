"""用 ReDimNet 为新增句子追加审计声纹嵌入。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import audio, config, manifests
from ..device import checked_device
from ..speakers import RedimNetEmbedder, l2norm


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    device = "cuda:0"
    if "--device" in argv:
        at = argv.index("--device")
        if at + 1 >= len(argv):
            raise SystemExit("--device 后需要 cuda:0 或 cpu")
        device = argv[at + 1]
        del argv[at:at + 2]
    device = checked_device(device)
    out = Path(argv[0]) if argv else config.EMB_REDIMNET
    use_orig = len(argv) > 1 and argv[1] == "orig"
    rows = manifests.read(config.UTT_MANIFEST)
    previous = np.load(out) if out.exists() else np.empty((0, 192), dtype=np.float32)
    if previous.ndim != 2 or previous.shape[1] != 192 or len(previous) > len(rows):
        raise SystemExit(f"现有嵌入 {previous.shape} 与 {len(rows)} 条句子不兼容")
    pending = rows[len(previous):]
    if not pending:
        print(f"没有待追加的新句子；嵌入保持 {previous.shape} -> {config.rel(out)}")
        return 0

    embedder = RedimNetEmbedder(device=device)
    print(f"ReDimNet 已加载 -> {embedder.dev}；待追加 {len(pending)} 句", flush=True)
    cache: dict[str, np.ndarray] = {}

    def load(src: str) -> np.ndarray:
        if src not in cache:
            path = config.mix_path(src) if use_orig else config.vocals_path(src)
            cache[src] = audio.decode(path, config.SR_MODEL)
        return cache[src]

    values = []
    for number, row in enumerate(pending, 1):
        wave = load(row["src"])
        i0 = round(float(row["t0"]) * config.SR_MODEL)
        i1 = round(float(row["t1"]) * config.SR_MODEL)
        values.append(embedder.from_wave(wave[i0:i1]))
        if number % 10 == 0 or number == len(pending):
            print(f"ReDimNet: {number}/{len(pending)}", flush=True)

    matrix = np.vstack([previous, np.asarray(values, dtype=np.float32)])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, matrix)
    print(f"嵌入 {matrix.shape} -> {config.rel(out)}", flush=True)
    normalized = l2norm(matrix)
    similarities = normalized @ normalized.T
    print(f"全体两两余弦: 中位={np.median(similarities):.3f} "
          f"p10={np.percentile(similarities, 10):.3f} "
          f"p90={np.percentile(similarities, 90):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
