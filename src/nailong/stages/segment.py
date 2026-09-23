"""对新分离源做 VAD 切句，并追加到稳定的句级主键空间。

默认只处理 ``utt_manifest.csv`` 中尚未出现的源。已有句子的 idx 永不重排，因而
扩数据时不会破坏视觉校准标签。VAD 只负责候选切句；说话人判断由后续校准模型完成。
"""

from __future__ import annotations

import sys

import numpy as np
import soundfile as sf

from .. import audio, config, manifests

SP_FMIN, SP_FMAX = 250, 4000
FIELDS = ["idx", "src", "t0", "t1", "dur"]


def detect(wave: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    frames = audio.frame_matrix(wave, config.FRAME, config.HOP)
    if frames is None:
        return []
    spectrum = audio.spectrum(frames)
    freqs = audio.freqs(config.FRAME, config.SR_MODEL)
    mask = ((audio.frame_db(frames) > config.DB_THRESH) &
            (audio.band_ratio(spectrum, freqs, SP_FMIN, SP_FMAX) > config.SP_THRESH))
    hop_seconds = config.HOP / config.SR_MODEL
    ranges: list[tuple[int, int]] = []
    start = last = None
    for number, active in enumerate(mask):
        if active:
            if start is None:
                start = number
            last = number
        elif start is not None and (number - last) * hop_seconds > config.GAP_TOL:
            ranges.append((start, last + 1))
            start = last = None
    if start is not None:
        ranges.append((start, last + 1))

    pad = round(config.SEG_PAD * config.SR_MODEL)
    found = []
    for first, final in ranges:
        t0, t1 = first * hop_seconds, final * hop_seconds
        if t1 - t0 < config.MIN_SEG:
            continue
        i0, i1 = first * config.HOP, final * config.HOP
        clip = wave[max(0, i0 - pad):min(len(wave), i1 + pad)]
        found.append((t0, t1, clip))
    return found


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    requested = list(sys.argv[1:] if argv is None else argv)
    config.ensure_dirs()
    existing = manifests.read(config.UTT_MANIFEST) if config.UTT_MANIFEST.exists() else []
    done = {row["src"] for row in existing}
    available = config.source_names()
    srcs = requested or [src for src in available if src not in done]
    unknown = sorted(set(srcs) - set(available))
    repeated = sorted(set(srcs) & done)
    if unknown:
        raise SystemExit(f"没有这些源文件: {unknown}")
    if repeated:
        raise SystemExit(f"这些源已经切句，拒绝产生重复 idx: {repeated}")
    if not srcs:
        print("没有待追加的新源")
        return 0

    next_idx = max((int(row["idx"]) for row in existing), default=0) + 1
    added = []
    for src in srcs:
        wave = audio.decode(config.vocals_path(src), config.SR_MODEL, dtype="float64")
        clips = detect(wave)
        for t0, t1, clip in clips:
            idx = next_idx
            next_idx += 1
            sf.write(str(config.UTTERANCES / f"utt_{idx:03d}.wav"), clip, config.SR_MODEL)
            added.append([idx, src, f"{t0:.2f}", f"{t1:.2f}", f"{len(clip) / config.SR_MODEL:.2f}"])
        print(f"{src}: +{len(clips)} 句", flush=True)

    preserved = [[row[field] for field in FIELDS] for row in existing]
    manifests.write(config.UTT_MANIFEST, FIELDS, [*preserved, *added])
    print(f"追加 {len(added)} 句；总计 {len(existing) + len(added)} 句 -> {config.rel(config.UTT_MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
