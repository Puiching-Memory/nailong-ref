"""用已校准的 ERes2NetV2 对本地候选源做廉价预筛。

该工具直接分析原始混音，不修改生产数据。它的目的只是决定哪些源值得花时间
跑 Demucs；最终是否入库仍由分离后声纹判断与 BGM/SFX 门控决定。

用法::

    nailong rank-sources data/eval_sep/visual/audio_index \
        data/eval_sep/visual/candidate_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .. import audio, config, manifests
from ..device import checked_device
from ..stages.embed_eres2net import ERes2NetV2Embedder

SP_FMIN, SP_FMAX = 250, 4000


def _segments(wave: np.ndarray) -> list[tuple[float, float]]:
    frames = audio.frame_matrix(wave, config.FRAME, config.HOP)
    if frames is None:
        return []
    spectrum = audio.spectrum(frames)
    freqs = audio.freqs(config.FRAME, config.SR_MODEL)
    mask = ((audio.frame_db(frames) > config.DB_THRESH) &
            (audio.band_ratio(spectrum, freqs, SP_FMIN, SP_FMAX) > config.SP_THRESH))
    hop_seconds = config.HOP / config.SR_MODEL
    found: list[tuple[int, int]] = []
    start = last = None
    for number, active in enumerate(mask):
        if active:
            if start is None:
                start = number
            last = number
        elif start is not None and (number - last) * hop_seconds > config.GAP_TOL:
            found.append((start, last + 1))
            start = last = None
    if start is not None:
        found.append((start, last + 1))
    return [(a * hop_seconds, b * hop_seconds) for a, b in found
            if (b - a) * hop_seconds >= config.MIN_SEG]


def _load_metadata(directory: Path) -> dict[str, dict[str, str]]:
    path = directory.parent / "candidate_metadata.csv"
    if not path.exists():
        return {}
    return {row["bvid"]: row for row in manifests.read(path)}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=config.DATA / "eval_sep" / "visual" / "audio_index")
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    directory = args.directory
    output = args.output or directory.parent / "candidate_scores.csv"
    device = checked_device(args.device)

    files = sorted(directory.glob("*.m4a"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"没有候选音频: {directory}")

    package = json.loads((config.CALIBRATION / "calibration.json").read_text(encoding="utf-8"))
    normalization = package["scoring"]["normalization"]
    thresholds = package["thresholds"]
    target = np.load(config.CALIBRATION / "eres2netv2_target_centroid.npy")
    other = np.load(config.CALIBRATION / "eres2netv2_other_centroid.npy")
    metadata = _load_metadata(directory)
    embedder = ERes2NetV2Embedder(device)

    rows = []
    segment_rows = []
    for number, path in enumerate(files, 1):
        wave = audio.decode(path, config.SR_MODEL)
        segments = _segments(wave)
        scores = []
        durations = []
        for t0, t1 in segments:
            sample = wave[round(t0 * config.SR_MODEL):round(t1 * config.SR_MODEL)]
            embedding = embedder(sample)
            margin = float(embedding @ target - embedding @ other)
            score = (margin - normalization["mean"]) / normalization["std"]
            duration = t1 - t0
            scores.append(score)
            durations.append(duration)
            decision = ("accept" if score >= thresholds["accept"] else
                        "reject" if score < thresholds["reject"] else "review")
            segment_rows.append([
                path.stem, f"{t0:.2f}", f"{t1:.2f}", f"{duration:.2f}",
                f"{margin:.6f}", f"{score:.6f}", decision,
            ])
        accepted = [duration for duration, score in zip(durations, scores, strict=True)
                    if score >= thresholds["accept"]]
        meta = metadata.get(path.stem, {})
        rows.append([
            path.stem, meta.get("title", ""), meta.get("owner", ""),
            meta.get("owner_mid", ""), meta.get("page_url", ""),
            f"{len(wave) / config.SR_MODEL:.2f}", len(segments),
            f"{sum(durations):.2f}", len(accepted), f"{sum(accepted):.2f}",
            f"{max(scores) if scores else float('-inf'):.6f}",
            f"{np.mean(scores) if scores else float('-inf'):.6f}", path.as_posix(),
        ])
        print(f"预筛 {number}/{len(files)} {path.stem}: "
              f"accept={len(accepted)}/{len(segments)} {sum(accepted):.1f}s", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bvid", "title", "owner", "owner_mid", "page_url", "duration", "segments",
                         "speech_seconds", "accept_segments", "accept_seconds",
                         "max_score", "mean_score", "local_file"])
        writer.writerows(rows)
    detail = output.with_name(f"{output.stem}_segments.csv")
    with open(detail, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bvid", "t0", "t1", "dur", "margin", "score", "decision"])
        writer.writerows(segment_rows)
    print(f"源排名 -> {output}; 片段明细 -> {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
