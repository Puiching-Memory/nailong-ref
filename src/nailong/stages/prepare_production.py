"""从视觉确认的奶龙句子构建保守的生产数据集。

只有 3--10 秒的连续目标对白才形成候选；再以 Demucs vocals/no_vocals 的 RMS
差做残留 BGM/SFX 门控。>=15 dB 进入 accepted，其余进入 quarantine，绝不让
音频模型的自动高分覆盖视觉真值或音质门控。
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np

from .. import audio, config, manifests

MIN_STEM_MARGIN_DB = 15.0


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))) + 1e-12)


def _probe(path) -> tuple[int, int, float]:
    command = ["ffprobe", "-v", "error", "-select_streams", "a:0",
               "-show_entries", "stream=sample_rate,channels,duration",
               "-of", "csv=p=0", str(path)]
    fields = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(fields[0]), int(fields[1]), float(fields[2])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.ensure_dirs()
    previous_files = []
    if config.PRODUCTION_MANIFEST.exists():
        previous_files = [row["file"] for row in manifests.read(config.PRODUCTION_MANIFEST)]
    labels = [row for row in manifests.read(config.VISUAL_LABELS) if row["class"] == "target"]
    visual_target_ids = {int(row["idx"]) for row in labels}
    scores = manifests.by_idx(config.PRODUCTION_SCORES)
    utterances = manifests.by_idx(config.UTT_MANIFEST)
    events = manifests.by_idx(config.SV_ALL)
    texts = manifests.texts()
    segments = []
    for idx, score in scores.items():
        if score["decision"] not in ("confirmed_target", "auto_accept"):
            continue
        if events[idx]["event"] != "Speech":
            continue
        if any("\uac00" <= character <= "\ud7a3" for character in texts[idx]):
            continue
        row = utterances[idx]
        # dur 含切句前后 padding，只用于声纹窗口；生产边界必须使用真实 t1，
        # 否则源尾片段会被清单虚增而 WAV 实际不足。
        t0, t1 = float(row["t0"]), float(row["t1"])
        segments.append((row["src"], t0, t1, idx, score["decision"]))
    segments.sort()

    runs: list[dict] = []
    for src, t0, t1, idx, decision in segments:
        if runs and runs[-1]["src"] == src and t0 - runs[-1]["segments"][-1][1] <= config.ADJ_GAP:
            runs[-1]["segments"].append((t0, t1, idx, decision))
        else:
            runs.append({"src": src, "segments": [(t0, t1, idx, decision)]})

    groups = []
    for run in runs:
        current = []
        for segment in run["segments"]:
            if current and segment[1] - current[0][0] > config.MAX_RUN:
                groups.append((run["src"], current))
                current = []
            current.append(segment)
        if current:
            groups.append((run["src"], current))

    vocals = audio.Decoder(config.SR_MODEL)
    export_vocals = audio.Decoder(config.SR_EXPORT, dtype="float64")
    background_cache: dict[str, np.ndarray] = {}
    made = []
    for src, group in groups:
        t0, t1 = group[0][0], group[-1][1]
        duration = t1 - t0
        if not (config.MIN_RUN <= duration <= config.MAX_RUN):
            continue
        if src not in background_cache:
            background_cache[src] = audio.decode(config.no_vocals_path(src), config.SR_MODEL)
        i0, i1 = round(t0 * config.SR_MODEL), round(t1 * config.SR_MODEL)
        vocal_part = vocals(src)[i0:i1]
        background_part = background_cache[src][i0:i1]
        margin_db = 20.0 * np.log10(_rms(vocal_part) / _rms(background_part))
        status = "accepted" if margin_db >= MIN_STEM_MARGIN_DB else "quarantine"
        filename = f"{src}_{t0:.2f}-{t1:.2f}.wav"
        relative = f"{status}/{filename}"
        output = config.PRODUCTION / relative
        export_i0, export_i1 = round(t0 * config.SR_EXPORT), round(t1 * config.SR_EXPORT)
        audio.write_wav(output, export_vocals(src)[export_i0:export_i1], config.SR_EXPORT,
                        highpass=config.HIGHPASS_HZ)
        indices = [segment[2] for segment in group]
        decisions = [segment[3] for segment in group]
        visual_ids = [idx for idx in indices if idx in visual_target_ids]
        speaker_basis = ("visual_confirmed" if len(visual_ids) == len(indices)
                         else "calibrated_audio")
        made.append([
            relative, src, f"{t0:.2f}", f"{t1:.2f}", f"{duration:.2f}",
            status, speaker_basis, f"{margin_db:.2f}", ";".join(map(str, indices)),
            ";".join(decisions), ";".join(map(str, visual_ids)),
            config.SEG_SEP.join(texts[idx] for idx in indices), _sha256(output),
        ])

    # 只清理由上一版清单登记、且这次已不再产出的文件，避免边界修正后留下幽灵样本。
    current_files = {row[0] for row in made}
    production_root = config.PRODUCTION.resolve()
    for relative in previous_files:
        stale = (config.PRODUCTION / relative).resolve()
        if stale.parent.parent != production_root or relative in current_files:
            continue
        if stale.exists():
            stale.unlink()

    manifests.write(config.PRODUCTION_MANIFEST,
                    ["file", "src", "t0", "t1", "dur", "status", "speaker_basis",
                     "stem_margin_db", "utterance_ids", "speaker_decisions",
                     "visual_label_ids", "text", "sha256"], made)

    separation_rows = []
    for src in config.source_names():
        vpath, npath = config.vocals_path(src), config.no_vocals_path(src)
        rate, channels, duration = _probe(vpath)
        separation_rows.append([src, "htdemucs", "two_stems_vocals", rate, channels,
                                f"{duration:.3f}", _sha256(config.source_path(src)),
                                _sha256(vpath), _sha256(npath)])
    manifests.write(config.MANIFESTS / "separation_manifest.csv",
                    ["src", "model", "mode", "sample_rate", "channels", "duration",
                     "source_sha256", "vocals_sha256", "no_vocals_sha256"], separation_rows)

    accepted = sum(row[5] == "accepted" for row in made)
    accepted_seconds = sum(float(row[4]) for row in made if row[5] == "accepted")
    print(f"候选 {len(made)} 段；accepted {accepted} 段 / {accepted_seconds:.2f}s；"
          f"quarantine {len(made) - accepted} 段")
    print(f"生产清单 -> {config.rel(config.PRODUCTION_MANIFEST)}")
    print("分离溯源 -> dataset/manifests/separation_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
