"""把 `dataset/production/accepted/` 打包成 TTS 训练器认的格式，并做质检。

依赖 `production_manifest.csv`（含每段台词与音质门控），它由
`nailong prepare-production` 产出。quarantine 行永远不会打包。
没有它就无法训练——TTS 需要文本-音频对齐，而片段是拼接出来的，
只有 prepare-production 知道每段由哪几句组成。

产出（默认写到 `tts/build/`）::

    filelist.txt    wav|speaker|lang|text        多数中文 TTS 训练器直接吃
    metadata.csv    file,path,dur,speaker_basis,stem_margin_db,text
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from nailong import config, manifests

SPEAKER = "nailong"
LANG = "zh"
BUILD = Path(__file__).resolve().parents[1] / "build"
DUR_EPS = 0.01          # 清单时长与实测时长超过这个差异就告警
MIN_CORPUS_SEC = 60.0   # 仅用于候选统计告警；正式门槛见 quality.py
# SenseVoice 会把情感标成尾随表情，训练器读不了这些字符
EMOJI = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\ufe0f]")


def speakable(text: str) -> str:
    """文件列表里只能出现能读出来的字：去掉表情符号与段内句界标记。"""
    return EMOJI.sub("", text).replace(config.SEG_SEP, "").strip()


@dataclass(frozen=True)
class Clip:
    file: str
    path: Path
    dur: float
    speaker_basis: str
    stem_margin_db: float
    text: str


def load(manifest: Path = config.PRODUCTION_MANIFEST,
         audio_dir: Path = config.PRODUCTION) -> list[Clip]:
    if not Path(manifest).exists():
        raise SystemExit(f"缺少 {config.rel(manifest)}；先跑 `nailong prepare-production`。")
    clips = []
    for r in manifests.read(manifest):
        if r["status"] != "accepted":
            continue
        p = Path(audio_dir) / r["file"]
        if not p.exists():
            raise SystemExit(f"清单里的 {r['file']} 不在 {config.rel(audio_dir)}")
        clips.append(Clip(r["file"], p, float(r["dur"]), r["speaker_basis"],
                          float(r["stem_margin_db"]), r["text"]))
    if not clips:
        raise SystemExit(f"{config.rel(manifest)} 是空的")
    return clips


def orphans(audio_dir: Path = config.PRODUCTION,
            manifest: Path = config.PRODUCTION_MANIFEST) -> list[str]:
    """磁盘上有、清单里没有的 wav。手工塞进去的片段没有台词，会污染训练集。"""
    if not Path(manifest).exists():
        return []
    known = {r["file"] for r in manifests.read(manifest) if r["status"] == "accepted"}
    return sorted(p.relative_to(audio_dir).as_posix()
                  for p in (Path(audio_dir) / "accepted").glob("*.wav")
                  if p.relative_to(audio_dir).as_posix() not in known)


def measure_drift(clips: list[Clip]) -> list[tuple[str, float, float]]:
    """清单时长 vs 实测时长。差异大说明音频被改过而清单没跟着更新。"""
    drift = []
    for c in clips:
        info = sf.info(c.path)
        real = info.frames / info.samplerate
        if abs(real - c.dur) > DUR_EPS:
            drift.append((c.file, c.dur, real))
    return drift


def report(clips: list[Clip]) -> None:
    total = sum(c.dur for c in clips)
    print(f"片段 {len(clips)} 段，总长 {total:.1f}s，"
          f"最短 {min(c.dur for c in clips):.2f}s，最长 {max(c.dur for c in clips):.2f}s")
    print(f"残留轨差: 最低 {min(c.stem_margin_db for c in clips):.2f}dB，"
          f"中位 {sorted(c.stem_margin_db for c in clips)[len(clips) // 2]:.2f}dB")
    visual = sum(c.speaker_basis == "visual_confirmed" for c in clips)
    print(f"说话人依据: visual_confirmed={visual}, calibrated_audio={len(clips) - visual}")

    missing = [c.file for c in clips if not c.text.strip()]
    if missing:
        print(f"警告：{len(missing)} 段没有台词，TTS 无法使用 -> {missing}", file=sys.stderr)

    drift = measure_drift(clips)
    if drift:
        print(f"警告：{len(drift)} 段时长与清单不符（音频被改过？）", file=sys.stderr)
        for f, listed, real in drift:
            print(f"  {f}  清单 {listed:.2f}s  实测 {real:.2f}s", file=sys.stderr)

    extra = orphans()
    if extra:
        print(f"警告：{len(extra)} 个 wav 不在清单里（无台词）-> {extra}", file=sys.stderr)

    if total < MIN_CORPUS_SEC:
        print(f"警告：总长 {total:.1f}s 低于音色克隆的经验下限 {MIN_CORPUS_SEC:.0f}s",
              file=sys.stderr)


def pack(out_dir: Path = BUILD, manifest: Path = config.PRODUCTION_MANIFEST,
         audio_dir: Path = config.PRODUCTION, *, experimental: bool = False,
         review: Path | None = None) -> Path:
    from . import quality

    result = quality.audit(manifest, audio_dir, review or quality.REVIEW,
                           min_seconds=0.1 if experimental else quality.MIN_TRAIN_SECONDS,
                           min_clips=1 if experimental else quality.MIN_TRAIN_CLIPS)
    quality.print_audit(result)
    if result.issues:
        # A previous successful run must not leave a trainable-looking stale package.
        for name in ("filelist.txt", "metadata.csv"):
            (Path(out_dir) / name).unlink(missing_ok=True)
        raise SystemExit("训练包未通过质量门禁；先运行 nailong-tts review 并修正上述问题。")
    clips = [Clip(row["file"], Path(audio_dir) / row["file"], float(row["dur"]),
                  row["speaker_basis"], float(row["stem_margin_db"]), text)
             for row, text in result.usable]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "filelist.txt").write_text(
        "".join(f"{c.path.as_posix()}|{SPEAKER}|{LANG}|{speakable(c.text)}\n"
                for c in clips),
        encoding="utf-8")

    manifests.write(out_dir / "metadata.csv",
                    ["file", "path", "dur", "speaker_basis", "stem_margin_db",
                     "src", "utterance_ids", "sha256", "text", "text_speakable"],
                    [[c.file, c.path.as_posix(), manifests.f2(c.dur),
                      c.speaker_basis, f"{c.stem_margin_db:.2f}", row.get("src", ""),
                      row.get("utterance_ids", ""), row["sha256"], c.text,
                      speakable(c.text)]
                     for c, (row, _) in zip(clips, result.usable, strict=True)])

    report(clips)
    print(f"\n-> {out_dir / 'filelist.txt'}")
    print(f"-> {out_dir / 'metadata.csv'}")
    return out_dir
