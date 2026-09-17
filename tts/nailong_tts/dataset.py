"""把 `dataset/final/` 打包成 TTS 训练器认的格式，并做质检。

依赖 `final_manifest.csv`（含每段台词），它由 `nailong finalize` 产出。
没有它就无法训练——TTS 需要文本-音频对齐，而片段是拼接出来的，
只有 finalize 知道每段由哪几句组成。

产出（默认写到 `tts/build/`）::

    filelist.txt    wav|speaker|lang|text        多数中文 TTS 训练器直接吃
    metadata.csv    file,path,dur,min_simA,text  带质检字段，便于人工复核
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
MIN_CORPUS_SEC = 60.0   # 音色克隆的经验下限（当前素材 75s）
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
    min_simA: float
    text: str


def load(manifest: Path = config.FINAL_MANIFEST,
         audio_dir: Path = config.FINAL) -> list[Clip]:
    if not Path(manifest).exists():
        raise SystemExit(f"缺少 {config.rel(manifest)}；先跑 `nailong finalize`。")
    clips = []
    for r in manifests.read(manifest):
        p = Path(audio_dir) / r["file"]
        if not p.exists():
            raise SystemExit(f"清单里的 {r['file']} 不在 {config.rel(audio_dir)}")
        clips.append(Clip(r["file"], p, float(r["dur"]), float(r["min_simA"]), r["text"]))
    if not clips:
        raise SystemExit(f"{config.rel(manifest)} 是空的")
    return clips


def orphans(audio_dir: Path = config.FINAL,
            manifest: Path = config.FINAL_MANIFEST) -> list[str]:
    """磁盘上有、清单里没有的 wav。手工塞进去的片段没有台词，会污染训练集。"""
    if not Path(manifest).exists():
        return []
    known = {r["file"] for r in manifests.read(manifest)}
    return sorted(p.name for p in Path(audio_dir).glob("*.wav") if p.name not in known)


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
    print(f"可信度 min_simA: 最低 {min(c.min_simA for c in clips):.3f}，"
          f"中位 {sorted(c.min_simA for c in clips)[len(clips) // 2]:.3f}")

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


def pack(out_dir: Path = BUILD, manifest: Path = config.FINAL_MANIFEST,
         audio_dir: Path = config.FINAL) -> Path:
    clips = load(manifest, audio_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "filelist.txt").write_text(
        "".join(f"{c.path.as_posix()}|{SPEAKER}|{LANG}|{speakable(c.text)}\n"
                for c in clips),
        encoding="utf-8")

    manifests.write(out_dir / "metadata.csv",
                    ["file", "path", "dur", "min_simA", "text", "text_speakable"],
                    [[c.file, c.path.as_posix(), manifests.f2(c.dur),
                      f"{c.min_simA:.4f}", c.text, speakable(c.text)] for c in clips])

    report(clips)
    print(f"\n-> {out_dir / 'filelist.txt'}")
    print(f"-> {out_dir / 'metadata.csv'}")
    return out_dir
