"""从 src_NN 音视频源生成 demucs 的输入与人声轨。

这一步原先只有手敲的命令、没留下脚本，所以 `data/sep_out/` 被删掉之后
整条链路静默失效——8 个下游脚本都在读 `vocals.wav` 却没人报错。
现在把它固化成可重跑的阶段。

产出::

    data/sep_in/src_NN.wav                    单声道原始混音（保持源采样率）
    data/sep_out/htdemucs/src_NN/vocals.wav   htdemucs --two-stems=vocals 的人声轨

用法::

    nailong separate               # 缺什么补什么
    nailong separate --force       # 全部重跑
    nailong separate src_03 src_05 # 只处理指定集
    nailong separate --device cuda:0 # 默认 GPU；显式 --device cpu 可诊断
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

from .. import config, manifests
from ..device import checked_device


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_mix(src: str, force: bool = False) -> bool:
    """mp4 -> 单声道 wav。保持源采样率，避免先降采样再被 demucs 升回去。"""
    out = config.SEP_IN / f"{src}.wav"
    other = config.SEP_OUT / "htdemucs" / src / "no_vocals.wav"
    if out.exists() and other.exists() and not force:
        print(f"  跳过 {config.rel(out)}（已存在）")
        return False
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(config.source_path(src)),
                    "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(out)], check=True)
    print(f"  {config.rel(config.source_path(src))} -> {config.rel(out)}")
    return True


def separate_vocals(src: str, force: bool = False, device: str = "cuda:0") -> bool:
    """htdemucs 两轨分离。近似能量守恒：orig ≈ vocals ⊕ no_vocals。"""
    out = config.SEP_OUT / "htdemucs" / src / "vocals.wav"
    other = out.with_name("no_vocals.wav")
    if out.exists() and other.exists() and not force:
        print(f"  跳过 {config.rel(out)}（已存在）")
        return False
    subprocess.run([sys.executable, "-m", "demucs.separate",
                    "--two-stems=vocals", "-n", "htdemucs", "-d", device,
                    "-o", str(config.SEP_OUT), str(config.SEP_IN / f"{src}.wav")],
                   check=True)
    if not out.exists():
        raise RuntimeError(f"demucs 未产出 {config.rel(out)}")
    print(f"  -> {config.rel(out)}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    force = args.force
    device = checked_device(args.device)
    available = config.source_names()
    srcs = args.sources or available
    unknown = [s for s in srcs if s not in available]
    if unknown:
        print(f"未知的集名 {unknown}；可选 {available}", file=sys.stderr)
        return 2

    config.ensure_dirs()
    separation_manifest = config.MANIFESTS / "separation_manifest.csv"
    previous = ({row["src"]: row for row in manifests.read(separation_manifest)}
                if separation_manifest.exists() else {})
    source_manifest = config.MANIFESTS / "source_manifest.csv"
    sources = ({row["src"]: row for row in manifests.read(source_manifest)}
               if source_manifest.exists() else {})
    print(f"分离 {len(srcs)} 集 ({device}) -> {config.rel(config.SEP_OUT)}"
          f"{'  (force)' if force else ''}")
    for src in srcs:
        print(f"[{src}]")
        source_hash = _sha256(config.source_path(src))
        if src in sources and sources[src]["sha256"] != source_hash:
            raise SystemExit(f"{src} 与 source_manifest.csv 的 SHA-256 不符；"
                             "请核对来源后更新清单，不能静默替换源音频")
        stale = src in previous and previous[src]["source_sha256"] != source_hash
        stem_stale = False
        if src in previous:
            for name, path in (("vocals_sha256", config.SEP_OUT / "htdemucs" / src / "vocals.wav"),
                               ("no_vocals_sha256", config.SEP_OUT / "htdemucs" / src / "no_vocals.wav")):
                if path.exists() and _sha256(path) != previous[src][name]:
                    stem_stale = True
                    print(f"  {name} 与清单不符，重新分离")
        if stale:
            print("  源 SHA-256 已变化，重新提取和分离")
        extract_mix(src, force or stale)
        separate_vocals(src, force or stale or stem_stale, device)
    print("\n完成。下游可用：nailong segment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
