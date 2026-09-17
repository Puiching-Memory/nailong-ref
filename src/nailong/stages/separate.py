"""从 src_NN.mp4 生成 demucs 的输入与人声轨。

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
"""

from __future__ import annotations

import subprocess
import sys

from .. import config


def extract_mix(src: str, force: bool = False) -> bool:
    """mp4 -> 单声道 wav。保持源采样率，避免先降采样再被 demucs 升回去。"""
    out = config.SEP_IN / f"{src}.wav"
    if out.exists() and not force:
        print(f"  跳过 {config.rel(out)}（已存在）")
        return False
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(config.source_path(src)),
                    "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(out)], check=True)
    print(f"  {config.rel(config.source_path(src))} -> {config.rel(out)}")
    return True


def separate_vocals(src: str, force: bool = False) -> bool:
    """htdemucs 两轨分离。近似能量守恒：orig ≈ vocals ⊕ no_vocals。"""
    out = config.SEP_OUT / "htdemucs" / src / "vocals.wav"
    if out.exists() and not force:
        print(f"  跳过 {config.rel(out)}（已存在）")
        return False
    subprocess.run([sys.executable, "-m", "demucs.separate",
                    "--two-stems=vocals", "-n", "htdemucs",
                    "-o", str(config.SEP_OUT), str(config.SEP_IN / f"{src}.wav")],
                   check=True)
    if not out.exists():
        raise RuntimeError(f"demucs 未产出 {config.rel(out)}")
    print(f"  -> {config.rel(out)}")
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    srcs = [a for a in argv if not a.startswith("-")] or config.SRCS
    unknown = [s for s in srcs if s not in config.SRCS]
    if unknown:
        print(f"未知的集名 {unknown}；可选 {config.SRCS}", file=sys.stderr)
        return 2

    config.ensure_dirs()
    print(f"分离 {len(srcs)} 集 -> {config.rel(config.SEP_OUT)}"
          f"{'  (force)' if force else ''}")
    for src in srcs:
        print(f"[{src}]")
        extract_mix(src, force)
        separate_vocals(src, force)
    print("\n完成。下游可用：nailong segment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
