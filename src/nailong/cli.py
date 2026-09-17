"""命令行入口。

    nailong <stage> [该阶段自己的参数...]

各阶段仍是脚本式的（模块级代码 + 自己的 argv 约定），这里只做分发，
参数原样透传，所以 `nailong segment` 与 `python -m nailong.stages.segment` 等价。
"""

from __future__ import annotations

import runpy
import sys

# 名称 -> (模块, 说明)。顺序即推荐执行顺序。
STAGES: dict[str, tuple[str, str]] = {
    # --- 抽取链路 ---
    "separate": ("nailong.stages.separate", "mp4 -> demucs 人声轨（缺这一步下游全废）"),
    "segment": ("nailong.stages.segment", "VAD 细粒度切句 -> utterances/ + utt_manifest.csv"),
    "transcribe": ("nailong.stages.transcribe", "SenseVoice 事件/情绪/文本 -> sv_all.csv"),
    "embed-redimnet": ("nailong.stages.embed_redimnet", "ReDimNet 声纹嵌入（主力）"),
    "embed-funasr": ("nailong.stages.embed_funasr", "FunASR 声纹嵌入（用于横向对比）"),
    "embed-episode": ("nailong.stages.embed_episode", "整集长参考嵌入（短句不可靠的解法）"),
    "embed-chunk": ("nailong.stages.embed_chunk", "滑窗分块嵌入（检测混说话人的句子）"),
    "cluster": ("nailong.stages.cluster", "无监督聚类扫描（已证明本素材不可用）"),
    "score-long-ref": ("nailong.stages.score_long_ref", "长参考初筛 -> utt_margin_long.csv"),
    "score-two-pass": ("nailong.stages.score_two_pass", "两遍迭代细化 -> utt_two_pass.csv"),
    "verify": ("nailong.stages.verify", "三条独立证据验证组A 是奶龙"),
    "finalize": ("nailong.stages.finalize", "重建交付片段 -> dataset/final/"),
    # --- 工具 ---
    "annotate": ("nailong.tools.annotate", "主动学习标注：seed / status / query / teach"),
    "inspect-group": ("nailong.tools.inspect_group", "列出两组全部成员，看 B 组是不是语气词堆"),
    "sanity": ("nailong.tools.sanity", "声纹模型自检（确认嵌入真有区分度）"),
    "reel": ("nailong.tools.reel", "把目录里的片段串成试听带"),
    "band": ("nailong.tools.band", "频段/RMS 量化分析（无播放设备时判 BGM）"),
}


def _print_help() -> None:
    print(__doc__.strip())
    print("\n可用阶段:\n")
    width = max(len(k) for k in STAGES)
    for name, (_, desc) in STAGES.items():
        print(f"  {name:<{width}}  {desc}")
    print("\n示例:")
    print("  nailong separate                缺什么补什么")
    print("  nailong segment                 切句")
    print("  nailong annotate status         看标注进度")
    print("  nailong band dataset/final/*.wav")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        _print_help()
        return 0

    name, rest = argv[0], argv[1:]
    if name not in STAGES:
        print(f"未知阶段 '{name}'。用 `nailong help` 看全部阶段。", file=sys.stderr)
        return 2

    module = STAGES[name][0]
    sys.argv = [f"nailong {name}", *rest]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
