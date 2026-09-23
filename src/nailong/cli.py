"""命令行入口。

    nailong <stage> [该阶段自己的参数...]

这里只做阶段分发并原样透传参数，所以 `nailong segment` 与
`python -m nailong.stages.segment` 等价。
"""

from __future__ import annotations

import runpy
import sys

# 名称 -> (模块, 说明)。顺序即推荐执行顺序。
STAGES: dict[str, tuple[str, str]] = {
    # --- 抽取链路 ---
    "build-dataset": ("nailong.stages.build_dataset", "一键运行增量抽取并审计训练就绪度"),
    "ingest-candidates": ("nailong.stages.ingest_candidates", "从本地候选排名导入新源"),
    "separate": ("nailong.stages.separate", "mp4 -> demucs 人声轨（缺这一步下游全废）"),
    "segment": ("nailong.stages.segment", "VAD 细粒度切句 -> utterances/ + utt_manifest.csv"),
    "transcribe": ("nailong.stages.transcribe", "SenseVoice 事件/情绪/文本 -> sv_all.csv"),
    "embed-redimnet": ("nailong.stages.embed_redimnet", "ReDimNet 声纹嵌入（审计旁证）"),
    "embed-eres2net": ("nailong.stages.embed_eres2net", "ERes2NetV2 声纹嵌入（视觉校准后主模型）"),
    "embed-funasr": ("nailong.stages.embed_funasr", "FunASR 声纹嵌入（用于横向对比）"),
    "calibrate-visual": ("nailong.stages.calibrate_visual", "视觉真值 -> 双模型原型、阈值、全量打分"),
    "prepare-production": ("nailong.stages.prepare_production", "严格门控 -> dataset/production/"),
    # --- 工具 ---
    "annotate": ("nailong.tools.annotate", "主动学习标注：seed / status / query / teach"),
    "sanity": ("nailong.tools.sanity", "声纹模型自检（确认嵌入真有区分度）"),
    "reel": ("nailong.tools.reel", "把目录里的片段串成试听带"),
    "band": ("nailong.tools.band", "频段/RMS 量化分析（无播放设备时判 BGM）"),
    "rank-sources": ("nailong.tools.rank_sources", "用校准声纹预筛本地候选源（不改生产数据）"),
    "fetch-public": ("nailong.tools.fetch_public", "缓存官方公开视频音频及来源元数据"),
}


def _print_help() -> None:
    print(__doc__.strip())
    print("\n可用阶段:\n")
    width = max(len(k) for k in STAGES)
    for name, (_, desc) in STAGES.items():
        print(f"  {name:<{width}}  {desc}")
    print("\n示例:")
    print("  nailong build-dataset --device cuda:0  增量 GPU 抽取")
    print("  nailong separate                缺什么补什么")
    print("  nailong segment                 切句")
    print("  nailong annotate status         看标注进度")
    print("  nailong band dataset/production/accepted/*.wav")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
