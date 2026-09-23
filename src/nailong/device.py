"""Explicit compute-device validation shared by GPU stages."""

from __future__ import annotations


def checked_device(requested: str = "cuda:0") -> str:
    if requested == "cpu":
        return requested
    if not requested.startswith("cuda:") or not requested[5:].isdigit():
        raise SystemExit(f"无效设备 {requested!r}；使用 cuda:0 或 cpu")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("缺少 PyTorch；先安装 CUDA 版 torch 与 torchaudio") from exc
    index = int(requested[5:])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise SystemExit(
            f"要求 {requested}，但当前 PyTorch {torch.__version__} 无法使用该 GPU。"
            "运行 scripts/setup-gpu.ps1 安装 CUDA 版依赖，或显式使用 --device cpu。"
        )
    return requested
