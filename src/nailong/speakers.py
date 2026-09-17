"""声纹嵌入后端。

两条来源合并到一处，避免每个脚本各自重复 `torch.hub.load` 与 `embed()`：

- ReDimNet：Vox1-O EER 0.53%（ft_mix 权重含 CN-Celeb），本素材上最强的中文模型。
  FunASR 只注册了 CAMPPlus / ERes2NetV2，要上更强模型必须走 torch.hub。
- FunASR：用于 CAMPPlus / ERes2NetV2 的横向对比与短句嵌入。

重依赖（torch / funasr）全部惰性导入，保证 `import nailong` 无 GPU 也能成功。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .audio import Decoder

REDIMNET_REPO = "IDRnD/ReDimNet"


def pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_redimnet(device=None, model_name: str = "M", train_type: str = "ft_mix",
                  dataset: str = "vb2+vox2+cnc"):
    """返回 (model, device)。权重约 19MB，输入 16k 单声道 [N,T]，输出 [N,192]。"""
    import torch
    dev = device or pick_device()
    model = torch.hub.load(REDIMNET_REPO, "ReDimNet", model_name=model_name,
                           train_type=train_type, dataset=dataset, trust_repo=True)
    return model.to(dev).eval(), dev


def embed_redimnet(model, dev, x: np.ndarray) -> np.ndarray:
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        return model(t).squeeze(0).float().cpu().numpy()


class RedimNetEmbedder:
    """载入一次、反复调用的便捷封装。"""

    def __init__(self, device=None, **kw):
        self.model, self.dev = load_redimnet(device, **kw)
        self.decoder = Decoder()

    def from_src(self, src: str, t0: float, dur: float) -> np.ndarray:
        return self.from_wave(self.decoder.clip(src, t0, dur))

    def from_wave(self, x: np.ndarray) -> np.ndarray:
        return embed_redimnet(self.model, self.dev, x)


def load_funasr(model_id: str, device: str = "cuda:0"):
    from funasr import AutoModel
    return AutoModel(model=model_id, device=device, hub="ms",
                     disable_update=True, disable_pbar=True, log_level="ERROR")


def embed_funasr(model, wav_path: Path) -> np.ndarray:
    res = model.generate(input=str(wav_path), extract_embedding=True)[0]
    e = res["spk_embedding"]
    return np.asarray(e.squeeze().cpu() if hasattr(e, "cpu") else e, dtype=np.float64)


def l2norm(E: np.ndarray) -> np.ndarray:
    """按行 L2 归一化；单向量也适用。"""
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 1:
        return E / (np.linalg.norm(E) + 1e-9)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def anchor_cosines(En: np.ndarray, pos_of: dict[int, int]) -> str:
    """早期人工锚点的两两余弦。锚点内部一致性高只说明它们同源，
    不代表它们都是奶龙——实测这 4 个锚点本身已跨说话人。"""
    A = config.ANCHORS
    return "  ".join(f"#{a}-#{b}={En[pos_of[a]] @ En[pos_of[b]]:.3f}"
                     for i, a in enumerate(A) for b in A[i + 1:])
