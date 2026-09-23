"""提取 ERes2NetV2 句级声纹嵌入，默认使用 CUDA。

视觉校准后它是生产主模型；ReDimNet 保留为独立审计旁证。

用法::

    nailong embed-eres2net
    nailong embed-eres2net out.npy --device cuda:0

模型：iic/speech_eres2netv2_sv_zh-cn_16k-common（ModelScope）。首次运行会下载，
之后从 ModelScope 缓存读取。需要 ``uv sync --extra asr``。
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

from .. import audio, config, manifests
from ..device import checked_device

MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
CKPT_NAME = "pretrained_eres2netv2.ckpt"


def _model_dir() -> Path:
    cache = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
    matches = [path for path in glob.glob(str(cache / "**" / CKPT_NAME), recursive=True)
               if MODEL_ID.split("/")[-1] in path]
    if matches:
        return Path(matches[0]).parent
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit("缺少 modelscope；请先运行 uv sync --extra asr") from exc
    return Path(snapshot_download(MODEL_ID))


class ERes2NetV2Embedder:
    """Load ERes2NetV2 once and embed many 16 kHz mono waveforms."""

    def __init__(self, device: str = "cuda:0"):
        try:
            import torch
            from modelscope.models.audio.sv.ERes2NetV2 import ERes2NetV2
        except ImportError as exc:
            raise SystemExit("缺少 torch/torchaudio/modelscope；请先运行 uv sync --extra asr") from exc

        self.device = device
        self.torch = torch
        self.model = ERes2NetV2(feat_dim=80, embed_dim=192, baseWidth=26, scale=2, expansion=2)
        state = torch.load(_model_dir() / CKPT_NAME, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        import torch.nn.functional as nnf
        import torchaudio.compliance.kaldi as kaldi

        wav = self.torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0) * (1 << 15)
        feat = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10,
                           sample_frequency=config.SR_MODEL, dither=0.0)
        feat = (feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            value = nnf.normalize(self.model(feat).float(), dim=-1)
        return value.cpu().numpy().reshape(-1)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    device = "cuda:0"
    if "--device" in argv:
        at = argv.index("--device")
        try:
            device = argv[at + 1]
        except IndexError as exc:
            raise SystemExit("--device 后需要 cpu / cuda:0") from exc
        del argv[at:at + 2]
    device = checked_device(device)
    out = Path(argv[0]) if argv else config.EMB_ERES2NETV2

    rows = manifests.read(config.UTT_MANIFEST)
    previous = np.load(out) if out.exists() else np.empty((0, 192), dtype=np.float32)
    if previous.ndim != 2 or previous.shape[1] != 192 or len(previous) > len(rows):
        raise SystemExit(f"现有嵌入 {previous.shape} 与 {len(rows)} 条句子不兼容")
    pending = rows[len(previous):]
    if not pending:
        print(f"没有待追加的新句子；嵌入保持 {previous.shape} -> {config.rel(out)}")
        return 0
    embedder = ERes2NetV2Embedder(device)
    decoder = audio.Decoder(config.SR_MODEL)
    values = []
    for number, row in enumerate(pending, 1):
        values.append(embedder(decoder.clip(row["src"], float(row["t0"]), float(row["dur"]))))
        if number % 10 == 0 or number == len(pending):
            print(f"ERes2NetV2 {device}: {number}/{len(pending)}", flush=True)

    matrix = np.vstack([previous, np.asarray(values, dtype=np.float32)])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, matrix)
    print(f"嵌入 {matrix.shape} -> {config.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
