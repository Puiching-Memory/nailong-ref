"""Fetch everything export_onnx.py needs. Keeps the HuggingFace repo layout so
paths are self-documenting.

Usage:  uv run --no-project --with huggingface_hub python fetch_gsv_models.py
"""

import os
import sys

from huggingface_hub import hf_hub_download

ROOT = r"C:\workspace\github\nailong-ref\native\gsv"

FILES = [
    # fine-tuned NaiLong weights (pengyichen)
    ("pengyichen/NaiLong-Voice-Clone", "GPT_weights/NaiLong-e30.ckpt"),
    ("pengyichen/NaiLong-Voice-Clone", "SoVITS_weights/NaiLong_e20_s720.pth"),
    # base frontend/feature models + speaker verification (lj1995 = official GPT-SoVITS mirror)
    ("lj1995/GPT-SoVITS", "chinese-hubert-base/pytorch_model.bin"),
    ("lj1995/GPT-SoVITS", "chinese-hubert-base/config.json"),
    ("lj1995/GPT-SoVITS", "chinese-hubert-base/preprocessor_config.json"),
    ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/pytorch_model.bin"),
    ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/config.json"),
    ("lj1995/GPT-SoVITS", "chinese-roberta-wwm-ext-large/tokenizer.json"),
    ("lj1995/GPT-SoVITS", "sv/pretrained_eres2netv2w24s4ep4.ckpt"),
]


def main():
    os.makedirs(ROOT, exist_ok=True)
    total = len(FILES)
    for i, (repo, fn) in enumerate(FILES, 1):
        dest = os.path.join(ROOT, fn.replace("/", os.sep))
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"[{i}/{total}] SKIP (exists) {fn}  ({os.path.getsize(dest)/1e6:.1f} MB)", flush=True)
            continue
        print(f"[{i}/{total}] {repo} :: {fn}", flush=True)
        try:
            p = hf_hub_download(repo_id=repo, filename=fn, local_dir=ROOT)
            print(f"          -> {os.path.getsize(p)/1e6:.1f} MB  {p}", flush=True)
        except Exception as ex:
            print(f"          FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            return 1
    print("\nALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
