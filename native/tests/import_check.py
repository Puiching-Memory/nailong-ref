"""Verify the import chain export_onnx.py depends on.

Inserts the GPT-SoVITS repo root on sys.path so this works regardless of CWD.
"""

import importlib
import os
import sys

REPO = r"C:\workspace\github\GPT-SoVITS_minimal_inference"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

CHECKS = [
    ("torch", None), ("torchaudio", None), ("transformers", None), ("numpy", None),
    ("pytorch_lightning", None), ("torchmetrics", None), ("librosa", None),
    ("einops", None), ("tqdm", None), ("yaml", None), ("scipy", None),
    ("onnx", None), ("onnxruntime", None),
    ("GPT_SoVITS.process_ckpt", "load_sovits_new,get_sovits_version_from_path_fast"),
    ("GPT_SoVITS.feature_extractor.cnhubert", None),
    ("GPT_SoVITS.text", "_symbol_to_id_v2"),
    ("GPT_SoVITS.AR.models.t2s_lightning_module", "Text2SemanticLightningModule"),
    ("GPT_SoVITS.module.models", "SynthesizerTrn"),
    ("onnx_validation", None),
]

failed = []
for mod, attrs in CHECKS:
    try:
        m = importlib.import_module(mod)
        extra = ""
        if attrs:
            for a in attrs.split(","):
                getattr(m, a)
            extra = f"  ({attrs})"
        print(f"  OK    {mod}{extra}")
    except Exception as ex:
        print(f"  FAIL  {mod}: {type(ex).__name__}: {str(ex)[:160]}")
        failed.append(mod)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    raise SystemExit(1)
print("ALL IMPORTS OK")

import torch
import onnx
print(f"  torch={torch.__version__}  onnx={onnx.__version__}")
