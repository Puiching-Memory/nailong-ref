"""FP16-ify exported ONNX models with the reference project's own per-model policy
(onnx_to_fp16.py: op sensitivity block list + shape-inference type repair + onnxsim).

Usage: make_fp16.py [comma,separated,names]
"""

import os
import sys
import time

REPO = r"C:\workspace\github\GPT-SoVITS_minimal_inference"
sys.path.insert(0, REPO)

SRC = r"C:\workspace\github\nailong-ref\native\gsv\onnx_out"
DST = r"C:\workspace\github\nailong-ref\native\gsv\onnx_fp16"

NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "gpt_step", "gpt_step_static", "gpt_encoder"]

import onnx_to_fp16 as F

os.makedirs(DST, exist_ok=True)
for name in NAMES:
    src = os.path.join(SRC, name + ".onnx")
    dst = os.path.join(DST, name + ".onnx")
    print("=" * 70)
    print(f"### {name}")
    t = time.perf_counter()
    F.optimize_single_model(src, dst)
    mb = lambda p: os.path.getsize(p) / 1e6
    print(f"  {mb(src):.1f} MB -> {mb(dst):.1f} MB  ({time.perf_counter() - t:.1f}s)")
