"""Is the FP16 model itself faithful? Compare CPU(FP32 model) against CPU(FP16 model)
on one identical seeded input set, so a GPU EP is never blamed for a broken graph.
"""

import os
import sys

import numpy as np
import onnxruntime as ort

FP32_DIR = r"C:\workspace\github\nailong-ref\native\gsv\onnx_out"
FP16_DIR = r"C:\workspace\github\nailong-ref\native\gsv\onnx_fp16"

DIM_BY_NAME = {"batch_size": 1, "one": 1, "seq_len": 32,
               "text_len": 64, "prompt_len": 64, "sem_len": 64, "ref_len": 64}
DYNAMIC_FILL = 64


def sess(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def resolve(shape):
    return [d if isinstance(d, int) else DIM_BY_NAME.get(d, DYNAMIC_FILL) for d in shape]


def make_feed(s):
    """Seeded input set, sized from the FP32 model's own signature."""
    rng = np.random.default_rng(0)
    feed = {}
    for i in s.get_inputs():
        n = i.name.lower()
        shp = resolve(i.shape)
        if "int" not in i.type:
            feed[i.name] = rng.standard_normal(shp).astype(np.float32)
        elif "token_type" in n:
            feed[i.name] = np.zeros(shp, dtype=np.int64)
        elif "mask" in n:
            feed[i.name] = np.ones(shp, dtype=np.int64)
        elif n in ("idx", "samples"):
            feed[i.name] = np.zeros(shp, dtype=np.int64)
        else:
            feed[i.name] = rng.integers(0, 64, size=shp).astype(np.int64)
    return feed


NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "gpt_step", "gpt_step_static", "gpt_encoder"]

for name in NAMES:
    a = sess(os.path.join(FP32_DIR, f"{name}.onnx"))
    b = sess(os.path.join(FP16_DIR, f"{name}.onnx"))
    feed32 = make_feed(a)

    want = {i.name: i.type for i in b.get_inputs()}
    feed16 = {k: (v.astype(np.float16) if "float16" in want[k] and v.dtype.kind == "f" else v)
              for k, v in feed32.items()}

    out32 = a.run(None, feed32)
    out16 = b.run(None, feed16)

    print("=" * 74)
    print(f"### {name}   CPU(FP32)  vs  CPU(FP16)")
    worst = 0.0
    for o, x, y in zip(a.get_outputs(), out32, out16):
        # Integer index outputs sit on a top-k boundary where fp16 rounding swaps
        # membership outright, so relative error is meaningless for them.
        is_index = np.asarray(x).dtype.kind in "iu"
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        if x.shape != y.shape:
            print(f"  {o.name:<16} SHAPE MISMATCH {x.shape} vs {y.shape}")
            continue
        d = float(np.max(np.abs(x - y))) if x.size else 0.0
        ref = float(np.max(np.abs(x))) or 1.0
        note = "  [index output - excluded]" if is_index else ""
        print(f"  {o.name:<16} {str(x.shape):<22} max|d|={d:.3e} refmax={ref:.3e} rel={d / ref:.3e}{note}")
        if not is_index:
            worst = max(worst, d / ref)
    verdict = "FAITHFUL" if worst < 1e-2 else "BROKEN BY CONVERSION"
    print(f"  worst rel = {worst:.3e}  ->  {verdict}")
