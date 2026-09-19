"""Run a GPT-SoVITS-shaped synthetic ONNX graph on three EPs and compare.

Op coverage mirrors what GPT-SoVITS actually needs:
  Conv1d / ConvTranspose1d  -> sovits.onnx VITS decoder (the risky ones for TensorRT)
  MatMul / LayerNormalization / Softmax / Erf-Erf-based GELU / Transpose
  -> gpt_step.onnx transformer half

Compares CPU vs CUDAExecutionProvider vs NvTensorRTRTXExecutionProvider.
"""

import os
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EP_DIR = r"C:\workspace\github\nailong-ref\native\third_party\trt-rtx-ep"
EP_NAME = "NvTensorRTRTXExecutionProvider"
EP_DLL = os.path.join(EP_DIR, "onnxruntime_providers_nv_tensorrt_rtx.dll")
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "_gsv_shape.onnx")
CACHE_DIR = os.path.join(HERE, "_trt_engine_cache")

B, C, L = 1, 64, 128          # input [B, C, L]
KT = 4                         # convtranspose kernel
WARMUP, ITERS = 3, 20


def build_model(path, conv=True):
    """conv=True  -> Conv1d + ConvTranspose1d head (mirrors sovits.onnx decoder)
       conv=False -> MatMul-only head      (mirrors gpt_step.onnx, needs no cuDNN)

    Both variants end at a tensor named 'tt' feeding a shared transformer tail.
    """
    rng = np.random.default_rng(0)

    def w(name, shape):
        return numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32) * 0.2, name)

    def const(name, value):
        return numpy_helper.from_array(np.array(value, dtype=np.float32), name)

    inits = [
        w("W3", (32, 32)), w("B3", (32,)), w("W4", (32, 32)),
        const("RCP_SQRT2", 0.70710678), const("HALF", 0.5), const("ONE", 1.0),
    ]
    tail = [
        helper.make_node("MatMul", ["tt", "W3"], ["mm"]),
        helper.make_node("Add", ["mm", "B3"], ["mmb"]),
        helper.make_node("Mul", ["mmb", "RCP_SQRT2"], ["scaled"]),
        helper.make_node("Erf", ["scaled"], ["erfed"]),
        helper.make_node("Add", ["erfed", "ONE"], ["erf1"]),
        helper.make_node("Mul", ["mmb", "erf1"], ["g1"]),
        helper.make_node("Mul", ["g1", "HALF"], ["gelu"]),
        helper.make_node("Softmax", ["gelu"], ["sm"], axis=-1),
        helper.make_node("MatMul", ["sm", "W4"], ["y"]),
    ]

    if conv:
        inits += [w("W1", (64, C, 3)), w("B1", (64,)),
                  w("LN_S", (L,)), w("LN_B", (L,)),
                  w("W2", (64, 32, KT)), w("B2", (32,))]
        head = [
            helper.make_node("Conv", ["x", "W1", "B1"], ["conv"], kernel_shape=[3], pads=[1, 1]),
            helper.make_node("LayerNormalization", ["conv", "LN_S", "LN_B"], ["ln"], axis=-1),
            helper.make_node("ConvTranspose", ["ln", "W2", "B2"], ["deconv"],
                             kernel_shape=[KT], strides=[2], pads=[1, 1]),
            helper.make_node("Transpose", ["deconv"], ["tt"], perm=[0, 2, 1]),
        ]
        out_shape, gname = [B, 2 * L, 32], "gsv_conv"
    else:
        inits += [w("W1", (L, L)), w("LN_S", (L,)), w("LN_B", (L,)), w("W2", (C, 32))]
        head = [
            helper.make_node("MatMul", ["x", "W1"], ["h0"]),
            helper.make_node("LayerNormalization", ["h0", "LN_S", "LN_B"], ["h1"], axis=-1),
            helper.make_node("Transpose", ["h1"], ["h2"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["h2", "W2"], ["tt"]),
        ]
        out_shape, gname = [B, L, 32], "gsv_noconv"

    graph = helper.make_graph(
        head + tail, gname,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, C, L])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    inferred = onnx.shape_inference.infer_shapes(m)
    dims = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in inferred.graph.output}
    print(f"  built {gname}  opset=17  output={dims}")
    onnx.save(m, path)
    return path


def bench(sess, x, tag):
    print(f"    [{tag}] session providers: {sess.get_providers()}")
    for _ in range(WARMUP):
        sess.run(None, {"x": x})
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter()
        out = sess.run(None, {"x": x})[0]
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    print(f"  {tag:<28} median={np.median(ts):7.2f} ms   min={ts.min():7.2f}   p90={np.percentile(ts,90):7.2f}")
    return out


# SKIP_CUDA_BIN=1 drops the CUDA toolkit dir from the DLL search path, which
# tests whether the TRT-RTX runtime really needs the toolkit at runtime.
_dll_dirs = [EP_DIR] if os.environ.get("SKIP_CUDA_BIN") == "1" else [EP_DIR, CUDA_BIN]
for p in _dll_dirs:
    if os.path.isdir(p):
        os.add_dll_directory(p)
        print(f"add_dll_directory: {p}")

if os.environ.get("SKIP_CUDA_BIN") == "1":
    print(f"CUDA bin dir SKIPPED: {CUDA_BIN}")
    print(f"  on PATH already? {CUDA_BIN.lower() in os.environ.get('PATH', '').lower()}")

import onnxruntime as ort

print(f"onnxruntime {ort.__version__}")
print(f"available providers: {ort.get_available_providers()}")
print()

def run_case(conv):
    label = ("Conv1d + ConvTranspose1d head  (sovits decoder shape)" if conv
             else "MatMul-only head                (gpt_step shape, no cuDNN ops)")
    path = os.path.join(HERE, "_gsv_conv.onnx" if conv else "_gsv_noconv.onnx")
    print("=" * 78)
    print(f"### {label}")
    print("=" * 78)
    build_model(path, conv=conv)

    x = np.random.default_rng(1).standard_normal((B, C, L)).astype(np.float32)
    print(f"  benchmark warmup={WARMUP} iters={ITERS}")

    results = {}

    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        results["cpu"] = bench(s, x, "CPUExecutionProvider")
    except Exception as ex:
        print(f"  {'CPUExecutionProvider':<28} FAILED: {type(ex).__name__}: {str(ex)[:200]}")

    try:
        so = ort.SessionOptions()
        so.log_severity_level = 1
        s = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider"])
        results["cuda"] = bench(s, x, "CUDAExecutionProvider")
    except Exception as ex:
        msg = str(ex)
        hint = "  [needs cuDNN]" if "cudnn" in msg.lower() else ""
        print(f"  {'CUDAExecutionProvider':<28} FAILED{hint}: {type(ex).__name__}: {msg[:200]}")

    try:
        dev = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", "") == EP_NAME]
        if not dev:
            raise RuntimeError("EP exposes no device")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.add_provider_for_devices([dev[0]], {"nv_runtime_cache_path": CACHE_DIR})
        t = time.perf_counter()
        s = ort.InferenceSession(path, so)
        print(f"  (engine build + load: {(time.perf_counter() - t) * 1e3:.0f} ms)")
        results["trt"] = bench(s, x, EP_NAME)
    except Exception as ex:
        print(f"  {EP_NAME:<28} FAILED: {type(ex).__name__}: {str(ex)[:200]}")

    ref = results.get("cpu")
    if ref is not None:
        print("  max |delta| vs CPU:")
        for k in ("cuda", "trt"):
            if k in results:
                d = float(np.max(np.abs(results[k] - ref)))
                print(f"    {k:<6} {d:.3e}  {'OK' if d < 1e-3 else 'DIVERGENT'}")
    print()


os.makedirs(CACHE_DIR, exist_ok=True)
ort.register_execution_provider_library(EP_NAME, EP_DLL)
print(f"registered plugin EP: {EP_NAME}")
print()

for conv in (False, True):
    run_case(conv)
