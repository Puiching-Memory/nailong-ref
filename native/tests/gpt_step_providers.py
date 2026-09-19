"""gpt_step 到底坏在哪：是图，还是 TRT-RTX？

把同一张图放到 ORT 能加载的每个后端上跑，逐个与 CPU 比。
- 只有 TRT-RTX 对不上  -> 图是好的，缺陷在 TRT-RTX（经典 TensorRT 大概率没这问题）
- CUDA EP 也对不上    -> 图/导出本身有问题（经典 TensorRT 会继承同样的毛病，得先修图）

判据用相对误差，且整数索引输出必须排除——top-50 边界 near-tie 会让索引交换，
那是舍入现象不是错误。
"""

import os
import sys

import numpy as np

ORT_LIB = r"C:\workspace\github\nailong-ref\native\third_party\onnxruntime-win-x64-gpu_cuda13-1.30.0\lib"
EP_DIR = r"C:\workspace\github\nailong-ref\native\third_party\trt-rtx-ep"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
ONNX_DIR = os.environ.get("NAILONG_ONNX_DIR", r"C:\workspace\github\nailong-ref\native\gsv\onnx_out")
MODEL = os.environ.get("NAILONG_MODEL", "gpt_step")

for _d in (ORT_LIB, EP_DIR, CUDA_BIN):
    if os.path.isdir(_d):
        os.add_dll_directory(_d)

import onnxruntime as ort

DIM_BY_NAME = {"batch_size": 1, "one": 1, "seq_len": 32,
               "text_len": 64, "prompt_len": 64, "sem_len": 64, "ref_len": 64}
DYNAMIC_FILL = 64


def resolve(shape):
    return [d if isinstance(d, int) else DIM_BY_NAME.get(d, DYNAMIC_FILL) for d in shape]


def make_feed(sess, rng):
    feed = {}
    for inp in sess.get_inputs():
        name = inp.name.lower()
        shape = resolve(inp.shape)
        if "int" not in inp.type:
            dtype = np.float16 if "float16" in inp.type else np.float32
            feed[inp.name] = rng.standard_normal(shape).astype(dtype)
        elif name in ("idx", "samples"):
            feed[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            feed[inp.name] = np.full(shape, 64, dtype=np.int64)
    return feed


def build_session(path, providers):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if providers == ["CPUExecutionProvider"]:
        return ort.InferenceSession(path, opts, providers=providers)
    return ort.InferenceSession(path, opts, providers=providers, enable_fallback=0)


def compare(tag, ref_outs, outs, outputs):
    print(f"  {tag}")
    worst = 0.0
    for spec, ref, got in zip(outputs, ref_outs, outs):
        is_index = np.asarray(ref).dtype.kind in "iu"
        a = np.asarray(ref, dtype=np.float64)
        b = np.asarray(got, dtype=np.float64)
        delta = float(np.max(np.abs(a - b))) if a.size else 0.0
        scale = float(np.max(np.abs(a))) or 1.0
        note = "  [index output - excluded]" if is_index else ""
        print(f"    {spec.name:<16} {str(a.shape):<22} max|d|={delta:.3e} "
              f"refmax={scale:.3e} rel={delta / scale:.3e}{note}")
        if not is_index:
            worst = max(worst, delta / scale)
    print(f"    worst rel = {worst:.3e}  ->  {'OK' if worst < 1e-2 else 'DIVERGENT'}")
    return worst


path = os.path.join(ONNX_DIR, f"{MODEL}.onnx")
print(f"model: {path}")
print(f"onnxruntime {ort.__version__}\n")

cpu = build_session(path, ["CPUExecutionProvider"])
rng = np.random.default_rng(0)
feed = make_feed(cpu, rng)

ref = cpu.run(None, feed)
print(f"CPUExecutionProvider  providers={cpu.get_providers()}")

for tag, providers in [("CUDAExecutionProvider", ["CUDAExecutionProvider"]),
                       ("TRT-RTX", ["NvTensorRTRTXExecutionProvider", "CPUExecutionProvider"])]:
    print()
    try:
        if providers[0].startswith("NvTensor"):
            ort.register_execution_provider_library(
                "NvTensorRTRTXExecutionProvider",
                os.path.join(EP_DIR, "onnxruntime_providers_nv_tensorrt_rtx.dll"))
            devices = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", "") == "NvTensorRTRTXExecutionProvider"]
            if not devices:
                print(f"{tag}: EP 未暴露设备，跳过")
                continue
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            cache = r"C:\workspace\github\nailong-ref\native\gsv\trt_cache"
            os.makedirs(cache, exist_ok=True)
            opts.add_provider_for_devices(devices, {"nv_runtime_cache_path": cache})
            sess = ort.InferenceSession(path, opts, enable_fallback=0)
        else:
            sess = build_session(path, providers)
    except Exception as exc:  # noqa: BLE001 - 诊断脚本，失败原因就是要看的东西
        print(f"{tag}: 加载/运行失败 -> {type(exc).__name__}: {str(exc)[:300]}")
        continue

    print(f"{tag}  providers={sess.get_providers()}")
    try:
        outs = sess.run(None, feed)
    except Exception as exc:  # noqa: BLE001
        print(f"  运行失败 -> {type(exc).__name__}: {str(exc)[:300]}")
        continue
    compare("vs CPU", ref, outs, cpu.get_outputs())
