"""Run the real exported GPT-SoVITS ONNX models on TensorRT-RTX and compare with CPU.

Feeds are generated from each model's own input signature, so this works without
hand-writing shapes for every graph. Dynamic dims are filled with DYNAMIC_FILL and
an optimization profile is derived from them when TensorRT needs one.
"""

import os
import sys
import time

import numpy as np

ONNX_DIR = os.environ.get(
    "NAILONG_ONNX_DIR", r"C:\workspace\github\nailong-ref\native\gsv\onnx_out")
EP_DIR = r"C:\workspace\github\nailong-ref\native\third_party\trt-rtx-ep"
EP_NAME = "NvTensorRTRTXExecutionProvider"
EP_DLL = os.path.join(EP_DIR, "onnxruntime_providers_nv_tensorrt_rtx.dll")
CACHE_DIR = r"C:\workspace\github\nailong-ref\native\gsv\trt_cache"

MODELS = ["ssl", "bert", "vq_encoder", "gpt_encoder", "gpt_step", "gpt_step_static",
          "sovits", "spectrogram", "sv_embedding"]

SKIP, ONLY, NO_CACHE = set(), set(), False
for _a in sys.argv[1:]:
    if _a == "--no-cache":
        NO_CACHE = True
    elif _a.startswith("--skip="):
        SKIP.update(_a.split("=", 1)[1].split(","))
    elif _a.startswith("--only="):
        ONLY.update(_a.split("=", 1)[1].split(","))

DYNAMIC_FILL = 64
WARMUP, ITERS = 2, 10

if os.path.isdir(EP_DIR):
    os.add_dll_directory(EP_DIR)

import onnxruntime as ort

print(f"onnxruntime {ort.__version__}")
ort.register_execution_provider_library(EP_NAME, EP_DLL)
TRT_DEV = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", "") == EP_NAME]
if not TRT_DEV:
    print("!! TRT-RTX EP exposes no device")
    sys.exit(2)
print(f"EP device ready: {TRT_DEV[0].ep_name}")
if not NO_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True)
print()


# Value used for dynamic dims, keyed by the ONNX dim_param NAME. Filling every
# dynamic dim with one number is wrong: batch_size must stay 1 or the KV-cache
# profile explodes (it demanded 4.19 GB and blew up the shape math).
DIM_BY_NAME = {"batch_size": 1, "one": 1, "seq_len": 32,
               "text_len": 64, "prompt_len": 64, "sem_len": 64, "ref_len": 64}

# Audio models need enough samples for their strided conv / STFT stacks to stay
# valid (a 64-sample clip makes hubert's conv output a negative dimension).
FILL = {"ssl": 16000, "spectrogram": 32000, "sv_embedding": 48000}


def fill_for(name):
    return FILL.get(name, DYNAMIC_FILL)


def resolve_shape(name, shape):
    """Concrete dims for a model input, resolving dim_param names semantically."""
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append(d)
        elif d == "time":
            out.append(fill_for(name))
        else:
            out.append(DIM_BY_NAME.get(d, DYNAMIC_FILL))
    return out


def feed_for(sess, name):
    """Generate plausible inputs from the model's own signature - random ints for
    token_type_ids/masks trip real bounds checks otherwise."""
    feed = {}
    for inp in sess.get_inputs():
        n = inp.name.lower()
        shp = resolve_shape(name, inp.shape)
        if "int" not in inp.type:
            dt = np.float16 if "float16" in inp.type else np.float32
            feed[inp.name] = np.random.randn(*shp).astype(dt)
        elif "token_type" in n:
            feed[inp.name] = np.zeros(shp, dtype=np.int64)
        elif "mask" in n:
            feed[inp.name] = np.ones(shp, dtype=np.int64)
        elif n in ("idx",):
            feed[inp.name] = np.zeros(shp, dtype=np.int64)
        else:
            feed[inp.name] = np.random.randint(0, 100, size=shp).astype(np.int64)
    return feed


def make_feed(name, sess):
    """gpt_step consumes the KV cache produced by gpt_encoder, so chain the two
    instead of inventing cache contents (wrong x_len/y_len blow up Concat)."""
    if not name.startswith("gpt_step"):
        return feed_for(sess, name)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc = ort.InferenceSession(os.path.join(ONNX_DIR, "gpt_encoder.onnx"), so,
                               providers=["CPUExecutionProvider"])
    _, _, k_cache, v_cache, x_len, y_len = enc.run(None, feed_for(enc, "gpt_encoder"))
    print(f"  (from gpt_encoder: k_cache{k_cache.shape} x_len={x_len} y_len={y_len})")

    feed = {}
    for inp in sess.get_inputs():
        n = inp.name
        want16 = "float16" in inp.type
        if n == "k_cache":
            feed[n] = k_cache.astype(np.float16) if want16 else k_cache
        elif n == "v_cache":
            feed[n] = v_cache.astype(np.float16) if want16 else v_cache
        elif n == "x_len":
            feed[n] = x_len.astype(np.int64)
        elif n == "y_len":
            feed[n] = y_len.astype(np.int64)
        elif n == "idx":
            feed[n] = np.zeros([1], dtype=np.int64)
        elif n == "samples":
            feed[n] = np.zeros([1, 1], dtype=np.int64)
        else:
            feed[n] = np.zeros(resolve_shape(name, inp.shape), dtype=np.int64)
    return feed


def profile_strings(name, sess):
    """Explicit single-shape (min=opt=max) profiles for every input. Supplied even
    for fully static graphs: TRT-RTX otherwise errors with "No explicit or implicit
    shapes were provided for dynamic shape inputs".
    """
    parts = []
    for inp in sess.get_inputs():
        parts.append(f"{inp.name}:{'x'.join(map(str, resolve_shape(name, inp.shape)))}")
    if not parts:
        return None
    joined = ";".join(parts)
    return joined, joined, joined


def timed(sess, feed, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        out = sess.run(None, feed)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        out = sess.run(None, feed)
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return out, float(np.median(ts)), float(ts.min())


for name in MODELS:
    if name in SKIP or (ONLY and name not in ONLY):
        continue
    path = os.path.join(ONNX_DIR, f"{name}.onnx")
    print("=" * 78)
    print(f"### {name}.onnx")
    print("=" * 78)
    if not os.path.isfile(path):
        print("  MISSING\n")
        continue

    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        cpu = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    except Exception as ex:
        print(f"  LOAD FAILED: {type(ex).__name__}: {str(ex)[:200]}\n")
        continue

    dyn = [i.name for i in cpu.get_inputs() if any(not isinstance(d, int) for d in i.shape)]
    print(f"  dynamic-axis inputs: {dyn if dyn else '(none - fully static)'}")

    try:
        feed = make_feed(name, cpu)
    except Exception as ex:
        print(f"  FEED FAILED: {type(ex).__name__}: {str(ex)[:200]}\n")
        continue

    try:
        ref, cpu_ms, cpu_min = timed(cpu, feed)
        print(f"  CPUExecutionProvider   median={cpu_ms:8.2f} ms  min={cpu_min:8.2f}")
    except Exception as ex:
        print(f"  CPU FAILED: {type(ex).__name__}: {str(ex)[:200]}\n")
        continue

    # Control: models with in-graph RNG (VITS noise injection) differ run to run,
    # which makes a cross-EP delta meaningless. Measure the model's own spread.
    ref2 = cpu.run(None, feed)
    _sd = [float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
           for a, b in zip(ref2, ref) if a.shape == b.shape]
    self_delta = max(_sd) if _sd else float("nan")
    stochastic = self_delta > 1e-5
    print(f"  CPU self-consistency (run twice) = {self_delta:.3e}"
          + ("   -> STOCHASTIC model, cross-EP delta is not comparable" if stochastic else ""))

    prof = profile_strings(name, cpu)

    # attempt 1: no profile options;  attempt 2: with derived profile
    for label, opts in (("no-profile", {}), ("with-profile", {
            "nv_profile_min_shapes": prof[0],
            "nv_profile_opt_shapes": prof[1],
            "nv_profile_max_shapes": prof[2]} if prof else {})):
        if label == "with-profile" and not prof:
            continue
        o = dict(opts)
        if not NO_CACHE:
            o["nv_runtime_cache_path"] = CACHE_DIR
        try:
            sopt = ort.SessionOptions()
            sopt.log_severity_level = 3
            sopt.add_provider_for_devices([TRT_DEV[0]], o)
            t = time.perf_counter()
            # enable_fallback=0 makes an EP failure raise instead of silently
            # running on CPU, which would make the "speedup" below a lie.
            trt = ort.InferenceSession(path, sopt, enable_fallback=0)
            build_ms = (time.perf_counter() - t) * 1e3
            print(f"  [{label}] providers: {trt.get_providers()}")
            out, trt_ms, trt_min = timed(trt, feed)
            rels = []
            print("    per-output:")
            for o, a, b in zip(trt.get_outputs(), out, ref):
                if a.shape != b.shape:
                    print(f"      {o.name:<16} SHAPE MISMATCH {a.shape} vs {b.shape}")
                    rels.append(float("inf"))
                    continue
                d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                scale = float(np.max(np.abs(b)))
                rel = d / scale if scale > 0 else float("inf")
                rels.append(rel)
                print(f"      {o.name:<16} {str(a.shape):<18} max|d|={d:.3e}  "
                      f"refmax={scale:.3e}  rel={rel:.2e}")
            # Relative error is the meaningful metric - an absolute threshold
            # misjudges wide-range tensors (sv_embedding reached 7.5 with 2.6e-3 rel).
            worst = max(rels) if rels else float("nan")
            if stochastic:
                verdict = "STOCHASTIC (not comparable)"
            else:
                verdict = "OK" if worst < 1e-2 else "DIVERGENT"
            print(f"  {EP_NAME} median={trt_ms:8.2f} ms  min={trt_min:8.2f}  "
                  f"speedup={cpu_ms / trt_ms:5.2f}x")
            print(f"    engine build+load = {build_ms:8.1f} ms")
            print(f"    worst relative error = {worst:.3e}  {verdict}")
            break
        except Exception as ex:
            msg = " ".join(str(ex).split())
            print(f"  [{label}] FAILED: {type(ex).__name__}: {msg[:400]}")
    print()
