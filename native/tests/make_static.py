"""Rewrite gpt_step.onnx with fully static input shapes.

TensorRT-RTX overflows its execution-context memory math on gpt_step's dynamic
axes (int64 underflow), so the engine cannot be executed. Our pipeline always
runs batch=1 against a fixed max_len KV cache, so a static graph is both a valid
workaround and the shape regime we actually use.
"""

import os
import sys

import onnx

ONNX_DIR = r"C:\workspace\github\nailong-ref\native\gsv\onnx_out"
SRC = os.path.join(ONNX_DIR, "gpt_step.onnx")
DST = os.path.join(ONNX_DIR, "gpt_step_static.onnx")

DIM_BY_NAME = {"batch_size": 1, "one": 1, "time": 16000, "seq_len": 32,
               "text_len": 64, "prompt_len": 64, "sem_len": 64, "ref_len": 64}


def fix_dims(tensor):
    changed = []
    for d in tensor.type.tensor_type.shape.dim:
        if d.HasField("dim_param") and d.dim_param:
            name = d.dim_param
            d.ClearField("dim_param")
            d.dim_value = DIM_BY_NAME.get(name, 1)
            changed.append(f"{name}->{d.dim_value}")
    return changed


print(f"loading {SRC} ...", flush=True)
m = onnx.load(SRC)

n = 0
for t in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
    got = fix_dims(t)
    if got:
        print(f"  {t.name:<16} {' '.join(got)}")
        n += len(got)

print(f"fixed {n} dynamic dims", flush=True)
onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"saved {DST}  ({os.path.getsize(DST)/1e6:.1f} MB)", flush=True)

# verify no dim_param survived
m2 = onnx.load(DST, load_external_data=False)
left = [f"{t.name}:{d.dim_param}"
        for t in list(m2.graph.input) + list(m2.graph.output)
        for d in t.type.tensor_type.shape.dim if d.HasField("dim_param")]
print("remaining dim_param:", left if left else "(none)")
sys.exit(0)
