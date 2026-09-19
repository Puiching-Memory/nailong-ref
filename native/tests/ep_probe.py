"""Probe: can ORT 1.30 load the standalone NVIDIA TensorRT-RTX plugin EP?

No system changes: everything runs out of native/third_party/trt-rtx-ep.
"""

import os
import sys

EP_DIR = r"C:\workspace\github\nailong-ref\native\third_party\trt-rtx-ep"
EP_NAME = "NvTensorRTRTXExecutionProvider"
EP_DLL = os.path.join(EP_DIR, "onnxruntime_providers_nv_tensorrt_rtx.dll")

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"


def show_devices(label):
    print(f"--- ep devices {label} ---")
    for d in ort.get_ep_devices():
        ep = getattr(d, "ep_name", None)
        hw = getattr(d, "hardware_device", None)
        hwtype = getattr(hw, "type", "?") if hw is not None else "?"
        vendor = getattr(hw, "vendor", "?") if hw is not None else "?"
        print(f"    {ep:<34} hw={hwtype:<10} vendor={vendor}")


for p in (EP_DIR, CUDA_BIN):
    if os.path.isdir(p):
        os.add_dll_directory(p)
        print(f"add_dll_directory OK: {p}")
    else:
        print(f"!! missing dir: {p}")

import onnxruntime as ort

print()
print("onnxruntime:", ort.__version__)
print("build info :", ort.get_build_info() if hasattr(ort, "get_build_info") else "(n/a)")
print("available providers:", ort.get_available_providers())
print()

show_devices("BEFORE register")

if not os.path.isfile(EP_DLL):
    print(f"!! EP dll missing: {EP_DLL}")
    sys.exit(1)

print()
print(f"registering '{EP_NAME}' from {EP_DLL}")
try:
    ort.register_execution_provider_library(EP_NAME, EP_DLL)
    print("register_execution_provider_library -> OK")
except Exception as ex:
    print(f"register FAILED: {type(ex).__name__}: {ex}")
    sys.exit(2)

print()
show_devices("AFTER register")

trt = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", "") == EP_NAME]
print()
if not trt:
    print(f"RESULT: plugin loaded but '{EP_NAME}' exposes NO device -> not usable")
    sys.exit(3)

print(f"RESULT: '{EP_NAME}' registered and exposes {len(trt)} device(s) -> USABLE")
