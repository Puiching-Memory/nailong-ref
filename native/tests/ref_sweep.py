"""挑参考音频：同一句台词换不同参考合成，用 sv_embedding 排序。

零样本克隆里参考音频决定音色与韵律。这里没有播放设备，听不出差异，
所以用声纹相似度做**可复现**的排序：固定 probe 文本，换参考合成，
再算 cos(合成嵌入, 11 段参考嵌入) 的均值。

参考文本来自 ASR，有明显错字（如"没血染"应为"没血缘"），
prompt 文本与音频不匹配会拖累 gpt_encoder 的 prompt，所以顺带做一次改字对照。

跑法（用参考仓库的 venv，它才有 onnxruntime / torch）：
    python native/tests/ref_sweep.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tts"))

from nailong_tts import synth  # noqa: E402

PROBE = "我是奶龙，你好呀，今天也要开开心心的。"
OUT_DIR = REPO / "native" / "gsv" / "_sweep"

# (参考文件名, 参考文本, 说明)
CANDIDATES: list[tuple[str, str, str]] = [
    ("src_05_3.54-7.82.wav", "小七，你给我做的。怎么跟没血染一样啊？", "4.28s simA .815 当前默认"),
    ("src_05_3.54-7.82.wav", "小七，你给我做的。怎么跟没血缘一样啊？", "4.28s 同上但改错字"),
    ("src_03_0.12-7.34.wav", "嗯，森林下的雪。你说那是保力奴小七。你怎么被冻住了？", "7.22s simA .811"),
    ("src_01_0.14-6.70.wav", "我就大肚嘟。真的很酷酷。今天的脊重。依旧那么稳定呢。", "6.56s simA .772"),
    ("src_06_14.22-22.22.wav", "还好没超过3秒呃，我刚刚出的三水一听就被你吃了，不吃不就浪费了呢。", "8.00s simA .759"),
]


def ref_names() -> list[str]:
    with (REPO / "dataset" / "manifests" / "final_manifest.csv").open(encoding="utf-8") as f:
        return [r["file"] for r in csv.DictReader(f)]


def main() -> int:
    import onnxruntime

    names = ref_names()
    ref_paths = [REPO / "dataset" / "final" / n for n in names]

    so = onnxruntime.SessionOptions()
    so.log_severity_level = 3
    sv = onnxruntime.InferenceSession(
        str(synth.ONNX_DIR / "sv_embedding.onnx"), sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    sys.path.insert(0, str(REPO / "native" / "tests"))
    from voice_id_check import embed  # noqa: E402

    ref_mat = np.stack([embed(sv, p) for p in ref_paths])
    print(f"probe text : {PROBE}")
    print(f"references : {len(ref_paths)} clips\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 只建一次引擎；参考音频只是每次调用时的参数，改 ref_wav/ref_text 即可复用。
    backend = synth.Backend(ref_wav=str(REPO / "dataset" / "final" / CANDIDATES[0][0]),
                            ref_text=CANDIDATES[0][1])

    rows = []
    for i, (wav_name, ref_text, note) in enumerate(CANDIDATES, 1):
        out = OUT_DIR / f"{i:02d}_{wav_name}"
        backend.ref_wav = REPO / "dataset" / "final" / wav_name
        backend.ref_text = ref_text
        print(f"[{i}/{len(CANDIDATES)}] ref={wav_name} ({note})")
        audio = backend.say(PROBE, out)
        dur = len(audio) / backend.sample_rate

        # 必须走 16k：sv_embedding 是 16k 模型，直接喂合成的 32k 音频会让嵌入整体偏低
        # （实测把 0.67 压到 0.53），排序就不可信了。embed() 内部会重采样。
        v = embed(sv, out)
        sims = ref_mat @ v
        rows.append((i, wav_name, note, dur, float(sims.mean()), float(sims.max())))

    print("\n--- 排序（按与 11 段参考的平均余弦）---")
    print(f"{'#':<3}{'ref':<26}{'dur':>7}{'mean':>9}{'max':>9}  note")
    for r in sorted(rows, key=lambda x: -x[4]):
        print(f"{r[0]:<3}{r[1]:<26}{r[3]:>6.2f}s{r[4]:>9.4f}{r[5]:>9.4f}  {r[2]}")

    best = max(rows, key=lambda r: r[4])
    print(f"\nbest: #{best[0]} {best[1]} mean={best[4]:.4f}")
    print(f"artifacts: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
