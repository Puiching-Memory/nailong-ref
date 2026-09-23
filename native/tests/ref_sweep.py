"""挑参考音频：同一句台词换不同参考合成，用 sv_embedding 排序。

零样本克隆里参考音频决定音色与韵律。这里没有播放设备，听不出差异，
所以用声纹相似度做**可复现**的排序：固定 probe 文本，换参考合成，
再算 cos(合成嵌入, 已听音批准参考嵌入) 的均值。

跑法（用参考仓库的 venv，它才有 onnxruntime / torch）：
    python native/tests/ref_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tts"))
sys.path.insert(0, str(REPO / "src"))

from nailong_tts import quality, synth  # noqa: E402

PROBE = "我是奶龙，你好呀，今天也要开开心心的。"
OUT_DIR = REPO / "native" / "gsv" / "_sweep"

PRODUCTION_DIR = REPO / "dataset" / "production"


def accepted_rows() -> list[dict[str, str]]:
    result = quality.audit(min_seconds=0, min_clips=0)
    return [{**row, "text": text} for row, text in result.usable]


def main() -> int:
    candidates = accepted_rows()
    if len(candidates) < 2:
        raise SystemExit("至少需要 2 段已听音批准的参考音频；先完成 training_review.csv")
    import onnxruntime
    names = [row["file"] for row in candidates]
    ref_paths = [PRODUCTION_DIR / name for name in names]

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
    backend = synth.Backend(ref_wav=str(PRODUCTION_DIR / candidates[0]["file"]),
                            ref_text=candidates[0]["text"])

    rows = []
    for i, row in enumerate(candidates, 1):
        wav_name, ref_text = row["file"], row["text"]
        note = f"{row['dur']}s {row['speaker_basis']} stem={row['stem_margin_db']}dB"
        out = OUT_DIR / f"{i:02d}_{Path(wav_name).name}"
        backend.ref_wav = PRODUCTION_DIR / wav_name
        backend.ref_text = ref_text
        print(f"[{i}/{len(candidates)}] ref={wav_name} ({note})")
        audio = backend.say(PROBE, out)
        dur = len(audio) / backend.sample_rate

        # 必须走 16k：sv_embedding 是 16k 模型，直接喂合成的 32k 音频会让嵌入整体偏低
        # （实测把 0.67 压到 0.53），排序就不可信了。embed() 内部会重采样。
        v = embed(sv, out)
        sims = ref_mat @ v
        rows.append((i, wav_name, note, dur, float(sims.mean()), float(sims.max())))

    print("\n--- 排序（按与已批准参考的平均余弦）---")
    print(f"{'#':<3}{'ref':<26}{'dur':>7}{'mean':>9}{'max':>9}  note")
    for r in sorted(rows, key=lambda x: -x[4]):
        print(f"{r[0]:<3}{r[1]:<26}{r[3]:>6.2f}s{r[4]:>9.4f}{r[5]:>9.4f}  {r[2]}")

    best = max(rows, key=lambda r: r[4])
    print(f"\nbest: #{best[0]} {best[1]} mean={best[4]:.4f}")
    print(f"artifacts: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
