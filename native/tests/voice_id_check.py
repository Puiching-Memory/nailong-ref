"""声纹一致性校验：合成出来的到底是不是奶龙的声音？

没有播放设备、人耳不可用，所以用一个可复现的数字判据代替听感：
拿与训练同源的 `sv_embedding.onnx` 给每段音频取 192 维嵌入，L2 归一化后算余弦。

判据是**相对**的，不是绝对阈值：
  - 先把 11 段参考音频互相之间的余弦分布测出来（这是同一个说话人的"自相似区间"）
  - 再看合成片段到这些参考的余弦落在什么位置
  - 落在区间内 => 音色没跑偏；明显低于区间下沿 => 音色不对

跑法：
    python native/tests/voice_id_check.py [要校验的 wav ...]
不给参数则校验 native/gsv/_smoke.wav。
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ONNX_DIR = Path(os.environ.get("NAILONG_GSV_ONNX", REPO / "native" / "gsv" / "onnx_out"))
FINAL_DIR = REPO / "dataset" / "final"
MANIFEST = REPO / "dataset" / "manifests" / "final_manifest.csv"
SR = 16_000


def load16k(path: Path) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=SR, mono=True)
    return wav.astype(np.float32)


def embed_wav(sess, wav: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: wav.astype(np.float32)[None, :]})[0]
    v = np.asarray(out).reshape(-1).astype(np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero embedding")
    return v / n


def embed(sess, path: Path) -> np.ndarray:
    return embed_wav(sess, load16k(path))


def cosine_matrix(vecs: list[np.ndarray]) -> np.ndarray:
    m = np.stack(vecs)
    return m @ m.T


def upper_tri(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(m), k=1)
    return m[iu]


def main(argv: list[str]) -> int:
    import onnxruntime

    targets = [Path(p) for p in argv] or [REPO / "native" / "gsv" / "_smoke.wav"]
    for t in targets:
        if not t.exists():
            print(f"missing: {t}", file=sys.stderr)
            return 2

    names = [r["file"] for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))]
    refs = [FINAL_DIR / n for n in names]
    missing = [r for r in refs if not r.exists()]
    if missing:
        print(f"missing {len(missing)} reference clip(s), e.g. {missing[0]}", file=sys.stderr)
        return 2

    so = onnxruntime.SessionOptions()
    so.log_severity_level = 3
    sess = onnxruntime.InferenceSession(
        str(ONNX_DIR / "sv_embedding.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )
    print(f"sv_embedding inputs : {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"sv_embedding outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    print(f"reference clips     : {len(refs)}\n")

    ref_vecs = [embed(sess, r) for r in refs]
    ref_mat = np.stack(ref_vecs)
    ref_sim = upper_tri(cosine_matrix(ref_vecs))
    lo, mid, hi = np.percentile(ref_sim, [5, 50, 95])
    print("--- 参考段互相似（同一说话人的自相似区间）---")
    print(f"  n={ref_sim.size}  min={ref_sim.min():.4f}  p5={lo:.4f}  median={mid:.4f}  "
          f"p95={hi:.4f}  max={ref_sim.max():.4f}\n")

    rc = 0
    for t in targets:
        wav = load16k(t)
        v = embed_wav(sess, wav)
        sims = ref_mat @ v
        print(f"--- {t.name} ---")
        print(f"  时长: {len(wav) / SR:.2f}s")
        print(f"  vs 参考段: mean={sims.mean():.4f}  min={sims.min():.4f}  max={sims.max():.4f}  "
              f"argmax={names[int(sims.argmax())]}")

        # 同长度对照：把每段参考截成与该目标等长的窗口，再与**其余**参考比。
        # 声纹嵌入的稳定性随时长上升，短片段天然得分低——不排除这个因素就没法
        # 判断低分到底是"音色不对"还是"片段太短"。
        win = len(wav)
        ctrl_means = []
        for i, r in enumerate(refs):
            rw = load16k(r)
            if len(rw) < win:
                continue
            rv = embed_wav(sess, rw[:win])
            others = np.delete(ref_mat, i, axis=0)
            ctrl_means.append(float((others @ rv).mean()))
        ctrl = np.array(ctrl_means)
        clo, cmid, chi = np.percentile(ctrl, [5, 50, 95])
        print(f"  同长度对照({win / SR:.2f}s 窗口, n={ctrl.size}): p5={clo:.4f}  "
              f"median={cmid:.4f}  p95={chi:.4f}")

        verdict = "OK" if sims.mean() >= clo else "SUSPECT"
        print(f"  判定: {verdict}（对照下沿 p5={clo:.4f}）")
        if verdict != "OK":
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
