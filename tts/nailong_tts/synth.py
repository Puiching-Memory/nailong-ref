"""离线批量预合成 —— 把封闭词表的台词渲染成 wav 资源。

**只在打包资源时跑一次**；运行期游戏只读 wav，不加载任何模型。
离线对速度不敏感，所以走 CPU 的 ONNX Runtime 即可，不需要 GPU / TensorRT。

引擎复用外部仓库 `GPT-SoVITS_minimal_inference` 的 ONNX 推理实现，
模型是我们自己用 `export_onnx.py` 导出的（见 native/gsv/onnx_out/）。

资产都不入库，用环境变量覆盖路径：

    NAILONG_GSV_REPO      GPT-SoVITS_minimal_inference 仓库根
    NAILONG_GSV_ONNX      导出的 onnx 目录（含 config.json 与 8 个 onnx）
    NAILONG_GSV_BERT      chinese-roberta-wwm-ext-large 目录
    NAILONG_GSV_REF       参考音频（3~10s 干净奶龙人声）
    NAILONG_GSV_REF_TEXT  参考音频的文本（零样本克隆要它做韵律对齐）

跑法（cwd 必须是 tts/，这样 `nailong_tts` 才在 import 路径上）：

    python -m nailong_tts.synth --say "你好呀，我是奶龙。" --out smoke.wav
    python -m nailong_tts.synth --batch
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import Line, write_wav

REPO = Path(__file__).resolve().parents[2]

GSV_REPO = Path(os.environ.get("NAILONG_GSV_REPO", r"C:\workspace\github\GPT-SoVITS_minimal_inference"))
ONNX_DIR = Path(os.environ.get("NAILONG_GSV_ONNX", REPO / "native" / "gsv" / "onnx_out"))
BERT_DIR = Path(os.environ.get("NAILONG_GSV_BERT", REPO / "native" / "gsv" / "chinese-roberta-wwm-ext-large"))

# 参考片段取自 dataset/final/。长度是关键：声纹一致性与参考时长正相关，
# 实测同一句 probe 换 5 条参考，8.00s 的 src_06 得分 0.7562（高于同长度对照中位 0.7396），
# 4.28s 的 src_05 只有 0.6696（低于对照 p5 0.7129）。见 native/tests/ref_sweep.py。
# src_06 还是唯一一条"到 11 段参考的最小余弦 0.6908"都高于参考互相似下沿 0.6843 的，
# 即所有参考都认可它像奶龙本身。
REF_WAV = Path(os.environ.get("NAILONG_GSV_REF", REPO / "dataset" / "final" / "src_06_14.22-22.22.wav"))
REF_TEXT = os.environ.get(
    "NAILONG_GSV_REF_TEXT",
    "还好没超过3秒呃，我刚刚出的三水一听就被你吃了，不吃不就浪费了呢。",
)

# 上游 sample_topk 没有固定种子，同一句每次韵律都不同。设了这个环境变量就能复现。
SEED_ENV = "NAILONG_GSV_SEED"


WORD = "zh"
PAUSE = 0.2

_ASCII = re.compile(r"[A-Za-z]+")


def _simple_segments(text: str, lang: str = "zh") -> list[dict]:
    """确定性分段：连续 ASCII 字母算 en，其余算 lang。

    项目台词来自**封闭词表**，语种在设计期就已知，不需要统计式语言检测。
    参考仓库的 `LangSegmenter` 走 fast_langdetect，要额外下 126MB 的 lid.176.bin
    并依赖外网；对构建可复现性是纯负担，所以替换掉。
    """
    out: list[dict] = []
    pos = 0
    for m in _ASCII.finditer(text):
        if m.start() > pos:
            out.append({"lang": lang, "text": text[pos:m.start()]})
        out.append({"lang": "en", "text": m.group(0)})
        pos = m.end()
    if pos < len(text):
        out.append({"lang": lang, "text": text[pos:]})
    return [s for s in out if s["text"]]


def _install_langseg_shim(R) -> None:
    """把 LangSegmenter.getTexts 换成上面的确定性分段。"""

    def getTexts(text, lang=None, default_lang=None, **_kw):
        base = default_lang or lang or "zh"
        if base not in ("zh", "ja", "ko", "yue", "en"):
            base = "zh"
        if base == "en":
            return [{"lang": "en", "text": text}]
        return _simple_segments(text, base)

    R.LangSegmenter.getTexts = getTexts


@dataclass
class Clip:
    """一条已渲染的语音资源。"""

    line_id: str
    text: str
    path: Path
    seconds: float


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def trim_silence(audio: np.ndarray, sample_rate: int, thresh_db: float = -45.0,
                 pad_sec: float = 0.05) -> np.ndarray:
    """裁掉首尾静音，各留 `pad_sec` 尾巴。

    游戏资源里带空白没有意义，而且空白会稀释声纹嵌入、影响对时长的手感。
    用 20ms 帧 RMS 包络找有声音帧，阈值相对峰值取，避免依赖绝对电平。
    """
    hop = max(1, int(0.02 * sample_rate))
    usable = (audio.size // hop) * hop
    if usable == 0:
        return audio
    rms = np.sqrt((audio[:usable].reshape(-1, hop) ** 2).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return audio
    loud = np.nonzero(rms >= peak * (10.0 ** (thresh_db / 20.0)))[0]
    if loud.size == 0:
        return audio
    pad = int(pad_sec * sample_rate)
    start = max(0, int(loud[0]) * hop - pad)
    end = min(audio.size, (int(loud[-1]) + 1) * hop + pad)
    return audio[start:end]


class Backend:
    """懒加载的 GPT-SoVITS ONNX 引擎。载权重比合成本身慢得多，所以只建一次。"""

    def __init__(self, device: str = "cpu", ref_wav: str | Path | None = None,
                 ref_text: str | None = None):
        self.ref_wav = Path(ref_wav) if ref_wav else REF_WAV
        self.ref_text = ref_text if ref_text is not None else REF_TEXT
        _require(ONNX_DIR / "config.json", "exported ONNX dir")
        _require(BERT_DIR, "roberta bert dir")
        _require(self.ref_wav, "reference wav")

        # 参考仓库的 run_onnx_inference 在模块顶层就把 GPT_SoVITS/ 也加进了 sys.path。
        # 另外它的中文前端把 G2PWModel 写成了相对路径 "GPT_SoVITS/text/G2PWModel"，
        # 所以必须把 cwd 切到参考仓库根——否则它连下载 zip 都打不开文件。
        # 这里用到的 onnx / bert / 参考音频路径都是绝对的，不受 chdir 影响。
        os.environ["bert_path"] = str(BERT_DIR)
        os.chdir(GSV_REPO)
        sys.path.insert(0, str(GSV_REPO))
        import run_onnx_inference as R  # type: ignore[import-not-found]

        self._R = R
        _install_langseg_shim(R)
        seed = os.environ.get(SEED_ENV)
        if seed:
            np.random.seed(int(seed))
        self.engine = R.GPTSoVITS_ONNX_Inference(str(ONNX_DIR), str(BERT_DIR), device=device)
        self.sample_rate = int(self.engine.hps["data"]["sampling_rate"])
        self.version = self.engine.version

    def say(self, text: str, out_path: Path, pause_length: float = PAUSE,
            trim: bool = True) -> np.ndarray:
        """合成一段文本并落盘，返回波形（默认已裁掉首尾静音）。"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine.infer(
            str(self.ref_wav), self.ref_text, WORD,
            text, WORD,
            output_path=str(out_path), pause_length=pause_length,
        )
        import soundfile as sf
        audio, _ = sf.read(str(out_path), dtype="float32")
        if trim:
            audio = trim_silence(audio, self.sample_rate)
            write_wav(out_path, audio, self.sample_rate)
        return audio


def synthesize(lines: list[Line], out_dir: Path, backend: Backend | None = None,
               force: bool = False, manifest: str | Path | None = None) -> list[Clip]:
    """把一批台词渲染成 wav，并写一份 line_id -> 文件的清单。

    已有 wav 默认跳过，所以词表改几条就只补合成那几条；`force=True` 全量重渲染。
    """
    import csv

    from . import corpus

    # 预检：短于 MIN_CHARS 的台词一定会让 GPT 阶段直接吐 EOS（只出 1 个 token），
    # 那个 EOS 语义码送进 SoVITS 的码本 Gather 会越界。先说清楚，
    # 别等 40s 载完模型才炸。
    bad = [(l.line_id, l.text) for l in lines if len(l.text) < corpus.MIN_CHARS]
    if bad:
        for line_id, text in bad:
            print(f"too short ({len(text)} < {corpus.MIN_CHARS}): {line_id} {text}",
                  file=sys.stderr)
        raise ValueError(f"{len(bad)} 条台词短于 {corpus.MIN_CHARS} 字，先改 corpus.py")

    backend = backend or Backend()
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Clip] = []
    failures: list[tuple[str, str]] = []
    skipped = 0
    t0 = time.perf_counter()
    for i, line in enumerate(lines, 1):
        out_path = out_dir / f"{line.line_id}.wav"
        if out_path.exists() and not force:
            import soundfile as sf

            clips.append(Clip(line.line_id, line.text, out_path, sf.info(str(out_path)).duration))
            skipped += 1
            print(f"[{i}/{len(lines)}] {line.line_id} skip (exists)", flush=True)
            continue
        try:
            audio = backend.say(line.text, out_path)
        except Exception as exc:  # 单条失败不该让整批白跑
            failures.append((line.line_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(lines)}] {line.line_id} FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        clip = Clip(line.line_id, line.text, out_path, len(audio) / backend.sample_rate)
        clips.append(clip)
        print(f"[{i}/{len(lines)}] {line.line_id} {clip.seconds:.2f}s", flush=True)

    mpath = Path(manifest) if manifest else corpus.MANIFEST
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with mpath.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "file", "dur", "sample_rate", "text"])
        for c in clips:
            w.writerow([c.line_id, c.path.name, f"{c.seconds:.3f}",
                        backend.sample_rate, c.text])

    dt = time.perf_counter() - t0
    total = sum(c.seconds for c in clips)
    print(f"\ndone: {len(clips)} clips ({skipped} skipped, {len(failures)} failed), "
          f"{total:.1f}s audio, {dt:.1f}s wall "
          f"(sample_rate={backend.sample_rate}, version={backend.version})")
    for line_id, err in failures:
        print(f"  FAILED {line_id}: {err}", file=sys.stderr)
    print(f"manifest: {mpath}")
    return clips


def check(audio: np.ndarray) -> str:
    """没有播放设备，只能靠数值判据说明「确实出声了」。"""
    if audio.size == 0:
        return "EMPTY"
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    # 数字静音或近似静音都视为失败
    if peak < 1e-3:
        return f"SILENT peak={peak:.2e}"
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    return f"peak={peak:.4f} rms={rms:.4f} clipped={clipped:.3%}"


def _smoke(text: str, out: Path) -> int:
    # 先把收到的文本回显出来：中文经命令行参数传递有可能被 shell 的 GBK 编码损坏，
    # 这里点名长度与码位，出错时能立刻看出来。
    print(f"text={text!r} chars={len(text)} cps={[hex(ord(c)) for c in text]}")
    backend = Backend()
    audio = backend.say(text, out)
    print(f"wrote {out}")
    print(f"  {len(audio) / backend.sample_rate:.2f}s @ {backend.sample_rate} Hz")
    print(f"  {check(audio)}")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="offline GPT-SoVITS batch synthesis")
    p.add_argument("--say", help="synthesize one ad-hoc line")
    p.add_argument("--out", default=str(REPO / "native" / "gsv" / "_smoke.wav"))
    p.add_argument("--batch", action="store_true", help="render the whole corpus")
    a = p.parse_args()

    if a.say:
        raise SystemExit(_smoke(a.say, Path(a.out)))

    if a.batch:
        from . import corpus
        lines = corpus.load()
        clips = synthesize(lines, corpus.VOICE_DIR)
        print(f"{len(clips)} clips -> {corpus.VOICE_DIR}")
        raise SystemExit(0)

    p.print_help()
