"""预录制语音库 —— 运行期唯一的 TTS 实现。

游戏不接 LLM、不接 ASR，也不需要任何模型：台词在构建期就合成好了（见 synth.py），
这里只按 `line_id` 查 wav 再解码。解码结果缓存在内存里，因为台词是复用的、
总量也小（封闭词表）。

找不到资源时**不抛异常**，退化成静音但保留估时长——资源没打全会没声音，
但不该让游戏直接崩掉。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import engine
from .engine import Line


@engine.register
class BankEngine:
    """按 line_id 查预合成 wav 的引擎。"""

    name = "bank"
    sample_rate = 32_000  # 上游 SoVITS v2ProPlus 的 sampling_rate，实际以 manifest 为准

    def __init__(self, root: str | Path | None = None, manifest: str | Path | None = None):
        from . import corpus

        self.root = Path(root) if root else corpus.VOICE_DIR
        self.manifest_path = Path(manifest) if manifest else corpus.MANIFEST
        self._index: dict[str, Path] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._missing: list[str] = []

    # --- 契约 ---------------------------------------------------------------
    def prepare(self) -> None:
        """读 manifest 建索引。缺少的文件记下来，不中断。"""
        self._index.clear()
        self._cache.clear()
        self._missing.clear()
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"语音清单不存在：{self.manifest_path}；先跑 `nailong-tts synth` 生成"
            )
        with self.manifest_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                path = self.root / row["file"]
                if path.exists():
                    self._index[row["line_id"]] = path
                    if "sample_rate" in row and row["sample_rate"]:
                        self.sample_rate = int(row["sample_rate"])
                else:
                    self._missing.append(row["line_id"])

    def say(self, line: Line) -> np.ndarray:
        if not self._index:
            self.prepare()
        cached = self._cache.get(line.line_id)
        if cached is not None:
            return cached
        path = self._index.get(line.line_id)
        if path is None:
            return self._silence(line)
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:  # 多声道压成单声道
            audio = audio.mean(axis=1)
        self.sample_rate = sr
        self._cache[line.line_id] = audio
        return audio

    # --- 自省 ---------------------------------------------------------------
    @property
    def missing(self) -> list[str]:
        return list(self._missing)

    @property
    def loaded(self) -> int:
        return len(self._index)

    def _silence(self, line: Line) -> np.ndarray:
        n = int(max(engine.NullEngine.min_sec,
                    len(line.text) * engine.NullEngine.sec_per_char) * self.sample_rate)
        return np.zeros(n, dtype=np.float32)
