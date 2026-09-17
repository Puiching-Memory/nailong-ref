"""TTS 引擎接口 —— 预留层。

引擎还没选型，但下游小游戏已经需要一个稳定的调用面，所以先把接口钉死，
用一个能跑的 `NullEngine` 占位；换引擎时只替换注册表里的实现。

硬约束（项目设定）：**不接 LLM、不接 ASR**，TTS 只做单向输出。
接口因此不留任何「运行时生成文本」的余地——输入只能是封闭词表里的一条台词，
这样才可能把所有台词离线预合成、打包进游戏，运行期零模型加载。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Line:
    """一条待合成台词。`line_id` 是封闭词表里的键，不是任意文本。"""

    line_id: str
    text: str
    emotion: str = "NEUTRAL"
    speed: float = 1.0


@runtime_checkable
class TtsEngine(Protocol):
    """所有引擎实现的契约。"""

    name: str
    sample_rate: int

    def prepare(self) -> None:
        """载入权重 / 预热。允许耗时，调用方可缓存。"""

    def say(self, line: Line) -> np.ndarray:
        """返回单声道 float32 [-1,1] 波形。"""


class NullEngine:
    """占位引擎：按字数估时长生成静音。

    存在的意义是让游戏侧的播放与字幕时序能先跑通——接口、时长估算、
    资源落盘格式全都定型后，再替换成真引擎。
    """

    name = "null"
    sample_rate = 24_000
    sec_per_char = 0.22
    min_sec = 0.6

    def __init__(self, sample_rate: int | None = None):
        if sample_rate:
            self.sample_rate = sample_rate

    def prepare(self) -> None:  # 无权重可载
        return

    def duration(self, line: Line) -> float:
        return max(self.min_sec, len(line.text) * self.sec_per_char) / line.speed

    def say(self, line: Line) -> np.ndarray:
        n = int(self.duration(line) * self.sample_rate)
        return np.zeros(n, dtype=np.float32)


_REGISTRY: dict[str, type] = {NullEngine.name: NullEngine}


def register(engine_cls: type) -> type:
    """注册一个引擎实现。用类装饰器形式调用。"""
    _REGISTRY[engine_cls.name] = engine_cls
    return engine_cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str = NullEngine.name, **kw) -> TtsEngine:
    if name not in _REGISTRY:
        raise KeyError(f"未知引擎 '{name}'；已注册 {available()}")
    return _REGISTRY[name](**kw)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """落盘 16bit wav。游戏直接引用这个路径，运行期不再调模型。"""
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
