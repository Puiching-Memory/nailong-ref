"""清单 I/O：跨阶段唯一的数据契约。

所有清单以 `idx` 为主键。原先各脚本用 `csv.DictReader(open("xxx.csv"))`
各写各的，行序与浮点格式不一致，导致重跑后 CSV 之间对不上号。
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import config

__all__ = ["read", "by_idx", "write", "f2", "f4"]


def read(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def by_idx(path) -> dict[int, dict]:
    """按 idx 索引，供随机访问。"""
    return {int(r["idx"]): r for r in read(path)}


def write(path, fields: list[str], rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        w.writerows(rows)


def f2(v) -> str:
    return f"{float(v):.2f}"


def f4(v) -> str:
    return f"{float(v):.4f}"


def texts() -> dict[int, str]:
    """idx -> 识别文本（来自 sv_all.csv）。"""
    return {i: r["text"] for i, r in by_idx(config.SV_ALL).items()}


def durations() -> dict[int, float]:
    return {i: float(r["dur"]) for i, r in by_idx(config.UTT_MANIFEST).items()}
