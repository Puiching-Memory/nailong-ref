"""把语音库和游戏打包成**单个 HTML**，玩家双击即玩。

为什么非得内联：浏览器的 `file://` 页面对子资源的限制毫无一致性——
Chrome/Edge 放行同盘的图片与媒体，但 fetch/XHR 一律拒绝；VS Code 内置浏览器
连 `<img>` 都读不到。只要 wav 还在 HTML 外面，玩家就得自己架一个静态服务器，
架不起来就是静音。把 wav 以 base64 内联进 HTML，播放路径退化成 `data:` URL，
于是：**不需要服务器、不需要联网、单个文件发出去就能玩**。

代价是体积：base64 把 wav 撑大 1/3，且全部常驻内存。词表只有 34 条、
都是短句，所以总体可控；词表再涨一个数量级就得换方案（见下）。

替代方案与为什么没选：
  · 附带一个静态服务器（`python -m http.server` + 批处理）——玩家看得见黑框、要环境
  · Electron / WebView 壳——为原型引入整个运行时，太重
  · 打包成 zip——解压后还是 `file://`，问题原样存在
  · Service Worker 缓存资源——`file://` 下根本注册不了

用法：
    nailong-tts bundle [--out 路径]
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from . import corpus

REPO = Path(__file__).resolve().parents[2]
SRC_HTML = REPO / "game" / "prototype" / "archive.html"
OUT_HTML = REPO / "game" / "dist" / "nailong.html"
MARKER = "<!--VOICE_BANK-->"


def voice_bank(voice_dir: Path | None = None) -> dict[str, str]:
    """{line_id: base64(wav)}，按 id 排序保证产物可复现。"""
    root = Path(voice_dir) if voice_dir else corpus.VOICE_DIR
    return {
        p.stem: base64.b64encode(p.read_bytes()).decode("ascii")
        for p in sorted(root.glob("*.wav"))
    }


def build(
    src: str | Path | None = None,
    out: str | Path | None = None,
    voice_dir: str | Path | None = None,
) -> dict:
    """内联语音库并落盘，返回一份可打印的构建报告。"""
    src = Path(src) if src else SRC_HTML
    out = Path(out) if out else OUT_HTML
    html = src.read_text(encoding="utf-8")
    if MARKER not in html:
        raise ValueError(f"{src} 里找不到注入点 {MARKER}，先确认它没被改掉")

    bank = voice_bank(voice_dir)
    if not bank:
        raise FileNotFoundError(f"语音库是空的：{voice_dir or corpus.VOICE_DIR}")

    payload = json.dumps(bank, ensure_ascii=True, separators=(",", ":"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        html.replace(MARKER, f"<script>window.VOICE_BANK={payload};</script>"),
        encoding="utf-8",
    )
    # 词表里有、语音库里没有的 → 运行时静音。这是构建期该知道的事。
    missing = [i for i, _, _ in corpus.entries() if i not in bank]
    return {
        "out": out,
        "voiced": len(bank),
        "audio_bytes": sum(len(base64.b64decode(v)) for v in bank.values()),
        "missing": missing,
    }
