"""封闭词表 + 穷举式模板 —— 游戏台词的全部来源。

运行期不接 LLM、不接 ASR，所以台词必须**有限可枚举**，全部离线预合成成 wav
（见 game/prototype/README.md 的「与 TTS 的衔接约束」）。这个模块就是那份词表。

词表直接取自 game/prototype/archive.html 的真实名词，不另起一套：
    材质 VARIANTS 12 种 · 形态 FORMS 3 种 · 律动 RHYS 3 种 · 阶段 STAGES 4 种
line_id 用数组下标（mat00..mat11 / form0..form2 / rhy0..rhy2 / stage0..stage3），
和 HTML 里的 MATCOST / CANON 下标一一对应，改名不会破坏对应关系。

**粒度小是刻意的**：12×3×3=108 种组合逐条合成既贵又没意义，
12+3+3 条名词就够——组合交给游戏在运行期把片段拼着播放。
词表改了只需重跑一次合成，禁止在运行期生成文本。
"""

from __future__ import annotations

import csv
from pathlib import Path

from .engine import Line

REPO = Path(__file__).resolve().parents[2]
TTS_DIR = REPO / "tts"
VOICE_DIR = TTS_DIR / "assets" / "voice"
MANIFEST = TTS_DIR / "assets" / "voice_manifest.csv"

# --- 词表：顺序与 archive.html 一致，不要重排 ---------------------------------
MATERIALS = [
    "线描", "蚀刻", "等高", "点彩", "星座", "大理石",
    "草木", "电路", "陶土", "乐谱", "木纹", "数字",
]
FORMS = ["奶娃形", "木形", "云形"]
RHYTHMS = ["静", "息", "游"]
STAGES = ["点", "芽", "幼体", "生灵"]

# --- 名词要说成自然短语，不能裸读 ---------------------------------------------
# 裸读名词（如"线描"）会让 GPT 阶段只吐出 1 个 token（直接 EOS），
# 那个 EOS 语义码 1024 送进 SoVITS 的码本 Gather 会越界报错——不是"质量差"，是直接崩。
# 实测 2 字台词 100% 触发。所以每个名词都套一个固定框架，保证有足够上下文。
FRAMES = {
    "material": "材质，{}。",
    "form": "形态，{}。",
    "rhythm": "律动就是{}。",
    "stage": "已经长到{}了。",
}

# --- 短句：模板固定，槽位只由上面的词表填 -------------------------------------
UI: list[tuple[str, str]] = [
    ("ui.greet", "你好呀，我是奶龙。"),
    ("ui.greet_back", "你又来看我了。"),
    ("ui.pick", "随便挑一样吧。"),
    ("ui.ok", "嗯，这就成了。"),
    ("ui.canon", "这是正典组合。"),
    ("ui.grow", "又长了一点。"),
    ("ui.no_light", "光还不够呢。"),
    ("ui.unlock", "新的材质醒过来了。"),
    ("ui.empty", "这里还空着。"),
    ("ui.full", "全都收齐了。"),
    ("ui.again", "那就再来一次。"),
    ("ui.bye", "下次再来玩。"),
]

# 台词少于这个字数就拒绝合成（见上面 FRAMES 的原因）。预检会先报出来。
MIN_CHARS = 6


def entries() -> list[tuple[str, str, str]]:
    """唯一的词表来源：(line_id, text, group)。"""
    out: list[tuple[str, str, str]] = []
    for group, names, fmt in (
        ("material", MATERIALS, "mat{:02d}"),
        ("form", FORMS, "form{}"),
        ("rhythm", RHYTHMS, "rhy{}"),
        ("stage", STAGES, "stage{}"),
    ):
        for i, name in enumerate(names):
            out.append((fmt.format(i), FRAMES[group].format(name), group))
    for line_id, text in UI:
        out.append((line_id, text, "ui"))
    return out


def too_short() -> list[tuple[str, str]]:
    """返回不满足 MIN_CHARS 的 (line_id, text)，供合成前预检。"""
    return [(i, t) for i, t, _ in entries() if len(t) < MIN_CHARS]


def load() -> list[Line]:
    return [Line(line_id=i, text=t) for i, t, _ in entries()]


def groups() -> dict[str, str]:
    return {i: g for i, _, g in entries()}


def emit(path: Path | None = None) -> Path:
    """把词表落成 csv，供人工审校与版本对比。"""
    path = path or (TTS_DIR / "build" / "voice_lines.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "group", "chars", "text"])
        for line_id, text, group in entries():
            w.writerow([line_id, group, len(text), text])
    return path


def stats() -> str:
    by_group: dict[str, int] = {}
    for _, _, g in entries():
        by_group[g] = by_group.get(g, 0) + 1
    combos = len(MATERIALS) * len(FORMS) * len(RHYTHMS)
    parts = " ".join(f"{k}={v}" for k, v in sorted(by_group.items()))
    return (f"{len(entries())} 条台词（{parts}），"
            f"运行期可拼出 {combos} 种 材质×形态×律动 组合")
