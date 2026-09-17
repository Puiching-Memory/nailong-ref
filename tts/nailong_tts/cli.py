"""nailong-tts 命令行。

    nailong-tts pack            打包训练数据 -> tts/build/
    nailong-tts stats           只跑质检，不写文件
    nailong-tts engines         列出已注册引擎
    nailong-tts say "台词"      用占位引擎合成一段 wav，验证播放链路
"""

from __future__ import annotations

import sys

from . import dataset, engine

SAY_DIR = dataset.BUILD / "say"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"

    if cmd == "pack":
        dataset.pack()
        return 0

    if cmd == "stats":
        dataset.report(dataset.load())
        return 0

    if cmd == "engines":
        print("已注册引擎: " + ", ".join(engine.available()))
        print("默认占位引擎不发音，只按字数估时长——用于先跑通游戏侧的播放与字幕时序。")
        return 0

    if cmd == "say":
        if len(argv) < 2:
            print("用法: nailong-tts say \"台词\" [引擎名]", file=sys.stderr)
            return 2
        name = argv[2] if len(argv) > 2 else engine.NullEngine.name
        tts = engine.get(name)
        tts.prepare()
        line = engine.Line(line_id="adhoc", text=argv[1])
        audio = tts.say(line)
        out = SAY_DIR / f"adhoc_{name}.wav"
        engine.write_wav(out, audio, tts.sample_rate)
        print(f"{tts.name}: {len(audio) / tts.sample_rate:.2f}s -> {out}")
        return 0

    print(__doc__.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
