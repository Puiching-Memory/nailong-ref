"""nailong-tts 命令行。

    nailong-tts pack            打包训练数据 -> tts/build/
    nailong-tts review          刷新待复核清单，保留哈希未变的人工修订
    nailong-tts audit           正式训练门禁；失败返回非零
    nailong-tts pack --experimental 仅豁免正式语料量门槛
    nailong-tts stats           候选统计，不代表训练就绪
    nailong-tts engines         列出已注册引擎
    nailong-tts say "台词"      用占位引擎合成一段 wav，验证播放链路
    nailong-tts corpus          打印封闭词表统计并落成 csv
    nailong-tts synth           离线批量合成词表 -> tts/assets/voice/
                                --force 全量重渲染（默认只补缺的）
                                --only mat00,rhy1 只渲染指定几条
    nailong-tts bank            检查预合成资源是否齐全
    nailong-tts bundle          把语音库与游戏合成单文件 -> game/dist/nailong.html
                                玩家双击即玩，不需要服务器

`synth` 是**构建期**命令，需要外部 GPT-SoVITS 环境（见 synth.py 模块头）；
其余命令只用本包依赖，运行期也只用 `bank` 引擎。
"""

from __future__ import annotations

import sys

from . import dataset, engine

SAY_DIR = dataset.BUILD / "say"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"

    if cmd == "pack":
        dataset.pack(experimental="--experimental" in argv)
        return 0

    if cmd == "review":
        from . import quality

        quality.make_review()
        return 0

    if cmd == "audit":
        from . import quality

        result = quality.audit()
        quality.print_audit(result)
        return 0 if not result.issues else 1

    if cmd == "stats":
        dataset.report(dataset.load())
        return 0

    if cmd == "engines":
        print("已注册引擎: " + ", ".join(engine.available()))
        print("默认占位引擎不发音，只按字数估时长——用于先跑通游戏侧的播放与字幕时序。")
        print("bank 是运行期引擎：按 line_id 查离线预合成的 wav，不加载任何模型。")
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

    if cmd == "corpus":
        from . import corpus

        path = corpus.emit()
        print(corpus.stats())
        print(f"-> {path}")
        return 0

    if cmd == "synth":
        from . import corpus, synth

        force = "--force" in argv
        only: set[str] | None = None
        if "--only" in argv:
            only = {s.strip() for s in argv[argv.index("--only") + 1].split(",") if s.strip()}
        lines = [line for line in corpus.load() if only is None or line.line_id in only]
        if only is not None and len(lines) != len(only):
            known = {line.line_id for line in corpus.load()}
            print(f"未知 line_id: {sorted(only - known)}", file=sys.stderr)
            return 2
        synth.synthesize(lines, corpus.VOICE_DIR, force=force)
        return 0

    if cmd == "bank":
        from . import bank

        b = bank.BankEngine()
        try:
            b.prepare()
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"bank: {b.loaded} 条就绪 @ {b.sample_rate} Hz")
        if b.missing:
            head = ", ".join(b.missing[:12])
            more = " ..." if len(b.missing) > 12 else ""
            print(f"缺 {len(b.missing)} 条: {head}{more}")
            return 1
        return 0

    if cmd == "bundle":
        from . import bundle

        out = argv[argv.index("--out") + 1] if "--out" in argv else None
        try:
            info = bundle.build(out=out)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        path = info["out"]
        print(f"bundle: {path}  {path.stat().st_size / 1e6:.1f} MB"
              f"（内联语音 {info['voiced']} 条 / {info['audio_bytes'] / 1e6:.1f} MB 原始音频）")
        if info["missing"]:
            print(f"静音 {len(info['missing'])} 条（尚未合成）: {', '.join(info['missing'])}")
        print("双击即可游玩：无服务器、无联网、无额外文件。")
        return 0

    print(__doc__.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
