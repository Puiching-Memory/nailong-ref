"""列出两组的完整成员，判断 B 组是真实说话人还是一堆语气词。"""
import sys

from .. import config, manifests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

txt = manifests.texts()
man = manifests.by_idx(config.UTT_MANIFEST)
dur = {i: float(r["dur"]) for i, r in man.items()}
src = {i: r["src"] for i, r in man.items()}
rows = manifests.read(config.TWO_PASS)

for g in ("A", "B"):
    m = [r for r in rows if r["group"] == g]
    tot = sum(dur[int(r["idx"])] for r in m)
    print(f"\n{'=' * 86}")
    print(f"组 {g}   {len(m)} 句   {tot:.1f}s")
    print("=" * 86)
    key = "simA" if g == "A" else "simB"
    for r in sorted(m, key=lambda r: -float(r[key])):
        i = int(r["idx"])
        t = txt[i]
        # 语气词：去标点后汉字数很少
        n = len([c for c in t if "\u4e00" <= c <= "\u9fff"])
        tag = "语气词?" if n <= 2 else ("        " if n >= 5 else "  短语? ")
        print(f"  #{i:>3} {src[i]} {dur[i]:>5.2f}s {float(r[key]):>6.3f} {tag} {t[:34]}")
