"""把转场帧条拼成 montage: 行=切点, 列=时间(左→右跨越切点)"""
import os, glob
from PIL import Image, ImageDraw

OUT = r"C:\workspace\nailong-ref\vid-analysis\strips"
for key in ["A", "B", "C"]:
    cuts = sorted(set(int(os.path.basename(p).split("_")[1]) for p in glob.glob(os.path.join(OUT, f"{key}_*_*.png"))))
    rows = []
    for ci in cuts:
        frames = sorted(glob.glob(os.path.join(OUT, f"{key}_{ci}_*.png")))
        ims = [Image.open(f) for f in frames]
        w, h = ims[0].size
        row = Image.new("RGB", (w*len(ims), h), (20, 20, 20))
        for i, im in enumerate(ims):
            row.paste(im, (i*w, 0))
        d = ImageDraw.Draw(row)
        d.line([(4*w, 0), (4*w, h)], fill=(255, 60, 60), width=2)  # 切点位置
        rows.append(row)
    W = max(r.width for r in rows); H = sum(r.height for r in rows) + 4*len(rows)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height + 4
    canvas.save(os.path.join(OUT, f"montage_{key}.png"))
    print(key, canvas.size)
