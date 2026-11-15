"""看 S1 留一法最差的那几张到底长什么样、错在哪。

预测点 = GT + (dX, dY)（Kaggle 日志给的最差 8 张逐图偏移），
原分辨率 700x700 裁块，绿=GT，黄=预测，青圈=检出的视盘。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from PIL import Image, ImageDraw

import t3_kaggle as K

ROOT = HERE / "task3" / "detection"
OUT = HERE / "t3_look"
OUT.mkdir(exist_ok=True)

gt = {}
with open(ROOT / "fovea_localization_train_GT.csv", "rb") as f:
    f.readline()
    for ln in f:
        p = ln.decode("utf-8").strip().split(",")
        if len(p) >= 3:
            gt[int(p[0])] = (float(p[1]), float(p[2]))

WORST8 = {72: (353, +234, -264), 49: (147, +143, -35), 62: (101, +73, +70),
          24: (98, +96, -19), 57: (69, +61, +33), 44: (65, -65, +3),
          16: (49, -47, -11), 34: (41, -1, -41)}

# 全训练集的亮度统计（缩略图上算），用来对比这 8 张是不是异常暗/异常亮
lums = {}
for sid in sorted(gt):
    p = ROOT / "train" / f"{sid:04d}.jpg"
    with Image.open(p) as im:
        w, h = im.size
        sm = im.convert("RGB").resize((256, 256), Image.BILINEAR)
    a = np.asarray(sm).astype(np.float32) / 255.0
    m = a.max(2) > 0.12
    L = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lums[sid] = (float(L[m].mean()) if m.any() else 0.0,
                 float(L[m].std()) if m.any() else 0.0, (w, h))
allv = np.array([lums[s][0] for s in sorted(gt)])
print(f"全 80 张视野内亮度: mean={allv.mean():.3f} std={allv.std():.3f} "
      f"min={allv.min():.3f} max={allv.max():.3f}")

print("\n最差 8 张的情况（亮度 / 视野对比度 / 尺寸 / 距视盘）:")
for sid, (err, dx, dy) in WORST8.items():
    p = ROOT / "train" / f"{sid:04d}.jpg"
    gx, gy = gt[sid]
    px, py = gx + dx, gy + dy
    d = K.disc_orig(p)
    dist = float(np.hypot(px - d[0], py - d[1])) if d else float("nan")
    lum, sd, wh = lums[sid]
    tag = "S" if max(wh) < 2500 else "L"
    print(f"  {sid:04d} err={err:3d}px  size={wh[0]}x{wh[1]}({tag})  "
          f"lum={lum:.3f} sd={sd:.3f}  |pred-disc|={dist:6.1f}px")

# 画 8 张近景
CELL, SCALE = 700, 1.0
rows, cols = 2, 4
sheet = Image.new("RGB", (cols * CELL, rows * CELL), (18, 18, 18))
dr = ImageDraw.Draw(sheet)
for n, (sid, (err, dx, dy)) in enumerate(WORST8.items()):
    p = ROOT / "train" / f"{sid:04d}.jpg"
    gx, gy = gt[sid]
    px, py = gx + dx, gy + dy
    with Image.open(p) as im:
        W, H = im.size
        L = int(CELL / SCALE)
        x0 = int(max(0, min(W - L, gx - L // 2)))
        y0 = int(max(0, min(H - L, gy - L // 2)))
        box = (x0, y0, x0 + L, y0 + L)
        crop = im.convert("RGB").crop(box).resize((CELL, CELL), Image.LANCZOS)
    ox, oy = (n % cols) * CELL, (n // cols) * CELL
    sheet.paste(crop, (ox, oy))

    def cp(pt, color, r=26):
        x = ox + (pt[0] - x0) * SCALE
        y = oy + (pt[1] - y0) * SCALE
        dr.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        dr.line([x - r * 1.6, y, x + r * 1.6, y], fill=color, width=3)
        dr.line([x, y - r * 1.6, x, y + r * 1.6], fill=color, width=3)

    cp((px, py), (255, 210, 0))
    cp((gx, gy), (0, 255, 90))
    d = K.disc_orig(p)
    if d:
        x = ox + (d[0] - x0) * SCALE
        y = oy + (d[1] - y0) * SCALE
        r = d[2] * SCALE
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(0, 220, 255), width=3)
    dr.text((ox + 8, oy + 8), f"#{sid:04d} S1 err={err}px  (yellow=pred, green=GT)",
            fill=(120, 255, 120))
    dr.text((ox + 8, oy + 26), f"dx={dx:+d} dy={dy:+d}  cyan=optic disc",
            fill=(120, 255, 120))

sheet.save(OUT / "worst8_s1.png")
print(f"\n已写 {OUT / 'worst8_s1.png'}")
