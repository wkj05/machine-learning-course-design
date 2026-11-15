# -*- coding: utf-8 -*-
"""任务三：把 GT 黄斑周围 320x320 的原始分辨率小块裁出来拼图，判断"黄斑的可见特征"到底有多清楚。"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
OUT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\t3_look')
OUT.mkdir(parents=True, exist_ok=True)

gt = {}
with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        gt[int(r['data'])] = (float(r['Fovea_X']), float(r['Fovea_Y']))

picks = [1, 4, 9, 15, 23, 34, 49, 58]
C = 160          # 半边
Z = 1.6          # 放大倍数
T = int(2 * C * Z)
cols, rows = 4, 2
sheet = Image.new('RGB', (T * cols, T * rows), 'white')
try:
    font = ImageFont.truetype('arial.ttf', 18)
except Exception:
    font = ImageFont.load_default()

for k, i in enumerate(picks):
    im = Image.open(ROOT / 'train' / f'{i:04d}.jpg').convert('RGB')
    W, H = im.size
    X, Y = gt[i]
    box = (int(X - C), int(Y - C), int(X + C), int(Y + C))
    crop = im.crop(box).resize((T, T), Image.LANCZOS)
    tx, ty = (k % cols) * T, (k // cols) * T
    sheet.paste(crop, (tx, ty))
    d = ImageDraw.Draw(sheet)
    m = T // 2
    d.line([(tx + m - 18, ty + m), (tx + m + 18, ty + m)], fill=(255, 0, 0), width=2)
    d.line([(tx + m, ty + m - 18), (tx + m, ty + m + 18)], fill=(255, 0, 0), width=2)
    # 100px 半径的参考圆（原图尺度 x Z）
    r100 = 100 * Z
    d.ellipse([tx + m - r100, ty + m - r100, tx + m + r100, ty + m + r100],
              outline=(0, 255, 255), width=2)
    d.text((tx + 5, ty + 3), f'#{i:04d}  {W}x{H}   (r=100px)', fill=(0, 255, 0), font=font)
    d.rectangle([tx, ty, tx + T - 1, ty + T - 1], outline=(80, 80, 80))

p = OUT / 'fovea_closeup.png'
sheet.save(p, optimize=True)
print('[saved]', p, sheet.size, p.stat().st_size, 'bytes')
print('每格 = 原图 320x320 像素，放大 1.6 倍；青圈 = 原图半径 100px（约 1 个视盘半径）')
