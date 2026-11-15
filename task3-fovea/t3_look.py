# -*- coding: utf-8 -*-
"""任务三：把若干 train 图缩略并标出 GT 黄斑位置，拼成一张 kontakt 表以便肉眼判断结构。"""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
OUTDIR = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\t3_look')
OUTDIR.mkdir(parents=True, exist_ok=True)

gt = {}
with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        gt[int(r['data'])] = (float(r['Fovea_X']), float(r['Fovea_Y']))

# 按 X 从小到大挑 6 张，覆盖整个跨度
picks = [1, 6, 49, 32, 3, 24]

TW, TH = 620, 420
sheet = Image.new('RGB', (TW * 3, TH * 2), 'white')
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype('arial.ttf', 22)
except Exception:
    font = ImageFont.load_default()

for k, i in enumerate(picks):
    im = Image.open(ROOT / 'train' / f'{i:04d}.jpg').convert('RGB')
    W, H = im.size
    s = min(TW / W, TH / H)
    thumb = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
    tx, ty = (k % 3) * TW, (k // 3) * TH
    sheet.paste(thumb, (tx, ty))
    d = ImageDraw.Draw(sheet)
    x, y = gt[i][0] * s, gt[i][1] * s
    # 十字准线
    d.line([(tx + x - 26, ty + y), (tx + x + 26, ty + y)], fill=(255, 0, 0), width=3)
    d.line([(tx + x, ty + y - 26), (tx + x, ty + y + 26)], fill=(255, 0, 0), width=3)
    d.ellipse([tx + x - 40, ty + y - 40, tx + x + 40, ty + y + 40], outline=(255, 255, 0), width=2)
    d.text((tx + 8, ty + 6),
           f'#{i:04d}  {W}x{H}\nGT=({gt[i][0]:.0f},{gt[i][1]:.0f})\n'
           f'norm=({gt[i][0]/W:.3f},{gt[i][1]/H:.3f})',
           fill=(0, 255, 0), font=font)
    d.rectangle([tx, ty, tx + TW - 1, ty + TH - 1], outline=(90, 90, 90))

p = OUTDIR / 'train_gt_overview.png'
sheet.save(p, optimize=True)
print('[saved]', p, sheet.size, p.stat().st_size, 'bytes')
