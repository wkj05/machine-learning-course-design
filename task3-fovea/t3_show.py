# -*- coding: utf-8 -*-
"""把预测点画到图上，用于目视核对（几何基线 / 模型输出都可用）。"""
import sys, json, argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
INB = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
LOOK = INB / 't3_look'


def draw_many(split, pred, out, cols=5, tw=380, gt=None):
    ids = sorted(int(k) for k in pred)
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, rows * tw), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for n, i in enumerate(ids):
        im = Image.open(ROOT / split / f'{i:04d}.jpg').convert('RGB')
        W, H = im.size
        s = tw / max(W, H)
        im = im.resize((int(W * s), int(H * s)), Image.BILINEAR)
        ox, oy = (n % cols) * tw, (n // cols) * tw
        sheet.paste(im, (ox, oy))
        x, y = pred[str(i)] if str(i) in pred else pred[i]
        cx, cy = ox + x * s, oy + y * s
        r = 26
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 255), width=2)
        d.line([cx - r - 8, cy, cx + r + 8, cy], fill=(255, 40, 40), width=2)
        d.line([cx, cy - r - 8, cx, cy + r + 8], fill=(255, 40, 40), width=2)
        # 视盘位置的粗指示（几何预测的锚点）
        d.text((ox + 6, oy + 4), f'#{i:04d} {W}x{H}', fill=(0, 255, 120))
        if gt and str(i) in gt:
            gx, gy = gt[str(i)]
            d.line([ox + gx * s - 10, oy + gy * s, ox + gx * s + 10, oy + gy * s],
                   fill=(255, 255, 0), width=2)
            d.line([ox + gx * s, oy + gy * s - 10, ox + gx * s, oy + gy * s + 10],
                   fill=(255, 255, 0), width=2)
    sheet.save(out)
    print(f'[written] {out}  {sheet.size[0]}x{sheet.size[1]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default=str(LOOK / 'test_pred.png'))
    ap.add_argument('--cols', type=int, default=5)
    a = ap.parse_args()
    LOOK.mkdir(exist_ok=True)
    draw_many(a.split, json.loads(Path(a.pred).read_text(encoding='utf-8')), a.out, a.cols)
