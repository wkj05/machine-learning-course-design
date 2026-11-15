# -*- coding: utf-8 -*-
"""1) 把几何基线按官方 RMSE 口径重算（RMSE = sqrt(mean(d^2)/2)，d 为逐图欧氏距离）
   2) 把 20 张测试图在「预测黄斑位置」处按原分辨率裁 400x400 拼图，用于判断中央凹是否可见。"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

INB = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
ROOT = INB / 'task3' / 'detection'
LOOK = INB / 't3_look'


def gt_load():
    g = {}
    with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8-sig') as f:
        next(f)
        for ln in f:
            if ln.strip():
                d, x, y = ln.split(',')[:3]
                g[int(d)] = (float(x), float(y))
    return g


def stats(tag, d):
    d = np.asarray(d, float)
    mean = d.mean()
    rms = math.sqrt((d ** 2).mean())
    print(f'{tag:22s} n={len(d):3d}  mean={mean:7.2f}  median={np.median(d):7.2f}  '
          f'p90={np.percentile(d,90):7.2f}  max={d.max():7.2f}  '
          f'RMS={rms:7.2f}  **RMSE={rms/math.sqrt(2):7.2f}**')


def main():
    gt = gt_load()
    geom = json.loads((INB / 't3_geom.json').read_text(encoding='utf-8'))
    d = []
    for k, v in geom['train'].items():
        X, Y = gt[int(k)]
        d.append(math.hypot(v[0] - X, v[1] - Y))
    print('=== 按官方 RMSE 口径（RMSE = RMS(d)/sqrt(2)）===')
    stats('几何基线 LOO', d)
    # 对照：常数点
    dc = [math.hypot(1446.11 - X, 1055.78 - Y) for X, Y in gt.values()]
    stats('常数点 (1446,1056)', dc)

    # 测试图中央凹近景
    cols, tw = 5, 300
    ids = list(range(81, 101))
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, rows * tw), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for n, i in enumerate(ids):
        px, py = geom['test'][f'{i:04d}']
        im = Image.open(ROOT / 'test' / f'{i:04d}.jpg').convert('RGB')
        W, H = im.size
        box = (int(px) - 200, int(py) - 200, int(px) + 200, int(py) + 200)
        box = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
        cr = im.crop(box).resize((tw, tw), Image.LANCZOS)
        ox, oy = (n % cols) * tw, (n // cols) * tw
        sheet.paste(cr, (ox, oy))
        c = tw / 2
        dr.ellipse([ox + c - 26, oy + c - 26, ox + c + 26, oy + c + 26],
                   outline=(0, 255, 255), width=2)
        dr.line([ox + c - 34, oy + c, ox + c + 34, oy + c], fill=(255, 60, 60), width=2)
        dr.line([ox + c, oy + c - 34, ox + c, oy + c + 34], fill=(255, 60, 60), width=2)
        dr.text((ox + 5, oy + 4), f'#{i:04d}  原图 400px 见方', fill=(0, 255, 120))
        im.close()
    sheet.save(LOOK / 'test_fovea_zoom.png')
    print(f'[written] {LOOK / "test_fovea_zoom.png"}  {sheet.size}')


main()
