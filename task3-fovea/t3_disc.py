# -*- coding: utf-8 -*-
"""任务三：视盘（optic disc）定位 + 几何法黄斑预测，并对 80 张训练图做留一验证。

做法（不含任何训练，纯测量 + 几何）：
  1. 缩到长边 800，取视网膜掩膜（非黑 + 最大连通域 + 填洞）。
  2. 视盘 = 视网膜内"亮度最高的一团"：亮度高分位阈值 -> 开运算 -> 连通域 -> 面积/紧凑度筛选 -> 取平均亮度最高者。
  3. 黄斑相对视盘：dx = X - cx, dy = Y - cy，用视盘半径 R 归一化。按 dfx 正负分左右两组。
  4. 留一法：用 79 张拟合 (u, v) 常数（分左右），预测被留那张，算原图像素欧氏距离。
输出：数值报告 + 一张视盘检出情况的 kontakt 表。
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
OUTDIR = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\t3_look')
OUTDIR.mkdir(parents=True, exist_ok=True)
LONG = 800

gt = {}
with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        gt[int(r['data'])] = (float(r['Fovea_X']), float(r['Fovea_Y']))


def load_small(i, split='train'):
    im = Image.open(ROOT / split / f'{i:04d}.jpg').convert('RGB')
    W, H = im.size
    s = LONG / max(W, H)
    sm = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
    return np.asarray(sm).astype(np.float32) / 255.0, (W, H), s


def retina_mask(rgb):
    mx = rgb.max(2)
    m = mx > 0.12
    m = ndi.binary_closing(m, np.ones((7, 7)))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    return m


def detect_disc(rgb, retina):
    L = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    L = ndi.gaussian_filter(L, 2.0)
    area_ret = int(retina.sum())
    info = {'area_ret': area_ret}
    if area_ret < 1000:
        return None, info
    best = None
    for pct in (98.5, 97.0, 95.0, 92.0):
        thr = np.percentile(L[retina], pct)
        m = (L >= thr) & retina
        m = ndi.binary_opening(m, np.ones((5, 5)))
        lab, n = ndi.label(m)
        if n == 0:
            continue
        objs = ndi.find_objects(lab)
        cands = []
        for k in range(1, n + 1):
            sl = objs[k - 1]
            if sl is None:
                continue
            comp = lab[sl] == k
            a = int(comp.sum())
            if not (0.0012 * area_ret <= a <= 0.06 * area_ret):
                continue
            h, w = comp.shape
            comp_ = area_ret and a / (np.pi * (max(h, w) / 2.0) ** 2)
            if comp_ < 0.35:
                continue
            ys, xs = np.nonzero(comp)
            cy = sl[0].start + ys.mean()
            cx = sl[1].start + xs.mean()
            meanL = float(L[sl][comp].mean())
            R = float(np.sqrt(a / np.pi))
            cands.append(dict(cx=cx, cy=cy, R=R, area=a, meanL=meanL,
                              compact=float(comp_), pct=pct))
        if cands:
            cands.sort(key=lambda c: (c['meanL'], c['area']), reverse=True)
            best = cands[0]
            info['n_cand'] = len(cands)
            break
    return best, info


rows = []
for i in sorted(gt):
    rgb, wh, s = load_small(i)
    retina = retina_mask(rgb)
    d, info = detect_disc(rgb, retina)
    if d is None:
        rows.append(dict(i=i, ok=False, wh=wh, s=s))
        continue
    cx_full, cy_full, R_full = d['cx'] / s, d['cy'] / s, d['R'] / s
    X, Y = gt[i]
    rows.append(dict(i=i, ok=True, wh=wh, s=s, cx=cx_full, cy=cy_full, R=R_full,
                     dx=X - cx_full, dy=Y - cy_full, area_frac=d['area'] / info['area_ret'],
                     meanL=d['meanL'], compact=d['compact'], pct=d['pct'], n=info.get('n_cand')))

ok = [r for r in rows if r['ok']]
print(f'检出视盘: {len(ok)}/{len(rows)}')
print(f'视盘半径 R(原图 px): mean={np.mean([r["R"] for r in ok]):.1f} '
      f'min={min(r["R"] for r in ok):.1f} max={max(r["R"] for r in ok):.1f}')
print(f'面积占比: mean={np.mean([r["area_frac"] for r in ok])*100:.2f}% '
      f'min={min(r["area_frac"] for r in ok)*100:.2f}% max={max(r["area_frac"] for r in ok)*100:.2f}%')

for r in ok:
    r['u'] = r['dx'] / r['R']
    r['v'] = r['dy'] / r['R']
    r['side'] = 1 if r['u'] > 0 else -1

print('\n=== dx/R（u）与 dy/R（v）分布 ===')
for sd, nm in ((1, '黄斑在视盘右侧'), (-1, '黄斑在视盘左侧')):
    g = [r for r in ok if r['side'] == sd]
    if not g:
        continue
    us = np.array([r['u'] for r in g])
    vs = np.array([r['v'] for r in g])
    print(f'{nm} (n={len(g)}):  u mean={us.mean():.3f} std={us.std():.3f} '
          f'[{us.min():.2f}, {us.max():.2f}]   v mean={vs.mean():.3f} std={vs.std():.3f} '
          f'[{vs.min():.2f}, {vs.max():.2f}]')

print('\n=== 留一法：几何模型 X = cx + med_u*R, Y = cy + med_v*R（u/v 已带符号，不再乘 side）===')
errs = []
for k, r in enumerate(ok):
    train = [o for j, o in enumerate(ok) if j != k]
    same = [o for o in train if o['side'] == r['side']] or train
    mu = float(np.median([o['u'] for o in same]))
    mv = float(np.median([o['v'] for o in same]))
    px = r['cx'] + mu * r['R']
    py = r['cy'] + mv * r['R']
    e = float(np.hypot(px - gt[r['i']][0], py - gt[r['i']][1]))
    errs.append(e)
    r['pred_err'] = e
    r['mu'] = mu
    r['mv'] = mv
errs = np.array(errs)
print(f'欧氏距离 LOO: mean={errs.mean():.1f}px  median={np.median(errs):.1f}px  '
      f'p90={np.percentile(errs,90):.1f}px  max={errs.max():.1f}px')
print(f'归一化(按长边 2992): mean={errs.mean()/2992:.4f}  中位={np.median(errs)/2992:.4f}')

print('\n=== u 排序（找视盘检出失败）===')
for r in sorted(ok, key=lambda z: z['u']):
    flag = '' if 4.5 <= abs(r['u']) <= 9.0 else '  <== 可疑'
    print(f'  #{r["i"]:04d} u={r["u"]:+7.2f} v={r["v"]:+6.2f} R={r["R"]:6.1f} '
          f'area={r["area_frac"]*100:5.2f}% err={r["pred_err"]:7.1f}px{flag}')

print('\n最差的 8 张：')
for r in sorted(ok, key=lambda z: -z['pred_err'])[:8]:
    print(f'  #{r["i"]:04d} err={r["pred_err"]:7.1f}px  u={r["u"]:+.2f} v={r["v"]:+.2f} '
          f'R={r["R"]:6.1f} area={r["area_frac"]*100:.2f}% L={r["meanL"]:.3f} '
          f'n={r["n"]} pct={r["pct"]}')

# 画检出情况
picks = [r['i'] for r in ok[:9]]
TW, TH = 620, 420
sheet = Image.new('RGB', (TW * 3, TH * 3), 'white')
try:
    font = ImageFont.truetype('arial.ttf', 20)
except Exception:
    font = ImageFont.load_default()
for k, i in enumerate(picks):
    r = next(z for z in ok if z['i'] == i)
    im = Image.open(ROOT / 'train' / f'{i:04d}.jpg').convert('RGB')
    W, H = im.size
    s = min(TW / W, TH / H)
    tx, ty = (k % 3) * TW, (k // 3) * TH
    sheet.paste(im.resize((int(W * s), int(H * s)), Image.LANCZOS), (tx, ty))
    d = ImageDraw.Draw(sheet)
    X, Y = gt[i]
    d.line([(tx + X * s - 24, ty + Y * s), (tx + X * s + 24, ty + Y * s)], fill=(255, 0, 0), width=3)
    d.line([(tx + X * s, ty + Y * s - 24), (tx + X * s, ty + Y * s + 24)], fill=(255, 0, 0), width=3)
    cx, cy, R = r['cx'] * s, r['cy'] * s, r['R'] * s
    d.ellipse([tx + cx - R, ty + cy - R, tx + cx + R, ty + cy + R], outline=(0, 255, 255), width=3)
    d.line([(tx + cx - 12, ty + cy), (tx + cx + 12, ty + cy)], fill=(0, 255, 255), width=2)
    d.line([(tx + cx, ty + cy - 12), (tx + cx, ty + cy + 12)], fill=(0, 255, 255), width=2)
    d.text((tx + 6, ty + 4), f'#{i:04d} R={R/s:.0f} err={r["pred_err"]:.0f}px', fill=(0, 255, 0), font=font)
    d.rectangle([tx, ty, tx + TW - 1, ty + TH - 1], outline=(90, 90, 90))
p = OUTDIR / 'disc_detect.png'
sheet.save(p, optimize=True)
print('\n[saved]', p)
