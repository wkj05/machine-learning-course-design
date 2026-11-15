# -*- coding: utf-8 -*-
"""任务三（视网膜眼底图 黄斑中心凹定位）数据探查：只读测量，不改任何文件。"""
import csv, os, re, statistics as st
from pathlib import Path
from PIL import Image

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
OUT = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s)


# ---------- 1. 尺寸 ----------
sizes = {}
for split in ('train', 'test'):
    for f in sorted((ROOT / split).iterdir()):
        with Image.open(f) as im:
            sizes.setdefault((split, im.size, im.mode), []).append(f.name)

p('=== 尺寸分布 ===')
for (split, size, mode), names in sorted(sizes.items()):
    p(f'{split:6s} {str(size):14s} mode={mode}  n={len(names)}  eg={names[0]}')

# ---------- 2. GT ----------
gt = {}
with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        gt[int(row['data'])] = (float(row['Fovea_X']), float(row['Fovea_Y']))

p('\n=== GT 统计 (n=%d) ===' % len(gt))
xs = [v[0] for v in gt.values()]
ys = [v[1] for v in gt.values()]
for nm, arr in (('X', xs), ('Y', ys)):
    p(f'{nm}: mean={st.mean(arr):8.2f} std={st.pstdev(arr):7.2f} '
      f'min={min(arr):8.2f} max={max(arr):8.2f}')

# 归一化坐标（用各图真实边长）
nx, ny, wh = [], [], {}
for i, (x, y) in sorted(gt.items()):
    with Image.open(ROOT / 'train' / f'{i:04d}.jpg') as im:
        w, h = im.size
    wh[i] = (w, h)
    nx.append(x / w)
    ny.append(y / h)
p(f'\n归一化 X: mean={st.mean(nx):.4f} std={st.pstdev(nx):.4f} min={min(nx):.4f} max={max(nx):.4f}')
p(f'归一化 Y: mean={st.mean(ny):.4f} std={st.pstdev(ny):.4f} min={min(ny):.4f} max={max(ny):.4f}')

# 各图宽高是否一致
uniq = set(wh.values())
p(f'\ntrain 图的 (w,h) 种类数 = {len(uniq)} -> {sorted(uniq)[:6]}')

# ---------- 3. XML 与 CSV 的一致性 ----------
p('\n=== train_location XML vs CSV ===')
rows = []
for f in sorted((ROOT / 'train_location').glob('*.xml')):
    txt = f.read_text(encoding='utf-8', errors='replace')
    i = int(f.stem)
    def g(tag):
        m = re.search(rf'<{tag}>\s*([\-\d\.]+)\s*</{tag}>', txt)
        return float(m.group(1)) if m else None
    xmin, ymin, xmax, ymax = (g(t) for t in ('xmin', 'ymin', 'xmax', 'ymax'))
    W, H = g('width'), g('height')
    if xmin is None:
        rows.append((i, 'NO_BNDBOX', None, None, None))
        continue
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    rows.append((i, 'ok', cx, cy, (W, H, xmax - xmin, ymax - ymin)))

bad = [r for r in rows if r[1] != 'ok']
p(f'XML 文件 {len(rows)} 个，缺失 bndbox 的 {len(bad)} 个')
dx = [r[2] - gt[r[0]][0] for r in rows if r[1] == 'ok']
dy = [r[3] - gt[r[0]][1] for r in rows if r[1] == 'ok']
p(f'box 中心 - CSV 点:  dX mean={st.mean(dx):7.2f} std={st.pstdev(dx):6.2f} max|{max(map(abs,dx)):.2f}|')
p(f'                  dY mean={st.mean(dy):7.2f} std={st.pstdev(dy):6.2f} max|{max(map(abs,dy)):.2f}|')
bw = [r[4][2] for r in rows if r[1] == 'ok']
bh = [r[4][3] for r in rows if r[1] == 'ok']
p(f'box 尺寸: w {min(bw)}~{max(bw)} (mean {st.mean(bw):.1f}), h {min(bh)}~{max(bh)} (mean {st.mean(bh):.1f})')
xmlWH = set((r[4][0], r[4][1]) for r in rows if r[1] == 'ok')
p(f'XML 声明的 (width,height) 种类数 = {len(xmlWH)} -> {sorted(xmlWH)[:6]}')

# XML 声明尺寸 vs 磁盘真实尺寸
mism = [(i, wh[i], r[4][:2]) for i, *rest in [(r[0], r) for r in rows if r[1] == 'ok']
        for r in [rest[0]] if wh.get(i) != tuple(int(v) for v in r[4][:2])]
p(f'XML 尺寸与磁盘不符的图数 = {len(mism)}')

# ---------- 4. 只看坐标，不加任何模型：常数预测的误差下界 ----------
def euclid(px, py):
    return [((px - gt[i][0]) ** 2 + (py - gt[i][1]) ** 2) ** 0.5 for i in gt]

mx, my = st.mean(xs), st.mean(ys)
d_const = euclid(mx, my)
p('\n=== 常数预测（均值点）基线 ===')
p(f'预测点 ({mx:.1f}, {my:.1f})')
p(f'欧氏距离: mean={st.mean(d_const):.2f}px  median={st.median(d_const):.2f}px  max={max(d_const):.2f}px')
p(f'50% 分位内的距离阈值(中位数) = {st.median(d_const):.2f}px ; 归一化后 mean={st.mean(d_const)/ (1956):.4f} (以 1956 宽为分母)')

Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\t3_probe.txt').write_text(
    '\n'.join(OUT), encoding='utf-8')
print('\n[written] _inbox/t3_probe.txt')
