# -*- coding: utf-8 -*-
"""任务三：几种"零模型/极简模型"基线的留一误差对比，用来定调模型上限与主战场在哪一维。"""
import csv, re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')

gt, wh = {}, {}
with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        gt[int(r['data'])] = (float(r['Fovea_X']), float(r['Fovea_Y']))
for i in gt:
    with Image.open(ROOT / 'train' / f'{i:04d}.jpg') as im:
        wh[i] = im.size

# XML 给出的 Fovea 框中心 = 独立的"人工标注点"，可作为 bndbox 精度参考
box = {}
for f in sorted((ROOT / 'train_location').glob('*.xml')):
    t = f.read_text(encoding='utf-8', errors='replace')
    g = lambda tag: float(re.search(rf'<{tag}>\s*([\-\d\.]+)\s*</{tag}>', t).group(1))
    box[int(f.stem)] = ((g('xmin') + g('xmax')) / 2, (g('ymin') + g('ymax')) / 2)

ids = sorted(gt)
size_key = {i: wh[i] for i in ids}


def loo(predict):
    """predict(train_idx, test_i) -> (x, y)"""
    errs = []
    for k, i in enumerate(ids):
        tr = [j for j in ids if j != i]
        px, py = predict(tr, i)
        errs.append(float(np.hypot(px - gt[i][0], py - gt[i][1])))
    return np.array(errs)


def rep(name, errs):
    print(f'{name:46s} mean={errs.mean():7.1f}  median={np.median(errs):7.1f}  '
          f'p90={np.percentile(errs,90):7.1f}  max={errs.max():7.1f}')


print(f'训练集 {len(ids)} 张；尺寸分组：' +
      ', '.join(f'{k}×{v}={sum(1 for i in ids if size_key[i]==k)}' for k, v in
                {s: None for s in set(size_key.values())}.items()))

# 1) 全局常数（不拟合，直接用全体均值；留一几乎不变）
gx, gy = np.mean([gt[i][0] for i in ids]), np.mean([gt[i][1] for i in ids])
rep(f'全局常数点 ({gx:.0f},{gy:.0f})',
    np.array([np.hypot(gx - gt[i][0], gy - gt[i][1]) for i in ids]))

# 2) 分尺寸常数（LOO）
def pred_size(tr, i):
    g = [j for j in tr if wh[j] == wh[i]]
    return np.mean([gt[j][0] for j in g]), np.mean([gt[j][1] for j in g])
rep('分尺寸常数点（LOO）', loo(pred_size))

# 3) 分尺寸 + 分左右（用 XML 框中心是否在图像左/右半边作为"眼别"代理：先用 GT 的 X 相对图中线）
def side(j):
    return 1 if gt[j][0] > wh[j][0] / 2 else -1

def pred_size_side(tr, i):
    g = [j for j in tr if wh[j] == wh[i] and side(j) == side(i)]
    if len(g) < 3:
        g = [j for j in tr if side(j) == side(i)] or tr
    return np.mean([gt[j][0] for j in g]), np.mean([gt[j][1] for j in g])
rep('分尺寸+分左右 常数点（LOO，用 GT 判眼别=作弊上限）', loo(pred_size_side))

# 4) 只预测 Y 的常数、X 用真值 -> 看 X 单独贡献多少误差
def pred_yonly(tr, i):
    return gt[i][0], np.mean([gt[j][1] for j in tr])
rep('X 用真值 + Y 取常数（下界参考）', loo(pred_yonly))

def pred_xonly(tr, i):
    return np.mean([gt[j][0] for j in tr]), gt[i][1]
rep('Y 用真值 + X 取常数（下界参考）', loo(pred_xonly))

# 5) XML 框中心当预测（评估 bndbox 本身的精度）
errs = np.array([float(np.hypot(box[i][0] - gt[i][0], box[i][1] - gt[i][1])) for i in ids])
rep('XML bndbox 中心 vs CSV 精确点', errs)

# 6) 各维度的"可解释上界"：X 与 Y 的绝对偏差分别多少
dx = np.array([gt[i][0] - gx for i in ids])
dy = np.array([gt[i][1] - gy for i in ids])
print(f'\nX 偏差 |dx|: mean={np.abs(dx).mean():.1f}  std={dx.std():.1f}')
print(f'Y 偏差 |dy|: mean={np.abs(dy).mean():.1f}  std={dy.std():.1f}')
print(f'若 X、Y 都完美预测，误差 = 0；常数点的误差主要来自 '
      f'X({np.abs(dx).mean():.0f}px) 与 Y({np.abs(dy).mean():.0f}px) 的联合。')

# 7) 尺寸组内的 X 是否存在"归一化后更稳"的结构
print('\n按尺寸分组的 GT 统计：')
for s in sorted(set(wh.values())):
    g = [i for i in ids if wh[i] == s]
    xs = np.array([gt[i][0] for i in g]); ys = np.array([gt[i][1] for i in g])
    print(f'  {s}: n={len(g)}  X mean={xs.mean():7.1f} std={xs.std():6.1f}  '
          f'Y mean={ys.mean():7.1f} std={ys.std():5.1f}  '
          f'X/W={np.mean([gt[i][0]/wh[i][0] for i in g]):.4f}')
