# -*- coding: utf-8 -*-
"""比较几何先验 / 第一级宽视野先验 / 两者混合，并给出「谁在哪张上赢」。

注意：在 OOF 上挑混合权重正是任务二翻车的那个坑。
这里只作为诊断；默认报告 w=0.5 的固定混合，扫权重曲线只用来看它是否平坦。
"""
import sys, json, math
from pathlib import Path
import numpy as np

INB = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
sys.path.insert(0, str(INB))
import t3_refine as R                                   # noqa: E402

gt = R.load_gt()
g1 = json.loads((INB / 't3_geom.json').read_text(encoding='utf-8'))['train']
g2 = json.loads((INB / 't3_geom2.json').read_text(encoding='utf-8'))['train']
ids = sorted(gt)


def rmse(P):
    T = np.array([gt[i] for i in ids], float)
    d = np.asarray(P, float) - T
    e = np.hypot(d[:, 0], d[:, 1])
    return math.sqrt((e ** 2).mean() / 2), e, d


A = np.array([g1[f'{i:04d}'] for i in ids], float)      # 几何
B = np.array([g2[f'{i:04d}'] for i in ids], float)      # 第一级
for name, P in (('几何先验', A), ('第一级宽视野', B)):
    r, e, d = rmse(P)
    print(f'{name:10s} mean={e.mean():6.2f} med={np.median(e):6.2f} '
          f'p90={np.percentile(e,90):6.2f} max={e.max():7.2f} RMSE={r:6.2f} '
          f'| |e|>60 的张数={(e>60).sum():2d} |e|>100 的张数={(e>100).sum():2d}')

print('\n混合 w*第一级 + (1-w)*几何：')
best = None
for w in np.arange(0, 1.0001, 0.1):
    r, e, _ = rmse(w * B + (1 - w) * A)
    print(f'  w={w:.1f}  RMSE={r:6.2f}  mean={e.mean():6.2f}  max={e.max():7.2f}  >60 张数={(e>60).sum():2d}')
    if best is None or r < best[1]:
        best = (w, r)
print(f'  OOF 最优 w={best[0]:.1f} RMSE={best[1]:.2f}（★在 OOF 上挑的，不可当证据）')

print('\n逐张对比（几何 err / 第一级 err / 谁赢）：')
rA, eA, _ = rmse(A)
rB, eB, _ = rmse(B)
win_b = (eB < eA)
print(f'  第一级赢 {win_b.sum()} 张 / 几何赢 {(~win_b).sum()} 张')
order = np.argsort(-(eA - eB))
print('  第一级改善最大的 8 张:', [(f'{ids[k]:04d}', round(float(eA[k]), 1), round(float(eB[k]), 1)) for k in order[:8]])
print('  第一级变差最多的 8 张:', [(f'{ids[k]:04d}', round(float(eA[k]), 1), round(float(eB[k]), 1)) for k in order[-8:]])
