# -*- coding: utf-8 -*-
"""任务三 · 多假设滑窗（用已训好的 1:1 细化器当「检测器 + 定位器」）。

动机（见 b23）：第二级在「先验误差 <60px」时 RMSE 15.13（榜首档），
但先验 ≥60px 的 31 张修不动——因为 CROP=256 只覆盖 ±128，黄斑根本不在裁块里。

做法：在基先验周围铺一个 G×G 的假设网格（步长 STEP），每个假设都让细化器
从其中心出发生成一个预测点。真正落在黄斑上的假设会彼此收敛到同一点，
所以对预测点做「人以群分」的聚类，取最大簇的质心。
这一步不训练任何东西。
"""
import sys, json, math, argparse
from pathlib import Path
import numpy as np
import torch

INB = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
sys.path.insert(0, str(INB))
import t3_refine as R                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--prior', default='t3_geom2.json')
ap.add_argument('--models', default='t3_run_refine_full2')
ap.add_argument('--grid', type=int, default=3)
ap.add_argument('--step', type=float, default=110.0)
ap.add_argument('--radius', type=float, default=30.0)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--emit', default='t3_pred_slide.json')
args = ap.parse_args()
torch.set_num_threads(args.threads)

gt = R.load_gt()
prior = json.loads((INB / args.prior).read_text(encoding='utf-8'))
mdir = INB / args.models
ids = sorted(gt)
rng = np.random.default_rng(42)
perm = rng.permutation(len(ids))
folds = [perm[j::args.folds] for j in range(args.folds)]
fold_of = {}
for fi, f in enumerate(folds):
    for k in f:
        fold_of[int(ids[k])] = fi

off = np.linspace(-(args.grid // 2), args.grid // 2, args.grid) * args.step
OFFS = [(float(dx), float(dy)) for dy in off for dx in off]
print(f'[slide] 网格 {args.grid}x{args.grid} 步长 {args.step}  共 {len(OFFS)} 个假设', flush=True)


def cluster(points, radius):
    """返回最大簇的质心 + 该簇大小；平票时取离基先验最近的簇。"""
    P = np.asarray(points, float)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    cnt = (D <= radius).sum(1)
    best = int(np.argmax(cnt))
    sel = D[best] <= radius
    return P[sel].mean(0), int(cnt[best])


def run(item, cx, cy, models_for_item):
    pts, per_h = [], []
    for dx, dy in OFFS:
        ps = [R.predict_one(m, item, cx + dx, cy + dy, 1, 3.0) for m in models_for_item]
        p = np.mean(ps, 0)
        pts.append(p)
        per_h.append((dx, dy, float(p[0]), float(p[1])))
    return pts, per_h


dev = torch.device('cpu')
models = {}
for fi in range(args.folds):
    m = R.Refiner().to(dev)
    m.load_state_dict(torch.load(mdir / f'fold{fi}.pt', map_location=dev))
    m.eval()
    models[fi] = m

# ---------- 训练集：OOF（每张用自己那折的模型）----------
tr_cache = R.ROICache(prior['train'], 'train')
items = {it['i']: it for it in tr_cache}
res, errs = [], []
for i in ids:
    it = items[i]
    cx, cy = prior['train'][f'{i:04d}']
    pts, _ = run(it, cx, cy, [models[fold_of[i]]])
    c, n = cluster(pts, args.radius)
    allmean = np.mean(pts, 0)
    X, Y = gt[i]
    e_clu = math.hypot(c[0] - X, c[1] - Y)
    e_mean = math.hypot(allmean[0] - X, allmean[1] - Y)
    e_prior = math.hypot(cx - X, cy - Y)
    errs.append((e_clu, e_mean, e_prior))
    res.append(dict(i=i, clu=[float(c[0]), float(c[1])], n=n, mean=[float(allmean[0]), float(allmean[1])],
                    prior=[cx, cy], true=[X, Y]))
E = np.array(errs)
for k, name in ((0, '聚类'), (1, '9点均值'), (2, '基先验')):
    e = E[:, k]
    print(f'{name:8s} mean={e.mean():6.2f} med={np.median(e):6.2f} p90={np.percentile(e,90):6.2f} '
          f'max={e.max():7.2f} RMSE={math.sqrt((e**2).mean()/2):6.2f}')
ns = np.array([r['n'] for r in res])
print(f'簇大小分布: 1点 {(ns==1).sum()} | 2点 {(ns==2).sum()} | 3+ {(ns>=3).sum()}')
print('聚类赢基先验的张数:', int((E[:, 0] < E[:, 2]).sum()), '/', len(ids))

# ---------- 测试集：5 折集成 ----------
te_cache = R.ROICache(prior['test'], 'test')
pred = {}
for it in te_cache:
    cx, cy = prior['test'][f'{it["i"]:04d}']
    pts, _ = run(it, cx, cy, [models[fi] for fi in range(args.folds)])
    c, n = cluster(pts, args.radius)
    pred[it['i']] = [float(c[0]), float(c[1])]
    print(f'  test {it["i"]:04d} prior=({cx:7.1f},{cy:7.1f}) -> ({c[0]:7.1f},{c[1]:7.1f}) 簇={n}', flush=True)
(INB / args.emit).write_text(json.dumps({str(k): v for k, v in pred.items()}), encoding='utf-8')
(INB / 't3_run_slide_oof.json').write_text(json.dumps(dict(
    per=res, rmse_clu=float(math.sqrt((E[:, 0] ** 2).mean() / 2)),
    rmse_mean=float(math.sqrt((E[:, 1] ** 2).mean() / 2)),
    rmse_prior=float(math.sqrt((E[:, 2] ** 2).mean() / 2))), ensure_ascii=False), encoding='utf-8')
print('[written]', INB / args.emit, '和 t3_run_slide_oof.json')
