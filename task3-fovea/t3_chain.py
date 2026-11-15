# -*- coding: utf-8 -*-
"""任务三 级联评估：第一级（宽视野）先验 + 第二级（1:1 细化，复用 t3_run_refine_full2 的折模型）。

不需要重训第二级：
- 训练集用「该图所属折的那个模型」在其验证折上预测（与训练时的评估协议完全一致，无泄漏）
- 测试集用 5 折集成
第一级的训练集先验是留一折预测，测试集先验是 5 折集成，同样无泄漏。
"""
import sys, json, math, argparse
from pathlib import Path
import numpy as np
import torch

INB = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
sys.path.insert(0, str(INB))
import t3_refine as R                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--prior', default='t3_geom2.json')     # 第一级产出的先验
ap.add_argument('--models', default='t3_run_refine_full2')
ap.add_argument('--iters', type=int, default=1)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--emit', default='t3_pred_chain.json')
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

print(f'[data] 训练 {len(ids)} 张，先验 mean/med 先算一下：', flush=True)
pr = np.array([prior['train'][f'{i:04d}'] for i in ids], float)
tr = np.array([gt[i] for i in ids], float)
e0 = np.hypot(*(pr - tr).T)
print(f'  第一级先验 mean={e0.mean():.2f} med={np.median(e0):.2f} '
      f'p90={np.percentile(e0,90):.2f} max={e0.max():.2f}', flush=True)

tr_cache = R.ROICache(prior['train'], 'train')
items = {it['i']: it for it in tr_cache}
print(f'[data] train ROI {len(items)} 张，{2*R.ROI_HALF}px 见方', flush=True)

dev = torch.device('cpu')
models = {}
for fi in range(args.folds):
    m = R.Refiner().to(dev)
    m.load_state_dict(torch.load(mdir / f'fold{fi}.pt', map_location=dev))
    m.eval()
    models[fi] = m

print('[eval] 逐张用「它所属折的模型」预测', flush=True)
errs, per = [], []
for i in ids:
    it = items[i]
    fi = fold_of[i]
    cx, cy = prior['train'][f'{i:04d}']
    px, py = R.predict_one(models[fi], it, cx, cy, args.iters, 3.0)
    X, Y = gt[i]
    d = math.hypot(px - X, py - Y)
    errs.append(d)
    per.append(dict(i=i, ref=[px, py], prior=[cx, cy], geom=prior['train'][f'{i:04d}'], true=[X, Y]))
e = np.array(errs)
print(f'\n级联 OOF n={len(e)}  mean={e.mean():.2f}  median={np.median(e):.2f}  '
      f'p90={np.percentile(e,90):.2f}  max={e.max():.2f}  RMSE={math.sqrt((e**2).mean()/2):.2f}')
print(f'对比：先验 alone mean={e0.mean():.2f} RMSE={math.sqrt((e0**2).mean()/2):.2f}')

print('\n[test] 从第一级先验出发，5 折集成', flush=True)
te_cache = R.ROICache(prior['test'], 'test')
pred = {}
for it in te_cache:
    cx, cy = prior['test'][f'{it["i"]:04d}']
    pts = [R.predict_one(models[fi], it, cx, cy, args.iters, 3.0) for fi in range(args.folds)]
    px = float(np.mean([p[0] for p in pts]))
    py = float(np.mean([p[1] for p in pts]))
    pred[it['i']] = [px, py]
    print(f'  test {it["i"]:04d} prior=({cx:7.1f},{cy:7.1f}) -> ({px:7.1f},{py:7.1f})  '
          f'分歧±({np.std([p[0] for p in pts]):.1f},{np.std([p[1] for p in pts]):.1f})', flush=True)

pj = INB / args.emit
pj.write_text(json.dumps({str(k): v for k, v in pred.items()}), encoding='utf-8')
print(f'[written] {pj}')
(INB / 't3_run_chain_oof.json').write_text(json.dumps(dict(
    per=per, mean=float(e.mean()), median=float(np.median(e)),
    prior_mean=float(e0.mean())), ensure_ascii=False), encoding='utf-8')
print('[written]', INB / 't3_run_chain_oof.json')
