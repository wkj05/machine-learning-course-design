# -*- coding: utf-8 -*-
"""任务三 · 翻转 TTA + 5 折集成的干净评估与提交件生成。

OOF 协议：每张训练图只用「它所属折的模型」（无泄漏），水平翻转两视图取平均。
测试集：5 折模型 × 2 视图 = 10 个预测取平均。
"""
import sys, json, math, argparse
from pathlib import Path
import numpy as np
import torch

INB = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
sys.path.insert(0, str(INB))
import t3_refine as R                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--prior', default='t3_geom.json')
ap.add_argument('--models', default='t3_run_refine_full2')
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--multi', default='')     # 额外模型目录，逗号分隔，与主模型并列平均
ap.add_argument('--emit', default='t3_pred_tta.json')
args = ap.parse_args()
torch.set_num_threads(args.threads)


def predict_tta(model, item, cx, cy):
    """1 次迭代 + 水平翻转 TTA。翻转图的偏移 x 要取反。"""
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        patch, (acx, acy), _ = R.sample_crop(item, cx, cy, None, False)
        x1 = R.to_tensor(patch).unsqueeze(0).to(dev)
        x2 = R.to_tensor(np.ascontiguousarray(patch[:, ::-1])).unsqueeze(0).to(dev)
        o1 = R.soft_argmax(model(x1), 4)[0].cpu().numpy()
        o2 = R.soft_argmax(model(x2), 4)[0].cpu().numpy()
        dx = (float(o1[0]) + (-float(o2[0]))) / 2.0
        dy = (float(o1[1]) + float(o2[1])) / 2.0
    return acx + dx, acy + dy


gt = R.load_gt()
prior = json.loads((INB / args.prior).read_text(encoding='utf-8'))
ids = sorted(gt)
rng = np.random.default_rng(42)
perm = rng.permutation(len(ids))
folds = [perm[j::args.folds] for j in range(args.folds)]
fold_of = {}
for fi, f in enumerate(folds):
    for k in f:
        fold_of[int(ids[k])] = fi

dev = torch.device('cpu')
groups = [args.models] + [m for m in args.multi.split(',') if m]
models = []          # (fold_index, model)
for g in groups:
    for fi in range(args.folds):
        p = INB / g / f'fold{fi}.pt'
        if not p.exists():
            print(f'[skip] 缺 {p}')
            continue
        m = R.Refiner().to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
        m.eval()
        models.append((fi, m))
print(f'[models] 载入 {len(models)} 个（{len(groups)} 组 × {args.folds} 折）')


def ens(items, cx, cy, want_fold=None):
    sel = [m for fi, m in models if want_fold is None or fi == want_fold]
    pts = [predict_tta(m, items, cx, cy) for m in sel]
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])), len(pts)


tr_cache = R.ROICache(prior['train'], 'train')
items = {it['i']: it for it in tr_cache}
errs = []
for i in ids:
    cx, cy = prior['train'][f'{i:04d}']
    px, py, n = ens(items[i], cx, cy, want_fold=fold_of[i])
    X, Y = gt[i]
    errs.append(math.hypot(px - X, py - Y))
e = np.array(errs)
print(f'\nOOF(TTA) n={len(e)} mean={e.mean():.2f} median={np.median(e):.2f} '
      f'p90={np.percentile(e,90):.2f} max={e.max():.2f} RMSE={math.sqrt((e**2).mean()/2):.2f}')
print(f'对照：单模型无 TTA RMSE 28.72 / 榜单 27.53739')

te_cache = R.ROICache(prior['test'], 'test')
pred = {}
for it in te_cache:
    cx, cy = prior['test'][f'{it["i"]:04d}']
    px, py, n = ens(it, cx, cy, want_fold=None)
    pred[it['i']] = [px, py]
    print(f'  test {it["i"]:04d} prior=({cx:7.1f},{cy:7.1f}) -> ({px:7.1f},{py:7.1f})')
(INB / args.emit).write_text(json.dumps({str(k): v for k, v in pred.items()}), encoding='utf-8')
print('[written]', INB / args.emit)
