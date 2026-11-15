# 只动后处理，不重训：在现有 OOF 概率上扫 tau / 最小域 / 膨胀 / 闭运算，全部用留一折验证。
import sys, importlib.util, json
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('t4', '_inbox/t4_kaggle.py')
t4 = importlib.util.module_from_spec(spec); _a = sys.argv; sys.argv = ['x']
spec.loader.exec_module(t4); sys.argv = _a

root = Path('_inbox/task4/segmentation')
labs = sorted([f for f in (root/'train'/'label').iterdir() if f.suffix.lower()=='.jpg'], key=lambda f: f.stem)
truth = np.stack([t4.gt_at_eval(t4.Image.open(f).convert('L')) for f in labs])
P = np.load('t4_oof_convnext.npy'); n = len(truth)
perm = np.random.default_rng(42).permutation(n)
fold_of = {int(p): j % 5 for j, p in enumerate(perm)}
folds = [np.array([i for i in range(n) if fold_of[i]==f]) for f in range(5)]

def morph(pm, r, close):
    if close:
        pm = ndi.binary_closing(pm, structure=np.ones((3,3), bool))
    if r > 0:
        pm = ndi.binary_dilation(pm, structure=np.ones((2*r+1, 2*r+1), bool))
    return pm

def d1(i, tau, ms, r, close):
    pm = P[i] > tau
    if ms: pm = t4.drop_small(pm, ms)
    pm = morph(pm, r, close)
    return t4.dice_at_eval(pm, truth[i])

def dset(idx, *cfg):
    return float(np.mean([d1(i, *cfg) for i in idx]))

TAUS = [0.08,0.10,0.12,0.14,0.16,0.20,0.26,0.32]
MSS  = [0,30,80]
GRID = [(r,c) for r in (0,1,2) for c in (0,1)]

print('== 全集 in-sample 最优（供参考，不作证据）')
best=(-1,None)
for tau in TAUS:
  for ms in MSS:
    for r,c in GRID:
      d=dset(np.arange(n), tau, ms, r, c)
      if d>best[0]: best=(d,(tau,ms,r,c))
print(f'   Dice={best[0]:.4f}  cfg(tau,ms,膨胀r,闭)={best[1]}')
base=dset(np.arange(n),0.12,30,0,0)
print(f'   基线（tau=0.12 ms=30 不形态学）= {base:.4f}')

print('== 留一折：4 折挑整套后处理，1 折评')
sel, fix = [], []
for f in range(5):
    tr = np.array([i for i in range(n) if fold_of[i]!=f]); va = folds[f]
    b=(-1,None)
    for tau in TAUS:
      for ms in MSS:
        for r,c in GRID:
          d=dset(tr, tau, ms, r, c)
          if d>b[0]: b=(d,(tau,ms,r,c))
    tau,ms,r,c = b[1]
    sel.append(dset(va, tau, ms, r, c)); fix.append(dset(va, 0.12, 30, 0, 0))
    print(f'  fold {f}: cfg={b[1]} -> held-out {sel[-1]:.4f}   (固定 tau.12/ms30/无形变: {fix[-1]:.4f})')
print(f'\n留一折平均  调好的后处理 = {np.mean(sel):.4f}   固定基线 = {np.mean(fix):.4f}   净收益 {np.mean(sel)-np.mean(fix):+.4f}')
print(f'in-sample 乐观量 = {best[0]-np.mean(sel):+.4f}')

# 单独看"膨胀"这一项的边际贡献（固定 tau=0.12 ms=30，只比 r）
print('\n== 膨胀的边际作用（固定 tau=0.12 ms=30，留一折平均）')
for r,c in GRID:
    v=[dset(folds[f], 0.12, 30, r, c) for f in range(5)]
    print(f'   r={r} 闭={c}: {np.mean(v):.4f}')
Path('t4_post_result.json').write_text(json.dumps({'best':best[1],'in_sample':best[0],
    'heldout_tuned':float(np.mean(sel)),'heldout_fixed':float(np.mean(fix))}, ensure_ascii=False, indent=1), encoding='utf-8')
