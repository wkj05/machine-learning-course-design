import sys, importlib.util
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('t4', '_inbox/t4_kaggle.py')
t4 = importlib.util.module_from_spec(spec); sys.argv=['x']; spec.loader.exec_module(t4)

npy = sys.argv[1] if len(sys.argv) > 1 else 't4_out/oof_prob512.npy'
root = Path('_inbox/task4/segmentation')
ids = sorted([f for f in (root/'train'/'label').iterdir() if f.suffix.lower()=='.jpg'], key=lambda f: f.stem)
truth = np.stack([t4.gt_at_eval(t4.Image.open(f).convert('L')) for f in ids])
P = np.load(npy); n = len(ids)
perm = np.random.default_rng(42).permutation(n)
fold_of = {int(p): j % 5 for j, p in enumerate(perm)}
folds = [np.array([i for i in range(n) if fold_of[i]==f]) for f in range(5)]
TAUS = np.linspace(0.08, 0.40, 33); MSS = (0, 30, 80)

def dice_set(idx, tau, ms):
    ds=[]
    for i in idx:
        pm = P[i] > tau
        if ms: pm = t4.drop_small(pm, ms)
        ds.append(t4.dice_at_eval(pm, truth[i]))
    return float(np.mean(ds))

def fit(idx):
    best=(-1,0.2,0)
    for tau in TAUS:
        for ms in MSS:
            d=dice_set(idx,tau,ms)
            if d>best[0]: best=(d,float(tau),ms)
    return best

print(f'== {npy}')
ins = fit(np.arange(n))
print(f'in-sample 全集最优: Dice={ins[0]:.4f}  tau={ins[1]:.2f}  ms={ins[2]}')
sel, fix = [], []
for f in range(5):
    tr = np.array([i for i in range(n) if fold_of[i]!=f]); va = folds[f]
    _, tau_s, ms_s = fit(tr)
    sel.append(dice_set(va, tau_s, ms_s)); fix.append(dice_set(va, 0.30, 0))
    print(f'  fold {f}: tau={tau_s:.2f} ms={ms_s} -> held-out {sel[-1]:.4f}  (固定 0.30: {fix[-1]:.4f})')
print(f'留一折挑阈值 = {np.mean(sel):.4f}   固定 0.30 = {np.mean(fix):.4f}   净收益 {np.mean(sel)-np.mean(fix):+.4f}')
print(f'in-sample 乐观量 = {ins[0]-np.mean(sel):+.4f}')
