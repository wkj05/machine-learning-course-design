# 留一折式检验：在 4/5 的图上挑 (tau, min_size)，在剩下的 1/5 上评。
# 这样衡量的是"挑阈值这个动作本身"的真实价值，而不是 in-sample 的最小值。
import sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location('t4', '_inbox/t4_kaggle.py')
t4 = importlib.util.module_from_spec(spec)
sys.argv = ['x']
spec.loader.exec_module(t4)          # 只用到常量与 drop_small/dice/gt 工具

root = Path('_inbox/task4/segmentation')
ids = sorted([f for f in (root/'train'/'label').iterdir() if f.suffix.lower() == '.jpg'],
             key=lambda f: f.stem)
truth = np.stack([t4.gt_at_eval(t4.Image.open(f).convert('L')) for f in ids])
P = np.load('t4_out/oof_prob512.npy')
n = len(ids)

seed, nf = 42, 5
perm = np.random.default_rng(seed).permutation(n)
fold_of = {int(p): j % nf for j, p in enumerate(perm)}
folds = [np.array([i for i in range(n) if fold_of[i] == f]) for f in range(nf)]

TAUS = np.linspace(0.10, 0.40, 31)
MSS = (0, 30, 80)


def dice_set(idx, tau, ms):
    ds = []
    for i in idx:
        pm = P[i] > tau
        if ms:
            pm = t4.drop_small(pm, ms)
        ds.append(t4.dice_at_eval(pm, truth[i]))
    return float(np.mean(ds))


def fit(idx):
    best = (-1, 0.2, 0)
    for tau in TAUS:
        for ms in MSS:
            d = dice_set(idx, tau, ms)
            if d > best[0]:
                best = (d, float(tau), ms)
    return best


# (a) in-sample：全部 20 张挑一个
ins = fit(np.arange(n))
print(f'in-sample 全集最优: Dice={ins[0]:.4f}  tau={ins[1]:.2f}  ms={ins[2]}')


# (b) 留一折：4 折挑，1 折评
gains_sel, gains_fix = [], []
for f in range(nf):
    tr = np.array([i for i in range(n) if fold_of[i] != f])
    va = folds[f]
    _, tau_s, ms_s = fit(tr)
    d_sel = dice_set(va, tau_s, ms_s)
    d_fix = dice_set(va, 0.30, 0)        # 上一版固定用的 0.30 / 无后处理
    gains_sel.append(d_sel)
    gains_fix.append(d_fix)
    print(f'  fold {f}: 选的 tau={tau_s:.2f} ms={ms_s}  ->  held-out Dice={d_sel:.4f}  '
          f'(对照 固定 tau=0.30/ms=0: {d_fix:.4f})')
print()
print(f'留一折平均  挑阈值 = {np.mean(gains_sel):.4f}')
print(f'留一折平均  固定 0.30/ms0 = {np.mean(gains_fix):.4f}')
print(f'挑阈值的真实净收益 = {np.mean(gains_sel) - np.mean(gains_fix):+.4f}')
print(f'in-sample 的乐观量 = {ins[0] - np.mean(gains_sel):+.4f}')
