import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
D = Path('_inbox/task4/segmentation'); E = 512
lab = sorted((D/'train/label').glob('*.jpg'), key=lambda p: int(p.stem))
gt = np.stack([np.asarray(Image.open(p).convert('L').resize((E, E), Image.BILINEAR)) > 0 for p in lab])
G = gt.reshape(len(gt), -1); gsum = G.sum(1).astype(np.float64)

def dice_vec(P, sel, t):
    """P: (n,E,E) float; returns mean dice over sel at threshold t (vectorised)."""
    B = (P[sel].reshape(len(sel), -1) > t)
    inter = (B & G[sel]).sum(1).astype(np.float64)
    s = B.sum(1) + gsum[sel]
    return float(np.mean(np.where(s == 0, 1.0, 2.0 * inter / np.maximum(s, 1))))

eg = np.load('t4_oof_eg.npy').astype(np.float32)
cv = np.load('t4_oof_convnext.npy').astype(np.float32)
n = len(gt)
print('shapes', eg.shape, cv.shape, 'corr=%.4f' % np.corrcoef(eg.ravel()[::997], cv.ravel()[::997])[0, 1])
perm = np.random.default_rng(42).permutation(n)
fold_of = np.empty(n, int)
for j, i in enumerate(perm): fold_of[i] = j % 5
WS = np.linspace(0, 1, 21); TAUS = np.linspace(0.10, 0.90, 33)
folds = [[i for i in range(n) if fold_of[i] == f] for f in range(5)]

print('--- 单模基线 ---')
for nm, P in (('eg(512网格)', eg), ('cv(原尺寸)', cv)):
    b = max(((dice_vec(P, range(n), t), t) for t in TAUS))
    print('  %s in-sample=%.4f (tau=%.2f)' % (nm, b[0], b[1]))

rows = []
for w in WS:
    P = (1 - w) * eg + w * cv
    d, t = max(((dice_vec(P, range(n), t), t) for t in TAUS))
    hold = []
    for f in range(5):
        tr = [i for i in range(n) if fold_of[i] != f]
        bw, bt, bd = 0.0, 0.5, -1
        for ww in WS:
            PP = (1 - ww) * eg + ww * cv
            for tt in TAUS:
                v = dice_vec(PP, tr, tt)
                if v > bd: bd, bw, bt = v, ww, tt
        PP = (1 - bw) * eg + bw * cv
        hold.append(dice_vec(PP, folds[f], bt))
    rows.append((float(w), d, float(t), float(np.mean(hold))))
    print('w=%.2f in=%.4f(tau=%.2f) held-out=%.4f' % (w, d, t, np.mean(hold)))
bi = max(rows, key=lambda r: r[1]); bh = max(rows, key=lambda r: r[3])
print('>>> in-sample 最优 w=%.2f -> %.4f' % (bi[0], bi[1]))
print('>>> held-out 最优 w=%.2f -> %.4f  (eg 单模 held-out 见下)' % (bh[0], bh[3]))
json.dump({'rows': rows}, open('_inbox/t4_blend2.json', 'w'))
