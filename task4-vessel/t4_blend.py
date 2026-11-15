# 两个架构概率平均：先看两者错误是否相关，再用留一折验证"该不该混、混多少"。
import sys, importlib.util, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('t4', '_inbox/t4_kaggle.py')
t4 = importlib.util.module_from_spec(spec); _argv = sys.argv; sys.argv = ['x']
spec.loader.exec_module(t4); sys.argv = _argv

root = Path('_inbox/task4/segmentation')
labs = sorted([f for f in (root/'train'/'label').iterdir() if f.suffix.lower()=='.jpg'], key=lambda f: f.stem)
truth = np.stack([t4.gt_at_eval(t4.Image.open(f).convert('L')) for f in labs])
A = np.load('t4_oof_convnext.npy'); B = np.load('t4_out_unet/oof_prob512.npy')
n = len(truth)
perm = np.random.default_rng(42).permutation(n)
fold_of = {int(p): j % 5 for j, p in enumerate(perm)}
folds = [np.array([i for i in range(n) if fold_of[i]==f]) for f in range(5)]

def dice_i(Pm, i, tau, ms):
    pm = Pm[i] > tau
    if ms: pm = t4.drop_small(pm, ms)
    return t4.dice_at_eval(pm, truth[i])

# 1) 两者的逐图 Dice 相关 —— 若高度相关，混合不可能有大收益
dA = np.array([dice_i(A, i, 0.12, 30) for i in range(n)])
dB = np.array([dice_i(B, i, 0.14, 30) for i in range(n)])
print(f'convnext 单图 Dice: mean={dA.mean():.4f}  unet: mean={dB.mean():.4f}')
print(f'两者逐图 Dice 的皮尔逊相关 r = {np.corrcoef(dA, dB)[0,1]:.4f}   (1=完全相关，混合无望)')
print('逐图对比（i: convnext / unet）:')
for i in range(n):
    flag = '  << 两个都差' if min(dA[i],dB[i]) < 0.75 else ''
    print(f'  {i+1:2d}: {dA[i]:.4f} / {dB[i]:.4f}{flag}')

WS = [0.0,0.2,0.4,0.5,0.6,0.8,1.0]; TAUS = [0.10,0.12,0.14,0.16,0.20,0.26]; MSS = [0,30]
def P(w):
    if w <= 0: return A
    if w >= 1: return B
    return (1-w)*A + w*B
def dset(Pm, idx, tau, ms):
    return float(np.mean([dice_i(Pm, i, tau, ms) for i in idx]))
def fit(idx):
    best=(-1,None)
    for w in WS:
        Pm=P(w)
        for tau in TAUS:
            for ms in MSS:
                d=dset(Pm, idx, tau, ms)
                if d>best[0]: best=(d,(w,tau,ms))
    return best

print('\n== in-sample（全集挑 w,tau,ms）')
ins = fit(np.arange(n)); print(f'   Dice={ins[0]:.4f}  cfg(w_unet,tau,ms)={ins[1]}')
print('== 留一折（4 折挑，1 折评）')
sel, uni = [], []
for f in range(5):
    tr = np.array([i for i in range(n) if fold_of[i]!=f]); va = folds[f]
    _, (w_s, tau_s, ms_s) = fit(tr)
    sel.append(dset(P(w_s), va, tau_s, ms_s)); uni.append(dset(A, va, 0.12, 30))
    print(f'  fold {f}: w={w_s:.1f} tau={tau_s:.2f} ms={ms_s} -> 混合 {sel[-1]:.4f}   单 convnext {uni[-1]:.4f}')
gain = float(np.mean(sel)-np.mean(uni))
print(f'\n留一折平均  混合 {np.mean(sel):.4f}   单 convnext {np.mean(uni):.4f}   净收益 {gain:+.4f}')
print(f'in-sample 乐观量 {ins[0]-np.mean(sel):+.4f}')
out = {'dice_corr': float(np.corrcoef(dA,dB)[0,1]), 'in_sample': ins[0], 'cfg': ins[1],
       'heldout_blend': float(np.mean(sel)), 'heldout_single': float(np.mean(uni)), 'gain': gain}
Path('t4_blend_result.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

if gain > 0.002:
    TA = np.load('t4_test_convnext.npy'); TB = np.load('t4_out_unet/test_prob512.npy')
    w,tau,ms = ins[1]; Tp = (1-w)*TA + w*TB
    te = sorted([f for f in (root/'test'/'image').iterdir() if f.suffix.lower()=='.jpg'], key=lambda f: f.stem)
    pred = {}
    for k,f in enumerate(te):
        pm = Tp[k] > tau
        if ms: pm = t4.drop_small(pm, ms)
        pred[f.stem] = t4.rle_encode(pm.astype(np.uint8))
    t4.write_submission(root/'sample_submission.csv', pred, Path('t4_sub_blend.csv'))
    print(f'\n已写出 t4_sub_blend.csv  (w_unet={w} tau={tau} ms={ms})')
else:
    print('\n留一折净收益 <= 0.002 ⇒ 判为无实质提升，不出混合提交。')
