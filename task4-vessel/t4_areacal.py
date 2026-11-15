"""逐图面积校准：模型的预测密度被向均值压缩了吗？用 1 参数展开能否救回尾部？"""
import numpy as np
from pathlib import Path
from PIL import Image
D = Path('_inbox/task4/segmentation'); E = 512
lab = sorted((D/'train/label').glob('*.jpg'), key=lambda p: int(p.stem))
img = sorted((D/'train/image').glob('*.jpg'), key=lambda p: int(p.stem))
gt = np.stack([np.asarray(Image.open(p).convert('L').resize((E, E), Image.BILINEAR)) > 0 for p in lab])
lex = sorted(range(len(lab)), key=lambda i: lab[i].name)
pos = np.empty(len(lab), int)
for k, i in enumerate(lex): pos[i] = k
P = np.stack([np.load('t4_oof_eg.npy')[pos[i]] for i in range(len(lab))])
G = np.stack([np.asarray(Image.open(p).convert('RGB').resize((E, E), Image.BILINEAR)) for p in img]).astype(np.float32)
fov = G.max(3) > 12
n = len(lab)
gf = np.array([gt[i][fov[i]].mean() for i in range(n)])            # GT 血管占比
pf = np.array([P[i][fov[i]].mean() for i in range(n)])             # 预测平均概率
print('GT 占比 %.4f±%.4f   预测平均概率 %.4f±%.4f   corr=%.3f' % (gf.mean(), gf.std(), pf.mean(), pf.std(), np.corrcoef(gf, pf)[0, 1]))
k, b = np.polyfit(pf, gf, 1); print('最小二乘 gf ~ %.3f*pf + %.4f  (斜率>1 = 存在向均值压缩)' % (k, b))

def dice_masks(ms): return np.array([2*(ms[i] & gt[i]).sum()/max(ms[i].sum()+gt[i].sum(), 1) for i in range(len(ms))])

# 基线：全局阈值
best = max(((dice_masks(P > t).mean(), t) for t in np.linspace(0.30, 0.85, 23)))
print('全局阈值基线: %.4f (tau=%.2f)' % best)

def area_masks(target):
    ms = []
    for i in range(n):
        pf_i = P[i][fov[i]]
        t = np.quantile(pf_i, 1.0 - np.clip(target[i], 0.02, 0.60))
        ms.append(P[i] > t)
    return np.array(ms)

print('\n逐图面积校准（target = a*pf + (1-a)*gfmean 形式，a 为展开强度）:')
for a in (0.0, 1.0, 1.2, 1.5, 2.0):
    tgt = gf.mean() + a * (pf - pf.mean())
    print('  a=%.1f  -> Dice=%.4f' % (a, dice_masks(area_masks(tgt)).mean()))
tgt = k * pf + b
print('  用最小二乘拟合 -> Dice=%.4f' % dice_masks(area_masks(tgt)).mean())

# 留一折诚实检验：在 4 折上选 a，在 1 折上评
perm = np.random.default_rng(42).permutation(n); fo = np.empty(n, int)
for j, i in enumerate(perm): fo[i] = j % 5
hold = []
for f in range(5):
    tr = [i for i in range(n) if fo[i] != f]; va = [i for i in range(n) if fo[i] == f]
    ba, bd = 0.0, -1
    for a in np.linspace(0, 2.5, 11):
        tgt = gf[tr].mean() + a * (pf - pf[tr].mean())
        v = dice_masks(area_masks(tgt))[va].mean()
        if v > bd: bd, ba = v, a
    hold.append((ba, bd))
print('\n留一折: 每折选到的 a 与验证折 Dice:', [(round(x, 2), round(y, 4)) for x, y in hold])
print('留一折均值 = %.4f' % np.mean([y for _, y in hold]))
