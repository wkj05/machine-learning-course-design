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
P = np.load('t4_oof_eg.npy'); n = len(lab)
Pm = np.stack([P[pos[i]] for i in range(n)]); pm = Pm > 0.60
d = np.array([2*(pm[i] & gt[i]).sum()/max(pm[i].sum()+gt[i].sum(), 1) for i in range(n)])
G = np.stack([np.asarray(Image.open(p).convert('RGB').resize((E, E), Image.BILINEAR)) for p in img]).astype(np.float32)
fov = (G.max(3) > 12)
print(' 图   Dice   绿>250占比  绿>230  p>0.6占比  GT占比')
for i in np.argsort(d):
    g = G[i, :, :, 1]
    clip = float(((g > 250) & fov[i]).sum() / max(fov[i].sum(), 1))
    hi = float(((g > 230) & fov[i]).sum() / max(fov[i].sum(), 1))
    print('%3d  %.3f   %.4f      %.4f     %.4f    %.4f' % (i+1, d[i], clip, hi, (pm[i] & fov[i]).mean()/fov[i].mean(), (gt[i] & fov[i]).mean()/fov[i].mean()))
r = np.corrcoef(d, [float((G[i,:,:,1] > 230)[fov[i]].mean()) for i in range(n)])[0,1]
print('corr(Dice, 绿>230 占比) = %.3f' % r)
