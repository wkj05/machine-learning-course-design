import os, sys, time
os.environ['T4_BB'] = 'convnext_tiny.fb_in22k_ft_in1k'
os.environ['T4_OUT'] = 't4_out_w'
sys.path.insert(0, str(__import__('pathlib').Path('_inbox').resolve()))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
import t4_kaggle as T

D = Path('_inbox/task4/segmentation'); E = T.EVAL_SIZE
labp = sorted((D/'train/label').glob('*.jpg'), key=lambda p: int(p.stem))
imgp = sorted((D/'train/image').glob('*.jpg'), key=lambda p: int(p.stem))
n = len(imgp)
imgs = [T.prep_one(np.asarray(Image.open(p).convert('RGB').resize((E, E), Image.BILINEAR))) for p in imgp]
gt = [T.gt_at_eval(Image.open(p).convert('L')) for p in labp]
perm = np.random.default_rng(42).permutation(n)
fo = np.empty(n, int)
for j, i in enumerate(perm): fo[i] = j % 5
dev = torch.device('cuda')

def d4_tta(model, img):
    x0 = torch.from_numpy(img).to(dev)[None]
    acc = None
    for r in range(4):
        xr = torch.rot90(x0, r, (-2, -1))
        for fh in (False, True):
            xi = torch.flip(xr, [-1]) if fh else xr
            o = torch.sigmoid(model(xi))
            o = torch.flip(o, [-1]) if fh else o
            o = torch.rot90(o, -r, (-2, -1))
            acc = o if acc is None else acc + o
    return (acc / 8.0)[0, 0].detach().cpu().numpy()

def tiled_prob(model, img, tile=256, stride=128):
    C, H, W = img.shape
    x = torch.from_numpy(img).to(dev)[None]
    acc = torch.zeros(1, 1, H, W, device=dev); ws = torch.zeros(1, 1, H, W, device=dev)
    w1 = torch.hann_window(tile, periodic=False, device=dev)
    win = (w1[:, None] * w1[None, :])[None, None]
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] != H - tile: ys.append(H - tile)
    if xs[-1] != W - tile: xs.append(W - tile)
    for y in ys:
        for x0 in xs:
            patch = x[:, :, y:y+tile, x0:x0+tile]
            o = T._forward_tta(model, patch, True)
            acc[:, :, y:y+tile, x0:x0+tile] += o * win
            ws[:, :, y:y+tile, x0:x0+tile] += win
    return (acc / ws.clamp_min(1e-6))[0, 0].detach().cpu().numpy()

TAUS = np.linspace(0.30, 0.85, 23)
def sc(P, sel, t):
    d = [2*( (P[i] > t) & gt[i]).sum()/max((P[i] > t).sum()+gt[i].sum(), 1) for i in sel]
    return float(np.mean(d))

variants = {}
t0 = time.time()
for f in range(5):
    ck = torch.load(f't4_out_w/m_f{f}s0.pt', map_location='cpu')
    m = T.build_model(pretrained=False); m.load_state_dict(ck); m.to(dev).eval()
    va = [i for i in range(n) if fo[i] == f]
    for nm, fn in (('full4', lambda im: T.predict_full(m, im, dev, True)),
                   ('full1', lambda im: T.predict_full(m, im, dev, False)),
                   ('fullD4', lambda im: d4_tta(m, im)),
                   ('tile256_4', lambda im: tiled_prob(m, im, 256, 128)),
                   ('tile128_4', lambda im: tiled_prob(m, im, 128, 64))):
        for i in va:
            with torch.no_grad():
                variants.setdefault(nm, {})[i] = fn(imgs[i]).astype(np.float32)
    print(f'fold {f} done  {time.time()-t0:.0f}s', flush=True)

print('\n== OOF（每个变体在 5 折上拼起来）==')
res = {}
for nm, dd in variants.items():
    P = [dd[i] for i in range(n)]
    bi = max(((sc(P, range(n), t), t) for t in TAUS))
    ho = []
    for f in range(5):
        tr = [i for i in range(n) if fo[i] != f]; va = [i for i in range(n) if fo[i] == f]
        bt = max(((sc(P, tr, t), t) for t in TAUS))[1]
        ho.append(sc(P, va, bt))
    res[nm] = (bi[0], bi[1], float(np.mean(ho)))
    print('%9s  in-sample=%.4f (tau=%.2f)  held-out=%.4f' % (nm, bi[0], bi[1], np.mean(ho)))
np.save('_inbox/t4_infer_variants.npy', np.array([0]))
import pickle
pickle.dump({k: [v[i] for i in range(n)] for k, v in variants.items()}, open('_inbox/t4_infer_variants.pkl', 'wb'))
print('saved')
