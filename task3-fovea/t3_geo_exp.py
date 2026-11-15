# -*- coding: utf-8 -*-
"""任务三 几何先验实验：换分辨率/模糊/打分方式/侧判定，并在留一法下对比。
目标：把先验的 80 张 LOO 误差 p90 从 110px 压到 60px 以下。
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
OUT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')


_CACHE = {}


def load_gray(p, long_side):
    key = (str(p), long_side)
    if key in _CACHE:
        return _CACHE[key]
    im = Image.open(p).convert('RGB')
    W, H = im.size
    s = long_side / max(W, H)
    L = np.asarray(im.convert('L'), np.float32) / 255.0
    if s != 1.0:
        L = np.asarray(Image.fromarray((L * 255).astype(np.uint8)).resize(
            (max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR), np.float32) / 255.0
    _CACHE[key] = (L, (W, H), s)
    return _CACHE[key]


def retina_mask(L):
    m = L > 0.05
    m = ndi.binary_closing(m, np.ones((7, 7)))
    lab, n = ndi.label(m)
    if n > 1:
        sz = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sz)))
    return ndi.binary_fill_holes(m)


def detect(L, retina, sigma, score_mode, pcts, warea):
    """返回 dict(cx,cy,R,score) 或 None。坐标是 LONG 尺度。"""
    S = ndi.gaussian_filter(L, sigma)
    a_ret = int(retina.sum())
    for pct in pcts:
        thr = np.percentile(S[retina], pct)
        m = ndi.binary_opening((S >= thr) & retina, np.ones((5, 5)))
        lab, n = ndi.label(m)
        if n == 0:
            continue
        objs = ndi.find_objects(lab)
        cands = []
        for k in range(1, n + 1):
            sl = objs[k - 1]
            if sl is None:
                continue
            comp = lab[sl] == k
            a = int(comp.sum())
            if not (0.0012 * a_ret <= a <= 0.06 * a_ret):
                continue
            h, w = comp.shape
            if a / (np.pi * (max(h, w) / 2.0) ** 2) < 0.35:
                continue
            ys, xs = np.nonzero(comp)
            # 亮度加权质心，去掉阈值带来的边界偏置
            wt = np.clip(S[sl][comp] - thr, 1e-6, None)
            cx = sl[1].start + float((xs * wt).sum() / wt.sum())
            cy = sl[0].start + float((ys * wt).sum() / wt.sum())
            meanL = float(S[sl][comp].mean())
            if score_mode == 'meanL':
                sc = meanL
            elif score_mode == 'area':
                sc = a
            else:
                sc = meanL + warea * (a / a_ret) * 100.0
            cands.append(dict(cx=cx, cy=cy, R=float(np.sqrt(a / np.pi)), area=a,
                              meanL=meanL, score=sc))
        if not cands:
            continue
        cands.sort(key=lambda c: c['score'], reverse=True)
        return cands[0]
    return None


def collect(long_side, sigma, score_mode, pcts, warea):
    gt = {}
    for line in (ROOT / 'fovea_localization_train_GT.csv').read_text().splitlines()[1:]:
        a, b, c = line.split(',')
        gt[a.strip()] = (float(b), float(c))
    rows = []
    for sid in sorted(gt):
        L, (W, H), s = load_gray(ROOT / 'train' / f'{int(sid):04d}.jpg', long_side)
        ret = retina_mask(L)
        det = detect(L, ret, sigma, score_mode, pcts, warea)
        if det is None:
            rows.append(dict(sid=sid, W=W, H=H, s=s, ok=False))
            continue
        rows.append(dict(sid=sid, W=W, H=H, s=s, ok=True,
                         cx=det['cx'] / s, cy=det['cy'] / s, R=det['R'] / s,
                         gy=gt[sid][1]))
    return rows


def eval_prior(rows, y_mode, alpha=0.5):
    """留一法：按尺寸组拟合 D 和 Y 偏移，再预测。"""
    ok = [r for r in rows if r['ok']]
    for r in ok:
        r['size'] = 'L' if r['W'] >= 2500 else 'S'
    gt = {}
    for line in (ROOT / 'fovea_localization_train_GT.csv').read_text().splitlines()[1:]:
        a, b, c = line.split(',')
        gt[a.strip()] = (float(b), float(c))
    n = len(ok)
    err = np.zeros(n)
    for i, r in enumerate(ok):
        gx, gy = gt[r['sid']]
        side = -1.0 if r['cx'] > r['W'] / 2.0 else 1.0
        rest = [q for j, q in enumerate(ok) if j != i]
        # 拟合参数用的都是「绝对偏移」，与目标无关
        dxs = [abs(gt[q['sid']][0] - q['cx']) for q in rest if q['size'] == r['size']]
        dys = [gt[q['sid']][1] - q['cy'] for q in rest if q['size'] == r['size']]
        constY = [gt[q['sid']][1] for q in rest if q['size'] == r['size']]
        D = float(np.median(dxs))
        X = r['cx'] + side * D
        if y_mode == 'const':
            Y = float(np.median(constY))
        elif y_mode == 'cy':
            Y = r['cy'] + float(np.median(dys))
        else:
            Y = alpha * float(np.median(constY)) + (1 - alpha) * (r['cy'] + float(np.median(dys)))
        err[i] = np.hypot(X - gx, Y - gy)
    return err


def rpt(name, e):
    print(f'{name:38s} mean={e.mean():6.2f} med={np.median(e):6.2f} '
          f'p90={np.percentile(e,90):6.2f} max={e.max():7.2f}  RMSE={np.sqrt((e**2).mean()/2):6.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--long', type=int, default=0)
    ap.add_argument('--sigma', type=float, default=0.0)
    args = ap.parse_args()
    longs = [args.long] if args.long else [768, 1536]
    sigmas = [args.sigma] if args.sigma else [1.5, 3.0, 6.0]
    pcts = (98.5, 97.0, 95.0, 92.0, 88.0)
    best = None
    for long_side in longs:
        for sigma in sigmas:
            rows = collect(long_side, sigma, 'meanL', pcts, 0.0)
            nok = sum(1 for r in rows if r['ok'])
            for ymode, alpha in (('const', 0.5), ('cy', 0.5),
                                 ('mix', 0.2), ('mix', 0.35), ('mix', 0.5), ('mix', 0.65), ('mix', 0.8)):
                e = eval_prior(rows, ymode, alpha)
                tag = f'long={long_side} sig={sigma} y={ymode} a={alpha}'
                rpt(tag, e)
                if best is None or e.mean() < best[0]:
                    best = (float(e.mean()), tag, nok)
            print(f'    (检出 {nok}/{len(rows)})')
    print()
    print('best:', best)
