# -*- coding: utf-8 -*-
"""任务三 几何基线：视盘位置 + 结构先验 -> 黄斑坐标。留一法评估 + 产出测试集 pred.json。"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
OUT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
LONG = 768


def load_small(p):
    im = Image.open(p).convert('RGB')
    W, H = im.size
    s = LONG / max(W, H)
    sm = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
    return np.asarray(sm).astype(np.float32) / 255.0, (W, H), s


def retina_mask(rgb):
    m = rgb.max(2) > 0.12
    m = ndi.binary_closing(m, np.ones((7, 7)))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(m)


def detect_disc(rgb, retina, refine):
    L = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    L = ndi.gaussian_filter(L, 2.0)
    a_ret = int(retina.sum())
    if a_ret < 1000:
        return None
    for pct in (98.5, 97.0, 95.0, 92.0):
        thr = np.percentile(L[retina], pct)
        m = ndi.binary_opening((L >= thr) & retina, np.ones((5, 5)))
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
            cands.append(dict(cx=sl[1].start + xs.mean(), cy=sl[0].start + ys.mean(),
                              R=float(np.sqrt(a / np.pi)), area=a,
                              meanL=float(L[sl][comp].mean())))
        if not cands:
            continue
        cands.sort(key=lambda c: (c['meanL'], c['area']), reverse=True)
        b = cands[0]
        if refine:
            # 以初检中心为圆心，在 1.8R 内用「峰值一半」阈值重新求质心，修正锁到亮杯的偏移
            R = max(b['R'], 25.0)
            yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
            disc = ((xx - b['cx']) ** 2 + (yy - b['cy']) ** 2) <= (1.8 * R) ** 2
            roi = L[disc & retina]
            if roi.size > 50:
                half = 0.5 * (roi.max() + np.median(L[retina]))
                m2 = disc & retina & (L >= half)
                m2 = ndi.binary_opening(m2, np.ones((5, 5)))
                lab2, n2 = ndi.label(m2)
                if n2 > 0:
                    sz = ndi.sum(m2, lab2, range(1, n2 + 1))
                    k2 = 1 + int(np.argmax(sz))
                    ys2, xs2 = np.nonzero(lab2 == k2)
                    b = dict(b, cx=float(xs2.mean()), cy=float(ys2.mean()),
                             R=float(np.sqrt(sz[k2 - 1] / np.pi)))
        return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refine', type=int, default=1)
    ap.add_argument('--emit', action='store_true')
    a = ap.parse_args()

    gt = {}
    with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8-sig') as f:
        next(f)
        for ln in f:
            if not ln.strip():
                continue
            d, x, y = ln.split(',')[:3]
            gt[int(d)] = (float(x), float(y))

    tr = []
    for i in sorted(gt):
        rgb, (W, H), s = load_small(ROOT / 'train' / f'{i:04d}.jpg')
        ret = retina_mask(rgb)
        d = detect_disc(rgb, ret, a.refine)
        if d is None:
            tr.append(dict(i=i, ok=False, W=W, H=H))
            continue
        tr.append(dict(i=i, ok=True, W=W, H=H, cx=d['cx'] / s, cy=d['cy'] / s, R=d['R'] / s))
    nok = sum(1 for r in tr if r['ok'])
    print(f'[disc] 检出 {nok}/{len(tr)}   refine={a.refine}')
    print(f'[disc] R 原图 px: mean={np.mean([r["R"] for r in tr if r["ok"]]):.1f} '
          f'min={min(r["R"] for r in tr if r["ok"]):.1f} max={max(r["R"] for r in tr if r["ok"]):.1f}')

    def size_key(r):
        return 'S' if r['W'] < 2500 else 'L'

    def predict(r, D, Ymed):
        # 视盘在左半 -> 黄斑在右(+1)；在右半 -> 黄斑在左(-1)
        side = -1.0 if r['cx'] > r['W'] / 2.0 else 1.0
        return r['cx'] + side * D[size_key(r)], Ymed[size_key(r)]

    def fit(rows):
        D, Y = {}, {}
        for k in ('L', 'S'):
            sel = [r for r in rows if r['ok'] and size_key(r) == k]
            if len(sel) < 3:
                sel = [r for r in rows if r['ok']]
            dxs = [abs(gt[r['i']][0] - r['cx']) for r in sel]
            ys = [gt[r['i']][1] for r in sel]
            D[k] = float(np.median(dxs))
            Y[k] = float(np.median(ys))
        gD = float(np.median([abs(gt[r['i']][0] - r['cx']) for r in rows if r['ok']]))
        gY = float(np.median([gt[r['i']][1] for r in rows if r['ok']]))
        for k in ('L', 'S'):
            D.setdefault(k, gD)
            Y.setdefault(k, gY)
        return D, Y

    # 留一法
    errs, loo = [], {}
    for j, r in enumerate(tr):
        others = [x for k, x in enumerate(tr) if k != j]
        D, Y = fit(others)
        if r['ok']:
            px, py = predict(r, D, Y)
        else:
            okr = [x for x in others if x['ok']]
            px = float(np.median([x['cx'] for x in okr]))
            py = Y[size_key(r)]
        X, Yt = gt[r['i']]
        loo[f'{r["i"]:04d}'] = [px, py]
        errs.append(float(np.hypot(px - X, py - Yt)))
    e = np.array(errs)
    print(f'[LOO] mean={e.mean():.2f} median={np.median(e):.2f} '
          f'p90={np.percentile(e,90):.2f} max={e.max():.2f}  n={len(e)}')

    if a.emit:
        D, Y = fit(tr)
        print(f'[fit] D={ {k: round(v,1) for k,v in D.items()} }  Y={ {k: round(v,1) for k,v in Y.items()} }')
        pred = {}
        for i in range(81, 101):
            rgb, (W, H), s = load_small(ROOT / 'test' / f'{i:04d}.jpg')
            ret = retina_mask(rgb)
            d = detect_disc(rgb, ret, a.refine)
            r = dict(i=i, ok=d is not None, W=W, H=H,
                     cx=(d['cx'] / s if d else W / 2.0), cy=(d['cy'] / s if d else H / 2.0))
            px, py = predict(r, D, Y)
            px = float(np.clip(px, 0, W - 1)); py = float(np.clip(py, 0, H - 1))
            pred[i] = [px, py]
            print(f'  {i:04d} {W}x{H} disc=({r["cx"]:.0f},{r["cy"]:.0f}) ok={r["ok"]} -> ({px:.1f},{py:.1f})')
        p = OUT / 't3_geo_pred.json'
        p.write_text(json.dumps({str(k): v for k, v in pred.items()}), encoding='utf-8')
        print(f'[written] {p}')
        # 供两级细化使用：train 用留一法（不泄漏），test 用全部 80 张拟合
        gj = OUT / 't3_geom.json'
        gj.write_text(json.dumps({'train': loo,
                                  'test': {f'{k:04d}': v for k, v in pred.items()}},
                                 ensure_ascii=False), encoding='utf-8')
        print(f'[written] {gj}  (train LOO {len(loo)} 条, test {len(pred)} 条)')


if __name__ == '__main__':
    main()
