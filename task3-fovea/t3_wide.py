# -*- coding: utf-8 -*-
"""任务三 两级方案第一级（宽视野）：以几何先验为中心取 512px 块、缩到 256 输入，
把 ±256px 的范围一次看全，专门解决「先验偏 60~155px 时黄斑落在 256 裁块之外」的尾巴。

训练样本：中心 = 几何预测 + 抖动(±160 px)，标签 = 真值 - 实际裁块中心。
推理：从几何预测出发裁块 -> 预测偏移（1 次即可，精度只要 40px）。
产出：t3_geom2.json（train=留一折预测，test=5 折集成），供第二级 1:1 细化当前验。
"""
import sys, json, time, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
INB = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')

CROP = 512          # 第一级：裁块边长（原图像素）—— 覆盖 ±256，能吞下最大 155px 的先验偏差
HALF = CROP // 2
NET = 256           # 送进网络的输入边长（把 CROP 缩到 NET）
ROI_HALF = 448      # 每张图预先缓存的 ROI 半边
STRIDE = 4          # 热图相对网络输入的降采样倍数
GRID = NET // STRIDE
STEP = CROP / GRID  # 每个热图 cell 覆盖的原图像素


def load_gt():
    gt = {}
    with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8-sig') as f:
        next(f)
        for ln in f:
            if ln.strip():
                d, x, y = ln.split(',')[:3]
                gt[int(d)] = (float(x), float(y))
    return gt


class ROICache:
    """每张图缓存一个以几何预测为中心、边长为 2*ROI_HALF 的原分辨率 ROI（uint8）。"""

    def __init__(self, geom, split):
        self.items = []
        for k in sorted(geom):
            i = int(k)
            cx, cy = geom[k]
            im = Image.open(ROOT / split / f'{i:04d}.jpg').convert('RGB')
            W, H = im.size
            x0 = int(round(cx)) - ROI_HALF
            y0 = int(round(cy)) - ROI_HALF
            box = (x0, y0, x0 + 2 * ROI_HALF, y0 + 2 * ROI_HALF)
            pad = (max(0, -box[0]), max(0, -box[1]),
                   max(0, box[2] - W), max(0, box[3] - H))
            crop = im.crop((max(0, box[0]), max(0, box[1]),
                            min(W, box[2]), min(H, box[3])))
            if any(pad):
                crop = crop.transform((2 * ROI_HALF, 2 * ROI_HALF), Image.EXTENT,
                                      (-pad[0], -pad[1], 2 * ROI_HALF - pad[0], 2 * ROI_HALF - pad[1]))
            self.items.append(dict(i=i, W=W, H=H, x0=x0, y0=y0,
                                   arr=np.asarray(crop, dtype=np.uint8)))
            im.close()

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


FACT = CROP // NET   # 缩放整数倍（2 表示 2x2 块均值降采样）
assert FACT * NET == CROP, (CROP, NET)


def sample_crop(item, ccx, ccy, rng, augment):
    """从 ROI 里取一个以 (ccx,ccy)（全图坐标）为中心的 CROP 块，缩到 NET，返回 (块, 实际中心, 是否翻转)。"""
    lx, ly = ccx - item['x0'], ccy - item['y0']
    left = int(round(lx - HALF))
    top = int(round(ly - HALF))
    left = max(0, min(left, 2 * ROI_HALF - CROP))
    top = max(0, min(top, 2 * ROI_HALF - CROP))
    patch = item['arr'][top:top + CROP, left:left + CROP]
    assert patch.shape[:2] == (CROP, CROP), patch.shape
    acx, acy = item['x0'] + left + HALF, item['y0'] + top + HALF
    if FACT > 1:
        # 2x2（或更大）块均值降采样，比 PIL resize 快两个数量级
        acc = patch[0::FACT, 0::FACT].astype(np.uint16)
        for dy in range(FACT):
            for dx in range(FACT):
                if dy or dx:
                    acc += patch[dy::FACT, dx::FACT]
        patch = (acc // (FACT * FACT)).astype(np.uint8)
    flip = False
    if augment and rng.random() < 0.5:
        patch = patch[:, ::-1]
        flip = True
    if not augment:
        return patch, (acx, acy), False
    p = patch.astype(np.float32) / 255.0
    if rng.random() < 0.8:
        p = p * rng.uniform(0.75, 1.35)                          # 亮度/对比度
        p = p + rng.uniform(-0.12, 0.12)
    if rng.random() < 0.5:
        p = np.clip(p, 0, 1) ** rng.uniform(0.75, 1.35)          # gamma
    return np.clip(p, 0, 1), (acx, acy), flip


def to_tensor(patch, flipped=False):
    a = np.ascontiguousarray(patch if patch.dtype == np.uint8 else patch)
    t = torch.from_numpy(a).permute(2, 0, 1).float()
    if a.dtype == np.uint8:
        t = t / 255.0
    return t


class Refiner(nn.Module):
    def __init__(self, w=(32, 64, 96, 128)):
        super().__init__()

        def blk(ci, co, s):
            return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                                 nn.Conv2d(co, co, 3, 1, 1, bias=False),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.e1 = blk(3, w[0], 2)        # 96
        self.e2 = blk(w[0], w[1], 2)     # 48
        self.e3 = blk(w[1], w[2], 2)     # 24
        self.e4 = blk(w[2], w[3], 2)     # 12
        self.d3 = blk(w[3] + w[2], w[2], 1)
        self.d2 = blk(w[2] + w[1], w[1], 1)
        self.head = nn.Conv2d(w[1], 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        u3 = F.interpolate(e4, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], 1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        return self.head(d2)             # (B,1,48,48) stride 4


_GX = None


def grid_px(dev):
    """热图每个 cell 的中心相对**裁块中心**的像素偏移，范围约 ±CROP/2。"""
    global _GX
    if _GX is None or _GX[0].device != dev:
        g = (torch.arange(GRID, device=dev, dtype=torch.float32) * STEP + STEP / 2.0
             - CROP / 2.0)
        _GX = (g, g)
    return _GX


def soft_argmax(hm, sharp=4):
    B = hm.shape[0]
    gx, gy = grid_px(hm.device)
    w = hm.clamp(min=0) ** sharp
    w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-8)
    px = (w * gx.view(1, 1, 1, -1)).sum(dim=(-1, -2)).view(B)
    py = (w * gy.view(1, 1, -1, 1)).sum(dim=(-1, -2)).view(B)
    return torch.stack([px, py], 1)                    # 相对裁块中心的像素偏移


def gauss_target(dev, tx, ty, sigma_px):
    gx, gy = grid_px(dev)
    d2 = ((gx.view(1, -1) - tx.view(-1, 1)) ** 2).unsqueeze(2) + \
         ((gy.view(1, -1) - ty.view(-1, 1)) ** 2).unsqueeze(1)
    return torch.exp(-d2 / (2 * sigma_px ** 2))


def predict_one(model, item, cx, cy, iters, sigma):
    """从 (cx,cy) 出发迭代 iters 次，返回最终全图坐标。"""
    dev = next(model.parameters()).device
    model.eval()
    cur = (cx, cy)
    with torch.no_grad():
        for _ in range(iters):
            patch, (acx, acy), _ = sample_crop(item, cur[0], cur[1], None, False)
            x = to_tensor(patch).unsqueeze(0).to(dev)
            hm = model(x)
            off = soft_argmax(hm, 4)[0].cpu().numpy()
            cur = (acx + float(off[0]), acy + float(off[1]))
    return cur


def run(args):
    torch.set_num_threads(args.threads)
    gt = load_gt()
    geom = json.loads((INB / 't3_geom.json').read_text(encoding='utf-8'))
    tr_cache = ROICache(geom['train'], 'train')
    print(f'[data] train ROI {len(tr_cache)} 张，ROI {2*ROI_HALF}px 见方', flush=True)

    ids = [it['i'] for it in tr_cache]
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(ids))
    folds = [perm[j::args.folds] for j in range(args.folds)]
    print(f'[cv] {args.folds} 折：' + ', '.join(f'{len(f)}' for f in folds), flush=True)

    errs_ref, errs_iter, errs_geom, per_img = [], [], [], []
    dev = torch.device('cpu')
    out = INB / f't3_run_refine_{args.tag}'
    out.mkdir(exist_ok=True)
    t_all = time.time()
    fdone = min(args.folds, args.max_folds or args.folds)
    for fi in range(fdone):
        va = set(int(ids[k]) for k in folds[fi])
        tr = [it for it in tr_cache if it['i'] not in va]
        va_items = [it for it in tr_cache if it['i'] in va]
        model = Refiner().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = max(1, len(tr) * args.crops // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps * args.epochs)
        rg = np.random.default_rng(1000 + fi)
        print(f'--- fold {fi}: train {len(tr)} / val {len(va_items)}  {steps} 步/epoch ---', flush=True)
        for ep in range(args.epochs):
            model.train()
            tl, nb = 0.0, 0
            idx = rg.permutation(len(tr) * args.crops)
            for s in range(steps):
                batch = idx[s * args.bs:(s + 1) * args.bs]
                xs, hs, tgt = [], [], []
                for b in batch:
                    it = tr[b // args.crops]
                    X, Y = gt[it['i']]
                    gx, gy = geom['train'][f'{it["i"]:04d}']
                    for _try in range(5):
                        jx = rg.uniform(-args.jit, args.jit)
                        jy = rg.uniform(-args.jit, args.jit)
                        patch, (acx, acy), flipped = sample_crop(it, gx + jx, gy + jy, rg, True)
                        tx, ty = X - acx, Y - acy
                        if flipped:
                            tx = -tx
                        if max(abs(tx), abs(ty)) <= args.guard:
                            break
                    xs.append(to_tensor(patch))
                    tgt.append([tx, ty])
                x = torch.stack(xs).to(dev)
                tt = torch.tensor(tgt, dtype=torch.float32, device=dev)
                hm = model(x)
                gh = gauss_target(dev, tt[:, 0], tt[:, 1], args.sigma * STEP)
                loss = F.mse_loss(torch.sigmoid(hm[:, 0]), gh) + \
                    args.w * F.l1_loss(soft_argmax(hm, 4), tt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tl += loss.detach().item(); nb += 1
            if args.log_every and (ep + 1) % args.log_every == 0:
                # 快速验证（1 次迭代）
                ve = []
                for it in va_items:
                    X, Y = gt[it['i']]
                    px, py = predict_one(model, it, *geom['train'][f'{it["i"]:04d}'], 1, args.sigma)
                    ve.append(math.hypot(px - X, py - Y))
                print(f'  f{fi} ep{ep+1:3d}/{args.epochs}  loss={tl/max(nb,1):.4f}  '
                      f'val(1it) mean={np.mean(ve):6.1f} med={np.median(ve):6.1f} '
                      f'max={max(ve):6.1f}  ({int(time.time()-t_all)}s)', flush=True)
        for it in va_items:
            X, Y = gt[it['i']]
            gx, gy = geom['train'][f'{it["i"]:04d}']
            px, py = predict_one(model, it, gx, gy, 1, args.sigma)
            qx, qy = predict_one(model, it, gx, gy, args.iters, args.sigma)
            errs_ref.append(math.hypot(px - X, py - Y))
            errs_iter.append(math.hypot(qx - X, qy - Y))
            errs_geom.append(math.hypot(gx - X, gy - Y))
            per_img.append(dict(i=it['i'], ref=[px, py], iter=[qx, qy],
                                geom=[gx, gy], true=[X, Y]))
        print(f'  fold {fi} done  ref(1it) mean={np.mean(errs_ref):.1f}  '
              f'ref({args.iters}it) mean={np.mean(errs_iter):.1f}  geom mean={np.mean(errs_geom):.1f}', flush=True)
        torch.save(model.state_dict(), out / f'fold{fi}.pt')

    e = np.array(errs_ref)
    e2 = np.array(errs_iter)
    g = np.array(errs_geom)
    print('=' * 60)
    print(f'细化1it n={len(e)}  mean={e.mean():.2f} median={np.median(e):.2f} '
          f'p90={np.percentile(e,90):.2f} max={e.max():.2f}')
    print(f'细化{args.iters}it n={len(e2)}  mean={e2.mean():.2f} median={np.median(e2):.2f} '
          f'p90={np.percentile(e2,90):.2f} max={e2.max():.2f}')
    print(f'几何基线 n={len(g)}  mean={g.mean():.2f} median={np.median(g):.2f} '
          f'p90={np.percentile(g,90):.2f} max={g.max():.2f}')
    print(f'胜率(1it): 细化更好 {int((e < g).sum())}/{len(e)} 张  平均改善 {g.mean()-e.mean():+.2f} px')
    (out / 'oof.json').write_text(json.dumps(dict(
        ref=errs_ref, iter=errs_iter, geom=errs_geom, per=per_img,
        mean=float(e.mean()), median=float(np.median(e)),
        mean_iter=float(e2.mean()), geom_mean=float(g.mean())), ensure_ascii=False), encoding='utf-8')
    print(f'[written] {out/"oof.json"}')

    if args.predict_test:
        te = ROICache(geom['test'], 'test')
        models = []
        for fi in range(fdone):
            m = Refiner().to(dev)
            m.load_state_dict(torch.load(out / f'fold{fi}.pt', map_location=dev))
            m.eval()
            models.append(m)
        pred = {}
        for it in te:
            gx, gy = geom['test'][f'{it["i"]:04d}']
            pts = [predict_one(m, it, gx, gy, args.iters, args.sigma) for m in models]
            px = float(np.mean([p[0] for p in pts]))
            py = float(np.mean([p[1] for p in pts]))
            pred[it['i']] = [px, py]
            print(f'  test {it["i"]:04d} geom=({gx:7.1f},{gy:7.1f}) -> ({px:7.1f},{py:7.1f})  '
                  f'分歧±({np.std([p[0] for p in pts]):.1f},{np.std([p[1] for p in pts]):.1f})', flush=True)
        pj = INB / f't3_pred_{args.tag}.json'
        pj.write_text(json.dumps({str(k): v for k, v in pred.items()}), encoding='utf-8')
        print(f'[written] {pj}')

        # 合并成第二级的先验文件：train = 留一折预测（无泄漏），test = 5 折集成
        geom2 = dict(
            train={f'{int(p["i"]):04d}': p['ref'] for p in per_img},
            test={f'{int(k):04d}': v for k, v in pred.items()},
        )
        g2 = INB / 't3_geom2.json'
        g2.write_text(json.dumps(geom2, ensure_ascii=False), encoding='utf-8')
        tr_e = np.array([math.hypot(p['ref'][0] - p['true'][0], p['ref'][1] - p['true'][1])
                         for p in per_img])
        print(f'[written] {g2}  (第一级留一法 mean={tr_e.mean():.2f} '
              f'med={np.median(tr_e):.2f} p90={np.percentile(tr_e,90):.2f} max={tr_e.max():.2f} '
              f'RMSE={np.sqrt((tr_e**2).mean()/2):.2f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--max-folds', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--crops', type=int, default=6)
    ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--sigma', type=float, default=2.5)
    ap.add_argument('--w', type=float, default=2.0)
    ap.add_argument('--jit', type=float, default=160.0)
    ap.add_argument('--guard', type=float, default=250.0)
    ap.add_argument('--iters', type=int, default=1)
    ap.add_argument('--predict-test', type=int, default=1)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--tag', default='w1')
    run(ap.parse_args())
