# -*- coding: utf-8 -*-
"""任务三 训练原型：全图 -> 热图 -> 软 argmax -> 坐标。5 折交叉验证，误差以"原图像素欧氏距离"报告。

关键设计
  * 输入 512x512 letterbox（由 t3_data 统一换算），输出 128x128 热图（stride 4）。
  * 损失 = MSE(热图, 高斯)  +  2.0 * L1(软argmax坐标)。后者直接对齐评价指标（距离）。
  * 软argmax 用 hm**4 锐化，避免 128 网格量化误差（1 格 = 4 box px ≈ 23 原图像素）。
  * 误差一律在 orig 像素域计算并报告 mean / median / p90 / max。
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 该 python 是 embeddable 版，sys.path 里没有脚本目录

import t3_data as D

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox')
torch.set_num_threads(int(__import__('os').environ.get('T3_THREADS', '12')))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------- 模型 ----------------
def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNetSmall(nn.Module):
    """从零训练的小 UNet：512 -> 128 热图（stride 4）。"""
    def __init__(self, w=(32, 64, 128, 256, 256)):
        super().__init__()
        c1, c2, c3, c4, c5 = w
        self.e1 = nn.Sequential(cbr(3, c1), cbr(c1, c1))              # 512
        self.e2 = nn.Sequential(cbr(c1, c2, s=2), cbr(c2, c2))        # 256
        self.e3 = nn.Sequential(cbr(c2, c3, s=2), cbr(c3, c3))        # 128
        self.e4 = nn.Sequential(cbr(c3, c4, s=2), cbr(c4, c4))        # 64
        self.e5 = nn.Sequential(cbr(c4, c5, s=2), cbr(c5, c5))        # 32
        self.d4 = cbr(c5 + c4, c3)
        self.d3 = cbr(c3 + c3, c2)
        self.d2 = cbr(c2 + c2, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        u = F.interpolate(e5, size=e4.shape[-2:], mode='nearest')
        u = self.d4(torch.cat([u, e4], 1))
        u = F.interpolate(u, size=e3.shape[-2:], mode='nearest')
        u = self.d3(torch.cat([u, e3], 1))
        u = F.interpolate(u, size=e2.shape[-2:], mode='nearest')
        u = self.d2(torch.cat([u, e2], 1))
        return self.out(u)                                            # 256 网格 (stride 2)


class CnxTHeat(nn.Module):
    """convnext_tiny（ImageNet 预训练）+ 轻量 FPN 头。"""
    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        self.bb = timm.create_model('convnext_tiny.fb_in22k_ft_in1k',
                                    features_only=True, pretrained=pretrained,
                                    out_indices=(1, 2, 3))   # stride 8,16,32
        chs = self.bb.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in chs])
        self.fuse = cbr(64, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        fs = self.bb(x)
        base = fs[0].shape[-2:]
        m = sum(F.interpolate(l(f), size=base, mode='nearest')
                for l, f in zip(self.lat, fs))
        return self.out(self.fuse(m))                                # stride 8


def soft_argmax(hm, sharpen=4.0):
    """hm: [B,1,H,W] 已 sigmoid。返回 [B,2] 归一化到 [0,1]。"""
    B, _, H, W = hm.shape
    w = (hm.view(B, -1) ** sharpen) + 1e-8
    p = w / w.sum(1, keepdim=True)
    ys = (torch.arange(H, device=hm.device, dtype=hm.dtype) + 0.5) / H
    xs = (torch.arange(W, device=hm.device, dtype=hm.dtype) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    gx = xx.reshape(1, -1)
    gy = yy.reshape(1, -1)
    return torch.stack([(p * gx).sum(1), (p * gy).sum(1)], 1)


def make_target(xy, H, W, sigma_cells):
    """xy: [B,2] 归一化坐标；返回 [B,1,H,W] 的高斯。"""
    B = xy.shape[0]
    ys = (torch.arange(H, device=xy.device, dtype=xy.dtype) + 0.5) / H
    xs = (torch.arange(W, device=xy.device, dtype=xy.dtype) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    cx = xy[:, 0].view(B, 1, 1)
    cy = xy[:, 1].view(B, 1, 1)
    s = sigma_cells / H
    g = torch.exp(-((xx.unsqueeze(0) - cx) ** 2 + (yy.unsqueeze(0) - cy) ** 2) / (2 * s * s))
    return g.unsqueeze(1)


class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, items, idxs, train, rng):
        self.imgs, self.items, self.idxs, self.train, self.rng = imgs, items, idxs, train, rng

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = self.idxs[k]
        it = self.items[i - 1]
        img = self.imgs[i]
        if self.train:
            img, nx, ny = D.rand_augment(img, it.bx, it.by, self.rng)
            xy = np.array([nx / D.S, ny / D.S], dtype=np.float32)
        else:
            xy = np.array([it.bx / D.S, it.by / D.S], dtype=np.float32)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        t = (t - MEAN[0]) / STD[0]
        return t, torch.from_numpy(xy)


def orig_dist(pred_xy, items, idxs):
    """pred_xy: [N,2] 归一化 box 坐标 -> orig 像素欧氏距离数组。"""
    out = []
    for k, i in enumerate(idxs):
        it = items[i - 1]
        bx, by = float(pred_xy[k, 0]) * D.S, float(pred_xy[k, 1]) * D.S
        X, Y = it.box_to_orig(bx, by)
        out.append(math.hypot(X - it.X, Y - it.Y))
    return np.array(out)


def run(args):
    items = D.load_items()
    print(f'加载 {len(items)} 张...', flush=True)
    imgs = {it.idx: D.load_box(it) for it in items}
    ids = [it.idx for it in items]

    rng0 = np.random.default_rng(42)
    perm = rng0.permutation(len(ids))
    folds = [[] for _ in range(args.folds)]
    for j, p in enumerate(perm):
        folds[j % args.folds].append(ids[p])
    if args.max_folds:
        folds = folds[:args.max_folds]

    all_err, all_pred, all_true = [], {}, {}
    for f, va in enumerate(folds):
        tr = [i for i in ids if i not in va]
        torch.manual_seed(42 + f)
        rng = np.random.default_rng(1000 + f)
        net = CnxTHeat(pretrained=not args.no_pretrained) if args.bb == 'cnxt' else UNetSmall()
        if args.freeze and args.bb == 'cnxt':
            for p in net.bb.parameters():
                p.requires_grad_(False)
        net.train()

        head = [p for n, p in net.named_parameters() if not n.startswith('bb.')]
        groups = [{'params': head, 'lr': args.lr}]
        if args.bb == 'cnxt' and not args.freeze:
            groups.append({'params': net.bb.parameters(), 'lr': args.lr * 0.1})
        opt = torch.optim.AdamW(groups, weight_decay=args.wd)
        steps = math.ceil(len(tr) / args.bs) * args.epochs
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups],
                                                  total_steps=steps, pct_start=0.25)

        dl = torch.utils.data.DataLoader(DS(imgs, items, tr, True, rng),
                                         batch_size=args.bs, shuffle=True,
                                         num_workers=0, drop_last=len(tr) > args.bs)
        t0 = time.time()
        for ep in range(args.epochs):
            tot, n = 0.0, 0
            for x, xy in dl:
                out = net(x)
                hm = torch.sigmoid(out)
                tgt = make_target(xy, hm.shape[-2], hm.shape[-1], args.sigma)
                l_hm = F.mse_loss(hm, tgt)
                l_xy = F.l1_loss(soft_argmax(hm), xy)
                loss = l_hm + args.w_xy * l_xy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
                tot += loss.detach().item() * x.shape[0]
                n += x.shape[0]
            if (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
                net.eval()
                with torch.no_grad():
                    dlv = torch.utils.data.DataLoader(DS(imgs, items, va, False, None),
                                                      batch_size=8, shuffle=False)
                    ps, ts_ = [], []
                    for x, xy in dlv:
                        ps.append(soft_argmax(torch.sigmoid(net(x))))
                        ts_.append(xy)
                    p = torch.cat(ps).numpy()
                    err = orig_dist(p, items, va)
                net.train()
                print(f'  f{f} ep{ep+1:3d}/{args.epochs}  loss={tot/max(n,1):.4f}  '
                      f'val: mean={err.mean():6.1f} med={np.median(err):6.1f} '
                      f'max={err.max():6.1f}  ({time.time()-t0:.0f}s)', flush=True)

        net.eval()
        with torch.no_grad():
            dlv = torch.utils.data.DataLoader(DS(imgs, items, va, False, None),
                                              batch_size=8, shuffle=False)
            ps = torch.cat([soft_argmax(torch.sigmoid(net(x))) for x, _ in dlv]).numpy()
        err = orig_dist(ps, items, va)
        all_err.append(err)
        for k, i in enumerate(va):
            all_pred[i] = (float(ps[k, 0]) * D.S, float(ps[k, 1]) * D.S)
            all_true[i] = (items[i - 1].X, items[i - 1].Y)
        print(f'[fold {f}] n_val={len(va)}  mean={err.mean():.1f} med={np.median(err):.1f} '
              f'p90={np.percentile(err,90):.1f} max={err.max():.1f}', flush=True)

    e = np.concatenate(all_err)
    print(f'\n===== OOF ({len(e)} 张, {args.folds} 折) =====')
    print(f'mean={e.mean():.2f}px  median={np.median(e):.2f}px  p90={np.percentile(e,90):.2f}px  '
          f'max={e.max():.2f}px')
    outd = ROOT / ('t3_run_' + args.tag)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / 'oof.json').write_text(json.dumps(
        {'args': vars(args), 'err': e.tolist(), 'pred': {str(k): v for k, v in all_pred.items()},
         'true': {str(k): v for k, v in all_true.items()},
         'mean': float(e.mean()), 'median': float(np.median(e))}, indent=1), encoding='utf-8')
    print('[saved]', outd / 'oof.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bb', default='scratch', choices=['scratch', 'cnxt'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--sigma', type=float, default=1.6)
    ap.add_argument('--w-xy', type=float, default=2.0)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--max-folds', type=int, default=0)
    ap.add_argument('--tag', default='proto')
    run(ap.parse_args())
