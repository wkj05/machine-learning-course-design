"""决定性验收测试：能不能把 5 张图 **过拟合** 到 30px 以内。
能 ⇒ 训练器正确（冒烟失败只是欠训练）；不能 ⇒ 训练器有真 bug。
顺便打印已训练模型的 logits 统计，看是否边界伪峰。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("T3_SMOKE", "1")
import numpy as np
import torch
import torch.nn.functional as F
import t3_kaggle as K

K.PREFLIGHT = False
root = K.find_root()
tr = K.load_train(root)
true_xy = np.array([[t[3], t[4]] for t in tr], np.float64)
cache = K.CanvasCache([{"path": root / "train" / t[0]} for t in tr], K.S1_CANVAS)
GH = K.S1_CANVAS // K.S1_STRIDE
print(f"canvas={K.S1_CANVAS} grid={GH} stride={K.S1_STRIDE} sigma={K.S1_SIGMA}")

xy_canvas = [cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1]) for i in range(len(tr))]

# ---------- A. 看已训练模型的 logits ----------
ck = Path(K.OUT) / "s1_0_0.pt"
if ck.is_file():
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    m = K.HeatNet(sd["cfg"]["name"], False, K.S1_CH, K.S1_STRIDE, K.S1_DROP_PATH).to("cpu")
    m.load_state_dict(sd["sd"])
    m.eval()
    print("\n[A] 已训练模型（冒烟 fold0）的 logits：")
    with torch.no_grad():
        for i in (1, 4, 9):
            x = K.normalize(torch.from_numpy(cache.arr[i]).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
            o = m(x)[0, 0]
            gx, gy = xy_canvas[i][0] / K.S1_STRIDE, xy_canvas[i][1] / K.S1_STRIDE
            gi, gj = int(gy), int(gx)
            print(f"  #{i:02d} logit min={o.min():7.2f} max={o.max():7.2f} mean={o.mean():7.2f} "
                  f"std={o.std():6.2f} | 峰@({int(o.argmax())%GH},{int(o.argmax())//GH}) "
                  f"| GT格({gj},{gi}) logit={o[gj,gi]:7.2f} | 边界均值={o[0,:].mean():6.2f} {o[-1,:].mean():6.2f}")

# ---------- B. 过拟合 5 张 ----------
print("\n[B] 过拟合 5 张图（300 epoch，canvas=%d）：" % K.S1_CANVAS)
sel = [0, 1, 2, 40, 41]
K.set_seed(123)
m = K.HeatNet("convnext_tiny.fb_in22k_ft_in1k", False, K.S1_CH, K.S1_STRIDE, K.S1_DROP_PATH).to("cpu")
opt = torch.optim.AdamW(m.parameters(), lr=K.S1_LR, weight_decay=K.S1_WD)
NB = 300
steps = NB * 3
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.S1_LR, total_steps=max(1, steps), pct_start=0.25)
ds = K.S1Dataset(cache, sel, [xy_canvas[i] for i in sel], True, K.S1_SIGMA, K.S1_STRIDE)
dl = torch.utils.data.DataLoader(ds, batch_size=K.S1_BATCH, shuffle=True, num_workers=0, drop_last=False)
for ep in range(NB):
    m.train()
    tot, n = 0.0, 0
    for img, tgt, xyn in dl:
        out = m(img)
        gauss = tgt
        lhm = F.binary_cross_entropy_with_logits(out, gauss)
        gh2, gw2 = out.shape[-2], out.shape[-1]
        sp = K.soft_argmax(out, T=4.0)
        tt = torch.stack([xyn[:, 0] * gw2, xyn[:, 1] * gh2], 1)   # ★ 换成格坐标
        lxy = F.smooth_l1_loss(sp, tt, beta=1.0)
        loss = lhm + K.S1_WXY * lxy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        try:
            sched.step()
        except Exception:
            pass
        tot += loss.item() * img.size(0)
        n += img.size(0)
    if ep % 25 == 0 or ep == NB - 1:
        # 评估（不加增广）
        m.eval()
        errs = []
        with torch.no_grad():
            for i in sel:
                x = K.normalize(torch.from_numpy(cache.arr[i]).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
                o = m(x)
                j = int(o.flatten(1).argmax(1)[0])
                gx, gy = (j % GH) + 0.5, (j // GH) + 0.5
                X, Y = cache.to_orig(i, gx * K.S1_STRIDE, gy * K.S1_STRIDE)
                errs.append(float(np.hypot(X - true_xy[i, 0], Y - true_xy[i, 1])))
        print(f"  ep{ep+1:3d} loss={tot/max(1,n):8.4f}  argmax误差 mean={np.mean(errs):7.1f} px  "
              f"逐张={[round(e, 1) for e in errs]}")
        m.train()

print("\n判据：60 epoch 后 5 张图的 mean 误差应 < 60 px。若仍 ~2000 px，训练器有真 bug。")
