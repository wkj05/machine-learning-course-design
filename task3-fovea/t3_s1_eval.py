"""决定性诊断：拿冒烟跑出来的 fold-0 模型，在**它自己训练过的图**上评估。
若在训练图上准 ⇒ 管线正确，Y 塌陷只是欠训练；
若在自己训练图上也不准 ⇒ 坐标/映射有真 bug。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("T3_SMOKE", "1")
import numpy as np
import torch
import t3_kaggle as K

K.PREFLIGHT = False
root = K.find_root()
tr = K.load_train(root)
true_xy = np.array([[t[3], t[4]] for t in tr], np.float64)
cache = K.CanvasCache([{"path": root / "train" / t[0]} for t in tr], K.S1_CANVAS)
GH = K.S1_CANVAS // K.S1_STRIDE

ck = Path(K.OUT) / "s1_0_0.pt"
print(f"checkpoint = {ck}  exists={ck.is_file()}")
sd = torch.load(ck, map_location="cpu", weights_only=False)
cfg = sd["cfg"]
m = K.HeatNet(cfg["name"], False, K.S1_CH, K.S1_STRIDE, K.S1_DROP_PATH).to("cpu")
m.load_state_dict(sd["sd"])
m.eval()

folds = K.make_folds(len(tr), K.N_FOLDS, K.SEED)
va0 = folds[0]
tr0 = [i for i in range(len(tr)) if i not in set(va0)]
print(f"fold0: 训练 {len(tr0)} 张 / 验证 {len(va0)} 张")

def eval_on(idx, tag, use_argmax=True):
    errs, gerr = [], []
    with torch.no_grad():
        for i in idx:
            x = torch.from_numpy(cache.arr[i]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x = K.normalize(x)
            o = m(x)
            H = o.shape[-2]
            if use_argmax:
                j = int(o.flatten(1).argmax(1)[0])
                gx, gy = (j % H) + 0.5, (j // H) + 0.5
            else:
                sp = K.soft_argmax(o, T=4.0)[0].numpy()
                gx, gy = sp[0], sp[1]
            X, Y = cache.to_orig(i, gx * K.S1_STRIDE, gy * K.S1_STRIDE)
            errs.append(float(np.hypot(X - true_xy[i, 0], Y - true_xy[i, 1])))
            xc, yc = cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1])
            gerr.append(float(np.hypot(gx - xc / K.S1_STRIDE, gy - yc / K.S1_STRIDE)))
    print(f"[{tag}] n={len(idx)} 原图误差 mean={np.mean(errs):7.1f} med={np.median(errs):7.1f} "
          f"max={np.max(errs):7.1f} | 网格误差 mean={np.mean(gerr):.2f} 格")
    return errs, gerr

eval_on(tr0, "argmax 在自己训练图上")
eval_on(tr0, "soft_argmax T=4 在训练图上", use_argmax=False)
eval_on(va0, "argmax 在验证图上")

# 训练图上的预测坐标 vs 真值（看是否是"全预测中心"）
print("\n抽查 6 张训练图（argmax）：")
with torch.no_grad():
    for i in tr0[:6]:
        x = torch.from_numpy(cache.arr[i]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        o = m(K.normalize(x))
        H = o.shape[-2]
        j = int(o.flatten(1).argmax(1)[0])
        gx, gy = (j % H) + 0.5, (j // H) + 0.5
        X, Y = cache.to_orig(i, gx * K.S1_STRIDE, gy * K.S1_STRIDE)
        xc, yc = cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1])
        print(f"  #{i:02d} 预测网格=({gx:5.2f},{gy:5.2f}) 真网格=({xc/K.S1_STRIDE:5.2f},"
              f"{yc/K.S1_STRIDE:5.2f}) | 预测=({X:6.1f},{Y:6.1f}) 真=({true_xy[i,0]:6.1f},{true_xy[i,1]:6.1f})")
print(f"\n网格={GH}x{GH} 中心={GH/2:.1f}")
