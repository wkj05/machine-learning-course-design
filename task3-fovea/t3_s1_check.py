"""第一阶段坐标链自检：标签 → canvas → gauss_target → soft_argmax → to_orig。
目的：判断冒烟里「Y 全堆在网格顶部（还原成 Y≈0）」是欠训练还是坐标 bug。
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
_tr = K.load_train(root)
_te = K.load_test(root)
true_xy = np.array([[_t[3], _t[4]] for _t in _tr], np.float64)
print(f"DATA_ROOT = {root}  train={len(_tr)} test={len(_te)}")

cache = K.CanvasCache([{"path": root / "train" / t[0]} for t in _tr], K.S1_CANVAS)
GH = K.S1_CANVAS // K.S1_STRIDE
GW = GH
print(f"canvas={K.S1_CANVAS} grid={GH}x{GW} stride={K.S1_STRIDE} sigma={K.S1_SIGMA}")

# [1] 真值 → 网格 → 高斯 → soft_argmax 应还原同一个网格坐标
e1 = []
for i in range(len(_tr)):
    xc, yc = cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1])
    gx, gy = xc / K.S1_STRIDE, yc / K.S1_STRIDE
    hm = K.gauss_target(GH, GW, gx, gy, K.S1_SIGMA, torch.device("cpu"))[None, None]
    sp = K.soft_argmax(hm, T=1.0)[0].numpy()
    e1.append(float(np.hypot(sp[0] - gx, sp[1] - gy)))
print(f"[1] gauss->soft_argmax 网格误差 mean={np.mean(e1):.3f} max={np.max(e1):.3f} 格")

# [2] 完整往返：到原图像素
e2 = []
for i in range(len(_tr)):
    xc, yc = cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1])
    gx, gy = xc / K.S1_STRIDE, yc / K.S1_STRIDE
    hm = K.gauss_target(GH, GW, gx, gy, K.S1_SIGMA, torch.device("cpu"))[None, None]
    sp = K.soft_argmax(hm, T=1.0)[0].numpy()
    X, Y = cache.to_orig(i, sp[0] * K.S1_STRIDE, sp[1] * K.S1_STRIDE)
    e2.append(float(np.hypot(X - true_xy[i, 0], Y - true_xy[i, 1])))
print(f"[2] 完整往返(原图像素) mean={np.mean(e2):.2f} max={np.max(e2):.2f} px")

# [3] dataset 实样本（train=False 才不做增广）
idx = list(range(len(_tr)))
xyl = [cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1]) for i in idx]
ds = K.S1Dataset(cache, idx, xyl, False, K.S1_SIGMA, K.S1_STRIDE)
for k in (0, 1, 40):
    img, hm, xyn = ds[k]
    sp = K.soft_argmax(hm[None], T=1.0)[0].numpy()
    X, Y = cache.to_orig(k, sp[0] * K.S1_STRIDE, sp[1] * K.S1_STRIDE)
    print(f"    #{k:02d} img={tuple(img.shape)} hm={tuple(hm.shape)} "
          f"峰=({sp[0]:5.2f},{sp[1]:5.2f})格 还原=({X:6.1f},{Y:6.1f}) "
          f"真=({true_xy[k,0]:6.1f},{true_xy[k,1]:6.1f}) xyn={xyn.numpy().round(4)}")

# [4] 画布上 GT 的纵向位置（期望网格行）
rows = []
for i in range(len(_tr)):
    xc, yc = cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1])
    rows.append(yc / K.S1_STRIDE)
print(f"[4] GT 网格行 mean={np.mean(rows):.2f} std={np.std(rows):.2f} "
      f"range=[{np.min(rows):.2f},{np.max(rows):.2f}]  (网格高 {GH})")

# [5] 打印 xyn 的统计（训练用的归一化标签）
xn = np.array([xyl[i][0] / K.S1_STRIDE / GW for i in idx])
yn = np.array([xyl[i][1] / K.S1_STRIDE / GH for i in idx])
print(f"[5] xyn 统计 x: mean={xn.mean():.4f} std={xn.std():.4f} | y: mean={yn.mean():.4f} std={yn.std():.4f}")
print("    若 y 的 std 极小或 mean 贴近 0/1，说明归一化把目标压扁了")
