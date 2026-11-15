"""看 S1Dataset 吐出来的图到底长什么样（怀疑 photometric/normalize 把图毁了）。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("T3_SMOKE", "1")
import numpy as np
import torch
import t3_kaggle as K
from PIL import Image

K.PREFLIGHT = False
root = K.find_root()
tr = K.load_train(root)
true_xy = np.array([[t[3], t[4]] for t in tr], np.float64)
cache = K.CanvasCache([{"path": root / "train" / t[0]} for t in tr], K.S1_CANVAS)
xy_canvas = [cache.to_canvas(i, true_xy[i, 0], true_xy[i, 1]) for i in range(len(tr))]

sel = [0, 1, 2, 40, 41, 42]
K.set_seed(7)
ds = K.S1Dataset(cache, sel, [xy_canvas[i] for i in sel], True, K.S1_SIGMA, K.S1_STRIDE)

print("原始画布图 vs dataset 输出的统计：")
for k, i in enumerate(sel):
    raw = cache.arr[i].astype(np.float32) / 255.0
    img, tgt, xyn = ds[k]
    a = img.numpy().transpose(1, 2, 0)
    print(f"  #{i:02d} raw: mean={raw.mean():.3f} std={raw.std():.3f} min={raw.min():.3f} max={raw.max():.3f}")
    print(f"        ds : mean={a.mean():.3f} std={a.std():.3f} min={a.min():.3f} max={a.max():.3f}"
          f"  tgt.max={float(tgt.max()):.3f} tgt.sum={float(tgt.sum()):.2f} xyn={xyn.numpy().round(3)}")

out = Path("_inbox/t3_look")
out.mkdir(parents=True, exist_ok=True)
tiles = []
for k in range(len(sel)):
    img, tgt, xyn = ds[k]
    # 反归一化回可看的范围
    a = img.numpy().transpose(1, 2, 0) * np.array(K.STD, np.float32) + np.array(K.MEAN, np.float32)
    a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    t = np.repeat((tgt.numpy()[0] * 255).astype(np.uint8)[..., None], 3, 2)
    tiles.append((a, t))

S = K.S1_CANVAS
sheet = np.zeros((2 * S, len(sel) * S, 3), np.uint8)
for k, (a, t) in enumerate(tiles):
    sheet[0:S, k * S:(k + 1) * S] = a
    sheet[S:2 * S, k * S:(k + 1) * S] = t
Image.fromarray(sheet).save(out / "ds_check.png")
print(f"\n已写出 {out / 'ds_check.png'}  (上排=增强后的图, 下排=高斯目标)")
print("若上排是灰色/纯色/看不出眼底 ⇒ photometric 或 normalize 有 bug")
