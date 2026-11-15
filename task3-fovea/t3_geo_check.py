"""几何一致性自测：apply_geom 与 geom_label 必须把同一个点送到同一个位置。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F

import importlib.util

spec = importlib.util.spec_from_file_location("t3k", Path(__file__).resolve().parent / "t3_kaggle.py")
t3k = importlib.util.module_from_spec(spec)
sys.modules["t3k"] = t3k
spec.loader.exec_module(t3k)

H = W = 192
worst = 0.0
worst_case = None
rng = torch.Generator().manual_seed(0)
cases = [
    (0.0, 1.0, 0.0, 0.0, False),
    (0.0, 1.0, 0.0, 0.0, True),
    (15.0, 1.0, 0.0, 0.0, False),
    (-20.0, 1.10, 0.0, 0.0, False),
    (0.0, 1.05, 0.03, -0.02, False),
    (8.0, 0.95, -0.03, 0.02, True),
    (-12.0, 1.10, 0.03, 0.03, True),
]

for (ang, sc, tx, ty, flip) in cases:
    for _ in range(6):
        xc = float(torch.rand(1, generator=rng).item()) * (W - 20) + 10
        yc = float(torch.rand(1, generator=rng).item()) * (H - 20) + 10
        # 造一张带亮点的高斯图
        ys = torch.arange(H, dtype=torch.float32).view(H, 1)
        xs = torch.arange(W, dtype=torch.float32).view(1, W)
        img = torch.exp(-((xs - xc) ** 2 + (ys - yc) ** 2) / (2 * 1.5 ** 2))
        img3 = img.unsqueeze(0).repeat(3, 1, 1)
        out = t3k.apply_geom(img3, ang, sc, tx, ty, flip)
        pk = float(out[0].max())
        idx = int(out[0].argmax())
        py, px = divmod(idx, W)
        ex, ey = t3k.geom_label(xc, yc, H, W, ang, sc, tx, ty, flip)
        d = math.hypot(px - ex, py - ey)
        if pk < 0.05:
            continue  # 亮点被转出画面，跳过
        if d > worst:
            worst, worst_case = d, (ang, sc, tx, ty, flip, xc, yc, px, py, ex, ey)
        assert d < 2.0, f"几何不一致: 变换后亮点在 ({px},{py}) 而 geom_label 给出 ({ex:.2f},{ey:.2f}) d={d:.2f} case={ang,sc,tx,ty,flip}"

print(f"OK 几何一致：最大偏差 {worst:.3f} px")
if worst_case:
    print("  最差 case:", worst_case)

# --- 第二阶段 flip 标签自测 ---
print("S2 GRID =", t3k.S2_NET // t3k.S2_STRIDE, " STRIDE =", t3k.S2_STRIDE,
      " 1 cell(原图 px) =", t3k.S2_STRIDE)
g = t3k.S2_NET // t3k.S2_STRIDE
for odx in (-100.0, -30.0, 0.0, 30.0, 100.0):
    gx = (odx + t3k.S2_CROP / 2.0 - t3k.S2_STRIDE / 2.0) / t3k.S2_STRIDE
    back = gx * t3k.S2_STRIDE - (t3k.S2_CROP / 2.0 - t3k.S2_STRIDE / 2.0)
    print(f"  offset {odx:+.1f} → grid {gx:.2f} → 回到 {back:+.2f}")
    assert abs(back - odx) < 1e-6
print("OK 第二阶段坐标往返一致")

# --- soft_argmax 精度自测 ---
for truth in [(3.5, 5.5), (10.2, 7.8), (20.0, 20.0)]:
    tgt = t3k.gauss_target(32, 32, truth[0], truth[1], 1.6, torch.device("cpu"))
    lg = torch.log(tgt.clamp_min(1e-9)).unsqueeze(0).unsqueeze(0)
    sp = t3k.soft_argmax(lg, T=1.0)[0]
    print(f"  soft_argmax truth={truth} → ({sp[0]:.3f},{sp[1]:.3f})")
    assert abs(float(sp[0]) - truth[0]) < 0.3 and abs(float(sp[1]) - truth[1]) < 0.3
print("OK soft_argmax 精度")
