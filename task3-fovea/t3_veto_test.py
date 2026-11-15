"""验证视盘否决是否真能救掉 S1 最差那几张。

用 Kaggle 那次跑出来的 S1 留一法逐图误差（日志里给了最差 8 张的 dX/dY）
反推每张的预测点 = GT + (dX, dY)，然后走一遍 t3_kaggle.py 里的
disc_orig / fit_geo / geo_point / veto_fix，看否决到底会不会触发、触发后误差变多少。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_inbox"))

import numpy as np
from PIL import Image

import t3_kaggle as K

ROOT = Path(r"C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection")
TRAIN = ROOT / "train"
TEST = ROOT / "test"

# ---- 复盘数据 ----
gt = {}
with open(ROOT / "fovea_localization_train_GT.csv", "rb") as f:
    head = f.readline().decode("utf-8").strip()
    for ln in f:
        p = ln.decode("utf-8").strip().split(",")
        if len(p) >= 3:
            gt[int(p[0])] = (float(p[1]), float(p[2]))
print(f"GT 行数 = {len(gt)}  表头 = {head}")

ids = sorted(gt)
paths = [TRAIN / f"{i:04d}.jpg" for i in ids]
sizes = []
for p in paths:
    with Image.open(p) as im:
        sizes.append(im.size)
true_xy = np.array([gt[i] for i in ids], np.float64)

# Kaggle 日志里 S1 留一法的最差 8 张（图号: 误差px, dX, dY）
WORST8 = {
    72: (353, +234, -264),
    49: (147, +143, -35),
    62: (101, +73, +70),
    24: (98, +96, -19),
    57: (69, +61, +33),
    44: (65, -65, +3),
    16: (49, -47, -11),
    34: (41, -1, -41),
}

print("\n--- 视盘检测 ---")
discs = []
for p in paths:
    d = K.disc_orig(p)
    discs.append(d)
nok = sum(d is not None for d in discs)
print(f"训练集检出 {nok}/{len(paths)}")
if nok:
    rs = [d[2] for d in discs if d is not None]
    print(f"半径 R: mean={np.mean(rs):.1f} min={min(rs):.1f} max={max(rs):.1f}")

geo_full = K.fit_geo(discs, true_xy, sizes)
print(f"\n全量拟合 D={ {k: round(v) for k, v in geo_full[0].items()} } "
      f"Y={ {k: round(v) for k, v in geo_full[1].items()} }")

print("\n--- 几何先验自身的留一法误差 ---")
loo = np.zeros((len(ids), 2))
for i in range(len(ids)):
    g = K.fit_geo(discs, true_xy, sizes, sel=[j for j in range(len(ids)) if j != i])
    pt = K.geo_point(discs[i], sizes[i], g)
    loo[i] = pt if pt is not None else (np.nan, np.nan)
okloo = ~np.isnan(loo[:, 0])
d = np.linalg.norm(loo[okloo] - true_xy[okloo], axis=1)
print(f"n={int(okloo.sum())} mean={d.mean():.2f} median={np.median(d):.2f} max={d.max():.2f}")
print(f"RMSE = {np.sqrt(np.mean(d ** 2) / 2):.3f}")

print("\n--- 最差 8 张：否决会不会触发 ---")
idx_of = {sid: i for i, sid in enumerate(ids)}
tot = 0.0
for sid, (err, dx, dy) in WORST8.items():
    i = idx_of[sid]
    g = K.fit_geo(discs, true_xy, sizes, sel=[j for j in range(len(ids)) if j != i])
    fb = K.geo_point(discs[i], sizes[i], g)
    pred = np.array([true_xy[i, 0] + dx, true_xy[i, 1] + dy])
    dd = discs[i]
    dist_disc = float(np.hypot(pred[0] - dd[0], pred[1] - dd[1])) if dd else float("nan")
    fire = dd is not None and fb is not None and dist_disc < K.VETO_MIN_DIST
    new = fb if fire else pred
    new_err = float(np.linalg.norm(new - true_xy[i]))
    tot += new_err ** 2 - err ** 2
    print(f"  {sid:04d}  原误差 {err:3d}px  预测点距视盘 {dist_disc:6.1f}px  "
          f"{'★否决' if fire else '不动'}  之后 {new_err:6.1f}px")

print("\n--- 若只改这 8 张，对整体 RMSE 的影响 ---")
# 已知 S1 留一法整体：mean=19.84 median=7.21 p90=36.04 max=352.83 RMSE=34.397
base_rmse = 34.397
base_tot = (base_rmse ** 2) * 2 * len(ids)
print(f"原总平方误差 = {base_tot:.0f}")
print(f"这 8 张的平方误差净变化 = {tot:+.0f}")
print(f"改后 RMSE = {np.sqrt((base_tot + tot) / 2 / len(ids)):.3f}")
print(f"改后榜分 ≈ {0.62 * np.sqrt((base_tot + tot) / 2 / len(ids)):.2f}")
