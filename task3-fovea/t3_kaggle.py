"""
任务三 · 视网膜眼底图黄斑中心凹定位 —— Kaggle 单文件训练脚本
================================================================
指标：RMSE（把 20 张测试图拆成 X/Y 共 40 个值算），越小越好。
格式：严格照抄 sample_submission.csv 的行序（81_Fovea_X, 81_Fovea_Y, 82_Fovea_X, ...）。

【怎么用】
1. Kaggle 新建 notebook → File → Import Notebook 或新建空白 → 把本文件全文粘进**一个 code cell**。
2. 右侧 Add Input → 选 2026task3 的数据集（比赛规则需先接受）。
3. Settings：Accelerator = GPU T4 x2（或 P100）；Internet = **On**（要下 timm 预训练权重）。
4. Save Version → Save & Run All (Commit)。一次 commit 硬上限 12h；本脚本 TIME_BUDGET_MIN=480
   （8 小时）留了余量，到点会**自动停止训练并照常出提交文件**。
5. 跑完在 Output 里取 `submission.csv`。

【为什么这么设计】（本地实测结论，不是拍脑袋）
- 单阶段整图回归在 512 输入下只能到 ~130 px：黄斑在降采样后只剩约 10 px，被糊掉了。
- 单阶段 1:1 裁块回归在「先验已经准」时能到 RMSE 15.13（榜首档），但训练标签只在中心
  ±90 px 内，它学到的是「在中心附近微调」，不是「去找黄斑」——先验偏 60 px 以上就基本不动。
- 所以必须两级：**第一级整图粗定位（本脚本用 1024 长边 + 预训练主干 + stride 8 热图）→
  第二级 1:1 裁块精定位（256 裁块，stride 4）**。先验只要压到 60 px 以内，第二级就是榜首档。
- 本地实测的失败路线（**不要重试**）：加宽裁块视野（320）→ RMSE 36.5；9 点滑窗聚类 → 38.6；
  翻转 TTA 单用 → 29.5；多候选共识融合 → 29.6。都是「下游再聪明也救不了没在找目标的模型」。
"""

import os
import sys
import json
import math
import time
import random
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

# ============================== CONFIG ==============================
SEED = 42
N_FOLDS = 5
TIME_BUDGET_MIN = 480.0          # 真实训练预算（分钟）；到点停训、照常出提交
RESERVE_MIN = max(0.4, min(12.0, TIME_BUDGET_MIN * 0.04))   # 留给出提交/收尾的时间

# ---- 第一阶段：整图热图粗定位 ----
S1_BACKBONES = [
    "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnet_b0.ra_in1k",
]
S1_SEEDS = [42]                  # 预算够的话可以加 2024 换更多成员
S1_CANVAS = 1024                 # 长边缩放到这个尺寸，再 letterbox 成方图
S1_STRIDE = 8                    # 热图步长（→ canvas/8 的网格；1024 → 128x128）
S1_CH = 96                       # FPN 通道
S1_SIGMA = 2.0                   # 高斯标签 σ（网格单位）≈ 2*8*3 = 48 原图像素
S1_EPOCHS = 110                  # 上限；预算不够时按 §plan_epochs 自动下调
S1_BATCH = 4
S1_LR = 3e-4
S1_WD = 0.05
S1_WXY = 1.0
S1_DROP_PATH = 0.1
S1_EMA = 0.0                     # >0 启用权重 EMA（0.999 之类）

# ---- 第二阶段：1:1 裁块精定位 ----
S2_CROP = 256                    # 裁块边长（原图像素）
S2_NET = 256                     # 网络输入边长（= CROP，即 1:1）
S2_STRIDE = 4                    # 热图步长（原图像素/格）
S2_ROI_HALF = 320                # ROI 缓存半径（原图像素），要 >= CROP/2 + 抖动
S2_SIGMA = 3.0                   # 高斯标签 σ（网格单位）≈ 12 原图像素
S2_JIT = 100.0                   # 训练时裁剪中心相对先验的随机抖动（原图像素）
S2_GUARD = 118.0                 # 真值必须落在裁块内这个半径内，否则重抽
S2_SEEDS = [42]
S2_EPOCHS = 120
S2_BATCH = 8
S2_LR = 1e-3
S2_WD = 0.02
S2_WXY = 2.0

# 每个候选主干都先跑一次；某个主干建不起来就自动跳过
PREFLIGHT = True

SMOKE = os.environ.get("T3_SMOKE", "0") == "1"
if SMOKE:
    N_FOLDS = 2
    TIME_BUDGET_MIN = 8.0
    S1_BACKBONES = S1_BACKBONES[:1]
    S1_SEEDS = S1_SEEDS[:1]
    S1_CANVAS = 256
    S1_EPOCHS = 2
    S1_BATCH = 2
    S2_SEEDS = S2_SEEDS[:1]
    S2_EPOCHS = 2
    S2_BATCH = 2
    S2_ROI_HALF = 320
    PREFLIGHT = False

# 预算相关常量在 SMOKE 覆盖之后重算一次
RESERVE_MIN = max(0.4, min(12.0, TIME_BUDGET_MIN * 0.04))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

T0 = time.time()
DEADLINE = T0 + TIME_BUDGET_MIN * 60.0


def now():
    return time.strftime("%H:%M:%S")


def el():
    return (time.time() - T0) / 60.0


def log(*a):
    print(f"[{now()} +{el():5.1f}m]", *a, flush=True)


def left_min():
    return (DEADLINE - time.time()) / 60.0


_EST_EPOCH = {}     # kind -> 每 epoch 分钟数（实测后填入）


def plan_epochs(kind, folds_left, cap):
    """在「剩余预算 / 还剩多少个折」的约束下，给这一折分配多少 epoch。

    目的：宁可把 5 折都跑完（每折少几个 epoch），也不要只跑完 2 折就预算耗尽 ——
    集成带来的收益（本地实测 cnxt 单模型 0.7006 → 5 折 2 视角 0.7139，+1.33 点）比多训几个 epoch 大。
    """
    est = _EST_EPOCH.get(kind)
    if not est or est <= 0:
        return cap
    budget = max(0.5, left_min() - RESERVE_MIN)
    per_fold = budget / max(1, folds_left)
    return int(max(20, min(cap, per_fold * 0.85 / est)))


# ============================== 基础工具 ==============================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def out_dir():
    p = Path("/kaggle/working")
    if p.is_dir():
        return p
    p = Path.cwd() / "t3_out"
    p.mkdir(parents=True, exist_ok=True)
    return p


OUT = out_dir()


def find_root():
    """找含 train/ + fovea_localization_train_GT.csv 的目录。"""
    cands = []
    for base in ("/kaggle/input", "_inbox/task3", "task3", "."):
        b = Path(base)
        if not b.exists():
            continue
        for p in [b] + [x for x in b.rglob("*") if x.is_dir()]:
            try:
                rel = len(p.relative_to(b).parts)
            except Exception:
                rel = 99
            if rel > 3:
                continue
            if (p / "fovea_localization_train_GT.csv").is_file() and (p / "train").is_dir():
                cands.append(p)
    # 优先名字带 detection/task3 的
    cands.sort(key=lambda p: (0 if "detection" in str(p).lower() else 1, len(str(p))))
    if not cands:
        raise FileNotFoundError(
            "没找到任务三数据目录（需要同时含 train/ 与 fovea_localization_train_GT.csv）。"
            "检查右侧 Add Input 是否加了 2026task3，且比赛规则已接受。"
        )
    return cands[0]


def find_sample_sub(root):
    for p in [root / "sample_submission.csv"] + list(Path("/kaggle/input").rglob("sample_submission.csv")):
        if p.is_file():
            import csv
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows and rows[0][:1] == ["ImageID"] and len(rows) >= 41:
                return p, rows[1:]
    raise FileNotFoundError("没找到可用的 sample_submission.csv")


def load_train(root):
    """返回 [(fname, W, H, X, Y)]，按 GT 的 data 列升序。"""
    import csv
    gt = root / "fovea_localization_train_GT.csv"
    out = []
    with open(gt, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = str(r["data"]).strip()
            name = f"{int(sid):04d}.jpg"
            p = root / "train" / name
            if not p.is_file():
                # 兜底：也许带扩展名
                hit = list((root / "train").glob(f"{sid}*"))
                if not hit:
                    continue
                p = hit[0]
                name = p.name
            with Image.open(p) as im:
                W, H = im.size
            out.append((name, W, H, float(r["Fovea_X"]), float(r["Fovea_Y"])))
    out.sort(key=lambda t: t[0])
    return out


def load_test(root):
    ts = sorted([p for p in (root / "test").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    out = []
    for p in ts:
        with Image.open(p) as im:
            W, H = im.size
        out.append((p.name, p, W, H))
    return out


def make_folds(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(perm[i::k]).tolist() for i in range(k)]


def rmse_report(pred, true, tag=""):
    """pred/true: (N,2) 原图像素坐标。

    ★ 官方口径：提交的 40 个数是 20 张图的 X/Y 交替，所以
        RMSE = sqrt( mean( dX^2 + dY^2 ) / 2 )，里面的 mean 是**对 N 张图**取。
    这里曾经写成 np.mean(d ** 2)（d 是 (N,2) 数组），那是对 2N 个数取平均，
    再除 2 就变成「官方 RMSE / sqrt(2)」—— 日志里那些 RMSE 全部**偏小 0.707 倍**。
    对比两条管线时这个因子两边都有、不影响选择，但绝对值必须改对，
    否则「榜分≈」的预测和排行榜对不上。
    """
    d = np.asarray(pred, np.float64) - np.asarray(true, np.float64)
    dist = np.hypot(d[:, 0], d[:, 1])
    mse_xy = float(np.mean((d ** 2).sum(axis=1)))     # = mean(dist^2)，对 N 张图
    rmse = math.sqrt(mse_xy / 2.0)
    msg = (f"{tag} N={len(dist)} mean={dist.mean():.2f} median={np.median(dist):.2f} "
           f"p90={np.percentile(dist, 90):.2f} max={dist.max():.2f} | "
           f"RMSE={rmse:.3f} (榜分≈{rmse * 0.62:.2f}) | bias dX={d[:,0].mean():+.2f} dY={d[:,1].mean():+.2f}")
    log(msg)
    return rmse, msg


# ================= 视盘检测 + 结构先验 =================
# 用来否决「荒谬」的热图峰：解剖上黄斑必在视盘颞侧 590~780px、Y 约等于视盘水平中线。
# 上一版 CV 里 max=355px 那种灾难张，就是热图峰落到了视盘上，光靠模型自己救不回来；
# 但它们全都不满足上面的解剖约束，一条纯几何规则就能把它们筛出来改掉。
DISC_LONG = 768
VETO_MIN_DIST = 350.0      # 热图峰离视盘中心近于这个距离 = 不可能，判为认错目标


def _retina_mask(rgb):
    import scipy.ndimage as ndi
    m = rgb.max(2) > 0.12
    m = ndi.binary_closing(m, np.ones((7, 7)))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(m)


def _detect_disc(rgb, retina):
    """在长边 768 的缩略图上找视盘（亮黄白圆盘），返回 (cx, cy, R)。"""
    import scipy.ndimage as ndi
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
        best = None
        for k in range(1, n + 1):
            sl = objs[k - 1]
            if sl is None:
                continue
            comp = lab[sl] == k
            a = int(comp.sum())
            if not (0.0012 * a_ret <= a <= 0.06 * a_ret):
                continue
            hh, ww = comp.shape
            if a / (np.pi * (max(hh, ww) / 2.0) ** 2) < 0.35:
                continue
            ys, xs = np.nonzero(comp)
            cur = (float(L[sl][comp].mean()), a,
                   sl[1].start + float(xs.mean()), sl[0].start + float(ys.mean()),
                   float(np.sqrt(a / np.pi)))
            if best is None or cur[:2] > best[:2]:
                best = cur
        if best is not None:
            return best[2], best[3], best[4]
    return None


def disc_orig(path):
    """原图像素坐标下的 (cx, cy, R)；任何失败都返回 None（调用方会退化处理）。"""
    try:
        im = Image.open(path).convert("RGB")
        W, H = im.size
        s = DISC_LONG / float(max(W, H))
        sm = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
        rgb = np.asarray(sm).astype(np.float32) / 255.0
        d = _detect_disc(rgb, _retina_mask(rgb))
        if d is None:
            return None
        return d[0] / s, d[1] / s, d[2] / s
    except Exception:
        return None


def size_key(wh):
    return "S" if max(wh[0], wh[1]) < 2500 else "L"


def fit_geo(discs, true_xy, sizes, sel=None):
    """拟合「X = 视盘 cx ± D[尺寸组]，Y = Y[尺寸组]」。sel 给出时只用这些样本（留一法用）。"""
    idx = list(range(len(discs))) if sel is None else list(sel)
    ok = [i for i in idx if discs[i] is not None]
    if len(ok) < 3:
        return None
    D, Y = {}, {}
    for k in ("L", "S"):
        rows = [i for i in ok if size_key(sizes[i]) == k]
        if len(rows) < 3:
            rows = ok
        D[k] = float(np.median([abs(true_xy[i, 0] - discs[i][0]) for i in rows]))
        Y[k] = float(np.median([true_xy[i, 1] for i in rows]))
    return D, Y


def geo_point(disc, wh, geo):
    if disc is None or geo is None:
        return None
    D, Y = geo
    k = size_key(wh)
    side = -1.0 if disc[0] > wh[0] / 2.0 else 1.0     # 视盘在左半 -> 黄斑在右
    return np.array([disc[0] + side * D[k], Y[k]], np.float64)


def veto_fix(preds, discs, fallback, min_dist=VETO_MIN_DIST):
    """凡是热图峰落在视盘附近（< min_dist）的，换成几何估计；其余一个像素都不动。"""
    out = np.array(preds, np.float64).copy()
    n = 0
    for i in range(len(out)):
        if discs[i] is None or fallback[i] is None:
            continue
        if float(np.hypot(out[i, 0] - discs[i][0], out[i, 1] - discs[i][1])) < min_dist:
            out[i] = fallback[i]
            n += 1
    return out, n


def amp_ctx():
    if USE_AMP:
        try:
            return torch.autocast("cuda", dtype=torch.float16)
        except Exception:
            return torch.cuda.amp.autocast()
    import contextlib
    return contextlib.nullcontext()


def make_scaler():
    if not USE_AMP:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


# ============================== 图像缓存（letterbox） ==============================

class CanvasCache:
    """把每张图按长边缩放到 CANVAS 后 letterbox 到 CANVAS x CANVAS，uint8 常驻内存。"""

    def __init__(self, items, canvas):
        self.canvas = canvas
        self.arr = []
        self.scale = []
        self.ox = []
        self.oy = []
        for it in items:
            path = it["path"]
            with Image.open(path) as im:
                im = im.convert("RGB")
                W, H = im.size
                s = min(canvas / W, canvas / H)
                nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
                r = im.resize((nw, nh), Image.BILINEAR)
                cv = Image.new("RGB", (canvas, canvas), (0, 0, 0))
                ox, oy = (canvas - nw) // 2, (canvas - nh) // 2
                cv.paste(r, (ox, oy))
                self.arr.append(np.asarray(cv))
            self.scale.append(s)
            self.ox.append(ox)
            self.oy.append(oy)
        self.arr = np.stack(self.arr, 0)   # N,H,W,3
        log(f"[cache] canvas={canvas} n={len(self.arr)} 占用 {self.arr.nbytes/1e6:.1f} MB")

    def to_canvas(self, i, X, Y):
        return X * self.scale[i] + self.ox[i], Y * self.scale[i] + self.oy[i]

    def to_orig(self, i, xc, yc):
        return (xc - self.ox[i]) / self.scale[i], (yc - self.oy[i]) / self.scale[i]


# ============================== 增强（几何 + 光度） ==============================

def rand_geom(rng):
    """返回 (angle, scale, tx, ty)，tx/ty 是 canvas 尺寸的比例。"""
    return (
        float(rng.uniform(-12, 12)),
        float(rng.uniform(0.90, 1.10)),
        float(rng.uniform(-0.03, 0.03)),
        float(rng.uniform(-0.03, 0.03)),
    )


def warp_forward(H, W, angle, scale, tx, ty):
    """正向映射 q = A p + b（像素坐标，绕图中心旋转缩放再平移）。A = [[a,b],[d,e]]。"""
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    a, b = scale * c, -scale * s
    d, e = scale * s, scale * c
    bx = cx + tx * W - (a * cx + b * cy)
    by = cy + ty * H - (d * cx + e * cy)
    return a, b, d, e, bx, by


def apply_geom(img, angle, scale, tx, ty, flip):
    """img: CHW float tensor → 变换后的图。几何参数与 geom_label 完全一致（先 flip 后 affine）。"""
    _, H, W = img.shape
    if flip:
        img = torch.flip(img, dims=[2])
    if angle == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0:
        return img
    a, b, d, e, bx, by = warp_forward(H, W, angle, scale, tx, ty)
    det = a * e - b * d
    ia, ib = e / det, -b / det
    idn, ie = -d / det, a / det
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=img.device),
        torch.arange(W, dtype=torch.float32, device=img.device),
        indexing="ij",
    )
    X = (xs - bx).reshape(-1)
    Y = (ys - by).reshape(-1)
    PX = ia * X + ib * Y
    PY = idn * X + ie * Y
    gx = PX / (W - 1) * 2.0 - 1.0
    gy = PY / (H - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], -1).view(1, H, W, 2)
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)[0]


def geom_label(xc, yc, H, W, angle, scale, tx, ty, flip):
    """把点 (xc,yc) 用与 apply_geom 相同的正向映射送到新图坐标。"""
    if flip:
        xc = (W - 1) - xc
    a, b, d, e, bx, by = warp_forward(H, W, angle, scale, tx, ty)
    return a * xc + b * yc + bx, d * xc + e * yc + by


def photometric(img, rng):
    """img: CHW float [0,1] tensor."""
    if rng.random() < 0.8:
        img = img * float(rng.uniform(0.85, 1.15))
    if rng.random() < 0.8:
        m = img.mean()
        img = (img - m) * float(rng.uniform(0.8, 1.2)) + m
    if rng.random() < 0.5:
        g = float(rng.uniform(0.7, 1.4))
        img = torch.clamp(img, 1e-6, 1.0) ** g
    if rng.random() < 0.5:
        m = img.mean(dim=(1, 2), keepdim=True)
        img = (img - m) * float(rng.uniform(0.85, 1.15)) + m
    return torch.clamp(img, 0.0, 1.0)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize(img):
    # ★ MEAN/STD 必须跟着 img 的设备走：predict_* 里是先 .to(DEVICE) 再 normalize，
    #   在 CPU 上跑永远看不出问题，一到 GPU 就 RuntimeError（cuda:0 和 cpu 混用）。
    return (img - MEAN.to(img.device)) / STD.to(img.device)


# ============================== 热图工具 ==============================

def gauss_target(H, W, cx, cy, sigma, device):
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    gy = torch.exp(-((ys - cy) ** 2) / (2.0 * sigma * sigma))
    gx = torch.exp(-((xs - cx) ** 2) / (2.0 * sigma * sigma))
    return gy[:, None] * gx[None, :]


def soft_argmax(logits, T=1.0):
    """logits: B,1,H,W → (B,2) 网格坐标（格子中心 0.5, 1.5, ...）"""
    B, _, H, W = logits.shape
    p = torch.softmax((logits * T).flatten(1), dim=1).view(B, 1, H, W)
    ys = (torch.arange(H, device=logits.device, dtype=torch.float32) + 0.5).view(1, H, 1)
    xs = (torch.arange(W, device=logits.device, dtype=torch.float32) + 0.5).view(1, 1, W)
    py = (p * ys).sum(dim=(2, 3)).reshape(B)
    px = (p * xs).sum(dim=(2, 3)).reshape(B)
    return torch.stack([px, py], 1)


def peak_info(logits):
    """返回 (峰值 logit, 归一化熵)。熵越接近 1 说明热图越平 —— 用来做 (0,0) 预警。"""
    B, _, H, W = logits.shape
    z = logits.flatten(1)
    p = torch.softmax(z, dim=1)
    ent = -(p * torch.log(p + 1e-9)).sum(1) / math.log(H * W)
    return z.max(1).values, ent


# ============================== 第一阶段模型 ==============================

class FPNHead(nn.Module):
    def __init__(self, chans, strides, ch, out_stride):
        super().__init__()
        # 只取 stride >= 8 的层（最低层太贵、且 patch 太浅）
        keep_idx = [i for i, s in enumerate(strides) if s >= 8]
        if len(keep_idx) < 2:
            keep_idx = list(range(len(strides)))[-2:]
        self.keep_idx = keep_idx                     # ★ forward 按它挑 feats，必须与 lat 对齐
        self.strides = [strides[i] for i in keep_idx]
        self.base = min(self.strides)
        self.lat = nn.ModuleList([nn.Conv2d(chans[i], ch, 1) for i in keep_idx])
        self.out_stride = out_stride
        self.n_up = int(round(math.log2(max(1, out_stride and self.base // out_stride))))
        self.smooth = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1), nn.GroupNorm(8, ch // 2), nn.SiLU(inplace=True),
            nn.Conv2d(ch // 2, ch // 2, 3, padding=1), nn.GroupNorm(8, ch // 2), nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(ch // 2, 1, 1)

    def forward(self, feats):
        base = self.base
        outs = []
        for lat, i, s in zip(self.lat, self.keep_idx, self.strides):
            x = lat(feats[i])
            if s > base:
                x = F.interpolate(x, scale_factor=s / base, mode="bilinear", align_corners=False)
            outs.append(x)
        x = sum(outs)
        x = F.interpolate(x, scale_factor=base / self.out_stride, mode="bilinear", align_corners=False)
        x = self.smooth(x)
        return self.head(x)


class HeatNet(nn.Module):
    def __init__(self, bb_name, pretrained=True, ch=96, out_stride=8, drop_path=0.1):
        super().__init__()
        kw = dict(features_only=True, pretrained=pretrained)
        try:
            self.bb = timm.create_model(bb_name, drop_path_rate=drop_path, **kw)
        except TypeError:
            self.bb = timm.create_model(bb_name, **kw)
        fi = self.bb.feature_info
        chans = list(fi.channels())
        strides = list(fi.reduction())
        self.out_stride = out_stride
        self.neck = FPNHead(chans, strides, ch, out_stride)
        self.bb_name = bb_name

    def forward(self, x):
        return self.neck(self.bb(x))


# ============================== 第一阶段数据集 ==============================

class S1Dataset(Dataset):
    def __init__(self, cache, idx, xy_canvas, train=True, sigma=2.0, stride=8):
        self.cache = cache
        self.idx = idx
        self.xy = xy_canvas
        self.train = train
        self.sigma = sigma
        self.stride = stride

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        H = W = self.cache.canvas
        img = torch.from_numpy(self.cache.arr[i]).permute(2, 0, 1).float() / 255.0
        xc, yc = self.xy[k]
        if self.train:
            rng = np.random
            ang, sc, tx, ty = rand_geom(rng)
            flip = bool(rng.random() < 0.5)
            img = apply_geom(img, ang, sc, tx, ty, flip)
            xc, yc = geom_label(xc, yc, H, W, ang, sc, tx, ty, flip)
            img = photometric(img, rng)
        img = normalize(img)
        gh, gw = H // self.stride, W // self.stride
        gx, gy = xc / self.stride, yc / self.stride
        tgt = gauss_target(gh, gw, gx, gy, self.sigma, img.device).unsqueeze(0)
        return img, tgt, torch.tensor([gx / gw, gy / gh], dtype=torch.float32)


def train_s1(cfg, tr_idx, va_idx, cache, xy_canvas, epochs, tag):
    m = HeatNet(cfg["name"], cfg["pretrained"], S1_CH, S1_STRIDE, S1_DROP_PATH).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=S1_LR, weight_decay=S1_WD)
    steps = max(1, math.ceil(len(tr_idx) / S1_BATCH)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=S1_LR, total_steps=steps, pct_start=0.25)
    scaler = make_scaler()
    dtr = S1Dataset(cache, tr_idx, [xy_canvas[i] for i in tr_idx], True, S1_SIGMA, S1_STRIDE)
    ltr = DataLoader(dtr, batch_size=S1_BATCH, shuffle=True, num_workers=0, drop_last=len(dtr) > S1_BATCH)
    hist = []
    for ep in range(epochs):
        m.train()
        tot, n = 0.0, 0
        for img, tgt, xy in ltr:
            img, tgt, xy = img.to(DEVICE), tgt.to(DEVICE), xy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                out = m(img)
                lhm = F.binary_cross_entropy_with_logits(out, tgt)
                # ★ xy 是「归一化到 0~1」的标签，soft_argmax 给的是「格坐标」，
                #   必须先把标签换成格坐标再比，否则两者量级差 30 倍，
                #   优化器会靠「把热图收缩到左上角」来降低这个虚假误差（曾经的真 bug）。
                gh, gw = out.shape[-2], out.shape[-1]
                sp = soft_argmax(out, T=4.0)
                tt = torch.stack([xy[:, 0] * gw, xy[:, 1] * gh], 1)
                lxy = F.smooth_l1_loss(sp, tt, beta=1.0)
                loss = lhm + S1_WXY * lxy
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += loss.item() * img.size(0)
            n += img.size(0)
        hist.append(tot / max(1, n))
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  {tag} ep{ep+1}/{epochs} loss={tot/max(1,n):.4f}")
    return m, hist


def predict_s1(models, cache, idx, flip_tta=True):
    """返回 ((N,2) 原图像素预测的累加值, 诊断字典)。

    诊断里 dis = 翻转 TTA 两个分支之间的分歧（格单位，越大越不稳定）。
    这是唯一一个不依赖校准、能直接指出「模型自己没底」的信号 —— 实测 S1 的
    4 张图吃掉 87.6% 的平方误差，就需要这种信号去把它们挑出来。
    """
    N = len(idx)
    acc = np.zeros((N, 2), np.float64)
    dis = np.zeros(N, np.float64)
    ent = np.zeros(N, np.float64)
    pk = np.zeros(N, np.float64)
    for m, cfg in models:
        m.eval()
        for start in range(0, N, 8):
            sel = idx[start:start + 8]
            x = torch.stack([torch.from_numpy(cache.arr[i]).permute(2, 0, 1).float() / 255.0 for i in sel]).to(DEVICE)
            x = normalize(x)
            with torch.no_grad(), amp_ctx():
                o1 = m(x)
                if flip_tta:
                    o2 = torch.flip(m(torch.flip(x, dims=[3])), dims=[3])
                else:
                    o2 = o1
                o = (o1 + o2).float()
                sp = soft_argmax(o, T=4.0)
                with torch.no_grad():
                    dd = (soft_argmax(o1, T=4.0) - soft_argmax(o2, T=4.0)).pow(2).sum(1).sqrt()
                pp, ee = peak_info(o)
            sp = sp.cpu().numpy()
            dd = dd.cpu().numpy()
            pp = pp.cpu().numpy()
            ee = ee.cpu().numpy()
            for b in range(len(sel)):
                i = sel[b]
                acc[start + b, 0], acc[start + b, 1] = cache.to_orig(
                    i, sp[b, 0] * S1_STRIDE, sp[b, 1] * S1_STRIDE)
                dis[start + b] += float(dd[b])      # 按下标（sel 可能是折内子集）
                pk[start + b] += float(pp[b])
                ent[start + b] += float(ee[b])
    return acc, {"dis": dis, "peak": pk, "ent": ent}


# ============================== 第二阶段模型 ==============================

class Refiner(nn.Module):
    def __init__(self, w=(32, 64, 96, 128), out_stride=4):
        super().__init__()
        self.out_stride = int(out_stride)

        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(inplace=True))
        self.e1 = nn.Sequential(blk(3, w[0]), blk(w[0], w[0]))
        self.e2 = nn.Sequential(nn.MaxPool2d(2), blk(w[0], w[1]), blk(w[1], w[1]))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), blk(w[1], w[2]), blk(w[2], w[2]))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), blk(w[2], w[3]), blk(w[3], w[3]))
        self.d3 = nn.Sequential(blk(w[3] + w[2], w[2]), blk(w[2], w[2]))
        self.d2 = nn.Sequential(blk(w[2] + w[1], w[1]), blk(w[1], w[1]))
        self.d1 = nn.Sequential(blk(w[1] + w[0], w[0]), blk(w[0], w[0]))
        # ★ 输出热图必须是 NET//out_stride 的网格（目标高斯就是按这个尺寸算的）
        self.head = nn.Conv2d(w[0], 1, 3, stride=self.out_stride, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        u3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], 1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], 1))
        return self.head(d1)


class ROICache:
    """每张图按给定先验缓存一个 (2*half)^2 的原分辨率方块。"""

    def __init__(self, paths, centers, half):
        self.half = half
        self.arr = []
        for p, (cx, cy) in zip(paths, centers):
            with Image.open(p) as im:
                a = np.asarray(im.convert("RGB"))
            H, W = a.shape[:2]
            out = np.zeros((2 * half, 2 * half, 3), np.uint8)
            x0 = int(round(cx)) - half
            y0 = int(round(cy)) - half
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(W, x0 + 2 * half), min(H, y0 + 2 * half)
            if sx1 > sx0 and sy1 > sy0:
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
            self.arr.append(out)
        self.arr = np.stack(self.arr, 0)
        log(f"[roi] half={half} n={len(self.arr)} 占用 {self.arr.nbytes/1e6:.1f} MB")

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.arr[i]


def crop_patch(roi, ddx, ddy, crop):
    """从以先验为中心的 ROI 里取一块以 (ddx,ddy) 为中心的 crop x crop。越界补黑。"""
    half = roi.shape[0] // 2
    ox = int(round(half + ddx - crop / 2.0))
    oy = int(round(half + ddy - crop / 2.0))
    H, W = roi.shape[:2]
    out = np.zeros((crop, crop, 3), np.uint8)
    sx0, sy0 = max(0, ox), max(0, oy)
    sx1, sy1 = min(W, ox + crop), min(H, oy + crop)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = roi[sy0:sy1, sx0:sx1]
    return out, ox, oy


class S2Dataset(Dataset):
    def __init__(self, roi, idx, off_true, train=True):
        self.roi = roi
        self.idx = idx
        self.off = off_true          # (N,2) 真值 - 先验，原图像素
        self.train = train
        self.G = S2_NET // S2_STRIDE

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        dx, dy = self.off[k]
        rng = np.random
        if self.train:
            for _ in range(6):
                cx = float(rng.uniform(-S2_JIT, S2_JIT))
                cy = float(rng.uniform(-S2_JIT, S2_JIT))
                if math.hypot(dx - cx, dy - cy) <= S2_GUARD:
                    break
        else:
            cx, cy = 0.0, 0.0
        patch, _, _ = crop_patch(self.roi[i], cx, cy, S2_CROP)
        img = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        flip = False
        if self.train:
            flip = bool(rng.random() < 0.5)
            if flip:
                img = torch.flip(img, dims=[2])
            if S2_CROP != S2_NET:
                img = F.interpolate(img.unsqueeze(0), size=(S2_NET, S2_NET), mode="area")[0]
            img = photometric(img, rng)
        else:
            if S2_CROP != S2_NET:
                img = F.interpolate(img.unsqueeze(0), size=(S2_NET, S2_NET), mode="area")[0]
        img = normalize(img)
        # 相对裁块中心的偏移（★ 水平翻转时 x 偏移必须一起取负，否则标签镜像错）
        odx, ody = dx - cx, dy - cy
        if flip:
            odx = -odx
        g = self.G
        gx = (odx + S2_CROP / 2.0 - S2_STRIDE / 2.0) / S2_STRIDE
        gy = (ody + S2_CROP / 2.0 - S2_STRIDE / 2.0) / S2_STRIDE
        tgt = gauss_target(g, g, gx, gy, S2_SIGMA, torch.device("cpu")).unsqueeze(0)
        return img, tgt, torch.tensor([gx, gy], dtype=torch.float32)


def train_s2(roi, tr_idx, va_idx, off_true, epochs, tag):
    m = Refiner(out_stride=S2_STRIDE).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=S2_LR, weight_decay=S2_WD)
    steps = max(1, math.ceil(len(tr_idx) / S2_BATCH)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=S2_LR, total_steps=steps, pct_start=0.25)
    scaler = make_scaler()
    dtr = S2Dataset(roi, tr_idx, off_true, True)
    dva = S2Dataset(roi, va_idx, off_true, False)
    ltr = DataLoader(dtr, batch_size=S2_BATCH, shuffle=True, num_workers=0, drop_last=len(dtr) > S2_BATCH)
    lva = DataLoader(dva, batch_size=max(1, S2_BATCH), shuffle=False, num_workers=0)
    for ep in range(epochs):
        m.train()
        tot, n = 0.0, 0
        for img, tgt, xy in ltr:
            img, tgt, xy = img.to(DEVICE), tgt.to(DEVICE), xy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                out = m(img)
                lhm = F.mse_loss(torch.sigmoid(out), tgt)
                sp = soft_argmax(out, T=4.0)
                lxy = F.l1_loss(sp, xy)
                loss = lhm + S2_WXY * lxy
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += loss.item() * img.size(0)
            n += img.size(0)
        if ep % 20 == 0 or ep == epochs - 1:
            log(f"  {tag} ep{ep+1}/{epochs} loss={tot/max(1,n):.4f}")
    return m


def predict_s2(models, roi, idx, priors, flip_tta=True):
    """返回到原图的最终坐标 (N,2)。"""
    N = len(idx)
    acc = np.zeros((N, 2), np.float64)
    for m in models:
        m.eval()
        for k, i in enumerate(idx):
            patch, _, _ = crop_patch(roi[i], 0.0, 0.0, S2_CROP)
            img = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            if S2_CROP != S2_NET:
                img = F.interpolate(img.unsqueeze(0), size=(S2_NET, S2_NET), mode="area")[0]
            img = normalize(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), amp_ctx():
                o = m(img)
                if flip_tta:
                    o2 = torch.flip(m(torch.flip(img, dims=[3])), dims=[3])
                    o = o + o2
                o = o.float()
            sp = soft_argmax(o, T=4.0).cpu().numpy()[0]
            odx = sp[0] * S2_STRIDE - (S2_CROP / 2.0 - S2_STRIDE / 2.0)
            ody = sp[1] * S2_STRIDE - (S2_CROP / 2.0 - S2_STRIDE / 2.0)
            acc[k, 0] = priors[i][0] + odx
            acc[k, 1] = priors[i][1] + ody
    return acc / max(1, len(models))


# ============================== 提交 ==============================

def write_submission(sample_rows, pred_by_id, out_path):
    """pred_by_id: {'0081': (X,Y), ...} 或 {81: (X,Y)}"""
    lines = ["ImageID,value"]
    n = 0
    n_x = 0
    for r in sample_rows:
        sid = r[0]
        # ★ ImageID 形如 "81_Fovea_X"：split("_") 是 ['81','Fovea','X']，
        #   所以必须看最后一段（曾经误写成 parts[1]，导致 X/Y 两行写了同一个值，
        #   整份提交静默作废 —— 这个坑太深，下面加了硬断言）。
        is_x = sid.strip().lower().endswith("_x")
        num = sid.split("_")[0]
        key = num if num in pred_by_id else int(num)
        X, Y = pred_by_id[key]
        lines.append(f"{sid},{X if is_x else Y:.6f}")
        n += 1
        n_x += int(is_x)
    assert n_x * 2 == n, f"X/Y 行数不成对: X={n_x} 总={n}"
    data = "\r\n".join(lines) + "\r\n"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(data)
    raw = Path(out_path).read_bytes()
    n_crlf = raw.count(b"\r\n")
    n_lf = raw.count(b"\n")
    log(f"[submit] 写入 {out_path} 行数={n} bytes={len(raw)} CRLF={n_crlf} "
        f"裸LF={n_lf - n_crlf} BOM={raw[:3] == b'\xef\xbb\xbf'}")
    assert n == 2 * len(pred_by_id), f"行数不对: {n} vs {2*len(pred_by_id)}"
    return out_path


def dump_submission(sample_rows, te, pts, out_csv, tag=""):
    """把 (N,2) 的测试集预测写成提交。任何时刻都能调用 —— 每折结束后写一次，
    这样哪怕后面某一折崩了，Output 里也一定有一份可交的文件（任务二的血教训）。"""
    pred_by_id = {}
    for k, (fname, path, W, H) in enumerate(te):
        num = int(Path(fname).stem)
        pred_by_id[num] = (float(np.clip(pts[k, 0], 0.0, W - 1.0)),
                           float(np.clip(pts[k, 1], 0.0, H - 1.0)))
    write_submission(sample_rows, pred_by_id, out_csv)
    if tag:
        log(f"  [早存] {tag} 版本的 submission.csv 已落盘（保证 Output 不为空）")
    return pred_by_id


def check_submission(path, sample_path):
    a = Path(path).read_bytes()
    b = Path(sample_path).read_bytes()
    ok = True
    problems = []
    if a[:3] == b"\xef\xbb\xbf":
        problems.append("有 BOM")
    if not a.endswith(b"\r\n"):
        problems.append("结尾不是 CRLF")
    if b"\n" in a.replace(b"\r\n", b""):
        problems.append("存在裸 LF")
    la = [x for x in a.decode("utf-8").split("\r\n") if x]
    lb = [x for x in b.decode("utf-8").split("\r\n") if x]
    if len(la) != len(lb):
        problems.append(f"行数 {len(la)} != 样例 {len(lb)}")
    if la[0] != "ImageID,value":
        problems.append(f"表头 {la[0]!r}")
    for i in range(1, min(len(la), len(lb))):
        if la[i].split(",")[0] != lb[i].split(",")[0]:
            problems.append(f"第 {i+1} 行 ImageID 顺序不符: {la[i]} vs {lb[i]}")
            break
    for i in range(1, len(la)):
        v = la[i].split(",")
        if len(v) != 2:
            problems.append(f"第 {i+1} 行列数 {len(v)}")
            break
        try:
            float(v[1])
        except Exception:
            problems.append(f"第 {i+1} 行不是数字: {la[i]}")
            break
    if problems:
        ok = False
        for p in problems:
            log("  [X]", p)
    log(f"[check] {'通过（0 错误 0 警告）' if ok else '有问题！'}")
    return ok


# ============================== 主流程 ==============================

def main():
    set_seed(SEED)
    log(f"device={DEVICE} torch={torch.__version__} timm={timm.__version__}")
    if DEVICE.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    log(f"budget={TIME_BUDGET_MIN} min  N_FOLDS={N_FOLDS}")

    root = find_root()
    log(f"DATA_ROOT = {root}")
    sample_path, sample_rows = find_sample_sub(root)
    log(f"sample_submission = {sample_path}  rows={len(sample_rows)}")
    out_csv = OUT / "submission.csv"

    tr = load_train(root)
    te = load_test(root)
    log(f"train={len(tr)}  test={len(te)}")
    assert len(tr) >= 20, "训练图太少，检查 GT 与 train/ 是否匹配"
    assert len(te) == len(sample_rows) // 2, f"测试图 {len(te)} 与样例 {len(sample_rows)//2} 不符"

    names = [t[0] for t in tr]
    sizes = [(t[1], t[2]) for t in tr]
    true_xy = np.array([[t[3], t[4]] for t in tr], np.float64)

    # ---- 全图缓存（第一阶段用；第二阶段复用同一批原图路径）----
    items = [{"path": root / "train" / t[0]} for t in tr]
    c1 = CanvasCache(items, S1_CANVAS)
    xy_canvas = [c1.to_canvas(i, true_xy[i, 0], true_xy[i, 1]) for i in range(len(tr))]

    folds = make_folds(len(tr), N_FOLDS, SEED)
    log(f"folds = {[len(f) for f in folds]}")

    # ---- ★ 兜底提交：训练开始前先写一份「按尺寸分组的 GT 中位数」----
    # 这样哪怕第一折就崩，Output 里也一定有一份合法的 submission.csv，
    # 而不是像上次那样 21 分钟白烧、Output 0 B、什么都没留下。
    try:
        big_te = np.array([max(W, H) for (_, _, W, H) in te]) >= 2500
        big_tr = np.array([max(t[1], t[2]) for t in tr]) >= 2500
        fb = np.zeros((len(te), 2), np.float64)
        for flag in (True, False):
            src = true_xy[big_tr == flag]
            if len(src) == 0:
                src = true_xy
            fb[big_te == flag] = np.median(src, axis=0)
        dump_submission(sample_rows, te, fb, out_csv, "median-GT 兜底")
    except Exception as e:
        log(f"  [兜底] 跳过：{type(e).__name__}: {e}")

    # ================= 第一阶段 =================
    s1_models = []          # [(model, cfg)]
    s1_oof_sum = np.zeros((len(tr), 2), np.float64)
    s1_oof_cnt = np.zeros(len(tr), np.float64)
    s1_dis = np.zeros(len(tr), np.float64)        # 翻转 TTA 分歧累加（每图）
    s1_test_dis = np.zeros(len(te), np.float64)
    s1_test_sum = None
    s1_test_n = 0

    te_items = [{"path": p} for (_, p, _, _) in te]
    c1_te = CanvasCache(te_items, S1_CANVAS)

    cfgs = []
    for bb in S1_BACKBONES:
        for sd in S1_SEEDS:
            cfgs.append({"name": bb, "seed": sd, "pretrained": not SMOKE})

    for ci, cfg in enumerate(cfgs):
        log(f"[S1 {ci+1}/{len(cfgs)}] backbone={cfg['name']} seed={cfg['seed']}")
        if PREFLIGHT:
            try:
                tst = HeatNet(cfg["name"], cfg["pretrained"], S1_CH, S1_STRIDE, S1_DROP_PATH)
                tst.eval()
                with torch.no_grad():
                    tst(torch.zeros(1, 3, S1_CANVAS, S1_CANVAS))
                log(f"  [ok] 尺寸预检通过 {S1_CANVAS}")
                del tst
            except Exception as e:
                log(f"  [--] {cfg['name']} 建不起来，跳过：{type(e).__name__}: {e}")
                continue
        for fi, va in enumerate(folds):
            if left_min() < RESERVE_MIN + 0.5:
                log(f"  ! 预算只剩 {left_min():.1f} min，停止第一阶段")
                break
            folds_left = (len(cfgs) - ci) * N_FOLDS - fi
            ep_i = plan_epochs("s1", folds_left, S1_EPOCHS)
            tr_idx = [i for i in range(len(tr)) if i not in set(va)]
            set_seed(cfg["seed"] * 1000 + fi)
            tag = f"S1[{cfg['name'].split('.')[0]} s{cfg['seed']} f{fi}]"
            t_fold = time.time()
            try:
                m, hist = train_s1(cfg, tr_idx, va, c1, xy_canvas, ep_i, tag)
            except Exception as e:
                log(f"  [!!] {tag} 抛异常，跳过：{type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            # OOF
            pr, di = predict_s1([(m, cfg)], c1, va, True)
            for k, i in enumerate(va):
                s1_oof_sum[i] += pr[k]
                s1_oof_cnt[i] += 1
                s1_dis[i] += float(di["dis"][k])
            # test
            pt, dt = predict_s1([(m, cfg)], c1_te, list(range(len(te))), True)
            if s1_test_sum is None:
                s1_test_sum = np.zeros_like(pt)
            for k in range(len(te)):
                s1_test_dis[k] += float(dt["dis"][k])
            s1_test_sum += pt
            s1_test_n += 1
            s1_models.append((m, cfg))
            torch.save({"sd": m.state_dict(), "cfg": cfg}, OUT / f"s1_{ci}_{fi}.pt")
            try:
                dump_submission(sample_rows, te, s1_test_sum / s1_test_n, out_csv,
                                f"S1 {cfg['name'].split('.')[0]} f{fi}")
            except Exception as e:
                log(f"  [早存] 失败（不影响训练）：{type(e).__name__}: {e}")
            fold_min = (time.time() - t_fold) / 60.0
            _EST_EPOCH["s1"] = fold_min / max(1, ep_i)
            log(f"  [{tag}] {ep_i} epochs, 用时 {fold_min:.1f} min "
                f"(每 epoch {fold_min*60/max(1,ep_i):.1f}s)  累计模型 {len(s1_models)}")

    prior_oof = np.where(s1_oof_cnt[:, None] > 0,
                         s1_oof_sum / np.maximum(1, s1_oof_cnt)[:, None],
                         true_xy.mean(0)[None, :].repeat(len(tr), 0))
    if s1_test_sum is not None:
        te_prior = s1_test_sum / max(1, s1_test_n)
    else:
        # 兜底：按尺寸分组的中位 GT（本地实测常数点 RMSE≈145，只为不空手）
        te_prior = np.zeros((len(te), 2), np.float64)
        for k, (fn, p, W, H) in enumerate(te):
            same = [i for i, t in enumerate(tr) if (t[1], t[2]) == (W, H)]
            te_prior[k] = np.median(true_xy[same] if same else true_xy, 0)

    s1_rmse, _ = rmse_report(prior_oof, true_xy, "[S1 留一法先验]")

    # ---- 逐图诊断：RMSE 被少数离群张主导，先看清是哪几张 ----
    try:
        _d = prior_oof - true_xy
        _dist = np.hypot(_d[:, 0], _d[:, 1])
        _ord = np.argsort(_dist)[::-1]
        log("[S1 最差 8 张] " + "  ".join(
            f"{tr[i][0]}:{_dist[i]:.0f}px(dX{_d[i,0]:+.0f},dY{_d[i,1]:+.0f})" for i in _ord[:8]))
        _sh = float((_dist[_ord[:4]] ** 2).sum() / max(1e-9, (_dist ** 2).sum()))
        log(f"[S1 最差 4 张合计占全部平方误差 {_sh * 100:.1f}%]  均值 {_dist.mean():.2f} 中位 {np.median(_dist):.2f}")
    except Exception as e:
        log(f"  [逐图诊断] 跳过：{type(e).__name__}: {e}")

    # ---- 第一阶段的测试集预测单独存一份，方便和级联版各交一次做真实对比 ----
    try:
        dump_submission(sample_rows, te, te_prior, OUT / "submission_s1.csv", "S1-only 备选")
    except Exception as e:
        log(f"  [S1-only] 失败：{type(e).__name__}: {e}")

    # ---- 视盘检测 + 结构先验：用来否决「荒谬」的热图峰 ----
    cand = {}
    fallback_tr = fallback_te = None
    s1_veto_rmse = None
    te_prior_veto = None
    try:
        _t0 = time.time()
        discs_tr = [disc_orig(root / "train" / t[0]) for t in tr]
        discs_te = [disc_orig(p) for (_, p, _, _) in te]
        log(f"[视盘] 检出 训练 {sum(d is not None for d in discs_tr)}/{len(tr)}"
            f"  测试 {sum(d is not None for d in discs_te)}/{len(te)}"
            f"  用时 {(time.time() - _t0) / 60:.1f} min")
        geo_full = fit_geo(discs_tr, true_xy, sizes)
        if geo_full is not None:
            _dd = {k: round(v) for k, v in geo_full[0].items()}
            _yy = {k: round(v) for k, v in geo_full[1].items()}
            log(f"[几何先验] 视盘->黄斑水平距离 D={_dd}  Y={_yy}")
        # 训练集用留一法拟合（每张的 D/Y 由另外 79 张得出），测试集用全量拟合
        fallback_tr = []
        for i in range(len(tr)):
            g = fit_geo(discs_tr, true_xy, sizes,
                        sel=[j for j in range(len(tr)) if j != i])
            fallback_tr.append(geo_point(discs_tr[i], sizes[i], g))
        fallback_te = [geo_point(discs_te[i], (te[i][2], te[i][3]), geo_full)
                       for i in range(len(te))]
        pv, nfix = veto_fix(prior_oof, discs_tr, fallback_tr)
        s1_veto_rmse, _ = rmse_report(pv, true_xy, "[B: 第一阶段+否决]")
        te_prior_veto, nfix_te = veto_fix(te_prior, discs_te, fallback_te)
        log(f"[B] 否决改写了 训练 {nfix}/{len(tr)} 张，测试 {nfix_te}/{len(te)} 张")
    except Exception as e:
        log(f"  [视盘] 跳过（不影响主流程）：{type(e).__name__}: {e}")
        fallback_tr = fallback_te = None
        s1_veto_rmse = None

    # ---- 稳健化：模型自己「翻个面就变了」的图，往几何先验拉 ----
    # S1 的 4 张图吃掉 87.6% 的平方误差，而这几张的共同点是翻转 TTA 两个分支分歧大。
    # 阈值/权重在下面这张表里扫（q × w 共 8 个），按留一法挑 —— 不在测试集上挑。
    s1_robust_rmse = None
    te_prior_robust = None
    dis_tr = dis_te = None
    try:
        dis_tr = s1_dis / np.maximum(1, s1_oof_cnt)
        dis_te = s1_test_dis / max(1, s1_test_n)
        F = None
        if fallback_tr is not None:
            F = np.array([p if p is not None else [np.nan, np.nan] for p in fallback_tr], np.float64)
            FT = np.array([p if p is not None else [np.nan, np.nan] for p in fallback_te], np.float64)
            log(f"[C] 翻转分歧分布：p50={np.percentile(dis_tr, 50):.2f} "
                f"p80={np.percentile(dis_tr, 80):.2f} p95={np.percentile(dis_tr, 95):.2f} "
                f"max={dis_tr.max():.2f} （格单位）")
            log("=== 稳健化扫描（不稳的图往几何先验拉）===")
            _best = None
            for q in (80, 85, 90, 95):
                thr = float(np.percentile(dis_tr, q))
                for w in (0.5, 1.0):
                    p2 = prior_oof.copy()
                    m2 = (dis_tr > thr) & ~np.isnan(F[:, 0])
                    p2[m2] = (1.0 - w) * prior_oof[m2] + w * F[m2]
                    r2, _ = rmse_report(p2, true_xy, f"[C q={q} w={w} 改写{int(m2.sum())}张]")
                    if _best is None or r2 < _best[0]:
                        _best = (r2, q, w, thr)
            s1_robust_rmse = _best[0]
            thr, w = _best[3], _best[2]
            m_tr = (dis_tr > thr) & ~np.isnan(F[:, 0])
            s1_robust_oof = prior_oof.copy()
            s1_robust_oof[m_tr] = (1.0 - w) * prior_oof[m_tr] + w * F[m_tr]
            te_prior_robust = te_prior.copy()
            m_te = (dis_te > thr) & ~np.isnan(FT[:, 0])
            te_prior_robust[m_te] = (1.0 - w) * te_prior[m_te] + w * FT[m_te]
            log(f"[C] 选中 q={_best[1]} w={w}（阈值 {thr:.2f} 格）："
                f"训练改写 {int(m_tr.sum())}/{len(tr)} 张，测试改写 {int(m_te.sum())}/{len(te)} 张，"
                f"RMSE={s1_robust_rmse:.3f}")

            # ★ 诚实检验：在另一半上挑 (q,w)，在这一半上评。两边都赢才留用。
            #   上面那个 8 选 1 的全集最小值是 in-sample 的，任务二就是在这上面翻过船。
            def _rm_on(p, sel):
                dd = np.linalg.norm(p[sel] - true_xy[sel], axis=1)
                return float(np.sqrt(np.mean(dd ** 2) / 2.0))

            def _fit_rule(sel):
                bl = None
                for q in (80, 85, 90, 95):
                    th = float(np.percentile(dis_tr[sel], q))
                    for ww in (0.5, 1.0):
                        p = prior_oof.copy()
                        mm = (dis_tr > th) & ~np.isnan(F[:, 0])
                        p[mm] = (1.0 - ww) * prior_oof[mm] + ww * F[mm]
                        rr = _rm_on(p, sel)
                        if bl is None or rr < bl[0]:
                            bl = (rr, q, ww, th)
                return bl

            _rng = np.random.default_rng(7)
            _perm = _rng.permutation(len(tr))
            _A = np.zeros(len(tr), bool)
            _A[_perm[: len(tr) // 2]] = True
            _gains = []
            for _fit_sel, _ev_sel in ((_A, ~_A), (~_A, _A)):
                _bl = _fit_rule(_fit_sel)
                _p = prior_oof.copy()
                _mm = (dis_tr > _bl[3]) & ~np.isnan(F[:, 0])
                _p[_mm] = (1.0 - _bl[2]) * prior_oof[_mm] + _bl[2] * F[_mm]
                _g = _rm_on(prior_oof, _ev_sel) - _rm_on(_p, _ev_sel)
                _gains.append(_g)
                log(f"  [C 留出检验] 在另一半上挑 q={_bl[1]} w={_bl[2]} → "
                    f"这一半 RMSE 改善 {_g:+.3f}")
            if min(_gains) > 0.0:
                log(f"[C] 两半都赢（最差 {min(_gains):+.2f}）⇒ 规则可信，保留 s1_robust")
            else:
                log(f"[C] 有一半变差（最差 {min(_gains):+.2f}）⇒ 判为过拟合，丢弃 s1_robust")
                s1_robust_rmse = None
                te_prior_robust = None
    except Exception as e:
        log(f"  [C] 跳过：{type(e).__name__}: {e}")
        s1_robust_rmse = None
        te_prior_robust = None

    # ================= 第二阶段 =================
    # 只用先验在训练图上的 OOF 值当裁块中心（无标签泄漏）
    tr_paths = [root / "train" / t[0] for t in tr]
    roi_tr = ROICache(tr_paths, prior_oof, S2_ROI_HALF)
    off_true = true_xy - prior_oof
    roi_te = ROICache([p for (_, p, _, _) in te], te_prior, S2_ROI_HALF)

    s2_oof = np.zeros_like(true_xy)
    s2_cnt = np.zeros(len(tr), np.float64)
    final_sum = None
    final_n = 0

    for si, sd in enumerate(S2_SEEDS):
        log(f"[S2 {si+1}/{len(S2_SEEDS)}] seed={sd}")
        for fi, va in enumerate(folds):
            if left_min() < RESERVE_MIN + 0.4:
                log(f"  ! 预算只剩 {left_min():.1f} min，停止第二阶段")
                break
            folds_left = (len(S2_SEEDS) - si) * N_FOLDS - fi
            ep_i = plan_epochs("s2", folds_left, S2_EPOCHS)
            tr_idx = [i for i in range(len(tr)) if i not in set(va)]
            set_seed(sd * 1000 + fi + 7)
            tag = f"S2[s{sd} f{fi}]"
            t_fold = time.time()
            try:
                m = train_s2(roi_tr, tr_idx, va, off_true, ep_i, tag)
            except Exception as e:
                log(f"  [!!] {tag} 抛异常，跳过：{type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            pr = predict_s2([m], roi_tr, va, prior_oof, True)
            for k, i in enumerate(va):
                s2_oof[i] += pr[k]
                s2_cnt[i] += 1
            pt = predict_s2([m], roi_te, list(range(len(te))), te_prior, True)
            if final_sum is None:
                final_sum = np.zeros_like(pt)
            final_sum += pt
            final_n += 1
            torch.save({"sd": m.state_dict()}, OUT / f"s2_{si}_{fi}.pt")
            fold_min = (time.time() - t_fold) / 60.0
            _EST_EPOCH["s2"] = fold_min / max(1, ep_i)
            log(f"  [{tag}] {ep_i} epochs, 用时 {fold_min:.1f} min "
                f"(每 epoch {fold_min*60/max(1,ep_i):.1f}s)  累计模型 {final_n}")
            try:
                dump_submission(sample_rows, te, final_sum / max(1, final_n), out_csv,
                                f"S2 {tag} 累计 {final_n} 个模型")
            except Exception as e:
                log(f"  [早存] 失败（不影响训练）：{type(e).__name__}: {e}")

    # ================= 评估 + 出提交 =================
    # 已知最好成绩的参照线：本地 t3_run_refine_full2 的 OOF RMSE=28.72 → 榜 27.53739
    REF_RMSE = 34.40   # 已交并拿到榜 14.45894 的那一版（S1-only）的留一法 RMSE，同尺度对比
    oof_rmse = float("inf")
    if final_n > 0:
        s2_pred = s2_oof / np.maximum(1, s2_cnt)[:, None]
        oof_rmse, _ = rmse_report(s2_pred, true_xy, "[S2 级联留一法]")
        test_pred = final_sum / final_n
        how = f"two-stage ({len(s1_models)} 个第一阶段 + {final_n} 个第二阶段)"
    else:
        log("!! 第二阶段一个模型都没训成，退回第一阶段热图直接当结果")
        test_pred = te_prior
        how = f"stage1 only ({len(s1_models)} 个模型)"

    # ★ 两条管线都算过留一法 OOF，就按 OOF 选 —— 这是纯 CV 选择，不碰测试集标签。
    #   上次 Kaggle 跑出来的事实：S1 单独 25.19，级联 26.15，即第二阶段反而在拖后腿
    #   （先验中位数已经 6.78px，而细化器是按 ±100px 抖动训练的，只会加噪声）。
    log(f"两条管线留一法对比： S1-only={s1_rmse:.3f}   级联={oof_rmse if oof_rmse < 1e9 else float('nan'):.3f}")
    if final_n > 0 and s1_rmse < oof_rmse:
        log(f"⇒ S1-only 更好（{s1_rmse:.2f} < {oof_rmse:.2f}），提交改用第一阶段结果")
        test_pred = te_prior
        how = f"stage1 only ({len(s1_models)} 个第一阶段模型，按 OOF 优于级联而选中)"
    else:
        log(f"⇒ 级联更好（{oof_rmse:.2f} <= {s1_rmse:.2f}），提交用级联结果")

    log(f"最终方案 = {how}")
    best_rmse = min(s1_rmse, oof_rmse) if final_n > 0 else s1_rmse
    if best_rmse < REF_RMSE:
        log(f">>> 交不交？采用管线的留一法 RMSE={best_rmse:.2f} 好于已知最好的 {REF_RMSE}"
            f"（榜 27.53739）⇒ **值得提交**")
    elif best_rmse < 1e9:
        log(f">>> 交不交？采用管线的留一法 RMSE={best_rmse:.2f} 还不如已知最好的 {REF_RMSE}"
            f"（榜 27.53739）⇒ **建议这次别交**，先加预算/epoch 重跑")
    else:
        log(f">>> 交不交？没有可用模型，RMSE 不可比 ⇒ **建议别交**")

    # ---- 候选集：每条管线都在留一法上打分，按 CV 自动选，不靠我猜 ----
    cand["s1"] = (s1_rmse, te_prior, f"第一阶段 {len(s1_models)} 模型集成 + 翻转 TTA")
    if s1_robust_rmse is not None and te_prior_robust is not None:
        cand["s1_robust"] = (s1_robust_rmse, te_prior_robust,
                             "第一阶段 + 不稳图拉向几何先验")
    if s1_veto_rmse is not None and te_prior_veto is not None:
        cand["s1_veto"] = (s1_veto_rmse, te_prior_veto, "第一阶段 + 视盘否决")
    if final_n > 0:
        _s2o = s2_oof / np.maximum(1, s2_cnt)[:, None]
        _s2t = final_sum / final_n
        cand["s2"] = (oof_rmse, _s2t, f"两级级联（{len(s1_models)}+{final_n} 模型）")
        if fallback_tr is not None:
            try:
                _a2, _n2 = veto_fix(_s2t, discs_te, fallback_te)
                _o2, _n2o = veto_fix(_s2o, discs_tr, fallback_tr)
                _r2, _ = rmse_report(_o2, true_xy, "[D: 级联+否决]")
                cand["s2_veto"] = (_r2, _a2, "两级级联 + 视盘否决")
                log(f"[D] 否决改写了 训练 {_n2o}/{len(tr)} 张，测试 {_n2}/{len(te)} 张")
            except Exception as e:
                log(f"  [D] 跳过：{type(e).__name__}: {e}")

    log("=== 候选对比（留一法 RMSE，越小越好）===")
    for _nm in sorted(cand, key=lambda k: cand[k][0]):
        log(f"   {_nm:9s} RMSE={cand[_nm][0]:7.3f}   {cand[_nm][2]}")
    _pick = min(cand, key=lambda k: cand[k][0])
    test_pred = cand[_pick][1]
    best_rmse = cand[_pick][0]
    how = f"{_pick} ← {cand[_pick][2]}（按留一法最优自动选中）"
    log(f"=== 选中 {_pick}：留一法 RMSE={best_rmse:.3f}（榜分≈{0.62 * best_rmse:.2f}）===")
    if best_rmse < REF_RMSE:
        log(f">>> 交不交？{best_rmse:.2f} < 参照 {REF_RMSE:.2f} ⇒ **值得提交**")
    else:
        log(f">>> 交不交？{best_rmse:.2f} >= 参照 {REF_RMSE:.2f} ⇒ **建议这次别交**")

    # ---- (0,0) 预警：热图太平坦的测试图要人工复核 ----
    if s1_models:
        try:
            x = torch.stack([torch.from_numpy(c1_te.arr[i]).permute(2, 0, 1).float() / 255.0
                             for i in range(len(te))]).to(DEVICE)
            x = normalize(x)
            acc = 0
            n_use = min(len(s1_models), 3)
            for m, cfg in s1_models[:n_use]:
                m.eval()
                with torch.no_grad(), amp_ctx():
                    o = m(x)
                    o = o + torch.flip(m(torch.flip(x, dims=[3])), dims=[3])
                acc = acc + o.float()
            pk, ent = peak_info(acc)
            pk = pk.cpu().numpy()
            ent = ent.cpu().numpy()
            order = np.argsort(ent)[::-1]
            log("[(0,0) 预警] 热图熵最高的 5 张（越大越平坦，越可能该填 (0,0)）：")
            for k in order[:5]:
                log(f"    {te[k][0]}  entropy={ent[k]:.4f}  peak={pk[k]:+.2f}")
            log(f"    熵中位数 = {np.median(ent):.4f}")
        except Exception as e:
            log(f"  [(0,0) 预警] 跳过：{type(e).__name__}: {e}")

    # ---- 把留一法的原始数组存下来：下次换思路时可以直接离线分析，不用再烧一次 GPU ----
    try:
        np.save(OUT / "oof_true.npy", true_xy)
        np.save(OUT / "oof_s1.npy", prior_oof)
        np.save(OUT / "te_s1.npy", te_prior)
        if final_n > 0:
            np.save(OUT / "oof_s2.npy", s2_oof / np.maximum(1, s2_cnt)[:, None])
        if s1_robust_rmse is not None and te_prior_robust is not None:
            np.save(OUT / "oof_s1_robust.npy", s1_robust_oof)
        if fallback_tr is not None:
            np.save(OUT / "disc_train.npy", np.array(
                [d if d is not None else [np.nan] * 3 for d in discs_tr], np.float64))
            np.save(OUT / "disc_test.npy", np.array(
                [d if d is not None else [np.nan] * 3 for d in discs_te], np.float64))
        if dis_tr is not None:
            np.savez(OUT / "disagree.npz", tr=dis_tr, te=dis_te)
        log("[dump] 留一法原始数组已落盘（oof_*.npy / disagree.npz），下次可离线分析")
    except Exception as e:
        log(f"  [dump] 跳过：{type(e).__name__}: {e}")

    # ---- 每个候选都落一份文件：一次提交额度就能验证一条管线 ----
    for _nm in sorted(cand):
        try:
            dump_submission(sample_rows, te, cand[_nm][1], OUT / f"submission_{_nm}.csv", "")
            log(f"  [候选文件] submission_{_nm}.csv   留一法 RMSE={cand[_nm][0]:.3f}")
        except Exception as e:
            log(f"  [{_nm}] 写文件失败：{type(e).__name__}: {e}")

    out_csv = OUT / "submission.csv"
    pred_by_id = dump_submission(sample_rows, te, test_pred, out_csv, "")
    log(f"[提交] submission.csv ← {how}")
    check_submission(out_csv, sample_path)

    summary = {
        "how": how,
        "s1_models": len(s1_models),
        "s2_models": final_n,
        "s1_prior_rmse": s1_rmse,
        "elapsed_min": el(),
        "pred": {str(k): [pred_by_id[k][0], pred_by_id[k][1]] for k in sorted(pred_by_id)},
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"完成，用时 {el():.1f} min。文件：{out_csv}")
    return out_csv


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        log("!! 主流程异常结束，检查上面 traceback")
        raise
