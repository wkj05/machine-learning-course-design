# ============================================================================
#  任务4: 眼底图像血管分割 (DRIVE-style)  --  Kaggle 单文件版
#  ---------------------------------------------------------------------------
#  指标: Dice     提交: run-length 编码的 csv     数据: 20 train / 20 test
#
#  ★★ 最重要的一件事（读这里）★★
#  竞赛页写「血管区域像素值为 0，其他为 255」，但这是**反的**。
#  证据有两条，互相印证：
#    1. train/label/*.jpg 是灰度图，**只有 0 和 255 两个值**，血管（细线网络）
#       在 255 上，占比 7.5%~9.8% —— 与解剖学一致；
#    2. 官方 segmentation_to_csv.py 里是
#          image[image > 0] = 1
#          dots = np.where(image.flatten() == 1)[0]
#       也就是把**非零像素**当作要编码的前景。
#  所以：mask 里血管必须写成 255（非零），背景 0。照页面那句话写会得到 Dice≈0。
#
#  ★★ 第二件：评测网格是 512x512 ★★
#  官方转换器会先 `image.resize((512, 512), BILINEAR)` 再取 `>0`。
#  双线性把 565x584 缩到 512x512 后，边缘任何非零插值像素都会变成前景，
#  等于给 mask 加了一圈膨胀。本脚本**直接输出 512x512 的 mask**，
#  这样 resize 是恒等变换，不受这个影响；并且训练时的 Dice 也按
#  「标签先 resize(512) > 0」来计算，与评测口径逐位一致。
#
#  用法
#  ----
#  Kaggle: 新建 notebook -> Add Input 挂 2026task-4 -> Accelerator 选 GPU ->
#          Internet 随意(本脚本不需要联网/不需要预训练权重) -> 把本文件整个
#          粘进一个 cell -> Run。跑完在 Output 里拿 submission.csv。
#  本机:  python t4_kaggle.py            (自动找 _inbox/task4/segmentation)
#  冒烟:  T4_SMOKE=1 python t4_kaggle.py (2 折 4 epoch 小分辨率, 只验链路)
# ============================================================================

import os
import sys
import math
import json
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:
    timm = None


# ============================== 配置 ==============================
TIME_BUDGET_MIN = 420.0      # 总时间预算(分钟)。Kaggle GPU 单次上限 9h, 留足余量。
                             # 这个数据集很小, 实际会用掉 20~60 min, 剩下的预算
                             # 会被用来多跑 epoch(见 MAX_EPOCHS)。
MAX_EPOCHS      = int(os.environ.get('T4_EPOCHS', 240))   # 每折 epoch 上限; 预算先用完则提前停
N_FOLDS         = 5          # 交叉验证折数(20 张 -> 每折 4 张验证)
N_SEEDS         = int(os.environ.get('T4_SEEDS', 2))      # 每个折重复训练几个种子(种子集成)
SEED            = 42

PATCH           = int(os.environ.get('T4_PATCH', 128))    # 训练裁块边长
PATCHES_PER_IMG = 10         # 每个 epoch 每张训练图采多少块
VESSEL_BIAS     = 0.60       # 采块时"以血管像素为中心"的比例(其余随机位置)
# 尺度抖动上限（乘性）：1.0 = 关闭。1.45 表示裁块边长在 patch/1.45 ~ patch*1.45 之间随机，
# 再缩回固定输入尺寸 —— 让血管以不同表观宽度出现。
SCALE_JIT       = float(os.environ.get('T4_SCALE_JIT', 1.0))
# 推理多尺度（乘性缩放原图再预测，概率缩回原尺寸平均）。"1.0" = 单尺度。
TTA_SCALES      = [float(v) for v in os.environ.get('T4_TTA_SCALES', '1.0').split(',') if v]
BATCH           = 16
LR              = 1.2e-3
WD              = 1e-4
BASE_CH         = 32         # UNet 基础通道 / 预训练版 stem 通道
DEPTH           = 5          # 下采样层数
EVAL_SIZE       = 512        # 评测网格(与官方转换器一致, 不要改)
# 训练网格：0 = 原尺寸(标签取 >127)；1 = 直接缩到 512 并用「resize 后 >0」的标签，
# 与官方转换器和评测口径逐位一致（标签被膨胀约 1px，模型学的就是被评分的那张图）。
TRAIN_ON_EVAL   = int(os.environ.get('T4_EVALGRID', '1')) == 1

# ---- 编码器 ----
# 空字符串 = 从零训练的 UNet（不联网、不需要权重，最稳）
# 非空     = ImageNet 预训练编码器 + UNet 解码头（需要联网下载权重，通常更强）
BB_NAME         = os.environ.get('T4_BB', 'convnext_tiny.fb_in22k_ft_in1k')
BB_LR_SCALE     = 0.15       # 预训练部分用更小的学习率（微调，别把好特征冲掉）

IN_CH           = 3          # [绿色通道, 局部对比度增强, 视野掩膜]
MEAN            = (0.0, 0.0, 0.0)
STD             = (1.0, 1.0, 1.0)

N_THRESH        = 33         # 阈值搜索点数
# 阈值搜索区间要覆盖两种训练网格的两个截然不同的最优点（实测）：
#   原尺寸训练      -> 最优 tau ≈ 0.12~0.20（模型吐出的是"锐利细血管"，要靠低阈值补厚度）
#   评测网格训练    -> 最优 tau ≈ 0.60    （模型直接学被膨胀过的标签，阈值回到正常位置）
# 所以区间取 [0.08, 0.88]，两端都留余量，保证最优点落在内部而不是被边界截断。
THR_LO          = 0.08
THR_HI          = 0.88
MIN_SIZE_LIST   = (0, 30, 80)  # 后处理: 去掉小于该像素数(512网格)的连通域
TTA             = True       # 推理时做 4 向翻转 TTA

OUT_DIR         = Path(os.environ.get('T4_OUT', 't4_out'))
SMOKE           = os.environ.get('T4_SMOKE', '') == '1'
if SMOKE:
    TIME_BUDGET_MIN = 8.0
    MAX_EPOCHS      = 4
    N_FOLDS         = 2
    PATCH           = 96
    PATCHES_PER_IMG = 3
    BATCH           = 8
    BASE_CH         = 16
    DEPTH           = 4

T0 = time.time()


def left_min():
    return TIME_BUDGET_MIN - (time.time() - T0) / 60.0


def log(*a):
    print(*a, flush=True)


# ============================== 基础工具 ==============================
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _walk_dirs(base, max_depth=3, max_dirs=4000):
    """有界的目录遍历（避免在巨大工作区上 rglob 卡死）。"""
    out = []
    base = Path(base)
    stack = [(base, 0)]
    n = 0
    while stack and n < max_dirs:
        d, dep = stack.pop()
        out.append(d)
        n += 1
        if dep >= max_depth:
            continue
        try:
            for e in d.iterdir():
                if e.is_dir():
                    stack.append((e, dep + 1))
        except Exception:
            pass
    return out


def find_root():
    """找到含 train/image, train/label, test/image 的那一层目录。"""
    cands = [
        '/kaggle/input/competitions/2026task-4/segmentation',
        '/kaggle/input/2026task-4/segmentation',
        '_inbox/task4/segmentation',
        'task4/segmentation',
        'segmentation',
        '.',
    ]
    seen = []
    for base in ['/kaggle/input', '_inbox/task4', 'segmentation', '.']:
        p = Path(base)
        if not p.exists():
            continue
        for d in _walk_dirs(p):
            try:
                if (d / 'train' / 'image').is_dir() and (d / 'test' / 'image').is_dir():
                    seen.append(d)
            except Exception:
                pass
        if seen:
            break
    cands += [str(x) for x in seen]
    for c in cands:
        p = Path(c)
        if (p / 'train' / 'image').is_dir() and (p / 'test' / 'image').is_dir():
            return p
    raise FileNotFoundError(
        '没找到数据集：需要 train/image, train/label, test/image。\n'
        'Kaggle 上请确认 Add Input 里已挂 2026task-4。\n'
        '当前目录: ' + str(Path('.').resolve()))


def list_ids(d):
    fs = [f for f in Path(d).iterdir()
          if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')]
    return sorted(fs, key=lambda f: f.stem)


# ---------------- 预处理：绿色通道 + 局部对比度 + 视野掩膜 ----------------
def _box_blur(a, r):
    """积分图实现的方框模糊(对 565x584 这种尺寸足够快, 且不依赖 scipy/cv2)。"""
    r = max(1, int(r))
    k = 2 * r + 1
    pad = np.pad(a, ((r, r), (r, r)), mode='edge')
    ii = pad.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    H, W = a.shape
    out = (ii[k:k + H, k:k + W] - ii[0:H, k:k + W]
           - ii[k:k + H, 0:W] + ii[0:H, 0:W])
    return out / float(k * k)


def prep_one(img_rgb):
    """返回 (3,H,W) float32: 绿色通道, 局部对比度增强, 视野掩膜。"""
    a = img_rgb.astype(np.float32) / 255.0
    g = a[:, :, 1]
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]

    # 局部对比度: 除法式背景归一化(对血管这类细结构很有效)
    bg = _box_blur(g, 24)
    loc = g / (bg + 0.02)
    loc = np.clip((loc - 0.6) / 0.9, 0.0, 1.0)

    # 视野: 亮区 -> 闭运算 -> 取最大块(这里用阈值+开闭, 不引入额外依赖)
    m = (lum > 0.06).astype(np.float32)
    m = (_box_blur(m, 8) > 0.35).astype(np.float32)
    m = (_box_blur(m, 12) > 0.55).astype(np.float32)

    return np.stack([g, loc, m], 0).astype(np.float32)


# ---------------- 标签：与评测口径一致（resize(512) > 0） ----------------
def gt_at_eval(label_img):
    """把 co 原尺寸标签映射到 512x512 的评测网格，返回 bool 数组。"""
    r = label_img.resize((EVAL_SIZE, EVAL_SIZE), Image.BILINEAR)
    return (np.asarray(r, dtype=np.uint8) > 0)


def prob_at_eval(prob_np):
    """把原尺寸概率图映射到 512x512。"""
    if prob_np.shape == (EVAL_SIZE, EVAL_SIZE):
        return prob_np
    t = torch.from_numpy(prob_np)[None, None]
    t = F.interpolate(t, size=(EVAL_SIZE, EVAL_SIZE), mode='bilinear', align_corners=False)
    return t[0, 0].numpy()


def dice_at_eval(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    s = pred_bool.sum() + gt_bool.sum()
    if s == 0:
        return 1.0
    return float(2.0 * inter / s)


def drop_small(mask, min_size):
    """去掉小于 min_size 的 8-邻域连通域。优先用 scipy，退化为纯 numpy BFS。"""
    if min_size <= 0:
        return mask
    try:
        from scipy import ndimage
        st = np.ones((3, 3), np.int32)
        lab, n = ndimage.label(mask, structure=st)
        if n == 0:
            return mask
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= min_size
        return keep[lab]
    except Exception:
        pass
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    cur = 0
    out = mask.copy()
    idx = np.argwhere(mask)
    for y0, x0 in idx:
        if lab[y0, x0] != 0:
            continue
        cur += 1
        stack = [(int(y0), int(x0))]
        lab[y0, x0] = cur
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
        if len(comp) < min_size:
            for y, x in comp:
                out[y, x] = False
    return out


# ============================== 模型 ==============================
class DoubleConv(nn.Module):
    def __init__(self, cin, cout, gn=8):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(gn, cout), cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(gn, cout), cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.b(x)


class UNet(nn.Module):
    def __init__(self, cin=IN_CH, base=BASE_CH, depth=DEPTH):
        super().__init__()
        chs = [base * (2 ** min(i, 4)) for i in range(depth)]
        self.enc = nn.ModuleList()
        c = cin
        for ch in chs:
            self.enc.append(DoubleConv(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(c, c * 2)
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        c = c * 2
        for ch in reversed(chs):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(DoubleConv(ch * 2, ch))
            c = ch
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        for i, (u, d) in enumerate(zip(self.up, self.dec)):
            x = u(x)
            s = skips[-1 - i]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode='nearest')
            x = torch.cat([s, x], 1)
            x = d(x)
        return self.head(x)


class ConvNextSeg(nn.Module):
    """ImageNet 预训练编码器(ConvNeXt/EfficientNet 等 timm features_only) + UNet 解码头。

    血管是 1~3 px 的细结构，所以解码要一路回到 stride 1：
        backbone   : stride 4 / 8 / 16 / 32
        自建 stem  : stride 2 / 4   （补上 backbone 给不出的高分辨率）
        解码       : 32 → 16 → 8 → 4 → 2 → 1
    输入高宽必须是 32 的倍数（backbone 的下采样要求）。
    """

    def __init__(self, name, pretrained=True, base=BASE_CH, drop_path=0.1):
        super().__init__()
        self.bb = timm.create_model(name, features_only=True, pretrained=pretrained,
                                    out_indices=(0, 1, 2, 3), drop_path_rate=drop_path)
        c = list(self.bb.feature_info.channels())      # 例如 convnext_tiny: [96,192,384,768]
        self.e2 = nn.Sequential(
            nn.Conv2d(IN_CH, base, 3, 2, 1, bias=False),
            nn.GroupNorm(8, base), nn.SiLU(inplace=True))
        self.e4 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1, bias=False),
            nn.GroupNorm(8, base * 2), nn.SiLU(inplace=True))

        self.p0 = nn.Conv2d(c[0], 64, 1)
        self.p1 = nn.Conv2d(c[1], 96, 1)
        self.p2 = nn.Conv2d(c[2], 128, 1)
        self.p3 = nn.Conv2d(c[3], 160, 1)

        self.d3 = DoubleConv(160, 128)
        self.d2 = DoubleConv(128 + 128, 96)
        self.d1 = DoubleConv(96 + 96, 96)
        self.d0 = DoubleConv(96 + 64 + base * 2, 64)          # 回到 stride 4
        self.u1 = nn.ConvTranspose2d(64, 64, 2, stride=2)      # stride 2
        self.dm = DoubleConv(64 + base, 64)
        self.u0 = nn.ConvTranspose2d(64, 32, 2, stride=2)      # stride 1
        self.df = DoubleConv(32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        fs = self.bb(x)                     # stride 4, 8, 16, 32
        x2 = self.e2(x)                     # stride 2
        x4 = self.e4(x2)                    # stride 4
        p0, p1, p2, p3 = (self.p0(fs[0]), self.p1(fs[1]),
                          self.p2(fs[2]), self.p3(fs[3]))
        d = self.d3(p3)
        d = self.d2(torch.cat([F.interpolate(d, size=p2.shape[-2:], mode='nearest'), p2], 1))
        d = self.d1(torch.cat([F.interpolate(d, size=p1.shape[-2:], mode='nearest'), p1], 1))
        d = self.d0(torch.cat([F.interpolate(d, size=p0.shape[-2:], mode='nearest'), p0, x4], 1))
        d = self.dm(torch.cat([self.u1(d), x2], 1))
        d = self.df(self.u0(d))
        return self.head(d)


def build_model(pretrained=True):
    """按 BB_NAME 建模型。BB_NAME 为空或 timm 不可用时退回从零训练的 UNet。"""
    if BB_NAME and timm is not None:
        try:
            m = ConvNextSeg(BB_NAME, pretrained=pretrained)
            n = sum(p.numel() for p in m.parameters()) / 1e6
            log(f'    [模型] {BB_NAME} 预训练编码器, 参数量 {n:.2f}M')
            return m
        except Exception as e:
            log(f'    [!!] 预训练编码器建不起来: {type(e).__name__}: {e}')
            log('    [..] 退回从零训练的 UNet')
    else:
        log('    [模型] 从零训练的 UNet')
    return UNet()


# ============================== 损失 ==============================
def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(3.0, device=logits.device))
    return 0.5 * bce + 0.5 * soft_dice_loss(logits, target)


# ============================== 数据 ==============================
def elastic_warp(img, mask, alpha=6.0, seed=None):
    """img: (C,H,W) tensor, mask: (1,H,W) tensor -> 同步弹性形变。"""
    C, H, W = img.shape
    g = torch.Generator(device='cpu').manual_seed(int(seed) if seed is not None else 0)
    low = max(4, H // 16)
    dy = torch.randn(1, 1, low, low, generator=g) * alpha
    dx = torch.randn(1, 1, low, low, generator=g) * alpha
    dy = F.interpolate(dy, size=(H, W), mode='bilinear', align_corners=False) * (H / 128.0)
    dx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False) * (W / 128.0)
    ys = torch.arange(H, dtype=torch.float32).view(1, H, 1).expand(1, H, W)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, W).expand(1, H, W)
    gy = (ys + dy[0, 0]) / max(H - 1, 1) * 2 - 1
    gx = (xs + dx[0, 0]) / max(W - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)
    oi = F.grid_sample(img[None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0]
    om = F.grid_sample(mask[None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0]
    return oi, om


_VESSEL_PTS = {}


def sample_patch(imgs, labs, i, rng, patch=PATCH):
    """从第 i 张图取一个 patch，返回 (3,patch,patch), (1,patch,patch)。

    带尺度抖动：同一根血管会以不同表观宽度出现在输入里（乘性 1/SCALE_JIT ~ SCALE_JIT）。
    动机：训练集里 #04 的 GT 血管明显比预测粗、#16 明显比预测细，一个固定尺度下的
    模型两头都顾不好；让尺度成为增广维度，可以让"血管性"判断与表观宽度解耦。"""
    img = imgs[i]              # (3,H,W) float32 np
    lab = labs[i]              # (H,W) bool
    H, W = lab.shape
    s = 1.0
    if SCALE_JIT > 1.0:
        s = float(np.exp(rng.uniform(-np.log(SCALE_JIT), np.log(SCALE_JIT))))
    sz = int(round(patch * s))
    sz = max(48, min(sz, min(H, W)))
    y = x = None
    if rng.random() < VESSEL_BIAS:
        if i not in _VESSEL_PTS:
            _VESSEL_PTS[i] = np.nonzero(lab)
        ys, xs = _VESSEL_PTS[i]
        if len(ys) > 0:
            k = rng.integers(len(ys))
            y = int(ys[k]) + int(rng.integers(-sz // 3, sz // 3 + 1))
            x = int(xs[k]) + int(rng.integers(-sz // 3, sz // 3 + 1))
    if y is None:
        y = int(rng.integers(0, max(1, H - sz)))
        x = int(rng.integers(0, max(1, W - sz)))
    y = int(np.clip(y, 0, H - sz))
    x = int(np.clip(x, 0, W - sz))
    pi = torch.from_numpy(np.ascontiguousarray(img[:, y:y + sz, x:x + sz]))
    pm = torch.from_numpy(np.ascontiguousarray(lab[y:y + sz, x:x + sz].astype(np.float32)))[None]

    if sz != patch:            # 回到固定输入尺寸
        pi = F.interpolate(pi[None], size=(patch, patch), mode='bilinear', align_corners=False)[0]
        pm = F.interpolate(pm[None], size=(patch, patch), mode='bilinear', align_corners=False)[0]
    pi, pm = pi.contiguous(), pm.contiguous()

    # 翻转 / 旋转
    if rng.random() < 0.5:
        pi, pm = torch.flip(pi, [-1]), torch.flip(pm, [-1])
    if rng.random() < 0.5:
        pi, pm = torch.flip(pi, [-2]), torch.flip(pm, [-2])
    k = int(rng.integers(0, 4))
    if k:
        pi, pm = torch.rot90(pi, k, (-2, -1)), torch.rot90(pm, k, (-2, -1))
    pi, pm = pi.contiguous(), pm.contiguous()

    # 弹性形变
    if rng.random() < 0.35:
        pi, pm = elastic_warp(pi, pm, alpha=5.0, seed=int(rng.integers(1 << 30)))

    # 光度
    g_ = float(np.exp(rng.normal(0, 0.18)))
    pi = torch.clamp(pi * g_, 0, 1)
    b_ = float(rng.normal(0, 0.04))
    pi = torch.clamp(pi + b_, 0, 1)
    c_ = 1.0 + float(rng.normal(0, 0.12))
    pi = torch.clamp((pi - 0.5) * c_ + 0.5, 0, 1)
    if rng.random() < 0.3:                      # 通道增益(模拟成像差异)
        pi = torch.clamp(pi * torch.tensor(
            [1 + rng.normal(0, .08), 1 + rng.normal(0, .08), 1.0]).view(3, 1, 1), 0, 1)
    pm = (pm > 0.5).float()
    return pi, pm


# ============================== 训练 ==============================
def train_fold(imgs, labs, tr_idx, device, epochs, seed, tag):
    seed_all(seed)
    model = build_model().to(device)
    # 预训练部分用小学习率（微调），新加的头部用正常学习率
    if hasattr(model, 'bb'):
        bb_ps = [p for p in model.bb.parameters() if p.requires_grad]
        head_ps = [p for p in model.parameters() if p.requires_grad and not any(p is q for q in bb_ps)]
        groups = [{'params': bb_ps, 'lr': LR * BB_LR_SCALE}, {'params': head_ps, 'lr': LR}]
    else:
        groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': LR}]
    opt = torch.optim.AdamW(groups, lr=LR, weight_decay=WD)
    total_steps = max(1, epochs * math.ceil(len(tr_idx) * PATCHES_PER_IMG / BATCH))
    warm = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))) * 0.98 + 0.02)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    rng = np.random.default_rng(seed)
    step = 0
    for ep in range(epochs):
        model.train()
        buf_i, buf_m = [], []
        for _ in range(len(tr_idx) * PATCHES_PER_IMG):
            i = int(tr_idx[rng.integers(len(tr_idx))])
            pi, pm = sample_patch(imgs, labs, i, rng)
            buf_i.append(pi)
            buf_m.append(pm)
        tot = 0.0
        nb = 0
        for b in range(0, len(buf_i), BATCH):
            xb = torch.stack(buf_i[b:b + BATCH]).to(device, non_blocking=True)
            yb = torch.stack(buf_m[b:b + BATCH]).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(xb)
                loss = seg_loss(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1
            step += 1
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            log(f'  [{tag}] ep{ep + 1}/{epochs} loss={tot / max(nb, 1):.4f} '
                f'lr={opt.param_groups[0]["lr"]:.2e} 已用 {((time.time() - T0) / 60):.1f} min')
    return model


@torch.no_grad()
def _forward_tta(model, x, tta):
    """x 已在 device 上，(1,C,H,W) -> 该尺寸下的概率图 (1,1,H,W)。4 向翻转平均。"""
    views = [(False, False), (True, False), (False, True), (True, True)] if tta else [(False, False)]
    acc = None
    for fh, fv in views:
        xi = x
        if fh:
            xi = torch.flip(xi, [-1])
        if fv:
            xi = torch.flip(xi, [-2])
        o = torch.sigmoid(model(xi))
        if fh:
            o = torch.flip(o, [-1])
        if fv:
            o = torch.flip(o, [-2])
        acc = o if acc is None else acc + o
    return acc / float(len(views))


def predict_full(model, img, device, tta=None):
    """整图推理(按 32 的倍数补边), 返回原尺寸概率图 (H,W) float32.

    默认做 4 向翻转 TTA（模型用 GroupNorm，不含 BatchNorm 的全局统计，翻转安全）。
    再做多尺度：按 TTA_SCALES 把原图缩放后分别预测，概率缩回原尺寸后平均。
    多尺度收益来自"同一根血管在不同表观宽度下都被判成血管"，互补粗/细血管。"""
    if tta is None:
        tta = TTA
    model.eval()
    C, H, W = img.shape
    scales = [float(s) for s in (TTA_SCALES or [1.0])]
    acc = None
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            im_s, hs, ws = img, H, W
        else:
            hs = max(32, int(round(H * s)))
            ws = max(32, int(round(W * s)))
            t = torch.from_numpy(img)[None]
            t = F.interpolate(t, size=(hs, ws), mode='bilinear', align_corners=False)[0]
            im_s = np.ascontiguousarray(t.numpy())
        ph = ((hs + 31) // 32) * 32      # 32 的倍数：backbone 有四层下采样，UNet 五层
        pw = ((ws + 31) // 32) * 32
        x = torch.zeros(1, C, ph, pw)
        x[0, :, :hs, :ws] = torch.from_numpy(im_s)
        o = _forward_tta(model, x.to(device), tta)[0, 0, :hs, :ws]
        if (hs, ws) != (H, W):
            o = F.interpolate(o[None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        acc = o if acc is None else acc + o
    acc = acc / float(len(scales))
    return acc.cpu().numpy().astype(np.float32)


# ============================== 提交（逐字复刻官方转换器） ==============================
def rle_encode(mask01):
    """与 segmentation_to_csv.py 完全一致: C 序展平, 1 基起点, 空格分隔。"""
    dots = np.where(np.asarray(mask01, dtype=np.uint8).flatten() == 1)[0]
    runs = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            runs.extend((int(b) + 1, 0))
        runs[-1] += 1
        prev = int(b)
    return ' '.join(str(r) for r in runs)


def rle_decode(s, n=EVAL_SIZE * EVAL_SIZE):
    out = np.zeros(n, np.uint8)
    if not str(s).strip():
        return out.reshape(EVAL_SIZE, EVAL_SIZE)
    v = [int(x) for x in str(s).split()]
    for i in range(0, len(v), 2):
        st = v[i] - 1
        ln = v[i + 1]
        out[st:st + ln] = 1
    return out.reshape(EVAL_SIZE, EVAL_SIZE)


def read_sample(sample_path):
    rows = []
    with open(sample_path, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    hdr = lines[0]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        rows.append(ln.split(',')[0].strip())
    return hdr, rows


def write_submission(sample_path, pred_by_id, out_path):
    hdr, ids = read_sample(sample_path)
    lines = [hdr]
    n_ok = 0
    for i in ids:
        s = pred_by_id.get(i, '')
        if s:
            n_ok += 1
        lines.append(f'{i},{s}')
    txt = '\r\n'.join(lines) + '\r\n'
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.write(txt)
    raw = Path(out_path).read_bytes()
    CRLF = b'\r\n'
    LF = b'\n'
    BOM = b'\xef\xbb\xbf'
    n_crlf = raw.count(CRLF)
    n_barelf = raw.count(LF) - n_crlf
    has_bom = raw[:3] == BOM
    log(f'[提交] {out_path}  行数={len(lines)}  非空={n_ok}/{len(ids)}  '
        f'bytes={len(raw)}  CRLF={n_crlf}  裸LF={n_barelf}  BOM={has_bom}')
    return ids


def self_test_rle():
    m = np.zeros((EVAL_SIZE, EVAL_SIZE), np.uint8)
    m[0, 0] = 1
    m[0, 1] = 1
    m[3, 7] = 1
    s = rle_encode(m)
    back = rle_decode(s)
    ok = bool((back == m).all())
    log(f'[自测] RLE 往返一致 = {ok}   (0,0),(0,1),(3,7) -> "{s}"')
    assert ok, 'RLE 编解码不一致, 说明下标/顺序有 bug'
    return ok


# ============================== 主流程 ==============================
def main():
    seed_all(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 78)
    log(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}  '
        f'device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'预算 {TIME_BUDGET_MIN:.0f} min  N_FOLDS={N_FOLDS}  MAX_EPOCHS={MAX_EPOCHS}  '
        f'PATCH={PATCH}  EVAL_SIZE={EVAL_SIZE}  SMOKE={SMOKE}')

    root = find_root()
    log(f'DATA_ROOT = {root}')
    tr_files = list_ids(root / 'train' / 'image')
    te_files = list_ids(root / 'test' / 'image')
    lb_map = {f.stem: f for f in list_ids(root / 'train' / 'label')}
    log(f'train={len(tr_files)}  label={len(lb_map)}  test={len(te_files)}')

    sample_path = root / 'sample_submission.csv'
    _, sample_ids = read_sample(sample_path)
    log(f'样例 {sample_path.name}: {len(sample_ids)} 行, Id = {sample_ids[0]}..{sample_ids[-1]}')

    # ---- 读数据 ----
    t = time.time()
    imgs, labs = [], []
    tr_stems = [f.stem for f in tr_files]
    log(f'训练网格 = {"EVAL(512, 与评测口径逐位一致)" if TRAIN_ON_EVAL else "原尺寸"}')
    for f in tr_files:
        lf = lb_map.get(f.stem)
        if lf is None:
            raise FileNotFoundError(f'缺少标签: {f.stem}')
        im_pil = Image.open(f).convert('RGB')
        lb_pil = Image.open(lf).convert('L')
        if TRAIN_ON_EVAL:
            im_pil = im_pil.resize((EVAL_SIZE, EVAL_SIZE), Image.BILINEAR)
            lb_pil = lb_pil.resize((EVAL_SIZE, EVAL_SIZE), Image.BILINEAR)
            labs.append(np.asarray(lb_pil, dtype=np.uint8) > 0)
        else:
            labs.append(np.asarray(lb_pil, dtype=np.uint8) > 127)
        imgs.append(prep_one(np.asarray(im_pil)))
    if TRAIN_ON_EVAL:
        te_imgs = [prep_one(np.asarray(Image.open(f).convert('RGB').resize(
            (EVAL_SIZE, EVAL_SIZE), Image.BILINEAR))) for f in te_files]
    else:
        te_imgs = [prep_one(np.asarray(Image.open(f).convert('RGB'))) for f in te_files]
    H0, W0 = imgs[0].shape[1:]
    log(f'读入 {len(imgs)} 训练 / {len(te_imgs)} 测试, 尺寸 {W0}x{H0}, 用时 {time.time() - t:.1f}s')
    pos = np.mean([l.mean() for l in labs])
    log(f'血管像素占比 mean={pos:.4f}  (min={min(l.mean() for l in labs):.4f}, '
        f'max={max(l.mean() for l in labs):.4f})')

    self_test_rle()

    # ---- 评测网格上的 GT（与官方转换器同口径）----
    gt512 = {}
    for f in tr_files:
        lf = lb_map[f.stem]
        gt512[f.stem] = gt_at_eval(Image.open(lf).convert('L'))
    fr = [gt512[s].mean() for s in gt512]
    log(f'评测网格(512) 上血管占比 mean={np.mean(fr):.4f}  '
        f'(比原尺寸高, 因为双线性 resize 后 >0 会膨胀约 1px)')

    # ---- 折划分 ----
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr_files))
    fold_of = {}
    for j, p in enumerate(perm):
        fold_of[int(p)] = j % N_FOLDS
    folds = [np.array([i for i in range(len(tr_files)) if fold_of[i] == f]) for f in range(N_FOLDS)]
    log(f'折大小 = {[len(f) for f in folds]}')

    # ---- 兜底提交（训练前先落一份，保证 Output 里永远有东西）----
    pred_by_id = {sid: '' for sid in sample_ids}
    sub_csv = OUT_DIR / 'submission.csv'
    write_submission(sample_path, pred_by_id, sub_csv)
    log('[早存] 空提交已落盘（全空 = Dice 0，仅用于占位）')

    # ---- 训练 + 每折结束立刻存一次提交 ----
    models = []
    fold_stats = []
    est_epoch = None
    test_sum = np.zeros((len(te_imgs), H0, W0), np.float64)
    test_cnt = 0
    oof_acc = np.zeros((len(tr_files), EVAL_SIZE, EVAL_SIZE), np.float64)
    oof_n = np.zeros(len(tr_files), np.float64)
    for fi in range(N_FOLDS):
        va = folds[fi]
        tr = np.array([i for i in range(len(tr_files)) if fold_of[i] != fi])
        remain_models = (N_FOLDS - fi) * N_SEEDS
        # 按剩余预算分配每折 epoch（宁可少跑几 epoch 也要把全部 折×种子 跑完）
        if est_epoch is None:
            ep_cap = MAX_EPOCHS
        else:
            per_model_min = est_epoch * MAX_EPOCHS / 60.0
            affordable = (left_min() - 2.0) / max(remain_models, 1) / max(per_model_min, 1e-6) * MAX_EPOCHS
            ep_cap = int(min(MAX_EPOCHS, max(8, affordable)))
        log(f'--- fold {fi}/{N_FOLDS}: train={len(tr)} val={len(va)} epochs={ep_cap} '
            f'seeds={N_SEEDS} 剩余预算 {left_min():.1f} min')
        for si in range(N_SEEDS):
            t_f = time.time()
            model = train_fold(imgs, labs, tr, device, ep_cap,
                               SEED + 100 * fi + 1000 * si, f'f{fi}s{si}')
            el = time.time() - t_f
            if ep_cap > 0:
                est_epoch = el / ep_cap
            models.append(model)
            if os.environ.get('T4_SAVE_W', '1') == '1':
                try:
                    torch.save(model.state_dict(), OUT_DIR / f'm_f{fi}s{si}.pt')
                except Exception as _e:
                    log(f'  [!!] 存权重失败: {_e}')
            log(f'  fold {fi} seed {si} 用时 {el / 60:.1f} min ({el / max(ep_cap, 1):.2f} s/epoch)')
            fold_stats.append({'fold': fi, 'seed': si, 'epochs': ep_cap,
                               'sec': round(el, 1), 's_per_epoch': round(el / max(ep_cap, 1), 2)})
            # OOF：在该折的验证集上累加（评估空间，供阈值扫描）
            for i in va:
                oof_acc[int(i)] += prob_at_eval(predict_full(model, imgs[i], device))
                oof_n[int(i)] += 1
            # 测试集累加（多模型集成）
            for k, ti in enumerate(te_imgs):
                test_sum[k] += predict_full(model, ti, device)
            test_cnt += 1

            # 用当前累积的模型先出一版（阈值 0.5, 无后处理）
            cur = test_sum / max(test_cnt, 1)
            pred_by_id = {}
            for k, f in enumerate(te_files):
                pm = (prob_at_eval(cur[k]) > 0.5)
                pred_by_id[f.stem] = rle_encode(pm.astype(np.uint8))
            write_submission(sample_path, pred_by_id, sub_csv)
            log(f'[早存] fold {fi} seed {si} 后已刷新（{test_cnt} 个模型）')

    # ---- OOF 概率矩阵 ----
    oof_p = np.zeros((len(tr_files), EVAL_SIZE, EVAL_SIZE), np.float32)
    for i in range(len(tr_files)):
        oof_p[i] = oof_acc[i] / max(oof_n[i], 1)
    np.save(OUT_DIR / 'oof_prob512.npy', oof_p)
    te_p = np.zeros((len(te_files), EVAL_SIZE, EVAL_SIZE), np.float32)
    cur_test = test_sum / max(test_cnt, 1)
    for k in range(len(te_files)):
        te_p[k] = prob_at_eval(cur_test[k])
    np.save(OUT_DIR / 'test_prob512.npy', te_p)
    truth = np.stack([gt512[f.stem] for f in tr_files])

    # ---- 阈值 × 最小连通域 扫描（在 OOF 上；评估口径与榜单一致）----
    log('--- 阈值 / 后处理 扫描（OOF，512 网格，Dice）')
    thr_list = np.linspace(THR_LO, THR_HI, N_THRESH)
    best = (-1.0, 0.5, 0, '')
    scan = []
    for tau in thr_list:
        for ms in MIN_SIZE_LIST:
            ds = []
            for i in range(len(tr_files)):
                pm = oof_p[i] > tau
                if ms > 0:
                    pm = drop_small(pm, ms)
                ds.append(dice_at_eval(pm, truth[i]))
            d = float(np.mean(ds))
            scan.append({'tau': round(float(tau), 3), 'min_size': ms, 'dice': round(d, 4)})
            if d > best[0]:
                best = (d, float(tau), ms, '')
            log(f'   tau={tau:.2f} min_size={ms:3d}  Dice={d:.4f}')
    oof_dice, TAU, MS, _ = best
    log(f'>>> 选中 tau={TAU:.2f}  min_size={MS}  OOF Dice={oof_dice:.4f}')
    (OUT_DIR / 'scan_oof.json').write_text(json.dumps(
        {'best': {'dice': oof_dice, 'tau': TAU, 'min_size': MS}, 'scan': scan},
        ensure_ascii=False, indent=1), encoding='utf-8')

    # 每张图的 OOF Dice（找尾部）
    per = [dice_at_eval(drop_small(oof_p[i] > TAU, MS) if MS > 0 else (oof_p[i] > TAU), truth[i])
           for i in range(len(tr_files))]
    worst = sorted(range(len(tr_files)), key=lambda i: per[i])[:6]
    log('[OOF 最差 6 张] ' + '  '.join(f'{tr_stems[i]}:{per[i]:.4f}' for i in worst))
    log(f'[OOF Dice 分布] mean={np.mean(per):.4f} median={np.median(per):.4f} '
        f'min={min(per):.4f} max={max(per):.4f}')

    # ---- 最终测试提交 ----
    pred_by_id = {}
    vf = []
    for k, f in enumerate(te_files):
        pm = te_p[k] > TAU
        if MS > 0:
            pm = drop_small(pm, MS)
        pred_by_id[f.stem] = rle_encode(pm.astype(np.uint8))
        vf.append(pm.mean())
    ids = write_submission(sample_path, pred_by_id, sub_csv)
    log(f'[test] 预测血管占比 mean={np.mean(vf):.4f} (min={min(vf):.4f}, max={max(vf):.4f})')

    # ---- 提交自检 ----
    raw = (OUT_DIR / 'submission.csv').read_bytes()
    ok = True
    if len(raw) < 50:
        ok = False
    txt = raw.decode('utf-8')
    lines = txt.split('\r\n')
    if lines[0] != 'Id,Predicted':
        ok = False
    body = [l for l in lines[1:] if l.strip()]
    if len(body) != len(sample_ids):
        ok = False
    got_ids = [l.split(',')[0] for l in body]
    if got_ids != sample_ids:
        ok = False
    empty = sum(1 for l in body if l.split(',', 1)[1].strip() == '')
    log(f'[check] 行数={len(body)}/{len(sample_ids)}  表头OK={lines[0] == "Id,Predicted"}  '
        f'Id顺序OK={got_ids == sample_ids}  空预测={empty}  =>  {"通过" if ok else "不通过"}')
    # 抽样解码一张，确认 RLE 能还原
    k = 0
    back = rle_decode(body[k].split(',', 1)[1])
    ref = te_p[k] > TAU
    if MS > 0:
        ref = drop_small(ref, MS)
    log(f'[check] 解码 {got_ids[k]} => mask 面积 {int(back.sum())} px '
        f'({back.mean():.4f}), 与写入时一致={bool((back == ref).all())}')

    summary = {
        'metric': 'Dice', 'n_train': len(tr_files), 'n_test': len(te_files),
        'folds': fold_stats, 'oof_dice': round(float(oof_dice), 4),
        'tau': float(TAU), 'min_size': int(MS),
        'oof_per_image': {tr_stems[i]: round(float(per[i]), 4) for i in range(len(tr_files))},
        'total_min': round((time.time() - T0) / 60.0, 1),
        'smoke': SMOKE,
    }
    (OUT_DIR / 'run_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    log('=' * 78)
    log(f'完成，用时 {(time.time() - T0) / 60:.1f} min  (预算 {TIME_BUDGET_MIN:.0f} min)')
    log(f'   OOF Dice = {oof_dice:.4f}   tau={TAU:.2f}  min_size={MS}')
    log(f'   提交文件: {sub_csv.resolve()}')
    log('=' * 78)
    return summary


if __name__ == '__main__':
    main()
