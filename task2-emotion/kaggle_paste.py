# ====================== code cell 1/16 ======================
import os, sys, math, time, random, json, hashlib, warnings
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)

import timm
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
matplotlib.rcParams['axes.unicode_minus'] = False

print('python  :', sys.version.split()[0])
print('torch   :', torch.__version__)
print('timm    :', timm.__version__)
print('cuda    :', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')


# ====================== code cell 2/16 ======================
# ============================== 全局配置 ==============================
QUICK_MODE    = False          # True = 只跑 1 折、少 epoch，用来快速验证流程能跑通

SEED          = 42             # 固定随机种子，保证可复现
N_FOLDS       = 1 if QUICK_MODE else 5

EPOCHS        = 3 if QUICK_MODE else 30
WARMUP_EPOCHS = 1 if QUICK_MODE else 3
BATCH_SIZE    = 8 if QUICK_MODE else 64
IMG_SIZE      = 96             # 训练分辨率：原图只有 48x48，会被放大到这么大
CACHE_SIZE    = 48             # 按原始分辨率缓存（原图就是 48x48，不做重采样，省内存也最灵活）
NUM_WORKERS   = 2

LR_HEAD       = 1e-3
LR_BACKBONE   = 2e-4
WEIGHT_DECAY  = 0.05
DROP_PATH     = 0.1
MIXUP_PROB    = 0.0 if QUICK_MODE else 0.5
LABEL_SMOOTH  = 0.1

TTA_SCALES    = [96, 128]      # 候选尺度；最终用哪些由 8.1 的 OOF 选优决定
TTA_FLIPS     = [None, 'h']    # 人脸不做上下翻转，所以只有原图与水平翻转

EXPECTED_TEST_N = 7065
TIME_BUDGET_MIN = 540

# 评分指标：任务二明确说明按"预测准确率"评分
SEL_METRIC    = 'acc'          # 'acc' 或 'f1_macro'

BACKBONES = [
    dict(tag='r18',    name='resnet18.a1_in1k',              lr_scale=1.0),
    dict(tag='r34',    name='resnet34.a1_in1k',              lr_scale=1.0),
    dict(tag='effb0',  name='efficientnet_b0.ra_in1k',       lr_scale=1.0),
    dict(tag='cnxt_t', name='convnext_tiny.fb_in22k_ft_in1k', lr_scale=1.0),
    # 老师思路第 4 条点名要"注意力机制与 Transformer"，所以这里加一个带真实自注意力的主干。
    # **不用 Swin/ViT** —— 这不是偏好，是在本机 timm 1.0.30 上实测出来的：
    #   swin_tiny_patch4_window7_224 在"输入尺寸 != 构造时的 img_size"时直接
    #   AssertionError: Input height (96) doesn't match model (224).
    #   （加 dynamic_img_size=True 也没用；img_size=96 只能跑 96、跑到 128 一样断言失败）
    #   而本 notebook 的 TTA 要在 TTA_SCALES=[96,128] 两个尺度上前向，
    #   所以 Swin 会**在训练跑完几个小时之后**才崩 —— 这正是必须先在本地试出来的那类坑。
    #   实测能同时通过 96 与 128 的注意力型主干只有
    #   mobilevit_s / mobilevit_xs / xcit_tiny_12_p16_224（CoAtNet、LeViT、PiT 的位置编码都不行）。
    # 选 mobilevit_s：MobileViT 块里是真正的多头自注意力；卷积 + 注意力的混合结构
    # 与另外 4 个纯 CNN 的错误相关性低，比再堆一个 ResNet 更值钱。
    # lr_scale=0.7：带注意力的主干对学习率比 CNN 敏感，同 LR 容易在早期发散。
    dict(tag='mvit_s', name='mobilevit_s', lr_scale=0.7,
         candidates=['mobilevit_s', 'mobilevit_xs']),
]
if QUICK_MODE:
    BACKBONES = BACKBONES[:1]


# ====================== code cell 3/16 ======================
# ---- 由上面的配置派生出来的运行期变量（不参与调参，故不放进配置格）----
OUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('./runs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_AMP = torch.cuda.is_available()
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.485, 0.456, 0.406]   # 仍是 ImageNet 的 mean/std：主干是在 ImageNet 预训练的
STD  = [0.229, 0.224, 0.225]

def autocast_ctx():
    return torch.cuda.amp.autocast(enabled=USE_AMP) if USE_AMP else nullcontext()

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(SEED)
torch.backends.cudnn.benchmark = True

print(f'QUICK_MODE={QUICK_MODE}  folds={N_FOLDS}  epochs={EPOCHS}  amp={USE_AMP}')
print(f'IMG_SIZE={IMG_SIZE}  CACHE_SIZE={CACHE_SIZE}  SEL_METRIC={SEL_METRIC}')
print(f'OUT_DIR = {OUT_DIR}')
print(f'集成规模 = {len(BACKBONES)} 主干 x {N_FOLDS} 折 = {len(BACKBONES)*N_FOLDS} 个模型')


# ====================== code cell 4/16 ======================
CLASS_ORDER = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # 0..5，任务说明规定
CLASS2IDX   = {c: i for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = 6

# 每类训练张数（实测，用来做数据完整性断言；数字对不上说明数据被截断或换了版本）
EXPECTED_TRAIN_PER_CLASS = {'Angry': 3963, 'Fear': 4097, 'Happy': 7192,
                            'Neutral': 4959, 'Sad': 4862, 'Surprise': 3202}

TRAIN_SUBDIRS = {'Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'}
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def find_data_root():
    """训练目录 = 同时含 6 个类别子目录的那个 train；测试目录是它的同级 test。

    不能换成"递归找名字叫 train 的目录" —— 那样会撞上 Kaggle 其它数据集里的
    train 目录。这里的要求是**类别子目录名必须正好是这 6 个**，所以先递归找
    名字长得像这个数据集的目录，再逐个验证，最后才用一个很弱的兜底。
    """
    def _cls_of(d):
        try:
            return {q.name for q in Path(d).iterdir() if q.is_dir()}
        except Exception:
            return set()

    def _ok(p):
        return ((p / 'train').is_dir() and (p / 'test').is_dir()
                and _cls_of(p / 'train') == TRAIN_SUBDIRS)

    cands = []

    # 1) 常见挂载点（Kaggle 比赛数据一般是 /kaggle/input/<dataset>/...；本地放在工作目录下）
    for p in [Path('/kaggle/input/fer_data/fer_data'),
              Path('/kaggle/input/fer_data'),
              Path('/kaggle/input') / 'fer2013',
              Path('fer_data/fer_data'),
              Path('./fer_data'),
              Path('../fer_data/fer_data'),
              Path('../_inbox/fer/fer_data/fer_data')]:
        if _ok(p):
            cands.append(p)

    # 2) 递归找名字里带 fer 的目录 / 名字叫 fer_data 的目录
    for base in ['/kaggle/input', '.', '..']:
        b = Path(base)
        if not b.exists():
            continue
        try:
            hits = [q for q in b.rglob('fer_data') if q.is_dir()]
            hits += [q for q in b.rglob('*fer*') if q.is_dir()]
        except Exception:
            continue
        for q in hits:
            if _ok(q):
                cands.append(q)
            if _ok(q.parent):
                cands.append(q.parent)

    # 3) 兜底：Kaggle 比赛的挂载点不一定带 "fer" 字样（例如
    #    /kaggle/input/<比赛slug>/train/<类名>/），所以把 /kaggle/input 下三层以内的
    #    目录都检一遍。放在最后，前面更具体的候选仍然优先。
    def _walk(base, depth):
        try:
            kids = [q for q in Path(base).iterdir() if q.is_dir()]
        except Exception:
            return
        for q in kids:
            yield q
            if depth > 0:
                yield from _walk(q, depth - 1)

    for base in ['/kaggle/input']:
        if not Path(base).exists():
            continue
        for q in _walk(base, 3):
            if _ok(q):
                cands.append(q)
            if _ok(q.parent):
                cands.append(q.parent)

    if not cands:
        raise FileNotFoundError(
            '没有找到数据集：需要一个目录，它下面同时有 train/ 和 test/，'
            f'且 train/ 的子目录名正好是 {sorted(TRAIN_SUBDIRS)}。'
            '请检查 Add Input 里是否挂载了比赛数据。')
    return cands[0]

DATA_ROOT = find_data_root()
TRAIN_DIR = DATA_ROOT / 'train'
TEST_DIR  = DATA_ROOT / 'test'

found = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
if set(found) != set(CLASS_ORDER):
    raise ValueError(
        '训练集类别与任务说明不一致，绝不能继续往下跑！\n'
        f'  实际找到 = {found}\n'
        f'  任务要求 = {CLASS_ORDER}\n'
        '  注意：类别顺序不是字母序，也不要按公开 FER2013 的编号硬套。')
assert found == sorted(CLASS_ORDER), f'类别集合相同但顺序异常: {found}'

def list_images(d):
    return sorted([p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT])

train_paths, train_labels = [], []
per_class_n = {}
for c in CLASS_ORDER:
    fs = list_images(TRAIN_DIR / c)
    per_class_n[c] = len(fs)
    train_paths += fs
    train_labels += [CLASS2IDX[c]] * len(fs)

test_paths = list_images(TEST_DIR)
test_names = [p.name for p in test_paths]
train_labels = np.array(train_labels, dtype=np.int64)

print(f'DATA_ROOT  = {DATA_ROOT}')
print(f'训练图 {len(train_paths)} 张  测试图 {len(test_paths)} 张')
print('每类数量 :', per_class_n)
print('标签分布 :', dict(zip(CLASS_ORDER, np.bincount(train_labels, minlength=NUM_CLASSES))))

for c in CLASS_ORDER:
    _exp = EXPECTED_TRAIN_PER_CLASS[c]
    if per_class_n[c] != _exp:
        print(f'[警告] {c} 应有 {_exp} 张，实际 {per_class_n[c]} 张 —— 数据集版本可能不同。')
assert len(test_paths) == EXPECTED_TEST_N, f'测试集应有 {EXPECTED_TEST_N} 张，实际 {len(test_paths)}'
assert len(set(test_names)) == len(test_names), '测试集里有重名文件，ID 会撞车'


# ====================== code cell 5/16 ======================
def find_sample_submission(expected_ids):
    """在 /kaggle/input 下递归找表头为 ID,Emotion 的样例提交；找不到就退回本地路径。

    只认表头 + 行数 + 自己是不是自带重复 ID，不认文件名 —— 这样
    (a) 能找到文件名不叫 submission 的官方样例（本任务的样例叫 submission.csv，
        但 Kaggle 上官方样例的命名并不保证），
    (b) 不会误抓我们自己上一轮写出来的结果文件（它的名字里也有 submission，
        但它每行 `ID,Emotion` 的 ID 集合和行数**故意**与本题一致，所以这里
        额外要求「样例自带的 ID 在它内部不重复」，用来挡住明显损坏的候选）。
    """
    n_expect = len(list(expected_ids))
    for base in ['/kaggle/input', '.', '..', 'reference']:
        b = Path(base)
        if not b.exists():
            continue
        try:
            cands = sorted(b.rglob('*.csv'))
        except Exception:
            continue
        for q in cands:
            try:
                head = pd.read_csv(q, nrows=0)
            except Exception:
                continue
            cols = {c.strip().lower(): c for c in head.columns}
            if 'id' not in cols or 'emotion' not in cols:
                continue
            try:
                df = pd.read_csv(q)
            except Exception:
                continue
            if len(df) != n_expect:
                continue
            ids = df[cols['id']].astype(str)
            if not ids.is_unique:
                continue
            return q, df
    return None, None

sample_path, sample_df = find_sample_submission(test_names)
if sample_path is not None:
    print(f'找到官方样例提交: {sample_path}  (shape={sample_df.shape})')
    print(f'  表头 = {list(sample_df.columns)}')
    _dir_head = test_names[:3]
    _smp_head = sample_df[sample_df.columns[0]].astype(str).head(3).tolist()
    print(f'  目录顺序前 3 个 = {_dir_head}')
    print(f'  样例顺序前 3 个 = {_smp_head}')
    print(f'  两者一致? {_dir_head == _smp_head}  '
          f'{"（一致，但第 9 节仍按 ID 对齐）" if _dir_head == _smp_head else "（不一致 -> 必须按 ID 对齐！）"}')
else:
    print('[警告] 没有找到官方样例提交 CSV（表头需为 ID,Emotion）。')
    print('       第 9 节将退回按文件名排序输出，ID 集合仍会校验。')


# ====================== code cell 6/16 ======================
def load_gray_rgb(path):
    """打开任意图 -> 8 位灰度 -> 复制成 3 通道 RGB。返回 PIL.Image（mode='RGB'）。"""
    img = Image.open(path)
    img = img.convert('L')          # 兜底：非灰度图也统一成灰度
    return img.convert('RGB')       # 复制成 3 通道，匹配 ImageNet 预训练主干


# ====================== code cell 7/16 ======================
# 可视化校验：每个类抽 1 张，画成 1x6，标注「类名 (编号)」。
# 这一格的意义不是好看 —— 是肉眼确认「文件夹名 -> 类别编号」的映射没接错，
# 因为提交里的 Emotion 是整数，一旦映射错位，准确率会直接掉到随机水平附近。
set_seed(SEED)
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(3 * NUM_CLASSES, 3.6))
for i, c in enumerate(CLASS_ORDER):
    fs = list_images(TRAIN_DIR / c)
    p = fs[np.random.randint(len(fs))]
    img = load_gray_rgb(p)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f'{c} ({CLASS2IDX[c]})\n{img.size[0]}x{img.size[1]} {img.mode}', fontsize=9)
plt.suptitle('每个类各抽 1 张：确认「文件夹名 -> Emotion 编号」映射正确', fontsize=12)
plt.tight_layout(); plt.show()

# 顺手打印形态统计：原图应当全是 48x48 单通道；出现别的尺寸/模式就说明数据里有异常图
from collections import Counter
sizes, modes = Counter(), Counter()
for c in CLASS_ORDER:
    for p in list_images(TRAIN_DIR / c)[:60]:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
print('抽样图像尺寸分布 :', dict(sizes))
print('抽样图像原始模式 :', dict(modes))
print('（补充：上面 imshow 的是 convert 之后的 RGB，原始模式见本行统计）')


# ====================== code cell 8/16 ======================
class ImageCache:
    """把预处理后的图像缓存在内存里：[N, S, S, 3] uint8"""
    def __init__(self, paths, size=CACHE_SIZE, desc=''):
        self.paths = list(paths)
        self.size = size
        n = len(self.paths)
        self.data = np.empty((n, size, size, 3), dtype=np.uint8)
        for i, p in enumerate(tqdm(self.paths, desc=f'cache {desc} ({n})', leave=False)):
            img = load_gray_rgb(p)
            if img.size != (size, size):
                img = img.resize((size, size), Image.BILINEAR)
            self.data[i] = np.asarray(img, dtype=np.uint8)

    def __len__(self):
        return self.data.shape[0]

    def pil(self, i):
        return Image.fromarray(self.data[i])

_CACHES = {}
def get_cache(paths, role, size=CACHE_SIZE):
    key = (role, size)
    if key not in _CACHES:
        _CACHES[key] = ImageCache(paths, size, desc=role)
        mb = _CACHES[key].data.nbytes / 1024**2
        print(f'[cache] {role:<5s} n={len(paths)}  size={size}  '
              f'占用 {mb:.1f} MB  (uint8, {_CACHES[key].data.nbytes/len(paths)/1024:.1f} KB/张)')
    return _CACHES[key]

_t = time.time()
train_cache = get_cache(train_paths, 'train')
test_cache  = get_cache(test_paths,  'test')
print(f'缓存训练+测试集共耗时 {time.time()-_t:.1f}s（后续 epoch 不再重复这个开销）')


# ====================== code cell 9/16 ======================
class CacheDataset(Dataset):
    def __init__(self, cache, indices, labels, transform):
        self.cache = cache
        self.indices = list(indices)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(self.cache.pil(j)), self.labels[i]

def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        # 注意：这里**没有** RandomVerticalFlip —— 人脸不做上下翻转
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

def eval_transform(size=IMG_SIZE):
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

def make_loader(cache, indices, labels, transform, shuffle, bs=BATCH_SIZE, workers=NUM_WORKERS):
    ds = CacheDataset(cache, indices, labels, transform)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle,
                      persistent_workers=bool(workers > 0))

def make_folds(labels, n_folds, seed):
    labels = np.asarray(labels)
    if n_folds <= 1:
        idx = np.arange(len(labels))
        tr, va = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
        return [(tr, va)]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))

FOLDS = make_folds(train_labels, N_FOLDS, SEED)
for k, (tr, va) in enumerate(FOLDS):
    print(f'fold {k}: train={len(tr)}  val={len(va)}')


# ====================== code cell 10/16 ======================
def rand_bbox(h, w, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, y1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
    x2, y2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)
    return int(x1), int(y1), int(x2), int(y2)

def mixup_cutmix(x, y_soft, prob=MIXUP_PROB, alpha_mix=0.4,
                 alpha_cut=1.0, switch=0.5):
    """x:[B,3,H,W]  y_soft:[B,C] one-hot/软标签 -> (mixed_x, mixed_soft_target)"""
    if prob <= 0 or np.random.rand() > prob:
        return x, y_soft
    idx = torch.randperm(x.size(0), device=x.device)
    y2 = y_soft[idx]
    if np.random.rand() < switch:
        lam = float(np.random.beta(alpha_mix, alpha_mix))
        x = lam * x + (1.0 - lam) * x[idx]
        y = lam * y_soft + (1.0 - lam) * y2
    else:
        lam = float(np.random.beta(alpha_cut, alpha_cut))
        x1, y1, x2, y2_ = rand_bbox(x.size(2), x.size(3), lam)
        if x2 - x1 > 0 and y2_ - y1 > 0:
            x[:, :, y1:y2_, x1:x2] = x[idx, :, y1:y2_, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2_ - y1) / (x.size(2) * x.size(3)))
        y = lam * y_soft + (1.0 - lam) * y2
    return x, y

class SoftTargetLoss(nn.Module):
    """软标签交叉熵；gamma>0 时等价于软标签 Focal Loss。"""
    def __init__(self, gamma=0.0, smoothing=LABEL_SMOOTH, num_classes=NUM_CLASSES):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.C = num_classes

    def smooth(self, t):
        if self.smoothing <= 0:
            return t
        return t * (1.0 - self.smoothing) + self.smoothing / self.C

    def forward(self, logits, target_soft):
        t = self.smooth(target_soft)
        logp = F.log_softmax(logits, dim=-1)
        loss = -(t * logp)
        if self.gamma > 0:
            loss = loss * (1.0 - logp.exp()).pow(self.gamma)
        return loss.sum(dim=-1).mean()


# ====================== code cell 11/16 ======================
def create_backbone(cfg, drop_path=DROP_PATH):
    """按配置尝试加载主干权重；全部失败返回 (None, None)。

    除了"权重能不能拿到"，这里还多做一步**输入尺寸预检**：对 IMG_SIZE 和
    max(TTA_SCALES) 各前向一次空张量，任一失败就跳过该主干。

    为什么需要它：本机 timm 1.0.30 实测，Swin 这类带固定位置编码/窗口假设的主干在
    输入尺寸 != 构造时的 img_size 时会抛
        AssertionError: Input height (96) doesn't match model (224).
    而 TTA 要在两个尺度上前向 —— 没有预检的话，这个错会在**训练几个小时之后**才爆，
    整个 Kaggle 运行白跑。有了预检，它退化成启动时的一行 [--] 跳过。
    """
    _probe_sizes = sorted({int(IMG_SIZE), int(max(TTA_SCALES))})
    for nm in cfg.get('candidates', [cfg['name']]):
        try:
            bb = timm.create_model(nm, pretrained=True, num_classes=0,
                                   drop_path_rate=drop_path)
        except Exception as e:
            print(f'  [--] {nm}: {type(e).__name__}: {str(e)[:90]}')
            continue
        # 尺寸预检：用 eval 模式，避免 BN running stats 被空张量污染。
        _was_training = bb.training
        bb.eval()
        _bad = None
        for _s in _probe_sizes:
            try:
                with torch.no_grad():
                    bb(torch.zeros(1, 3, _s, _s))
            except Exception as e:
                _bad = f'{_s}px -> {type(e).__name__}: {str(e)[:70]}'
                break
        bb.train(_was_training)
        if _bad:
            print(f'  [--] {nm}: 尺寸预检失败（{_bad}），跳过该主干')
            continue
        print(f'  [ok] backbone = {nm}   尺寸预检通过 {_probe_sizes}')
        return bb, nm
    print('  [!!] 所有候选权重都不可用，跳过该主干')
    return None, None

class FERNet(nn.Module):
    def __init__(self, backbone, num_classes=NUM_CLASSES, drop=0.2):
        super().__init__()
        self.backbone = backbone
        feat = getattr(backbone, 'num_features', None)
        if feat is None:
            with torch.no_grad():
                feat = backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.feat_dim = int(feat)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.backbone(x)))

def build_optimizer(model, lr_head=LR_HEAD, lr_backbone=LR_BACKBONE,
                    wd=WEIGHT_DECAY, lr_scale=1.0):
    head, bb = [], []
    for n, p in model.named_parameters():
        (head if n.startswith('fc') else bb).append(p)
    groups = []
    if bb:
        groups.append({'params': bb, 'lr': lr_backbone * lr_scale})
    if head:
        groups.append({'params': head, 'lr': lr_head})
    return torch.optim.AdamW(groups, weight_decay=wd)

def lr_lambda(step, total, warmup, min_ratio=0.02):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


# ====================== code cell 12/16 ======================
# 模型结构自检：不下载权重，只确认「主干 -> 分类头 -> logits」这条链路通的，
# 且 IMG_SIZE 这个分辨率下主干能正常输出特征维度。
_m = FERNet(timm.create_model('resnet18', pretrained=False, num_classes=0))
_trainable = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(f'模型结构自检: 输出 {tuple(_m(torch.randn(2, 3, IMG_SIZE, IMG_SIZE)).shape)}, '
      f'可训练参数 {_trainable/1e6:.2f}M')
del _m


# ====================== code cell 13/16 ======================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast_ctx():
            out = model(x)
        preds.append(out.float().argmax(1).cpu().numpy())
        gts.append(y.numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    return (accuracy_score(gts, preds),
            f1_score(gts, preds, average='macro', zero_division=0),
            preds, gts)

def score_of(y_true, y_pred, metric=SEL_METRIC):
    """统一的打分口径。默认按任务二的要求用准确率；也可切成 macro-F1。"""
    if metric == 'acc':
        return float((y_true == y_pred).mean())
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))

def tta_view_keys(scales, flips):
    """TTA 的每一个"视角"给一个稳定的字符串 key，形如 '96|h'。"""
    return [f'{size}|{flip or "orig"}' for size in scales for flip in flips]

@torch.no_grad()
def predict_tta_view_probs(model, cache, indices, device,
                           scales=(IMG_SIZE,), flips=(None,), bs=64):
    """逐视角返回概率，不平均。

    返回 dict: key -> [len(indices), NUM_CLASSES]。key 见 tta_view_keys()。

    为什么要逐视角而不是直接平均：TTA 不是"加了就一定涨"。把不同分辨率的视角
    混进来会改变输入的有效尺度，而模型是在单一分辨率（IMG_SIZE + 1.14x resize +
    center crop）下训出来的，多出来的尺度可能反而掉分。之前实测就有过这种回退。
    所以这里把每个视角单独存下来，交给 OOF 去"用数据选"，而不是拍脑袋定。
    """
    model.eval()
    idx = list(indices)
    out = {}
    for size in scales:
        base = eval_transform(size)
        for flip in flips:
            key = f'{size}|{flip or "orig"}'
            acc = []
            for s in range(0, len(idx), bs):
                chunk = idx[s:s + bs]
                xb = torch.stack([base(cache.pil(j)) for j in chunk]).to(device)
                if flip and 'h' in flip:
                    xb = torch.flip(xb, dims=[3])
                if flip and 'v' in flip:
                    xb = torch.flip(xb, dims=[2])
                with autocast_ctx():
                    logit = model(xb)
                acc.append(F.softmax(logit.float(), dim=-1).cpu().numpy())
            out[key] = np.concatenate(acc, axis=0)
    return out

@torch.no_grad()
def predict_tta_probs(model, cache, indices, device,
                      scales=(IMG_SIZE,), flips=(None,), bs=64):
    """返回 [len(indices), NUM_CLASSES] 的 TTA 平均概率（所有视角等权）"""
    views = predict_tta_view_probs(model, cache, indices, device,
                                   scales=scales, flips=flips, bs=bs)
    stack = np.stack(list(views.values()), axis=0)
    return stack.mean(axis=0)

def train_one_fold(cfg, fold, tr_idx, va_idx, device, train_cache, verbose=True):
    """训练单个 fold，返回 {'model','best_score','best_acc','best_f1','history'}"""
    # 这里不能用内建 hash()：Python 3 对 str 的 hash 每个进程都掺了随机盐
    # （PEP 456，PYTHONHASHSEED 默认随机），同一个 notebook 每次跑出来的种子都不同，
    # 结果就不可复现（同一份代码两次跑分数不一样，交作业和查错都会很难受）。
    # 改用 md5 前 8 位十六进制：跨进程、跨机器、跨版本都稳定。
    set_seed(SEED + 100 * fold + (int(hashlib.md5(cfg['tag'].encode()).hexdigest()[:8], 16) % 97))

    bb, bb_name = create_backbone(cfg)
    if bb is None:
        return None
    model = FERNet(bb).to(device)

    train_loader = make_loader(train_cache, tr_idx, train_labels[tr_idx],
                               train_transform(), shuffle=True)
    val_loader = make_loader(train_cache, va_idx, train_labels[va_idx],
                             eval_transform(), shuffle=False)

    criterion = SoftTargetLoss(gamma=0.0)
    opt = build_optimizer(model, lr_scale=cfg.get('lr_scale', 1.0))
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, total_steps, warmup_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best = {'score': -1.0, 'acc': 0.0, 'f1': 0.0, 'state': None, 'epoch': -1}
    history = {'loss': [], 'val_acc': [], 'val_f1': []}
    it = 0
    for ep in range(EPOCHS):
        model.train()
        run_loss, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f'{cfg["tag"]} f{fold} ep{ep+1}/{EPOCHS}',
                    leave=False, disable=not verbose)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y_soft = F.one_hot(y, NUM_CLASSES).float().to(device)
            xm, ym = mixup_cutmix(x, y_soft)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(xm)
                loss = criterion(logits, ym)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            it += 1
            run_loss += loss.item() * x.size(0); n_seen += x.size(0)
            pbar.set_postfix(loss=f'{run_loss/max(1,n_seen):.3f}',
                             lr=f'{opt.param_groups[0]["lr"]:.2e}')

        # evaluate 一次就把 acc / macro-F1 / 预测 / 真值都拿到；
        # 选优分数一律走 score_of()，这样 SEL_METRIC 一改，全流程口径一致地变。
        va, vf, va_pred, va_true = evaluate(model, val_loader, device)
        vs = score_of(va_true, va_pred, metric=SEL_METRIC)
        history['loss'].append(run_loss / max(1, n_seen))
        history['val_acc'].append(va); history['val_f1'].append(vf)
        if vs > best['score']:
            best = {'score': vs, 'acc': va, 'f1': vf, 'epoch': ep,
                    'state': {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if verbose:
            print(f'  [{cfg["tag"]} f{fold}] ep{ep+1:02d} '
                  f'loss={run_loss/max(1,n_seen):.3f} val_acc={va:.4f} val_f1={vf:.4f} '
                  f'(选优依据 {SEL_METRIC}={vs:.4f})')

    model.load_state_dict(best['state'])
    return {'model': model, 'best_score': best['score'],
            'best_acc': best['acc'], 'best_f1': best['f1'],
            'best_epoch': best['epoch'], 'history': history, 'backbone': bb_name}


# ====================== code cell 14/16 ======================
t_start = time.time()
N_TRAIN, N_TEST = len(train_paths), len(test_paths)

oof_sum   = np.zeros((N_TRAIN, NUM_CLASSES))
oof_cnt   = np.zeros(N_TRAIN)
test_sum  = np.zeros((N_TEST, NUM_CLASSES))
test_cnt  = 0

# 逐视角累加（key 形如 '96|h'）。TTA 视角的取舍不用手调，交给 8.1 的 OOF 去选。
VIEW_KEYS = tta_view_keys(TTA_SCALES, TTA_FLIPS)
oof_view_sum  = {k: np.zeros((N_TRAIN, NUM_CLASSES)) for k in VIEW_KEYS}
test_view_sum = {k: np.zeros((N_TEST,  NUM_CLASSES)) for k in VIEW_KEYS}

summary_rows = []
per_backbone_oof = {}
histories = {}
fold_secs = []
stopped_early = False

for cfg in BACKBONES:
    tag = cfg['tag']
    print(f'\n{"="*74}\n[主干 {tag}] {cfg["name"]}\n{"="*74}')

    for fold, (tr_idx, va_idx) in enumerate(FOLDS):
        # ---- 时间预算闸门：宁可少集成，也不能超时丢掉全部结果 ----
        elapsed_min = (time.time() - t_start) / 60
        if fold_secs:
            # 取"最近一折"与"历史均值"的较大者来估下一折：
            # 纯均值在"刚换到更慢的主干"时会低估（前面跑的是便宜的 resnet18），
            # 最近一折在"刚换到更快的主干"时会高估。时间闸门要的是安全的一侧，所以取 max。
            est_min = float(max(fold_secs[-1], float(np.mean(fold_secs)))) / 60
            if elapsed_min + est_min > TIME_BUDGET_MIN:
                print(f'\n[时间预算] 已用 {elapsed_min:.0f} min，再跑一折预计还要 {est_min:.0f} min，'
                      f'将超过预算 {TIME_BUDGET_MIN} min —— 停止训练。')
                print(f'          已完成 {test_cnt} 个模型，用它们出提交文件。')
                stopped_early = True
                break

        t0 = time.time()
        try:
            res = train_one_fold(cfg, fold, tr_idx, va_idx, DEVICE, train_cache)
        except Exception as _e:
            # 一折崩掉不该带走整个 cell。现实里最可能的是显存碎片导致的 CUDA OOM，
            # 也可能是某个主干在前向里不兼容（尺寸预检只探了空张量，没探真实数据）。
            # 打印出来、跳过这个主干剩下的折，让后面的主干照常跑。
            import traceback as _tb
            print(f'  [!!] {tag} f{fold} 抛异常，跳过该主干的剩余折：'
                  f'{type(_e).__name__}: {_e}')
            _tb.print_exc()
            break
        if res is None:
            print(f'  [skip] {tag} 权重不可用，跳过')
            break
        model = res['model']

        # 验证集 TTA 与测试集用完全相同的 scales/flips，
        # 这样 OOF 指标估计的就是"提交时那套推断流程"的表现，而不是另一套。
        # 逐视角存下来，最后用 OOF 选一个最好的视角子集（见 8.1）。
        val_views = predict_tta_view_probs(model, train_cache, va_idx, DEVICE,
                                           scales=TTA_SCALES, flips=TTA_FLIPS)
        for k, p in val_views.items():
            oof_view_sum[k][va_idx] += p
        val_probs = np.mean(list(val_views.values()), axis=0)   # 全视角平均，仅用于单主干对比
        oof_sum[va_idx] += val_probs
        oof_cnt[va_idx] += 1
        per_backbone_oof.setdefault(tag, np.zeros((N_TRAIN, NUM_CLASSES)))[va_idx] += val_probs

        val_pred = val_probs.argmax(1)
        tta_acc = accuracy_score(train_labels[va_idx], val_pred)
        tta_f1  = f1_score(train_labels[va_idx], val_pred, average='macro', zero_division=0)

        test_views = predict_tta_view_probs(model, test_cache, range(N_TEST), DEVICE,
                                            scales=TTA_SCALES, flips=TTA_FLIPS)
        for k, p in test_views.items():
            test_view_sum[k] += p
        tp = np.mean(list(test_views.values()), axis=0)
        test_sum += tp
        test_cnt += 1

        row = dict(backbone=tag, fold=fold,
                   best_val_acc=res['best_acc'], best_val_f1=res['best_f1'],
                   tta_val_acc=tta_acc, tta_val_f1=tta_f1,
                   best_epoch=res['best_epoch'] + 1, secs=round(time.time() - t0, 1))
        summary_rows.append(row)
        histories[(tag, fold)] = res['history']
        fold_secs.append(row['secs'])
        print(f'  [{tag} f{fold}] no-TTA acc={res["best_acc"]:.4f} f1={res["best_f1"]:.4f} | '
              f'TTA acc={tta_acc:.4f} f1={tta_f1:.4f} | {row["secs"]}s')
        print(f'   已用 {(time.time()-t_start)/60:.0f} min / 预算 {TIME_BUDGET_MIN} min，'
              f'已完成 {test_cnt}/{len(BACKBONES)*len(FOLDS)} 个模型')

        # 中途保险：每折结束就把当前集成结果落盘，而且**两份都写**。
        #
        # 这是实测教训换来的：`submission.csv` 原先只在最后一个 cell 里写，而那个 cell
        # 排在"OOF 汇总"cell 之后。只要训练 cell 被掐断（Kaggle 时间上限、手动停止）
        # 或在某个主干上抛异常，后面两个 cell 一个都不会跑，手上就只剩一份没做视角选优的
        # `submission_partial.csv` —— 实测那一份的排行榜分数是 0.653。
        # 现在每折都把同样的内容同时写成 `submission.csv`，保证任何时刻都有一份完整、
        # 可提交的文件；最后一个 cell 会用视角选优的结果把它覆盖掉。
        #
        # 行序也必须跟官方样例一致：Kaggle 有可能按行序而不是按 ID 对齐，
        # 而这份文件存在的唯一意义就是"最后一格挂了还能交"，不能自带顺序风险。
        _part_pred = test_sum.argmax(1).astype(int).tolist()
        _id2emo = dict(zip(test_names, _part_pred))
        _part_ids, _part_emo = test_names, _part_pred
        if sample_df is not None:
            _id_col = [c for c in sample_df.columns if c.strip().lower() == 'id'][0]
            _cand = sample_df[_id_col].astype(str).tolist()
            if set(_cand) == set(test_names):
                _part_ids, _part_emo = _cand, [_id2emo[i] for i in _cand]
                print(f'   [兜底] submission.csv + submission_partial.csv '
                      f'已按官方样例顺序写出（{len(_part_ids)} 行）')
            else:
                print('   [兜底] 样例 ID 集合与测试集不一致，退回目录顺序')
        _part = pd.DataFrame({'ID': _part_ids, 'Emotion': _part_emo})
        for _fn in ('submission_partial.csv', 'submission.csv'):
            _part.to_csv(OUT_DIR / _fn, index=False,
                         encoding='utf-8', lineterminator='\r\n')
        del model
        torch.cuda.empty_cache()

    print(f'  累计用时 {(time.time()-t_start)/60:.1f} min')
    if stopped_early:
        break

summary = pd.DataFrame(summary_rows)
print(f'\n总耗时 {(time.time()-t_start)/60:.1f} min')
print(f'集成模型数 = {test_cnt}')
display(summary)


# ====================== code cell 15/16 ======================
oof_probs = oof_sum / np.maximum(oof_cnt[:, None], 1)
oof_pred  = oof_probs.argmax(1)

# 关键：只统计"至少进过一次验证折"的样本（oof_cnt > 0）。
# 若直接对全量算指标，那些 cnt==0 的样本概率恒为 0 向量，argmax 会一律落在类别 0，
# 把指标压低且不报错（实验一 QUICK_MODE 下就踩了这个坑：真实 0.75 被显示成 0.31）。
covered = oof_cnt > 0
n_cov = int(covered.sum())
oof_acc = accuracy_score(train_labels[covered], oof_pred[covered])
oof_f1  = f1_score(train_labels[covered], oof_pred[covered], average='macro', zero_division=0)
oof_score = score_of(train_labels[covered], oof_pred[covered])
print(f'==== OOF (每张被覆盖的图都由没见过它的模型预测) ====')
print(f'覆盖样本 = {n_cov}/{N_TRAIN} ({n_cov/N_TRAIN:.1%})')
if n_cov < N_TRAIN:
    miss = np.bincount(train_labels[~covered], minlength=NUM_CLASSES)
    print(f'[警告] 有 {N_TRAIN - n_cov} 张从未进入任何验证折，已从 OOF 指标中剔除。')
    print(f'        未覆盖样本的类别分布: {dict(zip(CLASS_ORDER, miss.tolist()))}')
    if N_FOLDS > 1:
        print('        5 折正常情况下应当全覆盖；出现缺口说明某折没跑成功（如权重下载失败被 skip）。')
    else:
        print('        QUICK_MODE 下只跑 1 折，OOF 天然只覆盖验证折，指标仅供流程验证，不代表最终成绩。')
print(f'accuracy = {oof_acc:.4f}    macro-F1 = {oof_f1:.4f}    '
      f'（当前 SEL_METRIC={SEL_METRIC}，选优分数 = {oof_score:.4f}）')

# ---- 用 OOF 选一个最好的 TTA 视角子集 ----
# 多视图 TTA 不保证更好：把 IMG_SIZE 之外的尺度混进来会改变输入的有效尺度，
# 而模型是在单一分辨率（IMG_SIZE，配 1.14x resize + center crop）下训出来的，
# 额外尺度可能反而掉分（这正是"训练/测试分辨率不一致"的经典坑，FixRes 讲的也是这件事）。
# 所以不猜，直接把每个候选子集在 OOF 上打分，挑最好的那个去跑测试集。
def _oof_score_of(keys):
    # 逐视角累加的是"概率"（每折一份），所以分母既要去掉模型数、也要去掉视角数。
    # 少了这个除法，argmax 出来的标签**仍然是对的**（正的常数缩放不改变 argmax），
    # 但打印出来的置信度和存盘的 *_probs.npy 就不再是概率了 —— 那是会骗人的。
    p = sum(oof_view_sum[k] for k in keys) / (np.maximum(oof_cnt, 1)[:, None] * len(keys))
    return score_of(train_labels[covered], p[covered].argmax(1))

_cands = [(k,) for k in VIEW_KEYS]                                   # 每个单视角
for _size in TTA_SCALES:                                            # 同一尺度下的全部翻转
    _ks = tuple(k for k in VIEW_KEYS if k.startswith(f'{_size}|'))
    if len(_ks) > 1:
        _cands.append(_ks)
_ks_train = tuple(k for k in VIEW_KEYS if k.startswith(f'{IMG_SIZE}|'))
if _ks_train:                                                       # 训练分辨率下的全部翻转
    _cands.append(_ks_train)
_cands.append(tuple(VIEW_KEYS))                                     # 全部视角（旧行为）
_seen, _uniq = set(), []
for _c in _cands:
    if _c not in _seen:
        _seen.add(_c); _uniq.append(_c)
_cands = _uniq

_rowlist = [dict(views='+'.join(_c), n_views=len(_c),
                 oof_score=_oof_score_of(_c),
                 oof_acc=accuracy_score(train_labels[covered],
                                        (sum(oof_view_sum[k] for k in _c) /
                                         (np.maximum(oof_cnt, 1)[:, None] * len(_c)))[covered].argmax(1)))
            for _c in _cands]
view_tbl = pd.DataFrame(_rowlist).sort_values(['oof_score', 'n_views'],
                                              ascending=[False, True]).reset_index(drop=True)

print(f'\n所有候选视角子集在 OOF 上的打分（SEL_METRIC={SEL_METRIC}，仅供参考 —— 不要照抄第一名，原因见下）:')
display(view_tbl)

# 规则：**先在"尺度"这一层选择，再在该尺度内平均它的全部翻转；绝不挑单个视角。**
#
# 这条规则不是偏好，是实验一上两次独立跑逼出来的（本任务沿用同一条规则，
# 但下面提到的具体数值是**实验一植物任务**上测出来的，本任务还没有自己的两跑对照）：
#   实验一第 1 跑：单视角第一名是 288|hv (0.9010)，而 224|orig 只有 0.8974
#   实验一第 2 跑：单视角第一名是 224|orig (0.9133)，而 288|hv 掉到 0.8968
# **两次跑挑出的"最佳单视角"完全相反。** 这就是选择偏差的签名：8 个单视角各自的
# OOF 分数都带噪声（500 张图上 macro-F1 的标准误就有 1.3 个点），
# 取其中最大的那个，涨的是噪声，不是真实能力。
#
# 而"同一尺度下全部翻转的平均"是稳定量 —— 实验一两次跑一致地给出 224 那一组更好
# （224 四翻转 = 0.8991 / 0.9050，288 四翻转 = 0.8905 / 0.8943，方向一致）。
# 机理上也讲得通：288 是训练分辨率之外的尺度，测试时换尺度会引入
# train-test resolution discrepancy。
#
# 本任务的对应量：TTA_SCALES=[96,128]，96 与 IMG_SIZE 一致、128 是训练之外的尺度，
# 所以"该不该带上 128"正是这里要让 OOF 回答的问题。另外本任务训练集 28275 张、
# 5 折每折约 5655 张验证，OOF 的噪声比实验一（100 张验证折）小一个量级，
# 尺度层的比较会更可信；但即便如此也不回到"挑单个视角"。
_scale_sets = []
for _size in TTA_SCALES:
    _ks = tuple(k for k in VIEW_KEYS if k.startswith(f'{_size}|'))
    if _ks:
        _scale_sets.append((_size, _ks))
_scale_score = {s: _oof_score_of(ks) for s, ks in _scale_sets}
_best_scale = max(_scale_score, key=_scale_score.get)
BEST_VIEWS = dict(_scale_sets)[_best_scale]
print('\n每个尺度的"全部翻转"在 OOF 上的得分（尺度层比较比单视角稳得多）:')
_scale_rows = []
for s, ks in _scale_sets:
    _p = sum(oof_view_sum[k] for k in ks) / (np.maximum(oof_cnt, 1)[:, None] * len(ks))
    _scale_rows.append(dict(scale=s, n_views=len(ks), oof_score=_scale_score[s],
                            oof_acc=accuracy_score(train_labels[covered], _p[covered].argmax(1)),
                            oof_f1=f1_score(train_labels[covered], _p[covered].argmax(1),
                                            average='macro', zero_division=0)))
display(pd.DataFrame(_scale_rows).sort_values('oof_score', ascending=False).reset_index(drop=True))
print(f'>> 选中尺度 {_best_scale}（该尺度全部翻转的 OOF {SEL_METRIC} = {_scale_score[_best_scale]:.4f}）')
print(f'>> 提交用视角 = {"+".join(BEST_VIEWS)}（该尺度的全部 {len(BEST_VIEWS)} 个翻转取平均）')
print(f'   参考：全 {len(VIEW_KEYS)} 视角平均 = {_oof_score_of(tuple(VIEW_KEYS)):.4f}')

cm = confusion_matrix(train_labels[covered], oof_pred[covered])
plt.figure(figsize=(7.0, 5.8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
plt.title(f'Ensemble OOF Confusion Matrix\nAcc={oof_acc:.4f}  Macro-F1={oof_f1:.4f}')
plt.xlabel('predicted'); plt.ylabel('true')
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.show()

print('\n分类报告 (OOF):')
print(classification_report(train_labels[covered], oof_pred[covered],
                            labels=list(range(NUM_CLASSES)),
                            target_names=CLASS_ORDER, digits=4))

# 单主干对比
rows = []
for tag, p in per_backbone_oof.items():
    m = p.sum(1) > 0
    rows.append(dict(backbone=tag,
                     oof_acc=accuracy_score(train_labels[m], p[m].argmax(1)),
                     oof_f1=f1_score(train_labels[m], p[m].argmax(1),
                                     average='macro', zero_division=0)))
rows.append(dict(backbone='ENSEMBLE', oof_acc=oof_acc, oof_f1=oof_f1))
print('\n各主干 vs 集成 (OOF):')
display(pd.DataFrame(rows).sort_values('oof_acc', ascending=False))

# 训练曲线
n_show = len(histories)
if n_show:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for (tag, fold), h in histories.items():
        axes[0].plot(h['loss'], label=f'{tag} f{fold}', alpha=0.8)
        axes[1].plot(h['val_acc'], label=f'{tag} f{fold}', alpha=0.8)
        axes[2].plot(h['val_f1'], label=f'{tag} f{fold}', alpha=0.8)
    axes[0].set_title('train loss'); axes[0].set_xlabel('epoch'); axes[0].legend(fontsize=7)
    axes[1].set_title('val accuracy (no TTA)'); axes[1].set_xlabel('epoch'); axes[1].legend(fontsize=7)
    axes[2].set_title('val macro-F1 (no TTA)'); axes[2].set_xlabel('epoch'); axes[2].legend(fontsize=7)
    plt.tight_layout(); plt.show()

# 最容易被混淆的类别对
off = cm.copy(); np.fill_diagonal(off, 0)
pairs = [(CLASS_ORDER[i], CLASS_ORDER[j], int(off[i, j]))
         for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and off[i, j] > 0]
pairs.sort(key=lambda t: -t[2])
print('\n最容易混淆的类别对 (真实 -> 预测: 次数):')
for a, b, c in pairs[:8]:
    print(f'  {a:<10s} -> {b:<10s} {c}')


# ====================== code cell 16/16 ======================
# 分母要同时去掉"模型数"和"视角数"：*_view_sum[k] 是"每折一份 × 每个视角一份"的累加。
# 漏掉视角数不改变预测标签，但会让下面打印的平均置信度变成 3.3 之类的非概率值。
final_probs = sum(test_view_sum[k] for k in BEST_VIEWS) / (max(1, test_cnt) * len(BEST_VIEWS))
oof_probs   = sum(oof_view_sum[k] for k in BEST_VIEWS) / (np.maximum(oof_cnt, 1)[:, None] * len(BEST_VIEWS))
print(f'提交用的 TTA 视角 = {"+".join(BEST_VIEWS)}（由 8.1 的 OOF 选中）')
np.save(OUT_DIR / 'test_probs.npy', final_probs)
np.save(OUT_DIR / 'oof_probs.npy', oof_probs)
pd.DataFrame(summary).to_csv(OUT_DIR / 'fold_summary.csv', index=False)
view_tbl.to_csv(OUT_DIR / 'tta_view_scores.csv', index=False)
# 逐视角的概率和也一并存下来：以后想改选优规则或换一组视角重出提交文件，
# 直接读这个 npz 就够了，不必把 5 折 x 5 主干重训一遍。
np.savez_compressed(OUT_DIR / 'tta_view_probs.npz',
                    oof_cnt=oof_cnt, test_cnt=test_cnt,
                    **{f'oof__{k}': v for k, v in oof_view_sum.items()},
                    **{f'test__{k}': v for k, v in test_view_sum.items()})

# 兜底断言：一个模型都没跑成的话，test_sum 全 0、argmax 全落在类别 0，
# 会静默产出一份"7065 行全是 Angry(0)"的垃圾提交。这里必须拦住。
# 最常见的原因是 Kaggle 的 Internet 没开 → timm 下不到预训练权重 → 所有主干被 skip。
assert test_cnt > 0, (
    '没有任何模型训练成功，无法生成提交文件。\n'
    '最常见原因：Kaggle 的 Settings → Internet 没有打开，timm 无法下载预训练权重，'
    f'{len(BACKBONES)} 个主干全部被跳过。请打开 Internet 后重跑。')

test_pred = final_probs.argmax(1).astype(int)
sub = pd.DataFrame({'ID': test_names, 'Emotion': test_pred})
print(f'预测概率矩阵 {final_probs.shape}，平均置信度 {final_probs.max(1).mean():.4f}')
print('\n预测分布 (Emotion 值 -> 张数):')
_vc = sub['Emotion'].value_counts().sort_index()
for _i, _n in _vc.items():
    print(f'  {int(_i)} = {CLASS_ORDER[int(_i)]:<10s} {int(_n):5d} 张  ({_n/len(sub):5.1%})')

if sub['Emotion'].nunique() == 1:
    print(f'\n[严重警告] {len(sub)} 个预测全是同一个类别'
          f'（Emotion={int(sub["Emotion"].iloc[0])} = {CLASS_ORDER[int(sub["Emotion"].iloc[0])]}）。')
    print('           集成结果退化，多半是某个环节出错了，不要把这份文件交上去。')
if sub['Emotion'].nunique() < NUM_CLASSES:
    missing = [CLASS_ORDER[i] for i in range(NUM_CLASSES) if i not in set(sub['Emotion'])]
    print(f'\n[提示] 以下类别一张都没被预测到：{missing}（7065 张里可能确实没有，也可能模型偏了）。')

# ---- 按官方样例的 ID 顺序对齐 ----
sample_path2, sample_df2 = find_sample_submission(test_names)
if sample_path2 is not None:
    id_col = [c for c in sample_df2.columns if c.strip().lower() == 'id'][0]
    sample_ids = sample_df2[id_col].astype(str).tolist()
    print(f'\n官方样例提交 = {sample_path2}   行数 = {len(sample_ids)}   唯一 ID = {len(set(sample_ids))}')

    # 断言 1：行数 = EXPECTED_TEST_N
    assert len(sample_ids) == EXPECTED_TEST_N, \
        f'样例提交行数应为 {EXPECTED_TEST_N}，实际 {len(sample_ids)}'
    # 断言 2：没有重复 ID
    assert len(set(sample_ids)) == len(sample_ids), '样例提交里有重复 ID，无法按 ID 对齐'
    # 断言 3：预测 ID 集合与样例 ID 集合完全一致
    _pred_set, _samp_set = set(test_names), set(sample_ids)
    if _pred_set != _samp_set:
        raise AssertionError(
            '预测 ID 集合与样例 ID 集合不一致，绝不能靠目录顺序硬凑！\n'
            f'  预测独有 {len(_pred_set - _samp_set)} 个，例如 {sorted(_pred_set - _samp_set)[:5]}\n'
            f'  样例独有 {len(_samp_set - _pred_set)} 个，例如 {sorted(_samp_set - _pred_set)[:5]}')

    # 先按 ID 建 dict 再 reindex：这样行顺序完全由样例决定
    id2emo = dict(zip(sub['ID'].astype(str), sub['Emotion'].astype(int)))
    sub = pd.DataFrame({'ID': sample_ids,
                        'Emotion': [id2emo[i] for i in sample_ids]})
    print('已按官方样例的 ID 顺序逐行对齐（先建 ID->Emotion 字典，再 reindex）')
else:
    print('\n[警告] 未找到官方样例提交，按文件名排序输出（ID 集合仍会校验）')
    sub = sub.sort_values('ID').reset_index(drop=True)

# ---- 最终格式断言 ----
assert len(sub) == EXPECTED_TEST_N, f'行数应为 {EXPECTED_TEST_N}，实际 {len(sub)}'
assert list(sub.columns) == ['ID', 'Emotion'], f'表头必须是 ID,Emotion，实际 {list(sub.columns)}'
assert sub['ID'].is_unique and sub['ID'].notna().all(), 'ID 有缺失或重复'
assert set(sub['ID']) == set(test_names), 'ID 集合与测试集不一致'
assert sub['Emotion'].notna().all(), '存在未预测的样本'
assert sub['Emotion'].dtype.kind in 'iu', f'Emotion 必须是整数，实际 dtype={sub["Emotion"].dtype}'
assert set(sub['Emotion']).issubset(set(range(NUM_CLASSES))), '出现了非法 Emotion 值'

sub_path = OUT_DIR / 'submission.csv'
# lineterminator='\r\n' -> 强制 CRLF，与官方样例逐字节一致。
#   **不要写 newline=''** —— pandas 的 to_csv() 根本没有这个参数（那是标准库 csv 模块的），
#   传了会在"训练全跑完、就差写文件"的那一刻抛 TypeError（本 notebook 的本地冒烟就是这么崩的）。
#   pandas 内部已按 newline='' 打开句柄，所以这里给 '\r\n' 不会被再翻译成 '\r\r\n'。
# encoding='utf-8' -> 不带 BOM（不要用 utf-8-sig，BOM 会让评测方把第一列名读成 '\ufeffID'）
sub.to_csv(sub_path, index=False, encoding='utf-8', lineterminator='\r\n')
print(f'\n已写出: {sub_path}   形状={sub.shape}   表头={list(sub.columns)}')
print(sub.head(10).to_string(index=False))
print('...')
print(sub.tail(3).to_string(index=False))

# 复核落盘文件的行尾与 BOM（这两项出错的话评测方会直接判格式错）
_raw = sub_path.read_bytes()
print(f'\n落盘自检: CRLF={_raw.count(b"\r\n")} 行, 裸 LF={_raw.count(b"\n") - _raw.count(b"\r\n")} 行, '
      f'BOM={_raw[:3] == b"\\xef\\xbb\\xbf"}')
