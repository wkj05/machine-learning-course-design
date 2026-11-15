# -*- coding: utf-8 -*-
"""任务三 数据层：letterbox 到 512x512、坐标双向换算、随机裁剪增广，并做"增广后点是否还落在黄斑上"的视觉校验。

坐标约定（三层，全部显式命名，避免任务二那种"顺序/坐标对不上"的事故）：
  orig  : 原图像素 (0..W, 0..H)
  box   : letterbox 后 512x512 画布上的像素
  crop  : 随机裁剪再 resize 回 512x512 后的像素（= 模型输入）
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
OUT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\t3_look')
OUT.mkdir(parents=True, exist_ok=True)
S = 512          # 模型输入边长
PAD_V = 0        # letterbox 填充值（黑色，与眼底图背景一致）


@dataclass
class Item:
    idx: int
    path: str
    W: int
    H: int
    X: float          # orig
    Y: float
    s: float          # orig -> 512 缩放
    padx: float
    pady: float

    @property
    def bx(self):     # orig -> box
        return self.X * self.s + self.padx

    @property
    def by(self):
        return self.Y * self.s + self.pady

    def box_to_orig(self, bx, by):
        return (bx - self.padx) / self.s, (by - self.pady) / self.s


def load_items():
    gt = {}
    with open(ROOT / 'fovea_localization_train_GT.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            gt[int(r['data'])] = (float(r['Fovea_X']), float(r['Fovea_Y']))
    items = []
    for i in sorted(gt):
        p = ROOT / 'train' / f'{i:04d}.jpg'
        with Image.open(p) as im:
            W, H = im.size
        s = S / max(W, H)
        nw, nh = round(W * s), round(H * s)
        items.append(Item(i, str(p), W, H, gt[i][0], gt[i][1], s,
                          (S - nw) / 2, (S - nh) / 2))
    return items


def load_box(it):
    """返回 512x512 的 uint8 画布。"""
    with Image.open(it.path) as im:
        im = im.convert('RGB')
        nw, nh = round(it.W * it.s), round(it.H * it.s)
        im = im.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new('RGB', (S, S), (PAD_V, PAD_V, PAD_V))
        canvas.paste(im, (round(it.padx), round(it.pady)))
        return np.asarray(canvas, dtype=np.uint8)


def rand_augment(img, bx, by, rng):
    """随机裁剪 + resize + 翻转 + 颜色抖动。返回 (新图, 新点)，点留在 crop 坐标系。

    裁剪框以 (bx,by) 为中心加抖动，边长 = S * scale。这样黄斑几乎总在框内，
    同时框的偏移本身就是"目标不在画面中心"的增广 —— 正是推理时会遇到的分布。
    """
    scale = rng.uniform(0.78, 1.0)               # 裁边长 / S
    w = int(round(S * scale))
    jit = int(round(S * 0.16))                    # 中心抖动幅度
    cx = int(round(bx + rng.uniform(-jit, jit)))
    cy = int(round(by + rng.uniform(-jit, jit)))
    x0 = cx - w // 2
    y0 = cy - w // 2
    # 允许越界，用 fill 补黑（对应图外，合理）
    box_img = Image.fromarray(img)
    crop = box_img.crop((x0, y0, x0 + w, y0 + w)).resize((S, S), Image.BICUBIC)
    nx = (bx - x0) * S / w
    ny = (by - y0) * S / w

    if rng.random() < 0.5:                        # 水平翻转
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        nx = S - nx

    # 颜色抖动（不影响坐标）
    crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.82, 1.18))
    crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.82, 1.18))
    crop = ImageEnhance.Color(crop).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.3:                        # 随机 gamma
        a = np.asarray(crop, dtype=np.float32) / 255.0
        a = np.clip(a, 1e-4, 1.0) ** rng.uniform(0.8, 1.25)
        crop = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
    return np.asarray(crop, dtype=np.uint8), float(nx), float(ny)


if __name__ == '__main__':
    items = load_items()
    print(f'items={len(items)}')
    uniq = {}
    for it in items:
        uniq.setdefault((it.W, it.H), []).append(it.idx)
    for k, v in uniq.items():
        print(f'  {k}: n={len(v)}  s={S/max(k):.4f}  eg#{v[0]:04d} '
              f'box=({items[v[0]-1].bx:.1f},{items[v[0]-1].by:.1f})')
    # 反算校验：box -> orig 必须无损
    for it in items:
        X2, Y2 = it.box_to_orig(it.bx, it.by)
        assert abs(X2 - it.X) < 1e-6 and abs(Y2 - it.Y) < 1e-6, it.idx
    print('box<->orig 往返误差 < 1e-6  ✓')

    # 增广视觉校验
    rng = np.random.default_rng(0)
    picks = [1, 4, 9, 15, 34, 49]
    T = 340
    sheet = Image.new('RGB', (T * 3, T * 2), 'white')
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except Exception:
        font = ImageFont.load_default()
    cache = {it.idx: load_box(it) for it in items if it.idx in picks}
    for k, i in enumerate(picks):
        it = items[i - 1]
        img, nx, ny = rand_augment(cache[i], it.bx, it.by, rng)
        tx, ty = (k % 3) * T, (k // 3) * T
        sheet.paste(Image.fromarray(img).resize((T, T), Image.LANCZOS), (tx, ty))
        d = ImageDraw.Draw(sheet)
        x = nx / S * T
        y = ny / S * T
        d.line([(tx + x - 20, ty + y), (tx + x + 20, ty + y)], fill=(255, 0, 0), width=2)
        d.line([(tx + x, ty + y - 20), (tx + x, ty + y + 20)], fill=(255, 0, 0), width=2)
        d.text((tx + 4, ty + 3), f'#{i:04d} label=({nx:.0f},{ny:.0f})/512',
               fill=(0, 255, 0), font=font)
        d.rectangle([tx, ty, tx + T - 1, ty + T - 1], outline=(80, 80, 80))
    p = OUT / 'aug_check.png'
    sheet.save(p, optimize=True)
    print('[saved]', p)
