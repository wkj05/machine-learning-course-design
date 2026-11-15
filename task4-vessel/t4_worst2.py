"""每个 P 下标对应哪张图：脚本里用 glob 的字典序，不是数字序（雷！）"""
import numpy as np, json
from pathlib import Path
from PIL import Image, ImageDraw
D = Path('_inbox/task4/segmentation'); E = 512
lab = sorted((D/'train/label').glob('*.jpg'), key=lambda p: int(p.stem))
img = sorted((D/'train/image').glob('*.jpg'), key=lambda p: int(p.stem))
gt = np.stack([np.asarray(Image.open(p).convert('L').resize((E, E), Image.BILINEAR)) > 0 for p in lab])
# 字典序 -> 数字序 的映射：P 的第 k 个对应数字序第 map[k] 个
lex = sorted(range(len(lab)), key=lambda i: lab[i].name)
pos = np.empty(len(lab), int)
for k, i in enumerate(lex): pos[i] = k     # 数字序 i -> 字典序位次
P = np.load('t4_oof_eg.npy'); n = len(lab)
Pm = np.stack([P[pos[i]] for i in range(n)])   # 重排成数字序
pm = Pm > 0.60
d = np.array([2*(pm[i] & gt[i]).sum()/max(pm[i].sum()+gt[i].sum(), 1) for i in range(n)])
order = np.argsort(d)
print('per-image Dice mean=%.4f median=%.4f' % (d.mean(), np.median(d)))
print('worst 6:', [(int(i)+1, round(float(d[i]), 4)) for i in order[:6]])
print('best 3 :', [(int(i)+1, round(float(d[i]), 4)) for i in order[-3:]])
new = (d.sum() - d[order[0]] - d[order[1]] + 2*np.median(d))/n
print('若最差2张提到 median -> %.4f (榜 +%.2f)' % (new, (new-d.mean())*100))
json.dump({'lex_to_num': [int(i)+1 for i in lex]}, open('_inbox/t4_oof_order.json', 'w'))
S = 420
canvas = Image.new('RGB', (4*S, 4*S), (0, 0, 0)); dr = ImageDraw.Draw(canvas)
for k, i in enumerate(order[:16]):
    a = np.asarray(Image.open(img[i]).convert('RGB').resize((E, E), Image.BILINEAR)).astype(np.float32)
    ov = a.copy()
    ov[gt[i]] = ov[gt[i]]*0.35 + np.array([255, 40, 40])*0.65
    ov[pm[i]] = ov[pm[i]]*0.35 + np.array([40, 255, 40])*0.65
    both = gt[i] & pm[i]; ov[both] = ov[both]*0.25 + np.array([255, 235, 60])*0.75
    canvas.paste(Image.fromarray(ov.astype(np.uint8)).resize((S, S), Image.LANCZOS), ((k % 4)*S, (k//4)*S))
    dr.text(((k % 4)*S+6, (k//4)*S+4), '#%02d Dice=%.3f' % (int(i)+1, d[i]), fill=(255, 255, 255))
canvas.save('_inbox/t3_look/t4_worst16.png'); print('saved; lex order =', [int(i)+1 for i in lex])
