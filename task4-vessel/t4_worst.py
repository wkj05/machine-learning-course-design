import sys, importlib.util
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('t4', '_inbox/t4_kaggle.py')
t4 = importlib.util.module_from_spec(spec); _a=sys.argv; sys.argv=['x']
spec.loader.exec_module(t4); sys.argv=_a

root = Path('_inbox/task4/segmentation')
labs = sorted([f for f in (root/'train'/'label').iterdir() if f.suffix.lower()=='.jpg'], key=lambda f: f.stem)
P = np.load('t4_oof_convnext.npy')
tiles=[]
for k in (3, 15, 0):                       # #4, #16 最差 + #1 正常对照
    lab = np.asarray(Image.open(labs[k]).convert('L'))
    gtv = (lab > 0)
    pm  = P[k] > 0.12
    pm  = t4.drop_small(pm, 30)
    g512 = t4.gt_at_eval(Image.open(labs[k]).convert('L'))
    rgb = np.asarray(Image.open(root/'train'/'image'/labs[k].name).convert('RGB')).astype(np.float32)
    # 血管叠加：GT 红、预测 绿（重合处黄）
    ov = rgb.copy()
    up = np.asarray(Image.fromarray(pm.astype(np.uint8)*255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
    ug = np.asarray(Image.fromarray(gtv.astype(np.uint8)*255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
    ov[ug] = 0.35*ov[ug] + np.array([255,0,0])*0.65
    ov[up] = 0.35*ov[up] + np.array([0,255,0])*0.65
    # 裁一块有代表性的 220x220 区域（取 GT 血管最多的位置）
    ys,xs = np.where(ug)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0=max(0,cy-110); x0=max(0,cx-110)
    crop = Image.fromarray(ov.astype(np.uint8)[y0:y0+220, x0:x0+220]).resize((440,440), Image.LANCZOS)
    d=ImageDraw.Draw(crop)
    d.text((6,4), f'#{k+1:02d}  Dice={t4.dice_at_eval(pm, g512):.4f}  红=GT 绿=预测', fill=(255,255,0))
    tiles.append(crop)
W=440*3; out=Image.new('RGB',(W,450),(0,0,0))
for j,t in enumerate(tiles): out.paste(t,(j*440,10))
out.save('_inbox/t3_look/t4_worst.png')
print('saved _inbox/t3_look/t4_worst.png')
for k in (3,15,0):
    print(f'#{k+1:02d}  GT血管占比={(t4.gt_at_eval(Image.open(labs[k]).convert("L"))).mean()*100:.2f}%')
