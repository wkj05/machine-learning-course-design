# -*- coding: utf-8 -*-
"""离线评估：拿 t3_local.py 存下的 5 个折模型，只改推理尺度（TTA），
看留一法 RMSE 能不能再降。不需要重训。"""
import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import t3_local as T

OUTTXT = Path(__file__).resolve().parent / "t3_tta_offline.txt"
_lines = []


def say(s=""):
    print(s)
    _lines.append(s)


def rmse(pred, true):
    d = np.linalg.norm(pred - true, axis=1)
    return float(np.sqrt(np.mean(d ** 2) / 2.0)), float(d.mean()), float(np.median(d)), float(d.max())


root = T.find_root()
gt = {}
with open(root / "fovea_localization_train_GT.csv", newline="") as f:
    for r in csv.DictReader(f):
        gt[int(r["data"])] = (float(r["Fovea_X"]), float(r["Fovea_Y"]))
ids = sorted(gt)
true_xy = np.array([gt[i] for i in ids], np.float64)
tr_items = [{"path": str(root / "train" / ("%04d.jpg" % i))} for i in ids]
te_ids = sorted(int(p.stem) for p in (root / "test").glob("*.jpg"))
te_items = [{"path": str(root / "test" / ("%04d.jpg" % i))} for i in te_ids]

folds = T.make_folds(len(ids), T.N_FOLDS, T.SEED)
say("root = %s   train %d   test %d   folds=%s" % (root, len(ids), len(te_ids),
                                                   [len(v) for v in folds]))

models = []
for fi in range(len(folds)):
    ck = torch.load(T.OUT / ("s1_0_%d.pt" % fi), map_location="cpu")
    cfg = ck["cfg"]
    m = T.HeatNet(cfg["name"], pretrained=False, ch=T.S1_CH,
                  out_stride=T.S1_STRIDE, drop_path=0.0)
    m.load_state_dict(ck["sd"])
    m = m.to(T.DEVICE).eval()
    models.append((m, cfg))
say("loaded %d fold models (%s)" % (len(models), models[0][1]["name"]))

CANVASES = [896, 1024, 1152, 1280]
oof = {}
te = {}
for cv in CANVASES:
    c_tr = T.CanvasCache(tr_items, cv)
    c_te = T.CanvasCache(te_items, cv)
    p_tr = np.zeros_like(true_xy)
    for fi, va in enumerate(folds):
        acc, _d = T.predict_s1([models[fi]], c_tr, list(va), flip_tta=True)
        p_tr[va] = acc
    p_te, _d = T.predict_s1(models, c_te, list(range(len(te_ids))), flip_tta=True)
    p_te /= float(len(models))
    oof[cv] = p_tr
    te[cv] = p_te
    r = rmse(p_tr, true_xy)
    say("canvas=%4d  OOF  RMSE=%7.3f  mean=%6.2f  median=%5.2f  max=%7.2f" %
        (cv, r[0], r[1], r[2], r[3]))
    del c_tr, c_te
    torch.cuda.empty_cache()

combos = [(1024,), (1024, 1152), (896, 1024, 1152), (1024, 1152, 1280),
          (896, 1024, 1152, 1280), (1024, 1280), (1152,)]
say()
say("=== 多尺度平均 ===")
best = None
for cs in combos:
    p = np.mean([oof[c] for c in cs], axis=0)
    r = rmse(p, true_xy)
    say("canvas%s  OOF RMSE=%7.3f  mean=%6.2f  median=%5.2f  max=%7.2f" %
        (list(cs), r[0], r[1], r[2], r[3]))
    if best is None or r[0] < best[0]:
        best = (r[0], cs)
say()
say("best combo = %s  RMSE=%.3f  (baseline 1024 = %.3f)" %
    (list(best[1]), best[0], rmse(oof[1024], true_xy)[0]))

# 落盘最优组合的测试集预测，方便直接出提交
np.save(T.OUT / "te_tta_best.npy", np.mean([te[c] for c in best[1]], axis=0))
json.dump({"combo": list(best[1]), "rmse": best[0],
           "te_ids": te_ids, "baseline1024": rmse(oof[1024], true_xy)[0]},
          open(T.OUT / "tta_best.json", "w"), indent=1)
say("saved te_tta_best.npy / tta_best.json")

OUTTXT.write_text("\n".join(_lines), encoding="utf-8")
