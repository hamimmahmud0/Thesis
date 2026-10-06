"""Class-agnostic precision/recall of OBB predictions vs ground truth on chosen frames.
python eval_agnostic.py GT_DIR FRAMES(comma DJI ids) NAME=pred.json [NAME=pred.json ...] [--iou 0.5,0.3] [--conf 0.25]
GT_DIR = b1_small_annotated (images/<hash>__DJI_x.JPG + labels). Prediction json keys are the DJI stems (e.g. DJI_0614)."""
import glob, json, os, sys
import cv2, numpy as np
from PIL import Image
args = [x for x in sys.argv[1:]]
iou_list = [0.5, 0.3]; conf = 0.25
for k, f in (('--iou', 'iou'), ('--conf', 'conf')):
    if k in args:
        i = args.index(k); v = args[i + 1]; del args[i:i + 2]
        if f == 'iou': iou_list = [float(x) for x in v.split(',')]
        else: conf = float(v)
gt_dir, frames = args[0], args[1].split(','); models = [(a.split('=')[0], json.load(open(a.split('=', 1)[1]))) for a in args[2:]]

def piou(p, q):
    p = np.asarray(p, np.float32); q = np.asarray(q, np.float32)
    ap_, aq = cv2.contourArea(p), cv2.contourArea(q)
    if ap_ < 1 or aq < 1: return 0.0
    inter, _ = cv2.intersectConvexConvex(p, q)
    return inter / (ap_ + aq - inter + 1e-9)

gt = {}
for f in glob.glob(f'{gt_dir}/images/*.JPG'):
    stem = os.path.splitext(os.path.basename(f))[0]; dji = stem.split('__')[-1]
    if dji not in frames: continue
    W, H = Image.open(f).size; gt[dji] = []
    for l in open(f'{gt_dir}/labels/{stem}.txt'):
        v = l.split()
        if len(v) >= 9: gt[dji].append([(float(v[1 + 2 * j]) * W, float(v[2 + 2 * j]) * H) for j in range(4)])
n_gt = sum(len(v) for v in gt.values()); print(f'GT objects in {len(gt)} frames: {n_gt}')
for name, pj in models:
    for thr in iou_list:
        tp = fp = 0
        for dji, g in gt.items():
            dets = sorted([d for d in pj.get(dji, []) if d['conf'] >= conf], key=lambda d: -d['conf']); used = set()
            for d in dets:
                best, bj = 0, -1
                for j, q in enumerate(g):
                    if j in used: continue
                    u = piou(d['poly'], q)
                    if u > best: best, bj = u, j
                if best >= thr: tp += 1; used.add(bj)
                else: fp += 1
        P = tp / max(tp + fp, 1); R = tp / max(n_gt, 1)
        print(f'{name:<14} IoU>={thr}: dets={tp + fp:<5} TP={tp:<4} precision={P:.3f} recall={R:.3f} F1={2 * P * R / max(P + R, 1e-9):.3f}')
