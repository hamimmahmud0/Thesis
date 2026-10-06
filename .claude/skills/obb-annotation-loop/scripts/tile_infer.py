"""Tiled OBB inference: python tile_infer.py WEIGHTS IMAGES_DIR OUT.json [--tile 1024] [--overlap 0.25] [--conf 0.25] [--imgsz 1024] [--nms 0.5] [--max-det 1000]
Output: {image_stem: [{"cls": int, "name": str, "conf": float, "poly": [[x,y]*4]}]} in full-image pixel coords."""
import argparse, glob, json, os
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou


def nms_rotated(boxes, scores, thr):
    """Greedy rotated NMS on xywhr boxes (ProbIoU); returns kept indices, highest score first."""
    order = scores.argsort(descending=True); boxes = boxes[order]
    iou = batch_probiou(boxes, boxes).triu_(diagonal=1)
    keep = (iou.max(dim=0).values < thr).nonzero().squeeze(-1)
    return order[keep]

ap = argparse.ArgumentParser()
ap.add_argument('weights'); ap.add_argument('images'); ap.add_argument('out')
ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--overlap', type=float, default=0.25)
ap.add_argument('--conf', type=float, default=0.25); ap.add_argument('--imgsz', type=int, default=1024)
ap.add_argument('--nms', type=float, default=0.5); ap.add_argument('--max-det', type=int, default=1000); ap.add_argument('--device', default='cpu')
a = ap.parse_args()

def starts(n, t, stride):
    if n <= t: return [0]
    s = list(range(0, n - t, stride)); s.append(n - t); return s

m = YOLO(a.weights)
stride = int(a.tile * (1 - a.overlap)); res = {}
files = sorted(glob.glob(os.path.join(a.images, '*.JPG')) + glob.glob(os.path.join(a.images, '*.jpg')) + glob.glob(os.path.join(a.images, '*.png')))
for f in files:
    im = Image.open(f).convert('RGB'); W, H = im.size
    boxes, scores, clss, polys = [], [], [], []
    for y0 in starts(H, a.tile, stride):
        for x0 in starts(W, a.tile, stride):
            tile = im.crop((x0, y0, min(x0 + a.tile, W), min(y0 + a.tile, H)))
            r = m.predict(tile, imgsz=a.imgsz, conf=a.conf, device=a.device, max_det=a.max_det, verbose=False)[0]
            if r.obb is None or len(r.obb) == 0: continue
            xywhr = r.obb.xywhr.clone(); xywhr[:, 0] += x0; xywhr[:, 1] += y0
            p = r.obb.xyxyxyxy.clone(); p[..., 0] += x0; p[..., 1] += y0
            boxes.append(xywhr); scores.append(r.obb.conf.clone()); clss.append(r.obb.cls.clone()); polys.append(p)
    out = []
    if boxes:
        B, S, C, P = torch.cat(boxes), torch.cat(scores), torch.cat(clss), torch.cat(polys)
        keep = nms_rotated(B, S, a.nms)
        for i in keep.tolist():
            out.append({'cls': int(C[i]), 'name': m.names[int(C[i])], 'conf': float(S[i]), 'poly': P[i].tolist()})
    res[os.path.splitext(os.path.basename(f))[0]] = out
    print(os.path.basename(f), len(out), flush=True)
json.dump(res, open(a.out, 'w'))
