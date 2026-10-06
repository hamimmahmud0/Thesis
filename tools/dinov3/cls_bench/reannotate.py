#!/usr/bin/env python3
"""Shard worker: re-classify every detection of a COCO bucket dataset with frozen DINOv3 ViT-7B/16 (LVD-1689M) + LogReg.

Same recipe as the benchmark's best model: crop the bbox with 8 px padding (cv2, as in the exported training crops),
stretch to 224x224 with the model's own processor, fp16 forward, features = [CLS, mean of patch tokens (no registers)],
StandardScaler + LogisticRegression(C=0.01, class_weight=balanced), fit on ALL labelled crops (train+val+test).
Writes <out> = {"info":..., "preds": {ann_id: {"cls", "conf", "top3"}}} and syncs this shard's images to the target bucket.
"""
import argparse, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

PAD = 8


def crop(im, bbox):
    h, w = im.shape[:2]
    x, y, bw, bh = bbox
    x0, y0, x1, y1 = max(0, int(x) - PAD), max(0, int(y) - PAD), min(w, int(x + bw) + PAD), min(h, int(y + bh) + PAD)
    return im[y0:y1, x0:x1]


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hf://buckets/hamimmahmud0/SAM_COCO_b1/b1")
    ap.add_argument("--dst", default="", help="bucket path to sync this shard's images to, e.g. hf://buckets/u/name/b1")
    ap.add_argument("--train-feats", required=True)
    ap.add_argument("--model", default="facebook/dinov3-vit7b16-pretrain-lvd1689m")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--work", default="/kaggle/working/w")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many images")
    a = ap.parse_args()
    work = Path(a.work); (work / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not (work / "instances.json").exists():
        sh(f"hf buckets cp {a.src}/annotations/instances.json {work}/instances.json")
    coco = json.load(open(work / "instances.json"))
    by_img = {}
    for an in coco["annotations"]:
        by_img.setdefault(an["image_id"], []).append(an)
    imgs = sorted(coco["images"], key=lambda i: i["id"])
    loads = [0] * a.nshards; mine = []                       # balance shards by annotation count (greedy)
    for im in sorted(imgs, key=lambda i: -len(by_img.get(i["id"], []))):
        k = loads.index(min(loads)); loads[k] += len(by_img.get(im["id"], []))
        if k == a.shard:
            mine.append(im)
    mine.sort(key=lambda i: i["id"])
    if a.limit:
        mine = mine[:a.limit]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} images, {sum(len(by_img.get(i['id'], [])) for i in mine)} detections", flush=True)

    def fetch(im):
        p = work / "images" / im["file_name"]
        if not p.exists():
            sh(f"hf buckets cp {a.src}/images/{im['file_name']} {p}")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, mine))
    print(f"images downloaded {time.time() - t0:.0f}s", flush=True)

    z = np.load(a.train_feats, allow_pickle=True)
    Xtr = np.hstack([z["cls_stretch"], z["mean_stretch"]]); ytr = z["label"].astype(str)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000)).fit(Xtr, ytr)
    classes = list(clf.classes_)
    train_row = {Path(str(p)).stem.rsplit("_", 1)[1]: i for i, p in enumerate(z["ids"])}   # annotation id -> stored feature row
    print(f"classifier fit on {len(ytr)} crops, {len(classes)} classes, train acc {clf.score(Xtr, ytr):.3f}", flush=True)
    if a.shard == 0:
        joblib.dump({"pipeline": clf, "classes": classes, "recipe": "cls+mean / stretch / fp16, C=0.01 balanced, model=" + a.model},
                    work / "classifier.joblib")

    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, dtype=torch.float16, device_map="auto").eval()
    n_reg = getattr(model.config, "num_register_tokens", 0)
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def feats(crops):
        x = proc(images=crops, return_tensors="pt")["pixel_values"].to(model.device, torch.float16)
        h = model(pixel_values=x).last_hidden_state.float()
        return torch.cat([h[:, 0], h[:, 1 + n_reg:].mean(1)], 1).cpu().numpy()

    preds, cos, t1, done = {}, [], time.time(), 0
    for im in mine:
        frame = cv2.imread(str(work / "images" / im["file_name"]))
        if frame is None:
            raise SystemExit(f"cannot read {im['file_name']}")
        ans = by_img.get(im["id"], [])
        for i in range(0, len(ans), a.bs):
            chunk = ans[i:i + a.bs]
            crops = []
            for an in chunk:
                c = crop(frame, an["bbox"])
                if c.size == 0:
                    c = np.zeros((8, 8, 3), np.uint8)
                crops.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            f = feats(crops)
            assert np.isfinite(f).all(), "NaN/inf features"
            P = clf.predict_proba(f)
            for an, fi, p in zip(chunk, f, P):
                o = np.argsort(-p)[:3]
                preds[str(an["id"])] = {"cls": classes[o[0]], "conf": float(p[o[0]]), "top3": [[classes[j], float(p[j])] for j in o]}
                r = train_row.get(str(an["id"]))
                if r is not None:                                  # sanity: same crop recipe as the stored training features
                    s = Xtr[r]; cos.append(float(fi @ s / (np.linalg.norm(fi) * np.linalg.norm(s))))
        done += len(ans)
        print(f"{done} detections, {time.time() - t1:.0f}s", flush=True)
    info = {"shard": a.shard, "nshards": a.nshards, "images": len(mine), "detections": len(preds), "model": a.model,
            "seconds": round(time.time() - t0), "verify_n": len(cos),
            "verify_cos_min": min(cos) if cos else None, "verify_cos_mean": float(np.mean(cos)) if cos else None}
    json.dump({"info": info, "preds": preds}, open(a.out, "w"))
    print(json.dumps(info), flush=True)
    if a.dst:
        sh(f"hf buckets sync {work}/images {a.dst}/images")
        print("images synced", flush=True)


if __name__ == "__main__":
    main()
