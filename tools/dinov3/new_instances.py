"""Embed N fresh Rickshaw/Motorcycle crops (disjoint from the training samples) -> outputs/test_*.
Run on remote: ./remote.py run python3 new_instances.py   (needs outputs/samples.json already on remote)"""
import json, os, random, subprocess
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

SRC = "hf://buckets/hamimmahmud0/SAM_COCO_v1_b2_3024/annotate"
MODEL, WORK, OUT = "/root/work/model", "/root/work", "outputs"
CLASSES, N, MIN_SCORE, PAD, SEED = ["Rickshaw", "Motorcycle"], 10, 0.95, 0.1, 1
coco = json.load(open(f"{WORK}/instances.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}; imgs = {i["id"]: i for i in coco["images"]}
used = json.load(open(f"{OUT}/samples.json")); used_ids = {m["ann_id"] for m in used}; used_files = {m["file"] for m in used}
rng = random.Random(SEED); picked = []
for cid, name in cats.items():
    if name not in CLASSES: continue
    pool = [a for a in coco["annotations"] if a["category_id"] == cid and a["bbox"][2] >= 16 and a["bbox"][3] >= 16]
    hi = [a for a in pool if a["score"] >= MIN_SCORE]
    pool = hi if len(hi) >= 100 else sorted(pool, key=lambda a: -a["score"])[:100]   # same rule as training
    pool = [a for a in pool if a["id"] not in used_ids]
    fresh = [a for a in pool if imgs[a["image_id"]]["file_name"] not in used_files]   # also unseen frames
    src = fresh if len(fresh) >= N else pool
    print(name, "pool", len(pool), "unseen-frame", len(fresh))
    picked += [(name, a) for a in rng.sample(src, N)]

proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16, device_map="auto").eval()
os.makedirs(f"{OUT}/crops", exist_ok=True); feats, meta = [], []
for k, (name, a) in enumerate(picked):
    fn = imgs[a["image_id"]]["file_name"]; p = f"{WORK}/images/{fn}"
    if not os.path.exists(p):
        subprocess.run(["hf", "buckets", "cp", f"{SRC}/images/{fn}", p], check=True, capture_output=True)
    im = Image.open(p).convert("RGB"); x, y, w, h = a["bbox"]
    crop = im.crop((int(max(0, x - PAD * w)), int(max(0, y - PAD * h)), int(min(im.width, x + w * (1 + PAD))), int(min(im.height, y + h * (1 + PAD)))))
    tag = f"test_{name}_{sum(m['cat'] == name for m in meta):02d}"; crop.save(f"{OUT}/crops/{tag}.png")
    with torch.no_grad():
        f = model(**proc(images=crop, return_tensors="pt").to(model.device, torch.float16)).pooler_output.float()[0]
    feats.append(torch.nn.functional.normalize(f, dim=0).cpu().numpy())
    meta.append({"ann_id": a["id"], "file": fn, "cat": name, "score": a["score"], "tag": tag, "bbox": a["bbox"]})
np.save(f"{OUT}/test_features.npy", np.stack(feats)); json.dump(meta, open(f"{OUT}/test_samples.json", "w"), indent=1)
print("done", len(meta))
