"""Pick 100 Rickshaw + 100 Motorcycle high-score objects, crop them, assign IDs (R001.., M001..).
Run on remote: ./remote.py run python3 select_100.py"""
import json, os, random, subprocess
from collections import Counter
from PIL import Image
SRC = "hf://buckets/hamimmahmud0/SAM_COCO_v1_b2_3024/annotate"
WORK, OUT = "/root/work", "outputs/sel100"
N, MIN_SCORE, PAD, MAX_PER_FRAME, SEED = 100, 0.9, 0.1, 3, 42
os.makedirs(OUT + "/crops", exist_ok=True)
coco = json.load(open(f"{WORK}/instances.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}; imgs = {i["id"]: i for i in coco["images"]}
rng = random.Random(SEED); sel = []
for cid, name in cats.items():
    if name not in ("Rickshaw", "Motorcycle"): continue
    pool = [a for a in coco["annotations"] if a["category_id"] == cid and a["bbox"][2] >= 24 and a["bbox"][3] >= 24]
    hi = [a for a in pool if a["score"] >= MIN_SCORE]
    pool = hi if len(hi) >= 3 * N else sorted(pool, key=lambda a: -a["score"])[: 3 * N]
    print(name, "pool", len(pool), "min score", min(a["score"] for a in pool))
    rng.shuffle(pool); per = Counter(); chosen = []
    for a in pool:   # spread over frames
        if per[a["image_id"]] < MAX_PER_FRAME:
            per[a["image_id"]] += 1; chosen.append(a)
        if len(chosen) == N: break
    for k, a in enumerate(chosen, 1):
        sel.append((("R" if name == "Rickshaw" else "M") + f"{k:03d}", name, a))
for oid, name, a in sel:
    fn = imgs[a["image_id"]]["file_name"]; p = f"{WORK}/images/{fn}"
    if not os.path.exists(p):
        subprocess.run(["hf", "buckets", "cp", f"{SRC}/images/{fn}", p], check=True, capture_output=True)
for fn in sorted({imgs[a["image_id"]]["file_name"] for _, _, a in sel}):
    im = Image.open(f"{WORK}/images/{fn}").convert("RGB")
    for oid, name, a in sel:
        if imgs[a["image_id"]]["file_name"] != fn: continue
        x, y, w, h = a["bbox"]
        im.crop((int(max(0, x - PAD * w)), int(max(0, y - PAD * h)), int(min(im.width, x + w * (1 + PAD))),
                 int(min(im.height, y + h * (1 + PAD))))).save(f"{OUT}/crops/{oid}.png")
json.dump([{"id": oid, "cat": name, "ann_id": a["id"], "file": imgs[a["image_id"]]["file_name"], "bbox": a["bbox"], "score": a["score"]}
           for oid, name, a in sel], open(f"{OUT}/selection.json", "w"), indent=1)
print("done", len(sel))
