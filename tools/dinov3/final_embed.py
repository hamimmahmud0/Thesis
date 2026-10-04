"""Embed (a) curated set = sel100 minus dropped IDs, (b) 1000 random objects per class (disjoint from all used so far).
Run on remote after `submissions.log` exists:  ./remote.py run python3 final_embed.py"""
import json, os, random, subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
Image.MAX_IMAGE_PIXELS = None
SRC = "hf://buckets/hamimmahmud0/SAM_COCO_v1_b2_3024/annotate"
MODEL, WORK, OUT = "/root/work/model", "/root/work", "outputs/final"
N_RANDOM, MIN_SIDE, PAD, SEED = 1000, 16, 0.1, 7
os.makedirs(OUT, exist_ok=True)
coco = json.load(open(f"{WORK}/instances.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}; imgs = {i["id"]: i for i in coco["images"]}
by_ann = {a["id"]: a for a in coco["annotations"]}

sel = json.load(open("outputs/sel100/selection.json"))
dropped = set(json.loads(open("webapp/submissions.log").read().strip().splitlines()[-1])["dropped"])
curated = [s for s in sel if s["id"] not in dropped]
print("curated", len(curated), "dropped", sorted(dropped))
used = {s["ann_id"] for s in sel} | {m["ann_id"] for m in json.load(open("outputs/samples.json"))} \
     | {m["ann_id"] for m in json.load(open("outputs/test_samples.json"))}
rng = random.Random(SEED); rand = []
for cid, name in cats.items():
    if name not in ("Rickshaw", "Motorcycle"): continue
    pool = [a for a in coco["annotations"] if a["category_id"] == cid and a["id"] not in used
            and a["bbox"][2] >= MIN_SIDE and a["bbox"][3] >= MIN_SIDE]
    print(name, "random pool", len(pool))
    for a in rng.sample(pool, N_RANDOM):
        rand.append({"id": f"X{a['id']}", "cat": name, "ann_id": a["id"], "file": imgs[a["image_id"]]["file_name"], "bbox": a["bbox"], "score": a["score"]})

def fetch(fn):
    p = f"{WORK}/images/{fn}"
    if not os.path.exists(p):
        subprocess.run(["hf", "buckets", "cp", f"{SRC}/images/{fn}", p], check=True, capture_output=True)
files = sorted({s["file"] for s in curated + rand}); print("frames needed", len(files))
with ThreadPoolExecutor(8) as ex: list(ex.map(fetch, files))

proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float16, device_map="auto").eval()
def embed(items, name):
    crops = {}
    byf = defaultdict(list)
    for k, s in enumerate(items): byf[s["file"]].append(k)
    feats = [None] * len(items)
    buf = []   # (k, crop)
    def flush():
        if not buf: return
        with torch.no_grad():
            inp = proc(images=[c for _, c in buf], return_tensors="pt").to(model.device, torch.float16)
            f = torch.nn.functional.normalize(model(**inp).pooler_output.float(), dim=1).cpu().numpy()
        for (k, _), v in zip(buf, f): feats[k] = v
        buf.clear()
    for fn, ks in byf.items():
        im = Image.open(f"{WORK}/images/{fn}").convert("RGB")
        for k in ks:
            x, y, w, h = items[k]["bbox"]
            buf.append((k, im.crop((int(max(0, x - PAD * w)), int(max(0, y - PAD * h)), int(min(im.width, x + w * (1 + PAD))), int(min(im.height, y + h * (1 + PAD)))))))
            if len(buf) == 16: flush()
    flush()
    np.save(f"{OUT}/{name}_features.npy", np.stack(feats)); json.dump(items, open(f"{OUT}/{name}_meta.json", "w"))
    print("embedded", name, len(items), flush=True)
embed(curated, "curated"); embed(rand, "random")
