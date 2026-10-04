"""Class-vs-class similarity of DINOv3 ViT-7B (SAT-493M) embeddings of top-score object crops.

Runs on the remote GPU box:  ./remote.py run python3 class_similarity.py
"""
import json, os, random, subprocess
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

SRC = "hf://buckets/hamimmahmud0/SAM_COCO_v1_b2_3024/annotate"
MODEL = "/root/work/model"  # downloaded by `./remote.py model`
N, MIN_SCORE, PAD, SEED = 20, 0.95, 0.1, 0
WORK, OUT = "/root/work", "outputs"
os.makedirs(f"{WORK}/images", exist_ok=True); os.makedirs(f"{OUT}/crops", exist_ok=True)

def cp(src, dst):
    if not os.path.exists(dst):
        subprocess.run(["hf", "buckets", "cp", src, dst], check=True, capture_output=True)

cp(f"{SRC}/annotations/instances.json", f"{WORK}/instances.json")
coco = json.load(open(f"{WORK}/instances.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}

rng = random.Random(SEED)
picked = {}
for cid, name in cats.items():
    pool = [a for a in coco["annotations"] if a["category_id"] == cid and a["score"] >= MIN_SCORE
            and a["bbox"][2] >= 16 and a["bbox"][3] >= 16]
    if len(pool) < N:  # rare class: fall back to the N*5 best-scoring
        pool = sorted((a for a in coco["annotations"] if a["category_id"] == cid),
                      key=lambda a: -a["score"])[: N * 5]
    picked[cid] = rng.sample(pool, min(N, len(pool)))
    print(f"{name}: pool={len(pool)} picked={len(picked[cid])}")

proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()

feats, labels, meta = [], [], []
for cid, anns in picked.items():
    for a in anns:
        fn = imgs[a["image_id"]]["file_name"]
        cp(f"{SRC}/images/{fn}", f"{WORK}/images/{fn}")
        im = Image.open(f"{WORK}/images/{fn}").convert("RGB")
        x, y, w, h = a["bbox"]
        box = (max(0, x - PAD * w), max(0, y - PAD * h), min(im.width, x + w * (1 + PAD)), min(im.height, y + h * (1 + PAD)))
        crop = im.crop(tuple(int(v) for v in box))
        tag = f"{cats[cid]}_{sum(m['cat'] == cats[cid] for m in meta):02d}"
        crop.save(f"{OUT}/crops/{tag}.png")
        with torch.no_grad():
            inp = proc(images=crop, return_tensors="pt").to(model.device, torch.float16)
            f = model(**inp).pooler_output.float()[0]
        feats.append(torch.nn.functional.normalize(f, dim=0).cpu()); labels.append(cid)
        meta.append({"ann_id": a["id"], "file": fn, "cat": cats[cid], "score": a["score"], "tag": tag, "bbox": a["bbox"]})

F = torch.stack(feats); S = (F @ F.T).numpy(); L = np.array(labels); ids = list(cats)
M = np.zeros((len(ids), len(ids)))
for i, ci in enumerate(ids):
    for j, cj in enumerate(ids):
        blk = S[np.ix_(L == ci, L == cj)]
        if ci == cj:  # exclude self-similarity
            n = blk.shape[0]; M[i, j] = (blk.sum() - np.trace(blk)) / max(n * (n - 1), 1)
        else:
            M[i, j] = blk.mean()

names = [cats[i] for i in ids]
np.save(f"{OUT}/similarity_matrix.npy", M); np.save(f"{OUT}/features.npy", F.numpy()); json.dump(meta, open(f"{OUT}/samples.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 7)); im_ = ax.imshow(M, cmap="viridis"); plt.colorbar(im_)
ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w", fontsize=8)
ax.set_title("Mean cosine similarity (DINOv3 ViT-7B SAT, CLS)"); plt.tight_layout(); plt.savefig(f"{OUT}/similarity_confusion.png", dpi=150)
print("\n" + " " * 12 + "".join(f"{n[:9]:>10}" for n in names))
for n, row in zip(names, M): print(f"{n[:11]:<12}" + "".join(f"{v:10.3f}" for v in row))
