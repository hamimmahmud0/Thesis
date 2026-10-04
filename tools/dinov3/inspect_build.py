"""Builds webapp/public/inspect/{data.json,thumbs/*.jpg} for the results inspector (runs on remote)."""
import json, os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
Image.MAX_IMAGE_PIXELS = None
O, W, OUT = "outputs", "/root/work", "webapp/public/inspect"; CL = ["Rickshaw", "Motorcycle"]; PAD = 0.1
os.makedirs(OUT + "/thumbs", exist_ok=True)
F = np.load(f"{O}/features.npy"); M = json.load(open(f"{O}/samples.json")); keep = [i for i, m in enumerate(M) if m["cat"] in CL]
old = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1).fit(F[keep], [CL.index(M[i]["cat"]) for i in keep])
cm_ = json.load(open(f"{O}/final/curated_meta.json")); Xc = np.load(f"{O}/final/curated_features.npy")
new = LogisticRegression(max_iter=5000, class_weight="balanced", C=1).fit(Xc, [CL.index(s["cat"]) for s in cm_])
items = []
for setname in ("random", "curated"):
    meta = json.load(open(f"{O}/final/{setname}_meta.json")); X = np.load(f"{O}/final/{setname}_features.npy")
    po, pn = old.predict_proba(X)[:, 1], new.predict_proba(X)[:, 1]
    for s, a, b in zip(meta, po, pn):
        items.append({"set": setname, "id": s["id"], "cat": s["cat"], "score": round(s["score"], 3), "w": s["bbox"][2], "h": s["bbox"][3],
                      "file": s["file"], "bbox": s["bbox"], "p_old": round(float(a), 4), "p_new": round(float(b), 4)})
byf = {}
for k, it in enumerate(items): byf.setdefault(it["file"], []).append(k)
for fn, ks in byf.items():
    im = Image.open(f"{W}/images/{fn}").convert("RGB")
    for k in ks:
        x, y, w, h = items[k]["bbox"]
        c = im.crop((int(max(0, x - PAD * w)), int(max(0, y - PAD * h)), int(min(im.width, x + w * (1 + PAD))), int(min(im.height, y + h * (1 + PAD)))))
        c.thumbnail((200, 200)); c.save(f"{OUT}/thumbs/{items[k]['set'][0]}_{items[k]['id']}.jpg", quality=85)
json.dump(items, open(f"{OUT}/data.json", "w"), separators=(",", ":"))
print("items", len(items), "thumbs", len(os.listdir(OUT + "/thumbs")))
