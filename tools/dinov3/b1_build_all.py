"""Build the full-dataset viewer data for the FILTERED b1 annotations (runs on the server).
Per frame: 2000px JPEG + ann/<name>.json (kept annotations + removed pedestrians flagged r=1)."""
import json, os, subprocess
from collections import defaultdict
from multiprocessing import Pool
import cv2, numpy as np
from pycocotools import mask as mu
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SRC = "hf://buckets/hamimmahmud0/SAM_COCO_b1/b1"; W = "/root/work/b1"; OUT = "webapp/public/b1"; WIDTH = 2000
os.makedirs(f"{W}/images", exist_ok=True); os.makedirs(f"{OUT}/frames", exist_ok=True); os.makedirs(f"{OUT}/ann", exist_ok=True)
orig = json.load(open(f"{W}/instances.json")); filt = json.load(open("outputs/b1_filtered/instances.json")); rem = {r["id"] for r in json.load(open("outputs/b1_filtered/removed.json"))}
cats = {c["id"]: c["name"] for c in orig["categories"]}
by = defaultdict(list)
for a in orig["annotations"]: by[a["image_id"]].append(a)      # original = kept + removed (flagged)

def poly(a, sc):
    x, y, w, h = a["bbox"]
    try:
        m = mu.decode(a["segmentation"])[int(y):int(y + h) + 1, int(x):int(x + w) + 1]
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: return None
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), max(1.5, 0.01 * max(w, h)), True)[:, 0, :]
        return [[round((px + int(x)) * sc, 1), round((py + int(y)) * sc, 1)] for px, py in c] if len(c) >= 3 else None
    except Exception: return None

def job(im):
    fn = im["file_name"]; name = os.path.splitext(fn)[0]; p = f"{W}/images/{fn}"
    if not os.path.exists(p): subprocess.run(["hf", "buckets", "cp", f"{SRC}/images/{fn}", p], check=True, capture_output=True)
    img = Image.open(p).convert("RGB"); sc = WIDTH / img.width; h = int(img.height * sc)
    img.resize((WIDTH, h), Image.LANCZOS).save(f"{OUT}/frames/{name}.jpg", quality=85)
    anns = []
    for a in by[im["id"]]:
        d = {"id": a["id"], "c": a["category_id"], "s": round(a["score"], 3), "b": [round(v * sc, 1) for v in a["bbox"]], "p": poly(a, sc)}
        if a["id"] in rem: d["r"] = 1
        anns.append(d)
    json.dump(anns, open(f"{OUT}/ann/{name}.json", "w"), separators=(",", ":"))
    return {"file": fn, "name": name, "w": WIDTH, "h": h, "orig": [img.width, img.height], "n": sum(1 for d in anns if "r" not in d), "nr": sum(1 for d in anns if "r" in d)}
if __name__ == "__main__":
    ims = sorted(orig["images"], key=lambda i: i["file_name"])
    with Pool(4) as p: frames = p.map(job, ims, chunksize=2)
    json.dump({"categories": cats, "frames": frames, "note": filt.get("info", {}).get("note", "")}, open(f"{OUT}/index.json", "w"))
    print("frames", len(frames), "kept", sum(f["n"] for f in frames), "removed", sum(f["nr"] for f in frames))
    print(os.popen(f"du -sh {OUT}").read())
