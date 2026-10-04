"""Sample frames from SAM_COCO_b1 and build data for the /b1 viewer (runs on remote)."""
import json, os, random, subprocess
import cv2, numpy as np
from pycocotools import mask as mu
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SRC = "hf://buckets/hamimmahmud0/SAM_COCO_b1/b1"; W = "/root/work/b1"; OUT = "webapp/public/b1"
N_FRAMES, WIDTH, SEED = 12, 2000, 3
os.makedirs(f"{W}/images", exist_ok=True); os.makedirs(f"{OUT}/frames", exist_ok=True)
def cp(src, dst):
    if not os.path.exists(dst): subprocess.run(["hf", "buckets", "cp", src, dst], check=True, capture_output=True)
cp(f"{SRC}/annotations/instances.json", f"{W}/instances.json")
coco = json.load(open(f"{W}/instances.json")); cats = {c["id"]: c["name"] for c in coco["categories"]}
by = {}
for a in coco["annotations"]: by.setdefault(a["image_id"], []).append(a)
imgs = [i for i in coco["images"] if i["id"] in by]
frames = sorted(random.Random(SEED).sample(imgs, N_FRAMES), key=lambda i: i["file_name"])
out = {"categories": cats, "frames": []}
for im in frames:
    fn = im["file_name"]; cp(f"{SRC}/images/{fn}", f"{W}/images/{fn}")
    img = Image.open(f"{W}/images/{fn}").convert("RGB"); sc = WIDTH / img.width
    name = os.path.splitext(fn)[0] + ".jpg"
    img.resize((WIDTH, int(img.height * sc)), Image.LANCZOS).save(f"{OUT}/frames/{name}", quality=88)
    anns = []
    for a in by[im["id"]]:
        x, y, w, h = a["bbox"]; poly = None
        try:
            m = mu.decode(a["segmentation"])[int(y):int(y + h) + 1, int(x):int(x + w) + 1]
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), max(1.5, 0.01 * max(w, h)), True)[:, 0, :]
                poly = [[round((px + int(x)) * sc, 1), round((py + int(y)) * sc, 1)] for px, py in c] if len(c) >= 3 else None
        except Exception: pass
        anns.append({"id": a["id"], "c": a["category_id"], "s": round(a["score"], 3), "b": [round(v * sc, 1) for v in a["bbox"]], "p": poly})
    out["frames"].append({"img": f"/b1/frames/{name}", "file": fn, "w": WIDTH, "h": int(img.height * sc), "orig": [img.width, img.height], "anns": anns})
    print(fn, len(anns), flush=True)
json.dump(out, open(f"{OUT}/data.json", "w"), separators=(",", ":"))
print("done", os.popen(f"du -sh {OUT}").read())
