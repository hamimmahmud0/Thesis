"""Contact sheets of removed pedestrians (red) inside their vehicle (cyan). Runs on the server."""
import json, os, random, subprocess
import cv2, numpy as np
from pycocotools import mask as mu
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
SRC = "hf://buckets/hamimmahmud0/SAM_COCO_b1/b1"; W = "/root/work/b1"; OUT = "/root/work/b1_sheets"; os.makedirs(OUT, exist_ok=True)
coco = json.load(open(f"{W}/instances.json")); ann = {a["id"]: a for a in coco["annotations"]}; img = {i["id"]: i for i in coco["images"]}
rem = json.load(open("outputs/b1_filtered/removed.json")); rng = random.Random(1)
font = ImageFont.load_default(size=15); C = 330
def poly(a, ox, oy, s):
    m = mu.decode(a["segmentation"]); x, y, w, h = a["bbox"]
    cs, _ = cv2.findContours(m[int(y):int(y + h) + 1, int(x):int(x + w) + 1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[((p[0] + int(x)) - ox) * s, ((p[1] + int(y)) - oy) * s] for c in cs for p in c[:, 0, :]]
def cell(r):
    fn = img[r["image_id"]]["file_name"]; p = f"{W}/images/{fn}"
    if not os.path.exists(p): os.makedirs(f"{W}/images", exist_ok=True); subprocess.run(["hf", "buckets", "cp", f"{SRC}/images/{fn}", p], check=True, capture_output=True)
    im = Image.open(p).convert("RGB"); v = ann[r["veh_id"]]; pd = ann[r["id"]]
    x0, y0 = min(v["bbox"][0], pd["bbox"][0]), min(v["bbox"][1], pd["bbox"][1])
    x1, y1 = max(v["bbox"][0] + v["bbox"][2], pd["bbox"][0] + pd["bbox"][2]), max(v["bbox"][1] + v["bbox"][3], pd["bbox"][1] + pd["bbox"][3])
    pad = 0.25 * max(x1 - x0, y1 - y0, 40); box = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(im.width, x1 + pad)), int(min(im.height, y1 + pad)))
    crop = im.crop(box); s = C / max(crop.size); crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    for a, col in ((v, "#18ffff"), (pd, "#ff1744")):
        pts = poly(a, box[0], box[1], s)
        if len(pts) > 2: d.polygon([tuple(q) for q in pts], outline=col)
        x, y, w, h = a["bbox"]; d.rectangle([(x - box[0]) * s, (y - box[1]) * s, (x + w - box[0]) * s, (y + h - box[1]) * s], outline=col)
    canvas = Image.new("RGB", (C, C + 22), (20, 20, 20)); canvas.paste(crop, ((C - crop.width) // 2, 0))
    ImageDraw.Draw(canvas).text((4, C + 3), f"{r['veh_class']} IoA {r['ioa']:.2f} ped score {r['score']:.2f}", fill="white", font=font); return canvas
def sheet(rs, name, cols=5):
    cells = [cell(r) for r in rs]; rows = (len(cells) + cols - 1) // cols
    S = Image.new("RGB", (cols * (C + 6) + 6, rows * (C + 28) + 6), (40, 40, 40))
    for k, c in enumerate(cells): S.paste(c, (6 + (k % cols) * (C + 6), 6 + (k // cols) * (C + 28)))
    S.save(f"{OUT}/{name}.jpg", quality=88)
riders = [r for r in rem if r["veh_class"] in ("Motorcycle", "Rickshaw", "Bicycle")]
print(len(riders))
sheet(rng.sample(riders, 20), "removed_riders", cols=5)
