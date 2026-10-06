"""Convert works/b1 (YOLO-OBB pseudo-labels) into a Label Studio import copy.
Source is read-only. Output: images/ (flat copy), tasks.json (pre-annotations as
predictions), label_config.xml. Rotated boxes use LS RectangleLabels convention:
x,y = top-left corner (percent), width/height percent, rotation in degrees CW about (x,y)."""
import json, math, shutil
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "b1"
DST = Path(__file__).resolve().parent
W, H = 4000, 3000
PREFIX = "/data/local-files/?d=b1_labelstudio/images/"  # LOCAL_FILES_DOCUMENT_ROOT = works/
names = [l.strip() for l in (SRC / "classes.txt").read_text().splitlines() if l.strip()]

def obb_to_ls(pts):
    p = pts * [W, H]
    c = p.mean(0)
    e = p[1] - p[0]
    th = math.atan2(e[1], e[0])
    u = np.array([math.cos(th), math.sin(th)]); v = np.array([-math.sin(th), math.cos(th)])
    w = np.linalg.norm(p[1] - p[0]); h = np.linalg.norm(p[2] - p[1])
    tl = c - u * w / 2 - v * h / 2
    return dict(x=tl[0] / W * 100, y=tl[1] / H * 100, width=w / W * 100, height=h / H * 100,
                rotation=math.degrees(th))

tasks = []
for split in ("train", "val"):
    for img in sorted((SRC / "images" / split).glob("*.JPG")):
        shutil.copy2(img, DST / "images" / img.name)
        res = []
        lab = SRC / "labels" / split / (img.stem + ".txt")
        for i, line in enumerate(lab.read_text().splitlines() if lab.exists() else []):
            t = line.split()
            if len(t) != 9: continue
            pts = np.array(t[1:], float).reshape(4, 2)
            res.append(dict(id=f"{img.stem}_{i}", type="rectanglelabels", from_name="label", to_name="image",
                            original_width=W, original_height=H, image_rotation=0,
                            value=dict(**obb_to_ls(pts), rectanglelabels=[names[int(t[0])]])))
        tasks.append(dict(data=dict(image=PREFIX + img.name, split=split),
                          predictions=[dict(model_version="yolo26l-obb-drashti conf>=0.5", result=res)]))
(DST / "tasks.json").write_text(json.dumps(tasks))
colors = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe","#008080","#e6beff","#9a6324","#800000","#aaffc3"]
cfg = '<View>\n  <RectangleLabels name="label" toName="image" canRotate="true">\n' + "".join(
    f'    <Label value="{n}" background="{colors[i]}"/>\n' for i, n in enumerate(names)) + \
    '  </RectangleLabels>\n  <Image name="image" value="$image" rotateControl="true"/>\n</View>\n'
(DST / "label_config.xml").write_text(cfg)
print(len(tasks), "tasks", sum(len(t["predictions"][0]["result"]) for t in tasks), "boxes")
