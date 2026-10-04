"""Drop Pedestrian detections whose mask lies inside a vehicle mask (SAM_COCO_b1). Runs on the server.
Overlap = intersection / pedestrian-mask area (IoA), computed on the RLE masks with pycocotools (crowd trick)."""
import json, os, subprocess, sys
from collections import Counter, defaultdict
import numpy as np
from pycocotools import mask as mu
SRC = "hf://buckets/hamimmahmud0/SAM_COCO_b1/b1"; W = "/root/work/b1"; OUT = "outputs/b1_filtered"
THR = float(os.environ.get("THR", 0.5)); VEH = {int(c) for c in os.environ.get("VEH", "2,3,4,5").split(",")}   # Vehicle, Car, Bus, Truck
os.makedirs(W, exist_ok=True); os.makedirs(OUT, exist_ok=True)
if not os.path.exists(f"{W}/instances.json"):
    subprocess.run(["hf", "buckets", "cp", f"{SRC}/annotations/instances.json", f"{W}/instances.json"], check=True, capture_output=True)
coco = json.load(open(f"{W}/instances.json")); cats = {c["id"]: c["name"] for c in coco["categories"]}
PED = next(i for i, n in cats.items() if n == "Pedestrian")
by = defaultdict(list)
for a in coco["annotations"]: by[a["image_id"]].append(a)
rows = []   # one per pedestrian: best IoA per other class
for iid, anns in by.items():
    peds = [a for a in anns if a["category_id"] == PED]
    oth = [a for a in anns if a["category_id"] != PED]
    if not peds or not oth:
        rows += [(p, {}) for p in peds]; continue
    ioa = mu.iou([p["segmentation"] for p in peds], [o["segmentation"] for o in oth], [1] * len(oth))   # [ped x other], = inter / area(ped)
    for i, p in enumerate(peds):
        best = {}
        for j, o in enumerate(oth):
            v = float(ioa[i, j])
            if v > best.get(o["category_id"], (0, None))[0]: best[o["category_id"]] = (v, o["id"])
        rows.append((p, best))
print("pedestrians", len(rows))
for t in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9):
    print(f"thr {t}: " + ", ".join(f"{cats[c]} {sum(1 for _, b in rows if b.get(c, (0,))[0] >= t)}" for c in sorted(cats) if c != PED)
          + f" | any of {sorted(cats[c] for c in VEH)}: {sum(1 for _, b in rows if max([b.get(c, (0,))[0] for c in VEH] or [0]) >= t)}")
removed = []; drop_ids = set()
for p, b in rows:
    c, (v, oid) = max(((c, b.get(c, (0, None))) for c in VEH), key=lambda x: x[1][0])
    if v >= THR:
        drop_ids.add(p["id"]); removed.append({"id": p["id"], "image_id": p["image_id"], "score": p["score"], "bbox": p["bbox"], "ioa": round(v, 3), "veh_class": cats[c], "veh_id": oid})
kept = [a for a in coco["annotations"] if a["id"] not in drop_ids]
out = {**coco, "annotations": kept, "info": {**coco.get("info", {}), "note": f"removed {len(drop_ids)} Pedestrian detections with IoA>={THR} inside {sorted(cats[c] for c in VEH)}"}}
json.dump(out, open(f"{OUT}/instances.json", "w")); json.dump(removed, open(f"{OUT}/removed.json", "w"))
print(f"THR={THR}: removed {len(drop_ids)} of {len(rows)} pedestrians; annotations {len(coco['annotations'])} -> {len(kept)}")
print("removed by vehicle class:", dict(Counter(r["veh_class"] for r in removed)))
print("removed pedestrian score quantiles:", np.quantile([r["score"] for r in removed], [.1, .5, .9]).round(2).tolist() if removed else None)
