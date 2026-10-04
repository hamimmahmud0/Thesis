"""Stage + upload the filtered b1 dataset (runs on the server; HF_TOKEN in env)."""
import json, os, subprocess
B = "hf://buckets/z81980440/SAM_COCO_b1_filtered/b1"; S = "/root/work/b1_stage"; os.makedirs(S + "/annotations", exist_ok=True)
subprocess.run(["cp", "outputs/b1_filtered/instances.json", f"{S}/annotations/instances.json"], check=True)
rem = json.load(open("outputs/b1_filtered/removed.json")); subprocess.run(["cp", "outputs/b1_filtered/removed.json", f"{S}/annotations/removed_pedestrians.json"], check=True)
f = json.load(open("outputs/b1_filtered/instances.json")); o = json.load(open("/root/work/b1/instances.json"))
from collections import Counter
cats = {c["id"]: c["name"] for c in f["categories"]}
json.dump({"source": "hamimmahmud0/SAM_COCO_b1/b1", "images": len(f["images"]),
  "annotations_before_filter": len(o["annotations"]), "annotations": len(f["annotations"]),
  "pedestrians_removed": len(rem), "filter": "Pedestrian detections removed when >=50% of the pedestrian mask area (IoA, computed on RLE masks) lies inside a Vehicle, Car, Bus, Truck, Motorcycle, Bicycle or Rickshaw mask.",
  "removed_by_overlapping_class": dict(Counter(r["veh_class"] for r in rem)),
  "annotations_per_class": {cats[k]: v for k, v in sorted(Counter(a["category_id"] for a in f["annotations"]).items())},
  "files": {"annotations/instances.json": "filtered COCO annotations", "annotations/removed_pedestrians.json": "removed pedestrians: id, image_id, score, bbox, ioa, veh_class, veh_id", "images/": "original frames, unchanged"}},
  open(f"{S}/summary.json", "w"), indent=1)
for src, dst in ((f"{S}/summary.json", f"{B}/summary.json"), (f"{S}/annotations/instances.json", f"{B}/annotations/instances.json"), (f"{S}/annotations/removed_pedestrians.json", f"{B}/annotations/removed_pedestrians.json")):
    subprocess.run(["hf", "buckets", "cp", src, dst], check=True, capture_output=True)
subprocess.run(["hf", "buckets", "sync", "/root/work/b1/images", f"{B}/images"], check=True)
print("uploaded")
