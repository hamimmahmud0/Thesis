import json, numpy as np
from collections import defaultdict
coco = json.load(open("/root/work/b1/instances.json")); ann = {a["id"]: a for a in coco["annotations"]}
rem = json.load(open("outputs/b1_filtered/removed.json")); d = defaultdict(list)
for r in rem: d[r["veh_class"]].append(ann[r["veh_id"]]["area"] / max(1, ann[r["id"]]["area"]))
for c, v in d.items():
    v = np.array(v); print(f"{c:<8} n={len(v):3d} area ratio veh/ped  <1.5: {(v<1.5).sum():3d}  1.5-3: {((v>=1.5)&(v<3)).sum():3d}  3-10: {((v>=3)&(v<10)).sum():3d}  >=10: {(v>=10).sum():3d}  median {np.median(v):.1f}")
