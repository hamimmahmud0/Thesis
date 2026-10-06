#!/usr/bin/env python3
"""Clean the re-annotated dataset: (1) drop `not_an_object` detections, (2) drop overlapping detections.

Overlap (per image, on the segmentation masks): two detections overlap when mask IoU >= --iou or when >= --ioa of the
smaller mask lies inside the other. Of an overlapping pair the SMALLER mask is removed (ties: human label, then higher
model confidence). Greedy, biggest first, so a detection removed earlier no longer removes others.
Writes <out>/annotations/instances.json, removed.json (id, reason, overlapped_with, iou, ioa), summary.json, counts.png.
"""
import argparse, collections, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mu


def dedup_image(anns, iou_t, ioa_t):
    """-> (kept, [(removed_ann, kept_id, iou, ioa)])"""
    area0 = [float(mu.area(x["segmentation"])) for x in anns]
    idx = sorted(range(len(anns)), key=lambda i: (-area0[i], anns[i]["label_source"] != "human", -anns[i]["model_conf"], anns[i]["id"]))
    order = [anns[i] for i in idx]
    rl = [x["segmentation"] for x in order]
    area = np.array([area0[i] for i in idx])
    M = mu.iou(rl, rl, [0] * len(rl)) if len(rl) > 1 else None
    keep, removed = [], []
    for i in range(len(order)):
        hit = None
        for j in keep:
            iou = float(M[i, j]); inter = iou * (area[i] + area[j]) / (1 + iou); ioa = inter / max(min(area[i], area[j]), 1)
            if iou >= iou_t or ioa >= ioa_t:
                hit = (order[j]["id"], iou, ioa)
                break
        if hit is None:
            keep.append(i)
        else:
            removed.append((order[i],) + hit)
    return [order[i] for i in keep], removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", nargs="*", default=["not_an_object"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--ioa", type=float, default=0.8)
    a = ap.parse_args()
    out = Path(a.out); (out / "annotations").mkdir(parents=True, exist_ok=True)
    coco = json.load(open(a.instances))
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    before = collections.Counter(cat[x["category_id"]] for x in coco["annotations"])
    by, removed, kept_all = collections.defaultdict(list), [], []
    for x in coco["annotations"]:
        if cat[x["category_id"]] in a.drop:
            removed.append({"id": x["id"], "image_id": x["image_id"], "class": cat[x["category_id"]], "reason": "dropped_class",
                            "label_source": x["label_source"], "orig_category_id": x["orig_category_id"]})
        else:
            by[x["image_id"]].append(x)
    for iid, anns in by.items():
        kept, rem = dedup_image(anns, a.iou, a.ioa)
        kept_all += kept
        for x, kid, iou, ioa in rem:
            removed.append({"id": x["id"], "image_id": iid, "class": cat[x["category_id"]], "reason": "overlap", "overlapped_with": kid,
                            "iou": round(iou, 3), "ioa": round(ioa, 3), "label_source": x["label_source"], "orig_category_id": x["orig_category_id"],
                            "area": x["area"]})
    kept_all.sort(key=lambda x: x["id"])
    after = collections.Counter(cat[x["category_id"]] for x in kept_all)
    coco["annotations"] = kept_all
    coco["categories"] = [c for c in coco["categories"] if after[c["name"]] > 0]
    coco.setdefault("info", {})["cleaning"] = {"dropped_classes": a.drop, "overlap_rule": f"mask IoU >= {a.iou} or >= {a.ioa} of the smaller mask inside the other; smaller removed",
                                                "source": "re-annotated dataset (b1/annotations/instances.json)"}
    json.dump(coco, open(out / "annotations" / "instances.json", "w"))
    json.dump(removed, open(out / "removed.json", "w"))
    ov = [r for r in removed if r["reason"] == "overlap"]
    summary = {"images": len(coco["images"]), "annotations_before": sum(before.values()), "annotations_after": len(kept_all),
               "removed_not_an_object": sum(1 for r in removed if r["reason"] == "dropped_class"), "removed_overlap": len(ov),
               "removed_human_labelled": sum(1 for r in removed if r["label_source"] == "human"),
               "per_class_before": dict(before.most_common()), "per_class_after": dict(after.most_common()),
               "overlap_removed_per_class": dict(collections.Counter(r["class"] for r in ov).most_common()),
               "overlap_pairs_by_kept_vs_removed": dict(collections.Counter(
                   f"{r['class']} (removed)" for r in ov).most_common()),
               "images_without_detections": len(coco["images"]) - len({x["image_id"] for x in kept_all}),
               "rule": {"drop": a.drop, "iou": a.iou, "ioa": a.ioa, "removed": "smaller mask of an overlapping pair"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    names = sorted(before, key=lambda n: -before[n])
    fig, ax = plt.subplots(figsize=(12, 5)); xs = np.arange(len(names)); w = .4
    ax.bar(xs - w / 2, [before[n] for n in names], w, label="before cleaning"); ax.bar(xs + w / 2, [after[n] for n in names], w, label="after cleaning")
    ax.set_xticks(xs, names, rotation=45, ha="right"); ax.set_yscale("log"); ax.set_ylabel("detections (log)"); ax.legend()
    ax.set_title(f"Cleaning: {sum(before.values())} -> {len(kept_all)} detections"); plt.tight_layout(); plt.savefig(out / "counts.png", dpi=130)
    print(json.dumps({k: v for k, v in summary.items() if k != "overlap_pairs_by_kept_vs_removed"}, indent=1))


if __name__ == "__main__":
    main()
