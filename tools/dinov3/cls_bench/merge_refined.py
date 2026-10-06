#!/usr/bin/env python3
"""Build the re-annotated COCO file from the shard predictions.

category_id = human label where the reviewer decided (confirm / change in decisions.json, Autorickshaw merged into
Rickshaw), otherwise the classifier's prediction. Originals are kept in `orig_category_id`; the model's own
prediction, confidence and top-3 are stored on every annotation. The `Vehicle` category is not kept (nothing is left in it).
"""
import argparse, collections, hashlib, json
from pathlib import Path

MERGE = {"Autorickshaw": "Rickshaw"}
KEEP_IDS = {"Pedestrian": 1, "Car": 3, "Bus": 4, "Truck": 5, "Motorcycle": 6, "Bicycle": 7, "Rickshaw": 8}   # unchanged COCO ids
LOW_RES = 12          # min(w, h) below this: crop too small for a trustworthy class
LOW_CONF = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "annotations").mkdir(parents=True, exist_ok=True)
    coco = json.load(open(a.instances))
    old = {c["id"]: c["name"] for c in coco["categories"]}
    preds, infos = {}, []
    for p in a.preds:
        d = json.load(open(p)); preds.update(d["preds"]); infos.append(d["info"])
    dec = json.load(open(a.decisions))
    human = {k: MERGE.get(v["class"], v["class"]) for k, v in dec.items() if v["action"] in ("confirm", "change")}
    missing = [x["id"] for x in coco["annotations"] if str(x["id"]) not in preds]
    assert not missing, f"{len(missing)} detections without prediction"
    names = {x["cls"] for x in preds.values()} | set(human.values())
    ids = dict(KEEP_IDS)
    for n in sorted(names - set(ids)):
        ids[n] = max(ids.values()) + 1
    used = collections.Counter(); trans = collections.Counter(); src = collections.Counter()
    for x in coco["annotations"]:
        k = str(x["id"]); pr = preds[k]
        final = human.get(k, pr["cls"])
        x["orig_category_id"] = x["category_id"]
        x["category_id"] = ids[final]
        x["label_source"] = "human" if k in human else "model"
        x["model_pred"], x["model_conf"], x["model_top3"] = pr["cls"], round(pr["conf"], 4), [[c, round(p, 4)] for c, p in pr["top3"]]
        x["low_res"] = min(x["bbox"][2:]) < LOW_RES
        x["low_conf"] = pr["conf"] < LOW_CONF
        used[final] += 1; src[x["label_source"]] += 1; trans[(old[x["orig_category_id"]], final)] += 1
    coco["categories"] = [{"id": i, "name": n, "supercategory": ""} for n, i in sorted(ids.items(), key=lambda kv: kv[1]) if used[n]]
    coco.setdefault("info", {})["reannotation"] = {
        "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m (frozen) + LogisticRegression(C=0.01, balanced), cls+mean features, fit on all 1812 reviewed crops",
        "rule": "category_id = human label if reviewed else model prediction; original class in orig_category_id",
        "flags": {"low_res": f"min(w,h) < {LOW_RES}px", "low_conf": f"model top-1 probability < {LOW_CONF}"}}
    json.dump(coco, open(out / "annotations" / "instances.json", "w"))
    n = len(coco["annotations"])
    mp = [x for x in coco["annotations"] if x["label_source"] == "model"]
    changed = sum(1 for x in mp if x["category_id"] != x["orig_category_id"])
    summary = {"source": "hamimmahmud0/SAM_COCO_b1/b1", "images": len(coco["images"]), "annotations": n,
               "label_source": dict(src), "decisions_sha256": hashlib.sha256(Path(a.decisions).read_bytes()).hexdigest(),
               "per_class": dict(used.most_common()),
               "model_labelled": {"n": len(mp), "changed_vs_original": changed, "changed_pct": round(100 * changed / max(len(mp), 1), 1),
                                  "low_conf": sum(x["low_conf"] for x in mp), "low_res": sum(x["low_res"] for x in mp),
                                  "mean_conf": round(sum(x["model_conf"] for x in mp) / max(len(mp), 1), 3)},
               "original_to_final": {f"{o} -> {f}": c for (o, f), c in trans.most_common()},
               "shards": infos,
               "note": "Held-out quality of the classifier (ViT-7B LVD + LogReg, train-only fit): test macro-F1 0.758, acc 0.797 on 14 classes; "
                       "weak classes Cyclevan, pickup, Leguna. Human labels cover only the reviewed detections."}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("original_to_final", "shards")}, indent=1))


if __name__ == "__main__":
    main()
