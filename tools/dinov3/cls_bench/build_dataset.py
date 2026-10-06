#!/usr/bin/env python3
"""Snapshot the exported crops (class_annotator_from_coco/cls_dataset) into a train/val/test dataset.

- Autorickshaw is merged into Rickshaw (label map below); decisions.json / classes.yaml are not touched.
- Classes with fewer than --min-count crops (Pushcart, police_car) are dropped and reported.
- Split ~70/15/15, stratified by class and grouped by source image (crops of one frame never straddle splits).
Writes <out>/images/<label>/<file>.png, <out>/split.csv (path,label,group,split) and <out>/meta.json.
"""
import argparse, collections, csv, hashlib, json, shutil
from pathlib import Path

import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

MERGE = {"Autorickshaw": "Rickshaw"}


warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../utils/class_annotator_from_coco/cls_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="first seed tried")
    ap.add_argument("--tries", type=int, default=300, help="seeds tried; the most balanced split is kept")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    rows = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.png")):
            label = MERGE.get(d.name, d.name)
            group = f.stem.rsplit("_", 1)[0]           # <image stem>_<annotation id>
            rows.append((f, label, group))
    raw = collections.Counter(p.parent.name for p, _, _ in rows)
    cnt = collections.Counter(l for _, l, _ in rows)
    dropped = {l: n for l, n in cnt.items() if n < a.min_count}
    rows = [r for r in rows if r[1] not in dropped]
    y = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows])
    # 70/15/15 = two nested grouped stratified splits: 7 folds -> test fold(s)/val fold(s)
    idx = np.arange(len(rows)); classes = sorted(set(y))

    def make(seed):
        sp = np.array(["train"] * len(rows), dtype=object)
        for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed).split(idx, y, g)):
            sp[te] = "test" if k < 3 else "val" if k < 6 else "train"      # 3/20 test, 3/20 val, 14/20 train
        return sp

    def cost(sp):  # how far each class is from 15% val / 15% test (every class counts equally)
        c = 0.0
        for l in classes:
            m = y == l
            c += sum((((sp[m] == s).mean()) - 0.15) ** 2 for s in ("val", "test"))
        return c

    # grouping makes the split lumpy; keep the best of a few hundred seeds
    best = min(range(a.seed, a.seed + a.tries), key=lambda s: cost(make(s)))
    split = make(best); a.seed = best
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    with open(out / "split.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "group", "split"])
        for (f, l, gr), s in zip(rows, split):
            dst = out / "images" / l / f.name
            dst.parent.mkdir(exist_ok=True); shutil.copy2(f, dst)
            w.writerow([f"images/{l}/{f.name}", l, gr, s])
    per = {l: {s: int(((y == l) & (split == s)).sum()) for s in ("train", "val", "test")} for l in sorted(set(y))}
    dec = Path(a.src).parent / "cls_dataset" / "decisions.json"
    meta = {"source": str(src), "decisions_sha256": hashlib.sha256(dec.read_bytes()).hexdigest() if dec.exists() else None,
            "raw_counts": dict(raw), "merge": MERGE, "dropped_min_count": dropped, "seed": a.seed, "per_class": per,
            "total": {s: int((split == s).sum()) for s in ("train", "val", "test")},
            "groups": {s: len(set(g[split == s])) for s in ("train", "val", "test")}}
    assert not (set(g[split == "train"]) & set(g[split == "test"])) and not (set(g[split == "val"]) & set(g[split == "test"])) \
        and not (set(g[split == "train"]) & set(g[split == "val"])), "group leakage"
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
