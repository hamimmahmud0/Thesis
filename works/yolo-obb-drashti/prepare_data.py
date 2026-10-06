"""Download DRASHTI from the HF bucket, convert DOTA-style labels to Ultralytics OBB, resize images.

Source layout (bucket hamimmahmud0/DRASHTI): images/{train,val,test}/*.jpg (3840x2160) and
labels/{split}_original/*.txt. Each label file: line 1 is a header "flightHeight:<m>" (skipped); other
lines are "x1 y1 x2 y2 x3 y3 x4 y4 <class> <difficulty> <extra>" in pixels (11 fields).

Output (--out): images/{split}/, labels/{split}/ (class x1 y1 .. x4 y4, normalized to [0,1]),
data.yaml, val_subset.txt (fixed random subset of val used for per-epoch validation in training).
Images are resized so the long side is --max-side (labels are normalized so need no change).

Usage:  python prepare_data.py --out /tmp/drashti --max-side 1920 --val-subset 800
Run on the kaggle-vm (never locally). Idempotent: existing outputs are skipped.
"""
import argparse
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2

from common import CLASSES

SPLITS = ["train", "val", "test"]
CID = {c: i for i, c in enumerate(CLASSES)}


def convert(job):
    img_p, lbl_p, out_img, out_lbl, max_side = job
    if out_img.exists() and out_lbl.exists():
        return 0, 0
    im = cv2.imread(str(img_p))
    if im is None:
        return -1, 0
    h, w = im.shape[:2]
    lines, skipped = [], 0
    if lbl_p.exists():
        for ln in lbl_p.read_text().splitlines():
            p = ln.split()
            if len(p) < 10 or p[8] not in CID:  # header or malformed
                skipped += ln.strip() != "" and not ln.startswith("flightHeight")
                continue
            c = [float(v) for v in p[:8]]
            n = [min(max(v / (w if i % 2 == 0 else h), 0.0), 1.0) for i, v in enumerate(c)]
            lines.append(f"{CID[p[8]]} " + " ".join(f"{v:.6f}" for v in n))
    s = max_side / max(w, h)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_img), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
    return 1, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default="hamimmahmud0/DRASHTI")
    ap.add_argument("--raw", default="/tmp/drashti_raw", help="where the bucket is synced")
    ap.add_argument("--out", default="/tmp/drashti")
    ap.add_argument("--max-side", type=int, default=1920, help="resize long side to this (no upscaling)")
    ap.add_argument("--val-subset", type=int, default=800, help="images in val_subset.txt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)

    if not a.skip_download:
        subprocess.run(["hf", "buckets", "sync", f"hf://buckets/{a.bucket}", str(raw)], check=True)
    jobs = []
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for img in sorted((raw / "images" / sp).glob("*.jpg")):
            jobs.append((img, raw / "labels" / f"{sp}_original" / f"{img.stem}.txt",
                         out / "images" / sp / img.name, out / "labels" / sp / f"{img.stem}.txt", a.max_side))
    print(f"{len(jobs)} images")
    bad = skipped = 0
    with Pool(a.workers) as pool:
        for i, (ok, sk) in enumerate(pool.imap_unordered(convert, jobs, chunksize=8), 1):
            bad += ok < 0
            skipped += sk
            if i % 2000 == 0:
                print(i, "done", flush=True)
    print(f"unreadable images: {bad}, malformed label lines skipped: {skipped}")

    vals = sorted((out / "images" / "val").glob("*.jpg"))
    random.Random(0).shuffle(vals)
    (out / "val_subset.txt").write_text("\n".join(str(p) for p in vals[:a.val_subset]) + "\n")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: val_subset.txt\ntest: images/test\n"
        f"val_full: images/val\nnames:\n{names}\n")
    (out / "data_fullval.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
