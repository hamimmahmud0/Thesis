"""Draw converted OBB labels on a few images to sanity-check the conversion.

Usage: python visualize_labels.py --data-dir /tmp/drashti --split train --n 4 --out /tmp/viz
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from common import CLASSES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/tmp/drashti")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="/tmp/viz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d, out = Path(a.data_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((d / "images" / a.split).glob("*.jpg"))
    for p in random.Random(a.seed).sample(imgs, a.n):
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        for ln in (d / "labels" / a.split / f"{p.stem}.txt").read_text().split("\n"):
            v = ln.split()
            if len(v) != 9:
                continue
            pts = (np.array(v[1:], dtype=float).reshape(4, 2) * [w, h]).astype(np.int32)
            cv2.polylines(im, [pts], True, (0, 255, 0), 2)
            cv2.putText(im, CLASSES[int(v[0])], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"{p.stem}.jpg"), im)
    print("wrote", a.n, "images to", out)


if __name__ == "__main__":
    main()
