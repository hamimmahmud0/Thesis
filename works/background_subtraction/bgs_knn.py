"""KNN background subtraction (OpenCV) on a video.

Outputs: <out>/mask.mp4 (foreground masks), <out>/overlay.mp4 (side-by-side),
<out>/stats.csv (per-frame foreground ratio), <out>/sample_*.png.
"""
import argparse
import csv
import os
import urllib.request

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path or URL")
    ap.add_argument("--out", default="out")
    ap.add_argument("--history", type=int, default=500)
    ap.add_argument("--dist2", type=float, default=400.0)
    ap.add_argument("--shadows", action="store_true")
    ap.add_argument("--open", type=int, default=3, help="morph open kernel (0=off)")
    ap.add_argument("--width", type=int, default=2048, help="resize to this width (0=native)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = args.video
    if src.startswith("http"):
        local = os.path.join(args.out, "input.mp4")
        if not os.path.exists(local):
            urllib.request.urlretrieve(src, local)
        src = local

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.width and args.width < w:
        h = round(h * args.width / w / 2) * 2
        w = args.width
    print(f"video {w}x{h} @ {fps:.2f}fps, {n} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wm = cv2.VideoWriter(os.path.join(args.out, "mask.mp4"), fourcc, fps, (w, h), False)
    wo = cv2.VideoWriter(os.path.join(args.out, "overlay.mp4"), fourcc, fps, (w * 2, h))

    bg = cv2.createBackgroundSubtractorKNN(
        history=args.history, dist2Threshold=args.dist2, detectShadows=args.shadows
    )
    kernel = np.ones((args.open, args.open), np.uint8) if args.open else None

    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        if frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        m = bg.apply(frame)
        if args.shadows:
            m = (m == 255).astype(np.uint8) * 255
        if kernel is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        wm.write(m)
        vis = frame.copy()
        vis[m > 0] = (0, 0, 255)
        wo.write(np.hstack([frame, cv2.addWeighted(frame, 0.5, vis, 0.5, 0)]))
        rows.append((i, float((m > 0).mean())))
        if i % max(1, n // 5) == 0:
            cv2.imwrite(os.path.join(args.out, f"sample_{i:05d}.png"),
                        np.hstack([frame, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)]))
        i += 1
        if i % 100 == 0:
            print("frame", i, flush=True)

    cap.release(); wm.release(); wo.release()
    with open(os.path.join(args.out, "stats.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["frame", "fg_ratio"]); wr.writerows(rows)
    print("done", i, "frames; mean fg ratio", np.mean([r[1] for r in rows]))


if __name__ == "__main__":
    main()
