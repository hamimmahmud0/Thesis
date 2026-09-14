#!/usr/bin/env python3
"""Per-frame camera motion estimation from CoTracker3 tracks (similarity model, 4 DOF).

WINDOWED GRID RE-SEEDING (--window, default 100):
  Grid/query points are RE-DEFINED at every window anchor instead of being
  defined once and run across the whole video (they diminish over a long
  context as tracks leave the frame). Anchors sit at frames 0, w-1, 2w-1,
  3w-1, ... i.e. windows 0-99, 99-199, 199-299 for --window 100: consecutive
  windows share exactly one anchor frame. Within a window only points that
  are visible AND in-bounds at its anchor frame are used (per-pair
  visibility/bounds still gate every frame pair), so the correspondence set
  is refreshed at every anchor. Per-frame motion is chained through the
  shared anchor frames, which COMBINES all windows into one global
  frame-0-referenced trajectory (motion.csv) that stabilize.py consumes
  unchanged. --window 0 reproduces the legacy single-grid behaviour.

CONVENTION (applied consistently):
  cv2.estimateAffinePartial2D(pts_prev, pts_curr) returns the IMAGE-motion
  similarity M that maps point positions in frame t-1 to frame t. If the camera
  translates right by D, static scene content appears to shift LEFT, i.e. the
  induced image translation is -D. Camera motion is therefore the INVERSE of
  the induced image motion:
      camera_translation = -(tx, ty)
      camera_yaw         = -yaw_image
  The reported d_yaw_deg uses atan2(-M[1,0], M[0,0]) which ALREADY negates the
  image rotation angle, so it is directly the camera yaw rate (deg/frame).
  Scale is reported as the induced IMAGE magnification s = hypot(a, b):
      s > 1 -> image magnified -> camera descending (approaching ground)
      s < 1 -> image minified  -> camera ascending
  Cumulative trajectory composes the CAMERA transforms (translation/yaw) into
  the frame-0 reference; log-scale accumulates log(image scale).
"""

import argparse
import json
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_CORRESPONDENCES = 8


def squeeze_batch(arr):
    """If arr has a leading batch dim of 1, drop it."""
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def in_bounds(pts, w, h):
    return (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    )


def runs_of(mask):
    """Yield (start, end) index runs where mask is True (mask: 1-D bool)."""
    if not mask.any():
        return
    idx = np.flatnonzero(np.diff(np.concatenate(([False], mask.astype(int)))))

    for k in range(0, len(idx), 2):
        yield int(idx[k]), int(idx[k + 1])


def estimate_pair(pts_prev, pts_curr, max_points):
    """RANSAC similarity fit for one frame pair.

    Returns (M_2x3 or None, inliers, n_used).
    """
    n = pts_prev.shape[0]
    if n > max_points:
        sel = np.linspace(0, n - 1, max_points).astype(np.int64)
        pts_prev = pts_prev[sel]
        pts_curr = pts_curr[sel]

    try:
        M, inl = cv2.estimateAffinePartial2D(
            pts_prev.reshape(-1, 1, 2).astype(np.float32),
            pts_curr.reshape(-1, 1, 2).astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=5000,
            confidence=0.999,
        )
    except cv2.error:
        return None, 0, pts_prev.shape[0]

    if M is None or inl is None:
        return None, 0, pts_prev.shape[0]
    if not np.isfinite(M).all():
        return None, 0, pts_prev.shape[0]

    inliers = int(inl.sum())
    # Degenerate guard: zero/near-zero scale (collinear configurations can
    # produce this) or non-finite decomposition inputs.
    s = float(np.hypot(M[0, 0], M[1, 0]))
    if s < 1e-6:
        return None, inliers, pts_prev.shape[0]

    return M, inliers, pts_prev.shape[0]


def decompose_camera_motion(M):
    """IMAGE-motion M [[a,b,tx],[-b,a,ty]] -> CAMERA-motion (dx, dy, yaw_deg, scale)."""
    a, b = M[0, 0], M[0, 1]
    tx, ty = M[0, 2], M[1, 2]
    scale = float(np.hypot(a, b))
    yaw_img = float(np.arctan2(-M[1, 0], M[0, 0]))  # == -atan2(b, a): camera yaw
    # Inverse convention: camera translation opposes induced image shift.
    dx, dy = -float(tx), -float(ty)
    return dx, dy, float(np.degrees(yaw_img)), scale


def camera_matrix(dx, dy, yaw_deg, scale):
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [
            [scale * c, -scale * s, dx],
            [scale * s, scale * c, dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_tracks(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    tracks = squeeze_batch(np.asarray(d["tracks"], dtype=np.float32))
    visibility = squeeze_batch(np.asarray(d["visibility"], dtype=bool))
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be (T, N, 2), got {tracks.shape}")

    width = height = None
    fps = None
    if "meta" in d.files:
        try:
            meta = json.loads(str(d["meta"]))
            width = meta.get("width")
            height = meta.get("height")
            fps = meta.get("fps")
        except Exception:
            pass
    return tracks, visibility, width, height, fps


def window_anchors(T, window):
    """Grid re-seed anchor frames: 0, w-1, 2w-1, ... (windows 0-99, 99-199, ...).

    Window i covers pairs (anchor_i, anchor_{i+1}]; the last window runs to
    T-1. Consecutive windows share exactly one anchor frame, so per-frame
    motion chains across windows without a seam. window=0 -> single window
    [0, T-1] (legacy, grid never re-defined).
    """
    if window == 0:
        return [0]
    if window < 2:
        raise ValueError(f"--window must be 0 (off) or >= 2, got {window}")
    anchors = [0]
    k = 1
    while k * window - 1 < T - 1:
        anchors.append(k * window - 1)
        k += 1
    return anchors


def window_grid(tracks, visibility, anchor, width, height):
    """Re-defined grid for the window starting at `anchor`: points visible
    and in-bounds at the anchor frame."""
    grid = visibility[anchor]
    if width and height:
        grid = grid & in_bounds(tracks[anchor], width, height)
    return grid


def estimate_motion(tracks, visibility, width, height, max_points, window):
    """Windowed motion estimation; all point ops are vectorized.

    Grid points are re-defined at every window anchor (frames 0, w-1, 2w-1,
    ...; e.g. --window 100 -> windows 0-99, 99-199, 199-299) so usable
    correspondences are refreshed instead of diminishing over the video.
    Chaining the per-frame increments through the shared anchor frames
    combines the per-window motion into one global frame-0 reference.
    """
    T, N, _ = tracks.shape
    anchors = window_anchors(T, window)
    reseed = window != 0
    rows = []
    H = np.eye(3)
    cum_yaw = 0.0
    log_scale = 0.0

    # Frame 0 reference row (identity, no motion yet).
    grid0 = window_grid(tracks, visibility, 0, width, height) if reseed \
        else np.ones(N, dtype=bool)
    rows.append(dict(frame=0, dx=0.0, dy=0.0, d_yaw_deg=0.0, scale=1.0,
                     inliers=visibility[0].sum(), inlier_ratio=float(visibility[0].mean()),
                     cum_x=0.0, cum_y=0.0, cum_yaw_deg=0.0,
                     cum_log_scale=0.0, reliable=True, usable=int(visibility[0].sum()),
                     window=0, grid_size=int(grid0.sum())))

    for wi, s in enumerate(anchors):
        e = anchors[wi + 1] if wi + 1 < len(anchors) else T - 1
        # RE-DEFINE the grid points at this window's anchor frame.
        grid = window_grid(tracks, visibility, s, width, height) if reseed \
            else None
        grid_size = int(grid.sum()) if grid is not None else N

        for t in range(s + 1, e + 1):
            p_prev = tracks[t - 1]
            p_curr = tracks[t]
            # Usable for THIS pair: still-alive window grid points, visible
            # in both frames + inside bounds in both frames.
            usable = visibility[t - 1] & visibility[t]
            if width and height:
                usable &= in_bounds(p_prev, width, height) & in_bounds(p_curr, width, height)
            if grid is not None:
                usable &= grid

            n_usable = int(usable.sum())
            base = dict(
                frame=t,
                cum_x=H[0, 2],
                cum_y=H[1, 2],
                cum_yaw_deg=cum_yaw,
                cum_log_scale=log_scale,
                usable=n_usable,
                window=wi,
                grid_size=grid_size,
            )

            if n_usable < MIN_CORRESPONDENCES:
                # Unreliable pair: carry previous pose forward with IDENTITY motion,
                # flag stays in the output.
                rows.append(dict(base, dx=0.0, dy=0.0, d_yaw_deg=0.0, scale=1.0,
                                 inliers=0, inlier_ratio=0.0, reliable=False))
                continue

            M, inliers, n_used = estimate_pair(p_prev[usable], p_curr[usable], max_points)
            if M is None:
                rows.append(dict(base, dx=0.0, dy=0.0, d_yaw_deg=0.0, scale=1.0,
                                 inliers=inliers, inlier_ratio=(inliers / n_used),
                                 reliable=False))
                continue

            dx, dy, yaw_deg, scale = decompose_camera_motion(M)
            H = H @ camera_matrix(dx, dy, yaw_deg, scale)
            cum_yaw += yaw_deg
            log_scale += float(np.log(scale))

            rows.append(dict(base, dx=dx, dy=dy, d_yaw_deg=yaw_deg, scale=scale,
                             inliers=inliers, inlier_ratio=(inliers / n_used),
                             reliable=True))

    return pd.DataFrame(rows)


def make_plots(df, out_png, title):
    t = df["frame"].to_numpy()
    unreliable = ~df["reliable"].to_numpy()
    # Grid re-seed anchors (windowed motion only): frames where the window
    # index changes, minus 1 (the shared anchor frame of the two windows).
    if "window" in df.columns:
        w = df["window"].to_numpy()
        anchors = t[1:][np.diff(w) != 0] - 1 if len(w) > 1 else np.array([])
    else:
        anchors = np.array([])

    fig = plt.figure(figsize=(20, 10.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1, 1])
    fig.suptitle(title, fontsize=13)

    def shade(ax):
        for a, b in runs_of(unreliable):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="red", alpha=0.18)

    def mark_anchors(ax):
        for a in anchors:
            ax.axvline(a, color="gray", lw=0.6, alpha=0.35)

    # 1. Trajectory X-Y colored by time (left column, top two rows)
    ax1 = fig.add_subplot(gs[:2, 0])
    sc = ax1.scatter(df["cum_x"], df["cum_y"], c=t, cmap="viridis", s=4)
    fig.colorbar(sc, ax=ax1, label="frame")
    ax1.plot(df["cum_x"].iloc[0], df["cum_y"].iloc[0], marker="*", ms=18,
             color="lime", ls="none", label="start")
    ax1.plot(df["cum_x"].iloc[-1], df["cum_y"].iloc[-1], marker="X", ms=14,
             color="red", ls="none", label="end")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("cumulative X (px)")
    ax1.set_ylabel("cumulative Y (px)")
    ax1.set_title("Camera trajectory (frame-0 reference)")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # 2. dx, dy vs frame
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df["dx"], lw=0.7, label="dx")
    ax2.plot(t, df["dy"], lw=0.7, label="dy")
    shade(ax2)
    mark_anchors(ax2)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("px/frame")
    ax2.set_title("Per-frame translation (camera)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. yaw rate + cumulative yaw
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, df["d_yaw_deg"], lw=0.7, color="tab:blue", label="yaw rate")
    shade(ax3)
    ax3.set_xlabel("frame")
    ax3.set_ylabel("deg/frame")
    ax3b = ax3.twinx()
    ax3b.plot(t, df["cum_yaw_deg"], lw=1.2, color="tab:orange", label="cumulative")
    ax3b.set_ylabel("deg (cum)")
    l1, la1 = ax3.get_legend_handles_labels()
    l2, la2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1 + l2, la1 + la2, fontsize=8, loc="upper left")
    ax3.set_title("Yaw")
    ax3.grid(alpha=0.3)
    mark_anchors(ax3)

    # 4. scale vs frame (log y)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(t, df["scale"].clip(lower=1e-6), lw=0.7)
    ax4.axhline(1.0, color="k", lw=0.8, ls="--")
    shade(ax4)
    ax4.set_xlabel("frame")
    ax4.set_ylabel("image scale (>1 descending)")
    ax4.set_title("Zoom scale per frame")
    ax4.grid(alpha=0.3, which="both")
    mark_anchors(ax4)

    # 5. Diagnostics: usable point count + inlier ratio (grid re-seed anchors
    #    marked in gray - usable count jumps back up at each anchor)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(t, df["usable"], lw=0.7, color="tab:green", label="usable pts")
    shade(ax5)
    mark_anchors(ax5)
    ax5.set_xlabel("frame")
    ax5.set_ylabel("# usable")
    ax5b = ax5.twinx()
    ax5b.plot(t, df["inlier_ratio"], lw=0.7, color="tab:red", label="inlier ratio")
    ax5b.set_ylim(-0.02, 1.05)
    ax5b.set_ylabel("inlier ratio")
    l1, la1 = ax5.get_legend_handles_labels()
    l2, la2 = ax5b.get_legend_handles_labels()
    ax5.legend(l1 + l2, la1 + la2, fontsize=8, loc="lower left")
    ax5.set_title("Diagnostics")
    ax5.grid(alpha=0.3)

    # Bottom strip: key numbers
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    m = df[df["reliable"]]
    step = np.hypot(m["dx"], m["dy"])
    txt = (
        f"path length: {step.sum():,.0f} px   |   "
        f"mean speed: {step.mean():.3f} px/f   median: {step.median():.3f} px/f   |   "
        f"total yaw: {df['cum_yaw_deg'].iloc[-1]:+.2f} deg   |   "
        f"net scale: x{np.exp(df['cum_log_scale'].iloc[-1]):.4f}   |   "
        f"unreliable: {(~df['reliable']).mean() * 100:.2f}%"
    )
    if len(anchors):
        txt += f"   |   grid re-seed: {len(anchors) + 1} windows"
    ax6.text(0.5, 0.5, txt, ha="center", va="center", fontsize=12,
             family="monospace",
             bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#999999"))

    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", help="CoTracker3 result .npz (tracks, visibility[, meta])")
    ap.add_argument("--fps", type=float, default=None, help="video fps (default: from npz meta, else 23.976)")
    ap.add_argument("--gsd", type=float, default=None, help="ground sample distance, meters/pixel (optional)")
    ap.add_argument("--width", type=int, default=None, help="image width px (default: meta)")
    ap.add_argument("--height", type=int, default=None, help="image height px (default: meta)")
    ap.add_argument("--max-points", type=int, default=20000,
                    help="cap usable correspondences per pair for RANSAC speed")
    ap.add_argument("--window", type=int, default=100,
                    help="re-define grid points every WINDOW frames (anchors at "
                         "0, w-1, 2w-1, ...; e.g. 100 -> windows 0-99, 99-199, "
                         "199-299) so correspondences do not diminish over a "
                         "long video; motion is chained through the shared "
                         "anchor frames. 0 = legacy single grid, whole video")
    ap.add_argument("--out-dir", default=".", help="output directory")
    args = ap.parse_args()

    tracks, visibility, w, h, meta_fps = load_tracks(args.npz)
    T, N, _ = tracks.shape
    width = args.width or w or 3840
    height = args.height or h or 2160
    fps = args.fps or meta_fps or 23.976
    print(f"tracks {tracks.shape}, visibility {visibility.shape}, "
          f"{width}x{height} @ {fps:.3f} fps, GSD={'%.4g m/px' % args.gsd if args.gsd else 'n/a'}")

    if T < 2:
        print("T < 2: no frame pairs; nothing to do.")
        sys.exit(0)

    import os

    os.makedirs(args.out_dir, exist_ok=True)

    df = estimate_motion(tracks, visibility, width, height, args.max_points,
                         args.window)

    csv_path = os.path.join(args.out_dir, "motion.csv")
    out_cols = ["frame", "dx", "dy", "d_yaw_deg", "scale", "inliers",
                "inlier_ratio", "cum_x", "cum_y", "cum_yaw_deg", "reliable",
                "usable", "cum_log_scale", "window", "grid_size"]
    df[out_cols].to_csv(csv_path, index=False)

    png_path = os.path.join(args.out_dir, "trajectory.png")
    title = (f"Camera motion from CoTracker3 tracks ({os.path.basename(args.npz)}) - "
             f"{width}x{height} @ {fps:.3f} fps")
    make_plots(df, png_path, title)

    m = df[df["reliable"]]
    step = np.hypot(m["dx"], m["dy"])
    path_len_px = float(step.sum())
    total_yaw = float(df["cum_yaw_deg"].iloc[-1])
    net_scale = float(np.exp(df["cum_log_scale"].iloc[-1]))
    pct_unrel = float((~df["reliable"]).mean()) * 100

    print("\n===== SUMMARY =====")
    print(f"frames/pairs           : {T} ({len(df) - 1} pairs)")
    if args.window:
        n_windows = int(df["window"].max()) + 1
        if n_windows > 1:
            print(f"grid re-seed           : every {args.window} frames -> "
                  f"{n_windows} windows (0-{args.window - 1}, "
                  f"{args.window - 1}-{2 * args.window - 1}, ...)")
        else:
            print(f"grid re-seed           : video shorter than one window "
                  f"({args.window} frames); single grid used")
    else:
        print("grid re-seed           : off (legacy single grid)")
    print(f"path length            : {path_len_px:,.0f} px", end="")
    if args.gsd:
        print(f"  =  {path_len_px * args.gsd:,.1f} m", end="")
    print()
    print(f"mean speed             : {step.mean():.3f} px/frame", end="")
    if args.gsd:
        print(f"  =  {step.mean() * args.gsd * fps:.3f} m/s", end="")
    print()
    print(f"median speed           : {step.median():.3f} px/frame", end="")
    if args.gsd:
        print(f"  =  {step.median() * args.gsd * fps:.3f} m/s", end="")
    print()
    print(f"total yaw change       : {total_yaw:+.2f} deg")
    print(f"net scale change       : x{net_scale:.4f} ({(net_scale - 1) * 100:+.2f}%)"
          f"  [{'>1 => descended' if net_scale > 1 else '<1 => ascended'}]")
    print(f"unreliable frames      : {pct_unrel:.2f}%")

    if args.window:
        print("\n===== WINDOWS (grid re-defined at each anchor) =====")
        print(f"{'win':>4} {'anchor':>7} {'grid_pts':>9} {'mean_usable':>12} "
              f"{'unreliable':>11}")
        for wi in range(int(df["window"].max()) + 1):
            wdf = df[df["window"] == wi]
            anchor = 0 if wi == 0 else int(wdf["frame"].min()) - 1
            print(f"{wi:>4} {anchor:>7} {int(wdf['grid_size'].iloc[0]):>9} "
                  f"{wdf['usable'].mean():>12.1f} "
                  f"{(~wdf['reliable']).mean() * 100:>10.2f}%")
    print(f"\nwrote: {csv_path}")
    print(f"wrote: {png_path}")


if __name__ == "__main__":
    main()
