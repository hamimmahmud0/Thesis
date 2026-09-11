"""Camera motion estimation from CoTracker3 keypoint trajectories.

Estimates per-frame camera motion as a 4-DOF similarity transform
(translation, yaw, uniform scale) using RANSAC.  Stores **exact 3x3
homogeneous matrices** for pairwise and cumulative transforms, avoiding
the sign-convention ambiguities of decomposed CSV-only storage.

Conventions
-----------
- ``cv2.estimateAffinePartial2D(pts_prev, pts_curr)`` returns the
  IMAGE-motion M that maps points from frame t-1 to frame t.
- Camera translation = -(image tx, ty); camera yaw = -image yaw.
- ``pairwise[t]`` is the INCREMENTAL image transform (frame t-1 -> frame t).
- ``cumulative[t] = pairwise[t] @ cumulative[t-1]`` is the cumulative
  forward IMAGE transform mapping frame-0 coordinates to frame-t
  coordinates (identity for t=0).

The .npz stores exact 3x3 IMAGE transforms.  The .csv stores a
human-readable summary; its ``cum_x``/``cum_y`` columns are in CAMERA
convention (negated cumulative image translation), consistent with the
per-frame ``dx``/``dy`` camera columns.

Output files
------------
- ``motion.npz`` : pairwise (T,3,3), cumulative (T,3,3), reliable (T,),
  per-frame decomposition (dx, dy, d_yaw_deg, scale, inliers, inlier_ratio,
  usable).
- ``motion.csv`` : human-readable summary with cumulative values.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .utils import (
    decompose_similarity,
    eye3,
    in_bounds,
    load_tracks,
    runs_of,
    similarity_matrix,
)

MIN_CORRESPONDENCES = 8


def _estimate_pair(
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray | None, int, int]:
    """RANSAC similarity fit for one frame pair.

    Returns (M_2x3 or None, inlier_count, n_used).
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
    s = float(np.hypot(M[0, 0], M[1, 0]))
    if s < 1e-6:
        return None, inliers, pts_prev.shape[0]

    return M, inliers, pts_prev.shape[0]


def estimate_motion(
    tracks: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
    max_points: int = 20000,
) -> dict:
    """Estimate per-frame camera motion from keypoint trajectories.

    Parameters
    ----------
    tracks : (T, N, 2) float32 — keypoint positions.
    visibility : (T, N) bool — per-point visibility.
    width, height : image dimensions (for bounds checking).
    max_points : cap usable correspondences per pair for RANSAC speed.

    Returns
    -------
    dict with keys:
        pairwise     (T, 3, 3) float64 — incremental image transforms.
        cumulative   (T, 3, 3) float64 — cumulative forward transforms.
        reliable     (T,) bool
        dx, dy       (T,) float64 — per-frame camera translation.
        d_yaw_deg    (T,) float64 — per-frame camera yaw rate.
        scale        (T,) float64 — per-frame image scale factor.
        inliers      (T,) int
        inlier_ratio (T,) float
        usable       (T,) int
    """
    T, N, _ = tracks.shape

    pairwise = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    cumulative = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    reliable = np.zeros(T, dtype=bool)
    reliable[0] = True

    dx = np.zeros(T, dtype=np.float64)
    dy = np.zeros(T, dtype=np.float64)
    d_yaw = np.zeros(T, dtype=np.float64)
    scale = np.ones(T, dtype=np.float64)
    inliers = np.zeros(T, dtype=np.int64)
    inlier_ratio = np.zeros(T, dtype=np.float64)
    usable = np.zeros(T, dtype=np.int64)

    for t in range(1, T):
        p_prev = tracks[t - 1]
        p_curr = tracks[t]

        mask = visibility[t - 1] & visibility[t]
        if width and height:
            mask &= in_bounds(p_prev, width, height) & in_bounds(
                p_curr, width, height
            )

        n_usable = int(mask.sum())
        usable[t] = n_usable

        if n_usable < MIN_CORRESPONDENCES:
            # Unreliable: carry forward previous cumulative, zero motion.
            pairwise[t] = eye3()
            cumulative[t] = cumulative[t - 1]
            reliable[t] = False
            continue

        M, n_inliers, n_used = _estimate_pair(
            p_prev[mask], p_curr[mask], max_points
        )

        inliers[t] = n_inliers
        inlier_ratio[t] = n_inliers / max(n_used, 1)

        if M is None:
            pairwise[t] = eye3()
            cumulative[t] = cumulative[t - 1]
            reliable[t] = False
            continue

        # M is the IMAGE-motion matrix (2x3).  Embed to 3x3.
        A = eye3()
        A[:2] = M

        # Decompose for CSV / diagnostics (camera convention).
        img_tx, img_ty, img_yaw, s = decompose_similarity(M)
        dx[t] = -img_tx
        dy[t] = -img_ty
        d_yaw[t] = -img_yaw
        scale[t] = s

        # Store exact matrices — no re-decomposition/reconstruction later.
        pairwise[t] = A
        cumulative[t] = A @ cumulative[t - 1]
        reliable[t] = True

    return dict(
        pairwise=pairwise,
        cumulative=cumulative,
        reliable=reliable,
        dx=dx,
        dy=dy,
        d_yaw_deg=d_yaw,
        scale=scale,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        usable=usable,
    )


def save_motion(
    result: dict,
    output_path: str | Path,
    video_width: int | None = None,
    video_height: int | None = None,
    fps: float | None = None,
    tracks_source: str = "",
) -> None:
    """Save motion estimation results as .npz (matrices) and .csv (summary).

    Parameters
    ----------
    result : dict returned by :func:`estimate_motion`.
    output_path : base path (without extension).  Creates ``<path>.npz``
        and ``<path>.csv``.
    video_width, video_height, fps : optional metadata written into the CSV
        header comment and NPZ.
    tracks_source : name of the source tracks file (for the CSV header).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- NPZ: full precision matrices + arrays ----
    npz_path = out.with_suffix(".npz")
    meta = {}
    if video_width is not None:
        meta["width"] = video_width
    if video_height is not None:
        meta["height"] = video_height
    if fps is not None:
        meta["fps"] = fps
    if tracks_source:
        meta["tracks_source"] = tracks_source

    np.savez(
        npz_path,
        pairwise=result["pairwise"],
        cumulative=result["cumulative"],
        reliable=result["reliable"],
        dx=result["dx"],
        dy=result["dy"],
        d_yaw_deg=result["d_yaw_deg"],
        scale=result["scale"],
        inliers=result["inliers"],
        inlier_ratio=result["inlier_ratio"],
        usable=result["usable"],
        meta=json.dumps(meta) if meta else "",
    )

    # ---- CSV: human-readable summary ----
    csv_path = out.with_suffix(".csv")
    T = len(result["reliable"])

    # Recompute cumulative decomposition for the CSV (for human readability).
    # Camera convention: negate the cumulative IMAGE translation, consistent
    # with the per-frame dx/dy (camera) columns.
    cum_x = -result["cumulative"][:, 0, 2]
    cum_y = -result["cumulative"][:, 1, 2]
    cum_yaw = np.zeros(T)
    cum_log_scale = np.zeros(T)
    cyaw = 0.0
    clogs = 0.0
    for t in range(T):
        if result["reliable"][t]:
            cyaw += result["d_yaw_deg"][t]
            clogs += float(np.log(max(result["scale"][t], 1e-9)))
        cum_yaw[t] = cyaw
        cum_log_scale[t] = clogs

    df = pd.DataFrame(dict(
        frame=np.arange(T),
        dx=result["dx"],
        dy=result["dy"],
        d_yaw_deg=result["d_yaw_deg"],
        scale=result["scale"],
        inliers=result["inliers"],
        inlier_ratio=result["inlier_ratio"],
        usable=result["usable"],
        reliable=result["reliable"],
        cum_x=cum_x,
        cum_y=cum_y,
        cum_yaw_deg=cum_yaw,
        cum_log_scale=cum_log_scale,
    ))

    # Write a comment header with metadata.
    with open(csv_path, "w") as f:
        if meta:
            f.write("# " + json.dumps(meta) + "\n")
        df.to_csv(f, index=False)

    print(f"Wrote {npz_path}")
    print(f"Wrote {csv_path}")


def load_motion(npz_path: str | Path) -> dict:
    """Load a motion .npz file (as saved by :func:`save_motion`)."""
    d = np.load(npz_path, allow_pickle=True)
    result = dict(
        pairwise=d["pairwise"],
        cumulative=d["cumulative"],
        reliable=d["reliable"].astype(bool),
        dx=d["dx"],
        dy=d["dy"],
        d_yaw_deg=d["d_yaw_deg"],
        scale=d["scale"],
        inliers=d["inliers"],
        inlier_ratio=d["inlier_ratio"],
        usable=d["usable"],
    )
    if "meta" in d.files:
        try:
            result["meta"] = json.loads(str(d["meta"]))
        except Exception:
            result["meta"] = {}
    else:
        result["meta"] = {}
    return result


def motion_summary(result: dict) -> str:
    """Return a human-readable summary string."""
    reliable = result["reliable"]
    T = len(reliable)
    m = np.where(reliable)[0]
    if len(m) < 2:
        return f"frames: {T}, unreliable: {T}"

    step = np.hypot(result["dx"][m], result["dy"][m])
    path_len = float(step.sum())
    cum = result["cumulative"]
    total_yaw = float(result["d_yaw_deg"][m].sum())
    net_scale = float(np.exp(np.log(result["scale"][m]).sum()))
    pct_unrel = float((~reliable).mean()) * 100

    lines = [
        f"frames / pairs      : {T} ({T - 1} pairs)",
        f"path length         : {path_len:,.0f} px",
        f"mean speed          : {step.mean():.3f} px/frame",
        f"median speed        : {float(np.median(step)):.3f} px/frame",
        f"total yaw change    : {total_yaw:+.2f} deg",
        f"net scale change    : x{net_scale:.4f} ({(net_scale - 1) * 100:+.2f}%)",
        f"unreliable frames   : {pct_unrel:.2f}%",
    ]
    return "\n".join(lines)
