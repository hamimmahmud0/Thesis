"""Camera-path smoothing for video stabilisation.

Decomposes cumulative camera transforms into (translation, yaw, log-scale),
applies Gaussian smoothing with short-gap interpolation, and reconstructs
smoothed 3x3 homogeneous transforms.  Optionally also stores the combined
correction warp matrices for direct rendering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import (
    decompose_similarity,
    eye3,
    load_tracks,
    similarity_matrix,
    smooth_decomposed_params,
)


def smooth_motion(
    motion_npz: str | Path,
    output_npz: str | Path,
    sigma: float = 10.0,
    interp_gap: int = 5,
) -> dict:
    """Smooth a camera path and save the result.

    Parameters
    ----------
    motion_npz : path to ``motion.npz`` (output of ``stabilize estimate``).
    output_npz : path to write ``motion_smooth.npz``.
    sigma : Gaussian smoothing sigma in frames.  Larger values produce
        smoother but more delayed camera motion.  Typical range: 5–30.
    interp_gap : unreliable gaps shorter than this are linearly
        interpolated before smoothing.  Longer gaps hold the last
        reliable value.

    Returns
    -------
    dict with keys:
        smoothed_cumulative (T, 3, 3) — smoothed cumulative transforms.
        raw_cumulative      (T, 3, 3) — original cumulative (copy).
        tx, ty, yaw_deg, log_scale — smoothed decomposed parameters.
        reliable           (T,) bool.
    """
    from .motion import load_motion

    motion = load_motion(motion_npz)
    raw_cum = motion["cumulative"]
    reliable = motion["reliable"]
    T = len(reliable)

    # Decompose raw cumulative transforms.
    tx = raw_cum[:, 0, 2]
    ty = raw_cum[:, 1, 2]
    yaw_deg = np.zeros(T)
    log_scale = np.zeros(T)
    for t in range(T):
        _, _, yaw_deg[t], scale_t = decompose_similarity(raw_cum[t])
        log_scale[t] = np.log(max(scale_t, 1e-9))

    # Smooth decomposed parameters.
    stx, sty, syaw, slogs = smooth_decomposed_params(
        tx, ty, yaw_deg, log_scale, reliable, sigma=sigma, interp_range=interp_gap,
    )

    # Reconstruct smoothed cumulative transforms.
    smooth_cum = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    for t in range(T):
        smooth_cum[t] = similarity_matrix(stx[t], sty[t], syaw[t], np.exp(slogs[t]))

    out_path = Path(output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        smoothed_cumulative=smooth_cum,
        raw_cumulative=raw_cum,
        reliable=reliable,
        tx=stx,
        ty=sty,
        yaw_deg=syaw,
        log_scale=slogs,
        sigma=sigma,
        interp_gap=interp_gap,
    )
    print(f"Wrote smoothed path (sigma={sigma}) -> {out_path}")

    return dict(
        smoothed_cumulative=smooth_cum,
        raw_cumulative=raw_cum,
        reliable=reliable,
        tx=stx,
        ty=sty,
        yaw_deg=syaw,
        log_scale=slogs,
    )


def load_smoothed(npz_path: str | Path) -> dict:
    """Load a smoothed-motion .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    return dict(
        smoothed_cumulative=d["smoothed_cumulative"],
        raw_cumulative=d["raw_cumulative"],
        reliable=d["reliable"].astype(bool),
        tx=d["tx"],
        ty=d["ty"],
        yaw_deg=d["yaw_deg"],
        log_scale=d["log_scale"],
        sigma=float(d["sigma"]),
        interp_gap=int(d["interp_gap"]),
    )
