"""Camera motion estimation from CoTracker3 keypoint trajectories.

Architecture (drift-free, anchor-relative)
------------------------------------------
The camera trajectory is built from **persistent correspondences to periodic
anchor frames**.  Consecutive frame-to-frame transforms are *never* composed
to form the long-term pose -- that accumulates the tiny error of every
RANSAC fit and produces the classic "150 px over a 15-minute video" drift.

Transform conventions (identical to :mod:`stabilize.utils`)
-----------------------------------------------------------
- A transform is a 3x3 homogeneous matrix ``M`` acting on column vectors
  ``p = [x, y, 1]^T`` so that ``p_dst = M @ p_src``.
- ``M_{a->b}`` maps coordinates in frame ``a`` to coordinates in frame ``b``.
- ``compose(A, B) == A @ B`` applies ``B`` first, then ``A``.

Definitions used below
----------------------
``local[t]``
    ``M_{t -> K}``: transform mapping **frame t -> its segment anchor K**
    (current frame -> local anchor).  Estimated *directly* by RANSAC from
    points visible in both frame ``t`` and frame ``K``.  For ``t == K`` it is
    the identity.

``anchor_global[i]``
    ``M_{K_i -> 0}``: transform mapping **anchor K_i -> global (frame 0)
    coordinates** (anchor -> global).  ``anchor_global[0] = I``.  Consecutive
    anchors are related by a single RANSAC fit over their shared visible
    points: ``anchor_global[i] = anchor_global[i-1] @ M_{K_i -> K_{i-1}}``.
    Errors therefore accumulate once per anchor interval (default 10 s),
    never once per frame.

``frame_to_global[t]``
    ``M_{t -> 0} = anchor_global[i] @ local[t]`` -- the composed
    current-frame -> global transform (this is the transform drawn in the
    task diagram: frame -> local anchor -> global).

``cumulative[t]``
    ``M_{0 -> t} = inv(frame_to_global[t])``: the measured camera trajectory
    in the **frame 0 -> frame t** direction.  This is the convention the
    renderer/smoother already used, so it is stored inverted from the
    diagram's ``frame_to_global`` for drop-in compatibility.  At ``t == 0``
    it is the identity.

``pairwise[t]``
    ``M_{t-1 -> t}``, derived by *differencing the already-built global
    trajectory* (``cumulative[t] @ inv(cumulative[t-1])``).  It is used for
    diagnostics only and is never integrated into the trajectory.

Output files
------------
- ``motion.npz`` : ``pairwise``, ``cumulative``, ``local``, ``reliable``,
  per-frame decomposition (dx, dy, d_yaw_deg, scale), quality arrays
  (num_candidates, num_inliers, inlier_ratio, transform_valid,
  fallback_used), and anchor bookkeeping (anchor_frames, anchor_global,
  anchor_valid, anchor_index, anchor_* quality).
- ``motion.csv`` : human-readable summary with cumulative values.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .utils import (
    affine_2x3_to_3x3,
    blend_similarity,
    compose,
    decompose_similarity,
    eye3,
    in_bounds,
    invert_affine,
    load_tracks,  # re-exported for callers that used ``motion.load_tracks``
)

# Defaults (exposed through the CLI / runner; do not hardcode deep in callers).
DEFAULT_ANCHOR_INTERVAL_SECONDS = 10.0
DEFAULT_RANSAC_REPROJ_THRESHOLD = 2.0
DEFAULT_RANSAC_MAX_ITERS = 5000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_MIN_CORRESPONDENCES = 20
DEFAULT_MIN_INLIER_RATIO = 0.4
DEFAULT_MAX_POINTS = 20000
DEFAULT_ANCHOR_BLEND_FRAMES = 0


# ---------------------------------------------------------------------------
# Single transform estimation
# ---------------------------------------------------------------------------

def estimate_affine(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    max_iters: int = DEFAULT_RANSAC_MAX_ITERS,
    confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    min_correspondences: int = DEFAULT_MIN_CORRESPONDENCES,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    max_points: int = DEFAULT_MAX_POINTS,
) -> tuple[np.ndarray | None, dict]:
    """Robust 4-DOF similarity fit mapping ``src -> dst``.

    ``cv2.estimateAffinePartial2D`` is used (translation + rotation + uniform
    scale) rather than a full homography, so shearing / projective
    deformation is not introduced.

    Parameters
    ----------
    src, dst : (n, 2) correspondences in source and destination frames.
    ransac_reproj_threshold : RANSAC inlier threshold in pixels.
    max_iters, confidence : passed through to OpenCV.
    min_correspondences : minimum usable correspondences to attempt a fit.
    min_inlier_ratio : below this the transform is reported ``valid=False``.
    max_points : cap correspondences fed to RANSAC (speed).

    Returns
    -------
    (M, info)
        ``M`` is a 3x3 homogeneous matrix mapping ``src -> dst`` (or None if
        the fit failed catastrophically).  ``info`` always contains
        ``num_candidates``, ``num_inliers``, ``inlier_ratio`` and ``valid``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = int(src.shape[0])

    info = {
        "num_candidates": n,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "valid": False,
    }

    if n < min_correspondences:
        return None, info

    if n > max_points:
        sel = np.linspace(0, n - 1, max_points).astype(np.int64)
        src = src[sel]
        dst = dst[sel]

    try:
        M, inl = cv2.estimateAffinePartial2D(
            src.reshape(-1, 1, 2).astype(np.float32),
            dst.reshape(-1, 1, 2).astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_threshold),
            maxIters=int(max_iters),
            confidence=float(confidence),
        )
    except cv2.error:
        return None, info

    if M is None or inl is None or not np.isfinite(M).all():
        return None, info

    n_inliers = int(inl.sum())
    info["num_inliers"] = n_inliers
    info["inlier_ratio"] = n_inliers / max(n, 1)

    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if scale < 1e-6:
        return None, info

    info["valid"] = (
        n_inliers >= min_correspondences
        and info["inlier_ratio"] >= min_inlier_ratio
    )
    return affine_2x3_to_3x3(M), info


def _merge_info(*infos: dict) -> dict:
    """Combine quality info from a bridged (multi-hop) estimation."""
    cand = sum(i.get("num_candidates", 0) for i in infos)
    inl = sum(i.get("num_inliers", 0) for i in infos)
    return {
        "num_candidates": cand,
        "num_inliers": inl,
        "inlier_ratio": inl / max(cand, 1),
        "valid": True,
        "bridged": True,
    }


def _estimate_between(
    tracks: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
    t_src: int,
    t_dst: int,
    params: dict,
    depth: int = 0,
) -> tuple[np.ndarray | None, dict, bool]:
    """Estimate ``M_{t_src -> t_dst}`` from persistent correspondences.

    ``track[t_src]`` are the source points and ``track[t_dst]`` the
    destination points, so the returned matrix maps frame ``t_src``
    coordinates into frame ``t_dst`` coordinates.

    If the direct fit is not good enough, the interval is bisected and the
    two half-transforms are composed (recursively).  This only ever bridges
    *within* one anchor interval during a fallback, so it cannot reintroduce
    global frame-level drift.

    Returns ``(M_3x3 | None, info, used_bridge)``.
    """
    p_src = tracks[t_src]
    p_dst = tracks[t_dst]

    mask = visibility[t_src] & visibility[t_dst]
    mask &= np.isfinite(p_src).all(axis=1) & np.isfinite(p_dst).all(axis=1)
    if width and height:
        mask &= in_bounds(p_src, width, height) & in_bounds(p_dst, width, height)

    M, info = estimate_affine(p_src[mask], p_dst[mask], **params)
    if info["valid"] and M is not None:
        return M, info, False

    gap = abs(int(t_dst) - int(t_src))
    max_depth = int(params.get("bridge_max_depth", 6))
    if gap > 1 and depth < max_depth:
        mid = (int(t_src) + int(t_dst)) // 2
        M1, i1, _ = _estimate_between(
            tracks, visibility, width, height, t_src, mid, params, depth + 1
        )
        if M1 is not None and i1["valid"]:
            M2, i2, _ = _estimate_between(
                tracks, visibility, width, height, mid, t_dst, params, depth + 1
            )
            if M2 is not None and i2["valid"]:
                # M1: src -> mid, M2: mid -> dst  =>  compose(M2, M1) = dst@src.
                return compose(M2, M1), _merge_info(i1, i2), True

    return None, info, False


# ---------------------------------------------------------------------------
# Anchor bookkeeping
# ---------------------------------------------------------------------------

def anchor_frames_for(T: int, fps: float, anchor_interval_seconds: float) -> np.ndarray:
    """Return the anchor frame indices for a video of ``T`` frames.

    Anchors are ``0, step, 2*step, ...`` with ``step`` = round(fps *
    anchor_interval_seconds).  The final segment (after the last anchor)
    simply uses the last anchor.
    """
    if T <= 1:
        return np.array([0], dtype=np.int64)
    step = max(1, int(round(max(float(anchor_interval_seconds), 1e-6) * max(float(fps), 1e-6))))
    return np.arange(0, T, step, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def estimate_motion(
    tracks: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
    max_points: int = DEFAULT_MAX_POINTS,
    anchor_interval_seconds: float = DEFAULT_ANCHOR_INTERVAL_SECONDS,
    fps: float = 23.976,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    max_iters: int = DEFAULT_RANSAC_MAX_ITERS,
    confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    min_correspondences: int = DEFAULT_MIN_CORRESPONDENCES,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    anchor_blend_frames: int = DEFAULT_ANCHOR_BLEND_FRAMES,
    debug: bool = False,
) -> dict:
    """Estimate the camera trajectory relative to periodic anchor frames.

    Every frame is fit *directly* against its local anchor and then placed
    into the global (frame-0) coordinate system through one anchor->global
    composition.  See the module docstring for the full convention table.

    Returns a dict with ``cumulative`` (T,3,3) mapping frame 0 -> frame t,
    plus the diagnostic / quality arrays documented in the module header.
    """
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    T = int(tracks.shape[0])
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be (T, N, 2), got {tracks.shape}")
    if T == 0:
        raise ValueError("tracks has zero frames")
    N = int(tracks.shape[1])

    interval = max(1, int(round(max(float(anchor_interval_seconds), 1e-6) * max(float(fps), 1e-6))))
    anchor_frames = anchor_frames_for(T, fps, anchor_interval_seconds)
    n_anchors = len(anchor_frames)

    # Greatest anchor index <= t for every frame.
    anchor_index = np.searchsorted(anchor_frames, np.arange(T), side="right") - 1
    anchor_index = np.clip(anchor_index, 0, n_anchors - 1)

    est_params = dict(
        ransac_reproj_threshold=float(ransac_reproj_threshold),
        max_iters=int(max_iters),
        confidence=float(confidence),
        min_correspondences=int(min_correspondences),
        min_inlier_ratio=float(min_inlier_ratio),
        max_points=int(max_points),
    )

    # ---- 1. Anchor -> global transforms (one composition per anchor) ----
    anchor_global = np.tile(eye3(), (n_anchors, 1, 1))
    anchor_valid = np.zeros(n_anchors, dtype=bool)
    anchor_valid[0] = True
    anchor_num_candidates = np.zeros(n_anchors, dtype=np.int64)
    anchor_num_inliers = np.zeros(n_anchors, dtype=np.int64)
    anchor_inlier_ratio = np.zeros(n_anchors, dtype=np.float64)
    anchor_bridged = np.zeros(n_anchors, dtype=bool)

    for i in range(1, n_anchors):
        k_cur = int(anchor_frames[i])
        k_prev = int(anchor_frames[i - 1])
        # M maps anchor K_i -> anchor K_{i-1}.
        M, info, bridged = _estimate_between(
            tracks, visibility, width, height, k_cur, k_prev, est_params
        )
        anchor_num_candidates[i] = info["num_candidates"]
        anchor_num_inliers[i] = info["num_inliers"]
        anchor_inlier_ratio[i] = info["inlier_ratio"]
        anchor_bridged[i] = bridged
        if info["valid"] and M is not None:
            # anchor_global[i] : K_i -> global = (K_{i-1} -> global) @ (K_i -> K_{i-1})
            anchor_global[i] = compose(anchor_global[i - 1], M)
            anchor_valid[i] = True
        else:
            # Hold the previous anchor pose; this segment is flagged invalid.
            anchor_global[i] = anchor_global[i - 1].copy()
            anchor_valid[i] = False

    # ---- 2. Frame -> local anchor transforms (direct, no chaining) ----
    local = np.tile(eye3(), (T, 1, 1))          # M_{t -> K}
    transform_valid = np.zeros(T, dtype=bool)
    fallback_used = np.zeros(T, dtype=bool)
    num_candidates = np.zeros(T, dtype=np.int64)
    num_inliers = np.zeros(T, dtype=np.int64)
    inlier_ratio = np.zeros(T, dtype=np.float64)

    for i in range(n_anchors):
        K = int(anchor_frames[i])
        end = int(anchor_frames[i + 1]) if i + 1 < n_anchors else T
        local[K] = eye3()
        transform_valid[K] = True
        prev_valid = eye3()
        have_prev = True
        for t in range(K + 1, end):
            M, info, _ = _estimate_between(
                tracks, visibility, width, height, t, K, est_params
            )
            num_candidates[t] = info["num_candidates"]
            num_inliers[t] = info["num_inliers"]
            inlier_ratio[t] = info["inlier_ratio"]
            if info["valid"] and M is not None:
                local[t] = M
                transform_valid[t] = True
                prev_valid = M
                have_prev = True
            else:
                # Graceful fallback: hold the previous anchor-relative
                # transform, or identity if none exists yet.
                local[t] = prev_valid if have_prev else eye3()
                transform_valid[t] = False
                fallback_used[t] = True

    # ---- 3. Global trajectory ----
    # frame_to_global[t] = M_{t -> 0} = anchor_global[K] @ local[t]
    # cumulative[t]      = M_{0 -> t} = inv(frame_to_global[t])
    frame_to_global = np.tile(eye3(), (T, 1, 1))
    cumulative = np.tile(eye3(), (T, 1, 1))
    for t in range(T):
        frame_to_global[t] = compose(anchor_global[anchor_index[t]], local[t])
        cumulative[t] = invert_affine(frame_to_global[t])

    # ---- 4. Optional short blend across anchor transitions ----
    # Position is already continuous at anchors by construction (the
    # anchor-to-anchor fit equals the previous segment's prediction at the
    # anchor).  The blend below additionally smooths the *velocity* kink.
    blend = max(0, int(anchor_blend_frames))
    if blend > 0:
        cumulative = _blend_anchor_transitions(
            cumulative,
            tracks,
            visibility,
            width,
            height,
            anchor_frames,
            anchor_global,
            anchor_index,
            local,
            est_params,
            blend,
        )

    # ---- 5. Derived per-frame motion (diagnostics only) ----
    pairwise = np.tile(eye3(), (T, 1, 1))
    for t in range(1, T):
        # cumulative[t] = M_{0->t}; cumulative[t-1] = M_{0->t-1}
        # M_{t-1 -> t} = cumulative[t] @ inv(cumulative[t-1])
        pairwise[t] = compose(cumulative[t], invert_affine(cumulative[t - 1]))

    dx = np.zeros(T)
    dy = np.zeros(T)
    d_yaw = np.zeros(T)
    scale = np.ones(T)
    for t in range(T):
        img_tx, img_ty, img_yaw, s = decompose_similarity(pairwise[t])
        dx[t] = -img_tx
        dy[t] = -img_ty
        d_yaw[t] = -img_yaw
        scale[t] = s

    reliable = transform_valid & anchor_valid[anchor_index]

    if debug:
        print(
            f"[estimate] {T} frames, {N} points, {n_anchors} anchors "
            f"(interval {interval} frames = {anchor_interval_seconds:g}s @ {fps:.3f} fps)"
        )
        for i in range(1, n_anchors):
            print(
                f"[anchor {i:4d}] frame={int(anchor_frames[i]):6d} "
                f"cand={anchor_num_candidates[i]:6d} "
                f"inl={anchor_num_inliers[i]:6d} "
                f"ratio={anchor_inlier_ratio[i]:.3f} "
                f"bridged={bool(anchor_bridged[i])} "
                f"valid={bool(anchor_valid[i])}"
            )
        print(
            f"[estimate] reliable frames: {reliable.mean() * 100:.2f}%  "
            f"fallbacks: {fallback_used.mean() * 100:.2f}%"
        )

    return dict(
        # Core transforms.
        pairwise=pairwise,
        cumulative=cumulative,
        frame_to_global=frame_to_global,
        local=local,
        reliable=reliable,
        # Per-frame decomposition (camera convention).
        dx=dx,
        dy=dy,
        d_yaw_deg=d_yaw,
        scale=scale,
        # Per-frame quality (aliases kept for the existing viz/CSV).
        inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        usable=num_candidates,
        num_candidates=num_candidates,
        num_inliers=num_inliers,
        transform_valid=transform_valid,
        fallback_used=fallback_used,
        # Anchor bookkeeping.
        anchor_frames=anchor_frames,
        anchor_global=anchor_global,
        anchor_valid=anchor_valid,
        anchor_index=anchor_index,
        anchor_num_candidates=anchor_num_candidates,
        anchor_num_inliers=anchor_num_inliers,
        anchor_inlier_ratio=anchor_inlier_ratio,
        anchor_bridged=anchor_bridged,
        anchor_interval_seconds=float(anchor_interval_seconds),
        anchor_interval_frames=int(interval),
        method="anchor_relative",
    )


def _blend_anchor_transitions(
    cumulative: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
    anchor_frames: np.ndarray,
    anchor_global: np.ndarray,
    anchor_index: np.ndarray,
    local: np.ndarray,
    est_params: dict,
    blend: int,
) -> np.ndarray:
    """Cross-fade the first ``blend`` frames of each segment with the
    previous anchor's continuation, smoothing the velocity at transitions.

    At the anchor frame both predictions coincide (continuity is exact), so
    this cannot introduce a jump.
    """
    out = cumulative.copy()
    T = cumulative.shape[0]
    for i in range(1, len(anchor_frames)):
        K = int(anchor_frames[i])
        K_prev = int(anchor_frames[i - 1])
        for d in range(0, min(blend, T - K)):
            t = K + d
            M_prev, info_prev, _ = _estimate_between(
                tracks, visibility, width, height, t, K_prev, est_params
            )
            if M_prev is None or not info_prev["valid"]:
                continue
            # Previous segment's frame t -> global transform, converted to the
            # stored frame 0 -> frame t convention before blending.
            global_prev = compose(anchor_global[i - 1], M_prev)
            cum_prev = invert_affine(global_prev)
            w = (d + 1) / (blend + 1.0)
            out[t] = blend_similarity(out[t], cum_prev, w)
    return out


# ---------------------------------------------------------------------------
# Legacy estimator (frame-to-frame integration) -- for A/B drift comparison
# ---------------------------------------------------------------------------

def estimate_motion_pairwise(
    tracks: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
    max_points: int = DEFAULT_MAX_POINTS,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    max_iters: int = DEFAULT_RANSAC_MAX_ITERS,
    confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    min_correspondences: int = DEFAULT_MIN_CORRESPONDENCES,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
) -> dict:
    """**Deprecated** frame-to-frame estimator that accumulates drift.

    Kept only so the old and new long-horizon behaviour can be compared
    (see ``stabilize estimate --legacy``).  It composes
    ``cumulative[t] = pairwise[t] @ cumulative[t-1]`` for every frame, so
    each RANSAC fit's error is integrated into the camera pose.
    """
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    T = int(tracks.shape[0])

    est_params = dict(
        ransac_reproj_threshold=float(ransac_reproj_threshold),
        max_iters=int(max_iters),
        confidence=float(confidence),
        min_correspondences=int(min_correspondences),
        min_inlier_ratio=float(min_inlier_ratio),
        max_points=int(max_points),
    )

    pairwise = np.tile(eye3(), (T, 1, 1))
    cumulative = np.tile(eye3(), (T, 1, 1))
    reliable = np.zeros(T, dtype=bool)
    reliable[0] = True
    dx = np.zeros(T)
    dy = np.zeros(T)
    d_yaw = np.zeros(T)
    scale = np.ones(T)
    num_inliers = np.zeros(T, dtype=np.int64)
    inlier_ratio = np.zeros(T)
    num_candidates = np.zeros(T, dtype=np.int64)
    transform_valid = np.zeros(T, dtype=bool)
    transform_valid[0] = True
    fallback_used = np.zeros(T, dtype=bool)

    for t in range(1, T):
        M, info, _ = _estimate_between(
            tracks, visibility, width, height, t, t - 1, est_params
        )
        num_candidates[t] = info["num_candidates"]
        num_inliers[t] = info["num_inliers"]
        inlier_ratio[t] = info["inlier_ratio"]

        if info["valid"] and M is not None:
            # M = M_{t -> t-1}.  Store pairwise as M_{t-1 -> t} = inv(M) to
            # match the new estimator's convention.
            pairwise[t] = invert_affine(M)
            # cumulative maps frame 0 -> t:
            #   cumulative[t] = M_{t-1 -> t} @ cumulative[t-1]
            cumulative[t] = compose(pairwise[t], cumulative[t - 1])
            transform_valid[t] = True
            reliable[t] = True
        else:
            pairwise[t] = eye3()
            cumulative[t] = cumulative[t - 1].copy()
            reliable[t] = False
            fallback_used[t] = True

        img_tx, img_ty, img_yaw, s = decompose_similarity(pairwise[t])
        dx[t] = -img_tx
        dy[t] = -img_ty
        d_yaw[t] = -img_yaw
        scale[t] = s

    anchor_frames = np.array([0], dtype=np.int64)
    return dict(
        pairwise=pairwise,
        cumulative=cumulative,
        frame_to_global=cumulative.copy(),
        local=cumulative.copy(),
        reliable=reliable,
        dx=dx,
        dy=dy,
        d_yaw_deg=d_yaw,
        scale=scale,
        inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        usable=num_candidates,
        num_candidates=num_candidates,
        num_inliers=num_inliers,
        transform_valid=transform_valid,
        fallback_used=fallback_used,
        anchor_frames=anchor_frames,
        anchor_global=np.tile(eye3(), (1, 1, 1)),
        anchor_valid=np.array([True]),
        anchor_index=np.zeros(T, dtype=np.int64),
        anchor_num_candidates=np.array([num_candidates.max() if T else 0]),
        anchor_num_inliers=np.array([num_inliers.max() if T else 0]),
        anchor_inlier_ratio=np.array([inlier_ratio.max() if T else 0.0]),
        anchor_bridged=np.array([False]),
        anchor_interval_seconds=0.0,
        anchor_interval_frames=1,
        method="pairwise_legacy",
    )


# ---------------------------------------------------------------------------
# Drift diagnostics
# ---------------------------------------------------------------------------

def drift_diagnostic(
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchor: int = 0,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Measure absolute point displacement relative to a single anchor frame.

    For anchor frame ``k`` and every frame ``t`` where enough points are
    visible in both, the robust (median) displacement is::

        valid = visibility[k] & visibility[t]
        delta = tracks[t, valid] - tracks[k, valid]
        absolute_dx[t] = median(delta[:, 0])
        absolute_dy[t] = median(delta[:, 1])

    Because this uses only CoTracker trajectories, it isolates drift that is
    *already present in the tracker output* from drift introduced later by
    transform integration.
    """
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    T = int(tracks.shape[0])
    k = int(min(max(anchor, 0), T - 1))

    absolute_dx = np.zeros(T)
    absolute_dy = np.zeros(T)
    num_valid = np.zeros(T, dtype=np.int64)

    vis_k = visibility[k]
    p_k = tracks[k]
    for t in range(T):
        valid = vis_k & visibility[t]
        valid &= np.isfinite(p_k).all(axis=1) & np.isfinite(tracks[t]).all(axis=1)
        if width and height:
            valid &= in_bounds(p_k, width, height) & in_bounds(tracks[t], width, height)
        n = int(valid.sum())
        num_valid[t] = n
        if n == 0:
            continue
        delta = tracks[t, valid] - p_k[valid]
        absolute_dx[t] = float(np.median(delta[:, 0]))
        absolute_dy[t] = float(np.median(delta[:, 1]))

    return dict(
        anchor=k,
        absolute_dx=absolute_dx,
        absolute_dy=absolute_dy,
        num_valid=num_valid,
    )


def segment_drift_diagnostic(
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchor_frames: np.ndarray,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Absolute displacement measured relative to *each frame's own anchor*.

    Unlike :func:`drift_diagnostic` (which uses a single global anchor), this
    resets the reference every anchor interval, so the reported displacement
    is bounded by the anchor spacing and reveals per-segment tracker error.
    """
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    T = int(tracks.shape[0])
    anchor_frames = np.asarray(anchor_frames, dtype=np.int64)
    if anchor_frames.size == 0:
        anchor_frames = np.array([0], dtype=np.int64)

    anchor_index = np.searchsorted(anchor_frames, np.arange(T), side="right") - 1
    anchor_index = np.clip(anchor_index, 0, len(anchor_frames) - 1)

    dx = np.zeros(T)
    dy = np.zeros(T)
    num_valid = np.zeros(T, dtype=np.int64)

    for i, k in enumerate(anchor_frames):
        k = int(k)
        vis_k = visibility[k]
        p_k = tracks[k]
        for t in np.where(anchor_index == i)[0]:
            valid = vis_k & visibility[t]
            valid &= np.isfinite(p_k).all(axis=1) & np.isfinite(tracks[t]).all(axis=1)
            if width and height:
                valid &= in_bounds(p_k, width, height) & in_bounds(tracks[t], width, height)
            n = int(valid.sum())
            num_valid[t] = n
            if n == 0:
                continue
            delta = tracks[t, valid] - p_k[valid]
            dx[t] = float(np.median(delta[:, 0]))
            dy[t] = float(np.median(delta[:, 1]))

    return dict(
        absolute_dx=dx,
        absolute_dy=dy,
        num_valid=num_valid,
        anchor_frames=anchor_frames,
    )


def save_drift(
    result: dict,
    output_path: str | Path,
    fps: float | None = None,
    anchor_frames: np.ndarray | None = None,
) -> None:
    """Save drift-diagnostic arrays as ``<path>.npz`` and ``<path>.csv``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dx = np.asarray(result["absolute_dx"])
    dy = np.asarray(result["absolute_dy"])
    nv = np.asarray(result["num_valid"])
    T = len(dx)
    frames = np.arange(T)
    speed = np.hypot(dx, dy)
    time_s = frames / fps if fps else frames.astype(float)

    meta = {}
    if "anchor" in result:
        meta["anchor"] = int(result["anchor"])
    if fps is not None:
        meta["fps"] = float(fps)
    if anchor_frames is not None:
        meta["anchor_frames"] = [int(a) for a in np.asarray(anchor_frames).ravel()]

    npz_payload = dict(
        absolute_dx=dx,
        absolute_dy=dy,
        num_valid=nv,
        frame=frames,
    )
    if anchor_frames is not None:
        npz_payload["anchor_frames"] = np.asarray(anchor_frames, dtype=np.int64)
    np.savez(out.with_suffix(".npz"), **npz_payload, meta=json.dumps(meta))

    df = pd.DataFrame(dict(
        frame=frames,
        time_s=time_s,
        absolute_dx=dx,
        absolute_dy=dy,
        absolute_mag=speed,
        num_valid=nv,
    ))
    with open(out.with_suffix(".csv"), "w") as f:
        f.write("# " + json.dumps(meta) + "\n")
        df.to_csv(f, index=False)

    print(f"Wrote {out.with_suffix('.npz')}")
    print(f"Wrote {out.with_suffix('.csv')}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_motion(
    result: dict,
    output_path: str | Path,
    video_width: int | None = None,
    video_height: int | None = None,
    fps: float | None = None,
    tracks_source: str = "",
) -> None:
    """Save motion estimation results as .npz (matrices) and .csv (summary)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    npz_path = out.with_suffix(".npz")
    meta = {}
    if video_width is not None:
        meta["width"] = int(video_width)
    if video_height is not None:
        meta["height"] = int(video_height)
    if fps is not None:
        meta["fps"] = float(fps)
    if tracks_source:
        meta["tracks_source"] = tracks_source
    meta["method"] = result.get("method", "anchor_relative")
    meta["anchor_interval_seconds"] = float(result.get("anchor_interval_seconds", 0.0))
    meta["anchor_interval_frames"] = int(result.get("anchor_interval_frames", 1))

    np.savez(
        npz_path,
        pairwise=result["pairwise"],
        cumulative=result["cumulative"],
        frame_to_global=result.get("frame_to_global", result["cumulative"]),
        local=result.get("local", result["cumulative"]),
        reliable=result["reliable"],
        dx=result["dx"],
        dy=result["dy"],
        d_yaw_deg=result["d_yaw_deg"],
        scale=result["scale"],
        inliers=result["inliers"],
        inlier_ratio=result["inlier_ratio"],
        usable=result["usable"],
        num_candidates=result.get("num_candidates", result["usable"]),
        num_inliers=result.get("num_inliers", result["inliers"]),
        transform_valid=result.get("transform_valid", result["reliable"]),
        fallback_used=result.get("fallback_used", np.zeros(len(result["reliable"]), bool)),
        anchor_frames=result.get("anchor_frames", np.array([0], np.int64)),
        anchor_global=result.get("anchor_global", np.tile(eye3(), (1, 1, 1))),
        anchor_valid=result.get("anchor_valid", np.array([True])),
        anchor_index=result.get("anchor_index", np.zeros(len(result["reliable"]), np.int64)),
        anchor_num_candidates=result.get("anchor_num_candidates", np.array([0], np.int64)),
        anchor_num_inliers=result.get("anchor_num_inliers", np.array([0], np.int64)),
        anchor_inlier_ratio=result.get("anchor_inlier_ratio", np.array([0.0])),
        anchor_bridged=result.get("anchor_bridged", np.array([False])),
        meta=json.dumps(meta),
    )

    # ---- CSV: human-readable summary ----
    T = len(result["reliable"])
    cumulative = result["cumulative"]
    cum_x = -cumulative[:, 0, 2]
    cum_y = -cumulative[:, 1, 2]

    # Cumulative yaw / log-scale from the exact cumulative matrices.
    cum_yaw = np.zeros(T)
    cum_log_scale = np.zeros(T)
    for t in range(T):
        _, _, yaw_t, scale_t = decompose_similarity(cumulative[t])
        cum_yaw[t] = -yaw_t
        cum_log_scale[t] = float(np.log(max(scale_t, 1e-9)))

    anchor_frames = np.asarray(result.get("anchor_frames", [0]))
    anchor_index = np.asarray(
        result.get("anchor_index", np.zeros(T, dtype=np.int64))
    )
    if len(anchor_index) != T:
        anchor_index = np.zeros(T, dtype=np.int64)
    anchor_frame_col = anchor_frames[np.clip(anchor_index, 0, len(anchor_frames) - 1)]

    df = pd.DataFrame(dict(
        frame=np.arange(T),
        anchor_frame=anchor_frame_col,
        dx=result["dx"],
        dy=result["dy"],
        d_yaw_deg=result["d_yaw_deg"],
        scale=result["scale"],
        inliers=result["inliers"],
        inlier_ratio=result["inlier_ratio"],
        usable=result["usable"],
        transform_valid=result.get(
            "transform_valid", result["reliable"]
        ),
        fallback_used=result.get(
            "fallback_used", np.zeros(T, dtype=bool)
        ),
        reliable=result["reliable"],
        cum_x=cum_x,
        cum_y=cum_y,
        cum_yaw_deg=cum_yaw,
        cum_log_scale=cum_log_scale,
    ))

    with open(out.with_suffix(".csv"), "w") as f:
        if meta:
            f.write("# " + json.dumps(meta) + "\n")
        df.to_csv(f, index=False)

    print(f"Wrote {npz_path}")
    print(f"Wrote {out.with_suffix('.csv')}")


def load_motion(npz_path: str | Path) -> dict:
    """Load a motion .npz file (as saved by :func:`save_motion`).

    Files written by the older frame-to-frame estimator lack the anchor
    arrays; sensible defaults are synthesised so downstream code (smoother,
    renderer, viz) keeps working unchanged.
    """
    d = np.load(npz_path, allow_pickle=True)
    T = len(d["reliable"])

    def _get(key, default):
        return d[key] if key in d.files else default

    result = dict(
        pairwise=d["pairwise"],
        cumulative=d["cumulative"],
        frame_to_global=_get(
            "frame_to_global", np.linalg.inv(d["cumulative"])
        ),
        local=_get("local", d["cumulative"]),
        reliable=d["reliable"].astype(bool),
        dx=d["dx"],
        dy=d["dy"],
        d_yaw_deg=d["d_yaw_deg"],
        scale=d["scale"],
        inliers=_get("inliers", np.zeros(T, dtype=np.int64)),
        inlier_ratio=_get("inlier_ratio", np.zeros(T)),
        usable=_get("usable", np.zeros(T, dtype=np.int64)),
        num_candidates=_get("num_candidates", _get("usable", np.zeros(T, dtype=np.int64))),
        num_inliers=_get("num_inliers", _get("inliers", np.zeros(T, dtype=np.int64))),
        transform_valid=_get("transform_valid", d["reliable"].astype(bool)),
        fallback_used=_get("fallback_used", np.zeros(T, dtype=bool)),
        anchor_frames=_get("anchor_frames", np.array([0], dtype=np.int64)),
        anchor_global=_get("anchor_global", np.tile(eye3(), (1, 1, 1))),
        anchor_valid=_get("anchor_valid", np.array([True])),
        anchor_index=_get("anchor_index", np.zeros(T, dtype=np.int64)),
        anchor_num_candidates=_get("anchor_num_candidates", np.array([0], dtype=np.int64)),
        anchor_num_inliers=_get("anchor_num_inliers", np.array([0], dtype=np.int64)),
        anchor_inlier_ratio=_get("anchor_inlier_ratio", np.array([0.0])),
        anchor_bridged=_get("anchor_bridged", np.array([False])),
    )
    if "meta" in d.files:
        try:
            result["meta"] = json.loads(str(d["meta"]))
        except Exception:
            result["meta"] = {}
    else:
        result["meta"] = {}
    result["method"] = result["meta"].get("method", "legacy")
    return result


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def stabilization_stats(cumulative: np.ndarray) -> dict:
    """Long-horizon translation statistics of a measured/reference path.

    ``cumulative[t]`` maps frame 0 -> frame t (image motion).  Camera
    translation is the negated image translation, so net displacement is the
    difference between the last and first camera positions.
    """
    cum = np.asarray(cumulative, dtype=np.float64)
    if cum.ndim != 3 or cum.shape[0] == 0:
        return {}
    cam_x = -cum[:, 0, 2]
    cam_y = -cum[:, 1, 2]
    net = float(np.hypot(cam_x[-1] - cam_x[0], cam_y[-1] - cam_y[0]))
    disp = np.hypot(cam_x - cam_x[0], cam_y - cam_y[0])
    return dict(
        start=(float(cam_x[0]), float(cam_y[0])),
        end=(float(cam_x[-1]), float(cam_y[-1])),
        net_translation_px=net,
        max_translation_px=float(disp.max()),
        path_length_px=float(np.hypot(np.diff(cam_x), np.diff(cam_y)).sum()),
    )


def motion_summary(result: dict) -> str:
    """Return a human-readable summary string."""
    reliable = result["reliable"]
    T = len(reliable)
    m = np.where(reliable)[0]
    method = result.get("method", "anchor_relative")
    stats = stabilization_stats(result["cumulative"])

    if len(m) < 2:
        return f"frames: {T}, unreliable: {T}"

    step = np.hypot(result["dx"][m], result["dy"][m])
    path_len = float(step.sum())
    total_yaw = float(result["d_yaw_deg"][m].sum())
    net_scale = float(np.exp(np.log(np.clip(result["scale"][m], 1e-9, None)).sum()))
    pct_unrel = float((~reliable).mean()) * 100
    pct_fallback = float(
        np.asarray(result.get("fallback_used", np.zeros(T, bool))).mean()
    ) * 100

    lines = [
        f"method              : {method}",
        f"frames / pairs      : {T} ({T - 1} pairs)",
        f"anchor interval     : {result.get('anchor_interval_frames', 1)} frames "
        f"({result.get('anchor_interval_seconds', 0.0):g}s), "
        f"{len(np.asarray(result.get('anchor_frames', [0])))} anchors",
        f"path length         : {path_len:,.0f} px",
        f"mean speed          : {step.mean():.3f} px/frame",
        f"median speed        : {float(np.median(step)):.3f} px/frame",
        f"total yaw change    : {total_yaw:+.2f} deg",
        f"net scale change    : x{net_scale:.4f} ({(net_scale - 1) * 100:+.2f}%)",
        f"unreliable frames   : {pct_unrel:.2f}%",
        f"fallback frames     : {pct_fallback:.2f}%",
    ]
    if stats:
        lines += [
            f"start position      : ({stats['start'][0]:+.1f}, {stats['start'][1]:+.1f}) px",
            f"end position        : ({stats['end'][0]:+.1f}, {stats['end'][1]:+.1f}) px",
            f"net translation     : {stats['net_translation_px']:.1f} px",
            f"max translation     : {stats['max_translation_px']:.1f} px",
        ]
    return "\n".join(lines)
