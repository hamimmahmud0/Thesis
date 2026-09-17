"""Camera-path smoothing for video stabilisation.

Decomposes the measured global camera trajectory into
(translation, yaw, log-scale), optionally removes a robust long-term
translation trend (``locked`` mode), applies Gaussian smoothing with
short-gap interpolation, and reconstructs the smoothed reference
transforms.

Modes
-----
``natural``
    Preserve legitimate low-frequency pans.  Only high-frequency jitter is
    removed; the trajectory is **not** detrended.

``locked``
    For tripod / static-camera footage.  A robust (Theil-Sen) linear fit to
    ``tx``/``ty`` is used to estimate the very-low-frequency drift, and the
    ramp is subtracted so the net start->end translation collapses to zero.
    Only the ramp is removed -- the intercept (start position) is kept, so
    the output framing does not jump.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import (
    compose,
    decompose_similarity,
    eye3,
    invert_affine,
    robust_linear_fit,
    similarity_matrix,
    smooth_decomposed_params,
)

VALID_MODES = ("natural", "locked")


def _decompose_cumulative(cum: np.ndarray) -> tuple[np.ndarray, ...]:
    """Decompose (T,3,3) cumulative transforms into (tx, ty, yaw, log_scale)."""
    T = cum.shape[0]
    tx = np.zeros(T)
    ty = np.zeros(T)
    yaw_deg = np.zeros(T)
    log_scale = np.zeros(T)
    for t in range(T):
        tx_t, ty_t, yaw_t, scale_t = decompose_similarity(cum[t])
        tx[t] = tx_t
        ty[t] = ty_t
        yaw_deg[t] = yaw_t
        log_scale[t] = np.log(max(scale_t, 1e-9))
    return tx, ty, yaw_deg, log_scale


def _reconstruct(stx, sty, syaw, slogs) -> np.ndarray:
    T = len(stx)
    out = np.tile(eye3(), (T, 1, 1))
    for t in range(T):
        out[t] = similarity_matrix(stx[t], sty[t], syaw[t], float(np.exp(slogs[t])))
    return out


def _increments(cum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame relative motion of a trajectory: (translation, yaw, scale).

    ``inc[t] = inv(cum[t-1]) @ cum[t]`` is the frame t-1 -> t transform.
    """
    T = cum.shape[0]
    trans = np.zeros(T)
    yaw = np.zeros(T)
    scale = np.ones(T)
    for t in range(1, T):
        inc = compose(invert_affine(cum[t - 1]), cum[t])
        inc_tx, inc_ty, yaw_t, scale_t = decompose_similarity(inc)
        trans[t] = float(np.hypot(inc_tx, inc_ty))
        yaw[t] = yaw_t
        scale[t] = scale_t
    return trans, yaw, scale


def anchor_transition_report(
    cumulative: np.ndarray, anchor_frames: np.ndarray, info: dict | None = None
) -> dict:
    """Check for translation / rotation / scale jumps at anchor boundaries.

    A "jump" is a boundary frame whose per-frame increment is far above the
    local median.  Returns the worst offenders plus a boolean flag.
    """
    anchor_frames = np.asarray(anchor_frames, dtype=np.int64)
    boundaries = [int(b) for b in anchor_frames if 0 < int(b) < len(cumulative)]
    if not boundaries:
        return dict(boundaries=[], max_translation_jump_px=0.0,
                    max_rotation_jump_deg=0.0, max_scale_jump=0.0,
                    suspicious=False)

    trans, yaw, scale = _increments(cumulative)
    med_t = float(np.median(trans[1:])) if len(trans) > 1 else 0.0
    med_y = float(np.median(np.abs(yaw[1:]))) if len(yaw) > 1 else 0.0
    med_s = float(np.median(np.abs(scale[1:] - 1.0))) if len(scale) > 1 else 0.0

    max_tj = max((abs(trans[b] - med_t) for b in boundaries), default=0.0)
    max_yj = max((abs(yaw[b]) - med_y for b in boundaries), default=0.0)
    max_sj = max((abs(scale[b] - 1.0) - med_s for b in boundaries), default=0.0)

    result = dict(
        boundaries=boundaries,
        max_translation_jump_px=float(max_tj),
        max_rotation_jump_deg=float(max(0.0, max_yj)),
        max_scale_jump=float(max(0.0, max_sj)),
        suspicious=bool(
            max_tj > max(2.0, 5 * med_t)
            or max_yj > max(0.5, 5 * med_y)
            or max_sj > max(0.01, 5 * med_s)
        ),
    )
    if info is not None:
        info.setdefault("transitions", result)
    return result


def smooth_motion(
    motion_npz: str | Path,
    output_npz: str | Path,
    sigma: float = 10.0,
    interp_gap: int = 5,
    mode: str = "natural",
) -> dict:
    """Smooth a camera path and save the result.

    Parameters
    ----------
    motion_npz : path to ``motion.npz`` (output of ``stabilize estimate``).
    output_npz : path to write ``motion_smooth.npz``.
    sigma : Gaussian smoothing sigma in frames (5-30 typical).
    interp_gap : unreliable gaps shorter than this are linearly interpolated
        before smoothing; longer gaps hold the last reliable value.
    mode : ``"natural"`` (preserve slow pans) or ``"locked"`` (remove
        robust long-term translation drift).

    Returns
    -------
    dict with the smoothed reference transforms and diagnostics.
    """
    from .motion import load_motion, stabilization_stats

    mode = str(mode).lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    motion = load_motion(motion_npz)
    raw_cum = motion["cumulative"]
    reliable = motion["reliable"]
    T = len(reliable)
    t_axis = np.arange(T, dtype=np.float64)

    tx, ty, yaw_deg, log_scale = _decompose_cumulative(raw_cum)

    # ---- Optional locked-camera long-term translation constraint ----
    drift = None
    if mode == "locked":
        rel = np.where(reliable)[0]
        if len(rel) >= 2:
            slope_x, intercept_x = robust_linear_fit(t_axis[rel], tx[rel])
            slope_y, intercept_y = robust_linear_fit(t_axis[rel], ty[rel])
        else:
            slope_x = slope_y = 0.0
            intercept_x = float(tx[0]) if T else 0.0
            intercept_y = float(ty[0]) if T else 0.0
        # Subtract only the ramp (slope * t); keep the intercept so the
        # start position -- and therefore the framing -- is unchanged.
        tx = tx - slope_x * t_axis
        ty = ty - slope_y * t_axis
        drift = dict(
            slope_x=float(slope_x), slope_y=float(slope_y),
            intercept_x=float(intercept_x), intercept_y=float(intercept_y),
        )

    # ---- Smooth ----
    stx, sty, syaw, slogs = smooth_decomposed_params(
        tx, ty, yaw_deg, log_scale, reliable,
        sigma=sigma, interp_range=interp_gap,
    )

    ref = _reconstruct(stx, sty, syaw, slogs)

    # ---- Per-frame correction warp: W[t] = raw[t] @ inv(ref[t]) ----
    correction = np.tile(eye3(), (T, 1, 1))
    for t in range(T):
        correction[t] = compose(raw_cum[t], invert_affine(ref[t]))

    # ---- Diagnostics ----
    anchor_frames = motion.get("anchor_frames", np.array([0], dtype=np.int64))
    transitions = anchor_transition_report(raw_cum, anchor_frames)
    stats_raw = stabilization_stats(raw_cum)
    stats_ref = stabilization_stats(ref)

    out_path = Path(output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        smoothed_cumulative=ref,
        raw_cumulative=raw_cum,
        correction=correction,
        reliable=reliable,
        tx=stx,
        ty=sty,
        yaw_deg=syaw,
        log_scale=slogs,
        sigma=sigma,
        interp_gap=interp_gap,
        mode=mode,
        drift=json.dumps(drift),
        anchor_frames=np.asarray(anchor_frames, dtype=np.int64),
    )
    print(f"Wrote smoothed path (mode={mode}, sigma={sigma}) -> {out_path}")

    return dict(
        smoothed_cumulative=ref,
        raw_cumulative=raw_cum,
        correction=correction,
        reliable=reliable,
        tx=stx,
        ty=sty,
        yaw_deg=syaw,
        log_scale=slogs,
        mode=mode,
        drift=drift,
        transitions=transitions,
        stats_raw=stats_raw,
        stats_ref=stats_ref,
    )


def load_smoothed(npz_path: str | Path) -> dict:
    """Load a smoothed-motion .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    result = dict(
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
    if "correction" in d.files:
        result["correction"] = d["correction"]
    result["mode"] = str(d["mode"]) if "mode" in d.files else "natural"
    if "anchor_frames" in d.files:
        result["anchor_frames"] = d["anchor_frames"].astype(np.int64)
    if "drift" in d.files:
        try:
            raw = str(d["drift"])
            result["drift"] = json.loads(raw) if raw else None
        except Exception:
            result["drift"] = None
    return result
