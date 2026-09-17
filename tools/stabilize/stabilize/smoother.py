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
    robust_polyfit,
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

    A boundary is flagged only when its per-frame increment exceeds both the
    surrounding local motion and a sensible absolute floor.  This avoids
    false positives on footage whose camera simply moves faster near an
    anchor (the anchor-relative construction is positionally continuous by
    design).
    """
    anchor_frames = np.asarray(anchor_frames, dtype=np.int64)
    T = len(cumulative)
    boundaries = [int(b) for b in anchor_frames if 0 < int(b) < T]
    if not boundaries:
        return dict(boundaries=[], max_translation_jump_px=0.0,
                    max_rotation_jump_deg=0.0, max_scale_jump=0.0,
                    suspicious=False)

    trans, yaw, scale = _increments(cumulative)
    half = 8
    abs_t, abs_y, abs_s = 8.0, 1.0, 0.03

    max_tj = max_yj = max_sj = 0.0
    suspicious = False
    for b in boundaries:
        lo, hi = max(1, b - half), min(T, b + half + 1)
        local = [i for i in range(lo, hi) if i != b]
        if not local:
            continue
        local_t = float(np.max(trans[local]))
        local_y = float(np.max(np.abs(yaw[local])))
        local_s = float(np.max(np.abs(scale[local] - 1.0)))

        dj_t = max(0.0, trans[b] - max(abs_t, 2.0 * local_t))
        dj_y = max(0.0, abs(yaw[b]) - max(abs_y, 2.0 * local_y))
        dj_s = max(0.0, abs(scale[b] - 1.0) - max(abs_s, 2.0 * local_s))

        max_tj = max(max_tj, dj_t)
        max_yj = max(max_yj, dj_y)
        max_sj = max(max_sj, dj_s)
        if dj_t > 0 or dj_y > 0 or dj_s > 0:
            suspicious = True

    result = dict(
        boundaries=boundaries,
        max_translation_jump_px=float(max_tj),
        max_rotation_jump_deg=float(max_yj),
        max_scale_jump=float(max_sj),
        suspicious=bool(suspicious),
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
    locked_poly_degree: int = 3,
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
    locked_poly_degree : degree of the robust polynomial used to model the
        long-term drift in ``locked`` mode.  1 = linear (classic drift),
        3 (default) also removes gentle curved drift without oscillating.

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
    t_norm = t_axis / max(T - 1, 1)

    tx, ty, yaw_deg, log_scale = _decompose_cumulative(raw_cum)

    # ---- Optional locked-camera long-term translation constraint ----
    drift = None
    if mode == "locked":
        rel = np.where(reliable)[0]
        if len(rel) < 2:
            rel = np.arange(T)
        degree = max(0, int(locked_poly_degree))
        if degree == 1:
            slope_x, intercept_x = robust_linear_fit(t_axis[rel], tx[rel])
            slope_y, intercept_y = robust_linear_fit(t_axis[rel], ty[rel])
            trend_x = slope_x * t_axis + intercept_x
            trend_y = slope_y * t_axis + intercept_y
        else:
            c_x = robust_polyfit(t_norm[rel], tx[rel], degree=degree)
            c_y = robust_polyfit(t_norm[rel], ty[rel], degree=degree)
            trend_x = np.polyval(c_x, t_norm)
            trend_y = np.polyval(c_y, t_norm)
        # Remove only the *change* (trend - trend[0]); keep the intercept so
        # the start position -- and therefore the framing -- is unchanged.
        tx = tx - (trend_x - trend_x[0])
        ty = ty - (trend_y - trend_y[0])
        drift = dict(
            degree=degree,
            slope_x=float((trend_x[-1] - trend_x[0]) / max(T - 1, 1)),
            slope_y=float((trend_y[-1] - trend_y[0]) / max(T - 1, 1)),
            intercept_x=float(trend_x[0]), intercept_y=float(trend_y[0]),
            removed_x=float(trend_x[-1] - trend_x[0]),
            removed_y=float(trend_y[-1] - trend_y[0]),
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
