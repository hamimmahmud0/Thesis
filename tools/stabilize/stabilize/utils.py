"""Shared video I/O, matrix helpers, and track loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def open_video(path: str | Path) -> tuple[cv2.VideoCapture, int, int, int, float]:
    """Open a video and return (cap, frame_count, width, height, fps).

    Raises ``SystemExit`` with a human-readable message on failure.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976

    if n_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise SystemExit(
            f"Video probe failed ({n_frames} frames, {width}x{height}). "
            "Re-encode or check the file."
        )
    return cap, n_frames, width, height, fps


def read_all_frames(
    path: str | Path,
    max_dim: int = 0,
    convert_rgb: bool = True,
) -> tuple[np.ndarray, int, int, float]:
    """Read every frame from a video.

    Parameters
    ----------
    path : path to video file.
    max_dim : if > 0, resize longest side to this value.
    convert_rgb : if True, convert BGR (OpenCV default) to RGB.

    Returns
    -------
    frames : ndarray (T, H, W, 3) uint8.
    width, height : original video dimensions.
    fps : video frame rate.
    """
    cap, n_frames, orig_w, orig_h, fps = open_video(path)
    scale = min(1.0, max_dim / max(orig_w, orig_h)) if max_dim > 0 else 1.0
    tw = max(1, int(round(orig_w * scale)) // 2 * 2)
    th = max(1, int(round(orig_h * scale)) // 2 * 2)

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            if convert_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise SystemExit(f"No frames readable from: {path}")

    return np.stack(frames), orig_w, orig_h, fps


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def eye3() -> np.ndarray:
    """Return a 3x3 identity matrix (float64)."""
    return np.eye(3, dtype=np.float64)


def similarity_matrix(
    tx: float, ty: float, yaw_deg: float, scale: float
) -> np.ndarray:
    """Build a 3x3 homogeneous similarity matrix.

    This maps source points to destination points via translation (tx, ty),
    rotation by *yaw_deg* degrees, and uniform *scale*.
    """
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [scale * c, -scale * s, tx],
        [scale * s,  scale * c, ty],
        [0.0,        0.0,       1.0],
    ], dtype=np.float64)


def decompose_similarity(M: np.ndarray) -> tuple[float, float, float, float]:
    """Decompose a 2x3 or 3x3 similarity matrix into (tx, ty, yaw_deg, scale).

    For a matrix built by :func:`similarity_matrix`, this recovers the
    original parameters exactly (up to floating-point precision).
    """
    a, b = float(M[0, 0]), float(M[0, 1])
    tx, ty = float(M[0, 2]), float(M[1, 2])
    scale = float(np.hypot(a, b))
    # The matrix stores [-scale*sin(th)] in position [0,1], so we need
    # atan2(-b, a) to recover the original angle.
    yaw_deg = float(np.degrees(np.arctan2(-b, a)))
    return tx, ty, yaw_deg, scale


# ---------------------------------------------------------------------------
# Homogeneous affine matrix helpers
# ---------------------------------------------------------------------------
#
# CONVENTION (used everywhere in this package):
#
#   A transform is a 3x3 homogeneous matrix ``M`` acting on column vectors
#   ``p = [x, y, 1]^T`` so that ``p_dst = M @ p_src``.
#
#   ``M`` maps coordinates in the SOURCE frame to coordinates in the
#   DESTINATION frame.  A matrix written ``M_{a->b}`` therefore maps frame
#   ``a`` coordinates into frame ``b`` coordinates.
#
#   ``compose(A, B) == A @ B`` applies ``B`` first, then ``A``.  This is the
#   standard matrix-product order: ``compose(M_{a->b}, M_{b->c})`` yields
#   ``M_{a->c}``.


def affine_2x3_to_3x3(M: np.ndarray) -> np.ndarray:
    """Embed a 2x3 affine matrix into a 3x3 homogeneous matrix.

    ``M`` maps source -> destination.  The returned 3x3 matrix performs the
    same mapping and leaves the homogeneous coordinate unchanged.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return M.copy()
    if M.shape != (2, 3):
        raise ValueError(f"expected 2x3 or 3x3 affine matrix, got {M.shape}")
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = M
    return out


def affine_3x3_to_2x3(M: np.ndarray) -> np.ndarray:
    """Strip a 3x3 homogeneous affine matrix down to its 2x3 block."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (2, 3):
        return M.copy()
    if M.shape != (3, 3):
        raise ValueError(f"expected 2x3 or 3x3 affine matrix, got {M.shape}")
    return M[:2, :].copy()


def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compose two affine transforms: ``compose(A, B) == A @ B``.

    ``B`` is applied first, then ``A``.  Accepts 2x3 or 3x3 matrices and
    always returns a 3x3 homogeneous matrix.  With the package convention,
    ``compose(M_{a->b}, M_{b->c}) == M_{a->c}``.
    """
    return affine_2x3_to_3x3(A) @ affine_2x3_to_3x3(B)


def invert_affine(M: np.ndarray) -> np.ndarray:
    """Return the exact inverse of an affine transform as a 3x3 matrix.

    For ``M_{a->b}`` the result is ``M_{b->a}``.
    """
    return np.linalg.inv(affine_2x3_to_3x3(M))


def similarity_matrix_3x3(
    tx: float, ty: float, yaw_deg: float, scale: float
) -> np.ndarray:
    """3x3 alias of :func:`similarity_matrix` (explicit homogeneous form)."""
    return similarity_matrix(tx, ty, yaw_deg, scale)


def blend_similarity(A: np.ndarray, B: np.ndarray, w: float) -> np.ndarray:
    """Linearly blend two similarity transforms in parameter space.

    ``w == 0`` returns ``A``; ``w == 1`` returns ``B``.  Blending is done on
    ``(tx, ty, yaw_deg, log_scale)`` so the result stays a valid similarity
    transform (a naive matrix average would not preserve that).
    """
    ta = decompose_similarity(A)
    tb = decompose_similarity(B)
    w = float(min(max(w, 0.0), 1.0))

    # Interpolate the yaw along the shortest angular path.
    d_yaw = ((tb[2] - ta[2] + 180.0) % 360.0) - 180.0
    yaw = ta[2] + w * d_yaw

    log_scale = (1.0 - w) * np.log(max(ta[3], 1e-9)) + w * np.log(max(tb[3], 1e-9))
    return similarity_matrix(
        (1.0 - w) * ta[0] + w * tb[0],
        (1.0 - w) * ta[1] + w * tb[1],
        yaw,
        float(np.exp(log_scale)),
    )


def robust_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 2000,
    random_state: int = 0,
) -> tuple[float, float]:
    """Robust straight-line fit ``y = slope * x + intercept``.

    Uses the Theil-Sen estimator (median of pairwise slopes) on an evenly
    spaced subsample, which tolerates up to ~29% outliers without being
    dragged like ordinary least squares.  Falls back to least squares if the
    robust estimate is degenerate.

    Returns ``(slope, intercept)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return 0.0, 0.0
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0

    if n > max_samples:
        sel = np.linspace(0, n - 1, max_samples).astype(np.int64)
        xs, ys = x[sel], y[sel]
    else:
        xs, ys = x, y

    # Pairwise slopes (Theil-Sen).  O(m^2) with m <= max_samples.
    dx = xs[None, :] - xs[:, None]
    dy = ys[None, :] - ys[:, None]
    iu = np.triu_indices(xs.size, k=1)
    denom = dx[iu]
    num = dy[iu]
    ok = np.abs(denom) > 1e-12
    if ok.any():
        slope = float(np.median(num[ok] / denom[ok]))
        intercept = float(np.median(ys - slope * xs))
        if np.isfinite(slope) and np.isfinite(intercept):
            return slope, intercept

    # Fallback: ordinary least squares.
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def robust_polyfit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    iters: int = 20,
    max_samples: int = 4000,
) -> np.ndarray:
    """Robust low-order polynomial fit via iteratively reweighted least squares.

    Uses Tukey bisquare weights so that a minority of outlier frames cannot
    drag the estimated long-term trend.  Suitable for extracting the
    very-low-frequency drift of ``tx``/``ty`` in locked-camera mode.

    Returns the coefficient vector in ``np.polyval`` order (highest first).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    degree = max(0, int(degree))
    if n <= degree + 1:
        return np.zeros(degree + 1)

    if n > max_samples:
        sel = np.linspace(0, n - 1, max_samples).astype(np.int64)
        xs, ys = x[sel], y[sel]
    else:
        xs, ys = x, y

    try:
        c = np.polyfit(xs, ys, degree)
        for _ in range(max(0, int(iters))):
            r = ys - np.polyval(c, xs)
            s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9
            u = r / (4.685 * s)
            w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
            if w.sum() <= degree + 1:
                break
            c = np.polyfit(xs, ys, degree, w=np.sqrt(w))
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros(degree + 1)

    if not np.isfinite(c).all():
        return np.zeros(degree + 1)
    return np.asarray(c, dtype=np.float64)


def smooth_decomposed_params(
    tx: np.ndarray,
    ty: np.ndarray,
    yaw_deg: np.ndarray,
    log_scale: np.ndarray,
    reliable: np.ndarray,
    sigma: float,
    interp_range: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth decomposed similarity parameters with Gaussian filter.

    Short unreliable gaps (<= interp_range frames) are linearly interpolated
    before smoothing.  Longer gaps keep the last reliable value (hold).

    Returns smoothed (tx, ty, yaw_deg, log_scale) arrays.
    """
    from scipy.ndimage import gaussian_filter1d

    def _interp_hold(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Interpolate short gaps, hold over long gaps."""
        out = arr.copy()
        reliable_idx = np.where(mask)[0]
        if len(reliable_idx) < 2:
            return out
        for i in range(len(reliable_idx) - 1):
            lo, hi = reliable_idx[i], reliable_idx[i + 1]
            gap = hi - lo
            if gap <= 1:
                continue
            if gap <= interp_range:
                # linear interpolation
                frac = np.linspace(0, 1, gap + 1)[1:-1]
                out[lo + 1 : hi] = arr[lo] * (1 - frac) + arr[hi] * frac
            else:
                # hold last reliable value
                out[lo + 1 : hi] = arr[lo]
        return out

    stx = _interp_hold(tx, reliable)
    sty = _interp_hold(ty, reliable)
    syaw = _interp_hold(yaw_deg, reliable)
    slogs = _interp_hold(log_scale, reliable)

    stx = gaussian_filter1d(stx, sigma=sigma)
    sty = gaussian_filter1d(sty, sigma=sigma)
    syaw = gaussian_filter1d(syaw, sigma=sigma)
    slogs = gaussian_filter1d(slogs, sigma=sigma)

    return stx, sty, syaw, slogs


def in_bounds(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """Boolean mask: True where points are inside the image."""
    return (
        (pts[:, 0] >= 0) & (pts[:, 0] < w)
        & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )


def runs_of(mask: np.ndarray):
    """Yield (start, end) index runs where *mask* is True."""
    if not mask.any():
        return
    idx = np.flatnonzero(np.diff(np.concatenate(([False], mask.astype(int)))))
    for k in range(0, len(idx), 2):
        yield int(idx[k]), int(idx[k + 1])


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    """Drop a leading batch dimension of 1."""
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def load_tracks(npz_path: str | Path) -> dict:
    """Load a CoTracker3 .npz file and return a dict with standardised keys.

    Returns
    -------
    dict with keys:
        tracks       (T, N, 2) float32 — keypoint positions in original pixels
        visibility   (T, N)    bool    — per-point visibility
        width        int or None
        height       int or None
        fps          float or None
        query_points (N, 2) or None
    """
    d = np.load(npz_path, allow_pickle=True)
    tracks = squeeze_batch(np.asarray(d["tracks"], dtype=np.float32))
    visibility = squeeze_batch(np.asarray(d["visibility"], dtype=bool))

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be (T, N, 2), got {tracks.shape}")

    width = height = None
    fps = None
    query_points = None

    if "meta" in d.files:
        try:
            meta = json.loads(str(d["meta"]))
            width = meta.get("width")
            height = meta.get("height")
            fps = meta.get("fps")
        except Exception:
            pass

    if "query_points" in d.files:
        query_points = np.asarray(d["query_points"], dtype=np.float32)

    return dict(
        tracks=tracks,
        visibility=visibility,
        width=width,
        height=height,
        fps=fps,
        query_points=query_points,
    )


def human_time(sec: float) -> str:
    """Format seconds as M:SS or HhMMm."""
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}:{s:02d}"
    return f"{m // 60}h{m % 60:02d}m"
