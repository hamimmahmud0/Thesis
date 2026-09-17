"""Fresh-grid + bridge-point anchor architecture.

Motivation
----------
A single CoTracker query grid sampled on frame 0 degrades over long videos:
points are occluded, leave the frame, drift with accumulated tracking error,
sit on moving foreground, or get rejected by RANSAC.  The surviving population
shrinks and clusters.

This module segments the video into short tracking segments.  **Every anchor
gets a brand-new query grid**, which is the primary tracking population for
that segment.  The previous anchor's grid is retained only long enough to
*bridge* the new anchor into the existing global coordinate system through
the overlap region.

Transform conventions (identical to :mod:`stabilize.motion`)
------------------------------------------------------------
- A transform is a 3x3 homogeneous matrix acting on column vectors,
  ``p_dst = M @ p_src``.
- ``M_{a->b}`` maps coordinates in frame ``a`` into frame ``b``.
- ``compose(A, B) == A @ B`` applies ``B`` first, then ``A``.
- ``local[t]  = M_{t -> K}``   (current frame -> its segment anchor)
- ``G[K]      = M_{K -> 0}``   (anchor -> global / frame 0)
- ``bridge    = M_{K_new -> K_old}``
- ``G[K_new] = compose(G[K_old], bridge)``
- ``frame_to_global[t] = compose(G[K], local[t])``
- ``cumulative[t] = inv(frame_to_global[t])``  (frame 0 -> t, renderer)

Overlap region
--------------
When anchor ``K_old`` transitions to ``K_new``, the old segment was tracked
through ``K_new + overlap`` frames and the new segment starts at ``K_new``.
In the overlap window ``[K_new, K_new + overlap)`` both grids have tracks, so
the bridge can be estimated from **multiple frames** and the stabilization
transforms can be cross-faded (smoothstep) instead of snapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .utils import (
    blend_similarity,
    compose,
    decompose_similarity,
    eye3,
    in_bounds,
    invert_affine,
    robust_median_similarity,
    smoothstep,
    spatial_coverage,
    transform_residual,
)

# ---------------------------------------------------------------------------
# Anchor reasons
# ---------------------------------------------------------------------------


class AnchorReason(str, Enum):
    """Why a new tracking anchor was created."""

    INITIAL = "initial"
    MAX_INTERVAL = "max_interval"
    LOW_POINT_COUNT = "low_point_count"
    LOW_INLIER_RATIO = "low_inlier_ratio"
    LOW_SPATIAL_COVERAGE = "low_spatial_coverage"
    TRANSFORM_FAILURE = "transform_failure"
    VIDEO_END = "video_end"


REASON_NAMES = [r.value for r in AnchorReason]
REASON_CODES = {r.value: i for i, r in enumerate(AnchorReason)}
_NO_REASON = -1


# ---------------------------------------------------------------------------
# Configuration / data containers
# ---------------------------------------------------------------------------


@dataclass
class SegmentConfig:
    """All tunables for fresh-grid anchor planning (validated on use)."""

    anchor_interval_seconds: float = 10.0
    min_anchor_interval_seconds: float = 2.0
    anchor_overlap_seconds: float = 1.5

    min_remaining_point_ratio: float = 0.40
    min_inlier_ratio: float = 0.40
    min_inlier_count: int = 20
    min_spatial_coverage: float = 0.35
    quality_failure_patience_frames: int = 5

    coverage_grid_rows: int = 4
    coverage_grid_cols: int = 4

    # RANSAC / estimation.
    ransac_reproj_threshold: float = 2.0
    max_iters: int = 5000
    confidence: float = 0.999
    min_correspondences: int = 20
    max_points: int = 20000
    # If 0 the overlap length is used for cross-fading.
    blend_frames: int = 0

    def validate(self) -> None:
        if self.anchor_interval_seconds <= 0:
            raise ValueError("anchor_interval_seconds must be > 0")
        if self.min_anchor_interval_seconds <= 0:
            raise ValueError("min_anchor_interval_seconds must be > 0")
        if self.min_anchor_interval_seconds > self.anchor_interval_seconds:
            raise ValueError(
                "min_anchor_interval_seconds must be <= anchor_interval_seconds"
            )
        if self.anchor_overlap_seconds < 0:
            raise ValueError("anchor_overlap_seconds must be >= 0")
        for name in (
            "min_remaining_point_ratio",
            "min_inlier_ratio",
            "min_spatial_coverage",
        ):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.min_inlier_count < 3:
            raise ValueError("min_inlier_count must be >= 3")
        for name in ("coverage_grid_rows", "coverage_grid_cols"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.quality_failure_patience_frames < 1:
            raise ValueError("quality_failure_patience_frames must be >= 1")


@dataclass
class TrackingSegment:
    """One CoTracker inference run with a fresh query grid at its anchor."""

    segment_id: int
    anchor_frame: int
    start_frame: int
    end_frame: int  # exclusive
    query_points: np.ndarray  # (N, 2) original-video pixels at anchor_frame
    tracks: np.ndarray  # (L, N, 2) for frames [start_frame, end_frame)
    visibility: np.ndarray  # (L, N) bool
    width: int
    height: int
    fps: float
    reason: str = AnchorReason.INITIAL.value

    # Filled by :func:`analyze_segment`.
    local: np.ndarray | None = None  # (L,3,3) M_{t -> K}
    transform_valid: np.ndarray | None = None
    fallback_used: np.ndarray | None = None
    num_visible: np.ndarray | None = None
    num_candidates: np.ndarray | None = None
    num_inliers: np.ndarray | None = None
    inlier_ratio: np.ndarray | None = None
    spatial_coverage: np.ndarray | None = None
    remaining_point_ratio: np.ndarray | None = None
    residual: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.tracks.shape[0])

    @property
    def num_query_points(self) -> int:
        return int(self.tracks.shape[1])

    def abs_to_local(self, t: int) -> int:
        return int(t) - int(self.start_frame)


def make_segment(
    segment_id: int,
    anchor_frame: int,
    start_frame: int,
    end_frame: int,
    tracks: np.ndarray,
    visibility: np.ndarray,
    query_points: np.ndarray,
    width: int,
    height: int,
    fps: float,
    reason: str = AnchorReason.INITIAL.value,
) -> TrackingSegment:
    return TrackingSegment(
        segment_id=int(segment_id),
        anchor_frame=int(anchor_frame),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        query_points=np.asarray(query_points, dtype=np.float64),
        tracks=np.asarray(tracks, dtype=np.float64),
        visibility=np.asarray(visibility, dtype=bool),
        width=int(width),
        height=int(height),
        fps=float(fps),
        reason=str(reason),
    )


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------


def _estimation_kwargs(cfg: SegmentConfig) -> dict:
    return dict(
        ransac_reproj_threshold=cfg.ransac_reproj_threshold,
        max_iters=cfg.max_iters,
        confidence=cfg.confidence,
        min_correspondences=cfg.min_correspondences,
        min_inlier_ratio=cfg.min_inlier_ratio,
        max_points=cfg.max_points,
    )


def analyze_segment(seg: TrackingSegment, cfg: SegmentConfig) -> TrackingSegment:
    """Estimate ``frame t -> anchor K`` transforms and per-frame quality.

    The segment owns its fresh grid, so this only ever fits against its own
    anchor; nothing is chained frame-to-frame.
    """
    from .motion import estimate_affine

    L = seg.length
    N = seg.num_query_points
    a = seg.abs_to_local(seg.anchor_frame)
    if a < 0 or a >= L:
        raise ValueError(
            f"anchor {seg.anchor_frame} not inside segment "
            f"[{seg.start_frame}, {seg.end_frame})"
        )

    local = np.tile(eye3(), (L, 1, 1))
    transform_valid = np.zeros(L, dtype=bool)
    fallback_used = np.zeros(L, dtype=bool)
    num_visible = np.zeros(L, dtype=np.int64)
    num_candidates = np.zeros(L, dtype=np.int64)
    num_inliers = np.zeros(L, dtype=np.int64)
    inlier_ratio = np.zeros(L)
    coverage = np.zeros(L)
    remaining = np.zeros(L)
    residual = np.full(L, np.inf)

    kwargs = _estimation_kwargs(cfg)
    p_anchor = seg.tracks[a]
    vis_anchor = seg.visibility[a]

    prev = eye3()
    for i in range(L):
        p = seg.tracks[i]
        mask = vis_anchor & seg.visibility[i]
        mask &= np.isfinite(p).all(axis=1) & np.isfinite(p_anchor).all(axis=1)
        if seg.width and seg.height:
            mask &= in_bounds(p, seg.width, seg.height) & in_bounds(
                p_anchor, seg.width, seg.height
            )

        nv = int(mask.sum())
        num_visible[i] = nv
        remaining[i] = nv / max(N, 1)

        if i == a:
            local[i] = eye3()
            transform_valid[i] = True
            num_candidates[i] = nv
            num_inliers[i] = nv
            inlier_ratio[i] = 1.0 if nv else 0.0
            coverage[i] = spatial_coverage(
                p_anchor[mask], seg.width, seg.height,
                cfg.coverage_grid_rows, cfg.coverage_grid_cols,
            )
            residual[i] = 0.0
            continue

        M, info = estimate_affine(p[mask], p_anchor[mask], **kwargs)
        num_candidates[i] = info["num_candidates"]
        num_inliers[i] = info["num_inliers"]
        inlier_ratio[i] = info["inlier_ratio"]
        coverage[i] = spatial_coverage(
            p[mask], seg.width, seg.height,
            cfg.coverage_grid_rows, cfg.coverage_grid_cols,
        )

        if info["valid"] and M is not None:
            local[i] = M
            transform_valid[i] = True
            prev = M
            im = info.get("inlier_mask")
            if im is not None and len(im) == int(mask.sum()) and im.any():
                residual[i] = transform_residual(
                    M, p[mask][im], p_anchor[mask][im]
                )
        else:
            # Graceful fallback: hold the previous anchor-relative transform.
            local[i] = prev
            fallback_used[i] = True

    seg.local = local
    seg.transform_valid = transform_valid
    seg.fallback_used = fallback_used
    seg.num_visible = num_visible
    seg.num_candidates = num_candidates
    seg.num_inliers = num_inliers
    seg.inlier_ratio = inlier_ratio
    seg.spatial_coverage = coverage
    seg.remaining_point_ratio = remaining
    seg.residual = residual
    return seg


def frame_trigger_reason(
    seg: TrackingSegment, i: int, cfg: SegmentConfig
) -> AnchorReason | None:
    """Return the re-anchor reason for local frame ``i``, or None."""
    if i < 0 or i >= seg.length:
        return None
    # Point survival is checked first so total loss reports LOW_POINT_COUNT
    # rather than a generic transform failure.
    if seg.remaining_point_ratio[i] < cfg.min_remaining_point_ratio:
        return AnchorReason.LOW_POINT_COUNT
    if seg.num_candidates[i] < cfg.min_correspondences:
        return AnchorReason.TRANSFORM_FAILURE
    if not bool(seg.transform_valid[i]):
        return AnchorReason.TRANSFORM_FAILURE
    if seg.inlier_ratio[i] < cfg.min_inlier_ratio:
        return AnchorReason.LOW_INLIER_RATIO
    if seg.num_inliers[i] < cfg.min_inlier_count:
        return AnchorReason.LOW_POINT_COUNT
    if seg.spatial_coverage[i] < cfg.min_spatial_coverage:
        return AnchorReason.LOW_SPATIAL_COVERAGE
    return None


def choose_next_anchor(
    seg: TrackingSegment, cfg: SegmentConfig
) -> tuple[int, AnchorReason]:
    """Pick the next anchor frame, honoring min/max spacing and hysteresis.

    Returns ``(absolute_frame, reason)``.  The frame is the start of the
    sustained poor-quality run when quality-triggered, otherwise the
    configured maximum interval.
    """
    fps = seg.fps if seg.fps > 0 else 30.0
    k = int(seg.anchor_frame)
    min_gap = max(1, int(round(cfg.min_anchor_interval_seconds * fps)))
    max_gap = max(min_gap, int(round(cfg.anchor_interval_seconds * fps)))
    end = int(seg.start_frame + seg.length)

    limit = min(k + max_gap, end - 1)
    start = k + min_gap
    if start > limit:
        return k + max_gap, AnchorReason.MAX_INTERVAL

    patience = int(cfg.quality_failure_patience_frames)
    streak = 0
    streak_start = None
    streak_reason = None
    for t in range(start, limit + 1):
        i = seg.abs_to_local(t)
        reason = frame_trigger_reason(seg, i, cfg)
        if reason is not None:
            if streak == 0:
                streak_start = t
                streak_reason = reason
            streak += 1
            if streak >= patience:
                # Anchor on the last *good* frame before the collapse so the
                # old grid still has a valid transform for bridging.
                anchor_at = max(k + 1, int(streak_start) - 1)
                return int(anchor_at), streak_reason
        else:
            streak = 0
            streak_start = None
            streak_reason = None
    return k + max_gap, AnchorReason.MAX_INTERVAL


# ---------------------------------------------------------------------------
# Bridge registration
# ---------------------------------------------------------------------------


def estimate_bridge(
    seg_old: TrackingSegment,
    seg_new: TrackingSegment,
    cfg: SegmentConfig,
    overlap_frames: int | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Estimate ``M_{K_new -> K_old}`` from the overlap region.

    For every overlap frame ``t`` where both segments have a valid
    anchor-relative transform::

        bridge(t) = local_old[t] @ inv(local_new[t])
                  = M_{t -> K_old} @ M_{K_new -> t}
                  = M_{K_new -> K_old}

    The per-frame candidates are combined with a robust median in
    (translation, rotation, log-scale) space.

    Returns ``(bridge_3x3 | None, info)``.
    """
    fps = seg_old.fps if seg_old.fps > 0 else 30.0
    if overlap_frames is None:
        overlap_frames = max(1, int(round(cfg.anchor_overlap_seconds * fps)))

    k_new = int(seg_new.anchor_frame)
    lo = k_new
    hi = min(int(seg_old.end_frame), k_new + overlap_frames)
    candidates = []
    translations = []
    n_frames = 0
    for t in range(lo, hi):
        i_old = seg_old.abs_to_local(t)
        i_new = seg_new.abs_to_local(t)
        if not (0 <= i_old < seg_old.length and 0 <= i_new < seg_new.length):
            continue
        if not (seg_old.transform_valid[i_old] and seg_new.transform_valid[i_new]):
            continue
        cand = compose(seg_old.local[i_old], invert_affine(seg_new.local[i_new]))
        candidates.append(cand)
        translations.append(seg_new.local[i_new])
        n_frames += 1

    info = {
        "bridge": True,
        "degraded": False,
        "num_overlap_frames": n_frames,
        "num_candidates": len(candidates),
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "method": "multi_frame_median",
        "fallback": False,
    }

    if len(candidates) == 0:
        # Fallback: use the old segment's own prediction of the new anchor,
        # i.e. local_old[K_new] = M_{K_new -> K_old}.  Because analyzing a
        # segment always fills ``local`` (holding the previous transform when
        # a frame is unreliable), this preserves continuity at the transition
        # instead of introducing a jump.  Marked degraded for diagnostics.
        i_fb = seg_old.abs_to_local(k_new)
        if 0 <= i_fb < seg_old.length:
            fb = seg_old.local[i_fb]
            info.update(
                bridge=True, degraded=True,
                method="old_segment_prediction", fallback=True,
                num_inliers=0,
            )
            return fb, info
        info.update(bridge=False, degraded=True, method="hold", fallback=True)
        return None, info

    bridge = robust_median_similarity(candidates)
    if bridge is None or not np.isfinite(bridge).all():
        i_fb = seg_old.abs_to_local(k_new)
        if 0 <= i_fb < seg_old.length:
            info.update(
                bridge=True, degraded=True,
                method="old_segment_prediction", fallback=True,
            )
            return seg_old.local[i_fb], info
        info.update(bridge=False, degraded=True, method="hold", fallback=True)
        return None, info

    info["num_inliers"] = len(candidates)
    info["inlier_ratio"] = len(candidates) / max(n_frames, 1)
    return bridge, info


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def _fill_quality(
    out: dict,
    t: int,
    seg: TrackingSegment,
    i: int,
    anchor_valid: bool,
    reanchor_requested: bool = False,
    reanchor_reason: int = _NO_REASON,
) -> None:
    """Copy per-frame segment quality into the flat output arrays."""
    L = seg.length
    if not (0 <= i < L):
        return
    out["segment_id"][t] = seg.segment_id
    out["anchor_frame"][t] = seg.anchor_frame
    out["anchor_age_frames"][t] = t - seg.anchor_frame
    out["num_query_points"][t] = seg.num_query_points
    out["num_visible_points"][t] = seg.num_visible[i]
    out["num_ransac_candidates"][t] = seg.num_candidates[i]
    out["num_ransac_inliers"][t] = seg.num_inliers[i]
    out["inlier_ratio"][t] = seg.inlier_ratio[i]
    out["spatial_coverage"][t] = seg.spatial_coverage[i]
    out["remaining_point_ratio"][t] = seg.remaining_point_ratio[i]
    out["fallback_used"][t] = seg.fallback_used[i]
    out["transform_valid"][t] = bool(seg.transform_valid[i]) and anchor_valid
    out["reanchor_requested"][t] = bool(reanchor_requested)
    out["reanchor_reason"][t] = int(reanchor_reason)


def _base_cumulative(g: np.ndarray, local: np.ndarray) -> np.ndarray:
    """cumulative = inv(g @ local) = M_{0 -> t}."""
    return invert_affine(compose(g, local))


def plan_segments(
    track_fn,
    total_frames: int,
    width: int,
    height: int,
    fps: float,
    cfg: SegmentConfig,
    *,
    debug: bool = False,
) -> dict:
    """Run the fresh-grid + bridge planner over the whole video.

    ``track_fn(anchor_frame, end_frame) -> TrackingSegment`` must run CoTracker
    on ``[anchor_frame, end_frame)`` with a fresh grid queried at
    ``anchor_frame``.  It is injectable so the planner can be tested without a
    GPU.

    Returns a motion-style dict compatible with the smoother/renderer, plus
    segmented diagnostics.
    """
    cfg.validate()
    total_frames = int(total_frames)
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    fps = float(fps) if fps and fps > 0 else 30.0

    max_gap = max(1, int(round(cfg.anchor_interval_seconds * fps)))
    overlap = max(0, int(round(cfg.anchor_overlap_seconds * fps)))
    blend = int(cfg.blend_frames) if cfg.blend_frames > 0 else overlap

    # ---- Flat outputs ----
    out = {
        "cumulative": np.tile(eye3(), (total_frames, 1, 1)),
        "local": np.tile(eye3(), (total_frames, 1, 1)),
        "frame_to_global": np.tile(eye3(), (total_frames, 1, 1)),
        "transform_valid": np.zeros(total_frames, dtype=bool),
        "fallback_used": np.zeros(total_frames, dtype=bool),
        "segment_id": np.full(total_frames, -1, dtype=np.int64),
        "anchor_frame": np.full(total_frames, -1, dtype=np.int64),
        "anchor_age_frames": np.zeros(total_frames, dtype=np.int64),
        "num_query_points": np.zeros(total_frames, dtype=np.int64),
        "num_visible_points": np.zeros(total_frames, dtype=np.int64),
        "num_ransac_candidates": np.zeros(total_frames, dtype=np.int64),
        "num_ransac_inliers": np.zeros(total_frames, dtype=np.int64),
        "inlier_ratio": np.zeros(total_frames),
        "spatial_coverage": np.zeros(total_frames),
        "remaining_point_ratio": np.zeros(total_frames),
        "reanchor_requested": np.zeros(total_frames, dtype=bool),
        "reanchor_reason": np.full(total_frames, _NO_REASON, dtype=np.int64),
    }
    assigned = np.zeros(total_frames, dtype=bool)

    anchor_frames: list[int] = []
    anchor_globals: list[np.ndarray] = []
    anchor_valid: list[bool] = []
    anchor_reasons: list[str] = []
    anchor_stats: list[dict] = []
    segment_infos: list[dict] = []

    def _record_anchor(source_seg, trigger_frame, g, valid, reason, bridge_info):
        """Record anchor metadata.  ``source_seg`` is the *previous* segment
        whose deteriorating quality triggered this anchor (or the initial
        segment for anchor 0)."""
        i_trig = source_seg.abs_to_local(trigger_frame)
        i_trig = min(max(i_trig, 0), max(source_seg.length - 1, 0))
        anchor_frames.append(int(trigger_frame if reason != AnchorReason.INITIAL.value
                                 else source_seg.anchor_frame))
        anchor_globals.append(g.copy())
        anchor_valid.append(bool(valid))
        anchor_reasons.append(str(reason))
        prev = anchor_frames[-2] if len(anchor_frames) >= 2 else -1
        anchor_stats.append(dict(
            anchor_frame=int(anchor_frames[-1]),
            previous_anchor=int(prev),
            age_frames=int(anchor_frames[-1] - prev) if prev >= 0 else 0,
            reason=str(reason),
            surviving_tracks=int(source_seg.num_visible[i_trig]),
            num_query_points=int(source_seg.num_query_points),
            ransac_inliers=int(source_seg.num_inliers[i_trig]),
            inlier_ratio=float(source_seg.inlier_ratio[i_trig]),
            spatial_coverage=float(source_seg.spatial_coverage[i_trig]),
            bridge_inliers=int(bridge_info.get("num_inliers", 0)),
            bridge_degraded=bool(bridge_info.get("degraded", False)),
            bridge_method=str(bridge_info.get("method", "")),
            new_query_points=int(source_seg.num_query_points),
        ))

    # ---- First segment ----
    seg = track_fn(0, min(total_frames, max_gap + overlap))
    analyze_segment(seg, cfg)
    seg.segment_id = 0
    seg.reason = AnchorReason.INITIAL.value
    g_old = eye3()
    _record_anchor(seg, seg.anchor_frame, g_old, True,
                   AnchorReason.INITIAL.value, {})
    segment_infos.append(dict(segment_id=0, anchor=0,
                              start=seg.start_frame, end=seg.end_frame,
                              reason=AnchorReason.INITIAL.value))

    if debug:
        print(
            f"[fresh-grid] segment 0 anchor={seg.anchor_frame} "
            f"frames=[{seg.start_frame},{seg.end_frame}) points={seg.num_query_points}"
        )

    guard = 0
    while True:
        guard += 1
        if guard > total_frames:  # safety against pathological churn
            break

        k_new, reason = choose_next_anchor(seg, cfg)
        if k_new >= total_frames:
            # Finalize the remaining frames with the current segment.
            _finalize_remaining(out, assigned, seg, g_old,
                                anchor_valid[-1], total_frames)
            break
        if k_new <= seg.anchor_frame:
            k_new = seg.anchor_frame + max_gap

        end_new = min(total_frames, k_new + max_gap + overlap)
        new = track_fn(k_new, end_new)
        analyze_segment(new, cfg)
        new.segment_id = len(segment_infos)
        new.reason = reason.value
        segment_infos.append(dict(segment_id=new.segment_id, anchor=k_new,
                                  start=new.start_frame, end=new.end_frame,
                                  reason=reason.value))

        bridge, binfo = estimate_bridge(seg, new, cfg, overlap_frames=overlap)
        if bridge is not None:
            g_new = compose(g_old, bridge)
            valid = True
        else:
            # Last-resort continuity: align the new anchor to the old
            # segment's own prediction of it (local_old[K_new]).  This keeps
            # the frame-to-frame transform continuous rather than resetting
            # or jumping.  Geometric accuracy is sacrificed, not continuity.
            i_fb = seg.abs_to_local(k_new)
            if 0 <= i_fb < seg.length:
                g_new = compose(g_old, seg.local[i_fb])
            else:
                g_new = g_old.copy()
            valid = False

        if debug:
            print(
                f"[fresh-grid] anchor {k_new} reason={reason.value} "
                f"bridged={bridge is not None} bridge_inliers={binfo.get('num_inliers',0)} "
                f"degraded={binfo.get('degraded',False)}"
            )

        k_old = seg.anchor_frame

        # 1. Assign the old segment's owned frames [k_old, k_new).
        for t in range(k_old, k_new):
            if assigned[t]:
                continue
            i = seg.abs_to_local(t)
            if not (0 <= i < seg.length):
                continue
            out["cumulative"][t] = _base_cumulative(g_old, seg.local[i])
            out["local"][t] = seg.local[i]
            out["frame_to_global"][t] = compose(g_old, seg.local[i])
            _fill_quality(out, t, seg, i, anchor_valid[-1])
            assigned[t] = True

        # 2. Cross-fade the overlap [k_new, k_new + blend).
        for d in range(blend):
            t = k_new + d
            if t >= total_frames:
                break
            i_old = seg.abs_to_local(t)
            i_new = new.abs_to_local(t)
            can_old = 0 <= i_old < seg.length and bool(seg.transform_valid[i_old])
            can_new = 0 <= i_new < new.length and bool(new.transform_valid[i_new])
            if can_old and can_new:
                cum_old = _base_cumulative(g_old, seg.local[i_old])
                cum_new = _base_cumulative(g_new, new.local[i_new])
                alpha = smoothstep((d + 1) / (blend + 1.0))
                out["cumulative"][t] = blend_similarity(cum_old, cum_new, alpha)
            elif can_new:
                out["cumulative"][t] = _base_cumulative(g_new, new.local[i_new])
            elif can_old:
                out["cumulative"][t] = _base_cumulative(g_old, seg.local[i_old])
            else:
                continue
            out["local"][t] = new.local[i_new] if can_new else seg.local[i_old]
            out["frame_to_global"][t] = invert_affine(out["cumulative"][t])
            fill_seg, fill_i = (new, i_new) if can_new else (seg, i_old)
            _fill_quality(
                out, t, fill_seg, fill_i, valid,
                reanchor_requested=True,
                reanchor_reason=REASON_CODES[reason.value],
            )
            assigned[t] = True

        _record_anchor(seg, k_new, g_new, valid, reason.value, binfo)

        seg = new
        g_old = g_new

    # ---- Any frames still unassigned: extend the current segment. ----
    if not assigned.all():
        _finalize_remaining(out, assigned, seg, g_old, anchor_valid[-1],
                            total_frames)

    # ---- Assemble the motion result ----
    return _assemble_result(
        out, total_frames, width, height, fps, cfg,
        anchor_frames, anchor_globals, anchor_valid, anchor_reasons,
        anchor_stats, segment_infos, overlap,
    )


def _finalize_remaining(out, assigned, seg, g, anchor_ok, total_frames):
    for t in range(total_frames):
        if assigned[t]:
            continue
        i = seg.abs_to_local(t)
        if not (0 <= i < seg.length):
            # Outside the last segment (should not happen); hold the anchor.
            out["cumulative"][t] = _base_cumulative(g, eye3())
            out["local"][t] = eye3()
            out["frame_to_global"][t] = g
            assigned[t] = True
            continue
        out["cumulative"][t] = _base_cumulative(g, seg.local[i])
        out["local"][t] = seg.local[i]
        out["frame_to_global"][t] = compose(g, seg.local[i])
        _fill_quality(out, t, seg, i, anchor_ok)
        assigned[t] = True


def _assemble_result(
    out, T, width, height, fps, cfg,
    anchor_frames, anchor_globals, anchor_valid, anchor_reasons,
    anchor_stats, segment_infos, overlap,
):
    cumulative = out["cumulative"]
    pairwise = np.tile(eye3(), (T, 1, 1))
    for t in range(1, T):
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

    anchor_frames_arr = np.asarray(anchor_frames, dtype=np.int64)
    anchor_index = np.searchsorted(anchor_frames_arr, np.arange(T), side="right") - 1
    anchor_index = np.clip(anchor_index, 0, max(len(anchor_frames_arr) - 1, 0))
    anchor_valid_arr = np.asarray(anchor_valid, dtype=bool)
    reliable = out["transform_valid"] & anchor_valid_arr[anchor_index]

    # ---- Compute overhead statistics ----
    ideal = T
    processed = sum(int(s["end"] - s["start"]) for s in segment_infos)
    extra = max(0, processed - ideal)

    return dict(
        pairwise=pairwise,
        cumulative=cumulative,
        frame_to_global=out["frame_to_global"],
        local=out["local"],
        reliable=reliable,
        dx=dx,
        dy=dy,
        d_yaw_deg=d_yaw,
        scale=scale,
        inliers=out["num_ransac_inliers"],
        inlier_ratio=out["inlier_ratio"],
        usable=out["num_ransac_candidates"],
        num_candidates=out["num_ransac_candidates"],
        num_inliers=out["num_ransac_inliers"],
        transform_valid=out["transform_valid"],
        fallback_used=out["fallback_used"],
        # Segment diagnostics.
        segment_id=out["segment_id"],
        anchor_frame=out["anchor_frame"],
        anchor_age_frames=out["anchor_age_frames"],
        num_query_points=out["num_query_points"],
        num_visible_points=out["num_visible_points"],
        spatial_coverage=out["spatial_coverage"],
        remaining_point_ratio=out["remaining_point_ratio"],
        reanchor_requested=out["reanchor_requested"],
        reanchor_reason=out["reanchor_reason"],
        # Anchor bookkeeping (compatible with estimate_motion schema).
        anchor_frames=anchor_frames_arr,
        anchor_global=np.asarray(anchor_globals),
        anchor_valid=anchor_valid_arr,
        anchor_index=anchor_index,
        anchor_reasons=anchor_reasons,
        anchor_stats=anchor_stats,
        segments=segment_infos,
        segment_overhead=dict(
            num_segments=len(segment_infos),
            overlap_frames=overlap,
            frames_processed=processed,
            extra_frames=extra,
            overlap_compute_percent=100.0 * extra / max(ideal, 1),
        ),
        anchor_interval_seconds=float(cfg.anchor_interval_seconds),
        anchor_interval_frames=max(1, int(round(cfg.anchor_interval_seconds * fps))),
        method="fresh_grid_segments",
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def segment_report(result: dict) -> str:
    """Human-readable fresh-grid statistics for logs / summaries."""
    segs = result.get("segments", [])
    frames = result.get("anchor_frame", [])
    reasons: dict[str, int] = {}
    for r in result.get("anchor_reasons", []):
        reasons[r] = reasons.get(r, 0) + 1

    af = np.asarray(result.get("anchor_frames", []), dtype=np.int64)
    spacings = np.diff(af) if len(af) > 1 else np.array([])

    def _safe_mean(a):
        a = np.asarray(a, dtype=np.float64)
        return float(a.mean()) if a.size else 0.0

    lines = [
        f"segments            : {len(segs)}",
        f"anchors created     : {len(af)}",
        f"anchor spacing      : mean {_safe_mean(spacings):.1f} "
        f"min {int(spacings.min()) if spacings.size else 0} "
        f"max {int(spacings.max()) if spacings.size else 0} frames",
        f"reanchor reasons    : " + ", ".join(
            f"{k}={v}" for k, v in sorted(reasons.items())
        ),
        f"avg surviving ratio : {_safe_mean(result.get('remaining_point_ratio', [])):.3f}",
        f"avg inlier ratio    : {_safe_mean(result.get('inlier_ratio', [])):.3f}",
        f"avg spatial coverage: {_safe_mean(result.get('spatial_coverage', [])):.3f}",
    ]
    degraded = sum(
        1 for s in result.get("anchor_stats", []) if s.get("bridge_degraded")
    )
    lines.append(f"failed bridges      : {degraded}")
    ov = result.get("segment_overhead", {})
    if ov:
        lines.append(
            f"overlap compute     : {ov.get('overlap_compute_percent', 0):.2f}% "
            f"({ov.get('extra_frames', 0)} extra frames)"
        )
    return "\n".join(lines)


def save_segment_metadata(result: dict, output_path: str | Path) -> None:
    """Write per-segment metadata + stats as JSON next to the motion file."""
    out = Path(output_path).with_suffix(".segments.json")
    payload = dict(
        method=result.get("method"),
        anchor_reasons=result.get("anchor_reasons", []),
        anchor_stats=result.get("anchor_stats", []),
        segments=result.get("segments", []),
        segment_overhead=result.get("segment_overhead", {}),
        anchor_frames=[int(a) for a in result.get("anchor_frames", [])],
    )
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
