"""Tests for the fresh-grid + bridge-point anchor architecture.

These tests inject a synthetic ``track_fn`` so the planner, bridge
registration, quality-triggered re-anchoring and composition can be
exercised without a GPU or CoTracker.
"""

import numpy as np
import pytest

from stabilize import segments
from stabilize.segments import (
    AnchorReason,
    SegmentConfig,
    estimate_bridge,
    plan_segments,
)
from stabilize.utils import (
    compose,
    decompose_similarity,
    invert_affine,
    similarity_matrix,
)
from stabilize.tracker import make_segment_tracker  # noqa: F401  (import check)

WIDTH, HEIGHT = 1280, 720


def _apply(M, pts):
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (M @ p.T).T[:, :2]


class SyntheticWorld:
    """Known camera path + simulated per-segment fresh grids."""

    def __init__(
        self,
        total_frames,
        fps=30.0,
        grid_size=16,
        tx_per_frame=0.2,
        ty_per_frame=0.05,
        yaw_per_frame=0.01,
        lifetime=None,
        cluster_after=None,
        seed=0,
    ):
        self.T = total_frames
        self.fps = fps
        self.grid_size = grid_size
        self.tx = tx_per_frame
        self.ty = ty_per_frame
        self.yaw = yaw_per_frame
        self.lifetime = lifetime
        self.cluster_after = cluster_after
        self.seed = seed
        self._query_points = {}

    def truth(self, t):
        return similarity_matrix(
            self.tx * t, self.ty * t, self.yaw * t, 1.0
        )

    def track_fn(self, anchor_frame, end_frame):
        rng = np.random.default_rng(self.seed * 100000 + int(anchor_frame))
        n = self.grid_size
        xs = np.linspace(0.1 * WIDTH, 0.9 * WIDTH, n)
        ys = np.linspace(0.1 * HEIGHT, 0.9 * HEIGHT, n)
        X, Y = np.meshgrid(xs, ys)
        q = np.stack([X.ravel(), Y.ravel()], axis=1)
        q = q + rng.normal(0.0, 2.0, q.shape)
        self._query_points[int(anchor_frame)] = q.copy()

        M_anchor = self.truth(anchor_frame)
        p0 = _apply(invert_affine(M_anchor), q)  # frame-0 physical points

        L = int(end_frame) - int(anchor_frame)
        tracks = np.zeros((L, q.shape[0], 2))
        visibility = np.ones((L, q.shape[0]), dtype=bool)
        for i in range(L):
            t = int(anchor_frame) + i
            p = _apply(self.truth(t), p0)
            if self.cluster_after is not None and i >= self.cluster_after:
                # Keep every point visible but collapse them into a corner
                # while preserving the point-wise affine relationship.
                center = p.mean(axis=0)
                p = (p - center) * 0.02 + np.array([0.06 * WIDTH, 0.06 * HEIGHT])
            tracks[i] = p
            vis = (
                (p[:, 0] >= 0) & (p[:, 0] < WIDTH)
                & (p[:, 1] >= 0) & (p[:, 1] < HEIGHT)
            )
            if self.lifetime is not None and i >= self.lifetime:
                vis[:] = False
            visibility[i] = vis
        return segments.make_segment(
            segment_id=0,
            anchor_frame=int(anchor_frame),
            start_frame=int(anchor_frame),
            end_frame=int(end_frame),
            tracks=tracks,
            visibility=visibility,
            query_points=q,
            width=WIDTH,
            height=HEIGHT,
            fps=self.fps,
        )


def _cfg(**kw):
    base = dict(
        anchor_interval_seconds=2.0,
        min_anchor_interval_seconds=0.5,
        anchor_overlap_seconds=0.5,
        min_remaining_point_ratio=0.40,
        min_inlier_ratio=0.40,
        min_inlier_count=20,
        min_spatial_coverage=0.35,
        quality_failure_patience_frames=3,
        coverage_grid_rows=4,
        coverage_grid_cols=4,
        min_correspondences=20,
    )
    base.update(kw)
    return SegmentConfig(**base)


def test_fresh_grid_lifecycle():
    world = SyntheticWorld(300)
    cfg = _cfg()
    result = plan_segments(world.track_fn, world.T, WIDTH, HEIGHT, world.fps, cfg)

    assert len(result["anchor_frames"]) >= 2
    seg0 = result["segments"][0]["anchor"]
    seg1 = result["segments"][1]["anchor"]
    q0 = world._query_points[seg0]
    q1 = world._query_points[seg1]
    # Segment B must get its own grid, not reuse A's.
    assert not np.allclose(q0, q1)
    # Segment ownership changes at the boundary.
    assert result["segment_id"][seg0] == 0
    assert result["segment_id"][seg1] == 1


def test_bridge_recovers_B_to_A_direction():
    world = SyntheticWorld(400)
    cfg = _cfg()
    seg_a = segments.analyze_segment(world.track_fn(0, 200), cfg)
    k_b = 150
    seg_b = segments.analyze_segment(world.track_fn(k_b, 300), cfg)

    bridge, info = estimate_bridge(seg_a, seg_b, cfg, overlap_frames=30)
    assert bridge is not None
    assert info["num_candidates"] > 0

    # bridge must map Anchor B -> Anchor A: M_{K_B -> K_A}
    expected = compose(world.truth(0), invert_affine(world.truth(k_b)))
    assert np.allclose(bridge, expected, atol=1e-6)

    # It must NOT be the opposite direction.
    opposite = invert_affine(expected)
    assert not np.allclose(bridge, opposite, atol=1e-3)


def test_global_composition():
    world = SyntheticWorld(400)
    cfg = _cfg(anchor_interval_seconds=1.0, anchor_overlap_seconds=0.4)
    result = plan_segments(world.track_fn, world.T, WIDTH, HEIGHT, world.fps, cfg)

    af = result["anchor_frames"]
    ag = result["anchor_global"]
    for i, k in enumerate(af):
        # anchor_global[i] = M_{K_i -> 0} = inv(truth(K_i))
        assert np.allclose(ag[i], invert_affine(world.truth(k)), atol=1e-5)

    for t in range(world.T):
        # cumulative = M_{0 -> t}
        assert np.allclose(result["cumulative"][t], world.truth(t), atol=1e-3)


def test_grid_deterioration_triggers_reanchor():
    world = SyntheticWorld(400, lifetime=60)
    cfg = _cfg(anchor_interval_seconds=10.0, min_anchor_interval_seconds=0.5,
               quality_failure_patience_frames=3)
    result = plan_segments(world.track_fn, world.T, WIDTH, HEIGHT, world.fps, cfg)

    reasons = result["anchor_reasons"]
    assert any(r == AnchorReason.LOW_POINT_COUNT.value for r in reasons)
    # Reanchor happened far earlier than the 10 s (300 frame) maximum.
    spacings = np.diff(result["anchor_frames"])
    assert spacings.min() <= 70


def test_spatial_degeneration_triggers_reanchor():
    # Points stay fully visible but collapse into one corner.
    world = SyntheticWorld(400, cluster_after=60)
    cfg = _cfg(anchor_interval_seconds=10.0, min_anchor_interval_seconds=0.5,
               quality_failure_patience_frames=3)
    result = plan_segments(world.track_fn, world.T, WIDTH, HEIGHT, world.fps, cfg)

    reasons = result["anchor_reasons"]
    assert any(r == AnchorReason.LOW_SPATIAL_COVERAGE.value for r in reasons)
    # The anchor was created early (well before the 10 s / 300 frame max),
    # purely because the surviving points lost spatial spread.
    spacings = np.diff(result["anchor_frames"])
    assert spacings.min() <= 70


def test_anchor_transition_continuity():
    world = SyntheticWorld(500, tx_per_frame=0.3, yaw_per_frame=0.02)
    cfg = _cfg(anchor_interval_seconds=1.0, anchor_overlap_seconds=0.5)
    result = plan_segments(world.track_fn, world.T, WIDTH, HEIGHT, world.fps, cfg)

    cam = -result["cumulative"][:, :2, 2]
    inc = np.linalg.norm(np.diff(cam, axis=0), axis=1)
    tolerance = max(3.0, 6.0 * float(np.median(inc)))
    for b in result["anchor_frames"][1:]:
        assert inc[int(b) - 1] < tolerance
    # No frame-to-frame catastrophic jump at all.
    assert inc.max() < 10.0 * max(float(np.median(inc)), 1e-6)


def test_failed_bridge_preserves_continuity(monkeypatch):
    world = SyntheticWorld(400)

    def failing_bridge(*a, **k):
        return None, {"bridge": False, "degraded": True, "num_inliers": 0,
                      "method": "hold"}

    monkeypatch.setattr(segments, "estimate_bridge", failing_bridge)
    cfg = _cfg(anchor_interval_seconds=1.0, anchor_overlap_seconds=0.4)
    result = plan_segments(world.track_fn, world.T, WIDTH, HEIGHT, world.fps, cfg)

    # Transforms must remain finite and continuous (no reset / jump).
    assert np.isfinite(result["cumulative"]).all()
    cam = -result["cumulative"][:, :2, 2]
    inc = np.linalg.norm(np.diff(cam, axis=0), axis=1)
    assert inc.max() < 6.0
    # Degraded transitions are recorded.
    assert not result["anchor_valid"].all()


def test_long_sequence_with_periodic_population_loss():
    # Populations vanish every ~90 frames; refreshing grids must keep the
    # global estimate faithful to the known camera path throughout.
    T = 900
    cfg = _cfg(anchor_interval_seconds=10.0, min_anchor_interval_seconds=0.5,
               quality_failure_patience_frames=3)
    # Custom world where each segment loses its points after 90 frames.
    world = SyntheticWorld(T, lifetime=90)
    result = plan_segments(world.track_fn, T, WIDTH, HEIGHT, world.fps, cfg)

    truth = np.stack([world.truth(t) for t in range(T)])
    err = np.abs(result["cumulative"] - truth)
    assert err.max() < 5e-2
    # Multiple anchors were created, not one long grid.
    assert len(result["anchor_frames"]) >= 5

    # Report statistics must be well-formed.
    rep = segments.segment_report(result)
    assert "anchors created" in rep
    overhead = result["segment_overhead"]
    assert overhead["num_segments"] == len(result["segments"])
    assert overhead["extra_frames"] >= 0


def test_config_validation():
    with pytest.raises(ValueError):
        SegmentConfig(anchor_interval_seconds=0).validate()
    with pytest.raises(ValueError):
        SegmentConfig(min_anchor_interval_seconds=5.0,
                      anchor_interval_seconds=2.0).validate()
    with pytest.raises(ValueError):
        SegmentConfig(min_spatial_coverage=1.5).validate()
