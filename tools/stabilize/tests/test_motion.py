"""Tests for anchor-relative camera-motion estimation."""

import numpy as np
import pytest

from stabilize import motion
from stabilize.smoother import anchor_transition_report
from stabilize.utils import (
    compose,
    decompose_similarity,
    invert_affine,
    similarity_matrix,
)

WIDTH, HEIGHT = 1920, 1080


def _make_tracks(T, N, seed=0, tx_per_frame=0.0, ty_per_frame=0.0,
                 yaw_per_frame=0.0, scale_per_frame=0.0):
    """Synthetic tracks from a known camera transform M_t (frame 0 -> t)."""
    rng = np.random.default_rng(seed)
    base = np.stack([
        rng.uniform(0, WIDTH, N),
        rng.uniform(0, HEIGHT, N),
    ], axis=1)
    ones = np.ones((N, 1))
    pts = np.concatenate([base, ones], axis=1)
    tracks = np.zeros((T, N, 2))
    truth = np.zeros((T, 3, 3))
    for t in range(T):
        M = similarity_matrix(
            tx_per_frame * t,
            ty_per_frame * t,
            yaw_per_frame * t,
            1.0 + scale_per_frame * t,
        )
        truth[t] = M
        tracks[t] = (M @ pts.T).T[:, :2]
    return tracks, truth


def test_exact_motion_is_recovered():
    T, N = 60, 300
    tracks, truth = _make_tracks(
        T, N, tx_per_frame=0.5, ty_per_frame=0.25,
        yaw_per_frame=0.05, scale_per_frame=0.0002,
    )
    visibility = np.ones((T, N), dtype=bool)

    res = motion.estimate_motion(
        tracks, visibility, WIDTH, HEIGHT,
        anchor_interval_seconds=1.0, fps=10.0,
        min_correspondences=20, min_inlier_ratio=0.4,
    )

    assert res["method"] == "anchor_relative"
    assert res["anchor_valid"].all()
    # cumulative maps frame 0 -> t and must match the ground truth.
    for t in range(T):
        assert np.allclose(res["cumulative"][t], truth[t], atol=1e-3)
    # frame_to_global must be the inverse of cumulative.
    assert np.allclose(
        compose(res["cumulative"][10], res["frame_to_global"][10]),
        np.eye(3), atol=1e-6,
    )


def test_anchor_global_matches_anchor_pose():
    T, N = 50, 200
    tracks, truth = _make_tracks(T, N, tx_per_frame=0.7, ty_per_frame=-0.3)
    visibility = np.ones((T, N), dtype=bool)
    fps, interval_s = 10.0, 1.0
    res = motion.estimate_motion(
        tracks, visibility, WIDTH, HEIGHT,
        anchor_interval_seconds=interval_s, fps=fps,
    )
    # anchor_global[i] = M_{K_i -> 0} = inv(truth[K_i]).
    for i, k in enumerate(res["anchor_frames"]):
        assert np.allclose(res["anchor_global"][i], invert_affine(truth[k]), atol=1e-3)


def test_insufficient_points_do_not_crash():
    T, N = 40, 5  # far below min_correspondences
    rng = np.random.default_rng(3)
    tracks = np.stack([
        np.stack([rng.uniform(0, WIDTH, N), rng.uniform(0, HEIGHT, N)], axis=1)
        for _ in range(T)
    ])
    visibility = np.ones((T, N), dtype=bool)

    res = motion.estimate_motion(
        tracks, visibility, WIDTH, HEIGHT,
        anchor_interval_seconds=1.0, fps=10.0, min_correspondences=20,
    )
    assert np.isfinite(res["cumulative"]).all()
    # Almost everything must be flagged unreliable / fallback.
    assert res["fallback_used"].sum() > 0
    assert not res["anchor_valid"][1:].any()


def test_ransac_failure_uses_fallback(monkeypatch):
    T, N = 40, 100
    tracks, _ = _make_tracks(T, N, tx_per_frame=0.4)
    visibility = np.ones((T, N), dtype=bool)

    def failing_estimate_affine(src, dst, **kwargs):
        return None, dict(num_candidates=len(src), num_inliers=0,
                          inlier_ratio=0.0, valid=False)

    monkeypatch.setattr(motion, "estimate_affine", failing_estimate_affine)

    res = motion.estimate_motion(
        tracks, visibility, WIDTH, HEIGHT,
        anchor_interval_seconds=1.0, fps=10.0,
    )
    assert np.isfinite(res["cumulative"]).all()
    assert res["fallback_used"].sum() > 0
    assert not res["anchor_valid"][1:].any()


def test_anchor_transitions_are_continuous():
    T, N = 120, 250
    tracks, _ = _make_tracks(
        T, N, tx_per_frame=0.9, ty_per_frame=0.4, yaw_per_frame=0.03,
    )
    visibility = np.ones((T, N), dtype=bool)
    res = motion.estimate_motion(
        tracks, visibility, WIDTH, HEIGHT,
        anchor_interval_seconds=1.0, fps=10.0,
    )
    report = anchor_transition_report(res["cumulative"], res["anchor_frames"])
    assert report["boundaries"], "expected at least one anchor boundary"
    assert not report["suspicious"]
    # Continuity: the camera position must not jump at an anchor boundary.
    cam = -res["cumulative"][:, :2, 2]
    inc = np.linalg.norm(np.diff(cam, axis=0), axis=1)
    med = np.median(inc)
    for b in report["boundaries"]:
        assert inc[b - 1] < max(2.0, 10 * med)


def _fake_biased_estimator(bias):
    def fake(src, dst, **kwargs):
        info = dict(num_candidates=len(src), num_inliers=len(src),
                    inlier_ratio=1.0, valid=True)
        return similarity_matrix(bias, 0.0, 0.0, 1.0), info
    return fake


def test_naive_integration_accumulates_bias_but_anchor_relative_does_not(monkeypatch):
    """The core drift demonstration.

    A constant, tiny per-estimate bias (0.005 px) is injected into every
    RANSAC fit.  The legacy frame-to-frame estimator integrates it into the
    camera pose (~150 px after 30k frames), while anchor-relative estimation
    only accumulates one bias per anchor, keeping the net translation tiny.
    """
    T, N = 30000, 16
    bias = 0.005
    fps = 24.0
    anchor_interval_seconds = 10.0

    rng = np.random.default_rng(7)
    tracks = np.stack([
        np.stack([rng.uniform(0, WIDTH, N), rng.uniform(0, HEIGHT, N)], axis=1)
        for _ in range(T)
    ])
    visibility = np.ones((T, N), dtype=bool)

    monkeypatch.setattr(motion, "estimate_affine", _fake_biased_estimator(bias))

    legacy = motion.estimate_motion_pairwise(tracks, visibility, WIDTH, HEIGHT)
    legacy_stats = motion.stabilization_stats(legacy["cumulative"])

    new = motion.estimate_motion(
        tracks, visibility, WIDTH, HEIGHT,
        anchor_interval_seconds=anchor_interval_seconds, fps=fps,
    )
    new_stats = motion.stabilization_stats(new["cumulative"])

    # Naive integration: ~bias * T = 150 px of pure integration drift.
    assert legacy_stats["net_translation_px"] > 100.0
    # Anchor-relative: bounded by (number of anchors) * bias, i.e. < 1 px.
    n_anchors = len(new["anchor_frames"])
    assert new_stats["net_translation_px"] < bias * (n_anchors + 5)
    # And an order of magnitude better than the legacy path.
    assert new_stats["net_translation_px"] < legacy_stats["net_translation_px"] / 20.0


def test_drift_diagnostic_matches_manual_median():
    T, N = 30, 50
    rng = np.random.default_rng(11)
    base = np.stack([rng.uniform(0, WIDTH, N), rng.uniform(0, HEIGHT, N)], axis=1)
    tracks = np.zeros((T, N, 2))
    for t in range(T):
        tracks[t] = base + np.array([2.0 * t, -1.5 * t])
    visibility = np.ones((T, N), dtype=bool)

    diag = motion.drift_diagnostic(tracks, visibility, anchor=0)
    assert np.allclose(diag["absolute_dx"][5], 10.0)
    assert np.allclose(diag["absolute_dy"][5], -7.5)
    assert diag["num_valid"][5] == N
