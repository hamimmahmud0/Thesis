"""Tests for camera-path smoothing and stabilization modes."""

import numpy as np

from stabilize import motion
from stabilize.smoother import anchor_transition_report, load_smoothed, smooth_motion
from stabilize.utils import eye3, similarity_matrix


def _write_motion(tmp_path, cumulative, fps=24.0):
    T = cumulative.shape[0]
    zeros = np.zeros(T)
    ones = np.ones(T)
    result = dict(
        pairwise=np.tile(eye3(), (T, 1, 1)),
        cumulative=cumulative,
        frame_to_global=np.linalg.inv(cumulative),
        local=cumulative.copy(),
        reliable=np.ones(T, dtype=bool),
        dx=zeros, dy=zeros, d_yaw_deg=zeros, scale=ones,
        inliers=np.full(T, 100, dtype=np.int64),
        inlier_ratio=ones, usable=np.full(T, 100, dtype=np.int64),
        num_candidates=np.full(T, 100, dtype=np.int64),
        num_inliers=np.full(T, 100, dtype=np.int64),
        transform_valid=np.ones(T, dtype=bool),
        fallback_used=np.zeros(T, dtype=bool),
        anchor_frames=np.array([0], dtype=np.int64),
        anchor_global=np.tile(eye3(), (1, 1, 1)),
        anchor_valid=np.array([True]),
        anchor_index=zeros.astype(np.int64),
        anchor_num_candidates=np.array([100], dtype=np.int64),
        anchor_num_inliers=np.array([100], dtype=np.int64),
        anchor_inlier_ratio=np.array([1.0]),
        anchor_bridged=np.array([False]),
        anchor_interval_seconds=0.0,
        anchor_interval_frames=1,
        method="anchor_relative",
    )
    path = tmp_path / "motion.npz"
    motion.save_motion(result, tmp_path / "motion", fps=fps)
    return path


def test_locked_mode_removes_net_translation(tmp_path):
    T = 600
    drift = 0.5  # px/frame
    rng = np.random.default_rng(0)
    cum = np.tile(eye3(), (T, 1, 1))
    jitter = rng.normal(0.0, 0.3, size=T)
    for t in range(T):
        cum[t] = similarity_matrix(-drift * t + jitter[t], 0.0, 0.0, 1.0)

    path = _write_motion(tmp_path, cum)
    raw_net = motion.stabilization_stats(cum)["net_translation_px"]
    assert raw_net > 200.0

    res = smooth_motion(path, tmp_path / "smooth_locked.npz", sigma=5.0, mode="locked")
    ref_net = motion.stabilization_stats(res["smoothed_cumulative"])["net_translation_px"]
    assert ref_net < 5.0
    assert res["drift"]["slope_x"] < 0.0


def test_natural_mode_preserves_slow_pan(tmp_path):
    T = 600
    drift = 0.5
    rng = np.random.default_rng(1)
    cum = np.tile(eye3(), (T, 1, 1))
    jitter = rng.normal(0.0, 0.3, size=T)
    for t in range(T):
        cum[t] = similarity_matrix(-drift * t + jitter[t], 0.0, 0.0, 1.0)

    path = _write_motion(tmp_path, cum)
    raw_net = motion.stabilization_stats(cum)["net_translation_px"]

    res = smooth_motion(path, tmp_path / "smooth_natural.npz", sigma=5.0, mode="natural")
    ref_net = motion.stabilization_stats(res["smoothed_cumulative"])["net_translation_px"]
    # Slow pan must survive (no detrend in natural mode).
    assert ref_net > 0.8 * raw_net
    assert "drift" in res and res["drift"] is None


def test_smoothing_roundtrip_load(tmp_path):
    T = 120
    cum = np.tile(eye3(), (T, 1, 1))
    for t in range(T):
        cum[t] = similarity_matrix(0.0, 0.0, 0.01 * t, 1.0)
    path = _write_motion(tmp_path, cum)
    out = tmp_path / "smooth.npz"
    smooth_motion(path, out, sigma=3.0, mode="natural")
    loaded = load_smoothed(out)
    assert loaded["mode"] == "natural"
    assert loaded["smoothed_cumulative"].shape == (T, 3, 3)


def test_transition_report_flags_jump():
    T = 100
    smooth = np.tile(eye3(), (T, 1, 1))
    for t in range(T):
        smooth[t] = similarity_matrix(-0.2 * t, 0.0, 0.0, 1.0)
    anchors = np.array([0, 50], dtype=np.int64)
    assert not anchor_transition_report(smooth, anchors)["suspicious"]

    jumpy = smooth.copy()
    for t in range(50, T):
        jumpy[t] = similarity_matrix(-0.2 * t + 40.0, 0.0, 0.0, 1.0)
    assert anchor_transition_report(jumpy, anchors)["suspicious"]
