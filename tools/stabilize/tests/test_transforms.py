"""Tests for the homogeneous-matrix helpers and robust fitting."""

import numpy as np

from stabilize.utils import (
    affine_2x3_to_3x3,
    affine_3x3_to_2x3,
    blend_similarity,
    compose,
    decompose_similarity,
    invert_affine,
    robust_linear_fit,
    similarity_matrix,
)


def _apply(M, pts):
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (affine_2x3_to_3x3(M) @ p.T).T[:, :2]


def test_2x3_3x3_roundtrip():
    M = np.array([[1.1, -0.2, 3.5], [0.2, 1.1, -4.0]], dtype=np.float64)
    assert np.allclose(affine_2x3_to_3x3(M)[:2], M)
    assert np.allclose(affine_3x3_to_2x3(affine_2x3_to_3x3(M)), M)


def test_compose_order_is_apply_B_then_A():
    # B translates +10 in x, A then rotates 90 degrees about the origin.
    B = similarity_matrix(10.0, 0.0, 0.0, 1.0)
    A = similarity_matrix(0.0, 0.0, 90.0, 1.0)
    p = np.array([[0.0, 0.0], [1.0, 0.0]])

    expected = _apply(A, _apply(B, p))
    got = _apply(compose(A, B), p)
    assert np.allclose(got, expected, atol=1e-9)


def test_compose_chains_frame_transforms():
    # M_{a->b} followed by M_{b->c} must equal the direct M_{a->c}.
    M_ab = similarity_matrix(5.0, -2.0, 12.0, 1.02)
    M_bc = similarity_matrix(-3.0, 4.0, -7.0, 0.98)
    M_ac = compose(M_bc, M_ab)  # apply ab first, then bc

    pts = np.array([[0.0, 0.0], [100.0, 50.0], [640.0, 480.0]])
    assert np.allclose(_apply(M_ac, pts), _apply(M_bc, _apply(M_ab, pts)))


def test_invert_affine_roundtrip():
    M = similarity_matrix(11.0, -7.0, 33.0, 1.07)
    inv = invert_affine(M)
    assert np.allclose(compose(M, inv), np.eye(3), atol=1e-9)
    assert np.allclose(compose(inv, M), np.eye(3), atol=1e-9)


def test_blend_similarity_endpoints_and_midpoint():
    A = similarity_matrix(0.0, 0.0, 0.0, 1.0)
    B = similarity_matrix(100.0, -50.0, 20.0, 1.2)
    assert np.allclose(blend_similarity(A, B, 0.0), A, atol=1e-9)
    assert np.allclose(blend_similarity(A, B, 1.0), B, atol=1e-9)

    mid = blend_similarity(A, B, 0.5)
    tx, ty, yaw, scale = decompose_similarity(mid)
    assert abs(tx - 50.0) < 1e-6
    assert abs(ty + 25.0) < 1e-6
    assert abs(yaw - 10.0) < 1e-6
    # Scale is interpolated geometrically (in log space).
    assert abs(scale - np.sqrt(1.2)) < 1e-6


def test_blend_similarity_takes_short_rotation_path():
    A = similarity_matrix(0.0, 0.0, 170.0, 1.0)
    B = similarity_matrix(0.0, 0.0, -170.0, 1.0)
    _, _, yaw, _ = decompose_similarity(blend_similarity(A, B, 0.5))
    # Short path crosses +/-180, not through 0.
    assert abs(abs(yaw) - 180.0) < 1e-6


def test_robust_linear_fit_ignores_outliers():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1000.0, 500)
    y = 0.01 * x + 2.0 + rng.normal(0.0, 0.5, size=x.size)
    # Contaminate 20% of samples with large outliers.
    idx = rng.choice(x.size, size=100, replace=False)
    y[idx] += 500.0

    slope, intercept = robust_linear_fit(x, y)
    assert abs(slope - 0.01) < 0.005
    assert abs(intercept - 2.0) < 3.0


def test_robust_polyfit_removes_cubic_drift():
    from stabilize.utils import robust_polyfit

    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 1.0, 800)
    true = np.array([30.0, -18.0, 6.0, 1.0])  # cubic trend
    y = np.polyval(true, x) + rng.normal(0.0, 0.2, size=x.size)
    # Add outliers.
    idx = rng.choice(x.size, size=40, replace=False)
    y[idx] += 50.0

    c = robust_polyfit(x, y, degree=3)
    # Robust fit ignores the outliers: median residual must be tiny and the
    # recovered coefficients close to the true cubic.
    assert np.median(np.abs(y - np.polyval(c, x))) < 0.5
    assert np.allclose(c, true, atol=1.0)
