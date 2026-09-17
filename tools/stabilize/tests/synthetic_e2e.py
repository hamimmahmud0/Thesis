"""Synthetic end-to-end stabilisation check (no CoTracker / GPU required).

This builds a texture video with a known camera path (slow pan + high
frequency jitter), writes analytic tracks in the ``stabilize`` tracks.npz
format, then runs the real ``estimate -> smooth -> render`` CLI and measures
the residual temporal variation of the output.

Usage::

    python tests/synthetic_e2e.py --out-dir /tmp/stab_e2e --mode locked

The key metric is ``output_temporal_std``: high for the raw input (camera
moves) and low for a correctly stabilised locked output.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stabilize.utils import similarity_matrix  # noqa: E402

W, H = 640, 360
T = 180
FPS = 30.0


def build_scene(seed: int = 0) -> np.ndarray:
    """A textured canvas larger than the frame so panning stays in bounds."""
    cw, ch = W + 260, H + 200
    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 255, size=(ch, cw, 3), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    for _ in range(25):
        p = (int(rng.integers(20, cw - 20)), int(rng.integers(20, ch - 20)))
        cv2.circle(canvas, p, int(rng.integers(6, 18)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return canvas


def camera_path(T: int):
    """Known frame-0 -> frame-t similarity transforms (pan + jitter)."""
    mats = []
    for t in range(T):
        pan_x = 0.6 * t
        pan_y = 0.15 * t
        jitter_x = 2.0 * np.sin(2 * np.pi * t / 5.0)
        jitter_y = 1.5 * np.sin(2 * np.pi * t / 3.0 + 0.7)
        yaw = 0.02 * np.sin(2 * np.pi * t / 6.0)
        mats.append(similarity_matrix(pan_x + jitter_x, pan_y + jitter_y, yaw, 1.0))
    return mats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/stab_e2e")
    ap.add_argument("--mode", default="locked", choices=["natural", "locked", "off"])
    ap.add_argument("--anchor-interval-seconds", type=float, default=1.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    canvas = build_scene()
    mats = camera_path(T)

    # ---- Write the source video ----
    video_path = out / "source.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    frames = []
    for t in range(T):
        # Default warpAffine computes dst(x) = src(M^{-1} x), so a canvas
        # point p appears at M_t @ p -- matching the tracks below.
        frame = cv2.warpAffine(canvas, mats[t][:2], (W, H))
        frames.append(frame)
        writer.write(frame)
    writer.release()

    # ---- Analytic tracks (frame0 base points -> M_t @ base) ----
    rng = np.random.default_rng(1)
    N = 400
    base = np.stack([
        rng.uniform(0, W, N),
        rng.uniform(0, H, N),
    ], axis=1)
    tracks = np.zeros((T, N, 2), dtype=np.float32)
    visibility = np.zeros((T, N), dtype=bool)
    for t in range(T):
        p = np.concatenate([base, np.ones((N, 1))], axis=1)
        xy = (mats[t] @ p.T).T[:, :2]
        tracks[t] = xy
        visibility[t] = (
            (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
        )
    tracks_npz = out / "tracks.npz"
    np.savez(
        tracks_npz,
        tracks=tracks,
        visibility=visibility,
        query_points=base.astype(np.float32),
        meta=str({
            "width": W, "height": H, "fps": FPS,
            "total_frames": T, "processed_frames": T,
        }),
    )

    def run(*cmd):
        print("$", " ".join(str(c) for c in cmd), flush=True)
        subprocess.check_call([sys.executable, "-m", "stabilize", *map(str, cmd)])

    motion = out / "motion"
    smooth = out / "motion_smooth.npz"
    stable = out / "stabilized.mp4"

    run(
        "estimate", tracks_npz, "-o", motion,
        "--fps", str(FPS), "--width", str(W), "--height", str(H),
        "--anchor-interval-seconds", str(args.anchor_interval_seconds),
        "--drift", "--debug",
    )
    render_cmd = [
        "render", video_path, "--tracks", tracks_npz, "--motion",
        motion.with_suffix(".npz"),
    ]
    if args.mode != "off":
        run("smooth", motion.with_suffix(".npz"), "-o", smooth, "--mode", args.mode)
        render_cmd += ["--smooth", smooth]
    render_cmd += [
        "--crop-black-border", "-o", stable, "--confirm", "synthetic e2e",
    ]
    run(*render_cmd)

    # ---- Measure residual temporal variation ----
    def temporal_std(path):
        cap = cv2.VideoCapture(str(path))
        grays = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        cap.release()
        arr = np.stack(grays[:T])
        return float(arr.std(axis=0).mean())

    src_std = temporal_std(video_path)
    out_std = temporal_std(stable)
    print(f"\nsource_temporal_std = {src_std:.2f}")
    print(f"output_temporal_std = {out_std:.2f} (mode={args.mode})")

    # Locked output is (near) static -> temporal std must drop a lot.
    if args.mode == "locked":
        assert out_std < 0.35 * src_std, "locked stabilisation did not reduce motion"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
