"""Video stabilization renderer.

Reads each frame from the source video, applies a per-frame affine warp
that locks tracked keypoints to their first-frame positions (optionally
with camera-path smoothing), and writes the result via FFmpeg.

Supports rectangular crops (``--crop-width``, ``--crop-height``),
validates border safety, and handles FFmpeg failures properly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .utils import eye3, open_video, human_time

EST_SEC_PER_1024 = 0.09
EST_BYTES_PER_1024 = 31_000


def _combined_warp(
    raw_cum: np.ndarray,
    smooth_cum: np.ndarray | None,
    origin: tuple[int, int],
) -> np.ndarray:
    """Build the 2x3 warpAffine matrix (with WARP_INVERSE_MAP).

    Parameters
    ----------
    raw_cum : (3, 3) cumulative forward transform for frame t.
    smooth_cum : (3, 3) smoothed cumulative transform, or None for
        track-locked (no smoothing).
    origin : (x0, y0) top-left corner of the crop window in frame-0 coords.

    Returns
    -------
    (2, 3) float32 matrix M such that ``warpAffine(frame, M, ...,
    WARP_INVERSE_MAP)`` produces the stabilised output.
    """
    if smooth_cum is None:
        # Track-locked: no smoothing.  Lock to frame-0 positions.
        # M u = C_t (o + u)  =>  dst(u) = src(C_t (o + u))
        lin = raw_cum[:2, :2]
        off = raw_cum[:2, 2:3] + lin @ np.asarray(
            origin, dtype=np.float64
        ).reshape(2, 1)
    else:
        # Smoothed path: correction warp.
        # W = raw_cum @ inv(smooth_cum)
        W = raw_cum @ np.linalg.inv(smooth_cum)
        lin = W[:2, :2]
        off = W[:2, 2:3] + lin @ np.asarray(
            origin, dtype=np.float64
        ).reshape(2, 1)

    return np.hstack([lin, off]).astype(np.float32)


def _border_check(
    cumulative: np.ndarray,
    n_render: int,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    vid_w: int,
    vid_h: int,
) -> bool:
    """Check whether the crop window stays within frame bounds for all frames."""
    corners = np.array([
        [x0,      y0,      1],
        [x0 + crop_w, y0,      1],
        [x0 + crop_w, y0 + crop_h, 1],
        [x0,      y0 + crop_h, 1],
    ], dtype=np.float64)
    src_xy = np.einsum("tij,kj->tki", cumulative[:n_render], corners)
    ok = (
        (src_xy[..., 0] >= 0).all()
        and (src_xy[..., 0] <= vid_w - 1).all()
        and (src_xy[..., 1] >= 0).all()
        and (src_xy[..., 1] <= vid_h - 1).all()
    )
    return bool(ok)


def render(
    video_path: str | Path,
    output_path: str | Path,
    motion_npz: str | Path,
    smooth_npz: str | Path | None = None,
    crop_width: int = 1024,
    crop_height: int = 1024,
    shift_x: int = 0,
    shift_y: int = 0,
    n_frames: int = 0,
    fps_override: float | None = None,
    crf: int = 20,
) -> None:
    """Render a stabilised video.

    Parameters
    ----------
    video_path : source video file.
    output_path : output .mp4 file.
    motion_npz : ``motion.npz`` from ``stabilize estimate``.
    smooth_npz : optional ``motion_smooth.npz`` from ``stabilize smooth``.
    crop_width, crop_height : output frame dimensions in pixels.
    shift_x, shift_y : shift the crop window centre (frame-0 px).
    n_frames : render only the first N frames (0 = all).
    fps_override : override the output FPS (None = use source video FPS).
    crf : x264 CRF quality (lower = better, 18–28 typical).
    """
    from .motion import load_motion

    # ---- Load motion data ----
    motion = load_motion(motion_npz)
    raw_cum = motion["cumulative"]
    T_csv = len(raw_cum)

    smooth_cum = None
    if smooth_npz is not None:
        from .smoother import load_smoothed

        sm = load_smoothed(smooth_npz)
        smooth_cum = sm["smoothed_cumulative"]
        if len(smooth_cum) != T_csv:
            raise SystemExit(
                f"Smoothed path has {len(smooth_cum)} frames but motion "
                f"has {T_csv} — mismatched files?"
            )

    # ---- Open video ----
    cap, n_vid, W, H, fps = open_video(video_path)
    if fps_override is not None:
        fps = fps_override

    if crop_width > W or crop_height > H:
        cap.release()
        raise SystemExit(
            f"Crop {crop_width}x{crop_height} exceeds video {W}x{H}"
        )

    n_render = min(n_vid, T_csv, n_frames) if n_frames > 0 else min(n_vid, T_csv)

    x0 = (W - crop_width) // 2 + shift_x
    y0 = (H - crop_height) // 2 + shift_y
    if x0 < 0 or y0 < 0 or x0 + crop_width > W or y0 + crop_height > H:
        cap.release()
        raise SystemExit(
            f"Crop window ({x0},{y0}) size {crop_width}x{crop_height} "
            f"leaves video {W}x{H}. Reduce shift or crop size."
        )

    # ---- Border safety check ----
    safe = _border_check(raw_cum, n_render, x0, y0, crop_width, crop_height, W, H)
    scale_px = (crop_width / 1024.0) * (crop_height / 1024.0)
    est_sec = EST_SEC_PER_1024 * scale_px * n_render
    est_mb = EST_BYTES_PER_1024 * scale_px * n_render / 1e6

    if not safe:
        print("WARNING: border safety check FAILED — output may contain black edges")

    print(f"Video:       {n_vid} frames, {W}x{H} @ {fps:.3f} fps")
    print(f"Motion:      {T_csv} rows from {Path(motion_npz).name}")
    print(f"Render:      {n_render} frames, crop {crop_width}x{crop_height}")
    print(f"Window:      x=[{x0}..{x0 + crop_width}], y=[{y0}..{y0 + crop_height}]")
    print(f"Encoder:     libx264 CRF{crf} veryfast, yuv420p")
    print(f"Border:      {'OK' if safe else 'FAIL'}")
    print(f"Est. time:   ~{human_time(est_sec)} (4 vCPU reference)")
    print(f"Est. size:   ~{est_mb:.0f} MB")
    if smooth_cum is not None:
        print(f"Smoothing:   enabled ({Path(smooth_npz).name})")
    else:
        print("Smoothing:   disabled (track-locked)")

    # ---- Start FFmpeg ----
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{crop_width}x{crop_height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        cap.release()
        raise SystemExit("FFmpeg not found. Install FFmpeg and try again.")

    try:
        for t in range(n_render):
            ok, frame = cap.read()
            if not ok:
                print(f"Video ended early at frame {t}")
                break

            M = _combined_warp(raw_cum[t], None if smooth_cum is None else smooth_cum[t], (x0, y0))
            out = cv2.warpAffine(
                frame, M, (crop_width, crop_height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )

            try:
                proc.stdin.write(out.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe broke — encoder may have failed")
                break

            if t % 500 == 0:
                print(f"  frame {t}/{n_render}", flush=True)
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()

    return_code = proc.wait()
    if return_code != 0:
        raise SystemExit(f"FFmpeg failed with return code {return_code}")

    print(f"Wrote: {output_path}")
