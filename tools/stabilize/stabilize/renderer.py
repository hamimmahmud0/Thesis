"""Video stabilization renderer.

Reads each frame from the source video, applies a per-frame affine warp
that locks tracked keypoints to their first-frame positions (optionally
with camera-path smoothing), and writes the result via FFmpeg.

Supports rectangular crops (``--crop-width``, ``--crop-height``), full-frame
output (``--no-crop``), automatic black-border trimming
(``--crop-black-border``), validates border safety, and handles FFmpeg
failures properly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils import eye3, open_video, human_time

EST_SEC_PER_1024 = 0.09
EST_BYTES_PER_1024 = 31_000
EST_LOSSLESS_FACTOR = 16  # rough size multiplier of lossless vs CRF 20 (measured on 4K drone footage)

_FFMPEG_ENCODER: str | None = None

# Preferred H.264 encoders, in order.  NVENC is used first when it is
# actually usable on the system (fastest on NVIDIA GPUs); libopenh264 is
# the compact fallback, and libx264 the universally available last resort.
_ENCODER_PREFERENCE = ("h264_nvenc", "libopenh264", "libx264")


def _probe_encoder(name: str) -> bool:
    """Return True if the installed FFmpeg can *actually* encode with *name*.

    A functional one-frame encode is used rather than just grepping
    ``ffmpeg -encoders``, because an encoder can be compiled in yet still
    be unusable at runtime (e.g. ``h264_nvenc`` without a reachable GPU).
    """
    try:
        rc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:s=64x64:d=0.05",
                "-frames:v", "1", "-c:v", name,
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=30,
        ).returncode
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return rc == 0


def resolve_encoder() -> str:
    """Return the H.264 encoder to use (cached across calls).

    Preference order: ``h264_nvenc`` (NVIDIA GPU) -> ``libopenh264`` ->
    ``libx264``.  The probe verifies a real encode, so a compiled but
    unusable encoder is skipped.
    """
    global _FFMPEG_ENCODER
    if _FFMPEG_ENCODER is None:
        for name in _ENCODER_PREFERENCE:
            if _probe_encoder(name):
                _FFMPEG_ENCODER = name
                break
        _FFMPEG_ENCODER = _FFMPEG_ENCODER or "libx264"
    return _FFMPEG_ENCODER


def _encoder_preset(encoder: str) -> str:
    """Return the ``-preset`` value appropriate for *encoder*.

    NVENC uses quality presets ``p1``..``p7`` (``p4`` is the balanced
    default); the software encoders use x264-style presets.
    """
    return "p4" if encoder == "h264_nvenc" else "veryfast"


def _quality_args(
    encoder: str, crf: int, w: int, h: int, fps: float
) -> list[str]:
    """Return FFmpeg quality arguments appropriate for *encoder*.

    ``crf <= 0`` means lossless where the encoder supports it: ``-crf 0``
    for libx264 and ``-tune lossless`` for NVENC.  libopenh264 exposes no
    CRF/QP control at all, so it gets a bits-per-pixel bitrate heuristic
    (and cannot be truly lossless).
    """
    if encoder == "libx264":
        return ["-crf", str(max(crf, 0))]
    if encoder == "h264_nvenc":
        if crf <= 0:
            return ["-tune", "lossless"]
        return ["-cq", str(crf)]
    # libopenh264: bitrate-only wrapper.
    bpp = 1.0 if crf <= 0 else max(0.02, 0.15 - 0.004 * crf)
    return ["-b:v", str(int(w * h * fps * bpp))]


def _quality_label(encoder: str, crf: int) -> str:
    """Human-readable quality description for log lines."""
    if crf > 0:
        return f"CRF{crf}"
    if encoder == "libopenh264":
        return "high-bitrate (no lossless mode)"
    return "lossless"


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


def _black_border_crop(
    raw_cum: np.ndarray,
    smooth_cum: np.ndarray | None,
    n_render: int,
    vid_w: int,
    vid_h: int,
    shift_x: int = 0,
    shift_y: int = 0,
) -> tuple[int, int, int, int]:
    """Return the largest centred crop window ``(x0, y0, w, h)`` whose four
    corners stay inside the source frame for every rendered frame.

    The result is the biggest (by area) crop that removes the black borders
    introduced by stabilisation: the output pixel at ``(x0 + u, y0 + v)`` is
    never sampled from outside the source frame.  The warp used for the
    check is the *effective* per-frame warp (``raw @ inv(smooth)`` when
    smoothing is active, ``raw`` otherwise).
    """
    if n_render <= 0:
        return 0, 0, vid_w, vid_h

    raw = raw_cum[:n_render]
    if smooth_cum is None:
        warp = raw
    else:
        sm = smooth_cum[:n_render]
        if len(sm) != n_render:
            raise SystemExit(
                f"Smoothed path has {len(sm)} frames but motion has {n_render}"
            )
        warp = np.einsum("tij,tjk->tik", raw, np.linalg.inv(sm))

    cx = min(max(vid_w // 2 + shift_x, 0), vid_w - 1)
    cy = min(max(vid_h // 2 + shift_y, 0), vid_h - 1)

    def safe(w: int, h: int) -> bool:
        x0 = cx - w // 2
        y0 = cy - h // 2
        return _border_check(warp, n_render, x0, y0, w, h, vid_w, vid_h,
                             far_margin=1)

    def max_w(h: int) -> int:
        lo, hi = 0, vid_w
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if safe(mid, h):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def max_h(w: int) -> int:
        lo, hi = 0, vid_h
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if safe(w, mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    w, h = vid_w, vid_h
    for _ in range(8):
        nw, nh = max_w(h), max_h(w)
        if (nw, nh) == (w, h):
            break
        w, h = nw, nh

    return cx - w // 2, cy - h // 2, w, h


def _border_check(
    cumulative: np.ndarray,
    n_render: int,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    vid_w: int,
    vid_h: int,
    far_margin: int = 0,
) -> bool:
    """Check whether the crop window stays within frame bounds for all frames.

    Corners are warp-mapped and required to stay in ``[0, vid_w-1+far_margin]``
    x ``[0, vid_h-1+far_margin]``.  ``far_margin=1`` allows a corner to touch
    the outermost boundary (the exact corner point is never a rendered pixel).
    """
    corners = np.array([
        [x0,      y0,      1],
        [x0 + crop_w, y0,      1],
        [x0 + crop_w, y0 + crop_h, 1],
        [x0,      y0 + crop_h, 1],
    ], dtype=np.float64)
    src_xy = np.einsum("tij,kj->tki", cumulative[:n_render], corners)
    ok = (
        (src_xy[..., 0] >= 0).all()
        and (src_xy[..., 0] <= vid_w - 1 + far_margin).all()
        and (src_xy[..., 1] >= 0).all()
        and (src_xy[..., 1] <= vid_h - 1 + far_margin).all()
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
    crf: int = 18,
    no_crop: bool = False,
    crop_black_border: bool = False,
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
    crf : output quality.  18 (default) = visually lossless; 0 =
        mathematically lossless (very large files); otherwise x264 CRF.
    no_crop : render the full source frame (WxH) instead of cropping.
        Black borders may appear where content moved out of view.
    crop_black_border : auto-detect the largest centred crop that removes
        the black borders introduced by stabilisation, instead of using
        ``crop_width``/``crop_height``.  Mutually exclusive with ``no_crop``.
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

    if no_crop and crop_black_border:
        raise SystemExit(
            "--no-crop and --crop-black-border are mutually exclusive"
        )

    if crop_width is None:
        crop_width = crop_height or 1024
    if crop_height is None:
        crop_height = crop_width

    n_render = min(n_vid, T_csv, n_frames) if n_frames > 0 else min(n_vid, T_csv)

    # ---- Choose the output size and crop-window origin ----
    if no_crop:
        out_w, out_h = W, H
        x0 = y0 = 0
        crop_desc = f"full frame {W}x{H} (no-crop)"
    elif crop_black_border:
        x0, y0, out_w, out_h = _black_border_crop(
            raw_cum, smooth_cum, n_render, W, H, shift_x, shift_y
        )
        if out_w < 1 or out_h < 1:
            cap.release()
            raise SystemExit(
                f"Auto black-border crop found no safe region on a {W}x{H} "
                f"frame.  Reduce --shift-x/--shift-y or use --no-crop."
            )
        crop_desc = f"auto black-border {out_w}x{out_h}"
    else:
        if crop_width > W or crop_height > H:
            cap.release()
            raise SystemExit(
                f"Crop {crop_width}x{crop_height} exceeds video {W}x{H}"
            )
        out_w, out_h = crop_width, crop_height
        x0 = (W - out_w) // 2 + shift_x
        y0 = (H - out_h) // 2 + shift_y
        if x0 < 0 or y0 < 0 or x0 + out_w > W or y0 + out_h > H:
            cap.release()
            raise SystemExit(
                f"Crop window ({x0},{y0}) size {out_w}x{out_h} "
                f"leaves video {W}x{H}. Reduce shift or crop size."
            )
        crop_desc = f"{out_w}x{out_h}"

    # ---- H.264 with yuv420p requires even dimensions ----
    if out_w % 2 or out_h % 2:
        print(
            f"NOTE: crop {out_w}x{out_h} not divisible by 2 — "
            "trimmed by 1 px for yuv420p"
        )
        out_w -= out_w % 2
        out_h -= out_h % 2

    # ---- Border safety check ----
    if crop_black_border:
        # Auto-crop is safe by construction (largest in-bounds window).
        safe = True
    elif no_crop:
        # Full frame: black borders are expected; no check performed.
        safe = True
    else:
        safe = _border_check(raw_cum, n_render, x0, y0, out_w, out_h, W, H)
    scale_px = (out_w / 1024.0) * (out_h / 1024.0)
    est_sec = EST_SEC_PER_1024 * scale_px * n_render
    est_mb = EST_BYTES_PER_1024 * scale_px * n_render / 1e6
    if crf <= 0:
        est_mb *= EST_LOSSLESS_FACTOR

    encoder = resolve_encoder()
    preset = _encoder_preset(encoder)

    if not safe:
        print("WARNING: border safety check FAILED — output may contain black edges")
    if no_crop:
        print("NOTE: no-crop — black borders appear where the warp moves content out of frame")

    print(f"Video:       {n_vid} frames, {W}x{H} @ {fps:.3f} fps")
    print(f"Motion:      {T_csv} rows from {Path(motion_npz).name}")
    print(f"Render:      {n_render} frames, crop {crop_desc}")
    print(f"Window:      x=[{x0}..{x0 + out_w}], y=[{y0}..{y0 + out_h}]")
    print(f"Encoder:     {encoder} {_quality_label(encoder, crf)} yuv420p")
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
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-an",
        "-c:v", encoder, "-preset", preset,
        *_quality_args(encoder, crf, out_w, out_h, fps),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        cap.release()
        raise SystemExit("FFmpeg not found. Install FFmpeg and try again.")

    pbar = tqdm(total=n_render, desc="Rendering", unit="frame",
                bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")
    try:
        for t in range(n_render):
            ok, frame = cap.read()
            if not ok:
                pbar.write(f"Video ended early at frame {t}")
                break

            M = _combined_warp(raw_cum[t], None if smooth_cum is None else smooth_cum[t], (x0, y0))
            out = cv2.warpAffine(
                frame, M, (out_w, out_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )

            try:
                proc.stdin.write(out.tobytes())
            except BrokenPipeError:
                pbar.write("FFmpeg pipe broke — encoder may have failed")
                break

            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if proc.stdin:
            proc.stdin.close()

    return_code = proc.wait()
    if return_code != 0:
        raise SystemExit(f"FFmpeg failed with return code {return_code}")

    print(f"Wrote: {output_path}")
