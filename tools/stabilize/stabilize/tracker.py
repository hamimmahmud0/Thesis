"""Local CoTracker3 keypoint tracking.

Runs the CoTracker3 online model on a local GPU to extract per-frame
keypoint trajectories from a video.  Frames are read in BGR (OpenCV)
order and converted to RGB before being fed to the model.

Requires ``torch``, ``cotracker``, and a CUDA GPU.  For remote/server
tracking, use the ``cot_mcp_server`` separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def track_video(
    video_path: str | Path,
    output_npz: str | Path,
    checkpoint: str | Path,
    grid_size: int = 16,
    grid_query_frame: int = 0,
    max_video_dim: int = 1280,
    device: str | None = None,
) -> dict:
    """Track keypoint trajectories through *video_path* using CoTracker3.

    Parameters
    ----------
    video_path : path to the input video.
    output_npz : path to write the output .npz file.
    checkpoint : path to ``scaled_online.pth``.
    grid_size : NxN grid of query points (max 32).
    grid_query_frame : frame index the grid is sampled from (default 0).
    max_video_dim : resize longest side to this before tracking (GPU memory).
    device : torch device string, e.g. ``"cuda:0"``.  Auto-detected if None.

    Returns
    -------
    dict with keys: tracks, visibility, query_points, width, height, fps,
    total_frames, processed_frames, grid_size, num_points.
    """
    import torch
    from cotracker.models.core.model_utils import get_points_on_a_grid
    from cotracker.predictor import CoTrackerOnlinePredictor

    video_path = Path(video_path)
    output_npz = Path(output_npz)
    checkpoint = Path(checkpoint)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # ---- Probe video ----
    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()

    if total < 2:
        raise SystemExit(f"Video too short: {total} frame(s), need >= 2")

    # ---- Compute tracking resolution ----
    scale = min(1.0, max_video_dim / max(orig_w, orig_h))
    tw = max(1, int(round(orig_w * scale)) // 2 * 2)
    th = max(1, int(round(orig_h * scale)) // 2 * 2)

    # ---- Load model ----
    print(f"Loading CoTracker3 from {checkpoint} ...")
    model = CoTrackerOnlinePredictor(
        checkpoint=str(checkpoint), offline=False, window_len=16
    )
    model = model.to(torch_device).eval()

    # ---- Grid query points ----
    grid_size = max(1, min(grid_size, 32))
    grid_query_frame = max(0, min(grid_query_frame, total - 1))
    ish = model.interp_shape

    # ---- Streaming tracking loop ----
    WINDOW_LEN = 16
    STEP = 8

    window: list[np.ndarray] = []
    n = 0
    is_first_step = True

    def _process_step(first: bool):
        chunk = np.stack(window[-STEP * 2:])
        video_chunk = (
            torch.from_numpy(chunk)
            .float()
            .permute(0, 3, 1, 2)[None]
            .to(torch_device)
        )
        with torch.inference_mode():
            try:
                with torch.autocast(
                    device_type=torch_device.type, dtype=torch.float16
                ):
                    return model(
                        video_chunk,
                        is_first_step=first,
                        grid_size=grid_size,
                        grid_query_frame=grid_query_frame,
                    )
            except RuntimeError:
                # Fallback without AMP if model doesn't support fp16
                return model(
                    video_chunk,
                    is_first_step=first,
                    grid_size=grid_size,
                    grid_query_frame=grid_query_frame,
                )

    print(f"Tracking {total} frames at {tw}x{th} (grid {grid_size}x{grid_size}) ...")
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            # FIX: Convert BGR to RGB before tracking
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if scale < 1.0:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            if n % STEP == 0 and n != 0:
                _process_step(is_first_step)
                is_first_step = False
            window.append(frame)
            if len(window) > STEP * 2:
                window = window[-STEP * 2:]
            n += 1
            if n % 100 == 0:
                print(f"  read {n}/{total} frames ...", flush=True)
    finally:
        cap.release()

    if n < 2:
        raise SystemExit(f"Only {n} frame(s) readable from {video_path}")

    # Final step
    pred_tracks, pred_visibility = _process_step(is_first_step)
    if pred_tracks is None:
        pred_tracks, pred_visibility = _process_step(False)

    # ---- Map query points back to original coordinates ----
    query_pts = model.queries[0, :, 1:].detach().float().cpu().numpy()
    query_pts[:, 0] *= (orig_w - 1) / max(ish[1] - 1, 1)
    query_pts[:, 1] *= (orig_h - 1) / max(ish[0] - 1, 1)

    # ---- Map tracks back to original coordinates ----
    tracks = pred_tracks[0].detach().float().cpu().numpy()  # (T, N, 2)
    visibility = pred_visibility[0].detach().cpu().numpy()  # (T, N)
    T = tracks.shape[0]

    sx = (orig_w - 1) / max(tw - 1, 1)
    sy = (orig_h - 1) / max(th - 1, 1)
    tracks[:, :, 0] *= sx
    tracks[:, :, 1] *= sy

    # ---- Save ----
    meta = dict(
        width=orig_w,
        height=orig_h,
        fps=round(float(fps), 3),
        total_frames=total,
        processed_frames=T,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        num_points=tracks.shape[1],
        source=str(video_path.name),
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        tracks=tracks.astype(np.float32),
        visibility=visibility.astype(bool),
        query_points=query_pts.round(2).astype(np.float32),
        meta=json.dumps(meta),
    )
    print(f"Wrote {T} frames, {tracks.shape[1]} points -> {output_npz}")

    return dict(
        tracks=tracks,
        visibility=visibility,
        query_points=query_pts,
        width=orig_w,
        height=orig_h,
        fps=fps,
        total_frames=total,
        processed_frames=T,
        grid_size=grid_size,
        num_points=tracks.shape[1],
    )
