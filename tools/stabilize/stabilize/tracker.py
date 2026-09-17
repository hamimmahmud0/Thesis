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
from tqdm import tqdm


def _checkpoint_for_window(checkpoint: Path, window_len: int) -> Path:
    """Return a checkpoint whose ``time_emb`` matches *window_len*.

    The released online checkpoint ships with ``window_len=16``.  The model's
    ``time_emb`` is a deterministic sincos positional buffer (not a learned
    parameter), so when a different window is requested we resize it and cache
    a copy next to the original.  Returns the original path when no change is
    needed.
    """
    import torch
    import torch.nn.functional as F

    state = torch.load(checkpoint, map_location="cpu")
    model_sd = state.get("model", state)
    emb = model_sd.get("time_emb")
    if emb is None or emb.shape[1] == window_len:
        return checkpoint
    emb = (
        F.interpolate(
            emb.float().permute(0, 2, 1), size=window_len, mode="linear"
        )
        .permute(0, 2, 1)
        .to(emb.dtype)
    )
    model_sd["time_emb"] = emb
    out = checkpoint.with_name(f"{checkpoint.stem}_w{window_len}{checkpoint.suffix}")
    torch.save(state, out)
    return out


def track_video(
    video_path: str | Path,
    output_npz: str | Path,
    checkpoint: str | Path | None = None,
    grid_size: int = 128,
    grid_query_frame: int = 0,
    max_video_dim: int = 1280,
    step: int = 1,
    device: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """Track keypoint trajectories through *video_path* using CoTracker3.

    Parameters
    ----------
    video_path : path to the input video.
    output_npz : path to write the output .npz file.
    checkpoint : path to ``scaled_online.pth``.  If missing or None, it is
        auto-downloaded from the Hub (``facebook/cotracker3``) into the
        given path's parent, or into ``~/.cache/cotracker3/`` by default.
    grid_size : NxN grid of query points (no cap; GPU memory scales
        with N*N).
    grid_query_frame : frame index the grid is sampled from (default 0).
        Interpreted as an *absolute* frame index; inside a tracked range
        (``start_frame``..``end_frame``) it is converted to a local index.
    max_video_dim : resize longest side to this before tracking (GPU memory).
    step : online sliding-window stride in frames (default 1).  The CoTracker3
        online model processes ``window_len = 2 * step`` frames per call and
        advances by ``step``, so total compute is roughly constant (always 50%
        overlap).  Smaller steps give more frequent, smaller passes (lower
        peak memory); larger steps use larger windows.
    device : torch device string, e.g. ``"cuda:0"``.  Auto-detected if None.
    start_frame, end_frame : track only ``[start_frame, end_frame)``.  Used
        by the fresh-grid segment planner to run one CoTracker session per
        anchor.  ``end_frame=None`` means to the end of the video.

    Returns
    -------
    dict with keys: tracks, visibility, query_points, width, height, fps,
    total_frames, processed_frames, start_frame, end_frame, grid_size,
    num_points.
    """
    import torch
    from cotracker.models.core.model_utils import get_points_on_a_grid
    from cotracker.predictor import CoTrackerOnlinePredictor

    from .hfio import resolve_checkpoint

    video_path = Path(video_path)
    output_npz = Path(output_npz)
    checkpoint = resolve_checkpoint(checkpoint)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # ---- Probe video ----
    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    total_video = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()

    end = total_video if end_frame is None else min(int(end_frame), total_video)
    start = max(0, min(int(start_frame), end - 1))
    total = end - start
    if total < 2:
        raise SystemExit(
            f"Range [{start}, {end}) too short: {total} frame(s), need >= 2"
        )

    # Grid query frame as a LOCAL index inside [start, end).
    local_query = int(grid_query_frame) - start
    local_query = max(0, min(local_query, total - 1))

    # ---- Compute tracking resolution ----
    scale = min(1.0, max_video_dim / max(orig_w, orig_h))
    tw = max(1, int(round(orig_w * scale)) // 2 * 2)
    th = max(1, int(round(orig_h * scale)) // 2 * 2)

    # ---- Load model ----
    # The online model's internal stride is window_len // 2, so a custom step
    # requires rebuilding the model with window_len = 2 * step.
    step = max(1, int(step))
    window_len = 2 * step
    ckpt = _checkpoint_for_window(checkpoint, window_len)
    print(
        f"Loading CoTracker3 from {checkpoint} "
        f"(window {window_len}, step {step}) ..."
    )
    model = CoTrackerOnlinePredictor(
        checkpoint=str(ckpt), offline=False, window_len=window_len
    )
    model = model.to(torch_device).eval()

    # ---- Grid query points ----
    # Grid size is UNLOCKED (no cap); GPU memory scales with grid_size^2.
    grid_size = max(1, int(grid_size))
    grid_query_frame = local_query
    ish = model.interp_shape

    # ---- Streaming tracking loop ----
    WINDOW_LEN = window_len
    STEP = step

    window: list[np.ndarray] = []
    n = 0
    is_first_step = True

    def _process_step(first: bool):
        chunk = np.stack(window[-WINDOW_LEN:])
        video_chunk = (
            torch.from_numpy(chunk)
            .float()
            .permute(0, 3, 1, 2)[None]
            .to(torch_device)
        )
        kwargs = dict(
            is_first_step=first,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
        with torch.inference_mode():
            # AMP on CUDA only; autocast(fp16) on CPU hangs.
            if torch_device.type == "cuda":
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        return model(video_chunk, **kwargs)
                except RuntimeError:
                    pass
            return model(video_chunk, **kwargs)

    n_forwards = (total - 1) // STEP + 1
    print(
        f"Tracking {total} frames [{start},{end}) at {tw}x{th} "
        f"(grid {grid_size}x{grid_size}, query local {grid_query_frame}) ..."
    )
    print(
        f"  {n_forwards} forward passes "
        f"(stride {STEP}, window {WINDOW_LEN})"
    )
    cap = cv2.VideoCapture(str(video_path))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pbar = tqdm(
        total=n_forwards, desc="Tracking", unit="step",
        bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]",
    )
    try:
        while n < total:
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
                pbar.update(1)
                pbar.set_postfix_str(f"frame {n}/{total}")
            window.append(frame)
            if len(window) > WINDOW_LEN:
                window = window[-WINDOW_LEN:]
            n += 1

        if n < 2:
            raise SystemExit(f"Only {n} frame(s) readable from {video_path}")

        # Final step (produces the tracks)
        pred_tracks, pred_visibility = _process_step(is_first_step)
        if pred_tracks is None:
            pred_tracks, pred_visibility = _process_step(False)
        pbar.update(1)
        pbar.set_postfix_str(f"done {n} frames")
    finally:
        pbar.close()
        cap.release()

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
        start_frame=start,
        end_frame=end,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        step=step,
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
        start_frame=start,
        end_frame=end,
        grid_size=grid_size,
        step=step,
        num_points=tracks.shape[1],
    )


def make_segment_tracker(
    video_path: str | Path,
    checkpoint: str | Path | None = None,
    grid_size: int = 128,
    max_video_dim: int = 1280,
    step: int = 1,
    device: str | None = None,
    segment_dir: str | Path | None = None,
):
    """Return ``track_fn(anchor, end) -> TrackingSegment`` for the planner.

    Each call runs an **independent** CoTracker online session over
    ``[anchor, end)`` with a brand-new grid queried at ``anchor``.  Segment
    tracks are written to ``segment_dir/segment_<anchor>.npz`` when a
    directory is given, giving the persisted per-segment representation
    required for long-video debugging.
    """
    import tempfile
    from pathlib import Path as _Path

    from .segments import make_segment

    video_path = _Path(video_path)
    if segment_dir is not None:
        segment_dir = _Path(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)
    _tmp = None if segment_dir is not None else tempfile.mkdtemp(prefix="segs_")

    def track_fn(anchor_frame: int, end_frame: int):
        out_npz = _Path(segment_dir or _tmp) / f"segment_{int(anchor_frame):08d}.npz"
        res = track_video(
            video_path=video_path,
            output_npz=out_npz,
            checkpoint=checkpoint,
            grid_size=grid_size,
            grid_query_frame=int(anchor_frame),
            max_video_dim=max_video_dim,
            step=step,
            device=device,
            start_frame=int(anchor_frame),
            end_frame=int(end_frame),
        )
        return make_segment(
            segment_id=0,
            anchor_frame=int(anchor_frame),
            start_frame=int(res["start_frame"]),
            end_frame=int(res["end_frame"]),
            tracks=res["tracks"],
            visibility=res["visibility"],
            query_points=res["query_points"],
            width=int(res["width"]),
            height=int(res["height"]),
            fps=float(res["fps"]),
        )

    return track_fn
