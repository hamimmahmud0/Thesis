
def track_video(
    video_path: str | Path,
    output_npz: str | Path,
    checkpoint: str | Path | None = None,
    grid_size: int = 16,
    grid_query_frame: int = 0,
    max_video_dim: int = 1280,
    step: int = 8,
    device: str | None = None,
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
    max_video_dim : resize longest side to this before tracking (GPU memory).
    step : online sliding-window stride in frames (default 8).  The CoTracker3
        online model processes ``window_len = 2 * step`` frames per call and
        advances by ``step``, so total compute is roughly constant (always 50%
        overlap).  Smaller steps give more frequent, smaller passes (lower
        peak memory); larger steps use larger windows.
    device : torch device string, e.g. ``"cuda:0"``.  Auto-detected if None.

    Returns
    -------
    dict with keys: tracks, visibility, query_points, width, height, fps,
    total_frames, processed_frames, grid_size, num_points.
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
    grid_size = max(16, int(grid_size))
    grid_query_frame = max(0, min(grid_query_frame, total - 1))
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
    print(f"Tracking {total} frames at {tw}x{th} (grid {grid_size}x{grid_size}) ...")
    print(
        f"  {n_forwards} forward passes "
        f"(stride {STEP}, window {WINDOW_LEN})"
    )
    cap = cv2.VideoCapture(str(video_path))
    pbar = tqdm(
        total=n_forwards, desc="Tracking", unit="step",
        bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]",
    )
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
        grid_size=grid_size,
        step=step,
        num_points=tracks.shape[1],
    )
