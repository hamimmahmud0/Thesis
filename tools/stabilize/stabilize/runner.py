"""End-to-end pipeline runner.

Orchestrates a full stabilisation run:

    download video -> track -> estimate motion -> smooth -> render -> upload

All intermediate products land in a local run directory
(``<out_dir>/<run_name>/``) and are uploaded to an HF bucket under
``hf://buckets/<bucket>/<run_name>/``.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from . import hfio


def _steps_desc(use_smoothing: bool) -> str:
    base = ["download", "track", "estimate"]
    if use_smoothing:
        base.append("smooth")
    base += ["render", "upload"]
    return " -> ".join(base)


def run_pipeline(
    *,
    source: str,
    bucket: str | None,
    run_name: str,
    token: str | None = None,
    private: bool = False,
    out_dir: str | Path = "runs",
    checkpoint: str | None = None,
    tracks_npz: str | None = None,
    grid_size: int = 16,
    grid_query_frame: int = 0,
    max_video_dim: int = 1280,
    crop_width: int = 1024,
    crop_height: int | None = None,
    shift_x: int = 0,
    shift_y: int = 0,
    sigma: float = 10.0,
    interp_gap: int = 5,
    use_smoothing: bool = True,
    n_frames: int = 0,
    crf: int = 20,
    skip_upload: bool = False,
    device: str | None = None,
) -> dict:
    """Run the full download->track->estimate->smooth->render->upload pipeline.

    Parameters
    ----------
    source : ``hf://`` link OR a local path to a video file.
    bucket : HF bucket id (``user/bucket`` or ``bucket``).  Created if it
        does not exist.  Required unless ``skip_upload`` is True.
    run_name : sub-folder name under which outputs are uploaded.
    token : HF token for bucket access.  Defaults to ``HF_TOKEN`` env var.
    out_dir : local parent directory for run folders.
    checkpoint : CoTracker3 checkpoint path; enables local tracking.  If
        None, ``tracks_npz`` must be provided.
    tracks_npz : pre-computed tracks to use instead of tracking.
    skip_upload : do everything locally but skip the upload step.

    Returns
    -------
    dict of run metadata and output paths (also written to summary.json).
    """
    out_dir = Path(out_dir)
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if token is None:
        token = _env_token()

    if not checkpoint and not tracks_npz:
        raise ValueError(
            "Either --checkpoint (to track locally) or --tracks (pre-computed "
            "tracks) is required."
        )
    if bucket is None and not skip_upload:
        raise ValueError("--bucket is required unless --skip-upload is used.")

    if crop_height is None:
        crop_height = crop_width

    timing = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    step_times: dict[str, float] = {}

    def _tick(name: str) -> None:
        _t0 = time.time()

        def _done():
            step_times[name] = time.time() - _t0
            print(f"[{name}] {step_times[name]:.1f}s", flush=True)

        return _done

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    video_path: Path | None = None
    if hfio.is_local_path(source):
        video_path = Path(source)
        if not video_path.is_file():
            raise SystemExit(f"Local video not found: {source}")
        print(f"[download] using local file {video_path}")
    else:
        done = _tick("download")
        print(f"[download] {source} -> {run_dir}")
        video_path = hfio.download_link(source, run_dir, token=token)
        done()

    # ------------------------------------------------------------------
    # 2. Track
    # ------------------------------------------------------------------
    if checkpoint:
        from .tracker import track_video

        done = _tick("track")
        tracks = run_dir / "tracks.npz"
        print(f"[track] {video_path} (grid {grid_size}x{grid_size})")
        track_video(
            video_path=video_path,
            output_npz=tracks,
            checkpoint=checkpoint,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            max_video_dim=max_video_dim,
            device=device,
        )
        done()
    else:
        tracks = Path(tracks_npz)
        if not tracks.is_file():
            raise SystemExit(f"Tracks file not found: {tracks}")
        # Copy into the run dir so everything lives in one place.
        run_tracks = run_dir / tracks.name
        shutil.copy2(tracks, run_tracks)
        tracks = run_tracks
        print(f"[track] using pre-computed tracks {tracks.name}")

    # ------------------------------------------------------------------
    # 3. Estimate
    # ------------------------------------------------------------------
    from . import motion as motion_mod

    done = _tick("estimate")
    data = motion_mod.load_tracks(tracks)
    width = data["width"] or 3840
    height = data["height"] or 2160
    fps = data["fps"] or 23.976

    result = motion_mod.estimate_motion(
        data["tracks"], data["visibility"], width, height
    )
    motion_prefix = run_dir / "motion"
    motion_mod.save_motion(
        result,
        motion_prefix,
        video_width=width,
        video_height=height,
        fps=fps,
        tracks_source=tracks.name,
    )
    done()
    print(motion_mod.motion_summary(result))

    # ------------------------------------------------------------------
    # 4. Smooth
    # ------------------------------------------------------------------
    smooth_npz = None
    if use_smoothing:
        from .smoother import smooth_motion

        done = _tick("smooth")
        smooth_npz = run_dir / "motion_smooth.npz"
        smooth_motion(motion_prefix.with_suffix(".npz"), smooth_npz, sigma=sigma)
        done()

    # ------------------------------------------------------------------
    # 5. Visualisation
    # ------------------------------------------------------------------
    done = _tick("viz")
    from . import viz as viz_mod

    viz_mod.viz_from_npz(
        motion_prefix.with_suffix(".npz"),
        out_png=run_dir / "trajectory.png",
        video_name=video_path.name,
    )
    done()

    # ------------------------------------------------------------------
    # 6. Render
    # ------------------------------------------------------------------
    from .renderer import render

    stable = run_dir / "stabilized.mp4"
    done = _tick("render")
    print(f"[render] crop {crop_width}x{crop_height}, smoothing={'on' if use_smoothing else 'off'}")
    render(
        video_path=video_path,
        output_path=stable,
        motion_npz=motion_prefix.with_suffix(".npz"),
        smooth_npz=smooth_npz,
        crop_width=crop_width,
        crop_height=crop_height,
        shift_x=shift_x,
        shift_y=shift_y,
        n_frames=n_frames,
        crf=crf,
    )
    done()

    # ------------------------------------------------------------------
    # 7. Summary + upload
    # ------------------------------------------------------------------
    outputs = {
        "video": str(video_path),
        "tracks": str(tracks),
        "motion_npz": str(motion_prefix.with_suffix(".npz")),
        "motion_csv": str(motion_prefix.with_suffix(".csv")),
        "smooth_npz": str(smooth_npz) if smooth_npz else None,
        "trajectory_png": str(run_dir / "trajectory.png"),
        "stabilized_mp4": str(stable),
    }

    summary = {
        "run_name": run_name,
        "source": source,
        "bucket": bucket,
        "timestamp": timing["started_at"],
        "params": {
            "grid_size": grid_size,
            "grid_query_frame": grid_query_frame,
            "max_video_dim": max_video_dim,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "shift_x": shift_x,
            "shift_y": shift_y,
            "sigma": sigma if use_smoothing else None,
            "smoothing": use_smoothing,
            "frames": n_frames,
            "crf": crf,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "outputs": outputs,
        "step_times_s": step_times,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n===== RUN OUTPUTS =====")
    for key, val in outputs.items():
        if val:
            size = Path(val).stat().st_size if Path(val).is_file() else 0
            print(f"  {key}: {val} ({size/1e6:.1f} MB)")
    print(f"  summary: {summary_path}")

    uploaded = []
    if not skip_upload:
        done = _tick("upload")
        print(f"[upload] ensuring bucket {bucket} ...")
        bucket_id = hfio.ensure_bucket(bucket, token=token, private=private)
        summary["bucket"] = bucket_id
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[upload] upload {run_dir} -> hf://buckets/{bucket_id}/{run_name}/")
        uploaded = hfio.upload_dir(run_dir, bucket_id, run_name, token=token)
        done()
        print("\n===== UPLOADED =====")
        for uri in uploaded:
            print(f"  hf://buckets/{bucket_id}/{uri}")
    else:
        print("\n[skip-upload] not uploading (--skip-upload)")

    summary["uploaded"] = uploaded
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n===== DONE =====")
    print(f"local run dir: {run_dir}")
    print(f"pipeline:      {_steps_desc(use_smoothing)}")
    return summary


def _env_token() -> str | None:
    return None if "HF_TOKEN" not in __import__("os").environ else __import__("os").environ["HF_TOKEN"]