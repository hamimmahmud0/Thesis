"""End-to-end pipeline runner.

Orchestrates a full stabilisation run:

    download video -> track -> estimate motion -> smooth -> render -> upload

All intermediate products land in a local run directory
(``<out_dir>/<run_name>/``) and are uploaded to an HF bucket under
``hf://buckets/<bucket>/<run_name>/``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import hfio


def _steps_desc(mode: str, skip_overlay: bool = False) -> str:
    base = ["download", "track", "estimate", "drift"]
    if mode != "off":
        base.append(f"smooth[{mode}]")
    base += ["viz"]
    if not skip_overlay:
        base.append("overlay")
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
    step: int = 8,
    crop_width: int = 1024,
    crop_height: int | None = None,
    shift_x: int = 0,
    shift_y: int = 0,
    sigma: float = 10.0,
    interp_gap: int = 5,
    use_smoothing: bool = False,
    stabilization_mode: str = "off",
    anchor_interval_seconds: float = 10.0,
    ransac_reproj_threshold: float = 2.0,
    min_correspondences: int = 20,
    min_inlier_ratio: float = 0.4,
    anchor_blend_frames: int = 0,
    locked_poly_degree: int = 3,
    save_drift: bool = True,
    debug: bool = False,
    n_frames: int = 0,
    crf: int = 18,
    skip_upload: bool = False,
    device: str | None = None,
    no_crop: bool = False,
    crop_black_border: bool = False,
    skip_overlay: bool = False,
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
        None (and ``tracks_npz`` is also None), the default cached
        checkpoint is used and auto-downloaded from the Hub if missing.
    tracks_npz : pre-computed tracks to use instead of tracking.
    no_crop : render the full source frame (WxH) instead of cropping.
        Black borders may appear.  Overrides ``crop_width``/``crop_height``.
    crop_black_border : auto-detect the largest centred crop that removes
        the black borders introduced by stabilisation.  Mutually exclusive
        with ``no_crop``.
    skip_upload : do everything locally but skip the upload step.
    skip_overlay : skip generating the tracks-overlay video.

    Returns
    -------
    dict of run metadata and output paths (also written to summary.json).
    """
    out_dir = Path(out_dir)
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if token is None:
        token = _env_token()

    if not tracks_npz and not checkpoint:
        # Neither provided: resolve the default cached checkpoint,
        # downloading it from the Hub on first use.
        checkpoint = hfio.resolve_checkpoint(None, token=token)
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
            step=step,
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
    # 3. Estimate (anchor-relative, drift-free)
    # ------------------------------------------------------------------
    from . import motion as motion_mod

    done = _tick("estimate")
    data = motion_mod.load_tracks(tracks)
    width = data["width"] or 3840
    height = data["height"] or 2160
    fps = data["fps"] or 23.976

    result = motion_mod.estimate_motion(
        data["tracks"], data["visibility"], width, height,
        anchor_interval_seconds=anchor_interval_seconds,
        fps=fps,
        ransac_reproj_threshold=ransac_reproj_threshold,
        min_correspondences=min_correspondences,
        min_inlier_ratio=min_inlier_ratio,
        anchor_blend_frames=anchor_blend_frames,
        debug=debug,
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

    # ---- Drift diagnostic: isolate tracker drift from integration drift ----
    drift_npz = None
    if save_drift:
        done = _tick("drift")
        drift_res = motion_mod.drift_diagnostic(
            data["tracks"], data["visibility"], anchor=0, width=width, height=height,
        )
        seg_res = motion_mod.segment_drift_diagnostic(
            data["tracks"], data["visibility"], result["anchor_frames"],
            width=width, height=height,
        )
        drift_res["segment_absolute_dx"] = seg_res["absolute_dx"]
        drift_res["segment_absolute_dy"] = seg_res["absolute_dy"]
        drift_npz = run_dir / "motion_drift"
        motion_mod.save_drift(
            drift_res, drift_npz, fps=fps, anchor_frames=result["anchor_frames"],
        )
        done()

    # ------------------------------------------------------------------
    # 4. Smooth
    # ------------------------------------------------------------------
    # ``off`` => track-locked (no smoothing).  The legacy ``use_smoothing``
    # flag maps onto the ``natural`` mode for backward compatibility.
    mode = str(stabilization_mode).lower().strip()
    if use_smoothing and mode == "off":
        mode = "natural"
    if mode not in ("natural", "locked", "off"):
        raise ValueError(
            f"stabilization_mode must be natural|locked|off, got {mode!r}"
        )

    smooth_npz = None
    if mode != "off":
        from .smoother import smooth_motion

        done = _tick("smooth")
        smooth_npz = run_dir / "motion_smooth.npz"
        smooth_res = smooth_motion(
            motion_prefix.with_suffix(".npz"),
            smooth_npz,
            sigma=sigma,
            interp_gap=interp_gap,
            mode=mode,
            locked_poly_degree=locked_poly_degree,
        )
        tr = smooth_res.get("transitions", {})
        print(
            f"[smooth] mode={mode} boundary jumps: "
            f"translation={tr.get('max_translation_jump_px', 0):.3f}px "
            f"rotation={tr.get('max_rotation_jump_deg', 0):.3f}deg "
            f"scale={tr.get('max_scale_jump', 0):.4f} "
            f"suspicious={tr.get('suspicious', False)}"
        )
        st = smooth_res.get("stats_ref", {})
        if st:
            print(
                f"[smooth] reference net translation={st['net_translation_px']:.2f}px "
                f"max={st['max_translation_px']:.2f}px"
            )
        done()
    else:
        print("[smooth] stabilization_mode=off -> track-locked (no smoothing)")

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
    # 6. Track overlay
    # ------------------------------------------------------------------
    overlay = None
    if not skip_overlay:
        done = _tick("overlay")
        overlay = run_dir / "tracks_overlay.mp4"
        viz_mod.overlay_tracks(
            video_path=video_path,
            tracks_npz=tracks,
            output_path=overlay,
        )
        done()
    else:
        print("[skip-overlay] skipping tracks-overlay video (--skip-overlay)")

    # ------------------------------------------------------------------
    # 7. Render
    # ------------------------------------------------------------------
    from .renderer import render

    stable = run_dir / "stabilized.mp4"
    done = _tick("render")
    crop_desc = (
        "no-crop (full frame)" if no_crop else
        "crop-black-border (auto)" if crop_black_border else
        f"crop {crop_width}x{crop_height}"
    )
    print(f"[render] {crop_desc}, smoothing={'on' if use_smoothing else 'off'}")
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
        no_crop=no_crop,
        crop_black_border=crop_black_border,
    )
    done()

    # ------------------------------------------------------------------
    # 8. Summary + upload
    # ------------------------------------------------------------------
    outputs = {
        "video": str(video_path),
        "tracks": str(tracks),
        "motion_npz": str(motion_prefix.with_suffix(".npz")),
        "motion_csv": str(motion_prefix.with_suffix(".csv")),
        "drift_npz": str(drift_npz.with_suffix(".npz")) if drift_npz else None,
        "drift_csv": str(drift_npz.with_suffix(".csv")) if drift_npz else None,
        "smooth_npz": str(smooth_npz) if smooth_npz else None,
        "trajectory_png": str(run_dir / "trajectory.png"),
        "tracks_overlay_mp4": str(overlay) if overlay else None,
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
            "step": step,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "crop_mode": (
                "no_crop" if no_crop else
                "crop_black_border" if crop_black_border else "fixed"
            ),
            "shift_x": shift_x,
            "shift_y": shift_y,
            "sigma": sigma if mode != "off" else None,
            "smoothing": mode != "off",
            "stabilization_mode": mode,
            "anchor_interval_seconds": anchor_interval_seconds,
            "ransac_reproj_threshold": ransac_reproj_threshold,
            "min_correspondences": min_correspondences,
            "min_inlier_ratio": min_inlier_ratio,
            "anchor_blend_frames": anchor_blend_frames,
            "locked_poly_degree": locked_poly_degree,
            "motion_method": result.get("method", "anchor_relative"),
            "overlay": not skip_overlay,
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
    print(f"pipeline:      {_steps_desc(mode, skip_overlay)}")
    return summary


def _env_token() -> str | None:
    return os.environ.get("HF_TOKEN")