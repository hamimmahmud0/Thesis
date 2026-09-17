"""Command-line interface for drone video stabilisation.

Nine subcommands cover both the low-level pipeline and automated
Hugging-Face-integrated runs:

  run       Download, stabilise, and upload in one shot.
  download  Fetch a video from an hf:// link.
  upload    Upload local files to an HF bucket.
  track     Extract keypoint trajectories from a video using CoTracker3.
  estimate  Estimate per-frame camera motion from tracked keypoints.
  smooth    Gaussian-smooth the estimated camera path.
  render    Warp the source video to produce a stabilised output.
  viz       Generate a diagnostic trajectory plot.
  overlay   Visualise tracked keypoints on the source video.

Automated end-to-end workflow::

  stabilize run hf://datasets/user/repo/video.mp4 \\
      --token hf_xxx --bucket my-stabilized --run DJI_0260 \\
      --checkpoint ./scaled_online.pth \\
      --crop-width 1920 --crop-height 1080

Step-by-step workflow::

  stabilize track     input.mp4 -o tracks.npz --checkpoint ./scaled_online.pth
  stabilize estimate  tracks.npz -o motion
  stabilize smooth    motion.npz -o motion_smooth.npz --sigma 12
  stabilize plan      input.mp4 --tracks tracks.npz --motion motion.npz --crop 1024
  stabilize render    input.mp4 --tracks tracks.npz --motion motion.npz \\
                           --crop 1024 --output stable.mp4 --confirm "approved"
  stabilize viz       motion.npz -o trajectory.png
  stabilize overlay   input.mp4 --tracks tracks.npz -o tracks_overlay.mp4

Run ``stabilize <subcommand> --help`` for per-command details.
"""

from __future__ import annotations

import argparse
import sys


def _effective_crop(args: argparse.Namespace) -> tuple[bool, bool, int | None, int | None]:
    """Resolve crop flags into ``(no_crop, crop_black_border, width, height)``.

    Priority: ``--no-crop`` wins over everything; an explicit
    ``--crop-black-border`` selects the automatic mode.  Otherwise any
    explicit ``--crop-width``/``--crop-height`` selects fixed cropped
    output.  With none of these, the default is the automatic
    black-border crop.
    """
    if args.no_crop:
        return True, False, None, None
    if args.crop_black_border or (args.crop_width is None and args.crop_height is None):
        return False, True, None, None
    crop_w = args.crop_width if args.crop_width is not None else (args.crop_height or 1024)
    crop_h = args.crop_height if args.crop_height is not None else crop_w
    return False, False, crop_w, crop_h


def _add_common_render_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared by ``plan`` and ``render``."""
    p.add_argument(
        "video",
        help="Path to the source drone video file (MP4, MOV, etc.).",
    )
    p.add_argument(
        "--tracks", required=True, metavar="TRACKS_NPZ",
        help=(
            "Path to the tracks .npz file produced by ``stabilize track`` "
            "(or the CoTracker3 API).  Must contain keys 'tracks' [T,N,2], "
            "'visibility' [T,N], and 'meta' with width/height/fps."
        ),
    )
    p.add_argument(
        "--motion", required=True, metavar="MOTION_NPZ",
        help=(
            "Path to the motion .npz file produced by ``stabilize estimate``. "
            "Contains pairwise and cumulative 3x3 homogeneous transforms."
        ),
    )
    p.add_argument(
        "--smooth", default=None, metavar="SMOOTH_NPZ",
        help=(
            "Optional smoothed-motion .npz from ``stabilize smooth``. "
            "When provided, the renderer uses smoothed-path stabilisation "
            "(difference between raw and smoothed camera paths) instead of "
            "simple track-locking.  Produces more natural-looking motion."
        ),
    )
    p.add_argument(
        "--output", "-o", required=True, metavar="OUTPUT.mp4",
        help="Path for the output stabilised video file.",
    )
    p.add_argument(
        "--crop-width", type=int, default=None, metavar="PX",
        help=(
            "Width of a fixed-size crop window in pixels.  When neither "
            "--crop-width nor --crop-height is given, the automatic "
            "--crop-black-border mode is used by default.  If only this is "
            "given, --crop-height defaults to the same value (square).  "
            "Default: auto black-border crop."
        ),
    )
    p.add_argument(
        "--crop-height", type=int, default=None, metavar="PX",
        help=(
            "Height of a fixed-size crop window in pixels.  If omitted "
            "but --crop-width is given, defaults to --crop-width (square "
            "output).  Use --crop-height 1080 --crop-width 1920 for 16:9 "
            "output.  Default: auto black-border crop."
        ),
    )
    p.add_argument(
        "--shift-x", type=int, default=0, metavar="PX",
        help=(
            "Shift the crop window centre right by PX pixels (in frame-0 "
            "coordinates).  Positive values shift right.  Default: 0 (centred)."
        ),
    )
    p.add_argument(
        "--shift-y", type=int, default=0, metavar="PX",
        help=(
            "Shift the crop window centre down by PX pixels (in frame-0 "
            "coordinates).  Positive values shift down.  Default: 0 (centred)."
        ),
    )
    crop_mode = p.add_mutually_exclusive_group()
    crop_mode.add_argument(
        "--no-crop", action="store_true",
        help=(
            "Do not crop the output: stabilise the full source frame (WxH). "
            "Black borders may appear where content moved out of view."
        ),
    )
    crop_mode.add_argument(
        "--crop-black-border", action="store_true",
        help=(
            "Auto-detect the largest centred crop window that removes the "
            "black borders introduced by stabilisation.  This is the default "
            "mode; passing the flag is optional.  Overrides "
            "--crop-width/--crop-height."
        ),
    )
    p.add_argument(
        "--frames", type=int, default=0, metavar="N",
        help=(
            "Render only the first N frames.  0 (default) renders all "
            "frames.  Use a small value for previewing."
        ),
    )
    p.add_argument(
        "--fps", type=float, default=None, metavar="FPS",
        help=(
            "Override the output video frame rate.  If omitted, the source "
            "video FPS is used."
        ),
    )
    p.add_argument(
        "--crf", type=int, default=18, metavar="N",
        help=(
            "Output quality: x264 CRF, lower = better.  Default: 18 "
            "(visually lossless).  0 = mathematically lossless (very large)."
        ),
    )


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand — full end-to-end pipeline."""
    from .runner import run_pipeline

    no_crop, crop_black_border, crop_width, crop_height = _effective_crop(args)

    run_pipeline(
        source=args.video,
        bucket=args.bucket,
        run_name=args.run_name,
        token=args.token,
        private=args.private,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        tracks_npz=args.tracks,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        max_video_dim=args.max_dim,
        step=args.step,
        crop_width=crop_width,
        crop_height=crop_height,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        sigma=args.sigma,
        interp_gap=args.interp_gap,
        use_smoothing=args.smooth,
        stabilization_mode=args.stabilization_mode,
        anchor_interval_seconds=args.anchor_interval_seconds,
        min_anchor_interval_seconds=args.min_anchor_interval_seconds,
        anchor_overlap_seconds=args.anchor_overlap_seconds,
        min_remaining_point_ratio=args.min_remaining_point_ratio,
        min_inlier_count=args.min_inlier_count,
        min_spatial_coverage=args.min_spatial_coverage,
        coverage_grid_rows=args.coverage_grid_rows,
        coverage_grid_cols=args.coverage_grid_cols,
        quality_failure_patience_frames=args.quality_failure_patience_frames,
        fresh_grids=not args.no_fresh_grids,
        ransac_reproj_threshold=args.ransac_reproj_threshold,
        min_correspondences=args.min_correspondences,
        min_inlier_ratio=args.min_inlier_ratio,
        anchor_blend_frames=args.anchor_blend_frames,
        locked_poly_degree=args.locked_poly_degree,
        save_drift=not args.no_drift,
        debug=args.debug,
        n_frames=args.frames,
        crf=args.crf,
        skip_upload=args.skip_upload,
        device=args.device,
        no_crop=no_crop,
        crop_black_border=crop_black_border,
        skip_overlay=args.skip_overlay,
    )


def _cmd_download(args: argparse.Namespace) -> None:
    """Execute the ``download`` subcommand."""
    from pathlib import Path

    from .hfio import download_link

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = download_link(args.link, out, token=args.token)
    print(f"Downloaded: {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def _cmd_upload(args: argparse.Namespace) -> None:
    """Execute the ``upload`` subcommand."""
    from pathlib import Path

    from .hfio import ensure_bucket, upload_dir

    src = Path(args.local_dir)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory")
        sys.exit(1)

    bucket_id = ensure_bucket(args.bucket, token=args.token, private=args.private)
    remote = args.remote_prefix.rstrip("/")

    print(f"Uploading {src} -> hf://buckets/{bucket_id}/{remote}/")
    uploaded = upload_dir(src, bucket_id, remote, token=args.token)
    for uri in uploaded:
        print(f"  hf://buckets/{bucket_id}/{uri}")
    print(f"Done — {len(uploaded)} files.")


def _cmd_track(args: argparse.Namespace) -> None:
    """Execute the ``track`` subcommand."""
    from .tracker import track_video

    track_video(
        video_path=args.video,
        output_npz=args.output,
        checkpoint=args.checkpoint,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        max_video_dim=args.max_dim,
        step=args.step,
        device=args.device,
    )


def _cmd_estimate(args: argparse.Namespace) -> None:
    """Execute the ``estimate`` subcommand."""
    from .motion import (
        estimate_motion,
        estimate_motion_pairwise,
        load_tracks,
        save_drift,
        save_motion,
    )

    data = load_tracks(args.tracks)
    tracks = data["tracks"]
    visibility = data["visibility"]
    T, N, _ = tracks.shape

    width = args.width or data["width"] or 3840
    height = args.height or data["height"] or 2160
    fps = args.fps or data["fps"] or 23.976

    print(f"Tracks:     {T} frames, {N} points, {width}x{height} @ {fps:.3f} fps")

    if T < 2:
        print("T < 2: no frame pairs to estimate.")
        sys.exit(0)

    if args.legacy:
        print("Estimator:  legacy frame-to-frame accumulation (drift-prone, A/B only)")
        result = estimate_motion_pairwise(
            tracks, visibility, width, height, max_points=args.max_points,
        )
    else:
        print(
            f"Estimator:  anchor-relative "
            f"(interval {args.anchor_interval_seconds:g}s, "
            f"min_corr {args.min_correspondences}, "
            f"min_ratio {args.min_inlier_ratio})"
        )
        result = estimate_motion(
            tracks, visibility, width, height,
            max_points=args.max_points,
            anchor_interval_seconds=args.anchor_interval_seconds,
            fps=fps,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            min_correspondences=args.min_correspondences,
            min_inlier_ratio=args.min_inlier_ratio,
            anchor_blend_frames=args.anchor_blend_frames,
            debug=args.debug,
        )

    save_motion(
        result,
        args.output,
        video_width=width,
        video_height=height,
        fps=fps,
        tracks_source=str(args.tracks),
    )

    from .motion import drift_diagnostic, motion_summary, segment_drift_diagnostic

    print()
    print("===== SUMMARY =====")
    print(motion_summary(result))

    if args.drift:
        import math

        drift = drift_diagnostic(tracks, visibility, anchor=0, width=width, height=height)
        seg = segment_drift_diagnostic(
            tracks, visibility, result["anchor_frames"], width=width, height=height,
        )
        drift["segment_absolute_dx"] = seg["absolute_dx"]
        drift["segment_absolute_dy"] = seg["absolute_dy"]
        save_drift(
            drift, str(args.output) + "_drift",
            fps=fps, anchor_frames=result["anchor_frames"],
        )
        net = float(
            math.hypot(drift["absolute_dx"][-1], drift["absolute_dy"][-1])
        )
        print(f"Tracker drift (anchor 0, start->end): {net:.1f} px")


def _cmd_fresh_grid(args: argparse.Namespace) -> None:
    """Execute the ``fresh-grid`` subcommand (segmented tracking + estimate)."""
    from .segments import (
        SegmentConfig,
        plan_segments,
        save_segment_metadata,
        segment_report,
    )
    from .tracker import make_segment_tracker
    from .utils import open_video

    from .motion import motion_summary, save_motion

    from pathlib import Path

    cap, n_vid, width, height, fps = open_video(args.video)
    cap.release()

    cfg = SegmentConfig(
        anchor_interval_seconds=args.anchor_interval_seconds,
        min_anchor_interval_seconds=args.min_anchor_interval_seconds,
        anchor_overlap_seconds=args.anchor_overlap_seconds,
        min_remaining_point_ratio=args.min_remaining_point_ratio,
        min_inlier_ratio=args.min_inlier_ratio,
        min_inlier_count=args.min_inlier_count,
        min_spatial_coverage=args.min_spatial_coverage,
        coverage_grid_rows=args.coverage_grid_rows,
        coverage_grid_cols=args.coverage_grid_cols,
        quality_failure_patience_frames=args.quality_failure_patience_frames,
        ransac_reproj_threshold=args.ransac_reproj_threshold,
        min_correspondences=args.min_correspondences,
    )

    segment_dir = Path(args.segment_dir) if args.segment_dir else (
        Path(args.output).parent / (Path(args.output).name + "_segments")
    )

    print(
        f"Video:  {n_vid} frames, {width}x{height} @ {fps:.3f} fps\n"
        f"Fresh-grid: interval {args.anchor_interval_seconds}s "
        f"(min {args.min_anchor_interval_seconds}s), overlap "
        f"{args.anchor_overlap_seconds}s, grid {args.grid_size}x{args.grid_size}"
    )

    track_fn = make_segment_tracker(
        args.video,
        checkpoint=args.checkpoint,
        grid_size=args.grid_size,
        max_video_dim=args.max_dim,
        step=args.step,
        device=args.device,
        segment_dir=segment_dir,
    )
    result = plan_segments(
        track_fn, n_vid, width, height, fps, cfg, debug=args.debug,
    )

    save_motion(
        result,
        args.output,
        video_width=width,
        video_height=height,
        fps=fps,
        tracks_source=f"fresh-grid({len(result['segments'])} segments)",
    )
    save_segment_metadata(result, args.output)

    print()
    print("===== SUMMARY =====")
    print(motion_summary(result))
    print()
    print("===== FRESH-GRID =====")
    print(segment_report(result))
    for st in result.get("anchor_stats", []):
        print(
            f"  anchor {st['anchor_frame']:6d} reason={st['reason']:<20s} "
            f"age={st['age_frames']:4d} "
            f"surviving={st['surviving_tracks']:4d}/{st['num_query_points']} "
            f"inliers={st['ransac_inliers']:4d} "
            f"cov={st['spatial_coverage']:.2f} "
            f"bridge_inliers={st['bridge_inliers']:3d} "
            f"degraded={st['bridge_degraded']}"
        )


def _cmd_smooth(args: argparse.Namespace) -> None:
    """Execute the ``smooth`` subcommand."""
    from .smoother import smooth_motion

    res = smooth_motion(
        motion_npz=args.motion,
        output_npz=args.output,
        sigma=args.sigma,
        interp_gap=args.interp_gap,
        mode=args.mode,
        locked_poly_degree=args.locked_poly_degree,
    )

    tr = res.get("transitions", {})
    print(
        f"Anchor boundary jumps: translation={tr.get('max_translation_jump_px', 0):.3f}px "
        f"rotation={tr.get('max_rotation_jump_deg', 0):.3f}deg "
        f"scale={tr.get('max_scale_jump', 0):.4f} "
        f"suspicious={tr.get('suspicious', False)}"
    )
    st = res.get("stats_ref", {})
    if st:
        print(
            f"Stabilized reference: start={st['start']} end={st['end']} "
            f"net={st['net_translation_px']:.2f}px max={st['max_translation_px']:.2f}px"
        )


def _cmd_plan(args: argparse.Namespace) -> None:
    """Execute the ``plan`` subcommand."""
    from .motion import load_motion
    from .renderer import _border_check, _black_border_crop
    from .utils import open_video, human_time

    cap, n_vid, W, H, fps = open_video(args.video)
    if args.fps:
        fps = args.fps
    cap.release()

    motion = load_motion(args.motion)
    raw_cum = motion["cumulative"]
    T_csv = len(raw_cum)

    smooth_cum = None
    if args.smooth:
        from .smoother import load_smoothed

        smooth_cum = load_smoothed(args.smooth)["smoothed_cumulative"]

    n_render = min(n_vid, T_csv, args.frames) if args.frames > 0 else min(n_vid, T_csv)

    no_crop, crop_black_border, crop_w, crop_h = _effective_crop(args)

    # ---- Determine the output size and crop-window origin ----
    if no_crop:
        out_w, out_h = W, H
        x0 = y0 = 0
        crop_desc = f"full frame {W}x{H} (no-crop)"
    elif crop_black_border:
        x0, y0, out_w, out_h = _black_border_crop(
            raw_cum, smooth_cum, n_render, W, H, args.shift_x, args.shift_y
        )
        crop_desc = f"auto black-border {out_w}x{out_h}"
    else:
        out_w, out_h = crop_w, crop_h
        if out_w > W or out_h > H:
            print(f"ERROR: crop {out_w}x{out_h} exceeds video {W}x{H}")
            sys.exit(1)
        x0 = (W - out_w) // 2 + args.shift_x
        y0 = (H - out_h) // 2 + args.shift_y
        if x0 < 0 or y0 < 0 or x0 + out_w > W or y0 + out_h > H:
            print(
                f"ERROR: crop window ({x0},{y0}) size {out_w}x{out_h} "
                f"leaves video {W}x{H}. Reduce shift."
            )
            sys.exit(1)
        crop_desc = f"{out_w}x{out_h}"

    # Match the renderer: yuv420p H.264 needs even dimensions.
    out_w -= out_w % 2
    out_h -= out_h % 2

    if crop_black_border:
        safe = out_w >= 1 and out_h >= 1
    elif no_crop:
        safe = True
    else:
        safe = _border_check(raw_cum, n_render, x0, y0, out_w, out_h, W, H)

    unreliable = float((~motion["reliable"]).mean()) if T_csv else 1.0
    scale_px = (out_w / 1024.0) * (out_h / 1024.0)
    from .renderer import (
        EST_BYTES_PER_1024,
        EST_LOSSLESS_FACTOR,
        EST_SEC_PER_1024,
        _encoder_preset,
        _quality_label,
        resolve_encoder,
    )

    est_sec = EST_SEC_PER_1024 * scale_px * n_render
    est_mb = EST_BYTES_PER_1024 * scale_px * n_render / 1e6
    if args.crf <= 0:
        est_mb *= EST_LOSSLESS_FACTOR
    dur = human_time(n_render / fps)

    warnings = []
    if no_crop:
        warnings.append("no-crop: black borders may appear (use --crop-black-border to trim them)")
    elif not safe and crop_black_border:
        warnings.append("auto black-border crop found no safe region")
    elif not safe:
        warnings.append("BORDER WARNING: crop window leaves the frame on some frames")
    if T_csv < n_vid:
        warnings.append(f"Motion data covers only {T_csv}/{n_vid} video frames")
    if unreliable > 0.01:
        warnings.append(f"{unreliable:.1%} unreliable motion rows — inspect trajectory.png")

    mode = "smoothed" if args.smooth else "track-locked"

    print("=" * 24, "PROPOSED PARAMETERS", "=" * 24)
    print(f"video:             {args.video} ({n_vid} frames, {W}x{H} @ {fps:.3f} fps)")
    print(f"tracks npz:        {args.tracks}")
    print(f"motion npz:        {args.motion} ({T_csv} rows)")
    if args.smooth:
        print(f"smoothed npz:      {args.smooth}")
    print(f"output:            {args.output}")
    print(f"mode:              {mode}")
    print(f"crop size:         {crop_desc} px")
    print(f"crop window:       x [{x0}..{x0 + out_w}], "
          f"y [{y0}..{y0 + out_h}] "
          f"(centre shifted +{args.shift_x} x, +{args.shift_y} y)")
    full = min(n_vid, T_csv)
    rng = f"all {n_render}" if n_render == full else f"FIRST {n_render} (preview)"
    print(f"frames to render:  {rng} (~{dur} at {fps:.3f} fps)")
    encoder = resolve_encoder()
    q_label = _quality_label(encoder, args.crf)
    print(f"encoder:           {encoder} {q_label} {_encoder_preset(encoder)}, yuv420p")
    print(f"border safety:     {'OK' if safe else 'FAIL'}")
    print(f"est. render time:  ~{human_time(est_sec)} (4 vCPU reference)")
    print(f"est. output size:  ~{est_mb:.0f} MB")
    for w in warnings:
        print(f"!! {w}")
    print("-" * 67)
    print("Run 'render' with the same arguments plus --confirm <note>.")
    print("=" * 67)


def _cmd_render(args: argparse.Namespace) -> None:
    """Execute the ``render`` subcommand."""
    if not args.confirm or not args.confirm.strip():
        print("REFUSED: no user confirmation provided.")
        print("Protocol: run 'plan' with the same arguments, review the")
        print("PROPOSED PARAMETERS, then rerun with --confirm <note>.")
        sys.exit(2)

    print(f'Confirmation: "{args.confirm}"')

    from .renderer import render

    no_crop, crop_black_border, crop_width, crop_height = _effective_crop(args)
    render(
        video_path=args.video,
        output_path=args.output,
        motion_npz=args.motion,
        smooth_npz=args.smooth,
        crop_width=crop_width,
        crop_height=crop_height,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        n_frames=args.frames,
        fps_override=args.fps,
        crf=args.crf,
        no_crop=no_crop,
        crop_black_border=crop_black_border,
    )


def _cmd_viz(args: argparse.Namespace) -> None:
    """Execute the ``viz`` subcommand."""
    from pathlib import Path

    path = Path(args.motion)
    if path.suffix == ".npz":
        from .viz import viz_from_npz

        viz_from_npz(
            motion_npz=args.motion,
            out_png=args.output,
            video_name=args.video_name or "",
        )
    elif path.suffix == ".csv":
        from .viz import viz_from_csv

        viz_from_csv(
            motion_csv=args.motion,
            out_png=args.output,
            video_name=args.video_name or "",
            fps=args.fps,
            gsd=args.gsd,
        )
    else:
        print(f"Unknown file type: {path.suffix} (expected .npz or .csv)")
        sys.exit(1)


def _cmd_overlay(args: argparse.Namespace) -> None:
    """Execute the ``overlay`` subcommand."""
    from pathlib import Path

    from .viz import overlay_tracks

    out = args.output or str(
        Path(args.video).with_suffix(".tracks_overlay.mp4")
    )
    overlay_tracks(
        video_path=args.video,
        tracks_npz=args.tracks,
        output_path=out,
        n_frames=args.frames,
        fps_override=args.fps,
        radius=args.radius,
        trail=args.trail,
        max_points=args.max_points,
        crf=args.crf,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""

    # -----------------------------------------------------------------------
    # Top-level parser
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="stabilize",
        description=(
            "Drone video stabilisation using CoTracker3 keypoint trajectories. "
            "Nine subcommands cover the low-level pipeline and automated "
            "Hugging-Face-integrated runs."
        ),
        epilog=(
            "End-to-end (automated):\n"
            "  stabilize run hf://datasets/u/r/video.mp4 --token hf_xxx --bucket my-bucket --run run1 --checkpoint ./scaled_online.pth\n"
            "\n"
            "Step-by-step:\n"
            "  stabilize track    input.mp4 -o tracks.npz --checkpoint ./scaled_online.pth\n"
            "  stabilize estimate tracks.npz -o motion\n"
            "  stabilize smooth   motion.npz -o motion_smooth.npz --sigma 12\n"
            "  stabilize plan     input.mp4 --tracks tracks.npz --motion motion.npz --crop 1024\n"
            "  stabilize render   input.mp4 --tracks tracks.npz --motion motion.npz \\\n"
            "                          --crop 1024 --output stable.mp4 --confirm 'approved'\n"
            "  stabilize viz      motion.npz -o trajectory.png\n"
            "  stabilize overlay  input.mp4 --tracks tracks.npz -o tracks_overlay.mp4\n"
            "\n"
            "Run 'stabilize <subcommand> --help' for per-command details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -----------------------------------------------------------------------
    # run  (end-to-end)
    # -----------------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Full pipeline: download, stabilise, and upload to HF bucket.",
        description=(
            "Automates the complete workflow in one command: download a video "
            "from an hf:// link (or use a local file), track keypoints, "
            "estimate and smooth camera motion, render the stabilised output, "
            "and upload every artefact to an Hugging Face bucket."
        ),
        epilog=(
            "Authentication: set HF_TOKEN env var, pass --token, or\n"
            "run 'hf auth login' beforehand.\n\n"
            "Output structure in the bucket:\n"
            "  hf://buckets/<bucket>/<run>/tracks.npz\n"
            "  hf://buckets/<bucket>/<run>/motion.npz\n"
            "  hf://buckets/<bucket>/<run>/motion.csv\n"
            "  hf://buckets/<bucket>/<run>/motion_smooth.npz  (unless --no-smooth)\n"
            "  hf://buckets/<bucket>/<run>/stabilized.mp4\n"
            "  hf://buckets/<bucket>/<run>/trajectory.png\n"
            "  hf://buckets/<bucket>/<run>/tracks_overlay.mp4  (unless --skip-overlay)\n"
            "  hf://buckets/<bucket>/<run>/summary.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument(
        "video",
        help=(
            "Source video: a local path OR an hf:// link, e.g. "
            "hf://datasets/user/repo/video.mp4"
        ),
    )
    p_run.add_argument(
        "--token", default=None, metavar="hf_xxx",
        help="Hugging Face token.  Defaults to HF_TOKEN env var.",
    )
    p_run.add_argument(
        "--bucket", required=True, metavar="user/bucket",
        help="HF bucket to upload results to (created automatically if missing).",
    )
    p_run.add_argument(
        "--run", dest="run_name", required=True, metavar="NAME",
        help="Sub-folder name inside the bucket for this run.",
    )
    p_run.add_argument(
        "--private", action="store_true",
        help="Create the bucket as private (if creating a new one).",
    )
    p_run.add_argument(
        "--out-dir", default="runs", metavar="DIR",
        help="Local parent directory for run folders.  Default: ./runs",
    )
    p_run.add_argument(
        "--checkpoint", metavar="scaled_online.pth",
        help=(
            "Path to CoTracker3 checkpoint.  When provided, tracks are "
            "computed locally on a GPU.  If the file is missing, it is "
            "auto-downloaded from the Hub (facebook/cotracker3) into its "
            "parent directory.  If omitted, the default cache "
            "~/.cache/cotracker3/scaled_online.pth is used (downloaded on "
            "first use).  Mutually exclusive with --tracks."
        ),
    )
    p_run.add_argument(
        "--tracks", metavar="TRACKS.npz",
        help=(
            "Pre-computed tracks file.  When provided, local tracking is "
            "skipped.  Mutually exclusive with --checkpoint."
        ),
    )
    p_run.add_argument(
        "--grid-size", type=int, default=16, metavar="N",
        help="NxN keypoint grid (used with --checkpoint).  Default: 16.",
    )
    p_run.add_argument(
        "--grid-query-frame", type=int, default=0, metavar="F",
        help="Frame index for the keypoint grid.  Default: 0.",
    )
    p_run.add_argument(
        "--step", type=int, default=8, metavar="N",
        help=(
            "Online sliding-window stride in frames (window_len = 2*step).  "
            "Total compute is roughly constant (always 50%% overlap); smaller "
            "steps give more frequent, smaller passes.  Default: 8 (window 16)."
        ),
    )
    p_run.add_argument(
        "--max-dim", type=int, default=1280, metavar="PX",
        help="Longest-side resize for tracking.  Default: 1280.",
    )
    p_run.add_argument(
        "--crop-width", type=int, default=None, metavar="PX",
        help=(
            "Width of a fixed-size crop window in pixels.  When neither "
            "--crop-width nor --crop-height is given, the automatic "
            "--crop-black-border mode is used by default.  Default: auto "
            "black-border crop."
        ),
    )
    p_run.add_argument(
        "--crop-height", type=int, default=None, metavar="PX",
        help=(
            "Height of a fixed-size crop window in pixels.  If omitted but "
            "--crop-width is given, defaults to --crop-width (square).  "
            "Default: auto black-border crop."
        ),
    )
    p_run.add_argument(
        "--shift-x", type=int, default=0, metavar="PX",
        help="Crop centre horizontal shift.  Default: 0.",
    )
    p_run.add_argument(
        "--shift-y", type=int, default=0, metavar="PX",
        help="Crop centre vertical shift.  Default: 0.",
    )
    crop_mode = p_run.add_mutually_exclusive_group()
    crop_mode.add_argument(
        "--no-crop", action="store_true",
        help=(
            "Do not crop the output: stabilise the full source frame (WxH). "
            "Black borders may appear where content moved out of view."
        ),
    )
    crop_mode.add_argument(
        "--crop-black-border", action="store_true",
        help=(
            "Auto-detect the largest centred crop window that removes the "
            "black borders introduced by stabilisation.  This is the default "
            "mode; passing the flag is optional.  Overrides "
            "--crop-width/--crop-height."
        ),
    )
    p_run.add_argument(
        "--sigma", type=float, default=10.0, metavar="F",
        help="Gaussian smoothing sigma in frames.  Default: 10.",
    )
    p_run.add_argument(
        "--interp-gap", type=int, default=5, metavar="N",
        help="Interpolate unreliable gaps shorter than N frames.  Default: 5.",
    )
    p_run.add_argument(
        "--smooth", action="store_true",
        help=(
            "Deprecated alias that enables natural-motion smoothing "
            "(equivalent to --stabilization-mode natural).  Smoothing is "
            "OFF by default; track-locked rendering is the default."
        ),
    )
    p_run.add_argument(
        "--stabilization-mode", default="off",
        choices=["natural", "locked", "off"], metavar="MODE",
        help=(
            "off (default): track-lock to frame 0 with no smoothing.  "
            "natural: remove high-frequency jitter, preserve slow pans.  "
            "locked: tripod/static footage -- additionally remove robust "
            "long-term translation drift."
        ),
    )
    p_run.add_argument(
        "--anchor-interval-seconds", type=float, default=10.0, metavar="SEC",
        help=(
            "Maximum spacing between stabilization anchors.  Each frame is fit "
            "directly against its local anchor instead of chaining "
            "frame-to-frame transforms.  Typical 5-30.  Default: 10."
        ),
    )
    p_run.add_argument(
        "--no-fresh-grids", action="store_true",
        help=(
            "Disable the fresh-grid architecture and use one persistent "
            "CoTracker grid for the whole video (classic behaviour).  Fresh "
            "grids are the default: every anchor gets a new query grid and "
            "the previous grid bridges it into the global coordinate system."
        ),
    )
    p_run.add_argument(
        "--min-anchor-interval-seconds", type=float, default=2.0, metavar="SEC",
        help="Minimum spacing between anchors (anti-churn).  Default: 2.",
    )
    p_run.add_argument(
        "--anchor-overlap-seconds", type=float, default=1.5, metavar="SEC",
        help=(
            "Overlap between consecutive tracking segments, used for bridge "
            "registration and transform cross-fading.  Default: 1.5."
        ),
    )
    p_run.add_argument(
        "--min-remaining-point-ratio", type=float, default=0.40, metavar="R",
        help="Re-anchor when surviving points / grid points drops below R.  Default: 0.40.",
    )
    p_run.add_argument(
        "--min-inlier-count", type=int, default=20, metavar="N",
        help="Re-anchor when RANSAC inliers fall below N.  Default: 20.",
    )
    p_run.add_argument(
        "--min-spatial-coverage", type=float, default=0.35, metavar="R",
        help="Re-anchor when occupied image-cell fraction drops below R.  Default: 0.35.",
    )
    p_run.add_argument(
        "--coverage-grid-rows", type=int, default=4, metavar="N",
        help="Spatial-coverage grid rows.  Default: 4.",
    )
    p_run.add_argument(
        "--coverage-grid-cols", type=int, default=4, metavar="N",
        help="Spatial-coverage grid columns.  Default: 4.",
    )
    p_run.add_argument(
        "--quality-failure-patience-frames", type=int, default=5, metavar="N",
        help="Consecutive poor-quality frames required before re-anchoring.  Default: 5.",
    )
    p_run.add_argument(
        "--ransac-reproj-threshold", type=float, default=2.0, metavar="PX",
        help="RANSAC inlier reprojection threshold in pixels.  Default: 2.0.",
    )
    p_run.add_argument(
        "--min-correspondences", type=int, default=20, metavar="N",
        help="Minimum usable correspondences for a transform.  Default: 20.",
    )
    p_run.add_argument(
        "--min-inlier-ratio", type=float, default=0.4, metavar="R",
        help=(
            "Minimum inlier ratio (0-1) for a transform to be trusted; "
            "below this the fallback logic is used.  Default: 0.4."
        ),
    )
    p_run.add_argument(
        "--anchor-blend-frames", type=int, default=0, metavar="N",
        help=(
            "Cross-fade the first N frames of each anchor segment with the "
            "previous anchor to smooth velocity kinks at transitions.  "
            "Position is already continuous at anchors.  Default: 0."
        ),
    )
    p_run.add_argument(
        "--locked-poly-degree", type=int, default=3, metavar="D",
        help=(
            "Degree of the robust polynomial used to model long-term drift "
            "in locked mode.  1 = linear, 3 (default) also removes gentle "
            "curved drift without oscillating."
        ),
    )
    p_run.add_argument(
        "--no-drift", action="store_true",
        help="Skip writing the CoTracker drift diagnostic (motion_drift.*).",
    )
    p_run.add_argument(
        "--debug", action="store_true",
        help="Print per-anchor estimation diagnostics during estimate.",
    )
    p_run.add_argument(
        "--frames", type=int, default=0, metavar="N",
        help="Render only the first N frames (0 = all).",
    )
    p_run.add_argument(
        "--crf", type=int, default=18, metavar="N",
        help=(
            "Output quality: x264 CRF, lower = better.  Default: 18 "
            "(visually lossless).  0 = mathematically lossless (very large)."
        ),
    )
    p_run.add_argument(
        "--skip-upload", action="store_true",
        help="Run the full pipeline locally but skip the HF upload.",
    )
    p_run.add_argument(
        "--skip-overlay", action="store_true",
        help="Skip generating the tracks-overlay video.",
    )
    p_run.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="PyTorch device (cuda:0, cpu, etc.).  Default: auto-detect.",
    )
    p_run.set_defaults(func=_cmd_run)

    # -----------------------------------------------------------------------
    # download
    # -----------------------------------------------------------------------
    p_dl = sub.add_parser(
        "download",
        help="Download a video from an hf:// link.",
        description=(
            "Fetch a video file from an Hugging Face bucket or dataset "
            "using its hf:// link and save it to a local directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_dl.add_argument(
        "link",
        help=(
            "hf:// link to the video, e.g. "
            "hf://datasets/user/repo/video.mp4 or "
            "hf://buckets/user/bucket/path/video.mp4"
        ),
    )
    p_dl.add_argument(
        "-o", "--output-dir", default=".", metavar="DIR",
        help="Local directory to save the downloaded file.  Default: current dir.",
    )
    p_dl.add_argument(
        "--token", default=None, metavar="hf_xxx",
        help="HF token.  Defaults to HF_TOKEN env var.",
    )
    p_dl.set_defaults(func=_cmd_download)

    # -----------------------------------------------------------------------
    # upload
    # -----------------------------------------------------------------------
    p_up = sub.add_parser(
        "upload",
        help="Upload a local directory to an HF bucket.",
        description=(
            "Upload every file in a local directory to an HF bucket under "
            "a specified remote prefix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_up.add_argument(
        "local_dir",
        help="Path to the local directory whose contents will be uploaded.",
    )
    p_up.add_argument(
        "--bucket", required=True, metavar="user/bucket",
        help="Target HF bucket.",
    )
    p_up.add_argument(
        "--remote-prefix", required=True, metavar="run-name",
        help="Remote path prefix inside the bucket (e.g. DJI_0260-run1).",
    )
    p_up.add_argument(
        "--token", default=None, metavar="hf_xxx",
        help="HF token.  Defaults to HF_TOKEN env var.",
    )
    p_up.add_argument(
        "--private", action="store_true",
        help="Create the bucket as private (if creating a new one).",
    )
    p_up.set_defaults(func=_cmd_upload)

    # -----------------------------------------------------------------------
    # track
    # -----------------------------------------------------------------------
    p_track = sub.add_parser(
        "track",
        help="Extract keypoint trajectories from a video using CoTracker3.",
        description=(
            "Run the CoTracker3 online model on a local GPU to track an NxN "
            "grid of keypoints through every frame of the input video.  Output "
            "is an .npz file with per-frame (x,y) trajectories and visibility "
            "flags, suitable for ``stabilize estimate``."
        ),
        epilog=(
            "Requires: torch (CUDA), cotracker Python package, and a GPU.\n"
            "The model checkpoint (scaled_online.pth, ~100 MB) is downloaded\n"
            "automatically from the Hub (facebook/cotracker3) on first use.\n\n"
            "For remote/server tracking (no local GPU), use the CoTracker3\n"
            "MCP server or FastAPI backend instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_track.add_argument(
        "video",
        help="Path to the input video file (MP4, MOV, AVI, MKV, etc.).",
    )
    p_track.add_argument(
        "-o", "--output", required=True, metavar="TRACKS.npz",
        help="Output path for the tracks .npz file.",
    )
    p_track.add_argument(
        "--checkpoint", default=None, metavar="scaled_online.pth",
        help=(
            "Path to the CoTracker3 online model checkpoint.  If omitted or "
            "the file is missing, it is auto-downloaded from the Hub "
            "(facebook/cotracker3) into ~/.cache/cotracker3/ on first use."
        ),
    )
    p_track.add_argument(
        "--grid-size", type=int, default=16, metavar="N",
        help=(
            "NxN grid of keypoints over the query frame.  Number of points "
            "= N*N.  No cap (GPU memory scales with N*N).  Default: 16 "
            "(256 points). Higher values give denser tracking but use more "
            "GPU memory."
        ),
    )
    p_track.add_argument(
        "--grid-query-frame", type=int, default=0, metavar="F",
        help=(
            "Frame index from which the keypoint grid is sampled.  "
            "Default: 0 (first frame)."
        ),
    )
    p_track.add_argument(
        "--max-dim", type=int, default=1280, metavar="PX",
        help=(
            "Resize the longest side of the video to this many pixels "
            "before tracking (bounded GPU memory).  Tracked coordinates "
            "are mapped back to original resolution.  Default: 1280."
        ),
    )
    p_track.add_argument(
        "--step", type=int, default=8, metavar="N",
        help=(
            "Online sliding-window stride in frames.  CoTracker3's online "
            "model processes window_len = 2*step frames per call and advances "
            "by step, so total compute is roughly constant (always 50%% "
            "overlap).  Smaller steps give more frequent, smaller passes "
            "(lower peak memory).  Default: 8 (window 16)."
        ),
    )
    p_track.add_argument(
        "--device", default=None, metavar="DEVICE",
        help=(
            "PyTorch device, e.g. 'cuda:0', 'cuda:1', 'cpu'.  "
            "Default: auto-detect (first CUDA GPU)."
        ),
    )
    p_track.set_defaults(func=_cmd_track)

    # -----------------------------------------------------------------------
    # estimate
    # -----------------------------------------------------------------------
    p_est = sub.add_parser(
        "estimate",
        help="Estimate per-frame camera motion from tracked keypoints.",
        description=(
            "Estimates the camera trajectory from persistent CoTracker "
            "correspondences.  Each frame is fit DIRECTLY against a local "
            "anchor frame with RANSAC (4 DOF similarity: translation, yaw, "
            "uniform scale), then placed into a global (frame-0) coordinate "
            "system via one anchor->global composition.  Frame-to-frame "
            "transforms are never integrated, so estimation error does not "
            "accumulate over thousands of frames."
        ),
        epilog=(
            "The estimation uses cv2.estimateAffinePartial2D with RANSAC.\n"
            "Anchors default to every --anchor-interval-seconds (10 s).\n\n"
            "Convention (column vectors, p_dst = M @ p_src):\n"
            "  local[t]          maps frame t  -> segment anchor K\n"
            "  anchor_global[i]  maps anchor K -> global frame 0\n"
            "  frame_to_global   maps frame t  -> global frame 0\n"
            "  cumulative[t]     maps global frame 0 -> frame t (renderer)\n"
            "Use --legacy to reproduce the old drifting estimator for A/B\n"
            "comparison, and --drift to write a tracker-drift diagnostic."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_est.add_argument(
        "tracks",
        help="Path to the tracks .npz file (from ``stabilize track`` or CoTracker3 API).",
    )
    p_est.add_argument(
        "-o", "--output", required=True, metavar="PREFIX",
        help=(
            "Output base path (without extension).  Creates PREFIX.npz "
            "(full-precision matrices) and PREFIX.csv (human-readable summary)."
        ),
    )
    p_est.add_argument(
        "--fps", type=float, default=None, metavar="FPS",
        help=(
            "Override the video FPS.  Default: read from the .npz meta, "
            "or 23.976 if not available."
        ),
    )
    p_est.add_argument(
        "--width", type=int, default=None, metavar="PX",
        help="Image width in pixels.  Default: read from .npz meta, or 3840.",
    )
    p_est.add_argument(
        "--height", type=int, default=None, metavar="PX",
        help="Image height in pixels.  Default: read from .npz meta, or 2160.",
    )
    p_est.add_argument(
        "--max-points", type=int, default=20000, metavar="N",
        help=(
            "Maximum usable correspondences per transform for RANSAC.  "
            "Higher values are more accurate but slower.  Default: 20000."
        ),
    )
    p_est.add_argument(
        "--anchor-interval-seconds", type=float, default=10.0, metavar="SEC",
        help=(
            "Spacing between stabilization anchors.  Every frame is fit "
            "directly against its local anchor, so transform error does not "
            "accumulate frame-to-frame.  Typical 5-30.  Default: 10."
        ),
    )
    p_est.add_argument(
        "--ransac-reproj-threshold", type=float, default=2.0, metavar="PX",
        help="RANSAC inlier reprojection threshold in pixels.  Default: 2.0.",
    )
    p_est.add_argument(
        "--min-correspondences", type=int, default=20, metavar="N",
        help="Minimum usable correspondences to attempt a transform.  Default: 20.",
    )
    p_est.add_argument(
        "--min-inlier-ratio", type=float, default=0.4, metavar="R",
        help=(
            "Minimum inlier ratio (0-1) for a transform to be trusted; "
            "otherwise the fallback logic is used.  Default: 0.4."
        ),
    )
    p_est.add_argument(
        "--anchor-blend-frames", type=int, default=0, metavar="N",
        help="Cross-fade N frames after each anchor to smooth velocity.  Default: 0.",
    )
    p_est.add_argument(
        "--legacy", action="store_true",
        help=(
            "Use the deprecated frame-to-frame accumulation estimator.  "
            "Kept only for A/B drift comparison; do not use in production."
        ),
    )
    p_est.add_argument(
        "--drift", action="store_true",
        help=(
            "Also write a tracker-drift diagnostic (PREFIX_drift.npz/.csv): "
            "absolute point displacement relative to anchor 0 and to each "
            "segment anchor."
        ),
    )
    p_est.add_argument(
        "--debug", action="store_true",
        help="Print per-anchor diagnostics during estimation.",
    )
    p_est.set_defaults(func=_cmd_estimate)

    # -----------------------------------------------------------------------
    # fresh-grid (segmented tracking + estimate)
    # -----------------------------------------------------------------------
    p_fg = sub.add_parser(
        "fresh-grid",
        help="Segment-tracking estimate with a fresh CoTracker grid per anchor.",
        description=(
            "Runs an independent CoTracker session per tracking segment.  Every "
            "anchor gets a brand-new query grid; the previous grid bridges the "
            "new anchor into the global coordinate system through the overlap "
            "region.  Anchors are created on a max interval or early when "
            "tracking quality (point survival, inlier ratio, spatial coverage) "
            "deteriorates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_fg.add_argument("video", help="Path to the source video file.")
    p_fg.add_argument(
        "-o", "--output", required=True, metavar="PREFIX",
        help="Output base path; writes PREFIX.npz/.csv and PREFIX.segments.json.",
    )
    p_fg.add_argument("--checkpoint", default=None, metavar="scaled_online.pth",
                      help="CoTracker3 checkpoint (auto-downloaded if missing).")
    p_fg.add_argument("--grid-size", type=int, default=16, metavar="N",
                      help="NxN fresh query grid per anchor.  Default: 16.")
    p_fg.add_argument("--max-dim", type=int, default=1280, metavar="PX",
                      help="Longest-side resize for tracking.  Default: 1280.")
    p_fg.add_argument("--step", type=int, default=8, metavar="N",
                      help="Online sliding-window stride (window_len = 2*step).  "
                           "Default: 8; use 1 for per-frame stepping in tests.")
    p_fg.add_argument("--device", default=None, metavar="DEVICE",
                      help="torch device (cuda:0, cpu).  Default: auto.")
    p_fg.add_argument("--segment-dir", default=None, metavar="DIR",
                      help="Directory for per-segment tracks (default: "
                           "<output>_segments/).")
    p_fg.add_argument("--anchor-interval-seconds", type=float, default=10.0,
                      metavar="SEC", help="Maximum anchor spacing.  Default: 10.")
    p_fg.add_argument("--min-anchor-interval-seconds", type=float, default=2.0,
                      metavar="SEC", help="Minimum anchor spacing.  Default: 2.")
    p_fg.add_argument("--anchor-overlap-seconds", type=float, default=1.5,
                      metavar="SEC", help="Segment overlap.  Default: 1.5.")
    p_fg.add_argument("--min-remaining-point-ratio", type=float, default=0.40,
                      metavar="R", help="Re-anchor below this survival ratio.")
    p_fg.add_argument("--min-inlier-ratio", type=float, default=0.40, metavar="R",
                      help="Re-anchor below this RANSAC inlier ratio.")
    p_fg.add_argument("--min-inlier-count", type=int, default=20, metavar="N",
                      help="Re-anchor below this inlier count.")
    p_fg.add_argument("--min-spatial-coverage", type=float, default=0.35,
                      metavar="R", help="Re-anchor below this cell coverage.")
    p_fg.add_argument("--coverage-grid-rows", type=int, default=4, metavar="N",
                      help="Coverage grid rows.  Default: 4.")
    p_fg.add_argument("--coverage-grid-cols", type=int, default=4, metavar="N",
                      help="Coverage grid columns.  Default: 4.")
    p_fg.add_argument("--quality-failure-patience-frames", type=int, default=5,
                      metavar="N", help="Consecutive bad frames before re-anchor.")
    p_fg.add_argument("--ransac-reproj-threshold", type=float, default=2.0,
                      metavar="PX", help="RANSAC reprojection threshold.  Default: 2.")
    p_fg.add_argument("--min-correspondences", type=int, default=20, metavar="N",
                      help="Minimum correspondences for a transform.  Default: 20.")
    p_fg.add_argument("--debug", action="store_true",
                      help="Print per-anchor diagnostics.")
    p_fg.set_defaults(func=_cmd_fresh_grid)

    # -----------------------------------------------------------------------
    # smooth
    # -----------------------------------------------------------------------
    p_smooth = sub.add_parser(
        "smooth",
        help="Gaussian-smooth the estimated camera path.",
        description=(
            "Decomposes the measured global camera path into (translation, "
            "yaw, log-scale), applies Gaussian smoothing with short-gap "
            "interpolation, and reconstructs the smoothed reference "
            "transforms.  In 'locked' mode a robust (Theil-Sen) long-term "
            "translation ramp is also removed; in 'natural' mode slow pans "
            "are preserved."
        ),
        epilog=(
            "Sigma controls the smoothing window size in frames.  Larger "
            "values produce smoother but more delayed camera motion.  "
            "Typical range: 5-30.  Default: 10.\n\n"
            "Unreliable gaps shorter than --interp-gap frames are linearly "
            "interpolated.  Longer gaps hold the last reliable value."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_smooth.add_argument(
        "motion",
        help="Path to the motion .npz file (from ``stabilize estimate``).",
    )
    p_smooth.add_argument(
        "-o", "--output", required=True, metavar="SMOOTH.npz",
        help="Output path for the smoothed-motion .npz file.",
    )
    p_smooth.add_argument(
        "--sigma", type=float, default=10.0, metavar="SIGMA",
        help=(
            "Gaussian smoothing sigma in frames.  Larger = smoother but "
            "more delayed.  Typical: 5 (mild) to 30 (heavy).  Default: 10."
        ),
    )
    p_smooth.add_argument(
        "--interp-gap", type=int, default=5, metavar="N",
        help=(
            "Unreliable gaps shorter than N frames are linearly "
            "interpolated before smoothing.  Default: 5."
        ),
    )
    p_smooth.add_argument(
        "--mode", default="natural", choices=["natural", "locked"], metavar="MODE",
        help=(
            "natural: preserve legitimate low-frequency pans (no detrend).  "
            "locked: robustly remove long-term translation drift (tripod / "
            "static-camera footage).  Default: natural."
        ),
    )
    p_smooth.add_argument(
        "--locked-poly-degree", type=int, default=3, metavar="D",
        help=(
            "Degree of the robust polynomial used to model long-term drift "
            "in locked mode.  1 = linear, 3 (default) also removes gentle "
            "curved drift without oscillating."
        ),
    )
    p_smooth.set_defaults(func=_cmd_smooth)

    # -----------------------------------------------------------------------
    # plan
    # -----------------------------------------------------------------------
    p_plan = sub.add_parser(
        "plan",
        help="Validate inputs and print proposed render parameters.",
        description=(
            "Dry-run for ``render``.  Validates that the video, tracks, and "
            "motion data are consistent, checks border safety, estimates "
            "render time and file size, and prints a PROPOSED PARAMETERS "
            "block for review before actually rendering."
        ),
        epilog=(
            "Always run 'plan' before 'render' to verify parameters.\n"
            "Review the output, then run 'render' with --confirm <note>."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_render_args(p_plan)
    p_plan.set_defaults(func=_cmd_plan)

    # -----------------------------------------------------------------------
    # render
    # -----------------------------------------------------------------------
    p_render = sub.add_parser(
        "render",
        help="Render the stabilised video (requires --confirm).",
        description=(
            "Reads each frame from the source video, applies a per-frame "
            "affine warp that locks tracked keypoints to their first-frame "
            "positions (optionally with camera-path smoothing), and writes "
            "the result via FFmpeg as H.264/MP4."
        ),
        epilog=(
            "REFUSES to run without --confirm.  This is intentional: you\n"
            "must run 'plan' first, review the proposed parameters, and\n"
            "pass --confirm with a note of what was approved.\n\n"
            "Border safety is checked before rendering.  If the check fails,\n"
            "the output may contain black edges where the crop window leaves\n"
            "the source frame."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_render_args(p_render)
    p_render.add_argument(
        "--confirm", metavar="NOTE", default="",
        help=(
            "MANDATORY: quote what was approved, e.g. "
            "'user approved 1024px crop, +150 right, full length'.  "
            "Refuses to run without it (exit code 2)."
        ),
    )
    p_render.set_defaults(func=_cmd_render)

    # -----------------------------------------------------------------------
    # viz
    # -----------------------------------------------------------------------
    p_viz = sub.add_parser(
        "viz",
        help="Generate a camera-motion trajectory visualisation.",
        description=(
            "Creates a multi-panel diagnostic plot showing: camera trajectory "
            "(X-Y coloured by time), per-frame translation, yaw rate and "
            "cumulative yaw, zoom scale, usable point count and inlier ratio, "
            "and a summary strip.  Works from either .npz or .csv motion data."
        ),
        epilog=(
            "The plot uses red shading to highlight unreliable frame regions.\n"
            "For .csv input, missing columns (cum_log_scale, usable, etc.)\n"
            "are automatically reconstructed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_viz.add_argument(
        "motion",
        help=(
            "Path to the motion data file: either .npz (from "
            "``stabilize estimate``) or .csv (legacy)."
        ),
    )
    p_viz.add_argument(
        "-o", "--output", default="trajectory.png", metavar="PNG",
        help="Output path for the trajectory plot.  Default: trajectory.png.",
    )
    p_viz.add_argument(
        "--video-name", default="", metavar="NAME",
        help="Video name shown in the plot title.",
    )
    p_viz.add_argument(
        "--fps", type=float, default=None, metavar="FPS",
        help="FPS shown in the title (csv mode only).",
    )
    p_viz.add_argument(
        "--gsd", type=float, default=None, metavar="M/PX",
        help=(
            "Ground sample distance (metres per pixel).  If set, speeds "
            "are also shown in m/s.  Csv mode only."
        ),
    )
    p_viz.set_defaults(func=_cmd_viz)

    # -----------------------------------------------------------------------
    # overlay
    # -----------------------------------------------------------------------
    p_ov = sub.add_parser(
        "overlay",
        help="Visualise tracked keypoints on the source video.",
        description=(
            "Draws the CoTracker3 keypoint trajectories on top of the "
            "original video and encodes the result as an MP4.  Visible "
            "points are coloured dots (colour fixed per point id, with a "
            "short trail of recent positions); points the tracker currently "
            "considers invisible are drawn as red crosses.  The grid-query "
            "frame is marked with hollow cyan squares at the initial query "
            "locations."
        ),
        epilog=(
            "Points are drawn in original-video resolution (the .npz\n"
            "coordinates are mapped back to it by 'stabilize track').  Use\n"
            "--max-points to subsample a dense grid for a cleaner image.\n\n"
            "Requires FFmpeg with the libopenh264 encoder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ov.add_argument(
        "video",
        help="Path to the source video file (MP4, MOV, AVI, MKV, etc.).",
    )
    p_ov.add_argument(
        "--tracks", required=True, metavar="TRACKS_NPZ",
        help=(
            "Path to the tracks .npz file produced by ``stabilize track`` "
            "(or the CoTracker3 API).  Coordinates must be in original "
            "video pixels."
        ),
    )
    p_ov.add_argument(
        "-o", "--output", default=None, metavar="OUTPUT.mp4",
        help=(
            "Output path for the overlay video.  Default: "
            "<video>.tracks_overlay.mp4 next to the source video."
        ),
    )
    p_ov.add_argument(
        "--max-points", type=int, default=0, metavar="N",
        help=(
            "If the tracks contain more than N points, draw only N evenly "
            "spaced points (cleaner image on dense grids).  "
            "0 (default) draws all points."
        ),
    )
    p_ov.add_argument(
        "--radius", type=int, default=4, metavar="PX",
        help="Dot radius in pixels.  Default: 4.",
    )
    p_ov.add_argument(
        "--trail", type=int, default=8, metavar="N",
        help=(
            "Length of the position trail behind each visible point, in "
            "frames.  0 disables trails.  Default: 8."
        ),
    )
    p_ov.add_argument(
        "--frames", type=int, default=0, metavar="N",
        help="Overlay only the first N frames.  0 (default) overlays all.",
    )
    p_ov.add_argument(
        "--fps", type=float, default=None, metavar="FPS",
        help="Override the output frame rate.  Default: source video FPS.",
    )
    p_ov.add_argument(
        "--crf", type=int, default=20, metavar="N",
        help="x264 CRF for output quality.  Default: 20.",
    )
    p_ov.set_defaults(func=_cmd_overlay)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``stabilize`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
