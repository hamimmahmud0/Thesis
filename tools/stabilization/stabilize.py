#!/usr/bin/env python3
"""Video stabilization tool: fixed-window, track-locked (CoTracker3 + motion.csv).

Stabilizes a video so every tracked keypoint keeps its FIRST-FRAME pixel
position. Output is a fixed WxH window whose center can be shifted in
frame-0 coordinates.

TWO+ subcommands - the verification gate is MANDATORY:

  plan    Validate inputs and print a PROPOSED PARAMETERS block.
          Run this FIRST and paste the block to the user verbatim.

  render  Render the stabilized video. REFUSES to run unless you pass
          --confirm "<what the user approved>". Never bypass this: the
          user must verify crop size / shifts / frame range first.

  viz     Regenerate the camera-motion visualization sheet (trajectory,
          translation, yaw, zoom, diagnostics) from an existing motion.csv.

Pipeline prerequisites per video:
  1) tracks .npz   (keys: tracks [T,N,2] f32 original-px, visibility [T,N]
                    bool, query_points, meta JSON with width/height/fps)
      -> see COT3/llms.txt in this bucket to produce them.
  2) motion.csv    (camera-convention per-frame motion; produced by
                    motion_from_tracks.py in this folder, which re-defines
                    the grid points every --window frames (default 100,
                    windows 0-99, 99-199, 199-299, ...) so correspondences
                    do not diminish over a long video; the per-window motion
                    is combined by chaining through the shared anchor frames)
  3) stabilize.py plan / render (this script)

Math (do not "simplify"):
  IMAGE-motion A_t maps prev->curr points: p_t = A_t p_{t-1}.
  Cumulative forward map C_t = A_t @ ... @ A_1 maps frame-0 -> frame-t coords.
  Output relation for window origin o=(x0,y0): dst(u) = src_t(C_t (o+u)).
  cv2.warpAffine samples dst(u)=src(M^-1 u) by default, so we MUST pass
  flags=...|cv2.WARP_INVERSE_MAP with M built so that M u == C_t (o+u).
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

REQUIRED_CSV_COLS = {"frame", "dx", "dy", "d_yaw_deg", "scale", "reliable"}
EST_SEC_PER_FRAME_1024 = 0.09  # measured: ~11 min / 7852 frames on 4 vCPU
EST_BYTES_PER_FRAME_1024 = 31_000  # observed CRF20 veryfast @1024px


def incremental_image_matrix(dx, dy, d_yaw_deg, scale):
    """Rebuild IMAGE-motion A from stored CAMERA-convention parameters."""
    phi = np.radians(-d_yaw_deg)
    c, s = np.cos(phi), np.sin(phi)
    return np.array(
        [
            [scale * c, -scale * s, -dx],
            [scale * s, scale * c, -dy],
        ],
        dtype=np.float64,
    )


def load_cumulative(motion_csv):
    df = pd.read_csv(motion_csv)
    missing = REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        sys.exit(f"motion.csv missing columns: {sorted(missing)}")
    T = len(df)
    C = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    for t in range(1, T):
        r = df.iloc[t]
        A = incremental_image_matrix(r.dx, r.dy, r.d_yaw_deg, r.scale)
        At = np.vstack([A, [0.0, 0.0, 1.0]])
        C[t] = At @ C[t - 1]
    return C, df


def combined_warp(C3, origin):
    lin = C3[:2, :2]
    off = C3[:2, 2:3] + lin @ np.asarray(origin, dtype=np.float64).reshape(2, 1)
    return np.hstack([lin, off]).astype(np.float32)


def motion_window_info(df):
    """Windowed-grid info from a motion.csv produced with --window.

    Returns (n_windows, window_len or None). Windowed CSVs carry a `window`
    column; the grid was re-defined (anchored) every `window` frames and the
    per-frame motion was chained through the shared anchor frames, so the
    cumulative map C_t needs no special handling here.
    """
    if "window" not in df.columns:
        return None, None
    w = df["window"].to_numpy()
    n = int(w.max()) + 1
    if n < 2 or len(w) < 2:
        return n, None
    change = df["frame"].to_numpy()[1:][np.diff(w) != 0]
    spacing = np.diff(change)
    return n, (int(np.median(spacing)) if len(spacing) else None)


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open video {path}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    if n_vid <= 0 or W <= 0 or H <= 0:
        sys.exit("video probe failed (0 frames/size) - re-encode or check file")
    return cap, n_vid, W, H, fps


def check_npz_meta(npz_path, W, H):
    import json

    try:
        d = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot read npz meta ({e}); expected JSON string with width/height")
    mw, mh = int(meta["width"]), int(meta["height"])
    if (mw, mh) != (W, H):
        sys.exit(
            f"MISMATCH: tracks npz meta says {mw}x{mh} but video is {W}x{H} "
            "- wrong tracks/video pair?"
        )
    return meta


def border_check(C, n_render, x0, y0, crop, W, H):
    corners_o = np.array(
        [
            [x0, y0, 1],
            [x0 + crop, y0, 1],
            [x0 + crop, y0 + crop, 1],
            [x0, y0 + crop, 1],
        ],
        dtype=np.float64,
    )
    src_xy = np.einsum("tij,kj->tki", C[:n_render], corners_o)
    ok = (
        (src_xy[..., 0] >= 0).all()
        and (src_xy[..., 0] <= W - 1).all()
        and (src_xy[..., 1] >= 0).all()
        and (src_xy[..., 1] <= H - 1).all()
    )
    return bool(ok)


def validate(args):
    """Common validation for plan/render. Returns context dict."""
    cap, n_vid, W, H, fps = open_video(args.video)
    if args.fps:
        fps = args.fps
    meta = check_npz_meta(args.tracks_npz, W, H)
    if args.crop > min(W, H):
        sys.exit(f"crop {args.crop} exceeds sensor {W}x{H}")
    C, df = load_cumulative(args.motion_csv)
    T_csv = len(C)
    unreliable = float((~df["reliable"].astype(bool)).mean()) if T_csv else 1.0
    n_render = min(n_vid, T_csv, args.frames or max(n_vid, T_csv))
    x0 = (W - args.crop) // 2 + args.shift_x
    y0 = (H - args.crop) // 2 + args.shift_y
    if x0 < 0 or y0 < 0 or x0 + args.crop > W or y0 + args.crop > H:
        sys.exit(
            f"window origin ({x0},{y0}) size {args.crop} leaves sensor {W}x{H} "
            "- reduce shift"
        )
    safe = border_check(C, n_render, x0, y0, args.crop, W, H)
    scale_px = (args.crop / 1024.0) ** 2
    est_sec = EST_SEC_PER_FRAME_1024 * scale_px * n_render
    est_mb = EST_BYTES_PER_FRAME_1024 * scale_px * n_render / 1e6
    ctx = dict(
        cap=cap, n_vid=n_vid, W=W, H=H, fps=fps, meta=meta, C=C, df=df,
        T_csv=T_csv, n_render=n_render, x0=x0, y0=y0, safe=safe,
        est_sec=est_sec, est_mb=est_mb, unreliable=unreliable,
    )
    return ctx


def human_time(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def print_plan(args, c):
    warnings = []
    if not c["safe"]:
        warnings.append("BORDER WARNING: window samples leave the frame on some t")
    if c["T_csv"] < c["n_vid"]:
        warnings.append(f"motion.csv covers only {c['T_csv']}/{c['n_vid']} frames")
    if c["unreliable"] > 0.01:
        warnings.append(f"{c['unreliable']:.1%} unreliable motion rows - inspect tracks")
    dur = human_time(c["n_render"] / c["fps"])
    print("=" * 24, "PROPOSED PARAMETERS", "=" * 24)
    print(f"video:             {args.video} ({c['n_vid']} frames, "
          f"{c['W']}x{c['H']} @ {c['fps']:.3f} fps)")
    print(f"tracks npz:        {args.tracks_npz} (meta matches video)")
    print(f"motion csv:        {args.motion_csv} ({c['T_csv']} rows)")
    n_win, win_len = motion_window_info(c["df"])
    if n_win is not None and n_win > 1:
        note = f"every {win_len} frames" if win_len else "per window"
        print(f"motion windows:    {n_win} (grid points re-defined {note}; "
              f"motion chained through shared anchors)")
    else:
        print("motion windows:    none (single grid for the whole video)")
    print(f"output:            {args.out}")
    print(f"crop size:         {args.crop}x{args.crop} px (fixed, frame-0 locked)")
    print(f"crop window:       x [{c['x0']}..{c['x0'] + args.crop}], "
          f"y [{c['y0']}..{c['y0'] + args.crop}] "
          f"(center shifted +{args.shift_x} x, +{args.shift_y} y)")
    full = min(c["n_vid"], c["T_csv"])
    rng = f"all {c['n_render']}" if c["n_render"] == full \
        else f"FIRST {c['n_render']} (preview)"
    print(f"frames to render:  {rng} (~{dur} at {c['fps']:.3f} fps)")
    print(f"encoder:           libx264 CRF{args.crf} veryfast, yuv420p")
    print(f"border safety:     {'OK' if c['safe'] else 'FAIL'}")
    print(f"est. render time:  ~{human_time(c['est_sec'])} (4 vCPU reference)")
    print(f"est. output size:  ~{c['est_mb']:.0f} MB")
    for w in warnings:
        print(f"!! {w}")
    print("-" * 67)
    print("ASK THE USER: paste this block verbatim. Ask them to approve or adjust")
    print("(crop size, shift-x/y, frame range, CRF). For videos longer than ~2 min,")
    print("offer a --frames 1200 preview first.")
    print("Render ONLY after explicit approval:")
    print(f"  python3 stabilize.py render <same arguments> "
          f"--confirm \"<user approval note>\"")
    print("=" * 67)


def viz_motion(motion_csv, out_png, fps=None, video_name="", gsd=None):
    """Render the camera-motion visualization sheet from an existing motion.csv.

    Same style as motion_from_tracks.make_plots (trajectory XY colored by
    time, per-frame translation, yaw, zoom scale, diagnostics + summary
    strip). Works on CSVs that lack the extended columns by reconstructing:
      cum_log_scale[t] = sum(log(scale) over reliable rows up to t)
      usable approximated by inliers when inlier_ratio is present.
    """
    import matplotlib

    matplotlib.use("Agg")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from motion_from_tracks import make_plots
    except ImportError:
        sys.exit("motion_from_tracks.py must sit next to stabilize.py")

    df = pd.read_csv(motion_csv)
    missing = REQUIRED_CSV_COLS - set(df.columns)
    if missing:
        sys.exit(f"motion.csv missing columns: {sorted(missing)}")
    if "cum_log_scale" not in df.columns:
        reliable = df["reliable"].astype(bool).to_numpy()
        logs = np.log(df["scale"].clip(lower=1e-9)).to_numpy()
        logs[~reliable] = 0.0
        df["cum_log_scale"] = np.cumsum(logs)
    if "usable" not in df.columns:
        ratio = df["inlier_ratio"].to_numpy()
        df["usable"] = np.where(ratio > 0, (df["inliers"] / np.maximum(ratio, 1e-9)), 0)
    if "cum_yaw_deg" not in df.columns:
        df["cum_yaw_deg"] = np.cumsum(np.where(
            df["reliable"].astype(bool), df["d_yaw_deg"], 0.0))

    title = f"Camera motion ({video_name or os.path.basename(motion_csv)}) - " \
            f"{fps:.3f} fps" if fps else \
            f"Camera motion ({video_name or os.path.basename(motion_csv)})"
    make_plots(df, out_png, title)

    m = df[df["reliable"]]
    step = np.hypot(m["dx"], m["dy"])
    print("===== SUMMARY =====")
    print(f"rows             : {len(df)}")
    print(f"path length      : {step.sum():,.0f} px", end="")
    if gsd:
        print(f"  =  {step.sum() * gsd:,.1f} m", end="")
    print()
    if fps:
        print(f"mean speed       : {step.mean():.3f} px/frame", end="")
        if gsd:
            print(f"  =  {step.mean() * gsd * fps:.2f} m/s", end="")
        print()
    print(f"total yaw change : {df['cum_yaw_deg'].iloc[-1]:+.2f} deg")
    print(f"net scale change : x{np.exp(df['cum_log_scale'].iloc[-1]):.4f}")
    print(f"unreliable frames: {(~df['reliable']).mean() * 100:.2f}%")
    print(f"wrote: {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description="Track-locked video stabilization (plan/render).",
        epilog="ALWAYS run 'plan' and get user approval before 'render'.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--video", required=True)
        p.add_argument("--tracks-npz", required=True)
        p.add_argument("--motion-csv", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--crop", type=int, default=1024, help="square crop side px")
        p.add_argument("--shift-x", type=int, default=0,
                       help="shift crop center right, frame-0 px")
        p.add_argument("--shift-y", type=int, default=0,
                       help="shift crop center down, frame-0 px")
        p.add_argument("--frames", type=int, default=0,
                       help="render only first N frames, 0=all")
        p.add_argument("--fps", type=float, default=None)
        p.add_argument("--crf", type=int, default=20)

    p_plan = sub.add_parser("plan", help="validate + print parameters for user")
    common(p_plan)
    p_rend = sub.add_parser("render", help="render (requires user-approved params)")
    common(p_rend)
    p_rend.add_argument(
        "--confirm", metavar="NOTE",
        help='mandatory: quote what the user approved, e.g. '
             '"user approved 1024px crop, +150px right, full length". '
             'Refuses to run without it.',
    )
    p_viz = sub.add_parser(
        "viz",
        help="camera-motion visualization (trajectory.png) from an existing "
             "motion.csv - no RANSAC re-run needed",
    )
    p_viz.add_argument("--motion-csv", required=True)
    p_viz.add_argument("--out-png", default="trajectory.png")
    p_viz.add_argument("--fps", type=float, default=None)
    p_viz.add_argument("--video-name", default="", help="shown in the title")
    p_viz.add_argument("--gsd", type=float, default=None,
                       help="meters/pixel; if set, speeds shown in m/s too")
    args = ap.parse_args()

    if args.cmd == "render":
        if not args.confirm or not args.confirm.strip():
            print("REFUSED: no user confirmation provided.")
            print("Protocol: run 'plan' with the same arguments, paste the")
            print("PROPOSED PARAMETERS block to the user, get explicit approval,")
            print('then rerun with --confirm "<what the user approved>".')
            sys.exit(2)
        print(f'confirmation on record: "{args.confirm}"')

    if args.cmd == "viz":
        viz_motion(args.motion_csv, args.out_png, args.fps,
                   args.video_name, args.gsd)
        return

    ctx = validate(args)
    c = ctx
    if args.cmd == "plan":
        print_plan(args, c)
        c["cap"].release()
        return

    # ---------------- render ----------------
    n_render, crop = c["n_render"], args.crop
    C = c["C"]
    print(f"video: {c['n_vid']} frames {c['W']}x{c['H']} @ {c['fps']:.3f} fps")
    print(f"stabilizing {n_render} frames; crop {crop}x{crop}; "
          f"window x [{c['x0']},{c['x0'] + crop}], y [{c['y0']},{c['y0'] + crop}]")
    if not c["safe"]:
        print("WARNING: border safety FAILED - output will contain black edges")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{crop}x{crop}", "-r", f"{c['fps']:.6f}",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", args.out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(n_render):
        okread, frame = c["cap"].read()
        if not okread:
            print(f"video ended early at t={t}")
            break
        N = combined_warp(C[t], (c["x0"], c["y0"]))
        out = cv2.warpAffine(
            frame, N, (crop, crop),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        proc.stdin.write(out.tobytes())
        if t % 500 == 0:
            print(f"t={t}/{n_render}", flush=True)
    c["cap"].release()
    proc.stdin.close()
    proc.wait()
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
