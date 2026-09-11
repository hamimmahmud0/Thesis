# REPRODUCE.md - stabilize ANY video end-to-end

Prereqs: python3.10+ with numpy, opencv-python-headless, pandas, matplotlib;
ffmpeg on PATH; ~3 GB free RAM for 4K dense tracks. No GPU needed after
tracking (tracking itself uses the COT3 server).

## 1. Get CoTracker3 tracks (.npz)

Required schema: tracks (T,N,2) float32 ORIGINAL-pixel coords, visibility
(T,N) bool, query_points (N,2), meta JSON string with width/height/fps.

- Find the backend: GET http://163.61.236.112/v1/announce ("online": true).
- API usage: read COT3/llms.txt in this bucket.
- NOTE: the MCP/backend clamps query grids to NxN<=32; a rectangular
  fixed-density grid cannot be expressed through the API. Replicate the
  canonical online loop instead - see task_cot3-dense16-track-*/scripts/
  track_dense_amp.py in z81980440/agent_bucket (window_len=16, step=8,
  every frame tracked, longest side 1280 processing res, AMP fp16,
  coords mapped back to ORIGINAL pixels).
- Save npz + summary.json + scripts used in a folder named
  task_cot3-denseNN-track-<UTC ts>/ and publish per Agent.md protocol.

## 2. Camera motion (.csv) + visualization

    python3 motion_from_tracks.py tracks.npz --out-dir out [--gsd m_per_px]

Writes out/motion.csv (camera convention; columns frame,dx,dy,d_yaw_deg,
scale,inliers,inlier_ratio,cum_x,cum_y,cum_yaw_deg,reliable[,usable,
cum_log_scale]) and out/trajectory.png (the camera-motion sheet).
Runtime ~=90 s for T=7856,N=32400 on 4 vCPU.

Regenerate the sheet later from any motion.csv:

    python3 stabilize.py viz --motion-csv out/motion.csv --fps 23.976 \
        --video-name YOUR_VIDEO.MP4 [--gsd m/px] --out-png trajectory.png

(Older CSVs without usable/cum_log_scale are handled: values are
reconstructed.)

## 3. Stabilize - WITH user verification (mandatory)

    # a) propose parameters
    python3 stabilize.py plan --video VIDEO.mp4 --tracks-npz tracks.npz \
        --motion-csv out/motion.csv --out stabilized.mp4 \
        --crop 1024 --shift-x 0 --shift-y 0

    # b) paste the PROPOSED PARAMETERS block to the user verbatim; offer a
    #    preview (--frames 1200) for long videos; get EXPLICIT approval

    # c) render only after approval
    python3 stabilize.py render ...same args... \
        --confirm "<user approval note>"

Parameter guidance:
- crop: output side in px; must be <= min(W,H); leave margin to sensor
  edges for large motions (border-safety check will warn otherwise).
- shift-x/y: move the window center in frame-0 coords (e.g. +150 = 150 px
  right) to frame the region of interest.
- frames: 0=all, or N for a preview of the first N frames.
- crf: 20 default (smaller=better/larger).

Choosing shifts: run `viz` first - the trajectory panel shows where the
camera went; pick a window covering the scene content across time.

## 4. Verify before publishing

- Output frame count == expected (ffprobe -count_frames).
- Template-match a center patch of first vs last frame; residual drift
  should be small (<~1 px per 1000 frames on healthy tracks).
- Border safety OK in plan output; no black edges in corners.

## 5. Publish (bucket protocol, Agent.md)

New folder task_video-stabilize-<variant>-<UTC ts>/ in z81980440/agent_bucket
with the mp4, stabilize.py copy used, llms.txt (objective, params, result,
verification numbers), and update the bucket-root llms.txt.

## Reference implementation

z81980440/agent_bucket task_camera-motion-dense16-2026-08-22T0414Z (tracks ->
motion) and task_video-stabilize-full-2026-08-22T0544Z (full render, all
verification numbers).
