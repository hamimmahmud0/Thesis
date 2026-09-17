# Stabilize

Drone video stabilisation using CoTracker3 keypoint trajectories.

Two workflows are supported:

- **Automated (Hugging Face):** `stabilize run` downloads a video from an `hf://` link, runs the full pipeline locally, and uploads every artefact to an HF bucket.
- **Step-by-step:** `track` → `estimate` → `smooth` → `plan` → `render` → `viz` → `overlay` — each as a separate CLI command.

---

## Installation

```bash
cd /home/hamim-mahmud/Workspace/Thesis/tools/stabilize
pip install -e .
```

Verify:

```bash
stabilize --help
```

### Requirements

- Python 3.10+
- `ffmpeg` available on PATH
- For local tracking: PyTorch (CUDA) + the CoTracker3 checkpoint.  The
  checkpoint (`scaled_online.pth`, ~100 MB) is **auto-downloaded** from the
  Hub (`facebook/cotracker3`) on first use into `~/.cache/cotracker3/`, or
  into the parent directory of an explicitly given `--checkpoint` path.
- For HF workflows: the `hf` CLI (`pip install huggingface-hub[cli]`) + `HF_TOKEN` or `--token`

---

## End-to-end run (`run`)

Download a video from HF, stabilise it, and upload all results to an HF bucket:

```bash
stabilize run \
    hf://datasets/user/drone-raw/video.mp4 \
    --token hf_xxxxxxxx \
    --bucket my-stabilized \
    --run DJI_0260 \
    --checkpoint ./scaled_online.pth \
    --crop-width 1920 \
    --crop-height 1080 \
    --sigma 12
```

### What happens

1. **download** — fetches the video into `./runs/DJI_0260/`
2. **track** — CoTracker3 128×128 fresh grid per anchor, writes `tracks.npz`
3. **estimate** — anchor-relative RANSAC similarity → `motion.npz` + `motion.csv`
4. **drift** — CoTracker drift diagnostic → `motion_drift.npz` + `.csv`
5. **smooth** — Gaussian σ=12 with `--stabilization-mode` (skipped when `off`) → `motion_smooth.npz`
6. **viz** — trajectory diagnostics → `trajectory.png`
7. **overlay** — tracked keypoints drawn on the source video → `tracks_overlay.mp4`
8. **render** — FFmpeg H.264 (CRF 18, visually lossless), track-locked by default → `stabilized.mp4`
9. **upload** — pushes everything to `hf://buckets/user/my-stabilized/DJI_0260/`
10. **summary.json** — metadata, params, step timings

### Drift-free anchor-relative estimation

The camera trajectory is **not** built by composing consecutive frame-to-frame
transforms (`cumulative[t] = pairwise[t] @ cumulative[t-1]`).  That approach
integrates the tiny error of every RANSAC fit and drifts over long videos
(observed: ~150 px over 15 min).

Instead, every frame is fit **directly** against a periodic anchor frame and
placed into the global (frame-0) coordinate system in one step:

```
tracks ──visibility filter──▶ RANSAC similarity ──▶ frame -> local anchor
                                                        │
                                          anchor -> global (once per anchor)
                                                        ▼
                                          continuous global trajectory
                                                        ▼
                             smoothing ──▶ optional locked drift removal
                                                        ▼
                                          correction warp ──▶ render
```

Transform directions (3×3 homogeneous, `p_dst = M @ p_src`):

| Matrix | Maps |
|---|---|
| `local[t]` | frame `t` → segment anchor `K` |
| `anchor_global[i]` | anchor `K_i` → global frame 0 |
| `frame_to_global[t]` | frame `t` → global frame 0 |
| `cumulative[t]` | global frame 0 → frame `t` (renderer convention) |
| `pairwise[t]` | frame `t-1` → frame `t` (diagnostics only) |

Anchor-to-anchor errors therefore accumulate ~once per 10 s, never per frame.
Frames with too few correspondences or a poor RANSAC fit fall back to the
previous valid anchor-relative transform (identity if none), and are flagged
`transform_valid=False` / `fallback_used=True` rather than crashing.

### Stabilization modes

| Mode | Behaviour |
|---|---|
| `off` (default) | Track-lock to frame 0; no smoothing. |
| `natural` | Remove high-frequency jitter; **preserve** legitimate slow pans (no detrend). |
| `locked` | For tripod/static footage: additionally remove a robust low-order (default cubic) translation drift. |

### Fresh-grid + bridge anchors

A single query grid sampled on frame 0 degrades over long videos (occlusion,
points leaving frame, accumulated tracker error, foreground motion, RANSAC
rejection).  The default `fresh-grid` architecture instead gives **every
anchor its own new CoTracker grid**:

```
Anchor A ──fresh grid A──▶ CoTracker segment A ──▶ frame -> A transforms
                                                        │
                                   surviving A points bridge A <-> B
                                                        ▼
Anchor B ──fresh grid B──▶ CoTracker segment B ──▶ frame -> B transforms
```

* **Bridge**: the old grid's tracks are used only to register the new anchor
  (`B -> A`), estimated from **multiple overlap frames** and aggregated
  robustly in (translation, rotation, log-scale) space.
* **Global**: `G_B = G_A @ M_{B->A}` (same convention as the anchor-relative
  estimator), so coordinates never reset.
* **Overlap**: `anchor_overlap_seconds` (default 1.5 s) of shared coverage is
  used for bridging and for smoothstep cross-fading of the transforms.
* **Quality-triggered re-anchoring**: beyond the max interval, a new anchor is
  created early when point survival, RANSAC inlier ratio/count, or *spatial
  coverage* stays poor for `quality_failure_patience_frames`, subject to
  `min_anchor_interval_seconds`.  Spatial coverage is the fraction of occupied
  `4x4` image cells, so 100 points clustered in a corner cannot masquerade as
  a well-conditioned population.
* **Fallback**: if a bridge cannot be estimated, the new anchor is aligned to
  the old segment's own prediction of it — continuity is preserved instead of
  jumping, and the transition is flagged `degraded`.

Per-segment tracks are persisted under `<output>_segments/`, metadata and
per-anchor stats in `<output>.segments.json`, and the flat `motion.npz`
carries per-frame `segment_id`, `anchor_frame`, `anchor_age_frames`,
`num_query_points`, `num_visible_points`, `spatial_coverage`,
`remaining_point_ratio`, `reanchor_requested`/`reanchor_reason`.

Local outputs live at `./runs/DJI_0260/`. The bucket ends up with:

```
hf://buckets/user/my-stabilized/DJI_0260/
  tracks.npz
  motion.npz
  motion.csv
  motion_smooth.npz
  trajectory.png
  tracks_overlay.mp4
  stabilized.mp4
  summary.json
```

### Key flags

| Flag | Description | Default |
|---|---|---|
| `--checkpoint` | CoTracker3 `.pth` — enables local tracking. If missing, auto-downloaded from the Hub | `~/.cache/cotracker3/scaled_online.pth` |
| `--tracks` | Pre-computed tracks file — skips tracking | — |
| `--bucket` | HF bucket to upload to (`user/name`) | *required unless `--skip-upload`* |
| `--run` | Sub-folder name inside the bucket | *required* |
| `--token` | HF token (or set `HF_TOKEN` env var) | — |
| `--crop-width` | Fixed-crop output width | auto black-border |
| `--crop-height` | Fixed-crop output height (square if only width given) | auto black-border |
| `--no-crop` | Stabilise the full source frame (WxH); black borders may appear | off |
| `--crop-black-border` | Auto-detect the largest centred crop that removes the black borders from stabilisation | **on (default)** |
| `--sigma` | Gaussian smoothing strength (frames), used when mode != `off` | 10 |
| `--smooth` | Deprecated alias for `--stabilization-mode natural` | off |
| `--stabilization-mode` | `off` (default) / `natural` / `locked` | off |
| `--anchor-interval-seconds` | Spacing between local anchors (5–30 typical) | 10 |
| `--ransac-reproj-threshold` | RANSAC inlier threshold (px) | 2.0 |
| `--min-correspondences` | Min usable points for a transform | 20 |
| `--min-inlier-ratio` | Min inlier ratio to trust a transform | 0.4 |
| `--anchor-blend-frames` | Cross-fade N frames after each anchor | 0 |
| `--no-drift` | Skip the CoTracker drift diagnostic | off |
| `--debug` | Print per-anchor estimation diagnostics | off |
| `--frames` | Render only first N frames (0 = all) | 0 |
| `--crf` | Output quality: x264 CRF, 0 = mathematically lossless | 18 (visually lossless) |
| `--skip-upload` | Do everything locally, skip upload | off |
| `--skip-overlay` | Skip generating the tracks-overlay video | off |
| `--device` | torch device (`cuda:0`, `cpu`) | auto |

### Tracking from the API instead of locally

If you already have tracks from the CoTracker3 MCP/FastAPI server, skip local tracking entirely:

```bash
stabilize run \
    ./DJI_0260.mp4 \
    --tracks server-tracks.npz \
    --bucket my-stabilized \
    --run DJI_0260 \
    --skip-upload
```

### Crop modes

The output crop mode is chosen as follows (highest priority first):

- **`--no-crop`:** outputs the *full* source frame (W×H) so nothing is thrown
  away.  Because stabilisation warps the frame, black borders can appear at
  the edges where content moved out of view — expected, and this is what
  `--crop-black-border` fixes.
- **`--crop-black-border` (default, on unless `--crop-width`/`--crop-height`
  are given):** auto-detects the largest centred crop window (based on the
  estimated camera motion) that stays inside the source frame for every
  rendered frame, trimming exactly the black borders.
- **Fixed crop:** set `--crop-width` × `--crop-height` for a fixed-size
  output, centred on the frame-0 centre (`--shift-x`/`--shift-y` nudge the
  centre).  This is what you opt into when passing either crop size.

`--no-crop` and `--crop-black-border` are mutually exclusive, and either one
overrides `--crop-width`/`--crop-height`.

---

## Step-by-step commands

### `track`

```bash
stabilize track input.mp4 -o tracks.npz --grid-size 128
```

`--checkpoint` is optional — if omitted (or the file is missing), the
checkpoint auto-downloads from the Hub into `~/.cache/cotracker3/`.

Options: `--checkpoint` (autodownloads if missing), `--grid-size 128` (no cap; GPU memory scales with N²), `--grid-query-frame`, `--max-dim 1280`, `--step 1`, `--device cuda:0`.

`--step` sets the online sliding-window stride: the model processes
`window_len = 2*step` frames per call and advances by `step` (default 1 →
window 2). Because overlap is always 50%, total compute is roughly constant;
smaller steps give more frequent, smaller passes (lower peak memory). The
checkpoint's `time_emb` buffer is resized and cached automatically when a
non-default step is used.

---

### `estimate`

```bash
stabilize estimate tracks.npz -o motion
```

Produces `motion.npz` (full-precision 3×3 matrices) and `motion.csv` (human-readable).

Options: `--width 3840 --height 2160 --fps 23.976 --max-points 20000`,
`--anchor-interval-seconds 10`, `--ransac-reproj-threshold 2.0`,
`--min-correspondences 20`, `--min-inlier-ratio 0.4`,
`--anchor-blend-frames 0`, `--drift`, `--debug`.
`--legacy` reproduces the old drifting frame-to-frame estimator for A/B
comparison only.

Convention: stored matrices are IMAGE-motion transforms. Camera translation
= `-(tx, ty)`.

For a 15-minute comparison of old vs new drift::

    stabilize estimate tracks.npz -o motion_new  --drift --debug
    stabilize estimate tracks.npz -o motion_old  --legacy
    # Compare net_translation in each SUMMARY block; the diagnostic
    # motion_new_drift.csv isolates CoTracker drift from integration drift.

---

### `smooth`

```bash
stabilize smooth motion.npz -o motion_smooth.npz --sigma 12
```

Options: `--sigma` (5=mild, 30=heavy, default 10), `--interp-gap 5` (interpolate short unreliable gaps).

---

### `plan`

Dry-run for `render` — validates inputs, checks border safety, estimates render time and file size:

```bash
stabilize plan input.mp4 \
    --tracks tracks.npz \
    --motion motion.npz \
    --smooth motion_smooth.npz \
    --crop-width 1920 --crop-height 1080 \
    --shift-x 150
```

`plan` and `render` also accept `--no-crop` (full-frame output) and
`--crop-black-border` (auto-trim the black borders) instead of
`--crop-width`/`--crop-height`.

---

### `render`

**Refuses to run without `--confirm`.** Review the `plan` output first, then confirm:

```bash
stabilize render input.mp4 \
    --tracks tracks.npz \
    --motion motion.npz \
    --smooth motion_smooth.npz \
    --crop-width 1920 --crop-height 1080 \
    --output stable.mp4 \
    --confirm "user approved 1920x1080 crop, +150 right"
```

Without `--smooth`, the renderer uses track-locked mode (locks keypoints to frame-0 positions).

---

### `viz`

```bash
stabilize viz motion.npz -o trajectory.png --video-name "DJI_0260"
```

Works with both `.npz` and `.csv` motion files. Red shading marks unreliable regions.

---

### `overlay`

Draws the tracked keypoints on the source video and encodes the result as an MP4:

```bash
stabilize overlay input.mp4 --tracks tracks.npz -o tracks_overlay.mp4
```

Visible points are coloured dots (colour fixed per point id, with a short
trail of recent positions); points the tracker currently considers invisible
are drawn as red crosses. The grid-query frame is marked with hollow cyan
squares at the initial query locations.

Options: `--max-points` (evenly subsample a dense grid), `--radius`, `--trail` (trail length, 0 = off), `--frames`, `--fps`, `--crf`.

---

## `hf://` link format

| Kind | Format |
|---|---|
| Bucket | `hf://buckets/<namespace>/<bucket>/<path>` |
| Dataset | `hf://datasets/<user>/<repo>/<path>` |
| Model | `hf://models/<user>/<repo>/<path>` |
| Space | `hf://spaces/<user>/<repo>/<path>` |

---

## Other utilities

```bash
# Download a single file from HF
stabilize download hf://datasets/user/repo/video.mp4 -o ./downloads/

# Upload a local directory to an HF bucket
stabilize upload ./runs/DJI_0260/ --bucket user/my-bucket --remote-prefix DJI_0260
```

---

## Full file reference

| File | Purpose |
|---|---|
| `cli.py` | Argument parser + command dispatch |
| `runner.py` | Orchestrates the full pipeline |
| `hfio.py` | HF CLI wrappers (download, bucket, upload) |
| `tracker.py` | Local CoTracker3 tracking |
| `motion.py` | RANSAC similarity estimation |
| `smoother.py` | Gaussian path smoothing |
| `renderer.py` | FFmpeg affine-warped rendering |
| `viz.py` | Trajectory diagnostic plot + tracks-overlay video |
| `utils.py` | Video I/O, matrix helpers, shared utils |

---

## Output conventions

- **NPZ matrices** are exact 3×3 homogeneous transforms (cumulative = pairwise product, no rounding).
- **CSV columns** `cum_x`, `cum_y` are in **camera** convention (negated from matrix `[0,2]`/`[1,2]`).
- **Renderer** uses `WARP_INVERSE_MAP` with `INTER_LINEAR` (no black borders from inverse warping).
- **Rectangular output** supported: set `--crop-width` and `--crop-height` independently.
