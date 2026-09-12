# Stabilize

Drone video stabilisation using CoTracker3 keypoint trajectories.

Two workflows are supported:

- **Automated (Hugging Face):** `stabilize run` downloads a video from an `hf://` link, runs the full pipeline locally, and uploads every artefact to an HF bucket.
- **Step-by-step:** `track` → `estimate` → `smooth` → `plan` → `render` → `viz` — each as a separate CLI command.

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
2. **track** — CoTracker3 16×16 grid, writes `tracks.npz`
3. **estimate** — RANSAC similarity per frame pair → `motion.npz` + `motion.csv`
4. **smooth** — Gaussian σ=12, interpolates gaps ≤5 frames → `motion_smooth.npz`
5. **viz** — trajectory diagnostics → `trajectory.png`
6. **render** — FFmpeg H.264 CRF20, 1920×1080, smoothed path → `stabilized.mp4`
7. **upload** — pushes everything to `hf://buckets/user/my-stabilized/DJI_0260/`
8. **summary.json** — metadata, params, step timings

Local outputs live at `./runs/DJI_0260/`. The bucket ends up with:

```
hf://buckets/user/my-stabilized/DJI_0260/
  tracks.npz
  motion.npz
  motion.csv
  motion_smooth.npz
  trajectory.png
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
| `--sigma` | Gaussian smoothing strength (frames) | 10 |
| `--no-smooth` | Skip smoothing (track-locked only) | off |
| `--frames` | Render only first N frames (0 = all) | 0 |
| `--crf` | x264 quality (lower = better) | 20 |
| `--skip-upload` | Do everything locally, skip upload | off |
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
stabilize track input.mp4 -o tracks.npz --grid-size 16
```

`--checkpoint` is optional — if omitted (or the file is missing), the
checkpoint auto-downloads from the Hub into `~/.cache/cotracker3/`.

Options: `--checkpoint` (autodownloads if missing), `--grid-size` (max 32), `--grid-query-frame`, `--max-dim 1280`, `--step 8`, `--device cuda:0`.

`--step` sets the online sliding-window stride: the model processes
`window_len = 2*step` frames per call and advances by `step` (default 8 →
window 16). Because overlap is always 50%, total compute is roughly constant;
smaller steps give more frequent, smaller passes (lower peak memory). The
checkpoint's `time_emb` buffer is resized and cached automatically when a
non-default step is used.

---

### `estimate`

```bash
stabilize estimate tracks.npz -o motion
```

Produces `motion.npz` (full-precision 3×3 matrices) and `motion.csv` (human-readable).

Options: `--width 3840 --height 2160 --fps 23.976 --max-points 20000`.

Convention: matrices are IMAGE-motion transforms. Camera translation = `-(tx, ty)`.

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
| `viz.py` | Trajectory diagnostics plot |
| `utils.py` | Video I/O, matrix helpers, shared utils |

---

## Output conventions

- **NPZ matrices** are exact 3×3 homogeneous transforms (cumulative = pairwise product, no rounding).
- **CSV columns** `cum_x`, `cum_y` are in **camera** convention (negated from matrix `[0,2]`/`[1,2]`).
- **Renderer** uses `WARP_INVERSE_MAP` with `INTER_LINEAR` (no black borders from inverse warping).
- **Rectangular output** supported: set `--crop-width` and `--crop-height` independently.
