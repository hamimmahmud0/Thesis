# sam31 — SAM 3.1 Annotator

Text-prompted auto-annotation of image directories into **COCO 1.0
instance-segmentation datasets** using [SAM 3.1](https://github.com/facebookresearch/sam3)
(Segment Anything with Concepts). Give it a folder of images plus one or more
text prompts; every prompt becomes a COCO category, and all prompts for an
image are evaluated in a **single forward pass**. Inference is distributed
across **all CUDA devices**, one worker process per GPU with work stealing.

---

## Installation

```bash
cd tools/sam31
bash setup
```

The script creates a Python 3.12 environment (system-python venv, or conda
when necessary), installs PyTorch CUDA wheels, clones + installs the `sam3`
repo (sibling directory `tools/sam3`), and installs this tool. To always use
a conda env (e.g. on Kaggle-style VMs with a capable system python), run:

```bash
SAM31_FORCE_CONDA=1 bash setup     # env at /root/miniconda3/envs/sam31
```

Verify with:

```bash
sam31 --help
```

On first use the **3.5 GB checkpoint** (`sam3.1_multiplex_mapped.pt`) is
auto-downloaded to `~/.cache/sam31/` from the public HF bucket
`z81980440/agent-tools/SAM3.1`. Prefetch it with `sam31 download`.

---

## Typical workflow

### 1. Extract frames from a video

```bash
sam31 frames "https://huggingface.co/datasets/user/repo/resolve/main/clip.mp4" \
    -o frames/clip --step 150 --limit 48
```

Accepts a local path, `http(s)://` URL, or `hf://` link. URLs are downloaded
once into `~/.cache/sam31/videos/` and reused. `--step N` keeps every Nth
frame, `--limit N` stops after N frames, `--max-side PX` downscales.

### 2. Plan (dry run)

```bash
sam31 plan frames/clip --run clip-v1 \
    -p "pedestrian | car | van | bus | truck | rickshaw | motorcycle | bicycle"
```

Prints image count, prompt→category mapping, batch/device plan, output paths
— no model load.

### 3. Annotate

```bash
sam31 run frames/clip \
    --run clip-v1 \
    -p "pedestrian | car | van | bus | truck | rickshaw | motorcycle | bicycle" \
    --confidence 0.5 --batch-size 4
```

### Output structure

```
runs/clip-v1/
  annotations/instances.json    # COCO 1.0 dataset (compressed-RLE masks)
  images/                       # copy of the images (self-contained dataset)
  shards/batch_00000.jsonl ...  # per-batch results (resume artefacts)
  summary.json                  # params, per-GPU stats, per-category counts
  failures.jsonl                # only when images failed
```

Annotations carry the standard COCO fields (`bbox` XYWH, `area`,
`segmentation` compressed RLE, `iscrowd: 0`) plus a `score` field
(detector confidence).

### Key flags (`run`)

| Flag | Description | Default |
|---|---|---|
| `-p / --prompt` | Text prompt; repeatable; `\|` or `;` separate multiple concepts in one string | required (or `--prompts-file`) |
| `--prompts-file` | One prompt per line (`#` comments) | — |
| `--confidence` | Detection score threshold | 0.5 |
| `--batch-size` | Images per forward pass (auto-halved on OOM) | 4 |
| `--devices` | `cuda:0,cuda:1` / `0,1` / `all` | all CUDA devices |
| `--checkpoint` | Explicit checkpoint path | `~/.cache/sam31/sam3.1_multiplex_mapped.pt` |
| `--limit` | Only first N images (smoke tests) | 0 (all) |
| `--copy-images / --no-copy-images` | Copy images into the run dir | on |
| `--resume` | Continue a partial run from its shards | off |
| `--overwrite` | Discard an existing run folder | off |
| `--bucket` | HF storage bucket to upload the run to | — (no upload) |

Exit code is non-zero if any image failed.

### Tiled high-resolution wrapper

`sam31_tiled.py` keeps each model input at SAM's native 1008×1008 size,
merges overlapping tiles back into the source-image coordinate space, and
removes cross-category duplicate masks:

```bash
python sam31_tiled.py frames/clip \
    --run clip-v1 \
    -p "pedestrian; vehicle; car; bus" \
    --iou-threshold 0.85 \
    --non-vehicle-class "pedestrian; dog; cat"
```

For masks whose IoU reaches `--iou-threshold`, the higher-scoring instance is
kept. A generic `vehicle` instance is always removed in favor of an overlapping
vehicle subclass. Names listed by `--non-vehicle-class` are excluded from the
vehicle-subclass rule; matching is case-insensitive.

---

## Other utilities

```bash
sam31 download                       # prefetch the checkpoint
sam31 upload runs/clip-v1 --bucket user/my-bucket --remote-prefix clip-v1
```

---

## Full file reference

| File | Purpose |
|---|---|
| `cli.py` | Argument parser + command dispatch |
| `runner.py` | Orchestrates scan → workers → shards → COCO merge |
| `worker.py` | GPU worker: SAM 3.1 multi-prompt batched inference (fp16, OOM-safe) |
| `coco.py` | COCO 1.0 builder + RLE encoding |
| `prompts.py` | Prompt parsing (`-p`, file, `\|`/`;` separators) |
| `dataset.py` | Recursive image discovery, natural sort |
| `frames.py` | Video → frames extraction (local/URL/hf://) |
| `checkpoint.py` | Checkpoint resolution + auto-download |
| `hfio.py` | HTTPS downloads, hf:// links, bucket upload |
| `config.py` | Constants and defaults |
