---
name: obb-annotation-loop
description: Run one full round of the oriented-bounding-box (OBB) self-training loop for the Dhaka drone traffic data - export human annotations from every Label Studio project whose title starts with "BEDOBB", tile them, optionally add road-constrained copy-paste synthetic data, fine-tune a YOLO OBB model on Kaggle VMs, run tiled inference on new frames or videos (e.g. DRINF), and upload the model output to Label Studio as pre-annotations for the next annotation round. Use this whenever the user asks to retrain or fine-tune the OBB/YOLO model on the BEDOBB annotations, build synthetic data from annotated images, pre-annotate new frames or videos, extract frames from DRINF/Hugging Face videos for annotation, push model predictions to Label Studio, or start the next annotation round - even if they only name one of those steps.
---

# OBB annotation loop (annotate -> synthesize -> train -> pre-annotate -> re-annotate)

One round turns the annotators' corrections into a better model, and the better model into cheaper pre-annotations for the next batch of frames. The
scripts in `scripts/` were built and verified in round 1 (23 hand-annotated b1 frames -> model -> Label Studio projects for b1 and DRINF frames).
Read `references/gotchas.md` before the first VM command: most of the lost time in round 1 came from those traps.

## Ground rules

- **All execution on Kaggle VMs** (`kaggle-vm` MCP: `start_vm` -> `upload`/`run_command` -> `download` -> `stop_vm`). Local use is for editing files and light API
  calls. Stop every VM once its outputs are downloaded; they burn quota.
- **Secrets are supplied at run time and never written into the repo or memory**: Label Studio refresh token, HF token, Telegram bot token. On a VM, write
  the token to a file with `umask 077`, pass the file path to the scripts, and `shred -u` it when the job ends.
- **Never write into a Label Studio project that people are annotating.** Review output goes into a NEW project whose title starts with `BEDOBB`
  (so the next export picks it up). Check first with `ls_project.py ... list`: annotations > 0 or a second annotator means it is in use.
- **Tile everything by default** (training and inference). Large frames downscaled to model size lose pedestrians (15-25 px).
- **Dense frames: one image can hold ~500 objects.** Keep `max_det` at 1000 (the scripts do; Ultralytics' default of 300 silently truncates), expect
  heavy Label Studio rendering, and watch memory in rotated NMS (it builds an N x N matrix over all tile detections of a frame).
- Report honestly: tiny validation sets are noisy; say when a gain is within noise.
- Optional: send progress and previews to Telegram with `scripts/tg.sh` (needs `TG_BOT_TOKEN`, `TG_CHATS` in the environment; ask the user for them).
  Send 2x2 grids, not very wide strips.

## Inputs to confirm with the user (ask only what is missing)

Label Studio URL + refresh token; HF token; source of new frames (HF dataset repo + path substrings, or a folder) and frame spacing (seconds);
class merge/drop decisions (`references/class_map_round1.json` is what round 1 used - confirm, do not assume); validation block; which model version
goes into the review project.

## The round

Work in `~/work` on the VM (`mkdir -p ~/work` first; fresh VMs lack it). Upload the `scripts/` dir once: `upload` -> `/root/work/s`.
Bootstrap: `pip install -q -U ultralytics huggingface_hub requests` (the tokenizers/hub version warning is harmless).

### 1. Gather ground truth from ALL `BEDOBB*` projects
```
python s/ls_project.py URL tok.txt list --prefix BEDOBB            # see what will be used
python s/ls_export.py URL tok.txt gt --prefix BEDOBB --merge-map map.json --drop Tipper,Trailer --dry-run
python s/ls_export.py URL tok.txt gt --prefix BEDOBB --merge-map map.json --drop Tipper,Trailer
```
Merges every matching project, keeps only human annotations (latest per task; if an image is in several projects the most recently updated annotation wins),
removes classes with 0 instances, writes `gt/{images,labels,classes.txt,manifest.json}`. Read the printed summary: images per project, and how many
annotations **started from a model prediction** - an annotator who pressed Submit unedited has labelled model output as ground truth. Show these numbers to
the user and offer `--skip-unedited-prediction`. Classes with a handful of instances (<~10) cannot be learned or evaluated - say so.

### 2. Split by frame block, never randomly
Consecutive frames show the same street, so hold out whole contiguous blocks (or whole videos/clips): `build_tiles.py --val DJI_0614,DJI_0615,DJI_0616`
takes frame ids (`image stem.split('__')[-1]`). Pick a block with varied classes, print per-class val counts, and report classes with no val instances as
untested (round 1: Special-Purpose-Vehicles sat only in the val block, so it had nothing to train on). If annotation volume allows, rotate the block
(blocked k-fold) for a steadier estimate.

### 3. Tile
`python s/build_tiles.py gt tiles_real --val ...` - 1024 px tiles, 25% overlap, native resolution; an object stays in a tile if >=60% of it is inside
(corners clipped), slivers are dropped, 10% of empty tiles kept. **Draw polygons on 2 tiles and look** before training (see gotchas for a snippet idea).

### 4. Synthetic copy-paste data (optional - measure it)
`python s/synth_paste.py gt tiles_real tiles_synth [--ratio 1.0] [--road-masks DIR]` crops labelled objects from TRAIN frames only and pastes them onto
labelled TRAIN tiles (so no unlabelled real object is ever turned into a hidden negative), colour-matched, <10% overlap, class-balanced toward rare
classes, placed along the traffic direction of a real vehicle and required to sit on road.
- Default road prior = HSV colour heuristic. It leaks onto gray rooftops/bare ground. Look at ~8 tiles (green=real, red=pasted) every time.
- Better: SAM road masks. `kernels/sam` is a Kaggle kernel that batch-runs SAM 3.1 with text prompts over a bucket of images (config `PROMPTS`,
  `SOURCE_BUCKET`, `DEST_BUCKET`; copy `config.yaml.example`, keep its secrets local); `tools/sam31/README.md` documents running `sam31` directly on a VM.
  Prompt `road` on the same frame files, then `coco_to_road_masks.py instances.json masks/` and pass `--road-masks masks`. This path is written but not
  yet exercised end to end - overlay one mask on a frame and check it before trusting it.
- Round 1 result: real+synthetic did not beat real-only beyond noise (class-agnostic F1 0.730 vs 0.737). Treat synthetic as an experiment: train both
  and compare on the held-out frames; keep it only if it wins. More hand labels beat synthetic data.

### 5. Train (detached, both GPUs)
```
setsid nohup python3 s/train_obb.py tiles_real/data.yaml real_only > train_real.log 2>&1 < /dev/null &
```
Run synthetic after it, not alongside (each run holds ~14 GB per GPU). Chain follow-up work with a queue script that waits with `kill -0 <PID>`.
Weights: `runs/obb/runs/<name>/weights/best.pt` (note the extra `obb/`). Sync checkpoints off the VM (HF bucket or download) before stopping it.

### 6. Evaluate fairly
Tile-level mAP from training is noisy and distorted by edge-cut objects. Run tiled inference on the held-out frames and use
`python s/eval_agnostic.py gt DJI_0614,... "NEW=pred.json" "PREV=pred_prev.json" ...` (class-agnostic precision/recall at IoU 0.5 and 0.3; class names
differ across models, so this is the only like-for-like number). Compare against the previous round's model, not just against satellite models.

### 7. Pre-annotate new frames
```
python s/extract_frames.py frames --repo hamimmahmud0/DRINF --include "Doyel Chattor,Polashi Bazaar" --step 35     # streams, no full download
python s/tile_infer.py best.pt frames pred.json --device 0                   # 1024 tiles, 25% overlap, rotated NMS 0.5, conf 0.25, max_det 1000
```
`--step` is seconds between frames (user said "1/20 fps"= every 20 s, later changed to 35 s: confirm). Draw predictions on one frame per site and look: expect
rooftop false positives and missed vehicles under trees; if the scale/viewpoint differs from training (nadir video vs oblique stills) say so.
Keep the same NMS threshold for every model you compare (0.3 suppressed adjacent vehicles in round 1; 0.5 still removes cross-tile duplicates).

### 8. Upload to Label Studio as a NEW project
```
python s/ls_project.py URL tok.txt create "BEDOBB - <source> r<N> (pre-annotated)" --labels-from <BEDOBB project id> --model-version <tag>
python s/ls_import.py URL <new id> tok.txt frames gt_empty pred.json map.json --model-version <tag>
```
(`gt_empty` = a dir with empty `images/ labels/ classes.txt` when no frame has manual labels; with GT, frames present in `gt` become annotations and the
rest become predictions.) Title <=50 chars. The importer uploads each image, writes rotated rectangles (verified against Label Studio's own YOLO_OBB
export, error 0.002 px) and attaches predictions with per-box confidence. Verify via API (tasks / predictions counts, an image fetch returns 200), then
`shred -u` the token, download `pred.json`, the task map and previews locally, and stop the VM.
Label sets must match across `BEDOBB*` projects, or the next export will see mixed names; `ls_project.py add-label` extends a config safely.

## What to tell the user at the end

Models and metrics on held-out frames (with the caveat on size), what was merged/dropped, counts per project, where outputs are, anything unchecked
(SAM road masks, unreviewed prediction-started annotations), and that VMs are stopped.
