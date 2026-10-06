# YOLO-OBB on DRASHTI (drone traffic, 14 vehicle classes)

Train the most accurate Ultralytics YOLO OBB model we can on the DRASHTI dataset
(HF bucket `hamimmahmud0/DRASHTI`). **All execution happens on the kaggle-vm (T4 x2), never locally.**

## Dataset facts
- Splits: train 16,571 / val 8,266 / test 2,740 images, 3840x2160 (4K drone frames).
- Labels (`labels/<split>_original/*.txt`): DOTA-like. Line 1 is a header `flightHeight:<m>` (skipped);
  other lines `x1 y1 x2 y2 x3 y3 x4 y4 <class> <difficulty> <extra>` in pixels. Difficulty flag is ignored (all boxes kept).
- 14 classes (val counts): MotorCycle 163.8k, SUV 83.0k, AutoRicksaw 35.8k, Sedan 23.5k, Truck 18.4k, PickUp 13.1k,
  Auto3WCargo 9.3k, Container 8.3k, Bus 7.8k, Van 7.4k, Trailer 7.0k, Tipper 6.7k, Tanker 5.9k, Mixer 3.0k. ~47 objects/image.
- Object size at native 4K (short side of min-area rect, median): MotorCycle 37 px (p10 23), others 70-140 px.
  => halving to 1920 long side is safe; going to 1280 makes motorcycles ~12 px, so imgsz is a real accuracy lever.
- Bucket is readable on the VM without login; there is no HF write token on the VM (download checkpoints via the MCP instead).

## Scripts (all argparse; `--help` on each)
| Script | Purpose |
|---|---|
| `common.py` | class list/order (fixed, alphabetical) |
| `prepare_data.py` | sync bucket, convert labels to normalized OBB, resize to `--max-side`, write `data.yaml` (val = fixed 800-image subset), `data_fullval.yaml` |
| `visualize_labels.py` | overlay converted labels on random images to sanity-check |
| `train.py` | fine-tune from pretrained DOTA OBB weights; `--hours` time budget, `--resume` |
| `notify_tg.py` | VM-side watcher: per-epoch metrics to Telegram (`TG_TOKEN`/`TG_CHAT_ID` env vars, never stored) |
| `infer_tiled.py` | inference on an image folder; resizes long side to 1920, then overlapped 1280 tiles (25% min overlap, edge-cut boxes dropped, rotated NMS) |
| `infer_video_tiled.py` | same tiled inference on videos (first N seconds), writes annotated mp4 + per-frame jsonl |
| `pseudo_label.py` | build an Ultralytics OBB dataset from images using the trained model's tiled predictions |
| `evaluate.py` | full val / test metrics + per-class AP to JSON |

## Workflow (kaggle-vm)
```
upload this dir -> /kaggle/working/yolo-obb-drashti ; pip install ultralytics
nohup python prepare_data.py > prep.log 2>&1 &         # ~/tmp/drashti (disk: /tmp has >1TB, /kaggle/working only 20GB)
python visualize_labels.py                              # download /tmp/viz and eyeball
nohup python train.py --name exp ... > train.log 2>&1 &
python evaluate.py --weights .../best.pt --split test
```

## Model choice (timing probes on 2x T4, imgsz 1280, 5% of train, 1 epoch)
- yolo26x-obb: CUDA OOM at batch 8 (4/GPU). Not feasible on T4.
- yolo26l-obb: 11.6 GB/GPU, ~1.3 it/s at batch 8 => ~26 min/epoch (16.6k imgs) => ~20 epochs in 9 h. **Chosen.**
- Other available pretrained OBB weights: yolo26{n..x}-obb, yolo11{n..x}-obb (DOTA).
- Data prep: images resized to 1920 long side (~33 img/s on 4 CPUs, ~14 min); bucket download ~14 MB/s (~15 min).

## Log
- 2026-10-05: `python train.py --model yolo26l-obb.pt --imgsz 1280 --hours 9 --batch 8 --save-period 2 --name l1280`
  (run dir `/kaggle/working/runs/l1280` on the VM; flipud=0.5, cos LR, close_mosaic=5). Telegram notifier running.
- 2026-10-05: dataset inspected, scripts written; preparing data (max-side 1920).

## Results (run `l1280`: yolo26l-obb, imgsz 1280, 19 epochs, 9 h on 2x T4)
- Weights: private HF bucket **`hamimmahmud0/yolo-obb-drashti-models`** (`l1280/best.pt`, `last.pt`, `results.csv`, `args.yaml`); local copy in `results/l1280/`.
- Full val (8,266 imgs): mAP50 0.995, mAP50-95 0.9907. Test split (2,740 imgs): mAP50 0.995, mAP50-95 0.9905, P 0.999, R 0.999. Weakest class MotorCycle (AP50-95 0.955); all others >= 0.99.
- **Caveat:** these near-saturated numbers are very likely optimistic (random split of frames from the same drone videos / locations => near-duplicate train/val/test). Do not read them as generalisation.
- Out-of-distribution check: `infer_tiled.py` on bucket `hamimmahmud0/SAM_COCO_b1/b1/images` (174 imgs, 4000x3000, oblique/lower altitude views, 4 tiles each, ~0.6 s/img).
  Visual inspection: vehicles on roads are mostly found, but there are many false positives on rooftops/buildings
  (Truck/Trailer/Tipper/Auto3WCargo) and frequent class confusion (vans -> Truck, cars -> Auto3WCargo). Class counts are in `results/l1280/summary.json`.
  Suggested next steps: higher `--conf` (0.4-0.5), fine-tune on the SAM_COCO annotations (`b1/annotations/instances.json`), group-by-video splits.

## Further outputs (bucket `hamimmahmud0/yolo-obb-drashti-models`, private)
- `l1280/` weights + metrics_{val,test}.json + results.csv.
- `results/doyel_chattor_sep3_first20s/`: first 20 s of `DJI_0403_merged.mp4` and `DJI_0406_merged.mp4` (hf dataset `hamimmahmud0/DRINF`,
  `Nadirs/Doyel Chattor/Sep 3, 2026`), 4K@24fps -> 1920x1080 -> 2 overlapped 1280 tiles/frame, conf 0.3, ~5 fps. Annotated mp4 + detections.
  Nadir roundabout footage looks good (vehicles found, MotorCycle dominant); false positives on the metro track / pool edge, misses in the dark clip.
- Pseudo-labelled dataset of the b1 images: private bucket **`hamimmahmud0/b1-pseudo-obb`** (148 train / 26 val, conf>=0.5,
  `python pseudo_label.py --weights best.pt --source <b1 images> --out <dir> --conf 0.5`). Pseudo-labels, not human-verified.
- VM stopped after downloading/uploading everything (2026-10-05).
