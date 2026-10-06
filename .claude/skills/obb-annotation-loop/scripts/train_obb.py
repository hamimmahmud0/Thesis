"""Fine-tune a YOLO OBB model on a tiled dataset (data.yaml from build_tiles.py / synth_paste.py).
python train_obb.py DATA_YAML NAME [--weights yolo26l-obb.pt] [--epochs 120] [--patience 40] [--imgsz 1024] [--batch 16] [--devices 0,1] [--no-val]
Launch detached so it survives the tool call:  setsid nohup python3 train_obb.py data.yaml real_only > train.log 2>&1 < /dev/null &
Results land in runs/obb/runs/<NAME>/ (Ultralytics prepends 'obb/' to a relative project dir - sync that path, not runs/<NAME>).
Recipe notes: AdamW lr 5e-4 + cosine, flips on, rotation off (nadir tiles have no preferred up), scale 0.25 keeps 15-25 px pedestrians visible,
max_det 1000 because a single image can hold ~500 objects (default 300 silently truncates validation/inference on dense tiles).
Pretrained DOTA weights beat starting from the satellite fine-tunes (VEDAI/DIOR-R) and keep comparisons against those fair."""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('data'); ap.add_argument('name'); ap.add_argument('--weights', default='yolo26l-obb.pt')
ap.add_argument('--epochs', type=int, default=120); ap.add_argument('--patience', type=int, default=40)
ap.add_argument('--imgsz', type=int, default=1024); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--devices', default='0,1'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--no-val', action='store_true', help='validate only at the end (use for large val sets: val every epoch dominates the time)')
a = ap.parse_args()
YOLO(a.weights).train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch, device=a.devices, workers=4,
                      project='runs', name=a.name, exist_ok=True, optimizer='AdamW', lr0=5e-4, cos_lr=True, warmup_epochs=3, close_mosaic=15,
                      flipud=0.5, fliplr=0.5, scale=0.25, mixup=0.0, max_det=1000, val=not a.no_val, plots=True, seed=a.seed)
