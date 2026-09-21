from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from .config import load_config
from .notify import Notifier
from .notify import Notifier
from .storage import HubStore

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True); args = parser.parse_args(argv)
    config = load_config(args.config); spec = next(x for x in config.models if x.name == args.model)
    notifier = Notifier.from_config(config)
    store = HubStore(config.hf_repo, config.hf_bucket, config.hf_token, config.work_dir)
    notifier = Notifier.from_config(config)
    run_dir = store.artifacts / "models" / spec.name; last = run_dir / "weights" / "last.pt"
    import torch
    if torch.cuda.device_count() < 2: raise RuntimeError("Two CUDA GPUs are required")
    from ultralytics import YOLO
    model = YOLO(str(last if last.exists() else f"{spec.name}.pt"))
    def checkpoint(trainer):
        epoch = int(trainer.epoch) + 1
        store.update_model(spec.name, status="training", epoch=epoch, epochs=spec.epochs)
        store.sync()
        print(f"EPOCH {spec.name} {epoch}/{spec.epochs}", flush=True)
        notifier.major(spec.name, f"epoch {epoch}/{spec.epochs} checkpoint saved")
    def deadline(trainer):
        if time.time() >= args.deadline - config.handoff_minutes * 60: trainer.stop = True
    model.add_callback("on_model_save", checkpoint); model.add_callback("on_train_epoch_end", deadline)
    kwargs = dict(data=str(args.dataset / "dataset.yaml"), epochs=spec.epochs, batch=spec.batch,
        imgsz=spec.image_size, workers=spec.workers, save_period=spec.save_period, device=[0, 1],
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True, resume=last.exists())
    kwargs.update(spec.extra); model.train(**kwargs)
    # A deadline stop is resumable and not completion.
    trainer_epoch = int(getattr(model.trainer, "epoch", -1)) + 1
    if trainer_epoch >= spec.epochs:
        store.update_model(spec.name, status="complete", epoch=trainer_epoch, epochs=spec.epochs); store.sync()
    return 0
if __name__ == "__main__": raise SystemExit(main())
