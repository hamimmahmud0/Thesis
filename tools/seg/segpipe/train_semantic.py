from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from .config import load_config
from .data import CocoSemanticDataset
from .models import build_model
from .notify import Notifier
from .notify import Notifier
from .storage import HubStore

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True); args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    config = load_config(args.config); spec = next(x for x in config.models if x.name == args.model)
    notifier = Notifier.from_config(config)
    if torch.cuda.device_count() < 2: raise RuntimeError("Two CUDA GPUs are required")
    rank = int(os.environ["LOCAL_RANK"]); dist.init_process_group("nccl"); torch.cuda.set_device(rank)
    store = HubStore(config.hf_repo, config.hf_bucket, config.hf_token, config.work_dir)
    notifier = Notifier.from_config(config)
    classes = len(__import__("json").loads((args.dataset / "classes.json").read_text())) + 1
    model = build_model(spec, classes).cuda(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    checkpoint = store.artifacts / "models" / spec.name / "last.pt"; start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=f"cuda:{rank}", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = saved["epoch"]
    model = DistributedDataParallel(model, device_ids=[rank])
    dataset = CocoSemanticDataset(args.dataset, "train", spec.image_size)
    sampler = DistributedSampler(dataset, shuffle=True); loader = DataLoader(dataset, batch_size=spec.batch,
        sampler=sampler, num_workers=spec.workers, pin_memory=True, persistent_workers=spec.workers > 0)
    completed_epoch = start
    for epoch in range(start, spec.epochs):
        sampler.set_epoch(epoch); model.train()
        for images, masks in loader:
            images, masks = images.cuda(rank, non_blocking=True), masks.cuda(rank, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); output = model(images)
            logits = output.logits if hasattr(output, "logits") else output
            if logits.shape[-2:] != masks.shape[-2:]: logits = F.interpolate(logits, masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, masks); loss.backward(); optimizer.step()
        dist.barrier()
        if rank == 0:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}, checkpoint)
            store.update_model(spec.name, status="training", epoch=epoch + 1, epochs=spec.epochs); store.sync()
            print(f"EPOCH {spec.name} {epoch + 1}/{spec.epochs}", flush=True)
            notifier.major(spec.name, f"epoch {epoch + 1}/{spec.epochs} checkpoint saved")
        dist.barrier()
        completed_epoch = epoch + 1
        if time.time() >= args.deadline - config.handoff_minutes * 60: break
    if rank == 0 and completed_epoch >= spec.epochs:
        store.update_model(spec.name, status="complete", epoch=completed_epoch, epochs=spec.epochs); store.sync()
    dist.destroy_process_group(); return 0
if __name__ == "__main__": raise SystemExit(main())
