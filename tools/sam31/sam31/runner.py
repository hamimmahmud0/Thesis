"""End-to-end annotation pipeline orchestrator.

Scans an image directory, spawns one GPU worker per CUDA device, feeds them
batches through a shared queue (work stealing — every worker stays busy
regardless of per-image speed), writes per-batch shard files for crash
resume, and finally merges everything into a COCO 1.0 dataset.
"""

from __future__ import annotations

import json
import queue as queue_mod
import shutil
import threading
import time
from pathlib import Path

from .checkpoint import resolve_checkpoint
from .coco import CocoBuilder
from .config import MODEL_LOAD_TIMEOUT
from .dataset import scan_images
from .prompts import parse_prompts
from .worker import find_sam3_repo, gpu_worker


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_devices(devices_arg: str | None) -> list[int]:
    """Resolve the ``--devices`` flag into a list of CUDA device indices.

    Default: all CUDA devices.  Accepts ``cuda:0,cuda:1``, ``0,1`` or ``all``.
    """
    import torch

    count = torch.cuda.device_count()
    if not devices_arg or devices_arg.strip().lower() == "all":
        if count == 0:
            raise RuntimeError("No CUDA devices found — SAM 3.1 inference needs a GPU.")
        return list(range(count))

    ids = []
    for tok in devices_arg.split(","):
        tok = tok.strip().lower().replace("cuda:", "")
        if not tok:
            continue
        ids.append(int(tok))
    if not ids:
        raise ValueError(f"Could not parse --devices: {devices_arg!r}")
    for i in ids:
        if not 0 <= i < count:
            raise ValueError(
                f"Device cuda:{i} out of range (this host has {count} CUDA device(s))"
            )
    return ids


# ---------------------------------------------------------------------------
# Shard I/O (resume support)
# ---------------------------------------------------------------------------

def _load_shard_lines(shard_dir: Path) -> list[dict]:
    items = []
    if not shard_dir.is_dir():
        return items
    for f in sorted(shard_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _load_done(shard_dir: Path) -> set[str]:
    return {
        item["file_name"]
        for item in _load_shard_lines(shard_dir)
        if item.get("file_name")
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    images_dir: str | Path,
    prompt_args: list[str] | None,
    prompts_file: str | None = None,
    run_name: str,
    out_dir: str | Path = "runs",
    checkpoint: str | Path | None = None,
    devices: str | None = None,
    batch_size: int = 4,
    confidence: float = 0.5,
    input_size: int = 1008,
    copy_images: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    limit: int = 0,
    bucket: str | None = None,
    token: str | None = None,
    private: bool = False,
    model_load_timeout: int = MODEL_LOAD_TIMEOUT,
) -> dict:
    """Annotate *images_dir* with the given prompts into a COCO dataset.

    Returns the run summary dict (also written to ``<run_dir>/summary.json``).
    ``summary["failed_images"]`` is non-empty when some images could not be
    annotated — the CLI turns that into a non-zero exit code.
    """
    t0 = time.time()

    # ---- 1. Scan images / parse prompts / resolve resources ----
    images = scan_images(images_dir)
    if limit > 0:
        images = images[:limit]
    for i, rec in enumerate(images):
        rec.index = i
    if not images:
        raise SystemExit(f"No images found under {images_dir}")

    prompts = parse_prompts(prompt_args, prompts_file)
    cp = resolve_checkpoint(checkpoint, token=token)
    device_ids = resolve_devices(devices)

    run_dir = (Path(out_dir) / run_name).resolve()
    ann_dir = run_dir / "annotations"
    shard_dir = run_dir / "shards"
    images_out = run_dir / "images"
    final_json = ann_dir / "instances.json"

    if final_json.exists() and not (resume or overwrite):
        raise SystemExit(
            f"{final_json} already exists — use --resume to continue or "
            "--overwrite to redo the run"
        )
    if overwrite:
        shutil.rmtree(run_dir, ignore_errors=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {len(images)} images, {len(prompts)} prompts -> {run_dir}")
    print(f"[run] prompts: {', '.join(prompts)}")
    print(f"[run] checkpoint: {cp}")
    print(f"[run] devices: {', '.join(f'cuda:{i}' for i in device_ids)}")

    # ---- 2. Optionally copy images into the run dir (self-contained dataset) ----
    if copy_images:
        images_out.mkdir(parents=True, exist_ok=True)
        for rec in images:
            dest = images_out / rec.rel
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rec.path, dest)
            rec.path = dest
        print(f"[run] images copied to {images_out}")

    # ---- 3. Resume state ----
    done = _load_done(shard_dir) if resume else set()
    pending = [r for r in images if r.rel not in done]
    if done:
        print(f"[resume] {len(done)} image(s) already complete, {len(pending)} pending")

    stats = {
        "images": 0,
        "annotations": 0,
        "failed": [],
        "per_gpu": {},
    }

    # ---- 4. Spawn workers ----
    if pending:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        task_q = ctx.Queue(maxsize=max(4, 4 * len(device_ids)))
        result_q = ctx.Queue()
        repo_dir = find_sam3_repo()

        procs = []
        for gpu in device_ids:
            p = ctx.Process(
                target=gpu_worker,
                args=(gpu, task_q, result_q),
                kwargs={
                    "checkpoint": str(cp),
                    "prompts": prompts,
                    "confidence": confidence,
                    "input_size": input_size,
                    "max_batch": batch_size,
                    "repo_dir": str(repo_dir) if repo_dir else None,
                },
                daemon=True,
            )
            p.start()
            procs.append(p)

        # ---- 5. Wait for model load ----
        ready = set()
        deadline = time.time() + model_load_timeout
        while len(ready) < len(device_ids):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Workers not ready within {model_load_timeout}s "
                    f"(ready: {sorted(ready)})"
                )
            try:
                msg = result_q.get(timeout=10)
            except queue_mod.Empty:
                dead = [gpu for gpu, p in zip(device_ids, procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"Worker(s) on gpu {dead} died while loading the model"
                    )
                continue
            if msg.get("kind") == "worker_ready":
                ready.add(msg["gpu_id"])
                print(f"[worker] cuda:{msg['gpu_id']} model ready", flush=True)

        # ---- 6. Feed batches (work stealing) and collect results ----
        batches = [
            pending[i : i + batch_size] for i in range(0, len(pending), batch_size)
        ]

        def _feeder():
            for bi, batch in enumerate(batches):
                task_q.put(
                    {
                        "batch_index": bi,
                        "images": [
                            {
                                "index": r.index,
                                "path": str(r.path),
                                "file_name": r.rel,
                            }
                            for r in batch
                        ],
                    }
                )
            for _ in procs:
                task_q.put(None)

        feeder = threading.Thread(target=_feeder, daemon=True)
        feeder.start()

        from tqdm import tqdm

        pbar = tqdm(total=len(pending), unit="img", desc="annotate")

        def _handle(msg):
            kind = msg.get("kind")
            if kind == "batch_done":
                shard_file = shard_dir / f"batch_{msg['batch_index']:05d}.jsonl"
                with open(shard_file, "w", encoding="utf-8") as fh:
                    for item in msg["items"]:
                        fh.write(json.dumps(item) + "\n")
                stats["images"] += len(msg["items"])
                n_anns = sum(len(it["annotations"]) for it in msg["items"])
                stats["annotations"] += n_anns
                g = stats["per_gpu"].setdefault(
                    f"cuda:{msg['gpu_id']}", {"images": 0, "annotations": 0}
                )
                g["images"] += len(msg["items"])
                g["annotations"] += n_anns
                pbar.update(len(msg["items"]))
                if msg["failed"]:
                    stats["failed"].extend(msg["failed"])
                    pbar.update(len(msg["failed"]))
            elif kind == "batch_failed":
                stats["failed"].extend(
                    {"file_name": fn, "error": msg["error"]} for fn in msg["file_names"]
                )
                pbar.update(len(msg["file_names"]))

        def _append_failures():
            if not stats["failed"]:
                return
            with open(run_dir / "failures.jsonl", "a", encoding="utf-8") as fh:
                for f in stats["failed"]:
                    fh.write(json.dumps(f) + "\n")

        remaining_batches = len(batches)
        error: Exception | None = None
        while remaining_batches > 0:
            try:
                msg = result_q.get(timeout=15)
            except queue_mod.Empty:
                if not any(p.is_alive() for p in procs):
                    # Workers gone — drain anything still queued, then stop.
                    time.sleep(2)
                    while True:
                        try:
                            _handle(result_q.get(timeout=1))
                            remaining_batches -= 1
                        except queue_mod.Empty:
                            break
                    if remaining_batches > 0:
                        error = RuntimeError(
                            "All workers exited before finishing — "
                            "see worker output above"
                        )
                    break
                continue
            _handle(msg)
            remaining_batches -= 1

        pbar.close()

        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
        _append_failures()
        if error is not None:
            raise error
    else:
        print("[run] nothing pending — merging existing shards")

    # ---- 7. Merge shards into the COCO dataset ----
    from .dataset import read_sizes

    read_sizes(images)
    items_by_name = {
        item["file_name"]: item
        for item in _load_shard_lines(shard_dir)
        if item.get("file_name")
    }

    builder = CocoBuilder()
    builder.set_prompts(prompts)
    for rec in images:
        item = items_by_name.get(rec.rel)
        width = (item or {}).get("width") or rec.width or 0
        height = (item or {}).get("height") or rec.height or 0
        builder.add_image(rec.index + 1, rec.rel, width, height)
    builder.add_annotations(
        [items_by_name[rec.rel] for rec in images if rec.rel in items_by_name]
    )

    out_path = builder.write(final_json)

    # ---- 8. Summary ----
    summary = {
        "run_name": run_name,
        "images_dir": str(images_dir),
        "checkpoint": str(cp),
        "devices": [f"cuda:{i}" for i in device_ids],
        "params": {
            "prompts": prompts,
            "batch_size": batch_size,
            "confidence": confidence,
            "input_size": input_size,
            "copy_images": copy_images,
            "limit": limit,
        },
        "n_images": len(images),
        "n_annotated_images": stats["images"],
        "n_annotations": stats["annotations"],
        "counts_by_category": builder.counts_by_category(),
        "failed_images": stats["failed"],
        "per_gpu": stats["per_gpu"],
        "elapsed_s": round(time.time() - t0, 1),
        "output": str(out_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n===== ANNOTATION SUMMARY =====")
    print(f"images:       {len(images)} ({stats['images']} annotated, "
          f"{len(stats['failed'])} failed)")
    print(f"annotations:  {stats['annotations']}")
    for name, cnt in sorted(builder.counts_by_category().items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24} {cnt}")
    if stats["failed"]:
        print(f"FAILED ({len(stats['failed'])}):")
        for f in stats["failed"][:20]:
            print(f"  {f['file_name']}: {f['error'][:140]}")
    print(f"dataset:      {out_path}")
    print(f"summary:      {run_dir / 'summary.json'}")
    print(f"elapsed:      {summary['elapsed_s']}s")

    # ---- 9. Optional upload ----
    if bucket:
        from . import hfio

        bucket_id = hfio.ensure_bucket(bucket, token=token, private=private)
        print(f"[upload] {run_dir} -> hf://buckets/{bucket_id}/{run_name}/")
        hfio.upload_dir(run_dir, bucket_id, run_name, token=token)

    return summary
