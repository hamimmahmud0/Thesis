"""GPU worker process: SAM 3.1 text-prompted detection + segmentation.

One worker process owns one CUDA device.  Each incoming task is a batch of
images; every image becomes a single Datapoint whose ``find_queries`` contain
ALL text prompts, so ONE forward pass produces detections for every concept
on that image.  Results are post-processed, converted to COCO annotations
(compressed RLE masks) and streamed back to the orchestrator.

The inference recipe mirrors the reference SAM3.1 FastAPI server:
fp16 autocast, 1008px square resize, sub-batching with OOM-halving retries.
"""

from __future__ import annotations

import os
from pathlib import Path


def gpu_worker(
    gpu_id: int,
    task_queue,
    result_queue,
    *,
    checkpoint: str,
    prompts: list[str],
    confidence: float,
    input_size: int,
    max_batch: int,
    repo_dir: str | None,
) -> None:
    import torch

    if repo_dir is not None:
        os.chdir(repo_dir)
    torch.cuda.set_device(gpu_id)

    from PIL import Image

    from sam3 import build_sam3_image_model
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )

    from .coco import rle_encode

    device = torch.device(f"cuda:{gpu_id}")
    model = build_sam3_image_model(checkpoint_path=str(checkpoint))
    model = model.to(device).eval()

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=input_size,
                max_size=input_size,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=confidence,
        to_cpu=False,
    )

    result_queue.put({"kind": "worker_ready", "gpu_id": gpu_id})

    n_prompts = len(prompts)

    def _datapoint(rgb_image, qids):
        width, height = rgb_image.size
        queries = [
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=qid,
                    original_image_id=qid,
                    original_category_id=1,
                    original_size=[height, width],
                    object_id=0,
                    frame_index=0,
                ),
            )
            for prompt, qid in zip(prompts, qids)
        ]
        return Datapoint(
            find_queries=queries,
            images=[SAMImage(data=rgb_image, objects=[], size=[height, width])],
        )

    def _annotations_from(result, qids, width, height):
        import numpy as np

        anns = []
        for j, qid in enumerate(qids):
            res = result.get(qid)
            if res is None:
                continue
            boxes = res["boxes"].detach().float().cpu().tolist()
            scores = res["scores"].detach().float().cpu().tolist()
            masks = res["masks"]
            for k in range(len(boxes)):
                if k >= len(masks):
                    break
                mask = masks[k]
                if hasattr(mask, "detach"):
                    mask = mask.detach().cpu().numpy()
                mask = np.asarray(mask)
                if mask.ndim == 3:  # masks come back as [1, H, W]
                    mask = mask[0]
                segmentation, area = rle_encode(mask)

                x1, y1, x2, y2 = boxes[k]
                x1 = min(max(x1, 0.0), float(width))
                y1 = min(max(y1, 0.0), float(height))
                x2 = min(max(x2, 0.0), float(width))
                y2 = min(max(y2, 0.0), float(height))
                anns.append(
                    {
                        "image_id": None,  # filled by the CocoBuilder
                        "category_id": j + 1,
                        "bbox": [
                            round(x1, 2),
                            round(y1, 2),
                            round(max(0.0, x2 - x1), 2),
                            round(max(0.0, y2 - y1), 2),
                        ],
                        "area": area,
                        "iscrowd": 0,
                        "score": round(float(scores[k]), 4),
                        "segmentation": segmentation,
                    }
                )
        return anns

    while True:
        task = task_queue.get()
        if task is None:
            return

        batch_index = task["batch_index"]
        images = task["images"]
        try:
            # ---- Load images and build (transformed) datapoints ----
            entries = []
            failed = []
            for im in images:
                try:
                    with Image.open(im["path"]) as pil:
                        rgb = pil.convert("RGB")
                    qids = [im["index"] * n_prompts + j + 1 for j in range(n_prompts)]
                    entries.append(
                        {
                            "meta": im,
                            "width": rgb.width,
                            "height": rgb.height,
                            "qids": qids,
                            "dp": transform(_datapoint(rgb, qids)),
                        }
                    )
                except Exception as exc:
                    failed.append(
                        {
                            "file_name": im["file_name"],
                            "error": f"load: {type(exc).__name__}: {exc}",
                        }
                    )

            # ---- Forward in OOM-safe sub-batches ----
            items = []
            start = 0
            batch_size = max(1, min(max_batch, len(entries))) if entries else 0
            while start < len(entries):
                chunk = entries[start : start + batch_size]
                try:
                    batch = collate([e["dp"] for e in chunk], dict_key="batch")["batch"]
                    batch = copy_data_to_device(batch, device, non_blocking=True)
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.float16
                    ):
                        output = model(batch)
                    processed = postprocessor.process_results(
                        output, batch.find_metadatas
                    )
                except torch.cuda.OutOfMemoryError as oom:
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        continue
                    # A single image still OOMs: fail just this image.
                    e = chunk[0]
                    failed.append(
                        {
                            "file_name": e["meta"]["file_name"],
                            "error": f"CUDA OOM: {oom}",
                        }
                    )
                    start += 1
                    continue
                finally:
                    torch.cuda.empty_cache()

                for e in chunk:
                    anns = _annotations_from(
                        processed, e["qids"], e["width"], e["height"]
                    )
                    items.append(
                        {
                            "image_id": e["meta"]["index"] + 1,
                            "file_name": e["meta"]["file_name"],
                            "width": e["width"],
                            "height": e["height"],
                            "annotations": anns,
                        }
                    )
                start += batch_size

            result_queue.put(
                {
                    "kind": "batch_done",
                    "batch_index": batch_index,
                    "gpu_id": gpu_id,
                    "items": items,
                    "failed": failed,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "kind": "batch_failed",
                    "batch_index": batch_index,
                    "gpu_id": gpu_id,
                    "file_names": [im["file_name"] for im in images],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def find_sam3_repo() -> Path | None:
    """Locate the sam3 repository root (editable install) if importable."""
    import importlib.util

    spec = importlib.util.find_spec("sam3")
    if spec is None or spec.origin is None:
        return None
    # <repo>/sam3/__init__.py -> repo root
    return Path(spec.origin).resolve().parent.parent
