"""COCO 1.0 instance-segmentation dataset assembly.

Annotations carry the standard COCO fields (``bbox`` XYWH, ``area``,
``segmentation`` as compressed RLE, ``iscrowd`` 0) plus a ``score`` field
with the detector confidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def rle_encode(mask) -> tuple[dict, int]:
    """Encode a binary uint8/bool HxW mask as COCO compressed RLE.

    Returns ``(segmentation_dict, area)`` where the dict has ``size``
    (``[height, width]``) and an ASCII ``counts`` string.

    Note: the mask area is computed with ``count_nonzero`` instead of
    ``pycocotools.mask.area`` because some pycocotools builds compiled
    against numpy 1.x raise ``OverflowError`` under numpy 2 when re-parsing
    RLE dicts.  ``encode`` itself operates on the array buffer and is safe.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    area = int(np.count_nonzero(mask))
    segmentation = {
        "size": [int(s) for s in rle["size"]],
        "counts": rle["counts"].decode("ascii"),
    }
    return segmentation, area


class CocoBuilder:
    """Accumulates images / annotations / categories into a COCO 1.0 dict."""

    def __init__(self, description: str = "SAM 3.1 auto-annotation"):
        self.description = description
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self.categories: list[dict] = []

    def set_prompts(self, prompts: list[str]) -> None:
        """One COCO category per text prompt, ids 1..N in prompt order."""
        self.categories = [
            {"id": i + 1, "name": p, "supercategory": ""} for i, p in enumerate(prompts)
        ]

    def add_image(self, image_id: int, file_name: str, width, height) -> None:
        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

    def add_annotations(self, items: list[dict]) -> None:
        """Add shard items (``{file_name, annotations: [...]}``)."""
        by_name = {img["file_name"]: img["id"] for img in self.images}
        for item in items:
            image_id = by_name.get(item["file_name"], item.get("image_id"))
            for ann in item.get("annotations", []):
                ann = dict(ann)
                ann["image_id"] = image_id
                ann["id"] = len(self.annotations) + 1
                self.annotations.append(ann)

    def counts_by_category(self) -> dict[str, int]:
        names = {c["id"]: c["name"] for c in self.categories}
        counts: dict[str, int] = {}
        for ann in self.annotations:
            name = names.get(ann["category_id"], str(ann["category_id"]))
            counts[name] = counts.get(name, 0) + 1
        return counts

    def build(self) -> dict:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "info": {
                "description": self.description,
                "version": "1.0",
                "year": int(now[:4]),
                "contributor": "sam31 (SAM 3.1)",
                "date_created": now,
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.build(), fh)
        return path
