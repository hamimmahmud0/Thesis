from __future__ import annotations
import json
from pathlib import Path

class CocoSemanticDataset:
    def __init__(self, root: Path, split: str, size: int):
        import torch
        from PIL import Image
        self.torch, self.Image, self.root, self.size = torch, Image, root, size
        self.files = sorted((root / "images" / split).glob("*"))
        coco = json.loads((root / "instances.json").read_text(encoding="utf-8"))
        self.images = {Path(x["file_name"]).name: x for x in coco["images"]}
        self.anns = {}
        for ann in coco["annotations"]: self.anns.setdefault(int(ann["image_id"]), []).append(ann)
        cats = sorted(int(x["id"]) for x in coco["categories"]); self.cat = {v: i + 1 for i, v in enumerate(cats)}
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        import numpy as np
        from PIL import ImageDraw
        path = self.files[index]; item = self.images[path.name]
        image = self.Image.open(path).convert("RGB").resize((self.size, self.size))
        mask = self.Image.new("L", image.size, 0); draw = ImageDraw.Draw(mask)
        sx, sy = self.size / int(item["width"]), self.size / int(item["height"])
        for ann in self.anns.get(int(item["id"]), []):
            segmentation = ann.get("segmentation")
            if isinstance(segmentation, list):
                for poly in segmentation:
                    if len(poly) >= 6: draw.polygon([(poly[i] * sx, poly[i+1] * sy) for i in range(0, len(poly), 2)], fill=self.cat[int(ann["category_id"])])
            elif isinstance(segmentation, dict):
                from pycocotools import mask as mask_utils
                rle = segmentation
                if isinstance(rle.get("counts"), list): rle = mask_utils.frPyObjects(rle, int(item["height"]), int(item["width"]))
                binary = mask_utils.decode(rle).astype("uint8") * self.cat[int(ann["category_id"])]
                layer = self.Image.fromarray(binary, mode="L").resize((self.size, self.size), resample=self.Image.Resampling.NEAREST)
                mask.paste(layer, mask=layer.point(lambda value: 255 if value else 0))
        image_array = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255
        return self.torch.from_numpy(image_array), self.torch.from_numpy(np.asarray(mask, dtype="int64").copy())
