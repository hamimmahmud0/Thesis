from __future__ import annotations
import json
import random
import shutil
from pathlib import Path
from .errors import PipelineError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def deterministic_split(ids: list[int], train: float, val: float, seed: int) -> dict[str, set[int]]:
    values = list(ids); random.Random(seed).shuffle(values); count = len(values)
    a, b = int(count * train), int(count * (train + val))
    return {"train": set(values[:a]), "val": set(values[a:b]), "test": set(values[b:])}

def _find_annotation(source: Path) -> Path:
    candidates = sorted(source.rglob("*.json"))
    for path in candidates:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if {"images", "annotations", "categories"} <= raw.keys(): return path
        except (ValueError, OSError): pass
    raise PipelineError("source_not_found", "No COCO instances JSON found in source repository")

def _clean_polygon(segmentation, width: int, height: int) -> list[list[float]]:
    if not isinstance(segmentation, list): return []
    clean = []
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2: continue
        points = []
        for x, y in zip(polygon[::2], polygon[1::2]):
            points.extend((min(max(float(x) / width, 0), 1), min(max(float(y) / height, 0), 1)))
        if len(set(zip(points[::2], points[1::2]))) >= 3: clean.append(points)
    return clean

def _rle_polygons(segmentation, width: int, height: int) -> list[list[float]]:
    if not isinstance(segmentation, dict): return []
    try:
        import cv2
        from pycocotools import mask as mask_utils
        rle = segmentation
        if isinstance(rle.get("counts"), list): rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle).astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) >= 3:
                polygons.append([coordinate for x, y in points for coordinate in
                    (min(max(float(x) / width, 0), 1), min(max(float(y) / height, 0), 1))])
        return polygons
    except Exception as exc:
        raise PipelineError("invalid_annotation", f"Could not decode COCO RLE segmentation: {exc}") from exc

def convert_coco(source: Path, target: Path, train: float, val: float, seed: int) -> Path:
    annotation_path = _find_annotation(source); coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in coco["images"]}
    file_index = {p.name: p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    discovered = {name: set() for name in ("train", "val", "test")}
    for image_id, item in images.items():
        path = file_index.get(Path(item["file_name"]).name)
        if path:
            for split in discovered:
                if split in {part.lower() for part in path.parts}: discovered[split].add(image_id); break
    discovered_ids = set().union(*discovered.values())
    split_memberships = sum(len(ids) for ids in discovered.values())
    has_complete_split = (
        all(discovered.values())
        and discovered_ids == set(images)
        and split_memberships == len(discovered_ids)
    )
    splits = discovered if has_complete_split else deterministic_split(list(images), train, val, seed)
    category_ids = sorted(int(c["id"]) for c in coco["categories"]); class_index = {v: i for i, v in enumerate(category_ids)}
    annotations = {}
    for ann in coco["annotations"]: annotations.setdefault(int(ann["image_id"]), []).append(ann)
    for split, ids in splits.items():
        (target / "images" / split).mkdir(parents=True, exist_ok=True); (target / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            item = images[image_id]; filename = item["file_name"]
            src = file_index.get(Path(filename).name)
            if src is None or src.suffix.lower() not in IMAGE_EXTENSIONS: continue
            shutil.copy2(src, target / "images" / split / Path(filename).name)
            lines = []
            for ann in annotations.get(image_id, []):
                segmentation = ann.get("segmentation")
                polygons = _clean_polygon(segmentation, int(item["width"]), int(item["height"]))
                polygons += _rle_polygons(segmentation, int(item["width"]), int(item["height"]))
                for polygon in polygons:
                    lines.append(str(class_index[int(ann["category_id"])]) + " " + " ".join(f"{v:.6f}" for v in polygon))
            (target / "labels" / split / f"{Path(filename).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    names = [next(str(c["name"]) for c in coco["categories"] if int(c["id"]) == cid) for cid in category_ids]
    (target / "classes.json").write_text(json.dumps(names), encoding="utf-8")
    yaml_text = "path: " + str(target) + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml_text += "".join(f"  {i}: {json.dumps(name)}\n" for i, name in enumerate(names))
    (target / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    # Semantic masks are generated lazily by the semantic dataset to avoid duplicating large images.
    shutil.copy2(annotation_path, target / "instances.json")
    (target / ".prepared").write_text("1", encoding="ascii")
    return target

def prepare_dataset(config, store, dry_run=False) -> Path:
    prepared = config.work_dir / "dataset"
    if (prepared / ".prepared").exists(): return prepared
    if dry_run:
        prepared.mkdir(parents=True, exist_ok=True); return prepared
    source = store.download_source(config.work_dir / "source")
    return convert_coco(source, prepared, config.train_ratio, config.val_ratio, config.seed)
