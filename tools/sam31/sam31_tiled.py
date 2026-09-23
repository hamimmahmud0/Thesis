#!/usr/bin/env python3
"""Tiled wrapper for `sam31 run`.

Mirrors the public `sam31 run` CLI while running the model only at its native
1008x1008 resolution. Large source images are split into overlapping tiles,
SAM runs on the tiles, and the resulting COCO RLE masks are merged back into
original-image coordinates.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

SAM_SIZE = 1008
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Instance:
    category_id: int
    score: float
    x: int
    y: int
    mask: np.ndarray

    @property
    def h(self) -> int:
        return int(self.mask.shape[0])

    @property
    def w(self) -> int:
        return int(self.mask.shape[1])

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sam31_tiled.py",
        description=(
            "Drop-in tiled wrapper for `sam31 run`: split source images into "
            "overlapping 1008x1008 tiles, run SAM 3.1, merge masks back into "
            "the original image coordinates, and optionally upload the final "
            "merged run to an HF Storage Bucket."
        ),
    )

    # Keep the sam31 run interface/order as closely as possible.
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory containing the images (searched recursively).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        dest="prompts",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Text prompt (concept) to detect and segment — repeatable. "
            "Passed through to sam31."
        ),
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        metavar="FILE",
        help="File with one prompt per line; passed through to sam31.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        metavar="F",
        help="Detection score threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="Images/tiles per SAM forward pass per worker. Default: 4.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=SAM_SIZE,
        metavar="PX",
        help=(
            "Accepted for sam31 CLI compatibility. The tiled wrapper always "
            "runs the inner SAM model at 1008. Values such as 2016/3024 "
            "therefore select tiled high-resolution processing rather than "
            "native SAM inference at that size."
        ),
    )
    parser.add_argument(
        "--devices",
        default="all",
        metavar="LIST",
        help="CUDA devices, e.g. 'cuda:0,cuda:1', '0,1', or 'all'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        metavar="CKPT.pt",
        help="SAM 3.1 checkpoint path; passed through to sam31.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Annotate only the first N SOURCE images (0 = all). This is "
            "intentionally applied before tiling, not passed to the tile run."
        ),
    )
    parser.add_argument(
        "--token",
        metavar="hf_xxx",
        help=(
            "Hugging Face token. Passed to sam31 for checkpoint access and "
            "used for final HF Storage Bucket creation/upload."
        ),
    )
    parser.add_argument(
        "--run",
        required=True,
        metavar="NAME",
        help="Run folder name (outputs land in <out-dir>/<run>/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs"),
        metavar="DIR",
        help="Local parent directory for run folders. Default: ./runs",
    )

    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument(
        "--copy-images",
        dest="copy_images",
        action="store_true",
        help="Copy original source images into <run>/images/. Default: on.",
    )
    copy_group.add_argument(
        "--no-copy-images",
        dest="copy_images",
        action="store_false",
        help="Do not copy original source images into the final run.",
    )
    parser.set_defaults(copy_images=True)

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the persistent temporary tile run. The wrapper keeps "
            "<out-dir>/.sam31_tiled_work/<run>/ until a successful run unless "
            "--keep-workdir is specified."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing final and temporary run state and start over.",
    )
    parser.add_argument(
        "--bucket",
        metavar="user/bucket",
        help=(
            "HF Storage Bucket to upload the FINAL merged run to. The wrapper "
            "uploads to hf://buckets/<bucket>/<run>/, never the temporary tiles."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HF bucket as private if it has to be created.",
    )

    # Wrapper-only arguments. Prefix them with tile/merge names so they do not
    # collide with current sam31 flags.
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        metavar="F",
        help="Fractional overlap between 1008x1008 tiles. Default: 0.25.",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.35,
        metavar="F",
        help="Same-category mask IoU threshold for merging duplicates.",
    )
    parser.add_argument(
        "--merge-containment",
        type=float,
        default=0.60,
        metavar="F",
        help="Intersection/smaller-area threshold for merging duplicates.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.85,
        metavar="F",
        help=(
            "Mask IoU threshold for removing duplicate final instances across "
            "categories. The highest-scoring instance is kept, except a generic "
            "'vehicle' always loses to an overlapping vehicle subclass. "
            "Default: 0.85."
        ),
    )
    parser.add_argument(
        "--non-vehicle-class",
        default="pedestrian; dog; cat",
        metavar="NAMES",
        help=(
            "Semicolon- or pipe-separated category names that are not vehicle "
            "subclasses. Matching is case-insensitive. "
            "Default: 'pedestrian; dog; cat'."
        ),
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep generated tiles and raw tile-level SAM output after success.",
    )

    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    if args.private and not args.bucket:
        parser.error("--private requires --bucket")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not (0.0 <= args.tile_overlap < 1.0):
        parser.error("--tile-overlap must be >= 0 and < 1")
    if not (0.0 <= args.merge_iou <= 1.0):
        parser.error("--merge-iou must be between 0 and 1")
    if not (0.0 <= args.merge_containment <= 1.0):
        parser.error("--merge-containment must be between 0 and 1")
    if not (0.0 <= args.iou_threshold <= 1.0):
        parser.error("--iou-threshold must be between 0 and 1")

    return args


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_images(
    root: Path,
    out_root: Path,
    work_root: Path,
    limit: int,
) -> list[Path]:
    images: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        # Important for `... images_dir=.`: never re-ingest previous outputs.
        if is_under(path, out_root) or is_under(path, work_root):
            continue
        images.append(path)

    images.sort()
    if limit > 0:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No images found under {root}")
    return images


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def make_tiles(
    images: list[Path],
    root: Path,
    tile_dir: Path,
    overlap: float,
) -> tuple[list[dict], dict[str, dict], int]:
    stride = max(1, int(round(SAM_SIZE * (1.0 - overlap))))
    mapping: dict[str, dict] = {}
    originals: list[dict] = []
    tile_dir.mkdir(parents=True, exist_ok=True)

    for original_id, source in enumerate(images, start=1):
        with Image.open(source) as image:
            width, height = image.size
            relative = source.resolve().relative_to(root.resolve())
            originals.append(
                {
                    "id": original_id,
                    "source": str(source.resolve()),
                    "relative": relative.as_posix(),
                    "width": width,
                    "height": height,
                }
            )

            xs = tile_positions(width, SAM_SIZE, stride)
            ys = tile_positions(height, SAM_SIZE, stride)
            tile_index = 0

            for y in ys:
                for x in xs:
                    valid_w = min(SAM_SIZE, width - x)
                    valid_h = min(SAM_SIZE, height - y)
                    crop = image.crop((x, y, x + valid_w, y + valid_h)).convert("RGB")

                    # Keep every model input exactly 1008x1008. Edge crops are
                    # padded and the padded area is removed again after decode.
                    canvas = Image.new("RGB", (SAM_SIZE, SAM_SIZE), (0, 0, 0))
                    canvas.paste(crop, (0, 0))

                    name = (
                        f"tile_o{original_id:06d}_t{tile_index:05d}"
                        f"_x{x:06d}_y{y:06d}.png"
                    )
                    tile_path = tile_dir / name
                    canvas.save(tile_path, compress_level=1)

                    mapping[name] = {
                        "original_id": original_id,
                        "x": x,
                        "y": y,
                        "valid_w": valid_w,
                        "valid_h": valid_h,
                    }
                    tile_index += 1

    (tile_dir.parent / "tile_mapping.json").write_text(
        json.dumps(
            {"sam_size": SAM_SIZE, "stride": stride, "tiles": mapping},
            indent=2,
        ),
        encoding="utf-8",
    )
    return originals, mapping, stride


def redact_command(command: list[str]) -> str:
    result: list[str] = []
    hide_next = False
    for item in command:
        if hide_next:
            result.append("***")
            hide_next = False
            continue
        result.append(item)
        if item == "--token":
            hide_next = True
    return " ".join(result)


def run_sam31(
    args: argparse.Namespace,
    tile_dir: Path,
    raw_out_parent: Path,
) -> Path:
    command = [
        sys.executable,
        "-m",
        "sam31.cli",
        "run",
        str(tile_dir),
    ]

    for prompt in args.prompts:
        command += ["-p", prompt]

    if args.prompts_file is not None:
        command += ["--prompts-file", str(args.prompts_file.resolve())]

    command += [
        "--confidence",
        str(args.confidence),
        "--batch-size",
        str(args.batch_size),
        # Deliberately intercepted: SAM itself only receives its native size.
        "--input-size",
        str(SAM_SIZE),
        "--devices",
        args.devices,
    ]

    if args.checkpoint is not None:
        command += ["--checkpoint", str(args.checkpoint.expanduser().resolve())]
    if args.token:
        command += ["--token", args.token]

    command += [
        "--run",
        "tiles",
        "--out-dir",
        str(raw_out_parent),
        # The tiles are already present in our work directory; copying them
        # into another temporary images/ folder only wastes space.
        "--no-copy-images",
    ]

    tile_run_dir = raw_out_parent / "tiles"
    if args.resume and tile_run_dir.exists():
        command.append("--resume")
    elif tile_run_dir.exists():
        # This should normally only happen with stale state. We only destroy it
        # automatically when the top-level user requested --overwrite.
        raise SystemExit(
            f"Temporary SAM run already exists: {tile_run_dir}\n"
            "Use --resume to continue it or --overwrite to start over."
        )

    print(f"[sam31] {redact_command(command)}")
    subprocess.run(command, check=True)

    coco_path = tile_run_dir / "annotations" / "instances.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"SAM output not found: {coco_path}")
    return coco_path


def require_pycocotools() -> None:
    if mask_utils is None:
        raise SystemExit(
            "pycocotools is required for COCO RLE merge/re-encoding. Install it with:\n"
            "  pip install pycocotools"
        )


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    require_pycocotools()
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_utils.decode(rle)
    elif isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(mask_utils.merge(rles))
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)!r}")

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool, copy=False)


def annotation_to_instance(
    annotation: dict,
    tile_meta: dict,
    tile_height: int,
    tile_width: int,
) -> Instance | None:
    mask = decode_segmentation(annotation["segmentation"], tile_height, tile_width)
    mask = mask[: tile_meta["valid_h"], : tile_meta["valid_w"]]
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    tight = mask[y0:y1, x0:x1].copy()
    score = annotation.get("score", annotation.get("confidence", 1.0))

    return Instance(
        category_id=int(annotation["category_id"]),
        score=float(score),
        x=int(tile_meta["x"] + x0),
        y=int(tile_meta["y"] + y0),
        mask=tight,
    )


def mask_intersection(a: Instance, b: Instance) -> int:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x1 >= x2 or y1 >= y2:
        return 0

    ma = a.mask[y1 - a.y : y2 - a.y, x1 - a.x : x2 - a.x]
    mb = b.mask[y1 - b.y : y2 - b.y, x1 - b.x : x2 - b.x]
    return int(np.logical_and(ma, mb).sum())


def overlap_metrics(a: Instance, b: Instance) -> tuple[float, float]:
    intersection = mask_intersection(a, b)
    if intersection == 0:
        return 0.0, 0.0

    area_a, area_b = a.area, b.area
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    smaller = min(area_a, area_b)
    containment = intersection / smaller if smaller else 0.0
    return iou, containment


def union_instances(a: Instance, b: Instance) -> Instance:
    x0, y0 = min(a.x, b.x), min(a.y, b.y)
    x1 = max(a.x + a.w, b.x + b.w)
    y1 = max(a.y + a.h, b.y + b.h)
    merged = np.zeros((y1 - y0, x1 - x0), dtype=bool)

    merged[a.y - y0 : a.y - y0 + a.h, a.x - x0 : a.x - x0 + a.w] |= a.mask
    merged[b.y - y0 : b.y - y0 + b.h, b.x - x0 : b.x - x0 + b.w] |= b.mask

    return Instance(
        category_id=a.category_id,
        score=max(a.score, b.score),
        x=x0,
        y=y0,
        mask=merged,
    )


def deduplicate_instances(
    instances: list[Instance],
    iou_threshold: float,
    containment_threshold: float,
) -> list[Instance]:
    instances = sorted(instances, key=lambda item: item.score, reverse=True)
    kept: list[Instance] = []

    for candidate in instances:
        best_index = None
        best_metric = 0.0

        for index, previous in enumerate(kept):
            if candidate.category_id != previous.category_id:
                continue
            iou, containment = overlap_metrics(candidate, previous)
            if iou < iou_threshold and containment < containment_threshold:
                continue
            metric = max(iou, containment)
            if metric > best_metric:
                best_metric = metric
                best_index = index

        if best_index is None:
            kept.append(candidate)
        else:
            kept[best_index] = union_instances(kept[best_index], candidate)

    return kept


def parse_category_names(value: str) -> set[str]:
    """Return normalized category names from a semicolon/pipe-separated value."""
    return {
        part.strip().casefold()
        for part in re.split(r"[;|]", value)
        if part.strip()
    }


def suppress_cross_category_duplicates(
    instances: list[Instance],
    categories: dict[int, str],
    iou_threshold: float,
    non_vehicle_categories: set[str],
) -> tuple[list[Instance], int, int]:
    """Suppress highly overlapping final instances across categories.

    Generic ``vehicle`` masks are removed when they overlap a vehicle subclass,
    even when the generic mask has the higher score. All remaining overlaps use
    greedy score-based mask NMS, so only the highest-confidence instance remains.
    """
    if len(instances) < 2:
        return instances, 0, 0

    names = {
        category_id: name.strip().casefold()
        for category_id, name in categories.items()
    }
    generic_vehicle_ids = {
        category_id for category_id, name in names.items() if name == "vehicle"
    }
    vehicle_subclass_ids = {
        category_id
        for category_id, name in names.items()
        if name and name != "vehicle" and name not in non_vehicle_categories
    }

    removed_generic: set[int] = set()
    for generic_index, generic in enumerate(instances):
        if generic.category_id not in generic_vehicle_ids:
            continue
        for other_index, other in enumerate(instances):
            if other_index == generic_index:
                continue
            if other.category_id not in vehicle_subclass_ids:
                continue
            iou, _ = overlap_metrics(generic, other)
            if iou >= iou_threshold:
                removed_generic.add(generic_index)
                break

    candidates = [
        instance
        for index, instance in enumerate(instances)
        if index not in removed_generic
    ]
    candidates.sort(key=lambda item: item.score, reverse=True)

    kept: list[Instance] = []
    score_suppressed = 0
    for candidate in candidates:
        duplicate = False
        for previous in kept:
            iou, _ = overlap_metrics(candidate, previous)
            if iou >= iou_threshold:
                duplicate = True
                score_suppressed += 1
                break
        if not duplicate:
            kept.append(candidate)

    return kept, len(removed_generic), score_suppressed


def encode_full_rle(instance: Instance, height: int, width: int) -> dict:
    require_pycocotools()
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[
        instance.y : instance.y + instance.h,
        instance.x : instance.x + instance.w,
    ] = instance.mask.astype(np.uint8)

    rle = mask_utils.encode(np.asfortranarray(full_mask))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def load_tile_instances(
    coco_path: Path,
    mapping: dict[str, dict],
) -> tuple[dict, dict[int, list[Instance]]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    tile_images = {int(image["id"]): image for image in data.get("images", [])}
    by_original: dict[int, list[Instance]] = {}

    for annotation in data.get("annotations", []):
        tile_image = tile_images[int(annotation["image_id"])]
        tile_name = Path(tile_image["file_name"]).name
        tile_meta = mapping.get(tile_name)
        if tile_meta is None:
            raise KeyError(f"Could not map SAM tile back to source: {tile_name}")

        instance = annotation_to_instance(
            annotation,
            tile_meta,
            int(tile_image.get("height", SAM_SIZE)),
            int(tile_image.get("width", SAM_SIZE)),
        )
        if instance is not None:
            by_original.setdefault(tile_meta["original_id"], []).append(instance)

    return data, by_original


def write_final_dataset(
    args: argparse.Namespace,
    originals: list[dict],
    raw_coco: dict,
    by_original: dict[int, list[Instance]],
    stride: int,
    final_dir: Path,
) -> Path:
    # Merge is deterministic, so on resume it is simpler/safer to rebuild the
    # final folder from the completed tile-level COCO than to incrementally edit it.
    if final_dir.exists():
        shutil.rmtree(final_dir)

    images_dir = final_dir / "images"
    annotations_dir = final_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    final_images: list[dict] = []
    final_annotations: list[dict] = []
    annotation_id = 1
    total_before = 0
    total_after_tile_merge = 0
    total_after = 0
    total_generic_vehicle_removed = 0
    total_score_suppressed = 0
    categories = {
        int(category["id"]): str(category.get("name", ""))
        for category in raw_coco.get("categories", [])
    }
    non_vehicle_categories = parse_category_names(args.non_vehicle_class)

    for original in originals:
        source = Path(original["source"])
        relative = Path(original["relative"])

        if args.copy_images:
            destination = images_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        final_images.append(
            {
                "id": original["id"],
                "file_name": relative.as_posix(),
                "width": original["width"],
                "height": original["height"],
            }
        )

        instances = by_original.get(original["id"], [])
        before = len(instances)
        total_before += before
        instances = deduplicate_instances(
            instances,
            args.merge_iou,
            args.merge_containment,
        )
        after_tile_merge = len(instances)
        total_after_tile_merge += after_tile_merge
        instances, generic_removed, score_suppressed = (
            suppress_cross_category_duplicates(
                instances,
                categories,
                args.iou_threshold,
                non_vehicle_categories,
            )
        )
        total_generic_vehicle_removed += generic_removed
        total_score_suppressed += score_suppressed
        after = len(instances)
        total_after += after

        for instance in instances:
            rle = encode_full_rle(instance, original["height"], original["width"])
            final_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": original["id"],
                    "category_id": instance.category_id,
                    "segmentation": rle,
                    "area": instance.area,
                    "bbox": [instance.x, instance.y, instance.w, instance.h],
                    "iscrowd": 0,
                    "score": instance.score,
                }
            )
            annotation_id += 1

        print(
            f"[merge] {original['relative']}: "
            f"{before} tile instances -> {after_tile_merge} merged -> "
            f"{after} deduplicated"
        )

    output = {
        "info": {
            "description": "SAM 3.1 tiled inference",
            "sam_input_size": SAM_SIZE,
            "requested_input_size": args.input_size,
            "tile_stride": stride,
            "tile_overlap": args.tile_overlap,
            "merge_iou": args.merge_iou,
            "merge_containment": args.merge_containment,
            "iou_threshold": args.iou_threshold,
            "non_vehicle_class": sorted(non_vehicle_categories),
        },
        "images": final_images,
        "annotations": final_annotations,
        "categories": raw_coco.get("categories", []),
    }

    instances_path = annotations_dir / "instances.json"
    instances_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    summary = {
        "images": len(final_images),
        "annotations_before_merge": total_before,
        "annotations_after_tile_merge": total_after_tile_merge,
        "annotations": total_after,
        "generic_vehicle_removed": total_generic_vehicle_removed,
        "score_suppressed": total_score_suppressed,
        "dataset": str(instances_path),
        "sam_input_size": SAM_SIZE,
        "requested_input_size": args.input_size,
        "tile_size": SAM_SIZE,
        "stride": stride,
        "overlap": args.tile_overlap,
        "copy_images": args.copy_images,
    }
    (final_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n===== TILED ANNOTATION SUMMARY =====")
    print(f"images:       {len(final_images)}")
    print(f"tile masks:   {total_before}")
    print(f"tile-merged:  {total_after_tile_merge}")
    print(f"final masks:  {total_after}")
    print(f"vehicle drop: {total_generic_vehicle_removed}")
    print(f"score drop:   {total_score_suppressed}")
    print(f"dataset:      {instances_path}")
    if args.copy_images:
        print(f"images dir:   {images_dir}")
    else:
        print("images:       not copied (--no-copy-images)")

    return instances_path


def normalize_bucket_id(bucket: str) -> str:
    prefix = "hf://buckets/"
    value = bucket.strip().rstrip("/")
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if not value or "/" not in value:
        # HF supports implicit username for bucket creation, but sam31's help
        # advertises user/bucket, so keep the wrapper strict and predictable.
        raise SystemExit(
            "--bucket must be in 'user/bucket' form (or hf://buckets/user/bucket)"
        )
    return value


def upload_final_run(args: argparse.Namespace, final_dir: Path) -> None:
    if not args.bucket:
        return

    try:
        from huggingface_hub import create_bucket, sync_bucket
    except ImportError as exc:
        raise SystemExit(
            "HF Storage Bucket upload requires a recent huggingface_hub with "
            "create_bucket/sync_bucket support. Upgrade with:\n"
            "  pip install -U huggingface_hub"
        ) from exc

    bucket_id = normalize_bucket_id(args.bucket)
    print(f"[upload] ensuring bucket exists: {bucket_id}")

    # Matching sam31's help text: --private only affects bucket creation.
    bucket_url = create_bucket(
        bucket_id=bucket_id,
        private=True if args.private else None,
        exist_ok=True,
        token=args.token,
    )

    resolved_id = getattr(bucket_url, "bucket_id", bucket_id)
    destination = f"hf://buckets/{resolved_id}/{args.run}"
    print(f"[upload] syncing final merged run -> {destination}")
    sync_bucket(
        source=str(final_dir),
        dest=destination,
        token=args.token,
    )
    print(f"[upload] done: {destination}")


def main() -> None:
    args = parse_args()

    root = args.images_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"images_dir is not a directory: {root}")

    args.out_dir = args.out_dir.expanduser().resolve()
    final_dir = args.out_dir / args.run
    work_root = args.out_dir / ".sam31_tiled_work"
    work_dir = work_root / args.run
    tile_dir = work_dir / "tiles"
    raw_out_parent = work_dir / "sam_output"

    if args.overwrite:
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    elif not args.resume:
        if final_dir.exists():
            raise SystemExit(f"{final_dir} already exists. Use --overwrite.")
        if work_dir.exists():
            raise SystemExit(
                f"Stale tiled work directory exists: {work_dir}\n"
                "Use --resume to continue it or --overwrite to discard it."
            )

    work_dir.mkdir(parents=True, exist_ok=True)
    raw_out_parent.mkdir(parents=True, exist_ok=True)

    images = find_images(root, args.out_dir, work_root, args.limit)

    if args.input_size != SAM_SIZE:
        print(
            f"[tile] requested --input-size {args.input_size}; "
            f"inner SAM inference is fixed at {SAM_SIZE} and high resolution "
            "is preserved through tiling."
        )

    print(f"[tile] source images: {len(images)}")
    originals, mapping, stride = make_tiles(
        images,
        root,
        tile_dir,
        args.tile_overlap,
    )
    print(f"[tile] generated: {len(mapping)} tiles")
    print(f"[tile] size:      {SAM_SIZE}x{SAM_SIZE}")
    print(f"[tile] stride:    {stride}")
    print(f"[tile] overlap:   {args.tile_overlap:.0%}")

    coco_path = run_sam31(args, tile_dir, raw_out_parent)
    raw_coco, by_original = load_tile_instances(coco_path, mapping)
    write_final_dataset(
        args,
        originals,
        raw_coco,
        by_original,
        stride,
        final_dir,
    )

    # Upload only after the merged original-resolution dataset is complete.
    upload_final_run(args, final_dir)

    if args.keep_workdir:
        print(f"[debug] work directory kept: {work_dir}")
    else:
        shutil.rmtree(work_dir, ignore_errors=True)
        # Remove empty parent directory when possible.
        try:
            work_root.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: sam31 exited with status {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
