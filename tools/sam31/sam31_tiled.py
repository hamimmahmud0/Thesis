#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SAM_SIZE = 1008

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ---------------------------------------------------------------------
# Internal instance representation
# ---------------------------------------------------------------------

@dataclass
class Instance:
    category_id: int
    score: float

    # top-left position in ORIGINAL image
    x: int
    y: int

    # tightly cropped binary mask
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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Tile large images into overlapping 1008x1008 crops, "
            "run SAM 3.1 on the tiles, and merge the masks back "
            "into original-image COCO coordinates."
        )
    )

    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory containing source images.",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        action="append",
        default=[],
        help="SAM text prompt. Repeatable.",
    )

    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--devices",
        default="all",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--token",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--run",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help=(
            "Fractional overlap between tiles. "
            "Default 0.25 = 25%% overlap."
        ),
    )

    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.35,
        help=(
            "Merge same-category masks from overlapping tiles "
            "when mask IoU >= this value."
        ),
    )

    parser.add_argument(
        "--merge-containment",
        type=float,
        default=0.60,
        help=(
            "Merge when intersection / smaller-mask-area "
            ">= this value."
        ),
    )

    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep temporary tiles and raw SAM output.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------

def read_prompts(args) -> list[str]:

    prompts = list(args.prompt)

    if args.prompts_file:
        for line in args.prompts_file.read_text(
            encoding="utf-8"
        ).splitlines():

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            prompts.append(line)

    if not prompts:
        raise SystemExit(
            "At least one -p/--prompt or "
            "--prompts-file entry is required."
        )

    return prompts


# ---------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------

def is_under(path: Path, parent: Path) -> bool:

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_images(
    root: Path,
    out_root: Path,
    limit: int,
) -> list[Path]:

    root = root.resolve()

    images = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Important when running:
        #
        #     sam31_tiled.py .
        #
        # Prevent previous run output from becoming new input.
        if is_under(path, out_root):
            continue

        images.append(path)

    images.sort()

    if limit > 0:
        images = images[:limit]

    if not images:
        raise SystemExit(
            f"No images found under {root}"
        )

    return images


# ---------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------

def tile_positions(
    length: int,
    tile_size: int,
    stride: int,
) -> list[int]:

    if length <= tile_size:
        return [0]

    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    # Always anchor one tile to the far edge so no pixels
    # are missed.
    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def make_tiles(
    images: list[Path],
    root: Path,
    tile_dir: Path,
    overlap: float,
):

    stride = max(
        1,
        int(round(
            SAM_SIZE * (1.0 - overlap)
        )),
    )

    mapping = {}
    originals = []

    for original_id, source in enumerate(
        images,
        start=1,
    ):

        with Image.open(source) as image:

            width, height = image.size

            relative = (
                source.resolve()
                .relative_to(root.resolve())
            )

            originals.append(
                {
                    "id": original_id,
                    "source": str(source.resolve()),
                    "relative": relative.as_posix(),
                    "width": width,
                    "height": height,
                }
            )

            xs = tile_positions(
                width,
                SAM_SIZE,
                stride,
            )

            ys = tile_positions(
                height,
                SAM_SIZE,
                stride,
            )

            tile_index = 0

            for y in ys:
                for x in xs:

                    valid_w = min(
                        SAM_SIZE,
                        width - x,
                    )

                    valid_h = min(
                        SAM_SIZE,
                        height - y,
                    )

                    crop = image.crop(
                        (
                            x,
                            y,
                            x + valid_w,
                            y + valid_h,
                        )
                    ).convert("RGB")

                    # Edge tiles for images smaller than 1008
                    # are padded to the model's native size.
                    canvas = Image.new(
                        "RGB",
                        (SAM_SIZE, SAM_SIZE),
                        (0, 0, 0),
                    )

                    canvas.paste(
                        crop,
                        (0, 0),
                    )

                    name = (
                        f"tile_o{original_id:06d}"
                        f"_t{tile_index:05d}"
                        f"_x{x:06d}"
                        f"_y{y:06d}.png"
                    )

                    tile_path = tile_dir / name

                    canvas.save(
                        tile_path,
                        compress_level=1,
                    )

                    mapping[name] = {
                        "original_id": original_id,
                        "x": x,
                        "y": y,
                        "valid_w": valid_w,
                        "valid_h": valid_h,
                    }

                    tile_index += 1

    return originals, mapping, stride


# ---------------------------------------------------------------------
# Run your existing SAM 3.1 CLI
# ---------------------------------------------------------------------

def run_sam31(
    args,
    prompts: list[str],
    tile_dir: Path,
    raw_out: Path,
):

    command = [
        "sam31",
        "run",
        str(tile_dir),

        "--confidence",
        str(args.confidence),

        "--batch-size",
        str(args.batch_size),

        # IMPORTANT:
        # Never change this to 2016/3024.
        "--input-size",
        str(SAM_SIZE),

        "--devices",
        args.devices,

        "--run",
        "tiles",

        "--out-dir",
        str(raw_out),

        "--no-copy-images",
        "--overwrite",
    ]

    for prompt in prompts:
        command.extend(
            [
                "-p",
                prompt,
            ]
        )

    if args.checkpoint:
        command.extend(
            [
                "--checkpoint",
                str(args.checkpoint),
            ]
        )

    if args.token:
        command.extend(
            [
                "--token",
                args.token,
            ]
        )

    # Don't print an HF token.
    printable = []

    skip_next = False

    for item in command:

        if skip_next:
            printable.append("***")
            skip_next = False
            continue

        printable.append(item)

        if item == "--token":
            skip_next = True

    print()
    print(
        "[sam31]",
        " ".join(printable),
    )
    print()

    subprocess.run(
        command,
        check=True,
    )

    coco_path = (
        raw_out
        / "tiles"
        / "annotations"
        / "instances.json"
    )

    if not coco_path.exists():
        raise FileNotFoundError(
            f"SAM output not found: {coco_path}"
        )

    return coco_path


# ---------------------------------------------------------------------
# COCO mask decoding
# ---------------------------------------------------------------------

def decode_segmentation(
    segmentation,
    height: int,
    width: int,
) -> np.ndarray:

    # Normal compressed COCO RLE
    if isinstance(segmentation, dict):

        rle = dict(segmentation)

        # pycocotools normally expects bytes internally.
        if isinstance(
            rle.get("counts"),
            str,
        ):
            rle["counts"] = (
                rle["counts"].encode("ascii")
            )

        mask = mask_utils.decode(rle)

    # Also allow polygon segmentation for robustness.
    elif isinstance(segmentation, list):

        rles = mask_utils.frPyObjects(
            segmentation,
            height,
            width,
        )

        rle = mask_utils.merge(rles)

        mask = mask_utils.decode(rle)

    else:
        raise TypeError(
            "Unsupported segmentation type: "
            f"{type(segmentation)!r}"
        )

    if mask.ndim == 3:
        mask = np.any(
            mask,
            axis=2,
        )

    return mask.astype(
        bool,
        copy=False,
    )


# ---------------------------------------------------------------------
# Convert tile mask -> original-image mask coordinates
# ---------------------------------------------------------------------

def annotation_to_instance(
    annotation,
    tile_meta,
    tile_height: int,
    tile_width: int,
):

    mask = decode_segmentation(
        annotation["segmentation"],
        tile_height,
        tile_width,
    )

    # Remove padded region if this was an edge tile.
    mask = mask[
        :tile_meta["valid_h"],
        :tile_meta["valid_w"],
    ]

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    tight = mask[
        y0:y1,
        x0:x1,
    ].copy()

    score = annotation.get(
        "score",
        annotation.get(
            "confidence",
            1.0,
        ),
    )

    return Instance(
        category_id=int(
            annotation["category_id"]
        ),

        score=float(score),

        x=int(
            tile_meta["x"] + x0
        ),

        y=int(
            tile_meta["y"] + y0
        ),

        mask=tight,
    )


# ---------------------------------------------------------------------
# Mask-overlap calculations
# ---------------------------------------------------------------------

def mask_intersection(
    a: Instance,
    b: Instance,
) -> int:

    x1 = max(
        a.x,
        b.x,
    )

    y1 = max(
        a.y,
        b.y,
    )

    x2 = min(
        a.x + a.w,
        b.x + b.w,
    )

    y2 = min(
        a.y + a.h,
        b.y + b.h,
    )

    if x1 >= x2 or y1 >= y2:
        return 0

    mask_a = a.mask[
        y1 - a.y:y2 - a.y,
        x1 - a.x:x2 - a.x,
    ]

    mask_b = b.mask[
        y1 - b.y:y2 - b.y,
        x1 - b.x:x2 - b.x,
    ]

    return int(
        np.logical_and(
            mask_a,
            mask_b,
        ).sum()
    )


def overlap_metrics(
    a: Instance,
    b: Instance,
):

    intersection = mask_intersection(
        a,
        b,
    )

    if intersection == 0:
        return 0.0, 0.0

    area_a = a.area
    area_b = b.area

    union = (
        area_a
        + area_b
        - intersection
    )

    iou = (
        intersection / union
        if union
        else 0.0
    )

    smaller_area = min(
        area_a,
        area_b,
    )

    containment = (
        intersection / smaller_area
        if smaller_area
        else 0.0
    )

    return iou, containment


# ---------------------------------------------------------------------
# Union duplicate masks
# ---------------------------------------------------------------------

def union_instances(
    a: Instance,
    b: Instance,
) -> Instance:

    x0 = min(
        a.x,
        b.x,
    )

    y0 = min(
        a.y,
        b.y,
    )

    x1 = max(
        a.x + a.w,
        b.x + b.w,
    )

    y1 = max(
        a.y + a.h,
        b.y + b.h,
    )

    merged = np.zeros(
        (
            y1 - y0,
            x1 - x0,
        ),
        dtype=bool,
    )

    merged[
        a.y - y0:
        a.y - y0 + a.h,

        a.x - x0:
        a.x - x0 + a.w,
    ] |= a.mask

    merged[
        b.y - y0:
        b.y - y0 + b.h,

        b.x - x0:
        b.x - x0 + b.w,
    ] |= b.mask

    return Instance(
        category_id=a.category_id,
        score=max(
            a.score,
            b.score,
        ),
        x=x0,
        y=y0,
        mask=merged,
    )


# ---------------------------------------------------------------------
# Deduplicate overlapping tile detections
# ---------------------------------------------------------------------

def deduplicate_instances(
    instances: list[Instance],
    iou_threshold: float,
    containment_threshold: float,
):

    # Higher-confidence masks first.
    instances = sorted(
        instances,
        key=lambda item: item.score,
        reverse=True,
    )

    kept: list[Instance] = []

    for candidate in instances:

        best_index = None
        best_metric = 0.0

        for index, previous in enumerate(kept):

            if (
                candidate.category_id
                != previous.category_id
            ):
                continue

            iou, containment = overlap_metrics(
                candidate,
                previous,
            )

            duplicate = (
                iou >= iou_threshold
                or
                containment
                >= containment_threshold
            )

            if not duplicate:
                continue

            metric = max(
                iou,
                containment,
            )

            if metric > best_metric:
                best_metric = metric
                best_index = index

        if best_index is None:

            kept.append(candidate)

        else:

            kept[best_index] = union_instances(
                kept[best_index],
                candidate,
            )

    return kept


# ---------------------------------------------------------------------
# Re-encode into full original-image compressed RLE
# ---------------------------------------------------------------------

def encode_full_rle(
    instance: Instance,
    height: int,
    width: int,
):

    full_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    full_mask[
        instance.y:
        instance.y + instance.h,

        instance.x:
        instance.x + instance.w,
    ] = instance.mask.astype(
        np.uint8
    )

    rle = mask_utils.encode(
        np.asfortranarray(
            full_mask
        )
    )

    if isinstance(
        rle["counts"],
        bytes,
    ):
        rle["counts"] = (
            rle["counts"].decode("ascii")
        )

    return rle


# ---------------------------------------------------------------------
# Read raw SAM result
# ---------------------------------------------------------------------

def load_tile_instances(
    coco_path: Path,
    mapping: dict,
):

    data = json.loads(
        coco_path.read_text(
            encoding="utf-8"
        )
    )

    tile_images = {
        int(image["id"]): image
        for image in data["images"]
    }

    by_original = {}

    for annotation in data["annotations"]:

        tile_image = tile_images[
            int(annotation["image_id"])
        ]

        tile_name = Path(
            tile_image["file_name"]
        ).name

        tile_meta = mapping.get(
            tile_name
        )

        if tile_meta is None:
            raise KeyError(
                "Could not map SAM tile "
                f"back to source: {tile_name}"
            )

        instance = annotation_to_instance(
            annotation,

            tile_meta,

            int(
                tile_image.get(
                    "height",
                    SAM_SIZE,
                )
            ),

            int(
                tile_image.get(
                    "width",
                    SAM_SIZE,
                )
            ),
        )

        if instance is None:
            continue

        original_id = (
            tile_meta["original_id"]
        )

        by_original.setdefault(
            original_id,
            [],
        ).append(instance)

    return data, by_original


# ---------------------------------------------------------------------
# Final COCO dataset
# ---------------------------------------------------------------------

def write_final_dataset(
    args,
    originals,
    raw_coco,
    by_original,
    stride,
):

    final_dir = (
        args.out_dir.resolve()
        / args.run
    )

    if final_dir.exists():

        if not args.overwrite:
            raise SystemExit(
                f"{final_dir} already exists. "
                "Use --overwrite."
            )

        shutil.rmtree(
            final_dir
        )

    images_dir = (
        final_dir
        / "images"
    )

    annotations_dir = (
        final_dir
        / "annotations"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotations_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_images = []
    final_annotations = []

    annotation_id = 1

    total_before = 0
    total_after = 0

    for original in originals:

        source = Path(
            original["source"]
        )

        relative = Path(
            original["relative"]
        )

        destination = (
            images_dir
            / relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        final_images.append(
            {
                "id": original["id"],
                "file_name": relative.as_posix(),
                "width": original["width"],
                "height": original["height"],
            }
        )

        instances = by_original.get(
            original["id"],
            [],
        )

        before = len(instances)

        total_before += before

        instances = deduplicate_instances(
            instances,
            args.merge_iou,
            args.merge_containment,
        )

        after = len(instances)

        total_after += after

        for instance in instances:

            rle = encode_full_rle(
                instance,
                original["height"],
                original["width"],
            )

            final_annotations.append(
                {
                    "id": annotation_id,

                    "image_id":
                        original["id"],

                    "category_id":
                        instance.category_id,

                    "segmentation":
                        rle,

                    "area":
                        instance.area,

                    "bbox": [
                        instance.x,
                        instance.y,
                        instance.w,
                        instance.h,
                    ],

                    "iscrowd":
                        0,

                    "score":
                        instance.score,
                }
            )

            annotation_id += 1

        print(
            f"[merge] {original['relative']}: "
            f"{before} tile instances -> "
            f"{after} final instances"
        )

    output = {
        "info": {
            "description":
                "SAM 3.1 tiled inference",

            "sam_input_size":
                SAM_SIZE,

            "tile_stride":
                stride,

            "tile_overlap":
                args.overlap,

            "merge_iou":
                args.merge_iou,

            "merge_containment":
                args.merge_containment,
        },

        "images":
            final_images,

        "annotations":
            final_annotations,

        "categories":
            raw_coco.get(
                "categories",
                [],
            ),
    }

    instances_path = (
        annotations_dir
        / "instances.json"
    )

    instances_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "images":
            len(final_images),

        "annotations_before_merge":
            total_before,

        "annotations":
            total_after,

        "dataset":
            str(instances_path),

        "tile_size":
            SAM_SIZE,

        "stride":
            stride,

        "overlap":
            args.overlap,
    }

    (
        final_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "===== TILED ANNOTATION SUMMARY ====="
    )

    print(
        f"images:       {len(final_images)}"
    )

    print(
        f"tile masks:   {total_before}"
    )

    print(
        f"final masks:  {total_after}"
    )

    print(
        f"dataset:      {instances_path}"
    )

    print(
        f"images dir:   {images_dir}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    if not (
        0.0 <= args.overlap < 1.0
    ):
        raise SystemExit(
            "--overlap must be >= 0 and < 1"
        )

    if not (
        0.0 <= args.merge_iou <= 1.0
    ):
        raise SystemExit(
            "--merge-iou must be between 0 and 1"
        )

    if not (
        0.0
        <= args.merge_containment
        <= 1.0
    ):
        raise SystemExit(
            "--merge-containment must be "
            "between 0 and 1"
        )

    root = args.images_dir.resolve()

    args.out_dir = (
        args.out_dir.resolve()
    )

    final_dir = (
        args.out_dir
        / args.run
    )

    if (
        final_dir.exists()
        and
        not args.overwrite
    ):
        raise SystemExit(
            f"{final_dir} already exists. "
            "Use --overwrite."
        )

    prompts = read_prompts(args)

    images = find_images(
        root,
        args.out_dir,
        args.limit,
    )

    # -------------------------------------------------------------
    # Temporary workspace
    # -------------------------------------------------------------

    if args.keep_workdir:

        work = Path(
            tempfile.mkdtemp(
                prefix="sam31_tiles_"
            )
        )

        temp_context = None

    else:

        temp_context = (
            tempfile.TemporaryDirectory(
                prefix="sam31_tiles_"
            )
        )

        work = Path(
            temp_context.name
        )

    tile_dir = (
        work
        / "tiles"
    )

    raw_out = (
        work
        / "sam_output"
    )

    tile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[tile] source images: "
        f"{len(images)}"
    )

    # -------------------------------------------------------------
    # Make 1008 tiles
    # -------------------------------------------------------------

    originals, mapping, stride = make_tiles(
        images,
        root,
        tile_dir,
        args.overlap,
    )

    print(
        f"[tile] generated "
        f"{len(mapping)} tiles"
    )

    print(
        f"[tile] size:    "
        f"{SAM_SIZE}x{SAM_SIZE}"
    )

    print(
        f"[tile] stride:  "
        f"{stride}"
    )

    print(
        f"[tile] overlap: "
        f"{args.overlap:.0%}"
    )

    # -------------------------------------------------------------
    # Run installed SAM CLI
    # -------------------------------------------------------------

    coco_path = run_sam31(
        args,
        prompts,
        tile_dir,
        raw_out,
    )

    # -------------------------------------------------------------
    # Decode and map to source-image coordinates
    # -------------------------------------------------------------

    raw_coco, by_original = (
        load_tile_instances(
            coco_path,
            mapping,
        )
    )

    # -------------------------------------------------------------
    # Deduplicate + write final dataset
    # -------------------------------------------------------------

    write_final_dataset(
        args,
        originals,
        raw_coco,
        by_original,
        stride,
    )

    if args.keep_workdir:

        print()
        print(
            "[debug] temporary data kept at:"
        )

        print(
            work
        )

    elif temp_context is not None:

        temp_context.cleanup()


if __name__ == "__main__":
    main()