"""Command-line interface for the SAM 3.1 annotator.

Subcommands:

  run       Annotate an image directory with text prompts into a COCO dataset.
  plan      Dry-run: show what ``run`` would do (no model load).
  frames    Extract frames from a video (local path, http(s):// or hf://).
  download  Prefetch the SAM 3.1 checkpoint into the local cache.
  upload    Upload a local directory to an HF storage bucket.

Typical workflow::

  sam31 frames https://huggingface.co/datasets/user/repo/resolve/main/clip.mp4 \\
      -o frames/clip --step 150 --limit 48

  sam31 run frames/clip \\
      --run clip-annotated \\
      -p "pedestrian | car | bus | truck" \\
      --batch-size 4 --confidence 0.5

Run ``sam31 <subcommand> --help`` for per-command details.
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------

def _add_prompt_args(p: argparse.ArgumentParser, *, prompts_required: bool) -> None:
    p.add_argument(
        "-p", "--prompt", action="append", metavar="TEXT",
        help=(
            "Text prompt (concept) to detect and segment — repeatable. "
            "Multiple concepts can be combined in one string with '|' or ';' "
            "separators, e.g. -p 'car | bus | truck'. One COCO category is "
            "created per prompt."
        ),
    )
    p.add_argument(
        "--prompts-file", metavar="FILE",
        help="File with one prompt per line (# comments allowed); merged with -p.",
    )
    if prompts_required:
        p.set_defaults(require_prompts=True)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    _add_prompt_args(p, prompts_required=True)
    p.add_argument(
        "--confidence", type=float, default=0.5, metavar="F",
        help="Detection score threshold. Default: 0.5.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4, metavar="N",
        help=(
            "Images per forward pass per worker (all prompts run together in "
            "one pass). Halved automatically on CUDA OOM. Default: 4."
        ),
    )
    p.add_argument(
        "--input-size", type=int, default=1008, metavar="PX",
        help="Square inference resolution. Default: 1008.",
    )
    p.add_argument(
        "--devices", default=None, metavar="LIST",
        help=(
            "CUDA devices to use, e.g. 'cuda:0,cuda:1' or '0,1' or 'all'. "
            "Default: all CUDA devices, one worker each."
        ),
    )
    p.add_argument(
        "--checkpoint", default=None, metavar="CKPT.pt",
        help=(
            "SAM 3.1 checkpoint path. Default: ~/.cache/sam31/"
            "sam3.1_multiplex_mapped.pt (auto-downloaded from the HF bucket "
            "on first use)."
        ),
    )
    p.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Annotate only the first N images (0 = all). Useful for smoke tests.",
    )
    p.add_argument(
        "--token", default=None, metavar="hf_xxx",
        help="Hugging Face token (checkpoint download from private repos, uploads).",
    )


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    from .runner import run_pipeline

    summary = run_pipeline(
        images_dir=args.images_dir,
        prompt_args=args.prompt,
        prompts_file=args.prompts_file,
        run_name=args.run_name,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        devices=args.devices,
        batch_size=args.batch_size,
        confidence=args.confidence,
        input_size=args.input_size,
        copy_images=args.copy_images,
        resume=args.resume,
        overwrite=args.overwrite,
        limit=args.limit,
        bucket=args.bucket,
        token=args.token,
        private=args.private,
    )
    if summary.get("failed_images"):
        print("\nNOTE: some images failed — see summary.json / failures.jsonl")
        sys.exit(1)


def _cmd_plan(args: argparse.Namespace) -> None:
    from .dataset import read_sizes, scan_images
    from .prompts import parse_prompts

    images = scan_images(args.images_dir)
    if args.limit > 0:
        images = images[:limit_reindex(images, args.limit)]
    read_sizes(images)
    prompts = parse_prompts(args.prompt, args.prompts_file)

    import multiprocessing

    n_batches = (len(images) + args.batch_size - 1) // max(1, args.batch_size)

    devices_desc = "unknown (torch not importable here)"
    try:
        import torch

        count = torch.cuda.device_count()
        devices_desc = (
            f"{count} CUDA device(s): {', '.join(f'cuda:{i}' for i in range(count))}"
            if count
            else "NONE — at least one GPU is required for 'run'"
        )
    except ImportError:
        pass

    try:
        from .checkpoint import resolve_checkpoint  # noqa: F401
        from .config import CACHE_DIR, CHECKPOINT_FILENAME

        ckpt = CACHE_DIR / CHECKPOINT_FILENAME
        ckpt_desc = str(ckpt) + (" (present)" if ckpt.is_file() else " (will download ~3.5 GB)")
        if args.checkpoint:
            ckpt_desc = args.checkpoint
    except Exception:
        ckpt_desc = "n/a"

    size0 = images[0]
    print("=" * 24, "PROPOSED PARAMETERS", "=" * 24)
    print(f"images dir:        {args.images_dir}")
    print(f"images:            {len(images)} (first: {size0.rel}, "
          f"{size0.width or '?'}x{size0.height or '?'})")
    print(f"prompts:           {len(prompts)} -> one COCO category each")
    for i, p in enumerate(prompts, 1):
        print(f"  category {i}:      {p!r}")
    print(f"confidence:        {args.confidence}")
    print(f"batch size:        {args.batch_size} -> {n_batches} batch(es), "
          f"{len(prompts)} prompt(s) per image per pass")
    print(f"input size:        {args.input_size}px (square)")
    print(f"devices:           {devices_desc}")
    print(f"checkpoint:        {ckpt_desc}")
    print(f"output:            {args.out_dir}/{args.run_name}/annotations/instances.json")
    print(f"images copied:     {'yes' if args.copy_images else 'no (referenced in place)'}")
    print("=" * 67)
    print("Run 'sam31 run' with the same arguments to annotate.")


def limit_reindex(images, limit):
    for i, rec in enumerate(images[:limit]):
        rec.index = i
    return min(limit, len(images))


def _cmd_frames(args: argparse.Namespace) -> None:
    from .frames import extract_frames

    extract_frames(
        args.source,
        args.output_dir,
        step=args.step,
        limit=args.limit,
        quality=args.quality,
        max_side=args.max_side,
        token=args.token,
        delete_video=args.delete_video,
    )


def _cmd_download(args: argparse.Namespace) -> None:
    from .checkpoint import resolve_checkpoint

    cp = resolve_checkpoint(args.checkpoint, token=args.token)
    print(f"Checkpoint ready: {cp} ({cp.stat().st_size / 1e9:.2f} GB)")


def _cmd_upload(args: argparse.Namespace) -> None:
    from pathlib import Path

    from . import hfio

    src = Path(args.local_dir)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory")
        sys.exit(1)
    bucket_id = hfio.ensure_bucket(args.bucket, token=args.token, private=args.private)
    remote = args.remote_prefix.rstrip("/")
    print(f"Uploading {src} -> hf://buckets/{bucket_id}/{remote}/")
    uploaded = hfio.upload_dir(src, bucket_id, remote, token=args.token)
    print(f"Done — {len(uploaded)} file(s).")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sam31",
        description=(
            "SAM 3.1 text-prompted annotator: turns a directory of images "
            "plus text prompts into a COCO 1.0 instance-segmentation dataset, "
            "using one GPU worker per CUDA device."
        ),
        epilog=(
            "Examples:\n"
            "  sam31 frames https://.../clip.mp4 -o frames/clip --step 150 --limit 48\n"
            "  sam31 plan frames/clip -p 'car | bus | pedestrian'\n"
            "  sam31 run frames/clip --run clip-v1 -p car -p bus -p pedestrian\n"
            "\n"
            "Run 'sam31 <subcommand> --help' for per-command details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="Annotate an image directory into a COCO dataset.",
        description=(
            "Scans the image directory, spawns one SAM 3.1 GPU worker per "
            "CUDA device, runs every text prompt on every image (all prompts "
            "in a single forward pass per image), and writes a COCO 1.0 "
            "instance-segmentation dataset with compressed-RLE masks."
        ),
    )
    p_run.add_argument("images_dir", help="Directory containing the images (searched recursively).")
    _add_run_args(p_run)
    p_run.add_argument("--run", dest="run_name", required=True, metavar="NAME",
                       help="Run folder name (outputs land in <out-dir>/<run>/).")
    p_run.add_argument("--out-dir", default="runs", metavar="DIR",
                       help="Local parent directory for run folders. Default: ./runs")
    p_run.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True,
                       help="Copy images into <run>/images/ (self-contained dataset). Default: on.")
    p_run.add_argument("--resume", action="store_true",
                       help="Resume a partial run from its shard files.")
    p_run.add_argument("--overwrite", action="store_true",
                       help="Discard any existing run folder and start over.")
    p_run.add_argument("--bucket", default=None, metavar="user/bucket",
                       help="HF storage bucket to upload the run to (optional).")
    p_run.add_argument("--private", action="store_true",
                       help="Create the bucket as private (if creating a new one).")
    p_run.set_defaults(func=_cmd_run)

    # ---- plan ----
    p_plan = sub.add_parser(
        "plan",
        help="Dry-run: show what 'run' would do.",
        description="Validates the image directory and prompts without loading the model.",
    )
    p_plan.add_argument("images_dir", help="Directory containing the images.")
    _add_run_args(p_plan)
    p_plan.add_argument("--run", dest="run_name", default="run1", metavar="NAME",
                        help="Run folder name used to print the output path. Default: run1")
    p_plan.add_argument("--out-dir", default="runs", metavar="DIR",
                        help="Local parent directory for run folders. Default: ./runs")
    p_plan.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy images into <run>/images/. Default: on.")
    p_plan.set_defaults(func=_cmd_plan)

    # ---- frames ----
    p_fr = sub.add_parser(
        "frames",
        help="Extract frames from a video into a directory.",
        description=(
            "Downloads the video if given a URL/hf:// link (cached), then "
            "writes every --step-th frame as JPEG, up to --limit frames."
        ),
    )
    p_fr.add_argument("source", help="Local video path, http(s):// URL, or hf:// link.")
    p_fr.add_argument("-o", "--output-dir", required=True, metavar="DIR",
                      help="Directory to write frame_000001.jpg ... into.")
    p_fr.add_argument("--step", type=int, default=1, metavar="N",
                      help="Keep every Nth frame. Default: 1 (all frames).")
    p_fr.add_argument("--limit", type=int, default=0, metavar="N",
                      help="Stop after N frames (0 = all).")
    p_fr.add_argument("--quality", type=int, default=92, metavar="Q",
                      help="JPEG quality 1-100. Default: 92.")
    p_fr.add_argument("--max-side", type=int, default=0, metavar="PX",
                      help="Downscale so the longest side is at most PX (0 = keep original).")
    p_fr.add_argument("--token", default=None, metavar="hf_xxx",
                      help="HF token for private URLs/repos.")
    p_fr.add_argument("--delete-video", action="store_true",
                      help="Delete the downloaded video after extraction.")
    p_fr.set_defaults(func=_cmd_frames)

    # ---- download ----
    p_dl = sub.add_parser(
        "download",
        help="Prefetch the SAM 3.1 checkpoint.",
        description="Downloads the checkpoint into ~/.cache/sam31/ if not already present.",
    )
    p_dl.add_argument("--checkpoint", default=None, metavar="CKPT.pt",
                      help="Explicit checkpoint path to verify.")
    p_dl.add_argument("--token", default=None, metavar="hf_xxx")
    p_dl.set_defaults(func=_cmd_download)

    # ---- upload ----
    p_up = sub.add_parser(
        "upload",
        help="Upload a local directory to an HF storage bucket.",
    )
    p_up.add_argument("local_dir", help="Local directory to upload.")
    p_up.add_argument("--bucket", required=True, metavar="user/bucket")
    p_up.add_argument("--remote-prefix", required=True, metavar="PREFIX")
    p_up.add_argument("--token", default=None, metavar="hf_xxx")
    p_up.add_argument("--private", action="store_true")
    p_up.set_defaults(func=_cmd_upload)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
