"""Image-directory discovery and bookkeeping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import IMAGE_SUFFIXES

_NATURAL_RE = re.compile(r"(\d+)")


def natural_key(text: str) -> list:
    """Sort key ordering embedded numbers numerically (frame_2 < frame_10)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_RE.split(text)]


@dataclass
class ImageRec:
    """One discovered image.

    ``index``   0-based position in the sorted scan; the COCO image id is
                ``index + 1``.
    ``path``    absolute path of the file the worker should read (this points
                at the copied file when ``--copy-images`` is used).
    ``rel``     the ``file_name`` recorded in the COCO dataset, relative to
                the images root, posix style.
    """

    index: int
    path: Path
    rel: str
    width: int | None = None
    height: int | None = None


def scan_images(root: str | Path) -> list[ImageRec]:
    """Recursively find images under *root*, natural-sorted by relative path."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda p: natural_key(p.relative_to(root).as_posix()))
    return [
        ImageRec(index=i, path=p.resolve(), rel=p.relative_to(root).as_posix())
        for i, p in enumerate(files)
    ]


def read_sizes(recs: list[ImageRec]) -> None:
    """Best-effort fill of width/height via PIL header reads."""
    try:
        from PIL import Image
    except ImportError:
        return
    for rec in recs:
        if rec.width and rec.height:
            continue
        try:
            with Image.open(rec.path) as im:
                rec.width, rec.height = im.size
        except Exception:
            continue
