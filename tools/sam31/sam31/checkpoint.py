"""Checkpoint resolution: auto-download the SAM 3.1 checkpoint on first use."""

from __future__ import annotations

from pathlib import Path

from . import hfio
from .config import CACHE_DIR, CHECKPOINT_FILENAME, CHECKPOINT_URL


def resolve_checkpoint(
    checkpoint: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """Return a usable checkpoint path, downloading to the cache if missing.

    ``checkpoint`` may point at an explicit ``.pt`` file (must exist).  When
    omitted, ``~/.cache/sam31/sam3.1_multiplex_mapped.pt`` is used and
    downloaded from the public HF storage bucket on first use.
    """
    if checkpoint is not None:
        cp = Path(checkpoint).expanduser()
        if not cp.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")
        return cp

    cp = CACHE_DIR / CHECKPOINT_FILENAME
    if cp.is_file():
        return cp

    print(f"[checkpoint] not found: {cp} — downloading from HF bucket ...", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return hfio.https_download(CHECKPOINT_URL, cp, token=token, label=CHECKPOINT_FILENAME)
