"""Shared constants and defaults for the sam31 annotator."""

from __future__ import annotations

from pathlib import Path

# Hugging Face storage bucket hosting the SAM 3.1 checkpoint (public).
BUCKET_RESOLVE_BASE = (
    "https://huggingface.co/buckets/z81980440/agent-tools/resolve/SAM3.1"
)
CHECKPOINT_FILENAME = "sam3.1_multiplex_mapped.pt"
CHECKPOINT_URL = f"{BUCKET_RESOLVE_BASE}/{CHECKPOINT_FILENAME}"
CHECKPOINT_SIZE_BYTES = 3_502_880_784  # approx., for progress estimates

# Local cache location.
CACHE_DIR = Path.home() / ".cache" / "sam31"
VIDEO_CACHE_DIR = CACHE_DIR / "videos"

# Inference defaults (mirroring the reference SAM3.1 inference server).
DEFAULT_INPUT_SIZE = 1008
DEFAULT_BATCH_SIZE = 4
DEFAULT_CONFIDENCE = 0.5
DEFAULT_JPEG_QUALITY = 92

# Supported inputs.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts"}

# Prompt handling.
PROMPT_SEPARATORS = ("|", ";")
MAX_PROMPT_CHARS = 500
MAX_PROMPTS = 64

# Seconds to wait for all workers to load the model before giving up.
MODEL_LOAD_TIMEOUT = 3600
