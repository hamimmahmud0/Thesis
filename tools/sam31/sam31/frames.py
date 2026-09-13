"""Video -> frame-directory extraction (building annotation inputs).

Accepts a local video file, an ``http(s)://`` URL, or an ``hf://`` link.
Videos given by URL/link are downloaded once into the local video cache
(``~/.cache/sam31/videos/``) and reused on later invocations.
"""

from __future__ import annotations

from pathlib import Path

from .config import VIDEO_CACHE_DIR, VIDEO_SUFFIXES


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _resolve_source(
    source: str, cache_dir: Path, token: str | None = None
) -> tuple[Path, bool]:
    """Return (video_path, downloaded). Downloads URLs/links into the cache."""
    if _is_url(source):
        from . import hfio

        cache_dir.mkdir(parents=True, exist_ok=True)
        name = hfio.link_basename(source) or "video.mp4"
        dest = cache_dir / name
        if dest.is_file():
            print(f"[frames] using cached video {dest}")
            return dest, False
        return hfio.https_download(source, dest, token=token, label=name), True

    if source.startswith("hf://"):
        from . import hfio

        cache_dir.mkdir(parents=True, exist_ok=True)
        name = hfio.link_basename(source)
        dest = cache_dir / name
        if dest.is_file():
            print(f"[frames] using cached video {dest}")
            return dest, False
        path = hfio.download_link(source, cache_dir, token=token)
        return path, True

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video extension {path.suffix!r} "
            f"(supported: {sorted(VIDEO_SUFFIXES)})"
        )
    return path, False


def extract_frames(
    source: str,
    output_dir: str | Path,
    *,
    step: int = 1,
    limit: int = 0,
    quality: int = 92,
    max_side: int = 0,
    token: str | None = None,
    video_cache: str | Path | None = None,
    delete_video: bool = False,
) -> dict:
    """Extract every ``step``-th frame (up to ``limit``) into *output_dir*.

    Returns a dict with frame count, video metadata and the output directory.
    """
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(video_cache) if video_cache else VIDEO_CACHE_DIR

    video, downloaded = _resolve_source(str(source), cache, token=token)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(step))

    print(
        f"[frames] {video.name}: {total} frames, {width}x{height} @ "
        f"{fps:.3f} fps — extracting every {step} frame(s)"
    )

    written = 0
    frame_idx = 0
    while True:
        if not cap.grab():
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                if max_side and max(width, height) > max_side:
                    scale = max_side / max(width, height)
                    frame = cv2.resize(
                        frame,
                        (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2),
                    )
                out_path = output_dir / f"frame_{written + 1:06d}.jpg"
                cv2.imwrite(
                    str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                )
                written += 1
                if limit and written >= limit:
                    break
        frame_idx += 1
    cap.release()

    if delete_video and downloaded:
        video.unlink(missing_ok=True)
        print(f"[frames] deleted downloaded video {video}")

    print(f"[frames] wrote {written} frame(s) -> {output_dir}")
    return {
        "frames": written,
        "out_dir": str(output_dir),
        "video": str(video),
        "fps": fps,
        "total_frames": total,
        "width": width,
        "height": height,
    }
