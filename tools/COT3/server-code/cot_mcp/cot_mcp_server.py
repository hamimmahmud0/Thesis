#!/usr/bin/env python3
"""CoTracker3 MCP server (streamable HTTP).

Exposes CoTracker3 keypoint-trajectory extraction as MCP tools. Agents
connect through a TunnelMate tunnel (public TCP address) and call tools to
track keypoints through videos. Inference runs on the separate FastAPI
backend (cot_tracker_server.py); this server submits jobs and returns
encoded trajectories.

Tools:
  - track_media: extract keypoint trajectories from a video. Query points are
    an NxN grid (grid_size) over the query frame (default frame 0); returned
    tracks are mapped back to ORIGINAL video coordinates.
  - server_health: backend + queue state.
  - list_capabilities: conventions for agents.

Input sources:
  - upload: base64-encoded file content (max 5 MB)
  - hf: an hf:// dataset/model/bucket file address, e.g.
        hf://datasets/user/repo/path/video.mp4 (any size; required for videos
        larger than 5 MB)

Guardrails:
  - Hard cap on concurrent inference requests (COT_MCP_MAX_CONCURRENT, default
    1 = one GPU worker). Requests beyond the cap are rejected immediately with
    a clear "busy" error instead of piling up.
  - Bounded work per request: max_frames and grid_size are clamped, uploads
    are size-limited, file names sanitized.
  - Every failure is returned as {"is_error": true, ...}; exceptions never
    escape the tool, so a bad request cannot kill the server.
"""

import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
# Request-body cap for the StreamableHTTP transport. Must exceed the 5 MB
# upload guardrail (base64 inflates ~4/3) so oversized uploads reach the
# tool and get a structured error instead of an HTTP 413. Hard bound above
# that still returns 413 from the transport layer.
MAX_BODY_BYTES = int(os.environ.get("COT_MCP_MAX_BODY_BYTES", str(16 * 1024 * 1024)))
COT_SERVER_URL = os.environ.get("COT_SERVER_URL", "http://127.0.0.1:8003")
WORK_DIR = Path(os.environ.get("COT_MCP_WORK_DIR", "/tmp/cot-mcp"))
HF_CLI = shutil.which("hf") or "/usr/local/bin/hf"

MAX_CONCURRENT = int(os.environ.get("COT_MCP_MAX_CONCURRENT", "1"))
_concurrency_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
MAX_FRAMES = int(os.environ.get("COT_MCP_MAX_FRAMES", "600"))
MAX_GRID_SIZE = int(os.environ.get("COT_MCP_MAX_GRID_SIZE", "32"))
POLL_TIMEOUT_S = int(os.environ.get("COT_MCP_POLL_TIMEOUT", "900"))


def _hf_download(address: str, dest_dir: Path) -> Path:
    """Download a single file from an hf:// address into dest_dir.

    Supports hf://datasets/..., hf://models/... and hf://buckets/... . For
    datasets/models the hf CLI needs a repo_id + filename, so the hf:// URL is
    parsed into its components; buckets use `hf buckets cp`.
    """
    if not address.startswith("hf://"):
        raise ValueError(
            "hf_address must start with hf:// (e.g. hf://datasets/user/repo/file.mp4)"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    if address.startswith("hf://buckets/"):
        cmd = [HF_CLI, "buckets", "cp", address, str(dest_dir)]
    else:
        parts = address[len("hf://") :].split("/")
        repo_type = parts[0]  # datasets | models | spaces
        repo_owner = parts[1]
        repo_name = parts[2] if len(parts) > 2 else ""
        if not repo_owner or not repo_name:
            raise ValueError(f"Could not parse repo from hf_address: {address}")
        repo_id = f"{repo_owner}/{repo_name}"
        rel_path = "/".join(parts[3:])
        if not rel_path:
            raise ValueError(f"hf_address must point to a file: {address}")
        cmd = [
            HF_CLI, "download", repo_id, rel_path,
            "--type", repo_type.rstrip("s") if repo_type != "spaces" else "space",
            "--local-dir", str(dest_dir),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"hf download failed: {proc.stderr[-500:]}")
    files = sorted(p for p in dest_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"hf download produced no file for {address}")
    if len(files) == 1:
        return files[0]
    name = address.rstrip("/").split("/")[-1]
    for f in files:
        if f.name == name:
            return f
    return files[0]


def _submit_job(video_path: Path, grid_size: int, grid_query_frame: int, max_frames: int, local_path: str = "") -> dict:
    import requests

    data = {
        "grid_size": str(grid_size),
        "grid_query_frame": str(grid_query_frame),
        "max_frames": str(max_frames),
    }
    if local_path:
        # File already on the local filesystem (e.g. downloaded from hf://).
        # Pass the path so the backend reads it in place, bypassing the
        # upload size cap entirely.
        resp = requests.post(
            f"{COT_SERVER_URL}/add_to_track_queue",
            files={"video": (video_path.name, b"", "video/mp4")},
            data={**data, "local_path": local_path},
            timeout=180,
        )
    else:
        with open(video_path, "rb") as f:
            resp = requests.post(
                f"{COT_SERVER_URL}/add_to_track_queue",
                files={"video": (video_path.name, f, "video/mp4")},
                data=data,
                timeout=180,
            )
    resp.raise_for_status()
    return resp.json()


def _poll_job(job_id: str, timeout: int = POLL_TIMEOUT_S) -> dict:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{COT_SERVER_URL}/jobs/{job_id}", timeout=30)
        if resp.status_code == 404:
            time.sleep(1)
            continue
        resp.raise_for_status()
        job = resp.json()
        if job is None:
            time.sleep(1)
            continue
        status = job.get("status")
        if status == "completed":
            return job["result"]
        if status == "failed":
            raise RuntimeError(f"Tracking failed: {job.get('error')}")
        time.sleep(1)
    raise TimeoutError(f"Tracking job {job_id} timed out after {timeout}s")


def _track_media_impl(
    source: str,
    file_name: str,
    file_b64: str,
    hf_address: str,
    grid_size: int,
    grid_query_frame: int,
    max_frames: int,
) -> dict:
    work = WORK_DIR / f"job-{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        source = source.lower()
        if source == "upload":
            if file_b64 is None:
                return {"is_error": True, "error": "file_b64 is required when source=upload"}
            if not file_name:
                return {"is_error": True, "error": "file_name is required when source=upload"}
            safe_name = Path(file_name).name
            if not safe_name:
                return {"is_error": True, "error": "file_name must include a filename"}
            try:
                raw = base64.b64decode(file_b64, validate=True)
            except Exception:
                return {"is_error": True, "error": "file_b64 is not valid base64"}
            if len(raw) > MAX_UPLOAD_BYTES:
                return {
                    "is_error": True,
                    "error": (
                        f"Uploaded file is {len(raw) / 1e6:.1f} MB, larger than the "
                        f"5 MB upload limit. Use source=hf with hf_address instead."
                    ),
                }
            if len(raw) == 0:
                return {"is_error": True, "error": "Uploaded file is empty"}
            video_path = work / safe_name
            video_path.write_bytes(raw)
            local_path = ""
        elif source == "hf":
            if not hf_address:
                return {"is_error": True, "error": "hf_address is required when source=hf"}
            video_path = _hf_download(hf_address, work)
            local_path = str(video_path)
        else:
            return {"is_error": True, "error": "source must be 'upload' or 'hf'"}

        if not video_path.is_file():
            return {"is_error": True, "error": f"File not found: {video_path}"}
        if video_path.stat().st_size == 0:
            return {"is_error": True, "error": "File is empty"}

        suffix = video_path.suffix.lower()
        if suffix not in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif",
        }:
            return {
                "is_error": True,
                "error": (
                    "Unsupported file type; supported: mp4/mov/avi/mkv/webm/m4v "
                    "(video) or jpg/jpeg/png/webp/bmp/tiff (image)"
                ),
            }

        grid_size = max(1, min(int(grid_size), MAX_GRID_SIZE))
        grid_query_frame = max(0, int(grid_query_frame))
        max_frames = int(max_frames)
        if max_frames != 0:
            max_frames = max(2, min(max_frames, MAX_FRAMES))

        job = _submit_job(video_path, grid_size, grid_query_frame, max_frames, local_path)
        result = _poll_job(job["job_id"])
        return result
    except Exception as exc:
        return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def register_tools(mcp):
    @mcp.tool()
    def track_media(
        source: str = "upload",
        file_name: str = "",
        file_b64: str | None = None,
        hf_address: str = "",
        grid_size: int = 16,
        grid_query_frame: int = 0,
        max_frames: int = 0,  # 0 = all frames (no sampling)
    ) -> dict:
        """Extract keypoint trajectories from a video with CoTracker3.

        Args:
            source: "upload" (base64 file, max 5 MB) or "hf" (file address on Hugging Face).
            file_name: file name including extension, required when source="upload".
            file_b64: base64-encoded file content, required when source="upload".
            hf_address: hf:// dataset/model/bucket file address, required when
                source="hf". Examples:
                hf://datasets/user/repo/videos/clip.mp4
                hf://buckets/user/my-bucket/data.mp4
            grid_size: NxN grid of keypoints over the query frame (clamped to 32).
                grid_size=16 -> 256 keypoints, 1 per 16x16 grid cell on a 512x512
                reference grid.
            grid_query_frame: frame index the grid is sampled from (default 0).
            max_frames: maximum number of frames to track. 0 (default) tracks
                EVERY frame (no sampling); a positive value evenly samples that
                many frames when the video is longer (clamped).
        Returns:
            Metadata (width, height, fps, frame counts, num_points, query_points)
            plus encoded per-frame trajectories:
              tracks: zlib+base64 float32 array, shape [T, N, 2] (x,y in ORIGINAL
                      video pixels, per frame)
              visibility: zlib+base64 packbits bool array, shape [T, N]
            Decode: np.frombuffer(zlib.decompress(base64.b64decode(data)),
                      dtype=np.float32).reshape(shape)
        """
        if not 1 <= grid_size <= MAX_GRID_SIZE:
            return {
                "is_error": True,
                "error": f"grid_size must be between 1 and {MAX_GRID_SIZE}",
            }
        if grid_query_frame < 0:
            return {"is_error": True, "error": "grid_query_frame must be >= 0"}
        if max_frames != 0 and not 2 <= max_frames <= MAX_FRAMES:
            return {
                "is_error": True,
                "error": f"max_frames must be 0 (all frames) or between 2 and {MAX_FRAMES}",
            }

        if not _concurrency_slots.acquire(blocking=False):
            return {
                "is_error": True,
                "error": (
                    f"Server at capacity: max {MAX_CONCURRENT} concurrent inference "
                    "requests. Please retry after the in-flight request completes."
                ),
            }
        try:
            return _track_media_impl(
                source=source,
                file_name=file_name,
                file_b64=file_b64,
                hf_address=hf_address,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                max_frames=max_frames,
            )
        finally:
            _concurrency_slots.release()

    @mcp.tool()
    def server_health() -> dict:
        """Check the CoTracker3 inference backend health and queue state."""
        import requests

        try:
            resp = requests.get(f"{COT_SERVER_URL}/health", timeout=15)
            resp.raise_for_status()
            health = resp.json()
            health["mcp_max_concurrent"] = MAX_CONCURRENT
            health["mcp_available_slots"] = _concurrency_slots._value
            return health
        except Exception as exc:
            return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool()
    def list_capabilities() -> dict:
        """Describe what this MCP server can do and how to call the tools."""
        return {
            "server": "CoTracker3 MCP trajectory-extraction server",
            "tools": [
                {
                    "name": "track_media",
                    "description": (
                        "Extract keypoint trajectories from a video. Keypoints are an "
                        "NxN grid over the query frame (grid_size, default 16 = 256 "
                        "keypoints). Returns per-frame trajectories (T,N,2) in original "
                        "video pixels plus visibility (T,N). Small files (<=5 MB) can be "
                        "uploaded as base64; larger files must use an hf:// address."
                    ),
                    "input_size_limit": "upload: <=5 MB, hf: unlimited",
                },
                {
                    "name": "server_health",
                    "description": "Check inference backend health and queue state",
                },
                {
                    "name": "list_capabilities",
                    "description": "Describe available tools and conventions",
                },
            ],
            "conventions": [
                "By default track_media tracks EVERY frame of the video (max_frames=0); "
                "pass a positive max_frames to evenly sample fewer frames.",
                "For files > 5 MB use source=hf with an hf:// address.",
                "Tracks are zlib+base64 float32 [T,N,2]; visibility is zlib+base64+packbits [T,N].",
                "The server accepts at most 1 concurrent inference request; when busy, "
                "track_media returns is_error with a retry-after message.",
                "Videos are downscaled (longest side <= 1280) before tracking for bounded "
                "GPU memory; returned tracks are mapped back to original coordinates.",
            ],
        }


def main():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "cotracker3-tracking",
        host="0.0.0.0",
        port=int(os.environ.get("COT_MCP_PORT", "8004")),
        streamable_http_path="/mcp",
        max_request_body_size=MAX_BODY_BYTES,
    )
    register_tools(mcp)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()