#!/usr/bin/env python3
"""CoTracker3 trajectory-extraction inference backend (FastAPI).

Loads the CoTracker3 online model in a GPU worker process and tracks
keypoints through a video. Agents talk to this server indirectly through
the CoTracker3 MCP server (cot_mcp_server.py), which submits jobs here.

Endpoints:
  POST /add_to_track_queue  (multipart video + grid params) -> job_id
  GET  /jobs/{job_id}       poll job status / result
  GET  /health              worker + queue state

Guardrails:
  - Bounded task queue (COT_MAX_QUEUE); full queue -> HTTP 503.
  - Job TTL cleanup (COT_JOB_TTL) so the in-memory jobs dict cannot grow.
  - Per-job caps: frames (COT_MAX_FRAMES), total upload bytes
    (COT_MAX_TOTAL_UPLOAD). Grid size is UNLOCKED (no default cap); set
    COT_MAX_GRID_SIZE > 0 to enforce an operator ceiling.
  - Videos are resized down to at most COT_MAX_VIDEO_DIM on the longest side
    before tracking (bounded GPU memory); returned tracks are mapped back to
    the original video coordinates.
  - CUDA OOM recovery: on torch.cuda.OutOfMemoryError the frame budget is
    halved and retried; torch.cuda.empty_cache() after every forward.
  - Path traversal guarded: uploaded filenames sanitized via basename.
  - Any per-job error is reported as job status "failed" with a message;
    the worker loop never dies.
"""

import base64
import multiprocessing as mp
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from cotracker.models.core.model_utils import get_points_on_a_grid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

PORT = int(os.environ.get("COT_SERVER_PORT", "8003"))
PROJECT_DIR = Path(
    os.environ.get("COT_PROJECT_DIR", "/root/cotracker3")
).resolve()
CHECKPOINT_PATH = Path(
    os.environ.get(
        "COT_CHECKPOINT_PATH",
        PROJECT_DIR / "checkpoints/scaled_online.pth",
    )
).resolve()
UPLOAD_ROOT = Path(os.environ.get("COT_UPLOAD_DIR", tempfile.gettempdir())) / "cot3-api"

# Guardrails
MAX_QUEUE_SIZE = int(os.environ.get("COT_MAX_QUEUE", "4"))
MAX_TOTAL_UPLOAD_BYTES = int(
    os.environ.get("COT_MAX_TOTAL_UPLOAD", str(512 * 1024 * 1024))
)
JOB_TTL_SECONDS = int(os.environ.get("COT_JOB_TTL", str(30 * 60)))  # 30 min
MAX_FRAMES = int(os.environ.get("COT_MAX_FRAMES", "600"))
# Grid size is UNLOCKED: no default cap (any NxN grid is accepted). Set
# COT_MAX_GRID_SIZE > 0 to re-enable an operator ceiling.
MAX_GRID_SIZE = int(os.environ.get("COT_MAX_GRID_SIZE", "0"))
MAX_VIDEO_DIM = int(os.environ.get("COT_MAX_VIDEO_DIM", "1280"))
MIN_FRAMES = 2

app = FastAPI(title="CoTracker3 inference server")
jobs = {}
jobs_lock = threading.Lock()
task_queue = None
result_queue = None
workers = []
collector_thread = None
cleanup_thread = None


def encode_float_array(arr):
    """zlib+base64 encoding for float32 arrays (compact JSON payloads)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "dtype": "float32",
        "encoding": "zlib+base64",
        "data": base64.b64encode(zlib.compress(arr.tobytes())).decode("ascii"),
    }


def encode_bool_array(arr):
    """zlib+base64 encoding for boolean arrays (packed bits)."""
    packed = np.packbits(np.ascontiguousarray(arr, dtype=np.uint8).reshape(-1))
    return {
        "shape": list(arr.shape),
        "dtype": "bool",
        "encoding": "zlib+base64+packbits",
        "data": base64.b64encode(zlib.compress(packed.tobytes())).decode("ascii"),
    }


def read_video_bounded(video_path, max_frames):
    """Read up to max_frames frames from a video, evenly sampled, with a
    longest-side cap (MAX_VIDEO_DIM). Returns frames (T,H,W,3 uint8), and
    (orig_w, orig_h, fps, total)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video file: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if total <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")
    if total < MIN_FRAMES:
        raise ValueError(
            f"Video too short: {total} frame(s), need at least {MIN_FRAMES}"
        )

    scale = min(1.0, MAX_VIDEO_DIM / max(orig_w, orig_h))
    tw = max(1, int(round(orig_w * scale)) // 2 * 2)
    th = max(1, int(round(orig_h * scale)) // 2 * 2)

    if total <= max_frames:
        indices = list(range(total))
    else:
        indices = sorted({round(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)})

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if scale < 1.0:
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) < MIN_FRAMES:
        raise ValueError(f"Only {len(frames)} frame(s) could be read from {video_path}")
    return np.stack(frames), orig_w, orig_h, fps, total


def read_image_bounded(image_path, max_dim=MAX_VIDEO_DIM):
    """Read a single image as one frame (T=1), with longest-side cap.
    Returns (frame (1,H,W,3 uint8), orig_w, orig_h)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not open image file: {image_path}")
    orig_h, orig_w = img.shape[:2]
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    tw = max(1, int(round(orig_w * scale)) // 2 * 2)
    th = max(1, int(round(orig_h * scale)) // 2 * 2)
    if scale < 1.0:
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.stack([img]), orig_w, orig_h


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _clamp_grid_size(grid_size):
    """Sanitize grid_size: >= 1; ceiling only if COT_MAX_GRID_SIZE > 0
    (unlocked by default)."""
    grid_size = max(1, int(grid_size))
    if MAX_GRID_SIZE > 0:
        grid_size = min(grid_size, MAX_GRID_SIZE)
    return grid_size


def _is_image(path):
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def _track_image(model, task, device):
    """Static-image path: no temporal tracking is possible on a single frame,
    so the grid keypoints on the image are returned directly. Mirrors the
    model's grid-query computation (get_points_on_a_grid) so the keypoint
    semantics match the video path exactly."""
    frame, orig_w, orig_h = read_image_bounded(task["video_path"])
    grid_size = _clamp_grid_size(task["grid_size"])
    ish = model.interp_shape
    pts = get_points_on_a_grid(grid_size, ish)
    pts = pts[0].cpu().numpy()  # (N, 2) in interp coords
    pts[:, 0] *= (orig_w - 1) / (ish[1] - 1)
    pts[:, 1] *= (orig_h - 1) / (ish[0] - 1)
    tracks = pts[None, :, :].astype(np.float32)  # (1, N, 2)
    visibility = np.ones((1, grid_size * grid_size), dtype=bool)
    return {
        "job_id": task["job_id"],
        "kind": "image",
        "width": orig_w,
        "height": orig_h,
        "fps": 0.0,
        "total_frames": 1,
        "processed_frames": 1,
        "grid_size": grid_size,
        "grid_query_frame": 0,
        "num_points": tracks.shape[1],
        "query_points": pts.round(2).tolist(),
        "tracks": encode_float_array(tracks),
        "visibility": encode_bool_array(visibility),
    }


def run_tracking(model, task, device):
    """Run CoTracker3 online tracking over the job video. Returns a JSON-ready
    result dict with encoded tracks/visibility in ORIGINAL video coordinates.

    Mirrors the sliding-window streaming loop of online_demo.py: the model
    consumes chunks of at most window_len (16) frames and accumulates
    predictions across calls; the final call returns tracks for every frame.

    Frame selection: max_frames <= 0 means NO sampling - every frame of the
    video is tracked (the default). A positive max_frames evenly samples that
    many frames when the video is longer. Frames are streamed from disk with a
    bounded in-memory window so long 4K videos do not exhaust host RAM.

    For a static image input (T=1), tracking is degenerate: the grid query
    keypoints on the image are returned directly (all visible), matching the
    grid semantics of the video path.
    """
    if _is_image(task["video_path"]):
        return _track_image(model, task, device)

    video_path = task["video_path"]
    grid_size = _clamp_grid_size(task["grid_size"])
    max_frames = int(task["max_frames"])

    # Probe the video once.
    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()
    if total <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")
    if total < MIN_FRAMES:
        raise ValueError(
            f"Video too short: {total} frame(s), need at least {MIN_FRAMES}"
        )

    scale = min(1.0, MAX_VIDEO_DIM / max(orig_w, orig_h))
    tw = max(1, int(round(orig_w * scale)) // 2 * 2)
    th = max(1, int(round(orig_h * scale)) // 2 * 2)

    # max_frames <= 0 -> process EVERY frame (no sampling, the default).
    if max_frames and total > max_frames:
        indices = sorted(
            {round(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)}
        )
    else:
        indices = None

    grid_query_frame = max(0, min(int(task["grid_query_frame"]), total - 1))

    window = []

    def _process_step(is_first_step):
        chunk = np.stack(window[-model.step * 2 :])
        video_chunk = (
            torch.from_numpy(chunk).float().permute(0, 3, 1, 2)[None].to(device)
        )
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    # Stream frames from disk, keeping only a bounded window in memory.
    is_first_step = True
    n = 0
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if indices is not None and frame_idx not in indices:
                frame_idx += 1
                continue
            if scale < 1.0:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            if n % model.step == 0 and n != 0:
                _process_step(is_first_step)
                is_first_step = False
            window.append(frame)
            if len(window) > model.step * 2:
                window = window[-model.step * 2 :]
            n += 1
            frame_idx += 1
    finally:
        cap.release()

    if n < MIN_FRAMES:
        raise ValueError(f"Only {n} frame(s) could be read from {video_path}")

    pred_tracks, pred_visibility = _process_step(is_first_step)
    if pred_tracks is None:
        # Video shorter than one window: the step above only initialized the
        # queries. Run inference on the same chunk now.
        pred_tracks, pred_visibility = _process_step(False)

    query_pts = model.queries[0, :, 1:].detach().float().cpu().numpy()
    ish = model.interp_shape
    query_pts[:, 0] *= (orig_w - 1) / (ish[1] - 1)
    query_pts[:, 1] *= (orig_h - 1) / (ish[0] - 1)

    tracks = pred_tracks[0].detach().float().cpu().numpy()  # (T, N, 2) resized coords
    visibility = pred_visibility[0].detach().cpu().numpy()  # (T, N) bool
    T = tracks.shape[0]

    sx = (orig_w - 1) / max(tw - 1, 1)
    sy = (orig_h - 1) / max(th - 1, 1)
    tracks[:, :, 0] *= sx
    tracks[:, :, 1] *= sy

    return {
        "job_id": task["job_id"],
        "kind": "video",
        "width": orig_w,
        "height": orig_h,
        "fps": round(float(fps), 3),
        "total_frames": total,
        "processed_frames": T,
        "grid_size": grid_size,
        "grid_query_frame": grid_query_frame,
        "num_points": tracks.shape[1],
        "query_points": query_pts.round(2).tolist(),
        "tracks": encode_float_array(tracks),
        "visibility": encode_bool_array(visibility),
    }


def gpu_worker(gpu_id, incoming, outgoing):
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))
    torch.cuda.set_device(gpu_id)

    from cotracker.predictor import CoTrackerOnlinePredictor

    device = torch.device(f"cuda:{gpu_id}")
    model = CoTrackerOnlinePredictor(
        checkpoint=str(CHECKPOINT_PATH), offline=False, window_len=16
    )
    model = model.to(device).eval()
    outgoing.put({"kind": "worker_ready", "gpu_id": gpu_id})

    while True:
        task = incoming.get()
        if task is None:
            return

        job_id = task["job_id"]
        outgoing.put({"kind": "running", "job_id": job_id, "gpu_id": gpu_id})
        try:
            result = run_tracking(model, task, device)
            outgoing.put({"kind": "completed", "job_id": job_id, "gpu_id": gpu_id, "result": result})
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            # Retry once with a reduced frame budget. For all-frames jobs
            # (max_frames == 0) fall back to a capped sampled budget.
            if task["max_frames"] == 0:
                new_max = MAX_FRAMES
            else:
                new_max = max(MIN_FRAMES, task["max_frames"] // 2)
            if new_max != task["max_frames"]:
                task = dict(task)
                task["max_frames"] = new_max
                try:
                    result = run_tracking(model, task, device)
                    outgoing.put(
                        {"kind": "completed", "job_id": job_id, "gpu_id": gpu_id, "result": result}
                    )
                    continue
                except Exception as exc:
                    outgoing.put(
                        {
                            "kind": "failed",
                            "job_id": job_id,
                            "gpu_id": gpu_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            else:
                outgoing.put(
                    {
                        "kind": "failed",
                        "job_id": job_id,
                        "gpu_id": gpu_id,
                        "error": f"torch.cuda.OutOfMemoryError: {oom}",
                    }
                )
        except Exception as exc:
            outgoing.put(
                {
                    "kind": "failed",
                    "job_id": job_id,
                    "gpu_id": gpu_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            shutil.rmtree(task.get("upload_dir", ""), ignore_errors=True)


def collect_worker_messages():
    while True:
        try:
            message = result_queue.get(timeout=5)
        except queue.Empty:
            continue
        if message is None:
            return
        kind = message["kind"]
        if kind == "worker_ready":
            with jobs_lock:
                jobs.setdefault("_workers", {})[str(message["gpu_id"])] = "ready"
            continue

        job_id = message["job_id"]
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                continue
            job["status"] = kind
            job["gpu_id"] = message.get("gpu_id")
            job["updated_at"] = time.time()
            if kind == "completed":
                job["result"] = message["result"]
            elif kind == "failed":
                job["error"] = message["error"]


def cleanup_old_jobs():
    while True:
        time.sleep(60)
        now = time.time()
        with jobs_lock:
            for job_id in list(jobs.keys()):
                if job_id == "_workers":
                    continue
                job = jobs[job_id]
                if job["status"] in {"completed", "failed"} and (
                    now - job.get("updated_at", 0) > JOB_TTL_SECONDS
                ):
                    jobs.pop(job_id, None)


@app.on_event("startup")
def startup():
    global task_queue, result_queue, workers, collector_thread, cleanup_thread
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=MAX_QUEUE_SIZE)
    result_queue = ctx.Queue()

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPUs were found")

    for gpu_id in range(gpu_count):
        process = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue),
            daemon=True,
        )
        process.start()
        workers.append(process)

    collector_thread = threading.Thread(target=collect_worker_messages, daemon=True)
    collector_thread.start()

    cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
    cleanup_thread.start()


@app.on_event("shutdown")
def shutdown():
    for _ in workers:
        task_queue.put(None)
    result_queue.put(None)
    for process in workers:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()


@app.get("/health")
def health():
    with jobs_lock:
        ready_workers = len(jobs.get("_workers", {}))
    try:
        qsize = task_queue.qsize() if task_queue is not None else 0
    except Exception:
        qsize = 0
    return {
        "status": "ready" if ready_workers == len(workers) else "loading",
        "port": PORT,
        "gpu_count": len(workers),
        "ready_workers": ready_workers,
        "queue_size": qsize,
        "max_queue_size": MAX_QUEUE_SIZE,
    }


@app.post("/add_to_track_queue", status_code=202)
async def add_to_track_queue(
    video: UploadFile = File(...),
    grid_size: int = Form(16),
    grid_query_frame: int = Form(0),
    max_frames: int = Form(0),  # 0 = all frames (no sampling)
    local_path: str = Form(""),
):
    if not video.filename and not local_path:
        raise HTTPException(status_code=400, detail="video file is required")
    if grid_size < 1:
        raise HTTPException(status_code=400, detail="grid_size must be >= 1")
    if MAX_GRID_SIZE > 0 and grid_size > MAX_GRID_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"grid_size must be <= {MAX_GRID_SIZE} (COT_MAX_GRID_SIZE)",
        )
    if grid_query_frame < 0:
        raise HTTPException(status_code=400, detail="grid_query_frame must be >= 0")
    if max_frames != 0 and not MIN_FRAMES <= max_frames <= MAX_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"max_frames must be 0 (all frames) or between {MIN_FRAMES} and {MAX_FRAMES}",
        )

    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True)
    if local_path:
        # File already present on the local filesystem (e.g. downloaded by the
        # MCP layer from hf://). Bypass the upload size cap entirely; the
        # backend reads it in place. The MCP layer guarantees a sanitized
        # absolute path under its own work dir.
        candidate = Path(local_path).resolve()
        if not candidate.is_file():
            raise HTTPException(status_code=400, detail=f"local file not found: {candidate}")
        video_path = candidate
        total_bytes = candidate.stat().st_size
    else:
        safe_name = Path(video.filename or "").name or f"{job_id}.mp4"
        video_path = upload_dir / safe_name
        total_bytes = 0
        try:
            with video_path.open("wb") as destination:
                while chunk := await video.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413, detail="Total upload exceeds size limit"
                        )
                    destination.write(chunk)

            if total_bytes == 0:
                raise HTTPException(status_code=400, detail="Uploaded video is empty")

        except HTTPException:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise

    now = time.time()
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "gpu_id": None,
            "created_at": now,
            "updated_at": now,
        }
    try:
        task_queue.put_nowait(
            {
                "job_id": job_id,
                "upload_dir": str(upload_dir),
                "video_path": str(video_path),
                "grid_size": grid_size,
                "grid_query_frame": grid_query_frame,
                "max_frames": max_frames,
            }
        )
    except queue.Full:
        shutil.rmtree(upload_dir, ignore_errors=True)
        with jobs_lock:
            jobs.pop(job_id, None)
        raise HTTPException(status_code=503, detail="Inference queue is full")
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        with jobs_lock:
            jobs.pop(job_id, None)
        raise
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        with jobs_lock:
            jobs.pop(job_id, None)
        raise

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None
    # note: 404-style None returned so MCP can distinguish not-found
    # without raising (keeps pipeline resilient).


def main():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    mp.freeze_support()
    main()