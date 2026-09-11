"""Hugging Face I/O helpers built on the ``hf`` CLI.

Provides:
- ``download_video`` — fetch a video from an ``hf://`` bucket or repo link.
- ``ensure_bucket``  — create a bucket if it does not already exist.
- ``upload_dir``     — upload the contents of a local directory to a bucket
  under a given remote prefix.

Supports the full ``hf://`` address space:
- ``hf://buckets/<namespace>/<bucket>/<path>/<file>``
- ``hf://datasets/<user>/<repo>/<path>/<file>``
- ``hf://models/<user>/<repo>/<path>/<file>``
- ``hf://spaces/<user>/<repo>/<path>/<file>``

Authentication is done by setting ``HF_TOKEN`` in the subprocess environment
(never on the command line).  A token passed to these functions wins over
the ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HF_CLI = shutil.which("hf") or "hf"
_DEFAULT_TIMEOUT = 7200  # large video downloads/uploads can be slow

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts"}
REPO_TYPES = {
    "datasets": "dataset",
    "models": "model",
    "spaces": "space",
}
BUCKET_PREFIX = "hf://buckets/"


class HfError(RuntimeError):
    """Raised when an ``hf`` CLI invocation fails."""


def _env(token: str | None) -> dict:
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
    return env


def _run(
    args: list[str],
    token: str | None = None,
    check: bool = True,
    timeout: int = _DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run an ``hf`` CLI command with HF_TOKEN set in the environment."""
    try:
        proc = subprocess.run(
            [HF_CLI, *args],
            capture_output=True,
            text=True,
            env=_env(token),
            timeout=timeout,
        )
    except FileNotFoundError:
        raise HfError(
            f"hf CLI not found. Install it with: pip install huggingface-hub[cli]"
        )
    except subprocess.TimeoutExpired:
        raise HfError(f"hf {' '.join(args)} timed out after {timeout}s")
    if check and proc.returncode != 0:
        raise HfError(
            f"hf {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr[-800:]}"
        )
    return proc


# ---------------------------------------------------------------------------
# Link parsing
# ---------------------------------------------------------------------------

def parse_hf_link(link: str) -> dict:
    """Parse an ``hf://`` link into (kind, fields).

    Returns a dict with keys:
      kind   : "bucket" or "repo"
      bucket : bucket id (for kind=bucket)
      path   : remote path within the bucket (for kind=bucket)
      user   : repo owner (for kind=repo)
      repo   : repo name (for kind=repo)
      path   : path within the repo (for kind=repo)
    """
    link = link.strip()
    if not link.startswith("hf://"):
        raise ValueError(f"Link must start with hf:// — got: {link}")

    rest = link[len("hf://"):]
    parts = rest.split("/")
    kind = parts[0]

    if kind == "buckets":
        if len(parts) < 3:
            raise ValueError(
                f"Bucket link must be hf://buckets/<namespace>/<bucket>[/path]: {link}"
            )
        return dict(
            kind="bucket",
            bucket=f"{parts[1]}/{parts[2]}",
            path="/".join(parts[3:]),
        )

    if kind in REPO_TYPES:
        if len(parts) < 3:
            raise ValueError(
                f"{kind} link must be hf://{kind}/<user>/<repo>[/path]: {link}"
            )
        return dict(
            kind="repo",
            repo_type=REPO_TYPES[kind],
            user=parts[1],
            repo=parts[2],
            path="/".join(parts[3:]),
        )

    raise ValueError(f"Unsupported hf:// kind '{kind}' (expected buckets, datasets, models, spaces)")


def link_basename(link: str) -> str:
    """Return the file name at the end of an hf:// link."""
    parts = [p for p in link.rstrip("/").split("/")]
    return parts[-1] if parts else ""


def is_local_path(path: str) -> bool:
    """True if the string looks like a local filesystem path, not an hf:// link."""
    return not path.strip().startswith("hf://")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_link(
    link: str,
    dest_dir: str | Path,
    token: str | None = None,
) -> Path:
    """Download a file from an ``hf://`` link into *dest_dir*.

    Returns the path of the downloaded file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_hf_link(link)

    if parsed["kind"] == "bucket":
        _run(
            ["buckets", "cp", link, str(dest_dir) + os.sep],
            token=token,
        )
    else:
        cmd = [
            "download",
            f"{parsed['user']}/{parsed['repo']}",
            parsed["path"],
            "--type", parsed["repo_type"],
            "--local-dir", str(dest_dir),
        ]
        _run(cmd, token=token)

    return _locate_download(parsed, link, dest_dir)


def _locate_download(parsed: dict, link: str, dest_dir: Path) -> Path:
    """Find the file that a download produced inside dest_dir."""
    # Preferred: exact basename from the original link.
    name = link_basename(link)
    if name:
        candidate = dest_dir / name
        if candidate.is_file():
            return candidate
    # Fall back: if exactly one file was downloaded, use it.
    files = sorted(p for p in dest_dir.rglob("*") if p.is_file())
    if len(files) == 1:
        return files[0]
    if not files:
        raise HfError(f"Download produced no file: {link}")
    # Multiple files: pick the one that looks like a video, else the largest.
    vids = [f for f in files if f.suffix.lower() in VIDEO_SUFFIXES]
    if len(vids) == 1:
        return vids[0]
    return max(files, key=lambda f: f.stat().st_size)


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

def bucket_exists(bucket_id: str, token: str | None = None) -> bool:
    """Return True if the bucket exists (via ``hf buckets info``)."""
    try:
        _run(["buckets", "info", bucket_id], token=token, check=True, timeout=60)
        return True
    except HfError:
        return False


def ensure_bucket(
    bucket_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Create the bucket if it does not already exist.

    Returns the canonical bucket id (namespace may be added by HF, e.g.
    ``user/bucket``).
    """
    if bucket_exists(bucket_id, token=token):
        return bucket_id
    cmd = ["buckets", "create", bucket_id, "--exist-ok"]
    if private:
        cmd.append("--private")
    proc = _run(cmd, token=token, timeout=120)
    return _bucket_id_from_output(proc.stdout, bucket_id)


def _bucket_id_from_output(stdout: str, fallback: str) -> str:
    """Extract the bucket id from ``hf buckets create`` output.

    The CLI prints a single line like::

        uri=hf://buckets/hamimmahmud0/test-bucket url=https://huggingface.co/buckets/hamimmahmud0/test-bucket

    Returns the ``...`` part of ``uri=hf://buckets/...``.
    """
    for line in stdout.splitlines():
        line = line.strip()
        for tok in line.split():
            if tok.startswith("uri=hf://buckets/"):
                return tok[len("uri=hf://buckets/"):]
    return fallback


def upload_file(
    local_file: str | Path,
    bucket_id: str,
    remote_path: str,
    token: str | None = None,
) -> None:
    """Upload a single local file to ``hf://buckets/<bucket>/<remote_path>``."""
    local_file = Path(local_file)
    if not local_file.is_file():
        raise FileNotFoundError(f"No such file to upload: {local_file}")
    uri = f"hf://buckets/{bucket_id.rstrip('/')}/{remote_path.lstrip('/')}"
    _run(["buckets", "cp", str(local_file), uri], token=token)


def upload_dir(
    local_dir: str | Path,
    bucket_id: str,
    remote_prefix: str,
    token: str | None = None,
) -> list[str]:
    """Upload every file in *local_dir* to ``hf://buckets/<bucket>/<remote_prefix>``.

    Returns the list of remote URI paths that were uploaded.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"No such directory to upload: {local_dir}")

    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise HfError(f"No files found in {local_dir} to upload")

    uploaded = []
    for f in files:
        rel = f.relative_to(local_dir)
        remote = f"{remote_prefix.rstrip('/')}/{rel.as_posix()}"
        upload_file(f, bucket_id, remote, token=token)
        uploaded.append(remote)

    return uploaded