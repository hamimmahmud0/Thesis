"""Hugging Face I/O helpers.

- ``https_download`` — stream a public HF ``resolve`` URL to disk with
  progress logging (checkpoint, videos).
- ``download_link``  — fetch a file from an ``hf://`` link via the ``hf`` CLI.
- ``ensure_bucket`` / ``upload_dir`` — push results to an HF storage bucket
  via the ``hf`` CLI (only needed when uploading).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote

import requests

HF_CLI = shutil.which("hf") or "hf"
TRANSFER_TIMEOUT = 7200
REPORT_EVERY_BYTES = 256 * 1024 * 1024

REPO_TYPES = {
    "datasets": "dataset",
    "models": "model",
    "spaces": "space",
}


class HfError(RuntimeError):
    """Raised when a transfer to/from Hugging Face fails."""


# ---------------------------------------------------------------------------
# Plain HTTPS downloads (public resolve URLs)
# ---------------------------------------------------------------------------

def https_download(
    url: str,
    dest: str | Path,
    token: str | None = None,
    label: str | None = None,
) -> Path:
    """Stream *url* to *dest* (via a ``.part`` file) with progress prints."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = label or dest.name

    headers = {"User-Agent": "sam31/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    part = dest.with_name(dest.name + ".part")
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        next_report = REPORT_EVERY_BYTES
        with open(part, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    if total:
                        print(
                            f"[download] {label}: {done / 1e9:.2f} / {total / 1e9:.2f} GB",
                            flush=True,
                        )
                    else:
                        print(f"[download] {label}: {done / 1e9:.2f} GB", flush=True)
                    next_report += REPORT_EVERY_BYTES
    part.replace(dest)
    print(f"[download] {label}: done ({done / 1e9:.2f} GB)", flush=True)
    return dest


# ---------------------------------------------------------------------------
# hf:// links (via the hf CLI)
# ---------------------------------------------------------------------------

def _env(token: str | None) -> dict:
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
    return env


def _run(
    args: list[str],
    token: str | None = None,
    check: bool = True,
    timeout: int = TRANSFER_TIMEOUT,
) -> subprocess.CompletedProcess:
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
            "hf CLI not found. Install it with: pip install 'huggingface-hub[cli]'"
        )
    except subprocess.TimeoutExpired:
        raise HfError(f"hf {' '.join(args)} timed out after {timeout}s")
    if check and proc.returncode != 0:
        raise HfError(
            f"hf {' '.join(args)} failed (rc={proc.returncode}): {proc.stderr[-800:]}"
        )
    return proc


def parse_hf_link(link: str) -> dict:
    """Parse an ``hf://`` link into its kind and fields."""
    link = link.strip()
    if not link.startswith("hf://"):
        raise ValueError(f"Link must start with hf:// — got: {link}")
    parts = link[len("hf://"):].split("/")

    if parts[0] == "buckets":
        if len(parts) < 3:
            raise ValueError(
                f"Bucket link must be hf://buckets/<namespace>/<bucket>[/path]: {link}"
            )
        return dict(kind="bucket", bucket=f"{parts[1]}/{parts[2]}", path="/".join(parts[3:]))

    if parts[0] in REPO_TYPES:
        if len(parts) < 3:
            raise ValueError(
                f"{parts[0]} link must be hf://{parts[0]}/<user>/<repo>[/path]: {link}"
            )
        return dict(
            kind="repo",
            repo_type=REPO_TYPES[parts[0]],
            user=parts[1],
            repo=parts[2],
            path="/".join(parts[3:]),
        )

    raise ValueError(
        f"Unsupported hf:// kind {parts[0]!r} (expected buckets, datasets, models, spaces)"
    )


def link_basename(link: str) -> str:
    """File name at the end of an hf:// or http(s):// link (decoded)."""
    name = link.split("?")[0].rstrip("/").split("/")[-1]
    return unquote(name)


def download_link(link: str, dest_dir: str | Path, token: str | None = None) -> Path:
    """Download a file from an ``hf://`` link into *dest_dir*; returns its path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_hf_link(link)

    if parsed["kind"] == "bucket":
        _run(["buckets", "cp", link, str(dest_dir) + os.sep], token=token)
    else:
        _run(
            [
                "download",
                f"{parsed['user']}/{parsed['repo']}",
                parsed["path"],
                "--type",
                parsed["repo_type"],
                "--local-dir",
                str(dest_dir),
            ],
            token=token,
        )

    name = link_basename(link)
    if name and (dest_dir / name).is_file():
        return dest_dir / name
    files = sorted(p for p in dest_dir.rglob("*") if p.is_file())
    if not files:
        raise HfError(f"Download produced no file: {link}")
    if len(files) == 1:
        return files[0]
    return max(files, key=lambda f: f.stat().st_size)


# ---------------------------------------------------------------------------
# Bucket upload
# ---------------------------------------------------------------------------

def bucket_exists(bucket_id: str, token: str | None = None) -> bool:
    try:
        _run(["buckets", "info", bucket_id], token=token, timeout=60)
        return True
    except HfError:
        return False


def ensure_bucket(
    bucket_id: str, token: str | None = None, private: bool = False
) -> str:
    """Create the bucket if missing; returns the canonical bucket id."""
    if bucket_exists(bucket_id, token=token):
        return bucket_id
    cmd = ["buckets", "create", bucket_id, "--exist-ok"]
    if private:
        cmd.append("--private")
    proc = _run(cmd, token=token, timeout=120)
    for line in proc.stdout.splitlines():
        for tok in line.split():
            if tok.startswith("uri=hf://buckets/"):
                return tok[len("uri=hf://buckets/"):]
    return bucket_id


def upload_dir(
    local_dir: str | Path,
    bucket_id: str,
    remote_prefix: str,
    token: str | None = None,
) -> list[str]:
    """Upload every file under *local_dir* to the bucket; returns remote paths."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"No such directory to upload: {local_dir}")
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise HfError(f"No files found in {local_dir} to upload")

    uploaded = []
    for f in files:
        rel = f.relative_to(local_dir).as_posix()
        remote = f"{remote_prefix.strip('/')}/{rel}"
        _run(
            ["buckets", "cp", str(f), f"hf://buckets/{bucket_id.strip('/')}/{remote}"],
            token=token,
        )
        uploaded.append(remote)
        print(f"[upload] {remote}", flush=True)
    return uploaded
