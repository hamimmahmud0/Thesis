from __future__ import annotations
import json
import subprocess
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from .errors import PipelineError, classify_exception

@dataclass
class RunState:
    models: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    def is_complete(self, name: str) -> bool: return self.models.get(name, {}).get("status") == "complete"
    def all_complete(self, names) -> bool: return all(self.is_complete(n) for n in names)

class HubStore:
    """HF Bucket-backed durable state. Credentials only enter subprocess env."""
    def __init__(self, source_repo: str, bucket: str, token: str, work_dir: Path):
        self.source_repo, self.bucket, self.token = source_repo, bucket, token
        self.work_dir = work_dir; self.artifacts = work_dir / "artifacts"
        self.state_path = self.artifacts / "state.json"
    @property
    def env(self): return {"HF_TOKEN": self.token} if self.token else {}
    def _run(self, args: list[str], *, required=True) -> subprocess.CompletedProcess:
        import os
        result = subprocess.run(args, text=True, capture_output=True, env={**os.environ, **self.env})
        if required and result.returncode:
            message = (result.stderr or result.stdout).strip(); issue = classify_exception(RuntimeError(message))
            raise PipelineError(issue.code if issue.code != "unexpected" else "hub_error", message)
        return result
    def ensure_private_destination(self) -> None:
        if not self.token: raise PipelineError("invalid_hf_token", "HF token is missing")
        # Buckets are private by design; --exist-ok makes resume idempotent.
        self._run(["hf", "buckets", "create", self.bucket, "--exist-ok", "--private"])
    def restore(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True); self.artifacts.mkdir(parents=True, exist_ok=True)
        result = self._run(["hf", "buckets", "sync", f"hf://buckets/{self.bucket}", str(self.artifacts)], required=False)
        if result.returncode and "not found" not in (result.stderr + result.stdout).lower():
            raise PipelineError("hub_restore_failed", (result.stderr or result.stdout).strip())
    def sync(self) -> None:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._run(["hf", "buckets", "sync", str(self.artifacts), f"hf://buckets/{self.bucket}"])
    def download_source(self, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        source = self.source_repo.rstrip("/")
        if source.startswith("hf://buckets/"):
            bucket = source.removeprefix("hf://buckets/")
        elif "/buckets/" in source:
            bucket = urlparse(source).path.split("/buckets/", 1)[1]
        else:
            bucket = ""
        if bucket:
            self._run(["hf", "buckets", "sync", f"hf://buckets/{bucket}", str(destination)])
        else:
            self._run(["hf", "download", source, "--repo-type", "dataset", "--local-dir", str(destination)])
        return destination
    def read_state(self) -> RunState:
        if not self.state_path.exists(): return RunState()
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return RunState(raw.get("models", {}), raw.get("failures", []))
    def write_state(self, state: RunState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"models": state.models, "failures": state.failures}, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)
    def update_model(self, name: str, **values) -> RunState:
        state = self.read_state(); state.models.setdefault(name, {}).update(values); self.write_state(state); return state
    def append_failure(self, issue) -> None:
        state = self.read_state(); state.failures.append({"time": datetime.now(timezone.utc).isoformat(),
            "code": issue.code, "message": issue.safe_message}); self.write_state(state)
