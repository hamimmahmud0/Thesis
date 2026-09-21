from __future__ import annotations
import csv
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import urllib.request
import yaml
from .errors import PipelineError

class KaggleTokenPool:
    def __init__(self, tokens): self.tokens = tuple(tokens)
    def _identity(self, token: str) -> str:
        request = urllib.request.Request("https://www.kaggle.com/api/v1/oauth2/introspect",
            data=json.dumps({"token": token}).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response: return json.load(response)["username"]
        except Exception as exc: raise PipelineError("invalid_kaggle_token", "Kaggle token validation failed") from exc
    def select_available(self) -> str:
        if not self.tokens: raise PipelineError("kaggle_tokens_exhausted", "No Kaggle tokens configured")
        for token in self.tokens:
            try:
                self._identity(token)
                # CSV output has an explicit ``remaining`` column.  Parsing the
                # human-readable table is unsafe because its first number is the
                # used quota, which can be positive when remaining quota is zero.
                result = subprocess.run(["kaggle", "quota", "-v"], text=True, capture_output=True,
                    env={**os.environ, "KAGGLE_API_TOKEN": token})
                output = (result.stdout + result.stderr).lower()
                exhausted = "quota exceeded" in output or "no gpu quota" in output
                available = False
                if result.returncode == 0 and not exhausted:
                    reader = csv.DictReader(io.StringIO(result.stdout))
                    fields = {str(field).strip().lower() for field in (reader.fieldnames or [])}
                    if {"resource", "remaining"} <= fields:
                        for row in reader:
                            normalized = {str(key).strip().lower(): value for key, value in row.items()}
                            if normalized.get("resource", "").strip().lower() == "gpu":
                                try:
                                    remaining = str(normalized.get("remaining", "0")).strip().lower()
                                    available = float(remaining.removesuffix("h")) > 0
                                except (TypeError, ValueError): available = False
                                break
                    else:
                        # Compatibility with pre-CSV clients and mocked callers.
                        hours = [float(x) for x in re.findall(
                            r"(?:remaining|available)[^\n]*?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)?", output)]
                        available = max(hours, default=1.0) > 0
                if available:
                    return token
            except PipelineError: continue
        raise PipelineError("kaggle_tokens_exhausted", "No valid Kaggle token has available quota")

def _redacted_config(source: Path, destination: Path) -> None:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    for key in ("hf_token", "HF_TOKEN", "bot_token", "BOT_TOKEN", "KAGGLE_API_TOKEN"):
        if key in raw: raw[key] = f"env:{key.upper()}"
    if "kaggle_tokens" in raw: raw["kaggle_tokens"] = [f"env:KAGGLE_TOKEN_{i+1}" for i in range(len(raw["kaggle_tokens"]))]
    if "KAGGLE_TOKENS" in raw: raw["KAGGLE_TOKENS"] = [f"env:KAGGLE_TOKEN_{i+1}" for i in range(len(raw["KAGGLE_TOKENS"]))]
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

def launch_successor(config, token: str) -> None:
    username = KaggleTokenPool((token,))._identity(token)
    source_dir = config.path.parent
    tool_dir = source_dir / "tool"
    if not tool_dir.exists(): tool_dir = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="seg-handoff-") as temporary:
        target = Path(temporary); shutil.copytree(tool_dir, target / "tool")
        for filename in ("main.py", "bootstrap.py"):
            shutil.copy2(source_dir / filename, target / filename)
        _redacted_config(config.path, target / "config.yaml")
        metadata = {"id": f"{username}/{config.kernel_id}", "title": config.title,
            "code_file": "bootstrap.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
        (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        result = subprocess.run(["kaggle", "kernels", "push", "-p", str(target)], text=True,
            capture_output=True, env={**os.environ, "KAGGLE_API_TOKEN": token})
        if result.returncode: raise PipelineError("handoff_failed", (result.stderr or result.stdout).strip())
        info = config.work_dir / "artifacts" / "kernel.json"; info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps({"kernel": metadata["id"], "url": f"https://www.kaggle.com/code/{username}/{config.kernel_id}"}, indent=2), encoding="utf-8")
