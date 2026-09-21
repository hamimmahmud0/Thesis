#!/usr/bin/env python3
"""Resumable Kaggle entrypoint for segmentation training."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from segpipe.config import ConfigError, load_config
from segpipe.errors import PipelineError, classify_exception
from segpipe.kaggle import KaggleTokenPool, launch_successor
from segpipe.notify import Notifier
from segpipe.prepare import prepare_dataset
from segpipe.storage import HubStore


def _run(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"})
        assert process.stdout is not None
        tail = deque(maxlen=80)
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
            tail.append(line)
        code = process.wait()
        if code:
            issue = classify_exception(RuntimeError("".join(tail)))
            raise PipelineError(issue.code if issue.code != "unexpected" else "training_failed", issue.safe_message)
        return code


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def _check_disk(config) -> None:
    for path, minimum in ((config.work_dir, config.min_work_free_gb), (Path("/root"), config.min_root_free_gb)):
        if path.exists() and _free_gib(path) < minimum:
            raise PipelineError("disk_full", f"Only {_free_gib(path):.1f} GiB free at {path}")


def _training_command(config, model, dataset_dir: Path, deadline: float) -> list[str]:
    common = ["--config", str(config.path), "--model", model.name,
              "--dataset", str(dataset_dir), "--deadline", str(deadline)]
    if model.family == "yolo":
        return [sys.executable, "-m", "segpipe.train_yolo", *common]
    return ["torchrun", "--standalone", "--nproc_per_node=2",
            "-m", "segpipe.train_semantic", *common]


def run(config_path: Path, allow_local: bool = False, dry_run: bool = False) -> int:
    if not allow_local and not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ConfigError("Training may only execute inside a Kaggle kernel")
    config = load_config(config_path)
    notifier = Notifier.from_config(config)
    store = HubStore(config.hf_repo, config.hf_bucket, config.hf_token, config.work_dir)
    deadline = time.time() + config.vm_budget_minutes * 60
    try:
        _check_disk(config)
        notifier.major("pipeline", "starting segmentation pipeline")
        store.ensure_private_destination(); store.restore()
        dataset_dir = prepare_dataset(config, store, dry_run=dry_run)
        state = store.read_state()
        for model in config.models:
            if state.is_complete(model.name):
                continue
            if time.time() >= deadline - config.handoff_minutes * 60:
                break
            _check_disk(config)
            notifier.major(model.name, "training started")
            command = _training_command(config, model, dataset_dir, deadline)
            if dry_run:
                print(json.dumps({"model": model.name, "command": command})); continue
            code = _run(command, config.log_dir / f"{model.name}.log")
            store.sync(); state = store.read_state()
            if code != 0:
                raise PipelineError("training_failed", f"{model.name} exited with status {code}")
            if state.is_complete(model.name):
                notifier.major(model.name, "training complete")
        state = store.read_state()
        if state.all_complete(m.name for m in config.models) or dry_run:
            notifier.major("pipeline", "all configured models complete"); return 0
        notifier.major("handoff", "VM time window ending; launching successor")
        store.sync()
        token = KaggleTokenPool(config.kaggle_tokens).select_available()
        launch_successor(config, token)
        store.sync()  # includes kernel.json with the successor log URL
        return 0
    except Exception as exc:
        issue = classify_exception(exc)
        notifier.major("error", f"{issue.code}: {issue.safe_message}")
        try:
            store.append_failure(issue); store.sync()
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.getenv("SEG_CONFIG", "config.yaml")))
    parser.add_argument("--allow-local", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.config, args.allow_local, args.dry_run)
    except (ConfigError, PipelineError) as exc:
        print(f"fatal: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
