"""Settings + account discovery. Tokens live one-per-file in .tokens/<name>."""
import copy, os, tempfile, threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent.parent
CONFIG_PATH = ROOT / "config.yaml"
TOKENS_DIR = ROOT / ".tokens"
STATE_PATH = ROOT / "state.json"

DEFAULTS = {
    "selection": {
        "min_gpu_hours": 1.0,        # skip accounts with less GPU quota left
        "quota_cache_seconds": 60,
        "bad_account_seconds": 900,  # cool-down after a failed push/boot
    },
    "safety": {
        "max_vms": 3,
        "idle_stop_minutes": 30,     # 0 disables
        "stop_all_on_exit": False,
        "boot_timeout_seconds": 420,
    },
    "vm": {
        "gpu_template": str(REPO / "kernels" / "ssh"),
        "cpu_template": str(REPO / "kernels" / "ssh-cpu"),
        "gpu_slug": "ssh-gpu",
        "cpu_slug": "ssh-cpu",
        "ssh_user": "notebook",
        "ssh_local_host": "127.0.0.1",
        "ssh_local_port": 2222,
        "broker": "https://165.99.219.95",
        "broker_host": "165.99.219.95",
        "broker_control_port": 7000,
        "scope": "open",
        "protocol": "tcp",
        "hf_token": "",
    },
    "server": {"host": "127.0.0.1", "port": 8765},
    "accounts": {},   # name -> {enabled, priority, username, gpu, cpu}
}

_lock = threading.RLock()


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "accounts":
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load():
    with _lock:
        cfg = copy.deepcopy(DEFAULTS)
        if CONFIG_PATH.exists():
            _merge(cfg, yaml.safe_load(CONFIG_PATH.read_text()) or {})
        return cfg


def save(cfg):
    with _lock:
        fd, tmp = tempfile.mkstemp(dir=ROOT, prefix=".config.")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, CONFIG_PATH)


def update(patch):
    with _lock:
        cfg = load()
        _merge(cfg, patch)
        save(cfg)
        return cfg


def token_names():
    if not TOKENS_DIR.is_dir():
        return []
    return sorted(p.name for p in TOKENS_DIR.iterdir() if p.is_file() and not p.name.startswith("."))


def read_token(name):
    return (TOKENS_DIR / name).read_text().strip()


def write_token(name, token):
    TOKENS_DIR.mkdir(exist_ok=True, mode=0o700)
    p = TOKENS_DIR / name
    p.write_text(token.strip() + "\n")
    p.chmod(0o600)


def delete_token(name):
    (TOKENS_DIR / name).unlink(missing_ok=True)


def account_settings(cfg, name):
    base = {"enabled": True, "priority": 0, "gpu": True, "cpu": True, "username": None}
    base.update((cfg.get("accounts") or {}).get(name) or {})
    return base
