#!/usr/bin/env python3
"""Create a conda environment on Kaggle, then execute the pipeline in it."""
import base64
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOL = ROOT / "tool"
ENV = Path("/kaggle/working/conda-seg")
MARKER = ENV / ".seg-ready"
CONDA_ROOT = Path("/kaggle/working/miniforge3")
PAYLOAD = Path("/kaggle/working/seg-payload")
try:
    EMBEDDED_WHEEL_B64
    EMBEDDED_WHEEL_NAME
    EMBEDDED_CONFIG_B64
except NameError:
    EMBEDDED_WHEEL_B64 = ""
    EMBEDDED_WHEEL_NAME = "segpipe-1.0.0-py3-none-any.whl"
    EMBEDDED_CONFIG_B64 = ""

def run(args): subprocess.run(args, check=True)
def ensure_conda() -> str:
    existing = shutil.which("conda")
    if existing:
        return existing
    conda = CONDA_ROOT / "bin" / "conda"
    if conda.exists():
        return str(conda)
    machine = platform.machine().lower()
    architecture = "aarch64" if machine in {"aarch64", "arm64"} else "x86_64"
    installer = Path("/kaggle/working/miniforge-installer.sh")
    url = f"https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-{architecture}.sh"
    urllib.request.urlretrieve(url, installer)
    run(["bash", str(installer), "-b", "-p", str(CONDA_ROOT)])
    return str(conda)

def main():
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise SystemExit("This bootstrap may only run in Kaggle")
    conda = ensure_conda()
    config_path = ROOT / "config.yaml"
    embedded_wheel = None
    if EMBEDDED_WHEEL_B64:
        PAYLOAD.mkdir(parents=True, exist_ok=True)
        embedded_wheel = PAYLOAD / Path(EMBEDDED_WHEEL_NAME).name
        embedded_wheel.write_bytes(base64.b64decode(EMBEDDED_WHEEL_B64))
        config_path = PAYLOAD / "config.yaml"
        config_path.write_bytes(base64.b64decode(EMBEDDED_CONFIG_B64))
        shutil.copy2(Path(__file__), PAYLOAD / "kernel.py")
    if not MARKER.exists():
        run([conda, "create", "-y", "-p", str(ENV), f"python={sys.version_info.major}.{sys.version_info.minor}", "pip"])
        wheels = [embedded_wheel] if embedded_wheel else sorted(ROOT.glob("segpipe-*.whl"))
        package = wheels[-1] if wheels else TOOL
        if not package.exists():
            raise SystemExit("segpipe wheel/source missing from Kaggle payload")
        run([conda, "run", "-p", str(ENV), "python", "-m", "pip", "install", str(package)])
        MARKER.touch()
    os.execv(conda, [conda, "run", "--no-capture-output", "-p", str(ENV), "python", "-m", "segpipe.main", "--config", str(config_path)])
if __name__ == "__main__": main()
