#!/usr/bin/env python3
"""Create a conda environment on Kaggle, then execute the pipeline in it."""
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
    if not MARKER.exists():
        run([conda, "create", "-y", "-p", str(ENV), f"python={sys.version_info.major}.{sys.version_info.minor}", "pip"])
        wheels = sorted(ROOT.glob("segpipe-*.whl"))
        package = wheels[-1] if wheels else TOOL
        if not package.exists():
            raise SystemExit("segpipe wheel/source missing from Kaggle payload")
        run([conda, "run", "-p", str(ENV), "python", "-m", "pip", "install", str(package)])
        MARKER.touch()
    os.execv(conda, [conda, "run", "--no-capture-output", "-p", str(ENV), "python", str(ROOT / "main.py"), "--config", str(ROOT / "config.yaml")])
if __name__ == "__main__": main()
