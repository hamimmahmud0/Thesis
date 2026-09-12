#!/usr/bin/env bash
set -euo pipefail

# ──── Configuration ───────────────────────────────────────────────────────────
ENV_NAME="stabilize"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.4"        # for conda pytorch-cuda
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALL_DIR="$HOME/miniconda3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ──── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ──── 1. Locate or install conda ─────────────────────────────────────────────
find_conda() {
    # Search common locations in PATH and standard install dirs.
    local candidates=(
        conda
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "/opt/conda/bin/conda"
        "/opt/miniconda3/bin/conda"
    )
    for c in "${candidates[@]}"; do
        if command -v "$c" &>/dev/null || [ -x "$c" ]; then
            echo "$c"
            return
        fi
    done
    return 1
}

install_miniconda() {
    info "conda not found — installing Miniconda headlessly ..."

    local installer="/tmp/miniconda_installer.sh"
    if [ ! -f "$installer" ]; then
        info "Downloading Miniconda ..."
        curl -fsSL "$MINICONDA_URL" -o "$installer"
    fi

    bash "$installer" -b -p "$MINICONDA_INSTALL_DIR"
    rm -f "$installer"

    # Source the shell hook so conda is on PATH for this script invocation.
    # shellcheck disable=SC1091

    # Init conda
    "$MINICONDA_INSTALL_DIR/bin/conda" init
    source ~/.bashrc

    # Accept
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    # source "$MINICONDA_INSTALL_DIR/etc/profile.d/conda.sh"

    info "Miniconda installed -> $MINICONDA_INSTALL_DIR"
    warn "Add this to your shell profile for future sessions:"
    warn "  source $MINICONDA_INSTALL_DIR/etc/profile.d/conda.sh"
}

# Main entry — find or install, then activate.
CONDA_BIN=""
if CONDA_BIN="$(find_conda)"; then
    info "conda found: $CONDA_BIN"
    eval "$("$CONDA_BIN" shell.bash hook 2>/dev/null || true)"
else
    install_miniconda
    CONDA_BIN="$MINICONDA_INSTALL_DIR/bin/conda"
fi

# ──── 2. Create / update conda environment ────────────────────────────────────
if conda env list | grep -qw "$ENV_NAME"; then
    info "conda env '$ENV_NAME' already exists — keeping as-is"
else
    info "Creating conda env '$ENV_NAME' (python $PYTHON_VERSION) ..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
info "Activated: $(python --version) @ $(which python)"

# ──── 3. Install PyTorch + CUDA via conda (idempotent) ───────────────────────
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || true)
if [ -z "$TORCH_VER" ]; then
    info "Installing PyTorch + CUDA $CUDA_VERSION via conda ..."
    conda install -y -n "$ENV_NAME" \
        pytorch \
        torchvision \
        "pytorch-cuda=$CUDA_VERSION" \
        -c pytorch -c nvidia
else
    info "PyTorch already installed: $TORCH_VER"
fi

# ──── 4. Install pip requirements (non-torch) ────────────────────────────────
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    error "requirements.txt not found at $REQUIREMENTS"
    exit 1
fi

info "Installing pip requirements ..."
pip install --upgrade pip
# --no-deps for torch/torchvision is already satisfied via conda
pip install -r "$REQUIREMENTS"

# ──── 5. Install the package in editable mode ────────────────────────────────
info "Installing stabilize (editable) ..."
pip install -e "$SCRIPT_DIR"

# ──── 6. Verify ───────────────────────────────────────────────────────────────
info "Running smoke test ..."
if stabilize --help >/dev/null 2>&1; then
    info "stabilize CLI installed successfully"
    stabilize --help
else
    error "stabilize --help failed — check output above"
    exit 1
fi

echo ""
info "===== SETUP COMPLETE ====="
info "Activate with:  conda activate $ENV_NAME"
