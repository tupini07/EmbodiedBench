#!/usr/bin/env bash
# Reinstall habitat-sim in CPU-only headless mode for VirtualBox environments without GPU passthrough.
# Usage: ./scripts/reinstall_habitat_headless.sh [conda_env_name]
# Default conda env: embench
# Safe to re-run. Assumes current working directory is repo root.
set -euo pipefail

HAB_ENV="${1:-embench}"

echo "[0/8] Using conda environment: ${HAB_ENV}" >&2
if ! command -v conda &>/dev/null; then
  echo "Conda not found on PATH. Abort." >&2
  exit 1
fi
source "${HOME}/miniconda3/etc/profile.d/conda.sh" || true
if ! conda env list | awk '{print $1}' | grep -Fx "${HAB_ENV}" >/dev/null; then
  echo "Conda env '${HAB_ENV}' not found. Create it or pass a different name." >&2
  exit 1
fi
conda activate "${HAB_ENV}"
echo "Activated env: ${HAB_ENV}" >&2

echo "[1/8] Installing Mesa/EGL software stack (if not present)" >&2
sudo apt update -y
sudo apt install -y mesa-utils libegl1-mesa libegl1-mesa-dev libgles2-mesa-dev libgbm-dev libglu1-mesa

echo "[2/8] Uninstalling any existing habitat-sim wheel" >&2
pip uninstall -y habitat-sim || true

echo "[3/8] Preparing habitat-sim source" >&2
if [ -d habitat-sim ]; then
  echo "Found local habitat-sim source directory; will rebuild headless." >&2
else
  echo "Cloning habitat-sim source (depth=1)" >&2
  git clone --depth 1 https://github.com/facebookresearch/habitat-sim.git
fi

cd habitat-sim

echo "[4/8] Initializing submodules" >&2
git submodule update --init --recursive

echo "[5/8] Installing Python requirements" >&2
pip install -r requirements.txt

# Build flags: disable CUDA, build headless, include Bullet physics.
export HABITAT_SIM_BUILD_WITH_CUDA=OFF
export HABITAT_SIM_BUILD_WITH_MAGNUM=ON
export HABITAT_SIM_USE_SYSTEM_MAGNUM=OFF

echo "[6/8] Building and installing habitat-sim (CPU headless)" >&2
python setup.py install --headless --with-bullet

cd ..

echo "[7/8] Post-install validation" >&2
python - <<'EOF'
import habitat_sim
print('habitat-sim imported OK (CPU headless).')
EOF

echo "[8/8] Suggested environment variables for headless software rendering" >&2
cat <<'EOV'
# Export these before running EB-Habitat when in VirtualBox:
export CUDA_VISIBLE_DEVICES=""  # ensure no CUDA usage attempt
export MAGNUM_DEFAULT_GL_CONTEXT_VERSION=330
export GALLIUM_DRIVER=llvmpipe
export EGL_PLATFORM=surfaceless
unset DISPLAY  # guarantee offscreen; if you need GUI later set DISPLAY=:0
EOV

echo "Done. Re-run: HEADLESS=1 bash scripts/run_evals_basic.sh <exp_name>" >&2
