#!/usr/bin/env bash

set -ex

exp_name=$1

export REMOTE_URL=${REMOTE_URL:-"http://localhost:43289/v1"}
export remote_url="${REMOTE_URL}"
export OPENAI_API_KEY='empty'

export EXTRA_ARGS=${EXTRA_ARGS:-""}
export EXTRA_ARGS_EB_ALFRED=${EXTRA_ARGS_EB_ALFRED:-""}
export EXTRA_ARGS_EB_HAB=${EXTRA_ARGS_EB_HAB:-""}
export EXTRA_ARGS_EB_MAN=${EXTRA_ARGS_EB_MAN:-""}
export EXTRA_ARGS_EB_NAV=${EXTRA_ARGS_EB_NAV:-""}

export MODEL_NAME=${MODEL_NAME:-"Qwen2.5-VL-7B-Instruct"}

if [ -z "$exp_name" ]; then
    echo "Usage: $0 <exp_name>"
    exit 1
fi

# ensure that remoteURL is reachable
if ! curl -si "$REMOTE_URL/models" | head -n 1 | grep "200 OK" > /dev/null; then
    echo "Error: REMOTE_URL $REMOTE_URL is not reachable."
    echo ""
    echo "Please ensure SSH tunnel is running in another terminal:"
    echo ""
    echo "    amlt ssh \"job_name\" -o \"StrictHostKeyChecking=no\" -o \"-4 -L 43289:localhost:43289\""
    echo ""
    echo ""
    exit 1
fi

# -------------------------------------------------------------------
# print evaluation configuration
echo "================ Evaluation Configuration ================"
echo "Experiment Name: $exp_name"
echo "Model Name: $MODEL_NAME"
echo "REMOTE_URL: $REMOTE_URL"
echo "EXTRA_ARGS: $EXTRA_ARGS"
echo "EXTRA_ARGS_EB_ALFRED: $EXTRA_ARGS_EB_ALFRED"
echo "EXTRA_ARGS_EB_HAB: $EXTRA_ARGS_EB_HAB"
echo "EXTRA_ARGS_EB_MAN: $EXTRA_ARGS_EB_MAN"
echo "EXTRA_ARGS_EB_NAV: $EXTRA_ARGS_EB_NAV"
echo "=========================================================="


# -------------------------------------------------------------------

echo "Running evaluation with exp_name: $exp_name"

# remove results dir if exp_name was already present
find "./running"/ -type d -name "${MODEL_NAME}_${exp_name}" -exec rm -rf {} +

# Headless toggle: if HEADLESS=1, use software rendering (llvmpipe) and unset DISPLAY.
if [ "${HEADLESS:-}" = "1" ]; then
    echo "[HEADLESS] Enabling software rendering (EGL surfaceless).";
    export CUDA_VISIBLE_DEVICES=""  # force CPU-only
    export MAGNUM_DEFAULT_GL_CONTEXT_VERSION=330
    export GALLIUM_DRIVER=llvmpipe
    export EGL_PLATFORM=surfaceless
    unset DISPLAY
else
    # Set DISPLAY for WSL (WSLg handles display automatically) or local X11
    export DISPLAY=":1"
fi

dones_file="running/${exp_name}_dones.txt"
echo "" > "$dones_file"

source ~/miniconda3/etc/profile.d/conda.sh

# -------------------------------------------------------------------

conda activate embench

echo "Running EB-ALFRED evaluation..."
python -m embodiedbench.main env=eb-alf model_name="$MODEL_NAME" exp_name="$exp_name" ${EXTRA_ARGS} ${EXTRA_ARGS_EB_ALFRED}

echo "EB-ALFRED" > "$dones_file"

# -------------------------------------------------------------------

conda activate embench

echo "Running EB-Habitat evaluation..."
python -m embodiedbench.main env=eb-hab model_name="$MODEL_NAME" exp_name="$exp_name" ${EXTRA_ARGS} ${EXTRA_ARGS_EB_HAB}

echo "EB-Habitat" >> "$dones_file"

# -------------------------------------------------------------------

conda activate embench_man

# ---------------- EB-Manipulation setup & health checks ----------------
# Resolve CoppeliaSim install location. Prefer project root, fallback to env subdir.
ROOT_COPPELIA="$(pwd)/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
ALT_COPPELIA="$(pwd)/embodiedbench/envs/eb_manipulation/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
if [ -d "$ROOT_COPPELIA" ]; then
    export COPPELIASIM_ROOT="$ROOT_COPPELIA"
elif [ -d "$ALT_COPPELIA" ]; then
    export COPPELIASIM_ROOT="$ALT_COPPELIA"
else
    echo "[EB-Man] CoppeliaSim directory not found in either expected path." >&2
    echo "[EB-Man] Expected one of: $ROOT_COPPELIA or $ALT_COPPELIA" >&2
    echo "[EB-Man] Please re-run installation steps for EB-Manipulation before continuing." >&2
    exit 1
fi

# Ensure system/usrset.txt exists (PyRep setup.py touches/opens this file)
if [ ! -f "$COPPELIASIM_ROOT/system/usrset.txt" ]; then
    echo "[EB-Man] Creating missing usrset.txt at $COPPELIASIM_ROOT/system/usrset.txt"
    mkdir -p "$COPPELIASIM_ROOT/system" && touch "$COPPELIASIM_ROOT/system/usrset.txt"
fi

# Library paths (prepend safely, avoid leading colon if empty)
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}" 
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

# Attempt PyRep import; if missing, try local editable install automatically.
if ! python -c "import pyrep" 2>/dev/null; then
    echo "[EB-Man] PyRep not found. Attempting local install..."
    if [ -d "embodiedbench/envs/eb_manipulation/PyRep" ]; then
        (cd embodiedbench/envs/eb_manipulation/PyRep && pip install -e .) || {
            echo "[EB-Man] PyRep installation failed." >&2; exit 1; }
    else
        echo "[EB-Man] PyRep source directory missing at embodiedbench/envs/eb_manipulation/PyRep" >&2
        echo "[EB-Man] Please clone https://github.com/stepjam/PyRep.git into that path." >&2
        exit 1
    fi
fi

# Verify dynamic libs accessible for PyRep
if ! python -c "import pyrep" 2>/dev/null; then
    echo "[EB-Man] PyRep import still failing after install. Check LD_LIBRARY_PATH and COPPELIASIM_ROOT." >&2
    exit 1
fi

echo "Running EB-Manipulation evaluation..."
python -m embodiedbench.main env=eb-man model_name="$MODEL_NAME" exp_name="$exp_name" ${EXTRA_ARGS} ${EXTRA_ARGS_EB_MAN}

echo "EB-Manipulation" >> "$dones_file"

# -------------------------------------------------------------------

conda activate embench_nav

echo "Running EB-Navigation evaluation..."
python -m embodiedbench.main env=eb-nav model_name="$MODEL_NAME" exp_name="$exp_name" ${EXTRA_ARGS} ${EXTRA_ARGS_EB_NAV}

echo "EB-Navigation" >> "$dones_file"

# -------------------------------------------------------------------

echo "All evaluations completed."

echo "ALL DONE" >> "$dones_file"