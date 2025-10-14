#!/usr/bin/env bash

exp_name=$1

export REMOTE_URL=${REMOTE_URL:-"http://localhost:43289/v1"}
export OPENAI_API_KEY='empty'
export N_SHOTS=${N_SHOTS:-3}

if [ -z "$exp_name" ]; then
    echo "Usage: $0 <exp_name>"
    exit 1
fi

# ensure that remoteURL is reachable
if ! curl -si "$REMOTE_URL/models" | head -n 1 | grep "200 OK" > /dev/null; then
    echo "Error: REMOTE_URL $REMOTE_URL is not reachable."
    echo "Maybe need to start the SSH tunnel with 'scripts/start_ssh_tunnel.sh'?"
    exit 1
fi

# -------------------------------------------------------------------

echo "Running evaluation with exp_name: $exp_name"

export DISPLAY=":1"

python -m embodiedbench.envs.eb_alfred.scripts.startx 1 > /dev/null 2>&1 &
sleep 2  

dones_file="running/${exp_name}_dones.txt"

source ~/miniconda3/etc/profile.d/conda.sh

# -------------------------------------------------------------------

conda activate embench

echo "Running EB-ALFRED evaluation..."
python -m embodiedbench.main env=eb-alf model_name='vllm-model' exp_name="$exp_name" n_shots=$N_SHOTS

echo "EB-ALFRED" >> "$dones_file"

# -------------------------------------------------------------------

conda activate embench

echo "Running EB-Habitat evaluation..."
python -m embodiedbench.main env=eb-hab model_name="vllm-model" exp_name="$exp_name" n_shots=$N_SHOTS

echo "EB-Habitat" >> "$dones_file"

# -------------------------------------------------------------------

conda activate embench_man

# Set CoppeliaSim environment variables for PyRep
export COPPELIASIM_ROOT="$(pwd)/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

echo "Running EB-Manipulation evaluation..."
python -m embodiedbench.main env=eb-man model_name="vllm-model" exp_name="$exp_name" n_shots=$N_SHOTS

echo "EB-Manipulation" >> "$dones_file"

# -------------------------------------------------------------------

conda activate embench_nav

echo "Running EB-Navigation evaluation..."
python -m embodiedbench.main env=eb-nav model_name="vllm-model" exp_name="$exp_name" n_shots=$N_SHOTS

echo "EB-Navigation" >> "$dones_file"

# -------------------------------------------------------------------

echo "All evaluations completed."

echo "ALL DONE" >> "$dones_file"