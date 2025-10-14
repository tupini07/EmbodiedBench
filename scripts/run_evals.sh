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

source ~/miniconda3/etc/profile.d/conda.sh

conda activate embench

echo "Running EB-ALFRED evaluation..."
python -m embodiedbench.main env=eb-alf model_name='vllm-model' exp_name="$exp_name" n_shots=$N_SHOTS

echo "Running EB-Habitat evaluation..."
python -m embodiedbench.main env=eb-hab model_name="vllm-model" exp_name="$exp_name" n_shots=$N_SHOTS

# -------------------------------------------------------------------

conda activate embench_man

echo "Running EB-Manipulation evaluation..."
python -m embodiedbench.main env=eb-man model_name="vllm-model" exp_name="$exp_name" n_shots=$N_SHOTS

# -------------------------------------------------------------------

conda activate embench_nav

echo "Running EB-Navigation evaluation..."
python -m embodiedbench.main env=eb-nav model_name="vllm-model" exp_name="$exp_name" n_shots=$N_SHOTS

# -------------------------------------------------------------------

echo "All evaluations completed."