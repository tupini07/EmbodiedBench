#!/usr/bin/env bash

# MODEL_NAME="Qwen2.5-VL-7B-Instruct"
# MODEL_CHECKPOINT="Qwen/Qwen2.5-VL-7B-Instruct"


MODEL_NAME="Qwen2.5-VL-7B-Instruct_BASE--aokvqa+videor1"
MODEL_CHECKPOINT="/mnt/magmathor/training/checkpoints/Qwen2.5-VL-7B-Instruct_BASE__sft__llamafactory_ai2thor_aokvqa+videor1_tag_bs1_accum2_lr1e-5_cosine_0.1_ft_llm_qwen_base_pix_3136_12845056_model_singlenode/checkpoint-436/"


# MODEL_NAME="Qwen2.5-VL-7B-Instruct_BASE--magmathor0.1p+aokvqa+videor1"
# MODEL_CHECKPOINT="/mnt/magmathor/training/checkpoints/Qwen2.5-VL-7B-Instruct_BASE__sft__llamafactory_ai2thor_magmathor0.1p+aokvqa+videor1_tag_bs1_accum2_lr1e-5_cosine_0.1_ft_llm_qwen_base_pix_3136_12845056_model_singlenode/checkpoint-700/"


echo "Using model: $MODEL_NAME"
echo "Using checkpoint: $MODEL_CHECKPOINT"


CUR_DIR=$PWD

cd /tmp/

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

mkdir -p /tmp/vllm-server
cd /tmp/vllm-server
uv venv --python=3.12 --clear
source .venv/bin/activate

uv pip install "vllm[video]==0.10.2" --torch-backend=auto
uv pip install flashinfer-python==0.3.1 bitsandbytes==0.47.0

export CUDA_VISIBLE_DEVICES=0,1 

# Start GPU keep-alive monitor in background (meant to be run in AML)
echo "Starting GPU keep-alive monitor..."
python3 "$CUR_DIR/gpu_keepalive.py" > /tmp/gpu_keepalive.log 2>&1 &
GPU_MONITOR_PID=$!
echo "GPU monitor started with PID: $GPU_MONITOR_PID"

# Cleanup function to stop monitor on exit
cleanup() {
    echo "Stopping GPU keep-alive monitor (PID: $GPU_MONITOR_PID)..."
    kill $GPU_MONITOR_PID 2>/dev/null || true
    wait $GPU_MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

vllm serve "$MODEL_CHECKPOINT"  \
    --tensor-parallel-size 2  \
    --uvicorn-log-level critical  \
    --allowed-local-media-path /  \
    --mm-encoder-tp-mode data \
    --host 0.0.0.0 \
    --port 43289  \
    --served-model-name "vllm-model" 


