#!/usr/bin/env bash

# Parse command line arguments
PORT=${1:-43289}

# Qwen2.5-VL-7B-Instruct
# MODEL_CHECKPOINT="Qwen/Qwen2.5-VL-7B-Instruct"

# Qwen2.5-VL-7B-Instruct_BASE--rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128
MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/videor1-images-pxmc-verifier-forced-reasoning-glm4.5/rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_15/actor/huggingface/"

# Qwen2.5-VL-7B-Instruct_BASE--cleaned_videos_unprocessed_split1_min_1.0_num_4.9K_videos
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/sft-model-weights/videor1/sing/cleaned_videos_unprocessed_split1_min_1.0_num_4.9K_videos_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model/"

# combined-1-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_mode
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/sft-model-weights/combined/sing/combined-1-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model/"

# Qwen2.5-VL-7B-Instruct_BASE--aokvqa+videor1
# MODEL_CHECKPOINT="/mnt/magmathor/training/checkpoints/Qwen2.5-VL-7B-Instruct_BASE__sft__llamafactory_ai2thor_aokvqa+videor1_tag_bs1_accum2_lr1e-5_cosine_0.1_ft_llm_qwen_base_pix_3136_12845056_model_singlenode/checkpoint-436/"


# Qwen2.5-VL-7B-Instruct_BASE--magmathor0.1p+aokvqa+videor1
# MODEL_CHECKPOINT="/mnt/magmathor/training/checkpoints/Qwen2.5-VL-7B-Instruct_BASE__sft__llamafactory_ai2thor_magmathor0.1p+aokvqa+videor1_tag_bs1_accum2_lr1e-5_cosine_0.1_ft_llm_qwen_base_pix_3136_12845056_model_singlenode/checkpoint-700/"


echo "Using model: $MODEL_NAME"
echo "Using checkpoint: $MODEL_CHECKPOINT"
echo "Using port: $PORT"


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
    --port "$PORT"  \
    --served-model-name "vllm-model" 


