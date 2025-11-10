#!/usr/bin/env bash

# Parse command line arguments
PORT=${1:-43289}

# Qwen2.5-VL-7B-Instruct
# Qwen2.5-7b_NewJsonParsing_ActionCorrection_OneStepPlan_3Retries_WithReasoning_
# MODEL_CHECKPOINT="Qwen/Qwen2.5-VL-7B-Instruct"

# 20251109-total-gated-combined-5-rl-step-55
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/fixed-combined-5-reward-videor1-images-pxmc-verifier-forced-reasoning-glm4.5/total-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_55/actor/huggingface/"

# 20251109-combined-9-sft-baseline
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/sft-model-weights/combined/sing/single-combined-9-sft_pix_3136_262144_bs2_accum16_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model/"

# 20251109-combined-8-sft-baseline
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/sft-model-weights/combined/sing/single-combined-8-sft_pix_3136_262144_bs2_accum16_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model/"

# 20251109-combined-7-sft-baseline
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/sft-model-weights/combined/sing/single-combined-7-sft_pix_3136_262144_bs2_accum16_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model/"

# 20251108-combined-5-total-gated-rm2-step20--actor--huggingface
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/fixed-combined-5-reward-videor1-images-pxmc-verifier-forced-reasoning-glm4.5/total-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_20/actor/huggingface/"

# 20251108-combined-5-gated-rm2-step55--actor--huggingface
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/fixed-combined-5-reward-videor1-images-pxmc-verifier-forced-reasoning-glm4.5/gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_55/actor/huggingface/"

# 20251108-combined-5-gated-rm2-step65--actor--huggingface
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/fixed-combined-5-reward-videor1-images-pxmc-verifier-forced-reasoning-glm4.5/gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_65/actor/huggingface/"

# Script: baseline-videor1-sft-qwen2.5-vl-7b-instruct
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/public_pretrained_weights/videor1-Qwen2.5-VL-7B-Instruct/"

# Script: baseline-videor1-rl-qwen2.5-vl-7b-instruct
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/pretrained_weights/Video-R1-7B/"

# Script: 20251107-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2--global_step_40--ActCorr_1SPLoose_3Ret_WReason_
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/fixed-combined-2-reward-pxmc-verifier-forced-reasoning-glm4.5/gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_40/actor/huggingface/"

# Script: 20251107-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2--global_step_45--ActCorr_1SPLoose_3Ret_WReason_
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/fixed-combined-2-reward-pxmc-verifier-forced-reasoning-glm4.5/gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_45/actor/huggingface/"

# Script: 20251106-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2--global_step_30--actor--huggingface
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/videor1-images-pxmc-verifier-forced-reasoning-glm4.5/gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_30/actor/huggingface/"

# Script: 20251106-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2--global_step_35--actor--huggingface
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/videor1-images-pxmc-verifier-forced-reasoning-glm4.5/gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_35/actor/huggingface/"

# Script: 20251106-combined-5-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/sft-model-weights/combined/sing/combined-5-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model/"

# Script: 20251106-gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/videor1-images-pxmc-verifier-forced-reasoning-glm4.5/gated-rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_20/actor/huggingface/"

# Script: 20251106-combined-2-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/sft-model-weights/combined/sing/combined-2-sft_pix_3136_262144_bs1_accum8_gpus8_lr1e-5_cosine0.1_epoch3_freeze_vision-true-proj-true-lm-false_model/"

# Script: 20251106-evaluate_rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128
# MODEL_CHECKPOINT="/mnt/reuben_models/projects/reubenprojects/magma-reasoning/rl-model-weights/videor1-images-pxmc-verifier-forced-reasoning-glm4.5/rs-rm-2-vis1.0-sa1.0-agentic-functions-verifier-videor1_min_1.5-globalbs-128-updatebs-1-expbs-2-nnodes--2/global_step_15/actor/huggingface/"

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


