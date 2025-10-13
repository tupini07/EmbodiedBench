#!/usr/bin/env bash

# MODEL_NAME="Qwen2.5-VL-7B-Instruct_SFT_C"
# MODEL_CHECKPOINT="/mnt/magmathor/training/checkpoints/Qwen2.5-VL-7B-Instruct__sft__llamafactory_ai2thor_correct-only_tag_bs1_accum1_lr1e-5_cosine_0.1_ft_llm_qwen_base_pix_25088_50176_model_multinode/checkpoint-2100/"

MODEL_NAME="Qwen2.5-VL-7B-Instruct"
MODEL_CHECKPOINT="Qwen/Qwen2.5-VL-7B-Instruct"

# if modelname is qwen then don't load from disk but from huggingface
if [ "$MODEL_NAME" = "Qwen2.5-VL-7B-Instruct" ]; then
    echo "Loading model from Huggingface"
    MODEL_CHECKPOINT="$MODEL_PATH"
else
    echo "Loading model from disk"
    MODEL_CHECKPOINT="/mnt/$MODEL_PATH"
fi


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

vllm serve "$MODEL_CHECKPOINT"  \
    --tensor-parallel-size 2  \
    --uvicorn-log-level critical  \
    --allowed-local-media-path /  \
    --mm-encoder-tp-mode data \
    --host 0.0.0.0 \
    --port 43289  \
    --served-model-name "vllm-model" 
