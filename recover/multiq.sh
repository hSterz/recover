#!/bin/sh

MODEL_NAME=google/gemma-2-2b-it
# Qwen/Qwen2-7B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
STEER_PATH=<ADD_YOUR_STEERING_PATH_HERE>

python predict_multiQ.py \
    --crosslingual \
    --alpha 2 \
    --beta 1 \
    --scaling norm \
    --restore_norm \
    --version recover_plus \
    --model_name $MODEL_NAME \
    --path $STEER_PATH 