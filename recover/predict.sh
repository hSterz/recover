#!/bin/sh

python predict_lcb.py \
    --crosslingual \
    --alpha 1 \
    --beta 1 \
    --scaling norm \
    --restore_norm \
    --version recover \
    --model_name Qwen/Qwen2-7B-Instruct 
