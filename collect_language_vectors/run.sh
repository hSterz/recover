#!/bin/sh
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
# #Qwen/Qwen2-7B-Instruct or google/gemma-2-2b-it or meta-llama/Llama-3.1-8B-Instruct are used in the paper

for LANG in "eus" "swh" "arb" "cmn" "eng" "deu" "spa" "fra" "ita" "por" "rus" "jpn" "kor" "ind" "vie" "tha" "ara" "tur" "swh" "pes" "eus" "hin"
do 
    python collect.py \
        --language $LANG \
        --model_name_or_path $MODEL_NAME \
        --dataset_name openlanguagedata/flores_plus \
        --split dev 
done