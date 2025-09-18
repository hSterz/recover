export WANDB_PROJECT=steer_reft


LR=1e-4
WARM_UP=0
INTERVENTION_TYPE=low_rank_2
EPOCHS=1
MODEL=google/gemma-2-2b-it
SKIP=false
MAX_STEPS=2200
if [ "$MODEL" = "Qwen/Qwen2.5-7B-Instruct" ]; then
    MODEL_ID=qwen
elif [ "$MODEL" = "meta-llama/Llama-3.1-8B-Instruct" ]; then
    MODEL_ID=llama
elif [ "$MODEL" = "google/gemma-2-2b-it" ]; then
    MODEL_ID=gemma2
fi


LAYERWISE=true
CROSSLING=true
BUCKETIZE=true
RANK=64

RUN_NAME="${MODEL_ID}_adaptive_alpha_tulu_rank${RANK}_${EPOCHS}_epochs_${MAX_STEPS}_steps_${INTERVENTION_TYPE}_${LR}_skip_${SKIP}"

LAYER_CMD=""
if [ "$LAYERWISE" = true ]; then
    LAYER_CMD="--layer_wise_AD"
    RUN_NAME="${RUN_NAME}_layerwise"
fi

CROSSLING_CMD=""
if [ "$CROSSLING" = true ]; then
    CROSSLING_CMD="--crosslingual"
    RUN_NAME="${RUN_NAME}_crosslingual"
fi

BUCKETIZE_CMD=""
if [ "$BUCKETIZE" = true ]; then
    RUN_NAME="${RUN_NAME}_avg_vec"
    BUCKETIZE_CMD="--num_buckets 2"
fi

echo "Run name: $RUN_NAME"

python train_recover.py \
        --model_name_or_path $MODEL \
        --run_name $RUN_NAME \
        --output_dir models/${RUN_NAME} \
        --do_train \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --save_steps 1000 \
        --logging_steps 50 \
        --overwrite_output_dir \
        --learning_rate $LR \
        --max_seq_length 1024 \
        --save_total_limit 2 \
        --remove_unused_columns False \
        --num_train_epochs $EPOCHS \
        --report_to wandb \
        --label_names labels \
        --warmup_ratio ${WARM_UP} \
        --lr_scheduler_type linear \
        --intervention_type $INTERVENTION_TYPE \
        --max_steps $MAX_STEPS \
        --rank $RANK \
        $LAYER_CMD \
        $CROSSLING_CMD \
        $BUCKETIZE_CMD \
        --save_total_limit 1

