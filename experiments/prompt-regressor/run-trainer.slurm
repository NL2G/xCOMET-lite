#!/usr/bin/env bash

#SBATCH --job-name=TrainBloom560m
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --output=./experiments/prompt-regressor/logs/trainer_log_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --exclude=worker-8

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="

export SEED=42

export WANDB_NAME="trainer@run"
export WANDB_PROJECT="wmt23-trainer"
export WANDB_RUN_GROUP="run-1b-lora-test"
export WANDB_ENTITY="airi23-efficient-llm-metrics"
export OUTPUT_DIR="./trainer-model"

accelerate launch train.py \
    --max_train_samples=25600 \
    --max_eval_samples=6400 \
	--model_name_or_path="bigscience/bloomz-1b1" \
    --output_dir=$OUTPUT_DIR \
	--dataset_name="nllg/wmt-metrics-data" \
	--dataset_config_name="template" \
	--token="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" \
	--do_regression \
	--text_column_names="text" \
	--validation_split_name="test" \
	--remove_columns="lp,src,mt,ref,score_type" \
	--label_column_name="score" \
	--max_seq_length=512 \
	--no_pad_to_max_length \
	--shuffle_train_dataset \
	--shuffle_seed=$SEED --seed=$SEED --data_seed=$SEED \
	--do_train \
	--do_eval \
	--evaluation_strategy="steps" \
	--eval_steps=50 \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=16 \
	--gradient_accumulation_steps=8 \
	--eval_accumulation_steps=4 \
	--learning_rate=5e-5 \
	--weight_decay=0.01 \
	--lr_scheduler_type="cosine" \
	--warmup_ratio=0.06 \
	--log_level="info" \
	--logging_steps=50 \
	--save_strategy="steps" \
	--save_total_limit=3 \
	--save_steps=100 \
	--save_safetensors \
    --bf16 \
	--tf32=true \
    --include_inputs_for_metrics \
	--dataloader_num_workers=1 \
	--load_best_model_at_end \
	--metric_for_best_model="kendalltau|total" \
	--greater_is_better=true \
	--optim="paged_adamw_8bit" \
    --length_column_name="length" \
    --report_to="wandb" \
    --push_to_hub \
    --hub_model_id="nllg/bloomscore-1b1" \
    --hub_strategy="end" \
    --hub_private_repo \
    --hub_token="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"

