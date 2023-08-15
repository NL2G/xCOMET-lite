#!/usr/bin/env bash

#SBATCH --job-name=TrainBloom
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --output=./experiments/prompt-regressor/logs/train_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --exclude=worker-8


echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="

accelerate launch prompt-regressor/run.py \
    --n-bits 16 \
    --use-lora \
    --epochs 3 \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --scale-lr \
    --subsample-train=100000 \
    --lr=1e-4 \
    --activation="sigmoid" \
    --use-tf32 \
    --weight-decay=0.01 \
    --log-every=100 \
    --checkpoint-every=500 \
    --eval-every=500 \
    --warmup-ratio=0.1 \
    --max-length=512 \
    --optim="paged_adam_8bit" \
    --group-name="bloomz-1b1-4bit" \
    --save-path="./experiments/prompt-regressor/model-1b1" \
    --checkpoint-path="./experiments/prompt-regressor/checkpoint"

