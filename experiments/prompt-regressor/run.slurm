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

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="

accelerate launch prompt-regressor/run.py \
    --n-bits 4 \
    --epochs 3 \
    --batch-size 32 \
    --scale-lr \
    --lr=1e-4 \
    --use-tf32 \
    --weight-decay=0.1 \
    --log-every=100 \
    --checkpoint-every=2500 \
    --warmup-ratio=0.1 \
    --max-length=768 \
    --optim="paged_adamw_8bit" \
    --group-name="bloomz-1b1-4bit" \
    --save-path="./experiments/prompt-regressor/model" \
    --checkpoint-path="./experiments/prompt-regressor/checkpoint"

