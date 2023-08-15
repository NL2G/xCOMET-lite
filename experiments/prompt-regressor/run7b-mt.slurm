#!/usr/bin/env bash

#SBATCH --job-name=TrainBloom7B
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --output=./experiments/prompt-regressor/logs/train7b_%j.out
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
    --n-bits 4 \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --scale-lr \
    --lr=5e-4 \
    --activation="gelu_fast" \
    --use-lora \
    --use-tf32 \
    --weight-decay=0.01 \
    --log-every=400 \
    --checkpoint-every=8000 \
    --eval-every=8000 \
    --warmup-ratio=0.1 \
    --max-length=512 \
    --model="bigscience/bloomz-7b1-mt" \
    --optim="paged_adam_8bit" \
    --group-name="bloomz-7b1-mt-4bit" \
    --save-path="./experiments/prompt-regressor/model" \
    --checkpoint-path="./experiments/prompt-regressor/checkpoint"

