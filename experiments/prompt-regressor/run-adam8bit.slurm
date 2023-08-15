#!/usr/bin/env bash

#SBATCH --job-name=TrainBloom560mAdam
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --output=./experiments/prompt-regressor/logs/train_adam8bit_%j.out
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
    --epochs 3 \
    --use-lora \
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
    --model="bigscience/bloomz-560m" \
    --optim="paged_adamw_8bit" \
    --group-name="bloomz-560m" \
    --save-path="./experiments/prompt-regressor/model" \
    --checkpoint-path="./experiments/prompt-regressor/checkpoint"

