#!/usr/bin/env bash

#SBATCH --job-name=TrainBloom560mLion
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --output=./experiments/prompt-regressor/logs/train_lion8bit_%j.out
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
    --use-lora \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --scale-lr \
    --lr=5e-4 \
    --use-tf32 \
    --weight-decay=0.01 \
    --activation="gelu_fast" \
    --log-every=100 \
    --checkpoint-every=2000 \
    --eval-every=2000 \
    --warmup-ratio=0.1 \
    --max-length=512 \
    --model="bigscience/bloomz-560m" \
    --optim="paged_lion_8bit" \
    --group-name="bloomz-560m" \
    --save-path="./experiments/prompt-regressor/model" \
    --checkpoint-path="./experiments/prompt-regressor/checkpoint"

