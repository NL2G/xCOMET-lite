#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --job-name="COMET-Training-no-adapters"
#SBATCH --output=./logs/comet/R-%x-%j.out

echo "==="
echo "Starting iteration-1"
echo "==="
WANDB_RUN_GROUP="comet-no-adapters" WANDB_NAME="comet-no-adapters-21291" \
    accelerate launch run.py \
    --prepared-data="./data/comet-prepared" \
    --model-config="comet" \
    --seed=21291 \
    --no-tqdm

echo "==="
echo "Starting iteration-2"
echo "==="
WANDB_RUN_GROUP="comet-no-adapters" WANDB_NAME="comet-no-adapters-2169" \
    accelerate launch run.py \
    --prepared-data="./data/comet-prepared" \
    --model-config="comet" \
    --seed=2169 \
    --no-tqdm

echo "==="
echo "Starting iteration-3"
echo "==="
WANDB_RUN_GROUP="comet-no-adapters" WANDB_NAME="comet-no-adapters-70800" \
    accelerate launch run.py \
    --prepared-data="./data/comet-prepared" \
    --model-config="comet" \
    --seed=70800 \
    --no-tqdm