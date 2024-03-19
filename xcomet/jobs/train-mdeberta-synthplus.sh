#!/bin/bash
#SBATCH --job-name=train-mdeberta
#SBATCH --output=./logs/train-mdeberta-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4,gpu_8,gpu_4_a100,gpu_4_h100
#SBATCH --array=0-2

# Add your commands here



WANDB_PROJECT="xcomet-distillation" \
WANDB_NAME=mdeberta-splus-${SLURM_ARRAY_TASK_ID} \
WANDB_RUN_GROUP="mdeberta" \
python train.py \
    --seed=${SLURM_ARRAY_TASK_ID} \
    --output="distillation_results/synthplus-mdeberta-${SLURM_ARRAY_TASK_ID}" \
    --pretrained-model="microsoft/mdeberta-v3-base" \
    --encoder-model="DeBERTa" \
    --word-layer=8 \
    --word-level=True \
    --hidden-sizes 3072 1024 \
    --loss-lambda=0.055 \
    --layerwise-decay=0.983 \
    --lr=1e-05 \
    --encoder-lr=5e-06 \
    --train-dataset="nllg/mt-metric-synth-plus" \
    --val-dataset="data/mqm-spans-with-year-and-domain-but-no-news-2022.csv.zst" \
    --wandb-project-name="xcomet-distillation" \
    --use-wandb \
    --n-epochs=1 \
    --grad-accum-steps=1 \
    --batch-size=128 \
    --n-gpus=1
