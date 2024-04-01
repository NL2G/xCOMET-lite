#!/bin/bash
#SBATCH --job-name=eval-mdeberta
#SBATCH --output=./logs/eval-mdeberta-%A-[%a].txt
#SBATCH --error=./error-logs/eval-mdeberta-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=single
#SBATCH --array=0-8

# Add your commands here

seeds=(
    "0"
    "1"
    "2"
    "0"
    "1"
    "2"
    "0"
    "1"
    "2"
)

lps=(
    "en-de"
    "en-de"
    "en-de"
    "zh-en"
    "zh-en"
    "zh-en"
    "en-ru"
    "en-ru"
    "en-ru"
)

srun python eval_checkpoint.py \
    --dataset="RicardoRei/wmt-mqm-human-evaluation" \
    --domain="news" \
    --lp=${lps[$SLURM_ARRAY_TASK_ID]} \
    --seed=${seeds[$SLURM_ARRAY_TASK_ID]} \
    --year=2022 \
    --output="distillation_results/synthplus-mdeberta-${SLURM_ARRAY_TASK_ID}" \
    --pretrained-model="microsoft/mdeberta-v3-base" \
    --encoder-model="DeBERTa" \
    --word-layer=8 \
    --word-level=True \
    --hidden-sizes 3072 1024 

# srun --partition=single --gres=gpu:1 --mem=128G --cpus-per-task=16 --pty --time=4:00:00 bash