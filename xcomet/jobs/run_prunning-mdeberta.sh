#!/bin/bash
#SBATCH --job-name=prune-md
#SBATCH --output=./logs/prune-md-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=single
#SBATCH --array=0-2

prune_layers_options=(
    2
    4
    6
)

seeds=(
    0
    0
    0
)

n_layers_to_prune=${prune_layers_options[$SLURM_ARRAY_TASK_ID]}
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
exp_name="prune-mdeberta-${n_layers_to_prune}-layers-seed-${seed}"

model="distillation_results/synthplus-mdeberta-1epoch-2/training/checkpoint.pth"

python prune_finetune.py \
    -o pruning_results/$exp_name/ \
    --lp "en-de" \
    --dataset RicardoRei/wmt-mqm-human-evaluation \
    --n-layers-to-prune $n_layers_to_prune \
    --model $model \
    --seed $seed \
    --do-finetune

for lp in "en-ru" "en-de" "zh-en"
do
    python prune_finetune.py \
        -o pruning_results/$exp_name/ \
        --lp $lp \
        --dataset RicardoRei/wmt-mqm-human-evaluation \
        --n-layers-to-prune $n_layers_to_prune \
        --model $model
done

for lp in "en-de" "en-es" "en-zh"
do
    python prune_finetune.py \
        -o pruning_results/$exp_name/ \
        --lp $lp \
        --dataset data/mt_${lp}_ground_truth_cleaned_l.tsv \
        --n-layers-to-prune $n_layers_to_prune \
        --model $model
done