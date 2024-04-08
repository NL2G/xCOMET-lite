#!/bin/bash
#SBATCH --job-name=prune-xxl
#SBATCH --output=./logs/prune-xxl-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100,gpu_4_h100,gpu_4,gpu_8
#SBATCH --array=0-2

prune_layers_options=(
    0
    4
    8
    12
    16
    20
)
seeds=(

)

n_layers_to_prune=${prune_layers_options[$SLURM_ARRAY_TASK_ID]}
exp_name="prune-xxl-${n_layers_to_prune}-layers"

model="Unbabel/XCOMET-XXL"

python prune_finetune.py \
    -o pruning_results/$exp_name/ \
    --lp "en-de" \
    --dataset RicardoRei/wmt-mqm-human-evaluation \
    --n-layers-to-prune $n_layers_to_prune \
    --model $model \
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