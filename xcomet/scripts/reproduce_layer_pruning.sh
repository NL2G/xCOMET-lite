# Example of usage:
# bash scripts/reproduce_layer_pruning.sh "reproduction_layer_pruning" "Unbabel/XCOMET-XL" 0 0 8

export CUDA_DEVICE_ORDER=PCI_BUS_ID

exp_name=$1
model=$2
devices=$3
seed=$4
n_layers_to_prune=$5

# First, generic parameter-efficient finetuning is performed with --do-finetune flag on:
CUDA_VISIBLE_DEVICES=$devices python prune_finetune.py \
    -o pruning_results/$exp_name/ \
    --lp "en-de" \
    --dataset RicardoRei/wmt-mqm-human-evaluation \
    --n-layers-to-prune $n_layers_to_prune \
    --model $model \
    --do-finetune \
    --seed $seed

# Then, finetuned model is evaluated
for lp in "en-ru" "en-de" "zh-en"
do
    CUDA_VISIBLE_DEVICES=$devices python prune_finetune.py \
        -o pruning_results/$exp_name/ \
        --lp $lp \
        --dataset RicardoRei/wmt-mqm-human-evaluation \
        --n-layers-to-prune $n_layers_to_prune \
        --model $model \
        --seed $seed
done

for lp in "en-de" "en-es" "en-zh"
do
    CUDA_VISIBLE_DEVICES=$devices python prune_finetune.py \
        -o pruning_results/$exp_name/ \
        --lp $lp \
        --dataset data/mt_${lp}_ground_truth_cleaned_l.tsv \
        --n-layers-to-prune $n_layers_to_prune \
        --model $model \
        --seed $seed
done