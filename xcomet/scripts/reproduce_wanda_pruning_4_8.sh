# Example of usage:
# bash scripts/reproduce_wanda_pruning_4_8.sh "reproduction_wanda" "Unbabel/XCOMET-XL" 0 0

export CUDA_DEVICE_ORDER=PCI_BUS_ID

exp_name=$1
model=$2
devices=$3
seed=$4

if [ $model == "Unbabel/XCOMET-XL" ]; then
    model_tag="xl"
elif [ $model == "Unbabel/XCOMET-XXL" ]; then
    model_tag="xxl"
else
    echo "Unsupported model"
    exit 1
fi

for lp in "en-ru" "en-de" "zh-en"
do
    CUDA_VISIBLE_DEVICES=${devices} python prune_finetune.py \
        -o pruning_results/${exp_name}-${model_tag}-seed-${seed}/ \
        --lp ${lp} \
        --dataset RicardoRei/wmt-mqm-human-evaluation \
        --model ${model} \
        --use-wanda \
        --seed ${seed} \
        --structured-pruning-n 4 \
        --structured-pruning-m 8
done

for lp in "en-de" "en-es" "en-zh"
do
    CUDA_VISIBLE_DEVICES=${devices} python prune_finetune.py \
        -o pruning_results/${exp_name}-${model_tag}-seed-${seed}/ \
        --lp ${lp} \
        --dataset data/mt_${lp}_ground_truth_cleaned_l.tsv \
        --model ${model} \
        --use-wanda \
        --seed ${seed} \
        --structured-pruning-n 4 \
        --structured-pruning-m 8
done