# Example of usage:
# bash scripts/reproduce_gptq.sh "reproduction_gptq" "Unbabel/XCOMET-XL" 0 0 

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

for nbits in 2 3 4 8
do
    for lp in "en-ru" "en-de" "zh-en"
    do
        CUDA_VISIBLE_DEVICES=${devices} python eval.py \
            -o quantization_results/${exp_name}-${model_tag}-seed-${seed}/ \
            --lp ${lp} \
            --dataset RicardoRei/wmt-mqm-human-evaluation \
            --model ${model} \
            --seed ${seed} \
            --quantization-type gptq \
            --quantize-n-bits ${nbits} \
            --gpu
    done

    for lp in "en-de" "en-es" "en-zh"
    do
        CUDA_VISIBLE_DEVICES=${devices} python eval.py \
            -o quantization_results/${exp_name}-${model_tag}-seed-${seed}/ \
            --lp ${lp} \
            --dataset data/mt_${lp}_ground_truth_cleaned_l.tsv \
            --model ${model} \
            --seed ${seed} \
            --quantization-type gptq \
            --quantize-n-bits ${nbits} \
            --gpu
    done
done