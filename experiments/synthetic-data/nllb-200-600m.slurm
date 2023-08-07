#!/usr/bin/env bash

#SBATCH --job-name=translate-NLLB-600m
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --output=./experiments/synthetic-data/logs/translate-NLLB-600m-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="
python ./synthetic-data/gen_translations.py \
    --model="facebook/nllb-200-distilled-600M" \
    --nbits=4 \
    --sample=15000 \
    --save_path="./synthetic-data/data/he-en-nllb600m.csv" \
    --batch_size=16

echo "============="
echo "Job end: "
echo $(date)
echo "============="