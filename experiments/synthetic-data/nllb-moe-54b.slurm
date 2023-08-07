#!/usr/bin/env bash

#SBATCH --job-name=translate-NLLB-54B
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --output=./experiments/synthetic-data/logs/translate-NLLB-54B-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="
python ./synthetic-data/gen_translations.py \
    --model="facebook/nllb-moe-54b" \
    --nbits=4 \
    --seed=128 \
    --sample=15000 \
    --save_path="./synthetic-data/data/he-en-nllb54B.csv" \
    --batch_size=1

echo "============="
echo "Job end: "
echo $(date)
echo "============="