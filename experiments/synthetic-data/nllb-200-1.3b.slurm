#!/usr/bin/env bash

#SBATCH --job-name=translate-NLLB-1.3B
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --output=./experiments/synthetic-data/logs/translate-NLLB-1.3B-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

echo "============="
echo "Job start: "
echo $(date) $(hostname) $(pwd)
echo "============="
python ./synthetic-data/gen_translations.py \
    --model="facebook/nllb-200-distilled-1.3B" \
    --nbits=4 \
    --seed=128 \
    --sample=20000 \
    --save_path="./synthetic-data/data/he-en-nllb1.3B.csv" \
    --batch_size=8

echo "============="
echo "Job end: "
echo $(date)
echo "============="