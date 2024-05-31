#!/bin/bash
#SBATCH --job-name=speed-eval-quantization
#SBATCH --output=./logs/speed-xxl-eval-quantization-%A-[%a].txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dev_gpu_4_a100,gpu_4_a100
#SBATCH --array=0-11

max_cpu_threads=16
export OMP_NUM_THREADS=$max_cpu_threads
export OPENBLAS_NUM_THREADS=$max_cpu_threads
export MKL_NUM_THREADS=$max_cpu_threads
export VECLIB_MAXIMUM_THREADS=$max_cpu_threads
export NUMEXPR_NUM_THREADS=$max_cpu_threads
export NUMEXPR_MAX_THREADS=$max_cpu_threads

model="Unbabel/XCOMET-XXL"
script_name="eval.py"
root_exp_dir="speed_results"

n_bits_options=(
    "8"
    "8"
    "8"
    "4"
    "4"
    "4"
    "3"
    "3"
    "3"
    "2"
    "2"
    "2"
)

no_ref_language=(
    "en-de"
    "en-es"
    "en-zh"
    "en-de"
    "en-es"
    "en-zh"
    "en-de"
    "en-es"
    "en-zh"
    "en-de"
    "en-es"
    "en-zh"
)
n_bits=${n_bits_options[$SLURM_ARRAY_TASK_ID]}

no_ref=${no_ref_language[$SLURM_ARRAY_TASK_ID]}
output=${root_exp_dir}/a100_xxl_${n_bits}bits

echo "Running ${no_ref} with ${n_bits} bits"

echo "Running ${no_ref}"
python ${script_name} \
    -o ${output}/ \
    --lp $no_ref \
    --dataset data/mt_${no_ref}_ground_truth_cleaned_l.tsv \
    --model ${model} \
    --gpu \
    --quantize-n-bits ${n_bits} \
    --batch-size 0

