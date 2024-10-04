# efficient-llm-metrics
Efficient &amp; Open LLM-based metrics for NLG tasks

## Setup environment
```
conda create -n metric python=3.11.5 pandas=2.1.1 pytorch=2.1.1 torchvision=0.16.1 pytorch-cuda=11.8 scipy=1.11.3 -c pytorch -c nvidia

conda activate metric

# to use GPTQ quantization
pip install auto-gptq==0.5.1
pip install optimum==1.14.1 accelerate==0.24.1
pip install --upgrade git+https://github.com/huggingface/transformers.git

# to use BnB quantization
pip install bitsandbytes

# to access xCOMET models
pip install --upgrade pip
pip install "unbabel-comet>=2.2.0"
huggingface-cli login               # will have to enter your huggingface access token

# To train distilled models
pip install lightning==2.1.2 wandb

# Visualization
pip install jupyterlab==4.0.9 matplotlib rich

# Onnx for speeding up
pip install onnxruntime
```

## Quick start

To run magnitude pruning with 4:8 sparsity pattern on English-Russian language pair for XCOMET-XL model, run
```
devices="0"
model="Unbabel/XCOMET-XL"
exp_name="xl_magnitude_pruning_4_8"

CUDA_VISIBLE_DEVICES=${devices} python prune_finetune.py \
    -o pruning_results/${exp_name}/ \
    --lp en-ru \
    --dataset RicardoRei/wmt-mqm-human-evaluation \
    --model ${model} \
    --use-magnitude-pruning \
    --structured-pruning-n 4 \
    --structured-pruning-m 8
```

## Outline of code structure

There are three main scripts:
- `eval.py` is used to evaluate compressed models throughput. Supports quantization, pruning and loading of distilled models.
- `prune_finetune.py` has two modes, depending on the `--do-finetune` flag.
  - If the flag is set, the script prunes, finetunes and saves the model (more concretely, it only saves the small subset of parameters which were finetuned).
  - If the flag is not set, the script prunes, tries to load finetuned parameters and runs evaluation.
- `train_three_stages.py` is used to train the distilled model.

Results are usually saved as json reports in `{pruning, quantization, distillation, speed}_results` directories. To get summarized results, run `aggregate_results.py`.

#### Other scripts:
- `deberta_encoder.py` contains an implementation of DeBERTa encoder, which was used as a backbone for distilled model.
- `eval_checkpoint.py` was initially used to evaluate distilled models, until it was supported in `eval.py`.
- `merge_mqm_and_error_spans.ipynb` was used to prepare the finetuning dataset for `prune_finetune.py` script.
- `reduce_results.py` was used to summarize the results of pruning experiments with different random seeds, which were presented in the appendix of the paper.
- `train.py` was a first version of the script used to train distilled models, which used a simplified traning process.
- `utils.py` contains a variety of auxilary functions and classes.