# xCOMET-lite

[EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1223/) | [Arxiv](https://arxiv.org/abs/2406.14553) | [Distilled model on HF](https://huggingface.co/myyycroft/XCOMET-lite)

Efficient learned metrics for translation quality evaluation.


## Setup environment
```
conda create -n metric python=3.11.5 pandas=2.1.1 pytorch=2.1.1 torchvision=0.16.1 pytorch-cuda=11.8 scipy=1.11.3 -c pytorch -c nvidia

conda activate metric

# to use GPTQ quantization
pip install auto-gptq==0.5.1
pip install optimum==1.14.1 accelerate==0.24.1
pip install transformers

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

# Workaround for tokenizers
pip install protobuf==3.20
```

## Quick start

To try out xCOMET-lite, run the following code:

```
from xcomet.deberta_encoder import XCOMETLite

model = XCOMETLite().from_pretrained("myyycroft/XCOMET-lite")
data = [
    {
        "src": "Elon Musk has acquired Twitter and plans significant changes.",
        "mt": "Илон Маск приобрел Twitter и планировал значительные искажения.",
        "ref": "Илон Маск приобрел Twitter и планирует значительные изменения."
    },
    {
        "src": "Elon Musk has acquired Twitter and plans significant changes.",
        "mt": "Илон Маск приобрел Twitter.",
        "ref": "Илон Маск приобрел Twitter и планирует значительные изменения."
    }
]

model_output = model.predict(data, batch_size=2, gpus=1)

print("Segment-level scores:", model_output.scores)
```

## Quantization experiments

To run GPTq quantization to 3 bits on English-Russian language pair for XCOMET-XL model, run

```
devices="0"
model="Unbabel/XCOMET-XL"
exp_name="xl_gptq_3bit"

CUDA_VISIBLE_DEVICES=${devices} python eval.py \
    -o quantization_results/${exp_name}/ \
    --lp en-ru \
    --dataset RicardoRei/wmt-mqm-human-evaluation \
    --model ${model} \
    --quantization-type gptq \
    --quantize-n-bits 3 \
    --half \
    --gpu
```
from `xcomet` subdirectory.

> [!TIP]
> If you want to use quantized version of XCOMET in your code, you can just copy `quantize_model_gptq` function and make sure you imported `GPTQQuantizer` from `optimum.gptq`.

## Pruning experiments

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
from `xcomet` subdirectory.

Other experiments can be reproduced similarly.

## Outline of code structure

There are three main scripts:
- `eval.py` is used to evaluate compressed models quality and throughput. Supports quantization, pruning and loading distilled models.
- `prune_finetune.py` has two modes, depending on the `--do-finetune` flag.
  - If the flag is set, the script prunes, finetunes and saves the model (more concretely, it only saves the small subset of parameters which were finetuned).
  - If the flag is not set, the script prunes, tries to load finetuned parameters and runs evaluation.
- `train_three_stages.py` is used to train the distilled model.

Results are usually saved as json reports in `{pruning, quantization, distillation, speed}_results` directories. To get summarized results, run `aggregate_results.py`.

#### Other scripts:
- `deberta_encoder.py` contains an implementation of DeBERTa encoder, which was used as a backbone for distilled model, and a wrapper for XCOMET-lite model.
- `eval_checkpoint.py` was initially used to evaluate distilled models, until it was supported in `eval.py`.
- `merge_mqm_and_error_spans.ipynb` was used to prepare the finetuning dataset for `prune_finetune.py` script.
- `reduce_results.py` was used to summarize the results of pruning experiments with different random seeds, which were presented in the appendix of the paper.
- `train.py` was a first version of the script used to train distilled models, which used a simplified traning process.
- `utils.py` contains a variety of auxilary functions and classes.

## Citation

```
@misc{larionov2024xcometlitebridginggapefficiency,
      title={xCOMET-lite: Bridging the Gap Between Efficiency and Quality in Learned MT Evaluation Metrics}, 
      author={Daniil Larionov and Mikhail Seleznyov and Vasiliy Viskov and Alexander Panchenko and Steffen Eger},
      year={2024},
      eprint={2406.14553},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.14553}, 
}
```