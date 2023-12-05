import os
import time
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from utils import load_json, dump_json

###
# Implemetation scheme:
# - Data parser from tsv to something json-like
# - Hand-written dataset 
# - Model
# - CustomTrainer with compute_loss and compute_metrics
###

def make_parser():
    raise NotImplementedError("Not adapted from GPTQ eval yet.")

    parser = ArgumentParser(description="MQM finetuning.")
    parser.add_argument("-o", "--output", help="Where to save results, in format root_results_directory/experiment_name", required=True)
    parser.add_argument("--model", help="Which model to use (name on huggingface)", required=True)
    parser.add_argument("--dataset", help="Which dataset to use (huggingface dataset/path to tsv file)", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed to fix")
    parser.add_argument("--n-gpus", type=int, default=1, help="Amount of GPUs utilized")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--estimator-dropout", type=float, default=0.1, help="Rate of dropout in MQM-score predicting branch of the model")

    return parser

def print_summary(report: dict):
    print("Dataset load time:", report["dataset_load_time"])
    print("Model load time:", report["model_load_time"])
    print("Prediction time:", report["prediction_time"], "\n")
    print("Max memory:", report["peak_memory_mb"], "Mb")
    print("Kendall correlation:", report["kendall_correlation"])


@torch.inference_mode()
def main():
# Get arguments
    parser = make_parser()
    args = parser.parse_args()
    print(args)

# Setup environment
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Check for earlier launches
    output_path = Path(args.output) / args.lp

    if os.path.exists(output_path):
        print("Reusing previous results. Change output folder or delete this folder to recompute.")
        segment_scores = np.load(output_path / "model_segment_level_scores.npy")
        error_spans = load_json(output_path / "error_spans.json")
        report = load_json(output_path / "report.json")
        print_summary(report)

# Start logic

# Create directories
    os.makedirs(output_path, exist_ok=True)

# Data
    train_dataloader = ... # synthetic + some real data 
    val_dataloader = ... # some hold-out real data
    test_dataloader = ... # news 2022

# Model
    model = ... # multilingual miniml + mqm head
    # note: apparently, it has some elaborate pooling and learning rate distribution

# Custom trainer
    trainer = ... # custom trainer with redefined compute_loss, compute_metrics and preprocess_logis_for_metrics
    # track training time

# Construct report
    peak_memory_mb = torch.cuda.max_memory_allocated() // 2 ** 20
    kendall_corr = kendalltau(ground_truth, segment_scores)

    model_output = ...
    dataset_load_time = ...
    model_load_time = ...
    quantization_time = ...
    dataset = ...

    report = {
        "kendall_correlation": kendall_corr[0],
        "kendall_p_value": kendall_corr[1], 
        "peak_memory_mb": peak_memory_mb,
        "system_level_score": model_output.system_score,
        "dataset_load_time": round(dataset_load_time, 2),
        "model_load_time": round(model_load_time, 2),
        "quantization_time": round(quantization_time, 2),
        "prediction_time": round(prediction_time, 2),
        "dataset_length": len(dataset),
    }
    report = report | vars(args)
    report = report | {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch.version.cuda": torch.version.cuda,  # type: ignore[code]
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),  # type: ignore[code]
        "torch.cuda.nccl.version()": torch.cuda.nccl.version(),  # type: ignore[code]
    }    

# Save artefacts
    np.save(output_path / "model_segment_level_scores.npy", segment_scores)
    dump_json(report, output_path / "report.json")
    dump_json(model_output.metadata.error_spans, output_path / "error_spans.json")

# Finish
    print_summary(report)


if __name__ == "__main__":
    main()