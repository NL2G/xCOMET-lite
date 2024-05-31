import os
import time
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from deberta_encoder import DeBERTaEncoder
import comet.encoders

comet.encoders.str2encoder["DeBERTa"] = DeBERTaEncoder

import wandb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import torch
import torch.nn as nn
from datasets import load_dataset
from comet.models.multitask.xcomet_metric import XCOMETMetric

from utils import load_json, dump_json, load_tsv
from source.mqm_dataset import MQMDataset

def make_parser():
    parser = ArgumentParser(description="MQM evaluation.")
    parser.add_argument("-o", "--output", help="Where to save results, in format root_results_directory/experiment_name", required=True)
    parser.add_argument("--lp", help="On which language pair to evaluate model", required=True)
    parser.add_argument("--dataset", help="Which dataset to use (huggingface dataset/path to tsv file)", required=True)
    parser.add_argument("--domain", default="news", help="On which domain to evaluate model")
    parser.add_argument("--year", type=int, default=2022, help="For which year to compute metrics")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to fix")
    parser.add_argument("--n-gpus", type=int, default=1, help="Amount of GPUs utilized")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")

    parser.add_argument("--encoder-model", default="MiniLM", help="Backbone family [BERT, XLM-RoBERTa, MiniLM, XLM-RoBERTa-XL, RemBERT]")
    parser.add_argument("--pretrained-model", default="microsoft/Multilingual-MiniLM-L12-H384", help="Concrete pretrained checkpoint for the backbone (huggingface name)")
    parser.add_argument("--word-layer", type=int, default=8, help="From which layer of encoder to predict word tags")
    parser.add_argument("--word-level", type=bool, default=True, help="Whether to use word-level annotations")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=(3072, 1024), help="Size of hidden layers used in regression head")

    return parser

def print_summary(report: dict):
    print("Dataset load time:", report["dataset_load_time"])
    print("Model load time:", report["model_load_time"])
    print("Prediction time:", report["prediction_time"], "\n")
    print("Max memory:", report["peak_memory_mb"], "Mb")
    print("Kendall correlation:", report["kendall_correlation"])


def get_dataset(args, track_time):
    print("Loading dataset...")
    start = time.perf_counter()

    if args.dataset.endswith(".tsv"):
        print(f"Ignoring arguments domain={args.domain}, year={args.year} and lp={args.lp} -- not implemented for local .tsv datasets.")
        dataset = load_tsv(args.dataset)
        ground_truth = dataset["score"]
        dataset = list(dataset.T.to_dict().values())
    else:
        dataset = load_dataset(args.dataset, split="train")
        dataset = dataset.filter(lambda example: 
            example["year"] == args.year and example["domain"] == args.domain and example["lp"] == args.lp)
        ground_truth = dataset["score"]
        dataset = [sample for sample in dataset]

    dataset_load_time = time.perf_counter() - start
    print("N samples:", len(dataset))
    print("First sample:\n", dataset[0], "\n")

    if track_time:
        return dataset, ground_truth, dataset_load_time
    return dataset, ground_truth


def get_model(args, track_time):
    start = time.perf_counter()
    model = XCOMETMetric(
        encoder_model=args.encoder_model,
        pretrained_model=args.pretrained_model,
        word_layer=args.word_layer,
        validation_data=[],
        word_level_training=args.word_level,
        hidden_sizes=args.hidden_sizes,
        load_pretrained_weights=False,
    )

    checkpoint_path = Path(args.output) / "training" / "checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path))

    model_load_time = time.perf_counter() - start

    if track_time:
        return model, model_load_time
    return model


def run_metric(model, dataset, args):
    print("Computing metric...")
    start = time.perf_counter()
    model_output = model.predict(dataset, batch_size=args.batch_size, gpus=args.n_gpus)
    prediction_time = time.perf_counter() - start

    return model_output, prediction_time

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
    output_path = Path(args.output) / "evaluations" / ("no_reference" if args.dataset.endswith(".tsv") else "with_reference") / args.lp

    #if os.path.exists(output_path):
    #    print("Reusing previous results. Change output folder or delete this folder to recompute.")
    #    return

# Start logic

# Create directories
    os.makedirs(output_path, exist_ok=True)

# Data
    dataset, ground_truth, dataset_load_time = get_dataset(args, track_time=True)

# Model
    # note: apparently, it has some elaborate pooling and learning rate distribution
    device = "cuda:0"
    model, model_load_time = get_model(args, track_time=True)
    model.to(device)

    model_output, prediction_time = run_metric(model, dataset, args)

    segment_scores = np.array(model_output.scores)
# Construct report
    peak_memory_mb = torch.cuda.max_memory_allocated() // 2 ** 20
    kendall_corr = kendalltau(ground_truth, segment_scores)

    report = {
        "kendall_correlation": kendall_corr[0],
        "kendall_p_value": kendall_corr[1],
        "peak_memory_mb": peak_memory_mb,
        "system_level_score": model_output.system_score,
        "dataset_load_time": round(dataset_load_time, 2),
        "model_load_time": round(model_load_time, 2),
        "prediction_time": round(prediction_time, 2),
        "dataset_length": len(ground_truth),
    }
    report = report | vars(args)
    report = report | {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch.version.cuda": torch.version.cuda,  # type: ignore[code]
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),  # type: ignore[code]
        "torch.cuda.nccl.version()": torch.cuda.nccl.version(),  # type: ignore[code]
    }

# Save artifacts
    np.save(output_path / "model_segment_level_scores.npy", segment_scores)
    dump_json(report, output_path / "report.json")
    dump_json(model_output.metadata.error_spans, output_path / "error_spans.json")

# Finish
    print_summary(report)


if __name__ == "__main__":
    main()