import os
import time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import torch
import torch.nn as nn
import lightning as L
from datasets import load_dataset
from comet.models.multitask.xcomet_metric import XCOMETMetric

from utils import load_json, dump_json
from source.mqm_dataset import MQMDataset

###
# Implemetation scheme:
# - Data parser from tsv to something json-like
# - Hand-written dataset 
# - Model
# - CustomTrainer with compute_loss and compute_metrics

# Changes
# - Data parser -> use published parsed dataset
# - CustomTrainer -> hand-written train loop
###

def make_parser():
    parser = ArgumentParser(description="MQM finetuning.")
    parser.add_argument("-o", "--output", help="Where to save results, in format root_results_directory/experiment_name", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed to fix")
    parser.add_argument("--n-gpus", type=int, default=1, help="Amount of GPUs utilized")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--use-wandb", type=bool, default=False, action="store_true", help="Whether to use wandb logging")

    parser.add_argument("--encoder-model", default="MiniLM", help="Backbone family [BERT, XLM-RoBERTa, MiniLM, XLM-RoBERTa-XL, RemBERT]")
    parser.add_argument("--pretrained-model", default="microsoft/Multilingual-MiniLM-L12-H384", help="Concrete pretrained checkpoint for the backbone (huggingface name)")
    parser.add_argument("--word-level", type=bool, default=True, help="Whether to use word-level annotations")
    parser.add_argument("--word-layer", type=int, help="From which layer of encoder to predict word tags", required=True)

    return parser

def print_summary(report: dict):
    print("Dataset load time:", report["dataset_load_time"])
    print("Model load time:", report["model_load_time"])
    print("Train time:", report["train_time"], "\n")
    print("Max memory:", report["peak_memory_mb"], "Mb")
    print("Train kendall correlation:", report["train_kendall_correlation"])
    print("Validation kendall correlation:", report["val_kendall_correlation"])


# Option A: hardcode error-span dataset, hardcode splits into train/val/test
# Option B: Make distinct functions and argument sets for train, val and test

def get_datasets(args, track_time):
    start = time.perf_counter()

    path = "data/mqm-spans-with-year-and-domain-but-no-news-2022.csv"
    test_path = "data/wmt-mqm-human-evaluation.csv"

    val_predicate = lambda x: x["domain"] == "social" and x["year"] == "2022.0"
    train_predicate = lambda x: not val_predicate(x)

    train_dataset = MQMDataset(path)
    train_dataset.filter(train_predicate)

    val_dataset = MQMDataset(path)
    val_dataset.filter(val_predicate)
    
    # For debugging purposes
    train_dataset.data = train_dataset.data.iloc[:100]
    val_dataset.data = val_dataset.data.iloc[:100]

    dataset_load_time = time.perf_counter() - start

    if track_time:
        return train_dataset, val_dataset, dataset_load_time
    return train_dataset, val_dataset

def get_model(args, track_time):
    start = time.perf_counter()
    model = XCOMETMetric(
        encoder_model=args.encoder_model,
        pretrained_model=args.pretrained_model,
        word_level_training=args.word_level,
        word_layer=args.word_layer,
        validation_data=[]
    )
    model_load_time = time.perf_counter() - start

    if track_time:
        return model, model_load_time
    return model

def train_one_epoch(model, optimizer, train_dataloader, use_wandb):
    losses = []

    for batch in train_dataloader:
        optimizer.zero_grad()

        inputs, target = model.prepare_sample(batch)
        
        output = model(inputs)
        # output is a dict {"sentemb": Tensor, "wordemb": Tensor, "all_layers": Tensor, "attention_mask": Tensor}
        loss = model.compute_loss(output, target)
        
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if use_wandb:
            wandb.log({
                "loss": loss.item(),
            })
        
    return losses

@torch.inference_mode()
def evaluate_model(model, val_dataloader, prefix):
    true_scores = []
    model_scores = []

    for batch in val_dataloader:
        inputs, target = model.prepare_sample(batch)
        output = model(inputs)

        true_scores.append(target.score.detach().cpu())
        model_scores.append(output.score.detach().cpu())
    
    true_scores = torch.cat(true_scores).numpy()
    model_scores = torch.cat(model_scores).numpy()

    kendall_result = kendalltau(true_scores, model_scores)
    return {
        f"{prefix}kendall_correlation": kendall_result[0],
        f"{prefix}mse": np.mean(np.square(model_scores - true_scores)),
        f"{prefix}mae": np.mean(np.abs(model_scores - true_scores)),
        f"{prefix}kendall_p_value": kendall_result[1],
    }

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
    output_path = Path(args.output) / "training"

    if os.path.exists(output_path):
        print("Reusing previous results. Change output folder or delete this folder to recompute.")
        return

# Start logic

# Create directories
    os.makedirs(output_path, exist_ok=True)

# Data
    print("Loading datasets...")
    train_dataset, val_dataset, dataset_load_time = get_dataset(args, track_time=True)
    for d, part in zip((train_dataset, val_dataset), ("train", "val")):
        print(part)
        print("N samples:", len(d))
        print("First sample:\n", d[0], "\n")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shiffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shiffle=False)

# Model
    # note: apparently, it has some elaborate pooling and learning rate distribution
    model, model_load_time = get_model(args, track_time=True)

# Train loop
    optimizer = torch.AdamW(model.layerwise_lr(args.lr, args.lr_decay))

    if args.use_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project_name, config=vars(args))

    val_metrics = []
    train_metrics = []
    losses = []

    train_start = time.perf_counter()

    for epoch in range(args.n_epochs):
        losses.extend(train_one_epoch(model, optimizer, train_dataloader, args.use_wandb))
        torch.save(model.state_dict(), output_path / "checkpoint.pth")
        np.save(output_path / "losses.npy", losses)

        train_metrics.append(evaluate_model(model, train_dataloader, "train_"))
        pd.DataFrame({"epoch": epoch + 1} | train_metrics).to_csv(output_path / "train_metrics.csv", index=False)

        val_metrics.append(evaluate_model(model, val_dataloader, "val_"))
        pd.DataFrame({"epoch": epoch + 1} | val_metrics).to_csv(output_path / "val_metrics.csv", index=False)

        if args.use_wandb:
            wandb.log(train_metrics[-1] | val_metrics[-1])

    train_time = time.perf_counter() - train_start
# Construct report
    peak_memory_mb = torch.cuda.max_memory_allocated() // 2 ** 20

    report = {
        "peak_memory_mb": peak_memory_mb,
        "dataset_load_time": round(dataset_load_time, 2),
        "model_load_time": round(model_load_time, 2),
        "train_time": round(train_time, 2),
        "train_dataset_length": len(train_dataset),
        "val_dataset_length": len(val_dataset),
    }
    report = report | train_metrics[-1]
    report = report | val_metrics[-1]
    report = report | vars(args)
    report = report | {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch.version.cuda": torch.version.cuda,  # type: ignore[code]
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),  # type: ignore[code]
        "torch.cuda.nccl.version()": torch.cuda.nccl.version(),  # type: ignore[code]
    }

    dump_json(report, output_path / "report.json")

# Finish
    print_summary(report)


if __name__ == "__main__":
    main()