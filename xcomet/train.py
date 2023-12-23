import os
import time
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser
from collections import defaultdict

import wandb
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
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of passes through the train set")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use wandb logging")
    parser.add_argument("--wandb-project-name", type=str, default="xcomet-compression", help="The name of project in wandb")

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
    print("Train full kendall correlation:", report["train_full_kendall_correlation"])
    print("Validation full kendall correlation:", report["val_full_kendall_correlation"])


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
    #train_dataset.data = train_dataset.data.sample(n=1000, random_state=11)
    #val_dataset.data = val_dataset.data.sample(n=1000, random_state=11)

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

def train_one_epoch(model, optimizer, train_dataloader, use_wandb, device):
    model.train()
    losses = []

    for batch in tqdm(train_dataloader, desc="training"):
        optimizer.zero_grad()

        inputs_tuple, target = model.prepare_sample(batch)
        target.score = target.score.to(device)

        loss = 0
        
        # src + ref, src only, ref only
        for inputs in inputs_tuple:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            output = model(**inputs)

            seq_len = target.mt_length.max()
            output.logits = output.logits[:, :seq_len]

            loss = loss + model.compute_loss(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if use_wandb:
            wandb.log({
                "loss": loss.item(),
            })
        
    return losses

@torch.inference_mode()
def evaluate_model(model, val_dataloader, prefix, device):
    model.eval()
    input_types = ("src", "ref", "full")

    true_scores = []
    n_samples = 0
    model_scores = defaultdict(list)
    tag_accuracy = defaultdict(float)

    for batch in tqdm(val_dataloader, desc="evaluation"):
        inputs_tuple, target = model.prepare_sample(batch)
        assert len(inputs_tuple) == 3, "Only support training mode with (src, ref, full_input) as input"

        true_scores.append(target.score)
        n_samples += target.score.shape[0]
        
        target.score = target.score.to(device)
        target.labels = target.labels.to(device)

        # src + ref, src only, ref only
        for inputs, input_type in zip(inputs_tuple, input_types):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            output = model(**inputs)

            seq_len = target.mt_length.max()
            output.logits = output.logits[:, :seq_len]

            model_scores[input_type].append(output.score.detach().cpu())
            tag_accuracy[input_type] += (output.logits.argmax(-1) == target.labels).float().mean() * target.score.shape[0]

    true_scores = torch.cat(true_scores).numpy()
    model_scores = {k: torch.cat(v).numpy() for k, v in model_scores.items()}
    kendall_results = {
        k: kendalltau(true_scores, model_scores[k]) for k in input_types
    }
    tag_accuracy = {k: v / n_samples for k, v in tag_accuracy.items()}

    metrics = {
        f"{prefix}{k}_kendall_correlation": kendall_results[k][0] for k in input_types
    }

    metrics = metrics | {
        f"{prefix}{k}_tag_accuracy": tag_accuracy[k].item() for k in input_types
    }

    return metrics

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
    train_dataset, val_dataset, dataset_load_time = get_datasets(args, track_time=True)
    for d, part in zip((train_dataset, val_dataset), ("train", "val")):
        print(part)
        print("N samples:", len(d))
        print("First sample:\n", d[0], "\n")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=lambda x: x)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=lambda x: x)

# Model
    # note: apparently, it has some elaborate pooling and learning rate distribution
    device = "cuda:0"
    model, model_load_time = get_model(args, track_time=True)
    model.to(device)

# Train loop
    optimizers, schedulers = model.configure_optimizers()
    assert len(schedulers) == 0, len(optimizers) == 1
    optimizer = optimizers[0]

    if args.use_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project_name, config=vars(args), name=args.output.split("/")[-1])

    val_metrics = []
    train_metrics = []
    losses = []

    train_start = time.perf_counter()

    for epoch in range(args.n_epochs):
        losses.extend(train_one_epoch(model, optimizer, train_dataloader, args.use_wandb, device))
        torch.save(model.state_dict(), output_path / "checkpoint.pth")
        np.save(output_path / "losses.npy", losses)

        train_metrics.append(evaluate_model(model, train_dataloader, "train_", device))
        pd.DataFrame(train_metrics).to_csv(output_path / "train_metrics.csv", index=False)

        val_metrics.append(evaluate_model(model, val_dataloader, "val_", device))
        pd.DataFrame(val_metrics).to_csv(output_path / "val_metrics.csv", index=False)

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