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
from datasets import load_dataset
from comet.models.multitask.xcomet_metric import XCOMETMetric

from utils import load_json, dump_json, MQMDataset

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
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (equals effective batch size when grad-accum-steps=1)")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Number of batches processed before weights update")
    parser.add_argument("--phase-one-epochs", type=int, default=3, help="Number of passes through the train set with DA scores")
    parser.add_argument("--phase-two-epochs", type=int, default=3, help="Number of passes through the train set with MQM scores, with emphasis on word-level task")
    parser.add_argument("--phase-three-epochs", type=int, default=3, help="Number of passes through the train set with MQM scores, with emphasis on sentence-level task")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use wandb logging")
    parser.add_argument("--wandb-project-name", type=str, default="xcomet-compression", help="The name of project in wandb")

    parser.add_argument("--nr-frozen-epochs", type=float, default=0.9, help="Number of epochs (% of epoch) that the encoder is frozen")
    parser.add_argument("--encoder-lr", type=float, default=1e-06, help="Base learning rate for the encoder part")
    parser.add_argument("--lr", type=float, default=3.66e-06, help="Learning rate for top layers")
    parser.add_argument("--layerwise-decay", type=float, default=0.983, help="Learning rate decay from last to first encoder layers")
    parser.add_argument("--encoder-model", default="MiniLM", help="Backbone family [BERT, XLM-RoBERTa, MiniLM, XLM-RoBERTa-XL, RemBERT]")
    parser.add_argument("--pretrained-model", default="microsoft/Multilingual-MiniLM-L12-H384", help="Concrete pretrained checkpoint for the backbone (huggingface name)")
    parser.add_argument("--word-layer", type=int, default=8, help="From which layer of encoder to predict word tags")
    parser.add_argument("--word-level", type=bool, default=True, help="Whether to use word-level annotations")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=(3072, 1024), help="Size of hidden layers used in regression head")
    parser.add_argument("--loss-lambda-phase-two", type=float, default=0.983, help="Weight assigned to the word-level loss in Phase 2")
    parser.add_argument("--loss-lambda-phase-three", type=float, default=0.055, help="Weight assigned to the word-level loss in Phase 3")

    return parser

def print_summary(report: dict):
    print("Dataset load time:", report["dataset_load_time"])
    print("Model load time:", report["model_load_time"])
    print("Train time:", report["train_time"], "\n")
    print("Max memory:", report["peak_memory_mb"], "Mb")


# Option A: hardcode error-span dataset, hardcode splits into train/val/test
# Option B: Make distinct functions and argument sets for train, val and test

def get_datasets(args, track_time):
    start = time.perf_counter()

    mqm_path = "data/mqm-spans-with-year-and-domain-but-no-news-2022.csv"
    mqm_dataset = MQMDataset(mqm_path)

    da_path = "RicardoRei/wmt-da-human-evaluation"
    da_dataset = load_dataset(da_path, split="train")
    # xcomet paper states they use DA data from 2017 to 2020
    da_dataset = da_dataset.filter(lambda example: 2017 <= example["year"] <= 2020)

    lowest = da_dataset.filter(lambda example: example["annotators"] > 1 and example["raw"] == 0)
    highest = da_dataset.filter(lambda example: example["annotators"] > 1 and example["raw"] == 100)
    low = np.mean(lowest["score"])
    high = np.mean(highest["score"])
    scaled_scores = ((np.array(da_dataset["score"]) - low) / (high - low)).tolist()

    da_dataset = da_dataset.remove_columns("score")
    da_dataset = da_dataset.add_column("score", scaled_scores)

    dataset_load_time = time.perf_counter() - start

    # For debugging purposes
    # mqm_dataset.data = mqm_dataset.data.sample(n=1000, random_state=11)
    # da_dataset = da_dataset.select(range(1000))

    if track_time:
        return mqm_dataset, da_dataset, dataset_load_time
    return mqm_dataset, da_dataset

def get_model(args, track_time):
    start = time.perf_counter()
    model = XCOMETMetric(
        encoder_learning_rate=args.encoder_lr,
        learning_rate=args.lr,
        layerwise_decay=args.layerwise_decay,
        encoder_model=args.encoder_model,
        pretrained_model=args.pretrained_model,
        word_layer=args.word_layer,
        validation_data=[],
        word_level_training=args.word_level,
        hidden_sizes=args.hidden_sizes,
        loss_lambda=args.loss_lambda_phase_two,
    )
    model_load_time = time.perf_counter() - start

    if track_time:
        return model, model_load_time
    return model

def train_one_epoch(model, optimizer, scheduler, train_loader, use_wandb, grad_accum_steps, device):
    model.train()
    losses = []

    for step, batch in enumerate(tqdm(train_loader, desc="training")):
        inputs_tuple, target = model.prepare_sample(batch)
        assert len(inputs_tuple) == 3, "Only support mode with (src, ref, full_input) as input for now (important during evaluation)"
        target.score = target.score.to(device)

        loss = 0
        
        # src + ref, src only, ref only
        for inputs in inputs_tuple:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            output = model(**inputs)

            # keep only logits corresponding to "mt" part of input, as we only predict error spans there
            if model.word_level:
                seq_len = target.mt_length.max()
                output.logits = output.logits[:, :seq_len]

            loss = loss + model.compute_loss(output, target)

        # Without this scaling we will have effective lr = lr * grad_accum_steps
        (loss / grad_accum_steps).backward()

        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())
            
            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                })
        
    return losses

@torch.inference_mode()
def evaluate_model(model, val_dataloader, prefix, device):
    input_types = ("src", "ref", "full")
    model.eval()

    true_scores = []
    n_samples = 0
    model_scores = defaultdict(list)
    tag_accuracy = defaultdict(float)

    for batch in tqdm(val_dataloader, desc="evaluation"):
        inputs_tuple, target = model.prepare_sample(batch)
        assert len(inputs_tuple) == 3, "Only support mode with (src, ref, full_input) as input for now"

        true_scores.append(target.score)
        n_samples += target.score.shape[0]
        
        target.score = target.score.to(device)
        target.labels = target.labels.to(device)

        # 3 input types: src only, ref only, full (src + ref)
        for inputs, input_type in zip(inputs_tuple, input_types):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            output = model(**inputs)

            # keep only logits corresponding to "mt" part of input, as we only predict error spans there
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
    mqm_dataset, da_dataset, dataset_load_time = get_datasets(args, track_time=True)
    for d, part in zip((mqm_dataset, da_dataset), ("mqm", "da")):
        print(part)
        print("N samples:", len(d))
        print("First sample:\n", d[0], "\n")

    mqm_dataloader = torch.utils.data.DataLoader(mqm_dataset, args.batch_size, shuffle=True, collate_fn=lambda x: x)
    da_dataloader = torch.utils.data.DataLoader(da_dataset, args.batch_size, shuffle=False, collate_fn=lambda x: x)

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
    losses = []
    print("Unfreezing encoder (but keeping embeddings frozen).")

    model.encoder.unfreeze()
    model.encoder.freeze_embeddings()

    train_start = time.perf_counter()

    # Phase 1
    model.word_level = False
    model.hparams.loss_lambda = 0

    for epoch in range(args.phase_one_epochs):
        losses.extend(train_one_epoch(model, optimizer, None, mqm_dataloader, args.use_wandb, args.grad_accum_steps, device))
        torch.save(model.state_dict(), output_path / "checkpoint.pth")
        np.save(output_path / "losses.npy", losses)

    # Phase 2
    model.word_level = True
    model.hparams.loss_lambda = args.loss_lambda_phase_two
    
    for epoch in range(args.phase_two_epochs):
        losses.extend(train_one_epoch(model, optimizer, None, mqm_dataloader, args.use_wandb, args.grad_accum_steps, device))
        torch.save(model.state_dict(), output_path / "checkpoint.pth")
        np.save(output_path / "losses.npy", losses)

    # Phase 3
    model.word_level = True
    model.hparams.loss_lambda = args.loss_lambda_phase_three

    for epoch in range(args.phase_three_epochs):
        losses.extend(train_one_epoch(model, optimizer, None, mqm_dataloader, args.use_wandb, args.grad_accum_steps, device))
        torch.save(model.state_dict(), output_path / "checkpoint.pth")
        np.save(output_path / "losses.npy", losses)

    train_time = time.perf_counter() - train_start
# Construct report
    peak_memory_mb = torch.cuda.max_memory_allocated() // 2 ** 20

    # train_metrics = []
    # train_metrics.append(evaluate_model(model, mqm_dataloader, "train_", device))
    # pd.DataFrame(train_metrics).to_csv(output_path / "train_metrics.csv", index=False)

    report = {
        "peak_memory_mb": peak_memory_mb,
        "dataset_load_time": round(dataset_load_time, 2),
        "model_load_time": round(model_load_time, 2),
        "train_time": round(train_time, 2),
        "mqm_dataset_length": len(mqm_dataset),
        "da_dataset_length": len(da_dataset),
    }
    #report = report | train_metrics[-1]
    #report = report | val_metrics[-1]
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