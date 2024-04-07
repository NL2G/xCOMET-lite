import os
import time
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Tuple
from argparse import ArgumentParser
from collections import defaultdict
from deberta_encoder import DeBERTaEncoder
import comet.encoders

comet.encoders.str2encoder["DeBERTa"] = DeBERTaEncoder

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import comet
from torch.cuda.amp import GradScaler
from datasets import load_dataset

from source.mqm_dataset import MQMDataset
from train import train_one_epoch, prepare_sample, compute_loss
from utils import load_json, dump_json, load_tsv, enable_gradient_checkpointing, CosineAnnealingLRWarmup

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
    parser.add_argument("--n-layers-to-prune", type=int, help="Amount of layers to prune", required=True)
    parser.add_argument("--do-finetune", action="store_true", help="If chosen, script only prunes and finetunes the model; otherwise it only evaluates pruned model.")

    parser.add_argument("--model", default="Unbabel/XCOMET-XL", help="Which XCOMET model to load", required=True)

    return parser

def print_summary(report: dict):
    print("Dataset load time:", report.get("dataset_load_time"))
    print("Model load time:", report.get("model_load_time"))
    print("Prediction time:", report.get("prediction_time"), "\n")
    print("Max memory:", report.get("peak_memory_mb"), "Mb")
    print("Kendall correlation:", report.get("kendall_correlation"))


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
    model_path = comet.download_model(args.model)
    model = comet.load_from_checkpoint(model_path)

    model_load_time = time.perf_counter() - start

    if track_time:
        return model, model_load_time
    return model


def prune_layers(model, n_layers_to_prune: int, new_word_layer: Optional[int] = None):
    """Implements simple layer pruning heuristic as described in https://arxiv.org/abs/2403.17887v1.
    Prunes n layers, starting from a penultimate layer.
    """
    model.encoder.model.encoder.layer = model.encoder.model.encoder.layer[:-(1 + n_layers_to_prune)] + \
        model.encoder.model.encoder.layer[-1:]
    model.encoder.model.config.num_hidden_layers = model.encoder.model.config.num_hidden_layers - n_layers_to_prune

    pruned_layerwise_attention = comet.modules.LayerwiseAttention(
        num_layers=model.encoder.num_layers,
        dropout=model.hparams.dropout,
        layer_norm=model.hparams.layer_norm
    )
    pruned_layerwise_attention.scalar_parameters = model.layerwise_attention.scalar_parameters[:-1-n_layers_to_prune]\
        .append(model.layerwise_attention.scalar_parameters[-1])
    model.layerwise_attention = pruned_layerwise_attention

    model.hparams.word_layer = new_word_layer if new_word_layer is not None else len(model.encoder.model.encoder.layer)
    
    return model


def finetune(pruned_model, args, device):
    """Finetunes the model on WMT MQM evaluation dataset (news 2022 excluded).
    """
    # Data
    finetune_data_path = "data/mqm-spans-with-year-and-domain-but-no-news-2022.csv"
    train_dataset = MQMDataset(finetune_data_path)

    train_batch_size = 8
    grad_accum_steps = 16
    n_epochs = 3
    warmup_fraction = 0.1
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, collate_fn=lambda x: x, shuffle=True
    )

    # Prepare model
    def requires_grad_predicate(name):
        return "layerwise_attention.scalar_parameters" in name or \
            "estimator" in name or \
            "LayerNorm" in name or \
            name.endswith("bias")

    for name, param in pruned_model.named_parameters():
        param.requires_grad = requires_grad_predicate(name)
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    print(f"Total parameters: {sum(p.numel() for p in pruned_model.parameters()) / 1e6:0.2f} M")
    print(f"Trained parameters: {sum(p.numel() for p in pruned_model.parameters() if p.requires_grad) / 1e6:0.2f} M")

    optimizer = torch.optim.AdamW([p for p in pruned_model.parameters() if p.requires_grad], lr=1e-4)

    scheduler = CosineAnnealingLRWarmup(
        optimizer,
        T_max=len(train_dataloader) * n_epochs // grad_accum_steps,
        T_warmup=len(train_dataloader) * n_epochs * warmup_fraction // grad_accum_steps
    )

    scaler = GradScaler()

    pruned_model.to(device)
    enable_gradient_checkpointing(pruned_model)

    pruned_model.word_level = False
    pruned_model.hparams.loss_lambda = 0

    finetune_output_path = Path(args.output) / "training"
    finetune_output_path.mkdir(parents=True, exist_ok=True)

    # Finetune
    losses = []
    for epoch in range(n_epochs):
        losses.extend(
            train_one_epoch(pruned_model, optimizer, scheduler, train_dataloader, use_wandb=False,
                grad_accum_steps=grad_accum_steps, device=device, scaler=scaler)
        )

        # Save trained parameters
        torch.save(
            {name: param.detach().cpu() for name, param in pruned_model.named_parameters() if param.requires_grad},
            finetune_output_path / f"tuned_params_{epoch + 1}.pth"
        )
        np.save(finetune_output_path / "losses.npy", losses)
        plt.plot(losses)
        plt.title(args.output)
        plt.savefig(finetune_output_path / "losses.png")

def load_pruned_tuned_model(args):
    """Loads a model, prunes the layers, and loads finetuned parameters if there is a corresponding checkpoint.
    """
    start = time.perf_counter()
    model = get_model(args, track_time=False)

    prune_layers(model, args.n_layers_to_prune)
    
    finetune_output_path = Path(args.output) / "training"
    if finetune_output_path.exists():
        # _3 is hardcoded, assuming n_epoch=3 during finetuning
        tuned_params = torch.load(finetune_output_path / "tuned_params_3.pth")
        print(f"N tuned params groups found: {len(tuned_params)}")
        print(f"N tuned params found: {sum(p.numel() for p in tuned_params.values())}")
        model.load_state_dict(tuned_params, strict=False)

    return model.half(), time.perf_counter() - start

@torch.inference_mode()
def run_metric(model, dataset, args):
    print("Computing metric...")
    start = time.perf_counter()
    model_output = model.predict(dataset, batch_size=args.batch_size, gpus=args.n_gpus)
    prediction_time = time.perf_counter() - start

    return model_output, prediction_time

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
    if args.do_finetune:
        output_path = Path(args.output) / "training"
    else:
        output_path = Path(args.output) / "evaluations" / ("no_reference" if args.dataset.endswith(".tsv") else "with_reference") / args.lp
    
    if output_path.exists():
        print("Reusing previous results. Change output folder or delete this folder to recompute.")
        return

# Start logic

# Create directories
    os.makedirs(output_path, exist_ok=True)

# Data
    dataset, ground_truth, dataset_load_time = get_dataset(args, track_time=True)

# Layer pruning
    if args.do_finetune:
        model, model_load_time = get_model(args, track_time=True)
        prune_layers(model, args.n_layers_to_prune)

        finetune(model, args, device="cuda:0")

        print("Finetuning complete.")
        return

# Run evaluation
    model, model_load_time = load_pruned_tuned_model(args)
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