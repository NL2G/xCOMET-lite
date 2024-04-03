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

from utils import load_json, dump_json, load_tsv, LengthGroupedSampler, enable_gradient_checkpointing
from source.mqm_dataset import MQMDataset
from train import train_one_epoch, prepare_sample, compute_loss

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
    parser.add_argument("--do-finetune", action="store_true", help="Whether to tune the biases in the model to compensate for pruning")

    parser.add_argument("--model", default="Unbabel/XCOMET-XL", help="Which XCOMET model to load", required=True)

    return parser

def print_summary(report: dict):
    print("Dataset load time:", report.get("dataset_load_time"))
    print("Model load time:", report.get("model_load_time"))
    print("Prediction time:", report.get("prediction_time"), "\n")
    print("Max memory:", report.get("peak_memory_mb"), "Mb")
    print("Kendall correlation:", report.get("kendall_correlation"))
    print("Random Kendall correlation:", report.get("random_kendall_correlation"))


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


def structured_prune(model, fraction_to_prune: float):
    parameters_to_prune = [
        (module, "weight") for module in filter(lambda m: isinstance(m, nn.Linear), model.modules())
    ]

    for module, name in parameters_to_prune:
        prune.ln_structured(module, name=name, amount=fraction_to_prune, n=2, dim=1)

    for (module_, _ ) in parameters_to_prune:
        prune.remove(module_, 'weight')

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
    # Data
    train_dataset = load_dataset("RicardoRei/wmt-mqm-human-evaluation")["train"]
    train_dataset = train_dataset.filter(lambda example:
        not (example["year"] == args.year and example["domain"] == args.domain and example["lp"] == args.lp))
    train_dataset = train_dataset.shuffle(seed=11).select(range(16_000))

    train_batch_size = 16
    sampler = LengthGroupedSampler(
        batch_size=train_batch_size, dataset=train_dataset, model_input_name="src"
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, sampler=sampler, collate_fn=lambda x: x
    )

    # Prepare model
    def requires_grad_predicate(name):
        return name.endswith("bias") or \
            "layerwise_attention.scalar_parameters" in name or \
            "hidden2tag" in name or \
            "estimator" in name

    pruned_model = pruned_model.half()
    for name, param in pruned_model.named_parameters():
        param.requires_grad = requires_grad_predicate(name)
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    optimizer = torch.optim.AdamW([p for p in pruned_model.parameters() if p.requires_grad], lr=1e-5)
    print(f"Total parameters: {sum(p.numel() for p in pruned_model.parameters()) / 1e6:0.2f} M")
    print(f"Trained parameters: {sum(p.numel() for p in pruned_model.parameters() if p.requires_grad) / 1e6:0.2f} M")

    pruned_model.to(device)
    enable_gradient_checkpointing(pruned_model)

    # Finetune
    losses = train_one_epoch(pruned_model, optimizer, None, train_dataloader, use_wandb=False,
        grad_accum_steps=1, device=device, scaler=GradScaler())
    
    # Save trained parameters
    finetune_output_path = Path(args.output) / "training"
    finetune_output_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: param for name, param in pruned_model.state_dict().items() if requires_grad_predicate(name)},
        finetune_output_path / "tuned_params.pth"
    )
    np.save(finetune_output_path / "losses.npy", losses)
    plt.plot(losses)
    plt.title(args.output)
    plt.savefig(finetune_output_path / "losses.png")

def load_pruned_tuned_model(args):
    start = time.perf_counter()
    model = get_model(args, track_time=False)

    prune_layers(model, args.n_layers_to_prune)
    
    finetune_output_path = Path(args.output) / "training"
    model.load_state_dict(torch.load(finetune_output_path / "tuned_params.pth"), strict=False)

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
    output_path = Path(args.output) / "evaluations" / ("no_reference" if args.dataset.endswith(".tsv") else "with_reference") / args.lp

    if os.path.exists(output_path) and args.do_finetune:
        print("Reusing previous results. Change output folder or delete this folder to recompute.")
        return

# Start logic

# Create directories
    os.makedirs(output_path, exist_ok=True)

# Data
    dataset, ground_truth, dataset_load_time = get_dataset(args, track_time=True)

# channel pruning
    # def count_zeros_fraction(model):
    #     n_parameters = sum(p.numel() for p in model.parameters())
    #     # There is a lot of parameters in embeddings, which are not sparsified
    #     n_parameters = n_parameters - model.encoder.model.embeddings.word_embeddings.weight.numel()
    #     n_zeros = sum(torch.isclose(p, torch.zeros_like(p)).sum() for p in model.parameters())
    #     return n_zeros / n_parameters

    # print(f"Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    # print(f"Original\nSparsity: {count_zeros_fraction(model):.3f}\n")
    # structured_prune(model, 0.6)
    # print("="*70)
    # print(f"Pruned\nSparsity: {count_zeros_fraction(model):.3f}\n")
  
# Fancy layer pruning
    
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
    random_kendall_corr = kendalltau(ground_truth, np.random.rand(len(ground_truth)))

    report = {
        "kendall_correlation": kendall_corr[0],
        "kendall_p_value": kendall_corr[1],
        "random_kendall_correlation": random_kendall_corr[0],
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