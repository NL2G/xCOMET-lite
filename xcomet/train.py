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
from deberta_encoder import DeBERTaEncoder
import comet.encoders

comet.encoders.str2encoder["DeBERTa"] = DeBERTaEncoder
from comet.models.multitask.xcomet_metric import XCOMETMetric

from utils import load_json, dump_json, LengthGroupedSampler, MQMDataset

from rich.logging import RichHandler
from rich.console import Console
import logging
from torch import autocast
from torch.cuda.amp import GradScaler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            console=Console(color_system=None)
        )
    ],
)

logger = logging.getLogger(__name__)

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


def parse_args():
    parser = ArgumentParser(description="MQM finetuning.")
    parser.add_argument(
        "-o",
        "--output",
        help="Where to save results, in format root_results_directory/experiment_name",
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed to fix")
    parser.add_argument("--n-gpus", type=int, default=1, help="Amount of GPUs utilized")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (equals effective batch size when grad-accum-steps=1)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Number of batches processed before weights update",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of passes through the train set",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="xcomet-compression",
        help="The name of project in wandb",
    )
    parser.add_argument("--train-dataset", help="Dataset to train on", required=True)
    parser.add_argument(
        "--val-dataset",
        default="data/mqm-spans-with-year-and-domain-but-no-news-2022.csv",
        help="Dataset for validation",
    )

    parser.add_argument(
        "--nr-frozen-epochs",
        type=float,
        default=0.9,
        help="Number of epochs (%% of epoch) that the encoder is frozen",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-06,
        help="Base learning rate for the encoder part",
    )
    parser.add_argument(
        "--lr", type=float, default=3.66e-06, help="Learning rate for top layers"
    )
    parser.add_argument(
        "--layerwise-decay",
        type=float,
        default=0.983,
        help="Learning rate decay from last to first encoder layers",
    )
    parser.add_argument(
        "--encoder-model",
        default="MiniLM",
        help="Backbone family [BERT, XLM-RoBERTa, MiniLM, XLM-RoBERTa-XL, RemBERT]",
    )
    parser.add_argument(
        "--pretrained-model",
        default="microsoft/Multilingual-MiniLM-L12-H384",
        help="Concrete pretrained checkpoint for the backbone (huggingface name)",
    )
    parser.add_argument(
        "--word-layer",
        type=int,
        default=8,
        help="From which layer of encoder to predict word tags",
    )
    parser.add_argument(
        "--word-level",
        type=bool,
        default=True,
        help="Whether to use word-level annotations",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=(3072, 1024),
        help="Size of hidden layers used in regression head",
    )
    parser.add_argument(
        "--loss-lambda",
        type=float,
        default=0.055,
        help="Weight assigned to the word-level loss",
    )
    parser.add_argument(
        "--subsample-pct",
        type=float,
        default=1.0,
        help="Percentage of samples to use from the dataset",
    )

    return parser.parse_args()


def print_summary(report: dict):
    logger.info("=" * 70)
    logger.info(f'Dataset load time: {report["dataset_load_time"]}')
    logger.info(f'Model load time: {report["model_load_time"]}')
    logger.info(f'Train time: {report["train_time"]}\n')
    logger.info(f'Max memory: {report["peak_memory_mb"]} Mb')
    logger.info(f'Validation full kendall correlation: {report["val_full_kendall_correlation"]}')


# Option A: hardcode error-span dataset, hardcode splits into train/val/test
# Option B: Make distinct functions and argument sets for train, val and test


def get_datasets(args, track_time):
    start = time.perf_counter()

    train_dataset = load_dataset(args.train_dataset)["train"]
    

    if ".csv" in args.val_dataset:
        val_dataset = MQMDataset(args.val_dataset)
    elif args.val_dataset.endswith(".tsv"):
        raise ValueError(".tsv is not supported yet")
    else:
        # Assumes it is a huggingface dataset
        val_dataset = load_dataset(args.val_dataset)

    # For debugging purposes
    # train_dataset.data = train_dataset.data.sample(n=1000, random_state=11)
    # train_dataset = train_dataset.select(range(1000))
    # val_dataset.data = val_dataset.data.sample(n=1000, random_state=11)

    dataset_load_time = time.perf_counter() - start

    if track_time:
        return train_dataset, val_dataset, dataset_load_time
    return train_dataset, val_dataset


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
        loss_lambda=args.loss_lambda,
    )
    model_load_time = time.perf_counter() - start

    if track_time:
        return model, model_load_time
    return model


def prepare_sample(model, batch):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.prepare_sample(batch)

    return model.prepare_sample(batch)


def compute_loss(model, output, target):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.compute_loss(output, target)

    return model.compute_loss(output, target)


def train_one_epoch(
    model, optimizer, scheduler, train_dataloader, use_wandb, grad_accum_steps, device, scaler: GradScaler
):
    model.train()
    losses = []

    for step, batch in enumerate(tqdm(train_dataloader, desc="training")):
        inputs_tuple, target = prepare_sample(model, batch)
        assert (
            len(inputs_tuple) == 3
        ), "Only support mode with (src, ref, full_input) as input for now (important during evaluation)"
        target.score = target.score.to(device)

        loss = 0

        # src + ref, src only, ref only
        for inputs in inputs_tuple:
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(**inputs)

                if model.word_level:
                    # keep only logits corresponding to "mt" part of input, as we only predict error spans there
                    seq_len = target.mt_length.max()
                    output.logits = output.logits[:, :seq_len]

                loss = loss + compute_loss(model, output, target)

        # Without this scaling we will have effective lr = lr * grad_accum_steps
        if scaler is not None:
            scaler.scale((loss / grad_accum_steps)).backward()
        else:
            (loss / grad_accum_steps).backward()

        if step % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            losses.append(loss.detach())

            if use_wandb:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                    }
                )

    return [loss.item() for loss in losses]


@torch.inference_mode()
def evaluate_model(model, val_dataloader, prefix, device):
    input_types = ("src", "ref", "full")
    model.eval()

    true_scores = []
    n_samples = 0
    model_scores = defaultdict(list)
    tag_accuracy = defaultdict(float)

    for batch in tqdm(val_dataloader, desc="evaluation"):
        inputs_tuple, target = prepare_sample(model, batch)
        assert (
            len(inputs_tuple) == 3
        ), "Only support mode with (src, ref, full_input) as input for now"

        true_scores.append(target.score)
        n_samples += target.score.shape[0]

        target.score = target.score.to(device)
        target.labels = target.labels.to(device)

        # 3 input types: src only, ref only, full (src + ref)
        for inputs, input_type in zip(inputs_tuple, input_types):
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            output = model(**inputs)

            # keep only logits corresponding to "mt" part of input, as we only predict error spans there
            seq_len = target.mt_length.max()
            output.logits = output.logits[:, :seq_len]

            model_scores[input_type].append(output.score.detach().cpu())
            tag_accuracy[input_type] += (
                output.logits.argmax(-1) == target.labels
            ).float().mean() * target.score.shape[0]

    true_scores = torch.cat(true_scores).numpy()
    model_scores = {k: torch.cat(v).numpy() for k, v in model_scores.items()}
    kendall_results = {k: kendalltau(true_scores, model_scores[k]) for k in input_types}
    tag_accuracy = {k: v / n_samples for k, v in tag_accuracy.items()}

    metrics = {
        f"{prefix}{k}_kendall_correlation": kendall_results[k][0] for k in input_types
    }

    metrics = metrics | {
        f"{prefix}{k}_tag_accuracy": tag_accuracy[k].item() for k in input_types
    }

    logger.info(str(metrics))

    return metrics


def main():
    # Get arguments
    args = parse_args()
    logger.info(f'Running with args: {args}')

    # Setup environment
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Check for earlier launches
    output_path = Path(args.output) / "training"

    if os.path.exists(output_path):
        logger.warning(
            "Reusing previous results. Change output folder or delete this folder to recompute."
        )
        return

    # Start logic

    # Create directories
    os.makedirs(output_path, exist_ok=True)

    # Data
    logger.info("Loading datasets...")
    train_dataset, val_dataset, dataset_load_time = get_datasets(args, track_time=True)
    if args.subsample_pct < 1.0:
        logger.info(f"Subsampling train dataset to {args.subsample_pct * 100:.2f}%")
        len_train = len(train_dataset)
        subsampled_len = int(len_train * args.subsample_pct)
        logger.info(f"Original train dataset length: {len_train}, subsampled length: {subsampled_len}")
        train_dataset = train_dataset.select(range(subsampled_len))

    for d, part in zip((train_dataset, val_dataset), ("train", "val")):
        logger.info(str(part))
        logger.info(f"N samples: {len(d) if d is not None else 0}")
        logger.info(f"First sample:\n{d[0] if d is not None else None}\n")

    sampler = LengthGroupedSampler(
        batch_size=args.batch_size, dataset=train_dataset, model_input_name="src"
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda x: x
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=lambda x: x)

    # Model
    device = "cuda:0"
    model, model_load_time = get_model(args, track_time=True)

    # Train loop
    optimizers, schedulers = model.configure_optimizers()
    assert len(schedulers) == 0, len(optimizers) == 1
    optimizer = optimizers[0]
    scaler = GradScaler()

    if args.use_wandb:
        wandb.login()
        wandb.init(
            project=args.wandb_project_name,
            name=args.output.split("/")[-1],
            config=vars(args),
        )

    # val_metrics = []
    losses = []

    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    train_start = time.perf_counter()
    val_metrics = []
    for epoch in range(args.n_epochs):
        if epoch >= args.n_epochs * args.nr_frozen_epochs:
            logger.info("Unfreezing encoder (but keeping embeddings frozen).")
            if args.n_gpus == 1:
                model.encoder.unfreeze()
                model.encoder.freeze_embeddings()
            else:
                model.module.encoder.unfreeze()
                model.module.encoder.freeze_embeddings()

        losses.extend(
            train_one_epoch(
                model,
                optimizer,
                None,
                train_dataloader,
                args.use_wandb,
                args.grad_accum_steps,
                device,
                scaler,
            )
        )
        torch.save(model.state_dict(), output_path / "checkpoint.pth")
        np.save(output_path / "losses.npy", losses)

        val_metrics.append(evaluate_model(model, val_dataloader, "val_", device))
        pd.DataFrame(val_metrics).to_csv(output_path / "val_metrics.csv", index=False)

        if args.use_wandb:
            wandb.log(val_metrics[-1])

    train_time = time.perf_counter() - train_start
    # Construct report
    peak_memory_mb = torch.cuda.max_memory_allocated() // 2**20

    report = {
        "peak_memory_mb": peak_memory_mb,
        "dataset_load_time": round(dataset_load_time, 2),
        "model_load_time": round(model_load_time, 2),
        "train_time": round(train_time, 2),
        "train_dataset_length": len(train_dataset),
        "val_dataset_length": len(val_dataset),
    }
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
