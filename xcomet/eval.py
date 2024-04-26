import os
import time
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from datasets import load_dataset
from scipy.stats import kendalltau
from comet import download_model, load_from_checkpoint

from inference.utils import (
    load_json, dump_json, load_tsv,
    find_max_bs
)

from onnx_wrapper.xcomet import OnnxXCOMETMetric, OnnxXCOMETModel
from onnx_wrapper.utils import xcomet_to_onnx
from inference.utils import logger, get_memory_allocated


DEVICE = torch.device('cuda:0')


def make_parser():
    parser = ArgumentParser(description="xCOMET general pipeline evaluation.")
    parser.add_argument("-o", "--output", help="Where to save results, in format root_results_directory/experiment_name", required=True)
    parser.add_argument("--model", help="Which model to use (name on huggingface)", required=True)
    parser.add_argument("--onnx_path", help="Directory with ONNX .onnx file. If doesn't exist, create it")
    parser.add_argument("--lp", help="On which language pair to compute metrics", required=True)
    parser.add_argument("--dataset", help="Which dataset to use (huggingface dataset/path to tsv file)", required=True)
    parser.add_argument("--domain", default="news", help="On which domain to compute metrics")
    parser.add_argument("--year", type=int, default=2022, help="In which year to compute metrics")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to fix")
    parser.add_argument("--gpu", action="store_true", help="Neither use GPU or CPU. If GPU, use 0-th device - set CUDA_VISIBLE_DEVICES")
    parser.add_argument("--half", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--batch-size", type=int, default=8, help="Fixed inference batch size")

    return parser


def get_dataset(args):
    logger.info("Loading dataset...")
    start = time.perf_counter()

    if args.dataset.endswith(".tsv"):
        logger.info(f"Ignoring arguments domain={args.domain}, year={args.year} and lp={args.lp} -- not implemented for local .tsv datasets.")
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
    logger.info(f"N samples: {len(dataset)}")
    logger.info(f"First sample:\n{dataset[0]}\n")

    return dataset, ground_truth, dataset_load_time

def get_model(args):
    logger.info("Loading model...")
    start = time.perf_counter()
    model_path = args.model
    if args.model.startswith('Unbabel/'):
        model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)
    if args.half:
        assert args.onnx_path is None
        model = model.half()
    model.eval()
    if args.onnx_path is not None:
        logger.info("Loading ONNX model...")
        if not os.path.exists(args.onnx_path):
            xcomet_to_onnx(model, args.onnx_path)
        model = OnnxXCOMETMetric(
            OnnxXCOMETModel(args.onnx_path, model, use_gpu=args.gpu)
        )
    model_load_time = time.perf_counter() - start
    return model, model_load_time

def get_batch_size(model, args):
    batch_size = args.batch_size
    if batch_size > 0:
        return batch_size, 0
    logger.info(f"Searching for best batch size for {model.__class__.__name__}")
    start = time.perf_counter()
    torch_model = model
    if isinstance(model, OnnxXCOMETMetric):
        torch_model = model.model.xcomet_model
    batch_size, _, _ = find_max_bs(model, len(torch_model.encoder.tokenizer.vocab), DEVICE)
    batch_size_time = time.perf_counter() - start
    return batch_size, batch_size_time

def get_number_of_points(dataset):
    coef = 1
    if 'ref' in dataset[0]:
        coef = 3
    return len(dataset) * coef

def run_metric(model, dataset, batch_size, args):
    logger.info("Computing metric...")
    start = time.perf_counter()
    model_output = model.predict(dataset, batch_size=batch_size, gpus=int(args.gpu))
    prediction_time = time.perf_counter() - start
    return model_output, prediction_time

@torch.inference_mode()
def main():
# Get arguments
    parser = make_parser()
    args = parser.parse_args()
    logger.info(args)

# Setup environment
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Start logic
    output_path = Path(args.output) / args.lp

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

        dataset, ground_truth, dataset_load_time = get_dataset(args)

        model, model_load_time = get_model(args)

        batch_size, batch_size_time = get_batch_size(model, args)

        model_output, prediction_time = run_metric(model, dataset, batch_size, args)

        segment_scores = np.array(model_output.scores)
# Construct report
        peak_memory_mb = get_memory_allocated(DEVICE, model, is_max=True) // 2 ** 20
        throughput = get_number_of_points(dataset) / prediction_time
        kendall_corr = kendalltau(ground_truth, segment_scores)

        report = {
            "kendall_correlation": kendall_corr[0],
            "kendall_p_value": kendall_corr[1],
            "peak_memory_mb": peak_memory_mb,
            "samples_per_second": throughput,
            "system_level_score": model_output.system_score,
            "dataset_load_time": round(dataset_load_time, 2),
            "model_load_time": round(model_load_time, 2),
            "prediction_time": round(prediction_time, 2),
            "batch_size_time": round(batch_size_time, 2),
            "dataset_length": get_number_of_points(dataset),
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

    else:
        logger.info("Reusing previous results. Change output folder or delete this folder to recompute.")
        segment_scores = np.load(output_path / "model_segment_level_scores.npy")
        error_spans = load_json(output_path / "error_spans.json")
        report = load_json(output_path / "report.json")

    logger.info(f"Dataset load time: {report['dataset_load_time']}")
    logger.info(f"Model load time: {report['model_load_time']}")
    logger.info(f"Prediction time: {report['prediction_time']}\n")
    logger.info(f"Samples per second: {report['samples_per_second']}")
    logger.info(f"Max memory: {report['peak_memory_mb']} Mb")
    logger.info(f"Kendall correlation: {report['kendall_correlation']}")

if __name__ == "__main__":
    main()