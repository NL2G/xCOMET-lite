import os
import time
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import kendalltau
from comet import download_model, load_from_checkpoint
from optimum.gptq import GPTQQuantizer, load_quantized_model

from utils import load_json, dump_json

def make_parser():
    parser = ArgumentParser(description="xCOMET evaluation.")
    parser.add_argument("-o", "--output", help="Where to save results, in format root_results_directory/experiment_name", required=True)
    parser.add_argument("--model", help="Which model to use (name on huggingface)", required=True)
    parser.add_argument("--lp", help="On which language pair to compute metrics", required=True)
    parser.add_argument("--dataset", help="Which dataset to use (huggingface dataset/path to tsv file)", required=True)
    parser.add_argument("--domain", default="news", help="On which domain to compute metrics")
    parser.add_argument("--year", type=int, default=2022, help="In which year to compute metrics")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to fix")
    parser.add_argument("--n-gpus", type=int, default=1, help="Amount of GPUs utilized")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--nbits", type=int, default=8, help="N bits to quantize to")
    parser.add_argument("--calibration-dataset", default="c4", help="Calibration dataset for GPTQ algorithm")

    return parser

def load_tsv(path):
    data = pd.read_csv(path, sep="\t")
    data.index = np.arange(len(data))
    data = data.drop(columns=["Unnamed: 0"])
    return data

def get_dataset(args):
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

    return dataset, ground_truth, dataset_load_time

def get_model(args):
    print("Loading model...")
    start = time.perf_counter()
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path).half()
    model_load_time = time.perf_counter() - start

    return model, model_load_time

def quantize_model(model, args):
    print("Quantizing model...")
    start = time.perf_counter()
    # By default calibrates on c4 dataset, probably can do better with domain-specific dataset
    quantizer = GPTQQuantizer(bits=args.nbits, dataset=args.calibration_dataset, block_name_to_quantize = "encoder.layer", model_seqlen = 512)
    model.encoder.model = quantizer.quantize_model(model.encoder.model, model.encoder.tokenizer)
    quantization_time = time.perf_counter() - start

    return model, quantization_time

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

# Start logic
    output_path = Path(args.output) / args.lp

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

        dataset, ground_truth, dataset_load_time = get_dataset(args)

        model, model_load_time = get_model(args)

        model, quantization_time = quantize_model(model, args)

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

    else:
        print("Reusing previous results. Change output folder or delete this folder to recompute.")
        segment_scores = np.load(output_path / "model_segment_level_scores.npy")
        error_spans = load_json(output_path / "error_spans.json")
        report = load_json(output_path / "report.json")

    print("Dataset load time:", report["dataset_load_time"])
    print("Model load time:", report["model_load_time"])
    print("Prediction time:", report["prediction_time"], "\n")
    print("Max memory:", report["peak_memory_mb"], "Mb")
    print("Kendall correlation:", report["kendall_correlation"])

if __name__ == "__main__":
    main()