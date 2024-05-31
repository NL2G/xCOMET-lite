import os
import time
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from datasets import load_dataset
from scipy.stats import kendalltau
from comet import download_model, load_from_checkpoint
from optimum.gptq import GPTQQuantizer, load_quantized_model

from utils import load_json, dump_json, load_tsv, print_summary, is_oom_exception

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
    parser.add_argument("--tune-batch-size", action="store_true", help="Whether to tune evaluation batch size to find maximal possible speed & throughput")

    return parser

def get_dataset(dataset, domain, year, lp):
    print("Loading dataset...")
    start = time.perf_counter()

    if dataset.endswith(".tsv"):
        print(f"Ignoring arguments domain={domain}, year={year} and lp={lp} -- not implemented for local .tsv datasets.")
        dataset = load_tsv(dataset)
        ground_truth = dataset["score"]
        dataset = list(dataset.T.to_dict().values())
    else:
        dataset = load_dataset(dataset, split="train")
        dataset = dataset.filter(lambda example: 
            example["year"] == year and example["domain"] == domain and example["lp"] == lp)
        ground_truth = dataset["score"]
        dataset = [sample for sample in dataset]

    dataset_load_time = time.perf_counter() - start
    print("N samples:", len(dataset))
    print("First sample:\n", dataset[0], "\n")

    return dataset, ground_truth, dataset_load_time

def get_model(model_name):
    print("Loading model...")
    start = time.perf_counter()
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path).half()
    model_load_time = time.perf_counter() - start

    return model, model_load_time

def quantize_model(model, nbits, calibration_dataset):
    print("Quantizing model...")
    start = time.perf_counter()
    # By default calibrates on c4 dataset, probably can do better with domain-specific dataset
    quantizer = GPTQQuantizer(bits=nbits, dataset=calibration_dataset, block_name_to_quantize = "encoder.layer", model_seqlen = 512)
    model.encoder.model = quantizer.quantize_model(model.encoder.model, model.encoder.tokenizer)
    quantization_time = time.perf_counter() - start

    return model, quantization_time

@torch.inference_mode()
def run_metric(model, dataset, batch_size, n_gpus):
    print("Computing metric...")
    start = time.perf_counter()
    model_output = model.predict(dataset, batch_size=batch_size, gpus=n_gpus)
    torch.cuda.synchronize()
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

# Start logic
    output_path = Path(args.output) / args.lp

    if os.path.exists(output_path):
        print("Reusing previous results. Change output folder or delete this folder to recompute.")
        report = load_json(output_path / "report.json")
        print_summary(report)
        return

    os.makedirs(output_path, exist_ok=True)

    dataset, ground_truth, dataset_load_time = get_dataset(args.dataset, args.domain, args.year, args.lp)

    model, model_load_time = get_model(args.model)

    model, quantization_time = quantize_model(model, args.nbits, args.calibration_dataset)

    model_output, prediction_time = run_metric(model, dataset, args.batch_size, args.n_gpus)

    prev_batch_size = args.batch_size
    if args.tune_batch_size:
        batch_to_time = {prev_batch_size: prediction_time}
        new_batch_size = prev_batch_size * 2

        while new_batch_size > prev_batch_size:
            try:
                print(f"Trying batch size {new_batch_size}")
                model_output, prediction_time = run_metric(model, dataset, new_batch_size, args.n_gpus)
                batch_to_time[new_batch_size] = prediction_time
                prev_batch_size = new_batch_size
                new_batch_size = prev_batch_size * 2
            except RuntimeError as error:
                if is_oom_exception(error):
                    new_batch_size = (new_batch_size + prev_batch_size) // 2
                else:
                    raise error


    segment_scores = np.array(model_output.scores)
# Construct report
    peak_memory_mb = torch.cuda.max_memory_allocated() // 2 ** 20
    kendall_corr = kendalltau(ground_truth, segment_scores)

    prediction_time = min(batch_to_time.values()) if args.tune_batch_size else prediction_time

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
        "samples_per_second": round(len(dataset) / prediction_time, 2),
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
    print_summary(report)

if __name__ == "__main__":
    main()