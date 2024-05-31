import os
import time
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional
from deberta_encoder import DeBERTaEncoder
import comet.encoders

comet.encoders.str2encoder["DeBERTa"] = DeBERTaEncoder

import torch
import numpy as np
import comet
from datasets import load_dataset
from scipy.stats import kendalltau
from comet import download_model, load_from_checkpoint
from optimum.gptq import GPTQQuantizer
from comet.models.multitask.xcomet_metric import XCOMETMetric


from inference.utils import (
    load_json, dump_json, load_tsv,
    find_max_bs
)

from onnx_wrapper.xcomet import OnnxXCOMETMetric, OnnxXCOMETModel
from onnx_wrapper.utils import xcomet_to_onnx
from inference.utils import logger, get_memory_allocated


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
    parser.add_argument("--gpu", action="store_true", help="Either use GPU or CPU. If GPU, use 0-th device - set CUDA_VISIBLE_DEVICES")
    parser.add_argument("--trt", action="store_true", help="Either use TRT or general CUDA. Requires passing GPU flag")
    parser.add_argument("--half", action="store_true", default=False, help="Use fp16 precision")
    parser.add_argument("--batch-size", type=int, default=8, help="Fixed inference batch size. If set to 0, script automatically finds largest batch, which is a power of 2 and fits into current device.")
    parser.add_argument("--prune-n-layers", type=int, default=0, help="How many layers to prune")
    parser.add_argument("--quantize-n-bits", type=int, default=0, help="Quantize model into N bits. Available options are 8, 4, 3, and 2 bits. 0 means no quantization.")

    return parser

def print_summary(logger, report):
    logger.info(f"Dataset load time: {report['dataset_load_time']}")
    logger.info(f"Model load time: {report['model_load_time']}")
    logger.info(f"Prediction time: {report['prediction_time']}\n")
    logger.info(f"Samples per second: {report['samples_per_second']}")
    logger.info(f"Max memory: {report['peak_memory_mb']} Mb")
    logger.info(f"Kendall correlation: {report['kendall_correlation']}")

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

def prune_layers(model, n_layers_to_prune: int, new_word_layer: Optional[int] = None):
    """Implements simple layer pruning heuristic as described in https://arxiv.org/abs/2403.17887v1.
    Prunes n layers, starting from a penultimate layer.
    """
    print(f"Pruning {n_layers_to_prune} layers...")
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

def quantize_model(model, nbits, calibration_dataset="wikitext2"):
    print("Quantizing model...")
    start = time.perf_counter()
    # By default calibrates on c4 dataset, probably can do better with domain-specific dataset
    # NOTE: due to this issue https://github.com/huggingface/transformers/issues/28490 we switch to wikitext2
    quantizer = GPTQQuantizer(bits=nbits, dataset=calibration_dataset, block_name_to_quantize = "encoder.layer", model_seqlen = 512)
    model.encoder.model = quantizer.quantize_model(model.encoder.model, model.encoder.tokenizer)
    quantization_time = time.perf_counter() - start

    return model, quantization_time

def get_model(args, device):
    logger.info("Loading model...")
    start = time.perf_counter()

    model_path = args.model
    if model_path == "mdeberta":
        model = XCOMETMetric(
            encoder_model='DeBERTa',
            pretrained_model='microsoft/mdeberta-v3-base',
            word_layer=8,
            validation_data=[],
            word_level_training=True,
            hidden_sizes=[
                3072,
                1024
            ],
            load_pretrained_weights=False,
        )
    else:
        if args.model.startswith('Unbabel/'):
            model_path = download_model(args.model)
        model = load_from_checkpoint(model_path)

    if args.prune_n_layers > 0:
        model = prune_layers(model, args.prune_n_layers)

    if args.half:
        assert args.onnx_path is None
        model = model.half()
    elif args.quantize_n_bits > 0:
        assert not args.half, "At most one of --half and --quantize-n-bits must be specified"
        available_bitwidths = (2, 3, 4, 8)
        assert args.quantize_n_bits in available_bitwidths, f"Can only quantize into {available_bitwidths} bits"
        #model.to(device)
        model, _ = quantize_model(model, args.quantize_n_bits)

    model.eval()

    if args.onnx_path is not None:
        logger.info("Loading ONNX model...")
        if not os.path.exists(args.onnx_path):
            xcomet_to_onnx(model, args.onnx_path)
        model = OnnxXCOMETMetric(
            OnnxXCOMETModel(args.onnx_path, model, use_gpu=args.gpu, use_trt=args.trt)
        )
    model_load_time = time.perf_counter() - start
    return model, model_load_time

def get_batch_size(model, args, device):
    batch_size = args.batch_size
    if batch_size > 0:
        return batch_size, 0
    logger.info(f"Searching for best batch size for {model.__class__.__name__}")
    start = time.perf_counter()
    torch_model = model
    if isinstance(model, OnnxXCOMETMetric):
        torch_model = model.model.xcomet_model
    batch_size, _, _ = find_max_bs(model, len(torch_model.encoder.tokenizer.vocab), device)
    batch_size_time = time.perf_counter() - start
    return batch_size, batch_size_time

def get_number_of_points(dataset):
    # TODO: handle the fact that in with-reference setting each sample is actually requires 3 forward passes
    return len(dataset)

@torch.inference_mode()
def run_metric(model, dataset, batch_size, args):
    logger.info("Computing metric...")
    start = time.perf_counter()
    model_output = model.predict(dataset, batch_size=batch_size, gpus=int(args.gpu))
    torch.cuda.synchronize()
    prediction_time = time.perf_counter() - start
    return model_output, prediction_time

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
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

# Start logic
    output_path = Path(args.output) / "evaluations" / ("no_reference" if args.dataset.endswith(".tsv") else "with_reference") / args.lp

    if os.path.exists(output_path):
        logger.info("Reusing previous results. Change output folder or delete this folder to recompute.")
        return

    os.makedirs(output_path, exist_ok=True)

    dataset, ground_truth, dataset_load_time = get_dataset(args)

    model, model_load_time = get_model(args, device)

    batch_size, batch_size_time = get_batch_size(model, args, device)

    model_output, prediction_time = run_metric(model, dataset, batch_size, args)

    segment_scores = np.array(model_output.scores)
# Construct report
    peak_memory_mb = get_memory_allocated(device, model, is_max=True) // 2 ** 20
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
    # If batch size was selected, update it
    report["batch_size"] = batch_size
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

    print_summary(logger, report)

if __name__ == "__main__":
    main()