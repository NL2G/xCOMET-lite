import os
import torch
import numpy as np

from scipy.stats import kendalltau
from argparse import ArgumentParser
from datasets import load_dataset
from comet import download_model, load_from_checkpoint

def make_parser():
    parser = ArgumentParser(description="xCOMET evaluation.")
    parser.add_argument("-o", "--output", help="Where to save results")
    parser.add_argument("--model", help="Which model to use")
    parser.add_argument("--lp", help="On which language pair to compute metrics")
    parser.add_argument("--domain", default="news", help="On which domain to compute metrics")
    parser.add_argument("--year", default=2022, help="In which year to compute metrics")
    parser.add_argument("--force", default=False, action="store_true", help="Overwrite experiment with this output folder")

    return parser

@torch.inference_mode()
def main():
    torch.set_float32_matmul_precision("medium")

    parser = make_parser()
    args = parser.parse_args()
    print(args)
    if not args.force and os.path.exists(args.output):
        print(f"Folder {args.output} already exists. Change output folder or use --force flag.")
        return

    dataset = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train")
    dataset = dataset.filter(lambda example: example["year"] == args.year and example["domain"] == args.domain and example["lp"] == args.lp)
    print("N samples:", len(dataset))
    print("First sample:\n", dataset[0], "\n")

    segment_score_path = f"{args.output}/model_segment_level_scores.npy"

    if not os.path.exists(segment_score_path):
        model_path = download_model(args.model)
        model = load_from_checkpoint(model_path).half()
        
        # for name, parameter in model.named_parameters():
        #     if not ("Norm" in name or "bias" in name or "scalar_parameters" in name):
        #         model.state_dict()[name] = parameter.half()

        print(model.encoder.model.encoder.layer[0].output.dense.weight.dtype)

        print("Computing metric...")
        model_output = model.predict([sample for sample in dataset], batch_size=8, gpus=1)
        # Segment-level scores
        os.makedirs(args.output, exist_ok=True)
        segment_scores = np.array(model_output.scores)
        np.save(segment_score_path, segment_scores)
    else:
        segment_scores = np.load(segment_score_path)

    print("Max memory:", torch.cuda.max_memory_allocated() // 2 ** 20, "Mb")

    kendall_corr = kendalltau(dataset["score"], segment_scores)
    print("Kendall correlation:", kendall_corr)

    # # System-level score
    # print("System score:", model_output.system_score)

    # # Score explanation (error spans)
    # print(model_output.metadata.error_spans)

if __name__ == "__main__":
    main()