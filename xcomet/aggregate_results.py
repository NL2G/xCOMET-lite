import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from argparse import ArgumentParser

from utils import load_json

WITH_REFERENCE_LANGUAGE_PAIRS = ("en-de", "en-ru", "zh-en")
NO_REFERENCE_LANGUAGE_PAIRS = ("en-de", "en-es", "en-zh")

def gather_quantization_results(results_dir_name: str = "quantization_results") -> List[pd.DataFrame]:
    all_results = []

    for evaluation_mode in ("no_reference", "with_reference"):
        experiment_names = ["vanilla", "8bit", "3bit"]
        models = ["xcomet_xl", "xcomet_xxl"]
        
        paths = [
            f"{results_dir_name}/{evaluation_mode}/{experiment_name}_{model_name}/{language_pair}/report.json" 
                for language_pair in (NO_REFERENCE_LANGUAGE_PAIRS if evaluation_mode == "no_reference" else WITH_REFERENCE_LANGUAGE_PAIRS)
                for experiment_name in experiment_names
                for model_name in models
        ]
        paths = [path for path in paths if os.path.exists(path)]

        records = [load_json(path) for path in paths]
        results = pd.DataFrame(records)
        results["experiment_name"] = [path.split("/")[2].split("_")[0] for path in paths]
        results["setup"] = evaluation_mode

        interesting_columns = [
            "model", "lp", "experiment_name", "kendall_correlation", "peak_memory_mb", \
            "system_level_score", "prediction_time", "model_load_time", "domain", "year", "setup",
        ]

        results = results[interesting_columns]
        results.sort_values(by=["model", "lp", "experiment_name"], inplace=True)
        #results.to_csv(f"{results_dir_name}/{output_name}", index=False)
        all_results.append(results)

    return all_results

def gather_pruning_results(results_dir_name: str = "pruning_results") -> List[pd.DataFrame]:
    all_results = []
    for evaluation_mode in ("no_reference", "with_reference"):
        experiment_names = ["vanilla"] + [f"prune{k:02d}layers{suffix}"
            for k in (8, 16) for suffix in ("_finetune6",)]
        #models = ["xcomet_xl", "xcomet_xxl"]
        
        paths = [
            f"{results_dir_name}/{experiment_name}/evaluations/{evaluation_mode}/{language_pair}/report.json" 
                for language_pair in (NO_REFERENCE_LANGUAGE_PAIRS if evaluation_mode == "no_reference" else WITH_REFERENCE_LANGUAGE_PAIRS)
                for experiment_name in experiment_names
        ]
        paths = [path for path in paths if os.path.exists(path)]

        records = [load_json(path) for path in paths]
        results = pd.DataFrame(records)
        results["experiment_name"] = [path.split("/")[1] for path in paths]
        results["setup"] = evaluation_mode

        interesting_columns = [
            "model", "lp", "experiment_name", "kendall_correlation", "peak_memory_mb", \
            "system_level_score", "prediction_time", "model_load_time", "domain", "year", "setup",
        ]

        results = results[interesting_columns]
        results.sort_values(by=["model", "lp", "experiment_name"], inplace=True)
        all_results.append(results)

    return all_results

def gather_speed_results(results_dir_name: str = "speed_results") -> List[pd.DataFrame]:
    all_results = []

    for evaluation_mode in ("no_reference", "with_reference"):
        experiment_names = ["vanilla"] + [f"prune_{k:d}_layers"
            for k in (8, 16)]
        
        # experiment_names = ["vanilla"] + [f"{k}bits" for k in (3, 8)]
        
        models = ["xl", "xxl"]

        paths = [
            f"{results_dir_name}/{model_name}_{experiment_name}/evaluations/{evaluation_mode}/{language_pair}/report.json" 
                for language_pair in (NO_REFERENCE_LANGUAGE_PAIRS if evaluation_mode == "no_reference" else WITH_REFERENCE_LANGUAGE_PAIRS)
                for experiment_name in experiment_names
                for model_name in models
        ]
        paths = [path for path in paths if os.path.exists(path)]

        records = [load_json(path) for path in paths]
        results = pd.DataFrame(records)
        results["experiment_name"] = ["_".join(path.split("/")[1].split("_")[1:]) for path in paths]
        results["setup"] = evaluation_mode

        interesting_columns = [
            "model", "lp", "experiment_name", "kendall_correlation", "peak_memory_mb", "samples_per_second", \
            "prediction_time", "model_load_time", "domain", "year", "setup",
        ]

        results = results[interesting_columns]
        results.sort_values(by=["model", "lp", "experiment_name"], inplace=True)
        all_results.append(results)
    
    return all_results

def gather_distillation_results(results_dir_name: str = "mdeberta-checkpoints") -> List[pd.DataFrame]:
    all_results = []
    for prefix in ["with_reference"]:
        experiment_names = [
            # "synthplus-mdeberta-no-freeze-0",
            # "synthplus-mdeberta-no-freeze-1",
            # "synthplus-mdeberta-no-freeze-2",
            "synthplus-mdeberta-no-freeze-highbs-0",
            # "synthplus-mdeberta-no-freeze-highbs-1",
            # "synthplus-mdeberta-no-freeze-highbs-2",
            # "synthplus-mdeberta-no-freeze-lowlr-0",
            # "synthplus-mdeberta-no-freeze-lowlr-1",
            # "synthplus-mdeberta-no-freeze-lowlr-2",
        ]

        paths = [
            f"{results_dir_name}/{experiment_name}/evaluations/{prefix}/{language_pair}/report.json" 
                for language_pair in (NO_REFERENCE_LANGUAGE_PAIRS if prefix == "no_reference" else WITH_REFERENCE_LANGUAGE_PAIRS)
                for experiment_name in experiment_names
        ]
        print(paths)
        print(os.path.exists(f"{results_dir_name}/{experiment_names[0]}/evaluations/{prefix}/en-de/report.json"))
        paths = [path for path in paths if os.path.exists(path)]
        assert len(paths) > 0

        records = [load_json(path) for path in paths]
        results = pd.DataFrame(records)
        results["experiment_name"] = [path.split("/")[1] for path in paths]
        results["setup"] = prefix
        results["model"] = "mDeBERTa"

        interesting_columns = [
            "model", "experiment_name", "lp", "kendall_correlation", "peak_memory_mb", "dataset_length", \
            "system_level_score", "prediction_time", "model_load_time", "domain", "year", "setup",
        ]
        #print(results)

        results = results[interesting_columns]
        results.sort_values(by=["experiment_name", "lp"], inplace=True)
        all_results.append(results)
    
    return all_results

def gather(results_dir_name: str, glob_pattern: str) -> pd.DataFrame:
    paths = list(Path(results_dir_name).glob(glob_pattern))
    records = [load_json(path) for path in paths]
    results = pd.DataFrame(records)
    
    # paths are expected to have format 
    # results_dir_name/experiment_name/{training,evaluations}/{no_reference,with_reference}/language_pair/filename
    results["experiment_name"] = [str(path).split("/")[1] for path in paths]
    results["setup"] = [str(path).split("/")[3] for path in paths]

    def _get_model_name(experiment_name):
        if "xxl" in experiment_name:
            return "xCOMET-XXL"
        elif "xl" in experiment_name:
            return "xCOMET-XL"
        elif "mdeberta" in experiment_name:
            return "DeBERTa"
        raise ValueError(f"Can't determine model name from experiment name '{experiment_name}'")

    results["model"] = results["experiment_name"].map(_get_model_name)

    interesting_columns = [
        "model", "experiment_name", "lp", "kendall_correlation", "peak_memory_mb", "dataset_length", \
        "system_level_score", "prediction_time", "model_load_time", "domain", "year", "setup",
    ]
    print(results)
    results = results[interesting_columns]

    return results

def _reformat_folder(results_dir_name: str = "quantization_results", glob_pattern: str = "*/*/*/*"):
    paths = list(Path(results_dir_name).glob(glob_pattern))
    
    new_root = "reformatted_quantization_results"
    Path(new_root).mkdir()

    for path in paths:
        _, setup, experiment_name, lp, filename = str(path).split("/")
        print(setup, experiment_name, lp, filename)

        target_dir = Path(f"{new_root}/{experiment_name}/evaluations/{setup}/{lp}")
        target_dir.mkdir(exist_ok=True, parents=True)

        shutil.copy(path, target_dir / filename)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-r", "--results-dir-name", required=True)
    parser.add_argument("-s", "--setup", help="Setup: 'with_reference', 'no_reference' or '**' (both).", required=True)
    parser.add_argument("-v", "--videocard", default="", help="Videocard: 3090 or a100. Needed for speed results.")

    return parser.parse_args()

def main():
    args = parse_args()
    # paths are expected to have format 
    # results_dir_name/experiment_name/{training,evaluations}/{no_reference,with_reference}/language_pair/filename
    experiment_name_patterns = [
        f"{args.videocard}vanilla*",
        # f"{args.videocard}*mdeberta*",
        # f"{args.videocard}*8bits",
        # f"{args.videocard}*4bits",
        # f"{args.videocard}*3bits",
        # f"{args.videocard}*8_layers",
        # f"{args.videocard}*16_layers"
        f"{args.videocard}wanda*",
        f"{args.videocard}prune08layers_finetune6-xl",
        f"{args.videocard}prune16layers_finetune6-xl",
    ]

    results = pd.concat(
        [gather(args.results_dir_name, f"{exp}/**/{args.setup}/**/report.json") for exp in experiment_name_patterns]
    )
    results = results[~results.experiment_name.str.contains("halfbnb_8bits")]
    results = results[~results.experiment_name.str.contains("truebnb_4bits")]
    results = results[~results.experiment_name.str.contains("xl_4bits")]
    
    results = results.sort_values(["model", "setup"])
    print(results)

    print("Kendall")
    print(results.groupby(["model", "experiment_name"], sort=False)["kendall_correlation"].agg(["mean", "std"]).round(3))

    results["peak_memory_gb"] = np.round(results["peak_memory_mb"].values / 1000, 2)

    # print("Peak Memory")
    # print(results.groupby(["model", "experiment_name"], sort=False)["peak_memory_gb"].agg(["mean", "max"]).round(2))

    if "samples_per_second" not in results.columns:
        results["samples_per_second"] = results["dataset_length"] / results["prediction_time"]

    print("Speed")
    # print(results.groupby(["model", "experiment_name"], sort=False)[["samples_per_second", "prediction_time"]].agg(["mean", "std"]).round(1))
    print(results.groupby(["model", "experiment_name"], sort=False)[["samples_per_second", "prediction_time"]].agg(["min", "median", "max"]).round(1))

if __name__ == "__main__":
    main()