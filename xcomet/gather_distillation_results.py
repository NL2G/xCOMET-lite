import os
import pandas as pd
from utils import load_json

def main():
    for prefix in ("with_reference", "no_reference"):
        experiment_names = ["default", "more_word_level_loss", "unfrozen_encoder", "debug_3_stages", "3_stages"]
        
        results_dir_name = "distillation_results"
        all_language_pairs = ("en-de", "en-ru", "zh-en")
        other_language_pairs = ("en-de", "en-es", "en-zh")
        
        output_name = f"{prefix}_summary.csv"

        paths = [
            f"{results_dir_name}/{experiment_name}/evaluations/{prefix}/{language_pair}/report.json" 
                for language_pair in (other_language_pairs if prefix == "no_reference" else all_language_pairs)
                for experiment_name in experiment_names
        ]
        paths = [path for path in paths if os.path.exists(path)]

        records = [load_json(path) for path in paths]
        results = pd.DataFrame(records)
        results["experiment_name"] = [path.split("/")[1] for path in paths]
        results["setup"] = prefix

        interesting_columns = [
            "experiment_name", "lp", "kendall_correlation", "peak_memory_mb", \
            "system_level_score", "prediction_time", "model_load_time", "domain", "year", "setup",
        ]

        results = results[interesting_columns]
        results.sort_values(by=["experiment_name", "lp"], inplace=True)
        results.to_csv(f"{results_dir_name}/{output_name}", index=False)


if __name__ == "__main__":
    main()