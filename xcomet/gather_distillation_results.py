import os
import pandas as pd
from utils import load_json

def main():
    for prefix in ["with_reference"]:
        experiment_names = [
            "synthplus-mdeberta-no-freeze-0",
            "synthplus-mdeberta-no-freeze-1",
            "synthplus-mdeberta-no-freeze-2",
            "synthplus-mdeberta-no-freeze-highbs-0",
            "synthplus-mdeberta-no-freeze-highbs-1",
            "synthplus-mdeberta-no-freeze-highbs-2",
            "synthplus-mdeberta-no-freeze-lowlr-0",
            "synthplus-mdeberta-no-freeze-lowlr-1",
            "synthplus-mdeberta-no-freeze-lowlr-2",
        ]
        
        results_dir_name = "distillation_results"
        all_language_pairs = ("en-de", "en-ru", "zh-en")
        other_language_pairs = ("en-de", "en-es", "en-zh")
        
        output_name = f"synthplus_{prefix}_summary.csv"

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
        #print(results)

        results = results[interesting_columns]
        results.sort_values(by=["experiment_name", "lp"], inplace=True)
        results.to_csv(f"{results_dir_name}/{output_name}", index=False)


if __name__ == "__main__":
    main()