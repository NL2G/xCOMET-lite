import os
import pandas as pd
from utils import load_json

def main():
    results_dir_name = "results"
    all_language_pairs = ("en-de", "en-ru", "zh-en")
    other_language_pairs = ("en-de", "en-es", "en-zh")
    output_name = "summary.csv"

    paths = [
        *[f"{results_dir_name}/vanilla_xcomet_xl/{language_pair}/report.json" for language_pair in all_language_pairs],
        *[f"{results_dir_name}/vanilla_xcomet_xxl/{language_pair}/report.json" for language_pair in all_language_pairs],
    ]

    records = [load_json(path) for path in paths]
    results = pd.DataFrame(records)

    interesting_columns = [
        "model", "lp", "kendall_correlation", "peak_memory_mb", "system_level_score", "prediction_time", "model_load_time", "domain", "year"
    ]

    results = results[interesting_columns]
    results.sort_values(by=["model", "lp"])
    results.to_csv(f"{results_dir_name}/{output_name}", index=False)


if __name__ == "__main__":
    main()