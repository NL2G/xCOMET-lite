import os
import argparse
import json
import pathlib
import numpy as np


def make_result_dict():
    result = {
        key: 0.0
        for key in (
            'peak_memory_mb_max',
            'peak_memory_mb_mean',
            'samples_per_second_micro',
            'prediction_time',
            'kendall_score_weighted',
            'system_score_weighted',
            'kendall_score_mean',
            'kendall_score_std',
            'system_score_mean',
            'dataset_length',
            'total'
        )
    }
    result.update(
        {
            'samples_per_second_macro_max': 0,
            'samples_per_second_macro_mean': 0,
            'samples_per_second_macro_std': 0,
            'samples_per_second_macro_median': [],
            'samples_per_second_macro_min': float('+inf')
        }
    )
    return result


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir')
    parser.add_argument('--model_name')
    return parser


def parse_results(root, model_name):
    report = {}
    total_ = 0
    for type_ in ['with_reference', 'no_reference']:
        result = make_result_dict()
        base = root / model_name / 'evaluations' / type_
        for filename in os.listdir(base):
            path = base / filename
            if not os.path.exists(path / 'report.json'):
                continue
            with open(path / 'report.json') as fin:
                data = json.load(fin)
                result['dataset_length'] += data['dataset_length']
                result['peak_memory_mb_max'] = max(data['peak_memory_mb'], result['peak_memory_mb_max'])
                result['peak_memory_mb_mean'] += data['peak_memory_mb']
                result['prediction_time'] += data['prediction_time']
                result['samples_per_second_micro'] += data['prediction_time'] * data.get('samples_per_second', 0)
                result['samples_per_second_macro_min'] = min(result['samples_per_second_macro_min'], data.get('samples_per_second', 0))
                result['samples_per_second_macro_mean'] += data.get('samples_per_second', 0)
                result['samples_per_second_macro_std'] += data.get('samples_per_second', 0) ** 2
                result['samples_per_second_macro_median'].append(data.get('samples_per_second', 0))
                result['samples_per_second_macro_max'] = max(result['samples_per_second_macro_max'], data.get('samples_per_second', 0))
                result['kendall_score_weighted'] += data['dataset_length'] * data['kendall_correlation']
                result['system_score_weighted'] += data['dataset_length'] * data['system_level_score']
                result['kendall_score_mean'] += data['kendall_correlation']
                result['kendall_score_std'] += data['kendall_correlation'] ** 2
                result['system_score_mean'] += data['system_level_score']
                result['total'] += 1
        result['kendall_score_weighted'] /= result['dataset_length']
        result['system_score_weighted'] /= result['dataset_length']
        result['samples_per_second_micro'] /= result['prediction_time']
        result['samples_per_second_macro_median'] = np.median(result['samples_per_second_macro_median'])
        result['samples_per_second_macro_mean'] /= result['total']
        result['samples_per_second_macro_std'] = (result['samples_per_second_macro_std'] / result['total'] - result['samples_per_second_macro_mean'] ** 2) ** 0.5
        result['kendall_score_mean'] /= result['total']
        result['kendall_score_std'] = (result['kendall_score_std'] / result['total'] - result['kendall_score_mean'] ** 2) ** 0.5
        result['system_score_mean'] /= result['total']
        result['system_score_mean'] /= result['total']
        result['peak_memory_mb_mean'] /= result['total']
        result = {key: round(value, 3) for key, value in result.items()}
        report[type_] = result
    return report


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    path = pathlib.Path(args.root_dir)
    print(json.dumps(parse_results(path, args.model_name), sort_keys=True, indent=4))
