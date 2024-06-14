import os
import argparse
import json
import pathlib


def make_result_dict():
    return {
        key: 0.0
        for key in ('peak_memory_mb', 'samples_per_second_macro', 'samples_per_second_micro', 'prediction_time', 'kendall_score_weighted', 'system_score_weighted', 'kendall_score_mean', 'system_score_mean', 'dataset_length')
    }


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir')
    parser.add_argument('--model_name')
    return parser


def parse_results(root, model_name):
    result = make_result_dict()
    total_ = 0
    for type_ in ['with_reference', 'no_reference']:
        base = root / type_ / model_name / 'evaluations' / type_
        for filename in os.listdir(base):
            path = base / filename
            if not os.path.exists(path / 'report.json'):
                continue
            with open(path / 'report.json') as fin:
                data = json.load(fin)
                result['dataset_length'] += data['dataset_length']
                result['peak_memory_mb'] = max(data['peak_memory_mb'], result['peak_memory_mb'])
                result['prediction_time'] += data['prediction_time']
                result['samples_per_second_micro'] += data['prediction_time'] * data['samples_per_second']
                result['samples_per_second_macro'] += data['samples_per_second']
                result['kendall_score_weighted'] += data['dataset_length'] * data['kendall_correlation']
                result['system_score_weighted'] += data['dataset_length'] * data['system_level_score']
                result['kendall_score_mean'] += data['kendall_correlation']
                result['system_score_mean'] += data['system_level_score']
                total_ += 1
    result['kendall_score_weighted'] /= result['dataset_length']
    result['system_score_weighted'] /= result['dataset_length']
    result['samples_per_second_micro'] /= result['prediction_time']
    result['samples_per_second_macro'] /= total_
    result['kendall_score_mean'] /= total_
    result['system_score_mean'] /= total_
    result['system_score_mean'] /= total_
    result = {key: round(value, 3) for key, value in result.items()}
    for key in ('peak_memory_mb', 'samples_per_second_macro', 'samples_per_second_micro', 'prediction_time'):
        result[key] = round(result[key], 3)
    return result


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    path = pathlib.Path(args.root_dir)
    print(parse_results(path, args.model_name))
    # print(data)