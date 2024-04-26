import os
import numpy as np
import argparse
import json
import pathlib


def make_result_dict():
    return {
        key: 0.0
        for key in ('peak_memory_mb', 'samples_per_second', 'prediction_time', 'kendall_score', 'system_score')
    }


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir')
    parser.add_argument('--model_name')
    return parser


def parse_results(root, model_name):
    result = make_result_dict()
    total_len = 0
    cnt = 0
    for type_ in ['with_reference', 'no_reference']:
        base = root / type_ / model_name
        for filename in os.listdir(base):
            path = base / filename
            if not os.path.exists(path / 'report.json'):
                continue
            with open(path / 'report.json') as fin:
                data = json.load(fin)
                cnt += 1
                total_len += data['dataset_length']
                result['peak_memory_mb'] += data['dataset_length'] * data['peak_memory_mb']
                result['prediction_time'] += data['prediction_time']
                result['samples_per_second'] += data['samples_per_second']
                result['kendall_score'] += data['dataset_length'] * data['kendall_correlation']
                result['system_score'] += data['dataset_length'] * data['system_level_score']
    result['peak_memory_mb'] /= total_len
    result['kendall_score'] /= total_len
    result['system_score'] /= total_len
    result['samples_per_second'] /= cnt
    result = {key: np.round(value, 3) for key, value in result.items()}
    for key in ('peak_memory_mb', 'samples_per_second', 'prediction_time'):
        result[key] = int(np.round(result[key]))
    return result


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    path = pathlib.Path(args.root_dir)
    print(parse_results(path, args.model_name))
    # print(data)