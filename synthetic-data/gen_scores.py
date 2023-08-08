import torch
torch.set_float32_matmul_precision('medium')
import comet
import sys
from bleurt import score as bleurt_score
import argparse as ap
from datasets import Dataset
from tqdm.auto import tqdm
from rich.logging import RichHandler
import logging
from rich import inspect

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

def compute_comet(data: Dataset) -> Dataset:
    gpus = 1 if torch.cuda.is_available() else 0
    logger.info(f"Computing COMET scores with {comet} ...")
    model = comet.load_from_checkpoint(comet.download_model("Unbabel/wmt22-comet-da"))
    data_dict = {"src": data['src'], "mt": data['mt'], "ref": data['tgt']}
    data_dict = [dict(zip(data_dict, t)) for t in zip(*data_dict.values())]
    scores = model.predict(data_dict, gpus=gpus, progress_bar=True)['scores']
    logger.info(f"COMET scores: {len(scores)} for {len(data)} examples")
    data = data.add_column("comet", scores)
    return data

def compute_bleurt(data: Dataset) -> Dataset:
    bleurt = bleurt_score.BleurtScorer("./BLEURT-20/")
    logger.info(f"Computing BLEURT scores with {bleurt} ...")
    score = bleurt.score(candidates=data['mt'], references=data['tgt'])
    logger.info(f"BLEURT scores: {len(score)} for {len(data)} examples")
    data = data.add_column("bleurt", score)
    return data

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        description="Generate scores for synthetic data",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        prog="gen_scores",
    )

    parser.add_argument(
        "--data",
        type=str,
        help="path to csv file",
    )

    parser.add_argument(
        "--metrics",
        help="list of metrics to compute",
        action="append",
    )

    args = parser.parse_args()
    logger.info(f"Scoring {args.data} with {args.metrics} ...")

    dataset = Dataset.from_csv(args.data)

    if "comet" in args.metrics:
        if 'comet' in dataset.column_names:
            dataset = dataset.remove_columns(['comet'])
        dataset = compute_comet(dataset)

    if "bleurt" in args.metrics:
        if 'bleurt' in dataset.column_names:
            dataset = dataset.remove_columns(['bleurt'])
        dataset = compute_bleurt(dataset)

    if 'score' in dataset.column_names:
        dataset = dataset.remove_columns(['score'])

    dataset = dataset.map(
        lambda x: {
            'score': sum(x[name] for name in args.metrics) / len(args.metrics)
        }
    )

    dataset.to_csv(args.data)