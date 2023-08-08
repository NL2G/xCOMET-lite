import argparse as ap
import logging

from datasets import DatasetDict
from rich.logging import RichHandler
from transformers import XLMRobertaTokenizerFast

from config import DATA_CONFIG
from data_utils import load_from_config, make_preprocessing_fn

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def main():
    parser: ap.ArgumentParser = ap.ArgumentParser(
        prog="preprocess.py",
        description="Preprocess the dataset",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="ID of the data configuration",
    )

    parser.add_argument(
        "--encoder-model-name",
        type=str,
        required=True,
        help="Name of the encoder model",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes to use for preprocessing",
    )

    parser.add_argument(
        "--dev-size",
        type=float,
        default=0.01,
        help="Size of the development set",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum length of the input sequence",
    )

    args = parser.parse_args()

    data_args = DATA_CONFIG[args.data_config]
    args: ap.Namespace = ap.Namespace(**dict(**vars(args), **data_args))

    logger.info("Loading data")
    train, dev, test = load_from_config(args.train, args.test, args.dev_size, args.seed)

    _datasets = {
        "train": train,
        "dev": dev,
    }
    for k, v in test.items():
        _datasets[f"test_{k}"] = v

    datasets: DatasetDict = DatasetDict(_datasets)

    logger.info("Loading tokenizer")
    tokenizer: XLMRobertaTokenizerFast = XLMRobertaTokenizerFast.from_pretrained(
        args.encoder_model_name
    )

    logger.info("Preprocessing data")
    preprocessing_fn = make_preprocessing_fn(
        tokenizer=tokenizer, max_length=args.max_length, return_length=False
    )

    columns_to_remove = datasets["train"].column_names

    datasets = datasets.map(
        preprocessing_fn,
        batched=True,
        num_proc=args.num_processes,
        remove_columns=columns_to_remove,
        desc="Tokenization...",
    )

    logger.info("Saving data")
    datasets.save_to_disk(
        args.output_dir, 
        num_proc=args.num_processes,
        num_shards={
            key: 1 for key in datasets.keys()
        }
    )


if __name__ == "__main__":
    main()
    logger.info("Done!")
