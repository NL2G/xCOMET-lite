import os
os.environ['HF_HOME'] = "/pfs/work7/workspace/scratch/ma_dalarion-data-cache/hf"

from rich import print
import datasets as ds
import model_utils as mu
import transformers as tr
import argparse as ap
from rich.logging import RichHandler
from rich.console import Console
import logging
from omegaconf import OmegaConf

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=True, 
        console=Console(color_system=None)
    )]
)

logger = logging.getLogger(__name__)


def main(config_path: str):

    config = OmegaConf.load(config_path)

    logger.info(f"Loading dataset from {config.misc.dataset}...")
    dataset = ds.Dataset.from_json(config.misc.dataset)
    dataset = dataset.train_test_split(test_size=config.misc.dev_size, seed=config.misc.seed)
    tokenizer = tr.AutoTokenizer.from_pretrained(config.misc.base_model_name)

    logger.info("Tokenizing dataset...")
    tokenize_fn = mu.get_tokenize_fn(
        tokenizer=tokenizer,
        kind=config.misc.kind
    )

    dataset = dataset.map(
        tokenize_fn, 
        batched=True, 
        batch_size=1024, 
        num_proc=80,
        remove_columns=dataset['train'].column_names
    )
    logger.info("Tokenization complete.")

    logger.info("Calculating lengths...")
    dataset = dataset.map(
        mu.length_fn,
        batched=True,
        batch_size=1024,
        num_proc=40
    )
    logger.info("Length calculation complete.")

    dataset.save_to_disk(
        dataset_dict_path=config.misc.preprocessed_dataset_prefix + f"_{config.misc.kind}",
        num_shards={'train': 20, 'test': 1}, num_proc=20
    )


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)

    
