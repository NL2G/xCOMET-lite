#!/usr/bin/env python
# coding: utf-8

from datasets import Dataset as ds
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

model_encoder_name = "bigscience/mt0-large"
dataset_name = 'nllg/wmt-metrics-data'
config_upload_name = 'mt0-labse'
token = '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'
sentence_transformers_model = 'sentence-transformers/LaBSE'
random_seed = 42
prompt_template_simple = "{source_seg} [SEP] {reference_seg} [SEP] {target_seg}"
save_to_disk_path = "./wmt-labse"


if __name__ == "__main__":
    
    logger.info("Loading dataset")

    dataset = load_dataset(
        dataset_name, 
        token=token,
    )
    
    logger.info("Creating prompts")


    dataset = dataset.map(
        lambda x: {'prompt': prompt_template_simple.format(
            source_seg=x['src'],
            reference_seg=x['ref'],
            target_seg=x['mt']
        )},
        batched=False,
        num_proc=10,
        remove_columns=[
            'lp', 'score_type'
        ],
        keep_in_memory=True
    )
    
    logger.info("Loading tokenizer and embedding model")

    tokenizer = AutoTokenizer.from_pretrained(model_encoder_name)

    labse = SentenceTransformer(sentence_transformers_model, device='cuda:0')

    def encode_labse(example):
        batch_size = len(example['src'])
        flat_list = example['src'] + example['mt'] + example['ref']
        vectors = labse.encode(
            flat_list
        , device='cuda:0', show_progress_bar=False, batch_size=1024)

        src_vecs = vectors[:batch_size]
        mt_vecs = vectors[batch_size:(batch_size*2)]
        ref_vecs = vectors[(batch_size*2):]

        labse_emb = np.concatenate([
            src_vecs, mt_vecs, ref_vecs
        ], axis=1)

        return {
            'labse': labse_emb
        }

    def tokenize_fn(examples):
        embedding = tokenizer(
            examples["prompt"],
            padding=False,
            truncation=True,
            max_length=512,
        )

        return embedding
    
    logger.info("Tokenizing")

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=2048,
        num_proc=20,
        keep_in_memory=True
    )
    
    logger.info("Sorting")    

    dataset = dataset.map(
        lambda x: {'len': len(x['ref'])},
        batched=False,
        num_proc=10,
        keep_in_memory=True
    )
    dataset = dataset.sort(column_names=['len'], keep_in_memory=True)

    logger.info("Embedding")

    dataset = dataset.map(
        encode_labse,
        batched=True,
        batch_size=2048,
        keep_in_memory=True,
        remove_columns=['prompt', 'src', 'mt', 'ref', 'len']
    )

    dataset = dataset.rename_column("score", "labels")

    dataset = dataset.shuffle(random_seed)

    dataset.save_to_disk(save_to_disk_path)

    dataset.push_to_hub(
        dataset_name,
        config_name=config_upload_name,
        token=token, num_shards={
            'train': 10,
            'test': 1
        }
    )