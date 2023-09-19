from MT0Regressor import MT0Regressor, Args
import sentence_transformers as st
import json
import torch
import argparse as ap
from torch import nn
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
import datasets as ds
from scipy.stats import kendalltau
from transformers import AutoTokenizer, DataCollatorWithPadding

from rich.logging import RichHandler
import logging

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser: ap.ArgumentParser = ap.ArgumentParser(prog='evaluate.py')
    parser.add_argument(
        "--model-file", required=True, type=str,
    )
    parser.add_argument(
        "--use-labse", action='store_true', default=False
    )
    parser.add_argument(
        "--output-file", type=str, default="results_{model_file}.json"
    )
    args: ap.Namespace = parser.parse_args()

    logger.info(f"Running with args: {args}")
    logger.info(f"Loading model from {args.model_file}")

    config = Args(
        encoder_name='bigscience/mt0-large',
        size_labsell=512,
        sizes_mlp=[384, 96, 1], # starting from 2th linear layer; 1th layer size 1024 + 512 = 1536
        hidden_act=nn.Tanh,
        dropout_coef=0.1,
        need_lora=False,
        output_act=nn.Sigmoid,
        loss_fc=nn.MSELoss,
        use_labse=args.use_labse,
        checkpoint=None
    )
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    checkpoint = torch.load(args.model_file)
    
    checkpoint = OrderedDict(
        {key.replace("module.", ""): value for key, value in checkpoint.items()}
    )

    model: MT0Regressor = MT0Regressor(config)
    model.load_state_dict(checkpoint)

    device = torch.device("cuda")
    model = model.to(device)

    logger.info(f"Model loaded to {device}")

    prompt_template_simple = "{source_seg} [SEP] {reference_seg} [SEP] {target_seg}"
    def encode_fn(examples):
        prompts = [
            prompt_template_simple.format(
                source_seg=src,
                reference_seg=ref,
                target_seg=mt
            ) for src, ref, mt in zip(examples['src'], examples['mt'], examples['ref'])
        ]
        
        return tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding=False
        )
    
    logger.info(f"Encoding dataset")
    
    dataset = ds.load_dataset(
        "nllg/wmt-metrics-data",
        'default', 
        split='test',
        token="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"
    )

    dataset = dataset.map(
        encode_fn,
        batched=True, batch_size=1024,
        num_proc=10,
    )

    if args.use_labse:

        logger.info(f"Loading LaBSE")
        labse = st.SentenceTransformer('sentence-transformers/LaBSE', device='cuda')

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
        
        logger.info(f"Encoding with LaBSE")
        dataset = dataset.map(
            encode_labse,
            batched=True,
            batch_size=2048,
            keep_in_memory=True,
        )

    dataset = dataset.remove_columns(
        ['mt', 'src', 'ref', 'score_type']
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding='longest',
        pad_to_multiple_of=8,
        max_length=512,
        return_tensors='pt'
    )

    def evaluate(dataset, model, lp: str = 'en-ru'):
        logger.info(f"Evaluating for {lp}")
        ds = dataset.filter(lambda x: x['lp'] == lp, num_proc=10)
        ds_strip = ds.remove_columns(['lp'])
        dl = DataLoader(
            dataset=ds_strip,
            batch_size=32,
            shuffle=False,
            collate_fn=collator
        )
        scores = []
        model.eval()
        for batch in tqdm(dl, desc=f"Evaluating for {lp}"):
            batch_device = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            if args.use_labse:
                batch_device['labse'] = batch['labse'].to(device)

            with torch.no_grad():
                predictions = model(**batch_device)[1].cpu()
                
            for elem in predictions:
                scores.append(elem.item())
                
        return scores, ds['score']
    
    logger.info(f"Starting evaluation")
    lps = set(dataset['lp'])
    scores = {}
    for lp in lps:
        pred, true = evaluate(dataset, model, lp)
        scores[lp] = kendalltau(pred, true).statistic
        logger.info(f"Finished evaluation for {lp} => Kendall: {scores[lp]}")

    logger.info(f"Finished evaluation")
    logger.info(f"Scores: {scores}")
    filepath = args.output_file.format(model_file=args.model_file.replace("./", "").replace('-', '+'))
    logger.info(f"Saving scores to {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    
