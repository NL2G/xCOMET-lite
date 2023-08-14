import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from pathlib import Path

import logging
from rich.logging import RichHandler

import torch
from torch import nn
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset, Dataset, DatasetDict

import bitsandbytes as bnb
import wandb

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from accelerate import Accelerator

from transformers import AutoTokenizer, MT5EncoderModel
from transformers import DataCollatorWithPadding
from transformers import get_scheduler

import gc
from functools import partial

from specials import *
from modules import *


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':

    accelerator = Accelerator(log_with='wandb', gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    accelerator.init_trackers(
        project_name='wmtcl_mt0',
        init_kwargs={'entity': 'airi23-efficient-llm-metrics'}
    )

    logging.basicConfig(
        level="INFO" if accelerator.is_main_process else "ERROR",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger(__name__)

    wmt_dsets = load_dataset('nllg/wmt-metrics-data', token='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').filter(lambda x: x['lp'] in LANG_PAIRS)
    train_dset = wmt_dsets['train'].rename_column('score_type', 'score_type_str')
    test_dset = wmt_dsets['test'].rename_column('score_type', 'score_type_str').map(lambda x: {'score_type': SCORE_TYPE2ID[x['score_type_str']]})

    train_dset = train_dset.map(lambda x: {'score_type': SCORE_TYPE2ID[x['score_type_str']]}).remove_columns(['score_type_str'])
    test_dset = test_dset.map(lambda x: {'score_type': SCORE_TYPE2ID[x['score_type_str']]}).remove_columns(['score_type_str'])

    tokenizer = AutoTokenizer.from_pretrained(MT0_MODEL)

    lp2train_dset, lp2ref_index, lp2mt_index = get_lp_training_staff(train_dset, ST_MODEL)
    lp2train_dset = DatasetDict(lp2train_dset).map(partial(tokenize_, tokenizer=tokenizer))
    test_dset = test_dset.map(partial(tokenize_, tokenizer=tokenizer))

    wmtcl_train_dset = DatasetWMTCL(
        [(lp2train_dset[lp], lp2ref_index[lp], lp2mt_index[lp]) for lp in LANG_PAIRS],
        train_batch_size=TRAIN_BATCH_SIZE
    )
    wmtcl_test_dset = DatasetWMTCL(
        [test_dset],
        inference=True
    )

    train_collator = DataCollatorWithPaddingAndScore(tokenizer, pad_to_multiple_of=8, max_length=MAX_LENGTH)
    test_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, max_length=MAX_LENGTH)

    train_dataloader = DataLoader(wmtcl_train_dset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=train_collator)
    test_dataloader = DataLoader(wmtcl_test_dset, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=test_collator)

    model = MT5EncoderModel.from_pretrained(MT0_MODEL)
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters()
    )
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=1000, num_training_steps=(N_EPOCHS * len(train_dataloader)))
    loss = ContrastiveLossWMT(
        wmtcl_train_dset.n_neighbors,
        score_type_weights=SCORE_TYPE_WEIGHTS
    )

    train_dataloader, test_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer, scheduler
    )

    accelerator.wait_for_everyone()

    for epoch in range(N_EPOCHS):
        accelerator.print(f'TRAIN EPOCH {epoch + 1}')
        model.train()
        for batch in (pbar := tqdm(train_dataloader, disable=(not accelerator.is_local_main_process))):
            with accelerator.accumulate(model):
                score = batch.pop('score')[0]
                score_type = batch.pop('score_type')[0]

                outputs = model(**batch)
                outputs = mean_pooling(outputs.last_hidden_state, batch['attention_mask'])

                loss_ = loss(outputs, score, score_type)
                free_()

                loss_.backward()
                free_()

                optimizer.step()
                free_()

                scheduler.step()
                optimizer.zero_grad()

                log_ = {'loss': loss_.item()}
                pbar.set_postfix(log_)
                accelerator.log(log_)

        accelerator.print(f'EVAL EPOCH {epoch + 1}')
        model.eval()
        total_correlation = 0.0
        total_ = 0.0
        if accelerator.is_local_main_process:
            for batch in (pbar := tqdm(test_dataloader, disable=(not accelerator.is_local_main_process))):
                with torch.no_grad():
                    outputs = model(**batch)
                    outputs = mean_pooling(outputs.last_hidden_state, batch['attention_mask'])
                    outputs = torch.nn.functional.normalize(outputs, dim=1)
                    embs_src = outputs[0::2]
                    embs_ref = outputs[1::2]

                    correlation_ = (embs_src @ embs_ref.T).diag().sum().item()
                    total_correlation += correlation_
                    total_ += (len(batch) / 2)

                    log_ = {'src & ref correlation': total_correlation / total_}
                    pbar.set_postfix(log_)

            log_ = {'src & ref correlation': total_correlation / total_}
            accelerator.print(f'src & ref correlation: {total_correlation / total_}')
            accelerator.log(log_)

        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        accelerator.save(model.state_dict(), f'{DATA_DIR}/wmtcl_mt0_encoder.pth')
