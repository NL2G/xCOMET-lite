# coding: utf-8

import time
from typing import Dict, List

import torch
from torch import nn
import numpy as np
import logging

logger = logging
logger.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s> {%(module)s} %(message)s',
    datefmt='%Y-%m-%d_%H:%M:%S'
)


def to_onnx(
    model: nn.Module,
    batch: Dict[str, torch.tensor],
    filepath: str,
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]]
):
    logger.info('Export PyTorch to ONNX')
    start = time.perf_counter()
    torch.onnx.export(
        model,
        tuple(batch.values()),
        f=filepath,
        input_names=list(batch.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17,
    )
    logger.info(f'Exporting is done in {np.round(time.perf_counter()-start, 2)} seconds')


def xcomet_to_onnx(
    model: nn.Module,
    filepath: str,
    batch_size: int = 4
):
    text = 'text'
    tokens = model.encoder.tokenizer.encode(text)
    input_ids = torch.from_numpy(np.stack([tokens]*batch_size))
    batch = {
        'input_ids': input_ids,
        'attention_mask': torch.full_like(input_ids, True, dtype=torch.bool)
    }
    output_names = ['output']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'}
    }
    to_onnx(model, batch, filepath, output_names, dynamic_axes)
