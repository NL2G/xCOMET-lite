# coding: utf-8

import json
import numpy as np
import pandas as pd
import torch
import gc
import time
import logging
import functools


logger = logging
logger.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s> {%(module)s} %(message)s',
    datefmt='%Y-%m-%d_%H:%M:%S'
)

def dump_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_tsv(path):
    data = pd.read_csv(path, sep="\t")
    data.index = np.arange(len(data))
    data = data.drop(columns=["Unnamed: 0"])
    return data

def get_memory_allocated(device, model, is_max=False):
    if isinstance(model, torch.nn.Module):
        return torch.cuda.max_memory_allocated(device) if is_max else torch.cuda.memory_allocated(device)
    free, total = torch.cuda.mem_get_info(device)
    return total - free

@torch.inference_mode()
def find_max_bs(model, vocab_size, device, n_iter=10, max_length=512):
    if isinstance(model, torch.nn.Module):
        model.eval()
        model = model.to(device)
    b_sz = 1
    init_memory = get_memory_allocated(device, model)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    logger.info(f"Free memory: {(total_memory-init_memory) // 2 ** 20} / {total_memory // 2 ** 20}")
    while True:
        logger.info(f"Trying {b_sz}...")
        torch.cuda.reset_peak_memory_stats(device)
        input_ids = torch.randint(0, vocab_size, (b_sz, max_length), device=device)
        attention_mask = torch.full_like(input_ids, True, dtype=torch.bool)

        start = time.perf_counter()
        for _ in range(n_iter):
            _ = model.forward(input_ids, attention_mask)

        peak_memory = get_memory_allocated(device, model, is_max=True)
        diff_memory = peak_memory - init_memory
        end = time.perf_counter() - start
        throughput = b_sz * n_iter / end
        logger.info(f"Peak memory: {peak_memory // 2 ** 20}")
        logger.info(f"Diff memory: {diff_memory // 2 ** 20}")
        if total_memory < init_memory + 2 * diff_memory:
            break
        b_sz <<= 1
    logger.info(f"Results: {b_sz} points per batch, {peak_memory // 2 ** 20} peak memory, {throughput} samples per s")
    if isinstance(model, torch.nn.Module):
        model = model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return (b_sz, throughput, peak_memory // 2 ** 20)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
