import json
import math
import functools
import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple
import gc

from torch.utils.data import Sampler, Dataset

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

def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        model_input_name = model_input_name if model_input_name is not None else "input_ids"

        if dataset is not None:
            lengths = dataset.map(
                lambda x: {'len': len(x[model_input_name])},
                batched=False,
                num_proc=4,
            )
            
        lengths = lengths['len']

        self.lengths = lengths
        if generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(0)
        self.generator = generator
        gc.collect()

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)


class CheckpointedXLMRobertaLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    # signature was taken from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L380
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        return torch.utils.checkpoint.checkpoint(self.layer, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, use_reentrant=False)

def enable_gradient_checkpointing(model):
    # Gradient checkpointing saves memory during training
    model.encoder.model.encoder.layer = torch.nn.ModuleList([
        CheckpointedXLMRobertaLayer(layer) for layer in model.encoder.model.encoder.layer
    ])

#
# Taken from https://gist.github.com/akshaychawla/86d938bc6346cf535dce766c83f743ce
#
def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def _constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier

def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup, 
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler

def LinearWarmup(optimizer, T_warmup):
    _decay_func = functools.partial(
        _constant_warmup, 
        warmup_iterations=T_warmup
    )
    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler