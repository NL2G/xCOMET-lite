import transformers as tr
import peft
import datasets as ds
import torch
from transformers.activations import ACT2FN
import argparse as ap
import logging
import numpy as np
from scipy.stats import kendalltau

logger = logging.getLogger(__name__)

def load_model(path: str, is_peft: bool = True, n_bits: int = 4):

    if is_peft:
        config = peft.PeftConfig.from_pretrained(path)
        model_name = config.model_name_or_path

    else:
        model_name = path

    model_kwargs = {}

    if n_bits == 4:
        model_kwargs['quantization_config'] = tr.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif n_bits == 8:
        model_kwargs['quantization_config'] = tr.BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs['torch_dtype'] = torch.float16
    elif n_bits == 16:
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif n_bits == 32:
        model_kwargs['torch_dtype'] = torch.float32
    else:
        raise ValueError(f'Invalid number of bits: {n_bits}')
    
    model = tr.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, 
        **model_kwargs
    )

    if is_peft:
        model = peft.PeftModelForSequenceClassification.from_pretrained(
            model, model_id=path
        )

    return model


def main():
    parser: ap.ArgumentParser = ap.ArgumentParser(
        prog='evaluate.py',
        description='Evaluate a model on a dataset'
    )
    parser.add_argument(
        '--path',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="nllg/wmt-metrics-data"
    )
    parser.add_argument(
        '--n-bits',
        type=int,
        default=4
        choices=[4, 8, 16, 32]
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        default=False
    )



