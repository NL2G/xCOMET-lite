#!/usr/bin/env python
# coding: utf-8
# srun --pty --partition=single --time=1:00:00 --gres=gpu:2 --cpus-per-task=16 --mem=128G accelerate launch train.py -c ./config/debug.yaml 

from rich import print
import datasets as ds
import model_utils as mu
import transformers as tr
import os
import time
import torch
from omegaconf import OmegaConf
torch.set_float32_matmul_precision("medium")
import argparse as ap
from rich.logging import RichHandler
import logging

IS_RANK_0 = os.environ.get("RANK", "0") == "0"

logging.basicConfig(
    level="INFO" if IS_RANK_0 else "ERROR",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

logger.info(f"Env vars: {os.environ}")

def main(config_path: str, output_name: str = "model"):
    config = OmegaConf.load(config_path)
    
    arguments = tr.Seq2SeqTrainingArguments(
        output_dir=f"./output/{output_name}", 
        overwrite_output_dir=True,
        remove_unused_columns=False,
        do_train=True, do_eval=True,
        evaluation_strategy='steps', 
        eval_steps=500,
        logging_strategy='steps', 
        logging_steps=10,
        predict_with_generate=True, 
        prediction_loss_only=False, 
        per_device_train_batch_size=config.batch.train_size, 
        per_device_eval_batch_size=config.batch.eval_size,
        gradient_accumulation_steps=config.batch.accumulation,
        learning_rate=config.train.lr, 
        weight_decay=config.train.weight_decay,
        max_grad_norm=config.train.max_grad_norm, 
        max_steps=config.train.max_steps,
        lr_scheduler_type=config.train.lr_scheduler, 
        warmup_ratio=config.train.warmup_ratio,
        save_strategy='steps', 
        save_steps=500, 
        save_safetensors=True, 
        group_by_length=True, length_column_name='length',
    )

    logging.info(f"Running with arguments: {arguments}")

    logging.info(f"Loading dataset from {config.misc.dataset}")
    with arguments.main_process_first():
        t0 = time.perf_counter()
        dataset = ds.Dataset.from_json(config.misc.dataset)
        t_elapsed = time.perf_counter() - t0
        logging.info(f"Dataset Loaded in {t_elapsed:.4f} seconds")
    
    if config.misc.subsample:
        logger.info(f"Subsampling to {config.misc.subsample_size} examples")
        dataset = dataset.select(range(config.misc.subsample_size))

    logging.info(f"Splitting into train-dev")
    dataset = dataset.train_test_split(test_size=config.misc.dev_size, seed=config.misc.seed)
    
    effective_batch_size = arguments.per_device_train_batch_size * arguments.gradient_accumulation_steps
    n_epochs = (effective_batch_size * arguments.max_steps * arguments.n_gpu) / len(dataset['train'])
    logger.info(f"Effective # of epochs: {n_epochs:.4f}")

    logger.info(f"Loading model {config.misc.base_model_name}")
    t0 = time.perf_counter()
    model = tr.AutoModelForSeq2SeqLM.from_pretrained(config.misc.base_model_name)
    tokenizer = tr.AutoTokenizer.from_pretrained(config.misc.base_model_name)
    t_elapsed = time.perf_counter() - t0
    logger.info(f"Model loaded in {t_elapsed:.4f} seconds")
    
    
    tokenize_fn = mu.get_tokenize_fn(
        tokenizer=tokenizer,
        kind=config.misc.kind
    )

    with arguments.main_process_first():
        logger.info("Tokenizing")
        dataset = dataset.map(
            tokenize_fn, 
            batched=True, 
            batch_size=1024, 
            num_proc=20,
            remove_columns=dataset['train'].column_names
        )
        
        dataset = dataset.map(
            mu.length_fn,
            batched=True,
            batch_size=1024,
            num_proc=20
        )
    
    
    data_collator = mu.KIND_TO_DATACOLLATOR[config.misc.kind](tokenizer, padding=True, pad_to_multiple_of=8, max_length=1024)
    
    import numpy as np
    def compute_metrics_text(tokenizer):
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
            acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
    
            return {'accuracy': acc}
    
        return compute_metrics
    
    trainer = mu.TaskPrefixTrainer(
        alpha1=config.misc.alpha1,
        alpha2=config.misc.alpha2,
        kind=config.misc.kind,
        data_collator=data_collator,
        model=model,
        args=arguments,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_text(tokenizer)
    )

    logger.info("Training")
    
    trainer.train()
    
    def do_test(idx: int = 0, n: int = 2):
        examples = dataset['test'][idx:idx+n]
        result = []
        for i, ex in enumerate(examples['input_ids']):
            result.append(f"{i} => {tokenizer.decode(ex)} => [{tokenizer.decode(examples['labels'][i])}]")
    
        examples.pop('labels')
        examples.pop("length")
        if config.misc.kind != "1way":
            examples.pop("expl_input_ids")
            examples.pop("expl_labels")
            examples.pop("expl_attention_mask")
            if config.misc.kind == "3way":
                examples.pop("antiexpl_input_ids")
                examples.pop("antiexpl_labels")
                examples.pop("antiexpl_attention_mask")
        
        examples = tokenizer.pad(examples, return_attention_mask=True, return_tensors='pt')
        
        with torch.inference_mode():
            examples = {
                k: v.to(model.device) for k, v in examples.items()
            }
            outputs = model.generate(**examples)
            outputs = outputs.cpu().numpy()
    
        for i, out in enumerate(outputs):
            result[i] += f" *Predicted*: [{tokenizer.decode(out, skip_special_tokens=True)}]"
    
        for item in result:
            print(item)
    
    do_test(0, 10)


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="train.py")
    parser.add_argument("-c", "--config", type=str, help="Path to yaml config")
    parser.add_argument("-o", "--output", type=str, default="model", help="Output name for model")
    args = parser.parse_args()
    main(args.config, args.output)