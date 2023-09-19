from statistics import mean

import scipy.stats as stats
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
import argparse as ap
import os
from tqdm import tqdm
from datasets import load_from_disk
import datetime
from transformers import get_scheduler
from transformers import DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler

from MT0Regressor import MT0Regressor, Args

from accelerate import Accelerator

CHEKPOINT_EVERY: int = 3000
USE_LABSE: bool = True
BATCH_SIZE: int = 8


def eval(accelerator, model, dataloader_eval, progress_bar_eval, metric, postfix: str = ""):
    accelerator.print(f"EVAL {postfix}")
    model.eval()
    mse_metrics = []
    predicted = []
    labels = []
    if accelerator.is_local_main_process:
        for batch in dataloader_eval:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

                logits = outputs[1]

                mse_metric = metric(logits, batch["labels"]).item()

            for i in range(len(logits)):
                predicted.append(logits[i].item())
                labels.append(batch["labels"][i].item())
            mse_metrics.append(mse_metric)
            progress_bar_eval.set_postfix({"loss": mse_metric})
            progress_bar_eval.update(1)

        eval_mse = mean(mse_metrics)
        eval_kendall = stats.kendalltau(predicted, labels).statistic
        accelerator.print(f"Eval MSE: {eval_mse}")
        accelerator.print(f"Eval Kendall tau-b: {eval_kendall}")
        accelerator.log({"eval/mse": eval_mse, "eval/kendall": eval_kendall})

    model.train()

def main(model, dataloader_train, dataloader_eval, optimizer,
         scheduler, metric, num_epochs, accelerator, save_file_name):

    model, optimizer, dataloader_train, scheduler = accelerator.prepare(model, optimizer, dataloader_train, scheduler)

    num_local_training_steps = num_epochs * len(dataloader_train)
    accelerator.print(f"Train size: {len(dataloader_train)}")
    accelerator.print(f"Eval size: {len(dataloader_eval)}")

    progress_bar_train = tqdm(range(num_local_training_steps), disable=not accelerator.is_local_main_process)
    progress_bar_eval = tqdm(range(num_epochs * len(dataloader_eval)), disable=not accelerator.is_local_main_process)

    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        accelerator.print(f"TRAIN EPOCH {epoch + 1}")
        model.train()
        for i, batch in enumerate(dataloader_train):
            outputs = model(**batch)

            loss = outputs[2]

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar_train.set_postfix(
                {"loss": loss.item(), "logits": outputs[1][1].item()}
            )
            progress_bar_train.update(1)

            accelerator.log({"loss": loss.item()})

            if (i + 1) % CHEKPOINT_EVERY == 0:
                if accelerator.is_local_main_process:
                    accelerator.save(model.state_dict(), save_file_name)
            accelerator.wait_for_everyone()

        eval(accelerator, model, dataloader_eval, progress_bar_eval, metric, f"EPOCH {epoch + 1} # End")
        
        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        accelerator.save(model.state_dict(), save_file_name)


if __name__ == "__main__":

    parser: ap.ArgumentParser = ap.ArgumentParser(
        prog='train.py',
    )
    parser.add_argument(
        "--no-use-labse", action='store_true', default=False
    )
    parser.add_argument(
        "--save-file-name", default="model-large.pth", type=str,
    )
    parser.add_argument(
        "--model-name", type=str, default="bigscience/mt0-large"
    )
    parser.add_argument(
        "--batch-size", default=8, type=int
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None
    )

    args: ap.Namespace = parser.parse_args()

    accelerator = Accelerator(log_with='wandb')
    accelerator.init_trackers(
        project_name="t5regressor", 
        init_kwargs={'entity': 'airi23-efficient-llm-metrics'}
    )

    accelerator.print(f"Running with args: {args}")

    model_encoder_name = args.model_name

    use_labse = not args.no_use_labse

    config = Args(
        encoder_name=model_encoder_name,
        size_labsell=512,
        sizes_mlp=[384, 96, 1], # starting from 2th linear layer; 1th layer size 1024 + 512 = 1536
        hidden_act=nn.Tanh,
        dropout_coef=0.1,
        need_lora=False,
        output_act=nn.Sigmoid,
        loss_fc=nn.MSELoss,
        use_labse=use_labse,
        checkpoint=args.checkpoint
    )

    model = MT0Regressor(config)

    accelerator.print(f"Loaded model {model}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_encoder_name)

    dataset = load_from_disk("./wmt-labse")

    if not use_labse:
        accelerator.print("Dropping LaBSE columns")
        dataset = dataset.remove_columns(['labse'])

    collator_fn = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=512, return_tensors="pt", pad_to_multiple_of=8
    )
    with accelerator.main_process_first():
        lenghts = dataset['train'].map(lambda x: {'len': len(x['input_ids'])}, batched=False, num_proc=20)['len']
        
    length_sampler = LengthGroupedSampler(batch_size=args.batch_size, dataset=dataset['train'], lengths=lenghts)
    dataloader_train = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        sampler=length_sampler,
        collate_fn=collator_fn
    )
    dataloader_eval = DataLoader(
        dataset["test"],
        batch_size=args.batch_size,
        collate_fn=collator_fn,
        shuffle=False
    )

    accelerator.print("DataLoaders successfully loaded.")

    optimizer = AdamW(model.parameters(), lr=(5e-5*accelerator.num_processes))

    num_epochs = 1
    num_training_steps = num_epochs * len(dataloader_train)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    metric = nn.MSELoss()

    accelerator.print("Starting training =>")
    main(
        model=model, 
        dataloader_train=dataloader_train,
        dataloader_eval=dataloader_eval, 
        optimizer=optimizer,
        scheduler=lr_scheduler, 
        metric=metric, 
        num_epochs=num_epochs, 
        accelerator=accelerator,
        save_file_name=args.save_file_name
    )
