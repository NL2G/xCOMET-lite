from statistics import mean

import scipy.stats as stats
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from tqdm import tqdm
import datetime
from transformers import get_scheduler

from MT0Regressor import MT0Regressor, Args

from accelerate import Accelerator


def main(model, dataloader_train, dataloader_eval, optimizer,
         scheduler, metric, num_epochs, num_training_steps):
    accelerator = Accelerator(log_with='wandb')
    accelerator.init_trackers(
        project_name="t5regressor", 
        init_kwargs={'entity': 'airi23-efficient-llm-metrics'}
    )

    model, optimizer, dataloader_train, scheduler = accelerator.prepare(model, optimizer, dataloader_train, scheduler)

    accelerator.print(f"Train size: {len(dataloader_train)}")
    accelerator.print(f"Eval size: {len(dataloader_eval)}")

    progress_bar_train = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    progress_bar_eval = tqdm(range(num_epochs * len(dataloader_eval)), disable=not accelerator.is_local_main_process)

    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        accelerator.print(f"TRAIN EPOCH {epoch + 1}")
        model.train()
        for batch in dataloader_train:
            #batch = {k: v.to(device) for k, v in batch.items()}
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

        accelerator.print("EVAL")
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

                for i in range(8):
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
        
        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        accelerator.save(model.state_dict(), f"model_large.pth")


if __name__ == "__main__":
    

    model_encoder_name = "bigscience/mt0-large"
    #device = "cuda:0"

    config = Args(
        encoder_name=model_encoder_name,
        sizes_mlp=[1024, 192, 48, 1],
        hidden_act=nn.Tanh,
        dropout_coef=0.1,
        need_lora=False,
        output_act=nn.Sigmoid,
        loss_fc=nn.MSELoss,
    )

    model = MT0Regressor(config)
    #model.to(device)

    print("Model successfully loaded.")

    dataloader_train = torch.load('dataloader_train_large.pth')
    dataloader_eval = torch.load('dataloader_eval_large.pth')

    print("DataLoaders successfully loaded.")

    optimizer = AdamW(model.parameters(), lr=3e-4)

    num_epochs = 3
    num_training_steps = num_epochs * len(dataloader_train)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    metric = nn.MSELoss()

    main(model=model, dataloader_train=dataloader_train, dataloader_eval=dataloader_eval, optimizer=optimizer,
         scheduler=lr_scheduler, metric=metric, num_epochs=num_epochs, num_training_steps=num_training_steps)
