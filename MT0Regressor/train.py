from statistics import mean

import scipy.stats as stats
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler

from MT0Regressor import MT0Regressor, Args

from accelerate import Accelerator


def main(model, dataloader_train, dataloader_eval, optimizer,
         scheduler, metric, progress_bar_train, progress_bar_eval):
    accelerator = Accelerator()

    model, optimizer, dataloader_train, scheduler = accelerator.prepare(model, optimizer, dataloader_train, scheduler)

    print(f"Train size: {len(dataloader_train)}")
    print(f"Eval size: {len(dataloader_eval)}")

    for epoch in range(num_epochs):
        print(f"TRAIN EPOCH {epoch + 1}")
        model.train()
        for batch in dataloader_train:
            batch = {k: v.to(device) for k, v in batch.items()}
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

            wandb.log({"loss": loss.item()})

        print("EVAL")
        model.eval()
        mse_metrics = []
        predicted = []
        labels = []
        for batch in dataloader_eval:
            batch = {k: v.to(device) for k, v in batch.items()}

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

        print(f"Eval MSE: {mean(mse_metrics)}")
        print(f"Eval Kendall tau-b: {stats.kendalltau(predicted, labels)[0]}")

    torch.save(model.state_dict(), "checkpoints/model_arc_simple_4.pt")


if __name__ == "__main__":
    wandb.login()
    wandb.init(entity="airi23-efficient-llm-metrics", project="t5regressor")

    model_encoder_name = "bigscience/mt0-large"
    device = "cuda:0"

    config = Args(
        encoder_name=model_encoder_name,
        sizes_mlp=[768, 192, 48, 1],
        hidden_act=nn.Tanh,
        dropout_coef=0.1,
        need_lora=False,
        output_act=nn.Sigmoid,
        loss_fc=nn.MSELoss,
    )

    model = MT0Regressor(config)
    model.to(device)

    print("Model successfully loaded.")

    dataloader_train = torch.load('dataloader_train.pth')
    dataloader_eval = torch.load('dataloader_eval.pth')

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
    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_eval = tqdm(range(num_epochs * len(dataloader_eval)))
    metric = nn.MSELoss()

    main(model=model, dataloader_train=dataloader_train, dataloader_eval=dataloader_eval, optimizer=optimizer,
         scheduler=lr_scheduler, metric=metric, progress_bar_train=progress_bar_train,
         progress_bar_eval=progress_bar_train)
