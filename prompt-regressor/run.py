import argparse as ap
import torch
from torch.utils.data import DataLoader
import accelerate as acc
import numpy as np
from scipy.stats import kendalltau
import transformers as tr
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import peft
import datasets as ds
from transformers.activations import ACT2FN
import logging
import os
from tqdm.auto import tqdm

local_rank = int(os.environ.get("LOCAL_RANK", 0))
logging.basicConfig(
    level=logging.INFO if local_rank in [-1, 0] else logging.CRITICAL,
    format="%(asctime)s || %(levelname)s || %(message)s",
    datefmt="[%X]",
)

logger = logging.getLogger(__name__)

TEMPLATE: str = """Language-pair: {lp}

Score type: {score_type}

Source: [{src}]

Reference: [{ref}]

Hypothesis: [{mt}]"""

def get_tokenize_fn(template: str, tokenizer: tr.PreTrainedTokenizerFast, max_length: int) -> callable:
    def tokenize_fn(example: dict[str, str]) -> dict[str, np.ndarray]:
        score = example.pop("score")
        inputs = template.format(**example)
        tokenized = tokenizer(
            inputs,
            max_length=max_length,
            truncation='longest_first',
        )
        tokenized["labels"] = score
        tokenized["len"] = len(tokenized["input_ids"])
        return tokenized

    return tokenize_fn


def get_optimizer(
        model: tr.PreTrainedModel,
        optim: str, 
        lr: float,
        betas: tuple[float, float],
        weight_decay: float, 
        eps: float,
        use_lora: bool = False,
    ) -> torch.optim.Optimizer:

    logger.info(f"Using optimizer: {optim} with lr={lr}, betas={betas}, weight_decay={weight_decay}, eps={eps}")

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optim in ['adamw', 'adamw_fused']:
        from torch.optim import AdamW

        optimizer_kwargs = {
            "lr": lr,
            "eps": eps,
            "betas": betas,
        }

        if optim == 'adamw_fused':
            optimizer_kwargs["fused"] = True

        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    elif '8bit' in optim or 'lion' in optim:
        from bitsandbytes.optim import AdamW, Lion

        is_paged = 'paged' in optim
        is_lion = 'lion' in optim

        if is_lion:
            optim_cls = Lion
        else:
            optim_cls = AdamW

        optimizer_kwargs = {
            "lr": lr,
            "is_paged": is_paged,
            "optim_bits": 8,
            "betas": betas,
        }
        if not is_lion:
            optimizer_kwargs["eps"] = eps

        optimizer = optim_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if not use_lora:
            from bitsandbytes.optim import GlobalOptimManager
            GlobalOptimManager.get_instance().register_module_override(
                model.transformer.word_embeddings, 'weight', {'optim_bits': 32}
            )

    else:
        raise ValueError(f"Invalid optimizer: {optim}")
    
    logger.info(f"Initialized optimizer: {optimizer}")
    return optimizer


def evaluate(model, eval_dataloader, accelerate: acc.Accelerator, epoch: int, id2lp: dict[int, str], activation: callable):
    total_lps = []
    total_eval_labels = []
    total_eval_preds = []
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", disable=not accelerate.is_main_process)):
            lps = batch.pop("lp")
            labels = batch.pop("labels")
            outputs = model(**batch)
            
            loss = torch.nn.functional.mse_loss(
                activation(outputs.logits.squeeze()),
                labels.squeeze(),
            )

            preds = activation(outputs.logits.squeeze())
            preds, labels, loss, lps = accelerate.gather_for_metrics(
                (preds, labels, loss, lps)
            )
            total_loss.append(loss.mean().item())
            total_eval_labels += labels.cpu().tolist()
            total_eval_preds += preds.cpu().tolist()
            total_lps += lps.cpu().tolist()
    
    total_eval_labels = np.array(total_eval_labels)
    total_eval_preds = np.array(total_eval_preds)
    language_pairs = np.array([id2lp[x] for x in total_lps])
    logger.info(f"LPS: {set(language_pairs)}")
    total_kt = kendalltau(total_eval_labels, total_eval_preds).statistic
    lp_kt_results = {}
    for lp in set(language_pairs):
        lp_mask = language_pairs == lp
        lp_kt_results[lp] = kendalltau(total_eval_labels[lp_mask], total_eval_preds[lp_mask]).statistic

    eval_mse = np.mean(total_loss)
    eval_rmse = np.sqrt(eval_mse)

    logger_msg = f"Epoch {epoch:^8} | Eval MSE {eval_mse:^15} | Eval RMSE {eval_rmse:^15} | Eval KT {total_kt:^15} | "
    for lp, kt in lp_kt_results.items():
        logger_msg += f"{lp}: {kt:^10} | "

    logger.info(logger_msg.strip())

    logger_values = {
        "eval/mse": eval_mse,
        "eval/rmse": eval_rmse,
        "eval/kt": total_kt,
    }
    for lp, kt in lp_kt_results.items():
        logger_values[f"eval/kt_{lp}"] = kt

    accelerate.log(logger_values)




def init_model(model_name: str, n_bits: int, use_lora: bool):
    
    logger.info(f"Loading model {model_name} with {n_bits}-bit quantization with LoRA={use_lora}")

    if n_bits == 4 and not use_lora:
        raise ValueError("4-bit quantization is not supported without LoRA")
    
    model_kwargs = {}
    if n_bits == 4:
        config = tr.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = config
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif n_bits == 8:
        config = tr.BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["quantization_config"] = config
        model_kwargs["torch_dtype"] = torch.float16

    elif n_bits == 16:
        model_kwargs['torch_dtype'] = torch.bfloat16

    elif n_bits == 32:
        model_kwargs['torch_dtype'] = torch.float32

    else:
        raise ValueError(f"Invalid number of bits: {n_bits}")
    
    model = tr.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_kwargs,
        #device_map='auto',
        num_labels=1
    )

    #logger.error(f"[]===========> Device Map: {model.hf_device_map} <=============]")

    if n_bits in {4, 8}:
        #model.gradient_checkpointing_enable()
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if use_lora:
        lora_config = peft.LoraConfig(
            task_type=peft.TaskType.SEQ_CLS, lora_dropout=0.05, bias='lora_only', r=64, lora_alpha=32
        )
        logger.info(f"Using LoRA with config: {lora_config}")
        model = peft.get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    logger.info("Model loaded")
    logger.info(f"Model summary: {model}")

    return model



def main():
    parser = ap.ArgumentParser(
        prog="run.py",
        description="Train a model",
    )

    parser.add_argument("--model", type=str, default="bigscience/bloomz-1b1")

    parser.add_argument(
        "--use-lora",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="linear",
        choices=list(ACT2FN.keys()),
    )

    parser.add_argument(
        "--subsample-train",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--subsample-val",
        type=int,
        default=-1
    )

    parser.add_argument("--n-bits", type=int, choices=[4, 8, 16, 32])

    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
    )

    parser.add_argument("--dataset", type=str, default="nllg/wmt-metrics-data")

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--use-tf32",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--scale-lr",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--load-from-checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoint/",
    )

    parser.add_argument(
        "--group-name",
        type=str,
        default='default-group',
    )

    parser.add_argument("--save-path", type=str, default="./model/")

    args = parser.parse_args()

    # initializing accelerate

    if args.use_lora:
        kwargs = {
            'kwargs_handlers': [
                acc.DistributedDataParallelKwargs(find_unused_parameters=True)
            ]
        }
    else:
        kwargs = {}

    accelerate = acc.Accelerator(
        log_with='wandb',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **kwargs
    )
    accelerate.init_trackers(
        project_name="wmt23-prompt-regressor",
        config=vars(args),
        init_kwargs={
            'wandb': {
                'group': args.group_name,
            }
        }
    )

    logger.info(f"Arguments: {args}")

    if args.use_tf32:
        logger.info("Using TF32")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set seed
    acc.utils.set_seed(args.seed)

    # scaling learning rate
    if args.scale_lr:
        logger.info(f"Scaling learning rate according to number of processes: {args.lr} * {accelerate.num_processes} = {args.lr * accelerate.num_processes}")
        args.lr *= accelerate.num_processes

    # Load data
    logger.info("Loading data")
    with accelerate.main_process_first():
        dataset = ds.load_dataset(args.dataset)
        dataset = dataset.map(lambda x: {'mt': x['mt'] if x['mt'] is not None else ''}, num_proc=10)
        tokenizer = tr.AutoTokenizer.from_pretrained(args.model)
        tokenize_fn = get_tokenize_fn(
            TEMPLATE,
            tokenizer,
            args.max_length,
        )
        dataset = dataset.map(
            tokenize_fn,
            batched=False,
            num_proc=10,
            remove_columns=["mt", "src", "ref", "score", "score_type"],
        )

        dataset['train'] = dataset['train'].remove_columns(['lp'])
        test_lps = set(dataset['test']['lp'])
        lp2id = {lp: i for i, lp in enumerate(test_lps)}
        id2lp = {i: lp for i, lp in enumerate(test_lps)}
        dataset['test'] = dataset['test'].map(lambda x: {'lp': lp2id[x['lp']]}, num_proc=10)

        # Subsampling
        if args.subsample_train > 0:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(range(args.subsample_train))

        if args.subsample_val > 0:
            dataset['test'] = dataset['test'].shuffle(seed=args.seed).select(range(args.subsample_val))

        #dataset = dataset.sort(column_names=["len"])#, reverse=True)
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.remove_columns(["len"])

    accelerate.wait_for_everyone()

    # Load model
    logger.info("Loading model")
    with accelerate.main_process_first():
        model = init_model(args.model, args.n_bits, args.use_lora)

    accelerate.wait_for_everyone()

    # Initialize optimizer
    optimizer = get_optimizer(
        model=model, optim=args.optim,
        lr=args.lr, eps=args.eps, 
        weight_decay=args.weight_decay, 
        betas=(args.beta1, args.beta2),
        use_lora=args.use_lora,
    )

    # Initialize scheduler
    num_training_steps = int(args.epochs * (len(dataset['train']) / (args.batch_size * args.gradient_accumulation_steps)))
    logger.info(f"Total num of training steps: {num_training_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerate.num_processes}")
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = tr.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Initialize collator
    collator = tr.DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=args.max_length, pad_to_multiple_of=8, return_tensors="pt"
    )

    # Initialize data loader

    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=args.batch_size,
        num_workers=1,
        collate_fn=collator,
    )

    eval_dataloader = DataLoader(
        dataset['test'],
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        sampler=torch.utils.data.SequentialSampler(dataset['test']),
        collate_fn=collator,
    )

    # putting model on accelerator
    logger.info("Putting model on accelerator")
    accelerate.wait_for_everyone()

    model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerate.prepare(
        model, optimizer, scheduler, train_dataloader, eval_dataloader
    )

    # Load from checkpoint
    if args.load_from_checkpoint is not None:
        logger.info(f"Loading model from {args.load_from_checkpoint}")
        accelerate.load_state(args.load_from_checkpoint)

    logging_steps = int(args.log_every)
    logger.info(f"Logging every {logging_steps} steps")

    checkpoint_steps = int(args.checkpoint_every)
    logger.info(f"Checkpointing every {checkpoint_steps} steps")

    activation = ACT2FN[args.activation]

    logger.info("Starting training")
    accelerate.wait_for_everyone()
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}")
        model.train()
        for i, batch in enumerate(tqdm(
            train_dataloader, 
            mininterval=30, 
            maxinterval=120,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerate.is_main_process,
        )):
            with accelerate.accumulate(model):
                labels = batch.pop("labels")
                outputs = model(**batch)

                loss = torch.nn.functional.mse_loss(
                    activation(outputs.logits.squeeze()),
                    labels.squeeze(),
                )

                accelerate.backward(loss)
                if accelerate.sync_gradients:
                    accelerate.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (i + 1) % logging_steps == 0:
                    mse = loss.item()
                    rmse = np.sqrt(mse)
                    logger.info(f"Epoch {epoch + 1:^4} | Step {i:^5} out of {len(train_dataloader):^5} | MSE {mse:^10.6f} | RMSE {rmse:^10.6f}")
                    accelerate.log({
                        "train/mse": mse,
                        "train/rmse": rmse,
                        "train/lr": scheduler.get_last_lr()[0],
                    })

                if (i + 1) % checkpoint_steps == 0:
                    logger.info(f"Checkpointing model at step {i}")
                    accelerate.save_state(output_dir=args.checkpoint_path)

                if (i + 1) % args.eval_every == 0:
                    logger.info(f"Evaluating model at step {i}")
                    accelerate.wait_for_everyone()
                    evaluate(
                        model=model,
                        eval_dataloader=eval_dataloader, 
                        accelerate=accelerate, 
                        id2lp=id2lp, 
                        epoch=epoch + 1,
                        activation=activation,
                    )
                    model.train()
                    accelerate.wait_for_everyone()

        accelerate.wait_for_everyone()


    logger.info("Final evaluation")
    evaluate(
        model=model, 
        eval_dataloader=eval_dataloader, 
        accelerate=accelerate, 
        id2lp=id2lp, 
        epoch="Last",
        activation=activation,
    )
    accelerate.wait_for_everyone()

    accelerate.end_training()
    logger.info("Finished training, saving model")
    accelerate.save_state(output_dir=args.checkpoint_path)
    if accelerate.is_main_process:
        model = accelerate.unwrap_model(model)
        model.save_pretrained(args.save_path, safe_serialization=True)
        if args.use_lora:
            model.create_or_update_model_card(args.save_path)
        tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()