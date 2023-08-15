import transformers as tr
import peft
import datasets as ds
import torch
from transformers.activations import ACT2FN
import argparse as ap
import logging
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import kendalltau
import os

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

def load_model(path: str, is_peft: bool = True, n_bits: int = 4):

    logger.info(f'Loading model from {path} with {n_bits} bits per weight and {is_peft} as PEFT')

    if is_peft:
        config = peft.PeftConfig.from_pretrained(path)
        model_name = config.base_model_name_or_path

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
        model: peft.LoraModel = peft.PeftModelForSequenceClassification.from_pretrained(
            model, model_id=path
        )
        model = model.merge_and_unload(progressbar=True)

    logger.info(f'Loaded model {model}')

    return model

def evaluate(model, eval_dataloader, epoch: int, id2lp: dict[int, str], activation: callable):
    total_lps = []
    total_eval_labels = []
    total_eval_preds = []
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            lps = batch.pop("lp")
            labels = batch.pop("labels")
            outputs = model(**batch)
            
            loss = torch.nn.functional.mse_loss(
                activation(outputs.logits.squeeze()),
                labels.squeeze(),
            )

            preds = activation(outputs.logits.squeeze())
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
        default=4,
        choices=[4, 8, 16, 32]
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512
    )
    parser.add_argument(
        "--activation",
        type=str,
        default='linear',
        choices=list(ACT2FN.keys())
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.path, n_bits=args.n_bits, is_peft=args.use_lora)
    tokenizer = tr.AutoTokenizer.from_pretrained(args.path)
    if args.n_bits not in [4, 8]:
        model = model.to(device)

    model.eval()

    logger.info(f'Loaded tokenizer {tokenizer}')
    logger.info(f'Loading dataset {args.dataset}')
    dataset = ds.load_dataset(args.dataset, split='test')
    dataset = dataset.map(lambda x: {'mt': x['mt'] if x['mt'] is not None else ''}, num_proc=10)
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
    dataset = dataset.sort(['len'])
    dataset = dataset.remove_columns(['len'])

    collate_fn = tr.DataCollatorWithPadding(
        tokenizer, padding='longest',
        max_length=args.max_length, 
        pad_to_multiple_of=8,
        return_tensors='pt'
    )
    activation_fn = ACT2FN[args.activation]
    logger.info(f'Using activation function {activation_fn}')

    logger.info(f'Computing scores for {len(dataset)} examples')

    id2lp = {i: lp for i, lp in enumerate(set(dataset['lp']))}
    lp2id = {lp: i for i, lp in id2lp.items()}
    dataset = dataset.map(lambda x: {'lp': lp2id[x['lp']]}, num_proc=10)

    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    
    evaluate(model, eval_dataloader, 0, id2lp, activation_fn)
        

if __name__ == '__main__':
    main()



