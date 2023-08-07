from fnmatch import translate
import re
from datasets import load_dataset
import torch
import pandas as pd
import transformers as tr
from rich.logging import RichHandler
import argparse as ap
import logging
import xxhash

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="opus100", help="Dataset to use for translation"
    )
    parser.add_argument(
        "--sample", type=int, default=1000, help="Number of samples to translate"
    )
    parser.add_argument(
        "--ds-config", type=str, default="en-he", help="Configuration for the dataset"
    )
    parser.add_argument(
        "--ds_src_lang", type=str, default="he", help="Source language in the dataset"
    )
    parser.add_argument(
        "--ds_tgt_lang", type=str, default="en", help="Target language in the dataset"
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="heb_Hebr",
        help="Source language for translation",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="eng_Latn",
        help="Target language for translation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/nllb-moe-54b",
        help="Model to use for translation",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=8,
        choices=[4, 8, 16, 32],
        help="Number of bits to quantize to",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for translation"
    )
    parser.add_argument(
        "--max_new_length",
        type=int,
        default=512,
        help="Maximum length of the generated sequence",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data",
        help="Path to save the translations in csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--auto_device_map",
        action="store_true",
        default=False,
        help="Use auto device map",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.ds_config, split="train")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.sample))
    model_kwargs = {}
    if args.nbits == 8:
        model_kwargs["quantization_config"] = tr.BitsAndBytesConfig(load_in_8bit=True)
        dtype = torch.float16
    elif args.nbits == 4:
        model_kwargs["quantization_config"] = tr.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",
        )
        dtype = torch.bfloat16
    elif args.nbits == 16:
        dtype = torch.bfloat16
    elif args.nbits == 32:
        dtype = torch.float32
    else:
        raise ValueError("Invalid number of bits")

    if args.auto_device_map:
        model_kwargs["device_map"] = "auto"

    model_kwargs["torch_dtype"] = dtype

    logger.info(f"Loading model {args.model} with {args.nbits} bits")
    tokenizer = tr.AutoTokenizer.from_pretrained(args.model)
    model = tr.AutoModelForSeq2SeqLM.from_pretrained(args.model, **model_kwargs)

    logger.info("Preparing dataset")
    dataset = dataset.map(
        lambda x: {
            "src": x["translation"][args.ds_src_lang],
            "tgt": x["translation"][args.ds_tgt_lang],
        },
        remove_columns=["translation"],
    )
    logger.info("Sorting dataset by length")
    dataset = dataset.map(lambda x: {"len": len(tokenizer(x["src"])["input_ids"])})
    dataset = dataset.sort(["len"], reverse=True)
    dataset = dataset.remove_columns(["len"])

    def translate(examples):
        inputs = tokenizer._build_translation_inputs(
            raw_inputs=examples["src"],
            return_tensors="pt",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            padding="longest",
            truncation=True,
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=args.max_new_length)
        return {"mt": tokenizer.batch_decode(outputs, skip_special_tokens=True)}

    logger.info(f"Translating {len(dataset)} samples")
    _new_fingerprint = f'{dataset._fingerprint}#{args.dataset}-{args.ds_config}-{args.sample}-{args.src_lang}-{args.tgt_lang}-{args.model.replace("/", "#")}-{args.nbits}'
    _new_fingerprint_hash = xxhash.xxh64(_new_fingerprint).hexdigest()
    logger.info(f"New fingerprint: {_new_fingerprint} => {_new_fingerprint_hash}")
    dataset = dataset.map(
        translate,
        batched=True,
        batch_size=args.batch_size,
        desc="Translating",
        new_fingerprint=_new_fingerprint_hash,
    )

    logger.info(f"Saving translations to {args.save_path}")
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.add_column('model_name', [args.model] * len(dataset))
    dataset.to_pandas().to_csv(args.save_path, index=False, encoding="utf-8")
