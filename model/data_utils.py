import pandas as pd
import torch
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler


def load_data(
    path: str, lps: list[str] | str = "all", domain: str = "news"
) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    if lps != "all":
        if "lp" in dataset.columns:
            dataset = dataset[dataset["lp"].isin(lps)]
    if domain != "all":
        if "domain" in dataset.columns:
            dataset = dataset[dataset["domain"] == domain]

    # scaler = MinMaxScaler()
    dataset = dataset[["src", "mt", "ref", "score"]]
    # dataset["score"] = scaler.fit_transform(dataset[["score"]])
    return dataset


def load_from_config(
    train_config: list[dict[str, str | list[str]]],
    test_config: dict[str, str | list[str]],
    dev_size: float = 0.01,
    random_seed: int = 999,
) -> tuple[Dataset, Dataset, Dataset]:
    train_data = []
    for element in train_config:
        train_data.append(load_data(**element))
    train_data = pd.concat(train_data)
    train_dev = Dataset.from_pandas(train_data)
    train_dev = train_dev.train_test_split(test_size=dev_size, seed=random_seed)

    test_data = {}
    for key in test_config:
        test_data[key] = load_data(**test_config[key])

    test_data = {key: Dataset.from_pandas(value) for key, value in test_data.items()}

    return train_dev["train"], train_dev["test"], test_data


def make_preprocessing_fn(
    tokenizer, max_length: int = 512, return_length: bool = False
):
    def preprocessing_function(examples):
        src_inputs = tokenizer(
            [str(x) for x in examples["src"]],
            max_length=max_length,
            padding=False,
            truncation=True,
            return_length=return_length,
        )
        mt_inputs = tokenizer(
            [str(x) for x in examples["mt"]],
            max_length=max_length,
            padding=False,
            truncation=True,
            return_length=return_length,
        )
        ref_inputs = tokenizer(
            [str(x) for x in examples["ref"]],
            max_length=max_length,
            padding=False,
            truncation=True,
            return_length=return_length,
        )
        result = {
            "src_input_ids": src_inputs.input_ids,
            "mt_input_ids": mt_inputs.input_ids,
            "ref_input_ids": ref_inputs.input_ids,
        }
        if "score" in examples:
            result["labels"] = examples["score"]

        if return_length:
            result["src_length"] = src_inputs.length
            result["mt_length"] = mt_inputs.length
            result["ref_length"] = ref_inputs.length

        return result

    return preprocessing_function


def make_collate_fn(tokenizer, max_length: int = 512):
    def collate_fn(examples):
        batch = {}
        for segment in {"src", "mt", "ref"}:
            output = tokenizer.pad(
                [{"input_ids": i[f"{segment}_input_ids"]} for i in examples],
                padding="longest",
                max_length=max_length,
                pad_to_multiple_of=8,
                return_attention_mask=True,
                return_tensors="pt",
            )
            batch[f"{segment}_input_ids"] = output["input_ids"]
            batch[f"{segment}_attention_mask"] = output["attention_mask"]

        batch["labels"] = torch.tensor(
            [x["labels"] for x in examples], dtype=torch.float32
        )
        return batch

    return collate_fn
