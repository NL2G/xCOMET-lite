#!/usr/bin/env python
# coding: utf-8

import os

import datasets as ds
import numpy as np

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bigscience/mt0-xxl")

rng = np.random.default_rng(seed=0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading dataset")
dataset = ds.load_dataset("nllg/mt-metric-synth-plus", "wmt-languages", split="train")


def score_to_quality_class_improved(example):
    # Adjusted function to handle scores outside 0 to 1 range
    score = example["score"]
    # Define the plain text phrases for each class
    quality_classes = [
        "Very Poor",  # Below 0 or 0.0 to 0.1
        "Poor",  # 0.1 to 0.2
        "Fair",  # 0.2 to 0.3
        "Below Average",  # 0.3 to 0.4
        "Average",  # 0.4 to 0.5
        "Above Average",  # 0.5 to 0.6
        "Good",  # 0.6 to 0.7
        "Very Good",  # 0.7 to 0.8
        "Excellent",  # 0.8 to 0.9
        "Outstanding",  # 0.9 to 1.0 or above
    ]

    if score < 0:
        return {"class": quality_classes[0]}  # Very Poor for scores below 0
    elif score > 1.0:
        return {"class": quality_classes[-1]}  # Outstanding for scores above 1.0

    # Calculate the index for scores within the range 0 to 1
    index = int(score * 10)
    index = min(
        index, 9
    )  # Ensure index does not go beyond the last element for score = 1.0

    return {"class": quality_classes[index]}


def generate_translation_error_explanations(example):
    mt_text = example["mt"]
    annotations = example["annotations"]
    explanations = []

    # Start with a brief introduction
    explanation_intro = f"Found {len(annotations)} translation error(s):"
    explanations.append(explanation_intro)

    for annotation in annotations:
        severity = annotation["severity"]
        error_text = annotation["text"]
        start = annotation["start"]
        end = annotation["end"]

        # Generate context with highlighted error text
        context_before = mt_text[max(0, start - 10) : start]
        context_after = mt_text[end : min(len(mt_text), end + 10)]
        highlighted_error = f"<b>{error_text}</b>"

        context = f"...{context_before}{highlighted_error}{context_after}..."

        # Add error description to explanations
        error_description = f'Severity: {severity.capitalize()}, Error: "{error_text}", Context: {context}'
        explanations.append(error_description)

    rationale_text = "\n".join(explanations)
    return {"rationale": rationale_text}



def smart_split(example):
    return {
        "_tokens": [
            word[1:]
            for word, _ in tokenizer.pre_tokenizer.pre_tokenize_str(example["mt"])
        ]
    }


def generate_variable_anti_explanations(example):
    mt_text = example["mt"]
    annotations = example["annotations"]
    words = example["_tokens"]
    if len(words) == 0:
        return {"antirationale": ""}

    num_anti_explanations = rng.integers(0, 5 + 1)

    explanations = [f"Found {num_anti_explanations} translation error(s):"]

    # Randomly choose the number of anti-explanations to generate, from 0 to 5

    # Possible severity levels
    severity_levels = ["Minor", "Major", "Critical"]

    # Collect spans of all annotated errors to exclude them
    error_indices = set()
    for annotation in annotations:
        start = annotation["start"]
        end = annotation["end"]
        for i in range(start, end + 1):
            error_indices.add(i)

    selected_indices = set()
    max_tries = 100
    while len(selected_indices) < num_anti_explanations:
        
        index = rng.integers(0, len(words) - 1)
        # Ensure the selected word is not part of any annotated error
        if index not in error_indices and index not in selected_indices:
            selected_indices.add(index)

        if max_tries == 0:
            break

        max_tries -= 1

    for index in sorted(list(selected_indices)):
        selected_word = words[index]
        start = max(0, index - 1)
        end = min(len(words), index + 2)
        context = " ".join(words[start:end]).replace(
            selected_word, f"<b>{selected_word}</b>"
        )
        severity = rng.choice(severity_levels)

        explanations.append(
            f'Severity: {severity}, Error: "{selected_word}", Context: ...{context}...'
        )

    # If no anti-explanations were generated, adjust the message
    if len(explanations) == 1:  # Only the initial message exists
        explanations = [""]

    return {"antirationale": "\n".join(explanations)}

dataset = dataset.shuffle(seed=0).remove_columns(["__index_level_0__"])

dataset = dataset.map(smart_split, num_proc=40)

dataset = dataset.map(score_to_quality_class_improved, num_proc=40)


dataset = dataset.map(generate_translation_error_explanations, num_proc=40)


dataset = dataset.map(generate_variable_anti_explanations, num_proc=40)


for i in range(10):
    print("===" * 30)
    print(dataset[i])


dataset.push_to_hub(
    "nllg/mt-metric-synth-plus",
    config_name="wmt-languages",
    num_shards=4,
    commit_message="Added class labels and rationales",
)
