import json
import numpy as np
import pandas as pd

def dump_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_tsv(path):
    data = pd.read_csv(path, sep="\t")
    data.index = np.arange(len(data))
    data = data.drop(columns=["Unnamed: 0"])
    return data