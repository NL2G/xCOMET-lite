DATA_CONFIG = {
    "comet": {
        "train": [
            # {
            #    "path": "./data/2017-da.tar.gz",
            #    "lps": "all",
            # },
            # {
            #    "path": "./data/2018-da.tar.gz",
            #    "lps": "all",
            # },
            # {
            #    "path": "./data/2019-da.tar.gz",
            #    "lps": "all",
            # },
            {
                "path": "./data/2020-da.tar.gz",
                "lps": "all",
            }
        ],
        "test": {
            "en-ru": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "en-ru",
                ],
            },
            "en-de": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "en-de",
                ],
            },
            "zh-en": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "zh-en",
                ],
            },
        },
    },
    "cometinho": {
        "train": [
            {
                "path": "./data/cometinho-0.csv.gz",
                "lps": "all",
            },
            {
                "path": "./data/cometinho-1.csv.gz",
                "lps": "all",
            },
        ],
        "test": {
            "en-ru": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "en-ru",
                ],
            },
            "en-de": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "en-de",
                ],
            },
            "zh-en": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "zh-en",
                ],
            },
        },
    },
    "debug": {
        "train": [
            # {
            #    "path": "./data/2017-da.tar.gz",
            #    "lps": "all",
            # },
            # {
            #    "path": "./data/2018-da.tar.gz",
            #    "lps": "all",
            # },
            # {
            #    "path": "./data/2019-da.tar.gz",
            #    "lps": "all",
            # },
            {
                "path": "./data/2020-da.tar.gz",
                "lps": ["en-ru"],
            }
        ],
        "test": {
            "en-ru": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "en-ru",
                ],
            },
            "en-de": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "en-de",
                ],
            },
            "zh-en": {
                "path": "./data/wmt21-newstest.csv",
                "lps": [
                    "zh-en",
                ],
            },
        },
    },
}

TRAINING_CONFIG = {
    "comet": {
        "nr_frozen_epochs": 0.3,
        "keep_embeddings_freezed": True,
        "encoder_lr": 1.0e-6,
        "estimator_lr": 1.5e-5,
        "layerwise_decay": 0.95,
        "encoder_model_name": "xlm-roberta-large",
        "layer": "mix",
        "batch_size": 16,
        "hidden_sizes": [3072, 1024],
        "activations": "Tanh",
        "final_activation": None,
        "layer_transformation": "sparsemax",
        "max_epochs": 4,
        "patience": 2,
        "dropout": 0.1,
    },
    "cometinho": {
        "nr_frozen_epochs": 0.0,
        "keep_embeddings_freezed": True,
        "encoder_lr": 6.0e-5,
        "estimator_lr": 18.6e-5,
        "layerwise_decay": 0.95,
        "encoder_model_name": "microsoft/Multilingual-MiniLM-L12-H384",
        "layer": 12,
        "batch_size": 48,
        "hidden_sizes": [384],
        "activations": "Tanh",
        "final_activation": None,
        "layer_transformation": "sparsemax",
        "max_epochs": 4,
        "patience": 2,
        "dropout": 0.1,
    },
    "debug": {
        "nr_frozen_epochs": 0.1,
        "keep_embeddings_freezed": True,
        "encoder_lr": 1.0e-6,
        "estimator_lr": 1.5e-5,
        "layerwise_decay": 0.95,
        "encoder_model_name": "microsoft/Multilingual-MiniLM-L12-H384",
        "layer": "mix",
        "batch_size": 32,
        "hidden_sizes": [3072, 1024],
        "activations": "Tanh",
        "final_activation": None,
        "layer_transformation": "sparsemax",
        "max_epochs": 10,
        "patience": 2,
        "dropout": 0.1,
    },
}
