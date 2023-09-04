SEED = 101
DATA_DIR = '/homes/dlarionov/efficient-llm-metrics/WMTCL_MT0/data'
MODEL_DIR = f'{DATA_DIR}/ckpts'

ST_MODEL = 'sentence-transformers/sentence-t5-large'
MT0_MODEL = 'bigscience/mt0-large'

TARGET_LANG_PAIRS = ['en-de', 'zh-en', 'he-en']
LANG_PAIRS = TARGET_LANG_PAIRS + ['de-en', 'en-zh']

ID2LANG = {
    'en' : 'english',
    'de' : 'german',
    'zh' : 'chinese',
    'he' : 'hebrew'
}

SCORE_TYPE2ID = {
    'da': 1,
    'mqm': 2,
    'sqm': 3
}

SCORE_TYPE_WEIGHTS = {
    SCORE_TYPE2ID['da']: 0.8,
    SCORE_TYPE2ID['mqm']: 1.0,
    SCORE_TYPE2ID['sqm']: 0.8
}

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
N_EPOCHS = 1

MAX_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-2
WARMUP_STEPS_RATIO = 0.1

N_COPIES = 3
CHECKPOINT_EVERY_STEP = 10000
EVAL_EVERY_STEP = 10000
