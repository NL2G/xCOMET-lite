SEED = 101
DATA_DIR = '/home/vyskov/Effective_Metrics/data'

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
    SCORE_TYPE2ID['sqm']: 0.6
}

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
N_EPOCHS = 5

MAX_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 4
