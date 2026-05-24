import os

NUM_EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 0.001
NUM_POINTS = 8192

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')

OUTPUT_PATH = os.path.join(BASE_DIR, 'output', f'{NUM_EPOCHS}_{LEARNING_RATE}_{NUM_POINTS}')
MODELS_DIR = os.path.join(OUTPUT_PATH, 'saved_models')
SAMPLES_DIR = os.path.join(OUTPUT_PATH, 'samples')

CSV_FILE = os.path.join(OUTPUT_PATH, 'evaluation_results.csv')

OBJECTS = ['bunny', 'dragon_small', 'armadillo_small']
TEST_OBJECT = 'asian_dragon_really_small'
TARGET_OBJECT = 'teapot'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)