import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

OUTPUT_PATH = os.path.join(BASE_DIR, '..\\output') 
CSV_FILE = os.path.join(OUTPUT_PATH, 'dataset.csv')

SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

BATCH_SIZE = 16
NUM_EPOCHS = 150
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999

CONDITION_DIM = 10

LAMBDA_L1 = 10

GL_VERSION = (3, 3)
WINDOW_TITLE = 'SIGK 4'
WINDOW_SIZE = (128, 128)

FRAGMENT_SHADER_EXTENSION = ['.frag']
VERTEX_SHADER_EXTENSION = ['.vert']


def get_supported_extensions():
    return [
        *VERTEX_SHADER_EXTENSION,
        *FRAGMENT_SHADER_EXTENSION
    ]
