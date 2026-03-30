import os

# roots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "../../dataset/HDREye/images/Bracketed_images")
OUTPUT_DIR_PATH = os.path.join(BASE_DIR, "../outputs")
HDR_ORIGINAL_ROOT = os.path.join(BASE_DIR, "../../dataset/HDREye/images/HDR")

# training settings
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4