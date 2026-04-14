import os

# roots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "../../dataset/HDREye/images/Bracketed_images")
OUTPUT_DIR_PATH = os.path.join(BASE_DIR, "../outputs")
HDR_ORIGINAL_ROOT = os.path.join(BASE_DIR, "../../dataset/HDREye/images/HDR")

# training settings
ACCUMULATION_STEPS = 4
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 1e-4
CRITERION = "MSE_VGG"

# dataset settings
EXPOSURE_TIMES = [-2.7, 0.0, 2.7]
SAMPLES_LABELS_INPUT = "input"
SAMPLES_LABELS_TARGET_UNDER = "target_under"
SAMPLES_LABELS_TARGET_OVER = "target_over"
RESIZE_DIM = (256, 256)

# output
OUTPUT_MODEL_NAME = f"exposure_unet_{EPOCHS}_{RESIZE_DIM}_lr{LEARNING_RATE}_{CRITERION}.pth"