import os
from enum import IntEnum

NUM_FRAMES = 48
NUM_JOINTS = 15
DIM = 3

BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
TIMESTEPS = 1000

ACTIONS = {
    "walk": 0,
    "jump": 1
}

class Joint(IntEnum):
    HEAD = 0
    NECK = 1
    PELVIS = 2
    RIGHT_SHOULDER = 3
    RIGHT_ELBOW = 4
    RIGHT_WRIST = 5
    LEFT_SHOULDER = 6
    LEFT_ELBOW = 7
    LEFT_WRIST = 8
    RIGHT_HIP = 9
    RIGHT_KNEE = 10
    RIGHT_ANKLE = 11
    LEFT_HIP = 12
    LEFT_KNEE = 13
    LEFT_ANKLE = 14

JOINT_CONNECTIONS = [
    (Joint.PELVIS, Joint.NECK),
    (Joint.NECK, Joint.HEAD),
    (Joint.NECK, Joint.RIGHT_SHOULDER),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.NECK, Joint.LEFT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.PELVIS, Joint.RIGHT_HIP),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.PELVIS, Joint.LEFT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE)
]

MOTION_EXCEL = 'MOTION'
DESCRIPTION_EXCEL = 'DESCRIPTION from CMU web database'
SUBJECT_EXCEL = 'SUBJECT from CMU web database'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_METADATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'data')
OUTPUT_PATH = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(OUTPUT_PATH, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)