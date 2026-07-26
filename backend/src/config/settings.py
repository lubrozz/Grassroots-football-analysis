from pathlib import Path

#
# Project paths
#

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

MODEL_DIR = PROJECT_ROOT / "models"

#
# YOLO
#

YOLO_MODEL = MODEL_DIR / "yolo8n.pt"

YOLO_IMAGE_SIZE = 1280

YOLO_MAX_DETECTIONS = 60

PERSON_CONFIDENCE_THRESHOLD = 0.28
BALL_CONFIDENCE_THRESHOLD = 0.05

#
# Video
#

DEFAULT_WINDOW_NAME = "Football Analysis"
