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

YOLO_MODEL = MODEL_DIR / "yolov8n.pt" # Load a pre-trained YOLO model(optionally use yolov8s.pt for better accuracy)

YOLO_IMAGE_SIZE = 1280

YOLO_MAX_DETECTIONS = 60

PERSON_CONFIDENCE_THRESHOLD = 0.28  # Set a confidence threshold for detections
BALL_CONFIDENCE_THRESHOLD = 0.05  # Set a confidence threshold for ball detections

#
# Video
#

DEFAULT_WINDOW_NAME = "Football Analysis"
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
DISPLAY_RESIZABLE = True
