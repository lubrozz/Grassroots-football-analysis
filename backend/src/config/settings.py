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

YOLO_MODEL = (
    MODEL_DIR / "yolov8m.pt"
)  # Load a pre-trained YOLO model(optionally use yolov8s.pt for better accuracy)

YOLO_IMAGE_SIZE = 1280  # Resize input images to this size for YOLO (higher for better accuracy, lower for performance)
YOLO_MAX_DETECTIONS = 60  # How many detections to keep per frame (higher for crowded scenes, lower for performance)

PERSON_CONFIDENCE_THRESHOLD = 0.1  # Set a confidence threshold for detections
BALL_CONFIDENCE_THRESHOLD = 0.02  # Set a confidence threshold for ball detections

#
# ByteTrack
#

TRACK_ACTIVATION_THRESHOLD = (
    0.25  # Minimum confidence score for a detection to be considered for tracking
)
LOST_TRACK_BUFFER = 30  # Number of frames to keep a track after it is lost
MINIMUM_MATCHING_THRESHOLD = (
    0.8  # Minimum IoU threshold for matching detections to tracks
)

CMC_ENABLED = True  # Enable/disable CMC (Camera Motion Compensation) for tracking

#
# Video
#

DEFAULT_WINDOW_NAME = "Football Analysis"
DISPLAY_MAX_WIDTH = 1920
DISPLAY_MAX_HEIGHT = 1080
DISPLAY_RESIZABLE = True

INPUT_VIDEO = INPUT_DIR / "u16-women-goal.mp4"
OUTPUT_VIDEO_SUFFIX = "_annotated_teamAssignmentTest"
OUTPUT_VIDEO_CODEC = "mp4v"
PROGRESS_LOG_EVERY_N_FRAMES = 100
