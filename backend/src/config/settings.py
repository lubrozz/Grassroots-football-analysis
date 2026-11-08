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
    MODEL_DIR / "yolov8s.pt"
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
MINIMUM_CONSECUTIVE_FRAMES = (
    2  # Minimum number of consecutive frames to confirm a track
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


#
# Moment analysis
#
MOMENT_TIMESTAMPS = [
    "0:06",
    "0:11.5",
]  # Example moments in format a coach would write them down (minutes:seconds). These will be converted to seconds in the code.
MOMENT_SECONDS_BEFORE = 2.0  # Seconds before the moment to include in analysis
MOMENT_SECONDS_AFTER = 0.5  # Seconds after the moment to include in analysis (mostly extra observations for team vote)
MOMENT_MOVEMENT_SECONDS = (
    0.5  # Seconds to look back for movement estimation (for player movement arrows)
)
MOMENT_MAX_STEP_HEIGHTS = 0.25  # Maximum step height as a fraction of the player's bounding box height (for movement estimation)
MOMENT_DEBUG_IMAGES = (
    True  # Also save a PNG with foot positions and movement drawn on it
)
