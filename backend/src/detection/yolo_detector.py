import numpy as np
from src.config import settings
from src.core.video_boundingBox import BoundingBox
from src.core.video_detection import Detection
from ultralytics import YOLO
from ultralytics.engine.results import Results

from backend.src.core.detection_class import DetectionClass


class YoloDetector:
    def __init__(self):
        self._model = YOLO(
            settings.YOLO_MODEL
        )  # Load a pre-trained YOLO model (optionally use yolov8s.pt for better accuracy)
        self._person_confidence_threshold = (
            settings.PERSON_CONFIDENCE_THRESHOLD
        )  # Set a confidence threshold for detections
        self._ball_confidence_threshold = (
            settings.BALL_CONFIDENCE_THRESHOLD
        )  # Set a confidence threshold for ball detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model_results = list(
            self._model(
                frame,
                verbose=False,
                classes=[DetectionClass.SPORTS_BALL, DetectionClass.PERSON],
                conf=min(  # Use the lower of the two thresholds for inference
                    settings.PERSON_CONFIDENCE_THRESHOLD,
                    settings.BALL_CONFIDENCE_THRESHOLD,
                ),
                max_det=settings.YOLO_MAX_DETECTIONS,
                imgsz=settings.YOLO_IMAGE_SIZE,
            )
        )  # Run detection on the frame

        if not model_results:
            return []  # No detections

        first = model_results[0]  # Get the first result (assuming batch size of 1)

        if not isinstance(first, Results):
            return []  # Unexpected result type

        if first.boxes is None:
            return []  # No bounding boxes detected

        detections: list[Detection] = []
        for box in first.boxes:
            class_id = int(box.cls.item())  # Get class ID
            if class_id not in (
                DetectionClass.SPORTS_BALL,
                DetectionClass.PERSON,
            ):  # Only consider class IDs 0 (person) and 32 (sports ball)
                continue  # Skip other classes

            confidence = float(box.conf.item())  # Get confidence score
            if (
                class_id == DetectionClass.SPORTS_BALL
                and confidence < self._ball_confidence_threshold
            ):
                continue  # Skip ball detections below the ball confidence threshold
            elif (
                class_id == DetectionClass.PERSON
                and confidence < self._person_confidence_threshold
            ):
                continue  # Skip person detections below the confidence threshold

            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box coordinates

            width = x2 - x1
            height = y2 - y1

            detections.append(
                Detection(
                    class_id=class_id,
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(width),
                        height=float(height),
                    ),
                )
            )
        return detections
