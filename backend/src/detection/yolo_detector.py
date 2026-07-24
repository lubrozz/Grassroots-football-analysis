import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.core.video_detection import Detection
from src.core.video_boundingBox import BoundingBox

class YoloDetector:
    def __init__(self):
        self._model = YOLO("yolov8n.pt")  # Load a pre-trained YOLO model (optionally use yolov8s.pt for better accuracy)
        self._person_confidence_threshold = 0.28  # Set a confidence threshold for detections
        self._ball_confidence_threshold = 0.05  # Set a confidence threshold for ball detections
        self._inference_confidence = min(self._person_confidence_threshold, self._ball_confidence_threshold)  # Use the lower of the two thresholds for inference
        
    def detect(self, frame: np.ndarray) -> list[Detection]:
        results_any = list(self._model(frame, verbose=False, classes=[0, 32], conf=self._inference_confidence, max_det=60, imgsz=1280))  # Run detection on the frame
        
        if not results_any:
            return []  # No detections
        
        first = results_any[0]  # Get the first result (assuming batch size of 1)
        
        if not isinstance(first, Results):
            return []  # Unexpected result type
        
        if first.boxes is None:
            return []  # No bounding boxes detected
        
        detections: list[Detection] = []
        for box in first.boxes:
            class_id = int(box.cls.item())  # Get class ID
            if class_id not in (0, 32):  # Only consider class IDs 0 (person) and 32 (sports ball)
                continue  # Skip other classes
            
            confidence = float(box.conf.item())  # Get confidence score
            if class_id == 32 and confidence < self._ball_confidence_threshold:
                continue  # Skip ball detections below the ball confidence threshold
            elif class_id == 0 and confidence < self._person_confidence_threshold:
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