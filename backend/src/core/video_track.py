from dataclasses import dataclass

from src.core.detection_class import DetectionClass
from src.core.video_boundingBox import BoundingBox


@dataclass(frozen=True)  # frozen=True makes object immutable.
class Track:
    id: int
    class_id: DetectionClass
    confidence: float
    bounding_box: BoundingBox
