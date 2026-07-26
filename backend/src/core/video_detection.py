from dataclasses import dataclass

from .video_boundingBox import BoundingBox


@dataclass(frozen=True)  # frozen=True makes object immutable.
class Detection:
    class_id: int
    confidence: float
    bounding_box: BoundingBox
