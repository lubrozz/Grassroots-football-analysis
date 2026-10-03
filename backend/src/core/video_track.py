from dataclasses import dataclass

from src.core.video_boundingBox import BoundingBox


@dataclass(frozen=True)  # frozen=True makes object immutable.
class Track:
    id: int
    class_id: int
    confidence: float
    bounding_box: BoundingBox
