from dataclasses import dataclass


@dataclass(frozen=True)  # frozen=True makes object immutable.
class VideoMetaData:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: int
