from dataclasses import dataclass

from src.core.video_boundingBox import BoundingBox


@dataclass(frozen=True)
class Moment:
    """
    Represents a moment in time with an associated label.

    Attributes:
        time_seconds: float
        label: str
    """

    time_seconds: float
    label: str = ""


@dataclass(frozen=True)
class PlayerSnapshot:
    """
    One player at the moment's key frame

    """

    track_id: int
    team: int | None
    bounding_box: BoundingBox
    foot_position: tuple[float, float]  # (x, y) in key-frame pixels
    movement: tuple[float, float] | None  # Pixels per second, camera motion removed
    trail: list[tuple[float, float]]  # Foot positions in key-frame pixels, newest first


@dataclass(frozen=True)
class MomentSnapshot:
    """
    Everything that the frontend needs to draw a moment

    Attributes:
        moment: Moment

    """

    moment: Moment
    frame_index: int
    frame_width: int
    frame_height: int
    players: list[PlayerSnapshot]
