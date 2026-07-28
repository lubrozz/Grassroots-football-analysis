from dataclasses import dataclass


@dataclass(frozen=True)  # frozen=True makes object immutable.
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
