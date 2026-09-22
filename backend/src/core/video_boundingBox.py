from dataclasses import dataclass


@dataclass(frozen=True)  # frozen=True makes object immutable.
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height
