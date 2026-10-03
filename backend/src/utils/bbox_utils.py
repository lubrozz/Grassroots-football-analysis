from src.core.video_boundingBox import BoundingBox


def get_center_of_bbox(bbox: BoundingBox) -> tuple[int, int]:
    """
    Calculate the center coordinates of a bounding box.

    Args:
        bbox (BoundingBox): The bounding box for which to calculate the center.

    Returns:
        tuple[int, int]: The (x, y) coordinates of the center of the bounding box.
    """
    return int((bbox.x + bbox.x2) / 2), int((bbox.y + bbox.y2) / 2)


def get_bbox_width(bbox: BoundingBox) -> float:
    """Return the width of a bounding box."""
    return bbox.x2 - bbox.x


def measure_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Return the Euclidean distance between two points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(
    p1: tuple[float, float], p2: tuple[float, float]
) -> tuple[float, float]:
    """Return the x and y differences between two points."""
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox: BoundingBox) -> tuple[int, int]:
    """Return the center point on the bottom edge of a bounding box."""
    return int((bbox.x + bbox.x2) / 2), int(bbox.y2)
