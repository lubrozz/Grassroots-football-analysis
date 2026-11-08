import numpy as np
from src.core.video_track import Track


def foot_position(track: Track) -> tuple[float, float]:
    """Bottom centre of the box: where the player touches the pitch."""
    bbox = track.bounding_box
    return (bbox.x + bbox.width / 2, bbox.y2)


def build_trail(
    track: Track,
    key_index: int,
    tracks_by_frame: dict[int, dict[int, Track]],
    camera_transforms: dict[int, np.ndarray],
    max_offset: int,
    max_step_heights: float,
) -> list[tuple[int, np.ndarray]]:
    """
    Foot positions of the track, walking back from the key frame, converted to the
    key frame's image coordinates (camera motion removed). Returns (offset, point)
    pairs, newest first. Stops at a jump larger than a player can run in one frame,
    which points to an ID switch or a merged box.
    """
    key_transform = camera_transforms[key_index]
    max_step = max_step_heights * track.bounding_box.height
    trail: list[tuple[int, np.ndarray]] = []

    for offset in range(0, max_offset + 1):
        frame_index = key_index - offset
        earlier_track = tracks_by_frame.get(frame_index, {}).get(track.id)
        if earlier_track is None:
            continue  # Briefly missing (e.g. occluded) is fine

        to_key = key_transform @ np.linalg.inv(camera_transforms[frame_index])
        point = (to_key @ np.array([*foot_position(earlier_track), 1.0]))[:2]

        if trail:
            last_offset, last_point = trail[-1]
            step_per_frame = np.linalg.norm(point - last_point) / (offset - last_offset)
            if step_per_frame > max_step:
                break  # Everything before this belongs to a different box

        trail.append((offset, point))

    return trail


def velocity_from_trail(
    trail: list[tuple[int, np.ndarray]], min_offset: int, fps: float
) -> tuple[float, float] | None:
    """Pixels per second from the oldest to the newest trail point, or None if too short."""
    if not trail or trail[-1][0] < min_offset:
        return None

    (_, newest), (oldest_offset, oldest) = trail[0], trail[-1]
    velocity = (newest - oldest) / (oldest_offset / fps)
    return (float(velocity[0]), float(velocity[1]))
