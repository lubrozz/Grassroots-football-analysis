import numpy as np

from src.analytics.movement import build_trail, foot_position, velocity_from_trail
from src.core.video_boundingBox import BoundingBox
from src.core.video_track import Track

FPS = 60
KEY_INDEX = 100


def make_track(foot_x: float) -> Track:
    return Track(
        id=7,
        class_id=0,
        confidence=0.9,
        bounding_box=BoundingBox(x=foot_x - 10, y=300, width=20, height=60),
    )


def test_trail_stops_at_jump_and_gives_true_velocity() -> None:
    # Player runs right at 2 px per frame; before frame 75 the track was 60 px off
    tracks_by_frame: dict[int, dict[int, Track]] = {}
    camera_transforms: dict[int, np.ndarray] = {}
    for frame_index in range(60, KEY_INDEX + 1):
        foot_x = 500 + 2 * (frame_index - KEY_INDEX)
        if frame_index < 75:
            foot_x += 60
        tracks_by_frame[frame_index] = {7: make_track(foot_x)}
        camera_transforms[frame_index] = np.eye(3)

    key_track = tracks_by_frame[KEY_INDEX][7]
    trail = build_trail(
        key_track,
        KEY_INDEX,
        tracks_by_frame,
        camera_transforms,
        max_offset=30,
        max_step_heights=0.25,
    )

    assert trail[-1][0] == 25  # Stopped at the jump
    assert velocity_from_trail(trail, min_offset=15, fps=FPS) == (120.0, 0.0)
    assert foot_position(key_track) == (500.0, 360.0)
