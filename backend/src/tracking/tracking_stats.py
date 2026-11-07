import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from src.core.detection_class import DetectionClass
from src.core.video_track import Track


def print_tracking_stats(
    tracks_per_frame: list[list[Track]], fps: float, reference_frames: list[int]
) -> None:

    frames_per_id = Counter(
        track.id
        for tracks in tracks_per_frame
        for track in tracks
        if track.class_id == DetectionClass.PERSON
    )

    lengths_s = np.array(list(frames_per_id.values())) / fps
    players_per_frame = [
        sum(track.class_id == DetectionClass.PERSON for track in tracks)
        for tracks in tracks_per_frame
    ]

    print(f"Unique player IDs:        {len(frames_per_id)}")
    print(f"Median track length:      {np.median(lengths_s):.1f} s")
    print(f"Tracks shorter than 1 s:  {np.mean(lengths_s < 1.0):.0%}")
    print(f"Mean players per frame:   {np.mean(players_per_frame):.1f}")

    if reference_frames:
        for frame_index in reference_frames:
            count = sum(
                track.class_id == DetectionClass.PERSON
                for track in tracks_per_frame[frame_index]
            )
            print(f"Frame {frame_index}: {count} players tracked")


def pick_reference_frames(
    frame_count: int, count: int = 5, seed: int = 42
) -> list[int]:
    """
    Picks a specified number of reference frames from a video.

    Args:
        frame_count (int): Total number of frames in the video.
        count (int): Number of reference frames to pick.
        seed (int): Random seed for reproducibility.

    Returns:
        list[int]: List of selected reference frame indices.
    """
    rng = random.Random(seed)
    segment_length = frame_count // count

    return [
        rng.randrange(i * segment_length, (i + 1) * segment_length)
        for i in range(count)
    ]


def save_reference_frames(
    output_dir: Path, frame_index: int, frame: np.ndarray, annotated_frame: np.ndarray
) -> None:
    """
    Saves the original and annotated reference frames to the specified output directory.

    Args:
        output_dir (Path): Directory where the frames will be saved.
        frame_index (int): Index of the frame being saved.
        frame (np.ndarray): Original frame image.
        annotated_frame (np.ndarray): Annotated frame image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"frame_{frame_index:05d}_raw.png"), frame)
    cv2.imwrite(
        str(output_dir / f"frame_{frame_index:05d}_annotated.png"), annotated_frame
    )
