import numpy as np
from src.core.video_track import Track


class PlayerCropper:
    def __init__(self):
        self._crop_counts: dict[int, int] = {}

    def crop(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]:
        """
        Crop the image based on the bounding box of the track.

        Args:
            frame (np.ndarray): The original frame from which to crop.
            tracks (list[Track]): The tracks containing the bounding box information.

        Returns:
            np.ndarray: The cropped image.
        """
        crops: dict[int, np.ndarray] = {}

        for track in tracks:
            track_id = track.id

            bbox = track.bounding_box

            cropped_image = frame[
                int(bbox.y) : int(bbox.y2), int(bbox.x) : int(bbox.x2)
            ]

            if cropped_image.size == 0:
                continue  # Skip if the cropped image is empty

            crops[track_id] = cropped_image

        return crops
