import numpy as np


class JerseyCropper:
    def crop_jersey(self, player_crops: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """
        Crop the upper body of the player images to isolate the jersey area.

        Args:
            player_crops (dict[int, np.ndarray]): A dictionary mapping track IDs to cropped player images.

        Returns:
            dict[int, np.ndarray]: A dictionary mapping track IDs to cropped jersey images.
        """

        jersey_crops: dict[int, np.ndarray] = {}

        for track_id, player_crop in player_crops.items():
            height, width = player_crop.shape[:2]

            top = int(height * 0.25)
            bottom = int(height * 0.65)

            jersey_crop = player_crop[
                top:bottom, :
            ]  # Crop the upper half of the player image

            jersey_crops[track_id] = jersey_crop

        return jersey_crops
