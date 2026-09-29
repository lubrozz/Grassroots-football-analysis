import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColourExtractor:
    def __init__(self, x_margin: float = 0.25, min_pixels: int = 20):
        """
        Initializes the ColourExtractor with specified parameters.

        Args:
            x_margin (float): The margin to exclude from the left and right sides of the jersey crop.
            min_pixels (int): The minimum number of pixels required to consider a colour valid.
        """
        self.__x_margin = x_margin
        self.__min_pixels = min_pixels

    def extract_colour(
        self, jersey_crops: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """
        Extract the dominant colour from the cropped jersey images.

        Args:
            jersey_crops (dict[int, np.ndarray]): A dictionary mapping track IDs to cropped jersey images.

        Returns:
            dict[int, np.ndarray]: A dictionary mapping track IDs to the dominant colour of the jersey
        """

        colours: dict[int, np.ndarray] = {}

        for track_id, jersey_crop in jersey_crops.items():
            height, width = jersey_crop.shape[:2]

            # Keep only the central part of the jersey crop to avoid background influence
            x0 = int(width * self.__x_margin)
            x1 = int(width * (1 - self.__x_margin))
            core = jersey_crop[:, x0:x1]

            if core.shape[0] * core.shape[1] < self.__min_pixels:
                continue

            lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB).reshape(-1, 3)
            colours[track_id] = np.median(lab, axis=0).astype(np.float32)

        return colours
