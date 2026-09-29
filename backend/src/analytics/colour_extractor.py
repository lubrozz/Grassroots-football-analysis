import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColourExtractor:
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
            # Convert the jersey crop to HSV color space for better color segmentation
            jersey_hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)

            # Reshape the image into a 2D array for KMeans clustering
            image_2d = jersey_hsv.reshape(-1, 3)

            kmeans = KMeans(n_clusters=2, random_state=0)

            kmeans.fit(image_2d)

            labels = kmeans.labels_

            clustered_image = labels.reshape(jersey_hsv.shape[0], jersey_hsv.shape[1])

            corner_clusters = [
                clustered_image[0, 0],
                clustered_image[0, -1],
                clustered_image[-1, 0],
                clustered_image[-1, -1],
            ]

            non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

            player_cluster = 1 - non_player_cluster

            # Get representative colour in HSV space
            hsv_colour = kmeans.cluster_centers_[player_cluster]

            # # Set HSV values to uint8 for OpenCV compatibility
            # hsv_colour = np.uint8([[hsv_colour]])

            # # Convert the HSV colour to BGR colour space
            # bgr_colour = cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)[0][0]

            colours[track_id] = hsv_colour

        return colours
