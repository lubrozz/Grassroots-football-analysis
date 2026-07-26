import numpy as np
from src.core.video_detection import Detection


class DetectionBase:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detects objects in the given video frame.

        Args:
            frame (np.ndarray): The video frame in which to detect objects.

        Returns:
            list[Detection]: A list of detected objects in the frame.
        """

        # Placeholder implementation; replace with actual detection logic.
        return []
