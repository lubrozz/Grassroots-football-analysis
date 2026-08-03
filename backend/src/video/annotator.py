import cv2
import numpy as np
from src.core.video_track import Track


class FrameAnnotator:
    def annotate(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Annotates the given video frame with the provided tracks.

        Args:
            frame (np.ndarray): The video frame to annotate.
            tracks (list[Track]): A list of tracked objects to annotate on the frame.

        Returns:
            np.ndarray: The annotated video frame.
        """
        annotated = frame.copy()  # Create a copy of the frame to annotate

        self._draw_tracks(annotated, tracks)  # Draw each track on the frame

        return annotated

    # Private methods
    def _draw_tracks(self, frame: np.ndarray, tracks: list[Track]) -> None:
        for track in tracks:
            bbox = track.bounding_box
            x1 = int(bbox.x)
            y1 = int(bbox.y)
            x2 = int(bbox.x + bbox.width)
            y2 = int(bbox.y + bbox.height)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 0, 255), 2
            )  # Draw bounding box in red

            label_y = max(
                15, y1 - 8
            )  # Position label above the bounding box, ensuring it doesn't go off-screen

            if track.class_id == 0:
                label = f"player_id: #{track.id}, conf:{track.confidence:.2f}"  # Create label text with id, confidence
                cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )  # Draw label text in red
