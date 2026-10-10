import cv2
import numpy as np
import supervision as sv
from src.config import settings
from src.core.video_boundingBox import BoundingBox
from src.core.video_detection import Detection
from src.core.video_track import Track
from supervision.tracker.byte_tracker.core import ByteTrack


class ByteTracker:
    def __init__(self, frame_rate: float):
        # Initialize the tracker state here
        self._tracker = ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=settings.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=settings.LOST_TRACK_BUFFER,
            minimum_matching_threshold=settings.MINIMUM_MATCHING_THRESHOLD,
        )
        self._camera_transform = np.eye(
            3, dtype=np.float32
        )  # Reference -> Current frame

    def update(
        self, detections: list[Detection], camera_motion: np.ndarray | None = None
    ) -> list[Track]:
        """
        Updates the tracker with new detections and returns the current tracks.

        Args:
            detections (list[Detection]): A list of Detection objects for the current frame.
            camera_motion (np.ndarray | None): The camera motion matrix.

        Returns:
            list[Track]: A list of Track objects representing the current tracked objects.
        """
        if camera_motion is not None:
            # Accumulate before the early return, so no camera motion is ever skipped
            self._camera_transform = (
                np.vstack([camera_motion, [0, 0, 1]]) @ self._camera_transform
            )

        if not detections:
            return []

        sv_detections = self._to_supervision_detections(detections)
        sv_detections.data["frame_xyxy"] = (
            sv_detections.xyxy.copy()
        )  # Store original frame coordinates
        sv_detections.xyxy = self._to_reference(
            sv_detections.xyxy
        )  # Transform to reference coordinates

        tracked_detections = self._tracker.update_with_detections(sv_detections)
        tracked_detections.xyxy = tracked_detections.data[
            "frame_xyxy"
        ]  # Restore original frame coordinates

        assert tracked_detections.tracker_id is not None
        assert tracked_detections.confidence is not None

        return self._to_tracks(tracked_detections)

    def reset(self) -> None:
        # Reset the tracker state
        self.tracks = []

    # Private helpers
    def _to_supervision_detections(self, detections: list[Detection]) -> sv.Detections:
        """
        Converts a list of Detection objects to a supervision.Detections object.

        Args:
            detections (list[Detection]): List of Detection objects.

        Returns:
            sv.Detections: A supervision.Detections object containing the converted detections.
        """
        if not detections:
            return sv.Detections.empty()

        bounding_boxes = []
        class_ids = []
        confidences = []

        for detection in detections:
            x1 = detection.bounding_box.x
            y1 = detection.bounding_box.y
            x2 = detection.bounding_box.x + detection.bounding_box.width
            y2 = detection.bounding_box.y + detection.bounding_box.height

            bounding_boxes.append([x1, y1, x2, y2])
            class_ids.append(detection.class_id)
            confidences.append(detection.confidence)

        return sv.Detections(
            xyxy=np.array(bounding_boxes, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=np.int32),
        )

    def _to_tracks(self, tracked_detections: sv.Detections) -> list[Track]:
        """
        Converts a supervision.Detections object to a list of Track objects.

        Args:
            tracked_detections (sv.Detections): A supervision.Detections object containing tracked detections.

        Returns:
            list[Track]: A list of Track objects representing the current tracked objects.
        """
        if len(tracked_detections) == 0:
            return []

        tracks: list[Track] = []

        assert tracked_detections.confidence is not None, "Confidence array is None"
        assert tracked_detections.class_id is not None, "Class ID array is None"
        assert tracked_detections.tracker_id is not None, "Tracker ID array is None"

        for xyxy, confidence, class_id, tracker_id in zip(
            tracked_detections.xyxy,
            tracked_detections.confidence,
            tracked_detections.class_id,
            tracked_detections.tracker_id,
        ):
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1

            bounding_box = BoundingBox(
                x=float(x1), y=float(y1), width=float(width), height=float(height)
            )

            track = Track(
                id=int(tracker_id),
                class_id=int(class_id),
                confidence=float(confidence),
                bounding_box=bounding_box,
            )
            tracks.append(track)

        return tracks

    def _to_reference(self, xyxy: np.ndarray) -> np.ndarray:
        """
        Transforms bounding boxes from the current frame to the reference frame using the accumulated camera motion.

        Args:
            xyxy (np.ndarray): An array of bounding boxes in the format [x1, y1, x2, y2].

        Returns:
            np.ndarray: An array of transformed bounding boxes in the reference frame.
        """
        inverse = np.linalg.inv(self._camera_transform)[:2]
        corners = xyxy.reshape(-1, 2, 2)  # (x1, y1), (x2, y2) as seperate points
        return cv2.transform(corners, inverse).reshape(-1, 4)
