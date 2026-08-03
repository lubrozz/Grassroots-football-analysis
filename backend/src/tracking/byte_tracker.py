import numpy as np
import supervision as sv
from src.core.video_boundingBox import BoundingBox
from src.core.video_detection import Detection
from src.core.video_track import Track
from supervision.tracker.byte_tracker.core import ByteTrack


class ByteTracker:
    def __init__(self, frame_rate: float):
        # Initialize the tracker state here
        self._tracker = ByteTrack(frame_rate=frame_rate)

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Updates the tracker with new detections and returns the current tracks.

        Args:
            detections (list[Detection]): A list of Detection objects for the current frame.

        Returns:
            list[Track]: A list of Track objects representing the current tracked objects.
        """
        if not detections:
            return []

        sv_detections = self._to_supervision_detections(detections)

        tracked_detections = self._tracker.update_with_detections(sv_detections)

        assert tracked_detections.tracker_id is not None
        assert tracked_detections.confidence is not None

        for tracker_id, confidence, xyxy in zip(
            tracked_detections.tracker_id,
            tracked_detections.confidence,
            tracked_detections.xyxy,
        ):
            if tracker_id in (54, 105, 184):
                print(
                    f"#{tracker_id} "
                    f"conf={confidence:.2f} "
                    f"w={xyxy[2] - xyxy[0]:.1f} "
                    f"h={xyxy[3] - xyxy[1]:.1f}"
                )

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
