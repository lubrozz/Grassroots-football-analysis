import cv2
import numpy as np
from src.core.video_detection import Detection


class CameraMotionEstimator:
    """Estimates how the camera moved between two consecutive frames."""

    def __init__(
        self,
        downscale: float = 0.5,
        max_corners: int = 500,
        min_points: int = 20,
    ):
        self._downscale = downscale
        self._max_corners = max_corners
        self._min_points = min_points
        self._previous_gray: np.ndarray | None = None
        self._previous_points: np.ndarray | None = None

    def estimate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Returns a 2x3 affine matrix mapping points in the previous frame to the same
        points in the current frame (full-resolution pixels). Returns the identity
        on the first frame or when the motion cannot be estimated.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=self._downscale, fy=self._downscale)

        motion = np.eye(2, 3, dtype=np.float32)  # Identity: "camera did not move"

        if (
            self._previous_gray is not None
            and self._previous_points is not None
            and len(self._previous_points) >= self._min_points
        ):
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                self._previous_points,
                np.empty_like(self._previous_points),
            )
            found = status.ravel() == 1

            if found.sum() >= self._min_points:
                matrix, _ = cv2.estimateAffinePartial2D(
                    self._previous_points[found],
                    current_points[found],
                    method=cv2.RANSAC,
                )
                if matrix is not None:
                    motion = self._to_full_resolution(matrix)

        # Pick new background points in this frame, to follow into the next one
        self._previous_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_corners,
            qualityLevel=0.01,
            minDistance=10,
            mask=self._background_mask(gray.shape, detections),
        )
        self._previous_gray = gray

        return motion

    # Private helpers
    def _background_mask(
        self, shape: tuple[int, ...], detections: list[Detection]
    ) -> np.ndarray:
        """255 where points may be picked, 0 inside detection boxes."""
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        for detection in detections:
            bbox = detection.bounding_box
            x1, y1 = int(bbox.x * self._downscale), int(bbox.y * self._downscale)
            x2, y2 = int(bbox.x2 * self._downscale), int(bbox.y2 * self._downscale)
            mask[max(0, y1) : y2, max(0, x1) : x2] = 0
        return mask

    def _to_full_resolution(self, matrix: np.ndarray) -> np.ndarray:
        """Rotation/zoom are scale-independent; only the shift must be scaled up."""
        full = matrix.astype(np.float32).copy()
        full[:, 2] /= self._downscale
        return full
