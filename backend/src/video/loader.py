# Return dataclass such as "VideoMetaData" to give type safety, self-documenting code and makes interface between modules easier.
# VideoLoader will have helper functions such as:
# open(), close(), get_metadata(), read_frame(), reset()

from pathlib import Path

import cv2
import numpy as np

from core.exceptions import VideoLoaderError
from core.video_metadata import VideoMetaData


class VideoLoader:
    # Constructor
    def __init__(self, video_path: str | Path):
        self._video_path = Path(video_path)
        if not self._video_path.exists():
            raise VideoLoaderError(f"video file does not exist: {self._video_path}")

        self._capture = cv2.VideoCapture(str(self._video_path))
        if not self._capture.isOpened():
            raise VideoLoaderError(f"Could not open video: {self._video_path}")

        self._metadata = self._read_metadata()

    # Public properties
    @property  # represents data, not an action that changes state (like read()).
    def metadata(self) -> VideoMetaData:
        return self._metadata

    @property
    def current_frame(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    # Public methods
    def read(self) -> np.ndarray | None:
        """
        Reads the next frame from the video.

        Returns:
            The next video frame as a NumPy array,
            or None if there are no more frames.
        """
        success, frame = self._capture.read()

        if not success:
            return None

        return frame

    def reset(self) -> None:
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        if self._capture.isOpened():
            self._capture.release()

    # Private helpers
    def _read_metadata(self) -> VideoMetaData:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetaData(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )
