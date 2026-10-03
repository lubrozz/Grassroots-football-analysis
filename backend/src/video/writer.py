from pathlib import Path

import cv2
import numpy as np
from src.core.exceptions import VideoWriterError


class VideoWriter:
    # Constructor
    def __init__(
        self,
        output_path: str | Path,
        frame_rate: float,
        frame_width: int,
        frame_height: int,
        codec: str = "mp4v",
    ):
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_size = (frame_width, frame_height)
        self._frames_written = 0

        fourcc = cv2.VideoWriter.fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self._output_path), fourcc, frame_rate, self._frame_size
        )
        if not self._writer.isOpened():
            raise VideoWriterError(f"Could not open video writer: {self._output_path}")

    @property
    def output_path(self) -> Path:
        return self._output_path

    def write(self, frame: np.ndarray) -> None:
        frame_height, frame_width = frame.shape[:2]
        if (frame_width, frame_height) != self._frame_size:
            raise VideoWriterError(
                f"Frame size {frame_width}x{frame_height} does not match "
                f"writer size {self._frame_size[0]}x{self._frame_size[1]}"
            )
        self._writer.write(frame)
        self._frames_written += 1

    def close(self) -> None:
        if self._writer.isOpened():
            self._writer.release()
