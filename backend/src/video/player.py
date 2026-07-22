import cv2
import time

from src.video.loader import VideoLoader


class VideoPlayer:
    # Constructor
    def __init__(self, loader: VideoLoader, window_name: str = "Video"):
        self._loader = loader
        self._window_name = window_name

    # Public methods
    def play(self) -> None:
        try:
            frame_duration = self._frame_duration_seconds()
            next_frame_time = time.perf_counter()

            while True:
                frame = self._loader.read()
                if frame is None:
                    break

                cv2.imshow(self._window_name, frame)

                if frame_duration > 0:
                    next_frame_time += frame_duration
                    sleep_time = next_frame_time - time.perf_counter()

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # If decoding/rendering falls behind, resync to avoid drift.
                        next_frame_time = time.perf_counter()

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC or 'q' to quit
                    break
        finally:
            self.close()

    def close(self) -> None:
        cv2.destroyAllWindows()

    # Private helpers
    def _frame_duration_seconds(self) -> float:
        fps = self._loader.metadata.fps
        if fps <= 0:
            return 0.0

        return 1.0 / fps
