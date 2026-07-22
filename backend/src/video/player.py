import cv2
import time
import numpy as np

from src.video.loader import VideoLoader


class VideoPlayer:
    # Constructor
    def __init__(self, loader: VideoLoader, window_name: str = "Video"):
        self._loader = loader
        self._window_name = window_name
        self._playback_speed = 1.0  # Normal speed; can be adjusted for faster/slower playback.

    # Public methods
    def play(self) -> None:
        try:
            base_frame_duration = self._frame_duration_seconds()
            next_frame_time = time.perf_counter()

            while True:
                frame = self._loader.read()
                if frame is None:
                    break

                self._draw_speed_label(frame)
                cv2.imshow(self._window_name, frame)

                if base_frame_duration > 0:
                    effective_frame_duration = base_frame_duration / self._playback_speed
                    next_frame_time += effective_frame_duration
                    sleep_time = next_frame_time - time.perf_counter()

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # If decoding/rendering falls behind, resync to avoid drift.
                        next_frame_time = time.perf_counter()

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("z"), ord("x"), ord("c")):  # 'z' to slow down, 'x' to normal, 'c' to speed up
                    if key == ord("z"):
                        self._set_playback_speed(self._playback_speed * 0.5)
                    elif key == ord("x"):
                        self._set_playback_speed(1.0)
                    elif key == ord("c"):
                        self._set_playback_speed(self._playback_speed * 2.0)

                    # Resync scheduler right after changing speed.
                    next_frame_time = time.perf_counter()
                
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

    def _set_playback_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("Playback speed must be positive.")
        speed = max(0.25, min(speed, 4.0))  # Clamp speed between 0.25x and 4x
        self._playback_speed = speed

    def _draw_speed_label(self, frame: np.ndarray) -> None:
        label = f"Speed: {self._playback_speed:.2f}x"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (255, 255, 255)  # White
        position = (20, 40)  # Top-left corner

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # define padding around the text for better visibility
        padding_x = 10
        padding_y = 8
        x = 10
        y = 10

        # Calculate the bounding box dimensions
        box_w = text_width + (padding_x * 2)
        box_h = text_height + baseline + (padding_y * 2)

        # Calculate the top-left and bottom-right coordinates of the rectangle
        top_left = (x, y)
        bottom_right = (x + box_w, y + box_h)

        # Calculate the position for the text to be drawn
        text_x = x + padding_x
        text_y = y + padding_y + text_height
        position = (text_x, text_y)

        # Draw the rectangle and the text on the frame
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)  # Background rectangle for better visibility
        cv2.putText(frame, label, position, font, font_scale, color, thickness, cv2.LINE_AA)

