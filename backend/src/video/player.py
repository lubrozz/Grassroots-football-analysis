import time

import cv2
import numpy as np


class VideoPlayer:
    # Constructor
    def __init__(
        self,
        fps: float,
        window_name: str = "Video",
        max_display_width: int | None = None,
        max_display_height: int | None = None,
        resizable: bool = True,
    ):
        self._window_name = window_name
        self._playback_speed = (
            1.0  # Normal speed; can be adjusted to 0.5x, 2x, etc. using controls
        )
        self._base_frame_duration = (1.0 / fps) if fps > 0 else 0.0
        self._next_frame_time = time.perf_counter()
        self._max_display_width = max_display_width
        self._max_display_height = max_display_height
        self._resizable = resizable
        if self._resizable:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        else:
            cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)

    # Public methods
    def show(self, frame: np.ndarray) -> bool:
        self._draw_overlay_text(frame)  # Draw control-overlay text on the frame

        self._draw_overlay_text(display_frame)  # Draw control-overlay text on the frame
        self._ensure_display_size(
            display_frame
        )  # Ensure the display window is appropriately sized
        cv2.imshow(self._window_name, display_frame)

        if self._base_frame_duration > 0:
            effective_frame_duration = self._base_frame_duration / self._playback_speed

            self._next_frame_time += effective_frame_duration
            sleep_time = self._next_frame_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If decoding/rendering falls behind, resync to avoid drift.
                self._next_frame_time = time.perf_counter()

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("z"),
            ord("x"),
            ord("c"),
        ):  # 'z' to slow down, 'x' to normal, 'c' to speed up
            if key == ord("z"):
                self._set_playback_speed(self._playback_speed * 0.5)
            elif key == ord("x"):
                self._set_playback_speed(1.0)
            elif key == ord("c"):
                self._set_playback_speed(self._playback_speed * 2.0)

            # Resync scheduler right after changing speed.
            self._next_frame_time = time.perf_counter()

        if key in (27, ord("q")):  # ESC or 'q' to quit
            return False

        return True  # Continue playback

    def close(self) -> None:
        cv2.destroyAllWindows()

    def reset(self) -> None:
        self._next_frame_time = time.perf_counter()

    # Private helpers
    def _set_playback_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("Playback speed must be positive.")
        speed = max(0.25, min(speed, 4.0))  # Clamp speed between 0.25x and 4x
        self._playback_speed = speed

    def _ensure_display_size(self, frame: np.ndarray) -> None:
        max_w = self._max_display_width
        frame_w = frame.shape[1]
        max_h = self._max_display_height
        frame_h = frame.shape[0]
        scale = min(
            max_w / frame_w, max_h / frame_h, 1.0
        )  # Scale down if needed, but never scale up
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        cv2.resizeWindow(self._window_name, new_w, new_h)

    def _draw_overlay_text(self, frame: np.ndarray) -> None:
        lines = [
            f"Speed: {self._playback_speed:.2f}x",
            "Controls: 'z' slow, 'x' normal, 'c' fast",
            "'q' or ESC to quit",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)  # White
        box_color = (0, 0, 0)  # Black background for better visibility

        padding_x = 10
        padding_y = 8
        line_spacing = 5  # Space between lines

        # Calculate the size of the text box based on the lines of text
        sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
        max_text_width = max(size[0][0] for size in sizes)
        total_text_height = sum(size[0][1] for size in sizes)
        max_baseline = max(size[1] for size in sizes)

        box_w = max_text_width + (padding_x * 2)
        box_h = (
            total_text_height
            + line_spacing * (len(lines) - 1)
            + max_baseline
            + (padding_y * 2)
        )

        frame_w = frame.shape[1]
        margin = 10  # Margin from the edges of the frame
        x = frame_w - box_w - margin
        y = margin
        x = max(0, x)

        # Draw the background rectangle
        top_left = (x, y)
        bottom_right = (x + box_w, y + box_h)
        cv2.rectangle(frame, top_left, bottom_right, box_color, -1)

        text_x = x + padding_x
        text_y = y + padding_y

        # Draw each line of text
        for i, line in enumerate(lines):
            line_height = sizes[i][0][1]
            text_y += line_height
            cv2.putText(
                frame,
                line,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            text_y += line_spacing  # Add spacing after each line
