import cv2
import time
import numpy as np

from src.video.loader import VideoLoader
from src.core.video_detection import Detection
from src.core.video_boundingBox import BoundingBox


class VideoPlayer:
    # Constructor
    def __init__(self, loader: VideoLoader, window_name: str = "Video"):
        self._loader = loader
        self._window_name = window_name
        self._playback_speed = (
            1.0  # Normal speed; can be adjusted for faster/slower playback.
        )

    # Public methods
    def play(self) -> None:
        try:
            base_frame_duration = self._frame_duration_seconds()
            next_frame_time = time.perf_counter()
            
            frame_index = 0

            while True:
                frame = self._loader.read()
                frame_index += 1
                if frame is None:
                    break
                
                moving_x = 100 + (frame_index % 300)
                test_detections = [
                    Detection(
                        class_id=0,
                        confidence=0.99,
                        bounding_box=BoundingBox(x=moving_x, y=120, width=220, height=300),
                    )
                ]
                
                self._draw_detections(frame, test_detections, frame_index) # Draw test detections on the frame
                self._draw_overlay_text(frame) # Draw control-overlay text on the frame
                cv2.imshow(self._window_name, frame)

                if base_frame_duration > 0:
                    effective_frame_duration = (
                        base_frame_duration / self._playback_speed
                    )
                    next_frame_time += effective_frame_duration
                    sleep_time = next_frame_time - time.perf_counter()

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # If decoding/rendering falls behind, resync to avoid drift.
                        next_frame_time = time.perf_counter()

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
                    next_frame_time = time.perf_counter()

                if key in (27, ord("q")):  # ESC or 'q' to quit
                    break
        finally:
            self.close()

    def close(self) -> None:
        cv2.destroyAllWindows()

    # Private helpers
    def _draw_detections(self, frame: np.ndarray, detections: list[Detection], frame_index: int) -> None:
        for detection in detections:
            bbox = detection.bounding_box
            x1 = int(bbox.x)
            y1 = int(bbox.y)
            x2 = int(bbox.x + bbox.width)
            y2 = int(bbox.y + bbox.height)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw bounding box in red
            
            label = f"id:{detection.class_id}, conf:{detection.confidence:.2f}, frame:{frame_index}" # Create label text with id, confidence, and frame index
            label_y = max(15, y1 - 8) # Position label above the bounding box, ensuring it doesn't go off-screen
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # Draw label text in red
    
    
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

    def _draw_overlay_text(self, frame: np.ndarray) -> None:
        lines = [
            f"Speed: {self._playback_speed:.2f}x",
            f"Controls: 'z' slow, 'x' normal, 'c' fast",
            f"'q' or ESC to quit",
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
