from pathlib import Path

from src.config import settings
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTracker
from src.video.annotator import FrameAnnotator
from src.video.loader import VideoLoader
from src.video.player import VideoPlayer


def main() -> None:
    video_path = Path("data/input/veo-tracker-highlights-test.mp4")
    loader = VideoLoader(video_path)
    detector = YoloDetector()  # Initialize the YOLO detector
    tracker = ByteTracker()  # Initialize the ByteTracker for tracking
    annotator = FrameAnnotator()  # Initialize the frame annotator
    player = VideoPlayer(
        loader.metadata.fps,
        window_name="Football Analysis",
        max_display_width=settings.DISPLAY_MAX_WIDTH,
        max_display_height=settings.DISPLAY_MAX_HEIGHT,
    )  # Initialize the video player

    try:
        while True:
            frame = loader.read()

            if frame is None:
                break  # End of video

            detections = detector.detect(frame)  # Run detection on the frame

            tracker.update(detections)  # Update the tracker with the new detections

            annotated_frame = annotator.annotate(
                frame, detections
            )  # Annotate the frame

            if not player.show(annotated_frame):  # Show the annotated frame
                break  # Exit if the user closes the window or presses a key

    finally:
        player.close()
        loader.close()


if __name__ == "__main__":
    main()
