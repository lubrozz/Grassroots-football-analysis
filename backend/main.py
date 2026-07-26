from pathlib import Path

from src.detection.yolo_detector import YoloDetector
from src.video.loader import VideoLoader
from src.video.player import VideoPlayer

from backend.src.video.annotator import FrameAnnotator


def main() -> None:
    video_path = Path("data/input/veo-tracker-highlights-test.mp4")
    loader = VideoLoader(video_path)
    detector = YoloDetector()  # Initialize the YOLO detector
    annotator = FrameAnnotator()  # Initialize the frame annotator
    player = VideoPlayer(loader.metadata.fps)

    try:
        while True:
            frame = loader.read()

            if frame is None:
                break  # End of video

            detections = detector.detect(frame)  # Run detection on the frame

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
