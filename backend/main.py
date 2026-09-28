from pathlib import Path

import cv2
from src.analytics.team_classifier import TeamClassifier
from src.config import settings
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTracker
from src.video.annotator import FrameAnnotator
from src.video.loader import VideoLoader
from src.video.writer import VideoWriter


def main() -> None:
    video_path = settings.INPUT_VIDEO
    output_path = (
        settings.OUTPUT_DIR / f"{video_path.stem}{settings.OUTPUT_VIDEO_SUFFIX}.mp4"
    )
    loader = VideoLoader(video_path)
    detector = YoloDetector()  # Initialize the YOLO detector
    tracker = ByteTracker(
        frame_rate=loader.metadata.fps
    )  # Initialize the ByteTracker for tracking
    annotator = FrameAnnotator()  # Initialize the frame annotator

    writer = VideoWriter(
        output_path=output_path,
        frame_rate=loader.metadata.fps,
        frame_width=loader.metadata.width,
        frame_height=loader.metadata.height,
        codec=settings.OUTPUT_VIDEO_CODEC,
    )

    try:
        while True:
            frame = loader.read()

            if frame is None:
                break  # End of video

            detections = detector.detect(frame)  # Run detection on the frame

            tracks = tracker.update(
                detections
            )  # Update the tracker with the new detections

            annotated_frame = annotator.annotate(frame, tracks)  # Annotate the frame

            writer.write(annotated_frame)

            if writer._frames_written % settings.PROGRESS_LOG_EVERY_N_FRAMES == 0:
                print(
                    f"Processed {writer._frames_written}/{loader.metadata.frame_count} frames"
                )

    finally:
        writer.close()
        loader.close()


if __name__ == "__main__":
    main()
