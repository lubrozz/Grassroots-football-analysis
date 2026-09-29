import time

from src.analytics.team_classifier import TeamClassifier
from src.config import settings
from src.core.detection_class import DetectionClass
from src.core.video_track import Track
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTracker
from src.video.annotator import FrameAnnotator
from src.video.loader import VideoLoader
from src.video.writer import VideoWriter


def get_player_tracks(tracks: list[Track]) -> list[Track]:
    return [track for track in tracks if track.class_id == DetectionClass.PERSON]


def fit_team_classifier(
    loader: VideoLoader, detector: YoloDetector, team_classifier: TeamClassifier
) -> None:
    tracker = ByteTracker(frame_rate=loader.metadata.fps)
    frames_used = 0

    while not team_classifier.is_ready_to_fit():
        frame = loader.read()

        if frame is None:
            break  # Video ended before enough samples were collected

        tracks = tracker.update(detector.detect(frame))
        team_classifier.collect_samples(frame, get_player_tracks(tracks))
        frames_used += 1

    team_classifier.fit()  # Raises ValueError if too few samples were collected
    print(f"Fitted team classifier on {frames_used} frames")
    print(f"Team Colours (LAB): {team_classifier.team_colours}")


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
    team_classifier = TeamClassifier()
    annotator = FrameAnnotator()  # Initialize the frame annotator
    writer = VideoWriter(
        output_path=output_path,
        frame_rate=loader.metadata.fps,
        frame_width=loader.metadata.width,
        frame_height=loader.metadata.height,
        codec=settings.OUTPUT_VIDEO_CODEC,
    )

    start_time = time.perf_counter()

    try:
        # Pass 1: fit the team model
        fit_team_classifier(loader, detector, team_classifier)

        # Pass 2: classify, annotate, and write every frame
        loader.reset()
        tracker = ByteTracker(
            frame_rate=loader.metadata.fps
        )  # Fresh tracker for pass 2
        frames_written = 0

        while True:
            frame = loader.read()

            if frame is None:
                break  # End of video

            detections = detector.detect(frame)  # Run detection on the frame
            tracks = tracker.update(detections)  # Update the tracker

            team_assignments = team_classifier.assign_teams(
                frame, get_player_tracks(tracks)
            )

            annotated_frame = annotator.annotate(
                frame, tracks, team_assignments, team_classifier.team_colours
            )  # Annotate the frame

            writer.write(annotated_frame)
            frames_written += 1

            if writer._frames_written % settings.PROGRESS_LOG_EVERY_N_FRAMES == 0:
                print(
                    f"Processed {writer._frames_written}/{loader.metadata.frame_count} frames"
                )

    finally:
        writer.close()
        loader.close()

    elapsed = time.perf_counter() - start_time
    print(f"Wrote {frames_written} frames to {output_path} in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
