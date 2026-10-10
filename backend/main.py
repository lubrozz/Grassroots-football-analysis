import time

from src.analytics.team_classifier import TeamClassifier
from src.config import settings
from src.core.detection_class import DetectionClass
from src.core.video_track import Track
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTracker
from src.tracking.camera_motion import CameraMotionEstimator
from src.tracking.tracking_stats import (
    pick_reference_frames,
    print_tracking_stats,
    save_reference_frames,
)
from src.video.annotator import FrameAnnotator
from src.video.loader import VideoLoader
from src.video.writer import VideoWriter


def get_player_tracks(tracks: list[Track]) -> list[Track]:
    return [track for track in tracks if track.class_id == DetectionClass.PERSON]


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
    camera_motion_estimator = (
        CameraMotionEstimator() if settings.CMC_ENABLED else None
    )  # Initialize the camera motion estimator

    start_time = time.perf_counter()
    frames_written = 0

    try:
        # Pass 1: fit the team model
        tracks_per_frame: list[list[Track]] = []
        reference_frames = pick_reference_frames(loader.metadata.frame_count)
        reference_dir = settings.OUTPUT_DIR / "reference_frames" / video_path.stem

        while True:
            frame = loader.read()

            if frame is None:
                break  # End of video

            detections = detector.detect(frame)  # Run detection on the frame
            camera_motion = (
                camera_motion_estimator.estimate(frame, detections)
                if camera_motion_estimator is not None
                else None
            )  # Estimate camera motion if enabled

            # Temp_ check that the camera motion makes sense:
            # frame_index = len(tracks_per_frame)
            # if camera_motion is not None and frame_index % 60 == 0:
            #     print(
            #         f"Frame {frame_index}: camera shift "
            #         f"x={camera_motion[0, 2]:.1f}, y={camera_motion[1, 2]:.1f} px"
            # )

            tracks = tracker.update(detections, camera_motion)  # Update the tracker
            tracks_per_frame.append(tracks)

            team_classifier.add_observations(frame, get_player_tracks(tracks))

            if len(tracks_per_frame) % settings.PROGRESS_LOG_EVERY_N_FRAMES == 0:
                print(
                    f"Processed {len(tracks_per_frame)}/{loader.metadata.frame_count} frames"
                )

        # Fit the team model and decide one team per track ID
        team_classifier.fit()
        # print(f"Team colours (LAB): {team_classifier.team_colours}")
        # debug_track_ids = {2, 32, 203}  # Example track IDs to debug
        team_assignments = team_classifier.assign_teams(debug_track_ids=None)

        # Print tracking statistics
        print_tracking_stats(tracks_per_frame, loader.metadata.fps, reference_frames)

        # Pass 2: classify, annotate, and write every frame
        loader.reset()
        for frame_index, tracks in enumerate(tracks_per_frame):
            frame = loader.read()
            if frame is None:
                break

            annotated_frame = annotator.annotate(
                frame, tracks, team_assignments, team_classifier.team_colours
            )  # Annotate the frame

            writer.write(annotated_frame)
            frames_written += 1

            if frame_index in reference_frames:
                save_reference_frames(
                    reference_dir, frame_index, frame, annotated_frame
                )

    finally:
        writer.close()
        loader.close()

    elapsed = time.perf_counter() - start_time
    print(f"Wrote {frames_written} frames to {output_path} in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
