import numpy as np
from src.analytics.movement import build_trail, foot_position, velocity_from_trail
from src.analytics.team_classifier import TeamClassifier
from src.core.detection_class import DetectionClass
from src.core.moment import Moment, MomentSnapshot, PlayerSnapshot
from src.core.video_track import Track
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTracker
from src.tracking.camera_motion import CameraMotionEstimator
from src.video.loader import VideoLoader


class MomentAnalyzer:
    def __init__(
        self,
        detector: YoloDetector,
        seconds_before: float = 2.0,
        seconds_after: float = 0.5,
        movement_seconds: float = 0.5,
        max_step_heights: float = 0.25,
    ):
        self._detector = detector
        self._seconds_before = seconds_before
        self._seconds_after = seconds_after
        self._movement_seconds = movement_seconds
        self._max_step_heights = max_step_heights

    def analyse(
        self, loader: VideoLoader, moment: Moment
    ) -> tuple[MomentSnapshot, np.ndarray]:
        """
        Analyzes a moment in the video and returns a snapshot of the moment along with the camera motion matrix.

        Args:
            loader (VideoLoader): The video loader instance.
            moment (Moment): The moment to analyze.

        Returns:
            tuple[MomentSnapshot, np.ndarray]: A tuple containing the MomentSnapshot and the camera motion matrix.
        """
        fps = loader.metadata.fps
        last_index = loader.metadata.frame_count - 1
        key_index = min(int(round(moment.time_seconds * fps)), last_index)
        start_index = max(0, key_index - int(self._seconds_before * fps))
        end_index = min(last_index, key_index + int(self._seconds_after * fps))

        # Fresh state per moment, so one moment never influences another
        tracker = ByteTracker(frame_rate=fps)
        camera_motion_estimator = CameraMotionEstimator()
        team_classifier = TeamClassifier()

        tracks_by_frame: dict[int, dict[int, Track]] = {}
        camera_transforms: dict[int, np.ndarray] = {}
        key_frame: np.ndarray | None = None

        loader.seek(start_index)
        for frame_index in range(start_index, end_index + 1):
            frame = loader.read()
            if frame is None:
                break  # End of video

            detections = self._detector.detect(frame)
            camera_motion = camera_motion_estimator.estimate(frame, detections)
            players = [
                track
                for track in tracker.update(detections, camera_motion)
                if track.class_id == DetectionClass.PERSON
            ]

            tracks_by_frame[frame_index] = {track.id: track for track in players}
            camera_transforms[frame_index] = tracker.camera_transform
            team_classifier.add_observations(frame, players)

            if frame_index == key_index:
                key_frame = frame.copy()

        if key_frame is None:
            raise ValueError(
                f"Could not read the key frame at {moment.time_seconds:.2f} seconds (frame index {key_index})."
            )

        team_classifier.fit()
        teams = team_classifier.assign_teams()
        players = [
            self._snapshot_player(
                track, key_index, tracks_by_frame, camera_transforms, teams, fps
            )
            for track in tracks_by_frame[key_index].values()
        ]
        snapshot = MomentSnapshot(
            moment=moment,
            frame_index=key_index,
            frame_width=loader.metadata.width,
            frame_height=loader.metadata.height,
            players=players,
        )
        return snapshot, key_frame

    # Private helpers
    def _snapshot_player(
        self,
        track: Track,
        key_index: int,
        tracks_by_frame: dict[int, dict[int, Track]],
        camera_transforms: dict[int, np.ndarray],
        teams: dict[int, int],
        fps: float,
    ) -> PlayerSnapshot:
        """Creates a PlayerSnapshot for a given track at the key frame index.

        args:
            track (Track): The track to create a snapshot for.
            key_index (int): The index of the key frame.
            tracks_by_frame (dict[int, dict[int, Track]]): A dictionary mapping frame indices to track dictionaries.
            camera_transforms (dict[int, np.ndarray]): A dictionary mapping frame indices to camera transformation matrices.
            teams (dict[int, int]): A dictionary mapping track IDs to team IDs.
            fps (float): Frames per second of the video.

        returns:
            PlayerSnapshot: A snapshot of the player at the key frame.
        """
        max_offset = int(self._movement_seconds * fps)

        trail = build_trail(
            track,
            key_index,
            tracks_by_frame,
            camera_transforms,
            max_offset=int(self._movement_seconds * fps),
            max_step_heights=self._max_step_heights,
        )
        movement = velocity_from_trail(
            trail, min_offset=max(1, int(self._movement_seconds * fps // 2)), fps=fps
        )

        return PlayerSnapshot(
            track_id=track.id,
            team=teams.get(track.id),
            bounding_box=track.bounding_box,
            foot_position=foot_position(track),
            movement=movement,
            trail=[(float(point[0]), float(point[1])) for _, point in trail],
        )
