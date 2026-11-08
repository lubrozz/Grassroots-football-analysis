import numpy as np
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
    ):
        self._detector = detector
        self._seconds_before = seconds_before
        self._seconds_after = seconds_after
        self._movement_seconds = movement_seconds

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
            PlayerSnapshot(
                track_id=track.id,
                team=teams.get(track.id),
                bounding_box=track.bounding_box,
                foot_position=self._foot_position(track),
                movement=self._movement(
                    track, key_index, tracks_by_frame, camera_transforms, fps
                ),
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
    def _foot_position(self, track: Track) -> tuple[float, float]:
        """
        Estimates the foot position of a player based on their bounding box.

        Args:
            track (Track): The track object representing the player.

        Returns:
            tuple[float, float]: The estimated foot position (x, y) in pixels.
        """
        bbox = track.bounding_box
        return (bbox.x + bbox.width / 2, bbox.y2)  # Bottom center of the bounding box

    def _movement(
        self,
        track: Track,
        key_index: int,
        tracks_by_frame: dict[int, dict[int, Track]],
        camera_transforms: dict[int, np.ndarray],
        fps: float,
    ) -> tuple[float, float] | None:
        """
        Estimates the movement of a player in pixels per second.

        Args:
            track (Track): The track object representing the player.
            key_index (int): The index of the key frame.
            tracks_by_frame (dict[int, dict[int, Track]]): A dictionary mapping frame indices to player tracks.
            camera_transforms (dict[int, np.ndarray]): A dictionary mapping frame indices to camera motion matrices.
            fps (float): The frames per second of the video.

        Returns:
            tuple[float, float] | None: The estimated movement (dx, dy) in pixels per second, or None if not enough data.
        """
        max_offset = int(self._movement_seconds * fps)
        min_offset = max(1, max_offset // 2)  # Ensure at least one frame difference

        # Prefer the full window, but accept a shorter one for newer tracks
        for offset in range(max_offset, min_offset - 1, -1):
            earlier_track = tracks_by_frame.get(key_index - offset, {}).get(track.id)
            if earlier_track is not None:
                break
        else:
            return None  # Not enough data to estimate movement

            # Where the earlier foot position would be in the key frame's image
        earlier_point = np.array(
            [*self._foot_position(earlier_track), 1.0]
        )  # Homogeneous coordinates
        earlier_to_key_transform = camera_transforms[key_index] @ np.linalg.inv(
            camera_transforms[key_index - offset]
        )
        earlier_in_key_frame = earlier_to_key_transform @ earlier_point

        key_x, key_y = self._foot_position(track)
        seconds = offset / fps
        return (
            float((key_x - earlier_in_key_frame[0]) / seconds),
            float((key_y - earlier_in_key_frame[1]) / seconds),
        )
