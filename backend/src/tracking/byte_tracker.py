from src.core.video_detection import Detection
from src.core.video_track import Track


class ByteTracker:
    def __init__(self):
        # Initialize the tracker state here
        self.tracks = []

    def update(self, detections: list[Detection]) -> list[Track]:
        # Update the tracker with new detections and return the updated tracks
        # For simplicity, we'll just return the detections as tracks in this example
        self.tracks = detections  # In a real implementation, you'd match detections to existing tracks
        return self.tracks

    def reset(self) -> None:
        # Reset the tracker state
        self.tracks = []
