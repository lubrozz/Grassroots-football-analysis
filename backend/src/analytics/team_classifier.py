import numpy as np
from sklearn.cluster import KMeans
from src.analytics.colour_extractor import ColourExtractor
from src.analytics.jersey_cropper import JerseyCropper
from src.analytics.player_cropper import PlayerCropper
from src.core.video_track import Track


class TeamClassifier:
    def __init__(
        self, min_fit_samples: int = 100, min_votes: int = 8, vote_window: int = 30
    ):
        self.__player_cropper = PlayerCropper()
        self.__jersey_cropper = JerseyCropper()
        self.__colour_extractor = ColourExtractor()

        self.__samples: list[np.ndarray] = []
        self.__kmeans: KMeans | None = None
        self.__label_to_team: dict[int, int] = {}
        self.team_colours: dict[int, np.ndarray] = {}

        self.__votes: dict[int, list[int]] = {}
        self.__player_team: dict[int, int] = {}  # Locked team per track ID
        self.__min_fit_samples = min_fit_samples
        self.__min_votes = min_votes

    def __get_colours(
        self, frame: np.ndarray, tracks: list[Track]
    ) -> dict[int, np.ndarray]:
        crops = self.__player_cropper.crop(frame, tracks)
        jerseys = self.__jersey_cropper.crop_jersey(crops)
        return self.__colour_extractor.extract_colour(jerseys)

    def collect_samples(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """Call this on the first N frames, then call fit()."""
        self.__samples.extend(self.__get_colours(frame, tracks).values())

    def is_ready_to_fit(self) -> bool:
        return len(self.__samples) >= self.__min_fit_samples

    def fit(self) -> None:
        if len(self.__samples) < 2:
            raise ValueError("Not enough colour samples to fit teams.")

        data = np.array(self.__samples)
        self.__kmeans = KMeans(
            n_clusters=2, init="k-means++", n_init=10, random_state=0
        ).fit(data)

        order = np.argsort(
            -self.__kmeans.cluster_centers_[:, 0]
        )  # sort by L, descending
        self.__label_to_team = {int(label): team for team, label in enumerate(order)}
        self.team_colours = {
            team: self.__kmeans.cluster_centers_[label]
            for label, team in self.__label_to_team.items()
        }

    def get_team(self, frame: np.ndarray, track: Track) -> int | None:
        if self.__kmeans is None:
            raise RuntimeError("Call fit() before get_team().")

        # Team already decided for this track: keep it
        if track.id in self.__player_team:
            return self.__player_team[track.id]

        colour = self.__get_colours(frame, [track]).get(track.id)

        votes = self.__votes.setdefault(track.id, [])
        if colour is not None:
            label = int(self.__kmeans.predict(colour.reshape(1, -1))[0])
            votes.append(self.__label_to_team[label])

        if len(votes) < self.__min_votes:
            return None

        # Enough votes: decide by majority and lock the result
        team = int(np.bincount(votes, minlength=2).argmax())
        self.__player_team[track.id] = team
        del self.__votes[track.id]  # Votes are no longer needed

        return team

    def assign_teams(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, int]:
        assignments: dict[int, int] = {}

        for track in tracks:
            team = self.get_team(frame, track)
            if team is not None:
                assignments[track.id] = team

        return assignments
