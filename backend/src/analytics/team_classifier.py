import numpy as np
from sklearn.cluster import KMeans
from src.analytics.colour_extractor import ColourExtractor
from src.analytics.jersey_cropper import JerseyCropper
from src.analytics.player_cropper import PlayerCropper
from src.core.video_track import Track


class TeamClassifier:
    def __init__(self):
        self.__player_cropper = PlayerCropper()
        self.__jersey_cropper = JerseyCropper()
        self.__colour_extractor = ColourExtractor()

        self.__track_colours: dict[int, list[np.ndarray]] = {}  # All colours per track
        self.__kmeans: KMeans | None = None
        self.__label_to_team: dict[int, int] = {}
        self.team_colours: dict[int, np.ndarray] = {}

    def __get_colours(
        self, frame: np.ndarray, tracks: list[Track]
    ) -> dict[int, np.ndarray]:
        crops = self.__player_cropper.crop(frame, tracks)
        jerseys = self.__jersey_cropper.crop_jersey(crops)
        return self.__colour_extractor.extract_colour(jerseys)

    def add_observations(self, frame: np.ndarray, tracks: list[Track]) -> None:
        colours = self.__get_colours(frame, tracks)
        for track_id, colour in colours.items():
            self.__track_colours.setdefault(track_id, []).append(colour)

    def fit(self) -> None:
        data = np.concatenate(
            [np.array(colours) for colours in self.__track_colours.values()]
        )

        if len(data) < 2:
            raise ValueError("Not enough colour samples to fit teams.")

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

    def assign_teams(self, debug_track_ids: set[int] | None = None) -> dict[int, int]:
        if self.__kmeans is None:
            raise RuntimeError("Call fit() before assign_teams().")

        assignments: dict[int, int] = {}
        for track_id, colours in self.__track_colours.items():
            labels = self.__kmeans.predict(np.array(colours))
            teams = [self.__label_to_team[int(label)] for label in labels]
            votes = np.bincount(teams, minlength=2)
            assignments[track_id] = int(votes.argmax())

            if debug_track_ids is not None and track_id in debug_track_ids:
                self.__print_track_debug(track_id, colours, votes)

        return assignments

    def __print_track_debug(
        self, track_id: int, colours: list[np.ndarray], votes: np.ndarray
    ) -> None:
        median_colour = np.median(np.array(colours), axis=0)
        distances = {
            team: float(np.linalg.norm(median_colour - centre))
            for team, centre in self.team_colours.items()
        }
        print(
            f"Track {track_id}: {len(colours)} observations | "
            f"votes team 0/1 = {votes[0]}/{votes[1]} | "
            f"median LAB = {np.round(median_colour, 1)} | "
            f"distance to team 0 = {distances[0]:.1f}, team 1 = {distances[1]:.1f}"
        )
