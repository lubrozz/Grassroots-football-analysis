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
        self.__min_fit_samples = min_fit_samples
        self.__min_votes = min_votes
        self.__vote_window = vote_window

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

        colour = self.__get_colours(frame, [track]).get(track.id)

        votes = self.__votes.setdefault(track.id, [])
        if colour is not None:
            label = int(self.__kmeans.predict(colour.reshape(1, -1))[0])
            votes.append(self.__label_to_team[label])
            if len(votes) > self.__vote_window:
                votes.pop(0)

        if len(votes) < self.__min_votes:
            return None

        return int(np.bincount(votes, minlength=2).argmax())

    # def assign_team_colour(self, frame: np.ndarray, tracks: list[Track]) -> None:
    #     """Assigns team colours to the detected players.

    #     Args:
    #         frame (np.ndarray): The video frame containing the players.
    #         tracks (list[Track]): A list of tracked players.

    #     Raises:
    #         ValueError: If not enough distinct colours are found to assign teams.
    #     """

    #     player_crops = self.__player_cropper.crop(frame, tracks)

    #     jersey_crops = self.__jersey_cropper.crop_jersey(player_crops)

    #     colours = self.__colour_extractor.extract_colour(jersey_crops)

    #     if len(colours) < 2:
    #         raise ValueError(
    #             "Not enough distinct colours found to assign teams. Ensure that there are at least two players."
    #         )

    #     colour_data = np.array(list(colours.values()))

    #     self.__kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)

    #     self.__kmeans.fit(colour_data)

    #     self.team_colours = {
    #         0: self.__kmeans.cluster_centers_[0],
    #         1: self.__kmeans.cluster_centers_[1],
    #     }

    #     print(
    #         f"Team 0 Colour: {self.team_colours[0]}, Team 1 Colour: {self.team_colours[1]}"
    #     )

    # def get_team(self, frame: np.ndarray, track: Track) -> int | None:
    #     """Gets the team assigned to a detected player.

    #     Args:
    #         frame (np.ndarray): The video frame containing the player.
    #         track (Track): The tracked player.

    #     Raises:
    #         RuntimeError: If team colours have not been assigned.
    #         ValueError: If no colour is found for the player.

    #     Returns:
    #         int | None: The team assigned to the player, or None if not yet assigned.
    #     """
    #     if self.__kmeans is None:
    #         raise RuntimeError("Team colours must be assigned before getting a team.")

    #     # Already assigned team, return it
    #     if track.id in self.__player_team:
    #         return self.__player_team[track.id]

    #     # Extract current colour for the player
    #     player_crops = self.__player_cropper.crop(frame, [track])
    #     jersey_crops = self.__jersey_cropper.crop_jersey(player_crops)
    #     colours = self.__colour_extractor.extract_colour(jersey_crops)

    #     player_colour = colours.get(track.id)

    #     if player_colour is None:
    #         raise ValueError(f"No colour found for track ID {track.id}.")

    #     # Store this observation for future reference
    #     colour_samples = self.__player_colours.setdefault(track.id, [])
    #     colour_samples.append(player_colour)

    #     # Not enough observations yet
    #     if len(colour_samples) < self.__colour_observation_threshold:
    #         return None  # Not enough data to assign a team yet

    #     # Use the collected colours to determine the team
    #     representative_colour = np.median(colour_samples, axis=0)

    #     team = self.__kmeans.predict(representative_colour.reshape(1, -1))[0]

    #     # Cache the team assignment for future calls
    #     self.__player_team[track.id] = team

    #     if track.id == 41:
    #         print(
    #             f"Track ID: {track.id}, Representative Colour: {representative_colour}, Assigned Team: {team}"
    #         )

    #     return int(team)  # Ensure the team is returned as an integer
