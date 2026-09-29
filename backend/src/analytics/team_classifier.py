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

        self.__kmeans: KMeans | None = None  # Store the KMeans model for later use

        self.team_colours: dict[
            int, np.ndarray
        ] = {}  # Store team colours for later use

        # Store colour observations until enough is collected
        self.__player_colours: dict[int, list[np.ndarray]] = {}

        # Final team assignments after KMeans clustering
        self.__player_team: dict[int, int] = {}

        # Number of colour observations needed before performing KMeans clustering
        self.__colour_observation_threshold: int = 5

    def assign_team_colour(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """Assigns team colours to the detected players.

        Args:
            frame (np.ndarray): The video frame containing the players.
            tracks (list[Track]): A list of tracked players.

        Raises:
            ValueError: If not enough distinct colours are found to assign teams.
        """

        player_crops = self.__player_cropper.crop(frame, tracks)

        jersey_crops = self.__jersey_cropper.crop_jersey(player_crops)

        colours = self.__colour_extractor.extract_colour(jersey_crops)

        if len(colours) < 2:
            raise ValueError(
                "Not enough distinct colours found to assign teams. Ensure that there are at least two players."
            )

        colour_data = np.array(list(colours.values()))

        self.__kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)

        self.__kmeans.fit(colour_data)

        self.team_colours = {
            0: self.__kmeans.cluster_centers_[0],
            1: self.__kmeans.cluster_centers_[1],
        }

        print(
            f"Team 0 Colour: {self.team_colours[0]}, Team 1 Colour: {self.team_colours[1]}"
        )

    def get_team(self, frame: np.ndarray, track: Track) -> int | None:
        """Gets the team assigned to a detected player.

        Args:
            frame (np.ndarray): The video frame containing the player.
            track (Track): The tracked player.

        Raises:
            RuntimeError: If team colours have not been assigned.
            ValueError: If no colour is found for the player.

        Returns:
            int | None: The team assigned to the player, or None if not yet assigned.
        """
        if self.__kmeans is None:
            raise RuntimeError("Team colours must be assigned before getting a team.")

        # Already assigned team, return it
        if track.id in self.__player_team:
            return self.__player_team[track.id]

        # Extract current colour for the player
        player_crops = self.__player_cropper.crop(frame, [track])
        jersey_crops = self.__jersey_cropper.crop_jersey(player_crops)
        colours = self.__colour_extractor.extract_colour(jersey_crops)

        player_colour = colours.get(track.id)

        if player_colour is None:
            raise ValueError(f"No colour found for track ID {track.id}.")

        # Store this observation for future reference
        colour_samples = self.__player_colours.setdefault(track.id, [])
        colour_samples.append(player_colour)

        # Not enough observations yet
        if len(colour_samples) < self.__colour_observation_threshold:
            return None  # Not enough data to assign a team yet

        # Use the collected colours to determine the team
        representative_colour = np.median(colour_samples, axis=0)

        team = self.__kmeans.predict(representative_colour.reshape(1, -1))[0]

        # Cache the team assignment for future calls
        self.__player_team[track.id] = team

        if track.id == 41:
            print(
                f"Track ID: {track.id}, Representative Colour: {representative_colour}, Assigned Team: {team}"
            )

        return int(team)  # Ensure the team is returned as an integer
