import cv2
import numpy as np
from src.core.detection_class import DetectionClass
from src.core.video_boundingBox import BoundingBox
from src.core.video_track import Track
from src.utils.bbox_utils import get_bbox_width, get_center_of_bbox


class FrameAnnotator:
    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        team_assignments: dict[int, int],
        team_colours: dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Annotates the given video frame with the provided tracks.

        Args:
            frame (np.ndarray): The video frame to annotate.
            tracks (list[Track]): A list of tracked objects to annotate on the frame.
            team_assignments (dict[int, int]): A dictionary mapping track IDs to their assigned teams.
            team_colours (dict[int, np.ndarray]): A dictionary mapping team IDs to their corresponding colours.

        Returns:
            np.ndarray: The annotated video frame.
        """
        annotated = frame.copy()  # Create a copy of the frame to annotate

        self._draw_tracks(
            annotated, tracks, team_assignments, team_colours
        )  # Draw each track on the frame

        return annotated

    # Private methods
    def _draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        team_assignments: dict[int, int],
        team_colours: dict[int, np.ndarray],
    ) -> None:

        for track in tracks:
            bbox = track.bounding_box

            if track.class_id == DetectionClass.PERSON:
                team = team_assignments.get(track.id)

                if team is not None:
                    colour = team_colours[team]
                    colour = tuple(int(value) for value in colour)
                else:
                    colour = (255, 0, 255)  # Default to magenta if no team assigned

                self._draw_player_annotation(
                    frame, bbox, track, colour
                )  # Draw player annotation

            if track.class_id == DetectionClass.SPORTS_BALL:
                self._draw_triangle(frame, bbox)  # Draw triangle for ball

    def _draw_player_annotation(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        track: Track,
        colour: tuple,
    ) -> None:
        """
        Draws a player annotation on the frame.

        Args:
            frame (np.ndarray): The video frame to annotate.
            bbox: The bounding box of the detected object.
            track (Track): The tracked object.
            colour (tuple): The colour to use for the annotation.
        """
        y2 = int(bbox.y2)  # Bottom y-coordinate of the bounding box
        x_center, _ = get_center_of_bbox(bbox)
        bbox_width = int(get_bbox_width(bbox))
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(bbox_width), int(0.35 * bbox_width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        label = f"#{track.id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        rectangle_width = text_width + 8
        rectangle_height = text_height + baseline + 4
        x1_rectangle = x_center - rectangle_width // 2
        x2_rectangle = x_center + rectangle_width // 2
        y1_rectangle = y2 + 5
        y2_rectangle = y1_rectangle + rectangle_height

        if track.id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rectangle), int(y1_rectangle)),
                (int(x2_rectangle), int(y2_rectangle)),
                colour,
                cv2.FILLED,
            )

            x1_text = x_center - text_width // 2
            y_text = y1_rectangle + text_height + 2

            cv2.putText(
                frame,
                label,
                (int(x1_text), int(y_text)),
                font,
                font_scale,
                (0, 0, 0),
                text_thickness,
            )

    def _draw_triangle(self, frame: np.ndarray, bbox) -> None:
        """
        Draws a triangle on the frame representing the ball position of a detected object.

        Args:
            frame (np.ndarray): The video frame to annotate.
            bbox: The bounding box of the detected object.
        """

        y = int(bbox.y)
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],  # Top vertex
                [x - 10, y + 20],  # Bottom left vertex
                [x + 10, y + 20],  # Bottom right vertex
            ]
        )
        cv2.drawContours(
            frame, [triangle_points], 0, (255, 0, 255), cv2.FILLED
        )  # Draw filled triangle in purple
        cv2.drawContours(
            frame, [triangle_points], 0, (0, 0, 0), 2
        )  # Draw triangle border in black
