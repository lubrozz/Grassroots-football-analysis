import cv2
from src.analytics.moment_analyzer import MomentAnalyzer
from src.api.serialization import save_moment
from src.config import settings
from src.core.moment import Moment
from src.detection.yolo_detector import YoloDetector
from src.utils.time_utils import parse_timestamp
from src.video.annotator import FrameAnnotator
from src.video.loader import VideoLoader

""" Known limitations:
 - if a moments shows almost only one team, KMeans splits that one team into two. A later improvement would be to fit the team model once per match.
 - Match clock does not necessarily match the video time. Kickoff could be at 3:20 in the video, and an inputted timestamp of 12:34 would be 15:54 in the video. 
   A later improvement would be to allow the user to input the kickoff time, and then all timestamps would be relative to that.
"""


def main() -> None:
    video_path = settings.INPUT_VIDEO
    output_dir = settings.OUTPUT_DIR / "moments" / video_path.stem

    loader = VideoLoader(video_path)
    annotator = FrameAnnotator()
    analyzer = MomentAnalyzer(
        detector=YoloDetector(),
        seconds_before=settings.MOMENT_SECONDS_BEFORE,
        seconds_after=settings.MOMENT_SECONDS_AFTER,
        movement_seconds=settings.MOMENT_MOVEMENT_SECONDS,
        max_step_heights=settings.MOMENT_MAX_STEP_HEIGHTS,
    )

    try:
        for timestamp in settings.MOMENT_TIMESTAMPS:
            time_seconds = parse_timestamp(timestamp)
            snapshot, key_frame = analyzer.analyse(
                loader, Moment(time_seconds=time_seconds)
            )
            json_path = save_moment(snapshot, key_frame, output_dir)
            if settings.MOMENT_DEBUG_IMAGES:
                debug_path = json_path.with_name(f"{json_path.stem}_debug.png")
                cv2.imwrite(str(debug_path), annotator.draw_moment(key_frame, snapshot))
            print(
                f"{time_seconds:.1f} s: {len(snapshot.players)} players -> saved to {json_path}"
            )
    except Exception as e:
        print(f"Error occurred while analyzing moments: {e}")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
