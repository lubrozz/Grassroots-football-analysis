from pathlib import Path

from src.video.loader import VideoLoader
from src.video.player import VideoPlayer


def main() -> None:
    video_path = Path("data/input/veo-tracker-highlights-test.mp4")
    loader = VideoLoader(video_path)
    player = VideoPlayer(loader)

    try:
        metadata = loader.metadata

        print("Video information")
        print("-----------------")
        print(f"Resolution : {metadata.width}x{metadata.height}")
        print(f"FPS        : {metadata.fps:.2f}")
        print(f"Frames     : {metadata.frame_count}")
        print(f"Duration   : {metadata.duration_seconds:.2f} s")

        player.play()

    finally:
        player.close()
        loader.close()


if __name__ == "__main__":
    main()
