from pathlib import Path

from src.video.loader import VideoLoader


def main() -> None:
    video_path = Path("data/input/veo-tracker-highlights-test.mp4")
    loader = VideoLoader(video_path)

    try:
        metadata = loader.metadata

        print("Video information")
        print("-----------------")
        print(f"Resolution : {metadata.width}x{metadata.height}")
        print(f"FPS        : {metadata.fps:.2f}")
        print(f"Frames     : {metadata.frame_count}")
        print(f"Duration   : {metadata.duration_seconds:.2f} s")

        frame_count = 0

        while True:
            frame = loader.read()
            if frame is None:
                break

            frame_count += 1
        print(f"Frames read: {frame_count}")

    finally:
        loader.close


if __name__ == "__main__":
    main()
