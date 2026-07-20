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

        frames_read = 0

        while True:
            frame = loader.read()
            if frame is None:
                break

            frames_read += 1
        print(f"Frames read: {frames_read}")

    finally:
        loader.close()


if __name__ == "__main__":
    main()
