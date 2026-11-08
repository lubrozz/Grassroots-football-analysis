import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from src.core.moment import MomentSnapshot


def save_moment(
    snapshot: MomentSnapshot, key_frame: np.ndarray, output_dir: Path
) -> Path:
    """
    Saves the key frame as PNG and the snapshot as JSON.

    Args:
        snapshot (MomentSnapshot): The moment snapshot to save.
        key_frame (np.ndarray): The key frame image to save.
        output_dir (Path): The directory where the files will be saved.

    Returns:
        Path: The path to the saved moment file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"moment_{snapshot.frame_index:06d}"

    cv2.imwrite(str(output_dir / f"{name}.png"), key_frame)

    json_path = output_dir / f"{name}.json"
    json_path.write_text(json.dumps(asdict(snapshot), indent=2))
    return json_path
