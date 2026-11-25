"""Add each clip's frame size (width, height) to keypoint files extracted before isl.extract saved it.

features.build_frame_features needs width / height to put x in frame-height units; the Kaggle
videos come as 640x720, 760x720 and 880x720 and INCLUDE as 1920x1080, so a single constant would
be wrong. Files that already have the fields are left alone.

    python tools/add_frame_size.py data/keypoints data/keypoints_tilt data/include/keypoints
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isl.pixel import video_of  # noqa: E402  (keypoint folder -> video folder map lives there)


def main(dirs):
    done = skipped = 0
    for d in dirs:
        for p in sorted(Path(d).resolve().rglob("*.npz")):
            with np.load(p, allow_pickle=False) as z:
                if "width" in z.files:
                    skipped += 1
                    continue
                data = {k: z[k] for k in z.files}
            cap = cv2.VideoCapture(str(video_of(p)))
            data["width"], data["height"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            np.savez_compressed(p, **data)
            done += 1
    print(f"frame size added to {done} files, {skipped} already had it")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/keypoints", "data/keypoints_tilt", "data/include/keypoints"])
