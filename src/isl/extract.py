"""Stage 1 CLI: run MediaPipe over every INCLUDE video and cache landmarks as .npz.

    python -m isl.extract --videos data/videos --out data/keypoints --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import DATA_KP, DATA_VIDEOS, VIDEO_EXT


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."))


def label_of(path: Path, root: Path) -> str:
    """INCLUDE layout: <Category>/<N. Word>/<clip>.MOV -> label 'Word'."""
    rel = path.relative_to(root)
    word = rel.parent.name
    if ". " in word:
        word = word.split(". ", 1)[1]
    return word.strip().lower().replace(" ", "_")


def _out_path(video: Path, root: Path, out_dir: Path) -> Path:
    rel = video.relative_to(root).with_suffix(".npz")
    return out_dir / rel


def _worker(args) -> tuple[str, str]:
    video, root, out_dir, stride = (Path(args[0]), Path(args[1]), Path(args[2]), args[3])
    from .keypoints import LandmarkExtractor        # import inside the worker process
    dest = _out_path(video, root, out_dir)
    if dest.exists():
        return str(video), "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LandmarkExtractor() as ex:
            pose, left, right, mask, fps = ex.run_video(video, stride=stride)
        import cv2                                   # frame size, for the aspect correction in features.py
        cap = cv2.VideoCapture(str(video))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        np.savez_compressed(dest, pose=pose, left=left, right=right, mask=mask,
                            fps=fps, width=width, height=height, label=label_of(video, root),
                            category=video.relative_to(root).parts[0], source=video.name)
        return str(video), "ok"
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return str(video), "fail"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, default=DATA_VIDEOS)
    ap.add_argument("--out", type=Path, default=DATA_KP)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="", help="regex; matching paths are skipped")
    a = ap.parse_args()

    videos = find_videos(a.videos)
    if a.exclude:
        import re
        pat = re.compile(a.exclude)
        videos = [v for v in videos if not pat.search(str(v))]
    if a.limit:
        videos = videos[: a.limit]
    print(f"{len(videos)} videos under {a.videos}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)

    jobs = [(str(v), str(a.videos), str(a.out), a.stride) for v in videos]
    done = {"ok": 0, "skip": 0, "fail": 0}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_worker, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            _, status = f.result()
            done[status] += 1
            if i % 25 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {done}", flush=True)
    print("done:", done)


if __name__ == "__main__":
    main()
