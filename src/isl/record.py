"""Record your own ISL clips for an unseen-signer test.

For each word it plays a reference clip from the dataset (so you can copy the sign),
counts you in, then records you doing it. Clips land in data/my_clips/<Word>/.

    python -m isl.record                          # default 5 words, 3 takes each
    python -m isl.record --words Hello Tea Jump --takes 3
    python -m isl.record --list                   # show all 61 word classes

Keys while running:  r = redo the take just recorded,  s = skip this word,  q = quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .config import ROOT

VIDEOS = ROOT / "data" / "videos" / "Video_Dataset" / "Video_Dataset"
OUT = ROOT / "data" / "my_clips"
# visually distinct, easy-to-copy signs
DEFAULT_WORDS = ["Hello", "Thank you", "Jump", "Drink", "Tea"]
WIN = "ISL recorder"


def classes() -> list[str]:
    return sorted(p.name for p in VIDEOS.iterdir() if p.is_dir())


def banner(frame, title, subtitle="", colour=(20, 160, 120)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 92), (25, 30, 40), -1)
    cv2.putText(frame, title, (22, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
    if subtitle:
        cv2.putText(frame, subtitle, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def play_reference(word: str, repeats: int = 2, speed: float = 1.0):
    """Show the dataset's own clip of this sign so the signer can copy it."""
    folder = VIDEOS / word
    clips = sorted(p for p in folder.glob("*.mp4") if "tilt" not in p.name)
    if not clips:
        return
    for _ in range(repeats):
        cap = cv2.VideoCapture(str(clips[0]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = max(1, int(1000 / (fps * speed)))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (640, 720))
            banner(frame, f"REFERENCE: {word.upper()}", "watch the sign - recording starts after this")
            cv2.imshow(WIN, frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                cap.release()
                return "quit"
        cap.release()


def countdown(cap, seconds: int, word: str, take: int, takes: int):
    t0 = time.time()
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            return
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        banner(frame, f"GET READY: {word.upper()}", f"take {take} of {takes}", (60, 180, 255))
        cv2.putText(frame, str(int(left) + 1), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (60, 180, 255), 8)
        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt


def record_take(cap, dest: Path, seconds: float, fps: float, word: str, take: int, takes: int):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed >= seconds:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        writer.write(frame)                      # save unmirrored, like the dataset
        view = cv2.flip(frame.copy(), 1)
        banner(view, f"RECORDING: {word.upper()}", f"take {take} of {takes}", (60, 60, 240))
        bar = int((elapsed / seconds) * (view.shape[1] - 44))
        cv2.rectangle(view, (22, 100), (22 + bar, 112), (60, 60, 240), -1)
        cv2.imshow(WIN, view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            writer.release()
            raise KeyboardInterrupt
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--words", nargs="+", default=DEFAULT_WORDS)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-reference", action="store_true", help="skip the example clip")
    ap.add_argument("--list", action="store_true", help="print the 61 word classes and exit")
    a = ap.parse_args()

    if a.list:
        print("\n".join(classes()))
        return

    known = {c.lower(): c for c in classes()}
    words = []
    for w in a.words:
        if w.lower() not in known:
            raise SystemExit(f"'{w}' is not one of the 61 classes - run with --list")
        words.append(known[w.lower()])

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("cannot open the camera - grant camera access and retry")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print(f"recording {len(words)} words x {a.takes} takes into {OUT}")

    try:
        for word in words:
            if not a.no_reference and play_reference(word) == "quit":
                break
            take = 1
            while take <= a.takes:
                countdown(cap, 3, word, take, a.takes)
                dest = OUT / word / f"take_{take:02d}.mp4"
                record_take(cap, dest, a.seconds, fps, word, take, a.takes)
                # review: keep, redo or skip the rest of this word
                ok, frame = cap.read()
                frame = cv2.flip(frame, 1) if ok else frame
                if ok:
                    banner(frame, f"SAVED take {take}", "ENTER keep  ·  r redo  ·  s skip word  ·  q quit")
                    cv2.imshow(WIN, frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
                if key == ord("r"):
                    dest.unlink(missing_ok=True)
                    continue
                if key == ord("s"):
                    break
                take += 1
    except KeyboardInterrupt:
        print("stopped early - clips recorded so far are kept")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    n = len(list(OUT.rglob("*.mp4")))
    print(f"\n{n} clips in {OUT}")
    print("now run:  python -m isl.unseen_signer")


if __name__ == "__main__":
    main()
