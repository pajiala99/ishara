"""Check the trained model yourself, on clips it never saw during training.

    python -m isl.demo --random 10          # 10 random held-out test clips, scored
    python -m isl.demo --random 5 --play    # ...and watch each clip with its prediction
    python -m isl.demo --word Tea --play    # a held-out clip of one specific word
    python -m isl.demo --video path/to.mp4  # any video file you point at

Only clips from the held-out test split are used, so nothing here was trained on.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .config import DATA_KP, ROOT, TrainConfig
from .dataset import load_sequence
from .features import build_frame_features, resample, with_deltas
from .shipped import SHIPPED, aspect_of, held_out, load_model, video_size
from .train import device_auto

VIDEOS = ROOT / "data" / "videos" / "Video_Dataset" / "Video_Dataset"


def video_for(npz: Path) -> Path | None:
    """Map a cached landmark file back to the video it came from (Kaggle clips; None for others)."""
    if DATA_KP not in npz.parents:
        return None
    rel = npz.relative_to(DATA_KP)
    cand = VIDEOS / rel.with_suffix(".mp4")
    return cand if cand.exists() else None


def predict(model, device, seq: np.ndarray, classes, seq_len: int):
    x = torch.from_numpy(with_deltas(resample(seq, seq_len)))[None].to(device)
    with torch.no_grad():
        prob = model(x).softmax(-1)[0].cpu().numpy()
    order = prob.argsort()[::-1][:5]
    return [(classes[i], float(prob[i])) for i in order]


def play(video: Path, truth: str, top5, correct: bool, writer=None, show=True):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    colour = (20, 160, 120) if correct else (60, 60, 240)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 720))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 112), (25, 30, 40), -1)
        cv2.putText(frame, f"PREDICTED: {top5[0][0].replace('_',' ').upper()}", (18, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        cv2.putText(frame, f"{top5[0][1]*100:.1f}%   truth: {truth.replace('_',' ')}", (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        for i, (c, pr) in enumerate(top5[1:4], start=2):
            cv2.putText(frame, f"{i}. {c.replace('_',' ')} {pr*100:.0f}%", (18, 118 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
        if writer is not None:
            writer.write(frame)
        if not show:
            continue
        cv2.putText(frame, "space = next   q = quit", (18, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.imshow("ISL demo", frame)
        k = cv2.waitKey(int(1000 / fps)) & 0xFF
        if k == ord("q"):
            cap.release(); cv2.destroyAllWindows(); raise KeyboardInterrupt
        if k == ord(" "):
            break
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=SHIPPED)
    ap.add_argument("--random", type=int, default=10, help="how many held-out clips to test")
    ap.add_argument("--word", default=None, help="restrict to one word class")
    ap.add_argument("--video", type=Path, default=None, help="predict on any video file")
    ap.add_argument("--play", action="store_true", help="show each clip in a window")
    ap.add_argument("--speak", action="store_true", help="speak each prediction aloud")
    ap.add_argument("--save", type=Path, default=None,
                    help="record the played clips with their predictions to an mp4")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    cfg, device = TrainConfig(), device_auto()
    model, classes, state = load_model(a.ckpt, device)
    from .speech import Speaker
    speaker = Speaker(enabled=a.speak)
    print(f"loaded {a.ckpt.name}: {state['arch']}, {len(classes)} classes, device {device}"
          + (f" | speech: {speaker.describe()}" if a.speak else "") + "\n")

    # single arbitrary video
    if a.video:
        from .keypoints import LandmarkExtractor
        with LandmarkExtractor() as ex:
            pose, left, right, mask, _ = ex.run_video(a.video, stride=3)
        aspect = aspect_of(state, *video_size(a.video))
        top5 = predict(model, device, build_frame_features(pose, left, right, mask, aspect), classes, cfg.seq_len)
        print(f"{a.video.name}")
        for i, (c, p) in enumerate(top5, 1):
            print(f"  {i}. {c.replace('_',' '):<22} {p*100:5.1f}%")
        return

    test_s = held_out(state)
    if a.word:
        key = a.word.strip().lower().replace(" ", "_")
        test_s = [s for s in test_s if s.label == key]
        if not test_s:
            raise SystemExit(f"'{a.word}' has no held-out clip - try another word")
    rng = random.Random(a.seed)
    picks = rng.sample(test_s, min(a.random, len(test_s)))

    print(f"{len(picks)} held-out clips (never trained on), {len(classes)}-way choice:\n")
    hits = 0
    writer = None
    if a.save:
        import cv2
        writer = cv2.VideoWriter(str(a.save), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 720))
    try:
        for s in picks:
            top5 = predict(model, device, load_sequence(s.path, aspect_correct=bool(state.get("aspect"))),
                           classes, cfg.seq_len)
            ok = top5[0][0] == s.label
            hits += ok
            print(f"{'OK ' if ok else 'X  '}truth {s.label.replace('_',' '):<20} -> "
                  f"{top5[0][0].replace('_',' '):<20} {top5[0][1]*100:5.1f}%   "
                  f"next: {', '.join(c.replace('_',' ') for c, _ in top5[1:4])}")
            if a.speak:
                speaker.say(top5[0][0], force=True)
            if a.play or writer is not None:
                v = video_for(s.path)
                if v:
                    play(v, s.label, top5, ok, writer=writer, show=a.play)
    except KeyboardInterrupt:
        print("\nstopped")
    speaker.close()
    if writer is not None:
        writer.release()
        print(f"wrote {a.save}")
    if picks:
        print(f"\n{hits}/{len(picks)} correct ({hits/len(picks)*100:.0f}%) on held-out clips")


if __name__ == "__main__":
    main()
