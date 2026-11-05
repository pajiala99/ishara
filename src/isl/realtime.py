"""Live demo: webcam -> landmarks -> sliding-window prediction -> text + speech.

    python -m isl.realtime --ckpt models/bilstm_orig.pt --speak

Recognised words are spoken as they are confirmed and appended to a sentence line at
the bottom of the window, so a sequence of signs builds a sentence you can replay.

Keys:  space = speak the whole sentence   c = clear it   m = mute/unmute   q = quit
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import MODELS, SEQ_LEN, TrainConfig
from .features import build_frame_features, resample, with_deltas
from .keypoints import LandmarkExtractor
from .models import build_model
from .speech import Speaker
from .train import device_auto

WIN = "ISL recognition"


def draw(frame, label, conf, threshold, sentence, muted, voice):
    h, w = frame.shape[:2]
    live = conf >= threshold
    colour = (20, 160, 120) if live else (130, 130, 130)

    cv2.rectangle(frame, (0, 0), (w, 78), (25, 30, 40), -1)
    cv2.putText(frame, label.replace("_", " ").upper(), (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
    cv2.putText(frame, f"{conf*100:5.1f}%", (w - 150, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # confidence bar: shows how close the current guess is to the speaking threshold
    cv2.rectangle(frame, (20, 62), (w - 160, 68), (60, 65, 80), -1)
    cv2.rectangle(frame, (20, 62), (20 + int((w - 180) * min(conf, 1.0)), 68), colour, -1)
    gate = 20 + int((w - 180) * threshold)
    cv2.line(frame, (gate, 58), (gate, 72), (240, 200, 60), 2)

    cv2.rectangle(frame, (0, h - 86), (w, h), (25, 30, 40), -1)
    text = " ".join(x.replace("_", " ") for x in sentence) or "(sign to build a sentence)"
    if len(text) > 58:
        text = "..." + text[-55:]
    cv2.putText(frame, text, (20, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    status = f"voice: {'MUTED' if muted else voice}   space speak   c clear   m mute   q quit"
    cv2.putText(frame, status, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=MODELS / "bilstm_orig.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=int, default=SEQ_LEN)
    ap.add_argument("--stride", type=int, default=5, help="predict every N frames")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--hold", type=int, default=2,
                    help="consecutive agreeing predictions before a word is accepted")
    ap.add_argument("--speak", action="store_true", help="speak each confirmed word")
    ap.add_argument("--voice", default=None, help="macOS voice name (default: Indian English)")
    ap.add_argument("--rate", type=int, default=170)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes, cfg = state["classes"], TrainConfig()
    device = device_auto()
    model = build_model(state["arch"], len(classes), cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    speaker = Speaker(enabled=args.speak, voice=args.voice, rate=args.rate)
    print(f"{len(classes)} classes on {device} | speech: {speaker.describe()}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("cannot open the camera - grant camera access and retry")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    buf = {k: deque(maxlen=args.window) for k in ("pose", "left", "right", "mask")}
    sentence: list[str] = []
    label, conf, i = "...", 0.0, 0
    streak_word, streak = "", 0
    muted = not args.speak

    try:
        with LandmarkExtractor(running_mode="video") as ex:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # landmarks come from the frame as the camera filmed it, like the dataset clips (record.py
                # saves takes unmirrored for the same reason); only the window below shows a mirror image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                p, l, r, m = ex._frame(rgb, int(cap.get(cv2.CAP_PROP_POS_MSEC)) or i * 33)
                buf["pose"].append(p); buf["left"].append(l)
                buf["right"].append(r); buf["mask"].append(m)

                if len(buf["pose"]) == args.window and i % args.stride == 0:
                    seq = build_frame_features(np.stack(buf["pose"]), np.stack(buf["left"]),
                                               np.stack(buf["right"]), np.stack(buf["mask"]))
                    x = torch.from_numpy(with_deltas(resample(seq, cfg.seq_len)))[None].to(device)
                    with torch.no_grad():
                        prob = model(x).softmax(-1)[0].cpu().numpy()
                    k = int(prob.argmax())
                    conf = float(prob[k])
                    label = classes[k] if conf >= args.threshold else "..."

                    # a word counts only after `hold` agreeing predictions in a row,
                    # which stops transitions between signs being spoken as words
                    if label != "...":
                        streak = streak + 1 if label == streak_word else 1
                        streak_word = label
                        if streak == args.hold and (not sentence or sentence[-1] != label):
                            sentence.append(label)
                            if not muted:
                                speaker.say(label)
                    else:
                        streak, streak_word = 0, ""

                voice_name = speaker.voice or speaker.backend
                cv2.imshow(WIN, draw(cv2.flip(frame, 1), label, conf, args.threshold, sentence, muted, voice_name))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    speaker.say_sentence(sentence)
                if key == ord("c"):
                    sentence.clear()
                if key == ord("m"):
                    muted = not muted
                i += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sentence:
            print("sentence:", " ".join(w.replace("_", " ") for w in sentence))
        speaker.close()


if __name__ == "__main__":
    main()
