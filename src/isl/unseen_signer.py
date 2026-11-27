"""Protocol D: test the trained model on a signer it has never seen (you).

Extracts landmarks from data/my_clips/<Word>/*.mp4, runs the trained classifier over
all the model's classes, and reports top-1 / top-5 with a per-clip table and a slide figure.

    python -m isl.unseen_signer                               # the shipped 315-word Transformer
    python -m isl.unseen_signer --ckpt models/bilstm_orig.pt  # the original 61-word BiLSTM
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ARTIFACTS, ROOT, TrainConfig
from .features import build_frame_features, resample, with_deltas
from .shipped import SHIPPED, aspect_of, load_model, video_size
from .train import device_auto

CLIPS = ROOT / "data" / "my_clips"
ACCENT, AMBER, GREY = "#0F766E", "#F59E0B", "#94A3B8"


def label_from_dir(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def figure(rows, acc1, acc5, path: Path, n_classes: int = 61):
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.42 * len(rows) + 1.6)))
    words = [r["word"] for r in rows]
    conf = [r["confidence"] * 100 for r in rows]
    colors = [ACCENT if r["correct"] else (AMBER if r["in_top5"] else GREY) for r in rows]
    y = range(len(rows))
    ax.barh(list(y), conf, color=colors)
    for i, r in enumerate(rows):
        mark = "correct" if r["correct"] else ("in top-5" if r["in_top5"] else f"-> {r['predicted']}")
        ax.text(min(conf[i] + 1.5, 86), i, mark, va="center", fontsize=9, color="#1F2937")
    ax.set_yticks(list(y), words, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("model confidence in its top prediction (%)")
    ax.set_title(f"Unseen signer — top-1 {acc1*100:.0f}%, top-5 {acc5*100:.0f}%  "
                 f"({len(rows)} clips, {n_classes}-way)", fontsize=11)
    ax.grid(axis="x", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=SHIPPED)
    ap.add_argument("--clips", type=Path, default=CLIPS)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()

    videos = sorted(p for p in a.clips.rglob("*.mp4") if not p.name.startswith("."))
    if not videos:
        raise SystemExit(f"no clips in {a.clips} - record some first: python -m isl.record")

    cfg, device = TrainConfig(), device_auto()
    model, classes, state = load_model(a.ckpt, device)
    idx = {c: i for i, c in enumerate(classes)}

    from .keypoints import LandmarkExtractor
    rows, ys, p1, p5 = [], [], [], []
    with LandmarkExtractor() as ex:
        for v in videos:
            truth = label_from_dir(v.parent.name)
            if truth not in idx:
                print(f"skip {v.name}: '{truth}' is not a training class")
                continue
            pose, left, right, mask, _ = ex.run_video(v, stride=a.stride)
            seq = build_frame_features(pose, left, right, mask, aspect_of(state, *video_size(v)))
            x = torch.from_numpy(with_deltas(resample(seq, cfg.seq_len)))[None].to(device)
            with torch.no_grad():
                prob = model(x).softmax(-1)[0].cpu().numpy()
            top5 = prob.argsort()[-5:][::-1]
            pred = classes[top5[0]]
            rows.append({"clip": v.name, "word": v.parent.name, "predicted": pred,
                         "confidence": float(prob[top5[0]]),
                         "correct": pred == truth,
                         "in_top5": truth in [classes[i] for i in top5],
                         "top5": [classes[i] for i in top5]})
            ys.append(truth); p1.append(pred); p5.append([classes[i] for i in top5])
            flag = "OK " if pred == truth else ("~5 " if truth in rows[-1]["top5"] else "X  ")
            print(f"{flag}{v.parent.name:<20} -> {pred:<20} {prob[top5[0]]*100:5.1f}%   "
                  f"top5: {', '.join(rows[-1]['top5'][:5])}")

    acc1 = float(np.mean([t == p for t, p in zip(ys, p1)]))
    acc5 = float(np.mean([t in k for t, k in zip(ys, p5)]))
    words = sorted({r["word"] for r in rows})
    per_word = {w: float(np.mean([r["correct"] for r in rows if r["word"] == w])) for w in words}
    print(f"\nunseen signer: top-1 {acc1*100:.1f}%  top-5 {acc5*100:.1f}%  "
          f"over {len(rows)} clips of {len(words)} words ({len(classes)}-way, chance {100 / len(classes):.1f}%)")
    for w, s in per_word.items():
        print(f"  {w:<22} {s*100:5.1f}%")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "unseen_signer.json").write_text(json.dumps(
        {"top1": acc1, "top5": acc5, "n_clips": len(rows), "n_words": len(words),
         "per_word": per_word, "clips": rows}, indent=2))
    figure(rows, acc1, acc5, ARTIFACTS / "unseen_signer.png", len(classes))
    print("wrote artifacts/unseen_signer.json and .png")


if __name__ == "__main__":
    main()
