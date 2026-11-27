"""The model the apps run, and how to give it exactly the input it was trained on.

models/isl_transformer.pt is the Transformer that isl.compare_report chose on the combined
Kaggle + INCLUDE run (315 words). Its checkpoint records two things the apps must follow:

  aspect  x and the hands' depth z are put in frame-height units (features.build_frame_features),
          so every caller passes the frame's width / height
  split   the artifacts/split_<split>.json it was trained on, so "held-out" means its own test set

The older 61-word models/bilstm_orig.pt has neither, and still loads with the original features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import ARTIFACTS, DATA_KP, MODELS, ROOT, TrainConfig
from .dataset import Sample, group_of, index_keypoints, stratified_split
from .models import build_model

SHIPPED = MODELS / "isl_transformer.pt"


def load_model(ckpt: Path = SHIPPED, device="cpu"):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(state["arch"], len(state["classes"]), TrainConfig()).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state["classes"], state


def aspect_of(state: dict, width: float, height: float) -> float | None:
    """The `aspect` argument of build_frame_features for this model and a frame of this size."""
    return float(width) / float(height) if state.get("aspect") and width and height else None


def video_size(path: Path) -> tuple[int, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def held_out(state: dict, kaggle_only: bool = False) -> list[Sample]:
    """The test clips this model never trained on."""
    if state.get("split"):
        split = json.loads((ARTIFACTS / f"split_{state['split']}.json").read_text())
        test = []
        for rel in split["test"]:
            p = ROOT / rel
            with np.load(p, allow_pickle=False) as z:
                test.append(Sample(p, str(z["label"]), group_of(p)))
    else:
        cfg = TrainConfig()
        _, _, test = stratified_split(index_keypoints([DATA_KP]), cfg.val_frac, cfg.test_frac, cfg.seed)
    if kaggle_only:
        test = [s for s in test if DATA_KP in s.path.parents]
    return test
