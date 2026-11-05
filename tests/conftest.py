"""Synthetic landmarks in MediaPipe's shapes, so no test needs the Kaggle videos or a camera."""
from __future__ import annotations

import numpy as np
import pytest

from isl.config import N_HAND


def make_clip(rng, t: int = 30):
    """pose (T,33,4: x, y, z, visibility), left/right (T,21,3), hand-present mask (T,2)."""
    pose = rng.uniform(0.2, 0.8, (t, 33, 4)).astype(np.float32)
    pose[:, 11, :2] = [0.40, 0.45]       # shoulders far enough apart for a stable body scale
    pose[:, 12, :2] = [0.60, 0.45]
    left = rng.uniform(0.25, 0.45, (t, N_HAND, 3)).astype(np.float32)
    right = rng.uniform(0.55, 0.75, (t, N_HAND, 3)).astype(np.float32)
    for hand in (left, right):           # middle-finger MCP away from the wrist: a real hand size
        hand[:, 9, :2] = hand[:, 0, :2] + [0.0, -0.08]
    return pose, left, right, np.ones((t, 2), np.float32)


@pytest.fixture
def clip():
    rng = np.random.default_rng(0)
    return lambda t=30: make_clip(rng, t)
