"""Index the cached landmarks, split them, and serve augmented tensors to torch."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DATA_KP, SEQ_LEN, N_HAND, POSE_IDX
from .features import build_frame_features, resample, with_deltas


@dataclass
class Sample:
    path: Path
    label: str
    group: str = ""        # recording-session id: tilt variants of one take share it


TILT_RE = re.compile(r"_(left|right)_tilt$", re.I)


def group_of(path: Path) -> str:
    """This dataset ships each take three times (original, left tilt, right tilt).
    Those three clips are near-duplicates, so they must never be split across
    train and test - we key them to one group and split by group."""
    return TILT_RE.sub("", path.stem)


def index_keypoints(kp_dir=DATA_KP, min_per_class: int = 2) -> list[Sample]:
    dirs = [kp_dir] if isinstance(kp_dir, (str, Path)) else list(kp_dir)
    samples = []
    for p in sorted(q for d in dirs for q in Path(d).rglob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            label = str(z["label"])
        samples.append(Sample(p, label, group_of(p)))
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.label] = counts.get(s.label, 0) + 1
    return [s for s in samples if counts[s.label] >= min_per_class]


def stratified_split(samples: list[Sample], val_frac: float, test_frac: float, seed: int):
    """Per-class stratified, group-aware split.

    Stratified so every word appears in train/val/test; group-aware so the
    near-duplicate tilt variants of one recording stay on the same side of the
    split. Splitting those independently would leak the test set into training
    and inflate accuracy by a wide margin.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, dict[str, list[Sample]]] = {}
    for s in samples:
        by_class.setdefault(s.label, {}).setdefault(s.group or str(s.path), []).append(s)
    train, val, test = [], [], []
    for label, groups in sorted(by_class.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, int(round(n * test_frac))) if n >= 4 else 1
        n_val = max(1, int(round(n * val_frac))) if n >= 6 else (1 if n >= 3 else 0)
        for k in keys[:n_test]:
            test += groups[k]
        for k in keys[n_test:n_test + n_val]:
            val += groups[k]
        for k in keys[n_test + n_val:]:
            train += groups[k]
    return train, val, test


def load_sequence(path: Path) -> np.ndarray:
    """.npz of raw landmarks -> (T, F_BASE) engineered features."""
    with np.load(path, allow_pickle=False) as z:
        return build_frame_features(z["pose"], z["left"], z["right"], z["mask"])


# ----------------------------------------------------------------- augmentation
def _rot_scale_shift(seq: np.ndarray, rot_deg, scale_j, shift_j, rng) -> np.ndarray:
    """Apply a small 2-D similarity transform to every (x, y) pair in the frame."""
    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    s = 1.0 + rng.uniform(-scale_j, scale_j)
    dx, dy = rng.uniform(-shift_j, shift_j, 2)
    c, sn = np.cos(th) * s, np.sin(th) * s
    out = seq.copy()
    n_pose = len(POSE_IDX)
    # pose block: (x, y, vis) triples
    for i in range(n_pose):
        x, y = out[:, i * 3], out[:, i * 3 + 1]
        out[:, i * 3] = c * x - sn * y + dx
        out[:, i * 3 + 1] = sn * x + c * y + dy
    # global hand block: (x, y) pairs
    off = n_pose * 3
    for i in range(2 * N_HAND * 2 // 2):
        x, y = out[:, off + 2 * i], out[:, off + 2 * i + 1]
        out[:, off + 2 * i] = c * x - sn * y + dx
        out[:, off + 2 * i + 1] = sn * x + c * y + dy
    return out


# MediaPipe names pose landmarks from the signer's side, so a mirror image swaps these pairs
# (shoulders, elbows, wrists, hips); the nose stays where it is.
_POSE_LR = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23}
_POSE_MIRROR = [POSE_IDX.index(_POSE_LR.get(j, j)) for j in POSE_IDX]


def _mirror(seq: np.ndarray) -> np.ndarray:
    """Left/right flip, as if the sign were made with the other dominant hand: negate x
    and swap left and right everywhere - the paired arm and hip landmarks of the pose,
    the two hand blocks and their presence flags."""
    n_pose = len(POSE_IDX)
    out = seq.copy()
    pose = out[:, :n_pose * 3].reshape(len(out), n_pose, 3)[:, _POSE_MIRROR]
    pose[..., 0] *= -1.0
    out[:, :n_pose * 3] = pose.reshape(len(out), -1)
    g0 = n_pose * 3
    g1 = g0 + N_HAND * 2
    g2 = g1 + N_HAND * 2
    gl, gr = out[:, g0:g1].copy(), out[:, g1:g2].copy()
    gl[:, 0::2] *= -1.0
    gr[:, 0::2] *= -1.0
    out[:, g0:g1], out[:, g1:g2] = gr, gl
    l0 = g2
    l1 = l0 + N_HAND * 3
    l2 = l1 + N_HAND * 3
    ll, lr = out[:, l0:l1].copy(), out[:, l1:l2].copy()
    ll[:, 0::3] *= -1.0
    lr[:, 0::3] *= -1.0
    out[:, l0:l1], out[:, l1:l2] = lr, ll
    out[:, l2], out[:, l2 + 1] = out[:, l2 + 1].copy(), out[:, l2].copy()
    return out


def augment(seq: np.ndarray, cfg: dict, rng) -> np.ndarray:
    if rng.random() < cfg["p_mirror"]:
        seq = _mirror(seq)
    # temporal jitter: resample at a random speed, then crop/pad back
    if cfg["speed"] > 0:
        f = 1.0 + rng.uniform(-cfg["speed"], cfg["speed"])
        seq = resample(seq, max(8, int(round(len(seq) * f))))
    if rng.random() < cfg["p_frame_drop"]:
        keep = rng.random(len(seq)) > 0.1
        if keep.sum() >= 8:
            seq = seq[keep]
    seq = _rot_scale_shift(seq, cfg["rot_deg"], cfg["scale"], cfg["shift"], rng)
    if cfg["noise"] > 0:
        seq = seq + rng.normal(0, cfg["noise"], seq.shape).astype(np.float32)
    return seq.astype(np.float32)


class ISLSequenceDataset(Dataset):
    def __init__(self, samples, classes, seq_len=SEQ_LEN, train=False,
                 aug_cfg=None, seed=0, cache=True):
        self.samples = samples
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.seq_len = seq_len
        self.train = train
        self.aug_cfg = aug_cfg or {}
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, np.ndarray] = {} if cache else None

    def __len__(self):
        return len(self.samples)

    def raw(self, i: int) -> np.ndarray:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        seq = load_sequence(self.samples[i].path)
        if self._cache is not None:
            self._cache[i] = seq
        return seq

    def __getitem__(self, i):
        seq = self.raw(i)
        if self.train and self.aug_cfg:
            seq = augment(seq, self.aug_cfg, self.rng)
        seq = with_deltas(resample(seq, self.seq_len))
        y = self.cls_to_idx[self.samples[i].label]
        return torch.from_numpy(seq), y


def save_classes(classes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(classes, indent=2))


def load_classes(path: Path) -> list[str]:
    return json.loads(path.read_text())
