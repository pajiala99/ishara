"""The split that guards against leakage, augmentation, and the tensors the model trains on."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isl.config import F_BASE, F_TOTAL, N_HAND, N_POSE, SEQ_LEN, TrainConfig
from isl.dataset import (ISLSequenceDataset, Sample, _mirror, augment, group_of,
                         index_keypoints, stratified_split)
from isl.features import build_frame_features


def kaggle_like(n_words=6, takes=20):
    """Every take ships three times: original, left tilt and right tilt.
    Also returns each clip's true take, worked out here rather than by group_of."""
    samples, take_of = [], {}
    for w in range(n_words):
        for k in range(takes):
            for v in ("", "_left_tilt", "_right_tilt"):
                p = Path(f"word{w}/word{w}_{k:02d}{v}.npz")
                samples.append(Sample(p, f"word{w}", group_of(p)))
                take_of[p] = (w, k)
    return samples, take_of


def test_tilt_variants_of_one_take_share_a_group():
    assert {group_of(Path(f"Tea/Tea_07{v}.npz")) for v in ("", "_left_tilt", "_Right_Tilt")} == {"Tea_07"}


def test_split_keeps_each_take_on_one_side_and_every_word_in_every_split():
    cfg, (samples, take_of) = TrainConfig(), kaggle_like()
    parts = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    assert sum(map(len, parts)) == len(samples)
    train, val, test = ({take_of[s.path] for s in p} for p in parts)
    assert not (train & val or train & test or val & test)   # no near-duplicate leaks across
    words = {s.label for s in samples}
    assert all({s.label for s in p} == words for p in parts)
    # Same seed, same test set: reported numbers can be reproduced.
    again = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    assert [s.path for s in again[2]] == [s.path for s in parts[2]]


def test_mirroring_twice_changes_nothing_and_swaps_the_hands(clip):
    pose, left, right, mask = clip()
    mask[::2, 1] = 0                      # right hand missing in half the frames
    f = build_frame_features(pose, left, right, mask)
    m = _mirror(f)
    np.testing.assert_array_equal(_mirror(m), f)

    g = N_POSE * 3                        # the two global hand blocks start here
    flipped_right = f[:, g + 2 * N_HAND:g + 4 * N_HAND].copy()
    flipped_right[:, 0::2] *= -1          # x negated
    np.testing.assert_array_equal(m[:, g:g + 2 * N_HAND], flipped_right)
    np.testing.assert_array_equal(m[:, -2:], f[:, [-1, -2]])  # hand-present flags swap too


def test_augmentation_keeps_the_feature_layout(clip):
    rng, f = np.random.default_rng(0), build_frame_features(*clip(40))
    for _ in range(25):
        a = augment(f, TrainConfig().augment, rng)
        assert a.shape[1] == F_BASE and len(a) >= 8
        assert a.dtype == np.float32 and np.isfinite(a).all()


def test_cached_landmarks_become_fixed_size_training_tensors(tmp_path, clip):
    # Same keys extract.py writes; clips differ in length, as real recordings do.
    for word, n in (("hello", 3), ("tea", 3), ("rare", 1)):
        (tmp_path / word).mkdir()
        for k in range(n):
            pose, left, right, mask = clip(25 + 10 * k)
            np.savez_compressed(tmp_path / word / f"{word}_{k}.npz", pose=pose, left=left,
                                right=right, mask=mask, fps=30.0, label=word)

    samples = index_keypoints(tmp_path, min_per_class=2)
    assert sorted({s.label for s in samples}) == ["hello", "tea"]   # one-clip words dropped

    classes = ["hello", "tea"]
    for train in (False, True):
        ds = ISLSequenceDataset(samples, classes, train=train, aug_cfg=TrainConfig().augment)
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape == (SEQ_LEN, F_TOTAL) and x.dtype == torch.float32
            assert classes[y] == samples[i].label
