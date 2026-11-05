"""Feature engineering: the layout in config.py and the normalisation claims in features.py."""
from __future__ import annotations

import numpy as np

from isl.config import F_BASE, F_TOTAL, N_HAND, N_POSE, SEQ_LEN
from isl.features import build_frame_features, pooled_stats, resample, with_deltas

GLOBAL_RIGHT = slice(N_POSE * 3 + N_HAND * 2, N_POSE * 3 + N_HAND * 4)
LOCAL_RIGHT = slice(N_POSE * 3 + N_HAND * 7, N_POSE * 3 + N_HAND * 10)


def test_feature_layout_matches_config(clip):
    f = build_frame_features(*clip())
    assert f.shape == (30, F_BASE) and F_BASE == 239 and f.dtype == np.float32
    assert with_deltas(f).shape == (30, F_TOTAL) and F_TOTAL == 478
    assert pooled_stats(f).shape == (4 * F_BASE,)


def test_features_ignore_where_the_signer_stands_and_how_far_away(clip):
    pose, left, right, mask = clip()
    s, centre, shift = 1.7, np.array([0.3, 0.6]), np.array([0.12, -0.05])

    def moved(a, depth_too=False):
        # Same signer, standing elsewhere and closer to the camera: x, y scale about a
        # point and shift. Hand depth (z) is relative to the wrist, so it scales as well.
        out = a.copy()
        out[..., :2] = (a[..., :2] - centre) * s + centre + shift
        if depth_too:
            out[..., 2] *= s
        return out

    np.testing.assert_allclose(
        build_frame_features(moved(pose), moved(left, True), moved(right, True), mask),
        build_frame_features(pose, left, right, mask),
        atol=1e-5,
    )


def test_a_hand_flagged_missing_contributes_nothing(clip):
    pose, left, right, mask = clip()
    right[:5] = np.nan                    # whatever sits in an empty slot must not leak through
    mask[:, 1] = 0
    f = build_frame_features(pose, left, right, mask)
    assert np.isfinite(f).all()
    assert not f[:, GLOBAL_RIGHT].any() and not f[:, LOCAL_RIGHT].any()
    assert (f[:, -2] == 1).all() and (f[:, -1] == 0).all()   # hand-present flags


def test_resample_keeps_the_ends_and_straight_line_motion():
    t = np.linspace(0.0, 1.0, 37)[:, None]
    seq = np.hstack([t, 2 * t + 1]).astype(np.float32)
    out = resample(seq, SEQ_LEN)
    assert out.shape == (SEQ_LEN, 2)
    np.testing.assert_allclose(out[[0, -1]], seq[[0, -1]])
    np.testing.assert_allclose(out[:, 0], np.linspace(0.0, 1.0, SEQ_LEN), atol=1e-6)


def test_deltas_are_frame_to_frame_motion():
    seq = np.cumsum(np.random.default_rng(1).normal(size=(10, 3)), axis=0)
    d = with_deltas(seq)
    np.testing.assert_array_equal(d[:, :3], seq)
    assert not d[0, 3:].any()             # no motion before the first frame
    np.testing.assert_allclose(d[1:, 3:], np.diff(seq, axis=0))
