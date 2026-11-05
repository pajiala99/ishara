"""
================================================================================
 DATA PREPROCESSING, EXPLAINED LINE BY LINE
 Indian Sign Language recognition  ·  video  ->  64 x 478 tensor
================================================================================

This one file contains the whole preprocessing path, in the order it runs, with a
comment on every step explaining WHY it is there - not just what it does.

In the real project this logic is split across four modules, because training,
evaluation and the live demo all import it:

    src/isl/keypoints.py   stage 1-2   video frames -> MediaPipe landmarks
    src/isl/features.py    stage 3-5   normalise, resample, add velocity
    src/isl/dataset.py     stage 6     augmentation and the train/test split
    src/isl/extract.py                 runs stage 1-2 over the dataset, caches .npz

This file is the same code, flattened and annotated so it can be read top to
bottom. At the end, main() runs it on a real clip AND checks the output against
the production implementation, so you can prove the two agree.

    RUN IT:   PYTHONPATH=$PWD/src python3 preprocessing_explained.py
              PYTHONPATH=$PWD/src python3 preprocessing_explained.py --word Hello
================================================================================
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# MediaPipe's native layers (glog, TFLite) print startup noise straight to stderr
# the moment they load. These must be set BEFORE `import mediapipe` to take effect.
for _k, _v in (("GLOG_minloglevel", "2"), ("GLOG_stderrthreshold", "3"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"), ("ABSL_MIN_LOG_LEVEL", "3")):
    os.environ.setdefault(_k, _v)

import cv2                 # video decoding only - we never feed pixels to the model
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "data" / "videos" / "Video_Dataset" / "Video_Dataset"


# =============================================================================
# SECTION 0 - THE CONSTANTS THAT DEFINE THE FEATURE VECTOR
# =============================================================================

# MediaPipe Pose returns 33 body landmarks. We keep 9. The legs never take part
# in a sign, and every extra dimension is extra noise for the model to explain.
#   0 = nose          11, 12 = shoulders     13, 14 = elbows
#   15, 16 = wrists   23, 24 = hips
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

L_SHOULDER, R_SHOULDER = 11, 12   # indices into the FULL 33-point pose array
N_POSE = len(POSE_IDX)            # 9 kept pose points
N_HAND = 21                       # MediaPipe gives 21 landmarks per hand

# The 239 numbers that describe one frame, block by block:
F_POSE        = N_POSE * 3        # 27  = 9 points x (x, y, visibility)
F_HAND_GLOBAL = 2 * N_HAND * 2    # 84  = 2 hands x 21 points x (x, y)
F_HAND_LOCAL  = 2 * N_HAND * 3    # 126 = 2 hands x 21 points x (x, y, z)
F_FLAGS       = 2                 # 2   = was each hand detected at all?
F_BASE        = F_POSE + F_HAND_GLOBAL + F_HAND_LOCAL + F_FLAGS   # = 239
F_TOTAL       = F_BASE * 2        # = 478 once velocity is appended

SEQ_LEN = 64      # every clip is resampled onto this many frames
STRIDE  = 3       # keep every 3rd frame of the source video


# =============================================================================
# SECTION 1 - DECODE THE VIDEO AND SAMPLE FRAMES
# =============================================================================

def load_frames(video_path: Path, stride: int = STRIDE):
    """Read the clip and keep every `stride`-th frame.

    WHY SUBSAMPLE: the source is 30 fps, so consecutive frames are nearly
    identical - the hand has moved a few pixels. Two thirds of the landmark
    computation would buy almost no information. Keeping every 3rd frame leaves
    an effective 10 fps, still well above the rate at which sign movement starts
    to alias, and cut extraction over 1,210 clips from ~90 minutes to under 10.
    """
    cap = cv2.VideoCapture(str(video_path))          # open the file
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")

    frames, idx = [], 0
    while True:
        ok, frame = cap.read()                       # ok=False at end of file
        if not ok:
            break
        if idx % stride == 0:                        # the subsampling decision
            # OpenCV decodes as BGR; MediaPipe expects RGB. Getting this wrong
            # does not crash - it silently degrades detection, which is worse.
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


# =============================================================================
# SECTION 2 - PIXELS -> LANDMARKS  (this is where the background disappears)
# =============================================================================

def extract_landmarks(frames, hand_task: Path, pose_task: Path):
    """Run MediaPipe over the frames and return raw landmark arrays.

    WHY LANDMARKS AT ALL: this is the core decision of the project. A clip is
    ~280 million pixel values; the label is one word. Feeding pixels to a network
    means it spends most of its capacity learning to ignore clothing, lighting,
    skin tone and background. Landmarks throw all of that away by construction.

    Returns
        pose  (T, 33, 4)  x, y, z, visibility   - normalised image coordinates
        left  (T, 21, 3)  x, y, z               - signer's left hand
        right (T, 21, 3)                        - signer's right hand
        mask  (T, 2)      1.0 where that hand was detected on that frame
    """
    # Loading the two models logs several lines straight to the OS-level stderr,
    # below anything the env vars filter. Redirect fd 2 for the duration of the
    # construction only - no Python errors are hidden by this.
    devnull, saved_fd = os.open(os.devnull, os.O_WRONLY), os.dup(2)
    os.dup2(devnull, 2)

    # VIDEO running mode lets MediaPipe track landmarks across frames instead of
    # re-detecting from scratch, which is both faster and more stable.
    hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_task)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        # 0.4 is deliberately low: high enough to reject background noise, low
        # enough to keep tracking through motion blur during a fast sign.
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    ))
    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_task)),
        running_mode=RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    ))

    os.dup2(saved_fd, 2); os.close(devnull); os.close(saved_fd)   # stderr restored

    poses, lefts, rights, masks = [], [], [], []

    for i, rgb in enumerate(frames):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires STRICTLY INCREASING timestamps. Reusing one
        # extractor across clips (whose timestamps restart at 0) throws
        # "Input timestamp must be monotonically increasing" - a real bug we hit.
        ts_ms = i * 100
        if i == 0:
            # The very first inference warms up the graph and logs once more,
            # below the level the env vars filter. Silence that single call by
            # redirecting the OS-level stderr - nothing else is suppressed.
            devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(2)
            os.dup2(devnull, 2)
            hand_result = hands.detect_for_video(image, ts_ms)
            pose_result = pose_lm.detect_for_video(image, ts_ms)
            os.dup2(saved, 2); os.close(devnull); os.close(saved)
        else:
            hand_result = hands.detect_for_video(image, ts_ms)
            pose_result = pose_lm.detect_for_video(image, ts_ms)

        # ---- pose: 33 points, zeros if nothing was detected on this frame ----
        pose = np.zeros((33, 4), np.float32)
        if pose_result.pose_landmarks:
            for j, lm in enumerate(pose_result.pose_landmarks[0]):
                # visibility is MediaPipe's own confidence for that joint; we keep
                # it as a feature so the model knows when a point is unreliable.
                pose[j] = (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0) or 0.0)

        # ---- hands: slot 0 = signer's right, slot 1 = signer's left ----------
        hands_arr = np.zeros((2, N_HAND, 3), np.float32)
        mask = np.zeros(2, np.float32)
        for lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            # MediaPipe labels handedness FROM THE CAMERA'S POINT OF VIEW, which
            # is mirrored relative to the signer. So its "Left" is the signer's
            # right hand. Without this flip the two hand channels swap randomly
            # between clips and a two-handed sign becomes incoherent.
            slot = 1 if handedness[0].category_name == "Left" else 0
            if mask[slot]:            # already filled: ignore the duplicate
                continue
            hands_arr[slot] = [(lm.x, lm.y, lm.z) for lm in lms]
            mask[slot] = 1.0          # flag: this hand WAS seen on this frame

        poses.append(pose); lefts.append(hands_arr[0])
        rights.append(hands_arr[1]); masks.append(mask)

    hands.close(); pose_lm.close()
    return np.stack(poses), np.stack(lefts), np.stack(rights), np.stack(masks)


# =============================================================================
# SECTION 3 - THE BODY FRAME  (the single most important step)
# =============================================================================

def body_frame(pose: np.ndarray):
    """Per-frame origin and scale, derived from the shoulders.

    WHY: raw landmarks are in image coordinates, so they still encode where the
    signer stood and how far they were from the camera. Expressing everything
    relative to the shoulders removes both in one transform:

        origin = midpoint of the two shoulders   -> translation invariance
        scale  = distance between them           -> scale invariance

    Computed PER FRAME rather than once per clip, so it also absorbs the signer
    leaning or stepping sideways in the middle of a sign.
    """
    ls = pose[:, L_SHOULDER, :2]                 # (T, 2) left shoulder  (x, y)
    rs = pose[:, R_SHOULDER, :2]                 # (T, 2) right shoulder
    origin = (ls + rs) / 2.0                     # (T, 2) midpoint
    scale = np.linalg.norm(ls - rs, axis=-1, keepdims=True)   # (T, 1) shoulder width

    # GUARD: if the signer turns sideways or the detector fails, shoulder width
    # collapses toward zero and dividing by it would explode every coordinate.
    scale = np.where(scale < 1e-3, 1.0, scale)
    return origin, scale


# =============================================================================
# SECTION 4 - ASSEMBLE THE 239 NUMBERS FOR EVERY FRAME
# =============================================================================

def build_frame_features(pose, left, right, mask) -> np.ndarray:
    """(T, 33, 4) + two (T, 21, 3) hands + (T, 2) mask  ->  (T, 239)."""
    origin, scale = body_frame(pose)             # (T, 2) and (T, 1)

    # ---- BLOCK A: pose, 27 numbers -----------------------------------------
    sel = pose[:, POSE_IDX, :]                               # (T, 9, 4)
    # Broadcasting: origin becomes (T, 1, 2) so it subtracts from all 9 points.
    pose_xy = (sel[..., :2] - origin[:, None, :]) / scale[:, None, :]
    # Keep visibility (index 3) un-normalised: it is a confidence, not a position.
    pose_feat = np.concatenate([pose_xy, sel[..., 3:4]], -1).reshape(len(pose), -1)

    blocks = [pose_feat]
    local_blocks = []

    for hand, m in ((left, mask[:, 0]), (right, mask[:, 1])):
        # ---- BLOCK B: hand LOCATION, 42 numbers per hand -------------------
        # Same body-frame transform as the pose: this says WHERE the hand is -
        # chest, chin, forehead. In sign-language linguistics that is "location".
        g = (hand[..., :2] - origin[:, None, :]) / scale[:, None, :]
        g *= m[:, None, None]      # undetected hand -> all zeros, not garbage
        blocks.append(g.reshape(len(hand), -1))

        # ---- BLOCK C: hand SHAPE, 63 numbers per hand ----------------------
        # Re-centre on the wrist so position cancels out, then divide by the
        # hand's own size. What survives is only which fingers are extended -
        # "handshape", the other linguistic parameter. Giving the model location
        # and handshape separately means it never has to disentangle them.
        wrist = hand[:, 0:1, :]                      # landmark 0 is the wrist
        loc = hand - wrist                           # (T, 21, 3) wrist-centred
        # Landmark 9 is the middle-finger knuckle: wrist->knuckle is a stable
        # measure of hand size that does not change as the fingers move.
        hs = np.linalg.norm(loc[:, 9, :2], axis=-1, keepdims=True)
        hs = np.where(hs < 1e-3, 1.0, hs)            # same divide-by-zero guard
        loc = loc / hs[:, None]
        loc *= m[:, None, None]
        local_blocks.append(loc.reshape(len(hand), -1))

    # ---- BLOCK D: the two presence flags ------------------------------------
    # Without these, "hand absent" and "hand exactly at the origin" are both
    # zeros and the model cannot tell them apart.
    feats = np.concatenate(blocks + local_blocks + [mask], axis=-1).astype(np.float32)
    assert feats.shape[1] == F_BASE, (feats.shape, F_BASE)

    # MediaPipe occasionally emits NaN or inf on a failed frame. A single NaN
    # propagates through backpropagation and destroys an entire training run.
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# SECTION 5 - TIME: FIXED LENGTH, THEN VELOCITY
# =============================================================================

def resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Linearly interpolate a variable-length clip onto `length` frames.

    WHY: clips differ in duration and people sign at different speeds, but a sign
    performed slowly and the same sign performed quickly are the SAME sign, time-
    warped. Resampling removes that difference and gives fixed-size tensors, which
    the classical SVM / Random-Forest baselines also need.

    The alternative - zero-pad to the longest clip and mask - preserves absolute
    duration, which we judged not to be meaningful for isolated signs.
    """
    t = len(seq)
    if t == length:
        return seq.astype(np.float32)
    src = np.linspace(0.0, 1.0, t)          # original timeline, normalised to 0..1
    dst = np.linspace(0.0, 1.0, length)     # the fixed grid we want
    out = np.empty((length, seq.shape[1]), np.float32)
    for c in range(seq.shape[1]):           # interpolate each feature separately
        out[:, c] = np.interp(dst, src, seq[:, c])
    return out


def with_deltas(seq: np.ndarray) -> np.ndarray:
    """Append frame-to-frame change: (T, 239) -> (T, 478).

    WHY: movement is as much a part of a sign as posture - two signs can share a
    handshape and location and differ only in how the hand moves. The LSTM could
    learn to difference its own inputs, but it would spend capacity doing it.
    Differencing is free and removes any residual static offset. Same reasoning
    as delta-MFCCs in speech recognition.

    `prepend=seq[:1]` makes the first frame's velocity zero instead of dropping
    a frame, so the output keeps the same length as the input.
    """
    d = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, d], axis=-1)


# =============================================================================
# SECTION 6 - AUGMENTATION  (training clips only - never val or test)
# =============================================================================

def augment_geometric(seq: np.ndarray, rng, rot_deg=10.0, scale_j=0.1, shift_j=0.05):
    """Apply one small random 2-D similarity transform to every (x, y) pair.

    Each parameter imitates a real nuisance: rotation = camera not level, scale =
    distance from the camera, translation = signer off-centre. That is the test of
    a legitimate augmentation - it must imitate variation that exists in reality.

    NOTE from our ablation: these three HELP (86.5% vs 83.2% with none). Mirroring,
    which we also tried, HURT by ~10 points, because the dataset has one signer
    with a consistent dominant hand, so mirrored clips are inputs the test
    distribution never contains.
    """
    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg))      # rotation angle
    s = 1.0 + rng.uniform(-scale_j, scale_j)             # scale factor
    dx, dy = rng.uniform(-shift_j, shift_j, 2)           # translation
    c, sn = np.cos(th) * s, np.sin(th) * s               # rotation matrix entries
    out = seq.copy()

    # The pose block is stored as (x, y, visibility) triples, so x and y sit at
    # offsets 0 and 1 of every group of 3; visibility must NOT be transformed.
    for i in range(N_POSE):
        x, y = out[:, i * 3], out[:, i * 3 + 1]
        out[:, i * 3]     = c * x - sn * y + dx
        out[:, i * 3 + 1] = sn * x + c * y + dy

    # The global-hand block is (x, y) pairs, so x and y are at offsets 0 and 1
    # of every group of 2.
    off = N_POSE * 3
    for i in range(2 * N_HAND):
        x, y = out[:, off + 2 * i], out[:, off + 2 * i + 1]
        out[:, off + 2 * i]     = c * x - sn * y + dx
        out[:, off + 2 * i + 1] = sn * x + c * y + dy
    return out


# =============================================================================
# SECTION 7 - THE BASELINE REPRESENTATION
# =============================================================================

def pooled_stats(seq: np.ndarray) -> np.ndarray:
    """Collapse time into four summary statistics, for the SVM / Random Forest.

    WHY IT IS A WEAKER REPRESENTATION, deliberately: mean/std/min/max are
    order-invariant, so a sign and the same sign played backwards give identical
    features. The gap between these baselines and the BiLSTM is exactly what
    temporal modelling buys.
    """
    return np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0)])


# =============================================================================
# MAIN - run the whole chain on one real clip and verify it against production
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Run the preprocessing chain on one clip")
    ap.add_argument("--word", default="Tea", help="which word class to take a clip from")
    ap.add_argument("--video", type=Path, default=None, help="or a specific video file")
    args = ap.parse_args()

    clip = args.video
    if clip is None:
        folder = VIDEOS / args.word
        if not folder.is_dir():
            raise SystemExit(f"no folder {folder} - check --word")
        clip = sorted(p for p in folder.glob("*.mp4") if "tilt" not in p.name)[0]

    print(f"clip: {clip.name}  (word: {clip.parent.name})\n")

    # ---- stage 1 ----------------------------------------------------------
    frames = load_frames(clip)
    total = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"1. decode + sample     {total} frames in file -> {len(frames)} kept "
          f"(every {STRIDE}rd)")

    # ---- stage 2 ----------------------------------------------------------
    pose, left, right, mask = extract_landmarks(
        frames, ROOT / "models" / "hand_landmarker.task",
        ROOT / "models" / "pose_landmarker_lite.task")
    print(f"2. landmarks           pose {pose.shape}, hands {left.shape} x2, "
          f"mask {mask.shape}")
    print(f"                       both hands seen on "
          f"{int((mask.sum(1) == 2).sum())}/{len(mask)} frames")

    # ---- stage 3 + 4 ------------------------------------------------------
    origin, scale = body_frame(pose)
    print(f"3. body frame          shoulder width: mean {scale.mean():.4f}, "
          f"min {scale.min():.4f}, max {scale.max():.4f}")
    feats = build_frame_features(pose, left, right, mask)
    print(f"4. features per frame  {feats.shape}  "
          f"(27 pose + 84 location + 126 handshape + 2 flags)")

    # ---- stage 5 ----------------------------------------------------------
    fixed = resample(feats, SEQ_LEN)
    print(f"5. resample            {feats.shape} -> {fixed.shape}")
    final = with_deltas(fixed)
    print(f"6. + velocity          {fixed.shape} -> {final.shape}   <- model input")
    print(f"7. baseline vector     {pooled_stats(final).shape} "
          f"(mean/std/min/max over time)\n")

    # Raw video is uint8, so one pixel channel = one byte; the final tensor is
    # float32. Comparing BYTES is the honest comparison, and it is the ~2,000x
    # figure quoted in the presentation.
    raw_bytes = total * 640 * 720 * 3
    print(f"compression: {raw_bytes/1e6:.0f} MB of pixels -> "
          f"{final.nbytes/1024:.0f} KB of features  "
          f"({raw_bytes/final.nbytes:.0f}x smaller)")

    # ---- proof that this teaching copy matches the real pipeline ----------
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from isl.features import build_frame_features as prod_features
        from isl.features import resample as prod_resample, with_deltas as prod_deltas
        prod = prod_deltas(prod_resample(prod_features(pose, left, right, mask), SEQ_LEN))
        identical = np.allclose(final, prod, atol=1e-6)
        print(f"\nmatches src/isl/features.py exactly: {identical}")
    except Exception as e:
        print(f"\n(could not compare against the production module: {e})")


if __name__ == "__main__":
    main()
