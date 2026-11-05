# Indian Sign Language Recognition

[![ci](https://github.com/pajiala99/isl-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/pajiala99/isl-recognition/actions/workflows/ci.yml)

Word-level ISL recognition from video: MediaPipe skeletal keypoints → signer-invariant
features → BiLSTM with temporal attention → word, text and speech. Runs live on a laptop
webcam; no server, no video leaves the machine.

## Results

Dataset: [Indian Sign Language Video Dataset](https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset)
(Kaggle, MIT) — 61 word classes, 3,630 clips, 1210 original takes used.

| Model | Test top-1 |
|---|---|
| SVM (RBF) on pooled features | 97.8% |
| Random Forest on pooled features | 100.0% |
| BiLSTM + attention (3 seeds) | 98.7% ± 1.4 |

**The standard split is saturated** — every class was recorded by one signer in a single
~5-minute session, so its 20 takes are near-copies and even a Random Forest hits 100%. A
benchmark every model solves cannot rank models, so the real evaluation is below.

### Data efficiency (takes per class used for training)

| Takes/class | Train clips | Top-1 | Top-5 |
|---|---|---|---|
| 1 | 61 | 49.5% | 80.8% |
| 2 | 122 | 71.4% | 95.6% |
| 3 | 183 | 82.4% | 98.4% |
| 5 | 305 | 94.5% | 100.0% |
| 10 | 606 | 99.5% | 100.0% |
| 14 | 846 | 96.7% | 100.0% |

One example per word already gives 49.5% across 61 classes (chance 1.6%).

### Robustness (same model, perturbed inputs)

| Perturbation | Top-1 |
|---|---|
| clean | 96.7% |
| mirror | 96.7% |
| rotate 15 | 96.7% |
| scale 20 | 96.7% |
| shift 10 | 96.7% |
| speed 50 | 96.7% |
| frame drop 30 | 94.5% |
| noise 03 | 96.7% |
| occlude left hand | 31.9% |

Geometry and tempo are fully absorbed by the normalisation; occluding one hand collapses
accuracy, which is correct — many ISL signs are two-handed.

### Augmentation ablation (2 takes/class, mean of 2 seeds)

| Setting | Top-1 |
|---|---|
| none | 83.2% |
| full | 73.9% |
| no mirror | 86.0% |
| geometric only | 86.5% |
| mirror only | 74.5% |

Mirroring *hurts* on a single-signer dataset: mirrored clips are inputs the test distribution
never contains, and many ISL signs are not left-right symmetric. Geometric jitter and noise
help. An augmentation is only valid if it imitates variation that exists in deployment.

## See it working

Three recordings in [`media/`](media/) — all H.264, so they play in the browser.

| Video | What it shows |
|---|---|
| [`model_demo.mp4`](media/model_demo.mp4) | The trained model classifying six held-out clips it never saw, with its top-5 and confidence drawn on each frame |
| [`preprocessing_demo.mp4`](media/preprocessing_demo.mp4) | The preprocessing pipeline running frame by frame: raw video with landmarks, the normalised skeleton, the feature matrix building up, and live numbers |
| [`invariance_demo.mp4`](media/invariance_demo.mp4) | The same clip through two different "cameras" — one zoomed and shifted — producing near-identical normalised skeletons (mean abs. difference 0.011) |

Run them yourself:

```bash
export PYTHONPATH=$PWD/src
python -m isl.demo --random 6 --play --speak     # the model, on held-out clips
python -m isl.visualize --word Tea               # the preprocessing, four panels
python -m isl.visualize --word Tea --invariance  # the normalisation, proved
```

`--save out.mp4` records any of them. `isl.visualize --figure stages.png` writes a single
still, which is where [`artifacts/preprocessing_stages.png`](artifacts/preprocessing_stages.png)
and [`artifacts/invariance_proof.png`](artifacts/invariance_proof.png) came from.

The clips are from the source dataset, which is MIT-licensed and published publicly.

## Pipeline

```
video.mp4 → frame sampling (every 3rd) → MediaPipe pose + 2 hands
         → body-frame normalisation → 64-frame resample → +velocity
         → 64 × 478 tensor → BiLSTM + attention → word
```

Per frame, 239 features: 9 upper-body pose points (x, y, visibility), both hands
body-relative (x, y), both hands wrist-local (x, y, z), 2 presence flags; plus frame-to-frame
deltas → 478. The rationale for each step is in `CROSS_QUESTIONS.md`.

## Layout

```
src/isl/
  config.py          paths, feature layout, hyper-parameters
  keypoints.py       MediaPipe Tasks wrapper (hand + pose landmarkers)
  extract.py         stage 1: videos → cached .npz landmarks (multi-process)
  features.py        normalisation, resampling, deltas, pooled stats
  dataset.py         group-aware stratified split, torch Dataset, augmentation
  models.py          BiLSTM+attention, Transformer encoder
  train.py           stage 2: training loop (MPS/CUDA/CPU)
  baselines.py       SVM / Random Forest on pooled features
  evaluate.py        stage 3: metrics, confusion matrix, curves
  experiments.py     data efficiency, robustness, augmentation protocols
  ablation.py        which augmentations help; seed variance
  error_analysis.py  confusions in the regime where the model errs
  plots.py           figures used in the slides
  speech.py          text-to-speech: Indian-English voice, queued, de-duplicated
  demo.py            predict held-out clips (optionally play them, optionally speak)
  record.py          record your own clips for an unseen-signer test
  unseen_signer.py   score the model on a signer it has never seen
  realtime.py        webcam demo: live text + speech, sentence buffer
  visualize.py       four-panel preprocessing visualiser and the invariance demo
```

## Running it

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src

# 1. data: download + unzip the Kaggle dataset into data/videos/
# 2. landmarks (~7 min for 1,210 clips on 6 workers)
python -m isl.extract --videos data/videos/Video_Dataset/Video_Dataset \
                      --out data/keypoints --stride 3 --workers 6 --exclude "_tilt"
# 3. train + evaluate
python -m isl.train --model bilstm
python -m isl.baselines
python -m isl.evaluate --ckpt models/bilstm.pt
# 4. the protocols that actually measure generalisation
python -m isl.experiments
python -m isl.ablation
python -m isl.error_analysis
python -m isl.plots
# 5. check it yourself on held-out clips (add --play to watch, --speak to hear)
python -m isl.demo --random 15 --play --speak

# 6. live demo: text on screen + spoken output
python -m isl.realtime --ckpt models/bilstm_orig.pt --speak
```

## Tests

```bash
pip install pytest
pytest
```

13 tests run in seconds on synthetic landmarks, with no dataset or camera: the feature
layout and its invariance to where the signer stands and how far away, the group-aware
split (no take's tilt variants on both sides), mirroring and augmentation, both encoders,
and that `models/bilstm_orig.pt` still loads into the current code with the 61 classes in
`artifacts/classes.json`. CI runs them on every push to `main`. Not covered: landmark
extraction from video (needs the Kaggle dataset) and the live demo (needs a webcam).

## Text and speech output

A prediction becomes a spoken word only after three filters: softmax confidence ≥ 0.60,
the same word predicted on 2 consecutive windows, and not identical to the previous word.
Without them the model — which re-predicts every 5 frames — would speak one sign a dozen
times and read every transition between signs aloud.

Confirmed words accumulate into a sentence shown at the bottom of the window:
`space` speaks the whole sentence, `c` clears it, `m` mutes, `q` quits.

`speech.py` tries macOS `say` (preferring an Indian-English voice such as Aman, Rishi or
Veena), then pyttsx3, then a printing fallback, and speaks on a worker thread with a 1.2 s
repeat guard so recognition never blocks on the voice.

Trained on Apple Silicon GPU via PyTorch MPS; training itself takes 2.2 minutes.

## Known limitations

- Isolated words only — no continuous sentence recognition.
- Single-signer dataset, so signer-independent evaluation is impossible here; cross-dataset
  testing on INCLUDE (263 signs, multiple signers) is the next step.
- No facial / non-manual markers yet.
- An assistive aid, not a replacement for a human interpreter.

## Rebuilding the slides

The deck is generated from `artifacts/` — see [tools/README.md](tools/README.md).
To add the unseen-signer slide: `python -m isl.record` then `python -m isl.unseen_signer`.

---

**Hardik Pajiala** · [LinkedIn](https://linkedin.com/in/hardik-pajiala) · [Portfolio](https://pajiala99.github.io) · [GitHub](https://github.com/pajiala99)
