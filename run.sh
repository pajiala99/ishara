#!/usr/bin/env bash
# The final pipeline, end to end. Run from the project root.
# Data first: the Kaggle "Indian Sign Language Video Dataset" unzipped into data/videos/.
set -euo pipefail
export PYTHONPATH="$PWD/src"
KAGGLE=data/videos/Video_Dataset/Video_Dataset

# 1. keypoints: Kaggle originals, Kaggle tilted copies (training only), and INCLUDE
python -m isl.extract --videos "$KAGGLE" --out data/keypoints --stride 3 --exclude "_tilt"
python -m isl.extract --videos "$KAGGLE" --out data/keypoints_tilt --stride 3 --exclude "(?i)^(?!.*tilt)"
python tools/download_include.py                     # AI4Bharat INCLUDE, 57.8 GB, resumable
python -m isl.extract --videos data/include/videos --out data/include/keypoints --stride 2

# 2. compare the ten models on the combined 315-word set (aspect fix on) and rank them
python -m isl.compare --tag combined --keypoints data/keypoints data/include/keypoints \
    --train-extra data/keypoints_tilt --aspect
python -m isl.pixel --tag combined                   # pretrained ResNet-18 and R3D-18
python -m isl.compare --tag ns_aspect --aspect --cross data/include/keypoints   # new-signer test
python -m isl.compare_report                         # artifacts/model_comparison.md
python -m isl.dashboard                              # dashboard/index.html

# 3. the deployed Transformer: export to ONNX (with its check), then the live webcam app
python -m isl.export_onnx
python -m isl.realtime --speak
