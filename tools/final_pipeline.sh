#!/bin/bash
# Finish the INCLUDE stage end to end: download (resumable), keypoints, combined comparison, dashboard.
# Detached on purpose (nohup), so it survives the terminal or app that started it:
#   nohup caffeinate -i bash tools/final_pipeline.sh > data/final_pipeline.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
echo "== download $(date)"
python3 tools/download_include.py --workers 2 > data/include/download.log 2>&1
echo "== retry failures $(date)"
python3 tools/download_include.py --workers 2 >> data/include/download.log 2>&1
tail -1 data/include/download.log
echo "== keypoints $(date)"
.venv/bin/python -m isl.extract --videos data/include/videos --out data/include/keypoints --stride 2 --workers 3 2>&1 | tail -1
.venv/bin/python tools/add_frame_size.py data/include/keypoints | tail -1
echo "== combined comparison, keypoint models, aspect fix $(date)"
.venv/bin/python -m isl.compare --tag combined --keypoints data/keypoints data/include/keypoints \
    --train-extra data/keypoints_tilt --aspect 2>&1 | grep -vE "Warning|self.enc"
echo "== combined comparison, pretrained pixel models $(date)"
.venv/bin/python -m isl.pixel --tag combined 2>&1 | grep -vE "Warning|%\||features [0-9]" | tail -4
.venv/bin/python -m isl.compare_report | tail -1
.venv/bin/python -m isl.dashboard | tail -1
echo "== done $(date)"
