"""Classical baselines (SVM / Random Forest) on pooled keypoint statistics.

    python -m isl.baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ARTIFACTS, DATA_KP, TrainConfig
from .dataset import index_keypoints, load_sequence, stratified_split
from .features import pooled_stats, resample, with_deltas


def featurise(samples, seq_len, **load_kw):
    X = np.stack([pooled_stats(with_deltas(resample(load_sequence(s.path, **load_kw), seq_len)))
                  for s in samples])
    y = np.array([s.label for s in samples])
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keypoints", type=Path, nargs="+", default=[DATA_KP])
    a = ap.parse_args()

    cfg = TrainConfig()
    samples = index_keypoints(a.keypoints)
    if not samples:
        raise SystemExit("no keypoints - run `python -m isl.extract` first")
    train_s, val_s, test_s = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    train_s = train_s + val_s                      # classical models need no val set

    Xtr, ytr = featurise(train_s, cfg.seq_len)
    Xte, yte = featurise(test_s, cfg.seq_len)
    print(f"baseline features: train {Xtr.shape} test {Xte.shape}")

    models = {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", kernel="rbf")),
        "random_forest": RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                                random_state=cfg.seed, class_weight="balanced"),
    }
    out = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        out[name] = {"test_top1": float(accuracy_score(yte, pred)),
                     "test_macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0))}
        print(name, out[name])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
