"""Compare models on one split with one set of metrics, so the best can be chosen on evidence.

Keypoint models: SVM and Random Forest on pooled statistics, and six networks on the 64 x 478
sequence (one-way LSTM, BiLSTM, GRU, BiGRU, 1-D temporal CNN, Transformer). Every network gets the
same optimiser, schedule, early stopping and augmentation. Augmentation is geometric only (rotation,
scale, shift, speed, frame drop, noise): the ablation showed that mirroring hurts on this data.

Reported on the held-out test set: top-1, top-5, macro precision / recall / F1, parameters,
training minutes and CPU latency for one clip. With --cross, every model trained here is also
scored on clips from a different dataset whose words it knows: signers it has never seen.

    python -m isl.compare --tag kaggle61
    python -m isl.compare --tag kaggle61 --cross data/include/keypoints
    python -m isl.compare --tag combined --keypoints data/keypoints data/include/keypoints
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from .baselines import featurise
from .config import ARTIFACTS, DATA_KP, ROOT, SEQ_LEN, TrainConfig, F_TOTAL
from .dataset import ISLSequenceDataset, index_keypoints, stratified_split
from .models import build_model
from .train import device_auto

NETS = ["lstm", "bilstm", "gru", "bigru", "tcn", "transformer"]
CLASSICAL = ["svm", "random_forest"]
CKPTS = ROOT / "data" / "compare_models"


def source_of(path: Path) -> str:
    return "include" if "include" in path.parts else "kaggle"


def metrics(y: np.ndarray, scores: np.ndarray, n_classes: int) -> dict:
    top5 = np.argsort(-scores, axis=1)[:, :5]
    pred = top5[:, 0]
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=sorted(set(y)), average="macro", zero_division=0)
    return {"top1": float((pred == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()),
            "precision": float(p), "recall": float(r), "f1": float(f1), "n": int(len(y))}


def save_result(path: Path, key: str, value) -> None:
    """Merge one entry into the results file; isl.pixel writes to the same file, so re-read first."""
    results = json.loads(path.read_text()) if path.exists() else {}
    results[key] = value
    path.write_text(json.dumps(results, indent=2))


def train_net(name, train_s, val_s, classes, cfg, device, load_kw, p_mirror=0.0):
    torch.manual_seed(cfg.seed)
    aug = {**cfg.augment, "p_mirror": p_mirror}
    dl_tr = DataLoader(ISLSequenceDataset(train_s, classes, cfg.seq_len, True, aug, cfg.seed, **load_kw),
                       cfg.batch_size, shuffle=True, drop_last=len(train_s) > cfg.batch_size)
    dl_va = DataLoader(ISLSequenceDataset(val_s, classes, cfg.seq_len, **load_kw), cfg.batch_size)
    model = build_model(name, len(classes), cfg).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(dl_tr),
                                                pct_start=0.25)
    best, best_state, bad, t0, epochs = -1.0, None, 0, time.time(), 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(x.to(device)).argmax(1).cpu() == y).sum().item() for x, y in dl_va)
        acc, epochs = correct / len(val_s), ep
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, (time.time() - t0) / 60, epochs, best


@torch.no_grad()
def net_scores(model, samples, classes, cfg, device, load_kw) -> np.ndarray:
    model.eval()
    dl = DataLoader(ISLSequenceDataset(samples, classes, cfg.seq_len, **load_kw), cfg.batch_size)
    return np.concatenate([model(x.to(device)).softmax(-1).cpu().numpy() for x, _ in dl])


@torch.no_grad()
def net_latency_ms(model) -> float:
    m = model.to("cpu").eval()
    x = torch.randn(1, SEQ_LEN, F_TOTAL)
    for _ in range(5):
        m(x)
    t = time.perf_counter()
    for _ in range(30):
        m(x)
    return (time.perf_counter() - t) / 30 * 1000


def classical(name):
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", kernel="rbf"))
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)


def classical_scores(clf, X) -> np.ndarray:
    return clf.decision_function(X) if hasattr(clf, "decision_function") else clf.predict_proba(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--keypoints", type=Path, nargs="+", default=[DATA_KP])
    ap.add_argument("--models", nargs="+", default=CLASSICAL + NETS)
    ap.add_argument("--cross", type=Path, default=None, help="keypoints of another dataset: new-signer test")
    ap.add_argument("--train-extra", type=Path, nargs="+", default=[],
                    help="more copies of the same takes (the tilted clips), for training only")
    ap.add_argument("--aspect", action="store_true", help="x in frame-height units (features.py)")
    ap.add_argument("--trim", action="store_true", help="cut the resting frames before and after the sign")
    ap.add_argument("--mirror", type=float, default=0.0, help="probability of a mirrored training clip")
    a = ap.parse_args()
    load_kw = {"aspect_correct": a.aspect, "trim": a.trim}
    # absolute paths, so the saved split and isl.pixel can relate every clip to the repo root
    a.keypoints = [p.resolve() for p in a.keypoints]
    a.train_extra = [p.resolve() for p in a.train_extra]
    a.cross = a.cross.resolve() if a.cross else None

    cfg, device = TrainConfig(), device_auto()
    samples = index_keypoints(a.keypoints)
    classes = sorted({s.label for s in samples})
    idx = {c: i for i, c in enumerate(classes)}
    train_s, val_s, test_s = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    if a.train_extra:
        # a copy joins training only if its original take is a training take, so no near-duplicate
        # of a validation or test clip is ever trained on
        take = lambda s: (s.label, re.sub(r"(_(left|right)_tilt|_Trim_cropped)$", "", s.path.stem, flags=re.I))
        train_takes = {take(s) for s in train_s}
        extra = [s for s in index_keypoints(a.train_extra, min_per_class=1) if take(s) in train_takes]
        print(f"+ {len(extra)} extra training copies from {[str(p) for p in a.train_extra]}", flush=True)
        train_s = train_s + extra
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"split_{a.tag}.json").write_text(json.dumps(
        {k: [str(s.path.relative_to(ROOT)) for s in v] for k, v in
         (("train", train_s), ("val", val_s), ("test", test_s))}, indent=0))
    y_test = np.array([idx[s.label] for s in test_s])
    src_test = np.array([source_of(s.path) for s in test_s])

    cross_s = []
    if a.cross:
        cross_s = [s for s in index_keypoints([a.cross], min_per_class=1) if s.label in idx]
    # half the new-signer clips (alternating within each word) choose settings, the other half reports
    cross_s.sort(key=lambda s: (s.label, str(s.path)))
    half = np.array([i % 2 for i in range(len(cross_s))])
    y_cross = np.array([idx[s.label] for s in cross_s])
    print(f"{a.tag}: {len(samples)} clips, {len(classes)} words | train {len(train_s)} val {len(val_s)} "
          f"test {len(test_s)}" + (f" | new-signer clips {len(cross_s)} over "
                                   f"{len({s.label for s in cross_s})} words" if cross_s else ""), flush=True)

    out_path = ARTIFACTS / f"comparison_{a.tag}.json"
    save_result(out_path, "_meta", {"clips": len(samples), "classes": len(classes), "train": len(train_s),
                                    "val": len(val_s), "test": len(test_s), "cross_clips": len(cross_s),
                                    "cross_words": sorted({s.label for s in cross_s}),
                                    "settings": {"aspect": a.aspect, "trim": a.trim, "mirror": a.mirror,
                                                 "train_extra": [str(p) for p in a.train_extra]}})
    Xs = {}
    for name in a.models:
        t0 = time.time()
        if name in CLASSICAL:
            if not Xs:
                Xs = {k: featurise(v, cfg.seq_len, **load_kw)[0] for k, v in
                      (("train", train_s), ("test", test_s), ("cross", cross_s)) if v}
                y_train = np.array([idx[s.label] for s in train_s])
            clf = classical(name).fit(Xs["train"], y_train)
            minutes = (time.time() - t0) / 60
            scores = classical_scores(clf, Xs["test"])
            t = time.perf_counter()
            for row in Xs["test"][:30]:
                clf.predict(row[None])
            latency = (time.perf_counter() - t) / min(30, len(Xs["test"])) * 1000
            info = {"params": None, "epochs": None}
            cross_scores = classical_scores(clf, Xs["cross"]) if cross_s else None
        else:
            model, minutes, epochs, best_val = train_net(name, train_s, val_s, classes, cfg, device, load_kw, a.mirror)
            scores = net_scores(model, test_s, classes, cfg, device, load_kw)
            cross_scores = net_scores(model, cross_s, classes, cfg, device, load_kw) if cross_s else None
            info = {"params": int(sum(p.numel() for p in model.parameters())), "epochs": epochs,
                    "best_val": best_val}
            CKPTS.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": classes, "arch": name, "cfg": cfg.__dict__},
                       CKPTS / f"{a.tag}_{name}.pt")
            latency = net_latency_ms(model)
        r = {"test": metrics(y_test, scores, len(classes)), "train_minutes": minutes,
             "latency_ms": latency, **info}
        if len(set(src_test)) > 1:
            r["test_by_source"] = {src: metrics(y_test[src_test == src], scores[src_test == src], len(classes))
                                   for src in sorted(set(src_test))}
        if cross_scores is not None:
            r["new_signer"] = metrics(y_cross, cross_scores, len(classes))
            for h, part in ((0, "new_signer_dev"), (1, "new_signer_test")):
                r[part] = metrics(y_cross[half == h], cross_scores[half == h], len(classes))
        save_result(out_path, name, r)
        m = r["test"]
        line =(f"{name:14} top1 {m['top1']*100:5.1f}  top5 {m['top5']*100:5.1f}  F1 {m['f1']*100:5.1f}  "
                f"{minutes:4.1f} min  {latency:6.2f} ms")
        if "new_signer" in r:
            line += f" | new signers top1 {r['new_signer']['top1']*100:5.1f} top5 {r['new_signer']['top5']*100:5.1f}"
        print(line, flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
