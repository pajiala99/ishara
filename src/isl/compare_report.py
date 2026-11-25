"""Turn the comparison results into one table and one chart, and rank the models.

Reads artifacts/comparison_kaggle61.json (and comparison_combined.json when it exists) and writes
artifacts/model_comparison.md and artifacts/model_comparison_all.png.

The ranking puts generalisation first: macro F1 on the combined test set, then top-1 on signers
the model never saw, then latency. On the single-signer Kaggle split every model is near 100%,
so that column cannot separate them.

    python -m isl.compare_report
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ARTIFACTS

LABELS = {"svm": "SVM", "random_forest": "Random Forest", "lstm": "LSTM", "bilstm": "BiLSTM + attention",
          "gru": "GRU", "bigru": "BiGRU", "tcn": "1-D CNN (TCN)", "transformer": "Transformer",
          "resnet18_bilstm": "ResNet-18 + BiLSTM (ImageNet)", "r3d18": "R3D-18 3-D CNN (Kinetics)"}
KIND = {"svm": "classical", "random_forest": "classical", "resnet18_bilstm": "pretrained, pixels",
        "r3d18": "pretrained, pixels"}


def load(tag):
    p = ARTIFACTS / f"comparison_{tag}.json"
    return {k: v for k, v in json.loads(p.read_text()).items() if not k.startswith("_")} if p.exists() else {}


def pct(x):
    return "" if x is None else f"{x * 100:.1f}"


def num(x, fmt, scale=1.0, unit=""):
    return "" if x is None else f"{x / scale:{fmt}}{unit}"


def main():
    k61, comb = load("kaggle61"), load("combined")
    rows = []
    for n in LABELS:
        if n not in k61 and n not in comb:
            continue
        a, b = k61.get(n, {}), comb.get(n, {})
        t, ns, ct = a.get("test", {}), a.get("new_signer", {}), b.get("test", {})
        rows.append({
            "label": LABELS[n], "kind": KIND.get(n, "keypoints"),
            "k61_top1": t.get("top1"), "new_top1": ns.get("top1"), "new_top5": ns.get("top5"),
            "comb_top1": ct.get("top1"), "comb_top5": ct.get("top5"), "comb_p": ct.get("precision"),
            "comb_r": ct.get("recall"), "comb_f1": ct.get("f1"),
            "comb_include": b.get("test_by_source", {}).get("include", {}).get("top1"),
            "params": (b or a).get("params"), "latency": (b or a).get("latency_ms")})
    rows.sort(key=lambda r: (-(r["comb_f1"] or 0), -(r["new_top1"] or 0), r["latency"] or 1e9))

    lines = ["| Rank | Model | Type | Kaggle-61 top-1 | New signers top-1 / top-5 | Combined top-1 | Combined top-5 "
             "| Precision | Recall | Macro F1 | INCLUDE part top-1 | Params | ms per clip |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(" | ".join([
            f"| {i}", r["label"], r["kind"], pct(r["k61_top1"]), f"{pct(r['new_top1'])} / {pct(r['new_top5'])}",
            pct(r["comb_top1"]), pct(r["comb_top5"]), pct(r["comb_p"]), pct(r["comb_r"]), pct(r["comb_f1"]),
            pct(r["comb_include"]), num(r["params"], ".2f", 1e6, " M"), num(r["latency"], ".1f")]) + " |")
    (ARTIFACTS / "model_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    cols = [("k61_top1", "Kaggle-61 test (one signer)", "#94A3B8"),
            ("new_top1", "New signers (INCLUDE, shared words)", "#F59E0B"),
            ("comb_f1", "Combined test, macro F1", "#0F766E")]
    cols = [c for c in cols if any(r[c[0]] is not None for r in rows)]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.8))
    y = np.arange(len(rows))
    h = 0.8 / len(cols)
    for j, (k, title, colour) in enumerate(cols):
        vals = [(r[k] or 0) * 100 for r in rows]
        ax.barh(y + j * h, vals, h, label=title, color=colour)
        for yi, v in zip(y + j * h, vals):
            if v:
                ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(y + h * (len(cols) - 1) / 2, [r["label"] for r in rows], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 108); ax.set_xlabel("%")
    ax.legend(loc="lower right", fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Ishara: model comparison (ranked best first)", fontsize=11)
    fig.tight_layout(); fig.savefig(ARTIFACTS / "model_comparison_all.png", dpi=160); plt.close(fig)
    print("wrote model_comparison.md and model_comparison_all.png | best so far:", rows[0]["label"])


if __name__ == "__main__":
    main()
