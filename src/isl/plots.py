"""Render the figures used in the report and slides from artifacts/*.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ARTIFACTS

ACCENT, AMBER, GREY = "#0F766E", "#F59E0B", "#94A3B8"


def data_efficiency(exp, path: Path):
    c = exp["data_efficiency"]
    k = [r["takes_per_class"] for r in c]
    t1 = [r["test_top1"] * 100 for r in c]
    t5 = [r["test_top5"] * 100 for r in c]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(k, t5, "--o", color=AMBER, lw=2, ms=5, label="top-5")
    ax.plot(k, t1, "-o", color=ACCENT, lw=2.4, ms=6, label="top-1")
    for x, y in zip(k, t1):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=ACCENT, fontweight="bold")
    ax.set_xlabel("recording takes per class used for training")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"Data efficiency — {exp['n_classes']} ISL word classes")
    ax.set_ylim(0, 104); ax.set_xticks(k)
    ax.grid(alpha=.25); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def robustness(exp, path: Path):
    labels = {"clean": "clean", "mirror": "mirrored (left/right swapped)", "rotate_15": "camera rotated 15°",
              "scale_20": "scale ±20%", "shift_10": "off-centre ±10%", "speed_50": "signing speed ×0.5 / ×1.5",
              "frame_drop_30": "30% frames dropped", "noise_03": "landmark noise σ=0.03",
              "occlude_left_hand": "left hand occluded"}
    r = exp["robustness"]
    keys = [k for k in labels if k in r]
    vals = [r[k] * 100 for k in keys]
    colors = [ACCENT if k == "clean" else (AMBER if v < 90 else GREY) for k, v in zip(keys, vals)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = range(len(keys))
    ax.barh(list(y), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold", color="#1F2937")
    ax.set_yticks(list(y), [labels[k] for k in keys], fontsize=10)
    ax.invert_yaxis(); ax.set_xlim(0, 112); ax.set_xlabel("test accuracy (%)")
    ax.set_title("Robustness of the trained model under input perturbation")
    ax.grid(axis="x", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def saturation(exp, baselines, deep_top1, path: Path):
    names = ["SVM (RBF)\npooled features", "Random Forest\npooled features", "BiLSTM + attention"]
    full = [baselines["svm_rbf"]["test_top1"] * 100, baselines["random_forest"]["test_top1"] * 100,
            deep_top1 * 100]
    k2 = exp["augmentation_ablation"]["with_augmentation"]["test_top1"] * 100
    one = exp["data_efficiency"][0]["test_top1"] * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    a1.bar(names, full, color=[GREY, GREY, ACCENT])
    for i, v in enumerate(full):
        a1.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    a1.set_ylim(0, 112); a1.set_ylabel("test top-1 (%)")
    a1.set_title("Standard split: every model saturates", fontsize=11)
    a2.bar(["1 take\nper class", "2 takes\nper class", "all 14 takes\nper class"],
           [one, k2, full[2]], color=[AMBER, AMBER, ACCENT])
    for i, v in enumerate([one, k2, full[2]]):
        a2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    a2.set_ylim(0, 112)
    a2.set_title("Reduced supervision separates models", fontsize=11)
    for a in (a1, a2):
        a.grid(axis="y", alpha=.25); a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def augmentation(ab, path: Path):
    labels = {"none": "no\naugmentation", "full": "all\naugmentations",
              "mirror_only": "mirror\nonly", "no_mirror": "all except\nmirror",
              "geometric_only": "geometric\n+ noise"}
    keys = [k for k in ["none", "full", "mirror_only", "no_mirror", "geometric_only"] if k in ab["settings"]]
    vals = [ab["settings"][k]["mean"] * 100 for k in keys]
    best = max(vals)
    colors = [ACCENT if v == best else (AMBER if v < ab["settings"]["none"]["mean"] * 100 else GREY)
              for v in vals]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bars = ax.bar([labels[k] for k in keys], vals, color=colors)
    for b, k in zip(bars, keys):
        v = ab["settings"][k]["mean"] * 100
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center",
                fontsize=10, fontweight="bold")
        for s_ in ab["settings"][k]["scores"]:
            ax.plot(b.get_x() + b.get_width() / 2, s_ * 100, "o", color="#1F2937", ms=3.5, alpha=.65)
    ax.set_ylim(0, 100); ax.set_ylabel("test top-1 (%)")
    ax.set_title("Augmentation ablation — 2 takes per class, mean of 2 seeds", fontsize=11)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    exp = json.loads((ARTIFACTS / "experiments.json").read_text())
    base = json.loads((ARTIFACTS / "baselines.json").read_text())
    summary = json.loads((ARTIFACTS / "summary_bilstm_orig.json").read_text())
    data_efficiency(exp, ARTIFACTS / "data_efficiency.png")
    robustness(exp, ARTIFACTS / "robustness.png")
    saturation(exp, base, summary["test_top1"], ARTIFACTS / "saturation.png")
    ab_path = ARTIFACTS / "ablation.json"
    if ab_path.exists():
        augmentation(json.loads(ab_path.read_text()), ARTIFACTS / "augmentation.png")
    print("figures written")


if __name__ == "__main__":
    main()
