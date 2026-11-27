"""Ishara live dashboard: the project on one screen, read live from artifacts/ and data/.

    streamlit run dashboard/app.py

The offline page (python -m isl.dashboard -> dashboard/index.html) is for sharing; this one is for
presenting: it runs the shipped model on a held-out clip you pick, plays the clip, and shows the
INCLUDE download as it happens. Every number is read from the files the experiments write.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Streamlit's file watcher walks every module's __path__; torch.classes raises on that, so give it an empty one
torch.classes.__path__ = []

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isl.config import TrainConfig  # noqa: E402
from isl.dataset import load_sequence  # noqa: E402
from isl.features import resample, with_deltas  # noqa: E402
from isl.models import build_model  # noqa: E402

ART, DATA = ROOT / "artifacts", ROOT / "data"
TEAL, AMBER, SLATE, RED = "#0F766E", "#D97706", "#64748B", "#B91C1C"
TYPE_COLORS = {"keypoint network": TEAL, "classical": SLATE, "pretrained CNN (pixels)": AMBER}
NAMES = {"svm": "SVM", "random_forest": "Random Forest", "lstm": "LSTM", "bilstm": "BiLSTM + attention",
         "gru": "GRU", "bigru": "BiGRU", "tcn": "1-D CNN (TCN)", "transformer": "Transformer",
         "resnet18_bilstm": "ResNet-18 + BiLSTM (pretrained)", "r3d18": "R3D-18 3-D CNN (pretrained)"}
SETTINGS = [("ns_base", "Original features"), ("ns_aspect", "+ aspect fix"),
            ("ns_aspect_trim", "+ aspect fix + trim"), ("ns_aspect_trim_mirror", "+ aspect + trim + mirror")]
INCLUDE_TOTAL = 4284

st.set_page_config(page_title="Ishara dashboard", layout="wide")


def read(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def comparison(tag):
    d = read(f"comparison_{tag}.json") or {}
    return {k: v for k, v in d.items() if not k.startswith("_")}, d.get("_meta", {})


def new_signer(name, run=None):
    """New-signer metrics: the Kaggle-61 model with the aspect fix on the held-out half of the INCLUDE clips;
    pixel models use no keypoints, so theirs come from the Kaggle-61 run on all 167 clips."""
    ns, _ = comparison("ns_aspect")
    k61, _ = comparison("kaggle61")
    return (ns.get(name, {}).get("new_signer_test") or (run or {}).get("new_signer")
            or k61.get(name, {}).get("new_signer") or {})


def pct(x):
    return None if x is None else round(100 * x, 1)


def kind(name):
    if name in ("resnet18_bilstm", "r3d18"):
        return "pretrained CNN (pixels)"
    return "classical" if name in ("svm", "random_forest") else "keypoint network"


def style(fig, height=420):
    # transparent backgrounds so the charts sit right in both Streamlit themes
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1), font=dict(size=13))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.25)"); fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)")
    fig.for_each_trace(lambda t: t.update(texttemplate="%{text:.1f}") if t.type == "bar" and t.text is not None else None)
    return fig


st.title("Ishara: Indian Sign Language recognition")
st.caption("Team: Hardik Pajiala, Khushi Dangi, Geetika Parakh  ·  github.com/pajiala99/ishara  ·  "
           "every number here is read from the project's results files")
tabs = st.tabs(["Overview", "Try the model", "Model comparison", "New signers", "Data", "Earlier experiments",
                "Browser parity"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    k61, _ = comparison("kaggle61")
    comb, cmeta = comparison("combined")
    best = max(((new_signer(n).get("top1") or 0, n) for n in k61), default=(0, ""))
    c = st.columns(5)
    c[0].metric("Words", f"{cmeta.get('classes', 61)}", "Kaggle 61 + INCLUDE 264", delta_color="off")
    c[1].metric("Clips", f"{cmeta.get('clips', 1210):,}", "+1,640 tilted copies for training", delta_color="off")
    c[2].metric("Models compared", f"{len(k61)}", "classical, RNN, CNN, Transformer, pretrained", delta_color="off")
    c[3].metric("Chosen model", "Transformer",
                f"{pct(comb.get('transformer', {}).get('test', {}).get('top1'))}% on {cmeta.get('classes', '')} words",
                delta_color="off")
    c[4].metric("Best on new signers", f"{pct(best[0])}%", NAMES.get(best[1], best[1]), delta_color="off")
    st.markdown("**Video** → every 2nd–3rd frame → **MediaPipe** (33 body + 2 × 21 hand points) → **body-frame "
                "normalisation** (origin mid-shoulder, unit shoulder width) → **239 features per frame** → resample to "
                "**64 frames** + velocity → **64 × 478** → **sequence model** → word → **speech**")
    st.markdown(
        "- **One signer:** every model scores **96.7–100%**, so that test cannot choose a model.\n"
        "- **New signers** (INCLUDE, 9 shared words): with the aspect fix, keypoint models reach **15–41%** (chance 1.6%); "
        "pretrained pixel CNNs drop to chance.\n"
        "- **Combined 315 words:** the Transformer and the 1-D CNN tie at **95.8%** top-1; the Transformer is chosen because "
        "it does best on signers it never saw (41.0% against 28.9%).\n"
        "- **Bug found and fixed:** x was measured in frame-widths and y in frame-heights, so wide videos stretched every "
        "body. The fix and multi-signer training are the two levers measured in *New signers*.")
    cols = st.columns(2)
    for col, (f, cap) in zip(cols, (("preprocessing_stages.png", "Preprocessing on one clip"),
                                    ("invariance_proof.png", "Two cameras, one skeleton"))):
        if (ART / f).exists():
            col.image(str(ART / f), caption=cap, use_container_width=True)

# ------------------------------------------------------------------ try the model
with tabs[1]:
    clips = sorted((DATA / "parity" / "clips").rglob("*.mp4")) if (DATA / "parity" / "clips").exists() else []
    if not clips:
        st.info("No held-out clips found in data/parity/clips.")
    else:

        @st.cache_resource
        def shipped_model():
            state = torch.load(ROOT / "models" / "bilstm_orig.pt", map_location="cpu", weights_only=False)
            m = build_model(state["arch"], len(state["classes"]), TrainConfig())
            m.load_state_dict(state["model"]); m.eval()
            return m, state["classes"]

        words = sorted({c.parent.name for c in clips})
        left, right = st.columns(2)
        word = left.selectbox("Word", words, index=words.index("Hello") if "Hello" in words else 0)
        clip = left.selectbox("Held-out clip (the model never trained on it)",
                              [c for c in clips if c.parent.name == word], format_func=lambda p: p.stem)
        left.video(str(clip))
        kp = DATA / "keypoints" / word / f"{clip.stem}.npz"
        if kp.exists():
            m, classes = shipped_model()
            x = torch.from_numpy(with_deltas(resample(load_sequence(kp), 64)))[None]
            with torch.no_grad():
                prob = m(x).softmax(-1)[0].numpy()
            top = np.argsort(-prob)[:5]
            d = pd.DataFrame({"Word": [classes[i].replace("_", " ") for i in top],
                              "Probability %": (100 * prob[top]).astype(float)})
            fig = px.bar(d, x="Probability %", y="Word", orientation="h", text="Probability %",
                         title="Top 5 from the shipped BiLSTM + attention")
            fig.update_traces(marker_color=[TEAL] + [SLATE] * 4)
            fig.update_yaxes(autorange="reversed", title=None); fig.update_xaxes(range=[0, 100])
            right.plotly_chart(style(fig, 360), use_container_width=True)
            ok = classes[top[0]] == word.lower().replace(" ", "_")
            right.markdown(f"**{'Correct' if ok else 'Wrong'}:** *{classes[top[0]].replace('_', ' ')}* at "
                           f"{100 * prob[top[0]]:.1f}% (true word: *{word}*). It chooses among 61 words; guessing "
                           "would score 1.6%.")

# ------------------------------------------------------------------ model comparison
with tabs[2]:
    options = {"kaggle61": "Kaggle, 61 words (one signer)",
               "combined_partial": "Combined, early (INCLUDE partly downloaded)", "combined": "Combined, 315 words"}
    latest = max((i for i, t in enumerate(options) if read(f"comparison_{t}.json")), default=0)
    tag = st.radio("Dataset", list(options), index=latest, horizontal=True, format_func=options.get)
    res, meta = comparison(tag)
    if not res:
        st.info("The combined comparison starts when the INCLUDE download finishes.")
    else:
        df = pd.DataFrame([{
            "Model": NAMES.get(n, n), "Type": kind(n), "Top-1 %": pct(v["test"]["top1"]), "Top-5 %": pct(v["test"]["top5"]),
            "Precision %": pct(v["test"]["precision"]), "Recall %": pct(v["test"]["recall"]),
            "Macro F1 %": pct(v["test"]["f1"]), "New signers top-1 %": pct(new_signer(n, v).get("top1")),
            "New signers top-5 %": pct(new_signer(n, v).get("top5")),
            "Params (M)": None if v.get("params") is None else round(v["params"] / 1e6, 2),
            "ms per clip": round(v.get("latency_ms") or 0, 2), "Train min": round(v.get("train_minutes") or 0, 1)}
            for n, v in res.items()]).sort_values(["New signers top-1 %", "Macro F1 %"], ascending=False)
        st.caption(f"Test set {meta.get('test', '?')} clips, {meta.get('classes', '?')} words; new signers: "
                   "the Kaggle-61 model with the aspect fix on the held-out half of the INCLUDE clips of the 9 shared words "
                   "(pixel models: all 167 clips).")
        st.dataframe(df, hide_index=True, use_container_width=True)
        metric = st.selectbox("Compare on", ["New signers top-1 %", "Top-1 %", "Macro F1 %", "Precision %", "Recall %",
                                             "Top-5 %", "ms per clip", "Params (M)"])
        d = df.dropna(subset=[metric]).sort_values(metric, ascending=metric in ("ms per clip", "Params (M)"))
        fig = px.bar(d, x=metric, y="Model", orientation="h", color="Type", text=metric, color_discrete_map=TYPE_COLORS)
        fig.update_yaxes(categoryorder="array", categoryarray=list(d["Model"])[::-1], title=None)
        st.plotly_chart(style(fig), use_container_width=True)
        if df["New signers top-1 %"].notna().any():
            fig = go.Figure()
            fig.add_bar(y=df["Model"], x=df["Top-1 %"], name="Same signer (test set)", orientation="h", marker_color=SLATE)
            fig.add_bar(y=df["Model"], x=df["New signers top-1 %"], name="New signers", orientation="h", marker_color=TEAL)
            fig.update_layout(barmode="group", title="Same signer vs new signers, top-1 %")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 540), use_container_width=True)

# ------------------------------------------------------------------ new signers
with tabs[3]:
    st.markdown("Trained on the Kaggle signer, tested on **INCLUDE signers** doing the 9 shared words (61-way, chance "
                "1.6%). Settings are chosen on one half of those clips (**dev**) and reported on the other (**test**).")
    rows = []
    for tag, label in SETTINGS:
        res, _ = comparison(tag)
        for n, v in res.items():
            if n not in ("lstm", "gru", "bilstm"):    # the three models run in every setting
                continue
            for part, half in (("new_signer_dev", "dev"), ("new_signer_test", "test")):
                if part in v:
                    rows.append({"Setting": label, "Model": NAMES.get(n, n), "Half": half,
                                 "Top-1 %": pct(v[part]["top1"]), "Top-5 %": pct(v[part]["top5"])})
    if not rows:
        st.info("The improvement experiments are still running.")
    else:
        df = pd.DataFrame(rows)
        half = st.radio("Half", ["test", "dev"], horizontal=True)
        d = df[df["Half"] == half]
        fig = px.bar(d, x="Setting", y="Top-1 %", color="Model", barmode="group", text="Top-1 %",
                     category_orders={"Setting": [s for _, s in SETTINGS]},
                     color_discrete_sequence=[TEAL, AMBER, SLATE], title=f"New-signer top-1 by setting ({half} half)")
        st.plotly_chart(style(fig), use_container_width=True)
        st.dataframe(d.pivot_table(index="Setting", columns="Model", values="Top-1 %").reindex(
            [s for _, s in SETTINGS if s in set(d["Setting"])]), use_container_width=True)
    ns, _ = comparison("ns_aspect")
    if ns:
        d = pd.DataFrame([{"Model": NAMES.get(n, n), "Type": kind(n), "Top-1 %": pct(v["new_signer_test"]["top1"]),
                           "Top-5 %": pct(v["new_signer_test"]["top5"])}
                          for n, v in ns.items() if "new_signer_test" in v]).sort_values("Top-1 %")
        fig = px.bar(d, x="Top-1 %", y="Model", orientation="h", color="Type", text="Top-1 %",
                     color_discrete_map=TYPE_COLORS, title="Every keypoint model with the aspect fix (test half)")
        st.plotly_chart(style(fig), use_container_width=True)

# ------------------------------------------------------------------ data
with tabs[4]:
    vids = DATA / "include" / "videos"
    kps = DATA / "include" / "keypoints"
    n_vid = sum(1 for _ in vids.rglob("*.MOV")) if vids.exists() else 0
    n_kp = sum(1 for _ in kps.rglob("*.npz")) if kps.exists() else 0
    c = st.columns(4)
    c[0].metric("Kaggle ISL clips", "3,630", "1,210 originals + tilted copies", delta_color="off")
    c[1].metric("INCLUDE videos downloaded", f"{n_vid:,} / {INCLUDE_TOTAL:,}")
    c[2].metric("INCLUDE keypoints extracted", f"{n_kp:,}")
    c[3].metric("Words in both datasets", "9")
    st.progress(min(n_vid / INCLUDE_TOTAL, 1.0), text=f"INCLUDE download: {100 * n_vid / INCLUDE_TOTAL:.0f}%")
    st.dataframe(pd.DataFrame([
        {"Dataset": "Kaggle ISL video dataset", "Signers": "1", "Words": 61, "Videos": "3,630 (1,210 + tilts)",
         "Frame size": "640×720, 760×720, 880×720", "fps": "30", "Clip length": "≈ 6.8 s"},
        {"Dataset": "INCLUDE (AI4Bharat)", "Signers": "several", "Words": 264, "Videos": "4,284",
         "Frame size": "1920×1080", "fps": "25", "Clip length": "≈ 2.5 s"},
        {"Dataset": "Combined", "Signers": "several", "Words": 315, "Videos": "5,494 + 1,640 training-only tilts",
         "Frame size": "mixed (aspect fix)", "fps": "mixed", "Clip length": "resampled to 64 frames"},
    ]), hide_index=True, use_container_width=True)
    st.markdown("**Shared words** (the new-signer test): hello, thank you, good morning, good afternoon, man, wife, "
                "temple, key, clean.  \n**Tilted copies:** each Kaggle take ships three times; 1,630 copies whose "
                "original is a training take join training only, and 691 whose original is a test take are left out, "
                "so no near-duplicate of a test clip is ever trained on.")

# ------------------------------------------------------------------ earlier experiments
with tabs[5]:
    exp = read("experiments.json") or {}
    c1, c2 = st.columns(2)
    if exp.get("data_efficiency"):
        d = pd.DataFrame(exp["data_efficiency"])
        fig = go.Figure()
        fig.add_scatter(x=d["takes_per_class"], y=100 * d["test_top1"], name="top-1", mode="lines+markers", line_color=TEAL)
        fig.add_scatter(x=d["takes_per_class"], y=100 * d["test_top5"], name="top-5", mode="lines+markers", line_color=AMBER)
        fig.update_layout(title="Data efficiency: accuracy vs examples per word", xaxis_title="takes per word",
                          yaxis_title="%")
        c1.plotly_chart(style(fig), use_container_width=True)
    if exp.get("robustness"):
        d = pd.DataFrame({"Perturbation": list(exp["robustness"]), "Top-1 %": [pct(v) for v in exp["robustness"].values()]})
        fig = px.bar(d, x="Top-1 %", y="Perturbation", orientation="h", text="Top-1 %", title="Robustness of the trained model")
        fig.update_traces(marker_color=[RED if p == "occlude_left_hand" else TEAL for p in d["Perturbation"]])
        fig.update_yaxes(autorange="reversed", title=None)
        c2.plotly_chart(style(fig), use_container_width=True)
    abl = read("ablation.json")
    if abl:
        d = pd.DataFrame({"Augmentation": list(abl["settings"]), "Top-1 %": [pct(v["mean"]) for v in abl["settings"].values()]})
        fig = px.bar(d, x="Augmentation", y="Top-1 %", text="Top-1 %", title="Augmentation ablation (2 takes per word, 2 seeds)")
        fig.update_traces(marker_color=[AMBER if a in ("full", "mirror_only") else TEAL for a in d["Augmentation"]])
        st.plotly_chart(style(fig, 380), use_container_width=True)

# ------------------------------------------------------------------ browser parity
with tabs[6]:
    par, feat = read("parity.json"), read("feature_parity.json")
    c = st.columns(3)
    c[0].metric("Export check", "PyTorch = ONNX", "same word on 182 / 182", delta_color="off")
    if feat:
        c[1].metric("Feature check", f"{feat['bit_identical_clips']:,} / {feat['clips']:,} clips bit-identical",
                    f"{feat['values_compared']:,} values compared", delta_color="off")
    if par:
        s = par["summary"]
        c[2].metric("End to end", f"same word on {s['top1_agree']} / {s['compared']}",
                    f"browser correct on {s['browser_correct']} / {s['compared']}", delta_color="off")
        d = pd.DataFrame(par["clips"])
        d["same word"] = d["python_top1"] == d["browser_top1"]
        fig = px.scatter(d, x="python_p", y="browser_p", color="same word", hover_data=["clip"],
                         labels={"python_p": "Python confidence", "browser_p": "Browser confidence"},
                         color_discrete_map={True: TEAL, False: RED}, title="Python vs browser, 182 held-out clips")
        st.plotly_chart(style(fig), use_container_width=True)
