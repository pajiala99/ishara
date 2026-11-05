"""Export the trained BiLSTM to ONNX for the browser demo, and check it against PyTorch.

    python -m isl.export_onnx                         # models/bilstm_orig.pt -> models/isl_bilstm.onnx
    python -m isl.export_onnx --out ../hunter-portfolio/assets/isl

The graph takes the same (batch, 64, 478) feature tensor as the PyTorch model and returns
softmax probabilities, so the browser only has to reproduce features.py. The check runs every
held-out test clip through both runtimes and fails if any top-1 prediction differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import ARTIFACTS, DATA_KP, F_BASE, F_TOTAL, MODELS, N_HAND, POSE_IDX, SEQ_LEN, TrainConfig
from .dataset import ISLSequenceDataset, index_keypoints, stratified_split
from .models import build_model

ONNX_NAME = "isl_bilstm.onnx"
OPSET = 17
IR_VERSION = 8          # onnxruntime-web reads IR 8 everywhere; newer onnx defaults higher


class WithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).softmax(-1)


def load(ckpt: Path):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(state["arch"], len(state["classes"]), TrainConfig())
    model.load_state_dict(state["model"])
    return model.eval(), state["classes"]


def export(model, path: Path):
    import onnx

    x = torch.zeros(1, SEQ_LEN, F_TOTAL)
    # the wrapper must be in eval mode too: the exporter restores the wrapper's mode afterwards,
    # and a fresh module defaults to train(), which would switch dropout back on in `model`
    torch.onnx.export(WithSoftmax(model).eval(), (x,), str(path), dynamo=False, opset_version=OPSET,
                      input_names=["features"], output_names=["probs"],
                      dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}})
    proto = onnx.load(str(path))
    proto.ir_version = IR_VERSION
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))


def check(model, path: Path, classes) -> dict:
    """Every held-out test clip through PyTorch (CPU) and onnxruntime; compare the outputs."""
    import onnxruntime as ort

    cfg = TrainConfig()
    _, _, test_s = stratified_split(index_keypoints([DATA_KP]), cfg.val_frac, cfg.test_frac, cfg.seed)
    ds = ISLSequenceDataset(test_s, classes, cfg.seq_len)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = np.array([ds[i][1] for i in range(len(ds))])
    with torch.no_grad():
        p_torch = model(x).softmax(-1).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    p_onnx = sess.run(["probs"], {"features": x.numpy()})[0]
    same_top1 = int((p_torch.argmax(1) == p_onnx.argmax(1)).sum())
    return {
        "test_clips": len(ds),
        "top1_agree": same_top1,
        "max_abs_prob_diff": float(np.abs(p_torch - p_onnx).max()),
        "onnx_test_top1": float((p_onnx.argmax(1) == y).mean()),
        "torch_test_top1": float((p_torch.argmax(1) == y).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=MODELS / "bilstm_orig.pt")
    ap.add_argument("--out", type=Path, default=MODELS, help="folder for the .onnx and its metadata")
    a = ap.parse_args()

    model, classes = load(a.ckpt)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / ONNX_NAME
    export(model, path)
    report = check(model, path, classes)
    print(json.dumps(report, indent=2))
    if report["top1_agree"] != report["test_clips"]:
        raise SystemExit("ONNX and PyTorch disagree on a held-out clip")

    meta = {
        "model": ONNX_NAME, "source_checkpoint": a.ckpt.name,
        "source_sha256": hashlib.sha256(a.ckpt.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "opset": OPSET, "input": ["batch", SEQ_LEN, F_TOTAL], "classes": classes,
        "seq_len": SEQ_LEN, "f_base": F_BASE, "pose_idx": POSE_IDX, "n_hand": N_HAND,
        "export_check": report,
    }
    (a.out / "isl_meta.json").write_text(json.dumps(meta, indent=2))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "onnx_export.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "classes"}, indent=2))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and isl_meta.json")


if __name__ == "__main__":
    main()
