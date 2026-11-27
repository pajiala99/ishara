"""Both encoders, and the trained checkpoint committed in models/."""
from __future__ import annotations

import json

import pytest
import torch

from isl.config import ARTIFACTS, F_TOTAL, MODELS, SEQ_LEN, TrainConfig
from isl.models import build_model


@pytest.mark.parametrize("name", ["bilstm", "transformer"])
def test_encoders_give_class_logits_and_attention_over_time(name):
    torch.manual_seed(0)
    model = build_model(name, 7, TrainConfig()).eval()
    with torch.no_grad():
        logits, attn = model(torch.randn(3, SEQ_LEN, F_TOTAL), return_attn=True)
    assert logits.shape == (3, 7) and attn.shape == (3, SEQ_LEN)
    torch.testing.assert_close(attn.sum(1), torch.ones(3))


def test_committed_checkpoint_still_fits_the_code_and_the_class_list():
    """models/bilstm_orig.pt, the original 61-word model, still loads (any script with --ckpt)."""
    state = torch.load(MODELS / "bilstm_orig.pt", map_location="cpu", weights_only=True)
    classes = json.loads((ARTIFACTS / "classes.json").read_text())
    assert state["arch"] == "bilstm" and state["classes"] == classes and len(classes) == 61

    model = build_model(state["arch"], len(classes), TrainConfig())
    model.load_state_dict(state["model"])  # strict: every saved weight matches a layer
    model.eval()
    with torch.no_grad():
        probs = model(torch.zeros(2, SEQ_LEN, F_TOTAL)).softmax(-1)
    assert probs.shape == (2, 61) and torch.isfinite(probs).all()


def test_shipped_transformer_fits_the_code_and_asks_for_the_aspect_fix():
    """models/isl_transformer.pt is the default for demo.py, realtime.py, unseen_signer.py and the ONNX export."""
    from isl.shipped import SHIPPED, aspect_of

    state = torch.load(SHIPPED, map_location="cpu", weights_only=False)
    assert state["arch"] == "transformer" and len(state["classes"]) == 315
    assert state["aspect"] is True and (ARTIFACTS / f"split_{state['split']}.json").exists()
    assert aspect_of(state, 1920, 1080) == 1920 / 1080 and aspect_of({}, 1920, 1080) is None

    model = build_model(state["arch"], len(state["classes"]), TrainConfig())
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        probs = model(torch.zeros(2, SEQ_LEN, F_TOTAL)).softmax(-1)
    assert probs.shape == (2, 315) and torch.isfinite(probs).all()
