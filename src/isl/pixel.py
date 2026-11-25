"""Pretrained pixel models, for comparison with the keypoint models.

Two networks pretrained on large image / video datasets read the raw frames, frozen (transfer
learning: only a small head is trained on our data):

  resnet18_bilstm  ResNet-18 pretrained on ImageNet turns each of 16 frames into 512 numbers;
                   a small BiLSTM with attention reads the 16 vectors in order.
  r3d18            R3D-18, a 3-D CNN pretrained on Kinetics-400 (400 human actions), turns the
                   16-frame clip into one 512-number vector; a linear head names the word.

It uses the split isl.compare saved (artifacts/split_<tag>.json), so every model is tested on the
same clips, and appends its results to artifacts/comparison_<tag>.json with the same metrics.

    python -m isl.pixel --tag kaggle61
    python -m isl.pixel --tag kaggle61 --cross data/include/keypoints
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .compare import metrics, save_result
from .config import ARTIFACTS, DATA_KP, ROOT, TrainConfig
from .dataset import index_keypoints
from .models import AttentionPool
from .train import device_auto

N_FRAMES = 16
FEATS = ROOT / "data" / "pixel_feats"
# keypoint folder -> the video folder it was extracted from (same relative layout)
VIDEO_ROOTS = {DATA_KP: ROOT / "data" / "videos" / "Video_Dataset" / "Video_Dataset",
               ROOT / "data" / "keypoints_tilt": ROOT / "data" / "videos" / "Video_Dataset" / "Video_Dataset",
               ROOT / "data" / "include" / "keypoints": ROOT / "data" / "include" / "videos"}
IMAGENET = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
KINETICS = (np.array([0.43216, 0.394666, 0.37645]), np.array([0.22803, 0.22145, 0.216989]))


def video_of(kp_path: Path) -> Path:
    for kp_root, vid_root in VIDEO_ROOTS.items():
        try:
            rel = kp_path.relative_to(kp_root)
        except ValueError:
            continue
        hits = [p for p in (vid_root / rel.parent).glob(rel.stem + ".*") if p.suffix.lower() in (".mp4", ".mov")]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no video for {kp_path}")


def sample_frames(path: Path, n: int = N_FRAMES) -> np.ndarray:
    """n RGB frames spread evenly over the clip, read in one pass."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx])


def square(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize the shorter side to `size`, then centre-crop a square."""
    h, w = frames.shape[1:3]
    s = size / min(h, w)
    out = np.stack([cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) for f in frames])
    h, w = out.shape[1:3]
    y, x = (h - size) // 2, (w - size) // 2
    return out[:, y:y + size, x:x + size]


class Backbones:
    def __init__(self, device):
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.models.video import R3D_18_Weights, r3d_18
        self.device = device
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        self.r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.r3d.fc = nn.Identity()
        self.resnet.eval().to(device); self.r3d.eval().to(device)

    @torch.no_grad()
    def per_frame(self, frames: np.ndarray) -> np.ndarray:
        img = (square(frames, 224) / 255.0 - IMAGENET[0]) / IMAGENET[1]                 # (16, 224, 224, 3)
        x = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().to(self.device)
        return self.resnet(x).cpu().numpy().astype(np.float32)                          # (16, 512)

    @torch.no_grad()
    def clip(self, frames: np.ndarray) -> np.ndarray:
        clip = (square(frames, 112) / 255.0 - KINETICS[0]) / KINETICS[1]                 # (16, 112, 112, 3)
        v = torch.from_numpy(clip.transpose(3, 0, 1, 2)[None]).float().to(self.device)  # (1, 3, 16, 112, 112)
        return self.r3d(v)[0].cpu().numpy().astype(np.float32)                          # (512,)

    def __call__(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.per_frame(frames), self.clip(frames)


def features(kp_paths: list[Path], backbones: Backbones) -> dict:
    """{kp_path: (resnet (16, 512), r3d (512,))}, cached under data/pixel_feats/."""
    out, todo, t0 = {}, 0, time.time()
    for i, p in enumerate(kp_paths, 1):
        cache = FEATS / (str(p.relative_to(ROOT)).replace("/", "__") + ".npz")
        if cache.exists():
            with np.load(cache) as z:
                out[p] = (z["resnet"], z["r3d"])
            continue
        res, r3d = backbones(sample_frames(video_of(p)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, resnet=res, r3d=r3d)
        out[p] = (res, r3d)
        todo += 1
        if todo % 100 == 0:
            print(f"  features {i}/{len(kp_paths)} ({(time.time() - t0) / todo:.2f} s per new clip)", flush=True)
    return out


class FrameBiLSTM(nn.Module):
    def __init__(self, n_classes, dim=512, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(hidden * 2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        h, _ = self.lstm(self.norm(x))
        return self.head(self.pool(h)[0])


class ClipLinear(nn.Module):
    def __init__(self, n_classes, dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3), nn.Linear(dim, n_classes))

    def forward(self, x):
        return self.net(x)


def fit(model, Xtr, ytr, Xva, yva, device, epochs=100, patience=15):
    torch.manual_seed(42)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xtr, ytr = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    Xva = torch.from_numpy(Xva).to(device)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(Xtr), device=device).split(32):
            opt.zero_grad(set_to_none=True)
            crit(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1).cpu().numpy() == yva).mean()
        if acc > best:
            best, bad, state = acc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


@torch.no_grad()
def scores(model, X, device):
    model.eval()
    return model(torch.from_numpy(X).to(device)).softmax(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cross", type=Path, default=None)
    a = ap.parse_args()

    split = json.loads((ARTIFACTS / f"split_{a.tag}.json").read_text())
    sets = {k: [ROOT / p for p in v] for k, v in split.items()}
    all_paths = [p for v in sets.values() for p in v]
    label_of = {p: str(np.load(p, allow_pickle=False)["label"]) for p in all_paths}
    classes = sorted(set(label_of.values()))
    idx = {c: i for i, c in enumerate(classes)}
    cross = []
    if a.cross:
        known = [s for s in index_keypoints([a.cross.resolve()], min_per_class=1) if s.label in idx]
        cross = [s.path for s in known]
        label_of.update({s.path: s.label for s in known})

    device = device_auto()
    bb = Backbones(device)
    frames = sample_frames(video_of(all_paths[0]))
    backbone_ms = {}
    for which, fn in ((0, bb.per_frame), (1, bb.clip)):
        fn(frames)
        t = time.perf_counter()
        for _ in range(5):
            fn(frames)
        backbone_ms[which] = (time.perf_counter() - t) / 5 * 1000
    print(f"{a.tag}: {len(all_paths)} clips, {len(classes)} words | backbone per clip on {device}: "
          f"ResNet-18 x16 frames {backbone_ms[0]:.0f} ms, R3D-18 {backbone_ms[1]:.0f} ms", flush=True)
    feats = features(all_paths + cross, bb)

    def stack(paths, which):
        return np.stack([feats[p][which] for p in paths]), np.array([idx[label_of[p]] for p in paths])

    out_path = ARTIFACTS / f"comparison_{a.tag}.json"
    for name, which, make in (("resnet18_bilstm", 0, lambda: FrameBiLSTM(len(classes))),
                              ("r3d18", 1, lambda: ClipLinear(len(classes)))):
        Xtr, ytr = stack(sets["train"], which)
        Xva, yva = stack(sets["val"], which)
        Xte, yte = stack(sets["test"], which)
        t0 = time.time()
        model = fit(make(), Xtr, ytr, Xva, yva, device)
        minutes = (time.time() - t0) / 60
        sc = scores(model, Xte, device)
        r = {"test": metrics(yte, sc, len(classes)), "train_minutes": minutes,
             "params": int(sum(p.numel() for p in model.parameters())),
             "latency_ms": backbone_ms[which], "pretrained": "ImageNet" if which == 0 else "Kinetics-400",
             "note": f"frozen pretrained backbone + trained head; latency is the backbone on {device}"}
        src = np.array(["include" if "include" in p.parts else "kaggle" for p in sets["test"]])
        if len(set(src)) > 1:
            r["test_by_source"] = {s: metrics(yte[src == s], sc[src == s], len(classes)) for s in sorted(set(src))}
        if cross:
            Xc, yc = stack(cross, which)
            r["new_signer"] = metrics(yc, scores(model, Xc, device), len(classes))
        save_result(out_path, name, r)
        m = r["test"]
        print(f"{name:16} top1 {m['top1']*100:5.1f}  top5 {m['top5']*100:5.1f}  F1 {m['f1']*100:5.1f}"
              + (f" | new signers top1 {r['new_signer']['top1']*100:5.1f}" if cross else ""), flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
