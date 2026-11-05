"""Prove the browser demo (web/isl.js) computes what the Python pipeline computes.

Three checks, from the inside out:

  1. export   ONNX vs PyTorch on every held-out clip's features   (python -m isl.export_onnx)
  2. features the JavaScript feature code vs features.py, on the cached landmarks of all
              1,210 clips, compared bit for bit                    (features, then web/check_features.mjs)
  3. clips    end to end on the same video files: MediaPipe Python + numpy + PyTorch against
              MediaPipe JS + isl.js + onnxruntime-web in a real browser   (clips, serve, compare)

    python -m isl.parity features              # dump landmarks + numpy inputs for check 2
    node web/check_features.mjs
    python -m isl.parity clips                 # held-out test clips -> H.264 + Python reference
    python -m isl.parity serve                 # open http://127.0.0.1:8765/web/parity.html
    python -m isl.parity compare               # browser vs Python -> artifacts/parity.json
    python -m isl.parity site --out ../hunter-portfolio/assets/isl   # sample clips for the portfolio demo

The dataset's clips are MPEG-4 Part 2, which browsers can't decode, so both sides read the
same H.264 copies made with macOS `avconvert`. Python is also run on the original files to
show the conversion changes nothing that matters.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from .config import ARTIFACTS, DATA_KP, MODELS, ROOT, SEQ_LEN, TrainConfig
from .dataset import index_keypoints, stratified_split
from .demo import video_for
from .export_onnx import load
from .features import build_frame_features, resample, with_deltas

PARITY = ROOT / "data" / "parity"
PRESET = "PresetMediumQuality"      # H.264, 428x480 for these clips, ~350 KB each
FRAME_FLOATS = 33 * 4 + 21 * 3 * 2 + 2   # pose, left, right, mask = 260 per frame


def flat_frames(pose, left, right, mask) -> np.ndarray:
    """(T, 260) float32 rows in the order web/check_features.mjs and parity.html use."""
    t = len(pose)
    return np.concatenate([pose.reshape(t, -1), left.reshape(t, -1), right.reshape(t, -1), mask], 1).astype(np.float32)


# ------------------------------------------------------------------ check 2: features
def cmd_features(a):
    out = PARITY / "features"
    out.mkdir(parents=True, exist_ok=True)
    index, raw_off, n = [], 0, 0
    with open(out / "raw.f32", "wb") as raw, open(out / "x.f32", "wb") as xs:
        for s in index_keypoints([DATA_KP]):
            with np.load(s.path, allow_pickle=False) as z:
                pose, left, right, mask = z["pose"], z["left"], z["right"], z["mask"]
            x = with_deltas(resample(build_frame_features(pose, left, right, mask), SEQ_LEN))
            raw.write(flat_frames(pose, left, right, mask).tobytes())
            xs.write(np.ascontiguousarray(x, np.float32).tobytes())
            index.append({"clip": str(s.path.relative_to(DATA_KP)), "frames": len(pose), "raw_offset": raw_off})
            raw_off += len(pose) * FRAME_FLOATS
            n += 1
    (out / "index.json").write_text(json.dumps({"frame_floats": FRAME_FLOATS, "x_floats": SEQ_LEN * 478, "clips": index}))
    print(f"wrote {n} clips to {out}")


# ------------------------------------------------------------------ check 3: clips
def transcode(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        subprocess.run(["avconvert", "-s", str(src), "-p", PRESET, "-o", str(dst), "--replace"],
                       check=True, capture_output=True)


def python_reference(model, video: Path):
    """The Python pipeline exactly as demo.py --video runs it: fresh extractor, every 3rd frame."""
    import cv2
    from .keypoints import LandmarkExtractor

    with LandmarkExtractor() as ex:
        pose, left, right, mask, fps = ex.run_video(video, stride=3)
    cap = cv2.VideoCapture(str(video))
    frame_count = 0
    while cap.read()[0]:
        frame_count += 1
    cap.release()
    x = torch.from_numpy(with_deltas(resample(build_frame_features(pose, left, right, mask), SEQ_LEN)))[None]
    with torch.no_grad():
        probs = model(x).softmax(-1)[0].numpy()
    return probs, flat_frames(pose, left, right, mask), fps, frame_count


def cmd_clips(a):
    cfg = TrainConfig()
    _, _, test_s = stratified_split(index_keypoints([DATA_KP]), cfg.val_frac, cfg.test_frac, cfg.seed)
    if a.limit:
        test_s = test_s[: a.limit]
    model, classes = load(a.ckpt)
    (PARITY / "python").mkdir(parents=True, exist_ok=True)
    clips, ref = [], {}
    t0 = time.time()
    for i, s in enumerate(test_s, 1):
        src = video_for(s.path)
        rel = s.path.relative_to(DATA_KP).with_suffix(".mp4")
        dst = PARITY / "clips" / rel
        transcode(src, dst)
        probs, frames, fps, n = python_reference(model, dst)
        probs_orig, _, _, _ = python_reference(model, src)
        name = str(rel)
        np.save(PARITY / "python" / (name.replace("/", "__") + ".npy"), frames)
        clips.append({"name": name, "url": f"/data/parity/clips/{rel.as_posix()}", "fps": fps, "frame_count": n})
        ref[name] = {"label": s.label, "probs": [float(p) for p in probs], "frames": len(frames),
                     "top1_original_file": classes[int(probs_orig.argmax())],
                     "p_top1_original_file": float(probs_orig.max())}
        print(f"[{i}/{len(test_s)}] {name}: {classes[int(probs.argmax())]} {probs.max():.3f} "
              f"(original file: {classes[int(probs_orig.argmax())]})", flush=True)
    (PARITY / "clips.json").write_text(json.dumps({"classes": classes, "clips": clips}, indent=1))
    (PARITY / "python.json").write_text(json.dumps(ref))
    print(f"{len(clips)} clips in {time.time() - t0:.0f} s -> {PARITY}")


def cmd_site(a):
    """Copy a few held-out clips, with Python's answer for each, for the portfolio's live demo."""
    meta = json.loads((PARITY / "clips.json").read_text())
    ref = json.loads((PARITY / "python.json").read_text())
    classes, out = meta["classes"], []
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    for word in a.words:
        c = next(c for c in meta["clips"] if ref[c["name"]]["label"] == word)   # first held-out clip of the word
        probs = np.array(ref[c["name"]]["probs"])
        top = probs.argsort()[::-1][:5]
        shutil.copy(PARITY / "clips" / c["name"], a.out / "clips" / f"{word}.mp4")
        out.append({"file": f"{word}.mp4", "word": word, "source": c["name"], "fps": c["fps"],
                    "frame_count": c["frame_count"],
                    "python": {"top1": classes[int(top[0])], "p": float(probs[top[0]]),
                               "top5": [[classes[int(i)], float(probs[i])] for i in top]}})
    (a.out / "clips.json").write_text(json.dumps({"classes": classes, "clips": out}, indent=1))
    print(f"{len(out)} clips -> {a.out}")


# ------------------------------------------------------------------ serve the harness
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kw):
        super().__init__(*args, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/parity/results":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        (PARITY / "browser.json").write_bytes(body)
        self.send_response(204)
        self.end_headers()
        print(f"saved browser results ({len(body) / 1e6:.1f} MB)", flush=True)

    def log_message(self, fmt, *args):
        pass


def cmd_serve(a):
    print(f"serving {ROOT} on http://127.0.0.1:{a.port}/web/parity.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


# ------------------------------------------------------------------ compare
def cmd_compare(a):
    meta = json.loads((PARITY / "clips.json").read_text())
    classes = meta["classes"]
    py = json.loads((PARITY / "python.json").read_text())
    br = json.loads((PARITY / "browser.json").read_text())
    rows = []
    for c in meta["clips"]:
        name = c["name"]
        p, b = py[name], br["results"].get(name)
        if b is None:
            rows.append({"clip": name, "missing": True})
            continue
        pp, bp = np.array(p["probs"]), np.array(b["probs"])
        lp = np.load(PARITY / "python" / (name.replace("/", "__") + ".npy"))
        lb = np.frombuffer(base64.b64decode(b["landmarks"]), np.float32).reshape(-1, FRAME_FLOATS)
        same_t = len(lp) == len(lb)
        rows.append({
            "clip": name, "label": p["label"],
            "python_top1": classes[int(pp.argmax())], "browser_top1": classes[int(bp.argmax())],
            "python_p": float(pp.max()), "browser_p": float(bp[pp.argmax()]),
            "max_abs_prob_diff": float(np.abs(pp - bp).max()),
            "frames_python": len(lp), "frames_browser": len(lb),
            "landmark_mean_abs_diff": float(np.abs(lp[:, :-2] - lb[:, :-2]).mean()) if same_t else None,
            "hand_flags_agree": float((lp[:, -2:] == lb[:, -2:]).mean()) if same_t else None,
            "python_top1_on_original_file": p["top1_original_file"],
        })
    ok = [r for r in rows if not r.get("missing")]
    agree = sum(r["python_top1"] == r["browser_top1"] for r in ok)
    summary = {
        "clips": len(rows), "compared": len(ok), "top1_agree": agree,
        "python_correct": sum(r["python_top1"] == r["label"] for r in ok),
        "browser_correct": sum(r["browser_top1"] == r["label"] for r in ok),
        "transcode_keeps_python_top1": sum(r["python_top1"] == r["python_top1_on_original_file"] for r in ok),
        "same_frame_count": sum(r["frames_python"] == r["frames_browser"] for r in ok),
        "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in ok),
        "median_abs_prob_diff": float(np.median([r["max_abs_prob_diff"] for r in ok])),
        "median_landmark_abs_diff": float(np.median([r["landmark_mean_abs_diff"] for r in ok if r["landmark_mean_abs_diff"] is not None])),
        "hand_flags_agree": float(np.mean([r["hand_flags_agree"] for r in ok if r["hand_flags_agree"] is not None])),
        "browser": br.get("env", {}), "preset": PRESET,
    }
    out = {"summary": summary, "clips": rows}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "parity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=2))
    for r in ok:
        if r["python_top1"] != r["browser_top1"]:
            print(f"DISAGREE {r['clip']}: python {r['python_top1']} {r['python_p']:.3f} / browser {r['browser_top1']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features")
    c = sub.add_parser("clips")
    c.add_argument("--ckpt", type=Path, default=MODELS / "bilstm_orig.pt")
    c.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8765)
    sub.add_parser("compare")
    w = sub.add_parser("site")
    w.add_argument("--out", type=Path, required=True, help="e.g. ../hunter-portfolio/assets/isl")
    w.add_argument("--words", nargs="+", default=["hello", "thank_you", "good_morning", "tea", "elephant", "what_is_your_name"])
    a = ap.parse_args()
    if shutil.which("avconvert") is None and a.cmd == "clips":
        raise SystemExit("clips needs macOS avconvert to make browser-playable H.264 copies")
    {"features": cmd_features, "clips": cmd_clips, "serve": cmd_serve, "compare": cmd_compare, "site": cmd_site}[a.cmd](a)


if __name__ == "__main__":
    main()
