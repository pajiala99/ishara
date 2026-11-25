"""Download the INCLUDE dataset (264 ISL words, 4,284 videos, 57.8 GB) from its Kaggle mirror.

The official host (Zenodo, record 4010759) serves about 0.6 MB/s; the Kaggle mirror
daskoushik/include needs no login and gives about 4.4 MB/s with 4 parallel downloads
(16 in parallel made Kaggle drop connections). Files land in data/include/videos/ in the
INCLUDE layout that isl.extract already understands: <Category>/<N. Word>/<clip>.MOV.

Resumable: a file whose size already matches is skipped, partial files are re-fetched.

    python tools/download_include.py               # everything
    python tools/download_include.py --workers 4 --limit 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "daskoushik/include"
API = "https://www.kaggle.com/api/v1/datasets"
VIDEO_EXT = (".mov", ".mp4")


def curl_json(url: str) -> dict:
    # Python's urllib gets 404s from this API where curl does not, so every request goes through curl
    out = subprocess.run(["curl", "-s", "--retry", "3", url], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def list_files() -> list[tuple[str, int]]:
    files, token = [], None
    while True:
        url = f"{API}/list/{DATASET}?pageSize=200" + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        page = curl_json(url)
        files += [(f["name"], f.get("totalBytes", 0)) for f in page["datasetFiles"]]
        token = page.get("nextPageToken")
        if not token:
            return files


def local_path(name: str, out: Path) -> Path:
    # "Animals_1of2/Animals/1. Dog/MVI_2978.MOV" -> out/Animals/1. Dog/MVI_2978.MOV
    return out / name.split("/", 1)[1]


def fetch(name: str, size: int, out: Path) -> tuple[str, str]:
    dest = local_path(name, out)
    if dest.exists() and dest.stat().st_size == size:
        return name, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".zip.part")
    tmp = dest.with_name(dest.name + ".part")
    url = f"{API}/download/{DATASET}/" + urllib.parse.quote(name, safe="")
    r = subprocess.run(["curl", "-s", "-L", "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
                        "--max-time", "900", "-o", str(part), url])
    try:
        if r.returncode != 0 or not part.exists():
            return name, "fail"
        # Kaggle wraps every single-file download in a one-member zip
        try:
            with zipfile.ZipFile(part) as z, z.open(z.infolist()[0]) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile:
            part.rename(tmp)
        if tmp.stat().st_size != size:
            return name, "fail"
        tmp.rename(dest)
        return name, "ok"
    finally:
        part.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "include" / "videos")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (a quick test)")
    ap.add_argument("--first", nargs="+", default=[], help="download these words before the rest")
    a = ap.parse_args()

    files = [(n, s) for n, s in list_files() if n.lower().endswith(VIDEO_EXT)]
    if a.first:
        # same normalisation as isl.extract.label_of: "48. Hello" -> "hello"
        word = lambda n: n.split("/")[-2].split(". ", 1)[-1].strip().lower().replace(" ", "_")
        first = {w.lower() for w in a.first}
        files.sort(key=lambda f: word(f[0]) not in first)
    if a.limit:
        files = files[: a.limit]
    total = sum(s for _, s in files)
    print(f"{len(files)} videos, {total / 1e9:.1f} GB -> {a.out}", flush=True)

    done_bytes, counts, t0 = 0, {"ok": 0, "skip": 0, "fail": 0}, time.time()
    failed = []
    sizes = dict(files)
    with ThreadPoolExecutor(a.workers) as ex:
        futures = [ex.submit(fetch, n, s, a.out) for n, s in files]
        for i, fut in enumerate(as_completed(futures), 1):
            name, status = fut.result()
            counts[status] += 1
            if status == "fail":
                failed.append(name)
            else:
                done_bytes += sizes[name]
            if i % 50 == 0 or i == len(files):
                rate = done_bytes / 1e6 / max(time.time() - t0, 1)
                eta = (total - done_bytes) / 1e6 / max(rate, 1e-6) / 60
                print(f"{i}/{len(files)} | {done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB | {rate:.1f} MB/s | "
                      f"ok {counts['ok']} skip {counts['skip']} fail {counts['fail']} | ETA {eta:.0f} min", flush=True)

    (a.out.parent / "download_failed.json").write_text(json.dumps(failed, indent=1))
    print(f"done: {counts} in {(time.time() - t0) / 60:.0f} min"
          + (f"; rerun to retry the {len(failed)} failures" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
