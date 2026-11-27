"""Build the results dashboard: one offline HTML page with every dataset, model and metric.

Reads the comparison and experiment results in artifacts/ and writes dashboard/index.html. No
server and no internet needed: the charts are inline SVG and the figures are embedded, so the
page opens from the file in any browser. Rebuild after each new run.

    python -m isl.dashboard
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import json

from .config import ARTIFACTS, ROOT

OUT = ROOT / "dashboard" / "index.html"
NAMES = {"svm": "SVM", "random_forest": "Random Forest", "lstm": "LSTM", "bilstm": "BiLSTM + attention",
         "gru": "GRU", "bigru": "BiGRU", "tcn": "1-D CNN (TCN)", "transformer": "Transformer",
         "resnet18_bilstm": "ResNet-18 + BiLSTM", "r3d18": "R3D-18 (3-D CNN)"}
KIND = {"svm": "Classical", "random_forest": "Classical", "resnet18_bilstm": "Pretrained (ImageNet), pixels",
        "r3d18": "Pretrained (Kinetics-400), pixels"}
SETTINGS = [("ns_base", "Original features"), ("ns_aspect", "+ aspect fix"),
            ("ns_aspect_trim", "+ aspect fix + trim"), ("ns_aspect_trim_mirror", "+ aspect + trim + mirror")]


def load(name):
    p = ARTIFACTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def pct(x, d=1):
    return "&ndash;" if x is None else f"{x * 100:.{d}f}"


def get(d, *keys):
    for k in keys:
        if isinstance(d, list) and isinstance(k, int) and k < len(d):
            d = d[k]
        elif isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def hbars(rows, series, width=640):
    """Grouped horizontal bars, 0-100 %. rows: [(label, [value per series])]; series: [(name, css var)]."""
    bar, gap, group_gap, left, right = 9, 2, 12, 190, 46
    plot = width - left - right
    group_h = len(series) * (bar + gap) - gap
    height = len(rows) * (group_h + group_gap) + 26
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for t in (0, 25, 50, 75, 100):
        x = left + plot * t / 100
        out.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height - 20}" class="grid"/>'
                   f'<text x="{x}" y="{height - 6}" class="tick" text-anchor="middle">{t}%</text>')
    y = 0
    for label, values in rows:
        out.append(f'<text x="{left - 10}" y="{y + group_h / 2 + 4}" class="label" text-anchor="end">'
                   f'{html.escape(label)}</text>')
        for (name, var), v in zip(series, values):
            if v is not None:
                w = max(plot * v, 2)
                out.append(f'<g class="mark"><title>{html.escape(label)} · {html.escape(name)}: {v * 100:.1f}%</title>'
                           f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar}" rx="4" fill="var({var})"/>'
                           f'<text x="{left + w + 5:.1f}" y="{y + bar - 1}" class="value">{v * 100:.1f}</text></g>')
            y += bar + gap
        y += group_gap - gap
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:var({v})"></i>{html.escape(n)}</span>' for n, v in series)
    return f'<div class="legend">{legend}</div>' + "".join(out)


def img(name):
    p = ARTIFACTS / name
    if not p.exists():
        return ""
    return f'<img alt="{html.escape(name)}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def main():
    k61, comb = load("comparison_kaggle61.json"), load("comparison_combined.json")
    meta, cmeta = k61.get("_meta", {}), comb.get("_meta", {})
    models = [m for m in NAMES if m in k61 or m in comb]

    ns_run = load("comparison_ns_aspect.json")

    def new(m):
        """New-signer metrics: aspect-fixed Kaggle-61 model on the held-out half of the INCLUDE clips;
        the pixel models use no keypoints, so theirs is the Kaggle-61 run on all 167 clips."""
        return get(ns_run, m, "new_signer_test") or get(k61, m, "new_signer")

    best = max((m for m in models if get(new(m), "top1") is not None),
               key=lambda m: new(m)["top1"], default=None)
    words = cmeta.get("classes") or meta.get("classes", 61)
    tiles = [("Words", f"{words}", "Kaggle 61 + INCLUDE" if cmeta else "Kaggle dataset; combined run pending"),
             ("Clips", f"{cmeta.get('clips', meta.get('clips', 0)):,}", "videos turned into keypoints"),
             ("Models compared", f"{len(models)}", "classical, sequence, CNN, pretrained"),
             ("Best on new signers", (pct(get(new(best), "top1")) + "%") if best else "&ndash;",
              f"{NAMES.get(best, '')}, top-1; chance is 1.6%")]

    # the choice, by the rule in compare_report: combined macro F1, within 1 point counts as a tie, ties go to
    # new-signer top-1
    f1 = {m: get(comb, m, "test", "f1") for m in models if get(comb, m, "test", "f1") is not None}
    chosen = runner = None
    if f1:
        tied = [m for m in f1 if f1[m] >= max(f1.values()) - 0.01]
        ranked = sorted(tied, key=lambda m: -(get(new(m), "top1") or 0))
        chosen, runner = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    choice = ""
    if chosen:
        c, r = comb[chosen]["test"], (comb[runner]["test"] if runner else None)
        choice = (f"<section><h2>Chosen model: {NAMES[chosen]}</h2><p>On the combined test set ({cmeta.get('test')} clips, "
                  f"{cmeta.get('classes')} words) it scores <b>{pct(c['top1'])}%</b> top-1, {pct(c['top5'])}% top-5 and "
                  f"{pct(c['f1'])} macro F1, in {comb[chosen].get('latency_ms', 0):.1f} ms per clip on a CPU. "
                  + (f"{NAMES[runner]} is level on that test ({pct(r['top1'])}% top-1, {pct(r['f1'])} F1), so the tie goes to "
                     f"signers neither model has seen: <b>{pct(get(new(chosen), 'top1'))}%</b> against "
                     f"{pct(get(new(runner), 'top1'))}% (chance 1.6%)." if runner else "")
                  + "</p><p class='muted'>Rule: rank by combined macro F1; models within 1 point of the best count as "
                  "tied (1 point is about 8 test clips) and are ranked by new-signer top-1.</p></section>")
    kp = [get(new(m), "top1") for m in models if m in k61 and "pixels" not in KIND.get(m, "")
          and get(new(m), "top1") is not None]

    same_vs_new = hbars([(NAMES[m], [get(k61, m, "test", "top1"), get(new(m), "top1")])
                         for m in models if m in k61],
                        [("Same signer (Kaggle test)", "--s1"), ("New signers (INCLUDE)", "--s2")])

    rows = []
    for m in models:
        a, c = k61.get(m, {}), comb.get(m, {})
        cells = [NAMES[m], KIND.get(m, "Keypoints, sequence")]
        cells += [pct(get(a, "test", k)) for k in ("top1", "top5", "precision", "recall", "f1")]
        cells += [pct(get(new(m), k)) for k in ("top1", "top5")]
        if comb:
            cells += [pct(get(c, "test", k)) for k in ("top1", "f1")]
        params = (c or a).get("params")
        cells += ["&ndash;" if params is None else f"{params / 1e6:.2f} M", f"{(c or a).get('latency_ms', 0):.1f}"]
        rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    head = ["Model", "Type", "Top-1", "Top-5", "Precision", "Recall", "Macro F1", "New signers top-1",
            "New signers top-5"] + (["Combined top-1", "Combined F1"] if comb else []) + ["Params", "ms per clip"]
    table = ("<table class='sortable'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) +
             "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    abl = {tag: load(f"comparison_{tag}.json") for tag, _ in SETTINGS}
    abl_rows = [(f"{NAMES[m]} · {label}", [get(abl[tag], m, "new_signer_dev", "top1"),
                                           get(abl[tag], m, "new_signer_test", "top1")])
                for m in ("lstm", "gru", "bilstm") for tag, label in SETTINGS if get(abl[tag], m)]
    ablation = (hbars(abl_rows, [("Dev half (chooses the setting)", "--s1"), ("Test half (reported)", "--s3")])
                if abl_rows else "<p class='muted'>Running: results appear after the next rebuild.</p>")

    exp = load("experiments.json")
    datasets = f"""<table><thead><tr><th>Dataset</th><th>Words</th><th>Clips</th><th>Signers</th><th>Used for</th></tr></thead><tbody>
<tr><td>Kaggle ISL video dataset</td><td>61</td><td>1,210 originals</td><td>1</td><td>train / validation / test</td></tr>
<tr><td>Kaggle tilted copies</td><td>61</td><td>1,640 used of 2,531</td><td>1</td><td>training only, and only when the original take is a training take</td></tr>
<tr><td>INCLUDE (AI4Bharat)</td><td>263</td><td>4,284</td><td>several</td><td>new-signer test (9 shared words) and the combined set</td></tr>
<tr><td>Combined</td><td>{cmeta.get('classes', 316)}</td><td>{cmeta.get('clips') or 'pending'}</td><td>several</td><td>the final comparison</td></tr>
</tbody></table>"""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ishara results</title>
<style>
:root{{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--accent:#0f766e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--card:#232321;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent:#2dd4bf}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:28px 16px 60px}} h1{{margin:0;font-size:28px}} h2{{font-size:19px;margin:0 0 4px}}
.sub{{color:var(--ink2);margin:4px 0 24px}} .muted{{color:var(--ink2);font-size:13px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin:0 0 18px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:18px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.tile b{{display:block;font-size:30px;line-height:1.2}} .tile span{{color:var(--ink2);font-size:13px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:10px 0 4px}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.chart{{width:100%;height:auto}} .chart .grid{{stroke:var(--line)}} .chart text{{fill:var(--ink2);font-size:11px}}
.chart .label{{fill:var(--ink);font-size:12px}} .chart .mark:hover rect{{opacity:.75}}
.callout{{border-left:3px solid var(--accent);padding:8px 14px;margin:12px 0 0;background:color-mix(in srgb,var(--accent) 8%,transparent)}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{color:var(--ink2);font-weight:600;cursor:pointer;user-select:none}} td:nth-child(n+3){{font-variant-numeric:tabular-nums}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}} .figs img{{width:100%;border-radius:6px;background:#fff}}
ul{{margin:8px 0 0;padding-left:20px}}
</style></head><body><main>
<h1>Ishara: Indian Sign Language recognition</h1>
<p class="sub">Team: Hardik Pajiala, Khushi Dangi, Geetika Parakh &middot; results dashboard, built {dt.date.today():%d %b %Y} from <code>artifacts/</code></p>
<div class="tiles">{"".join(f'<div class="tile"><span>{t}</span><b>{v}</b><span>{n}</span></div>' for t, v, n in tiles)}</div>
{choice}

<section><h2>Same signer vs new signers</h2>
<p class="muted">Every model is trained on the Kaggle dataset (one signer), then scored on its own test clips and on {meta.get('cross_clips', 167)} INCLUDE clips of the {len(meta.get('cross_words', [])) or 9} shared words, signed by other people. Choice of 61 words; guessing scores 1.6%.</p>
{same_vs_new}
<p class="callout">On the same signer every model scores 97&ndash;100%, so that test cannot choose a model. On new signers, with the aspect fix, keypoint models reach {pct(min(kp), 0) if kp else "&ndash;"}&ndash;{pct(max(kp), 0) if kp else "&ndash;"}% (chance 1.6%), while the pretrained pixel CNNs fall to chance: they learned one person's look rather than the sign. New signers: the Kaggle model with the aspect fix on the held-out half of the INCLUDE clips of the 9 shared words; pixel models on all 167.</p></section>

<section><h2>Every model, every metric</h2><p class="muted">Click a column to sort. Kaggle test: {meta.get('test', 182)} clips. Macro scores weight every word equally. Latency: keypoint models on CPU for one clip; pixel models include their frozen backbone on the Mac GPU.</p>
<div class="scroll">{table}</div></section>

<section><h2>Improving accuracy on new signers</h2><p class="muted">The new-signer clips are split in half: the dev half chooses the setting, the test half is reported. Aspect fix: MediaPipe measures x in frame widths and y in frame heights, so INCLUDE's 1920&times;1080 bodies looked twice as tall as Kaggle's 640&times;720 ones; x is now put in frame-height units. Trim: cut the resting frames before and after the sign.</p>
{ablation}</section>

<section><h2>Datasets</h2><div class="scroll">{datasets}</div></section>

<section><h2>Earlier experiments (Kaggle, one signer)</h2><p class="muted">Data efficiency: {pct(get(exp, 'data_efficiency', 0, 'test_top1'))}% top-1 from one example per word, {pct(get(exp, 'data_efficiency', 3, 'test_top1'))}% from five.</p>
<div class="figs">{img("data_efficiency.png")}{img("robustness.png")}{img("augmentation.png")}</div></section>

<section><h2>Bugs found and fixed along the way</h2><ul>
<li><b>Frame shape.</b> Features depended on the video's width-to-height ratio; fixed with the aspect correction, which the deployed 315-word Transformer uses.</li>
<li><b>Mirror augmentation.</b> Swapped the hands but not the arms; fixed and every experiment rerun.</li>
<li><b>Live app.</b> Fed the model a mirrored frame; now only the window is mirrored.</li></ul></section>
</main>
<script>
document.querySelectorAll("table.sortable th").forEach((th, i) => th.addEventListener("click", () => {{
  const body = th.closest("table").tBodies[0], rows = [...body.rows], asc = th.dataset.asc !== "1";
  const val = r => {{ const t = r.cells[i].textContent.replace(/[^0-9.\\-]/g, ""); return t === "" ? NaN : parseFloat(t); }};
  rows.sort((a, b) => {{ const x = val(a), y = val(b);
    if (isNaN(x) || isNaN(y)) return asc ? a.cells[i].textContent.localeCompare(b.cells[i].textContent) : b.cells[i].textContent.localeCompare(a.cells[i].textContent);
    return asc ? x - y : y - x; }});
  th.dataset.asc = asc ? "1" : "0"; rows.forEach(r => body.appendChild(r));
}}));
</script></body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
