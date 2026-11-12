const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const PROJ = "/Users/hardikpajiala/Desktop/isl-recognition";
const ART = path.join(PROJ, "artifacts");
const readJSON = (f, d = null) => { try { return JSON.parse(fs.readFileSync(path.join(ART, f), "utf8")); } catch { return d; } };
const img = (f) => { const p = path.join(ART, f); return fs.existsSync(p) ? p : null; };

const S = readJSON("summary_bilstm_orig.json", {}) || {};
const M = readJSON("test_metrics.json", {}) || {};
const B = readJSON("baselines.json", {}) || {};
const FULL = readJSON("summary_bilstm_full.json", null);
const E = readJSON("experiments.json", {}) || {};
const EA = readJSON("error_analysis.json", {}) || {};
const AB = readJSON("ablation.json", {}) || {};
const US = readJSON("unseen_signer.json", null);
const abs = (k) => ((AB.settings || {})[k] || {}).mean;
const shot = (k) => (E.data_efficiency || []).find((r) => r.takes_per_class === k) || {};
const rob = (k) => (E.robustness || {})[k];
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");

// ---------------------------------------------------------------- palette
const PANEL = "0F3B3C", ACCENT = "2DD4BF", ACC_D = "0F766E", TINT = "E6F6F3";
const DARK = "1F2937", MUTED = "4B5563", LINE = "D8DDE4";
const W = 13.333, H = 7.5, MX = 0.62;

async function icon(Comp, color, px = 256) {
  const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: px }));
  const buf = await sharp(Buffer.from(svg)).resize(px, px).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Ishara: Indian Sign Language Recognition";

function slide() { const s = pres.addSlide(); s.background = { color: "FFFFFF" }; return s; }

function header(s, kicker, title) {
  s.addText(kicker.toUpperCase(), { x: MX, y: 0.45, w: 8, h: 0.25, margin: 0, fontFace: "Calibri",
    fontSize: 11, bold: true, color: ACC_D, charSpacing: 2.5, isTextBox: true });
  s.addText(title, { x: MX, y: 0.75, w: W - 2 * MX, h: 0.6, margin: 0, fontFace: "Cambria",
    fontSize: 32, bold: true, color: DARK, isTextBox: true });
}

function body(s, items, x, y, w, size = 14.5) {
  s.addText(items.map((t, i) => ({
    text: typeof t === "string" ? t : t.text,
    options: { bullet: true, breakLine: i < items.length - 1, bold: typeof t === "object" && t.bold,
      color: typeof t === "object" && t.muted ? MUTED : DARK },
  })), { x, y, w, h: 3.6, margin: 0, fontFace: "Calibri", fontSize: size, paraSpaceAfter: 9,
    lineSpacingMultiple: 1.06, valign: "top", isTextBox: true });
}

function card(s, x, y, w, h, fill = TINT) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill }, line: { color: fill } });
}

function stat(s, x, y, w, value, label, color = ACC_D) {
  s.addText(value, { x, y, w, h: 0.75, margin: 0, fontFace: "Cambria", fontSize: 34, bold: true,
    color, align: "center", valign: "bottom", isTextBox: true });
  s.addText(label, { x, y: y + 0.78, w, h: 0.5, margin: 0, fontFace: "Calibri", fontSize: 11.5,
    color: MUTED, align: "center", valign: "top", isTextBox: true });
}

function tableRows(s, headers, rows, x, y, w, colW) {
  s.addTable([headers.map((h) => ({ text: h, options: { bold: true, color: ACC_D, fill: { color: TINT } } })),
    ...rows.map((r) => r.map((c) => ({ text: String(c) })))], {
    x, y, w, colW, border: { pt: 0.75, color: LINE }, fontFace: "Calibri", fontSize: 12.5,
    color: DARK, rowH: 0.34, valign: "middle", margin: 6,
  });
}

async function chipRow(s, items, x, y, w, accent = ACCENT) {
  let cx = x, cy = y;
  for (const t of items) {
    const cw = t.length * 12 * 0.0075 + 0.4;
    if (cx + cw > x + w + 0.001) { cx = x; cy += 0.52; }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: cw, h: 0.4, rectRadius: 0.2,
      fill: { color: "FFFFFF" }, line: { color: accent, width: 1.25 } });
    s.addText(t, { x: cx, y: cy, w: cw, h: 0.4, margin: 0, fontFace: "Calibri", fontSize: 12,
      color: DARK, align: "center", valign: "middle", isTextBox: true });
    cx += cw + 0.12;
  }
  return cy + 0.4;
}

async function flowRow(s, steps, y, h = 1.0, x = MX, width = W - 2 * MX) {
  const n = steps.length, gap = 0.26, bw = (width - gap * (n - 1)) / n;
  for (let i = 0; i < n; i++) {
    const [name, ic] = steps[i];
    const bx = x + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: bw, h, rectRadius: 0.1,
      fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 },
      shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 90, opacity: 0.12 } });
    if (ic) s.addImage({ data: await icon(ic, ACC_D), x: bx + bw / 2 - 0.15, y: y + 0.13, w: 0.3, h: 0.3 });
    s.addText(name, { x: bx + 0.05, y: y + (ic ? 0.5 : 0.2), w: bw - 0.1, h: h - (ic ? 0.55 : 0.25),
      margin: 0, fontFace: "Calibri", fontSize: 10.5, bold: true, color: DARK,
      align: "center", valign: "middle", isTextBox: true });
    if (i < n - 1) s.addShape(pres.shapes.LINE, { x: bx + bw + 0.04, y: y + h / 2, w: gap - 0.08, h: 0,
      line: { color: ACC_D, width: 1.75, endArrowType: "triangle" } });
  }
}

async function build() {
  // ------------------------------------------------------------- 1. title
  {
    const s = slide();
    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: PANEL }, line: { color: PANEL } });
    s.addShape(pres.shapes.OVAL, { x: MX, y: 1.05, w: 1.05, h: 1.05, fill: { color: ACCENT }, line: { color: ACCENT } });
    s.addImage({ data: await icon(fa.FaSignLanguage, PANEL), x: MX + 0.26, y: 1.31, w: 0.53, h: 0.53 });
    s.addText("MINOR PROJECT  ·  COMPUTER VISION & DEEP LEARNING", { x: MX, y: 2.45, w: 9, h: 0.3,
      margin: 0, fontFace: "Calibri", fontSize: 12, bold: true, color: ACCENT, charSpacing: 3, isTextBox: true });
    s.addText([{ text: "Ishara", options: { fontSize: 54, breakLine: true } },
      { text: "Indian Sign Language Recognition", options: { fontSize: 30 } }], { x: MX, y: 2.85, w: 9.5, h: 1.9, margin: 0,
      fontFace: "Cambria", bold: true, color: "FFFFFF", lineSpacingMultiple: 1.0, isTextBox: true });
    s.addText("Real-time word-level recognition from video using MediaPipe skeletal keypoints and a BiLSTM with temporal attention",
      { x: MX, y: 4.85, w: 8.4, h: 0.8, margin: 0, fontFace: "Calibri", fontSize: 15, italic: true,
        color: "C8DEDB", isTextBox: true });
    const chips = ["Python", "MediaPipe", "PyTorch", "OpenCV", "scikit-learn"];
    let cx = MX;
    for (const t of chips) {
      const cw = t.length * 12 * 0.0075 + 0.45;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: 6.05, w: cw, h: 0.42, rectRadius: 0.21,
        fill: { color: PANEL }, line: { color: ACCENT, width: 1 } });
      s.addText(t, { x: cx, y: 6.05, w: cw, h: 0.42, margin: 0, fontFace: "Calibri", fontSize: 12,
        color: ACCENT, align: "center", valign: "middle", isTextBox: true });
      cx += cw + 0.14;
    }
    if (S.test_top1 != null) {
      card(s, 9.9, 2.85, 2.82, 1.9, "14504F");
      s.addText(pct(S.test_top1), { x: 9.9, y: 3.0, w: 2.82, h: 0.95, margin: 0, fontFace: "Cambria",
        fontSize: 44, bold: true, color: ACCENT, align: "center", valign: "middle", isTextBox: true });
      s.addText(`on the standard split\n— and why that is not the result`, { x: 9.9, y: 3.9, w: 2.82, h: 0.7,
        margin: 0, fontFace: "Calibri", fontSize: 12, color: "C8DEDB", align: "center", isTextBox: true });
    }
    s.addNotes("Isolated word-level ISL recognition. Camera to text and speech, running live on a laptop.");
  }

  // ------------------------------------------------------- 2. the problem
  {
    const s = slide();
    header(s, "Motivation", "A communication gap of millions, bridged by a webcam");
    body(s, [
      "India has roughly 63 million people with hearing loss, and only a few hundred certified ISL interpreters (ISLRTC estimates).",
      "Outside specialist settings, a deaf signer cannot be understood without a third person present.",
      "ISL is a full language with its own grammar, and is heavily two-handed — models trained on American Sign Language transfer poorly.",
      "Goal: a laptop-camera system that turns an isolated ISL sign into text and speech, on-device and in real time.",
    ], MX, 1.75, 7.0);
    card(s, 8.1, 1.75, 4.6, 3.1);
    s.addText("Why it is hard", { x: 8.4, y: 1.95, w: 4.0, h: 0.35, margin: 0, fontFace: "Calibri",
      fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, [
      "A sign = handshape + orientation + location + movement + facial expression, all at once",
      "Signs differ only in movement direction, so a single frame is not enough",
      "Signer size, distance, lighting and background all vary",
      "Two hands occlude each other constantly",
    ], 8.4, 2.4, 4.0, 12);
    await flowRow(s, [["Camera", fa.FaVideo], ["Keypoints", fa.FaHandPaper], ["Sequence model", fa.FaBrain],
      ["Word", fa.FaFont], ["Speech", fa.FaVolumeUp]], 5.35, 1.0);
    s.addNotes("Frame the problem as a sequence problem, not an image problem.");
  }

  // ------------------------------------------------------------ 3. dataset
  {
    const s = slide();
    header(s, "Dataset", "Indian Sign Language Video Dataset");
    card(s, MX, 1.7, W - 2 * MX, 1.5);
    s.addText([{ text: "Kaggle · MIT licence · 3.48 GB of video", options: { bold: true, fontSize: 15, breakLine: true } },
      { text: "61 word signs recorded as webcam video at 640×720, 30 fps, about 6.8 s per clip. Each take is also provided as a left-tilt and a right-tilt variant, giving 60 clips per class.", options: { fontSize: 12.5, color: MUTED } }],
      { x: MX + 0.28, y: 1.8, w: 6.6, h: 1.3, margin: 0, fontFace: "Calibri", valign: "middle", isTextBox: true });
    stat(s, 7.7, 1.78, 1.6, "61", "word classes");
    stat(s, 9.3, 1.78, 1.6, "3,630", "video clips");
    stat(s, 10.9, 1.78, 1.6, "1,210", "original takes");
    s.addText("Leakage control: the three tilt variants of one take are near-duplicates", { x: MX, y: 3.45, w: 11, h: 0.3,
      margin: 0, fontFace: "Calibri", fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, [
      "Splitting them independently would put a near-copy of a test clip into training — accuracy would look excellent and mean nothing.",
      "Our splitter groups every variant of a take and assigns the whole group to one split, then stratifies per class so all 61 words appear in train, validation and test.",
      `Split actually used: ${S.train ?? "—"} train · ${S.val ?? "—"} validation · ${S.test ?? "—"} test clips.`,
    ], MX, 3.85, 11.5, 13.5);
    s.addText("Sample classes:  Bear · Thank you · What is your Name · Umbrella · Temple · Vegetables · Writer · Volcano",
      { x: MX, y: 5.95, w: 11.5, h: 0.4, margin: 0, fontFace: "Calibri", fontSize: 12.5, italic: true,
        color: MUTED, isTextBox: true });
    s.addNotes("Mention the tilt-variant leakage trap — examiners like this.");
  }

  // -------------------------------------------------- 4. why keypoints
  {
    const s = slide();
    header(s, "Design decision", "Why skeletal keypoints instead of raw pixels");
    const colW = (W - 2 * MX - 0.4) / 2;
    card(s, MX, 1.75, colW, 3.5, "F3F4F6");
    s.addText("Raw video → 3D-CNN", { x: MX + 0.3, y: 1.95, w: colW - 0.6, h: 0.35, margin: 0,
      fontFace: "Calibri", fontSize: 15, bold: true, color: MUTED, isTextBox: true });
    body(s, ["≈ 280 M pixel values per clip", "Needs hundreds of clips per class", "Learns background, clothing and skin tone as shortcuts",
      "Hours of GPU time per experiment", "Fails on a new room or a new signer"], MX + 0.3, 2.4, colW - 0.6, 12.5);
    card(s, MX + colW + 0.4, 1.75, colW, 3.5);
    s.addText("Keypoints → sequence model", { x: MX + colW + 0.7, y: 1.95, w: colW - 0.6, h: 0.35, margin: 0,
      fontFace: "Calibri", fontSize: 15, bold: true, color: ACC_D, isTextBox: true });
    body(s, ["64 × 478 numbers ≈ 120 KB per clip", "Trains on tens of clips per class", "Background, clothing and lighting are gone by construction",
      "Minutes per experiment on a laptop GPU", "Runs live at interactive frame rates"], MX + colW + 0.7, 2.4, colW - 0.6, 12.5);
    card(s, MX, 5.5, W - 2 * MX, 1.15, PANEL);
    s.addText([{ text: "≈ 2,000× smaller input, ", options: { bold: true, color: ACCENT } },
      { text: "with the linguistically meaningful information kept: where the hands are, what shape they make, and how they move.", options: { color: "D7E9E7" } }],
      { x: MX + 0.35, y: 5.5, w: W - 2 * MX - 0.7, h: 1.15, margin: 0, fontFace: "Calibri", fontSize: 15,
        valign: "middle", isTextBox: true });
    s.addNotes("Data efficiency, invariance and cost — the three reasons.");
  }

  // ------------------------------------------------------- 5. pipeline
  {
    const s = slide();
    header(s, "System architecture", "End-to-end pipeline");
    await flowRow(s, [["Video\n6.8 s @ 30 fps", null], ["Frame sampling\nevery 3rd frame", null],
      ["MediaPipe\npose + 2 hands", null], ["Normalisation\nbody-relative", null],
      ["Resample\n64 frames", null], ["BiLSTM\n+ attention", null], ["Word\n+ speech", null]], 1.85, 1.15);
    s.addText("Stage outputs", { x: MX, y: 3.4, w: 6, h: 0.3, margin: 0, fontFace: "Calibri", fontSize: 14,
      bold: true, color: ACC_D, isTextBox: true });
    tableRows(s, ["Stage", "Output shape", "Purpose"], [
      ["Frame sampling", "~68 frames", "Drop redundant frames, 3× faster extraction"],
      ["MediaPipe landmarks", "33 pose + 2 × 21 hand points", "Discard pixels, keep geometry"],
      ["Feature engineering", "T × 239", "Signer-invariant posture + handshape"],
      ["Temporal resampling + deltas", "64 × 478", "Fixed size, tempo-invariant, motion explicit"],
      ["BiLSTM + attention", `${S.classes ?? 61} logits`, "Word prediction with confidence"],
    ], MX, 3.75, W - 2 * MX, [2.6, 3.0, 6.5]);
    card(s, MX, 5.95, W - 2 * MX, 0.8, TINT);
    s.addText("Keypoint extraction is the only expensive stage — it runs once over the dataset (multi-process, ~7 minutes for 1,210 clips) and is cached to .npz, so every later experiment trains in minutes.",
      { x: MX + 0.35, y: 5.95, w: W - 2 * MX - 0.7, h: 0.8, margin: 0, fontFace: "Calibri",
        fontSize: 13, color: DARK, valign: "middle", isTextBox: true });
    s.addNotes("Keypoint extraction is the expensive stage and is cached once to .npz.");
  }

  // --------------------------------------------- 6. preprocessing maths
  {
    const s = slide();
    header(s, "Preprocessing", "Making the features signer-invariant");
    card(s, MX, 1.75, 6.3, 2.0, "F3F4F6");
    s.addText("origin = (L_shoulder + R_shoulder) / 2\nscale  = || L_shoulder − R_shoulder ||\nx′ = (x − origin ) / scale", {
      x: MX + 0.3, y: 1.9, w: 5.7, h: 1.7, margin: 0, fontFace: "Courier New", fontSize: 14,
      color: DARK, valign: "middle", lineSpacingMultiple: 1.25, isTextBox: true });
    body(s, [
      "Every landmark is re-expressed relative to the shoulder mid-point and divided by shoulder width — recomputed per frame.",
      "A tall signer far from the camera and a short signer close to it now produce nearly identical features.",
      "Hands are encoded twice: body-relative (where the sign happens) and wrist-local, scaled by hand size (what the handshape is).",
      "Missing hand → zeros plus a presence flag, so 'absent' differs from 'at the origin'.",
    ], 7.3, 1.75, 5.4, 12.5);
    s.addText("Feature vector per frame", { x: MX, y: 4.05, w: 6, h: 0.3, margin: 0, fontFace: "Calibri",
      fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    tableRows(s, ["Block", "Dims", "Encodes"], [
      ["Pose: 9 upper-body points (x, y, visibility)", "27", "Posture and arm configuration"],
      ["Hands, body-relative (2 × 21 × x, y)", "84", "Location of the sign"],
      ["Hands, wrist-local (2 × 21 × x, y, z)", "126", "Handshape"],
      ["Hand-present flags", "2", "Detection status"],
      ["+ frame-to-frame velocity", "478 total", "Movement, the defining parameter of a sign"],
    ], MX, 4.4, W - 2 * MX, [5.6, 1.7, 4.8]);
    s.addNotes("This slide is the answer to 'what preprocessing did you do'.");
  }

  // ------------------------------------------------------ 7. augmentation
  {
    const s = slide();
    header(s, "Preprocessing", "Augmentation — training set only");
    tableRows(s, ["Transform", "Range", "Real-world variation it imitates"], [
      ["Horizontal mirror, left and right swapped", "p = 0.3", "Left- vs right-dominant signers"],
      ["Rotation", "±10°", "Camera not level"],
      ["Scale", "±10%", "Distance from camera"],
      ["Translation", "±0.05", "Signer off-centre"],
      ["Temporal speed jitter", "±20%", "Faster or slower signing"],
      ["Random frame dropout", "10% of frames", "Dropped frames, detector misses"],
      ["Gaussian coordinate noise", "σ = 0.01", "Landmark jitter"],
    ], MX, 1.8, W - 2 * MX, [4.4, 2.0, 5.7]);
    card(s, MX, 5.3, W - 2 * MX, 1.35);
    s.addText([{ text: "Mirroring is not a sign flip.  ", options: { bold: true, color: ACC_D } },
      { text: "Negating x alone would leave the left hand's data in the left-hand channel while the body is mirrored. The implementation negates x and swaps every left/right pair: the two hand blocks with their presence flags, and the shoulders, elbows, wrists and hips of the pose — otherwise a mirrored sign becomes incoherent.", options: { color: DARK } }],
      { x: MX + 0.35, y: 5.3, w: W - 2 * MX - 0.7, h: 1.35, margin: 0, fontFace: "Calibri", fontSize: 13.5,
        valign: "middle", isTextBox: true });
    s.addNotes("Validation and test clips are never augmented.");
  }

  // -------------------------------------------------------- 8. the model
  {
    const s = slide();
    header(s, "Model", "BiLSTM with temporal attention");
    await flowRow(s, [["Input\n64 × 478", null], ["LayerNorm\n+ Linear 192", null], ["BiLSTM × 2\n192 hidden", null],
      ["Attention\npooling", null], [`Softmax\n${S.classes ?? 61} words`, null]], 1.8, 1.05, MX, 8.2);
    card(s, 9.2, 1.8, 3.5, 1.05, PANEL);
    s.addText(`${S.params ? (S.params / 1e6).toFixed(2) : "1.67"} M parameters`, { x: 9.2, y: 1.8, w: 3.5, h: 1.05,
      margin: 0, fontFace: "Cambria", fontSize: 18, bold: true, color: ACCENT, align: "center",
      valign: "middle", isTextBox: true });
    body(s, [
      { text: "Bidirectional LSTM — a recorded clip is complete, so the end of a sign may inform the beginning.", bold: false },
      "Attention pooling learns which frames carry the sign: every clip has idle lead-in and lead-out that mean-pooling would blend in.",
      "Velocity features and tempo normalisation let a small recurrent model beat a Transformer at this data scale (the Transformer encoder is implemented as an ablation).",
      "Regularisation sized for a small dataset: dropout 0.3, label smoothing 0.1, weight decay 1e-4, gradient clipping, early stopping.",
    ], MX, 3.35, 7.1, 13.5);
    card(s, 8.1, 3.35, 4.6, 2.9, "F3F4F6");
    s.addText("Attention, concretely", { x: 8.4, y: 3.55, w: 4.0, h: 0.3, margin: 0, fontFace: "Calibri",
      fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    s.addText("score_t = w₂ · tanh(W₁ h_t)\nα = softmax(score)\nclip = Σ α_t · h_t", { x: 8.4, y: 3.95, w: 4.0, h: 1.1,
      margin: 0, fontFace: "Courier New", fontSize: 12.5, color: DARK, lineSpacingMultiple: 1.2, isTextBox: true });
    s.addText("Additive (Bahdanau-style) attention with a single query — not multi-head self-attention. The weights are also a built-in explanation of which frames the model used.",
      { x: 8.4, y: 5.1, w: 4.0, h: 1.0, margin: 0, fontFace: "Calibri", fontSize: 12, color: MUTED, isTextBox: true });
    s.addNotes("Be ready to explain LSTM gates and why bidirectional is allowed offline.");
  }

  // ----------------------------------------------------- 9. training setup
  {
    const s = slide();
    header(s, "Training", "Setup and protocol");
    tableRows(s, ["Setting", "Value"], [
      ["Loss", "Cross-entropy, label smoothing 0.1"],
      ["Optimiser", "AdamW, weight decay 1e-4"],
      ["Schedule", "OneCycle, peak LR 1e-3"],
      ["Batch size / epochs", `32 / up to ${S.epochs_run ?? 60} (early stopping, patience 15)`],
      ["Model selection", "Best validation accuracy; test set read once"],
      ["Hardware", `${(S.device ?? "mps") === "mps" ? "Apple Silicon GPU (PyTorch MPS)" : S.device}`],
      ["Training time", S.minutes ? `${S.minutes.toFixed(1)} minutes` : "minutes, not hours"],
    ], MX, 1.8, 6.3, [2.5, 3.8]);
    const curves = img("training_curves.png");
    if (curves) s.addImage({ path: curves, x: 7.2, y: 1.8, w: 5.5, h: 2.0, sizing: { type: "contain", w: 5.5, h: 2.0 } });
    card(s, MX, 4.6, W - 2 * MX, 2.0);
    s.addText("Honest protocol", { x: MX + 0.35, y: 4.78, w: 6, h: 0.3, margin: 0, fontFace: "Calibri",
      fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, [
      "Split computed once, before training, with a fixed seed; the test set is read only by the evaluation script.",
      "Augmentation applied to training samples only — validation and test see the clean pipeline.",
      "Baseline StandardScaler fitted inside a pipeline on training data only, so no test statistics leak.",
      "Known limitation: the split is signer-mixed, not signer-independent — accuracy on a brand-new signer will be lower.",
    ], MX + 0.35, 5.15, W - 2 * MX - 0.7, 12.5);
    s.addNotes("Say the limitation before the examiner finds it.");
  }

  // ------------------------------- 10. results: the benchmark saturates
  {
    const s = slide();
    header(s, "Results", "First finding: the benchmark saturates");
    const sat = img("saturation.png");
    if (sat) s.addImage({ path: sat, x: MX, y: 1.7, w: 7.4, h: 3.0, sizing: { type: "contain", w: 7.4, h: 3.0 } });
    card(s, 8.3, 1.7, 4.4, 3.0, "FEF4E2");
    s.addText("Why 100% is a warning, not a win", { x: 8.6, y: 1.9, w: 3.8, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: "B45309", isTextBox: true });
    body(s, ["One signer, one room, one camera position",
      "Each word recorded in a single five-minute session — its 20 takes are near-copies",
      "A Random Forest on time-pooled features also reaches 100%",
      "A benchmark that every model solves cannot tell models apart"], 8.6, 2.3, 3.8, 11.5);
    card(s, MX, 4.95, W - 2 * MX, 1.6, PANEL);
    s.addText([{ text: "So we built harder protocols.  ", options: { bold: true, color: ACCENT } },
      { text: "Reduced supervision (how few examples does it need?), perturbation robustness (did it learn the sign or the session?), and an augmentation ablation. Those are the numbers in the rest of this deck.", options: { color: "D7E9E7" } }],
      { x: MX + 0.35, y: 4.95, w: W - 2 * MX - 0.7, h: 1.6, margin: 0, fontFace: "Calibri",
        fontSize: 14.5, valign: "middle", isTextBox: true });
    s.addNotes("Lead with the diagnosis. Examiners reward finding the flaw yourself.");
  }

  // ------------------------------------------------ 11. data efficiency
  {
    const s = slide();
    header(s, "Protocol A", "Data efficiency — how few examples does it need?");
    const de = img("data_efficiency.png");
    if (de) s.addImage({ path: de, x: MX, y: 1.7, w: 7.0, h: 4.1, sizing: { type: "contain", w: 7.0, h: 4.1 } });
    tableRows(s, ["Takes per class", "Train clips", "Top-1", "Top-5"],
      (E.data_efficiency || []).map((r) => [r.takes_per_class, r.train_clips,
        pct(r.test_top1), pct(r.test_top5)]), 7.9, 1.75, 4.8, [1.5, 1.2, 1.05, 1.05]);
    body(s, [
      `One example per word already gives ${pct(shot(1).test_top1)} top-1 across ${S.classes ?? 61} classes — chance is ${(100 / (S.classes || 61)).toFixed(1)}%.`,
      `Five takes per class reach ${pct(shot(5).test_top1)}; the curve is flat well before the full training set.`,
      "That is the practical argument for keypoints: a new word can be added from a handful of recordings, not hundreds.",
    ], 7.9, 4.3, 4.8, 12.5);
    s.addNotes("This is the number to quote when asked how much data the approach needs.");
  }

  // ---------------------------------------------------- 12. robustness
  {
    const s = slide();
    header(s, "Protocol B", "Robustness — did it learn the sign or the session?");
    const rb = img("robustness.png");
    if (rb) s.addImage({ path: rb, x: MX, y: 1.7, w: 7.2, h: 4.2, sizing: { type: "contain", w: 7.2, h: 4.2 } });
    body(s, [
      "The trained model is re-tested on the same clips after camera rotation, scaling, off-centring, speed changes, dropped frames, landmark noise, mirroring and occlusion of one hand.",
      rob("rotate_15") != null ? `Geometry is handled: ${pct(rob("rotate_15"))} under a 15° camera rotation, ${pct(rob("scale_20"))} under ±20% scale — the body-frame normalisation doing its job.` : "",
      rob("occlude_left_hand") != null ? `Occluding one hand costs the most (${pct(rob("occlude_left_hand"))}), which is expected: many ISL signs are two-handed and the information is genuinely gone.` : "",
      rob("speed_50") != null ? `Speed changes of ×0.5 and ×1.5 land at ${pct(rob("speed_50"))}, confirming the temporal resampling gives real tempo invariance.` : "",
    ].filter(Boolean), 8.1, 1.85, 4.6, 12);
    s.addNotes("Each perturbation maps to a preprocessing decision — say which one.");
  }

  // --------------------------------------- 13. augmentation ablation
  {
    const s = slide();
    header(s, "Protocol C", "Augmentation ablation — a negative result worth reporting");
    const ag = img("augmentation.png");
    if (ag) s.addImage({ path: ag, x: MX, y: 1.7, w: 7.0, h: 3.9, sizing: { type: "contain", w: 7.0, h: 3.9 } });
    body(s, [
      `Switching on the full augmentation stack made accuracy ${abs("full") != null && abs("none") != null ? "worse" : "worse"}: ${pct(abs("none"))} without it, ${pct(abs("full"))} with it.`,
      `Isolating the cause: mirror-only is ${pct(abs("mirror_only"))}, everything-except-mirror is ${pct(abs("no_mirror"))}. Mirroring alone accounts for the loss.`,
      "Why: one signer with a consistent dominant hand. Mirrored clips are inputs the test distribution never contains, and many ISL signs are not left-right symmetric.",
      `Geometric jitter and noise do help: ${pct(abs("geometric_only"))} versus ${pct(abs("none"))} without augmentation.`,
      "Lesson: an augmentation is only valid if it imitates variation that actually exists in deployment.",
    ], 7.9, 1.8, 4.8, 12.5);
    card(s, MX, 5.85, W - 2 * MX, 0.85, TINT);
    s.addText(`Seed variance check: the full model scores ${AB.full_training_seeds ? (AB.full_training_seeds.mean * 100).toFixed(1) : "—"}% ± ${AB.full_training_seeds ? (AB.full_training_seeds.stdev * 100).toFixed(1) : "—"} over 3 seeds — single-run differences below ~1.5 points are noise, not signal.`,
      { x: MX + 0.35, y: 5.85, w: W - 2 * MX - 0.7, h: 0.85, margin: 0, fontFace: "Calibri",
        fontSize: 13, color: DARK, valign: "middle", isTextBox: true });
    s.addNotes("A negative result, measured and explained, is stronger than an unverified claim.");
  }

  // ------------------------------------------- 13b. unseen signer
  if (US) {
    const s = slide();
    header(s, "Protocol D", "The real test: a signer the model has never seen");
    const usimg = img("unseen_signer.png");
    if (usimg) s.addImage({ path: usimg, x: MX, y: 1.72, w: 7.1, h: 4.6, sizing: { type: "contain", w: 7.1, h: 4.6 } });
    stat(s, 8.1, 1.72, 2.2, pct(US.top1), "top-1, new signer");
    stat(s, 10.4, 1.72, 2.2, pct(US.top5), "top-5, new signer");
    body(s, [
      `${US.n_clips} clips of ${US.n_words} words recorded on a different laptop webcam, by a signer who appears nowhere in the training data, classified against all ${S.classes ?? 61} classes (chance ${(100 / (S.classes || 61)).toFixed(1)}%).`,
      "Nothing about the recording matches the training set: different person, room, lighting, clothing, camera and signing style — only the sign itself carries over.",
      "This is the number that says whether the system works for a new user, and it is the honest ceiling on what the saturated benchmark could ever tell us.",
      "The gap to the in-dataset score is the signer-generalisation gap, and closing it needs multi-signer data — not a better model.",
    ], 8.1, 3.3, 4.6, 12);
    s.addNotes("State this number plainly, whatever it is. It is the most honest slide in the deck.");
  }

  // ------------------------------------------------ 14. error analysis
  {
    const s = slide();
    header(s, "Error analysis", "Where the model still confuses signs");
    const cm = img("confusion_matrix_k2.png");
    if (cm) s.addImage({ path: cm, x: MX, y: 1.7, w: 6.3, h: 4.7, sizing: { type: "contain", w: 6.3, h: 4.7 } });
    s.addText(`Analysed on the ${EA.takes_per_class ?? 2}-take model (${pct(EA.test_top1)}) — the full model makes no errors to study.`,
      { x: 7.2, y: 1.78, w: 5.5, h: 0.6, margin: 0, fontFace: "Calibri", fontSize: 13, italic: true,
        color: MUTED, isTextBox: true });
    tableRows(s, ["True sign", "Predicted as"], (EA.top_confusions || []).slice(0, 4)
      .map((c) => [c.true.replace(/_/g, " "), c.predicted.replace(/_/g, " ")]), 7.2, 2.45, 5.5, [2.75, 2.75]);
    body(s, [
      "Confusions concentrate on signs sharing a handshape and body location, differing only in a short movement or a facial marker the model never sees.",
      "The top-1 to top-5 gap shows these are near-misses inside a small confusable set, not random failures.",
      "Fixes: a face-landmark channel for non-manual markers, more takes for weak classes, and a two-stage classifier that resolves the cluster first.",
    ], 7.2, 4.35, 5.5, 12);
    s.addNotes("Name two confused pairs out loud; it shows you read your own matrix.");
  }

  // --------------------------------------------------------- 12. live demo
  {
    const s = slide();
    header(s, "Deployment", "Real-time recognition on a laptop");
    await flowRow(s, [["Webcam\nframe", fa.FaVideo], ["MediaPipe\nlandmarks", fa.FaHandPaper],
      ["Rolling 64-frame\nbuffer", fa.FaLayerGroup], ["BiLSTM\nevery 5 frames", fa.FaBrain],
      ["Confidence\ngate 0.6", fa.FaFilter], ["Text + speech", fa.FaVolumeUp]], 1.8, 1.15);
    body(s, [
      "The same normalisation code path runs live, so the training and inference distributions match exactly.",
      "A prediction is spoken only above the confidence threshold and only when it changes, so idle hand motion stays silent.",
      "MediaPipe landmarking is the bottleneck; the classifier head is sub-millisecond.",
      "Confirmed words accumulate into a sentence; space replays the whole sentence aloud, m mutes, c clears.",
      "No server and no upload — video never leaves the machine, which matters because this is face and hand data.",
    ], MX, 3.4, 7.1, 13.5);
    card(s, 8.1, 3.4, 4.6, 2.6, PANEL);
    s.addText("Ready to deploy", { x: 8.4, y: 3.6, w: 4.0, h: 0.3, margin: 0, fontFace: "Calibri",
      fontSize: 14, bold: true, color: ACCENT, isTextBox: true });
    s.addText("python -m isl.realtime \\\n  --ckpt models/bilstm_orig.pt \\\n  --speak", { x: 8.4, y: 4.0, w: 4.0, h: 0.95,
      margin: 0, fontFace: "Courier New", fontSize: 12, color: "D7E9E7", isTextBox: true });
    s.addText("Exporting to TFLite or Core ML puts the same pipeline on a phone — MediaPipe was built for mobile and the classifier is a couple of megabytes.",
      { x: 8.4, y: 5.0, w: 4.0, h: 0.85, margin: 0, fontFace: "Calibri", fontSize: 12, color: "C8DEDB", isTextBox: true });
    s.addNotes("Run the live demo here if the room has decent light.");
  }

  // ------------------------------------------- 12b. text + speech output
  {
    const s = slide();
    header(s, "Output stage", "From a prediction to text and speech");
    await flowRow(s, [["Prediction\nsoftmax over 61", null], ["Confidence gate\n≥ 0.60", null],
      ["Hold\n2 agreeing frames", null], ["De-duplicate\nno repeats", null],
      ["Text + speech\nsentence buffer", null]], 1.75, 1.0);
    const colW = (W - 2 * MX - 0.4) / 2;
    card(s, MX, 3.1, colW, 2.45);
    s.addText("TEXT — for the screen", { x: MX + 0.3, y: 3.3, w: colW - 0.6, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, ["Each confirmed word is appended to a sentence line at the bottom of the window",
      "The hearing person reads it, and the signer can see that they were understood correctly",
      "Confidence bar shows how close the current sign is to the speaking threshold"],
      MX + 0.3, 3.7, colW - 0.6, 12);
    card(s, MX + colW + 0.4, 3.1, colW, 2.45, PANEL);
    s.addText("SPEECH — for the room", { x: MX + colW + 0.7, y: 3.3, w: colW - 0.6, h: 0.3,
      margin: 0, fontFace: "Calibri", fontSize: 14, bold: true, color: ACCENT, isTextBox: true });
    s.addText([{ text: "Indian-English voice", options: { bold: true, color: ACCENT, breakLine: true } },
      { text: "An ISL translator should not sound American — the system picks an en_IN voice (Aman / Rishi / Veena) when one is installed.", options: { color: "D7E9E7", breakLine: true } },
      { text: "macOS say → pyttsx3 → print", options: { bold: true, color: ACCENT, breakLine: true } },
      { text: "Three backends tried in order, so the pipeline never crashes on a machine with no speech engine.", options: { color: "D7E9E7" } }],
      { x: MX + colW + 0.7, y: 3.7, w: colW - 0.6, h: 1.7, margin: 0, fontFace: "Calibri",
        fontSize: 12, valign: "top", isTextBox: true });
    card(s, MX, 5.75, W - 2 * MX, 0.95, TINT);
    s.addText([{ text: "Why the gate and the hold:  ", options: { bold: true, color: ACC_D } },
      { text: "the model re-predicts every 5 frames, so without them a single sign would be spoken a dozen times and every transition between signs would be read aloud as a word. Speech runs on its own thread with a 1.2 s repeat guard, so recognition never stalls waiting for the voice.", options: { color: DARK } }],
      { x: MX + 0.35, y: 5.75, w: W - 2 * MX - 0.7, h: 0.95, margin: 0, fontFace: "Calibri",
        fontSize: 12.5, valign: "middle", isTextBox: true });
    s.addNotes("Demo it: sign two words, then press space to replay the whole sentence aloud.");
  }

  // -------------------------------------------------- 13. limits & future
  {
    const s = slide();
    header(s, "Honest assessment", "Limitations and what comes next");
    const colW = (W - 2 * MX - 0.4) / 2;
    card(s, MX, 1.75, colW, 4.3, "F3F4F6");
    s.addText("Limitations", { x: MX + 0.3, y: 1.95, w: colW - 0.6, h: 0.3, margin: 0, fontFace: "Calibri",
      fontSize: 15, bold: true, color: MUTED, isTextBox: true });
    body(s, ["Isolated words only — no continuous sentences",
      `Vocabulary limited to the ${S.classes ?? 61} classes in the dataset`,
      "No facial / non-manual markers yet, which disambiguate some sign pairs",
      "Signer-mixed split: performance on a completely new signer will be lower",
      "Degrades in very low light or heavy occlusion, where MediaPipe itself fails"], MX + 0.3, 2.4, colW - 0.6, 12.5);
    card(s, MX + colW + 0.4, 1.75, colW, 4.3);
    s.addText("Next steps", { x: MX + colW + 0.7, y: 1.95, w: colW - 0.6, h: 0.3, margin: 0, fontFace: "Calibri",
      fontSize: 15, bold: true, color: ACC_D, isTextBox: true });
    body(s, ["Signer-independent evaluation as the headline metric",
      "Add face landmarks for non-manual markers",
      "Continuous signing with CTC loss — no manual clip boundaries",
      "Sentence-level output with a small language model to fix word order",
      "On-device mobile build (TFLite / Core ML) for genuine field use"], MX + colW + 0.7, 2.4, colW - 0.6, 12.5);
    card(s, MX, 6.25, W - 2 * MX, 0.75, PANEL);
    s.addText("An assistive aid, not a replacement for a human interpreter — in medical, legal or educational settings errors carry real cost.",
      { x: MX + 0.35, y: 6.25, w: W - 2 * MX - 0.7, h: 0.75, margin: 0, fontFace: "Calibri", fontSize: 13,
        italic: true, color: "D7E9E7", valign: "middle", isTextBox: true });
    s.addNotes("Ending on limitations plus ethics reads as maturity, not weakness.");
  }

  // ------------------------------------------------------------ 14. close
  {
    const s = slide();
    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: PANEL }, line: { color: PANEL } });
    s.addText("What we built", { x: MX, y: 1.4, w: 9, h: 0.4, margin: 0, fontFace: "Calibri", fontSize: 12,
      bold: true, color: ACCENT, charSpacing: 3, isTextBox: true });
    const seedMean = AB.full_training_seeds ? pct(AB.full_training_seeds.mean) : pct(S.test_top1);
    const seedSd = AB.full_training_seeds ? (AB.full_training_seeds.stdev * 100).toFixed(1) : "—";
    s.addText(`A reproducible ISL recognition pipeline:\n${S.clips ?? 1210} clips → MediaPipe keypoints → ${S.classes ?? 61}-way BiLSTM`,
      { x: MX, y: 1.85, w: 11.9, h: 1.35, margin: 0, fontFace: "Cambria", fontSize: 30, bold: true,
        color: "FFFFFF", lineSpacingMultiple: 1.1, isTextBox: true });
    s.addText(`${seedMean} ± ${seedSd} on the standard split  ·  ${pct(shot(5).test_top1)} from five takes per word  ·  ${pct(shot(1).test_top1)} from a single example`,
      { x: MX, y: 3.35, w: 11.9, h: 0.45, margin: 0, fontFace: "Calibri", fontSize: 16,
        color: ACCENT, isTextBox: true });
    const items = [["Signer-invariant keypoint preprocessing", fa.FaRulerCombined],
      ["Leakage-safe, group-aware data splits", fa.FaShieldAlt],
      ["Diagnosed a saturated benchmark, then built harder protocols", fa.FaBalanceScale],
      ["Live webcam demo with speech output", fa.FaVolumeUp]];
    for (let i = 0; i < items.length; i++) {
      const [t, ic] = items[i];
      const y = 4.15 + Math.floor(i / 2) * 1.05, x = MX + (i % 2) * 6.2;
      s.addShape(pres.shapes.OVAL, { x, y, w: 0.6, h: 0.6, fill: { color: "14504F" }, line: { color: ACCENT, width: 1 } });
      s.addImage({ data: await icon(ic, ACCENT), x: x + 0.16, y: y + 0.16, w: 0.28, h: 0.28 });
      s.addText(t, { x: x + 0.8, y, w: 5.2, h: 0.6, margin: 0, fontFace: "Calibri", fontSize: 14,
        color: "D7E9E7", valign: "middle", isTextBox: true });
    }
    s.addText("Thank you  ·  questions welcome", { x: MX, y: 6.35, w: 9, h: 0.4, margin: 0, fontFace: "Calibri",
      fontSize: 14, italic: true, color: ACCENT, isTextBox: true });
    s.addNotes("Close by offering the live demo.");
  }

  const out = process.argv[2] || "ISL_Project.pptx";
  await pres.writeFile({ fileName: out });
  console.log("wrote", out);
}

build().catch((e) => { console.error(e); process.exit(1); });
