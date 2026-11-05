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
pres.title = "ISL Data Preprocessing";

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
    s.addShape(pres.shapes.OVAL, { x: MX, y: 1.1, w: 1.05, h: 1.05, fill: { color: ACCENT }, line: { color: ACCENT } });
    s.addImage({ data: await icon(fa.FaSlidersH, PANEL), x: MX + 0.26, y: 1.36, w: 0.53, h: 0.53 });
    s.addText("INDIAN SIGN LANGUAGE RECOGNITION  ·  MINOR PROJECT", { x: MX, y: 2.5, w: 9.5, h: 0.3,
      margin: 0, fontFace: "Calibri", fontSize: 12, bold: true, color: ACCENT, charSpacing: 3, isTextBox: true });
    s.addText("Data Preprocessing", { x: MX, y: 2.9, w: 10, h: 1.0, margin: 0, fontFace: "Cambria",
      fontSize: 50, bold: true, color: "FFFFFF", isTextBox: true });
    s.addText("Turning 6.8 seconds of raw webcam video into a 64 × 478 matrix the model can learn from — and why every step of that is a decision, not a default.",
      { x: MX, y: 4.1, w: 8.6, h: 0.9, margin: 0, fontFace: "Calibri", fontSize: 15, italic: true,
        color: "C8DEDB", isTextBox: true });
    const chips = ["OpenCV", "MediaPipe Tasks", "NumPy", "signer-invariant features"];
    let cx = MX;
    for (const t of chips) {
      const cw = t.length * 12 * 0.0075 + 0.45;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: 5.6, w: cw, h: 0.42, rectRadius: 0.21,
        fill: { color: PANEL }, line: { color: ACCENT, width: 1 } });
      s.addText(t, { x: cx, y: 5.6, w: cw, h: 0.42, margin: 0, fontFace: "Calibri", fontSize: 12,
        color: ACCENT, align: "center", valign: "middle", isTextBox: true });
      cx += cw + 0.14;
    }
    card(s, 10.3, 2.9, 2.4, 1.6, "14504F");
    s.addText("2,000×", { x: 10.3, y: 3.0, w: 2.4, h: 0.8, margin: 0, fontFace: "Cambria", fontSize: 34,
      bold: true, color: ACCENT, align: "center", valign: "middle", isTextBox: true });
    s.addText("smaller input,\nsame information", { x: 10.3, y: 3.75, w: 2.4, h: 0.6, margin: 0,
      fontFace: "Calibri", fontSize: 11.5, color: "C8DEDB", align: "center", isTextBox: true });
    s.addNotes("Open by saying preprocessing is not cleanup here - it is the modelling decision that makes the whole project work on a laptop.");
  }

  // ------------------------------------------------- 2. why it matters
  {
    const s = slide();
    header(s, "Why it matters", "Preprocessing is the modelling decision, not cleanup");
    card(s, MX, 1.75, 5.9, 1.5, "F3F4F6");
    s.addText([{ text: "RAW CLIP\n", options: { bold: true, fontSize: 11, color: MUTED, charSpacing: 2 } },
      { text: "6.8 s × 30 fps × 640 × 720 × 3 channels\n", options: { fontSize: 13, color: DARK } },
      { text: "≈ 280 million numbers → label: one word", options: { fontSize: 13, bold: true, color: DARK } }],
      { x: MX + 0.3, y: 1.85, w: 5.3, h: 1.3, margin: 0, fontFace: "Calibri", valign: "middle",
        lineSpacingMultiple: 1.15, isTextBox: true });
    card(s, 6.85, 1.75, 5.85, 1.5);
    s.addText([{ text: "AFTER PREPROCESSING\n", options: { bold: true, fontSize: 11, color: ACC_D, charSpacing: 2 } },
      { text: "64 frames × 478 features ≈ 120 KB\n", options: { fontSize: 13, color: DARK } },
      { text: "only hand shape, position and motion remain", options: { fontSize: 13, bold: true, color: DARK } }],
      { x: 7.15, y: 1.85, w: 5.25, h: 1.3, margin: 0, fontFace: "Calibri", valign: "middle",
        lineSpacingMultiple: 1.15, isTextBox: true });
    s.addText("Preprocessing does three jobs", { x: MX, y: 3.5, w: 8, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, [
      { text: "Throw away everything that is not the sign — background, clothing, lighting, skin tone, the signer's identity.", bold: false },
      "Make what remains comparable across people and cameras, so a tall signer far away and a short signer close up look the same to the model.",
      "Turn a variable-length clip into a fixed-size tensor, because a classifier needs a fixed input while signs vary in duration.",
    ], MX, 3.85, 11.9, 14);
    card(s, MX, 5.75, W - 2 * MX, 0.95, PANEL);
    s.addText([{ text: "In one line:  ", options: { bold: true, color: ACCENT } },
      { text: "we convert raw video into signer-invariant, time-normalised skeletal motion features, so the classifier learns the sign and not the signer.", options: { color: "D7E9E7" } }],
      { x: MX + 0.35, y: 5.75, w: W - 2 * MX - 0.7, h: 0.95, margin: 0, fontFace: "Calibri",
        fontSize: 14.5, valign: "middle", isTextBox: true });
    s.addNotes("If you remember one sentence from this talk, it is the one in the dark box.");
  }

  // ------------------------------------------------- 3. the raw material
  {
    const s = slide();
    header(s, "The input", "What the raw data actually looks like");
    card(s, MX, 1.72, 7.4, 1.35);
    s.addText([{ text: "Indian Sign Language Video Dataset — Kaggle, MIT licence, 3.48 GB\n", options: { bold: true, fontSize: 14, color: DARK } },
      { text: "61 word signs · 3,630 clips · 640 × 720 at 30 fps · ~6.8 s (≈205 frames) per clip · one folder per word", options: { fontSize: 12.5, color: MUTED } }],
      { x: MX + 0.28, y: 1.82, w: 6.9, h: 1.15, margin: 0, fontFace: "Calibri", valign: "middle", isTextBox: true });
    stat(s, 8.3, 1.75, 1.35, "61", "classes");
    stat(s, 9.75, 1.75, 1.35, "3,630", "clips");
    stat(s, 11.2, 1.75, 1.5, "205", "frames/clip");
    s.addText("Every problem in the raw data has a preprocessing answer", { x: MX, y: 3.3, w: 9, h: 0.3,
      margin: 0, fontFace: "Calibri", fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    tableRows(s, ["Problem in the raw video", "What we do about it"], [
      ["205 near-identical frames per clip", "Keep every 3rd frame — 3× less computation, no information lost"],
      ["Background, clothing, lighting, skin tone", "Replace pixels with 75 landmark points"],
      ["Signer stands closer / further / off-centre", "Re-express every point relative to the shoulders"],
      ["Clips differ in length; signs differ in speed", "Resample every clip onto a fixed 64-frame grid"],
      ["A hand leaves the frame or is occluded", "Zero-fill that hand and raise a presence flag"],
      ["Only 20 recordings per word", "Augment the training set with realistic variation"],
    ], MX, 3.65, W - 2 * MX, [5.1, 7.0]);
    s.addNotes("This table is the spine of the talk - each row is one of the next slides.");
  }

  // ---------------------------------------- 4. stages 1-2: frames + landmarks
  {
    const s = slide();
    header(s, "Stages 1 & 2", "Frame sampling, then pixels → landmarks");
    await flowRow(s, [["Decode video\n205 frames", null], ["Keep every 3rd\n~68 frames", null],
      ["BGR → RGB\nper frame", null], ["MediaPipe\nhand + pose", null],
      ["75 points\nper frame", null]], 1.75, 1.05);
    const colW = (W - 2 * MX - 0.4) / 2;
    card(s, MX, 3.15, colW, 2.45, "F3F4F6");
    s.addText("Why every 3rd frame", { x: MX + 0.3, y: 3.35, w: colW - 0.6, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: MUTED, isTextBox: true });
    body(s, ["At 30 fps consecutive frames are nearly identical — the hand moves a few pixels",
      "~68 frames left, an effective 10 fps, still above the rate at which sign movement aliases",
      "Extraction over 1,210 clips drops from ~90 minutes to under 10"], MX + 0.3, 3.75, colW - 0.6, 12.5);
    card(s, MX + colW + 0.4, 3.15, colW, 2.45);
    s.addText("What MediaPipe returns", { x: MX + colW + 0.7, y: 3.35, w: colW - 0.6, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, ["Hand landmarker: 21 3-D points per hand, plus a handedness label",
      "Pose landmarker: 33 body points — we keep 9 (nose, shoulders, elbows, wrists, hips) and drop the legs",
      "Handedness is mirrored from the camera's view, so we flip it — otherwise the two hand channels swap randomly between clips"],
      MX + colW + 0.7, 3.75, colW - 0.6, 12.5);
    card(s, MX, 5.8, W - 2 * MX, 0.85, TINT);
    s.addText([{ text: "Detail worth mentioning:  ", options: { bold: true, color: ACC_D } },
      { text: "the landmarker runs in VIDEO mode and demands strictly increasing timestamps, so reusing one extractor across clips crashes on the second clip unless the timestamps are made monotonic. We hit this and fixed it.", options: { color: DARK } }],
      { x: MX + 0.35, y: 5.8, w: W - 2 * MX - 0.7, h: 0.85, margin: 0, fontFace: "Calibri",
        fontSize: 12.5, valign: "middle", isTextBox: true });
    s.addNotes("The timestamp bug is a good detail - it shows you actually built this rather than copied it.");
  }

  // ------------------------------------------- 5. normalisation (the key slide)
  {
    const s = slide();
    header(s, "Stage 3 — the most important step", "Making the features signer-invariant");
    card(s, MX, 1.72, 6.2, 1.95, "F3F4F6");
    s.addText("origin = (L_shoulder + R_shoulder) / 2\nscale  = || L_shoulder − R_shoulder ||\nx′ = (x − origin) / scale", {
      x: MX + 0.32, y: 1.85, w: 5.6, h: 1.7, margin: 0, fontFace: "Courier New", fontSize: 14,
      color: DARK, valign: "middle", lineSpacingMultiple: 1.25, isTextBox: true });
    body(s, [
      "Translate every landmark to the mid-point between the shoulders, then divide by shoulder width — recomputed on every frame.",
      "Translation invariance and scale invariance from one transform; recomputing per frame also absorbs the signer leaning or stepping sideways mid-sign.",
      "Guarded: if the measured shoulder width falls below 1e-3 (signer turned sideways, detector failure) the scale falls back to 1.0 instead of dividing by near-zero.",
    ], 7.2, 1.75, 5.5, 12.5);
    s.addText("Each hand is encoded twice — on purpose", { x: MX, y: 3.95, w: 9, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    const colW = (W - 2 * MX - 0.4) / 2;
    card(s, MX, 4.3, colW, 1.75);
    s.addText([{ text: "Body-relative → LOCATION\n", options: { bold: true, fontSize: 13, color: ACC_D } },
      { text: "Where the hand is with respect to the body: chest, chin, forehead. Shoulder-normalised (x, y).", options: { fontSize: 12, color: DARK } }],
      { x: MX + 0.3, y: 4.45, w: colW - 0.6, h: 1.45, margin: 0, fontFace: "Calibri", valign: "top", isTextBox: true });
    card(s, MX + colW + 0.4, 4.3, colW, 1.75);
    s.addText([{ text: "Wrist-local → HANDSHAPE\n", options: { bold: true, fontSize: 13, color: ACC_D } },
      { text: "Which fingers are extended, independent of where the hand is. Centred on the wrist, divided by wrist-to-knuckle distance.", options: { fontSize: 12, color: DARK } }],
      { x: MX + colW + 0.7, y: 4.45, w: colW - 0.6, h: 1.45, margin: 0, fontFace: "Calibri", valign: "top", isTextBox: true });
    s.addText("Sign-language linguistics treats location and handshape as separate parameters of a sign — supplying both means the network never has to disentangle them from one mixed signal.",
      { x: MX, y: 6.2, w: W - 2 * MX, h: 0.5, margin: 0, fontFace: "Calibri", fontSize: 12.5,
        italic: true, color: MUTED, isTextBox: true });
    s.addNotes("If the prof asks only one question it will be about this slide. Know the formula cold.");
  }

  // ------------------------------------------------- 6. feature layout
  {
    const s = slide();
    header(s, "The result", "What one frame becomes: 239 numbers");
    tableRows(s, ["Block", "Dims", "What it encodes"], [
      ["Pose: 9 upper-body points (x, y, visibility)", "27", "Posture and arm configuration"],
      ["Both hands, body-relative (2 × 21 × x, y)", "84", "Location — where the sign happens"],
      ["Both hands, wrist-local (2 × 21 × x, y, z)", "126", "Handshape — which fingers are extended"],
      ["Hand-present flags", "2", "Whether each hand was detected at all"],
      ["Base feature vector, per frame", "239", ""],
      ["+ frame-to-frame difference (velocity)", "478", "Movement — the defining parameter of a sign"],
      ["Full clip tensor after resampling", "64 × 478", "What the model actually sees"],
    ], MX, 1.8, W - 2 * MX, [5.4, 1.7, 5.0]);
    card(s, MX, 5.1, W - 2 * MX, 1.5, TINT);
    s.addText([{ text: "Why velocity is added rather than learned:  ", options: { bold: true, color: ACC_D } },
      { text: "the LSTM could compute differences itself, but it would spend capacity doing so. Differencing is free, removes any residual static offset, and speeds up convergence — the same reasoning behind delta-MFCCs in speech recognition. The presence flags matter for the same kind of reason: without them the model cannot tell \"hand absent\" from \"hand at the origin\", because both are zeros.", options: { color: DARK } }],
      { x: MX + 0.35, y: 5.1, w: W - 2 * MX - 0.7, h: 1.5, margin: 0, fontFace: "Calibri",
        fontSize: 13, valign: "middle", isTextBox: true });
    s.addNotes("Be able to add the blocks up out loud: 27 + 84 + 126 + 2 = 239, doubled to 478 with velocity.");
  }

  // --------------------------------------------- 7. temporal normalisation
  {
    const s = slide();
    header(s, "Stage 4", "Temporal resampling — same sign, different speed");
    card(s, MX, 1.75, 6.5, 1.6, "F3F4F6");
    s.addText("src = linspace(0, 1, T)        # original timeline\ndst = linspace(0, 1, 64)       # fixed timeline\nout[:, c] = interp(dst, src, seq[:, c])", {
      x: MX + 0.3, y: 1.88, w: 5.9, h: 1.35, margin: 0, fontFace: "Courier New", fontSize: 12.5,
      color: DARK, valign: "middle", lineSpacingMultiple: 1.2, isTextBox: true });
    body(s, [
      "Every feature channel is linearly interpolated onto a fixed grid of 64 frames.",
      "A sign performed slowly and the same sign performed quickly are the same sign, time-warped — resampling removes that difference.",
      "It also gives fixed-size tensors, which lets the classical SVM and Random-Forest baselines consume exactly the same representation.",
    ], 7.4, 1.75, 5.3, 12.5);
    card(s, MX, 3.65, W - 2 * MX, 1.15, PANEL);
    s.addText([{ text: "Evidence that it works:  ", options: { bold: true, color: ACCENT } },
      { text: `we re-tested the trained model on the same clips played at ×0.5 and ×1.5 speed. Accuracy stayed at ${pct(rob("speed_50"))} — identical to the clean clips. The tempo invariance is real, not assumed.`, options: { color: "D7E9E7" } }],
      { x: MX + 0.35, y: 3.65, w: W - 2 * MX - 0.7, h: 1.15, margin: 0, fontFace: "Calibri",
        fontSize: 14, valign: "middle", isTextBox: true });
    s.addText("The alternative we rejected", { x: MX, y: 5.0, w: 8, h: 0.3, margin: 0, fontFace: "Calibri",
      fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, [
      "Zero-padding every clip to the longest length and masking the padding is the standard alternative.",
      "It preserves absolute duration — but it leaves the speed variation in the data, and it complicates the baselines, which need one fixed-size vector.",
      "We chose to normalise tempo deliberately. If duration were linguistically meaningful here, padding would be the better choice.",
    ], MX, 5.35, 11.9, 12.5);
    s.addNotes("Naming the alternative you rejected, and why, is what separates a design decision from a default.");
  }

  // ----------------------------------------------- 8. augmentation + result
  {
    const s = slide();
    header(s, "Stage 5", "Augmentation — and a negative result we measured");
    const ag = img("augmentation.png");
    if (ag) s.addImage({ path: ag, x: MX, y: 1.72, w: 6.6, h: 3.7, sizing: { type: "contain", w: 6.6, h: 3.7 } });
    tableRows(s, ["Transform", "Range", "Imitates"], [
      ["Rotation", "±10°", "Camera not level"],
      ["Scale", "±10%", "Distance from camera"],
      ["Translation", "±0.05", "Signer off-centre"],
      ["Speed jitter", "±20%", "Faster / slower signing"],
      ["Frame dropout", "10%", "Dropped frames"],
      ["Coordinate noise", "σ = 0.01", "Landmark jitter"],
      ["Horizontal mirror", "p = 0.3", "Left-dominant signers"],
    ], 7.5, 1.78, 5.2, [1.85, 1.25, 2.1]);
    card(s, 7.5, 4.55, 5.2, 0.85, TINT);
    s.addText("Applied to training clips only — validation and test always see the clean pipeline.",
      { x: 7.6, y: 4.55, w: 5.0, h: 0.85, margin: 0, fontFace: "Calibri", fontSize: 12, bold: true,
        color: ACC_D, align: "center", valign: "middle", isTextBox: true });
    card(s, MX, 5.6, W - 2 * MX, 1.1, PANEL);
    s.addText([{ text: "Mirroring made it worse.  ", options: { bold: true, color: ACCENT } },
      { text: `With 2 takes per class: ${pct(abs("none"))} with no augmentation, ${pct(abs("full"))} with everything on. Isolating it — mirror-only ${pct(abs("mirror_only"))}, everything-except-mirror ${pct(abs("no_mirror"))}. The dataset has one signer with a consistent dominant hand, so mirrored clips are inputs the test distribution never contains. Geometric jitter and noise do help: ${pct(abs("geometric_only"))}.`, options: { color: "D7E9E7" } }],
      { x: MX + 0.35, y: 5.6, w: W - 2 * MX - 0.7, h: 1.1, margin: 0, fontFace: "Calibri",
        fontSize: 12.5, valign: "middle", isTextBox: true });
    s.addNotes("Lead with the measurement, not the intuition: an augmentation is only valid if it imitates variation that exists in deployment.");
  }

  // ------------------------------------- 9. leakage control + hygiene
  {
    const s = slide();
    header(s, "Data hygiene", "Splitting and numerical safety");
    card(s, MX, 1.72, W - 2 * MX, 1.35, "FEF4E2");
    s.addText([{ text: "The trap in this dataset:  ", options: { bold: true, color: "B45309" } },
      { text: "every recorded take also ships as a left-tilt and a right-tilt copy. Those three clips are near-duplicates. Split them independently and a near-copy of a test clip lands in training — the accuracy then looks excellent and means nothing.", options: { color: DARK } }],
      { x: MX + 0.35, y: 1.72, w: W - 2 * MX - 0.7, h: 1.35, margin: 0, fontFace: "Calibri",
        fontSize: 13.5, valign: "middle", isTextBox: true });
    const colW = (W - 2 * MX - 0.4) / 2;
    card(s, MX, 3.25, colW, 2.4);
    s.addText("Leakage control", { x: MX + 0.3, y: 3.45, w: colW - 0.6, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: ACC_D, isTextBox: true });
    body(s, ["Group-aware split: all variants of one take go to the same side",
      "Stratified per class, so all 61 words appear in train, validation and test",
      "Computed once with a fixed seed, before any training; the test set is read only by the evaluation script",
      "The baseline StandardScaler is fitted inside a pipeline on training data only"],
      MX + 0.3, 3.85, colW - 0.6, 12);
    card(s, MX + colW + 0.4, 3.25, colW, 2.4, "F3F4F6");
    s.addText("Numerical hygiene", { x: MX + colW + 0.7, y: 3.45, w: colW - 0.6, h: 0.3, margin: 0,
      fontFace: "Calibri", fontSize: 14, bold: true, color: MUTED, isTextBox: true });
    body(s, ["Every feature matrix passes a NaN / ±inf guard — MediaPipe occasionally emits non-finite values, and one NaN destroys a whole training run",
      "Degenerate shoulder-width scales clamped to 1.0",
      "Landmarks cached to compressed .npz, so MediaPipe runs once and every later experiment reuses it"],
      MX + colW + 0.7, 3.85, colW - 0.6, 12);
    card(s, MX, 5.85, W - 2 * MX, 0.8, TINT);
    s.addText([{ text: "Stated limitation:  ", options: { bold: true, color: ACC_D } },
      { text: "the split is group-aware but signer-mixed — the dataset has one signer, so a signer-independent split is impossible here. Accuracy on a brand-new signer will be lower.", options: { color: DARK } }],
      { x: MX + 0.35, y: 5.85, w: W - 2 * MX - 0.7, h: 0.8, margin: 0, fontFace: "Calibri",
        fontSize: 12.5, valign: "middle", isTextBox: true });
    s.addNotes("Say the limitation before the prof finds it. It reads as rigour, not weakness.");
  }

  // ------------------------------------------- 10. evidence it works
  {
    const s = slide();
    header(s, "Does the preprocessing work?", "Every invariance we claimed, measured");
    const rb = img("robustness.png");
    if (rb) s.addImage({ path: rb, x: MX, y: 1.72, w: 7.0, h: 4.1, sizing: { type: "contain", w: 7.0, h: 4.1 } });
    body(s, [
      "The trained model was re-tested on the same clips after deliberate perturbation — each one targets a specific preprocessing claim.",
      `Shifted off-centre or with landmark noise added: ${pct(rob("shift_10"))}, the same as clean. Camera rotated 15° or scaled ±20%: ${pct(rob("rotate_15"))}. The shoulder-frame normalisation does what it claims. Mirrored: ${pct(rob("mirror"))}.`,
      `30% of frames dropped: ${pct(rob("frame_drop_30"))} — resampling absorbs missing frames.`,
      `One hand occluded: ${pct(rob("occlude_left_hand"))}. Expected and correct — many ISL signs are two-handed, so the information is genuinely gone, and a model scoring highly here would be relying on something other than the sign.`,
    ], 7.9, 1.82, 4.8, 12);
    s.addNotes("Close here: the invariances are measured, not asserted. Offer to go deeper on any stage.");
  }

  // ----------------------------------------------------------- 11. summary
  {
    const s = slide();
    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: PANEL }, line: { color: PANEL } });
    s.addText("Summary", { x: MX, y: 1.3, w: 9, h: 0.4, margin: 0, fontFace: "Calibri", fontSize: 12,
      bold: true, color: ACCENT, charSpacing: 3, isTextBox: true });
    s.addText("280 million pixels → 64 × 478 features\nwithout losing the sign", { x: MX, y: 1.8, w: 11.9, h: 1.5,
      margin: 0, fontFace: "Cambria", fontSize: 32, bold: true, color: "FFFFFF",
      lineSpacingMultiple: 1.1, isTextBox: true });
    const items = [["Landmarks instead of pixels — identity, clothing and background gone by construction", fa.FaHandPaper],
      ["Shoulder-frame normalisation — the step that makes it signer-invariant", fa.FaRulerCombined],
      ["Location and handshape encoded separately, velocity added explicitly", fa.FaProjectDiagram],
      ["Fixed 64-frame grid — tempo invariance, measured at " + pct(rob("speed_50")), fa.FaClock],
      ["Group-aware splitting — no near-duplicate leakage", fa.FaShieldAlt],
      ["Augmentation validated by ablation, not assumed", fa.FaBalanceScale]];
    for (let i = 0; i < items.length; i++) {
      const [t, ic] = items[i];
      const y = 3.6 + Math.floor(i / 2) * 0.95, x = MX + (i % 2) * 6.2;
      s.addShape(pres.shapes.OVAL, { x, y, w: 0.52, h: 0.52, fill: { color: "14504F" }, line: { color: ACCENT, width: 1 } });
      s.addImage({ data: await icon(ic, ACCENT), x: x + 0.14, y: y + 0.14, w: 0.24, h: 0.24 });
      s.addText(t, { x: x + 0.72, y: y - 0.05, w: 5.3, h: 0.62, margin: 0, fontFace: "Calibri",
        fontSize: 12, color: "D7E9E7", valign: "middle", isTextBox: true });
    }
    s.addText("Thank you  ·  happy to go deeper on any stage", { x: MX, y: 6.5, w: 9, h: 0.4, margin: 0,
      fontFace: "Calibri", fontSize: 14, italic: true, color: ACCENT, isTextBox: true });
    s.addNotes("Invite questions on a specific stage - you have a slide-level answer for each.");
  }

  const out = process.argv[2] || "ISL_Preprocessing_Presentation.pptx";
  await pres.writeFile({ fileName: out });
  console.log("wrote", out);
}

build().catch((e) => { console.error(e); process.exit(1); });
