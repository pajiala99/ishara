const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow,
        TableCell, WidthType, ShadingType, BorderStyle, LevelFormat } = require("docx");

const TEAL = "0F766E", DARK = "1F2937", GREY = "4B5563", PAGE_W = 9360;
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, color: TEAL })] });
const p = (t, o = {}) => new Paragraph({ spacing: { after: 120, line: 290 }, ...o,
  children: Array.isArray(t) ? t : [new TextRun(t)] });
const b = (t) => new TextRun({ text: t, bold: true });
const bullet = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(t) ? t : [new TextRun(t)] });
const say = (t) => new Paragraph({ spacing: { after: 150, line: 300 }, indent: { left: 260 },
  children: [new TextRun({ text: t, italics: true, color: "1F2937" })] });

function slideHead(n, title, secs) {
  return new Paragraph({ spacing: { before: 280, after: 80 }, children: [
    new TextRun({ text: `Slide ${n} · `, bold: true, color: TEAL }),
    new TextRun({ text: title, bold: true, color: DARK }),
    new TextRun({ text: `   (${secs})`, color: GREY, size: 19 }),
  ]});
}

function table(headers, rows, widths) {
  const cell = (txt, o = {}) => new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: "E6F4F1" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ children: [new TextRun({ text: txt, bold: !!o.head, size: 20,
      color: o.head ? TEAL : DARK })] })],
  });
  return new Table({
    columnWidths: widths, width: { size: PAGE_W, type: WidthType.DXA },
    borders: ["top","bottom","left","right","insideHorizontal","insideVertical"]
      .reduce((a, k) => (a[k] = { style: BorderStyle.SINGLE, size: 2, color: "D1D5DB" }, a), {}),
    rows: [new TableRow({ tableHeader: true, children: headers.map((hd, i) => cell(hd, { head: true, w: widths[i] })) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, { w: widths[i] })) }))],
  });
}

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Preprocessing Talk Track", bold: true, size: 44, font: "Cambria", color: TEAL })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 280 },
    children: [new TextRun({ text: "ISL_Preprocessing_Presentation.pptx  ·  11 slides  ·  about 8 minutes", size: 24, color: GREY })] }),

  h1("Before you start"),
  bullet([b("The one sentence to land: "), new TextRun("we convert raw video into signer-invariant, time-normalised skeletal motion features, so the classifier learns the sign and not the signer.")]),
  bullet([b("The frame to hold: "), new TextRun("preprocessing here is not data cleaning, it is the modelling decision. Every step is a choice with an alternative you rejected for a reason.")]),
  bullet([b("If you are cut to 3 minutes: "), new TextRun("slides 2, 5 and 8 — why it matters, the normalisation maths, and the augmentation ablation. Those three carry the argument on their own.")]),
  bullet([b("Speaker notes are already in the .pptx"), new TextRun(" — open Presenter View and you will see a one-line reminder per slide.")]),

  h1("Timing"),
  table(["Slide", "Time", "Job it does"], [
    ["1 · Title", "15 s", "Name the topic and the 2,000× compression hook"],
    ["2 · Why it matters", "60 s", "Reframe preprocessing as the modelling decision"],
    ["3 · The input", "60 s", "Dataset facts and the problem→answer table"],
    ["4 · Stages 1 & 2", "60 s", "Frame sampling and MediaPipe landmarks"],
    ["5 · Stage 3", "90 s", "Shoulder-frame normalisation — the core slide"],
    ["6 · Feature layout", "60 s", "239 → 478 numbers, and why velocity is explicit"],
    ["7 · Stage 4", "60 s", "Temporal resampling, with the alternative rejected"],
    ["8 · Stage 5", "75 s", "Augmentation table and the mirroring negative result"],
    ["9 · Data hygiene", "60 s", "Leakage control and numerical safety"],
    ["10 · Evidence", "60 s", "Every invariance measured, not asserted"],
    ["11 · Summary", "20 s", "Six takeaways, invite questions"],
  ], [1900, 1100, 6360]),

  h1("What to say, slide by slide"),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Italic text is roughly what to say out loud. Do not read it word for word — it is there so you know the shape of each slide.", color: GREY, size: 20 })] }),

  slideHead(1, "Title", "15 s"),
  say("\"This is the data preprocessing half of our Indian Sign Language recognition project. The short version: a clip arrives as roughly 280 million pixel values and leaves as 64 by 478 numbers — about two thousand times smaller — with everything that identifies the sign still in it.\""),

  slideHead(2, "Preprocessing is the modelling decision, not cleanup", "60 s"),
  say("\"A clip is 6.8 seconds at 30 frames a second, 640 by 720, three colour channels — about 280 million numbers, and the label attached to all of that is a single word. If you hand that tensor to a network, almost all of its capacity goes on learning to ignore the shirt, the wall, the lighting and where the person happened to stand.\""),
  say("\"So preprocessing does three jobs here. It throws away everything that is not the sign. It makes what is left comparable across people and cameras. And it turns a variable-length clip into a fixed-size tensor. In one line: signer-invariant, time-normalised skeletal motion features — so the classifier learns the sign, not the signer.\""),
  p([b("Expect: "), new TextRun("\"why not just use a CNN on the video?\" Answer on slide 2 — data efficiency, invariance, and cost. We have 20 recordings per word; a 3D-CNN on pixels needs hundreds.")]),

  slideHead(3, "What the raw data actually looks like", "60 s"),
  say("\"The dataset is a public Indian Sign Language video set from Kaggle, MIT licensed, 3.48 gigabytes: 61 word signs, 3,630 clips, 640 by 720 at 30 fps, about 205 frames each, one folder per word.\""),
  say("\"This table is the rest of my talk. Every row is a real problem in the raw video, and the right-hand column is the preprocessing step that answers it. I'll walk the rows in order.\""),

  slideHead(4, "Frame sampling, then pixels to landmarks", "60 s"),
  say("\"First, we keep every third frame. At 30 fps consecutive frames are nearly identical — the hand has moved a few pixels — so two thirds of the work buys almost nothing. That leaves about 68 frames, an effective 10 fps, which is still well above the rate at which sign movement starts to alias. In wall-clock terms it took landmark extraction over 1,210 clips from about ninety minutes to under ten.\""),
  say("\"Then MediaPipe replaces pixels with geometry: 21 three-dimensional landmarks per hand with a handedness label, and 33 body points of which we keep nine — nose, shoulders, elbows, wrists, hips. The legs never participate in a sign.\""),
  say("\"One detail we had to fix: MediaPipe labels handedness from the camera's point of view, which is mirrored relative to the signer, so we flip it. Without that, the two hand channels swap randomly between clips and a two-handed sign becomes incoherent.\""),

  slideHead(5, "Making the features signer-invariant — the core slide", "90 s"),
  p([b("Slow down here. This is the slide you will be questioned on.")]),
  say("\"Raw landmarks are in image coordinates, so they still encode where the signer stood and how far they were from the camera. So on every frame we compute a body frame: the origin is the midpoint between the shoulders, and the scale is the distance between them — shoulder width. Every point becomes x minus origin, divided by scale.\""),
  say("\"After that, a tall signer two metres away and a short signer one metre away produce almost the same numbers. Translation invariance and scale invariance from a single transform. We recompute it per frame rather than once per clip, so it also absorbs the signer leaning or stepping sideways mid-sign. And it is guarded — if the measured shoulder width collapses below one times ten to the minus three, because the person turned sideways or the detector failed, we fall back to a scale of one instead of dividing by near-zero.\""),
  say("\"The second idea on this slide is that each hand is encoded twice. Sign language linguistics treats location and handshape as separate parameters of a sign. So we give the model both: a body-relative copy, which says where the hand is — chest, chin, forehead — and a wrist-centred copy divided by hand size, which says what shape the hand is making regardless of where it is. Supplying both means the network never has to disentangle them from one mixed signal.\""),

  slideHead(6, "What one frame becomes: 239 numbers", "60 s"),
  say("\"Concretely: 27 numbers of pose, 84 for both hands body-relative, 126 for both hands wrist-local, and 2 presence flags — 239 per frame. We then append the frame-to-frame difference of that vector, which doubles it to 478, and resample to 64 frames. So one clip is a 64 by 478 matrix.\""),
  say("\"Two choices worth defending. Velocity: the LSTM could learn to difference its own inputs, but it would spend capacity doing it — differencing is free and it is the same reasoning as delta-MFCCs in speech. And the presence flags: without them the model cannot tell 'hand absent' from 'hand at the origin', because a missing hand is zero-filled and the origin is also zero.\""),

  slideHead(7, "Temporal resampling", "60 s"),
  say("\"Clips differ in length and people sign at different speeds, but a sign performed slowly and the same sign performed quickly are the same sign, time-warped. So every feature channel is linearly interpolated onto a fixed grid of 64 frames. That gives tempo invariance and fixed-size tensors in one step, and it lets our classical SVM and Random Forest baselines consume exactly the same representation.\""),
  say("\"We tested that claim rather than assuming it: replaying the test clips at half speed and one-and-a-half times speed costs one test clip of 182: 98.4 percent against 98.9 clean.\""),
  say("\"The alternative is zero-padding to the longest clip and masking. That keeps absolute duration, which we judged not to be linguistically meaningful here, and it complicates the baselines. If duration mattered for these signs, padding would be the better choice.\""),

  slideHead(8, "Augmentation — and the negative result", "75 s"),
  p([b("This is the slide that makes the talk memorable. Lead with the measurement.")]),
  say("\"Augmentation applies to training clips only — validation and test always see the clean pipeline, otherwise the reported accuracy would not describe real inputs. Each transform imitates a real source of variation: rotation for a camera that is not level, scaling for distance, translation for being off-centre, speed jitter, frame dropout, and coordinate noise for landmark jitter.\""),
  say("\"Then there is mirroring, which we added because signs can be made with either dominant hand — and it made things worse. With two takes per class: 83.2 percent with no augmentation at all, 75.5 with everything switched on. So we isolated it: mirror-only is 74.7, everything-except-mirror is 86.0. Mirroring alone accounts for the whole loss.\""),
  say("\"The explanation is in the dataset. It has one signer with a consistent dominant hand, so mirrored clips are inputs the test distribution never contains, and many ISL signs are not left-right symmetric anyway. The general lesson is that an augmentation is only valid if it imitates variation that actually exists in deployment. Geometric jitter and noise do help — 86.5 percent.\""),

  slideHead(9, "Splitting and numerical safety", "60 s"),
  say("\"One trap in this dataset: every recorded take also ships as a left-tilt and a right-tilt copy. Those three clips are near-duplicates. If they land on opposite sides of the train-test split, you are testing on something the model has effectively memorised, and the accuracy is meaningless. So our splitter is group-aware — all variants of one take go to the same side — and stratified per class so all 61 words appear everywhere.\""),
  say("\"The split is computed once with a fixed seed before any training, and the test set is read only by the evaluation script. For the classical baselines the scaler is fitted inside a pipeline on training data only, so no test statistics leak in.\""),
  say("\"On the numerical side: every feature matrix passes a NaN and infinity guard, because MediaPipe occasionally emits non-finite values on a failed frame and a single NaN propagates through backpropagation and destroys the whole run. Landmarks are cached to compressed npz files so MediaPipe runs once and every later experiment reuses it.\""),
  p([b("Say the limitation yourself: "), new TextRun("the split is group-aware but signer-mixed, because the dataset has one signer. A signer-independent split is impossible here and accuracy on a new signer will be lower.")]),

  slideHead(10, "Every invariance we claimed, measured", "60 s"),
  say("\"Finally, we checked whether the preprocessing does what we claim. We re-tested the trained model on the same clips after deliberate perturbation, with each perturbation targeting a specific claim. Shifting off-centre and adding landmark noise leave accuracy at 98.9 percent, identical to clean; rotating the camera 15 degrees or scaling by 20 percent costs one clip of 182. That is the shoulder-frame normalisation doing its job. A mirrored signer costs about three points.\""),
  say("\"Dropping 30 percent of frames costs about two points, because resampling absorbs the gaps. Occluding one hand collapses accuracy to 25.8 percent — and that is the correct result, not a failure. Many ISL signs are two-handed, so the information is genuinely gone; a model that still scored well there would be relying on something other than the sign.\""),

  slideHead(11, "Summary", "20 s"),
  say("\"So: 280 million pixels down to 64 by 478 features without losing the sign. Landmarks instead of pixels, shoulder-frame normalisation for signer invariance, location and handshape encoded separately with velocity explicit, a fixed 64-frame grid for tempo invariance, group-aware splitting to prevent leakage, and augmentation validated by ablation rather than assumed. Happy to go deeper on any stage.\""),

  h1("Six numbers to memorise"),
  table(["Number", "What it is"], [
    ["280 M → 120 KB", "Raw pixel values per clip vs the preprocessed tensor (~2,000×)"],
    ["239 → 478", "Features per frame, before and after adding velocity"],
    ["27 / 84 / 126 / 2", "Pose / hands body-relative / hands wrist-local / presence flags"],
    ["64 frames", "Fixed temporal grid every clip is resampled onto"],
    ["98.4–98.9%", "Accuracy under rotation, scale, shift, noise and speed change (98.9% clean)"],
    ["83.2 vs 75.5 vs 86.0", "No augmentation vs all vs all-except-mirror (2 takes/class)"],
  ], [2700, 6660]),

  h1("Questions to expect"),
  ...[
    ["Why keypoints instead of feeding the video to a CNN?",
     "Data efficiency, invariance and cost. We have 20 takes per word; a 3D-CNN on pixels needs hundreds per class and still latches onto background and clothing. Keypoints give a 2,000× smaller input that trains in minutes on a laptop GPU."],
    ["What is the single most important preprocessing step?",
     "Shoulder-centred, shoulder-width-scaled normalisation. Without it the model partly learns where the signer stood and how far from the camera they were."],
    ["Why recompute the body frame every frame instead of once per clip?",
     "The signer moves during the clip. Per-frame normalisation also absorbs leaning and stepping sideways, not just the starting position."],
    ["Isn't dividing by shoulder width unstable?",
     "It is clamped — below 1e-3 we fall back to a scale of 1.0, and the visibility channel tells the model the pose estimate was weak on that frame."],
    ["Why 64 frames? Why not pad and mask?",
     "Signs vary in duration but not in structure, so resampling normalises tempo and gives fixed-size tensors the baselines can also use. Padding preserves absolute duration, which we judged not to be meaningful for isolated signs. We measured the result: 96.7% under ×0.5 and ×1.5 speed."],
    ["Why both global and wrist-local hand coordinates?",
     "Location and handshape are separate linguistic parameters of a sign. Giving both means the model does not have to separate them from one entangled signal."],
    ["What happens when a hand is not detected?",
     "That hand's 63 numbers are zeroed and its presence flag goes to 0, so 'absent' is distinguishable from 'at the origin'. Short gaps are additionally smoothed by resampling."],
    ["Do you augment the validation and test sets?",
     "No. Augmentation lives inside the training dataset only. Otherwise the reported accuracy would not describe real inputs."],
    ["How do you know your augmentations help?",
     "We ran the ablation — that is slide 8. Most help; mirroring hurt by about 10 points and we can explain why. We report it rather than quietly dropping it."],
    ["How do you know there is no data leakage?",
     "The tilt variants of one take are near-duplicates, so the splitter keeps them together; the split is computed once with a fixed seed before training; the test set is read only by the evaluation script; and the baseline scaler is fitted on training data only."],
    ["Why every 3rd frame and not every 10th?",
     "A sign's movement phase lasts a few tenths of a second. Below roughly 8–10 fps the trajectory starts to alias and fast signs become ambiguous. Every 3rd frame leaves ~10 fps, which is comfortably above that."],
    ["Did you try background subtraction or skin-colour segmentation?",
     "Deliberately not. MediaPipe already discards the background, and skin-colour segmentation fails across skin tones and warm lighting — an accuracy problem and a fairness problem. Our features are geometric, so pixel-level colour normalisation is irrelevant."],
  ].flatMap(([q, a]) => [
    new Paragraph({ spacing: { before: 150, after: 40 }, children: [new TextRun({ text: q, bold: true, color: DARK })] }),
    p(a, { indent: { left: 180 } }),
  ]),

  h1("If the prof wants to see it run"),
  p("Two commands, both quick, both safe to run live:"),
  p([new TextRun({ text: "python -m isl.demo --random 6 --play --speak", font: "Consolas", size: 20 })], { indent: { left: 180 } }),
  p("Plays held-out clips with the model's prediction overlaid and spoken aloud — proof the pipeline works end to end.", { indent: { left: 180 } }),
  p([new TextRun({ text: "python -m isl.realtime --ckpt models/bilstm_orig.pt --speak", font: "Consolas", size: 20 })], { indent: { left: 180 } }),
  p("Live webcam recognition with text and speech. Warn them it will be weaker than the clip demo, because you are a signer the model has never seen — that gap is the known limitation, not a bug.", { indent: { left: 180 } }),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22, color: DARK } } },
    paragraphStyles: [{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
      quickFormat: true, run: { size: 30, bold: true, font: "Cambria" },
      paragraph: { spacing: { before: 340, after: 170 }, outlineLevel: 0 } }],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } }, children }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(process.argv[2], buf); console.log("wrote", process.argv[2]); });
