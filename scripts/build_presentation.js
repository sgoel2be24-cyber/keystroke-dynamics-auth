// Keystroke Dynamics: 7-slide deck. Usage: node build.js <out.pptx> [previewDir]
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "Keystroke",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2433", lt1: "FFFFFF", dk2: "2E3A4D", lt2: "EEF1F5",
    accent1: "E8A33D", accent2: "3F7F8C", accent3: "8A94A6", accent4: "C8CED8",
    accent5: "C8553D", accent6: "5B8C5A", hlink: "3F7F8C", folHlink: "8A94A6",
  },
};
const HEX = THEME.colors;

function sec(pres, t) { if (!pres._secs.has(t)) { pres.addSection({ title: t }); pres._secs.add(t); } return t; }

function makePres() {
  const pres = new pptxgen();
  pres._secs = new Set();
  pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Keystroke Dynamics: Behavioral Biometric Authentication";
  const C = pres.SchemeColor;
  pres.defineSlideMaster({
    title: "DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.25, w: 8.8, h: 1.0, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.3, w: 8.8, h: 0.6, fontSize: 18, color: C.accent4, margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: "Keystroke Dynamics · CMU benchmark (Killourhy & Maxion, 2009)", options: { x: 0.5, y: 5.25, w: 6.5, h: 0.25, fontSize: 9, color: C.accent3, margin: 0 } } },
    ],
    slideNumber: { x: 9.1, y: 5.22, w: 0.4, h: 0.3, fontSize: 9, color: C.accent3, align: "right" },
  });
  return pres;
}

// "Keycap" motif: a rounded tile with a big value and a small label underneath.
function keycap(slide, pres, { x, y, w, h, value, label, dark = false, accent = false, valueSize = 30, name }) {
  const C = pres.SchemeColor;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, objectName: name + " tile",
    fill: { color: accent ? C.accent1 : dark ? C.text2 : C.background2 },
    line: { color: accent ? C.accent1 : dark ? C.text2 : C.background2, width: 0.75 },
    shadow: { type: "outer", color: "000000", opacity: 0.18, blur: 3, offset: 2, angle: 90 },
  });
  slide.addText(value, { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: h * 0.55, fontSize: valueSize, bold: true, fontFace: "Cambria",
    color: accent ? C.text1 : dark ? C.background1 : C.text1, valign: "bottom", margin: 0, isTextBox: true, objectName: name + " value" });
  slide.addText(label, { x: x + 0.12, y: y + h * 0.62, w: w - 0.24, h: h * 0.34, fontSize: 12,
    color: accent ? C.text1 : dark ? C.accent4 : C.text2, valign: "top", margin: 0, isTextBox: true, objectName: name + " label" });
}

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  dataLabelFontSize: 10, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.accent3, dataLabelColor: HEX.dk1 };

const SLIDES = [
  // 1. Title
  (pres) => {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: sec(pres, "Introduction") });
    s.addText("Keystroke Dynamics", { placeholder: "title" });
    s.addText("Recognising people by how they type a password", { placeholder: "body" });
    const keys = [["51", "users"], ["400", "typings each"], ["8", "sessions"], ["31", "timings per sample"]];
    keys.forEach(([v, l], i) => keycap(s, pres, { x: 0.6 + i * 2.25, y: 3.25, w: 2.0, h: 1.3, value: v, label: l, dark: true, accent: i === 0, valueSize: 32, name: `stat-${i}` }));
    s.addText("ML semester project · Person A (identification) · Person B (classical authentication) · Person C (deep authentication, drift)",
      { x: 0.6, y: 4.85, w: 8.8, h: 0.4, fontSize: 12, color: pres.SchemeColor.accent3, margin: 0, isTextBox: true });
    s.addNotes("Passwords can be stolen; typing rhythm is much harder to copy. We use the CMU benchmark: 51 people typed .tie5Roanl 400 times over 8 sessions.");
  },
  // 2. Data
  (pres) => {
    const C = pres.SchemeColor;
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec(pres, "Data and method") });
    s.addText("Each sample is 31 timings across 11 keystrokes", { placeholder: "title" });
    const keys = [".", "t", "i", "e", "5", "R", "o", "a", "n", "l", "⏎"];
    keys.forEach((k, i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5 + i * 0.52, y: 1.35, w: 0.44, h: 0.44, rectRadius: 0.06,
        fill: { color: i === 5 ? C.accent1 : C.background2 }, line: { color: C.accent4, width: 0.75 }, objectName: `key-${i}` });
      s.addText(k, { x: 0.5 + i * 0.52, y: 1.35, w: 0.44, h: 0.44, fontSize: 14, bold: true, align: "center", valign: "middle", color: C.text1, margin: 0, isTextBox: true });
    });
    s.addText([
      { text: "H  ", options: { bold: true, color: C.accent2 } }, { text: "hold time: press to release (11)", options: { breakLine: true } },
      { text: "DD  ", options: { bold: true, color: C.accent2 } }, { text: "down-down: press to next press (10)", options: { breakLine: true } },
      { text: "UD  ", options: { bold: true, color: C.accent2 } }, { text: "up-down: release to next press (10)" },
    ], { x: 0.5, y: 2.05, w: 5.6, h: 1.0, fontSize: 14, color: C.text2, margin: 0, paraSpaceAfter: 4, isTextBox: true });
    s.addText([
      { text: "Clean and balanced: no missing values, exactly 400 samples per user", options: { bullet: { indent: 14 }, breakLine: true } },
      { text: "Sessions at least a day apart, so typing can change between them", options: { bullet: { indent: 14 }, breakLine: true } },
      { text: "DD = H + UD exactly, so only 21 of the 31 features are independent", options: { bullet: { indent: 14 } } },
    ], { x: 0.5, y: 3.25, w: 5.6, h: 1.6, fontSize: 14, color: C.text1, margin: 0, paraSpaceAfter: 6, isTextBox: true });
    keycap(s, pres, { x: 6.55, y: 1.35, w: 2.95, h: 1.1, value: "27% faster", label: "typing by session 8 vs session 1", accent: true, valueSize: 26, name: "eda-speed" });
    keycap(s, pres, { x: 6.55, y: 2.6, w: 2.95, h: 1.1, value: "Top 8", label: "most telling features are all hold times", valueSize: 26, name: "eda-hold" });
    keycap(s, pres, { x: 6.55, y: 3.85, w: 2.95, h: 1.1, value: "1.5-5.4 s", label: "range of users' mean typing time", valueSize: 24, name: "eda-range" });
    s.addNotes("Hold times carry the most signal. Typing drifts: people get faster with practice, which matters for every split we choose.");
  },
  // 3. Method
  (pres) => {
    const C = pres.SchemeColor;
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec(pres, "Data and method") });
    s.addText("Three tasks, each tested on unseen sessions", { placeholder: "title" });
    const cards = [
      ["Task 1 · Identification", "Who typed this? 51-class classification", "Train sessions 1-5 and 8\nTest sessions 6-7", "Accuracy, macro F1"],
      ["Task 2 · Authentication", "Is this really the claimed user? One detector per user, trained on genuine typing only", "Train sessions 1-4\nTest 5-8 + 250 impostor tries", "Equal error rate (EER)"],
      ["Task 3 · Session drift", "Does typing change, and does that hurt a detector?", "Train sessions 1-2\nTest each of 3-8", "EER per session"],
    ];
    cards.forEach(([t, q, split, metric], i) => {
      const x = 0.5 + i * 3.05;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 2.85, h: 2.75, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `task-card-${i}` });
      s.addText(t, { x: x + 0.2, y: 1.45, w: 2.45, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
      s.addText(q, { x: x + 0.2, y: 1.9, w: 2.45, h: 0.85, fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true });
      s.addText(split, { x: x + 0.2, y: 2.8, w: 2.45, h: 0.6, fontSize: 12, bold: true, color: C.accent2, margin: 0, valign: "top", isTextBox: true });
      s.addText(metric, { x: x + 0.2, y: 3.5, w: 2.45, h: 0.35, fontSize: 12, color: C.text2, italic: true, margin: 0, isTextBox: true });
    });
    s.addText([
      { text: "EER: ", options: { bold: true } },
      { text: "the error rate at the strictness where impostors let in = real users locked out. Lower is better. Every scaler is fit on training data only; every hyperparameter is tuned on training sessions." },
    ], { x: 0.5, y: 4.3, w: 9.0, h: 0.8, fontSize: 13, color: C.text1, margin: 0, isTextBox: true });
    s.addNotes("Samples from one session are near-identical, so we never split at random. Task 2 follows the published protocol so our numbers compare directly with the paper.");
  },
  // 4. Task 1
  (pres) => {
    const C = pres.SchemeColor;
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec(pres, "Results") });
    s.addText("A 1D-CNN identifies 94.7% of users from one typing", { placeholder: "title" });
    const models = ["Logistic Regression", "SVM (RBF)", "BiLSTM", "Random Forest", "XGBoost", "MLP", "1D-CNN"];
    const acc = [81.6, 87.9, 90.9, 92.9, 93.2, 93.9, 94.7];
    s.addChart(pres.charts.BAR, [{ name: "Accuracy (%)", labels: models, values: acc }], {
      x: 0.4, y: 1.15, w: 5.9, h: 3.95, barDir: "bar", objectName: "task1-accuracy-chart",
      chartColors: [HEX.accent3, HEX.accent3, HEX.accent3, HEX.accent3, HEX.accent3, HEX.accent3, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"',
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      showLegend: false, barGapWidthPct: 45, ...chartText,
    });
    keycap(s, pres, { x: 6.6, y: 1.3, w: 2.9, h: 1.15, value: "94.7%", label: "1D-CNN, train 1-5 + 8, test 6-7", accent: true, valueSize: 30, name: "t1-main" });
    keycap(s, pres, { x: 6.6, y: 2.6, w: 1.38, h: 1.05, value: "85.7%", label: "paper split", valueSize: 20, name: "t1-paper" });
    keycap(s, pres, { x: 8.12, y: 2.6, w: 1.38, h: 1.05, value: "89.5%", label: "past data only", valueSize: 20, name: "t1-past" });
    s.addText("Paper split: train 1-4, test 5-8. Past only: train 1-6, test 7-8. Convolutions read neighbouring keystrokes, where personal patterns live; training on session 8 makes 94.7% a best case.",
      { x: 6.6, y: 3.8, w: 2.9, h: 1.3, fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    s.addNotes("Main split: train on sessions 1-5 and 8, test on 6-7. It includes typing from after the test days, so it is the best case; with only past data the CNN scores 89.5%, and on the paper's split 85.7%. The ranking of models is the same on all three splits.");
  },
  // 5. Task 2
  (pres) => {
    const C = pres.SchemeColor;
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec(pres, "Results") });
    s.addText("kNN beats the paper's best: 7.8% vs 9.6% EER", { placeholder: "title" });
    const dets = ["LSTM autoencoder", "Dense autoencoder", "Scaled Manhattan", "One-class SVM", "Isolation Forest", "kNN, scaled Manhattan"];
    const eer = [11.5, 10.0, 9.5, 9.1, 8.1, 7.8];
    s.addChart(pres.charts.BAR, [{ name: "Mean EER (%)", labels: dets, values: eer }], {
      x: 0.4, y: 1.15, w: 5.9, h: 3.95, barDir: "bar", objectName: "task2-eer-chart",
      chartColors: [HEX.accent4, HEX.accent4, HEX.accent3, HEX.accent3, HEX.accent3, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"',
      valAxisMinVal: 0, valAxisMaxVal: 14, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      showLegend: false, barGapWidthPct: 45, ...chartText,
    });
    keycap(s, pres, { x: 6.6, y: 1.3, w: 2.9, h: 1.15, value: "7.8% EER", label: "kNN; better than the paper for 39 of 51 users", accent: true, valueSize: 28, name: "t2-main" });
    keycap(s, pres, { x: 6.6, y: 2.6, w: 2.9, h: 1.05, value: "7 of 10", label: "paper results reproduced (±0.1)", valueSize: 22, name: "t2-repro" });
    s.addText("Deep autoencoders (light bars) reach 10.0% and 11.5%: with 200 samples per user, a good distance measure beats a neural network.",
      { x: 6.6, y: 3.8, w: 2.9, h: 1.3, fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    s.addNotes("Each user gets a detector trained only on their own typing. 7 of the paper's 10 detectors match within 0.1 points, so the protocol is right. kNN compares a sample to the user's closest past sample instead of their average. Paired Wilcoxon p = 0.0002.");
  },
  // 6. Task 3
  (pres) => {
    const C = pres.SchemeColor;
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec(pres, "Results") });
    s.addText("Typing drifts, and retraining keeps the error falling", { placeholder: "title" });
    const sess = ["S3", "S4", "S5", "S6", "S7", "S8"];
    s.addChart(pres.charts.LINE, [
      { name: "Trained once", labels: sess, values: [9.3, 9.6, 10.4, 10.8, 12.3, 13.1] },
      { name: "Retrained on 2 latest sessions", labels: sess, values: [9.3, 7.7, 7.3, 6.3, 6.7, 6.5] },
    ], {
      x: 0.4, y: 1.15, w: 5.9, h: 3.95, objectName: "task3-drift-chart",
      chartColors: [HEX.accent3, HEX.accent1], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 7,
      valAxisMinVal: 5, valAxisMaxVal: 14, valAxisMajorUnit: 3, valAxisLabelFormatCode: '0"%"',
      valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: HEX.dk2,
      showValAxisTitle: true, valAxisTitle: "Mean EER, scaled Manhattan", valAxisTitleFontSize: 10, valAxisTitleColor: HEX.accent3, valAxisTitleFontFace: "+mn-lt",
      ...chartText,
    });
    keycap(s, pres, { x: 6.6, y: 1.3, w: 2.9, h: 1.05, value: "9.3 → 13.1%", label: "EER of a detector trained once", valueSize: 22, name: "t3-static" });
    keycap(s, pres, { x: 6.6, y: 2.5, w: 2.9, h: 1.05, value: "9.3 → 6.5%", label: "retrained on recent sessions", accent: true, valueSize: 22, name: "t3-sliding" });
    s.addText("71% of (user, feature) pairs change significantly between sessions 1 and 8. A real system should keep updating each user's profile.",
      { x: 6.6, y: 3.75, w: 2.9, h: 1.35, fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    s.addNotes("Hold times move most, about one standard deviation by session 8. Impostor samples always come from session 1, so read the retrained line as: drift is recoverable. The LSTM autoencoder shows the same pattern (12.5% to 13.6% static, to 9.4% retrained).");
  },
  // 7. Conclusion
  (pres) => {
    const C = pres.SchemeColor;
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: sec(pres, "Conclusion") });
    s.addText("Typing rhythm is a usable biometric", { x: 0.6, y: 0.45, w: 8.8, h: 0.8, fontSize: 36, bold: true, fontFace: "Cambria", color: C.background1, margin: 0, isTextBox: true });
    const take = [["94.7%", "identification accuracy, 51 users"], ["7.8%", "authentication EER, below the paper's 9.6%"], ["Retrain", "profiles drift; recent typing keeps error low"]];
    take.forEach(([v, l], i) => keycap(s, pres, { x: 0.6 + i * 3.0, y: 1.45, w: 2.75, h: 1.25, value: v, label: l, dark: true, accent: i === 1, valueSize: 28, name: `take-${i}` }));
    s.addText([
      { text: "Demo: ", options: { bold: true, color: C.accent1 } },
      { text: "Streamlit app with a tab per task: identify a sample, try a login as the real user or an impostor, adjust strictness, see drift.", options: { breakLine: true } },
      { text: "Limits: ", options: { bold: true, color: C.accent1 } },
      { text: "one fixed password, 51 users from one lab study; Task 1's best split trains on later typing.", options: { breakLine: true } },
      { text: "Next: ", options: { bold: true, color: C.accent1 } },
      { text: "Siamese network for few-shot enrolment, adaptive templates, live typing capture." },
    ], { x: 0.6, y: 3.0, w: 8.8, h: 1.8, fontSize: 14, color: C.background1, margin: 0, paraSpaceAfter: 8, valign: "top", isTextBox: true });
    s.addText("Code, notebooks and results: github.com/sgoel2be24-cyber/keystroke-dynamics-auth",
      { x: 0.6, y: 4.95, w: 8.8, h: 0.3, fontSize: 11, color: C.accent3, margin: 0, isTextBox: true });
    s.addNotes("Simple methods win on small per-user data; drift is the main practical challenge.");
  },
];

async function build(out, only) {
  const pres = makePres();
  SLIDES.forEach((f, i) => { if (only === undefined || only === i) f(pres); });
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
}

(async () => {
  const out = process.argv[2], previews = process.argv[3];
  await build(out);
  if (previews) for (let i = 0; i < SLIDES.length; i++) await build(path.join(previews, `slide${i + 1}.pptx`), i);
  console.log("built", out);
})();
