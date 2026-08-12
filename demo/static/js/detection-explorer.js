/* ─────────────────────────────────────────────────────────────
   Detection threshold explorer — engineering log.

   Plots precision, recall and F1 against the confidence threshold for one
   detector on one dataset, with a draggable operating point.

   Unlike the recognition explorer this is NOT an approximation of something
   finer: the confidence sweep stored by benchmark_detection is a resampling of
   the exact precision-recall curve that AP is computed from, taken at 200
   fixed confidences so detectors are directly comparable.
   ───────────────────────────────────────────────────────────── */

(function () {
  const root = document.getElementById("det-explorer");
  if (!root) return;

  let sweeps;
  try {
    sweeps = JSON.parse(root.dataset.sweeps || "{}");
  } catch (err) {
    return;
  }
  const datasets = Object.keys(sweeps);
  if (datasets.length === 0) return;

  const shipped = parseFloat(root.dataset.shipped);
  const hasShipped = !Number.isNaN(shipped);
  const shippedDetector = root.dataset.shippedDetector || "";
  let shippedDatasets = [];
  try {
    shippedDatasets = JSON.parse(root.dataset.shippedDatasets || "[]");
  } catch (err) {
    shippedDatasets = [];
  }

  const dsSelect = document.getElementById("det-dataset");
  const detSelect = document.getElementById("det-detector");
  const slider = document.getElementById("det-threshold");
  const output = document.getElementById("det-value");
  const stats = document.getElementById("det-stats");
  const scopeNote = document.getElementById("det-scope");
  const ghostLegend = document.getElementById("det-ghost-legend");
  const canvas = document.getElementById("det-canvas");
  const ctx = canvas.getContext("2d");

  const COLORS = { precision: "#3ad19a", recall: "#d19a3a", f1: "#7c9ad1" };
  const GHOST = "rgba(124, 154, 209, 0.38)";
  const ACCENT = "#7c3aed";
  const PANEL = "#101018";   // --surface, the colour the canvas sits on

  datasets.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    dsSelect.appendChild(option);
  });
  dsSelect.value = datasets.includes("widerface") ? "widerface" : datasets[0];

  function fillDetectors() {
    const available = Object.keys(sweeps[dsSelect.value] || {});
    detSelect.innerHTML = "";
    available.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      // Which detector actually ships is the whole point of the benchmark, and
      // the chart alone does not say it — the picker has to.
      option.textContent = name === shippedDetector ? name + "  · shipped" : name;
      detSelect.appendChild(option);
    });
    const preferred = shippedDetector || "yolo";
    detSelect.value = available.includes(preferred) ? preferred : available[0];
  }

  function currentSweep() {
    return (sweeps[dsSelect.value] || {})[detSelect.value] || [];
  }

  /* Two DIFFERENT reasons the shipped threshold may not belong on a curve, and
     they do not deserve the same treatment.

     WRONG DETECTOR — not drawn at all. A confidence score is whatever a given
     model's head emits; 0.57 from YOLO is not the same quantity as 0.57 from
     MTCNN or BlazeFace, and the two cannot be compared even in principle.
     Drawing the line there is not weak evidence, it is a category error, so
     the marker is omitted and the note says why.

     WRONG DATASET, RIGHT DETECTOR — drawn dim. Same detector, same score
     scale, so the line is meaningful; FDDB simply had no vote in choosing the
     value (it is the easy set — see pipeline_config.py). Worth seeing, not
     worth presenting as that curve's operating point. */
  function onShippedDetector() {
    if (!shippedDetector) return true;               // no provenance to honour
    return detSelect.value === shippedDetector;
  }

  function onDecidingDataset() {
    if (shippedDatasets.length === 0) return true;   // no provenance to honour
    return shippedDatasets.includes(dsSelect.value);
  }

  /* The shipped detector's curve on the SAME dataset, so a detector that lost
     can be seen losing rather than just asserted to have lost in a table. */
  function ghostSweep() {
    if (onShippedDetector()) return null;
    return (sweeps[dsSelect.value] || {})[shippedDetector] || null;
  }

  function describeScope() {
    const detector = detSelect.value;
    if (!onShippedDetector()) {
      scopeNote.className = "eng-scope-note off-scope";
      scopeNote.innerHTML =
        "<strong>" + detector + " is not the shipped detector</strong> — " +
        shippedDetector + " is. Confidence is whatever each model's head " +
        "emits, so " + shipped.toFixed(2) + " does not transfer between " +
        "detectors and no shipped marker is drawn here." +
        (ghostSweep() ? " The faint line is " + shippedDetector +
                        "'s F1 on this same dataset, for comparison." : "");
      return;
    }
    scopeNote.className = "eng-scope-note";
    if (onDecidingDataset()) {
      scopeNote.innerHTML =
        "<strong>" + detector + " at " + shipped.toFixed(2) +
        " is what actually runs.</strong> " + dsSelect.value + " is one of " +
        "the datasets that decided this threshold, so the dashed marker is a " +
        "live operating point.";
    } else {
      scopeNote.innerHTML =
        "<strong>" + dsSelect.value + " had no vote in this threshold.</strong> " +
        "It is the easy set, and was deliberately excluded so it could not " +
        "bias the value upward. The marker is dimmed: same detector, same " +
        "confidence scale, but this curve is not evidence for it.";
    }
  }

  /* Precision is tp/(tp+fp) — UNDEFINED, not zero, once the confidence is high
     enough that the detector returns nothing at all. The stored sweep records
     0.0 there, so plotting it verbatim draws a cliff to the floor that the
     detector never fell off. */
  function precisionKnown(point) {
    return (point.tp + point.fp) > 0;
  }

  /* Nearest stored grid point to the requested confidence. */
  function pointAt(sweep, threshold) {
    let best = sweep[0];
    let bestGap = Infinity;
    for (const point of sweep) {
      const gap = Math.abs(point.threshold - threshold);
      if (gap < bestGap) { bestGap = gap; best = point; }
    }
    return best;
  }

  function render() {
    const sweep = currentSweep();
    if (sweep.length === 0) return;
    const threshold = parseFloat(slider.value);
    const point = pointAt(sweep, threshold);
    // The stats come from the nearest STORED grid point, so the readout has to
    // be that point's confidence at the grid's own resolution. Rounding it to
    // 2dp printed a number the figures below were not computed at, and hid
    // every other step of the 0.005 slider.
    output.textContent = point.threshold.toFixed(3);

    const known = precisionKnown(point);

    if (hasShipped) {
      scopeNote.hidden = false;
      describeScope();
    } else {
      scopeNote.hidden = true;
    }
    ghostLegend.hidden = !ghostSweep();

    stats.innerHTML = "";
    [
      ["precision",
       known ? (point.precision * 100).toFixed(1) + "%" : "—",
       known ? "of accepts are faces" : "nothing accepted here",
       known && point.precision > 0.8 ? "good" : ""],
      ["recall", (point.recall * 100).toFixed(1) + "%", "of faces found",
       point.recall > 0.8 ? "good" : ""],
      ["F1", point.f1.toFixed(3), "balance of the two", ""],
      ["true positives", point.tp.toLocaleString(), "faces detected", ""],
      ["false positives", point.fp.toLocaleString(), "boxes to discard",
       point.fp > point.tp ? "bad" : ""],
    ].forEach(([label, value, sub, tone]) => {
      const div = document.createElement("div");
      div.className = "eng-stat" + (tone ? " " + tone : "");
      div.innerHTML =
        '<span class="eng-stat-value">' + value + "</span>" +
        '<span class="eng-stat-label">' + label + "</span>" +
        '<span class="eng-stat-label" style="opacity:.6">' + sub + "</span>";
      stats.appendChild(div);
    });

    draw(sweep, threshold);
  }

  function draw(sweep, threshold) {
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = 240;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const pad = { left: 34, right: 14, top: 12, bottom: 26 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const x = (t) => pad.left + t * plotW;
    const y = (v) => pad.top + plotH - v * plotH;

    // Horizontal gridlines at 0, .25, .5, .75, 1
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.fillStyle = "#6a6a78";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.lineWidth = 1;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      ctx.beginPath();
      ctx.moveTo(pad.left, y(v));
      ctx.lineTo(pad.left + plotW, y(v));
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(v.toFixed(2), pad.left - 6, y(v) + 3);
    });

    /* Text tied to a vertical marker: a short tick back to the line, and a
       backdrop in the panel colour so a curve passing behind it stays
       readable. Flips to the left of the line when the right runs off plot. */
    function markerLabel(text, lineX, color) {
      const textW = ctx.measureText(text).width;
      const fitsRight = lineX + 7 + textW + 3 <= pad.left + plotW;
      const textX = fitsRight ? lineX + 7 : lineX - 7 - textW;
      const midY = pad.top + 9;
      ctx.fillStyle = PANEL;
      ctx.fillRect(textX - 3, midY - 7, textW + 6, 14);
      ctx.beginPath();
      ctx.moveTo(lineX, midY);
      ctx.lineTo(fitsRight ? textX - 3 : textX + textW + 3, midY);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = color;
      ctx.fillText(text, textX, midY);
      ctx.textBaseline = "alphabetic";
    }

    /* One series, broken wherever it is undefined rather than plotted as 0. */
    function stroke(points, series, color, width, defined) {
      ctx.beginPath();
      let pen = false;
      points.forEach((point) => {
        if (defined && !defined(point)) { pen = false; return; }
        const px = x(point.threshold);
        const py = y(point[series]);
        if (pen) ctx.lineTo(px, py); else ctx.moveTo(px, py);
        pen = true;
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    }

    // The shipped detector's F1 on this same dataset, behind everything, so a
    // losing detector can be SEEN losing instead of only said to have lost.
    const ghost = ghostSweep();
    if (ghost) {
      ctx.setLineDash([2, 3]);
      stroke(ghost, "f1", GHOST, 1.4, null);
      ctx.setLineDash([]);
    }

    ["precision", "recall", "f1"].forEach((series) => {
      // Only precision can be undefined; recall and F1 are honestly 0 when the
      // detector returns nothing, so their lines stay continuous.
      stroke(sweep, series, COLORS[series], 1.8,
             series === "precision" ? precisionKnown : null);
    });

    // Everything at or above the operating point is accepted — the same
    // shading the recognition explorer uses for the same idea.
    const tx = x(threshold);
    ctx.fillStyle = "rgba(124, 58, 237, 0.10)";
    ctx.fillRect(tx, pad.top, pad.left + plotW - tx, plotH);

    // Draggable operating point. Drawn BEFORE the shipped marker: the two sit
    // at the same confidence on load, and the wider opaque line would
    // otherwise hide the reference it is supposed to be compared against.
    ctx.beginPath();
    ctx.moveTo(tx, pad.top);
    ctx.lineTo(tx, pad.top + plotH);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.stroke();

    /* Shipped threshold marker. Only ever drawn on the detector it was
       measured on — see onShippedDetector: across detectors the confidence
       scales are not the same quantity, so the line would be meaningless
       rather than merely unsupported. */
    if (hasShipped && onShippedDetector()) {
      const decided = onDecidingDataset();
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x(shipped), pad.top);
      ctx.lineTo(x(shipped), pad.top + plotH);
      ctx.strokeStyle = decided ? "rgba(255,255,255,0.55)"
                                : "rgba(255,255,255,0.2)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      markerLabel(
        "shipped " + shipped.toFixed(2) + (decided ? "" : " · not decided here"),
        x(shipped),
        decided ? "#b9b9c8" : "#6a6a78");
    }

    // "accept →" only where it has room to sit without colliding with the
    // shipped label above it or running off the plot.
    if (tx < pad.left + plotW - 66) {
      ctx.textAlign = "left";
      ctx.fillStyle = ACCENT;
      ctx.fillText("accept →", tx + 6, pad.top + plotH - 6);
    }

    ctx.textAlign = "center";
    ctx.fillStyle = "#6a6a78";
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      ctx.fillText(t.toFixed(2), x(t), height - 9);
    });
  }

  dsSelect.addEventListener("change", () => { fillDetectors(); render(); });
  detSelect.addEventListener("change", render);
  slider.addEventListener("input", render);
  window.addEventListener("resize", render);

  fillDetectors();
  if (hasShipped) slider.value = shipped;
  render();
})();
