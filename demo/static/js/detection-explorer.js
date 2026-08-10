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

  const dsSelect = document.getElementById("det-dataset");
  const detSelect = document.getElementById("det-detector");
  const slider = document.getElementById("det-threshold");
  const output = document.getElementById("det-value");
  const stats = document.getElementById("det-stats");
  const canvas = document.getElementById("det-canvas");
  const ctx = canvas.getContext("2d");

  const COLORS = { precision: "#3ad19a", recall: "#d19a3a", f1: "#7c9ad1" };

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
      option.textContent = name;
      detSelect.appendChild(option);
    });
    detSelect.value = available.includes("yolo") ? "yolo" : available[0];
  }

  function currentSweep() {
    return (sweeps[dsSelect.value] || {})[detSelect.value] || [];
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
    output.textContent = threshold.toFixed(2);
    const point = pointAt(sweep, threshold);

    stats.innerHTML = "";
    [
      ["precision", (point.precision * 100).toFixed(1) + "%", "of accepts are faces",
       point.precision > 0.8 ? "good" : ""],
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

    ["precision", "recall", "f1"].forEach((series) => {
      ctx.beginPath();
      sweep.forEach((point, i) => {
        const px = x(point.threshold);
        const py = y(point[series]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = COLORS[series];
      ctx.lineWidth = 1.8;
      ctx.stroke();
    });

    // Shipped threshold marker, so the deployed choice is visible on the curve.
    if (hasShipped) {
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x(shipped), pad.top);
      ctx.lineTo(x(shipped), pad.top + plotH);
      ctx.strokeStyle = "rgba(255,255,255,0.28)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.textAlign = "left";
      ctx.fillStyle = "#8a8a99";
      ctx.fillText("shipped " + shipped.toFixed(2), x(shipped) + 5, pad.top + 10);
    }

    // Draggable operating point.
    ctx.beginPath();
    ctx.moveTo(x(threshold), pad.top);
    ctx.lineTo(x(threshold), pad.top + plotH);
    ctx.strokeStyle = "#7c3aed";
    ctx.lineWidth = 2;
    ctx.stroke();

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
