/* ─────────────────────────────────────────────────────────────
   Curve explorer — recognition engineering page.

   The interactive form of results/recognition_pr.svg and _roc.svg: precision-
   recall and ROC for every encoder under one gallery aggregation, with the
   threshold as a control rather than a caption.

   NOTHING NEW IS SHIPPED TO DRAW THESE. The curves come from the same tp/fp
   arrays the threshold explorer reads — precision, recall, TAR and FAR are all
   ratios of accepted-genuine and accepted-non-mate at a grid threshold, so the
   plots and the readout below them are the same numbers by construction and
   cannot drift apart. Only the exact AUC/AP scalars and the two off-grid
   operating points come from the payload's `curve` block, because those are
   the parts a 0.01 grid genuinely cannot reproduce.

   The static SVGs stay in results/ for anything that is not a browser; the
   plotter reads the same JSON, so all three agree.
   ───────────────────────────────────────────────────────────── */

(function () {
  const root = document.getElementById("curves");
  if (!root) return;

  let cells;
  try {
    cells = JSON.parse(root.dataset.distributions || "{}");
  } catch (err) {
    return;
  }

  /* Only cells carrying both the grid and the scalars can be plotted. */
  const plottable = {};
  Object.keys(cells).forEach((key) => {
    if (cells[key].exact && cells[key].curve) plottable[key] = cells[key];
  });
  const keys = Object.keys(plottable);
  if (keys.length === 0) return;

  const split = (key) => ({ encoder: key.split("|")[0], aggregation: key.split("|")[1] });
  const aggregations = [];
  const encoders = [];
  keys.forEach((key) => {
    const { encoder, aggregation } = split(key);
    if (!aggregations.includes(aggregation)) aggregations.push(aggregation);
    if (!encoders.includes(encoder)) encoders.push(encoder);
  });

  // Encoder colours avoid amber and green: on this page those two already mean
  // "yardstick" and "deployment", and the markers wear them.
  const LINE_COLORS = ["#7c3aed", "#4a9eff", "#e0559e", "#35c5cf", "#c0c0d0"];
  const YARDSTICK = "#d19a3a";
  const DEPLOYMENT = "#3ad19a";
  const AXIS = "#8a8a99";
  const GRID = "#22222c";
  const TEXT = "#e0e0e0";

  const colorOf = {};
  encoders.forEach((encoder, i) => { colorOf[encoder] = LINE_COLORS[i % LINE_COLORS.length]; });

  const aggSelect = document.getElementById("curve-aggregation");
  const slider = document.getElementById("curve-threshold");
  const output = document.getElementById("curve-value");
  const logToggle = document.getElementById("curve-log");
  const legendBox = document.getElementById("curve-legend");
  const readout = document.getElementById("curve-readout");
  const prCanvas = document.getElementById("curve-pr");
  const rocCanvas = document.getElementById("curve-roc");

  aggregations.forEach((aggregation) => {
    const option = document.createElement("option");
    option.value = aggregation;
    option.textContent = aggregation;
    aggSelect.appendChild(option);
  });

  const preferredAgg = root.dataset.defaultAggregation;
  aggSelect.value = aggregations.includes(preferredAgg) ? preferredAgg : aggregations[0];
  slider.value = parseFloat(root.dataset.defaultThreshold || "0.3").toFixed(2);

  const hidden = new Set();

  /* ── Curve construction ─────────────────────────────────── */

  /* Every plottable point for one cell, derived from the stored counts.
     Points where nothing is accepted have undefined precision (0/0) and are
     marked so the PR line can break rather than dive to a fabricated zero. */
  function pointsFor(cell) {
    const { min, step, tp, fp, genuine_total, nonmate_total } = cell.exact;
    return tp.map((accepted, i) => {
      const falseAccepts = fp[i];
      const total = accepted + falseAccepts;
      return {
        threshold: min + i * step,
        recall: accepted / genuine_total,
        tar: accepted / genuine_total,
        far: falseAccepts / nonmate_total,
        precision: total > 0 ? accepted / total : null,
        tp: accepted,
        fp: falseAccepts,
      };
    });
  }

  function activeCells() {
    return keys
      .filter((key) => split(key).aggregation === aggSelect.value)
      .map((key) => ({ key, encoder: split(key).encoder, cell: plottable[key] }))
      .filter((entry) => !hidden.has(entry.encoder));
  }

  /* The smallest FAR this run can resolve — one non-mate out of the set. The
     log axis starts there, so its left edge means "resolution limit", not 0. */
  function floorFar() {
    let smallest = 1;
    keys.forEach((key) => {
      const total = plottable[key].exact.nonmate_total;
      if (total > 0) smallest = Math.min(smallest, 1 / total);
    });
    return smallest || 1e-4;
  }

  /* ── Drawing ────────────────────────────────────────────── */

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = 300;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }

  function draw(canvas, kind, threshold) {
    const { ctx, width, height } = setupCanvas(canvas);
    const pad = { left: 46, right: 14, top: 14, bottom: 34 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const isRoc = kind === "roc";
    const useLog = isRoc && logToggle.checked;
    const low = floorFar();

    const xOf = (value) => {
      if (!useLog) return pad.left + value * plotW;
      const clamped = Math.max(value, low);
      const span = Math.log10(1) - Math.log10(low);
      return pad.left + ((Math.log10(clamped) - Math.log10(low)) / span) * plotW;
    };
    const yOf = (value) => pad.top + plotH - value * plotH;

    // Grid + ticks.
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.fillStyle = AXIS;
    ctx.font = "10px 'JetBrains Mono', monospace";

    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((tick) => {
      const y = yOf(tick);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(tick.toFixed(1), pad.left - 6, y + 3);
    });

    const xTicks = useLog ? [low, 0.001, 0.01, 0.1, 1] : [0, 0.2, 0.4, 0.6, 0.8, 1];
    ctx.textAlign = "center";
    xTicks.forEach((tick) => {
      if (useLog && tick < low) return;
      const x = xOf(tick);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
      const label = useLog
        ? (tick >= 0.01 ? String(tick) : tick.toExponential(0))
        : tick.toFixed(1);
      ctx.fillText(label, x, height - 12);
    });

    // Axis titles.
    ctx.fillStyle = AXIS;
    ctx.textAlign = "center";
    ctx.fillText(isRoc ? (useLog ? "false accept rate (log)" : "false accept rate")
                       : "recall", pad.left + plotW / 2, height - 1);
    ctx.save();
    ctx.translate(11, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(isRoc ? "true accept rate" : "precision", 0, 0);
    ctx.restore();

    // Curves.
    activeCells().forEach(({ encoder, cell }) => {
      const points = pointsFor(cell);
      ctx.strokeStyle = colorOf[encoder];
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      let drawing = false;
      points.forEach((point) => {
        const value = isRoc ? point.tar : point.precision;
        const across = isRoc ? point.far : point.recall;
        // Undefined precision, or a FAR of 0 with no position on a log axis:
        // lift the pen instead of inventing a point.
        if (value === null || (useLog && across <= 0)) { drawing = false; return; }
        const x = xOf(across);
        const y = yOf(value);
        if (!drawing) { ctx.moveTo(x, y); drawing = true; } else { ctx.lineTo(x, y); }
      });
      ctx.stroke();

      // Shipped operating points, at their exact off-grid positions.
      const markers = cell.curve.markers;
      [[markers.yardstick, YARDSTICK], [markers.deployment, DEPLOYMENT]].forEach(([marker, color]) => {
        if (!marker) return;
        const value = isRoc ? marker.tar : marker.precision;
        const across = isRoc ? marker.far : marker.recall;
        if (value === null || value === undefined) return;
        if (useLog && across <= 0) return;
        ctx.beginPath();
        ctx.arc(xOf(across), yOf(value), 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#0a0a0f";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      });
    });

    // Chance line, ROC only — a diagonal on linear axes, a curve on log.
    if (isRoc) {
      ctx.strokeStyle = "rgba(138, 138, 153, 0.4)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let i = 0; i <= 60; i++) {
        const value = useLog ? low * Math.pow(1 / low, i / 60) : i / 60;
        const x = xOf(value);
        const y = yOf(value);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Where the current threshold sits on each curve.
    activeCells().forEach(({ encoder, cell }) => {
      const point = pointAt(cell, threshold);
      if (!point) return;
      const value = isRoc ? point.tar : point.precision;
      const across = isRoc ? point.far : point.recall;
      if (value === null) return;
      if (useLog && across <= 0) return;
      const x = xOf(across);
      const y = yOf(value);
      ctx.beginPath();
      ctx.arc(x, y, 6.5, 0, Math.PI * 2);
      ctx.strokeStyle = colorOf[encoder];
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = colorOf[encoder];
      ctx.fill();
    });
  }

  /* The grid point for `threshold`, by index — same lookup the threshold
     explorer uses, so a dot on the curve and the readout below always agree. */
  function pointAt(cell, threshold) {
    const points = pointsFor(cell);
    const { min, step } = cell.exact;
    let i = Math.round((threshold - min) / step);
    i = Math.max(0, Math.min(points.length - 1, i));
    return points[i];
  }

  /* ── Legend and readout ─────────────────────────────────── */

  function renderLegend() {
    legendBox.innerHTML = "";
    encoders
      .filter((encoder) => keys.some((key) => key === encoder + "|" + aggSelect.value))
      .forEach((encoder) => {
        const cell = plottable[encoder + "|" + aggSelect.value];
        const off = hidden.has(encoder);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "eng-curve-toggle" + (off ? " off" : "");
        button.innerHTML =
          '<i class="swatch" style="background:' + colorOf[encoder] + '"></i>' +
          "<span>" + encoder + "</span>" +
          '<span class="eng-curve-scalars">AP ' + cell.curve.ap.toFixed(3) +
          " · AUC " + cell.curve.auc.toFixed(3) + "</span>";
        button.addEventListener("click", () => {
          // Never let the last visible encoder be switched off — an empty
          // pair of axes reads as "no data" rather than "you hid everything".
          if (!off && activeCells().length <= 1) return;
          if (off) hidden.delete(encoder); else hidden.add(encoder);
          render();
        });
        legendBox.appendChild(button);
      });
  }

  function renderReadout(threshold) {
    const rows = activeCells().map(({ encoder, cell }) => {
      const point = pointAt(cell, threshold);
      const precision = point.precision === null ? "—" : (point.precision * 100).toFixed(1) + "%";
      return (
        "<tr><td class='mono' style='color:" + colorOf[encoder] + "'>" + encoder + "</td>" +
        "<td class='num'>" + (point.tar * 100).toFixed(1) + "%</td>" +
        "<td class='num'>" + (point.far * 100).toFixed(2) + "%</td>" +
        "<td class='num'>" + precision + "</td>" +
        "<td class='num'>" + point.tp + "</td>" +
        "<td class='num'>" + point.fp + "</td></tr>"
      );
    });
    readout.innerHTML =
      "<table class='eng-table'><thead><tr><th>encoder</th><th class='num'>TAR</th>" +
      "<th class='num'>FAR</th><th class='num'>precision</th>" +
      "<th class='num'>genuine in</th><th class='num'>non-mates in</th></tr></thead>" +
      "<tbody>" + rows.join("") + "</tbody></table>";
  }

  function render() {
    const threshold = parseFloat(slider.value);
    output.textContent = threshold.toFixed(2);
    renderLegend();
    draw(prCanvas, "pr", threshold);
    draw(rocCanvas, "roc", threshold);
    renderReadout(threshold);
  }

  aggSelect.addEventListener("change", render);
  slider.addEventListener("input", render);
  logToggle.addEventListener("change", render);
  window.addEventListener("resize", render);
  render();
})();
