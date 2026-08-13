/* ─────────────────────────────────────────────────────────────
   Threshold explorer — recognition engineering page.

   Draws the three score populations (genuine / impostor / stranger) and
   recomputes the accept/reject trade as you move the threshold.

   TWO SOURCES, DELIBERATELY. The canvas draws the 40-bin histograms, which is
   all a distribution SHAPE needs. The readout above it reads `exact`: the
   threshold sweep counted from the real scores when the benchmark ran, on the
   same 0.01 grid as the slider, so a quoted count is the true count.

   The binned path below it is the fallback for runs recorded before the sweep
   was persisted. It splits the straddled bin by interpolation, which is why it
   can report a third of a false accept — a real limitation, not a rounding
   artifact, and the page shows a caveat whenever this path is in use.
   ───────────────────────────────────────────────────────────── */

(function () {
  const root = document.getElementById("explorer");
  if (!root) return;

  let distributions;
  try {
    distributions = JSON.parse(root.dataset.distributions || "{}");
  } catch (err) {
    return;
  }
  const keys = Object.keys(distributions);
  if (keys.length === 0) return;

  const select = document.getElementById("explorer-cell");
  const slider = document.getElementById("explorer-threshold");
  const output = document.getElementById("explorer-value");
  const stats = document.getElementById("explorer-stats");
  const canvas = document.getElementById("explorer-canvas");
  const ctx = canvas.getContext("2d");

  const COLORS = {
    genuine: "#3ad19a",
    impostor: "#d19a3a",
    stranger: "#6b6b8a",
  };

  keys.forEach((key) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = key.replace("|", "  ·  ");
    select.appendChild(option);
  });

  const preferred = root.dataset.default;
  select.value = keys.includes(preferred) ? preferred : keys[0];
  slider.value = parseFloat(root.dataset.defaultThreshold || "0.3").toFixed(2);

  const shipped = parseFloat(root.dataset.shipped);
  const hasShipped = !Number.isNaN(shipped);
  const shippedEncoder = root.dataset.shippedEncoder || "";
  const shippedAggregation = root.dataset.shippedAggregation || "";
  const scopeNote = document.getElementById("explorer-scope");

  const encoderOf = (key) => key.split("|")[0];
  const aggregationOf = (key) => key.split("|")[1];

  // Which cell actually ships is the whole point of the benchmark, and the
  // dropdown is where someone chooses what to look at.
  if (shippedEncoder && shippedAggregation) {
    const shippedKey = shippedEncoder + "|" + shippedAggregation;
    Array.prototype.forEach.call(select.options, (option) => {
      if (option.value === shippedKey) option.textContent += "  ·  shipped";
    });
  }

  /* Two different reasons the shipped threshold may not belong on this curve,
     and they do not deserve the same treatment — the same distinction the
     detection explorer draws.

     WRONG ENCODER — not drawn at all. A cosine similarity is a distance in
     that encoder's own embedding space; 0.34 in arcface's space is not the
     same quantity as 0.34 in facenet512's, and the populations sit at
     different places on the axis. Drawing the line there is a category error
     rather than weak evidence, so it is omitted and the note says why.

     WRONG AGGREGATION, RIGHT ENCODER — drawn dim. Same encoder, so the same
     score scale and a meaningful position; the value was simply tuned against
     a different way of building the gallery. Worth seeing, not worth
     presenting as this curve's operating point. */
  function onShippedEncoder() {
    if (!shippedEncoder) return true;                // no provenance to honour
    return encoderOf(select.value) === shippedEncoder;
  }

  function onShippedAggregation() {
    if (!shippedAggregation) return true;
    return aggregationOf(select.value) === shippedAggregation;
  }

  function describeScope() {
    if (!scopeNote || !hasShipped) return;
    const encoder = encoderOf(select.value);
    if (!onShippedEncoder()) {
      scopeNote.className = "eng-scope-note off-scope";
      scopeNote.innerHTML =
        "<strong>" + encoder + " is not the shipped encoder</strong> — " +
        shippedEncoder + " is. A cosine similarity only means something " +
        "inside one encoder's embedding space, so " + shipped.toFixed(3) +
        " does not transfer here and no shipped marker is drawn.";
      return;
    }
    scopeNote.className = "eng-scope-note";
    if (onShippedAggregation()) {
      scopeNote.innerHTML =
        "<strong>" + encoder + " at " + shipped.toFixed(3) +
        " is what actually runs.</strong> This is the cell the gallery was " +
        "built from, so the dashed marker is a live operating point.";
    } else {
      scopeNote.innerHTML =
        "<strong>" + aggregationOf(select.value) + " did not decide this " +
        "threshold.</strong> " + shippedAggregation + " did. Same encoder, so " +
        shipped.toFixed(3) + " sits on the same scale and the marker is drawn " +
        "dim — a reference point, not this curve's operating point.";
    }
  }

  /* Count of samples at or above `threshold`, interpolating the straddled
     bin so dragging the slider moves the numbers smoothly. */
  function countAbove(histogram, threshold) {
    if (!histogram) return 0;
    const { edges, counts } = histogram;
    let total = 0;
    for (let i = 0; i < counts.length; i++) {
      const low = edges[i];
      const high = edges[i + 1];
      if (threshold <= low) {
        total += counts[i];
      } else if (threshold < high) {
        const fraction = (high - threshold) / (high - low);
        total += counts[i] * fraction;
      }
    }
    return total;
  }

  function sum(histogram) {
    if (!histogram) return 0;
    return histogram.counts.reduce((a, b) => a + b, 0);
  }

  /* Exact accepted counts at `threshold`, straight off the recorded sweep.
     The grid and the slider share a 0.01 step, so this is an index lookup, not
     a search or an interpolation. Thresholds outside the grid clamp to its
     ends, where the answer is "everything" / "nothing" anyway. */
  function exactAt(exact, threshold) {
    const last = exact.tp.length - 1;
    let i = Math.round((threshold - exact.min) / exact.step);
    i = Math.max(0, Math.min(last, i));
    return {
      trueAccepts: exact.tp[i],
      falseAccepts: exact.fp[i],
      genuineTotal: exact.genuine_total,
      nonmateTotal: exact.nonmate_total,
    };
  }

  /* Interpolated counts, for runs with no sweep recorded. */
  function binnedAt(cell, threshold) {
    return {
      trueAccepts: countAbove(cell.genuine, threshold),
      falseAccepts: countAbove(cell.impostor, threshold) + countAbove(cell.stranger, threshold),
      genuineTotal: sum(cell.genuine),
      nonmateTotal: sum(cell.impostor) + sum(cell.stranger),
    };
  }

  function render() {
    const cell = distributions[select.value] || {};
    const threshold = parseFloat(slider.value);
    output.textContent = threshold.toFixed(2);

    const isExact = Boolean(cell.exact);
    const { trueAccepts, falseAccepts, genuineTotal, nonmateTotal } =
      isExact ? exactAt(cell.exact, threshold) : binnedAt(cell, threshold);

    const tar = genuineTotal ? trueAccepts / genuineTotal : 0;
    const far = nonmateTotal ? falseAccepts / nonmateTotal : 0;
    // Precision is undefined, not zero, at a threshold that accepts nobody —
    // "0% of accepts are right" claims a wrongness that never happened. TAR
    // and FAR are honestly 0 there, since their denominators are populations.
    const accepts = trueAccepts + falseAccepts;
    const anyAccepts = accepts > 1e-9;
    const precision = anyAccepts ? trueAccepts / accepts : 0;
    const missed = genuineTotal - trueAccepts;

    // Exact counts are whole samples and are shown as such. A fractional
    // reading is the honest rendering of the interpolated fallback: "0.3 false
    // accepts" at least admits it is an estimate, where a rounded "0" would
    // quietly assert a fact the histograms cannot support.
    const asCount = (value) => (isExact ? String(value) : value.toFixed(1));

    stats.innerHTML = "";
    [
      ["TAR", (tar * 100).toFixed(1) + "%", "accepted of genuine", tar > 0.95 ? "good" : ""],
      ["FAR", (far * 100).toFixed(2) + "%", "accepted of non-mates", far > 0.01 ? "bad" : "good"],
      ["precision",
       anyAccepts ? (precision * 100).toFixed(1) + "%" : "—",
       anyAccepts ? "of accepts are right" : "nothing accepted here", ""],
      ["missed", asCount(missed), "genuine rejected", missed > 0 ? "bad" : "good"],
      ["false accepts", asCount(falseAccepts), "strangers let in", falseAccepts > 0 ? "bad" : "good"],
    ].forEach(([label, value, sub, tone]) => {
      const div = document.createElement("div");
      div.className = "eng-stat" + (tone ? " " + tone : "");
      div.innerHTML =
        '<span class="eng-stat-value">' + value + "</span>" +
        '<span class="eng-stat-label">' + label + "</span>" +
        '<span class="eng-stat-label" style="opacity:.6">' + sub + "</span>";
      stats.appendChild(div);
    });

    describeScope();
    draw(cell, threshold);
  }

  function draw(cell, threshold) {
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = 220;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // Left/right padding leaves room for the -1.0 and 1.0 tick labels, which
    // are centre-aligned on the axis ends and would otherwise clip.
    const pad = { left: 18, right: 18, top: 10, bottom: 26 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Share one y-scale across populations, normalised per population so the
    // 3000 strangers do not flatten the 65 genuine scores into the axis.
    const populations = ["stranger", "impostor", "genuine"];
    const x = (score) => pad.left + ((score + 1) / 2) * plotW;

    populations.forEach((name) => {
      const histogram = cell[name];
      if (!histogram) return;
      const total = sum(histogram) || 1;
      const peak = Math.max(...histogram.counts) / total || 1;

      ctx.beginPath();
      histogram.counts.forEach((count, i) => {
        const px = x((histogram.edges[i] + histogram.edges[i + 1]) / 2);
        const py = pad.top + plotH - (count / total / peak) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = COLORS[name];
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.lineTo(x(histogram.edges[histogram.edges.length - 1]), pad.top + plotH);
      ctx.lineTo(x(histogram.edges[0]), pad.top + plotH);
      ctx.closePath();
      ctx.fillStyle = COLORS[name] + "22";
      ctx.fill();
    });

    // Threshold line + accepted region.
    const tx = x(threshold);
    ctx.fillStyle = "rgba(124, 58, 237, 0.10)";
    ctx.fillRect(tx, pad.top, pad.left + plotW - tx, plotH);
    ctx.beginPath();
    ctx.moveTo(tx, pad.top);
    ctx.lineTo(tx, pad.top + plotH);
    ctx.strokeStyle = "#7c3aed";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#8a8a99";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    [-1, -0.5, 0, 0.5, 1].forEach((tick) => {
      ctx.fillText(tick.toFixed(1), x(tick), height - 10);
    });

    /* Shipped threshold marker. Only ever drawn on the encoder it was measured
       on — see onShippedEncoder: across encoders a cosine score is not the
       same quantity, so the line would be meaningless rather than merely
       unsupported. */
    if (hasShipped && onShippedEncoder()) {
      const decided = onShippedAggregation();
      const sx = x(shipped);
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(sx, pad.top);
      ctx.lineTo(sx, pad.top + plotH);
      ctx.strokeStyle = decided ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.2)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      markerLabel(
        "shipped " + shipped.toFixed(3) + (decided ? "" : " · not decided here"),
        sx,
        decided ? "#b9b9c8" : "#6a6a78");
    }

    ctx.textAlign = "left";
    ctx.fillStyle = "#7c3aed";
    // Only where it has room to sit without colliding with the shipped label.
    if (tx < width - 70) ctx.fillText("accept →", tx + 6, pad.top + 24);

    /* Label pinned to a vertical marker, with a short leader line and a panel
       behind the text so it stays readable over the distribution curves.
       Mirrors detection-explorer's markerLabel. */
    function markerLabel(text, lineX, color) {
      ctx.font = "10px 'JetBrains Mono', monospace";
      const textW = ctx.measureText(text).width;
      const fitsRight = lineX + 7 + textW + 3 <= pad.left + plotW;
      const textX = fitsRight ? lineX + 7 : lineX - 7 - textW;
      const midY = pad.top + 9;
      ctx.fillStyle = "#101018";
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
  }

  select.addEventListener("change", render);
  slider.addEventListener("input", render);
  window.addEventListener("resize", render);
  render();
})();
