/* ─────────────────────────────────────────────────────────────
   Threshold explorer — recognition engineering page.

   Draws the three score populations (genuine / impostor / stranger) and
   recomputes the accept/reject trade as you move the threshold.

   BINNED APPROXIMATION, ON PURPOSE. The benchmark stores 40-bin histograms
   over cosine [-1, 1], not raw scores, so counts are resolved to 0.05. A bin
   straddling the threshold is split by linear interpolation rather than
   assigned whole, which keeps the curves smooth as you drag; it is still an
   approximation and the page says so. The exact figures live in the table.
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

  function render() {
    const cell = distributions[select.value] || {};
    const threshold = parseFloat(slider.value);
    output.textContent = threshold.toFixed(2);

    const genuineTotal = sum(cell.genuine);
    const impostorTotal = sum(cell.impostor);
    const strangerTotal = sum(cell.stranger);

    const trueAccepts = countAbove(cell.genuine, threshold);
    const falseImpostor = countAbove(cell.impostor, threshold);
    const falseStranger = countAbove(cell.stranger, threshold);
    const falseAccepts = falseImpostor + falseStranger;
    const nonmateTotal = impostorTotal + strangerTotal;

    const tar = genuineTotal ? trueAccepts / genuineTotal : 0;
    const far = nonmateTotal ? falseAccepts / nonmateTotal : 0;
    const precision = trueAccepts + falseAccepts
      ? trueAccepts / (trueAccepts + falseAccepts)
      : 0;
    const missed = genuineTotal - trueAccepts;

    stats.innerHTML = "";
    [
      ["TAR", (tar * 100).toFixed(1) + "%", "accepted of genuine", tar > 0.95 ? "good" : ""],
      ["FAR", (far * 100).toFixed(2) + "%", "accepted of non-mates", far > 0.01 ? "bad" : "good"],
      ["precision", (precision * 100).toFixed(1) + "%", "of accepts are right", ""],
      ["missed", missed.toFixed(1), "genuine rejected", missed > 0 ? "bad" : "good"],
      ["false accepts", falseAccepts.toFixed(1), "strangers let in", falseAccepts > 0 ? "bad" : "good"],
    ].forEach(([label, value, sub, tone]) => {
      const div = document.createElement("div");
      div.className = "eng-stat" + (tone ? " " + tone : "");
      div.innerHTML =
        '<span class="eng-stat-value">' + value + "</span>" +
        '<span class="eng-stat-label">' + label + "</span>" +
        '<span class="eng-stat-label" style="opacity:.6">' + sub + "</span>";
      stats.appendChild(div);
    });

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
    ctx.textAlign = "left";
    ctx.fillStyle = "#7c3aed";
    if (tx < width - 70) ctx.fillText("accept →", tx + 6, pad.top + 12);
  }

  select.addEventListener("change", render);
  slider.addEventListener("input", render);
  window.addEventListener("resize", render);
  render();
})();
