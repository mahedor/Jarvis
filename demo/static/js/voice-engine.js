// ═══════════════════════════════════════════════
// ─── Voice Waveform Engine ─────────────────────
// ═══════════════════════════════════════════════
const voiceCanvas = document.getElementById('voice-canvas');
const vCtx = voiceCanvas.getContext('2d');
let voiceStyle = 'waveform';
let voiceState = 'idle';
let voiceAnimRunning = false;
let vTime = 0;

function resizeVoiceCanvas() {
  const area = voiceCanvas.parentElement;
  if (!area) return;
  voiceCanvas.width = area.clientWidth * devicePixelRatio;
  voiceCanvas.height = area.clientHeight * devicePixelRatio;
  vCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}
resizeVoiceCanvas();

function setVoiceStyle(name, el) {
  voiceStyle = name;
  document.querySelectorAll('.voice-style-chip').forEach(c => c.classList.remove('active'));
  if (el) el.classList.add('active');
}

function setVoiceState(state) {
  voiceState = state;
  const label = document.getElementById('voice-status');
  if (label) {
    label.textContent = state;
    label.classList.toggle('active-state', state !== 'idle');
  }
}

function getVAmp() {
  switch (voiceState) {
    case 'idle': return 0.03 + Math.sin(vTime * 0.8) * 0.02;
    case 'listening': return 0.2 + Math.sin(vTime * 3) * 0.15 + Math.sin(vTime * 7.3) * 0.08 + Math.random() * 0.1;
    case 'thinking': return 0.08 + Math.sin(vTime * 1.5) * 0.05;
    case 'speaking': return 0.25 + Math.sin(vTime * 4.5) * 0.12 + Math.sin(vTime * 11) * 0.06 + Math.sin(vTime * 2.1) * 0.08;
    default: return 0.05;
  }
}

function getVFreqData(count) {
  const out = [];
  const amp = getVAmp();
  for (let i = 0; i < count; i++) {
    const f = i / count;
    const base = amp * (1 - f * 0.6);
    const wave = Math.sin(vTime * 3 + i * 0.4) * 0.3 + Math.sin(vTime * 7 + i * 0.8) * 0.15;
    out.push(Math.max(0, Math.min(1, base + wave * amp)));
  }
  return out;
}

// ─── Main draw loop ────────────────────────────
function drawVoice() {
  if (currentMode !== 'voice') { voiceAnimRunning = false; return; }
  vTime += 0.016;
  const w = voiceCanvas.width / devicePixelRatio;
  const h = voiceCanvas.height / devicePixelRatio;
  const cx = w / 2;
  const cy = h / 2;
  const r = accentRGB;
  const amp = getVAmp();

  vCtx.clearRect(0, 0, w, h);

  switch (voiceStyle) {
    case 'waveform': vDrawWaveform(w, h, cx, cy, r, amp); break;
    case 'bars': vDrawBars(w, h, cx, cy, r, amp); break;
    case 'orb': vDrawOrb(w, h, cx, cy, r, amp); break;
    case 'ring': vDrawRing(w, h, cx, cy, r, amp); break;
    case 'pulse': vDrawPulse(w, h, cx, cy, r, amp); break;
    case 'spiral': vDrawSpiral(w, h, cx, cy, r, amp); break;
  }

  requestAnimationFrame(drawVoice);
}

// ── 1. Waveform (oscilloscope) ──────────────────
function vDrawWaveform(w, h, cx, cy, r, amp) {
  const count = 128;
  const spread = w * 0.6;
  const startX = cx - spread / 2;

  function yAt(i) {
    return cy + Math.sin(vTime * 2.5 + i * 0.08) * amp * 120
              + Math.sin(vTime * 4.1 + i * 0.15) * amp * 60
              + Math.sin(vTime * 6.7 + i * 0.05) * amp * 30;
  }

  // Glow layer
  vCtx.beginPath();
  for (let i = 0; i < count; i++) {
    const x = startX + (i / count) * spread;
    if (i === 0) vCtx.moveTo(x, yAt(i));
    else vCtx.lineTo(x, yAt(i));
  }
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.15)`;
  vCtx.lineWidth = 8;
  vCtx.stroke();

  // Main line
  vCtx.beginPath();
  for (let i = 0; i < count; i++) {
    const x = startX + (i / count) * spread;
    if (i === 0) vCtx.moveTo(x, yAt(i));
    else vCtx.lineTo(x, yAt(i));
  }
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.7)`;
  vCtx.lineWidth = 2;
  vCtx.stroke();

  // Mirror line (fainter)
  vCtx.beginPath();
  for (let i = 0; i < count; i++) {
    const x = startX + (i / count) * spread;
    const y = cy - (yAt(i) - cy) * 0.4;
    if (i === 0) vCtx.moveTo(x, y);
    else vCtx.lineTo(x, y);
  }
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.2)`;
  vCtx.lineWidth = 1;
  vCtx.stroke();

  // Center line
  vCtx.beginPath();
  vCtx.moveTo(startX, cy);
  vCtx.lineTo(startX + spread, cy);
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.06)`;
  vCtx.lineWidth = 1;
  vCtx.stroke();
}

// ── 2. Bars (frequency visualizer) ──────────────
function vDrawBars(w, h, cx, cy, r, amp) {
  const count = 48;
  const barWidth = 4;
  const gap = 4;
  const totalWidth = count * (barWidth + gap) - gap;
  const startX = cx - totalWidth / 2;
  const maxH = h * 0.35;
  const freq = getVFreqData(count);

  for (let i = 0; i < count; i++) {
    const x = startX + i * (barWidth + gap);
    const barH = Math.max(2, freq[i] * maxH);

    // Upward bar
    const grad = vCtx.createLinearGradient(x, cy, x, cy - barH);
    grad.addColorStop(0, `rgba(${r.join(',')}, 0.6)`);
    grad.addColorStop(1, `rgba(${r.join(',')}, 0.15)`);
    vCtx.fillStyle = grad;
    vCtx.fillRect(x, cy - barH, barWidth, barH);

    // Reflection
    const grad2 = vCtx.createLinearGradient(x, cy, x, cy + barH * 0.4);
    grad2.addColorStop(0, `rgba(${r.join(',')}, 0.2)`);
    grad2.addColorStop(1, `rgba(${r.join(',')}, 0)`);
    vCtx.fillStyle = grad2;
    vCtx.fillRect(x, cy + 2, barWidth, barH * 0.4);

    // Top cap
    vCtx.fillStyle = `rgba(${r.join(',')}, 0.9)`;
    vCtx.fillRect(x, cy - barH - 2, barWidth, 2);
  }

  // Center line
  vCtx.beginPath();
  vCtx.moveTo(startX - 10, cy);
  vCtx.lineTo(startX + totalWidth + 10, cy);
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.06)`;
  vCtx.lineWidth = 1;
  vCtx.stroke();
}

// ── 3. Orb (morphing sphere) ────────────────────
function vDrawOrb(w, h, cx, cy, r, amp) {
  const baseRadius = 80;
  const points = 120;

  // Ambient glow
  const glowR = baseRadius + amp * 100 + 40;
  const glow = vCtx.createRadialGradient(cx, cy, baseRadius * 0.3, cx, cy, glowR);
  glow.addColorStop(0, `rgba(${r.join(',')}, ${0.08 + amp * 0.1})`);
  glow.addColorStop(0.5, `rgba(${r.join(',')}, ${0.03 + amp * 0.04})`);
  glow.addColorStop(1, 'rgba(0,0,0,0)');
  vCtx.fillStyle = glow;
  vCtx.fillRect(0, 0, w, h);

  // Morphing shape — multiple layers
  for (let layer = 2; layer >= 0; layer--) {
    const layerScale = 1 + layer * 0.15;
    const layerAlpha = layer === 0 ? 0.6 : layer === 1 ? 0.2 : 0.08;

    vCtx.beginPath();
    for (let i = 0; i <= points; i++) {
      const angle = (i / points) * Math.PI * 2;
      const noise = Math.sin(angle * 3 + vTime * 2) * amp * 40
                  + Math.sin(angle * 5 + vTime * 3.3) * amp * 20
                  + Math.sin(angle * 7 + vTime * 1.7) * amp * 10;
      const radius = (baseRadius + noise) * layerScale;
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      if (i === 0) vCtx.moveTo(x, y);
      else vCtx.lineTo(x, y);
    }
    vCtx.closePath();

    if (layer === 0) {
      const fill = vCtx.createRadialGradient(cx, cy - 20, 10, cx, cy, baseRadius + amp * 50);
      fill.addColorStop(0, `rgba(${r.join(',')}, 0.12)`);
      fill.addColorStop(1, `rgba(${r.join(',')}, 0.02)`);
      vCtx.fillStyle = fill;
      vCtx.fill();
    }

    vCtx.strokeStyle = `rgba(${r.join(',')}, ${layerAlpha})`;
    vCtx.lineWidth = layer === 0 ? 2 : 1;
    vCtx.stroke();
  }

  // Inner core dot
  const coreR = 3 + amp * 8;
  vCtx.beginPath();
  vCtx.arc(cx, cy, coreR, 0, Math.PI * 2);
  vCtx.fillStyle = `rgba(${r.join(',')}, ${0.5 + amp * 0.4})`;
  vCtx.fill();
}

// ── 4. Ring (circular waveform) ─────────────────
function vDrawRing(w, h, cx, cy, r, amp) {
  const baseRadius = 100;
  const count = 180;

  // Glow
  const glow = vCtx.createRadialGradient(cx, cy, baseRadius - 20, cx, cy, baseRadius + 60);
  glow.addColorStop(0, `rgba(${r.join(',')}, 0.03)`);
  glow.addColorStop(0.5, `rgba(${r.join(',')}, ${0.02 + amp * 0.06})`);
  glow.addColorStop(1, 'rgba(0,0,0,0)');
  vCtx.fillStyle = glow;
  vCtx.fillRect(0, 0, w, h);

  // Base circle
  vCtx.beginPath();
  vCtx.arc(cx, cy, baseRadius, 0, Math.PI * 2);
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.08)`;
  vCtx.lineWidth = 1;
  vCtx.stroke();

  // Outer waveform
  vCtx.beginPath();
  for (let i = 0; i <= count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const wave = Math.sin(angle * 8 + vTime * 3) * amp * 35
               + Math.sin(angle * 13 + vTime * 5) * amp * 15
               + Math.sin(angle * 3 + vTime * 1.5) * amp * 20;
    const radius = baseRadius + wave;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    if (i === 0) vCtx.moveTo(x, y);
    else vCtx.lineTo(x, y);
  }
  vCtx.closePath();
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.6)`;
  vCtx.lineWidth = 1.5;
  vCtx.stroke();

  // Inner waveform
  vCtx.beginPath();
  for (let i = 0; i <= count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const wave = Math.sin(angle * 8 + vTime * 3) * amp * 25
               + Math.sin(angle * 13 + vTime * 5) * amp * 10;
    const radius = baseRadius - wave * 0.5;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    if (i === 0) vCtx.moveTo(x, y);
    else vCtx.lineTo(x, y);
  }
  vCtx.closePath();
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.2)`;
  vCtx.lineWidth = 1;
  vCtx.stroke();

  // Spinning dot
  const dotAngle = vTime * 1.2;
  const dx = cx + Math.cos(dotAngle) * baseRadius;
  const dy = cy + Math.sin(dotAngle) * baseRadius;
  vCtx.beginPath();
  vCtx.arc(dx, dy, 3 + amp * 4, 0, Math.PI * 2);
  vCtx.fillStyle = `rgba(${r.join(',')}, 0.8)`;
  vCtx.fill();
}

// ── 5. Pulse (expanding concentric rings) ───────
function vDrawPulse(w, h, cx, cy, r, amp) {
  const maxRings = 8;
  const maxRadius = Math.min(w, h) * 0.4;
  const speed = 1 + amp * 2;

  // Central glow
  const coreGlow = vCtx.createRadialGradient(cx, cy, 0, cx, cy, 50 + amp * 40);
  coreGlow.addColorStop(0, `rgba(${r.join(',')}, ${0.15 + amp * 0.2})`);
  coreGlow.addColorStop(1, 'rgba(0,0,0,0)');
  vCtx.fillStyle = coreGlow;
  vCtx.fillRect(0, 0, w, h);

  for (let i = 0; i < maxRings; i++) {
    const phase = ((vTime * speed * 0.3 + i / maxRings) % 1);
    const radius = phase * maxRadius;
    const alpha = (1 - phase) * (0.3 + amp * 0.3);

    vCtx.beginPath();
    const segments = 60;
    for (let j = 0; j <= segments; j++) {
      const angle = (j / segments) * Math.PI * 2;
      const distort = amp * 15 * Math.sin(angle * 4 + vTime * 2 + i);
      const rad = radius + distort;
      const x = cx + Math.cos(angle) * rad;
      const y = cy + Math.sin(angle) * rad;
      if (j === 0) vCtx.moveTo(x, y);
      else vCtx.lineTo(x, y);
    }
    vCtx.closePath();
    vCtx.strokeStyle = `rgba(${r.join(',')}, ${Math.max(0, alpha)})`;
    vCtx.lineWidth = 1.5 * (1 - phase) + 0.5;
    vCtx.stroke();
  }

  // Core dot
  const coreR = 5 + amp * 10;
  vCtx.beginPath();
  vCtx.arc(cx, cy, coreR, 0, Math.PI * 2);
  vCtx.fillStyle = `rgba(${r.join(',')}, ${0.6 + amp * 0.3})`;
  vCtx.fill();
}

// ── 6. Spiral ───────────────────────────────────
function vDrawSpiral(w, h, cx, cy, r, amp) {
  const turns = 4;
  const points = 300;
  const maxRadius = 140 + amp * 60;

  // Glow
  const glow = vCtx.createRadialGradient(cx, cy, 0, cx, cy, maxRadius);
  glow.addColorStop(0, `rgba(${r.join(',')}, 0.06)`);
  glow.addColorStop(1, 'rgba(0,0,0,0)');
  vCtx.fillStyle = glow;
  vCtx.fillRect(0, 0, w, h);

  // Main spiral
  vCtx.beginPath();
  for (let i = 0; i < points; i++) {
    const t = i / points;
    const angle = t * Math.PI * 2 * turns + vTime * 1.5;
    const radius = t * maxRadius;
    const wobble = Math.sin(t * 20 + vTime * 4) * amp * 20;
    const x = cx + Math.cos(angle) * (radius + wobble);
    const y = cy + Math.sin(angle) * (radius + wobble);
    if (i === 0) vCtx.moveTo(x, y);
    else vCtx.lineTo(x, y);
  }
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.5)`;
  vCtx.lineWidth = 1.5;
  vCtx.stroke();

  // Counter-rotating spiral
  vCtx.beginPath();
  for (let i = 0; i < points; i++) {
    const t = i / points;
    const angle = -t * Math.PI * 2 * turns - vTime * 1.2;
    const radius = t * maxRadius * 0.85;
    const wobble = Math.sin(t * 15 + vTime * 3) * amp * 15;
    const x = cx + Math.cos(angle) * (radius + wobble);
    const y = cy + Math.sin(angle) * (radius + wobble);
    if (i === 0) vCtx.moveTo(x, y);
    else vCtx.lineTo(x, y);
  }
  vCtx.strokeStyle = `rgba(${r.join(',')}, 0.15)`;
  vCtx.lineWidth = 1;
  vCtx.stroke();

  // Center
  vCtx.beginPath();
  vCtx.arc(cx, cy, 3 + amp * 5, 0, Math.PI * 2);
  vCtx.fillStyle = `rgba(${r.join(',')}, 0.7)`;
  vCtx.fill();
}
