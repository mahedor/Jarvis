// ═══════════════════════════════════════════════
// ─── Screensaver ───────────────────────────────
// ═══════════════════════════════════════════════
const screensaver = document.getElementById('screensaver');
const ssCore = document.getElementById('ss-cursor-core');
const ssRing = document.getElementById('ss-cursor-ring');
let ssActive = true;
let ssClockInterval;
let ssAnimFrame;

// Cursor: core follows mouse exactly, ring trails behind with smooth lerp
let mouseX = -100, mouseY = -100;
let ringX = -100, ringY = -100;

screensaver.addEventListener('mousemove', e => {
  mouseX = e.clientX;
  mouseY = e.clientY;
  ssCore.style.left = mouseX + 'px';
  ssCore.style.top = mouseY + 'px';
});

screensaver.addEventListener('mouseenter', () => {
  ssCore.classList.remove('hidden');
  ssRing.classList.remove('hidden');
});
screensaver.addEventListener('mouseleave', () => {
  ssCore.classList.add('hidden');
  ssRing.classList.add('hidden');
});

function animateRing() {
  ringX += (mouseX - ringX) * 0.12;
  ringY += (mouseY - ringY) * 0.12;
  ssRing.style.left = ringX + 'px';
  ssRing.style.top = ringY + 'px';
  if (ssActive) requestAnimationFrame(animateRing);
}
animateRing();

// Expand ring on hoverable elements
document.querySelectorAll('.ss-toggle, .ss-check, .theme-dot, .theme-trigger').forEach(el => {
  el.addEventListener('mouseenter', () => ssRing.classList.add('hover'));
  el.addEventListener('mouseleave', () => ssRing.classList.remove('hover'));
});

function updateSSClock() {
  const now = new Date();
  const h = now.getHours().toString().padStart(2, '0');
  const m = now.getMinutes().toString().padStart(2, '0');
  document.getElementById('ss-time').textContent = h + ':' + m;
  const days = ['sunday','monday','tuesday','wednesday','thursday','friday','saturday'];
  const months = ['january','february','march','april','may','june','july','august','september','october','november','december'];
  document.getElementById('ss-date').textContent = days[now.getDay()] + '  \u00b7  ' + months[now.getMonth()] + ' ' + now.getDate();
}

updateSSClock();
ssClockInterval = setInterval(updateSSClock, 10000);

// ─── Canvas Effects (particles + orbits + all 10) ───────────
const ssCanvas = document.getElementById('ss-canvas');
const ctx = ssCanvas.getContext('2d');
let particles = [];
let orbitDots = [];
let effectsOn = { particles: false, grid: false, pulse: false, orbits: false, warpgrid: false, neural: false, tendrils: false, starfield: false, circuits: false, helix: false };
let warpTime = 0;
let helixTime = 0;
let neuralTime = 0;

// Tendrils
let tendrilBranches = [];
let tendrilTimer = 0;

function spawnTendril(cx, cy) {
  const angle = Math.random() * Math.PI * 2;
  const startR = 125;
  tendrilBranches.push({
    points: [{ x: cx + Math.cos(angle) * startR, y: cy + Math.sin(angle) * startR }],
    angle: angle,
    speed: 0.8 + Math.random() * 1.2,
    curl: (Math.random() - 0.5) * 0.06,
    life: 1,
    fade: 0.002 + Math.random() * 0.003,
    thickness: 0.3 + Math.random() * 0.8,
    branched: false,
    depth: 0,
  });
}

function branchTendril(parent) {
  if (parent.depth > 2) return;
  const last = parent.points[parent.points.length - 1];
  for (let i = 0; i < 2; i++) {
    const spread = (Math.random() - 0.5) * 0.8;
    tendrilBranches.push({
      points: [{ x: last.x, y: last.y }],
      angle: parent.angle + spread,
      speed: parent.speed * (0.6 + Math.random() * 0.3),
      curl: (Math.random() - 0.5) * 0.08,
      life: parent.life * 0.7,
      fade: parent.fade * 1.3,
      thickness: parent.thickness * 0.6,
      branched: false,
      depth: parent.depth + 1,
    });
  }
}

// Neural network nodes
let neurons = [];
let synapses = [];

function initNeural() {
  neurons = [];
  synapses = [];
  const w = window.innerWidth;
  const h = window.innerHeight;
  const count = 35;
  for (let i = 0; i < count; i++) {
    neurons.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 1 + Math.random() * 1.5,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      energy: Math.random(),
      pulsePhase: Math.random() * Math.PI * 2,
    });
  }
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      const dx = neurons[i].x - neurons[j].x;
      const dy = neurons[i].y - neurons[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 250) {
        synapses.push({ from: i, to: j, dist: dist });
      }
    }
  }
}
initNeural();

// Starfield
let stars = [];
for (let i = 0; i < 200; i++) {
  stars.push({
    x: (Math.random() - 0.5) * 2,
    y: (Math.random() - 0.5) * 2,
    z: Math.random() * 1,
    pz: 0,
  });
}

// Circuit traces
let circuits = [];
let circuitTimer = 0;

function spawnCircuit(cx, cy) {
  const angle = Math.random() * Math.PI * 2;
  const dist = 130 + Math.random() * 40;
  circuits.push({
    points: [{ x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist }],
    dir: Math.floor(Math.random() * 4),
    steps: 0,
    maxSteps: 15 + Math.floor(Math.random() * 30),
    speed: 2 + Math.random() * 2,
    life: 1,
    fade: 0.003 + Math.random() * 0.005,
  });
}

function resizeSSCanvas() {
  ssCanvas.width = window.innerWidth;
  ssCanvas.height = window.innerHeight;
  initNeural();
}
resizeSSCanvas();
window.addEventListener('resize', () => { resizeSSCanvas(); resizeVoiceCanvas(); });

// Create particles
for (let i = 0; i < 60; i++) {
  particles.push({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    r: Math.random() * 1.5 + 0.5,
    o: Math.random() * 0.3 + 0.05,
  });
}

// Create orbit dots
for (let i = 0; i < 12; i++) {
  orbitDots.push({
    angle: (Math.PI * 2 / 12) * i + Math.random() * 0.5,
    radius: 150 + Math.random() * 80,
    speed: 0.002 + Math.random() * 0.003,
    dir: Math.random() > 0.5 ? 1 : -1,
    size: Math.random() * 2 + 1,
    o: Math.random() * 0.3 + 0.1,
  });
}

function getAccentRGB() {
  const s = getComputedStyle(document.documentElement).getPropertyValue('--accent-rgb').trim();
  const parts = s.split(',').map(Number);
  return parts.length === 3 ? parts : [124, 58, 237];
}

function drawEffects() {
  if (!ssActive) return;
  ctx.clearRect(0, 0, ssCanvas.width, ssCanvas.height);
  const ring = document.querySelector('.ss-ring');
  const rRect = ring.getBoundingClientRect();
  const cx = rRect.left + rRect.width / 2;
  const cy = rRect.top + rRect.height / 2;
  const rgb = getAccentRGB();

  if (effectsOn.particles) {
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = ssCanvas.width;
      if (p.x > ssCanvas.width) p.x = 0;
      if (p.y < 0) p.y = ssCanvas.height;
      if (p.y > ssCanvas.height) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + p.o + ')';
      ctx.fill();
    });
  }

  if (effectsOn.orbits) {
    orbitDots.forEach(d => {
      d.angle += d.speed * d.dir;
      const x = cx + Math.cos(d.angle) * d.radius;
      const y = cy + Math.sin(d.angle) * d.radius;
      ctx.beginPath();
      ctx.arc(x, y, d.size, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + d.o + ')';
      ctx.fill();
    });
  }

  if (effectsOn.warpgrid) {
    warpTime += 0.008;
    const w = ssCanvas.width;
    const h = ssCanvas.height;
    const spacing = 50;
    const warpStrength = 40;
    ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',0.07)';
    ctx.lineWidth = 0.5;

    for (let row = -2; row <= h / spacing + 2; row++) {
      ctx.beginPath();
      for (let col = 0; col <= w; col += 4) {
        const baseY = row * spacing;
        const dx = col - cx;
        const dy = baseY - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const warp = warpStrength * Math.sin(dist * 0.01 - warpTime * 3) / (1 + dist * 0.005);
        const fy = baseY + warp;
        if (col === 0) ctx.moveTo(col, fy);
        else ctx.lineTo(col, fy);
      }
      ctx.stroke();
    }

    for (let col = -2; col <= w / spacing + 2; col++) {
      ctx.beginPath();
      for (let row = 0; row <= h; row += 4) {
        const baseX = col * spacing;
        const dx = baseX - cx;
        const dy = row - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const warp = warpStrength * Math.sin(dist * 0.01 - warpTime * 3) / (1 + dist * 0.005);
        const fx = baseX + warp;
        if (row === 0) ctx.moveTo(fx, row);
        else ctx.lineTo(fx, row);
      }
      ctx.stroke();
    }
  }

  // Neural Network
  if (effectsOn.neural) {
    neuralTime += 0.01;

    neurons.forEach(n => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < -20) n.x = ssCanvas.width + 20;
      if (n.x > ssCanvas.width + 20) n.x = -20;
      if (n.y < -20) n.y = ssCanvas.height + 20;
      if (n.y > ssCanvas.height + 20) n.y = -20;
      n.energy = 0.3 + Math.sin(n.pulsePhase + neuralTime * 2) * 0.3;
    });

    synapses.forEach(s => {
      const a = neurons[s.from];
      const b = neurons[s.to];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 280) return;
      const alpha = (1 - dist / 280) * 0.25;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      const mx = (a.x + b.x) / 2 + (a.y - b.y) * 0.08;
      const my = (a.y + b.y) / 2 + (b.x - a.x) * 0.08;
      ctx.quadraticCurveTo(mx, my, b.x, b.y);
      ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + alpha + ')';
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });

    neurons.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + (0.2 + n.energy * 0.25) + ')';
      ctx.fill();
    });
  }

  // Tendrils
  if (effectsOn.tendrils) {
    tendrilTimer++;
    if (tendrilTimer % 40 === 0 && tendrilBranches.filter(t => t.depth === 0).length < 8) {
      spawnTendril(cx, cy);
    }

    tendrilBranches.forEach(t => {
      if (t.life > 0.2) {
        const last = t.points[t.points.length - 1];
        t.angle += t.curl;
        const nx = last.x + Math.cos(t.angle) * t.speed;
        const ny = last.y + Math.sin(t.angle) * t.speed;
        t.points.push({ x: nx, y: ny });

        if (!t.branched && t.points.length > 15 && Math.random() < 0.02 && t.depth < 3) {
          t.branched = true;
          branchTendril(t);
        }
      }

      t.life -= t.fade;

      if (t.points.length > 1 && t.life > 0) {
        ctx.beginPath();
        ctx.moveTo(t.points[0].x, t.points[0].y);
        for (let i = 1; i < t.points.length; i++) {
          ctx.lineTo(t.points[i].x, t.points[i].y);
        }
        ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, t.life * 0.3) + ')';
        ctx.lineWidth = t.thickness;
        ctx.stroke();

        if (t.life > 0.2) {
          const tip = t.points[t.points.length - 1];
          ctx.beginPath();
          ctx.arc(tip.x, tip.y, t.thickness + 1, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, t.life * 0.4) + ')';
          ctx.fill();
        }
      }
    });
    tendrilBranches = tendrilBranches.filter(t => t.life > 0);
  }

  // Starfield
  if (effectsOn.starfield) {
    const hw = ssCanvas.width / 2;
    const hh = ssCanvas.height / 2;
    stars.forEach(s => {
      s.pz = s.z;
      s.z -= 0.005;
      if (s.z <= 0) { s.z = 1; s.pz = 1; s.x = (Math.random() - 0.5) * 2; s.y = (Math.random() - 0.5) * 2; }
      const sx = (s.x / s.z) * hw + hw;
      const sy = (s.y / s.z) * hh + hh;
      const px = (s.x / s.pz) * hw + hw;
      const py = (s.y / s.pz) * hh + hh;
      const size = (1 - s.z) * 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(sx, sy);
      ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + (1 - s.z) * 0.5 + ')';
      ctx.lineWidth = size;
      ctx.stroke();
    });
  }

  // Circuit Traces
  if (effectsOn.circuits) {
    circuitTimer++;
    if (circuitTimer % 20 === 0 && circuits.length < 30) spawnCircuit(cx, cy);
    const dirs = [[1,0],[0,1],[-1,0],[0,-1]];
    circuits.forEach(c => {
      if (c.steps < c.maxSteps) {
        const last = c.points[c.points.length - 1];
        if (Math.random() < 0.2) c.dir = Math.floor(Math.random() * 4);
        const d = dirs[c.dir];
        c.points.push({ x: last.x + d[0] * c.speed * 3, y: last.y + d[1] * c.speed * 3 });
        c.steps++;
      }
      c.life -= c.fade;
      if (c.points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(c.points[0].x, c.points[0].y);
        for (let i = 1; i < c.points.length; i++) ctx.lineTo(c.points[i].x, c.points[i].y);
        ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, c.life * 0.4) + ')';
        ctx.lineWidth = 0.8;
        ctx.stroke();
        const tip = c.points[c.points.length - 1];
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, c.life * 0.6) + ')';
        ctx.fill();
      }
    });
    circuits = circuits.filter(c => c.life > 0);
  }

  // DNA Helix
  if (effectsOn.helix) {
    helixTime += 0.015;
    const helixX = ssCanvas.width - 60;
    const helixH = ssCanvas.height;
    const nodes = 40;
    const amp = 20;
    const spacing = helixH / nodes;
    for (let i = 0; i < nodes; i++) {
      const yy = i * spacing;
      const phase = (i * 0.3) + helixTime;
      const x1 = helixX + Math.sin(phase) * amp;
      const x2 = helixX + Math.sin(phase + Math.PI) * amp;
      const depth1 = (Math.sin(phase) + 1) / 2;
      const depth2 = (Math.sin(phase + Math.PI) + 1) / 2;
      ctx.beginPath();
      ctx.moveTo(x1, yy);
      ctx.lineTo(x2, yy);
      ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',0.06)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x1, yy, 2 + depth1, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + (0.1 + depth1 * 0.25) + ')';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x2, yy, 2 + depth2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + (0.1 + depth2 * 0.25) + ')';
      ctx.fill();
    }
  }

  ssAnimFrame = requestAnimationFrame(drawEffects);
}

drawEffects();

// ─── Effect toggles ────────────────────────────
function toggleEffect(name, el) {
  effectsOn[name] = !effectsOn[name];
  el.classList.toggle('on', effectsOn[name]);

  if (name === 'grid') document.getElementById('ss-grid').classList.toggle('active', effectsOn[name]);
  if (name === 'pulse') document.getElementById('ss-pulse').classList.toggle('active', effectsOn[name]);
}

// Prevent control clicks from dismissing screensaver
const ssControls = document.getElementById('ss-controls');
ssControls.addEventListener('click', e => e.stopPropagation());
ssControls.addEventListener('touchstart', e => e.stopPropagation());
ssControls.addEventListener('touchend', e => e.stopPropagation());

function dismissScreensaver(e) {
  if (!ssActive) return;
  if (ssControls.contains(e.target)) return;
  ssActive = false;
  screensaver.classList.add('dismissed');
  clearInterval(ssClockInterval);
  cancelAnimationFrame(ssAnimFrame);
  setTimeout(() => {
    fetch('/greeting', { method: 'POST' }).then(r => r.json()).then(d => speakText(d.text));
    screensaver.remove();
  }, 600);
}

document.addEventListener('click', dismissScreensaver);
document.addEventListener('keydown', dismissScreensaver);
document.addEventListener('touchstart', dismissScreensaver);
