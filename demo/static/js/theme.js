// ─── Theme Switching ───────────────────────────
function setTheme(hex, rgb, dot) {
  currentAccent = hex;
  accentRGB = rgb.split(',').map(Number);
  document.documentElement.style.setProperty('--accent', hex);
  document.documentElement.style.setProperty('--accent-rgb', rgb);
  document.querySelectorAll('.mic-btn svg, .voice-mic-btn svg').forEach(svg => {
    if (!svg.closest('.recording')) svg.setAttribute('stroke', hex);
  });
  document.getElementById('theme-trigger').style.background = hex;
  document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('active'));
  if (dot) dot.classList.add('active');
}

// ─── Mode Switching ────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById('mode-chat').classList.toggle('active', mode === 'chat');
  document.getElementById('mode-voice').classList.toggle('active', mode === 'voice');
  document.getElementById('chat-mode').classList.toggle('hidden', mode === 'voice');
  document.getElementById('voice-mode').classList.toggle('active', mode === 'voice');

  if (mode === 'voice') {
    resizeVoiceCanvas();
    if (!voiceAnimRunning) { voiceAnimRunning = true; drawVoice(); }
  }
  if (mode === 'chat') { input.focus(); }
}
