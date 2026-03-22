// ═══════════════════════════════════════════════
// ─── Shared State ──────────────────────────────
// ═══════════════════════════════════════════════
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const typing = document.getElementById('typing');
const sendBtn = document.getElementById('send');
const suggestions = document.getElementById('suggestions');
let currentAccent = '#7c3aed';
let accentRGB = [124, 58, 237];
let currentMode = 'chat';


// ─── Device Chips ──────────────────────────────
const deviceChipMap = {
  'light.bedroom': 'dev-light-bedroom',
  'light.bedroom_lamp': 'dev-light-lamp',
  'switch.bedroom_fan': 'dev-switch-fan',
  'cover.bedroom_blinds': 'dev-cover-blinds',
};

const deviceLabels = {
  'light.bedroom': 'bedroom light',
  'light.bedroom_lamp': 'bedroom lamp',
  'switch.bedroom_fan': 'bedroom fan',
  'cover.bedroom_blinds': 'blinds',
};

function syncDeviceChips(states) {
  if (!states) return;
  for (const [entityId, info] of Object.entries(states)) {
    const chipId = deviceChipMap[entityId];
    const chip = document.getElementById(chipId);
    if (!chip) continue;
    const label = deviceLabels[entityId] || entityId;
    const isOn = info.state === 'on' || info.state === 'open';
    let stateText = info.state;
    if (info.brightness && info.state === 'on') {
      const pct = Math.round(info.brightness / 255 * 100);
      stateText = 'on (' + pct + '%)';
    }
    chip.textContent = label + ': ' + stateText;
    chip.classList.toggle('on', isOn);
  }
}

function addMessage(role, text, actions) {
  const div = document.createElement('div');
  div.className = 'message ' + role;

  if (role === 'user') {
    div.textContent = text;
  } else {
    div.innerHTML = '<div class="label">Jarvis</div><div class="text">' + text + '</div>';
    if (actions && actions.length > 0) {
      actions.forEach(a => {
        const card = document.createElement('div');
        card.className = 'action-card';
        card.innerHTML = '<span class="bolt">&#9889;</span> ' + (a.service||'') + ' &rarr; ' + (a.entity_id||'');
        div.appendChild(card);
      });
    }
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
