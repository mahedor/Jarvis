// ═══════════════════════════════════════════════
// ─── Chat Mode Send ────────────────────────────
// ═══════════════════════════════════════════════
async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage('user', text);
  input.value = '';
  sendBtn.disabled = true;
  typing.style.display = 'block';
  suggestions.style.display = 'none';
  chat.scrollTop = chat.scrollHeight;

  try {
    // Fire both requests simultaneously — filler returns in ~5ms
    const chatPromise = fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const fillerRes = await fetch('/filler', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const fillerData = await fillerRes.json();
    if (fillerData.filler) speakText(fillerData.filler);

    const res = await chatPromise;
    const data = await res.json();
    typing.style.display = 'none';
    addMessage('jarvis', data.response, data.actions);
    syncDeviceChips(data.device_states);
    speakText(data.response);  // cancels filler if still playing
  } catch (err) {
    typing.style.display = 'none';
    addMessage('jarvis', 'Something went wrong. Check that the server is running.');
  }

  sendBtn.disabled = false;
  input.focus();
}

function useSuggestion(btn) {
  input.value = btn.textContent;
  sendMessage();
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});

input.focus();

// ═══════════════════════════════════════════════
// ─── Voice Mode Send ───────────────────────────
// ═══════════════════════════════════════════════
async function sendVoiceMessage(text) {
  if (!text.trim()) return;
  setVoiceState('thinking');

  try {
    // Fire both requests simultaneously — filler returns in ~5ms
    const chatPromise = fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const fillerRes = await fetch('/filler', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const fillerData = await fillerRes.json();
    if (fillerData.filler) speakText(fillerData.filler);

    const res = await chatPromise;
    const data = await res.json();
    syncDeviceChips(data.device_states);
    addMessage('user', text);
    addMessage('jarvis', data.response, data.actions);
    speakText(data.response);  // cancels filler if still playing
  } catch (err) {
    setVoiceState('idle');
  }
}
