// ═══════════════════════════════════════════════
// ─── Browser TTS (Web Speech API) ──────────────
// ═══════════════════════════════════════════════
let ttsOn = document.querySelector('meta[name="tts-enabled"]').content === 'true';
let ttsVoice = null;
let ttsSpeaking = false;

function initTTS() {
  const voices = speechSynthesis.getVoices();
  const prefs = ['ryan', 'george', 'ryanneural'];
  for (const p of prefs) {
    const v = voices.find(v => v.name.toLowerCase().includes(p));
    if (v) { ttsVoice = v; break; }
  }
  if (!ttsVoice) {
    ttsVoice = voices.find(v => v.lang.startsWith('en')) || voices[0];
  }
}

speechSynthesis.onvoiceschanged = initTTS;
initTTS();

function speakText(text) {
  if (!ttsOn || !text.trim()) return;
  speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  if (ttsVoice) utter.voice = ttsVoice;
  utter.rate = 1.05;
  utter.pitch = 0.95;
  utter.onstart = () => { ttsSpeaking = true; setVoiceState('speaking'); };
  utter.onend = () => { ttsSpeaking = false; setVoiceState('idle'); };
  speechSynthesis.speak(utter);
}

function toggleTTS() {
  ttsOn = !ttsOn;
  if (!ttsOn) speechSynthesis.cancel();
  const btn = document.getElementById('tts-toggle');
  btn.textContent = 'TTS: ' + (ttsOn ? 'on' : 'off');
  btn.classList.toggle('off', !ttsOn);
}

// ═══════════════════════════════════════════════
// ─── Speech Recognition (shared) ───────────────
// ═══════════════════════════════════════════════
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isRecording = false;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onresult = (event) => {
    let transcript = '';
    let isFinal = false;
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
      if (event.results[i].isFinal) isFinal = true;
    }
    if (currentMode === 'chat') input.value = transcript;
    if (isFinal) {
      stopAllMics();
      if (currentMode === 'chat') {
        sendMessage();
      } else {
        sendVoiceMessage(transcript);
      }
    }
  };

  recognition.onend = () => { stopAllMics(); };
  recognition.onerror = (e) => {
    console.log('Speech recognition error:', e.error);
    stopAllMics();
  };
}

// ─── Chat Mode Mic ─────────────────────────────
function toggleMic() {
  if (isRecording) { stopAllMics(); }
  else { startMic('chat'); }
}

// ─── Voice Mode Mic ────────────────────────────
function toggleVoiceMic() {
  if (isRecording) { stopAllMics(); }
  else { startMic('voice'); }
}

function startMic(mode) {
  if (!recognition) {
    alert('Speech recognition not supported in this browser. Try Edge or Chrome.');
    return;
  }
  speechSynthesis.cancel();
  isRecording = true;
  setVoiceState('listening');

  if (mode === 'chat') {
    document.getElementById('mic-btn').classList.add('recording');
    document.getElementById('mic-btn').querySelector('svg').setAttribute('stroke', '#e24b4a');
    input.placeholder = 'Listening...';
  } else {
    document.getElementById('voice-mic-btn').classList.add('recording');
    document.getElementById('voice-mic-btn').querySelector('svg').setAttribute('stroke', '#e24b4a');
  }
  recognition.start();
}

function stopAllMics() {
  isRecording = false;
  // Chat mic reset
  document.getElementById('mic-btn').classList.remove('recording');
  document.getElementById('mic-btn').querySelector('svg').setAttribute('stroke', currentAccent);
  input.placeholder = 'Talk to Jarvis or click the mic...';
  // Voice mic reset
  document.getElementById('voice-mic-btn').classList.remove('recording');
  document.getElementById('voice-mic-btn').querySelector('svg').setAttribute('stroke', currentAccent);

  if (!ttsSpeaking) setVoiceState('idle');
  try { recognition.stop(); } catch(e) {}
}
