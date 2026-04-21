import js from "@eslint/js";
import globals from "globals";

export default [
  {
    files: ["demo/static/js/**/*.js"],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: "script",
      globals: {
        ...globals.browser,
        // ── Cross-file globals ─────────────────────────────────────
        // These are defined in one script and used in others — all
        // loaded into the same page scope via <script> tags.
        // Declaring them here lets no-undef catch typos in consuming files.
        // no-redeclare is disabled below so the defining file can also
        // declare them without ESLint complaining.
        //
        // state.js
        chat: "readonly",
        input: "readonly",
        typing: "readonly",
        sendBtn: "readonly",
        suggestions: "readonly",
        currentAccent: "writable",   // theme.js reassigns
        accentRGB: "writable",       // theme.js reassigns
        currentMode: "writable",     // theme.js reassigns
        syncDeviceChips: "readonly",
        addMessage: "readonly",
        // tts.js
        speakText: "readonly",
        toggleTTS: "readonly",
        recognition: "writable",     // tts.js assigns on init
        isRecording: "writable",     // tts.js toggles
        ttsSpeaking: "writable",     // tts.js toggles
        toggleMic: "readonly",
        toggleVoiceMic: "readonly",
        startMic: "readonly",
        stopAllMics: "readonly",
        // voice-engine.js
        voiceAnimRunning: "writable",
        voiceState: "writable",      // setVoiceState reassigns
        voiceStyle: "writable",      // setVoiceStyle reassigns
        drawVoice: "readonly",
        resizeVoiceCanvas: "readonly",
        setVoiceState: "readonly",
        setVoiceStyle: "readonly",
        // chat.js
        sendMessage: "readonly",
        sendVoiceMessage: "readonly",
        useSuggestion: "readonly",
        // screensaver.js
        toggleEffect: "readonly",
        dismissScreensaver: "readonly",
        // theme.js
        setTheme: "readonly",
        setMode: "readonly",
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      "no-console": "off",
      // Empty catch blocks are used intentionally for cleanup (e.g. recognition.stop())
      "no-empty": ["error", { allowEmptyCatch: true }],
      // Only flag unused vars in local scopes — top-level functions are shared across
      // files via global scope and would otherwise all appear "unused" in the file
      // that defines them. Args and caught errors are never flagged.
      "no-unused-vars": ["error", { vars: "local", args: "none", caughtErrors: "none" }],
      // Multi-script browser apps intentionally re-declare globals (the defining file
      // uses const/let/function while consuming files rely on the global scope).
      "no-redeclare": "off",
    },
  },
];
