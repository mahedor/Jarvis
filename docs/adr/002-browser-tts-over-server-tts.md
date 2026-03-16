# ADR 002: Browser-native TTS over server-side TTS

**Date:** March 16, 2026
**Status:** Accepted

## Context

Jarvis needs text-to-speech to speak responses aloud. We evaluated multiple approaches:

1. **Windows SAPI (pyttsx3)** — local, built into Windows
2. **Server-side edge-tts** — calls Microsoft's cloud from Python, saves mp3, plays via pygame
3. **Browser Web Speech API** — browser speaks text directly, no server involvement

## Decision

Use the browser's Web Speech API for TTS. Recommend Microsoft Edge for best voice quality.

## Reasoning

Server-side edge-tts added ~1-2 seconds of overhead after Claude's response: cloud API call (~100ms), audio generation (~500-1000ms), file save to disk (~50ms), file load + playback startup (~200ms). Six steps between "response ready" and "audio starts."

Browser Web Speech API eliminates ALL of those steps. `speechSynthesis.speak()` starts audio in ~10ms. The browser's speech engine is always loaded and ready.

| Approach | Time after Claude responds | Voice quality | Dependencies |
|----------|---------------------------|---------------|-------------|
| Windows SAPI | ~200ms | Robotic | pyttsx3 (broken threading) |
| Server-side edge-tts | ~1-2 seconds | Great (neural) | edge-tts, pygame |
| Browser Web Speech API | ~10ms | Great on Edge, decent on Chrome | None |

## Tradeoffs

- Voice quality varies by browser. Edge is best (Microsoft neural voices), Chrome is decent, Firefox uses robotic OS voices.
- Less control over voice parameters from Python — voice selection happens in JavaScript.
- Won't work headless (no browser). For the mini PC deployment, production TTS engine is TBD — evaluating Kokoro-82M (local neural), ElevenLabs (cloud, best quality), and future speech-to-speech models. Decision will be made as a separate ADR when hardware is ready.

## Consequences

- Removed edge-tts, pygame, and pyttsx3 as dependencies. Only need anthropic, flask, python-dotenv.
- TTS logic moved from Python to ~30 lines of JavaScript in the frontend.
- Demo feels significantly more responsive — audio starts almost immediately after text appears.
- This decision applies to the demo phase. Production TTS on the mini PC will be chosen separately — the bar is realistic, fast, and ideally local. Current candidates: Kokoro-82M, ElevenLabs, or whatever speech-to-speech models look like by then.