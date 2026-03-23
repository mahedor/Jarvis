# Jarvis — Definitive Phase Roadmap (v4)

> Every single feature discussed, organized by phase. Last updated: March 21, 2026.
> 
> Tags: `[NEW]` = added during planning · `[LATEST]` = just added · `[CAREER]` = portfolio enhancer · `[DONE]` = completed

---

## Phase 1: Voice + Home Control (Bedroom) — 2-4 weeks

### Core
1. ~~Voice pipeline (Whisper STT → LLM → TTS)~~ — partially done: browser TTS + STT working, Whisper needs hardware `[DONE partial]`
2. Home Assistant + Zigbee smart devices (lights, blinds, fan) — needs hardware
3. Wake word — "Hey Jarvis" (OpenWakeWord) — needs hardware
4. Bedroom speaker system — needs hardware
5. ~~Jarvis orchestrator (Flask service with device state tracking)~~ `[DONE]`

### Demo Features (completed without hardware)
- ~~Browser-native TTS via Web Speech API (Edge recommended)~~ `[DONE]`
- ~~Speech input via browser SpeechRecognition API~~ `[DONE]`
- ~~Futuristic web UI with dark theme~~ `[DONE]`
- ~~6-color theme system (purple default)~~ `[DONE]`
- ~~Screensaver with 10 toggleable effects~~ `[DONE]`
- ~~Custom cursor (gray dot + accent trailing ring)~~ `[DONE]`
- ~~Claude streaming (internal, faster response generation)~~ `[DONE]`
- ~~Interaction logging to JSONL~~ `[DONE]`
- Voice mode — toggleable waveform/orb UI for voice-only interaction (no chat bubbles, just speak and listen) `[LATEST]`

### ⚡ NEXT UP
10. **Development agents — automated code quality pipeline** `[LATEST]`
    - See `jarvis-dev-agents.md` for full details, implementation code, and build order
    - **Core (run every change):**
      - Linter (ESLint + ruff) — catches syntax errors instantly (~15 min to build)
      - QA Agent (Playwright) — boots server + clicks through all UI + takes screenshots (~1-2 hrs to build)
    - **Extras (run when needed):**
      - Diff verification — catches missing code after refactors
      - Performance — page load, FPS, API response time
    - Single command runs full pipeline: `python run_agents.py`

### Career Enhancements
6. ~~Local intent classifier (rules + spaCy + embeddings)~~ `[DONE]` `[CAREER]` ⏳ *review code deeper*
7. Eval suite — test commands + accuracy tracking `[CAREER]`
8. ~~GitHub repo + ADRs + documentation~~ `[DONE]` `[CAREER]`
9. Local failsafe LLM (Llama 3.2 3B via Ollama) — offline fallback when internet drops `[LATEST]`

**Hardware needed:** Beelink Mini S12 Pro, ReSpeaker USB Mic Array, Zigbee dongle, smart bulbs, speaker

---

## Phase 2: Identity, Presence + Vision — 3-4 weeks

### Core
9. Facial recognition greeting (DeepFace/InsightFace)
10. Room presence detection + duration tracking (in bed, at desk, on couch, out of room) `[LATEST]`
    - Camera detects location/position, orchestrator logs timestamped state changes
    - Enables queries like "How long was I on the couch today?"
11. Multi-person face support
12. Activity recognition — sleeping, showering, getting ready, at desk `[NEW]`
13. Bed made / room cleanliness detection — custom trained classifier `[NEW]`
14. Speaker identification — who's talking, personality swap per person `[NEW]`
    - Train custom voice biometrics model (SpeechBrain / Resemblyzer)
    - Different system prompt per identified speaker
15. Cat escape detection + roommate alerts `[NEW]`
    - Object detection on entry camera
    - Cat near exit + door opens + cat gone from indoor feed = push notification
    - Cat returns inside = "Crisis averted" notification

### Architecture Upgrade
16. Migrate REST → MQTT message bus (Mosquitto) `[CAREER]`
17. ADR: document the REST → MQTT migration `[CAREER]`

**Hardware:** USB webcam or Pi Camera (entry + bedroom)
**Depends on:** Phase 1

---

## Phase 3: Calendar, Habits + Daily Intelligence — 2-3 weeks

### Core
18. Google Calendar integration (morning briefing, proactive nudges)
19. Habit app sync (API or export)
20. ~~Notion integration — read databases, voice queries~~ `[DONE]`
    - Reads Work Table (30-min focus blocks with productivity ratings)
    - Reads Start Table (session start tracking)
    - Data injected into system prompt for Claude to answer questions
    - Direct API calls with database ID targeting
21. Morning briefing routine (sleep, schedule, habits, weather, room status)
22. Smart alarm — Oura sleep stages + gradual light increase
23. Web search — Claude API web_search tool for live info `[NEW]`
24. Notion data visualization — charts for productivity trends, focus patterns `[LATEST]`

**Depends on:** Phase 1 (voice), Phase 2 (presence for wake detection)

---

## Phase 4: Personal Data Warehouse + Tracking — 3-4 weeks

### Infrastructure
25. Central data warehouse (Postgres + SQLAlchemy)
26. Multi-modal data pipeline (ETL, schema design) `[CAREER]`

### Data Sources
27. Sleep tracking — Oura Ring API V2 (stages, HRV, readiness)
28. Medication logging — voice or app, timestamped `[NEW]`
29. Location tracking — phone geofencing, labeled zones `[NEW]`
30. Screen time — Windows + Android via ActivityWatch `[NEW]`
31. Computer awareness agent — active window/app reporting `[NEW]`
32. Meal timing — voice log or kitchen camera
33. Spending tracker + savings goals `[NEW]`
    - Plaid API for bank transactions OR voice logging ("Jarvis, I spent $45 on gas")
    - Motorcycle fund + car fund progress tracking
34. Tesla vehicle integration (TeslaPy) `[NEW]`
35. Presence duration analytics — aggregate couch/desk/bed/room time from Phase 2 camera data `[LATEST]`

**Hardware:** Oura Ring (Gen 3/4 + membership)
**Depends on:** Phase 1 (voice for logging), Phase 3 (calendar/habits for correlation)

---

## Phase 5: Coaching, Reflection + Intelligence — 3-4 weeks

### Core
36. End-of-day check-in conversation
37. Pattern recognition + coaching ("You skip gym on 4+ meeting days")
38. Goal tracking + accountability (proactive nudges)
39. Proactive habit calls — Jarvis initiates check-ins `[LATEST]`
40. Medication impact reports — Benadryl effect on sleep, etc. `[NEW]`
41. Proactive outreach — Jarvis texts/calls your phone when you need intervention `[LATEST]`
    - Twilio API for SMS and voice calls
    - Triggered by: doom scrolling (ActivityWatch), missed tasks (calendar), broken patterns (no gym)
    - Auto-escalation: notification → text → phone call → app lock → screen lock
    - Phone locking via Android Device Admin API on companion app
    - Jarvis voice conversations during calls — talks you through what to do
    - Configurable frequency limits, quiet hours, snooze option
    - Research-backed: nudge theory, temporal discounting, varied interventions to prevent habituation
    - Nudge variation pools per level — Claude generates contextually unique messages to prevent habituation
    - Scaffolding withdrawal system — Jarvis gradually reduces intervention as behavior improves (SDT-based internalization)
    - Companion app lock screen shows Jarvis interface: why you're locked, unlock conditions, voice interaction

### Career Enhancements
42. RAG over personal data — vector search for coaching context `[CAREER]`
43. Statistical modeling — Bayesian inference for correlations `[CAREER]`
44. Evaluate MCP (Model Context Protocol) for multi-source data access `[CAREER]`
45. Per-user nudge experimentation framework — multi-armed bandit optimization on nudge strategies `[LATEST]` `[CAREER]`
    - Baseline measurement week per user (no nudges, just observe)
    - Randomized A/B testing across nudge types, tones, channels
    - Compliance rate, time-to-comply, relapse rate, escalation depth tracked per strategy
    - Epsilon-greedy or Thompson Sampling for real-time optimization
    - Cross-user insights improve starting defaults for new users

**Depends on:** Phase 4 (need 2-4 weeks of data history)

---

## Phase 6: Companion App (Phone + Wall Tablet) — 3-4 weeks

### Core
45. Phone app (React Native / Flutter) `[LATEST update]`
    - Companion app with Device Admin permissions for screen locking and app blocking
    - Receives commands from Jarvis server via Firebase Cloud Messaging
    - Commands: lock_screen, block_app, grayscale_on, show_message, play_tts
46. Wall tablet dashboard (kiosk mode web app)
47. Tablet screensaver / ambient display
    - Clock, weather, sleep score, subtle animation
    - Wakes to full Jarvis UI on tap or approach
48. Dynamic screensaver states based on system status `[LATEST]`
    - Red glow when something is down
    - Pulse effect when updating
    - Color/effect changes for notifications
49. Data visualizations (sleep trends, habits, spending, screen time)
50. Notification routing (push to phone + surface on tablet)
51. Multi-user onboarding system — goals, priorities, shortcomings, nudge preferences `[LATEST]`

### Career
52. Demo video — 2-3 min walkthrough for LinkedIn `[CAREER]`

**Hardware:** Cheap Android tablet + wall mount
**Depends on:** All previous phases

---

## Summary

| Phase | Features | Timeline | Status |
|---|---|---|---|
| Phase 1 | 8 core + demo features | 2-4 weeks | 🟡 Demo complete, needs hardware |
| Phase 2 | 9 | 3-4 weeks | ⬜ Planned |
| Phase 3 | 7 | 2-3 weeks | 🟡 Notion done, rest planned |
| Phase 4 | 11 | 3-4 weeks | ⬜ Planned |
| Phase 5 | 10 | 3-4 weeks | ⬜ Planned |
| Phase 6 | 8 | 3-4 weeks | ⬜ Planned |
| **Total** | **56 features** | **16-23 weeks** | |

Career enhancements: 13 items