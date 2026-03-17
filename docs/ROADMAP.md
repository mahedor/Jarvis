# Jarvis — Definitive Phase Roadmap (v4)

> Every single feature discussed, organized by phase. Last updated: March 16, 2026.
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

### Career Enhancements
6. Local intent classifier (DistilBERT fine-tuned) `[CAREER]`
7. Eval suite — test commands + accuracy tracking `[CAREER]`
8. ~~GitHub repo + ADRs + documentation~~ `[DONE]` `[CAREER]`

**Hardware needed:** Beelink Mini S12 Pro, ReSpeaker USB Mic Array, Zigbee dongle, smart bulbs, speaker

---

## Phase 2: Identity, Presence + Vision — 3-4 weeks

### Core
9. Facial recognition greeting (DeepFace/InsightFace)
10. Room presence detection (in bed, at desk, etc.)
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

**Hardware:** Oura Ring (Gen 3/4 + membership)
**Depends on:** Phase 1 (voice for logging), Phase 3 (calendar/habits for correlation)

---

## Phase 5: Coaching, Reflection + Intelligence — 3-4 weeks

### Core
35. End-of-day check-in conversation
36. Pattern recognition + coaching ("You skip gym on 4+ meeting days")
37. Goal tracking + accountability (proactive nudges)
38. Proactive habit calls — Jarvis initiates check-ins `[LATEST]`
39. Medication impact reports — Benadryl effect on sleep, etc. `[NEW]`

### Career Enhancements
40. RAG over personal data — vector search for coaching context `[CAREER]`
41. Statistical modeling — Bayesian inference for correlations `[CAREER]`

**Depends on:** Phase 4 (need 2-4 weeks of data history)

---

## Phase 6: Companion App (Phone + Wall Tablet) — 3-4 weeks

### Core
42. Phone app (React Native / Flutter)
43. Wall tablet dashboard (kiosk mode web app)
44. Tablet screensaver / ambient display
    - Clock, weather, sleep score, subtle animation
    - Wakes to full Jarvis UI on tap or approach
45. Dynamic screensaver states based on system status `[LATEST]`
    - Red glow when something is down
    - Pulse effect when updating
    - Color/effect changes for notifications
46. Data visualizations (sleep trends, habits, spending, screen time)
47. Notification routing (push to phone + surface on tablet)

### Career
48. Demo video — 2-3 min walkthrough for LinkedIn `[CAREER]`

**Hardware:** Cheap Android tablet + wall mount
**Depends on:** All previous phases

---

## Summary

| Phase | Features | Timeline | Status |
|---|---|---|---|
| Phase 1 | 8 core + demo features | 2-4 weeks | 🟡 Demo complete, needs hardware |
| Phase 2 | 9 | 3-4 weeks | ⬜ Planned |
| Phase 3 | 7 | 2-3 weeks | 🟡 Notion done, rest planned |
| Phase 4 | 10 | 3-4 weeks | ⬜ Planned |
| Phase 5 | 7 | 3-4 weeks | ⬜ Planned |
| Phase 6 | 7 | 3-4 weeks | ⬜ Planned |
| **Total** | **48 features** | **16-23 weeks** | |

Career enhancements: 9 items