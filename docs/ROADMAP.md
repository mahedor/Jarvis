# Jarvis — Definitive Phase Roadmap (v3)

> Every single feature discussed, organized by phase. Last updated: March 15, 2026.
> 
> Tags: `[NEW]` = added during planning · `[LATEST]` = just added · `[CAREER]` = portfolio enhancer

---

## Phase 1: Voice + Home Control (Bedroom) — 2-4 weeks

### Core
1. Voice pipeline (Whisper STT → LLM → Piper TTS)
2. Home Assistant + Zigbee smart devices (lights, blinds, fan)
3. Wake word — "Hey Jarvis" (OpenWakeWord)
4. Bedroom speaker system
5. Jarvis orchestrator (FastAPI service with device state tracking)

### Career Enhancements
6. Local intent classifier (DistilBERT fine-tuned) `[CAREER]`
7. Eval suite — test commands + accuracy tracking `[CAREER]`
8. GitHub repo + ADRs + documentation `[CAREER]`

**Hardware:** Beelink Mini S12 Pro, ReSpeaker USB Mic Array, Zigbee dongle, smart bulbs, speaker

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
15. Cat escape detection + roommate alerts `[LATEST]`
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
20. Notion integration — read/write databases, voice queries `[NEW]`
21. Morning briefing routine (sleep, schedule, habits, weather, room status)
22. Smart alarm — Oura sleep stages + gradual light increase
23. Web search — Claude API web_search tool for live info `[NEW]`
    - Weather, news, general questions
    - Makes morning briefing much more useful

**Depends on:** Phase 1 (voice), Phase 2 (presence for wake detection)

---

## Phase 4: Personal Data Warehouse + Tracking — 3-4 weeks

### Infrastructure
24. Central data warehouse (Postgres + SQLAlchemy)
25. Multi-modal data pipeline (ETL, schema design) `[CAREER]`

### Data Sources
26. Sleep tracking — Oura Ring API V2 (stages, HRV, readiness)
27. Medication logging — voice or app, timestamped `[NEW]`
28. Location tracking — phone geofencing, labeled zones `[NEW]`
29. Screen time — Windows + Android via ActivityWatch `[NEW]`
30. Computer awareness agent — active window/app reporting `[NEW]`
31. Meal timing — voice log or kitchen camera
32. Spending tracker + savings goals `[LATEST]`
    - Plaid API for bank transactions OR voice logging ("Jarvis, I spent $45 on gas")
    - Motorcycle fund + car fund progress tracking
    - "You've saved $3,200 of your $8,000 motorcycle goal. At this rate, September."

**Hardware:** Oura Ring (Gen 3/4 + membership)
**Depends on:** Phase 1 (voice for logging), Phase 3 (calendar/habits for correlation)

---

## Phase 5: Coaching, Reflection + Intelligence — 3-4 weeks

### Core
33. End-of-day check-in conversation
34. Pattern recognition + coaching ("You skip gym on 4+ meeting days")
35. Goal tracking + accountability (proactive nudges)
36. Medication impact reports — Benadryl effect on sleep, etc. `[NEW]`

### Career Enhancements
37. RAG over personal data — vector search for coaching context `[CAREER]`
38. Statistical modeling — Bayesian inference for correlations `[CAREER]`

**Depends on:** Phase 4 (need 2-4 weeks of data history)

---

## Phase 6: Companion App (Phone + Wall Tablet) — 3-4 weeks

### Core
39. Phone app (React Native / Flutter)
40. Wall tablet dashboard (kiosk mode web app)
41. Tablet screensaver / ambient display `[NEW]`
    - Clock, weather, sleep score, subtle animation
    - Wakes to full Jarvis UI on tap or approach
42. Data visualizations (sleep trends, habits, spending, screen time)
43. Notification routing (push to phone + surface on tablet)

### Career
44. Demo video — 2-3 min walkthrough for LinkedIn `[CAREER]`

**Hardware:** Cheap Android tablet + wall mount
**Depends on:** All previous phases

---

## Summary

| | Features | Timeline |
|---|---|---|
| Phase 1 | 8 | 2-4 weeks |
| Phase 2 | 9 | 3-4 weeks |
| Phase 3 | 6 | 2-3 weeks |
| Phase 4 | 9 | 3-4 weeks |
| Phase 5 | 6 | 3-4 weeks |
| Phase 6 | 6 | 3-4 weeks |
| **Total** | **44 features** | **16-23 weeks** |

Career enhancements: 9 items
New items added during planning: 15 items
