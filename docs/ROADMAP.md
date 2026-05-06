# Jarvis — Definitive Phase Roadmap (v4)

> Every single feature discussed, organized by phase. Last updated: May 5, 2026.
> 
> Tags: `[NEW]` = added during planning · `[LATEST]` = just added · `[CAREER]` = portfolio enhancer · `[DONE]` = completed

---

## Project Scope (April 12, 2026)

Jarvis is a personal tool built by Michael, for Michael. Productization, multi-user deployment, and ethics-of-deployment-at-scale are explicitly out of scope. If those become relevant later, they'll be reopened then. For now, every design decision optimizes for "does this help Michael" — not "would this work for other users." The interview/portfolio value comes from the process and technical work, not from shipping a product.

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
- ~~Prompt caching on Tier 3 system prompt (cache_control ephemeral)~~ `[DONE April 12]`
- ~~Intent classifier hardening — negation guard, word-boundary fix, normalize unit tests, multi-pass prefix removal~~ `[DONE April]`
- Voice mode — toggleable waveform/orb UI for voice-only interaction (no chat bubbles, just speak and listen) `[LATEST]`
- Tier-aware filler phrases — spoken "Let me think..." for Tier 3 commands while Claude processes `[LATEST]`
    - Current: random from categorized pools (question/command/statement/complaint/reflection)
    - Upgrade: local Llama 3.2 3B generates contextual fillers in ~200ms (depends on Ollama setup)

### ⚡ NEXT UP
10. **Development agents — automated code quality pipeline** `[LATEST]`
    - See `jarvis-dev-agents.md` for full details, implementation code, and build order
    - **Core (run every change):**
      - ~~Linter (ESLint + ruff) — catches syntax errors instantly~~ `[DONE April 22]` — `run_lint.sh` wraps both, configs in `eslint.config.mjs` and `ruff.toml`
      - QA Agent (Playwright) — boots server + clicks through all UI + takes screenshots (~1-2 hrs to build)
    - **Extras (run when needed):**
      - Diff verification — catches missing code after refactors
      - Performance — page load, FPS, API response time
    - Single command runs full pipeline: `python run_agents.py`

### Career Enhancements
6. ~~Local intent classifier (rules + spaCy + embeddings)~~ `[DONE]` `[CAREER]` ⏳ *review code deeper*
7. Eval suite — test commands + accuracy tracking `[CAREER]` 🟡 *in progress*
    - `tests/eval/routing_eval.jsonl` — 160-case route-classification suite covering all 25 planned route types (home_assistant, calendar, gmail, notion, drone, vehicle, oura, spending, coaching, code_assist, etc.) plus adversarial and compound cases
    - Registry rule: every new capability must add cases here AND a sub-suite under `tests/eval/suites/`
    - Runner + per-suite results writer (`tests/eval/results/`) not yet built
8. ~~GitHub repo + ADRs + documentation~~ `[DONE]` `[CAREER]`
9. Local failsafe LLM (Llama 3.2 3B via Ollama) — offline fallback when internet drops `[LATEST]`
10. ~~Benchmarks — intent classifier (`tools/benchmark.py`, 1000x per command, per-tier aggregates → `logs/benchmarks.json`) and Claude API prompt-caching latency (`tools/benchmark_claude.py`, cache ON vs OFF, `--with-history` flag to push tokens past the cache-activation threshold)~~ `[DONE April]` `[CAREER]`

**Hardware needed:** Beelink Mini S12 Pro, ReSpeaker USB Mic Array, speaker
**IoT starter kit (~$85):**
- Sonoff Zigbee 3.0 USB Dongle Plus (~$20) — coordinator, plugs into Beelink
- IKEA TRÅDFRI bulb (~$10) — cheapest Zigbee bulb, dimmable
- Aqara FP2 Presence Sensor (~$55) — mmWave radar, detects presence even when still
**Future IoT additions:** Aqara temp/humidity sensor, door/window sensor, Sonoff smart plug, IKEA STYRBAR remote

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
    - Two-camera system: outside (front door) + inside (entry) with cross-camera event correlation
18. Cough detection model — audio classification via spectrograms + CNN or YAMNet embeddings `[LATEST]`
    - ReSpeaker mic captures audio, model classifies cough vs not-cough
    - Datasets: ESC-50, COUGHVID, FluSense
    - "You've been coughing a lot today — need cold medicine?"
    - Interview story: addresses an actual interview question Michael couldn't solve
19. Full-duplex interruption mechanism `[LATEST]` `[CAREER]`
    - ReSpeaker hardware echo cancellation keeps mic active during TTS
    - Silero VAD (~1MB) detects human speech in real-time, 200ms buffer prevents false triggers
    - Interrupt pipeline: stop TTS + cancel LLM stream + start STT on new input
    - Conversation state machine: jarvis_speaking / user_speaking / both_speaking / idle
    - Strictness-gated: user's personality preference determines if Jarvis allows interruption
20. Contextual filler generation via tiny distilled model `[LATEST]` `[CAREER]`
    - Michael's original research idea — potentially publishable
    - Tiny distilled model (100-300M params) generates semantically-independent fillers
    - "Hmm, good question about dinner..." plays in ~100ms while big model prefills
    - Handoff is clean because filler is grammatically self-contained — no continuity problem
    - Optional: pass filler to big model as context, or keep them independent
    - Benchmark target: time-to-first-speech, perceived naturalness vs static fillers + vanilla streaming
21. Autonomous drone deployment `[LATEST]` `[CAREER]`
    - Extends cat escape detection (#15) with automatic drone launch for aerial visual confirmation and pursuit tracking that fixed cameras can't offer
    - Also available as manual voice command ("Jarvis, launch the drone") for yard checks, finding things outside, security sweeps, generic "what's happening outside" queries
    - Hardware candidates (drones with developer APIs/SDKs — see learning notes on what an SDK is):
        - **DJI Tello EDU (~$129, recommended for Phase 2)** — hobbyist favorite, official Python SDK (`djitellopy`), simple command interface, indoor/outdoor capable, ~13 min flight time. Get the EDU version (not base Tello) — same hardware, $30 more, unlocks full SDK command set. Cheap enough to crash without crying, and you will crash it.
        - Parrot Anafi (~$600-900) — upgrade path. GPS, gimbal, 25min flight time, Olympe Python SDK. More capable than Tello, more complex to set up.
        - Skydio 2+ (~$1100+) — dream tier. Best-in-class autonomous "follow this subject" tracking, *perfect* for cat/bike pursuit. SDK access may be gated — verify before buying.
        - DJI Mavic/Mini with Mobile SDK — prosumer but requires companion phone app architecture, harder to integrate than Tello.
        - Custom ArduPilot / PX4 build — maximum flexibility, steepest learning curve, overkill for Phase 2.
    - Integration sketch:
        - New `drone_control` service in the orchestrator
        - Cat escape detection event → orchestrator triggers drone launch via Tello SDK
        - Drone video stream piped to additional camera input on Jarvis for tracking
        - Voice command route: "launch the drone" / "send the drone to the backyard" / "land the drone"
        - Safety: geofence, low-battery auto-return, manual override always available
    - Regulatory note: check local FAA / civil aviation rules on autonomous drone operation. Hobby flight is generally fine; autonomous operation without a pilot-in-command may have restrictions depending on jurisdiction and visual line of sight. Worth a quick search before deployment, not a blocker for experimentation.
    - Career framing: "I integrated autonomous aerial drone tracking into my AI assistant using the DJI Tello Python SDK, triggered by a computer vision pipeline that detects when my cat escapes the house." Combines robotics, computer vision, voice interfaces, and event-driven architecture in one concrete story.
22. Motorcycle anti-theft with gesture authentication + drone pursuit `[LATEST]` `[CAREER]`
    - Extends garage camera + drone deployment (#21) with motorcycle theft detection. Cheap camera in the garage watches the bike. When a person approaches, the system decides "authorized owner" or "intruder."
    - Authentication flow: Michael authenticates with a hand gesture (e.g., 2 fingers then 5 fingers — arbitrary but memorable, can be changed). Gesture detected + face recognition confirms Michael → drone deployment disabled for this session. No gesture, or wrong person, or ambiguous → as soon as the bike leaves the property geofence, drone auto-deploys to pursue and track, and Jarvis triggers proactive outreach (SMS + call to Michael, push notification, 911 option).
    - Why gesture auth instead of just face recognition: face alone fails in edge cases (helmet on, hoodie up, back turned, bad lighting). A gesture is a deliberate *act of authentication* proving intent, not just identity. Combines "something you are" (face) with "something you know" (gesture sequence) — classic 2-factor, expressed physically.
    - Technical components:
        - Garage camera (cheap USB or PoE IP camera)
        - Hand gesture recognition — MediaPipe Hands (Google, free, real-time, 21 hand landmarks) as upstream feature extractor + custom LSTM sequence classifier trained on Michael's specific gesture
        - Face recognition (already on roadmap for #9 entry greeting — reuse)
        - Bike-on-property detection — motion + object detection, or a BLE tag on the bike, or a GPS tracker on the bike reporting to Jarvis
        - Geofence via bike GPS tracker or phone tracking
        - Drone deployment pipeline (shared with cat-chase system — same infrastructure, different trigger)
        - Proactive outreach via Twilio (already planned in Phase 5 #41 — reuse)
    - Integration with existing roadmap items (this feature is "free" in engineering terms — every component is already planned for other reasons):
        - Cat escape detection (#15) — same drone pipeline, different CV trigger
        - Facial recognition greeting (#9) — same face model, different use case
        - Proactive outreach (Phase 5 #41) — existing SMS/call infrastructure, new trigger type
        - Garage automation (Phase 3 #24) — Meross garage opener + presence sensor stack already planned
        - Motorcycle tracker (Phase 3 #24, part of garage automation logic)
    - The genuinely new work is (a) the gesture recognition classifier and (b) the state machine coordinating "authorized vs. intruder" logic.
    - Career framing: "My AI assistant uses a hand-gesture 2FA system to distinguish me from intruders near my motorcycle, and if it detects theft, it autonomously deploys a drone from a servo-controlled hatch to pursue the bike while calling my phone." Four distinct technical claims in one sentence (custom ML model, real-time CV, event-driven autonomy, hardware integration), every one defensible.
23. Drone docking — outdoor vs. indoor deployment strategies `[LATEST]`
    - Design concern for #21 / #22. The drone needs a home when not in use. Four options under consideration:
        - **Option A — Outdoor dock (drone-as-security-camera + rapid deployment):** Weatherproof dock outside, drone doubles as a fixed surveillance camera when idle, launches on trigger. Existing products: DJI Dock 2 (~$30k, enterprise tier), Skydio Dock, custom 3D-printed enclosures with weather shielding. Pros: fastest deployment, always-ready surveillance. Cons: weather exposure, theft risk, expensive off-the-shelf, hobbyist drones aren't weather-rated. Not viable at hobbyist budget.
        - **Option B — Front-door dock with autonomous exit (likely production choice):** Drone lives in a covered area near the front door (porch, garage alcove, dedicated shed). On trigger, a servo-controlled door/hatch opens, drone flies out. Pros: weather-protected, still outdoor-adjacent for fast launch, cheap DIY. Cons: requires mechanical design (servo hatch), drone must navigate the door without collision.
        - **Option C — Indoor dock with window exit (technically coolest, practically most complex):** Drone lives in Michael's room, servos open the window on trigger, drone flies out. Pros: fully protected, fully integrated into home automation stack, drone can dual-purpose as indoor patrol. Cons: window servo retrofit is non-trivial (casement vs double-hung matters), navigation-out-of-a-room-and-through-a-window is hard, drone noise in the bedroom, and Michael has to be okay with a drone living in his room. Natural extension if smart blinds motor retrofit (Phase 1) generalizes to windows.
        - **Option D — Manual deployment (Phase 2 MVP):** Drone lives wherever, Jarvis notifies Michael to launch manually, future phases automate. Pros: zero hardware work, ship the feature immediately, validate CV/tracking pipeline before solving the hardware problem. Cons: not actually autonomous, defeats some of the interview-story appeal.
    - **Build order:** Ship Option D first to validate the full pipeline (CV → decision → drone control → tracking). Once that works reliably, build Option B (front-door dock with autonomous hatch) as production hardware. Save Option C (window servos) as a "future flex" if Option B proves limiting. Skip Option A until prosumer-drone budget exists.

### Architecture Upgrade
16. Migrate REST → MQTT message bus (Mosquitto) `[CAREER]`
17. ADR: document the REST → MQTT migration `[CAREER]`

**Hardware:** USB webcam × 2 (one bedroom/inside, one front door/outside — outside needs weather rating or housing), DJI Tello EDU (~$129) for drone prototyping
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
22. Smart alarm — Oura sleep stages + gradual light increase `[LATEST update]`
    - Color-changing lights escalate by severity: green (on time) → amber (snoozing) → red (late)
    - Requires RGB Zigbee bulb (IKEA TRÅDFRI color ~$25 or Philips Hue color ~$40)
23. Email reading — Gmail API integration `[LATEST]`
24. Garage automation — geofencing + presence-aware open/close `[LATEST]`
    - Auto-close: phone GPS >500ft from home OR gone >10min AND garage presence sensor empty AND motorcycle tracker confirms bike with you
    - Auto-open: phone approaching home within 0.25 miles during typical return window
    - Hardware: Meross Smart Garage Opener (~$40) + Aqara FP2 presence sensor (~$55)
    - Depends on Phase 6 companion app for reliable background geofencing
    - "Hey Jarvis, do I have any important emails?"
    - Morning briefing includes email summary (unread count, flagged senders, urgent subjects)
    - Gmail API is free, OAuth2, well-documented for developers
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
33. Meal/macro tracker integration `[LATEST]`
    - Connect to calorie/macro tracking app
    - Preloaded meals Michael makes regularly (Trader Joe's orange chicken + rice, etc.)
    - "Hey Jarvis, what can I eat to hit my macros?" → suggests specific preloaded meals with portions
    - Tracks remaining calories/protein/carbs/fat for the day
34. Spending tracker + savings goals `[NEW]`
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
    - **Sharper theoretical framing (April 12, 2026):** social media recommender algorithms are the empirical ground truth for what actually modifies behavior at scale. Same mechanisms (variable reinforcement, receptivity-window optimization, personalization) inverted toward goal-achievement instead of engagement-maximization. See learning notes "Social media algorithms as nudging research (inversion framing)" for the full reading list and technique catalog.
46. Personality adaptation system — Jarvis learns user's preferred voice `[LATEST]`
    - Four dimensions: encouragement, strictness, humor, commitment enforcement (each 0-1 slider)
    - Explicit feedback: thumbs up/down with reason tags ("too harsh", "not funny", "perfect")
    - Implicit feedback: follow-through rate, response engagement, rephrasing signals
    - A/B testing: occasionally try responses outside current profile to explore
    - Context awareness: time-of-day patterns (morning humor vs work-time seriousness)
    - System prompt dynamically adjusts based on learned personality vector
    - Long-term goal: help users stick to habits by matching their psychological profile
47. Identity-based goal framing — tie goals to who the user wants to BE `[LATEST]`
    - Based on James Clear's "identity-based habits" research from Atomic Habits
    - "I'm becoming a person who works out" vs "I'm trying to work out more"
    - Jarvis references identity ("runners show up even on hard days") not outcomes
    - User sets aspirational identity during onboarding, system reinforces it
48. Positive association reinforcement — reframe dread triggers `[LATEST]`
    - Detect negative sentiment around goal-related topics ("ugh, gym")
    - Intervention: pair goal mentions with user's recent wins, favorite music, rewards
    - Avoid guilt/shame language entirely — research shows it backfires
    - Connect to reward pathways: post-workout message celebrates instead of nagging
49. Novelty injection to prevent habituation `[LATEST]`
    - Vary voice, phrasing, nudge timing, delivery channel to prevent tuning out
    - Research: habituation causes even effective nudges to fail after ~2-3 weeks
    - Occasional surprise elements — different TTS voice, unexpected humor, new framings
    - Track response rate per novelty level; find personal sweet spot
50. Biometric-informed coaching via Oura Ring `[LATEST]`
    - Use HRV, sleep score, readiness to time interventions
    - High stress day → softer tone, lower goals, focus on recovery
    - High readiness → push harder, suggest stretch goals
    - Poor sleep → skip non-critical nudges, let user rest
    - Creates "Jarvis knows how I actually feel today" experience

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

#### 49.5 — Self-Insight Engine `[LATEST]` `[CAREER]`

A local-first psychological self-portrait. Not a metrics dashboard — a living document about who Michael is, generated from accumulated behavioral data + conversation history + reflection patterns. Reads like therapist intake notes: "Michael does well when X. He tends to struggle when Y. To keep him focused, Z works better than W." Every claim grounded in actual logged behavior, not vibes.

**Sources feeding the self-portrait:**
- Conversation history with Jarvis (themes, framing, language patterns)
- Behavioral telemetry (sleep, screen time, focus sessions, calendar adherence, location/presence)
- Reaction patterns (how Michael responds to nudges, what works vs backfires)
- Mood signals (explicit logs, voice tone analysis if available, sentiment over time)
- Goal/habit data (stated vs revealed)

**Surfacing modes:**
- Long-form "Who is Michael" document — updated weekly or monthly
- Targeted insights on demand — "Why do I keep skipping the gym?" returns evidence-backed answer
- Probability-of-success modeling — "Given today's sleep, stress, and schedule, here's how likely you are to hit your goals"
- Pattern surfacing for blind spots — recurring traps, hidden strengths

**Connection to nudging (Phase 5 #41, #45):** The self-insight engine and the nudging system are the same system viewed from two angles. Self-insight = the model's understanding of Michael. Nudging = acting on that understanding. The personalization framework (multi-armed bandit nudge optimization) is the output layer of the self-insight engine. Deep self-knowledge enables nudges that work for this specific person rather than generic interventions.

**Framing:** "Personal Palantir" — self-owned, local-first version of behavioral data fusion + pattern recognition + predictive modeling, pointed inward at one person, owned by that person, serving that person. Same tech as surveillance tooling; the politics invert when subject and operator are the same person.

MUST be local-first. The whole point. A self-portrait this intimate is a liability if it lives anywhere else.

**Architectural dependencies:** Phase 4 (data warehouse), Phase 5 (pattern recognition + coaching brain), strong-enough local LLM to generate the writeup (likely needs GPU upgrade past N100). Replaces/absorbs prior item #49 (basic data visualizations) — keep #49 as the MVP visualization layer underneath.

**Open design tensions to resolve later:**
- Cadence of self-portrait updates
- Feedback loop for correcting model when wrong
- Risk of self-fulfilling prophecy (telling Michael "you struggle with X" might make him more X)
- Transparency vs selective surfacing — does the model ever withhold what it's noticed? (Tension: full transparency lets user pressure model into flattery; hidden insights = paternalism. No clean answer yet.)
- Success metric: "amount accomplished" is dangerous as the only metric (optimizes toward burnout). Needs balancing with wellbeing signals.

50. Notification routing (push to phone + surface on tablet)
51. Multi-user onboarding system — goals, priorities, shortcomings, nudge preferences `[LATEST]`
53. Audio-native speech-to-speech model — bypass STT→LLM→TTS pipeline `[LATEST]`
    - Candidates: Moshi (Kyutai, open source), Spirit LM (Meta), or future models
    - Requires GPU hardware (minimum 8GB VRAM) — not feasible on N100
    - Eliminates ~300-800ms latency from STT/TTS conversion
    - Preserves tone, emphasis, emotion that text conversion loses
    - Monitor open-source progress — implement when models mature + hardware available
    - ⏰ Checkpoint: revisit state of audio-native models quarterly to evaluate feasibility

### Career
54. Demo video — 2-3 min walkthrough for LinkedIn `[CAREER]`

**Hardware:** Cheap Android tablet + wall mount
**Depends on:** All previous phases

---

## Summary

| Phase | Features | Timeline | Status |
|---|---|---|---|
| Phase 1 | 8 core + demo features | 2-4 weeks | 🟡 Demo + linter + benchmarks done; eval suite scaffolded; needs hardware |
| Phase 2 | 15 | 3-4 weeks | ⬜ Planned |
| Phase 3 | 8 | 2-3 weeks | 🟡 Notion done, rest planned |
| Phase 4 | 11 | 3-4 weeks | ⬜ Planned |
| Phase 5 | 15 | 3-4 weeks | ⬜ Planned |
| Phase 6 | 10 | 3-4 weeks | ⬜ Planned |
| **Total** | **73 features** | **16-23 weeks** | |

Career enhancements: 16 items