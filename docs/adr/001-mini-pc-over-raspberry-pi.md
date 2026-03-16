# ADR 001: Mini PC over Raspberry Pi for the Server

**Date:** March 15, 2026
**Status:** Accepted

## Context

Need a 24/7 server to run Home Assistant, Whisper STT, Piper TTS, OpenWakeWord, and the Jarvis orchestrator. The two main options are a Raspberry Pi 5 (~$120 all-in) or a mini PC like the Beelink Mini S12 Pro (~$150).

## Decision

Use the Beelink Mini S12 Pro (Intel N100, 16GB RAM, 500GB SSD).

## Reasoning

- The Pi 5 handles basic Home Assistant fine, but struggles when you add Whisper + camera processing + LLM orchestration simultaneously.
- The N100 has significantly more headroom for future phases (facial recognition, activity detection, multiple camera feeds).
- Cost difference is only ~$30-50 more than a fully kitted Pi 5 (board + NVMe HAT + SSD + case + PSU + cooling).
- Power consumption difference is ~5W ($7/year) — negligible.

## Tradeoffs

- Pi 5 has better community support for GPIO/hardware hacking (not relevant for this project).
- Pi 5 consumes less power at idle (~5W vs ~10W).
- Mini PC is slightly larger physically, but still very small.

## Consequences

- Server will comfortably run all Phase 1-6 services without hardware swaps.
- Docker Compose deployment works identically on both — this decision doesn't lock us into anything.
