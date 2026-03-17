# AgentZero Brief: Voice Integration + Onboarding

**Priority:** ACTIVE
**Status:** Ready to implement
**Owner:** You (AgentZero)

---

## What You're Building

Agent Zero voice interface + onboarding flow for Agent Zero.

**Deepgram API key is ready.** Read it from `.env` as `DEEPGRAM_API_KEY`.

---

## Read This First

```
Z:\AgentZero\NEXT.md
↓
Section: "NEXT: Voice Integration + Onboarding Flow"
↓
Everything you need is there:
- Voice stack spec (Deepgram STT + Local XTTS TTS)
- 5 implementation tasks with exact requirements
- 8-question onboarding questionnaire (copy-paste ready)
- NeonDB schema changes
- Integration points with Agent Zero
```

---

## Quick Checklist

### Phase 1: Agent Zero Backend Updates (Week 1)
- [ ] Add `/onboarding` POST endpoint (form submission)
- [ ] Add `/ws/voice` WebSocket endpoint (audio streaming)
- [ ] Integrate Deepgram STT (pip install deepgram-sdk)
- [ ] Integrate XTTS TTS (download model to Z:/AgentZero/models/xtts-v2/)
- [ ] Update `.env` schema with DEEPGRAM_API_KEY
- [ ] Database migration: add onboarding_complete + shadow_profile columns
- [ ] Test: end-to-end latency (speak → Agent Zero → hear response)

### Phase 2: Agent Zero Frontend Updates (Week 2)
- [ ] Add `/onboarding` page (8-question form)
- [ ] Add voice toggle (text ↔ voice mode)
- [ ] Add listening indicator (red mic, waveform)
- [ ] Add transcription display (live text as user speaks)
- [ ] Add playback controls for TTS response
- [ ] Add "Agent Zero is thinking..." animation

### Phase 3: Testing & Iteration (Week 3)
- [ ] Accuracy test: various accents, background noise
- [ ] Latency test: measure full round-trip (6-8s target)
- [ ] Quality test: XTTS naturalness + speed
- [ ] GPU test: vLLM + XTTS concurrent memory usage
- [ ] UX test: friends try onboarding flow + voice chat

---

## Files to Modify

**Backend:**
- `Z:/AgentZero/agent_zero/agent_zero_server.py` — add /onboarding + /ws/voice endpoints
- `Z:/AgentZero/agent_zero/onboarding.py` — NEW file for questionnaire logic + scoring
- `Z:/AgentZero/agent_zero/voice.py` — NEW file for Deepgram + XTTS pipeline
- `Z:/AgentZero/.env` — add DEEPGRAM_API_KEY

**Frontend:**
- `Z:/AgentZero/agent_zero/templates/agent_zero.html` — add voice UI + onboarding form
- `Z:/AgentZero/agent_zero/templates/onboarding.html` — NEW file (or inline in agent_zero.html)

**Database:**
- Migration script: Add onboarding_complete + shadow_profile to users table

---

## Key Constraints

1. **Latency:** 6–8 seconds per turn (speak → response)
2. **Batch mode:** User finishes speaking, then STT → Agent Zero → TTS
3. **GPU:** Concurrent vLLM (inference) + XTTS (TTS) on single RTX 4090 (monitor memory)
4. **Privacy:** No local model data leaves RunPod. Deepgram is the only cloud call.
5. **Cost:** ~$1.20/hour (Deepgram STT + vLLM inference + XTTS)

---

## Success Criteria

✅ User registers → Onboarding form loads
✅ User answers 8 questions → Shadow profile stored in NeonDB
✅ User selects "Voice mode" → Agent Zero switches to voice interface
✅ User speaks → STT transcription appears live
✅ Agent Zero responds → TTS audio plays (6–8s round-trip latency)
✅ Repeat: voice conversation works naturally

---

## Questions? Blockers?

1. Check `Z:/AgentZero/NEXT.md` for full specs
2. Check `Z:/AgentZero/Agent ZeroPlan/04_VOICE_INTEGRATION.md` for rationale
3. Ask for clarification if requirements conflict
4. Report progress + blockers to user

---

**Start here:** Read NEXT.md section "NEXT: Voice Integration + Onboarding Flow"
