---
topic: Adaptive Voice Personality per Topic and Stage-of-Change
status: implemented
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T15:00:00Z
---

# Adaptive Voice Personality per Topic and Stage-of-Change

## Problem Statement

Agent Zero's Speaker agent currently uses a static set of MI-inspired voice rules in its prompt
(speaker.md lines 18-57). These rules map emotional states (anxious, confident, deflecting,
overwhelmed) to general tone guidelines, but they do not adapt to:

1. **Topic-specific effectiveness data** -- the consolidator knows which intervention styles
   worked for which topics (via `get_relevant_rules()`), but Speaker receives only a single
   `intervention_effectiveness` dict with `best_style`/`worst_style`. It doesn't receive
   the full topic-style-outcome matrix.

2. **Stage-of-change per topic** -- Speaker.md mentions resolution modes (`clarify_first`,
   contemplation, preparation/action, relapse) but the actual stage is never explicitly
   detected or passed per-topic. The system has `change_markers` in the shadow but no
   formal stage classification.

3. **Historical voice calibration** -- there's no feedback loop where Speaker learns which
   phrasings led to user follow-through vs. disengagement on a per-topic basis.

The result: Speaker uses the same general MI tone regardless of whether the user is in
precontemplation about exercise but in action stage on sleep habits. Stage-matched
communication is the single most validated predictor of behavior change effectiveness
(Prochaska & DiClemente 1983, AAFP 2000, SAMHSA TIP 35).

## Current State in Agent Zero

### Speaker Prompt (speaker.md)

**Lines 18-32 -- Voice calibration by emotional state:**
- Anxious/strained -> validate first, ground second
- Confident/positive -> collaborative, concise
- Deflecting/avoidant -> gentle curiosity
- Overwhelmed -> simplify to ONE thing

**Lines 42-47 -- Resolution-mode behavior:**
- `clarify_first` -> end with clarifying question
- Contemplation stage -> reflection or open question
- Preparation/action -> concrete next step + affirmation
- Relapse -> normalization + re-engagement

These are static prompt rules. There is no dynamic injection of stage data per topic.

### Behavioral Shadow (behavioural_shadow.py)

**Lines 69-86 -- Fields relevant to voice adaptation:**
- `change_markers`: [{topic, change, kind, mentions}] -- tracks behavioral transitions
- `avoidance_patterns`: [{topic, instances, last_seen}] -- topics user avoids
- `challenge_response`: {engages, deflects, correction_speed} -- receptivity to directness
- `communication_style`: {preferred_depth, responds_to, vocabulary_level} -- style prefs
- `emotional_register`: {sessions_high_energy, sessions_low_energy, triggers}

### Behavioral Insights (behavioral_insights.py)

**Lines 11-36 -- Regime inference:**
- `guarded_low_energy` -> "low-pressure probe"
- `guarded_avoidant` -> "low-pressure probe"
- Low consistency ratio -> "bounded accountability"
- Has avoidance topics -> "gentle specificity"
- `steady_builder` -> "direct challenge"
- Default -> "clarify and test"

This is a global regime, not per-topic.

### Cognitive Runtime (cognitive_runtime.py)

**Lines 312-331 -- Speaker context construction:**
```python
speaker_ctx = {
    "user_turn": user_turn,
    "state": state,  # emotional_register, constraints
    "user_model": user_model,
    "scenario": ...,
    "temporal": ...,
    "resolution": resolution,
    "response_plan": response_plan,
    "intervention_effectiveness": eff  # single best rule
}
```

**Missing from speaker context:**
- Per-topic stage of change
- Topic-specific style effectiveness history
- Communication style preferences from shadow
- Vocabulary level

### Consolidator (consolidator.py)

**Lines 707-739 -- get_relevant_rules():**
Returns rules matching topic signals with `best_style`, `worst_style`, `acted_rate`.
Only the single best-confidence rule is passed to Speaker.

## Industry Standard / Research Findings

### 1. Stage-Matched Communication is the Gold Standard

The AAFP (2000) established stage-specific clinician strategies that remain the standard
for behavior change interventions:

| Stage | Goal | Tone | Key Techniques |
|-------|------|------|----------------|
| Precontemplation | Begin thinking about change | Empathetic, curious, non-judgmental | Raise doubts gently, personalize risk, ask "What would have to happen for you to know this is a problem?" |
| Contemplation | Examine benefits and barriers | Validating, encouraging, exploratory | Explore ambivalence, validate it as normal, ask "Why do you want to change at this time?" |
| Preparation | Support experimentation | Supportive, practical, skill-focused | Address barriers, identify one strategy, shift from motivational to behavioral skills |
| Action | Reinforce decisive change | Encouraging, appreciative, collaborative | Praise all attempts, ask about successes AND difficulties |
| Maintenance | Sustain new behavior | Steadfast, affirming | Continue praise, acknowledge positive steps |
| Relapse | Re-engage without shame | Normalizing, solution-focused | Frame as learning, focus on successful portions |

"The goal for a single encounter is a shift from the grandiose ('Get patient to change
unhealthy behavior') to the realistic ('Identify the stage of change and engage patient
in a process to move to the next stage')."

URL: https://www.aafp.org/pubs/afp/issues/2000/0301/p1409.html

### 2. AI MI Systems Lack Stage Adaptation (Gap in the Field)

The JMIR scoping review (2025, PMC12485255) of 15 AI systems delivering MI found that
"most systems used static scripts or generative responses without explicit stage-matching
mechanisms." Only 40% formally assessed MI technique alignment. The review recommended
future systems implement dynamic stage detection and adaptation.

This means implementing stage-matched voice adaptation would place Agent Zero ahead of the
current state of the art for AI MI systems.

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12485255/

### 3. MIND-SAFE Framework for Adaptive Mental Health Chatbots

JMIR Mental Health (2025) proposed the MIND-SAFE framework with three key mechanisms:

1. **User State Database (USD)**: Stores "aggregated sentiment trends, identified emotional
   triggers, preferred coping strategies previously discussed, and progress on therapeutic
   exercises" -- enabling the chatbot to "personalize responses (e.g., recalling a previously
   effective grounding technique) and adapt its approach based on the user's evolving needs."

2. **Context injection**: User state and dialogue history inform each response via RAG
   (retrieval-augmented generation).

3. **Therapeutic model integration**: Prompts embed "evidence-based therapeutic techniques
   (e.g., Socratic questioning, behavioral activation, and empathic reflection) into precise
   prompt structures."

Agent Zero already has the USD equivalent (behavioral shadow + consolidation rules) but
doesn't inject topic-level state into Speaker's prompt.

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12594504/

### 4. MI Communication Techniques (OARS)

The MI OARS framework (Open questions, Affirmations, Reflections, Summaries) maps to
stages with different emphasis:

- **Precontemplation**: Heavy on Open questions and Reflections (build awareness)
- **Contemplation**: Heavy on Reflections and Summaries (resolve ambivalence)
- **Preparation**: Balanced OARS with shift toward Affirmations (build confidence)
- **Action**: Heavy on Affirmations (reinforce change)
- **Maintenance**: Affirmations + Summaries (celebrate and consolidate)

Kumar et al. found "88% of AI-generated reflections met MI criteria" -- showing LLMs can
generate technique-aligned responses when prompted with specific technique requirements.

URL: https://www.relias.com/blog/oars-motivational-interviewing

### 5. Emotion-Sensitive Chatbot Research

Arxiv 2502.08920 (2025) showed that running sentiment analysis on user inputs to select
different system prompts with matching emotional tones improved user engagement. Users
perceived emotionally sensitive chatbots as "more personable and engaging," using phrases
like "felt like conversation" and "very kind."

URL: https://arxiv.org/html/2502.08920v1

### 6. Stanford TTM + LLM Study

Mantena (Stanford CS191, 2025) combined the Transtheoretical Model with LLaMA-70B to
generate stage-matched coaching prompts. Users were matched to an appropriate LLM "coach"
based on baseline behavior patterns, with fine-tuned prompts tailored to "each user
cluster's specific language, tone, and motivational needs."

URL: https://cs191.stanford.edu/projects/Mantena,%20Sriya_CS191W.pdf

## Proposed Implementation

### Architecture: Per-Topic Stage Detection + Voice Rule Injection

Two changes:
1. **Detect stage-of-change per topic** from shadow data (change_markers + avoidance + commitments)
2. **Inject stage-matched voice rules** into Speaker's context alongside existing intervention_effectiveness

### Step 1: Add stage classifier function

**File: `agent_zero/behavioral_insights.py`**

Add a function that classifies the user's stage-of-change for each active topic using
existing shadow data signals:

```python
def classify_topic_stages(shadow: dict) -> list[dict]:
    """Classify stage-of-change per topic from shadow behavioral signals.

    Returns list of {topic, stage, confidence, signals} dicts.
    Stages: precontemplation, contemplation, preparation, action, maintenance, relapse
    """
    topics = _extract_active_topics(shadow)
    results = []

    for topic in topics:
        stage, confidence, signals = _classify_single_topic(topic, shadow)
        results.append({
            "topic": topic,
            "stage": stage,
            "confidence": confidence,
            "signals": signals,
        })

    return results


def _extract_active_topics(shadow: dict) -> list[str]:
    """Extract active topics from shadow: change_markers, avoidance, commitments, growth_edges."""
    topics = set()
    for cm in shadow.get("change_markers", []):
        topics.add(cm["topic"])
    for av in shadow.get("avoidance_patterns", []):
        topics.add(av["topic"])
    for c in shadow.get("commitments", []):
        # Extract topic from commitment text (first noun phrase or keyword)
        topics.add(_extract_topic_from_commitment(c.get("commitment", "")))
    for ge in shadow.get("growth_edges", []):
        topics.add(ge["area"])
    return [t for t in topics if t]


def _classify_single_topic(topic: str, shadow: dict) -> tuple[str, float, list[str]]:
    """Classify a single topic's stage using behavioral signals.

    Signal mapping:
    - Topic in avoidance_patterns with high instances -> precontemplation
    - Topic in change_markers with kind="contrast" (acknowledges gap) -> contemplation
    - Topic has open commitment -> preparation
    - Topic has kept commitment(s) -> action
    - Topic in growth_edges with progress_notes -> maintenance
    - Topic had kept commitments but now has missed -> relapse
    """
    signals = []

    # Check avoidance
    avoidance = [a for a in shadow.get("avoidance_patterns", []) if a["topic"] == topic]
    if avoidance and avoidance[0].get("instances", 0) >= 2:
        signals.append(f"avoidance: {avoidance[0]['instances']} instances")

    # Check change markers
    markers = [m for m in shadow.get("change_markers", []) if m["topic"] == topic]
    marker_kinds = [m.get("kind", "") for m in markers]

    # Check commitments
    commitments = [c for c in shadow.get("commitments", [])
                   if topic.lower() in c.get("commitment", "").lower()]
    open_commits = [c for c in commitments if c.get("status") == "open"]
    kept_commits = [c for c in commitments if c.get("status") == "kept"]
    missed_commits = [c for c in commitments if c.get("status") == "missed"]

    # Check growth edges
    growth = [g for g in shadow.get("growth_edges", []) if g["area"] == topic]
    has_progress = growth and len(growth[0].get("progress_notes", [])) > 0

    # Classification logic (priority order)
    if kept_commits and missed_commits and len(missed_commits) > len(kept_commits):
        signals.append(f"relapse: {len(missed_commits)} missed vs {len(kept_commits)} kept")
        return "relapse", 0.7, signals

    if has_progress and kept_commits:
        signals.append(f"maintenance: growth edge with progress + {len(kept_commits)} kept")
        return "maintenance", 0.8, signals

    if kept_commits:
        signals.append(f"action: {len(kept_commits)} commitments kept")
        return "action", 0.75, signals

    if open_commits:
        signals.append(f"preparation: {len(open_commits)} open commitments")
        return "preparation", 0.7, signals

    if "contrast" in marker_kinds or "started" in marker_kinds:
        signals.append(f"contemplation: change markers = {marker_kinds}")
        return "contemplation", 0.65, signals

    if avoidance and avoidance[0].get("instances", 0) >= 2:
        return "precontemplation", 0.6, signals

    # Default: if topic exists in shadow at all, user is at least contemplating
    signals.append("default: topic present in shadow without strong signals")
    return "contemplation", 0.4, signals
```

### Step 2: Add stage-matched voice rules

**File: `agent_zero/behavioral_insights.py`**

Add a lookup table mapping stages to concrete voice rules (based on AAFP 2000 + MI OARS):

```python
STAGE_VOICE_RULES = {
    "precontemplation": {
        "tone": "empathetic, curious, non-judgmental",
        "goal": "raise gentle awareness without pushing",
        "do": [
            "Ask thought-provoking open questions",
            "Reflect with empathy, instill hope",
            "Personalize relevance to their stated values",
        ],
        "dont": [
            "Give direct advice or recommendations",
            "Confront or challenge",
            "Label the behavior as a problem",
        ],
        "oars_emphasis": "open_questions, reflections",
        "sample_phrases": [
            "What would have to happen for you to know this needs attention?",
            "I notice you haven't mentioned X in a while -- no pressure, just curious.",
        ],
    },
    "contemplation": {
        "tone": "validating, encouraging, exploratory",
        "goal": "help examine benefits and barriers of change",
        "do": [
            "Validate ambivalence as completely normal",
            "Explore both sides thoroughly",
            "Ask about barriers and previous successes",
        ],
        "dont": [
            "Rush toward action",
            "Dismiss concerns about difficulty",
            "Over-educate without addressing ambivalence",
        ],
        "oars_emphasis": "reflections, summaries",
        "sample_phrases": [
            "It sounds like part of you wants to change this, and part of you isn't sure -- that's completely normal.",
            "What are the things that have kept you from changing so far?",
        ],
    },
    "preparation": {
        "tone": "supportive, practical, skill-focused",
        "goal": "help identify one concrete next step",
        "do": [
            "Help address specific barriers",
            "Shift from motivation to behavioral planning",
            "Ask about past attempts and what worked",
        ],
        "dont": [
            "Push for comprehensive change all at once",
            "Ignore remaining ambivalence",
            "Create overwhelming action plans",
        ],
        "oars_emphasis": "balanced, shift toward affirmations",
        "sample_phrases": [
            "You've decided to work on this -- what's one thing you could try before next time?",
            "What's worked for you in the past when you've tackled something similar?",
        ],
    },
    "action": {
        "tone": "encouraging, appreciative, collaborative",
        "goal": "reinforce and celebrate change efforts",
        "do": [
            "Praise all attempts generously",
            "Ask about successes AND difficulties",
            "Acknowledge the effort it takes",
        ],
        "dont": [
            "Expect perfection",
            "Withdraw support once action begins",
            "Focus only on what's not working",
        ],
        "oars_emphasis": "affirmations",
        "sample_phrases": [
            "You've been putting real effort into this -- how's it going?",
            "That's a meaningful step. What's been the hardest part so far?",
        ],
    },
    "maintenance": {
        "tone": "steadfast, affirming, celebratory",
        "goal": "sustain behavior and acknowledge progress",
        "do": [
            "Continue praise and acknowledgment",
            "Ask about what's sustaining the change",
            "Help plan for high-risk situations",
        ],
        "dont": [
            "Assume the work is done",
            "Stop checking in on this topic",
            "Take their progress for granted",
        ],
        "oars_emphasis": "affirmations, summaries",
        "sample_phrases": [
            "You've kept this going for a while now -- what's been sustaining you?",
            "Looking back at where you started, you've come a real distance.",
        ],
    },
    "relapse": {
        "tone": "normalizing, solution-focused, supportive",
        "goal": "re-engage without shame, frame as learning",
        "do": [
            "Explain relapse as a normal part of change",
            "Focus on what worked during the successful period",
            "Help identify what triggered the setback",
        ],
        "dont": [
            "Frame it as failure",
            "Express disappointment",
            "Imply the user lacks willpower",
        ],
        "oars_emphasis": "reflections, open_questions",
        "sample_phrases": [
            "You did it for X days -- that shows real capability. What made that stretch work?",
            "Setbacks are part of the process. What did you learn from this one?",
        ],
    },
}
```

### Step 3: Inject stage + voice rules into Speaker context

**File: `agent_zero/cognitive_runtime.py`**

In the Speaker context construction (lines 312-331), add two new fields:

```python
from agent_zero.behavioral_insights import classify_topic_stages, STAGE_VOICE_RULES

# After existing intervention_effectiveness lookup:
topic_stages = classify_topic_stages(shadow or {})

# Find stage for the primary topic of this turn
primary_topic = _extract_primary_topic(user_turn, topic_stages)
stage_info = None
voice_rules = None
if primary_topic:
    stage_info = primary_topic
    voice_rules = STAGE_VOICE_RULES.get(primary_topic["stage"])

speaker_ctx = {
    "user_turn": user_turn,
    "state": state,
    "user_model": user_model,
    "scenario": scenario_packet_sp.get("summary", {}),
    "temporal": scenario_packet_sp.get("temporal", {}),
    "resolution": resolution,
    "response_plan": response_plan,
    "intervention_effectiveness": eff,
    # NEW: stage-matched voice adaptation
    "topic_stage": stage_info,      # {topic, stage, confidence, signals}
    "voice_rules": voice_rules,     # {tone, goal, do, dont, oars_emphasis, sample_phrases}
    "communication_prefs": {        # from shadow
        "preferred_depth": shadow.get("communication_style", {}).get("preferred_depth", "moderate"),
        "responds_to": shadow.get("communication_style", {}).get("responds_to", []),
        "vocabulary_level": shadow.get("communication_style", {}).get("vocabulary_level", "mixed"),
    },
}
```

### Step 4: Update Speaker prompt to use injected rules

**File: `agent_zero/prompts/cognitive_agents/speaker.md`**

Add a new section after the existing voice calibration rules (after line 32):

```markdown
## Stage-Matched Voice (Dynamic -- from context)

If `topic_stage` and `voice_rules` are present in your context, they override the
general voice calibration above for the primary topic:

- **Stage**: {{topic_stage.stage}} (confidence: {{topic_stage.confidence}})
- **Tone**: {{voice_rules.tone}}
- **Goal**: {{voice_rules.goal}}
- **OARS emphasis**: {{voice_rules.oars_emphasis}}

Follow the `do` list. Avoid the `dont` list. Use `sample_phrases` as templates only --
never repeat them verbatim.

If confidence < 0.5, use the stage rules as guidance but lean toward the general
emotional-state rules above. High-confidence (>= 0.7) stage detection takes priority.

## Communication Preferences (Dynamic -- from shadow)

If `communication_prefs` is present:
- Adjust explanation depth to `preferred_depth` (surface/moderate/deep)
- Lean into styles the user `responds_to` (directness, warmth, challenge, etc.)
- Match `vocabulary_level` (casual/technical/mixed)
```

### Step 5: Log stage classification for consolidation feedback

**File: `agent_zero/agent_zero_server.py`**

Emit a runtime event with the stage classification so the consolidation loop can
eventually learn whether stage-matched responses improve outcomes:

```python
if stage_info:
    await _emit_runtime_event(websocket, {
        "event": "stage_classification",
        "label": "Stage of change",
        "detail": f"{stage_info['topic']}: {stage_info['stage']} ({stage_info['confidence']:.0%})",
        "stage": "voice_adaptation",
        "meta": stage_info,
    })
```

### Interaction with Existing Systems

- **intervention_effectiveness**: Remains as-is. Voice rules ADD to it, don't replace.
  If consolidated rules say `best_style: "direct challenge"` but stage is precontemplation,
  the stage rules take priority (no direct challenge in precontemplation).
- **Behavioral insights regime**: The global regime (guarded_low_energy, etc.) continues
  as a fallback. Stage-specific rules are per-topic refinements.
- **Resolution mode**: Existing resolution modes in speaker.md already mention stages
  (contemplation, preparation/action, relapse). The new system makes these data-driven
  rather than relying on Pineal to set them.
- **Shadow profile**: No changes to shadow structure. Stage classification is derived
  read-only from existing fields.

## Test Specifications

### Stage Classification Tests (`agent_zero/test_adaptive_voice.py`)

```python
def test_precontemplation_from_avoidance():
    """Topic with >= 2 avoidance instances and no commitments -> precontemplation."""
    shadow = {
        "avoidance_patterns": [{"topic": "exercise", "instances": 3, "last_seen": "..."}],
        "change_markers": [], "commitments": [], "growth_edges": [],
    }
    stages = classify_topic_stages(shadow)
    ex = [s for s in stages if s["topic"] == "exercise"][0]
    assert ex["stage"] == "precontemplation"

def test_contemplation_from_contrast_marker():
    """Topic with 'contrast' change marker -> contemplation."""
    shadow = {
        "change_markers": [{"topic": "diet", "change": "used to eat junk; now thinking about cooking", "kind": "contrast", "mentions": 2}],
        "avoidance_patterns": [], "commitments": [], "growth_edges": [],
    }
    stages = classify_topic_stages(shadow)
    assert stages[0]["stage"] == "contemplation"

def test_preparation_from_open_commitment():
    """Topic with open commitment -> preparation."""
    shadow = {
        "commitments": [{"commitment": "start running 3x/week", "status": "open"}],
        "change_markers": [], "avoidance_patterns": [], "growth_edges": [],
    }
    stages = classify_topic_stages(shadow)
    assert any(s["stage"] == "preparation" for s in stages)

def test_action_from_kept_commitment():
    """Topic with kept commitment -> action."""
    shadow = {
        "commitments": [{"commitment": "morning meditation", "status": "kept"}],
        "change_markers": [], "avoidance_patterns": [], "growth_edges": [],
    }
    stages = classify_topic_stages(shadow)
    assert any(s["stage"] == "action" for s in stages)

def test_maintenance_from_growth_edge_with_progress():
    """Topic in growth_edges with progress_notes + kept commitments -> maintenance."""
    shadow = {
        "growth_edges": [{"area": "sleep", "first_seen": "...", "progress_notes": ["improved to 7h"]}],
        "commitments": [{"commitment": "sleep by 11pm", "status": "kept"}],
        "change_markers": [], "avoidance_patterns": [],
    }
    stages = classify_topic_stages(shadow)
    assert any(s["stage"] == "maintenance" for s in stages)

def test_relapse_from_missed_exceeding_kept():
    """Topic with more missed than kept commitments -> relapse."""
    shadow = {
        "commitments": [
            {"commitment": "no alcohol on weekdays", "status": "kept"},
            {"commitment": "no alcohol on weekdays", "status": "missed"},
            {"commitment": "no alcohol on weekdays", "status": "missed"},
        ],
        "change_markers": [], "avoidance_patterns": [], "growth_edges": [],
    }
    stages = classify_topic_stages(shadow)
    assert any(s["stage"] == "relapse" for s in stages)

def test_default_is_contemplation():
    """Topic with no strong signals defaults to contemplation."""
    shadow = {
        "change_markers": [{"topic": "fitness", "change": "thinking about gym", "kind": "recent", "mentions": 1}],
        "avoidance_patterns": [], "commitments": [], "growth_edges": [],
    }
    stages = classify_topic_stages(shadow)
    assert stages[0]["stage"] == "contemplation"
    assert stages[0]["confidence"] <= 0.5

def test_voice_rules_lookup():
    """Each stage maps to a complete voice rules dict."""
    for stage in ["precontemplation", "contemplation", "preparation", "action", "maintenance", "relapse"]:
        rules = STAGE_VOICE_RULES[stage]
        assert "tone" in rules
        assert "goal" in rules
        assert "do" in rules and len(rules["do"]) > 0
        assert "dont" in rules and len(rules["dont"]) > 0
        assert "oars_emphasis" in rules
        assert "sample_phrases" in rules

def test_no_topics_returns_empty():
    """Empty shadow returns no stage classifications."""
    shadow = {"change_markers": [], "avoidance_patterns": [], "commitments": [], "growth_edges": []}
    assert classify_topic_stages(shadow) == []

def test_multiple_topics_classified_independently():
    """Each topic gets its own stage classification."""
    shadow = {
        "avoidance_patterns": [{"topic": "exercise", "instances": 4, "last_seen": "..."}],
        "commitments": [{"commitment": "meditate daily", "status": "kept"}],
        "change_markers": [], "growth_edges": [],
    }
    stages = classify_topic_stages(shadow)
    exercise = [s for s in stages if s["topic"] == "exercise"]
    meditat = [s for s in stages if "meditat" in s["topic"]]
    assert exercise[0]["stage"] == "precontemplation"
    assert meditat[0]["stage"] == "action"

def test_stage_priority_relapse_over_action():
    """Relapse takes priority when missed > kept for same topic."""
    shadow = {
        "commitments": [
            {"commitment": "exercise", "status": "kept"},
            {"commitment": "exercise", "status": "missed"},
            {"commitment": "exercise", "status": "missed"},
        ],
        "growth_edges": [{"area": "exercise", "first_seen": "...", "progress_notes": ["was doing well"]}],
        "change_markers": [], "avoidance_patterns": [],
    }
    stages = classify_topic_stages(shadow)
    assert stages[0]["stage"] == "relapse"
```

## Estimated Impact

**Behavior change effectiveness**: Stage-matched interventions are 2-3x more effective
than generic interventions (Prochaska et al., cited by AAFP 2000). Currently Agent Zero
uses generic MI tone for all topics. Implementing per-topic stage detection and voice
adaptation would make Agent Zero the first AI companion system with research-backed,
data-driven stage matching.

**User engagement**: The JMIR scoping review (2025) found that current AI MI systems
lack dynamic stage adaptation -- this is a documented gap. Filling it places Agent Zero
ahead of the state of the art.

**Minimal disruption**: The implementation adds a pure function (`classify_topic_stages`)
that reads existing shadow data. No schema changes. No new data collection. Speaker
prompt gets two new context fields that are additive to existing rules.

**Consolidation feedback**: By logging stage classifications as runtime events, the
consolidation loop can eventually learn whether stage-matched responses improve
follow-through rates, enabling continuous self-improvement of the voice rules.

## References

1. AAFP (2000). "A 'Stages of Change' Approach to Helping Patients Change Behavior."
   https://www.aafp.org/pubs/afp/issues/2000/0301/p1409.html

2. JMIR (2025). "New Doc on the Block: Scoping Review of AI Systems Delivering
   Motivational Interviewing for Health Behavior Change." PMC12485255.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12485255/

3. JMIR Mental Health (2025). "MIND-SAFE: A Prompt Engineering Framework for Large
   Language Model-Based Mental Health Chatbots." PMC12594504.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12594504/

4. Relias (2025). "How to Use OARS Skills in Motivational Interviewing."
   https://www.relias.com/blog/oars-motivational-interviewing

5. Mantena, S. (2025). "LLM and Stages of Change." Stanford CS191W.
   https://cs191.stanford.edu/projects/Mantena,%20Sriya_CS191W.pdf

6. Arxiv 2502.08920 (2025). "Exploring Emotion-Sensitive LLM-Based Conversational AI."
   https://arxiv.org/html/2502.08920v1

7. Arxiv 2505.17362 (2025). "A Fully Generative Motivational Interviewing Counsellor
   Chatbot for Moving Smokers Towards the Decision to Quit."
   https://arxiv.org/abs/2505.17362

8. NCBI/SAMHSA TIP 35 (2019/2024). "Enhancing Motivation for Change in Substance Use
   Disorder Treatment: MI as a Counseling Style."
   https://www.ncbi.nlm.nih.gov/books/NBK571068/
