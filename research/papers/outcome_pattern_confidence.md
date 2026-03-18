---
topic: Outcome Pattern Confidence Scoring
status: ready_for_implementation
priority: medium
estimated_complexity: small
researched_at: 2026-03-18T15:30:00Z
---

# Outcome Pattern Confidence Scoring

## Problem Statement

Agent Zero uses regex pattern matching to detect whether users followed through on commitments. Currently, `classify_outcome()` in `outcome_patterns.py` returns a flat label ("acted", "ignored", "pushed_back", "neutral") with **no confidence score**. This means:

1. **"I did it" and "I tried it" are treated identically** -- both match `POSITIVE_OUTCOME` and return "acted", but "I tried it" is much weaker evidence of actual follow-through
2. **"I started" vs "I finished" carry the same weight** -- completion and initiation are conflated
3. **Hedging language is invisible**: "I sort of did it" and "I kind of tried" match as positive with full confidence
4. **Downstream consumers can't differentiate**: Consolidation (`_compute_agent_calibration`), intervention tracking, and outcome resolution all treat every match as equally certain

## Current State in Agent Zero

### outcome_patterns.py (full file, 74 lines)

```python
POSITIVE_OUTCOME = (
    r"\bi did it\b", r"\bi followed through\b", r"\bi went through with it\b",
    r"\bi tried it\b", r"\bi started\b", r"\bi finished\b", r"\bi sent\b",
    r"\bi asked\b", r"\bi booked\b", r"\bi spoke to\b", r"\bi took the step\b",
)

def classify_outcome(text: str) -> str:
    if matches_any(text, POSITIVE_OUTCOME):
        return "acted"
    if matches_any(text, PUSHBACK_OUTCOME):
        return "pushed_back"
    if matches_any(text, IGNORED_OUTCOME):
        return "ignored"
    return "neutral"
```

- No confidence score returned
- Pattern order determines priority (first match wins)
- No hedging detection
- No distinction between strong/weak positive signals

### Consumers of classify_outcome

1. **episode_store.py:resolve_episode_outcomes()** -- line ~260: uses classify_outcome to set `ep["outcome"]["user_followed_up"]` as boolean True/False
2. **consolidator.py:_compute_agent_calibration()** -- line ~127: counts `user_followed_up` as binary signal for calibration metrics
3. **consolidator.py:_compute_intervention_effectiveness()** -- line ~303: groups outcomes into acted/ignored/pushed_back counts

## Industry Standard / Research Findings

### 1. DARN-CAT Change Talk Framework (Miller & Rollnick, MI)

Motivational Interviewing distinguishes between **preparatory change talk** (DARN: Desire, Ability, Reason, Need) and **mobilizing change talk** (CAT: Commitment, Activation, Taking Steps). "I tried it" is Activation (preparatory), while "I did it" and "I finished" are Taking Steps (mobilizing). Research shows that mobilizing change talk is a significantly stronger predictor of actual behavior change than preparatory talk.

**URL**: https://casaa.unm.edu/tools/misc.html

### 2. Automated MISC Coding with NLP (NLP-AI4Health 2025)

Recent NLP work on automated coding of MI transcripts achieves 88% accuracy for client behavior classification. The key finding: **the slope of change talk vs sustain talk across sessions correlates with treatment outcomes** (r=0.28, p<0.005). This validates that distinguishing change talk strength (not just presence/absence) is predictive of actual follow-through.

**URL**: https://aclanthology.org/2025.nlpai4health-main.4/

### 3. MISC 2.5 Commitment Strength Scale

The MISC codes commitment language on a strength scale from 1 (weak) to 10 (strong). Key markers:
- **Strong (8-10)**: "I will", "I guarantee", "I am going to do it", "I did it", "I finished"
- **Medium (5-7)**: "I plan to", "I intend to", "I started", "I spoke to"
- **Weak (1-4)**: "I tried", "I might", "I sort of did", "I thought about it"

**URL**: https://casaa.unm.edu/assets/docs/misc25.pdf

### 4. Hedging Language Detection (Sentiment Analysis, 2025)

Production sentiment analysis systems return confidence scores between 0 and 1 for each classification. Azure Language Service and similar tools compute confidence from pattern strength, negation detection, and hedge word presence. The principle: **classification without confidence is incomplete**.

**URL**: https://learn.microsoft.com/en-us/azure/ai-services/language-service/sentiment-opinion-mining/overview

### 5. Commitment Language Coding (Amrhein et al.)

Research shows that specific verb choices map to commitment strength: "will" and "going to" are strong commitment markers (0.8-1.0), "try to" and "might" are weak (0.3-0.5), and "I did" past-tense completion is the strongest signal (0.9-1.0) because it reports accomplished action rather than intention.

**URL**: https://digitalcommons.montclair.edu/psychology-facpubs/27/

### 6. Change Talk Automated Classification Pipeline

The state-of-the-art pipeline for automated change talk coding uses hierarchical classification: first classify utterance type (change talk vs sustain talk vs follow/neutral), then score strength within type. This two-level approach achieves 82% accuracy at coarse level, 68-76% at fine-grained level.

**URL**: https://aclanthology.org/2025.nlpai4health-main.4/

## Proposed Implementation

### Design: Pattern-Level Confidence Scores + Hedge Detection

Replace flat pattern tuples with scored patterns, add a hedge detection layer, and return `(outcome, confidence)` from `classify_outcome`.

### Step 1: Define Scored Outcome Patterns

Replace the flat tuples in `outcome_patterns.py` with scored patterns:

```python
# Scored positive patterns: (pattern, base_confidence)
# Based on MISC commitment strength scale mapped to 0-1
POSITIVE_OUTCOME_SCORED = (
    # Strong completion (Taking Steps / CAT) -- 0.9-1.0
    (r"\bi did it\b", 0.95),
    (r"\bi finished\b", 0.95),
    (r"\bi followed through\b", 0.95),
    (r"\bi went through with it\b", 0.90),
    (r"\bi completed\b", 0.95),

    # Action taken (Taking Steps) -- 0.8-0.9
    (r"\bi sent\b", 0.85),
    (r"\bi booked\b", 0.85),
    (r"\bi spoke to\b", 0.85),
    (r"\bi asked\b", 0.80),
    (r"\bi took the step\b", 0.85),
    (r"\bi signed up\b", 0.85),

    # Initiation (Activation / DARN) -- 0.5-0.7
    (r"\bi started\b", 0.65),
    (r"\bi tried it\b", 0.50),
    (r"\bi gave it a go\b", 0.55),
    (r"\bi attempted\b", 0.50),
    (r"\bi began\b", 0.60),
)

IGNORED_OUTCOME_SCORED = (
    (r"\bi did not\b.+\b(do|send|ask|book|start|finish|try|follow through)\b", 0.85),
    (r"\bi didn't\b.+\b(do|send|ask|book|start|finish|try|follow through)\b", 0.85),
    (r"\bi haven't\b.+\b(done|sent|asked|booked|started|finished|tried)\b", 0.80),
    (r"\bi have not\b.+\b(done|sent|asked|booked|started|finished|tried)\b", 0.80),
    (r"\bnot yet\b", 0.60),  # Weaker -- implies intention to do it later
    (r"\bi put it off\b", 0.75),
    (r"\bi avoided it\b", 0.80),
    (r"\bi forgot\b", 0.70),
    (r"\bstill haven't\b", 0.85),
)

PUSHBACK_OUTCOME_SCORED = (
    (r"\bthat won't work\b", 0.85),
    (r"\bthat does not work\b", 0.85),
    (r"\bi'm not doing that\b", 0.90),
    (r"\bi am not doing that\b", 0.90),
    (r"\bi do not want to do that\b", 0.90),
    (r"\bi don't want to do that\b", 0.90),
    (r"\bno thanks\b", 0.70),
    (r"\btoo much\b", 0.60),
    (r"\bstop pushing\b", 0.90),
    (r"\bi disagree with that\b", 0.75),
)

# Keep backward-compatible flat tuples (extract patterns only)
POSITIVE_OUTCOME = tuple(p for p, _ in POSITIVE_OUTCOME_SCORED)
IGNORED_OUTCOME = tuple(p for p, _ in IGNORED_OUTCOME_SCORED)
PUSHBACK_OUTCOME = tuple(p for p, _ in PUSHBACK_OUTCOME_SCORED)
```

### Step 2: Add Hedge Detection

```python
# Hedging markers that reduce confidence
HEDGE_PATTERNS = (
    (r"\bsort of\b", 0.15),
    (r"\bkind of\b", 0.15),
    (r"\bkinda\b", 0.15),
    (r"\bmaybe\b", 0.10),
    (r"\bi think\b", 0.10),
    (r"\bi guess\b", 0.15),
    (r"\bnot really\b", 0.20),
    (r"\bnot sure\b", 0.15),
    (r"\bpartially\b", 0.10),
    (r"\ba little\b", 0.10),
    (r"\bbut\b", 0.05),  # "I did it, but..."
)


def _compute_hedge_penalty(text: str) -> float:
    """Compute confidence reduction from hedging language."""
    penalty = 0.0
    for pattern, weight in HEDGE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            penalty += weight
    return min(penalty, 0.4)  # Cap total hedge penalty at 0.4
```

### Step 3: Add Scored classify_outcome

```python
def classify_outcome_scored(text: str) -> tuple[str, float]:
    """Classify user text with confidence score.

    Returns (outcome_label, confidence) where confidence is 0.0-1.0.
    Uses MISC-inspired commitment strength and hedge detection.
    """
    hedge_penalty = _compute_hedge_penalty(text)

    # Check each category, return highest-confidence match
    best_label = "neutral"
    best_conf = 0.0

    for pattern, base_conf in POSITIVE_OUTCOME_SCORED:
        if re.search(pattern, text, flags=re.IGNORECASE):
            conf = max(0.1, base_conf - hedge_penalty)
            if conf > best_conf:
                best_conf = conf
                best_label = "acted"

    for pattern, base_conf in PUSHBACK_OUTCOME_SCORED:
        if re.search(pattern, text, flags=re.IGNORECASE):
            conf = max(0.1, base_conf - hedge_penalty * 0.5)  # Less hedge impact on pushback
            if conf > best_conf:
                best_conf = conf
                best_label = "pushed_back"

    for pattern, base_conf in IGNORED_OUTCOME_SCORED:
        if re.search(pattern, text, flags=re.IGNORECASE):
            conf = max(0.1, base_conf - hedge_penalty * 0.3)  # Less hedge impact on ignored
            if conf > best_conf:
                best_conf = conf
                best_label = "ignored"

    return (best_label, round(best_conf, 2))


# Keep backward-compatible classify_outcome
def classify_outcome(text: str) -> str:
    """Classify user text as 'acted', 'ignored', 'pushed_back', or 'neutral'."""
    label, _ = classify_outcome_scored(text)
    return label
```

### Step 4: Use Confidence in Episode Outcome Resolution

In `episode_store.py:resolve_episode_outcomes()`, store confidence alongside the outcome:

```python
# Where outcome is resolved from user text:
from outcome_patterns import classify_outcome_scored

label, confidence = classify_outcome_scored(user_content)
if label != "neutral":
    ep_outcome["intervention_outcome"] = label
    ep_outcome["outcome_confidence"] = confidence
    ep_outcome["user_followed_up"] = label == "acted" and confidence >= 0.5
```

Note: `user_followed_up` now requires confidence >= 0.5, meaning "I sort of tried it" (acted, ~0.35) no longer counts as followed_up.

## Test Specifications

### test_outcome_confidence.py

```python
from outcome_patterns import classify_outcome_scored, _compute_hedge_penalty

# Test 1: Strong completion has high confidence
def test_strong_completion_high_confidence():
    label, conf = classify_outcome_scored("I did it!")
    assert label == "acted"
    assert conf >= 0.90

# Test 2: "I tried it" has lower confidence than "I did it"
def test_tried_lower_than_did():
    _, conf_did = classify_outcome_scored("I did it")
    _, conf_tried = classify_outcome_scored("I tried it")
    assert conf_did > conf_tried
    assert conf_tried < 0.70

# Test 3: Hedging reduces confidence
def test_hedging_reduces_confidence():
    _, conf_plain = classify_outcome_scored("I tried it")
    _, conf_hedged = classify_outcome_scored("I sort of tried it")
    assert conf_hedged < conf_plain

# Test 4: Multiple hedges stack
def test_multiple_hedges_stack():
    _, conf_single = classify_outcome_scored("I kind of did it")
    _, conf_double = classify_outcome_scored("I kind of sort of did it")
    assert conf_double < conf_single

# Test 5: Hedge penalty is capped
def test_hedge_penalty_capped():
    penalty = _compute_hedge_penalty("sort of kind of maybe i think i guess not really a little but partially")
    assert penalty <= 0.4

# Test 6: "not yet" is weaker ignored than "I didn't do it"
def test_not_yet_weaker_than_didnt():
    _, conf_didnt = classify_outcome_scored("I didn't do it")
    _, conf_not_yet = classify_outcome_scored("not yet")
    assert conf_didnt > conf_not_yet

# Test 7: Neutral text returns 0 confidence
def test_neutral_zero_confidence():
    label, conf = classify_outcome_scored("the weather is nice today")
    assert label == "neutral"
    assert conf == 0.0

# Test 8: backward-compatible classify_outcome still works
def test_backward_compat():
    assert classify_outcome("I did it") == "acted"
    assert classify_outcome("I forgot") == "ignored"
    assert classify_outcome("hello") == "neutral"

# Test 9: "I started" is lower confidence than "I finished"
def test_started_lower_than_finished():
    _, conf_started = classify_outcome_scored("I started")
    _, conf_finished = classify_outcome_scored("I finished")
    assert conf_finished > conf_started

# Test 10: Pushback is high confidence
def test_pushback_high_confidence():
    label, conf = classify_outcome_scored("I'm not doing that")
    assert label == "pushed_back"
    assert conf >= 0.85
```

## Estimated Impact

1. **More accurate calibration data**: Consolidation's `_compute_agent_calibration` currently counts "I tried it" (weak) the same as "I did it" (strong). With confidence scoring, downstream consumers can weight outcomes proportionally, producing more accurate agent calibration.

2. **Better outcome resolution**: `user_followed_up` now requires confidence >= 0.5, filtering out hedged and tentative reports. This prevents inflating follow-through rates with weak signals like "I sort of tried it."

3. **Aligns with MI clinical standards**: The DARN-CAT framework is the gold standard for coding commitment language strength. Mapping Agent Zero patterns to this framework makes the system clinically informed.

4. **Backward compatible**: `classify_outcome()` is preserved with identical behavior. The new `classify_outcome_scored()` is opt-in. Existing consumers continue working unchanged.

5. **Foundation for future ML**: The scored patterns provide labeled training data for eventually training a small classifier (from user corrections) to improve beyond regex matching.
