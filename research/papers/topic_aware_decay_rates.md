---
topic: Topic-Aware Memory Decay Rates
status: ready_for_implementation
priority: medium
estimated_complexity: small
researched_at: 2026-03-18T15:00:00Z
---

# Topic-Aware Memory Decay Rates

## Problem Statement

Agent Zero applies a uniform exponential decay rate (`lambda=0.005/hr`) to all episodes regardless of topic. A career decision episode and a casual fitness check-in episode decay at exactly the same rate. This is cognitively unrealistic and practically harmful:

1. **High-stakes topics decay too fast**: Career decisions, relationship milestones, and health diagnoses should persist longer because they inform long-term behavioral patterns
2. **Low-stakes topics persist too long**: Casual check-ins about weather or mood decay at the same rate as major life events
3. **No personalization**: The decay rate is hardcoded, not adapted from user behavior

## Current State in Agent Zero

### decay_episodes() -- episode_store.py:316-355

```python
DEFAULT_LAMBDA_PER_HOUR = 0.005  # line 41

def decay_episodes(shadow: dict, lambda_per_hour: float = DEFAULT_LAMBDA_PER_HOUR) -> dict:
    # ...
    base_strength = math.exp(-lambda_per_hour * hours_elapsed)  # line 346
    boost = retrieval_count * RETRIEVAL_BOOST  # line 350
    ep["decay_strength"] = round(min(1.0, max(0.0, base_strength + boost)), 4)  # line 353
```

- Single `lambda_per_hour=0.005` for all topics
- No topic-dependent modulation
- The `lambda_per_hour` parameter exists but is never varied by caller

### Topic extraction -- outcome_patterns.py

Topics are extracted via `TOPIC_KEYWORDS` dict with categories like: career, health, fitness, relationships, finance, learning, creativity, social, habits, mental_health, sleep, nutrition, productivity, mindfulness, self_reflection

### Episode structure

Each episode has `topic_signals: list[str]` containing 0+ topic categories.

## Industry Standard / Research Findings

### 1. FSRS: Difficulty-Modulated Decay (Ye et al., 2023-2025)

The Free Spaced Repetition Scheduler (FSRS-6) uses per-item difficulty to modulate stability gains. Key formula: `R(t, S) = (1 + factor * t/S)^(-w20)` where stability `S` varies per item based on difficulty `D` (1-10 scale). **Harder items (higher D) gain less stability per review**, meaning they need more frequent reinforcement. FSRS demonstrates 20-30% fewer reviews needed vs uniform scheduling by adapting to item difficulty.

**URL**: https://github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm

### 2. Emotional Arousal Flattens Forgetting Curves (Cognitive Science, 2025)

Research shows that "emotional arousal enhances consolidation and flattens the curve for affectively charged content, deviating from the standard exponential decay observed in neutral material." High-stakes topics (career crises, relationship breakups, health scares) are emotionally arousing and should decay more slowly than neutral topics.

**URL**: https://www.growthengineering.co.uk/forgetting-curve/

### 3. ACT-R Memory Architecture for LLM Agents (HAI 2025)

The ACT-R-inspired memory architecture for LLM agents uses base-level activation that integrates recency, frequency, and context-dependent decay. Different memory chunks decay at different rates based on their creation context and subsequent retrieval pattern. This is the closest analog to topic-dependent decay in cognitive architectures.

**URL**: https://dl.acm.org/doi/10.1145/3765766.3765803

### 4. Context-Dependent Decay in AI Agent Memory (2025 Industry Practice)

Multiple 2025 frameworks note that "context decays at different rates -- a project deadline becomes irrelevant after completion, while a communication preference may stay valid indefinitely." Oracle, Mem0, and DigitalOcean all document per-category forgetting policies as best practice for production AI agents.

**URL**: https://blogs.oracle.com/developers/agent-memory-why-your-ai-has-amnesia-and-how-to-fix-it

### 5. Distribution Model of Forgetting (MDPI Behavioral Sciences, 2025)

Recent research shows that empirical forgetting curves deviate from actual forgetting rates because of the distribution of initial memory strengths. This implies that averaging decay across topics with different inherent strengths produces misleading results -- each topic category has its own effective decay rate.

**URL**: https://www.mdpi.com/2076-328X/15/7/924

### 6. Deep Knowledge Tracing with Memory Decay (Knowledge-Based Systems, 2025)

Modeling memory decay in deep knowledge tracing shows that per-concept decay modeling significantly outperforms uniform decay in predicting knowledge retention, particularly for concepts with different inherent difficulty levels.

**URL**: https://www.sciencedirect.com/science/article/pii/S0950705125019227

## Proposed Implementation

### Design: Topic-Category Decay Multipliers

Add a mapping from topic categories to decay rate multipliers. The base `lambda=0.005` is scaled per topic. This is simple, backward-compatible, and requires no ML training.

### Step 1: Define Topic Decay Multipliers

Add to `episode_store.py` after line 44:

```python
# Topic-dependent decay multipliers (lower = slower decay = longer retention)
# Based on emotional arousal and decision stakes (cognitive science literature)
TOPIC_DECAY_MULTIPLIERS = {
    # High-stakes, emotionally charged -- slow decay
    "career":          0.5,   # Career decisions persist 2x longer
    "relationships":   0.5,   # Relationship patterns persist 2x longer
    "mental_health":   0.6,   # Mental health patterns persist ~1.7x longer
    "finance":         0.6,   # Financial decisions persist ~1.7x longer
    "health":          0.7,   # Health patterns persist ~1.4x longer

    # Medium-stakes -- normal decay
    "self_reflection":  0.8,
    "learning":         0.8,
    "creativity":       0.9,
    "productivity":     1.0,   # Default rate
    "habits":           1.0,

    # Low-stakes, routine -- faster decay
    "fitness":          1.2,   # Routine check-ins decay 1.2x faster
    "nutrition":        1.2,
    "sleep":            1.3,   # Sleep logs are very routine
    "social":           1.3,
    "mindfulness":      1.0,
}

DEFAULT_DECAY_MULTIPLIER = 1.0
```

### Step 2: Modify decay_episodes() to Use Topic-Aware Rates

Replace lines 331-353 in `decay_episodes()`:

```python
for ep in episodes:
    ts_str = ep.get("timestamp")
    if not ts_str:
        continue

    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        continue

    hours_elapsed = max(0, (now - ts).total_seconds() / 3600.0)

    # Topic-aware decay: use the slowest decay rate among the episode's topics
    # (if an episode touches career AND fitness, career's slower rate wins)
    topic_signals = ep.get("topic_signals", [])
    if topic_signals:
        multiplier = min(
            TOPIC_DECAY_MULTIPLIERS.get(t, DEFAULT_DECAY_MULTIPLIER)
            for t in topic_signals
        )
    else:
        multiplier = DEFAULT_DECAY_MULTIPLIER

    effective_lambda = lambda_per_hour * multiplier

    # Base decay
    base_strength = math.exp(-effective_lambda * hours_elapsed)

    # Spaced repetition boost from retrievals
    retrieval_count = ep.get("retrieval_count", 0)
    boost = retrieval_count * RETRIEVAL_BOOST

    # Final strength: base + boost, clamped to [0, 1]
    ep["decay_strength"] = round(min(1.0, max(0.0, base_strength + boost)), 4)
```

### Step 3: Store Effective Lambda for Observability

Optionally store the computed multiplier on the episode for debugging:

```python
ep["_decay_multiplier"] = multiplier
```

## Test Specifications

### test_topic_decay.py

```python
import math
from episode_store import (
    decay_episodes, DEFAULT_LAMBDA_PER_HOUR,
    TOPIC_DECAY_MULTIPLIERS, DEFAULT_DECAY_MULTIPLIER,
)

# Test 1: Career episodes decay slower than fitness episodes
def test_career_decays_slower_than_fitness():
    """Career episodes should retain higher strength than fitness after same elapsed time."""
    now_iso = datetime.now(timezone.utc).isoformat()
    one_week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()

    shadow = {"episodes": [
        {"episode_id": "career1", "topic_signals": ["career"],
         "timestamp": one_week_ago, "retrieval_count": 0},
        {"episode_id": "fitness1", "topic_signals": ["fitness"],
         "timestamp": one_week_ago, "retrieval_count": 0},
    ]}
    decay_episodes(shadow)
    career_ep = shadow["episodes"][0]
    fitness_ep = shadow["episodes"][1]
    assert career_ep["decay_strength"] > fitness_ep["decay_strength"]

# Test 2: Multi-topic episode uses slowest decay
def test_multi_topic_uses_slowest_decay():
    """An episode tagged [career, fitness] should use career's slower rate."""
    one_week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
    shadow = {"episodes": [
        {"episode_id": "multi1", "topic_signals": ["career", "fitness"],
         "timestamp": one_week_ago, "retrieval_count": 0},
        {"episode_id": "career1", "topic_signals": ["career"],
         "timestamp": one_week_ago, "retrieval_count": 0},
    ]}
    decay_episodes(shadow)
    # Multi-topic should match career-only (career has lowest multiplier)
    assert shadow["episodes"][0]["decay_strength"] == shadow["episodes"][1]["decay_strength"]

# Test 3: No topics uses default rate
def test_no_topics_uses_default():
    """An episode with no topic_signals should use DEFAULT_DECAY_MULTIPLIER=1.0."""
    one_week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
    shadow = {"episodes": [
        {"episode_id": "none1", "topic_signals": [],
         "timestamp": one_week_ago, "retrieval_count": 0},
    ]}
    decay_episodes(shadow)
    hours = 168
    expected = math.exp(-DEFAULT_LAMBDA_PER_HOUR * DEFAULT_DECAY_MULTIPLIER * hours)
    assert abs(shadow["episodes"][0]["decay_strength"] - round(expected, 4)) < 0.001

# Test 4: Unknown topic uses default multiplier
def test_unknown_topic_default():
    """A topic not in TOPIC_DECAY_MULTIPLIERS should use DEFAULT_DECAY_MULTIPLIER."""
    one_week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
    shadow = {"episodes": [
        {"episode_id": "unk1", "topic_signals": ["unknown_topic"],
         "timestamp": one_week_ago, "retrieval_count": 0},
        {"episode_id": "prod1", "topic_signals": ["productivity"],
         "timestamp": one_week_ago, "retrieval_count": 0},
    ]}
    decay_episodes(shadow)
    # Both should have same strength (both multiplier=1.0)
    assert shadow["episodes"][0]["decay_strength"] == shadow["episodes"][1]["decay_strength"]

# Test 5: Retrieval boost still works with topic decay
def test_retrieval_boost_with_topic_decay():
    """Retrieval boost should stack on top of topic-aware decay."""
    one_week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
    shadow = {"episodes": [
        {"episode_id": "ret1", "topic_signals": ["career"],
         "timestamp": one_week_ago, "retrieval_count": 3},
        {"episode_id": "noret1", "topic_signals": ["career"],
         "timestamp": one_week_ago, "retrieval_count": 0},
    ]}
    decay_episodes(shadow)
    assert shadow["episodes"][0]["decay_strength"] > shadow["episodes"][1]["decay_strength"]

# Test 6: All multiplier values are positive
def test_all_multipliers_positive():
    """All topic decay multipliers should be positive numbers."""
    for topic, mult in TOPIC_DECAY_MULTIPLIERS.items():
        assert mult > 0, f"{topic} has non-positive multiplier: {mult}"
        assert mult <= 3.0, f"{topic} has unreasonably high multiplier: {mult}"

# Test 7: Decay function is monotonically decreasing over time
def test_decay_monotonic():
    """Older episodes should always have lower or equal strength (ignoring retrieval)."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    shadow = {"episodes": [
        {"episode_id": f"ep{i}", "topic_signals": ["career"],
         "timestamp": (now - timedelta(hours=i*24)).isoformat(),
         "retrieval_count": 0}
        for i in range(7)
    ]}
    decay_episodes(shadow)
    strengths = [ep["decay_strength"] for ep in shadow["episodes"]]
    for i in range(len(strengths) - 1):
        assert strengths[i] >= strengths[i+1]
```

## Estimated Impact

1. **Career/relationship episodes retain 2x longer**: At 1 week, career episode strength with multiplier 0.5 is `e^(-0.0025*168) = 0.657` vs `e^(-0.005*168) = 0.432` with uniform decay. This means career context persists meaningfully for ~2 weeks instead of ~1 week.

2. **Routine check-ins clear faster**: Fitness/sleep episodes with multiplier 1.2-1.3 decay ~25% faster, reducing noise in retrieval for routine topics.

3. **Minimal code change**: Only `decay_episodes()` changes (the loop body). The `TOPIC_DECAY_MULTIPLIERS` dict is the only new artifact. No database migration, no API changes.

4. **Backward compatible**: Episodes without `topic_signals` or with unknown topics use `DEFAULT_DECAY_MULTIPLIER=1.0`, producing identical behavior to current code.

5. **Aligns with FSRS principle**: FSRS demonstrates that per-item difficulty modulation reduces unnecessary reviews by 20-30%. The same principle applied to episodic memory means higher-stakes episodes surface more reliably in retrieval, while routine episodes clear out faster.
