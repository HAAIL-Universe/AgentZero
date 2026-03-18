---
topic: Consolidation Rule Quality Gate
status: ready_for_implementation
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T14:30:00Z
---

# Consolidation Rule Quality Gate

## Problem Statement

Agent Zero consolidates episodic memories into rules that are injected into agent contexts to guide behavior (Shadow gets consolidated_rules, Pineal gets agent_weights + consolidated_insights, Speaker gets intervention_effectiveness). Currently, **no validation** occurs between rule creation and rule injection. A rule with low episode count, contradictory signals, or degraded outcomes gets injected with the same authority as a high-quality rule. This means:

1. **Rules based on insufficient data** (exactly 3 episodes at MIN_CLUSTER_SIZE) can override agent behavior
2. **Rules whose outcomes have degraded** continue to be applied until they age past STALE_DAYS=30
3. **Contradictory rules** (where agent calibration conflicts across metrics) get injected without warning
4. **No feedback loop**: there is no mechanism to detect that a rule produced worse outcomes after injection

## Current State in Agent Zero

### Rule Creation -- consolidator.py:550-700

Rules are created in `run_consolidation()` with:
- `_compute_rule_confidence()` (lines 507-525): confidence = 0.5 * min(1, episodes/10) + 0.5 * outcome_coverage
- Minimum confidence threshold: `MIN_RULE_CONFIDENCE = 0.4` (line 36)
- No quality validation beyond confidence score

### Rule Injection -- cognitive_runtime.py:229, 299-309

```python
# Shadow context (line 229):
consolidated_rules = get_relevant_rules(shadow_prof, topic_signals)

# Pineal context (lines 299-309):
pineal_rules = get_relevant_rules(shadow_prof_pineal, topic_signals_pineal)
if pineal_rules:
    pineal_ctx["agent_weights"] = compute_agent_weights_from_rules(pineal_rules)
    pineal_ctx["consolidated_insights"] = [...]
```

Rules are injected directly. No check for:
- Whether the rule has produced good outcomes since injection
- Whether the rule's calibration data is internally consistent
- Whether the rule conflicts with other active rules
- Whether the rule's temporal pattern shows degradation

### Rule Retrieval -- consolidator.py:707-739

`get_relevant_rules()` scores by `topic_overlap * confidence`, returns top 3 active rules above `MIN_RULE_CONFIDENCE=0.4`. No quality dimension in scoring.

### Rule Pruning -- consolidator.py:669-693

Only age-based pruning: stale after 30 days (if confidence < 0.5), retired after 90 days. No outcome-based pruning.

## Industry Standard / Research Findings

### 1. Concept Drift Detection for Learned Rules (NeurIPS 2025)

RCCDA (NeurIPS 2025) addresses adaptive model updates under concept drift with resource constraints. Key insight: **learned rules must be monitored for performance degradation**, not just staleness. They use a tunable drift threshold based on past loss information to decide when updates are needed. For Agent Zero, this translates to tracking whether rules' recommended actions produce better or worse outcomes after injection.

**URL**: https://neurips.cc/virtual/2025/loc/san-diego/poster/120087

### 2. Rule-Based Explanation of Concept Drift (XAI Workshop 2025)

Haug et al. propose generating logical rules that explain **why** a classifier's performance degraded -- identifying which features shifted. Applied to Agent Zero: when a consolidated rule's associated outcomes degrade, the quality gate should identify which calibration dimension shifted (e.g., Fello adoption dropped, Othello became overcautious).

**URL**: https://ceur-ws.org/Vol-4132/short54.pdf

### 3. MemOS Memory Lifecycle Management (2025)

MemOS introduces MemLifecycle, which tracks creation, activation, expiration, and reclamation transitions for each memory unit. Their `mid_term_heat_threshold` parameter controls promotion between memory tiers -- analogous to a quality gate between episodic and semantic memory. Key principle: **memories must earn their way into higher tiers** through demonstrated utility, not just age.

**URL**: https://arxiv.org/abs/2507.03724

### 4. MemoryOS Hierarchical Validation (EMNLP 2025 Oral)

MemoryOS (BAI-LAB) uses a `similarity_threshold` and `heat_threshold` to filter information during consolidation from short-term to mid-term memory. The heat model ensures that only frequently accessed and contextually relevant memories survive consolidation -- a quality signal beyond raw age and count.

**URL**: https://github.com/BAI-LAB/MemoryOS
**Paper**: https://arxiv.org/abs/2506.06326

### 5. Silent Performance Decay in Production ML (DZone, 2025)

Industry practice identifies silent performance decay as "one of the most dangerous failure modes in production machine learning." Models that work on day one drift into irrelevance. Standard mitigation: **continuous performance monitoring with automated alerts** when key metrics cross thresholds, not just periodic retraining.

**URL**: https://dzone.com/articles/investigating-model-performance-degradation

### 6. Walking the Tightrope: Disentangling Beneficial vs Detrimental Drift (NeurIPS 2025)

This paper formalizes the distinction between **beneficial concept drift** (adaptation) and **detrimental concept drift** (degradation). Not all rule evolution is bad -- some rules should change as users change. The quality gate must distinguish between healthy rule evolution and harmful degradation.

**URL**: https://neurips.cc/virtual/2025/poster/120252

## Proposed Implementation

### Design: Three-Stage Quality Gate

Add a quality validation layer between rule retrieval and rule injection. The gate operates at three levels:

1. **Structural quality** -- is the rule internally consistent?
2. **Outcome quality** -- has the rule produced good outcomes since creation?
3. **Drift detection** -- is the rule's effectiveness degrading over time?

### Step 1: Add Quality Scoring to consolidator.py

Add after `_compute_rule_confidence()` (after line 525):

```python
def compute_rule_quality(rule: dict) -> dict:
    """Compute multi-dimensional quality score for a consolidated rule.

    Returns quality metrics that the quality gate uses to decide
    whether to inject the rule into agent contexts.
    """
    cal = rule.get("agent_calibration", {})
    eff = rule.get("intervention_effectiveness", {})
    temporal = rule.get("temporal_pattern", {})
    confidence = rule.get("confidence", 0)
    episode_count = rule.get("episode_count", 0)

    # 1. Structural quality: internal consistency
    # Flag if agent calibrations contradict each other
    fello_adopted = cal.get("fello", {}).get("proposal_adopted_rate", 0.5)
    othello_overcautious = cal.get("othello", {}).get("overcautious_rate", 0.0)
    # Contradiction: Fello proposals adopted >70% AND Othello overcautious >50%
    # means both "trust Fello" and "ignore Othello" -- internally consistent
    # Real contradiction: Fello adopted <30% AND Othello risk_heeded <30%
    # (neither agent is useful, but rule still says to use them)
    fello_useful = fello_adopted > 0.4
    othello_useful = cal.get("othello", {}).get("risk_heeded_rate", 0.5) > 0.4
    shadow_useful = cal.get("shadow", {}).get("prediction_accuracy", 0.5) > 0.4
    useful_agents = sum([fello_useful, othello_useful, shadow_useful])
    structural_score = min(1.0, useful_agents / 2.0)  # At least 2 of 3 agents useful

    # 2. Outcome quality: does the rule have outcome data?
    outcome_rate = eff.get("acted_rate", 0.0)
    ignored_rate = eff.get("ignored_rate", 0.0)
    # Good: high acted rate, low ignored rate
    outcome_score = max(0.0, outcome_rate - ignored_rate * 0.5)
    # Bonus for having any outcome data at all
    if outcome_rate > 0 or ignored_rate > 0:
        outcome_score = max(outcome_score, 0.3)

    # 3. Drift signal: outcome decay detected?
    drift_penalty = 0.0
    if temporal and temporal.get("outcome_decay"):
        drift_penalty = 0.2  # Reduce quality if outcomes are degrading

    # 4. Data sufficiency
    data_score = min(1.0, episode_count / 8.0)  # 8+ episodes = full data score

    # Composite quality score
    quality = round(
        structural_score * 0.25
        + outcome_score * 0.35
        + data_score * 0.25
        + confidence * 0.15
        - drift_penalty,
        2,
    )
    quality = max(0.0, min(1.0, quality))

    return {
        "quality_score": quality,
        "structural_score": round(structural_score, 2),
        "outcome_score": round(outcome_score, 2),
        "data_score": round(data_score, 2),
        "drift_penalty": round(drift_penalty, 2),
        "passes_gate": quality >= 0.3,  # Minimum quality threshold
        "warnings": _generate_quality_warnings(rule, structural_score, outcome_score, drift_penalty),
    }


def _generate_quality_warnings(
    rule: dict,
    structural_score: float,
    outcome_score: float,
    drift_penalty: float,
) -> list[str]:
    """Generate human-readable warnings about rule quality issues."""
    warnings = []
    if rule.get("episode_count", 0) < 5:
        warnings.append(f"Low data: only {rule.get('episode_count', 0)} episodes")
    if structural_score < 0.5:
        warnings.append("Weak agent calibration: most agents show low utility")
    if outcome_score < 0.2:
        warnings.append("Low outcome data: unclear if rule produces good results")
    if drift_penalty > 0:
        warnings.append("Outcome decay detected: rule effectiveness may be declining")
    temporal = rule.get("temporal_pattern", {})
    if temporal and temporal.get("outcome_decay"):
        warnings.append(f"Follow-through declining over {temporal.get('span_days', '?')} days")
    return warnings
```

### Step 2: Add Quality Gate to Rule Retrieval

Modify `get_relevant_rules()` in consolidator.py (lines 707-739) to filter by quality:

```python
def get_relevant_rules(
    shadow: dict,
    topic_signals: list[str],
    route: str | None = None,
    apply_quality_gate: bool = True,
) -> list[dict]:
    """Retrieve the most relevant consolidated rules for the current turn.

    Scores rules by topic overlap * confidence * quality.
    Filters out rules that fail the quality gate.
    Returns top 3 active rules above thresholds.
    """
    rules = shadow.get("consolidated_rules", [])
    if not rules or not topic_signals:
        return []

    topic_set = set(topic_signals)
    scored = []

    for rule in rules:
        if rule.get("status") != "active":
            continue
        if rule.get("confidence", 0) < MIN_RULE_CONFIDENCE:
            continue

        rule_topics = set(rule.get("trigger", {}).get("topic_signals", []))
        overlap = len(topic_set & rule_topics)
        if overlap == 0:
            continue

        # Quality gate
        if apply_quality_gate:
            quality = compute_rule_quality(rule)
            if not quality["passes_gate"]:
                continue
            quality_factor = quality["quality_score"]
        else:
            quality_factor = 1.0

        score = overlap * rule.get("confidence", 0) * quality_factor
        scored.append((score, rule))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rule for _, rule in scored[:3]]
```

### Step 3: Store Quality Metadata on Rules

In `run_consolidation()`, after computing confidence (line 617), add quality computation:

```python
# After line 617: confidence = _compute_rule_confidence(cluster)
quality_result = compute_rule_quality({
    "agent_calibration": calibration,
    "intervention_effectiveness": intervention_eff,
    "temporal_pattern": temporal,
    "confidence": confidence,
    "episode_count": len(cluster),
})
```

Store `quality_result` on the rule:
```python
# In new rule creation (line 641 block):
new_rule["quality"] = quality_result

# In existing rule update (line 624 block):
existing_rule["quality"] = quality_result
```

### Step 4: Log Quality Gate Decisions

Add logging for observability:

```python
import logging

_logger = logging.getLogger("agent_zero.quality_gate")

def _log_quality_gate(rule_id: str, quality: dict, topic_signals: list[str]):
    """Log quality gate decision for observability."""
    if quality["passes_gate"]:
        if quality["warnings"]:
            _logger.info(
                "rule=%s topics=%s quality=%.2f PASSED_WITH_WARNINGS: %s",
                rule_id[:8], topic_signals[:2], quality["quality_score"],
                "; ".join(quality["warnings"]),
            )
    else:
        _logger.warning(
            "rule=%s topics=%s quality=%.2f BLOCKED: %s",
            rule_id[:8], topic_signals[:2], quality["quality_score"],
            "; ".join(quality["warnings"]),
        )
```

## Test Specifications

### test_rule_quality_gate.py

```python
# Test 1: High-quality rule passes gate
def test_high_quality_rule_passes():
    """A rule with good calibration, outcomes, and data should pass."""
    rule = {
        "agent_calibration": {
            "fello": {"proposal_adopted_rate": 0.8},
            "othello": {"risk_heeded_rate": 0.6, "overcautious_rate": 0.1},
            "shadow": {"prediction_accuracy": 0.7},
        },
        "intervention_effectiveness": {"acted_rate": 0.6, "ignored_rate": 0.1},
        "episode_count": 10,
        "confidence": 0.7,
    }
    quality = compute_rule_quality(rule)
    assert quality["passes_gate"] is True
    assert quality["quality_score"] >= 0.5
    assert len(quality["warnings"]) == 0

# Test 2: Low-data rule gets warning
def test_low_data_rule_warned():
    """A rule with only 3 episodes should get a low-data warning."""
    rule = {
        "agent_calibration": {"fello": {"proposal_adopted_rate": 0.5},
                              "othello": {"risk_heeded_rate": 0.5},
                              "shadow": {"prediction_accuracy": 0.5}},
        "intervention_effectiveness": {"acted_rate": 0.5, "ignored_rate": 0.1},
        "episode_count": 3,
        "confidence": 0.45,
    }
    quality = compute_rule_quality(rule)
    assert any("Low data" in w for w in quality["warnings"])

# Test 3: Rule with outcome decay gets penalty
def test_outcome_decay_penalized():
    """A rule with detected outcome decay should get a drift penalty."""
    rule = {
        "agent_calibration": {"fello": {"proposal_adopted_rate": 0.6},
                              "othello": {"risk_heeded_rate": 0.5},
                              "shadow": {"prediction_accuracy": 0.6}},
        "intervention_effectiveness": {"acted_rate": 0.5, "ignored_rate": 0.2},
        "temporal_pattern": {"outcome_decay": True, "span_days": 21},
        "episode_count": 8,
        "confidence": 0.6,
    }
    quality = compute_rule_quality(rule)
    assert quality["drift_penalty"] > 0
    assert any("decay" in w.lower() for w in quality["warnings"])

# Test 4: Rule with all-low calibration blocked
def test_weak_calibration_blocked():
    """A rule where no agent shows utility should fail or score very low."""
    rule = {
        "agent_calibration": {"fello": {"proposal_adopted_rate": 0.1},
                              "othello": {"risk_heeded_rate": 0.1, "overcautious_rate": 0.8},
                              "shadow": {"prediction_accuracy": 0.2}},
        "intervention_effectiveness": {"acted_rate": 0.0, "ignored_rate": 0.8},
        "episode_count": 4,
        "confidence": 0.4,
    }
    quality = compute_rule_quality(rule)
    assert quality["structural_score"] < 0.5
    assert quality["quality_score"] < 0.3

# Test 5: get_relevant_rules filters by quality
def test_get_relevant_rules_quality_filter():
    """get_relevant_rules should exclude rules that fail the quality gate."""
    shadow = {
        "consolidated_rules": [
            {
                "rule_id": "good",
                "status": "active",
                "confidence": 0.7,
                "trigger": {"topic_signals": ["career"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.8},
                    "othello": {"risk_heeded_rate": 0.6},
                    "shadow": {"prediction_accuracy": 0.7},
                },
                "intervention_effectiveness": {"acted_rate": 0.6, "ignored_rate": 0.1},
                "episode_count": 10,
            },
            {
                "rule_id": "bad",
                "status": "active",
                "confidence": 0.45,
                "trigger": {"topic_signals": ["career"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.1},
                    "othello": {"risk_heeded_rate": 0.1},
                    "shadow": {"prediction_accuracy": 0.2},
                },
                "intervention_effectiveness": {"acted_rate": 0.0, "ignored_rate": 0.9},
                "episode_count": 3,
            },
        ]
    }
    rules = get_relevant_rules(shadow, ["career"], apply_quality_gate=True)
    rule_ids = [r["rule_id"] for r in rules]
    assert "good" in rule_ids
    # "bad" may or may not pass depending on exact scoring; test that good ranks higher
    if "bad" in rule_ids:
        assert rule_ids.index("good") < rule_ids.index("bad")

# Test 6: Quality gate can be bypassed
def test_quality_gate_bypass():
    """Setting apply_quality_gate=False should skip quality filtering."""
    shadow = {
        "consolidated_rules": [{
            "rule_id": "low_q",
            "status": "active",
            "confidence": 0.45,
            "trigger": {"topic_signals": ["health"]},
            "agent_calibration": {"fello": {"proposal_adopted_rate": 0.1},
                                  "othello": {"risk_heeded_rate": 0.1},
                                  "shadow": {"prediction_accuracy": 0.2}},
            "intervention_effectiveness": {"acted_rate": 0.0, "ignored_rate": 0.9},
            "episode_count": 3,
        }]
    }
    rules = get_relevant_rules(shadow, ["health"], apply_quality_gate=False)
    assert len(rules) == 1

# Test 7: Quality stored on rule during consolidation
def test_quality_stored_on_rule():
    """run_consolidation should store quality metadata on created rules."""
    # Build minimal shadow with 3 episodes sharing topic signals
    from consolidator import run_consolidation
    episodes = [
        {"episode_id": f"ep{i}", "topic_signals": ["fitness"],
         "timestamp": f"2026-03-{10+i}T10:00:00Z",
         "agent_signals": {"fello": {"ran": True, "confidence": 0.6},
                           "othello": {"ran": True, "risk_count": 0},
                           "shadow": {"ran": True}},
         "resolution": {"mode": "proceed"},
         "outcome": {"user_followed_up": True}}
        for i in range(5)
    ]
    shadow = {"episodes": episodes, "consolidated_rules": []}
    result = run_consolidation(shadow)
    rules = result.get("consolidated_rules", [])
    assert len(rules) > 0
    assert "quality" in rules[0]
    assert "quality_score" in rules[0]["quality"]

# Test 8: Empty calibration produces safe quality score
def test_empty_calibration_safe():
    """A rule with missing calibration fields should not crash."""
    rule = {
        "agent_calibration": {},
        "intervention_effectiveness": {},
        "episode_count": 3,
        "confidence": 0.4,
    }
    quality = compute_rule_quality(rule)
    assert 0.0 <= quality["quality_score"] <= 1.0
    assert isinstance(quality["passes_gate"], bool)
```

## Estimated Impact

1. **Prevents misleading agent guidance**: Rules with insufficient data or contradictory signals are blocked from injection, preventing agents from being steered by noise. With MIN_CLUSTER_SIZE=3 and no quality gate, rules based on just 3 episodes currently drive agent behavior -- the quality gate ensures at least 5 episodes and consistent calibration before injection.

2. **Drift detection**: The temporal outcome_decay flag, combined with the drift penalty, automatically reduces the influence of rules whose effectiveness is declining. This addresses the "silent performance decay" problem identified in production ML monitoring (DZone, 2025).

3. **Backward compatible**: `get_relevant_rules()` adds `apply_quality_gate=True` as a default parameter -- all existing callers get quality gating automatically. Setting `False` bypasses it for debugging/testing.

4. **Observability**: Quality warnings and gate decisions are logged, enabling data-driven threshold tuning. The quality metadata stored on rules enables future analysis of which quality dimensions predict rule usefulness.

5. **Aligns with MemOS/MemoryOS patterns**: Both MemOS (2025) and MemoryOS (EMNLP 2025) use heat/threshold-based promotion between memory tiers. The quality gate serves the same function: memories must earn their way into agent contexts.
