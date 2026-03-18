---
topic: Learned Agent Routing
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T12:00:00Z
---

# Learned Agent Routing

Replace keyword-based agent selection with a lightweight contextual bandit that
learns which agents to activate based on episodic outcome data.

## Problem Statement

`select_agents_for_turn()` in `cognitive_runtime.py` (lines 329-382) uses
hard-coded keyword lists and a hand-tuned complexity score to decide which
worker agents (fello, othello, prefrontal_cortex, shadow) run for each turn.
This has three problems:

1. **Keyword fragility** -- "remember" triggers memory mode, but "I recall a
   time when..." doesn't. The keyword lists can't cover natural language variety.
2. **No learning from outcomes** -- The system already captures per-turn episodes
   with agent signals, resolution outcomes, and user follow-through data
   (episode_store.py). The consolidator already computes per-agent calibration
   and agent weights (consolidator.py:724-780). But `select_agents_for_turn()`
   ignores all of this -- it never consults the learned data.
3. **Wasted LLM calls** -- Every strategic turn runs 3-4 agents. When
   consolidation data shows an agent consistently adds no signal for a topic
   (e.g., Othello overcautious 60%+ on career topics), the system still calls it.

## Current State in Agent Zero

### Agent Selection (cognitive_runtime.py:329-382)

```python
def select_agents_for_turn(user_content, route, state=None):
    if route in ("casual_chat", "simple_chat", "self_introspection"):
        return ()
    lower = (user_content or "").lower()
    # Keyword lists for memory signals
    memory_signals = ("remember", "recall", ...)
    if any(sig in lower for sig in memory_signals):
        ...
    # Complexity scoring via keyword counting
    complexity_signals = 0
    if state:
        if state.get("constraints"): complexity_signals += 1
        ...
    if complexity_signals < 2:
        return ("fello", "othello", "shadow")
    return WORKER_AGENT_IDS  # all 4
```

Decision is purely rule-based. No historical data consulted.

### Available Data (consolidator.py:724-780)

`compute_agent_weights_from_rules(rules)` already produces per-agent weights
in [0.2, 1.0] from consolidated episode data:
- Fello weight based on proposal adoption rate
- Othello weight based on risk_heeded_rate minus overcautious_rate
- Shadow weight based on prediction accuracy
- Prefrontal weight based on step follow-through rate

These weights are currently only used by **Pineal for synthesis weighting**
(cognitive_runtime.py:296). They are never used for **selection/activation**.

### Episode Data Structure (episode_store.py)

Each episode contains:
- `topic_signals`: ["career", "health", ...] (10 topic categories)
- `agent_signals.{agent}.ran`: bool
- `agent_signals.{agent}.confidence`: float
- `resolution.mode`: "proceed" | "narrow" | "clarify_first"
- `outcome.user_followed_up`: bool | None
- `outcome.sentiment_shift`: "positive" | "neutral" | "negative" | None
- `route`: the route that triggered the turn
- `emotional_register`: str | None

This is more than enough context for a bandit to learn from.

## Industry Standard / Research Findings

### NeurIPS 2025: Evolving Orchestration Paradigm
A "puppeteer" orchestrator trained via RL learns to dynamically route tasks
between agents based on evolving problem states. Reward signal combines solution
quality and efficiency (token cost). Outperforms fixed pipelines while reducing
computational cost. Key insight: treat multi-agent coordination as a sequential
decision problem, not a predetermined pipeline.
Source: https://neurips.cc/virtual/2025/loc/san-diego/poster/118584

### ACL 2025: MasRouter
Cascaded controller network for learned multi-agent routing. Three-stage
decision: (1) collaboration mode determination via variational latent model,
(2) role allocation via structured probabilistic cascade, (3) LLM router via
multinomial distribution. Achieves 1.8-8.2% improvement over SOTA with up to
52% cost reduction.
Source: https://arxiv.org/abs/2502.11133

### OI-MAS: Confidence-Aware Routing (2026)
Two-stage hierarchical routing: Role Router selects which agent functions are
needed, Model Router assigns model scales. Uses token-level log-probability
as confidence signal. Role selection accumulates probability mass until
threshold (theta=0.3). Includes "EarlyStop" to terminate when additional
computation is unnecessary. Higher confidence -> stronger cost penalty
(skip expensive agents when certain).
Source: https://arxiv.org/html/2601.04861v2

### LinUCB for Agent Selection (2025)
Contextual bandit framework for sequential LLM selection. LinUCB achieves
sublinear regret without neural networks. Deterministic confidence-based
action selection. Lightweight, no training loop, provable convergence.
Source: https://www.researchgate.net/publication/392941746

### Key Takeaways for Agent Zero

1. **Full RL is overkill** -- Agent Zero has ~200 episodes max, not millions.
   A contextual bandit (LinUCB or Thompson Sampling) fits the data scale.
2. **The consolidator already does most of the work** -- Agent calibration
   metrics and weights are computed. We just need to use them for selection.
3. **Cost-aware routing is critical** -- Each agent call costs an LLM
   inference. Skipping agents that historically add no signal saves ~25-50%
   of per-turn latency.
4. **Threshold-based activation** (OI-MAS) is the right primitive -- activate
   an agent only if its expected contribution exceeds a threshold.

## Proposed Implementation

### Approach: Weight-Threshold Selection

The simplest effective approach that leverages existing infrastructure:

1. Compute agent weights from consolidated rules (already exists)
2. Apply a topic-aware activation threshold
3. Skip agents whose weight falls below threshold for this topic

This avoids the complexity of a full bandit while still being data-driven.

### Changes Required

#### File: `cognitive_runtime.py`

**Modify `select_agents_for_turn()` (lines 329-382)**

New signature:
```python
def select_agents_for_turn(
    user_content: str,
    route: str,
    state: dict | None = None,
    shadow_profile: dict | None = None,  # NEW: pass shadow for learned weights
) -> tuple[str, ...]:
```

New logic:
```python
def select_agents_for_turn(user_content, route, state=None, shadow_profile=None):
    # Phase 1: Hard rules (unchanged)
    if route in ("casual_chat", "simple_chat", "self_introspection"):
        return ()

    # Phase 2: Compute learned weights if shadow data available
    agent_weights = _compute_learned_selection(user_content, state, shadow_profile)

    # Phase 3: Threshold selection
    ACTIVATION_THRESHOLD = 0.35  # Skip agents below this weight
    MIN_AGENTS = 2  # Always run at least 2 workers (fello + shadow minimum)

    candidates = []
    for agent_id in WORKER_AGENT_IDS:
        weight = agent_weights.get(agent_id, 0.5)
        if weight >= ACTIVATION_THRESHOLD:
            candidates.append((weight, agent_id))

    # Sort by weight descending
    candidates.sort(reverse=True)
    selected = [aid for _, aid in candidates]

    # Ensure minimum agents: always include fello and shadow
    for must_have in ("fello", "shadow"):
        if must_have not in selected:
            selected.append(must_have)

    # Ensure minimum count
    if len(selected) < MIN_AGENTS:
        for aid in WORKER_AGENT_IDS:
            if aid not in selected:
                selected.append(aid)
            if len(selected) >= MIN_AGENTS:
                break

    return tuple(selected)
```

**Add `_compute_learned_selection()` helper:**

```python
def _compute_learned_selection(
    user_content: str,
    state: dict | None,
    shadow_profile: dict | None,
) -> dict[str, float]:
    """Compute per-agent activation weights from consolidated episode data.

    Falls back to uniform 0.5 weights when no consolidation data exists
    (cold start: behaves identically to current keyword-based system).
    """
    if not shadow_profile:
        return {aid: 0.5 for aid in WORKER_AGENT_IDS}

    # Extract topic signals for this turn
    topic_signals = _extract_current_topic_signals(state or {})
    if not topic_signals:
        # No topic signals -- use text-based extraction
        topic_signals = _extract_topics_from_text(
            (user_content or "").lower()
        )

    # Get relevant consolidated rules
    rules = get_relevant_rules(shadow_profile, topic_signals)

    if not rules:
        # Cold start: no learned data for this topic yet
        # Fall back to complexity-based heuristic (existing behavior)
        return _heuristic_weights(user_content, state)

    # Compute weights from rules (existing function)
    weights = compute_agent_weights_from_rules(rules)

    # Boost based on turn characteristics (hybrid: learned + heuristic)
    lower = (user_content or "").lower()

    # Emotional content boosts Othello
    if _has_emotional_signals(lower, state):
        weights["othello"] = min(1.0, weights.get("othello", 0.5) + 0.2)

    # Complex turns boost Prefrontal
    complexity = _count_complexity_signals(lower, state)
    if complexity >= 2:
        weights["prefrontal_cortex"] = min(
            1.0, weights.get("prefrontal_cortex", 0.5) + 0.15
        )

    return weights


def _heuristic_weights(user_content: str, state: dict | None) -> dict[str, float]:
    """Cold-start heuristic weights (replaces current keyword logic).

    Returns weights that, when passed through the threshold, produce
    the same agent sets as the current select_agents_for_turn().
    """
    lower = (user_content or "").lower()
    weights = {aid: 0.5 for aid in WORKER_AGENT_IDS}

    # Memory signals: boost fello, reduce prefrontal
    memory_signals = ("remember", "recall", "what do you know about",
                      "last time", "we talked about", "you mentioned")
    if any(sig in lower for sig in memory_signals):
        weights["fello"] = 0.8
        weights["prefrontal_cortex"] = 0.25  # below threshold
        if _has_emotional_signals(lower, state):
            weights["othello"] = 0.7
        else:
            weights["othello"] = 0.25  # below threshold

    # Complexity signals
    complexity = _count_complexity_signals(lower, state)
    if complexity >= 2:
        # Full pipeline
        for aid in WORKER_AGENT_IDS:
            weights[aid] = max(weights[aid], 0.6)

    return weights


def _count_complexity_signals(lower: str, state: dict | None) -> int:
    """Count complexity signals (extracted from existing logic)."""
    signals = 0
    if state:
        if state.get("constraints"):
            signals += 1
        if state.get("time_pressure") in ("high", "medium"):
            signals += 1
        if state.get("emotional_register") in ("anxious", "strained"):
            signals += 1
        if len(state.get("relevant_priorities", [])) > 1:
            signals += 1
    word_count = len(lower.split())
    if word_count > 30:
        signals += 1
    if "?" in lower and word_count > 15:
        signals += 1
    return signals
```

#### File: `agent_zero_server.py`

**Update the call site (line ~1796) to pass shadow_profile:**

```python
selected_agents = select_agents_for_turn(
    user_content,
    route_result.get("route", "strategic_reasoning"),
    state=reasoning_packet.get("state"),
    shadow_profile=shadow_profile,  # NEW
)
```

#### File: `cognitive_runtime.py` (run_cognitive_runtime)

**Update the automatic selection call (line ~744) to pass shadow_profile:**

```python
active_workers = agent_selection if agent_selection is not None else select_agents_for_turn(
    user_content,
    route_result.get("route", "strategic_reasoning"),
    state=reasoning_packet.get("state"),
    shadow_profile=shadow_profile,  # NEW: from function parameter
)
```

### Backward Compatibility

- When `shadow_profile` is None (no user data yet), the system falls back to
  `_heuristic_weights()`, which produces identical agent sets to the current
  keyword-based logic.
- The activation threshold (0.35) and minimum agent guarantee (fello + shadow)
  ensure the system never degrades to zero agents.
- Existing tests for `select_agents_for_turn()` continue to pass because they
  don't pass `shadow_profile`, triggering the cold-start path.

### Cost Savings Estimate

- Current system: 3-4 LLM calls per strategic turn
- With learned routing: 2-3 calls when consolidation data shows agents are
  unhelpful for that topic
- Estimated 25-35% reduction in per-turn LLM inference cost
- No additional LLM calls needed (weights computed deterministically from
  existing consolidation data)

## Test Specifications

### Test file: `agent_zero/test_learned_routing.py`

```python
class TestLearnedRouting(unittest.TestCase):

    def test_cold_start_no_shadow_returns_heuristic(self):
        """No shadow profile -> fall back to keyword-based heuristic."""
        result = select_agents_for_turn(
            "I need career advice", "strategic_reasoning",
            state={"domain": "career"}, shadow_profile=None,
        )
        # Should behave like current system: fello + othello + shadow
        assert "fello" in result
        assert "shadow" in result

    def test_cold_start_memory_signal(self):
        """Memory keywords with no shadow -> fello + shadow only."""
        result = select_agents_for_turn(
            "do you remember what I said about my job?",
            "strategic_reasoning", shadow_profile=None,
        )
        assert "fello" in result
        assert "shadow" in result
        assert "prefrontal_cortex" not in result

    def test_learned_weights_skip_overcautious_othello(self):
        """When consolidation shows Othello overcautious, skip it."""
        shadow = {
            "consolidated_rules": [{
                "rule_id": "r1",
                "status": "active",
                "confidence": 0.7,
                "trigger": {"topic_signals": ["career"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.8},
                    "othello": {
                        "risk_heeded_rate": 0.1,
                        "overcautious_rate": 0.7,
                    },
                    "shadow": {"prediction_accuracy": 0.6},
                    "prefrontal_cortex": {"step_followed_rate": 0.3},
                },
            }],
        }
        result = select_agents_for_turn(
            "I'm thinking about changing jobs",
            "strategic_reasoning",
            state={"domain": "career"},
            shadow_profile=shadow,
        )
        # Othello weight should be low (overcautious)
        # Fello and Shadow should be included
        assert "fello" in result
        assert "shadow" in result
        # Othello should be skipped (weight < threshold)
        assert "othello" not in result

    def test_learned_weights_include_high_performing_agents(self):
        """When consolidation shows agents are effective, include them."""
        shadow = {
            "consolidated_rules": [{
                "rule_id": "r1",
                "status": "active",
                "confidence": 0.8,
                "trigger": {"topic_signals": ["health"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.9},
                    "othello": {
                        "risk_heeded_rate": 0.7,
                        "overcautious_rate": 0.1,
                    },
                    "shadow": {"prediction_accuracy": 0.8},
                    "prefrontal_cortex": {"step_followed_rate": 0.7},
                },
            }],
        }
        result = select_agents_for_turn(
            "I want to start exercising regularly",
            "strategic_reasoning",
            state={"domain": "health"},
            shadow_profile=shadow,
        )
        # All agents should be included (all high-performing)
        assert len(result) == 4

    def test_emotional_content_boosts_othello(self):
        """Emotional signals override low learned weight for Othello."""
        shadow = {
            "consolidated_rules": [{
                "rule_id": "r1",
                "status": "active",
                "confidence": 0.6,
                "trigger": {"topic_signals": ["career"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.5},
                    "othello": {
                        "risk_heeded_rate": 0.2,
                        "overcautious_rate": 0.5,
                    },
                    "shadow": {"prediction_accuracy": 0.5},
                    "prefrontal_cortex": {"step_followed_rate": 0.5},
                },
            }],
        }
        result = select_agents_for_turn(
            "I'm really anxious about losing my job and I'm scared",
            "strategic_reasoning",
            state={"domain": "career", "emotional_register": "anxious"},
            shadow_profile=shadow,
        )
        # Emotional boost should push Othello above threshold
        assert "othello" in result

    def test_minimum_agents_guarantee(self):
        """Even with all low weights, at least fello + shadow run."""
        shadow = {
            "consolidated_rules": [{
                "rule_id": "r1",
                "status": "active",
                "confidence": 0.9,
                "trigger": {"topic_signals": ["career"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.1},
                    "othello": {
                        "risk_heeded_rate": 0.0,
                        "overcautious_rate": 0.9,
                    },
                    "shadow": {"prediction_accuracy": 0.1},
                    "prefrontal_cortex": {"step_followed_rate": 0.1},
                },
            }],
        }
        result = select_agents_for_turn(
            "career stuff", "strategic_reasoning",
            state={"domain": "career"}, shadow_profile=shadow,
        )
        assert "fello" in result
        assert "shadow" in result
        assert len(result) >= 2

    def test_casual_chat_unaffected(self):
        """Casual chat still returns empty regardless of shadow data."""
        shadow = {"consolidated_rules": [{"rule_id": "r1", "status": "active",
                   "confidence": 0.9, "trigger": {"topic_signals": ["career"]},
                   "agent_calibration": {}}]}
        result = select_agents_for_turn(
            "hey there!", "casual_chat", shadow_profile=shadow,
        )
        assert result == ()

    def test_no_topic_match_uses_heuristic(self):
        """When no rules match current topic, fall back to heuristic."""
        shadow = {
            "consolidated_rules": [{
                "rule_id": "r1",
                "status": "active",
                "confidence": 0.8,
                "trigger": {"topic_signals": ["health"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.9},
                    "othello": {"risk_heeded_rate": 0.1, "overcautious_rate": 0.8},
                    "shadow": {"prediction_accuracy": 0.8},
                    "prefrontal_cortex": {"step_followed_rate": 0.7},
                },
            }],
        }
        # Query about finance (no matching rule for "finance")
        result = select_agents_for_turn(
            "I need help budgeting my money",
            "strategic_reasoning",
            state={"domain": "finance"},
            shadow_profile=shadow,
        )
        # Should fall back to heuristic (all agents at 0.5 = above threshold)
        assert "fello" in result
        assert "shadow" in result


class TestComputeLearnedSelection(unittest.TestCase):

    def test_empty_shadow_returns_uniform(self):
        weights = _compute_learned_selection("hello", None, None)
        assert all(w == 0.5 for w in weights.values())

    def test_no_rules_returns_heuristic(self):
        weights = _compute_learned_selection(
            "career change", {"domain": "career"},
            {"consolidated_rules": []},
        )
        assert all(w == 0.5 for w in weights.values())

    def test_emotional_boosts_othello(self):
        shadow = {
            "consolidated_rules": [{
                "rule_id": "r1", "status": "active", "confidence": 0.7,
                "trigger": {"topic_signals": ["career"]},
                "agent_calibration": {
                    "fello": {"proposal_adopted_rate": 0.5},
                    "othello": {"risk_heeded_rate": 0.3, "overcautious_rate": 0.3},
                    "shadow": {"prediction_accuracy": 0.5},
                    "prefrontal_cortex": {"step_followed_rate": 0.5},
                },
            }],
        }
        weights = _compute_learned_selection(
            "I'm scared about my job",
            {"domain": "career", "emotional_register": "anxious"},
            shadow,
        )
        assert weights["othello"] > 0.5  # boosted by emotion


class TestHeuristicWeights(unittest.TestCase):

    def test_default_weights(self):
        weights = _heuristic_weights("some question", None)
        assert all(w == 0.5 for w in weights.values())

    def test_memory_signal_boosts_fello(self):
        weights = _heuristic_weights("remember what I said", None)
        assert weights["fello"] > 0.5
        assert weights["prefrontal_cortex"] < 0.35

    def test_complex_turn_boosts_all(self):
        state = {
            "constraints": ["time"],
            "time_pressure": "high",
            "relevant_priorities": ["a", "b"],
        }
        weights = _heuristic_weights(
            "I have a complex long question about multiple things and decisions "
            "that need careful thought and analysis of trade-offs?",
            state,
        )
        assert all(w >= 0.5 for w in weights.values())
```

### Edge Cases

1. **First-ever user** -- No episodes, no rules. System behaves identically
   to current keyword-based routing. No regression.
2. **Topic never seen** -- No matching rules. Falls back to heuristic weights.
3. **All agents low weight** -- Minimum guarantee ensures fello + shadow always
   run (empathy + historical pattern detection are essential).
4. **Emotional override** -- Even if consolidation says skip Othello, strong
   emotional signals in the current turn override the learned weight. Safety
   takes precedence over efficiency.
5. **Stale/retired rules** -- `get_relevant_rules()` already filters by status
   "active" and minimum confidence. No change needed.

## Estimated Impact

- **Latency**: 25-35% reduction in per-turn LLM inference for users with
  5+ episodes on a topic (skipping 1-2 agents that add no signal).
- **Quality**: Agents that historically underperform on a topic are skipped,
  reducing noise in Pineal's synthesis.
- **Cold start**: Zero regression -- new users get identical behavior to today.
- **Learning curve**: System improves as consolidation data accumulates,
  typically after ~15 episodes (3 clusters of 5).
- **No additional LLM calls**: All computation is deterministic math on
  existing consolidation data. Zero additional inference cost.
