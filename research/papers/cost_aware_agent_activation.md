---
topic: Cost-Aware Agent Activation
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T18:00:00Z
---

# Cost-Aware Agent Activation

Skip cognitive agents when consolidation data shows they historically add no
signal for the current topic. Penalise unnecessary calls to reduce latency and
token cost without degrading response quality.

## Problem Statement

Every strategic turn in Agent Zero currently runs 2-4 worker agents (fello,
othello, prefrontal_cortex, shadow). The learned routing upgrade
(`_compute_learned_selection` in `cognitive_runtime.py:380-424`) adjusts
per-agent activation weights based on consolidation data, but:

1. **No cost penalty** -- An agent that consistently adds zero signal (e.g.,
   Othello overcautious 60%+ on career topics, Prefrontal steps never followed
   on emotional topics) is merely *weighted lower* but still runs if its weight
   exceeds 0.35. There is no explicit mechanism to suppress an agent that wastes
   an LLM call.

2. **Minimum agent floor is too rigid** -- `MIN_AGENTS = 2` forces at least
   fello + shadow on every strategic turn regardless of topic evidence. For
   well-understood topics (high consolidation confidence), the system should be
   able to run a single agent when data supports it.

3. **No value-of-information calculation** -- The current threshold is static
   (0.35). Research shows that adaptive thresholds based on expected value of
   information (VOI) yield 22-34% cost reductions without quality loss (see
   citations below).

4. **No per-agent cost tracking** -- As of 2026-03-18, all cognitive agents are
   hybrid (LLM-backed via Qwen3). Every skipped agent saves a full vLLM
   inference call (~1500-2000 tokens, ~1-3s latency). The activation decision
   should be cost-weighted, with shadow costing more due to larger context.

## Current State in Agent Zero

### Agent Selection (`cognitive_runtime.py:329-377`)

```python
def select_agents_for_turn(user_content, route, state=None, shadow_profile=None):
    if route in ("casual_chat", "simple_chat", "self_introspection"):
        return ()
    agent_weights = _compute_learned_selection(user_content, state, shadow_profile)
    ACTIVATION_THRESHOLD = 0.35
    MIN_AGENTS = 2
    candidates = []
    for agent_id in WORKER_AGENT_IDS:
        weight = agent_weights.get(agent_id, 0.5)
        if weight >= ACTIVATION_THRESHOLD:
            candidates.append((weight, agent_id))
    candidates.sort(reverse=True)
    selected = [aid for _, aid in candidates]
    # Force fello and shadow
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

### Agent Execution Cost (`cognitive_agents.py:53-80`)

**UPDATE (2026-03-18):** All 7 cognitive agents now have `execution_mode: hybrid`
in their prompt frontmatter (`agent_zero/prompts/cognitive_agents/*.md`). This means
every agent invocation is a full LLM call to Qwen3-235B via vLLM when the model
is loaded. Deterministic fallback only fires when the LLM call fails or the model
isn't loaded.

- Each hybrid agent call: ~500 max_tokens output, ~1500-2000 tokens context,
  ~1-3s latency via vLLM.
- Shadow is the most expensive worker (largest context: shadow profile, episodes,
  consolidated rules).
- Pineal always runs after workers (synthesis), cost is fixed regardless of
  worker count.
- **Every skipped worker agent saves a full LLM call** -- not just shadow.
  This makes cost-aware activation significantly more impactful than when only
  shadow was hybrid.

### Consolidation Data Available (`consolidator.py:124-274, 724-780`)

The consolidation engine already computes per-agent calibration per topic cluster:

- `fello.proposal_adopted_rate` -- How often Fello's proposals are accepted.
- `othello.overcautious_rate` -- How often Othello flags risk that turns out fine.
- `shadow.prediction_accuracy` -- How accurate Shadow's predictions are.
- `prefrontal.step_followed_rate` -- How often Prefrontal steps are followed.
- `confidence` -- Per-rule confidence (0.0-1.0), based on episode count and
  outcome coverage.

This data is sufficient to compute a **value-of-information score** per agent
per topic, but is currently used only to adjust weights, not to suppress agents.

### Execution Pipeline (`cognitive_runtime.py:849-876`)

Worker agents run in parallel via `asyncio.as_completed`. Removing an agent
from `active_workers` is zero-cost -- no API call, no context assembly, no
wait. The pipeline already handles variable worker counts gracefully.

## Industry Standard / Research Findings

### 1. Difficulty-Aware Agentic Orchestration (DAAO)

**Source**: [DAAO, arXiv 2509.11079](https://arxiv.org/abs/2509.11079)

Adapts workflow depth based on query difficulty. Easy queries get fewer
processing stages. Key insight: the number of active agents/operators is
`L = ceil(difficulty * max_layers)`. Achieves **11.21% higher accuracy at 64%
of inference cost**. Uses a Mixture-of-Experts gating mechanism to threshold
per-operator activation scores.

**Applicability to Agent Zero**: Instead of a fixed activation threshold (0.35),
scale the threshold by topic familiarity. Well-consolidated topics (high
confidence, many episodes) should have a *higher* threshold, skipping agents
that historically underperform. Novel topics should keep the low threshold to
explore.

### 2. Value-of-Information (VOI) Agent Gating

**Source**: [Bayesian Orchestration, arXiv 2601.01522](https://arxiv.org/abs/2601.01522)

Computes expected value of information before each agent call:
`VOI(agent) = E[cost_improvement_from_agent_output] - cost_of_calling_agent`.
Only calls an agent when VOI > 0. Achieves **34% cost reduction** across 1000
decisions.

**Applicability to Agent Zero**: For each agent, compute:
- `expected_signal = calibration_metric * rule_confidence`
- `call_cost = 0.01 for deterministic, 1.0 for hybrid`
- `VOI = expected_signal - call_cost_penalty`
Skip when VOI < 0.

### 3. SupervisorAgent Token Reduction

**Source**: [SupervisorAgent, arXiv 2510.26585](https://arxiv.org/abs/2510.26585)

Uses an **LLM-free adaptive filter** to intercept and suppress unnecessary
agent activations. Reduces token consumption by **29.45%** without accuracy
loss. Key insight: the filter is a prioritized conditional chain -- check
cheapest conditions first, only escalate to expensive checks if needed.

**Applicability to Agent Zero**: The consolidation data IS the cheap signal.
Check it first. If a rule with confidence >= 0.6 says an agent's calibration
metric is below a "no-signal" threshold, skip that agent without further
analysis.

### 4. Contextual Bandit Agent Penalty

**Source**: [Multi-Agent Contextual Bandits, ITM 2025](https://www.itm-conferences.org/articles/itmconf/pdf/2025/09/itmconf_cseit2025_01033.pdf)

Uses a penalty parameter gamma (= 0.5) to define minimum acceptable reward.
Agents producing rewards below gamma are penalised, reducing their selection
probability over time. Heterogeneous agent configurations with penalty
mechanisms converge faster.

**Applicability to Agent Zero**: Define a "no-signal threshold" per agent. When
the calibration metric (adoption rate, risk heeded, prediction accuracy, step
followed) falls below 0.25 AND rule confidence >= 0.5, apply a suppression
penalty that forces the weight below the activation threshold.

## Proposed Implementation

### Step 1: Define Per-Agent Cost and Signal Constants

File: `cognitive_runtime.py`, new constants near line 351.

```python
# Per-agent inference cost (all hybrid since 2026-03-18).
# Shadow costs more due to larger context (profile + episodes + rules).
AGENT_INFERENCE_COST = {
    "fello": 1.0,       # hybrid -- full LLM call
    "othello": 1.0,     # hybrid -- full LLM call
    "prefrontal_cortex": 1.0,  # hybrid -- full LLM call
    "shadow": 1.5,      # hybrid -- larger context = higher cost
}

# Minimum calibration signal below which an agent is considered no-signal
# Only applied when rule confidence >= MIN_SUPPRESSION_CONFIDENCE
NO_SIGNAL_THRESHOLD = 0.25
MIN_SUPPRESSION_CONFIDENCE = 0.5
```

### Step 2: Add `_compute_agent_voi()` Function

File: `cognitive_runtime.py`, new function after `_compute_learned_selection`.

```python
def _compute_agent_voi(
    agent_id: str,
    weight: float,
    rules: list[dict],
) -> float:
    """Compute value-of-information score for an agent.

    Returns a score where positive = worth calling, negative = skip.
    Based on calibration signal strength vs inference cost.
    """
    cost = AGENT_INFERENCE_COST.get(agent_id, 0.0)

    if not rules:
        # Cold start: no data to suppress, default positive VOI
        return weight - cost * 0.1

    # Find the highest-confidence rule's calibration for this agent
    best_conf = 0.0
    best_signal = 0.5  # neutral default
    for rule in rules:
        conf = rule.get("confidence", 0)
        if conf <= best_conf:
            continue
        cal = rule.get("agent_calibration", {}).get(agent_id, {})
        signal = _extract_signal_metric(agent_id, cal)
        if signal is not None:
            best_conf = conf
            best_signal = signal

    # Suppression: if signal is below threshold AND confidence is high,
    # penalise this agent heavily
    if best_conf >= MIN_SUPPRESSION_CONFIDENCE and best_signal < NO_SIGNAL_THRESHOLD:
        return -1.0  # force skip

    # VOI = signal strength * confidence - cost penalty
    return (best_signal * best_conf) - (cost * 0.15)


def _extract_signal_metric(agent_id: str, calibration: dict) -> float | None:
    """Extract the primary effectiveness metric for an agent."""
    if agent_id == "fello":
        return calibration.get("proposal_adopted_rate")
    if agent_id == "othello":
        # Invert overcautious: high overcautious = low signal
        overcautious = calibration.get("overcautious_rate", 0)
        heeded = calibration.get("risk_heeded_rate", 0.5)
        return max(0.0, heeded - overcautious)
    if agent_id == "shadow":
        return calibration.get("prediction_accuracy")
    if agent_id == "prefrontal_cortex":
        return calibration.get("step_followed_rate")
    return None
```

### Step 3: Modify `select_agents_for_turn()` to Use VOI

File: `cognitive_runtime.py`, modify `select_agents_for_turn` (lines 329-377).

Replace the simple weight-threshold loop with VOI-aware selection:

```python
def select_agents_for_turn(user_content, route, state=None, shadow_profile=None):
    if route in ("casual_chat", "simple_chat", "self_introspection"):
        return ()

    agent_weights = _compute_learned_selection(user_content, state, shadow_profile)

    # Get relevant rules for VOI computation
    topic_signals = _extract_current_topic_signals(state or {})
    if not topic_signals:
        topic_signals = _extract_topics_from_text((user_content or "").lower())
    rules = get_relevant_rules(shadow_profile or {}, topic_signals) if shadow_profile else []

    # Adaptive threshold: higher for well-consolidated topics
    max_rule_conf = max((r.get("confidence", 0) for r in rules), default=0)
    # Scale threshold from 0.35 (no data) to 0.50 (high confidence data)
    ACTIVATION_THRESHOLD = 0.35 + (max_rule_conf * 0.15)

    candidates = []
    for agent_id in WORKER_AGENT_IDS:
        weight = agent_weights.get(agent_id, 0.5)
        voi = _compute_agent_voi(agent_id, weight, rules)

        # Skip if VOI is negative (agent historically adds no signal)
        if voi < 0:
            continue

        if weight >= ACTIVATION_THRESHOLD:
            candidates.append((weight, agent_id))

    candidates.sort(reverse=True)
    selected = [aid for _, aid in candidates]

    # Relaxed minimum: require at least 1 agent (not 2) when data supports it
    MIN_AGENTS = 1 if max_rule_conf >= 0.6 else 2

    # Still force fello if no workers selected (empathy baseline)
    if not selected:
        selected.append("fello")

    if len(selected) < MIN_AGENTS:
        for aid in WORKER_AGENT_IDS:
            if aid not in selected:
                voi = _compute_agent_voi(aid, agent_weights.get(aid, 0.5), rules)
                if voi >= 0:  # Don't force-add suppressed agents
                    selected.append(aid)
                if len(selected) >= MIN_AGENTS:
                    break

    return tuple(selected)
```

### Step 4: Track Agent Skip Counts in Episode Data

File: `cognitive_runtime.py`, in `run_cognitive_pipeline` (after line 840).

Add skip tracking to the blackboard so the consolidator can monitor skip
patterns and detect if skipping degrades outcomes:

```python
# Track which agents were skipped for cost analysis
all_worker_ids = set(WORKER_AGENT_IDS)
skipped_workers = all_worker_ids - set(active_workers)
blackboard.write_section(
    "meta",
    {"skipped_agents": sorted(skipped_workers)},
    actor="orchestrator",
    merge=True,
)
```

### Step 5: Monitor Skip Impact in Consolidator

File: `consolidator.py`, add to `_compute_agent_calibration` (after line 274).

Track whether turns where an agent was skipped had better or worse outcomes
than turns where it ran, to detect if skipping is hurting quality:

```python
# -- Skip impact tracking --
skip_impact = {}
for agent_id in ("fello", "othello", "prefrontal_cortex", "shadow"):
    ran_eps = [ep for ep in episodes_with_outcomes
               if ep.get("agent_signals", {}).get(agent_id, {}).get("ran")]
    skipped_eps = [ep for ep in episodes_with_outcomes
                   if agent_id in (ep.get("meta", {}).get("skipped_agents", []))]

    ran_followed = sum(1 for ep in ran_eps if ep.get("outcome", {}).get("user_followed_up"))
    skipped_followed = sum(1 for ep in skipped_eps if ep.get("outcome", {}).get("user_followed_up"))

    ran_rate = ran_followed / max(len(ran_eps), 1) if ran_eps else None
    skipped_rate = skipped_followed / max(len(skipped_eps), 1) if skipped_eps else None

    skip_impact[agent_id] = {
        "ran_follow_rate": round(ran_rate, 2) if ran_rate is not None else None,
        "skipped_follow_rate": round(skipped_rate, 2) if skipped_rate is not None else None,
        "skip_safe": (skipped_rate is None or ran_rate is None or
                      skipped_rate >= ran_rate - 0.1) if (ran_rate is not None) else True,
    }
```

Include `skip_impact` in the returned calibration dict. If `skip_safe` is
False for an agent, the VOI function should boost that agent's score to
prevent further skipping (feedback loop).

## Test Specifications

### Test File: `agent_zero/test_cost_aware_activation.py`

```
test_cold_start_all_agents_selected:
    - No shadow_profile, strategic route
    - All 4 workers should be selected (no suppression without data)
    - Passes: len(result) == 4

test_high_overcautious_othello_suppressed:
    - Shadow profile with rule: othello.overcautious_rate=0.7, confidence=0.6
    - Othello should NOT be in selected agents
    - Fello, shadow, prefrontal should still be in

test_low_prediction_accuracy_shadow_suppressed:
    - Rule: shadow.prediction_accuracy=0.15, confidence=0.7
    - Shadow should NOT be selected (hybrid agent, high cost)
    - This is the highest-value skip (saves an LLM call)

test_low_step_followed_prefrontal_suppressed:
    - Rule: prefrontal.step_followed_rate=0.10, confidence=0.5
    - Prefrontal should NOT be selected

test_fello_always_selected_as_fallback:
    - All agents suppressed except fello
    - At least fello should be in result (empathy baseline)

test_adaptive_threshold_rises_with_confidence:
    - Rule confidence=0.0: threshold should be 0.35
    - Rule confidence=1.0: threshold should be 0.50
    - Agent with weight=0.40 selected at conf=0, skipped at conf=1.0

test_min_agents_relaxed_with_high_confidence:
    - Rule confidence=0.7, only fello passes threshold
    - Should accept 1 agent (MIN_AGENTS=1 when conf>=0.6)
    - Without the change, would force-add a second agent

test_voi_positive_for_deterministic_cold_start:
    - No rules, deterministic agent (cost=0)
    - VOI should be positive (weight - 0 > 0)

test_voi_negative_for_suppressed_hybrid:
    - Rule with low signal (0.1) and high confidence (0.8)
    - Hybrid agent (cost=1.0)
    - VOI should be negative

test_skip_tracking_in_blackboard:
    - Run pipeline with 2 of 4 agents selected
    - Blackboard meta.skipped_agents should list the other 2

test_skip_impact_monitoring:
    - Cluster with episodes where agent ran (good outcomes) + skipped (bad outcomes)
    - skip_safe should be False
    - This signals the system to stop skipping that agent

test_casual_chat_still_returns_empty:
    - Route="casual_chat"
    - Should return () regardless of cost-aware logic

test_no_suppression_below_confidence_threshold:
    - Rule with low signal BUT confidence=0.3 (below MIN_SUPPRESSION_CONFIDENCE)
    - Agent should NOT be suppressed (not enough data to be sure)
```

## Estimated Impact

### Latency Reduction

- **All agents are now hybrid** (LLM-backed). Skipping any agent saves ~1-3s
  latency and ~1500-2000 tokens per call.
- On well-understood topics (5+ consolidated episodes), expect 1-2 agents
  skipped per turn on average.
- Estimated **25-50% latency reduction** on repeat topics (1-2 of 4 workers
  skipped, each saving a full vLLM call).

### Token Cost Reduction

- Each hybrid agent call costs ~1500-2000 tokens (context + output).
- Skipping 1 agent saves ~1500-2000 tokens; skipping 2 saves ~3000-4000.
- 4 workers + pineal + speaker = ~9000-12000 tokens per strategic turn.
- Estimated **17-33% token savings** per turn with 1-2 agents suppressed.
- At scale (100 turns/day), this is ~150K-400K tokens saved daily.

### Quality Preservation

- The VOI check ensures agents are only skipped when data shows they add
  no signal. The skip-impact monitor in the consolidator provides a feedback
  loop: if skipping hurts outcomes, the system self-corrects.
- The adaptive threshold ensures novel topics (no consolidation data) still
  get the full agent pipeline.
- Cold-start behavior is identical to current system (all agents run).

### User Experience

- Faster responses on familiar topics (latency savings).
- No degradation on novel topics (full pipeline preserved).
- Reasoning transparency already shows which agents ran; skipped agents
  will simply not appear in the thought bubbles.

## Citations

1. [Difficulty-Aware Agentic Orchestration](https://arxiv.org/abs/2509.11079) -- DAAO, adaptive workflow depth by query difficulty, 11.21% accuracy gain at 64% cost.
2. [Bayesian Orchestration for Cost-Aware Decision-Making](https://arxiv.org/abs/2601.01522) -- VOI-based agent gating, 34% cost reduction.
3. [Stop Wasting Your Tokens: SupervisorAgent](https://arxiv.org/abs/2510.26585) -- LLM-free adaptive filter, 29.45% token reduction.
4. [Multi-Agent Contextual Bandits](https://www.itm-conferences.org/articles/itmconf/pdf/2025/09/itmconf_cseit2025_01033.pdf) -- Penalty parameter gamma for minimum acceptable agent reward.
5. [OPTIMA: Optimizing Effectiveness and Efficiency for LLM-Based MAS](https://arxiv.org/abs/2410.08115) -- ACL 2025, 2.8x performance at <10% tokens via token-penalised reward.
6. [CoRL: Cost-Controlled Multi-Agent via RL](https://arxiv.org/abs/2511.02755) -- Multiplicative reward (perf * cost), budget-regime conditioning.
7. [Budget-Aware Tool-Use Enables Effective Agent Scaling](https://arxiv.org/abs/2511.17006) -- BATS, continuous budget awareness plug-in.
