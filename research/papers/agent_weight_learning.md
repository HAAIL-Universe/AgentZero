---
topic: Agent Weight Learning via Multi-Armed Bandits
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T15:00:00Z
---

# Agent Weight Learning via Multi-Armed Bandits

## Problem Statement

Agent Zero's cognitive runtime uses hardcoded agent weights to decide which of the 7 agents to activate per turn. These weights are static keyword-matching rules that never learn from outcomes. When Fello gives advice that the user acts on, the system doesn't increase Fello's weight for that topic. When Othello flags a risk that the user ignores, the system doesn't decrease Othello's activation. The agent orchestration is permanently frozen at the developer's initial guesses, even as consolidation data accumulates evidence about which agents actually help.

## Current State in Agent Zero

### cognitive_runtime.py (lines 570-591) -- `compute_agent_weights()`
```python
weights = {aid: 0.5 for aid in WORKER_AGENT_IDS}

# Memory signals: boost fello, reduce prefrontal
memory_signals = ("remember", "recall", ...)
if any(sig in lower for sig in memory_signals):
    weights["fello"] = 0.8
    weights["prefrontal_cortex"] = 0.25
    if _has_emotional_signals(lower, state):
        weights["othello"] = 0.7
    else:
        weights["othello"] = 0.25
```

Problems:
- All agents default to 0.5 -- no differentiation by topic or user history
- Keyword matching is brittle: "remember" triggers Fello boost regardless of whether Fello has been useful
- Weights are symmetric (same for all users) despite different users responding differently to different agents
- No feedback loop: outcomes (acted/ignored/pushed_back) never update weights
- No exploration: once a weight pattern is set, the system never tries alternatives

### cognitive_runtime.py (lines 504-541) -- Cost-aware activation
```python
if max_rule_conf >= 0.5:
    activation_threshold = 0.35 + (max_rule_conf * 0.15)
```
This uses consolidation rule confidence to raise the activation threshold, effectively suppressing agents. But it doesn't learn which specific agents to suppress -- it raises the bar uniformly.

### cognitive_runtime.py (lines 654-693) -- Disagreement detection
The disagreement detector uses hardcoded thresholds (0.3, 0.15, 0.25) for confidence deltas, directional disagreement, and shadow tension. These never adapt to actual disagreement resolution outcomes.

### WORKER_AGENT_IDS
The 7 agents: fello, othello, prefrontal_cortex, shadow, pineal, forecaster, curiosity.
Currently all treated as equal candidates with keyword-based weight adjustments.

## Industry Standard / Research Findings

### 1. Multi-Armed Bandit Tutorial for LLMs (KDD 2024)
IBM Research tutorial covering foundational MAB concepts for LLM systems. Key insight: treat text generation options (here, agent selections) as arms in a bandit problem. Covers epsilon-greedy, UCB, and Thompson Sampling. Recommends Thompson Sampling for LLM applications because it naturally balances exploration/exploitation without tuning epsilon.

Reference: IBM Research. (2024). "A Tutorial on Multi-Armed Bandit Applications for Large Language Models." KDD 2024. https://dl.acm.org/doi/abs/10.1145/3637528.3671440

### 2. Thompson Sampling for AI Model Selection (SourcePilot, 2025)
Production implementation of Thompson Sampling for selecting between AI models. Each model maintains Beta(alpha, beta) parameters tracking successes/failures. On each request, sample from each model's Beta distribution and pick the highest sample. Key implementation details:
- Start with Beta(1,1) uniform priors
- Update alpha on success, beta on failure
- "Probability matching": chance of selecting equals estimated probability of being optimal
- Wide distributions (uncertain models) occasionally sample high, ensuring exploration

Reference: SourcePilot. (2025). "How Thompson Sampling Works: The Algorithm Behind SourcePilot's Smart AI Selection." https://sourcepilot.co/blog/2025/11/22/how-thompson-sampling-works

### 3. LLM-Enhanced Multi-Armed Bandits (arXiv, 2025)
Shows that direct arm selection by LLMs is suboptimal. Instead, uses classical MAB (Thompson Sampling) as the high-level framework with LLM for reward prediction. The hybrid approach outperforms both pure LLM selection and pure classical MAB.

Reference: Li, Y. et al. (2025). "Large Language Model-Enhanced Multi-Armed Bandits." arXiv:2502.01118. https://arxiv.org/abs/2502.01118

### 4. Contextual Bandits for Dynamic Environments (MDPI Electronics, 2023)
LLM-informed multi-armed bandit strategies for non-stationary environments. Uses context features (user state, topic, emotional register) to condition arm selection. Key finding: contextual bandits outperform context-free bandits in environments where optimal actions change based on state -- directly applicable to agent selection conditioned on turn context.

Reference: "LLM-Informed Multi-Armed Bandit Strategies for Non-Stationary Environments." MDPI Electronics. https://www.mdpi.com/2079-9292/12/13/2814

### 5. Thompson Sampling for Dynamic Multi-Armed Bandits (IEEE, 2012)
Foundational work showing Thompson Sampling naturally adapts to non-stationary reward distributions through its probabilistic sampling mechanism. In behavioral coaching, user responsiveness to different agents changes over time (stage of change shifts, trust builds). Thompson Sampling with a sliding window or exponential discounting handles this drift.

Reference: Gupta, N., Granmo, O-C., Agrawala, A. (2012). "Thompson Sampling for Dynamic Multi-armed Bandits." IEEE ICIIP. https://ieeexplore.ieee.org/document/6147024/

### 6. Scalable Contextual Bandits (Springer, 2025)
Literature review on contextual bandits with practical retail applications. Recommends LinUCB or Thompson Sampling with context features for personalized decision-making at scale. Demonstrates that per-user bandit models outperform global models when enough per-user data is available.

Reference: "Scalable and Interpretable Contextual Bandits: A Literature Review and Retail Offer Prototype." Springer. https://link.springer.com/chapter/10.1007/978-3-032-03769-5_39

## Proposed Implementation

### New file: `agent_zero/agent_bandit.py`

```python
"""Thompson Sampling multi-armed bandit for agent weight learning.

Each (user, topic, agent) triple maintains a Beta(alpha, beta) distribution.
On each turn, sample from each agent's distribution to produce learned weights.
After outcome resolution, update the relevant agent's alpha (success) or beta (failure).
"""

import json
import math
import random
from typing import Optional

from database import execute, fetch_all, fetch_one


# Default prior: weakly informative Beta(2, 2)
# Effective sample size = 4, equivalent to 2 successes + 2 failures
# Centers at 0.5 but with less variance than Beta(1,1)
DEFAULT_ALPHA = 2.0
DEFAULT_BETA = 2.0

# Decay factor for non-stationary environments
# Multiply alpha/beta by this factor periodically to "forget" old data
DECAY_FACTOR = 0.95
DECAY_INTERVAL_HOURS = 168  # weekly


async def get_agent_priors(user_id: str, topic: str) -> dict[str, tuple[float, float]]:
    """Fetch Beta distribution parameters for all agents for this user+topic.

    Returns: {"fello": (alpha, beta), "othello": (alpha, beta), ...}
    """
    rows = await fetch_all(
        """SELECT agent_id, alpha, beta
           FROM agent_bandit_params
           WHERE user_id = $1::uuid AND topic = $2""",
        user_id, topic,
    )
    params = {}
    for row in rows:
        params[row["agent_id"]] = (float(row["alpha"]), float(row["beta"]))
    return params


def sample_weights(
    agent_ids: list[str],
    priors: dict[str, tuple[float, float]],
    *,
    exploration_bonus: float = 0.0,
) -> dict[str, float]:
    """Thompson Sampling: sample from each agent's Beta distribution.

    Returns dict of agent_id -> sampled weight (0.0 to 1.0).
    """
    weights = {}
    for aid in agent_ids:
        alpha, beta = priors.get(aid, (DEFAULT_ALPHA, DEFAULT_BETA))
        # Sample from Beta(alpha, beta)
        sample = random.betavariate(alpha, beta)
        # Optional exploration bonus for under-sampled agents
        n_eff = alpha + beta
        if exploration_bonus > 0 and n_eff < 10:
            sample = min(1.0, sample + exploration_bonus * (10 - n_eff) / 10)
        weights[aid] = round(sample, 3)
    return weights


async def update_agent_outcome(
    user_id: str,
    topic: str,
    agent_id: str,
    success: bool,
) -> None:
    """Update Beta parameters after observing outcome.

    success=True increments alpha (acted on agent's advice).
    success=False increments beta (ignored/pushed back).
    """
    await execute(
        """INSERT INTO agent_bandit_params (user_id, topic, agent_id, alpha, beta)
           VALUES ($1::uuid, $2, $3, $4, $5)
           ON CONFLICT (user_id, topic, agent_id)
           DO UPDATE SET
               alpha = agent_bandit_params.alpha + $4 - 2.0,
               beta = agent_bandit_params.beta + $5 - 2.0,
               updated_at = now()""",
        user_id, topic, agent_id,
        DEFAULT_ALPHA + (1.0 if success else 0.0),
        DEFAULT_BETA + (0.0 if success else 1.0),
    )
```

### Database schema addition

```sql
CREATE TABLE IF NOT EXISTS agent_bandit_params (
    user_id UUID NOT NULL,
    topic VARCHAR(64) NOT NULL,
    agent_id VARCHAR(64) NOT NULL,
    alpha DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    beta DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (user_id, topic, agent_id)
);
```

### Modify: `agent_zero/cognitive_runtime.py`

**In `compute_agent_weights()` (lines 570-591):**

Blend hardcoded rules with learned bandit weights:

```python
async def compute_agent_weights(user_content, state, *, user_id=None, topic=None):
    # 1. Start with existing keyword-based weights (backward compat)
    base_weights = _keyword_weights(user_content, state)

    # 2. If user+topic available, blend with learned weights
    if user_id and topic:
        from agent_bandit import get_agent_priors, sample_weights
        priors = await get_agent_priors(user_id, topic)
        learned = sample_weights(WORKER_AGENT_IDS, priors)

        # Blend: 60% learned + 40% keyword-based (gradually shift to learned)
        blend_factor = min(0.6, sum(a + b for a, b in priors.values()) / 100)
        for aid in WORKER_AGENT_IDS:
            base_weights[aid] = (
                blend_factor * learned[aid] +
                (1 - blend_factor) * base_weights[aid]
            )

    return base_weights
```

**In consolidation outcome resolution (episode_store.py or intervention_tracker.py):**

After an episode outcome is resolved, update the bandit for agents that participated:

```python
# After outcome resolution:
if outcome == "acted":
    for agent_id in participating_agents:
        await update_agent_outcome(user_id, topic, agent_id, success=True)
elif outcome in ("ignored", "pushed_back"):
    for agent_id in participating_agents:
        await update_agent_outcome(user_id, topic, agent_id, success=False)
```

### Integration with existing cost-aware activation

The bandit weights feed naturally into the existing activation threshold system:
- Low bandit weight (agent historically unhelpful) -> below activation threshold -> skipped
- High bandit weight (agent historically helpful) -> above threshold -> activated
- The cost-aware activation in lines 504-541 already skips agents below threshold; bandit weights just make the weights adaptive instead of static.

## Test Specifications

### test_agent_bandit.py

```
test_sample_weights_uniform_prior -- Beta(2,2) prior produces weights near 0.5
test_sample_weights_strong_success -- Beta(20,2) produces weight near 0.9
test_sample_weights_strong_failure -- Beta(2,20) produces weight near 0.1
test_sample_weights_exploration -- under-sampled agents get exploration bonus
test_sample_weights_all_agents -- returns weight for every agent_id
test_sample_weights_stochastic -- repeated calls produce different values (exploration)
test_sample_weights_bounded -- all weights in [0.0, 1.0]

test_update_outcome_success -- alpha incremented, beta unchanged
test_update_outcome_failure -- beta incremented, alpha unchanged
test_update_outcome_upsert -- creates row if not exists, updates if exists

test_blend_factor_cold_start -- no learned data: blend_factor near 0, keyword weights dominate
test_blend_factor_warm -- with 50+ observations: blend_factor approaches 0.6
test_blend_preserves_keyword_signals -- memory/emotional keywords still influence weights
test_blend_backward_compat -- without user_id, returns pure keyword weights

test_bandit_learns_from_acted -- after 10 "acted" outcomes, agent weight increases
test_bandit_learns_from_ignored -- after 10 "ignored" outcomes, agent weight decreases
test_bandit_per_user -- different users have independent parameters
test_bandit_per_topic -- different topics have independent parameters
test_bandit_exploration_decreases -- exploration bonus shrinks as n_eff grows

test_schema_creation -- agent_bandit_params table created with correct columns
test_schema_primary_key -- (user_id, topic, agent_id) is unique constraint
```

## Estimated Impact

- **Personalization**: Each user develops a unique agent activation profile based on which agents' advice they actually follow. Users who respond well to Fello's memory-based coaching see more Fello; users who prefer Othello's emotional awareness see more Othello.
- **Topic specialization**: The system learns that Forecaster is valuable for career decisions but unhelpful for daily habits, activating it selectively.
- **Efficiency**: Agents with low historical effectiveness are suppressed, reducing unnecessary LLM calls and token usage. Estimated 15-30% reduction in agent activations after 20+ turns per topic.
- **Exploration**: Thompson Sampling's natural exploration prevents premature convergence. New topics start with uniform priors, giving all agents equal chance until evidence accumulates.
- **Graceful degradation**: The blend factor starts near 0 (keyword weights dominate) and gradually shifts to 0.6 as data accumulates. No behavior change for cold-start users.
- **Non-stationarity**: Weekly decay (DECAY_FACTOR=0.95) ensures old data gradually loses influence as user preferences evolve.
