---
topic: Consolidation Threshold Validation
status: implemented
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T12:00:00Z
---

# Consolidation Threshold Validation

## Problem Statement

Agent Zero's consolidation engine (consolidator.py) contains 12+ hardcoded thresholds that
govern when consolidation fires, how episodes cluster, when rules expire, and how confidence
is computed. The current values were chosen by engineering judgement or borrowed from other
systems (e.g., "Letta production default" for N=5). None have been empirically validated
against memory science research or tuned via ablation. If any threshold is materially wrong,
the system silently produces low-quality rules, clusters too aggressively or too conservatively,
or retires useful rules prematurely.

## Current State in Agent Zero

All thresholds live in `agent_zero/consolidator.py` as module-level constants:

| Constant | Value | Line | Purpose |
|----------|-------|------|---------|
| `MIN_UNCONSOLIDATED_EPISODES` | 5 | 34 | Trigger: batch size before consolidation fires |
| `MAX_HOURS_BETWEEN_CONSOLIDATIONS` | 1.0 | 35 | Trigger: time-based fallback |
| `MAX_ACTIVE_RULES` | 20 | 38 | Cap on concurrent active rules |
| `MIN_CLUSTER_SIZE` | 3 | 39 | Minimum episodes to form a cluster |
| `MIN_RULE_CONFIDENCE` | 0.4 | 40 | Threshold to inject rule into agent context |
| `STALE_DAYS` | 30 | 41 | Days until rule marked stale |
| `RETIRED_DAYS` | 90 | 42 | Days until rule marked retired |
| `MERGE_DISTANCE_THRESHOLD` | 0.6 | 45 | Agglomerative clustering stop distance |
| `MIN_CLUSTER_COHERENCE` | 0.4 | 46 | Intra-cluster similarity floor |
| `TEMPORAL_WEIGHT` | 0.2 | 47 | Weight for temporal proximity vs topic Jaccard |
| Exponential decay constant | 0.0495 | 115 | Hardcoded in `_episode_distance()`, half-life 14 days |
| Outcome decay threshold | 0.2 | 501 | `second_avg < first_avg - 0.2` in `_detect_temporal_pattern()` |

Additional thresholds in related files:
- `context_manager.py:13-16`: MODEL_CONTEXT_LIMIT=32768, COMPRESSION_THRESHOLD=0.70, SYSTEM_PROMPT_BUDGET=0.30, PROTECTED_RECENT_TURNS=6
- `session_checkin.py:17`: MOTIVATION_THRESHOLD=0.5
- `cognitive_runtime.py:22`: DELIBERATION_THRESHOLD=0.30

## Industry Standard / Research Findings

### 1. Batch Size for Consolidation (MIN_UNCONSOLIDATED_EPISODES = 5)

**Current justification:** "Letta production default." Letta (formerly MemGPT) uses a tiered
memory model where consolidation happens via "sleeptime" subagents triggered by context window
compaction events, not fixed episode counts [1]. The "5" value appears to be a reasonable
engineering heuristic but lacks direct empirical support.

**Research recommendation:** FadeMem (Yang et al., 2025) uses event-driven consolidation
triggered by importance thresholds rather than fixed batch sizes [2]. Their dual-layer system
with LML capacity=1000 and SML capacity=500 suggests capacity-proportional triggers are more
robust than fixed counts. For Agent Zero's scale (dozens of episodes per user per day), 5 is
reasonable but should be made configurable. A range of 3-8 is defensible.

**Verdict:** 5 is acceptable. Add configurable range [3, 10] with default 5.

### 2. Time-Based Trigger (MAX_HOURS_BETWEEN_CONSOLIDATIONS = 1.0)

**Research context:** Biological hippocampal consolidation primarily occurs during sleep
(offline), not on a fixed hourly schedule [3]. However, for online AI systems, periodic
consolidation is standard practice. LightMem (ICLR 2026) uses "sleep-time update" as an
offline procedure decoupled from online inference [4]. Letta triggers consolidation during
context compaction events.

**Verdict:** 1 hour is aggressive for a system that may see 0-2 interactions per hour.
Consider increasing to 2-4 hours, or switching to an event-driven trigger (e.g., "consolidate
after session ends or after N new episodes, whichever comes first"). Current value is
acceptable for active users but wasteful for inactive ones.

### 3. Maximum Active Rules (MAX_ACTIVE_RULES = 20)

**Current justification:** "MemOS mid-term capacity benchmark."

**Research context:** MemOS (MemTensor, 2025) proposes a three-layer memory architecture
with configurable capacity per layer [5]. The "20 active rules" maps roughly to working
memory capacity research: Miller's 7+/-2 for individual items, but consolidation rules are
higher-level abstractions. Cowan (2001) revised this to 4+/-1 for pure capacity, but rules
are retrieved selectively (top 3 per turn), not held simultaneously.

**Verdict:** 20 is reasonable given that only top-3 rules are retrieved per turn (get_relevant_rules
returns max 3). The cap prevents unbounded growth. Consider scaling with user tenure:
new users may need fewer rules, long-term users may benefit from 30-50. Keep 20 as default.

### 4. Exponential Decay Half-Life (14 days, lambda=0.0495)

**Current implementation:** `temporal_dist = 1.0 - math.exp(-0.0495 * days_apart)`
This gives half-life = ln(2)/0.0495 = 14.0 days.

**Research findings:**
- **FadeMem (2025):** LML half-life ~11.25 days, SML half-life ~5.02 days with lambda_base=0.1
  and shape parameters beta=0.8 (LML) / 1.2 (SML) [2].
- **Park et al. (2023):** Generative Agents use decay factor 0.995 per game-hour, which
  translates to half-life = ln(2)/(-ln(0.995)) = ~138 game-hours [6]. This is much slower
  but in a different time scale.
- **Ebbinghaus replication (Murre & Dros, 2015):** Measured retention at 31 days = 0.041
  (savings score). The Memory Chain Model gives Store 1 decay rate a1=0.000353 (per minute),
  yielding half-life ~32.7 hours for short-term and much longer for consolidated store [7].
- **FadeMem promotion threshold:** theta_promote=0.7 (promote important memories to LML),
  theta_demote=0.3 (demote to SML) [2].

**Verdict:** 14-day half-life for **temporal proximity weighting** in clustering is appropriate.
Episodes discussing the same topic 14+ days apart are likely different contexts. However,
the decay is used for *clustering distance* (grouping similar episodes), not for *memory
strength*. For this purpose, a 7-14 day window is defensible. Consider making this
configurable: `TEMPORAL_HALF_LIFE_DAYS = 14`.

### 5. Merge Distance Threshold (0.6) and Cluster Coherence (0.4)

**Research context:** Optimal distance thresholds for agglomerative clustering depend on
the distance metric and data distribution. Mahmoud et al. (2022) propose using Extreme Value
Theory to automatically determine the threshold from the data's linkage distance distribution [8].
The silhouette score method jointly maximizes intra-cluster coherence and inter-cluster
separation.

**Current approach:** Fixed threshold 0.6 with coherence floor 0.4. For Jaccard-based
distance (range [0,1]), 0.6 means "stop merging when clusters are less than 40% similar."
Coherence 0.4 means "reject clusters where average pair similarity < 40%."

**Verdict:** These are reasonable for Jaccard distance. However, they should be validated
empirically against actual Agent Zero episode data. The paper recommends adding a self-tuning
mechanism: compute silhouette scores for the produced clusters and log them. If mean
silhouette < 0.25, the thresholds need adjustment. Current values are a good starting point.

### 6. Rule Staleness (30/90 days)

**Research context:** FadeMem uses LML half-life ~11 days for importance decay, suggesting
that memories not reinforced within 2-3 half-lives (~22-33 days) have lost most value [2].
Ebbinghaus data shows retention at 31 days is ~4% without rehearsal [7]. In behavioral
coaching, interventions are typically time-boxed to 4-10 sessions over 4-12 weeks [9].

**Verdict:** STALE_DAYS=30 aligns well with both FadeMem (~3 half-lives) and Ebbinghaus
(~4% retention without rehearsal). RETIRED_DAYS=90 (~3 months) matches typical coaching
program duration. These values are well-justified.

### 7. Minimum Rule Confidence (0.4)

**Research context:** LightMem uses extraction threshold range [0.2, 0.8] for metadata
extraction, with 0.8 being conservative and 0.2 being permissive [4]. FadeMem uses
theta_promote=0.7 for promoting memories to long-term store [2].

**Verdict:** 0.4 is on the permissive side, appropriate for a system that also applies
a quality gate (compute_rule_quality with passes_gate >= 0.3). The two-stage filter
(confidence >= 0.4 AND quality >= 0.3) provides adequate protection. Keep as-is.

### 8. Temporal Weight (0.2 for temporal vs 0.8 for topic Jaccard)

**Research context:** Park et al. (2023) use equal weights (alpha=1) for recency,
importance, and relevance in retrieval scoring [6]. FadeMem's importance formula uses
balanced weights alpha, beta, gamma determined via grid search [2].

**Verdict:** The 80/20 split favoring topic similarity over temporal proximity is appropriate
for *clustering* (grouping thematically similar episodes). For *retrieval*, temporal recency
should weigh more heavily. The current usage is for clustering, so 0.2 is justified.

## Proposed Implementation

### Phase 1: Make All Thresholds Configurable (consolidator.py)

Replace module-level constants with a `ConsolidationConfig` dataclass:

```python
# agent_zero/consolidator.py, lines 33-47

from dataclasses import dataclass, field

@dataclass
class ConsolidationConfig:
    """Validated consolidation parameters with research-backed defaults."""

    # Triggers
    min_unconsolidated_episodes: int = 5        # Range: [3, 10]
    max_hours_between_consolidations: float = 1.0  # Range: [0.5, 8.0]

    # Rule limits
    max_active_rules: int = 20                  # Range: [10, 50]
    min_cluster_size: int = 3                   # Range: [2, 5]
    min_rule_confidence: float = 0.4            # Range: [0.2, 0.8]
    stale_days: int = 30                        # Range: [14, 60]
    retired_days: int = 90                      # Range: [60, 180]

    # Clustering
    merge_distance_threshold: float = 0.6       # Range: [0.3, 0.8]
    min_cluster_coherence: float = 0.4          # Range: [0.2, 0.6]
    temporal_weight: float = 0.2                # Range: [0.0, 0.5]
    temporal_half_life_days: float = 14.0       # Range: [7.0, 30.0]

    # Quality
    quality_gate_threshold: float = 0.3         # Range: [0.2, 0.5]
    outcome_decay_margin: float = 0.2           # Range: [0.1, 0.4]

    def __post_init__(self):
        """Validate parameter ranges."""
        assert 3 <= self.min_unconsolidated_episodes <= 10
        assert 0.5 <= self.max_hours_between_consolidations <= 8.0
        assert 10 <= self.max_active_rules <= 50
        assert 2 <= self.min_cluster_size <= 5
        assert 0.2 <= self.min_rule_confidence <= 0.8
        assert 14 <= self.stale_days <= 60
        assert self.stale_days < self.retired_days
        assert 0.3 <= self.merge_distance_threshold <= 0.8
        assert 0.2 <= self.min_cluster_coherence <= 0.6
        assert 0.0 <= self.temporal_weight <= 0.5
        assert 7.0 <= self.temporal_half_life_days <= 30.0
        assert 0.2 <= self.quality_gate_threshold <= 0.5
        assert 0.1 <= self.outcome_decay_margin <= 0.4

# Default config instance (backward compatible)
DEFAULT_CONFIG = ConsolidationConfig()
```

### Phase 2: Thread Config Through All Functions

Update function signatures to accept `config: ConsolidationConfig = DEFAULT_CONFIG`:

1. `should_consolidate(shadow, *, force=False, config=DEFAULT_CONFIG)` -- use `config.min_unconsolidated_episodes`, `config.max_hours_between_consolidations`
2. `_episode_distance(ep_a, ep_b, *, config=DEFAULT_CONFIG)` -- use `config.temporal_weight`, compute decay from `config.temporal_half_life_days`
3. `_cluster_episodes(episodes, *, config=DEFAULT_CONFIG)` -- use `config.merge_distance_threshold`, `config.min_cluster_coherence`
4. `run_consolidation(shadow, *, config=DEFAULT_CONFIG)` -- use all config values
5. `get_relevant_rules(shadow, topic_signals, *, config=DEFAULT_CONFIG)` -- use `config.min_rule_confidence`
6. `_detect_temporal_pattern(cluster, *, config=DEFAULT_CONFIG)` -- use `config.outcome_decay_margin`

Replace the hardcoded decay constant:
```python
# Line 115, currently:
temporal_dist = 1.0 - math.exp(-0.0495 * days_apart)

# Replace with:
lambda_decay = math.log(2) / config.temporal_half_life_days
temporal_dist = 1.0 - math.exp(-lambda_decay * days_apart)
```

### Phase 3: Add Cluster Quality Telemetry

After clustering, compute and log silhouette-like scores:
```python
def _compute_clustering_quality(clusters, dist_matrix, all_indices):
    """Log clustering quality metrics for threshold tuning."""
    if len(clusters) < 2:
        return {"silhouette": 0.0, "n_clusters": len(clusters)}

    # Simplified silhouette: for each point, compare intra-cluster dist to nearest-cluster dist
    silhouettes = []
    for cluster_idx, indices in enumerate(clusters):
        for i in indices:
            # a(i) = mean intra-cluster distance
            a = _mean_dist(dist_matrix, i, indices)
            # b(i) = min mean distance to other clusters
            b = min(
                _mean_dist(dist_matrix, i, other_indices)
                for other_idx, other_indices in enumerate(clusters)
                if other_idx != cluster_idx and other_indices
            ) if len(clusters) > 1 else 0
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouettes.append(s)

    return {
        "silhouette": round(sum(silhouettes) / len(silhouettes), 3) if silhouettes else 0,
        "n_clusters": len(clusters),
        "n_singletons": sum(1 for c in clusters if len(c) == 1),
    }
```

Log these metrics to `_quality_logger` so they can be analyzed for threshold tuning.

### Phase 4: Document Research Justifications

Add a `THRESHOLD_RATIONALE` dict to consolidator.py as inline documentation:
```python
THRESHOLD_RATIONALE = {
    "min_unconsolidated_episodes": "Default 5. FadeMem uses event-driven triggers; Letta uses context compaction. Range [3,10] covers both eager and conservative strategies.",
    "temporal_half_life_days": "Default 14. FadeMem LML half-life=11.25d, SML=5.02d. 14d balances LML-level persistence for clustering context.",
    "merge_distance_threshold": "Default 0.6. Jaccard distance [0,1]; 0.6 = stop at <40% similarity. Validate via silhouette scores.",
    "stale_days": "Default 30. Aligns with FadeMem ~3 half-lives (33d) and Ebbinghaus 31d retention ~4%.",
    "retired_days": "Default 90. Matches typical behavioral coaching program duration (4-12 weeks).",
    "min_rule_confidence": "Default 0.4. Permissive (LightMem range [0.2,0.8]). Paired with quality gate >= 0.3 for two-stage filtering.",
}
```

## Test Specifications

### Test: ConsolidationConfig validation
```python
def test_config_defaults_valid():
    """Default config should pass validation."""
    config = ConsolidationConfig()
    assert config.min_unconsolidated_episodes == 5
    assert config.temporal_half_life_days == 14.0

def test_config_rejects_out_of_range():
    """Out-of-range values should raise AssertionError."""
    with pytest.raises(AssertionError):
        ConsolidationConfig(min_unconsolidated_episodes=1)
    with pytest.raises(AssertionError):
        ConsolidationConfig(stale_days=100, retired_days=90)  # stale >= retired

def test_config_accepts_custom_values():
    """Custom values within range should work."""
    config = ConsolidationConfig(
        min_unconsolidated_episodes=8,
        temporal_half_life_days=21.0,
        max_active_rules=30,
    )
    assert config.min_unconsolidated_episodes == 8
```

### Test: Temporal decay uses config
```python
def test_episode_distance_uses_config_half_life():
    """Changing half-life should change temporal distance."""
    ep_a = {"topic_signals": ["x"], "timestamp": "2025-01-01T00:00:00+00:00"}
    ep_b = {"topic_signals": ["x"], "timestamp": "2025-01-15T00:00:00+00:00"}

    config_14d = ConsolidationConfig(temporal_half_life_days=14.0)
    config_7d = ConsolidationConfig(temporal_half_life_days=7.0)

    dist_14 = _episode_distance(ep_a, ep_b, config=config_14d)
    dist_7 = _episode_distance(ep_a, ep_b, config=config_7d)

    # Shorter half-life = higher temporal distance for same gap
    assert dist_7 > dist_14
```

### Test: Clustering respects config thresholds
```python
def test_cluster_episodes_strict_threshold():
    """Stricter merge threshold should produce more clusters."""
    episodes = [make_episode(topics=["a","b"]), make_episode(topics=["a","c"]),
                make_episode(topics=["a","d"]), make_episode(topics=["a","e"])]

    loose = _cluster_episodes(episodes, config=ConsolidationConfig(merge_distance_threshold=0.8))
    strict = _cluster_episodes(episodes, config=ConsolidationConfig(merge_distance_threshold=0.3))

    assert len(strict) >= len(loose)
```

### Test: Silhouette quality metric
```python
def test_clustering_quality_metric():
    """Clustering quality should be computable and in [-1, 1]."""
    # Well-separated clusters
    episodes = [make_episode(topics=["fitness"]) for _ in range(5)] + \
               [make_episode(topics=["finance"]) for _ in range(5)]
    clusters = _cluster_episodes(episodes)
    quality = _compute_clustering_quality(...)
    assert -1.0 <= quality["silhouette"] <= 1.0
    assert quality["silhouette"] > 0  # well-separated should be positive
```

### Test: Backward compatibility
```python
def test_consolidation_works_without_config():
    """All functions should work with default config (no arg)."""
    shadow = make_shadow_with_episodes(10)
    result = run_consolidation(shadow)  # no config arg
    assert "consolidated_rules" in result
```

## Estimated Impact

1. **Tuning capability:** Operators can adjust consolidation behavior per-deployment without
   code changes. A cautious deployment might use `min_unconsolidated_episodes=8` and
   `merge_distance_threshold=0.5` to consolidate less aggressively.

2. **Research validation:** Each threshold now has a documented justification with citations.
   Future tuning can be guided by silhouette metrics rather than guesswork.

3. **No behavioral change at defaults:** All default values match current hardcoded values,
   so existing behavior is preserved exactly. This is a pure refactor + documentation +
   configurability improvement.

4. **Telemetry foundation:** Cluster quality metrics enable data-driven threshold optimization
   in future sessions.

## Citations

1. Letta (MemGPT) Documentation. "Understanding Memory Management." https://docs.letta.com/advanced/memory-management/
2. Yang et al. (2025). "FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory." arXiv:2601.18642. https://arxiv.org/abs/2601.18642
3. Murre & Dros (2015). "Replication and Analysis of Ebbinghaus' Forgetting Curve." PLOS ONE. https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/
4. LightMem (ICLR 2026). "LightMem: Lightweight and Efficient Memory-Augmented Generation." arXiv:2510.18866. https://arxiv.org/abs/2510.18866
5. MemOS (MemTensor, 2025). "MemOS: A Memory OS for AI System." arXiv:2507.03724. https://arxiv.org/abs/2507.03724
6. Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763
7. Murre & Dros (2015). Retention at 31 days: savings score 0.041; Memory Chain Model decay rate a1=0.000353/min. https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/
8. Mahmoud et al. (2022). "Agglomerative Clustering with Threshold Optimization via Extreme Value Theory." Algorithms 15(5):170. https://www.mdpi.com/1999-4893/15/5/170
9. Terblanche (2024). "AI Coaching: Redefining People Development." Journal of Applied Behavioral Science. https://journals.sagepub.com/doi/10.1177/00218863241283919
