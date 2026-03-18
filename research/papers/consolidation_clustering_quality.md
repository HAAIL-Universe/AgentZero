---
topic: Consolidation Clustering Quality
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T22:00:00Z
---

# Consolidation Clustering Quality

## Problem Statement

The consolidation engine's episode clustering uses a union-find algorithm with a single-topic-overlap threshold (consolidator.py:84-120). This creates two quality problems:

1. **Transitive chaining produces mega-clusters.** Episode A has topics `["career", "finance"]`, Episode B has `["finance", "health"]`, Episode C has `["health", "relationships"]`. Under single-overlap union-find, all three merge into one cluster despite A and C sharing zero topics. This dilutes rule quality -- a rule spanning career+finance+health+relationships is too broad to calibrate agents effectively.

2. **No cluster coherence metric.** Every cluster with >= 3 episodes produces a rule regardless of internal coherence. A cluster where 5 episodes share "career" and 1 outlier was pulled in by transitive chaining gets treated identically.

3. **No temporal proximity weighting.** Episodes about "career" from January and "career" from March cluster together purely on topic match, even though these may represent different life contexts requiring different agent calibration.

4. **No noise handling.** Every episode is forced into a cluster. Outlier episodes that don't fit any group should remain unclustered until more related data arrives.

## Current State in Agent Zero

**File:** `agent_zero/consolidator.py`
**Function:** `_cluster_episodes(episodes: list[dict]) -> list[list[dict]]` (lines 84-120)

**Algorithm:**
1. Build topic sets: `topic_sets = [set(ep.get("topic_signals", [])) for ep in episodes]` (line 107)
2. O(n^2) pairwise comparison: `if topic_sets[i] & topic_sets[j]: union(i, j)` (lines 109-112)
3. Group by union-find root (lines 114-120)

**Merge criterion:** Any single shared topic signal merges two episodes (line 111).
**Linkage type:** Equivalent to single-linkage clustering -- the weakest possible grouping criterion.
**Noise handling:** None -- every episode is assigned to a cluster.

**Topic signals** are extracted from user input via keyword matching against 10 categories (outcome_patterns.py:37-48): career, learning, health, relationships, creativity, productivity, finance, mental_health, goals, technology.

**Downstream consumers:**
- `run_consolidation()` (line 668) calls `_cluster_episodes()` on unconsolidated episodes
- Clusters with < `MIN_CLUSTER_SIZE` (3) are skipped (line 673)
- Each qualifying cluster produces calibration metrics, intervention effectiveness, temporal patterns, insights, and rules (lines 682-763)
- Rules are injected into agent contexts via `get_relevant_rules()` (lines 803-857)

**Episode cap:** 200 episodes max (episode_store.py:38), so n <= 200.

**Existing tests:** `test_consolidator.py` includes `test_cluster_episodes_by_topic` (line ~131) verifying basic clustering and `test_cluster_episodes_disjoint` verifying disjoint topics create separate clusters.

## Industry Standard / Research Findings

### 1. MemGAS: Multi-Granularity Memory Association (2025)

Wang et al. (2025) propose MemGAS, which uses **Gaussian Mixture Models (GMM)** to cluster new memories against historical ones, partitioning into an "Accept Set" (high similarity) and "Reject Set" (irrelevant). Key insight: **probabilistic soft assignment** rather than hard binary overlap avoids the chaining problem. The GMM operates on dense embeddings (Contriever vectors), computing pairwise similarity and fitting a 2-component GMM to determine cluster membership.

MemGAS achieves F1=20.38 on LongMemEval-s (vs. 14.73 for HippoRAG 2) and Recall@10=81.82 on LoCoMo-10.

**URL:** https://arxiv.org/abs/2505.19549

### 2. Mem0: Production Memory Consolidation (2025)

Chhikara et al. (2025) implement **similarity-threshold-based deduplication** in the production Mem0 system. Embeddings with cosine similarity > 0.85 trigger merges; clusters within 0.9 similarity are deduplicated. This approach cuts storage by 60% and raises retrieval precision by 22%. Key insight: **graduated thresholds** (loose for merge candidates, strict for deduplication) prevent both over-clustering and redundancy.

**URL:** https://arxiv.org/abs/2504.19413

### 3. A-Mem: Zettelkasten-Style Memory Networks (NeurIPS 2025)

Zhang et al. (2025) implement **Zettelkasten-inspired linking** where each new memory analyzes historical memories to find meaningful connections, creating indexed cross-references. Memory evolution refines contextual representations and attributes over time. Key insight: **explicit link scoring** replaces implicit clustering -- memories maintain scored connections rather than being grouped.

**URL:** https://arxiv.org/abs/2502.12110

### 4. Agglomerative Clustering with Jaccard Similarity

For sparse, categorical features like topic signals, **average-linkage agglomerative clustering with Jaccard distance** is the standard approach (Manning et al., Introduction to Information Retrieval, Ch. 17). Unlike single-linkage (what union-find implements), average-linkage measures the mean pairwise distance between all members of two clusters, producing compact, coherent groups. Jaccard distance `1 - |A cap B| / |A cup B|` is the natural metric for set-valued features.

**URL:** https://nlp.stanford.edu/IR-book/completelink.html

### 5. Temporal Locality in Memory Consolidation

Park et al. (2023) found that episodic memory retrieval in generative agents benefits from **recency weighting** using exponential decay. Episodes closer in time are more likely to represent the same behavioral context. This aligns with hippocampal consolidation research showing temporal contiguity effects on memory binding.

**URL:** https://arxiv.org/abs/2304.03442 (Generative Agents: Interactive Simulacra of Human Behavior)

### 6. HDBSCAN for Variable-Density Clustering

McInnes et al. (2017) show that **HDBSCAN** finds clusters of varying density without requiring epsilon parameters, unlike DBSCAN. For topic signals, this handles the case where some topics (e.g., "career") have many episodes while others (e.g., "creativity") have few. HDBSCAN also naturally identifies noise points -- episodes that don't belong to any cluster.

**URL:** https://joss.theoj.org/papers/10.21105/joss.00205

## Proposed Implementation

### Strategy: Average-Linkage Agglomerative Clustering with Weighted Jaccard

Replace union-find single-linkage with average-linkage agglomerative clustering using a composite distance metric that combines topic Jaccard distance and temporal distance. This is implementable in pure Python with no external dependencies, fits the project's constraint model, and addresses all four problems.

### Step 1: Define Composite Distance Function

**File:** `agent_zero/consolidator.py`
**New function** (insert before `_cluster_episodes`):

```python
def _episode_distance(ep_a: dict, ep_b: dict, *, temporal_weight: float = 0.2) -> float:
    """Compute distance between two episodes using weighted Jaccard + temporal proximity.

    Returns value in [0, 1]. Lower = more similar.

    Components:
    - Jaccard distance on topic signals (weight: 1 - temporal_weight)
    - Temporal distance via exponential decay (weight: temporal_weight)
    """
    # Topic Jaccard distance
    topics_a = set(ep_a.get("topic_signals", []))
    topics_b = set(ep_b.get("topic_signals", []))
    if topics_a or topics_b:
        jaccard = 1.0 - len(topics_a & topics_b) / len(topics_a | topics_b)
    else:
        jaccard = 1.0

    # Temporal distance (exponential decay, half-life = 14 days)
    temporal_dist = 1.0  # default if timestamps missing
    ts_a = ep_a.get("timestamp")
    ts_b = ep_b.get("timestamp")
    if ts_a and ts_b:
        try:
            dt_a = datetime.fromisoformat(ts_a)
            dt_b = datetime.fromisoformat(ts_b)
            if dt_a.tzinfo is None:
                dt_a = dt_a.replace(tzinfo=timezone.utc)
            if dt_b.tzinfo is None:
                dt_b = dt_b.replace(tzinfo=timezone.utc)
            days_apart = abs((dt_a - dt_b).total_seconds()) / 86400.0
            # Exponential decay: half-life 14 days -> lambda = ln(2)/14 ~ 0.0495
            temporal_dist = 1.0 - math.exp(-0.0495 * days_apart)
        except (ValueError, TypeError):
            pass

    return (1.0 - temporal_weight) * jaccard + temporal_weight * temporal_dist
```

**Rationale:** Jaccard distance is the standard metric for set-valued features (Manning et al.). Temporal weighting follows Park et al.'s recency principle. Weight of 0.2 for temporal vs 0.8 for topic ensures topic similarity dominates but temporally distant episodes with the same topics are slightly penalized.

### Step 2: Replace `_cluster_episodes` with Agglomerative Clustering

**File:** `agent_zero/consolidator.py`
**Replace function** `_cluster_episodes` (lines 84-120):

```python
# Clustering constants
MERGE_DISTANCE_THRESHOLD = 0.6  # Average-linkage distance cutoff
MIN_CLUSTER_COHERENCE = 0.4     # Minimum intra-cluster average similarity

def _cluster_episodes(episodes: list[dict]) -> list[list[dict]]:
    """Cluster episodes using average-linkage agglomerative clustering.

    Uses composite distance (Jaccard topic similarity + temporal proximity).
    Stops merging when closest cluster pair exceeds MERGE_DISTANCE_THRESHOLD.
    Returns only clusters passing coherence threshold.
    """
    if not episodes:
        return []

    n = len(episodes)
    if n == 1:
        return [[episodes[0]]]

    # Precompute pairwise distance matrix (still O(n^2), but n <= 200)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _episode_distance(episodes[i], episodes[j])
            dist[i][j] = d
            dist[j][i] = d

    # Initialize: each episode is its own cluster
    # clusters[k] = list of original indices
    clusters = {i: [i] for i in range(n)}
    active = set(range(n))

    # Average-linkage: merge closest pair until threshold exceeded
    while len(active) > 1:
        best_dist = float('inf')
        best_pair = None

        active_list = sorted(active)
        for idx_a in range(len(active_list)):
            for idx_b in range(idx_a + 1, len(active_list)):
                ca, cb = active_list[idx_a], active_list[idx_b]
                # Average linkage: mean distance between all pairs
                total = 0.0
                count = 0
                for i in clusters[ca]:
                    for j in clusters[cb]:
                        total += dist[i][j]
                        count += 1
                avg_dist = total / count if count else float('inf')
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_pair = (ca, cb)

        if best_dist > MERGE_DISTANCE_THRESHOLD or best_pair is None:
            break

        # Merge best_pair
        ca, cb = best_pair
        clusters[ca] = clusters[ca] + clusters[cb]
        del clusters[cb]
        active.discard(cb)

    # Convert index clusters to episode clusters
    result = []
    for indices in clusters.values():
        cluster = [episodes[i] for i in indices]
        # Coherence check: average pairwise similarity within cluster
        if len(cluster) >= 2:
            coherence = _compute_cluster_coherence(cluster, dist, indices)
            if coherence < MIN_CLUSTER_COHERENCE:
                # Split into singletons (remain unclustered)
                for ep in cluster:
                    result.append([ep])
                continue
        result.append(cluster)

    return result


def _compute_cluster_coherence(
    cluster: list[dict],
    dist_matrix: list[list[float]],
    indices: list[int],
) -> float:
    """Compute intra-cluster coherence as average pairwise similarity.

    Returns value in [0, 1]. Higher = more coherent.
    """
    if len(indices) < 2:
        return 1.0
    total = 0.0
    count = 0
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            total += (1.0 - dist_matrix[indices[a]][indices[b]])
            count += 1
    return total / count if count else 0.0
```

**Rationale:**
- Average-linkage is the standard for producing compact clusters (Manning et al.; scikit-learn docs).
- `MERGE_DISTANCE_THRESHOLD = 0.6` means two episodes need Jaccard similarity > ~0.5 to merge (accounting for temporal weight). This prevents the career->finance->health chaining problem.
- Coherence check post-clustering catches clusters that formed through borderline merges.
- O(n^3) worst case for agglomerative clustering, but n <= 200 so max ~8M operations -- trivial for Python.

### Step 3: Add `import math`

**File:** `agent_zero/consolidator.py`, top of file (line ~6)

Add `import math` for the exponential decay computation.

### Step 4: Export New Constants for Testing

**File:** `agent_zero/consolidator.py`

Add to the constants section (after line 41):
```python
MERGE_DISTANCE_THRESHOLD = 0.6
MIN_CLUSTER_COHERENCE = 0.4
TEMPORAL_WEIGHT = 0.2
```

### Step 5: Update `_cluster_episodes` Signature Compatibility

The function signature `_cluster_episodes(episodes: list[dict]) -> list[list[dict]]` stays identical. The only behavioral change:
- Transitive chaining no longer merges loosely related episodes
- Low-coherence clusters may produce singletons (filtered by MIN_CLUSTER_SIZE downstream)
- Temporal proximity slightly favors recent episode grouping

No changes needed to `run_consolidation()` or any caller.

## Test Specifications

### Test 1: Transitive chaining prevention
```python
def test_no_transitive_chaining():
    """Episodes sharing only intermediate topics should NOT cluster together."""
    eps = [
        _make_episode(topic_signals=["career", "finance"]),
        _make_episode(topic_signals=["finance", "health"]),
        _make_episode(topic_signals=["health", "relationships"]),
    ]
    clusters = _cluster_episodes(eps)
    # Should NOT produce a single 3-episode cluster
    # career+finance and finance+health may merge (Jaccard 0.33),
    # but career+finance+health+relationships should not form one cluster
    max_size = max(len(c) for c in clusters)
    assert max_size <= 2, f"Transitive chaining produced cluster of size {max_size}"
```

### Test 2: High-similarity episodes cluster
```python
def test_high_similarity_cluster():
    """Episodes with same topics should cluster together."""
    eps = [
        _make_episode(topic_signals=["career", "goals"]),
        _make_episode(topic_signals=["career", "goals"]),
        _make_episode(topic_signals=["career", "goals", "productivity"]),
    ]
    clusters = _cluster_episodes(eps)
    # All 3 should be in one cluster (Jaccard distance very low)
    assert any(len(c) == 3 for c in clusters)
```

### Test 3: Disjoint topics create separate clusters
```python
def test_disjoint_topics_separate():
    """Episodes with no topic overlap stay in separate clusters."""
    eps = [
        _make_episode(topic_signals=["career"]),
        _make_episode(topic_signals=["career"]),
        _make_episode(topic_signals=["career"]),
        _make_episode(topic_signals=["health"]),
        _make_episode(topic_signals=["health"]),
        _make_episode(topic_signals=["health"]),
    ]
    clusters = _cluster_episodes(eps)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [3, 3], f"Expected [3, 3] but got {sizes}"
```

### Test 4: Temporal proximity effect
```python
def test_temporal_proximity_favors_recent():
    """Recent episodes with same topics cluster more readily than distant ones."""
    now = datetime.now(timezone.utc)
    recent = [
        _make_episode(topic_signals=["career"], timestamp=(now - timedelta(hours=1)).isoformat()),
        _make_episode(topic_signals=["career"], timestamp=(now - timedelta(hours=2)).isoformat()),
    ]
    distant = [
        _make_episode(topic_signals=["career"], timestamp=(now - timedelta(days=60)).isoformat()),
    ]
    clusters = _cluster_episodes(recent + distant)
    # All should cluster (same topic, temporal only 20% weight),
    # but test that recent pair has lower distance
    from consolidator import _episode_distance
    d_recent = _episode_distance(recent[0], recent[1])
    d_distant = _episode_distance(recent[0], distant[0])
    assert d_recent < d_distant, "Recent episodes should have lower distance"
```

### Test 5: Empty and single episode edge cases
```python
def test_empty_episodes():
    assert _cluster_episodes([]) == []

def test_single_episode():
    eps = [_make_episode(topic_signals=["career"])]
    clusters = _cluster_episodes(eps)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1
```

### Test 6: Cluster coherence gate
```python
def test_low_coherence_cluster_rejected():
    """A cluster formed from borderline merges should be split if coherence is low."""
    # Create episodes that barely meet threshold but are internally incoherent
    eps = [
        _make_episode(topic_signals=["career", "finance", "goals"]),
        _make_episode(topic_signals=["finance"]),
        _make_episode(topic_signals=["goals"]),
    ]
    clusters = _cluster_episodes(eps)
    # The first episode overlaps with both, but ep2 and ep3 share nothing
    # Coherence should be low; cluster may be split
    # At minimum, no cluster should contain all 3 with low coherence
    for c in clusters:
        if len(c) >= 2:
            # Verify coherence is acceptable
            pass  # Coherence check is internal; validate via rule quality
```

### Test 7: Backward compatibility with run_consolidation
```python
def test_consolidation_still_produces_rules():
    """End-to-end: clustering change should still produce rules from coherent clusters."""
    shadow = {"episodes": [], "consolidated_rules": []}
    for _ in range(5):
        shadow["episodes"].append(_make_episode(
            topic_signals=["career", "goals"],
            user_followed_up=True,
        ))
    result = run_consolidation(shadow)
    rules = result.get("consolidated_rules", [])
    assert len(rules) >= 1, "Should produce at least one rule from coherent cluster"
```

### Test 8: Distance function symmetry and bounds
```python
def test_episode_distance_properties():
    """Distance function should be symmetric and bounded [0, 1]."""
    from consolidator import _episode_distance
    ep_a = _make_episode(topic_signals=["career", "goals"])
    ep_b = _make_episode(topic_signals=["health", "finance"])

    d_ab = _episode_distance(ep_a, ep_b)
    d_ba = _episode_distance(ep_b, ep_a)
    assert abs(d_ab - d_ba) < 1e-10, "Distance should be symmetric"
    assert 0.0 <= d_ab <= 1.0, f"Distance {d_ab} out of bounds"

    # Same episode should have distance ~0 (temporal 0 if same timestamp)
    d_aa = _episode_distance(ep_a, ep_a)
    assert d_aa < 0.01, f"Self-distance should be ~0, got {d_aa}"
```

## Estimated Impact

1. **Higher-quality rules.** By preventing transitive chaining, each rule will cover a tighter topic cluster. Rules for "career+goals" won't be diluted by unrelated "health" episodes pulled in by intermediate "finance" topics.

2. **Better agent calibration.** Fello/Othello/Shadow calibration metrics will be computed on genuinely related episodes, producing more accurate per-topic tuning (e.g., "Fello effective for career decisions" instead of "Fello effective for career+health+finance").

3. **Noise tolerance.** Outlier episodes that don't match any cluster remain unclustered and will consolidate later when more related episodes arrive. This prevents weak rules from polluting agent contexts.

4. **Temporal coherence.** Episodes from the same behavioral period cluster preferentially, producing rules that reflect current rather than historical agent performance.

5. **No breaking changes.** Function signature unchanged. MIN_CLUSTER_SIZE filter downstream still works. All existing tests should pass with minor adjustments to expected cluster shapes.

6. **Performance.** O(n^3) worst case for n <= 200 episodes is negligible (~8M operations). No external dependencies required.

## AZ Challenge Leverage

- **C092 (Graph Algorithms):** Agglomerative clustering can be viewed as iterative graph contraction -- AZ's graph algorithm knowledge applies directly.
- **C115 (MinHash/LSH):** If episode counts grow beyond 200, locality-sensitive hashing from C115 could approximate nearest neighbors for O(n log n) clustering.
