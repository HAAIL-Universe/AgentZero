---
topic: Consolidated Rules Growth Cap
status: ready_for_implementation
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T00:00:00Z
---

# Consolidated Rules Growth Cap

## Problem Statement

The `consolidated_rules` array in the behavioural shadow stores rules with status "active", "stale", and "retired". While MAX_ACTIVE_RULES (20) caps active rules, **retired and stale rules accumulate indefinitely**. After months of use, a user could have hundreds of retired rules in the array -- each containing episode IDs (up to 50), agent calibration dicts, intervention effectiveness dicts, quality scores, and temporal patterns. At ~2KB per rule, 500 retired rules = ~1MB of dead weight per user profile, growing without bound. This bloats the JSONB shadow profile stored in PostgreSQL, increases serialization/deserialization time, and wastes bandwidth on every shadow read/write.

Additionally, there is no audit trail when rules are retired -- no log of what was removed or why, making it impossible to debug rule quality regressions.

## Current State in Agent Zero

**File:** `agent_zero/consolidator.py`

### Rule Lifecycle (lines 918-942):
1. **Creation** (line 911): New rules appended to `shadow["consolidated_rules"]` with status "active"
2. **Staleness** (line 934): Rules marked "stale" after 30 days if confidence < 0.5
3. **Retirement** (line 933): Rules marked "retired" after 90 days regardless of confidence
4. **Cap enforcement** (lines 938-942): If active count > MAX_ACTIVE_RULES (20), oldest active rules retired

### What's missing:
- **No removal of retired rules from array** -- status changes to "retired" but rule object stays forever
- **No total array size cap** -- only active count is bounded
- **No audit log** -- when a rule transitions status, nothing is recorded
- **No archival** -- retired rules could be moved to a separate table for historical analysis
- **No memory-aware pruning** -- no composite scoring for which rules to keep vs archive

### Configuration (config.py lines 134-153):
```python
MAX_ACTIVE_RULES = 20    # range [5, 100]
STALE_DAYS = 30          # range [7, 180]
RETIRED_DAYS = 90        # range [30, 365]
```

### Data flow:
- Rules stored in `shadow["consolidated_rules"]` (JSONB in behavioural_shadow table)
- Entire shadow profile read/written as single JSONB blob (behavioural_shadow.py lines 153-181)
- `get_relevant_rules()` (line 968) filters by status="active" at retrieval time but loads all rules

## Industry Standard / Research Findings

### 1. Three-Tier Pruning with Composite Importance Score

OneUptime's memory consolidation framework (2026) defines a production-proven approach to bounded knowledge stores:

**Composite importance score** (three weighted factors):
- **Recency (30%)**: `exp(-ln(2) * days_since_use / half_life)` with configurable half-life (30 days default)
- **Usage (40%)**: `min(1.0, log10(usage_count + 1) / 2)` -- logarithmic scaling prevents outliers
- **Confidence (30%)**: direct 0-1 probability measure

**Three-tier classification**:
- **Retain** (score >= 0.6): Keep in active store
- **Archive** (0.3-0.6): Move to cold storage, retrievable on demand
- **Delete** (< 0.3): Remove permanently

**Protected categories**: Error recoveries, edge cases, explicit user feedback, and high-usage items (>10 uses) go to archive instead of delete.

Source: https://oneuptime.com/blog/post/2026-01-30-memory-consolidation/view

### 2. FSRS Retrievability Decay

The Free Spaced Repetition Scheduler (FSRS-6, default in Anki) models memory decay as:

`R(t,S) = (1 + FACTOR * t/S)^DECAY`

Where R = retrievability (probability of recall), t = time since last access, S = stability (time for R to drop from 100% to 90%). This power-law decay is more accurate than exponential for long-term knowledge retention.

Applied to rules: each time a rule is matched by `get_relevant_rules()`, its stability increases. Rules that go unmatched for long periods see retrievability drop below threshold, triggering archival.

Source: https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm

### 3. RETE/Phreak Rule Lifecycle Management

Production rule systems (Drools Phreak algorithm) manage large rule sets via:
- **Salience-ordered priority queues** -- only evaluate rules in active agenda groups
- **Lazy evaluation** -- don't evaluate rules until needed
- **Refraction** -- prevent fired rules from re-firing on same data

The key principle: **active forgetting** is essential. Cognitive science research confirms that memories (rules) should be actively removed or modified based on relevance and recency, not just passively decayed.

Sources:
- https://docs.redhat.com/en/documentation/red_hat_decision_manager/7.4/html/decision_engine_in_red_hat_decision_manager/phreak-algorithm-con_decision-engine
- https://dl.acm.org/doi/10.1145/3748302 (Survey on Memory Mechanism of LLM-based Agents)

### 4. Knowledge Acquisition Dynamics -- Permanence Measure

Research on knowledge acquisition dynamics (EmergentMind) uses a "permanence" measure based on Minimum Message Length (MML) principle. Rules are promoted or pruned based on payoffs in compression and evidence coverage. Working hypothesis pools with promotion/demotion thresholds maintain balance between stability and plasticity.

Source: https://www.emergentmind.com/topics/knowledge-acquisition-dynamics

### 5. Agent Cognitive Compressor (ACC)

ArXiv 2025: Bio-inspired memory controller that uses bounded internal state with controlled replacement rather than unbounded growth. Key insight: cap the store, replace lowest-value entries.

Source: https://arxiv.org/html/2601.11653

## Proposed Implementation

### Step 1: Add Total Rules Cap (consolidator.py)

Add configuration:
```python
# config.py
MAX_TOTAL_RULES = 50   # range [20, 200], default 50
```

In `run_consolidation()` after the retirement logic (line 942), add total-array pruning:

```python
# After line 942 in consolidator.py
# --- Total rules cap: remove lowest-scoring retired rules ---
if len(rules) > config.MAX_TOTAL_RULES:
    # Score each retired rule by composite importance
    def _rule_importance(rule):
        if rule.get("status") == "active":
            return float('inf')  # never prune active rules here

        days_since_update = (now - parse_iso(rule["last_updated"])).total_seconds() / 86400
        recency = math.exp(-0.693 * days_since_update / 30)  # 30-day half-life
        usage = min(1.0, math.log10(rule.get("episode_count", 1) + 1) / 2)
        confidence = rule.get("confidence", 0.0)
        return 0.3 * recency + 0.4 * usage + 0.3 * confidence

    # Sort by importance ascending, prune lowest
    scored = [(r, _rule_importance(r)) for r in rules]
    scored.sort(key=lambda x: x[1])

    to_remove = len(rules) - config.MAX_TOTAL_RULES
    removed = []
    for rule, score in scored:
        if to_remove <= 0:
            break
        if rule.get("status") != "active":
            removed.append(rule)
            rules.remove(rule)
            to_remove -= 1

    # Log pruned rules
    if removed:
        _log_rule_retirement(shadow, removed, "total_cap_exceeded")
```

### Step 2: Add Audit Log for Rule Transitions (consolidator.py)

Add a `rule_audit_log` list to the shadow profile (capped at last 200 entries):

```python
def _log_rule_retirement(shadow, rules, reason):
    """Record rule retirements for debugging and quality analysis."""
    audit = shadow.setdefault("rule_audit_log", [])
    now_iso = datetime.utcnow().isoformat()
    for rule in rules:
        audit.append({
            "rule_id": rule["rule_id"],
            "action": "pruned",
            "reason": reason,
            "timestamp": now_iso,
            "confidence": rule.get("confidence", 0),
            "episode_count": rule.get("episode_count", 0),
            "age_days": (datetime.utcnow() - parse_iso(rule["created_at"])).days,
        })
    # Cap audit log at 200 entries
    if len(audit) > 200:
        shadow["rule_audit_log"] = audit[-200:]
```

Also log status transitions (stale, retired) in existing retirement logic at lines 932-935.

### Step 3: Update empty_shadow() (behavioural_shadow.py)

Add `rule_audit_log` to the shadow profile template:

```python
# In empty_shadow(), after consolidated_rules
"rule_audit_log": [],
```

### Step 4: Add Configuration (config.py)

```python
MAX_TOTAL_RULES = Field(
    default=50, ge=20, le=200,
    description="Maximum total rules (active + stale + retired) before pruning"
)
```

### Step 5: Optimize get_relevant_rules() (consolidator.py, line 968)

Currently loads all rules then filters by status. Add early filter:

```python
# Line 975: change from
active_rules = [r for r in rules if r.get("status") == "active"]
# This already exists, but ensure we never iterate retired rules for scoring
```

No change needed -- existing filter at line 978 already skips non-active rules. The optimization is that with the total cap, the iteration is bounded.

## Test Specifications

```python
def test_total_rules_cap_enforced():
    """When rules exceed MAX_TOTAL_RULES, lowest-importance retired rules are pruned."""
    # Setup: shadow with 60 rules (20 active, 40 retired with varying ages/confidence)
    # Run consolidation
    # Assert: total rules <= MAX_TOTAL_RULES
    # Assert: all 20 active rules still present
    # Assert: removed rules had lowest composite importance scores

def test_active_rules_never_pruned_by_total_cap():
    """Active rules are never removed by total cap enforcement."""
    # Setup: shadow with MAX_TOTAL_RULES + 10 rules, all active
    # Run consolidation
    # Assert: active rules untouched (only active->retired transition via MAX_ACTIVE_RULES)

def test_audit_log_records_pruning():
    """Pruned rules are recorded in rule_audit_log with reason and metadata."""
    # Setup: shadow exceeding total cap
    # Run consolidation
    # Assert: rule_audit_log entries exist for each pruned rule
    # Assert: entries contain rule_id, action, reason, timestamp, confidence, episode_count, age_days

def test_audit_log_capped_at_200():
    """Audit log never exceeds 200 entries."""
    # Setup: shadow with existing 195 audit entries, prune 10 rules
    # Assert: audit log has exactly 200 entries (oldest 5 dropped)

def test_composite_importance_score():
    """Composite score correctly weights recency (30%), usage (40%), confidence (30%)."""
    # Test cases:
    # - Recent (1 day), high usage (50 episodes), high confidence (0.9) -> high score
    # - Old (180 days), low usage (1 episode), low confidence (0.1) -> low score
    # - Recent but low confidence -> medium score

def test_stale_transition_logged():
    """Status transition from active to stale is recorded in audit log."""

def test_retired_transition_logged():
    """Status transition from active/stale to retired is recorded in audit log."""

def test_empty_shadow_has_audit_log():
    """empty_shadow() includes rule_audit_log as empty list."""

def test_max_total_rules_config():
    """MAX_TOTAL_RULES is configurable with range [20, 200], default 50."""
```

## Estimated Impact

- **Storage**: Prevents unbounded JSONB growth. At 50 total rules max, shadow profile stays under ~100KB for rules (vs potentially MBs after months of use).
- **Performance**: Faster shadow read/write cycles due to bounded JSONB size. Estimated 20-50ms savings per shadow operation for long-running users.
- **Debugging**: Audit log enables tracing rule quality regressions -- when did a good rule get retired? What replaced it?
- **Memory**: Reduced memory pressure on server during consolidation runs (fewer rules to iterate).
- **Data integrity**: Composite importance scoring ensures valuable retired rules are kept longer than worthless ones, rather than simple FIFO removal.
