---
topic: Episode-Intervention Outcome Synchronization
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T14:00:00Z
---

# Episode-Intervention Outcome Synchronization

## Problem Statement

Episodes and interventions in Agent Zero are tracked by two independent systems that never
link to each other. The episode store (in-memory shadow dict) and intervention tracker
(PostgreSQL) maintain separate outcome pattern sets, separate topic taxonomies, and no
bidirectional references. The cross-reference code at `episode_store.py:305-320` is
dead code -- it searches for intervention_ids that are never populated in episodes.

This means:
1. Consolidation rules derived from episodes have no knowledge of which interventions worked
2. Intervention effectiveness cannot be measured because outcomes diverge between systems
3. The ENGAGE framework (Saldanha et al., 2025) identifies that "interaction with a DHI on
   its own is rarely sufficient" -- explicit outcome-intervention linking is required

## Current State in Agent Zero

### Episode Creation (NO intervention link)
- `cognitive_runtime.py:1081`: `capture_episode(blackboard.snapshot())`
- `episode_store.py:112-219`: `capture_episode()` accepts optional `intervention_id` parameter
- The parameter is **never passed** from cognitive_runtime.py
- Result: all episodes have `intervention: None`

### Intervention Logging (NO episode link)
- `agent_zero_server.py:2835-2861`: `log_intervention()` happens AFTER response generation
- Returns `intervention_id` UUID
- No mechanism to link this back to the episode created earlier in the same turn
- `database.py:213-231`: `intervention_log` table has no `episode_id` column

### Dead Cross-Reference Code
- `episode_store.py:305-320`: Loops through `intervention_resolutions` trying to match
  `intervention_id` in episodes -- but episodes never have this field populated
- This code path has never executed successfully in production

### Duplicated Outcome Patterns
Two separate pattern sets that have already diverged:

**episode_store.py (lines 38-56):**
- `_POSITIVE_OUTCOME`: "i did", "completed", "finished", "accomplished", etc.
- `_IGNORED_OUTCOME`: "i didn't", "forgot", "skipped", etc.
- `_PUSHBACK_OUTCOME`: "that doesn't work", "i can't", etc.

**intervention_tracker.py (lines 21-58):**
- Same categories but different patterns
- intervention_tracker has "i disagree with that" (line 57) -- episode_store doesn't
- Different regex compilation and matching logic

### Divergent Topic Taxonomies
**episode_store.py (lines 59-70):**
- Topics: career, learning, health, relationships, creativity, productivity, finance,
  mental_health, goals, technology

**intervention_tracker.py (lines 60-66):**
- Topics: career, productivity, health, relationships, technology
- Missing: learning, creativity, finance, mental_health, goals
- Has "procrast" keyword in productivity -- episode_store doesn't

### Temporal Mismatch
Turn processing order (agent_zero_server.py):
1. Line 1297: `resolve_pending_interventions_from_user_turn()` -- updates DB
2. Line 1870: `resolve_episode_outcomes(shadow, user_content, resolved)` -- updates shadow
3. Agents run, episode captured (without intervention_id)
4. Line 1871: `append_episode(shadow, episode)`
5. Line 2835: `log_intervention()` -- creates NEW intervention in DB
6. No link back to episode from step 4

## Industry Standard / Research Findings

### 1. ENGAGE Framework -- Outcome-Intervention Linking
Saldanha et al. (2025) propose a six-step cyclical framework for digital health interventions.
Step 5 ("Generate Evidence") explicitly requires:
- Measuring clinical, behavioral, and subjective outcomes **simultaneously**
- Closing feedback loops where outcome data continuously refines strategy
- Embedding measurement as a dedicated operational step, not an afterthought

Agent Zero's current architecture treats outcome measurement as an afterthought --
interventions are logged after response generation with no link to the episode that
captured the user's behavioral signals.

Citation: Saldanha et al., "Achieving clinically meaningful outcomes in digital health:
a six-step cyclical precision engagement framework (ENGAGE)," Frontiers in Digital Health,
2025. https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1713334/full

### 2. Event Sourcing for Behavioral Audit Trails
PubNub (2026) identifies event sourcing as essential for behavioral health data integrity:
"indisputable audit trails via event sourcing verifying when, where, and how mistakes were
made." The pattern: every state change is an immutable event, and current state is derived
by replaying events.

For Agent Zero, this means episodes and interventions should share a common event stream
(the turn) rather than being independently derived from the same user message.

Citation: PubNub, "Architecting the Synchronized Digital Health System: Top Trends for 2026,"
https://www.pubnub.com/blog/architecting-the-synchronized-digital-health-system-2026-trends/

### 3. AI Agent Behavioral Science -- Hypothesis-Driven Intervention
Anthropic, Google, and academic researchers (2025) propose AI Agent Behavioral Science:
"systematic observation of behavior, hypothesis-driven intervention design, and
theory-informed interpretation." This requires:
- Each intervention to be a testable hypothesis (did it work?)
- Outcomes to be linked to specific interventions for measurement
- Feedback loops that update strategy based on measured outcomes

Citation: "AI Agent Behavioral Science," arXiv:2506.06366v2, 2025.
https://arxiv.org/html/2506.06366v2

### 4. Bidirectional Outcome Resolution
The scoping review by PMC (2025) on AI-driven digital interventions found that
"almost half of all DHIs reported at least one evaluation of behavioral outcomes,
mostly employing RCTs." The key finding: interventions without outcome tracking
cannot be evaluated, and systems that track outcomes independently from interventions
produce unreliable effect estimates.

Citation: "A Scoping Review of AI-Driven Digital Interventions in Mental Health Care,"
PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12110772/

## Proposed Implementation

### Step 1: Shared Outcome Patterns (episode_store.py + intervention_tracker.py)

Create a shared module `agent_zero/outcome_patterns.py` (~40 lines):
```python
"""Shared outcome detection patterns used by both episode and intervention systems."""

import re

POSITIVE_PATTERNS = [
    r"\bi did\b", r"\bcompleted\b", r"\bfinished\b", r"\baccomplished\b",
    r"\bi('ve| have) (done|started|tried|practiced)\b",
    r"\bfollowed through\b", r"\bkept my\b", r"\bstuck with\b",
]

IGNORED_PATTERNS = [
    r"\bi didn'?t\b", r"\bforgot\b", r"\bskipped\b", r"\bdidn'?t get to\b",
    r"\bnever got around\b", r"\bi disagree with that\b",
]

PUSHBACK_PATTERNS = [
    r"\bthat doesn'?t work\b", r"\bi can'?t\b", r"\btoo (hard|much|difficult)\b",
    r"\bnot realistic\b", r"\bnot helpful\b",
]

TOPIC_KEYWORDS = {
    "career": ["career", "job", "work", "promotion", "interview", "salary", "manager", "raise", "role"],
    "learning": ["learn", "study", "course", "understand", "practice"],
    "health": ["health", "exercise", "sleep", "diet", "weight", "fitness", "walk", "run"],
    "relationships": ["relationship", "partner", "friend", "family", "social", "connect"],
    "creativity": ["creative", "art", "music", "write", "design", "build"],
    "productivity": ["focus", "habit", "routine", "procrast", "stuck", "task", "organize"],
    "finance": ["money", "budget", "save", "invest", "debt", "spend"],
    "mental_health": ["anxiety", "stress", "depression", "therapy", "mindful", "overwhelm"],
    "goals": ["goal", "plan", "milestone", "progress", "achieve", "target"],
    "technology": ["code", "program", "tech", "software", "app", "automate"],
}

_compiled = {
    "positive": [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS],
    "ignored": [re.compile(p, re.IGNORECASE) for p in IGNORED_PATTERNS],
    "pushback": [re.compile(p, re.IGNORECASE) for p in PUSHBACK_PATTERNS],
}

def classify_outcome(text: str) -> str:
    """Return 'acted', 'ignored', 'pushed_back', or 'neutral'."""
    for pat in _compiled["positive"]:
        if pat.search(text):
            return "acted"
    for pat in _compiled["pushback"]:
        if pat.search(text):
            return "pushed_back"
    for pat in _compiled["ignored"]:
        if pat.search(text):
            return "ignored"
    return "neutral"

def extract_topics(text: str) -> list[str]:
    """Return list of matching topic tags."""
    lower = text.lower()
    return [topic for topic, kws in TOPIC_KEYWORDS.items()
            if any(kw in lower for kw in kws)]
```

Then update both `episode_store.py` and `intervention_tracker.py` to import from this module
instead of maintaining separate pattern sets.

### Step 2: Add Turn-Level Linking

#### 2a. Add `turn_id` to episodes (episode_store.py)
The `capture_episode()` function already accepts context that includes a turn identifier.
Ensure every episode gets a `turn_id` field (UUID generated at start of each turn).

#### 2b. Add `turn_id` to interventions (intervention_tracker.py + database.py)
Add `turn_id` column to `intervention_log` table:
```sql
ALTER TABLE intervention_log ADD COLUMN turn_id TEXT;
```

Pass `turn_id` from `agent_zero_server.py` when calling `log_intervention()`.

#### 2c. Generate turn_id at turn start (agent_zero_server.py)
At the top of `_run_conversation_turn()`, generate:
```python
turn_id = str(uuid.uuid4())
```
Pass this to both `capture_episode()` (via blackboard) and `log_intervention()`.

### Step 3: Retroactive Episode-Intervention Linking

After `log_intervention()` returns the `intervention_id` (agent_zero_server.py ~line 2835),
link it back to the episode created in the same turn:

```python
# After log_intervention returns intervention_id:
if intervention_id and shadow.get("episodes"):
    for ep in reversed(shadow["episodes"]):
        if ep.get("turn_id") == turn_id:
            ep["intervention"] = {"logged": True, "intervention_id": intervention_id}
            break
```

This fixes the dead code at `episode_store.py:305-320` by ensuring episodes actually
have intervention_ids.

### Step 4: Bidirectional Outcome Resolution

Update `resolve_episode_outcomes()` to also update the intervention in DB:

```python
# In resolve_episode_outcomes(), after determining outcome for an episode
# that has an intervention_id:
if ep_intervention_id:
    # Queue DB update (don't await in hot path)
    pending_intervention_updates.append(
        (ep_intervention_id, episode_outcome)
    )
```

Then after the function returns, batch-update interventions:
```python
for iid, outcome in pending_intervention_updates:
    await resilient_call(
        execute, "UPDATE intervention_log SET outcome=$1, resolved_at=now() WHERE id=$2",
        outcome, iid, circuit=db_circuit, max_retries=1, timeout_s=3.0,
        fallback=None, operation_name="sync_intervention_outcome"
    )
```

### Step 5: Consolidation Rule Enhancement

With episodes now linked to interventions, `run_consolidation()` can generate rules
that reference intervention effectiveness:

```python
# In consolidation rule generation, when an episode has intervention data:
if episode.get("intervention", {}).get("logged"):
    rule["intervention_id"] = episode["intervention"]["intervention_id"]
    rule["intervention_effective"] = episode["outcome"].get("intervention_outcome") == "acted"
```

This enables the routing system to weight agents based on which interventions
actually led to behavior change.

## Test Specifications

### test_outcome_patterns.py (~10 tests)
```python
def test_classify_positive_outcome():
    """'I did my homework' -> 'acted'."""

def test_classify_ignored_outcome():
    """'I forgot about it' -> 'ignored'."""

def test_classify_pushback_outcome():
    """'That doesn't work for me' -> 'pushed_back'."""

def test_classify_neutral():
    """'What should I do next?' -> 'neutral'."""

def test_extract_topics_career():
    """'my job interview' -> ['career']."""

def test_extract_topics_multiple():
    """'I need to exercise and study' -> ['learning', 'health']."""

def test_patterns_are_superset():
    """All patterns from both systems are present in shared module."""
```

### test_episode_intervention_sync.py (~12 tests)
```python
def test_turn_id_propagated_to_episode():
    """Episode created in turn gets turn_id from blackboard."""

def test_turn_id_propagated_to_intervention():
    """Intervention logged in turn gets same turn_id."""

def test_retroactive_linking():
    """After log_intervention(), episode in same turn gets intervention_id."""

def test_retroactive_linking_finds_correct_episode():
    """Links to most recent episode with matching turn_id, not older ones."""

def test_bidirectional_outcome_sync():
    """Episode outcome updates intervention outcome in DB."""

def test_outcome_sync_handles_missing_intervention():
    """Episodes without intervention_id don't attempt DB update."""

def test_consolidation_includes_intervention_effectiveness():
    """Rules from linked episodes include intervention_effective field."""

def test_cross_reference_code_now_executes():
    """episode_store.py:305-320 finds matches when intervention_ids are populated."""

def test_shared_patterns_match_both_systems():
    """Same text produces same outcome in episode and intervention systems."""

def test_topic_taxonomy_unified():
    """Both systems use identical topic keywords from shared module."""

def test_missing_turn_id_graceful():
    """If turn_id is missing, linking is skipped without error."""

def test_batch_intervention_update():
    """Multiple intervention outcomes updated in single batch."""
```

## Estimated Impact

- **Intervention effectiveness measurement**: For the first time, Agent Zero can answer
  "which interventions led to behavior change?" by linking episodes to interventions
- **Consolidation quality**: Rules based on outcome data include intervention context,
  enabling better routing decisions
- **Pattern consistency**: Single source of truth for outcome patterns eliminates
  divergence risk (currently 5+ patterns differ between systems)
- **Dead code elimination**: Cross-reference code at episode_store.py:305-320 becomes functional
- **Research alignment**: Implements ENGAGE framework Step 5 (Generate Evidence) and
  AI Agent Behavioral Science principles (hypothesis-driven intervention)

## Citations

1. Saldanha et al., "Achieving clinically meaningful outcomes in digital health: a six-step cyclical precision engagement framework (ENGAGE)," Frontiers in Digital Health, 2025. https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1713334/full
2. PubNub, "Architecting the Synchronized Digital Health System: Top Trends for 2026," 2025. https://www.pubnub.com/blog/architecting-the-synchronized-digital-health-system-2026-trends/
3. "AI Agent Behavioral Science," arXiv:2506.06366v2, 2025. https://arxiv.org/html/2506.06366v2
4. "A Scoping Review of AI-Driven Digital Interventions in Mental Health Care," PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12110772/
5. Ali et al., "Artificial intelligence for mental health: A narrative review," SAGE Digital Health, 2025. https://journals.sagepub.com/doi/10.1177/20552076251395548
6. "Sustainability of AI-Assisted Mental Health Intervention: A Review 2020-2025," PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12469610/
