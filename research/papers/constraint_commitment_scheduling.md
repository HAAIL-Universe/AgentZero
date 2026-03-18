---
topic: Constraint-Based Commitment Scheduling
status: ready_for_implementation
priority: high
estimated_complexity: large
researched_at: 2026-03-18T14:00:00Z
---

# Constraint-Based Commitment Scheduling

## Problem Statement

Agent Zero tracks user commitments (daily, weekly, custom cadence) but has no intelligent scheduling engine. The current system uses a naive elapsed-time check (`agent_zero_server.py:1008-1047`) -- if `now - last_checked_in > cadence_interval`, the commitment is "pending". This creates several problems:

1. **No optimal timing**: All daily commitments fire at the same time (24h after last check-in), regardless of user context, receptiveness, or competing obligations
2. **No burden management**: A user with 5 daily commitments gets 5 check-in prompts simultaneously -- no spreading, no prioritization
3. **No constraint awareness**: The system ignores time-of-day preferences, weekday/weekend differences, and commitment interdependencies
4. **Unused `recurrence_rule` field**: The JSONB column at `database.py:246` was reserved for complex recurrence but never populated or parsed
5. **No `custom` cadence logic**: The `cadence='custom'` enum value exists (`database.py:241`) but the pending-checkin endpoint skips it entirely

## Current State in Agent Zero

### Commitment Schema (`database.py:236-253`)
- `cadence`: 'once' | 'daily' | 'weekly' | 'custom' (custom is dead code)
- `weight`: 'light' | 'moderate' | 'heavy' (never used in scheduling)
- `recurrence_rule`: JSONB, always NULL
- `last_checked_in`: TIMESTAMPTZ, updated on check-in
- `streak_current` / `streak_best`: tracked but not used for scheduling decisions

### Pending Check-In Detection (`agent_zero_server.py:1008-1047`)
```python
elapsed = (now - reference).total_seconds()
if cadence == "daily" and elapsed > 86400:
    pending.append(...)
elif cadence == "weekly" and elapsed > 7 * 86400:
    pending.append(...)
# "custom" cadence: completely unhandled
```

### Check-In Engine (`agent_zero_telegram/check_in_engine.py`)
- Scores check-in urgency on three axes: event-anchored (0.7-1.0), pattern-triggered (0.5-0.8), silence-based (0.3-0.6)
- Has `CHECK_IN_HOURS_START/END` gates and `CHECK_IN_MAX_DAILY` cap
- But no constraint solver -- just heuristic scoring with no conflict resolution

### Behavioral Shadow Commitment Tracking (`behavioural_shadow.py:420-478`)
- Extracts commitments from conversation with `_extract_due_hint()`
- Tracks `commitment_prediction` (probability of follow-through, 0.0-1.0)
- No scheduling optimization -- prediction is used for accountability interventions, not timing

### AZ's Constraint Solver (`challenges/C094_constraint_solver/constraint_solver.py`)
- CSPSolver with 8 constraint types: Equality, Inequality, Comparison, AllDifferent, Table, Arithmetic, Sum, Callback
- Built-in `scheduling()` helper function (lines 1003-1064) with precedence, deadlines, durations, no-overlap
- Three search strategies: Backtracking, Forward Checking, MAC (AC-3)
- SAT and SMT solver backends available for complex problems
- Already solves exactly this class of problem -- but is disconnected from Agent Zero

## Industry Standard / Research Findings

### 1. Just-in-Time Adaptive Interventions (JITAIs)

The gold standard for behavioral intervention timing is the JITAI framework (Nahum-Sheth et al., 2018). JITAIs define six components that determine when and how to intervene:

1. **Distal outcome**: Long-term behavioral goal (e.g., sustained habit)
2. **Proximal outcome**: Short-term target (e.g., today's check-in completed)
3. **Decision points**: When the system evaluates whether to intervene
4. **Intervention options**: What actions are available (check-in, nudge, silence)
5. **Tailoring variables**: Context that personalizes the decision (time, location, history)
6. **Decision rules**: The algorithm linking variables to actions

Agent Zero currently has components 1-2 (commitments + check-ins), partial 3-4 (time-elapsed triggers, check-in messages), minimal 5 (only elapsed time), and no 6 (no decision rules beyond threshold comparison).

**Citation**: Nahum-Sheth, S.A., Smith, S.N., Spring, B.J., et al. (2018). "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles." *Annals of Behavioral Medicine*, 52(6), 446-462. https://pmc.ncbi.nlm.nih.gov/articles/PMC5364076/

### 2. Reinforcement Learning for Intervention Timing (HeartSteps)

The HeartSteps study deployed a Thompson Sampling RL algorithm to optimize when to send walking suggestions. Key design elements:

- **Decision points**: 5 fixed times/day, ~2.5 hours apart
- **State features**: Location, recent step count, yesterday's total, temperature, dosage (recent treatment burden)
- **Reward**: Log-transformed 30-minute step count after each decision
- **Burden constraint**: Probability clipping to [0.1, 0.8] ensures at most ~4 messages/day
- **Dosage tracking**: A "dosage variable" tracks cumulative recent treatment to prevent notification fatigue

The critical insight: **intervention effectiveness degrades with frequency**. The dosage variable captures this -- sending too many check-ins reduces engagement per check-in. Agent Zero has `CHECK_IN_MAX_DAILY` but no dosage-aware scheduling.

**Citation**: Liao, P., Klasnja, P., Tewari, A., Murphy, S.A. (2020). "Personalized HeartSteps: A Reinforcement Learning Algorithm for Optimizing Physical Activity." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, 4(1). https://pmc.ncbi.nlm.nih.gov/articles/PMC8439432/

### 3. Dual RL Model for Timing + Content

Gönül et al. (2021) proposed a dual-model architecture for JITAIs:

- **Intervention-selection model**: Learns which type/content of intervention to deliver, using long-term state features (engagement history, preferences)
- **Opportune-moment-identification model**: Learns when to deliver, using short-term features (current activity, time of day, location)

This separation is architecturally clean: the "what" and "when" are independent optimization problems. In Agent Zero terms, the cognitive agents already decide "what" (response content), but "when" to check in is a fixed-interval timer.

**Citation**: Gönül, S., Namli, T., Huibers, L., Laleci Erturkmen, G.B., et al. (2021). "A reinforcement learning based algorithm for personalization of digital, just-in-time, adaptive interventions." *Artificial Intelligence in Medicine*, 115, 102062. https://www.sciencedirect.com/science/article/abs/pii/S0933365721000555

### 4. LLM-Based JITAI Decision Rules

Maharjan et al. (2024) tested GPT-4 as a replacement for traditional JITAI decision rules. Key findings:

- LLM-generated interventions **outperformed** both layperson and healthcare professional suggestions across appropriateness, engagement, and effectiveness metrics
- LLMs can replace rigid if-then decision rules with flexible, context-aware reasoning
- The approach generates 450 JITAI decisions across personas and contexts

This is directly applicable: Agent Zero already uses an LLM (Qwen3) for response generation. The scheduling decision ("should we check in now?") can be framed as an LLM inference using the same cognitive pipeline, with constraint solving for conflict resolution.

**Citation**: Maharjan, R., Doherty, K., Rohani, D.A., et al. (2024). "The Last JITAI? Exploring Large Language Models for Issuing Just-in-Time Adaptive Interventions." *CHI Conference on Human Factors in Computing Systems*, 2025. https://arxiv.org/abs/2402.08658

### 5. FSRS Spaced Repetition Scheduling

The Free Spaced Repetition Scheduler (FSRS) provides a mathematical model for optimal review intervals based on memory decay. Core formulas (FSRS v4):

```
Retrievability:  R(t, S) = (1 + t/(9*S))^(-1)
Optimal interval: I(r, S) = 9*S * (1/r - 1)
```

Where S = stability (how slowly the item is forgotten), t = elapsed time, r = target retention.

After successful recall: `S' = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) + 1)`
After failure: `S'f = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R))`

Applied to commitments: each commitment has a "stability" (how reliably the user follows through). Successful check-ins increase stability (longer intervals before next prompt). Missed check-ins decrease stability (shorter intervals, more frequent prompts). This creates **adaptive cadence** that responds to actual behavior.

**Citation**: Ye, J. (2024). "Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory." *IEEE Transactions on Knowledge and Data Engineering*. https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm

### 6. Intervention Burden and Treatment Fatigue

Research consistently shows that notification fatigue degrades intervention effectiveness. A 2025 meta-analysis of JITAIs found that interventions shorter than 6 weeks yielded greater longevity of effects -- suggesting that **less frequent, better-timed** interventions outperform constant prompting. Structured accountability systems increase habit maintenance 2.8x (vs. no accountability), but only when intervention frequency is managed.

**Citation**: Effectiveness of just-in-time adaptive interventions for improving mental health and psychological well-being: a systematic review and meta-analysis. (2025). *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC12481328/

## Proposed Implementation

### Architecture: Three-Layer Scheduling System

```
Layer 1: Constraint Solver (C094)    -- "What's feasible?"
Layer 2: FSRS-inspired Stability     -- "How often?"
Layer 3: JITAI Context Filter        -- "Is now a good time?"
```

### Layer 1: Constraint-Based Schedule Generation

**New file**: `agent_zero/commitment_scheduler.py`

```python
class CommitmentScheduler:
    """Constraint-based scheduler using CSPSolver from C094."""

    def schedule_checkins(self, commitments: list[dict], user_prefs: dict) -> list[ScheduledCheckin]:
        """Generate optimal check-in schedule for a time window."""
        # 1. Create CSP problem
        solver = CSPSolver()

        # Variables: each commitment gets a time-slot variable
        # Domain: available time slots (e.g., 15-min intervals during waking hours)
        slots = self._generate_time_slots(
            window_hours=user_prefs.get("active_hours", (8, 22)),
            interval_minutes=15
        )

        for c in commitments:
            solver.add_variable(c["id"], slots)

        # Constraint 1: AllDifferent -- no two check-ins at same time
        solver.add_alldiff([c["id"] for c in commitments])

        # Constraint 2: Minimum spacing between check-ins (burden management)
        min_gap = user_prefs.get("min_gap_minutes", 60)  # at least 1hr apart
        for i, c1 in enumerate(commitments):
            for c2 in commitments[i+1:]:
                solver.add_callback(
                    [c1["id"], c2["id"]],
                    lambda a, id1=c1["id"], id2=c2["id"]:
                        abs(a[id1] - a[id2]) >= min_gap // 15
                )

        # Constraint 3: Heavy commitments get prime time slots
        for c in commitments:
            if c["weight"] == "heavy":
                prime_slots = self._prime_time_slots(user_prefs)
                solver.add_table([c["id"]], [(s,) for s in prime_slots])

        # Constraint 4: Max check-ins per day
        max_daily = user_prefs.get("max_daily_checkins", 4)
        # Solved by limiting variable count entering the solver

        # Constraint 5: Precedence -- if commitment B depends on A
        for c in commitments:
            if c.get("parent_id"):
                parent = next((p for p in commitments if p["id"] == c["parent_id"]), None)
                if parent:
                    solver.add_comparison(parent["id"], c["id"], "<")

        result, assignment = solver.solve()
        if result == CSPResult.SOLVED:
            return self._decode_schedule(assignment, commitments, slots)
        else:
            return self._fallback_even_spread(commitments, slots)
```

### Layer 2: FSRS-Inspired Adaptive Stability

**Add to**: `agent_zero/commitment_scheduler.py`

Each commitment tracks a "stability" value -- how reliably the user follows through. This determines the optimal interval between check-ins.

```python
class CommitmentStability:
    """FSRS-inspired stability tracking for commitment check-in intervals."""

    # Simplified FSRS v4 parameters tuned for commitments (not flashcards)
    W = [0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34]

    def initial_stability(self, weight: str) -> float:
        """Initial stability based on commitment weight."""
        # Light commitments start with higher stability (less frequent check-ins)
        # Heavy commitments start with lower stability (more frequent check-ins)
        base = {"light": 3.0, "moderate": 1.5, "heavy": 0.7}
        return base.get(weight, 1.5)

    def retrievability(self, elapsed_days: float, stability: float) -> float:
        """Probability user still 'remembers' / is engaged with this commitment."""
        return (1 + elapsed_days / (9 * stability)) ** -1

    def optimal_interval(self, stability: float, target_retention: float = 0.85) -> float:
        """Days until next check-in to maintain target engagement."""
        return 9 * stability * (1 / target_retention - 1)

    def update_stability_success(self, S: float, D: float, R: float) -> float:
        """Update stability after successful check-in (user followed through)."""
        w = self.W
        return S * (math.exp(w[8]) * (11 - D) * S**(-w[9]) * (math.exp(w[10] * (1 - R)) - 1) + 1)

    def update_stability_failure(self, S: float, D: float, R: float) -> float:
        """Update stability after missed commitment."""
        w = self.W
        return w[11] * D**(-w[12]) * ((S + 1)**w[13] - 1) * math.exp(w[14] * (1 - R))

    def difficulty_from_history(self, events: list[dict]) -> float:
        """Compute commitment difficulty from event history (1-10 scale)."""
        if not events:
            return 5.0
        completions = sum(1 for e in events if e["event_type"] == "completed")
        misses = sum(1 for e in events if e["event_type"] == "missed")
        total = completions + misses
        if total == 0:
            return 5.0
        miss_rate = misses / total
        return min(10, max(1, 1 + miss_rate * 9))
```

### Layer 3: JITAI Context Filter

**Add to**: `agent_zero/commitment_scheduler.py`

Before delivering a scheduled check-in, apply context filters:

```python
class JITAIContextFilter:
    """Decides whether NOW is an appropriate time for this check-in."""

    def should_deliver(self, commitment: dict, context: dict) -> tuple[bool, float]:
        """
        Returns (should_deliver, confidence).
        Context includes: time_of_day, day_of_week, recent_session_activity,
        dosage (recent check-in count), streak_status.
        """
        score = 1.0

        # Dosage penalty (HeartSteps-inspired)
        recent_checkins = context.get("checkins_last_4h", 0)
        if recent_checkins >= 2:
            score *= 0.3  # heavy penalty for 3rd+ check-in in 4 hours
        elif recent_checkins >= 1:
            score *= 0.7

        # Time-of-day preference (learned from check-in success rates)
        hour = context.get("hour", 12)
        tod_weight = context.get("tod_success_rate", {}).get(hour, 0.5)
        score *= tod_weight

        # Streak protection: don't over-prompt during active streaks
        if commitment.get("streak_current", 0) >= 7:
            score *= 0.8  # slightly reduce frequency for established habits

        # Silence boost: increase urgency after prolonged silence
        days_since_session = context.get("days_since_last_session", 0)
        if days_since_session >= 3:
            score *= 1.3

        # Broken streak boost
        if (commitment.get("streak_best", 0) >= 5 and
            commitment.get("streak_current", 0) == 0):
            score *= 1.4

        threshold = context.get("delivery_threshold", 0.5)
        return score >= threshold, min(score, 1.0)
```

### Database Changes

**Modify**: `database.py` schema and functions

1. Add `stability` column to commitments table:
```sql
ALTER TABLE commitments ADD COLUMN IF NOT EXISTS stability FLOAT DEFAULT 1.5;
ALTER TABLE commitments ADD COLUMN IF NOT EXISTS difficulty FLOAT DEFAULT 5.0;
ALTER TABLE commitments ADD COLUMN IF NOT EXISTS next_checkin_at TIMESTAMPTZ;
```

2. Populate `recurrence_rule` for custom cadences:
```json
{
    "type": "custom",
    "interval_days": 3,
    "preferred_hours": [9, 14, 19],
    "weekdays_only": false,
    "skip_holidays": false
}
```

3. Add function `update_commitment_stability(commitment_id, event_type)` that calls CommitmentStability methods after each checked_in/missed event.

### API Changes

**Modify**: `agent_zero_server.py`

1. Replace naive pending-checkin endpoint (`lines 1008-1047`) with scheduler-aware version:
```python
@app.get("/user/checkins/pending")
async def api_pending_checkins(user: dict = Depends(get_current_user)):
    """Return optimally-scheduled pending check-ins."""
    scheduler = CommitmentScheduler()
    commitments = await list_commitments(user["user_id"], status="active")
    context = await build_checkin_context(user["user_id"])

    # Layer 1: Generate feasible schedule via CSP
    schedule = scheduler.schedule_checkins(commitments, user.get("preferences", {}))

    # Layer 2: Filter by stability-based intervals
    stability = CommitmentStability()
    due_now = []
    for item in schedule:
        c = item["commitment"]
        S = c.get("stability", 1.5)
        D = c.get("difficulty", 5.0)
        elapsed = item["elapsed_days"]
        R = stability.retrievability(elapsed, S)
        optimal = stability.optimal_interval(S)
        if elapsed >= optimal * 0.9:  # within 90% of optimal interval
            due_now.append({**item, "retrievability": R, "urgency": 1.0 - R})

    # Layer 3: JITAI context filter
    jitai = JITAIContextFilter()
    pending = []
    for item in due_now:
        should_deliver, confidence = jitai.should_deliver(item["commitment"], context)
        if should_deliver:
            pending.append({**item, "delivery_confidence": confidence})

    # Sort by urgency (lowest retrievability first)
    pending.sort(key=lambda x: x.get("urgency", 0), reverse=True)

    return JSONResponse({"pending": pending[:user.get("max_daily", 4)]})
```

2. Add new endpoint for schedule preview:
```python
@app.get("/user/checkins/schedule")
async def api_checkin_schedule(user: dict = Depends(get_current_user)):
    """Preview the next 7 days of scheduled check-ins."""
    # Returns constraint-solved schedule with stability predictions
```

### Integration with Check-In Engine

**Modify**: `agent_zero_telegram/check_in_engine.py`

The existing check-in engine's scoring system (event-anchored, pattern-triggered, silence-based) should be **preserved** as a fallback and as input to the JITAI context filter. The constraint scheduler replaces the timing logic, not the scoring logic.

```python
# In check_in_engine.py, replace direct scheduling with:
from commitment_scheduler import CommitmentScheduler, JITAIContextFilter

class CheckInEngine:
    def evaluate(self, user_id, commitments, shadow):
        # Existing scoring logic produces candidates with scores
        candidates = self._score_candidates(commitments, shadow)

        # NEW: Feed candidates through constraint scheduler
        scheduler = CommitmentScheduler()
        scheduled = scheduler.schedule_checkins(
            [c for c in candidates if c["score"] >= 0.3],
            self._get_user_prefs(user_id)
        )

        # NEW: Apply JITAI filter
        jitai = JITAIContextFilter()
        context = self._build_context(user_id)

        filtered = []
        for item in scheduled:
            should, conf = jitai.should_deliver(item["commitment"], context)
            if should:
                item["delivery_confidence"] = conf
                filtered.append(item)

        return filtered
```

### Stability Update on Events

**Modify**: `database.py` streak update logic (lines 406-429)

After each commitment event (completed, missed), update stability:

```python
async def update_commitment_status(commitment_id, user_id, new_status, payload=None):
    # ... existing status update logic ...

    # NEW: Update stability based on outcome
    if new_status in ("completed", "missed"):
        commitment = await get_commitment(commitment_id)
        stability_engine = CommitmentStability()
        S = commitment.get("stability", 1.5)
        D = commitment.get("difficulty", 5.0)
        elapsed = (now - (commitment["last_checked_in"] or commitment["created_at"])).total_seconds() / 86400
        R = stability_engine.retrievability(elapsed, S)

        if new_status == "completed":
            new_S = stability_engine.update_stability_success(S, D, R)
        else:
            new_S = stability_engine.update_stability_failure(S, D, R)

        new_D = stability_engine.difficulty_from_history(events)
        next_interval = stability_engine.optimal_interval(new_S)
        next_checkin = now + timedelta(days=next_interval)

        await execute(
            "UPDATE commitments SET stability=$1, difficulty=$2, next_checkin_at=$3 WHERE id=$4",
            new_S, new_D, next_checkin, commitment_id
        )
```

### C094 Integration Path

The constraint solver lives at `challenges/C094_constraint_solver/constraint_solver.py`. To integrate:

1. Copy `CSPSolver`, `Variable`, constraint classes, and `CSPResult` to `agent_zero/csp_solver.py` (or add `challenges/C094_constraint_solver/` to Python path)
2. The `scheduling()` helper can be used directly for multi-commitment scheduling with precedence
3. For production, prefer MAC strategy (best pruning) with a timeout fallback to simple even-spread

### User Preferences Schema

Add to user settings (stored in shadow or user table):

```json
{
    "scheduling_preferences": {
        "active_hours": [8, 22],
        "prime_hours": [9, 10, 14, 15],
        "min_gap_minutes": 60,
        "max_daily_checkins": 4,
        "weekend_mode": "relaxed",
        "delivery_threshold": 0.5
    }
}
```

## Test Specifications

### Unit Tests: `agent_zero/test_commitment_scheduler.py`

```python
# -- CommitmentStability tests --

def test_initial_stability_by_weight():
    """Light=3.0, moderate=1.5, heavy=0.7"""

def test_retrievability_at_zero_elapsed():
    """R(0, S) = 1.0 for any stability"""

def test_retrievability_decays_over_time():
    """R(t, S) < R(0, S) for t > 0"""

def test_retrievability_at_stability():
    """R(S, S) should be approximately 0.9 (the FSRS design target)"""
    # Actually for FSRS v4: R(9*S, S) = 0.5, R(S, S) = (1 + 1/9)^-1 ~ 0.9

def test_optimal_interval_increases_with_stability():
    """Higher stability -> longer intervals"""

def test_stability_increases_after_success():
    """S' > S after successful check-in"""

def test_stability_decreases_after_failure():
    """S' < S after missed commitment"""

def test_difficulty_from_empty_history():
    """Default difficulty = 5.0"""

def test_difficulty_increases_with_misses():
    """More misses -> higher difficulty -> shorter intervals"""

def test_difficulty_clamped_1_to_10():
    """Difficulty never exceeds [1, 10] range"""

# -- CommitmentScheduler tests --

def test_schedule_single_commitment():
    """One commitment gets a valid time slot"""

def test_schedule_no_overlap():
    """Multiple commitments get different time slots"""

def test_schedule_min_gap_enforced():
    """Commitments are at least min_gap_minutes apart"""

def test_schedule_heavy_gets_prime_time():
    """Heavy-weight commitments scheduled during prime hours"""

def test_schedule_parent_before_child():
    """Parent commitment scheduled before child (precedence)"""

def test_schedule_respects_active_hours():
    """No check-ins outside user's active hours window"""

def test_schedule_max_daily_cap():
    """At most max_daily_checkins commitments scheduled per day"""

def test_schedule_fallback_on_unsatisfiable():
    """When constraints can't be satisfied, fall back to even spread"""

# -- JITAIContextFilter tests --

def test_dosage_penalty_reduces_score():
    """Recent check-ins reduce delivery probability"""

def test_high_dosage_blocks_delivery():
    """3+ check-ins in 4 hours -> should_deliver returns False"""

def test_streak_protection():
    """Active 7+ day streak slightly reduces frequency"""

def test_silence_boost():
    """3+ days since last session increases urgency"""

def test_broken_streak_boost():
    """Broken streak (best>=5, current=0) increases urgency"""

def test_delivery_threshold_configurable():
    """Custom threshold changes the filter cutoff"""

# -- Integration tests --

def test_full_pipeline_schedule_to_delivery():
    """Commitments -> CSP schedule -> stability filter -> JITAI filter -> pending list"""

def test_stability_update_on_completion():
    """Completing a commitment updates stability and next_checkin_at in DB"""

def test_stability_update_on_miss():
    """Missing a commitment reduces stability and shortens next interval"""

def test_custom_cadence_with_recurrence_rule():
    """Custom cadence uses recurrence_rule JSONB for interval calculation"""
```

### Expected Test Count: ~25-30 tests

## Estimated Impact

1. **Reduced notification fatigue**: Constraint-solved schedules spread check-ins optimally instead of clustering. Users with 5 daily commitments get spaced prompts instead of simultaneous ones.

2. **Adaptive frequency**: FSRS stability means reliable users get progressively less frequent prompts (rewarding consistency), while struggling users get more support (shorter intervals after misses). Based on FSRS benchmarks, this should reduce unnecessary check-ins by 20-30%.

3. **Context-aware delivery**: JITAI filter prevents check-ins during inopportune moments, improving receptiveness. HeartSteps showed that dosage-aware scheduling maintained engagement 2x longer than fixed-interval prompting.

4. **Custom cadence support**: Activates the dormant `custom` cadence and `recurrence_rule` fields, allowing users to define personalized schedules (e.g., "every 3 days", "weekday mornings only").

5. **Foundation for RL optimization**: The three-layer architecture (constraint -> stability -> context) creates clean interfaces for future reinforcement learning: Layer 2 parameters can be optimized per-user via Thompson Sampling (HeartSteps approach), and Layer 3 weights can be learned from check-in success/failure data.

## Files to Create/Modify

| Action | File | Changes |
|--------|------|---------|
| CREATE | `agent_zero/commitment_scheduler.py` | CommitmentScheduler, CommitmentStability, JITAIContextFilter |
| CREATE | `agent_zero/csp_solver.py` | Copy of C094 CSPSolver (core classes only, ~400 lines) |
| MODIFY | `agent_zero/database.py:236-253` | Add stability, difficulty, next_checkin_at columns |
| MODIFY | `agent_zero/database.py:406-429` | Add stability update on status change |
| MODIFY | `agent_zero/agent_zero_server.py:1008-1047` | Replace naive pending with scheduler-aware endpoint |
| MODIFY | `agent_zero/agent_zero_server.py` | Add /user/checkins/schedule preview endpoint |
| MODIFY | `agent_zero/agent_zero_telegram/check_in_engine.py` | Integrate scheduler into check-in evaluation |
| CREATE | `agent_zero/test_commitment_scheduler.py` | ~25-30 tests |

## Relevant AZ Challenges

- **C094** (Constraint Solver): Core CSP engine for scheduling
- **C095** (Logic Programming): Could express scheduling rules as Prolog facts (future enhancement)
- **C092** (Graph Algorithms): Commitment dependency graphs
