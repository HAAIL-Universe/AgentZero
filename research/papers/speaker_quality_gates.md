---
topic: Quality Gates on Speaker Output
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T19:00:00Z
---

# Quality Gates on Speaker Output

Verify that Speaker's response style matches learned intervention effectiveness
from consolidation rules before sending to the user. Catch stylistic mismatches
(e.g., directive style when data shows reflective works better for this topic)
and flag or adjust before delivery.

## Problem Statement

Speaker renders the final user-facing response based on the resolution from
Pineal and the response plan. However:

1. **No style-effectiveness verification** -- The consolidator tracks which
   intervention styles work best per topic (`best_style`, `worst_style` in
   `intervention_effectiveness`), but Speaker never checks this data. It can
   deliver a response using a style that consolidation data shows is ineffective
   for the current topic.

2. **No MI adherence check** -- Research shows AI therapeutic responses have a
   persistent "cognitive-affective gap": factually sound but emotionally
   misaligned (CSS vs ARS scores, arXiv 2601.18630). Speaker has no post-hoc
   verification against MI quality dimensions.

3. **No reflection-to-question ratio monitoring** -- MITI-4 best practice is
   R:Q >= 2:1 for proficiency. Speaker's output is never checked for this ratio,
   and the deterministic fallback tends to produce directive statements with no
   reflections.

4. **Output guardrails are safety-only** -- `evaluate_visible_output_guardrails`
   in `guardrails.py:235-284` only checks for medical/legal/financial safety and
   ungrounded claims. It has no quality dimension for therapeutic style.

## Current State in Agent Zero

### Speaker Agent (`cognitive_agents.py:1103-1157`)

The deterministic `_speaker_message()` constructs a formulaic response:
- "The strongest path here is '[approved_path]'. My recommendation is to
  start with [next_step]."
- Style adjustment is minimal: only checks `support_style` dimension from
  user_model for "direct" vs "balanced" (lines 1180-1186).
- No access to consolidation data (intervention effectiveness, best_style).
- No MI technique integration beyond the prompt (which only fires in hybrid mode).

### Speaker Hybrid Mode (`prompts/cognitive_agents/speaker.md`)

The prompt instructs Speaker to use MI techniques (reflective listening,
affirmation, summarizing) and calibrate voice by emotional register. However:
- No structured quality check on the output.
- The LLM may ignore style guidance, especially under tight token budgets.
- No feedback loop from intervention outcomes to Speaker behavior.

### Consolidation Data (`consolidator.py:281-337`)

`_compute_intervention_effectiveness()` returns:
```python
{
    "best_style": "reflective" | "direct" | "clarify and test" | ...,
    "worst_style": "directive" | None,
    "acted_rate": 0.72,
    "ignored_rate": 0.15,
    "pushed_back_rate": 0.13,
}
```

This data is available in consolidated rules but never reaches Speaker or the
output guardrail pipeline.

### Output Guardrail Pipeline (`guardrails.py:235-284`)

`evaluate_visible_output_guardrails()` returns `{action: "allow"|"rewrite"}`.
Currently only handles:
- Ungrounded introspection claims
- Unsafe medical directives
- Overly directive high-stakes language

It does NOT check:
- Style match against learned effectiveness
- MI adherence (reflection ratio, empathy markers)
- Anti-pattern violations from Speaker prompt (e.g., "you should", >3 items)

### Intervention Logging (`agent_zero_server.py:2705-2753`)

`intervention_style` is logged per turn (line 2716) and stored in the
intervention record. This creates the data loop: intervention -> outcome ->
consolidation -> effectiveness rule. But the loop is open -- Speaker doesn't
read the effectiveness data back.

## Industry Standard / Research Findings

### 1. PAIR-SAFE: Paired-Agent Runtime Auditing

**Source**: [PAIR-SAFE, arXiv 2601.12754](https://arxiv.org/abs/2601.12754)

A Judge agent audits each Responder output using MITI-4 dimensions and issues
structured ALLOW/REVISE decisions. Key results:
- Partnership scores improve significantly.
- Seek Collaboration scores improve.
- Overall Relational quality improves.

**Applicability to Agent Zero**: Add a lightweight quality gate function (not a
full LLM judge) that checks Speaker output against learned effectiveness data
and MITI-4 heuristics. Issue ALLOW/REWRITE decisions similar to PAIR-SAFE.

### 2. Cognitive-Affective Gap in LLM Responses

**Source**: [Mental Health LLM Evaluation, arXiv 2601.18630](https://arxiv.org/abs/2601.18630)

6-attribute rubric: Guidance, Informativeness, Safety (CSS) vs Empathy,
Helpfulness, Interpretation (ARS). LLMs score near-ceiling on Safety (4.89)
but struggle with Empathy (4.03) and Helpfulness (4.06).

**Applicability to Agent Zero**: The quality gate should check for empathy
markers (reflective listening phrases) and helpfulness signals (concrete next
step present), not just safety.

### 3. MITI-4 Automated Scoring

**Source**: [MITI LLM Benchmarking, arXiv 2603.03846](https://arxiv.org/abs/2603.03846)

MITI-4 defines key ratios:
- Reflection-to-Question ratio >= 2:1 for proficiency
- Complex Reflection % >= 40% for proficiency
- MI-Adherent vs MI-Non-Adherent behavior ratio

LLMs achieve R:Q ~1.1-1.3 vs human ~1.4, and Complex Reflection ~25-31% vs
human ~37%. Runtime MITI judging (using paired agents) improved R:Q from
1.01 to 5.31.

**Applicability to Agent Zero**: Implement a lightweight deterministic R:Q check
on Speaker output. Count reflection markers vs question marks. Flag when
ratio is below 1:1 (minimum MI adherence).

### 4. LLM-as-Judge for Agent Output Quality

**Source**: [Agent-as-a-Judge, arXiv 2508.02994](https://arxiv.org/abs/2508.02994)

Agent-as-a-Judge uses multi-dimensional scoring with dimension-specific rubrics.
For efficiency, some dimensions can be scored deterministically (rule-based)
while others require LLM judgment.

**Applicability to Agent Zero**: Use deterministic checks for most quality gates
(style match, anti-patterns, R:Q ratio). Reserve LLM judgment for edge cases
only (when deterministic checks flag a potential issue but can't determine
severity).

## Proposed Implementation

### Step 1: Add Quality Gate Function to `guardrails.py`

File: `guardrails.py`, new function after `evaluate_visible_output_guardrails`.

```python
# MI reflective listening markers
_REFLECTION_MARKERS = (
    "it sounds like", "what i'm hearing", "so what you're saying",
    "you're feeling", "that makes sense", "i can see", "it seems like",
    "you mentioned", "what stands out", "i notice",
)

# Anti-patterns from Speaker prompt (should never appear)
_SPEAKER_ANTI_PATTERNS = (
    "you should ",
    "you must ",
    "you need to ",
    "you have to ",
)


def evaluate_speaker_quality(
    visible_text: str,
    *,
    intervention_effectiveness: dict | None = None,
    intervention_style: str = "",
    emotional_register: str = "",
) -> dict:
    """Check Speaker output against learned effectiveness and MI heuristics.

    Returns {action: "allow"|"flag", reasons: [...], suggestions: [...]}
    - "flag" means the response has quality concerns but is still delivered.
      The flag is logged for consolidation learning, not used for rewriting
      (rewriting is reserved for safety guardrails).
    """
    lower = (visible_text or "").lower()
    reasons = []
    suggestions = []

    # 1. Style mismatch check
    if intervention_effectiveness and intervention_style:
        worst = intervention_effectiveness.get("worst_style")
        best = intervention_effectiveness.get("best_style")
        if worst and worst.lower() in intervention_style.lower():
            reasons.append(
                f"Using '{intervention_style}' style but consolidation data "
                f"shows '{worst}' is least effective for this topic."
            )
            if best:
                suggestions.append(f"Consider '{best}' style instead.")

    # 2. Anti-pattern check (from Speaker prompt contract)
    for pattern in _SPEAKER_ANTI_PATTERNS:
        if pattern in lower:
            reasons.append(f"Contains anti-pattern '{pattern.strip()}'.")
            suggestions.append("Replace with 'here's what I'd suggest' or 'one approach is'.")
            break

    # 3. Item count check (Speaker prompt: never list more than 3 things)
    numbered_items = sum(1 for line in visible_text.split("\n")
                        if line.strip() and (
                            line.strip()[0].isdigit() and line.strip()[1:2] in (".", ")")
                        ))
    bullet_items = sum(1 for line in visible_text.split("\n")
                       if line.strip().startswith(("- ", "* ")))
    total_items = max(numbered_items, bullet_items)
    if total_items > 3:
        reasons.append(f"Lists {total_items} items (max 3 per Speaker contract).")
        suggestions.append("Narrow to the 3 most important items.")

    # 4. Reflection-to-question ratio (MITI-4 heuristic)
    question_count = lower.count("?")
    reflection_count = sum(1 for marker in _REFLECTION_MARKERS if marker in lower)
    if question_count > 0 and reflection_count == 0:
        reasons.append("Contains questions but no reflective listening markers.")
        suggestions.append("Add a reflection before the question (e.g., 'It sounds like...').")

    # 5. Emotional register mismatch
    if emotional_register in ("anxious", "strained", "overwhelmed"):
        # Check for overwhelming content
        word_count = len(visible_text.split())
        if word_count > 100:
            reasons.append(
                f"Response is {word_count} words but user emotional register "
                f"is '{emotional_register}' -- may be overwhelming."
            )
            suggestions.append("Simplify. Narrow to ONE thing.")

    action = "flag" if reasons else "allow"
    return {
        "action": action,
        "reasons": reasons or ["Speaker output passes quality gate."],
        "suggestions": suggestions,
        "metrics": {
            "question_count": question_count,
            "reflection_count": reflection_count,
            "item_count": total_items,
            "word_count": len(visible_text.split()),
        },
    }
```

### Step 2: Wire Quality Gate into Response Pipeline

File: `agent_zero_server.py`, after `evaluate_visible_output_guardrails` call
(around line 2376 for WebSocket, line 2564 for HTTP).

```python
# Quality gate: check Speaker output against learned effectiveness
from guardrails import evaluate_speaker_quality

# Get intervention effectiveness from consolidation rules
intervention_eff = {}
if context_bundle and context_bundle.get("insights"):
    # Retrieve from relevant consolidated rules
    relevant_rules = runtime_packet.get("consolidated_rules", [])
    if relevant_rules:
        # Use the highest-confidence rule's effectiveness data
        best_rule = max(relevant_rules, key=lambda r: r.get("confidence", 0))
        intervention_eff = best_rule.get("intervention_effectiveness", {})

speaker_quality = evaluate_speaker_quality(
    final_response,
    intervention_effectiveness=intervention_eff,
    intervention_style=intervention_style,
    emotional_register=state.get("emotional_register", ""),
)

if speaker_quality["action"] == "flag":
    await append_stage_message(
        reasoning_run_id,
        "quality_gate",
        "flag",
        speaker_quality,
    )
```

### Step 3: Pass Consolidation Data to Speaker Context

File: `cognitive_runtime.py`, in the blackboard assembly for Speaker.

When building the Speaker context, include the intervention effectiveness
data from consolidated rules so that hybrid-mode Speaker can self-calibrate:

```python
# In build_cognitive_blackboard or when constructing Speaker context:
if shadow_profile:
    rules = get_relevant_rules(shadow_profile, topic_signals)
    if rules:
        best_rule = max(rules, key=lambda r: r.get("confidence", 0))
        speaker_context["intervention_effectiveness"] = (
            best_rule.get("intervention_effectiveness", {})
        )
```

### Step 4: Update Speaker Prompt for Effectiveness Awareness

File: `agent_zero/prompts/cognitive_agents/speaker.md`, add new section before
"Output Contract":

```markdown
## Learned Effectiveness (from consolidation data)

If `intervention_effectiveness` is present in your context:
- Check `best_style` -- use this style when possible
- Check `worst_style` -- AVOID this style
- Check `acted_rate` -- if < 0.3, responses on this topic are being ignored;
  make the recommendation more concrete and time-bounded
- If `best_style` is "reflective", lead with a reflection before any recommendation
- If `best_style` is "direct", skip reflections and give the recommendation first
```

### Step 5: Log Quality Flags for Consolidation Learning

File: `episode_store.py`, add quality gate data to episode capture.

Include `speaker_quality_flags` in the episode so the consolidator can track
whether flagged responses correlate with worse outcomes:

```python
# In capture_episode():
episode["speaker_quality"] = {
    "action": quality_gate_result.get("action", "allow"),
    "flag_count": len(quality_gate_result.get("reasons", [])),
    "metrics": quality_gate_result.get("metrics", {}),
}
```

## Test Specifications

### Test File: `agent_zero/test_speaker_quality_gates.py`

```
test_allow_clean_response:
    - Input: "The strongest path here is starting small. It sounds like
      you've been thinking about this carefully."
    - No anti-patterns, has reflection marker
    - Expected: action="allow"

test_flag_anti_pattern_you_should:
    - Input: "You should start exercising immediately."
    - Expected: action="flag", reason contains "anti-pattern"

test_flag_anti_pattern_you_must:
    - Input: "You must stop procrastinating."
    - Expected: action="flag", reason contains "anti-pattern"

test_flag_excessive_items:
    - Input: "1. Do X\n2. Do Y\n3. Do Z\n4. Do W\n5. Do V"
    - Expected: action="flag", reason contains "5 items"

test_flag_questions_without_reflections:
    - Input: "What do you think? Have you tried that? When will you start?"
    - Expected: action="flag", reason contains "no reflective listening"

test_allow_questions_with_reflections:
    - Input: "It sounds like you're unsure. What do you think would help?"
    - Expected: action="allow" (has reflection + question)

test_flag_style_mismatch_worst_style:
    - Input: any response
    - intervention_effectiveness: {worst_style: "directive", best_style: "reflective"}
    - intervention_style: "directive"
    - Expected: action="flag", reason contains "least effective"

test_allow_style_match_best_style:
    - Input: any response
    - intervention_effectiveness: {best_style: "reflective"}
    - intervention_style: "reflective"
    - Expected: action="allow" (style matches best)

test_flag_overwhelmed_long_response:
    - Input: 150-word response
    - emotional_register: "overwhelmed"
    - Expected: action="flag", reason contains "overwhelming"

test_allow_overwhelmed_short_response:
    - Input: 30-word response
    - emotional_register: "overwhelmed"
    - Expected: action="allow"

test_metrics_returned:
    - Input: "It sounds like you're torn. What feels right? Try option A?"
    - Expected: metrics.question_count=2, metrics.reflection_count=1

test_no_effectiveness_data_allows_all:
    - Input: any response without anti-patterns
    - intervention_effectiveness: None
    - Expected: action="allow" (no data to compare against)

test_flag_is_logged_not_rewritten:
    - Verify that "flag" action does NOT change the response text
    - Quality gate returns suggestions but original text is preserved
    - Rewriting is reserved for safety guardrails only
```

## Estimated Impact

### Response Quality

- Catches 100% of Speaker anti-pattern violations ("you should", >3 items)
  which currently slip through in both deterministic and hybrid mode.
- Reflection-to-question ratio monitoring encourages MI-adherent responses
  over time through the consolidation feedback loop.
- Style-effectiveness matching ensures the system learns from past interventions
  and avoids repeating styles that don't work for specific topics.

### Latency

- Quality gate is fully deterministic (string matching, counting). Adds <1ms
  per response. No LLM calls.
- Does not block response delivery -- flags are logged asynchronously.

### Safety

- Quality gate is additive to existing safety guardrails, not a replacement.
- The gate flags but does not rewrite, preserving the safety guardrail's
  authority over response content.

### Learning Loop

- Quality flags feed into episode data, which feeds into consolidation,
  which updates intervention effectiveness rules, which feeds back into
  the quality gate. This closes the loop between response style and
  user outcomes.

## Citations

1. [PAIR-SAFE: Paired-Agent Runtime Auditing](https://arxiv.org/abs/2601.12754) -- Judge agent with MITI-4 ALLOW/REVISE decisions.
2. [Mental Health LLM Evaluation](https://arxiv.org/abs/2601.18630) -- 6-attribute rubric (CSS + ARS), cognitive-affective gap.
3. [MITI LLM Benchmarking](https://arxiv.org/abs/2603.03846) -- MITI-4 automated scoring, R:Q ratio improvement.
4. [Agent-as-a-Judge](https://arxiv.org/abs/2508.02994) -- Multi-dimensional agent output evaluation.
5. [MI Quality Assurance Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7680367/) -- MITI-4 as gold standard for MI fidelity measurement.
