---
topic: Explicit Agree/Disagree Deliberation Protocol
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T22:00:00Z
---

# Explicit Agree/Disagree Deliberation Protocol

## Problem Statement

Agent Zero's deliberation protocol fires when agents disagree (magnitude >= 0.30), but the
current implementation has critical gaps:

1. **Binary revision logic**: Fello concedes only when opposing confidence exceeds its own by
   0.15; otherwise it maintains. Othello checks if Fello conceded by 0.10 and adjusts by
   +/-0.05. These are crude binary decisions, not genuine deliberation.

2. **No justification structure**: Agents produce revised `content` strings but no structured
   `justification` field explaining WHY they revised or maintained. Pineal receives content
   truncated to 300 chars with no reasoning chain.

3. **No explicit agree/disagree signal**: The transcript tracks only `converged` (boolean)
   and `pre/post magnitude`. There's no per-agent "I agree with X on Y, but disagree on Z"
   structure. Pineal cannot distinguish partial agreement from total disagreement.

4. **Shadow is a non-participant**: Shadow doesn't revise confidence or engage with the
   challenge. It only escalates watch signals. This wastes historical data that could
   ground the debate.

5. **No argument validity weighting**: All agent reactions are treated equally. Research shows
   that weighting updates by argument validity (not just confidence) maximizes improvement
   (DCI framework, ICLR Blogposts 2025).

6. **Conformity pressure**: The current protocol resembles the failure mode identified in
   "Can LLM Agents Really Debate?" (arXiv 2511.07784) where majority pressure suppresses
   independent correction.

## Current State in Agent Zero

### Disagreement Detection
- **File**: `agent_zero/cognitive_runtime.py`, lines 396-498
- **Function**: `compute_agent_disagreement(blackboard)`
- **Threshold**: `DELIBERATION_THRESHOLD = 0.30` (line 17)
- **Components**: confidence_delta + directional_disagreement + shadow_tension
- **Output**: `{has_disagreement, magnitude, between, topic, fello_position, othello_position, shadow_position}`

### Challenge Construction
- **File**: `agent_zero/cognitive_runtime.py`, lines 525-549
- **Function**: `_build_challenge_question(disagreement)`
- Constructs natural language: "Fello says X, Othello says Y. Do you revise or maintain?"
- No structured argument format; no criteria for evaluation

### Reaction Context
- **File**: `agent_zero/cognitive_runtime.py`, lines 552-577
- **Function**: `_build_reaction_context(agent_id, between, agent_outputs, challenge_question, disagreement)`
- Passes: mode, challenge_question, own position (200 chars), own confidence, opposing positions (200 chars + confidence), topic, magnitude
- No argument validity criteria, no scoring rubric

### Fello Reaction
- **File**: `agent_zero/cognitive_agents.py`, lines 480-522
- **Function**: `_fello_reaction(context, challenge)`
- Logic: if max_opposing_conf > own_conf + 0.15 -> average confidences, else maintain
- No justification of WHY it's revising/maintaining beyond template text

### Othello Reaction
- **File**: `agent_zero/cognitive_agents.py`, lines 577-617
- **Function**: `_othello_reaction(context, challenge)`
- Logic: if fello_conf < own_conf - 0.10 -> relax by 0.05, else strengthen by 0.05
- No justification of WHY the risk assessment changed

### Shadow Reaction
- **File**: `agent_zero/cognitive_agents.py`, lines 839-884 (per explore agent)
- Does not revise confidence. Only escalates/maintains watch signals.
- Checks if other agents mentioned "pattern" or "history"

### Deliberation Transcript
- **File**: `agent_zero/cognitive_runtime.py`, lines 686-693
- Structure: `{challenge_question, reactions: {aid: content[:300]}, pre_magnitude, post_magnitude, converged, participants}`
- No structured justification, no per-agent agree/disagree signals, no argument scoring

### Agent Output Schemas
- **File**: `agent_zero/agent_schemas.py`, lines 10-102
- No `justification`, `agree_with`, `disagree_with`, or `revision_reason` fields on any schema

## Industry Standard / Research Findings

### 1. DCI Framework -- Typed Epistemic Acts (2026)
**Source**: [From Debate to Deliberation: Structured Collective Reasoning with Typed Epistemic Acts](https://arxiv.org/html/2603.11781)

DCI defines deliberation as a phased process with:
- **14 typed epistemic acts** organized in 6 families (orienting, generative, critical, integrative, epistemic, decisional)
- **Shared workspace** with 6 sections including explicit **Tensions** (disagreements as first-class objects)
- **Decision packets** at closure: selected option + residual objections + minority report + reopen triggers
- **Delegate differentiation**: Framer, Explorer, Challenger, Integrator archetypes

Key finding: On hidden-profile tasks (where different agents have unique information), DCI significantly beats single-agent generation. This maps directly to Agent Zero where Fello, Othello, and Shadow each have unique analytical perspectives.

**Applicable pattern**: Explicit tension tracking and minority reports. Pineal should receive not just "converged/held" but structured agree/disagree with reasons from each agent.

### 2. ConfMAD -- Confidence-Modulated Debate (2026)
**Source**: [Demystifying Multi-Agent Debate: The Role of Confidence and Diversity](https://arxiv.org/html/2601.19921)

Key findings:
- Calibrated confidence paired with conditional updates creates a **submartingale** (expected belief strictly improves over rounds)
- Agents should **condition updates on others' confidence**, not just their content
- Diversity-aware initialization improves debate by ensuring the correct hypothesis is present
- On 6 benchmarks, ConfMAD improved accuracy 2-10 percentage points over baseline

**Applicable pattern**: Agent Zero already has calibrated confidence via consolidation-based self-calibration. But agents don't currently condition their updates on the opposing agent's calibration history. Adding the agent_weights data to the reaction context would enable this.

### 3. "Can LLM Agents Really Debate?" (2025)
**Source**: [arXiv 2511.07784](https://arxiv.org/abs/2511.07784)

Key findings:
- **Majority pressure suppresses independent correction** -- the dominant failure mode
- **Group diversity and intrinsic reasoning strength** are the primary drivers of debate success
- **Confidence visibility has limited benefit** and can induce over-confidence cascades
- **Effective teams overturn incorrect consensus** through rational, validity-aligned reasoning

**Applicable pattern**: Agent Zero should NOT show raw confidence numbers to agents during deliberation (risk of anchoring). Instead, share the REASONING and let agents evaluate argument validity.

### 4. DMAD -- Diverse Multi-Agent Debate (ICLR 2025)
**Source**: [Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate](https://github.com/MraDonkey/DMAD)

Key finding: Assigning **distinct reasoning strategies** to different agents breaks "fixed mental set" where uniform approaches consistently fail. DMAD outperforms standard MAD in fewer rounds.

**Applicable pattern**: Agent Zero already has distinct agent roles (optionality, risk, behavioral, sequencing). This is a strength. The gap is in how they EXCHANGE and EVALUATE each other's reasoning, not in diversity itself.

### 5. Lazy Agent Problem (2025)
**Source**: [Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation](https://arxiv.org/abs/2511.02303)

Key finding: In multi-agent debate, one agent often dominates while others contribute little ("lazy agent" problem). Solution: verifiable reward mechanisms and causal influence measurement.

**Applicable pattern**: Shadow's current non-participation in revision IS the lazy agent problem. Shadow should actively engage with the debate using historical evidence, not just passively escalate.

### 6. ICLR Blogposts MAD Analysis (2025)
**Source**: [Multi-LLM-Agents Debate -- Performance, Efficiency, and Scaling Challenges](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-mad-159/blog/mad/)

Key finding: Current MAD frameworks fail because agents evaluate "full responses" rather than analyzing specific reasoning gaps. Agents assign weight to final answers instead of reasoning steps.

**Applicable pattern**: Agent Zero's challenge question asks "do you revise or maintain?" when it should ask "what specific reasoning in the opposing position do you find valid or invalid?"

## Proposed Implementation

### Phase 1: Structured Agree/Disagree Fields (agent_schemas.py + cognitive_agents.py)

**1a. Add deliberation fields to Fello and Othello schemas**

In `agent_zero/agent_schemas.py`, add optional deliberation fields to fello and othello schemas:

```python
# Add to fello schema (after "confidence"):
"agree_with": list,        # ["othello: risk X is real", "shadow: pattern Y is relevant"]
"disagree_with": list,     # ["othello: severity is overstated because..."]
"revision_reason": str,    # "Revised because Othello's cascade-failure argument is sound"
# (or "Maintained because the risk is bounded by exit criteria")

# Same fields for othello schema
```

These fields are OPTIONAL -- only populated during reaction mode. The validate_agent_output
function already handles missing optional fields (they just won't be in the payload).

**Implementation approach**: Don't change validate_agent_output logic. Instead, add an
`"optional"` key to schemas alongside `"required"`:

```python
"fello": {
    "required": { ... existing ... },
    "optional": {
        "agree_with": list,
        "disagree_with": list,
        "revision_reason": str,
    }
}
```

**1b. Rewrite `_fello_reaction` (cognitive_agents.py, lines 480-522)**

Replace the binary if/else with multi-factor evaluation:

```python
def _fello_reaction(context: dict, challenge: dict) -> dict:
    own_position = challenge.get("your_round1_position", "")
    own_confidence = challenge.get("your_round1_confidence") or 0.5
    opposing = challenge.get("opposing_positions", {})
    topic = challenge.get("disagreement_topic", "")
    agent_weights = challenge.get("agent_weights", {})

    agree_with = []
    disagree_with = []

    # Evaluate each opposing position on argument merit, not just confidence
    for aid, pos in opposing.items():
        opp_content = pos.get("content", "")
        opp_conf = float(pos.get("confidence") or 0.5)
        opp_weight = agent_weights.get(aid, 0.5)

        # Check for specific risk mentions that Fello should acknowledge
        risk_keywords = ["irreversible", "cascade", "overcommit", "relapse", "pattern"]
        has_substantive_risk = any(kw in opp_content.lower() for kw in risk_keywords)

        if has_substantive_risk:
            agree_with.append(f"{aid}: identified real constraint -- factor into experiment design")
        if opp_conf > 0.7 and opp_weight > 0.6:
            agree_with.append(f"{aid}: high historical accuracy on this topic (weight {opp_weight:.0%})")

        # Fello disagrees when risk is generic / not tied to specific failure mode
        if not has_substantive_risk and opp_conf < own_confidence:
            disagree_with.append(f"{aid}: risk assessment is generic, not tied to specific failure mode")

    # Revision logic: agree count vs disagree count, weighted by agent accuracy
    total_agree_weight = len(agree_with)
    total_disagree_weight = len(disagree_with)

    if total_agree_weight > total_disagree_weight:
        # Genuine concession: moderate confidence based on strength of agreement
        shift = min(0.15, 0.05 * total_agree_weight)
        revised_confidence = round(max(own_confidence - shift, 0.25), 2)
        revision_reason = (
            f"Revised downward because {total_agree_weight} substantive point(s) "
            f"in opposing position address real constraints. "
            f"Adjusted experiment scope to account for these."
        )
        # Build content incorporating specific concessions
        content = (
            f"On {topic}: I acknowledge the specific constraints raised. "
            f"My revised recommendation narrows the experiment to reduce exposure "
            f"to the identified risks while preserving the core signal. "
            f"Confidence adjusted to {revised_confidence:.0%}."
        )
    else:
        revised_confidence = own_confidence
        revision_reason = (
            f"Maintained because opposing arguments are {'generic risk flags' if disagree_with else 'lower confidence'} "
            f"that don't outweigh the information value of the proposed experiment."
        )
        content = (
            f"I maintain my position on {topic}. {own_position[:120]} "
            f"The concerns raised are noted but don't change the optionality calculus."
        )

    return {
        "sender": "FELLO",
        "target": "pineal",
        "message_type": "proposal",
        "content": content,
        "tags": ["optionality", "upside", "deliberation"],
        "alternative_paths": [],
        "upside_opportunities": [],
        "low_cost_experiments": [],
        "confidence": revised_confidence,
        "agree_with": agree_with,
        "disagree_with": disagree_with,
        "revision_reason": revision_reason,
    }
```

**1c. Rewrite `_othello_reaction` (cognitive_agents.py, lines 577-617)**

Same pattern -- evaluate Fello's specific claims, not just confidence delta:

```python
def _othello_reaction(context: dict, challenge: dict) -> dict:
    own_position = challenge.get("your_round1_position", "")
    own_confidence = challenge.get("your_round1_confidence") or 0.5
    opposing = challenge.get("opposing_positions", {})
    topic = challenge.get("disagreement_topic", "")
    agent_weights = challenge.get("agent_weights", {})

    agree_with = []
    disagree_with = []

    fello_data = opposing.get("fello", {})
    fello_content = fello_data.get("content", "")
    fello_conf = float(fello_data.get("confidence") or 0.5)
    fello_weight = agent_weights.get("fello", 0.5)

    # Check if Fello addressed specific risk with concrete mitigation
    mitigation_keywords = ["exit criteria", "bounded", "reversible", "probe", "7 days", "test"]
    has_concrete_mitigation = any(kw in fello_content.lower() for kw in mitigation_keywords)

    if has_concrete_mitigation:
        agree_with.append("fello: proposed concrete mitigation that bounds the downside")
    if fello_conf < own_confidence - 0.1:
        agree_with.append("fello: conceded on confidence, showing genuine consideration of risk")

    # Othello disagrees when optimism is unbounded
    if not has_concrete_mitigation and fello_conf > 0.6:
        disagree_with.append("fello: maintains high confidence without concrete exit criteria")

    # Shadow data check
    shadow_data = opposing.get("shadow", {})
    if shadow_data:
        shadow_content = shadow_data.get("content", "")
        if any(kw in shadow_content.lower() for kw in ["pattern", "history", "previous"]):
            agree_with.append("shadow: historical data supports caution")

    if agree_with and len(agree_with) > len(disagree_with):
        revised_confidence = round(own_confidence - 0.05, 2)
        revision_reason = (
            f"Relaxed slightly because opposing position includes concrete "
            f"mitigation ({len(agree_with)} point(s) acknowledged)."
        )
        content = (
            f"The specific mitigations proposed on {topic} partially address my concerns. "
            f"A bounded probe with the stated exit criteria is acceptable. "
            f"Core risk remains -- monitor closely."
        )
    else:
        revised_confidence = min(round(own_confidence + 0.05, 2), 0.95)
        revision_reason = (
            f"Strengthened because opposing position lacks concrete mitigation "
            f"for the identified risks."
        )
        content = (
            f"I maintain my caution on {topic}. {own_position[:120]} "
            f"The optimistic position has not provided concrete exit criteria. "
            f"Recommend adding a hard stop-loss condition before proceeding."
        )

    return {
        "sender": "Othello",
        "target": "pineal",
        "message_type": "risk_flag",
        "content": content,
        "tags": ["risk", "safety", "deliberation"],
        "risks": [f"Unresolved tension on {topic}"],
        "blocked_moves": [],
        "conditions_for_safe_proceeding": ["Set hard exit criteria", "Monitor for early warnings"],
        "reversibility_requirements": [],
        "confidence": revised_confidence,
        "agree_with": agree_with,
        "disagree_with": disagree_with,
        "revision_reason": revision_reason,
    }
```

### Phase 2: Shadow Active Participation (cognitive_agents.py)

Upgrade Shadow's reaction to provide grounded historical judgment:

```python
def _shadow_reaction(context: dict, challenge: dict) -> dict:
    # ... existing setup ...
    opposing = challenge.get("opposing_positions", {})
    topic = challenge.get("disagreement_topic", "")

    agree_with = []
    disagree_with = []

    # Shadow evaluates: does the optimistic position match historical patterns?
    fello_data = opposing.get("fello", {})
    fello_content = fello_data.get("content", "")

    # Use commitment_prediction from round 1 as ground truth
    own_prediction = challenge.get("your_round1_confidence") or 0.5  # commitment_prediction

    if own_prediction < 0.4:
        # Historical data predicts low follow-through
        if "bounded" in fello_content.lower() or "small" in fello_content.lower():
            agree_with.append("fello: scoped experiment matches what user can realistically sustain")
        else:
            disagree_with.append("fello: proposed commitment exceeds historical follow-through capacity")

    # Shadow now produces a revision_reason tied to data
    revision_reason = (
        f"Based on {len(context.get('consolidated_insights', []))} consolidated episodes: "
        f"commitment prediction is {own_prediction:.0%}. "
        f"{'Data supports bounded approach.' if agree_with else 'Data suggests scope is too large.'}"
    )

    # ... return with agree_with, disagree_with, revision_reason ...
```

### Phase 3: Enriched Reaction Context (cognitive_runtime.py)

**3a. Add agent_weights to reaction context**

In `_build_reaction_context` (line 552-577), add:

```python
# After existing context building, add:
agent_weights = blackboard.read("agent_weights", {})
context["agent_weights"] = {
    aid: agent_weights.get(aid, 0.5)
    for aid in between
}
```

This lets agents know the historical accuracy of their opponents -- enabling the ConfMAD
pattern of conditioning updates on calibrated trust.

**3b. Hide raw opposing confidence, expose reasoning instead**

Per arXiv 2511.07784 finding that confidence visibility induces anchoring, change the
opposing_positions structure to emphasize content over confidence:

```python
opposing = {
    aid: {
        "position": (agent_outputs.get(aid, {}).get("content") or "")[:300],  # more chars
        "key_claims": _extract_key_claims(agent_outputs.get(aid, {})),  # new helper
        "historical_accuracy": agent_weights.get(aid, 0.5),
        # Note: raw confidence deliberately omitted to prevent anchoring
    }
    for aid in between
    if aid != agent_id
}
```

Add helper function:

```python
def _extract_key_claims(agent_output: dict) -> list[str]:
    """Extract the main claims from an agent output for deliberation."""
    claims = []
    # Fello claims
    for path in agent_output.get("alternative_paths", []):
        claims.append(f"proposes: {path[:100]}")
    for exp in agent_output.get("low_cost_experiments", []):
        claims.append(f"experiment: {exp[:100]}")
    # Othello claims
    for risk in agent_output.get("risks", []):
        claims.append(f"risk: {risk[:100]}")
    for cond in agent_output.get("conditions_for_safe_proceeding", []):
        claims.append(f"condition: {cond[:100]}")
    # Shadow claims
    for pattern in agent_output.get("pattern_matches", []):
        claims.append(f"pattern: {str(pattern)[:100]}")
    return claims[:5]  # cap at 5 claims
```

### Phase 4: Enriched Deliberation Transcript (cognitive_runtime.py)

Replace the current transcript structure (lines 686-693) with:

```python
transcript = {
    "challenge_question": challenge_question,
    "reactions": {},
    "agreements": [],       # NEW: cross-agent agreements
    "disagreements": [],    # NEW: cross-agent disagreements
    "revision_reasons": {}, # NEW: per-agent reason for revision/maintenance
    "pre_magnitude": pre_magnitude,
    "post_magnitude": post_magnitude,
    "converged": converged,
    "participants": between,
}

for aid, r in reactions.items():
    transcript["reactions"][aid] = (r.get("content") or "")[:300]

    # Collect structured agree/disagree signals
    for agreement in r.get("agree_with", []):
        transcript["agreements"].append(f"{aid} agrees: {agreement}")
    for disagreement in r.get("disagree_with", []):
        transcript["disagreements"].append(f"{aid} disagrees: {disagreement}")

    reason = r.get("revision_reason", "")
    if reason:
        transcript["revision_reasons"][aid] = reason
```

### Phase 5: Pineal Consumes Structured Deliberation (pineal prompt + cognitive_agents.py)

Update `agent_zero/prompts/cognitive_agents/pineal.md` step 4 to reference the new fields:

```markdown
4. **Handle disagreement**: If agent_disagreement.magnitude > 0.3:
   - Read deliberation_transcript.agreements -- these are resolved points
   - Read deliberation_transcript.disagreements -- these are unresolved tensions
   - Read deliberation_transcript.revision_reasons -- understand WHY each agent moved
   - If more agreements than disagreements: trust revised positions, mode "proceed"
   - If disagreements dominate and no revision_reasons show genuine engagement:
     mode "narrow" (the debate failed to produce resolution)
   - Include residual disagreements in rationale_summary (minority report)
```

Also update `_compute_pineal_confidence` to factor in agreement/disagreement ratios:

```python
# In _compute_pineal_confidence, replace the simple "converged" check:
transcript = context.get("deliberation_transcript", {})
agreements = transcript.get("agreements", [])
disagreements = transcript.get("disagreements", [])
if agreements and len(agreements) > len(disagreements):
    base += 0.12  # strong convergence signal
elif disagreements and len(disagreements) > len(agreements):
    base -= 0.10  # persistent tension
```

### Phase 6: UI Thought Bubble Enhancement

The deliberation thought bubbles emitted via `on_step` (lines 658-670) should include
the new structured data. In the `_extract_agent_details` function, when in deliberation
stage, include:

```python
if output.get("agree_with"):
    details["agree_with"] = output["agree_with"]
if output.get("disagree_with"):
    details["disagree_with"] = output["disagree_with"]
if output.get("revision_reason"):
    details["revision_reason"] = output["revision_reason"]
```

The frontend `ThoughtBubbles.tsx` can then render these as:
- Green chip: "Agrees: [point]"
- Red chip: "Disagrees: [point]"
- Italicized text: revision reason

## Test Specifications

### Test 1: Schema validation accepts optional deliberation fields
```python
def test_fello_schema_accepts_agree_disagree_fields():
    """Optional deliberation fields should not cause validation errors."""
    payload = {
        "sender": "FELLO", "target": "pineal", "message_type": "proposal",
        "content": "test", "tags": [], "alternative_paths": [],
        "upside_opportunities": [], "low_cost_experiments": [], "confidence": 0.7,
        "agree_with": ["othello: valid risk"], "disagree_with": [],
        "revision_reason": "Revised due to valid risk"
    }
    ok, errors = validate_agent_output("fello", payload)
    assert ok
```

### Test 2: Fello concedes on substantive risk
```python
def test_fello_concedes_on_substantive_risk():
    """Fello should concede when opposing position contains specific risk keywords."""
    context = {"state": {}, "decision_brief": {"fallback_path": "probe"}}
    challenge = {
        "your_round1_position": "High optionality",
        "your_round1_confidence": 0.7,
        "opposing_positions": {
            "othello": {"content": "Irreversible cascade failure risk", "confidence": 0.8}
        },
        "disagreement_topic": "career change",
        "agent_weights": {"othello": 0.75},
    }
    result = _fello_reaction(context, challenge)
    assert result["confidence"] < 0.7  # conceded
    assert len(result["agree_with"]) > 0
    assert "revision_reason" in result
    assert "revised" in result["revision_reason"].lower() or "substantive" in result["revision_reason"].lower()
```

### Test 3: Fello maintains on generic risk
```python
def test_fello_maintains_on_generic_risk():
    """Fello should maintain when opposing risk is generic."""
    context = {"state": {}, "decision_brief": {}}
    challenge = {
        "your_round1_position": "High optionality",
        "your_round1_confidence": 0.7,
        "opposing_positions": {
            "othello": {"content": "Something might go wrong", "confidence": 0.5}
        },
        "disagreement_topic": "experiment",
        "agent_weights": {"othello": 0.4},
    }
    result = _fello_reaction(context, challenge)
    assert result["confidence"] == 0.7  # maintained
    assert len(result["disagree_with"]) > 0
    assert "maintained" in result["revision_reason"].lower()
```

### Test 4: Othello relaxes on concrete mitigation
```python
def test_othello_relaxes_on_concrete_mitigation():
    """Othello should relax when Fello proposes bounded experiment with exit criteria."""
    context = {"state": {}}
    challenge = {
        "your_round1_position": "High risk",
        "your_round1_confidence": 0.8,
        "opposing_positions": {
            "fello": {"content": "Bounded probe with exit criteria within 7 days", "confidence": 0.6}
        },
        "disagreement_topic": "commitment",
        "agent_weights": {"fello": 0.6},
    }
    result = _othello_reaction(context, challenge)
    assert result["confidence"] < 0.8
    assert any("mitigation" in a or "exit criteria" in a or "bounded" in a for a in result["agree_with"])
```

### Test 5: Othello strengthens on unbounded optimism
```python
def test_othello_strengthens_on_unbounded_optimism():
    """Othello should strengthen when Fello lacks concrete mitigation."""
    context = {"state": {}}
    challenge = {
        "your_round1_position": "High risk",
        "your_round1_confidence": 0.75,
        "opposing_positions": {
            "fello": {"content": "Great opportunity, lots of upside potential", "confidence": 0.8}
        },
        "disagreement_topic": "investment",
        "agent_weights": {"fello": 0.5},
    }
    result = _othello_reaction(context, challenge)
    assert result["confidence"] >= 0.75
    assert len(result["disagree_with"]) > 0
```

### Test 6: Enriched transcript contains agreements and disagreements
```python
def test_deliberation_transcript_has_structured_signals():
    """Transcript should contain agreements, disagreements, and revision reasons."""
    # Mock a blackboard with agent outputs that have agree/disagree fields
    # Run run_deliberation_round
    # Assert transcript has "agreements", "disagreements", "revision_reasons" keys
    # Assert they contain entries from the reacting agents
```

### Test 7: Pineal confidence adjusts based on agreement ratio
```python
def test_pineal_confidence_from_agreement_ratio():
    """Pineal confidence should increase when agreements outnumber disagreements."""
    context_converged = {
        "deliberation_transcript": {
            "agreements": ["fello agrees: X", "othello agrees: Y"],
            "disagreements": [],
            "converged": True,
        }
    }
    conf_converged = _compute_pineal_confidence(context_converged)

    context_held = {
        "deliberation_transcript": {
            "agreements": [],
            "disagreements": ["fello disagrees: X", "othello disagrees: Y"],
            "converged": False,
        }
    }
    conf_held = _compute_pineal_confidence(context_held)
    assert conf_converged > conf_held
```

### Test 8: Agent weights flow into reaction context
```python
def test_reaction_context_includes_agent_weights():
    """Reaction context should include historical agent weights."""
    # Build a blackboard with agent_weights = {"fello": 0.8, "othello": 0.6}
    # Call _build_reaction_context
    # Assert "agent_weights" in result
    # Assert result["agent_weights"]["fello"] == 0.8
```

### Test 9: Key claims extraction
```python
def test_extract_key_claims():
    output = {
        "risks": ["cascade failure", "burnout"],
        "conditions_for_safe_proceeding": ["Set exit criteria"],
        "alternative_paths": ["try a smaller probe"],
    }
    claims = _extract_key_claims(output)
    assert len(claims) == 4
    assert any("risk: cascade failure" in c for c in claims)
    assert any("proposes: try a smaller probe" in c for c in claims)
```

### Test 10: Backward compatibility -- agents without deliberation fields still validate
```python
def test_non_deliberation_outputs_still_valid():
    """Round-1 outputs (no agree_with/disagree_with) must still pass validation."""
    payload = {
        "sender": "FELLO", "target": "pineal", "message_type": "proposal",
        "content": "test", "tags": [], "alternative_paths": [],
        "upside_opportunities": [], "low_cost_experiments": [], "confidence": 0.6,
    }
    ok, errors = validate_agent_output("fello", payload)
    assert ok
```

## Estimated Impact

### User Experience
- **Transparency**: Users see WHY agents changed positions, not just that confidence shifted
- **Trust**: Structured agree/disagree shows genuine deliberation, not arbitrary number changes
- **Insight**: Minority reports surface dissenting views that may be valid

### System Quality
- **Better Pineal synthesis**: With structured justifications, Pineal can make more informed resolutions
- **Richer consolidation data**: agree_with/disagree_with patterns feed back into agent calibration over time
- **Shadow activation**: Shadow becomes an active deliberation participant, grounding debate in data

### Measurable Outcomes
- Deliberation convergence rate should improve (more targeted revisions)
- Pineal confidence should be more calibrated (reasoning-informed, not just magnitude-based)
- Agent reaction quality visible in thought bubbles (new green/red chips)

## Files Changed

| File | Change |
|------|--------|
| `agent_zero/agent_schemas.py` | Add optional deliberation fields to fello, othello, shadow schemas |
| `agent_zero/cognitive_agents.py` | Rewrite _fello_reaction, _othello_reaction, _shadow_reaction |
| `agent_zero/cognitive_runtime.py` | Enrich _build_reaction_context, transcript, add _extract_key_claims |
| `agent_zero/prompts/cognitive_agents/pineal.md` | Update step 4 for structured deliberation data |
| `agent_zero-ui/src/components/ThoughtBubbles.tsx` | Render agree/disagree chips (optional, can be separate PR) |

## References

1. [From Debate to Deliberation: Structured Collective Reasoning with Typed Epistemic Acts](https://arxiv.org/html/2603.11781) -- DCI framework, 2026
2. [Demystifying Multi-Agent Debate: The Role of Confidence and Diversity](https://arxiv.org/html/2601.19921) -- ConfMAD, 2026
3. [Can LLM Agents Really Debate?](https://arxiv.org/abs/2511.07784) -- Controlled study, 2025
4. [Breaking Mental Set through Diverse Multi-Agent Debate (DMAD)](https://github.com/MraDonkey/DMAD) -- ICLR 2025
5. [Unlocking the Power of Multi-Agent LLM: From Lazy Agents to Deliberation](https://arxiv.org/abs/2511.02303) -- 2025
6. [Multi-LLM-Agents Debate: Performance, Efficiency, and Scaling Challenges](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-mad-159/blog/mad/) -- ICLR Blogposts 2025
