---
topic: Logic Programming for Transparent Reasoning
status: ready_for_implementation
priority: medium
estimated_complexity: large
researched_at: 2026-03-18T15:00:00Z
---

# Logic Programming for Transparent Reasoning

## Problem Statement

Agent Zero has a rich multi-agent cognitive pipeline that generates reasoning (thought bubbles, deliberation transcripts, agent details). But the reasoning is opaque in a critical way: **users can see what agents concluded but not why**. There is no formal inference trace showing how evidence led to conclusions. The pipeline is:

1. Agents emit natural-language thoughts ("I see a fallback here...")
2. Pineal synthesizes with a `rationale_summary` field
3. But the logical chain -- "because pattern X was detected AND commitment Y has miss_rate > 0.3 AND stage_of_change is precontemplation, THEREFORE recommend Z" -- is never formally represented or shown

AZ's logic programming engine (C095) can express these reasoning chains as Prolog-style rules, unify them against user data, and generate proof trees that trace exactly why a recommendation was made. This creates **contestable, auditable reasoning** -- the user can see and challenge any step.

## Current State in Agent Zero

### Reasoning Display (`cognitive_agents.py:1193-1278`)
- `build_agent_thought()` converts structured output to natural language
- Each agent has a thought builder: `_fello_thought()`, `_othello_thought()`, etc.
- Thoughts are 1-2 sentences, not formal inference chains

### Agent Details Panel (`cognitive_runtime.py:730-747`)
- `_extract_agent_details()` provides expandable data (alternative_paths, risks, patterns)
- Shows raw structured output, not reasoning steps
- No lineage from evidence to conclusion

### Deliberation Transcript (`cognitive_runtime.py:854-976`)
- Records agree/disagree/revision between agents
- Shows what changed, but not the logical argument structure
- No formal support/attack relations

### Pineal Synthesis (`cognitive_agents.py:420-453`)
- Confidence computed from heuristic formula (base + bonuses/penalties)
- `sources` field lists which agents influenced the decision
- `rationale_summary` is free-text, not structured proof

### Missing Components
1. No formal representation of reasoning rules
2. No proof trees showing derivation chains
3. No way for users to challenge specific premises
4. No counterfactual explanations ("what would change this recommendation?")
5. No connection to C095's SLD resolution engine

### C095 Logic Programming Engine
- Full Prolog interpreter with SLD resolution
- Trace mode: `interpreter.trace = True` generates `CALL: goal depth=N` output
- 60+ builtins including findall, bagof, setof for collecting solutions
- CLP(FD) constraint logic programming
- Unification, backtracking, cut, negation as failure
- **Already built** -- just disconnected from Agent Zero

## Industry Standard / Research Findings

### 1. Argumentative Human-AI Decision Making

Hadoux & Hunter (2025) propose a three-task architecture for AI that "reasons with humans, not for them":

1. **Argumentation Mining**: Extract argument structures from unstructured text
2. **Argumentation Synthesis**: Generate new premises and counterarguments
3. **Argumentative Reasoning**: Formal evaluation with transparent semantics

Key design principle: **"Separate generation from evaluation"** -- the LLM generates candidate arguments, while a formal solver determines acceptability. This maps directly to Agent Zero: cognitive agents (LLM) generate reasoning, while a logic engine (C095) evaluates and traces it.

The paper emphasizes **bidirectional propagation**: when a user challenges one premise, the system recomputes acceptability across the entire argument graph. This creates genuinely contestable AI.

**Citation**: Hadoux, E., Hunter, A. (2025). "Argumentative Human-AI Decision-Making: Toward AI Agents That Reason With Us, Not For Us." *arXiv:2603.15946*. https://arxiv.org/html/2603.15946

### 2. Interactive Reasoning Visualization

Pang et al. (2025) designed "Interactive Reasoning" (Hippo), which visualizes chain-of-thought as a **hierarchy of topics** rather than linear text. Key findings from a 16-person user study:

- Users reported significantly higher sense-making (M=6.44 vs 5.19, p=0.004)
- Greater awareness of reasoning assumptions (M=6.25 vs 4.69, p=0.012)
- Higher decision confidence (M=6.06 vs 5.06, p=0.049)
- Tree-based visualization outperformed linear reasoning display

The design uses: topic decomposition, collapsible subtrees, feedback nodes for user input, and visual linking between reasoning nodes and response sentences.

**Citation**: Pang, Y., et al. (2025). "Interactive Reasoning: Visualizing and Controlling Chain-of-Thought Reasoning in Large Language Models." *UIST 2025*. https://arxiv.org/html/2506.23678v1

### 3. Explainable AI in Multi-Agent Systems via Layered Prompting

Research published February 2025 proposes **layered prompting** for multi-agent explainability: structuring agent interactions into hierarchical, interpretable steps. Each layer produces a reasoning artifact that connects to the next, creating an end-to-end **cognitive lineage** -- an inspectable chain from input to output.

This aligns with Agent Zero's existing multi-layer pipeline (workers -> pineal -> speaker) but adds formal traceability between layers.

**Citation**: "Explainable AI in Multi-Agent Systems: Advancing Transparency with Layered Prompting." (2025). *ResearchGate*. https://www.researchgate.net/publication/388835453_Explainable_AI_in_Multi-Agent_Systems_Advancing_Transparency_with_Layered_Prompting

### 4. Argumentation + Logic Programming for XAI

Calegari et al. (2021) establish that argumentation combined with logic programming can address both explainability and ethical concerns in AI. Key techniques:

- **Abductive logic programming**: Generate hypotheses that explain observations
- **Defeasible logic**: Rules with exceptions -- a recommendation holds UNLESS a defeating condition is met
- **Argumentation semantics**: Formal attack/support relations between claims

The insight: SLD resolution proof trees ARE explanations. Each successful derivation shows exactly which rules fired and which facts grounded the conclusion. Converting this to natural language produces human-readable explanations.

**Citation**: Calegari, R., Ciatto, A., Omicini, A. (2021). "Explainable and Ethical AI: A Perspective on Argumentation and Logic Programming." *CEUR Workshop Proceedings*, Vol. 2742. https://ceur-ws.org/Vol-2742/paper5.pdf

### 5. Neurosymbolic Integration for Reasoning

Belle (2025) argues that logic programming remains uniquely suited for AI explainability because it provides **compositional, inspectable reasoning** that neural networks lack. The key advantage: logic programs can be read, verified, and modified by humans, creating a "glass box" that complements the LLM's "black box".

**Citation**: Belle, V. (2025). "On the Relevance of Logic for Artificial Intelligence, and the Promise of Neurosymbolic Learning." *SAGE Journals*. https://journals.sagepub.com/doi/10.1177/29498732251339951

### 6. Proof Trees as User-Facing Explanations

SLD proof trees show successful derivation paths. Each node represents a subgoal, and edges represent the clause used to resolve it. For user-facing explanations, proof trees can be rendered as:

- **Natural language**: "I recommended X because [rule R1] matched your pattern of [fact F1], and [rule R2] confirmed that [fact F2]"
- **Collapsible tree**: Interactive UI where users expand/collapse proof steps
- **Argument graph**: Support/attack visualization showing which evidence supports which conclusion

**Citation**: "Simply Logical: Intelligent Reasoning by Example" -- SLD Resolution chapter. https://book.simply-logical.space/src/text/1_part_i/3.1.html

## Proposed Implementation

### Architecture: Reasoning Knowledge Base + Proof Renderer

```
Cognitive Agents (LLM)          Logic Engine (C095)         UI
       |                              |                      |
  emit structured output  -->  assert as facts/rules  -->  render proof tree
  (patterns, risks, etc)       query for recommendation     as explanation
                               generate proof trace
```

### Step 1: Reasoning Knowledge Base

**New file**: `agent_zero/reasoning_kb.py`

Define Agent Zero's decision logic as Prolog rules that mirror what the cognitive agents do implicitly:

```python
REASONING_RULES = """
% -- Recommendation rules --

% Recommend accountability intervention when follow-through is low
recommend(accountability, User) :-
    commitment_prediction(User, Pred), Pred < 0.4,
    stage_of_change(User, Stage), Stage \\= action,
    recent_miss_rate(User, Rate), Rate > 0.3.

% Recommend encouragement when user is making progress
recommend(encouragement, User) :-
    stage_of_change(User, action),
    streak_active(User, Streak), Streak >= 3,
    commitment_prediction(User, Pred), Pred >= 0.6.

% Recommend clarification when ambiguity is high
recommend(clarify, User) :-
    agent_disagreement(Magnitude), Magnitude >= 0.3,
    evidence_quality(low).

% Recommend risk warning when safety concern detected
recommend(risk_warning, User) :-
    risk_detected(Risk, Severity),
    Severity >= 0.7,
    reversibility(Risk, Reversibility),
    Reversibility < 0.5.

% -- Pattern rules --

% User is in avoidance pattern
pattern(avoidance, User) :-
    topic_frequency(User, Topic, Freq), Freq >= 3,
    topic_depth(User, Topic, Depth), Depth < 2,
    commitment_on_topic(User, Topic, Status), Status = missed.

% User shows commitment fatigue
pattern(commitment_fatigue, User) :-
    active_commitments(User, Count), Count > 5,
    recent_miss_rate(User, Rate), Rate > 0.4,
    weight_distribution(User, heavy, HeavyCount), HeavyCount >= 3.

% -- Explanation rules --

% Why was this recommendation made?
explanation(Rec, Reasons) :-
    findall(R, reason_for(Rec, R), Reasons).

reason_for(accountability, 'low follow-through prediction') :-
    commitment_prediction(_, Pred), Pred < 0.4.
reason_for(accountability, 'not yet in action stage') :-
    stage_of_change(_, Stage), Stage \\= action.
reason_for(accountability, 'recent misses exceed threshold') :-
    recent_miss_rate(_, Rate), Rate > 0.3.
"""
```

### Step 2: Fact Assertion from Agent Outputs

**Add to**: `agent_zero/reasoning_kb.py`

After each cognitive pipeline run, assert current-turn facts into the logic engine:

```python
from challenges.C095_logic_programming.logic_programming import Interpreter, run

class ReasoningKB:
    """Logic-based reasoning knowledge base for transparent inference."""

    def __init__(self):
        self.interpreter = Interpreter()
        self.interpreter.trace = True  # Enable proof tracing
        self.interpreter.consult(REASONING_RULES)

    def assert_turn_facts(self, agent_outputs: dict, context: dict):
        """Assert facts from current cognitive pipeline run."""
        facts = []

        # From Shadow agent
        shadow = agent_outputs.get("shadow", {})
        if "commitment_prediction" in shadow:
            pred = shadow["commitment_prediction"]
            facts.append(f"commitment_prediction(user, {pred}).")

        if "stage_of_change" in shadow:
            stage = shadow["stage_of_change"].lower().replace(" ", "_")
            facts.append(f"stage_of_change(user, {stage}).")

        for pattern in shadow.get("pattern_matches", []):
            name = pattern.get("pattern", "unknown").lower().replace(" ", "_")
            conf = pattern.get("confidence", 0.5)
            facts.append(f"detected_pattern(user, {name}, {conf}).")

        # From Othello agent
        othello = agent_outputs.get("othello", {})
        for risk in othello.get("risks", []):
            severity = risk.get("severity", 0.5) if isinstance(risk, dict) else 0.5
            name = (risk.get("risk", "unknown") if isinstance(risk, dict) else str(risk)).lower().replace(" ", "_")
            facts.append(f"risk_detected({name}, {severity}).")

        # From context
        if "miss_rate" in context:
            facts.append(f"recent_miss_rate(user, {context['miss_rate']}).")
        if "active_commitment_count" in context:
            facts.append(f"active_commitments(user, {context['active_commitment_count']}).")

        # From deliberation
        if "disagreement_magnitude" in context:
            facts.append(f"agent_disagreement({context['disagreement_magnitude']}).")
        if "evidence_quality" in context:
            eq = context["evidence_quality"].lower()
            facts.append(f"evidence_quality({eq}).")

        # Assert all facts
        for fact in facts:
            self.interpreter.consult(fact)

    def query_recommendations(self) -> list[dict]:
        """Query the KB for recommendations with proof traces."""
        self.interpreter.output = []  # Reset trace
        results = self.interpreter.query_string("recommend(Action, user)")

        recommendations = []
        for result in results:
            action = str(result.bindings.get("Action", "unknown"))

            # Get explanation
            self.interpreter.output = []
            self.interpreter.query_string(f"explanation({action}, Reasons)")

            recommendations.append({
                "action": action,
                "proof_trace": list(self.interpreter.output),
                "bindings": {k: str(v) for k, v in result.bindings.items()},
            })

        return recommendations

    def get_proof_tree(self, query_str: str) -> dict:
        """Run a query and return the proof tree as structured data."""
        self.interpreter.output = []  # Reset trace
        results = self.interpreter.query_string(query_str)

        # Parse trace output into tree structure
        trace_lines = self.interpreter.output
        tree = self._parse_trace_to_tree(trace_lines)

        return {
            "query": query_str,
            "success": len(results) > 0,
            "solutions": [{k: str(v) for k, v in r.bindings.items()} for r in results],
            "proof_tree": tree,
        }

    def _parse_trace_to_tree(self, trace_lines: list[str]) -> list[dict]:
        """Convert CALL trace lines into nested proof tree."""
        nodes = []
        for line in trace_lines:
            if line.startswith("CALL:"):
                parts = line.split("depth=")
                goal = parts[0].replace("CALL:", "").strip()
                depth = int(parts[1]) if len(parts) > 1 else 0
                nodes.append({"goal": goal, "depth": depth})
        return nodes
```

### Step 3: Proof Tree Rendering for UI

**Add to**: `agent_zero/reasoning_kb.py`

```python
class ProofRenderer:
    """Renders proof trees into user-facing explanations."""

    def to_natural_language(self, proof_tree: dict) -> str:
        """Convert proof tree to natural language explanation."""
        if not proof_tree["success"]:
            return "No recommendation could be derived from the available evidence."

        nodes = proof_tree["proof_tree"]
        if not nodes:
            return f"Recommendation based on query: {proof_tree['query']}"

        # Build explanation from proof nodes
        explanation_parts = []
        for node in nodes:
            goal = node["goal"]
            indent = "  " * node["depth"]
            # Convert Prolog goals to natural language
            nl = self._goal_to_natural_language(goal)
            if nl:
                explanation_parts.append(f"{indent}{nl}")

        return "\n".join(explanation_parts)

    def to_argument_graph(self, proof_tree: dict) -> dict:
        """Convert proof tree to argument graph (support/attack relations)."""
        nodes = []
        edges = []

        for i, node in enumerate(proof_tree.get("proof_tree", [])):
            node_id = f"n{i}"
            nodes.append({
                "id": node_id,
                "label": self._goal_to_natural_language(node["goal"]) or node["goal"],
                "depth": node["depth"],
                "type": "fact" if node["depth"] > 2 else "rule",
            })
            # Connect to parent (previous node at depth-1)
            for j in range(i - 1, -1, -1):
                if proof_tree["proof_tree"][j]["depth"] == node["depth"] - 1:
                    edges.append({
                        "from": f"n{j}",
                        "to": node_id,
                        "relation": "supports",
                    })
                    break

        return {"nodes": nodes, "edges": edges}

    def _goal_to_natural_language(self, goal: str) -> str:
        """Map Prolog goals to human-readable statements."""
        mappings = {
            "commitment_prediction": "Your follow-through prediction is {1}",
            "stage_of_change": "You are in the '{1}' stage of change",
            "recent_miss_rate": "Your recent miss rate is {1}",
            "risk_detected": "Risk detected: {0} (severity {1})",
            "active_commitments": "You have {1} active commitments",
            "agent_disagreement": "Agent disagreement level is {0}",
            "evidence_quality": "Evidence quality is {0}",
            "streak_active": "You have an active streak of {1} days",
            "detected_pattern": "Pattern detected: {0} (confidence {1})",
        }

        # Parse goal into functor and args
        if "(" in goal:
            functor = goal[:goal.index("(")]
            args_str = goal[goal.index("(")+1:goal.rindex(")")]
            args = [a.strip() for a in args_str.split(",")]
        else:
            return goal

        template = mappings.get(functor)
        if template:
            try:
                return template.format(*args)
            except (IndexError, KeyError):
                return goal
        return None
```

### Step 4: Integration with Cognitive Runtime

**Modify**: `cognitive_runtime.py`

After the cognitive pipeline runs, assert facts and query for formal recommendations:

```python
# After all agents have run and Pineal has synthesized:

from reasoning_kb import ReasoningKB, ProofRenderer

async def run_cognitive_pipeline(user_input, context, on_step=None):
    # ... existing pipeline code ...

    # NEW: After agent outputs collected, run logic-based reasoning
    kb = ReasoningKB()
    kb.assert_turn_facts(agent_outputs, pipeline_context)
    recommendations = kb.query_recommendations()

    # Get proof tree for the primary recommendation
    if recommendations:
        primary = recommendations[0]
        proof = kb.get_proof_tree(f"recommend({primary['action']}, user)")
        renderer = ProofRenderer()

        # Emit proof as a reasoning step
        if on_step:
            await on_step({
                "agent": "reasoning_engine",
                "agent_id": "reasoning_engine",
                "thought": renderer.to_natural_language(proof),
                "stage": "proof",
                "confidence": None,
                "details": {
                    "proof_tree": proof["proof_tree"],
                    "argument_graph": renderer.to_argument_graph(proof),
                    "recommendations": recommendations,
                },
            })

    # Include proof in response plan for Speaker
    response_plan["reasoning_proof"] = proof if recommendations else None

    # ... rest of pipeline ...
```

### Step 5: API Endpoint for Proof Queries

**Add to**: `agent_zero_server.py`

```python
@app.post("/reasoning/query")
async def reasoning_query(
    body: dict,
    user: dict = Depends(get_current_user)
):
    """Query the reasoning KB and return proof tree."""
    query_str = body.get("query", "")
    shadow = await get_shadow(user["user_id"])

    kb = ReasoningKB()
    # Assert user-specific facts from shadow
    kb.assert_user_facts(shadow)

    proof = kb.get_proof_tree(query_str)
    renderer = ProofRenderer()

    return JSONResponse({
        "query": query_str,
        "success": proof["success"],
        "explanation": renderer.to_natural_language(proof),
        "proof_tree": proof["proof_tree"],
        "argument_graph": renderer.to_argument_graph(proof),
    })
```

### Step 6: Frontend Proof Display

**Modify**: Agent Zero UI to render proof trees

The existing agent detail panel (expandable reasoning chips) can be extended with:

1. A "Why?" button on each recommendation that opens the proof tree
2. Collapsible tree visualization (matching the Interactive Reasoning pattern from Pang et al.)
3. Argument graph showing support/attack relations between evidence and conclusions

```typescript
// ProofTreeView component (new)
interface ProofNode {
    goal: string;
    depth: number;
    naturalLanguage?: string;
}

// Render as indented collapsible tree
// Each node shows the natural language translation
// Clicking a node expands to show the raw Prolog goal
```

## Test Specifications

### Unit Tests: `agent_zero/test_reasoning_kb.py`

```python
# -- ReasoningKB tests --

def test_assert_shadow_facts():
    """Shadow output with commitment_prediction, stage_of_change asserted correctly."""

def test_assert_othello_risks():
    """Othello risks become risk_detected/2 facts."""

def test_assert_context_metrics():
    """miss_rate, active_commitment_count become facts."""

def test_query_accountability_recommendation():
    """Low prediction + high miss rate -> recommend(accountability, user)."""

def test_query_encouragement_recommendation():
    """Action stage + active streak -> recommend(encouragement, user)."""

def test_query_clarify_recommendation():
    """High disagreement + low evidence -> recommend(clarify, user)."""

def test_query_risk_warning():
    """High severity + low reversibility -> recommend(risk_warning, user)."""

def test_no_recommendation_when_facts_missing():
    """No facts asserted -> no recommendations."""

def test_multiple_recommendations():
    """Multiple rules can fire simultaneously."""

def test_proof_trace_generated():
    """Trace mode produces CALL lines for proof tree."""

def test_proof_tree_structure():
    """Parsed trace has correct depth hierarchy."""

# -- ProofRenderer tests --

def test_natural_language_commitment_prediction():
    """commitment_prediction(user, 0.3) -> 'Your follow-through prediction is 0.3'"""

def test_natural_language_stage_of_change():
    """stage_of_change(user, precontemplation) -> 'You are in the precontemplation stage'"""

def test_natural_language_full_proof():
    """Full proof tree renders as multi-line explanation."""

def test_argument_graph_structure():
    """Graph has nodes and support edges matching tree depth."""

def test_argument_graph_root_node():
    """Root node is the query, children are supporting facts."""

def test_empty_proof_tree():
    """No solutions -> 'No recommendation could be derived'"""

# -- Integration tests --

def test_full_pipeline_with_reasoning():
    """Agent outputs -> KB assertion -> query -> proof -> natural language."""

def test_proof_included_in_on_step():
    """on_step callback receives reasoning_engine event with proof."""

def test_reasoning_query_api():
    """POST /reasoning/query returns proof tree and explanation."""

def test_counterfactual_query():
    """Removing one fact changes recommendation (proof of contestability)."""
```

### Expected Test Count: ~20-25 tests

## Estimated Impact

1. **Contestable recommendations**: Users can see exactly which facts led to which conclusion. If they disagree with a premise (e.g., "my miss rate isn't really that high"), they can challenge it -- and the system can show what would change.

2. **Trust through transparency**: Pang et al. showed that reasoning visualization increases decision confidence by 20% (M=6.06 vs 5.06). Formal proof trees provide even stronger guarantees than free-text reasoning.

3. **Debuggable cognitive pipeline**: When agents produce unexpected recommendations, developers can inspect the proof tree to see exactly which rules fired and why. This replaces ad-hoc debugging with structured reasoning inspection.

4. **Foundation for contestability**: Following Hadoux & Hunter's architecture, the formal argumentation graph enables bidirectional propagation -- changing one fact recomputes the entire recommendation, making the system genuinely interactive.

5. **Regulatory compliance**: Explainable reasoning chains satisfy emerging AI transparency requirements (EU AI Act, etc.) by providing auditable decision traces.

## Files to Create/Modify

| Action | File | Changes |
|--------|------|---------|
| CREATE | `agent_zero/reasoning_kb.py` | ReasoningKB, ProofRenderer, REASONING_RULES |
| MODIFY | `agent_zero/cognitive_runtime.py` | Add reasoning engine step after agent pipeline |
| MODIFY | `agent_zero/agent_zero_server.py` | Add /reasoning/query endpoint |
| MODIFY | `agent_zero-ui/src/` | ProofTreeView component (optional, UI enhancement) |
| CREATE | `agent_zero/test_reasoning_kb.py` | ~20-25 tests |

## Relevant AZ Challenges

- **C095** (Logic Programming): Core Prolog interpreter for reasoning
- **C094** (Constraint Solver): Used by C095's CLP(FD) for constraint reasoning
- **C096** (Datalog): Could be used for bottom-up reasoning over episode data (future)
- **C098** (Program Verifier): Hoare-logic verification for formal guarantees (future)
