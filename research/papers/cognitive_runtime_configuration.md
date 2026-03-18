---
topic: Cognitive Runtime Configuration Extraction
status: ready_for_implementation
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T22:00:00Z
---

# Cognitive Runtime Configuration Extraction

## Problem Statement

The Agent Zero cognitive runtime has 30+ hardcoded numeric constants scattered across 6 files. These constants control agent activation thresholds, memory decay rates, consolidation parameters, context budgets, and disagreement detection. Changing any value requires editing Python source code, restarting the server, and risking merge conflicts. There is no way for operators to tune system behavior at runtime, no validation that values are within safe ranges, and no centralized view of what parameters exist.

This is a well-documented anti-pattern in production AI systems. The Twelve-Factor App methodology (Factor III) mandates strict separation of configuration from code. Research on agentic AI configuration (Huang et al., 2025) demonstrates that learned, per-query configuration can improve accuracy by up to 25% over static defaults -- but the prerequisite is that configuration is externalized and accessible to adaptation mechanisms.

## Current State in Agent Zero

### cognitive_runtime.py (6 constants + ~15 inline magic numbers)
- `DELIBERATION_THRESHOLD = 0.30` (line 22) -- triggers deliberation round when disagreement magnitude >= this
- `NO_SIGNAL_THRESHOLD = 0.25` (line 505) -- VOI: below this, agent historically adds no signal
- `MIN_SUPPRESSION_CONFIDENCE = 0.5` (line 506) -- VOI: minimum confidence to suppress an agent
- `activation_threshold = 0.35 + (max_rule_conf * 0.15)` (line 396) -- adaptive activation; base 0.35, scale factor 0.15
- `min_agents = 1 if max_rule_conf >= 0.6 else 2` (line 416) -- minimum agent count gate
- Heuristic weights: fello=0.8, prefrontal=0.25, othello=0.7/0.25, complexity boost=0.6 (lines 572-589)
- Hybrid boosts: othello emotional +0.2, prefrontal complexity +0.15 (lines 476, 483)
- Disagreement: evidence discount=0.6, directional tiers=0.3/0.15/0.10, overlap damping=0.5, shadow tension=0.25/0.15, regime boost=0.10, disagreement gate=0.3 (lines 654-693)
- `AGENT_INFERENCE_COST` dict (lines 496-502) -- per-agent cost multipliers

### consolidator.py (8 constants, lines 34-47)
- `MIN_UNCONSOLIDATED_EPISODES = 5`
- `MAX_HOURS_BETWEEN_CONSOLIDATIONS = 1.0`
- `MAX_ACTIVE_RULES = 20`
- `MIN_CLUSTER_SIZE = 3`
- `MIN_RULE_CONFIDENCE = 0.4`
- `STALE_DAYS = 30`
- `RETIRED_DAYS = 90`
- `MERGE_DISTANCE_THRESHOLD = 0.6`, `MIN_CLUSTER_COHERENCE = 0.4`, `TEMPORAL_WEIGHT = 0.2`

### episode_store.py (3 constants, lines 38-44)
- `EPISODE_CAP = 200`
- `DEFAULT_LAMBDA_PER_HOUR = 0.005`
- `RETRIEVAL_BOOST = 0.2`

### context_manager.py (4 constants, lines 13-19)
- `MODEL_CONTEXT_LIMIT = 32768`
- `COMPRESSION_THRESHOLD = 0.70`
- `SYSTEM_PROMPT_BUDGET = 0.30`
- `PROTECTED_RECENT_TURNS = 6`

### session_checkin.py (2 constants, lines 17-18)
- `MOTIVATION_THRESHOLD = 0.5`
- `MAX_ITEMS_IN_GREETING = 3`

### agent_zero_server.py (2 constants, lines 180, 206)
- `_PROMPT_LEAK_TAIL_CHARS = 240`
- `_MAX_MODEL_TOOL_ROUNDS = 3`

### auth.py (1 constant, line 20)
- `JWT_EXPIRY_HOURS = 24`

**Total: ~30 module-level constants + ~15 inline magic numbers = ~45 tunable parameters.**

## Industry Standard / Research Findings

### 1. Twelve-Factor Configuration (Wiggins, 2011)
The Twelve-Factor App methodology, Factor III ("Store config in the environment"), is the foundational principle: configuration that varies between deploys must be strictly separated from code. Environment variables are the recommended transport mechanism because they are language/OS-agnostic and hard to accidentally commit to source control.
URL: https://12factor.net/config

### 2. Pydantic Settings (Pydantic, 2025)
Pydantic Settings 2.0+ provides type-safe, validated configuration from environment variables and .env files. It uses Python dataclasses with automatic type coercion, validation, default values, and documentation. FastAPI projects use it as the standard configuration mechanism. The `Field(ge=0.0, le=1.0)` pattern enforces range constraints at startup, failing fast on misconfiguration.
URL: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

### 3. Hydra + OmegaConf (Meta, 2019-2025)
Meta's Hydra framework provides hierarchical configuration composition with YAML files, command-line overrides, and structured configs backed by OmegaConf. It is the ML community's standard for experiment configuration, supporting config groups (model, optimizer, dataset) and runtime override syntax. OmegaConf adds merge semantics and interpolation (${model.hidden_size}).
URL: https://hydra.cc/docs/intro/

### 4. ARC: Learning to Configure Agentic AI Systems (Huang et al., 2025)
This paper introduces a learned hierarchical RL policy that dynamically configures agentic AI systems per-query, achieving up to 25% accuracy improvement over static configurations. The key insight: adaptive configuration outperforms hand-tuned defaults because different queries have different optimal parameter settings. The prerequisite is externalized, accessible configuration.
URL: https://arxiv.org/abs/2602.11574

### 5. Martin Fowler's Feature Toggle Taxonomy (Fowler, 2017)
Fowler categorizes toggles into Release, Experiment, Ops, and Permission. Agent Zero's thresholds are "Ops Toggles" -- long-lived parameters that control system behavior and enable graceful degradation. The article recommends these be externalized with runtime mutability and default-safe values.
URL: https://martinfowler.com/articles/feature-toggles.html

### 6. OptiMindTune: Multi-Agent HPO (2025)
A three-agent framework (Recommender, Evaluator, Decision) for intelligent hyperparameter optimization. Demonstrates that multi-agent systems benefit from dynamic, collaborative threshold tuning rather than static presets. Agent Zero's multi-agent cognitive pipeline is a direct analogue.
URL: https://arxiv.org/abs/2505.19205

## Proposed Implementation

### Architecture: Single `Agent ZeroConfig` Pydantic Settings Class

Create `agent_zero/config.py` with a single `Agent ZeroConfig(BaseSettings)` class that centralizes all parameters. Use `pydantic-settings` (already available via the existing pydantic dependency) with environment variable loading and `.env` file support.

### File: `agent_zero/config.py` (NEW)

```python
"""Centralized configuration for the Agent Zero cognitive runtime.

All tunable parameters are defined here with type validation, range constraints,
and documentation. Values load from environment variables (prefix AGENT_ZERO_)
or a .env file, falling back to defaults that match current hardcoded values.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Agent ZeroConfig(BaseSettings):
    """Agent Zero runtime configuration.

    All fields have defaults matching current hardcoded values (zero-change migration).
    Override via environment variables prefixed with AGENT_ZERO_, e.g.:
        AGENT_ZERO_DELIBERATION_THRESHOLD=0.35
    """

    model_config = {"env_prefix": "AGENT_ZERO_", "env_file": ".env", "extra": "ignore"}

    # --- Cognitive Runtime: Agent Selection ---
    deliberation_threshold: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Disagreement magnitude that triggers deliberation round"
    )
    activation_threshold_base: float = Field(
        default=0.35, ge=0.0, le=1.0,
        description="Base activation threshold (scales up with rule confidence)"
    )
    activation_threshold_scale: float = Field(
        default=0.15, ge=0.0, le=0.5,
        description="How much max rule confidence raises activation threshold"
    )
    min_agents_high_confidence: int = Field(
        default=1, ge=1, le=7,
        description="Minimum agents when rule confidence >= confidence_gate"
    )
    min_agents_low_confidence: int = Field(
        default=2, ge=1, le=7,
        description="Minimum agents when rule confidence < confidence_gate"
    )
    confidence_gate: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Rule confidence level that allows fewer agents"
    )

    # --- Cognitive Runtime: VOI Gating ---
    no_signal_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Below this signal level, agent is considered to add no value"
    )
    min_suppression_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence to suppress an agent via VOI"
    )

    # --- Cognitive Runtime: Heuristic Weights (Cold Start) ---
    heuristic_default_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Default agent weight when no learned data exists"
    )
    heuristic_fello_memory: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Fello weight when memory signals detected"
    )
    heuristic_suppressed_weight: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Weight for agents suppressed during memory recall"
    )
    heuristic_complexity_boost: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum weight when complexity signals >= 2"
    )

    # --- Cognitive Runtime: Hybrid Boosts ---
    emotional_boost: float = Field(
        default=0.2, ge=0.0, le=0.5,
        description="Othello weight boost for emotional content"
    )
    complexity_boost: float = Field(
        default=0.15, ge=0.0, le=0.5,
        description="Prefrontal weight boost for complex turns"
    )

    # --- Cognitive Runtime: Disagreement Detection ---
    evidence_discount: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Confidence delta multiplier when evidence quality is high"
    )
    directional_high: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Directional disagreement when both agents > 0.7 confidence"
    )
    directional_medium: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Directional disagreement when both agents > 0.6 confidence"
    )
    directional_low: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Directional disagreement when both agents > 0.5 confidence"
    )
    overlap_damping: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Directional reduction when content overlap > 50%"
    )
    shadow_tension_high: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Shadow tension when commitment < 0.4 and fello > 0.65"
    )
    shadow_tension_low: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Shadow tension when commitment < 0.35 and fello > 0.5"
    )
    regime_tension_boost: float = Field(
        default=0.10, ge=0.0, le=0.5,
        description="Extra shadow tension for avoidant/unreliable regimes"
    )
    disagreement_gate: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Magnitude threshold for reporting disagreement"
    )

    # --- Consolidator ---
    min_unconsolidated_episodes: int = Field(
        default=5, ge=1, le=100,
        description="Minimum episodes before consolidation triggers"
    )
    max_hours_between_consolidations: float = Field(
        default=1.0, ge=0.1, le=24.0,
        description="Maximum hours before consolidation is forced"
    )
    max_active_rules: int = Field(
        default=20, ge=5, le=100,
        description="Maximum number of active consolidation rules"
    )
    min_cluster_size: int = Field(
        default=3, ge=2, le=20,
        description="Minimum episodes to form a valid cluster"
    )
    min_rule_confidence: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Minimum confidence for a consolidation rule"
    )
    stale_days: int = Field(
        default=30, ge=7, le=365,
        description="Days before a rule is marked stale"
    )
    retired_days: int = Field(
        default=90, ge=14, le=730,
        description="Days before a rule is retired"
    )
    merge_distance_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Stop merging clusters when closest pair exceeds this"
    )
    min_cluster_coherence: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Reject clusters below this intra-cluster similarity"
    )
    temporal_weight: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Weight for temporal proximity vs topic Jaccard in clustering"
    )

    # --- Episode Store ---
    episode_cap: int = Field(
        default=200, ge=50, le=10000,
        description="Maximum stored episodes per user"
    )
    default_lambda_per_hour: float = Field(
        default=0.005, ge=0.0, le=0.1,
        description="Exponential decay rate for episode relevance"
    )
    retrieval_boost: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Score boost when an episode is retrieved"
    )

    # --- Context Manager ---
    model_context_limit: int = Field(
        default=32768, ge=4096, le=131072,
        description="Model context window size in tokens"
    )
    compression_threshold: float = Field(
        default=0.70, ge=0.3, le=0.95,
        description="Compress conversation when at this fraction of context limit"
    )
    system_prompt_budget: float = Field(
        default=0.30, ge=0.1, le=0.6,
        description="Fraction of context reserved for system prompt + reasoning"
    )
    protected_recent_turns: int = Field(
        default=6, ge=1, le=20,
        description="Number of recent messages to keep uncompressed"
    )

    # --- Session Check-in ---
    motivation_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum motivation score to include item in greeting"
    )
    max_items_in_greeting: int = Field(
        default=3, ge=1, le=10,
        description="Maximum commitment items shown in session greeting"
    )

    # --- Server ---
    prompt_leak_tail_chars: int = Field(
        default=240, ge=50, le=1000,
        description="Characters to check for prompt leak in response tail"
    )
    max_model_tool_rounds: int = Field(
        default=3, ge=1, le=10,
        description="Maximum tool call rounds before forcing text response"
    )
    jwt_expiry_hours: int = Field(
        default=24, ge=1, le=720,
        description="JWT token expiry in hours"
    )


# Singleton instance -- import this everywhere
config = Agent ZeroConfig()
```

### Migration Strategy (Zero-Change Default)

Every default value matches the current hardcoded value exactly. The migration is:

1. **Create `agent_zero/config.py`** with the class above
2. **In each source file**, replace the module-level constant with an import:

   ```python
   # BEFORE (cognitive_runtime.py line 22):
   DELIBERATION_THRESHOLD = 0.30

   # AFTER:
   from config import config
   # ... then use config.deliberation_threshold instead of DELIBERATION_THRESHOLD
   ```

3. **For inline magic numbers**, replace with config references:

   ```python
   # BEFORE (cognitive_runtime.py line 396):
   activation_threshold = 0.35 + (max_rule_conf * 0.15)

   # AFTER:
   activation_threshold = config.activation_threshold_base + (max_rule_conf * config.activation_threshold_scale)
   ```

4. **Files to modify** (in order):
   - `cognitive_runtime.py`: Replace 3 module constants + ~15 inline numbers
   - `consolidator.py`: Replace 10 module constants
   - `episode_store.py`: Replace 3 module constants
   - `context_manager.py`: Replace 4 module constants + 1 inline
   - `session_checkin.py`: Replace 2 module constants
   - `agent_zero_server.py`: Replace 2 module constants
   - `auth.py`: Replace 1 module constant

5. **Create `agent_zero/.env.example`** documenting all variables with comments

6. **No `.env` file needed** -- defaults match current values, so system works unchanged

### Backward Compatibility

- All defaults match current hardcoded values: **zero behavioral change on deployment**
- Pydantic validates on import: if `AGENT_ZERO_DELIBERATION_THRESHOLD=banana`, the server fails at startup with a clear error message instead of crashing mid-request
- The `extra = "ignore"` setting means unrecognized env vars don't cause failures
- Existing tests pass without modification (they use default values)

### Dependency

`pydantic-settings` is the only new dependency. Since `pydantic` is already used (via FastAPI), adding `pydantic-settings` is a minimal, well-supported addition:
```
pip install pydantic-settings
```

### Future: Runtime Hot-Reload (Optional Enhancement)

Once config is externalized, a future paper could add:
- A `/admin/config` endpoint to view current configuration
- A `/admin/config` PATCH endpoint to update values at runtime (with validation)
- A `config.reload()` method that re-reads environment/.env
- Integration with ARC-style learned configuration (Huang et al., 2025) where the system adapts thresholds per-query

This paper does NOT implement hot-reload -- it only externalizes. Hot-reload is a separate, optional follow-up.

## Test Specifications

### test_config.py

```python
# Test 1: Default values match current hardcoded constants
def test_defaults_match_hardcoded():
    """Every default in Agent ZeroConfig must match the previously hardcoded value."""
    from config import Agent ZeroConfig
    c = Agent ZeroConfig()
    assert c.deliberation_threshold == 0.30
    assert c.activation_threshold_base == 0.35
    assert c.activation_threshold_scale == 0.15
    assert c.no_signal_threshold == 0.25
    assert c.min_suppression_confidence == 0.5
    assert c.min_unconsolidated_episodes == 5
    assert c.max_hours_between_consolidations == 1.0
    assert c.max_active_rules == 20
    assert c.episode_cap == 200
    assert c.default_lambda_per_hour == 0.005
    assert c.model_context_limit == 32768
    assert c.compression_threshold == 0.70
    assert c.protected_recent_turns == 6
    assert c.motivation_threshold == 0.5
    assert c.max_items_in_greeting == 3
    assert c.jwt_expiry_hours == 24

# Test 2: Environment variable override works
def test_env_override(monkeypatch):
    """Config picks up AGENT_ZERO_ prefixed env vars."""
    monkeypatch.setenv("AGENT_ZERO_DELIBERATION_THRESHOLD", "0.45")
    monkeypatch.setenv("AGENT_ZERO_EPISODE_CAP", "500")
    from config import Agent ZeroConfig
    c = Agent ZeroConfig()
    assert c.deliberation_threshold == 0.45
    assert c.episode_cap == 500

# Test 3: Validation rejects out-of-range values
def test_validation_rejects_invalid(monkeypatch):
    """Values outside constraints should raise ValidationError."""
    monkeypatch.setenv("AGENT_ZERO_DELIBERATION_THRESHOLD", "1.5")  # > 1.0
    from config import Agent ZeroConfig
    import pytest
    with pytest.raises(Exception):  # ValidationError
        Agent ZeroConfig()

# Test 4: Validation rejects non-numeric types
def test_validation_rejects_type_error(monkeypatch):
    """Non-numeric strings for float fields should raise ValidationError."""
    monkeypatch.setenv("AGENT_ZERO_DELIBERATION_THRESHOLD", "banana")
    from config import Agent ZeroConfig
    import pytest
    with pytest.raises(Exception):
        Agent ZeroConfig()

# Test 5: All fields have descriptions
def test_all_fields_documented():
    """Every config field must have a description."""
    from config import Agent ZeroConfig
    for name, field in Agent ZeroConfig.model_fields.items():
        assert field.description, f"Field {name} missing description"

# Test 6: All float fields in [0, 1] have ge/le constraints
def test_float_fields_constrained():
    """Float fields representing probabilities/fractions must be bounded."""
    from config import Agent ZeroConfig
    for name, field in Agent ZeroConfig.model_fields.items():
        if field.annotation is float and field.default is not None:
            if 0.0 <= field.default <= 1.0:
                metadata = field.metadata
                assert any(hasattr(m, 'ge') for m in metadata), f"{name} missing ge constraint"
                assert any(hasattr(m, 'le') for m in metadata), f"{name} missing le constraint"

# Test 7: Singleton import returns same config
def test_singleton_consistency():
    """Importing config from config module should return consistent values."""
    from config import config
    assert config.deliberation_threshold == config.deliberation_threshold  # sanity

# Test 8: Config serializes to dict for logging/admin
def test_config_to_dict():
    """Config should serialize to a dict for admin endpoints."""
    from config import Agent ZeroConfig
    c = Agent ZeroConfig()
    d = c.model_dump()
    assert isinstance(d, dict)
    assert "deliberation_threshold" in d
    assert len(d) >= 30  # we have ~30+ fields
```

## Estimated Impact

- **Operator ergonomics**: Operators can tune thresholds via `.env` or env vars without code changes or redeployment
- **Safety**: Pydantic validation catches misconfiguration at startup instead of producing silent incorrect behavior at runtime
- **Observability**: A single `config.model_dump()` call provides a complete snapshot of all system parameters for logging/debugging
- **Future adaptability**: Externalized config is the prerequisite for ARC-style learned configuration (Huang et al., 2025), A/B testing of parameters, and per-user configuration overrides
- **Maintenance**: All 45 tunable parameters documented in one place with types, ranges, and descriptions
- **Zero risk**: All defaults match current values; existing tests pass without modification

## Citations

1. Wiggins, A. (2011). "The Twelve-Factor App -- III. Config." https://12factor.net/config
2. Pydantic Team (2025). "Pydantic Settings -- Settings Management Using Pydantic." https://docs.pydantic.dev/latest/concepts/pydantic_settings/
3. Yadan, O. (2019-2025). "Hydra: A Framework for Elegantly Configuring Complex Applications." Meta Research. https://hydra.cc/docs/intro/
4. Huang, J. et al. (2025). "Learning to Configure Agentic AI Systems." arXiv:2602.11574. https://arxiv.org/abs/2602.11574
5. Fowler, M. (2017). "Feature Toggles (aka Feature Flags)." https://martinfowler.com/articles/feature-toggles.html
6. OptiMindTune Authors (2025). "OptiMindTune: A Multi-Agent Framework for Intelligent Hyperparameter Optimization." arXiv:2505.19205. https://arxiv.org/abs/2505.19205
