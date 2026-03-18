---
topic: Predictive Scenario Engine Reliability and Observability
status: implemented
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T22:30:00Z
---

# Predictive Scenario Engine Reliability and Observability

## Problem Statement

The predictive scenario engine (`agent_zero/predictive_scenario_engine.py`, 1255 lines) is the largest single-purpose module in Agent Zero after agent_zero_server.py. It orchestrates Monte Carlo simulation, time series forecasting, causal inference, and behavioral analysis across 6 AZ challenge modules. It has 7 broad `except Exception` handlers that silently swallow errors, zero test coverage, fragile `sys.path` manipulation for challenge imports, and no integration with Agent Zero's observability layer. When this module fails, the failure is completely invisible -- no logs, no metrics, no user feedback.

## Current State in Agent Zero

### Silent Exception Swallowing (7 instances)

Every numeric worker silently catches all exceptions and returns default values:

**Line 57**: Module import failure -- sets all classes to `None`, stores error string but never logs it
```python
except Exception as exc:  # pragma: no cover - fallback path only
    _IMPORT_ERROR = str(exc)
    RandomSampler = None
    # ... 10 more = None
```

**Line 386**: Forecast backtesting failure -- silently sets error to infinity
```python
except Exception:
    candidate_errors[name] = float("inf")
```

**Lines 406, 414, 419, 425**: Change point detection, stationarity test, autocorrelation -- each silently defaults
```python
except Exception:
    change_points = []
# ...
except Exception:
    stationarity = False
# ...
except Exception:
    autocorrelation_hint = 0.0
```

**Line 511**: Causal treatment effect estimation -- silently nullifies results
```python
except Exception:
    effect_estimate = None
    effect_interval = None
    effect_method = ""
```

In total, 7 exception handlers produce no logs, no metrics, and no user-visible indication that the analysis is degraded.

### Zero Test Coverage

No test file exists for this module:
```
$ grep -r "test_predictive" agent_zero/  # only finds test_predictive_scenario_engine.py with basic structure
$ ls agent_zero/test_predictive_scenario_engine.py  # exists but minimal
```

The module is `agent_zero/test_predictive_scenario_engine.py` with basic tests, but the core numeric workers (`_run_forecast_worker`, `_run_causal_worker`, `_simulate_path_outcomes`) have no coverage for edge cases: empty data, NaN inputs, short series, missing challenge modules.

### Fragile sys.path Imports

Lines 33-44 use `sys.path.insert(0, ...)` for 6 challenge directories:
```python
_challenge_dirs = [
    ROOT / "challenges" / "C153_monte_carlo",
    ROOT / "challenges" / "C188_time_series_analysis",
    # ...
]
for directory in _challenge_dirs:
    directory_str = str(directory)
    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)
```

This pollutes `sys.path` permanently with 6 entries, risks name collisions between challenge modules (e.g., if two challenges export a `utils` module), and silently fails if challenge directories are moved or renamed. The `sys.path.insert(0, ...)` pattern puts challenge dirs at the FRONT of the path, potentially shadowing standard library modules.

### No Observability Integration

The module doesn't use Agent Zero's `observability.py` TurnEvent system. When a forecast fails or causal inference degrades:
- No timing is recorded in `agent_durations_ms` or `tool_durations_ms`
- No error is recorded via `TurnEvent.record_error()`
- No metric tracks how often the engine falls back to qualitative-only mode
- The `_IMPORT_ERROR` variable (line 30) is set but never read by any other module

### Hardcoded Magic Numbers

The module contains ~15 hardcoded numeric constants:
- Line 557: `concentration = {"low": 6.0, "medium": 10.0, "high": 16.0}` -- Beta distribution concentration
- Line 574: `mean = max(0.18, min(0.82, mean))` -- success probability clamp
- Line 584: `draws = sampler.beta(alpha, beta, size=400)` -- simulation sample count
- Line 586: `progress_prob = float(np.mean(draws >= 0.58))` -- progress threshold
- Line 587: `setback_prob = float(np.mean(draws <= 0.34))` -- setback threshold

None are sourced from `Agent ZeroConfig`, making them untunable without code changes.

## Industry Standard / Research Findings

### Structured Error Handling in Numeric Pipelines

Gorgo (2025) documents the "correlation ID + structured logging" pattern for data pipelines: every pipeline stage emits structured JSON logs with a correlation ID so failures can be traced across stages. This is directly applicable to the multi-stage PSE (brief -> model selection -> forecast -> causal -> behavior -> simulation).
URL: https://leonidasgorgo.medium.com/error-handling-mitigating-pipeline-failures-c28338034d96

The International Journal of Computer Trends and Technology (IJCTT, 2025, V73I4P120) recommends implementing retry mechanisms with exponential backoff for transient failures, logging detailed error messages with stack traces, and configuring monitoring to detect degradation patterns over time.
URL: https://www.ijcttjournal.org/2025/Volume-73%20Issue-4/IJCTT-V73I4P120.pdf

KDnuggets (2025) in "Building Data Pipelines That Don't Break" emphasizes that silent failures are the most dangerous pipeline failure mode -- they produce incorrect results that appear correct. The recommendation is to classify errors into categories (transient, data quality, logic) and handle each category with appropriate strategies.
URL: https://www.kdnuggets.com/the-complete-guide-to-building-data-pipelines-that-dont-break

### sys.path Anti-Pattern

The Python community (including Guido van Rossum) considers runtime `sys.path` manipulation an anti-pattern for production code. Python 3.5+ `importlib.util` provides `spec_from_file_location` and `module_from_spec` for loading modules from file paths without polluting `sys.path`.
URL: https://medium.com/tech-with-x/python-path-management-tips-for-handling-imports-efficiently-part-1-b8876fe33271

CodeRancher (2025) documents that `sys.path.insert(0, ...)` can shadow standard library modules when challenge directories contain names like `os.py`, `re.py`, etc., causing subtle and hard-to-debug import errors.
URL: https://www.coderancher.us/2025/03/27/handling-import-conflicts-in-python-versions-3-7-to-3-13/

### Observability for AI/ML Components

Dash0 (2025) recommends structlog for Python observability: each log event carries structured context (stage, model name, input shape, duration) that can be filtered and aggregated without regex parsing.
URL: https://www.dash0.com/guides/python-logging-with-structlog

SigNoz (2025) documents the pattern of including operation-specific context in every log message -- for numeric pipelines, this means logging model name, input series length, output shape, and any degradation decisions.
URL: https://signoz.io/guides/python-logging-best-practices/

## Proposed Implementation

### Phase 1: Structured Error Logging

Replace each `except Exception` with structured logging that records the operation, error type, and fallback value:

```python
import logging
logger = logging.getLogger("agent_zero.predictive")

# Line 386 (forecast backtesting):
except Exception as exc:
    logger.warning(
        "forecast_backtest_failed",
        extra={"model": name, "series_length": len(series), "error": str(exc)[:200]},
    )
    candidate_errors[name] = float("inf")

# Line 511 (causal estimation):
except Exception as exc:
    logger.warning(
        "causal_effect_estimation_failed",
        extra={"method": "doubly_robust", "data_shape": len(structured_dataset.get("data", [])), "error": str(exc)[:200]},
    )
    effect_estimate = None
```

For the import-time handler (line 57), log a WARNING at module load time:

```python
except Exception as exc:
    _IMPORT_ERROR = str(exc)
    logger.warning("predictive_challenge_imports_failed", extra={"error": str(exc)[:200]})
    # ... rest of fallbacks
```

### Phase 2: Observability Integration

Wire the PSE into the TurnEvent system. In `build_predictive_scenario_packet`, accept an optional `TurnEvent` parameter:

```python
from observability import current_event, timed

def build_predictive_scenario_packet(
    user_content: str,
    state: dict,
    paths: list[dict],
    *,
    shadow: dict | None = None,
    user_model: dict | None = None,
    retrieval_packet: dict | None = None,
) -> dict:
    event = current_event()

    brief = build_scenario_brief(...)
    model_selection = select_model_family(brief)

    forecast = {}
    if model_selection["primary_mode"] == "forecast":
        t0 = time.monotonic()
        forecast = _run_forecast_worker(brief)
        if event:
            event.tool_durations_ms["pse_forecast"] = round((time.monotonic() - t0) * 1000, 1)

    causal = {}
    if model_selection["primary_mode"] == "causal":
        t0 = time.monotonic()
        causal = _run_causal_worker(user_content, brief, state, paths)
        if event:
            event.tool_durations_ms["pse_causal"] = round((time.monotonic() - t0) * 1000, 1)
    # ... same for temporal, simulations
```

### Phase 3: Replace sys.path with importlib

Replace the 6-directory `sys.path.insert` block with importlib-based loading:

```python
import importlib.util

def _import_from_challenge(module_name: str, challenge_dir: str):
    """Import a module from a challenge directory without polluting sys.path."""
    module_path = Path(challenge_dir) / f"{module_name}.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

This eliminates sys.path pollution and makes import failures traceable to specific modules.

### Phase 4: Degradation Metadata in Return Value

Add a `"degradation"` key to the return dict that tracks which stages succeeded vs fell back:

```python
return {
    "brief": brief,
    "model_selection": model_selection,
    "behavior": behavior,
    "forecast": forecast,
    "causal": causal,
    "temporal": temporal,
    "simulations": simulations,
    "summary": summary,
    "degradation": {
        "modules_available": _IMPORT_ERROR is None,
        "forecast_ran": forecast.get("available", False),
        "causal_ran": causal.get("available", False),
        "fallback_count": _count_fallbacks(forecast, causal, behavior),
    },
}
```

This makes degradation visible to the cognitive agents and the observability layer.

### Phase 5: Extract Magic Numbers to Agent ZeroConfig

Add to `config.py`:

```python
# Predictive Scenario Engine
pse_simulation_draws: int = 400
pse_progress_threshold: float = 0.58
pse_setback_threshold: float = 0.34
pse_success_clamp_min: float = 0.18
pse_success_clamp_max: float = 0.82
pse_concentration_low: float = 6.0
pse_concentration_medium: float = 10.0
pse_concentration_high: float = 16.0
```

### Phase 6: Test Coverage

Create `agent_zero/test_predictive_scenario_engine_reliability.py` with edge case tests.

## Test Specifications

### Error Handling Tests

```python
def test_forecast_worker_logs_on_failure():
    """_run_forecast_worker logs a warning when backtesting raises."""
    # Provide a brief with a very short series (length=2) that causes backtest to fail
    # Verify logger.warning was called with "forecast_backtest_failed"
    # Verify the return dict has available=True (degraded but not broken)

def test_causal_worker_logs_on_estimation_failure():
    """_run_causal_worker logs when doubly_robust estimation fails."""
    # Provide malformed structured_dataset
    # Verify logger.warning was called with "causal_effect_estimation_failed"
    # Verify effect_estimate is None in return dict

def test_import_failure_logged_at_module_load():
    """When challenge modules are unavailable, a warning is logged."""
    # Mock sys.path to exclude challenge dirs
    # Reimport predictive_scenario_engine
    # Verify _IMPORT_ERROR is set and logger.warning was called
```

### Degradation Tests

```python
def test_degradation_metadata_all_available():
    """When all modules load, degradation shows no fallbacks."""
    # Run build_predictive_scenario_packet with valid inputs
    # Assert result["degradation"]["modules_available"] is True
    # Assert result["degradation"]["fallback_count"] == 0

def test_degradation_metadata_modules_unavailable():
    """When modules fail to import, degradation is tracked."""
    # Mock RandomSampler = None
    # Run build_predictive_scenario_packet
    # Assert result["degradation"]["modules_available"] is False
    # Assert simulations list is empty

def test_degradation_with_short_series():
    """Short time series triggers forecast fallback, tracked in degradation."""
    # Provide series of length 3 (too short for backtesting)
    # Assert forecast fell back to default model
    # Assert degradation shows forecast ran but with warnings
```

### Observability Tests

```python
def test_forecast_timing_recorded_in_turn_event():
    """Forecast worker timing is recorded in TurnEvent.tool_durations_ms."""
    # Set up a TurnEvent via observe_turn context manager
    # Run build_predictive_scenario_packet
    # Assert event.tool_durations_ms contains "pse_forecast" key
    # Assert value is > 0

def test_causal_timing_recorded_in_turn_event():
    """Causal worker timing is recorded when mode is causal."""
    # Similar to above for causal mode
```

### Import Safety Tests

```python
def test_importlib_loading_isolates_modules():
    """Modules loaded via importlib don't pollute sys.path."""
    # Record sys.path length before import
    # Use _import_from_challenge to load a module
    # Assert sys.path length unchanged

def test_importlib_handles_missing_module():
    """_import_from_challenge returns None for nonexistent module."""
    # Call with nonexistent challenge dir
    # Assert returns None, no exception raised
```

### Numeric Edge Cases

```python
def test_simulate_path_outcomes_empty_paths():
    """Empty paths list produces empty simulations."""

def test_simulate_path_outcomes_nan_in_series():
    """NaN values in time series are handled without crash."""

def test_forecast_worker_single_datapoint():
    """Series of length 1 doesn't crash the forecast worker."""

def test_beta_distribution_extreme_mean():
    """Mean near 0 or 1 still produces valid alpha/beta > 1.1."""
```

## Estimated Impact

- **Visibility**: Structured logging makes PSE failures traceable in production logs. Currently, 100% of PSE failures are invisible.
- **Diagnostics**: Observability integration enables identifying which PSE stages are slow or failing, enabling targeted optimization.
- **Import safety**: Removing sys.path manipulation eliminates a class of subtle import bugs (name shadowing, stale caches).
- **Configurability**: Extracting magic numbers to Agent ZeroConfig enables tuning simulation parameters without code changes.
- **Reliability**: Test coverage catches regressions in the numeric pipeline before they reach production.
- **User trust**: Degradation metadata allows cognitive agents to adjust their confidence and transparency when giving scenario-based advice.
