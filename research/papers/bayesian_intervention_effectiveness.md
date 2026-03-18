---
topic: Bayesian Intervention Effectiveness Estimation
status: ready_for_implementation
priority: high
estimated_complexity: small
researched_at: 2026-03-18T14:30:00Z
---

# Bayesian Intervention Effectiveness Estimation

## Problem Statement

Agent Zero's intervention tracker calculates effectiveness using raw count ratios: `hit_rate = acted_count / resolved_count`. With small sample sizes (typical for new users or rare topics), this produces wildly unstable estimates. A user with 2 resolved interventions where 1 was acted shows 50% hit rate, identical confidence to a user with 200 resolved where 100 were acted. The system makes style recommendations and adjusts intervention strategy based on these unreliable point estimates, leading to premature strategy shifts and oscillating recommendations.

## Current State in Agent Zero

### intervention_tracker.py (lines 248-250)
```python
"hit_rate": round(counts.get("acted", 0) / denom, 2) if resolved_count else 0.0,
"ignore_rate": round(counts.get("ignored", 0) / denom, 2) if resolved_count else 0.0,
"fail_rate": round(counts.get("pushed_back", 0) / denom, 2) if resolved_count else 0.0,
```
These are maximum likelihood point estimates with no uncertainty quantification.

### intervention_tracker.py (lines 369-371)
```python
if top_success_style and acted_count >= max(ignored_count, pushed_back_count):
    return top_success_style
if pushed_back_count / float(resolved_count) >= 0.34:
    return "low-pressure probe"
```
Threshold-based decisions using raw ratios. With 3 resolved interventions and 2 pushbacks (67%), the system immediately switches to "low-pressure probe" despite high uncertainty.

### intervention_tracker.py (lines 392-393)
```python
if ignored_count / float(resolved_count) > 0.4:
    suggestions.append("Reduce pressure or narrow the next step before escalating.")
```
Same pattern: raw ratio comparison with no sample size consideration.

### behavioral_insights.py (line 32)
```python
if consistency >= 0.7:  # regime classification threshold
```
Hardcoded thresholds applied to small-sample estimates.

## Industry Standard / Research Findings

### 1. Beta-Binomial Model for Behavioral Interventions (Wen et al., 2024)
The beta-binomial model is the standard Bayesian approach for binary outcome data (acted vs not-acted) in behavioral research. It naturally handles overdispersion and small samples by maintaining a posterior distribution rather than a point estimate.

Prior: Beta(alpha_0, beta_0) -- typically Beta(1,1) uniform or Beta(2,2) weakly informative
Likelihood: Binomial(n, p)
Posterior: Beta(alpha_0 + successes, beta_0 + failures)

The posterior mean `(alpha_0 + s) / (alpha_0 + beta_0 + n)` naturally shrinks toward the prior with small samples, preventing extreme estimates.

Reference: Wen, C. et al. (2024). "A Bayesian beta-binomial piecewise growth mixture model for longitudinal overdispersed binomial data." Statistical Methods in Medical Research. https://journals.sagepub.com/doi/10.1177/09622802241279109

### 2. Bayesian Evaluation of Behavior Change Interventions (van de Schoot et al., 2018)
Demonstrates that Bayesian methods perform well with small sample sizes, which is critical for health psychology and behavioral coaching. Even with "non-informed" (uniform) priors, Bayesian estimators outperform frequentist point estimates in small samples. Key recommendation: use credible intervals rather than point estimates for decision-making.

Reference: van de Schoot, R. et al. (2018). "Bayesian evaluation of behavior change interventions: a brief introduction and a practical example." Health Psychology and Behavioral Medicine. https://www.tandfonline.com/doi/full/10.1080/21642850.2018.1428102

### 3. Bayesian Design for Behavioral Intervention Studies (Bulus & Tong, 2026)
Recent work on a priori sample size determination in the Bayesian hypothesis testing framework for behavioral interventions. Uses Approximate Adjusted Fractional Bayes Factors with default priors from minimal training samples. Key insight: probability of the Bayes factor exceeding a threshold should reach target level (eta=0.80) before acting on estimates.

Reference: Bulus, M. & Tong, X. (2026). "Design and analysis of behavioral intervention studies: A Bayesian approach." PLOS ONE. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0342163

### 4. Thompson Sampling for Model Selection (SourcePilot, 2025)
Production implementation of Beta-distribution-based effectiveness tracking for AI model selection. Tracks alpha (successes) and beta (failures) per model, samples from Beta(alpha, beta) to make probabilistic selections. Starts with Beta(1,1) uniform prior, updates after each interaction.

Reference: SourcePilot. (2025). "How Thompson Sampling Works: The Algorithm Behind Smart AI Selection." https://sourcepilot.co/blog/2025/11/22/how-thompson-sampling-works

### 5. Bayesian Sample Size Estimation (Gao, 2025)
Hybrid frequentist-Bayesian approach for improving effectiveness of sample size re-estimation. Demonstrates that Bayesian methods with informative priors from historical data outperform pure frequentist designs, especially when interim data is limited.

Reference: Gao, L. (2025). "Improving the Effectiveness of Sample Size Re-Estimation: An Operating Characteristic Focused, Hybrid Frequentist-Bayesian Approach." Statistics in Medicine. https://onlinelibrary.wiley.com/doi/full/10.1002/sim.10310

### 6. Effective Sample Size of Parametric Priors (Morita, Thall, Mueller, 2008)
Foundational work on determining how much information a Bayesian prior contributes. The effective sample size of a Beta(a,b) prior is a+b, meaning Beta(2,2) contributes the equivalent of 4 observations. This provides a principled way to set prior strength: match it to the amount of "pseudo-data" you want to start with.

Reference: Morita, S., Thall, P.F., Mueller, P. (2008). "Determining the Effective Sample Size of a Parametric Prior." Biometrics. https://web.ma.utexas.edu/users/pmueller/pap/MTM08.pdf

## Proposed Implementation

### New file: `agent_zero/bayesian_rates.py`

```python
"""Bayesian effectiveness estimation using Beta-Binomial model."""
import math


def beta_posterior(
    successes: int,
    failures: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict:
    """Compute Beta posterior statistics.

    Returns posterior mean, credible interval, and effective sample size.
    """
    alpha = alpha_prior + successes
    beta = beta_prior + failures
    n_eff = alpha + beta  # effective sample size

    mean = alpha / n_eff
    # 90% credible interval via normal approximation (good for alpha,beta > 2)
    variance = (alpha * beta) / (n_eff ** 2 * (n_eff + 1))
    std = math.sqrt(variance)
    ci_low = max(0.0, mean - 1.645 * std)
    ci_high = min(1.0, mean + 1.645 * std)

    return {
        "mean": round(mean, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "ci_width": round(ci_high - ci_low, 3),
        "n_eff": round(n_eff, 1),
        "n_observed": successes + failures,
        "confidence": "high" if n_eff >= 20 else ("medium" if n_eff >= 8 else "low"),
    }


def is_reliable(posterior: dict, min_n_eff: int = 8, max_ci_width: float = 0.4) -> bool:
    """Check if the estimate is reliable enough for strategy decisions."""
    return posterior["n_eff"] >= min_n_eff and posterior["ci_width"] <= max_ci_width


def compare_styles(style_outcomes: dict[str, tuple[int, int]]) -> list[dict]:
    """Compare intervention styles using Bayesian posteriors.

    style_outcomes: {"gentle_probe": (acted, not_acted), "direct_challenge": (acted, not_acted)}
    Returns sorted list of styles with posterior statistics.
    """
    results = []
    for style, (successes, failures) in style_outcomes.items():
        posterior = beta_posterior(successes, failures)
        results.append({"style": style, **posterior})
    results.sort(key=lambda x: -x["mean"])
    return results
```

### Modify: `agent_zero/intervention_tracker.py`

**In `summarize_interventions()` (lines 239-296):**

Replace raw ratio calculations with Bayesian posteriors:

```python
from bayesian_rates import beta_posterior, is_reliable

# Replace lines 248-250:
acted_count = counts.get("acted", 0)
not_acted = resolved_count - acted_count
hit_posterior = beta_posterior(acted_count, not_acted)
ignore_posterior = beta_posterior(counts.get("ignored", 0), resolved_count - counts.get("ignored", 0))
fail_posterior = beta_posterior(counts.get("pushed_back", 0), resolved_count - counts.get("pushed_back", 0))

summary.update({
    "hit_rate": hit_posterior["mean"],
    "ignore_rate": ignore_posterior["mean"],
    "fail_rate": fail_posterior["mean"],
    "hit_rate_ci": [hit_posterior["ci_low"], hit_posterior["ci_high"]],
    "rate_confidence": hit_posterior["confidence"],
    # ... existing fields preserved
})
```

**In `_derive_recommended_style()` (lines 358-377):**

Add sample-size guard:
```python
def _derive_recommended_style(...):
    if resolved_count == 0:
        return ""
    # NEW: don't make strategy shifts until we have enough data
    n_eff = 2 + resolved_count  # Beta(1,1) prior + observations
    if n_eff < 8:
        return top_success_style or ""  # keep current style, don't switch
    # ... existing logic with Bayesian rates instead of raw ratios
```

**In `_build_suggestions()` (lines 380-402):**

Replace raw ratio thresholds with credible interval checks:
```python
if resolved_count:
    ignore_post = beta_posterior(ignored_count, resolved_count - ignored_count)
    pushback_post = beta_posterior(pushed_back_count, resolved_count - pushed_back_count)
    if ignore_post["ci_low"] > 0.3:  # Even lower bound of CI is high
        suggestions.append("Reduce pressure or narrow the next step before escalating.")
    if pushback_post["ci_low"] > 0.2:
        suggestions.append("The user is pushing back on current framing; soften the intervention style.")
```

## Test Specifications

### test_bayesian_rates.py

```
test_beta_posterior_uniform_prior -- Beta(1,1) prior with 0 observations gives mean=0.5
test_beta_posterior_with_data -- 8 successes, 2 failures gives mean close to 0.75
test_beta_posterior_small_sample_shrinkage -- 1 success, 0 failures gives mean < 1.0 (shrunk toward prior)
test_beta_posterior_large_sample_converges -- 100 successes, 0 failures gives mean close to 1.0
test_beta_posterior_ci_narrows_with_data -- CI width decreases as n increases
test_beta_posterior_ci_bounds -- CI always within [0, 1]
test_beta_posterior_confidence_levels -- low/medium/high at n_eff thresholds 8, 20
test_is_reliable_low_n -- n_eff < 8 returns False
test_is_reliable_wide_ci -- CI width > 0.4 returns False
test_is_reliable_good_estimate -- sufficient n_eff and narrow CI returns True
test_compare_styles_ranking -- styles sorted by posterior mean descending
test_compare_styles_empty -- empty dict returns empty list
test_compare_styles_with_uncertainty -- low-data style has wider CI than high-data style

test_summarize_interventions_uses_bayesian -- hit_rate is posterior mean, not raw ratio
test_summarize_interventions_ci_present -- hit_rate_ci field present in summary
test_summarize_interventions_confidence_field -- rate_confidence reflects sample size
test_derive_style_guards_small_sample -- < 6 observations doesn't switch style
test_suggestions_use_credible_intervals -- suggestions triggered by CI lower bound, not point estimate
test_backward_compat_hit_rate_field -- hit_rate field still present and numeric
```

## Estimated Impact

- **Stability**: New users (< 6 interventions) will not receive oscillating style recommendations. The system will maintain a stable default until enough evidence accumulates.
- **Accuracy**: With 20+ resolved interventions, Bayesian estimates converge to MLE, so no loss for high-data users.
- **Transparency**: The `rate_confidence` field lets agents know when to trust the effectiveness data vs. when to maintain default behavior.
- **Zero dependencies**: Pure Python math (Beta distribution via alpha/beta arithmetic). No scipy/numpy needed for the simple normal approximation.
- **Quantified uncertainty**: Credible intervals give agents a principled way to decide when to act on effectiveness data vs. when to explore.
