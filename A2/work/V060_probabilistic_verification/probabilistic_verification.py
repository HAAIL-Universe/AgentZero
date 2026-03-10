"""
V060: Probabilistic Verification -- Statistical Model Checking

Composes:
- V054 (verification-driven fuzzing) -- mutation engine, test generation
- V001 (guided symbolic execution) -- symbolic path exploration
- C010 (parser/VM) -- concrete execution
- C037 (SMT solver) -- constraint solving

Statistical model checking approach:
1. Define stochastic properties (probability of assertion holding, expected bounds)
2. Sample concrete executions via random/symbolic input generation
3. Apply hypothesis testing (Wald's SPRT, Chernoff-Hoeffding bounds)
4. Estimate probability of property satisfaction with confidence intervals

This is the first tool that gives QUANTITATIVE answers:
  "P(assertion holds) >= 0.95 with 99% confidence"
instead of just "holds" or "violated".
"""

import os, sys, math, random, time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Set, Callable

# Path setup
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
_challenges = os.path.join(_az, "challenges")

for p in [
    os.path.join(_work, "V054_verification_driven_fuzzing"),
    os.path.join(_work, "V001_guided_symbolic_execution"),
    os.path.join(_work, "V028_fault_localization"),
    os.path.join(_work, "V018_concolic_testing"),
    os.path.join(_challenges, "C010_stack_vm"),
    os.path.join(_challenges, "C037_smt_solver"),
    os.path.join(_challenges, "C038_symbolic_execution"),
    os.path.join(_challenges, "C039_abstract_interpreter"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stack_vm import lex, Parser, Compiler, VM


# =============================================================================
# Data Model
# =============================================================================

class StatVerdict(Enum):
    ACCEPT = "accept"       # Property likely holds
    REJECT = "reject"       # Property likely violated
    INCONCLUSIVE = "inconclusive"  # Not enough samples


class PropertyKind(Enum):
    PROBABILITY_GE = "probability_ge"    # P(prop) >= threshold
    PROBABILITY_LE = "probability_le"    # P(prop) <= threshold
    EXPECTED_BOUND = "expected_bound"    # E[expr] in [lo, hi]
    QUANTILE = "quantile"               # P(expr <= bound) >= q


@dataclass
class StatProperty:
    """A stochastic property to check."""
    kind: PropertyKind
    description: str
    threshold: float = 0.0        # For probability properties
    bound_lo: float = float('-inf')  # For expected bound
    bound_hi: float = float('inf')
    quantile: float = 0.95        # For quantile properties


@dataclass
class SampleResult:
    """Result of a single concrete execution."""
    inputs: Dict[str, int]
    output: Any
    passed: bool             # Did the property hold for this sample?
    value: float = 0.0       # Numeric value (for expected bound/quantile)
    error: Optional[str] = None


@dataclass
class StatCheckResult:
    """Result of statistical model checking."""
    verdict: StatVerdict
    property: StatProperty
    total_samples: int
    passing_samples: int
    estimated_probability: float
    confidence: float            # Confidence level (e.g., 0.99)
    confidence_interval: Tuple[float, float]  # (lower, upper) bound on probability
    samples: List[SampleResult] = field(default_factory=list)
    sprt_log_ratio: Optional[float] = None   # For SPRT
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def failing_samples(self) -> int:
        return self.total_samples - self.passing_samples

    def summary(self) -> str:
        ci_lo, ci_hi = self.confidence_interval
        lines = [
            f"StatCheck: {self.verdict.value}",
            f"  Property: {self.property.description}",
            f"  Samples: {self.total_samples} ({self.passing_samples} pass, {self.failing_samples} fail)",
            f"  P(prop) ~ {self.estimated_probability:.4f}",
            f"  {self.confidence*100:.0f}% CI: [{ci_lo:.4f}, {ci_hi:.4f}]",
        ]
        return "\n".join(lines)


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo probability estimation."""
    estimated_probability: float
    confidence_interval: Tuple[float, float]
    total_samples: int
    passing_samples: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Concrete Execution Engine
# =============================================================================

class ProbabilisticExecutor:
    """Execute C10 programs with random inputs and collect statistics."""

    def __init__(self, source: str, input_vars: List[str],
                 input_ranges: Optional[Dict[str, Tuple[int, int]]] = None):
        self.source = source
        self.input_vars = input_vars
        self.input_ranges = input_ranges or {}
        # Default range for unspecified vars
        self.default_range = (-100, 100)

    def generate_random_inputs(self) -> Dict[str, int]:
        """Generate random input values within specified ranges."""
        inputs = {}
        for var in self.input_vars:
            lo, hi = self.input_ranges.get(var, self.default_range)
            inputs[var] = random.randint(lo, hi)
        return inputs

    def execute_with_inputs(self, inputs: Dict[str, int]) -> Tuple[Any, Optional[str]]:
        """Execute the program with given inputs, return (result, error_or_None)."""
        # Build source with input assignments prepended
        input_lines = [f"let {var} = {val};" for var, val in inputs.items()]
        full_source = "\n".join(input_lines) + "\n" + self.source

        try:
            tokens = lex(full_source)
            ast = Parser(tokens).parse()
            compiler = Compiler()
            chunk = compiler.compile(ast)
            vm = VM(chunk)
            result = vm.run()
            return result, None
        except Exception as e:
            return None, str(e)

    def sample(self, oracle: Callable[[Dict[str, int], Any, Optional[str]], bool],
               value_fn: Optional[Callable[[Dict[str, int], Any], float]] = None,
               inputs: Optional[Dict[str, int]] = None) -> SampleResult:
        """Run one sample and evaluate the oracle."""
        if inputs is None:
            inputs = self.generate_random_inputs()
        result, error = self.execute_with_inputs(inputs)
        passed = oracle(inputs, result, error)
        value = 0.0
        if value_fn is not None and error is None:
            try:
                value = value_fn(inputs, result)
            except Exception:
                pass
        return SampleResult(
            inputs=inputs, output=result, passed=passed,
            value=value, error=error,
        )


# =============================================================================
# Statistical Tests
# =============================================================================

def wilson_confidence_interval(n: int, k: int, confidence: float) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    # z-score for confidence level (approximate)
    z = _z_score(confidence)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _z_score(confidence: float) -> float:
    """Approximate z-score for common confidence levels."""
    # Common values
    if confidence >= 0.999:
        return 3.291
    elif confidence >= 0.995:
        return 2.807
    elif confidence >= 0.99:
        return 2.576
    elif confidence >= 0.975:
        return 2.241
    elif confidence >= 0.95:
        return 1.960
    elif confidence >= 0.90:
        return 1.645
    else:
        return 1.282  # 80%


def chernoff_hoeffding_samples(epsilon: float, delta: float) -> int:
    """Minimum samples for Chernoff-Hoeffding bound.

    To estimate P within epsilon with probability >= 1-delta:
    n >= ln(2/delta) / (2 * epsilon^2)
    """
    if epsilon <= 0 or delta <= 0:
        return 10000
    return math.ceil(math.log(2.0 / delta) / (2 * epsilon * epsilon))


def sprt_test(
    samples: List[bool],
    p0: float,
    p1: float,
    alpha: float = 0.01,
    beta: float = 0.01,
) -> Tuple[StatVerdict, float]:
    """Sequential Probability Ratio Test (Wald's SPRT).

    H0: P(pass) >= p0  (property holds with high probability)
    H1: P(pass) <= p1  (property violated)

    Returns (verdict, log_likelihood_ratio).
    """
    if p0 <= p1:
        return StatVerdict.INCONCLUSIVE, 0.0
    if p0 <= 0 or p0 >= 1 or p1 <= 0 or p1 >= 1:
        return StatVerdict.INCONCLUSIVE, 0.0

    log_A = math.log((1 - beta) / alpha)    # Accept H0 threshold
    log_B = math.log(beta / (1 - alpha))    # Accept H1 threshold

    log_ratio = 0.0
    for s in samples:
        if s:  # pass
            log_ratio += math.log(p1 / p0)
        else:  # fail
            log_ratio += math.log((1 - p1) / (1 - p0))

        if log_ratio <= log_B:
            return StatVerdict.ACCEPT, log_ratio   # Accept H0 (property holds)
        elif log_ratio >= log_A:
            return StatVerdict.REJECT, log_ratio    # Accept H1 (property violated)

    return StatVerdict.INCONCLUSIVE, log_ratio


# =============================================================================
# Main API: Statistical Model Checking
# =============================================================================

def stat_check(
    source: str,
    input_vars: List[str],
    oracle: Callable[[Dict[str, int], Any, Optional[str]], bool],
    prop: StatProperty,
    confidence: float = 0.99,
    max_samples: int = 1000,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    value_fn: Optional[Callable[[Dict[str, int], Any], float]] = None,
    seed: Optional[int] = None,
) -> StatCheckResult:
    """
    Statistical model checking: estimate probability of property with confidence.

    Args:
        source: C10 source code (without input variable declarations)
        input_vars: Variable names treated as random inputs
        oracle: (inputs, result, error) -> bool (does this sample satisfy the property?)
        prop: Statistical property to check
        confidence: Desired confidence level (e.g., 0.99)
        max_samples: Maximum number of samples
        input_ranges: Per-variable (min, max) ranges
        value_fn: Optional function to extract numeric value from execution
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    executor = ProbabilisticExecutor(source, input_vars, input_ranges)
    samples = []
    pass_count = 0

    for _ in range(max_samples):
        sample = executor.sample(oracle, value_fn)
        samples.append(sample)
        if sample.passed:
            pass_count += 1

    # Compute statistics
    n = len(samples)
    p_hat = pass_count / n if n > 0 else 0.0
    ci = wilson_confidence_interval(n, pass_count, confidence)

    # Determine verdict based on property kind
    verdict = _evaluate_verdict(prop, p_hat, ci, samples)

    return StatCheckResult(
        verdict=verdict,
        property=prop,
        total_samples=n,
        passing_samples=pass_count,
        estimated_probability=p_hat,
        confidence=confidence,
        confidence_interval=ci,
        samples=samples[:50],  # Keep first 50 for inspection
        metadata={"seed": seed},
    )


def stat_check_sprt(
    source: str,
    input_vars: List[str],
    oracle: Callable[[Dict[str, int], Any, Optional[str]], bool],
    p0: float = 0.95,
    p1: float = 0.90,
    alpha: float = 0.01,
    beta: float = 0.01,
    max_samples: int = 5000,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> StatCheckResult:
    """
    SPRT-based statistical model checking.

    Tests H0: P(oracle) >= p0 vs H1: P(oracle) <= p1.
    Terminates early when evidence is sufficient (sequential test).

    Args:
        p0: Null hypothesis probability (property holds with at least this probability)
        p1: Alternative hypothesis probability (property holds with at most this probability)
        alpha: Type I error rate (false reject)
        beta: Type II error rate (false accept)
    """
    if seed is not None:
        random.seed(seed)

    executor = ProbabilisticExecutor(source, input_vars, input_ranges)
    sample_bools = []
    samples = []
    pass_count = 0

    for _ in range(max_samples):
        sample = executor.sample(oracle)
        samples.append(sample)
        sample_bools.append(sample.passed)
        if sample.passed:
            pass_count += 1

        # Run SPRT after each sample
        verdict, log_ratio = sprt_test(sample_bools, p0, p1, alpha, beta)
        if verdict != StatVerdict.INCONCLUSIVE:
            n = len(samples)
            p_hat = pass_count / n
            ci = wilson_confidence_interval(n, pass_count, 1 - max(alpha, beta))
            prop = StatProperty(
                kind=PropertyKind.PROBABILITY_GE,
                description=f"P(oracle) >= {p0}",
                threshold=p0,
            )
            return StatCheckResult(
                verdict=verdict,
                property=prop,
                total_samples=n,
                passing_samples=pass_count,
                estimated_probability=p_hat,
                confidence=1 - max(alpha, beta),
                confidence_interval=ci,
                samples=samples[:50],
                sprt_log_ratio=log_ratio,
                metadata={"p0": p0, "p1": p1, "alpha": alpha, "beta": beta,
                          "early_termination": True},
            )

    # Exhausted samples
    n = len(samples)
    p_hat = pass_count / n if n > 0 else 0.0
    ci = wilson_confidence_interval(n, pass_count, 1 - max(alpha, beta))
    _, log_ratio = sprt_test(sample_bools, p0, p1, alpha, beta)
    prop = StatProperty(
        kind=PropertyKind.PROBABILITY_GE,
        description=f"P(oracle) >= {p0}",
        threshold=p0,
    )
    return StatCheckResult(
        verdict=StatVerdict.INCONCLUSIVE,
        property=prop,
        total_samples=n,
        passing_samples=pass_count,
        estimated_probability=p_hat,
        confidence=1 - max(alpha, beta),
        confidence_interval=ci,
        samples=samples[:50],
        sprt_log_ratio=log_ratio,
        metadata={"p0": p0, "p1": p1, "alpha": alpha, "beta": beta,
                  "early_termination": False},
    )


def monte_carlo_estimate(
    source: str,
    input_vars: List[str],
    oracle: Callable[[Dict[str, int], Any, Optional[str]], bool],
    n_samples: int = 1000,
    confidence: float = 0.99,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Simple Monte Carlo probability estimation with confidence interval."""
    if seed is not None:
        random.seed(seed)

    executor = ProbabilisticExecutor(source, input_vars, input_ranges)
    pass_count = 0

    for _ in range(n_samples):
        sample = executor.sample(oracle)
        if sample.passed:
            pass_count += 1

    p_hat = pass_count / n_samples if n_samples > 0 else 0.0
    ci = wilson_confidence_interval(n_samples, pass_count, confidence)

    return MonteCarloResult(
        estimated_probability=p_hat,
        confidence_interval=ci,
        total_samples=n_samples,
        passing_samples=pass_count,
        confidence=confidence,
    )


def expected_value_check(
    source: str,
    input_vars: List[str],
    value_fn: Callable[[Dict[str, int], Any], float],
    bound_lo: float = float('-inf'),
    bound_hi: float = float('inf'),
    n_samples: int = 500,
    confidence: float = 0.95,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> StatCheckResult:
    """Check if E[value_fn(inputs, result)] is within [bound_lo, bound_hi]."""
    if seed is not None:
        random.seed(seed)

    executor = ProbabilisticExecutor(source, input_vars, input_ranges)
    values = []

    def oracle(inputs, result, error):
        if error:
            return False
        try:
            v = value_fn(inputs, result)
            values.append(v)
            return bound_lo <= v <= bound_hi
        except Exception:
            return False

    samples = []
    pass_count = 0
    for _ in range(n_samples):
        sample = executor.sample(oracle, value_fn)
        samples.append(sample)
        if sample.passed:
            pass_count += 1

    n = len(samples)
    p_hat = pass_count / n if n > 0 else 0.0

    # Compute mean and std of values
    if values:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        std = math.sqrt(variance)
        z = _z_score(confidence)
        margin = z * std / math.sqrt(len(values))
        mean_ci = (mean - margin, mean + margin)
    else:
        mean = 0.0
        mean_ci = (0.0, 0.0)

    ci = wilson_confidence_interval(n, pass_count, confidence)

    in_bounds = bound_lo <= mean <= bound_hi
    verdict = StatVerdict.ACCEPT if in_bounds else StatVerdict.REJECT

    prop = StatProperty(
        kind=PropertyKind.EXPECTED_BOUND,
        description=f"E[value] in [{bound_lo}, {bound_hi}]",
        bound_lo=bound_lo,
        bound_hi=bound_hi,
    )

    return StatCheckResult(
        verdict=verdict,
        property=prop,
        total_samples=n,
        passing_samples=pass_count,
        estimated_probability=p_hat,
        confidence=confidence,
        confidence_interval=ci,
        samples=samples[:50],
        metadata={"mean": mean, "mean_ci": mean_ci, "seed": seed},
    )


def required_samples(epsilon: float = 0.01, delta: float = 0.01) -> int:
    """Compute minimum samples for given accuracy and confidence.

    Returns n such that P(|p_hat - p| > epsilon) <= delta.
    """
    return chernoff_hoeffding_samples(epsilon, delta)


# =============================================================================
# Convenience API
# =============================================================================

def check_assertion_probability(
    source: str,
    input_vars: List[str],
    threshold: float = 0.95,
    confidence: float = 0.99,
    n_samples: int = 500,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> StatCheckResult:
    """Check P(no runtime error) >= threshold.

    Useful for programs with assertions or division-by-zero risks.
    """
    def oracle(inputs, result, error):
        return error is None

    prop = StatProperty(
        kind=PropertyKind.PROBABILITY_GE,
        description=f"P(no error) >= {threshold}",
        threshold=threshold,
    )

    return stat_check(
        source, input_vars, oracle, prop,
        confidence=confidence, max_samples=n_samples,
        input_ranges=input_ranges, seed=seed,
    )


def check_output_probability(
    source: str,
    input_vars: List[str],
    output_predicate: Callable[[Any], bool],
    threshold: float = 0.95,
    confidence: float = 0.99,
    n_samples: int = 500,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> StatCheckResult:
    """Check P(output_predicate(result)) >= threshold."""
    def oracle(inputs, result, error):
        if error:
            return False
        try:
            return output_predicate(result)
        except Exception:
            return False

    prop = StatProperty(
        kind=PropertyKind.PROBABILITY_GE,
        description=f"P(output predicate) >= {threshold}",
        threshold=threshold,
    )

    return stat_check(
        source, input_vars, oracle, prop,
        confidence=confidence, max_samples=n_samples,
        input_ranges=input_ranges, seed=seed,
    )


def compare_statistical_vs_exact(
    source: str,
    input_vars: List[str],
    oracle: Callable[[Dict[str, int], Any, Optional[str]], bool],
    n_samples: int = 500,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare Monte Carlo estimate vs SPRT on the same program."""
    mc = monte_carlo_estimate(
        source, input_vars, oracle,
        n_samples=n_samples, confidence=0.99,
        input_ranges=input_ranges, seed=seed,
    )

    sprt = stat_check_sprt(
        source, input_vars, oracle,
        p0=0.95, p1=0.85,
        max_samples=n_samples,
        input_ranges=input_ranges,
        seed=seed,
    )

    return {
        "monte_carlo": {
            "probability": mc.estimated_probability,
            "ci": mc.confidence_interval,
            "samples": mc.total_samples,
        },
        "sprt": {
            "verdict": sprt.verdict.value,
            "probability": sprt.estimated_probability,
            "samples": sprt.total_samples,
            "early_termination": sprt.metadata.get("early_termination", False),
            "log_ratio": sprt.sprt_log_ratio,
        },
    }


# =============================================================================
# Helpers
# =============================================================================

def _evaluate_verdict(prop: StatProperty, p_hat: float,
                      ci: Tuple[float, float], samples: List[SampleResult]) -> StatVerdict:
    """Evaluate verdict based on property kind and statistics."""
    ci_lo, ci_hi = ci

    if prop.kind == PropertyKind.PROBABILITY_GE:
        # Accept if lower CI bound >= threshold
        if ci_lo >= prop.threshold:
            return StatVerdict.ACCEPT
        elif ci_hi < prop.threshold:
            return StatVerdict.REJECT
        else:
            return StatVerdict.INCONCLUSIVE

    elif prop.kind == PropertyKind.PROBABILITY_LE:
        if ci_hi <= prop.threshold:
            return StatVerdict.ACCEPT
        elif ci_lo > prop.threshold:
            return StatVerdict.REJECT
        else:
            return StatVerdict.INCONCLUSIVE

    elif prop.kind == PropertyKind.EXPECTED_BOUND:
        # Use p_hat as proxy (fraction within bounds)
        if p_hat >= 0.95:
            return StatVerdict.ACCEPT
        elif p_hat < 0.5:
            return StatVerdict.REJECT
        else:
            return StatVerdict.INCONCLUSIVE

    elif prop.kind == PropertyKind.QUANTILE:
        if p_hat >= prop.quantile:
            return StatVerdict.ACCEPT
        else:
            return StatVerdict.REJECT

    return StatVerdict.INCONCLUSIVE
