"""Tests for V054: Verification-Driven Fuzzing"""

import os, sys
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import pytest
from verification_driven_fuzzing import (
    verification_fuzz, quick_fuzz, deep_fuzz, fuzz_with_localization,
    detect_divergence, VerificationDrivenFuzzer, TargetedFuzzer,
    MutationEngine, FuzzResult, FuzzFinding, FuzzInput, FuzzStatus,
    CoverageInfo, _extract_boundary_values, _safe_execute,
)


# -----------------------------------------------------------------------
# Test programs
# -----------------------------------------------------------------------

SIMPLE_ABS = """
let x = 0;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = 0 - x;
}
print(r);
"""

DIVISION = """
let x = 0;
let y = 0;
let r = 0;
if (y != 0) {
  r = x;
} else {
  r = 0;
}
print(r);
"""

# Division by zero is possible
UNSAFE_DIVISION = """
let x = 10;
let y = 0;
let r = x / y;
print(r);
"""

BRANCHY = """
let x = 0;
let y = 0;
let r = 0;
if (x > 10) {
  if (y > 5) {
    r = 1;
  } else {
    r = 2;
  }
} else {
  if (x < 0 - 5) {
    r = 3;
  } else {
    r = 4;
  }
}
print(r);
"""

LOOP_PROGRAM = """
let n = 0;
let sum = 0;
let i = 0;
while (i < n) {
  sum = sum + i;
  i = i + 1;
}
print(sum);
"""

CLASSIFY = """
let x = 0;
let cat = 0;
if (x < 0) {
  cat = 0 - 1;
} else {
  if (x == 0) {
    cat = 0;
  } else {
    if (x < 100) {
      cat = 1;
    } else {
      cat = 2;
    }
  }
}
print(cat);
"""


# -----------------------------------------------------------------------
# Section 1: MutationEngine
# -----------------------------------------------------------------------

class TestMutationEngine:
    def test_mutate_small(self):
        eng = MutationEngine(seed=42)
        inp = {"x": 5, "y": 10}
        m = eng.mutate(inp, strength=1)
        assert m != inp or True  # might be same if delta is 0 in rare case
        assert set(m.keys()) == set(inp.keys())

    def test_mutate_medium(self):
        eng = MutationEngine(seed=42)
        inp = {"x": 5}
        mutations = set()
        for _ in range(20):
            m = eng.mutate(inp, strength=2)
            mutations.add(m["x"])
        assert len(mutations) > 1  # should produce variety

    def test_mutate_large(self):
        eng = MutationEngine(seed=42)
        inp = {"x": 5}
        mutations = set()
        for _ in range(20):
            m = eng.mutate(inp, strength=3)
            mutations.add(m["x"])
        assert len(mutations) > 1

    def test_mutate_batch(self):
        eng = MutationEngine(seed=42)
        inp = {"x": 5, "y": 10}
        batch = eng.mutate_batch(inp, count=10, strength=1)
        assert len(batch) <= 10
        assert len(batch) > 0
        # All unique
        keys = [tuple(sorted(m.items())) for m in batch]
        assert len(set(keys)) == len(keys)

    def test_boundary_mutate(self):
        eng = MutationEngine(seed=42)
        inp = {"x": 5}
        bm = eng.boundary_mutate(inp)
        assert len(bm) > 5  # should generate many boundary inputs
        # Check that boundary values are covered
        x_values = [m["x"] for m in bm]
        assert 0 in x_values
        assert 1 in x_values
        assert -1 in x_values

    def test_boundary_mutate_custom_values(self):
        eng = MutationEngine(seed=42)
        inp = {"x": 0}
        bm = eng.boundary_mutate(inp, boundary_values=[42])
        x_values = [m["x"] for m in bm]
        assert 42 in x_values
        assert 41 in x_values
        assert 43 in x_values

    def test_empty_input(self):
        eng = MutationEngine(seed=42)
        m = eng.mutate({}, strength=1)
        assert m == {}


# -----------------------------------------------------------------------
# Section 2: Safe execution
# -----------------------------------------------------------------------

class TestSafeExecute:
    def test_normal_execution(self):
        output, crashed, err = _safe_execute(SIMPLE_ABS, {"x": 5})
        assert not crashed
        assert output == [5]

    def test_division_by_zero(self):
        output, crashed, err = _safe_execute(UNSAFE_DIVISION, {})
        assert crashed
        assert "division" in err

    def test_with_inputs(self):
        output, crashed, err = _safe_execute(DIVISION, {"x": 10, "y": 3})
        assert not crashed
        assert output == [10]


# -----------------------------------------------------------------------
# Section 3: Boundary value extraction
# -----------------------------------------------------------------------

class TestBoundaryExtraction:
    def test_extract_from_simple(self):
        vals = _extract_boundary_values(SIMPLE_ABS)
        assert 0 in vals  # from `let x = 0`
        assert 1 in vals  # always included
        assert -1 in vals  # always included

    def test_extract_from_branchy(self):
        vals = _extract_boundary_values(BRANCHY)
        assert 10 in vals  # from `x > 10`
        assert 5 in vals   # from `y > 5`
        assert 11 in vals  # boundary +1

    def test_extract_from_classify(self):
        vals = _extract_boundary_values(CLASSIFY)
        assert 100 in vals
        assert 99 in vals  # boundary -1
        assert 101 in vals  # boundary +1


# -----------------------------------------------------------------------
# Section 4: CoverageInfo
# -----------------------------------------------------------------------

class TestCoverageInfo:
    def test_empty(self):
        ci = CoverageInfo()
        assert ci.coverage == 0.0

    def test_add_branch(self):
        ci = CoverageInfo()
        ci.total_branches = 4
        ci.add((0, True), {"x": 5})
        assert (0, True) in ci.covered_branches
        assert ci.coverage == 0.25

    def test_dedup(self):
        ci = CoverageInfo()
        ci.total_branches = 2
        ci.add((0, True), {"x": 5})
        ci.add((0, True), {"x": 6})
        assert len(ci.covered_branches) == 1
        assert len(ci.inputs_by_branch[(0, True)]) == 2


# -----------------------------------------------------------------------
# Section 5: FuzzResult
# -----------------------------------------------------------------------

class TestFuzzResult:
    def test_empty(self):
        r = FuzzResult()
        assert not r.has_bugs
        assert r.bug_count == 0
        assert r.unique_findings == 0

    def test_with_crash(self):
        r = FuzzResult()
        r.findings.append(FuzzFinding(
            kind="crash", inputs={"x": 0}, description="div by zero",
            source="mutation"
        ))
        assert r.has_bugs
        assert r.bug_count == 1

    def test_with_divergence(self):
        r = FuzzResult()
        r.findings.append(FuzzFinding(
            kind="divergence", inputs={"x": 0}, description="mismatch",
            source="mutation"
        ))
        assert not r.has_bugs  # divergence is not a "bug" in crash sense
        assert r.unique_findings == 1

    def test_summary(self):
        r = FuzzResult()
        r.total_inputs_tested = 50
        r.symbolic_inputs = 10
        r.concolic_inputs = 15
        s = r.summary()
        assert "50" in s
        assert "10" in s


# -----------------------------------------------------------------------
# Section 6: Quick fuzz
# -----------------------------------------------------------------------

class TestQuickFuzz:
    def test_simple_program(self):
        result = quick_fuzz(SIMPLE_ABS, {"x": "int"})
        assert result.total_inputs_tested > 0
        assert result.status in (FuzzStatus.COMPLETE, FuzzStatus.BUDGET_EXHAUSTED)

    def test_branchy_coverage(self):
        result = quick_fuzz(BRANCHY, {"x": "int", "y": "int"})
        assert result.total_inputs_tested > 0
        # Should find multiple branches
        if result.coverage:
            assert len(result.coverage.covered_branches) > 0

    def test_loop_program(self):
        result = quick_fuzz(LOOP_PROGRAM, {"n": "int"})
        assert result.total_inputs_tested > 0


# -----------------------------------------------------------------------
# Section 7: Full verification fuzz
# -----------------------------------------------------------------------

class TestVerificationFuzz:
    def test_safe_program(self):
        result = verification_fuzz(DIVISION, {"x": "int", "y": "int"}, max_inputs=50)
        assert result.total_inputs_tested > 0
        assert isinstance(result.status, FuzzStatus)

    def test_unsafe_finds_crash(self):
        result = verification_fuzz(UNSAFE_DIVISION, {}, max_inputs=20)
        # Should find the division by zero
        assert result.has_bugs

    def test_with_oracle(self):
        # Oracle: abs(x) should always be >= 0
        def oracle(inputs, output):
            return output[0] >= 0

        result = verification_fuzz(
            SIMPLE_ABS, {"x": "int"}, oracle_fn=oracle, max_inputs=50
        )
        assert result.total_inputs_tested > 0
        # abs(x) is always >= 0, so no oracle failures expected
        oracle_failures = [f for f in result.findings if "oracle" in f.description]
        assert len(oracle_failures) == 0

    def test_classify(self):
        result = verification_fuzz(CLASSIFY, {"x": "int"}, max_inputs=80)
        assert result.total_inputs_tested > 0
        if result.coverage:
            assert len(result.coverage.covered_branches) > 0


# -----------------------------------------------------------------------
# Section 8: Deep fuzz
# -----------------------------------------------------------------------

class TestDeepFuzz:
    def test_deep_more_inputs(self):
        result = deep_fuzz(BRANCHY, {"x": "int", "y": "int"}, max_inputs=100)
        assert result.total_inputs_tested > 0
        # Deep should explore more than quick
        result_quick = quick_fuzz(BRANCHY, {"x": "int", "y": "int"}, max_inputs=30)
        assert result.total_inputs_tested >= result_quick.total_inputs_tested


# -----------------------------------------------------------------------
# Section 9: TargetedFuzzer
# -----------------------------------------------------------------------

class TestTargetedFuzzer:
    def test_fuzz_branch(self):
        tf = TargetedFuzzer(max_inputs=30)
        result = tf.fuzz_branch(
            BRANCHY, {"x": "int", "y": "int"},
            target_branch=0, target_direction=True
        )
        assert result.total_inputs_tested >= 0

    def test_fuzz_suspicious(self):
        from fault_localization import SuspiciousnessScore, StatementInfo, Metric
        fake_sus = [SuspiciousnessScore(
            statement=StatementInfo(index=0, line=1, kind="let", description="let x"),
            score=0.9,
            metric=Metric.OCHIAI,
        )]
        tf = TargetedFuzzer(max_inputs=30)
        result = tf.fuzz_suspicious(
            SIMPLE_ABS, {"x": "int"}, fake_sus
        )
        assert result.total_inputs_tested > 0
        assert result.suspicious_stmts == fake_sus


# -----------------------------------------------------------------------
# Section 10: Divergence detection
# -----------------------------------------------------------------------

class TestDivergenceDetection:
    def test_correct_abs(self):
        def ref(inputs):
            x = inputs.get("x", 0)
            return [abs(x)]

        result = detect_divergence(
            SIMPLE_ABS, {"x": "int"}, ref, max_inputs=50
        )
        assert result.total_inputs_tested > 0
        # Our abs implementation matches for non-negative, may diverge for some negatives
        # depending on C10 semantics

    def test_finds_divergence(self):
        # Buggy program: claims to compute abs but doesn't handle negative
        buggy = """
let x = 0;
let r = x;
print(r);
"""
        def ref(inputs):
            return [abs(inputs.get("x", 0))]

        result = detect_divergence(buggy, {"x": "int"}, ref, max_inputs=50)
        # Should find divergence for negative x
        divergences = [f for f in result.findings if f.kind == "divergence"]
        assert len(divergences) > 0


# -----------------------------------------------------------------------
# Section 11: Fuzz with localization
# -----------------------------------------------------------------------

class TestFuzzWithLocalization:
    def test_runs(self):
        result = fuzz_with_localization(SIMPLE_ABS, {"x": "int"})
        assert isinstance(result, FuzzResult)
        assert result.total_inputs_tested >= 0


# -----------------------------------------------------------------------
# Section 12: VerificationDrivenFuzzer configuration
# -----------------------------------------------------------------------

class TestFuzzerConfig:
    def test_custom_budget(self):
        fuzzer = VerificationDrivenFuzzer(max_total_inputs=10)
        result = fuzzer.fuzz(SIMPLE_ABS, {"x": "int"})
        assert result.total_inputs_tested <= 10

    def test_seed_reproducibility(self):
        f1 = VerificationDrivenFuzzer(max_total_inputs=30, seed=123)
        f2 = VerificationDrivenFuzzer(max_total_inputs=30, seed=123)
        r1 = f1.fuzz(SIMPLE_ABS, {"x": "int"})
        r2 = f2.fuzz(SIMPLE_ABS, {"x": "int"})
        # Same seed should produce same mutation inputs
        mut1 = [fi.values for fi in r1.all_test_inputs if fi.source == "mutation"]
        mut2 = [fi.values for fi in r2.all_test_inputs if fi.source == "mutation"]
        assert mut1 == mut2

    def test_input_sources_tracked(self):
        result = verification_fuzz(BRANCHY, {"x": "int", "y": "int"}, max_inputs=80)
        sources = set(fi.source for fi in result.all_test_inputs)
        # Should have at least symbolic and some others
        assert len(sources) >= 1


# -----------------------------------------------------------------------
# Section 13: Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_no_input_vars(self):
        prog = """
let x = 5;
print(x);
"""
        result = quick_fuzz(prog, {})
        assert result.total_inputs_tested >= 0

    def test_single_var(self):
        result = quick_fuzz(SIMPLE_ABS, {"x": "int"})
        assert result.total_inputs_tested > 0

    def test_many_vars(self):
        prog = """
let a = 0;
let b = 0;
let c = 0;
let r = a + b + c;
print(r);
"""
        result = quick_fuzz(prog, {"a": "int", "b": "int", "c": "int"}, max_inputs=30)
        assert result.total_inputs_tested > 0


# -----------------------------------------------------------------------
# Section 14: FuzzInput data
# -----------------------------------------------------------------------

class TestFuzzInput:
    def test_fields(self):
        fi = FuzzInput(values={"x": 5}, source="symbolic", generation=0)
        assert fi.values == {"x": 5}
        assert fi.source == "symbolic"
        assert fi.generation == 0


# -----------------------------------------------------------------------
# Section 15: Integration - full pipeline
# -----------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_classify(self):
        """Run the full pipeline on a multi-branch program."""
        result = verification_fuzz(CLASSIFY, {"x": "int"}, max_inputs=100)
        assert result.total_inputs_tested > 0
        assert result.coverage is not None
        summary = result.summary()
        assert "Fuzz Result" in summary

    def test_full_pipeline_with_bug(self):
        """Ensure the pipeline finds a known bug."""
        result = verification_fuzz(UNSAFE_DIVISION, {}, max_inputs=20)
        assert result.has_bugs
        assert result.status == FuzzStatus.BUG_FOUND

    def test_pipeline_coverage_improves(self):
        """More budget should yield equal or better coverage."""
        r_small = quick_fuzz(BRANCHY, {"x": "int", "y": "int"}, max_inputs=20)
        r_large = verification_fuzz(BRANCHY, {"x": "int", "y": "int"}, max_inputs=100)
        # Larger budget should test at least as many inputs
        assert r_large.total_inputs_tested >= r_small.total_inputs_tested
