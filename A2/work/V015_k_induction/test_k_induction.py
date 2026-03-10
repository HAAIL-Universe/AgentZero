"""Tests for V015: k-Induction Model Checking."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from smt_solver import SMTSolver, Op, IntConst, BoolConst, App, INT, BOOL
from pdr import TransitionSystem
from k_induction import (
    k_induction_check, incremental_k_induction, k_induction_with_strengthening,
    check_base_case, check_inductive_step, check_inductive_step_with_uniqueness,
    bmc_check, verify_loop, verify_loop_with_invariants, compare_with_pdr,
    KIndResult, _negate
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eq(a, b):
    return App(Op.EQ, [a, b], BOOL)

def _neq(a, b):
    return App(Op.NEQ, [a, b], BOOL)

def _lt(a, b):
    return App(Op.LT, [a, b], BOOL)

def _le(a, b):
    return App(Op.LE, [a, b], BOOL)

def _ge(a, b):
    return App(Op.GE, [a, b], BOOL)

def _gt(a, b):
    return App(Op.GT, [a, b], BOOL)

def _add(a, b):
    return App(Op.ADD, [a, b], INT)

def _sub(a, b):
    return App(Op.SUB, [a, b], INT)

def _mul(a, b):
    return App(Op.MUL, [a, b], INT)

def _and(*args):
    if len(args) == 1:
        return args[0]
    return App(Op.AND, list(args), BOOL)

def _or(*args):
    if len(args) == 1:
        return args[0]
    return App(Op.OR, list(args), BOOL)

def _ite(c, t, e):
    return App(Op.ITE, [c, t, e], INT)

def _const(n):
    return IntConst(n)


# ===================================================================
# Section 1: Simple 1-inductive systems (k=0 suffices)
# ===================================================================

class TestSimple1Inductive:
    """Systems where P is directly inductive (k=0)."""

    def test_trivially_safe(self):
        """Property is always true: True."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _add(x, _const(1))))
        ts.set_property(BoolConst(True))

        r = k_induction_check(ts, k=0)
        assert r.result == "SAFE"
        assert r.k == 0

    def test_identity_system(self):
        """x=0, x'=x, property x==0. 1-inductive."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_eq(x, _const(0)))

        r = k_induction_check(ts, k=0)
        assert r.result == "SAFE"

    def test_monotone_increasing_nonneg(self):
        """x=0, x'=x+1, property x>=0. 1-inductive since x>=0 AND x'=x+1 => x'>=0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _add(x, _const(1))))
        ts.set_property(_ge(x, _const(0)))

        r = k_induction_check(ts, k=0)
        assert r.result == "SAFE"

    def test_conservation_law(self):
        """x=3, y=7, x'=x+1, y'=y-1. Property x+y==10. 1-inductive."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")
        ts.set_init(_and(_eq(x, _const(3)), _eq(y, _const(7))))
        ts.set_trans(_and(_eq(xp, _add(x, _const(1))), _eq(yp, _sub(y, _const(1)))))
        ts.set_property(_eq(_add(x, y), _const(10)))

        r = k_induction_check(ts, k=0)
        assert r.result == "SAFE"


# ===================================================================
# Section 2: Immediate counterexamples
# ===================================================================

class TestImmediate:
    """Property violated at init or within 1 step."""

    def test_init_violation(self):
        """Property fails at initial state."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(-1)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_ge(x, _const(0)))

        r = k_induction_check(ts, k=0)
        assert r.result == "UNSAFE"
        assert r.counterexample is not None
        assert r.counterexample[0]["x"] == -1

    def test_one_step_violation(self):
        """Property holds at init but fails after one step."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(5)))
        ts.set_trans(_eq(xp, _sub(x, _const(10))))
        ts.set_property(_ge(x, _const(0)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "UNSAFE"
        assert r.k == 1

    def test_two_step_violation(self):
        """x=5, x'=x-3. Property x>=0 fails at step 2 (5->2->-1)."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(5)))
        ts.set_trans(_eq(xp, _sub(x, _const(3))))
        ts.set_property(_ge(x, _const(0)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "UNSAFE"
        assert r.k == 2


# ===================================================================
# Section 3: k-Inductive properties (require k > 0)
# ===================================================================

class TestKInductive:
    """Properties that need k > 0 for the induction to go through."""

    def test_two_phase_counter(self):
        """Two-variable system: x counts 0,1,0,1,... y increments.
        Property y>=0 is 1-inductive. But x in {0,1} needs k=1."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")

        # x flips between 0 and 1, y always increments
        ts.set_init(_and(_eq(x, _const(0)), _eq(y, _const(0))))
        ts.set_trans(_and(
            _eq(xp, _ite(_eq(x, _const(0)), _const(1), _const(0))),
            _eq(yp, _add(y, _const(1)))
        ))
        ts.set_property(_ge(y, _const(0)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"
        assert r.k <= 2  # Should be provable at small k

    def test_increment_bounded_by_init(self):
        """x=0, x'=x+2. Property x>=0. 1-inductive (x>=0 AND x'=x+2 => x'>=0)."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _add(x, _const(2))))
        ts.set_property(_ge(x, _const(0)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"


# ===================================================================
# Section 4: Incremental k search
# ===================================================================

class TestIncremental:
    """Test that incremental search finds the right k."""

    def test_finds_minimal_k(self):
        """Identity system is 0-inductive."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_eq(x, _const(0)))

        r = incremental_k_induction(ts, max_k=10)
        assert r.result == "SAFE"
        assert r.k == 0

    def test_max_k_reached(self):
        """Unbounded counter with bounded property: can't be proven by k-induction alone."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _add(x, _const(1))))
        ts.set_property(_lt(x, _const(100)))

        # k-induction can't prove x<100 without an invariant, because
        # x>=0 AND x<100 AND x'=x+1 => x'<100 fails when x=99.
        # But the base case passes up to k=99.
        # With small max_k, we get UNKNOWN.
        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "UNKNOWN"

    def test_stats_populated(self):
        """Stats dict should contain useful info."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_eq(x, _const(0)))

        r = incremental_k_induction(ts, max_k=5)
        assert "time" in r.stats
        assert "base_checks" in r.stats
        assert "ind_checks" in r.stats


# ===================================================================
# Section 5: Invariant strengthening
# ===================================================================

class TestStrengthening:
    """Test k-induction with auxiliary invariant strengthening."""

    def test_bounded_counter_with_invariant(self):
        """x=0, x'=x+1, prop x<100. Not k-inductive alone.
        With invariant x>=0, it's still not provable.
        But with invariant x>=0 AND x<=99, it works at k=0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        # x increments until 99, then stays (guarded transition)
        ts.set_trans(_eq(xp, _ite(_lt(x, _const(99)), _add(x, _const(1)), x)))
        ts.set_property(_lt(x, _const(100)))

        # Without strengthening: may not prove
        r1 = incremental_k_induction(ts, max_k=3)

        # With invariant x>=0 AND x<=99
        inv = _and(_ge(x, _const(0)), _le(x, _const(99)))
        r2 = k_induction_with_strengthening(ts, max_k=3, invariants=[inv])
        assert r2.result == "SAFE"

    def test_two_var_with_conservation_invariant(self):
        """x+y==10, x'=x+1,y'=y-1. Property x<=10.
        With x+y==10 as invariant, provable since x<=10 iff y>=0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")
        ts.set_init(_and(_eq(x, _const(3)), _eq(y, _const(7))))
        # Guarded: only increment if y > 0
        ts.set_trans(_and(
            _eq(xp, _ite(_gt(y, _const(0)), _add(x, _const(1)), x)),
            _eq(yp, _ite(_gt(y, _const(0)), _sub(y, _const(1)), y))
        ))
        ts.set_property(_le(x, _const(10)))

        inv = _eq(_add(x, y), _const(10))
        r = k_induction_with_strengthening(ts, max_k=5, invariants=[inv])
        assert r.result == "SAFE"

    def test_multiple_invariants(self):
        """Test passing multiple invariants."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _ite(_lt(x, _const(10)), _add(x, _const(1)), x)))
        ts.set_property(_le(x, _const(10)))

        inv1 = _ge(x, _const(0))
        inv2 = _le(x, _const(10))
        r = k_induction_with_strengthening(ts, max_k=3, invariants=[inv1, inv2])
        assert r.result == "SAFE"


# ===================================================================
# Section 6: Uniqueness constraints
# ===================================================================

class TestUniqueness:
    """Test the uniqueness variant of the inductive step."""

    def test_uniqueness_helps_finite_state(self):
        """For a finite-state system, uniqueness can help prove properties
        that would otherwise need a large k."""
        # x toggles 0,1,0,1,...
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _ite(_eq(x, _const(0)), _const(1), _const(0))))
        ts.set_property(_le(x, _const(1)))

        r = incremental_k_induction(ts, max_k=5, use_uniqueness=True)
        assert r.result == "SAFE"

    def test_uniqueness_basic(self):
        """Basic uniqueness test on a simple system."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(5)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_eq(x, _const(5)))

        r_normal = k_induction_check(ts, k=0, use_uniqueness=False)
        r_unique = k_induction_check(ts, k=0, use_uniqueness=True)
        assert r_normal.result == "SAFE"
        assert r_unique.result == "SAFE"


# ===================================================================
# Section 7: BMC-only mode
# ===================================================================

class TestBMC:
    """Test bounded model checking (bug-finding only)."""

    def test_bmc_finds_bug(self):
        """BMC should find bug within depth."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(3)))
        ts.set_trans(_eq(xp, _sub(x, _const(1))))
        ts.set_property(_ge(x, _const(0)))

        r = bmc_check(ts, max_depth=10)
        assert r.result == "UNSAFE"
        assert r.k == 4  # 3 -> 2 -> 1 -> 0 -> -1

    def test_bmc_no_bug_unknown(self):
        """If no bug within bounds, BMC returns UNKNOWN."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _add(x, _const(1))))
        ts.set_property(_ge(x, _const(0)))

        r = bmc_check(ts, max_depth=5)
        assert r.result == "UNKNOWN"

    def test_bmc_init_violation(self):
        """Bug at depth 0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(-5)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_ge(x, _const(0)))

        r = bmc_check(ts, max_depth=3)
        assert r.result == "UNSAFE"
        assert r.k == 0


# ===================================================================
# Section 8: Multi-variable systems
# ===================================================================

class TestMultiVar:
    """Systems with multiple state variables."""

    def test_two_counters_independent(self):
        """Two independent counters, both non-negative."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")
        ts.set_init(_and(_eq(x, _const(0)), _eq(y, _const(0))))
        ts.set_trans(_and(_eq(xp, _add(x, _const(1))), _eq(yp, _add(y, _const(2)))))
        ts.set_property(_and(_ge(x, _const(0)), _ge(y, _const(0))))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"

    def test_swap_system(self):
        """x,y swap each step. Property x+y == init sum."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")
        ts.set_init(_and(_eq(x, _const(3)), _eq(y, _const(7))))
        ts.set_trans(_and(_eq(xp, y), _eq(yp, x)))
        ts.set_property(_eq(_add(x, y), _const(10)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"

    def test_three_vars(self):
        """x=a, y=b, z=c, all rotate. Property x+y+z == a+b+c."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        z = ts.add_int_var("z")
        xp = ts.prime("x")
        yp = ts.prime("y")
        zp = ts.prime("z")
        ts.set_init(_and(_eq(x, _const(1)), _eq(y, _const(2)), _eq(z, _const(3))))
        ts.set_trans(_and(_eq(xp, y), _eq(yp, z), _eq(zp, x)))
        ts.set_property(_eq(_add(_add(x, y), z), _const(6)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"


# ===================================================================
# Section 9: Conditional transitions
# ===================================================================

class TestConditional:
    """Systems with conditional (ITE) transitions."""

    def test_abs_system(self):
        """x starts anywhere, x' = abs(x). After one step x>=0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(-5)))
        ts.set_trans(_eq(xp, _ite(_ge(x, _const(0)), x, _sub(_const(0), x))))
        ts.set_property(_ge(x, _const(0)))

        # Fails at step 0 (x=-5) but passes after step 1
        r = bmc_check(ts, max_depth=0)
        assert r.result == "UNSAFE"

    def test_clamp_system(self):
        """x = clamp(x, 0, 10). Property 0<=x<=10 after first step."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(5)))
        ts.set_trans(_eq(xp, _ite(_lt(x, _const(0)), _const(0),
                                  _ite(_gt(x, _const(10)), _const(10), x))))
        ts.set_property(_and(_ge(x, _const(0)), _le(x, _const(10))))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"

    def test_guarded_counter(self):
        """x increments only while x < 5, then stays. Property x<=5."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _ite(_lt(x, _const(5)), _add(x, _const(1)), x)))
        ts.set_property(_le(x, _const(5)))

        inv = _and(_ge(x, _const(0)), _le(x, _const(5)))
        r = k_induction_with_strengthening(ts, max_k=5, invariants=[inv])
        assert r.result == "SAFE"


# ===================================================================
# Section 10: Nondeterministic systems
# ===================================================================

class TestNondeterministic:
    """Systems with nondeterminism (under-constrained transitions)."""

    def test_nondet_stays_positive(self):
        """x=10, x'>=x (nondeterministic increase). Property x>=10."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(10)))
        ts.set_trans(_ge(xp, x))
        ts.set_property(_ge(x, _const(10)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"

    def test_nondet_bounded(self):
        """x=0, 0<=x'<=x+1. Property x>=0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_and(_ge(xp, _const(0)), _le(xp, _add(x, _const(1)))))
        ts.set_property(_ge(x, _const(0)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"


# ===================================================================
# Section 11: Source-level API
# ===================================================================

class TestSourceLevel:
    """Test verify_loop with C10 source code."""

    def test_countdown_nonneg(self):
        """Countdown loop: i=10; while(i>0) { i=i-1; }. Property i>=0."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        r = verify_loop(source, "i >= 0", max_k=5)
        assert r.result == "SAFE"

    def test_countup_bounded(self):
        """Countup: i=0; while(i<10) { i=i+1; }. Property i<=10."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        r = verify_loop(source, "i <= 10", max_k=5)
        assert r.result == "SAFE"

    def test_accumulator_nonneg(self):
        """sum=0, i=0; while(i<5) { sum=sum+i; i=i+1; }. Property sum>=0.
        Needs i>=0 invariant to prove sum stays non-negative."""
        source = "let sum = 0; let i = 0; while (i < 5) { sum = sum + i; i = i + 1; }"
        r = verify_loop_with_invariants(source, "sum >= 0", ["i >= 0", "sum >= 0"], max_k=5)
        assert r.result == "SAFE"

    def test_conservation(self):
        """x=3, y=7; while(x<10) { x=x+1; y=y-1; }. Property x+y==10."""
        source = "let x = 3; let y = 7; while (x < 10) { x = x + 1; y = y - 1; }"
        r = verify_loop(source, "x + y == 10", max_k=5)
        assert r.result == "SAFE"

    def test_source_with_invariants(self):
        """Test source-level API with invariants."""
        source = "let i = 0; while (i < 100) { i = i + 1; }"
        r = verify_loop_with_invariants(source, "i <= 100", ["i >= 0", "i <= 100"], max_k=3)
        assert r.result == "SAFE"


# ===================================================================
# Section 12: Counterexample traces
# ===================================================================

class TestCounterexamples:
    """Test counterexample trace quality."""

    def test_trace_length(self):
        """Trace should have k+1 states."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(3)))
        ts.set_trans(_eq(xp, _sub(x, _const(2))))
        ts.set_property(_ge(x, _const(0)))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "UNSAFE"
        assert r.counterexample is not None
        assert len(r.counterexample) == r.k + 1

    def test_trace_is_valid(self):
        """Trace should follow the transition relation."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(10)))
        ts.set_trans(_eq(xp, _sub(x, _const(3))))
        ts.set_property(_ge(x, _const(0)))

        r = incremental_k_induction(ts, max_k=10)
        assert r.result == "UNSAFE"
        trace = r.counterexample
        # Check transitions
        for i in range(len(trace) - 1):
            assert trace[i + 1]["x"] == trace[i]["x"] - 3

    def test_trace_violates_property(self):
        """Last state in trace should violate the property."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(5)))
        ts.set_trans(_eq(xp, _sub(x, _const(1))))
        ts.set_property(_gt(x, _const(0)))

        r = incremental_k_induction(ts, max_k=10)
        assert r.result == "UNSAFE"
        # Some state in the trace should violate x > 0
        violations = [s for s in r.counterexample if s["x"] <= 0]
        assert len(violations) > 0


# ===================================================================
# Section 13: PDR comparison
# ===================================================================

class TestPDRComparison:
    """Compare k-induction with PDR results."""

    def test_both_agree_safe(self):
        """Both should agree on SAFE."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, _add(x, _const(1))))
        ts.set_property(_ge(x, _const(0)))

        result = compare_with_pdr(ts)
        assert result["k_induction"]["result"] == "SAFE"
        assert result["pdr"]["result"] == "SAFE"

    def test_both_agree_unsafe(self):
        """Both should agree on UNSAFE."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(5)))
        ts.set_trans(_eq(xp, _sub(x, _const(1))))
        ts.set_property(_gt(x, _const(0)))

        result = compare_with_pdr(ts)
        assert result["k_induction"]["result"] == "UNSAFE"
        assert result["pdr"]["result"] == "UNSAFE"


# ===================================================================
# Section 14: Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_k_zero(self):
        """k=0 is just checking init => property."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_eq(x, _const(0)))

        base_ok, _ = check_base_case(ts, 0)
        assert base_ok is True
        ind_ok, _ = check_inductive_step(ts, 0)
        assert ind_ok is True

    def test_single_var_system(self):
        """Minimal single-variable system."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, _const(42)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_eq(x, _const(42)))

        r = k_induction_check(ts, k=0)
        assert r.result == "SAFE"

    def test_result_repr(self):
        """Test KIndResult string representation."""
        r = KIndResult("SAFE", k=3)
        assert "SAFE" in repr(r)
        assert "3" in repr(r)

    def test_negate_constants(self):
        """Test negation of boolean constants."""
        assert isinstance(_negate(BoolConst(True)), BoolConst)
        assert _negate(BoolConst(True)).value is False

    def test_negate_operators(self):
        """Test negation with complement operators."""
        x = IntConst(1)
        y = IntConst(2)
        eq = _eq(x, y)
        neg = _negate(eq)
        assert isinstance(neg, App) and neg.op == Op.NEQ


# ===================================================================
# Section 15: Performance and practical patterns
# ===================================================================

class TestPractical:
    """Real-world-ish patterns."""

    def test_producer_consumer(self):
        """Buffer: produced increments, consumed increments, buffer = produced - consumed.
        Property: buffer >= 0. Invariant: consumed <= produced."""
        ts = TransitionSystem()
        p = ts.add_int_var("p")  # produced
        c = ts.add_int_var("c")  # consumed
        pp = ts.prime("p")
        cp = ts.prime("c")

        ts.set_init(_and(_eq(p, _const(0)), _eq(c, _const(0))))
        # p always increments, c increments only if c < p
        ts.set_trans(_and(
            _eq(pp, _add(p, _const(1))),
            _eq(cp, _ite(_lt(c, p), _add(c, _const(1)), c))
        ))
        ts.set_property(_ge(_sub(p, c), _const(0)))

        inv = _ge(_sub(p, c), _const(0))
        r = k_induction_with_strengthening(ts, max_k=5, invariants=[inv])
        assert r.result == "SAFE"

    def test_fibonacci_like_bounded(self):
        """a=1, b=1. a'=b, b'=a+b. Property a>=1 AND b>=1."""
        ts = TransitionSystem()
        a = ts.add_int_var("a")
        b = ts.add_int_var("b")
        ap = ts.prime("a")
        bp = ts.prime("b")

        ts.set_init(_and(_eq(a, _const(1)), _eq(b, _const(1))))
        ts.set_trans(_and(_eq(ap, b), _eq(bp, _add(a, b))))
        ts.set_property(_and(_ge(a, _const(1)), _ge(b, _const(1))))

        r = incremental_k_induction(ts, max_k=5)
        assert r.result == "SAFE"

    def test_traffic_light(self):
        """State machine: 0=red, 1=green, 2=yellow, cycling.
        Property: state in {0,1,2}."""
        ts = TransitionSystem()
        s = ts.add_int_var("s")
        sp = ts.prime("s")

        ts.set_init(_eq(s, _const(0)))
        ts.set_trans(_eq(sp, _ite(_eq(s, _const(0)), _const(1),
                                  _ite(_eq(s, _const(1)), _const(2), _const(0)))))
        ts.set_property(_and(_ge(s, _const(0)), _le(s, _const(2))))

        inv = _and(_ge(s, _const(0)), _le(s, _const(2)))
        r = k_induction_with_strengthening(ts, max_k=5, invariants=[inv])
        assert r.result == "SAFE"
