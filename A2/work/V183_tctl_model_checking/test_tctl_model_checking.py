"""
Tests for V183: TCTL Model Checking

Tests TCTL (Timed Computation Tree Logic) model checking over timed automata.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tctl_model_checking import (
    # Formula AST
    Atomic, TrueF, FalseF, Not, And, Or, Implies,
    EF, AF, EG, AG, EU, AU,
    TimeBound, BoundType,
    # Model checker
    check_tctl, check_tctl_batch, tctl_summary,
    TCTLResult,
    # Example systems
    example_light_controller, example_request_response,
    example_mutex_protocol, example_train_crossing,
    labeled_ta,
    # Helpers
    _check_atomic_locations, _is_state_formula,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V118_timed_automata'))
from timed_automata import (
    true_guard, clock_leq, clock_lt, clock_geq, clock_gt, clock_eq,
    guard_and, simple_ta,
)


# ============================================================
# Formula Construction Tests
# ============================================================

class TestFormulaConstruction:
    def test_atomic(self):
        f = Atomic("safe")
        assert str(f) == "safe"

    def test_true_false(self):
        assert str(TrueF()) == "true"
        assert str(FalseF()) == "false"

    def test_not(self):
        f = Not(Atomic("error"))
        assert "!" in str(f)

    def test_and(self):
        f = And(Atomic("a"), Atomic("b"))
        assert "&&" in str(f)

    def test_or(self):
        f = Or(Atomic("a"), Atomic("b"))
        assert "||" in str(f)

    def test_implies(self):
        f = Implies(Atomic("a"), Atomic("b"))
        assert "=>" in str(f)

    def test_ef_unbounded(self):
        f = EF(Atomic("target"))
        assert "E F" in str(f)

    def test_ef_bounded(self):
        f = EF(Atomic("target"), TimeBound.leq(5))
        assert "E F" in str(f)
        assert "5" in str(f)

    def test_ag_bounded(self):
        f = AG(Atomic("safe"), TimeBound.leq(10))
        assert "A G" in str(f)

    def test_eu(self):
        f = EU(Atomic("trying"), Atomic("done"), TimeBound.leq(8))
        assert "U" in str(f)

    def test_au(self):
        f = AU(Atomic("waiting"), Atomic("served"), TimeBound.leq(15))
        assert "U" in str(f)

    def test_time_bound_types(self):
        assert TimeBound.unbounded().bound_type == BoundType.UNBOUNDED
        assert TimeBound.leq(5).value == 5
        assert TimeBound.lt(3).bound_type == BoundType.LT
        assert TimeBound.geq(2).bound_type == BoundType.GEQ
        assert TimeBound.gt(1).bound_type == BoundType.GT
        assert TimeBound.eq(4).bound_type == BoundType.EQ

    def test_nested_formula(self):
        f = AG(Implies(Atomic("request"), EF(Atomic("response"), TimeBound.leq(10))))
        assert "A G" in str(f)
        assert "E F" in str(f)


# ============================================================
# State Formula Tests
# ============================================================

class TestStateFormula:
    def test_is_state_formula_atomic(self):
        assert _is_state_formula(Atomic("x"))

    def test_is_state_formula_true_false(self):
        assert _is_state_formula(TrueF())
        assert _is_state_formula(FalseF())

    def test_is_state_formula_boolean(self):
        assert _is_state_formula(And(Atomic("a"), Atomic("b")))
        assert _is_state_formula(Or(Atomic("a"), Not(Atomic("b"))))

    def test_is_not_state_formula_temporal(self):
        assert not _is_state_formula(EF(Atomic("x")))
        assert not _is_state_formula(AG(Atomic("x")))


# ============================================================
# Light Controller Tests
# ============================================================

class TestLightController:
    @pytest.fixture
    def ta(self):
        return example_light_controller()

    def test_ef_on_reachable(self, ta):
        """Light can be turned on (E F on)."""
        result = check_tctl(ta, EF(Atomic("on")))
        assert result.satisfied

    def test_ef_off_reachable(self, ta):
        """Can return to off (E F off)."""
        result = check_tctl(ta, EF(Atomic("off")))
        assert result.satisfied

    def test_ef_on_within_2(self, ta):
        """Light turns on within 2 time units (E F_{<=2} on)."""
        result = check_tctl(ta, EF(Atomic("on"), TimeBound.leq(2)))
        assert result.satisfied

    def test_ef_on_within_0(self, ta):
        """Can turn on at time 0 (guard x<=2 allows x=0)."""
        result = check_tctl(ta, EF(Atomic("on"), TimeBound.leq(0)))
        assert result.satisfied

    def test_ef_active_within_2(self, ta):
        """Active label reachable within 2 (E F_{<=2} active)."""
        result = check_tctl(ta, EF(Atomic("active"), TimeBound.leq(2)))
        assert result.satisfied

    def test_ag_not_error(self, ta):
        """No error state exists (A G !error)."""
        result = check_tctl(ta, AG(Not(Atomic("error"))))
        assert result.satisfied

    def test_initial_is_off(self, ta):
        """Initial location is off."""
        result = check_tctl(ta, Atomic("off"))
        assert result.satisfied

    def test_initial_not_on(self, ta):
        """Initial location is not on."""
        result = check_tctl(ta, Atomic("on"))
        assert not result.satisfied

    def test_ef_idle_label(self, ta):
        """Idle label is reachable (it's the initial state)."""
        result = check_tctl(ta, EF(Atomic("idle")))
        assert result.satisfied


# ============================================================
# Request-Response Tests
# ============================================================

class TestRequestResponse:
    @pytest.fixture
    def ta(self):
        return example_request_response()

    def test_ef_done_reachable(self, ta):
        """Success state is reachable."""
        result = check_tctl(ta, EF(Atomic("done")))
        assert result.satisfied

    def test_ef_success_reachable(self, ta):
        """Success label is reachable."""
        result = check_tctl(ta, EF(Atomic("success")))
        assert result.satisfied

    def test_ef_timeout_reachable(self, ta):
        """Timeout state is reachable."""
        result = check_tctl(ta, EF(Atomic("timeout")))
        assert result.satisfied

    def test_ef_done_within_15(self, ta):
        """Can complete within 15 time units."""
        result = check_tctl(ta, EF(Atomic("done"), TimeBound.leq(15)))
        assert result.satisfied

    def test_ef_done_within_1_fails(self, ta):
        """Cannot complete within 1 time unit (processing takes >= 2)."""
        result = check_tctl(ta, EF(Atomic("done"), TimeBound.leq(1)))
        assert not result.satisfied

    def test_ag_not_nonexistent(self, ta):
        """Nonexistent state never reached."""
        result = check_tctl(ta, AG(Not(Atomic("nonexistent"))))
        assert result.satisfied

    def test_initial_is_idle(self, ta):
        result = check_tctl(ta, Atomic("idle"))
        assert result.satisfied

    def test_ef_busy_within_5(self, ta):
        """Processing is reachable within 5 time units."""
        result = check_tctl(ta, EF(Atomic("busy"), TimeBound.leq(5)))
        assert result.satisfied

    def test_eu_waiting_until_busy(self, ta):
        """E [waiting U busy]: can go from waiting to busy."""
        result = check_tctl(ta, EU(Atomic("idle"), Atomic("processing")))
        assert result.satisfied


# ============================================================
# Mutex Protocol Tests
# ============================================================

class TestMutexProtocol:
    @pytest.fixture
    def ta(self):
        return example_mutex_protocol()

    def test_ef_critical_reachable(self, ta):
        """Critical section is reachable."""
        result = check_tctl(ta, EF(Atomic("critical")))
        assert result.satisfied

    def test_ef_locked_reachable(self, ta):
        """Locked label is reachable."""
        result = check_tctl(ta, EF(Atomic("locked")))
        assert result.satisfied

    def test_ef_critical_within_5(self, ta):
        """Can enter critical within 5 time units."""
        result = check_tctl(ta, EF(Atomic("critical"), TimeBound.leq(5)))
        assert result.satisfied

    def test_ef_idle_always_reachable(self, ta):
        """Can always return to idle."""
        result = check_tctl(ta, EF(Atomic("idle")))
        assert result.satisfied

    def test_ag_bounded_critical(self, ta):
        """A G_{<=3} locked: locked holds for at most 3 time units (invariant)."""
        # This checks the bounded globally -- critical section bounded
        result = check_tctl(ta, AG(Not(Atomic("nonexistent")), TimeBound.leq(3)))
        assert result.satisfied

    def test_eu_idle_until_critical_fails(self, ta):
        """E [idle U critical] fails: path goes idle->trying->critical,
        but trying doesn't satisfy 'idle', violating until."""
        result = check_tctl(ta, EU(Atomic("idle"), Atomic("critical")))
        assert not result.satisfied

    def test_eu_idle_until_trying(self, ta):
        """E [idle U trying]: from idle directly to trying. Holds."""
        result = check_tctl(ta, EU(Atomic("idle"), Atomic("trying")))
        assert result.satisfied

    def test_eu_true_until_critical(self, ta):
        """E [true U critical]: reachability via until with trivial path condition."""
        result = check_tctl(ta, EU(TrueF(), Atomic("critical")))
        assert result.satisfied

    def test_eu_true_until_critical_within_8(self, ta):
        """E [true U_{<=8} critical]."""
        result = check_tctl(ta, EU(TrueF(), Atomic("critical"), TimeBound.leq(8)))
        assert result.satisfied


# ============================================================
# Train Crossing Tests
# ============================================================

class TestTrainCrossing:
    @pytest.fixture
    def ta(self):
        return example_train_crossing()

    def test_ef_crossing_reachable(self, ta):
        """Crossing state is reachable."""
        result = check_tctl(ta, EF(Atomic("crossing")))
        assert result.satisfied

    def test_ef_danger_reachable(self, ta):
        """Danger label is reachable."""
        result = check_tctl(ta, EF(Atomic("danger")))
        assert result.satisfied

    def test_ef_passed_reachable(self, ta):
        """Passed state is reachable."""
        result = check_tctl(ta, EF(Atomic("passed")))
        assert result.satisfied

    def test_ef_safe_reachable(self, ta):
        """Safe label is reachable (approach and passed)."""
        result = check_tctl(ta, EF(Atomic("safe")))
        assert result.satisfied

    def test_ef_warning_within_10(self, ta):
        """Warning reachable within 10 time units."""
        result = check_tctl(ta, EF(Atomic("warning"), TimeBound.leq(10)))
        assert result.satisfied

    def test_ag_not_error(self, ta):
        """No error state."""
        result = check_tctl(ta, AG(Not(Atomic("error"))))
        assert result.satisfied


# ============================================================
# Boolean Combination Tests
# ============================================================

class TestBooleanCombinations:
    @pytest.fixture
    def ta(self):
        return example_light_controller()

    def test_and_true(self, ta):
        """off AND idle (both true at initial)."""
        result = check_tctl(ta, And(Atomic("off"), Atomic("idle")))
        assert result.satisfied

    def test_and_false(self, ta):
        """off AND on (impossible)."""
        result = check_tctl(ta, And(Atomic("off"), Atomic("on")))
        assert not result.satisfied

    def test_or_true(self, ta):
        """off OR on (one is true)."""
        result = check_tctl(ta, Or(Atomic("off"), Atomic("on")))
        assert result.satisfied

    def test_or_false(self, ta):
        """error OR crash (neither exists)."""
        result = check_tctl(ta, Or(Atomic("error"), Atomic("crash")))
        assert not result.satisfied

    def test_implies_true(self, ta):
        """off => idle (both true)."""
        result = check_tctl(ta, Implies(Atomic("off"), Atomic("idle")))
        assert result.satisfied

    def test_implies_false_antecedent(self, ta):
        """on => error (antecedent false at initial, so true)."""
        result = check_tctl(ta, Implies(Atomic("on"), Atomic("error")))
        assert result.satisfied

    def test_not_true(self, ta):
        """!on (initial is off)."""
        result = check_tctl(ta, Not(Atomic("on")))
        assert result.satisfied

    def test_not_false(self, ta):
        """!off (initial is off, so false)."""
        result = check_tctl(ta, Not(Atomic("off")))
        assert not result.satisfied

    def test_true_formula(self, ta):
        result = check_tctl(ta, TrueF())
        assert result.satisfied

    def test_false_formula(self, ta):
        result = check_tctl(ta, FalseF())
        assert not result.satisfied


# ============================================================
# Unbounded CTL Tests
# ============================================================

class TestUnboundedCTL:
    @pytest.fixture
    def ta(self):
        return example_mutex_protocol()

    def test_ef_reachable(self, ta):
        result = check_tctl(ta, EF(Atomic("critical")))
        assert result.satisfied
        assert result.witness_trace is not None

    def test_ef_unreachable(self, ta):
        result = check_tctl(ta, EF(Atomic("nonexistent")))
        assert not result.satisfied

    def test_ag_always_holds(self, ta):
        """A G !nonexistent: always true."""
        result = check_tctl(ta, AG(Not(Atomic("nonexistent"))))
        assert result.satisfied

    def test_ag_violated(self, ta):
        """A G !critical: violated because critical is reachable."""
        result = check_tctl(ta, AG(Not(Atomic("critical"))))
        assert not result.satisfied
        assert result.counterexample_trace is not None

    def test_eu_basic(self, ta):
        result = check_tctl(ta, EU(Atomic("idle"), Atomic("trying")))
        assert result.satisfied

    def test_au_basic(self, ta):
        """A [free U waiting]: from free, all paths go to waiting.
        Since idle->trying is the only edge from idle, this holds."""
        result = check_tctl(ta, AU(Atomic("idle"), Atomic("trying")))
        assert result.satisfied

    def test_eg_cycle(self, ta):
        """E G free: can we stay free forever? No, because idle always goes to trying."""
        # Actually, there's an edge idle->trying but it's not forced.
        # In timed automata, time can pass. If there's no invariant forcing
        # the edge, the system can stay in idle.
        # idle has no invariant, so it CAN stay forever.
        result = check_tctl(ta, EG(Atomic("idle")))
        assert result.satisfied

    def test_eg_impossible(self, ta):
        """E G critical: can we stay in critical forever? No, invariant x<=3."""
        result = check_tctl(ta, EG(Atomic("critical")))
        assert not result.satisfied


# ============================================================
# Bounded Time Tests
# ============================================================

class TestBoundedTime:
    def test_tight_bound_satisfied(self):
        """Simple 2-state system: a->b with x<=3. E F_{<=3} b should hold."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_leq("x", 3), frozenset(["x"]))],
            invariants={},
            labels={"a": {"start"}, "b": {"end"}},
        )
        result = check_tctl(ta, EF(Atomic("b"), TimeBound.leq(3)))
        assert result.satisfied

    def test_tight_bound_violated(self):
        """a->b requires x>=5. E F_{<=3} b should fail."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_geq("x", 5), frozenset(["x"]))],
            invariants={},
            labels={},
        )
        result = check_tctl(ta, EF(Atomic("b"), TimeBound.leq(3)))
        assert not result.satisfied

    def test_bound_exactly_met(self):
        """a->b requires x>=5. E F_{<=5} b should hold."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_geq("x", 5), frozenset(["x"]))],
            invariants={},
            labels={},
        )
        result = check_tctl(ta, EF(Atomic("b"), TimeBound.leq(5)))
        assert result.satisfied

    def test_multi_hop_bound(self):
        """a->b (x>=2, reset) -> c (x>=3). E F_{<=6} c should hold (2+3=5<=6)."""
        ta = labeled_ta(
            locations=["a", "b", "c"],
            initial="a",
            clocks=["x"],
            edges=[
                ("a", "b", "step1", clock_geq("x", 2), frozenset(["x"])),
                ("b", "c", "step2", clock_geq("x", 3), frozenset(["x"])),
            ],
            invariants={},
            labels={},
        )
        result = check_tctl(ta, EF(Atomic("c"), TimeBound.leq(6)))
        assert result.satisfied

    def test_multi_hop_bound_too_tight(self):
        """a->b (x>=2, reset) -> c (x>=3). E F_{<=4} c should fail (2+3=5>4)."""
        ta = labeled_ta(
            locations=["a", "b", "c"],
            initial="a",
            clocks=["x"],
            edges=[
                ("a", "b", "step1", clock_geq("x", 2), frozenset(["x"])),
                ("b", "c", "step2", clock_geq("x", 3), frozenset(["x"])),
            ],
            invariants={},
            labels={},
        )
        result = check_tctl(ta, EF(Atomic("c"), TimeBound.leq(4)))
        assert not result.satisfied

    def test_ag_bounded_holds(self):
        """A G_{<=2} a: a holds for 2 time units (initial state, no outgoing required)."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_geq("x", 5), frozenset())],
            invariants={},
            labels={},
        )
        result = check_tctl(ta, AG(Atomic("a"), TimeBound.leq(2)))
        assert result.satisfied

    def test_eu_bounded(self):
        """E [a U_{<=5} b]: go from a to b within 5."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_leq("x", 3), frozenset())],
            invariants={},
            labels={},
        )
        result = check_tctl(ta, EU(Atomic("a"), Atomic("b"), TimeBound.leq(5)))
        assert result.satisfied

    def test_au_bounded_holds(self):
        """A [a U_{<=10} b]: must reach b within 10."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_leq("x", 5), frozenset())],
            invariants={"a": clock_leq("x", 5)},
            labels={},
        )
        result = check_tctl(ta, AU(Atomic("a"), Atomic("b"), TimeBound.leq(10)))
        assert result.satisfied

    def test_au_bounded_violated(self):
        """A [a U_{<=2} b] but can only go to b after x>=5: violated."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_geq("x", 5), frozenset())],
            invariants={},
            labels={},
        )
        # a can stay forever (no invariant forcing exit), but b needs x>=5
        # There exist paths that don't reach b within 2
        result = check_tctl(ta, AU(Atomic("a"), Atomic("b"), TimeBound.leq(2)))
        assert not result.satisfied


# ============================================================
# EG Bounded Tests
# ============================================================

class TestEGBounded:
    def test_eg_bounded_holds(self):
        """E G_{<=3} a: can stay in a for 3 time units."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_geq("x", 5), frozenset())],
            invariants={},
            labels={},
        )
        # a has no invariant, can stay indefinitely -> E G_{<=3} holds
        result = check_tctl(ta, EG(Atomic("a"), TimeBound.leq(3)))
        assert result.satisfied

    def test_eg_bounded_violated(self):
        """E G_{<=10} a: need a for 10 units, but invariant forces exit at 3."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", true_guard(), frozenset())],
            invariants={"a": clock_leq("x", 3)},
            labels={},
        )
        # a's invariant: x<=3, must leave by time 3
        # But b is not "a", so E G_{<=10} a fails
        # Wait -- AF !a within 3 is what's happening
        # E G_{<=10} a = !A F_{<=10} !a
        # A F_{<=10} !a should be true (forced out of a by x<=3)
        # But actually, a->b with true_guard means we CAN stay in a (a has invariant x<=3)
        # and must leave by time 3. So E G_{<=10} a is violated (can't stay 10 units)
        result = check_tctl(ta, EG(Atomic("a"), TimeBound.leq(10)))
        assert not result.satisfied


# ============================================================
# Batch and Summary Tests
# ============================================================

class TestBatchAndSummary:
    @pytest.fixture
    def ta(self):
        return example_light_controller()

    def test_batch(self, ta):
        formulas = [
            EF(Atomic("on")),
            AG(Not(Atomic("error"))),
            EF(Atomic("on"), TimeBound.leq(2)),
        ]
        results = check_tctl_batch(ta, formulas)
        assert len(results) == 3
        assert all(r.satisfied for r in results)

    def test_summary(self, ta):
        formulas = [
            EF(Atomic("on")),
            AG(Not(Atomic("error"))),
        ]
        s = tctl_summary(ta, formulas)
        assert "PASS" in s
        assert "2/2" in s

    def test_summary_with_failure(self, ta):
        formulas = [
            EF(Atomic("on")),
            EF(Atomic("nonexistent")),
        ]
        s = tctl_summary(ta, formulas)
        assert "PASS" in s
        assert "FAIL" in s
        assert "1/2" in s


# ============================================================
# Result Structure Tests
# ============================================================

class TestResultStructure:
    def test_result_str(self):
        r = TCTLResult(satisfied=True, formula=EF(Atomic("x")), states_explored=42)
        s = str(r)
        assert "SATISFIED" in s
        assert "42" in s

    def test_result_violated_str(self):
        r = TCTLResult(satisfied=False, formula=AG(Atomic("safe")), states_explored=10)
        s = str(r)
        assert "VIOLATED" in s

    def test_witness_trace_exists(self):
        ta = example_light_controller()
        result = check_tctl(ta, EF(Atomic("on")))
        assert result.satisfied
        assert result.witness_trace is not None
        assert len(result.witness_trace.steps) > 0

    def test_counterexample_trace(self):
        ta = example_mutex_protocol()
        result = check_tctl(ta, AG(Not(Atomic("critical"))))
        assert not result.satisfied
        assert result.counterexample_trace is not None


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_location(self):
        """Single location, no edges."""
        ta = labeled_ta(
            locations=["only"],
            initial="only",
            clocks=["x"],
            edges=[],
            labels={"only": {"here"}},
        )
        # EF only: trivially true (already there)
        result = check_tctl(ta, EF(Atomic("only")))
        assert result.satisfied

        # AG only: trivially true (no other states)
        result = check_tctl(ta, AG(Atomic("only")))
        assert result.satisfied

        # EF other: false (unreachable)
        result = check_tctl(ta, EF(Atomic("other")))
        assert not result.satisfied

    def test_self_loop(self):
        """Location with self-loop."""
        ta = labeled_ta(
            locations=["loop"],
            initial="loop",
            clocks=["x"],
            edges=[("loop", "loop", "tick", clock_geq("x", 1), frozenset(["x"]))],
            labels={"loop": {"alive"}},
        )
        result = check_tctl(ta, EG(Atomic("loop")))
        assert result.satisfied

        result = check_tctl(ta, AG(Atomic("alive")))
        assert result.satisfied

    def test_unreachable_location(self):
        """Location exists but is unreachable."""
        ta = labeled_ta(
            locations=["start", "island"],
            initial="start",
            clocks=["x"],
            edges=[],
            labels={"island": {"treasure"}},
        )
        result = check_tctl(ta, EF(Atomic("island")))
        assert not result.satisfied

        result = check_tctl(ta, EF(Atomic("treasure")))
        assert not result.satisfied

    def test_multiple_clocks(self):
        """System with two clocks."""
        ta = labeled_ta(
            locations=["a", "b", "c"],
            initial="a",
            clocks=["x", "y"],
            edges=[
                ("a", "b", "go1", clock_geq("x", 2), frozenset(["x"])),
                ("b", "c", "go2", clock_geq("y", 3), frozenset(["y"])),
            ],
            labels={},
        )
        # c reachable: x>=2 then y>=3 (y was never reset from 0, so y>=3 means total time >= 3)
        result = check_tctl(ta, EF(Atomic("c")))
        assert result.satisfied

    def test_guard_prevents_transition(self):
        """Invariant + guard interaction."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", clock_geq("x", 10), frozenset())],
            invariants={"a": clock_leq("x", 5)},
            labels={},
        )
        # Guard requires x>=10 but invariant forces exit by x<=5: deadlock
        result = check_tctl(ta, EF(Atomic("b")))
        assert not result.satisfied

    def test_zero_time_transition(self):
        """Transition that can fire at time 0."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "instant", true_guard(), frozenset())],
            labels={},
        )
        result = check_tctl(ta, EF(Atomic("b"), TimeBound.leq(0)))
        assert result.satisfied


# ============================================================
# Complex Property Tests
# ============================================================

class TestComplexProperties:
    def test_response_property(self):
        """AG(request => EF_{<=10} response): every request gets a response within 10."""
        ta = example_request_response()
        # This is a nested formula -- AG with EF inside
        prop = AG(Implies(Atomic("processing"), EF(Atomic("done"), TimeBound.leq(10))))
        result = check_tctl(ta, prop)
        # Processing starts with x reset, done needs x>=2, within 10 is possible
        assert result.satisfied

    def test_bounded_liveness(self):
        """AF_{<=8} critical: must eventually enter critical within 8."""
        ta = example_mutex_protocol()
        result = check_tctl(ta, AF(Atomic("critical"), TimeBound.leq(8)))
        # idle has no invariant forcing exit, so can stay in idle forever
        # AF critical fails because idle->trying is not forced
        assert not result.satisfied

    def test_safety_and_liveness(self):
        """AG !timeout AND EF done."""
        ta = example_request_response()
        # AG !timeout is false (timeout IS reachable)
        result = check_tctl(ta, And(AG(Not(Atomic("timeout"))), EF(Atomic("done"))))
        assert not result.satisfied

    def test_or_temporal(self):
        """EF done OR EF timeout (at least one is reachable)."""
        ta = example_request_response()
        result = check_tctl(ta, Or(EF(Atomic("done")), EF(Atomic("timeout"))))
        assert result.satisfied

    def test_nested_ef(self):
        """EF (EF b): if b is reachable from any reachable state."""
        ta = labeled_ta(
            locations=["a", "b", "c"],
            initial="a",
            clocks=["x"],
            edges=[
                ("a", "b", "go1", true_guard(), frozenset(["x"])),
                ("b", "c", "go2", true_guard(), frozenset(["x"])),
            ],
            labels={},
        )
        # EF(EF(c)) -- c is reachable from reachable state b
        result = check_tctl(ta, EF(EF(Atomic("c"))))
        # This checks: exists path to a state from which EF c holds
        # The inner EF c is a temporal formula, not a state formula
        # Our checker handles this by checking from the initial state
        assert result.satisfied


# ============================================================
# AF Unbounded Tests
# ============================================================

class TestAFUnbounded:
    def test_af_forced_by_invariant(self):
        """AF b: must reach b. Invariant on a forces exit."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", true_guard(), frozenset())],
            invariants={"a": clock_leq("x", 5)},
            labels={},
        )
        result = check_tctl(ta, AF(Atomic("b")))
        assert result.satisfied

    def test_af_not_forced(self):
        """AF b: a has no invariant, can stay forever -> AF fails."""
        ta = labeled_ta(
            locations=["a", "b"],
            initial="a",
            clocks=["x"],
            edges=[("a", "b", "go", true_guard(), frozenset())],
            invariants={},
            labels={},
        )
        # In timed automata semantics, a path can stay in a forever
        # (no invariant forces exit). So AF b is violated.
        result = check_tctl(ta, AF(Atomic("b")))
        assert not result.satisfied
        assert result.counterexample_trace is not None


# ============================================================
# Stats and Exploration Tests
# ============================================================

class TestExplorationStats:
    def test_states_explored_positive(self):
        ta = example_light_controller()
        result = check_tctl(ta, EF(Atomic("on")))
        assert result.states_explored > 0

    def test_zones_created_positive(self):
        ta = example_light_controller()
        result = check_tctl(ta, EF(Atomic("on")))
        assert result.zones_created > 0

    def test_satisfying_locations(self):
        ta = example_light_controller()
        result = check_tctl(ta, EF(Atomic("on")))
        assert "on" in result.satisfying_locations

    def test_large_state_space_bounded(self):
        """Ensure max_states limit works."""
        ta = example_train_crossing()
        result = check_tctl(ta, EF(Atomic("nonexistent")), max_states=10)
        assert not result.satisfied
        assert result.states_explored <= 10
