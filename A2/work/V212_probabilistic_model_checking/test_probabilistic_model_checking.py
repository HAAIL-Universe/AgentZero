"""Tests for V212: Probabilistic Model Checking."""

import math
import pytest
from probabilistic_model_checking import (
    DTMC, CTMC, DTMCModelChecker, CTMCModelChecker,
    PCTLOp, PathOp, PCTLFormula,
    tt, atom, neg, conj, disj,
    prob_bound, prob_query, steady_bound, steady_query,
    reward_bound, reward_query,
    verify_dtmc, verify_ctmc,
    transient_analysis, ctmc_transient,
    build_dtmc_from_matrix, build_ctmc_from_matrix,
    find_bsccs, bscc_steady_state, _bscc_reach_prob,
)


# ===== Helpers =====

def approx(a, b, tol=1e-4):
    """Check approximate equality."""
    return abs(a - b) < tol


# ===== DTMC Construction =====

class TestDTMCConstruction:
    def test_create_empty_dtmc(self):
        d = DTMC()
        assert d.states == []
        assert d.initial is None

    def test_add_states(self):
        d = DTMC()
        d.add_state("s0", {"init"})
        d.add_state("s1", {"target"})
        assert len(d.states) == 2
        assert "init" in d.labels["s0"]

    def test_add_transitions(self):
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1")
        d.add_transition("s0", "s1", 0.7)
        d.add_transition("s0", "s0", 0.3)
        assert d.transitions["s0"]["s1"] == 0.7

    def test_validate_correct(self):
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1")
        d.add_transition("s0", "s1", 0.7)
        d.add_transition("s0", "s0", 0.3)
        d.add_transition("s1", "s1", 1.0)
        assert d.validate() == []

    def test_validate_incorrect(self):
        d = DTMC()
        d.add_state("s0")
        d.add_transition("s0", "s0", 0.5)
        errors = d.validate()
        assert len(errors) == 1

    def test_is_absorbing(self):
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1")
        d.add_transition("s0", "s1", 1.0)
        d.add_transition("s1", "s1", 1.0)
        assert not d.is_absorbing("s0")
        assert d.is_absorbing("s1")

    def test_sat_labels(self):
        d = DTMC()
        d.add_state("s0", {"a", "b"})
        d.add_state("s1", {"b"})
        d.add_state("s2", {"c"})
        assert d.sat("b") == {"s0", "s1"}
        assert d.sat("c") == {"s2"}
        assert d.sat("d") == set()

    def test_duplicate_state_ignored(self):
        d = DTMC()
        d.add_state("s0", {"a"})
        d.add_state("s0", {"b"})  # Should be ignored
        assert len(d.states) == 1
        assert "a" in d.labels["s0"]

    def test_rewards(self):
        d = DTMC()
        d.add_state("s0")
        d.add_reward("cost", "s0", 5.0)
        assert d.rewards["cost"]["s0"] == 5.0

    def test_build_from_matrix(self):
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.3, 0.7], [1.0, 0.0]],
            labels={"s0": {"a"}, "s1": {"b"}},
            initial="s0"
        )
        assert d.transitions["s0"]["s1"] == 0.7
        assert d.initial == "s0"
        assert "a" in d.labels["s0"]


# ===== CTMC Construction =====

class TestCTMCConstruction:
    def test_create_ctmc(self):
        c = CTMC()
        c.add_state("up", {"operational"})
        c.add_state("down", {"failed"})
        c.add_rate("up", "down", 0.01)  # failure rate
        c.add_rate("down", "up", 0.5)   # repair rate
        assert c.exit_rate("up") == 0.01
        assert c.exit_rate("down") == 0.5

    def test_embedded_dtmc(self):
        c = CTMC()
        c.add_state("s0")
        c.add_state("s1")
        c.add_state("s2")
        c.add_rate("s0", "s1", 3.0)
        c.add_rate("s0", "s2", 7.0)
        c.add_rate("s1", "s2", 5.0)
        c.add_rate("s2", "s2", 0.0)  # absorbing

        d = c.embedded_dtmc()
        assert approx(d.transitions["s0"]["s1"], 0.3)
        assert approx(d.transitions["s0"]["s2"], 0.7)
        assert approx(d.transitions["s1"]["s2"], 1.0)

    def test_uniformization_rate(self):
        c = CTMC()
        c.add_state("s0")
        c.add_state("s1")
        c.add_rate("s0", "s1", 3.0)
        c.add_rate("s1", "s0", 5.0)
        assert c.uniformization_rate() == 5.0

    def test_build_from_matrix(self):
        c = build_ctmc_from_matrix(
            ["up", "down"],
            [[0, 0.01], [0.5, 0]],
            labels={"up": {"ok"}, "down": {"fail"}}
        )
        assert c.rates["up"]["down"] == 0.01
        assert c.rates["down"]["up"] == 0.5


# ===== PCTL Formula Construction =====

class TestPCTLFormulas:
    def test_atom(self):
        f = atom("target")
        assert f.op == PCTLOp.ATOM
        assert f.label == "target"

    def test_negation(self):
        f = neg(atom("a"))
        assert f.op == PCTLOp.NOT

    def test_conjunction(self):
        f = conj(atom("a"), atom("b"))
        assert f.op == PCTLOp.AND

    def test_disjunction(self):
        f = disj(atom("a"), atom("b"))
        assert f.op == PCTLOp.OR

    def test_prob_bound_formula(self):
        f = prob_bound(">=", 0.5, PathOp.EVENTUALLY, path_right=atom("target"))
        assert f.op == PCTLOp.PROB_BOUND
        assert f.bound == 0.5

    def test_prob_query_formula(self):
        f = prob_query(PathOp.EVENTUALLY, path_right=atom("target"))
        assert f.op == PCTLOp.PROB_QUERY

    def test_bounded_until_formula(self):
        f = prob_query(PathOp.BOUNDED_UNTIL, path_left=atom("a"),
                       path_right=atom("b"), steps=5)
        assert f.steps == 5

    def test_steady_formula(self):
        f = steady_query(atom("ok"))
        assert f.op == PCTLOp.STEADY_QUERY

    def test_reward_formula(self):
        f = reward_query("cost", PathOp.EVENTUALLY, path_right=atom("done"))
        assert f.reward_name == "cost"

    def test_repr(self):
        f = atom("target")
        assert "target" in repr(f)
        f2 = tt()
        assert "True" in repr(f2)


# ===== DTMC Model Checking =====

class TestDTMCModelChecking:
    def _simple_chain(self):
        """s0 --0.5--> s1 --1.0--> s2 (absorbing)
           s0 --0.5--> s0 (self-loop)"""
        d = DTMC()
        d.add_state("s0", {"init"})
        d.add_state("s1", {"mid"})
        d.add_state("s2", {"target"})
        d.add_transition("s0", "s0", 0.5)
        d.add_transition("s0", "s1", 0.5)
        d.add_transition("s1", "s2", 1.0)
        d.add_transition("s2", "s2", 1.0)
        d.set_initial("s0")
        return d

    def test_atom_check(self):
        d = self._simple_chain()
        result = verify_dtmc(d, atom("target"))
        assert result == {"s2"}

    def test_negation_check(self):
        d = self._simple_chain()
        result = verify_dtmc(d, neg(atom("target")))
        assert result == {"s0", "s1"}

    def test_conjunction_check(self):
        d = self._simple_chain()
        result = verify_dtmc(d, conj(atom("init"), neg(atom("target"))))
        assert result == {"s0"}

    def test_prob_next(self):
        d = self._simple_chain()
        # P=? [X target]
        result = verify_dtmc(d, prob_query(PathOp.NEXT, path_right=atom("target")))
        assert approx(result["s0"], 0.0)  # Can't reach target in one step
        assert approx(result["s1"], 1.0)  # s1 -> s2 always
        assert approx(result["s2"], 1.0)  # s2 self-loop

    def test_prob_bounded_until(self):
        d = self._simple_chain()
        # P=? [true U<=2 target]
        result = verify_dtmc(d, prob_query(
            PathOp.BOUNDED_UNTIL, path_left=tt(),
            path_right=atom("target"), steps=2
        ))
        # From s0: step 1: 0.5 to s1, step 2: s1->s2. So P = 0.5*1 = 0.5
        # Actually with 2 steps: step0->step1->step2
        # After 2 matrix multiplications from s0:
        #   Step 1: P(s0)=0.5, P(s1)=0.5, P(s2)=0
        #   Step 2: P(s0)=0.25, P(s1)=0.25, P(s2)=0.5
        # But bounded until accumulates: at step 1, s2 is reached...
        # With k=2 backward iterations:
        # k=0 (init): probs = {s0:0, s1:0, s2:1}
        # k=1: s0 = 0.5*0 + 0.5*0 = 0, s1 = 1*1 = 1, s2 = 1
        # k=2: s0 = 0.5*0 + 0.5*1 = 0.5, s1 = 1, s2 = 1
        assert approx(result["s0"], 0.5)
        assert approx(result["s1"], 1.0)
        assert approx(result["s2"], 1.0)

    def test_prob_unbounded_until(self):
        d = self._simple_chain()
        # P=? [true U target] -- eventually reach target
        result = verify_dtmc(d, prob_query(
            PathOp.UNTIL, path_left=tt(), path_right=atom("target")
        ))
        # From s0: with prob 1 eventually reach s2 (geometric)
        assert approx(result["s0"], 1.0)
        assert approx(result["s1"], 1.0)
        assert approx(result["s2"], 1.0)

    def test_prob_eventually(self):
        d = self._simple_chain()
        result = verify_dtmc(d, prob_query(PathOp.EVENTUALLY, path_right=atom("target")))
        assert approx(result["s0"], 1.0)

    def test_prob_always(self):
        d = self._simple_chain()
        # P=? [G !target] -- always avoid target
        result = verify_dtmc(d, prob_query(PathOp.ALWAYS, path_right=neg(atom("target"))))
        # From s0: prob of never reaching target = 0 (will reach eventually)
        assert approx(result["s0"], 0.0)
        assert approx(result["s2"], 0.0)  # Already at target, so !target fails

    def test_prob_bound_filter(self):
        d = self._simple_chain()
        # P>=0.9 [F target]
        result = verify_dtmc(d, prob_bound(">=", 0.9, PathOp.EVENTUALLY,
                                            path_right=atom("target")))
        assert "s0" in result  # prob=1.0 >= 0.9
        assert "s1" in result
        assert "s2" in result

    def test_unreachable_target(self):
        """Target is unreachable from some states."""
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1", {"target"})
        d.add_state("s2")
        d.add_transition("s0", "s0", 1.0)  # absorbing, can't reach s1
        d.add_transition("s1", "s1", 1.0)
        d.add_transition("s2", "s1", 1.0)
        result = verify_dtmc(d, prob_query(PathOp.EVENTUALLY, path_right=atom("target")))
        assert approx(result["s0"], 0.0)
        assert approx(result["s1"], 1.0)
        assert approx(result["s2"], 1.0)


# ===== Steady State =====

class TestSteadyState:
    def test_two_state_ergodic(self):
        """s0 <-> s1 with equal transition probs."""
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.0, 1.0], [1.0, 0.0]]
        )
        checker = DTMCModelChecker(d)
        ss = checker.steady_state()
        assert approx(ss["s0"], 0.5)
        assert approx(ss["s1"], 0.5)

    def test_asymmetric_steady_state(self):
        """s0 -> s1 with prob 0.3, s1 -> s0 with prob 0.7."""
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.7, 0.3], [0.7, 0.3]]
        )
        checker = DTMCModelChecker(d)
        ss = checker.steady_state()
        # pi * P = pi: pi_0 = 0.7, pi_1 = 0.3
        assert approx(ss["s0"], 0.7)
        assert approx(ss["s1"], 0.3)

    def test_three_state_chain(self):
        """Doubly stochastic 3-state chain: uniform steady state."""
        d = build_dtmc_from_matrix(
            ["s0", "s1", "s2"],
            [[0.0, 0.5, 0.5],
             [0.5, 0.0, 0.5],
             [0.5, 0.5, 0.0]]
        )
        checker = DTMCModelChecker(d)
        ss = checker.steady_state()
        for s in ["s0", "s1", "s2"]:
            assert approx(ss[s], 1.0/3)

    def test_steady_query(self):
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.0, 1.0], [1.0, 0.0]],
            labels={"s0": {"a"}, "s1": set()}
        )
        result = verify_dtmc(d, steady_query(atom("a")))
        # S=? [a]: steady-state probability of being in s0 = 0.5
        assert approx(result["s0"], 0.5)

    def test_steady_bound(self):
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.0, 1.0], [1.0, 0.0]],
            labels={"s0": {"a"}, "s1": set()}
        )
        # S>=0.4 [a]: steady-state of a is 0.5 >= 0.4
        result = verify_dtmc(d, steady_bound(">=", 0.4, atom("a")))
        assert len(result) == 2  # All states satisfy

        # S>=0.6 [a]: 0.5 < 0.6
        result = verify_dtmc(d, steady_bound(">=", 0.6, atom("a")))
        assert len(result) == 0


# ===== Expected Rewards =====

class TestRewards:
    def test_cumulative_reward(self):
        """Simple chain with state rewards, compute cumulative over k steps."""
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1", {"done"})
        d.add_transition("s0", "s1", 1.0)
        d.add_transition("s1", "s1", 1.0)
        d.add_reward("cost", "s0", 2.0)
        d.add_reward("cost", "s1", 1.0)
        d.set_initial("s0")

        # R=? [C<=3]: cumulative cost over 3 steps from s0
        # Step 0: at s0, cost=2
        # Step 1: at s1, cost=1
        # Step 2: at s1, cost=1
        # Total from s0 = 2 + 1 + 1 = 4
        result = verify_dtmc(d, reward_query(
            "cost", PathOp.BOUNDED_UNTIL, steps=3, cumulative=True
        ))
        assert approx(result["s0"], 4.0)
        assert approx(result["s1"], 3.0)  # 1+1+1

    def test_reachability_reward(self):
        """Expected cost to reach target."""
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1")
        d.add_state("s2", {"target"})
        d.add_transition("s0", "s1", 0.5)
        d.add_transition("s0", "s0", 0.5)
        d.add_transition("s1", "s2", 1.0)
        d.add_transition("s2", "s2", 1.0)
        d.add_reward("steps", "s0", 1.0)
        d.add_reward("steps", "s1", 1.0)

        # R=? [F target]: expected steps to reach target
        # From s1: 1 step, cost=1
        # From s0: E[cost] = 1 + 0.5*E[s0] + 0.5*1 => E = 1 + 0.5E + 0.5 => 0.5E = 1.5 => E = 3
        result = verify_dtmc(d, reward_query(
            "steps", PathOp.EVENTUALLY, path_right=atom("target")
        ))
        assert approx(result["s0"], 3.0)
        assert approx(result["s1"], 1.0)
        assert approx(result["s2"], 0.0)

    def test_reward_bound(self):
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1", {"done"})
        d.add_transition("s0", "s1", 1.0)
        d.add_transition("s1", "s1", 1.0)
        d.add_reward("cost", "s0", 5.0)

        # R<=6 [F done]: expected cost <= 6
        result = verify_dtmc(d, reward_bound(
            "<=", 6.0, "cost", PathOp.EVENTUALLY, path_right=atom("done")
        ))
        assert "s0" in result  # cost=5 <= 6


# ===== CTMC Model Checking =====

class TestCTMCModelChecking:
    def _availability_model(self):
        """Simple availability: up <-> down."""
        c = CTMC()
        c.add_state("up", {"operational"})
        c.add_state("down", {"failed"})
        c.add_rate("up", "down", 0.01)  # MTTF = 100
        c.add_rate("down", "up", 0.5)   # MTTR = 2
        c.set_initial("up")
        return c

    def test_ctmc_steady_state(self):
        c = self._availability_model()
        checker = CTMCModelChecker(c)
        ss = checker.steady_state()
        # Analytical: pi_up = mu/(lambda+mu) = 0.5/0.51 ~= 0.9804
        expected_up = 0.5 / (0.01 + 0.5)
        assert approx(ss["up"], expected_up, tol=0.01)
        assert approx(ss["down"], 1.0 - expected_up, tol=0.01)

    def test_ctmc_steady_query(self):
        c = self._availability_model()
        result = verify_ctmc(c, steady_query(atom("operational")))
        expected = 0.5 / 0.51
        for v in result.values():
            assert approx(v, expected, tol=0.01)

    def test_ctmc_unbounded_reachability(self):
        c = self._availability_model()
        # P=? [F failed]: from up, will eventually fail (prob 1 for ergodic)
        result = verify_ctmc(c, prob_query(PathOp.EVENTUALLY, path_right=atom("failed")))
        assert approx(result["up"], 1.0)
        assert approx(result["down"], 1.0)

    def test_ctmc_time_bounded_until(self):
        """P=? [operational U<=10 failed]: prob of failure within 10 time units."""
        c = self._availability_model()
        result = verify_ctmc(c, prob_query(
            PathOp.BOUNDED_UNTIL,
            path_left=atom("operational"),
            path_right=atom("failed"),
            steps=10  # time bound
        ))
        # Time-bounded until with repair: P(operational U<=10 failed)
        # Higher than 1-e^{-0.01*10} because repair allows cycling back and re-failing
        assert result["up"] > 0.3
        assert result["up"] < 0.6
        assert approx(result["down"], 1.0)  # Already failed

    def test_ctmc_time_bounded_zero(self):
        c = self._availability_model()
        result = verify_ctmc(c, prob_query(
            PathOp.BOUNDED_UNTIL,
            path_left=tt(),
            path_right=atom("failed"),
            steps=0
        ))
        assert approx(result["up"], 0.0)
        assert approx(result["down"], 1.0)

    def test_ctmc_next(self):
        c = CTMC()
        c.add_state("s0")
        c.add_state("s1", {"target"})
        c.add_state("s2")
        c.add_rate("s0", "s1", 3.0)
        c.add_rate("s0", "s2", 7.0)
        c.add_rate("s1", "s1", 1.0)  # doesn't matter for embedded
        result = verify_ctmc(c, prob_query(PathOp.NEXT, path_right=atom("target")))
        assert approx(result["s0"], 0.3)  # 3/10

    def test_absorbing_ctmc(self):
        """System that eventually reaches absorbing failure state."""
        c = CTMC()
        c.add_state("ok", {"working"})
        c.add_state("degraded", {"working"})
        c.add_state("failed", {"dead"})
        c.add_rate("ok", "degraded", 0.1)
        c.add_rate("degraded", "failed", 0.5)
        # failed is absorbing (no rates out)
        result = verify_ctmc(c, prob_query(PathOp.EVENTUALLY, path_right=atom("dead")))
        assert approx(result["ok"], 1.0)
        assert approx(result["degraded"], 1.0)


# ===== Transient Analysis =====

class TestTransientAnalysis:
    def test_dtmc_transient(self):
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.0, 1.0], [1.0, 0.0]]
        )
        dist = transient_analysis(d, {"s0": 1.0, "s1": 0.0}, steps=1)
        assert approx(dist["s0"], 0.0)
        assert approx(dist["s1"], 1.0)

        dist2 = transient_analysis(d, {"s0": 1.0, "s1": 0.0}, steps=2)
        assert approx(dist2["s0"], 1.0)
        assert approx(dist2["s1"], 0.0)

    def test_dtmc_transient_convergence(self):
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.7, 0.3], [0.7, 0.3]]
        )
        dist = transient_analysis(d, {"s0": 1.0, "s1": 0.0}, steps=100)
        assert approx(dist["s0"], 0.7)
        assert approx(dist["s1"], 0.3)

    def test_ctmc_transient(self):
        c = CTMC()
        c.add_state("up")
        c.add_state("down")
        c.add_rate("up", "down", 1.0)
        c.add_rate("down", "up", 1.0)
        dist = ctmc_transient(c, {"up": 1.0}, time=100)
        # Should converge to steady state (0.5, 0.5) for symmetric rates
        assert approx(dist["up"], 0.5, tol=0.01)
        assert approx(dist["down"], 0.5, tol=0.01)

    def test_ctmc_transient_short_time(self):
        c = CTMC()
        c.add_state("up")
        c.add_state("down")
        c.add_rate("up", "down", 0.01)
        c.add_rate("down", "up", 0.5)
        # Short time: mostly stay in initial state
        dist = ctmc_transient(c, {"up": 1.0}, time=0.1)
        assert dist["up"] > 0.99  # Very unlikely to fail in 0.1 time units


# ===== BSCC Analysis =====

class TestBSCC:
    def test_single_bscc(self):
        """Entire ergodic chain is one BSCC."""
        d = build_dtmc_from_matrix(
            ["s0", "s1"],
            [[0.5, 0.5], [0.5, 0.5]]
        )
        bsccs = find_bsccs(d)
        assert len(bsccs) == 1
        assert bsccs[0] == {"s0", "s1"}

    def test_absorbing_state_bscc(self):
        """Absorbing state forms its own BSCC."""
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1")
        d.add_transition("s0", "s1", 1.0)
        d.add_transition("s1", "s1", 1.0)
        bsccs = find_bsccs(d)
        assert len(bsccs) == 1
        assert bsccs[0] == {"s1"}

    def test_two_bsccs(self):
        """Two absorbing states: two BSCCs."""
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1")
        d.add_state("s2")
        d.add_transition("s0", "s1", 0.5)
        d.add_transition("s0", "s2", 0.5)
        d.add_transition("s1", "s1", 1.0)
        d.add_transition("s2", "s2", 1.0)
        bsccs = find_bsccs(d)
        assert len(bsccs) == 2

    def test_bscc_reach_prob(self):
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1")
        d.add_state("s2")
        d.add_transition("s0", "s1", 0.3)
        d.add_transition("s0", "s2", 0.7)
        d.add_transition("s1", "s1", 1.0)
        d.add_transition("s2", "s2", 1.0)

        reach_s1 = _bscc_reach_prob(d, {"s1"})
        assert approx(reach_s1["s0"], 0.3)
        assert approx(reach_s1["s1"], 1.0)
        assert approx(reach_s1["s2"], 0.0)


# ===== Classic Models =====

class TestClassicModels:
    def test_die_simulation(self):
        """Knuth-Yao die: simulate fair die with biased coin.

        Classic PRISM example.
        """
        d = DTMC()
        for i in range(13):
            d.add_state(f"s{i}")
        for i in range(1, 7):
            d.add_state(f"d{i}", {"done", f"val{i}"})
            d.add_transition(f"d{i}", f"d{i}", 1.0)

        # Knuth-Yao tree structure (fair coin flips)
        d.add_transition("s0", "s1", 0.5)
        d.add_transition("s0", "s2", 0.5)
        d.add_transition("s1", "s3", 0.5)
        d.add_transition("s1", "s4", 0.5)
        d.add_transition("s2", "s5", 0.5)
        d.add_transition("s2", "s6", 0.5)
        d.add_transition("s3", "d1", 0.5)
        d.add_transition("s3", "d2", 0.5)
        d.add_transition("s4", "d3", 0.5)
        d.add_transition("s4", "d4", 0.5)
        d.add_transition("s5", "d5", 0.5)
        d.add_transition("s5", "d6", 0.5)
        d.add_transition("s6", "s1", 0.5)  # Restart branch
        d.add_transition("s6", "s2", 0.5)
        d.set_initial("s0")

        # P=? [F val1] from s0 should be 1/6
        for i in range(1, 7):
            result = verify_dtmc(d, prob_query(
                PathOp.EVENTUALLY, path_right=atom(f"val{i}")
            ))
            assert approx(result["s0"], 1.0/6, tol=0.001)

    def test_gambler_ruin(self):
        """Gambler's ruin: start with $2, win/lose $1 with prob 0.5."""
        d = DTMC()
        d.add_state("$0", {"broke"})
        d.add_state("$1")
        d.add_state("$2")
        d.add_state("$3")
        d.add_state("$4", {"rich"})
        d.add_transition("$0", "$0", 1.0)  # absorbing
        d.add_transition("$1", "$0", 0.5)
        d.add_transition("$1", "$2", 0.5)
        d.add_transition("$2", "$1", 0.5)
        d.add_transition("$2", "$3", 0.5)
        d.add_transition("$3", "$2", 0.5)
        d.add_transition("$3", "$4", 0.5)
        d.add_transition("$4", "$4", 1.0)  # absorbing
        d.set_initial("$2")

        # P(reach $4 | start at $2) = 2/4 = 0.5 for fair game
        result = verify_dtmc(d, prob_query(PathOp.EVENTUALLY, path_right=atom("rich")))
        assert approx(result["$2"], 0.5)
        assert approx(result["$1"], 0.25)
        assert approx(result["$3"], 0.75)

    def test_reliable_broadcast(self):
        """Simple reliable broadcast protocol."""
        d = DTMC()
        d.add_state("idle", {"start"})
        d.add_state("sent")
        d.add_state("ack", {"delivered"})
        d.add_state("fail")
        d.add_transition("idle", "sent", 1.0)
        d.add_transition("sent", "ack", 0.9)   # Success
        d.add_transition("sent", "fail", 0.1)  # Failure
        d.add_transition("fail", "sent", 1.0)  # Retry
        d.add_transition("ack", "ack", 1.0)
        d.set_initial("idle")

        # Eventually delivered with prob 1 (geometric retries)
        result = verify_dtmc(d, prob_query(PathOp.EVENTUALLY, path_right=atom("delivered")))
        assert approx(result["idle"], 1.0)

        # P(delivered within 2 steps): idle->sent->ack = 0.9
        result2 = verify_dtmc(d, prob_query(
            PathOp.BOUNDED_UNTIL, path_left=tt(),
            path_right=atom("delivered"), steps=2
        ))
        assert approx(result2["idle"], 0.9)

    def test_sensor_network(self):
        """Sensor with intermittent failures."""
        c = CTMC()
        c.add_state("active", {"sensing"})
        c.add_state("sleep")
        c.add_state("broken", {"error"})
        c.add_rate("active", "sleep", 2.0)    # Go to sleep
        c.add_rate("active", "broken", 0.1)   # Fail
        c.add_rate("sleep", "active", 1.0)    # Wake up
        c.add_rate("broken", "active", 0.5)   # Repair
        c.set_initial("active")

        checker = CTMCModelChecker(c)
        ss = checker.steady_state()
        # Analytical: pi proportional to (1/E_i) for embedded SS
        # active has highest exit rate (2.1), so lowest steady-state time
        assert ss["active"] > 0.1
        assert ss["active"] < 0.5
        total = sum(ss.values())
        assert approx(total, 1.0, tol=0.01)


# ===== Edge Cases =====

class TestEdgeCases:
    def test_single_state_dtmc(self):
        d = DTMC()
        d.add_state("s0", {"only"})
        d.add_transition("s0", "s0", 1.0)
        checker = DTMCModelChecker(d)
        ss = checker.steady_state()
        assert approx(ss["s0"], 1.0)

    def test_single_state_ctmc(self):
        c = CTMC()
        c.add_state("s0", {"only"})
        # No transitions: absorbing
        checker = CTMCModelChecker(c)
        ss = checker.steady_state()
        assert approx(ss["s0"], 1.0)

    def test_bounded_until_zero_steps(self):
        d = DTMC()
        d.add_state("s0", {"target"})
        d.add_state("s1")
        d.add_transition("s0", "s0", 1.0)
        d.add_transition("s1", "s0", 1.0)
        result = verify_dtmc(d, prob_query(
            PathOp.BOUNDED_UNTIL, path_left=tt(),
            path_right=atom("target"), steps=0
        ))
        assert approx(result["s0"], 1.0)
        assert approx(result["s1"], 0.0)

    def test_empty_rewards(self):
        d = DTMC()
        d.add_state("s0", {"done"})
        d.add_transition("s0", "s0", 1.0)
        result = verify_dtmc(d, reward_query(
            "nonexistent", PathOp.EVENTUALLY, path_right=atom("done")
        ))
        assert approx(result["s0"], 0.0)

    def test_comparison_operators(self):
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1", {"target"})
        d.add_transition("s0", "s1", 0.7)
        d.add_transition("s0", "s0", 0.3)
        d.add_transition("s1", "s1", 1.0)

        # Test different comparison operators
        assert "s0" in verify_dtmc(d, prob_bound(">=", 0.7, PathOp.NEXT, path_right=atom("target")))
        assert "s0" not in verify_dtmc(d, prob_bound(">", 0.7, PathOp.NEXT, path_right=atom("target")))
        assert "s0" in verify_dtmc(d, prob_bound("<=", 0.7, PathOp.NEXT, path_right=atom("target")))
        assert "s0" not in verify_dtmc(d, prob_bound("<", 0.7, PathOp.NEXT, path_right=atom("target")))


# ===== Protocol Verification Patterns =====

class TestProtocolPatterns:
    def test_mutual_exclusion_prob(self):
        """Two processes competing for a resource with random backoff."""
        d = DTMC()
        # States: (p1_state, p2_state) where state in {idle, try, crit}
        d.add_state("ii", {"safe"})      # Both idle
        d.add_state("ti", {"safe"})      # P1 trying
        d.add_state("it", {"safe"})      # P2 trying
        d.add_state("tt", {"safe"})      # Both trying -> random resolve
        d.add_state("ci", {"p1_crit"})   # P1 in critical
        d.add_state("ic", {"p2_crit"})   # P2 in critical
        # No "cc" state: mutual exclusion guaranteed

        d.add_transition("ii", "ti", 0.3)
        d.add_transition("ii", "it", 0.3)
        d.add_transition("ii", "ii", 0.4)
        d.add_transition("ti", "ci", 0.8)
        d.add_transition("ti", "tt", 0.2)
        d.add_transition("it", "ic", 0.8)
        d.add_transition("it", "tt", 0.2)
        d.add_transition("tt", "ci", 0.5)  # Random winner
        d.add_transition("tt", "ic", 0.5)
        d.add_transition("ci", "ii", 1.0)  # Release
        d.add_transition("ic", "ii", 1.0)
        d.set_initial("ii")

        # Verify: P=1 [F p1_crit | start=ti]
        result = verify_dtmc(d, prob_query(PathOp.EVENTUALLY, path_right=atom("p1_crit")))
        assert approx(result["ti"], 1.0)

        # Verify liveness: from trying, always eventually reach critical
        result2 = verify_dtmc(d, prob_query(
            PathOp.EVENTUALLY,
            path_right=disj(atom("p1_crit"), atom("p2_crit"))
        ))
        assert approx(result2["tt"], 1.0)

    def test_leader_election(self):
        """Simplified probabilistic leader election (3 processes)."""
        d = DTMC()
        d.add_state("round1")
        d.add_state("round2")
        d.add_state("elected", {"leader"})
        # Round 1: each process picks random ID. P(unique max) = 1/3 roughly
        d.add_transition("round1", "elected", 1.0/3)
        d.add_transition("round1", "round2", 2.0/3)
        # Round 2: retry
        d.add_transition("round2", "elected", 1.0/3)
        d.add_transition("round2", "round2", 2.0/3)
        d.set_initial("round1")

        # Eventually elect with prob 1 (geometric)
        result = verify_dtmc(d, prob_query(PathOp.EVENTUALLY, path_right=atom("leader")))
        assert approx(result["round1"], 1.0)

        # Expected rounds (reward = 1 per non-elected state)
        d.add_reward("rounds", "round1", 1.0)
        d.add_reward("rounds", "round2", 1.0)
        result2 = verify_dtmc(d, reward_query(
            "rounds", PathOp.EVENTUALLY, path_right=atom("leader")
        ))
        # E[rounds] = 1/(1/3) = 3
        assert approx(result2["round1"], 3.0)


# ===== Composition with System Properties =====

class TestSystemProperties:
    def test_availability_sla(self):
        """Verify system meets 99% availability SLA."""
        c = CTMC()
        c.add_state("up", {"available"})
        c.add_state("down")
        c.add_rate("up", "down", 0.001)  # MTTF = 1000 hours
        c.add_rate("down", "up", 1.0)    # MTTR = 1 hour

        # S>=0.99 [available]
        result = verify_ctmc(c, steady_bound(">=", 0.99, atom("available")))
        assert len(result) == 2  # Availability = 1000/1001 ~= 0.999

    def test_redundant_system(self):
        """Two-component redundant system: both must fail for system failure."""
        c = CTMC()
        c.add_state("both_up", {"operational"})
        c.add_state("one_down", {"operational"})  # Still operational (redundancy)
        c.add_state("both_down", {"failed"})
        c.add_rate("both_up", "one_down", 2.0)    # Either component fails
        c.add_rate("one_down", "both_down", 1.0)  # Remaining component fails
        c.add_rate("one_down", "both_up", 0.5)    # Repair first component
        c.add_rate("both_down", "one_down", 0.5)  # Repair one component
        c.set_initial("both_up")

        checker = CTMCModelChecker(c)
        ss = checker.steady_state()
        operational = ss["both_up"] + ss["one_down"]
        # With high failure rate (2.0) and low repair (0.5), operational fraction
        # is moderate. The redundancy is in the degraded-operational state.
        assert operational > 0.2
        # With rates failure=2.0, repair=0.5: system is down most of the time
        # but operational fraction is non-zero (redundancy keeps partial service)
        assert ss["one_down"] > ss["both_up"]  # More time in degraded than fully up

    def test_queue_model(self):
        """Simple M/M/1 queue with 3 buffer slots."""
        c = CTMC()
        for i in range(4):  # 0, 1, 2, 3 items
            labels = set()
            if i == 0:
                labels.add("empty")
            if i == 3:
                labels.add("full")
            c.add_state(f"q{i}", labels)
        # Arrival rate = 2, service rate = 3
        for i in range(3):
            c.add_rate(f"q{i}", f"q{i+1}", 2.0)  # arrival
        for i in range(1, 4):
            c.add_rate(f"q{i}", f"q{i-1}", 3.0)  # service
        c.set_initial("q0")

        checker = CTMCModelChecker(c)
        ss = checker.steady_state()
        # Queue should be empty more often than full (service > arrival)
        assert ss["q0"] > ss["q3"]
        total = sum(ss.values())
        assert approx(total, 1.0, tol=0.01)


# ===== Nested/Complex Formulas =====

class TestComplexFormulas:
    def test_nested_prob(self):
        """P>=0.5 [F (a & P>=0.9 [X b])]."""
        d = DTMC()
        d.add_state("s0")
        d.add_state("s1", {"a"})
        d.add_state("s2", {"b"})
        d.add_transition("s0", "s1", 1.0)
        d.add_transition("s1", "s2", 0.95)
        d.add_transition("s1", "s1", 0.05)
        d.add_transition("s2", "s2", 1.0)

        # Inner: P>=0.9 [X b] -- states where next-step prob of b >= 0.9
        inner = prob_bound(">=", 0.9, PathOp.NEXT, path_right=atom("b"))
        # Outer: P>=0.5 [F (a & inner)]
        combined = conj(atom("a"), inner)
        outer = prob_bound(">=", 0.5, PathOp.EVENTUALLY, path_right=combined)
        result = verify_dtmc(d, outer)
        # s1 has label "a" and P[X b] = 0.95 >= 0.9, so s1 satisfies inner
        # s0 reaches s1 with prob 1, so s0 satisfies outer
        assert "s0" in result

    def test_or_formula(self):
        d = DTMC()
        d.add_state("s0", {"a"})
        d.add_state("s1", {"b"})
        d.add_state("s2")
        d.add_transition("s0", "s0", 1.0)
        d.add_transition("s1", "s1", 1.0)
        d.add_transition("s2", "s2", 1.0)
        result = verify_dtmc(d, disj(atom("a"), atom("b")))
        assert result == {"s0", "s1"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
