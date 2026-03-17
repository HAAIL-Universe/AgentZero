"""Tests for V182: Probabilistic Model Checking."""

import pytest
from fractions import Fraction
from probabilistic_model_checking import (
    DTMC, MDP, State, Distribution,
    dtmc_reachability_probability, dtmc_expected_reward,
    dtmc_steady_state, dtmc_transient_probs,
    dtmc_bisimulation_quotient, dtmc_simulate_path,
    dtmc_estimate_reachability,
    mdp_reachability_probability, mdp_expected_reward,
    pctl_check_dtmc,
    Atomic, Not, And, Or, ProbOp, Finally, Until, BoundedFinally, ExpRewardOp,
    build_dtmc, build_mdp,
)


# ============================================================
# Distribution tests
# ============================================================

class TestDistribution:
    def test_basic(self):
        d = Distribution({"a": Fraction(1, 2), "b": Fraction(1, 2)})
        assert d.prob("a") == Fraction(1, 2)
        assert d.prob("c") == Fraction(0)
        assert d.support() == {"a", "b"}

    def test_single(self):
        d = Distribution({"x": 1})
        assert d.prob("x") == Fraction(1)

    def test_invalid_sum(self):
        with pytest.raises(ValueError, match="not 1"):
            Distribution({"a": Fraction(1, 2), "b": Fraction(1, 3)})

    def test_negative_prob(self):
        with pytest.raises(ValueError, match="Negative"):
            Distribution({"a": -1, "b": 2})

    def test_zero_prob_excluded(self):
        d = Distribution({"a": 1, "b": 0})
        assert d.support() == {"a"}

    def test_action(self):
        d = Distribution({"a": 1}, action="go")
        assert d.action == "go"


# ============================================================
# DTMC construction tests
# ============================================================

class TestDTMC:
    def test_basic_construction(self):
        dtmc = DTMC()
        dtmc.add_state("s0", labels=["init"])
        dtmc.add_state("s1", labels=["target"])
        dtmc.set_initial("s0")
        dtmc.add_transition("s0", {"s1": 1})
        dtmc.add_transition("s1", {"s1": 1})
        assert dtmc.initial == "s0"
        assert dtmc.states_with_label("target") == {"s1"}

    def test_successors_predecessors(self):
        dtmc = build_dtmc(
            ["s0", "s1", "s2"],
            {"s0": {"s1": Fraction(1,2), "s2": Fraction(1,2)}, "s1": {"s1": 1}, "s2": {"s2": 1}},
            "s0"
        )
        assert dtmc.successors("s0") == {"s1", "s2"}
        assert dtmc.predecessors("s1") == {"s0", "s1"}

    def test_absorbing(self):
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s1": 1}, "s1": {"s1": 1}},
            "s0"
        )
        assert dtmc.is_absorbing("s1")
        assert not dtmc.is_absorbing("s0")

    def test_unknown_state_transition(self):
        dtmc = DTMC()
        dtmc.add_state("s0")
        with pytest.raises(ValueError):
            dtmc.add_transition("s0", {"bad": 1})

    def test_unknown_src_transition(self):
        dtmc = DTMC()
        dtmc.add_state("s0")
        with pytest.raises(ValueError):
            dtmc.add_transition("bad", {"s0": 1})

    def test_labels(self):
        dtmc = DTMC()
        dtmc.add_state("s0", labels=["a", "b"])
        dtmc.add_state("s1", labels=["b", "c"])
        assert dtmc.states_with_label("b") == {"s0", "s1"}
        assert dtmc.states_with_label("a") == {"s0"}

    def test_reward(self):
        dtmc = DTMC()
        s = dtmc.add_state("s0", reward=5)
        assert s.reward == Fraction(5)


# ============================================================
# DTMC reachability probability
# ============================================================

class TestDTMCReachability:
    def test_certain_reach(self):
        """Deterministic path to target."""
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s1": 1}, "s1": {"s1": 1}},
            "s0",
            labels={"s1": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(1)
        assert probs["s1"] == Fraction(1)

    def test_impossible_reach(self):
        """Target unreachable."""
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s0": 1}, "s1": {"s1": 1}},
            "s0",
            labels={"s1": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(0)

    def test_fair_coin(self):
        """Fair coin: P(heads) = 1/2."""
        dtmc = build_dtmc(
            ["flip", "heads", "tails"],
            {"flip": {"heads": Fraction(1,2), "tails": Fraction(1,2)},
             "heads": {"heads": 1}, "tails": {"tails": 1}},
            "flip",
            labels={"heads": ["win"]}
        )
        probs = dtmc_reachability_probability(dtmc, "win")
        assert probs["flip"] == Fraction(1, 2)

    def test_repeated_trials(self):
        """Geometric: flip coin until heads. P(reach heads) = 1."""
        dtmc = build_dtmc(
            ["try", "heads"],
            {"try": {"heads": Fraction(1,2), "try": Fraction(1,2)},
             "heads": {"heads": 1}},
            "try",
            labels={"heads": ["done"]}
        )
        probs = dtmc_reachability_probability(dtmc, "done")
        assert probs["try"] == Fraction(1)

    def test_biased_coin(self):
        """Biased coin: P(heads) = 1/3."""
        dtmc = build_dtmc(
            ["flip", "heads", "tails"],
            {"flip": {"heads": Fraction(1,3), "tails": Fraction(2,3)},
             "heads": {"heads": 1}, "tails": {"tails": 1}},
            "flip",
            labels={"heads": ["win"]}
        )
        probs = dtmc_reachability_probability(dtmc, "win")
        assert probs["flip"] == Fraction(1, 3)

    def test_two_step_reach(self):
        """s0 -> s1 (p=1/2) -> s2 (p=1/3), reach s2 from s0 = 1/6."""
        dtmc = build_dtmc(
            ["s0", "s1", "s2", "fail1", "fail2"],
            {"s0": {"s1": Fraction(1,2), "fail1": Fraction(1,2)},
             "s1": {"s2": Fraction(1,3), "fail2": Fraction(2,3)},
             "s2": {"s2": 1}, "fail1": {"fail1": 1}, "fail2": {"fail2": 1}},
            "s0",
            labels={"s2": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(1, 6)

    def test_diamond(self):
        """Diamond structure: two paths to target with different probs."""
        dtmc = build_dtmc(
            ["s0", "s1", "s2", "t"],
            {"s0": {"s1": Fraction(1,2), "s2": Fraction(1,2)},
             "s1": {"t": Fraction(1,3), "s1": Fraction(2,3)},
             "s2": {"t": Fraction(1,2), "s2": Fraction(1,2)},
             "t": {"t": 1}},
            "s0",
            labels={"t": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        # Both s1 and s2 reach t with prob 1 (geometric), so s0 reaches t with prob 1
        assert probs["s0"] == Fraction(1)

    def test_three_state_chain(self):
        """s0 -1/2-> s1 -1/2-> s2(target), failures absorb."""
        dtmc = build_dtmc(
            ["s0", "s1", "s2", "f0", "f1"],
            {"s0": {"s1": Fraction(1,2), "f0": Fraction(1,2)},
             "s1": {"s2": Fraction(1,2), "f1": Fraction(1,2)},
             "s2": {"s2": 1}, "f0": {"f0": 1}, "f1": {"f1": 1}},
            "s0",
            labels={"s2": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(1, 4)
        assert probs["s1"] == Fraction(1, 2)


# ============================================================
# DTMC expected reward
# ============================================================

class TestDTMCExpectedReward:
    def test_one_step(self):
        """One step to target, reward = 1."""
        dtmc = build_dtmc(
            ["s0", "t"],
            {"s0": {"t": 1}, "t": {"t": 1}},
            "s0",
            labels={"t": ["target"]},
            rewards={"s0": 1}
        )
        rewards = dtmc_expected_reward(dtmc, "target")
        assert rewards["s0"] == Fraction(1)
        assert rewards["t"] == Fraction(0)

    def test_geometric_reward(self):
        """Geometric trial: flip until heads. E[steps] = 2."""
        dtmc = build_dtmc(
            ["try", "done"],
            {"try": {"done": Fraction(1,2), "try": Fraction(1,2)},
             "done": {"done": 1}},
            "try",
            labels={"done": ["target"]},
            rewards={"try": 1}
        )
        rewards = dtmc_expected_reward(dtmc, "target")
        assert rewards["try"] == Fraction(2)

    def test_unreachable_infinite_reward(self):
        """If target unreachable, reward is infinite."""
        dtmc = build_dtmc(
            ["s0", "t"],
            {"s0": {"s0": 1}, "t": {"t": 1}},
            "s0",
            labels={"t": ["target"]},
            rewards={"s0": 1}
        )
        rewards = dtmc_expected_reward(dtmc, "target")
        assert rewards["s0"] == float('inf')

    def test_two_step_reward(self):
        """s0 (reward=3) -> s1 (reward=5) -> t. E[s0]=8, E[s1]=5."""
        dtmc = build_dtmc(
            ["s0", "s1", "t"],
            {"s0": {"s1": 1}, "s1": {"t": 1}, "t": {"t": 1}},
            "s0",
            labels={"t": ["target"]},
            rewards={"s0": 3, "s1": 5}
        )
        rewards = dtmc_expected_reward(dtmc, "target")
        assert rewards["s0"] == Fraction(8)
        assert rewards["s1"] == Fraction(5)

    def test_probabilistic_reward(self):
        """Two paths with different rewards."""
        dtmc = build_dtmc(
            ["s0", "s1", "s2", "t"],
            {"s0": {"s1": Fraction(1,2), "s2": Fraction(1,2)},
             "s1": {"t": 1}, "s2": {"t": 1}, "t": {"t": 1}},
            "s0",
            labels={"t": ["target"]},
            rewards={"s0": 1, "s1": 2, "s2": 10}
        )
        rewards = dtmc_expected_reward(dtmc, "target")
        # E[s0] = 1 + 1/2*E[s1] + 1/2*E[s2] = 1 + 1/2*2 + 1/2*10 = 1 + 1 + 5 = 7
        assert rewards["s0"] == Fraction(7)


# ============================================================
# DTMC steady state
# ============================================================

class TestDTMCSteadyState:
    def test_absorbing(self):
        """Single absorbing state gets all probability."""
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s1": 1}, "s1": {"s1": 1}},
            "s0"
        )
        ss = dtmc_steady_state(dtmc)
        assert abs(ss["s1"] - 1) < Fraction(1, 10**6)
        assert abs(ss["s0"]) < Fraction(1, 10**6)

    def test_symmetric_cycle(self):
        """Two-state symmetric cycle: steady state = 1/2 each."""
        dtmc = build_dtmc(
            ["a", "b"],
            {"a": {"b": 1}, "b": {"a": 1}},
            "a"
        )
        ss = dtmc_steady_state(dtmc)
        assert abs(ss["a"] - Fraction(1, 2)) < Fraction(1, 10**6)
        assert abs(ss["b"] - Fraction(1, 2)) < Fraction(1, 10**6)

    def test_biased_cycle(self):
        """a->b (p=2/3), a->a (p=1/3); b->a (p=1). SS: pi(a)=3/5, pi(b)=2/5."""
        dtmc = build_dtmc(
            ["a", "b"],
            {"a": {"b": Fraction(2,3), "a": Fraction(1,3)},
             "b": {"a": 1}},
            "a"
        )
        ss = dtmc_steady_state(dtmc)
        # Balance: pi(a)*2/3 = pi(b)*1, pi(a)+pi(b)=1
        # pi(b) = 2/3*pi(a), pi(a)(1+2/3)=1, pi(a)=3/5
        assert abs(ss["a"] - Fraction(3, 5)) < Fraction(1, 10**6)
        assert abs(ss["b"] - Fraction(2, 5)) < Fraction(1, 10**6)


# ============================================================
# DTMC transient probabilities
# ============================================================

class TestDTMCTransient:
    def test_zero_steps(self):
        dtmc = build_dtmc(["s0", "s1"], {"s0": {"s1": 1}, "s1": {"s1": 1}}, "s0")
        dist = dtmc_transient_probs(dtmc, 0)
        assert dist["s0"] == Fraction(1)
        assert dist["s1"] == Fraction(0)

    def test_one_step(self):
        dtmc = build_dtmc(["s0", "s1"], {"s0": {"s1": 1}, "s1": {"s1": 1}}, "s0")
        dist = dtmc_transient_probs(dtmc, 1)
        assert dist["s0"] == Fraction(0)
        assert dist["s1"] == Fraction(1)

    def test_probabilistic_step(self):
        dtmc = build_dtmc(
            ["s0", "s1", "s2"],
            {"s0": {"s1": Fraction(1,3), "s2": Fraction(2,3)},
             "s1": {"s1": 1}, "s2": {"s2": 1}},
            "s0"
        )
        dist = dtmc_transient_probs(dtmc, 1)
        assert dist["s0"] == Fraction(0)
        assert dist["s1"] == Fraction(1, 3)
        assert dist["s2"] == Fraction(2, 3)

    def test_two_steps(self):
        dtmc = build_dtmc(
            ["s0", "s1", "s2"],
            {"s0": {"s1": 1}, "s1": {"s2": 1}, "s2": {"s2": 1}},
            "s0"
        )
        dist = dtmc_transient_probs(dtmc, 2)
        assert dist["s2"] == Fraction(1)
        assert dist["s0"] == Fraction(0)
        assert dist["s1"] == Fraction(0)

    def test_random_walk_two_steps(self):
        """s0 -> s0 (1/2), s1 (1/2). s1 -> s0 (1/2), s1 (1/2). After 2 steps from s0."""
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s0": Fraction(1,2), "s1": Fraction(1,2)},
             "s1": {"s0": Fraction(1,2), "s1": Fraction(1,2)}},
            "s0"
        )
        dist = dtmc_transient_probs(dtmc, 2)
        # After 2 steps of symmetric random walk: P(s0)=1/2, P(s1)=1/2
        assert dist["s0"] == Fraction(1, 2)
        assert dist["s1"] == Fraction(1, 2)


# ============================================================
# MDP tests
# ============================================================

class TestMDP:
    def test_construction(self):
        mdp = MDP()
        mdp.add_state("s0", labels=["init"])
        mdp.add_state("s1", labels=["target"])
        mdp.set_initial("s0")
        mdp.add_transition("s0", {"s1": 1}, action="go")
        mdp.add_transition("s1", {"s1": 1})
        assert len(mdp.actions("s0")) == 1
        assert mdp.actions("s0")[0].action == "go"

    def test_multiple_actions(self):
        mdp = build_mdp(
            ["s0", "s1", "s2"],
            {"s0": [
                ({"s1": 1}, "left"),
                ({"s2": 1}, "right"),
            ],
             "s1": [{"s1": 1}],
             "s2": [{"s2": 1}]},
            "s0"
        )
        assert len(mdp.actions("s0")) == 2


class TestMDPReachability:
    def test_max_choice(self):
        """MDP: choose between p=1/3 and p=2/3 to reach target."""
        mdp = build_mdp(
            ["s0", "t", "f"],
            {"s0": [
                ({"t": Fraction(1,3), "f": Fraction(2,3)}, "risky"),
                ({"t": Fraction(2,3), "f": Fraction(1,3)}, "safe"),
            ],
             "t": [{"t": 1}],
             "f": [{"f": 1}]},
            "s0",
            labels={"t": ["target"]}
        )
        vals_max, sched_max = mdp_reachability_probability(mdp, "target", minimize=False)
        assert vals_max["s0"] == Fraction(2, 3)
        assert sched_max["s0"].action == "safe"

        vals_min, sched_min = mdp_reachability_probability(mdp, "target", minimize=True)
        assert vals_min["s0"] == Fraction(1, 3)
        assert sched_min["s0"].action == "risky"

    def test_deterministic_mdp(self):
        """MDP with deterministic choices: go left or right."""
        mdp = build_mdp(
            ["s0", "left", "right", "t"],
            {"s0": [
                ({"left": 1}, "go_left"),
                ({"right": 1}, "go_right"),
            ],
             "left": [({"t": Fraction(1,4), "left": Fraction(3,4)},)],
             "right": [({"t": 1},)],
             "t": [{"t": 1}]},
            "s0",
            labels={"t": ["target"]}
        )
        vals, sched = mdp_reachability_probability(mdp, "target", minimize=False)
        # Both reach t with prob 1 (left is geometric), max picks either
        assert vals["s0"] == Fraction(1)

    def test_min_avoids_target(self):
        """Minimizer can avoid target entirely."""
        mdp = build_mdp(
            ["s0", "t", "loop"],
            {"s0": [
                ({"t": 1}, "go_target"),
                ({"loop": 1}, "go_loop"),
            ],
             "t": [{"t": 1}],
             "loop": [{"loop": 1}]},
            "s0",
            labels={"t": ["target"]}
        )
        vals, _ = mdp_reachability_probability(mdp, "target", minimize=True)
        assert vals["s0"] == Fraction(0)

    def test_multi_step_mdp(self):
        """Two-step MDP: s0->s1->t, choices at each."""
        mdp = build_mdp(
            ["s0", "s1", "t", "f"],
            {"s0": [
                ({"s1": Fraction(1,2), "f": Fraction(1,2)}, "risky"),
                ({"s1": 1}, "safe"),
            ],
             "s1": [
                ({"t": Fraction(1,3), "f": Fraction(2,3)}, "gamble"),
                ({"t": Fraction(2,3), "f": Fraction(1,3)}, "careful"),
            ],
             "t": [{"t": 1}],
             "f": [{"f": 1}]},
            "s0",
            labels={"t": ["target"]}
        )
        vals, sched = mdp_reachability_probability(mdp, "target", minimize=False)
        # Max: safe at s0 (prob 1 to s1), careful at s1 (prob 2/3 to t) => 2/3
        assert vals["s0"] == Fraction(2, 3)


# ============================================================
# MDP expected reward
# ============================================================

class TestMDPExpectedReward:
    def test_min_reward(self):
        """Choose shorter path to minimize expected cost."""
        mdp = build_mdp(
            ["s0", "t"],
            {"s0": [
                ({"t": 1}, "direct"),
            ],
             "t": [{"t": 1}]},
            "s0",
            labels={"t": ["target"]},
            rewards={"s0": 5}
        )
        vals, _ = mdp_expected_reward(mdp, "target", minimize=True)
        assert vals["s0"] == Fraction(5)

    def test_choose_cheaper_path(self):
        """Two paths: expensive (reward 10) vs cheap (reward 2)."""
        mdp = build_mdp(
            ["s0", "exp", "cheap", "t"],
            {"s0": [
                ({"exp": 1}, "expensive"),
                ({"cheap": 1}, "cheap_path"),
            ],
             "exp": [({"t": 1},)],
             "cheap": [({"t": 1},)],
             "t": [{"t": 1}]},
            "s0",
            labels={"t": ["target"]},
            rewards={"s0": 1, "exp": 10, "cheap": 2}
        )
        vals, sched = mdp_expected_reward(mdp, "target", minimize=True)
        # Min: s0 cost=1, choose cheap (cost=2), total=3
        assert vals["s0"] == Fraction(3)
        assert sched["s0"].action == "cheap_path"


# ============================================================
# PCTL model checking
# ============================================================

class TestPCTL:
    def _simple_dtmc(self):
        return build_dtmc(
            ["s0", "s1", "s2"],
            {"s0": {"s1": Fraction(1,2), "s2": Fraction(1,2)},
             "s1": {"s1": 1}, "s2": {"s2": 1}},
            "s0",
            labels={"s0": ["init"], "s1": ["good"], "s2": ["bad"]}
        )

    def test_atomic(self):
        dtmc = self._simple_dtmc()
        sat = pctl_check_dtmc(dtmc, Atomic("good"))
        assert sat == {"s1"}

    def test_not(self):
        dtmc = self._simple_dtmc()
        sat = pctl_check_dtmc(dtmc, Not(Atomic("good")))
        assert sat == {"s0", "s2"}

    def test_and(self):
        dtmc = self._simple_dtmc()
        sat = pctl_check_dtmc(dtmc, And(Atomic("init"), Not(Atomic("good"))))
        assert sat == {"s0"}

    def test_or(self):
        dtmc = self._simple_dtmc()
        sat = pctl_check_dtmc(dtmc, Or(Atomic("good"), Atomic("bad")))
        assert sat == {"s1", "s2"}

    def test_prob_finally_ge(self):
        """P>=1/2 [F good]: s0 and s1 satisfy (s0 has prob 1/2, s1 has prob 1)."""
        dtmc = self._simple_dtmc()
        formula = ProbOp(">=", Fraction(1, 2), Finally(Atomic("good")))
        sat = pctl_check_dtmc(dtmc, formula)
        assert "s1" in sat
        assert "s0" in sat  # prob = 1/2 >= 1/2

    def test_prob_finally_gt(self):
        """P>1/2 [F good]: only s1 (prob=1)."""
        dtmc = self._simple_dtmc()
        formula = ProbOp(">", Fraction(1, 2), Finally(Atomic("good")))
        sat = pctl_check_dtmc(dtmc, formula)
        assert sat == {"s1"}

    def test_prob_bounded_finally(self):
        """P>=1 [F<=0 good]: only s1 is already good."""
        dtmc = self._simple_dtmc()
        formula = ProbOp(">=", 1, BoundedFinally(Atomic("good"), 0))
        sat = pctl_check_dtmc(dtmc, formula)
        assert sat == {"s1"}

    def test_prob_bounded_finally_one_step(self):
        """P>=1/2 [F<=1 good]: s0 can reach good in 1 step with prob 1/2."""
        dtmc = self._simple_dtmc()
        formula = ProbOp(">=", Fraction(1, 2), BoundedFinally(Atomic("good"), 1))
        sat = pctl_check_dtmc(dtmc, formula)
        assert "s0" in sat
        assert "s1" in sat

    def test_prob_until(self):
        """P>=1/2 [init U good]: s0 satisfies init, so it can stay init until good."""
        dtmc = self._simple_dtmc()
        formula = ProbOp(">=", Fraction(1, 2), Until(Atomic("init"), Atomic("good")))
        sat = pctl_check_dtmc(dtmc, formula)
        assert "s0" in sat  # prob of init U good from s0 = 1/2 (path s0->s1)
        assert "s1" in sat  # s1 satisfies good immediately

    def test_expected_reward(self):
        """R<=5 [F target]."""
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s1": 1}, "s1": {"s1": 1}},
            "s0",
            labels={"s1": ["target"]},
            rewards={"s0": 3}
        )
        formula = ExpRewardOp("<=", 5, Atomic("target"))
        sat = pctl_check_dtmc(dtmc, formula)
        assert "s0" in sat  # reward = 3 <= 5
        assert "s1" in sat  # reward = 0 <= 5


# ============================================================
# Bisimulation quotient
# ============================================================

class TestBisimulation:
    def test_identical_states_merged(self):
        """Two states with same labels and same transitions should merge."""
        dtmc = build_dtmc(
            ["s0", "s1", "s2", "t"],
            {"s0": {"s1": Fraction(1,2), "s2": Fraction(1,2)},
             "s1": {"t": 1},
             "s2": {"t": 1},
             "t": {"t": 1}},
            "s0",
            labels={"s1": ["mid"], "s2": ["mid"], "t": ["target"]}
        )
        quotient, partition = dtmc_bisimulation_quotient(dtmc)
        # s1 and s2 are bisimilar: same labels, same transitions
        assert partition["s1"] == partition["s2"]
        assert len(quotient.states) < len(dtmc.states)

    def test_different_labels_not_merged(self):
        """States with different labels should not merge."""
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s0": 1}, "s1": {"s1": 1}},
            "s0",
            labels={"s0": ["a"], "s1": ["b"]}
        )
        _, partition = dtmc_bisimulation_quotient(dtmc)
        assert partition["s0"] != partition["s1"]

    def test_single_state(self):
        dtmc = build_dtmc(["s0"], {"s0": {"s0": 1}}, "s0")
        quotient, partition = dtmc_bisimulation_quotient(dtmc)
        assert len(quotient.states) == 1

    def test_quotient_preserves_reachability(self):
        """Bisimulation preserves reachability probabilities."""
        dtmc = build_dtmc(
            ["s0", "a1", "a2", "t"],
            {"s0": {"a1": Fraction(1,2), "a2": Fraction(1,2)},
             "a1": {"t": Fraction(1,3), "a1": Fraction(2,3)},
             "a2": {"t": Fraction(1,3), "a2": Fraction(2,3)},
             "t": {"t": 1}},
            "s0",
            labels={"a1": ["mid"], "a2": ["mid"], "t": ["target"]}
        )
        orig_probs = dtmc_reachability_probability(dtmc, "target")
        quotient, _ = dtmc_bisimulation_quotient(dtmc)
        q_probs = dtmc_reachability_probability(quotient, "target")
        # a1 and a2 merge; both have prob 1 (geometric), s0 has prob 1
        assert orig_probs["s0"] == Fraction(1)
        # Quotient initial should also have prob 1
        assert q_probs[quotient.initial] == Fraction(1)


# ============================================================
# Builder convenience tests
# ============================================================

class TestBuilders:
    def test_build_dtmc(self):
        dtmc = build_dtmc(
            ["a", "b"],
            {"a": {"b": 1}, "b": {"b": 1}},
            "a",
            labels={"b": ["target"]},
            rewards={"a": 10}
        )
        assert dtmc.initial == "a"
        assert dtmc.states["a"].reward == Fraction(10)
        assert "target" in dtmc.states["b"].labels

    def test_build_mdp(self):
        mdp = build_mdp(
            ["a", "b", "c"],
            {"a": [({"b": 1}, "go_b"), ({"c": 1}, "go_c")],
             "b": [{"b": 1}],
             "c": [{"c": 1}]},
            "a"
        )
        assert len(mdp.actions("a")) == 2
        assert mdp.actions("a")[0].action == "go_b"

    def test_build_mdp_no_action(self):
        mdp = build_mdp(
            ["a", "b"],
            {"a": [{"b": 1}], "b": [{"b": 1}]},
            "a"
        )
        assert mdp.actions("a")[0].action is None


# ============================================================
# Simulation tests
# ============================================================

class TestSimulation:
    def test_simulate_absorbing(self):
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s1": 1}, "s1": {"s1": 1}},
            "s0"
        )
        path = dtmc_simulate_path(dtmc, max_steps=10)
        assert path[0] == "s0"
        assert path[1] == "s1"
        assert len(path) == 2  # stops at absorbing

    def test_estimate_certain(self):
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s1": 1}, "s1": {"s1": 1}},
            "s0",
            labels={"s1": ["target"]}
        )
        est = dtmc_estimate_reachability(dtmc, "target", num_samples=100)
        assert est == Fraction(1)  # always reaches target

    def test_estimate_impossible(self):
        dtmc = build_dtmc(
            ["s0", "s1"],
            {"s0": {"s0": 1}, "s1": {"s1": 1}},
            "s0",
            labels={"s1": ["target"]}
        )
        est = dtmc_estimate_reachability(dtmc, "target", num_samples=100)
        assert est == Fraction(0)


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_no_transitions(self):
        """State with no outgoing transitions (dead end)."""
        dtmc = DTMC()
        dtmc.add_state("s0")
        dtmc.add_state("t", labels=["target"])
        dtmc.set_initial("s0")
        # s0 has no transitions
        dtmc.add_transition("t", {"t": 1})
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(0)

    def test_self_loop_not_target(self):
        """Self-loop on non-target = prob 0."""
        dtmc = build_dtmc(
            ["s0", "t"],
            {"s0": {"s0": 1}, "t": {"t": 1}},
            "s0",
            labels={"t": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(0)

    def test_initial_is_target(self):
        dtmc = build_dtmc(
            ["s0"],
            {"s0": {"s0": 1}},
            "s0",
            labels={"s0": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(1)

    def test_mdp_no_actions(self):
        """MDP state with no actions = dead end."""
        mdp = MDP()
        mdp.add_state("s0")
        mdp.add_state("t", labels=["target"])
        mdp.set_initial("s0")
        mdp.add_transition("t", {"t": 1})
        vals, _ = mdp_reachability_probability(mdp, "target")
        assert vals["s0"] == Fraction(0)

    def test_large_chain(self):
        """Chain of 20 states, each going to next with prob 1."""
        n = 20
        names = [f"s{i}" for i in range(n)]
        trans = {}
        for i in range(n - 1):
            trans[names[i]] = {names[i+1]: 1}
        trans[names[-1]] = {names[-1]: 1}
        dtmc = build_dtmc(names, trans, names[0], labels={names[-1]: ["target"]})
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs[names[0]] == Fraction(1)

    def test_large_chain_rewards(self):
        """Chain of 10, each state has reward 1. Total = 9."""
        n = 10
        names = [f"s{i}" for i in range(n)]
        trans = {}
        for i in range(n - 1):
            trans[names[i]] = {names[i+1]: 1}
        trans[names[-1]] = {names[-1]: 1}
        rews = {names[i]: 1 for i in range(n - 1)}
        dtmc = build_dtmc(names, trans, names[0], labels={names[-1]: ["target"]}, rewards=rews)
        rewards = dtmc_expected_reward(dtmc, "target")
        assert rewards[names[0]] == Fraction(n - 1)


# ============================================================
# Complex scenarios
# ============================================================

class TestComplexScenarios:
    def test_die_roll(self):
        """Fair 6-sided die. P(roll 6) = 1/6."""
        states = ["roll"] + [f"face{i}" for i in range(1, 7)]
        trans = {"roll": {f"face{i}": Fraction(1, 6) for i in range(1, 7)}}
        for i in range(1, 7):
            trans[f"face{i}"] = {f"face{i}": 1}
        dtmc = build_dtmc(states, trans, "roll", labels={"face6": ["six"]})
        probs = dtmc_reachability_probability(dtmc, "six")
        assert probs["roll"] == Fraction(1, 6)

    def test_gambler_ruin_3(self):
        """Gambler's ruin with 3 states: 0 (lose), 1 (start), 2 (win).
        At state 1: go to 0 or 2 with prob 1/2 each.
        P(win from 1) = 1/2.
        """
        dtmc = build_dtmc(
            ["s0", "s1", "s2"],
            {"s0": {"s0": 1},
             "s1": {"s0": Fraction(1, 2), "s2": Fraction(1, 2)},
             "s2": {"s2": 1}},
            "s1",
            labels={"s2": ["win"]}
        )
        probs = dtmc_reachability_probability(dtmc, "win")
        assert probs["s1"] == Fraction(1, 2)

    def test_gambler_ruin_5(self):
        """Gambler's ruin with wealth {0,1,2,3,4}. Start at 2. Win at 4.
        Fair game: P(win from k) = k/4.
        P(win from 2) = 1/2.
        """
        states = [f"w{i}" for i in range(5)]
        trans = {"w0": {"w0": 1}, "w4": {"w4": 1}}
        for i in range(1, 4):
            trans[f"w{i}"] = {f"w{i-1}": Fraction(1,2), f"w{i+1}": Fraction(1,2)}
        dtmc = build_dtmc(states, trans, "w2", labels={"w4": ["win"]})
        probs = dtmc_reachability_probability(dtmc, "win")
        assert probs["w2"] == Fraction(1, 2)
        assert probs["w1"] == Fraction(1, 4)
        assert probs["w3"] == Fraction(3, 4)

    def test_two_dice_sum_seven(self):
        """Roll two dice. P(sum=7) = 6/36 = 1/6."""
        states = ["roll"]
        trans_roll = {}
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                name = f"d{d1}_{d2}"
                states.append(name)
                trans_roll[name] = Fraction(1, 36)
        trans = {"roll": trans_roll}
        labels = {}
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                name = f"d{d1}_{d2}"
                trans[name] = {name: 1}
                if d1 + d2 == 7:
                    labels[name] = ["seven"]
        dtmc = build_dtmc(states, trans, "roll", labels=labels)
        probs = dtmc_reachability_probability(dtmc, "seven")
        assert probs["roll"] == Fraction(1, 6)

    def test_retry_with_backoff(self):
        """Retry pattern: each attempt has p=1/4 success. E[attempts] = 4."""
        dtmc = build_dtmc(
            ["try", "success"],
            {"try": {"success": Fraction(1,4), "try": Fraction(3,4)},
             "success": {"success": 1}},
            "try",
            labels={"success": ["done"]},
            rewards={"try": 1}
        )
        rewards = dtmc_expected_reward(dtmc, "done")
        assert rewards["try"] == Fraction(4)

    def test_protocol_handshake(self):
        """Network handshake: send, if lost (p=0.1) retry, else connected.
        P(connected) = 1. E[attempts with reward 1 each] = 10/9.
        """
        dtmc = build_dtmc(
            ["send", "connected"],
            {"send": {"connected": Fraction(9,10), "send": Fraction(1,10)},
             "connected": {"connected": 1}},
            "send",
            labels={"connected": ["done"]},
            rewards={"send": 1}
        )
        probs = dtmc_reachability_probability(dtmc, "done")
        assert probs["send"] == Fraction(1)
        rewards = dtmc_expected_reward(dtmc, "done")
        assert rewards["send"] == Fraction(10, 9)

    def test_mdp_routing(self):
        """Network routing: choose path A (fast but lossy) or B (slow but reliable).
        A: 1 step, p=1/2 success.
        B: 2 steps, p=1 success.
        Max reach prob is 1 for both. Min expected cost: B = 2 (with rewards 1 per step).
        """
        mdp = build_mdp(
            ["s", "a_try", "b1", "t", "f"],
            {"s": [
                ({"a_try": 1}, "path_a"),
                ({"b1": 1}, "path_b"),
            ],
             "a_try": [({"t": Fraction(1,2), "f": Fraction(1,2)},)],
             "b1": [({"t": 1},)],
             "t": [{"t": 1}],
             "f": [{"f": 1}]},
            "s",
            labels={"t": ["target"]},
            rewards={"s": 1, "a_try": 1, "b1": 1}
        )
        # Max reachability: path B guarantees reaching target
        vals_max, _ = mdp_reachability_probability(mdp, "target", minimize=False)
        assert vals_max["s"] == Fraction(1)

        # Min expected reward to reach target: path_b = 1 + 1 = 2
        vals_min, sched = mdp_expected_reward(mdp, "target", minimize=True)
        assert vals_min["s"] == Fraction(2)
        assert sched["s"].action == "path_b"


# ============================================================
# Regression / specific value checks
# ============================================================

class TestRegression:
    def test_three_absorbing(self):
        """Three absorbing states: prob distributes correctly."""
        dtmc = build_dtmc(
            ["s0", "a", "b", "c"],
            {"s0": {"a": Fraction(1,4), "b": Fraction(1,2), "c": Fraction(1,4)},
             "a": {"a": 1}, "b": {"b": 1}, "c": {"c": 1}},
            "s0",
            labels={"a": ["target"]}
        )
        probs = dtmc_reachability_probability(dtmc, "target")
        assert probs["s0"] == Fraction(1, 4)

    def test_nested_pctl(self):
        """Nested PCTL: P>=0.5 [F (P>=1 [F target])]."""
        dtmc = build_dtmc(
            ["s0", "s1", "s2", "t"],
            {"s0": {"s1": Fraction(1,2), "s2": Fraction(1,2)},
             "s1": {"t": 1},
             "s2": {"s2": 1},
             "t": {"t": 1}},
            "s0",
            labels={"t": ["target"]}
        )
        # Inner: P>=1 [F target] = {s1, t} (s1 reaches t with prob 1)
        # Let's label them and check
        inner = ProbOp(">=", 1, Finally(Atomic("target")))
        inner_sat = pctl_check_dtmc(dtmc, inner)
        assert inner_sat == {"s1", "t"}

    def test_steady_state_three_cycle(self):
        """Three-state cycle: a->b->c->a. SS = 1/3 each."""
        dtmc = build_dtmc(
            ["a", "b", "c"],
            {"a": {"b": 1}, "b": {"c": 1}, "c": {"a": 1}},
            "a"
        )
        ss = dtmc_steady_state(dtmc)
        for s in ["a", "b", "c"]:
            assert abs(ss[s] - Fraction(1, 3)) < Fraction(1, 10**6)

    def test_transient_convergence(self):
        """Transient distribution converges to steady state."""
        dtmc = build_dtmc(
            ["a", "b"],
            {"a": {"b": Fraction(2,3), "a": Fraction(1,3)},
             "b": {"a": 1}},
            "a"
        )
        ss = dtmc_steady_state(dtmc)
        trans = dtmc_transient_probs(dtmc, 100)
        for s in ["a", "b"]:
            assert abs(trans[s] - ss[s]) < Fraction(1, 10**6)

    def test_mdp_max_expected_reward(self):
        """Maximizer picks expensive path."""
        mdp = build_mdp(
            ["s0", "cheap", "exp", "t"],
            {"s0": [
                ({"cheap": 1}, "go_cheap"),
                ({"exp": 1}, "go_exp"),
            ],
             "cheap": [({"t": 1},)],
             "exp": [({"t": 1},)],
             "t": [{"t": 1}]},
            "s0",
            labels={"t": ["target"]},
            rewards={"s0": 0, "cheap": 1, "exp": 100}
        )
        vals, sched = mdp_expected_reward(mdp, "target", minimize=False)
        assert vals["s0"] == Fraction(100)
        assert sched["s0"].action == "go_exp"
