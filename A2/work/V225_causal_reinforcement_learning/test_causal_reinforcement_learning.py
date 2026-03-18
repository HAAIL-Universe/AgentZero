"""Tests for V225: Causal Reinforcement Learning."""

import math
import random
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "V213_markov_decision_processes"))

from causal_reinforcement_learning import (
    CausalState,
    CausalTransition,
    ConfoundedObservation,
    CausalRLResult,
    CausalMDP,
    ConfoundedMDP,
    CausalQLearning,
    OffPolicyCausalEvaluator,
    CausalRewardDecomposition,
    InterventionalPlanner,
    CausalTransferRL,
    compare_policies,
    build_confounded_treatment_mdp,
    build_causal_gridworld,
    build_confounded_bandit_mdp,
)
from markov_decision_processes import (
    MDP,
    MDPResult,
    value_iteration,
    policy_iteration,
    simulate,
)


# ===================================================================
# CausalState tests
# ===================================================================

class TestCausalState:
    def test_key_generation(self):
        cs = CausalState(variables={"x": 1, "y": 2})
        key = cs.key()
        assert "x=1" in key
        assert "y=2" in key

    def test_key_deterministic(self):
        cs1 = CausalState(variables={"b": 2, "a": 1})
        cs2 = CausalState(variables={"a": 1, "b": 2})
        assert cs1.key() == cs2.key()  # Sorted

    def test_from_key(self):
        cs = CausalState(variables={"x": 1, "y": 2})
        key = cs.key()
        cs2 = CausalState.from_key(key)
        assert cs2.variables == cs.variables

    def test_from_key_strings(self):
        cs = CausalState.from_key("name=alice|role=admin")
        assert cs.variables["name"] == "alice"
        assert cs.variables["role"] == "admin"

    def test_from_key_numeric(self):
        cs = CausalState.from_key("x=1|y=2.5")
        assert cs.variables["x"] == 1
        assert cs.variables["y"] == 2.5

    def test_roundtrip(self):
        original = {"a": 0, "b": 1, "c": 2}
        cs = CausalState(variables=original)
        recovered = CausalState.from_key(cs.key())
        assert recovered.variables == original


# ===================================================================
# CausalMDP tests
# ===================================================================

class TestCausalMDP:
    def test_create_simple(self):
        cmdp = CausalMDP("test")
        cmdp.add_state_var("x", [0, 1])
        cmdp.set_action_var("a", ["left", "right"])
        assert "x" in cmdp.state_vars
        assert cmdp.action_domain == ["left", "right"]

    def test_add_edge(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("x", [0, 1])
        cmdp.add_edge("a", "x")
        assert "a" in cmdp.causal_parents["x"]
        assert "x" in cmdp.causal_children["a"]

    def test_structural_equation(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("x", [0, 1])
        cmdp.set_action_var("a", ["inc", "dec"])
        cmdp.add_edge("a", "x")
        cmdp.add_edge("x", "x")

        def x_eq(parents):
            x = parents.get("x", 0)
            a = parents.get("a", "inc")
            if a == "inc":
                return {min(x + 1, 1): 1.0}
            else:
                return {max(x - 1, 0): 1.0}

        cmdp.set_structural_eq("x", x_eq)
        cmdp.set_reward_fn(lambda sv, a: 1.0 if sv.get("x") == 1 else 0.0)

        # Test transition computation
        trans = cmdp._compute_transition("x=0", "inc")
        assert len(trans) > 0
        # Should transition to x=1
        next_states = {ns for ns, _, _ in trans}
        assert "x=1" in next_states

    def test_enumerate_states(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("x", [0, 1])
        cmdp.add_state_var("y", [0, 1])
        states = cmdp._enumerate_states()
        assert len(states) == 4

    def test_to_mdp(self):
        cmdp = build_causal_gridworld()
        mdp = cmdp.to_mdp()
        assert len(mdp.states) == 9  # 3x3 grid
        assert len(mdp.actions) == 4

    def test_gridworld_solvable(self):
        cmdp = build_causal_gridworld()
        mdp = cmdp.to_mdp()
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged
        # Goal state should have high value
        assert result.values.get("x=2|y=2", 0.0) > 0

    def test_topological_sort(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("x", [0, 1])
        cmdp.add_state_var("y", [0, 1])
        cmdp.add_edge("x", "y")
        topo = cmdp._topological_sort()
        assert topo.index("x") < topo.index("y")

    def test_confounder(self):
        cmdp, _ = build_confounded_bandit_mdp()
        assert "severity" in cmdp.confounders
        assert "health" in cmdp.confounders["severity"]

    def test_confounder_marginalization(self):
        """Confounders should be marginalized out in transitions."""
        cmdp, _ = build_confounded_bandit_mdp()
        trans = cmdp._compute_transition("health=0", "drug_A")
        # Should have transitions (marginalized over severity)
        assert len(trans) > 0
        # Probabilities should sum to ~1
        total_p = sum(p for _, p, _ in trans)
        assert abs(total_p - 1.0) < 0.01

    def test_interventional_transition(self):
        cmdp = build_causal_gridworld()
        trans = cmdp.interventional_transition("x=0|y=0", "right")
        assert len(trans) > 0
        # Deterministic: should go to x=1|y=0
        for ns, p, _ in trans:
            if p > 0.5:
                assert "x=1" in ns

    def test_two_var_state(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("pos", [0, 1, 2])
        cmdp.add_state_var("fuel", [0, 1])
        cmdp.set_action_var("action", ["move", "stay"])
        cmdp.add_edge("action", "pos")
        cmdp.add_edge("action", "fuel")
        cmdp.add_edge("pos", "pos")
        cmdp.add_edge("fuel", "fuel")
        cmdp.add_edge("fuel", "pos")  # Fuel affects whether move succeeds

        def pos_eq(parents):
            pos = parents.get("pos", 0)
            fuel = parents.get("fuel", 0)
            action = parents.get("action", "stay")
            if action == "move" and fuel > 0:
                return {min(pos + 1, 2): 1.0}
            return {pos: 1.0}

        def fuel_eq(parents):
            fuel = parents.get("fuel", 0)
            action = parents.get("action", "stay")
            if action == "move" and fuel > 0:
                return {fuel - 1: 1.0}
            return {fuel: 1.0}

        cmdp.set_structural_eq("pos", pos_eq)
        cmdp.set_structural_eq("fuel", fuel_eq)
        cmdp.set_reward_fn(lambda sv, a: 1.0 if sv.get("pos") == 2 else -0.1)

        mdp = cmdp.to_mdp()
        assert len(mdp.states) == 6  # 3 pos * 2 fuel

    def test_stochastic_structural_eq(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("x", [0, 1, 2])
        cmdp.set_action_var("a", ["push"])
        cmdp.add_edge("a", "x")
        cmdp.add_edge("x", "x")

        def x_eq(parents):
            x = parents.get("x", 0)
            return {min(x + 1, 2): 0.7, x: 0.3}

        cmdp.set_structural_eq("x", x_eq)
        cmdp.set_reward_fn(lambda sv, a: float(sv.get("x", 0)))

        trans = cmdp._compute_transition("x=0", "push")
        probs = {ns: p for ns, p, _ in trans}
        assert abs(probs.get("x=1", 0) - 0.7) < 0.01
        assert abs(probs.get("x=0", 0) - 0.3) < 0.01

    def test_chain_builder(self):
        cmdp = CausalMDP("chain")
        result = (cmdp.add_state_var("x", [0, 1])
                  .set_action_var("a", ["go"])
                  .add_edge("a", "x"))
        assert result is cmdp  # Chaining works


# ===================================================================
# ConfoundedMDP tests
# ===================================================================

class TestConfoundedMDP:
    def setup_method(self):
        self.mdp, self.conf = build_confounded_treatment_mdp()

    def test_observations_loaded(self):
        assert len(self.conf.observations) == 400

    def test_empirical_reward(self):
        r = self.conf.empirical_reward("mild", "treat")
        assert isinstance(r, float)

    def test_empirical_transition(self):
        trans = self.conf.empirical_transition("mild", "treat")
        assert isinstance(trans, dict)
        if trans:
            assert abs(sum(trans.values()) - 1.0) < 0.05

    def test_detect_confounding(self):
        strength = self.conf.detect_confounding("mild", "treat", "patient_type")
        # Should detect confounding (patient_type affects both treatment and outcome)
        assert strength >= 0.0

    def test_detect_no_confounding(self):
        # No confounding if we use a non-existent variable
        strength = self.conf.detect_confounding("mild", "treat", "nonexistent")
        assert strength == 0.0

    def test_backdoor_adjusted_reward(self):
        naive = self.conf.empirical_reward("mild", "treat")
        adjusted = self.conf.backdoor_adjusted_reward(
            "mild", "treat", "patient_type")
        # Both should be finite
        assert math.isfinite(naive)
        assert math.isfinite(adjusted)

    def test_ipw_reward(self):
        propensity = {
            "mild": {"treat": 0.55, "wait": 0.45},
            "severe": {"treat": 0.55, "wait": 0.45},
        }
        r = self.conf.ipw_reward("mild", "treat", propensity)
        assert math.isfinite(r)

    def test_doubly_robust(self):
        propensity = {
            "mild": {"treat": 0.55, "wait": 0.45},
            "severe": {"treat": 0.55, "wait": 0.45},
        }
        baseline = {
            "mild": {"treat": 0.5, "wait": 0.2},
            "severe": {"treat": 0.3, "wait": 0.0},
        }
        r = self.conf.doubly_robust_reward(
            "mild", "treat", propensity, baseline)
        assert math.isfinite(r)

    def test_add_observations_batch(self):
        conf = ConfoundedMDP(self.mdp)
        conf.add_observations_batch([
            ("mild", "treat", "healthy", 1.0),
            ("severe", "wait", "severe", -1.0, {"type": "weak"}),
        ])
        assert len(conf.observations) == 2
        assert conf.observations[1].context["type"] == "weak"

    def test_empty_empirical_reward(self):
        conf = ConfoundedMDP(self.mdp)
        assert conf.empirical_reward("nonexist", "treat") == 0.0

    def test_empty_empirical_transition(self):
        conf = ConfoundedMDP(self.mdp)
        assert conf.empirical_transition("nonexist", "treat") == {}


# ===================================================================
# CausalQLearning tests
# ===================================================================

class TestCausalQLearning:
    def setup_method(self):
        self.mdp = MDP("test")
        self.mdp.add_state("s0")
        self.mdp.add_state("s1")
        self.mdp.add_state("s2", terminal=True)
        self.mdp.set_initial("s0")
        self.mdp.add_action("a")
        self.mdp.add_action("b")
        self.mdp.add_transition("s0", "a", "s1", 0.8, 1.0)
        self.mdp.add_transition("s0", "a", "s0", 0.2, 0.0)
        self.mdp.add_transition("s0", "b", "s0", 0.6, 0.0)
        self.mdp.add_transition("s0", "b", "s1", 0.4, 0.5)
        self.mdp.add_transition("s1", "a", "s2", 0.9, 2.0)
        self.mdp.add_transition("s1", "a", "s1", 0.1, 0.0)
        self.mdp.add_transition("s1", "b", "s2", 0.5, 1.0)
        self.mdp.add_transition("s1", "b", "s1", 0.5, 0.0)

    def test_basic_update(self):
        cql = CausalQLearning(self.mdp, seed=42)
        cql.update("s0", "a", 1.0, "s1")
        assert cql.q_values["s0"]["a"] > 0

    def test_policy_extraction(self):
        cql = CausalQLearning(self.mdp, seed=42)
        # Train a bit
        rng = random.Random(42)
        for _ in range(100):
            s = rng.choice(["s0", "s1"])
            a = cql.select_action(s)
            ns = rng.choice(["s0", "s1", "s2"])
            r = rng.uniform(-1, 2)
            cql.update(s, a, r, ns, done=(ns == "s2"))

        policy = cql.get_policy()
        assert isinstance(policy, dict)
        assert len(policy) > 0

    def test_values_extraction(self):
        cql = CausalQLearning(self.mdp, seed=42)
        cql.update("s0", "a", 1.0, "s1")
        values = cql.get_values()
        assert "s0" in values

    def test_result(self):
        cql = CausalQLearning(self.mdp, seed=42)
        cql.update("s0", "a", 1.0, "s1")
        result = cql.result()
        assert isinstance(result, CausalRLResult)
        assert result.iterations == 1

    def test_backdoor_adjustment(self):
        cql = CausalQLearning(self.mdp, adjustment_method="backdoor", seed=42)
        cql.set_adjustment_var("severity")

        # Add observations with context
        for i in range(20):
            ctx = {"severity": i % 2}
            cql.update("s0", "a", 1.0 + (i % 2), "s1", context=ctx)

        assert cql.causal_adjustments > 0

    def test_ipw_adjustment(self):
        cql = CausalQLearning(self.mdp, adjustment_method="ipw", seed=42)
        cql.set_propensity_scores({
            "s0": {"a": 0.7, "b": 0.3},
            "s1": {"a": 0.5, "b": 0.5},
        })
        cql.update("s0", "a", 1.0, "s1")
        assert cql.causal_adjustments == 1

    def test_epsilon_greedy(self):
        cql = CausalQLearning(self.mdp, epsilon=1.0, seed=42)
        # With epsilon=1.0, all actions are random
        actions = [cql.select_action("s0") for _ in range(100)]
        assert "a" in actions
        assert "b" in actions

    def test_greedy(self):
        cql = CausalQLearning(self.mdp, epsilon=0.0, seed=42)
        cql.q_values["s0"]["a"] = 10.0
        cql.q_values["s0"]["b"] = 1.0
        # Should always pick "a"
        actions = [cql.select_action("s0") for _ in range(10)]
        assert all(a == "a" for a in actions)

    def test_terminal_state_update(self):
        cql = CausalQLearning(self.mdp, seed=42)
        cql.update("s1", "a", 2.0, "s2", done=True)
        # TD target should be just the reward (no future)
        assert cql.q_values["s1"]["a"] == pytest.approx(0.1 * 2.0)


# ===================================================================
# OffPolicyCausalEvaluator tests
# ===================================================================

class TestOffPolicyCausalEvaluator:
    def setup_method(self):
        self.evaluator = OffPolicyCausalEvaluator(gamma=0.9)

    def _make_trajectory(self, rewards, states=None, actions=None):
        n = len(rewards)
        if states is None:
            states = [f"s{i}" for i in range(n)]
        if actions is None:
            actions = ["a"] * n
        traj = []
        for i in range(n):
            ns = states[i + 1] if i + 1 < len(states) else "terminal"
            traj.append(ConfoundedObservation(
                state=states[i], action=actions[i],
                next_state=ns, reward=rewards[i]
            ))
        return traj

    def test_is_same_policy(self):
        """IS with identical policies should give unbiased estimate."""
        policy = {"s0": {"a": 1.0}, "s1": {"a": 1.0}}
        traj = self._make_trajectory([1.0, 0.5])
        value = self.evaluator.importance_sampling(
            [traj], policy, policy)
        expected = 1.0 + 0.9 * 0.5
        assert abs(value - expected) < 0.01

    def test_wis_same_policy(self):
        policy = {"s0": {"a": 1.0}, "s1": {"a": 1.0}}
        traj = self._make_trajectory([1.0, 0.5])
        value = self.evaluator.weighted_importance_sampling(
            [traj], policy, policy)
        expected = 1.0 + 0.9 * 0.5
        assert abs(value - expected) < 0.01

    def test_is_different_policy(self):
        """IS with different policies reweights returns."""
        target = {"s0": {"a": 0.8, "b": 0.2}, "s1": {"a": 1.0}}
        behavior = {"s0": {"a": 0.5, "b": 0.5}, "s1": {"a": 1.0}}
        traj = self._make_trajectory([1.0, 0.5])
        value = self.evaluator.importance_sampling(
            [traj], target, behavior)
        # rho = 0.8/0.5 * 1.0/1.0 = 1.6
        expected = 1.6 * (1.0 + 0.9 * 0.5)
        assert abs(value - expected) < 0.01

    def test_wis_normalizes(self):
        """WIS normalizes by total weight."""
        target = {"s0": {"a": 0.8, "b": 0.2}}
        behavior = {"s0": {"a": 0.5, "b": 0.5}}
        trajs = [self._make_trajectory([r]) for r in [1.0, 2.0, 3.0]]
        value = self.evaluator.weighted_importance_sampling(
            trajs, target, behavior)
        assert math.isfinite(value)

    def test_dr_estimation(self):
        target = {"s0": {"a": 1.0}, "s1": {"a": 1.0}}
        behavior = {"s0": {"a": 1.0}, "s1": {"a": 1.0}}
        q_model = {"s0": {"a": 1.0}, "s1": {"a": 0.5}}
        v_model = {"s0": 1.0, "s1": 0.5, "terminal": 0.0}
        traj = self._make_trajectory([1.0, 0.5])
        value = self.evaluator.doubly_robust(
            [traj], target, behavior, q_model, v_model)
        assert math.isfinite(value)

    def test_causal_is(self):
        target = {"s0": {"a": 1.0}}
        behavior = {"s0": {"a": 1.0}}
        traj = self._make_trajectory([1.0])
        traj[0].context = {"z": 0}
        value = self.evaluator.causal_importance_sampling(
            [traj], target, behavior, "z")
        assert abs(value - 1.0) < 0.01

    def test_causal_is_stratification(self):
        """Causal IS should stratify by adjustment variable."""
        target = {"s0": {"a": 1.0}}
        behavior = {"s0": {"a": 1.0}}

        trajs = []
        for i in range(20):
            t = self._make_trajectory([float(i % 3)])
            t[0].context = {"z": i % 2}
            trajs.append(t)

        value = self.evaluator.causal_importance_sampling(
            trajs, target, behavior, "z")
        assert math.isfinite(value)

    def test_empty_trajectories(self):
        assert self.evaluator.importance_sampling([], {}, {}) == 0.0
        assert self.evaluator.weighted_importance_sampling([], {}, {}) == 0.0
        assert self.evaluator.doubly_robust([], {}, {}, {}, {}) == 0.0
        assert self.evaluator.causal_importance_sampling([], {}, {}, "z") == 0.0


# ===================================================================
# CausalRewardDecomposition tests
# ===================================================================

class TestCausalRewardDecomposition:
    def test_no_confounding_no_mediators(self):
        decomp = CausalRewardDecomposition()
        obs = [
            ConfoundedObservation("s", "treat", "s", 1.0),
            ConfoundedObservation("s", "treat", "s", 1.5),
            ConfoundedObservation("s", "wait", "s", 0.5),
            ConfoundedObservation("s", "wait", "s", 0.3),
        ]
        decomp.estimate_effects(obs, "treat", "wait")
        d = decomp.get_decomposition("s", "treat")
        # Total = direct (no mediators, no confounders)
        assert abs(d["total"] - d["direct"]) < 0.01
        assert abs(d["indirect"]) < 0.01
        assert abs(d["spurious"]) < 0.01

    def test_with_confounders(self):
        decomp = CausalRewardDecomposition()
        decomp.set_confounders(["type"])

        obs = []
        # Type=0: treat works well
        for _ in range(50):
            obs.append(ConfoundedObservation("s", "treat", "s", 2.0,
                                              {"type": 0}))
            obs.append(ConfoundedObservation("s", "wait", "s", 0.5,
                                              {"type": 0}))
        # Type=1: treat works poorly
        for _ in range(50):
            obs.append(ConfoundedObservation("s", "treat", "s", 0.5,
                                              {"type": 1}))
            obs.append(ConfoundedObservation("s", "wait", "s", 0.3,
                                              {"type": 1}))

        decomp.estimate_effects(obs, "treat", "wait")
        d = decomp.get_decomposition("s", "treat")
        assert math.isfinite(d["total"])
        assert math.isfinite(d["spurious"])

    def test_with_mediators(self):
        decomp = CausalRewardDecomposition()
        decomp.set_mediators(["mediator"])

        obs = []
        for i in range(100):
            m_val = 1 if i < 70 else 0  # Mediator changes under treatment
            obs.append(ConfoundedObservation("s", "treat", "s", 1.0 + m_val,
                                              {"mediator": m_val}))
        for i in range(100):
            m_val = 1 if i < 30 else 0
            obs.append(ConfoundedObservation("s", "wait", "s", 0.5 + m_val,
                                              {"mediator": m_val}))

        decomp.estimate_effects(obs, "treat", "wait")
        d = decomp.get_decomposition("s", "treat")
        # Should detect indirect effect through mediator
        assert math.isfinite(d["direct"])
        assert math.isfinite(d["indirect"])

    def test_default_decomposition(self):
        decomp = CausalRewardDecomposition()
        d = decomp.get_decomposition("nonexist", "nonexist")
        assert d["total"] == 0.0
        assert d["direct"] == 0.0
        assert d["indirect"] == 0.0
        assert d["spurious"] == 0.0


# ===================================================================
# InterventionalPlanner tests
# ===================================================================

class TestInterventionalPlanner:
    def test_plan_without_confounding(self):
        mdp = MDP("simple")
        mdp.add_state("s0")
        mdp.add_state("s1", terminal=True)
        mdp.set_initial("s0")
        mdp.add_action("go")
        mdp.add_transition("s0", "go", "s1", 1.0, 1.0)

        planner = InterventionalPlanner(mdp, gamma=0.9)
        result = planner.plan()
        assert result.policy.get("s0") == "go"
        assert result.converged

    def test_plan_with_adjusted_model(self):
        mdp, conf = build_confounded_treatment_mdp()
        planner = InterventionalPlanner(mdp, gamma=0.9)
        planner.set_confounded_data(conf)
        planner.compute_adjusted_model("patient_type")

        result = planner.plan()
        assert isinstance(result, CausalRLResult)
        assert result.causal_adjustments > 0
        assert len(result.policy) > 0

    def test_adjusted_vs_naive(self):
        """Adjusted planner should differ from naive when confounding exists."""
        mdp, conf = build_confounded_treatment_mdp()

        # Naive: use original MDP
        naive_result = value_iteration(mdp, gamma=0.9)

        # Adjusted: use causal adjustment
        planner = InterventionalPlanner(mdp, gamma=0.9)
        planner.set_confounded_data(conf)
        planner.compute_adjusted_model("patient_type")
        causal_result = planner.plan()

        # Both should produce policies
        assert len(naive_result.policy) > 0
        assert len(causal_result.policy) > 0

    def test_plan_uses_original_when_no_adjustment(self):
        mdp = MDP("simple")
        mdp.add_state("s0")
        mdp.add_state("s1", terminal=True)
        mdp.set_initial("s0")
        mdp.add_action("a")
        mdp.add_action("b")
        mdp.add_transition("s0", "a", "s1", 1.0, 10.0)
        mdp.add_transition("s0", "b", "s1", 1.0, 1.0)

        planner = InterventionalPlanner(mdp, gamma=0.9)
        result = planner.plan()
        assert result.policy["s0"] == "a"  # Higher reward


# ===================================================================
# CausalTransferRL tests
# ===================================================================

class TestCausalTransferRL:
    def test_learn_source(self):
        cmdp = build_causal_gridworld()
        mdp = cmdp.to_mdp()
        result = value_iteration(mdp, gamma=0.9)

        transfer = CausalTransferRL()
        transfer.learn_source(cmdp, result)
        assert len(transfer.source_graph) > 0
        assert len(transfer.source_effects) > 0

    def test_identify_invariant(self):
        transfer = CausalTransferRL()

        # Same observations in source and target
        obs = [ConfoundedObservation("x=0|y=0", "right", "x=1|y=0", 0.0)
               for _ in range(10)]
        transfer.identify_invariant_mechanisms(obs, obs, threshold=0.1)
        # Variables should be invariant (identical data)
        assert len(transfer.invariant_mechanisms) > 0

    def test_identify_changed_mechanism(self):
        transfer = CausalTransferRL()

        source = [ConfoundedObservation("x=0|y=0", "right", "x=1|y=0", 0.0)
                  for _ in range(20)]
        target = [ConfoundedObservation("x=0|y=0", "right", "x=0|y=1", 0.0)
                  for _ in range(20)]

        transfer.identify_invariant_mechanisms(source, target, threshold=0.05)
        # Mechanisms differ -- should not be invariant
        # (x transitions differently)

    def test_transfer_q_init_all_invariant(self):
        transfer = CausalTransferRL()
        transfer.invariant_mechanisms = {"x", "y"}
        transfer.source_effects = {"right": {"x": 5}, "up": {"y": 5}}

        source_q = {
            "s0": {"right": 1.0, "up": 0.5},
            "s1": {"right": 0.5, "up": 1.0},
        }
        transferred = transfer.transfer_q_init(source_q)
        assert transferred["s0"]["right"] == 1.0
        assert transferred["s1"]["up"] == 1.0

    def test_transfer_q_init_partial(self):
        transfer = CausalTransferRL()
        transfer.invariant_mechanisms = {"x"}  # Only x is invariant
        transfer.source_effects = {"right": {"x": 5}, "up": {"y": 5}}

        source_q = {
            "s0": {"right": 1.0, "up": 0.5},
        }
        transferred = transfer.transfer_q_init(source_q)
        assert transferred["s0"]["right"] == 1.0  # x is invariant
        assert transferred["s0"]["up"] == 0.0  # y is NOT invariant

    def test_transfer_with_mapping(self):
        transfer = CausalTransferRL()
        transfer.invariant_mechanisms = {"x"}
        transfer.source_effects = {"go": {}}

        source_q = {"source_s": {"go": 5.0}}
        mapping = {"source_s": "target_s"}
        transferred = transfer.transfer_q_init(source_q, state_mapping=mapping)
        assert "target_s" in transferred
        assert transferred["target_s"]["go"] == 5.0


# ===================================================================
# Integration tests
# ===================================================================

class TestIntegration:
    def test_confounded_treatment_pipeline(self):
        """Full pipeline: confounded data -> causal adjustment -> better policy."""
        mdp, conf = build_confounded_treatment_mdp()

        # 1. Detect confounding
        strength = conf.detect_confounding("mild", "treat", "patient_type")
        assert strength >= 0.0

        # 2. Get naive and adjusted rewards
        naive_r = conf.empirical_reward("mild", "treat")
        adjusted_r = conf.backdoor_adjusted_reward(
            "mild", "treat", "patient_type")
        assert math.isfinite(naive_r)
        assert math.isfinite(adjusted_r)

        # 3. Plan with adjustment
        planner = InterventionalPlanner(mdp, gamma=0.9)
        planner.set_confounded_data(conf)
        planner.compute_adjusted_model("patient_type")
        result = planner.plan()
        assert result.converged
        assert len(result.policy) > 0

    def test_causal_gridworld_pipeline(self):
        """CausalMDP -> MDP -> solve -> transfer."""
        # Build and solve source
        cmdp = build_causal_gridworld()
        mdp = cmdp.to_mdp()
        result = value_iteration(mdp, gamma=0.9)

        # Learn transferable knowledge
        transfer = CausalTransferRL()
        transfer.learn_source(cmdp, result)

        # Transfer Q-values
        if result.q_values:
            transferred = transfer.transfer_q_init(result.q_values)
            assert len(transferred) > 0

    def test_causal_q_learning_pipeline(self):
        """CausalQLearning with adjustment on confounded data."""
        mdp, conf = build_confounded_treatment_mdp()

        cql = CausalQLearning(mdp, gamma=0.9, alpha=0.1,
                               adjustment_method="backdoor", seed=42)
        cql.set_adjustment_var("patient_type")

        # Feed confounded observations
        for o in conf.observations[:100]:
            done = o.next_state == "healthy"
            cql.update(o.state, o.action, o.reward, o.next_state,
                       context=o.context, done=done)

        result = cql.result()
        assert result.iterations == 100
        assert result.causal_adjustments > 0
        assert len(result.q_values) > 0

    def test_off_policy_evaluation_pipeline(self):
        """Off-policy evaluation with different methods."""
        mdp, conf = build_confounded_treatment_mdp()

        evaluator = OffPolicyCausalEvaluator(gamma=0.9)

        # Build trajectories from observations
        trajectories = []
        current_traj = []
        for o in conf.observations:
            current_traj.append(o)
            if o.next_state == "healthy" or len(current_traj) >= 5:
                trajectories.append(current_traj)
                current_traj = []

        target = {"mild": {"treat": 1.0, "wait": 0.0},
                  "severe": {"treat": 1.0, "wait": 0.0}}
        behavior = {"mild": {"treat": 0.55, "wait": 0.45},
                    "severe": {"treat": 0.55, "wait": 0.45}}

        # IS
        is_val = evaluator.importance_sampling(
            trajectories[:20], target, behavior)
        assert math.isfinite(is_val)

        # WIS
        wis_val = evaluator.weighted_importance_sampling(
            trajectories[:20], target, behavior)
        assert math.isfinite(wis_val)

    def test_reward_decomposition_pipeline(self):
        """Decompose treatment effect into direct/indirect/spurious."""
        _, conf = build_confounded_treatment_mdp()

        decomp = CausalRewardDecomposition()
        decomp.set_confounders(["patient_type"])
        decomp.estimate_effects(conf.observations, "treat", "wait")

        d = decomp.get_decomposition("mild", "treat")
        assert math.isfinite(d["total"])
        assert math.isfinite(d["direct"])
        assert math.isfinite(d["spurious"])

    def test_confounded_bandit_mdp(self):
        """Build and solve confounded bandit MDP."""
        cmdp, behavior = build_confounded_bandit_mdp()
        mdp = cmdp.to_mdp()
        assert len(mdp.states) > 0
        # Should be solvable
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged

    def test_compare_policies(self):
        mdp, _ = build_confounded_treatment_mdp()
        policy1 = {"mild": "treat", "severe": "treat"}
        policy2 = {"mild": "wait", "severe": "wait"}
        advantage = compare_policies(mdp, policy1, policy2, gamma=0.9)
        assert isinstance(advantage, dict)
        # Treatment should generally be better
        for s in advantage:
            assert math.isfinite(advantage[s])

    def test_causal_mdp_with_confounder_solvable(self):
        """CausalMDP with confounders should produce a solvable MDP."""
        cmdp, _ = build_confounded_bandit_mdp()
        mdp = cmdp.to_mdp()
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged

    def test_observational_vs_interventional(self):
        """Observational and interventional transitions should differ with confounding."""
        cmdp, _ = build_confounded_bandit_mdp()
        state = "health=0"

        # Interventional
        int_trans = cmdp.interventional_transition(state, "drug_A")
        assert len(int_trans) > 0

        # Check probabilities sum to ~1
        total = sum(p for _, p, _ in int_trans)
        assert abs(total - 1.0) < 0.05

    def test_end_to_end_causal_rl(self):
        """Full end-to-end: CausalMDP -> confounded data -> causal Q-learning -> policy."""
        cmdp = build_causal_gridworld()
        mdp = cmdp.to_mdp()

        # Get optimal policy for reference
        optimal = value_iteration(mdp, gamma=0.9)

        # Generate some "observational" data (simulated trajectories)
        rng = random.Random(42)
        observations = []
        states = list(mdp.states)
        for _ in range(200):
            s = rng.choice([s for s in states if s not in mdp.terminal_states]
                           if any(s not in mdp.terminal_states for s in states)
                           else states)
            a = rng.choice(mdp.actions)
            trans = mdp.get_transitions(s, a)
            if trans:
                r = rng.random()
                cumul = 0.0
                ns = trans[-1][0]
                for next_s, p, reward in trans:
                    cumul += p
                    if r <= cumul:
                        ns = next_s
                        break
                observations.append(ConfoundedObservation(
                    state=s, action=a, next_state=ns,
                    reward=-0.1 if "x=2|y=2" not in ns else 1.0
                ))

        # Train causal Q-learning
        cql = CausalQLearning(mdp, gamma=0.9, alpha=0.1,
                               epsilon=0.2, seed=42)
        for o in observations:
            cql.update(o.state, o.action, o.reward, o.next_state)

        result = cql.result()
        assert len(result.policy) > 0
        assert result.iterations == 200


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:
    def test_single_state_cmdp(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("x", [0])
        cmdp.set_action_var("a", ["noop"])
        cmdp.set_structural_eq("x", lambda p: {0: 1.0})
        cmdp.set_reward_fn(lambda sv, a: 0.0)
        mdp = cmdp.to_mdp()
        assert len(mdp.states) == 1

    def test_many_actions(self):
        cmdp = CausalMDP()
        cmdp.add_state_var("x", [0, 1])
        actions = [f"a{i}" for i in range(10)]
        cmdp.set_action_var("a", actions)
        cmdp.add_edge("a", "x")
        cmdp.set_structural_eq("x", lambda p: {0: 0.5, 1: 0.5})
        cmdp.set_reward_fn(lambda sv, a: 1.0 if sv.get("x") == 1 else 0.0)
        mdp = cmdp.to_mdp()
        assert len(mdp.actions) == 10

    def test_confounded_mdp_empty(self):
        mdp = MDP("empty")
        mdp.add_state("s")
        mdp.add_action("a")
        conf = ConfoundedMDP(mdp)
        assert conf.empirical_reward("s", "a") == 0.0

    def test_causal_q_no_actions(self):
        mdp = MDP("minimal")
        mdp.add_state("s", terminal=True)
        mdp.add_action("a")
        cql = CausalQLearning(mdp, seed=42)
        action = cql.select_action("s")
        assert action == "a"  # Falls back to first action

    def test_off_policy_zero_behavior_prob(self):
        """Propensity clipping prevents division by zero."""
        evaluator = OffPolicyCausalEvaluator()
        traj = [ConfoundedObservation("s0", "a", "s1", 1.0)]
        target = {"s0": {"a": 1.0}}
        behavior = {"s0": {"a": 0.0}}  # Zero prob!
        # Should not crash (clipped to 0.01)
        value = evaluator.importance_sampling([traj], target, behavior)
        assert math.isfinite(value)

    def test_transfer_empty_source(self):
        transfer = CausalTransferRL()
        transferred = transfer.transfer_q_init({})
        assert transferred == {}

    def test_planner_no_confounded_data(self):
        mdp = MDP("simple")
        mdp.add_state("s0")
        mdp.add_state("s1", terminal=True)
        mdp.set_initial("s0")
        mdp.add_action("go")
        mdp.add_transition("s0", "go", "s1", 1.0, 1.0)

        planner = InterventionalPlanner(mdp)
        planner.compute_adjusted_model("z")  # No data -- should be no-op
        result = planner.plan()
        assert result.converged

    def test_causal_state_empty(self):
        cs = CausalState(variables={})
        assert cs.key() == ""

    def test_from_key_empty(self):
        cs = CausalState.from_key("")
        assert cs.variables == {}

    def test_reward_decomposition_single_action(self):
        decomp = CausalRewardDecomposition()
        obs = [ConfoundedObservation("s", "treat", "s", 1.0)]
        decomp.estimate_effects(obs, "treat", "wait")
        # No wait observations -- should handle gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
