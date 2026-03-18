"""Tests for V213: Markov Decision Processes."""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from markov_decision_processes import (
    MDP, MDPResult, Transition,
    value_iteration, policy_iteration, linear_programming,
    q_learning, rtdp,
    simulate, expected_total_reward, policy_advantage, occupancy_measure,
    mdp_to_influence_diagram, influence_diagram_to_mdp, mdp_transition_bn,
    compare_solvers, mdp_summary,
    gridworld, inventory_management, gambling, two_state_mdp,
)


# ─────────────────────────────────────────────────────────────────────────────
#  MDP Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestMDPConstruction:
    def test_add_state(self):
        mdp = MDP()
        mdp.add_state("s0")
        mdp.add_state("s1")
        assert len(mdp.states) == 2
        assert "s0" in mdp._state_set

    def test_add_state_idempotent(self):
        mdp = MDP()
        mdp.add_state("s0")
        mdp.add_state("s0")
        assert len(mdp.states) == 1

    def test_add_action(self):
        mdp = MDP()
        mdp.add_action("a")
        mdp.add_action("b")
        assert len(mdp.actions) == 2

    def test_add_action_idempotent(self):
        mdp = MDP()
        mdp.add_action("a")
        mdp.add_action("a")
        assert len(mdp.actions) == 1

    def test_terminal_state(self):
        mdp = MDP()
        mdp.add_state("s0", terminal=True)
        assert "s0" in mdp.terminal_states

    def test_set_initial(self):
        mdp = MDP()
        mdp.add_state("s0")
        mdp.set_initial("s0")
        assert mdp.initial_state == "s0"

    def test_add_transition_auto_registers(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0, 5.0)
        assert "s0" in mdp._state_set
        assert "s1" in mdp._state_set
        assert "a" in mdp._action_set

    def test_get_transitions(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 0.7, 1.0)
        mdp.add_transition("s0", "a", "s0", 0.3, 0.0)
        trans = mdp.get_transitions("s0", "a")
        assert len(trans) == 2
        probs = [p for _, p, _ in trans]
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_set_reward(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0, 2.0)
        mdp.set_reward("s0", "a", 3.0)
        trans = mdp.get_transitions("s0", "a")
        assert trans[0][2] == 5.0  # 2.0 + 3.0

    def test_get_actions_terminal(self):
        mdp = MDP()
        mdp.add_state("goal", terminal=True)
        mdp.add_transition("s0", "go", "goal", 1.0)
        assert mdp.get_actions("goal") == []

    def test_get_actions_from_transitions(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0)
        mdp.add_transition("s0", "b", "s1", 1.0)
        actions = mdp.get_actions("s0")
        assert set(actions) == {"a", "b"}

    def test_available_actions_override(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0)
        mdp.add_transition("s0", "b", "s1", 1.0)
        mdp.set_available_actions("s0", ["a"])
        assert mdp.get_actions("s0") == ["a"]

    def test_successor_states(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 0.5)
        mdp.add_transition("s0", "a", "s2", 0.5)
        assert mdp.successor_states("s0") == {"s1", "s2"}

    def test_reachable_states(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0)
        mdp.add_transition("s1", "a", "s2", 1.0)
        mdp.add_state("s3")  # unreachable
        mdp.set_initial("s0")
        reachable = mdp.reachable_states()
        assert "s0" in reachable
        assert "s1" in reachable
        assert "s2" in reachable
        assert "s3" not in reachable


class TestMDPValidation:
    def test_valid_mdp(self):
        mdp = two_state_mdp()
        assert mdp.validate() == []

    def test_no_states(self):
        mdp = MDP()
        issues = mdp.validate()
        assert any("No states" in i for i in issues)

    def test_invalid_probability_sum(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 0.3)
        mdp.add_transition("s0", "a", "s0", 0.3)
        issues = mdp.validate()
        assert any("sums to" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
#  Value Iteration
# ─────────────────────────────────────────────────────────────────────────────

class TestValueIteration:
    def test_two_state(self):
        mdp = two_state_mdp()
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged
        assert result.iterations < 1000
        assert "s0" in result.values
        assert "s1" in result.values
        assert "s0" in result.policy
        assert "s1" in result.policy

    def test_deterministic(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 10.0)
        mdp.add_state("s1", terminal=True)
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged
        assert abs(result.values["s0"] - 10.0) < 0.01
        assert result.values["s1"] == 0.0
        assert result.policy["s0"] == "go"

    def test_optimal_action_selection(self):
        mdp = MDP()
        mdp.add_transition("s0", "good", "s1", 1.0, 10.0)
        mdp.add_transition("s0", "bad", "s1", 1.0, 1.0)
        mdp.add_state("s1", terminal=True)
        result = value_iteration(mdp, gamma=0.9)
        assert result.policy["s0"] == "good"

    def test_q_values_returned(self):
        mdp = two_state_mdp()
        result = value_iteration(mdp, gamma=0.9)
        assert result.q_values is not None
        assert "s0" in result.q_values
        assert "a" in result.q_values["s0"]

    def test_gridworld_basic(self):
        mdp = gridworld(rows=3, cols=3, goal=(2, 2), trap=None)
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged
        # Goal state has value 0 (terminal)
        assert result.values["(2,2)"] == 0.0
        # States near goal should have higher value than far states
        assert result.values["(2,1)"] > result.values["(0,0)"]


class TestPolicyIteration:
    def test_two_state(self):
        mdp = two_state_mdp()
        result = policy_iteration(mdp, gamma=0.9)
        assert result.converged

    def test_agrees_with_vi(self):
        mdp = two_state_mdp()
        vi = value_iteration(mdp, gamma=0.9)
        pi = policy_iteration(mdp, gamma=0.9)
        for s in mdp.states:
            assert abs(vi.values[s] - pi.values[s]) < 0.01
        assert vi.policy == pi.policy

    def test_deterministic(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0, 5.0)
        mdp.add_transition("s0", "b", "s1", 1.0, 3.0)
        mdp.add_state("s1", terminal=True)
        result = policy_iteration(mdp, gamma=0.9)
        assert result.policy["s0"] == "a"


class TestLinearProgramming:
    def test_two_state(self):
        mdp = two_state_mdp()
        result = linear_programming(mdp, gamma=0.9)
        assert result.converged

    def test_agrees_with_vi(self):
        mdp = two_state_mdp()
        vi = value_iteration(mdp, gamma=0.9)
        lp = linear_programming(mdp, gamma=0.9)
        for s in mdp.states:
            assert abs(vi.values[s] - lp.values[s]) < 0.01


class TestQLearning:
    def test_simple_mdp(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 10.0)
        mdp.add_state("s1", terminal=True)
        mdp.set_initial("s0")
        result = q_learning(mdp, gamma=0.9, episodes=5000, seed=42)
        assert result.policy["s0"] == "go"

    def test_learns_optimal_direction(self):
        mdp = MDP()
        mdp.add_transition("s0", "good", "goal", 1.0, 10.0)
        mdp.add_transition("s0", "bad", "goal", 1.0, 1.0)
        mdp.add_state("goal", terminal=True)
        mdp.set_initial("s0")
        result = q_learning(mdp, gamma=0.9, episodes=5000, seed=42)
        assert result.policy["s0"] == "good"

    def test_q_values_populated(self):
        mdp = two_state_mdp()
        mdp.set_initial("s0")
        result = q_learning(mdp, gamma=0.9, episodes=2000, seed=42)
        assert result.q_values is not None
        assert len(result.q_values["s0"]) > 0


class TestRTDP:
    def test_simple(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 10.0)
        mdp.add_state("s1", terminal=True)
        mdp.set_initial("s0")
        result = rtdp(mdp, gamma=0.9, trials=500, seed=42)
        assert abs(result.values["s0"] - 10.0) < 0.1
        assert result.policy["s0"] == "go"

    def test_agrees_approximately(self):
        mdp = two_state_mdp()
        mdp.set_initial("s0")
        vi = value_iteration(mdp, gamma=0.9)
        rt = rtdp(mdp, gamma=0.9, trials=3000, seed=42)
        for s in mdp.states:
            assert abs(vi.values[s] - rt.values[s]) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Solver Comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareSolvers:
    def test_all_agree(self):
        mdp = two_state_mdp()
        results = compare_solvers(mdp, gamma=0.9)
        vi_vals = results["value_iteration"].values
        pi_vals = results["policy_iteration"].values
        lp_vals = results["lp"].values
        for s in mdp.states:
            assert abs(vi_vals[s] - pi_vals[s]) < 0.01
            assert abs(vi_vals[s] - lp_vals[s]) < 0.01

    def test_all_converge(self):
        mdp = two_state_mdp()
        results = compare_solvers(mdp, gamma=0.9)
        for name, r in results.items():
            assert r.converged, f"{name} did not converge"


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation & Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulation:
    def test_simulate_deterministic(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 5.0)
        mdp.add_state("s1", terminal=True)
        mdp.set_initial("s0")
        traj = simulate(mdp, {"s0": "go", "s1": ""}, steps=10, seed=42)
        assert len(traj) == 1
        assert traj[0] == ("s0", "go", "s1", 5.0)

    def test_simulate_terminates_at_terminal(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 1.0)
        mdp.add_state("s1", terminal=True)
        traj = simulate(mdp, {"s0": "go"}, steps=100, start="s0", seed=42)
        assert len(traj) == 1

    def test_simulate_multi_step(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 1.0)
        mdp.add_transition("s1", "go", "s2", 1.0, 2.0)
        mdp.add_state("s2", terminal=True)
        traj = simulate(mdp, {"s0": "go", "s1": "go"}, steps=10, start="s0", seed=42)
        assert len(traj) == 2
        assert traj[0][3] == 1.0
        assert traj[1][3] == 2.0

    def test_expected_total_reward(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 10.0)
        mdp.add_state("s1", terminal=True)
        mdp.set_initial("s0")
        etr = expected_total_reward(mdp, {"s0": "go"}, gamma=0.9,
                                    n_simulations=1000, seed=42)
        assert abs(etr - 10.0) < 0.5

    def test_policy_advantage(self):
        mdp = MDP()
        mdp.add_transition("s0", "good", "goal", 1.0, 10.0)
        mdp.add_transition("s0", "bad", "goal", 1.0, 1.0)
        mdp.add_state("goal", terminal=True)
        adv = policy_advantage(mdp, {"s0": "good", "goal": ""},
                               {"s0": "bad", "goal": ""}, gamma=0.9)
        assert adv["s0"] > 0  # good policy has higher value


class TestOccupancy:
    def test_deterministic_chain(self):
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 0.0)
        mdp.add_transition("s1", "go", "s2", 1.0, 0.0)
        mdp.add_state("s2", terminal=True)
        mdp.set_initial("s0")
        occ = occupancy_measure(mdp, {"s0": "go", "s1": "go"}, gamma=0.9)
        assert ("s0", "go") in occ
        assert ("s1", "go") in occ
        assert abs(occ[("s0", "go")] - 1.0) < 0.01  # visited at t=0
        assert abs(occ[("s1", "go")] - 0.9) < 0.01  # visited at t=1, discounted

    def test_stationary_occupancy(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s0", 1.0, 1.0)
        mdp.set_initial("s0")
        occ = occupancy_measure(mdp, {"s0": "a"}, gamma=0.9)
        # Should be geometric series: 1 + 0.9 + 0.81 + ... = 1/(1-0.9) = 10
        assert abs(occ[("s0", "a")] - 10.0) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  V210 Composition: MDP <-> Influence Diagram
# ─────────────────────────────────────────────────────────────────────────────

class TestInfluenceDiagramConversion:
    def test_mdp_to_id_structure(self):
        mdp = two_state_mdp()
        mdp.set_initial("s0")
        diagram = mdp_to_influence_diagram(mdp, horizon=2)
        assert "S_0" in diagram.bn.nodes
        assert "S_1" in diagram.bn.nodes
        assert "S_2" in diagram.bn.nodes
        assert "A_0" in diagram.bn.nodes
        assert "A_1" in diagram.bn.nodes

    def test_mdp_to_id_decisions(self):
        mdp = two_state_mdp()
        mdp.set_initial("s0")
        diagram = mdp_to_influence_diagram(mdp, horizon=2)
        decisions = diagram.decision_nodes()
        assert len(decisions) == 2

    def test_mdp_to_id_utilities(self):
        mdp = two_state_mdp()
        mdp.set_initial("s0")
        diagram = mdp_to_influence_diagram(mdp, horizon=2)
        utilities = diagram.utility_nodes()
        assert len(utilities) == 2

    def test_mdp_to_id_initial_distribution(self):
        mdp = two_state_mdp()
        mdp.set_initial("s0")
        diagram = mdp_to_influence_diagram(mdp, horizon=1)
        cpt_factor = diagram.bn.cpts["S_0"]
        assert cpt_factor.get({"S_0": "s0"}) == 1.0
        assert cpt_factor.get({"S_0": "s1"}) == 0.0

    def test_mdp_to_id_transition_cpt(self):
        mdp = MDP()
        mdp.add_transition("A", "x", "B", 1.0, 5.0)
        mdp.add_state("B", terminal=True)
        mdp.set_initial("A")
        diagram = mdp_to_influence_diagram(mdp, horizon=1)
        cpt_factor = diagram.bn.cpts["S_1"]
        assert cpt_factor.get({"S_0": "A", "A_0": "x", "S_1": "B"}) == 1.0
        assert cpt_factor.get({"S_0": "A", "A_0": "x", "S_1": "A"}) == 0.0


class TestBNConversion:
    def test_transition_bn(self):
        mdp = two_state_mdp()
        bn = mdp_transition_bn(mdp, "s0", "a")
        assert "next_state" in bn.nodes
        cpt_factor = bn.cpts["next_state"]
        # P(s0|s0,a) = 0.5, P(s1|s0,a) = 0.5
        assert abs(cpt_factor.get({"next_state": "s0"}) - 0.5) < 1e-9
        assert abs(cpt_factor.get({"next_state": "s1"}) - 0.5) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  Example MDPs
# ─────────────────────────────────────────────────────────────────────────────

class TestGridworld:
    def test_structure(self):
        mdp = gridworld(rows=3, cols=3, goal=(2, 2), trap=None)
        assert len(mdp.states) == 9
        assert "(2,2)" in mdp.terminal_states
        assert mdp.initial_state == "(0,0)"

    def test_with_trap(self):
        mdp = gridworld(rows=3, cols=3, goal=(2, 2), trap=(1, 1))
        assert "(1,1)" in mdp.terminal_states

    def test_valid(self):
        mdp = gridworld(rows=3, cols=3, goal=(2, 2), trap=None)
        assert mdp.validate() == []

    def test_optimal_policy_moves_toward_goal(self):
        mdp = gridworld(rows=3, cols=3, goal=(2, 2), trap=None, step_reward=-0.1)
        result = value_iteration(mdp, gamma=0.95)
        # From (0,0), should go S or E
        assert result.policy["(0,0)"] in ("S", "E")

    def test_with_slip(self):
        mdp = gridworld(rows=3, cols=3, goal=(2, 2), trap=None, slip_prob=0.1)
        assert mdp.validate() == []
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged

    def test_values_decrease_from_goal(self):
        mdp = gridworld(rows=4, cols=4, goal=(3, 3), trap=None, step_reward=-0.04)
        result = value_iteration(mdp, gamma=0.9)
        # Adjacent to goal should have higher value than far away
        assert result.values["(3,2)"] > result.values["(0,0)"]
        assert result.values["(2,3)"] > result.values["(0,0)"]


class TestInventory:
    def test_structure(self):
        mdp = inventory_management(max_stock=3, max_order=2)
        assert "stock_0" in mdp._state_set
        assert "stock_3" in mdp._state_set

    def test_valid(self):
        mdp = inventory_management(max_stock=3, max_order=2)
        assert mdp.validate() == []

    def test_solvable(self):
        mdp = inventory_management(max_stock=3, max_order=2)
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged

    def test_policy_orders_when_empty(self):
        mdp = inventory_management(max_stock=3, max_order=2,
                                   holding_cost=-0.5, stockout_cost=-10.0)
        result = value_iteration(mdp, gamma=0.9)
        # When stock is 0, should order something
        assert result.policy["stock_0"] != "order_0"


class TestGambling:
    def test_structure(self):
        mdp = gambling(states_count=10)
        assert "$0" in mdp.terminal_states
        assert "$10" in mdp.terminal_states
        assert mdp.initial_state == "$50"

    def test_small_solvable(self):
        mdp = gambling(states_count=10)
        mdp.set_initial("$5")
        result = value_iteration(mdp, gamma=1.0)
        assert result.converged
        # P(reaching $10) should be positive but < 1 (unfair coin)
        assert 0.0 < result.values["$5"] < 1.0

    def test_terminal_values_zero(self):
        mdp = gambling(states_count=10)
        result = value_iteration(mdp, gamma=1.0)
        assert result.values["$0"] == 0.0
        assert result.values["$10"] == 0.0


class TestTwoStateMDP:
    def test_structure(self):
        mdp = two_state_mdp()
        assert len(mdp.states) == 2
        assert len(mdp.actions) == 2

    def test_valid(self):
        assert two_state_mdp().validate() == []


# ─────────────────────────────────────────────────────────────────────────────
#  Expected Reward
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectedReward:
    def test_deterministic(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0, 10.0)
        mdp.add_state("s1", terminal=True)
        V = {"s0": 0.0, "s1": 0.0}
        assert abs(mdp.expected_reward("s0", "a", V, 0.9) - 10.0) < 1e-9

    def test_stochastic(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 0.5, 10.0)
        mdp.add_transition("s0", "a", "s0", 0.5, 0.0)
        V = {"s0": 5.0, "s1": 0.0}
        # 0.5*(10 + 0.9*0) + 0.5*(0 + 0.9*5) = 5 + 2.25 = 7.25
        assert abs(mdp.expected_reward("s0", "a", V, 0.9) - 7.25) < 1e-9

    def test_with_discount(self):
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0, 0.0)
        V = {"s0": 0.0, "s1": 10.0}
        # 0 + 0.5 * 10 = 5.0
        assert abs(mdp.expected_reward("s0", "a", V, 0.5) - 5.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_fields(self):
        mdp = two_state_mdp()
        s = mdp_summary(mdp)
        assert s["name"] == "TwoState"
        assert s["states"] == 2
        assert s["actions"] == 2
        assert s["transitions"] > 0
        assert s["issues"] == []

    def test_gridworld_summary(self):
        mdp = gridworld(rows=3, cols=3, goal=(2, 2))
        s = mdp_summary(mdp)
        assert s["states"] == 9
        assert s["initial_state"] == "(0,0)"


# ─────────────────────────────────────────────────────────────────────────────
#  Discount Factor Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscountEdgeCases:
    def test_gamma_zero(self):
        """With gamma=0, only immediate rewards matter."""
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0, 5.0)
        mdp.add_transition("s1", "a", "s0", 1.0, 100.0)
        result = value_iteration(mdp, gamma=0.0)
        assert result.converged
        assert abs(result.values["s0"] - 5.0) < 0.01
        assert abs(result.values["s1"] - 100.0) < 0.01

    def test_high_gamma_propagates(self):
        """With high gamma, future rewards propagate."""
        mdp = MDP()
        mdp.add_transition("s0", "a", "s1", 1.0, 0.0)
        mdp.add_transition("s1", "a", "s2", 1.0, 100.0)
        mdp.add_state("s2", terminal=True)
        result = value_iteration(mdp, gamma=0.9)
        assert result.values["s0"] > 80.0  # 0.9 * 90 = 81


class TestMultiActionOptimality:
    def test_chooses_best_action(self):
        mdp = MDP()
        mdp.add_transition("s0", "safe", "s1", 1.0, 5.0)
        mdp.add_transition("s0", "risky", "s1", 0.5, 20.0)
        mdp.add_transition("s0", "risky", "s0", 0.5, -5.0)
        mdp.add_state("s1", terminal=True)
        result = value_iteration(mdp, gamma=0.9)
        # risky: 0.5*20 + 0.5*(-5 + 0.9*V(s0))
        # safe: 5
        # risky yields higher expected value
        assert result.policy["s0"] == "risky"

    def test_risk_averse_with_punishment(self):
        mdp = MDP()
        mdp.add_transition("s0", "safe", "goal", 1.0, 5.0)
        mdp.add_transition("s0", "risky", "goal", 0.5, 12.0)
        mdp.add_transition("s0", "risky", "doom", 0.5, -100.0)
        mdp.add_state("goal", terminal=True)
        mdp.add_state("doom", terminal=True)
        result = value_iteration(mdp, gamma=0.9)
        assert result.policy["s0"] == "safe"  # risky EV = 0.5*12 + 0.5*(-100) = -44


class TestLargerMDP:
    def test_chain_mdp(self):
        """Linear chain: s0 -> s1 -> ... -> s9 (terminal), reward at end."""
        mdp = MDP()
        for i in range(10):
            mdp.add_state(f"s{i}", terminal=(i == 9))
        for i in range(9):
            mdp.add_transition(f"s{i}", "go", f"s{i+1}", 1.0,
                               10.0 if i == 8 else 0.0)
        mdp.set_initial("s0")
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged
        # V(s0) = 0.9^8 * 10 = ~4.3
        assert abs(result.values["s0"] - 0.9**8 * 10.0) < 0.01

    def test_cycle_mdp(self):
        """Cyclic MDP: keep going around collecting reward."""
        mdp = MDP()
        mdp.add_transition("s0", "go", "s1", 1.0, 1.0)
        mdp.add_transition("s1", "go", "s0", 1.0, 1.0)
        mdp.set_initial("s0")
        result = value_iteration(mdp, gamma=0.9)
        assert result.converged
        # V(s0) = 1 + 0.9*(1 + 0.9*V(s0)) => V(s0) = (1+0.9) / (1-0.81) ~= 10.0
        expected = (1.0 + 0.9) / (1.0 - 0.81)
        assert abs(result.values["s0"] - expected) < 0.01
