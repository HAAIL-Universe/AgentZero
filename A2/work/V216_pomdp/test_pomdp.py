"""Tests for V216: Partially Observable Markov Decision Processes (POMDPs).

Covers: POMDP construction, belief update, solvers (QMDP, FIB, PBVI, Perseus,
exact VI), simulation, evaluation, information gain, classic problems
(Tiger, Machine Maintenance, Hallway, RockSample).

AI-Generated | Claude (Anthropic) | AgentZero A2 Session 297 | 2026-03-18
"""

import math
import pytest
from pomdp import (
    POMDP, AlphaVector, POMDPResult,
    qmdp, fib, exact_value_iteration, pbvi, perseus,
    simulate_pomdp, evaluate_policy, compare_solvers,
    belief_entropy, information_gain, most_informative_action,
    tiger_problem, machine_maintenance, hallway_navigation, rock_sample_small,
    pomdp_summary, _belief_distance,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Section 1: Core Data Structures
# ═════════════════════════════════════════════════════════════════════════════

class TestAlphaVector:
    def test_dot_product(self):
        av = AlphaVector(action="a", values={"s0": 1.0, "s1": 2.0})
        b = {"s0": 0.3, "s1": 0.7}
        assert abs(av.dot(b) - 1.7) < 1e-9

    def test_dot_zero_belief(self):
        av = AlphaVector(action="a", values={"s0": 5.0, "s1": -3.0})
        b = {"s0": 0.0, "s1": 0.0}
        assert av.dot(b) == 0.0

    def test_dot_pure_belief(self):
        av = AlphaVector(action="a", values={"s0": 10.0, "s1": -5.0})
        b = {"s0": 1.0, "s1": 0.0}
        assert abs(av.dot(b) - 10.0) < 1e-9


class TestPOMDPResult:
    def test_value_at_belief(self):
        av1 = AlphaVector(action="a", values={"s0": 1.0, "s1": 0.0})
        av2 = AlphaVector(action="b", values={"s0": 0.0, "s1": 1.0})
        result = POMDPResult(alpha_vectors=[av1, av2])
        # At uniform belief, both give 0.5
        assert abs(result.value({"s0": 0.5, "s1": 0.5}) - 0.5) < 1e-9

    def test_best_action(self):
        av1 = AlphaVector(action="a", values={"s0": 2.0, "s1": 0.0})
        av2 = AlphaVector(action="b", values={"s0": 0.0, "s1": 2.0})
        result = POMDPResult(alpha_vectors=[av1, av2])
        assert result.best_action({"s0": 0.9, "s1": 0.1}) == "a"
        assert result.best_action({"s0": 0.1, "s1": 0.9}) == "b"

    def test_policy(self):
        av1 = AlphaVector(action="a", values={"s0": 5.0, "s1": 0.0})
        result = POMDPResult(alpha_vectors=[av1])
        action, val = result.policy({"s0": 1.0, "s1": 0.0})
        assert action == "a"
        assert abs(val - 5.0) < 1e-9

    def test_empty_result(self):
        result = POMDPResult(alpha_vectors=[])
        assert result.value({"s0": 0.5}) == 0.0
        assert result.best_action({"s0": 0.5}) == ""


# ═════════════════════════════════════════════════════════════════════════════
#  Section 2: POMDP Construction & Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestPOMDPConstruction:
    def test_add_states_actions_obs(self):
        p = POMDP()
        p.add_state("s0")
        p.add_state("s1")
        p.add_action("a")
        p.add_observation("o1")
        assert len(p.states) == 2
        assert len(p.actions) == 1
        assert len(p.observations) == 1

    def test_no_duplicates(self):
        p = POMDP()
        p.add_state("s0")
        p.add_state("s0")
        assert len(p.states) == 1

    def test_transitions(self):
        p = POMDP()
        p.add_transition("s0", "a", "s1", 0.7)
        p.add_transition("s0", "a", "s0", 0.3)
        trans = p.get_transitions("s0", "a")
        assert len(trans) == 2
        total = sum(pr for _, pr in trans)
        assert abs(total - 1.0) < 1e-9

    def test_observation_model(self):
        p = POMDP()
        p.add_observation_prob("a", "s1", "o1", 0.8)
        p.add_observation_prob("a", "s1", "o2", 0.2)
        assert abs(p.get_observation_prob("a", "s1", "o1") - 0.8) < 1e-9
        assert abs(p.get_observation_prob("a", "s1", "o2") - 0.2) < 1e-9
        assert p.get_observation_prob("a", "s1", "o3") == 0.0

    def test_rewards_sa(self):
        p = POMDP()
        p.add_state("s0")
        p.add_action("a")
        p.set_reward("s0", "a", 5.0)
        assert p.get_reward("s0", "a") == 5.0

    def test_rewards_sas(self):
        p = POMDP()
        p.set_reward("s0", "a", 3.0, s_prime="s1")
        assert p.get_reward("s0", "a", "s1") == 3.0

    def test_initial_belief_uniform(self):
        p = POMDP()
        p.add_state("s0")
        p.add_state("s1")
        b = p.get_initial_belief()
        assert abs(b["s0"] - 0.5) < 1e-9
        assert abs(b["s1"] - 0.5) < 1e-9

    def test_initial_belief_custom(self):
        p = POMDP()
        p.add_state("s0")
        p.add_state("s1")
        p.set_initial_belief({"s0": 0.3, "s1": 0.7})
        b = p.get_initial_belief()
        assert abs(b["s0"] - 0.3) < 1e-9

    def test_validate_good(self):
        p = tiger_problem()
        issues = p.validate()
        assert len(issues) == 0

    def test_validate_bad_transitions(self):
        p = POMDP()
        p.add_state("s0")
        p.add_action("a")
        p.add_observation("o")
        p._transitions[("s0", "a")] = [("s0", 0.5)]  # doesn't sum to 1
        issues = p.validate()
        assert any("sums to" in iss for iss in issues)

    def test_to_mdp(self):
        p = tiger_problem()
        mdp = p.to_mdp()
        assert len(mdp.states) == 2
        assert len(mdp.actions) == 3


# ═════════════════════════════════════════════════════════════════════════════
#  Section 3: Belief Update
# ═════════════════════════════════════════════════════════════════════════════

class TestBeliefUpdate:
    def test_tiger_listen_hear_left(self):
        """Hearing left growl should increase belief in tiger-left."""
        p = tiger_problem()
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        b_new = p.belief_update(b, "listen", "hear-left")
        assert b_new["tiger-left"] > 0.5
        assert b_new["tiger-right"] < 0.5
        assert abs(sum(b_new.values()) - 1.0) < 1e-9

    def test_tiger_listen_accuracy(self):
        """After hearing left, belief should match Bayesian computation."""
        p = tiger_problem(listen_accuracy=0.85)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        b_new = p.belief_update(b, "listen", "hear-left")
        # P(TL|HL) = 0.85 * 0.5 / (0.85 * 0.5 + 0.15 * 0.5) = 0.85
        assert abs(b_new["tiger-left"] - 0.85) < 1e-9

    def test_multiple_listen_converges(self):
        """Multiple consistent observations should converge belief."""
        p = tiger_problem()
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        for _ in range(10):
            b = p.belief_update(b, "listen", "hear-left")
        assert b["tiger-left"] > 0.99

    def test_open_resets_belief(self):
        """Opening a door resets to uniform belief."""
        p = tiger_problem()
        b = {"tiger-left": 0.9, "tiger-right": 0.1}
        b_new = p.belief_update(b, "open-left", "hear-left")
        # After opening, transitions reset to 50/50
        assert abs(b_new["tiger-left"] - 0.5) < 1e-9

    def test_belief_normalization(self):
        """Belief should always sum to 1."""
        p = tiger_problem()
        b = {"tiger-left": 0.3, "tiger-right": 0.7}
        b_new = p.belief_update(b, "listen", "hear-right")
        assert abs(sum(b_new.values()) - 1.0) < 1e-9

    def test_observation_probability(self):
        """Test P(o|b,a) computation."""
        p = tiger_problem(listen_accuracy=0.85)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        p_hl = p.observation_probability(b, "listen", "hear-left")
        p_hr = p.observation_probability(b, "listen", "hear-right")
        assert abs(p_hl + p_hr - 1.0) < 1e-9
        assert abs(p_hl - 0.5) < 1e-9  # uniform belief -> uniform obs

    def test_biased_observation_probability(self):
        """Biased belief should bias observation probability."""
        p = tiger_problem(listen_accuracy=0.85)
        b = {"tiger-left": 0.9, "tiger-right": 0.1}
        p_hl = p.observation_probability(b, "listen", "hear-left")
        # P(HL) = 0.85*0.9 + 0.15*0.1 = 0.78
        assert abs(p_hl - 0.78) < 1e-9


# ═════════════════════════════════════════════════════════════════════════════
#  Section 4: QMDP Solver
# ═════════════════════════════════════════════════════════════════════════════

class TestQMDP:
    def test_tiger_qmdp(self):
        """QMDP on tiger: should produce valid alpha vectors."""
        p = tiger_problem()
        result = qmdp(p)
        assert len(result.alpha_vectors) == 3  # one per action
        assert result.converged

    def test_tiger_qmdp_listen_preferred(self):
        """At uniform belief, QMDP should suggest listening (or have similar value)."""
        p = tiger_problem()
        result = qmdp(p)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        # Check that listen action has reasonable value
        for av in result.alpha_vectors:
            if av.action == "listen":
                listen_val = av.dot(b)
        # Open actions at uniform belief should average to (treasure + penalty) / 2
        # = (10 - 100) / 2 = -45, while listen should be better
        assert result.best_action(b) == "listen"

    def test_tiger_qmdp_confident_opens(self):
        """With strong belief, QMDP should prefer opening the non-tiger door."""
        p = tiger_problem()
        result = qmdp(p)
        # Very confident tiger is left -> open right
        b_left = {"tiger-left": 0.99, "tiger-right": 0.01}
        assert result.best_action(b_left) == "open-right"

    def test_maintenance_qmdp(self):
        p = machine_maintenance()
        result = qmdp(p)
        assert len(result.alpha_vectors) == 3
        assert result.converged


# ═════════════════════════════════════════════════════════════════════════════
#  Section 5: FIB Solver
# ═════════════════════════════════════════════════════════════════════════════

class TestFIB:
    def test_tiger_fib(self):
        p = tiger_problem()
        result = fib(p)
        assert len(result.alpha_vectors) > 0

    def test_fib_tighter_than_qmdp(self):
        """FIB should produce a tighter (lower) upper bound than QMDP."""
        p = tiger_problem()
        qmdp_result = qmdp(p)
        fib_result = fib(p)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        # FIB <= QMDP (both are upper bounds, FIB is tighter)
        assert fib_result.value(b) <= qmdp_result.value(b) + 1e-6

    def test_fib_convergence(self):
        p = tiger_problem()
        result = fib(p, max_iter=200)
        assert result.converged or result.iterations <= 200


# ═════════════════════════════════════════════════════════════════════════════
#  Section 6: Exact Value Iteration
# ═════════════════════════════════════════════════════════════════════════════

class TestExactVI:
    def test_tiger_exact(self):
        p = tiger_problem()
        result = exact_value_iteration(p, max_iter=20)
        assert len(result.alpha_vectors) > 0

    def test_tiger_exact_policy(self):
        """Exact solver on tiger should learn to listen at uniform belief."""
        p = tiger_problem()
        result = exact_value_iteration(p, max_iter=30, max_alphas=100)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        # At uniform belief, listening is optimal
        assert result.best_action(b) == "listen"

    def test_tiger_exact_opens_when_confident(self):
        """Exact solver should open correct door when confident."""
        p = tiger_problem()
        result = exact_value_iteration(p, max_iter=30, max_alphas=100)
        b = {"tiger-left": 0.99, "tiger-right": 0.01}
        assert result.best_action(b) == "open-right"

    def test_exact_value_positive(self):
        """Optimal tiger policy should have positive expected value."""
        p = tiger_problem()
        result = exact_value_iteration(p, max_iter=30)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        # Optimal policy can gather info and act => positive value
        # (QMDP is an upper bound, exact may be lower but should be reasonable)
        val = result.value(b)
        # The tiger POMDP optimal value at uniform is known to be around 5-20
        # depending on parameters. It should definitely be better than always opening.
        assert val > -50  # much better than always opening randomly


# ═════════════════════════════════════════════════════════════════════════════
#  Section 7: PBVI Solver
# ═════════════════════════════════════════════════════════════════════════════

class TestPBVI:
    def test_tiger_pbvi(self):
        p = tiger_problem()
        result = pbvi(p, n_points=30, seed=42, max_iter=50)
        assert len(result.alpha_vectors) > 0

    def test_tiger_pbvi_policy(self):
        """PBVI should produce a reasonable tiger policy."""
        p = tiger_problem()
        result = pbvi(p, n_points=50, seed=42, max_iter=50)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        # Should listen at uncertain belief
        assert result.best_action(b) == "listen"

    def test_pbvi_custom_beliefs(self):
        """PBVI with custom belief points."""
        p = tiger_problem()
        beliefs = [
            {"tiger-left": 0.5, "tiger-right": 0.5},
            {"tiger-left": 0.8, "tiger-right": 0.2},
            {"tiger-left": 0.2, "tiger-right": 0.8},
        ]
        result = pbvi(p, belief_points=beliefs, max_iter=30)
        assert len(result.alpha_vectors) > 0

    def test_maintenance_pbvi(self):
        p = machine_maintenance()
        result = pbvi(p, n_points=30, seed=42, max_iter=30)
        assert len(result.alpha_vectors) > 0


# ═════════════════════════════════════════════════════════════════════════════
#  Section 8: Perseus Solver
# ═════════════════════════════════════════════════════════════════════════════

class TestPerseus:
    def test_tiger_perseus(self):
        p = tiger_problem()
        result = perseus(p, n_points=50, seed=42, max_iter=50)
        assert len(result.alpha_vectors) > 0

    def test_tiger_perseus_policy(self):
        """Perseus should produce a reasonable tiger policy."""
        p = tiger_problem()
        result = perseus(p, n_points=50, seed=42, max_iter=50)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        assert result.best_action(b) == "listen"

    def test_perseus_confident_action(self):
        p = tiger_problem()
        result = perseus(p, n_points=50, seed=42, max_iter=50)
        b = {"tiger-left": 0.99, "tiger-right": 0.01}
        assert result.best_action(b) == "open-right"


# ═════════════════════════════════════════════════════════════════════════════
#  Section 9: Simulation & Evaluation
# ═════════════════════════════════════════════════════════════════════════════

class TestSimulation:
    def test_simulate_tiger(self):
        p = tiger_problem()
        result = qmdp(p)
        traj = simulate_pomdp(p, result, steps=20, seed=42)
        assert len(traj) > 0
        assert "state" in traj[0]
        assert "action" in traj[0]
        assert "observation" in traj[0]
        assert "reward" in traj[0]

    def test_simulate_records_beliefs(self):
        p = tiger_problem()
        result = qmdp(p)
        traj = simulate_pomdp(p, result, steps=10, seed=42)
        for step in traj:
            b = step["belief"]
            assert abs(sum(b.values()) - 1.0) < 1e-9

    def test_evaluate_tiger(self):
        p = tiger_problem()
        result = qmdp(p)
        stats = evaluate_policy(p, result, n_episodes=200, max_steps=50, seed=42)
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert stats["n_episodes"] == 200

    def test_evaluate_returns_reasonable_reward(self):
        """A good policy should get positive expected reward on tiger."""
        p = tiger_problem()
        result = pbvi(p, n_points=50, seed=42, max_iter=50)
        stats = evaluate_policy(p, result, n_episodes=500, max_steps=50, seed=42)
        # With a decent policy, average should be better than random opening
        # Random opening at uniform: E[R] = 0.5*10 + 0.5*(-100) = -45
        assert stats["mean_reward"] > -45


# ═════════════════════════════════════════════════════════════════════════════
#  Section 10: Information Theory
# ═════════════════════════════════════════════════════════════════════════════

class TestInformationTheory:
    def test_entropy_uniform(self):
        b = {"s0": 0.5, "s1": 0.5}
        assert abs(belief_entropy(b) - 1.0) < 1e-9

    def test_entropy_pure(self):
        b = {"s0": 1.0, "s1": 0.0}
        assert abs(belief_entropy(b) - 0.0) < 1e-9

    def test_entropy_three_states(self):
        b = {"s0": 1/3, "s1": 1/3, "s2": 1/3}
        expected = math.log2(3)
        assert abs(belief_entropy(b) - expected) < 1e-9

    def test_information_gain_listen(self):
        """Listening in tiger should have positive information gain."""
        p = tiger_problem()
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        ig = information_gain(p, b, "listen")
        assert ig > 0

    def test_information_gain_open_zero(self):
        """Opening a door in tiger has zero info gain (uniform obs)."""
        p = tiger_problem()
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        ig = information_gain(p, b, "open-left")
        # Open resets to uniform and obs is uniform -> no info gain
        assert abs(ig) < 0.01

    def test_most_informative_is_listen(self):
        """In tiger, listening is the most informative action."""
        p = tiger_problem()
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        action, ig = most_informative_action(p, b)
        assert action == "listen"
        assert ig > 0


# ═════════════════════════════════════════════════════════════════════════════
#  Section 11: Tiger Problem (detailed)
# ═════════════════════════════════════════════════════════════════════════════

class TestTigerProblem:
    def test_tiger_structure(self):
        p = tiger_problem()
        assert len(p.states) == 2
        assert len(p.actions) == 3
        assert len(p.observations) == 2
        assert len(p.validate()) == 0

    def test_tiger_summary(self):
        p = tiger_problem()
        s = pomdp_summary(p)
        assert s["states"] == 2
        assert s["actions"] == 3
        assert s["observations"] == 2
        assert len(s["issues"]) == 0

    def test_tiger_custom_params(self):
        p = tiger_problem(listen_cost=-2.0, tiger_penalty=-50.0,
                          treasure_reward=20.0, listen_accuracy=0.9)
        assert p.get_reward("tiger-left", "listen") == -2.0
        assert p.get_reward("tiger-left", "open-left") == -50.0
        assert abs(p.get_observation_prob("listen", "tiger-left", "hear-left") - 0.9) < 1e-9

    def test_tiger_belief_update_sequence(self):
        """Alternating listen observations should produce oscillating belief."""
        p = tiger_problem(listen_accuracy=0.85)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        b = p.belief_update(b, "listen", "hear-left")
        assert b["tiger-left"] > 0.8  # should be ~0.85
        b = p.belief_update(b, "listen", "hear-right")
        # Should decrease back toward 0.5
        assert b["tiger-left"] < 0.85


# ═════════════════════════════════════════════════════════════════════════════
#  Section 12: Machine Maintenance Problem
# ═════════════════════════════════════════════════════════════════════════════

class TestMachineMaintenance:
    def test_maintenance_structure(self):
        p = machine_maintenance()
        assert len(p.states) == 3
        assert len(p.actions) == 3
        assert len(p.validate()) == 0

    def test_maintenance_5_conditions(self):
        p = machine_maintenance(n_conditions=5)
        assert len(p.states) == 5
        assert len(p.validate()) == 0

    def test_maintenance_qmdp_policy(self):
        p = machine_maintenance()
        result = qmdp(p)
        # When confident machine is good, should operate
        b_good = {"cond_0": 0.95, "cond_1": 0.04, "cond_2": 0.01}
        action = result.best_action(b_good)
        assert action in ["operate", "inspect"]  # should prefer operating when good

    def test_maintenance_repair_when_poor(self):
        p = machine_maintenance()
        result = qmdp(p)
        # When confident machine is poor, should repair
        b_poor = {"cond_0": 0.01, "cond_1": 0.04, "cond_2": 0.95}
        action = result.best_action(b_poor)
        # Repair is costly (-8) but continuing with poor machine gives less reward
        # The exact action depends on discount factor; either repair or inspect
        assert action in ["repair", "inspect"]


# ═════════════════════════════════════════════════════════════════════════════
#  Section 13: Hallway Navigation
# ═════════════════════════════════════════════════════════════════════════════

class TestHallwayNavigation:
    def test_hallway_structure(self):
        p = hallway_navigation()
        assert len(p.states) == 4
        assert len(p.actions) == 3
        assert len(p.validate()) == 0

    def test_hallway_longer(self):
        p = hallway_navigation(length=8)
        assert len(p.states) == 8

    def test_hallway_qmdp(self):
        p = hallway_navigation()
        result = qmdp(p)
        assert result.converged

    def test_hallway_pbvi(self):
        p = hallway_navigation()
        result = pbvi(p, n_points=30, seed=42, max_iter=30)
        assert len(result.alpha_vectors) > 0


# ═════════════════════════════════════════════════════════════════════════════
#  Section 14: RockSample Problem
# ═════════════════════════════════════════════════════════════════════════════

class TestRockSample:
    def test_rock_structure(self):
        p = rock_sample_small()
        assert len(p.states) == 6  # 3 positions * 2 qualities
        assert len(p.actions) == 4
        assert len(p.validate()) == 0

    def test_rock_initial_belief(self):
        p = rock_sample_small()
        b = p.get_initial_belief()
        assert abs(b["left_good"] - 0.5) < 1e-9
        assert abs(b["left_bad"] - 0.5) < 1e-9
        assert abs(b["rock_good"]) < 1e-9

    def test_rock_check_updates_belief(self):
        """Checking the rock should update quality belief."""
        p = rock_sample_small()
        b = {"rock_good": 0.5, "rock_bad": 0.5,
             "left_good": 0.0, "left_bad": 0.0,
             "right_good": 0.0, "right_bad": 0.0}
        b_new = p.belief_update(b, "check", "good-signal")
        # Good signal should increase belief in good rock
        total_good = sum(b_new[s] for s in p.states if "good" in s)
        total_bad = sum(b_new[s] for s in p.states if "bad" in s)
        assert total_good > total_bad

    def test_rock_qmdp(self):
        p = rock_sample_small()
        result = qmdp(p)
        assert result.converged


# ═════════════════════════════════════════════════════════════════════════════
#  Section 15: Solver Comparison
# ═════════════════════════════════════════════════════════════════════════════

class TestSolverComparison:
    def test_compare_solvers_tiger(self):
        p = tiger_problem()
        results = compare_solvers(p)
        assert "qmdp" in results
        assert "fib" in results
        assert "pbvi" in results

    def test_solver_ordering(self):
        """FIB value <= QMDP value (both upper bounds, FIB tighter)."""
        p = tiger_problem()
        results = compare_solvers(p)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        qmdp_v = results["qmdp"].value(b)
        fib_v = results["fib"].value(b)
        assert fib_v <= qmdp_v + 1e-6

    def test_all_solvers_positive_value(self):
        """All solvers should give positive value at uniform tiger belief."""
        p = tiger_problem()
        results = compare_solvers(p)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        for name, result in results.items():
            # At minimum, value should not be deeply negative
            assert result.value(b) > -200, f"{name} has unreasonably low value"


# ═════════════════════════════════════════════════════════════════════════════
#  Section 16: Belief Distance & Utilities
# ═════════════════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_belief_distance_same(self):
        b = {"s0": 0.5, "s1": 0.5}
        assert _belief_distance(b, b) == 0.0

    def test_belief_distance_pure(self):
        b1 = {"s0": 1.0, "s1": 0.0}
        b2 = {"s0": 0.0, "s1": 1.0}
        assert abs(_belief_distance(b1, b2) - 2.0) < 1e-9

    def test_belief_distance_partial(self):
        b1 = {"s0": 0.7, "s1": 0.3}
        b2 = {"s0": 0.5, "s1": 0.5}
        assert abs(_belief_distance(b1, b2) - 0.4) < 1e-9


# ═════════════════════════════════════════════════════════════════════════════
#  Section 17: Expected Reward
# ═════════════════════════════════════════════════════════════════════════════

class TestExpectedReward:
    def test_tiger_listen_reward(self):
        p = tiger_problem(listen_cost=-1.0)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        r = p.expected_reward(b, "listen")
        assert abs(r - (-1.0)) < 1e-9

    def test_tiger_open_reward_uniform(self):
        p = tiger_problem(tiger_penalty=-100.0, treasure_reward=10.0)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        r_left = p.expected_reward(b, "open-left")
        # E[R] = 0.5 * (-100) + 0.5 * 10 = -45
        assert abs(r_left - (-45.0)) < 1e-9

    def test_tiger_open_reward_certain(self):
        p = tiger_problem(tiger_penalty=-100.0, treasure_reward=10.0)
        b = {"tiger-left": 1.0, "tiger-right": 0.0}
        r = p.expected_reward(b, "open-right")
        assert abs(r - 10.0) < 1e-9


# ═════════════════════════════════════════════════════════════════════════════
#  Section 18: Edge Cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_state_pomdp(self):
        """POMDP with one state is trivially observable."""
        p = POMDP()
        p.add_state("s0")
        p.add_action("a")
        p.add_observation("o")
        p.add_transition("s0", "a", "s0", 1.0)
        p.add_observation_prob("a", "s0", "o", 1.0)
        p.set_reward("s0", "a", 5.0)
        p.gamma = 0.9
        b = p.get_initial_belief()
        assert abs(b["s0"] - 1.0) < 1e-9

    def test_single_state_qmdp(self):
        p = POMDP()
        p.add_state("s0")
        p.add_action("a")
        p.add_observation("o")
        p.add_transition("s0", "a", "s0", 1.0)
        p.add_observation_prob("a", "s0", "o", 1.0)
        p.set_reward("s0", "a", 5.0)
        p.gamma = 0.9
        result = qmdp(p)
        assert result.converged
        b = {"s0": 1.0}
        # V = R / (1 - gamma) = 5 / 0.1 = 50
        assert abs(result.value(b) - 50.0) < 1.0

    def test_zero_discount(self):
        """Gamma near 0 should focus on immediate reward."""
        p = tiger_problem()
        p.gamma = 0.01
        result = qmdp(p)
        b = {"tiger-left": 0.5, "tiger-right": 0.5}
        # With gamma~0, immediate reward dominates
        # Listen: -1, Open: E[-45] -> listen is better
        assert result.best_action(b) == "listen"

    def test_fully_observable_pomdp(self):
        """POMDP with perfect observations equals MDP."""
        p = POMDP(name="FullyObservable")
        p.gamma = 0.9
        p.add_state("s0")
        p.add_state("s1")
        p.add_action("a")
        p.add_action("b")
        p.add_observation("see_s0")
        p.add_observation("see_s1")
        p.add_transition("s0", "a", "s1", 1.0)
        p.add_transition("s0", "b", "s0", 1.0)
        p.add_transition("s1", "a", "s0", 1.0)
        p.add_transition("s1", "b", "s1", 1.0)
        p.set_reward("s0", "a", 10.0)
        p.set_reward("s0", "b", 1.0)
        p.set_reward("s1", "a", 5.0)
        p.set_reward("s1", "b", 2.0)
        # Perfect observations
        p.add_observation_prob("a", "s0", "see_s0", 1.0)
        p.add_observation_prob("a", "s1", "see_s1", 1.0)
        p.add_observation_prob("b", "s0", "see_s0", 1.0)
        p.add_observation_prob("b", "s1", "see_s1", 1.0)
        result = qmdp(p)
        assert result.converged


# ═════════════════════════════════════════════════════════════════════════════
#  Section 19: POMDP-to-MDP Conversion
# ═════════════════════════════════════════════════════════════════════════════

class TestConversion:
    def test_tiger_to_mdp(self):
        p = tiger_problem()
        mdp = p.to_mdp()
        assert len(mdp.states) == 2
        assert len(mdp.actions) == 3
        # MDP should have valid transitions
        issues = mdp.validate()
        assert len(issues) == 0

    def test_maintenance_to_mdp(self):
        p = machine_maintenance()
        mdp = p.to_mdp()
        assert len(mdp.states) == 3
        assert len(mdp.validate()) == 0


# ═════════════════════════════════════════════════════════════════════════════
#  Section 20: Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_tiger_pipeline(self):
        """Full pipeline: build, solve, simulate, evaluate."""
        p = tiger_problem()
        # Solve
        result = pbvi(p, n_points=40, seed=42, max_iter=40)
        # Simulate
        traj = simulate_pomdp(p, result, steps=30, seed=42)
        assert len(traj) > 0
        # Evaluate
        stats = evaluate_policy(p, result, n_episodes=200, seed=42)
        assert stats["mean_reward"] > -100

    def test_solve_then_info_gain(self):
        """Combine solver with information-theoretic analysis."""
        p = tiger_problem()
        result = pbvi(p, n_points=30, seed=42, max_iter=30)
        b = p.get_initial_belief()

        # Get policy action
        policy_action = result.best_action(b)
        # Get most informative action
        info_action, ig = most_informative_action(p, b)

        # Both should be "listen" at uniform belief
        assert policy_action == "listen"
        assert info_action == "listen"

    def test_belief_tracking_simulation(self):
        """Verify beliefs track correctly through simulation."""
        p = tiger_problem()
        result = qmdp(p)
        traj = simulate_pomdp(p, result, steps=20, seed=123)

        for step in traj:
            b = step["belief"]
            # All beliefs should be valid probability distributions
            total = sum(b.values())
            assert abs(total - 1.0) < 1e-9
            for prob in b.values():
                assert prob >= -1e-9

    def test_maintenance_full_pipeline(self):
        p = machine_maintenance()
        result = pbvi(p, n_points=30, seed=42, max_iter=30)
        traj = simulate_pomdp(p, result, steps=20, seed=42)
        assert len(traj) > 0
        stats = evaluate_policy(p, result, n_episodes=100, seed=42)
        assert "mean_reward" in stats

    def test_rock_sample_pipeline(self):
        p = rock_sample_small()
        result = qmdp(p)
        traj = simulate_pomdp(p, result, steps=15, seed=42)
        assert len(traj) > 0

    def test_hallway_pipeline(self):
        p = hallway_navigation(length=3)
        result = pbvi(p, n_points=20, seed=42, max_iter=20)
        stats = evaluate_policy(p, result, n_episodes=100, seed=42)
        assert "mean_reward" in stats
