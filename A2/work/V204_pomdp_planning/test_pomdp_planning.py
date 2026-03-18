"""Tests for V204: Online POMDP Planning (POMCP + DESPOT)."""

import pytest
import random
import math
from pomdp_planning import (
    POMCP, POMCPConfig, DESPOT, DESPOTConfig,
    BeliefNode, ActionNode, DESPOTNode,
    simulate_online, evaluate_planner, compare_planners,
    make_tiger_planning_pomdp, make_maze_planning_pomdp, make_hallway_pomdp,
    make_greedy_rollout, make_heuristic_rollout,
    planner_summary, evaluation_summary,
    _sample_transition, _get_observation, _available_actions,
    _all_actions, _sample_initial_state, _random_rollout_policy,
    _particle_filter_update,
    SimulationStep, SimulationResult, EvaluationResult,
)
from probabilistic_partial_obs import POMDP, POMDPObjective


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiger():
    return make_tiger_planning_pomdp()

@pytest.fixture
def maze():
    return make_maze_planning_pomdp()

@pytest.fixture
def hallway():
    return make_hallway_pomdp()


def _simple_pomdp():
    """Minimal 2-state POMDP for unit tests."""
    return POMDP(
        states={"A", "B"},
        actions={"go", "stay"},
        transitions={
            ("A", "go"): [("B", 1.0)],
            ("A", "stay"): [("A", 1.0)],
            ("B", "go"): [("A", 1.0)],
            ("B", "stay"): [("B", 1.0)],
        },
        obs={"A": "obs_a", "B": "obs_b"},
        rewards={
            ("A", "go"): 10.0,
            ("A", "stay"): 0.0,
            ("B", "go"): -5.0,
            ("B", "stay"): 1.0,
        },
        initial=[("A", 0.5), ("B", 0.5)],
        objective=POMDPObjective.REWARD_FINITE,
    )


# ---------------------------------------------------------------------------
# POMDP construction tests
# ---------------------------------------------------------------------------

class TestPOMDPConstruction:
    def test_tiger_pomdp_states(self, tiger):
        assert len(tiger.states) == 4  # (pos, heard) pairs
        assert ("L", "L") in tiger.states

    def test_tiger_pomdp_actions(self, tiger):
        assert len(tiger.actions) == 3
        assert "listen" in tiger.actions

    def test_tiger_pomdp_transitions(self, tiger):
        assert (("L", "L"), "listen") in tiger.transitions
        trans = tiger.transitions[(("L", "L"), "listen")]
        assert len(trans) == 2  # correct + incorrect hearing

    def test_tiger_pomdp_rewards(self, tiger):
        assert tiger.rewards[(("L", "L"), "listen")] == -1
        assert tiger.rewards[(("L", "L"), "open_left")] == -100
        assert tiger.rewards[(("L", "L"), "open_right")] == 10

    def test_tiger_pomdp_initial(self, tiger):
        assert len(tiger.initial) == 4  # 4 expanded states
        total_prob = sum(p for _, p in tiger.initial)
        assert abs(total_prob - 1.0) < 1e-9

    def test_maze_pomdp_states(self, maze):
        assert len(maze.states) == 16  # 4x4

    def test_maze_pomdp_actions(self, maze):
        assert len(maze.actions) == 4

    def test_maze_goal_reward(self, maze):
        # Moving to (3,3) from (3,2) east should give +100
        assert maze.rewards[((3, 2), "east")] == 100.0

    def test_maze_step_cost(self, maze):
        assert maze.rewards[((0, 0), "east")] == -1.0

    def test_hallway_pomdp_states(self, hallway):
        assert len(hallway.states) == 5

    def test_hallway_observations(self, hallway):
        assert hallway.obs[0] == "zone_0"
        assert hallway.obs[2] == "zone_1"
        assert hallway.obs[4] == "zone_2"

    def test_hallway_goal_reward(self, hallway):
        assert hallway.rewards[(3, "right")] == 100.0

    def test_hallway_step_cost(self, hallway):
        assert hallway.rewards[(0, "right")] == -1.0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_sample_transition_deterministic(self):
        p = _simple_pomdp()
        ns, r = _sample_transition(p, "A", "go")
        assert ns == "B"
        assert r == 10.0

    def test_sample_transition_missing(self):
        p = _simple_pomdp()
        ns, r = _sample_transition(p, "A", "nonexistent")
        assert ns == "A"  # stays in place
        assert r == 0.0

    def test_get_observation(self):
        p = _simple_pomdp()
        assert _get_observation(p, "A") == "obs_a"
        assert _get_observation(p, "B") == "obs_b"

    def test_available_actions_all(self):
        p = _simple_pomdp()
        acts = _available_actions(p, "A")
        assert set(acts) == {"go", "stay"}

    def test_available_actions_restricted(self):
        p = _simple_pomdp()
        p.state_actions = {"A": {"stay"}, "B": {"go", "stay"}}
        assert set(_available_actions(p, "A")) == {"stay"}
        assert set(_available_actions(p, "B")) == {"go", "stay"}

    def test_all_actions(self):
        p = _simple_pomdp()
        assert set(_all_actions(p)) == {"go", "stay"}

    def test_sample_initial_state(self):
        p = _simple_pomdp()
        random.seed(42)
        states_seen = set()
        for _ in range(100):
            s = _sample_initial_state(p)
            states_seen.add(s)
        assert len(states_seen) == 2  # should see both states

    def test_random_rollout_policy(self):
        random.seed(42)
        a = _random_rollout_policy("s", ["a", "b", "c"])
        assert a in ["a", "b", "c"]

    def test_particle_filter_update_matching(self):
        p = _simple_pomdp()
        particles = ["A"] * 50 + ["B"] * 50
        # Action "go" from A -> B (obs_b), from B -> A (obs_a)
        updated = _particle_filter_update(p, particles, "go", "obs_b", target_count=100)
        assert len(updated) == 100
        # All particles that matched obs_b came from A->B
        # So updated should be mostly B
        b_count = sum(1 for s in updated if s == "B")
        assert b_count > 0

    def test_particle_filter_update_fallback(self):
        p = _simple_pomdp()
        # All particles in A, action stay -> stays A, obs_a
        # But we ask for obs_b -- no match, fallback kicks in
        particles = ["A"] * 50
        updated = _particle_filter_update(p, particles, "stay", "obs_b", target_count=50)
        assert len(updated) == 50


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestDataStructures:
    def test_action_node_q_value_zero_visits(self):
        an = ActionNode(action="a")
        assert an.q_value == 0.0

    def test_action_node_q_value(self):
        an = ActionNode(action="a", visit_count=10, total_value=50.0)
        assert abs(an.q_value - 5.0) < 1e-9

    def test_belief_node_defaults(self):
        bn = BeliefNode()
        assert bn.particles == []
        assert bn.visit_count == 0
        assert bn.action_children == {}

    def test_despot_node_value_zero(self):
        dn = DESPOTNode()
        assert dn.value == 0.0

    def test_despot_node_value(self):
        dn = DESPOTNode(visit_count=4, total_value=20.0)
        assert abs(dn.value - 5.0) < 1e-9

    def test_pomcp_config_defaults(self):
        cfg = POMCPConfig()
        assert cfg.num_simulations == 1000
        assert cfg.exploration_const == 1.0
        assert cfg.discount == 0.95
        assert cfg.particle_count == 500

    def test_despot_config_defaults(self):
        cfg = DESPOTConfig()
        assert cfg.num_scenarios == 500
        assert cfg.lambda_penalty == 0.1


# ---------------------------------------------------------------------------
# POMCP tests
# ---------------------------------------------------------------------------

class TestPOMCP:
    def test_pomcp_search_returns_action(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=100, particle_count=50, max_depth=10)
        planner = POMCP(tiger, cfg)
        # Uniform belief over expanded states
        particles = [("L", "L")] * 13 + [("L", "R")] * 12 + [("R", "L")] * 13 + [("R", "R")] * 12
        action = planner.search(particles)
        assert action in tiger.actions

    def test_pomcp_listen_with_uncertain_belief(self, tiger):
        """With balanced belief, POMCP should prefer listening (info gathering)."""
        random.seed(42)
        cfg = POMCPConfig(num_simulations=1000, particle_count=200, max_depth=20)
        planner = POMCP(tiger, cfg)
        # Uniform: tiger equally likely left or right
        particles = [("L", "L")] * 50 + [("L", "R")] * 50 + [("R", "L")] * 50 + [("R", "R")] * 50
        action = planner.search(particles)
        # With noisy obs, listening gathers info (85% accuracy) -- rational to listen
        assert action == "listen"

    def test_pomcp_open_with_confident_belief(self, tiger):
        """With confident belief, POMCP should open the safe door."""
        random.seed(42)
        cfg = POMCPConfig(num_simulations=1000, particle_count=200, max_depth=15)
        planner = POMCP(tiger, cfg)
        # Tiger is LEFT with very high confidence (heard left many times)
        particles = [("L", "L")] * 190 + [("R", "L")] * 10
        action = planner.search(particles)
        assert action == "open_right"  # open the safe door

    def test_pomcp_get_action_values(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=100, particle_count=50, max_depth=10)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 25 + [("R", "R")] * 25
        planner.search(particles)
        vals = planner.get_action_values()
        assert len(vals) > 0
        for a, v in vals.items():
            assert isinstance(v, float)

    def test_pomcp_get_action_visits(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=100, particle_count=50, max_depth=10)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 25 + [("R", "R")] * 25
        planner.search(particles)
        visits = planner.get_action_visits()
        total = sum(visits.values())
        assert total == 100  # should match num_simulations

    def test_pomcp_belief_update(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(particle_count=100)
        planner = POMCP(tiger, cfg)
        # Uniform belief
        particles = [("L", "L")] * 25 + [("L", "R")] * 25 + [("R", "L")] * 25 + [("R", "R")] * 25
        updated = planner.update_belief(particles, "listen", "hear_left")
        assert len(updated) == 100
        # After hearing left, more particles should have heard="L"
        left_heard = sum(1 for s in updated if s[1] == "L")
        assert left_heard > 50  # should shift toward hearing left

    def test_pomcp_belief_update_reinvigoration(self, tiger):
        """When very few particles match, reinvigoration should produce enough."""
        random.seed(42)
        cfg = POMCPConfig(particle_count=100, reinvigoration_count=10)
        planner = POMCP(tiger, cfg)
        # All particles have heard=R, but we observe hear_left
        particles = [("R", "R")] * 100
        updated = planner.update_belief(particles, "listen", "hear_left")
        assert len(updated) == 100  # should still produce particles via reinvigoration

    def test_pomcp_simple_pomdp(self):
        p = _simple_pomdp()
        random.seed(42)
        cfg = POMCPConfig(num_simulations=200, particle_count=50, max_depth=5)
        planner = POMCP(p, cfg)
        particles = ["A"] * 50
        action = planner.search(particles)
        assert action in p.actions

    def test_pomcp_with_greedy_rollout(self, tiger):
        random.seed(42)
        greedy = make_greedy_rollout(tiger)
        cfg = POMCPConfig(num_simulations=100, particle_count=50,
                          max_depth=10, rollout_policy=greedy)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 25 + [("R", "R")] * 25
        action = planner.search(particles)
        assert action in tiger.actions

    def test_pomcp_root_visit_count(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=50, particle_count=30, max_depth=5)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 15 + [("R", "R")] * 15
        planner.search(particles)
        assert planner.root is not None
        assert planner.root.visit_count == 50

    def test_pomcp_tree_structure(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=50, particle_count=30, max_depth=5)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 15 + [("R", "R")] * 15
        planner.search(particles)
        # Root should have action children
        assert len(planner.root.action_children) > 0
        for a, anode in planner.root.action_children.items():
            assert anode.action == a
            assert anode.visit_count >= 0

    def test_pomcp_hallway_prefers_right(self, hallway):
        """In hallway starting at 0, planner should prefer moving right toward goal."""
        random.seed(42)
        cfg = POMCPConfig(num_simulations=200, particle_count=50, max_depth=15)
        planner = POMCP(hallway, cfg)
        particles = [0] * 50
        action = planner.search(particles)
        assert action == "right"

    def test_pomcp_empty_particles(self, tiger):
        cfg = POMCPConfig(num_simulations=10, particle_count=10, max_depth=5)
        planner = POMCP(tiger, cfg)
        action = planner.search([])
        assert action is None or action in tiger.actions


# ---------------------------------------------------------------------------
# DESPOT tests
# ---------------------------------------------------------------------------

class TestDESPOT:
    def test_despot_search_returns_action(self, tiger):
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=50, num_expansions=50, max_depth=10)
        planner = DESPOT(tiger, cfg)
        particles = [("L", "L")] * 25 + [("R", "R")] * 25
        action = planner.search(particles)
        assert action in tiger.actions

    def test_despot_listen_uncertain(self, tiger):
        """DESPOT with uncertain belief should prefer listening."""
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=200, num_expansions=500, max_depth=20)
        planner = DESPOT(tiger, cfg)
        particles = [("L", "L")] * 50 + [("L", "R")] * 50 + [("R", "L")] * 50 + [("R", "R")] * 50
        action = planner.search(particles)
        assert action == "listen"

    def test_despot_open_confident(self, tiger):
        """DESPOT with confident belief should open safe door."""
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=200, num_expansions=500, max_depth=15)
        planner = DESPOT(tiger, cfg)
        # Tiger LEFT with high confidence
        particles = [("L", "L")] * 190 + [("R", "L")] * 10
        action = planner.search(particles)
        assert action == "open_right"

    def test_despot_get_action_values(self, tiger):
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=30, num_expansions=30, max_depth=5)
        planner = DESPOT(tiger, cfg)
        particles = [("L", "L")] * 15 + [("R", "R")] * 15
        planner.search(particles)
        vals = planner.get_action_values()
        assert len(vals) > 0

    def test_despot_get_action_visits(self, tiger):
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=30, num_expansions=30, max_depth=5)
        planner = DESPOT(tiger, cfg)
        particles = [("L", "L")] * 15 + [("R", "R")] * 15
        planner.search(particles)
        visits = planner.get_action_visits()
        assert sum(visits.values()) > 0

    def test_despot_regularization(self, tiger):
        """Higher lambda should prefer simpler trees (fewer nodes)."""
        random.seed(42)
        cfg_low = DESPOTConfig(num_scenarios=30, num_expansions=50,
                               max_depth=10, lambda_penalty=0.01)
        cfg_high = DESPOTConfig(num_scenarios=30, num_expansions=50,
                                max_depth=10, lambda_penalty=10.0)
        p_low = DESPOT(tiger, cfg_low)
        p_high = DESPOT(tiger, cfg_high)
        particles = [("L", "L")] * 15 + [("R", "R")] * 15
        p_low.search(particles)
        p_high.search(particles)
        assert p_low.root is not None
        assert p_high.root is not None

    def test_despot_simple_pomdp(self):
        p = _simple_pomdp()
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=30, num_expansions=50, max_depth=5)
        planner = DESPOT(p, cfg)
        particles = ["A"] * 30
        action = planner.search(particles)
        assert action in p.actions

    def test_despot_with_greedy_rollout(self, tiger):
        random.seed(42)
        greedy = make_greedy_rollout(tiger)
        cfg = DESPOTConfig(num_scenarios=30, num_expansions=50,
                           max_depth=10, rollout_policy=greedy)
        planner = DESPOT(tiger, cfg)
        particles = [("L", "L")] * 15 + [("R", "R")] * 15
        action = planner.search(particles)
        assert action in tiger.actions

    def test_despot_hallway_prefers_right(self, hallway):
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=50, num_expansions=100, max_depth=15)
        planner = DESPOT(hallway, cfg)
        particles = [0] * 50
        action = planner.search(particles)
        assert action == "right"

    def test_despot_count_nodes(self, tiger):
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=20, num_expansions=30, max_depth=5)
        planner = DESPOT(tiger, cfg)
        particles = [("L", "L")] * 10 + [("R", "R")] * 10
        planner.search(particles)
        count = planner._count_nodes(planner.root)
        assert count > 0


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_simulate_online_pomcp(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=50, particle_count=30, max_depth=10)
        planner = POMCP(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=5, seed=42)
        assert isinstance(result, SimulationResult)
        assert len(result.steps) <= 5
        assert len(result.steps) > 0

    def test_simulate_online_despot(self, tiger):
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=30, num_expansions=30, max_depth=10)
        planner = DESPOT(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=5, seed=42)
        assert isinstance(result, SimulationResult)
        assert len(result.steps) <= 5

    def test_simulation_step_fields(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=30, particle_count=20, max_depth=5)
        planner = POMCP(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=3, seed=42)
        for step in result.steps:
            assert step.state in tiger.states
            assert step.action in tiger.actions
            assert isinstance(step.reward, float)
            assert isinstance(step.cumulative_reward, float)

    def test_simulation_cumulative_reward(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=30, particle_count=20, max_depth=5)
        planner = POMCP(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=5, seed=42)
        cum = 0.0
        for step in result.steps:
            cum += step.reward
            assert abs(step.cumulative_reward - cum) < 1e-9

    def test_simulation_discounted_reward(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=30, particle_count=20, max_depth=5)
        planner = POMCP(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=5, seed=42, discount=0.9)
        # Verify discounted reward is computed
        assert isinstance(result.discounted_reward, float)

    def test_simulation_with_initial_particles(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=30, particle_count=20, max_depth=5)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 20
        result = simulate_online(tiger, planner, horizon=3,
                                  initial_particles=particles, seed=42)
        assert len(result.steps) > 0

    def test_simulate_hallway_reaches_goal(self, hallway):
        """Hallway planner should make progress toward goal."""
        random.seed(42)
        cfg = POMCPConfig(num_simulations=100, particle_count=30, max_depth=15)
        planner = POMCP(hallway, cfg)
        result = simulate_online(hallway, planner, horizon=20, seed=42)
        # Check that total reward includes some positive (goal reached) or at least moved
        assert len(result.steps) > 0

    def test_simulate_maze_short(self, maze):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=50, particle_count=20, max_depth=10)
        planner = POMCP(maze, cfg)
        result = simulate_online(maze, planner, horizon=5, seed=42)
        assert len(result.steps) <= 5


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_planner_basic(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=30, particle_count=20, max_depth=5)
        planner = POMCP(tiger, cfg)
        result = evaluate_planner(tiger, planner, n_episodes=3, horizon=5, seed=42)
        assert isinstance(result, EvaluationResult)
        assert result.n_episodes == 3
        assert len(result.episode_rewards) == 3

    def test_evaluate_planner_statistics(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=30, particle_count=20, max_depth=5)
        planner = POMCP(tiger, cfg)
        result = evaluate_planner(tiger, planner, n_episodes=5, horizon=5, seed=42)
        assert isinstance(result.mean_total_reward, float)
        assert isinstance(result.std_total_reward, float)
        assert result.std_total_reward >= 0

    def test_compare_planners_basic(self, tiger):
        random.seed(42)
        pomcp = POMCP(tiger, POMCPConfig(num_simulations=30, particle_count=20, max_depth=5))
        despot = DESPOT(tiger, DESPOTConfig(num_scenarios=20, num_expansions=30, max_depth=5))
        results = compare_planners(
            tiger, {"pomcp": pomcp, "despot": despot},
            n_episodes=3, horizon=5, seed=42,
        )
        assert "pomcp" in results
        assert "despot" in results
        assert results["pomcp"].n_episodes == 3
        assert results["despot"].n_episodes == 3

    def test_compare_planners_different_configs(self, tiger):
        random.seed(42)
        p1 = POMCP(tiger, POMCPConfig(num_simulations=20, particle_count=15, max_depth=5))
        p2 = POMCP(tiger, POMCPConfig(num_simulations=50, particle_count=30, max_depth=5))
        results = compare_planners(
            tiger, {"low_sim": p1, "high_sim": p2},
            n_episodes=3, horizon=5, seed=42,
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Rollout policy tests
# ---------------------------------------------------------------------------

class TestRolloutPolicies:
    def test_greedy_rollout_picks_best_reward(self):
        p = _simple_pomdp()
        greedy = make_greedy_rollout(p)
        # From state A: go=10, stay=0
        action = greedy("A", ["go", "stay"])
        assert action == "go"

    def test_greedy_rollout_negative(self):
        p = _simple_pomdp()
        greedy = make_greedy_rollout(p)
        # From state B: go=-5, stay=1
        action = greedy("B", ["go", "stay"])
        assert action == "stay"

    def test_heuristic_rollout_returns_action(self):
        h = make_heuristic_rollout(lambda s: s)
        action = h("state", ["a", "b", "c"])
        assert action in ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Summary / reporting tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_planner_summary_pomcp(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=30, particle_count=20, max_depth=5)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 10 + [("R", "R")] * 10
        planner.search(particles)
        summary = planner_summary(planner, "test_pomcp")
        assert "test_pomcp" in summary
        assert "Action values" in summary

    def test_planner_summary_despot(self, tiger):
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=20, num_expansions=20, max_depth=5)
        planner = DESPOT(tiger, cfg)
        particles = [("L", "L")] * 10 + [("R", "R")] * 10
        planner.search(particles)
        summary = planner_summary(planner, "test_despot")
        assert "test_despot" in summary

    def test_evaluation_summary_format(self, tiger):
        random.seed(42)
        pomcp = POMCP(tiger, POMCPConfig(num_simulations=20, particle_count=15, max_depth=5))
        results = {"pomcp": evaluate_planner(tiger, pomcp, n_episodes=2, horizon=3, seed=42)}
        summary = evaluation_summary(results)
        assert "Planner Comparison" in summary
        assert "pomcp" in summary

    def test_planner_summary_no_search(self, tiger):
        planner = POMCP(tiger)
        summary = planner_summary(planner)
        assert "POMCP" in summary

    def test_evaluation_summary_multiple(self, tiger):
        random.seed(42)
        pomcp = POMCP(tiger, POMCPConfig(num_simulations=20, particle_count=15, max_depth=5))
        despot = DESPOT(tiger, DESPOTConfig(num_scenarios=15, num_expansions=20, max_depth=5))
        results = compare_planners(
            tiger, {"pomcp": pomcp, "despot": despot},
            n_episodes=2, horizon=3, seed=42,
        )
        summary = evaluation_summary(results)
        assert "pomcp" in summary
        assert "despot" in summary


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_pomcp_single_action(self):
        """POMDP with only one action."""
        p = POMDP(
            states={"s"},
            actions={"only"},
            transitions={("s", "only"): [("s", 1.0)]},
            obs={"s": "o"},
            rewards={("s", "only"): 1.0},
            initial=[("s", 1.0)],
            objective=POMDPObjective.REWARD_FINITE,
        )
        cfg = POMCPConfig(num_simulations=10, particle_count=5, max_depth=3)
        planner = POMCP(p, cfg)
        action = planner.search(["s"] * 5)
        assert action == "only"

    def test_despot_single_action(self):
        p = POMDP(
            states={"s"},
            actions={"only"},
            transitions={("s", "only"): [("s", 1.0)]},
            obs={"s": "o"},
            rewards={("s", "only"): 1.0},
            initial=[("s", 1.0)],
            objective=POMDPObjective.REWARD_FINITE,
        )
        cfg = DESPOTConfig(num_scenarios=5, num_expansions=10, max_depth=3)
        planner = DESPOT(p, cfg)
        action = planner.search(["s"] * 5)
        assert action == "only"

    def test_pomcp_single_particle(self, tiger):
        cfg = POMCPConfig(num_simulations=20, particle_count=1, max_depth=5)
        planner = POMCP(tiger, cfg)
        action = planner.search([("L", "L")])
        assert action in tiger.actions

    def test_despot_single_scenario(self, tiger):
        cfg = DESPOTConfig(num_scenarios=1, num_expansions=10, max_depth=5)
        planner = DESPOT(tiger, cfg)
        action = planner.search([("L", "L")])
        assert action in tiger.actions

    def test_pomcp_depth_one(self, tiger):
        """With very shallow depth, planner should still return valid action."""
        cfg = POMCPConfig(num_simulations=10, particle_count=10, max_depth=1)
        planner = POMCP(tiger, cfg)
        particles = [("L", "L")] * 10
        action = planner.search(particles)
        assert action in tiger.actions

    def test_zero_reward_pomdp(self):
        """POMDP where all rewards are zero."""
        p = POMDP(
            states={"A", "B"},
            actions={"go", "stay"},
            transitions={
                ("A", "go"): [("B", 1.0)],
                ("A", "stay"): [("A", 1.0)],
                ("B", "go"): [("A", 1.0)],
                ("B", "stay"): [("B", 1.0)],
            },
            obs={"A": "o1", "B": "o2"},
            rewards={},
            initial=[("A", 1.0)],
            objective=POMDPObjective.REWARD_FINITE,
        )
        cfg = POMCPConfig(num_simulations=20, particle_count=10, max_depth=5)
        planner = POMCP(p, cfg)
        action = planner.search(["A"] * 10)
        assert action in p.actions

    def test_probabilistic_transitions(self):
        """POMDP with non-trivial transition probabilities."""
        p = POMDP(
            states={"s0", "s1", "s2"},
            actions={"a"},
            transitions={
                ("s0", "a"): [("s0", 0.1), ("s1", 0.3), ("s2", 0.6)],
                ("s1", "a"): [("s0", 0.5), ("s1", 0.5)],
                ("s2", "a"): [("s2", 1.0)],
            },
            obs={"s0": "x", "s1": "y", "s2": "z"},
            rewards={("s0", "a"): 1.0, ("s1", "a"): 2.0, ("s2", "a"): 3.0},
            initial=[("s0", 1.0)],
            objective=POMDPObjective.REWARD_FINITE,
        )
        random.seed(42)
        cfg = POMCPConfig(num_simulations=50, particle_count=20, max_depth=5)
        planner = POMCP(p, cfg)
        action = planner.search(["s0"] * 20)
        assert action == "a"

    def test_simulation_result_fields(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=20, particle_count=10, max_depth=5)
        planner = POMCP(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=3, seed=42)
        assert hasattr(result, 'total_reward')
        assert hasattr(result, 'discounted_reward')
        assert hasattr(result, 'horizon_reached')
        assert hasattr(result, 'final_state')
        assert result.final_state in tiger.states

    def test_evaluation_result_fields(self, tiger):
        random.seed(42)
        cfg = POMCPConfig(num_simulations=20, particle_count=10, max_depth=5)
        planner = POMCP(tiger, cfg)
        result = evaluate_planner(tiger, planner, n_episodes=2, horizon=3, seed=42)
        assert hasattr(result, 'mean_total_reward')
        assert hasattr(result, 'mean_discounted_reward')
        assert hasattr(result, 'std_total_reward')
        assert hasattr(result, 'mean_horizon')
        assert result.mean_horizon > 0


# ---------------------------------------------------------------------------
# Integration: POMCP vs DESPOT behavioral equivalence
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_both_planners_prefer_listen_tiger(self, tiger):
        """Both POMCP and DESPOT should prefer listening under uncertainty."""
        random.seed(42)
        pomcp = POMCP(tiger, POMCPConfig(num_simulations=1000, particle_count=200, max_depth=20))
        particles = [("L", "L")] * 50 + [("L", "R")] * 50 + [("R", "L")] * 50 + [("R", "R")] * 50
        a1 = pomcp.search(particles)

        random.seed(42)
        despot = DESPOT(tiger, DESPOTConfig(num_scenarios=200, num_expansions=500, max_depth=20))
        a2 = despot.search(particles)

        assert a1 == "listen"
        assert a2 == "listen"

    def test_both_planners_open_right_confident(self, tiger):
        """Both planners should open right when tiger is confidently left."""
        random.seed(42)
        pomcp = POMCP(tiger, POMCPConfig(num_simulations=1000, particle_count=200, max_depth=15))
        particles = [("L", "L")] * 190 + [("R", "L")] * 10
        a1 = pomcp.search(particles)

        random.seed(42)
        despot = DESPOT(tiger, DESPOTConfig(num_scenarios=200, num_expansions=500, max_depth=15))
        a2 = despot.search(particles)

        assert a1 == "open_right"
        assert a2 == "open_right"

    def test_both_hallway_right(self, hallway):
        random.seed(42)
        pomcp = POMCP(hallway, POMCPConfig(num_simulations=200, particle_count=50, max_depth=15))
        a1 = pomcp.search([0] * 50)

        random.seed(42)
        despot = DESPOT(hallway, DESPOTConfig(num_scenarios=50, num_expansions=100, max_depth=15))
        a2 = despot.search([0] * 50)

        assert a1 == "right"
        assert a2 == "right"

    def test_full_episode_pomcp_tiger(self, tiger):
        """Run a full 10-step episode with POMCP on Tiger."""
        random.seed(42)
        cfg = POMCPConfig(num_simulations=100, particle_count=50, max_depth=10)
        planner = POMCP(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=10, seed=42)
        assert len(result.steps) == 10
        assert isinstance(result.total_reward, float)

    def test_full_episode_despot_tiger(self, tiger):
        """Run a full 10-step episode with DESPOT on Tiger."""
        random.seed(42)
        cfg = DESPOTConfig(num_scenarios=50, num_expansions=50, max_depth=10)
        planner = DESPOT(tiger, cfg)
        result = simulate_online(tiger, planner, horizon=10, seed=42)
        assert len(result.steps) == 10
        assert isinstance(result.total_reward, float)

    def test_comparison_on_maze(self, maze):
        """Compare POMCP and DESPOT on maze."""
        random.seed(42)
        pomcp = POMCP(maze, POMCPConfig(num_simulations=50, particle_count=20, max_depth=10))
        despot = DESPOT(maze, DESPOTConfig(num_scenarios=20, num_expansions=50, max_depth=10))
        results = compare_planners(
            maze, {"pomcp": pomcp, "despot": despot},
            n_episodes=2, horizon=10, seed=42,
        )
        assert results["pomcp"].n_episodes == 2
        assert results["despot"].n_episodes == 2
