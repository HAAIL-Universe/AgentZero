"""Tests for V220: Decentralized POMDPs."""

import pytest
import math
from dec_pomdp import (
    DecPOMDP, LocalPolicy, JointPolicy, DecPOMDPResult,
    decentralized_tiger, cooperative_box_pushing, multi_agent_meeting,
    communication_channel,
    evaluate_joint_policy, exhaustive_dp, jesp, cpde,
    occupancy_state, information_loss, simulate,
    compare_solvers, dec_pomdp_summary,
)


# ========================================================================
# Core data structure tests
# ========================================================================

class TestDecPOMDPConstruction:
    """Test Dec-POMDP model construction."""

    def test_empty_model(self):
        dec = DecPOMDP(name="empty")
        assert dec.name == "empty"
        assert dec.agents == []
        assert dec.states == []

    def test_add_agents(self):
        dec = DecPOMDP()
        dec.add_agent("alice")
        dec.add_agent("bob")
        assert dec.agents == ["alice", "bob"]

    def test_add_agent_idempotent(self):
        dec = DecPOMDP()
        dec.add_agent("alice")
        dec.add_agent("alice")
        assert dec.agents == ["alice"]

    def test_add_states(self):
        dec = DecPOMDP()
        dec.add_state("s0")
        dec.add_state("s1")
        assert dec.states == ["s0", "s1"]

    def test_add_state_idempotent(self):
        dec = DecPOMDP()
        dec.add_state("s0")
        dec.add_state("s0")
        assert dec.states == ["s0"]

    def test_add_actions(self):
        dec = DecPOMDP()
        dec.add_agent("a1")
        dec.add_action("a1", "left")
        dec.add_action("a1", "right")
        assert dec.actions["a1"] == ["left", "right"]

    def test_add_observations(self):
        dec = DecPOMDP()
        dec.add_agent("a1")
        dec.add_observation("a1", "o1")
        dec.add_observation("a1", "o2")
        assert dec.observations["a1"] == ["o1", "o2"]

    def test_set_transition(self):
        dec = DecPOMDP()
        dec.add_state("s0")
        dec.add_state("s1")
        dec.set_transition("s0", ("a", "b"), "s1", 0.7)
        dec.set_transition("s0", ("a", "b"), "s0", 0.3)
        trans = dec.get_transitions("s0", ("a", "b"))
        assert len(trans) == 2
        assert ("s1", 0.7) in trans
        assert ("s0", 0.3) in trans

    def test_transition_update(self):
        dec = DecPOMDP()
        dec.set_transition("s0", ("a",), "s1", 0.5)
        dec.set_transition("s0", ("a",), "s1", 0.8)  # Update
        trans = dec.get_transitions("s0", ("a",))
        assert len(trans) == 1
        assert trans[0] == ("s1", 0.8)

    def test_set_reward(self):
        dec = DecPOMDP()
        dec.set_reward("s0", ("a", "b"), 5.0)
        assert dec.get_reward("s0", ("a", "b")) == 5.0

    def test_default_reward(self):
        dec = DecPOMDP()
        assert dec.get_reward("s0", ("a",)) == 0.0

    def test_set_observation_prob(self):
        dec = DecPOMDP()
        dec.set_observation_prob("agent1", "s1", ("a", "b"), "o1", 0.9)
        assert dec.get_observation_prob("agent1", "s1", ("a", "b"), "o1") == 0.9

    def test_observation_prob_update(self):
        dec = DecPOMDP()
        dec.set_observation_prob("a", "s", ("x",), "o", 0.5)
        dec.set_observation_prob("a", "s", ("x",), "o", 0.8)
        assert dec.get_observation_prob("a", "s", ("x",), "o") == 0.8

    def test_observation_dist(self):
        dec = DecPOMDP()
        dec.set_observation_prob("a", "s", ("x",), "o1", 0.6)
        dec.set_observation_prob("a", "s", ("x",), "o2", 0.4)
        dist = dec.get_observation_dist("a", "s", ("x",))
        assert len(dist) == 2

    def test_initial_state(self):
        dec = DecPOMDP()
        dec.add_state("s0")
        dec.add_state("s1")
        dec.set_initial_state("s0", 0.3)
        dec.set_initial_state("s1", 0.7)
        b0 = dec.get_initial_belief()
        assert abs(b0["s0"] - 0.3) < 1e-10
        assert abs(b0["s1"] - 0.7) < 1e-10

    def test_initial_uniform(self):
        dec = DecPOMDP()
        dec.add_state("s0")
        dec.add_state("s1")
        b0 = dec.get_initial_belief()
        assert abs(b0["s0"] - 0.5) < 1e-10
        assert abs(b0["s1"] - 0.5) < 1e-10

    def test_joint_actions(self):
        dec = DecPOMDP()
        dec.add_agent("a1")
        dec.add_agent("a2")
        dec.add_action("a1", "x")
        dec.add_action("a1", "y")
        dec.add_action("a2", "p")
        dec.add_action("a2", "q")
        ja = dec.get_joint_actions()
        assert len(ja) == 4
        assert ("x", "p") in ja
        assert ("y", "q") in ja

    def test_joint_observations(self):
        dec = DecPOMDP()
        dec.add_agent("a1")
        dec.add_agent("a2")
        dec.add_observation("a1", "o1")
        dec.add_observation("a2", "o2")
        dec.add_observation("a2", "o3")
        jo = dec.get_joint_observations()
        assert len(jo) == 2
        assert ("o1", "o2") in jo
        assert ("o1", "o3") in jo

    def test_joint_observation_prob(self):
        dec = DecPOMDP()
        dec.add_agent("a1")
        dec.add_agent("a2")
        dec.set_observation_prob("a1", "s", ("x", "y"), "o1", 0.8)
        dec.set_observation_prob("a2", "s", ("x", "y"), "o2", 0.5)
        p = dec.get_joint_observation_prob("s", ("x", "y"), ("o1", "o2"))
        assert abs(p - 0.4) < 1e-10

    def test_joint_observation_prob_zero(self):
        dec = DecPOMDP()
        dec.add_agent("a1")
        dec.add_agent("a2")
        dec.set_observation_prob("a1", "s", ("x",), "o1", 0.0)
        dec.set_observation_prob("a2", "s", ("x",), "o2", 0.5)
        p = dec.get_joint_observation_prob("s", ("x",), ("o1", "o2"))
        assert p == 0.0

    def test_gamma(self):
        dec = DecPOMDP(gamma=0.9)
        assert dec.gamma == 0.9

    def test_horizon(self):
        dec = DecPOMDP(horizon=5)
        assert dec.horizon == 5


class TestValidation:
    """Test model validation."""

    def test_valid_model(self):
        dec = decentralized_tiger()
        issues = dec.validate()
        assert len(issues) == 0

    def test_no_agents(self):
        dec = DecPOMDP()
        dec.add_state("s")
        issues = dec.validate()
        assert any("No agents" in i for i in issues)

    def test_no_states(self):
        dec = DecPOMDP()
        dec.add_agent("a")
        dec.add_action("a", "x")
        dec.add_observation("a", "o")
        issues = dec.validate()
        assert any("No states" in i for i in issues)

    def test_missing_actions(self):
        dec = DecPOMDP()
        dec.add_agent("a")
        dec.add_state("s")
        dec.add_observation("a", "o")
        issues = dec.validate()
        assert any("no actions" in i for i in issues)

    def test_missing_observations(self):
        dec = DecPOMDP()
        dec.add_agent("a")
        dec.add_state("s")
        dec.add_action("a", "x")
        issues = dec.validate()
        assert any("no observations" in i for i in issues)

    def test_bad_transition_sum(self):
        dec = DecPOMDP()
        dec.add_agent("a")
        dec.add_state("s0")
        dec.add_state("s1")
        dec.add_action("a", "x")
        dec.add_observation("a", "o")
        dec.set_transition("s0", ("x",), "s1", 0.5)
        # Sum = 0.5, not 1.0
        issues = dec.validate()
        assert any("sum to" in i for i in issues)

    def test_bad_initial_sum(self):
        dec = DecPOMDP()
        dec.add_agent("a")
        dec.add_state("s0")
        dec.add_state("s1")
        dec.add_action("a", "x")
        dec.add_observation("a", "o")
        dec.set_initial_state("s0", 0.3)
        dec.set_initial_state("s1", 0.3)
        issues = dec.validate()
        assert any("Initial belief" in i for i in issues)


# ========================================================================
# Policy tests
# ========================================================================

class TestPolicies:
    """Test local and joint policies."""

    def test_local_policy_set_get(self):
        pol = LocalPolicy(agent="a1")
        pol.set_action((), "left")
        pol.set_action(("o1",), "right")
        assert pol.get_action(()) == "left"
        assert pol.get_action(("o1",)) == "right"

    def test_local_policy_default(self):
        pol = LocalPolicy(agent="a1")
        assert pol.get_action(("unknown",)) == ""

    def test_joint_policy(self):
        jp = JointPolicy()
        p1 = LocalPolicy(agent="a1")
        p1.set_action((), "x")
        p2 = LocalPolicy(agent="a2")
        p2.set_action((), "y")
        jp.set_policy("a1", p1)
        jp.set_policy("a2", p2)
        assert jp.get_action("a1", ()) == "x"
        assert jp.get_action("a2", ()) == "y"

    def test_joint_policy_missing_agent(self):
        jp = JointPolicy()
        assert jp.get_action("unknown", ()) == ""


# ========================================================================
# Example problem tests
# ========================================================================

class TestDecTiger:
    """Test decentralized tiger problem construction."""

    def test_structure(self):
        dec = decentralized_tiger()
        assert len(dec.agents) == 2
        assert len(dec.states) == 2
        assert len(dec.actions["agent1"]) == 3
        assert len(dec.observations["agent1"]) == 2

    def test_valid(self):
        dec = decentralized_tiger()
        assert len(dec.validate()) == 0

    def test_joint_actions_count(self):
        dec = decentralized_tiger()
        assert len(dec.get_joint_actions()) == 9  # 3x3

    def test_listen_reward(self):
        dec = decentralized_tiger()
        r = dec.get_reward("tiger-left", ("listen", "listen"))
        assert r == -2.0  # 2 * listen_cost

    def test_tiger_penalty(self):
        dec = decentralized_tiger()
        r = dec.get_reward("tiger-left", ("open-left", "open-left"))
        assert r == -100.0

    def test_treasure_reward(self):
        dec = decentralized_tiger()
        r = dec.get_reward("tiger-left", ("open-right", "open-right"))
        assert r == 20.0  # Both get treasure

    def test_listen_stays(self):
        dec = decentralized_tiger()
        trans = dec.get_transitions("tiger-left", ("listen", "listen"))
        assert len(trans) == 1
        assert trans[0] == ("tiger-left", 1.0)

    def test_open_resets(self):
        dec = decentralized_tiger()
        trans = dec.get_transitions("tiger-left", ("open-left", "listen"))
        probs = dict(trans)
        assert abs(probs.get("tiger-left", 0) - 0.5) < 1e-10
        assert abs(probs.get("tiger-right", 0) - 0.5) < 1e-10

    def test_listen_observation(self):
        dec = decentralized_tiger()
        p = dec.get_observation_prob(
            "agent1", "tiger-left", ("listen", "listen"), "hear-left")
        assert abs(p - 0.85) < 1e-10

    def test_initial_belief(self):
        dec = decentralized_tiger()
        b0 = dec.get_initial_belief()
        assert abs(b0["tiger-left"] - 0.5) < 1e-10


class TestBoxPushing:
    """Test cooperative box pushing problem."""

    def test_structure(self):
        dec = cooperative_box_pushing()
        assert len(dec.agents) == 2
        assert "agent1" in dec.agents
        assert "agent2" in dec.agents

    def test_has_states(self):
        dec = cooperative_box_pushing()
        assert len(dec.states) >= 5

    def test_actions(self):
        dec = cooperative_box_pushing()
        assert len(dec.actions["agent1"]) == 4

    def test_observations(self):
        dec = cooperative_box_pushing()
        assert len(dec.observations["agent1"]) == 4

    def test_large_box_reward(self):
        dec = cooperative_box_pushing()
        # Large done gives 100
        for ja in dec.get_joint_actions():
            r = dec.get_reward("large_done", ja)
            assert r == 100.0


class TestMeeting:
    """Test multi-agent meeting problem."""

    def test_structure(self):
        dec = multi_agent_meeting(grid_size=3)
        assert len(dec.agents) == 2
        assert len(dec.states) == 9  # 3x3

    def test_meeting_reward(self):
        dec = multi_agent_meeting(grid_size=3)
        # Co-located states should have positive reward
        r = dec.get_reward("1_1", ("stay", "stay"))
        assert r > 0

    def test_non_meeting_cost(self):
        dec = multi_agent_meeting(grid_size=3)
        r = dec.get_reward("0_2", ("stay", "stay"))
        assert r < 0


class TestCommunication:
    """Test communication channel problem."""

    def test_structure(self):
        dec = communication_channel()
        assert len(dec.agents) == 2
        assert "sender" in dec.agents
        assert "receiver" in dec.agents

    def test_states(self):
        dec = communication_channel()
        assert len(dec.states) == 2

    def test_correct_match_reward(self):
        dec = communication_channel()
        r = dec.get_reward("signal-A", ("msg-A", "act-A"))
        assert r == 10.0

    def test_mismatch_penalty(self):
        dec = communication_channel()
        r = dec.get_reward("signal-A", ("msg-A", "act-B"))
        assert r == -10.0

    def test_sender_sees_signal(self):
        dec = communication_channel()
        p = dec.get_observation_prob(
            "sender", "signal-A", ("msg-A", "act-A"), "see-A")
        assert p == 1.0

    def test_receiver_hears_message(self):
        dec = communication_channel()
        p = dec.get_observation_prob(
            "receiver", "signal-A", ("msg-A", "act-A"), "heard-A")
        assert abs(p - 0.9) < 1e-10

    def test_valid(self):
        dec = communication_channel()
        issues = dec.validate()
        assert len(issues) == 0


# ========================================================================
# Evaluation tests
# ========================================================================

class TestEvaluation:
    """Test policy evaluation."""

    def test_evaluate_random_tiger(self):
        dec = decentralized_tiger()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="agent1")
        p1.set_action((), "listen")
        p2 = LocalPolicy(agent="agent2")
        p2.set_action((), "listen")
        jp.set_policy("agent1", p1)
        jp.set_policy("agent2", p2)
        val = evaluate_joint_policy(dec, jp, horizon=3, n_simulations=5000,
                                    seed=42)
        # Both always listen: -2 per step * 3 steps ~ -6
        assert -8 < val < -4

    def test_evaluate_deterministic(self):
        """A known-optimal policy should have positive value."""
        dec = communication_channel()
        jp = JointPolicy()
        # Sender: always match signal
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")  # First step, no info
        p1.set_action(("see-A",), "msg-A")
        p1.set_action(("see-B",), "msg-B")
        # Receiver: act on what heard
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")  # First step, no info
        p2.set_action(("heard-A",), "act-A")
        p2.set_action(("heard-B",), "act-B")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)
        val = evaluate_joint_policy(dec, jp, horizon=3, n_simulations=5000,
                                    seed=42)
        # Should be positive: mostly correct matches
        assert val > 0

    def test_evaluate_horizon_1(self):
        dec = communication_channel()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)
        val = evaluate_joint_policy(dec, jp, horizon=1, n_simulations=10000,
                                    seed=42)
        # 50% chance signal-A (reward 10) + 50% signal-B (reward -10) = 0
        assert abs(val) < 2.0


# ========================================================================
# Solver tests
# ========================================================================

class TestExhaustiveDP:
    """Test exhaustive dynamic programming solver."""

    def test_communication_h1(self):
        """Exhaustive should find optimal policy for comm channel at h=1."""
        dec = communication_channel()
        result = exhaustive_dp(dec, horizon=1, seed=42)
        assert result.solver == "exhaustive_dp"
        assert result.converged
        # At h=1, no observations yet -- must pick blindly
        # Best: sender msg-A, receiver act-A or msg-B/act-B both give 0 expected
        # Any fixed (msg, act) that disagrees gives 0 expected
        assert result.value is not None

    def test_communication_h2(self):
        """At h=2, policies can condition on observations."""
        dec = communication_channel()
        result = exhaustive_dp(dec, horizon=2, seed=42)
        # At h=2, sender can signal correctly, receiver can respond to observation
        # Should find a positive value
        assert result.value > -5


class TestJESP:
    """Test JESP solver."""

    def test_communication_channel(self):
        """JESP should find a good communication strategy."""
        dec = communication_channel()
        result = jesp(dec, horizon=2, max_iter=30, n_restarts=5, seed=42)
        assert result.solver == "jesp"
        assert result.value is not None
        # Should find positive value through communication
        assert result.value > -5

    def test_tiger_h1(self):
        """JESP on tiger at horizon 1."""
        dec = decentralized_tiger()
        result = jesp(dec, horizon=1, max_iter=20, n_restarts=3, seed=42)
        assert result.solver == "jesp"
        assert result.horizon == 1

    def test_convergence(self):
        dec = communication_channel()
        result = jesp(dec, horizon=1, max_iter=50, n_restarts=3, seed=42)
        assert result.iterations >= 1

    def test_jesp_improves_over_random(self):
        """JESP should do better than random."""
        dec = communication_channel()
        import random as _rng
        r = _rng.Random(42)

        # Random policy value
        from dec_pomdp import _random_joint_policy
        rp = _random_joint_policy(dec, 2, r)
        random_val = evaluate_joint_policy(dec, rp, 2, seed=42)

        # JESP value
        result = jesp(dec, horizon=2, seed=42)
        # JESP should be at least as good
        assert result.value >= random_val - 2.0  # Allow MC noise margin


class TestCPDE:
    """Test Centralized Planning Decentralized Execution."""

    def test_communication(self):
        dec = communication_channel()
        result = cpde(dec, horizon=2, seed=42)
        assert result.solver == "cpde"
        assert result.value is not None

    def test_tiger(self):
        dec = decentralized_tiger()
        result = cpde(dec, horizon=2, seed=42)
        assert result.solver == "cpde"

    def test_returns_joint_policy(self):
        dec = communication_channel()
        result = cpde(dec, horizon=2, seed=42)
        assert result.joint_policy is not None
        # Policy should have entries for both agents
        assert "sender" in result.joint_policy.policies
        assert "receiver" in result.joint_policy.policies


# ========================================================================
# Analysis tests
# ========================================================================

class TestOccupancyState:
    """Test occupancy state computation."""

    def test_initial_occupancy(self):
        """At step 0, occupancy should match initial belief."""
        dec = communication_channel()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)

        occ = occupancy_state(dec, jp, horizon=2, step=0, n_samples=10000,
                              seed=42)
        # At step 0: state distribution matches initial, empty histories
        state_probs = {}
        for (s, hists), prob in occ.items():
            state_probs[s] = state_probs.get(s, 0) + prob
        assert abs(state_probs.get("signal-A", 0) - 0.5) < 0.05
        assert abs(state_probs.get("signal-B", 0) - 0.5) < 0.05

    def test_occupancy_sums_to_one(self):
        dec = communication_channel()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)
        occ = occupancy_state(dec, jp, horizon=3, step=2, n_samples=5000,
                              seed=42)
        total = sum(occ.values())
        assert abs(total - 1.0) < 0.01


class TestInformationLoss:
    """Test information loss measurement."""

    def test_communication_sender_low_loss(self):
        """Sender sees signal directly -- low information loss."""
        dec = communication_channel()
        loss = information_loss(dec, horizon=2, seed=42)
        # Sender sees signal perfectly
        assert loss["sender"] < 0.5

    def test_communication_receiver_higher_loss(self):
        """Receiver has higher info loss than sender."""
        dec = communication_channel()
        loss = information_loss(dec, horizon=2, seed=42)
        # Receiver gets noisy messages, not direct signal
        assert loss["receiver"] >= loss["sender"] - 0.1  # Allow MC noise

    def test_returns_all_agents(self):
        dec = decentralized_tiger()
        loss = information_loss(dec, horizon=2, seed=42)
        assert "agent1" in loss
        assert "agent2" in loss


# ========================================================================
# Simulation tests
# ========================================================================

class TestSimulation:
    """Test episode simulation."""

    def test_simulate_returns_trace(self):
        dec = communication_channel()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)
        trace = simulate(dec, jp, horizon=3, seed=42)
        assert len(trace) == 3

    def test_trace_structure(self):
        dec = communication_channel()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)
        trace = simulate(dec, jp, horizon=2, seed=42)
        step = trace[0]
        assert "step" in step
        assert "state" in step
        assert "joint_action" in step
        assert "observations" in step
        assert "reward" in step
        assert "next_state" in step

    def test_simulate_tiger(self):
        dec = decentralized_tiger()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="agent1")
        p1.set_action((), "listen")
        p2 = LocalPolicy(agent="agent2")
        p2.set_action((), "listen")
        jp.set_policy("agent1", p1)
        jp.set_policy("agent2", p2)
        trace = simulate(dec, jp, horizon=5, seed=42)
        assert len(trace) == 5
        # Both always listen -> reward = -2 each step
        for step in trace:
            assert step["reward"] == -2.0

    def test_simulate_observations_match_agents(self):
        dec = communication_channel()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)
        trace = simulate(dec, jp, horizon=2, seed=42)
        for step in trace:
            if step["next_state"] is not None:
                assert "sender" in step["observations"]
                assert "receiver" in step["observations"]


# ========================================================================
# Compare solvers test
# ========================================================================

class TestCompareSolvers:
    """Test solver comparison."""

    def test_compare_on_communication(self):
        dec = communication_channel()
        results = compare_solvers(dec, horizon=1, seed=42)
        assert "jesp" in results
        assert "cpde" in results
        # Small enough for exhaustive
        assert "exhaustive" in results

    def test_all_results_have_values(self):
        dec = communication_channel()
        results = compare_solvers(dec, horizon=1, seed=42)
        for name, result in results.items():
            assert result.value is not None
            assert result.solver == name


# ========================================================================
# Summary test
# ========================================================================

class TestSummary:
    """Test dec_pomdp_summary."""

    def test_tiger_summary(self):
        dec = decentralized_tiger()
        s = dec_pomdp_summary(dec)
        assert s["name"] == "dec_tiger"
        assert s["n_states"] == 2
        assert s["agents"] == ["agent1", "agent2"]
        assert s["n_joint_actions"] == 9
        assert s["gamma"] == 0.95
        assert len(s["issues"]) == 0

    def test_comm_summary(self):
        dec = communication_channel()
        s = dec_pomdp_summary(dec)
        assert s["name"] == "communication"
        assert s["n_states"] == 2
        assert s["n_joint_actions"] == 4

    def test_meeting_summary(self):
        dec = multi_agent_meeting(grid_size=3)
        s = dec_pomdp_summary(dec)
        assert s["n_states"] == 9


# ========================================================================
# Edge case tests
# ========================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_agent_dec_pomdp(self):
        """Single agent Dec-POMDP should behave like a POMDP."""
        dec = DecPOMDP(name="single")
        dec.add_agent("solo")
        dec.add_state("s0")
        dec.add_state("s1")
        dec.add_action("solo", "a")
        dec.add_action("solo", "b")
        dec.add_observation("solo", "o0")
        dec.add_observation("solo", "o1")
        dec.set_initial_state("s0", 1.0)
        dec.set_transition("s0", ("a",), "s0", 0.5)
        dec.set_transition("s0", ("a",), "s1", 0.5)
        dec.set_transition("s0", ("b",), "s0", 1.0)
        dec.set_transition("s1", ("a",), "s1", 1.0)
        dec.set_transition("s1", ("b",), "s0", 1.0)
        dec.set_reward("s1", ("a",), 10.0)
        dec.set_reward("s1", ("b",), 5.0)
        for sp in dec.states:
            for ja in dec.get_joint_actions():
                if sp == "s0":
                    dec.set_observation_prob("solo", sp, ja, "o0", 1.0)
                else:
                    dec.set_observation_prob("solo", sp, ja, "o1", 1.0)
        assert len(dec.validate()) == 0
        result = jesp(dec, horizon=2, seed=42)
        assert result.value is not None

    def test_three_agents(self):
        """Three-agent Dec-POMDP."""
        dec = DecPOMDP(name="three_agents")
        for s in ["s0", "s1"]:
            dec.add_state(s)
        dec.set_initial_state("s0", 1.0)
        for agent in ["a1", "a2", "a3"]:
            dec.add_agent(agent)
            dec.add_action(agent, "x")
            dec.add_action(agent, "y")
            dec.add_observation(agent, "o")
        # All joint actions lead to same transitions
        for ja in dec.get_joint_actions():
            dec.set_transition("s0", ja, "s1", 0.5)
            dec.set_transition("s0", ja, "s0", 0.5)
            dec.set_transition("s1", ja, "s1", 1.0)
            dec.set_reward("s1", ja, 1.0)
            for sp in dec.states:
                for agent in dec.agents:
                    dec.set_observation_prob(agent, sp, ja, "o", 1.0)
        assert len(dec.get_joint_actions()) == 8  # 2^3
        assert len(dec.validate()) == 0

    def test_no_transition_terminates(self):
        """Simulation handles missing transitions gracefully."""
        dec = DecPOMDP(name="terminal")
        dec.add_agent("a")
        dec.add_state("s0")
        dec.add_action("a", "x")
        dec.add_observation("a", "o")
        dec.set_initial_state("s0", 1.0)
        # No transitions defined -- should terminate immediately
        jp = JointPolicy()
        p = LocalPolicy(agent="a")
        p.set_action((), "x")
        jp.set_policy("a", p)
        trace = simulate(dec, jp, horizon=5, seed=42)
        assert len(trace) == 1  # Terminates after first step

    def test_discount_factor_effect(self):
        """Different discount factors produce different total values."""
        dec1 = communication_channel(gamma=0.99)
        dec2 = communication_channel(gamma=0.5)
        jp = JointPolicy()
        p1 = LocalPolicy(agent="sender")
        p1.set_action((), "msg-A")
        p1.set_action(("see-A",), "msg-A")
        p1.set_action(("see-B",), "msg-B")
        p2 = LocalPolicy(agent="receiver")
        p2.set_action((), "act-A")
        p2.set_action(("heard-A",), "act-A")
        p2.set_action(("heard-B",), "act-B")
        jp.set_policy("sender", p1)
        jp.set_policy("receiver", p2)
        val_high = evaluate_joint_policy(dec1, jp, horizon=5,
                                         n_simulations=5000, seed=42)
        val_low = evaluate_joint_policy(dec2, jp, horizon=5,
                                        n_simulations=5000, seed=42)
        # Higher gamma weights later steps more -- values should differ
        assert abs(val_high - val_low) > 0.01

    def test_result_dataclass(self):
        result = DecPOMDPResult(
            joint_policy=JointPolicy(),
            value=5.0,
            iterations=10,
            converged=True,
            solver="test",
            horizon=3
        )
        assert result.value == 5.0
        assert result.iterations == 10
        assert result.converged
        assert result.solver == "test"
        assert result.horizon == 3


# ========================================================================
# Integration tests
# ========================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_solve_and_simulate(self):
        """Solve then simulate with the found policy."""
        dec = communication_channel()
        result = jesp(dec, horizon=2, seed=42)
        trace = simulate(dec, result.joint_policy, horizon=2, seed=42)
        assert len(trace) == 2

    def test_solve_evaluate_compare(self):
        """Solve, evaluate independently, compare with solver's value."""
        dec = communication_channel()
        result = jesp(dec, horizon=2, seed=42)
        independent_val = evaluate_joint_policy(
            dec, result.joint_policy, horizon=2, n_simulations=10000, seed=123
        )
        # Should be in rough agreement (Monte Carlo noise)
        assert abs(independent_val - result.value) < 5.0

    def test_information_loss_matches_observations(self):
        """Agents with perfect observations should have near-zero info loss."""
        dec = DecPOMDP(name="perfect_obs")
        for s in ["s0", "s1"]:
            dec.add_state(s)
        dec.set_initial_state("s0", 0.5)
        dec.set_initial_state("s1", 0.5)
        dec.add_agent("a1")
        dec.add_action("a1", "x")
        dec.add_observation("a1", "see_s0")
        dec.add_observation("a1", "see_s1")
        for ja in [("x",)]:
            for s in dec.states:
                dec.set_transition(s, ja, s, 1.0)
                dec.set_reward(s, ja, 0.0)
            dec.set_observation_prob("a1", "s0", ja, "see_s0", 1.0)
            dec.set_observation_prob("a1", "s1", ja, "see_s1", 1.0)
        loss = information_loss(dec, horizon=3, seed=42)
        assert loss["a1"] < 0.1  # Near-perfect observability

    def test_all_solvers_on_meeting(self):
        """All solvers produce results on meeting problem."""
        dec = multi_agent_meeting(grid_size=2)  # Small
        result_jesp = jesp(dec, horizon=1, max_iter=10, n_restarts=2, seed=42)
        result_cpde = cpde(dec, horizon=1, seed=42)
        assert result_jesp.value is not None
        assert result_cpde.value is not None

    def test_occupancy_evolves(self):
        """Occupancy state at step 1 should differ from step 0."""
        dec = decentralized_tiger()
        jp = JointPolicy()
        p1 = LocalPolicy(agent="agent1")
        p1.set_action((), "listen")
        p2 = LocalPolicy(agent="agent2")
        p2.set_action((), "listen")
        jp.set_policy("agent1", p1)
        jp.set_policy("agent2", p2)
        occ0 = occupancy_state(dec, jp, horizon=3, step=0, n_samples=5000,
                               seed=42)
        occ1 = occupancy_state(dec, jp, horizon=3, step=1, n_samples=5000,
                               seed=42)
        # After one step, agents have observation histories
        max_hist_len_0 = max(
            max(len(h) for h in hists) if hists else 0
            for (_, hists) in occ0.keys()
        ) if occ0 else 0
        max_hist_len_1 = max(
            max(len(h) for h in hists) if hists else 0
            for (_, hists) in occ1.keys()
        ) if occ1 else 0
        assert max_hist_len_1 > max_hist_len_0
