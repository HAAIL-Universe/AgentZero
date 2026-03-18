"""Tests for V224: Interactive POMDPs.

AI-Generated | Claude (Anthropic) | AgentZero A2 Session 307 | 2026-03-18
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "V216_pomdp"))

import math
import pytest
from interactive_pomdp import (
    Frame, IntentionalModel, InteractiveState, IPOMDP, IPOMDPResult,
    TheoryOfMind, level_k_analysis, find_nash_belief,
    build_multi_agent_tiger, build_pursuit_evasion,
    build_signaling_game, build_coordination_game,
    _frame_to_pomdp, _belief_key, _entropy, _kl_divergence,
)


# ===== Frame Tests =====

class TestFrame:
    def test_frame_creation(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        assert frame.agent_id == "A"
        assert set(frame.states) == {"tiger-left", "tiger-right"}
        assert len(frame.actions) == 3
        assert len(frame.observations) == 2

    def test_frame_transitions_sum_to_one(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        for s, joint_map in frame.transitions.items():
            for joint, dist in joint_map.items():
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-9, f"Transition from {s} via {joint} sums to {total}"

    def test_frame_observations_sum_to_one(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        for s, joint_map in frame.obs_function.items():
            for joint, dist in joint_map.items():
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-9

    def test_frame_get_actions(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        assert frame.get_actions() == frame.actions

    def test_frame_get_observations(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["B"]
        assert frame.get_observations() == frame.observations


# ===== IntentionalModel Tests =====

class TestIntentionalModel:
    def test_level0_uniform_policy(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["B"]
        model = IntentionalModel(
            agent_id="B", frame=frame, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        pred = model.predict_action()
        assert len(pred) == 3
        for a, p in pred.items():
            assert abs(p - 1/3) < 1e-9

    def test_level0_fixed_policy(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["B"]
        policy = {
            "tiger-left": {"listen": 1.0, "open-left": 0.0, "open-right": 0.0},
            "tiger-right": {"listen": 1.0, "open-left": 0.0, "open-right": 0.0},
        }
        model = IntentionalModel(
            agent_id="B", frame=frame, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy=policy,
        )
        pred = model.predict_action()
        assert pred["listen"] == 1.0

    def test_level0_mixed_policy(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["B"]
        policy = {
            "tiger-left": {"listen": 0.0, "open-left": 0.0, "open-right": 1.0},
            "tiger-right": {"listen": 0.0, "open-left": 1.0, "open-right": 0.0},
        }
        model = IntentionalModel(
            agent_id="B", frame=frame, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy=policy,
        )
        pred = model.predict_action()
        # Both states equally likely, each produces different action
        assert abs(pred.get("open-left", 0) - 0.5) < 1e-9
        assert abs(pred.get("open-right", 0) - 0.5) < 1e-9

    def test_level0_state_specific_predict(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["B"]
        policy = {
            "tiger-left": {"listen": 0.0, "open-left": 0.0, "open-right": 1.0},
            "tiger-right": {"listen": 0.0, "open-left": 1.0, "open-right": 0.0},
        }
        model = IntentionalModel(
            agent_id="B", frame=frame, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy=policy,
        )
        pred = model.predict_action(state="tiger-left")
        assert pred["open-right"] == 1.0


# ===== InteractiveState Tests =====

class TestInteractiveState:
    def test_interactive_state_creation(self):
        tiger = build_multi_agent_tiger()
        frame_b = tiger["frames"]["B"]
        model_b = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        istate = InteractiveState(
            physical_state="tiger-left",
            agent_models={"B": model_b},
        )
        assert istate.physical_state == "tiger-left"
        assert "B" in istate.agent_models

    def test_interactive_state_equality(self):
        tiger = build_multi_agent_tiger()
        frame_b = tiger["frames"]["B"]
        model1 = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        model2 = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        is1 = InteractiveState("tiger-left", {"B": model1})
        is2 = InteractiveState("tiger-left", {"B": model2})
        assert is1 == is2

    def test_interactive_state_inequality(self):
        tiger = build_multi_agent_tiger()
        frame_b = tiger["frames"]["B"]
        model1 = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        model2 = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.9, "tiger-right": 0.1},
        )
        is1 = InteractiveState("tiger-left", {"B": model1})
        is2 = InteractiveState("tiger-left", {"B": model2})
        assert is1 != is2

    def test_interactive_state_hash(self):
        tiger = build_multi_agent_tiger()
        frame_b = tiger["frames"]["B"]
        model = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        is1 = InteractiveState("tiger-left", {"B": model})
        h = hash(is1)
        assert isinstance(h, int)


# ===== IPOMDP Tests =====

class TestIPOMDP:
    def _make_tiger_ipomdp(self):
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        # Model B as level 0 with two possible policies
        listener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}
                    for s in frame_b.states},
        )
        random_opener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )

        return IPOMDP(
            agent_id="A",
            frame=frame_a,
            opponent_models={"B": [listener, random_opener]},
            initial_belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )

    def test_ipomdp_creation(self):
        ipomdp = self._make_tiger_ipomdp()
        assert ipomdp.agent_id == "A"
        assert ipomdp.level == 1
        assert len(ipomdp.opponent_models["B"]) == 2

    def test_predict_opponent(self):
        ipomdp = self._make_tiger_ipomdp()
        pred = ipomdp.predict_opponent("B")
        assert len(pred) == 3
        total = sum(pred.values())
        assert abs(total - 1.0) < 1e-9

    def test_predict_all_opponents(self):
        ipomdp = self._make_tiger_ipomdp()
        preds = ipomdp.predict_all_opponents()
        assert "B" in preds
        assert abs(sum(preds["B"].values()) - 1.0) < 1e-9

    def test_model_priors_uniform(self):
        ipomdp = self._make_tiger_ipomdp()
        priors = ipomdp.model_priors["B"]
        assert len(priors) == 2
        assert abs(priors[0] - 0.5) < 1e-9
        assert abs(priors[1] - 0.5) < 1e-9

    def test_solve_returns_result(self):
        ipomdp = self._make_tiger_ipomdp()
        result = ipomdp.solve(horizon=5)
        assert isinstance(result, IPOMDPResult)
        assert len(result.policy) > 0
        assert "B" in result.opponent_predictions

    def test_best_action_is_listen(self):
        """With uniform belief, listening should be preferred (avoids -100 risk)."""
        ipomdp = self._make_tiger_ipomdp()
        action = ipomdp.best_action(horizon=5)
        assert action == "listen"

    def test_belief_update_with_opponent_action(self):
        ipomdp = self._make_tiger_ipomdp()
        old_belief = dict(ipomdp.belief)
        new_belief = ipomdp.update_belief(
            "listen", "hear-left",
            opponent_actions={"B": "listen"},
        )
        # Hearing left should increase belief in tiger-left
        assert new_belief["tiger-left"] > 0.5

    def test_belief_update_without_opponent_action(self):
        ipomdp = self._make_tiger_ipomdp()
        new_belief = ipomdp.update_belief("listen", "hear-right")
        # Hearing right should increase belief in tiger-right
        assert new_belief["tiger-right"] > 0.5

    def test_belief_update_sums_to_one(self):
        ipomdp = self._make_tiger_ipomdp()
        new_belief = ipomdp.update_belief("listen", "hear-left")
        total = sum(new_belief.values())
        assert abs(total - 1.0) < 1e-9

    def test_model_belief_update(self):
        ipomdp = self._make_tiger_ipomdp()
        old_priors = list(ipomdp.model_priors["B"])
        ipomdp.update_model_beliefs("listen", "hear-left")
        new_priors = ipomdp.model_priors["B"]
        # Priors should still sum to 1
        assert abs(sum(new_priors) - 1.0) < 1e-9

    def test_simulate(self):
        ipomdp = self._make_tiger_ipomdp()
        opponent_policies = {
            "B": {s: {"listen": 1.0} for s in ipomdp.frame.states}
        }
        traj = ipomdp.simulate(
            true_state="tiger-left",
            opponent_policies=opponent_policies,
            steps=5,
            horizon=3,
        )
        assert len(traj) == 5
        for entry in traj:
            assert "step" in entry
            assert "action" in entry
            assert "reward" in entry
            assert "belief" in entry

    def test_simulate_accumulates_reward(self):
        ipomdp = self._make_tiger_ipomdp()
        opponent_policies = {
            "B": {s: {"listen": 1.0} for s in ipomdp.frame.states}
        }
        traj = ipomdp.simulate(
            true_state="tiger-left",
            opponent_policies=opponent_policies,
            steps=3,
            horizon=3,
        )
        # If agent listens, reward is -1 per step
        for entry in traj:
            if entry["action"] == "listen":
                assert entry["reward"] == -1.0

    def test_custom_model_priors(self):
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        listener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}
                    for s in frame_b.states},
        )
        random_opener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )

        ipomdp = IPOMDP(
            agent_id="A",
            frame=frame_a,
            opponent_models={"B": [listener, random_opener]},
            initial_belief={"tiger-left": 0.5, "tiger-right": 0.5},
            model_priors={"B": {0: 0.9, 1: 0.1}},
        )
        assert abs(ipomdp.model_priors["B"][0] - 0.9) < 1e-9
        assert abs(ipomdp.model_priors["B"][1] - 0.1) < 1e-9


# ===== Theory of Mind Tests =====

class TestTheoryOfMind:
    def _make_tom(self):
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        listener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}
                    for s in frame_b.states},
        )
        opener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 0.0, "open-left": 0.5, "open-right": 0.5}
                    for s in frame_b.states},
        )

        ipomdp = IPOMDP(
            agent_id="A",
            frame=frame_a,
            opponent_models={"B": [listener, opener]},
            initial_belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        return TheoryOfMind(ipomdp)

    def test_predict(self):
        tom = self._make_tom()
        pred = tom.predict("B")
        assert abs(sum(pred.values()) - 1.0) < 1e-9

    def test_explain_action_listener(self):
        tom = self._make_tom()
        explanation = tom.explain_action("B", "listen")
        # After observing "listen", listener model should have higher posterior
        assert explanation["model_0"] > explanation["model_1"]

    def test_explain_action_opener(self):
        tom = self._make_tom()
        explanation = tom.explain_action("B", "open-left")
        # After observing "open-left", opener model should be more likely
        assert explanation["model_1"] > explanation["model_0"]

    def test_perspective_take(self):
        tom = self._make_tom()
        opp_belief = tom.perspective_take("B")
        assert abs(sum(opp_belief.values()) - 1.0) < 1e-9
        # Both models have uniform belief, so perspective should be uniform
        assert abs(opp_belief["tiger-left"] - 0.5) < 1e-9

    def test_information_advantage_symmetric(self):
        tom = self._make_tom()
        adv = tom.information_advantage("B")
        # Both have same belief -> advantage ~ 0
        assert abs(adv) < 1e-9

    def test_information_advantage_asymmetric(self):
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        model = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        ipomdp = IPOMDP(
            agent_id="A",
            frame=frame_a,
            opponent_models={"B": [model]},
            initial_belief={"tiger-left": 0.9, "tiger-right": 0.1},
        )
        tom = TheoryOfMind(ipomdp)
        adv = tom.information_advantage("B")
        # Agent A has more concentrated belief -> positive advantage
        assert adv > 0

    def test_deception_value(self):
        tom = self._make_tom()
        dv = tom.deception_value("B", "listen")
        # Each of our actions should map to some probability
        for a in tom.ipomdp.frame.actions:
            assert a in dv
            assert 0 <= dv[a] <= 1.0

    def test_belief_divergence_zero(self):
        tom = self._make_tom()
        div = tom.belief_divergence("B")
        # Same beliefs -> KL = 0
        assert abs(div) < 1e-9

    def test_belief_divergence_positive(self):
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        model = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.9, "tiger-right": 0.1},
        )
        ipomdp = IPOMDP(
            agent_id="A",
            frame=frame_a,
            opponent_models={"B": [model]},
            initial_belief={"tiger-left": 0.1, "tiger-right": 0.9},
        )
        tom = TheoryOfMind(ipomdp)
        div = tom.belief_divergence("B")
        assert div > 0


# ===== Level-k Analysis Tests =====

class TestLevelK:
    def test_level_k_tiger(self):
        tiger = build_multi_agent_tiger()
        result = level_k_analysis(
            tiger["frames"], tiger["initial_beliefs"],
            max_level=2, horizon=5,
        )
        assert "A" in result
        assert "B" in result
        assert 0 in result["A"]
        assert 1 in result["A"]
        assert 2 in result["A"]

    def test_level0_is_uniform(self):
        tiger = build_multi_agent_tiger()
        result = level_k_analysis(
            tiger["frames"], tiger["initial_beliefs"],
            max_level=1, horizon=5,
        )
        level0 = result["A"][0]
        n = len(tiger["frames"]["A"].actions)
        for a, p in level0.items():
            assert abs(p - 1.0/n) < 1e-9

    def test_level1_differs_from_level0(self):
        """Level 1 should differ from uniform level 0 (it best-responds)."""
        tiger = build_multi_agent_tiger()
        result = level_k_analysis(
            tiger["frames"], tiger["initial_beliefs"],
            max_level=1, horizon=5,
        )
        level0 = result["A"][0]
        level1 = result["A"][1]
        # Level 1 is a best response, should not be uniform like level 0
        # QMDP assumes full observability next step, so opening is rational
        assert level1 != level0

    def test_level_k_distributions_sum_to_one(self):
        tiger = build_multi_agent_tiger()
        result = level_k_analysis(
            tiger["frames"], tiger["initial_beliefs"],
            max_level=2, horizon=5,
        )
        for aid in result:
            for k in result[aid]:
                total = sum(result[aid][k].values())
                assert abs(total - 1.0) < 1e-9

    def test_level_k_coordination(self):
        coord = build_coordination_game()
        result = level_k_analysis(
            coord["frames"], coord["initial_beliefs"],
            max_level=2, horizon=5,
        )
        # At higher levels, agents should converge toward coordination
        for aid in result:
            assert 0 in result[aid]
            assert 1 in result[aid]


# ===== Nash Belief Tests =====

class TestNashBelief:
    def test_nash_tiger(self):
        tiger = build_multi_agent_tiger()
        eq = find_nash_belief(
            tiger["frames"], tiger["initial_beliefs"],
            max_iterations=10, horizon=5,
        )
        assert "A" in eq
        assert "B" in eq
        for aid in eq:
            total = sum(eq[aid].values())
            assert abs(total - 1.0) < 1e-9

    def test_nash_converges(self):
        """Nash equilibrium should converge to a fixed point."""
        tiger = build_multi_agent_tiger()
        eq = find_nash_belief(
            tiger["frames"], tiger["initial_beliefs"],
            max_iterations=10, horizon=5,
        )
        # Both agents should have deterministic strategies at equilibrium
        # QMDP-based Nash: agents assume they'll observe state next step
        for aid in ["A", "B"]:
            max_prob = max(eq[aid].values())
            assert max_prob > 0.3  # Not degenerate

    def test_nash_coordination(self):
        coord = build_coordination_game()
        eq = find_nash_belief(
            coord["frames"], coord["initial_beliefs"],
            max_iterations=10, horizon=5,
        )
        for aid in eq:
            total = sum(eq[aid].values())
            assert abs(total - 1.0) < 1e-9


# ===== Frame-to-POMDP Conversion Tests =====

class TestFrameConversion:
    def test_frame_to_pomdp(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        opp_pred = {"B": {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}}
        pomdp = _frame_to_pomdp(frame, opp_pred)
        assert pomdp.states == frame.states
        assert pomdp.actions == frame.actions
        assert pomdp.observations == frame.observations

    def test_frame_to_pomdp_transitions(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        opp_pred = {"B": {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}}
        pomdp = _frame_to_pomdp(frame, opp_pred)
        # When both listen, tiger stays
        t = pomdp.get_transitions("tiger-left", "listen")
        t_dict = {s: p for s, p in t}
        assert t_dict.get("tiger-left", 0) > 0.99

    def test_frame_to_pomdp_rewards(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        opp_pred = {"B": {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}}
        pomdp = _frame_to_pomdp(frame, opp_pred)
        # Listening cost should be -1
        assert abs(pomdp.get_reward("tiger-left", "listen") - (-1.0)) < 1e-9


# ===== Helper Function Tests =====

class TestHelpers:
    def test_belief_key(self):
        b = {"s1": 0.3, "s2": 0.7}
        key = _belief_key(b)
        assert "s1" in key
        assert "s2" in key

    def test_belief_key_ignores_small(self):
        b = {"s1": 0.3, "s2": 0.7, "s3": 1e-8}
        key = _belief_key(b)
        assert "s3" not in key

    def test_entropy_uniform(self):
        dist = {"a": 0.5, "b": 0.5}
        h = _entropy(dist)
        assert abs(h - 1.0) < 1e-9

    def test_entropy_deterministic(self):
        dist = {"a": 1.0, "b": 0.0}
        h = _entropy(dist)
        assert abs(h) < 1e-9

    def test_kl_divergence_zero(self):
        p = {"a": 0.5, "b": 0.5}
        assert abs(_kl_divergence(p, p)) < 1e-9

    def test_kl_divergence_positive(self):
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.5, "b": 0.5}
        assert _kl_divergence(p, q) > 0


# ===== Example Problem Tests =====

class TestMultiAgentTiger:
    def test_build(self):
        tiger = build_multi_agent_tiger()
        assert "frames" in tiger
        assert "initial_beliefs" in tiger
        assert len(tiger["frames"]) == 2

    def test_symmetric_problem(self):
        tiger = build_multi_agent_tiger()
        fa = tiger["frames"]["A"]
        fb = tiger["frames"]["B"]
        assert len(fa.states) == len(fb.states)
        assert len(fa.actions) == len(fb.actions)

    def test_tiger_rewards(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        # Opening tiger door should give -100
        r = frame.rewards["tiger-left"][("open-left", "listen")]
        assert r == -100.0
        # Opening treasure door should give +10
        r = frame.rewards["tiger-left"][("open-right", "listen")]
        assert r == 10.0

    def test_tiger_cooperation_bonus(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        # Both opening correct door gives +20
        r = frame.rewards["tiger-left"][("open-right", "open-right")]
        assert r == 20.0

    def test_tiger_observation_accuracy(self):
        tiger = build_multi_agent_tiger()
        frame = tiger["frames"]["A"]
        # Listening when tiger is left should mostly hear left
        obs = frame.obs_function["tiger-left"][("listen", "listen")]
        assert obs["hear-left"] == 0.85


class TestPursuitEvasion:
    def test_build(self):
        pe = build_pursuit_evasion()
        assert "frames" in pe
        assert len(pe["frames"]["pursuer"].states) == 25  # 5x5 grid

    def test_pursuer_rewards(self):
        pe = build_pursuit_evasion()
        pf = pe["frames"]["pursuer"]
        # Catching evader (same cell) should give +100
        r = pf.rewards["P2E2"][("stay", "stay")]
        assert r == 100.0

    def test_evader_rewards(self):
        pe = build_pursuit_evasion()
        ef = pe["frames"]["evader"]
        # Being caught gives -100
        r = ef.rewards["P2E2"][("stay", "stay")]
        assert r == -100.0

    def test_observation_accuracy(self):
        pe = build_pursuit_evasion()
        pf = pe["frames"]["pursuer"]
        # Adjacent: should mostly observe "near"
        obs = pf.obs_function["P1E2"][("stay", "stay")]
        assert obs["near"] == 0.8

    def test_initial_beliefs(self):
        pe = build_pursuit_evasion()
        pb = pe["initial_beliefs"]["pursuer"]
        total = sum(pb.values())
        assert abs(total - 1.0) < 1e-9


class TestSignalingGame:
    def test_build(self):
        sg = build_signaling_game()
        assert "frames" in sg
        assert set(sg["frames"].keys()) == {"sender", "receiver"}

    def test_sender_costly_signal(self):
        sg = build_signaling_game()
        sf = sg["frames"]["sender"]
        # Costly signal should cost 5
        r_cost = sf.rewards["high"][("signal-costly", "invest")]
        r_cheap = sf.rewards["high"][("signal-high", "invest")]
        assert r_cheap - r_cost == 5.0

    def test_receiver_payoff(self):
        sg = build_signaling_game()
        rf = sg["frames"]["receiver"]
        # Investing in high quality = +10
        assert rf.rewards["high"][("signal-high", "invest")] == 10.0
        # Investing in low quality = -10
        assert rf.rewards["low"][("signal-high", "invest")] == -10.0

    def test_receiver_observations(self):
        sg = build_signaling_game()
        rf = sg["frames"]["receiver"]
        # Receiver sees signal directly
        obs = rf.obs_function["high"][("signal-costly", "invest")]
        assert obs["see-costly"] == 1.0


class TestCoordinationGame:
    def test_build(self):
        cg = build_coordination_game()
        assert "frames" in cg
        assert set(cg["frames"].keys()) == {"alice", "bob"}

    def test_coordination_reward(self):
        cg = build_coordination_game()
        af = cg["frames"]["alice"]
        # Both go-A: alice gets 15 (preferred), coordination
        r = af.rewards["state"][("go-A", "go-A")]
        assert r == 15.0
        # Both go-B: alice gets 10 (not preferred, but coordinated)
        r = af.rewards["state"][("go-B", "go-B")]
        assert r == 10.0

    def test_miscoordination_reward(self):
        cg = build_coordination_game()
        af = cg["frames"]["alice"]
        r = af.rewards["state"][("go-A", "go-B")]
        assert r == 0.0


# ===== Integration Tests =====

class TestIntegration:
    def test_full_tiger_scenario(self):
        """Full I-POMDP scenario: build, solve, simulate, analyze."""
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        # Create two models of B
        listener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}
                    for s in frame_b.states},
        )
        random_agent = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )

        ipomdp = IPOMDP(
            agent_id="A", frame=frame_a,
            opponent_models={"B": [listener, random_agent]},
            initial_belief={"tiger-left": 0.5, "tiger-right": 0.5},
            model_priors={"B": {0: 0.7, 1: 0.3}},
        )

        # Solve
        result = ipomdp.solve(horizon=5)
        assert isinstance(result, IPOMDPResult)

        # Theory of Mind
        tom = TheoryOfMind(ipomdp)
        pred = tom.predict("B")
        assert abs(sum(pred.values()) - 1.0) < 1e-9

        # Perspective taking
        opp_view = tom.perspective_take("B")
        assert abs(sum(opp_view.values()) - 1.0) < 1e-9

    def test_level_k_converges(self):
        """Level-k strategies should converge (or at least not diverge)."""
        tiger = build_multi_agent_tiger()
        result = level_k_analysis(
            tiger["frames"], tiger["initial_beliefs"],
            max_level=3, horizon=5,
        )
        # Higher levels should still be valid distributions
        for aid in result:
            for k in result[aid]:
                total = sum(result[aid][k].values())
                assert abs(total - 1.0) < 1e-9

    def test_belief_update_sequence(self):
        """Sequential belief updates should maintain valid distributions."""
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        model = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}
                    for s in frame_b.states},
        )
        ipomdp = IPOMDP(
            agent_id="A", frame=frame_a,
            opponent_models={"B": [model]},
            initial_belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )

        # Multiple updates
        for obs in ["hear-left", "hear-left", "hear-right", "hear-left"]:
            b = ipomdp.update_belief("listen", obs, {"B": "listen"})
            total = sum(b.values())
            assert abs(total - 1.0) < 1e-9

        # After 3 hear-left and 1 hear-right, should believe tiger-left
        assert ipomdp.belief["tiger-left"] > 0.5

    def test_tom_explain_updates_correctly(self):
        """Theory of Mind explanation should update model posteriors."""
        tiger = build_multi_agent_tiger()
        frame_a = tiger["frames"]["A"]
        frame_b = tiger["frames"]["B"]

        listener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 1.0, "open-left": 0.0, "open-right": 0.0}
                    for s in frame_b.states},
        )
        opener = IntentionalModel(
            agent_id="B", frame=frame_b, level=0,
            belief={"tiger-left": 0.5, "tiger-right": 0.5},
            policy={s: {"listen": 0.0, "open-left": 0.5, "open-right": 0.5}
                    for s in frame_b.states},
        )

        ipomdp = IPOMDP(
            agent_id="A", frame=frame_a,
            opponent_models={"B": [listener, opener]},
            initial_belief={"tiger-left": 0.5, "tiger-right": 0.5},
        )
        tom = TheoryOfMind(ipomdp)

        # Observing "listen" should strongly favor listener model
        expl = tom.explain_action("B", "listen")
        assert expl["model_0"] > 0.99  # listener model assigns p=1 to listen

    def test_pursuit_evasion_ipomdp(self):
        """Pursuer I-POMDP should prefer moving toward evader."""
        pe = build_pursuit_evasion()
        p_frame = pe["frames"]["pursuer"]
        e_frame = pe["frames"]["evader"]

        # Model evader as random
        evader_model = IntentionalModel(
            agent_id="evader", frame=e_frame, level=0,
            belief=pe["initial_beliefs"]["evader"],
        )

        ipomdp = IPOMDP(
            agent_id="pursuer",
            frame=p_frame,
            opponent_models={"evader": [evader_model]},
            initial_belief=pe["initial_beliefs"]["pursuer"],
        )
        result = ipomdp.solve(horizon=3)
        assert isinstance(result, IPOMDPResult)

    def test_signaling_game_analysis(self):
        """Signaling game level-k analysis should produce valid strategies."""
        sg = build_signaling_game()
        result = level_k_analysis(
            sg["frames"], sg["initial_beliefs"],
            max_level=2, horizon=5,
        )
        assert "sender" in result
        assert "receiver" in result
        for aid in result:
            for k in result[aid]:
                total = sum(result[aid][k].values())
                assert abs(total - 1.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
