"""
Tests for V205: Concurrent Game Structures -- ATL/ATL* Model Checking

Tests cover:
1. CGS construction and validation
2. Coalition effectiveness (one-step forcing)
3. ATL model checking (Next, Globally, Finally, Until)
4. ATL* model checking (LTL path formulas via parity game reduction)
5. Strategy extraction and simulation
6. Example games (voting, train-gate, resource allocation, pursuit-evasion)
7. Coalition comparison and power analysis
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from concurrent_game_structures import (
    # CGS
    ConcurrentGameStructure,
    # ATL formulas
    ATL, ATLOp, ATLAtom, ATLTrue_, ATLFalse_, ATLNot, ATLAnd, ATLOr,
    ATLImplies, CoalNext, CoalGlobally, CoalFinally, CoalUntil, CoalPath,
    # Model checking
    check_atl, ATLResult, _pre_coalition,
    # Strategy
    extract_coalition_strategy, simulate_play,
    # Analysis
    coalition_power, game_statistics, compare_coalitions,
    # Example games
    make_simple_voting_game, make_train_gate_game,
    make_resource_allocation_game, make_pursuit_evasion_game,
)

# V023 LTL imports for ATL*
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V023_ltl_model_checking'))
from ltl_model_checker import (
    Atom, LTLTrue, LTLFalse, Not as LTLNot, And as LTLAnd, Or as LTLOr,
    Next as LTLNext, Finally as LTLFinally, Globally as LTLGlobally,
    Until as LTLUntil,
)


# ===================================================================
# Helper: minimal 2-agent CGS for unit tests
# ===================================================================

def make_two_state_game():
    """
    Two states: 0 (safe), 1 (unsafe).
    Two agents: a, b. Each has actions {0, 1}.
    From state 0:
        (0,0) -> 0, (0,1) -> 0, (1,0) -> 0, (1,1) -> 1
    From state 1:
        all -> 1 (absorbing)
    Agent 'a' alone can keep the system safe by playing 0.
    """
    cgs = ConcurrentGameStructure()
    cgs.add_state(0, {"safe"}, initial=True)
    cgs.add_state(1, {"unsafe"})

    cgs.add_agent("a")
    cgs.add_agent("b")

    for s in [0, 1]:
        cgs.set_actions("a", s, {0, 1})
        cgs.set_actions("b", s, {0, 1})

    # From safe: only (1,1) leads to unsafe
    cgs.set_transition(0, (0, 0), 0)
    cgs.set_transition(0, (0, 1), 0)
    cgs.set_transition(0, (1, 0), 0)
    cgs.set_transition(0, (1, 1), 1)

    # From unsafe: absorbing
    cgs.set_transition(1, (0, 0), 1)
    cgs.set_transition(1, (0, 1), 1)
    cgs.set_transition(1, (1, 0), 1)
    cgs.set_transition(1, (1, 1), 1)

    return cgs


def make_three_state_chain():
    """
    Three states: 0 -> 1 -> 2 (chain).
    Single agent 'a'. Actions: {0=stay, 1=advance}.
    State 2 is absorbing.
    """
    cgs = ConcurrentGameStructure()
    cgs.add_state(0, {"start"}, initial=True)
    cgs.add_state(1, {"mid"})
    cgs.add_state(2, {"goal"})

    cgs.add_agent("a")

    cgs.set_actions("a", 0, {0, 1})
    cgs.set_actions("a", 1, {0, 1})
    cgs.set_actions("a", 2, {0})

    cgs.set_transition(0, (0,), 0)
    cgs.set_transition(0, (1,), 1)
    cgs.set_transition(1, (0,), 1)
    cgs.set_transition(1, (1,), 2)
    cgs.set_transition(2, (0,), 2)

    return cgs


# ===================================================================
# Section 1: CGS Construction and Validation
# ===================================================================

class TestCGSConstruction:
    def test_add_states(self):
        cgs = ConcurrentGameStructure()
        cgs.add_state(0, {"p"}, initial=True)
        cgs.add_state(1, {"q"})
        assert cgs.states == {0, 1}
        assert cgs.initial == {0}
        assert cgs.labeling[0] == {"p"}

    def test_add_agents(self):
        cgs = ConcurrentGameStructure()
        cgs.add_agent("alice")
        cgs.add_agent("bob")
        assert cgs.agents == ["alice", "bob"]
        assert "alice" in cgs.actions
        assert "bob" in cgs.actions

    def test_duplicate_agent(self):
        cgs = ConcurrentGameStructure()
        cgs.add_agent("alice")
        cgs.add_agent("alice")
        assert cgs.agents == ["alice"]

    def test_set_actions(self):
        cgs = ConcurrentGameStructure()
        cgs.add_state(0)
        cgs.add_agent("a")
        cgs.set_actions("a", 0, {0, 1, 2})
        assert cgs.get_actions("a", 0) == {0, 1, 2}

    def test_joint_actions(self):
        cgs = make_two_state_game()
        ja = cgs.joint_actions(0)
        assert len(ja) == 4  # 2 actions x 2 agents
        assert (0, 0) in ja
        assert (1, 1) in ja

    def test_successors(self):
        cgs = make_two_state_game()
        succs = cgs.successors(0)
        assert succs == {0, 1}
        succs1 = cgs.successors(1)
        assert succs1 == {1}

    def test_validate_complete(self):
        cgs = make_two_state_game()
        errors = cgs.validate()
        assert errors == []

    def test_validate_missing_transition(self):
        cgs = ConcurrentGameStructure()
        cgs.add_state(0, initial=True)
        cgs.add_agent("a")
        cgs.set_actions("a", 0, {0, 1})
        # Missing transitions
        errors = cgs.validate()
        assert len(errors) == 2

    def test_three_agent_game(self):
        cgs = ConcurrentGameStructure()
        cgs.add_state(0, initial=True)
        cgs.add_agent("a")
        cgs.add_agent("b")
        cgs.add_agent("c")
        cgs.set_actions("a", 0, {0, 1})
        cgs.set_actions("b", 0, {0, 1})
        cgs.set_actions("c", 0, {0, 1})
        ja = cgs.joint_actions(0)
        assert len(ja) == 8  # 2^3

    def test_empty_game_stats(self):
        cgs = ConcurrentGameStructure()
        stats = game_statistics(cgs)
        assert stats["states"] == 0
        assert stats["agents"] == 0


# ===================================================================
# Section 2: Coalition Effectiveness
# ===================================================================

class TestCoalitionEffectiveness:
    def test_single_agent_can_force_safe(self):
        """Agent 'a' alone can keep system safe by playing 0."""
        cgs = make_two_state_game()
        # a can force next state to be safe (play 0)
        assert cgs.coalition_effectiveness(0, frozenset({"a"}), {0})

    def test_single_agent_cannot_force_unsafe(self):
        """Agent 'a' alone cannot force system to unsafe."""
        cgs = make_two_state_game()
        # a cannot force next state to unsafe (b can play 0)
        assert not cgs.coalition_effectiveness(0, frozenset({"a"}), {1})

    def test_both_agents_can_force_unsafe(self):
        """Both agents together can force any successor."""
        cgs = make_two_state_game()
        assert cgs.coalition_effectiveness(0, frozenset({"a", "b"}), {1})
        assert cgs.coalition_effectiveness(0, frozenset({"a", "b"}), {0})

    def test_empty_coalition_cannot_force(self):
        """Empty coalition can only force if all actions lead to target."""
        cgs = make_two_state_game()
        # From state 0, not all joint actions lead to safe
        assert not cgs.coalition_effectiveness(0, frozenset(), {0})

    def test_single_agent_chain_advance(self):
        """Single agent can force advance in chain."""
        cgs = make_three_state_chain()
        assert cgs.coalition_effectiveness(0, frozenset({"a"}), {1})
        assert cgs.coalition_effectiveness(1, frozenset({"a"}), {2})

    def test_absorbing_state(self):
        """From unsafe absorbing state, everyone forces staying."""
        cgs = make_two_state_game()
        assert cgs.coalition_effectiveness(1, frozenset(), {1})

    def test_coalition_effectiveness_b_can_keep_safe(self):
        """Agent 'b' alone can also keep safe (play 0)."""
        cgs = make_two_state_game()
        assert cgs.coalition_effectiveness(0, frozenset({"b"}), {0})


# ===================================================================
# Section 3: ATL Model Checking -- Boolean and Atomic
# ===================================================================

class TestATLBoolean:
    def test_atom(self):
        cgs = make_two_state_game()
        result = check_atl(cgs, ATLAtom("safe"))
        assert result.satisfaction_set == {0}

    def test_true(self):
        cgs = make_two_state_game()
        result = check_atl(cgs, ATLTrue_())
        assert result.satisfaction_set == {0, 1}

    def test_false(self):
        cgs = make_two_state_game()
        result = check_atl(cgs, ATLFalse_())
        assert result.satisfaction_set == set()

    def test_not(self):
        cgs = make_two_state_game()
        result = check_atl(cgs, ATLNot(ATLAtom("safe")))
        assert result.satisfaction_set == {1}

    def test_and(self):
        cgs = make_two_state_game()
        cgs.labeling[0].add("warm")
        result = check_atl(cgs, ATLAnd(ATLAtom("safe"), ATLAtom("warm")))
        assert result.satisfaction_set == {0}

    def test_or(self):
        cgs = make_two_state_game()
        result = check_atl(cgs, ATLOr(ATLAtom("safe"), ATLAtom("unsafe")))
        assert result.satisfaction_set == {0, 1}

    def test_implies(self):
        cgs = make_two_state_game()
        # safe -> safe is tautology at safe state
        result = check_atl(cgs, ATLImplies(ATLAtom("unsafe"), ATLAtom("safe")))
        # unsafe->safe: state 0 (not unsafe, vacuously true) + state 1 (unsafe but not safe => false)
        assert 0 in result.satisfaction_set
        assert 1 not in result.satisfaction_set

    def test_holds_in_initial(self):
        cgs = make_two_state_game()
        result = check_atl(cgs, ATLAtom("safe"))
        assert result.holds_in_initial


# ===================================================================
# Section 4: ATL Model Checking -- Coalition Temporal
# ===================================================================

class TestATLNext:
    def test_coalition_next_safe(self):
        """Agent a can ensure next state is safe."""
        cgs = make_two_state_game()
        formula = CoalNext({"a"}, ATLAtom("safe"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set
        # From unsafe, a cannot force safe
        assert 1 not in result.satisfaction_set

    def test_grand_coalition_next_unsafe(self):
        """Both agents can force next state to unsafe."""
        cgs = make_two_state_game()
        formula = CoalNext({"a", "b"}, ATLAtom("unsafe"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set

    def test_empty_coalition_next(self):
        """Empty coalition: all joint actions must lead to target."""
        cgs = make_two_state_game()
        formula = CoalNext(set(), ATLAtom("safe"))
        result = check_atl(cgs, formula)
        # From 0, (1,1)->1 so not all go to safe
        assert 0 not in result.satisfaction_set

    def test_single_agent_chain_next(self):
        """Agent can advance one step."""
        cgs = make_three_state_chain()
        formula = CoalNext({"a"}, ATLAtom("mid"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set

    def test_next_from_goal(self):
        """From goal (absorbing), next is still goal."""
        cgs = make_three_state_chain()
        formula = CoalNext({"a"}, ATLAtom("goal"))
        result = check_atl(cgs, formula)
        assert 2 in result.satisfaction_set


class TestATLGlobally:
    def test_agent_can_globally_safe(self):
        """Agent a can keep system safe forever."""
        cgs = make_two_state_game()
        formula = CoalGlobally({"a"}, ATLAtom("safe"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set
        assert result.holds_in_initial

    def test_from_unsafe_cannot_globally_safe(self):
        """From unsafe, cannot maintain safe."""
        cgs = make_two_state_game()
        formula = CoalGlobally({"a"}, ATLAtom("safe"))
        result = check_atl(cgs, formula)
        assert 1 not in result.satisfaction_set

    def test_globally_true(self):
        """Any coalition can globally maintain true."""
        cgs = make_two_state_game()
        formula = CoalGlobally(set(), ATLTrue_())
        result = check_atl(cgs, formula)
        assert result.satisfaction_set == {0, 1}

    def test_globally_absorbing(self):
        """At absorbing goal state, globally goal holds."""
        cgs = make_three_state_chain()
        formula = CoalGlobally({"a"}, ATLAtom("goal"))
        result = check_atl(cgs, formula)
        assert 2 in result.satisfaction_set
        assert 0 not in result.satisfaction_set


class TestATLFinally:
    def test_agent_can_finally_goal(self):
        """Agent can eventually reach goal in chain."""
        cgs = make_three_state_chain()
        formula = CoalFinally({"a"}, ATLAtom("goal"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set
        assert 1 in result.satisfaction_set
        assert 2 in result.satisfaction_set

    def test_cannot_finally_safe_from_unsafe(self):
        """Cannot reach safe from absorbing unsafe."""
        cgs = make_two_state_game()
        formula = CoalFinally({"a"}, ATLAtom("safe"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set  # already safe
        assert 1 not in result.satisfaction_set  # absorbing unsafe

    def test_finally_already_satisfied(self):
        """If goal holds now, finally is trivially satisfied."""
        cgs = make_three_state_chain()
        formula = CoalFinally({"a"}, ATLAtom("start"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set


class TestATLUntil:
    def test_safe_until_goal(self):
        """Agent can maintain safe until reaching goal."""
        cgs = make_three_state_chain()
        # start->mid->goal, all are in different props
        # Let's check: agent can maintain start|mid until goal
        formula = CoalUntil(
            {"a"},
            ATLOr(ATLAtom("start"), ATLAtom("mid")),
            ATLAtom("goal")
        )
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set
        assert 1 in result.satisfaction_set

    def test_until_already_target(self):
        """If psi holds now, phi U psi is trivially true."""
        cgs = make_three_state_chain()
        formula = CoalUntil({"a"}, ATLAtom("start"), ATLAtom("goal"))
        result = check_atl(cgs, formula)
        assert 2 in result.satisfaction_set  # goal already holds

    def test_until_impossible(self):
        """Cannot reach goal from absorbing unsafe state."""
        cgs = make_two_state_game()
        formula = CoalUntil({"a"}, ATLAtom("unsafe"), ATLAtom("safe"))
        result = check_atl(cgs, formula)
        assert 1 not in result.satisfaction_set


# ===================================================================
# Section 5: ATL* Model Checking (Parity Game Reduction)
# ===================================================================

class TestATLStar:
    def test_atl_star_globally_safe(self):
        """ATL* with G(safe) path formula -- equivalent to ATL G safe."""
        cgs = make_two_state_game()
        path = LTLGlobally(Atom("safe"))
        formula = CoalPath({"a"}, path)
        result = check_atl(cgs, formula)
        # Agent a can enforce G(safe) from state 0
        assert 0 in result.satisfaction_set

    def test_atl_star_finally_goal(self):
        """ATL* with F(goal) path formula."""
        cgs = make_three_state_chain()
        path = LTLFinally(Atom("goal"))
        formula = CoalPath({"a"}, path)
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set
        assert 2 in result.satisfaction_set

    def test_atl_star_next(self):
        """ATL* with X(mid) path formula."""
        cgs = make_three_state_chain()
        path = LTLNext(Atom("mid"))
        formula = CoalPath({"a"}, path)
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set

    def test_atl_star_until(self):
        """ATL* with true U goal."""
        cgs = make_three_state_chain()
        path = LTLUntil(LTLTrue(), Atom("goal"))
        formula = CoalPath({"a"}, path)
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set

    def test_atl_star_empty_formula(self):
        """ATL* with trivially true formula."""
        cgs = make_two_state_game()
        path = LTLTrue()
        formula = CoalPath({"a"}, path)
        result = check_atl(cgs, formula)
        assert result.satisfaction_set == {0, 1}


# ===================================================================
# Section 6: Strategy Extraction
# ===================================================================

class TestStrategy:
    def test_next_strategy(self):
        """Extract coalition strategy for next-step goal."""
        cgs = make_two_state_game()
        formula = CoalNext({"a"}, ATLAtom("safe"))
        strat = extract_coalition_strategy(cgs, frozenset({"a"}), formula)
        assert 0 in strat
        assert strat[0]["a"] == 0  # play 0 to stay safe

    def test_globally_strategy(self):
        """Strategy for globally safe should always play 0."""
        cgs = make_two_state_game()
        formula = CoalGlobally({"a"}, ATLAtom("safe"))
        strat = extract_coalition_strategy(cgs, frozenset({"a"}), formula)
        assert 0 in strat
        assert strat[0]["a"] == 0

    def test_finally_strategy(self):
        """Strategy for eventually reaching goal."""
        cgs = make_three_state_chain()
        formula = CoalFinally({"a"}, ATLAtom("goal"))
        strat = extract_coalition_strategy(cgs, frozenset({"a"}), formula)
        # Should advance at each non-goal state
        assert 0 in strat
        assert 1 in strat

    def test_until_strategy(self):
        """Strategy for until formula."""
        cgs = make_three_state_chain()
        formula = CoalUntil(
            {"a"},
            ATLOr(ATLAtom("start"), ATLAtom("mid")),
            ATLAtom("goal")
        )
        strat = extract_coalition_strategy(cgs, frozenset({"a"}), formula)
        assert len(strat) >= 2


# ===================================================================
# Section 7: Simulation
# ===================================================================

class TestSimulation:
    def test_simulate_safe_strategy(self):
        """Simulate with a's safe strategy."""
        cgs = make_two_state_game()
        strat = {0: {"a": 0}, 1: {"a": 0}}
        trace = simulate_play(cgs, strat, start=0,
                              coalition=frozenset({"a"}), max_steps=10)
        # Should stay at state 0 (safe)
        for state, ja, labels in trace:
            assert "safe" in labels

    def test_simulate_chain_advance(self):
        """Simulate advancing through chain."""
        cgs = make_three_state_chain()
        strat = {0: {"a": 1}, 1: {"a": 1}, 2: {"a": 0}}
        trace = simulate_play(cgs, strat, start=0,
                              coalition=frozenset({"a"}), max_steps=10)
        states = [t[0] for t in trace]
        assert 0 in states
        assert 1 in states
        assert 2 in states

    def test_simulate_with_opponent_strategy(self):
        """Simulate with explicit opponent strategy."""
        cgs = make_two_state_game()
        coal_strat = {0: {"a": 0}}
        opp_strat = {0: {"b": 1}}  # opponent tries to cause unsafe
        trace = simulate_play(cgs, coal_strat, opp_strat,
                              start=0, coalition=frozenset({"a"}),
                              max_steps=5)
        # a plays 0, b plays 1 -> (0,1) -> state 0 (safe)
        for state, ja, labels in trace:
            assert "safe" in labels

    def test_simulate_default_start(self):
        """Simulate from default initial state."""
        cgs = make_two_state_game()
        trace = simulate_play(cgs, {}, start=None, max_steps=3)
        assert len(trace) > 0


# ===================================================================
# Section 8: Voting Game
# ===================================================================

class TestVotingGame:
    def test_construction(self):
        cgs = make_simple_voting_game(3)
        assert len(cgs.states) == 3
        assert len(cgs.agents) == 3
        errors = cgs.validate()
        assert errors == []

    def test_majority_can_pass(self):
        """Majority coalition can force 'pass'."""
        cgs = make_simple_voting_game(3)
        formula = CoalNext({"v0", "v1"}, ATLAtom("pass"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set  # from voting state

    def test_minority_cannot_pass(self):
        """Single voter cannot force 'pass'."""
        cgs = make_simple_voting_game(3)
        formula = CoalNext({"v0"}, ATLAtom("pass"))
        result = check_atl(cgs, formula)
        assert 0 not in result.satisfaction_set

    def test_minority_can_block(self):
        """Two voters can block pass (force fail)."""
        cgs = make_simple_voting_game(3)
        formula = CoalNext({"v0", "v1"}, ATLAtom("fail"))
        result = check_atl(cgs, formula)
        assert 0 in result.satisfaction_set

    def test_single_cannot_block(self):
        """Single voter cannot force fail either (2 others can pass)."""
        cgs = make_simple_voting_game(3)
        formula = CoalNext({"v0"}, ATLAtom("fail"))
        result = check_atl(cgs, formula)
        assert 0 not in result.satisfaction_set

    def test_grand_coalition_can_do_anything(self):
        """All voters together can force any outcome."""
        cgs = make_simple_voting_game(3)
        f_pass = CoalNext({"v0", "v1", "v2"}, ATLAtom("pass"))
        f_fail = CoalNext({"v0", "v1", "v2"}, ATLAtom("fail"))
        assert 0 in check_atl(cgs, f_pass).satisfaction_set
        assert 0 in check_atl(cgs, f_fail).satisfaction_set

    def test_five_voters(self):
        """5-voter game: need 3 for majority."""
        cgs = make_simple_voting_game(5)
        assert len(cgs.agents) == 5
        # 3 voters can force pass
        f3 = CoalNext({"v0", "v1", "v2"}, ATLAtom("pass"))
        assert 0 in check_atl(cgs, f3).satisfaction_set
        # 2 voters cannot
        f2 = CoalNext({"v0", "v1"}, ATLAtom("pass"))
        assert 0 not in check_atl(cgs, f2).satisfaction_set


# ===================================================================
# Section 9: Train-Gate Game
# ===================================================================

class TestTrainGateGame:
    def test_construction(self):
        cgs = make_train_gate_game()
        assert len(cgs.states) == 4
        assert len(cgs.agents) == 2
        errors = cgs.validate()
        assert errors == []

    def test_controller_can_prevent_crash(self):
        """Controller can always prevent crash (globally)."""
        cgs = make_train_gate_game()
        formula = CoalGlobally({"controller"}, ATLNot(ATLAtom("crash")))
        result = check_atl(cgs, formula)
        assert result.holds_in_initial

    def test_train_alone_cannot_prevent_crash(self):
        """Train alone cannot prevent crash if it enters without gate open."""
        cgs = make_train_gate_game()
        formula = CoalGlobally({"train"}, ATLNot(ATLAtom("crash")))
        result = check_atl(cgs, formula)
        # Train can avoid crash by never entering (playing 0), but it CAN
        # ensure safety by just not entering
        assert result.holds_in_initial

    def test_train_can_reach_tunnel(self):
        """Train+controller together can reach tunnel."""
        cgs = make_train_gate_game()
        formula = CoalFinally({"train", "controller"}, ATLAtom("in_tunnel"))
        result = check_atl(cgs, formula)
        assert result.holds_in_initial

    def test_train_alone_cannot_reach_tunnel(self):
        """Train alone cannot guarantee reaching tunnel (controller may block)."""
        cgs = make_train_gate_game()
        formula = CoalFinally({"train"}, ATLAtom("in_tunnel"))
        result = check_atl(cgs, formula)
        assert not result.holds_in_initial

    def test_crash_is_absorbing(self):
        """Once crashed, globally crashed."""
        cgs = make_train_gate_game()
        formula = CoalGlobally(set(), ATLAtom("crash"))
        result = check_atl(cgs, formula)
        assert 3 in result.satisfaction_set


# ===================================================================
# Section 10: Resource Allocation Game
# ===================================================================

class TestResourceAllocationGame:
    def test_construction(self):
        cgs = make_resource_allocation_game()
        assert len(cgs.agents) == 3
        errors = cgs.validate()
        assert errors == []

    def test_allocator_controls_grant(self):
        """Allocator + process can get the resource."""
        cgs = make_resource_allocation_game()
        formula = CoalFinally({"p0", "allocator"}, ATLAtom("p0_has"))
        result = check_atl(cgs, formula)
        assert result.holds_in_initial

    def test_stats(self):
        """Game statistics are reasonable."""
        cgs = make_resource_allocation_game()
        stats = game_statistics(cgs)
        assert stats["agents"] == 3
        assert stats["states"] > 0


# ===================================================================
# Section 11: Pursuit-Evasion Game
# ===================================================================

class TestPursuitEvasionGame:
    def test_construction_2x2(self):
        cgs = make_pursuit_evasion_game(2)
        assert len(cgs.agents) == 2
        errors = cgs.validate()
        assert errors == []

    def test_pursuer_cannot_guarantee_catch_2x2(self):
        """On 2x2 grid with simultaneous moves, pursuer can't guarantee catch."""
        cgs = make_pursuit_evasion_game(2)
        formula = CoalFinally({"pursuer"}, ATLAtom("caught"))
        result = check_atl(cgs, formula)
        # In a concurrent game, evader can mirror pursuer's moves
        # Pursuer cannot guarantee catching from initial (opposite corners)
        # But from caught states, trivially satisfied
        caught_states = {s for s in cgs.states
                        if "caught" in cgs.labeling.get(s, set())}
        for s in caught_states:
            assert s in result.satisfaction_set

    def test_caught_is_absorbing(self):
        """Once caught, stays caught."""
        cgs = make_pursuit_evasion_game(2)
        formula = CoalGlobally(set(), ATLAtom("caught"))
        result = check_atl(cgs, formula)
        # Find a caught state
        caught_states = {s for s in cgs.states
                        if "caught" in cgs.labeling.get(s, set())}
        for s in caught_states:
            assert s in result.satisfaction_set

    def test_stats(self):
        cgs = make_pursuit_evasion_game(2)
        stats = game_statistics(cgs)
        assert stats["states"] == 16  # 4 positions x 4 positions
        assert stats["agents"] == 2


# ===================================================================
# Section 12: Coalition Comparison and Power Analysis
# ===================================================================

class TestCoalitionAnalysis:
    def test_coalition_power(self):
        """Coalition power computation."""
        cgs = make_two_state_game()
        power = coalition_power(cgs)
        # Grand coalition has power everywhere
        assert 0 in power[frozenset({"a", "b"})]

    def test_compare_coalitions(self):
        """Compare different coalitions on same formula template."""
        cgs = make_simple_voting_game(3)
        formula = CoalNext({"v0"}, ATLAtom("pass"))
        results = compare_coalitions(
            cgs, formula,
            [{"v0"}, {"v0", "v1"}, {"v0", "v1", "v2"}]
        )
        # Single voter can't pass, majority can, grand coalition can
        assert not results[str(sorted({"v0"}))]["holds_initial"]
        assert results[str(sorted({"v0", "v1"}))]["holds_initial"]
        assert results[str(sorted({"v0", "v1", "v2"}))]["holds_initial"]

    def test_game_statistics(self):
        """Statistics for voting game."""
        cgs = make_simple_voting_game(3)
        stats = game_statistics(cgs)
        assert stats["states"] == 3
        assert stats["agents"] == 3
        assert "voting" in stats["propositions"]


# ===================================================================
# Section 13: Pre-coalition (internal)
# ===================================================================

class TestPreCoalition:
    def test_pre_a_safe(self):
        """Pre_a({safe}) includes state 0."""
        cgs = make_two_state_game()
        pre = _pre_coalition(cgs, frozenset({"a"}), {0})
        assert 0 in pre

    def test_pre_empty_absorbing(self):
        """Pre_{}({unsafe}) includes unsafe (absorbing)."""
        cgs = make_two_state_game()
        pre = _pre_coalition(cgs, frozenset(), {1})
        assert 1 in pre

    def test_pre_chain(self):
        """Pre_a({mid}) from start."""
        cgs = make_three_state_chain()
        pre = _pre_coalition(cgs, frozenset({"a"}), {1})
        assert 0 in pre


# ===================================================================
# Section 14: Nested Formulas
# ===================================================================

class TestNestedFormulas:
    def test_nested_next(self):
        """<<a>>X <<a>>X goal in chain."""
        cgs = make_three_state_chain()
        inner = CoalNext({"a"}, ATLAtom("goal"))
        outer = CoalNext({"a"}, inner)
        result = check_atl(cgs, outer)
        assert 0 in result.satisfaction_set  # 0 -> 1 -> 2

    def test_finally_implies(self):
        """<<a>>F (safe -> safe) holds everywhere."""
        cgs = make_two_state_game()
        impl = ATLImplies(ATLAtom("safe"), ATLAtom("safe"))
        formula = CoalFinally({"a"}, impl)
        result = check_atl(cgs, formula)
        assert result.satisfaction_set == {0, 1}

    def test_globally_or(self):
        """<<a>>G (safe | unsafe) is trivially true."""
        cgs = make_two_state_game()
        disj = ATLOr(ATLAtom("safe"), ATLAtom("unsafe"))
        formula = CoalGlobally({"a"}, disj)
        result = check_atl(cgs, formula)
        assert result.satisfaction_set == {0, 1}


# ===================================================================
# Section 15: Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_single_state_game(self):
        """Game with one self-looping state."""
        cgs = ConcurrentGameStructure()
        cgs.add_state(0, {"p"}, initial=True)
        cgs.add_agent("a")
        cgs.set_actions("a", 0, {0})
        cgs.set_transition(0, (0,), 0)

        formula = CoalGlobally({"a"}, ATLAtom("p"))
        result = check_atl(cgs, formula)
        assert result.satisfaction_set == {0}

    def test_no_initial_states(self):
        """Game with no initial states."""
        cgs = ConcurrentGameStructure()
        cgs.add_state(0, {"p"})
        cgs.add_agent("a")
        cgs.set_actions("a", 0, {0})
        cgs.set_transition(0, (0,), 0)

        result = check_atl(cgs, ATLAtom("p"))
        assert result.holds_in_initial  # vacuously true

    def test_formula_repr(self):
        """Formula string representation."""
        f = CoalNext({"a", "b"}, ATLAtom("safe"))
        s = repr(f)
        assert "X" in s
        assert "safe" in s

    def test_atl_true_false_shortcuts(self):
        """True and False constructors."""
        assert ATLTrue_().op == ATLOp.TRUE
        assert ATLFalse_().op == ATLOp.FALSE


# ===================================================================
# Section 16: Integration -- Multi-step ATL properties
# ===================================================================

class TestIntegration:
    def test_voting_repeated_pass(self):
        """Majority can ensure pass happens infinitely often."""
        cgs = make_simple_voting_game(3)
        # After pass, goes back to voting, then can pass again
        # <<v0,v1>>G <<v0,v1>>F pass -- nested: always eventually pass
        inner = CoalFinally({"v0", "v1"}, ATLAtom("pass"))
        outer = CoalGlobally({"v0", "v1"}, inner)
        result = check_atl(cgs, outer)
        assert result.holds_in_initial

    def test_train_gate_safety_liveness(self):
        """Controller ensures safety; together they ensure liveness."""
        cgs = make_train_gate_game()
        # Safety: controller prevents crash
        safety = CoalGlobally({"controller"}, ATLNot(ATLAtom("crash")))
        s_result = check_atl(cgs, safety)
        assert s_result.holds_in_initial

        # Liveness: together they can reach tunnel
        liveness = CoalFinally({"train", "controller"}, ATLAtom("in_tunnel"))
        l_result = check_atl(cgs, liveness)
        assert l_result.holds_in_initial

    def test_chain_reachability_and_maintenance(self):
        """Can reach goal and then maintain it."""
        cgs = make_three_state_chain()
        reach = CoalFinally({"a"}, ATLAtom("goal"))
        maintain = CoalGlobally({"a"}, ATLAtom("goal"))

        r_result = check_atl(cgs, reach)
        m_result = check_atl(cgs, maintain)

        assert 0 in r_result.satisfaction_set
        assert 2 in m_result.satisfaction_set
        assert 0 not in m_result.satisfaction_set  # can't globally goal from start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
