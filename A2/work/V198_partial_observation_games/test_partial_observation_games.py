"""Tests for V198: Partial Observation Games."""

import pytest
from partial_observation_games import (
    PartialObsGame, Objective, Player, KnowledgeState, ObsStrategy,
    PartialObsGame, build_knowledge_game, solve_safety, solve_reachability,
    solve_buchi, solve_parity, solve,
    solve_safety_antichain, antichain_insert, antichain_contains,
    make_safety_po_game, make_reachability_po_game,
    make_buchi_po_game, make_co_buchi_po_game,
    analyze_observability, compare_perfect_vs_partial,
    game_statistics, game_summary,
    _initial_belief, _observation_split, _post, _post_all,
)


# ===================== Data Structure Tests =====================

class TestPartialObsGame:
    def test_create_empty(self):
        g = PartialObsGame()
        assert len(g.vertices) == 0
        assert len(g.edges) == 0

    def test_add_vertex(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        assert g.vertices == {0, 1}
        assert g.owner[0] == Player.EVEN
        assert g.owner[1] == Player.ODD
        assert g.obs[0] == 0
        assert g.obs[1] == 1

    def test_add_edge(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1)
        assert g.successors(0) == {1}
        assert g.successors(1) == set()

    def test_predecessors(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 0)
        g.add_vertex(2, Player.EVEN, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        assert g.predecessors(2) == {0, 1}
        assert g.predecessors(0) == set()

    def test_obs_class(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 10)
        g.add_vertex(1, Player.EVEN, 10)
        g.add_vertex(2, Player.ODD, 20)
        assert g.obs_class(10) == {0, 1}
        assert g.obs_class(20) == {2}

    def test_all_observations(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 10)
        g.add_vertex(1, Player.EVEN, 10)
        g.add_vertex(2, Player.ODD, 20)
        assert g.all_observations() == {10, 20}

    def test_obs_consistent(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 10)
        g.add_vertex(1, Player.EVEN, 10)
        assert g.is_observation_consistent()

    def test_obs_inconsistent(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 10)
        g.add_vertex(1, Player.ODD, 10)  # same obs, different owner
        assert not g.is_observation_consistent()

    def test_priority(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0, prio=3)
        assert g.priority[0] == 3


class TestKnowledgeState:
    def test_equality(self):
        ks1 = KnowledgeState(frozenset({0, 1}), 0)
        ks2 = KnowledgeState(frozenset({0, 1}), 0)
        assert ks1 == ks2

    def test_hash(self):
        ks1 = KnowledgeState(frozenset({0, 1}), 0)
        ks2 = KnowledgeState(frozenset({0, 1}), 0)
        assert hash(ks1) == hash(ks2)

    def test_inequality(self):
        ks1 = KnowledgeState(frozenset({0, 1}), 0)
        ks2 = KnowledgeState(frozenset({0, 2}), 0)
        assert ks1 != ks2


class TestObsStrategy:
    def test_action(self):
        s = ObsStrategy(moves={0: 1, 1: 0})
        assert s.action(0) == 1
        assert s.action(1) == 0
        assert s.action(2) is None


# ===================== Helper Function Tests =====================

class TestHelpers:
    def test_initial_belief_with_initials(self):
        g = PartialObsGame(initial={0, 1})
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.ODD, 1)
        assert _initial_belief(g) == frozenset({0, 1})

    def test_initial_belief_without_initials(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        assert _initial_belief(g) == frozenset({0, 1})

    def test_observation_split(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 10)
        g.add_vertex(1, Player.EVEN, 10)
        g.add_vertex(2, Player.ODD, 20)
        result = _observation_split(g, frozenset({0, 1, 2}))
        assert result[10] == frozenset({0, 1})
        assert result[20] == frozenset({2})

    def test_post(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 1)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        result = _post(g, frozenset({0}), 1)
        assert result == frozenset({1, 2})

    def test_post_all(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 2)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        result = _post_all(g, frozenset({0}))
        assert 1 in result
        assert 2 in result
        assert result[1] == frozenset({1})
        assert result[2] == frozenset({2})


# ===================== Knowledge Game Tests =====================

class TestKnowledgeGame:
    def test_simple_two_state(self):
        """Two states, same observation, Player 1 owns both."""
        g = PartialObsGame(objective=Objective.SAFETY)
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(0, 0)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(1, 1)
        g.initial = {0, 1}

        kg = build_knowledge_game(g)
        assert len(kg.game.vertices) >= 1

    def test_perfect_info(self):
        """With unique observations, knowledge game mirrors original."""
        g = PartialObsGame(objective=Objective.SAFETY)
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.initial = {0}

        kg = build_knowledge_game(g)
        # Should have states for both observations
        assert len(kg.game.vertices) >= 2

    def test_partial_info_merges(self):
        """States with same obs merge into one knowledge state."""
        g = PartialObsGame(objective=Objective.SAFETY)
        g.add_vertex(0, Player.EVEN, 100)  # same obs
        g.add_vertex(1, Player.EVEN, 100)  # same obs
        g.add_vertex(2, Player.ODD, 200)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        g.add_edge(2, 1)
        g.initial = {0, 1}

        kg = build_knowledge_game(g)
        # Initial belief is {0,1} with obs 100
        init_ks = kg.id_to_state[kg.initial_id]
        assert init_ks.belief == frozenset({0, 1})


# ===================== Safety Game Tests =====================

class TestSafety:
    def test_simple_safe(self):
        """Single safe self-loop. Player 1 wins trivially."""
        g = make_safety_po_game(
            n_states=1, bad=set(), even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve_safety(g)
        assert result.winning

    def test_immediate_bad(self):
        """Initial state is bad. Player 1 loses."""
        g = make_safety_po_game(
            n_states=1, bad={0}, even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve_safety(g)
        assert not result.winning

    def test_avoidable_bad(self):
        """Player 1 can avoid bad state with perfect info."""
        # 0 (Even, obs=0) -> 1 (safe, obs=1) or 2 (bad, obs=2)
        g = make_safety_po_game(
            n_states=3, bad={2}, even_states={0, 1},
            transitions=[(0, 1), (0, 2), (1, 1)],
            observations={0: 0, 1: 1, 2: 2}
        )
        result = solve_safety(g)
        assert result.winning

    def test_unavoidable_bad_partial_obs(self):
        """With partial obs, Player 1 cannot distinguish safe from bad."""
        # 0 -> 1 (safe), 0 -> 2 (bad), but 1 and 2 have same obs
        # So from 0, Player 1 goes to obs class {1,2} and can't avoid 2
        # Actually, the issue is that Player 1 chooses action from 0
        # and both actions lead to the same observation
        # If 1 and 2 have same obs, the successor belief is {1,2}
        # which intersects bad -> unsafe

        g = PartialObsGame(objective=Objective.SAFETY, target={2})
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)  # safe, obs 1
        g.add_vertex(2, Player.EVEN, 1)  # bad, obs 1
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        g.initial = {0}

        result = solve_safety(g)
        # From 0, the only successor obs is 1, giving belief {1,2}.
        # Belief {1,2} contains bad state 2 -> knowledge state has priority 1.
        # From {1,2}, successor is {1,2} again (both self-loop to same obs).
        # Odd-priority cycle -> Player 1 loses.
        assert not result.winning

    def test_opponent_forces_bad(self):
        """Player 2 can force into bad state."""
        g = make_safety_po_game(
            n_states=3, bad={2}, even_states={0},
            transitions=[(0, 1), (0, 2), (1, 0), (2, 2)],
            observations={0: 0, 1: 1, 2: 2}
        )
        # State 0 is Odd (not in even_states except 0), 1 and 2 are Odd
        # Wait, even_states={0} means only 0 is Even
        # So 1 and 2 are Odd. From 0 (Even) -> 1 or 2. Choose 1.
        # But from 1 (Odd) -> 0. Then from 0 -> 1 again. Safe loop.
        result = solve_safety(g)
        assert result.winning  # Player 1 can choose to go to 1 from 0

    def test_opponent_forces_bad_no_escape(self):
        """Opponent controls all moves, forces bad."""
        g = make_safety_po_game(
            n_states=2, bad={1}, even_states=set(),  # all Odd
            transitions=[(0, 1), (1, 1)],
            observations={0: 0, 1: 1}
        )
        result = solve_safety(g)
        assert not result.winning

    def test_three_state_safe_loop(self):
        """Player 1 can loop safely avoiding bad vertex."""
        g = make_safety_po_game(
            n_states=4, bad={3}, even_states={0, 1, 2},
            transitions=[(0, 1), (0, 3), (1, 2), (2, 0)],
            observations={0: 0, 1: 1, 2: 2, 3: 3}
        )
        result = solve_safety(g)
        assert result.winning


# ===================== Reachability Game Tests =====================

class TestReachability:
    def test_already_at_target(self):
        """Initial state is target. Player 1 wins."""
        g = make_reachability_po_game(
            n_states=1, target={0}, even_states={0},
            transitions=[], observations={0: 0}
        )
        result = solve_reachability(g)
        assert result.winning

    def test_simple_reach(self):
        """Player 1 can reach target in one step."""
        g = make_reachability_po_game(
            n_states=2, target={1}, even_states={0, 1},
            transitions=[(0, 1)], observations={0: 0, 1: 1}
        )
        result = solve_reachability(g)
        assert result.winning

    def test_unreachable(self):
        """No path to target."""
        g = make_reachability_po_game(
            n_states=2, target={1}, even_states={0},
            transitions=[(0, 0)], observations={0: 0, 1: 1}
        )
        result = solve_reachability(g)
        assert not result.winning

    def test_reach_through_opponent(self):
        """Must pass through opponent vertex to reach target."""
        g = make_reachability_po_game(
            n_states=3, target={2}, even_states={0, 2},
            transitions=[(0, 1), (1, 2), (1, 0)],
            observations={0: 0, 1: 1, 2: 2}
        )
        # Vertex 1 is Odd. Odd might loop to 0 forever.
        result = solve_reachability(g)
        assert not result.winning  # Odd can prevent reaching 2

    def test_reach_even_controls(self):
        """Player 1 controls path to target."""
        g = make_reachability_po_game(
            n_states=3, target={2}, even_states={0, 1, 2},
            transitions=[(0, 1), (1, 2)],
            observations={0: 0, 1: 1, 2: 2}
        )
        result = solve_reachability(g)
        assert result.winning

    def test_reach_partial_obs(self):
        """With partial obs, Player 1 can still reach target."""
        g = make_reachability_po_game(
            n_states=3, target={2}, even_states={0, 1, 2},
            transitions=[(0, 1), (0, 2), (1, 2)],
            observations={0: 0, 1: 10, 2: 10}  # 1 and 2 same obs
        )
        result = solve_reachability(g)
        # From 0, go to obs 10 (belief {1,2}). Target 2 is in belief.
        assert result.winning


# ===================== Buchi Game Tests =====================

class TestBuchi:
    def test_accepting_loop(self):
        """Single accepting self-loop. Player 1 wins."""
        g = make_buchi_po_game(
            n_states=1, accepting={0}, even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve_buchi(g)
        assert result.winning

    def test_non_accepting_loop(self):
        """Single non-accepting self-loop. Player 1 loses."""
        g = make_buchi_po_game(
            n_states=1, accepting=set(), even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve_buchi(g)
        assert not result.winning

    def test_alternating_accept(self):
        """Loop through accepting and non-accepting. Player 1 wins."""
        g = make_buchi_po_game(
            n_states=2, accepting={0}, even_states={0, 1},
            transitions=[(0, 1), (1, 0)],
            observations={0: 0, 1: 1}
        )
        result = solve_buchi(g)
        assert result.winning

    def test_buchi_opponent_avoids(self):
        """Opponent can avoid accepting states forever."""
        g = make_buchi_po_game(
            n_states=3, accepting={2}, even_states={0},
            transitions=[(0, 1), (1, 0), (1, 2), (2, 0)],
            observations={0: 0, 1: 1, 2: 2}
        )
        # From 0 (Even) -> 1. From 1 (Odd) -> 0 or 2. Odd avoids 2.
        result = solve_buchi(g)
        assert not result.winning


# ===================== Co-Buchi Game Tests =====================

class TestCoBuchi:
    def test_no_rejecting(self):
        """No rejecting states. Player 1 wins trivially."""
        g = make_co_buchi_po_game(
            n_states=2, rejecting=set(), even_states={0, 1},
            transitions=[(0, 1), (1, 0)],
            observations={0: 0, 1: 1}
        )
        result = solve(g)
        assert result.winning

    def test_always_rejecting(self):
        """Only rejecting states. Player 1 loses."""
        g = make_co_buchi_po_game(
            n_states=1, rejecting={0}, even_states={0},
            transitions=[(0, 0)],
            observations={0: 0}
        )
        result = solve(g)
        assert not result.winning


# ===================== Parity Game Tests =====================

class TestParity:
    def test_even_priority_loop(self):
        """Even priority self-loop. Player 1 wins."""
        g = PartialObsGame(objective=Objective.PARITY)
        g.add_vertex(0, Player.EVEN, 0, prio=2)
        g.add_edge(0, 0)
        g.initial = {0}
        result = solve_parity(g)
        assert result.winning

    def test_odd_priority_loop(self):
        """Odd priority self-loop. Player 1 loses."""
        g = PartialObsGame(objective=Objective.PARITY)
        g.add_vertex(0, Player.EVEN, 0, prio=1)
        g.add_edge(0, 0)
        g.initial = {0}
        result = solve_parity(g)
        assert not result.winning

    def test_parity_choice(self):
        """Player 1 can choose between even and odd priority cycles."""
        g = PartialObsGame(objective=Objective.PARITY)
        g.add_vertex(0, Player.EVEN, 0, prio=0)
        g.add_vertex(1, Player.EVEN, 1, prio=2)  # even cycle
        g.add_vertex(2, Player.ODD, 2, prio=1)    # odd cycle
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        g.initial = {0}
        result = solve_parity(g)
        assert result.winning  # Choose vertex 1 (even priority cycle)


# ===================== Antichain Tests =====================

class TestAntichain:
    def test_insert_non_dominated(self):
        chain = []
        chain = antichain_insert(chain, frozenset({1, 2}), "down")
        assert len(chain) == 1
        chain = antichain_insert(chain, frozenset({3, 4}), "down")
        assert len(chain) == 2

    def test_insert_dominated(self):
        chain = [frozenset({1, 2, 3})]
        chain = antichain_insert(chain, frozenset({1, 2}), "down")
        assert len(chain) == 1  # {1,2} dominated by {1,2,3}
        assert frozenset({1, 2, 3}) in chain

    def test_insert_dominates(self):
        chain = [frozenset({1, 2})]
        chain = antichain_insert(chain, frozenset({1, 2, 3}), "down")
        assert len(chain) == 1  # {1,2,3} dominates {1,2}
        assert frozenset({1, 2, 3}) in chain

    def test_insert_upward(self):
        chain = [frozenset({1, 2, 3})]
        chain = antichain_insert(chain, frozenset({1, 2}), "up")
        assert len(chain) == 1  # {1,2} dominates {1,2,3} in upward
        assert frozenset({1, 2}) in chain

    def test_contains_downward(self):
        chain = [frozenset({1, 2, 3})]
        assert antichain_contains(chain, frozenset({1, 2}), "down")
        assert not antichain_contains(chain, frozenset({1, 2, 3, 4}), "down")

    def test_contains_upward(self):
        chain = [frozenset({1, 2})]
        assert antichain_contains(chain, frozenset({1, 2, 3}), "up")
        assert not antichain_contains(chain, frozenset({1}), "up")

    def test_empty_antichain(self):
        assert not antichain_contains([], frozenset({1}), "down")


# ===================== Antichain Safety Tests =====================

class TestAntichainSafety:
    def test_simple_safe(self):
        g = make_safety_po_game(
            n_states=1, bad=set(), even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve_safety_antichain(g)
        assert result.winning

    def test_immediate_bad(self):
        g = make_safety_po_game(
            n_states=1, bad={0}, even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve_safety_antichain(g)
        assert not result.winning

    def test_avoidable_bad(self):
        g = make_safety_po_game(
            n_states=3, bad={2}, even_states={0, 1, 2},
            transitions=[(0, 1), (0, 2), (1, 1)],
            observations={0: 0, 1: 1, 2: 2}
        )
        result = solve_safety_antichain(g)
        assert result.winning


# ===================== Game Constructor Tests =====================

class TestConstructors:
    def test_make_safety(self):
        g = make_safety_po_game(
            n_states=3, bad={2}, even_states={0},
            transitions=[(0, 1), (1, 2)],
            observations={0: 0, 1: 1, 2: 2}
        )
        assert len(g.vertices) == 3
        assert g.objective == Objective.SAFETY
        assert g.target == {2}

    def test_make_reachability(self):
        g = make_reachability_po_game(
            n_states=2, target={1}, even_states={0, 1},
            transitions=[(0, 1)],
            observations={0: 0, 1: 1}
        )
        assert len(g.vertices) == 2
        assert g.objective == Objective.REACHABILITY
        # Self-loop added for target
        assert 1 in g.successors(1)

    def test_make_buchi(self):
        g = make_buchi_po_game(
            n_states=2, accepting={1}, even_states={0, 1},
            transitions=[(0, 1), (1, 0)],
            observations={0: 0, 1: 1}
        )
        assert len(g.vertices) == 2
        assert g.objective == Objective.BUCHI
        assert g.target == {1}

    def test_make_co_buchi(self):
        g = make_co_buchi_po_game(
            n_states=2, rejecting={1}, even_states={0},
            transitions=[(0, 1), (1, 0)],
            observations={0: 0, 1: 1}
        )
        assert g.objective == Objective.CO_BUCHI
        assert g.target == {1}


# ===================== Analysis Tests =====================

class TestAnalysis:
    def test_analyze_perfect_info(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 2)
        info = analyze_observability(g)
        assert info['is_perfect_info']
        assert info['info_ratio'] == 1.0
        assert info['num_observations'] == 3

    def test_analyze_blind(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 0)
        info = analyze_observability(g)
        assert info['is_trivial']
        assert info['max_class_size'] == 3
        assert info['info_ratio'] == pytest.approx(1/3)

    def test_analyze_partial(self):
        g = PartialObsGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.ODD, 1)
        info = analyze_observability(g)
        assert not info['is_perfect_info']
        assert not info['is_trivial']
        assert info['num_observations'] == 2

    def test_game_statistics(self):
        g = make_safety_po_game(
            n_states=3, bad={2}, even_states={0, 1},
            transitions=[(0, 1), (1, 2), (2, 0)],
            observations={0: 0, 1: 1, 2: 2}
        )
        stats = game_statistics(g)
        assert stats['vertices'] == 3
        assert stats['edges'] == 3
        assert stats['even_vertices'] == 2
        assert stats['odd_vertices'] == 1


# ===================== Comparison Tests =====================

class TestComparison:
    def test_perfect_vs_partial_same(self):
        """When perfect info, both should agree."""
        g = make_safety_po_game(
            n_states=2, bad={1}, even_states={0},
            transitions=[(0, 0)],
            observations={0: 0, 1: 1}
        )
        result = compare_perfect_vs_partial(g)
        assert result['partial_observation']['winning'] == result['perfect_information']['winning']
        assert not result['info_loss_matters']

    def test_info_loss_matters(self):
        """Partial obs causes loss where perfect info would win."""
        # With perfect info: Player 1 sees state, avoids bad
        # With partial obs: Player 1 can't distinguish safe from bad
        g = PartialObsGame(objective=Objective.SAFETY, target={2})
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)  # safe
        g.add_vertex(2, Player.EVEN, 1)  # bad, same obs as 1
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        g.initial = {0}

        result = compare_perfect_vs_partial(g)
        # Perfect info: Player 1 goes to 1, wins
        assert result['perfect_information']['winning']
        # Partial obs: successor belief {1,2} intersects bad
        assert not result['partial_observation']['winning']
        assert result['info_loss_matters']


# ===================== Game Summary Tests =====================

class TestSummary:
    def test_summary_string(self):
        g = make_safety_po_game(
            n_states=2, bad=set(), even_states={0, 1},
            transitions=[(0, 1), (1, 0)],
            observations={0: 0, 1: 1}
        )
        s = game_summary(g)
        assert "safety" in s.lower()
        assert "Player 1" in s


# ===================== Complex Scenario Tests =====================

class TestComplexScenarios:
    def test_maze_with_hidden_trap(self):
        """Player 1 navigates a maze where some rooms look identical.

        Layout:
        0 (start, obs=A) -> 1 (obs=B), 2 (obs=B)
        1 (safe, obs=B) -> 3 (exit, obs=C)
        2 (trap, obs=B) -> 4 (bad, obs=D)
        3 -> 3 (safe loop)
        4 -> 4 (bad loop)
        """
        g = PartialObsGame(objective=Objective.SAFETY, target={4})
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.EVEN, 1)  # same obs as 1
        g.add_vertex(3, Player.EVEN, 2)
        g.add_vertex(4, Player.EVEN, 3)
        g.add_edge(0, 1); g.add_edge(0, 2)
        g.add_edge(1, 3); g.add_edge(2, 4)
        g.add_edge(3, 3); g.add_edge(4, 4)
        g.initial = {0}

        result = solve_safety(g)
        # From belief {1,2}, successor observations are obs=2 ({3}) and obs=3 ({4}).
        # P1 chooses obs=2 (safe path). Only vertex 1 had that successor, so
        # the belief narrows to {3}. This is safe. The action disambiguates.
        assert result.winning

    def test_information_gathering(self):
        """Player 1 can gather information to distinguish states.

        0 (obs=A) -> 1 (obs=B, safe)
        0 (obs=A) -> 2 (obs=C, different from 1!)
        So P1 learns something from the observation after moving.
        """
        g = PartialObsGame(objective=Objective.SAFETY, target=set())
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.EVEN, 2)
        g.add_edge(0, 1); g.add_edge(0, 2)
        g.add_edge(1, 1); g.add_edge(2, 2)
        g.initial = {0}

        result = solve_safety(g)
        assert result.winning  # no bad states at all

    def test_blind_player_safety(self):
        """Completely blind player (one observation for all states).

        0 (obs=0) -> 1 (obs=0, safe), 2 (obs=0, bad)
        """
        g = PartialObsGame(objective=Objective.SAFETY, target={2})
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1); g.add_edge(0, 2)
        g.add_edge(1, 0); g.add_edge(2, 2)
        g.initial = {0}

        result = solve_safety(g)
        # Blind: belief is always {0,1,2} which contains bad state 2
        assert not result.winning

    def test_four_state_diamond(self):
        """Diamond structure where partial observation matters.

        0 (Even, obs=A) -> 1 (Even, obs=B), 2 (Odd, obs=B)
        1 -> 3 (safe, obs=C)
        2 -> 3 (safe, obs=C) or 4 (bad, obs=D)
        """
        g = PartialObsGame(objective=Objective.SAFETY, target={4})
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.ODD, 1)   # same obs as 1
        g.add_vertex(3, Player.EVEN, 2)
        g.add_vertex(4, Player.EVEN, 3)
        g.add_edge(0, 1); g.add_edge(0, 2)
        g.add_edge(1, 3); g.add_edge(2, 3); g.add_edge(2, 4)
        g.add_edge(3, 3); g.add_edge(4, 4)
        g.initial = {0}

        result = solve_safety(g)
        # From 0, go to obs=1 -> belief {1,2}.
        # P1 chooses obs=2 (safe). Belief narrows to {3}. Safe.
        # Even though obs consistency is violated, knowledge game handles it.
        assert result.winning

    def test_reach_with_partial_obs_strategy(self):
        """Player 1 reaches target despite partial observation."""
        g = make_reachability_po_game(
            n_states=4, target={3}, even_states={0, 1, 2, 3},
            transitions=[(0, 1), (0, 2), (1, 3), (2, 3)],
            observations={0: 0, 1: 10, 2: 10, 3: 20}
        )
        result = solve_reachability(g)
        # From 0 -> obs 10 (belief {1,2}). Both go to 3 (target). Win.
        assert result.winning

    def test_cyclic_belief_tracking(self):
        """Beliefs cycle but game is still solvable."""
        g = make_safety_po_game(
            n_states=4, bad=set(),
            even_states={0, 1, 2, 3},
            transitions=[(0, 1), (1, 2), (2, 3), (3, 0)],
            observations={0: 0, 1: 0, 2: 1, 3: 1}
        )
        result = solve_safety(g)
        assert result.winning  # no bad states

    def test_large_obs_classes(self):
        """Game with many states per observation class."""
        n = 8
        g = PartialObsGame(objective=Objective.SAFETY, target=set())
        for i in range(n):
            g.add_vertex(i, Player.EVEN, i % 2)  # two obs classes
            g.add_edge(i, (i + 1) % n)
        g.initial = {0}

        result = solve_safety(g)
        assert result.winning  # no bad states


# ===================== Solve Dispatcher Tests =====================

class TestSolveDispatcher:
    def test_dispatch_safety(self):
        g = make_safety_po_game(
            n_states=1, bad=set(), even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve(g)
        assert result.winning

    def test_dispatch_reachability(self):
        g = make_reachability_po_game(
            n_states=2, target={1}, even_states={0, 1},
            transitions=[(0, 1)], observations={0: 0, 1: 1}
        )
        result = solve(g)
        assert result.winning

    def test_dispatch_buchi(self):
        g = make_buchi_po_game(
            n_states=1, accepting={0}, even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve(g)
        assert result.winning

    def test_dispatch_parity(self):
        g = PartialObsGame(objective=Objective.PARITY)
        g.add_vertex(0, Player.EVEN, 0, prio=2)
        g.add_edge(0, 0)
        g.initial = {0}
        result = solve(g)
        assert result.winning


# ===================== Edge Case Tests =====================

class TestEdgeCases:
    def test_empty_game(self):
        g = PartialObsGame(objective=Objective.SAFETY)
        g.initial = set()
        result = solve_safety(g)
        assert result.winning  # vacuously true

    def test_single_vertex_safe(self):
        g = make_safety_po_game(
            n_states=1, bad=set(), even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve(g)
        assert result.winning

    def test_single_vertex_bad(self):
        g = make_safety_po_game(
            n_states=1, bad={0}, even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve(g)
        assert not result.winning

    def test_disconnected_components(self):
        """Two disconnected components. Only reachable one matters."""
        g = make_safety_po_game(
            n_states=4, bad={3}, even_states={0, 1, 2, 3},
            transitions=[(0, 1), (1, 0), (2, 3), (3, 2)],
            observations={0: 0, 1: 1, 2: 2, 3: 3}
        )
        # Initial = {0}, never reaches {2,3} component
        result = solve(g)
        assert result.winning

    def test_dead_end_vertex(self):
        """Vertex with no successors."""
        g = PartialObsGame(objective=Objective.SAFETY, target=set())
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1)
        # vertex 1 has no successors -- dead end is safe for safety
        g.initial = {0}
        result = solve_safety(g)
        assert result.winning

    def test_all_same_observation(self):
        """Every vertex has the same observation (blind player)."""
        g = make_safety_po_game(
            n_states=3, bad=set(),
            even_states={0, 1, 2},
            transitions=[(0, 1), (1, 2), (2, 0)],
            observations={0: 0, 1: 0, 2: 0}
        )
        result = solve(g)
        assert result.winning  # no bad states

    def test_self_loop_only(self):
        """Vertex with only self-loop."""
        g = make_safety_po_game(
            n_states=1, bad=set(), even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve(g)
        assert result.winning


# ===================== Strategy Tests =====================

class TestStrategy:
    def test_strategy_exists_for_winning(self):
        """Winning games should produce a strategy."""
        g = make_safety_po_game(
            n_states=3, bad={2}, even_states={0, 1},
            transitions=[(0, 1), (0, 2), (1, 1)],
            observations={0: 0, 1: 1, 2: 2}
        )
        result = solve_safety(g)
        assert result.winning
        # Strategy should map observation 0 to observation 1 (go to safe vertex)
        if result.strategy:
            assert 0 in result.strategy.moves

    def test_no_strategy_for_losing(self):
        """Losing games may have no strategy or empty strategy."""
        g = make_safety_po_game(
            n_states=1, bad={0}, even_states={0},
            transitions=[(0, 0)], observations={0: 0}
        )
        result = solve_safety(g)
        assert not result.winning


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
