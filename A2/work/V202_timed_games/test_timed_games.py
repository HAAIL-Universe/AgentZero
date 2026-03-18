"""Tests for V202: Timed Games."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from timed_games import (
    TimedGame, Player, TimedGameResult, TimedEnergyResult,
    SymState, ZoneStore, TimedGameTrace,
    solve_reachability, solve_safety, solve_buchi, solve_timed_energy,
    simulate_play, check_timed_strategy, game_statistics, game_summary,
    compare_reachability_safety, make_timed_game, _parse_guard_str,
    cat_mouse_game, resource_game, traffic_light_game, fischer_game,
    _explore_game, _successor_zone, _zone_past, _zone_union_overapprox,
    _backward_edge, _undo_resets,
)
from timed_automata import (
    Edge, Guard, Zone, ClockConstraint, CompOp,
    true_guard, clock_leq, clock_lt, clock_geq, clock_gt, clock_eq,
    guard_and, initial_zone,
)


# ===========================================================================
# Data structure tests
# ===========================================================================

class TestTimedGame:
    def test_create_empty(self):
        g = TimedGame()
        assert len(g.locations) == 0
        assert len(g.edges) == 0

    def test_add_location(self):
        g = TimedGame()
        g.add_location('A', Player.EVEN)
        g.add_location('B', Player.ODD)
        assert 'A' in g.locations
        assert g.owner['A'] == Player.EVEN
        assert g.owner['B'] == Player.ODD

    def test_add_location_with_invariant(self):
        g = TimedGame(clocks={'x'})
        g.add_location('A', Player.EVEN, clock_leq('x', 5))
        assert g.invariants['A'].constraints[0].value == 5

    def test_add_location_with_priority(self):
        g = TimedGame()
        g.add_location('A', Player.EVEN, priority=3)
        assert g.priorities['A'] == 3

    def test_add_edge(self):
        g = TimedGame(clocks={'x'})
        g.add_location('A', Player.EVEN)
        g.add_location('B', Player.ODD)
        idx = g.add_edge('A', 'B', 'go', clock_leq('x', 3))
        assert idx == 0
        assert g.edges[0].source == 'A'
        assert g.edges[0].target == 'B'

    def test_add_edge_with_weight(self):
        g = TimedGame(clocks={'x'})
        g.add_location('A', Player.EVEN)
        g.add_location('B', Player.ODD)
        idx = g.add_edge('A', 'B', 'go', weight=5)
        assert g.weights[idx] == 5

    def test_get_edges_from(self):
        g = TimedGame(clocks={'x'})
        g.add_location('A', Player.EVEN)
        g.add_location('B', Player.ODD)
        g.add_location('C', Player.EVEN)
        g.add_edge('A', 'B', 'ab')
        g.add_edge('A', 'C', 'ac')
        g.add_edge('B', 'C', 'bc')
        edges = g.get_edges_from('A')
        assert len(edges) == 2
        assert edges[0][1].target == 'B'
        assert edges[1][1].target == 'C'

    def test_get_invariant_default(self):
        g = TimedGame()
        g.add_location('A', Player.EVEN)
        inv = g.get_invariant('A')
        assert inv.is_true()

    def test_to_timed_automaton(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD},
            accepting={'B'}
        )
        g.add_edge('A', 'B', 'go')
        ta = g.to_timed_automaton()
        assert ta.initial == 'A'
        assert 'B' in ta.accepting

    def test_max_constant(self):
        g = TimedGame(clocks={'x'})
        g.add_location('A', Player.EVEN, clock_leq('x', 7))
        g.add_edge('A', 'A', 'loop', clock_geq('x', 3))
        assert g.max_constant() == 7


class TestPlayer:
    def test_opponent(self):
        assert Player.EVEN.opponent == Player.ODD
        assert Player.ODD.opponent == Player.EVEN


class TestZoneStore:
    def test_add_get(self):
        store = ZoneStore()
        z = initial_zone(['x'])
        idx = store.add(z)
        assert idx == 0
        assert store.get(0) is z
        assert len(store) == 1


# ===========================================================================
# Zone operation tests
# ===========================================================================

class TestZoneOps:
    def test_successor_zone_basic(self):
        edge = Edge('A', 'B', 'go', true_guard(), frozenset())
        z = initial_zone(['x'])
        z.future()
        z.canonicalize()
        succ = _successor_zone(z, edge, true_guard())
        assert succ is not None
        assert not succ.is_empty()

    def test_successor_zone_with_guard(self):
        edge = Edge('A', 'B', 'go', clock_geq('x', 5), frozenset())
        z = initial_zone(['x'])
        z.future()
        z.apply_guard(clock_leq('x', 3))
        z.canonicalize()
        # x in [0,3] -- guard requires x >= 5 -> should fail
        succ = _successor_zone(z, edge, true_guard())
        assert succ is None

    def test_successor_zone_with_reset(self):
        edge = Edge('A', 'B', 'go', true_guard(), frozenset({'x'}))
        z = initial_zone(['x'])
        z.future()
        z.canonicalize()
        succ = _successor_zone(z, edge, true_guard())
        assert succ is not None
        # After reset, x should be able to be 0
        sample = succ.get_sample()
        assert sample is not None

    def test_zone_past(self):
        z = initial_zone(['x'])
        z.future()
        z.apply_guard(clock_geq('x', 3))
        z.apply_guard(clock_leq('x', 5))
        z.canonicalize()
        past = _zone_past(z, ['x'])
        assert not past.is_empty()
        # Past should include x=0 (can reach [3,5] by waiting)
        sample = past.get_sample()
        assert sample is not None

    def test_zone_union_overapprox(self):
        z1 = initial_zone(['x'])
        z1.future()
        z1.apply_guard(clock_leq('x', 2))
        z1.canonicalize()

        z2 = initial_zone(['x'])
        z2.future()
        z2.apply_guard(clock_geq('x', 4))
        z2.apply_guard(clock_leq('x', 6))
        z2.canonicalize()

        union = _zone_union_overapprox(z1, z2, ['x'])
        assert not union.is_empty()
        # Should contain x=0, x=5, and also x=3 (over-approximation)

    def test_zone_union_empty_left(self):
        z1 = initial_zone(['x'])
        z1.constrain(ClockConstraint('x', None, CompOp.LT, 0))
        z1.canonicalize()
        assert z1.is_empty()

        z2 = initial_zone(['x'])
        z2.future()
        z2.canonicalize()

        union = _zone_union_overapprox(z1, z2, ['x'])
        assert not union.is_empty()


class TestExploreGame:
    def test_simple_exploration(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')
        reachable = _explore_game(g)
        assert len(reachable['A']) > 0
        assert len(reachable['B']) > 0

    def test_guarded_exploration(self):
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD, 'C': Player.EVEN},
            invariants={'A': clock_leq('x', 5)}
        )
        g.add_edge('A', 'B', 'ab', clock_leq('x', 3))
        g.add_edge('A', 'C', 'ac', clock_geq('x', 10))  # unreachable (inv x<=5)
        reachable = _explore_game(g)
        assert len(reachable['B']) > 0
        assert len(reachable['C']) == 0  # x can't be >= 10


# ===========================================================================
# Reachability game tests
# ===========================================================================

class TestReachability:
    def test_trivial_reach(self):
        """Even starts at target -> Even wins."""
        g = TimedGame(
            locations={'A'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN}
        )
        result = solve_reachability(g, {'A'})
        assert result.winner['A'] == Player.EVEN

    def test_one_step_reach(self):
        """Even can reach target in one step."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')
        result = solve_reachability(g, {'B'})
        assert result.winner['A'] == Player.EVEN
        assert result.winner['B'] == Player.EVEN  # target

    def test_even_choice(self):
        """Even chooses between two paths, one reaches target."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD, 'C': Player.ODD}
        )
        g.add_edge('A', 'B', 'good')  # B is target
        g.add_edge('A', 'C', 'bad')   # C is not
        result = solve_reachability(g, {'B'})
        assert result.winner['A'] == Player.EVEN

    def test_odd_blocks(self):
        """Odd controls the only path and can avoid target by looping."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.ODD, 'B': Player.ODD, 'C': Player.EVEN},
            invariants={'B': clock_leq('x', 10)}
        )
        g.add_edge('A', 'B', 'safe')  # Odd goes here (not target)
        g.add_edge('A', 'C', 'target')  # Odd avoids this
        g.add_edge('B', 'B', 'loop', true_guard(), frozenset({'x'}))  # Odd loops forever
        result = solve_reachability(g, {'C'})
        # Odd owns A and can choose 'safe' edge to B, then loop forever
        assert result.winner.get('A', Player.ODD) == Player.ODD

    def test_timed_reach(self):
        """Reachability with clock constraints."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD},
            invariants={'A': clock_leq('x', 5)}
        )
        g.add_edge('A', 'B', 'go', clock_geq('x', 2))
        result = solve_reachability(g, {'B'})
        assert result.winner['A'] == Player.EVEN

    def test_timed_reach_impossible(self):
        """Clock constraint makes target unreachable."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD},
            invariants={'A': clock_leq('x', 3)}
        )
        # Guard requires x >= 5 but invariant limits x <= 3
        g.add_edge('A', 'B', 'go', clock_geq('x', 5))
        result = solve_reachability(g, {'B'})
        assert result.winner['A'] == Player.ODD

    def test_two_step_reach(self):
        """Even reaches target in two steps."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN, 'C': Player.ODD}
        )
        g.add_edge('A', 'B', 'ab')
        g.add_edge('B', 'C', 'bc')
        result = solve_reachability(g, {'C'})
        assert result.winner['A'] == Player.EVEN

    def test_dead_end_odd_loses(self):
        """Odd at dead end -> Even wins."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')
        # B has no outgoing edges -- Odd is stuck
        result = solve_reachability(g, {'B'})
        assert result.winner['B'] == Player.EVEN

    def test_iterations_bounded(self):
        """Solver terminates within max iterations."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN, 'C': Player.ODD}
        )
        g.add_edge('A', 'B', 'ab', true_guard(), frozenset({'x'}))
        g.add_edge('B', 'C', 'bc')
        g.add_edge('B', 'A', 'ba', true_guard(), frozenset({'x'}))
        result = solve_reachability(g, {'C'}, max_iterations=50)
        assert result.iterations <= 50


# ===========================================================================
# Safety game tests
# ===========================================================================

class TestSafety:
    def test_trivial_safe(self):
        """No unsafe locations -> Even wins."""
        g = TimedGame(
            locations={'A'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN}
        )
        result = solve_safety(g, set())
        assert result.winner['A'] == Player.EVEN

    def test_start_unsafe(self):
        """Start location is unsafe -> Odd wins."""
        g = TimedGame(
            locations={'A'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN}
        )
        result = solve_safety(g, {'A'})
        assert result.winner['A'] == Player.ODD

    def test_even_avoids_unsafe(self):
        """Even can choose safe path."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD, 'C': Player.ODD}
        )
        g.add_edge('A', 'B', 'safe')
        g.add_edge('A', 'C', 'unsafe')
        result = solve_safety(g, {'C'})
        assert result.winner['A'] == Player.EVEN

    def test_forced_unsafe(self):
        """All paths from Even lead to unsafe."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')  # only option -> unsafe
        result = solve_safety(g, {'B'})
        assert result.winner['A'] == Player.ODD

    def test_timed_safety(self):
        """Safety with clock constraints -- safe as long as invariant holds."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD},
            invariants={'A': clock_leq('x', 10)}
        )
        g.add_edge('A', 'B', 'go', clock_geq('x', 5))
        g.add_edge('A', 'A', 'stay', clock_leq('x', 4), frozenset({'x'}))
        result = solve_safety(g, {'B'})
        # Even can loop at A by resetting x before 5
        assert result.winner['A'] == Player.EVEN


# ===========================================================================
# Buchi game tests
# ===========================================================================

class TestBuchi:
    def test_no_accepting(self):
        """No accepting states -> Odd wins."""
        g = TimedGame(
            locations={'A'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN}
        )
        result = solve_buchi(g, set())
        assert result.winner['A'] == Player.ODD

    def test_accepting_loop(self):
        """Even can loop through accepting state."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN},
            invariants={'A': clock_leq('x', 5), 'B': clock_leq('x', 5)}
        )
        g.add_edge('A', 'B', 'ab', clock_geq('x', 1), frozenset({'x'}))
        g.add_edge('B', 'A', 'ba', clock_geq('x', 1), frozenset({'x'}))
        result = solve_buchi(g, {'B'})
        assert result.winner['A'] == Player.EVEN

    def test_buchi_no_cycle(self):
        """One-shot reach of accepting, no loop back -> Odd wins."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN}
        )
        g.add_edge('A', 'B', 'go')
        # B has no edges back to accepting -> can't visit infinitely
        result = solve_buchi(g, {'B'})
        # B is a dead end -- can't revisit B infinitely
        assert result.winner.get('B', Player.ODD) == Player.ODD


# ===========================================================================
# Timed energy game tests
# ===========================================================================

class TestTimedEnergy:
    def test_positive_cycle(self):
        """Even can maintain positive energy via positive-weight cycle."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN},
            invariants={'A': clock_leq('x', 5), 'B': clock_leq('x', 5)}
        )
        g.add_edge('A', 'B', 'ab', clock_geq('x', 1), frozenset({'x'}), weight=3)
        g.add_edge('B', 'A', 'ba', clock_geq('x', 1), frozenset({'x'}), weight=-1)
        result = solve_timed_energy(g)
        assert 'A' in result.winning_locations

    def test_negative_only(self):
        """All edges negative -> Even eventually loses."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN},
            invariants={'A': clock_leq('x', 5), 'B': clock_leq('x', 5)}
        )
        g.add_edge('A', 'B', 'ab', clock_geq('x', 1), frozenset({'x'}), weight=-3)
        g.add_edge('B', 'A', 'ba', clock_geq('x', 1), frozenset({'x'}), weight=-2)
        result = solve_timed_energy(g)
        # With only negative weights and forced cycling, Even loses
        assert 'A' not in result.winning_locations

    def test_energy_min_value(self):
        """Check minimum energy computation."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN},
            invariants={'A': clock_leq('x', 5), 'B': clock_leq('x', 5)}
        )
        g.add_edge('A', 'B', 'ab', clock_geq('x', 1), frozenset({'x'}), weight=5)
        g.add_edge('B', 'A', 'ba', clock_geq('x', 1), frozenset({'x'}), weight=-2)
        result = solve_timed_energy(g)
        assert result.min_energy['A'] is not None
        assert result.min_energy['A'] >= 0


# ===========================================================================
# Simulation tests
# ===========================================================================

class TestSimulation:
    def test_simulate_basic(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        idx = g.add_edge('A', 'B', 'go')
        trace = simulate_play(g, {'A': (0.0, idx)}, {}, max_steps=5)
        assert len(trace.steps) == 1
        assert trace.steps[0][0] == 'A'
        assert trace.steps[0][3] == 'B'

    def test_simulate_with_delay(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD},
            invariants={'A': clock_leq('x', 10)}
        )
        idx = g.add_edge('A', 'B', 'go', clock_geq('x', 3))
        trace = simulate_play(g, {'A': (3.0, idx)}, {}, max_steps=5)
        assert len(trace.steps) == 1

    def test_simulate_guard_fail(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD},
            invariants={'A': clock_leq('x', 10)}
        )
        idx = g.add_edge('A', 'B', 'go', clock_geq('x', 5))
        # Delay only 2 -> guard fails
        trace = simulate_play(g, {'A': (2.0, idx)}, {}, max_steps=5)
        assert len(trace.steps) == 0

    def test_check_strategy(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        idx = g.add_edge('A', 'B', 'go')
        assert check_timed_strategy(g, {'B'}, {'A': (0.0, idx)})

    def test_check_strategy_fail(self):
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD, 'C': Player.EVEN}
        )
        idx = g.add_edge('A', 'B', 'wrong')
        # Goes to B not C (target)
        assert not check_timed_strategy(g, {'C'}, {'A': (0.0, idx)})


# ===========================================================================
# Example game tests
# ===========================================================================

class TestExampleGames:
    def test_cat_mouse_structure(self):
        g = cat_mouse_game()
        assert len(g.locations) == 5
        assert g.owner['start'] == Player.EVEN
        assert g.owner['path1'] == Player.ODD
        stats = game_statistics(g)
        assert stats['locations'] == 5
        assert stats['edges'] == 6

    def test_cat_mouse_reachability(self):
        g = cat_mouse_game()
        result = solve_reachability(g, {'hole'})
        # Mouse should be able to reach hole from start
        assert result.winner['start'] == Player.EVEN

    def test_cat_mouse_safety(self):
        g = cat_mouse_game()
        result = solve_safety(g, {'caught'})
        # Mouse can try to avoid being caught
        # (but cat controls paths, so it depends on timing)

    def test_resource_game_structure(self):
        g = resource_game()
        assert len(g.locations) == 4
        assert g.owner['idle'] == Player.EVEN
        assert g.owner['working'] == Player.ODD
        stats = game_statistics(g)
        assert stats['has_weights'] is True

    def test_resource_energy(self):
        g = resource_game()
        result = solve_timed_energy(g)
        # Controller should be able to manage energy by charging
        assert isinstance(result, TimedEnergyResult)

    def test_traffic_light_structure(self):
        g = traffic_light_game()
        assert len(g.locations) == 7
        assert g.owner['crash'] == Player.ODD
        stats = game_statistics(g)
        assert stats['even_locations'] == 6
        assert stats['odd_locations'] == 1

    def test_traffic_light_safety(self):
        g = traffic_light_game()
        result = solve_safety(g, {'crash'})
        # Controller should be able to avoid crash
        assert result.winner['green_ns'] == Player.EVEN

    def test_fischer_structure(self):
        g = fischer_game()
        assert len(g.locations) == 6
        assert g.owner['idle'] == Player.EVEN
        assert g.owner['interfere_point'] == Player.ODD

    def test_fischer_reachability(self):
        g = fischer_game()
        result = solve_reachability(g, {'cs'})
        # In adversarial model, Odd can always interfere -> Even can't guarantee CS
        # But Even wins at 'wait' (if past interference, can enter CS)
        assert result.winner['wait'] == Player.EVEN
        assert result.winner['cs'] == Player.EVEN
        # Odd can block at interfere_point
        assert result.winner['interfere_point'] == Player.ODD


# ===========================================================================
# Builder / parser tests
# ===========================================================================

class TestBuilder:
    def test_make_timed_game(self):
        g = make_timed_game(
            locations=[('A', 'even', 'x<=5'), ('B', 'odd', None)],
            edges=[('A', 'B', 'go', 'x>=2', 'x')],
            initial='A',
            clocks=['x']
        )
        assert 'A' in g.locations
        assert g.owner['A'] == Player.EVEN
        assert len(g.edges) == 1

    def test_parse_guard_leq(self):
        g = _parse_guard_str('x<=5')
        assert len(g.constraints) == 1
        assert g.constraints[0].op == CompOp.LE
        assert g.constraints[0].value == 5

    def test_parse_guard_geq(self):
        g = _parse_guard_str('x>=3')
        assert g.constraints[0].op == CompOp.GE
        assert g.constraints[0].value == 3

    def test_parse_guard_lt(self):
        g = _parse_guard_str('x<10')
        assert g.constraints[0].op == CompOp.LT

    def test_parse_guard_gt(self):
        g = _parse_guard_str('x>0')
        assert g.constraints[0].op == CompOp.GT

    def test_parse_guard_eq(self):
        g = _parse_guard_str('x==4')
        assert g.constraints[0].op == CompOp.EQ

    def test_parse_guard_conjunction(self):
        g = _parse_guard_str('x>=2 && x<=8')
        assert len(g.constraints) == 2

    def test_parse_diff_constraint(self):
        g = _parse_guard_str('x-y<=3')
        assert g.constraints[0].clock1 == 'x'
        assert g.constraints[0].clock2 == 'y'
        assert g.constraints[0].value == 3

    def test_make_game_with_resets(self):
        g = make_timed_game(
            locations=[('A', 'even', None), ('B', 'odd', None)],
            edges=[('A', 'B', 'go', None, 'x,y')],
            initial='A',
            clocks=['x', 'y']
        )
        assert 'x' in g.edges[0].resets
        assert 'y' in g.edges[0].resets


# ===========================================================================
# Statistics and summary tests
# ===========================================================================

class TestAnalysis:
    def test_game_statistics(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')
        stats = game_statistics(g)
        assert stats['locations'] == 2
        assert stats['even_locations'] == 1
        assert stats['odd_locations'] == 1
        assert stats['edges'] == 1
        assert stats['clocks'] == 1

    def test_game_summary(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')
        s = game_summary(g)
        assert 'Timed Game' in s
        assert '2 locations' in s

    def test_game_summary_energy(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go', weight=3)
        s = game_summary(g)
        assert 'Energy' in s or 'weighted' in s

    def test_compare_reach_safety(self):
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')
        comp = compare_reachability_safety(g, {'B'})
        assert 'reachability' in comp
        assert 'safety' in comp
        assert 'iterations' in comp['reachability']


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    def test_no_edges(self):
        """Game with no edges -- Even stuck at start."""
        g = TimedGame(
            locations={'A'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN}
        )
        result = solve_reachability(g, set())
        assert result.winner['A'] == Player.ODD

    def test_self_loop_even(self):
        """Even has self-loop only."""
        g = TimedGame(
            locations={'A'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN},
            invariants={'A': clock_leq('x', 5)}
        )
        g.add_edge('A', 'A', 'loop', true_guard(), frozenset({'x'}))
        result = solve_reachability(g, {'A'})
        assert result.winner['A'] == Player.EVEN

    def test_multi_clock(self):
        """Game with multiple clocks."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x', 'y'},
            owner={'A': Player.EVEN, 'B': Player.EVEN, 'C': Player.ODD},
            invariants={'A': clock_leq('x', 5), 'B': clock_leq('y', 3)}
        )
        g.add_edge('A', 'B', 'ab', clock_geq('x', 2), frozenset({'y'}))
        g.add_edge('B', 'C', 'bc', clock_geq('y', 1))
        result = solve_reachability(g, {'C'})
        assert result.winner['A'] == Player.EVEN

    def test_empty_reachable(self):
        """Unreachable location."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD, 'C': Player.EVEN}
        )
        g.add_edge('A', 'B', 'ab')
        # C is unreachable
        result = solve_reachability(g, {'C'})
        assert result.winner['C'] == Player.ODD

    def test_result_fields(self):
        """Check result structure."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.ODD}
        )
        g.add_edge('A', 'B', 'go')
        result = solve_reachability(g, {'B'})
        assert isinstance(result, TimedGameResult)
        assert isinstance(result.winner, dict)
        assert isinstance(result.winning_zones_even, dict)
        assert isinstance(result.winning_zones_odd, dict)
        assert isinstance(result.iterations, int)

    def test_energy_result_fields(self):
        """Check energy result structure."""
        g = TimedGame(
            locations={'A'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN}
        )
        result = solve_timed_energy(g)
        assert isinstance(result, TimedEnergyResult)
        assert isinstance(result.winning_locations, set)
        assert isinstance(result.min_energy, dict)


# ===========================================================================
# Regression / integration tests
# ===========================================================================

class TestIntegration:
    def test_reach_then_safety(self):
        """Solve same game for reachability and safety -- results consistent."""
        g = TimedGame(
            locations={'A', 'B', 'C'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN, 'C': Player.ODD},
            invariants={'A': clock_leq('x', 5), 'B': clock_leq('x', 5)}
        )
        g.add_edge('A', 'B', 'ab', clock_geq('x', 1), frozenset({'x'}))
        g.add_edge('B', 'C', 'bc', clock_geq('x', 2))
        g.add_edge('B', 'A', 'ba', clock_leq('x', 1), frozenset({'x'}))

        reach = solve_reachability(g, {'C'})
        safety = solve_safety(g, {'C'})

        # Both solvers should return valid results
        assert isinstance(reach, TimedGameResult)
        assert isinstance(safety, TimedGameResult)
        # Even controls A and B; it can choose to go to C (reach) or loop (safe)
        # Reachability: Even can reach C -> Even wins
        assert reach.winner.get('A') == Player.EVEN
        # Safety: Even can also choose to loop A->B->A avoiding C -> Even wins safety too
        assert safety.winner.get('A') == Player.EVEN

    def test_all_solvers(self):
        """Run all solvers on a game."""
        g = TimedGame(
            locations={'A', 'B'}, initial='A', clocks={'x'},
            owner={'A': Player.EVEN, 'B': Player.EVEN},
            invariants={'A': clock_leq('x', 5), 'B': clock_leq('x', 5)},
            accepting={'B'}
        )
        g.add_edge('A', 'B', 'ab', clock_geq('x', 1), frozenset({'x'}), weight=1)
        g.add_edge('B', 'A', 'ba', clock_geq('x', 1), frozenset({'x'}), weight=1)

        reach = solve_reachability(g, {'B'})
        safety = solve_safety(g, set())
        buchi = solve_buchi(g, {'B'})
        energy = solve_timed_energy(g)

        assert reach.winner['A'] == Player.EVEN
        assert safety.winner['A'] == Player.EVEN
        assert isinstance(buchi, TimedGameResult)
        assert isinstance(energy, TimedEnergyResult)

    def test_cat_mouse_full(self):
        """Full analysis of cat-mouse game."""
        g = cat_mouse_game()
        stats = game_statistics(g)
        summary = game_summary(g)
        reach = solve_reachability(g, {'hole'})
        safety = solve_safety(g, {'caught'})

        assert stats['locations'] == 5
        assert 'Timed Game' in summary
        assert isinstance(reach, TimedGameResult)
        assert isinstance(safety, TimedGameResult)

    def test_fischer_full(self):
        """Full analysis of Fischer's game."""
        g = fischer_game()
        reach = solve_reachability(g, {'cs'})
        safety = solve_safety(g, {'conflict'})
        summary = game_summary(g)

        assert isinstance(reach, TimedGameResult)
        assert isinstance(safety, TimedGameResult)
        assert 'Timed Game' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
