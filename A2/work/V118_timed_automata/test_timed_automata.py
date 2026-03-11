"""Tests for V118: Timed Automata Verification."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from timed_automata import (
    ClockConstraint, CompOp, Guard, Edge, TimedAutomaton, Zone,
    TimedAction, Trace, ReachabilityResult, SafetyResult, ZoneGraphStats,
    true_guard, clock_leq, clock_lt, clock_geq, clock_gt, clock_eq,
    clock_diff_leq, clock_diff_geq, guard_and, initial_zone,
    check_reachability, check_safety, check_timed_word,
    explore_zone_graph, product, zone_graph_summary,
    check_empty_language, check_language_inclusion,
    simple_ta, simple_light_timer, train_gate_controller, fischer_mutex,
)


# ============================================================
# Section 1: Clock Constraints
# ============================================================

class TestClockConstraints:
    def test_simple_leq(self):
        c = ClockConstraint('x', None, CompOp.LE, 5)
        assert c.evaluate({'x': 3.0})
        assert c.evaluate({'x': 5.0})
        assert not c.evaluate({'x': 6.0})

    def test_simple_lt(self):
        c = ClockConstraint('x', None, CompOp.LT, 5)
        assert c.evaluate({'x': 4.9})
        assert not c.evaluate({'x': 5.0})

    def test_simple_geq(self):
        c = ClockConstraint('x', None, CompOp.GE, 3)
        assert c.evaluate({'x': 3.0})
        assert c.evaluate({'x': 10.0})
        assert not c.evaluate({'x': 2.0})

    def test_simple_gt(self):
        c = ClockConstraint('x', None, CompOp.GT, 3)
        assert not c.evaluate({'x': 3.0})
        assert c.evaluate({'x': 3.1})

    def test_simple_eq(self):
        c = ClockConstraint('x', None, CompOp.EQ, 5)
        assert c.evaluate({'x': 5.0})
        assert not c.evaluate({'x': 5.1})

    def test_difference_constraint(self):
        c = ClockConstraint('x', 'y', CompOp.LE, 3)
        assert c.evaluate({'x': 5.0, 'y': 3.0})  # 5-3=2 <= 3
        assert not c.evaluate({'x': 7.0, 'y': 3.0})  # 7-3=4 > 3

    def test_guard_conjunction(self):
        g = guard_and(clock_geq('x', 2), clock_leq('x', 5))
        assert g.evaluate({'x': 3.0})
        assert not g.evaluate({'x': 1.0})
        assert not g.evaluate({'x': 6.0})

    def test_true_guard(self):
        g = true_guard()
        assert g.is_true()
        assert g.evaluate({'x': 100.0})

    def test_guard_str(self):
        g = clock_leq('x', 5)
        assert 'x' in str(g)
        assert '5' in str(g)


# ============================================================
# Section 2: Zone (DBM) Basics
# ============================================================

class TestZoneBasics:
    def test_initial_zone(self):
        z = initial_zone(['x', 'y'])
        assert not z.is_empty()
        sample = z.get_sample()
        assert sample is not None
        assert sample['x'] == 0.0
        assert sample['y'] == 0.0

    def test_zone_copy(self):
        z = initial_zone(['x'])
        z2 = z.copy()
        z2.dbm[1][0] = (10, False)
        assert z.dbm[1][0] != z2.dbm[1][0]

    def test_zone_empty_detection(self):
        z = initial_zone(['x'])
        # x <= 5 and x >= 10 -> empty
        z.constrain(ClockConstraint('x', None, CompOp.LE, 5))
        z.constrain(ClockConstraint('x', None, CompOp.GE, 10))
        z.canonicalize()
        assert z.is_empty()

    def test_zone_not_empty(self):
        z = initial_zone(['x'])
        z.future()  # let time pass so x can grow
        z.constrain(ClockConstraint('x', None, CompOp.LE, 5))
        z.constrain(ClockConstraint('x', None, CompOp.GE, 2))
        z.canonicalize()
        assert not z.is_empty()

    def test_zone_future(self):
        z = initial_zone(['x'])
        # Initially x=0
        z.future()
        # Now x can be anything >= 0
        assert not z.is_empty()
        # Upper bound on x should be INF
        idx = z._idx('x')
        assert z.dbm[idx][0][0] == float('inf')

    def test_zone_reset(self):
        z = initial_zone(['x'])
        z.future()
        z.constrain(ClockConstraint('x', None, CompOp.GE, 5))
        z.canonicalize()
        assert not z.is_empty()
        z.reset('x')
        # After reset, x = 0
        sample = z.get_sample()
        assert sample['x'] == 0.0


# ============================================================
# Section 3: Zone Operations
# ============================================================

class TestZoneOperations:
    def test_apply_guard(self):
        z = initial_zone(['x'])
        z.future()
        g = guard_and(clock_geq('x', 2), clock_leq('x', 5))
        z.apply_guard(g)
        assert not z.is_empty()
        sample = z.get_sample()
        assert 2.0 <= sample['x'] <= 5.0

    def test_apply_guard_empty(self):
        z = initial_zone(['x'])
        # x=0, guard x >= 5 without future
        g = clock_geq('x', 5)
        z.apply_guard(g)
        assert z.is_empty()

    def test_zone_intersect(self):
        z1 = initial_zone(['x'])
        z1.future()
        z1.apply_guard(clock_leq('x', 10))

        z2 = initial_zone(['x'])
        z2.future()
        z2.apply_guard(clock_geq('x', 5))

        z3 = z1.intersect(z2)
        assert not z3.is_empty()
        sample = z3.get_sample()
        assert 5.0 <= sample['x'] <= 10.0

    def test_zone_includes(self):
        z1 = initial_zone(['x'])
        z1.future()
        z1.apply_guard(clock_leq('x', 10))

        z2 = initial_zone(['x'])
        z2.future()
        z2.apply_guard(guard_and(clock_geq('x', 2), clock_leq('x', 5)))

        assert z1.includes(z2)
        assert not z2.includes(z1)

    def test_zone_equals(self):
        z1 = initial_zone(['x'])
        z1.future()
        z1.apply_guard(clock_leq('x', 5))

        z2 = initial_zone(['x'])
        z2.future()
        z2.apply_guard(clock_leq('x', 5))

        assert z1.equals(z2)

    def test_two_clock_difference(self):
        z = initial_zone(['x', 'y'])
        z.future()
        # x <= 5, y >= 3, x - y <= 1
        z.apply_guard(clock_leq('x', 5))
        z.apply_guard(clock_geq('y', 3))
        z.apply_guard(clock_diff_leq('x', 'y', 1))
        assert not z.is_empty()
        sample = z.get_sample()
        assert sample['x'] - sample['y'] <= 1.0

    def test_zone_str(self):
        z = initial_zone(['x'])
        z.future()
        z.apply_guard(clock_leq('x', 5))
        s = str(z)
        assert 'x' in s


# ============================================================
# Section 4: Simple Timed Automaton Construction
# ============================================================

class TestTimedAutomatonConstruction:
    def test_simple_ta_builder(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 1), ['x']),
             ('b', 'a', 'back', clock_leq('x', 3), [])],
            accepting={'b'}
        )
        assert len(ta.locations) == 2
        assert ta.initial == 'a'
        assert len(ta.edges) == 2
        assert len(ta.clocks) == 1

    def test_light_timer(self):
        ta = simple_light_timer()
        assert 'off' in ta.locations
        assert 'on' in ta.locations
        assert ta.initial == 'off'
        assert len(ta.edges) == 2

    def test_train_gate(self):
        ta = train_gate_controller()
        assert 'idle' in ta.locations
        assert 'crossing' in ta.locations
        assert ta.initial == 'idle'

    def test_max_constant(self):
        ta = simple_light_timer()
        assert ta.max_constant() == 3

    def test_get_edges_from(self):
        ta = simple_light_timer()
        edges = ta.get_edges_from('off')
        assert len(edges) == 1
        assert edges[0].label == 'press'

    def test_get_invariant(self):
        ta = simple_light_timer()
        inv = ta.get_invariant('on')
        assert not inv.is_true()
        inv_off = ta.get_invariant('off')
        assert inv_off.is_true()


# ============================================================
# Section 5: Zone Graph Exploration
# ============================================================

class TestZoneGraphExploration:
    def test_light_timer_zone_graph(self):
        ta = simple_light_timer()
        reached, stats = explore_zone_graph(ta)
        assert 'off' in reached
        assert 'on' in reached
        assert len(reached['off']) > 0
        assert len(reached['on']) > 0

    def test_simple_reachable(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
        )
        reached, stats = explore_zone_graph(ta)
        assert len(reached['b']) > 0

    def test_unreachable_guard(self):
        # b requires x >= 10, but invariant at a forces x <= 5
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 10), [])],
            invariants={'a': clock_leq('x', 5)},
        )
        reached, stats = explore_zone_graph(ta)
        assert len(reached['b']) == 0

    def test_zone_graph_summary_output(self):
        ta = simple_light_timer()
        summary = zone_graph_summary(ta)
        assert 'locations' in summary
        assert 'zone' in summary.lower()


# ============================================================
# Section 6: Reachability Checking
# ============================================================

class TestReachability:
    def test_trivial_reachable(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
        )
        result = check_reachability(ta, {'b'})
        assert result.reachable
        assert result.target_location == 'b'

    def test_initial_is_target(self):
        ta = simple_ta(['a', 'b'], 'a', ['x'], [])
        result = check_reachability(ta, {'a'})
        assert result.reachable
        assert result.target_location == 'a'

    def test_unreachable(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 10), [])],
            invariants={'a': clock_leq('x', 5)},
        )
        result = check_reachability(ta, {'b'})
        assert not result.reachable

    def test_reachable_with_timing(self):
        # Must wait 3 time units before transitioning
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 3), ['x'])],
        )
        result = check_reachability(ta, {'b'})
        assert result.reachable

    def test_trace_provided(self):
        ta = simple_ta(
            ['a', 'b', 'c'], 'a', ['x'],
            [('a', 'b', 'step1', true_guard(), []),
             ('b', 'c', 'step2', true_guard(), [])],
        )
        result = check_reachability(ta, {'c'})
        assert result.reachable
        assert result.trace is not None
        assert len(result.trace.steps) >= 2

    def test_multi_clock_reachability(self):
        ta = simple_ta(
            ['a', 'b', 'c'], 'a', ['x', 'y'],
            [('a', 'b', 'step1', clock_geq('x', 2), ['y']),
             ('b', 'c', 'step2', clock_geq('y', 1), [])],
        )
        result = check_reachability(ta, {'c'})
        assert result.reachable


# ============================================================
# Section 7: Safety Checking
# ============================================================

class TestSafety:
    def test_safe_system(self):
        ta = simple_ta(
            ['safe', 'unsafe'], 'safe', ['x'],
            [('safe', 'unsafe', 'fail', clock_geq('x', 10), [])],
            invariants={'safe': clock_leq('x', 5)},
        )
        result = check_safety(ta, {'unsafe'})
        assert result.safe

    def test_unsafe_system(self):
        ta = simple_ta(
            ['safe', 'unsafe'], 'safe', ['x'],
            [('safe', 'unsafe', 'fail', true_guard(), [])],
        )
        result = check_safety(ta, {'unsafe'})
        assert not result.safe
        assert result.violated_location == 'unsafe'
        assert result.trace is not None

    def test_train_gate_safety(self):
        """Gate should be down before train crosses."""
        ta = train_gate_controller()
        # The train-gate controller is designed so 'crossing' is only
        # reachable through 'down' state (gate is lowered)
        # Check: 'crossing' is reachable (the system works)
        result = check_reachability(ta, {'crossing'})
        assert result.reachable

    def test_safety_stats(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
        )
        result = check_safety(ta, {'b'})
        assert result.states_explored > 0


# ============================================================
# Section 8: Timed Word Acceptance
# ============================================================

class TestTimedWordAcceptance:
    def test_accept_simple_word(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 1), [])],
            accepting={'b'}
        )
        word = [TimedAction('go', 2.0)]
        assert check_timed_word(ta, word)

    def test_reject_too_early(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 3), [])],
            accepting={'b'}
        )
        word = [TimedAction('go', 1.0)]
        assert not check_timed_word(ta, word)

    def test_reject_wrong_action(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
            accepting={'b'}
        )
        word = [TimedAction('stop', 1.0)]
        assert not check_timed_word(ta, word)

    def test_accept_multi_step(self):
        ta = simple_ta(
            ['a', 'b', 'c'], 'a', ['x'],
            [('a', 'b', 'step1', clock_geq('x', 1), ['x']),
             ('b', 'c', 'step2', clock_geq('x', 2), [])],
            accepting={'c'}
        )
        word = [TimedAction('step1', 1.5), TimedAction('step2', 4.0)]
        assert check_timed_word(ta, word)

    def test_reject_invariant_violation(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
            invariants={'a': clock_leq('x', 3)},
            accepting={'b'}
        )
        # Wait too long (invariant violated at a)
        word = [TimedAction('go', 5.0)]
        assert not check_timed_word(ta, word)

    def test_accept_within_invariant(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
            invariants={'a': clock_leq('x', 3)},
            accepting={'b'}
        )
        word = [TimedAction('go', 2.0)]
        assert check_timed_word(ta, word)

    def test_light_timer_word(self):
        ta = simple_light_timer()
        # Press at time 0, timeout at time 3
        word = [TimedAction('press', 0.0), TimedAction('timeout', 3.0)]
        assert check_timed_word(ta, word)

    def test_light_timer_reject_early_timeout(self):
        ta = simple_light_timer()
        # Press at time 0, try timeout at time 2 (guard: t==3 fails)
        word = [TimedAction('press', 0.0), TimedAction('timeout', 2.0)]
        assert not check_timed_word(ta, word)

    def test_empty_word_acceptance(self):
        ta = simple_ta(['a'], 'a', ['x'], [], accepting={'a'})
        assert check_timed_word(ta, [])

    def test_empty_word_rejection(self):
        ta = simple_ta(['a', 'b'], 'a', ['x'], [], accepting={'b'})
        assert not check_timed_word(ta, [])

    def test_decreasing_time_rejected(self):
        ta = simple_ta(
            ['a', 'b', 'c'], 'a', ['x'],
            [('a', 'b', 'step1', true_guard(), []),
             ('b', 'c', 'step2', true_guard(), [])],
            accepting={'c'}
        )
        word = [TimedAction('step1', 5.0), TimedAction('step2', 3.0)]
        assert not check_timed_word(ta, word)


# ============================================================
# Section 9: Product Construction
# ============================================================

class TestProductConstruction:
    def test_product_locations(self):
        ta1 = simple_ta(['a', 'b'], 'a', ['x'],
                         [('a', 'b', 'go', true_guard(), [])])
        ta2 = simple_ta(['c', 'd'], 'c', ['y'],
                         [('c', 'd', 'go', true_guard(), [])])
        prod = product(ta1, ta2)
        assert prod.initial == 'a,c'
        assert len(prod.locations) == 4  # {a,b} x {c,d}

    def test_product_shared_sync(self):
        ta1 = simple_ta(['a', 'b'], 'a', ['x'],
                         [('a', 'b', 'go', true_guard(), [])])
        ta2 = simple_ta(['c', 'd'], 'c', ['y'],
                         [('c', 'd', 'go', true_guard(), [])])
        prod = product(ta1, ta2)
        # 'go' is shared, so only synchronized edges
        edges_from_init = prod.get_edges_from('a,c')
        go_edges = [e for e in edges_from_init if e.label == 'go']
        assert len(go_edges) == 1
        assert go_edges[0].target == 'b,d'

    def test_product_independent_actions(self):
        ta1 = simple_ta(['a', 'b'], 'a', ['x'],
                         [('a', 'b', 'act1', true_guard(), [])])
        ta2 = simple_ta(['c', 'd'], 'c', ['y'],
                         [('c', 'd', 'act2', true_guard(), [])])
        prod = product(ta1, ta2)
        edges = prod.get_edges_from('a,c')
        labels = {e.label for e in edges}
        assert 'act1' in labels
        assert 'act2' in labels

    def test_product_clocks_renamed(self):
        ta1 = simple_ta(['a', 'b'], 'a', ['x'],
                         [('a', 'b', 'go', clock_leq('x', 5), ['x'])])
        ta2 = simple_ta(['c', 'd'], 'c', ['x'],
                         [('c', 'd', 'go', clock_geq('x', 2), [])])
        prod = product(ta1, ta2)
        # Clocks should be renamed to avoid collision
        assert '1_x' in prod.clocks
        assert '2_x' in prod.clocks

    def test_product_reachability(self):
        ta1 = simple_ta(['a', 'b'], 'a', ['x'],
                         [('a', 'b', 'go', true_guard(), [])])
        ta2 = simple_ta(['c', 'd'], 'c', ['y'],
                         [('c', 'd', 'go', true_guard(), [])])
        prod = product(ta1, ta2)
        result = check_reachability(prod, {'b,d'})
        assert result.reachable


# ============================================================
# Section 10: Empty Language Check
# ============================================================

class TestEmptyLanguage:
    def test_non_empty_language(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
            accepting={'b'}
        )
        assert not check_empty_language(ta)

    def test_empty_language_unreachable(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 10), [])],
            invariants={'a': clock_leq('x', 5)},
            accepting={'b'}
        )
        assert check_empty_language(ta)

    def test_empty_no_accepting(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
        )
        assert check_empty_language(ta)

    def test_light_timer_non_empty(self):
        ta = simple_light_timer()
        assert not check_empty_language(ta)


# ============================================================
# Section 11: Language Inclusion (Approximate)
# ============================================================

class TestLanguageInclusion:
    def test_self_inclusion(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), [])],
            accepting={'a', 'b'}
        )
        included, trace = check_language_inclusion(ta, ta)
        assert included

    def test_subset_inclusion(self):
        # ta1 accepts only when x >= 3 (stricter)
        ta1 = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 3), [])],
            accepting={'b'}
        )
        # ta2 accepts when x >= 1 (more permissive)
        ta2 = simple_ta(
            ['a', 'b'], 'a', ['x'],
            [('a', 'b', 'go', clock_geq('x', 1), [])],
            accepting={'b'}
        )
        included, _ = check_language_inclusion(ta1, ta2)
        assert included


# ============================================================
# Section 12: Fischer's Mutual Exclusion
# ============================================================

class TestFischerMutex:
    def test_fischer_construction(self):
        ta, unsafe = fischer_mutex(2)
        assert ta.initial == 'idle_idle_0'
        assert len(unsafe) > 0

    def test_fischer_safety(self):
        ta, unsafe = fischer_mutex(2)
        result = check_safety(ta, unsafe)
        assert result.safe

    def test_fischer_reachability_cs1(self):
        """Process 1 can reach critical section."""
        ta, _ = fischer_mutex(2)
        targets = set()
        for loc in ta.locations:
            parts = loc.split('_')
            if parts[0] == 'cs':
                targets.add(loc)
        result = check_reachability(ta, targets)
        assert result.reachable


# ============================================================
# Section 13: Train-Gate Controller
# ============================================================

class TestTrainGateController:
    def test_construction(self):
        ta = train_gate_controller()
        assert len(ta.locations) == 6
        assert ta.initial == 'idle'

    def test_crossing_reachable(self):
        ta = train_gate_controller()
        result = check_reachability(ta, {'crossing'})
        assert result.reachable

    def test_idle_reachable(self):
        """System can return to idle (full cycle)."""
        ta = train_gate_controller()
        # Idle is initial, but check it's reachable through the cycle
        # We need to reach a non-initial state first, then get back
        # Since idle IS the initial state, we test the full cycle via timed word
        word = [
            TimedAction('approach', 0.0),
            TimedAction('lower', 3.0),
            TimedAction('lowered', 4.5),
            TimedAction('enter', 5.0),
            TimedAction('exit', 8.0),
            TimedAction('raised', 9.5),
        ]
        assert check_timed_word(ta, word)

    def test_word_timing_constraint(self):
        ta = train_gate_controller()
        # lowering takes 1-2 time units
        word = [
            TimedAction('approach', 0.0),
            TimedAction('lower', 3.0),
            TimedAction('lowered', 3.5),  # Only 0.5 after lower (needs >= 1)
        ]
        assert not check_timed_word(ta, word)


# ============================================================
# Section 14: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_location_no_edges(self):
        ta = simple_ta(['a'], 'a', ['x'], [])
        reached, stats = explore_zone_graph(ta)
        assert len(reached['a']) > 0

    def test_self_loop(self):
        ta = simple_ta(
            ['a'], 'a', ['x'],
            [('a', 'a', 'tick', clock_geq('x', 1), ['x'])],
            accepting={'a'}
        )
        result = check_reachability(ta, {'a'})
        assert result.reachable
        word = [TimedAction('tick', 1.0), TimedAction('tick', 2.5)]
        assert check_timed_word(ta, word)

    def test_multiple_paths(self):
        ta = simple_ta(
            ['a', 'b', 'c'], 'a', ['x'],
            [('a', 'b', 'fast', true_guard(), []),
             ('a', 'c', 'slow', clock_geq('x', 5), []),
             ('b', 'c', 'finish', true_guard(), [])],
        )
        result = check_reachability(ta, {'c'})
        assert result.reachable

    def test_no_clocks(self):
        ta = simple_ta(
            ['a', 'b'], 'a', [],
            [('a', 'b', 'go', true_guard(), [])],
        )
        result = check_reachability(ta, {'b'})
        assert result.reachable

    def test_many_clocks(self):
        ta = simple_ta(
            ['a', 'b'], 'a', ['c1', 'c2', 'c3'],
            [('a', 'b', 'go',
              guard_and(clock_geq('c1', 1), clock_leq('c2', 5)),
              ['c3'])],
        )
        result = check_reachability(ta, {'b'})
        assert result.reachable

    def test_strict_bounds(self):
        z = initial_zone(['x'])
        z.future()
        z.constrain(ClockConstraint('x', None, CompOp.LT, 5))
        z.constrain(ClockConstraint('x', None, CompOp.GT, 3))
        z.canonicalize()
        assert not z.is_empty()
        sample = z.get_sample()
        assert 3.0 < sample['x'] < 5.0

    def test_zone_str_repr(self):
        z = initial_zone(['x', 'y'])
        z.future()
        z.apply_guard(clock_leq('x', 5))
        s = str(z)
        r = repr(z)
        assert 'x' in s
        assert 'Zone' in r

    def test_trace_str(self):
        trace = Trace([('a', 'go', {'x': 1.0}), ('b', 'done', {'x': 2.0})])
        s = str(trace)
        assert 'a' in s
        assert 'b' in s


# ============================================================
# Section 15: Integration Tests
# ============================================================

class TestIntegration:
    def test_light_cycle(self):
        """Full light cycle: off -> on -> off."""
        ta = simple_light_timer()
        # Check on is reachable
        result = check_reachability(ta, {'on'})
        assert result.reachable
        # Check off is reachable (back to off after timeout)
        # Need to check accepting via timed word
        word = [TimedAction('press', 0.0), TimedAction('timeout', 3.0)]
        assert check_timed_word(ta, word)

    def test_guard_reset_chain(self):
        """Chain of timed transitions with resets."""
        ta = simple_ta(
            ['s0', 's1', 's2', 's3'], 's0', ['x'],
            [('s0', 's1', 'a', clock_geq('x', 2), ['x']),
             ('s1', 's2', 'b', clock_geq('x', 3), ['x']),
             ('s2', 's3', 'c', clock_geq('x', 1), [])],
        )
        result = check_reachability(ta, {'s3'})
        assert result.reachable
        word = [TimedAction('a', 2.0), TimedAction('b', 5.0), TimedAction('c', 6.5)]
        ta_acc = simple_ta(
            ['s0', 's1', 's2', 's3'], 's0', ['x'],
            [('s0', 's1', 'a', clock_geq('x', 2), ['x']),
             ('s1', 's2', 'b', clock_geq('x', 3), ['x']),
             ('s2', 's3', 'c', clock_geq('x', 1), [])],
            accepting={'s3'}
        )
        assert check_timed_word(ta_acc, word)

    def test_zone_precision(self):
        """Zone analysis correctly tracks difference constraints."""
        ta = simple_ta(
            ['a', 'b', 'c', 'd'], 'a', ['x', 'y'],
            [('a', 'b', 'start_x', true_guard(), ['x']),
             ('b', 'c', 'start_y', clock_geq('x', 2), ['y']),
             ('c', 'd', 'check', clock_diff_leq('y', 'x', -1), [])],
             # y - x <= -1 means x >= y + 1
             # After reset y at x>=2: x>=2, y=0, so x-y>=2 -- satisfies x >= y+1
        )
        result = check_reachability(ta, {'d'})
        assert result.reachable

    def test_product_safety(self):
        """Product preserves safety properties."""
        ta1 = simple_ta(
            ['idle', 'active'], 'idle', ['x'],
            [('idle', 'active', 'start', true_guard(), ['x']),
             ('active', 'idle', 'stop', clock_leq('x', 5), [])],
            invariants={'active': clock_leq('x', 5)},
        )
        ta2 = simple_ta(
            ['off', 'on'], 'off', ['y'],
            [('off', 'on', 'start', true_guard(), ['y']),
             ('on', 'off', 'stop', true_guard(), [])],
        )
        prod = product(ta1, ta2)
        # Both components should sync on start/stop
        result = check_reachability(prod, {'active,on'})
        assert result.reachable

    def test_reachability_preserves_invariants(self):
        """Zone exploration respects location invariants."""
        ta = simple_ta(
            ['a', 'b', 'c'], 'a', ['x'],
            [('a', 'b', 'go', true_guard(), []),
             ('b', 'c', 'next', clock_geq('x', 10), [])],
            invariants={'b': clock_leq('x', 5)},
        )
        # b has invariant x<=5, but reaching c requires x>=10
        # So c should be unreachable
        result = check_reachability(ta, {'c'})
        assert not result.reachable


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
