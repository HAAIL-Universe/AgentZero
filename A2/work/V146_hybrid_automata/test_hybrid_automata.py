"""
Tests for V146: Hybrid Automata Verification
"""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_automata import (
    # Constraints
    CompOp, LinearExpr, LinearConstraint, Predicate,
    var_expr, const_expr, linear_expr,
    var_leq, var_geq, var_lt, var_gt, var_eq, diff_leq, diff_geq,
    true_pred, pred_and, pred_from,
    # Flow
    FlowInterval, Flow, CLOCK_RATE, STOPPED,
    flow_from, clock_flow, stopped_flow,
    # Reset
    Reset, reset_to, reset_vars, no_reset,
    # Hybrid automaton
    HybridEdge, HybridAutomaton,
    # Zone
    RectZone, initial_zone,
    # Reachability
    check_reachability, check_safety,
    # Simulation
    simulate, simulate_step,
    # Product
    product,
    # Examples
    thermostat, water_tank, railroad_crossing, bouncing_ball, two_tank,
    # Verification
    verify_safety, verify_invariant, verify_bounded_liveness,
    analyze_modes, zone_graph_summary, compare_hybrid_vs_timed, batch_verify,
    VerifyResult,
)


# ============================================================
# Section 1: Linear Constraints
# ============================================================

class TestLinearConstraints:
    def test_var_expr(self):
        e = var_expr('x')
        assert e.evaluate({'x': 5.0}) == 5.0
        assert e.evaluate({'x': -3.0}) == -3.0

    def test_const_expr(self):
        e = const_expr(42.0)
        assert e.evaluate({}) == 42.0

    def test_linear_expr(self):
        e = linear_expr({'x': 2.0, 'y': -1.0}, 3.0)
        assert e.evaluate({'x': 5.0, 'y': 2.0}) == 2*5 - 2 + 3

    def test_var_leq(self):
        c = var_leq('x', 10.0)
        assert c.evaluate({'x': 5.0})
        assert c.evaluate({'x': 10.0})
        assert not c.evaluate({'x': 11.0})

    def test_var_geq(self):
        c = var_geq('x', 5.0)
        assert c.evaluate({'x': 5.0})
        assert c.evaluate({'x': 10.0})
        assert not c.evaluate({'x': 4.0})

    def test_var_lt(self):
        c = var_lt('x', 10.0)
        assert c.evaluate({'x': 9.0})
        assert not c.evaluate({'x': 10.0})

    def test_var_gt(self):
        c = var_gt('x', 5.0)
        assert c.evaluate({'x': 6.0})
        assert not c.evaluate({'x': 5.0})

    def test_var_eq(self):
        c = var_eq('x', 5.0)
        assert c.evaluate({'x': 5.0})
        assert not c.evaluate({'x': 5.1})

    def test_diff_constraint(self):
        c = diff_leq('x', 'y', 3.0)
        assert c.evaluate({'x': 5.0, 'y': 3.0})  # 5 - 3 = 2 <= 3
        assert not c.evaluate({'x': 7.0, 'y': 3.0})  # 7 - 3 = 4 > 3

    def test_predicate_conjunction(self):
        p = pred_from(var_geq('x', 0), var_leq('x', 10))
        assert p.evaluate({'x': 5.0})
        assert not p.evaluate({'x': -1.0})
        assert not p.evaluate({'x': 11.0})

    def test_predicate_true(self):
        p = true_pred()
        assert p.is_true()
        assert p.evaluate({'x': 999.0})

    def test_predicate_and(self):
        p1 = pred_from(var_geq('x', 0))
        p2 = pred_from(var_leq('x', 10))
        p = pred_and(p1, p2)
        assert len(p.constraints) == 2
        assert p.evaluate({'x': 5.0})

    def test_constraint_variables(self):
        c = diff_leq('x', 'y', 3.0)
        assert c.variables() == {'x', 'y'}

    def test_predicate_variables(self):
        p = pred_from(var_geq('x', 0), var_leq('y', 10))
        assert p.variables() == {'x', 'y'}


# ============================================================
# Section 2: Flow Dynamics
# ============================================================

class TestFlowDynamics:
    def test_clock_rate(self):
        assert CLOCK_RATE.is_clock()
        assert not CLOCK_RATE.is_stopped()
        assert CLOCK_RATE.is_exact()

    def test_stopped_rate(self):
        assert STOPPED.is_stopped()
        assert not STOPPED.is_clock()
        assert STOPPED.is_exact()

    def test_interval_rate(self):
        r = FlowInterval(1.0, 2.0)
        assert not r.is_clock()
        assert not r.is_stopped()
        assert not r.is_exact()

    def test_flow_from_exact(self):
        f = flow_from(x=1.0, y=-0.5)
        assert f.get_rate('x') == FlowInterval(1.0, 1.0)
        assert f.get_rate('y') == FlowInterval(-0.5, -0.5)

    def test_flow_from_interval(self):
        f = flow_from(x=(1, 2))
        assert f.get_rate('x') == FlowInterval(1.0, 2.0)

    def test_flow_default_stopped(self):
        f = flow_from(x=1.0)
        assert f.get_rate('y') == STOPPED

    def test_clock_flow(self):
        f = clock_flow('x', 'y')
        assert f.get_rate('x') == CLOCK_RATE
        assert f.get_rate('y') == CLOCK_RATE


# ============================================================
# Section 3: Reset Operations
# ============================================================

class TestReset:
    def test_reset_to_constant(self):
        r = reset_to('x', 0.0)
        result = r.apply({'x': 5.0, 'y': 3.0})
        assert result['x'] == 0.0
        assert result['y'] == 3.0

    def test_reset_vars(self):
        r = reset_vars(x=0.0, y=1.0)
        result = r.apply({'x': 5.0, 'y': 3.0})
        assert result['x'] == 0.0
        assert result['y'] == 1.0

    def test_no_reset(self):
        r = no_reset()
        vals = {'x': 5.0, 'y': 3.0}
        result = r.apply(vals)
        assert result == vals

    def test_reset_copy(self):
        r = Reset((('x', 'y', 1.0),))  # x := y + 1
        result = r.apply({'x': 0, 'y': 5.0})
        assert result['x'] == 6.0
        assert result['y'] == 5.0


# ============================================================
# Section 4: RectZone (Extended DBM)
# ============================================================

class TestRectZone:
    def test_empty_zone(self):
        z = RectZone(['x'])
        assert not z.is_empty()

    def test_constrain_leq(self):
        z = RectZone(['x'])
        z = z.constrain(var_leq('x', 10))
        lo, hi = z.get_bounds('x')
        assert hi == 10.0

    def test_constrain_geq(self):
        z = RectZone(['x'])
        z = z.constrain(var_geq('x', 5))
        lo, hi = z.get_bounds('x')
        assert lo == 5.0

    def test_constrain_eq(self):
        z = RectZone(['x'])
        z = z.constrain(var_eq('x', 7))
        lo, hi = z.get_bounds('x')
        assert lo == 7.0
        assert hi == 7.0

    def test_constrain_empty(self):
        z = RectZone(['x'])
        z = z.constrain(var_geq('x', 10))
        z = z.constrain(var_leq('x', 5))
        assert z.is_empty()

    def test_constrain_predicate(self):
        z = RectZone(['x', 'y'])
        p = pred_from(var_geq('x', 0), var_leq('x', 10), var_geq('y', 0), var_leq('y', 5))
        z = z.constrain_pred(p)
        assert z.get_bounds('x') == (0, 10)
        assert z.get_bounds('y') == (0, 5)
        assert not z.is_empty()

    def test_difference_constraint(self):
        z = RectZone(['x', 'y'])
        z = z.constrain(var_geq('x', 0))
        z = z.constrain(var_geq('y', 0))
        z = z.constrain(diff_leq('x', 'y', 3))
        # x - y <= 3
        lo_x, hi_x = z.get_bounds('x')
        assert not z.is_empty()

    def test_time_elapse_clock(self):
        z = RectZone(['x'])
        z = z.constrain(var_eq('x', 0))
        f = clock_flow('x')
        z2 = z.time_elapse(f)
        # After time elapse, x >= 0 (was 0, grows at rate 1)
        lo, hi = z2.get_bounds('x')
        assert lo == 0.0
        assert hi == float('inf')

    def test_time_elapse_negative_rate(self):
        z = RectZone(['x'])
        z = z.constrain(var_eq('x', 10))
        f = flow_from(x=-1)
        z2 = z.time_elapse(f)
        lo, hi = z2.get_bounds('x')
        assert hi == 10.0  # can't grow (rate -1)
        assert lo == -float('inf')  # can decrease

    def test_time_elapse_stopped(self):
        z = RectZone(['x'])
        z = z.constrain(var_eq('x', 5))
        f = stopped_flow()
        z2 = z.time_elapse(f)
        lo, hi = z2.get_bounds('x')
        assert lo == 5.0
        assert hi == 5.0

    def test_time_elapse_different_rates(self):
        z = RectZone(['x', 'y'])
        z = z.constrain(var_eq('x', 0))
        z = z.constrain(var_eq('y', 0))
        f = flow_from(x=1, y=2)
        z2 = z.time_elapse(f)
        # Both grow from 0, but at different rates
        # x >= 0, y >= 0 (lower bounds preserved)
        lo_x, _ = z2.get_bounds('x')
        lo_y, _ = z2.get_bounds('y')
        assert lo_x == 0.0
        assert lo_y == 0.0

    def test_time_elapse_same_rate_preserves_diff(self):
        z = RectZone(['x', 'y'])
        z = z.constrain(var_eq('x', 0))
        z = z.constrain(var_eq('y', 5))
        f = flow_from(x=1, y=1)
        z2 = z.time_elapse(f)
        # Same rate: x - y = -5 preserved
        # dbm[idx_x][idx_y] should still be -5
        idx_x = z2._idx('x')
        idx_y = z2._idx('y')
        assert z2.dbm[idx_x][idx_y] == (-5, False)

    def test_reset_constant(self):
        z = RectZone(['x', 'y'])
        z = z.constrain(var_eq('x', 5))
        z = z.constrain(var_eq('y', 3))
        z2 = z.reset(reset_to('x', 0))
        lo, hi = z2.get_bounds('x')
        assert lo == 0.0
        assert hi == 0.0
        lo_y, hi_y = z2.get_bounds('y')
        assert lo_y == 3.0
        assert hi_y == 3.0

    def test_includes(self):
        z1 = RectZone(['x'])
        z1 = z1.constrain(var_geq('x', 0))
        z1 = z1.constrain(var_leq('x', 10))

        z2 = RectZone(['x'])
        z2 = z2.constrain(var_geq('x', 2))
        z2 = z2.constrain(var_leq('x', 8))

        assert z1.includes(z2)
        assert not z2.includes(z1)

    def test_intersect(self):
        z1 = RectZone(['x'])
        z1 = z1.constrain(var_geq('x', 0))
        z1 = z1.constrain(var_leq('x', 10))

        z2 = RectZone(['x'])
        z2 = z2.constrain(var_geq('x', 5))
        z2 = z2.constrain(var_leq('x', 15))

        z3 = z1.intersect(z2)
        assert z3.get_bounds('x') == (5, 10)

    def test_sample(self):
        z = RectZone(['x'])
        z = z.constrain(var_geq('x', 2))
        z = z.constrain(var_leq('x', 8))
        s = z.sample()
        assert s is not None
        assert 2.0 <= s['x'] <= 8.0

    def test_sample_empty(self):
        z = RectZone(['x'])
        z = z.constrain(var_geq('x', 10))
        z = z.constrain(var_leq('x', 5))
        assert z.sample() is None

    def test_copy(self):
        z = RectZone(['x'])
        z = z.constrain(var_leq('x', 10))
        z2 = z.copy()
        z2 = z2.constrain(var_leq('x', 5))
        assert z.get_bounds('x')[1] == 10
        assert z2.get_bounds('x')[1] == 5


# ============================================================
# Section 5: Hybrid Automaton Structure
# ============================================================

class TestHybridAutomaton:
    def test_thermostat_creation(self):
        ha = thermostat()
        assert 'heat' in ha.modes
        assert 'cool' in ha.modes
        assert ha.initial_mode == 'heat'
        assert 'temp' in ha.variables

    def test_thermostat_is_rectangular(self):
        ha = thermostat()
        assert ha.is_rectangular()

    def test_water_tank_creation(self):
        ha = water_tank()
        assert 'filling' in ha.modes
        assert 'draining' in ha.modes

    def test_get_edges_from(self):
        ha = thermostat()
        edges = ha.get_edges_from('heat')
        assert len(edges) == 1
        assert edges[0].target == 'cool'

    def test_get_flow(self):
        ha = thermostat()
        f = ha.get_flow('heat')
        r = f.get_rate('temp')
        assert r.lo == 1.0 and r.hi == 1.0

    def test_get_invariant(self):
        ha = thermostat()
        inv = ha.get_invariant('heat')
        assert not inv.is_true()  # temp <= 22

    def test_max_constant(self):
        ha = thermostat()
        assert ha.max_constant() >= 18.0

    def test_two_tank_creation(self):
        ha = two_tank()
        assert 'fill1' in ha.modes
        assert 'fill2' in ha.modes
        assert 'h1' in ha.variables
        assert 'h2' in ha.variables


# ============================================================
# Section 6: Simulation
# ============================================================

class TestSimulation:
    def test_simulate_step(self):
        ha = thermostat()
        vals = simulate_step(ha, 'heat', {'temp': 18.0}, 2.0)
        assert vals['temp'] == 20.0  # 18 + 1*2

    def test_simulate_cooling(self):
        ha = thermostat()
        vals = simulate_step(ha, 'cool', {'temp': 22.0}, 2.0)
        assert vals['temp'] == 21.0  # 22 + (-0.5)*2

    def test_simulate_trajectory(self):
        ha = thermostat()
        trace = simulate(ha, {'temp': 18.0}, [
            (4.0, 'off'),   # heat for 4 units: 18 -> 22
            (8.0, 'on'),    # cool for 8 units: 22 + (-0.5)*8 = 18
        ])
        assert len(trace.steps) == 3  # 2 transitions + final state
        assert trace.reaches_target

    def test_simulate_invalid_edge(self):
        ha = thermostat()
        trace = simulate(ha, {'temp': 18.0}, [
            (1.0, 'off'),  # try to turn off at temp=19, guard requires >= 22
        ])
        assert not trace.reaches_target


# ============================================================
# Section 7: Reachability Analysis
# ============================================================

class TestReachability:
    def test_thermostat_reach_cool(self):
        ha = thermostat()
        result = check_reachability(ha, {'cool'})
        assert result.reachable

    def test_thermostat_safety(self):
        ha = thermostat()
        # There's no "danger" mode, so safety should hold
        result = check_safety(ha, {'nonexistent_mode'})
        assert not result.reachable

    def test_water_tank_reach_draining(self):
        ha = water_tank()
        result = check_reachability(ha, {'draining'})
        assert result.reachable

    def test_simple_unreachable(self):
        """Mode with no incoming edges from reachable modes."""
        ha = HybridAutomaton(
            modes={'a', 'b', 'c'},
            initial_mode='a',
            variables=['x'],
            flows={'a': flow_from(x=1), 'b': flow_from(x=1), 'c': flow_from(x=1)},
            invariants={},
            edges=[
                HybridEdge('a', 'b', 'go', pred_from(var_geq('x', 5)), no_reset()),
                # No edge to 'c'
            ],
            initial_condition=pred_from(var_eq('x', 0))
        )
        assert check_reachability(ha, {'b'}).reachable
        assert not check_reachability(ha, {'c'}).reachable

    def test_initial_mode_is_target(self):
        ha = thermostat()
        result = check_reachability(ha, {'heat'})
        assert result.reachable
        assert result.explored_states == 1

    def test_two_tank_reachability(self):
        ha = two_tank()
        result = check_reachability(ha, {'fill2'})
        assert result.reachable

    def test_reachability_trace(self):
        ha = thermostat()
        result = check_reachability(ha, {'cool'})
        assert result.trace is not None
        assert result.trace[0][0] == 'heat'
        assert result.trace[-1][0] == 'cool'


# ============================================================
# Section 8: Safety Verification
# ============================================================

class TestSafetyVerification:
    def test_thermostat_safety_no_danger(self):
        ha = thermostat()
        result = verify_safety(ha, {'nonexistent'}, "no danger")
        assert result.verdict == 'SAFE'

    def test_thermostat_cool_reachable(self):
        ha = thermostat()
        result = verify_safety(ha, {'cool'}, "no cooling")
        assert result.verdict == 'UNSAFE'
        assert result.trace is not None

    def test_water_tank_safety(self):
        ha = water_tank()
        result = verify_safety(ha, {'nonexistent'}, "tank safety")
        assert result.verdict == 'SAFE'

    def test_verify_result_details(self):
        ha = thermostat()
        result = verify_safety(ha, {'cool'}, "test")
        assert 'reachable' in result.details.lower() or 'Unsafe' in result.details


# ============================================================
# Section 9: Invariant Verification
# ============================================================

class TestInvariantVerification:
    def test_thermostat_temp_bounded(self):
        """Temperature should stay within bounds due to invariants."""
        ha = thermostat()
        # The invariant is that temp stays in [18-2, 22+2] = [16, 24]
        # But the mode invariants actually enforce [18, 22] for switching
        result = verify_invariant(ha, pred_from(var_geq('temp', 0)), "temp_positive")
        assert result.verdict == 'SAFE'

    def test_water_tank_level_nonneg(self):
        ha = water_tank()
        result = verify_invariant(ha, pred_from(var_geq('level', 0)), "level_nonneg")
        assert result.verdict == 'SAFE'


# ============================================================
# Section 10: Bounded Liveness
# ============================================================

class TestBoundedLiveness:
    def test_thermostat_reaches_cool(self):
        ha = thermostat()
        result = verify_bounded_liveness(ha, {'cool'}, "eventually_cool")
        assert result.verdict == 'SATISFIED'

    def test_unreachable_liveness(self):
        ha = thermostat()
        result = verify_bounded_liveness(ha, {'nonexistent'}, "unreachable")
        assert result.verdict == 'UNKNOWN'


# ============================================================
# Section 11: Product Construction
# ============================================================

class TestProduct:
    def test_product_modes(self):
        ha1 = HybridAutomaton(
            modes={'a', 'b'}, initial_mode='a', variables=['x'],
            flows={'a': flow_from(x=1), 'b': flow_from(x=-1)},
            invariants={},
            edges=[HybridEdge('a', 'b', 'go', true_pred(), no_reset())],
            initial_condition=pred_from(var_eq('x', 0))
        )
        ha2 = HybridAutomaton(
            modes={'c', 'd'}, initial_mode='c', variables=['y'],
            flows={'c': flow_from(y=1), 'd': flow_from(y=-1)},
            invariants={},
            edges=[HybridEdge('c', 'd', 'go', true_pred(), no_reset())],
            initial_condition=pred_from(var_eq('y', 0))
        )
        p = product(ha1, ha2)
        assert p.initial_mode == 'a_c'
        assert len(p.modes) == 4
        assert 'x' in p.variables and 'y' in p.variables

    def test_product_sync_edges(self):
        ha1 = HybridAutomaton(
            modes={'a', 'b'}, initial_mode='a', variables=['x'],
            flows={'a': flow_from(x=1), 'b': flow_from(x=0)},
            invariants={},
            edges=[HybridEdge('a', 'b', 'sync', true_pred(), no_reset())],
            initial_condition=pred_from(var_eq('x', 0))
        )
        ha2 = HybridAutomaton(
            modes={'c', 'd'}, initial_mode='c', variables=['y'],
            flows={'c': flow_from(y=1), 'd': flow_from(y=0)},
            invariants={},
            edges=[HybridEdge('c', 'd', 'sync', true_pred(), no_reset())],
            initial_condition=pred_from(var_eq('y', 0))
        )
        p = product(ha1, ha2, sync_labels={'sync'})
        # Should have synchronized edge a_c -> b_d
        sync_edges = [e for e in p.edges if e.label == 'sync']
        assert any(e.source == 'a_c' and e.target == 'b_d' for e in sync_edges)

    def test_product_independent_edges(self):
        ha1 = HybridAutomaton(
            modes={'a', 'b'}, initial_mode='a', variables=['x'],
            flows={'a': flow_from(x=1), 'b': flow_from(x=0)},
            invariants={},
            edges=[HybridEdge('a', 'b', 'move1', true_pred(), no_reset())],
            initial_condition=pred_from(var_eq('x', 0))
        )
        ha2 = HybridAutomaton(
            modes={'c', 'd'}, initial_mode='c', variables=['y'],
            flows={'c': flow_from(y=1), 'd': flow_from(y=0)},
            invariants={},
            edges=[HybridEdge('c', 'd', 'move2', true_pred(), no_reset())],
            initial_condition=pred_from(var_eq('y', 0))
        )
        p = product(ha1, ha2, sync_labels=set())  # No sync
        # move1: a_c -> b_c, a_d -> b_d
        move1_edges = [e for e in p.edges if e.label == 'move1']
        assert len(move1_edges) == 2  # one per ha2 mode


# ============================================================
# Section 12: Example Systems
# ============================================================

class TestExampleSystems:
    def test_thermostat_structure(self):
        ha = thermostat()
        assert len(ha.modes) == 2
        assert len(ha.edges) == 2
        assert len(ha.variables) == 1

    def test_water_tank_structure(self):
        ha = water_tank()
        assert len(ha.modes) == 2
        assert len(ha.edges) == 2

    def test_railroad_crossing_structure(self):
        ha = railroad_crossing()
        assert len(ha.modes) == 6
        assert len(ha.variables) == 2
        assert 'train' in ha.variables
        assert 'gate' in ha.variables

    def test_bouncing_ball_structure(self):
        ha = bouncing_ball()
        assert 'fly' in ha.modes
        assert 'h' in ha.variables and 'v' in ha.variables

    def test_two_tank_structure(self):
        ha = two_tank()
        assert len(ha.modes) == 2
        assert len(ha.edges) == 4

    def test_thermostat_custom_params(self):
        ha = thermostat(temp_low=15, temp_high=25, heat_rate=2, cool_rate=-1)
        f = ha.get_flow('heat')
        assert f.get_rate('temp') == FlowInterval(2.0, 2.0)

    def test_water_tank_custom_params(self):
        ha = water_tank(inflow_rate=3.0, outflow_rate=-2.0, low_level=2.0, high_level=8.0)
        assert ha.initial_condition.evaluate({'level': 5.0})
        assert not ha.initial_condition.evaluate({'level': 1.0})


# ============================================================
# Section 13: Zone Graph Summary
# ============================================================

class TestZoneGraphSummary:
    def test_thermostat_summary(self):
        ha = thermostat()
        summary = zone_graph_summary(ha)
        assert summary['explored'] > 0
        assert 'heat' in summary['modes_reached']
        assert 'cool' in summary['modes_reached']

    def test_water_tank_summary(self):
        ha = water_tank()
        summary = zone_graph_summary(ha)
        assert summary['explored'] > 0


# ============================================================
# Section 14: Mode Analysis
# ============================================================

class TestModeAnalysis:
    def test_thermostat_modes(self):
        ha = thermostat()
        reach = analyze_modes(ha)
        assert reach['heat'] is True
        assert reach['cool'] is True

    def test_two_tank_modes(self):
        ha = two_tank()
        reach = analyze_modes(ha)
        assert reach['fill1'] is True
        assert reach['fill2'] is True


# ============================================================
# Section 15: Comparison API
# ============================================================

class TestComparison:
    def test_thermostat_comparison(self):
        ha = thermostat()
        comp = compare_hybrid_vs_timed(ha)
        assert comp['is_rectangular']
        assert comp['num_variables'] == 1
        assert comp['num_modes'] == 2

    def test_water_tank_comparison(self):
        ha = water_tank()
        comp = compare_hybrid_vs_timed(ha)
        assert comp['is_rectangular']

    def test_non_clock_detection(self):
        ha = thermostat(heat_rate=2.0)  # rate != 1 -> non-clock
        comp = compare_hybrid_vs_timed(ha)
        assert 'temp' in comp['non_clock_variables']
        assert not comp['timed_automata_expressible']


# ============================================================
# Section 16: Batch Verification
# ============================================================

class TestBatchVerification:
    def test_batch_verify(self):
        ha = thermostat()
        results = batch_verify(ha, [
            ("no_danger", {'nonexistent'}),
            ("no_cool", {'cool'}),
        ])
        assert len(results) == 2
        assert results[0].verdict == 'SAFE'
        assert results[1].verdict == 'UNSAFE'


# ============================================================
# Section 17: Rectangular Automaton Check
# ============================================================

class TestRectangularCheck:
    def test_simple_rectangular(self):
        ha = thermostat()
        assert ha.is_rectangular()

    def test_non_rectangular_diff_constraint(self):
        """An automaton with difference constraints in guards is non-rectangular."""
        ha = HybridAutomaton(
            modes={'a'},
            initial_mode='a',
            variables=['x', 'y'],
            flows={'a': flow_from(x=1, y=1)},
            invariants={},
            edges=[
                HybridEdge('a', 'a', 'go',
                            pred_from(diff_leq('x', 'y', 5)),  # x - y <= 5 is non-rectangular
                            no_reset())
            ],
            initial_condition=pred_from(var_eq('x', 0), var_eq('y', 0))
        )
        assert not ha.is_rectangular()


# ============================================================
# Section 18: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_initial_zone(self):
        """Initial condition that is unsatisfiable."""
        ha = HybridAutomaton(
            modes={'a'},
            initial_mode='a',
            variables=['x'],
            flows={'a': flow_from(x=1)},
            invariants={},
            edges=[],
            initial_condition=pred_from(var_geq('x', 10), var_leq('x', 5))
        )
        result = check_reachability(ha, {'a'})
        assert not result.reachable

    def test_zone_equality(self):
        z1 = RectZone(['x'])
        z1 = z1.constrain(var_leq('x', 10))
        z2 = RectZone(['x'])
        z2 = z2.constrain(var_leq('x', 10))
        assert z1 == z2

    def test_single_mode_no_edges(self):
        ha = HybridAutomaton(
            modes={'only'},
            initial_mode='only',
            variables=['t'],
            flows={'only': clock_flow('t')},
            invariants={},
            edges=[],
            initial_condition=pred_from(var_eq('t', 0))
        )
        result = check_reachability(ha, {'nonexistent'})
        assert not result.reachable

    def test_self_loop(self):
        ha = HybridAutomaton(
            modes={'a'},
            initial_mode='a',
            variables=['x'],
            flows={'a': flow_from(x=1)},
            invariants={},
            edges=[
                HybridEdge('a', 'a', 'tick',
                            pred_from(var_geq('x', 5)), reset_to('x', 0))
            ],
            initial_condition=pred_from(var_eq('x', 0))
        )
        summary = zone_graph_summary(ha, max_iterations=20)
        assert summary['explored'] > 0

    def test_interval_flow_time_elapse(self):
        """Variable with interval rate [1, 2]."""
        z = RectZone(['x'])
        z = z.constrain(var_eq('x', 0))
        f = flow_from(x=(1, 2))
        z2 = z.time_elapse(f)
        lo, hi = z2.get_bounds('x')
        assert lo == 0.0  # can't go below 0 (rate >= 1 > 0)
        assert hi == float('inf')  # grows unboundedly


# ============================================================
# Section 19: String Representations
# ============================================================

class TestStringRepresentations:
    def test_constraint_str(self):
        c = var_leq('x', 10)
        assert 'x' in str(c) and '10' in str(c)

    def test_predicate_str(self):
        p = pred_from(var_geq('x', 0), var_leq('x', 10))
        s = str(p)
        assert 'x' in s

    def test_flow_str(self):
        f = flow_from(x=1, y=(2, 3))
        s = str(f)
        assert 'x' in s or 'y' in s

    def test_reset_str(self):
        r = reset_to('x', 0)
        assert 'x' in str(r) and '0' in str(r)

    def test_edge_str(self):
        e = HybridEdge('a', 'b', 'go', true_pred(), no_reset())
        s = str(e)
        assert 'a' in s and 'b' in s

    def test_zone_str(self):
        z = RectZone(['x'])
        z = z.constrain(var_geq('x', 0))
        z = z.constrain(var_leq('x', 10))
        s = str(z)
        assert 'x' in s

    def test_true_pred_str(self):
        assert str(true_pred()) == 'true'


# ============================================================
# Section 20: Integration -- Full Verification Pipelines
# ============================================================

class TestIntegration:
    def test_thermostat_full_pipeline(self):
        """Full verification pipeline for thermostat."""
        ha = thermostat()
        # Safety: no "danger" mode
        safety = verify_safety(ha, {'nonexistent'}, "safety")
        assert safety.verdict == 'SAFE'
        # Invariant: temp >= 0
        inv = verify_invariant(ha, pred_from(var_geq('temp', 0)), "temp_pos")
        assert inv.verdict == 'SAFE'
        # Liveness: eventually reaches cool
        live = verify_bounded_liveness(ha, {'cool'}, "reaches_cool")
        assert live.verdict == 'SATISFIED'
        # Summary
        summary = zone_graph_summary(ha)
        assert summary['explored'] > 0

    def test_water_tank_full_pipeline(self):
        ha = water_tank()
        safety = verify_safety(ha, {'nonexistent'}, "safety")
        assert safety.verdict == 'SAFE'
        live = verify_bounded_liveness(ha, {'draining'}, "reaches_drain")
        assert live.verdict == 'SATISFIED'

    def test_two_tank_full_pipeline(self):
        ha = two_tank()
        # Both modes should be reachable
        reach1 = check_reachability(ha, {'fill1'})
        reach2 = check_reachability(ha, {'fill2'})
        assert reach1.reachable
        assert reach2.reachable
        # Comparison
        comp = compare_hybrid_vs_timed(ha)
        assert comp['num_variables'] == 2

    def test_bouncing_ball_reachability(self):
        ha = bouncing_ball()
        # The ball stays in fly mode (with self-loop bounces)
        summary = zone_graph_summary(ha, max_iterations=50)
        assert summary['explored'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
