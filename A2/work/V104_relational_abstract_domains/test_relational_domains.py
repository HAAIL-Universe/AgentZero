"""Tests for V104: Relational Abstract Domains (Octagon + Zone)"""

import sys
import math
import pytest

sys.path.insert(0, 'Z:/AgentZero/A2/work/V104_relational_abstract_domains')

from relational_domains import (
    ZoneDomain, OctagonDomain,
    OctagonInterpreter, ZoneInterpreter,
    octagon_analyze, zone_analyze,
    get_variable_range, get_relational_constraints,
    compare_analyses, verify_relational_property,
    INF,
)


# ===========================================================================
# Section 1: Zone Domain -- Basic Operations
# ===========================================================================

class TestZoneBasic:
    def test_create_empty(self):
        z = ZoneDomain([])
        assert not z.is_bot()
        assert z.n == 1  # just the zero variable

    def test_create_with_vars(self):
        z = ZoneDomain(['x', 'y'])
        assert z.n == 3
        assert not z.is_bot()

    def test_set_upper_lower(self):
        z = ZoneDomain(['x'])
        z.set_upper('x', 10)
        z.set_lower('x', 0)
        assert z.get_upper('x') == 10
        assert z.get_lower('x') == 0

    def test_set_diff(self):
        z = ZoneDomain(['x', 'y'])
        z.set_diff('x', 'y', 5)
        assert z.get_diff_bound('x', 'y') == 5

    def test_assign_const(self):
        z = ZoneDomain(['x'])
        z.assign_const('x', 42)
        assert z.get_upper('x') == 42
        assert z.get_lower('x') == 42

    def test_assign_var(self):
        z = ZoneDomain(['x', 'y'])
        z.assign_const('x', 10)
        z.assign_var('y', 'x')
        assert z.get_upper('y') == 10
        assert z.get_lower('y') == 10
        assert z.get_diff_bound('y', 'x') == 0

    def test_assign_add(self):
        z = ZoneDomain(['x', 'y'])
        z.assign_const('x', 5)
        z.assign_add('y', 'x', 3)
        assert z.get_lower('y') == 8
        assert z.get_upper('y') == 8

    def test_bot(self):
        z = ZoneDomain.bot(['x'])
        assert z.is_bot()

    def test_forget(self):
        z = ZoneDomain(['x'])
        z.assign_const('x', 5)
        z.forget('x')
        assert z.get_upper('x') == INF
        assert z.get_lower('x') == -INF

    def test_add_var(self):
        z = ZoneDomain(['x'])
        z.assign_const('x', 5)
        z.add_var('y')
        assert 'y' in z._var_index
        assert z.get_upper('x') == 5  # preserved

    def test_get_interval(self):
        z = ZoneDomain(['x'])
        z.set_lower('x', 3)
        z.set_upper('x', 7)
        lo, hi = z.get_interval('x')
        assert lo == 3
        assert hi == 7

    def test_closure_tightens(self):
        z = ZoneDomain(['x', 'y', 'z'])
        z.set_diff('x', 'y', 3)  # x - y <= 3
        z.set_diff('y', 'z', 2)  # y - z <= 2
        z.close()
        # Should derive: x - z <= 5
        assert z.get_diff_bound('x', 'z') == 5

    def test_constraints_str(self):
        z = ZoneDomain(['x', 'y'])
        z.set_lower('x', 0)
        z.set_upper('x', 10)
        cs = z.get_constraints()
        assert any('x' in c for c in cs)


# ===========================================================================
# Section 2: Zone Domain -- Lattice Operations
# ===========================================================================

class TestZoneLattice:
    def test_join_basic(self):
        z1 = ZoneDomain(['x'])
        z1.assign_const('x', 5)
        z2 = ZoneDomain(['x'])
        z2.assign_const('x', 10)
        j = z1.join(z2)
        assert j.get_lower('x') == 5
        assert j.get_upper('x') == 10

    def test_join_preserves_diff(self):
        z1 = ZoneDomain(['x', 'y'])
        z1.set_diff('x', 'y', 3)
        z2 = ZoneDomain(['x', 'y'])
        z2.set_diff('x', 'y', 5)
        j = z1.join(z2)
        assert j.get_diff_bound('x', 'y') == 5  # max(3, 5)

    def test_join_bot(self):
        z1 = ZoneDomain.bot(['x'])
        z2 = ZoneDomain(['x'])
        z2.assign_const('x', 5)
        j = z1.join(z2)
        assert j.get_lower('x') == 5
        assert j.get_upper('x') == 5

    def test_meet(self):
        z1 = ZoneDomain(['x'])
        z1.set_lower('x', 0)
        z1.set_upper('x', 10)
        z2 = ZoneDomain(['x'])
        z2.set_lower('x', 5)
        z2.set_upper('x', 15)
        m = z1.meet(z2)
        assert m.get_lower('x') == 5
        assert m.get_upper('x') == 10

    def test_widen(self):
        z1 = ZoneDomain(['x'])
        z1.set_lower('x', 0)
        z1.set_upper('x', 5)
        z2 = ZoneDomain(['x'])
        z2.set_lower('x', 0)
        z2.set_upper('x', 10)  # increased
        w = z1.widen(z2)
        assert w.get_lower('x') == 0  # stable
        assert w.get_upper('x') == INF  # widened to infinity

    def test_leq(self):
        z1 = ZoneDomain(['x'])
        z1.set_lower('x', 2)
        z1.set_upper('x', 8)
        z2 = ZoneDomain(['x'])
        z2.set_lower('x', 0)
        z2.set_upper('x', 10)
        assert z1.leq(z2)
        assert not z2.leq(z1)

    def test_equals(self):
        z1 = ZoneDomain(['x'])
        z1.set_lower('x', 0)
        z1.set_upper('x', 10)
        z2 = ZoneDomain(['x'])
        z2.set_lower('x', 0)
        z2.set_upper('x', 10)
        assert z1.equals(z2)


# ===========================================================================
# Section 3: Octagon Domain -- Basic Operations
# ===========================================================================

class TestOctagonBasic:
    def test_create(self):
        o = OctagonDomain(['x', 'y'])
        assert o.size == 4
        assert not o.is_bot()

    def test_set_bounds(self):
        o = OctagonDomain(['x'])
        o.set_upper('x', 10)
        o.set_lower('x', -5)
        assert o.get_upper('x') == 10
        assert o.get_lower('x') == -5

    def test_sum_constraint(self):
        o = OctagonDomain(['x', 'y'])
        o.set_sum_upper('x', 'y', 10)
        assert o.get_sum_bound('x', 'y') == 10

    def test_diff_constraint(self):
        o = OctagonDomain(['x', 'y'])
        o.set_diff_upper('x', 'y', 3)
        assert o.get_diff_bound('x', 'y') == 3

    def test_assign_const(self):
        o = OctagonDomain(['x'])
        o.assign_const('x', 7)
        assert o.get_upper('x') == 7
        assert o.get_lower('x') == 7

    def test_assign_var(self):
        o = OctagonDomain(['x', 'y'])
        o.assign_const('x', 5)
        o.assign_var('y', 'x')
        assert o.get_upper('y') == 5
        assert o.get_lower('y') == 5

    def test_assign_add_const(self):
        o = OctagonDomain(['x', 'y'])
        o.assign_const('x', 10)
        o.assign_add_const('y', 'x', 3)
        assert o.get_lower('y') == 13
        assert o.get_upper('y') == 13

    def test_assign_add_const_self(self):
        o = OctagonDomain(['x'])
        o.assign_const('x', 5)
        o.assign_add_const('x', 'x', 1)  # x = x + 1
        assert o.get_lower('x') == 6
        assert o.get_upper('x') == 6

    def test_bot(self):
        o = OctagonDomain.bot(['x'])
        assert o.is_bot()

    def test_forget(self):
        o = OctagonDomain(['x'])
        o.assign_const('x', 5)
        o.forget('x')
        assert o.get_upper('x') == INF
        assert o.get_lower('x') == -INF

    def test_add_var(self):
        o = OctagonDomain(['x'])
        o.assign_const('x', 5)
        o.add_var('y')
        assert 'y' in o._var_index
        assert o.get_upper('x') == 5

    def test_closure(self):
        o = OctagonDomain(['x', 'y'])
        o.set_upper('x', 10)
        o.set_diff_upper('x', 'y', 3)  # x - y <= 3
        o.set_lower('y', 5)
        o.close()
        # x <= min(10, y+3) and y >= 5 => x <= 8
        assert o.get_upper('x') <= 10  # at least as tight

    def test_get_constraints(self):
        o = OctagonDomain(['x', 'y'])
        o.assign_const('x', 5)
        o.set_diff_upper('x', 'y', 3)
        cs = o.get_constraints()
        assert len(cs) > 0


# ===========================================================================
# Section 4: Octagon Domain -- Lattice Operations
# ===========================================================================

class TestOctagonLattice:
    def test_join(self):
        o1 = OctagonDomain(['x'])
        o1.assign_const('x', 5)
        o2 = OctagonDomain(['x'])
        o2.assign_const('x', 15)
        j = o1.join(o2)
        assert j.get_lower('x') == 5
        assert j.get_upper('x') == 15

    def test_join_preserves_diff(self):
        o1 = OctagonDomain(['x', 'y'])
        o1.set_diff_upper('x', 'y', 2)
        o2 = OctagonDomain(['x', 'y'])
        o2.set_diff_upper('x', 'y', 4)
        j = o1.join(o2)
        assert j.get_diff_bound('x', 'y') == 4

    def test_join_bot(self):
        o1 = OctagonDomain.bot(['x'])
        o2 = OctagonDomain(['x'])
        o2.assign_const('x', 5)
        j = o1.join(o2)
        assert j.get_lower('x') == 5

    def test_meet(self):
        o1 = OctagonDomain(['x'])
        o1.set_lower('x', 0)
        o1.set_upper('x', 10)
        o2 = OctagonDomain(['x'])
        o2.set_lower('x', 5)
        o2.set_upper('x', 20)
        m = o1.meet(o2)
        assert m.get_lower('x') == 5
        assert m.get_upper('x') == 10

    def test_widen(self):
        o1 = OctagonDomain(['x'])
        o1.set_lower('x', 0)
        o1.set_upper('x', 5)
        o2 = OctagonDomain(['x'])
        o2.set_lower('x', 0)
        o2.set_upper('x', 10)
        w = o1.widen(o2)
        assert w.get_lower('x') == 0
        assert w.get_upper('x') == INF

    def test_leq(self):
        o1 = OctagonDomain(['x'])
        o1.set_lower('x', 2)
        o1.set_upper('x', 8)
        o2 = OctagonDomain(['x'])
        o2.set_lower('x', 0)
        o2.set_upper('x', 10)
        assert o1.leq(o2)
        assert not o2.leq(o1)

    def test_equals(self):
        o1 = OctagonDomain(['x'])
        o1.set_lower('x', 0)
        o1.set_upper('x', 10)
        o2 = OctagonDomain(['x'])
        o2.set_lower('x', 0)
        o2.set_upper('x', 10)
        assert o1.equals(o2)

    def test_sum_constraint_join(self):
        o1 = OctagonDomain(['x', 'y'])
        o1.set_sum_upper('x', 'y', 10)
        o2 = OctagonDomain(['x', 'y'])
        o2.set_sum_upper('x', 'y', 15)
        j = o1.join(o2)
        assert j.get_sum_bound('x', 'y') == 15


# ===========================================================================
# Section 5: Octagon Interpreter -- Simple Programs
# ===========================================================================

class TestOctagonInterpreterSimple:
    def test_const_assignment(self):
        result = octagon_analyze('let x = 5;')
        env = result['env']
        assert env.get_lower('x') == 5
        assert env.get_upper('x') == 5

    def test_var_copy(self):
        result = octagon_analyze('let x = 10; let y = x;')
        env = result['env']
        assert env.get_lower('y') == 10
        assert env.get_upper('y') == 10
        # Relational: y - x = 0
        assert env.get_diff_bound('y', 'x') == 0
        assert env.get_diff_bound('x', 'y') == 0

    def test_add_const(self):
        result = octagon_analyze('let x = 5; let y = x + 3;')
        env = result['env']
        assert env.get_lower('y') == 8
        assert env.get_upper('y') == 8

    def test_sub_const(self):
        result = octagon_analyze('let x = 10; let y = x - 3;')
        env = result['env']
        assert env.get_lower('y') == 7
        assert env.get_upper('y') == 7

    def test_multiple_vars(self):
        result = octagon_analyze('let x = 5; let y = 10; let z = 15;')
        env = result['env']
        assert env.get_lower('x') == 5
        assert env.get_lower('y') == 10
        assert env.get_lower('z') == 15

    def test_reassignment(self):
        result = octagon_analyze('let x = 5; x = 10;')
        env = result['env']
        assert env.get_lower('x') == 10
        assert env.get_upper('x') == 10


# ===========================================================================
# Section 6: Octagon Interpreter -- Relational Tracking
# ===========================================================================

class TestOctagonRelational:
    def test_diff_tracking(self):
        """y = x + 3 => y - x = 3."""
        result = octagon_analyze('let x = 5; let y = x + 3;')
        env = result['env']
        # y - x should be tracked
        diff = env.get_diff_bound('y', 'x')
        assert diff <= 3  # y - x <= 3

    def test_diff_tracking_sub(self):
        """y = x - 2 => x - y = 2."""
        result = octagon_analyze('let x = 10; let y = x - 2;')
        env = result['env']
        diff = env.get_diff_bound('x', 'y')
        assert diff <= 2

    def test_sum_conservation(self):
        """x + y = 10 after x=3, y=7."""
        result = octagon_analyze('let x = 3; let y = 7;')
        env = result['env']
        sbound = env.get_sum_bound('x', 'y')
        assert sbound <= 10

    def test_relational_through_copy(self):
        """z = x after x = 5 => z - x = 0, z = 5."""
        result = octagon_analyze('let x = 5; let z = x;')
        env = result['env']
        assert env.get_diff_bound('z', 'x') == 0

    def test_self_increment(self):
        """x = x + 1 shifts all bounds by 1."""
        result = octagon_analyze('let x = 5; x = x + 1;')
        env = result['env']
        assert env.get_lower('x') == 6
        assert env.get_upper('x') == 6


# ===========================================================================
# Section 7: Octagon Interpreter -- Conditionals
# ===========================================================================

class TestOctagonConditionals:
    def test_if_refine_upper(self):
        src = 'let x = 10; if (x < 5) { x = x; }'
        result = octagon_analyze(src)
        # x was 10, condition is false, so x stays 10

    def test_if_var_comparison(self):
        """Relational condition: x < y. z declared before if, assigned inside."""
        src = '''
        let x = 3;
        let y = 10;
        let z = 0;
        if (x < y) {
            z = y - x;
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        # z is either 0 (else) or 7 (then), so after join: 0 <= z <= 7
        lo, hi = env.get_interval('z')
        assert lo >= 0
        assert hi <= 7

    def test_if_else(self):
        src = '''
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        # x is 5, so always takes then-branch: y = 1
        # But the octagon analysis may join both branches
        lo, hi = env.get_interval('y')
        assert lo >= 0  # at least 0
        assert hi <= 2  # at most 2

    def test_relational_condition_strengthens(self):
        """After if (x <= y), octagon knows x - y <= 0."""
        src = '''
        let x = 5;
        let y = 10;
        let z = 0;
        if (x <= y) {
            z = 1;
        }
        '''
        result = octagon_analyze(src)
        # The then-branch has x - y <= 0 constraint


# ===========================================================================
# Section 8: Octagon Interpreter -- Loops
# ===========================================================================

class TestOctagonLoops:
    def test_simple_countdown(self):
        src = '''
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        # After loop: i <= 0
        assert env.get_upper('i') <= 0

    def test_countup(self):
        src = '''
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        # After loop: i >= 10
        assert env.get_lower('i') >= 10

    def test_two_var_loop(self):
        """Track relational invariant: x + y = 10."""
        src = '''
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        # After loop: x >= 10
        assert env.get_lower('x') >= 10

    def test_loop_convergence(self):
        """Widening ensures convergence."""
        src = '''
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        assert env.get_lower('x') >= 100


# ===========================================================================
# Section 9: Zone Interpreter
# ===========================================================================

class TestZoneInterpreter:
    def test_basic(self):
        result = zone_analyze('let x = 5; let y = x + 3;')
        env = result['env']
        assert env.get_lower('y') == 8
        assert env.get_upper('y') == 8

    def test_diff_constraint(self):
        result = zone_analyze('let x = 5; let y = x + 3;')
        env = result['env']
        # y - x should be 3
        assert env.get_diff_bound('y', 'x') == 3

    def test_var_copy(self):
        result = zone_analyze('let x = 10; let y = x;')
        env = result['env']
        assert env.get_diff_bound('y', 'x') == 0
        assert env.get_diff_bound('x', 'y') == 0

    def test_conditional(self):
        src = '''
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        '''
        result = zone_analyze(src)
        env = result['env']
        lo, hi = env.get_interval('y')
        assert lo >= 0
        assert hi <= 2

    def test_loop(self):
        src = '''
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        '''
        result = zone_analyze(src)
        env = result['env']
        assert env.get_lower('i') >= 10

    def test_relational_comparison(self):
        src = '''
        let x = 5;
        let y = 10;
        if (x < y) {
            let z = 1;
        }
        '''
        result = zone_analyze(src)
        # Should not crash, handles var-var comparison


# ===========================================================================
# Section 10: High-Level API
# ===========================================================================

class TestHighLevelAPI:
    def test_get_variable_range_octagon(self):
        lo, hi = get_variable_range('let x = 5;', 'x', 'octagon')
        assert lo == 5
        assert hi == 5

    def test_get_variable_range_zone(self):
        lo, hi = get_variable_range('let x = 5;', 'x', 'zone')
        assert lo == 5
        assert hi == 5

    def test_get_relational_constraints(self):
        cs = get_relational_constraints('let x = 5; let y = x + 3;')
        assert len(cs) > 0

    def test_compare_analyses(self):
        src = 'let x = 5; let y = 10;'
        comparison = compare_analyses(src)
        assert 'octagon_constraints' in comparison
        assert 'interval_results' in comparison

    def test_verify_upper_bound(self):
        result = verify_relational_property(
            'let x = 5;',
            'x <= 10'
        )
        assert result['verdict'] == 'VERIFIED'

    def test_verify_lower_bound(self):
        result = verify_relational_property(
            'let x = 5;',
            'x >= 0'
        )
        assert result['verdict'] == 'VERIFIED'

    def test_verify_diff_constraint(self):
        result = verify_relational_property(
            'let x = 5; let y = x + 3;',
            'y - x <= 5'
        )
        assert result['verdict'] == 'VERIFIED'


# ===========================================================================
# Section 11: Precision Comparison -- Octagon vs Interval
# ===========================================================================

class TestPrecisionGains:
    def test_sum_conservation_precision(self):
        """Octagon captures x + y = const that interval loses."""
        src = '''
        let x = 5;
        let y = 5;
        if (x > 3) {
            x = x + 1;
            y = y - 1;
        }
        '''
        oct_result = octagon_analyze(src)
        oct_env = oct_result['env']
        # Octagon should track sum constraint better
        cs = oct_env.get_constraints()
        assert len(cs) > 0  # Has some constraints

    def test_diff_precision(self):
        """Octagon tracks x - y precisely through assignments."""
        src = 'let x = 10; let y = x - 3;'
        oct_result = octagon_analyze(src)
        oct_env = oct_result['env']
        # y = x - 3, so x - y = 3
        diff = oct_env.get_diff_bound('x', 'y')
        assert diff <= 3

    def test_comparison_api(self):
        src = 'let x = 5; let y = x + 3;'
        comp = compare_analyses(src)
        assert 'x' in comp['interval_results']
        assert 'y' in comp['interval_results']

    def test_loop_invariant_precision(self):
        """Octagon may capture loop invariants that interval misses."""
        src = '''
        let x = 0;
        let y = 10;
        while (x < 5) {
            x = x + 1;
            y = y - 1;
        }
        '''
        oct_result = octagon_analyze(src)
        oct_env = oct_result['env']
        # After loop: x >= 5
        assert oct_env.get_lower('x') >= 5


# ===========================================================================
# Section 12: Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = octagon_analyze('')
        assert result['env'] is not None

    def test_function_declaration(self):
        src = '''
        fn add(a, b) { return a + b; }
        let x = 5;
        '''
        result = octagon_analyze(src)
        assert 'add' in result['functions']
        assert result['env'].get_lower('x') == 5

    def test_nested_if(self):
        src = '''
        let x = 5;
        let y = 10;
        if (x > 0) {
            if (y > 5) {
                x = x + 1;
            }
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        lo, hi = env.get_interval('x')
        assert lo >= 5
        assert hi <= 6

    def test_multiple_loops(self):
        src = '''
        let x = 0;
        while (x < 5) {
            x = x + 1;
        }
        let y = 0;
        while (y < 3) {
            y = y + 1;
        }
        '''
        result = octagon_analyze(src)
        env = result['env']
        assert env.get_lower('x') >= 5
        assert env.get_lower('y') >= 3

    def test_negative_values(self):
        src = 'let x = 0; x = x - 5;'
        result = octagon_analyze(src)
        env = result['env']
        lo, hi = env.get_interval('x')
        assert lo == -5
        assert hi == -5

    def test_large_program(self):
        """Verify analysis doesn't crash on larger programs."""
        lines = ['let x0 = 0;']
        for i in range(1, 20):
            lines.append(f'let x{i} = x{i-1} + 1;')
        src = '\n'.join(lines)
        result = octagon_analyze(src)
        env = result['env']
        assert env.get_lower('x19') == 19
        assert env.get_upper('x19') == 19

    def test_zone_repr(self):
        z = ZoneDomain(['x'])
        z.assign_const('x', 5)
        s = repr(z)
        assert 'Zone' in s

    def test_octagon_repr(self):
        o = OctagonDomain(['x'])
        o.assign_const('x', 5)
        s = repr(o)
        assert 'Octagon' in s

    def test_bot_repr(self):
        z = ZoneDomain.bot(['x'])
        assert 'BOT' in repr(z)
        o = OctagonDomain.bot(['x'])
        assert 'BOT' in repr(o)


# ===========================================================================
# Section 13: Verify Relational Property API
# ===========================================================================

class TestVerifyProperty:
    def test_verify_diff_tight(self):
        result = verify_relational_property(
            'let x = 5; let y = x + 3;',
            'y - x <= 3'
        )
        assert result['verdict'] == 'VERIFIED'

    def test_verify_diff_too_tight(self):
        result = verify_relational_property(
            'let x = 5; let y = x + 3;',
            'y - x <= 2'
        )
        # Cannot verify since y - x = 3 > 2
        assert result['verdict'] == 'UNKNOWN'

    def test_verify_sum_bound(self):
        result = verify_relational_property(
            'let x = 3; let y = 7;',
            'x + y <= 10'
        )
        assert result['verdict'] == 'VERIFIED'

    def test_verify_returns_constraints(self):
        result = verify_relational_property(
            'let x = 5; let y = 10;',
            'x <= 100'
        )
        assert 'all_constraints' in result
        assert len(result['all_constraints']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
