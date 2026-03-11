"""
Tests for V108: Abstract Domain Composition Framework
"""

import sys
import os
import pytest

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from domain_composition import (
    # Reducers
    reduce_sign_interval, reduce_const_interval, reduce_const_sign,
    reduce_parity_interval, reduce_parity_sign,
    find_builtin_reducers, BUILTIN_REDUCERS,
    # Builder
    ReducedProductBuilder, compose_domains,
    # Combinators
    DisjunctiveDomain, LiftedDomain, LiftState,
    CardinalPowerDomain,
    # Interpreter
    CompositionInterpreter, CompositionEnv,
    # Analysis
    analyze_with_composition, analyze_single_domain,
    compare_compositions, full_composition_analysis, composition_summary,
    ComparisonResult,
)

# Import domain types
_v020 = os.path.normpath(os.path.join(_here, '..', 'V020_abstract_domain_functor'))
if _v020 not in sys.path:
    sys.path.insert(0, _v020)

from domain_functor import (
    AbstractDomain, SignDomain, SignValue, IntervalDomain,
    ConstDomain, ParityDomain, ParityValue,
    ProductDomain, ReducedProductDomain,
    INF, NEG_INF,
)


# ===========================================================================
# Section 1: Sign-Interval Reducer
# ===========================================================================

class TestReduceSignInterval:
    def test_pos_tightens_interval(self):
        components = [SignDomain(SignValue.POS), IntervalDomain(-5, 10)]
        result = reduce_sign_interval(components)
        assert isinstance(result[1], IntervalDomain)
        assert result[1].lo == 1
        assert result[1].hi == 10

    def test_neg_tightens_interval(self):
        components = [SignDomain(SignValue.NEG), IntervalDomain(-5, 10)]
        result = reduce_sign_interval(components)
        assert result[1].lo == -5
        assert result[1].hi == -1

    def test_zero_tightens_interval(self):
        components = [SignDomain(SignValue.ZERO), IntervalDomain(-5, 10)]
        result = reduce_sign_interval(components)
        assert result[1].lo == 0
        assert result[1].hi == 0

    def test_non_neg_tightens_interval(self):
        components = [SignDomain(SignValue.NON_NEG), IntervalDomain(-5, 10)]
        result = reduce_sign_interval(components)
        assert result[1].lo == 0

    def test_interval_tightens_sign_to_pos(self):
        components = [SignDomain(SignValue.TOP), IntervalDomain(3, 10)]
        result = reduce_sign_interval(components)
        assert result[0].value == SignValue.POS

    def test_interval_tightens_sign_to_neg(self):
        components = [SignDomain(SignValue.TOP), IntervalDomain(-10, -1)]
        result = reduce_sign_interval(components)
        assert result[0].value == SignValue.NEG

    def test_interval_tightens_sign_to_zero(self):
        components = [SignDomain(SignValue.TOP), IntervalDomain(0, 0)]
        result = reduce_sign_interval(components)
        assert result[0].value == SignValue.ZERO

    def test_contradictory_becomes_bot(self):
        components = [SignDomain(SignValue.POS), IntervalDomain(-5, -1)]
        result = reduce_sign_interval(components)
        assert result[0].is_bot()
        assert result[1].is_bot()

    def test_no_sign_no_interval_passthrough(self):
        components = [ConstDomain.from_concrete(5)]
        result = reduce_sign_interval(components)
        assert len(result) == 1

    def test_bot_propagates(self):
        components = [SignDomain(SignValue.BOT), IntervalDomain(0, 10)]
        result = reduce_sign_interval(components)
        assert all(c.is_bot() for c in result)


# ===========================================================================
# Section 2: Const-Interval Reducer
# ===========================================================================

class TestReduceConstInterval:
    def test_const_tightens_interval(self):
        components = [ConstDomain.from_concrete(5), IntervalDomain(-10, 10)]
        result = reduce_const_interval(components)
        assert result[1].lo == 5
        assert result[1].hi == 5

    def test_singleton_interval_sets_const(self):
        components = [ConstDomain(), IntervalDomain(7, 7)]  # TOP const
        result = reduce_const_interval(components)
        assert result[0].is_const()
        assert result[0].value == 7

    def test_const_contradicts_interval_bot(self):
        components = [ConstDomain.from_concrete(20), IntervalDomain(0, 10)]
        result = reduce_const_interval(components)
        assert all(c.is_bot() for c in result)

    def test_wide_interval_no_const_change(self):
        components = [ConstDomain(), IntervalDomain(-10, 10)]
        result = reduce_const_interval(components)
        assert result[0].is_top()

    def test_bot_propagates(self):
        components = [ConstDomain.from_concrete(5), IntervalDomain(1, 0)]  # bot interval
        result = reduce_const_interval(components)
        assert all(c.is_bot() for c in result)


# ===========================================================================
# Section 3: Const-Sign Reducer
# ===========================================================================

class TestReduceConstSign:
    def test_positive_const_tightens_sign(self):
        components = [ConstDomain.from_concrete(5), SignDomain(SignValue.TOP)]
        result = reduce_const_sign(components)
        assert result[1].value == SignValue.POS

    def test_negative_const_tightens_sign(self):
        components = [ConstDomain.from_concrete(-3), SignDomain(SignValue.TOP)]
        result = reduce_const_sign(components)
        assert result[1].value == SignValue.NEG

    def test_zero_const_tightens_sign(self):
        components = [ConstDomain.from_concrete(0), SignDomain(SignValue.TOP)]
        result = reduce_const_sign(components)
        assert result[1].value == SignValue.ZERO

    def test_sign_zero_sets_const_zero(self):
        components = [ConstDomain(), SignDomain(SignValue.ZERO)]
        result = reduce_const_sign(components)
        assert result[0].is_const() and result[0].value == 0

    def test_contradictory_bot(self):
        components = [ConstDomain.from_concrete(5), SignDomain(SignValue.NEG)]
        result = reduce_const_sign(components)
        assert all(c.is_bot() for c in result)


# ===========================================================================
# Section 4: Parity-Interval Reducer
# ===========================================================================

class TestReduceParityInterval:
    def test_singleton_even_interval_sets_parity(self):
        components = [ParityDomain(ParityValue.TOP), IntervalDomain(4, 4)]
        result = reduce_parity_interval(components)
        assert result[0].value == ParityValue.EVEN

    def test_singleton_odd_interval_sets_parity(self):
        components = [ParityDomain(ParityValue.TOP), IntervalDomain(5, 5)]
        result = reduce_parity_interval(components)
        assert result[0].value == ParityValue.ODD

    def test_even_parity_tightens_odd_lo(self):
        components = [ParityDomain(ParityValue.EVEN), IntervalDomain(3, 10)]
        result = reduce_parity_interval(components)
        assert result[1].lo == 4

    def test_odd_parity_tightens_even_lo(self):
        components = [ParityDomain(ParityValue.ODD), IntervalDomain(4, 10)]
        result = reduce_parity_interval(components)
        assert result[1].lo == 5

    def test_even_parity_tightens_odd_hi(self):
        components = [ParityDomain(ParityValue.EVEN), IntervalDomain(0, 9)]
        result = reduce_parity_interval(components)
        assert result[1].hi == 8

    def test_contradictory_singleton_bot(self):
        components = [ParityDomain(ParityValue.ODD), IntervalDomain(4, 4)]
        result = reduce_parity_interval(components)
        assert all(c.is_bot() for c in result)

    def test_wide_interval_no_parity_change(self):
        components = [ParityDomain(ParityValue.TOP), IntervalDomain(-10, 10)]
        result = reduce_parity_interval(components)
        assert result[0].value == ParityValue.TOP


# ===========================================================================
# Section 5: Parity-Sign Reducer
# ===========================================================================

class TestReduceParitySign:
    def test_zero_sign_sets_even(self):
        components = [ParityDomain(ParityValue.TOP), SignDomain(SignValue.ZERO)]
        result = reduce_parity_sign(components)
        assert result[0].value == ParityValue.EVEN

    def test_odd_parity_excludes_zero_from_nonneg(self):
        components = [ParityDomain(ParityValue.ODD), SignDomain(SignValue.NON_NEG)]
        result = reduce_parity_sign(components)
        assert result[1].value == SignValue.POS

    def test_odd_parity_excludes_zero_from_nonpos(self):
        components = [ParityDomain(ParityValue.ODD), SignDomain(SignValue.NON_POS)]
        result = reduce_parity_sign(components)
        assert result[1].value == SignValue.NEG

    def test_odd_zero_bot(self):
        components = [ParityDomain(ParityValue.ODD), SignDomain(SignValue.ZERO)]
        result = reduce_parity_sign(components)
        assert all(c.is_bot() for c in result)


# ===========================================================================
# Section 6: Auto-Discovery of Reducers
# ===========================================================================

class TestFindBuiltinReducers:
    def test_finds_sign_interval(self):
        reducers = find_builtin_reducers([SignDomain, IntervalDomain])
        assert len(reducers) == 1

    def test_finds_all_for_three_domains(self):
        reducers = find_builtin_reducers([SignDomain, IntervalDomain, ConstDomain])
        assert len(reducers) == 3  # sign-interval, const-interval, const-sign

    def test_finds_all_for_four_domains(self):
        reducers = find_builtin_reducers([SignDomain, IntervalDomain, ConstDomain, ParityDomain])
        assert len(reducers) == 5

    def test_empty_for_single_domain(self):
        reducers = find_builtin_reducers([SignDomain])
        assert len(reducers) == 0

    def test_reverse_order_works(self):
        reducers = find_builtin_reducers([IntervalDomain, SignDomain])
        assert len(reducers) == 1


# ===========================================================================
# Section 7: ReducedProductBuilder
# ===========================================================================

class TestReducedProductBuilder:
    def test_build_two_domains(self):
        builder = ReducedProductBuilder()
        builder.add(SignDomain).add(IntervalDomain).auto_reduce()
        factory = builder.build()
        val = factory(5)
        assert isinstance(val, ReducedProductDomain)
        assert len(val.components) == 2

    def test_factory_concrete_value(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        val = factory(5)
        # Sign should be POS, interval [5,5]
        assert val[0].value == SignValue.POS
        assert val[1].lo == 5
        assert val[1].hi == 5

    def test_factory_top(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        val = factory()
        assert val[0].is_top()
        assert val[1].is_top()

    def test_three_domains(self):
        factory = compose_domains(SignDomain, IntervalDomain, ParityDomain)
        val = factory(6)
        assert val[0].value == SignValue.POS
        assert val[1].lo == 6 and val[1].hi == 6
        assert val[2].value == ParityValue.EVEN

    def test_four_domains(self):
        factory = compose_domains(SignDomain, IntervalDomain, ConstDomain, ParityDomain)
        val = factory(7)
        assert len(val.components) == 4

    def test_fixpoint_reduction(self):
        factory = compose_domains(SignDomain, IntervalDomain, ConstDomain, fixpoint=True)
        val = factory(3)
        assert val[0].value == SignValue.POS

    def test_builder_interpreter(self):
        builder = ReducedProductBuilder()
        builder.add(SignDomain).add(IntervalDomain).auto_reduce()
        interp = builder.build_interpreter()
        result = interp.analyze("let x = 5;")
        assert 'bindings' in result


# ===========================================================================
# Section 8: DisjunctiveDomain
# ===========================================================================

class TestDisjunctiveDomain:
    def test_single_disjunct(self):
        d = DisjunctiveDomain.wrap(IntervalDomain(1, 5))
        assert not d.is_bot()
        assert not d.is_top()
        assert len(d.disjuncts) == 1

    def test_join_creates_disjuncts(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(1, 3))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(7, 9))
        joined = d1.join(d2)
        assert len(joined.disjuncts) == 2

    def test_bounded_cardinality(self):
        d = DisjunctiveDomain.wrap(IntervalDomain(1, 1), max_disjuncts=3)
        for i in range(2, 10):
            d = d.join(DisjunctiveDomain.wrap(IntervalDomain(i*10, i*10), max_disjuncts=3))
        assert len(d.disjuncts) <= 3

    def test_bot(self):
        d = DisjunctiveDomain(max_disjuncts=4, _is_bot=True)
        assert d.is_bot()

    def test_top(self):
        d = DisjunctiveDomain.wrap(IntervalDomain())
        assert d.is_top()

    def test_meet_pairwise(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(0, 10))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(5, 15))
        m = d1.meet(d2)
        assert len(m.disjuncts) == 1
        assert m.disjuncts[0].lo == 5
        assert m.disjuncts[0].hi == 10

    def test_meet_bot(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(0, 3))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(5, 10))
        m = d1.meet(d2)
        assert m.is_bot()

    def test_add(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(1, 3))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(10, 20))
        s = d1.add(d2)
        assert len(s.disjuncts) == 1
        assert s.disjuncts[0].lo == 11
        assert s.disjuncts[0].hi == 23

    def test_neg(self):
        d = DisjunctiveDomain.wrap(IntervalDomain(3, 5))
        n = d.neg()
        assert n.disjuncts[0].lo == -5
        assert n.disjuncts[0].hi == -3

    def test_leq(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(2, 3))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(0, 10))
        assert d1.leq(d2)
        assert not d2.leq(d1)

    def test_collapse(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(1, 3))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(7, 9))
        joined = d1.join(d2)
        collapsed = joined.collapse()
        assert collapsed.lo == 1
        assert collapsed.hi == 9

    def test_widen(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(0, 5))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(0, 10))
        w = d1.widen(d2)
        assert not w.is_bot()

    def test_eq(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(1, 5))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(1, 5))
        assert d1.eq(d2)

    def test_refine_lt(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(0, 10))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(0, 10))
        r1, r2 = d1.refine_lt(d2)
        assert not r1.is_bot()

    def test_refine_eq(self):
        d1 = DisjunctiveDomain.wrap(IntervalDomain(0, 10))
        d2 = DisjunctiveDomain.wrap(IntervalDomain(5, 15))
        r1, r2 = d1.refine_eq(d2)
        assert not r1.is_bot()


# ===========================================================================
# Section 9: LiftedDomain
# ===========================================================================

class TestLiftedDomain:
    def test_lift_normal(self):
        ld = LiftedDomain.lift(IntervalDomain(1, 5))
        assert ld.state == LiftState.NORMAL
        assert not ld.has_error()
        assert ld.may_be_normal()

    def test_error_state(self):
        ld = LiftedDomain.error(IntervalDomain(), "div by zero")
        assert ld.state == LiftState.ERROR
        assert ld.has_error()
        assert not ld.may_be_normal()
        assert ld.error_info == "div by zero"

    def test_join_normal_error(self):
        n = LiftedDomain.lift(IntervalDomain(1, 5))
        e = LiftedDomain.error(IntervalDomain(), "overflow")
        j = n.join(e)
        assert j.state == LiftState.BOTH
        assert j.has_error()
        assert j.may_be_normal()

    def test_bot(self):
        ld = LiftedDomain.lift(IntervalDomain(1, 5))
        b = ld.bot()
        assert b.is_bot()

    def test_top(self):
        ld = LiftedDomain.lift(IntervalDomain(1, 5))
        t = ld.top()
        assert t.is_top()
        assert t.state == LiftState.BOTH

    def test_leq(self):
        n1 = LiftedDomain.lift(IntervalDomain(2, 4))
        n2 = LiftedDomain.lift(IntervalDomain(0, 10))
        assert n1.leq(n2)
        assert not n2.leq(n1)

    def test_meet(self):
        n1 = LiftedDomain.lift(IntervalDomain(0, 10))
        n2 = LiftedDomain.lift(IntervalDomain(5, 15))
        m = n1.meet(n2)
        assert m.value.lo == 5
        assert m.value.hi == 10

    def test_add(self):
        n1 = LiftedDomain.lift(IntervalDomain(1, 3))
        n2 = LiftedDomain.lift(IntervalDomain(10, 20))
        s = n1.add(n2)
        assert s.value.lo == 11
        assert s.value.hi == 23

    def test_sub(self):
        n1 = LiftedDomain.lift(IntervalDomain(10, 20))
        n2 = LiftedDomain.lift(IntervalDomain(1, 3))
        s = n1.sub(n2)
        assert s.value.lo == 7
        assert s.value.hi == 19

    def test_neg(self):
        n = LiftedDomain.lift(IntervalDomain(3, 5))
        m = n.neg()
        assert m.value.lo == -5
        assert m.value.hi == -3

    def test_error_propagates_through_ops(self):
        e = LiftedDomain.error(IntervalDomain(), "err")
        n = LiftedDomain.lift(IntervalDomain(1, 5))
        s = n.add(e)
        assert s.is_bot()  # error state with bot value

    def test_widen(self):
        n1 = LiftedDomain.lift(IntervalDomain(0, 5))
        n2 = LiftedDomain.lift(IntervalDomain(0, 10))
        w = n1.widen(n2)
        assert w.value.hi == INF  # widened

    def test_eq(self):
        n1 = LiftedDomain.lift(IntervalDomain(1, 5))
        n2 = LiftedDomain.lift(IntervalDomain(1, 5))
        assert n1.eq(n2)

    def test_refine_lt(self):
        n1 = LiftedDomain.lift(IntervalDomain(0, 10))
        n2 = LiftedDomain.lift(IntervalDomain(0, 10))
        r1, r2 = n1.refine_lt(n2)
        assert isinstance(r1, LiftedDomain)


# ===========================================================================
# Section 10: CardinalPowerDomain
# ===========================================================================

class TestCardinalPowerDomain:
    def test_create_and_get(self):
        d = CardinalPowerDomain(
            {'a': IntervalDomain(1, 5), 'b': IntervalDomain(10, 20)},
            IntervalDomain()
        )
        assert d.get('a').lo == 1
        assert d.get('b').lo == 10
        assert d.get('c').is_top()  # default

    def test_set(self):
        d = CardinalPowerDomain({}, IntervalDomain())
        d2 = d.set('x', IntervalDomain(3, 7))
        assert d2.get('x').lo == 3

    def test_join(self):
        d1 = CardinalPowerDomain({'a': IntervalDomain(1, 5)}, IntervalDomain(1, 0))
        d2 = CardinalPowerDomain({'a': IntervalDomain(3, 8)}, IntervalDomain(1, 0))
        j = d1.join(d2)
        assert j.get('a').lo == 1
        assert j.get('a').hi == 8

    def test_meet(self):
        d1 = CardinalPowerDomain({'a': IntervalDomain(0, 10)}, IntervalDomain())
        d2 = CardinalPowerDomain({'a': IntervalDomain(5, 15)}, IntervalDomain())
        m = d1.meet(d2)
        assert m.get('a').lo == 5
        assert m.get('a').hi == 10

    def test_top(self):
        d = CardinalPowerDomain({'a': IntervalDomain(1, 5)}, IntervalDomain())
        t = d.top()
        assert t.is_top()

    def test_bot(self):
        d = CardinalPowerDomain({'a': IntervalDomain(1, 5)}, IntervalDomain())
        b = d.bot()
        assert b.is_bot()

    def test_leq(self):
        d1 = CardinalPowerDomain({'a': IntervalDomain(2, 4)}, IntervalDomain(1, 0))
        d2 = CardinalPowerDomain({'a': IntervalDomain(0, 10)}, IntervalDomain())
        assert d1.leq(d2)

    def test_eq(self):
        d1 = CardinalPowerDomain({'a': IntervalDomain(1, 5)}, IntervalDomain())
        d2 = CardinalPowerDomain({'a': IntervalDomain(1, 5)}, IntervalDomain())
        assert d1.eq(d2)

    def test_widen(self):
        d1 = CardinalPowerDomain({'a': IntervalDomain(0, 5)}, IntervalDomain())
        d2 = CardinalPowerDomain({'a': IntervalDomain(0, 10)}, IntervalDomain())
        w = d1.widen(d2)
        assert w.get('a').hi == INF


# ===========================================================================
# Section 11: CompositionInterpreter with Single Domain
# ===========================================================================

class TestCompositionInterpreterSingle:
    def test_let_assignment(self):
        r = analyze_single_domain("let x = 5;", IntervalDomain)
        assert 'x' in r['bindings']

    def test_arithmetic(self):
        r = analyze_single_domain("let x = 3; let y = x + 2;", IntervalDomain)
        env = r['env']
        y = env.get('y')
        assert y.lo == 5 and y.hi == 5

    def test_subtraction(self):
        r = analyze_single_domain("let x = 10; let y = x - 3;", IntervalDomain)
        env = r['env']
        y = env.get('y')
        assert y.lo == 7 and y.hi == 7

    def test_multiplication(self):
        r = analyze_single_domain("let x = 3; let y = x * 4;", IntervalDomain)
        env = r['env']
        y = env.get('y')
        assert y.lo == 12 and y.hi == 12

    def test_negation(self):
        r = analyze_single_domain("let x = 5; let y = -x;", IntervalDomain)
        env = r['env']
        y = env.get('y')
        assert y.lo == -5 and y.hi == -5

    def test_if_branch(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        """
        r = analyze_single_domain(source, IntervalDomain)
        env = r['env']
        y = env.get('y')
        # Both branches possible, join
        assert y.lo <= 1 and y.hi >= 2

    def test_while_loop(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        r = analyze_single_domain(source, IntervalDomain)
        env = r['env']
        i = env.get('i')
        # After loop, i >= 10
        assert i.lo >= 10

    def test_div_zero_warning(self):
        source = """
        let x = 0;
        let y = 10 / x;
        """
        r = analyze_single_domain(source, IntervalDomain)
        assert any("division by zero" in w for w in r['warnings'])

    def test_function_call(self):
        source = """
        fn add(a, b) { return a + b; }
        let r = add(3, 4);
        """
        r = analyze_single_domain(source, IntervalDomain)
        assert 'add' in r['functions']

    def test_bool_literal(self):
        r = analyze_single_domain("let x = true;", IntervalDomain)
        env = r['env']
        x = env.get('x')
        assert x.lo == 1 and x.hi == 1


# ===========================================================================
# Section 12: CompositionInterpreter with Composed Domains
# ===========================================================================

class TestCompositionInterpreterComposed:
    def test_sign_interval_composition(self):
        r = analyze_with_composition(
            "let x = 5; let y = x + 1;",
            SignDomain, IntervalDomain
        )
        env = r['env']
        y = env.get('y')
        assert isinstance(y, ReducedProductDomain)
        assert y[0].value == SignValue.POS
        assert y[1].lo == 6 and y[1].hi == 6

    def test_three_domain_composition(self):
        r = analyze_with_composition(
            "let x = 4;",
            SignDomain, IntervalDomain, ParityDomain
        )
        env = r['env']
        x = env.get('x')
        assert x[0].value == SignValue.POS
        assert x[1].lo == 4 and x[1].hi == 4
        assert x[2].value == ParityValue.EVEN

    def test_composed_div_zero_check(self):
        source = """
        let x = 0;
        let y = 10 / x;
        """
        r = analyze_with_composition(source, SignDomain, IntervalDomain)
        assert any("division by zero" in w for w in r['warnings'])

    def test_composed_no_false_div_zero(self):
        source = """
        let x = 5;
        let y = 10 / x;
        """
        r = analyze_with_composition(source, SignDomain, IntervalDomain)
        div_warnings = [w for w in r['warnings'] if "division by zero" in w]
        assert len(div_warnings) == 0

    def test_composed_while_loop(self):
        source = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        r = analyze_with_composition(source, SignDomain, IntervalDomain)
        env = r['env']
        i = env.get('i')
        # After loop, sign should be non-negative
        sign_i = i[0]
        assert sign_i.value in (SignValue.POS, SignValue.NON_NEG, SignValue.TOP)

    def test_full_four_domain(self):
        source = "let x = 6; let y = x + 1;"
        r = analyze_with_composition(
            source,
            SignDomain, IntervalDomain, ConstDomain, ParityDomain
        )
        env = r['env']
        y = env.get('y')
        assert len(y.components) == 4
        # y = 7: POS, [7,7], Const(7), ODD
        assert y[0].value == SignValue.POS
        assert y[1].lo == 7 and y[1].hi == 7


# ===========================================================================
# Section 13: CompositionEnv
# ===========================================================================

class TestCompositionEnv:
    def test_get_default_top(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        env = CompositionEnv(factory)
        x = env.get('x')
        assert x[0].is_top()
        assert x[1].is_top()

    def test_set_and_get(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        env = CompositionEnv(factory)
        env.set('x', factory(5))
        x = env.get('x')
        assert x[0].value == SignValue.POS

    def test_copy_independent(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        env = CompositionEnv(factory)
        env.set('x', factory(5))
        env2 = env.copy()
        env2.set('x', factory(-3))
        assert env.get('x')[0].value == SignValue.POS
        assert env2.get('x')[0].value == SignValue.NEG

    def test_join(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        e1 = CompositionEnv(factory)
        e1.set('x', factory(3))
        e2 = CompositionEnv(factory)
        e2.set('x', factory(7))
        joined = e1.join(e2)
        x = joined.get('x')
        assert x[1].lo == 3 and x[1].hi == 7

    def test_equals(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        e1 = CompositionEnv(factory)
        e1.set('x', factory(5))
        e2 = CompositionEnv(factory)
        e2.set('x', factory(5))
        assert e1.equals(e2)

    def test_widen(self):
        factory = compose_domains(SignDomain, IntervalDomain)
        e1 = CompositionEnv(factory)
        e1.set('x', factory(0))
        e2 = CompositionEnv(factory)
        e2.set('x', factory(10))
        w = e1.widen(e2)
        x = w.get('x')
        assert x[1].hi == INF  # widened upper bound


# ===========================================================================
# Section 14: PrecisionComparator
# ===========================================================================

class TestPrecisionComparator:
    def test_compare_sign_vs_interval(self):
        source = "let x = 0; let y = 10 / x;"
        f1 = compose_domains(SignDomain)
        f2 = compose_domains(IntervalDomain)

        # Create single-domain factories
        def sign_factory(value=None):
            if value is None:
                return SignDomain(SignValue.TOP)
            return SignDomain.from_concrete(value)

        def interval_factory(value=None):
            if value is None:
                return IntervalDomain()
            return IntervalDomain.from_concrete(value)

        result = compare_compositions(source, sign_factory, "Sign",
                                       interval_factory, "Interval")
        assert isinstance(result, ComparisonResult)
        assert result.domain1_name == "Sign"
        assert result.domain2_name == "Interval"

    def test_compare_has_bindings(self):
        source = "let x = 5;"
        def f1(value=None):
            if value is None:
                return IntervalDomain()
            return IntervalDomain.from_concrete(value)
        def f2(value=None):
            if value is None:
                return SignDomain(SignValue.TOP)
            return SignDomain.from_concrete(value)
        result = compare_compositions(source, f1, "Interval", f2, "Sign")
        assert 'x' in result.domain1_bindings
        assert 'x' in result.domain2_bindings


# ===========================================================================
# Section 15: Full Composition Analysis
# ===========================================================================

class TestFullCompositionAnalysis:
    def test_returns_all_configs(self):
        source = "let x = 5;"
        result = full_composition_analysis(source)
        assert 'Sign' in result['results']
        assert 'Interval' in result['results']
        assert 'Sign+Interval' in result['results']
        assert 'Sign+Interval+Parity' in result['results']
        assert 'Sign+Interval+Const' in result['results']

    def test_bindings_present(self):
        source = "let x = 5;"
        result = full_composition_analysis(source)
        for name, data in result['results'].items():
            assert 'x' in data['bindings'], f"Missing x in {name}"

    def test_div_zero_warnings(self):
        source = "let x = 0; let y = 10 / x;"
        result = full_composition_analysis(source)
        # All should detect div-by-zero
        for name, data in result['results'].items():
            assert data['warning_count'] > 0, f"No warnings in {name}"

    def test_summary_string(self):
        source = "let x = 5; let y = x + 1;"
        s = composition_summary(source)
        assert "Domain Composition Analysis" in s
        assert "Sign" in s
        assert "Interval" in s


# ===========================================================================
# Section 16: Cross-Domain Reduction Correctness
# ===========================================================================

class TestReductionCorrectness:
    def test_sign_interval_mutual_reduction(self):
        """Sign POS + Interval [-5, 10] should reduce to POS + [1, 10]"""
        factory = compose_domains(SignDomain, IntervalDomain)
        # Manually create a product with inconsistency
        from domain_functor import ReducedProductDomain
        reducers = find_builtin_reducers([SignDomain, IntervalDomain])
        val = ReducedProductDomain(
            [SignDomain(SignValue.POS), IntervalDomain(-5, 10)],
            reducers
        )
        assert val[0].value == SignValue.POS
        assert val[1].lo == 1  # tightened

    def test_const_propagates_to_both(self):
        """Const(5) + Sign(TOP) + Interval(-inf, inf) should reduce to POS + [5,5]"""
        reducers = find_builtin_reducers([ConstDomain, SignDomain, IntervalDomain])
        val = ReducedProductDomain(
            [ConstDomain.from_concrete(5), SignDomain(SignValue.TOP), IntervalDomain()],
            reducers
        )
        assert val[1].value == SignValue.POS
        assert val[2].lo == 5 and val[2].hi == 5

    def test_contradiction_detected(self):
        """Const(5) + Sign(NEG) should be BOT"""
        reducers = find_builtin_reducers([ConstDomain, SignDomain])
        val = ReducedProductDomain(
            [ConstDomain.from_concrete(5), SignDomain(SignValue.NEG)],
            reducers
        )
        assert val[0].is_bot()
        assert val[1].is_bot()

    def test_parity_interval_tightening(self):
        """Even + [3, 10] should become Even + [4, 10]"""
        reducers = find_builtin_reducers([ParityDomain, IntervalDomain])
        val = ReducedProductDomain(
            [ParityDomain(ParityValue.EVEN), IntervalDomain(3, 10)],
            reducers
        )
        assert val[1].lo == 4

    def test_full_chain_reduction(self):
        """Const(0) should set: Sign=ZERO, Interval=[0,0], Parity=EVEN"""
        reducers = find_builtin_reducers([ConstDomain, SignDomain, IntervalDomain, ParityDomain])
        val = ReducedProductDomain(
            [ConstDomain.from_concrete(0), SignDomain(SignValue.TOP),
             IntervalDomain(), ParityDomain(ParityValue.TOP)],
            reducers
        )
        assert val[1].value == SignValue.ZERO
        assert val[2].lo == 0 and val[2].hi == 0
        assert val[3].value == ParityValue.EVEN


# ===========================================================================
# Section 17: Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_source(self):
        r = analyze_single_domain("", IntervalDomain)
        assert r['bindings'] == {}
        assert r['warnings'] == []

    def test_print_statement(self):
        r = analyze_single_domain("let x = 5; print(x);", IntervalDomain)
        assert 'x' in r['bindings']

    def test_multiple_assignments(self):
        source = "let x = 1; x = 2; x = 3;"
        r = analyze_single_domain(source, IntervalDomain)
        env = r['env']
        x = env.get('x')
        assert x.lo == 3 and x.hi == 3

    def test_nested_if(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            if (x > 3) {
                y = 1;
            } else {
                y = 2;
            }
        } else {
            y = 3;
        }
        """
        r = analyze_single_domain(source, IntervalDomain)
        env = r['env']
        y = env.get('y')
        assert y.lo <= 1 and y.hi >= 3

    def test_large_constant(self):
        r = analyze_single_domain("let x = 1000000;", IntervalDomain)
        env = r['env']
        x = env.get('x')
        assert x.lo == 1000000

    def test_negative_literal(self):
        r = analyze_single_domain("let x = -7;", IntervalDomain)
        env = r['env']
        x = env.get('x')
        assert x.lo == -7 and x.hi == -7

    def test_modulo_zero_warning(self):
        source = "let x = 0; let y = 10 % x;"
        r = analyze_single_domain(source, IntervalDomain)
        assert any("modulo by zero" in w for w in r['warnings'])


# ===========================================================================
# Section 18: Disjunctive Domain with Interpreter
# ===========================================================================

class TestDisjunctiveInterpreter:
    def test_disjunctive_wrap_interval(self):
        """DisjunctiveDomain wrapping IntervalDomain preserves precision."""
        def factory(value=None):
            if value is None:
                return DisjunctiveDomain.wrap(IntervalDomain())
            return DisjunctiveDomain.wrap(IntervalDomain.from_concrete(value))

        interp = CompositionInterpreter(factory)
        r = interp.analyze("let x = 5;")
        env = r['env']
        x = env.get('x')
        assert isinstance(x, DisjunctiveDomain)
        assert len(x.disjuncts) == 1
        assert x.disjuncts[0].lo == 5

    def test_disjunctive_if_preserves_both(self):
        """DisjunctiveDomain should keep both branches as disjuncts."""
        def factory(value=None):
            if value is None:
                return DisjunctiveDomain.wrap(IntervalDomain())
            return DisjunctiveDomain.wrap(IntervalDomain.from_concrete(value))

        interp = CompositionInterpreter(factory)
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 100;
        }
        """
        r = interp.analyze(source)
        env = r['env']
        y = env.get('y')
        # With disjunctive, we should have separate disjuncts
        assert isinstance(y, DisjunctiveDomain)


# ===========================================================================
# Section 19: Lifted Domain with Interpreter
# ===========================================================================

class TestLiftedInterpreter:
    def test_lifted_basic(self):
        def factory(value=None):
            if value is None:
                return LiftedDomain.lift(IntervalDomain())
            return LiftedDomain.lift(IntervalDomain.from_concrete(value))

        interp = CompositionInterpreter(factory)
        r = interp.analyze("let x = 5; let y = x + 1;")
        env = r['env']
        y = env.get('y')
        assert isinstance(y, LiftedDomain)
        assert y.state == LiftState.NORMAL
        assert y.value.lo == 6 and y.value.hi == 6

    def test_lifted_negation(self):
        def factory(value=None):
            if value is None:
                return LiftedDomain.lift(IntervalDomain())
            return LiftedDomain.lift(IntervalDomain.from_concrete(value))

        interp = CompositionInterpreter(factory)
        r = interp.analyze("let x = 5; let y = -x;")
        env = r['env']
        y = env.get('y')
        assert y.value.lo == -5 and y.value.hi == -5


# ===========================================================================
# Section 20: Cardinal Power Domain Integration
# ===========================================================================

class TestCardinalPowerIntegration:
    def test_create_with_named_keys(self):
        d = CardinalPowerDomain(
            {0: IntervalDomain(0, 0), 1: IntervalDomain(1, 1), 2: IntervalDomain(2, 2)},
            IntervalDomain()
        )
        assert d.get(0).lo == 0
        assert d.get(1).lo == 1
        assert d.get(2).lo == 2

    def test_update_key(self):
        d = CardinalPowerDomain({0: IntervalDomain(0, 0)}, IntervalDomain())
        d2 = d.set(0, IntervalDomain(5, 5))
        assert d.get(0).lo == 0  # original unchanged
        assert d2.get(0).lo == 5

    def test_join_merges_values(self):
        d1 = CardinalPowerDomain(
            {0: IntervalDomain(1, 3), 1: IntervalDomain(5, 5)},
            IntervalDomain(1, 0)
        )
        d2 = CardinalPowerDomain(
            {0: IntervalDomain(2, 7), 1: IntervalDomain(10, 10)},
            IntervalDomain(1, 0)
        )
        j = d1.join(d2)
        assert j.get(0).lo == 1 and j.get(0).hi == 7
        assert j.get(1).lo == 5 and j.get(1).hi == 10

    def test_repr(self):
        d = CardinalPowerDomain({'x': IntervalDomain(1, 5)}, IntervalDomain())
        s = repr(d)
        assert 'CardPow' in s


# ===========================================================================
# Section 21: May-Be-Zero Detection
# ===========================================================================

class TestMayBeZero:
    def test_interval_contains_zero(self):
        interp = CompositionInterpreter(lambda v=None: IntervalDomain() if v is None else IntervalDomain.from_concrete(v))
        assert interp._may_be_zero(IntervalDomain(-5, 5))
        assert interp._may_be_zero(IntervalDomain(0, 0))
        assert not interp._may_be_zero(IntervalDomain(1, 10))
        assert not interp._may_be_zero(IntervalDomain(-10, -1))

    def test_sign_may_be_zero(self):
        interp = CompositionInterpreter(lambda v=None: SignDomain(SignValue.TOP))
        assert interp._may_be_zero(SignDomain(SignValue.ZERO))
        assert interp._may_be_zero(SignDomain(SignValue.NON_NEG))
        assert interp._may_be_zero(SignDomain(SignValue.TOP))
        assert not interp._may_be_zero(SignDomain(SignValue.POS))
        assert not interp._may_be_zero(SignDomain(SignValue.NEG))

    def test_const_may_be_zero(self):
        interp = CompositionInterpreter(lambda v=None: ConstDomain())
        assert interp._may_be_zero(ConstDomain.from_concrete(0))
        assert not interp._may_be_zero(ConstDomain.from_concrete(5))
        assert interp._may_be_zero(ConstDomain())  # TOP

    def test_product_all_must_agree(self):
        interp = CompositionInterpreter(lambda v=None: ProductDomain([SignDomain(SignValue.TOP), IntervalDomain()]))
        # Sign=POS, Interval=[-5,5] -> sign says no zero
        p = ProductDomain([SignDomain(SignValue.POS), IntervalDomain(-5, 5)])
        assert not interp._may_be_zero(p)


# ===========================================================================
# Section 22: Builder Fluent API
# ===========================================================================

class TestBuilderFluentAPI:
    def test_chain_adds(self):
        builder = ReducedProductBuilder()
        result = builder.add(SignDomain).add(IntervalDomain)
        assert result is builder

    def test_chain_auto_reduce(self):
        builder = ReducedProductBuilder()
        result = builder.add(SignDomain).add(IntervalDomain).auto_reduce()
        assert result is builder

    def test_chain_fixpoint(self):
        builder = ReducedProductBuilder()
        result = builder.add(SignDomain).add(IntervalDomain).auto_reduce().fixpoint()
        assert result is builder

    def test_chain_add_reducer(self):
        builder = ReducedProductBuilder()
        result = builder.add(SignDomain).add_reducer(lambda c: c)
        assert result is builder


# ===========================================================================
# Section 23: Composition with Programs
# ===========================================================================

class TestCompositionPrograms:
    def test_accumulator(self):
        source = """
        let sum = 0;
        let i = 0;
        while (i < 5) {
            sum = sum + i;
            i = i + 1;
        }
        """
        r = analyze_with_composition(source, SignDomain, IntervalDomain)
        env = r['env']
        i = env.get('i')
        assert i[0].value in (SignValue.POS, SignValue.NON_NEG, SignValue.TOP)

    def test_conditional_sign(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = -1;
        }
        """
        r = analyze_with_composition(source, SignDomain, IntervalDomain)
        env = r['env']
        y = env.get('y')
        # Both branches merge
        assert not y.is_bot()

    def test_nested_arithmetic(self):
        source = "let a = 2; let b = 3; let c = a * b + 1;"
        r = analyze_with_composition(source, SignDomain, IntervalDomain, ConstDomain)
        env = r['env']
        c = env.get('c')
        assert c[2].is_const() and c[2].value == 7  # constant propagation


# ===========================================================================
# Section 24: Stress Tests
# ===========================================================================

class TestStress:
    def test_long_chain(self):
        stmts = ["let x0 = 1;"]
        for i in range(1, 20):
            stmts.append(f"let x{i} = x{i-1} + 1;")
        source = "\n".join(stmts)
        r = analyze_with_composition(source, SignDomain, IntervalDomain)
        env = r['env']
        x19 = env.get('x19')
        assert x19[1].lo == 20 and x19[1].hi == 20

    def test_deep_nesting(self):
        source = """
        let x = 10;
        if (x > 5) {
            if (x > 7) {
                if (x > 9) {
                    x = x + 1;
                }
            }
        }
        """
        r = analyze_with_composition(source, SignDomain, IntervalDomain)
        assert not r['env'].get('x').is_bot()

    def test_multiple_warnings(self):
        source = """
        let a = 0;
        let b = 0;
        let c = 10 / a;
        let d = 20 / b;
        """
        r = analyze_single_domain(source, IntervalDomain)
        div_warnings = [w for w in r['warnings'] if "division by zero" in w]
        assert len(div_warnings) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
