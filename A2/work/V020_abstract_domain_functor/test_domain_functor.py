"""
Tests for V020: Abstract Domain Functor
"""

import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain_functor import (
    # Protocol
    AbstractDomain,
    # Concrete domains
    SignDomain, SignValue, IntervalDomain, ConstDomain, ParityDomain, ParityValue,
    FlatDomain,
    # Functors
    ProductDomain, ReducedProductDomain, PowersetDomain,
    # Reducers
    sign_interval_reducer, const_all_reducer, interval_const_reducer,
    parity_interval_reducer, standard_reducers,
    # Factories
    make_sign_interval, make_full_product, create_custom_domain,
    # Analysis
    analyze_with_domain, analyze_sign, analyze_interval,
    analyze_sign_interval, analyze_full, compare_domains,
    get_variable_info, FunctorInterpreter, DomainEnv,
    INF, NEG_INF,
)


# ===================================================================
# Section 1: Sign Domain
# ===================================================================

class TestSignDomain:
    def test_from_concrete(self):
        assert SignDomain.from_concrete(5).value == SignValue.POS
        assert SignDomain.from_concrete(-3).value == SignValue.NEG
        assert SignDomain.from_concrete(0).value == SignValue.ZERO

    def test_top_bot(self):
        t = SignDomain(SignValue.TOP)
        b = SignDomain(SignValue.BOT)
        assert t.is_top()
        assert b.is_bot()
        assert not t.is_bot()
        assert not b.is_top()

    def test_join(self):
        pos = SignDomain(SignValue.POS)
        neg = SignDomain(SignValue.NEG)
        zero = SignDomain(SignValue.ZERO)
        assert pos.join(zero).value == SignValue.NON_NEG
        assert neg.join(zero).value == SignValue.NON_POS
        assert pos.join(neg).value == SignValue.TOP

    def test_meet(self):
        nn = SignDomain(SignValue.NON_NEG)
        np = SignDomain(SignValue.NON_POS)
        assert nn.meet(np).value == SignValue.ZERO

    def test_leq(self):
        bot = SignDomain(SignValue.BOT)
        pos = SignDomain(SignValue.POS)
        nn = SignDomain(SignValue.NON_NEG)
        top = SignDomain(SignValue.TOP)
        assert bot.leq(pos)
        assert pos.leq(nn)
        assert nn.leq(top)
        assert not top.leq(pos)

    def test_add(self):
        pos = SignDomain(SignValue.POS)
        neg = SignDomain(SignValue.NEG)
        assert pos.add(pos).value == SignValue.POS
        assert neg.add(neg).value == SignValue.NEG
        assert pos.add(neg).value == SignValue.TOP

    def test_mul(self):
        pos = SignDomain(SignValue.POS)
        neg = SignDomain(SignValue.NEG)
        assert neg.mul(neg).value == SignValue.NON_NEG
        assert pos.mul(neg).value == SignValue.NON_POS

    def test_neg(self):
        assert SignDomain(SignValue.POS).neg().value == SignValue.NEG
        assert SignDomain(SignValue.ZERO).neg().value == SignValue.ZERO

    def test_widen(self):
        # Sign widen is just join
        pos = SignDomain(SignValue.POS)
        neg = SignDomain(SignValue.NEG)
        assert pos.widen(neg).value == SignValue.TOP


# ===================================================================
# Section 2: Interval Domain
# ===================================================================

class TestIntervalDomain:
    def test_from_concrete(self):
        iv = IntervalDomain.from_concrete(5)
        assert iv.lo == 5 and iv.hi == 5

    def test_top_bot(self):
        t = IntervalDomain()
        b = IntervalDomain(1, 0)
        assert t.is_top()
        assert b.is_bot()

    def test_join(self):
        a = IntervalDomain(1, 5)
        b = IntervalDomain(3, 10)
        j = a.join(b)
        assert j.lo == 1 and j.hi == 10

    def test_meet(self):
        a = IntervalDomain(1, 5)
        b = IntervalDomain(3, 10)
        m = a.meet(b)
        assert m.lo == 3 and m.hi == 5

    def test_widen(self):
        a = IntervalDomain(0, 5)
        b = IntervalDomain(0, 10)
        w = a.widen(b)
        assert w.lo == 0 and w.hi == INF

    def test_narrow(self):
        a = IntervalDomain(0, INF)
        b = IntervalDomain(0, 10)
        n = a.narrow(b)
        assert n.lo == 0 and n.hi == 10

    def test_add(self):
        a = IntervalDomain(1, 5)
        b = IntervalDomain(2, 3)
        r = a.add(b)
        assert r.lo == 3 and r.hi == 8

    def test_sub(self):
        a = IntervalDomain(5, 10)
        b = IntervalDomain(1, 3)
        r = a.sub(b)
        assert r.lo == 2 and r.hi == 9

    def test_mul(self):
        a = IntervalDomain(-2, 3)
        b = IntervalDomain(1, 4)
        r = a.mul(b)
        assert r.lo == -8 and r.hi == 12

    def test_neg(self):
        a = IntervalDomain(2, 5)
        n = a.neg()
        assert n.lo == -5 and n.hi == -2

    def test_refine_lt(self):
        a = IntervalDomain(0, 10)
        b = IntervalDomain(5, 15)
        ra, rb = a.refine_lt(b)
        assert ra.hi == 10  # min(10, 15-1)=10
        assert rb.lo == 5   # max(5, 0+1)=5

    def test_refine_eq(self):
        a = IntervalDomain(0, 10)
        b = IntervalDomain(5, 15)
        ra, rb = a.refine_eq(b)
        assert ra.lo == 5 and ra.hi == 10
        assert rb.lo == 5 and rb.hi == 10

    def test_contains(self):
        a = IntervalDomain(1, 5)
        assert a.contains(3)
        assert not a.contains(6)


# ===================================================================
# Section 3: Constant Domain
# ===================================================================

class TestConstDomain:
    def test_from_concrete(self):
        c = ConstDomain.from_concrete(42)
        assert c.is_const() and c.value == 42

    def test_top_bot(self):
        assert ConstDomain().is_top()
        assert ConstDomain(kind='bot').is_bot()

    def test_join(self):
        a = ConstDomain.from_concrete(5)
        b = ConstDomain.from_concrete(5)
        assert a.join(b).value == 5
        c = ConstDomain.from_concrete(3)
        assert a.join(c).is_top()

    def test_meet(self):
        a = ConstDomain.from_concrete(5)
        b = ConstDomain()
        assert a.meet(b).value == 5
        c = ConstDomain.from_concrete(3)
        assert a.meet(c).is_bot()

    def test_add(self):
        a = ConstDomain.from_concrete(3)
        b = ConstDomain.from_concrete(4)
        assert a.add(b).value == 7

    def test_mul(self):
        a = ConstDomain.from_concrete(3)
        b = ConstDomain.from_concrete(4)
        assert a.mul(b).value == 12

    def test_neg(self):
        a = ConstDomain.from_concrete(5)
        assert a.neg().value == -5

    def test_top_arithmetic(self):
        a = ConstDomain.from_concrete(5)
        b = ConstDomain()
        assert a.add(b).is_top()


# ===================================================================
# Section 4: Parity Domain
# ===================================================================

class TestParityDomain:
    def test_from_concrete(self):
        assert ParityDomain.from_concrete(4).value == ParityValue.EVEN
        assert ParityDomain.from_concrete(7).value == ParityValue.ODD

    def test_join(self):
        e = ParityDomain(ParityValue.EVEN)
        o = ParityDomain(ParityValue.ODD)
        assert e.join(o).is_top()
        assert e.join(e).value == ParityValue.EVEN

    def test_add(self):
        e = ParityDomain(ParityValue.EVEN)
        o = ParityDomain(ParityValue.ODD)
        assert e.add(e).value == ParityValue.EVEN
        assert o.add(o).value == ParityValue.EVEN
        assert e.add(o).value == ParityValue.ODD

    def test_mul(self):
        e = ParityDomain(ParityValue.EVEN)
        o = ParityDomain(ParityValue.ODD)
        assert e.mul(o).value == ParityValue.EVEN
        assert o.mul(o).value == ParityValue.ODD

    def test_neg(self):
        e = ParityDomain(ParityValue.EVEN)
        assert e.neg().value == ParityValue.EVEN


# ===================================================================
# Section 5: Flat Domain
# ===================================================================

class TestFlatDomain:
    def test_basic(self):
        a = FlatDomain.from_concrete("hello")
        assert a.is_val() and a.value == "hello"

    def test_join(self):
        a = FlatDomain.from_concrete("a")
        b = FlatDomain.from_concrete("a")
        assert a.join(b).value == "a"
        c = FlatDomain.from_concrete("b")
        assert a.join(c).is_top()

    def test_meet(self):
        a = FlatDomain.from_concrete("a")
        t = FlatDomain()
        assert a.meet(t).value == "a"
        b = FlatDomain.from_concrete("b")
        assert a.meet(b).is_bot()

    def test_leq(self):
        bot = FlatDomain(kind='bot')
        a = FlatDomain.from_concrete("x")
        top = FlatDomain()
        assert bot.leq(a)
        assert a.leq(top)
        assert not top.leq(a)


# ===================================================================
# Section 6: Product Domain
# ===================================================================

class TestProductDomain:
    def test_creation(self):
        p = ProductDomain([SignDomain(SignValue.POS), IntervalDomain(1, 10)])
        assert len(p) == 2
        assert p[0].value == SignValue.POS
        assert p[1].lo == 1

    def test_from_concrete(self):
        p = ProductDomain.from_concrete_with_types(5, [SignDomain, IntervalDomain])
        assert p[0].value == SignValue.POS
        assert p[1].lo == 5 and p[1].hi == 5

    def test_join(self):
        a = ProductDomain.from_concrete_with_types(5, [SignDomain, IntervalDomain])
        b = ProductDomain.from_concrete_with_types(-3, [SignDomain, IntervalDomain])
        j = a.join(b)
        assert j[0].value == SignValue.TOP
        assert j[1].lo == -3 and j[1].hi == 5

    def test_meet(self):
        a = ProductDomain([SignDomain(SignValue.NON_NEG), IntervalDomain(0, 10)])
        b = ProductDomain([SignDomain(SignValue.NON_POS), IntervalDomain(5, 15)])
        m = a.meet(b)
        assert m[0].value == SignValue.ZERO
        assert m[1].lo == 5 and m[1].hi == 10

    def test_add(self):
        a = ProductDomain.from_concrete_with_types(3, [SignDomain, IntervalDomain])
        b = ProductDomain.from_concrete_with_types(4, [SignDomain, IntervalDomain])
        r = a.add(b)
        assert r[0].value == SignValue.POS
        assert r[1].lo == 7 and r[1].hi == 7

    def test_is_bot(self):
        p = ProductDomain([SignDomain(SignValue.BOT), IntervalDomain()])
        assert p.is_bot()

    def test_is_top(self):
        p = ProductDomain([SignDomain(), IntervalDomain()])
        assert p.is_top()


# ===================================================================
# Section 7: Reduced Product Domain
# ===================================================================

class TestReducedProductDomain:
    def test_sign_interval_reduction(self):
        # Constant 5: sign should be POS, interval [5,5]
        rp = ReducedProductDomain(
            [SignDomain(SignValue.TOP), IntervalDomain(5, 5)],
            [sign_interval_reducer])
        assert rp[0].value == SignValue.POS
        assert rp[1].lo == 5

    def test_const_all_reduction(self):
        rp = ReducedProductDomain(
            [SignDomain(), IntervalDomain(), ConstDomain.from_concrete(7)],
            [const_all_reducer])
        assert rp[0].value == SignValue.POS
        assert rp[1].lo == 7 and rp[1].hi == 7

    def test_interval_const_reduction(self):
        rp = ReducedProductDomain(
            [IntervalDomain(3, 3), ConstDomain()],
            [interval_const_reducer])
        assert rp[1].value == 3

    def test_parity_interval_reduction(self):
        rp = ReducedProductDomain(
            [IntervalDomain(1, 5), ParityDomain(ParityValue.EVEN)],
            [parity_interval_reducer])
        assert rp[0].lo == 2 and rp[0].hi == 4

    def test_bot_propagation(self):
        rp = ReducedProductDomain(
            [IntervalDomain(5, 5), ParityDomain(ParityValue.EVEN)],
            [parity_interval_reducer])
        # 5 is odd but parity says EVEN -> BOT
        assert rp.is_bot()

    def test_full_reduction_chain(self):
        rp = ReducedProductDomain(
            [SignDomain(), IntervalDomain(5, 5), ConstDomain(), ParityDomain()],
            standard_reducers())
        # Singleton [5,5] -> const 5 -> POS sign -> ODD parity
        assert rp[0].value == SignValue.POS
        assert rp[2].value == 5
        assert rp[3].value == ParityValue.ODD

    def test_add_preserves_reduction(self):
        factory = make_full_product()
        a = factory(3)
        b = factory(4)
        r = a.add(b)
        # 3+4=7: POS, [7,7], Const(7), ODD
        assert r[0].value == SignValue.POS
        assert r[1].lo == 7 and r[1].hi == 7
        assert r[2].value == 7
        assert r[3].value == ParityValue.ODD


# ===================================================================
# Section 8: Powerset Domain
# ===================================================================

class TestPowersetDomain:
    def test_singleton(self):
        p = PowersetDomain.singleton(IntervalDomain(1, 5))
        assert not p.is_bot() and not p.is_top()
        assert len(p.elements) == 1

    def test_join(self):
        a = PowersetDomain.singleton(IntervalDomain(1, 5))
        b = PowersetDomain.singleton(IntervalDomain(10, 15))
        j = a.join(b)
        assert len(j.elements) == 2

    def test_meet(self):
        a = PowersetDomain.singleton(IntervalDomain(1, 10))
        b = PowersetDomain.singleton(IntervalDomain(5, 15))
        m = a.meet(b)
        assert len(m.elements) == 1
        elem = list(m.elements)[0]
        assert elem.lo == 5 and elem.hi == 10

    def test_add(self):
        a = PowersetDomain.singleton(IntervalDomain(1, 5))
        b = PowersetDomain.singleton(IntervalDomain(10, 20))
        r = a.add(b)
        assert len(r.elements) == 1
        elem = list(r.elements)[0]
        assert elem.lo == 11 and elem.hi == 25

    def test_bot_operations(self):
        b = PowersetDomain(element_type=IntervalDomain)
        assert b.is_bot()
        a = PowersetDomain.singleton(IntervalDomain(1, 5))
        assert a.add(b).is_bot()

    def test_collapse(self):
        # max_size=2, add 3 elements -> should collapse
        p1 = PowersetDomain.singleton(IntervalDomain(1, 1), max_size=2)
        p2 = PowersetDomain.singleton(IntervalDomain(5, 5), max_size=2)
        p3 = PowersetDomain.singleton(IntervalDomain(10, 10), max_size=2)
        j = p1.join(p2).join(p3)
        assert len(j.elements) <= 2

    def test_leq(self):
        a = PowersetDomain.singleton(IntervalDomain(1, 5))
        b = PowersetDomain.singleton(IntervalDomain(0, 10))
        assert a.leq(b)
        assert not b.leq(a)


# ===================================================================
# Section 9: DomainEnv
# ===================================================================

class TestDomainEnv:
    def test_get_set(self):
        factory = lambda v=None: IntervalDomain.from_concrete(v) if v is not None else IntervalDomain()
        env = DomainEnv(factory)
        env.set('x', IntervalDomain(0, 10))
        assert env.get('x').lo == 0
        assert env.get('y').is_top()  # unknown -> TOP

    def test_join(self):
        factory = lambda v=None: IntervalDomain.from_concrete(v) if v is not None else IntervalDomain()
        e1 = DomainEnv(factory)
        e1.set('x', IntervalDomain(0, 5))
        e2 = DomainEnv(factory)
        e2.set('x', IntervalDomain(3, 10))
        j = e1.join(e2)
        assert j.get('x').lo == 0 and j.get('x').hi == 10

    def test_widen(self):
        factory = lambda v=None: IntervalDomain.from_concrete(v) if v is not None else IntervalDomain()
        e1 = DomainEnv(factory)
        e1.set('x', IntervalDomain(0, 5))
        e2 = DomainEnv(factory)
        e2.set('x', IntervalDomain(0, 10))
        w = e1.widen(e2)
        assert w.get('x').lo == 0 and w.get('x').hi == INF

    def test_equals(self):
        factory = lambda v=None: IntervalDomain.from_concrete(v) if v is not None else IntervalDomain()
        e1 = DomainEnv(factory)
        e1.set('x', IntervalDomain(0, 5))
        e2 = DomainEnv(factory)
        e2.set('x', IntervalDomain(0, 5))
        assert e1.equals(e2)

    def test_copy(self):
        factory = lambda v=None: IntervalDomain.from_concrete(v) if v is not None else IntervalDomain()
        e1 = DomainEnv(factory)
        e1.set('x', IntervalDomain(0, 5))
        e2 = e1.copy()
        e2.set('x', IntervalDomain(0, 10))
        assert e1.get('x').hi == 5  # original unchanged


# ===================================================================
# Section 10: FunctorInterpreter - Basic
# ===================================================================

class TestFunctorInterpreterBasic:
    def test_let_decl(self):
        result = analyze_interval("let x = 5;")
        assert result['env'].get('x').lo == 5

    def test_assignment(self):
        result = analyze_interval("let x = 5; x = 10;")
        assert result['env'].get('x').lo == 10

    def test_arithmetic(self):
        result = analyze_interval("let x = 3; let y = 4; let z = x + y;")
        assert result['env'].get('z').lo == 7

    def test_subtraction(self):
        result = analyze_interval("let x = 10; let y = 3; let z = x - y;")
        assert result['env'].get('z').lo == 7

    def test_multiplication(self):
        result = analyze_interval("let x = 3; let y = 4; let z = x * y;")
        assert result['env'].get('z').lo == 12

    def test_negation(self):
        result = analyze_interval("let x = 5; let y = -x;")
        assert result['env'].get('y').lo == -5


# ===================================================================
# Section 11: FunctorInterpreter - Conditionals
# ===================================================================

class TestFunctorInterpreterConditionals:
    def test_if_then_else(self):
        src = "let x = 5; let y = 0; if (x > 0) { y = 1; } else { y = -1; }"
        result = analyze_interval(src)
        # Both branches: y is 1 or -1
        y = result['env'].get('y')
        assert y.lo == -1 and y.hi == 1

    def test_condition_refinement(self):
        src = "let x = 5; if (x > 3) { let y = x; }"
        result = analyze_sign(src)
        # x is POS (from constant 5), in then-branch it's still POS

    def test_nested_if(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            if (x > 10) {
                y = 2;
            } else {
                y = 1;
            }
        } else {
            y = -1;
        }
        """
        result = analyze_interval(src)
        y = result['env'].get('y')
        assert y.contains(1)


# ===================================================================
# Section 12: FunctorInterpreter - Loops
# ===================================================================

class TestFunctorInterpreterLoops:
    def test_simple_countdown(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_interval(src)
        i = result['env'].get('i')
        assert i.lo <= 0

    def test_countup(self):
        src = "let i = 0; while (i < 10) { i = i + 1; }"
        result = analyze_interval(src)
        i = result['env'].get('i')
        # After loop: i >= 10
        assert i.lo >= 10

    def test_accumulator(self):
        src = "let s = 0; let i = 0; while (i < 5) { s = s + i; i = i + 1; }"
        result = analyze_sign(src)
        s = result['env'].get('s')
        # Sum of non-negatives should be non-negative
        assert s.value in (SignValue.NON_NEG, SignValue.TOP, SignValue.POS, SignValue.ZERO)


# ===================================================================
# Section 13: Domain Composition via Functor
# ===================================================================

class TestDomainComposition:
    def test_sign_interval_product(self):
        src = "let x = 5;"
        result = analyze_sign_interval(src)
        v = result['env'].get('x')
        assert v[0].value == SignValue.POS
        assert v[1].lo == 5 and v[1].hi == 5

    def test_full_product(self):
        src = "let x = 6;"
        result = analyze_full(src)
        v = result['env'].get('x')
        assert v[0].value == SignValue.POS
        assert v[1].lo == 6 and v[1].hi == 6
        assert v[2].value == 6
        assert v[3].value == ParityValue.EVEN

    def test_reduction_in_analysis(self):
        src = "let x = 5; let y = x + 1;"
        result = analyze_full(src)
        y = result['env'].get('y')
        # 5+1=6: POS, [6,6], Const(6), EVEN
        assert y[0].value == SignValue.POS
        assert y[1].lo == 6 and y[1].hi == 6
        assert y[2].value == 6
        assert y[3].value == ParityValue.EVEN

    def test_custom_domain(self):
        factory = create_custom_domain(SignDomain, ParityDomain)
        result = analyze_with_domain("let x = 7;", factory)
        v = result['env'].get('x')
        assert v[0].value == SignValue.POS
        assert v[1].value == ParityValue.ODD

    def test_custom_with_reducers(self):
        factory = create_custom_domain(
            SignDomain, IntervalDomain,
            reducers=[sign_interval_reducer])
        result = analyze_with_domain("let x = 5;", factory)
        v = result['env'].get('x')
        assert v[0].value == SignValue.POS
        assert v[1].lo == 5

    def test_compare_domains(self):
        src = "let x = 5; let y = x + 1;"
        results = compare_domains(src)
        assert 'sign' in results
        assert 'interval' in results
        assert 'sign_interval' in results
        assert 'full' in results


# ===================================================================
# Section 14: Cross-Domain Precision
# ===================================================================

class TestCrossDomainPrecision:
    def test_const_propagation(self):
        src = "let x = 5; let y = x + x;"
        result = analyze_full(src)
        y = result['env'].get('y')
        assert y[2].value == 10

    def test_parity_from_arithmetic(self):
        src = "let x = 3; let y = 2; let z = x + y;"
        result = analyze_full(src)
        z = result['env'].get('z')
        # 3+2=5: ODD
        assert z[3].value == ParityValue.ODD

    def test_sign_from_subtraction(self):
        src = "let x = 10; let y = 3; let z = x - y;"
        result = analyze_full(src)
        z = result['env'].get('z')
        assert z[0].value == SignValue.POS

    def test_interval_arithmetic(self):
        src = "let x = 3; let y = 4; let z = x * y;"
        result = analyze_full(src)
        z = result['env'].get('z')
        assert z[1].lo == 12 and z[1].hi == 12

    def test_get_variable_info(self):
        src = "let a = 7; let b = a + 3;"
        v = get_variable_info(src, 'b')
        assert v[1].lo == 10 and v[1].hi == 10
        assert v[2].value == 10


# ===================================================================
# Section 15: Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = analyze_interval("")
        assert result['env'].names == set()

    def test_bot_propagation_in_product(self):
        rp = ReducedProductDomain(
            [SignDomain(SignValue.BOT), IntervalDomain(1, 5)],
            standard_reducers())
        assert rp.is_bot()

    def test_powerset_neg(self):
        a = PowersetDomain.singleton(IntervalDomain(1, 5))
        n = a.neg()
        elem = list(n.elements)[0]
        assert elem.lo == -5 and elem.hi == -1

    def test_flat_domain_arithmetic(self):
        # Flat domain has no meaningful arithmetic -> TOP
        a = FlatDomain.from_concrete("hello")
        b = FlatDomain.from_concrete("world")
        assert a.add(b).is_top()

    def test_domain_env_missing_var(self):
        factory = make_full_product()
        env = DomainEnv(factory)
        v = env.get('nonexistent')
        assert v.is_top()

    def test_multiple_reducers(self):
        rp = ReducedProductDomain(
            [SignDomain(), IntervalDomain(0, 0), ConstDomain(), ParityDomain()],
            standard_reducers())
        assert rp[0].value == SignValue.ZERO
        assert rp[2].value == 0
        assert rp[3].value == ParityValue.EVEN

    def test_product_widen(self):
        a = ProductDomain([IntervalDomain(0, 5), SignDomain(SignValue.POS)])
        b = ProductDomain([IntervalDomain(0, 10), SignDomain(SignValue.POS)])
        w = a.widen(b)
        assert w[0].hi == INF

    def test_reduced_product_narrow(self):
        a = ReducedProductDomain(
            [IntervalDomain(0, INF)], [])
        b = ReducedProductDomain(
            [IntervalDomain(0, 10)], [])
        n = a.narrow(b)
        assert n[0].hi == 10

    def test_sign_bot_arithmetic(self):
        bot = SignDomain(SignValue.BOT)
        pos = SignDomain(SignValue.POS)
        assert bot.add(pos).is_bot()
        assert bot.mul(pos).is_bot()

    def test_interval_bot_arithmetic(self):
        bot = IntervalDomain(1, 0)
        iv = IntervalDomain(1, 5)
        assert bot.add(iv).is_bot()

    def test_print_statement(self):
        result = analyze_interval("let x = 5; print(x);")
        assert result['env'].get('x').lo == 5

    def test_function_analysis(self):
        src = """
        fn add(a, b) { return a + b; }
        let result = add(3, 4);
        """
        result = analyze_full(src)
        assert 'add' in result['functions']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
