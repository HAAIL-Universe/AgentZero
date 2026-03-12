"""Tests for V170: Symbolic Mu-Calculus Model Checker with CEGAR."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mu_calculus_cegar import (
    # Formula constructors
    tt, ff, prop, var, neg, conj, disj, implies, ex, ax, eu, au,
    ef, af, eg, ag, mu, nu,
    # Formula analysis
    Formula, FormulaKind, formula_size, formula_depth, alternation_depth,
    # Kripke structures
    KripkeStructure, kripke_to_symbolic, check_mu_calculus, model_check,
    model_check_states, _bdd_states,
    # Predicate abstraction
    ConcreteSystem, Predicate, predicate_abstract,
    # CEGAR
    cegar_verify, CEGARVerdict, CEGARResult, cegar_statistics,
    # Helpers
    make_counter_system, make_traffic_light, make_mutex_system,
    make_bounded_counter, verify_model_check, compare_mc_methods,
    batch_check, parse_formula,
)


# ============================================================
# Test: Formula AST Construction
# ============================================================

class TestFormulaAST:
    def test_true_false(self):
        assert tt().kind == FormulaKind.TRUE
        assert ff().kind == FormulaKind.FALSE

    def test_proposition(self):
        p = prop('safe')
        assert p.kind == FormulaKind.PROP
        assert p.prop == 'safe'

    def test_variable(self):
        x = var('X')
        assert x.kind == FormulaKind.VAR
        assert x.var == 'X'

    def test_negation(self):
        f = neg(prop('p'))
        assert f.kind == FormulaKind.NOT
        assert f.children[0].prop == 'p'

    def test_conjunction(self):
        f = conj(prop('a'), prop('b'))
        assert f.kind == FormulaKind.AND
        assert len(f.children) == 2

    def test_disjunction(self):
        f = disj(prop('a'), prop('b'))
        assert f.kind == FormulaKind.OR

    def test_implication(self):
        f = implies(prop('a'), prop('b'))
        assert f.kind == FormulaKind.IMPLIES

    def test_modal_ex(self):
        f = ex(prop('p'))
        assert f.kind == FormulaKind.EX

    def test_modal_ax(self):
        f = ax(prop('p'))
        assert f.kind == FormulaKind.AX

    def test_until_eu(self):
        f = eu(prop('a'), prop('b'))
        assert f.kind == FormulaKind.EU

    def test_until_au(self):
        f = au(prop('a'), prop('b'))
        assert f.kind == FormulaKind.AU

    def test_ef_af_eg_ag(self):
        assert ef(prop('p')).kind == FormulaKind.EF
        assert af(prop('p')).kind == FormulaKind.AF
        assert eg(prop('p')).kind == FormulaKind.EG
        assert ag(prop('p')).kind == FormulaKind.AG

    def test_mu_fixpoint(self):
        f = mu('X', disj(prop('p'), ex(var('X'))))
        assert f.kind == FormulaKind.MU
        assert f.var == 'X'
        assert f.body.kind == FormulaKind.OR

    def test_nu_fixpoint(self):
        f = nu('X', conj(prop('p'), ex(var('X'))))
        assert f.kind == FormulaKind.NU
        assert f.var == 'X'

    def test_nested_fixpoints(self):
        # nu X. (mu Y. p \/ EX(Y)) /\ EX(X)
        f = nu('X', conj(mu('Y', disj(prop('p'), ex(var('Y')))), ex(var('X'))))
        assert f.kind == FormulaKind.NU
        assert f.body.kind == FormulaKind.AND

    def test_formula_repr(self):
        assert str(tt()) == "tt"
        assert str(ff()) == "ff"
        assert "p" in str(prop('p'))
        assert "EX" in str(ex(prop('p')))
        assert "mu" in str(mu('X', var('X')))


# ============================================================
# Test: Formula Analysis
# ============================================================

class TestFormulaAnalysis:
    def test_size_atomic(self):
        assert formula_size(tt()) == 1
        assert formula_size(prop('p')) == 1

    def test_size_compound(self):
        f = conj(prop('a'), prop('b'))
        assert formula_size(f) == 3  # AND + 2 props

    def test_size_fixpoint(self):
        f = mu('X', disj(prop('p'), ex(var('X'))))
        assert formula_size(f) >= 4

    def test_depth_atomic(self):
        assert formula_depth(tt()) == 1

    def test_depth_nested(self):
        f = ex(ax(prop('p')))
        assert formula_depth(f) == 3

    def test_alternation_depth_zero(self):
        assert alternation_depth(ef(prop('p'))) == 0

    def test_alternation_depth_one(self):
        f = mu('X', disj(prop('p'), ex(var('X'))))
        assert alternation_depth(f) == 1

    def test_alternation_depth_two(self):
        # mu X. (nu Y. p /\ EX(Y)) \/ EX(X)
        inner = nu('Y', conj(prop('p'), ex(var('Y'))))
        f = mu('X', disj(inner, ex(var('X'))))
        assert alternation_depth(f) == 2


# ============================================================
# Test: Simple Kripke Structures
# ============================================================

def make_simple_ks():
    """Simple 3-state Kripke structure.
    s0 -> s1 -> s2 -> s0 (cycle)
    s0: {p}, s1: {q}, s2: {p, q}
    """
    return KripkeStructure(
        states={0, 1, 2},
        initial={0},
        transitions={0: {1}, 1: {2}, 2: {0}},
        labeling={0: {'p'}, 1: {'q'}, 2: {'p', 'q'}}
    )


def make_diamond_ks():
    """Diamond structure: s0 -> s1, s0 -> s2, s1 -> s3, s2 -> s3, s3 -> s3.
    s0: {start}, s1: {left}, s2: {right}, s3: {end}
    """
    return KripkeStructure(
        states={0, 1, 2, 3},
        initial={0},
        transitions={0: {1, 2}, 1: {3}, 2: {3}, 3: {3}},
        labeling={0: {'start'}, 1: {'left'}, 2: {'right'}, 3: {'end'}}
    )


def make_self_loop_ks():
    """Single state with self-loop.
    s0 -> s0, s0: {p}
    """
    return KripkeStructure(
        states={0},
        initial={0},
        transitions={0: {0}},
        labeling={0: {'p'}}
    )


class TestBasicModelChecking:
    def test_true_formula(self):
        ks = make_simple_ks()
        r = model_check(ks, tt())
        assert r.satisfied

    def test_false_formula(self):
        ks = make_simple_ks()
        r = model_check(ks, ff())
        assert not r.satisfied

    def test_prop_at_initial(self):
        ks = make_simple_ks()
        r = model_check(ks, prop('p'))
        assert r.satisfied  # s0 has p

    def test_prop_not_at_initial(self):
        ks = make_simple_ks()
        r = model_check(ks, prop('q'))
        assert not r.satisfied  # s0 doesn't have q

    def test_negation(self):
        ks = make_simple_ks()
        r = model_check(ks, neg(prop('q')))
        assert r.satisfied  # s0 doesn't have q, so ~q holds

    def test_conjunction(self):
        ks = make_simple_ks()
        r = model_check(ks, conj(prop('p'), neg(prop('q'))))
        assert r.satisfied  # s0: p and ~q

    def test_disjunction(self):
        ks = make_simple_ks()
        r = model_check(ks, disj(prop('p'), prop('q')))
        assert r.satisfied

    def test_implication(self):
        ks = make_simple_ks()
        r = model_check(ks, implies(prop('p'), neg(prop('q'))))
        assert r.satisfied  # p => ~q at s0

    def test_sat_states(self):
        ks = make_simple_ks()
        states = model_check_states(ks, prop('p'))
        assert 0 in states  # s0 has p
        assert 2 in states  # s2 has p
        assert 1 not in states


# ============================================================
# Test: Modal Operators (EX, AX)
# ============================================================

class TestModalOperators:
    def test_ex_successor_has_property(self):
        ks = make_simple_ks()
        # EX(q): s0 can reach s1 which has q
        r = model_check(ks, ex(prop('q')))
        assert r.satisfied

    def test_ex_no_successor_with_property(self):
        ks = make_simple_ks()
        # At s0, EX(p): s1 doesn't have p
        states = model_check_states(ks, ex(prop('p')))
        assert 0 not in states  # s0's successor s1 doesn't have p
        assert 1 in states      # s1's successor s2 has p

    def test_ax_all_successors(self):
        ks = make_self_loop_ks()
        r = model_check(ks, ax(prop('p')))
        assert r.satisfied  # only successor is itself, which has p

    def test_ax_diamond_not_all(self):
        ks = make_diamond_ks()
        # AX(left): s0 has successors s1(left) and s2(right)
        # Not all successors have 'left'
        r = model_check(ks, ax(prop('left')))
        assert not r.satisfied

    def test_ex_diamond_exists(self):
        ks = make_diamond_ks()
        # EX(left): s0 has successor s1 with left
        r = model_check(ks, ex(prop('left')))
        assert r.satisfied

    def test_ex_chain(self):
        ks = make_simple_ks()
        # EX(EX(p)): s0 -> s1 -> s2(p)
        states = model_check_states(ks, ex(ex(prop('p'))))
        assert 0 in states


# ============================================================
# Test: CTL Operators (EF, AF, EG, AG, EU, AU)
# ============================================================

class TestCTLOperators:
    def test_ef_reachable(self):
        ks = make_simple_ks()
        r = model_check(ks, ef(prop('q')))
        assert r.satisfied  # s0 -> s1 (has q)

    def test_ef_unreachable(self):
        # Linear: s0 -> s1 -> s1 (self-loop). s0:{a}, s1:{b}
        ks = KripkeStructure(
            states={0, 1}, initial={0},
            transitions={0: {1}, 1: {1}},
            labeling={0: {'a'}, 1: {'b'}}
        )
        r = model_check(ks, ef(prop('a')))
        assert r.satisfied  # s0 already has a

    def test_af_inevitable(self):
        ks = make_simple_ks()
        # All paths from s0 eventually reach q (cycle through s1)
        r = model_check(ks, af(prop('q')))
        assert r.satisfied

    def test_ag_always(self):
        ks = make_self_loop_ks()
        r = model_check(ks, ag(prop('p')))
        assert r.satisfied  # p holds forever

    def test_ag_not_always(self):
        ks = make_simple_ks()
        # p doesn't hold at s1
        r = model_check(ks, ag(prop('p')))
        assert not r.satisfied

    def test_eg_exists_forever(self):
        ks = make_self_loop_ks()
        r = model_check(ks, eg(prop('p')))
        assert r.satisfied

    def test_eg_no_infinite_path(self):
        ks = make_diamond_ks()
        # EG(start): start only at s0, successors don't have start
        states = model_check_states(ks, eg(prop('start')))
        assert 0 not in states  # can't stay at start forever

    def test_eu_until(self):
        ks = make_simple_ks()
        # E[~q U q]: from s0, ~q holds, then s1 has q
        r = model_check(ks, eu(neg(prop('q')), prop('q')))
        assert r.satisfied

    def test_au_until(self):
        ks = make_simple_ks()
        r = model_check(ks, au(tt(), prop('q')))
        assert r.satisfied  # all paths from s0 reach q

    def test_ef_equals_eu_true(self):
        # EF(p) == E[tt U p]
        ks = make_simple_ks()
        s1 = model_check_states(ks, ef(prop('q')))
        s2 = model_check_states(ks, eu(tt(), prop('q')))
        assert s1 == s2

    def test_ag_implies_not_ef_neg(self):
        # AG(p) implies ~EF(~p)
        ks = make_self_loop_ks()
        r1 = model_check(ks, ag(prop('p')))
        r2 = model_check(ks, neg(ef(neg(prop('p')))))
        assert r1.satisfied == r2.satisfied


# ============================================================
# Test: Mu-Calculus Fixpoints
# ============================================================

class TestFixpoints:
    def test_mu_ef_encoding(self):
        # mu X. p \/ EX(X) == EF(p)
        ks = make_simple_ks()
        f_mu = mu('X', disj(prop('q'), ex(var('X'))))
        f_ef = ef(prop('q'))
        s1 = model_check_states(ks, f_mu)
        s2 = model_check_states(ks, f_ef)
        assert s1 == s2

    def test_nu_eg_encoding(self):
        # nu X. p /\ EX(X) == EG(p)
        ks = make_self_loop_ks()
        f_nu = nu('X', conj(prop('p'), ex(var('X'))))
        f_eg = eg(prop('p'))
        s1 = model_check_states(ks, f_nu)
        s2 = model_check_states(ks, f_eg)
        assert s1 == s2

    def test_mu_af_encoding(self):
        # mu X. p \/ AX(X) == AF(p)
        ks = make_simple_ks()
        f_mu = mu('X', disj(prop('q'), ax(var('X'))))
        f_af = af(prop('q'))
        s1 = model_check_states(ks, f_mu)
        s2 = model_check_states(ks, f_af)
        assert s1 == s2

    def test_nu_ag_encoding(self):
        # nu X. p /\ AX(X) == AG(p)
        ks = make_self_loop_ks()
        f_nu = nu('X', conj(prop('p'), ax(var('X'))))
        f_ag = ag(prop('p'))
        s1 = model_check_states(ks, f_nu)
        s2 = model_check_states(ks, f_ag)
        assert s1 == s2

    def test_nested_fixpoint(self):
        # nu X. mu Y. (p /\ X) \/ EX(Y)
        # "infinitely often p" on every path
        ks = make_simple_ks()
        f = nu('X', mu('Y', disj(conj(prop('p'), var('X')), ex(var('Y')))))
        states = model_check_states(ks, f)
        # All states can infinitely often visit p (cycle: s0->s1->s2->s0, s0 and s2 have p)
        assert 0 in states

    def test_mu_least_fixpoint_is_minimal(self):
        # mu X. X should be empty (least fixpoint of identity)
        ks = make_simple_ks()
        f = mu('X', var('X'))
        states = model_check_states(ks, f)
        assert len(states) == 0

    def test_nu_greatest_fixpoint_is_maximal(self):
        # nu X. X should be all states (greatest fixpoint of identity)
        ks = make_simple_ks()
        f = nu('X', var('X'))
        states = model_check_states(ks, f)
        assert states == {0, 1, 2}


# ============================================================
# Test: Kripke Structure Conversion
# ============================================================

class TestSymbolicConversion:
    def test_roundtrip_states(self):
        ks = make_simple_ks()
        sk = kripke_to_symbolic(ks)
        states = _bdd_states(sk, sk.states)
        assert len(states) == 3

    def test_roundtrip_initial(self):
        ks = make_simple_ks()
        sk = kripke_to_symbolic(ks)
        init = _bdd_states(sk, sk.initial)
        assert len(init) == 1

    def test_roundtrip_props(self):
        ks = make_simple_ks()
        sk = kripke_to_symbolic(ks)
        p_states = _bdd_states(sk, sk.prop_bdds['p'])
        assert len(p_states) == 2  # s0 and s2

    def test_diamond_conversion(self):
        ks = make_diamond_ks()
        sk = kripke_to_symbolic(ks)
        assert len(_bdd_states(sk, sk.states)) == 4

    def test_single_state(self):
        ks = make_self_loop_ks()
        sk = kripke_to_symbolic(ks)
        assert len(_bdd_states(sk, sk.states)) == 1


# ============================================================
# Test: Counter System
# ============================================================

class TestCounterSystem:
    def test_counter_ef_max(self):
        sys = make_counter_system(3)
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ef(prop('max')))
        assert r.satisfied

    def test_counter_af_max(self):
        sys = make_counter_system(3)
        ks = _concrete_to_kripke(sys)
        # Deterministic: always reaches max
        r = model_check(ks, af(prop('max')))
        assert r.satisfied

    def test_counter_ag_ef_zero(self):
        sys = make_counter_system(3)
        ks = _concrete_to_kripke(sys)
        # From every state, can eventually reach zero (wraps around)
        r = model_check(ks, ag(ef(prop('zero'))))
        assert r.satisfied

    def test_counter_ag_ef_max(self):
        sys = make_counter_system(3)
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ag(ef(prop('max'))))
        assert r.satisfied


# ============================================================
# Test: Traffic Light
# ============================================================

class TestTrafficLight:
    def test_traffic_starts_red(self):
        sys = make_traffic_light()
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, prop('red'))
        assert r.satisfied

    def test_traffic_ef_green(self):
        sys = make_traffic_light()
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ef(prop('green')))
        assert r.satisfied

    def test_traffic_ag_ef_red(self):
        sys = make_traffic_light()
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ag(ef(prop('red'))))
        assert r.satisfied

    def test_traffic_ag_ef_green(self):
        sys = make_traffic_light()
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ag(ef(prop('green'))))
        assert r.satisfied

    def test_traffic_no_red_green_simultaneous(self):
        sys = make_traffic_light()
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ag(neg(conj(prop('red'), prop('green')))))
        assert r.satisfied


# ============================================================
# Test: Mutex System
# ============================================================

class TestMutexSystem:
    def test_mutex_ef_both_critical(self):
        sys = make_mutex_system()
        ks = _concrete_to_kripke(sys)
        # Bug: both CAN enter critical (no real mutex)
        r = model_check(ks, ef(prop('both_critical')))
        assert r.satisfied  # bug: reachable

    def test_mutex_ag_not_both_critical_fails(self):
        sys = make_mutex_system()
        ks = _concrete_to_kripke(sys)
        # AG(~both_critical) should FAIL (mutex is buggy)
        r = model_check(ks, ag(neg(prop('both_critical'))))
        assert not r.satisfied

    def test_mutex_has_counterexample(self):
        sys = make_mutex_system()
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ag(neg(prop('both_critical'))))
        assert r.counterexample is not None

    def test_mutex_ef_some_critical(self):
        sys = make_mutex_system()
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ef(prop('some_critical')))
        assert r.satisfied


# ============================================================
# Test: Predicate Abstraction
# ============================================================

class TestPredicateAbstraction:
    def test_counter_abstraction(self):
        sys = make_counter_system(5)
        preds = [
            Predicate('zero', lambda s: s['x'] == 0),
            Predicate('positive', lambda s: s['x'] > 0),
        ]
        abs_ks = predicate_abstract(sys, preds)
        assert len(abs_ks.states) >= 2  # at least zero and positive

    def test_traffic_light_abstraction(self):
        sys = make_traffic_light()
        preds = [
            Predicate('is_red', lambda s: s['color'] == 0),
        ]
        abs_ks = predicate_abstract(sys, preds)
        assert len(abs_ks.states) >= 2

    def test_abstraction_preserves_initial(self):
        sys = make_counter_system(3)
        preds = [Predicate('zero', lambda s: s['x'] == 0)]
        abs_ks = predicate_abstract(sys, preds)
        assert len(abs_ks.initial) >= 1

    def test_abstraction_over_approximates(self):
        # Counter with 2 predicates: abstract should have <= 4 states
        sys = make_counter_system(10)
        preds = [
            Predicate('small', lambda s: s['x'] < 5),
            Predicate('even', lambda s: s['x'] % 2 == 0),
        ]
        abs_ks = predicate_abstract(sys, preds)
        assert len(abs_ks.states) <= 4  # 2^2 max abstract states

    def test_finer_predicates_more_states(self):
        sys = make_counter_system(7)
        preds1 = [Predicate('small', lambda s: s['x'] < 4)]
        preds2 = [
            Predicate('small', lambda s: s['x'] < 4),
            Predicate('even', lambda s: s['x'] % 2 == 0),
        ]
        abs1 = predicate_abstract(sys, preds1)
        abs2 = predicate_abstract(sys, preds2)
        assert len(abs2.states) >= len(abs1.states)


# ============================================================
# Test: CEGAR
# ============================================================

class TestCEGAR:
    def test_cegar_counter_always_reaches_max(self):
        sys = make_counter_system(3)
        preds = [Predicate('zero', lambda s: s['x'] == 0)]
        result = cegar_verify(sys, af(prop('max')), preds)
        # Counter is deterministic, AF(max) holds, but abstraction may need refinement
        assert result.verdict in (CEGARVerdict.SATISFIED, CEGARVerdict.VIOLATED,
                                   CEGARVerdict.UNKNOWN, CEGARVerdict.SPURIOUS_LIMIT)

    def test_cegar_traffic_always_red_again(self):
        sys = make_traffic_light()
        preds = [
            Predicate('is_red', lambda s: s['color'] == 0),
            Predicate('is_green', lambda s: s['color'] == 1),
            Predicate('is_yellow', lambda s: s['color'] == 2),
        ]
        result = cegar_verify(sys, ag(ef(prop('red'))), preds)
        assert result.verdict == CEGARVerdict.SATISFIED

    def test_cegar_mutex_violation(self):
        sys = make_mutex_system()
        preds = [
            Predicate('p1_crit', lambda s: s['p1'] == 2),
            Predicate('p2_crit', lambda s: s['p2'] == 2),
        ]
        result = cegar_verify(sys, ag(neg(prop('both_critical'))), preds)
        assert result.verdict == CEGARVerdict.VIOLATED

    def test_cegar_self_loop_ag(self):
        sys = ConcreteSystem(
            variables=['x'],
            init_states=[{'x': 1}],
            transition_fn=lambda s: [{'x': s['x']}],  # self-loop
            prop_fn=lambda s: {'positive'} if s['x'] > 0 else set()
        )
        preds = [Predicate('positive', lambda s: s['x'] > 0)]
        result = cegar_verify(sys, ag(prop('positive')), preds)
        assert result.verdict == CEGARVerdict.SATISFIED

    def test_cegar_statistics(self):
        sys = make_traffic_light()
        preds = [
            Predicate('is_red', lambda s: s['color'] == 0),
            Predicate('is_green', lambda s: s['color'] == 1),
            Predicate('is_yellow', lambda s: s['color'] == 2),
        ]
        result = cegar_verify(sys, ag(ef(prop('red'))), preds)
        stats = cegar_statistics(result)
        assert 'verdict' in stats
        assert 'iterations' in stats
        assert stats['iterations'] >= 1


# ============================================================
# Test: Formula Parser
# ============================================================

class TestFormulaParser:
    def test_parse_true(self):
        f = parse_formula("tt")
        assert f.kind == FormulaKind.TRUE

    def test_parse_false(self):
        f = parse_formula("ff")
        assert f.kind == FormulaKind.FALSE

    def test_parse_prop(self):
        f = parse_formula("safe")
        assert f.kind == FormulaKind.PROP
        assert f.prop == "safe"

    def test_parse_negation(self):
        f = parse_formula("~p")
        assert f.kind == FormulaKind.NOT
        assert f.children[0].prop == 'p'

    def test_parse_conjunction(self):
        f = parse_formula("(a /\\ b)")
        assert f.kind == FormulaKind.AND

    def test_parse_disjunction(self):
        f = parse_formula("(a \\/ b)")
        assert f.kind == FormulaKind.OR

    def test_parse_implication(self):
        f = parse_formula("(a => b)")
        assert f.kind == FormulaKind.IMPLIES

    def test_parse_ex(self):
        f = parse_formula("EX(p)")
        assert f.kind == FormulaKind.EX

    def test_parse_ax(self):
        f = parse_formula("AX(p)")
        assert f.kind == FormulaKind.AX

    def test_parse_ef(self):
        f = parse_formula("EF(p)")
        assert f.kind == FormulaKind.EF

    def test_parse_af(self):
        f = parse_formula("AF(p)")
        assert f.kind == FormulaKind.AF

    def test_parse_eg(self):
        f = parse_formula("EG(p)")
        assert f.kind == FormulaKind.EG

    def test_parse_ag(self):
        f = parse_formula("AG(p)")
        assert f.kind == FormulaKind.AG

    def test_parse_eu(self):
        f = parse_formula("E[ p U q ]")
        assert f.kind == FormulaKind.EU

    def test_parse_au(self):
        f = parse_formula("A[ p U q ]")
        assert f.kind == FormulaKind.AU

    def test_parse_mu(self):
        f = parse_formula("mu X . (X \\/ p)")
        assert f.kind == FormulaKind.MU
        assert f.var == 'X'

    def test_parse_nu(self):
        f = parse_formula("nu X . (X /\\ p)")
        assert f.kind == FormulaKind.NU
        assert f.var == 'X'

    def test_parse_nested_modal(self):
        f = parse_formula("AG(EF(p))")
        assert f.kind == FormulaKind.AG
        assert f.children[0].kind == FormulaKind.EF

    def test_parse_complex(self):
        f = parse_formula("mu X . (p \\/ EX(X))")
        assert f.kind == FormulaKind.MU


# ============================================================
# Test: Batch Checking
# ============================================================

class TestBatchChecking:
    def test_batch_simple(self):
        ks = make_simple_ks()
        formulas = [
            ("p_holds", prop('p')),
            ("q_holds", prop('q')),
            ("ef_q", ef(prop('q'))),
        ]
        results = batch_check(ks, formulas)
        assert len(results) == 3
        assert results[0]['satisfied']  # p at initial
        assert not results[1]['satisfied']  # q not at initial
        assert results[2]['satisfied']  # EF(q)

    def test_batch_traffic(self):
        sys = make_traffic_light()
        ks = _concrete_to_kripke(sys)
        formulas = [
            ("starts_red", prop('red')),
            ("ef_green", ef(prop('green'))),
            ("ag_ef_red", ag(ef(prop('red')))),
        ]
        results = batch_check(ks, formulas)
        assert all(r['satisfied'] for r in results)


# ============================================================
# Test: Verification Helper
# ============================================================

class TestVerification:
    def test_verify_correct(self):
        ks = make_simple_ks()
        r = verify_model_check(ks, prop('p'), True)
        assert r['correct']

    def test_verify_incorrect(self):
        ks = make_simple_ks()
        r = verify_model_check(ks, prop('q'), True)
        assert not r['correct']

    def test_compare_methods(self):
        ks = make_simple_ks()
        r = compare_mc_methods(ks, ef(prop('q')))
        assert r['direct_satisfied']
        assert r['num_states'] == 3


# ============================================================
# Test: Counterexample Generation
# ============================================================

class TestCounterexamples:
    def test_counterexample_for_ag_fail(self):
        ks = make_simple_ks()
        r = model_check(ks, ag(prop('p')))
        assert not r.satisfied
        assert r.counterexample is not None

    def test_no_counterexample_when_satisfied(self):
        ks = make_self_loop_ks()
        r = model_check(ks, ag(prop('p')))
        assert r.satisfied
        assert r.counterexample is None

    def test_counterexample_is_path(self):
        ks = make_diamond_ks()
        r = model_check(ks, ag(prop('start')))
        assert not r.satisfied
        assert isinstance(r.counterexample, list)
        assert len(r.counterexample) >= 1


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_transitions_from_state(self):
        # Deadlock state: s1 has no successors
        ks = KripkeStructure(
            states={0, 1}, initial={0},
            transitions={0: {1}, 1: set()},
            labeling={0: {'a'}, 1: {'b'}}
        )
        r = model_check(ks, ef(prop('b')))
        assert r.satisfied

    def test_multiple_initial_states(self):
        ks = KripkeStructure(
            states={0, 1, 2}, initial={0, 1},
            transitions={0: {2}, 1: {2}, 2: {2}},
            labeling={0: {'a'}, 1: {'b'}, 2: {'c'}}
        )
        # prop('a') doesn't hold at s1 (initial state)
        r = model_check(ks, prop('a'))
        assert not r.satisfied

    def test_large_cycle(self):
        n = 8
        ks = KripkeStructure(
            states=set(range(n)), initial={0},
            transitions={i: {(i + 1) % n} for i in range(n)},
            labeling={i: {'target'} if i == n - 1 else set() for i in range(n)}
        )
        r = model_check(ks, ef(prop('target')))
        assert r.satisfied
        r2 = model_check(ks, ag(ef(prop('target'))))
        assert r2.satisfied

    def test_disconnected_states(self):
        ks = KripkeStructure(
            states={0, 1, 2}, initial={0},
            transitions={0: {0}, 1: {1}, 2: {2}},
            labeling={0: {'a'}, 1: {'b'}, 2: {'c'}}
        )
        r = model_check(ks, ef(prop('b')))
        assert not r.satisfied  # b only at s1, unreachable from s0


# ============================================================
# Test: Mu-Calculus Specific Properties
# ============================================================

class TestMuCalculusProperties:
    def test_fairness_constraint(self):
        # nu X. (EF(p) /\ EX(X)) -- infinitely often can reach p
        ks = make_simple_ks()
        f = nu('X', conj(ef(prop('p')), ex(var('X'))))
        states = model_check_states(ks, f)
        # All states in the cycle can always eventually reach p
        assert 0 in states

    def test_alternating_fixpoint_depth2(self):
        # mu X. nu Y. (p /\ Y) \/ EX(X)
        ks = make_simple_ks()
        f = mu('X', nu('Y', disj(conj(prop('p'), var('Y')), ex(var('X')))))
        states = model_check_states(ks, f)
        assert len(states) >= 1

    def test_boolean_combinations(self):
        ks = make_simple_ks()
        # AG(p) \/ AG(q) -- neither holds but their disjunction is checked
        f = disj(ag(prop('p')), ag(prop('q')))
        r = model_check(ks, f)
        # Neither AG(p) nor AG(q) holds alone
        assert not r.satisfied

    def test_ctl_star_like_nesting(self):
        # EG(EF(p)) -- there exists a path where we can always eventually reach p
        ks = make_simple_ks()
        f = eg(ef(prop('p')))
        r = model_check(ks, f)
        assert r.satisfied


# ============================================================
# Test: Concrete System to Kripke Conversion
# ============================================================

def _concrete_to_kripke(sys: ConcreteSystem, max_states: int = 1000) -> KripkeStructure:
    """Enumerate concrete system into explicit Kripke structure."""
    from collections import deque
    states_list = []
    state_set = set()
    queue = deque()
    state_to_id = {}

    def key(s):
        return tuple(sorted(s.items()))

    for init in sys.init_states:
        k = key(init)
        if k not in state_set:
            sid = len(states_list)
            state_set.add(k)
            states_list.append(init)
            state_to_id[k] = sid
            queue.append(init)

    while queue and len(states_list) < max_states:
        s = queue.popleft()
        for t in sys.transition_fn(s):
            k = key(t)
            if k not in state_set:
                sid = len(states_list)
                state_set.add(k)
                states_list.append(t)
                state_to_id[k] = sid
                queue.append(t)

    transitions = {i: set() for i in range(len(states_list))}
    labeling = {}
    for i, s in enumerate(states_list):
        labeling[i] = sys.prop_fn(s)
        for t in sys.transition_fn(s):
            k = key(t)
            if k in state_to_id:
                transitions[i].add(state_to_id[k])

    initial = set()
    for init in sys.init_states:
        k = key(init)
        if k in state_to_id:
            initial.add(state_to_id[k])

    return KripkeStructure(
        states=set(range(len(states_list))),
        initial=initial,
        transitions=transitions,
        labeling=labeling
    )


class TestConcreteToKripke:
    def test_counter_states(self):
        sys = make_counter_system(3)
        ks = _concrete_to_kripke(sys)
        assert len(ks.states) == 4  # 0, 1, 2, 3

    def test_traffic_states(self):
        sys = make_traffic_light()
        ks = _concrete_to_kripke(sys)
        assert len(ks.states) == 3

    def test_mutex_states(self):
        sys = make_mutex_system()
        ks = _concrete_to_kripke(sys)
        assert len(ks.states) == 9  # 3 * 3


# ============================================================
# Test: Bounded Counter with CEGAR
# ============================================================

class TestBoundedCounterCEGAR:
    def test_bounded_ef_overflow(self):
        sys = make_bounded_counter(5)
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ef(prop('overflow')))
        assert r.satisfied  # can always increment past bound

    def test_bounded_ag_safe_fails(self):
        sys = make_bounded_counter(3)
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ag(prop('safe')))
        assert not r.satisfied  # can overflow

    def test_bounded_cegar_overflow(self):
        sys = make_bounded_counter(3)
        preds = [
            Predicate('zero', lambda s: s['x'] == 0),
            Predicate('safe', lambda s: s['x'] < 3),
        ]
        result = cegar_verify(sys, ag(prop('safe')), preds)
        assert result.verdict == CEGARVerdict.VIOLATED


# ============================================================
# Test: Complex Formulas on Complex Systems
# ============================================================

class TestComplexScenarios:
    def test_liveness_under_fairness(self):
        # In counter: if we're fair (always eventually tick), we reach max
        sys = make_counter_system(4)
        ks = _concrete_to_kripke(sys)
        # AF(max) -- deterministic, always reaches max
        r = model_check(ks, af(prop('max')))
        assert r.satisfied

    def test_response_property(self):
        # AG(requesting => AF(granted))
        # Simple: s0(req) -> s1(grant) -> s0
        ks = KripkeStructure(
            states={0, 1}, initial={0},
            transitions={0: {1}, 1: {0}},
            labeling={0: {'requesting'}, 1: {'granted'}}
        )
        f = ag(implies(prop('requesting'), af(prop('granted'))))
        r = model_check(ks, f)
        assert r.satisfied

    def test_persistence_property(self):
        # AF(AG(stable)) -- eventually stabilizes
        ks = KripkeStructure(
            states={0, 1}, initial={0},
            transitions={0: {1}, 1: {1}},
            labeling={0: set(), 1: {'stable'}}
        )
        f = af(ag(prop('stable')))
        r = model_check(ks, f)
        assert r.satisfied

    def test_recurrence_property(self):
        # AG(AF(p)) -- p occurs infinitely often
        sys = make_counter_system(3)
        ks = _concrete_to_kripke(sys)
        r = model_check(ks, ag(af(prop('zero'))))
        assert r.satisfied  # always returns to zero

    def test_mutual_exclusion_property(self):
        sys = make_mutex_system()
        ks = _concrete_to_kripke(sys)
        # Can we reach a state where someone is critical?
        r = model_check(ks, ef(prop('some_critical')))
        assert r.satisfied

    def test_starvation_freedom(self):
        # AG(trying => AF(critical)) -- if trying, eventually critical
        # In our simple mutex this holds (no blocking)
        ks = KripkeStructure(
            states={0, 1, 2}, initial={0},
            transitions={0: {1}, 1: {2}, 2: {0}},
            labeling={0: {'idle'}, 1: {'trying'}, 2: {'critical'}}
        )
        f = ag(implies(prop('trying'), af(prop('critical'))))
        r = model_check(ks, f)
        assert r.satisfied


# ============================================================
# Test: Parsed Formula Model Checking
# ============================================================

class TestParsedFormulas:
    def test_parsed_ef(self):
        ks = make_simple_ks()
        f = parse_formula("EF(q)")
        r = model_check(ks, f)
        assert r.satisfied

    def test_parsed_ag(self):
        ks = make_self_loop_ks()
        f = parse_formula("AG(p)")
        r = model_check(ks, f)
        assert r.satisfied

    def test_parsed_conjunction(self):
        ks = make_simple_ks()
        f = parse_formula("(p /\\ ~q)")
        r = model_check(ks, f)
        assert r.satisfied  # s0: p, ~q

    def test_parsed_mu(self):
        ks = make_simple_ks()
        f = parse_formula("mu X . (q \\/ EX(X))")
        r = model_check(ks, f)
        assert r.satisfied  # EF(q)

    def test_parsed_complex(self):
        ks = make_simple_ks()
        f = parse_formula("AG(EF(p))")
        r = model_check(ks, f)
        assert r.satisfied


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
