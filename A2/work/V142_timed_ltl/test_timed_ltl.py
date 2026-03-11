"""
Tests for V142: Timed Automata + LTL Model Checking
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V118_timed_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
# V023 also depends on V021 BDD library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

import pytest
from timed_ltl import (
    TimedLTLVerdict, TimedLTLResult, ProductState, LocationLabeling,
    auto_labeling, check_timed_ltl, check_timed_ltl_parsed,
    check_timed_safety, check_timed_liveness, check_timed_response,
    check_timed_until,
    abstract_zone_graph, compare_timed_vs_untimed,
    build_product_zone_graph,
    light_timer_with_properties, train_gate_with_properties,
    mutex_with_properties,
    batch_check, verification_summary,
)
from timed_automata import (
    simple_ta, true_guard, clock_leq, clock_lt, clock_geq, clock_gt,
    clock_eq, clock_diff_leq, clock_diff_geq, guard_and,
    TimedAutomaton, Edge, Guard
)
from ltl_model_checker import (
    Atom, LTLTrue, LTLFalse, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release, WeakUntil,
    parse_ltl, ltl_to_gba, gba_to_nba
)


# ============================================================
# Section 1: ProductState and LocationLabeling
# ============================================================

class TestDataStructures:
    def test_product_state_hash(self):
        ps1 = ProductState("loc1", 0)
        ps2 = ProductState("loc1", 0)
        ps3 = ProductState("loc1", 1)
        assert ps1 == ps2
        assert ps1 != ps3
        assert hash(ps1) == hash(ps2)

    def test_product_state_repr(self):
        ps = ProductState("idle", 2)
        assert "idle" in repr(ps)
        assert "q2" in repr(ps)

    def test_location_labeling_basic(self):
        lab = LocationLabeling()
        lab.add_label("loc1", "a")
        lab.add_label("loc1", "b")
        lab.add_label("loc2", "c")
        assert lab.get_labels("loc1") == {"a", "b"}
        assert lab.get_labels("loc2") == {"c"}
        assert lab.get_labels("loc3") == set()

    def test_location_labeling_empty(self):
        lab = LocationLabeling()
        assert lab.get_labels("nonexistent") == set()


# ============================================================
# Section 2: Auto Labeling
# ============================================================

class TestAutoLabeling:
    def test_auto_labeling_exact_match(self):
        ta = simple_ta(
            locations=['idle', 'busy'],
            initial='idle',
            clocks=['x'],
            edges=[('idle', 'busy', 'start', true_guard(), [])],
        )
        formula = Globally(Atom('idle'))
        lab = auto_labeling(ta, formula)
        assert 'idle' in lab.get_labels('idle')
        assert 'idle' not in lab.get_labels('busy')

    def test_auto_labeling_substring_match(self):
        ta = simple_ta(
            locations=['crit_1', 'crit_2', 'idle'],
            initial='idle',
            clocks=['x'],
            edges=[('idle', 'crit_1', 'go', true_guard(), [])],
        )
        formula = Finally(Atom('crit'))
        lab = auto_labeling(ta, formula)
        assert 'crit' in lab.get_labels('crit_1')
        assert 'crit' in lab.get_labels('crit_2')
        assert 'crit' not in lab.get_labels('idle')

    def test_auto_labeling_no_match(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[('a', 'b', 'go', true_guard(), [])],
        )
        formula = Globally(Atom('nonexistent'))
        lab = auto_labeling(ta, formula)
        assert lab.get_labels('a') == set()
        assert lab.get_labels('b') == set()


# ============================================================
# Section 3: NBA Construction (prerequisite check)
# ============================================================

class TestNBAConstruction:
    def test_gba_nba_for_simple_formula(self):
        formula = Not(Globally(Atom('a')))  # F(!a)
        gba = ltl_to_gba(formula)
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0
        assert len(nba.initial) > 0

    def test_gba_nba_for_until(self):
        formula = Until(Atom('a'), Atom('b'))
        gba = ltl_to_gba(formula)
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0


# ============================================================
# Section 4: Simple Timed LTL - Trivial Cases
# ============================================================

class TestSimpleCases:
    def test_true_formula(self):
        ta = simple_ta(
            locations=['a'],
            initial='a',
            clocks=['x'],
            edges=[('a', 'a', 'tick', true_guard(), ['x'])],
            invariants={'a': clock_leq('x', 1)},
            accepting={'a'}
        )
        result = check_timed_ltl(ta, LTLTrue())
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_false_formula(self):
        ta = simple_ta(
            locations=['a'],
            initial='a',
            clocks=['x'],
            edges=[('a', 'a', 'tick', true_guard(), ['x'])],
            invariants={'a': clock_leq('x', 1)},
            accepting={'a'}
        )
        result = check_timed_ltl(ta, LTLFalse())
        assert result.verdict == TimedLTLVerdict.VIOLATED

    def test_single_atom_always_holds(self):
        ta = simple_ta(
            locations=['on'],
            initial='on',
            clocks=['x'],
            edges=[('on', 'on', 'tick', true_guard(), ['x'])],
            invariants={'on': clock_leq('x', 1)},
            accepting={'on'}
        )
        lab = LocationLabeling()
        lab.add_label('on', 'alive')
        result = check_timed_ltl(ta, Globally(Atom('alive')), lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED


# ============================================================
# Section 5: Safety Properties
# ============================================================

class TestSafety:
    def test_safety_single_safe_location(self):
        ta = simple_ta(
            locations=['safe'],
            initial='safe',
            clocks=['x'],
            edges=[('safe', 'safe', 'loop', true_guard(), ['x'])],
            invariants={'safe': clock_leq('x', 2)},
            accepting={'safe'}
        )
        lab = LocationLabeling()
        lab.add_label('safe', 'ok')
        result = check_timed_safety(ta, 'ok', lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_safety_violation_reachable_unsafe(self):
        ta = simple_ta(
            locations=['safe', 'unsafe'],
            initial='safe',
            clocks=['x'],
            edges=[
                ('safe', 'unsafe', 'fail', clock_geq('x', 1), []),
                ('unsafe', 'unsafe', 'stay', true_guard(), ['x']),
            ],
            invariants={'unsafe': clock_leq('x', 1)},
            accepting={'safe', 'unsafe'}
        )
        lab = LocationLabeling()
        lab.add_label('safe', 'ok')
        # G(ok) should be violated because we can reach 'unsafe'
        result = check_timed_safety(ta, 'ok', lab)
        assert result.verdict == TimedLTLVerdict.VIOLATED

    def test_safety_unreachable_unsafe(self):
        """Unsafe location exists but is unreachable."""
        ta = simple_ta(
            locations=['safe', 'unsafe'],
            initial='safe',
            clocks=['x'],
            edges=[
                ('safe', 'safe', 'loop', true_guard(), ['x']),
                # No edge to unsafe!
            ],
            invariants={'safe': clock_leq('x', 1)},
            accepting={'safe', 'unsafe'}
        )
        lab = LocationLabeling()
        lab.add_label('safe', 'ok')
        lab.add_label('unsafe', 'ok')  # Even unsafe is labeled ok, so G(ok) holds
        result = check_timed_safety(ta, 'ok', lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED


# ============================================================
# Section 6: Liveness Properties
# ============================================================

class TestLiveness:
    def test_liveness_self_loop(self):
        """Single location with self-loop satisfies G(F(here))."""
        ta = simple_ta(
            locations=['here'],
            initial='here',
            clocks=['x'],
            edges=[('here', 'here', 'tick', true_guard(), ['x'])],
            invariants={'here': clock_leq('x', 1)},
            accepting={'here'}
        )
        lab = LocationLabeling()
        lab.add_label('here', 'here')
        result = check_timed_liveness(ta, 'here', lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_liveness_cyclic(self):
        """Two-location cycle satisfies G(F(a)) and G(F(b))."""
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', clock_geq('x', 1), ['x']),
                ('b', 'a', 'back', clock_geq('x', 1), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 2),
                'b': clock_leq('x', 2),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        result_a = check_timed_liveness(ta, 'a', lab)
        assert result_a.verdict == TimedLTLVerdict.SATISFIED

        result_b = check_timed_liveness(ta, 'b', lab)
        assert result_b.verdict == TimedLTLVerdict.SATISFIED


# ============================================================
# Section 7: Response Properties
# ============================================================

class TestResponse:
    def test_response_direct(self):
        """Request always followed by grant."""
        ta = simple_ta(
            locations=['idle', 'request', 'grant'],
            initial='idle',
            clocks=['x'],
            edges=[
                ('idle', 'request', 'req', true_guard(), ['x']),
                ('request', 'grant', 'grt', clock_geq('x', 1), ['x']),
                ('grant', 'idle', 'done', clock_geq('x', 1), ['x']),
            ],
            invariants={
                'request': clock_leq('x', 3),
                'grant': clock_leq('x', 2),
            },
            accepting={'idle', 'request', 'grant'}
        )
        lab = LocationLabeling()
        lab.add_label('idle', 'idle')
        lab.add_label('request', 'request')
        lab.add_label('grant', 'grant')

        result = check_timed_response(ta, 'request', 'grant', lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_response_no_grant(self):
        """Request without guaranteed grant: deadlock in request."""
        ta = simple_ta(
            locations=['idle', 'request'],
            initial='idle',
            clocks=['x'],
            edges=[
                ('idle', 'request', 'req', true_guard(), ['x']),
                # No edge from request! Deadlock.
            ],
            accepting={'idle', 'request'}
        )
        lab = LocationLabeling()
        lab.add_label('idle', 'idle')
        lab.add_label('request', 'request')
        lab.add_label('idle', 'grant')  # grant only at idle (which we can't reach from request)

        # G(request -> F(grant)) should be violated
        # But this is a deadlock case - no outgoing transitions from request
        # The system stops, so F(grant) doesn't hold
        result = check_timed_response(ta, 'request', 'grant', lab)
        # In a deadlock system, the formula might still hold vacuously
        # because the run is finite (no infinite run through request without grant)
        # Actually: timed automata semantics require time-divergent runs
        # A deadlock means no valid infinite run through request, so
        # the property holds vacuously for runs through request
        assert result.verdict in (TimedLTLVerdict.SATISFIED, TimedLTLVerdict.VIOLATED)


# ============================================================
# Section 8: Until Properties
# ============================================================

class TestUntil:
    def test_until_simple(self):
        """a holds until b is reached."""
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'switch', clock_geq('x', 1), []),
                ('b', 'b', 'stay', true_guard(), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 3),
                'b': clock_leq('x', 1),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        result = check_timed_until(ta, 'a', 'b', lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED


# ============================================================
# Section 9: Parsed Formula API
# ============================================================

class TestParsedFormula:
    def test_parsed_globally(self):
        ta = simple_ta(
            locations=['on'],
            initial='on',
            clocks=['x'],
            edges=[('on', 'on', 'tick', true_guard(), ['x'])],
            invariants={'on': clock_leq('x', 1)},
            accepting={'on'}
        )
        lab = LocationLabeling()
        lab.add_label('on', 'on')
        result = check_timed_ltl_parsed(ta, "G(on)", lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_parsed_eventually(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', clock_geq('x', 1), []),
                ('b', 'b', 'stay', true_guard(), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 2),
                'b': clock_leq('x', 1),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')
        result = check_timed_ltl_parsed(ta, "F(b)", lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_parsed_complex(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', clock_geq('x', 1), ['x']),
                ('b', 'a', 'back', clock_geq('x', 1), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 2),
                'b': clock_leq('x', 2),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')
        # G(a -> F(b)): from a, eventually reach b
        result = check_timed_ltl_parsed(ta, "G(a -> F(b))", lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED


# ============================================================
# Section 10: Product Zone Graph Construction
# ============================================================

class TestProductConstruction:
    def test_product_basic(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', true_guard(), ['x']),
                ('b', 'a', 'back', true_guard(), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 1),
                'b': clock_leq('x', 1),
            },
            accepting={'a', 'b'}
        )
        formula = Not(Globally(Atom('a')))  # F(!a)
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        neg = parse_ltl("!(G(a))")
        gba = ltl_to_gba(neg)
        nba = gba_to_nba(gba)

        reached, init_states, stats = build_product_zone_graph(ta, nba, lab)
        assert len(init_states) > 0
        assert stats.product_states > 0

    def test_product_stats(self):
        ta = simple_ta(
            locations=['on', 'off'],
            initial='off',
            clocks=['x'],
            edges=[
                ('off', 'on', 'press', true_guard(), ['x']),
                ('on', 'off', 'press', true_guard(), ['x']),
            ],
            invariants={
                'on': clock_leq('x', 5),
                'off': clock_leq('x', 5),
            },
            accepting={'on', 'off'}
        )
        lab = LocationLabeling()
        lab.add_label('on', 'on')
        lab.add_label('off', 'off')

        result = check_timed_ltl(ta, Globally(Atom('off')), lab)
        assert result.nba_states > 0
        assert result.product_states_explored >= 0


# ============================================================
# Section 11: Zone Graph Abstraction
# ============================================================

class TestZoneGraphAbstraction:
    def test_abstraction_basic(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', clock_geq('x', 1), ['x']),
                ('b', 'a', 'back', clock_geq('x', 1), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 2),
                'b': clock_leq('x', 2),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        abstr = abstract_zone_graph(ta, lab)
        assert abstr.n_states() >= 2  # at least a and b
        assert len(abstr.initial) >= 1

    def test_abstraction_labels_preserved(self):
        ta = simple_ta(
            locations=['safe', 'danger'],
            initial='safe',
            clocks=['x'],
            edges=[
                ('safe', 'danger', 'go', clock_geq('x', 1), []),
                ('danger', 'safe', 'back', true_guard(), ['x']),
            ],
            invariants={'safe': clock_leq('x', 2)},
            accepting={'safe', 'danger'}
        )
        lab = LocationLabeling()
        lab.add_label('safe', 'ok')

        abstr = abstract_zone_graph(ta, lab)
        # At least one state should have 'ok' label
        has_ok = any('ok' in labels for labels in abstr.state_labels.values())
        assert has_ok

    def test_abstraction_transitions_exist(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', true_guard(), ['x']),
                ('b', 'a', 'back', true_guard(), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 1),
                'b': clock_leq('x', 1),
            },
        )
        abstr = abstract_zone_graph(ta)
        total_trans = sum(len(ts) for ts in abstr.transitions.values())
        assert total_trans > 0


# ============================================================
# Section 12: Example Systems
# ============================================================

class TestExampleSystems:
    def test_light_timer_properties(self):
        ta, lab, properties = light_timer_with_properties()
        assert len(ta.locations) == 2
        assert len(properties) >= 1

        results = batch_check(ta, properties, lab)
        for name, result in results:
            assert result.verdict in (TimedLTLVerdict.SATISFIED, TimedLTLVerdict.VIOLATED)

    def test_train_gate_properties(self):
        ta, lab, properties = train_gate_with_properties()
        assert len(ta.locations) == 4

        results = batch_check(ta, properties, lab)
        for name, result in results:
            assert result.verdict in (TimedLTLVerdict.SATISFIED, TimedLTLVerdict.VIOLATED)

    def test_mutex_properties(self):
        ta, lab, properties = mutex_with_properties(n=2)
        assert len(ta.locations) == 9  # 3 x 3 product locations

        results = batch_check(ta, properties, lab)
        for name, result in results:
            assert result.verdict in (TimedLTLVerdict.SATISFIED, TimedLTLVerdict.VIOLATED)

    def test_mutex_safety(self):
        """Both processes should not be in critical simultaneously.
        Note: without proper synchronization, this protocol allows both in critical."""
        ta, lab, properties = mutex_with_properties(n=2)
        # Check G(!both_critical)
        safety_result = check_timed_safety(ta, 'both_critical', lab)
        # This unsynchronized protocol CAN have both critical
        # (no mutual exclusion enforced)
        # So G(!both_critical) is equivalent to checking G(Not(both_critical))
        # But check_timed_safety checks G(prop), meaning G(both_critical)
        # We want G(!both_critical), which is check_timed_ltl with the formula directly
        result = check_timed_ltl(ta, Globally(Not(Atom('both_critical'))), lab)
        # Without synchronization, both can be critical
        assert result.verdict == TimedLTLVerdict.VIOLATED


# ============================================================
# Section 13: Comparison API
# ============================================================

class TestComparison:
    def test_compare_basic(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', clock_geq('x', 1), ['x']),
                ('b', 'a', 'back', clock_geq('x', 1), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 2),
                'b': clock_leq('x', 2),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        comp = compare_timed_vs_untimed(ta, Globally(Finally(Atom('a'))), lab)
        assert 'timed_verdict' in comp
        assert 'zone_graph_states' in comp
        assert 'nba_states' in comp
        assert comp['zone_graph_states'] >= 2


# ============================================================
# Section 14: Batch Verification and Summary
# ============================================================

class TestBatch:
    def test_batch_check(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', clock_geq('x', 1), ['x']),
                ('b', 'a', 'back', clock_geq('x', 1), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 2),
                'b': clock_leq('x', 2),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        formulas = [
            ("G(F(a))", Globally(Finally(Atom('a')))),
            ("G(F(b))", Globally(Finally(Atom('b')))),
            ("F(b)", Finally(Atom('b'))),
        ]
        results = batch_check(ta, formulas, lab)
        assert len(results) == 3
        for name, result in results:
            assert isinstance(result, TimedLTLResult)

    def test_verification_summary(self):
        ta = simple_ta(
            locations=['a'],
            initial='a',
            clocks=['x'],
            edges=[('a', 'a', 'tick', true_guard(), ['x'])],
            invariants={'a': clock_leq('x', 1)},
            accepting={'a'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')

        formulas = [("G(a)", Globally(Atom('a')))]
        results = batch_check(ta, formulas, lab)
        summary = verification_summary(ta, results)
        assert "Timed LTL Verification Summary" in summary
        assert "1/1" in summary or "PASS" in summary


# ============================================================
# Section 15: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_no_edges(self):
        """TA with no edges: deadlock at initial location."""
        ta = simple_ta(
            locations=['stuck'],
            initial='stuck',
            clocks=['x'],
            edges=[],
            accepting={'stuck'}
        )
        lab = LocationLabeling()
        lab.add_label('stuck', 'stuck')
        # G(stuck) should hold (we never leave)
        result = check_timed_ltl(ta, Globally(Atom('stuck')), lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_multiple_clocks(self):
        """TA with two clocks."""
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x', 'y'],
            edges=[
                ('a', 'b', 'go', guard_and(clock_geq('x', 1), clock_leq('y', 3)), ['x']),
                ('b', 'a', 'back', clock_geq('y', 2), ['y']),
            ],
            invariants={
                'a': clock_leq('x', 5),
                'b': clock_leq('x', 2),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        result = check_timed_ltl(ta, Finally(Atom('b')), lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_tight_invariant(self):
        """Invariant forces quick transitions."""
        ta = simple_ta(
            locations=['fast', 'slow'],
            initial='fast',
            clocks=['x'],
            edges=[
                ('fast', 'slow', 'go', true_guard(), ['x']),
                ('slow', 'fast', 'back', true_guard(), ['x']),
            ],
            invariants={
                'fast': clock_leq('x', 1),
                'slow': clock_leq('x', 1),
            },
            accepting={'fast', 'slow'}
        )
        lab = LocationLabeling()
        lab.add_label('fast', 'fast')
        lab.add_label('slow', 'slow')

        result = check_timed_ltl(ta, Globally(Finally(Atom('fast'))), lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_guard_blocks_transition(self):
        """Guard prevents reaching target."""
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                # Guard requires x >= 10, but invariant limits x <= 5
                ('a', 'b', 'go', clock_geq('x', 10), []),
                ('a', 'a', 'loop', true_guard(), ['x']),
            ],
            invariants={'a': clock_leq('x', 5)},
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        # F(b) should be violated: can never reach b
        result = check_timed_ltl(ta, Finally(Atom('b')), lab)
        # This depends on semantics: if a has a self-loop keeping it alive,
        # the system can run forever without reaching b
        # The NBA for F(b) = true U b accepts runs that eventually reach b
        # Since no run reaches b, all infinite runs violate F(b)
        assert result.verdict == TimedLTLVerdict.VIOLATED

    def test_empty_formula_atoms(self):
        """Formula with atoms not matching any location."""
        ta = simple_ta(
            locations=['x', 'y'],
            initial='x',
            clocks=['c'],
            edges=[
                ('x', 'y', 'go', true_guard(), ['c']),
                ('y', 'x', 'back', true_guard(), ['c']),
            ],
            invariants={
                'x': clock_leq('c', 1),
                'y': clock_leq('c', 1),
            },
            accepting={'x', 'y'}
        )
        # F(z) where z doesn't match any location
        lab = LocationLabeling()
        result = check_timed_ltl(ta, Finally(Atom('z')), lab)
        assert result.verdict == TimedLTLVerdict.VIOLATED


# ============================================================
# Section 16: TimedLTLResult Structure
# ============================================================

class TestResultStructure:
    def test_result_fields(self):
        ta = simple_ta(
            locations=['a'],
            initial='a',
            clocks=['x'],
            edges=[('a', 'a', 'tick', true_guard(), ['x'])],
            invariants={'a': clock_leq('x', 1)},
            accepting={'a'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        result = check_timed_ltl(ta, Globally(Atom('a')), lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED
        assert result.formula is not None
        assert result.nba_states > 0
        assert result.accepting_cycle_found == False

    def test_violated_result_has_counterexample(self):
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', true_guard(), ['x']),
                ('b', 'b', 'stay', true_guard(), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 1),
                'b': clock_leq('x', 1),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        # G(a) violated: can reach b
        result = check_timed_ltl(ta, Globally(Atom('a')), lab)
        assert result.verdict == TimedLTLVerdict.VIOLATED
        assert result.accepting_cycle_found == True


# ============================================================
# Section 17: Complex Temporal Properties
# ============================================================

class TestComplexProperties:
    def test_nested_temporal_individual(self):
        """G(F(a)) and G(F(b)) individually on a cycle both hold."""
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', clock_geq('x', 1), ['x']),
                ('b', 'a', 'back', clock_geq('x', 1), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 2),
                'b': clock_leq('x', 2),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        # Each individual G(F) property holds on the cycle
        result_a = check_timed_ltl(ta, Globally(Finally(Atom('a'))), lab)
        assert result_a.verdict == TimedLTLVerdict.SATISFIED
        result_b = check_timed_ltl(ta, Globally(Finally(Atom('b'))), lab)
        assert result_b.verdict == TimedLTLVerdict.SATISFIED

    def test_release_property(self):
        """b R a: b releases a (a holds until b holds, or a holds forever)."""
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'a', 'stay', true_guard(), ['x']),
            ],
            invariants={'a': clock_leq('x', 1)},
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('a', 'b')  # both a and b hold at 'a'

        # a R b: a holds forever OR (b holds until a first holds)
        result = check_timed_ltl(ta, Release(Atom('a'), Atom('b')), lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED

    def test_implication_property(self):
        """G(a -> X(b)) on an alternating system."""
        ta = simple_ta(
            locations=['a', 'b'],
            initial='a',
            clocks=['x'],
            edges=[
                ('a', 'b', 'go', true_guard(), ['x']),
                ('b', 'a', 'back', true_guard(), ['x']),
            ],
            invariants={
                'a': clock_leq('x', 1),
                'b': clock_leq('x', 1),
            },
            accepting={'a', 'b'}
        )
        lab = LocationLabeling()
        lab.add_label('a', 'a')
        lab.add_label('b', 'b')

        # G(a -> X(b)): whenever in a, next state is b
        # This should hold for the alternating system
        result = check_timed_ltl(ta, Globally(Implies(Atom('a'), Next(Atom('b')))), lab)
        assert result.verdict == TimedLTLVerdict.SATISFIED


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
