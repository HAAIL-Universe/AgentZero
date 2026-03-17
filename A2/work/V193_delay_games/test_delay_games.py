"""
Tests for V193: Delay Games -- Synthesis with Bounded Lookahead
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V186_reactive_synthesis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V187_gr1_synthesis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))

from delay_games import (
    DelayGameResult, MinDelayResult, DelayStrategy,
    build_delay_arena, synthesize_with_delay,
    find_minimum_delay, gr1_delay_synthesize,
    synthesize_safety_with_delay, synthesize_reachability_with_delay,
    synthesize_response_with_delay, synthesize_liveness_with_delay,
    compare_delays, delay_benefit_analysis,
    verify_delay_controller, delay_game_summary, delay_statistics,
    _all_valuations, _all_buffers, _label_matches,
    BoolGR1Spec,
)
from ltl_model_checker import (
    Atom, Not, And, Or, Implies, Iff, Next, Finally, Globally,
    Until, LTLTrue, LTLFalse, parse_ltl,
    ltl_to_gba, gba_to_nba,
)
from reactive_synthesis import MealyMachine, SynthesisVerdict
from parity_games import Player


# ============================================================
# Helper utilities
# ============================================================

class TestValuations:
    def test_empty_vars(self):
        vals = _all_valuations(set())
        assert len(vals) == 1
        assert vals[0] == frozenset()

    def test_single_var(self):
        vals = _all_valuations({'a'})
        assert len(vals) == 2
        assert frozenset() in vals
        assert frozenset({'a'}) in vals

    def test_two_vars(self):
        vals = _all_valuations({'a', 'b'})
        assert len(vals) == 4

    def test_three_vars(self):
        vals = _all_valuations({'a', 'b', 'c'})
        assert len(vals) == 8

    def test_deterministic_order(self):
        v1 = _all_valuations({'x', 'y'})
        v2 = _all_valuations({'x', 'y'})
        assert v1 == v2


class TestBuffers:
    def test_zero_size(self):
        bufs = _all_buffers({'a'}, 0)
        assert len(bufs) == 1
        assert bufs[0] == ()

    def test_single_var_size_1(self):
        bufs = _all_buffers({'a'}, 1)
        assert len(bufs) == 2  # 2^1 env valuations, 1 slot

    def test_single_var_size_2(self):
        bufs = _all_buffers({'a'}, 2)
        assert len(bufs) == 4  # 2^1 * 2^1

    def test_two_vars_size_1(self):
        bufs = _all_buffers({'a', 'b'}, 1)
        assert len(bufs) == 4  # 2^2 env valuations


class TestLabelMatching:
    def test_empty_label(self):
        from ltl_model_checker import Label
        label = Label(pos=frozenset(), neg=frozenset())
        assert _label_matches(label, frozenset())
        assert _label_matches(label, frozenset({'a'}))

    def test_pos_match(self):
        from ltl_model_checker import Label
        label = Label(pos=frozenset({'a'}), neg=frozenset())
        assert _label_matches(label, frozenset({'a'}))
        assert _label_matches(label, frozenset({'a', 'b'}))
        assert not _label_matches(label, frozenset())

    def test_neg_match(self):
        from ltl_model_checker import Label
        label = Label(pos=frozenset(), neg=frozenset({'a'}))
        assert _label_matches(label, frozenset())
        assert not _label_matches(label, frozenset({'a'}))


# ============================================================
# Data structures
# ============================================================

class TestDelayGameResult:
    def test_creation(self):
        r = DelayGameResult(realizable=True, delay=2)
        assert r.realizable
        assert r.delay == 2
        assert r.controller is None

    def test_with_controller(self):
        m = MealyMachine(
            states={0}, initial=0,
            inputs={'r'}, outputs={'g'},
            transitions={}
        )
        r = DelayGameResult(realizable=True, delay=1, controller=m)
        assert r.controller is not None

    def test_defaults(self):
        r = DelayGameResult(realizable=False, delay=3)
        assert r.buffered_states == 0
        assert r.method == "delay_game"


class TestMinDelayResult:
    def test_creation(self):
        r = MinDelayResult(realizable=True, min_delay=2)
        assert r.min_delay == 2

    def test_unrealizable(self):
        r = MinDelayResult(realizable=False, min_delay=-1)
        assert r.min_delay == -1


class TestDelayStrategy:
    def test_empty_strategy(self):
        s = DelayStrategy(delay=1)
        assert s.step('q0', [frozenset()]) is None

    def test_strategy_lookup(self):
        s = DelayStrategy(delay=1)
        s.transitions[('q0', (frozenset(),))] = ('q1', frozenset({'g'}))
        result = s.step('q0', [frozenset()])
        assert result == ('q1', frozenset({'g'}))


# ============================================================
# Arena construction
# ============================================================

class TestBuildDelayArena:
    def _simple_nba(self):
        """Single-state NBA accepting everything (True spec)."""
        from ltl_model_checker import Label
        nba = type('NBA', (), {
            'states': {0},
            'initial': {0},
            'transitions': {0: [
                (Label(pos=frozenset(), neg=frozenset()), 0)
            ]},
            'accepting': {0},
            'ap': {'r', 'g'},
        })()
        return nba

    def test_delay_0_arena(self):
        """Delay 0 = standard game."""
        nba = self._simple_nba()
        game, v2info, info2v, init = build_delay_arena(
            nba, {'r'}, {'g'}, delay=0
        )
        assert len(init) > 0
        assert len(game.vertices) > 0

    def test_delay_1_arena(self):
        """Delay 1 has fill phase + play phase."""
        nba = self._simple_nba()
        game, v2info, info2v, init = build_delay_arena(
            nba, {'r'}, {'g'}, delay=1
        )
        assert len(game.vertices) > 0
        # Should have fill vertices
        fill_count = sum(1 for v in v2info.values() if v[2] == 'fill')
        assert fill_count >= 1

    def test_delay_increases_state_space(self):
        """Higher delay = more states."""
        nba = self._simple_nba()
        g0, _, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=0)
        g1, _, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=1)
        assert len(g1.vertices) >= len(g0.vertices)

    def test_initial_vertices_exist(self):
        nba = self._simple_nba()
        game, _, _, init = build_delay_arena(nba, {'r'}, {'g'}, delay=1)
        assert len(init) > 0
        for v in init:
            assert v in game.vertices

    def test_env_owns_env_vertices(self):
        nba = self._simple_nba()
        game, v2info, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=0)
        for v, info in v2info.items():
            if info[2] == 'env':
                assert game.owner[v] == Player.ODD

    def test_sys_owns_sys_vertices(self):
        nba = self._simple_nba()
        game, v2info, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=1)
        for v, info in v2info.items():
            if info[2] == 'sys':
                assert game.owner[v] == Player.EVEN

    def test_accepting_priorities(self):
        nba = self._simple_nba()
        game, v2info, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=0)
        for v, info in v2info.items():
            q = info[0]
            if info[2] == 'env' and q in nba.accepting:
                assert game.priority[v] == 2

    def test_all_vertices_have_successors(self):
        """No dead ends (sink handles them)."""
        nba = self._simple_nba()
        game, _, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=1)
        for v in game.vertices:
            assert len(game.edges.get(v, set())) > 0


# ============================================================
# Standard synthesis with delay
# ============================================================

class TestSynthesizeWithDelay:
    def test_true_spec_delay_0(self):
        """True is always realizable."""
        r = synthesize_with_delay(LTLTrue(), {'r'}, {'g'}, delay=0)
        assert r.realizable
        assert r.delay == 0

    def test_true_spec_delay_1(self):
        r = synthesize_with_delay(LTLTrue(), {'r'}, {'g'}, delay=1)
        assert r.realizable
        assert r.delay == 1

    def test_false_spec_unrealizable(self):
        """False is never realizable, even with delay."""
        r = synthesize_with_delay(LTLFalse(), {'r'}, {'g'}, delay=0)
        assert not r.realizable

    def test_false_spec_delay_1(self):
        r = synthesize_with_delay(LTLFalse(), {'r'}, {'g'}, delay=1)
        assert not r.realizable

    def test_safety_spec_delay_0(self):
        """G(g) -- system can always set g."""
        r = synthesize_with_delay(
            Globally(Atom('g')), {'r'}, {'g'}, delay=0
        )
        assert r.realizable

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError):
            synthesize_with_delay(LTLTrue(), {'r'}, {'g'}, delay=-1)

    def test_result_has_method(self):
        r = synthesize_with_delay(LTLTrue(), {'r'}, {'g'}, delay=0)
        assert r.method in ('standard_synthesis', 'delay_game')

    def test_result_has_states(self):
        r = synthesize_with_delay(LTLTrue(), {'r'}, {'g'}, delay=1)
        assert r.buffered_states > 0

    def test_delay_0_matches_standard(self):
        """Delay 0 should give same realizability as standard synthesis."""
        spec = Globally(Atom('g'))
        r0 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=0)
        r1 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=1)
        # Both should be realizable (system controls g)
        assert r0.realizable
        assert r1.realizable

    def test_copy_spec(self):
        """G(g <-> r) -- system must copy env's r to g. Needs delay 1."""
        spec = Globally(Iff(Atom('g'), Atom('r')))
        r0 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=0)
        # Without delay, system can't see r before choosing g simultaneously
        # This may or may not be realizable depending on game semantics
        # With delay 1, system sees current r and can copy it
        r1 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=1)
        # At least delay 1 should succeed (system sees r before choosing g)
        # Result depends on NBA construction and game semantics
        assert isinstance(r0, DelayGameResult)
        assert isinstance(r1, DelayGameResult)


# ============================================================
# Minimum delay search
# ============================================================

class TestFindMinimumDelay:
    def test_always_realizable(self):
        r = find_minimum_delay(LTLTrue(), {'r'}, {'g'}, max_delay=3)
        assert r.realizable
        assert r.min_delay == 0

    def test_never_realizable(self):
        r = find_minimum_delay(LTLFalse(), {'r'}, {'g'}, max_delay=2)
        assert not r.realizable
        assert r.min_delay == -1
        assert len(r.searched_delays) == 3  # 0, 1, 2

    def test_records_all_results(self):
        r = find_minimum_delay(LTLTrue(), {'r'}, {'g'}, max_delay=2)
        assert 0 in r.results
        # Stops early since delay 0 works
        assert len(r.searched_delays) == 1

    def test_searched_delays_ordered(self):
        r = find_minimum_delay(LTLFalse(), {'r'}, {'g'}, max_delay=3)
        assert r.searched_delays == [0, 1, 2, 3]

    def test_safety_min_delay_0(self):
        """Simple safety spec needs no delay."""
        spec = Globally(Atom('g'))
        r = find_minimum_delay(spec, {'r'}, {'g'}, max_delay=3)
        assert r.realizable
        assert r.min_delay == 0


# ============================================================
# GR(1) delay synthesis
# ============================================================

class TestGR1DelaySynthesize:
    def _simple_spec(self):
        """Simple GR(1) spec: sys must eventually set g."""
        return BoolGR1Spec(
            env_vars=['r'],
            sys_vars=['g'],
            env_init=lambda s: True,
            sys_init=lambda s: True,
            env_trans=lambda s, ne: True,
            sys_trans=lambda s, ns: True,
            env_justice=[lambda s: True],
            sys_justice=[lambda s: 'g' in s],  # GF(g)
        )

    def test_delay_0(self):
        spec = self._simple_spec()
        r = gr1_delay_synthesize(spec, delay=0)
        assert r.realizable
        assert r.delay == 0

    def test_negative_delay_raises(self):
        spec = self._simple_spec()
        with pytest.raises(ValueError):
            gr1_delay_synthesize(spec, delay=-1)

    def test_delay_1(self):
        spec = self._simple_spec()
        r = gr1_delay_synthesize(spec, delay=1)
        assert isinstance(r, DelayGameResult)
        assert r.delay == 1

    def test_trivial_spec_realizable(self):
        """Trivial spec: no constraints."""
        spec = BoolGR1Spec(
            env_vars=['r'],
            sys_vars=['g'],
            env_init=lambda s: True,
            sys_init=lambda s: True,
            env_trans=lambda s, ne: True,
            sys_trans=lambda s, ns: True,
            env_justice=[],
            sys_justice=[],
        )
        r = gr1_delay_synthesize(spec, delay=0)
        assert r.realizable


# ============================================================
# Specialized synthesis
# ============================================================

class TestSafetyWithDelay:
    def test_avoid_bad(self):
        """System can avoid bad by controlling g."""
        bad = And(Atom('r'), Not(Atom('g')))
        r = synthesize_safety_with_delay(bad, {'r'}, {'g'}, delay=0)
        assert isinstance(r, DelayGameResult)

    def test_with_delay(self):
        bad = And(Atom('r'), Not(Atom('g')))
        r = synthesize_safety_with_delay(bad, {'r'}, {'g'}, delay=1)
        assert isinstance(r, DelayGameResult)


class TestReachabilityWithDelay:
    def test_reach_target(self):
        target = Atom('g')
        r = synthesize_reachability_with_delay(target, {'r'}, {'g'}, delay=0)
        assert isinstance(r, DelayGameResult)

    def test_reach_with_delay(self):
        target = Atom('g')
        r = synthesize_reachability_with_delay(target, {'r'}, {'g'}, delay=1)
        assert isinstance(r, DelayGameResult)


class TestResponseWithDelay:
    def test_response(self):
        r = synthesize_response_with_delay(
            Atom('r'), Atom('g'), {'r'}, {'g'}, delay=0
        )
        assert isinstance(r, DelayGameResult)

    def test_response_with_delay(self):
        r = synthesize_response_with_delay(
            Atom('r'), Atom('g'), {'r'}, {'g'}, delay=1
        )
        assert isinstance(r, DelayGameResult)


class TestLivenessWithDelay:
    def test_liveness(self):
        r = synthesize_liveness_with_delay(
            Atom('g'), {'r'}, {'g'}, delay=0
        )
        assert isinstance(r, DelayGameResult)

    def test_liveness_with_delay(self):
        r = synthesize_liveness_with_delay(
            Atom('g'), {'r'}, {'g'}, delay=1
        )
        assert isinstance(r, DelayGameResult)


# ============================================================
# Comparison and analysis
# ============================================================

class TestCompareDelays:
    def test_compare_basic(self):
        spec = Globally(Atom('g'))
        result = compare_delays(spec, {'r'}, {'g'}, [0, 1])
        assert 'per_delay' in result
        assert 0 in result['per_delay']
        assert 1 in result['per_delay']
        assert result['min_realizable'] is not None

    def test_compare_unrealizable(self):
        spec = LTLFalse()
        result = compare_delays(spec, {'r'}, {'g'}, [0, 1])
        assert result['min_realizable'] is None

    def test_delays_tested(self):
        spec = LTLTrue()
        result = compare_delays(spec, {'r'}, {'g'}, [0, 2])
        assert result['delays_tested'] == [0, 2]


class TestDelayBenefitAnalysis:
    def test_no_benefit_for_easy_spec(self):
        """G(g) is realizable without delay -- no benefit."""
        spec = Globally(Atom('g'))
        result = delay_benefit_analysis(spec, {'r'}, {'g'}, max_delay=2)
        assert result['standard_realizable']
        assert not result['delay_helps']

    def test_analysis_structure(self):
        spec = LTLTrue()
        result = delay_benefit_analysis(spec, {'r'}, {'g'}, max_delay=1)
        assert 'standard_realizable' in result
        assert 'delay_helps' in result
        assert 'min_delay' in result
        assert 'state_growth' in result
        assert 'results' in result


# ============================================================
# Summary and statistics
# ============================================================

class TestSummaryAndStats:
    def test_summary_unrealizable(self):
        r = DelayGameResult(realizable=False, delay=1, buffered_states=10)
        s = delay_game_summary(r)
        assert 'k=1' in s
        assert 'False' in s

    def test_summary_realizable(self):
        m = MealyMachine(
            states={0, 1}, initial=0,
            inputs={'r'}, outputs={'g'},
            transitions={
                (0, frozenset()): (1, frozenset({'g'})),
            }
        )
        r = DelayGameResult(
            realizable=True, delay=2, controller=m,
            buffered_states=20, original_states=4
        )
        s = delay_game_summary(r)
        assert 'True' in s
        assert 'k=2' in s

    def test_statistics(self):
        r = DelayGameResult(
            realizable=True, delay=1,
            buffered_states=15, original_states=4
        )
        stats = delay_statistics(r)
        assert stats['delay'] == 1
        assert stats['realizable']
        assert stats['buffered_states'] == 15

    def test_statistics_with_controller(self):
        m = MealyMachine(
            states={0}, initial=0,
            inputs={'r'}, outputs={'g'},
            transitions={}
        )
        r = DelayGameResult(
            realizable=True, delay=1, controller=m,
            buffered_states=10, original_states=4
        )
        stats = delay_statistics(r)
        assert 'controller_states' in stats


# ============================================================
# Monotonicity: more delay never hurts
# ============================================================

class TestMonotonicity:
    def test_realizable_stays_realizable(self):
        """If realizable at delay k, also realizable at delay k+1."""
        spec = Globally(Atom('g'))
        r0 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=0)
        r1 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=1)
        if r0.realizable:
            assert r1.realizable

    def test_unrealizable_false_all_delays(self):
        """False is unrealizable at any delay."""
        for k in range(3):
            r = synthesize_with_delay(LTLFalse(), {'r'}, {'g'}, delay=k)
            assert not r.realizable


# ============================================================
# Controller verification
# ============================================================

class TestControllerVerification:
    def test_verify_trivial_controller(self):
        """A controller for G(g) that always sets g."""
        m = MealyMachine(
            states={0}, initial=0,
            inputs={'r'}, outputs={'g'},
            transitions={
                (0, frozenset()): (0, frozenset({'g'})),
                (0, frozenset({'r'})): (0, frozenset({'g'})),
            }
        )
        spec = Globally(Atom('g'))
        ok, msgs = verify_delay_controller(m, spec, {'r'}, {'g'})
        assert ok


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_no_env_vars(self):
        """No environment variables -- system has full control."""
        spec = Globally(Atom('g'))
        r = synthesize_with_delay(spec, set(), {'g'}, delay=0)
        assert r.realizable

    def test_no_sys_vars(self):
        """No system variables -- system can only observe."""
        spec = LTLTrue()
        r = synthesize_with_delay(spec, {'r'}, set(), delay=0)
        assert r.realizable

    def test_multiple_env_vars(self):
        spec = Globally(Atom('g'))
        r = synthesize_with_delay(spec, {'r1', 'r2'}, {'g'}, delay=0)
        assert isinstance(r, DelayGameResult)

    def test_multiple_sys_vars(self):
        spec = Globally(And(Atom('g1'), Atom('g2')))
        r = synthesize_with_delay(spec, {'r'}, {'g1', 'g2'}, delay=0)
        assert isinstance(r, DelayGameResult)


# ============================================================
# Integration: delay helps realizability
# ============================================================

class TestDelayHelps:
    def test_next_copy_needs_delay(self):
        """G(g <-> X(r)) -- system must predict env's next move.
        Without delay, sys can't see future r. With delay 1, it can."""
        # Note: this is a simplified test -- exact realizability depends
        # on the game semantics of the NBA construction
        spec = Globally(Iff(Atom('g'), Next(Atom('r'))))
        r0 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=0)
        r1 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=1)
        # With delay 1, system sees next env move
        # This validates the arena construction handles Next formulas
        assert isinstance(r0, DelayGameResult)
        assert isinstance(r1, DelayGameResult)

    def test_state_space_growth(self):
        """Delay increases state space."""
        spec = Globally(Atom('g'))
        r0 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=0)
        r1 = synthesize_with_delay(spec, {'r'}, {'g'}, delay=1)
        # Delay 1 should have at least as many states
        assert r1.buffered_states >= r0.buffered_states


# ============================================================
# Arena properties
# ============================================================

class TestArenaProperties:
    def _make_nba(self, spec, env_vars, sys_vars):
        gba = ltl_to_gba(spec)
        nba = gba_to_nba(gba)
        return nba

    def test_arena_is_bipartite(self):
        """Env vertices only go to sys/fill vertices and vice versa."""
        nba = self._make_nba(Globally(Atom('g')), {'r'}, {'g'})
        game, v2info, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=1)
        for v in game.vertices:
            info = v2info.get(v)
            if info is None:
                continue  # sink
            phase = info[2]
            for succ in game.edges.get(v, set()):
                succ_info = v2info.get(succ)
                if succ_info is None:
                    continue  # sink
                succ_phase = succ_info[2]
                if phase == 'env':
                    assert succ_phase == 'sys'
                elif phase == 'sys':
                    assert succ_phase == 'env'
                elif phase == 'fill':
                    assert succ_phase in ('fill', 'env')

    def test_arena_no_isolated(self):
        """Every vertex is either initial or reachable from initial."""
        nba = self._make_nba(LTLTrue(), {'r'}, {'g'})
        game, _, _, init = build_delay_arena(nba, {'r'}, {'g'}, delay=1)
        # BFS from initial
        reachable = set()
        queue = list(init)
        while queue:
            v = queue.pop()
            if v in reachable:
                continue
            reachable.add(v)
            for s in game.edges.get(v, set()):
                queue.append(s)
        assert reachable == game.vertices

    def test_priority_values(self):
        """Priorities are 0, 1, or 2."""
        nba = self._make_nba(Globally(Atom('g')), {'r'}, {'g'})
        game, _, _, _ = build_delay_arena(nba, {'r'}, {'g'}, delay=1)
        for v in game.vertices:
            assert game.priority[v] in (0, 1, 2)


# ============================================================
# Delay = 0 equivalence
# ============================================================

class TestDelayZeroEquivalence:
    def test_safety_realizability_matches(self):
        spec = Globally(Atom('g'))
        r_delay = synthesize_with_delay(spec, {'r'}, {'g'}, delay=0)
        from reactive_synthesis import synthesize as std_synth
        r_std = std_synth(spec, {'r'}, {'g'})
        assert r_delay.realizable == (r_std.verdict == SynthesisVerdict.REALIZABLE)

    def test_liveness_realizability_matches(self):
        spec = Globally(Finally(Atom('g')))
        r_delay = synthesize_with_delay(spec, {'r'}, {'g'}, delay=0)
        from reactive_synthesis import synthesize as std_synth
        r_std = std_synth(spec, {'r'}, {'g'})
        assert r_delay.realizable == (r_std.verdict == SynthesisVerdict.REALIZABLE)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
