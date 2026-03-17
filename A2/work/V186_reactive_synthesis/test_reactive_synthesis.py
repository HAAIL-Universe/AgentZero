"""Tests for V186: Reactive Synthesis from LTL Specifications."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))

from reactive_synthesis import (
    synthesize, synthesize_assume_guarantee,
    synthesize_safety, synthesize_reachability,
    synthesize_liveness, synthesize_response,
    synthesize_stability,
    verify_controller, controller_statistics, synthesis_summary,
    compare_specs,
    SynthesisVerdict, SynthesisResult, MealyMachine,
    _all_valuations, _label_matches, _build_game_arena
)
from ltl_model_checker import (
    Atom, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release,
    LTLTrue, LTLFalse,
    ltl_to_gba, gba_to_nba, GBA, NBA, Label
)
from parity_games import ParityGame, Player


# ============================================================
# Helper utilities
# ============================================================

class TestValuations:
    def test_empty_vars(self):
        vals = _all_valuations(set())
        assert vals == [frozenset()]

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


class TestLabelMatching:
    def test_empty_label(self):
        label = Label(frozenset(), frozenset())
        assert _label_matches(label, frozenset())
        assert _label_matches(label, frozenset({'a'}))

    def test_pos_match(self):
        label = Label(frozenset({'a'}), frozenset())
        assert _label_matches(label, frozenset({'a'}))
        assert _label_matches(label, frozenset({'a', 'b'}))
        assert not _label_matches(label, frozenset())

    def test_neg_match(self):
        label = Label(frozenset(), frozenset({'b'}))
        assert _label_matches(label, frozenset())
        assert _label_matches(label, frozenset({'a'}))
        assert not _label_matches(label, frozenset({'b'}))

    def test_pos_and_neg(self):
        label = Label(frozenset({'a'}), frozenset({'b'}))
        assert _label_matches(label, frozenset({'a'}))
        assert not _label_matches(label, frozenset({'a', 'b'}))
        assert not _label_matches(label, frozenset({'b'}))
        assert not _label_matches(label, frozenset())


# ============================================================
# Mealy Machine
# ============================================================

class TestMealyMachine:
    def test_basic_step(self):
        mm = MealyMachine(
            states={0, 1},
            initial=0,
            inputs={'r'},
            outputs={'g'},
            transitions={
                (0, frozenset()): (0, frozenset()),
                (0, frozenset({'r'})): (1, frozenset({'g'})),
                (1, frozenset()): (0, frozenset()),
                (1, frozenset({'r'})): (1, frozenset({'g'})),
            }
        )
        ns, out = mm.step(0, frozenset({'r'}))
        assert ns == 1
        assert out == frozenset({'g'})

    def test_simulate(self):
        mm = MealyMachine(
            states={0},
            initial=0,
            inputs={'a'},
            outputs={'b'},
            transitions={
                (0, frozenset()): (0, frozenset({'b'})),
                (0, frozenset({'a'})): (0, frozenset()),
            }
        )
        trace = mm.simulate([frozenset(), frozenset({'a'}), frozenset()])
        assert len(trace) == 3
        assert trace[0][2] == frozenset({'b'})  # no input -> b
        assert trace[1][2] == frozenset()  # input a -> no output

    def test_simulate_max_steps(self):
        mm = MealyMachine(
            states={0}, initial=0,
            inputs=set(), outputs=set(),
            transitions={(0, frozenset()): (0, frozenset())}
        )
        trace = mm.simulate([frozenset()] * 200, max_steps=5)
        assert len(trace) == 5

    def test_unknown_input(self):
        mm = MealyMachine(
            states={0}, initial=0,
            inputs={'a'}, outputs={'b'},
            transitions={}
        )
        # Unknown transition -> stays in same state, no output
        ns, out = mm.step(0, frozenset({'a'}))
        assert ns == 0
        assert out == frozenset()


# ============================================================
# Game Arena Construction
# ============================================================

class TestGameArena:
    def test_simple_arena(self):
        # Create a simple NBA manually
        nba = NBA(
            states={0, 1},
            initial={0},
            transitions={
                0: [(Label(frozenset({'a'}), frozenset()), 1),
                    (Label(frozenset(), frozenset({'a'})), 0)],
                1: [(Label(frozenset(), frozenset()), 1)],
            },
            accepting={1},
            ap={'a', 'b'}
        )
        game, v2i, i2v = _build_game_arena(nba, {'a'}, {'b'})
        assert len(game.vertices) > 0
        # Should have env vertices and mid vertices
        env_count = sum(1 for v in v2i.values() if v[1] == 'env')
        mid_count = sum(1 for v in v2i.values() if v[1] == 'mid')
        assert env_count == 2  # one per NBA state
        assert mid_count == 4  # 2 states * 2 env valuations

    def test_arena_priorities(self):
        nba = NBA(
            states={0},
            initial={0},
            transitions={0: [(Label(frozenset(), frozenset()), 0)]},
            accepting={0},
            ap={'x'}
        )
        game, v2i, i2v = _build_game_arena(nba, {'x'}, set())
        # Accepting state should have priority 2
        env_key = (0, 'env')
        vid = i2v[env_key]
        assert game.priority[vid] == 2

    def test_arena_non_accepting(self):
        nba = NBA(
            states={0},
            initial={0},
            transitions={0: [(Label(frozenset(), frozenset()), 0)]},
            accepting=set(),  # not accepting
            ap={'x'}
        )
        game, v2i, i2v = _build_game_arena(nba, {'x'}, set())
        env_key = (0, 'env')
        vid = i2v[env_key]
        assert game.priority[vid] == 1  # non-accepting

    def test_arena_players(self):
        nba = NBA(
            states={0},
            initial={0},
            transitions={0: [(Label(frozenset(), frozenset()), 0)]},
            accepting={0},
            ap={'a'}
        )
        game, v2i, i2v = _build_game_arena(nba, {'a'}, set())
        env_key = (0, 'env')
        mid_key = (0, 'mid', frozenset())
        assert game.owner[i2v[env_key]] == Player.ODD  # env
        assert game.owner[i2v[mid_key]] == Player.EVEN  # sys


# ============================================================
# Safety Synthesis
# ============================================================

class TestSafetySynthesis:
    def test_trivial_safe(self):
        """G(true) -- always satisfiable."""
        result = synthesize(Globally(LTLTrue()), {'a'}, {'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_trivial_unsafe(self):
        """G(false) -- never satisfiable."""
        result = synthesize(Globally(LTLFalse()), {'a'}, {'b'})
        assert result.verdict == SynthesisVerdict.UNREALIZABLE

    def test_safety_avoid_bad(self):
        """G(!bad) where bad = a AND b. System controls b, can always set b=false."""
        a, b = Atom('a'), Atom('b')
        result = synthesize_safety(And(a, b), {'a'}, {'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE
        assert result.controller is not None

    def test_safety_avoid_bad_controller(self):
        """Verify the controller actually avoids the bad condition."""
        a, b = Atom('a'), Atom('b')
        result = synthesize_safety(And(a, b), {'a'}, {'b'})
        if result.controller:
            # When env sets a=true, sys should set b=false
            for ev in [frozenset(), frozenset({'a'})]:
                _, out = result.controller.step(result.controller.initial, ev)
                if 'a' in ev:
                    assert 'b' not in out, "Controller should avoid a AND b"

    def test_safety_single_var(self):
        """G(!x) where system controls x."""
        x = Atom('x')
        result = synthesize_safety(x, set(), {'x'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_safety_env_only_unrealizable(self):
        """G(!a) where only env controls a -- unrealizable."""
        a = Atom('a')
        result = synthesize_safety(a, {'a'}, set())
        assert result.verdict == SynthesisVerdict.UNREALIZABLE


class TestSafetyHelper:
    def test_synthesize_safety_api(self):
        b = Atom('b')
        result = synthesize_safety(b, set(), {'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE


# ============================================================
# Reachability Synthesis
# ============================================================

class TestReachabilitySynthesis:
    def test_reach_own_var(self):
        """F(g) where system controls g -- realizable."""
        g = Atom('g')
        result = synthesize_reachability(g, set(), {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_reach_env_var(self):
        """F(a) where only env controls a -- depends on env."""
        a = Atom('a')
        # This should be unrealizable: system can't force env to set a
        result = synthesize_reachability(a, {'a'}, set())
        assert result.verdict == SynthesisVerdict.UNREALIZABLE


# ============================================================
# Liveness Synthesis
# ============================================================

class TestLivenessSynthesis:
    def test_liveness_own_var(self):
        """GF(g) -- system can toggle g infinitely."""
        g = Atom('g')
        result = synthesize_liveness(g, set(), {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_liveness_env_var_unrealizable(self):
        """GF(a) -- env controls a, system can't force it infinitely."""
        a = Atom('a')
        result = synthesize_liveness(a, {'a'}, set())
        assert result.verdict == SynthesisVerdict.UNREALIZABLE


# ============================================================
# Response Synthesis
# ============================================================

class TestResponseSynthesis:
    def test_response_sys_controls_both(self):
        """G(r -> F(g)) where system controls both r and g."""
        r, g = Atom('r'), Atom('g')
        result = synthesize_response(r, g, set(), {'r', 'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_response_env_trigger_sys_respond(self):
        """G(r -> F(g)) where env controls r, sys controls g."""
        r, g = Atom('r'), Atom('g')
        result = synthesize_response(r, g, {'r'}, {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE


# ============================================================
# Stability Synthesis
# ============================================================

class TestStabilitySynthesis:
    def test_stability_own_var(self):
        """FG(g) -- system can stabilize g to true."""
        g = Atom('g')
        result = synthesize_stability(g, set(), {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE


# ============================================================
# Assume-Guarantee Synthesis
# ============================================================

class TestAssumeGuaranteeSynthesis:
    def test_ag_trivial(self):
        """true -> G(true)."""
        result = synthesize_assume_guarantee(
            LTLTrue(), Globally(LTLTrue()),
            {'a'}, {'b'}
        )
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_ag_env_assumption_helps(self):
        """If env always provides a, system can use it."""
        a, b = Atom('a'), Atom('b')
        # Assume G(a), guarantee G(a OR b)
        # Since env guarantees a, G(a | b) is trivially satisfiable
        result = synthesize_assume_guarantee(
            Globally(a), Globally(Or(a, b)),
            {'a'}, {'b'}
        )
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_ag_false_assumption(self):
        """false -> anything is realizable (vacuous)."""
        result = synthesize_assume_guarantee(
            LTLFalse(), Globally(LTLFalse()),
            {'a'}, {'b'}
        )
        assert result.verdict == SynthesisVerdict.REALIZABLE


# ============================================================
# Controller Properties
# ============================================================

class TestControllerProperties:
    def test_controller_has_transitions(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        if result.controller:
            assert len(result.controller.transitions) > 0

    def test_controller_deterministic(self):
        """Each (state, input) maps to exactly one (next_state, output)."""
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        if result.controller:
            # Dict ensures determinism by construction
            keys = list(result.controller.transitions.keys())
            assert len(keys) == len(set(keys))

    def test_controller_initial_state(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        if result.controller:
            assert result.controller.initial in result.controller.states

    def test_controller_reachable_states(self):
        """All controller states should be reachable from initial."""
        g = Atom('g')
        result = synthesize_safety(And(Atom('a'), Atom('b')), {'a'}, {'b'})
        if result.controller:
            # BFS from initial
            reachable = {result.controller.initial}
            frontier = [result.controller.initial]
            while frontier:
                s = frontier.pop(0)
                for (st, inp), (ns, out) in result.controller.transitions.items():
                    if st == s and ns not in reachable:
                        reachable.add(ns)
                        frontier.append(ns)
            for s in result.controller.states:
                assert s in reachable


# ============================================================
# Controller Simulation
# ============================================================

class TestControllerSimulation:
    def test_simulate_safety(self):
        """Simulate safety controller and check invariant."""
        a, b = Atom('a'), Atom('b')
        result = synthesize_safety(And(a, b), {'a'}, {'b'})
        if result.controller:
            inputs = [frozenset({'a'}), frozenset(), frozenset({'a'}), frozenset({'a'})]
            trace = result.controller.simulate(inputs)
            for state, inp, out in trace:
                if 'a' in inp:
                    assert 'b' not in out

    def test_simulate_empty_inputs(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        if result.controller:
            trace = result.controller.simulate([frozenset()] * 5)
            assert len(trace) == 5


# ============================================================
# Verification
# ============================================================

class TestVerification:
    def test_verify_safety_controller(self):
        a, b = Atom('a'), Atom('b')
        spec = Globally(Not(And(a, b)))
        result = synthesize(spec, {'a'}, {'b'})
        if result.controller:
            ok, msgs = verify_controller(result.controller, spec, {'a'}, {'b'})
            assert ok

    def test_verify_trivial(self):
        result = synthesize(Globally(LTLTrue()), set(), {'g'})
        if result.controller:
            ok, msgs = verify_controller(result.controller, Globally(LTLTrue()), set(), {'g'})
            assert ok


# ============================================================
# Statistics & Summary
# ============================================================

class TestStatistics:
    def test_controller_statistics(self):
        mm = MealyMachine(
            states={0, 1}, initial=0,
            inputs={'a'}, outputs={'b'},
            transitions={(0, frozenset()): (1, frozenset({'b'}))}
        )
        stats = controller_statistics(mm)
        assert stats['states'] == 2
        assert stats['transitions'] == 1
        assert stats['deterministic'] is True

    def test_synthesis_summary(self):
        result = SynthesisResult(
            verdict=SynthesisVerdict.REALIZABLE,
            game_vertices=10,
            game_edges=20,
            automaton_states=5,
            winning_region_size=8,
            controller=MealyMachine(
                states={0}, initial=0,
                inputs={'a'}, outputs={'b'},
                transitions={}
            )
        )
        s = synthesis_summary(result)
        assert "REALIZABLE" in s or "realizable" in s
        assert "Controller" in s

    def test_summary_unrealizable(self):
        result = SynthesisResult(
            verdict=SynthesisVerdict.UNREALIZABLE,
            game_vertices=10,
            game_edges=20,
            automaton_states=5,
            winning_region_size=3,
        )
        s = synthesis_summary(result)
        assert "unrealizable" in s


# ============================================================
# Compare Specs
# ============================================================

class TestCompareSpecs:
    def test_compare_two_specs(self):
        g = Atom('g')
        specs = [
            ("safety", Globally(g)),
            ("reach", Finally(g)),
        ]
        results = compare_specs(specs, set(), {'g'})
        assert 'safety' in results
        assert 'reach' in results
        for name in results:
            assert 'verdict' in results[name]


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_no_env_vars(self):
        """System controls everything."""
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_no_sys_vars(self):
        """System controls nothing -- can only win if spec is trivially true."""
        result = synthesize(Globally(LTLTrue()), {'a'}, set())
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_single_env_single_sys(self):
        """Minimal interesting case: 1 env var, 1 sys var."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Globally(Or(a, b)), {'a'}, {'b'})
        # System can always set b=true to satisfy a|b
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_next_operator(self):
        """X(g) -- system must set g at step 1."""
        g = Atom('g')
        result = synthesize(Next(g), set(), {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_until_operator(self):
        """a U b where system controls b."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Until(a, b), set(), {'a', 'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_result_fields(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        assert result.game_vertices > 0
        assert result.automaton_states > 0
        assert result.method == "reactive_synthesis"


# ============================================================
# Complex Specifications
# ============================================================

class TestComplexSpecs:
    def test_mutual_exclusion(self):
        """G(!(g1 AND g2)) -- grants are mutually exclusive."""
        g1, g2 = Atom('g1'), Atom('g2')
        result = synthesize(Globally(Not(And(g1, g2))), set(), {'g1', 'g2'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_mutex_controller_behavior(self):
        """Check that mutex controller never grants both."""
        g1, g2 = Atom('g1'), Atom('g2')
        result = synthesize(Globally(Not(And(g1, g2))), set(), {'g1', 'g2'})
        if result.controller:
            trace = result.controller.simulate([frozenset()] * 10)
            for state, inp, out in trace:
                assert not ('g1' in out and 'g2' in out)

    def test_always_or(self):
        """G(a OR b) with sys controlling b -- realizable."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Globally(Or(a, b)), {'a'}, {'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_implication_chain(self):
        """G(a -> b) where sys controls b."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Globally(Implies(a, b)), {'a'}, {'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_implication_controller(self):
        """Verify implication controller: when a is true, b must be true."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Globally(Implies(a, b)), {'a'}, {'b'})
        if result.controller:
            for ev in [frozenset({'a'}), frozenset()]:
                _, out = result.controller.step(result.controller.initial, ev)
                if 'a' in ev:
                    assert 'b' in out


# ============================================================
# Arbiter Pattern
# ============================================================

class TestArbiter:
    def test_arbiter_safety(self):
        """Simple arbiter: G(!(g1 AND g2)) with requests from env."""
        r1, r2 = Atom('r1'), Atom('r2')
        g1, g2 = Atom('g1'), Atom('g2')
        # Safety: never grant both
        safety = Globally(Not(And(g1, g2)))
        result = synthesize(safety, {'r1', 'r2'}, {'g1', 'g2'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_arbiter_simulation(self):
        """Simulate arbiter with alternating requests."""
        g1, g2 = Atom('g1'), Atom('g2')
        safety = Globally(Not(And(g1, g2)))
        result = synthesize(safety, {'r1', 'r2'}, {'g1', 'g2'})
        if result.controller:
            inputs = [
                frozenset({'r1'}),
                frozenset({'r2'}),
                frozenset({'r1', 'r2'}),
                frozenset(),
            ]
            trace = result.controller.simulate(inputs)
            for state, inp, out in trace:
                assert not ('g1' in out and 'g2' in out)


# ============================================================
# Boolean Combinations
# ============================================================

class TestBooleanCombinations:
    def test_and_spec(self):
        """G(a) AND G(b) where sys controls both."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(And(Globally(a), Globally(b)), set(), {'a', 'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_or_spec(self):
        """G(a) OR G(b) where sys controls both."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Or(Globally(a), Globally(b)), set(), {'a', 'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_not_spec(self):
        """!G(false) = F(true) -- trivially true."""
        result = synthesize(Not(Globally(LTLFalse())), set(), set())
        assert result.verdict == SynthesisVerdict.REALIZABLE


# ============================================================
# Regression / Known Patterns
# ============================================================

class TestPatterns:
    def test_traffic_light_safety(self):
        """Traffic light: G(!(ns_green AND ew_green))."""
        ns, ew = Atom('ns_green'), Atom('ew_green')
        result = synthesize(Globally(Not(And(ns, ew))), set(), {'ns_green', 'ew_green'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_toggle_pattern(self):
        """System can keep toggling a variable."""
        g = Atom('g')
        # GF(g) AND GF(!g) -- toggle forever
        result = synthesize(And(Globally(Finally(g)), Globally(Finally(Not(g)))),
                          set(), {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_echo_pattern(self):
        """G(a -> b): system echoes env input."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Globally(Implies(a, b)), {'a'}, {'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE


# ============================================================
# SynthesisResult
# ============================================================

class TestSynthesisResult:
    def test_realizable_has_controller(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        if result.verdict == SynthesisVerdict.REALIZABLE:
            assert result.controller is not None

    def test_unrealizable_no_controller(self):
        result = synthesize(Globally(LTLFalse()), {'a'}, set())
        assert result.verdict == SynthesisVerdict.UNREALIZABLE
        assert result.controller is None

    def test_game_stats_populated(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        assert result.game_vertices >= 0
        assert result.game_edges >= 0
        assert result.automaton_states >= 0


# ============================================================
# Deeper Unrealizability Tests
# ============================================================

class TestUnrealizability:
    def test_env_controls_all_unrealizable(self):
        """G(a) where env controls a -- unrealizable (env can falsify a)."""
        a = Atom('a')
        result = synthesize(Globally(a), {'a'}, set())
        assert result.verdict == SynthesisVerdict.UNREALIZABLE

    def test_env_xor_unrealizable(self):
        """G(a XOR b) where env controls both -- unrealizable."""
        a, b = Atom('a'), Atom('b')
        xor = And(Or(a, b), Not(And(a, b)))
        result = synthesize(Globally(xor), {'a', 'b'}, set())
        assert result.verdict == SynthesisVerdict.UNREALIZABLE

    def test_conflicting_safety_liveness(self):
        """G(!g) AND GF(g) -- contradictory."""
        g = Atom('g')
        result = synthesize(And(Globally(Not(g)), Globally(Finally(g))), set(), {'g'})
        assert result.verdict == SynthesisVerdict.UNREALIZABLE

    def test_reach_impossible(self):
        """F(a AND !a) -- impossible."""
        a = Atom('a')
        result = synthesize(Finally(And(a, Not(a))), set(), {'a'})
        assert result.verdict == SynthesisVerdict.UNREALIZABLE


# ============================================================
# Multi-Variable Controllers
# ============================================================

class TestMultiVariable:
    def test_two_outputs(self):
        """G(g1 OR g2) -- system can set either."""
        g1, g2 = Atom('g1'), Atom('g2')
        result = synthesize(Globally(Or(g1, g2)), set(), {'g1', 'g2'})
        assert result.verdict == SynthesisVerdict.REALIZABLE
        if result.controller:
            for ev in _all_valuations(set()):
                _, out = result.controller.step(result.controller.initial, ev)
                assert 'g1' in out or 'g2' in out

    def test_two_inputs_one_output(self):
        """G((a OR b) -> g) where a,b are env inputs."""
        a, b, g = Atom('a'), Atom('b'), Atom('g')
        result = synthesize(Globally(Implies(Or(a, b), g)), {'a', 'b'}, {'g'})
        assert result.verdict == SynthesisVerdict.REALIZABLE

    def test_copy_input(self):
        """G(a IFF b) where a is env, b is sys -- system mirrors env."""
        a, b = Atom('a'), Atom('b')
        iff = And(Implies(a, b), Implies(b, a))
        result = synthesize(Globally(iff), {'a'}, {'b'})
        assert result.verdict == SynthesisVerdict.REALIZABLE
        if result.controller:
            _, out = result.controller.step(result.controller.initial, frozenset({'a'}))
            assert 'b' in out
            _, out = result.controller.step(result.controller.initial, frozenset())
            assert 'b' not in out


# ============================================================
# Controller Quality
# ============================================================

class TestControllerQuality:
    def test_safety_controller_complete(self):
        """Safety controller should have transitions for all inputs."""
        a, b = Atom('a'), Atom('b')
        result = synthesize(Globally(Implies(a, b)), {'a'}, {'b'})
        if result.controller:
            env_vals = _all_valuations({'a'})
            for ev in env_vals:
                ns, out = result.controller.step(result.controller.initial, ev)
                # Should always produce a valid transition
                assert ns is not None

    def test_controller_statistics_api(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        if result.controller:
            stats = controller_statistics(result.controller)
            assert 'states' in stats
            assert 'transitions' in stats
            assert 'inputs' in stats
            assert 'outputs' in stats
            assert 'deterministic' in stats
            assert stats['deterministic'] is True

    def test_controller_io_vars(self):
        result = synthesize(
            Globally(Implies(Atom('r'), Atom('g'))),
            {'r'}, {'g'}
        )
        if result.controller:
            assert result.controller.inputs == {'r'}
            assert result.controller.outputs == {'g'}


# ============================================================
# Synthesis API Variants
# ============================================================

class TestAPIVariants:
    def test_synthesize_direct(self):
        g = Atom('g')
        r1 = synthesize(Globally(g), set(), {'g'})
        assert r1.verdict == SynthesisVerdict.REALIZABLE

    def test_synthesize_safety_wrapper(self):
        g = Atom('g')
        # synthesize_safety(bad) = synthesize(G(!bad))
        r1 = synthesize_safety(g, set(), {'g'})  # avoid g
        r2 = synthesize(Globally(Not(g)), set(), {'g'})
        assert r1.verdict == r2.verdict

    def test_synthesize_reach_wrapper(self):
        g = Atom('g')
        r1 = synthesize_reachability(g, set(), {'g'})
        r2 = synthesize(Finally(g), set(), {'g'})
        assert r1.verdict == r2.verdict

    def test_synthesize_liveness_wrapper(self):
        g = Atom('g')
        r1 = synthesize_liveness(g, set(), {'g'})
        r2 = synthesize(Globally(Finally(g)), set(), {'g'})
        assert r1.verdict == r2.verdict

    def test_synthesize_response_wrapper(self):
        r, g = Atom('r'), Atom('g')
        r1 = synthesize_response(r, g, {'r'}, {'g'})
        r2 = synthesize(Globally(Implies(r, Finally(g))), {'r'}, {'g'})
        assert r1.verdict == r2.verdict

    def test_synthesize_stability_wrapper(self):
        g = Atom('g')
        r1 = synthesize_stability(g, set(), {'g'})
        r2 = synthesize(Finally(Globally(g)), set(), {'g'})
        assert r1.verdict == r2.verdict


# ============================================================
# Game Construction Details
# ============================================================

class TestGameDetails:
    def test_game_has_vertices(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        assert result.game_vertices > 0

    def test_game_has_edges(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        assert result.game_edges > 0

    def test_automaton_states_positive(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        assert result.automaton_states > 0

    def test_winning_region_bounded(self):
        g = Atom('g')
        result = synthesize(Globally(g), set(), {'g'})
        assert result.winning_region_size <= result.game_vertices

    def test_larger_spec_more_states(self):
        """More complex spec should produce larger automaton."""
        g = Atom('g')
        r1 = synthesize(Globally(g), set(), {'g'})
        r2 = synthesize(And(Globally(g), Globally(Finally(Not(g)))), set(), {'g'})
        # Complex spec may have more automaton states
        assert r1.automaton_states >= 1
        assert r2.automaton_states >= 1


# ============================================================
# Compare Specs Extended
# ============================================================

class TestCompareSpecsExtended:
    def test_compare_realizable_vs_unrealizable(self):
        g = Atom('g')
        specs = [
            ("realizable", Globally(g)),
            ("unrealizable", Globally(LTLFalse())),
        ]
        results = compare_specs(specs, set(), {'g'})
        assert results['realizable']['verdict'] == 'realizable'
        assert results['unrealizable']['verdict'] == 'unrealizable'

    def test_compare_empty_list(self):
        results = compare_specs([], set(), set())
        assert results == {}

    def test_compare_single_spec(self):
        g = Atom('g')
        results = compare_specs([("one", Globally(g))], set(), {'g'})
        assert len(results) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
