"""Tests for V075: Reactive Synthesis (GR(1))"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

from reactive_synthesis import (
    BDD, GR1Spec, GR1Arena, SynthResult, SynthesisOutput, MealyMachine,
    gr1_synthesis, safety_synthesis, reachability_synthesis, buchi_synthesis,
    make_gr1_game, synthesize_arbiter, synthesize_traffic_light,
    simulate_strategy, extract_counterstrategy, compare_synthesis_approaches,
    check_realizability, explicit_to_gr1, extract_mealy_machine,
    verify_controller
)


# ===== Section 1: Basic GR1Arena =====

class TestGR1Arena:
    def test_arena_creation(self):
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        e_next = bdd.named_var("e'")
        s_next = bdd.named_var("s'")
        arena = GR1Arena(bdd, ['e'], ['s'])
        assert arena.env_vars == ['e']
        assert arena.sys_vars == ['s']
        assert len(arena.env_indices) == 1
        assert len(arena.sys_indices) == 1

    def test_to_next_rename(self):
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        e_next = bdd.named_var("e'")
        s_next = bdd.named_var("s'")
        arena = GR1Arena(bdd, ['e'], ['s'])
        # e AND s should become e' AND s'
        formula = bdd.AND(e, s)
        renamed = arena._to_next(formula)
        # Check: renamed should be equivalent to e' AND s'
        expected = bdd.AND(e_next, s_next)
        assert renamed._id == expected._id

    def test_cpre_trivial_true(self):
        """System can always force staying in TRUE."""
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        e_next = bdd.named_var("e'")
        s_next = bdd.named_var("s'")
        arena = GR1Arena(bdd, ['e'], ['s'])
        result = arena.controllable_predecessor(bdd.TRUE, bdd.TRUE, bdd.TRUE)
        assert result._id == bdd.TRUE._id

    def test_cpre_false(self):
        """System cannot force reaching FALSE from anywhere."""
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        e_next = bdd.named_var("e'")
        s_next = bdd.named_var("s'")
        arena = GR1Arena(bdd, ['e'], ['s'])
        result = arena.controllable_predecessor(bdd.FALSE, bdd.TRUE, bdd.TRUE)
        assert result._id == bdd.FALSE._id


# ===== Section 2: Safety Synthesis =====

class TestSafetySynthesis:
    def test_trivial_safe(self):
        """Everything safe => realizable."""
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        bdd.named_var("e'")
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=['e'], sys_vars=['s'])
        result = safety_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_impossible_safety(self):
        """sys_safe = FALSE => unrealizable."""
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        bdd.named_var("e'")
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=['e'], sys_vars=['s'], sys_safe=bdd.FALSE)
        result = safety_synthesis(bdd, spec)
        assert result.result == SynthResult.UNREALIZABLE

    def test_mutual_exclusion_safety(self):
        """System can always avoid both grants being on."""
        bdd = BDD()
        g0 = bdd.named_var('g0')
        g1 = bdd.named_var('g1')
        r0 = bdd.named_var('r0')
        r1 = bdd.named_var('r1')
        g0n = bdd.named_var("g0'")
        g1n = bdd.named_var("g1'")
        r0n = bdd.named_var("r0'")
        r1n = bdd.named_var("r1'")

        spec = GR1Spec(
            env_vars=['r0', 'r1'],
            sys_vars=['g0', 'g1'],
            sys_init=bdd.AND(bdd.NOT(g0), bdd.NOT(g1)),
            sys_safe=bdd.NOT(bdd.AND(g0n, g1n))  # NOT(g0' AND g1')
        )
        result = safety_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE


# ===== Section 3: Reachability Synthesis =====

class TestReachabilitySynthesis:
    def test_already_at_target(self):
        """Init state is already the target."""
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'], sys_init=s)
        target = s
        result = reachability_synthesis(bdd, spec, target)
        assert result.result == SynthResult.REALIZABLE

    def test_reachable_in_one_step(self):
        """System can reach target in one step."""
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        # Start at NOT s, target is s, system can set s'=TRUE
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_init=bdd.NOT(s),
            sys_safe=bdd.TRUE
        )
        result = reachability_synthesis(bdd, spec, s)
        assert result.result == SynthResult.REALIZABLE

    def test_unreachable_target(self):
        """System cannot reach target if safety prevents it."""
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        # Start at NOT s, target is s, but safety says s' must be FALSE
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_init=bdd.NOT(s),
            sys_safe=bdd.NOT(sn)  # s' must always be FALSE
        )
        result = reachability_synthesis(bdd, spec, s)
        assert result.result == SynthResult.UNREALIZABLE


# ===== Section 4: Buchi Synthesis =====

class TestBuchiSynthesis:
    def test_trivial_buchi(self):
        """Acceptance = TRUE => always accepting => realizable."""
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        result = buchi_synthesis(bdd, spec, bdd.TRUE)
        assert result.result == SynthResult.REALIZABLE

    def test_impossible_buchi(self):
        """Acceptance = FALSE => never accepting => unrealizable."""
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        result = buchi_synthesis(bdd, spec, bdd.FALSE)
        assert result.result == SynthResult.UNREALIZABLE

    def test_buchi_toggle(self):
        """System must infinitely often visit s=TRUE."""
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'], sys_safe=bdd.TRUE)
        result = buchi_synthesis(bdd, spec, s)
        assert result.result == SynthResult.REALIZABLE


# ===== Section 5: Full GR(1) Synthesis =====

class TestGR1Synthesis:
    def test_trivial_realizable(self):
        """No constraints => realizable."""
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_simple_liveness(self):
        """System must satisfy GF(s)."""
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_live=[s]
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_multiple_liveness(self):
        """System must satisfy GF(a) AND GF(b)."""
        bdd = BDD()
        a = bdd.named_var('a')
        b = bdd.named_var('b')
        bdd.named_var("a'")
        bdd.named_var("b'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['a', 'b'],
            sys_live=[a, b]
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_liveness_with_safety(self):
        """GF(s) but NOT(s') => unrealizable."""
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_safe=bdd.NOT(sn),  # s must always be FALSE next
            sys_init=bdd.NOT(s),
            sys_live=[s]  # but must infinitely often be TRUE
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.UNREALIZABLE

    def test_env_assumption_helps(self):
        """Env assumption makes otherwise-hard spec realizable.

        System must GF(s AND e) -- both must be TRUE infinitely often.
        Without env assumption, env can keep e=FALSE, so s AND e never holds.
        With assumption GF(e), env must set e=TRUE sometimes, and sys can
        set s=TRUE at those moments.
        """
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        en = bdd.named_var("e'")
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=['e'], sys_vars=['s'],
            env_live=[e],  # GF(e): env must visit e=TRUE infinitely often
            sys_live=[bdd.AND(s, e)]  # GF(s AND e): both TRUE infinitely often
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE


# ===== Section 6: Arbiter Synthesis =====

class TestArbiterSynthesis:
    def test_arbiter_2_clients(self):
        result = synthesize_arbiter(n_clients=2)
        assert result.result == SynthResult.REALIZABLE
        assert result.winning_region is not None

    def test_arbiter_has_strategy(self):
        result = synthesize_arbiter(n_clients=2)
        assert result.strategy_bdd is not None

    def test_arbiter_3_clients(self):
        result = synthesize_arbiter(n_clients=3)
        assert result.result == SynthResult.REALIZABLE


# ===== Section 7: Traffic Light Synthesis =====

class TestTrafficLightSynthesis:
    def test_traffic_light_realizable(self):
        result = synthesize_traffic_light()
        assert result.result == SynthResult.REALIZABLE

    def test_traffic_light_has_strategy(self):
        result = synthesize_traffic_light()
        assert result.strategy_bdd is not None

    def test_traffic_light_winning_region(self):
        result = synthesize_traffic_light()
        assert result.winning_region is not None


# ===== Section 8: make_gr1_game API =====

class TestMakeGR1Game:
    def test_basic_game(self):
        bdd, spec = make_gr1_game(
            env_vars=['e'],
            sys_vars=['s'],
            sys_live_fns=[lambda bdd, c, n: c['s']]
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_game_with_safety(self):
        bdd, spec = make_gr1_game(
            env_vars=['e'],
            sys_vars=['s'],
            sys_safe_fn=lambda bdd, c, n: bdd.TRUE,
            sys_live_fns=[lambda bdd, c, n: c['s']]
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_game_with_init(self):
        bdd, spec = make_gr1_game(
            env_vars=['e'],
            sys_vars=['s'],
            sys_init_fn=lambda bdd, c, n: bdd.NOT(c['s']),
            sys_live_fns=[lambda bdd, c, n: c['s']]
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE


# ===== Section 9: Counterstrategy Extraction =====

class TestCounterstrategy:
    def test_counterstrategy_on_unrealizable(self):
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_safe=bdd.NOT(sn),
            sys_init=bdd.NOT(s),
            sys_live=[s]
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.UNREALIZABLE
        cs = extract_counterstrategy(bdd, spec, result)
        assert cs is not None
        assert 'losing_region' in cs
        assert 'losing_state_count' in cs

    def test_counterstrategy_on_realizable(self):
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        result = gr1_synthesis(bdd, spec)
        cs = extract_counterstrategy(bdd, spec, result)
        assert cs is None


# ===== Section 10: Strategy Simulation =====

class TestSimulation:
    def test_simulate_trivial(self):
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'], sys_init=bdd.NOT(s))
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE
        # No env trace needed for sys-only
        trace = simulate_strategy(bdd, spec, result, [{}]*5)
        assert len(trace) >= 1  # at least initial state

    def test_simulate_unrealizable_returns_empty(self):
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_safe=bdd.FALSE
        )
        result = gr1_synthesis(bdd, spec)
        trace = simulate_strategy(bdd, spec, result, [{}]*5)
        assert trace == []


# ===== Section 11: Verification of Controllers =====

class TestVerification:
    def test_verify_arbiter(self):
        result = synthesize_arbiter(n_clients=2)
        # We need the bdd to verify -- reconstruct
        bdd = BDD()
        env_vars = ['req_0', 'req_1']
        sys_vars = ['grant_0', 'grant_1']
        curr = {}
        nxt = {}
        for v in env_vars + sys_vars:
            curr[v] = bdd.named_var(v)
            nxt[v] = bdd.named_var(v + "'")

        spec = GR1Spec(env_vars=env_vars, sys_vars=sys_vars)
        spec.sys_init = bdd.AND(bdd.NOT(curr['grant_0']), bdd.NOT(curr['grant_1']))
        spec.sys_safe = bdd.NOT(bdd.AND(nxt['grant_0'], nxt['grant_1']))
        no_spurious = bdd.AND(
            bdd.OR(bdd.NOT(nxt['grant_0']), curr['req_0']),
            bdd.OR(bdd.NOT(nxt['grant_1']), curr['req_1'])
        )
        spec.sys_safe = bdd.AND(spec.sys_safe, no_spurious)
        spec.sys_live = [
            bdd.OR(bdd.NOT(curr['req_0']), curr['grant_0']),
            bdd.OR(bdd.NOT(curr['req_1']), curr['grant_1']),
        ]
        spec.env_live = [
            bdd.OR(bdd.NOT(curr['req_0']), bdd.NOT(curr['grant_0'])),
            bdd.OR(bdd.NOT(curr['req_1']), bdd.NOT(curr['grant_1'])),
        ]

        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE
        checks = verify_controller(bdd, spec, result)
        assert checks['init_in_winning']

    def test_verify_unrealizable(self):
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'], sys_safe=bdd.FALSE)
        result = gr1_synthesis(bdd, spec)
        checks = verify_controller(bdd, spec, result)
        assert not checks['verified']


# ===== Section 12: Check Realizability API =====

class TestCheckRealizability:
    def test_realizable(self):
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        assert check_realizability(bdd, spec) is True

    def test_unrealizable(self):
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_safe=bdd.NOT(sn),
            sys_init=bdd.NOT(s),
            sys_live=[s]
        )
        assert check_realizability(bdd, spec) is False


# ===== Section 13: Explicit to GR1 Conversion =====

class TestExplicitToGR1:
    def test_simple_explicit_game(self):
        # 2-state game: {s=F} -> {s=T} -> {s=F} cycle
        transitions = [
            ({'s': False}, {'s': True}),
            ({'s': True}, {'s': False}),
        ]
        bdd, spec = explicit_to_gr1(
            states=['s0', 's1'],
            env_vars=[], sys_vars=['s'],
            transitions=transitions,
            init_state={'s': False},
            sys_live_states=[[{'s': True}]]  # GF(s=True)
        )
        result = gr1_synthesis(bdd, spec)
        # Realizable because the cycle visits s=True
        assert result.result == SynthResult.REALIZABLE

    def test_explicit_with_env(self):
        # Environment controls 'e', system controls 's'
        transitions = [
            ({'e': False, 's': False}, {'e': False, 's': True}),
            ({'e': False, 's': False}, {'e': True, 's': False}),
            ({'e': True, 's': False}, {'e': False, 's': True}),
            ({'e': False, 's': True}, {'e': False, 's': False}),
            ({'e': True, 's': True}, {'e': False, 's': False}),
        ]
        bdd, spec = explicit_to_gr1(
            states=['s0', 's1', 's2', 's3'],
            env_vars=['e'], sys_vars=['s'],
            transitions=transitions,
            init_state={'e': False, 's': False}
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result in [SynthResult.REALIZABLE, SynthResult.UNREALIZABLE]


# ===== Section 14: Compare Synthesis Approaches =====

class TestCompare:
    def test_compare_all(self):
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_live=[s]
        )
        results = compare_synthesis_approaches(bdd, spec, target=s, acceptance=s)
        assert 'safety' in results
        assert 'gr1' in results
        assert 'reachability' in results
        assert 'buchi' in results
        assert results['gr1']['result'] == 'realizable'

    def test_compare_safety_only(self):
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        results = compare_synthesis_approaches(bdd, spec)
        assert 'safety' in results
        assert 'gr1' in results
        assert results['safety']['result'] == 'realizable'


# ===== Section 15: Mealy Machine Extraction =====

class TestMealyMachine:
    def test_extract_from_trivial(self):
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE
        if result.strategy:
            mm = extract_mealy_machine(bdd, spec, result)
            if mm is not None:
                assert len(mm.states) >= 1

    def test_extract_none_on_unrealizable(self):
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'], sys_safe=bdd.FALSE)
        result = gr1_synthesis(bdd, spec)
        mm = extract_mealy_machine(bdd, spec, result)
        assert mm is None

    def test_mealy_step(self):
        mm = MealyMachine(
            states=[{'s': False}, {'s': True}],
            initial=0,
            transitions={0: {(): 1}, 1: {(): 0}},
            outputs={0: {(): {'s': True}}, 1: {(): {'s': False}}}
        )
        next_idx, output = mm.step(0, {})
        assert next_idx == 1
        assert output == {'s': True}
        next_idx2, output2 = mm.step(1, {})
        assert next_idx2 == 0


# ===== Section 16: Multi-Variable Games =====

class TestMultiVariable:
    def test_two_sys_vars(self):
        """System controls two variables, must toggle both."""
        bdd = BDD()
        a = bdd.named_var('a')
        b = bdd.named_var('b')
        bdd.named_var("a'")
        bdd.named_var("b'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['a', 'b'],
            sys_live=[a, b]  # GF(a) AND GF(b)
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_two_env_one_sys(self):
        """Two env vars, one sys var. System must track env activity."""
        bdd = BDD()
        e1 = bdd.named_var('e1')
        e2 = bdd.named_var('e2')
        s = bdd.named_var('s')
        e1n = bdd.named_var("e1'")
        e2n = bdd.named_var("e2'")
        sn = bdd.named_var("s'")
        # System freely controls s, no safety coupling to env
        # With env assumptions GF(e1 OR e2), system can satisfy GF(s)
        spec = GR1Spec(
            env_vars=['e1', 'e2'], sys_vars=['s'],
            env_live=[bdd.OR(e1, e2)],  # GF(e1 OR e2)
            sys_live=[s]  # GF(s)
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE


# ===== Section 17: Edge Cases =====

class TestEdgeCases:
    def test_empty_liveness(self):
        """No liveness conditions => pure safety game."""
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'])
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_single_state(self):
        """Single boolean variable, initial is TRUE, must stay TRUE."""
        bdd = BDD()
        s = bdd.named_var('s')
        sn = bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_init=s,
            sys_safe=sn,  # must always be TRUE
            sys_live=[s]
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_contradictory_init(self):
        """Init = FALSE => unrealizable (no initial states)."""
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=[], sys_vars=['s'],
            sys_init=bdd.FALSE
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.UNREALIZABLE

    def test_env_init_restricts(self):
        """Env init restricts initial states."""
        bdd = BDD()
        e = bdd.named_var('e')
        s = bdd.named_var('s')
        bdd.named_var("e'")
        bdd.named_var("s'")
        spec = GR1Spec(
            env_vars=['e'], sys_vars=['s'],
            env_init=e,  # env starts TRUE
            sys_init=bdd.NOT(s),  # sys starts FALSE
        )
        result = gr1_synthesis(bdd, spec)
        assert result.result == SynthResult.REALIZABLE

    def test_statistics_populated(self):
        """Synthesis returns iteration statistics."""
        bdd = BDD()
        s = bdd.named_var('s')
        bdd.named_var("s'")
        spec = GR1Spec(env_vars=[], sys_vars=['s'], sys_live=[s])
        result = gr1_synthesis(bdd, spec)
        assert 'outer_iterations' in result.statistics
        assert result.statistics['outer_iterations'] >= 1
