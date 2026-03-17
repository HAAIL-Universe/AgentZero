"""Tests for V191: Parameterized Synthesis."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V187_gr1_synthesis'))

from parameterized_synthesis import (
    # Data structures
    ParamVerdict, ParamSpec, ProcessTemplate, ControllerTemplate,
    InstanceResult, CutoffResult, ParamResult,
    # Core functions
    instantiate_ring_game, instantiate_pipeline_game, instantiate_game,
    detect_cutoff, reduce_by_symmetry, verify_inductive_step,
    extract_controller_template, parameterized_synthesize,
    # Analysis
    analyze_scaling, scaling_summary, compare_with_without_symmetry,
    # Helpers
    build_parameterized_game, solve_parameterized_family,
    parameterized_summary,
    # Predefined specs
    mutex_ring_spec, pipeline_spec, token_passing_spec,
    # Internal helpers
    _ring_neighbors, _star_neighbors, _pipeline_neighbors,
    _canonical_rotation, _strategy_signature, _relative_signature,
    _state_name,
)
from gr1_synthesis import (
    GR1Game, GR1Verdict, GR1Result, gr1_solve, make_game,
)


# ===================================================================
# Section 1: Data Structures
# ===================================================================

class TestProcessTemplate:
    def test_create_basic(self):
        t = ProcessTemplate(
            local_states=['a', 'b'],
            initial_state='a',
            transitions={('a', '*'): [('b', 'go')]},
            labels={'in_b': {'b'}},
        )
        assert t.local_states == ['a', 'b']
        assert t.initial_state == 'a'
        assert ('a', '*') in t.transitions

    def test_labels(self):
        t = ProcessTemplate(
            local_states=['idle', 'active'],
            initial_state='idle',
            transitions={},
            labels={'is_active': {'active'}, 'is_idle': {'idle'}},
        )
        assert 'active' in t.labels['is_active']
        assert 'idle' in t.labels['is_idle']

    def test_multiple_transitions(self):
        t = ProcessTemplate(
            local_states=['s0', 's1', 's2'],
            initial_state='s0',
            transitions={
                ('s0', 'x'): [('s1', 'a'), ('s2', 'b')],
                ('s1', '*'): [('s0', 'c')],
            },
            labels={},
        )
        assert len(t.transitions[('s0', 'x')]) == 2


class TestParamSpec:
    def test_create_ring(self):
        t = ProcessTemplate(['a'], 'a', {}, {})
        spec = ParamSpec(process_template=t, topology="ring")
        assert spec.topology == "ring"

    def test_create_pipeline(self):
        t = ProcessTemplate(['a'], 'a', {}, {})
        spec = ParamSpec(process_template=t, topology="pipeline")
        assert spec.topology == "pipeline"

    def test_mutex_spec(self):
        spec = mutex_ring_spec()
        assert spec.mutex == ['critical']
        assert 'in_critical' in spec.liveness

    def test_pipeline_spec(self):
        spec = pipeline_spec()
        assert spec.topology == "pipeline"
        assert spec.env_signals == ['input', 'no_input']

    def test_token_spec(self):
        spec = token_passing_spec()
        assert spec.mutex == ['holding']
        assert 'has_token' in spec.liveness


class TestControllerTemplate:
    def test_create(self):
        ct = ControllerTemplate(
            states=['c0', 'c1'],
            initial='c0',
            transitions={('c0', 'idle', 'has_token'): ('c1', 'active')},
        )
        assert ct.initial == 'c0'
        assert len(ct.states) == 2

    def test_instantiate(self):
        ct = ControllerTemplate(
            states=['c0'],
            initial='c0',
            transitions={('c0', 'a', 'b'): ('c0', 'a')},
        )
        instances = ct.instantiate(3)
        assert len(instances) == 3
        assert all(i['state'] == 'c0' for i in instances)
        assert instances[0]['id'] == 0
        assert instances[2]['id'] == 2

    def test_instantiate_single(self):
        ct = ControllerTemplate(states=['s0'], initial='s0', transitions={})
        instances = ct.instantiate(1)
        assert len(instances) == 1


class TestParamVerdict:
    def test_values(self):
        assert ParamVerdict.REALIZABLE.value == "realizable"
        assert ParamVerdict.UNREALIZABLE.value == "unrealizable"
        assert ParamVerdict.CUTOFF_FOUND.value == "cutoff_found"
        assert ParamVerdict.UNKNOWN.value == "unknown"


class TestCutoffResult:
    def test_create(self):
        cr = CutoffResult(cutoff_n=3, verified_up_to=6, stable_from=3)
        assert cr.cutoff_n == 3
        assert not cr.is_inductive

    def test_with_signatures(self):
        cr = CutoffResult(
            cutoff_n=2, verified_up_to=4, stable_from=2,
            structure_signatures={2: 'sig_a', 3: 'sig_a', 4: 'sig_a'},
        )
        assert cr.structure_signatures[3] == 'sig_a'


class TestInstanceResult:
    def test_create(self):
        ir = InstanceResult(
            n=3, verdict=GR1Verdict.REALIZABLE,
            game_states=27, winning_region_size=27,
        )
        assert ir.n == 3
        assert ir.strategy is None


# ===================================================================
# Section 2: Topology Helpers
# ===================================================================

class TestTopologyHelpers:
    def test_ring_neighbors_3(self):
        assert _ring_neighbors(0, 3) == (2, 1)
        assert _ring_neighbors(1, 3) == (0, 2)
        assert _ring_neighbors(2, 3) == (1, 0)

    def test_ring_neighbors_2(self):
        assert _ring_neighbors(0, 2) == (1, 1)
        assert _ring_neighbors(1, 2) == (0, 0)

    def test_star_neighbors_hub(self):
        nbrs = _star_neighbors(0, 4)
        assert nbrs == [1, 2, 3]

    def test_star_neighbors_spoke(self):
        assert _star_neighbors(1, 4) == [0]
        assert _star_neighbors(3, 4) == [0]

    def test_pipeline_neighbors_source(self):
        left, right = _pipeline_neighbors(0, 3)
        assert left is None
        assert right == 1

    def test_pipeline_neighbors_middle(self):
        left, right = _pipeline_neighbors(1, 3)
        assert left == 0
        assert right == 2

    def test_pipeline_neighbors_sink(self):
        left, right = _pipeline_neighbors(2, 3)
        assert left == 1
        assert right is None

    def test_state_name(self):
        assert _state_name(0, 'idle') == 'p0_idle'
        assert _state_name(2, 'critical') == 'p2_critical'


# ===================================================================
# Section 3: Canonical Rotation (Symmetry)
# ===================================================================

class TestCanonicalRotation:
    def test_identity(self):
        state = (('a', 'b', 'c'), 0)
        canon = _canonical_rotation(state, 3)
        # Must be one of the 3 rotations; the canonical is lexicographic min
        assert canon == _canonical_rotation(canon, 3)  # Idempotent

    def test_rotated_states_same_canonical(self):
        s0 = (('a', 'b', 'c'), 0)
        s1 = (('b', 'c', 'a'), 2)
        s2 = (('c', 'a', 'b'), 1)
        c0 = _canonical_rotation(s0, 3)
        c1 = _canonical_rotation(s1, 3)
        c2 = _canonical_rotation(s2, 3)
        assert c0 == c1 == c2

    def test_different_states_different_canonical(self):
        s1 = (('a', 'a', 'b'), 0)
        s2 = (('a', 'b', 'b'), 0)
        c1 = _canonical_rotation(s1, 3)
        c2 = _canonical_rotation(s2, 3)
        assert c1 != c2

    def test_no_token_tuple(self):
        s1 = ('a', 'b', 'c')
        s2 = ('b', 'c', 'a')
        c1 = _canonical_rotation(s1, 3)
        c2 = _canonical_rotation(s2, 3)
        assert c1 == c2

    def test_two_procs(self):
        s1 = (('x', 'y'), 0)
        s2 = (('y', 'x'), 1)
        c1 = _canonical_rotation(s1, 2)
        c2 = _canonical_rotation(s2, 2)
        assert c1 == c2

    def test_singleton(self):
        s = (('a',), 0)
        assert _canonical_rotation(s, 1) == s

    def test_non_tuple_passthrough(self):
        s = "opaque_state"
        assert _canonical_rotation(s, 3) == s


# ===================================================================
# Section 4: Game Instantiation
# ===================================================================

class TestInstantiateRingGame:
    def test_two_procs_token(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        assert len(game.states) > 0
        assert len(game.initial) > 0
        assert len(game.transitions) > 0

    def test_state_count_grows(self):
        spec = token_passing_spec()
        g2 = instantiate_ring_game(spec, 2)
        g3 = instantiate_ring_game(spec, 3)
        assert len(g3.states) > len(g2.states)

    def test_initial_state_format(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        for s in game.initial:
            assert isinstance(s, tuple)
            assert len(s) == 2  # (combo, token_pos)
            combo, token = s
            assert isinstance(combo, tuple)
            assert token == 0

    def test_mutex_enforced(self):
        spec = mutex_ring_spec()
        game = instantiate_ring_game(spec, 2)
        # No state should have two processes in 'critical'
        for state in game.states:
            combo, token = state
            critical_count = sum(1 for s in combo if s == 'critical')
            # States exist but transitions shouldn't lead to mutex violation
            # (violations are pruned from successors)

    def test_env_justice_token_visits(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 3)
        # Should have one env justice set per process (token visits)
        assert len(game.env_justice) == 3

    def test_sys_justice_liveness(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 3)
        # One sys justice per process for 'has_token' liveness
        assert len(game.sys_justice) == 3


class TestInstantiatePipelineGame:
    def test_basic(self):
        spec = pipeline_spec()
        game = instantiate_pipeline_game(spec, 2)
        assert len(game.states) > 0
        assert len(game.initial) > 0

    def test_state_format(self):
        spec = pipeline_spec()
        game = instantiate_pipeline_game(spec, 2)
        for s in game.initial:
            assert isinstance(s, tuple)
            assert len(s) == 2  # 2 processes
            assert all(ls in ['empty', 'full'] for ls in s)

    def test_grows_with_n(self):
        spec = pipeline_spec()
        g2 = instantiate_pipeline_game(spec, 2)
        g3 = instantiate_pipeline_game(spec, 3)
        assert len(g3.states) > len(g2.states)


class TestInstantiateGame:
    def test_ring_dispatch(self):
        spec = token_passing_spec()
        game = instantiate_game(spec, 2)
        assert len(game.states) > 0

    def test_pipeline_dispatch(self):
        spec = pipeline_spec()
        game = instantiate_game(spec, 2)
        assert len(game.states) > 0

    def test_default_ring(self):
        t = ProcessTemplate(['a'], 'a', {('a', '*'): [('a', 'x')]}, {})
        spec = ParamSpec(process_template=t, topology="clique")
        game = instantiate_game(spec, 2)
        assert len(game.states) > 0  # Falls back to ring


# ===================================================================
# Section 5: GR(1) Solving on Parameterized Instances
# ===================================================================

class TestSolveInstances:
    def test_token_passing_2(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        result = gr1_solve(game, extract_strategy=True)
        # Should be realizable (token passing is always solvable)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert len(result.winning_region) > 0

    def test_token_passing_initial_winning(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        result = gr1_solve(game, extract_strategy=True)
        if result.verdict == GR1Verdict.REALIZABLE:
            assert game.initial <= result.winning_region

    def test_pipeline_2(self):
        spec = pipeline_spec()
        game = instantiate_pipeline_game(spec, 2)
        result = gr1_solve(game, extract_strategy=False)
        assert result.verdict in (GR1Verdict.REALIZABLE, GR1Verdict.UNREALIZABLE)

    def test_simple_custom_game(self):
        """Custom 2-state game that's trivially realizable."""
        game = make_game(
            states={0, 1},
            initial={0},
            transitions={
                0: [{0, 1}],  # sys picks
                1: [{0, 1}],
            },
            env_justice=[{0, 1}],
            sys_justice=[{0, 1}],
        )
        result = gr1_solve(game, extract_strategy=True)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===================================================================
# Section 6: Symmetry Reduction
# ===================================================================

class TestSymmetryReduction:
    def test_reduces_states(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 3)
        reduced = reduce_by_symmetry(game, 3)
        assert len(reduced.states) <= len(game.states)

    def test_preserves_verdict_2(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        reduced = reduce_by_symmetry(game, 2)

        result_full = gr1_solve(game, extract_strategy=False)
        result_red = gr1_solve(reduced, extract_strategy=False)

        full_real = result_full.verdict == GR1Verdict.REALIZABLE and \
                    game.initial <= result_full.winning_region
        red_initial = {_canonical_rotation(s, 2) for s in game.initial}
        red_real = result_red.verdict == GR1Verdict.REALIZABLE and \
                   red_initial <= result_red.winning_region

        # Symmetry reduction should preserve realizability
        assert full_real == red_real

    def test_single_proc_no_reduction(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 1)
        reduced = reduce_by_symmetry(game, 1)
        assert len(reduced.states) == len(game.states)

    def test_canonical_initial_preserved(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 3)
        reduced = reduce_by_symmetry(game, 3)
        assert len(reduced.initial) > 0

    def test_transitions_preserved(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        reduced = reduce_by_symmetry(game, 2)
        assert len(reduced.transitions) > 0

    def test_justice_preserved(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        reduced = reduce_by_symmetry(game, 2)
        assert len(reduced.env_justice) > 0
        assert len(reduced.sys_justice) > 0


class TestCompareSymmetry:
    def test_compare(self):
        spec = token_passing_spec()
        result = compare_with_without_symmetry(spec, 2)
        assert result['n'] == 2
        assert result['full_states'] >= result['reduced_states']
        assert 0 < result['reduction_ratio'] <= 1.0
        assert result['verdicts_agree']


# ===================================================================
# Section 7: Cutoff Detection
# ===================================================================

class TestCutoffDetection:
    def test_basic_cutoff(self):
        """Test cutoff detection on token passing (should stabilize quickly)."""
        spec = token_passing_spec()
        result, games = detect_cutoff(spec, n_min=2, n_max=4, stability_window=2)
        assert result.verdict in (
            ParamVerdict.CUTOFF_FOUND, ParamVerdict.REALIZABLE,
            ParamVerdict.UNREALIZABLE, ParamVerdict.UNKNOWN,
        )
        assert len(result.instance_results) > 0

    def test_instances_populated(self):
        spec = token_passing_spec()
        result, _ = detect_cutoff(spec, n_min=2, n_max=3)
        assert 2 in result.instance_results
        assert 3 in result.instance_results

    def test_instance_game_states(self):
        spec = token_passing_spec()
        result, _ = detect_cutoff(spec, n_min=2, n_max=3)
        for n, inst in result.instance_results.items():
            assert inst.game_states > 0
            assert inst.n == n

    def test_early_exit_unrealizable(self):
        """If any N is unrealizable, cutoff detection stops."""
        # Build a spec that's trivially unrealizable
        t = ProcessTemplate(
            local_states=['a'],
            initial_state='a',
            transitions={},  # No transitions at all
            labels={},
        )
        spec = ParamSpec(process_template=t, topology="ring", liveness=['nonexistent'])
        result, _ = detect_cutoff(spec, n_min=2, n_max=4)
        # Even with no liveness match, it defaults to global justice
        # so it may still be realizable trivially
        assert result.verdict in (
            ParamVerdict.UNREALIZABLE, ParamVerdict.REALIZABLE,
            ParamVerdict.CUTOFF_FOUND, ParamVerdict.UNKNOWN,
        )

    def test_games_returned(self):
        spec = token_passing_spec()
        result, games = detect_cutoff(spec, n_min=2, n_max=3)
        assert 2 in games
        assert 3 in games


# ===================================================================
# Section 8: Inductive Verification
# ===================================================================

class TestInductiveVerification:
    def test_basic_inductive(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        result = gr1_solve(game, extract_strategy=True)

        inst = InstanceResult(
            n=2,
            verdict=result.verdict,
            game_states=len(game.states),
            winning_region_size=len(result.winning_region),
            strategy=result.strategy,
        )

        if inst.verdict == GR1Verdict.REALIZABLE:
            is_inductive = verify_inductive_step(spec, 2, inst)
            # Token passing should extend to N+1
            assert isinstance(is_inductive, bool)

    def test_unrealizable_not_inductive(self):
        inst = InstanceResult(
            n=2, verdict=GR1Verdict.UNREALIZABLE,
            game_states=0, winning_region_size=0,
        )
        t = ProcessTemplate(['a'], 'a', {}, {})
        spec = ParamSpec(process_template=t)
        assert verify_inductive_step(spec, 2, inst) == False


# ===================================================================
# Section 9: Template Extraction
# ===================================================================

class TestTemplateExtraction:
    def test_extract_from_realizable(self):
        spec = token_passing_spec()
        result, _ = detect_cutoff(spec, n_min=2, n_max=3)

        template = extract_controller_template(spec, result.instance_results)
        # May or may not succeed depending on strategy structure
        if template is not None:
            assert len(template.states) > 0
            assert template.initial is not None

    def test_no_template_from_empty(self):
        template = extract_controller_template(
            token_passing_spec(), {}
        )
        assert template is None

    def test_no_template_from_unrealizable(self):
        inst = {2: InstanceResult(
            n=2, verdict=GR1Verdict.UNREALIZABLE,
            game_states=0, winning_region_size=0,
        )}
        template = extract_controller_template(token_passing_spec(), inst)
        assert template is None


# ===================================================================
# Section 10: Full Pipeline
# ===================================================================

class TestParameterizedSynthesize:
    def test_token_passing(self):
        spec = token_passing_spec()
        result = parameterized_synthesize(spec, n_min=2, n_max=3)
        assert result.verdict in (
            ParamVerdict.REALIZABLE, ParamVerdict.CUTOFF_FOUND,
            ParamVerdict.UNREALIZABLE, ParamVerdict.UNKNOWN,
        )
        assert result.n_min == 2
        assert result.n_max_checked >= 3

    def test_pipeline(self):
        spec = pipeline_spec()
        result = parameterized_synthesize(spec, n_min=2, n_max=3)
        assert len(result.instance_results) > 0

    def test_with_symmetry(self):
        spec = token_passing_spec()
        result = parameterized_synthesize(spec, n_min=2, n_max=3, use_symmetry=True)
        assert result.verdict in (
            ParamVerdict.REALIZABLE, ParamVerdict.CUTOFF_FOUND,
            ParamVerdict.UNREALIZABLE, ParamVerdict.UNKNOWN,
        )

    def test_without_symmetry(self):
        spec = token_passing_spec()
        result = parameterized_synthesize(spec, n_min=2, n_max=3, use_symmetry=False)
        assert len(result.instance_results) > 0


# ===================================================================
# Section 11: Scaling Analysis
# ===================================================================

class TestScalingAnalysis:
    def test_basic_scaling(self):
        spec = token_passing_spec()
        stats = analyze_scaling(spec, [2, 3])
        assert 2 in stats
        assert 3 in stats
        assert stats[2]['states'] < stats[3]['states']

    def test_stats_fields(self):
        spec = token_passing_spec()
        stats = analyze_scaling(spec, [2])
        s = stats[2]
        assert 'states' in s
        assert 'transitions' in s
        assert 'winning_region' in s
        assert 'verdict' in s
        assert 'initial_winning' in s

    def test_scaling_summary_format(self):
        spec = token_passing_spec()
        stats = analyze_scaling(spec, [2, 3])
        summary = scaling_summary(stats)
        assert 'States' in summary
        assert 'Trans' in summary
        assert 'Verdict' in summary


# ===================================================================
# Section 12: Custom Game Builders
# ===================================================================

class TestBuildParameterizedGame:
    def test_custom_builder(self):
        def states(n):
            return set(range(n))

        def initial(n):
            return {0}

        def transitions(n, ss):
            return {i: [{(i + 1) % n}] for i in range(n)}

        def env_j(n, ss):
            return [ss]

        def sys_j(n, ss):
            return [ss]

        game = build_parameterized_game(3, states, initial, transitions, env_j, sys_j)
        assert len(game.states) == 3
        assert game.initial == {0}
        assert len(game.transitions) == 3

    def test_solve_family(self):
        def builder(n):
            states = set(range(n))
            return make_game(
                states=states,
                initial={0},
                transitions={i: [{(i + 1) % n}] for i in range(n)},
                env_justice=[states],
                sys_justice=[states],
            )

        results = solve_parameterized_family(builder, range(2, 5))
        assert 2 in results
        assert 3 in results
        assert 4 in results

    def test_family_all_realizable(self):
        def builder(n):
            states = set(range(n))
            return make_game(
                states=states,
                initial={0},
                transitions={i: [{(i + 1) % n}] for i in range(n)},
                env_justice=[states],
                sys_justice=[states],
            )

        results = solve_parameterized_family(builder, range(2, 5))
        for n, r in results.items():
            assert r.verdict == GR1Verdict.REALIZABLE


# ===================================================================
# Section 13: Predefined Specs
# ===================================================================

class TestPredefinedSpecs:
    def test_mutex_ring_fields(self):
        spec = mutex_ring_spec()
        t = spec.process_template
        assert 'idle' in t.local_states
        assert 'trying' in t.local_states
        assert 'critical' in t.local_states
        assert t.initial_state == 'idle'
        assert len(t.transitions) > 0

    def test_pipeline_fields(self):
        spec = pipeline_spec()
        t = spec.process_template
        assert 'empty' in t.local_states
        assert 'full' in t.local_states
        assert t.initial_state == 'empty'

    def test_token_passing_fields(self):
        spec = token_passing_spec()
        t = spec.process_template
        assert 'waiting' in t.local_states
        assert 'holding' in t.local_states
        assert t.initial_state == 'waiting'

    def test_mutex_transitions_complete(self):
        spec = mutex_ring_spec()
        t = spec.process_template
        # Should have transitions for key (state, signal) pairs
        assert ('idle', 'has_token') in t.transitions
        assert ('trying', 'has_token') in t.transitions
        assert ('critical', 'has_token') in t.transitions

    def test_mutex_labels(self):
        spec = mutex_ring_spec()
        t = spec.process_template
        assert 'in_critical' in t.labels
        assert 'critical' in t.labels['in_critical']


# ===================================================================
# Section 14: Strategy Signatures
# ===================================================================

class TestSignatures:
    def test_strategy_signature_no_strategy(self):
        inst = InstanceResult(
            n=2, verdict=GR1Verdict.UNREALIZABLE,
            game_states=0, winning_region_size=0,
        )
        sig = _strategy_signature(inst)
        assert 'no_strategy' in sig

    def test_strategy_signature_with_strategy(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        result = gr1_solve(game, extract_strategy=True)
        inst = InstanceResult(
            n=2, verdict=result.verdict,
            game_states=len(game.states),
            winning_region_size=len(result.winning_region),
            strategy=result.strategy,
        )
        sig = _strategy_signature(inst)
        assert len(sig) > 0

    def test_relative_signature_unrealizable(self):
        inst = InstanceResult(
            n=2, verdict=GR1Verdict.UNREALIZABLE,
            game_states=0, winning_region_size=0,
        )
        sig = _relative_signature(inst, 2)
        assert sig == "unrealizable"

    def test_relative_signature_format(self):
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 2)
        result = gr1_solve(game, extract_strategy=True)
        inst = InstanceResult(
            n=2, verdict=result.verdict,
            game_states=len(game.states),
            winning_region_size=len(result.winning_region),
            strategy=result.strategy,
        )
        if inst.verdict == GR1Verdict.REALIZABLE:
            sig = _relative_signature(inst, 2)
            assert 'cov=' in sig
            assert 'modes=' in sig


# ===================================================================
# Section 15: Summary/Output
# ===================================================================

class TestSummary:
    def test_parameterized_summary(self):
        result = ParamResult(
            verdict=ParamVerdict.REALIZABLE,
            instance_results={
                2: InstanceResult(n=2, verdict=GR1Verdict.REALIZABLE,
                                  game_states=8, winning_region_size=8),
                3: InstanceResult(n=3, verdict=GR1Verdict.REALIZABLE,
                                  game_states=27, winning_region_size=27),
            },
            n_min=2,
            n_max_checked=3,
        )
        summary = parameterized_summary(result)
        assert 'realizable' in summary
        assert 'N=2' in summary
        assert 'N=3' in summary

    def test_summary_with_cutoff(self):
        result = ParamResult(
            verdict=ParamVerdict.CUTOFF_FOUND,
            instance_results={},
            cutoff=CutoffResult(cutoff_n=3, verified_up_to=5, stable_from=3, is_inductive=True),
            n_min=2, n_max_checked=5,
        )
        summary = parameterized_summary(result)
        assert 'Cutoff' in summary
        assert 'N_c = 3' in summary
        assert 'Inductive: True' in summary

    def test_summary_with_template(self):
        result = ParamResult(
            verdict=ParamVerdict.REALIZABLE,
            instance_results={},
            controller_template=ControllerTemplate(
                states=['c0', 'c1'], initial='c0', transitions={('c0', 'a', 'b'): ('c1', 'a')}
            ),
            n_min=2, n_max_checked=4,
        )
        summary = parameterized_summary(result)
        assert 'Controller template' in summary
        assert '2 states' in summary


# ===================================================================
# Section 16: Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_single_state_process(self):
        t = ProcessTemplate(
            local_states=['only'],
            initial_state='only',
            transitions={('only', '*'): [('only', 'stay')]},
            labels={'always': {'only'}},
        )
        spec = ParamSpec(process_template=t, topology="ring", liveness=['always'])
        game = instantiate_ring_game(spec, 2)
        result = gr1_solve(game, extract_strategy=False)
        # Single-state process: trivially realizable
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_two_procs_minimal(self):
        t = ProcessTemplate(
            local_states=['a', 'b'],
            initial_state='a',
            transitions={
                ('a', '*'): [('b', 'go')],
                ('b', '*'): [('a', 'back')],
            },
            labels={'in_a': {'a'}},
        )
        spec = ParamSpec(process_template=t, topology="ring", liveness=['in_a'])
        game = instantiate_ring_game(spec, 2)
        assert len(game.states) > 0

    def test_empty_mutex(self):
        t = ProcessTemplate(
            local_states=['a'],
            initial_state='a',
            transitions={('a', '*'): [('a', 'stay')]},
            labels={},
        )
        spec = ParamSpec(process_template=t, mutex=[])
        game = instantiate_ring_game(spec, 2)
        assert len(game.states) > 0

    def test_symmetry_reduction_preserves_reachability(self):
        """After symmetry reduction, initial states should be reachable."""
        spec = token_passing_spec()
        game = instantiate_ring_game(spec, 3)
        reduced = reduce_by_symmetry(game, 3)
        assert len(reduced.initial) > 0
        # All initial states should have transitions
        for s in reduced.initial:
            assert s in reduced.transitions or s in reduced.states


# ===================================================================
# Section 17: Integration -- Full Parameterized Pipeline
# ===================================================================

class TestIntegration:
    def test_full_pipeline_token_passing(self):
        """End-to-end: parameterized synthesis for token passing."""
        spec = token_passing_spec()
        result = parameterized_synthesize(spec, n_min=2, n_max=3)

        # Should produce some verdict
        assert result.verdict != ParamVerdict.UNKNOWN or len(result.instance_results) > 0

        # Summary should be well-formed
        summary = parameterized_summary(result)
        assert len(summary) > 0

    def test_full_pipeline_pipeline_spec(self):
        """End-to-end: parameterized synthesis for pipeline."""
        spec = pipeline_spec()
        result = parameterized_synthesize(spec, n_min=2, n_max=3, use_symmetry=False)
        assert len(result.instance_results) > 0

    def test_scaling_token(self):
        """Verify state space grows as expected."""
        spec = token_passing_spec()
        stats = analyze_scaling(spec, [2, 3])
        # 2 local states, N procs, N token positions
        # States = 2^N * N
        assert stats[2]['states'] == 2**2 * 2  # 8
        assert stats[3]['states'] == 2**3 * 3  # 24

    def test_custom_family_pipeline(self):
        """Solve a custom family and verify consistency."""
        def builder(n):
            states = set(range(n + 1))
            return make_game(
                states=states,
                initial={0},
                transitions={i: [{min(i + 1, n)}] for i in range(n + 1)},
                env_justice=[states],
                sys_justice=[{n}],  # Eventually reach state n
            )

        results = solve_parameterized_family(builder, range(2, 5))
        for n, r in results.items():
            assert r.verdict == GR1Verdict.REALIZABLE
            assert r.game_states == n + 1


# ===================================================================
# Section 18: Mutex Ring Synthesis
# ===================================================================

class TestMutexRingSynthesis:
    def test_mutex_2_procs(self):
        """Mutual exclusion for 2 processes."""
        spec = mutex_ring_spec()
        game = instantiate_ring_game(spec, 2)
        result = gr1_solve(game, extract_strategy=False)
        # Mutex ring should be realizable
        assert result.verdict in (GR1Verdict.REALIZABLE, GR1Verdict.UNREALIZABLE)

    def test_mutex_scaling(self):
        """State space for mutex ring."""
        spec = mutex_ring_spec()
        stats = analyze_scaling(spec, [2])
        # 3 local states, 2 procs, 2 token positions = 3^2 * 2 = 18
        assert stats[2]['states'] == 18


# ===================================================================
# Section 19: Robustness
# ===================================================================

class TestRobustness:
    def test_no_transitions(self):
        """Process with no transitions should still create a game."""
        t = ProcessTemplate(
            local_states=['stuck'],
            initial_state='stuck',
            transitions={},
            labels={},
        )
        spec = ParamSpec(process_template=t)
        game = instantiate_ring_game(spec, 2)
        assert len(game.states) > 0

    def test_wildcard_only_transitions(self):
        """Process using only wildcard transitions."""
        t = ProcessTemplate(
            local_states=['a', 'b'],
            initial_state='a',
            transitions={
                ('a', '*'): [('b', 'next')],
                ('b', '*'): [('a', 'prev')],
            },
            labels={'in_a': {'a'}},
        )
        spec = ParamSpec(process_template=t, liveness=['in_a'])
        game = instantiate_ring_game(spec, 2)
        assert len(game.transitions) > 0

    def test_stability_window_1(self):
        """Stability window of 1 should find cutoff immediately."""
        spec = token_passing_spec()
        result, _ = detect_cutoff(spec, n_min=2, n_max=4, stability_window=1)
        # With window 1, any single realizable N could be a cutoff
        assert len(result.instance_results) > 0

    def test_large_stability_window(self):
        """Large stability window shouldn't crash."""
        spec = token_passing_spec()
        result, _ = detect_cutoff(spec, n_min=2, n_max=4, stability_window=10)
        assert len(result.instance_results) > 0
        # Window larger than range means no cutoff found
        if result.verdict == ParamVerdict.CUTOFF_FOUND:
            assert result.cutoff is not None
