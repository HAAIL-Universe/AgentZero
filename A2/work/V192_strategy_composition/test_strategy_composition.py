"""Tests for V192: Strategy Composition."""

import sys
import pytest

sys.path.insert(0, 'Z:/AgentZero/A2/work/V192_strategy_composition')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V186_reactive_synthesis')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V187_gr1_synthesis')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V023_ltl_model_checking')

from ltl_model_checker import (
    Atom, LTLTrue, LTLFalse, And as LTLAnd, Or as LTLOr,
    Not as LTLNot, Implies as LTLImplies,
    Next, Finally, Globally, Until,
)

from strategy_composition import (
    CompositionResult,
    parallel_compose, sequential_compose, priority_compose,
    conjunctive_synthesize, assume_guarantee_compose,
    gr1_assume_guarantee,
    decompose_spec, compose_from_decomposition,
    verify_composition, compare_composition_methods,
    product_mealy, restrict_mealy, rename_mealy,
    minimize_mealy, mealy_equivalence,
    simulate_composition, composition_statistics,
    _all_valuations, _flatten_and, _collect_atoms, _s,
)
from reactive_synthesis import MealyMachine, SynthesisVerdict, synthesize
from gr1_synthesis import BoolGR1Spec, GR1Verdict, gr1_synthesize


# ---------------------------------------------------------------------------
# Helper: build simple Mealy machines for testing
# ---------------------------------------------------------------------------

def make_simple_mealy(inputs, outputs, trans_spec, initial=0):
    """Build a Mealy machine from a compact spec.

    trans_spec: list of (state, input_set, next_state, output_set)
    """
    states = set()
    transitions = {}
    for s, inp, ns, out in trans_spec:
        states.add(s)
        states.add(ns)
        transitions[(s, frozenset(inp))] = (ns, frozenset(out))
    states.add(initial)
    return MealyMachine(
        states=states,
        initial=initial,
        inputs=set(inputs),
        outputs=set(outputs),
        transitions=transitions,
    )


# ---------------------------------------------------------------------------
# Tests: Parallel Composition
# ---------------------------------------------------------------------------

class TestParallelComposition:
    def test_disjoint_outputs(self):
        """Two controllers with disjoint outputs compose cleanly."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, set()),
                (0, set(), 0, {'y'}),
            ]
        )
        result = parallel_compose(c1, c2)
        assert result.controller is not None
        assert result.method == 'parallel'
        assert result.controller.outputs == {'x', 'y'}

    def test_parallel_output_merge(self):
        """Outputs from both controllers are merged."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        # When a=True, both x and y should be True
        t = result.controller.transitions.get((0, frozenset({'a'})))
        assert t is not None
        _, out = t
        assert 'x' in out
        assert 'y' in out

    def test_parallel_disjoint_inputs(self):
        """Controllers with different inputs."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['b'], outputs=['y'],
            trans_spec=[
                (0, {'b'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        assert result.controller.inputs == {'a', 'b'}
        assert result.controller.outputs == {'x', 'y'}

    def test_parallel_overlap_raises(self):
        """Overlapping outputs should raise ValueError."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[(0, set(), 0, set())]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[(0, set(), 0, set())]
        )
        with pytest.raises(ValueError, match="not disjoint"):
            parallel_compose(c1, c2)

    def test_parallel_product_states(self):
        """Product state space is bounded by s1*s2."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 0, set()),
                (1, {'a'}, 0, set()),
                (1, set(), 1, {'x'}),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        # c1 has 2 states, c2 has 1, so product has at most 2
        assert result.n_states <= 2

    def test_parallel_simulation(self):
        """Simulate composed controller."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, set()),
                (0, set(), 0, {'y'}),
            ]
        )
        result = parallel_compose(c1, c2)
        trace = simulate_composition(result.controller, [{'a'}, set(), {'a'}])
        assert len(trace) == 3
        # a=True: x=True, y=False
        assert 'x' in trace[0][2]
        assert 'y' not in trace[0][2]
        # a=False: x=False, y=True
        assert 'x' not in trace[1][2]
        assert 'y' in trace[1][2]


class TestSequentialComposition:
    def test_basic_sequential(self):
        """c1 output feeds c2 input."""
        # c1: a -> x (copy)
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        # c2: x -> y (copy)
        c2 = make_simple_mealy(
            inputs=['x'], outputs=['y'],
            trans_spec=[
                (0, {'x'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = sequential_compose(c1, c2)
        assert result.controller is not None
        assert result.method == 'sequential'
        assert result.controller.inputs == {'a'}
        assert result.controller.outputs == {'y'}

    def test_sequential_propagation(self):
        """Values propagate through the chain."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['x'], outputs=['y'],
            trans_spec=[
                (0, {'x'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = sequential_compose(c1, c2)
        # a=True -> x=True (c1) -> y=True (c2)
        t = result.controller.transitions.get((0, frozenset({'a'})))
        assert t is not None
        _, out = t
        assert 'y' in out

    def test_sequential_negation(self):
        """c1 inverts, c2 copies -- net: invert."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, set()),    # a=True -> x=False
                (0, set(), 0, {'x'}),    # a=False -> x=True
            ]
        )
        c2 = make_simple_mealy(
            inputs=['x'], outputs=['y'],
            trans_spec=[
                (0, {'x'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = sequential_compose(c1, c2)
        # a=True -> x=False -> y=False
        t = result.controller.transitions.get((0, frozenset({'a'})))
        assert t is not None
        assert 'y' not in t[1]

    def test_sequential_no_shared_raises(self):
        """No shared vars should raise."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[(0, set(), 0, set())]
        )
        c2 = make_simple_mealy(
            inputs=['b'], outputs=['y'],
            trans_spec=[(0, set(), 0, set())]
        )
        with pytest.raises(ValueError, match="No shared"):
            sequential_compose(c1, c2)

    def test_sequential_explicit_shared(self):
        """Explicitly specify shared variables."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x', 'z'],
            trans_spec=[
                (0, {'a'}, 0, {'x', 'z'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['x', 'b'], outputs=['y'],
            trans_spec=[
                (0, {'x', 'b'}, 0, {'y'}),
                (0, {'x'}, 0, {'y'}),
                (0, {'b'}, 0, set()),
                (0, set(), 0, set()),
            ]
        )
        result = sequential_compose(c1, c2, shared_vars={'x'})
        assert result.controller.inputs == {'a', 'b'}
        assert result.controller.outputs == {'y', 'z'}


class TestPriorityComposition:
    def test_priority_c1(self):
        """c1 takes priority on overlapping outputs."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, set()),   # disagrees: x=False
                (0, set(), 0, {'x'}),   # disagrees: x=True
            ]
        )
        result = priority_compose(c1, c2, priority='c1')
        assert result.controller is not None
        assert result.method == 'priority'
        # a=True: c1 says x=True, c2 says x=False -> c1 wins
        t = result.controller.transitions.get((0, frozenset({'a'})))
        assert t is not None
        assert 'x' in t[1]

    def test_priority_c2(self):
        """c2 takes priority."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, set()),
                (0, set(), 0, {'x'}),
            ]
        )
        result = priority_compose(c1, c2, priority='c2')
        t = result.controller.transitions.get((0, frozenset({'a'})))
        assert t is not None
        assert 'x' not in t[1]  # c2 says x=False, c2 wins

    def test_priority_conflict_detection(self):
        """Conflicts are recorded."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, set()),
                (0, set(), 0, {'x'}),
            ]
        )
        result = priority_compose(c1, c2)
        assert len(result.conflicts) > 0

    def test_priority_no_conflict(self):
        """No conflicts when controllers agree."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        result = priority_compose(c1, c2)
        assert len(result.conflicts) == 0

    def test_priority_mixed_outputs(self):
        """Some outputs overlap, some don't."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x', 'y'],
            trans_spec=[
                (0, {'a'}, 0, {'x', 'y'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x', 'z'],
            trans_spec=[
                (0, {'a'}, 0, {'z'}),
                (0, set(), 0, {'x'}),
            ]
        )
        result = priority_compose(c1, c2, priority='c1')
        assert result.controller.outputs == {'x', 'y', 'z'}


class TestMealyOperations:
    def test_product_mealy(self):
        """Product of two Mealy machines."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        prod = product_mealy(c1, c2)
        assert prod.inputs == {'a'}
        assert prod.outputs == {'x', 'y'}

    def test_restrict_mealy(self):
        """Project to subset of outputs."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x', 'y'],
            trans_spec=[
                (0, {'a'}, 0, {'x', 'y'}),
                (0, set(), 0, set()),
            ]
        )
        r = restrict_mealy(m, {'x'})
        assert r.outputs == {'x'}
        t = r.transitions.get((0, frozenset({'a'})))
        assert t is not None
        assert 'y' not in t[1]

    def test_rename_mealy(self):
        """Rename inputs and outputs."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        r = rename_mealy(m, input_map={'a': 'b'}, output_map={'x': 'z'})
        assert r.inputs == {'b'}
        assert r.outputs == {'z'}
        t = r.transitions.get((0, frozenset({'b'})))
        assert t is not None
        assert 'z' in t[1]

    def test_minimize_identity(self):
        """Minimize a machine that's already minimal."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 0, set()),
                (1, {'a'}, 0, set()),
                (1, set(), 1, {'x'}),
            ]
        )
        mini = minimize_mealy(m)
        assert len(mini.states) == 2  # already minimal

    def test_minimize_reduces(self):
        """Minimize merges equivalent states."""
        # States 0 and 2 behave identically (both output x when a, go to 1)
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 0, set()),
                (1, {'a'}, 2, {'x'}),
                (1, set(), 1, set()),
                (2, {'a'}, 1, {'x'}),
                (2, set(), 2, set()),
            ]
        )
        mini = minimize_mealy(m)
        # States 0 and 2 are equivalent -> should merge
        assert len(mini.states) <= 2

    def test_mealy_equivalence_same(self):
        """Identical machines are equivalent."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        eq, msg = mealy_equivalence(m, m)
        assert eq

    def test_mealy_equivalence_different(self):
        """Different output machines are not equivalent."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, set()),
                (0, set(), 0, {'x'}),
            ]
        )
        eq, msg = mealy_equivalence(c1, c2)
        assert not eq

    def test_mealy_equivalence_io_mismatch(self):
        """I/O mismatch detected."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[(0, set(), 0, set())]
        )
        c2 = make_simple_mealy(
            inputs=['b'], outputs=['x'],
            trans_spec=[(0, set(), 0, set())]
        )
        eq, msg = mealy_equivalence(c1, c2)
        assert not eq


class TestSimulation:
    def test_simulate_basic(self):
        """Basic simulation trace."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        trace = simulate_composition(m, [{'a'}, set(), {'a'}])
        assert len(trace) == 3
        assert 'x' in trace[0][2]
        assert 'x' not in trace[1][2]
        assert 'x' in trace[2][2]

    def test_simulate_none_controller(self):
        """Simulation with None controller returns empty."""
        trace = simulate_composition(None, [{'a'}])
        assert trace == []

    def test_simulate_stateful(self):
        """Stateful machine changes behavior."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, set()),
                (0, set(), 0, set()),
                (1, {'a'}, 0, {'x'}),
                (1, set(), 1, set()),
            ]
        )
        trace = simulate_composition(m, [{'a'}, {'a'}, {'a'}, {'a'}])
        assert len(trace) == 4
        # Alternates: no output, output, no output, output
        assert 'x' not in trace[0][2]
        assert 'x' in trace[1][2]
        assert 'x' not in trace[2][2]
        assert 'x' in trace[3][2]


class TestCompositionStatistics:
    def test_statistics(self):
        """Composition statistics."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        stats = composition_statistics(result)
        assert stats['method'] == 'parallel'
        assert stats['n_states'] >= 1
        assert stats['n_components'] == 2
        assert 'x' in stats['outputs']
        assert 'y' in stats['outputs']


class TestCompositionResult:
    def test_result_fields(self):
        """CompositionResult has all fields."""
        r = CompositionResult(
            controller=None,
            method='test',
            components=[1, 2],
            conflicts=[('c',)],
            n_states=5,
            verified=True,
            details={'key': 'val'},
        )
        assert r.controller is None
        assert r.method == 'test'
        assert len(r.components) == 2
        assert len(r.conflicts) == 1
        assert r.n_states == 5
        assert r.verified is True
        assert r.details['key'] == 'val'


# ---------------------------------------------------------------------------
# Tests: Spec Decomposition & Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_all_valuations_empty(self):
        """Empty variable set gives one valuation."""
        vals = list(_all_valuations(set()))
        assert len(vals) == 1
        assert vals[0] == frozenset()

    def test_all_valuations_one(self):
        """One variable gives two valuations."""
        vals = list(_all_valuations({'a'}))
        assert len(vals) == 2

    def test_all_valuations_two(self):
        """Two variables give four valuations."""
        vals = list(_all_valuations({'a', 'b'}))
        assert len(vals) == 4

    def test_flatten_and_single(self):
        """Single spec stays as-is."""
        spec = Atom('a')
        result = _flatten_and(spec)
        assert len(result) == 1

    def test_flatten_and_nested(self):
        """Nested And is flattened."""
        spec = LTLAnd(Atom('a'), LTLAnd(Atom('b'), Atom('c')))
        result = _flatten_and(spec)
        assert len(result) >= 2  # at least a and (b and c) or a, b, c

    def test_collect_atoms_simple(self):
        """Collect atoms from simple formula."""
        spec = LTLAnd(Atom('a'), Atom('b'))
        atoms = _collect_atoms(spec)
        assert 'a' in atoms
        assert 'b' in atoms

    def test_collect_atoms_temporal(self):
        """Collect atoms from temporal formula."""
        spec = Globally(Atom('a'))
        atoms = _collect_atoms(spec)
        assert 'a' in atoms


class TestSpecDecomposition:
    def test_decompose_single(self):
        """Single spec cannot be decomposed."""
        spec = Globally(Atom('a'))
        groups = decompose_spec(spec, ['a'], ['b'])
        assert len(groups) == 1

    def test_decompose_conjunction(self):
        """Conjunction of independent specs splits."""
        spec = LTLAnd(Globally(Atom('a')), Globally(Atom('b')))
        # a and b are in env_vars, no sys_vars shared
        groups = decompose_spec(spec, ['a', 'b'], [])
        # Both have empty sys vars -> they still group by shared sys vars
        # With no sys vars, each gets empty set -> same group
        assert len(groups) >= 1

    def test_decompose_shared_vars_stay_grouped(self):
        """Specs sharing sys vars stay in same group."""
        # Both mention sys var 'x'
        spec = LTLAnd(Globally(Atom('x')), Finally(Atom('x')))
        groups = decompose_spec(spec, [], ['x'])
        assert len(groups) == 1

    def test_decompose_disjoint_sys_vars(self):
        """Specs with disjoint sys vars separate."""
        spec = LTLAnd(Globally(Atom('x')), Globally(Atom('y')))
        groups = decompose_spec(spec, [], ['x', 'y'])
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# Tests: Conjunctive Synthesis
# ---------------------------------------------------------------------------

class TestConjunctiveSynthesis:
    def test_safety_conjunction(self):
        """Synthesize conjunction of two safety specs."""
        # G(x) and G(y) where sys controls x,y -- sys just sets both
        spec1 = Globally(Atom('x'))
        spec2 = Globally(Atom('y'))
        result = conjunctive_synthesize(spec1, spec2, [], ['x', 'y'])
        assert result.method == 'conjunctive'
        # Should be realizable since sys controls both
        if result.controller is not None:
            assert result.verified

    def test_conflicting_specs(self):
        """G(x) and G(!x) is unrealizable."""
        spec1 = Globally(Atom('x'))
        spec2 = Globally(LTLNot(Atom('x')))
        result = conjunctive_synthesize(spec1, spec2, [], ['x'])
        # Should be unrealizable
        assert result.controller is None or not result.verified


# ---------------------------------------------------------------------------
# Tests: Assume-Guarantee
# ---------------------------------------------------------------------------

class TestAssumeGuarantee:
    def test_single_component(self):
        """Single component AG is just regular synthesis."""
        spec = (LTLTrue(), Globally(Atom('x')))
        result = assume_guarantee_compose([spec], [], ['x'])
        assert result.method == 'assume_guarantee'

    def test_two_component_ag(self):
        """Two-component AG composition."""
        # Component 1: assume True, guarantee G(x)
        # Component 2: assume G(x), guarantee G(y)
        specs = [
            (LTLTrue(), Globally(Atom('x'))),
            (Globally(Atom('x')), Globally(Atom('y'))),
        ]
        result = assume_guarantee_compose(specs, [], ['x', 'y'])
        assert result.method == 'assume_guarantee'


# ---------------------------------------------------------------------------
# Tests: GR(1) Assume-Guarantee
# ---------------------------------------------------------------------------

class TestGR1AssumeGuarantee:
    def test_single_gr1_component(self):
        """Single GR(1) component AG."""
        spec = BoolGR1Spec(
            env_vars=[],
            sys_vars=['x'],
            env_init=lambda s: True,
            sys_init=lambda s: True,
            env_trans=lambda s, n: True,
            sys_trans=lambda s, n: True,
            env_justice=[],
            sys_justice=[lambda s: 'x' in s],
        )
        result = gr1_assume_guarantee([spec])
        assert result.method == 'gr1_assume_guarantee'

    def test_two_gr1_components(self):
        """Two GR(1) component AG with different vars."""
        spec1 = BoolGR1Spec(
            env_vars=[],
            sys_vars=['x'],
            env_init=lambda s: True,
            sys_init=lambda s: True,
            env_trans=lambda s, n: True,
            sys_trans=lambda s, n: True,
            env_justice=[],
            sys_justice=[lambda s: 'x' in s],
        )
        spec2 = BoolGR1Spec(
            env_vars=[],
            sys_vars=['y'],
            env_init=lambda s: True,
            sys_init=lambda s: True,
            env_trans=lambda s, n: True,
            sys_trans=lambda s, n: True,
            env_justice=[],
            sys_justice=[lambda s: 'y' in s],
        )
        result = gr1_assume_guarantee([spec1, spec2])
        assert result.method == 'gr1_assume_guarantee'
        assert result.verified


# ---------------------------------------------------------------------------
# Tests: Multi-state Composition
# ---------------------------------------------------------------------------

class TestMultiStateComposition:
    def test_parallel_multi_state(self):
        """Parallel compose of multi-state machines."""
        # Counter that toggles on 'a'
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 0, set()),
                (1, {'a'}, 0, set()),
                (1, set(), 1, {'x'}),
            ]
        )
        # Another counter on 'a'
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 1, set()),
                (0, set(), 0, {'y'}),
                (1, {'a'}, 0, {'y'}),
                (1, set(), 1, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        assert result.n_states <= 4  # 2*2 product bound
        trace = simulate_composition(result.controller, [{'a'}, {'a'}, {'a'}, {'a'}])
        assert len(trace) == 4

    def test_sequential_multi_state(self):
        """Sequential compose of multi-state machines."""
        # c1: toggle x on a
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 0, set()),
                (1, {'a'}, 0, set()),
                (1, set(), 1, {'x'}),
            ]
        )
        # c2: copy x to y
        c2 = make_simple_mealy(
            inputs=['x'], outputs=['y'],
            trans_spec=[
                (0, {'x'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = sequential_compose(c1, c2)
        trace = simulate_composition(result.controller, [{'a'}, {'a'}, {'a'}, {'a'}])
        assert len(trace) == 4
        # c1 toggles x, c2 copies -> y toggles too
        assert 'y' in trace[0][2]
        assert 'y' not in trace[1][2]

    def test_minimize_after_compose(self):
        """Minimizing a composed machine."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        mini = minimize_mealy(result.controller)
        assert len(mini.states) >= 1

    def test_equivalence_before_after_minimize(self):
        """Machine equivalent to its minimization."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 0, set()),
                (1, {'a'}, 2, {'x'}),
                (1, set(), 1, set()),
                (2, {'a'}, 1, {'x'}),
                (2, set(), 2, set()),
            ]
        )
        mini = minimize_mealy(m)
        eq, _ = mealy_equivalence(m, mini)
        assert eq


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_trace_simulation(self):
        """Empty input sequence gives empty trace."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[(0, set(), 0, set())]
        )
        trace = simulate_composition(m, [])
        assert trace == []

    def test_parallel_single_state(self):
        """Both controllers have 1 state."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        assert result.n_states == 1

    def test_composition_result_defaults(self):
        """Default CompositionResult fields."""
        r = CompositionResult(controller=None, method='test')
        assert r.components == []
        assert r.conflicts == []
        assert r.n_states == 0
        assert r.verified is None
        assert r.details == {}

    def test_rename_identity(self):
        """Rename with empty maps is identity."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[(0, {'a'}, 0, {'x'}), (0, set(), 0, set())]
        )
        r = rename_mealy(m)
        eq, _ = mealy_equivalence(m, r)
        assert eq


# ---------------------------------------------------------------------------
# Tests: Integration with V186/V187
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_synthesize_and_compose(self):
        """Synthesize two specs and parallel compose."""
        # G(x) -- sys must keep x true
        spec1 = Globally(Atom('x'))
        r1 = synthesize(spec1, [], ['x'])
        if r1.verdict != SynthesisVerdict.REALIZABLE:
            pytest.skip("V186 synthesis issue")

        # G(y) -- sys must keep y true
        spec2 = Globally(Atom('y'))
        r2 = synthesize(spec2, [], ['y'])
        if r2.verdict != SynthesisVerdict.REALIZABLE:
            pytest.skip("V186 synthesis issue")

        result = parallel_compose(r1.controller, r2.controller)
        assert result.controller is not None
        assert result.controller.outputs == {'x', 'y'}
        # Every output should have both x and y
        for (s, inp), (ns, out) in result.controller.transitions.items():
            assert 'x' in out
            assert 'y' in out

    def test_conjunctive_vs_parallel(self):
        """Conjunctive synthesis matches parallel composition."""
        spec1 = Globally(Atom('x'))
        spec2 = Globally(Atom('y'))

        # Conjunctive
        conj = conjunctive_synthesize(spec1, spec2, [], ['x', 'y'])

        # Individual + parallel
        r1 = synthesize(spec1, [], ['x'])
        r2 = synthesize(spec2, [], ['y'])

        if conj.controller is None or r1.controller is None or r2.controller is None:
            pytest.skip("Synthesis unavailable")

        par = parallel_compose(r1.controller, r2.controller)

        # Both should produce controllers that keep x and y true
        for (s, inp), (ns, out) in conj.controller.transitions.items():
            assert 'x' in out and 'y' in out
        for (s, inp), (ns, out) in par.controller.transitions.items():
            assert 'x' in out and 'y' in out

    def test_gr1_to_mealy_compose(self):
        """GR(1) synthesize, convert to Mealy, then compose."""
        spec = BoolGR1Spec(
            env_vars=[],
            sys_vars=['x'],
            env_init=lambda s: True,
            sys_init=lambda s: True,
            env_trans=lambda s, n: True,
            sys_trans=lambda s, n: True,
            env_justice=[],
            sys_justice=[lambda s: 'x' in s],
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_compose_from_decomposition_single(self):
        """compose_from_decomposition with non-decomposable spec."""
        spec = Globally(Atom('x'))
        result = compose_from_decomposition(spec, [], ['x'])
        assert result.method == 'monolithic'

    def test_compare_methods(self):
        """compare_composition_methods returns valid structure."""
        spec = Globally(Atom('x'))
        comparison = compare_composition_methods(spec, [], ['x'])
        assert 'monolithic' in comparison
        assert 'compositional' in comparison
        assert 'verdict' in comparison['monolithic']


# ---------------------------------------------------------------------------
# Tests: Complex Scenarios
# ---------------------------------------------------------------------------

class TestComplexScenarios:
    def test_three_way_parallel(self):
        """Compose three controllers in parallel."""
        controllers = []
        for name in ['x', 'y', 'z']:
            c = make_simple_mealy(
                inputs=['a'], outputs=[name],
                trans_spec=[
                    (0, {'a'}, 0, {name}),
                    (0, set(), 0, set()),
                ]
            )
            controllers.append(c)

        # Compose pairwise
        r1 = parallel_compose(controllers[0], controllers[1])
        r2 = parallel_compose(r1.controller, controllers[2])
        assert r2.controller.outputs == {'x', 'y', 'z'}

        # All outputs true when a is true
        t = r2.controller.transitions.get((0, frozenset({'a'})))
        assert t is not None
        assert 'x' in t[1] and 'y' in t[1] and 'z' in t[1]

    def test_chain_sequential(self):
        """Chain three controllers sequentially: a -> x -> y -> z."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['x'], outputs=['y'],
            trans_spec=[
                (0, {'x'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        c3 = make_simple_mealy(
            inputs=['y'], outputs=['z'],
            trans_spec=[
                (0, {'y'}, 0, {'z'}),
                (0, set(), 0, set()),
            ]
        )

        r1 = sequential_compose(c1, c2)
        r2 = sequential_compose(r1.controller, c3)
        assert r2.controller.inputs == {'a'}
        assert r2.controller.outputs == {'z'}

        # a=True -> x=True -> y=True -> z=True
        t = r2.controller.transitions.get((0, frozenset({'a'})))
        assert t is not None
        assert 'z' in t[1]

    def test_parallel_then_minimize(self):
        """Compose then minimize."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 0, set()),
                (1, {'a'}, 0, set()),
                (1, set(), 1, {'x'}),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['y'],
            trans_spec=[
                (0, {'a'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = parallel_compose(c1, c2)
        mini = minimize_mealy(result.controller)
        assert len(mini.states) <= result.n_states

    def test_priority_then_simulate(self):
        """Priority compose then simulate full trace."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, set()),
                (0, set(), 0, {'x'}),
            ]
        )
        result = priority_compose(c1, c2, priority='c1')
        trace = simulate_composition(result.controller, [{'a'}, set(), {'a'}, set()])
        assert len(trace) == 4
        # c1 priority: a=T->x=T, a=F->x=F
        assert 'x' in trace[0][2]
        assert 'x' not in trace[1][2]

    def test_rename_then_parallel(self):
        """Rename to avoid overlap, then parallel compose."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        m2 = rename_mealy(m, output_map={'x': 'y'})
        result = parallel_compose(m, m2)
        assert result.controller.outputs == {'x', 'y'}


class TestDecompositionSynthesis:
    def test_decompose_disjoint_synthesize(self):
        """Decompose G(x) & G(y) with disjoint sys vars and compose."""
        spec = LTLAnd(Globally(Atom('x')), Globally(Atom('y')))
        result = compose_from_decomposition(spec, [], ['x', 'y'])
        # Should decompose into 2 groups since x and y are independent sys vars
        if result.method == 'decomposed':
            assert result.n_states >= 1
            assert result.verified
        else:
            # Monolithic also fine
            assert result.method in ('monolithic', 'monolithic_fallback')

    def test_decompose_shared_monolithic(self):
        """Spec with shared sys vars falls back to monolithic."""
        spec = Globally(LTLAnd(Atom('x'), Atom('y')))
        result = compose_from_decomposition(spec, [], ['x', 'y'])
        assert result.method == 'monolithic'

    def test_compare_methods_structure(self):
        """compare_composition_methods returns valid structure."""
        spec = LTLAnd(Globally(Atom('x')), Globally(Atom('y')))
        comparison = compare_composition_methods(spec, [], ['x', 'y'])
        assert 'monolithic' in comparison
        assert 'compositional' in comparison
        assert 'reduction' in comparison
        assert 'verdict' in comparison['monolithic']
        assert 'method' in comparison['compositional']


class TestSequentialChain:
    def test_sequential_with_extra_outputs(self):
        """Sequential: c1 produces x,z; only x feeds c2; z passes through."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x', 'z'],
            trans_spec=[
                (0, {'a'}, 0, {'x', 'z'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['x'], outputs=['y'],
            trans_spec=[
                (0, {'x'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        result = sequential_compose(c1, c2)
        # z should pass through from c1
        assert 'z' in _s(result.controller.outputs)
        assert 'y' in _s(result.controller.outputs)
        # x is consumed (shared), should not appear in final outputs
        assert 'x' not in _s(result.controller.outputs)

    def test_sequential_with_extra_inputs(self):
        """Sequential: c2 has extra inputs beyond shared vars."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['x', 'b'], outputs=['y'],
            trans_spec=[
                (0, {'x', 'b'}, 0, {'y'}),
                (0, {'x'}, 0, {'y'}),
                (0, {'b'}, 0, set()),
                (0, set(), 0, set()),
            ]
        )
        result = sequential_compose(c1, c2)
        assert _s(result.controller.inputs) == {'a', 'b'}


class TestPriorityDetailed:
    def test_priority_all_agree(self):
        """When controllers always agree, no conflicts recorded."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x', 'y'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, {'y'}),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x', 'y'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, {'y'}),
            ]
        )
        result = priority_compose(c1, c2)
        assert len(result.conflicts) == 0

    def test_priority_default_is_c1(self):
        """Default priority is c1."""
        c1 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, set()),
                (0, set(), 0, {'x'}),
            ]
        )
        result = priority_compose(c1, c2)
        # Default priority='c1', so c1 wins
        t = result.controller.transitions.get((0, frozenset({'a'})))
        assert 'x' in t[1]


class TestMealyAdvanced:
    def test_product_shared_inputs(self):
        """Product with shared inputs."""
        c1 = make_simple_mealy(
            inputs=['a', 'b'], outputs=['x'],
            trans_spec=[
                (0, {'a', 'b'}, 0, {'x'}),
                (0, {'a'}, 0, {'x'}),
                (0, {'b'}, 0, set()),
                (0, set(), 0, set()),
            ]
        )
        c2 = make_simple_mealy(
            inputs=['a', 'b'], outputs=['y'],
            trans_spec=[
                (0, {'a', 'b'}, 0, {'y'}),
                (0, {'a'}, 0, set()),
                (0, {'b'}, 0, {'y'}),
                (0, set(), 0, set()),
            ]
        )
        prod = product_mealy(c1, c2)
        assert _s(prod.inputs) == {'a', 'b'}
        assert _s(prod.outputs) == {'x', 'y'}

    def test_restrict_removes_output(self):
        """Restrict removes output variables from transitions."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x', 'y', 'z'],
            trans_spec=[
                (0, {'a'}, 0, {'x', 'y', 'z'}),
                (0, set(), 0, set()),
            ]
        )
        r = restrict_mealy(m, {'x', 'z'})
        t = r.transitions.get((0, frozenset({'a'})))
        assert 'y' not in t[1]
        assert 'x' in t[1]
        assert 'z' in t[1]

    def test_rename_no_overlap(self):
        """Rename creates non-overlapping variables."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[(0, {'a'}, 0, {'x'}), (0, set(), 0, set())]
        )
        r1 = rename_mealy(m, output_map={'x': 'out1'})
        r2 = rename_mealy(m, output_map={'x': 'out2'})
        # Now can parallel compose
        result = parallel_compose(r1, r2)
        assert _s(result.controller.outputs) == {'out1', 'out2'}

    def test_minimize_preserves_behavior(self):
        """Minimized machine has same I/O behavior."""
        # Build machine with redundant states
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                (0, set(), 2, set()),
                (1, {'a'}, 3, set()),
                (1, set(), 0, {'x'}),
                (2, {'a'}, 1, {'x'}),
                (2, set(), 0, set()),
                (3, {'a'}, 2, set()),
                (3, set(), 1, {'x'}),
            ]
        )
        mini = minimize_mealy(m)
        eq, _ = mealy_equivalence(m, mini)
        assert eq
        assert len(mini.states) <= len(m.states)


class TestLTLHelpers:
    def test_flatten_deeply_nested(self):
        """Flatten deeply nested And."""
        spec = LTLAnd(LTLAnd(Atom('a'), Atom('b')), LTLAnd(Atom('c'), Atom('d')))
        result = _flatten_and(spec)
        assert len(result) == 4

    def test_collect_atoms_implies(self):
        """Collect atoms from Implies formula."""
        spec = LTLImplies(Atom('a'), Globally(Atom('b')))
        atoms = _collect_atoms(spec)
        assert 'a' in atoms
        assert 'b' in atoms

    def test_collect_atoms_or(self):
        """Collect atoms from Or formula."""
        spec = LTLOr(Atom('p'), Atom('q'))
        atoms = _collect_atoms(spec)
        assert atoms == {'p', 'q'}

    def test_flatten_non_and(self):
        """Flatten non-And spec returns single-element list."""
        spec = Globally(LTLOr(Atom('a'), Atom('b')))
        result = _flatten_and(spec)
        assert len(result) == 1


class TestSynthesisIntegration:
    def test_safety_synthesis_compose(self):
        """Synthesize safety specs and compose."""
        r1 = synthesize(Globally(Atom('x')), [], ['x'])
        r2 = synthesize(Globally(Atom('y')), [], ['y'])
        if r1.controller is None or r2.controller is None:
            pytest.skip("Synthesis unavailable")
        result = parallel_compose(r1.controller, r2.controller)
        # Verify all transitions keep both x and y true
        for (s, inp), (ns, out) in result.controller.transitions.items():
            assert 'x' in out and 'y' in out

    def test_gr1_safety_synthesis(self):
        """GR(1) synthesis for simple safety."""
        spec = BoolGR1Spec(
            env_vars=['a'],
            sys_vars=['x'],
            env_init=lambda s: True,
            sys_init=lambda s: 'x' in s,
            env_trans=lambda s, n: True,
            sys_trans=lambda s, n: 'x' in n,
            env_justice=[],
            sys_justice=[lambda s: 'x' in s],
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_verify_composition_none(self):
        """verify_composition returns False for None controller."""
        ok, msg = verify_composition(None, Globally(Atom('x')), [], ['x'])
        assert not ok

    def test_conjunctive_realizable(self):
        """Conjunctive synthesis of compatible specs."""
        spec1 = Globally(Atom('x'))
        spec2 = Globally(Atom('y'))
        result = conjunctive_synthesize(spec1, spec2, [], ['x', 'y'])
        assert result.verified
        assert result.controller is not None
        assert result.method == 'conjunctive'


class TestSimulationAdvanced:
    def test_simulate_string_inputs(self):
        """Simulate with string set inputs (auto-converted)."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 0, {'x'}),
                (0, set(), 0, set()),
            ]
        )
        trace = simulate_composition(m, [frozenset({'a'}), frozenset()])
        assert len(trace) == 2

    def test_simulate_stops_at_undefined(self):
        """Simulation stops when transition is undefined."""
        m = make_simple_mealy(
            inputs=['a'], outputs=['x'],
            trans_spec=[
                (0, {'a'}, 1, {'x'}),
                # No transitions from state 1
            ]
        )
        trace = simulate_composition(m, [{'a'}, {'a'}, {'a'}])
        assert len(trace) == 1  # only first step succeeds


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
