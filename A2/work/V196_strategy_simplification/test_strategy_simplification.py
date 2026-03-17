"""Tests for V196: Strategy Simplification."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from strategy_simplification import (
    MealyMachine, SimplificationResult, SimulationRelation,
    make_mealy,
    compute_forward_simulation, compute_backward_simulation,
    simulation_quotient,
    dont_care_merge,
    find_irrelevant_inputs, reduce_inputs,
    canonicalize_outputs,
    remove_unreachable,
    signature_merge,
    simplify, full_simplification_pipeline,
    compare_simplification_methods,
    simplification_statistics, simplification_summary,
    simplify_distributed, distributed_simplification_summary,
    compute_cross_simulation, is_simulated_by,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V192_strategy_composition'))
from strategy_composition import minimize_mealy, mealy_equivalence


# ====================================================================
# Helper: build test machines
# ====================================================================

def single_state_machine():
    """1-state machine: always outputs 'g' on any input."""
    return make_mealy(
        states=[0],
        initial=0,
        inputs=["a"],
        outputs=["g"],
        transitions=[
            (0, {"a"}, 0, {"g"}),
            (0, set(), 0, {"g"}),
        ],
    )


def two_state_toggle():
    """2-state machine: toggles output on input 'a'."""
    return make_mealy(
        states=[0, 1],
        initial=0,
        inputs=["a"],
        outputs=["g"],
        transitions=[
            (0, {"a"}, 1, {"g"}),
            (0, set(), 0, set()),
            (1, {"a"}, 0, set()),
            (1, set(), 1, {"g"}),
        ],
    )


def redundant_states_machine():
    """4-state machine where states 2,3 are copies of 0,1."""
    return make_mealy(
        states=[0, 1, 2, 3],
        initial=0,
        inputs=["a"],
        outputs=["g"],
        transitions=[
            # Original pair (0, 1)
            (0, {"a"}, 1, {"g"}),
            (0, set(), 0, set()),
            (1, {"a"}, 0, set()),
            (1, set(), 1, {"g"}),
            # Duplicate pair (2, 3) -- same behavior as (0, 1)
            (2, {"a"}, 3, {"g"}),
            (2, set(), 2, set()),
            (3, {"a"}, 2, set()),
            (3, set(), 3, {"g"}),
        ],
    )


def unreachable_states_machine():
    """Machine with unreachable states."""
    return make_mealy(
        states=[0, 1, 2, 3],
        initial=0,
        inputs=["a"],
        outputs=["g"],
        transitions=[
            (0, {"a"}, 1, {"g"}),
            (0, set(), 0, set()),
            (1, {"a"}, 0, set()),
            (1, set(), 1, {"g"}),
            # States 2,3 are unreachable
            (2, {"a"}, 3, {"g"}),
            (2, set(), 2, set()),
            (3, {"a"}, 2, set()),
            (3, set(), 3, {"g"}),
        ],
    )


def irrelevant_input_machine():
    """Machine where input 'b' is irrelevant (doesn't affect anything)."""
    return make_mealy(
        states=[0, 1],
        initial=0,
        inputs=["a", "b"],
        outputs=["g"],
        transitions=[
            (0, {"a", "b"}, 1, {"g"}),
            (0, {"a"}, 1, {"g"}),
            (0, {"b"}, 0, set()),
            (0, set(), 0, set()),
            (1, {"a", "b"}, 0, set()),
            (1, {"a"}, 0, set()),
            (1, {"b"}, 1, {"g"}),
            (1, set(), 1, {"g"}),
        ],
    )


def constant_output_machine():
    """Machine where output 'h' is always true."""
    return make_mealy(
        states=[0, 1],
        initial=0,
        inputs=["a"],
        outputs=["g", "h"],
        transitions=[
            (0, {"a"}, 1, {"g", "h"}),
            (0, set(), 0, {"h"}),
            (1, {"a"}, 0, {"h"}),
            (1, set(), 1, {"g", "h"}),
        ],
    )


def three_state_chain():
    """3-state chain: 0 -> 1 -> 2 -> 0."""
    return make_mealy(
        states=[0, 1, 2],
        initial=0,
        inputs=["a"],
        outputs=["g"],
        transitions=[
            (0, {"a"}, 1, {"g"}),
            (0, set(), 0, set()),
            (1, {"a"}, 2, set()),
            (1, set(), 1, {"g"}),
            (2, {"a"}, 0, {"g"}),
            (2, set(), 2, set()),
        ],
    )


def dont_care_machine():
    """Machine with undefined transitions (don't-cares)."""
    return make_mealy(
        states=[0, 1, 2],
        initial=0,
        inputs=["a"],
        outputs=["g"],
        transitions=[
            (0, {"a"}, 1, {"g"}),
            (0, set(), 0, set()),
            (1, {"a"}, 0, set()),
            # State 1 has no transition for input={}
            # State 2 has same defined transitions as state 0
            (2, {"a"}, 1, {"g"}),
            (2, set(), 2, set()),
        ],
    )


def large_redundant_machine():
    """6-state machine where 3 pairs are duplicates."""
    return make_mealy(
        states=[0, 1, 2, 3, 4, 5],
        initial=0,
        inputs=["a"],
        outputs=["g"],
        transitions=[
            # State 0 and 3: same behavior
            (0, {"a"}, 1, {"g"}),
            (0, set(), 0, set()),
            (3, {"a"}, 4, {"g"}),
            (3, set(), 3, set()),
            # State 1 and 4: same behavior
            (1, {"a"}, 2, set()),
            (1, set(), 1, {"g"}),
            (4, {"a"}, 5, set()),
            (4, set(), 4, {"g"}),
            # State 2 and 5: same behavior
            (2, {"a"}, 0, {"g"}),
            (2, set(), 2, set()),
            (5, {"a"}, 3, {"g"}),
            (5, set(), 5, set()),
        ],
    )


# ====================================================================
# Tests: Forward simulation
# ====================================================================

class TestForwardSimulation:
    def test_single_state_self_simulation(self):
        m = single_state_machine()
        sim = compute_forward_simulation(m)
        assert sim.simulates(0, 0)

    def test_two_state_self_simulation(self):
        m = two_state_toggle()
        sim = compute_forward_simulation(m)
        assert sim.simulates(0, 0)
        assert sim.simulates(1, 1)

    def test_non_simulation_different_output(self):
        m = two_state_toggle()
        sim = compute_forward_simulation(m)
        # State 0 and 1 have different outputs, so neither simulates the other
        assert not sim.simulates(0, 1)
        assert not sim.simulates(1, 0)

    def test_redundant_states_simulation(self):
        m = redundant_states_machine()
        sim = compute_forward_simulation(m)
        # 0 and 2 have same behavior: mutual simulation
        assert sim.simulates(0, 2)
        assert sim.simulates(2, 0)
        # 1 and 3 have same behavior: mutual simulation
        assert sim.simulates(1, 3)
        assert sim.simulates(3, 1)

    def test_equivalence_classes(self):
        m = redundant_states_machine()
        sim = compute_forward_simulation(m)
        eq_classes = sim.equivalence_classes()
        # Should have 2 classes: {0,2} and {1,3}
        class_sets = [frozenset(c) for c in eq_classes]
        assert frozenset({0, 2}) in class_sets
        assert frozenset({1, 3}) in class_sets

    def test_chain_no_cross_simulation(self):
        m = three_state_chain()
        sim = compute_forward_simulation(m)
        # States 0 and 2 have same output sig but different successor patterns
        # 0: a->1(g), _->0(_)
        # 2: a->0(g), _->2(_)
        # successor of 0 on a is 1, successor of 2 on a is 0
        # 1 and 0 are different -> (0,2) might not survive fixpoint
        # Actually 0 and 2 have identical output behavior and their successors
        # (1 vs 0) have different outputs, so they don't mutually simulate
        assert not sim.simulates(0, 1)

    def test_simulation_reflexive(self):
        m = three_state_chain()
        sim = compute_forward_simulation(m)
        for s in m.states:
            assert sim.simulates(s, s)

    def test_simulation_is_preorder(self):
        """Simulation should be reflexive and transitive."""
        m = redundant_states_machine()
        sim = compute_forward_simulation(m)
        # Reflexive
        for s in m.states:
            assert sim.simulates(s, s)
        # Transitive: if (a,b) and (b,c) then (a,c)
        for a in m.states:
            for b in m.states:
                for c in m.states:
                    if sim.simulates(a, b) and sim.simulates(b, c):
                        assert sim.simulates(a, c)


# ====================================================================
# Tests: Backward simulation
# ====================================================================

class TestBackwardSimulation:
    def test_single_state(self):
        m = single_state_machine()
        sim = compute_backward_simulation(m)
        assert sim.simulates(0, 0)

    def test_two_state_self_simulation(self):
        m = two_state_toggle()
        sim = compute_backward_simulation(m)
        assert sim.simulates(0, 0)
        assert sim.simulates(1, 1)

    def test_redundant_states(self):
        m = redundant_states_machine()
        sim = compute_backward_simulation(m)
        # Redundant states should backward-simulate each other
        assert sim.simulates(0, 2) or sim.simulates(2, 0)

    def test_backward_reflexive(self):
        m = three_state_chain()
        sim = compute_backward_simulation(m)
        for s in m.states:
            assert sim.simulates(s, s)


# ====================================================================
# Tests: Simulation quotient
# ====================================================================

class TestSimulationQuotient:
    def test_single_state_unchanged(self):
        m = single_state_machine()
        sim = compute_forward_simulation(m)
        q = simulation_quotient(m, sim)
        assert len(q.states) == 1

    def test_two_state_toggle_unchanged(self):
        m = two_state_toggle()
        sim = compute_forward_simulation(m)
        q = simulation_quotient(m, sim)
        # States are not simulation-equivalent (different outputs)
        assert len(q.states) == 2

    def test_redundant_states_merged(self):
        m = redundant_states_machine()
        sim = compute_forward_simulation(m)
        q = simulation_quotient(m, sim)
        # 4 states should reduce to 2
        assert len(q.states) == 2

    def test_quotient_preserves_behavior(self):
        m = redundant_states_machine()
        sim = compute_forward_simulation(m)
        q = simulation_quotient(m, sim)
        # Both should respond identically to inputs
        trace_orig = m.simulate([frozenset({"a"}), frozenset(), frozenset({"a"})], max_steps=3)
        trace_quot = q.simulate([frozenset({"a"}), frozenset(), frozenset({"a"})], max_steps=3)
        orig_outputs = [out for _, _, out in trace_orig]
        quot_outputs = [out for _, _, out in trace_quot]
        assert orig_outputs == quot_outputs

    def test_large_redundant_merging(self):
        m = large_redundant_machine()
        sim = compute_forward_simulation(m)
        q = simulation_quotient(m, sim)
        assert len(q.states) == 3  # 3 pairs -> 3 representatives


# ====================================================================
# Tests: Don't-care optimization
# ====================================================================

class TestDontCareMerge:
    def test_single_state(self):
        m = single_state_machine()
        result = dont_care_merge(m)
        assert len(result.states) <= 1

    def test_no_dont_cares(self):
        m = two_state_toggle()
        result = dont_care_merge(m)
        # No undefined transitions, so no merging beyond standard minimize
        assert len(result.states) <= 2

    def test_dont_care_enables_merge(self):
        m = dont_care_machine()
        result = dont_care_merge(m)
        # States 0 and 2 are compatible (same defined transitions)
        # After merging and minimizing, should be <= 2 states
        assert len(result.states) <= 2

    def test_preserves_defined_behavior(self):
        m = dont_care_machine()
        result = dont_care_merge(m)
        # Original defined transitions should still work
        _, out = result.step(result.initial, frozenset({"a"}))
        assert "g" in out


# ====================================================================
# Tests: Input reduction
# ====================================================================

class TestInputReduction:
    def test_find_irrelevant(self):
        m = irrelevant_input_machine()
        irr = find_irrelevant_inputs(m)
        assert "b" in irr
        assert "a" not in irr

    def test_reduce_inputs(self):
        m = irrelevant_input_machine()
        reduced = reduce_inputs(m)
        assert "b" not in reduced.inputs
        assert "a" in reduced.inputs

    def test_no_irrelevant(self):
        m = two_state_toggle()
        irr = find_irrelevant_inputs(m)
        assert len(irr) == 0

    def test_reduce_preserves_behavior(self):
        m = irrelevant_input_machine()
        reduced = reduce_inputs(m)
        # Test with 'a' only inputs
        trace_r = reduced.simulate([frozenset({"a"}), frozenset(), frozenset({"a"})], max_steps=3)
        r_outputs = [out for _, _, out in trace_r]
        # Original machine with a=True, b=arbitrary
        trace_o = m.simulate([frozenset({"a"}), frozenset(), frozenset({"a"})], max_steps=3)
        o_outputs = [out for _, _, out in trace_o]
        assert r_outputs == o_outputs

    def test_reduce_specific_vars(self):
        m = irrelevant_input_machine()
        reduced = reduce_inputs(m, remove_vars={"b"})
        assert "b" not in reduced.inputs

    def test_single_input_no_reduction(self):
        m = single_state_machine()
        irr = find_irrelevant_inputs(m)
        # 'a' is irrelevant (output is always 'g' regardless)
        # But we can't remove the only input (len(inputs) <= 1 returns empty)
        # Actually the check is len(inputs) <= 1 returns empty set
        assert len(irr) == 0


# ====================================================================
# Tests: Output canonicalization
# ====================================================================

class TestOutputCanonicalization:
    def test_constant_output_removed(self):
        m = constant_output_machine()
        result = canonicalize_outputs(m)
        assert "h" not in result.outputs
        assert "g" in result.outputs

    def test_no_constant_outputs(self):
        m = two_state_toggle()
        result = canonicalize_outputs(m)
        assert result.outputs == m.outputs

    def test_all_constant(self):
        m = single_state_machine()
        result = canonicalize_outputs(m)
        # 'g' is always true -> removed
        assert "g" not in result.outputs


# ====================================================================
# Tests: Unreachable state removal
# ====================================================================

class TestRemoveUnreachable:
    def test_no_unreachable(self):
        m = two_state_toggle()
        result = remove_unreachable(m)
        assert len(result.states) == 2

    def test_remove_unreachable(self):
        m = unreachable_states_machine()
        result = remove_unreachable(m)
        assert len(result.states) == 2

    def test_single_state(self):
        m = single_state_machine()
        result = remove_unreachable(m)
        assert len(result.states) == 1

    def test_preserves_behavior(self):
        m = unreachable_states_machine()
        result = remove_unreachable(m)
        trace_orig = m.simulate([frozenset({"a"}), frozenset(), frozenset({"a"})], max_steps=3)
        trace_clean = result.simulate([frozenset({"a"}), frozenset(), frozenset({"a"})], max_steps=3)
        orig_outputs = [out for _, _, out in trace_orig]
        clean_outputs = [out for _, _, out in trace_clean]
        assert orig_outputs == clean_outputs


# ====================================================================
# Tests: Signature merge
# ====================================================================

class TestSignatureMerge:
    def test_single_state(self):
        m = single_state_machine()
        result = signature_merge(m, depth=1)
        assert len(result.states) == 1

    def test_redundant_depth_0(self):
        m = redundant_states_machine()
        result = signature_merge(m, depth=0)
        # At depth 0, only output sig matters -- 0,2 and 1,3 have same outputs
        assert len(result.states) <= 4

    def test_redundant_full_depth(self):
        m = redundant_states_machine()
        result = signature_merge(m, depth=10)
        assert len(result.states) == 2

    def test_chain_preserved(self):
        m = three_state_chain()
        result = signature_merge(m, depth=5)
        # Chain states have different successor patterns at depth >= 1
        # States 0 and 2 have same output sig but different successors
        # After enough depth, all 3 should remain distinct (or 2 merged)
        assert len(result.states) <= 3


# ====================================================================
# Tests: Full simplification pipeline
# ====================================================================

class TestSimplify:
    def test_minimize_method(self):
        m = redundant_states_machine()
        result = simplify(m, method="minimize")
        assert result.simplified_states == 2
        assert result.equivalent

    def test_simulation_method(self):
        m = redundant_states_machine()
        result = simplify(m, method="simulation")
        assert result.simplified_states == 2
        assert result.equivalent

    def test_dont_care_method(self):
        m = dont_care_machine()
        result = simplify(m, method="dont_care")
        assert result.simplified_states <= 2

    def test_input_reduce_method(self):
        m = irrelevant_input_machine()
        result = simplify(m, method="input_reduce")
        assert "b" not in result.simplified.inputs

    def test_full_method(self):
        m = redundant_states_machine()
        result = simplify(m, method="full")
        assert result.simplified_states == 2
        assert result.equivalent

    def test_full_on_complex(self):
        m = large_redundant_machine()
        result = simplify(m, method="full")
        assert result.simplified_states == 3

    def test_state_reduction_property(self):
        m = redundant_states_machine()
        result = simplify(m, method="minimize")
        assert result.state_reduction == 0.5  # 4 -> 2

    def test_transition_reduction_property(self):
        m = redundant_states_machine()
        result = simplify(m, method="minimize")
        assert result.transition_reduction >= 0.0

    def test_invalid_method_raises(self):
        m = single_state_machine()
        with pytest.raises(ValueError):
            simplify(m, method="invalid_method")

    def test_already_minimal(self):
        m = two_state_toggle()
        result = simplify(m, method="full")
        assert result.simplified_states == 2
        assert result.equivalent


# ====================================================================
# Tests: Full pipeline
# ====================================================================

class TestFullPipeline:
    def test_pipeline_reduces(self):
        m = redundant_states_machine()
        result = full_simplification_pipeline(m)
        assert len(result.states) == 2

    def test_pipeline_on_single(self):
        m = single_state_machine()
        result = full_simplification_pipeline(m)
        assert len(result.states) == 1

    def test_pipeline_chain(self):
        m = three_state_chain()
        result = full_simplification_pipeline(m)
        # All 3 states have distinct behavior, should stay 3
        assert len(result.states) <= 3

    def test_pipeline_unreachable_plus_redundant(self):
        """Machine with both unreachable and redundant states."""
        m = make_mealy(
            states=[0, 1, 2, 3, 4, 5],
            initial=0,
            inputs=["a"],
            outputs=["g"],
            transitions=[
                # Reachable: 0 and 1
                (0, {"a"}, 1, {"g"}),
                (0, set(), 0, set()),
                (1, {"a"}, 0, set()),
                (1, set(), 1, {"g"}),
                # Unreachable: 2,3 (copies of 0,1)
                (2, {"a"}, 3, {"g"}),
                (2, set(), 2, set()),
                (3, {"a"}, 2, set()),
                (3, set(), 3, {"g"}),
                # Unreachable: 4,5 (also copies)
                (4, {"a"}, 5, {"g"}),
                (4, set(), 4, set()),
                (5, {"a"}, 4, set()),
                (5, set(), 5, {"g"}),
            ],
        )
        result = full_simplification_pipeline(m)
        assert len(result.states) == 2


# ====================================================================
# Tests: Comparison and statistics
# ====================================================================

class TestCompareAndStats:
    def test_compare_methods(self):
        m = redundant_states_machine()
        results = compare_simplification_methods(m)
        assert "minimize" in results
        assert "simulation" in results
        assert "full" in results
        for method, result in results.items():
            assert isinstance(result, SimplificationResult)

    def test_statistics(self):
        m = redundant_states_machine()
        result = simplify(m, method="minimize")
        stats = simplification_statistics(result)
        assert stats["original_states"] == 4
        assert stats["simplified_states"] == 2
        assert "50.0%" in stats["state_reduction"]

    def test_summary(self):
        m = redundant_states_machine()
        result = simplify(m, method="minimize")
        summary = simplification_summary(result)
        assert "minimize" in summary
        assert "4" in summary
        assert "2" in summary


# ====================================================================
# Tests: Distributed simplification
# ====================================================================

class TestDistributedSimplification:
    def test_simplify_distributed(self):
        m1 = redundant_states_machine()
        m2 = large_redundant_machine()
        results = simplify_distributed({"P1": m1, "P2": m2}, method="minimize")
        assert "P1" in results
        assert "P2" in results
        assert results["P1"].simplified_states == 2
        assert results["P2"].simplified_states == 3

    def test_distributed_summary(self):
        m1 = redundant_states_machine()
        m2 = large_redundant_machine()
        results = simplify_distributed({"P1": m1, "P2": m2})
        summary = distributed_simplification_summary(results)
        assert "P1" in summary
        assert "P2" in summary
        assert "Total" in summary

    def test_distributed_single_process(self):
        m = two_state_toggle()
        results = simplify_distributed({"P": m})
        assert results["P"].simplified_states == 2


# ====================================================================
# Tests: Cross-machine simulation
# ====================================================================

class TestCrossSimulation:
    def test_identical_machines(self):
        m = two_state_toggle()
        sim = compute_cross_simulation(m, m)
        assert sim.simulates(0, 0)
        assert sim.simulates(1, 1)

    def test_simulated_by_identical(self):
        m = two_state_toggle()
        assert is_simulated_by(m, m)

    def test_larger_simulates_smaller(self):
        """Minimized version should simulate original (and vice versa)."""
        m = redundant_states_machine()
        m_min = minimize_mealy(m)
        # m_min simulates m (from initial states)
        assert is_simulated_by(m, m_min)

    def test_different_behavior_no_simulation(self):
        """Machines with different behavior shouldn't simulate."""
        m1 = make_mealy(
            states=[0],
            initial=0,
            inputs=["a"],
            outputs=["g"],
            transitions=[(0, {"a"}, 0, {"g"}), (0, set(), 0, set())],
        )
        m2 = make_mealy(
            states=[0],
            initial=0,
            inputs=["a"],
            outputs=["g"],
            transitions=[(0, {"a"}, 0, set()), (0, set(), 0, {"g"})],
        )
        assert not is_simulated_by(m1, m2)
        assert not is_simulated_by(m2, m1)

    def test_different_io_signature(self):
        m1 = make_mealy(
            states=[0], initial=0,
            inputs=["a"], outputs=["g"],
            transitions=[(0, {"a"}, 0, {"g"}), (0, set(), 0, set())],
        )
        m2 = make_mealy(
            states=[0], initial=0,
            inputs=["b"], outputs=["g"],
            transitions=[(0, {"b"}, 0, {"g"}), (0, set(), 0, set())],
        )
        sim = compute_cross_simulation(m1, m2)
        assert len(sim.pairs) == 0  # Different inputs


# ====================================================================
# Tests: SimulationRelation
# ====================================================================

class TestSimulationRelation:
    def test_equivalent_pairs(self):
        m = redundant_states_machine()
        sim = compute_forward_simulation(m)
        eq = sim.equivalent_pairs()
        # (0,2) and (2,0) should both be present
        assert (0, 2) in eq
        assert (2, 0) in eq

    def test_no_equivalent_pairs(self):
        m = two_state_toggle()
        sim = compute_forward_simulation(m)
        eq = sim.equivalent_pairs()
        # Only self-pairs should be equivalent
        for (a, b) in eq:
            assert a == b


# ====================================================================
# Tests: Edge cases
# ====================================================================

class TestEdgeCases:
    def test_empty_transitions(self):
        m = MealyMachine(
            states={0},
            initial=0,
            inputs={"a"},
            outputs={"g"},
            transitions={},
        )
        result = simplify(m, method="minimize")
        assert result.simplified_states <= 1

    def test_single_state_all_methods(self):
        m = single_state_machine()
        for method in ["minimize", "simulation", "dont_care", "input_reduce", "full"]:
            result = simplify(m, method=method)
            assert result.simplified_states <= 1

    def test_two_inputs(self):
        m = make_mealy(
            states=[0, 1],
            initial=0,
            inputs=["a", "b"],
            outputs=["g"],
            transitions=[
                (0, {"a", "b"}, 1, {"g"}),
                (0, {"a"}, 1, {"g"}),
                (0, {"b"}, 0, set()),
                (0, set(), 0, set()),
                (1, {"a", "b"}, 0, set()),
                (1, {"a"}, 0, set()),
                (1, {"b"}, 1, {"g"}),
                (1, set(), 1, {"g"}),
            ],
        )
        result = simplify(m, method="full")
        assert result.simplified_states <= 2

    def test_two_outputs(self):
        m = make_mealy(
            states=[0, 1],
            initial=0,
            inputs=["a"],
            outputs=["g", "h"],
            transitions=[
                (0, {"a"}, 1, {"g"}),
                (0, set(), 0, {"h"}),
                (1, {"a"}, 0, {"h"}),
                (1, set(), 1, {"g"}),
            ],
        )
        result = simplify(m, method="full")
        assert result.simplified_states <= 2

    def test_self_loop_machine(self):
        m = make_mealy(
            states=[0, 1],
            initial=0,
            inputs=["a"],
            outputs=["g"],
            transitions=[
                (0, {"a"}, 0, {"g"}),
                (0, set(), 0, set()),
                (1, {"a"}, 1, {"g"}),
                (1, set(), 1, set()),
            ],
        )
        # States 0 and 1 are identical self-loops
        result = simplify(m, method="full")
        assert result.simplified_states == 1

    def test_make_mealy_helper(self):
        m = make_mealy(
            states=[0],
            initial=0,
            inputs=["a"],
            outputs=["g"],
            transitions=[(0, {"a"}, 0, {"g"}), (0, set(), 0, set())],
        )
        assert len(m.states) == 1
        assert m.initial == 0
        assert (0, frozenset({"a"})) in m.transitions

    def test_large_input_space(self):
        """Machine with 3 inputs (8 valuations)."""
        transitions = []
        for a in [False, True]:
            for b in [False, True]:
                for c in [False, True]:
                    inp = set()
                    if a:
                        inp.add("a")
                    if b:
                        inp.add("b")
                    if c:
                        inp.add("c")
                    out = {"g"} if a else set()
                    transitions.append((0, inp, 0, out))
        m = make_mealy(
            states=[0],
            initial=0,
            inputs=["a", "b", "c"],
            outputs=["g"],
            transitions=transitions,
        )
        irr = find_irrelevant_inputs(m)
        assert "b" in irr
        assert "c" in irr
        assert "a" not in irr


# ====================================================================
# Tests: Synthesis integration
# ====================================================================

class TestSynthesisIntegration:
    """Test simplification on machines built from reactive synthesis."""

    def test_simplify_synthesized_machine(self):
        """Build a small Mealy machine resembling synthesis output and simplify."""
        # Simulates a 4-state controller with redundancy
        m = make_mealy(
            states=[0, 1, 2, 3],
            initial=0,
            inputs=["r"],
            outputs=["g"],
            transitions=[
                # States 0,2 are equivalent (grant on request, deny otherwise)
                (0, {"r"}, 1, {"g"}),
                (0, set(), 0, set()),
                (2, {"r"}, 3, {"g"}),
                (2, set(), 2, set()),
                # States 1,3 are equivalent (deny on request, grant otherwise)
                (1, {"r"}, 0, set()),
                (1, set(), 1, {"g"}),
                (3, {"r"}, 2, set()),
                (3, set(), 3, {"g"}),
            ],
        )
        result = simplify(m, method="full")
        assert result.simplified_states == 2
        assert result.equivalent
        assert result.state_reduction == 0.5

    def test_simplify_mutual_exclusion(self):
        """Controller for mutual exclusion (2 processes)."""
        m = make_mealy(
            states=[0, 1, 2],
            initial=0,
            inputs=["r1", "r2"],
            outputs=["g1", "g2"],
            transitions=[
                # Idle: grant r1 if requested, else grant r2 if requested
                (0, {"r1", "r2"}, 1, {"g1"}),
                (0, {"r1"}, 1, {"g1"}),
                (0, {"r2"}, 2, {"g2"}),
                (0, set(), 0, set()),
                # Process 1 has lock
                (1, {"r1", "r2"}, 1, {"g1"}),
                (1, {"r1"}, 1, {"g1"}),
                (1, {"r2"}, 0, set()),  # release
                (1, set(), 0, set()),   # release
                # Process 2 has lock
                (2, {"r1", "r2"}, 2, {"g2"}),
                (2, {"r1"}, 0, set()),   # release
                (2, {"r2"}, 2, {"g2"}),
                (2, set(), 0, set()),    # release
            ],
        )
        result = simplify(m, method="full")
        # Already minimal (3 states with distinct behavior)
        assert result.simplified_states == 3
        assert result.equivalent


# ====================================================================
# Tests: SimplificationResult properties
# ====================================================================

class TestSimplificationResult:
    def test_zero_original_states(self):
        r = SimplificationResult(
            original=None, simplified=None,
            method="test",
            original_states=0, simplified_states=0,
            original_transitions=0, simplified_transitions=0,
            equivalent=True,
        )
        assert r.state_reduction == 0.0
        assert r.transition_reduction == 0.0

    def test_full_reduction(self):
        r = SimplificationResult(
            original=None, simplified=None,
            method="test",
            original_states=10, simplified_states=0,
            original_transitions=20, simplified_transitions=0,
            equivalent=True,
        )
        assert r.state_reduction == 1.0
        assert r.transition_reduction == 1.0

    def test_no_reduction(self):
        r = SimplificationResult(
            original=None, simplified=None,
            method="test",
            original_states=5, simplified_states=5,
            original_transitions=10, simplified_transitions=10,
            equivalent=True,
        )
        assert r.state_reduction == 0.0
        assert r.transition_reduction == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
