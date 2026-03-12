"""Tests for V171: Interpolation-Based Model Checking."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from interpolation_mc import (
    SymbolicTS, InterpolantState, Interpolant,
    BMCResult, bounded_model_check,
    compute_interpolant, compute_interpolant_sequence,
    interpolation_model_check, incremental_interpolation_mc,
    IMCVerdict, IMCResult,
    kripke_to_ts, concrete_to_ts,
    make_safe_counter, make_unsafe_counter, make_mutual_exclusion,
    make_token_ring, make_producer_consumer, make_two_phase_commit,
    verify_imc_result, compare_imc_methods, imc_statistics, batch_verify,
    _check_inductive, _all_states_from_domain,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V170_mu_calculus_cegar'))
from mu_calculus_cegar import (
    KripkeStructure, ConcreteSystem,
    make_counter_system, make_traffic_light, make_mutex_system,
)


# ============================================================
# Test: BMC (Bounded Model Checking)
# ============================================================

class TestBMC:
    def test_safe_counter_bmc(self):
        ts = make_safe_counter(4)
        r = bounded_model_check(ts, 10)
        assert r.safe  # wrapping counter never exceeds bound

    def test_unsafe_counter_bmc(self):
        ts = make_unsafe_counter(3)
        r = bounded_model_check(ts, 5)
        assert not r.safe
        assert r.counterexample is not None

    def test_unsafe_counter_cex_valid(self):
        ts = make_unsafe_counter(3)
        r = bounded_model_check(ts, 5)
        trace = r.counterexample
        assert ts.init_pred(trace[0])
        assert ts.bad_pred(trace[-1])
        for i in range(len(trace) - 1):
            assert ts.trans_pred(trace[i], trace[i + 1])

    def test_mutex_bmc_finds_violation(self):
        ts = make_mutual_exclusion()
        r = bounded_model_check(ts, 10)
        assert not r.safe

    def test_producer_consumer_bmc_safe(self):
        ts = make_producer_consumer()
        r = bounded_model_check(ts, 10)
        assert r.safe

    def test_two_phase_commit_bmc(self):
        ts = make_two_phase_commit()
        r = bounded_model_check(ts, 10)
        assert r.safe

    def test_token_ring_bmc(self):
        ts = make_token_ring(5)
        r = bounded_model_check(ts, 10)
        assert r.safe  # no bad states

    def test_bmc_depth_zero(self):
        ts = make_unsafe_counter(3)
        r = bounded_model_check(ts, 0)
        # At depth 0, initial state x=0 is not bad (bad = x>=3)
        assert r.safe

    def test_bmc_cex_length(self):
        ts = make_unsafe_counter(3)
        r = bounded_model_check(ts, 5)
        assert len(r.counterexample) >= 4  # 0 -> 1 -> 2 -> 3 (bad)


# ============================================================
# Test: Interpolant Computation
# ============================================================

class TestInterpolant:
    def test_interpolant_safe_counter(self):
        ts = make_safe_counter(4)
        interp = compute_interpolant(ts, 5)
        assert interp is not None
        # Interpolant should include init state
        assert interp.predicate({'x': 0})

    def test_interpolant_excludes_bad(self):
        ts = make_safe_counter(4)
        interp = compute_interpolant(ts, 5)
        # Interpolant should exclude bad states
        assert not interp.predicate({'x': 4})

    def test_interpolant_sequence_length(self):
        ts = make_safe_counter(3)
        seq = compute_interpolant_sequence(ts, 5)
        assert len(seq) == 6  # I_0 through I_5

    def test_interpolant_sequence_names(self):
        ts = make_safe_counter(3)
        seq = compute_interpolant_sequence(ts, 3)
        for i, interp in enumerate(seq):
            assert interp.name == f"I_{i}"

    def test_interpolant_init_included(self):
        ts = make_safe_counter(4)
        seq = compute_interpolant_sequence(ts, 3)
        # I_0 should include initial state
        assert seq[0].predicate({'x': 0})

    def test_interpolant_producer_consumer(self):
        ts = make_producer_consumer()
        interp = compute_interpolant(ts, 5)
        assert interp is not None
        assert interp.predicate({'buf': 0})
        assert not interp.predicate({'buf': 4})


# ============================================================
# Test: Interpolation-Based Model Checking
# ============================================================

class TestIMC:
    def test_safe_counter_imc(self):
        ts = make_safe_counter(4)
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE

    def test_safe_counter_has_invariant(self):
        ts = make_safe_counter(4)
        r = interpolation_model_check(ts, max_depth=10)
        assert r.invariant is not None
        # Invariant should be inductive
        all_states = _all_states_from_domain(ts)
        assert _check_inductive(ts, r.invariant, all_states)

    def test_unsafe_counter_imc(self):
        ts = make_unsafe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.UNSAFE

    def test_unsafe_counter_has_cex(self):
        ts = make_unsafe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        assert r.counterexample is not None
        assert len(r.counterexample) >= 2

    def test_mutex_unsafe(self):
        ts = make_mutual_exclusion()
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.UNSAFE

    def test_producer_consumer_safe(self):
        ts = make_producer_consumer()
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE

    def test_two_phase_commit_safe(self):
        ts = make_two_phase_commit()
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE

    def test_token_ring_safe(self):
        ts = make_token_ring(5)
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE

    def test_imc_iterations_counted(self):
        ts = make_safe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        assert r.iterations >= 1

    def test_imc_history_populated(self):
        ts = make_safe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        assert len(r.history) >= 1


# ============================================================
# Test: Incremental Interpolation MC
# ============================================================

class TestIncrementalIMC:
    def test_safe_counter_incremental(self):
        ts = make_safe_counter(4)
        r = incremental_interpolation_mc(ts, max_depth=10)
        assert r.verdict in (IMCVerdict.SAFE, IMCVerdict.UNKNOWN)

    def test_unsafe_counter_incremental(self):
        ts = make_unsafe_counter(3)
        r = incremental_interpolation_mc(ts, max_depth=10)
        assert r.verdict == IMCVerdict.UNSAFE

    def test_mutex_incremental(self):
        ts = make_mutual_exclusion()
        r = incremental_interpolation_mc(ts, max_depth=10)
        assert r.verdict == IMCVerdict.UNSAFE

    def test_producer_consumer_incremental(self):
        ts = make_producer_consumer()
        r = incremental_interpolation_mc(ts, max_depth=10)
        assert r.verdict in (IMCVerdict.SAFE, IMCVerdict.UNKNOWN)

    def test_incremental_history(self):
        ts = make_safe_counter(3)
        r = incremental_interpolation_mc(ts, max_depth=10)
        assert len(r.history) >= 1


# ============================================================
# Test: Inductiveness Checking
# ============================================================

class TestInductiveness:
    def test_true_is_not_inductive_for_unsafe(self):
        ts = make_unsafe_counter(3)
        all_states = _all_states_from_domain(ts)
        # "All states" is not inductive because it includes bad
        true_interp = Interpolant(lambda s: True, "all", 0)
        assert not _check_inductive(ts, true_interp, all_states)

    def test_init_only_may_not_be_inductive(self):
        ts = make_safe_counter(4)
        all_states = _all_states_from_domain(ts)
        # Only init state is not inductive (successors not included)
        init_interp = Interpolant(lambda s: s['x'] == 0, "init", 0)
        assert not _check_inductive(ts, init_interp, all_states)

    def test_safe_set_is_inductive(self):
        ts = make_safe_counter(4)
        all_states = _all_states_from_domain(ts)
        # All states x < 4 should be inductive for safe counter
        safe_interp = Interpolant(lambda s: s['x'] < 4, "safe", 0)
        assert _check_inductive(ts, safe_interp, all_states)

    def test_token_ring_all_is_inductive(self):
        ts = make_token_ring(3)
        all_states = _all_states_from_domain(ts)
        # All states is inductive when no bad states
        all_interp = Interpolant(lambda s: True, "all", 0)
        assert _check_inductive(ts, all_interp, all_states)


# ============================================================
# Test: Kripke to TS Conversion
# ============================================================

class TestKripkeConversion:
    def test_kripke_to_ts_basic(self):
        ks = KripkeStructure(
            states={0, 1, 2}, initial={0},
            transitions={0: {1}, 1: {2}, 2: {0}},
            labeling={0: {'safe'}, 1: {'safe'}, 2: {'bad'}}
        )
        ts = kripke_to_ts(ks, 'bad')
        assert ts.init_pred({'state': 0})
        assert ts.bad_pred({'state': 2})

    def test_kripke_to_ts_transitions(self):
        ks = KripkeStructure(
            states={0, 1}, initial={0},
            transitions={0: {1}, 1: {0}},
            labeling={0: set(), 1: {'bad'}}
        )
        ts = kripke_to_ts(ks, 'bad')
        assert ts.trans_pred({'state': 0}, {'state': 1})
        assert ts.trans_pred({'state': 1}, {'state': 0})
        assert not ts.trans_pred({'state': 0}, {'state': 0})

    def test_kripke_imc(self):
        ks = KripkeStructure(
            states={0, 1, 2}, initial={0},
            transitions={0: {1}, 1: {2}, 2: {0}},
            labeling={0: set(), 1: set(), 2: {'bad'}}
        )
        ts = kripke_to_ts(ks, 'bad')
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.UNSAFE

    def test_kripke_safe_imc(self):
        ks = KripkeStructure(
            states={0, 1}, initial={0},
            transitions={0: {1}, 1: {0}},
            labeling={0: {'safe'}, 1: {'safe'}}
        )
        ts = kripke_to_ts(ks, 'bad')
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE


# ============================================================
# Test: Concrete System Conversion
# ============================================================

class TestConcreteConversion:
    def test_counter_to_ts(self):
        sys = make_counter_system(3)
        ts = concrete_to_ts(sys, 'nonexistent')
        assert ts.init_pred(sys.init_states[0])

    def test_traffic_to_ts(self):
        sys = make_traffic_light()
        ts = concrete_to_ts(sys, 'nonexistent')
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE  # no bad states


# ============================================================
# Test: Verification of Results
# ============================================================

class TestVerification:
    def test_verify_safe_result(self):
        ts = make_safe_counter(4)
        r = interpolation_model_check(ts, max_depth=10)
        v = verify_imc_result(ts, r)
        assert v['verdict'] == 'SAFE'
        assert v['invariant_inductive']

    def test_verify_unsafe_result(self):
        ts = make_unsafe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        v = verify_imc_result(ts, r)
        assert v['verdict'] == 'UNSAFE'
        assert v['cex_starts_init']
        assert v['cex_ends_bad']
        assert v['cex_valid_transitions']

    def test_verify_mutex_cex(self):
        ts = make_mutual_exclusion()
        r = interpolation_model_check(ts, max_depth=10)
        v = verify_imc_result(ts, r)
        assert v['cex_starts_init']
        assert v['cex_ends_bad']
        assert v['cex_valid_transitions']


# ============================================================
# Test: Comparison of Methods
# ============================================================

class TestComparison:
    def test_compare_safe(self):
        ts = make_safe_counter(3)
        c = compare_imc_methods(ts)
        assert c['agree'] or c['standard']['verdict'] == 'SAFE'

    def test_compare_unsafe(self):
        ts = make_unsafe_counter(3)
        c = compare_imc_methods(ts)
        assert c['agree']
        assert c['standard']['verdict'] == 'UNSAFE'

    def test_compare_mutex(self):
        ts = make_mutual_exclusion()
        c = compare_imc_methods(ts)
        assert c['agree']
        assert c['standard']['verdict'] == 'UNSAFE'


# ============================================================
# Test: Statistics
# ============================================================

class TestStatistics:
    def test_imc_statistics_safe(self):
        ts = make_safe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        s = imc_statistics(r)
        assert s['verdict'] == 'SAFE'
        assert s['has_invariant']
        assert not s['has_cex']

    def test_imc_statistics_unsafe(self):
        ts = make_unsafe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        s = imc_statistics(r)
        assert s['verdict'] == 'UNSAFE'
        assert not s['has_invariant']
        assert s['has_cex']


# ============================================================
# Test: Batch Verification
# ============================================================

class TestBatch:
    def test_batch_verify(self):
        systems = [
            ("safe_counter", make_safe_counter(3)),
            ("unsafe_counter", make_unsafe_counter(3)),
            ("producer_consumer", make_producer_consumer()),
        ]
        results = batch_verify(systems)
        assert len(results) == 3
        assert results[0]['verdict'] == 'SAFE'
        assert results[1]['verdict'] == 'UNSAFE'
        assert results[2]['verdict'] == 'SAFE'

    def test_batch_names(self):
        systems = [
            ("sys_a", make_safe_counter(2)),
            ("sys_b", make_unsafe_counter(2)),
        ]
        results = batch_verify(systems)
        assert results[0]['name'] == 'sys_a'
        assert results[1]['name'] == 'sys_b'


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_state_safe(self):
        ts = SymbolicTS(
            variables=['x'],
            init_pred=lambda s: s['x'] == 0,
            trans_pred=lambda s, t: t['x'] == 0,
            bad_pred=lambda s: False,
            domain={'x': [0]}
        )
        r = interpolation_model_check(ts, max_depth=5)
        assert r.verdict == IMCVerdict.SAFE

    def test_single_state_bad(self):
        ts = SymbolicTS(
            variables=['x'],
            init_pred=lambda s: s['x'] == 0,
            trans_pred=lambda s, t: t['x'] == 0,
            bad_pred=lambda s: s['x'] == 0,
            domain={'x': [0]}
        )
        r = interpolation_model_check(ts, max_depth=5)
        assert r.verdict == IMCVerdict.UNSAFE

    def test_no_transitions(self):
        ts = SymbolicTS(
            variables=['x'],
            init_pred=lambda s: s['x'] == 0,
            trans_pred=lambda s, t: False,
            bad_pred=lambda s: s['x'] == 1,
            domain={'x': [0, 1]}
        )
        r = interpolation_model_check(ts, max_depth=5)
        assert r.verdict == IMCVerdict.SAFE

    def test_immediate_bad(self):
        ts = SymbolicTS(
            variables=['x'],
            init_pred=lambda s: s['x'] == 0,
            trans_pred=lambda s, t: t['x'] == 1,
            bad_pred=lambda s: s['x'] == 1,
            domain={'x': [0, 1]}
        )
        r = interpolation_model_check(ts, max_depth=5)
        assert r.verdict == IMCVerdict.UNSAFE
        assert len(r.counterexample) == 2  # init -> bad

    def test_two_variables_safe(self):
        ts = SymbolicTS(
            variables=['x', 'y'],
            init_pred=lambda s: s['x'] == 0 and s['y'] == 0,
            trans_pred=lambda s, t: (
                t['x'] == (s['x'] + 1) % 3 and t['y'] == s['y']
            ),
            bad_pred=lambda s: s['x'] == 2 and s['y'] == 1,
            domain={'x': [0, 1, 2], 'y': [0, 1]}
        )
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE  # y never changes from 0


# ============================================================
# Test: Custom Transition Systems
# ============================================================

class TestCustomSystems:
    def test_simple_protocol(self):
        """Client-server: client sends, server processes, client receives."""
        ts = SymbolicTS(
            variables=['client', 'server'],
            init_pred=lambda s: s['client'] == 0 and s['server'] == 0,
            trans_pred=lambda s, t: (
                (s['client'] == 0 and s['server'] == 0 and
                 t['client'] == 1 and t['server'] == 0) or
                (s['client'] == 1 and s['server'] == 0 and
                 t['client'] == 1 and t['server'] == 1) or
                (s['client'] == 1 and s['server'] == 1 and
                 t['client'] == 2 and t['server'] == 0) or
                (s['client'] == 2 and t['client'] == 0 and t['server'] == 0)
            ),
            bad_pred=lambda s: s['client'] == 2 and s['server'] == 1,
            domain={'client': [0, 1, 2], 'server': [0, 1]}
        )
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE

    def test_nondeterministic_safe(self):
        """Nondeterministic system that stays safe."""
        ts = SymbolicTS(
            variables=['x'],
            init_pred=lambda s: s['x'] == 1,
            trans_pred=lambda s, t: (
                (s['x'] == 1 and t['x'] in (1, 2)) or
                (s['x'] == 2 and t['x'] in (1, 2))
            ),
            bad_pred=lambda s: s['x'] == 0 or s['x'] == 3,
            domain={'x': [0, 1, 2, 3]}
        )
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE

    def test_nondeterministic_unsafe(self):
        """Nondeterministic system that can reach bad."""
        ts = SymbolicTS(
            variables=['x'],
            init_pred=lambda s: s['x'] == 1,
            trans_pred=lambda s, t: (
                (s['x'] == 1 and t['x'] in (1, 2, 3)) or
                (s['x'] == 2 and t['x'] in (1, 2))
            ),
            bad_pred=lambda s: s['x'] == 3,
            domain={'x': [0, 1, 2, 3]}
        )
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.UNSAFE

    def test_deep_reachability(self):
        """Bad state reachable only after many steps."""
        n = 6
        vals = list(range(n + 1))
        ts = SymbolicTS(
            variables=['x'],
            init_pred=lambda s: s['x'] == 0,
            trans_pred=lambda s, t, n=n: s['x'] < n and t['x'] == s['x'] + 1,
            bad_pred=lambda s, n=n: s['x'] == n,
            domain={'x': vals}
        )
        r = interpolation_model_check(ts, max_depth=n + 2)
        assert r.verdict == IMCVerdict.UNSAFE
        assert len(r.counterexample) == n + 1


# ============================================================
# Test: Invariant Properties
# ============================================================

class TestInvariantProperties:
    def test_invariant_includes_init(self):
        ts = make_safe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        assert r.invariant is not None
        assert r.invariant.predicate({'x': 0})

    def test_invariant_excludes_bad(self):
        ts = make_safe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        assert not r.invariant.predicate({'x': 3})

    def test_invariant_includes_reachable(self):
        ts = make_safe_counter(3)
        r = interpolation_model_check(ts, max_depth=10)
        # All reachable states (0, 1, 2) should be in invariant
        for x in range(3):
            assert r.invariant.predicate({'x': x})

    def test_producer_consumer_invariant(self):
        ts = make_producer_consumer()
        r = interpolation_model_check(ts, max_depth=10)
        assert r.invariant is not None
        # Buffer values 0..3 should be in invariant
        for b in range(4):
            assert r.invariant.predicate({'buf': b})
        # Buffer > 3 should not be
        assert not r.invariant.predicate({'buf': 4})


# ============================================================
# Test: Two-Phase Commit Details
# ============================================================

class TestTwoPhaseCommit:
    def test_two_phase_safe(self):
        ts = make_two_phase_commit()
        r = interpolation_model_check(ts, max_depth=10)
        assert r.verdict == IMCVerdict.SAFE

    def test_two_phase_invariant(self):
        ts = make_two_phase_commit()
        r = interpolation_model_check(ts, max_depth=10)
        assert r.invariant is not None
        # Init state should be in invariant
        assert r.invariant.predicate({'coord': 0, 'part': 0})
        # Bad states should not
        assert not r.invariant.predicate({'coord': 2, 'part': 3})
        assert not r.invariant.predicate({'coord': 3, 'part': 2})

    def test_two_phase_verification(self):
        ts = make_two_phase_commit()
        r = interpolation_model_check(ts, max_depth=10)
        v = verify_imc_result(ts, r)
        assert v['invariant_inductive']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
