"""
Tests for V197: Delay Game Optimization
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V186_reactive_synthesis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V193_delay_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))

from delay_game_optimization import (
    # Data structures
    SymbolicArena, OptimizationResult, IncrementalResult, ArenaStats, ReductionReport,
    # Utilities
    _bits_needed, _encode_value, _decode_assignment,
    # Arena construction
    build_symbolic_arena,
    # Solving
    symbolic_parity_solve,
    # Reduction
    reduce_arena,
    # Incremental search
    incremental_find_minimum_delay,
    # Synthesis
    symbolic_synthesize,
    compare_symbolic_vs_explicit,
    # Shortcuts
    symbolic_safety_synthesize,
    symbolic_reachability_synthesize,
    symbolic_response_synthesize,
    symbolic_liveness_synthesize,
    # Analysis
    arena_statistics,
    compare_arena_sizes,
    enhanced_delay_analysis,
    # Summary
    optimization_summary,
    incremental_summary,
)

from bdd_model_checker import BDD
from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies, Iff,
    Next, Finally, Globally, Until, Release,
    LTLTrue, LTLFalse, parse_ltl,
    ltl_to_gba, gba_to_nba, Label, NBA
)
from delay_games import (
    DelayGameResult, MinDelayResult,
    synthesize_with_delay as v193_synthesize,
    find_minimum_delay as v193_find_minimum_delay,
)


# ============================================================
# Test Utilities
# ============================================================

class TestBitsNeeded:
    def test_bits_for_1(self):
        assert _bits_needed(1) == 1

    def test_bits_for_2(self):
        assert _bits_needed(2) == 1

    def test_bits_for_3(self):
        assert _bits_needed(3) == 2

    def test_bits_for_4(self):
        assert _bits_needed(4) == 2

    def test_bits_for_5(self):
        assert _bits_needed(5) == 3

    def test_bits_for_8(self):
        assert _bits_needed(8) == 3

    def test_bits_for_9(self):
        assert _bits_needed(9) == 4

    def test_bits_for_16(self):
        assert _bits_needed(16) == 4


class TestEncoding:
    def test_encode_zero(self):
        bdd = BDD()
        bits = ["b0", "b1"]
        for b in bits:
            bdd.named_var(b)
        enc = _encode_value(bdd, 0, bits)
        assert enc._id != bdd.FALSE._id
        # Value 0: both bits false
        r = bdd.AND(enc, bdd.AND(bdd.NOT(bdd.named_var("b0")), bdd.NOT(bdd.named_var("b1"))))
        assert r._id == enc._id

    def test_encode_one(self):
        bdd = BDD()
        bits = ["b0", "b1"]
        for b in bits:
            bdd.named_var(b)
        enc = _encode_value(bdd, 1, bits)
        # Value 1: b0=True, b1=False
        r = bdd.AND(enc, bdd.AND(bdd.named_var("b0"), bdd.NOT(bdd.named_var("b1"))))
        assert r._id == enc._id

    def test_encode_three(self):
        bdd = BDD()
        bits = ["b0", "b1"]
        for b in bits:
            bdd.named_var(b)
        enc = _encode_value(bdd, 3, bits)
        # Value 3: both bits true
        r = bdd.AND(enc, bdd.AND(bdd.named_var("b0"), bdd.named_var("b1")))
        assert r._id == enc._id

    def test_decode_round_trip(self):
        bdd = BDD()
        bits = ["b0", "b1", "b2"]
        for b in bits:
            bdd.named_var(b)
        for v in range(8):
            enc = _encode_value(bdd, v, bits)
            asgn = bdd.any_sat(enc)
            decoded = _decode_assignment(asgn, bits, bdd)
            assert decoded == v, f"Round trip failed for {v}: got {decoded}"


# ============================================================
# Test Data Structures
# ============================================================

class TestDataStructures:
    def test_optimization_result_defaults(self):
        r = OptimizationResult(realizable=True, delay=1)
        assert r.realizable
        assert r.delay == 1
        assert r.controller is None
        assert r.method == "symbolic"
        assert r.arena_bdd_nodes == 0
        assert r.solve_time_ms == 0.0

    def test_incremental_result_defaults(self):
        r = IncrementalResult(realizable=False)
        assert not r.realizable
        assert r.min_delay == -1
        assert r.results == {}
        assert r.searched_delays == []
        assert r.reuse_ratio == 0.0

    def test_arena_stats(self):
        s = ArenaStats(delay=2, nba_states=3, bdd_nodes=10,
                       reachable_states=24, env_vertices=12, sys_vertices=12,
                       accepting_count=8, max_priority=2, buffer_width=1)
        assert s.delay == 2
        assert s.nba_states == 3
        assert s.buffer_width == 1

    def test_reduction_report(self):
        r = ReductionReport(original_nodes=100, reduced_nodes=60,
                           dead_states_removed=5, priority_levels_compressed=0,
                           reduction_ratio=0.4, techniques_applied=["dead_state_removal"])
        assert r.reduction_ratio == 0.4
        assert "dead_state_removal" in r.techniques_applied


# ============================================================
# Test Symbolic Arena Construction
# ============================================================

class TestBuildSymbolicArena:
    def _make_simple_nba(self):
        """NBA for G(g): single accepting state, stays accepting when g=True."""
        states = {0, 1}
        initial = {0}
        accepting = {0}
        transitions = {
            0: [(Label(frozenset({"g"}), frozenset()), 0),    # g=T -> stay accepting
                (Label(frozenset(), frozenset({"g"})), 1)],   # g=F -> go to 1
            1: [(Label(frozenset({"g"}), frozenset()), 0),    # g=T -> back to accepting
                (Label(frozenset(), frozenset({"g"})), 1)],   # g=F -> stay in 1
        }
        return NBA(states=states, initial=initial, accepting=accepting,
                   transitions=transitions, ap={"g"})

    def test_arena_has_bdd(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        assert arena.bdd is not None
        assert isinstance(arena.bdd, BDD)

    def test_arena_has_initial_states(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        assert arena.initial._id != arena.bdd.FALSE._id

    def test_arena_delay_stored(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=2)
        assert arena.delay == 2

    def test_arena_vars_stored(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        assert arena.env_vars == ["r"]
        assert arena.sys_vars == ["g"]

    def test_arena_nba_states(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        assert arena.nba_states == 2

    def test_arena_delay0_has_transitions(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=0)
        assert arena.trans_env._id != arena.bdd.FALSE._id

    def test_arena_delay1_has_both_transitions(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        assert arena.trans_env._id != arena.bdd.FALSE._id
        assert arena.trans_sys._id != arena.bdd.FALSE._id

    def test_arena_accepting_encoded(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        assert arena.accepting._id != arena.bdd.FALSE._id

    def test_arena_buffer_bits_match_delay(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=3)
        assert len(arena.buffer_bits) == 3
        for pos_bits in arena.buffer_bits:
            assert len(pos_bits) == 1  # one env var "r"

    def test_arena_multiple_env_vars(self):
        nba = self._make_simple_nba()
        arena = build_symbolic_arena(nba, {"r1", "r2"}, {"g"}, delay=2)
        assert len(arena.env_vars) == 2
        for pos_bits in arena.buffer_bits:
            assert len(pos_bits) == 2  # two env vars


# ============================================================
# Test Symbolic Parity Solving
# ============================================================

class TestSymbolicParitySolve:
    def _make_nba_true(self):
        """NBA accepting everything (single accepting state with self-loop)."""
        return NBA(
            states={0}, initial={0}, accepting={0},
            transitions={0: [(Label(frozenset(), frozenset()), 0)]},
            ap=set()
        )

    def _make_nba_false(self):
        """NBA accepting nothing (no accepting states reachable)."""
        return NBA(
            states={0}, initial={0}, accepting=set(),
            transitions={0: [(Label(frozenset(), frozenset()), 0)]},
            ap=set()
        )

    def test_true_spec_winning(self):
        """Trivial spec (always true) should be winning for system."""
        nba = self._make_nba_true()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=0)
        sys_win, _ = symbolic_parity_solve(arena)
        # Initial should be winning
        bdd = arena.bdd
        init_win = bdd.AND(arena.initial, sys_win)
        assert init_win._id != bdd.FALSE._id

    def test_false_spec_not_winning(self):
        """Spec with no accepting states should not be winning."""
        nba = self._make_nba_false()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=0)
        sys_win, _ = symbolic_parity_solve(arena)
        bdd = arena.bdd
        init_win = bdd.AND(arena.initial, sys_win)
        # With no accepting states, system cannot win Buchi game
        assert init_win._id == bdd.FALSE._id

    def test_delay1_solve(self):
        """Solve with delay=1 should complete without error."""
        nba = self._make_nba_true()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        sys_win, _ = symbolic_parity_solve(arena)
        assert sys_win is not None


# ============================================================
# Test Arena Reduction
# ============================================================

class TestReduceArena:
    def _make_nba(self):
        return NBA(
            states={0, 1}, initial={0}, accepting={0},
            transitions={
                0: [(Label(frozenset({"g"}), frozenset()), 0),
                    (Label(frozenset(), frozenset({"g"})), 1)],
                1: [(Label(frozenset({"g"}), frozenset()), 0),
                    (Label(frozenset(), frozenset({"g"})), 1)],
            },
            ap={"g"}
        )

    def test_reduce_returns_arena_and_report(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        reduced, report = reduce_arena(arena)
        assert isinstance(reduced, SymbolicArena)
        assert isinstance(report, ReductionReport)

    def test_reduce_preserves_initial(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        reduced, _ = reduce_arena(arena)
        assert reduced.initial._id != reduced.bdd.FALSE._id

    def test_reduce_preserves_delay(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=2)
        reduced, _ = reduce_arena(arena)
        assert reduced.delay == 2

    def test_reduction_report_fields(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        _, report = reduce_arena(arena)
        assert report.original_nodes >= 0
        assert report.reduced_nodes >= 0
        assert 0.0 <= report.reduction_ratio <= 1.0


# ============================================================
# Test Symbolic Synthesis
# ============================================================

class TestSymbolicSynthesize:
    def test_true_spec_realizable(self):
        spec = LTLTrue()
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert result.realizable

    def test_false_spec_unrealizable(self):
        spec = LTLFalse()
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert not result.realizable

    def test_safety_spec_delay0(self):
        g = Atom("g")
        spec = Globally(g)  # G(g) -- sys controls g, always realizable
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert result.realizable

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError):
            symbolic_synthesize(LTLTrue(), {"r"}, {"g"}, delay=-1)

    def test_result_has_timing(self):
        spec = LTLTrue()
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert result.total_time_ms >= 0
        assert result.arena_build_time_ms >= 0
        assert result.solve_time_ms >= 0

    def test_result_method(self):
        spec = LTLTrue()
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert result.method == "v193_standard"  # delay=0 delegates to V193

    def test_delay1_safety(self):
        g = Atom("g")
        spec = Globally(g)
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=1)
        assert result.realizable

    def test_delay2_safety(self):
        g = Atom("g")
        spec = Globally(g)
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=2)
        assert result.realizable


# ============================================================
# Test Incremental Delay Search
# ============================================================

class TestIncrementalSearch:
    def test_true_spec_delay0(self):
        spec = LTLTrue()
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=3)
        assert result.realizable
        assert result.min_delay == 0

    def test_false_spec_no_delay(self):
        spec = LTLFalse()
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=2)
        assert not result.realizable
        assert result.min_delay == -1

    def test_safety_spec_delay0(self):
        g = Atom("g")
        spec = Globally(g)
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=3)
        assert result.realizable
        assert result.min_delay == 0

    def test_all_delays_searched_when_unrealizable(self):
        spec = LTLFalse()
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=2)
        assert len(result.searched_delays) == 3  # 0, 1, 2

    def test_early_stop_when_realizable(self):
        spec = LTLTrue()
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=5)
        assert result.searched_delays == [0]  # found at delay 0, stopped

    def test_results_dict_populated(self):
        spec = LTLTrue()
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=3)
        assert 0 in result.results
        assert result.results[0].realizable

    def test_reuse_ratio(self):
        spec = LTLFalse()
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=2)
        # 3 delays searched (0,1,2), reuse ratio > 0
        assert len(result.searched_delays) == 3
        assert result.reuse_ratio > 0

    def test_total_time_tracked(self):
        spec = LTLTrue()
        result = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=1)
        assert result.total_time_ms >= 0


# ============================================================
# Test Compare Symbolic vs Explicit
# ============================================================

class TestCompareSymbolicVsExplicit:
    def test_agreement_true_spec(self):
        spec = LTLTrue()
        cmp = compare_symbolic_vs_explicit(spec, {"r"}, {"g"}, delay=0)
        assert cmp["agreement"]
        assert cmp["symbolic"]["realizable"]
        assert cmp["explicit"]["realizable"]

    def test_agreement_false_spec(self):
        spec = LTLFalse()
        cmp = compare_symbolic_vs_explicit(spec, {"r"}, {"g"}, delay=0)
        assert cmp["agreement"]
        assert not cmp["symbolic"]["realizable"]

    def test_agreement_safety(self):
        g = Atom("g")
        spec = Globally(g)
        cmp = compare_symbolic_vs_explicit(spec, {"r"}, {"g"}, delay=0)
        assert cmp["agreement"]

    def test_agreement_safety_delay1(self):
        g = Atom("g")
        spec = Globally(g)
        cmp = compare_symbolic_vs_explicit(spec, {"r"}, {"g"}, delay=1)
        assert cmp["agreement"]

    def test_has_timing_info(self):
        spec = LTLTrue()
        cmp = compare_symbolic_vs_explicit(spec, {"r"}, {"g"}, delay=0)
        assert "time_ms" in cmp["symbolic"]
        assert "time_ms" in cmp["explicit"]

    def test_delay_recorded(self):
        spec = LTLTrue()
        cmp = compare_symbolic_vs_explicit(spec, {"r"}, {"g"}, delay=2)
        assert cmp["delay"] == 2


# ============================================================
# Test Safety/Reachability/Response/Liveness Shortcuts
# ============================================================

class TestSpecializedSynthesis:
    def test_safety(self):
        bad = Atom("bad")
        result = symbolic_safety_synthesize(bad, {"r"}, {"bad"}, delay=0)
        # sys controls "bad", can avoid it
        assert result.realizable

    def test_reachability(self):
        target = Atom("g")
        result = symbolic_reachability_synthesize(target, {"r"}, {"g"}, delay=0)
        assert result.realizable

    def test_response(self):
        trigger = Atom("r")
        response = Atom("g")
        result = symbolic_response_synthesize(trigger, response, {"r"}, {"g"}, delay=0)
        assert result.realizable

    def test_liveness(self):
        cond = Atom("g")
        result = symbolic_liveness_synthesize(cond, {"r"}, {"g"}, delay=0)
        assert result.realizable

    def test_safety_with_delay(self):
        bad = Atom("bad")
        result = symbolic_safety_synthesize(bad, {"r"}, {"bad"}, delay=1)
        assert result.realizable

    def test_reachability_with_delay(self):
        target = Atom("g")
        result = symbolic_reachability_synthesize(target, {"r"}, {"g"}, delay=1)
        assert result.realizable


# ============================================================
# Test Arena Statistics
# ============================================================

class TestArenaStatistics:
    def _make_nba(self):
        return NBA(
            states={0, 1}, initial={0}, accepting={0},
            transitions={
                0: [(Label(frozenset({"g"}), frozenset()), 0),
                    (Label(frozenset(), frozenset({"g"})), 1)],
                1: [(Label(frozenset({"g"}), frozenset()), 0),
                    (Label(frozenset(), frozenset({"g"})), 1)],
            },
            ap={"g"}
        )

    def test_stats_delay(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        stats = arena_statistics(arena)
        assert stats.delay == 1

    def test_stats_nba_states(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        stats = arena_statistics(arena)
        assert stats.nba_states == 2

    def test_stats_max_priority(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        stats = arena_statistics(arena)
        assert stats.max_priority == 2

    def test_stats_buffer_width(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        stats = arena_statistics(arena)
        assert stats.buffer_width == 1

    def test_stats_bdd_nodes_positive(self):
        nba = self._make_nba()
        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        stats = arena_statistics(arena)
        assert stats.bdd_nodes >= 0


class TestCompareArenaSizes:
    def test_compare_delays(self):
        g = Atom("g")
        spec = Globally(g)
        results = compare_arena_sizes(spec, {"r"}, {"g"}, [0, 1, 2])
        assert 0 in results
        assert 1 in results
        assert 2 in results

    def test_results_have_fields(self):
        g = Atom("g")
        spec = Globally(g)
        results = compare_arena_sizes(spec, {"r"}, {"g"}, [0])
        assert "bdd_nodes" in results[0]
        assert "reachable_states" in results[0]


# ============================================================
# Test Enhanced Delay Analysis
# ============================================================

class TestEnhancedDelayAnalysis:
    def test_analysis_realizable(self):
        spec = LTLTrue()
        analysis = enhanced_delay_analysis(spec, {"r"}, {"g"}, max_delay=2)
        assert analysis["realizable"]
        assert analysis["min_delay"] == 0

    def test_analysis_unrealizable(self):
        spec = LTLFalse()
        analysis = enhanced_delay_analysis(spec, {"r"}, {"g"}, max_delay=1)
        assert not analysis["realizable"]
        assert analysis["recommendation"] == "unrealizable_at_max_delay"

    def test_analysis_no_delay_needed(self):
        g = Atom("g")
        spec = Globally(g)
        analysis = enhanced_delay_analysis(spec, {"r"}, {"g"}, max_delay=2)
        assert analysis["recommendation"] == "no_delay_needed"

    def test_analysis_has_node_counts(self):
        spec = LTLFalse()
        analysis = enhanced_delay_analysis(spec, {"r"}, {"g"}, max_delay=1)
        assert "bdd_node_counts" in analysis
        assert len(analysis["bdd_node_counts"]) > 0

    def test_analysis_has_timing(self):
        spec = LTLTrue()
        analysis = enhanced_delay_analysis(spec, {"r"}, {"g"}, max_delay=1)
        assert analysis["total_time_ms"] >= 0


# ============================================================
# Test Summaries
# ============================================================

class TestSummaries:
    def test_optimization_summary(self):
        result = OptimizationResult(realizable=True, delay=1, method="symbolic",
                                     arena_bdd_nodes=42, explicit_states_equivalent=128)
        s = optimization_summary(result)
        assert "Realizable: True" in s
        assert "Delay: 1" in s
        assert "42" in s

    def test_optimization_summary_with_controller(self):
        from reactive_synthesis import MealyMachine
        ctrl = MealyMachine(
            states={0}, initial=0, inputs=[frozenset()],
            outputs=[frozenset()], transitions={(0, frozenset()): (0, frozenset())}
        )
        result = OptimizationResult(realizable=True, delay=0, controller=ctrl)
        s = optimization_summary(result)
        assert "Controller" in s

    def test_incremental_summary(self):
        r0 = OptimizationResult(realizable=True, delay=0, arena_bdd_nodes=10)
        result = IncrementalResult(realizable=True, min_delay=0,
                                    results={0: r0}, searched_delays=[0],
                                    total_time_ms=5.0, reuse_ratio=0.5)
        s = incremental_summary(result)
        assert "Minimum delay: 0" in s
        assert "REAL" in s


# ============================================================
# Test Monotonicity Property
# ============================================================

class TestMonotonicity:
    def test_realizable_stays_realizable(self):
        """If realizable at delay k, must be realizable at k+1."""
        g = Atom("g")
        spec = Globally(g)
        r0 = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        r1 = symbolic_synthesize(spec, {"r"}, {"g"}, delay=1)
        if r0.realizable:
            assert r1.realizable, "Monotonicity violated: realizable at k=0 but not k=1"

    def test_unrealizable_at_all_delays(self):
        """False spec should be unrealizable at all delays."""
        spec = LTLFalse()
        for k in range(3):
            r = symbolic_synthesize(spec, {"r"}, {"g"}, delay=k)
            assert not r.realizable


# ============================================================
# Test Agreement with V193
# ============================================================

class TestAgreementWithV193:
    def test_true_agrees(self):
        spec = LTLTrue()
        sym = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        exp = v193_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert sym.realizable == exp.realizable

    def test_false_agrees(self):
        spec = LTLFalse()
        sym = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        exp = v193_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert sym.realizable == exp.realizable

    def test_safety_agrees_delay0(self):
        g = Atom("g")
        spec = Globally(g)
        sym = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        exp = v193_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert sym.realizable == exp.realizable

    def test_safety_agrees_delay1(self):
        g = Atom("g")
        spec = Globally(g)
        sym = symbolic_synthesize(spec, {"r"}, {"g"}, delay=1)
        exp = v193_synthesize(spec, {"r"}, {"g"}, delay=1)
        assert sym.realizable == exp.realizable


# ============================================================
# Test Edge Cases
# ============================================================

class TestEdgeCases:
    def test_no_env_vars(self):
        """System has full control -- should always be realizable for safety."""
        g = Atom("g")
        spec = Globally(g)
        result = symbolic_synthesize(spec, set(), {"g"}, delay=0)
        assert result.realizable

    def test_no_sys_vars(self):
        """System has no output -- pure observation."""
        r = Atom("r")
        spec = Globally(r)  # sys cannot control r
        result = symbolic_synthesize(spec, {"r"}, set(), delay=0)
        # Unrealizable: sys cannot guarantee G(r) when env controls r
        assert not result.realizable

    def test_multiple_env_sys_vars(self):
        """Multiple variables in both sets."""
        g1 = Atom("g1")
        spec = Globally(g1)  # only need g1=true always
        result = symbolic_synthesize(spec, {"r1", "r2"}, {"g1", "g2"}, delay=0)
        assert result.realizable

    def test_delay0_no_buffer(self):
        """Delay 0 should not use buffer (minimal state space)."""
        spec = LTLTrue()
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=0)
        assert result.delay == 0
        assert result.realizable


# ============================================================
# Test Full Pipeline
# ============================================================

class TestFullPipeline:
    def test_build_reduce_solve(self):
        """Full pipeline: build arena -> reduce -> solve."""
        g = Atom("g")
        gba = ltl_to_gba(Globally(g))
        nba = gba_to_nba(gba)

        arena = build_symbolic_arena(nba, {"r"}, {"g"}, delay=1)
        reduced, report = reduce_arena(arena)
        sys_win, _ = symbolic_parity_solve(reduced)
        assert sys_win is not None

    def test_incremental_then_synthesize(self):
        """Find minimum delay, then synthesize at that delay."""
        g = Atom("g")
        spec = Globally(g)
        inc = incremental_find_minimum_delay(spec, {"r"}, {"g"}, max_delay=2)
        assert inc.realizable
        result = symbolic_synthesize(spec, {"r"}, {"g"}, delay=inc.min_delay)
        assert result.realizable

    def test_analysis_pipeline(self):
        """Run full analysis pipeline."""
        g = Atom("g")
        spec = Globally(g)
        analysis = enhanced_delay_analysis(spec, {"r"}, {"g"}, max_delay=2)
        assert analysis["realizable"]
        sizes = compare_arena_sizes(spec, {"r"}, {"g"}, [0, 1])
        assert len(sizes) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
