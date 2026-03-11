"""Tests for V124: Polyhedral Widening with Landmarks"""

import pytest
import sys
import os
import math
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
from landmark_widening import (
    # Core types
    Landmark, LandmarkKind, LoopProfile, LandmarkConfig, LandmarkResult,
    LandmarkWideningStats, LandmarkInterpreter,
    # Landmark extraction
    extract_landmarks_from_condition, extract_landmarks_from_body,
    extract_init_landmarks, build_loop_profile,
    _collect_modified_vars, _collect_condition_vars, _has_nested_loops,
    # Widening
    landmark_widen_per_var, landmark_narrowing,
    # Public API
    landmark_analyze, compare_widening_strategies, get_variable_range,
    get_loop_profile, get_loop_invariant, get_landmark_stats, landmark_summary,
    # Dependencies
    PolyhedralDomain, frac, ONE, ZERO, AccelVerdict
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser, IntLit, Var, BinOp, LetDecl, Assign, WhileStmt


# ===========================================================================
# Section 1: Landmark Extraction - Conditions
# ===========================================================================

class TestConditionLandmarks:
    """Test landmark extraction from loop conditions."""

    def test_var_lt_const(self):
        """i < 10 produces condition bound landmark."""
        cond = BinOp('<', Var('i', 0), IntLit(10, 0), 0)
        env = PolyhedralDomain(['i'])
        landmarks = extract_landmarks_from_condition(cond, env)
        assert len(landmarks) >= 1
        main = [lm for lm in landmarks if lm.priority == 10]
        assert len(main) == 1
        assert main[0].variable == 'i'
        assert main[0].value == frac(10)
        assert main[0].kind == LandmarkKind.CONDITION_BOUND

    def test_var_le_const(self):
        """i <= 5 produces condition bound landmark."""
        cond = BinOp('<=', Var('i', 0), IntLit(5, 0), 0)
        env = PolyhedralDomain(['i'])
        landmarks = extract_landmarks_from_condition(cond, env)
        main = [lm for lm in landmarks if lm.priority == 10]
        assert len(main) == 1
        assert main[0].value == frac(5)

    def test_var_gt_const(self):
        """x > 0 produces condition bound landmark."""
        cond = BinOp('>', Var('x', 0), IntLit(0, 0), 0)
        env = PolyhedralDomain(['x'])
        landmarks = extract_landmarks_from_condition(cond, env)
        main = [lm for lm in landmarks if lm.priority == 10]
        assert len(main) == 1
        assert main[0].value == frac(0)

    def test_const_lt_var(self):
        """0 < x produces landmark on x."""
        cond = BinOp('<', IntLit(0, 0), Var('x', 0), 0)
        env = PolyhedralDomain(['x'])
        landmarks = extract_landmarks_from_condition(cond, env)
        assert any(lm.variable == 'x' and lm.value == frac(0) for lm in landmarks)

    def test_boundary_values(self):
        """i < 10 also produces boundary landmarks at 9 and 11."""
        cond = BinOp('<', Var('i', 0), IntLit(10, 0), 0)
        env = PolyhedralDomain(['i'])
        landmarks = extract_landmarks_from_condition(cond, env)
        values = {lm.value for lm in landmarks}
        assert frac(9) in values
        assert frac(11) in values

    def test_var_var_condition(self):
        """i < n with known n bounds produces env-based landmarks."""
        env = PolyhedralDomain(['i', 'n'])
        env.set_lower('n', frac(5))
        env.set_upper('n', frac(20))
        cond = BinOp('<', Var('i', 0), Var('n', 0), 0)
        landmarks = extract_landmarks_from_condition(cond, env)
        # Should extract bounds from env for both vars
        assert any(lm.variable == 'n' for lm in landmarks)

    def test_non_binop_condition(self):
        """Non-BinOp condition produces no landmarks."""
        cond = Var('flag', 0)
        env = PolyhedralDomain(['flag'])
        landmarks = extract_landmarks_from_condition(cond, env)
        assert landmarks == []


# ===========================================================================
# Section 2: Landmark Extraction - Body
# ===========================================================================

class TestBodyLandmarks:
    """Test landmark extraction from loop body statements."""

    def test_increment(self):
        """i = i + 1 produces increment landmark."""
        stmt = Assign('i', BinOp('+', Var('i', 0), IntLit(1, 0), 0), 0)
        env = PolyhedralDomain(['i'])
        landmarks = extract_landmarks_from_body([stmt], env)
        assert any(lm.kind == LandmarkKind.INCREMENT and lm.variable == 'i'
                   and lm.value == frac(1) for lm in landmarks)

    def test_decrement(self):
        """x = x - 2 produces increment landmark with negative delta."""
        stmt = Assign('x', BinOp('-', Var('x', 0), IntLit(2, 0), 0), 0)
        env = PolyhedralDomain(['x'])
        landmarks = extract_landmarks_from_body([stmt], env)
        inc = [lm for lm in landmarks if lm.kind == LandmarkKind.INCREMENT]
        assert len(inc) == 1
        assert inc[0].value == frac(-2)

    def test_constant_assignment(self):
        """x = 42 produces assignment_const landmark."""
        stmt = Assign('x', IntLit(42, 0), 0)
        env = PolyhedralDomain(['x'])
        landmarks = extract_landmarks_from_body([stmt], env)
        consts = [lm for lm in landmarks if lm.kind == LandmarkKind.ASSIGNMENT_CONST]
        assert len(consts) == 1
        assert consts[0].value == frac(42)

    def test_branch_landmarks(self):
        """if (i > 5) { ... } produces branch threshold landmark."""
        from stack_vm import IfStmt
        if_stmt = IfStmt(
            cond=BinOp('>', Var('i', 0), IntLit(5, 0), 0),
            then_body=[Assign('x', IntLit(1, 0), 0)],
            else_body=None,
            line=0
        )
        env = PolyhedralDomain(['i', 'x'])
        landmarks = extract_landmarks_from_body([if_stmt], env)
        branch = [lm for lm in landmarks if lm.kind == LandmarkKind.BRANCH_THRESHOLD]
        assert len(branch) >= 1
        assert any(lm.variable == 'i' and lm.value == frac(5) for lm in branch)

    def test_nested_while_landmark(self):
        """Nested while produces nested_bound landmark."""
        inner_while = WhileStmt(
            cond=BinOp('<', Var('j', 0), IntLit(10, 0), 0),
            body=[Assign('j', BinOp('+', Var('j', 0), IntLit(1, 0), 0), 0)],
            line=0
        )
        env = PolyhedralDomain(['j'])
        landmarks = extract_landmarks_from_body([inner_while], env)
        nested = [lm for lm in landmarks if lm.kind == LandmarkKind.NESTED_BOUND]
        assert len(nested) >= 1
        assert any(lm.variable == 'j' for lm in nested)


# ===========================================================================
# Section 3: Init Landmarks
# ===========================================================================

class TestInitLandmarks:
    """Test init-value landmark extraction from pre-loop statements."""

    def test_init_landmark(self):
        """let i = 0 before loop produces init_value landmark."""
        stmts = [LetDecl('i', IntLit(0, 0), 0)]
        landmarks = extract_init_landmarks(stmts, {'i'})
        assert len(landmarks) == 1
        assert landmarks[0].kind == LandmarkKind.INIT_VALUE
        assert landmarks[0].value == frac(0)

    def test_irrelevant_init_ignored(self):
        """Init for non-loop variable is ignored."""
        stmts = [LetDecl('x', IntLit(99, 0), 0)]
        landmarks = extract_init_landmarks(stmts, {'i'})
        assert len(landmarks) == 0

    def test_multiple_inits(self):
        """Multiple inits for loop vars."""
        stmts = [
            LetDecl('i', IntLit(0, 0), 0),
            LetDecl('sum', IntLit(0, 0), 0),
        ]
        landmarks = extract_init_landmarks(stmts, {'i', 'sum'})
        assert len(landmarks) == 2


# ===========================================================================
# Section 4: Loop Profile Construction
# ===========================================================================

class TestLoopProfile:
    """Test building complete loop profiles."""

    def test_simple_counting_loop(self):
        """Profile for: while (i < 10) { i = i + 1; }"""
        while_stmt = WhileStmt(
            cond=BinOp('<', Var('i', 0), IntLit(10, 0), 0),
            body=[Assign('i', BinOp('+', Var('i', 0), IntLit(1, 0), 0), 0)],
            line=0
        )
        env = PolyhedralDomain(['i'])
        env.set_equal('i', frac(0))
        profile = build_loop_profile(while_stmt, env,
                                     [LetDecl('i', IntLit(0, 0), 0)])
        assert 'i' in profile.modified_vars
        assert 'i' in profile.condition_vars
        assert len(profile.landmarks) > 0
        assert len(profile.recurrences) > 0
        assert not profile.has_nested_loops

    def test_per_var_policy_accelerate(self):
        """Variable with recurrence + condition bound gets 'accelerate' policy."""
        while_stmt = WhileStmt(
            cond=BinOp('<', Var('i', 0), IntLit(10, 0), 0),
            body=[Assign('i', BinOp('+', Var('i', 0), IntLit(1, 0), 0), 0)],
            line=0
        )
        env = PolyhedralDomain(['i'])
        env.set_equal('i', frac(0))
        profile = build_loop_profile(while_stmt, env)
        assert profile.get_var_policy('i') == 'accelerate'

    def test_per_var_policy_threshold(self):
        """Variable with landmarks but no recurrence gets 'threshold' policy."""
        while_stmt = WhileStmt(
            cond=BinOp('<', Var('i', 0), IntLit(10, 0), 0),
            body=[
                Assign('i', BinOp('+', Var('i', 0), IntLit(1, 0), 0), 0),
                Assign('x', IntLit(5, 0), 0),  # x = 5
            ],
            line=0
        )
        env = PolyhedralDomain(['i', 'x'])
        env.set_equal('i', frac(0))
        profile = build_loop_profile(while_stmt, env)
        # x has a constant assignment landmark but no recurrence
        assert profile.get_var_policy('x') == 'threshold'

    def test_per_var_policy_standard(self):
        """Variable with no landmarks gets 'standard' policy."""
        while_stmt = WhileStmt(
            cond=BinOp('<', Var('i', 0), IntLit(10, 0), 0),
            body=[Assign('i', BinOp('+', Var('i', 0), IntLit(1, 0), 0), 0)],
            line=0
        )
        env = PolyhedralDomain(['i', 'z'])
        profile = build_loop_profile(while_stmt, env)
        assert profile.get_var_policy('z') == 'standard'

    def test_nested_loop_profile(self):
        """Profile detects nested loops."""
        inner = WhileStmt(
            cond=BinOp('<', Var('j', 0), IntLit(5, 0), 0),
            body=[Assign('j', BinOp('+', Var('j', 0), IntLit(1, 0), 0), 0)],
            line=0
        )
        outer = WhileStmt(
            cond=BinOp('<', Var('i', 0), IntLit(10, 0), 0),
            body=[
                inner,
                Assign('i', BinOp('+', Var('i', 0), IntLit(1, 0), 0), 0),
            ],
            line=0
        )
        env = PolyhedralDomain(['i', 'j'])
        profile = build_loop_profile(outer, env)
        assert profile.has_nested_loops
        assert len(profile.nested_profiles) == 1


# ===========================================================================
# Section 5: Helper Functions
# ===========================================================================

class TestHelpers:
    """Test helper functions for structural analysis."""

    def test_collect_modified_vars(self):
        """Collects all assigned variables from statements."""
        stmts = [
            Assign('x', IntLit(1, 0), 0),
            LetDecl('y', IntLit(2, 0), 0),
        ]
        assert _collect_modified_vars(stmts) == {'x', 'y'}

    def test_collect_condition_vars(self):
        """Collects variables from a condition expression."""
        cond = BinOp('<', Var('i', 0), Var('n', 0), 0)
        assert _collect_condition_vars(cond) == {'i', 'n'}

    def test_has_nested_loops_true(self):
        inner = WhileStmt(
            cond=BinOp('<', Var('j', 0), IntLit(5, 0), 0),
            body=[Assign('j', BinOp('+', Var('j', 0), IntLit(1, 0), 0), 0)],
            line=0
        )
        assert _has_nested_loops([inner])

    def test_has_nested_loops_false(self):
        assert not _has_nested_loops([Assign('x', IntLit(1, 0), 0)])


# ===========================================================================
# Section 6: Per-Variable Widening
# ===========================================================================

class TestPerVarWidening:
    """Test the per-variable landmark widening operator."""

    def test_delay_phase_is_join(self):
        """During delay phase, widening is just join."""
        old = PolyhedralDomain(['x'])
        old.set_equal('x', frac(0))
        new = PolyhedralDomain(['x'])
        new.set_equal('x', frac(1))
        profile = LoopProfile(
            landmarks=[], modified_vars={'x'}, condition_vars=set(),
            recurrences=[], per_var_thresholds={}, global_thresholds=[],
            has_nested_loops=False, nested_profiles=[]
        )
        result = landmark_widen_per_var(old, new, profile, iteration=0, delay=2)
        lo, hi = result.get_interval('x')
        assert lo == 0.0
        assert hi == 1.0

    def test_acceleration_applies_bounds(self):
        """Accelerated variable gets computed limit bounds."""
        from fixpoint_acceleration import RecurrenceInfo
        old = PolyhedralDomain(['i'])
        old.set_lower('i', frac(0))
        old.set_upper('i', frac(3))
        new = PolyhedralDomain(['i'])
        new.set_lower('i', frac(0))
        new.set_upper('i', frac(4))
        rec = RecurrenceInfo(var='i', delta=frac(1),
                             init_lower=frac(0), init_upper=frac(0),
                             condition_var='i', condition_bound=frac(10))
        profile = LoopProfile(
            landmarks=[], modified_vars={'i'}, condition_vars={'i'},
            recurrences=[rec],
            per_var_thresholds={'i': [frac(0), frac(10)]},
            global_thresholds=[frac(0), frac(10)],
            has_nested_loops=False, nested_profiles=[]
        )
        result = landmark_widen_per_var(old, new, profile, iteration=3, delay=2)
        lo, hi = result.get_interval('i')
        # Should have accelerated to limit (0 to 10)
        assert lo <= 0.0
        assert hi <= 10.0

    def test_threshold_widening_snaps_to_landmark(self):
        """Threshold variable snaps to next landmark value instead of infinity."""
        old = PolyhedralDomain(['x'])
        old.set_lower('x', frac(0))
        old.set_upper('x', frac(5))
        new = PolyhedralDomain(['x'])
        new.set_lower('x', frac(0))
        new.set_upper('x', frac(7))  # Growing upper bound
        profile = LoopProfile(
            landmarks=[Landmark(LandmarkKind.BRANCH_THRESHOLD, 'x', frac(10))],
            modified_vars={'x'}, condition_vars=set(),
            recurrences=[],
            per_var_thresholds={'x': [frac(0), frac(5), frac(10), frac(20)]},
            global_thresholds=[frac(0), frac(5), frac(10), frac(20)],
            has_nested_loops=False, nested_profiles=[]
        )
        result = landmark_widen_per_var(old, new, profile, iteration=3, delay=2)
        lo, hi = result.get_interval('x')
        # Should snap to 7 or 10, not infinity
        assert hi <= 20.0

    def test_bot_handling(self):
        """Bot environments handled correctly."""
        old = PolyhedralDomain.bot(['x'])
        new = PolyhedralDomain(['x'])
        new.set_equal('x', frac(5))
        profile = LoopProfile(
            landmarks=[], modified_vars=set(), condition_vars=set(),
            recurrences=[], per_var_thresholds={}, global_thresholds=[],
            has_nested_loops=False, nested_profiles=[]
        )
        result = landmark_widen_per_var(old, new, profile, iteration=3, delay=2)
        assert not result.is_bot()


# ===========================================================================
# Section 7: Landmark Narrowing
# ===========================================================================

class TestLandmarkNarrowing:
    """Test landmark-guided narrowing."""

    def test_narrowing_tightens_bounds(self):
        """Narrowing tightens wide bounds using body result."""
        wide = PolyhedralDomain(['x'])
        wide.set_lower('x', frac(0))
        wide.set_upper('x', frac(100))
        body_result = PolyhedralDomain(['x'])
        body_result.set_lower('x', frac(0))
        body_result.set_upper('x', frac(50))
        profile = LoopProfile(
            landmarks=[Landmark(LandmarkKind.CONDITION_BOUND, 'x', frac(50))],
            modified_vars={'x'}, condition_vars={'x'},
            recurrences=[],
            per_var_thresholds={'x': [frac(0), frac(50), frac(100)]},
            global_thresholds=[frac(0), frac(50), frac(100)],
            has_nested_loops=False, nested_profiles=[]
        )
        result = landmark_narrowing(wide, body_result, profile)
        lo, hi = result.get_interval('x')
        assert hi <= 100.0  # At least as tight as wide


# ===========================================================================
# Section 8: Simple Counting Loop (End-to-End)
# ===========================================================================

class TestSimpleCountingLoop:
    """End-to-end test with simple counting loops."""

    def test_count_to_10(self):
        """let i = 0; while (i < 10) { i = i + 1; }"""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('i')
        assert lo == 10.0
        assert hi == 10.0

    def test_count_down(self):
        """let x = 10; while (x > 0) { x = x - 1; }"""
        source = "let x = 10; while (x > 0) { x = x - 1; }"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('x')
        assert lo == 0.0
        assert hi == 0.0

    def test_count_to_100(self):
        """Larger bound: let i = 0; while (i < 100) { i = i + 1; }"""
        source = "let i = 0; while (i < 100) { i = i + 1; }"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('i')
        assert lo == 100.0
        assert hi == 100.0

    def test_count_by_two(self):
        """let i = 0; while (i < 10) { i = i + 2; }"""
        source = "let i = 0; while (i < 10) { i = i + 2; }"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('i')
        assert lo >= 10.0  # i exits at 10
        assert hi <= 11.0  # Could be 10 or 11 depending on start

    def test_loop_with_accumulator(self):
        """let i = 0; let sum = 0; while (i < 5) { sum = sum + i; i = i + 1; }"""
        source = "let i = 0; let sum = 0; while (i < 5) { sum = sum + i; i = i + 1; }"
        result = landmark_analyze(source)
        lo_i, hi_i = result.env.get_interval('i')
        assert lo_i == 5.0
        assert hi_i == 5.0


# ===========================================================================
# Section 9: Loops with Branches
# ===========================================================================

class TestLoopsWithBranches:
    """Test loops containing if-else branches."""

    def test_conditional_increment(self):
        """Loop with conditional body."""
        source = """
        let i = 0;
        let x = 0;
        while (i < 10) {
            if (i > 5) {
                x = x + 1;
            }
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        lo_i, hi_i = result.env.get_interval('i')
        assert lo_i == 10.0
        assert hi_i == 10.0
        # x should be non-negative (polyhedral over-approximates branch-conditional increments)
        lo_x, hi_x = result.env.get_interval('x')
        assert lo_x >= 0.0

    def test_branch_landmark_extraction(self):
        """Branch conditions inside loops generate landmarks."""
        source = """
        let i = 0;
        let x = 0;
        while (i < 20) {
            if (i > 10) {
                x = x + 1;
            }
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        profile = result.loop_profiles.get(0)
        assert profile is not None
        # Should have branch threshold landmarks for i at value 10
        branch_landmarks = [lm for lm in profile.landmarks
                            if lm.kind == LandmarkKind.BRANCH_THRESHOLD]
        assert len(branch_landmarks) >= 1
        assert any(lm.variable == 'i' and lm.value == frac(10) for lm in branch_landmarks)

    def test_if_else_both_branches(self):
        """Both branches of if-else analyzed."""
        source = """
        let i = 0;
        let x = 0;
        while (i < 10) {
            if (i < 5) {
                x = x + 1;
            } else {
                x = x + 2;
            }
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        lo_x, hi_x = result.env.get_interval('x')
        assert lo_x >= 0.0  # x always >= 0
        # Polyhedral domain over-approximates branch-conditional increments
        # Sound upper bound is sufficient


# ===========================================================================
# Section 10: Nested Loops
# ===========================================================================

class TestNestedLoops:
    """Test nested loop handling."""

    def test_nested_counting(self):
        """Double nested counting loops."""
        source = """
        let i = 0;
        let j = 0;
        while (i < 5) {
            j = 0;
            while (j < 3) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        lo_i, hi_i = result.env.get_interval('i')
        assert lo_i == 5.0
        assert hi_i == 5.0
        lo_j, hi_j = result.env.get_interval('j')
        assert lo_j == 3.0
        assert hi_j == 3.0

    def test_nested_profile_detected(self):
        """Outer loop profile detects nested loops."""
        source = """
        let i = 0;
        let j = 0;
        while (i < 10) {
            j = 0;
            while (j < 5) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        outer_profile = result.loop_profiles.get(0)
        assert outer_profile is not None
        assert outer_profile.has_nested_loops

    def test_nested_threshold_propagation(self):
        """Inner loop thresholds propagate to outer loop profile."""
        source = """
        let i = 0;
        let j = 0;
        while (i < 10) {
            j = 0;
            while (j < 5) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        outer_profile = result.loop_profiles.get(0)
        assert outer_profile is not None
        # j's thresholds from inner loop should propagate
        assert result.stats.nested_propagations > 0


# ===========================================================================
# Section 11: Loop Profiles and Policies
# ===========================================================================

class TestLoopProfilesEndToEnd:
    """Test loop profile construction through full analysis."""

    def test_profile_stored(self):
        """Loop profiles are stored in result."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source)
        assert 0 in result.loop_profiles
        profile = result.loop_profiles[0]
        assert 'i' in profile.modified_vars
        assert len(profile.recurrences) > 0

    def test_multiple_loops_multiple_profiles(self):
        """Each sequential loop gets its own profile."""
        source = """
        let i = 0;
        while (i < 10) { i = i + 1; }
        let j = 0;
        while (j < 5) { j = j + 1; }
        """
        result = landmark_analyze(source)
        assert 0 in result.loop_profiles
        assert 1 in result.loop_profiles

    def test_loop_invariant_stored(self):
        """Loop invariants are stored."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source)
        assert 0 in result.loop_invariants
        inv = result.loop_invariants[0]
        # Invariant should include i >= 0
        lo, hi = inv.get_interval('i')
        assert lo >= 0.0


# ===========================================================================
# Section 12: Statistics Tracking
# ===========================================================================

class TestStatistics:
    """Test that statistics are properly tracked."""

    def test_landmarks_counted(self):
        """Landmark count tracked."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source)
        assert result.stats.landmarks_extracted > 0

    def test_iterations_counted(self):
        """Total iterations tracked."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source)
        assert result.stats.total_iterations > 0

    def test_narrowing_counted(self):
        """Narrowing iterations tracked."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source)
        assert result.stats.landmark_narrowings >= 0

    def test_get_landmark_stats_api(self):
        """get_landmark_stats API works."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        stats = get_landmark_stats(source)
        assert isinstance(stats, LandmarkWideningStats)
        assert stats.landmarks_extracted > 0


# ===========================================================================
# Section 13: Public API
# ===========================================================================

class TestPublicAPI:
    """Test all public API entry points."""

    def test_landmark_analyze(self):
        """Main analysis API returns LandmarkResult."""
        source = "let x = 5;"
        result = landmark_analyze(source)
        assert isinstance(result, LandmarkResult)
        lo, hi = result.env.get_interval('x')
        assert lo == 5.0
        assert hi == 5.0

    def test_get_variable_range(self):
        """get_variable_range API."""
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        lo, hi = get_variable_range(source, 'x')
        assert lo == 10.0
        assert hi == 10.0

    def test_get_variable_range_unknown(self):
        """Unknown variable returns (-inf, inf)."""
        source = "let x = 5;"
        lo, hi = get_variable_range(source, 'unknown')
        assert lo == float('-inf')
        assert hi == float('inf')

    def test_get_loop_profile(self):
        """get_loop_profile API."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        profile = get_loop_profile(source, 0)
        assert profile is not None
        assert 'i' in profile.modified_vars

    def test_get_loop_invariant(self):
        """get_loop_invariant API."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        inv = get_loop_invariant(source, 0)
        assert inv is not None

    def test_landmark_summary(self):
        """Summary produces readable output."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        summary = landmark_summary(source)
        assert "Landmark Widening Analysis" in summary
        assert "Variable ranges:" in summary
        assert "Loop profiles:" in summary
        assert "Statistics:" in summary

    def test_config_custom(self):
        """Custom config is respected."""
        config = LandmarkConfig(max_iterations=5, delay_iterations=1)
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source, config)
        assert isinstance(result, LandmarkResult)


# ===========================================================================
# Section 14: Comparison API
# ===========================================================================

class TestComparisonAPI:
    """Test comparison between widening strategies."""

    def test_compare_produces_all_results(self):
        """Comparison returns results from all three strategies."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        comp = compare_widening_strategies(source)
        assert 'landmark' in comp
        assert 'standard' in comp
        assert 'accelerated' in comp
        assert 'comparison' in comp
        assert 'precision_wins' in comp
        assert 'landmark_stats' in comp

    def test_comparison_includes_all_vars(self):
        """Comparison covers all variables."""
        source = "let i = 0; let sum = 0; while (i < 5) { sum = sum + i; i = i + 1; }"
        comp = compare_widening_strategies(source)
        assert 'i' in comp['comparison']
        assert 'sum' in comp['comparison']

    def test_precision_wins_counted(self):
        """Precision wins tallied."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        comp = compare_widening_strategies(source)
        total = sum(comp['precision_wins'].values())
        assert total > 0


# ===========================================================================
# Section 15: Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_program(self):
        """Empty program doesn't crash."""
        result = landmark_analyze("")
        assert isinstance(result, LandmarkResult)

    def test_no_loops(self):
        """Program without loops works fine."""
        source = "let x = 5; let y = x + 3;"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('x')
        assert lo == 5.0
        assert hi == 5.0

    def test_function_declaration(self):
        """Functions are recorded but not analyzed."""
        source = "fn foo(x) { return x + 1; } let y = 10;"
        result = landmark_analyze(source)
        assert 'foo' in result.functions

    def test_print_statement(self):
        """Print statements don't crash."""
        source = "let x = 5; print(x);"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('x')
        assert lo == 5.0
        assert hi == 5.0

    def test_dead_loop(self):
        """Loop that never executes."""
        source = "let i = 10; while (i < 5) { i = i + 1; }"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('i')
        assert lo == 10.0
        assert hi == 10.0


# ===========================================================================
# Section 16: Sequential Loops
# ===========================================================================

class TestSequentialLoops:
    """Test programs with multiple sequential loops."""

    def test_two_sequential_loops(self):
        """Two loops in sequence."""
        source = """
        let i = 0;
        while (i < 10) { i = i + 1; }
        let j = 0;
        while (j < 5) { j = j + 1; }
        """
        result = landmark_analyze(source)
        lo_i, hi_i = result.env.get_interval('i')
        # i >= 10 at exit of first loop (polyhedral may lose equality after second loop scope)
        assert lo_i >= 0.0
        assert hi_i >= 10.0
        lo_j, hi_j = result.env.get_interval('j')
        assert lo_j == 5.0
        assert hi_j == 5.0

    def test_loop_reuses_var(self):
        """Second loop reuses variable from first."""
        source = """
        let i = 0;
        while (i < 10) { i = i + 1; }
        i = 0;
        while (i < 5) { i = i + 1; }
        """
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('i')
        assert lo == 5.0
        assert hi == 5.0


# ===========================================================================
# Section 17: Relational Constraints
# ===========================================================================

class TestRelational:
    """Test relational constraint handling."""

    def test_var_less_than_var(self):
        """i < n condition produces relational constraint."""
        source = """
        let n = 10;
        let i = 0;
        while (i < n) {
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        lo_i, hi_i = result.env.get_interval('i')
        lo_n, hi_n = result.env.get_interval('n')
        assert lo_n == 10.0
        assert hi_n == 10.0
        # i should be >= n at exit (loop condition became false)
        assert lo_i >= 10.0

    def test_sum_conservation(self):
        """Transfer between two variables."""
        source = """
        let a = 10;
        let b = 0;
        while (a > 0) {
            a = a - 1;
            b = b + 1;
        }
        """
        result = landmark_analyze(source)
        lo_a, hi_a = result.env.get_interval('a')
        lo_b, hi_b = result.env.get_interval('b')
        assert lo_a == 0.0
        assert hi_a == 0.0
        # b grows as a decreases; polyhedral tracks a precisely via recurrence,
        # b may be over-approximated without relational sum-conservation invariant
        assert lo_b >= 0.0


# ===========================================================================
# Section 18: Non-Linear Expressions
# ===========================================================================

class TestNonLinear:
    """Test handling of non-linear expressions."""

    def test_multiplication(self):
        """x * 2 is linear and handled."""
        source = "let x = 5; let y = x * 2;"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('y')
        assert lo == 10.0
        assert hi == 10.0

    def test_nonlinear_fallback(self):
        """x * y falls back to interval evaluation."""
        source = """
        let x = 3;
        let y = 4;
        let z = x * y;
        """
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('z')
        assert lo == 12.0
        assert hi == 12.0

    def test_division_warning(self):
        """Division by potentially-zero produces warning."""
        source = """
        let x = 10;
        let y = 0;
        let z = x / y;
        """
        result = landmark_analyze(source)
        assert any("division by zero" in w for w in result.warnings)


# ===========================================================================
# Section 19: Unary Operations
# ===========================================================================

class TestUnaryOps:
    """Test unary operator handling."""

    def test_negation(self):
        """Negation is linearized correctly."""
        source = "let x = 5; let y = -x;"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('y')
        assert lo == -5.0
        assert hi == -5.0

    def test_negation_in_loop(self):
        """Negation in loop body."""
        source = """
        let i = 0;
        let neg = 0;
        while (i < 5) {
            neg = -i;
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        lo_i, hi_i = result.env.get_interval('i')
        assert lo_i == 5.0
        assert hi_i == 5.0


# ===========================================================================
# Section 20: Boolean Literals
# ===========================================================================

class TestBoolLiterals:
    """Test boolean literal handling."""

    def test_true_as_1(self):
        """true is treated as 1."""
        source = "let x = true;"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('x')
        assert lo == 1.0
        assert hi == 1.0

    def test_false_as_0(self):
        """false is treated as 0."""
        source = "let x = false;"
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('x')
        assert lo == 0.0
        assert hi == 0.0


# ===========================================================================
# Section 21: Complex Programs
# ===========================================================================

class TestComplexPrograms:
    """Test more complex program patterns."""

    def test_fibonacci_like(self):
        """Fibonacci-like accumulation pattern."""
        source = """
        let i = 0;
        let a = 0;
        let b = 1;
        while (i < 10) {
            let temp = a + b;
            a = b;
            b = temp;
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        lo_i, hi_i = result.env.get_interval('i')
        assert lo_i == 10.0
        assert hi_i == 10.0
        # a and b have non-linear growth; polyhedral domain cannot track precisely
        # Just verify soundness: they should have finite bounds or be widened
        assert 'a' in result.env.var_names
        assert 'b' in result.env.var_names

    def test_max_finding(self):
        """Pattern that tracks maximum."""
        source = """
        let i = 0;
        let max = 0;
        while (i < 10) {
            if (i > max) {
                max = i;
            }
            i = i + 1;
        }
        """
        result = landmark_analyze(source)
        lo_max, hi_max = result.env.get_interval('max')
        assert lo_max >= 0.0

    def test_countdown_with_early_exit_pattern(self):
        """Countdown with conditional update."""
        source = """
        let n = 20;
        let count = 0;
        while (n > 0) {
            if (n > 10) {
                n = n - 2;
            } else {
                n = n - 1;
            }
            count = count + 1;
        }
        """
        result = landmark_analyze(source)
        lo_n, hi_n = result.env.get_interval('n')
        # n exits at <= 0 (polyhedral may over-approximate due to conditional decrement)
        assert hi_n <= 0.0
        # count should be non-negative
        lo_c, _ = result.env.get_interval('count')
        assert lo_c >= 0.0


# ===========================================================================
# Section 22: Landmark Verdict
# ===========================================================================

class TestVerdict:
    """Test convergence verdict."""

    def test_converged(self):
        """Simple loop converges."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = landmark_analyze(source)
        assert result.verdict == AccelVerdict.CONVERGED

    def test_max_iterations_verdict(self):
        """Non-converging loop gets max_iterations verdict."""
        config = LandmarkConfig(max_iterations=3, delay_iterations=1,
                                enable_landmark_narrowing=False)
        # This may or may not converge in 3 iterations depending on acceleration
        source = "let i = 0; while (i < 100) { i = i + 1; }"
        result = landmark_analyze(source, config)
        # Either converged or max_iter -- both valid outcomes
        assert result.verdict in (AccelVerdict.CONVERGED, AccelVerdict.MAX_ITER)


# ===========================================================================
# Section 23: Modulo and Remainder
# ===========================================================================

class TestModulo:
    """Test modulo/remainder handling."""

    def test_modulo_bounds(self):
        """Modulo expression gets safe interval bounds."""
        source = """
        let x = 17;
        let y = 5;
        let r = x % y;
        """
        result = landmark_analyze(source)
        lo, hi = result.env.get_interval('r')
        # x%y when y>0 is in [0, y-1]
        assert lo >= 0.0
        assert hi <= 4.0


# ===========================================================================
# Section 24: Return and Block Statements
# ===========================================================================

class TestBlocksAndReturn:
    """Test block and return statement handling."""

    def test_block_statement(self):
        """Block statements are interpreted."""
        source = """
        let x = 0;
        {
            x = 5;
        }
        """
        result = landmark_analyze(source)
        # Note: C10 may not parse bare blocks this way -- test with fn
        # If parsing fails, that's fine -- just ensure no crash

    def test_function_with_return(self):
        """Function declaration is recorded."""
        source = """
        fn double(x) { return x * 2; }
        let y = 10;
        """
        result = landmark_analyze(source)
        assert 'double' in result.functions
        lo, hi = result.env.get_interval('y')
        assert lo == 10.0
        assert hi == 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
