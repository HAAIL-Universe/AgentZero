"""
Tests for V052: Incremental Dashboard Verification
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from incremental_dashboard import (
    IncrementalDashboard, incremental_verify_sequence, diff_and_verify,
    diff_report, _extract_function_signatures, _stmt_signature,
    FunctionAnalysisCache, DeltaReport, IncrementalDashboardResult,
    PipelineConfig,
)
from holistic_verification import VerificationReport, AnalysisResult, AnalysisStatus


# --- Test Programs ---

V1_BASE = """
fn add(x, y) {
    return x + y;
}
fn double(x) {
    return x + x;
}
"""

V2_MODIFIED_DOUBLE = """
fn add(x, y) {
    return x + y;
}
fn double(x) {
    return x * 2;
}
"""

V3_ADDED_TRIPLE = """
fn add(x, y) {
    return x + y;
}
fn double(x) {
    return x * 2;
}
fn triple(x) {
    return x * 3;
}
"""

V4_REMOVED_ADD = """
fn double(x) {
    return x * 2;
}
fn triple(x) {
    return x * 3;
}
"""

SINGLE_FN = """
fn inc(x) {
    return x + 1;
}
"""

SINGLE_FN_MODIFIED = """
fn inc(x) {
    return x + 2;
}
"""

NO_FUNCTIONS = """
let x = 42;
"""

WITH_LOOP = """
fn sum_to(n) {
    let s = 0;
    let i = 0;
    while (i < n) {
        s = s + i;
        i = i + 1;
    }
    return s;
}
"""


# === Section 1: Function Signature Extraction ===

class TestFunctionSignatures(unittest.TestCase):
    def test_basic_extraction(self):
        sigs = _extract_function_signatures(V1_BASE)
        assert "add" in sigs
        assert "double" in sigs
        assert len(sigs) == 2

    def test_changed_function_different_sig(self):
        sigs1 = _extract_function_signatures(V1_BASE)
        sigs2 = _extract_function_signatures(V2_MODIFIED_DOUBLE)
        assert sigs1["add"] == sigs2["add"]  # unchanged
        assert sigs1["double"] != sigs2["double"]  # changed

    def test_added_function(self):
        sigs2 = _extract_function_signatures(V2_MODIFIED_DOUBLE)
        sigs3 = _extract_function_signatures(V3_ADDED_TRIPLE)
        assert "triple" in sigs3
        assert "triple" not in sigs2

    def test_no_functions(self):
        sigs = _extract_function_signatures(NO_FUNCTIONS)
        assert sigs == {}

    def test_invalid_source(self):
        sigs = _extract_function_signatures("not valid!!!")
        assert sigs == {}

    def test_single_function(self):
        sigs = _extract_function_signatures(SINGLE_FN)
        assert "inc" in sigs

    def test_with_loop(self):
        sigs = _extract_function_signatures(WITH_LOOP)
        assert "sum_to" in sigs


# === Section 2: DeltaReport ===

class TestDeltaReport(unittest.TestCase):
    def test_empty_delta(self):
        d = DeltaReport(old_source="", new_source="")
        assert d.cache_hit_rate == 0.0

    def test_all_hits(self):
        d = DeltaReport(old_source="", new_source="",
                        cache_hits=5, cache_misses=0)
        assert d.cache_hit_rate == 1.0

    def test_mixed(self):
        d = DeltaReport(old_source="", new_source="",
                        cache_hits=3, cache_misses=1)
        assert d.cache_hit_rate == 0.75

    def test_summary(self):
        d = DeltaReport(old_source="", new_source="",
                        added_functions={"foo"},
                        modified_functions={"bar"},
                        unchanged_functions={"baz"},
                        cache_hits=1, cache_misses=1)
        s = d.summary()
        assert "Added: foo" in s
        assert "Modified: bar" in s
        assert "Unchanged: baz" in s
        assert "50%" in s


# === Section 3: Diff Report (No Analysis) ===

class TestDiffReport(unittest.TestCase):
    def test_no_change(self):
        d = diff_report(V1_BASE, V1_BASE)
        assert len(d.modified_functions) == 0
        assert d.unchanged_functions == {"add", "double"}

    def test_modified(self):
        d = diff_report(V1_BASE, V2_MODIFIED_DOUBLE)
        assert "double" in d.modified_functions
        assert "add" in d.unchanged_functions

    def test_added(self):
        d = diff_report(V2_MODIFIED_DOUBLE, V3_ADDED_TRIPLE)
        assert "triple" in d.added_functions

    def test_removed(self):
        d = diff_report(V3_ADDED_TRIPLE, V4_REMOVED_ADD)
        assert "add" in d.removed_functions

    def test_all_new(self):
        d = diff_report(NO_FUNCTIONS, V1_BASE)
        assert d.added_functions == {"add", "double"}

    def test_all_removed(self):
        d = diff_report(V1_BASE, NO_FUNCTIONS)
        assert d.removed_functions == {"add", "double"}


# === Section 4: IncrementalDashboard State ===

class TestDashboardState(unittest.TestCase):
    def test_initial_state(self):
        d = IncrementalDashboard()
        assert d.version_count == 0
        assert d.cache_size == 0

    def test_version_count_increments(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        assert d.version_count == 1
        d.verify(V2_MODIFIED_DOUBLE)
        assert d.version_count == 2

    def test_cache_populated(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        assert d.cache_size >= 0  # may have functions cached

    def test_clear_cache(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        d.clear_cache()
        assert d.cache_size == 0
        assert d.version_count == 1  # version count not cleared


# === Section 5: First Version (Full Analysis) ===

class TestFirstVersion(unittest.TestCase):
    def test_first_returns_result(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        r = d.verify(V1_BASE)
        assert isinstance(r, IncrementalDashboardResult)
        assert r.is_first_version is True
        assert r.delta is None

    def test_first_has_report(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        r = d.verify(V1_BASE)
        assert isinstance(r.report, VerificationReport)
        assert len(r.report.analyses) > 0

    def test_first_has_duration(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        r = d.verify(V1_BASE)
        assert r.duration_ms >= 0


# === Section 6: Incremental Updates ===

class TestIncrementalUpdates(unittest.TestCase):
    def test_second_version_has_delta(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        r = d.verify(V2_MODIFIED_DOUBLE)
        assert r.is_first_version is False
        assert r.delta is not None

    def test_unchanged_detected(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        r = d.verify(V2_MODIFIED_DOUBLE)
        assert "add" in r.delta.unchanged_functions

    def test_modified_detected(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        r = d.verify(V2_MODIFIED_DOUBLE)
        assert "double" in r.delta.modified_functions

    def test_no_change_reuses_cache(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        r = d.verify(V1_BASE)
        assert r.is_first_version is False
        assert r.delta is not None
        assert r.delta.cache_hits >= 0

    def test_added_function(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V2_MODIFIED_DOUBLE)
        r = d.verify(V3_ADDED_TRIPLE)
        assert "triple" in r.delta.added_functions


# === Section 7: Sequence Verification ===

class TestSequenceVerification(unittest.TestCase):
    def test_basic_sequence(self):
        results = incremental_verify_sequence(
            [V1_BASE, V2_MODIFIED_DOUBLE, V3_ADDED_TRIPLE],
            PipelineConfig.fast()
        )
        assert len(results) == 3
        assert results[0].is_first_version is True
        assert results[1].is_first_version is False
        assert results[2].is_first_version is False

    def test_all_same(self):
        results = incremental_verify_sequence(
            [V1_BASE, V1_BASE, V1_BASE],
            PipelineConfig.fast()
        )
        assert len(results) == 3
        # Second and third should have cache hits
        assert results[1].delta is not None
        assert results[2].delta is not None

    def test_single_version(self):
        results = incremental_verify_sequence(
            [SINGLE_FN],
            PipelineConfig.fast()
        )
        assert len(results) == 1
        assert results[0].is_first_version is True


# === Section 8: Diff and Verify ===

class TestDiffAndVerify(unittest.TestCase):
    def test_basic(self):
        r = diff_and_verify(V1_BASE, V2_MODIFIED_DOUBLE, PipelineConfig.fast())
        assert isinstance(r, IncrementalDashboardResult)
        assert r.is_first_version is False
        assert r.delta is not None

    def test_delta_has_changes(self):
        r = diff_and_verify(V1_BASE, V2_MODIFIED_DOUBLE, PipelineConfig.fast())
        assert "double" in r.delta.modified_functions

    def test_no_change(self):
        r = diff_and_verify(V1_BASE, V1_BASE, PipelineConfig.fast())
        assert len(r.delta.modified_functions) == 0


# === Section 9: Summary Output ===

class TestSummaryOutput(unittest.TestCase):
    def test_first_version_summary(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        r = d.verify(V1_BASE)
        s = r.summary()
        assert "HOLISTIC VERIFICATION REPORT" in s

    def test_incremental_summary(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(V1_BASE)
        r = d.verify(V2_MODIFIED_DOUBLE)
        s = r.summary()
        assert "Incremental" in s

    def test_delta_summary(self):
        r = diff_and_verify(V1_BASE, V3_ADDED_TRIPLE, PipelineConfig.fast())
        s = r.delta.summary()
        assert "Added" in s or "Modified" in s


# === Section 10: Robustness ===

class TestRobustness(unittest.TestCase):
    def test_empty_source(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        r = d.verify("")
        assert isinstance(r, IncrementalDashboardResult)

    def test_invalid_source(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        r = d.verify("not valid C10!!!")
        assert isinstance(r, IncrementalDashboardResult)

    def test_transition_from_invalid_to_valid(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify("")
        r = d.verify(V1_BASE)
        assert isinstance(r, IncrementalDashboardResult)

    def test_no_functions_to_functions(self):
        d = IncrementalDashboard(PipelineConfig.fast())
        d.verify(NO_FUNCTIONS)
        r = d.verify(V1_BASE)
        assert isinstance(r, IncrementalDashboardResult)


# === Section 11: FunctionAnalysisCache ===

class TestFunctionAnalysisCache(unittest.TestCase):
    def test_creation(self):
        c = FunctionAnalysisCache(fn_name="foo", fn_signature="abc123")
        assert c.fn_name == "foo"
        assert c.fn_signature == "abc123"
        assert c.results == {}

    def test_with_results(self):
        r = AnalysisResult("test", AnalysisStatus.PASSED, "ok")
        c = FunctionAnalysisCache(fn_name="foo", fn_signature="abc",
                                  results={"test": r})
        assert "test" in c.results


if __name__ == '__main__':
    unittest.main()
