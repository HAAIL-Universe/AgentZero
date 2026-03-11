"""Tests for V092: Regex Repair"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V084_symbolic_regex'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V088_regex_synthesis'))

import pytest
from regex_repair import (
    diagnose_regex, repair_regex, repair_regex_targeted,
    suggest_repairs, compare_repairs, batch_repair,
    repair_from_counterexample, semantic_distance,
    _compile_and_test, _is_correct, _ast_edit_distance,
    _enumerate_subtrees, _replace_at_path,
    _quantifier_mutations, _char_class_mutations,
    _structural_mutations, _literal_mutations,
    _localize_fault, _chars_to_ranges,
    RepairResult, DiagnosticResult
)
from symbolic_regex import (
    parse_regex, regex_to_string, RLit, RDot, RClass, RStar, RPlus,
    ROptional, RConcat, RAlt, REpsilon, REmpty, regex_size
)


# ===========================================================================
# Section 1: Helpers
# ===========================================================================

class TestHelpers:
    def test_compile_and_test_correct(self):
        r = parse_regex("ab")
        ap, rn, fn, fp = _compile_and_test(r, ["ab"], ["cd"])
        assert ap and rn
        assert fn == [] and fp == []

    def test_compile_and_test_false_negative(self):
        r = parse_regex("ab")
        ap, rn, fn, fp = _compile_and_test(r, ["ab", "ac"], ["cd"])
        assert not ap
        assert "ac" in fn

    def test_compile_and_test_false_positive(self):
        r = parse_regex("a.")
        ap, rn, fn, fp = _compile_and_test(r, ["ab"], ["ac"])
        assert not rn
        assert "ac" in fp

    def test_is_correct(self):
        r = parse_regex("a(b|c)")
        assert _is_correct(r, ["ab", "ac"], ["ad", "bc"])

    def test_is_correct_fails(self):
        r = parse_regex("ab")
        assert not _is_correct(r, ["ab", "ac"], [])

    def test_ast_edit_distance_same(self):
        r = parse_regex("ab")
        assert _ast_edit_distance(r, r) == 0

    def test_ast_edit_distance_different(self):
        r1 = parse_regex("a")
        r2 = parse_regex("b")
        assert _ast_edit_distance(r1, r2) > 0

    def test_enumerate_subtrees(self):
        r = parse_regex("ab")
        subtrees = list(_enumerate_subtrees(r))
        assert len(subtrees) >= 3  # concat, a, b

    def test_replace_at_path(self):
        r = parse_regex("ab")
        # Replace 'a' with 'c'
        subtrees = list(_enumerate_subtrees(r))
        # Find 'a' literal
        for path, node in subtrees:
            s = regex_to_string(node)
            if s == "a":
                repaired = _replace_at_path(r, path, RLit("c"))
                assert regex_to_string(repaired) == "cb"
                break

    def test_chars_to_ranges_contiguous(self):
        ranges = _chars_to_ranges(['a', 'b', 'c'])
        assert ranges == [('a', 'c')]

    def test_chars_to_ranges_gap(self):
        ranges = _chars_to_ranges(['a', 'c', 'e'])
        assert len(ranges) == 3


# ===========================================================================
# Section 2: Quantifier mutations
# ===========================================================================

class TestQuantifierMutations:
    def test_star_mutations(self):
        node = RStar(RLit("a"))
        muts = _quantifier_mutations(node)
        kinds = {m.kind for m in muts}
        assert any(m.kind.name == "PLUS" for m in muts)
        assert any(m.kind.name == "OPTIONAL" for m in muts)
        assert any(m.kind.name == "LITERAL" for m in muts)  # remove quantifier

    def test_plus_mutations(self):
        node = RPlus(RLit("a"))
        muts = _quantifier_mutations(node)
        assert any(m.kind.name == "STAR" for m in muts)
        assert any(m.kind.name == "OPTIONAL" for m in muts)

    def test_no_quantifier_mutations(self):
        node = RLit("a")
        muts = _quantifier_mutations(node)
        assert any(m.kind.name == "STAR" for m in muts)
        assert any(m.kind.name == "PLUS" for m in muts)
        assert any(m.kind.name == "OPTIONAL" for m in muts)


# ===========================================================================
# Section 3: Character class mutations
# ===========================================================================

class TestCharClassMutations:
    def test_literal_to_class(self):
        node = RLit("a")
        muts = _char_class_mutations(node, ["abc"], [])
        assert any(m.kind.name == "CHAR_CLASS" for m in muts)
        assert any(m.kind.name == "DOT" for m in muts)

    def test_digit_literal_to_digit_class(self):
        node = RLit("5")
        muts = _char_class_mutations(node, ["123"], [])
        assert any(m.kind.name == "CHAR_CLASS" and m.ranges == (('0', '9'),) for m in muts)

    def test_dot_narrowing(self):
        node = RDot()
        muts = _char_class_mutations(node, ["abc"], ["123"])
        assert any(m.kind.name == "CHAR_CLASS" for m in muts)


# ===========================================================================
# Section 4: Structural mutations
# ===========================================================================

class TestStructuralMutations:
    def test_concat_remove_child(self):
        node = RConcat(RLit("a"), RLit("b"), RLit("c"))
        muts = _structural_mutations(node, ["ab"])
        # Should have variants with one child removed
        assert len(muts) > 0
        # One of them should be "ab" (remove c)
        patterns = [regex_to_string(m) for m in muts]
        assert "ab" in patterns

    def test_concat_make_optional(self):
        node = RConcat(RLit("a"), RLit("b"))
        muts = _structural_mutations(node, ["a", "ab"])
        assert any("?" in regex_to_string(m) for m in muts)

    def test_alt_remove_branch(self):
        node = RAlt(RLit("a"), RLit("b"), RLit("c"))
        muts = _structural_mutations(node, ["a"])
        patterns = [regex_to_string(m) for m in muts]
        assert any("a|b" in p or "b|a" in p for p in patterns)


# ===========================================================================
# Section 5: Literal mutations
# ===========================================================================

class TestLiteralMutations:
    def test_literal_substitution(self):
        node = RLit("x")
        muts = _literal_mutations(node, ["abc"], [])
        chars = {m.char for m in muts}
        assert "a" in chars
        assert "b" in chars
        assert "c" in chars
        assert "x" not in chars  # Don't include original

    def test_non_literal_no_mutations(self):
        node = RDot()
        muts = _literal_mutations(node, ["abc"], [])
        assert muts == []


# ===========================================================================
# Section 6: Fault localization
# ===========================================================================

class TestFaultLocalization:
    def test_localize_too_restrictive(self):
        regex = parse_regex("ab")
        faults = _localize_fault(regex, ["ab", "ac"], [])
        assert len(faults) > 0

    def test_localize_correct_regex(self):
        regex = parse_regex("a(b|c)")
        faults = _localize_fault(regex, ["ab", "ac"], ["ad"])
        assert len(faults) == 0

    def test_localize_too_permissive(self):
        regex = parse_regex("a.")
        faults = _localize_fault(regex, ["ab"], ["ac"])
        assert len(faults) > 0


# ===========================================================================
# Section 7: Diagnosis
# ===========================================================================

class TestDiagnosis:
    def test_diagnose_false_negatives(self):
        result = diagnose_regex("ab", ["ab", "ac", "ad"], [])
        assert len(result.false_negatives) == 2
        assert "ac" in result.false_negatives
        assert "ad" in result.false_negatives
        assert len(result.suggestions) > 0

    def test_diagnose_false_positives(self):
        result = diagnose_regex("a.", ["ab"], ["ac", "ad"])
        assert len(result.false_positives) == 2
        assert len(result.suggestions) > 0

    def test_diagnose_correct(self):
        result = diagnose_regex("a(b|c)", ["ab", "ac"], ["ad"])
        assert len(result.false_negatives) == 0
        assert len(result.false_positives) == 0

    def test_diagnose_has_fault_nodes(self):
        result = diagnose_regex("ab", ["ab", "ac"], [])
        assert len(result.fault_nodes) > 0


# ===========================================================================
# Section 8: Basic repair - quantifier fixes
# ===========================================================================

class TestQuantifierRepair:
    def test_repair_star_to_plus(self):
        # a* matches "", but we want at least one a
        result = repair_regex("a*", ["a", "aa"], [""], timeout=10.0)
        assert result.success
        # Repaired should accept "a", "aa" but reject ""
        assert _is_correct(result.repaired_regex, ["a", "aa"], [""])

    def test_repair_plus_to_star(self):
        # a+ doesn't match "", but we want it to
        result = repair_regex("a+", ["", "a", "aa"], [], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["", "a", "aa"], [])

    def test_repair_add_optional(self):
        # "ab" doesn't match "a", make b optional
        result = repair_regex("ab", ["a", "ab"], ["b"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["a", "ab"], ["b"])

    def test_repair_remove_quantifier(self):
        # "ab*" matches "a" but we only want "ab"
        result = repair_regex("ab*", ["ab"], ["a", "abb"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["ab"], ["a", "abb"])


# ===========================================================================
# Section 9: Character class repair
# ===========================================================================

class TestCharClassRepair:
    def test_widen_literal_to_class(self):
        # "a1" should match "a2", "a3" too
        result = repair_regex("a1", ["a1", "a2", "a3"], ["b1"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["a1", "a2", "a3"], ["b1"])

    def test_narrow_dot_to_class(self):
        # ".b" is too broad -- only want digits before b
        result = repair_regex(".b", ["1b", "2b"], ["ab", "xb"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["1b", "2b"], ["ab", "xb"])

    def test_repair_wrong_literal(self):
        # "ax" should be "ab"
        result = repair_regex("ax", ["ab"], ["ax"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["ab"], ["ax"])


# ===========================================================================
# Section 10: Structural repair
# ===========================================================================

class TestStructuralRepair:
    def test_remove_extra_part(self):
        # "abc" should just be "ab"
        result = repair_regex("abc", ["ab"], ["abc"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["ab"], ["abc"])

    def test_add_alternative(self):
        # "a" should also match "b"
        result = repair_regex("a", ["a", "b"], ["c"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["a", "b"], ["c"])

    def test_make_part_optional(self):
        # "abc" should match both "abc" and "ac"
        result = repair_regex("abc", ["abc", "ac"], ["ab"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["abc", "ac"], ["ab"])


# ===========================================================================
# Section 11: Already correct
# ===========================================================================

class TestAlreadyCorrect:
    def test_no_repair_needed(self):
        result = repair_regex("a(b|c)", ["ab", "ac"], ["ad"])
        assert result.success
        assert result.method == "already_correct"
        assert result.edit_distance == 0

    def test_simple_correct(self):
        result = repair_regex("abc", ["abc"], ["abd"])
        assert result.success
        assert result.method == "already_correct"


# ===========================================================================
# Section 12: Targeted repair
# ===========================================================================

class TestTargetedRepair:
    def test_targeted_replace(self):
        # Replace 'b' in "ab" with 'c'
        result = repair_regex_targeted("ab", ["ac"], ["ab"], (1,), "c")
        assert result.success
        assert result.method == "targeted"

    def test_targeted_fails(self):
        result = repair_regex_targeted("ab", ["cd"], ["ab"], (0,), "x")
        assert not result.success


# ===========================================================================
# Section 13: Suggest repairs
# ===========================================================================

class TestSuggestRepairs:
    def test_multiple_suggestions(self):
        suggestions = suggest_repairs("a.", ["ab", "ac"], ["a1", "a2"], timeout=10.0)
        assert len(suggestions) > 0
        for s in suggestions:
            assert s.success
            assert _is_correct(s.repaired_regex, ["ab", "ac"], ["a1", "a2"])

    def test_suggestions_sorted_by_distance(self):
        suggestions = suggest_repairs("ab*", ["ab"], ["a", "abb"], timeout=10.0)
        if len(suggestions) > 1:
            for i in range(len(suggestions) - 1):
                assert suggestions[i].edit_distance <= suggestions[i+1].edit_distance

    def test_no_suggestions_for_correct(self):
        suggestions = suggest_repairs("ab", ["ab"], ["cd"], timeout=5.0)
        # May return 0 or some suggestions (correct regex still has mutations)
        # Just check no crash


# ===========================================================================
# Section 14: Compare repairs
# ===========================================================================

class TestCompareRepairs:
    def test_compare_strategies(self):
        result = compare_repairs("a*", ["a", "aa"], [""], timeout=15.0)
        assert "strategies" in result
        assert "best_strategy" in result
        assert result["num_strategies_found"] >= 1

    def test_compare_has_best(self):
        result = compare_repairs("ab", ["ab", "ac"], ["ad"], timeout=15.0)
        if result["best_result"]:
            assert result["best_result"].success


# ===========================================================================
# Section 15: Batch repair
# ===========================================================================

class TestBatchRepair:
    def test_batch_multiple(self):
        problems = [
            ("a*", ["a", "aa"], [""]),
            ("ab", ["ab", "ac"], ["ad"]),
        ]
        results = batch_repair(problems, timeout_per=10.0)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, RepairResult)

    def test_batch_all_succeed(self):
        problems = [
            ("ab", ["a", "ab"], ["b"]),
            ("a*", ["a"], [""]),
        ]
        results = batch_repair(problems, timeout_per=10.0)
        for r in results:
            assert r.success


# ===========================================================================
# Section 16: Repair from counterexample
# ===========================================================================

class TestRepairFromCounterexample:
    def test_false_negative_counterexample(self):
        # "ab" doesn't match "ac", but should
        result = repair_from_counterexample("ab", "ac", should_match=True, timeout=10.0)
        assert result.success
        r = result.repaired_regex
        from symbolic_regex import RegexCompiler
        from symbolic_automata import CharAlgebra
        sfa = RegexCompiler(CharAlgebra()).compile(r).determinize()
        assert sfa.accepts("ac")

    def test_false_positive_counterexample(self):
        # "a." matches "a1", but shouldn't
        result = repair_from_counterexample("a.", "a1", should_match=False, timeout=10.0)
        assert result.success
        r = result.repaired_regex
        from symbolic_regex import RegexCompiler
        from symbolic_automata import CharAlgebra
        sfa = RegexCompiler(CharAlgebra()).compile(r).determinize()
        assert not sfa.accepts("a1")


# ===========================================================================
# Section 17: Semantic distance
# ===========================================================================

class TestSemanticDistance:
    def test_equivalent_patterns(self):
        result = semantic_distance("a|b", "b|a")
        assert result["equivalent"]
        assert result["jaccard_similarity"] == 1.0

    def test_different_patterns(self):
        result = semantic_distance("a", "b")
        assert not result["equivalent"]
        assert len(result["only_in_first"]) > 0 or len(result["only_in_second"]) > 0

    def test_subset_patterns(self):
        result = semantic_distance("a", "a|b")
        assert not result["equivalent"]
        assert len(result["only_in_second"]) > 0

    def test_jaccard_disjoint(self):
        result = semantic_distance("a", "b")
        assert result["jaccard_similarity"] < 0.5


# ===========================================================================
# Section 18: Complex repair scenarios
# ===========================================================================

class TestComplexRepair:
    def test_repair_multi_char_class(self):
        # Pattern matches digits but should match letters
        result = repair_regex("[0-9]", ["a", "b", "c"], ["1", "2"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["a", "b", "c"], ["1", "2"])

    def test_repair_alternation_missing_branch(self):
        # "a|b" should also match "c"
        result = repair_regex("a|b", ["a", "b", "c"], ["d"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["a", "b", "c"], ["d"])

    def test_repair_quantifier_in_context(self):
        # "ab+c" -- the b+ is wrong, should be b?
        result = repair_regex("ab+c", ["ac", "abc"], ["abbc"], timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["ac", "abc"], ["abbc"])

    def test_repair_preserves_structure(self):
        # Prefer minimal edit
        result = repair_regex("a(b|c)d", ["abd", "acd", "aed"], ["af"], timeout=10.0)
        assert result.success
        # Should have found a small edit, not a complete rewrite
        assert result.edit_distance < 20


# ===========================================================================
# Section 19: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_positives(self):
        result = repair_regex("a", [], ["a"], timeout=5.0)
        assert result.success

    def test_empty_negatives(self):
        result = repair_regex("b", ["a"], [], timeout=5.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["a"], [])

    def test_epsilon_repair(self):
        # Should match empty string
        result = repair_regex("a", [""], ["a"], timeout=10.0)
        assert result.success

    def test_single_char_repair(self):
        result = repair_regex("a", ["b"], ["a"], timeout=5.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["b"], ["a"])

    def test_max_edit_distance_constraint(self):
        result = repair_regex("abcdef", ["xyz"], ["abcdef"], max_edit_distance=1, timeout=5.0)
        # May fail because edit distance is too large
        if result.success:
            assert result.edit_distance <= 1


# ===========================================================================
# Section 20: Repair result properties
# ===========================================================================

class TestRepairResultProperties:
    def test_result_has_pattern(self):
        result = repair_regex("a*", ["a"], [""], timeout=10.0)
        assert result.success
        assert result.repaired_pattern != ""
        assert result.pattern == result.repaired_pattern

    def test_result_has_edit_distance(self):
        result = repair_regex("ab", ["ac"], ["ab"], timeout=10.0)
        assert result.success
        assert result.edit_distance > 0

    def test_result_has_method(self):
        result = repair_regex("ab", ["ac"], ["ab"], timeout=10.0)
        assert result.success
        assert result.method in ("single_mutation", "double_mutation", "synthesis",
                                  "already_correct", "targeted")

    def test_result_has_stats(self):
        result = repair_regex("ab", ["ac"], ["ab"], timeout=10.0)
        assert result.success
        assert isinstance(result.stats, dict)


# ===========================================================================
# Section 21: Diagnosis + repair integration
# ===========================================================================

class TestDiagnosisRepairIntegration:
    def test_diagnose_then_repair(self):
        pattern = "a[0-9]"
        positives = ["ab", "ac"]
        negatives = ["a1", "a2"]

        diag = diagnose_regex(pattern, positives, negatives)
        assert len(diag.false_negatives) > 0 or len(diag.false_positives) > 0

        result = repair_regex(pattern, positives, negatives, timeout=10.0)
        assert result.success
        assert _is_correct(result.repaired_regex, positives, negatives)

    def test_diagnose_gives_useful_info(self):
        diag = diagnose_regex("a.", ["ab"], ["ac"])
        assert len(diag.suggestions) > 0
        assert diag.pattern == "a."


# ===========================================================================
# Section 22: Synthesis fallback
# ===========================================================================

class TestSynthesisFallback:
    def test_synthesis_when_mutations_fail(self):
        # Hard repair that likely needs synthesis
        result = repair_regex("x", ["abc", "def"], ["ghi"], timeout=20.0)
        assert result.success
        assert _is_correct(result.repaired_regex, ["abc", "def"], ["ghi"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
