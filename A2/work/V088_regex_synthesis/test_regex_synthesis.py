"""Tests for V088: Regex Synthesis from Examples"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V084_symbolic_regex'))

from regex_synthesis import (
    synthesize_regex, verify_synthesis, compare_strategies,
    pattern_synthesize, enumerative_synthesize, rpni_synthesize,
    lstar_synthesize, cegis_synthesize, synthesize_from_language,
    SynthesisResult, _sfa_to_regex, _build_prefix_tree,
    _common_prefix, _common_suffix, _string_to_regex, _chars_to_regex,
)
from symbolic_automata import CharAlgebra, sfa_from_string, SFA
from symbolic_regex import (
    compile_regex, compile_regex_dfa, regex_to_string, regex_size,
    RegexCompiler, parse_regex, RLit, RConcat, RAlt, RStar, REpsilon,
    REmpty, RPlus, RDot, RClass, regex_equivalent
)


# ============================================================
# Section 1: Helper functions
# ============================================================

class TestHelpers:
    def test_common_prefix(self):
        assert _common_prefix(["abc", "abd", "abe"]) == "ab"
        assert _common_prefix(["hello", "help", "heap"]) == "he"
        assert _common_prefix(["abc"]) == "abc"
        assert _common_prefix([]) == ""
        assert _common_prefix(["abc", "xyz"]) == ""

    def test_common_suffix(self):
        assert _common_suffix(["abc", "dbc", "ebc"]) == "bc"
        assert _common_suffix(["testing", "coding", "running"]) == "ing"
        assert _common_suffix(["abc"]) == "abc"
        assert _common_suffix([]) == ""

    def test_string_to_regex(self):
        r = _string_to_regex("abc")
        assert regex_to_string(r) == "abc"

    def test_string_to_regex_empty(self):
        r = _string_to_regex("")
        assert r.kind.name == "EPSILON"

    def test_string_to_regex_single(self):
        r = _string_to_regex("x")
        assert r.kind.name == "LITERAL"

    def test_chars_to_regex_single(self):
        r = _chars_to_regex(['a'])
        assert r.kind.name == "LITERAL"

    def test_chars_to_regex_range(self):
        r = _chars_to_regex(['a', 'b', 'c'])
        assert r.kind.name == "CHAR_CLASS"


# ============================================================
# Section 2: SFA to Regex conversion
# ============================================================

class TestSFAToRegex:
    def test_single_string(self):
        alg = CharAlgebra()
        sfa = sfa_from_string("abc", alg).determinize()
        regex = _sfa_to_regex(sfa)
        # Verify it accepts "abc"
        compiled = RegexCompiler(alg).compile(regex).determinize()
        assert compiled.accepts("abc")
        assert not compiled.accepts("ab")
        assert not compiled.accepts("abcd")

    def test_empty_language(self):
        alg = CharAlgebra()
        from symbolic_automata import sfa_empty
        sfa = sfa_empty(alg)
        regex = _sfa_to_regex(sfa)
        assert regex.kind.name == "EMPTY"

    def test_epsilon_language(self):
        alg = CharAlgebra()
        from symbolic_automata import sfa_epsilon
        sfa = sfa_epsilon(alg)
        regex = _sfa_to_regex(sfa)
        compiled = RegexCompiler(alg).compile(regex).determinize()
        assert compiled.accepts("")
        assert not compiled.accepts("a")

    def test_two_strings(self):
        alg = CharAlgebra()
        from symbolic_automata import sfa_union
        sfa1 = sfa_from_string("ab", alg)
        sfa2 = sfa_from_string("cd", alg)
        sfa = sfa_union(sfa1, sfa2).determinize()
        regex = _sfa_to_regex(sfa)
        compiled = RegexCompiler(alg).compile(regex).determinize()
        assert compiled.accepts("ab")
        assert compiled.accepts("cd")
        assert not compiled.accepts("ac")


# ============================================================
# Section 3: Prefix tree construction
# ============================================================

class TestPrefixTree:
    def test_build_prefix_tree_single(self):
        alg = CharAlgebra()
        pta = _build_prefix_tree(["abc"], alg)
        assert pta.accepts("abc")
        assert not pta.accepts("ab")
        assert not pta.accepts("abd")

    def test_build_prefix_tree_multiple(self):
        alg = CharAlgebra()
        pta = _build_prefix_tree(["abc", "abd"], alg)
        assert pta.accepts("abc")
        assert pta.accepts("abd")
        assert not pta.accepts("ab")
        assert not pta.accepts("abe")

    def test_build_prefix_tree_shared_prefix(self):
        alg = CharAlgebra()
        pta = _build_prefix_tree(["cat", "car", "cap"], alg)
        assert pta.accepts("cat")
        assert pta.accepts("car")
        assert pta.accepts("cap")
        assert not pta.accepts("ca")
        assert not pta.accepts("cab")

    def test_build_prefix_tree_empty_string(self):
        alg = CharAlgebra()
        pta = _build_prefix_tree(["", "a"], alg)
        assert pta.accepts("")
        assert pta.accepts("a")
        assert not pta.accepts("b")


# ============================================================
# Section 4: Pattern synthesis
# ============================================================

class TestPatternSynthesis:
    def test_single_string(self):
        r = pattern_synthesize(["hello"], ["world"])
        assert r.success
        v = verify_synthesis(r.regex, ["hello"], ["world"])
        assert v["valid"]

    def test_fixed_length_digits(self):
        pos = ["123", "456", "789"]
        neg = ["12", "1234", "abc"]
        r = pattern_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_common_prefix(self):
        pos = ["test_a", "test_b", "test_c"]
        neg = ["best_a", "tests"]
        r = pattern_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_common_suffix(self):
        pos = ["running", "coding", "testing"]
        neg = ["runner", "coder"]
        r = pattern_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_alternation(self):
        pos = ["yes", "no"]
        neg = ["maybe"]
        r = pattern_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_char_class_repeat(self):
        pos = ["aaa", "bbb", "ab", "ba", "aabb"]
        neg = ["123", "xyz"]
        r = pattern_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_empty_positives(self):
        r = pattern_synthesize([], ["abc"])
        assert not r.success

    def test_prefix_suffix_combo(self):
        pos = ["<a>", "<b>", "<c>"]
        neg = ["a>", "<a", "abc"]
        r = pattern_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]


# ============================================================
# Section 5: Enumerative synthesis
# ============================================================

class TestEnumerativeSynthesis:
    def test_single_char(self):
        r = enumerative_synthesize(["a"], ["b", "c"])
        assert r.success
        v = verify_synthesis(r.regex, ["a"], ["b", "c"])
        assert v["valid"]

    def test_dot_star(self):
        pos = ["a", "bb", "ccc", ""]
        neg = []
        r = enumerative_synthesize(pos, neg, max_size=2)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_digit_class(self):
        pos = ["1", "5", "9"]
        neg = ["a", "z"]
        r = enumerative_synthesize(pos, neg, max_size=2)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_no_match(self):
        # Very constrained, unlikely to find a small regex
        pos = ["abc123xyz"]
        neg = ["abc", "123", "xyz", "abc123", "123xyz"]
        r = enumerative_synthesize(pos, neg, max_size=2)
        # May not find a regex this small
        if r.success:
            v = verify_synthesis(r.regex, pos, neg)
            assert v["valid"]


# ============================================================
# Section 6: RPNI synthesis
# ============================================================

class TestRPNISynthesis:
    def test_simple_merge(self):
        pos = ["ab", "cd"]
        neg = ["ac", "bd"]
        r = rpni_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_common_structure(self):
        pos = ["aa", "ab", "ba", "bb"]
        neg = ["a", "b", "aaa"]
        r = rpni_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_no_negatives(self):
        pos = ["hello", "world"]
        r = rpni_synthesize(pos, [])
        assert r.success
        # Should at least accept the positives
        assert r.accepts_all(pos)

    def test_single_example(self):
        r = rpni_synthesize(["test"], ["toast"])
        assert r.success
        v = verify_synthesis(r.regex, ["test"], ["toast"])
        assert v["valid"]


# ============================================================
# Section 7: L* synthesis
# ============================================================

class TestLStarSynthesis:
    def test_simple_language(self):
        pos = ["a", "aa", "aaa"]
        neg = ["b", "ab", "ba", ""]
        r = lstar_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_two_char_language(self):
        pos = ["ab"]
        neg = ["a", "b", "ba", "aba"]
        r = lstar_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_epsilon_in_language(self):
        pos = ["", "a"]
        neg = ["b", "aa"]
        r = lstar_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_alternation_language(self):
        pos = ["a", "b"]
        neg = ["c", "ab", "ba", ""]
        r = lstar_synthesize(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]


# ============================================================
# Section 8: CEGIS synthesis
# ============================================================

class TestCEGISSynthesis:
    def test_simple(self):
        r = cegis_synthesize(["abc"], ["abd", "ab"])
        assert r.success
        v = verify_synthesis(r.regex, ["abc"], ["abd", "ab"])
        assert v["valid"]

    def test_converges(self):
        pos = ["aa", "bb"]
        neg = ["ab", "ba", "a", "b"]
        r = cegis_synthesize(pos, neg, max_rounds=10)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_needs_refinement(self):
        pos = ["test1", "test2", "test3"]
        neg = ["test", "1test", "tset1"]
        r = cegis_synthesize(pos, neg, max_rounds=10)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]


# ============================================================
# Section 9: Main API (synthesize_regex)
# ============================================================

class TestSynthesizeRegex:
    def test_auto_strategy(self):
        r = synthesize_regex(["abc", "def"], ["ghi"])
        assert r.success
        v = verify_synthesis(r.regex, ["abc", "def"], ["ghi"])
        assert v["valid"]

    def test_explicit_pattern(self):
        r = synthesize_regex(["hello"], [], strategy="pattern")
        assert r.success

    def test_explicit_enumerative(self):
        r = synthesize_regex(["a"], ["b"], strategy="enumerative")
        assert r.success

    def test_explicit_rpni(self):
        r = synthesize_regex(["ab", "cd"], [], strategy="rpni")
        assert r.success

    def test_explicit_lstar(self):
        r = synthesize_regex(["a", "b"], ["c"], strategy="lstar")
        assert r.success

    def test_explicit_cegis(self):
        r = synthesize_regex(["yes"], ["no"], strategy="cegis")
        assert r.success

    def test_no_positives(self):
        r = synthesize_regex([], ["abc"])
        assert not r.success

    def test_no_negatives(self):
        r = synthesize_regex(["abc", "def"])
        assert r.success
        assert r.accepts_all(["abc", "def"])

    def test_digits_only(self):
        pos = ["123", "456", "789", "000"]
        neg = ["abc", "12a", "a23"]
        r = synthesize_regex(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_prefixed_strings(self):
        pos = ["log_error", "log_warn", "log_info"]
        neg = ["error_log", "log", "warning"]
        r = synthesize_regex(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_email_like(self):
        pos = ["a@b", "x@y"]
        neg = ["ab", "@", "a@", "@b"]
        r = synthesize_regex(pos, neg, strategy="pattern")
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_repeated_char(self):
        pos = ["aaa", "aaaa", "aaaaa"]
        neg = ["a", "aa", "b", "aab"]
        r = synthesize_regex(pos, neg, strategy="pattern")
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]


# ============================================================
# Section 10: Verification
# ============================================================

class TestVerification:
    def test_verify_valid(self):
        regex = RConcat(RLit('a'), RLit('b'))
        v = verify_synthesis(regex, ["ab"], ["cd"])
        assert v["valid"]
        assert v["false_negatives"] == []
        assert v["false_positives"] == []

    def test_verify_false_negative(self):
        regex = RLit('a')
        v = verify_synthesis(regex, ["a", "b"], [])
        assert not v["valid"]
        assert "b" in v["false_negatives"]

    def test_verify_false_positive(self):
        regex = RDot()
        v = verify_synthesis(regex, ["a"], ["b"])
        assert not v["valid"]
        assert "b" in v["false_positives"]

    def test_verify_from_string(self):
        v = verify_synthesis("ab", ["ab"], ["cd"])
        assert v["valid"]

    def test_verify_counts(self):
        regex = RLit('a')
        v = verify_synthesis(regex, ["a"], ["b", "c"])
        assert v["accepts_count"] == 1
        assert v["rejects_count"] == 2


# ============================================================
# Section 11: Strategy comparison
# ============================================================

class TestCompareStrategies:
    def test_compare_basic(self):
        results = compare_strategies(["a", "b"], ["c"])
        assert "pattern" in results
        assert "enumerative" in results
        assert "rpni" in results
        assert "lstar" in results
        assert "cegis" in results

    def test_compare_all_succeed(self):
        results = compare_strategies(["a"], ["b"])
        successes = sum(1 for r in results.values() if r.get("success"))
        assert successes >= 1  # At least one strategy should work

    def test_compare_reports_size(self):
        results = compare_strategies(["a", "b"], [])
        for strat, r in results.items():
            if r.get("success"):
                assert r["size"] is not None
                assert r["size"] > 0


# ============================================================
# Section 12: Synthesis from language
# ============================================================

class TestSynthesizeFromLanguage:
    def test_literal(self):
        r = synthesize_from_language("abc")
        assert r.success
        assert r.accepts_all(["abc"])
        assert r.rejects_all(["ab", "abcd"])

    def test_alternation(self):
        r = synthesize_from_language("a|b")
        assert r.success
        assert r.accepts_all(["a", "b"])

    def test_star(self):
        r = synthesize_from_language("a*")
        assert r.success
        assert r.accepts_all(["", "a", "aa"])


# ============================================================
# Section 13: SynthesisResult methods
# ============================================================

class TestSynthesisResult:
    def test_accepts_all(self):
        r = SynthesisResult(True, RLit('a'), "a", "test")
        assert r.accepts_all(["a"])
        assert not r.accepts_all(["b"])

    def test_rejects_all(self):
        r = SynthesisResult(True, RLit('a'), "a", "test")
        assert r.rejects_all(["b", "c"])
        assert not r.rejects_all(["a"])

    def test_failed_result(self):
        r = SynthesisResult(False)
        assert not r.accepts_all(["a"])
        assert r.rejects_all(["a"])  # vacuously true


# ============================================================
# Section 14: Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_string_only(self):
        r = synthesize_regex([""], ["a", "b"])
        assert r.success
        v = verify_synthesis(r.regex, [""], ["a", "b"])
        assert v["valid"]

    def test_single_char_alphabet(self):
        r = synthesize_regex(["a", "aa"], ["aaa"])
        assert r.success
        v = verify_synthesis(r.regex, ["a", "aa"], ["aaa"])
        assert v["valid"]

    def test_long_string(self):
        r = synthesize_regex(["abcdefghij"], [])
        assert r.success
        assert r.accepts_all(["abcdefghij"])

    def test_many_positives_same_length(self):
        pos = [f"{chr(ord('a')+i)}{chr(ord('a')+j)}" for i in range(3) for j in range(3)]
        neg = ["abc", "a", ""]
        r = synthesize_regex(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_overlapping_pos_neg_rejected(self):
        # If a string is in both pos and neg, it's contradictory
        # Synthesis should still try (pos wins for that string)
        r = synthesize_regex(["a"], ["a"])
        # Result depends on strategy -- it may fail or succeed with false_pos
        # Just verify it doesn't crash

    def test_special_chars(self):
        pos = ["a.b", "c.d"]
        neg = ["ab", "cd"]
        r = synthesize_regex(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_unicode_like_chars(self):
        # Test with chars outside basic alpha range
        pos = ["~!", "@#"]
        neg = ["ab", "12"]
        r = synthesize_regex(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]


# ============================================================
# Section 15: Integration with V084
# ============================================================

class TestV084Integration:
    def test_synthesis_matches_v084_compilation(self):
        """Synthesized regex should compile to same language as V084."""
        pos = ["ab", "cd"]
        neg = ["ac", "bd"]
        r = synthesize_regex(pos, neg)
        assert r.success

        alg = CharAlgebra()
        compiled = RegexCompiler(alg).compile(r.regex).determinize()
        for p in pos:
            assert compiled.accepts(p)
        for n in neg:
            assert not compiled.accepts(n)

    def test_roundtrip_pattern_string(self):
        """Pattern string from synthesis should recompile correctly."""
        r = synthesize_regex(["hello"], ["world"])
        assert r.success
        assert r.pattern is not None

        # The pattern string should be valid
        alg = CharAlgebra()
        parsed = parse_regex(r.pattern)
        compiled = RegexCompiler(alg).compile(parsed).determinize()
        assert compiled.accepts("hello")

    def test_equivalence_check(self):
        """Two synthesis runs should produce equivalent regexes."""
        pos = ["abc"]
        neg = ["abd"]
        r1 = synthesize_regex(pos, neg, strategy="pattern")
        r2 = synthesize_regex(pos, neg, strategy="lstar")
        if r1.success and r2.success:
            # Both should at least agree on the examples
            v1 = verify_synthesis(r1.regex, pos, neg)
            v2 = verify_synthesis(r2.regex, pos, neg)
            assert v1["valid"]
            assert v2["valid"]


# ============================================================
# Section 16: Regression tests (known tricky cases)
# ============================================================

class TestRegression:
    def test_single_string_exact(self):
        """Single positive string should synthesize exact match."""
        r = synthesize_regex(["test"], ["tes", "tess", "tests"])
        assert r.success
        v = verify_synthesis(r.regex, ["test"], ["tes", "tess", "tests"])
        assert v["valid"]

    def test_all_same_string(self):
        """Duplicate positives should work fine."""
        r = synthesize_regex(["abc", "abc", "abc"], ["abd"])
        assert r.success
        v = verify_synthesis(r.regex, ["abc"], ["abd"])
        assert v["valid"]

    def test_varied_lengths(self):
        pos = ["a", "ab", "abc", "abcd"]
        neg = ["b", "bc", "bcd"]
        r = synthesize_regex(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]

    def test_digit_sequences(self):
        pos = ["0", "1", "2", "3"]
        neg = ["a", "b", "00", ""]
        r = synthesize_regex(pos, neg)
        assert r.success
        v = verify_synthesis(r.regex, pos, neg)
        assert v["valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
