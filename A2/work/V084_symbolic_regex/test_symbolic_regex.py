"""Tests for V084: Symbolic Regex."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))

from symbolic_regex import (
    # Parser
    parse_regex, RegexParseError,
    RegexKind, RLit, RDot, RClass, RNegClass, RConcat, RAlt, RStar, RPlus,
    ROptional, REpsilon, REmpty,
    # Compiler
    compile_regex, compile_regex_dfa, compile_regex_min,
    RegexCompiler,
    # Equivalence / comparison
    regex_equivalent, regex_subset, regex_intersects,
    regex_difference_witness, regex_intersection_witness,
    regex_is_empty, regex_accepts_epsilon, regex_sample,
    regex_compare,
    # Derivatives
    nullable, derivative, derivative_match,
    # Operations
    regex_union, regex_intersection, regex_complement, regex_difference,
    # Utilities
    regex_to_string, regex_size,
)
from symbolic_automata import CharAlgebra, SFA


# ============================================================================
# 1. Regex Parser Tests
# ============================================================================

class TestParser:
    def test_empty_pattern(self):
        r = parse_regex("")
        assert r.kind == RegexKind.EPSILON

    def test_single_char(self):
        r = parse_regex("a")
        assert r.kind == RegexKind.LITERAL
        assert r.char == "a"

    def test_concat(self):
        r = parse_regex("abc")
        assert r.kind == RegexKind.CONCAT
        assert len(r.children) == 3
        assert all(c.kind == RegexKind.LITERAL for c in r.children)
        assert [c.char for c in r.children] == ['a', 'b', 'c']

    def test_alternation(self):
        r = parse_regex("a|b")
        assert r.kind == RegexKind.ALT
        assert len(r.children) == 2

    def test_star(self):
        r = parse_regex("a*")
        assert r.kind == RegexKind.STAR
        assert r.child.kind == RegexKind.LITERAL

    def test_plus(self):
        r = parse_regex("a+")
        assert r.kind == RegexKind.PLUS

    def test_optional(self):
        r = parse_regex("a?")
        assert r.kind == RegexKind.OPTIONAL

    def test_grouping(self):
        r = parse_regex("(ab)")
        assert r.kind == RegexKind.CONCAT
        assert len(r.children) == 2

    def test_dot(self):
        r = parse_regex(".")
        assert r.kind == RegexKind.DOT

    def test_char_class(self):
        r = parse_regex("[abc]")
        assert r.kind == RegexKind.CHAR_CLASS
        assert len(r.ranges) == 3

    def test_char_range(self):
        r = parse_regex("[a-z]")
        assert r.kind == RegexKind.CHAR_CLASS
        assert r.ranges == (('a', 'z'),)

    def test_negated_class(self):
        r = parse_regex("[^abc]")
        assert r.kind == RegexKind.NEG_CLASS

    def test_escape_dot(self):
        r = parse_regex("\\.")
        assert r.kind == RegexKind.LITERAL
        assert r.char == "."

    def test_escape_d(self):
        r = parse_regex("\\d")
        assert r.kind == RegexKind.CHAR_CLASS
        assert r.ranges == (('0', '9'),)

    def test_escape_w(self):
        r = parse_regex("\\w")
        assert r.kind == RegexKind.CHAR_CLASS

    def test_escape_s(self):
        r = parse_regex("\\s")
        assert r.kind == RegexKind.CHAR_CLASS

    def test_escape_D(self):
        r = parse_regex("\\D")
        assert r.kind == RegexKind.NEG_CLASS

    def test_complex_pattern(self):
        r = parse_regex("(a|b)*c")
        assert r.kind == RegexKind.CONCAT
        assert r.children[0].kind == RegexKind.STAR
        assert r.children[0].child.kind == RegexKind.ALT

    def test_nested_groups(self):
        r = parse_regex("((a|b)c)*")
        assert r.kind == RegexKind.STAR

    def test_multi_alternation(self):
        r = parse_regex("a|b|c")
        assert r.kind == RegexKind.ALT
        assert len(r.children) == 3

    def test_empty_group(self):
        r = parse_regex("()")
        assert r.kind == RegexKind.EPSILON

    def test_parse_error_unmatched_paren(self):
        with pytest.raises(RegexParseError):
            parse_regex("(abc")

    def test_mixed_ranges_and_chars(self):
        r = parse_regex("[a-zA-Z0-9_]")
        assert r.kind == RegexKind.CHAR_CLASS
        assert len(r.ranges) == 4  # a-z, A-Z, 0-9, _-_

    def test_quantifier_on_group(self):
        r = parse_regex("(ab)+")
        assert r.kind == RegexKind.PLUS
        assert r.child.kind == RegexKind.CONCAT

    def test_escape_in_class(self):
        r = parse_regex("[\\n\\t]")
        assert r.kind == RegexKind.CHAR_CLASS


# ============================================================================
# 2. Regex Compilation Tests
# ============================================================================

class TestCompilation:
    def test_literal_accepts(self):
        sfa = compile_regex("a")
        assert sfa.accepts("a")
        assert not sfa.accepts("b")
        assert not sfa.accepts("")
        assert not sfa.accepts("aa")

    def test_concat_accepts(self):
        sfa = compile_regex("ab")
        assert sfa.accepts("ab")
        assert not sfa.accepts("a")
        assert not sfa.accepts("b")
        assert not sfa.accepts("abc")

    def test_alt_accepts(self):
        sfa = compile_regex("a|b")
        assert sfa.accepts("a")
        assert sfa.accepts("b")
        assert not sfa.accepts("c")
        assert not sfa.accepts("ab")

    def test_star_accepts(self):
        sfa = compile_regex("a*")
        assert sfa.accepts("")
        assert sfa.accepts("a")
        assert sfa.accepts("aa")
        assert sfa.accepts("aaa")
        assert not sfa.accepts("b")

    def test_plus_accepts(self):
        sfa = compile_regex("a+")
        assert not sfa.accepts("")
        assert sfa.accepts("a")
        assert sfa.accepts("aa")
        assert not sfa.accepts("b")

    def test_optional_accepts(self):
        sfa = compile_regex("a?")
        assert sfa.accepts("")
        assert sfa.accepts("a")
        assert not sfa.accepts("aa")

    def test_dot_accepts(self):
        sfa = compile_regex(".")
        assert sfa.accepts("a")
        assert sfa.accepts("z")
        assert sfa.accepts("5")
        assert not sfa.accepts("")
        assert not sfa.accepts("ab")

    def test_char_class_accepts(self):
        sfa = compile_regex("[abc]")
        assert sfa.accepts("a")
        assert sfa.accepts("b")
        assert sfa.accepts("c")
        assert not sfa.accepts("d")
        assert not sfa.accepts("")

    def test_char_range_accepts(self):
        sfa = compile_regex("[a-z]")
        assert sfa.accepts("a")
        assert sfa.accepts("m")
        assert sfa.accepts("z")
        assert not sfa.accepts("A")
        assert not sfa.accepts("0")

    def test_negated_class_accepts(self):
        sfa = compile_regex("[^0-9]")
        assert sfa.accepts("a")
        assert not sfa.accepts("5")
        assert not sfa.accepts("")

    def test_complex_pattern(self):
        sfa = compile_regex("(a|b)*c")
        assert sfa.accepts("c")
        assert sfa.accepts("ac")
        assert sfa.accepts("bc")
        assert sfa.accepts("abc")
        assert sfa.accepts("aabc")
        assert not sfa.accepts("a")
        assert not sfa.accepts("")

    def test_email_like(self):
        """Simplified email-like pattern."""
        sfa = compile_regex("[a-z]+@[a-z]+\\.[a-z]+")
        assert sfa.accepts("test@mail.com")
        assert sfa.accepts("a@b.c")
        assert not sfa.accepts("@mail.com")
        assert not sfa.accepts("test@.com")

    def test_digit_sequence(self):
        sfa = compile_regex("\\d+")
        assert sfa.accepts("0")
        assert sfa.accepts("123")
        assert not sfa.accepts("")
        assert not sfa.accepts("abc")

    def test_epsilon_pattern(self):
        sfa = compile_regex("")
        assert sfa.accepts("")
        assert not sfa.accepts("a")

    def test_nested_star(self):
        sfa = compile_regex("(a*b)*")
        assert sfa.accepts("")
        assert sfa.accepts("b")
        assert sfa.accepts("ab")
        assert sfa.accepts("aab")
        assert sfa.accepts("bab")
        assert sfa.accepts("aabab")
        assert not sfa.accepts("a")

    def test_compile_dfa(self):
        dfa = compile_regex_dfa("a|b")
        assert dfa.is_deterministic()
        assert dfa.accepts("a")
        assert dfa.accepts("b")
        assert not dfa.accepts("c")

    def test_compile_min(self):
        min_dfa = compile_regex_min("a|a")
        assert min_dfa.is_deterministic()
        assert min_dfa.accepts("a")

    def test_optional_concat(self):
        sfa = compile_regex("a?b")
        assert sfa.accepts("b")
        assert sfa.accepts("ab")
        assert not sfa.accepts("aab")
        assert not sfa.accepts("")


# ============================================================================
# 3. Regex Equivalence Tests
# ============================================================================

class TestEquivalence:
    def test_identical_patterns(self):
        assert regex_equivalent("abc", "abc")

    def test_star_idempotent(self):
        """a** = a*"""
        assert regex_equivalent("a**", "a*")

    def test_alt_commutative(self):
        assert regex_equivalent("a|b", "b|a")

    def test_alt_idempotent(self):
        assert regex_equivalent("a|a", "a")

    def test_not_equivalent(self):
        assert not regex_equivalent("a", "b")

    def test_star_plus(self):
        """a+ = aa*"""
        assert regex_equivalent("a+", "aa*")

    def test_optional_alt(self):
        """a? = (|a) = epsilon | a"""
        # a? should match "" and "a"
        # (|a) should match "" and "a"
        assert regex_equivalent("a?", "|a")

    def test_distributivity(self):
        """a(b|c) = ab|ac"""
        assert regex_equivalent("a(b|c)", "ab|ac")

    def test_star_epsilon(self):
        """(a|)* = a*"""
        assert regex_equivalent("(a|)*", "a*")

    def test_complex_equivalence(self):
        """(a|b)* = (a*b*)*"""
        assert regex_equivalent("(a|b)*", "(a*b*)*")

    def test_not_equivalent_different_length(self):
        assert not regex_equivalent("ab", "abc")

    def test_digit_patterns(self):
        """[0-9] = \\d"""
        assert regex_equivalent("[0-9]", "\\d")


# ============================================================================
# 4. Regex Subset/Inclusion Tests
# ============================================================================

class TestSubset:
    def test_literal_subset_of_alt(self):
        assert regex_subset("a", "a|b")

    def test_alt_not_subset_of_literal(self):
        assert not regex_subset("a|b", "a")

    def test_plus_subset_of_star(self):
        assert regex_subset("a+", "a*")

    def test_star_not_subset_of_plus(self):
        assert not regex_subset("a*", "a+")

    def test_literal_subset_of_dot(self):
        assert regex_subset("a", ".")

    def test_concat_subset_of_star(self):
        assert regex_subset("aaa", "a*")

    def test_class_subset_of_dot(self):
        assert regex_subset("[a-z]", ".")

    def test_reflexive(self):
        assert regex_subset("(a|b)*c", "(a|b)*c")


# ============================================================================
# 5. Regex Intersection Tests
# ============================================================================

class TestIntersection:
    def test_disjoint(self):
        assert not regex_intersects("a", "b")

    def test_overlapping(self):
        assert regex_intersects("a|b", "b|c")

    def test_subset_intersects(self):
        assert regex_intersects("a", "a|b")

    def test_star_intersects_plus(self):
        assert regex_intersects("a*", "a+")

    def test_intersection_witness(self):
        w = regex_intersection_witness("a|b", "b|c")
        assert w is not None
        assert ''.join(w) == "b"

    def test_disjoint_no_witness(self):
        w = regex_intersection_witness("[a-c]", "[d-f]")
        assert w is None


# ============================================================================
# 6. Difference / Witness Tests
# ============================================================================

class TestDifference:
    def test_star_minus_plus(self):
        """a* - a+ = {epsilon}"""
        w = regex_difference_witness("a*", "a+")
        assert w is not None
        assert w == []  # empty string

    def test_no_difference(self):
        w = regex_difference_witness("a", "a|b")
        assert w is None  # L(a) subset L(a|b)

    def test_difference_exists(self):
        w = regex_difference_witness("a|b", "a")
        assert w is not None
        assert ''.join(w) == "b"

    def test_complex_difference(self):
        w = regex_difference_witness("[a-z]+", "[a-m]+")
        assert w is not None
        word = ''.join(w)
        assert any(c > 'm' for c in word)


# ============================================================================
# 7. Empty/Epsilon/Sample Tests
# ============================================================================

class TestMisc:
    def test_empty_language(self):
        # Intersection of disjoint should be empty
        sfa = regex_intersection("[a-c]", "[d-f]")
        assert sfa.is_empty()

    def test_not_empty(self):
        assert not regex_is_empty("a")

    def test_accepts_epsilon(self):
        assert regex_accepts_epsilon("a*")
        assert regex_accepts_epsilon("a?")
        assert regex_accepts_epsilon("")

    def test_not_accepts_epsilon(self):
        assert not regex_accepts_epsilon("a")
        assert not regex_accepts_epsilon("a+")

    def test_sample(self):
        w = regex_sample("abc")
        assert w is not None
        assert ''.join(w) == "abc"

    def test_sample_star(self):
        w = regex_sample("a*")
        # Should be shortest: empty string
        assert w is not None
        assert w == []


# ============================================================================
# 8. Brzozowski Derivative Tests
# ============================================================================

class TestDerivatives:
    def test_nullable_epsilon(self):
        assert nullable(REpsilon())

    def test_nullable_star(self):
        assert nullable(RStar(RLit('a')))

    def test_nullable_optional(self):
        assert nullable(ROptional(RLit('a')))

    def test_not_nullable_literal(self):
        assert not nullable(RLit('a'))

    def test_not_nullable_plus(self):
        assert not nullable(RPlus(RLit('a')))

    def test_nullable_concat(self):
        assert nullable(RConcat(RStar(RLit('a')), REpsilon()))

    def test_derivative_literal_match(self):
        d = derivative(RLit('a'), 'a')
        assert nullable(d)

    def test_derivative_literal_no_match(self):
        d = derivative(RLit('a'), 'b')
        assert d.kind == RegexKind.EMPTY

    def test_derivative_match_word(self):
        assert derivative_match("abc", "abc")

    def test_derivative_no_match(self):
        assert not derivative_match("abc", "abd")

    def test_derivative_star(self):
        assert derivative_match("a*", "")
        assert derivative_match("a*", "a")
        assert derivative_match("a*", "aaa")
        assert not derivative_match("a*", "b")

    def test_derivative_alt(self):
        assert derivative_match("a|b", "a")
        assert derivative_match("a|b", "b")
        assert not derivative_match("a|b", "c")

    def test_derivative_complex(self):
        assert derivative_match("(a|b)*c", "abc")
        assert derivative_match("(a|b)*c", "c")
        assert not derivative_match("(a|b)*c", "d")

    def test_derivative_optional(self):
        assert derivative_match("a?b", "b")
        assert derivative_match("a?b", "ab")
        assert not derivative_match("a?b", "aab")


# ============================================================================
# 9. Regex Compare (Full Analysis) Tests
# ============================================================================

class TestCompare:
    def test_equivalent_compare(self):
        result = regex_compare("a|b", "b|a")
        assert result['equivalent']
        assert result['subset_1_2']
        assert result['subset_2_1']
        assert result['witness_in_1_not_2'] is None
        assert result['witness_in_2_not_1'] is None

    def test_subset_compare(self):
        result = regex_compare("a", "a|b")
        assert not result['equivalent']
        assert result['subset_1_2']
        assert not result['subset_2_1']
        assert result['witness_in_1_not_2'] is None
        assert result['witness_in_2_not_1'] is not None

    def test_incomparable_compare(self):
        result = regex_compare("[a-c]", "[b-d]")
        assert not result['equivalent']
        assert not result['subset_1_2']
        assert not result['subset_2_1']
        assert result['witness_in_1_not_2'] is not None
        assert result['witness_in_2_not_1'] is not None


# ============================================================================
# 10. Regex Operations via SFA Tests
# ============================================================================

class TestOperations:
    def test_union(self):
        sfa = regex_union("a", "b")
        assert sfa.accepts("a")
        assert sfa.accepts("b")
        assert not sfa.accepts("c")

    def test_intersection_sfa(self):
        sfa = regex_intersection("[a-z]", "[m-z]")
        assert sfa.accepts("m")
        assert sfa.accepts("z")
        assert not sfa.accepts("a")

    def test_complement(self):
        sfa = regex_complement("a")
        assert not sfa.accepts("a")
        assert sfa.accepts("b")
        assert sfa.accepts("")
        assert sfa.accepts("aa")

    def test_difference_sfa(self):
        sfa = regex_difference("[a-z]", "[a-m]")
        assert sfa.accepts("n")
        assert sfa.accepts("z")
        assert not sfa.accepts("a")
        assert not sfa.accepts("m")


# ============================================================================
# 11. Regex Utilities Tests
# ============================================================================

class TestUtilities:
    def test_regex_to_string_literal(self):
        r = parse_regex("abc")
        s = regex_to_string(r)
        assert 'a' in s and 'b' in s and 'c' in s

    def test_regex_to_string_star(self):
        r = parse_regex("a*")
        s = regex_to_string(r)
        assert '*' in s

    def test_regex_to_string_class(self):
        r = parse_regex("[a-z]")
        s = regex_to_string(r)
        assert '[' in s and ']' in s

    def test_regex_size(self):
        r = parse_regex("a")
        assert regex_size(r) == 1

    def test_regex_size_concat(self):
        r = parse_regex("abc")
        assert regex_size(r) == 4  # concat + 3 literals

    def test_regex_size_star(self):
        r = parse_regex("a*")
        assert regex_size(r) == 2  # star + literal

    def test_roundtrip_compile_match(self):
        """Compile and derivative match should agree."""
        patterns = ["a", "ab", "a|b", "a*", "a+", "a?", "(a|b)*c", "[a-z]+"]
        words = ["", "a", "b", "ab", "abc", "c", "aac", "test"]
        algebra = CharAlgebra()
        for p in patterns:
            sfa = compile_regex(p, algebra)
            for w in words:
                sfa_result = sfa.accepts(w)
                deriv_result = derivative_match(p, w, algebra)
                assert sfa_result == deriv_result, \
                    f"Mismatch for pattern={p!r} word={w!r}: sfa={sfa_result} deriv={deriv_result}"


# ============================================================================
# 12. Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    def test_single_char_alphabet(self):
        alg = CharAlgebra("a")
        sfa = compile_regex("a*", alg)
        assert sfa.accepts("")
        assert sfa.accepts("a")
        assert sfa.accepts("aaa")

    def test_empty_alternation_branch(self):
        """a| means a or epsilon."""
        sfa = compile_regex("a|")
        assert sfa.accepts("a")
        assert sfa.accepts("")
        assert not sfa.accepts("b")

    def test_multiple_quantifiers(self):
        """a*+ should parse as (a*)+ which = a*"""
        sfa = compile_regex("a*+")
        assert sfa.accepts("")
        assert sfa.accepts("a")
        assert sfa.accepts("aaa")

    def test_nested_alternatives(self):
        sfa = compile_regex("(a|(b|c))")
        assert sfa.accepts("a")
        assert sfa.accepts("b")
        assert sfa.accepts("c")
        assert not sfa.accepts("d")

    def test_long_alternation(self):
        sfa = compile_regex("a|b|c|d|e")
        for c in "abcde":
            assert sfa.accepts(c)
        assert not sfa.accepts("f")

    def test_escaped_special_chars(self):
        sfa = compile_regex("\\(\\)")
        assert sfa.accepts("()")
        assert not sfa.accepts("a")

    def test_char_class_with_dash(self):
        """[a-] should treat - as literal at end."""
        # Our parser requires - to be between chars, so [a\\-] is safer
        r = parse_regex("[a\\-]")
        assert r.kind == RegexKind.CHAR_CLASS

    def test_determinization_correctness(self):
        """Ensure NFA and DFA accept same language."""
        pattern = "(a|b)*abb"
        nfa = compile_regex(pattern)
        dfa = compile_regex_dfa(pattern)
        test_words = ["abb", "aabb", "babb", "ababb", "ab", "a", ""]
        for w in test_words:
            assert nfa.accepts(w) == dfa.accepts(w), f"Mismatch for {w!r}"

    def test_minimization_correctness(self):
        """Minimized DFA should accept same language."""
        pattern = "(a|b)*abb"
        dfa = compile_regex_dfa(pattern)
        min_dfa = compile_regex_min(pattern)
        test_words = ["abb", "aabb", "babb", "ababb", "ab", "a", ""]
        for w in test_words:
            assert dfa.accepts(w) == min_dfa.accepts(w), f"Mismatch for {w!r}"
        assert min_dfa.count_states() <= dfa.count_states()

    def test_minimization_reduces_states(self):
        """Redundant pattern should minimize."""
        # (a|a) is same as a, should minimize to 2-state DFA
        min_dfa = compile_regex_min("a|a")
        assert min_dfa.count_states() <= 3

    def test_complement_double(self):
        """Complement of complement = original."""
        pattern = "abc"
        sfa = compile_regex_dfa(pattern)
        comp = regex_complement(pattern)
        comp2 = sfa_complement(comp)
        test_words = ["abc", "ab", "abcd", "", "a"]
        for w in test_words:
            assert sfa.accepts(w) == comp2.accepts(w), f"Mismatch for {w!r}"


# Import for complement
from symbolic_automata import sfa_complement


# ============================================================================
# 13. Practical Pattern Tests
# ============================================================================

class TestPractical:
    def test_identifier_pattern(self):
        """[a-zA-Z_][a-zA-Z0-9_]* -- identifier regex."""
        sfa = compile_regex("[a-zA-Z_][a-zA-Z0-9_]*")
        assert sfa.accepts("foo")
        assert sfa.accepts("_bar")
        assert sfa.accepts("x123")
        assert not sfa.accepts("123")
        assert not sfa.accepts("")

    def test_integer_literal(self):
        """Optional sign + digits."""
        # Using alternation for optional minus
        sfa = compile_regex("(\\-)?\\d+")
        assert sfa.accepts("0")
        assert sfa.accepts("123")
        assert sfa.accepts("-42")
        assert not sfa.accepts("")
        assert not sfa.accepts("-")

    def test_simple_url(self):
        """Very simplified URL pattern."""
        alg = CharAlgebra("abcdefghijklmnopqrstuvwxyz0123456789.:/")
        sfa = compile_regex("[a-z]+://[a-z0-9.]+", alg)
        assert sfa.accepts("http://example.com")
        assert not sfa.accepts("example.com")

    def test_phone_number(self):
        """Simple phone: ddd-ddd-dddd."""
        sfa = compile_regex("\\d\\d\\d\\-\\d\\d\\d\\-\\d\\d\\d\\d")
        assert sfa.accepts("123-456-7890")
        assert not sfa.accepts("123-456-789")
        assert not sfa.accepts("1234567890")

    def test_hex_color(self):
        """#[0-9a-f]{6} -- simulated with explicit repetition."""
        sfa = compile_regex("#[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]")
        assert sfa.accepts("#ff00aa")
        assert sfa.accepts("#000000")
        assert not sfa.accepts("#fff")
        assert not sfa.accepts("ff00aa")

    def test_csv_field(self):
        """Simple CSV field: word chars or empty."""
        sfa = compile_regex("\\w*")
        assert sfa.accepts("")
        assert sfa.accepts("hello")
        assert sfa.accepts("test_123")

    def test_equivalence_practical(self):
        """Two ways to write 'one or two digits'."""
        assert regex_equivalent("\\d\\d?", "\\d|\\d\\d")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
