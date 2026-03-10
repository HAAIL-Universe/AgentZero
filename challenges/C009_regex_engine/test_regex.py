"""Tests for regex engine -- C009."""
import pytest
from regex_engine import Regex, fullmatch, search, findall, compile


class TestLiterals:
    def test_exact_match(self):
        assert fullmatch("hello", "hello")

    def test_no_match(self):
        assert not fullmatch("hello", "world")

    def test_single_char(self):
        assert fullmatch("a", "a")

    def test_wrong_length(self):
        assert not fullmatch("ab", "abc")

    def test_empty_pattern_empty_string(self):
        assert fullmatch("", "")

    def test_empty_pattern_nonempty_string(self):
        assert not fullmatch("", "a")


class TestDot:
    def test_dot_matches_char(self):
        assert fullmatch("a.c", "abc")
        assert fullmatch("a.c", "axc")

    def test_dot_no_newline(self):
        assert not fullmatch("a.c", "a\nc")

    def test_multiple_dots(self):
        assert fullmatch("...", "xyz")

    def test_dot_wrong_length(self):
        assert not fullmatch(".", "ab")


class TestStar:
    def test_zero_matches(self):
        assert fullmatch("a*", "")

    def test_one_match(self):
        assert fullmatch("a*", "a")

    def test_many_matches(self):
        assert fullmatch("a*", "aaaa")

    def test_star_with_other(self):
        assert fullmatch("ba*c", "bc")
        assert fullmatch("ba*c", "bac")
        assert fullmatch("ba*c", "baaac")

    def test_dot_star(self):
        assert fullmatch(".*", "anything goes")

    def test_star_no_match(self):
        assert not fullmatch("a*", "b")


class TestPlus:
    def test_one_match(self):
        assert fullmatch("a+", "a")

    def test_many_matches(self):
        assert fullmatch("a+", "aaa")

    def test_zero_no_match(self):
        assert not fullmatch("a+", "")

    def test_plus_with_other(self):
        assert fullmatch("ba+c", "bac")
        assert fullmatch("ba+c", "baaac")
        assert not fullmatch("ba+c", "bc")


class TestOptional:
    def test_present(self):
        assert fullmatch("colou?r", "colour")

    def test_absent(self):
        assert fullmatch("colou?r", "color")

    def test_optional_only(self):
        assert fullmatch("a?", "")
        assert fullmatch("a?", "a")


class TestAlternation:
    def test_left(self):
        assert fullmatch("cat|dog", "cat")

    def test_right(self):
        assert fullmatch("cat|dog", "dog")

    def test_neither(self):
        assert not fullmatch("cat|dog", "bird")

    def test_nested(self):
        assert fullmatch("a|b|c", "c")

    def test_alternation_with_concat(self):
        assert fullmatch("ab|cd", "ab")
        assert fullmatch("ab|cd", "cd")
        assert not fullmatch("ab|cd", "ac")


class TestGrouping:
    def test_group_repeat(self):
        assert fullmatch("(ab)+", "ababab")

    def test_group_alternation(self):
        assert fullmatch("(a|b)c", "ac")
        assert fullmatch("(a|b)c", "bc")

    def test_nested_groups(self):
        assert fullmatch("((ab))+", "abab")

    def test_group_star(self):
        assert fullmatch("(ab)*", "")
        assert fullmatch("(ab)*", "abab")

    def test_group_optional(self):
        assert fullmatch("a(bc)?d", "ad")
        assert fullmatch("a(bc)?d", "abcd")


class TestCharClass:
    def test_basic_class(self):
        assert fullmatch("[abc]", "a")
        assert fullmatch("[abc]", "b")
        assert not fullmatch("[abc]", "d")

    def test_range(self):
        assert fullmatch("[a-z]", "m")
        assert not fullmatch("[a-z]", "A")

    def test_negated_class(self):
        assert fullmatch("[^abc]", "d")
        assert not fullmatch("[^abc]", "a")

    def test_class_with_quantifier(self):
        assert fullmatch("[0-9]+", "12345")
        assert not fullmatch("[0-9]+", "123a5")

    def test_multiple_ranges(self):
        assert fullmatch("[a-zA-Z]", "Z")
        assert fullmatch("[a-zA-Z]", "m")
        assert not fullmatch("[a-zA-Z]", "5")


class TestAnchors:
    def test_caret_search(self):
        assert search("^hello", "hello world")
        assert not search("^hello", "say hello")

    def test_dollar_search(self):
        assert search("world$", "hello world")
        assert not search("world$", "world hello")

    def test_both_anchors(self):
        assert fullmatch("^abc$", "abc")
        assert not search("^abc$", "xabc")

    def test_anchor_with_quantifier(self):
        assert search("^a+$", "aaa")
        assert not search("^a+$", "aab")


class TestEscaping:
    def test_escape_dot(self):
        assert fullmatch("a\\.b", "a.b")
        assert not fullmatch("a\\.b", "axb")

    def test_escape_star(self):
        assert fullmatch("a\\*", "a*")

    def test_escape_backslash(self):
        assert fullmatch("a\\\\b", "a\\b")

    def test_escape_paren(self):
        assert fullmatch("\\(a\\)", "(a)")


class TestShorthandClasses:
    def test_digit(self):
        assert fullmatch("\\d+", "12345")
        assert not fullmatch("\\d+", "123a5")

    def test_word(self):
        assert fullmatch("\\w+", "hello_world123")
        assert not fullmatch("\\w+", "hello world")

    def test_whitespace(self):
        assert fullmatch("\\s+", "  \t\n")
        assert not fullmatch("\\s+", "  a ")


class TestSearch:
    def test_find_in_middle(self):
        assert search("world", "hello world!")

    def test_find_at_start(self):
        assert search("hello", "hello world")

    def test_not_found(self):
        assert not search("xyz", "hello world")

    def test_pattern_with_quantifiers(self):
        assert search("a+b", "xxxaaabyyy")


class TestFindall:
    def test_basic(self):
        assert findall("a+", "aabaaab") == ["aa", "aaa"]

    def test_no_matches(self):
        assert findall("x", "hello") == []

    def test_overlapping_region(self):
        result = findall("[0-9]+", "abc123def456")
        assert result == ["123", "456"]

    def test_single_char_matches(self):
        assert findall(".", "abc") == ["a", "b", "c"]

    def test_anchored_findall(self):
        result = findall("^a+", "aabaa")
        assert result == ["aa"]


class TestEdgeCases:
    def test_catastrophic_backtracking_immunity(self):
        """Thompson NFA should handle this in linear time."""
        r = compile("a*a*a*a*a*b")
        # Should be fast (not exponential)
        assert not r.fullmatch("a" * 25)

    def test_nested_quantifiers(self):
        assert fullmatch("(a*)*", "aaa")

    def test_empty_alternation_branch(self):
        assert fullmatch("a|", "")
        assert fullmatch("a|", "a")

    def test_complex_pattern(self):
        r = compile("[a-z]+@[a-z]+\\.[a-z]+")
        assert r.fullmatch("user@example.com")
        assert not r.fullmatch("USER@example.com")
        assert not r.fullmatch("user@example")

    def test_repeated_groups_with_alternation(self):
        assert fullmatch("(ab|cd)+", "abcdab")
        assert not fullmatch("(ab|cd)+", "abce")

    def test_dot_star_greedy(self):
        result = findall("a.*b", "axbyazb")
        # Should find the longest match (greedy)
        assert result == ["axbyazb"]

    def test_compile_reuse(self):
        r = compile("hello")
        assert r.search("say hello there")
        assert r.fullmatch("hello")
        assert not r.fullmatch("hello!")


class TestErrorHandling:
    def test_unmatched_paren(self):
        with pytest.raises(ValueError):
            compile("(abc")

    def test_trailing_backslash(self):
        with pytest.raises(ValueError):
            compile("abc\\")

    def test_unterminated_char_class(self):
        with pytest.raises(ValueError):
            compile("[abc")
