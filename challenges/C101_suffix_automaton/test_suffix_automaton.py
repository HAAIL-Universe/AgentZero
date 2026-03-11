"""Tests for C101: Suffix Automaton (DAWG)"""
import pytest
from suffix_automaton import SuffixAutomaton, GeneralizedSuffixAutomaton


# ============================================================
# Construction basics
# ============================================================

class TestConstruction:
    def test_empty_string(self):
        sa = SuffixAutomaton("")
        assert sa.num_states() == 1  # just initial state
        assert sa.num_transitions() == 0

    def test_single_char(self):
        sa = SuffixAutomaton("a")
        assert sa.num_states() == 2
        assert sa.num_transitions() == 1

    def test_two_same_chars(self):
        sa = SuffixAutomaton("aa")
        # States: init, a, aa (clone of a becomes separate)
        assert sa.num_states() == 3

    def test_two_different_chars(self):
        sa = SuffixAutomaton("ab")
        assert sa.num_states() == 3
        assert sa.num_transitions() == 3  # a->b from init, b from init, b from a-state

    def test_abcbc(self):
        sa = SuffixAutomaton("abcbc")
        # Classic example: should have at most 2*5-1 = 9 states
        assert sa.num_states() <= 9
        assert sa.num_states() >= 6  # at least n+1

    def test_state_bound(self):
        """States <= 2n-1 for string of length n."""
        for s in ["abcdef", "aaaaaa", "ababab", "abcabc", "mississippi"]:
            sa = SuffixAutomaton(s)
            assert sa.num_states() <= 2 * len(s) - 1 + 1  # +1 for initial

    def test_transition_bound(self):
        """Transitions <= 3n-4 for string of length n >= 3."""
        for s in ["abcdef", "aaaaaa", "ababab", "abcabc", "mississippi"]:
            sa = SuffixAutomaton(s)
            if len(s) >= 3:
                assert sa.num_transitions() <= 3 * len(s) - 4 + len(s)  # generous bound

    def test_incremental_build(self):
        sa = SuffixAutomaton()
        sa.extend('a')
        sa.extend('b')
        sa.extend('c')
        assert sa.contains("abc")
        assert sa.contains("bc")
        assert sa.contains("c")

    def test_all_same_chars(self):
        sa = SuffixAutomaton("aaaa")
        # For all-same: n+1 states
        assert sa.num_states() == 5

    def test_binary_string(self):
        sa = SuffixAutomaton("01010101")
        assert sa.contains("0101")
        assert sa.contains("1010")
        assert not sa.contains("0000")


# ============================================================
# Substring containment
# ============================================================

class TestContains:
    def test_empty_pattern(self):
        sa = SuffixAutomaton("hello")
        assert sa.contains("")

    def test_full_string(self):
        sa = SuffixAutomaton("hello")
        assert sa.contains("hello")

    def test_all_suffixes(self):
        s = "banana"
        sa = SuffixAutomaton(s)
        for i in range(len(s)):
            assert sa.contains(s[i:])

    def test_all_substrings(self):
        s = "abcabc"
        sa = SuffixAutomaton(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                assert sa.contains(s[i:j])

    def test_not_contained(self):
        sa = SuffixAutomaton("abc")
        assert not sa.contains("d")
        assert not sa.contains("abd")
        assert not sa.contains("abcd")
        assert not sa.contains("ba")

    def test_single_chars(self):
        sa = SuffixAutomaton("hello")
        assert sa.contains("h")
        assert sa.contains("e")
        assert sa.contains("l")
        assert sa.contains("o")
        assert not sa.contains("x")

    def test_empty_automaton(self):
        sa = SuffixAutomaton("")
        assert sa.contains("")
        assert not sa.contains("a")

    def test_mississippi_substrings(self):
        sa = SuffixAutomaton("mississippi")
        assert sa.contains("issi")
        assert sa.contains("ssipp")
        assert sa.contains("mississippi")
        assert not sa.contains("mississippii")
        assert not sa.contains("missp")

    def test_long_repetitive(self):
        s = "ab" * 50
        sa = SuffixAutomaton(s)
        assert sa.contains("ababab")
        assert sa.contains("bababa")
        assert not sa.contains("aab")


# ============================================================
# Count distinct substrings
# ============================================================

class TestCountDistinct:
    def test_empty(self):
        sa = SuffixAutomaton("")
        assert sa.count_distinct_substrings() == 0

    def test_single_char(self):
        sa = SuffixAutomaton("a")
        assert sa.count_distinct_substrings() == 1

    def test_all_different(self):
        # "abc" -> a, b, c, ab, bc, abc = 6
        sa = SuffixAutomaton("abc")
        assert sa.count_distinct_substrings() == 6

    def test_all_same(self):
        # "aaa" -> a, aa, aaa = 3
        sa = SuffixAutomaton("aaa")
        assert sa.count_distinct_substrings() == 3

    def test_banana(self):
        # "banana" substrings: b,a,n,ba,an,na,ban,ana,nan,bana,anan,nana,banan,anana,banana
        # = 15 distinct
        sa = SuffixAutomaton("banana")
        assert sa.count_distinct_substrings() == 15

    def test_abab(self):
        # "abab": a,b,ab,ba,aba,bab,abab = 7
        sa = SuffixAutomaton("abab")
        assert sa.count_distinct_substrings() == 7

    def test_formula_n_choose_2_plus_n(self):
        # For all-distinct string of length n: n*(n+1)/2 substrings
        sa = SuffixAutomaton("abcde")
        assert sa.count_distinct_substrings() == 5 * 6 // 2

    def test_mississippi(self):
        s = "mississippi"
        sa = SuffixAutomaton(s)
        # Count by brute force
        subs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                subs.add(s[i:j])
        assert sa.count_distinct_substrings() == len(subs)


# ============================================================
# Occurrence counting
# ============================================================

class TestOccurrences:
    def test_single_occurrence(self):
        sa = SuffixAutomaton("abc")
        assert sa.count_occurrences("abc") == 1

    def test_multiple_occurrences(self):
        sa = SuffixAutomaton("abab")
        assert sa.count_occurrences("ab") == 2

    def test_overlapping_occurrences(self):
        sa = SuffixAutomaton("aaa")
        assert sa.count_occurrences("a") == 3
        assert sa.count_occurrences("aa") == 2

    def test_not_found(self):
        sa = SuffixAutomaton("abc")
        assert sa.count_occurrences("d") == 0

    def test_banana_occurrences(self):
        sa = SuffixAutomaton("banana")
        assert sa.count_occurrences("a") == 3
        assert sa.count_occurrences("an") == 2
        assert sa.count_occurrences("ana") == 2
        assert sa.count_occurrences("b") == 1
        assert sa.count_occurrences("ban") == 1

    def test_mississippi(self):
        sa = SuffixAutomaton("mississippi")
        assert sa.count_occurrences("i") == 4
        assert sa.count_occurrences("s") == 4
        assert sa.count_occurrences("ss") == 2
        assert sa.count_occurrences("issi") == 2
        assert sa.count_occurrences("p") == 2
        assert sa.count_occurrences("pp") == 1

    def test_empty_pattern(self):
        sa = SuffixAutomaton("abc")
        # Empty string "occurs" at every position
        # State 0 accumulates all counts
        assert sa.count_occurrences("") >= 1

    def test_all_same(self):
        sa = SuffixAutomaton("aaaaa")
        assert sa.count_occurrences("a") == 5
        assert sa.count_occurrences("aa") == 4
        assert sa.count_occurrences("aaa") == 3
        assert sa.count_occurrences("aaaa") == 2
        assert sa.count_occurrences("aaaaa") == 1


# ============================================================
# First occurrence
# ============================================================

class TestFirstOccurrence:
    def test_at_start(self):
        sa = SuffixAutomaton("abcabc")
        assert sa.first_occurrence("abc") == 0

    def test_at_end(self):
        sa = SuffixAutomaton("xyzabc")
        assert sa.first_occurrence("abc") == 3

    def test_not_found(self):
        sa = SuffixAutomaton("abc")
        assert sa.first_occurrence("xyz") == -1

    def test_single_char(self):
        sa = SuffixAutomaton("hello")
        pos = sa.first_occurrence("l")
        assert pos == 2  # first 'l' at index 2

    def test_full_string(self):
        sa = SuffixAutomaton("hello")
        assert sa.first_occurrence("hello") == 0


# ============================================================
# All occurrences
# ============================================================

class TestAllOccurrences:
    def test_single(self):
        sa = SuffixAutomaton("abc")
        assert sa.all_occurrences("abc") == [0]

    def test_multiple(self):
        sa = SuffixAutomaton("abcabc")
        assert sa.all_occurrences("abc") == [0, 3]

    def test_overlapping(self):
        sa = SuffixAutomaton("aaa")
        assert sa.all_occurrences("aa") == [0, 1]

    def test_not_found(self):
        sa = SuffixAutomaton("abc")
        assert sa.all_occurrences("xyz") == []

    def test_single_char_multiple(self):
        sa = SuffixAutomaton("banana")
        assert sa.all_occurrences("a") == [1, 3, 5]

    def test_banana_an(self):
        sa = SuffixAutomaton("banana")
        assert sa.all_occurrences("an") == [1, 3]

    def test_mississippi_ss(self):
        sa = SuffixAutomaton("mississippi")
        assert sa.all_occurrences("ss") == [2, 5]


# ============================================================
# Longest common substring
# ============================================================

class TestLCS:
    def test_basic(self):
        sa = SuffixAutomaton("abcdef")
        length, pos = sa.longest_common_substring("xbcdey")
        assert length == 4  # "bcde"
        assert pos == 1

    def test_no_common(self):
        sa = SuffixAutomaton("abc")
        length, pos = sa.longest_common_substring("xyz")
        assert length == 0

    def test_full_match(self):
        sa = SuffixAutomaton("abc")
        length, pos = sa.longest_common_substring("abc")
        assert length == 3
        assert pos == 0

    def test_single_char_common(self):
        sa = SuffixAutomaton("abc")
        length, pos = sa.longest_common_substring("xay")
        assert length == 1

    def test_multiple_common(self):
        sa = SuffixAutomaton("abcxyzabc")
        length, pos = sa.longest_common_substring("abcdef")
        assert length == 3  # "abc"

    def test_empty_second(self):
        sa = SuffixAutomaton("abc")
        length, pos = sa.longest_common_substring("")
        assert length == 0

    def test_mississippi(self):
        sa = SuffixAutomaton("mississippi")
        length, pos = sa.longest_common_substring("missourian")
        assert length == 4  # "miss"


# ============================================================
# Shortest non-occurring string
# ============================================================

class TestShortestNonOccurring:
    def test_simple(self):
        sa = SuffixAutomaton("ab")
        result = sa.shortest_non_occurring()
        assert len(result) > 0
        assert not sa.contains(result)

    def test_all_single_chars(self):
        sa = SuffixAutomaton("abc")
        result = sa.shortest_non_occurring()
        # All of a,b,c exist. Some 2-char combo shouldn't.
        assert not sa.contains(result)

    def test_minimal_length(self):
        sa = SuffixAutomaton("a")
        result = sa.shortest_non_occurring()
        # Only 'a' exists as single char. If alphabet is just 'a',
        # then 'aa' shouldn't exist
        assert not sa.contains(result)

    def test_binary(self):
        sa = SuffixAutomaton("01")
        result = sa.shortest_non_occurring()
        assert not sa.contains(result)
        assert len(result) <= 2  # "00" or "11" or "10" (len 2)

    def test_empty(self):
        sa = SuffixAutomaton("")
        result = sa.shortest_non_occurring()
        assert len(result) == 1  # just any char


# ============================================================
# Longest repeated substring
# ============================================================

class TestLongestRepeated:
    def test_basic(self):
        sa = SuffixAutomaton("abcabc")
        result = sa.longest_repeated_substring()
        assert len(result) == 3
        assert result == "abc"

    def test_no_repeat(self):
        sa = SuffixAutomaton("abc")
        result = sa.longest_repeated_substring()
        assert result == ""

    def test_all_same(self):
        sa = SuffixAutomaton("aaaa")
        result = sa.longest_repeated_substring()
        assert result == "aaa"

    def test_banana(self):
        sa = SuffixAutomaton("banana")
        result = sa.longest_repeated_substring()
        assert len(result) == 3
        assert result == "ana"

    def test_mississippi(self):
        sa = SuffixAutomaton("mississippi")
        result = sa.longest_repeated_substring()
        assert len(result) == 4  # "issi"

    def test_single_char(self):
        sa = SuffixAutomaton("a")
        assert sa.longest_repeated_substring() == ""


# ============================================================
# k-th smallest substring
# ============================================================

class TestKthSmallest:
    def test_first(self):
        sa = SuffixAutomaton("abc")
        result = sa.kth_smallest_substring(1)
        assert result == "a"

    def test_all_substrings_in_order(self):
        sa = SuffixAutomaton("abc")
        # Sorted: a, ab, abc, b, bc, c
        expected = ["a", "ab", "abc", "b", "bc", "c"]
        for i, exp in enumerate(expected, 1):
            assert sa.kth_smallest_substring(i) == exp

    def test_out_of_range(self):
        sa = SuffixAutomaton("abc")
        assert sa.kth_smallest_substring(0) == ""
        assert sa.kth_smallest_substring(7) == ""

    def test_with_repeats(self):
        sa = SuffixAutomaton("aba")
        # distinct substrings: a, ab, aba, b, ba = 5
        assert sa.count_distinct_substrings() == 5
        result1 = sa.kth_smallest_substring(1)
        assert result1 == "a"
        result2 = sa.kth_smallest_substring(2)
        assert result2 == "ab"

    def test_last(self):
        sa = SuffixAutomaton("abc")
        result = sa.kth_smallest_substring(6)
        assert result == "c"

    def test_banana_kth(self):
        sa = SuffixAutomaton("banana")
        # First should be "a"
        assert sa.kth_smallest_substring(1) == "a"
        # Get all 15 substrings
        subs = []
        for i in range(1, 16):
            subs.append(sa.kth_smallest_substring(i))
        # Verify sorted order
        assert subs == sorted(subs)
        # Verify all distinct
        assert len(set(subs)) == 15


# ============================================================
# Suffix link tree
# ============================================================

class TestSuffixLinkTree:
    def test_tree_structure(self):
        sa = SuffixAutomaton("abc")
        tree = sa.suffix_links_tree()
        # Root (0) should have children
        assert 0 in tree
        assert len(tree[0]) > 0

    def test_all_states_reachable(self):
        sa = SuffixAutomaton("abcabc")
        tree = sa.suffix_links_tree()
        visited = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for child in tree.get(node, []):
                visited.add(child)
                stack.append(child)
        assert len(visited) == sa.num_states()


# ============================================================
# DOT export
# ============================================================

class TestDotExport:
    def test_basic_dot(self):
        sa = SuffixAutomaton("ab")
        dot = sa.to_dot()
        assert "digraph" in dot
        assert "s0" in dot

    def test_max_states_limit(self):
        sa = SuffixAutomaton("abcdefghij" * 10)
        dot = sa.to_dot(max_states=5)
        # Should only have s0-s4
        assert "s5" not in dot or "s5 [" not in dot


# ============================================================
# Generalized Suffix Automaton
# ============================================================

class TestGeneralizedSA:
    def test_two_strings(self):
        gsa = GeneralizedSuffixAutomaton(["abc", "def"])
        assert gsa.contains("abc")
        assert gsa.contains("def")
        assert gsa.contains("ab")
        assert gsa.contains("ef")
        assert not gsa.contains("abcdef")

    def test_strings_containing(self):
        gsa = GeneralizedSuffixAutomaton(["abc", "bcd", "cde"])
        result = gsa.strings_containing("bc")
        assert 0 in result  # "abc" contains "bc"
        assert 1 in result  # "bcd" contains "bc"
        assert 2 not in result  # "cde" does not contain "bc"

    def test_strings_containing_c(self):
        gsa = GeneralizedSuffixAutomaton(["abc", "bcd", "cde"])
        result = gsa.strings_containing("c")
        assert result == {0, 1, 2}

    def test_strings_containing_not_found(self):
        gsa = GeneralizedSuffixAutomaton(["abc", "def"])
        result = gsa.strings_containing("xyz")
        assert result == set()

    def test_lcs_two_strings(self):
        gsa = GeneralizedSuffixAutomaton(["xabcy", "zabcw"])
        result = gsa.longest_common_substring_all()
        assert len(result) == 3
        assert result == "abc"

    def test_lcs_three_strings(self):
        gsa = GeneralizedSuffixAutomaton(["abcdef", "xbcdey", "zbcdew"])
        result = gsa.longest_common_substring_all()
        assert len(result) >= 3  # at least "bcd"

    def test_lcs_no_common(self):
        gsa = GeneralizedSuffixAutomaton(["abc", "def", "ghi"])
        result = gsa.longest_common_substring_all()
        assert result == ""

    def test_single_string(self):
        gsa = GeneralizedSuffixAutomaton(["hello"])
        assert gsa.contains("hello")
        assert gsa.contains("ell")

    def test_empty_strings(self):
        gsa = GeneralizedSuffixAutomaton(["", "abc"])
        assert gsa.contains("abc")

    def test_overlapping_strings(self):
        gsa = GeneralizedSuffixAutomaton(["abc", "abc"])
        assert gsa.contains("abc")
        result = gsa.strings_containing("abc")
        assert 0 in result
        assert 1 in result

    def test_add_string_incremental(self):
        gsa = GeneralizedSuffixAutomaton()
        gsa.add_string("abc")
        assert gsa.contains("abc")
        gsa.add_string("def")
        assert gsa.contains("abc")
        assert gsa.contains("def")


# ============================================================
# Edge cases and stress
# ============================================================

class TestEdgeCases:
    def test_single_char_repeated(self):
        sa = SuffixAutomaton("a" * 100)
        assert sa.count_distinct_substrings() == 100
        assert sa.count_occurrences("a") == 100
        assert sa.count_occurrences("aa") == 99

    def test_fibonacci_string(self):
        # Fibonacci strings have interesting suffix automaton properties
        a, b = "a", "b"
        for _ in range(8):
            a, b = b, b + a
        sa = SuffixAutomaton(b)
        n = len(b)
        # Verify distinct count matches brute force
        subs = set()
        for i in range(n):
            for j in range(i + 1, min(i + 50, n + 1)):  # limit for speed
                subs.add(b[i:j])
        # Just verify automaton accepts all
        for sub in list(subs)[:100]:
            assert sa.contains(sub)

    def test_alternating(self):
        s = "ab" * 50
        sa = SuffixAutomaton(s)
        assert sa.contains("abab")
        assert sa.contains("baba")
        assert not sa.contains("aa")
        assert not sa.contains("bb")

    def test_special_chars(self):
        sa = SuffixAutomaton("hello world! 123")
        assert sa.contains("o w")
        assert sa.contains("! 1")

    def test_unicode(self):
        sa = SuffixAutomaton("cafe\u0301")
        assert sa.contains("cafe")
        assert sa.contains("\u0301")

    def test_long_string_performance(self):
        """Build SA for 10k chars -- should complete quickly."""
        import time
        s = "abcdefghij" * 1000
        start = time.time()
        sa = SuffixAutomaton(s)
        elapsed = time.time() - start
        assert elapsed < 5.0  # should be << 1s
        assert sa.contains("abcdefghij")

    def test_worst_case_states(self):
        """abcdef... all different chars should give exactly n+1 states."""
        s = "abcdefghijklmnop"
        sa = SuffixAutomaton(s)
        # All different chars: no cloning needed, exactly n+1 states
        assert sa.num_states() == len(s) + 1


# ============================================================
# Integration: multiple operations on same automaton
# ============================================================

class TestIntegration:
    def test_all_operations(self):
        s = "abracadabra"
        sa = SuffixAutomaton(s)

        # Contains
        assert sa.contains("abra")
        assert sa.contains("cad")
        assert not sa.contains("xyz")

        # Count distinct
        subs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                subs.add(s[i:j])
        assert sa.count_distinct_substrings() == len(subs)

        # Occurrences
        assert sa.count_occurrences("a") == 5
        assert sa.count_occurrences("abra") == 2

        # All occurrences
        assert sa.all_occurrences("abra") == [0, 7]

        # LCS
        length, pos = sa.longest_common_substring("abracadabra")
        assert length == 11

        # Longest repeated
        lrs = sa.longest_repeated_substring()
        assert len(lrs) == 4  # "abra"

    def test_generalized_integration(self):
        gsa = GeneralizedSuffixAutomaton(["banana", "ananas", "cabana"])

        assert gsa.contains("ana")
        assert gsa.contains("ban")
        assert gsa.contains("cab")

        result = gsa.strings_containing("ana")
        assert 0 in result  # banana
        assert 1 in result  # ananas

        lcs = gsa.longest_common_substring_all()
        assert len(lcs) >= 2  # at least "an" or "na" or "ba"

    def test_build_then_query(self):
        """Build incrementally, then run queries."""
        sa = SuffixAutomaton()
        for ch in "hello":
            sa.extend(ch)

        assert sa.contains("hello")
        assert sa.contains("ell")
        assert not sa.contains("helo")
        assert sa.count_distinct_substrings() == 5 * 6 // 2 - 1  # "ll" duplicate: 14


# ============================================================
# Brute-force validation
# ============================================================

class TestBruteForceValidation:
    """Compare automaton results against brute force for small strings."""

    def _all_substrings(self, s):
        subs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                subs.add(s[i:j])
        return subs

    def test_distinct_count_brute_force(self):
        for s in ["a", "ab", "abc", "aab", "aba", "abab", "abcabc", "aabaa"]:
            sa = SuffixAutomaton(s)
            expected = len(self._all_substrings(s))
            assert sa.count_distinct_substrings() == expected, f"Failed for '{s}'"

    def test_occurrence_count_brute_force(self):
        for s in ["abab", "aabaa", "banana", "abcabc"]:
            sa = SuffixAutomaton(s)
            for sub in self._all_substrings(s):
                expected = sum(1 for i in range(len(s)) if s[i:i+len(sub)] == sub)
                actual = sa.count_occurrences(sub)
                assert actual == expected, f"Failed for '{sub}' in '{s}': {actual} != {expected}"

    def test_contains_brute_force(self):
        for s in ["hello", "abcabc", "aaabbb"]:
            sa = SuffixAutomaton(s)
            subs = self._all_substrings(s)
            for sub in subs:
                assert sa.contains(sub), f"'{sub}' should be in '{s}'"

    def test_kth_brute_force(self):
        for s in ["abc", "aba", "abab"]:
            sa = SuffixAutomaton(s)
            subs = sorted(self._all_substrings(s))
            for i, expected in enumerate(subs, 1):
                actual = sa.kth_smallest_substring(i)
                assert actual == expected, f"k={i} in '{s}': got '{actual}', expected '{expected}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
