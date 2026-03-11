"""Tests for C087: Suffix Array."""
import pytest
from suffix_array import (
    SuffixArray, LCPArray, SuffixArraySearcher,
    EnhancedSuffixArray, MultiStringSuffixArray, sa_is
)


# =============================================================================
# SA-IS core algorithm tests
# =============================================================================

class TestSAIS:
    """Tests for the SA-IS algorithm."""

    def test_empty(self):
        assert sa_is([]) == []

    def test_single(self):
        assert sa_is([0]) == [0]

    def test_two_elements(self):
        assert sa_is([1, 0]) == [1, 0]

    def test_simple(self):
        # "ba$" -> [2, 1, 0]
        sa = sa_is([2, 1, 0])
        assert sa[0] == 2  # $ (sentinel)
        assert sa[1] == 1  # a$
        assert sa[2] == 0  # ba$

    def test_banana(self):
        # "banana$" -> [6, 5, 3, 1, 0, 4, 2]
        s = [2, 1, 14, 1, 14, 1, 0]  # b=2, a=1, n=14, sentinel=0
        sa = sa_is(s)
        assert sa[0] == 6  # $
        # Verify sorted order
        for i in range(1, len(sa)):
            s1 = s[sa[i - 1]:]
            s2 = s[sa[i]:]
            assert s1 < s2

    def test_all_same(self):
        s = [1, 1, 1, 0]
        sa = sa_is(s)
        assert sa == [3, 2, 1, 0]

    def test_descending(self):
        s = [4, 3, 2, 1, 0]
        sa = sa_is(s)
        assert sa[0] == 4  # sentinel
        for i in range(1, len(sa)):
            assert s[sa[i - 1]:] < s[sa[i]:]


# =============================================================================
# SuffixArray class tests
# =============================================================================

class TestSuffixArray:
    """Tests for SuffixArray class."""

    def test_string_construction(self):
        sa = SuffixArray("abc")
        assert len(sa) == 4  # includes sentinel

    def test_list_construction(self):
        sa = SuffixArray([0, 1, 2])
        assert len(sa) == 4

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            SuffixArray(123)

    def test_sorted_suffixes(self):
        sa = SuffixArray("banana")
        suffixes = sa.suffixes_sorted()
        # Should be sorted
        for i in range(1, len(suffixes)):
            assert suffixes[i - 1] < suffixes[i]

    def test_banana_suffixes(self):
        sa = SuffixArray("banana")
        suffixes = sa.suffixes_sorted()
        expected = ["a", "ana", "anana", "banana", "na", "nana"]
        assert suffixes == expected

    def test_getitem(self):
        sa = SuffixArray("ab")
        # SA should have 3 entries (a, b, sentinel)
        assert len(sa) == 3

    def test_suffix_method(self):
        sa = SuffixArray("abc")
        # suffix(0) should return the lexicographically smallest suffix
        s = sa.suffix(0)
        # It should be the sentinel or "abc" etc -- just check it's a string
        assert isinstance(s, str)

    def test_single_char(self):
        sa = SuffixArray("a")
        suffixes = sa.suffixes_sorted()
        assert suffixes == ["a"]

    def test_repeated_chars(self):
        sa = SuffixArray("aaa")
        suffixes = sa.suffixes_sorted()
        assert suffixes == ["a", "aa", "aaa"]

    def test_two_chars(self):
        sa = SuffixArray("ba")
        suffixes = sa.suffixes_sorted()
        assert suffixes == ["a", "ba"]

    def test_integer_list(self):
        sa = SuffixArray([3, 1, 2])
        suffixes = sa.suffixes_sorted()
        assert len(suffixes) == 3

    def test_longer_string(self):
        sa = SuffixArray("mississippi")
        suffixes = sa.suffixes_sorted()
        for i in range(1, len(suffixes)):
            assert suffixes[i - 1] < suffixes[i]


# =============================================================================
# LCPArray tests
# =============================================================================

class TestLCPArray:
    """Tests for LCP array (Kasai's algorithm)."""

    def test_basic_lcp(self):
        lcp = LCPArray("banana")
        assert len(lcp) > 0

    def test_no_repeated(self):
        lcp = LCPArray("abcde")
        # No repeated substrings longer than 0
        assert max(lcp.lcp[1:]) <= 1

    def test_longest_repeated_banana(self):
        lcp = LCPArray("banana")
        lrs = lcp.longest_repeated_substring()
        assert lrs == "ana"

    def test_longest_repeated_abcabc(self):
        lcp = LCPArray("abcabc")
        lrs = lcp.longest_repeated_substring()
        assert lrs == "abc"

    def test_longest_repeated_no_repeat(self):
        lcp = LCPArray("abcde")
        lrs = lcp.longest_repeated_substring()
        # Might be empty or single char
        assert len(lrs) <= 1

    def test_count_distinct_substrings(self):
        lcp = LCPArray("abc")
        # Distinct substrings: a, b, c, ab, bc, abc = 6
        count = lcp.count_distinct_substrings()
        assert count == 6

    def test_count_distinct_aab(self):
        lcp = LCPArray("aab")
        # a, aa, aab, ab, b = 5
        count = lcp.count_distinct_substrings()
        assert count == 5

    def test_from_suffix_array(self):
        sa = SuffixArray("hello")
        lcp = LCPArray("hello", sa)
        assert len(lcp) == sa.n

    def test_all_same_chars(self):
        lcp = LCPArray("aaaa")
        lrs = lcp.longest_repeated_substring()
        assert lrs == "aaa"

    def test_single_char(self):
        lcp = LCPArray("a")
        lrs = lcp.longest_repeated_substring()
        assert lrs == ""

    def test_count_distinct_single(self):
        lcp = LCPArray("a")
        assert lcp.count_distinct_substrings() == 1


# =============================================================================
# SuffixArraySearcher tests
# =============================================================================

class TestSuffixArraySearcher:
    """Tests for pattern matching on suffix arrays."""

    def test_search_found(self):
        searcher = SuffixArraySearcher("banana")
        results = searcher.search("ana")
        assert sorted(results) == [1, 3]

    def test_search_not_found(self):
        searcher = SuffixArraySearcher("banana")
        results = searcher.search("xyz")
        assert results == []

    def test_search_single_char(self):
        searcher = SuffixArraySearcher("banana")
        results = searcher.search("a")
        assert sorted(results) == [1, 3, 5]

    def test_search_full_string(self):
        searcher = SuffixArraySearcher("banana")
        results = searcher.search("banana")
        assert results == [0]

    def test_count(self):
        searcher = SuffixArraySearcher("banana")
        assert searcher.count("a") == 3
        assert searcher.count("na") == 2
        assert searcher.count("xyz") == 0

    def test_contains(self):
        searcher = SuffixArraySearcher("banana")
        assert searcher.contains("ban") is True
        assert searcher.contains("nan") is True
        assert searcher.contains("xyz") is False

    def test_search_prefix(self):
        searcher = SuffixArraySearcher("banana")
        results = searcher.search("ban")
        assert results == [0]

    def test_search_suffix(self):
        searcher = SuffixArraySearcher("banana")
        results = searcher.search("ana")
        assert 1 in results and 3 in results

    def test_empty_pattern(self):
        searcher = SuffixArraySearcher("abc")
        results = searcher.search("")
        # Empty pattern matches everywhere
        assert len(results) >= 3

    def test_pattern_longer_than_text(self):
        searcher = SuffixArraySearcher("ab")
        results = searcher.search("abcdef")
        assert results == []

    def test_from_existing_sa(self):
        sa = SuffixArray("hello")
        searcher = SuffixArraySearcher("hello", sa)
        assert searcher.contains("ell")

    def test_longest_common_prefix(self):
        searcher = SuffixArraySearcher("abcabdabc")
        result = searcher.longest_common_prefix_of("abc", "abd")
        assert result == "ab"

    def test_mississippi_search(self):
        searcher = SuffixArraySearcher("mississippi")
        assert sorted(searcher.search("issi")) == [1, 4]
        assert searcher.count("ss") == 2
        assert searcher.contains("mississippi")
        assert not searcher.contains("mississippix")

    def test_repeated_pattern(self):
        searcher = SuffixArraySearcher("aaaa")
        assert len(searcher.search("aa")) == 3  # positions 0, 1, 2

    def test_single_char_text(self):
        searcher = SuffixArraySearcher("a")
        assert searcher.search("a") == [0]
        assert searcher.search("b") == []


# =============================================================================
# EnhancedSuffixArray tests
# =============================================================================

class TestEnhancedSuffixArray:
    """Tests for EnhancedSuffixArray with combined queries."""

    def test_search(self):
        esa = EnhancedSuffixArray("banana")
        assert sorted(esa.search("ana")) == [1, 3]

    def test_count(self):
        esa = EnhancedSuffixArray("banana")
        assert esa.count("a") == 3

    def test_contains(self):
        esa = EnhancedSuffixArray("banana")
        assert esa.contains("nan")
        assert not esa.contains("xyz")

    def test_longest_repeated(self):
        esa = EnhancedSuffixArray("banana")
        assert esa.longest_repeated_substring() == "ana"

    def test_count_distinct(self):
        esa = EnhancedSuffixArray("abc")
        assert esa.count_distinct_substrings() == 6

    def test_longest_common_extension_same(self):
        esa = EnhancedSuffixArray("abcabc")
        # LCE(0, 3) should be 3 (abc == abc)
        assert esa.longest_common_extension(0, 3) == 3

    def test_longest_common_extension_different(self):
        esa = EnhancedSuffixArray("abcdef")
        assert esa.longest_common_extension(0, 3) == 0

    def test_longest_common_extension_partial(self):
        esa = EnhancedSuffixArray("abcabd")
        assert esa.longest_common_extension(0, 3) == 2  # "ab"

    def test_longest_common_extension_self(self):
        esa = EnhancedSuffixArray("abc")
        assert esa.longest_common_extension(1, 1) == 2  # "bc"

    def test_top_k_repeated(self):
        esa = EnhancedSuffixArray("banana")
        top = esa.top_k_repeated(2)
        assert len(top) >= 1
        assert top[0][0] == "ana"  # longest repeated

    def test_top_k_repeated_longer(self):
        esa = EnhancedSuffixArray("abcabcabc")
        top = esa.top_k_repeated(3)
        assert len(top) >= 1
        # "abcabc" appears twice
        assert any(t[0] == "abcabc" for t in top)

    def test_all_repeated_substrings(self):
        esa = EnhancedSuffixArray("abab")
        repeated = esa.all_repeated_substrings()
        assert "ab" in repeated
        assert "a" in repeated
        assert "b" in repeated

    def test_all_repeated_min_length(self):
        esa = EnhancedSuffixArray("abab")
        repeated = esa.all_repeated_substrings(min_length=2)
        assert "ab" in repeated
        assert "a" not in repeated

    def test_kth_substring(self):
        esa = EnhancedSuffixArray("abc")
        # Sorted distinct substrings: a, ab, abc, b, bc, c
        assert esa.kth_substring(1) == "a"
        assert esa.kth_substring(2) == "ab"
        assert esa.kth_substring(3) == "abc"
        assert esa.kth_substring(4) == "b"
        assert esa.kth_substring(5) == "bc"
        assert esa.kth_substring(6) == "c"

    def test_kth_substring_out_of_range(self):
        esa = EnhancedSuffixArray("ab")
        assert esa.kth_substring(100) is None

    def test_kth_substring_first(self):
        esa = EnhancedSuffixArray("ba")
        assert esa.kth_substring(1) == "a"

    def test_rank_array(self):
        esa = EnhancedSuffixArray("abc")
        # rank[sa[i]] == i for all i
        for i in range(esa.n):
            assert esa.rank[esa.sa[i]] == i


# =============================================================================
# MultiStringSuffixArray tests
# =============================================================================

class TestMultiStringSuffixArray:
    """Tests for generalized suffix array."""

    def test_construction(self):
        msa = MultiStringSuffixArray(["abc", "def"])
        assert msa.num_texts == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            MultiStringSuffixArray([])

    def test_search_single_text(self):
        msa = MultiStringSuffixArray(["banana"])
        results = msa.search("ana")
        assert len(results) == 2
        assert all(ti == 0 for ti, _ in results)

    def test_search_multiple_texts(self):
        msa = MultiStringSuffixArray(["banana", "panama"])
        results = msa.search("ana")
        text_indices = set(ti for ti, _ in results)
        assert 0 in text_indices  # banana has "ana"
        assert 1 in text_indices  # panama has "ana"

    def test_search_not_found(self):
        msa = MultiStringSuffixArray(["abc", "def"])
        results = msa.search("xyz")
        assert results == []

    def test_search_in_specific_text(self):
        msa = MultiStringSuffixArray(["abc", "abcdef"])
        results = msa.search_in_text("abc", 0)
        assert len(results) >= 1
        assert all(ti == 0 for ti, _ in results)

    def test_longest_common_substring(self):
        msa = MultiStringSuffixArray(["abcde", "bcdef"])
        lcs = msa.longest_common_substring()
        assert lcs == "bcde"

    def test_longest_common_substring_no_common(self):
        msa = MultiStringSuffixArray(["abc", "xyz"])
        lcs = msa.longest_common_substring()
        assert lcs == ""

    def test_longest_common_substring_full_overlap(self):
        msa = MultiStringSuffixArray(["abc", "abc"])
        lcs = msa.longest_common_substring()
        assert lcs == "abc"

    def test_common_substrings(self):
        msa = MultiStringSuffixArray(["abcde", "bcdef"])
        common = msa.common_substrings(min_length=2)
        assert "bcd" in common or "bcde" in common

    def test_three_texts(self):
        msa = MultiStringSuffixArray(["abcx", "abcy", "abcz"])
        results = msa.search("abc")
        text_indices = set(ti for ti, _ in results)
        assert text_indices == {0, 1, 2}

    def test_single_text(self):
        msa = MultiStringSuffixArray(["hello"])
        results = msa.search("ell")
        assert len(results) == 1
        assert results[0] == (0, 1)

    def test_common_substrings_min_length(self):
        msa = MultiStringSuffixArray(["ab", "ab"])
        common = msa.common_substrings(min_length=1)
        assert "a" in common
        assert "b" in common
        assert "ab" in common

    def test_search_positions_correct(self):
        msa = MultiStringSuffixArray(["hello", "world"])
        results = msa.search("lo")
        assert (0, 3) in results  # "lo" starts at position 3 in "hello"

    def test_search_positions_second_text(self):
        msa = MultiStringSuffixArray(["abc", "xabcy"])
        results = msa.search("abc")
        positions = dict(results)
        assert 0 in positions  # first text
        assert 1 in positions  # second text
        assert positions[1] == 1  # "abc" at position 1 in "xabcy"


# =============================================================================
# Edge cases and stress tests
# =============================================================================

class TestEdgeCases:
    """Edge cases and special scenarios."""

    def test_sa_correctness_abracadabra(self):
        sa = SuffixArray("abracadabra")
        suffixes = sa.suffixes_sorted()
        for i in range(1, len(suffixes)):
            assert suffixes[i - 1] < suffixes[i]

    def test_sa_all_same(self):
        sa = SuffixArray("aaaa")
        suffixes = sa.suffixes_sorted()
        assert suffixes == ["a", "aa", "aaa", "aaaa"]

    def test_lcp_mississippi(self):
        lcp = LCPArray("mississippi")
        lrs = lcp.longest_repeated_substring()
        assert lrs == "issi"

    def test_search_overlapping(self):
        searcher = SuffixArraySearcher("aaa")
        results = searcher.search("aa")
        assert sorted(results) == [0, 1]

    def test_enhanced_long_text(self):
        text = "the quick brown fox jumps over the lazy dog"
        esa = EnhancedSuffixArray(text)
        assert esa.contains("quick")
        assert esa.contains("fox")
        assert not esa.contains("cat")

    def test_enhanced_search_the(self):
        text = "the quick brown fox jumps over the lazy dog"
        esa = EnhancedSuffixArray(text)
        results = esa.search("the")
        assert len(results) == 2  # "the" appears twice

    def test_multi_lcs_with_overlap(self):
        msa = MultiStringSuffixArray(["xabcx", "yabcy"])
        lcs = msa.longest_common_substring()
        assert lcs == "abc"

    def test_sa_binary_alphabet(self):
        sa = SuffixArray([0, 1, 0, 1, 0])
        suffixes = sa.suffixes_sorted()
        for i in range(1, len(suffixes)):
            assert suffixes[i - 1] < suffixes[i]

    def test_enhanced_abracadabra(self):
        esa = EnhancedSuffixArray("abracadabra")
        assert esa.longest_repeated_substring() == "abra"
        assert esa.count("abra") == 2
        assert esa.contains("cad")

    def test_kth_substring_banana(self):
        esa = EnhancedSuffixArray("banana")
        # Should return valid substrings in order
        first = esa.kth_substring(1)
        second = esa.kth_substring(2)
        assert first < second

    def test_count_distinct_mississippi(self):
        lcp = LCPArray("mississippi")
        count = lcp.count_distinct_substrings()
        # mississippi has 53 distinct substrings
        assert count == 53

    def test_multi_search_all_same(self):
        msa = MultiStringSuffixArray(["aaa", "aaa"])
        results = msa.search("aa")
        assert len(results) == 4  # 2 per text

    def test_single_char_texts(self):
        msa = MultiStringSuffixArray(["a", "b"])
        lcs = msa.longest_common_substring()
        assert lcs == ""

    def test_enhanced_single_char(self):
        esa = EnhancedSuffixArray("a")
        assert esa.kth_substring(1) == "a"
        assert esa.kth_substring(2) is None

    def test_searcher_integer_list(self):
        searcher = SuffixArraySearcher([3, 1, 4, 1, 5])
        results = searcher.search([1])
        assert sorted(results) == [1, 3]

    def test_lcp_integer_list(self):
        lcp = LCPArray([3, 1, 2, 1, 2])
        lrs = lcp.longest_repeated_substring()
        assert lrs == (1, 2)
