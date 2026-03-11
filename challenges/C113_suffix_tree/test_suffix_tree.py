"""Tests for C113: Suffix Tree -- Ukkonen's Algorithm."""
import pytest
from suffix_tree import (
    SuffixTree, SuffixTreeNode, EndRef,
    GeneralizedSuffixTree, SuffixTreeWithLCP,
    SuffixTreeSearcher, SuffixTreeAnalyzer,
)


# ============================================================
# SuffixTreeNode tests
# ============================================================

class TestSuffixTreeNode:
    def test_node_creation(self):
        node = SuffixTreeNode(0, 5)
        assert node.start == 0
        assert node.end == 5
        assert node.suffix_index == -1
        assert node.children == {}
        assert node.suffix_link is None

    def test_node_is_leaf(self):
        node = SuffixTreeNode(0, 5)
        assert node.is_leaf()
        node.children['a'] = SuffixTreeNode(1, 2)
        assert not node.is_leaf()

    def test_node_is_root(self):
        root = SuffixTreeNode(-1)
        assert root.is_root()
        non_root = SuffixTreeNode(0, 5)
        assert not non_root.is_root()

    def test_edge_length_fixed(self):
        node = SuffixTreeNode(2, 7)
        assert node.edge_length == 6  # 7 - 2 + 1

    def test_edge_length_endref(self):
        end = EndRef(10)
        node = SuffixTreeNode(3, end)
        assert node.edge_length == 8  # 10 - 3 + 1

    def test_edge_length_none(self):
        node = SuffixTreeNode(0, None)
        assert node.edge_length == 0

    def test_string_ids_empty(self):
        node = SuffixTreeNode()
        assert node.string_ids == set()


class TestEndRef:
    def test_endref_creation(self):
        e = EndRef(5)
        assert e.value == 5

    def test_endref_mutable(self):
        e = EndRef(0)
        e.value = 10
        assert e.value == 10

    def test_endref_shared(self):
        e = EndRef(5)
        n1 = SuffixTreeNode(0, e)
        n2 = SuffixTreeNode(3, e)
        e.value = 20
        assert n1.edge_length == 21
        assert n2.edge_length == 18


# ============================================================
# SuffixTree core tests
# ============================================================

class TestSuffixTree:
    def test_build_simple(self):
        st = SuffixTree("abc")
        assert st.contains("a")
        assert st.contains("b")
        assert st.contains("c")
        assert st.contains("ab")
        assert st.contains("bc")
        assert st.contains("abc")
        assert not st.contains("d")
        assert not st.contains("ac")

    def test_build_single_char(self):
        st = SuffixTree("a")
        assert st.contains("a")
        assert not st.contains("b")

    def test_build_repeated_char(self):
        st = SuffixTree("aaa")
        assert st.contains("a")
        assert st.contains("aa")
        assert st.contains("aaa")
        assert not st.contains("aaaa")

    def test_build_banana(self):
        st = SuffixTree("banana")
        assert st.contains("banana")
        assert st.contains("anana")
        assert st.contains("nana")
        assert st.contains("ana")
        assert st.contains("na")
        assert st.contains("a")
        assert st.contains("ban")
        assert st.contains("an")
        assert not st.contains("banan ")

    def test_contains_empty_pattern(self):
        st = SuffixTree("abc")
        assert st.contains("")

    def test_count_occurrences(self):
        st = SuffixTree("banana")
        assert st.count_occurrences("a") == 3
        assert st.count_occurrences("an") == 2
        assert st.count_occurrences("ana") == 2
        assert st.count_occurrences("nan") == 1
        assert st.count_occurrences("banana") == 1
        assert st.count_occurrences("x") == 0

    def test_find_all(self):
        st = SuffixTree("banana")
        assert st.find_all("a") == [1, 3, 5]
        assert st.find_all("an") == [1, 3]
        assert st.find_all("ana") == [1, 3]
        assert st.find_all("banana") == [0]
        assert st.find_all("x") == []

    def test_find_all_repeated(self):
        st = SuffixTree("abcabc")
        assert st.find_all("abc") == [0, 3]
        assert st.find_all("bc") == [1, 4]

    def test_has_suffix(self):
        st = SuffixTree("banana")
        assert st.has_suffix("banana")
        assert st.has_suffix("anana")
        assert st.has_suffix("nana")
        assert st.has_suffix("ana")
        assert st.has_suffix("na")
        assert st.has_suffix("a")
        assert not st.has_suffix("ban")
        assert not st.has_suffix("an")

    def test_leaf_count(self):
        st = SuffixTree("abc")
        # "abc\0" has 4 suffixes -> 4 leaves
        assert st.leaf_count() == 4

    def test_leaf_count_banana(self):
        st = SuffixTree("banana")
        assert st.leaf_count() == 7  # 7 suffixes including terminal

    def test_edge_count(self):
        st = SuffixTree("abc")
        assert st.edge_count() > 0

    def test_node_count(self):
        st = SuffixTree("abc")
        assert st.node_count() >= st.leaf_count() + 1  # leaves + root at minimum

    def test_get_edge_label(self):
        st = SuffixTree("abc")
        for child in st.root.children.values():
            label = st.get_edge_label(child)
            assert len(label) > 0

    def test_get_edge_label_root(self):
        st = SuffixTree("abc")
        assert st.get_edge_label(st.root) == ""

    def test_get_all_suffixes(self):
        st = SuffixTree("abc")
        suffixes = st.get_all_suffixes()
        assert "abc" in suffixes
        assert "bc" in suffixes
        assert "c" in suffixes
        assert len(suffixes) == 3

    def test_get_all_suffixes_banana(self):
        st = SuffixTree("banana")
        suffixes = st.get_all_suffixes()
        expected = ["a", "ana", "anana", "banana", "na", "nana"]
        assert suffixes == expected

    def test_longest_repeated_substring(self):
        st = SuffixTree("banana")
        lrs = st.longest_repeated_substring()
        assert lrs == "ana"

    def test_longest_repeated_no_repeat(self):
        st = SuffixTree("abcdef")
        lrs = st.longest_repeated_substring()
        # No substring repeats (of length > 0), empty or single char
        assert len(lrs) <= 0 or st.count_occurrences(lrs) >= 2

    def test_longest_repeated_all_same(self):
        st = SuffixTree("aaaa")
        lrs = st.longest_repeated_substring()
        assert lrs == "aaa"

    def test_shortest_unique_substring(self):
        st = SuffixTree("banana")
        sus = st.shortest_unique_substring()
        assert st.count_occurrences(sus) == 1
        # Should be short
        assert len(sus) <= 2

    def test_build_with_terminal(self):
        # If text already has terminal, should still work
        st = SuffixTree("abc\x00")
        assert st.contains("abc")

    def test_constructor_no_args(self):
        st = SuffixTree()
        assert st.text == ""

    def test_mississippi(self):
        st = SuffixTree("mississippi")
        assert st.contains("missis")
        assert st.contains("issi")
        assert st.contains("ippi")
        assert st.count_occurrences("issi") == 2
        assert st.count_occurrences("ss") == 2
        assert st.count_occurrences("i") == 4
        assert st.find_all("ss") == [2, 5]


# ============================================================
# GeneralizedSuffixTree tests
# ============================================================

class TestGeneralizedSuffixTree:
    def test_build_two_strings(self):
        gst = GeneralizedSuffixTree(["abc", "bcd"])
        assert gst.contains("abc")
        assert gst.contains("bcd")
        assert gst.contains("bc")

    def test_find_strings_containing(self):
        gst = GeneralizedSuffixTree(["abc", "bcd", "cde"])
        assert gst.find_strings_containing("bc") == {0, 1}
        assert gst.find_strings_containing("cd") == {1, 2}
        assert gst.find_strings_containing("abc") == {0}
        assert gst.find_strings_containing("xyz") == set()

    def test_find_strings_single_char(self):
        gst = GeneralizedSuffixTree(["abc", "bcd", "cde"])
        assert gst.find_strings_containing("c") == {0, 1, 2}
        assert gst.find_strings_containing("a") == {0}
        assert gst.find_strings_containing("e") == {2}

    def test_longest_common_substring(self):
        gst = GeneralizedSuffixTree(["abcdef", "xbcdey"])
        lcs = gst.longest_common_substring()
        assert lcs == "bcde"

    def test_lcs_identical(self):
        gst = GeneralizedSuffixTree(["abc", "abc"])
        assert gst.longest_common_substring() == "abc"

    def test_lcs_no_common(self):
        gst = GeneralizedSuffixTree(["abc", "xyz"])
        assert gst.longest_common_substring() == ""

    def test_lcs_three_strings(self):
        gst = GeneralizedSuffixTree(["xabcy", "yabcz", "zabcw"])
        lcs = gst.longest_common_substring()
        assert lcs == "abc"

    def test_lcs_specific_indices(self):
        gst = GeneralizedSuffixTree(["abc", "bcd", "cde"])
        lcs01 = gst.longest_common_substring(indices=[0, 1])
        assert lcs01 == "bc"
        lcs12 = gst.longest_common_substring(indices=[1, 2])
        assert lcs12 == "cd"

    def test_common_substrings(self):
        gst = GeneralizedSuffixTree(["abc", "abx"])
        common = gst.common_substrings(min_length=1)
        assert "ab" in common
        assert "b" in common

    def test_common_substrings_min_length(self):
        gst = GeneralizedSuffixTree(["abc", "abx"])
        common = gst.common_substrings(min_length=2)
        assert "ab" in common
        assert "a" not in common

    def test_contains_pattern(self):
        gst = GeneralizedSuffixTree(["hello", "world"])
        assert gst.contains("hell")
        assert gst.contains("orld")
        assert not gst.contains("xyz")

    def test_empty_constructor(self):
        gst = GeneralizedSuffixTree()
        assert gst.strings == []

    def test_single_string(self):
        gst = GeneralizedSuffixTree(["hello"])
        assert gst.contains("hello")
        assert gst.contains("ell")
        assert gst.find_strings_containing("ell") == {0}

    def test_lcs_single_string(self):
        gst = GeneralizedSuffixTree(["hello"])
        # With one string, LCS with itself should be empty (need 2)
        assert gst.longest_common_substring() == ""


# ============================================================
# SuffixTreeWithLCP tests
# ============================================================

class TestSuffixTreeWithLCP:
    def test_lcp_same_position(self):
        st = SuffixTreeWithLCP("banana")
        assert st.lcp(0, 0) == 6

    def test_lcp_different_positions(self):
        st = SuffixTreeWithLCP("banana")
        # suffix at 1: "anana", suffix at 3: "ana"
        assert st.lcp(1, 3) == 3  # "ana"

    def test_lcp_no_common(self):
        st = SuffixTreeWithLCP("banana")
        assert st.lcp(0, 1) == 0  # "banana" vs "anana"

    def test_lcp_out_of_range(self):
        st = SuffixTreeWithLCP("abc")
        assert st.lcp(-1, 0) == 0
        assert st.lcp(0, 100) == 0

    def test_lcp_array(self):
        st = SuffixTreeWithLCP("banana")
        sa, lcp = st.lcp_array()
        # SA for "banana": sorted suffixes
        assert len(sa) == 6
        assert len(lcp) == 6
        assert lcp[0] == 0  # first element always 0

    def test_lcp_array_sorted(self):
        st = SuffixTreeWithLCP("abc")
        sa, lcp = st.lcp_array()
        # Verify SA is in sorted order
        text = "abc"
        suffixes = [text[i:] for i in sa]
        assert suffixes == sorted(suffixes)

    def test_contains_delegates(self):
        st = SuffixTreeWithLCP("banana")
        assert st.contains("ana")
        assert not st.contains("xyz")

    def test_build_and_root(self):
        st = SuffixTreeWithLCP("abc")
        assert st.root is not None
        assert st.root.is_root()

    def test_empty_constructor(self):
        st = SuffixTreeWithLCP()
        assert st.text == ""


# ============================================================
# SuffixTreeSearcher tests
# ============================================================

class TestSuffixTreeSearcher:
    def test_search(self):
        s = SuffixTreeSearcher("abracadabra")
        assert s.search("abra") == [0, 7]
        assert s.search("bra") == [1, 8]
        assert s.search("cad") == [4]
        assert s.search("xyz") == []

    def test_count(self):
        s = SuffixTreeSearcher("abracadabra")
        assert s.count("a") == 5
        assert s.count("abra") == 2
        assert s.count("abracadabra") == 1
        assert s.count("z") == 0

    def test_contains(self):
        s = SuffixTreeSearcher("hello world")
        assert s.contains("hello")
        assert s.contains("world")
        assert s.contains("lo wo")
        assert not s.contains("xyz")

    def test_is_suffix(self):
        s = SuffixTreeSearcher("banana")
        assert s.is_suffix("banana")
        assert s.is_suffix("ana")
        assert s.is_suffix("a")
        assert not s.is_suffix("ban")

    def test_is_prefix(self):
        s = SuffixTreeSearcher("banana")
        assert s.is_prefix("ban")
        assert s.is_prefix("banana")
        assert s.is_prefix("")
        assert not s.is_prefix("ana")

    def test_longest_common_extension(self):
        s = SuffixTreeSearcher("abcabc")
        assert s.longest_common_extension(0, 3) == 3  # "abc" == "abc"
        assert s.longest_common_extension(0, 0) == 6
        assert s.longest_common_extension(1, 4) == 2  # "bc" == "bc"

    def test_distinct_substrings(self):
        s = SuffixTreeSearcher("abc")
        # "a", "ab", "abc", "b", "bc", "c" = 6
        assert s.distinct_substrings() == 6

    def test_distinct_substrings_repeated(self):
        s = SuffixTreeSearcher("aab")
        # "a", "aa", "aab", "ab", "b" = 5
        assert s.distinct_substrings() == 5

    def test_kth_substring(self):
        s = SuffixTreeSearcher("abc")
        # Sorted: a, ab, abc, b, bc, c
        assert s.kth_substring(1) == "a"
        assert s.kth_substring(2) == "ab"
        assert s.kth_substring(3) == "abc"
        assert s.kth_substring(4) == "b"
        assert s.kth_substring(5) == "bc"
        assert s.kth_substring(6) == "c"

    def test_kth_substring_out_of_range(self):
        s = SuffixTreeSearcher("ab")
        assert s.kth_substring(100) is None

    def test_longest_repeated_substring(self):
        s = SuffixTreeSearcher("banana")
        assert s.longest_repeated_substring() == "ana"

    def test_empty_constructor(self):
        s = SuffixTreeSearcher()
        assert s.text == ""

    def test_single_char(self):
        s = SuffixTreeSearcher("x")
        assert s.search("x") == [0]
        assert s.count("x") == 1
        assert s.distinct_substrings() == 1


# ============================================================
# SuffixTreeAnalyzer tests
# ============================================================

class TestSuffixTreeAnalyzer:
    def test_longest_repeated(self):
        a = SuffixTreeAnalyzer("banana")
        assert a.longest_repeated_substring() == "ana"

    def test_longest_repeated_mississippi(self):
        a = SuffixTreeAnalyzer("mississippi")
        lrs = a.longest_repeated_substring()
        assert lrs == "issi"

    def test_shortest_unique(self):
        a = SuffixTreeAnalyzer("banana")
        sus = a.shortest_unique_substring()
        assert sus is not None
        # Verify it occurs exactly once
        assert a.text.count(sus) == 1

    def test_tandem_repeats_simple(self):
        a = SuffixTreeAnalyzer("abab")
        repeats = a.tandem_repeats()
        # "ab" repeats twice starting at 0
        found = any(pos == 0 and unit == "ab" and count == 2
                     for pos, unit, count in repeats)
        assert found

    def test_tandem_repeats_aaa(self):
        a = SuffixTreeAnalyzer("aaa")
        repeats = a.tandem_repeats()
        # "a" repeats 3 times starting at 0
        found = any(unit == "a" and count == 3 for _, unit, count in repeats)
        assert found

    def test_tandem_repeats_no_repeat(self):
        a = SuffixTreeAnalyzer("abcdef")
        repeats = a.tandem_repeats()
        # No tandem repeats of length >= 1
        assert len(repeats) == 0

    def test_maximal_repeats(self):
        a = SuffixTreeAnalyzer("abcabc")
        mr = a.maximal_repeats()
        # "abc" should be a maximal repeat
        substrings = [s for s, _ in mr]
        assert "abc" in substrings

    def test_maximal_repeats_banana(self):
        a = SuffixTreeAnalyzer("banana")
        mr = a.maximal_repeats()
        assert len(mr) > 0
        substrings = [s for s, _ in mr]
        # "a" occurs at 1,3,5 with different left contexts
        assert "a" in substrings

    def test_supermaximal_repeats(self):
        a = SuffixTreeAnalyzer("abcabc")
        smr = a.supermaximal_repeats()
        # "abc" should be supermaximal
        substrings = [s for s, _ in smr]
        assert "abc" in substrings

    def test_palindromes(self):
        a = SuffixTreeAnalyzer("abacaba")
        pals = a.palindromes(min_length=2)
        assert "aba" in pals
        assert "abacaba" in pals
        assert "aca" in pals

    def test_palindromes_none(self):
        a = SuffixTreeAnalyzer("abcdef")
        pals = a.palindromes(min_length=2)
        assert len(pals) == 0

    def test_palindromes_all_same(self):
        a = SuffixTreeAnalyzer("aaa")
        pals = a.palindromes(min_length=2)
        assert "aa" in pals
        assert "aaa" in pals

    def test_substring_frequency(self):
        a = SuffixTreeAnalyzer("abracadabra")
        assert a.substring_frequency("a") == 5
        assert a.substring_frequency("abra") == 2
        assert a.substring_frequency("xyz") == 0

    def test_most_frequent_substring(self):
        a = SuffixTreeAnalyzer("banana")
        sub, freq = a.most_frequent_substring(1)
        assert sub == "a"
        assert freq == 3

    def test_most_frequent_length2(self):
        a = SuffixTreeAnalyzer("banana")
        sub, freq = a.most_frequent_substring(2)
        assert sub == "an" or sub == "na"
        assert freq == 2

    def test_most_frequent_none(self):
        a = SuffixTreeAnalyzer("ab")
        result = a.most_frequent_substring(5)
        assert result == (None, 0)

    def test_empty_constructor(self):
        a = SuffixTreeAnalyzer()
        assert a.text == ""

    def test_single_char_analysis(self):
        a = SuffixTreeAnalyzer("x")
        assert a.longest_repeated_substring() == ""
        assert a.tandem_repeats() == []


# ============================================================
# Edge cases and stress tests
# ============================================================

class TestEdgeCases:
    def test_all_same_characters(self):
        st = SuffixTree("aaaaaaa")
        assert st.contains("aaa")
        assert st.count_occurrences("a") == 7
        assert st.count_occurrences("aa") == 6
        assert st.find_all("aaa") == [0, 1, 2, 3, 4]

    def test_alternating(self):
        st = SuffixTree("abababab")
        assert st.count_occurrences("ab") == 4
        assert st.count_occurrences("aba") == 3
        assert st.find_all("bab") == [1, 3, 5]

    def test_long_text(self):
        text = "abcdefghij" * 50
        st = SuffixTree(text)
        assert st.contains("abcdefghij")
        assert st.count_occurrences("abc") == 50

    def test_binary_string(self):
        st = SuffixTree("01001010")
        assert st.contains("010")
        assert st.count_occurrences("01") == 3
        assert st.find_all("10") == [1, 4, 6]

    def test_two_char_text(self):
        st = SuffixTree("ab")
        assert st.contains("a")
        assert st.contains("b")
        assert st.contains("ab")
        assert not st.contains("ba")
        assert st.leaf_count() == 3  # "ab\0", "b\0", "\0"

    def test_generalized_overlapping(self):
        gst = GeneralizedSuffixTree(["abc", "cab", "bca"])
        assert gst.find_strings_containing("a") == {0, 1, 2}
        assert gst.find_strings_containing("ab") == {0, 1}
        assert gst.find_strings_containing("bc") == {0, 2}

    def test_generalized_lcs_overlap(self):
        gst = GeneralizedSuffixTree(["abcxyz", "xyzabc"])
        lcs = gst.longest_common_substring()
        assert lcs in ("abc", "xyz")
        assert len(lcs) == 3

    def test_searcher_multiword(self):
        s = SuffixTreeSearcher("the cat sat on the mat")
        assert s.search("the") == [0, 15]
        assert s.count("at") == 3
        assert s.contains("cat sat")

    def test_analyzer_dna(self):
        a = SuffixTreeAnalyzer("ATCGATCGATCG")
        assert a.longest_repeated_substring() == "ATCGATCG"
        assert a.substring_frequency("ATCG") == 3

    def test_lcp_mississippi(self):
        st = SuffixTreeWithLCP("mississippi")
        sa, lcp = st.lcp_array()
        assert len(sa) == 11
        assert len(lcp) == 11
        # Verify SA order
        text = "mississippi"
        for i in range(len(sa) - 1):
            assert text[sa[i]:] < text[sa[i + 1]:]


# ============================================================
# Integration / composition tests
# ============================================================

class TestIntegration:
    def test_tree_structure_banana(self):
        """Verify structural properties of banana suffix tree."""
        st = SuffixTree("banana")
        # n+1 leaves (including terminal)
        assert st.leaf_count() == 7
        # Every internal node (except root) has >= 2 children
        def check_internal(node):
            if not node.is_leaf() and not node.is_root():
                assert len(node.children) >= 2
            for child in node.children.values():
                check_internal(child)
        check_internal(st.root)

    def test_suffix_links_exist(self):
        """Verify that internal nodes have suffix links."""
        st = SuffixTree("banana")
        def check_links(node):
            if not node.is_leaf() and not node.is_root():
                assert node.suffix_link is not None
            for child in node.children.values():
                check_links(child)
        check_links(st.root)

    def test_all_suffixes_findable(self):
        """Every suffix of the text should be findable."""
        text = "abracadabra"
        st = SuffixTree(text)
        for i in range(len(text)):
            suffix = text[i:]
            assert st.contains(suffix), f"suffix '{suffix}' not found"
            assert st.has_suffix(suffix), f"'{suffix}' not recognized as suffix"

    def test_find_all_consistency(self):
        """find_all results should match count_occurrences."""
        text = "mississippi"
        st = SuffixTree(text)
        for pattern in ["i", "is", "ss", "issi", "p", "pp"]:
            positions = st.find_all(pattern)
            count = st.count_occurrences(pattern)
            assert len(positions) == count
            # Verify positions are correct
            for pos in positions:
                assert text[pos:pos + len(pattern)] == pattern

    def test_generalized_vs_individual(self):
        """Generalized tree should find same patterns as individual trees."""
        strings = ["hello", "world", "help"]
        gst = GeneralizedSuffixTree(strings)
        for i, s in enumerate(strings):
            st = SuffixTree(s)
            for j in range(len(s)):
                for k in range(j + 1, len(s) + 1):
                    pattern = s[j:k]
                    assert gst.contains(pattern)
                    ids = gst.find_strings_containing(pattern)
                    assert i in ids

    def test_distinct_substrings_formula(self):
        """For no-repeat text, distinct substrings = n*(n+1)/2."""
        text = "abcdef"
        s = SuffixTreeSearcher(text)
        n = len(text)
        assert s.distinct_substrings() == n * (n + 1) // 2

    def test_kth_substring_all(self):
        """Enumerate all distinct substrings via kth."""
        text = "abc"
        s = SuffixTreeSearcher(text)
        total = s.distinct_substrings()
        substrings = []
        for k in range(1, total + 1):
            sub = s.kth_substring(k)
            assert sub is not None
            substrings.append(sub)
        # Should be sorted
        assert substrings == sorted(substrings)
        # All distinct
        assert len(set(substrings)) == total

    def test_analyzer_all_methods(self):
        """Run all analyzer methods on same text."""
        a = SuffixTreeAnalyzer("abcabcabc")
        lrs = a.longest_repeated_substring()
        assert len(lrs) >= 3
        sus = a.shortest_unique_substring()
        assert sus is not None
        tr = a.tandem_repeats()
        assert len(tr) > 0
        mr = a.maximal_repeats()
        assert len(mr) > 0
        pals = a.palindromes(min_length=1)
        # Single chars are palindromes
        assert len(pals) >= 1
        sub, freq = a.most_frequent_substring(3)
        assert sub == "abc" or sub == "bca" or sub == "cab"

    def test_lcp_correctness(self):
        """LCP values should match actual common prefixes."""
        text = "banana"
        st = SuffixTreeWithLCP(text)
        sa, lcp_arr = st.lcp_array()
        for i in range(1, len(sa)):
            s1 = text[sa[i - 1]:]
            s2 = text[sa[i]:]
            # Compute actual LCP
            actual = 0
            for c1, c2 in zip(s1, s2):
                if c1 == c2:
                    actual += 1
                else:
                    break
            assert lcp_arr[i] == actual, f"LCP mismatch at {i}: {lcp_arr[i]} vs {actual}"
