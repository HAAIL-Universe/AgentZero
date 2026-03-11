"""Tests for C114: Aho-Corasick Multi-Pattern String Matching."""

import pytest
from aho_corasick import (
    AhoCorasick, TrieNode,
    StreamingAhoCorasick,
    WeightedAhoCorasick,
    WildcardAhoCorasick,
    AhoCorasickReplacer,
    AhoCorasickCounter,
)


# ===========================================================================
# Variant 1: Core AhoCorasick
# ===========================================================================

class TestAhoCorasickBasic:
    def test_single_pattern(self):
        ac = AhoCorasick(["hello"])
        results = ac.search("say hello world")
        assert len(results) == 1
        assert results[0] == (4, 0, "hello")

    def test_no_match(self):
        ac = AhoCorasick(["xyz"])
        assert ac.search("hello world") == []

    def test_multiple_patterns(self):
        ac = AhoCorasick(["he", "she", "his", "hers"])
        results = ac.search("ushers")
        patterns_found = {r[2] for r in results}
        assert "she" in patterns_found
        assert "he" in patterns_found
        assert "hers" in patterns_found

    def test_overlapping_matches(self):
        ac = AhoCorasick(["ab", "abc", "bc"])
        results = ac.search("abc")
        patterns_found = {r[2] for r in results}
        assert "ab" in patterns_found
        assert "abc" in patterns_found
        assert "bc" in patterns_found

    def test_repeated_pattern(self):
        ac = AhoCorasick(["aa"])
        results = ac.search("aaa")
        assert len(results) == 2
        positions = [r[0] for r in results]
        assert 0 in positions
        assert 1 in positions

    def test_empty_text(self):
        ac = AhoCorasick(["test"])
        assert ac.search("") == []

    def test_pattern_at_start(self):
        ac = AhoCorasick(["hello"])
        results = ac.search("hello world")
        assert results[0][0] == 0

    def test_pattern_at_end(self):
        ac = AhoCorasick(["world"])
        results = ac.search("hello world")
        assert results[0][0] == 6

    def test_single_char_patterns(self):
        ac = AhoCorasick(["a", "b", "c"])
        results = ac.search("abc")
        assert len(results) == 3

    def test_long_pattern(self):
        pat = "abcdefghijklmnop"
        ac = AhoCorasick([pat])
        results = ac.search("xxx" + pat + "yyy")
        assert len(results) == 1
        assert results[0][0] == 3

    def test_constructor_with_patterns(self):
        ac = AhoCorasick(["foo", "bar"])
        assert len(ac) == 2
        assert ac._built

    def test_repr(self):
        ac = AhoCorasick(["a", "b"])
        assert "patterns=2" in repr(ac)
        assert "built=True" in repr(ac)


class TestAhoCorasickFailureLinks:
    """Tests specifically targeting failure link behavior."""

    def test_classic_ushers(self):
        """The classic Aho-Corasick example."""
        ac = AhoCorasick(["he", "she", "his", "hers"])
        results = ac.search("ushers")
        found = sorted([(r[0], r[2]) for r in results])
        assert (1, "she") in found
        assert (2, "he") in found
        assert (2, "hers") in found

    def test_nested_patterns(self):
        ac = AhoCorasick(["a", "ab", "abc", "abcd"])
        results = ac.search("abcd")
        # All 4 patterns match at position 0
        assert len(results) == 4
        patterns = [r[2] for r in results]
        assert "a" in patterns
        assert "ab" in patterns
        assert "abc" in patterns
        assert "abcd" in patterns

    def test_failure_chain(self):
        """Pattern where failure links chain through multiple levels."""
        ac = AhoCorasick(["abc", "bc", "c"])
        results = ac.search("abc")
        patterns = {r[2] for r in results}
        assert patterns == {"abc", "bc", "c"}

    def test_shared_prefix(self):
        ac = AhoCorasick(["abc", "abd", "abe"])
        results = ac.search("abcabdabe")
        assert len(results) == 3

    def test_shared_suffix(self):
        ac = AhoCorasick(["xbc", "ybc", "zbc"])
        results = ac.search("xbcybczbc")
        assert len(results) == 3


class TestAhoCorasickEdgeCases:
    def test_all_same_char(self):
        ac = AhoCorasick(["aaa"])
        results = ac.search("aaaaa")
        assert len(results) == 3  # positions 0,1,2

    def test_binary_alphabet(self):
        ac = AhoCorasick(["01", "10", "010"])
        results = ac.search("01010")
        assert len(results) >= 4

    def test_large_alphabet(self):
        ac = AhoCorasick(["alpha", "beta", "gamma", "delta"])
        results = ac.search("alphabetagammadelta")
        assert len(results) == 4

    def test_substring_patterns(self):
        ac = AhoCorasick(["the", "there", "her", "here"])
        results = ac.search("there")
        patterns = {r[2] for r in results}
        assert "the" in patterns
        assert "there" in patterns
        assert "her" in patterns
        assert "here" in patterns

    def test_many_patterns(self):
        patterns = [f"pat{i}" for i in range(100)]
        ac = AhoCorasick(patterns)
        text = " ".join(patterns)
        results = ac.search(text)
        # More than 100 because "pat1" is a prefix of "pat10"-"pat19" etc.
        assert len(results) >= 100
        found_pats = {r[2] for r in results}
        for p in patterns:
            assert p in found_pats

    def test_cannot_add_after_build(self):
        ac = AhoCorasick(["a"])
        with pytest.raises(RuntimeError):
            ac.add_pattern("b")

    def test_search_before_build(self):
        ac = AhoCorasick()
        ac.add_pattern("test")
        with pytest.raises(RuntimeError):
            ac.search("test")

    def test_contains_any(self):
        ac = AhoCorasick(["needle"])
        assert ac.contains_any("find the needle here")
        assert not ac.contains_any("nothing here")

    def test_search_first(self):
        ac = AhoCorasick(["world", "hello"])
        result = ac.search_first("hello world")
        assert result is not None
        assert result[0] == 0  # hello at position 0

    def test_search_first_none(self):
        ac = AhoCorasick(["xyz"])
        assert ac.search_first("hello") is None

    def test_double_build(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        ac.build()  # should be idempotent
        assert ac.search("a") == [(0, 0, "a")]


class TestAhoCorasickPerformance:
    def test_dna_patterns(self):
        """Simulate DNA motif searching."""
        patterns = ["ATCG", "GCTA", "TTAA", "CCGG", "ATAT"]
        ac = AhoCorasick(patterns)
        dna = "ATCGATCGGCTATTAACCGGATAT" * 10
        results = ac.search(dna)
        assert len(results) > 0

    def test_many_short_patterns(self):
        import string
        patterns = [a + b for a in string.ascii_lowercase[:10] for b in string.ascii_lowercase[:10]]
        ac = AhoCorasick(patterns)
        text = string.ascii_lowercase * 5
        results = ac.search(text)
        assert len(results) > 0

    def test_long_text_few_patterns(self):
        ac = AhoCorasick(["needle"])
        text = "haystack " * 1000 + "needle" + " haystack" * 1000
        results = ac.search(text)
        assert len(results) == 1


# ===========================================================================
# Variant 2: StreamingAhoCorasick
# ===========================================================================

class TestStreamingAhoCorasick:
    def test_basic_streaming(self):
        sac = StreamingAhoCorasick(["hello", "world"])
        r1 = sac.feed("hel")
        r2 = sac.feed("lo wor")
        r3 = sac.feed("ld")
        all_results = r1 + r2 + r3
        patterns = {r[2] for r in all_results}
        assert "hello" in patterns
        assert "world" in patterns

    def test_streaming_positions_are_global(self):
        sac = StreamingAhoCorasick(["ab"])
        sac.feed("xxxx")  # offset becomes 4
        results = sac.feed("ab")
        assert results[0][0] == 4  # global position

    def test_match_across_chunks(self):
        sac = StreamingAhoCorasick(["abc"])
        r1 = sac.feed("a")
        r2 = sac.feed("b")
        r3 = sac.feed("c")
        all_results = r1 + r2 + r3
        assert len(all_results) == 1
        assert all_results[0][0] == 0

    def test_reset(self):
        sac = StreamingAhoCorasick(["test"])
        sac.feed("xxxx")
        assert sac.total_bytes_processed == 4
        sac.reset()
        assert sac.total_bytes_processed == 0
        results = sac.feed("test")
        assert len(results) == 1
        assert results[0][0] == 0

    def test_empty_chunks(self):
        sac = StreamingAhoCorasick(["abc"])
        assert sac.feed("") == []
        assert sac.feed("abc") == [(0, 0, "abc")]

    def test_patterns_property(self):
        sac = StreamingAhoCorasick(["a", "b"])
        assert sac.patterns == ["a", "b"]

    def test_multiple_matches_in_chunk(self):
        sac = StreamingAhoCorasick(["ab"])
        results = sac.feed("ababab")
        assert len(results) == 3

    def test_streaming_same_as_batch(self):
        patterns = ["he", "she", "his", "hers"]
        text = "ushers"
        batch = AhoCorasick(patterns).search(text)
        sac = StreamingAhoCorasick(patterns)
        streaming = []
        for ch in text:
            streaming.extend(sac.feed(ch))
        assert sorted(batch) == sorted(streaming)

    def test_large_chunks(self):
        sac = StreamingAhoCorasick(["xyz"])
        chunk = "a" * 10000 + "xyz" + "b" * 10000
        results = sac.feed(chunk)
        assert len(results) == 1
        assert results[0][0] == 10000

    def test_build_required(self):
        sac = StreamingAhoCorasick()
        sac.add_pattern("test")
        with pytest.raises(RuntimeError):
            sac.feed("test")


# ===========================================================================
# Variant 3: WeightedAhoCorasick
# ===========================================================================

class TestWeightedAhoCorasick:
    def test_basic_weights(self):
        wac = WeightedAhoCorasick()
        wac.add_pattern("cat", weight=1.0)
        wac.add_pattern("catch", weight=5.0)
        wac.build()
        results = wac.search("catch")
        assert any(r[3] == 5.0 for r in results)

    def test_search_best(self):
        wac = WeightedAhoCorasick()
        wac.add_pattern("he", weight=1.0)
        wac.add_pattern("her", weight=3.0)
        wac.add_pattern("here", weight=5.0)
        wac.build()
        best = wac.search_best("here")
        # At position 0, "here" (weight 5) should be best
        assert any(r[2] == "here" for r in best)

    def test_search_top_k(self):
        wac = WeightedAhoCorasick()
        for i in range(10):
            wac.add_pattern(f"p{i}", weight=float(i))
        wac.build()
        text = " ".join(f"p{i}" for i in range(10))
        top3 = wac.search_top_k(text, k=3)
        assert len(top3) == 3
        assert top3[0][3] >= top3[1][3] >= top3[2][3]

    def test_default_weight(self):
        wac = WeightedAhoCorasick()
        wac.add_pattern("test")
        wac.build()
        results = wac.search("test")
        assert results[0][3] == 1.0

    def test_patterns_property(self):
        wac = WeightedAhoCorasick()
        wac.add_pattern("a")
        wac.add_pattern("b")
        assert wac.patterns == ["a", "b"]

    def test_best_at_same_position(self):
        wac = WeightedAhoCorasick()
        wac.add_pattern("ab", weight=1.0)
        wac.add_pattern("abc", weight=10.0)
        wac.build()
        best = wac.search_best("abc")
        pos0_matches = [r for r in best if r[0] == 0]
        assert len(pos0_matches) == 1
        assert pos0_matches[0][2] == "abc"


# ===========================================================================
# Variant 4: WildcardAhoCorasick
# ===========================================================================

class TestWildcardAhoCorasick:
    def test_basic_wildcard(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a?c")
        wac.build()
        results = wac.search("abc")
        assert len(results) == 1
        assert results[0][0] == 0

    def test_wildcard_no_match(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a?c")
        wac.build()
        assert wac.search("adc def") == [(0, 0, "a?c")]
        assert wac.search("xyz") == []

    def test_multiple_wildcards(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a??d")
        wac.build()
        results = wac.search("abcd")
        assert len(results) == 1

    def test_wildcard_at_start(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("?bc")
        wac.build()
        results = wac.search("abc")
        assert len(results) == 1

    def test_wildcard_at_end(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("ab?")
        wac.build()
        results = wac.search("abc")
        assert len(results) == 1

    def test_multiple_matches(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a?c")
        wac.build()
        results = wac.search("abc adc aec")
        assert len(results) == 3

    def test_no_wildcard_pattern(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("exact")
        wac.build()
        results = wac.search("exact match")
        assert len(results) == 1

    def test_patterns_property(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a?b")
        assert wac.patterns == ["a?b"]

    def test_custom_wildcard_char(self):
        wac = WildcardAhoCorasick(wildcard='*')
        wac.add_pattern("a*c")
        wac.build()
        results = wac.search("abc")
        assert len(results) == 1

    def test_adjacent_wildcards(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a??b")
        wac.build()
        results = wac.search("axxb")
        assert len(results) == 1
        assert wac.search("axb") == []

    def test_build_required(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a?b")
        with pytest.raises(RuntimeError):
            wac.search("aab")


# ===========================================================================
# Variant 5: AhoCorasickReplacer
# ===========================================================================

class TestAhoCorasickReplacer:
    def test_basic_replace(self):
        r = AhoCorasickReplacer()
        r.add_rule("cat", "dog")
        r.build()
        assert r.replace("the cat sat") == "the dog sat"

    def test_multiple_replacements(self):
        r = AhoCorasickReplacer()
        r.add_rule("foo", "bar")
        r.add_rule("baz", "qux")
        r.build()
        assert r.replace("foo and baz") == "bar and qux"

    def test_no_match_unchanged(self):
        r = AhoCorasickReplacer()
        r.add_rule("xyz", "abc")
        r.build()
        assert r.replace("hello world") == "hello world"

    def test_overlapping_longest_wins(self):
        r = AhoCorasickReplacer()
        r.add_rule("ab", "X")
        r.add_rule("abc", "Y")
        r.build()
        result = r.replace("abcd", mode='longest')
        assert result == "Yd"

    def test_overlapping_first_wins(self):
        r = AhoCorasickReplacer()
        r.add_rule("ab", "X")
        r.add_rule("abc", "Y")
        r.build()
        result = r.replace("abcd", mode='first')
        assert result == "Xcd"

    def test_priority_mode(self):
        r = AhoCorasickReplacer()
        r.add_rule("ab", "X", priority=10)
        r.add_rule("abc", "Y", priority=1)
        r.build()
        result = r.replace("abcd", mode='priority')
        assert result == "Xcd"

    def test_non_overlapping(self):
        r = AhoCorasickReplacer()
        r.add_rule("aa", "X")
        r.build()
        assert r.replace("aaa") == "Xa"

    def test_adjacent_replacements(self):
        r = AhoCorasickReplacer()
        r.add_rule("ab", "X")
        r.add_rule("cd", "Y")
        r.build()
        assert r.replace("abcd") == "XY"

    def test_empty_replacement(self):
        r = AhoCorasickReplacer()
        r.add_rule("remove", "")
        r.build()
        assert r.replace("please remove this") == "please  this"

    def test_longer_replacement(self):
        r = AhoCorasickReplacer()
        r.add_rule("a", "xyz")
        r.build()
        assert r.replace("bab") == "bxyzb"

    def test_patterns_property(self):
        r = AhoCorasickReplacer()
        r.add_rule("a", "b")
        assert r.patterns == ["a"]

    def test_replace_all_text(self):
        r = AhoCorasickReplacer()
        r.add_rule("hello", "goodbye")
        r.build()
        assert r.replace("hello") == "goodbye"


# ===========================================================================
# Variant 6: AhoCorasickCounter
# ===========================================================================

class TestAhoCorasickCounter:
    def test_basic_count(self):
        c = AhoCorasickCounter(["the", "he"])
        counts = c.count("the the the")
        assert counts["the"] == 3
        assert counts["he"] == 3  # overlaps

    def test_no_matches(self):
        c = AhoCorasickCounter(["xyz"])
        counts = c.count("hello")
        assert counts["xyz"] == 0

    def test_total_matches(self):
        c = AhoCorasickCounter(["a", "b"])
        assert c.total_matches("aabba") == 5

    def test_most_common(self):
        c = AhoCorasickCounter(["a", "bb", "ccc"])
        text = "a" * 10 + "bb" * 5 + "ccc" * 2
        mc = c.most_common(text)
        assert mc[0][0] == "a"
        assert mc[0][1] == 10

    def test_most_common_k(self):
        c = AhoCorasickCounter(["a", "b", "c"])
        mc = c.most_common("aabbbc", k=2)
        assert len(mc) == 2

    def test_matched_patterns(self):
        c = AhoCorasickCounter(["yes", "no", "maybe"])
        matched = c.matched_patterns("yes or no")
        assert matched == {"yes", "no"}

    def test_empty_text(self):
        c = AhoCorasickCounter(["a"])
        counts = c.count("")
        assert counts["a"] == 0

    def test_patterns_property(self):
        c = AhoCorasickCounter(["x", "y"])
        assert c.patterns == ["x", "y"]

    def test_overlapping_counts(self):
        c = AhoCorasickCounter(["aba"])
        counts = c.count("ababa")
        assert counts["aba"] == 2

    def test_build_required(self):
        c = AhoCorasickCounter()
        c.add_pattern("test")
        with pytest.raises(RuntimeError):
            c.count("test")

    def test_many_patterns_counting(self):
        # Use patterns that don't overlap (fixed-width, unique prefixes)
        patterns = [f"word_{i:03d}" for i in range(50)]
        c = AhoCorasickCounter(patterns)
        text = " ".join(patterns * 3)
        counts = c.count(text)
        for p in patterns:
            assert counts[p] == 3


# ===========================================================================
# Integration / Cross-variant tests
# ===========================================================================

class TestIntegration:
    def test_streaming_matches_batch(self):
        """Streaming and batch should produce identical results."""
        patterns = ["abc", "bc", "c", "ab"]
        text = "xabcyz"

        batch = AhoCorasick(patterns).search(text)

        sac = StreamingAhoCorasick(patterns)
        streaming = []
        for ch in text:
            streaming.extend(sac.feed(ch))

        assert sorted(batch) == sorted(streaming)

    def test_counter_matches_search_count(self):
        """Counter totals should match search result count."""
        patterns = ["he", "she", "his", "hers"]
        text = "ushers and his hers"
        ac = AhoCorasick(patterns)
        search_count = len(ac.search(text))

        c = AhoCorasickCounter(patterns)
        total = c.total_matches(text)
        assert search_count == total

    def test_replacer_with_many_rules(self):
        r = AhoCorasickReplacer()
        for i in range(20):
            r.add_rule(f"old{i}", f"new{i}")
        r.build()
        text = " ".join(f"old{i}" for i in range(20))
        result = r.replace(text)
        for i in range(20):
            assert f"new{i}" in result
            assert f"old{i}" not in result

    def test_weighted_ranking_consistency(self):
        wac = WeightedAhoCorasick()
        wac.add_pattern("a", weight=1.0)
        wac.add_pattern("ab", weight=2.0)
        wac.add_pattern("abc", weight=3.0)
        wac.build()
        top = wac.search_top_k("abc", k=1)
        assert top[0][2] == "abc"
        assert top[0][3] == 3.0

    def test_real_world_keyword_detection(self):
        """Simulate keyword/spam detection."""
        keywords = ["viagra", "casino", "lottery", "winner", "free money"]
        ac = AhoCorasick(keywords)
        clean = "Hello, I wanted to discuss the project timeline."
        spam = "Congratulations winner! Free money at casino. Viagra lottery!"
        assert not ac.contains_any(clean)
        assert ac.contains_any(spam)

    def test_real_world_log_parsing(self):
        """Simulate log level detection."""
        counter = AhoCorasickCounter(["ERROR", "WARN", "INFO", "DEBUG"])
        log = "INFO: start\nDEBUG: init\nWARN: slow\nERROR: fail\nINFO: done\nERROR: crash"
        counts = counter.count(log)
        assert counts["ERROR"] == 2
        assert counts["WARN"] == 1
        assert counts["INFO"] == 2
        assert counts["DEBUG"] == 1

    def test_real_world_censoring(self):
        """Simulate word censoring."""
        r = AhoCorasickReplacer()
        r.add_rule("bad", "***")
        r.add_rule("evil", "****")
        r.build()
        assert r.replace("this is bad and evil") == "this is *** and ****"

    def test_unicode_text(self):
        ac = AhoCorasick(["cafe", "hello"])
        results = ac.search("hello from the cafe")
        assert len(results) == 2

    def test_single_char_text(self):
        ac = AhoCorasick(["a"])
        results = ac.search("a")
        assert len(results) == 1

    def test_pattern_equals_text(self):
        ac = AhoCorasick(["exact"])
        results = ac.search("exact")
        assert len(results) == 1
        assert results[0] == (0, 0, "exact")

    def test_text_shorter_than_pattern(self):
        ac = AhoCorasick(["longpattern"])
        assert ac.search("short") == []
