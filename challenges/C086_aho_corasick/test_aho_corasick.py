"""Tests for C086: Aho-Corasick multi-pattern string matching."""

import pytest
from aho_corasick import (
    AhoCorasick, ACNode, Match,
    AhoCorasickStream, AhoCorasickReplacer,
    WildcardAC, ACPatternSet,
)


# =============================================================================
# Variant 1: Classic Aho-Corasick
# =============================================================================

class TestAhoCorasickBasic:
    """Basic construction and single-pattern search."""

    def test_empty_automaton(self):
        ac = AhoCorasick([])
        assert ac.pattern_count == 0
        assert ac.find_all("hello") == []
        assert ac.search("hello") is False

    def test_single_pattern(self):
        ac = AhoCorasick(["hello"])
        assert ac.pattern_count == 1
        assert ac.search("say hello world") is True
        assert ac.search("hi there") is False

    def test_single_pattern_matches(self):
        ac = AhoCorasick(["ab"])
        matches = ac.find_all("ababab")
        assert len(matches) == 3
        assert matches[0].start == 0
        assert matches[1].start == 2
        assert matches[2].start == 4

    def test_pattern_at_boundaries(self):
        ac = AhoCorasick(["abc"])
        m = ac.find_all("abc")
        assert len(m) == 1
        assert m[0].start == 0 and m[0].end == 3

    def test_no_match(self):
        ac = AhoCorasick(["xyz"])
        assert ac.find_all("abcdef") == []

    def test_empty_text(self):
        ac = AhoCorasick(["a"])
        assert ac.find_all("") == []

    def test_empty_pattern(self):
        ac = AhoCorasick([""])
        # Empty pattern is stored but never matched (no chars to traverse)
        matches = ac.find_all("ab")
        assert len(matches) == 0

    def test_patterns_property(self):
        ac = AhoCorasick(["foo", "bar"])
        assert ac.patterns == ["foo", "bar"]


class TestAhoCorasickMultiPattern:
    """Multiple pattern matching."""

    def test_two_patterns_no_overlap(self):
        ac = AhoCorasick(["he", "she"])
        matches = ac.find_all("ahshe")
        patterns_found = {m.pattern for m in matches}
        assert "she" in patterns_found
        assert "he" in patterns_found  # "he" is suffix of "she"

    def test_classic_example(self):
        """Classic AC example: {he, she, his, hers}."""
        ac = AhoCorasick(["he", "she", "his", "hers"])
        matches = ac.find_all("ushers")
        patterns = {m.pattern for m in matches}
        assert "she" in patterns
        assert "he" in patterns
        assert "hers" in patterns

    def test_overlapping_patterns(self):
        ac = AhoCorasick(["a", "ab", "abc"])
        matches = ac.find_all("abc")
        patterns = {m.pattern for m in matches}
        assert patterns == {"a", "ab", "abc"}

    def test_multiple_occurrences(self):
        ac = AhoCorasick(["cat", "dog"])
        matches = ac.find_all("catdogcat")
        assert len(matches) == 3
        assert matches[0].pattern == "cat" and matches[0].start == 0
        assert matches[1].pattern == "dog" and matches[1].start == 3
        assert matches[2].pattern == "cat" and matches[2].start == 6

    def test_substring_patterns(self):
        ac = AhoCorasick(["a", "aa", "aaa"])
        matches = ac.find_all("aaa")
        # "a" at 0,1,2; "aa" at 0,1; "aaa" at 0
        assert len(matches) == 6

    def test_many_patterns(self):
        patterns = [f"p{i}" for i in range(100)]
        ac = AhoCorasick(patterns)
        text = " ".join(patterns)
        matches = ac.find_all(text)
        found_indices = {m.pattern_idx for m in matches}
        assert len(found_indices) == 100

    def test_shared_prefix_patterns(self):
        ac = AhoCorasick(["abc", "abd", "abe"])
        matches = ac.find_all("abcabdabe")
        assert len(matches) == 3
        assert {m.pattern for m in matches} == {"abc", "abd", "abe"}

    def test_shared_suffix_patterns(self):
        ac = AhoCorasick(["xbc", "ybc", "zbc"])
        matches = ac.find_all("xbcybczbc")
        assert len(matches) == 3


class TestAhoCorasickFailureLinks:
    """Tests that exercise failure link traversal."""

    def test_failure_link_basic(self):
        """Pattern "abc" with text "aababc" -- failure links traverse back."""
        ac = AhoCorasick(["abc"])
        matches = ac.find_all("aababc")
        assert len(matches) == 1
        assert matches[0].start == 3

    def test_failure_link_chain(self):
        """Long chain of failure links."""
        ac = AhoCorasick(["abcabc", "bcabc", "cabc", "abc"])
        matches = ac.find_all("abcabc")
        patterns = {m.pattern for m in matches}
        assert "abc" in patterns
        assert "abcabc" in patterns

    def test_dict_suffix_link(self):
        """Dict suffix link shortcuts to next output node."""
        ac = AhoCorasick(["he", "she", "her", "hers"])
        matches = ac.find_all("shers")
        patterns = {m.pattern for m in matches}
        assert "she" in patterns
        assert "he" in patterns
        assert "her" in patterns
        assert "hers" in patterns

    def test_repeated_char_failure(self):
        ac = AhoCorasick(["aab"])
        matches = ac.find_all("aaab")
        assert len(matches) == 1
        assert matches[0].start == 1


class TestAhoCorasickFindFirst:
    """find_first returns the earliest match."""

    def test_find_first_basic(self):
        ac = AhoCorasick(["dog", "cat"])
        m = ac.find_first("I have a cat and a dog")
        assert m is not None
        assert m.pattern == "cat"

    def test_find_first_none(self):
        ac = AhoCorasick(["xyz"])
        assert ac.find_first("abcdef") is None

    def test_find_first_at_start(self):
        ac = AhoCorasick(["hello"])
        m = ac.find_first("hello world")
        assert m.start == 0

    def test_find_first_overlapping(self):
        ac = AhoCorasick(["a", "ab"])
        m = ac.find_first("xab")
        assert m.pattern == "a"
        assert m.start == 1


class TestAhoCorasickNonOverlapping:
    """find_non_overlapping returns leftmost-longest non-overlapping matches."""

    def test_non_overlapping_basic(self):
        ac = AhoCorasick(["ab", "bc"])
        matches = ac.find_non_overlapping("abc")
        assert len(matches) == 1
        assert matches[0].pattern == "ab"

    def test_non_overlapping_longest(self):
        ac = AhoCorasick(["a", "ab", "abc"])
        matches = ac.find_non_overlapping("abc")
        # "a" at 0 is leftmost, takes precedence
        assert matches[0].start == 0

    def test_non_overlapping_multiple(self):
        ac = AhoCorasick(["ab", "cd"])
        matches = ac.find_non_overlapping("abcd")
        assert len(matches) == 2

    def test_non_overlapping_skip(self):
        ac = AhoCorasick(["aba", "ba"])
        matches = ac.find_non_overlapping("ababa")
        assert matches[0].pattern == "aba"
        assert matches[0].start == 0


class TestAhoCorasickCaseInsensitive:
    """Case-insensitive matching."""

    def test_case_insensitive_basic(self):
        ac = AhoCorasick(["hello"], case_sensitive=False)
        assert ac.search("HELLO WORLD") is True
        assert ac.search("Hello") is True

    def test_case_insensitive_matches(self):
        ac = AhoCorasick(["ABC"], case_sensitive=False)
        matches = ac.find_all("abcABCabc")
        assert len(matches) == 3

    def test_case_insensitive_mixed(self):
        ac = AhoCorasick(["Cat", "DOG"], case_sensitive=False)
        matches = ac.find_all("my cat chased a dog")
        assert len(matches) == 2

    def test_case_sensitive_by_default(self):
        ac = AhoCorasick(["Hello"])
        assert ac.search("hello") is False
        assert ac.search("Hello") is True


class TestAhoCorasickUtilities:
    """count_matches, match_at, which_patterns, pattern_positions."""

    def test_count_matches(self):
        ac = AhoCorasick(["ab"])
        assert ac.count_matches("ababab") == 3

    def test_match_at_found(self):
        ac = AhoCorasick(["cat"])
        m = ac.match_at("the cat sat", 4)
        assert m is not None
        assert m.pattern == "cat"

    def test_match_at_not_found(self):
        ac = AhoCorasick(["cat"])
        assert ac.match_at("the cat sat", 0) is None

    def test_which_patterns(self):
        ac = AhoCorasick(["foo", "bar", "baz"])
        found = ac.which_patterns("foobar")
        assert found == {0, 1}

    def test_pattern_positions(self):
        ac = AhoCorasick(["ab", "cd"])
        pos = ac.pattern_positions("abcdab")
        assert pos[0] == [0, 4]  # "ab" at 0 and 4
        assert pos[1] == [2]     # "cd" at 2


class TestAhoCorasickErrors:
    """Error handling."""

    def test_add_after_build_raises(self):
        ac = AhoCorasick(["a"])
        with pytest.raises(RuntimeError):
            ac.add_pattern("b")

    def test_search_before_build_raises(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        with pytest.raises(RuntimeError):
            ac.find_all("text")

    def test_double_build_ok(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        ac.build()  # idempotent
        assert ac.search("a") is True


class TestMatchObject:
    """Match object behavior."""

    def test_match_equality(self):
        m1 = Match(0, 3, 0, "abc")
        m2 = Match(0, 3, 0, "abc")
        assert m1 == m2

    def test_match_inequality(self):
        m1 = Match(0, 3, 0, "abc")
        m2 = Match(1, 4, 0, "abc")
        assert m1 != m2

    def test_match_hash(self):
        m1 = Match(0, 3, 0, "abc")
        m2 = Match(0, 3, 0, "abc")
        assert hash(m1) == hash(m2)
        assert len({m1, m2}) == 1

    def test_match_repr(self):
        m = Match(5, 8, 0, "abc")
        assert "5" in repr(m) and "8" in repr(m) and "abc" in repr(m)


# =============================================================================
# Variant 2: Streaming Aho-Corasick
# =============================================================================

class TestAhoCorasickStream:
    """Streaming/incremental matching."""

    def test_stream_basic(self):
        stream = AhoCorasickStream(["hello"])
        m1 = stream.feed("hel")
        assert m1 == []
        m2 = stream.feed("lo world")
        assert len(m2) == 1
        assert m2[0].pattern == "hello"
        assert m2[0].start == 0

    def test_stream_multiple_chunks(self):
        stream = AhoCorasickStream(["ab", "cd"])
        stream.feed("a")
        stream.feed("b")
        stream.feed("c")
        stream.feed("d")
        assert len(stream.matches) == 2
        assert stream.matches[0].pattern == "ab"
        assert stream.matches[1].pattern == "cd"

    def test_stream_cross_boundary(self):
        """Pattern spans chunk boundary."""
        stream = AhoCorasickStream(["cross"])
        stream.feed("abc cr")
        m = stream.feed("oss def")
        assert len(m) == 1
        assert m[0].pattern == "cross"
        assert m[0].start == 4

    def test_stream_total_processed(self):
        stream = AhoCorasickStream(["x"])
        stream.feed("abc")
        stream.feed("def")
        assert stream.total_processed == 6

    def test_stream_reset(self):
        stream = AhoCorasickStream(["ab"])
        stream.feed("ab")
        assert len(stream.matches) == 1
        stream.reset()
        assert len(stream.matches) == 0
        assert stream.total_processed == 0

    def test_stream_feed_all(self):
        stream = AhoCorasickStream(["xyz"])
        m = stream.feed_all(["x", "y", "z"])
        assert len(m) == 1
        assert m[0].pattern == "xyz"

    def test_stream_positions_correct(self):
        """Positions should be global (across chunks)."""
        stream = AhoCorasickStream(["bc"])
        stream.feed("ab")     # "bc" starts at position 1 but 'c' comes in next chunk
        m = stream.feed("cd")
        assert len(m) == 1
        assert m[0].start == 1
        assert m[0].end == 3

    def test_stream_many_matches(self):
        stream = AhoCorasickStream(["a"])
        m = stream.feed("aaaa")
        assert len(m) == 4

    def test_stream_no_build_raises(self):
        stream = AhoCorasickStream()
        stream.add_pattern("x")
        with pytest.raises(RuntimeError):
            stream.feed("x")

    def test_stream_manual_build(self):
        stream = AhoCorasickStream()
        stream.add_pattern("hi")
        stream.build()
        m = stream.feed("hi")
        assert len(m) == 1


# =============================================================================
# Variant 3: Aho-Corasick Replacer
# =============================================================================

class TestAhoCorasickReplacer:
    """Multi-pattern simultaneous replacement."""

    def test_replace_basic(self):
        r = AhoCorasickReplacer({"cat": "dog", "bird": "fish"})
        assert r.replace("I have a cat and a bird") == "I have a dog and a fish"

    def test_replace_no_match(self):
        r = AhoCorasickReplacer({"xyz": "abc"})
        assert r.replace("hello world") == "hello world"

    def test_replace_multiple_same(self):
        r = AhoCorasickReplacer({"ab": "XY"})
        assert r.replace("ababab") == "XYXYXY"

    def test_replace_overlapping_prefers_leftmost(self):
        r = AhoCorasickReplacer({"ab": "X", "bc": "Y"})
        result = r.replace("abc")
        assert result == "Xc"  # "ab" matched first

    def test_replace_empty_result(self):
        r = AhoCorasickReplacer({"hello": ""})
        assert r.replace("hello world") == " world"

    def test_replace_longer_replacement(self):
        r = AhoCorasickReplacer({"a": "xyz"})
        assert r.replace("aaa") == "xyzxyzxyz"

    def test_replace_case_insensitive(self):
        r = AhoCorasickReplacer({"Hello": "Hi"}, case_sensitive=False)
        assert r.replace("HELLO hello Hello") == "Hi Hi Hi"

    def test_replace_with_callback(self):
        r = AhoCorasickReplacer({"cat": "x", "dog": "y"})
        result = r.replace_with("cat and dog", lambda m: m.pattern.upper())
        assert result == "CAT and DOG"

    def test_replace_count(self):
        r = AhoCorasickReplacer({"a": "b"})
        text, count = r.replace_count("aaa")
        assert text == "bbb"
        assert count == 3

    def test_replace_count_zero(self):
        r = AhoCorasickReplacer({"x": "y"})
        text, count = r.replace_count("abc")
        assert text == "abc"
        assert count == 0

    def test_replace_preserves_non_matched(self):
        r = AhoCorasickReplacer({"world": "earth"})
        assert r.replace("hello world!") == "hello earth!"


# =============================================================================
# Variant 4: Wildcard Aho-Corasick
# =============================================================================

class TestWildcardAC:
    """Wildcard pattern matching."""

    def test_no_wildcard(self):
        wac = WildcardAC(["abc"])
        matches = wac.find_all("xabcy")
        assert len(matches) == 1
        assert matches[0].start == 1

    def test_single_wildcard(self):
        wac = WildcardAC(["a?c"])
        matches = wac.find_all("abc adc aec")
        assert len(matches) == 3

    def test_wildcard_at_start(self):
        wac = WildcardAC(["?bc"])
        matches = wac.find_all("abc xbc")
        assert len(matches) == 2

    def test_wildcard_at_end(self):
        wac = WildcardAC(["ab?"])
        matches = wac.find_all("abc abd")
        assert len(matches) == 2

    def test_multiple_wildcards(self):
        wac = WildcardAC(["a??d"])
        matches = wac.find_all("abcd axyd")
        assert len(matches) == 2

    def test_all_wildcards(self):
        wac = WildcardAC(["???"])
        matches = wac.find_all("abcde")
        assert len(matches) == 3  # positions 0,1,2

    def test_wildcard_search(self):
        wac = WildcardAC(["h?llo"])
        assert wac.search("hello") is True
        assert wac.search("hxllo") is True
        assert wac.search("hllo") is False

    def test_wildcard_find_first(self):
        wac = WildcardAC(["?at"])
        m = wac.find_first("the cat sat")
        assert m is not None
        assert m.start == 4
        assert m.pattern == "?at"

    def test_wildcard_no_match(self):
        wac = WildcardAC(["x?z"])
        assert wac.find_all("abc") == []

    def test_wildcard_case_insensitive(self):
        wac = WildcardAC(["H?LLO"], case_sensitive=False)
        assert wac.search("hello") is True

    def test_wildcard_multiple_patterns(self):
        wac = WildcardAC(["a?", "?b"])
        matches = wac.find_all("ab")
        patterns = {m.pattern for m in matches}
        assert "a?" in patterns
        assert "?b" in patterns

    def test_wildcard_custom_char(self):
        wac = WildcardAC(["a*c"], wildcard='*')
        assert wac.search("abc") is True


# =============================================================================
# ACPatternSet
# =============================================================================

class TestACPatternSet:
    """Pattern set operations."""

    def test_contains_any(self):
        ps = ACPatternSet(["cat", "dog", "fish"])
        assert ps.contains_any("I have a cat") is True
        assert ps.contains_any("I have nothing") is False

    def test_contains_all(self):
        ps = ACPatternSet(["cat", "dog"])
        assert ps.contains_all("cat and dog") is True
        assert ps.contains_all("just a cat") is False

    def test_which_found(self):
        ps = ACPatternSet(["alpha", "beta", "gamma"])
        found = ps.which_found("alpha and gamma")
        assert "alpha" in found
        assert "gamma" in found
        assert "beta" not in found

    def test_filter_texts(self):
        ps = ACPatternSet(["error", "warning"])
        texts = ["no issues", "error found", "all clear", "warning issued"]
        filtered = ps.filter_texts(texts)
        assert filtered == ["error found", "warning issued"]

    def test_pattern_count(self):
        ps = ACPatternSet(["a", "b", "c"])
        assert ps.pattern_count == 3

    def test_empty_set(self):
        ps = ACPatternSet([])
        assert ps.contains_any("hello") is False
        assert ps.contains_all("hello") is True  # vacuously true


# =============================================================================
# Edge cases and stress tests
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_char_patterns(self):
        ac = AhoCorasick(["a", "b", "c"])
        matches = ac.find_all("abc")
        assert len(matches) == 3

    def test_long_pattern(self):
        pattern = "a" * 1000
        ac = AhoCorasick([pattern])
        text = "b" * 999 + pattern + "b" * 999
        matches = ac.find_all(text)
        assert len(matches) == 1
        assert matches[0].start == 999

    def test_many_patterns_same_length(self):
        patterns = [chr(ord('a') + i) + chr(ord('a') + j) for i in range(26) for j in range(26)]
        ac = AhoCorasick(patterns)
        text = "abcdefghij"
        matches = ac.find_all(text)
        assert len(matches) == 9  # ab, bc, cd, de, ef, fg, gh, hi, ij

    def test_pattern_is_whole_text(self):
        ac = AhoCorasick(["exact"])
        matches = ac.find_all("exact")
        assert len(matches) == 1
        assert matches[0].start == 0 and matches[0].end == 5

    def test_repeated_same_pattern(self):
        ac = AhoCorasick(["aa"])
        matches = ac.find_all("aaaa")
        assert len(matches) == 3  # positions 0, 1, 2

    def test_unicode_text(self):
        ac = AhoCorasick(["hello"])
        matches = ac.find_all("say hello!")
        assert len(matches) == 1

    def test_special_chars_in_pattern(self):
        ac = AhoCorasick(["a.b", "c*d"])
        matches = ac.find_all("a.b and c*d")
        assert len(matches) == 2

    def test_binary_like_text(self):
        ac = AhoCorasick(["\x00\x01"])
        matches = ac.find_all("abc\x00\x01def")
        assert len(matches) == 1

    def test_newlines_in_pattern(self):
        ac = AhoCorasick(["line1\nline2"])
        matches = ac.find_all("start\nline1\nline2\nend")
        assert len(matches) == 1

    def test_duplicate_patterns(self):
        ac = AhoCorasick(["abc", "abc"])
        matches = ac.find_all("abc")
        # Both pattern indices should match
        assert len(matches) == 2

    def test_prefix_suffix_overlap(self):
        """Pattern where prefix = suffix (like KMP failure function)."""
        ac = AhoCorasick(["abab"])
        matches = ac.find_all("ababab")
        assert len(matches) == 2  # at 0 and 2

    def test_stress_many_short_patterns(self):
        """1000 single-char patterns."""
        import string
        patterns = list(string.ascii_lowercase)
        ac = AhoCorasick(patterns)
        text = string.ascii_lowercase * 10
        matches = ac.find_all(text)
        assert len(matches) == 260

    def test_stream_equivalence(self):
        """Streaming should produce same results as batch."""
        patterns = ["ab", "bc", "abc"]
        text = "xabcaby"

        ac = AhoCorasick(patterns)
        batch = ac.find_all(text)

        stream = AhoCorasickStream(patterns)
        for ch in text:
            stream.feed(ch)
        stream_matches = stream.matches

        assert len(batch) == len(stream_matches)
        for b, s in zip(
            sorted(batch, key=lambda m: (m.start, m.pattern_idx)),
            sorted(stream_matches, key=lambda m: (m.start, m.pattern_idx))
        ):
            assert b.start == s.start
            assert b.end == s.end
            assert b.pattern_idx == s.pattern_idx


class TestComposition:
    """Tests showing composition of AC with other systems."""

    def test_log_analysis(self):
        """Simulate log file analysis -- find multiple error patterns."""
        errors = ACPatternSet([
            "ERROR", "FATAL", "CRITICAL",
            "NullPointerException", "OutOfMemory"
        ])
        logs = [
            "2024-01-01 INFO: Starting up",
            "2024-01-01 ERROR: Connection failed",
            "2024-01-01 FATAL: NullPointerException in main",
            "2024-01-01 INFO: Retrying...",
            "2024-01-01 CRITICAL: OutOfMemory detected",
        ]
        bad_logs = errors.filter_texts(logs)
        assert len(bad_logs) == 3

    def test_content_filter(self):
        """Content filtering with replacement."""
        r = AhoCorasickReplacer({
            "badword": "***",
            "offensive": "***",
        })
        text = "This has badword and offensive content"
        result = r.replace(text)
        assert "badword" not in result
        assert "offensive" not in result
        assert "***" in result

    def test_dna_pattern_search(self):
        """DNA motif search."""
        motifs = ["ATCG", "GCTA", "ATAT"]
        ac = AhoCorasick(motifs)
        dna = "GATCGATCGATATATCG"
        matches = ac.find_all(dna)
        found = {m.pattern for m in matches}
        assert "ATCG" in found
        assert "ATAT" in found

    def test_streaming_network_inspection(self):
        """Simulate network packet inspection."""
        stream = AhoCorasickStream(["malware", "exploit", "payload"])
        packets = ["GET /index mal", "ware HTTP expl", "oit payl", "oad end"]
        all_matches = []
        for pkt in packets:
            all_matches.extend(stream.feed(pkt))
        patterns = {m.pattern for m in all_matches}
        assert patterns == {"malware", "exploit", "payload"}

    def test_wildcard_log_patterns(self):
        """Wildcard patterns for semi-structured log matching."""
        wac = WildcardAC(["ERR??", "WARN?"])
        assert wac.search("ERROR occurred") is True
        assert wac.search("WARNS about something") is True
        assert wac.search("INFO message") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
