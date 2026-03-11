"""Tests for C102: Aho-Corasick Multi-Pattern String Matching"""

import pytest
from aho_corasick import (
    AhoCorasick, AhoCorasickNode, Match,
    StreamingAhoCorasick, WildcardAhoCorasick, AhoCorasickSet
)


# ============================================================
# Basic Construction
# ============================================================

class TestConstruction:
    def test_empty_automaton(self):
        ac = AhoCorasick()
        ac.build()
        assert ac.search("hello") == []

    def test_single_pattern(self):
        ac = AhoCorasick()
        ac.add_pattern("he")
        ac.build()
        matches = ac.search("he")
        assert len(matches) == 1
        assert matches[0].start == 0
        assert matches[0].end == 2
        assert matches[0].pattern == "he"

    def test_add_empty_pattern_raises(self):
        ac = AhoCorasick()
        with pytest.raises(ValueError):
            ac.add_pattern("")

    def test_add_after_build_raises(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        with pytest.raises(RuntimeError):
            ac.add_pattern("b")

    def test_search_before_build_raises(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        with pytest.raises(RuntimeError):
            ac.search("abc")

    def test_double_build_is_safe(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        ac.build()  # should be idempotent
        assert len(ac.search("a")) == 1

    def test_add_patterns_bulk(self):
        ac = AhoCorasick()
        indices = ac.add_patterns(["he", "she", "his"])
        ac.build()
        assert indices == [0, 1, 2]
        assert len(ac.patterns) == 3

    def test_add_patterns_with_tuples(self):
        ac = AhoCorasick()
        indices = ac.add_patterns([("he", "pronoun"), ("she", "pronoun")])
        ac.build()
        matches = ac.search("she")
        assert any(m.label == "pronoun" for m in matches)

    def test_pattern_index_returned(self):
        ac = AhoCorasick()
        idx0 = ac.add_pattern("abc")
        idx1 = ac.add_pattern("def")
        assert idx0 == 0
        assert idx1 == 1


# ============================================================
# Classic Aho-Corasick Example
# ============================================================

class TestClassicExample:
    """The classic 'ushers' example from the original paper."""

    @pytest.fixture
    def ac(self):
        ac = AhoCorasick()
        ac.add_pattern("he")
        ac.add_pattern("she")
        ac.add_pattern("his")
        ac.add_pattern("hers")
        ac.build()
        return ac

    def test_ushers_all_matches(self, ac):
        matches = ac.search("ushers")
        patterns = {m.pattern for m in matches}
        assert "she" in patterns
        assert "he" in patterns
        assert "hers" in patterns

    def test_ushers_positions(self, ac):
        matches = ac.search("ushers")
        match_map = {m.pattern: m for m in matches}
        assert match_map["she"].start == 1
        assert match_map["she"].end == 4
        assert match_map["he"].start == 2
        assert match_map["he"].end == 4
        assert match_map["hers"].start == 2
        assert match_map["hers"].end == 6

    def test_no_match(self, ac):
        matches = ac.search("xyz")
        assert matches == []

    def test_empty_text(self, ac):
        matches = ac.search("")
        assert matches == []


# ============================================================
# Overlapping Matches
# ============================================================

class TestOverlapping:
    def test_overlapping_patterns(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.add_pattern("abc")
        ac.add_pattern("bc")
        ac.build()
        matches = ac.search("abc", overlapping=True)
        patterns = {m.pattern for m in matches}
        assert patterns == {"ab", "abc", "bc"}

    def test_non_overlapping(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.add_pattern("abc")
        ac.add_pattern("bc")
        ac.build()
        matches = ac.search("abc", overlapping=False)
        # Leftmost-longest: "abc" starts at 0
        assert len(matches) >= 1
        assert matches[0].start == 0

    def test_non_overlapping_skips(self):
        ac = AhoCorasick()
        ac.add_pattern("aa")
        ac.build()
        matches = ac.search("aaa", overlapping=False)
        assert len(matches) == 1
        assert matches[0].start == 0

    def test_overlapping_repeating(self):
        ac = AhoCorasick()
        ac.add_pattern("aa")
        ac.build()
        matches = ac.search("aaa", overlapping=True)
        assert len(matches) == 2
        assert matches[0].start == 0
        assert matches[1].start == 1

    def test_nested_patterns(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.add_pattern("aa")
        ac.add_pattern("aaa")
        ac.build()
        matches = ac.search("aaa", overlapping=True)
        assert len(matches) == 6  # a at 0,1,2; aa at 0,1; aaa at 0


# ============================================================
# Failure Links
# ============================================================

class TestFailureLinks:
    def test_failure_link_basic(self):
        ac = AhoCorasick()
        ac.add_pattern("abc")
        ac.add_pattern("bc")
        ac.build()
        matches = ac.search("abc")
        patterns = {m.pattern for m in matches}
        assert "abc" in patterns
        assert "bc" in patterns

    def test_failure_chain(self):
        ac = AhoCorasick()
        ac.add_pattern("abcde")
        ac.add_pattern("cde")
        ac.add_pattern("e")
        ac.build()
        matches = ac.search("abcde")
        patterns = {m.pattern for m in matches}
        assert patterns == {"abcde", "cde", "e"}

    def test_repeated_failure_fallback(self):
        ac = AhoCorasick()
        ac.add_pattern("aba")
        ac.add_pattern("a")
        ac.build()
        matches = ac.search("ababa")
        a_matches = [m for m in matches if m.pattern == "a"]
        aba_matches = [m for m in matches if m.pattern == "aba"]
        assert len(a_matches) == 3  # positions 0, 2, 4
        assert len(aba_matches) == 2  # positions 0-2, 2-4

    def test_long_failure_chain(self):
        ac = AhoCorasick()
        ac.add_pattern("abab")
        ac.add_pattern("ab")
        ac.build()
        matches = ac.search("ababab")
        ab_matches = [m for m in matches if m.pattern == "ab"]
        abab_matches = [m for m in matches if m.pattern == "abab"]
        assert len(ab_matches) == 3  # 0,2,4
        assert len(abab_matches) == 2  # 0,2


# ============================================================
# Dictionary Links
# ============================================================

class TestDictionaryLinks:
    def test_dict_link_shorter_pattern(self):
        ac = AhoCorasick()
        ac.add_pattern("abcdef")
        ac.add_pattern("cde")
        ac.add_pattern("ef")
        ac.build()
        matches = ac.search("abcdef")
        patterns = {m.pattern for m in matches}
        assert patterns == {"abcdef", "cde", "ef"}

    def test_multiple_dict_links(self):
        ac = AhoCorasick()
        ac.add_pattern("abcd")
        ac.add_pattern("bcd")
        ac.add_pattern("cd")
        ac.add_pattern("d")
        ac.build()
        matches = ac.search("abcd")
        patterns = {m.pattern for m in matches}
        assert patterns == {"abcd", "bcd", "cd", "d"}


# ============================================================
# Case Sensitivity
# ============================================================

class TestCaseSensitivity:
    def test_case_sensitive_default(self):
        ac = AhoCorasick()
        ac.add_pattern("Hello")
        ac.build()
        assert len(ac.search("Hello")) == 1
        assert len(ac.search("hello")) == 0

    def test_case_insensitive(self):
        ac = AhoCorasick(case_sensitive=False)
        ac.add_pattern("Hello")
        ac.build()
        assert len(ac.search("HELLO")) == 1
        assert len(ac.search("hello")) == 1
        assert len(ac.search("hElLo")) == 1

    def test_case_insensitive_preserves_original(self):
        ac = AhoCorasick(case_sensitive=False)
        ac.add_pattern("HeLLo")
        ac.build()
        matches = ac.search("hello world")
        assert matches[0].pattern == "HeLLo"  # original case preserved

    def test_case_insensitive_multiple(self):
        ac = AhoCorasick(case_sensitive=False)
        ac.add_pattern("cat")
        ac.add_pattern("CAR")
        ac.build()
        matches = ac.search("THE CAT AND CAR")
        patterns = {m.pattern for m in matches}
        assert "cat" in patterns
        assert "CAR" in patterns


# ============================================================
# Match Object
# ============================================================

class TestMatchObject:
    def test_match_repr(self):
        m = Match(0, 3, "abc", 0)
        assert "abc" in repr(m)
        assert "0" in repr(m)
        assert "3" in repr(m)

    def test_match_equality(self):
        m1 = Match(0, 3, "abc", 0)
        m2 = Match(0, 3, "abc", 0)
        assert m1 == m2

    def test_match_inequality(self):
        m1 = Match(0, 3, "abc", 0)
        m2 = Match(1, 4, "abc", 0)
        assert m1 != m2

    def test_match_hash(self):
        m1 = Match(0, 3, "abc", 0)
        m2 = Match(0, 3, "abc", 0)
        assert hash(m1) == hash(m2)
        assert len({m1, m2}) == 1

    def test_match_label(self):
        m = Match(0, 3, "abc", 0, label="keyword")
        assert m.label == "keyword"

    def test_match_not_equal_to_other_type(self):
        m = Match(0, 3, "abc", 0)
        assert m != "abc"


# ============================================================
# Contains Any
# ============================================================

class TestContainsAny:
    def test_contains_any_true(self):
        ac = AhoCorasick()
        ac.add_patterns(["bad", "evil", "wrong"])
        ac.build()
        assert ac.contains_any("this is bad") is True

    def test_contains_any_false(self):
        ac = AhoCorasick()
        ac.add_patterns(["bad", "evil", "wrong"])
        ac.build()
        assert ac.contains_any("this is good") is False

    def test_contains_any_empty_text(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        assert ac.contains_any("") is False

    def test_contains_any_before_build_raises(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        with pytest.raises(RuntimeError):
            ac.contains_any("a")

    def test_contains_any_via_dict_link(self):
        ac = AhoCorasick()
        ac.add_pattern("abc")
        ac.add_pattern("c")
        ac.build()
        # "c" is found via dict link when matching "abc"
        assert ac.contains_any("xabc") is True


# ============================================================
# Match Count
# ============================================================

class TestMatchCount:
    def test_match_count_zero(self):
        ac = AhoCorasick()
        ac.add_pattern("xyz")
        ac.build()
        assert ac.match_count("abc") == 0

    def test_match_count_multiple(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        assert ac.match_count("aaa") == 3

    def test_match_count_overlapping(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.add_pattern("b")
        ac.build()
        assert ac.match_count("ab") == 2

    def test_matched_patterns_set(self):
        ac = AhoCorasick()
        ac.add_patterns(["he", "she", "his", "hers"])
        ac.build()
        found = ac.matched_patterns("ushers")
        assert found == {"he", "she", "hers"}


# ============================================================
# Replace
# ============================================================

class TestReplace:
    def test_replace_dict(self):
        ac = AhoCorasick()
        ac.add_patterns(["bad", "evil"])
        ac.build()
        result = ac.replace("this is bad and evil", {"bad": "good", "evil": "nice"})
        assert result == "this is good and nice"

    def test_replace_no_match(self):
        ac = AhoCorasick()
        ac.add_pattern("xyz")
        ac.build()
        assert ac.replace("hello world", {"xyz": "abc"}) == "hello world"

    def test_replace_callback(self):
        ac = AhoCorasick()
        ac.add_pattern("cat")
        ac.build()
        result = ac.replace("the cat sat on the cat", lambda m: m.pattern.upper())
        assert result == "the CAT sat on the CAT"

    def test_replace_overlapping_takes_leftmost(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.add_pattern("bc")
        ac.build()
        result = ac.replace("abc", {"ab": "X", "bc": "Y"})
        assert result == "Xc"

    def test_replace_empty_text(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        assert ac.replace("", {"a": "b"}) == ""

    def test_replace_adjacent(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.add_pattern("cd")
        ac.build()
        result = ac.replace("abcd", {"ab": "X", "cd": "Y"})
        assert result == "XY"

    def test_replace_preserves_unmatched(self):
        ac = AhoCorasick()
        ac.add_pattern("o")
        ac.build()
        result = ac.replace("hello world", {"o": "0"})
        assert result == "hell0 w0rld"


# ============================================================
# Callback Search
# ============================================================

class TestCallbackSearch:
    def test_callback_all(self):
        ac = AhoCorasick()
        ac.add_patterns(["a", "b", "c"])
        ac.build()
        found = []
        ac.search_callback("abc", lambda m: found.append(m.pattern))
        assert set(found) == {"a", "b", "c"}

    def test_callback_early_stop(self):
        ac = AhoCorasick()
        ac.add_patterns(["a", "b", "c"])
        ac.build()
        found = []
        def cb(m):
            found.append(m.pattern)
            if m.pattern == "a":
                return False  # stop after first
        ac.search_callback("abc", cb)
        assert found == ["a"]

    def test_callback_before_build_raises(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        with pytest.raises(RuntimeError):
            ac.search_callback("a", lambda m: None)


# ============================================================
# Streaming
# ============================================================

class TestStreaming:
    def test_streaming_single_chunk(self):
        ac = AhoCorasick()
        ac.add_pattern("hello")
        ac.build()
        stream = StreamingAhoCorasick(ac)
        matches = stream.feed("hello world")
        assert len(matches) == 1
        assert matches[0].pattern == "hello"

    def test_streaming_split_pattern(self):
        ac = AhoCorasick()
        ac.add_pattern("hello")
        ac.build()
        stream = StreamingAhoCorasick(ac)
        m1 = stream.feed("hel")
        m2 = stream.feed("lo")
        assert len(m1) == 0
        assert len(m2) == 1
        assert m2[0].pattern == "hello"

    def test_streaming_positions_are_global(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.build()
        stream = StreamingAhoCorasick(ac)
        stream.feed("xa")
        m = stream.feed("b")
        assert len(m) == 1
        assert m[0].start == 1
        assert m[0].end == 3

    def test_streaming_multiple_matches_across_chunks(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.build()
        stream = StreamingAhoCorasick(ac)
        stream.feed("ab")
        stream.feed("ab")
        all_m = stream.get_all_matches()
        assert len(all_m) == 2
        assert all_m[0].start == 0
        assert all_m[1].start == 2

    def test_streaming_reset(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        stream = StreamingAhoCorasick(ac)
        stream.feed("aaa")
        assert len(stream.get_all_matches()) == 3
        stream.reset()
        assert len(stream.get_all_matches()) == 0
        assert stream.position == 0

    def test_streaming_unbuilt_raises(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        with pytest.raises(RuntimeError):
            StreamingAhoCorasick(ac)

    def test_streaming_empty_chunks(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.build()
        stream = StreamingAhoCorasick(ac)
        m1 = stream.feed("")
        m2 = stream.feed("a")
        m3 = stream.feed("")
        m4 = stream.feed("b")
        assert len(m1) == 0
        assert len(m2) == 0
        assert len(m3) == 0
        assert len(m4) == 1


# ============================================================
# Wildcard Patterns
# ============================================================

class TestWildcard:
    def test_wildcard_single(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("h?llo")
        wac.build()
        matches = wac.search("hello")
        assert len(matches) == 1
        assert matches[0].start == 0

    def test_wildcard_multiple_chars(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("h?llo")
        wac.build()
        m1 = wac.search("hello")
        m2 = wac.search("hallo")
        m3 = wac.search("hullo")
        assert len(m1) == 1
        assert len(m2) == 1
        assert len(m3) == 1

    def test_wildcard_no_match(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("h?llo")
        wac.build()
        matches = wac.search("hllo")  # too short
        assert len(matches) == 0

    def test_wildcard_at_start(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("?bc")
        wac.build()
        matches = wac.search("abc")
        assert len(matches) == 1

    def test_wildcard_at_end(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("ab?")
        wac.build()
        matches = wac.search("abx")
        assert len(matches) == 1

    def test_multiple_wildcards(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("a?c?e")
        wac.build()
        matches = wac.search("abcde")
        assert len(matches) == 1

    def test_all_wildcards(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("???")
        wac.build()
        matches = wac.search("abcde")
        assert len(matches) == 3  # at positions 0,1,2

    def test_wildcard_multiple_patterns(self):
        wac = WildcardAhoCorasick()
        wac.add_pattern("c?t")
        wac.add_pattern("d?g")
        wac.build()
        matches = wac.search("the cat and dog")
        patterns = {m.pattern for m in matches}
        assert "c?t" in patterns
        assert "d?g" in patterns

    def test_wildcard_case_insensitive(self):
        wac = WildcardAhoCorasick(case_sensitive=False)
        wac.add_pattern("H?LLO")
        wac.build()
        matches = wac.search("hello")
        assert len(matches) == 1

    def test_wildcard_empty_raises(self):
        wac = WildcardAhoCorasick()
        with pytest.raises(ValueError):
            wac.add_pattern("")


# ============================================================
# AhoCorasickSet
# ============================================================

class TestAhoCorasickSet:
    def test_set_add_and_search(self):
        s = AhoCorasickSet()
        s.add("greeting", "hello")
        s.add("farewell", "bye")
        matches = s.search("hello and bye")
        labels = {m.label for m in matches}
        assert "greeting" in labels
        assert "farewell" in labels

    def test_set_remove(self):
        s = AhoCorasickSet()
        s.add("a", "hello")
        s.add("b", "bye")
        s.remove("b")
        matches = s.search("hello and bye")
        patterns = {m.pattern for m in matches}
        assert "hello" in patterns
        assert "bye" not in patterns

    def test_set_contains_any(self):
        s = AhoCorasickSet()
        s.add("bad1", "bad")
        s.add("bad2", "evil")
        assert s.contains_any("this is bad") is True
        assert s.contains_any("this is good") is False

    def test_set_pattern_count(self):
        s = AhoCorasickSet()
        s.add("a", "hello")
        s.add("b", "world")
        assert s.pattern_count == 2
        s.remove("a")
        assert s.pattern_count == 1

    def test_set_auto_rebuild(self):
        s = AhoCorasickSet()
        s.add("a", "hello")
        assert len(s.search("hello")) == 1
        s.add("b", "world")
        # Should auto-rebuild
        assert len(s.search("hello world")) == 2


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_char_pattern(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        matches = ac.search("aaa")
        assert len(matches) == 3

    def test_single_char_text(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.build()
        matches = ac.search("a")
        assert len(matches) == 1

    def test_pattern_longer_than_text(self):
        ac = AhoCorasick()
        ac.add_pattern("abcdef")
        ac.build()
        matches = ac.search("abc")
        assert len(matches) == 0

    def test_identical_patterns(self):
        ac = AhoCorasick()
        ac.add_pattern("abc")
        ac.add_pattern("abc")
        ac.build()
        matches = ac.search("abc")
        assert len(matches) == 2  # both pattern indices match

    def test_prefix_patterns(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.add_pattern("ab")
        ac.add_pattern("abc")
        ac.build()
        matches = ac.search("abc")
        patterns = {m.pattern for m in matches}
        assert patterns == {"a", "ab", "abc"}

    def test_suffix_patterns(self):
        ac = AhoCorasick()
        ac.add_pattern("c")
        ac.add_pattern("bc")
        ac.add_pattern("abc")
        ac.build()
        matches = ac.search("abc")
        patterns = {m.pattern for m in matches}
        assert patterns == {"c", "bc", "abc"}

    def test_many_patterns(self):
        ac = AhoCorasick()
        for i in range(100):
            ac.add_pattern(f"pat{i}")
        ac.build()
        matches = ac.search("pat0 pat50 pat99")
        patterns = {m.pattern for m in matches}
        assert "pat0" in patterns
        assert "pat50" in patterns
        assert "pat99" in patterns

    def test_special_characters(self):
        ac = AhoCorasick()
        ac.add_pattern("$100")
        ac.add_pattern("50%")
        ac.build()
        matches = ac.search("costs $100 or 50%")
        patterns = {m.pattern for m in matches}
        assert "$100" in patterns
        assert "50%" in patterns

    def test_unicode_text(self):
        ac = AhoCorasick()
        ac.add_pattern("hello")
        ac.build()
        matches = ac.search("say hello!")
        assert len(matches) == 1

    def test_long_text(self):
        ac = AhoCorasick()
        ac.add_pattern("needle")
        ac.build()
        text = "haystack " * 10000 + "needle" + " haystack" * 10000
        matches = ac.search(text)
        assert len(matches) == 1

    def test_all_same_char(self):
        ac = AhoCorasick()
        ac.add_pattern("aaa")
        ac.build()
        matches = ac.search("aaaaa")
        assert len(matches) == 3  # at 0,1,2


# ============================================================
# Sorting and Order
# ============================================================

class TestSortingOrder:
    def test_matches_sorted_by_start(self):
        ac = AhoCorasick()
        ac.add_patterns(["a", "b", "c"])
        ac.build()
        matches = ac.search("cba")
        assert matches[0].start <= matches[1].start <= matches[2].start

    def test_same_start_sorted_by_length_desc(self):
        ac = AhoCorasick()
        ac.add_pattern("a")
        ac.add_pattern("ab")
        ac.add_pattern("abc")
        ac.build()
        matches = ac.search("abc")
        # At position 0: abc (len 3), ab (len 2), a (len 1)
        at_zero = [m for m in matches if m.start == 0]
        assert len(at_zero) == 3
        assert at_zero[0].pattern == "abc"  # longest first
        assert at_zero[1].pattern == "ab"
        assert at_zero[2].pattern == "a"


# ============================================================
# Labels and Categories
# ============================================================

class TestLabels:
    def test_pattern_labels(self):
        ac = AhoCorasick()
        ac.add_pattern("error", label="severity_high")
        ac.add_pattern("warning", label="severity_medium")
        ac.add_pattern("info", label="severity_low")
        ac.build()
        matches = ac.search("error: file not found")
        assert matches[0].label == "severity_high"

    def test_label_none_default(self):
        ac = AhoCorasick()
        ac.add_pattern("test")
        ac.build()
        matches = ac.search("test")
        assert matches[0].label is None

    def test_bulk_add_with_label(self):
        ac = AhoCorasick()
        ac.add_patterns(["cat", "dog", "bird"], label="animal")
        ac.build()
        matches = ac.search("the cat and dog")
        for m in matches:
            assert m.label == "animal"


# ============================================================
# Practical Use Cases
# ============================================================

class TestPracticalUseCases:
    def test_keyword_filtering(self):
        """Content moderation: detect banned words."""
        ac = AhoCorasick(case_sensitive=False)
        ac.add_patterns(["spam", "scam", "fraud"])
        ac.build()
        assert ac.contains_any("This is a SCAM!") is True
        assert ac.contains_any("This is legitimate") is False

    def test_dna_pattern_search(self):
        """Bioinformatics: find DNA motifs."""
        ac = AhoCorasick()
        ac.add_patterns(["ATCG", "GCTA", "TTAA"])
        ac.build()
        dna = "AATCGATCGCTATTAAGCTA"
        matches = ac.search(dna)
        assert len(matches) >= 3

    def test_log_pattern_detection(self):
        """Log analysis: detect error patterns."""
        ac = AhoCorasick()
        ac.add_pattern("ERROR", label="error")
        ac.add_pattern("WARN", label="warning")
        ac.add_pattern("FATAL", label="fatal")
        ac.build()
        log = "2024-01-01 ERROR: disk full\n2024-01-01 WARN: low memory\n2024-01-01 FATAL: crash"
        matches = ac.search(log)
        labels = [m.label for m in matches]
        assert "error" in labels
        assert "warning" in labels
        assert "fatal" in labels

    def test_url_detection(self):
        """Find protocol prefixes in text."""
        ac = AhoCorasick()
        ac.add_patterns(["http://", "https://", "ftp://"])
        ac.build()
        text = "visit https://example.com or ftp://files.example.com"
        matches = ac.search(text)
        assert len(matches) == 2

    def test_multi_language_keywords(self):
        """Find programming language keywords."""
        ac = AhoCorasick()
        ac.add_pattern("def", label="python")
        ac.add_pattern("function", label="javascript")
        ac.add_pattern("fn", label="rust")
        ac.add_pattern("func", label="go")
        ac.build()
        code = "def hello(): function world() fn greet() func main()"
        matches = ac.search(code)
        labels = {m.label for m in matches}
        assert labels == {"python", "javascript", "rust", "go"}

    def test_censoring(self):
        """Replace offensive words with asterisks."""
        ac = AhoCorasick(case_sensitive=False)
        ac.add_patterns(["bad", "evil"])
        ac.build()
        result = ac.replace("This is BAD and evil stuff",
                           lambda m: "*" * len(m.pattern))
        assert result == "This is *** and **** stuff"


# ============================================================
# Performance / Stress
# ============================================================

class TestPerformance:
    def test_many_patterns_many_matches(self):
        """100 patterns, long text -- patterns may overlap (w1 in w10)."""
        ac = AhoCorasick()
        words = [f"w{i}" for i in range(100)]
        ac.add_patterns(words)
        ac.build()
        text = " ".join(words * 10)
        matches = ac.search(text)
        # At least 1000 matches (exact count higher due to w1 matching in w10, etc.)
        assert len(matches) >= 1000

    def test_deep_trie(self):
        """Pattern with many characters (deep trie)."""
        ac = AhoCorasick()
        long_pattern = "a" * 500
        ac.add_pattern(long_pattern)
        ac.build()
        text = "a" * 1000
        matches = ac.search(text)
        assert len(matches) == 501

    def test_wide_trie(self):
        """Many patterns with different first characters."""
        ac = AhoCorasick()
        import string
        for ch in string.ascii_lowercase:
            ac.add_pattern(ch + "xyz")
        ac.build()
        text = "axyz bxyz cxyz"
        matches = ac.search(text)
        assert len(matches) == 3

    def test_streaming_large(self):
        """Streaming with many small chunks."""
        ac = AhoCorasick()
        ac.add_pattern("abcdef")
        ac.build()
        stream = StreamingAhoCorasick(ac)
        text = "xxxabcdefxxx"
        for ch in text:
            stream.feed(ch)
        all_m = stream.get_all_matches()
        assert len(all_m) == 1
        assert all_m[0].start == 3


# ============================================================
# Non-overlapping Advanced
# ============================================================

class TestNonOverlappingAdvanced:
    def test_non_overlapping_leftmost_priority(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.add_pattern("b")
        ac.build()
        matches = ac.search("ab", overlapping=False)
        # "ab" at 0 should be selected, "b" at 1 is overlapping
        assert len(matches) == 1
        assert matches[0].pattern == "ab"

    def test_non_overlapping_chain(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.add_pattern("cd")
        ac.build()
        matches = ac.search("abcd", overlapping=False)
        assert len(matches) == 2

    def test_non_overlapping_gap(self):
        ac = AhoCorasick()
        ac.add_pattern("ab")
        ac.build()
        matches = ac.search("ab_ab_ab", overlapping=False)
        assert len(matches) == 3


# ============================================================
# Node internals
# ============================================================

class TestNodeInternals:
    def test_node_defaults(self):
        n = AhoCorasickNode()
        assert n.children == {}
        assert n.fail is None
        assert n.dict_link is None
        assert n.output == []
        assert n.depth == 0
        assert n.char == ''

    def test_node_with_args(self):
        n = AhoCorasickNode('x', 3)
        assert n.char == 'x'
        assert n.depth == 3

    def test_root_fail_is_self(self):
        ac = AhoCorasick()
        assert ac.root.fail is ac.root


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
