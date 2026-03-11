"""Tests for C119: Trie / Patricia Trie / Radix Tree."""
import pytest
from trie import (
    Trie, PatriciaTrie, TernarySearchTree,
    AutocompleteTrie, GeneralizedSuffixTrie, TrieMap,
)


# =============================================================
# Trie
# =============================================================

class TestTrie:
    def test_empty(self):
        t = Trie()
        assert len(t) == 0
        assert not t
        assert "hello" not in t

    def test_insert_search(self):
        t = Trie()
        t.insert("hello")
        assert t.search("hello")
        assert "hello" in t
        assert not t.search("hell")
        assert not t.search("helloo")

    def test_insert_value(self):
        t = Trie()
        assert t.insert("key", 42) is None
        assert t.get("key") == 42
        assert t.insert("key", 99) == 42  # returns old value
        assert t.get("key") == 99

    def test_size(self):
        t = Trie()
        t.insert("a")
        t.insert("ab")
        t.insert("abc")
        assert len(t) == 3
        assert t

    def test_delete(self):
        t = Trie()
        t.insert("hello")
        t.insert("help")
        assert t.delete("hello")
        assert not t.search("hello")
        assert t.search("help")
        assert len(t) == 1

    def test_delete_nonexistent(self):
        t = Trie()
        t.insert("hello")
        assert not t.delete("world")
        assert not t.delete("hell")

    def test_delete_prefix_key(self):
        t = Trie()
        t.insert("abc")
        t.insert("ab")
        t.insert("a")
        assert t.delete("ab")
        assert t.search("abc")
        assert t.search("a")
        assert not t.search("ab")

    def test_starts_with(self):
        t = Trie()
        t.insert("apple")
        t.insert("application")
        assert t.starts_with("app")
        assert t.starts_with("apple")
        assert not t.starts_with("banana")

    def test_keys_with_prefix(self):
        t = Trie()
        for w in ["apple", "app", "application", "banana", "band"]:
            t.insert(w)
        result = t.keys_with_prefix("app")
        assert sorted(result) == ["app", "apple", "application"]

    def test_count_with_prefix(self):
        t = Trie()
        for w in ["apple", "app", "application", "banana"]:
            t.insert(w)
        assert t.count_with_prefix("app") == 3
        assert t.count_with_prefix("ban") == 1
        assert t.count_with_prefix("xyz") == 0

    def test_longest_prefix_of(self):
        t = Trie()
        t.insert("a")
        t.insert("ab")
        t.insert("abc")
        assert t.longest_prefix_of("abcdef") == "abc"
        assert t.longest_prefix_of("abd") == "ab"
        assert t.longest_prefix_of("xyz") == ""

    def test_all_keys(self):
        t = Trie()
        words = ["cat", "car", "card", "care", "careful"]
        for w in words:
            t.insert(w)
        assert t.all_keys() == sorted(words)

    def test_empty_string(self):
        t = Trie()
        t.insert("")
        assert t.search("")
        assert len(t) == 1
        assert t.delete("")
        assert not t.search("")

    def test_single_char_keys(self):
        t = Trie()
        for ch in "abcdefghij":
            t.insert(ch)
        assert len(t) == 10
        assert t.all_keys() == list("abcdefghij")

    def test_get_default(self):
        t = Trie()
        assert t.get("missing") is None
        assert t.get("missing", 42) == 42

    def test_overlapping_inserts(self):
        t = Trie()
        t.insert("the")
        t.insert("then")
        t.insert("there")
        t.insert("these")
        t.insert("them")
        assert len(t) == 5
        assert t.keys_with_prefix("the") == ["the", "them", "then", "there", "these"]

    def test_delete_all(self):
        t = Trie()
        words = ["a", "ab", "abc"]
        for w in words:
            t.insert(w)
        for w in words:
            assert t.delete(w)
        assert len(t) == 0
        assert t.all_keys() == []

    def test_reinsert_after_delete(self):
        t = Trie()
        t.insert("hello", 1)
        t.delete("hello")
        t.insert("hello", 2)
        assert t.get("hello") == 2


# =============================================================
# Patricia Trie
# =============================================================

class TestPatriciaTrie:
    def test_empty(self):
        t = PatriciaTrie()
        assert len(t) == 0
        assert not t.search("hello")

    def test_insert_search(self):
        t = PatriciaTrie()
        t.insert("hello")
        assert t.search("hello")
        assert not t.search("hell")
        assert not t.search("helloo")

    def test_insert_value(self):
        t = PatriciaTrie()
        t.insert("key", 42)
        assert t.get("key") == 42

    def test_size(self):
        t = PatriciaTrie()
        t.insert("abc")
        t.insert("abd")
        t.insert("xyz")
        assert len(t) == 3

    def test_compression(self):
        """Patricia trie should compress single-child chains."""
        t = PatriciaTrie()
        t.insert("abcdefgh")
        # Single key -> root has one edge with full string
        assert len(t.root.children) == 1
        edge, child = t.root.children['a']
        assert edge == "abcdefgh"

    def test_split_on_insert(self):
        t = PatriciaTrie()
        t.insert("abc")
        t.insert("abd")
        assert t.search("abc")
        assert t.search("abd")
        # "ab" edge splits into "ab" -> mid with 'c' and 'd' children
        edge, mid = t.root.children['a']
        assert edge == "ab"
        assert 'c' in mid.children
        assert 'd' in mid.children

    def test_delete(self):
        t = PatriciaTrie()
        t.insert("hello")
        t.insert("help")
        assert t.delete("hello")
        assert not t.search("hello")
        assert t.search("help")

    def test_delete_merges_nodes(self):
        t = PatriciaTrie()
        t.insert("abc")
        t.insert("abd")
        t.delete("abd")
        # Should merge back to single "abc" edge
        edge, child = t.root.children['a']
        assert edge == "abc"
        assert child.is_end

    def test_delete_nonexistent(self):
        t = PatriciaTrie()
        t.insert("hello")
        assert not t.delete("world")
        assert not t.delete("hell")

    def test_starts_with(self):
        t = PatriciaTrie()
        t.insert("application")
        t.insert("apple")
        assert t.starts_with("app")
        assert t.starts_with("apple")
        assert not t.starts_with("banana")

    def test_keys_with_prefix(self):
        t = PatriciaTrie()
        for w in ["apple", "app", "application", "banana"]:
            t.insert(w)
        result = t.keys_with_prefix("app")
        assert sorted(result) == ["app", "apple", "application"]

    def test_all_keys(self):
        t = PatriciaTrie()
        words = ["cat", "car", "card", "care"]
        for w in words:
            t.insert(w)
        assert t.all_keys() == sorted(words)

    def test_longest_prefix_of(self):
        t = PatriciaTrie()
        t.insert("a")
        t.insert("ab")
        t.insert("abc")
        assert t.longest_prefix_of("abcdef") == "abc"
        assert t.longest_prefix_of("abd") == "ab"
        assert t.longest_prefix_of("xyz") == ""

    def test_empty_string_key(self):
        t = PatriciaTrie()
        t.insert("")
        assert t.search("")
        assert t.delete("")
        assert not t.search("")

    def test_overwrite_value(self):
        t = PatriciaTrie()
        assert t.insert("key", 1) is None
        assert t.insert("key", 2) == 1
        assert t.get("key") == 2

    def test_prefix_is_key(self):
        t = PatriciaTrie()
        t.insert("abc")
        t.insert("ab")
        assert t.search("ab")
        assert t.search("abc")

    def test_many_keys(self):
        t = PatriciaTrie()
        words = [f"word{i:04d}" for i in range(100)]
        for w in words:
            t.insert(w)
        assert len(t) == 100
        for w in words:
            assert t.search(w)

    def test_delete_all_and_reinsert(self):
        t = PatriciaTrie()
        for w in ["abc", "abd", "xyz"]:
            t.insert(w)
        for w in ["abc", "abd", "xyz"]:
            assert t.delete(w)
        assert len(t) == 0
        t.insert("abc")
        assert t.search("abc")
        assert len(t) == 1


# =============================================================
# Ternary Search Tree
# =============================================================

class TestTernarySearchTree:
    def test_empty(self):
        t = TernarySearchTree()
        assert len(t) == 0
        assert not t.search("x")

    def test_insert_search(self):
        t = TernarySearchTree()
        t.insert("hello")
        assert t.search("hello")
        assert not t.search("hell")

    def test_insert_value(self):
        t = TernarySearchTree()
        t.insert("key", 42)
        assert t.get("key") == 42
        assert t.get("missing") is None
        assert t.get("missing", -1) == -1

    def test_size(self):
        t = TernarySearchTree()
        t.insert("a")
        t.insert("ab")
        t.insert("abc")
        assert len(t) == 3

    def test_delete(self):
        t = TernarySearchTree()
        t.insert("hello")
        t.insert("help")
        assert t.delete("hello")
        assert not t.search("hello")
        assert t.search("help")
        assert len(t) == 1

    def test_delete_nonexistent(self):
        t = TernarySearchTree()
        assert not t.delete("x")

    def test_starts_with(self):
        t = TernarySearchTree()
        t.insert("apple")
        t.insert("application")
        assert t.starts_with("app")
        assert not t.starts_with("banana")

    def test_keys_with_prefix(self):
        t = TernarySearchTree()
        for w in ["apple", "app", "application", "banana"]:
            t.insert(w)
        result = t.keys_with_prefix("app")
        assert sorted(result) == ["app", "apple", "application"]

    def test_all_keys(self):
        t = TernarySearchTree()
        words = ["the", "then", "there", "these", "them"]
        for w in words:
            t.insert(w)
        assert sorted(t.all_keys()) == sorted(words)

    def test_longest_prefix_of(self):
        t = TernarySearchTree()
        t.insert("a")
        t.insert("ab")
        t.insert("abc")
        assert t.longest_prefix_of("abcdef") == "abc"
        assert t.longest_prefix_of("abd") == "ab"
        assert t.longest_prefix_of("xyz") == ""

    def test_empty_string_ignored(self):
        t = TernarySearchTree()
        t.insert("")
        assert len(t) == 0

    def test_single_char(self):
        t = TernarySearchTree()
        t.insert("a")
        assert t.search("a")
        assert not t.search("b")

    def test_contains(self):
        t = TernarySearchTree()
        t.insert("hello")
        assert "hello" in t
        assert "world" not in t

    def test_starts_with_empty(self):
        t = TernarySearchTree()
        t.insert("a")
        assert t.starts_with("")

    def test_overwrite(self):
        t = TernarySearchTree()
        t.insert("key", 1)
        t.insert("key", 2)
        assert t.get("key") == 2
        assert len(t) == 1

    def test_many_keys_sorted(self):
        t = TernarySearchTree()
        words = ["delta", "alpha", "charlie", "bravo", "echo"]
        for w in words:
            t.insert(w)
        assert sorted(t.all_keys()) == sorted(words)


# =============================================================
# AutocompleteTrie
# =============================================================

class TestAutocompleteTrie:
    def test_empty(self):
        t = AutocompleteTrie()
        assert len(t) == 0
        assert t.suggest("a") == []

    def test_record_and_search(self):
        t = AutocompleteTrie()
        t.record("hello")
        assert t.search("hello")
        assert not t.search("world")

    def test_frequency(self):
        t = AutocompleteTrie()
        t.record("hello")
        t.record("hello")
        t.record("hello")
        assert t.frequency("hello") == 3
        assert t.frequency("world") == 0

    def test_suggest_by_frequency(self):
        t = AutocompleteTrie()
        t.record("apple", 10)
        t.record("application", 5)
        t.record("app", 20)
        result = t.suggest("app")
        assert result == ["app", "apple", "application"]

    def test_suggest_top_k(self):
        t = AutocompleteTrie()
        for i, w in enumerate(["a1", "a2", "a3", "a4", "a5", "a6"]):
            t.record(w, 10 - i)
        result = t.suggest("a", k=3)
        assert len(result) == 3
        assert result == ["a1", "a2", "a3"]

    def test_delete(self):
        t = AutocompleteTrie()
        t.record("hello")
        assert t.delete("hello")
        assert not t.search("hello")
        assert t.frequency("hello") == 0

    def test_all_keys_by_frequency(self):
        t = AutocompleteTrie()
        t.record("rare", 1)
        t.record("common", 100)
        t.record("medium", 50)
        keys = t.all_keys()
        assert keys == ["common", "medium", "rare"]

    def test_suggest_no_match(self):
        t = AutocompleteTrie()
        t.record("hello")
        assert t.suggest("xyz") == []

    def test_accumulate_frequency(self):
        t = AutocompleteTrie()
        t.record("x", 5)
        t.record("x", 3)
        assert t.frequency("x") == 8

    def test_suggest_alphabetical_tiebreak(self):
        t = AutocompleteTrie()
        t.record("ab", 5)
        t.record("aa", 5)
        t.record("ac", 5)
        result = t.suggest("a")
        assert result == ["aa", "ab", "ac"]


# =============================================================
# GeneralizedSuffixTrie
# =============================================================

class TestGeneralizedSuffixTrie:
    def test_empty(self):
        t = GeneralizedSuffixTrie()
        assert not t.contains_substring("a")

    def test_add_string(self):
        t = GeneralizedSuffixTrie()
        idx = t.add_string("hello")
        assert idx == 0

    def test_contains_substring(self):
        t = GeneralizedSuffixTrie()
        t.add_string("hello")
        assert t.contains_substring("hello")
        assert t.contains_substring("hell")
        assert t.contains_substring("ello")
        assert t.contains_substring("llo")
        assert t.contains_substring("lo")
        assert t.contains_substring("o")
        assert not t.contains_substring("world")

    def test_find_occurrences(self):
        t = GeneralizedSuffixTrie()
        t.add_string("banana")
        occ = t.find_occurrences("ana")
        assert sorted(occ) == [(0, 1), (0, 3)]

    def test_multiple_strings(self):
        t = GeneralizedSuffixTrie()
        t.add_string("abc")
        t.add_string("bcd")
        assert t.contains_substring("bc")
        occ = t.find_occurrences("bc")
        assert sorted(occ) == [(0, 1), (1, 0)]

    def test_longest_common_substring(self):
        t = GeneralizedSuffixTrie()
        t.add_string("abcdef")
        t.add_string("xbcdey")
        assert t.longest_common_substring() == "bcde"

    def test_lcs_no_common(self):
        t = GeneralizedSuffixTrie()
        t.add_string("abc")
        t.add_string("xyz")
        assert t.longest_common_substring() == ""

    def test_lcs_single_string(self):
        t = GeneralizedSuffixTrie()
        t.add_string("hello")
        assert t.longest_common_substring() == "hello"

    def test_count_strings_with(self):
        t = GeneralizedSuffixTrie()
        t.add_string("apple")
        t.add_string("pineapple")
        t.add_string("banana")
        assert t.count_strings_with("apple") == 2
        assert t.count_strings_with("banana") == 1
        assert t.count_strings_with("xyz") == 0

    def test_lcs_three_strings(self):
        t = GeneralizedSuffixTrie()
        t.add_string("xhelloworld")
        t.add_string("yhelloz")
        t.add_string("zhelloa")
        assert t.longest_common_substring() == "hello"


# =============================================================
# TrieMap
# =============================================================

class TestTrieMap:
    def test_empty(self):
        m = TrieMap()
        assert len(m) == 0
        assert "key" not in m

    def test_setitem_getitem(self):
        m = TrieMap()
        m["hello"] = 42
        assert m["hello"] == 42

    def test_contains(self):
        m = TrieMap()
        m["hello"] = 1
        assert "hello" in m
        assert "world" not in m

    def test_delitem(self):
        m = TrieMap()
        m["hello"] = 1
        del m["hello"]
        assert "hello" not in m

    def test_delitem_missing(self):
        m = TrieMap()
        with pytest.raises(KeyError):
            del m["missing"]

    def test_getitem_missing(self):
        m = TrieMap()
        with pytest.raises(KeyError):
            _ = m["missing"]

    def test_get(self):
        m = TrieMap()
        m["key"] = 42
        assert m.get("key") == 42
        assert m.get("missing") is None
        assert m.get("missing", -1) == -1

    def test_keys(self):
        m = TrieMap()
        m["c"] = 3
        m["a"] = 1
        m["b"] = 2
        assert m.keys() == ["a", "b", "c"]

    def test_values(self):
        m = TrieMap()
        m["a"] = 1
        m["b"] = 2
        m["c"] = 3
        assert m.values() == [1, 2, 3]

    def test_items(self):
        m = TrieMap()
        m["b"] = 2
        m["a"] = 1
        assert m.items() == [("a", 1), ("b", 2)]

    def test_prefix_keys(self):
        m = TrieMap()
        m["apple"] = 1
        m["app"] = 2
        m["banana"] = 3
        assert m.prefix_keys("app") == ["app", "apple"]

    def test_prefix_items(self):
        m = TrieMap()
        m["apple"] = 1
        m["app"] = 2
        m["banana"] = 3
        result = m.prefix_items("app")
        assert result == [("app", 2), ("apple", 1)]

    def test_longest_prefix_of(self):
        m = TrieMap()
        m["a"] = 1
        m["ab"] = 2
        m["abc"] = 3
        assert m.longest_prefix_of("abcdef") == "abc"

    def test_overwrite(self):
        m = TrieMap()
        m["key"] = 1
        m["key"] = 2
        assert m["key"] == 2
        assert len(m) == 1

    def test_many_items(self):
        m = TrieMap()
        for i in range(50):
            m[f"key{i:03d}"] = i
        assert len(m) == 50
        assert m["key025"] == 25


# =============================================================
# Cross-variant tests
# =============================================================

class TestCrossVariant:
    """Tests that exercise behavior across multiple trie types."""

    @pytest.fixture(params=["Trie", "PatriciaTrie", "TernarySearchTree"])
    def trie(self, request):
        cls_map = {
            "Trie": Trie,
            "PatriciaTrie": PatriciaTrie,
            "TernarySearchTree": TernarySearchTree,
        }
        return cls_map[request.param]()

    def test_insert_search_all(self, trie):
        words = ["apple", "app", "application", "banana", "band"]
        for w in words:
            trie.insert(w)
        for w in words:
            assert trie.search(w), f"{w} not found"

    def test_prefix_search_all(self, trie):
        trie.insert("test")
        trie.insert("testing")
        assert trie.starts_with("test")
        assert not trie.starts_with("xyz")

    def test_delete_all(self, trie):
        trie.insert("hello")
        trie.insert("help")
        assert trie.delete("hello")
        assert not trie.search("hello")
        assert trie.search("help")

    def test_longest_prefix_all(self, trie):
        trie.insert("a")
        trie.insert("ab")
        trie.insert("abc")
        assert trie.longest_prefix_of("abcdef") == "abc"

    def test_keys_with_prefix_all(self, trie):
        for w in ["cat", "car", "card", "care", "dog"]:
            trie.insert(w)
        result = trie.keys_with_prefix("car")
        assert sorted(result) == ["car", "card", "care"]

    def test_size_tracking_all(self, trie):
        trie.insert("a")
        trie.insert("b")
        trie.insert("c")
        assert len(trie) == 3
        trie.insert("a")  # duplicate
        assert len(trie) == 3

    def test_contains_all(self, trie):
        trie.insert("hello")
        assert "hello" in trie
        assert "world" not in trie


# =============================================================
# Edge cases
# =============================================================

class TestEdgeCases:
    def test_trie_unicode(self):
        t = Trie()
        t.insert("cafe")
        t.insert("cat")
        assert t.search("cafe")
        assert t.search("cat")

    def test_patricia_single_char_keys(self):
        t = PatriciaTrie()
        for ch in "abcde":
            t.insert(ch)
        assert len(t) == 5
        for ch in "abcde":
            assert t.search(ch)

    def test_trie_long_key(self):
        t = Trie()
        key = "a" * 1000
        t.insert(key)
        assert t.search(key)
        assert not t.search("a" * 999)

    def test_patricia_long_key(self):
        t = PatriciaTrie()
        key = "a" * 1000
        t.insert(key)
        assert t.search(key)

    def test_tst_long_key(self):
        t = TernarySearchTree()
        key = "a" * 500
        t.insert(key)
        assert t.search(key)

    def test_patricia_shared_prefix_many(self):
        t = PatriciaTrie()
        words = [f"prefix{i}" for i in range(20)]
        for w in words:
            t.insert(w)
        assert len(t) == 20
        assert sorted(t.all_keys()) == sorted(words)

    def test_trie_delete_reinsert_prefix(self):
        t = Trie()
        t.insert("abc")
        t.insert("ab")
        t.delete("abc")
        assert not t.search("abc")
        assert t.search("ab")
        t.insert("abc")
        assert t.search("abc")

    def test_triemap_empty_operations(self):
        m = TrieMap()
        assert m.keys() == []
        assert m.values() == []
        assert m.items() == []
        assert m.prefix_keys("x") == []
        assert m.prefix_items("x") == []

    def test_autocomplete_delete_nonexistent(self):
        t = AutocompleteTrie()
        assert not t.delete("missing")

    def test_suffix_trie_empty_string(self):
        t = GeneralizedSuffixTrie()
        t.add_string("")
        assert t.longest_common_substring() == ""

    def test_patricia_delete_leaf_then_merge(self):
        """Delete leaf child, verify parent merges with remaining sibling."""
        t = PatriciaTrie()
        t.insert("test")
        t.insert("team")
        t.insert("tea")
        assert t.delete("team")
        assert t.search("test")
        assert t.search("tea")
        assert len(t) == 2

    def test_trie_all_keys_empty(self):
        t = Trie()
        assert t.all_keys() == []

    def test_patricia_keys_with_prefix_empty(self):
        t = PatriciaTrie()
        assert t.keys_with_prefix("x") == []

    def test_trie_longest_prefix_empty_trie(self):
        t = Trie()
        assert t.longest_prefix_of("hello") == ""

    def test_tst_keys_with_prefix_empty(self):
        t = TernarySearchTree()
        assert t.keys_with_prefix("x") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
