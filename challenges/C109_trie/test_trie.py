"""Tests for C109: Trie / Patricia Tree / Ternary Search Tree."""
import pytest
from trie import Trie, PatriciaTrie, TernarySearchTree, AutocompleteTrie, IPRoutingTrie


# ===================================================================
# 1. Standard Trie
# ===================================================================

class TestTrieBasic:
    def test_empty_trie(self):
        t = Trie()
        assert len(t) == 0
        assert not t.search("anything")

    def test_insert_and_search(self):
        t = Trie()
        assert t.insert("hello") is True
        assert t.search("hello")
        assert not t.search("hell")
        assert not t.search("helloo")

    def test_insert_duplicate(self):
        t = Trie()
        assert t.insert("abc") is True
        assert t.insert("abc") is False
        assert len(t) == 1

    def test_insert_with_value(self):
        t = Trie()
        t.insert("key", 42)
        assert t.get("key") == 42
        assert t.get("missing") is None
        assert t.get("missing", -1) == -1

    def test_contains(self):
        t = Trie()
        t.insert("foo")
        assert "foo" in t
        assert "bar" not in t

    def test_delete(self):
        t = Trie()
        t.insert("hello")
        t.insert("help")
        assert t.delete("hello") is True
        assert not t.search("hello")
        assert t.search("help")
        assert len(t) == 1

    def test_delete_nonexistent(self):
        t = Trie()
        assert t.delete("ghost") is False

    def test_delete_prefix_of_another(self):
        t = Trie()
        t.insert("abc")
        t.insert("abcdef")
        t.delete("abc")
        assert not t.search("abc")
        assert t.search("abcdef")

    def test_delete_leaves_prefix(self):
        t = Trie()
        t.insert("abc")
        t.insert("abcdef")
        t.delete("abcdef")
        assert t.search("abc")
        assert not t.search("abcdef")

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
        assert result == ["app", "apple", "application"]

    def test_keys_with_prefix_empty(self):
        t = Trie()
        t.insert("hello")
        assert t.keys_with_prefix("xyz") == []

    def test_count_with_prefix(self):
        t = Trie()
        for w in ["cat", "car", "card", "care", "dog"]:
            t.insert(w)
        assert t.count_with_prefix("car") == 3
        assert t.count_with_prefix("dog") == 1
        assert t.count_with_prefix("xyz") == 0

    def test_all_keys_sorted(self):
        t = Trie()
        words = ["banana", "apple", "cherry", "avocado"]
        for w in words:
            t.insert(w)
        assert t.all_keys() == sorted(words)

    def test_longest_prefix_of(self):
        t = Trie()
        for w in ["a", "ab", "abc"]:
            t.insert(w)
        assert t.longest_prefix_of("abcdef") == "abc"
        assert t.longest_prefix_of("ab") == "ab"
        assert t.longest_prefix_of("xyz") == ""
        assert t.longest_prefix_of("a") == "a"

    def test_wildcard_match(self):
        t = Trie()
        for w in ["cat", "car", "bat", "bar", "cap"]:
            t.insert(w)
        assert t.wildcard_match("c.t") == ["cat"]
        assert t.wildcard_match("..t") == ["bat", "cat"]
        assert t.wildcard_match("...") == ["bar", "bat", "cap", "car", "cat"]
        assert t.wildcard_match("b.r") == ["bar"]

    def test_wildcard_no_match(self):
        t = Trie()
        t.insert("abc")
        assert t.wildcard_match("a..d") == []
        assert t.wildcard_match("..") == []

    def test_empty_string_key(self):
        t = Trie()
        t.insert("")
        assert t.search("")
        assert len(t) == 1
        t.delete("")
        assert not t.search("")

    def test_single_char_keys(self):
        t = Trie()
        for ch in "abcde":
            t.insert(ch)
        assert len(t) == 5
        assert t.all_keys() == ["a", "b", "c", "d", "e"]

    def test_long_key(self):
        t = Trie()
        long_key = "a" * 1000
        t.insert(long_key)
        assert t.search(long_key)
        assert not t.search("a" * 999)

    def test_unicode_keys(self):
        t = Trie()
        t.insert("cafe")
        t.insert("hello")
        assert t.search("cafe")
        assert t.search("hello")

    def test_many_keys(self):
        t = Trie()
        words = [f"word{i:04d}" for i in range(500)]
        for w in words:
            t.insert(w)
        assert len(t) == 500
        for w in words:
            assert t.search(w)

    def test_value_update(self):
        t = Trie()
        t.insert("key", 1)
        assert t.get("key") == 1
        t.insert("key", 2)
        assert t.get("key") == 2
        assert len(t) == 1

    def test_delete_all(self):
        t = Trie()
        words = ["a", "ab", "abc", "abcd"]
        for w in words:
            t.insert(w)
        for w in words:
            t.delete(w)
        assert len(t) == 0
        assert t.all_keys() == []


# ===================================================================
# 2. Patricia Trie (Radix Tree)
# ===================================================================

class TestPatriciaTrieBasic:
    def test_empty(self):
        pt = PatriciaTrie()
        assert len(pt) == 0
        assert not pt.search("x")

    def test_insert_search(self):
        pt = PatriciaTrie()
        assert pt.insert("hello") is True
        assert pt.search("hello")
        assert not pt.search("hell")

    def test_insert_duplicate(self):
        pt = PatriciaTrie()
        pt.insert("abc")
        assert pt.insert("abc") is False
        assert len(pt) == 1

    def test_insert_with_value(self):
        pt = PatriciaTrie()
        pt.insert("key", 99)
        assert pt.get("key") == 99
        assert pt.get("nope") is None

    def test_contains(self):
        pt = PatriciaTrie()
        pt.insert("foo")
        assert "foo" in pt
        assert "bar" not in pt

    def test_split_on_insert(self):
        pt = PatriciaTrie()
        pt.insert("test")
        pt.insert("team")
        assert pt.search("test")
        assert pt.search("team")
        assert not pt.search("te")

    def test_insert_prefix_of_existing(self):
        pt = PatriciaTrie()
        pt.insert("testing")
        pt.insert("test")
        assert pt.search("test")
        assert pt.search("testing")

    def test_insert_extension_of_existing(self):
        pt = PatriciaTrie()
        pt.insert("test")
        pt.insert("testing")
        assert pt.search("test")
        assert pt.search("testing")

    def test_delete(self):
        pt = PatriciaTrie()
        pt.insert("hello")
        pt.insert("help")
        assert pt.delete("hello") is True
        assert not pt.search("hello")
        assert pt.search("help")

    def test_delete_merges_nodes(self):
        pt = PatriciaTrie()
        pt.insert("test")
        pt.insert("testing")
        pt.insert("team")
        pt.delete("test")
        assert not pt.search("test")
        assert pt.search("testing")
        assert pt.search("team")

    def test_delete_nonexistent(self):
        pt = PatriciaTrie()
        assert pt.delete("ghost") is False

    def test_starts_with(self):
        pt = PatriciaTrie()
        pt.insert("apple")
        pt.insert("application")
        assert pt.starts_with("app")
        assert pt.starts_with("apple")
        assert not pt.starts_with("banana")

    def test_starts_with_partial_label(self):
        pt = PatriciaTrie()
        pt.insert("testing")
        assert pt.starts_with("te")
        assert pt.starts_with("test")
        assert pt.starts_with("testing")
        assert not pt.starts_with("testingg")

    def test_keys_with_prefix(self):
        pt = PatriciaTrie()
        for w in ["apple", "app", "application", "banana", "band"]:
            pt.insert(w)
        result = pt.keys_with_prefix("app")
        assert result == ["app", "apple", "application"]

    def test_keys_with_prefix_partial(self):
        pt = PatriciaTrie()
        pt.insert("testing")
        pt.insert("temperature")
        result = pt.keys_with_prefix("te")
        assert result == ["temperature", "testing"]

    def test_all_keys_sorted(self):
        pt = PatriciaTrie()
        words = ["banana", "apple", "cherry", "avocado"]
        for w in words:
            pt.insert(w)
        assert pt.all_keys() == sorted(words)

    def test_longest_prefix_of(self):
        pt = PatriciaTrie()
        for w in ["a", "ab", "abc"]:
            pt.insert(w)
        assert pt.longest_prefix_of("abcdef") == "abc"
        assert pt.longest_prefix_of("ab") == "ab"
        assert pt.longest_prefix_of("xyz") == ""

    def test_empty_string(self):
        pt = PatriciaTrie()
        pt.insert("")
        assert pt.search("")
        pt.delete("")
        assert not pt.search("")

    def test_many_keys(self):
        pt = PatriciaTrie()
        words = [f"word{i:04d}" for i in range(500)]
        for w in words:
            pt.insert(w)
        assert len(pt) == 500
        for w in words:
            assert pt.search(w), f"Missing: {w}"

    def test_shared_prefix_group(self):
        pt = PatriciaTrie()
        words = ["romane", "romanus", "romulus", "rubens", "ruber", "rubicon", "rubicundus"]
        for w in words:
            pt.insert(w)
        assert len(pt) == 7
        assert pt.all_keys() == sorted(words)

    def test_delete_all_reverse(self):
        pt = PatriciaTrie()
        words = ["abc", "abcdef", "ab", "a"]
        for w in words:
            pt.insert(w)
        for w in reversed(words):
            pt.delete(w)
        assert len(pt) == 0


# ===================================================================
# 3. Ternary Search Tree
# ===================================================================

class TestTSTBasic:
    def test_empty(self):
        tst = TernarySearchTree()
        assert len(tst) == 0
        assert not tst.search("x")

    def test_insert_search(self):
        tst = TernarySearchTree()
        assert tst.insert("hello") is True
        assert tst.search("hello")
        assert not tst.search("hell")

    def test_insert_duplicate(self):
        tst = TernarySearchTree()
        tst.insert("abc")
        assert tst.insert("abc") is False
        assert len(tst) == 1

    def test_insert_with_value(self):
        tst = TernarySearchTree()
        tst.insert("key", 42)
        assert tst.get("key") == 42
        assert tst.get("nope") is None

    def test_contains(self):
        tst = TernarySearchTree()
        tst.insert("foo")
        assert "foo" in tst
        assert "bar" not in tst

    def test_empty_key_rejected(self):
        tst = TernarySearchTree()
        assert tst.insert("") is False
        assert len(tst) == 0

    def test_delete(self):
        tst = TernarySearchTree()
        tst.insert("hello")
        tst.insert("help")
        assert tst.delete("hello") is True
        assert not tst.search("hello")
        assert tst.search("help")

    def test_delete_nonexistent(self):
        tst = TernarySearchTree()
        assert tst.delete("ghost") is False

    def test_keys_with_prefix(self):
        tst = TernarySearchTree()
        for w in ["apple", "app", "application", "banana", "band"]:
            tst.insert(w)
        result = tst.keys_with_prefix("app")
        assert sorted(result) == ["app", "apple", "application"]

    def test_all_keys_sorted(self):
        tst = TernarySearchTree()
        words = ["banana", "apple", "cherry", "avocado"]
        for w in words:
            tst.insert(w)
        assert tst.all_keys() == sorted(words)

    def test_near_search_exact(self):
        tst = TernarySearchTree()
        for w in ["cat", "car", "bat", "bar"]:
            tst.insert(w)
        result = tst.near_search("cat", 0)
        assert result == ["cat"]

    def test_near_search_distance_1(self):
        tst = TernarySearchTree()
        for w in ["cat", "car", "bat", "bar", "cap", "cup"]:
            tst.insert(w)
        result = tst.near_search("cat", 1)
        assert "cat" in result  # exact match
        assert "bat" in result  # substitution
        assert "car" in result  # substitution
        assert "cap" in result  # substitution

    def test_near_search_distance_2(self):
        tst = TernarySearchTree()
        words = ["hello", "hallo", "hullo", "hero", "help"]
        for w in words:
            tst.insert(w)
        result = tst.near_search("hello", 2)
        assert "hello" in result
        assert "hallo" in result
        assert "hullo" in result

    def test_starts_with(self):
        tst = TernarySearchTree()
        tst.insert("testing")
        tst.insert("team")
        assert tst.starts_with("te")
        assert not tst.starts_with("xyz")

    def test_many_keys(self):
        tst = TernarySearchTree()
        words = [f"word{i:04d}" for i in range(200)]
        for w in words:
            tst.insert(w)
        assert len(tst) == 200
        for w in words:
            assert tst.search(w)

    def test_delete_all(self):
        tst = TernarySearchTree()
        words = ["abc", "abd", "xyz", "ab"]
        for w in words:
            tst.insert(w)
        for w in words:
            tst.delete(w)
        assert len(tst) == 0

    def test_single_char(self):
        tst = TernarySearchTree()
        tst.insert("a")
        assert tst.search("a")
        tst.delete("a")
        assert not tst.search("a")


# ===================================================================
# 4. AutocompleteTrie
# ===================================================================

class TestAutocompleteTrie:
    def test_empty(self):
        at = AutocompleteTrie()
        assert len(at) == 0
        assert at.autocomplete("x") == []

    def test_insert_and_autocomplete(self):
        at = AutocompleteTrie()
        at.insert("hello", 10)
        at.insert("help", 20)
        at.insert("hero", 5)
        result = at.autocomplete("hel")
        words = [w for w, f in result]
        assert "help" in words
        assert "hello" in words
        assert "hero" not in words

    def test_frequency_ordering(self):
        at = AutocompleteTrie()
        at.insert("python", 100)
        at.insert("pytorch", 50)
        at.insert("pycharm", 30)
        result = at.autocomplete("py")
        assert result[0] == ("python", 100)
        assert result[1] == ("pytorch", 50)
        assert result[2] == ("pycharm", 30)

    def test_frequency_accumulates(self):
        at = AutocompleteTrie()
        at.insert("word", 5)
        at.insert("word", 3)
        assert at.get_frequency("word") == 8

    def test_top_k(self):
        at = AutocompleteTrie()
        for i in range(20):
            at.insert(f"item{i:02d}", 20 - i)
        result = at.autocomplete("item", k=5)
        assert len(result) == 5
        assert result[0][0] == "item00"

    def test_autocomplete_exact_prefix(self):
        at = AutocompleteTrie()
        at.insert("app", 10)
        at.insert("apple", 20)
        result = at.autocomplete("app")
        words = [w for w, f in result]
        assert "app" in words
        assert "apple" in words

    def test_no_match(self):
        at = AutocompleteTrie()
        at.insert("hello", 1)
        assert at.autocomplete("xyz") == []

    def test_get_frequency(self):
        at = AutocompleteTrie()
        at.insert("test", 42)
        assert at.get_frequency("test") == 42
        assert at.get_frequency("nope") == 0

    def test_fuzzy_autocomplete_substitution(self):
        at = AutocompleteTrie()
        at.insert("hello", 10)
        at.insert("world", 5)
        result = at.fuzzy_autocomplete("hallo", k=10, max_edits=1)
        words = [w for w, f in result]
        assert "hello" in words

    def test_fuzzy_autocomplete_deletion(self):
        at = AutocompleteTrie()
        at.insert("test", 10)
        at.insert("toast", 5)
        result = at.fuzzy_autocomplete("tst", k=10, max_edits=1)
        words = [w for w, f in result]
        assert "test" in words

    def test_fuzzy_autocomplete_exact(self):
        at = AutocompleteTrie()
        at.insert("exact", 10)
        result = at.fuzzy_autocomplete("exact", k=10, max_edits=1)
        words = [w for w, f in result]
        assert "exact" in words

    def test_size(self):
        at = AutocompleteTrie()
        at.insert("a", 1)
        at.insert("b", 1)
        at.insert("a", 1)  # duplicate
        assert len(at) == 2


# ===================================================================
# 5. IP Routing Trie
# ===================================================================

class TestIPRoutingTrie:
    def test_empty(self):
        rt = IPRoutingTrie()
        assert len(rt) == 0
        assert rt.lookup("192.168.1.1") is None

    def test_insert_and_lookup(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "gateway-a")
        result = rt.lookup("10.1.2.3")
        assert result is not None
        assert result[0] == "gateway-a"
        assert result[1] == 8

    def test_longest_prefix_match(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "broad")
        rt.insert("10.1.0.0/16", "medium")
        rt.insert("10.1.1.0/24", "specific")
        result = rt.lookup("10.1.1.5")
        assert result[0] == "specific"
        result2 = rt.lookup("10.1.2.5")
        assert result2[0] == "medium"
        result3 = rt.lookup("10.2.0.1")
        assert result3[0] == "broad"

    def test_no_match(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "gw")
        assert rt.lookup("192.168.1.1") is None

    def test_default_route(self):
        rt = IPRoutingTrie()
        rt.insert("0.0.0.0/0", "default")
        rt.insert("10.0.0.0/8", "private")
        result = rt.lookup("8.8.8.8")
        assert result[0] == "default"
        result2 = rt.lookup("10.0.0.1")
        assert result2[0] == "private"

    def test_host_route(self):
        rt = IPRoutingTrie()
        rt.insert("192.168.1.100/32", "host")
        rt.insert("192.168.1.0/24", "subnet")
        result = rt.lookup("192.168.1.100")
        assert result[0] == "host"
        result2 = rt.lookup("192.168.1.99")
        assert result2[0] == "subnet"

    def test_delete(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "gw")
        assert rt.delete("10.0.0.0/8") is True
        assert rt.lookup("10.0.0.1") is None
        assert len(rt) == 0

    def test_delete_preserves_longer(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "broad")
        rt.insert("10.1.0.0/16", "specific")
        rt.delete("10.0.0.0/8")
        assert rt.lookup("10.1.0.1")[0] == "specific"
        assert rt.lookup("10.2.0.1") is None

    def test_delete_nonexistent(self):
        rt = IPRoutingTrie()
        assert rt.delete("10.0.0.0/8") is False

    def test_all_routes(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "a")
        rt.insert("192.168.0.0/16", "b")
        routes = rt.all_routes()
        assert len(routes) == 2
        cidrs = [r[0] for r in routes]
        assert "10.0.0.0/8" in cidrs
        assert "192.168.0.0/16" in cidrs

    def test_overlapping_prefixes(self):
        rt = IPRoutingTrie()
        rt.insert("172.16.0.0/12", "private-b")
        rt.insert("172.16.0.0/16", "specific")
        result = rt.lookup("172.16.0.1")
        assert result[0] == "specific"
        result2 = rt.lookup("172.17.0.1")
        assert result2[0] == "private-b"

    def test_size(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "a")
        rt.insert("10.0.0.0/8", "b")  # update
        assert len(rt) == 1
        rt.insert("10.1.0.0/16", "c")
        assert len(rt) == 2


# ===================================================================
# Cross-variant consistency tests
# ===================================================================

class TestCrossVariant:
    """Test that all trie variants agree on basic operations."""

    WORDS = ["apple", "app", "application", "banana", "band", "bandana", "cat", "car", "card"]

    def _make_tries(self):
        t = Trie()
        pt = PatriciaTrie()
        tst = TernarySearchTree()
        for w in self.WORDS:
            t.insert(w)
            pt.insert(w)
            tst.insert(w)
        return t, pt, tst

    def test_same_size(self):
        t, pt, tst = self._make_tries()
        assert len(t) == len(pt) == len(tst) == len(self.WORDS)

    def test_same_search_results(self):
        t, pt, tst = self._make_tries()
        for w in self.WORDS:
            assert t.search(w) and pt.search(w) and tst.search(w)
        for w in ["xyz", "ap", "ban", "cards", "ba"]:
            assert not t.search(w) and not pt.search(w) and not tst.search(w)

    def test_same_all_keys(self):
        t, pt, tst = self._make_tries()
        assert t.all_keys() == pt.all_keys() == sorted(tst.all_keys())

    def test_same_prefix_keys(self):
        t, pt, tst = self._make_tries()
        for prefix in ["app", "ban", "ca", ""]:
            t_keys = t.keys_with_prefix(prefix)
            pt_keys = pt.keys_with_prefix(prefix)
            tst_keys = sorted(tst.keys_with_prefix(prefix))
            assert t_keys == pt_keys == tst_keys, f"Mismatch on prefix '{prefix}'"

    def test_delete_consistency(self):
        t, pt, tst = self._make_tries()
        for w in ["app", "banana", "car"]:
            t.delete(w)
            pt.delete(w)
            tst.delete(w)
        assert t.all_keys() == pt.all_keys() == sorted(tst.all_keys())

    def test_longest_prefix(self):
        t = Trie()
        pt = PatriciaTrie()
        for w in ["a", "ab", "abc", "abcde"]:
            t.insert(w)
            pt.insert(w)
        for text in ["abcdef", "abcd", "ab", "xyz", "a"]:
            assert t.longest_prefix_of(text) == pt.longest_prefix_of(text), f"Mismatch on '{text}'"


# ===================================================================
# Edge cases and stress tests
# ===================================================================

class TestEdgeCases:
    def test_trie_overlapping_inserts_deletes(self):
        t = Trie()
        t.insert("a")
        t.insert("ab")
        t.insert("abc")
        t.insert("abcd")
        t.delete("ab")
        assert t.search("a")
        assert not t.search("ab")
        assert t.search("abc")
        assert t.search("abcd")
        assert len(t) == 3

    def test_patricia_many_splits(self):
        pt = PatriciaTrie()
        # Insert words that force many splits
        pt.insert("romane")
        pt.insert("romanus")
        pt.insert("romulus")
        pt.insert("rubens")
        pt.insert("ruber")
        pt.insert("rubicon")
        pt.insert("rubicundus")
        assert len(pt) == 7
        assert pt.search("romane")
        assert pt.search("rubicundus")
        assert not pt.search("rom")
        assert not pt.search("rub")

    def test_tst_balanced_insert(self):
        """Insert in sorted order -- TST degenerates but should still work."""
        tst = TernarySearchTree()
        words = sorted(["delta", "alpha", "echo", "bravo", "charlie"])
        for w in words:
            tst.insert(w)
        assert len(tst) == 5
        for w in words:
            assert tst.search(w)

    def test_autocomplete_empty_prefix(self):
        at = AutocompleteTrie()
        at.insert("a", 1)
        at.insert("b", 2)
        result = at.autocomplete("")
        assert len(result) == 2

    def test_ip_trie_boundary_addresses(self):
        rt = IPRoutingTrie()
        rt.insert("0.0.0.0/0", "default")
        rt.insert("255.255.255.255/32", "broadcast")
        assert rt.lookup("0.0.0.0")[0] == "default"
        assert rt.lookup("255.255.255.255")[0] == "broadcast"
        assert rt.lookup("128.0.0.0")[0] == "default"

    def test_trie_wildcard_all_dots(self):
        t = Trie()
        t.insert("abc")
        t.insert("xyz")
        result = t.wildcard_match("...")
        assert "abc" in result
        assert "xyz" in result

    def test_patricia_empty_all_keys(self):
        pt = PatriciaTrie()
        assert pt.all_keys() == []

    def test_tst_prefix_of_existing_not_found(self):
        tst = TernarySearchTree()
        tst.insert("testing")
        assert not tst.search("test")
        assert tst.search("testing")

    def test_patricia_single_char_words(self):
        pt = PatriciaTrie()
        for ch in "zyxwvutsrqponmlkjihgfedcba":
            pt.insert(ch)
        assert len(pt) == 26
        assert pt.all_keys() == list("abcdefghijklmnopqrstuvwxyz")

    def test_autocomplete_alphabetical_tiebreak(self):
        at = AutocompleteTrie()
        at.insert("beta", 10)
        at.insert("alpha", 10)
        at.insert("gamma", 10)
        result = at.autocomplete("")
        words = [w for w, f in result]
        assert words == ["alpha", "beta", "gamma"]

    def test_ip_routing_update_value(self):
        rt = IPRoutingTrie()
        rt.insert("10.0.0.0/8", "old")
        rt.insert("10.0.0.0/8", "new")
        assert rt.lookup("10.0.0.1")[0] == "new"
        assert len(rt) == 1


# ===================================================================
# Stress / Randomized tests
# ===================================================================

class TestStress:
    def test_trie_1000_words(self):
        import random
        random.seed(42)
        t = Trie()
        words = set()
        for _ in range(1000):
            length = random.randint(1, 20)
            word = "".join(random.choice("abcdefghij") for _ in range(length))
            words.add(word)
            t.insert(word)
        assert len(t) == len(words)
        for w in words:
            assert t.search(w)

    def test_patricia_1000_words(self):
        import random
        random.seed(42)
        pt = PatriciaTrie()
        words = set()
        for _ in range(1000):
            length = random.randint(1, 20)
            word = "".join(random.choice("abcdefghij") for _ in range(length))
            words.add(word)
            pt.insert(word)
        assert len(pt) == len(words)
        for w in words:
            assert pt.search(w), f"Missing: {w}"

    def test_tst_500_words(self):
        import random
        random.seed(123)
        tst = TernarySearchTree()
        words = set()
        for _ in range(500):
            length = random.randint(1, 15)
            word = "".join(random.choice("abcdefgh") for _ in range(length))
            words.add(word)
            tst.insert(word)
        assert len(tst) == len(words)
        for w in words:
            assert tst.search(w)

    def test_patricia_insert_delete_cycle(self):
        import random
        random.seed(77)
        pt = PatriciaTrie()
        words = []
        for _ in range(200):
            w = "".join(random.choice("abc") for _ in range(random.randint(1, 10)))
            pt.insert(w)
            words.append(w)
        random.shuffle(words)
        for w in words:
            if pt.search(w):
                pt.delete(w)
        assert len(pt) == 0

    def test_ip_routing_many_subnets(self):
        rt = IPRoutingTrie()
        # Insert /24 subnets for 10.0.0-255.0/24
        for i in range(256):
            rt.insert(f"10.0.{i}.0/24", f"subnet-{i}")
        rt.insert("10.0.0.0/8", "broad")
        # Specific /24 should win
        result = rt.lookup("10.0.42.100")
        assert result[0] == "subnet-42"
        assert result[1] == 24
        # Outside 10.0.x.0 subnets but in /8
        result2 = rt.lookup("10.1.0.1")
        assert result2[0] == "broad"
