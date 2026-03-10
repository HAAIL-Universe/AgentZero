"""Tests for C085: Trie / Radix Tree."""
import pytest
from trie import Trie, RadixTree, PersistentTrie, TernarySearchTree


# =============================================================================
# Trie Tests
# =============================================================================

class TestTrieBasic:
    def test_empty(self):
        t = Trie()
        assert len(t) == 0
        assert not t
        assert "hello" not in t

    def test_insert_search(self):
        t = Trie()
        t.insert("hello")
        assert "hello" in t
        assert "hell" not in t
        assert "helloo" not in t
        assert len(t) == 1

    def test_insert_multiple(self):
        t = Trie()
        for w in ["apple", "app", "application", "banana"]:
            t.insert(w)
        assert len(t) == 4
        assert "apple" in t
        assert "app" in t
        assert "application" in t
        assert "banana" in t
        assert "ban" not in t

    def test_insert_returns_old_value(self):
        t = Trie()
        assert t.insert("key", 1) is None
        assert t.insert("key", 2) == 1
        assert t.get("key") == 2
        assert len(t) == 1

    def test_delete(self):
        t = Trie()
        t.insert("hello")
        t.insert("hell")
        assert t.delete("hello")
        assert "hello" not in t
        assert "hell" in t
        assert len(t) == 1

    def test_delete_nonexistent(self):
        t = Trie()
        t.insert("hello")
        assert not t.delete("xyz")
        assert not t.delete("hell")
        assert len(t) == 1

    def test_delete_prunes_branch(self):
        t = Trie()
        t.insert("abc")
        assert t.delete("abc")
        assert len(t) == 0
        assert not t.root.children

    def test_delete_preserves_shared_prefix(self):
        t = Trie()
        t.insert("abc")
        t.insert("abd")
        t.delete("abc")
        assert "abd" in t
        assert "abc" not in t

    def test_get_default(self):
        t = Trie()
        assert t.get("missing") is None
        assert t.get("missing", 42) == 42
        t.insert("key", "val")
        assert t.get("key") == "val"

    def test_empty_string(self):
        t = Trie()
        t.insert("")
        assert "" in t
        assert len(t) == 1
        assert t.delete("")
        assert "" not in t


class TestTriePrefix:
    def test_starts_with(self):
        t = Trie()
        t.insert("hello")
        t.insert("help")
        assert t.starts_with("hel")
        assert t.starts_with("hello")
        assert not t.starts_with("hex")

    def test_keys_with_prefix(self):
        t = Trie()
        for w in ["apple", "app", "application", "banana", "band"]:
            t.insert(w)
        result = t.keys_with_prefix("app")
        assert sorted(result) == ["app", "apple", "application"]

    def test_keys_with_prefix_empty(self):
        t = Trie()
        t.insert("hello")
        assert t.keys_with_prefix("xyz") == []

    def test_count_with_prefix(self):
        t = Trie()
        for w in ["apple", "app", "application"]:
            t.insert(w)
        assert t.count_with_prefix("app") == 3
        assert t.count_with_prefix("apple") == 1
        assert t.count_with_prefix("xyz") == 0

    def test_autocomplete(self):
        t = Trie()
        for w in ["car", "card", "care", "careful", "cars"]:
            t.insert(w)
        result = t.autocomplete("car", limit=3)
        assert len(result) == 3
        assert result == ["car", "card", "care"]  # sorted order

    def test_longest_prefix_of(self):
        t = Trie()
        for w in ["a", "ab", "abc", "abcdef"]:
            t.insert(w)
        assert t.longest_prefix_of("abcde") == "abc"
        assert t.longest_prefix_of("abcdefgh") == "abcdef"
        assert t.longest_prefix_of("xyz") == ""
        assert t.longest_prefix_of("a") == "a"


class TestTrieWildcard:
    def test_wildcard_dot(self):
        t = Trie()
        for w in ["cat", "car", "cap", "bat", "bar"]:
            t.insert(w)
        assert sorted(t.wildcard_search("ca.")) == ["cap", "car", "cat"]
        assert sorted(t.wildcard_search(".at")) == ["bat", "cat"]
        assert sorted(t.wildcard_search("...")) == ["bar", "bat", "cap", "car", "cat"]

    def test_wildcard_exact(self):
        t = Trie()
        t.insert("abc")
        assert t.wildcard_search("abc") == ["abc"]
        assert t.wildcard_search("abd") == []

    def test_wildcard_all_dots(self):
        t = Trie()
        t.insert("ab")
        t.insert("cd")
        assert sorted(t.wildcard_search("..")) == ["ab", "cd"]


class TestTrieAll:
    def test_all_keys(self):
        t = Trie()
        words = ["zebra", "apple", "mango"]
        for w in words:
            t.insert(w)
        assert t.all_keys() == sorted(words)

    def test_items(self):
        t = Trie()
        t.insert("a", 1)
        t.insert("b", 2)
        assert t.items() == [("a", 1), ("b", 2)]

    def test_contains_and_bool(self):
        t = Trie()
        assert not t
        t.insert("x")
        assert t
        assert "x" in t


# =============================================================================
# Radix Tree Tests
# =============================================================================

class TestRadixBasic:
    def test_empty(self):
        r = RadixTree()
        assert len(r) == 0
        assert "hello" not in r

    def test_insert_search(self):
        r = RadixTree()
        r.insert("hello")
        assert "hello" in r
        assert "hell" not in r

    def test_insert_shared_prefix(self):
        r = RadixTree()
        r.insert("test")
        r.insert("testing")
        assert "test" in r
        assert "testing" in r
        assert len(r) == 2

    def test_insert_split(self):
        r = RadixTree()
        r.insert("romane")
        r.insert("romanus")
        r.insert("romulus")
        assert "romane" in r
        assert "romanus" in r
        assert "romulus" in r
        assert "rom" not in r
        assert len(r) == 3

    def test_insert_returns_old(self):
        r = RadixTree()
        assert r.insert("key", 1) is None
        assert r.insert("key", 2) == 1
        assert r.get("key") == 2
        assert len(r) == 1

    def test_delete(self):
        r = RadixTree()
        r.insert("test")
        r.insert("testing")
        assert r.delete("test")
        assert "test" not in r
        assert "testing" in r
        assert len(r) == 1

    def test_delete_merges(self):
        r = RadixTree()
        r.insert("abc")
        r.insert("abd")
        r.delete("abc")
        assert "abd" in r
        assert "abc" not in r

    def test_delete_nonexistent(self):
        r = RadixTree()
        r.insert("hello")
        assert not r.delete("xyz")
        assert not r.delete("hell")

    def test_get_default(self):
        r = RadixTree()
        assert r.get("x") is None
        assert r.get("x", 99) == 99

    def test_many_insertions(self):
        r = RadixTree()
        words = ["the", "their", "there", "they", "them", "then", "these"]
        for w in words:
            r.insert(w)
        assert len(r) == len(words)
        for w in words:
            assert w in r

    def test_prefix_is_key(self):
        r = RadixTree()
        r.insert("abc")
        r.insert("ab")
        r.insert("a")
        assert all(w in r for w in ["a", "ab", "abc"])
        assert len(r) == 3


class TestRadixPrefix:
    def test_starts_with(self):
        r = RadixTree()
        r.insert("hello")
        r.insert("help")
        assert r.starts_with("hel")
        assert r.starts_with("hello")
        assert not r.starts_with("hex")

    def test_starts_with_partial_edge(self):
        r = RadixTree()
        r.insert("testing")
        assert r.starts_with("test")
        assert r.starts_with("t")
        assert not r.starts_with("testx")

    def test_keys_with_prefix(self):
        r = RadixTree()
        for w in ["apple", "app", "application", "banana"]:
            r.insert(w)
        result = r.keys_with_prefix("app")
        assert sorted(result) == ["app", "apple", "application"]

    def test_keys_with_prefix_partial_edge(self):
        r = RadixTree()
        r.insert("testing")
        r.insert("tested")
        result = r.keys_with_prefix("test")
        assert sorted(result) == ["tested", "testing"]

    def test_longest_prefix_of(self):
        r = RadixTree()
        for w in ["a", "ab", "abc", "abcdef"]:
            r.insert(w)
        assert r.longest_prefix_of("abcde") == "abc"
        assert r.longest_prefix_of("abcdefgh") == "abcdef"
        assert r.longest_prefix_of("xyz") == ""


class TestRadixAll:
    def test_all_keys(self):
        r = RadixTree()
        words = ["zebra", "apple", "mango"]
        for w in words:
            r.insert(w)
        assert r.all_keys() == sorted(words)

    def test_items(self):
        r = RadixTree()
        r.insert("a", 1)
        r.insert("b", 2)
        assert r.items() == [("a", 1), ("b", 2)]

    def test_delete_all(self):
        r = RadixTree()
        words = ["abc", "abd", "xyz"]
        for w in words:
            r.insert(w)
        for w in words:
            assert r.delete(w)
        assert len(r) == 0

    def test_insert_empty_string(self):
        r = RadixTree()
        r.insert("", 42)
        assert r.search("")
        assert r.get("") == 42
        assert len(r) == 1


# =============================================================================
# Persistent Trie Tests
# =============================================================================

class TestPersistentTrieBasic:
    def test_empty(self):
        t = PersistentTrie()
        assert len(t) == 0
        assert "hello" not in t

    def test_insert_returns_new(self):
        t1 = PersistentTrie()
        t2 = t1.insert("hello")
        assert "hello" not in t1
        assert "hello" in t2
        assert len(t1) == 0
        assert len(t2) == 1

    def test_multiple_versions(self):
        t0 = PersistentTrie()
        t1 = t0.insert("a")
        t2 = t1.insert("b")
        t3 = t2.insert("c")
        assert len(t0) == 0
        assert len(t1) == 1
        assert len(t2) == 2
        assert len(t3) == 3
        assert "a" in t1 and "b" not in t1
        assert "a" in t2 and "b" in t2

    def test_delete_returns_new(self):
        t1 = PersistentTrie()
        t2 = t1.insert("hello")
        t3 = t2.delete("hello")
        assert "hello" in t2
        assert "hello" not in t3
        assert len(t2) == 1
        assert len(t3) == 0

    def test_delete_nonexistent(self):
        t = PersistentTrie().insert("a")
        t2 = t.delete("b")
        assert t is t2  # same object returned

    def test_branching_history(self):
        base = PersistentTrie().insert("a").insert("b")
        branch1 = base.insert("c")
        branch2 = base.insert("d")
        assert "c" in branch1 and "d" not in branch1
        assert "d" in branch2 and "c" not in branch2
        assert "a" in branch1 and "a" in branch2

    def test_get(self):
        t = PersistentTrie().insert("key", 42)
        assert t.get("key") == 42
        assert t.get("missing") is None

    def test_overwrite(self):
        t1 = PersistentTrie().insert("k", 1)
        t2 = t1.insert("k", 2)
        assert t1.get("k") == 1
        assert t2.get("k") == 2
        assert len(t1) == 1
        assert len(t2) == 1

    def test_keys_with_prefix(self):
        t = PersistentTrie()
        for w in ["apple", "app", "banana"]:
            t = t.insert(w)
        assert sorted(t.keys_with_prefix("app")) == ["app", "apple"]

    def test_all_keys(self):
        t = PersistentTrie()
        for w in ["c", "a", "b"]:
            t = t.insert(w)
        assert t.all_keys() == ["a", "b", "c"]


# =============================================================================
# Ternary Search Tree Tests
# =============================================================================

class TestTSTBasic:
    def test_empty(self):
        t = TernarySearchTree()
        assert len(t) == 0
        assert "hello" not in t

    def test_insert_search(self):
        t = TernarySearchTree()
        t.insert("hello")
        assert "hello" in t
        assert "hell" not in t
        assert len(t) == 1

    def test_insert_multiple(self):
        t = TernarySearchTree()
        words = ["cat", "car", "card", "care", "bat"]
        for w in words:
            t.insert(w)
        assert len(t) == 5
        for w in words:
            assert w in t

    def test_insert_returns_old(self):
        t = TernarySearchTree()
        assert t.insert("key", 1) is None
        assert t.insert("key", 2) == 1
        assert t.get("key") == 2
        assert len(t) == 1

    def test_delete(self):
        t = TernarySearchTree()
        t.insert("hello")
        assert t.delete("hello")
        assert "hello" not in t
        assert len(t) == 0

    def test_delete_nonexistent(self):
        t = TernarySearchTree()
        t.insert("hello")
        assert not t.delete("xyz")
        assert not t.delete("")

    def test_get_default(self):
        t = TernarySearchTree()
        assert t.get("x") is None
        assert t.get("x", 99) == 99

    def test_empty_string_rejected(self):
        t = TernarySearchTree()
        t.insert("")
        assert "" not in t
        assert len(t) == 0


class TestTSTPrefix:
    def test_starts_with(self):
        t = TernarySearchTree()
        t.insert("hello")
        t.insert("help")
        assert t.starts_with("hel")
        assert t.starts_with("hello")
        assert not t.starts_with("hex")

    def test_starts_with_empty(self):
        t = TernarySearchTree()
        t.insert("a")
        assert t.starts_with("")

    def test_keys_with_prefix(self):
        t = TernarySearchTree()
        for w in ["apple", "app", "application", "banana"]:
            t.insert(w)
        result = t.keys_with_prefix("app")
        assert sorted(result) == ["app", "apple", "application"]

    def test_keys_with_prefix_empty(self):
        t = TernarySearchTree()
        for w in ["a", "b", "c"]:
            t.insert(w)
        assert sorted(t.keys_with_prefix("")) == ["a", "b", "c"]

    def test_longest_prefix_of(self):
        t = TernarySearchTree()
        for w in ["a", "ab", "abc"]:
            t.insert(w)
        assert t.longest_prefix_of("abcd") == "abc"
        assert t.longest_prefix_of("ab") == "ab"
        assert t.longest_prefix_of("xyz") == ""
        assert t.longest_prefix_of("") == ""


class TestTSTAll:
    def test_all_keys(self):
        t = TernarySearchTree()
        words = ["zebra", "apple", "mango"]
        for w in words:
            t.insert(w)
        assert sorted(t.all_keys()) == sorted(words)

    def test_near_search(self):
        t = TernarySearchTree()
        for w in ["cat", "car", "cap", "bat", "bar"]:
            t.insert(w)
        # Hamming distance 1 from "cat"
        result = sorted(t.near_search("cat", 1))
        assert "cat" in result  # distance 0
        assert "car" in result  # distance 1
        assert "cap" in result  # distance 1
        assert "bat" in result  # distance 1

    def test_near_search_exact(self):
        t = TernarySearchTree()
        t.insert("abc")
        t.insert("xyz")
        assert t.near_search("abc", 0) == ["abc"]

    def test_near_search_high_distance(self):
        t = TernarySearchTree()
        for w in ["aaa", "bbb", "ccc"]:
            t.insert(w)
        result = t.near_search("aaa", 3)
        assert sorted(result) == ["aaa", "bbb", "ccc"]


# =============================================================================
# Cross-variant consistency tests
# =============================================================================

class TestCrossVariant:
    """Ensure all four variants agree on basic operations."""

    WORDS = ["the", "their", "there", "they", "them", "then",
             "these", "a", "an", "and", "are", "as", "at"]

    def _make_all(self):
        t = Trie()
        r = RadixTree()
        p = PersistentTrie()
        s = TernarySearchTree()
        for w in self.WORDS:
            t.insert(w)
            r.insert(w)
            p = p.insert(w)
            s.insert(w)
        return t, r, p, s

    def test_sizes_agree(self):
        t, r, p, s = self._make_all()
        assert len(t) == len(r) == len(p) == len(s) == len(self.WORDS)

    def test_search_agrees(self):
        t, r, p, s = self._make_all()
        for w in self.WORDS:
            assert w in t
            assert w in r
            assert w in p
            assert w in s
        for w in ["th", "the!", "xyz", ""]:
            assert (w in t) == (w in r) == (w in p)

    def test_prefix_keys_agree(self):
        t, r, p, s = self._make_all()
        for prefix in ["th", "a", "the", "z"]:
            tk = sorted(t.keys_with_prefix(prefix))
            rk = sorted(r.keys_with_prefix(prefix))
            pk = sorted(p.keys_with_prefix(prefix))
            sk = sorted(s.keys_with_prefix(prefix))
            assert tk == rk == pk == sk, f"Mismatch for prefix '{prefix}'"

    def test_all_keys_agree(self):
        t, r, p, s = self._make_all()
        tk = sorted(t.all_keys())
        rk = sorted(r.all_keys())
        pk = sorted(p.all_keys())
        sk = sorted(s.all_keys())
        assert tk == rk == pk == sk

    def test_delete_agrees(self):
        t, r, _, s = self._make_all()
        for w in ["the", "they", "a"]:
            t.delete(w)
            r.delete(w)
            s.delete(w)
        remaining = [w for w in self.WORDS if w not in ["the", "they", "a"]]
        assert len(t) == len(r) == len(s) == len(remaining)
        for w in remaining:
            assert w in t and w in r and w in s


# =============================================================================
# Edge cases and stress tests
# =============================================================================

class TestEdgeCases:
    def test_single_char_keys(self):
        t = Trie()
        for ch in "abcdefghij":
            t.insert(ch)
        assert len(t) == 10
        assert t.all_keys() == list("abcdefghij")

    def test_long_key(self):
        t = Trie()
        long_key = "a" * 1000
        t.insert(long_key)
        assert long_key in t
        assert t.delete(long_key)

    def test_unicode_keys(self):
        t = Trie()
        t.insert("cafe")
        t.insert("caff")
        assert "cafe" in t
        assert "caff" in t

    def test_radix_progressive_split(self):
        r = RadixTree()
        r.insert("abcdefghij")
        r.insert("abcde12345")
        r.insert("abcxy")
        r.insert("ab000")
        assert len(r) == 4
        for k in ["abcdefghij", "abcde12345", "abcxy", "ab000"]:
            assert k in r

    def test_trie_overwrite_value(self):
        t = Trie()
        t.insert("key", "v1")
        t.insert("key", "v2")
        assert t.get("key") == "v2"
        assert len(t) == 1

    def test_persistent_many_versions(self):
        t = PersistentTrie()
        versions = [t]
        for i in range(50):
            t = t.insert(f"key{i}")
            versions.append(t)
        assert len(versions[-1]) == 50
        assert len(versions[1]) == 1
        assert len(versions[25]) == 25

    def test_radix_delete_all_and_reinsert(self):
        r = RadixTree()
        words = ["abc", "abd", "xyz", "xy"]
        for w in words:
            r.insert(w)
        for w in words:
            r.delete(w)
        assert len(r) == 0
        for w in words:
            r.insert(w)
        assert len(r) == 4

    def test_tst_sorted_insertion(self):
        """TST degenerates with sorted input but should still work."""
        t = TernarySearchTree()
        words = [f"key{i:04d}" for i in range(100)]
        for w in words:
            t.insert(w)
        assert len(t) == 100
        for w in words:
            assert w in t

    def test_trie_autocomplete_limit(self):
        t = Trie()
        for i in range(100):
            t.insert(f"prefix_{i:03d}")
        result = t.autocomplete("prefix_", limit=5)
        assert len(result) == 5

    def test_radix_longest_prefix_empty_trie(self):
        r = RadixTree()
        assert r.longest_prefix_of("anything") == ""

    def test_tst_longest_prefix_single(self):
        t = TernarySearchTree()
        t.insert("a")
        assert t.longest_prefix_of("abc") == "a"

    def test_trie_wildcard_no_match(self):
        t = Trie()
        t.insert("abc")
        assert t.wildcard_search("ab") == []
        assert t.wildcard_search("abcd") == []

    def test_radix_empty_string_and_regular(self):
        r = RadixTree()
        r.insert("", "empty")
        r.insert("hello", "world")
        assert r.get("") == "empty"
        assert r.get("hello") == "world"
        assert len(r) == 2


class TestStress:
    def test_trie_1000_words(self):
        t = Trie()
        words = [f"word_{i}" for i in range(1000)]
        for w in words:
            t.insert(w)
        assert len(t) == 1000
        for w in words:
            assert w in t
        for w in words[:500]:
            t.delete(w)
        assert len(t) == 500

    def test_radix_1000_words(self):
        r = RadixTree()
        words = [f"prefix_{i}" for i in range(1000)]
        for w in words:
            r.insert(w)
        assert len(r) == 1000
        for w in words:
            assert w in r

    def test_persistent_chain(self):
        t = PersistentTrie()
        for i in range(200):
            t = t.insert(f"k{i}")
        assert len(t) == 200

    def test_tst_1000_words(self):
        t = TernarySearchTree()
        words = [f"item_{i}" for i in range(1000)]
        for w in words:
            t.insert(w)
        assert len(t) == 1000
        for w in words:
            assert w in t
