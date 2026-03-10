"""
Tests for C077: Rope Data Structure
"""

import pytest
from rope import Rope, Leaf, Branch, RopeNode, MAX_LEAF, _EMPTY_LEAF


# ============================================================
# Construction
# ============================================================

class TestConstruction:
    def test_empty_rope(self):
        r = Rope.empty()
        assert r.length == 0
        assert str(r) == ""

    def test_from_string(self):
        r = Rope("hello")
        assert str(r) == "hello"
        assert r.length == 5

    def test_from_string_static(self):
        r = Rope.from_string("world")
        assert str(r) == "world"

    def test_from_empty_string(self):
        r = Rope("")
        assert r.length == 0
        assert str(r) == ""

    def test_from_long_string(self):
        s = "a" * 200
        r = Rope(s)
        assert str(r) == s
        assert r.length == 200

    def test_from_string_splits_into_leaves(self):
        s = "x" * (MAX_LEAF * 3)
        r = Rope(s)
        assert str(r) == s
        assert r.leaf_count() >= 3

    def test_constructor_type_error(self):
        with pytest.raises(TypeError):
            Rope(42)

    def test_repr_short(self):
        r = Rope("hello")
        assert "hello" in repr(r)

    def test_repr_long(self):
        r = Rope("a" * 100)
        assert "..." in repr(r)

    def test_bool_empty(self):
        assert not Rope.empty()

    def test_bool_nonempty(self):
        assert Rope("x")


# ============================================================
# Character Access
# ============================================================

class TestCharAt:
    def test_first_char(self):
        r = Rope("hello")
        assert r.char_at(0) == 'h'

    def test_last_char(self):
        r = Rope("hello")
        assert r.char_at(4) == 'o'

    def test_middle_char(self):
        r = Rope("hello")
        assert r.char_at(2) == 'l'

    def test_negative_index(self):
        r = Rope("hello")
        assert r.char_at(-1) == 'o'
        assert r.char_at(-5) == 'h'

    def test_index_out_of_range(self):
        r = Rope("hello")
        with pytest.raises(IndexError):
            r.char_at(5)

    def test_negative_out_of_range(self):
        r = Rope("hello")
        with pytest.raises(IndexError):
            r.char_at(-6)

    def test_getitem_single(self):
        r = Rope("abcde")
        assert r[0] == 'a'
        assert r[4] == 'e'
        assert r[-1] == 'e'

    def test_getitem_out_of_range(self):
        r = Rope("abc")
        with pytest.raises(IndexError):
            r[3]

    def test_char_at_across_leaves(self):
        s = "abcdefghij" * 20  # 200 chars, spans multiple leaves
        r = Rope(s)
        for i in range(len(s)):
            assert r.char_at(i) == s[i]


# ============================================================
# Concatenation
# ============================================================

class TestConcat:
    def test_concat_two_ropes(self):
        a = Rope("hello")
        b = Rope(" world")
        c = a.concat(b)
        assert str(c) == "hello world"

    def test_concat_with_string(self):
        r = Rope("hello")
        c = r.concat(" world")
        assert str(c) == "hello world"

    def test_concat_empty_left(self):
        a = Rope.empty()
        b = Rope("world")
        assert a.concat(b) is b

    def test_concat_empty_right(self):
        a = Rope("hello")
        b = Rope.empty()
        assert a.concat(b) is a

    def test_concat_preserves_original(self):
        a = Rope("hello")
        b = Rope(" world")
        _ = a.concat(b)
        assert str(a) == "hello"
        assert str(b) == " world"

    def test_concat_many(self):
        parts = [Rope(str(i)) for i in range(100)]
        result = Rope.empty()
        for p in parts:
            result = result.concat(p)
        assert str(result) == ''.join(str(i) for i in range(100))

    def test_add_operator(self):
        a = Rope("hello")
        b = Rope(" world")
        assert str(a + b) == "hello world"

    def test_add_string(self):
        r = Rope("hello")
        assert str(r + " world") == "hello world"

    def test_radd_string(self):
        r = Rope("world")
        assert str("hello " + r) == "hello world"

    def test_concat_type_error(self):
        r = Rope("hello")
        with pytest.raises(TypeError):
            r.concat(42)

    def test_concat_auto_rebalance(self):
        # Building a deeply unbalanced rope via repeated concat
        r = Rope("a")
        for _ in range(200):
            r = r.concat(Rope("b"))
        # Should auto-rebalance and still work
        assert r.length == 201
        assert r.char_at(0) == 'a'


# ============================================================
# Split
# ============================================================

class TestSplit:
    def test_split_middle(self):
        r = Rope("hello world")
        left, right = r.split(5)
        assert str(left) == "hello"
        assert str(right) == " world"

    def test_split_at_zero(self):
        r = Rope("hello")
        left, right = r.split(0)
        assert str(left) == ""
        assert str(right) == "hello"

    def test_split_at_end(self):
        r = Rope("hello")
        left, right = r.split(5)
        assert str(left) == "hello"
        assert str(right) == ""

    def test_split_at_one(self):
        r = Rope("hello")
        left, right = r.split(1)
        assert str(left) == "h"
        assert str(right) == "ello"

    def test_split_preserves_original(self):
        r = Rope("hello world")
        _ = r.split(5)
        assert str(r) == "hello world"

    def test_split_long_string(self):
        s = "the quick brown fox jumps over the lazy dog"
        r = Rope(s)
        for i in range(len(s) + 1):
            left, right = r.split(i)
            assert str(left) == s[:i]
            assert str(right) == s[i:]

    def test_split_negative(self):
        r = Rope("hello")
        left, right = r.split(-1)
        assert str(left) == ""
        assert str(right) == "hello"

    def test_split_beyond_end(self):
        r = Rope("hello")
        left, right = r.split(100)
        assert str(left) == "hello"
        assert str(right) == ""


# ============================================================
# Insert
# ============================================================

class TestInsert:
    def test_insert_middle(self):
        r = Rope("heo")
        r2 = r.insert(2, "ll")
        assert str(r2) == "hello"

    def test_insert_beginning(self):
        r = Rope("world")
        r2 = r.insert(0, "hello ")
        assert str(r2) == "hello world"

    def test_insert_end(self):
        r = Rope("hello")
        r2 = r.insert(5, " world")
        assert str(r2) == "hello world"

    def test_insert_preserves_original(self):
        r = Rope("hello")
        r.insert(2, "XX")
        assert str(r) == "hello"

    def test_insert_rope(self):
        r = Rope("hello")
        r2 = r.insert(5, Rope(" world"))
        assert str(r2) == "hello world"

    def test_insert_at_negative(self):
        r = Rope("world")
        r2 = r.insert(-5, "hello ")
        assert str(r2) == "hello world"

    def test_insert_beyond_end(self):
        r = Rope("hello")
        r2 = r.insert(100, "!")
        assert str(r2) == "hello!"


# ============================================================
# Delete
# ============================================================

class TestDelete:
    def test_delete_range(self):
        r = Rope("hello world")
        r2 = r.delete(5, 11)
        assert str(r2) == "hello"

    def test_delete_single_char(self):
        r = Rope("hello")
        r2 = r.delete(1)
        assert str(r2) == "hllo"

    def test_delete_beginning(self):
        r = Rope("hello world")
        r2 = r.delete(0, 6)
        assert str(r2) == "world"

    def test_delete_end(self):
        r = Rope("hello world")
        r2 = r.delete(4)
        assert str(r2) == "hell world"

    def test_delete_preserves_original(self):
        r = Rope("hello world")
        r.delete(0, 6)
        assert str(r) == "hello world"

    def test_delete_nothing(self):
        r = Rope("hello")
        r2 = r.delete(2, 2)
        assert str(r2) == "hello"

    def test_delete_all(self):
        r = Rope("hello")
        r2 = r.delete(0, 5)
        assert str(r2) == ""

    def test_delete_clamps_bounds(self):
        r = Rope("hello")
        r2 = r.delete(-5, 100)
        assert str(r2) == ""


# ============================================================
# Substring
# ============================================================

class TestSubstring:
    def test_substring_middle(self):
        r = Rope("hello world")
        s = r.substring(6, 11)
        assert str(s) == "world"

    def test_substring_full(self):
        r = Rope("hello")
        s = r.substring(0, 5)
        assert str(s) == "hello"

    def test_substring_empty(self):
        r = Rope("hello")
        s = r.substring(2, 2)
        assert str(s) == ""

    def test_substring_single_char(self):
        r = Rope("hello")
        s = r.substring(1, 2)
        assert str(s) == "e"

    def test_substring_default_end(self):
        r = Rope("hello world")
        s = r.substring(6)
        assert str(s) == "world"

    def test_getitem_slice(self):
        r = Rope("hello world")
        assert str(r[0:5]) == "hello"
        assert str(r[6:]) == "world"
        assert str(r[:5]) == "hello"

    def test_getitem_slice_step(self):
        r = Rope("abcdef")
        assert str(r[::2]) == "ace"


# ============================================================
# Balance
# ============================================================

class TestBalance:
    def test_balance_preserves_content(self):
        r = Rope("hello world this is a test")
        b = r.balance()
        assert str(b) == str(r)

    def test_balance_reduces_depth(self):
        # Build deeply unbalanced rope
        r = Rope("a")
        for i in range(100):
            r = Rope(Branch(r._root, Leaf("b")))
        balanced = r.balance()
        assert balanced.depth < r.depth
        assert str(balanced) == str(r)

    def test_balance_empty(self):
        r = Rope.empty()
        b = r.balance()
        assert b.length == 0

    def test_is_balanced(self):
        r = Rope("hello")
        assert r.is_balanced()

    def test_unbalanced_detection(self):
        # Build pathological rope
        node = Leaf("a")
        for _ in range(50):
            node = Branch(node, Leaf("b"))
        r = Rope(node)
        # May or may not be balanced depending on length vs depth
        # But after balance() it should be
        b = r.balance()
        assert b.is_balanced()


# ============================================================
# Iteration
# ============================================================

class TestIteration:
    def test_iter_chars(self):
        r = Rope("hello")
        assert list(r) == ['h', 'e', 'l', 'l', 'o']

    def test_iter_empty(self):
        r = Rope.empty()
        assert list(r) == []

    def test_iter_long(self):
        s = "abcdef" * 50
        r = Rope(s)
        assert ''.join(r) == s

    def test_contains(self):
        r = Rope("hello world")
        assert "hello" in r
        assert "world" in r
        assert "xyz" not in r
        assert "" in r


# ============================================================
# Find / Search
# ============================================================

class TestFind:
    def test_find_present(self):
        r = Rope("hello world")
        assert r.find("world") == 6

    def test_find_absent(self):
        r = Rope("hello world")
        assert r.find("xyz") == -1

    def test_find_at_start(self):
        r = Rope("hello world")
        assert r.find("hello") == 0

    def test_find_empty_pattern(self):
        r = Rope("hello")
        assert r.find("") == 0
        assert r.find("", 3) == 3

    def test_find_with_start(self):
        r = Rope("abcabc")
        assert r.find("abc", 1) == 3

    def test_find_all(self):
        r = Rope("abcabcabc")
        assert r.find_all("abc") == [0, 3, 6]

    def test_find_all_none(self):
        r = Rope("hello")
        assert r.find_all("xyz") == []

    def test_find_all_overlapping(self):
        r = Rope("aaaa")
        assert r.find_all("aa") == [0, 1, 2]

    def test_find_long_pattern(self):
        # Triggers BMH search (pattern > 3 chars)
        r = Rope("the quick brown fox jumps over the lazy dog")
        assert r.find("jumps") == 20
        assert r.find("missing") == -1

    def test_find_rope_pattern(self):
        r = Rope("hello world")
        assert r.find(Rope("world")) == 6

    def test_find_pattern_longer_than_rope(self):
        r = Rope("hi")
        assert r.find("hello") == -1


# ============================================================
# Replace
# ============================================================

class TestReplace:
    def test_replace_first(self):
        r = Rope("hello world")
        r2 = r.replace("world", "there")
        assert str(r2) == "hello there"

    def test_replace_not_found(self):
        r = Rope("hello world")
        r2 = r.replace("xyz", "abc")
        assert str(r2) == "hello world"

    def test_replace_all(self):
        r = Rope("aXbXcXd")
        r2 = r.replace_all("X", "-")
        assert str(r2) == "a-b-c-d"

    def test_replace_with_longer(self):
        r = Rope("ab")
        r2 = r.replace("a", "xyz")
        assert str(r2) == "xyzb"

    def test_replace_with_shorter(self):
        r = Rope("hello")
        r2 = r.replace("ell", "i")
        assert str(r2) == "hio"

    def test_replace_empty_old(self):
        r = Rope("hello")
        r2 = r.replace("", "x")
        assert str(r2) == "hello"

    def test_replace_with_rope(self):
        r = Rope("hello world")
        r2 = r.replace("world", Rope("there"))
        assert str(r2) == "hello there"

    def test_replace_all_multiple(self):
        r = Rope("aaa")
        r2 = r.replace_all("a", "bb")
        assert str(r2) == "bbbbbb"


# ============================================================
# String Operations
# ============================================================

class TestStringOps:
    def test_upper(self):
        r = Rope("hello")
        assert str(r.upper()) == "HELLO"

    def test_lower(self):
        r = Rope("HELLO")
        assert str(r.lower()) == "hello"

    def test_strip(self):
        r = Rope("  hello  ")
        assert str(r.strip()) == "hello"

    def test_startswith(self):
        r = Rope("hello world")
        assert r.startswith("hello")
        assert not r.startswith("world")
        assert r.startswith("")

    def test_endswith(self):
        r = Rope("hello world")
        assert r.endswith("world")
        assert not r.endswith("hello")
        assert r.endswith("")

    def test_startswith_rope(self):
        r = Rope("hello world")
        assert r.startswith(Rope("hello"))

    def test_endswith_rope(self):
        r = Rope("hello world")
        assert r.endswith(Rope("world"))

    def test_startswith_too_long(self):
        r = Rope("hi")
        assert not r.startswith("hello")

    def test_endswith_too_long(self):
        r = Rope("hi")
        assert not r.endswith("hello")

    def test_reverse(self):
        r = Rope("hello")
        assert str(r.reverse()) == "olleh"

    def test_repeat(self):
        r = Rope("ab")
        assert str(r.repeat(3)) == "ababab"

    def test_repeat_zero(self):
        r = Rope("ab")
        assert str(r.repeat(0)) == ""

    def test_repeat_one(self):
        r = Rope("ab")
        r2 = r.repeat(1)
        assert str(r2) == "ab"

    def test_repeat_large(self):
        r = Rope("x")
        r2 = r.repeat(1000)
        assert r2.length == 1000
        assert str(r2) == "x" * 1000


# ============================================================
# Line Operations
# ============================================================

class TestLines:
    def test_lines_basic(self):
        r = Rope("hello\nworld\nfoo")
        assert list(r.lines()) == ["hello", "world", "foo"]

    def test_lines_single(self):
        r = Rope("hello")
        assert list(r.lines()) == ["hello"]

    def test_lines_empty(self):
        r = Rope("")
        assert list(r.lines()) == []

    def test_lines_trailing_newline(self):
        r = Rope("hello\n")
        assert list(r.lines()) == ["hello", ""]

    def test_line_count(self):
        r = Rope("a\nb\nc")
        assert r.line_count() == 3

    def test_line_count_single(self):
        r = Rope("hello")
        assert r.line_count() == 1

    def test_line_at(self):
        r = Rope("hello\nworld\nfoo")
        assert r.line_at(0) == "hello"
        assert r.line_at(1) == "world"
        assert r.line_at(2) == "foo"

    def test_line_at_out_of_range(self):
        r = Rope("hello")
        with pytest.raises(IndexError):
            r.line_at(1)

    def test_char_to_line(self):
        r = Rope("hello\nworld")
        assert r.char_to_line(0) == 0
        assert r.char_to_line(4) == 0
        assert r.char_to_line(5) == 0  # the \n itself is on line 0
        assert r.char_to_line(6) == 1

    def test_char_to_line_out_of_range(self):
        r = Rope("hello")
        with pytest.raises(IndexError):
            r.char_to_line(5)

    def test_line_to_char(self):
        r = Rope("hello\nworld\nfoo")
        assert r.line_to_char(0) == 0
        assert r.line_to_char(1) == 6
        assert r.line_to_char(2) == 12

    def test_line_to_char_out_of_range(self):
        r = Rope("hello")
        with pytest.raises(IndexError):
            r.line_to_char(1)


# ============================================================
# Join / Split at String
# ============================================================

class TestJoinSplit:
    def test_join(self):
        parts = [Rope("a"), Rope("b"), Rope("c")]
        r = Rope.join(", ", parts)
        assert str(r) == "a, b, c"

    def test_join_strings(self):
        parts = ["hello", "world"]
        r = Rope.join(" ", parts)
        assert str(r) == "hello world"

    def test_join_empty(self):
        r = Rope.join(", ", [])
        assert str(r) == ""

    def test_join_single(self):
        r = Rope.join(", ", [Rope("only")])
        assert str(r) == "only"

    def test_join_rope_separator(self):
        r = Rope.join(Rope("-"), [Rope("a"), Rope("b")])
        assert str(r) == "a-b"

    def test_split_at_string(self):
        r = Rope("a,b,c")
        parts = r.split_at_string(",")
        assert [str(p) for p in parts] == ["a", "b", "c"]

    def test_split_at_string_not_found(self):
        r = Rope("hello")
        parts = r.split_at_string(",")
        assert [str(p) for p in parts] == ["hello"]

    def test_split_at_string_empty(self):
        r = Rope.empty()
        parts = r.split_at_string(",")
        assert len(parts) == 1
        assert str(parts[0]) == ""

    def test_split_at_rope_delimiter(self):
        r = Rope("a::b::c")
        parts = r.split_at_string(Rope("::"))
        assert [str(p) for p in parts] == ["a", "b", "c"]


# ============================================================
# Equality and Hashing
# ============================================================

class TestEquality:
    def test_equal_ropes(self):
        a = Rope("hello")
        b = Rope("hello")
        assert a == b

    def test_unequal_ropes(self):
        a = Rope("hello")
        b = Rope("world")
        assert a != b

    def test_equal_to_string(self):
        r = Rope("hello")
        assert r == "hello"

    def test_hash_equal(self):
        a = Rope("hello")
        b = Rope("hello")
        assert hash(a) == hash(b)

    def test_hash_in_set(self):
        s = {Rope("hello"), Rope("world")}
        assert Rope("hello") in s

    def test_not_equal_to_int(self):
        r = Rope("hello")
        assert r != 42


# ============================================================
# Persistence
# ============================================================

class TestPersistence:
    def test_concat_persistence(self):
        a = Rope("hello")
        b = Rope(" world")
        c = a.concat(b)
        assert str(a) == "hello"
        assert str(b) == " world"
        assert str(c) == "hello world"

    def test_split_persistence(self):
        r = Rope("hello world")
        left, right = r.split(5)
        assert str(r) == "hello world"
        assert str(left) == "hello"
        assert str(right) == " world"

    def test_insert_persistence(self):
        r = Rope("hllo")
        r2 = r.insert(1, "e")
        assert str(r) == "hllo"
        assert str(r2) == "hello"

    def test_delete_persistence(self):
        r = Rope("hello")
        r2 = r.delete(1, 3)
        assert str(r) == "hello"
        assert str(r2) == "hlo"

    def test_multiple_versions(self):
        v0 = Rope("hello")
        v1 = v0.insert(5, " world")
        v2 = v1.insert(11, "!")
        v3 = v0.insert(0, "say: ")
        assert str(v0) == "hello"
        assert str(v1) == "hello world"
        assert str(v2) == "hello world!"
        assert str(v3) == "say: hello"

    def test_branching_history(self):
        base = Rope("document")
        branch_a = base.insert(8, " version A")
        branch_b = base.insert(8, " version B")
        assert str(base) == "document"
        assert str(branch_a) == "document version A"
        assert str(branch_b) == "document version B"


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_char(self):
        r = Rope("x")
        assert r.length == 1
        assert r.char_at(0) == 'x'
        assert str(r) == "x"

    def test_unicode(self):
        r = Rope("hello \u2603 world")
        assert r.length == 13
        assert r.char_at(6) == '\u2603'

    def test_newlines(self):
        r = Rope("line1\nline2\nline3")
        assert r.line_count() == 3

    def test_tabs(self):
        r = Rope("a\tb\tc")
        assert r.length == 5
        assert r.char_at(1) == '\t'

    def test_very_long_rope(self):
        s = "abcdefghij" * 1000  # 10K chars
        r = Rope(s)
        assert r.length == 10000
        assert str(r) == s

    def test_rapid_insertions(self):
        r = Rope("")
        for i in range(100):
            r = r.insert(r.length, chr(65 + (i % 26)))
        assert r.length == 100

    def test_rapid_deletions(self):
        r = Rope("a" * 100)
        for _ in range(100):
            r = r.delete(0)
        assert r.length == 0

    def test_split_and_rejoin(self):
        s = "the quick brown fox"
        r = Rope(s)
        for i in range(len(s)):
            left, right = r.split(i)
            rejoined = left.concat(right)
            assert str(rejoined) == s

    def test_deep_tree_operations(self):
        # Build a deep tree, then do operations on it
        r = Rope("a")
        for i in range(50):
            r = r.concat(Rope(chr(65 + (i % 26))))
        assert r.length == 51
        # All operations should still work
        assert r.char_at(0) == 'a'
        left, right = r.split(25)
        assert left.length + right.length == 51


# ============================================================
# Tree Properties
# ============================================================

class TestTreeProperties:
    def test_depth_leaf(self):
        r = Rope("hello")
        assert r.depth == 0

    def test_depth_branch(self):
        s = "x" * (MAX_LEAF * 4)
        r = Rope(s)
        assert r.depth > 0

    def test_leaf_count(self):
        s = "x" * (MAX_LEAF * 4)
        r = Rope(s)
        assert r.leaf_count() == 4

    def test_len(self):
        r = Rope("hello")
        assert len(r) == 5


# ============================================================
# Stress Tests
# ============================================================

class TestStress:
    def test_edit_simulation(self):
        """Simulate a text editing session."""
        r = Rope("")
        # Type a paragraph
        text = "The quick brown fox jumps over the lazy dog. "
        for ch in text:
            r = r.insert(r.length, ch)
        assert str(r) == text

        # Insert in the middle
        r = r.insert(10, "VERY ")
        assert "VERY brown" in str(r)

        # Delete a word
        idx = r.find("quick ")
        r = r.delete(idx, idx + 6)
        assert "quick" not in str(r)

        # Replace
        r = r.replace("lazy", "energetic")
        assert "energetic" in str(r)

    def test_undo_redo_simulation(self):
        """Simulate undo/redo with persistent versions."""
        history = [Rope("hello")]
        history.append(history[-1].insert(5, " world"))
        history.append(history[-1].insert(11, "!"))
        history.append(history[-1].delete(0, 6))

        assert str(history[0]) == "hello"
        assert str(history[1]) == "hello world"
        assert str(history[2]) == "hello world!"
        assert str(history[3]) == "world!"

        # "Undo" to version 1 and branch
        history.append(history[1].insert(11, "?"))
        assert str(history[4]) == "hello world?"

    def test_large_document(self):
        """Test with a large document."""
        lines = [f"Line {i}: " + "x" * 70 for i in range(100)]
        text = '\n'.join(lines)
        r = Rope(text)
        assert r.length == len(text)
        assert r.line_count() == 100

        # Edit in the middle
        mid = r.length // 2
        r2 = r.insert(mid, "\n--- INSERTED ---\n")
        assert r2.line_count() == 102

        # Delete some lines
        start = r2.line_to_char(50)
        end = r2.line_to_char(52)
        r3 = r2.delete(start, end)
        assert r3.line_count() == 100

    def test_concat_balance_stress(self):
        """Stress test concat with many small ropes."""
        ropes = [Rope(f"segment{i}") for i in range(200)]
        result = Rope.empty()
        for r in ropes:
            result = result.concat(r)
        expected = ''.join(f"segment{i}" for i in range(200))
        assert str(result) == expected
        # Explicit balance should produce a balanced tree
        result = result.balance()
        assert result.is_balanced()

    def test_find_in_large_rope(self):
        """Test search in a large rope."""
        text = "abc" * 1000 + "NEEDLE" + "def" * 1000
        r = Rope(text)
        assert r.find("NEEDLE") == 3000
        assert r.find("MISSING") == -1

    def test_replace_all_stress(self):
        """Replace all in a rope with many occurrences."""
        r = Rope("aXaXaXaXaXa")
        r2 = r.replace_all("X", "YY")
        assert str(r2) == "aYYaYYaYYaYYaYYa"


# ============================================================
# Leaf Merging
# ============================================================

class TestLeafMerging:
    def test_small_concat_merges(self):
        """Two small leaves should merge into one."""
        a = Rope("hi")
        b = Rope(" there")
        c = a.concat(b)
        # Result should be a single leaf since combined < MAX_LEAF
        assert isinstance(c._root, Leaf)

    def test_large_concat_branches(self):
        """Two large leaves should create a branch."""
        a = Rope("a" * MAX_LEAF)
        b = Rope("b" * MAX_LEAF)
        c = a.concat(b)
        assert isinstance(c._root, Branch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
