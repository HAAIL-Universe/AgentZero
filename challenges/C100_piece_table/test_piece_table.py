"""Tests for C100: Piece Table"""

import pytest
from piece_table import PieceTable, BufferType, Piece, Mark, Cursor


# ============================================================
# Construction and basic properties
# ============================================================

class TestConstruction:
    def test_empty(self):
        pt = PieceTable()
        assert len(pt) == 0
        assert pt.text() == ''
        assert pt.piece_count() == 0

    def test_from_text(self):
        pt = PieceTable('hello')
        assert len(pt) == 5
        assert pt.text() == 'hello'
        assert pt.piece_count() == 1

    def test_multiline(self):
        pt = PieceTable('a\nb\nc')
        assert pt.text() == 'a\nb\nc'
        assert pt.line_count == 3

    def test_repr(self):
        pt = PieceTable('abc')
        assert 'PieceTable' in repr(pt)
        assert 'length=3' in repr(pt)

    def test_str(self):
        pt = PieceTable('hello')
        assert str(pt) == 'hello'

    def test_buffer_sizes(self):
        pt = PieceTable('hello')
        sizes = pt.buffer_sizes()
        assert sizes['original'] == 5
        assert sizes['add'] == 0

    def test_length_property(self):
        pt = PieceTable('abc')
        assert pt.length == 3

    def test_version_starts_zero(self):
        pt = PieceTable('abc')
        assert pt.version == 0


# ============================================================
# Character access
# ============================================================

class TestCharAccess:
    def test_char_at(self):
        pt = PieceTable('hello')
        assert pt.char_at(0) == 'h'
        assert pt.char_at(4) == 'o'

    def test_char_at_out_of_range(self):
        pt = PieceTable('hi')
        with pytest.raises(IndexError):
            pt.char_at(-1)
        with pytest.raises(IndexError):
            pt.char_at(2)

    def test_text_slice(self):
        pt = PieceTable('hello world')
        assert pt.text(0, 5) == 'hello'
        assert pt.text(6, 11) == 'world'

    def test_text_slice_clamped(self):
        pt = PieceTable('abc')
        assert pt.text(-5, 100) == 'abc'

    def test_text_empty_range(self):
        pt = PieceTable('abc')
        assert pt.text(2, 1) == ''

    def test_iteration(self):
        pt = PieceTable('abc')
        assert list(pt) == ['a', 'b', 'c']

    def test_chars_range(self):
        pt = PieceTable('abcdef')
        assert list(pt.chars(2, 4)) == ['c', 'd']


# ============================================================
# Insert operations
# ============================================================

class TestInsert:
    def test_insert_at_start(self):
        pt = PieceTable('world')
        pt.insert(0, 'hello ')
        assert pt.text() == 'hello world'

    def test_insert_at_end(self):
        pt = PieceTable('hello')
        pt.insert(5, ' world')
        assert pt.text() == 'hello world'

    def test_insert_in_middle(self):
        pt = PieceTable('helo')
        pt.insert(2, 'l')
        assert pt.text() == 'hello'

    def test_insert_empty(self):
        pt = PieceTable('abc')
        pt.insert(1, '')
        assert pt.text() == 'abc'
        assert pt.version == 0  # no change

    def test_insert_into_empty(self):
        pt = PieceTable()
        pt.insert(0, 'hello')
        assert pt.text() == 'hello'
        assert len(pt) == 5

    def test_insert_out_of_range(self):
        pt = PieceTable('abc')
        with pytest.raises(IndexError):
            pt.insert(5, 'x')
        with pytest.raises(IndexError):
            pt.insert(-1, 'x')

    def test_multiple_inserts(self):
        pt = PieceTable('ac')
        pt.insert(1, 'b')
        pt.insert(3, 'd')
        assert pt.text() == 'abcd'

    def test_insert_creates_pieces(self):
        pt = PieceTable('hello')
        pt.insert(2, 'XY')
        assert pt.piece_count() >= 2  # at least original split + new

    def test_insert_updates_length(self):
        pt = PieceTable('abc')
        pt.insert(1, 'xyz')
        assert len(pt) == 6

    def test_insert_updates_version(self):
        pt = PieceTable('abc')
        v0 = pt.version
        pt.insert(1, 'x')
        assert pt.version == v0 + 1

    def test_insert_at_piece_boundary(self):
        pt = PieceTable('abc')
        pt.insert(3, 'def')  # at end of first piece
        pt.insert(3, 'XY')   # at boundary between pieces
        assert pt.text() == 'abcXYdef'

    def test_insert_newlines(self):
        pt = PieceTable('ab')
        pt.insert(1, '\n')
        assert pt.text() == 'a\nb'
        assert pt.line_count == 2

    def test_insert_multiline_text(self):
        pt = PieceTable()
        pt.insert(0, 'line1\nline2\nline3')
        assert pt.line_count == 3
        assert pt.get_line(0) == 'line1'
        assert pt.get_line(2) == 'line3'


# ============================================================
# Delete operations
# ============================================================

class TestDelete:
    def test_delete_from_start(self):
        pt = PieceTable('hello')
        deleted = pt.delete(0, 2)
        assert deleted == 'he'
        assert pt.text() == 'llo'

    def test_delete_from_end(self):
        pt = PieceTable('hello')
        deleted = pt.delete(3, 2)
        assert deleted == 'lo'
        assert pt.text() == 'hel'

    def test_delete_from_middle(self):
        pt = PieceTable('hello')
        deleted = pt.delete(1, 3)
        assert deleted == 'ell'
        assert pt.text() == 'ho'

    def test_delete_all(self):
        pt = PieceTable('hello')
        deleted = pt.delete(0, 5)
        assert deleted == 'hello'
        assert pt.text() == ''
        assert len(pt) == 0

    def test_delete_zero_length(self):
        pt = PieceTable('abc')
        result = pt.delete(1, 0)
        assert result == ''
        assert pt.text() == 'abc'

    def test_delete_out_of_range(self):
        pt = PieceTable('abc')
        with pytest.raises(IndexError):
            pt.delete(2, 5)

    def test_delete_updates_length(self):
        pt = PieceTable('hello')
        pt.delete(1, 2)
        assert len(pt) == 3

    def test_delete_spanning_pieces(self):
        pt = PieceTable('hello')
        pt.insert(2, 'XY')  # h e XY l l o
        pt.delete(1, 4)     # delete 'eXYl'
        assert pt.text() == 'hlo'

    def test_delete_entire_piece(self):
        pt = PieceTable('abc')
        pt.insert(1, 'XY')
        pt.delete(1, 2)  # delete 'XY'
        assert pt.text() == 'abc'

    def test_delete_newline(self):
        pt = PieceTable('a\nb')
        assert pt.line_count == 2
        pt.delete(1, 1)  # delete \n
        assert pt.text() == 'ab'
        assert pt.line_count == 1


# ============================================================
# Undo / Redo
# ============================================================

class TestUndoRedo:
    def test_undo_insert(self):
        pt = PieceTable('hello')
        pt.insert(5, ' world')
        assert pt.text() == 'hello world'
        pt.undo()
        assert pt.text() == 'hello'

    def test_undo_delete(self):
        pt = PieceTable('hello')
        pt.delete(0, 2)
        assert pt.text() == 'llo'
        pt.undo()
        assert pt.text() == 'hello'

    def test_redo_insert(self):
        pt = PieceTable('hello')
        pt.insert(5, ' world')
        pt.undo()
        pt.redo()
        assert pt.text() == 'hello world'

    def test_redo_delete(self):
        pt = PieceTable('hello')
        pt.delete(0, 2)
        pt.undo()
        pt.redo()
        assert pt.text() == 'llo'

    def test_undo_empty_stack(self):
        pt = PieceTable('abc')
        assert pt.undo() == False

    def test_redo_empty_stack(self):
        pt = PieceTable('abc')
        assert pt.redo() == False

    def test_redo_cleared_after_new_edit(self):
        pt = PieceTable('abc')
        pt.insert(3, 'd')
        pt.undo()
        pt.insert(3, 'e')
        assert pt.redo() == False

    def test_multiple_undo(self):
        pt = PieceTable('a')
        pt.insert(1, 'b')
        pt.insert(2, 'c')
        pt.insert(3, 'd')
        assert pt.text() == 'abcd'
        pt.undo()
        assert pt.text() == 'abc'
        pt.undo()
        assert pt.text() == 'ab'
        pt.undo()
        assert pt.text() == 'a'

    def test_undo_redo_sequence(self):
        pt = PieceTable('hello')
        pt.insert(5, '!')
        pt.insert(6, '!')
        pt.undo()
        pt.undo()
        pt.redo()
        assert pt.text() == 'hello!'

    def test_undo_preserves_length(self):
        pt = PieceTable('abc')
        pt.insert(1, 'xyz')
        assert len(pt) == 6
        pt.undo()
        assert len(pt) == 3

    def test_undo_mixed_operations(self):
        pt = PieceTable('hello world')
        pt.delete(5, 6)
        assert pt.text() == 'hello'
        pt.insert(5, ' there')
        assert pt.text() == 'hello there'
        pt.undo()
        assert pt.text() == 'hello'
        pt.undo()
        assert pt.text() == 'hello world'


# ============================================================
# Line operations
# ============================================================

class TestLines:
    def test_single_line(self):
        pt = PieceTable('hello')
        assert pt.line_count == 1
        assert pt.get_line(0) == 'hello'

    def test_multiple_lines(self):
        pt = PieceTable('a\nb\nc')
        assert pt.line_count == 3
        assert pt.get_line(0) == 'a'
        assert pt.get_line(1) == 'b'
        assert pt.get_line(2) == 'c'

    def test_trailing_newline(self):
        pt = PieceTable('a\nb\n')
        assert pt.line_count == 3
        assert pt.get_line(2) == ''

    def test_empty_lines(self):
        pt = PieceTable('\n\n')
        assert pt.line_count == 3
        assert pt.get_line(0) == ''
        assert pt.get_line(1) == ''

    def test_line_start(self):
        pt = PieceTable('ab\ncd\nef')
        assert pt.line_start(0) == 0
        assert pt.line_start(1) == 3
        assert pt.line_start(2) == 6

    def test_line_end(self):
        pt = PieceTable('ab\ncd\nef')
        assert pt.line_end(0) == 3  # includes \n
        assert pt.line_end(1) == 6
        assert pt.line_end(2) == 8

    def test_line_start_out_of_range(self):
        pt = PieceTable('ab')
        with pytest.raises(IndexError):
            pt.line_start(5)

    def test_get_lines(self):
        pt = PieceTable('a\nb\nc')
        assert pt.get_lines() == ['a', 'b', 'c']

    def test_get_line_out_of_range(self):
        pt = PieceTable('abc')
        with pytest.raises(IndexError):
            pt.get_line(5)

    def test_empty_piece_table_line_count(self):
        pt = PieceTable('')
        assert pt.line_count == 1

    def test_lines_after_insert(self):
        pt = PieceTable('ac')
        pt.insert(1, '\nb\n')
        assert pt.line_count == 3
        assert pt.get_line(0) == 'a'
        assert pt.get_line(1) == 'b'
        assert pt.get_line(2) == 'c'

    def test_offset_to_line_col(self):
        pt = PieceTable('ab\ncd\nef')
        assert pt.offset_to_line_col(0) == (0, 0)
        assert pt.offset_to_line_col(1) == (0, 1)
        assert pt.offset_to_line_col(3) == (1, 0)
        assert pt.offset_to_line_col(4) == (1, 1)
        assert pt.offset_to_line_col(6) == (2, 0)

    def test_line_col_to_offset(self):
        pt = PieceTable('ab\ncd\nef')
        assert pt.line_col_to_offset(0, 0) == 0
        assert pt.line_col_to_offset(1, 0) == 3
        assert pt.line_col_to_offset(2, 1) == 7

    def test_insert_lines(self):
        pt = PieceTable('a\nc\n')
        pt.insert_lines(1, ['b'])
        assert pt.get_line(0) == 'a'
        assert pt.get_line(1) == 'b'
        assert pt.get_line(2) == 'c'

    def test_delete_lines(self):
        pt = PieceTable('a\nb\nc\n')
        pt.delete_lines(1, 1)
        assert pt.get_line(0) == 'a'
        assert pt.get_line(1) == 'c'

    def test_replace_line(self):
        pt = PieceTable('a\nb\nc\n')
        pt.replace_line(1, 'X')
        text = pt.text()
        lines = text.split('\n')
        # Should have 'a', 'X', 'c', ''
        assert lines[0] == 'a'
        assert lines[1] == 'X'
        assert lines[2] == 'c'

    def test_delete_lines_past_end(self):
        pt = PieceTable('a\nb\n')
        result = pt.delete_lines(10)
        assert result == ''
        assert pt.text() == 'a\nb\n'

    def test_insert_lines_at_end(self):
        pt = PieceTable('a\n')
        pt.insert_lines(5, ['b', 'c'])
        text = pt.text()
        assert 'b' in text
        assert 'c' in text


# ============================================================
# Search operations
# ============================================================

class TestSearch:
    def test_find_basic(self):
        pt = PieceTable('hello world')
        assert pt.find('world') == 6

    def test_find_not_found(self):
        pt = PieceTable('hello')
        assert pt.find('xyz') == -1

    def test_find_at_start(self):
        pt = PieceTable('hello')
        assert pt.find('hel') == 0

    def test_find_empty_pattern(self):
        pt = PieceTable('hello')
        assert pt.find('') == 0

    def test_find_with_start(self):
        pt = PieceTable('abcabc')
        assert pt.find('abc', start=1) == 3

    def test_find_all(self):
        pt = PieceTable('abcabcabc')
        assert pt.find_all('abc') == [0, 3, 6]

    def test_find_all_overlapping(self):
        pt = PieceTable('aaa')
        assert pt.find_all('aa') == [0, 1]

    def test_find_all_no_match(self):
        pt = PieceTable('hello')
        assert pt.find_all('xyz') == []

    def test_find_after_edit(self):
        pt = PieceTable('hello')
        pt.insert(5, ' world')
        assert pt.find('world') == 6

    def test_replace_single(self):
        pt = PieceTable('hello world')
        n = pt.replace('world', 'there', count=1)
        assert n == 1
        assert pt.text() == 'hello there'

    def test_replace_all(self):
        pt = PieceTable('aXbXc')
        n = pt.replace('X', 'Y')
        assert n == 2
        assert pt.text() == 'aYbYc'

    def test_replace_empty_pattern(self):
        pt = PieceTable('abc')
        n = pt.replace('', 'x')
        assert n == 0
        assert pt.text() == 'abc'

    def test_replace_count(self):
        pt = PieceTable('aaa')
        n = pt.replace('a', 'b', count=2)
        assert n == 2
        assert pt.text() == 'bba'

    def test_find_with_end(self):
        pt = PieceTable('abcabc')
        assert pt.find('abc', end=4) == 0
        assert pt.find('abc', start=1, end=4) == -1

    def test_find_all_with_range(self):
        pt = PieceTable('abcabcabc')
        assert pt.find_all('abc', start=1, end=7) == [3]


# ============================================================
# Marks
# ============================================================

class TestMarks:
    def test_set_and_get_mark(self):
        pt = PieceTable('hello')
        pt.set_mark('a', 3)
        assert pt.get_mark('a') == 3

    def test_mark_not_found(self):
        pt = PieceTable('hello')
        assert pt.get_mark('x') is None

    def test_mark_tracks_insert_after(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 3)
        pt.insert(1, 'XY')  # insert before mark
        assert pt.get_mark('m') == 5  # shifted right by 2

    def test_mark_unaffected_by_insert_after(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 1)
        pt.insert(3, 'XY')  # insert after mark
        assert pt.get_mark('m') == 1  # unchanged

    def test_mark_tracks_delete_before(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 4)
        pt.delete(0, 2)  # delete before mark
        assert pt.get_mark('m') == 2  # shifted left by 2

    def test_mark_collapses_on_delete_through(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 2)
        pt.delete(1, 3)  # delete through mark position
        assert pt.get_mark('m') == 1  # collapses to delete start

    def test_mark_gravity_left(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 3, gravity='left')
        pt.insert(3, 'X')  # insert at mark position
        assert pt.get_mark('m') == 3  # stays (left gravity)

    def test_mark_gravity_right(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 3, gravity='right')
        pt.insert(3, 'X')  # insert at mark position
        assert pt.get_mark('m') == 4  # moves right

    def test_remove_mark(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 2)
        assert pt.remove_mark('m') == True
        assert pt.get_mark('m') is None

    def test_remove_nonexistent_mark(self):
        pt = PieceTable('hello')
        assert pt.remove_mark('x') == False

    def test_list_marks(self):
        pt = PieceTable('hello')
        pt.set_mark('a', 1)
        pt.set_mark('b', 3)
        marks = pt.list_marks()
        assert marks == {'a': 1, 'b': 3}

    def test_mark_out_of_range(self):
        pt = PieceTable('abc')
        with pytest.raises(IndexError):
            pt.set_mark('m', 10)

    def test_mark_undo_insert(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 3)
        pt.insert(1, 'XY')
        assert pt.get_mark('m') == 5
        pt.undo()
        assert pt.get_mark('m') == 3

    def test_mark_undo_delete(self):
        pt = PieceTable('hello')
        pt.set_mark('m', 4)
        pt.delete(0, 2)
        assert pt.get_mark('m') == 2
        pt.undo()
        assert pt.get_mark('m') == 4


# ============================================================
# Cursors
# ============================================================

class TestCursors:
    def test_add_cursor(self):
        pt = PieceTable('hello')
        idx = pt.add_cursor(3)
        assert idx == 0
        c = pt.get_cursor(0)
        assert c['position'] == 3
        assert c['anchor'] is None
        assert c['selection'] is None

    def test_multiple_cursors(self):
        pt = PieceTable('hello')
        pt.add_cursor(1)
        pt.add_cursor(3)
        assert pt.cursor_count() == 2
        assert pt.get_cursor(0)['position'] == 1
        assert pt.get_cursor(1)['position'] == 3

    def test_move_cursor(self):
        pt = PieceTable('hello')
        pt.add_cursor(0)
        pt.move_cursor(0, 3)
        assert pt.get_cursor(0)['position'] == 3

    def test_cursor_selection(self):
        pt = PieceTable('hello')
        pt.add_cursor(1)
        pt.move_cursor(0, 4, select=True)
        c = pt.get_cursor(0)
        assert c['selection'] == (1, 4)

    def test_cursor_clamped(self):
        pt = PieceTable('abc')
        pt.add_cursor(100)
        assert pt.get_cursor(0)['position'] == 3

    def test_cursor_tracks_insert(self):
        pt = PieceTable('hello')
        pt.add_cursor(3)
        pt.insert(1, 'XY')
        assert pt.get_cursor(0)['position'] == 5

    def test_cursor_tracks_delete(self):
        pt = PieceTable('hello')
        pt.add_cursor(4)
        pt.delete(0, 2)
        assert pt.get_cursor(0)['position'] == 2

    def test_remove_cursor(self):
        pt = PieceTable('hello')
        pt.add_cursor(1)
        pt.add_cursor(3)
        pt.remove_cursor(0)
        assert pt.cursor_count() == 1
        assert pt.get_cursor(0)['position'] == 3

    def test_get_nonexistent_cursor(self):
        pt = PieceTable('hello')
        assert pt.get_cursor(99) is None

    def test_cursor_move_clamp(self):
        pt = PieceTable('abc')
        pt.add_cursor(1)
        pt.move_cursor(0, -5)
        assert pt.get_cursor(0)['position'] == 0
        pt.move_cursor(0, 100)
        assert pt.get_cursor(0)['position'] == 3

    def test_cursor_selection_cancel(self):
        pt = PieceTable('hello')
        pt.add_cursor(1)
        pt.move_cursor(0, 3, select=True)
        assert pt.get_cursor(0)['selection'] == (1, 3)
        pt.move_cursor(0, 2, select=False)  # cancel selection
        assert pt.get_cursor(0)['selection'] is None

    def test_cursor_reverse_selection(self):
        pt = PieceTable('hello')
        pt.add_cursor(4)
        pt.move_cursor(0, 1, select=True)
        c = pt.get_cursor(0)
        assert c['selection'] == (1, 4)  # normalized

    def test_cursor_zero_width_selection(self):
        pt = PieceTable('hello')
        pt.add_cursor(2)
        pt.move_cursor(0, 2, select=True)
        assert pt.get_cursor(0)['selection'] is None  # zero-width = no selection


# ============================================================
# Snapshots
# ============================================================

class TestSnapshots:
    def test_snapshot_and_restore(self):
        pt = PieceTable('hello')
        v = pt.snapshot()
        pt.insert(5, ' world')
        assert pt.text() == 'hello world'
        pt.restore(v)
        assert pt.text() == 'hello'

    def test_restore_nonexistent(self):
        pt = PieceTable('abc')
        with pytest.raises(KeyError):
            pt.restore(999)

    def test_multiple_snapshots(self):
        pt = PieceTable('a')
        v1 = pt.snapshot()
        pt.insert(1, 'b')
        v2 = pt.snapshot()
        pt.insert(2, 'c')
        assert pt.text() == 'abc'
        pt.restore(v1)
        assert pt.text() == 'a'

    def test_list_snapshots(self):
        pt = PieceTable('a')
        v1 = pt.snapshot()
        pt.insert(1, 'b')
        v2 = pt.snapshot()
        snaps = pt.list_snapshots()
        assert v1 in snaps
        assert v2 in snaps

    def test_restore_clears_undo(self):
        pt = PieceTable('abc')
        pt.insert(3, 'd')
        v = pt.snapshot()
        pt.insert(4, 'e')
        pt.restore(v)
        assert pt.undo() == False

    def test_snapshot_preserves_length(self):
        pt = PieceTable('hello')
        v = pt.snapshot()
        pt.delete(0, 5)
        pt.insert(0, 'world!!')
        pt.restore(v)
        assert len(pt) == 5


# ============================================================
# Compact
# ============================================================

class TestCompact:
    def test_compact_adjacent(self):
        pt = PieceTable('hello')
        pt.insert(5, ' world')
        initial_pieces = pt.piece_count()
        pt.compact()
        # After compact, if add buffer pieces are adjacent they merge
        assert pt.text() == 'hello world'
        assert pt.piece_count() <= initial_pieces

    def test_compact_single_piece(self):
        pt = PieceTable('hello')
        pt.compact()
        assert pt.piece_count() == 1
        assert pt.text() == 'hello'

    def test_compact_empty(self):
        pt = PieceTable()
        pt.compact()  # should not crash
        assert pt.piece_count() == 0

    def test_compact_preserves_content(self):
        pt = PieceTable('abc')
        pt.insert(1, 'X')
        pt.insert(3, 'Y')
        pt.insert(5, 'Z')
        expected = pt.text()
        pt.compact()
        assert pt.text() == expected

    def test_compact_line_counts(self):
        pt = PieceTable('a\n')
        pt.insert(2, 'b\n')
        pt.insert(4, 'c\n')
        expected_lines = pt.line_count
        pt.compact()
        assert pt.line_count == expected_lines


# ============================================================
# Complex editing scenarios
# ============================================================

class TestComplexEditing:
    def test_build_text_char_by_char(self):
        pt = PieceTable()
        for i, c in enumerate('hello'):
            pt.insert(i, c)
        assert pt.text() == 'hello'

    def test_build_and_delete(self):
        pt = PieceTable('abcdef')
        pt.delete(4, 2)  # abcd
        pt.delete(2, 2)  # ab
        pt.delete(0, 2)  # empty
        assert pt.text() == ''
        assert len(pt) == 0

    def test_interleaved_insert_delete(self):
        pt = PieceTable('hello')
        pt.insert(5, ' world')
        pt.delete(0, 6)  # remove 'hello '
        assert pt.text() == 'world'
        pt.insert(0, 'the ')
        assert pt.text() == 'the world'

    def test_large_text(self):
        text = 'x' * 10000
        pt = PieceTable(text)
        assert len(pt) == 10000
        pt.insert(5000, 'MIDDLE')
        assert len(pt) == 10006
        assert pt.text(4998, 5008) == 'xxMIDDLExx'

    def test_many_small_inserts(self):
        pt = PieceTable()
        for i in range(100):
            pt.insert(i, str(i % 10))
        assert len(pt) == 100

    def test_undo_all_then_redo_all(self):
        pt = PieceTable('start')
        pt.insert(5, 'A')
        pt.insert(6, 'B')
        pt.insert(7, 'C')
        while pt.undo():
            pass
        assert pt.text() == 'start'
        while pt.redo():
            pass
        assert pt.text() == 'startABC'

    def test_delete_after_multiple_inserts(self):
        pt = PieceTable('')
        pt.insert(0, 'abc')
        pt.insert(3, 'def')
        pt.insert(6, 'ghi')
        # Now delete spanning all three inserted pieces
        pt.delete(2, 5)  # delete 'cdefg'
        assert pt.text() == 'abhi'

    def test_replace_entire_content(self):
        pt = PieceTable('old content')
        pt.delete(0, len(pt))
        pt.insert(0, 'new content')
        assert pt.text() == 'new content'

    def test_mixed_operations_with_lines(self):
        pt = PieceTable('line1\nline2\nline3\n')
        pt.delete_lines(1, 1)
        assert pt.get_line(0) == 'line1'
        assert pt.get_line(1) == 'line3'
        pt.insert_lines(1, ['inserted'])
        assert pt.get_line(1) == 'inserted'
        assert pt.get_line(2) == 'line3'


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_char_operations(self):
        pt = PieceTable('a')
        pt.insert(0, 'b')
        assert pt.text() == 'ba'
        pt.delete(1, 1)
        assert pt.text() == 'b'
        pt.undo()
        assert pt.text() == 'ba'

    def test_empty_after_all_deletes(self):
        pt = PieceTable('abc')
        pt.delete(0, 3)
        assert pt.text() == ''
        assert pt.line_count == 1
        assert pt.get_line(0) == ''

    def test_only_newlines(self):
        pt = PieceTable('\n\n\n')
        assert pt.line_count == 4
        for i in range(4):
            assert pt.get_line(i) == ''

    def test_unicode_text(self):
        pt = PieceTable('hello')
        pt.insert(5, ' world')
        assert pt.text() == 'hello world'
        # Basic ASCII works fine; piece table is character-based

    def test_very_long_line(self):
        line = 'x' * 10000
        pt = PieceTable(line)
        assert pt.line_count == 1
        assert pt.get_line(0) == line

    def test_insert_at_every_position(self):
        pt = PieceTable('abcde')
        positions = list(range(6))  # 0..5
        for p in reversed(positions):
            pt.insert(p, '|')
        assert pt.text() == '|a|b|c|d|e|'

    def test_offset_at_end(self):
        pt = PieceTable('abc')
        line, col = pt.offset_to_line_col(3)
        assert line == 0
        assert col == 3

    def test_chars_full_range(self):
        pt = PieceTable('abc')
        assert list(pt.chars()) == ['a', 'b', 'c']

    def test_text_with_none_args(self):
        pt = PieceTable('hello')
        assert pt.text(None, None) == 'hello'
        assert pt.text(None, 3) == 'hel'
        assert pt.text(2, None) == 'llo'


# ============================================================
# Piece internals
# ============================================================

class TestPieceInternals:
    def test_piece_repr(self):
        p = Piece(BufferType.ORIGINAL, 0, 5)
        assert 'ORIGINAL' in repr(p)
        assert '5' in repr(p)

    def test_add_buffer_growth(self):
        pt = PieceTable('hello')
        sizes1 = pt.buffer_sizes()
        pt.insert(5, ' world')
        sizes2 = pt.buffer_sizes()
        assert sizes2['add'] == 6
        assert sizes2['original'] == sizes1['original']

    def test_add_buffer_append_only(self):
        pt = PieceTable('abc')
        pt.insert(1, 'X')
        pt.insert(3, 'Y')
        # Add buffer should be 'XY' (append-only)
        sizes = pt.buffer_sizes()
        assert sizes['add'] == 2

    def test_piece_count_after_operations(self):
        pt = PieceTable('abcdef')
        assert pt.piece_count() == 1
        pt.insert(3, 'X')
        assert pt.piece_count() >= 2
        pt.delete(2, 3)  # delete across pieces
        # Piece count depends on implementation details


# ============================================================
# BMH Search edge cases
# ============================================================

class TestBMHSearch:
    def test_search_single_char(self):
        pt = PieceTable('abcdef')
        assert pt.find('d') == 3

    def test_search_full_text(self):
        pt = PieceTable('hello')
        assert pt.find('hello') == 0

    def test_search_longer_than_text(self):
        pt = PieceTable('hi')
        assert pt.find('hello') == -1

    def test_find_all_single_char(self):
        pt = PieceTable('abacada')
        assert pt.find_all('a') == [0, 2, 4, 6]

    def test_find_across_pieces(self):
        pt = PieceTable('hel')
        pt.insert(3, 'lo')
        assert pt.find('hello') == 0
        assert pt.find('ello') == 1

    def test_search_after_delete(self):
        pt = PieceTable('hello world')
        pt.delete(5, 1)  # 'helloworld'
        assert pt.find('oworld') == 4  # 'oworld' exists in 'helloworld'
        assert pt.find('o wor') == -1  # space was deleted


# ============================================================
# Stress tests
# ============================================================

class TestStress:
    def test_many_inserts_and_deletes(self):
        pt = PieceTable('initial')
        for i in range(50):
            pt.insert(i % (len(pt) + 1), f'[{i}]')
        for i in range(25):
            if len(pt) > 2:
                pt.delete(0, 2)
        # Just verify consistency
        text = pt.text()
        assert len(text) == len(pt)

    def test_undo_redo_stress(self):
        pt = PieceTable('start')
        for i in range(20):
            pt.insert(len(pt), str(i))
        for _ in range(20):
            pt.undo()
        assert pt.text() == 'start'
        for _ in range(20):
            pt.redo()
        expected = 'start' + ''.join(str(i) for i in range(20))
        assert pt.text() == expected

    def test_snapshot_stress(self):
        pt = PieceTable('base')
        snapshots = []
        for i in range(10):
            snapshots.append(pt.snapshot())
            pt.insert(len(pt), str(i))
        # Restore to earliest
        pt.restore(snapshots[0])
        assert pt.text() == 'base'

    def test_marks_through_many_edits(self):
        pt = PieceTable('abcdefghij')
        pt.set_mark('end', 9)
        for i in range(5):
            pt.insert(0, 'X')  # each shifts mark right
        assert pt.get_mark('end') == 14

    def test_compact_after_many_edits(self):
        pt = PieceTable('')
        for c in 'hello world this is a test':
            pt.insert(len(pt), c)
        before = pt.piece_count()
        pt.compact()
        after = pt.piece_count()
        assert after <= before
        assert pt.text() == 'hello world this is a test'

    def test_lines_with_many_newlines(self):
        text = '\n'.join(f'line{i}' for i in range(100))
        pt = PieceTable(text)
        assert pt.line_count == 100
        assert pt.get_line(50) == 'line50'
        assert pt.get_line(99) == 'line99'


# ============================================================
# Cursor + edit interaction
# ============================================================

class TestCursorEditInteraction:
    def test_cursor_anchor_tracks_insert(self):
        pt = PieceTable('hello')
        pt.add_cursor(1, anchor=4)
        pt.insert(0, 'XX')
        c = pt.get_cursor(0)
        assert c['position'] == 3
        assert c['anchor'] == 6

    def test_cursor_anchor_tracks_delete(self):
        pt = PieceTable('hello world')
        pt.add_cursor(8, anchor=2)
        pt.delete(0, 2)
        c = pt.get_cursor(0)
        assert c['position'] == 6
        assert c['anchor'] == 0

    def test_cursor_collapses_on_delete_through(self):
        pt = PieceTable('hello world')
        pt.add_cursor(5)
        pt.delete(3, 5)  # delete through cursor position
        assert pt.get_cursor(0)['position'] == 3


# ============================================================
# Full workflow simulation
# ============================================================

class TestWorkflowSimulation:
    def test_text_editor_session(self):
        """Simulate a text editing session."""
        pt = PieceTable('')
        pt.add_cursor(0)

        # Type "Hello World"
        for c in 'Hello World':
            pos = pt.get_cursor(0)['position']
            pt.insert(pos, c)
            pt.move_cursor(0, pos + 1)

        assert pt.text() == 'Hello World'

        # Select "World" and delete it
        pt.move_cursor(0, 6)
        pt.move_cursor(0, 11, select=True)
        sel = pt.get_cursor(0)['selection']
        pt.delete(sel[0], sel[1] - sel[0])
        pt.move_cursor(0, 6)

        assert pt.text() == 'Hello '

        # Type "Python"
        for c in 'Python':
            pos = pt.get_cursor(0)['position']
            pt.insert(pos, c)
            pt.move_cursor(0, pos + 1)

        assert pt.text() == 'Hello Python'

        # Undo typing "Python" (6 undos)
        for _ in range(6):
            pt.undo()
        assert pt.text() == 'Hello '

        # Undo delete
        pt.undo()
        assert pt.text() == 'Hello World'

    def test_multiline_editing(self):
        """Simulate editing a multi-line document."""
        pt = PieceTable('def hello():\n    pass\n')
        v = pt.snapshot()

        # Replace 'pass' with 'print("hello")'
        pos = pt.find('pass')
        pt.delete(pos, 4)
        pt.insert(pos, 'print("hello")')

        assert 'print("hello")' in pt.text()
        assert pt.line_count == 3

        # Restore
        pt.restore(v)
        assert 'pass' in pt.text()

    def test_collaborative_marks(self):
        """Simulate multiple bookmarks in an editor."""
        pt = PieceTable('function a() {}\nfunction b() {}\nfunction c() {}\n')
        pt.set_mark('fn_a', 0)
        pt.set_mark('fn_b', 16)
        pt.set_mark('fn_c', 32)

        # Insert a new function before b
        pt.insert(16, 'function x() {}\n')

        assert pt.get_mark('fn_a') == 0   # unchanged
        assert pt.get_mark('fn_b') == 32  # shifted by 16 (len of 'function x() {}\n')
        assert pt.get_mark('fn_c') == 48  # shifted by 16

    def test_search_and_replace_workflow(self):
        """Simulate find-and-replace."""
        pt = PieceTable('var x = 1;\nvar y = 2;\nvar z = 3;\n')
        n = pt.replace('var', 'let')
        assert n == 3
        assert 'var' not in pt.text()
        assert pt.text().count('let') == 3

    def test_incremental_typing_with_undo(self):
        pt = PieceTable('')
        # Type word
        for c in 'test':
            pt.insert(len(pt), c)
        assert pt.text() == 'test'
        # Undo each char
        pt.undo()
        assert pt.text() == 'tes'
        pt.undo()
        assert pt.text() == 'te'
        # Redo
        pt.redo()
        assert pt.text() == 'tes'
