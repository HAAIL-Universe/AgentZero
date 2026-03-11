"""
C100: Piece Table -- Text Editing Data Structure

The piece table is the data structure used by VS Code for text editing.
Instead of modifying text in place, it maintains two buffers (original + additions)
and a table of "pieces" that reference slices of those buffers.

Features:
- Two buffers: original (immutable) and add (append-only)
- Piece descriptors: (buffer, start, length)
- O(pieces) insert/delete, O(1) append to add buffer
- Undo/redo via command history
- Line index for O(log n) line-based access
- Snapshot/versioning support
- Search with Boyer-Moore-Horspool
- Multiple cursors
- Marks (bookmarks/anchors that track position through edits)
"""

from enum import Enum
from typing import Optional
import bisect


class BufferType(Enum):
    ORIGINAL = 0
    ADD = 1


class Piece:
    """A descriptor referencing a slice of a buffer."""
    __slots__ = ('buffer_type', 'start', 'length', 'line_starts')

    def __init__(self, buffer_type, start, length, line_starts=None):
        self.buffer_type = buffer_type
        self.start = start
        self.length = length
        # Offsets within this piece where newlines occur (positions after \n)
        self.line_starts = line_starts if line_starts is not None else []

    def __repr__(self):
        return f"Piece({self.buffer_type.name}, {self.start}, {self.length})"


class EditCommand:
    """Represents a reversible edit operation."""
    __slots__ = ('kind', 'offset', 'text', 'old_pieces', 'new_pieces',
                 'old_piece_index', 'old_piece_count')

    def __init__(self, kind, offset, text='',
                 old_pieces=None, new_pieces=None,
                 old_piece_index=0, old_piece_count=0):
        self.kind = kind  # 'insert' or 'delete'
        self.offset = offset
        self.text = text
        self.old_pieces = old_pieces or []
        self.new_pieces = new_pieces or []
        self.old_piece_index = old_piece_index
        self.old_piece_count = old_piece_count


class Mark:
    """A bookmark/anchor that tracks position through edits."""
    __slots__ = ('name', 'offset', 'gravity')

    def __init__(self, name, offset, gravity='right'):
        """
        gravity: 'left' means mark stays at left edge of an insert at its position.
                 'right' means mark moves right with inserts at its position.
        """
        self.name = name
        self.offset = offset
        self.gravity = gravity


class Cursor:
    """A cursor with position and optional selection."""
    __slots__ = ('position', 'anchor')

    def __init__(self, position, anchor=None):
        self.position = position
        self.anchor = anchor  # None means no selection; otherwise selection is anchor..position

    @property
    def selection(self):
        if self.anchor is None:
            return None
        lo = min(self.anchor, self.position)
        hi = max(self.anchor, self.position)
        if lo == hi:
            return None
        return (lo, hi)


class PieceTable:
    """
    Piece Table text buffer.

    Maintains original buffer (immutable) + add buffer (append-only).
    Edits modify the piece table, not the text buffers.
    """

    def __init__(self, text=''):
        self._original = text
        self._add = ''
        self._pieces = []
        if text:
            line_starts = self._compute_line_starts(text, 0)
            self._pieces.append(Piece(BufferType.ORIGINAL, 0, len(text), line_starts))
        self._undo_stack = []
        self._redo_stack = []
        self._marks = {}  # name -> Mark
        self._cursors = []  # list of Cursor
        self._version = 0
        self._snapshots = {}  # version -> (pieces_copy, add_buffer_len)
        # Cache
        self._length_cache = len(text)
        self._line_count_cache = None

    # ---- Buffer access ----

    def _get_buffer(self, buffer_type):
        if buffer_type == BufferType.ORIGINAL:
            return self._original
        return self._add

    def _piece_text(self, piece):
        buf = self._get_buffer(piece.buffer_type)
        return buf[piece.start:piece.start + piece.length]

    # ---- Line start computation ----

    def _compute_line_starts(self, text, piece_offset):
        """Find positions within text where new lines start (after each \\n)."""
        starts = []
        for i, ch in enumerate(text):
            if ch == '\n':
                starts.append(i + 1)  # line starts after the newline
        return starts

    # ---- Core: text retrieval ----

    def __len__(self):
        return self._length_cache

    @property
    def length(self):
        return self._length_cache

    def text(self, start=None, end=None):
        """Get text content, optionally a slice."""
        if start is None and end is None:
            return ''.join(self._piece_text(p) for p in self._pieces)
        if start is None:
            start = 0
        if end is None:
            end = self._length_cache
        start = max(0, start)
        end = min(end, self._length_cache)
        if start >= end:
            return ''
        result = []
        offset = 0
        for piece in self._pieces:
            piece_end = offset + piece.length
            if piece_end <= start:
                offset = piece_end
                continue
            if offset >= end:
                break
            slice_start = max(start - offset, 0)
            slice_end = min(end - offset, piece.length)
            buf = self._get_buffer(piece.buffer_type)
            result.append(buf[piece.start + slice_start:piece.start + slice_end])
            offset = piece_end
        return ''.join(result)

    def char_at(self, offset):
        """Get character at offset."""
        if offset < 0 or offset >= self._length_cache:
            raise IndexError(f"offset {offset} out of range [0, {self._length_cache})")
        pos = 0
        for piece in self._pieces:
            if pos + piece.length > offset:
                buf = self._get_buffer(piece.buffer_type)
                return buf[piece.start + (offset - pos)]
            pos += piece.length
        raise IndexError("offset out of range")

    # ---- Core: find piece at offset ----

    def _find_piece(self, offset):
        """
        Find which piece contains the given offset.
        Returns (piece_index, offset_within_piece).
        If offset == total length, returns (len(pieces), 0) for append position.
        """
        pos = 0
        for i, piece in enumerate(self._pieces):
            if offset < pos + piece.length:
                return (i, offset - pos)
            pos += piece.length
        return (len(self._pieces), 0)

    # ---- Core: insert ----

    def insert(self, offset, text):
        """Insert text at the given offset."""
        if not text:
            return
        if offset < 0 or offset > self._length_cache:
            raise IndexError(f"insert offset {offset} out of range [0, {self._length_cache}]")

        # Append to add buffer
        add_start = len(self._add)
        self._add += text
        line_starts = self._compute_line_starts(text, 0)
        new_piece = Piece(BufferType.ADD, add_start, len(text), line_starts)

        # Find insertion point
        pi, po = self._find_piece(offset)

        if po == 0:
            # Insert before piece pi
            old_pieces = []
            new_pieces = [new_piece]
            old_index = pi
            old_count = 0
            self._pieces.insert(pi, new_piece)
        else:
            # Split piece pi at offset po
            piece = self._pieces[pi]
            left_line_starts = [ls for ls in piece.line_starts if ls <= po]
            right_line_starts = [ls - po for ls in piece.line_starts if ls > po]

            left = Piece(piece.buffer_type, piece.start, po, left_line_starts)
            right = Piece(piece.buffer_type, piece.start + po, piece.length - po, right_line_starts)

            old_pieces = [piece]
            new_pieces_list = [left, new_piece]
            if right.length > 0:
                new_pieces_list.append(right)
            new_pieces = new_pieces_list

            old_index = pi
            old_count = 1
            self._pieces[pi:pi + 1] = new_pieces_list

        cmd = EditCommand('insert', offset, text,
                          old_pieces=old_pieces, new_pieces=new_pieces,
                          old_piece_index=old_index, old_piece_count=old_count)
        self._undo_stack.append(cmd)
        self._redo_stack.clear()
        self._length_cache += len(text)
        self._line_count_cache = None
        self._version += 1

        # Update marks
        for mark in self._marks.values():
            if mark.offset > offset or (mark.offset == offset and mark.gravity == 'right'):
                mark.offset += len(text)

        # Update cursors
        for cursor in self._cursors:
            if cursor.position >= offset:
                cursor.position += len(text)
            if cursor.anchor is not None and cursor.anchor >= offset:
                cursor.anchor += len(text)

    # ---- Core: delete ----

    def delete(self, offset, length):
        """Delete 'length' characters starting at offset."""
        if length <= 0:
            return ''
        if offset < 0 or offset + length > self._length_cache:
            raise IndexError(f"delete range [{offset}, {offset + length}) out of bounds [0, {self._length_cache})")

        deleted_text = self.text(offset, offset + length)

        # Find start and end pieces
        start_pi, start_po = self._find_piece(offset)
        end_pi, end_po = self._find_piece(offset + length)

        # Build replacement pieces
        new_pieces = []
        old_index = start_pi
        old_count = end_pi - start_pi
        if end_po > 0 and end_pi < len(self._pieces):
            old_count = end_pi - start_pi + 1

        # Left remainder of start piece
        if start_po > 0:
            piece = self._pieces[start_pi]
            left_ls = [ls for ls in piece.line_starts if ls <= start_po]
            left = Piece(piece.buffer_type, piece.start, start_po, left_ls)
            new_pieces.append(left)
            if start_pi == end_pi and end_po > 0:
                # Delete is within a single piece
                old_count = 1

        # Right remainder of end piece
        if end_po > 0 and end_pi < len(self._pieces):
            piece = self._pieces[end_pi]
            if end_po < piece.length:
                right_ls = [ls - end_po for ls in piece.line_starts if ls > end_po]
                right = Piece(piece.buffer_type, piece.start + end_po, piece.length - end_po, right_ls)
                new_pieces.append(right)
                if end_pi >= start_pi and end_pi - start_pi + 1 > old_count:
                    old_count = end_pi - start_pi + 1

        # Recalculate old_count properly
        if end_pi < len(self._pieces) and end_po > 0:
            old_count = end_pi - start_pi + 1
        elif end_pi <= len(self._pieces) and end_po == 0:
            old_count = end_pi - start_pi
        if start_po > 0 and start_pi == end_pi and end_po > 0:
            old_count = 1

        old_pieces = self._pieces[old_index:old_index + old_count]

        cmd = EditCommand('delete', offset, deleted_text,
                          old_pieces=old_pieces, new_pieces=new_pieces,
                          old_piece_index=old_index, old_piece_count=old_count)
        self._undo_stack.append(cmd)
        self._redo_stack.clear()

        self._pieces[old_index:old_index + old_count] = new_pieces
        self._length_cache -= length
        self._line_count_cache = None
        self._version += 1

        # Update marks
        for mark in self._marks.values():
            if mark.offset > offset + length:
                mark.offset -= length
            elif mark.offset > offset:
                mark.offset = offset

        # Update cursors
        for cursor in self._cursors:
            if cursor.position > offset + length:
                cursor.position -= length
            elif cursor.position > offset:
                cursor.position = offset
            if cursor.anchor is not None:
                if cursor.anchor > offset + length:
                    cursor.anchor -= length
                elif cursor.anchor > offset:
                    cursor.anchor = offset

        return deleted_text

    # ---- Undo / Redo ----

    def undo(self):
        """Undo the last edit. Returns True if something was undone."""
        if not self._undo_stack:
            return False
        cmd = self._undo_stack.pop()
        # Reverse the operation
        idx = cmd.old_piece_index
        new_count = len(cmd.new_pieces)
        self._pieces[idx:idx + new_count] = cmd.old_pieces
        if cmd.kind == 'insert':
            self._length_cache -= len(cmd.text)
            # Reverse mark updates
            for mark in self._marks.values():
                if mark.offset >= cmd.offset + len(cmd.text):
                    mark.offset -= len(cmd.text)
                elif mark.offset > cmd.offset:
                    mark.offset = cmd.offset
        else:  # delete
            self._length_cache += len(cmd.text)
            # Reverse mark updates
            for mark in self._marks.values():
                if mark.offset >= cmd.offset:
                    mark.offset += len(cmd.text)
        self._redo_stack.append(cmd)
        self._line_count_cache = None
        self._version += 1
        return True

    def redo(self):
        """Redo the last undone edit. Returns True if something was redone."""
        if not self._redo_stack:
            return False
        cmd = self._redo_stack.pop()
        idx = cmd.old_piece_index
        self._pieces[idx:idx + cmd.old_piece_count] = cmd.new_pieces
        if cmd.kind == 'insert':
            self._length_cache += len(cmd.text)
            for mark in self._marks.values():
                if mark.offset > cmd.offset or (mark.offset == cmd.offset and mark.gravity == 'right'):
                    mark.offset += len(cmd.text)
        else:  # delete
            self._length_cache -= len(cmd.text)
            for mark in self._marks.values():
                if mark.offset > cmd.offset + len(cmd.text):
                    mark.offset -= len(cmd.text)
                elif mark.offset > cmd.offset:
                    mark.offset = cmd.offset
        self._undo_stack.append(cmd)
        self._line_count_cache = None
        self._version += 1
        return True

    # ---- Line operations ----

    @property
    def line_count(self):
        """Number of lines (text always has at least 1 line)."""
        if self._line_count_cache is not None:
            return self._line_count_cache
        count = 1
        for piece in self._pieces:
            count += len(piece.line_starts)
        self._line_count_cache = count
        return count

    def _build_line_offsets(self):
        """Build array of (global_offset) for each line start."""
        offsets = [0]  # Line 0 starts at offset 0
        pos = 0
        for piece in self._pieces:
            for ls in piece.line_starts:
                offsets.append(pos + ls)
            pos += piece.length
        return offsets

    def line_start(self, line_number):
        """Get the offset where line_number starts (0-indexed)."""
        offsets = self._build_line_offsets()
        if line_number < 0 or line_number >= len(offsets):
            raise IndexError(f"line {line_number} out of range [0, {len(offsets)})")
        return offsets[line_number]

    def line_end(self, line_number):
        """Get the offset where line_number ends (exclusive, includes \\n if present)."""
        offsets = self._build_line_offsets()
        if line_number < 0 or line_number >= len(offsets):
            raise IndexError(f"line {line_number} out of range")
        if line_number + 1 < len(offsets):
            return offsets[line_number + 1]
        return self._length_cache

    def get_line(self, line_number):
        """Get the text of a specific line (without trailing \\n)."""
        start = self.line_start(line_number)
        end = self.line_end(line_number)
        line_text = self.text(start, end)
        if line_text.endswith('\n'):
            line_text = line_text[:-1]
        return line_text

    def get_lines(self):
        """Get all lines as a list."""
        return [self.get_line(i) for i in range(self.line_count)]

    def offset_to_line_col(self, offset):
        """Convert offset to (line, col) tuple (0-indexed)."""
        if offset < 0 or offset > self._length_cache:
            raise IndexError(f"offset {offset} out of range")
        offsets = self._build_line_offsets()
        # Binary search for the line
        line = bisect.bisect_right(offsets, offset) - 1
        col = offset - offsets[line]
        return (line, col)

    def line_col_to_offset(self, line, col):
        """Convert (line, col) to offset."""
        start = self.line_start(line)
        end = self.line_end(line)
        offset = start + col
        if offset > end:
            offset = end
        return offset

    # ---- Search ----

    def find(self, pattern, start=0, end=None):
        """Find first occurrence of pattern. Returns offset or -1."""
        if not pattern:
            return start
        text = self.text(start, end)
        idx = self._bmh_search(text, pattern)
        if idx == -1:
            return -1
        return start + idx

    def find_all(self, pattern, start=0, end=None):
        """Find all occurrences of pattern. Returns list of offsets."""
        if not pattern:
            return []
        results = []
        text = self.text(start, end)
        pos = 0
        while pos <= len(text) - len(pattern):
            idx = self._bmh_search(text[pos:], pattern)
            if idx == -1:
                break
            results.append(start + pos + idx)
            pos += idx + 1
        return results

    def _bmh_search(self, text, pattern):
        """Boyer-Moore-Horspool string search."""
        n, m = len(text), len(pattern)
        if m == 0:
            return 0
        if m > n:
            return -1

        # Bad character table
        skip = {}
        for i in range(m - 1):
            skip[pattern[i]] = m - 1 - i

        i = 0
        while i <= n - m:
            j = m - 1
            while j >= 0 and text[i + j] == pattern[j]:
                j -= 1
            if j < 0:
                return i
            i += skip.get(text[i + m - 1], m)
        return -1

    def replace(self, pattern, replacement, start=0, end=None, count=0):
        """Replace occurrences of pattern with replacement.
        count=0 means replace all. Returns number of replacements made."""
        positions = self.find_all(pattern, start, end)
        if count > 0:
            positions = positions[:count]
        # Replace in reverse order to maintain offset validity
        for pos in reversed(positions):
            self.delete(pos, len(pattern))
            self.insert(pos, replacement)
        return len(positions)

    # ---- Marks ----

    def set_mark(self, name, offset, gravity='right'):
        """Set a named mark at the given offset."""
        if offset < 0 or offset > self._length_cache:
            raise IndexError(f"mark offset {offset} out of range")
        self._marks[name] = Mark(name, offset, gravity)

    def get_mark(self, name):
        """Get mark offset by name. Returns None if not found."""
        mark = self._marks.get(name)
        return mark.offset if mark else None

    def remove_mark(self, name):
        """Remove a mark."""
        return self._marks.pop(name, None) is not None

    def list_marks(self):
        """List all marks as dict of name -> offset."""
        return {name: mark.offset for name, mark in self._marks.items()}

    # ---- Cursors ----

    def add_cursor(self, position, anchor=None):
        """Add a cursor. Returns cursor index."""
        c = Cursor(min(position, self._length_cache), anchor)
        self._cursors.append(c)
        return len(self._cursors) - 1

    def get_cursor(self, index=0):
        """Get cursor position (and anchor) by index."""
        if index >= len(self._cursors):
            return None
        c = self._cursors[index]
        return {'position': c.position, 'anchor': c.anchor, 'selection': c.selection}

    def move_cursor(self, index, position, select=False):
        """Move cursor to position. If select=True, extends selection."""
        if index >= len(self._cursors):
            return
        c = self._cursors[index]
        if select and c.anchor is None:
            c.anchor = c.position
        elif not select:
            c.anchor = None
        c.position = max(0, min(position, self._length_cache))

    def remove_cursor(self, index):
        """Remove cursor by index."""
        if index < len(self._cursors):
            self._cursors.pop(index)

    def cursor_count(self):
        return len(self._cursors)

    # ---- Snapshots ----

    def snapshot(self):
        """Take a snapshot of the current state. Returns version number."""
        pieces_copy = [Piece(p.buffer_type, p.start, p.length, list(p.line_starts))
                       for p in self._pieces]
        self._snapshots[self._version] = (pieces_copy, len(self._add), self._length_cache)
        return self._version

    def restore(self, version):
        """Restore to a previous snapshot."""
        if version not in self._snapshots:
            raise KeyError(f"no snapshot at version {version}")
        pieces_copy, add_len, length = self._snapshots[version]
        self._pieces = [Piece(p.buffer_type, p.start, p.length, list(p.line_starts))
                        for p in pieces_copy]
        self._length_cache = length
        self._line_count_cache = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._version += 1

    def list_snapshots(self):
        """List all snapshot versions."""
        return sorted(self._snapshots.keys())

    # ---- Statistics ----

    def piece_count(self):
        """Number of pieces in the table."""
        return len(self._pieces)

    def buffer_sizes(self):
        """Return sizes of original and add buffers."""
        return {'original': len(self._original), 'add': len(self._add)}

    @property
    def version(self):
        return self._version

    # ---- Merge adjacent pieces ----

    def compact(self):
        """Merge adjacent pieces that reference contiguous buffer regions."""
        if len(self._pieces) <= 1:
            return
        merged = [self._pieces[0]]
        for piece in self._pieces[1:]:
            prev = merged[-1]
            if (prev.buffer_type == piece.buffer_type and
                    prev.start + prev.length == piece.start):
                # Merge
                combined_ls = list(prev.line_starts)
                combined_ls.extend(ls + prev.length for ls in piece.line_starts)
                merged[-1] = Piece(prev.buffer_type, prev.start,
                                   prev.length + piece.length, combined_ls)
            else:
                merged.append(piece)
        self._pieces = merged

    # ---- Iteration ----

    def chars(self, start=0, end=None):
        """Iterate over characters."""
        if end is None:
            end = self._length_cache
        offset = 0
        for piece in self._pieces:
            piece_end = offset + piece.length
            if piece_end <= start:
                offset = piece_end
                continue
            if offset >= end:
                break
            buf = self._get_buffer(piece.buffer_type)
            s = max(start - offset, 0)
            e = min(end - offset, piece.length)
            for i in range(s, e):
                yield buf[piece.start + i]
            offset = piece_end

    def __iter__(self):
        return self.chars()

    def __str__(self):
        return self.text()

    def __repr__(self):
        n = self._length_cache
        pc = len(self._pieces)
        return f"PieceTable(length={n}, pieces={pc}, version={self._version})"

    # ---- Batch operations ----

    def insert_lines(self, line_number, lines):
        """Insert lines before the given line number."""
        if line_number >= self.line_count:
            offset = self._length_cache
        else:
            offset = self.line_start(line_number)
        text = '\n'.join(lines) + '\n'
        self.insert(offset, text)

    def delete_lines(self, start_line, count=1):
        """Delete 'count' lines starting at start_line."""
        if start_line >= self.line_count:
            return ''
        s = self.line_start(start_line)
        end_line = min(start_line + count, self.line_count)
        if end_line >= self.line_count:
            e = self._length_cache
        else:
            e = self.line_start(end_line)
        return self.delete(s, e - s)

    def replace_line(self, line_number, new_text):
        """Replace a specific line's content."""
        self.delete_lines(line_number, 1)
        if line_number >= self.line_count:
            if self._length_cache > 0 and not self.text().endswith('\n'):
                self.insert(self._length_cache, '\n')
            self.insert(self._length_cache, new_text + '\n')
        else:
            offset = self.line_start(line_number)
            self.insert(offset, new_text + '\n')
