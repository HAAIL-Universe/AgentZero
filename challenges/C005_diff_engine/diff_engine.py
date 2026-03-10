"""
Diff Engine -- AgentZero Challenge C005

Compares two text files and produces unified diff output showing
additions, deletions, and context lines.

Uses Myers' diff algorithm (optimized shortest edit script) for
computing the minimal edit sequence between two files.

Usage:
    python diff_engine.py file_a file_b [--context N]

    # Or as a library:
    from diff_engine import diff, unified_diff
    edits = diff(lines_a, lines_b)
    output = unified_diff(lines_a, lines_b, name_a, name_b, context=3)
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def _myers_diff(a: List[str], b: List[str]) -> List[Tuple[str, str]]:
    """
    Myers' diff algorithm. Returns a list of (tag, line) tuples where
    tag is one of: 'equal', 'delete', 'insert'.

    This finds the shortest edit script (SES) -- the minimum number of
    insertions and deletions to transform a into b.
    """
    n, m = len(a), len(b)

    # Edge cases
    if n == 0:
        return [("insert", line) for line in b]
    if m == 0:
        return [("delete", line) for line in a]

    # Myers algorithm: find shortest edit path through the edit graph.
    # V[k] stores the furthest-reaching x on diagonal k.
    # Diagonal k = x - y. We want to reach diagonal n - m.
    max_d = n + m

    # V arrays for each d (edit distance). We store them for traceback.
    v_history = []
    v = {0: 0}

    found = False
    for d in range(max_d + 1):
        v_history.append(dict(v))
        new_v = {}
        for k in range(-d, d + 1, 2):
            # Decide whether to go down (insert) or right (delete)
            if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
                x = v.get(k + 1, 0)  # move down from diagonal k+1
            else:
                x = v.get(k - 1, 0) + 1  # move right from diagonal k-1

            y = x - k

            # Follow diagonal (equal lines)
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1

            new_v[k] = x

            if x >= n and y >= m:
                v_history.append(dict(new_v))
                found = True
                break

        v = new_v
        if found:
            break

    # Traceback: reconstruct the edit script from v_history
    edits = []
    x, y = n, m

    for d in range(len(v_history) - 2, -1, -1):
        v_d = v_history[d]
        k = x - y

        if k == -d or (k != d and v_d.get(k - 1, -1) < v_d.get(k + 1, -1)):
            prev_k = k + 1
        else:
            prev_k = k - 1

        prev_x = v_d.get(prev_k, 0)
        prev_y = prev_x - prev_k

        # Diagonal moves (equal lines) from (prev_x, prev_y) snake end to (x, y) or intermediate
        while x > prev_x and y > prev_y:
            x -= 1
            y -= 1
            edits.append(("equal", a[x]))

        if d > 0:
            if x > prev_x:
                # Moved right: delete
                x -= 1
                edits.append(("delete", a[x]))
            elif y > prev_y:
                # Moved down: insert
                y -= 1
                edits.append(("insert", b[y]))

    edits.reverse()
    return edits


def diff(a: List[str], b: List[str]) -> List[Tuple[str, str]]:
    """
    Compute the diff between two lists of lines.
    Returns list of (tag, line) where tag is 'equal', 'insert', or 'delete'.
    """
    return _myers_diff(a, b)


def unified_diff(
    a: List[str],
    b: List[str],
    name_a: str = "a",
    name_b: str = "b",
    context: int = 3,
    timestamp_a: str = "",
    timestamp_b: str = "",
) -> str:
    """
    Produce unified diff format output.

    Args:
        a: Lines of original file
        b: Lines of modified file
        name_a: Name for original file
        name_b: Name for modified file
        context: Number of context lines around changes (default 3)
        timestamp_a: Optional timestamp for file a
        timestamp_b: Optional timestamp for file b

    Returns:
        Unified diff string
    """
    edits = diff(a, b)

    if not edits:
        return ""

    # Check if there are any changes at all
    if all(tag == "equal" for tag, _ in edits):
        return ""

    # Build indexed edit list: (tag, line, a_lineno, b_lineno)
    indexed = []
    a_line = 0
    b_line = 0
    for tag, line in edits:
        indexed.append((tag, line, a_line, b_line))
        if tag == "equal":
            a_line += 1
            b_line += 1
        elif tag == "delete":
            a_line += 1
        elif tag == "insert":
            b_line += 1

    # Find change regions and group into hunks with context
    change_indices = [i for i, (tag, _, _, _) in enumerate(indexed) if tag != "equal"]

    if not change_indices:
        return ""

    # Group changes into hunks (merge if context overlaps)
    hunks = []
    hunk_start = max(0, change_indices[0] - context)
    hunk_end = min(len(indexed) - 1, change_indices[0] + context)

    for idx in change_indices[1:]:
        new_start = max(0, idx - context)
        new_end = min(len(indexed) - 1, idx + context)

        if new_start <= hunk_end + 1:
            # Merge with current hunk
            hunk_end = new_end
        else:
            # Close current hunk, start new one
            hunks.append((hunk_start, hunk_end))
            hunk_start = new_start
            hunk_end = new_end

    hunks.append((hunk_start, hunk_end))

    # Format output
    lines = []
    ts_a = f"\t{timestamp_a}" if timestamp_a else ""
    ts_b = f"\t{timestamp_b}" if timestamp_b else ""
    lines.append(f"--- {name_a}{ts_a}")
    lines.append(f"+++ {name_b}{ts_b}")

    for hunk_start, hunk_end in hunks:
        hunk_edits = indexed[hunk_start:hunk_end + 1]

        # Calculate line ranges for the hunk header
        a_start = None
        b_start = None
        a_count = 0
        b_count = 0

        for tag, line, a_ln, b_ln in hunk_edits:
            if a_start is None and tag in ("equal", "delete"):
                a_start = a_ln + 1  # 1-indexed
            if b_start is None and tag in ("equal", "insert"):
                b_start = b_ln + 1  # 1-indexed
            if tag in ("equal", "delete"):
                a_count += 1
            if tag in ("equal", "insert"):
                b_count += 1

        if a_start is None:
            a_start = 0
        if b_start is None:
            b_start = 0

        # Format hunk header
        a_range = f"{a_start},{a_count}" if a_count != 1 else str(a_start)
        b_range = f"{b_start},{b_count}" if b_count != 1 else str(b_start)
        lines.append(f"@@ -{a_range} +{b_range} @@")

        # Format hunk lines
        for tag, line, _, _ in hunk_edits:
            if tag == "equal":
                lines.append(f" {line}")
            elif tag == "delete":
                lines.append(f"-{line}")
            elif tag == "insert":
                lines.append(f"+{line}")

    return "\n".join(lines) + "\n"


def diff_files(path_a: str, path_b: str, context: int = 3) -> str:
    """Diff two files and return unified diff string."""
    file_a = Path(path_a)
    file_b = Path(path_b)

    if not file_a.exists():
        return f"Error: {path_a} not found"
    if not file_b.exists():
        return f"Error: {path_b} not found"

    a_lines = file_a.read_text(encoding="utf-8").splitlines()
    b_lines = file_b.read_text(encoding="utf-8").splitlines()

    a_time = datetime.fromtimestamp(file_a.stat().st_mtime).isoformat()
    b_time = datetime.fromtimestamp(file_b.stat().st_mtime).isoformat()

    return unified_diff(
        a_lines, b_lines,
        name_a=str(file_a),
        name_b=str(file_b),
        context=context,
        timestamp_a=a_time,
        timestamp_b=b_time,
    )


def main():
    if len(sys.argv) < 3:
        print("Usage: python diff_engine.py FILE_A FILE_B [--context N]")
        sys.exit(1)

    path_a = sys.argv[1]
    path_b = sys.argv[2]
    context = 3

    if "--context" in sys.argv:
        idx = sys.argv.index("--context")
        if idx + 1 < len(sys.argv):
            context = int(sys.argv[idx + 1])

    result = diff_files(path_a, path_b, context)
    if result:
        print(result, end="")
    else:
        print("Files are identical.")


if __name__ == "__main__":
    main()
