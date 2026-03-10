"""
Tests for the diff engine (Challenge C005).
"""

import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from diff_engine import diff, unified_diff, diff_files


def test_identical():
    """Identical inputs produce no diff."""
    a = ["hello", "world"]
    b = ["hello", "world"]
    result = unified_diff(a, b)
    assert result == "", f"Expected empty diff, got: {result}"
    print("PASS: identical files produce no diff")


def test_empty_to_content():
    """Empty file to content = all insertions."""
    a = []
    b = ["line1", "line2", "line3"]
    result = unified_diff(a, b, "old", "new")
    assert "+line1" in result
    assert "+line2" in result
    assert "+line3" in result
    # Body lines should have no deletions (only + and @@ lines after header)
    body_lines = result.split("\n")[3:]  # skip ---, +++, @@
    assert all(not l.startswith("-") for l in body_lines if l.strip())
    print("PASS: empty to content shows all insertions")


def test_content_to_empty():
    """Content to empty file = all deletions."""
    a = ["line1", "line2", "line3"]
    b = []
    result = unified_diff(a, b, "old", "new")
    assert "-line1" in result
    assert "-line2" in result
    assert "-line3" in result
    print("PASS: content to empty shows all deletions")


def test_single_insertion():
    """Insert one line in the middle."""
    a = ["aaa", "ccc"]
    b = ["aaa", "bbb", "ccc"]
    result = unified_diff(a, b, "old", "new")
    assert "+bbb" in result
    assert " aaa" in result  # context
    assert " ccc" in result  # context
    print("PASS: single insertion detected")


def test_single_deletion():
    """Delete one line from the middle."""
    a = ["aaa", "bbb", "ccc"]
    b = ["aaa", "ccc"]
    result = unified_diff(a, b, "old", "new")
    assert "-bbb" in result
    assert " aaa" in result
    assert " ccc" in result
    print("PASS: single deletion detected")


def test_modification():
    """Modify a line (shows as delete + insert)."""
    a = ["aaa", "bbb", "ccc"]
    b = ["aaa", "BBB", "ccc"]
    result = unified_diff(a, b, "old", "new")
    assert "-bbb" in result
    assert "+BBB" in result
    print("PASS: modification shows as delete + insert")


def test_multiple_changes():
    """Multiple scattered changes."""
    a = ["line1", "line2", "line3", "line4", "line5",
         "line6", "line7", "line8", "line9", "line10"]
    b = ["line1", "LINE2", "line3", "line4", "line5",
         "line6", "line7", "LINE8", "line9", "line10"]
    result = unified_diff(a, b, "old", "new", context=1)
    assert "-line2" in result
    assert "+LINE2" in result
    assert "-line8" in result
    assert "+LINE8" in result
    print("PASS: multiple scattered changes detected")


def test_context_lines():
    """Context lines are shown around changes."""
    a = ["a", "b", "c", "d", "e", "f", "g"]
    b = ["a", "b", "c", "D", "e", "f", "g"]
    result = unified_diff(a, b, "old", "new", context=2)
    # Should show 2 context lines before and after the change
    assert " b" in result
    assert " c" in result
    assert "-d" in result
    assert "+D" in result
    assert " e" in result
    assert " f" in result
    print("PASS: context lines shown correctly")


def test_unified_format_header():
    """Output has correct unified diff headers."""
    a = ["old"]
    b = ["new"]
    result = unified_diff(a, b, "file_a.txt", "file_b.txt")
    lines = result.split("\n")
    assert lines[0].startswith("--- file_a.txt")
    assert lines[1].startswith("+++ file_b.txt")
    assert lines[2].startswith("@@")
    print("PASS: unified format headers correct")


def test_hunk_header_ranges():
    """Hunk headers show correct line ranges."""
    a = ["a", "b", "c"]
    b = ["a", "X", "c"]
    result = unified_diff(a, b, "old", "new", context=1)
    # Should have a hunk like @@ -1,3 +1,3 @@ or similar
    assert "@@" in result
    lines = result.split("\n")
    hunk_line = [l for l in lines if l.startswith("@@")][0]
    assert "-" in hunk_line and "+" in hunk_line
    print("PASS: hunk headers have correct format")


def test_diff_files_with_real_files():
    """Test the file-based diff interface."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
        f1.write("hello\nworld\n")
        path_a = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
        f2.write("hello\nearth\n")
        path_b = f2.name

    try:
        result = diff_files(path_a, path_b)
        assert "-world" in result
        assert "+earth" in result
        print("PASS: file-based diff works")
    finally:
        os.unlink(path_a)
        os.unlink(path_b)


def test_large_diff():
    """Handles larger files (100+ lines)."""
    a = [f"line {i}" for i in range(100)]
    b = list(a)
    b[10] = "CHANGED 10"
    b[50] = "CHANGED 50"
    b[90] = "CHANGED 90"
    result = unified_diff(a, b, "old", "new", context=2)
    assert "-line 10" in result
    assert "+CHANGED 10" in result
    assert "-line 50" in result
    assert "+CHANGED 50" in result
    assert "-line 90" in result
    assert "+CHANGED 90" in result
    print("PASS: large file diff works (100 lines, 3 changes)")


def test_raw_diff_output():
    """The raw diff function returns correct edit tags."""
    a = ["a", "b", "c"]
    b = ["a", "c"]
    edits = diff(a, b)
    tags = [(tag, line) for tag, line in edits]
    assert ("equal", "a") in tags
    assert ("delete", "b") in tags
    assert ("equal", "c") in tags
    assert len(tags) == 3
    print("PASS: raw diff returns correct edit script")


def test_append_lines():
    """Lines appended at the end."""
    a = ["a", "b"]
    b = ["a", "b", "c", "d"]
    result = unified_diff(a, b, "old", "new")
    assert "+c" in result
    assert "+d" in result
    print("PASS: appended lines detected")


def test_prepend_lines():
    """Lines prepended at the start."""
    a = ["c", "d"]
    b = ["a", "b", "c", "d"]
    result = unified_diff(a, b, "old", "new")
    assert "+a" in result
    assert "+b" in result
    print("PASS: prepended lines detected")


if __name__ == "__main__":
    tests = [
        test_identical,
        test_empty_to_content,
        test_content_to_empty,
        test_single_insertion,
        test_single_deletion,
        test_modification,
        test_multiple_changes,
        test_context_lines,
        test_unified_format_header,
        test_hunk_header_ranges,
        test_diff_files_with_real_files,
        test_large_diff,
        test_raw_diff_output,
        test_append_lines,
        test_prepend_lines,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} -- {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
