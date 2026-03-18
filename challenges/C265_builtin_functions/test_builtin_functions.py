"""
Tests for C265: SQL Built-in Functions

Tests: string functions, math functions, null-handling, type conversion,
       CAST syntax, function composition, functions in WHERE/ORDER BY.
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from builtin_functions import BuiltinDB, _builtin_apply, _do_cast, SqlCast


@pytest.fixture
def db():
    d = BuiltinDB()
    d.execute("CREATE TABLE t (id INT, name TEXT, val REAL)")
    d.execute("INSERT INTO t VALUES (1, 'Alice', 10.5)")
    d.execute("INSERT INTO t VALUES (2, 'Bob', 20.3)")
    d.execute("INSERT INTO t VALUES (3, 'Charlie', 30.7)")
    d.execute("INSERT INTO t VALUES (4, 'Diana', NULL)")
    d.execute("INSERT INTO t VALUES (5, NULL, 50.0)")
    return d


@pytest.fixture
def str_db():
    d = BuiltinDB()
    d.execute("CREATE TABLE s (id INT, txt TEXT)")
    d.execute("INSERT INTO s VALUES (1, 'Hello World')")
    d.execute("INSERT INTO s VALUES (2, '  spaces  ')")
    d.execute("INSERT INTO s VALUES (3, 'abcdef')")
    d.execute("INSERT INTO s VALUES (4, NULL)")
    d.execute("INSERT INTO s VALUES (5, 'UPPER lower')")
    return d


# =============================================================================
# String Functions
# =============================================================================

class TestUpper:
    def test_basic(self, db):
        r = db.execute("SELECT UPPER(name) AS u FROM t WHERE id = 1")
        assert r.rows[0][0] == 'ALICE'

    def test_null(self, db):
        r = db.execute("SELECT UPPER(name) AS u FROM t WHERE id = 5")
        assert r.rows[0][0] is None


class TestLower:
    def test_basic(self, db):
        r = db.execute("SELECT LOWER(name) AS l FROM t WHERE id = 1")
        assert r.rows[0][0] == 'alice'

    def test_mixed_case(self, str_db):
        r = str_db.execute("SELECT LOWER(txt) AS l FROM s WHERE id = 5")
        assert r.rows[0][0] == 'upper lower'


class TestLength:
    def test_basic(self, db):
        r = db.execute("SELECT LENGTH(name) AS l FROM t WHERE id = 1")
        assert r.rows[0][0] == 5

    def test_null(self, db):
        r = db.execute("SELECT LENGTH(name) AS l FROM t WHERE id = 5")
        assert r.rows[0][0] is None


class TestTrim:
    def test_basic(self, str_db):
        r = str_db.execute("SELECT TRIM(txt) AS t FROM s WHERE id = 2")
        assert r.rows[0][0] == 'spaces'

    def test_ltrim(self, str_db):
        r = str_db.execute("SELECT LTRIM(txt) AS t FROM s WHERE id = 2")
        assert r.rows[0][0] == 'spaces  '

    def test_rtrim(self, str_db):
        r = str_db.execute("SELECT RTRIM(txt) AS t FROM s WHERE id = 2")
        assert r.rows[0][0] == '  spaces'


class TestSubstring:
    def test_basic(self, str_db):
        r = str_db.execute("SELECT SUBSTRING(txt, 1, 5) AS s FROM s WHERE id = 1")
        assert r.rows[0][0] == 'Hello'

    def test_from_middle(self, str_db):
        r = str_db.execute("SELECT SUBSTRING(txt, 7) AS s FROM s WHERE id = 1")
        assert r.rows[0][0] == 'World'

    def test_substr_alias(self, str_db):
        r = str_db.execute("SELECT SUBSTR(txt, 1, 3) AS s FROM s WHERE id = 3")
        assert r.rows[0][0] == 'abc'

    def test_null(self, str_db):
        r = str_db.execute("SELECT SUBSTRING(txt, 1, 3) AS s FROM s WHERE id = 4")
        assert r.rows[0][0] is None


class TestReplace:
    def test_basic(self, str_db):
        r = str_db.execute("SELECT REPLACE(txt, 'World', 'SQL') AS r FROM s WHERE id = 1")
        assert r.rows[0][0] == 'Hello SQL'

    def test_remove(self, str_db):
        r = str_db.execute("SELECT REPLACE(txt, ' ', '') AS r FROM s WHERE id = 1")
        assert r.rows[0][0] == 'HelloWorld'


class TestConcat:
    def test_basic(self, db):
        r = db.execute("SELECT CONCAT(name, ' - ', name) AS c FROM t WHERE id = 1")
        assert r.rows[0][0] == 'Alice - Alice'

    def test_with_null(self, db):
        # CONCAT skips NULLs
        r = db.execute("SELECT CONCAT(name, '-suffix') AS c FROM t WHERE id = 5")
        assert r.rows[0][0] == '-suffix'


class TestConcatWs:
    def test_basic(self):
        result = _builtin_apply('CONCAT_WS', [', ', 'a', 'b', 'c'])
        assert result == 'a, b, c'

    def test_with_nulls(self):
        result = _builtin_apply('CONCAT_WS', ['-', 'a', None, 'c'])
        assert result == 'a-c'


class TestReverse:
    def test_basic(self, str_db):
        r = str_db.execute("SELECT REVERSE(txt) AS r FROM s WHERE id = 3")
        assert r.rows[0][0] == 'fedcba'

    def test_null(self, str_db):
        r = str_db.execute("SELECT REVERSE(txt) AS r FROM s WHERE id = 4")
        assert r.rows[0][0] is None


class TestRepeat:
    def test_basic(self):
        assert _builtin_apply('REPEAT', ['ab', 3]) == 'ababab'

    def test_null(self):
        assert _builtin_apply('REPEAT', [None, 3]) is None


class TestLpad:
    def test_basic(self):
        assert _builtin_apply('LPAD', ['hi', 5]) == '   hi'

    def test_with_char(self):
        assert _builtin_apply('LPAD', ['hi', 5, '0']) == '000hi'

    def test_longer(self):
        assert _builtin_apply('LPAD', ['hello', 3]) == 'hel'


class TestRpad:
    def test_basic(self):
        assert _builtin_apply('RPAD', ['hi', 5]) == 'hi   '

    def test_with_char(self):
        assert _builtin_apply('RPAD', ['hi', 5, '0']) == 'hi000'


class TestPosition:
    def test_basic(self):
        assert _builtin_apply('POSITION', ['lo', 'Hello']) == 4

    def test_not_found(self):
        assert _builtin_apply('POSITION', ['xyz', 'Hello']) == 0


class TestInstr:
    def test_basic(self):
        assert _builtin_apply('INSTR', ['Hello', 'lo']) == 4

    def test_not_found(self):
        assert _builtin_apply('INSTR', ['Hello', 'xyz']) == 0


class TestLeftRight:
    def test_left(self, str_db):
        r = str_db.execute("SELECT LEFT(txt, 5) AS l FROM s WHERE id = 1")
        assert r.rows[0][0] == 'Hello'

    def test_right(self, str_db):
        r = str_db.execute("SELECT RIGHT(txt, 5) AS r FROM s WHERE id = 1")
        assert r.rows[0][0] == 'World'


class TestStartsEndsWith:
    def test_starts(self):
        assert _builtin_apply('STARTS_WITH', ['Hello World', 'Hello']) == 1
        assert _builtin_apply('STARTS_WITH', ['Hello World', 'World']) == 0

    def test_ends(self):
        assert _builtin_apply('ENDS_WITH', ['Hello World', 'World']) == 1
        assert _builtin_apply('ENDS_WITH', ['Hello World', 'Hello']) == 0


class TestAsciiChar:
    def test_ascii(self):
        assert _builtin_apply('ASCII', ['A']) == 65

    def test_char(self):
        assert _builtin_apply('CHAR', [65]) == 'A'
        assert _builtin_apply('CHR', [97]) == 'a'


class TestInitcap:
    def test_basic(self):
        assert _builtin_apply('INITCAP', ['hello world']) == 'Hello World'


class TestTranslate:
    def test_basic(self):
        assert _builtin_apply('TRANSLATE', ['hello', 'el', 'ip']) == 'hippo'


# =============================================================================
# Math Functions
# =============================================================================

class TestAbs:
    def test_positive(self, db):
        r = db.execute("SELECT ABS(val) AS a FROM t WHERE id = 1")
        assert r.rows[0][0] == 10.5

    def test_negative(self):
        assert _builtin_apply('ABS', [-42]) == 42


class TestRound:
    def test_basic(self, db):
        r = db.execute("SELECT ROUND(val) AS r FROM t WHERE id = 1")
        assert r.rows[0][0] == 10.0

    def test_decimals(self):
        assert _builtin_apply('ROUND', [3.14159, 2]) == 3.14

    def test_null(self):
        assert _builtin_apply('ROUND', [None]) is None


class TestCeilFloor:
    def test_ceil(self):
        assert _builtin_apply('CEIL', [3.2]) == 4
        assert _builtin_apply('CEILING', [3.2]) == 4

    def test_floor(self):
        assert _builtin_apply('FLOOR', [3.7]) == 3


class TestPower:
    def test_basic(self):
        assert _builtin_apply('POWER', [2, 10]) == 1024.0
        assert _builtin_apply('POW', [3, 2]) == 9.0

    def test_null(self):
        assert _builtin_apply('POWER', [None, 2]) is None


class TestSqrt:
    def test_basic(self):
        assert _builtin_apply('SQRT', [16]) == 4.0

    def test_negative(self):
        assert _builtin_apply('SQRT', [-1]) is None


class TestMod:
    def test_basic(self):
        assert _builtin_apply('MOD', [10, 3]) == 1

    def test_div_zero(self):
        assert _builtin_apply('MOD', [10, 0]) is None


class TestSign:
    def test_positive(self):
        assert _builtin_apply('SIGN', [42]) == 1

    def test_negative(self):
        assert _builtin_apply('SIGN', [-7]) == -1

    def test_zero(self):
        assert _builtin_apply('SIGN', [0]) == 0


class TestLog:
    def test_natural(self):
        assert abs(_builtin_apply('LN', [math.e]) - 1.0) < 1e-10

    def test_log10(self):
        assert abs(_builtin_apply('LOG10', [100]) - 2.0) < 1e-10

    def test_log2(self):
        assert abs(_builtin_apply('LOG2', [8]) - 3.0) < 1e-10

    def test_log_with_base(self):
        assert abs(_builtin_apply('LOG', [8, 2]) - 3.0) < 1e-10

    def test_log_negative(self):
        assert _builtin_apply('LN', [-1]) is None


class TestExp:
    def test_basic(self):
        assert abs(_builtin_apply('EXP', [0]) - 1.0) < 1e-10
        assert abs(_builtin_apply('EXP', [1]) - math.e) < 1e-10


class TestPi:
    def test_basic(self):
        assert abs(_builtin_apply('PI', []) - math.pi) < 1e-10


class TestGreatestLeast:
    def test_greatest(self):
        assert _builtin_apply('GREATEST', [1, 5, 3]) == 5

    def test_least(self):
        assert _builtin_apply('LEAST', [1, 5, 3]) == 1

    def test_with_null(self):
        assert _builtin_apply('GREATEST', [1, None, 3]) == 3
        assert _builtin_apply('LEAST', [None, 5, 3]) == 3


class TestTruncate:
    def test_basic(self):
        assert _builtin_apply('TRUNCATE', [3.7]) == 3
        assert _builtin_apply('TRUNC', [3.7]) == 3

    def test_with_decimals(self):
        assert _builtin_apply('TRUNCATE', [3.14159, 2]) == 3.14


class TestTrig:
    def test_sin(self):
        assert abs(_builtin_apply('SIN', [0]) - 0.0) < 1e-10

    def test_cos(self):
        assert abs(_builtin_apply('COS', [0]) - 1.0) < 1e-10

    def test_tan(self):
        assert abs(_builtin_apply('TAN', [0]) - 0.0) < 1e-10

    def test_asin(self):
        assert abs(_builtin_apply('ASIN', [1]) - math.pi / 2) < 1e-10

    def test_acos(self):
        assert abs(_builtin_apply('ACOS', [1]) - 0.0) < 1e-10

    def test_atan(self):
        assert abs(_builtin_apply('ATAN', [0]) - 0.0) < 1e-10

    def test_atan2(self):
        assert abs(_builtin_apply('ATAN2', [1, 1]) - math.pi / 4) < 1e-10

    def test_asin_out_of_range(self):
        assert _builtin_apply('ASIN', [2]) is None


class TestDegreesRadians:
    def test_degrees(self):
        assert abs(_builtin_apply('DEGREES', [math.pi]) - 180.0) < 1e-10

    def test_radians(self):
        assert abs(_builtin_apply('RADIANS', [180]) - math.pi) < 1e-10


class TestRandom:
    def test_basic(self):
        val = _builtin_apply('RANDOM', [])
        assert 0.0 <= val <= 1.0


# =============================================================================
# Null-handling Functions
# =============================================================================

class TestCoalesce:
    def test_basic(self, db):
        r = db.execute("SELECT COALESCE(name, 'unknown') AS n FROM t WHERE id = 5")
        assert r.rows[0][0] == 'unknown'

    def test_first_non_null(self):
        assert _builtin_apply('COALESCE', [None, None, 'x']) == 'x'

    def test_all_null(self):
        assert _builtin_apply('COALESCE', [None, None]) is None


class TestNullif:
    def test_equal(self):
        assert _builtin_apply('NULLIF', [1, 1]) is None

    def test_not_equal(self):
        assert _builtin_apply('NULLIF', [1, 2]) == 1


class TestIfnull:
    def test_null(self):
        assert _builtin_apply('IFNULL', [None, 'default']) == 'default'

    def test_not_null(self):
        assert _builtin_apply('IFNULL', ['value', 'default']) == 'value'

    def test_nvl(self):
        assert _builtin_apply('NVL', [None, 0]) == 0


class TestIif:
    def test_true(self):
        assert _builtin_apply('IIF', [1, 'yes', 'no']) == 'yes'

    def test_false(self):
        assert _builtin_apply('IIF', [0, 'yes', 'no']) == 'no'

    def test_null(self):
        assert _builtin_apply('IIF', [None, 'yes', 'no']) == 'no'


# =============================================================================
# Type Functions
# =============================================================================

class TestTypeof:
    def test_integer(self):
        assert _builtin_apply('TYPEOF', [42]) == 'integer'

    def test_real(self):
        assert _builtin_apply('TYPEOF', [3.14]) == 'real'

    def test_text(self):
        assert _builtin_apply('TYPEOF', ['hello']) == 'text'

    def test_null(self):
        assert _builtin_apply('TYPEOF', [None]) == 'null'


class TestCastFunction:
    def test_to_int(self):
        assert _do_cast('42', 'INTEGER') == 42

    def test_to_float(self):
        assert _do_cast('3.14', 'REAL') == 3.14

    def test_to_text(self):
        assert _do_cast(42, 'TEXT') == '42'

    def test_to_bool_true(self):
        assert _do_cast('true', 'BOOLEAN') is True

    def test_to_bool_false(self):
        assert _do_cast('false', 'BOOLEAN') is False

    def test_null(self):
        assert _do_cast(None, 'INTEGER') is None

    def test_float_to_int(self):
        assert _do_cast(3.7, 'INTEGER') == 3


class TestCastSyntax:
    def test_cast_as_int(self, db):
        r = db.execute("SELECT CAST(val AS INTEGER) AS v FROM t WHERE id = 1")
        assert r.rows[0][0] == 10

    def test_cast_as_text(self, db):
        r = db.execute("SELECT CAST(id AS TEXT) AS v FROM t WHERE id = 2")
        assert r.rows[0][0] == '2'


class TestPrintf:
    def test_basic(self):
        assert _builtin_apply('PRINTF', ['%d items', 5]) == '5 items'

    def test_float(self):
        assert _builtin_apply('PRINTF', ['%.2f', 3.14159]) == '3.14'


# =============================================================================
# Functions in SQL Context (integration)
# =============================================================================

class TestFunctionsInWhere:
    def test_upper_in_where(self, db):
        r = db.execute("SELECT id FROM t WHERE UPPER(name) = 'ALICE'")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_length_in_where(self, db):
        r = db.execute("SELECT id, name FROM t WHERE LENGTH(name) > 4")
        ids = [row[0] for row in r.rows]
        assert 1 in ids  # Alice (5)
        assert 3 in ids  # Charlie (7)
        assert 4 in ids  # Diana (5)

    def test_coalesce_in_where(self, db):
        r = db.execute("SELECT id FROM t WHERE COALESCE(name, 'NONE') = 'NONE'")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 5


class TestFunctionsInOrderBy:
    def test_order_by_length(self, db):
        r = db.execute("SELECT name FROM t WHERE name IS NOT NULL ORDER BY LENGTH(name)")
        names = [row[0] for row in r.rows]
        assert names[0] == 'Bob'  # 3
        # Alice (5) and Diana (5) before Charlie (7)
        assert names[-1] == 'Charlie'  # 7


class TestNestedFunctions:
    def test_upper_of_substring(self, str_db):
        r = str_db.execute("SELECT UPPER(SUBSTRING(txt, 1, 5)) AS u FROM s WHERE id = 1")
        assert r.rows[0][0] == 'HELLO'

    def test_concat_with_upper(self, db):
        r = db.execute("SELECT CONCAT(UPPER(name), '!') AS c FROM t WHERE id = 1")
        assert r.rows[0][0] == 'ALICE!'

    def test_coalesce_with_upper(self, db):
        r = db.execute("SELECT UPPER(COALESCE(name, 'none')) AS u FROM t WHERE id = 5")
        assert r.rows[0][0] == 'NONE'

    def test_round_of_sqrt(self):
        assert _builtin_apply('ROUND', [_builtin_apply('SQRT', [2]), 4]) == 1.4142


class TestFunctionComposition:
    def test_lpad_with_cast(self):
        result = _builtin_apply('LPAD', [str(42), 5, '0'])
        assert result == '00042'

    def test_greatest_with_abs(self):
        result = _builtin_apply('GREATEST', [
            _builtin_apply('ABS', [-5]),
            _builtin_apply('ABS', [-3]),
            _builtin_apply('ABS', [-8]),
        ])
        assert result == 8


class TestSelectExpression:
    """Test SELECT without FROM (pure expression evaluation)."""

    def test_select_literal(self):
        db = BuiltinDB()
        r = db.execute("SELECT 1 AS one")
        assert r.rows[0][0] == 1

    def test_select_function(self):
        db = BuiltinDB()
        r = db.execute("SELECT UPPER('hello') AS u")
        assert r.rows[0][0] == 'HELLO'

    def test_select_math(self):
        db = BuiltinDB()
        r = db.execute("SELECT ROUND(3.14159, 2) AS r")
        assert r.rows[0][0] == 3.14

    def test_select_coalesce(self):
        db = BuiltinDB()
        r = db.execute("SELECT COALESCE(NULL, 'fallback') AS c")
        assert r.rows[0][0] == 'fallback'


class TestMultipleFunctionsInQuery:
    def test_multiple_columns(self, db):
        r = db.execute("SELECT UPPER(name) AS u, LENGTH(name) AS l, ROUND(val) AS v FROM t WHERE id = 1")
        assert r.rows[0] == ['ALICE', 5, 10.0]

    def test_function_with_alias(self, db):
        r = db.execute("SELECT LOWER(name) AS lowered FROM t WHERE id = 2")
        assert r.columns == ['lowered']
        assert r.rows[0][0] == 'bob'


class TestEdgeCases:
    def test_null_propagation(self):
        """Most functions return NULL when given NULL input."""
        null_fns = ['UPPER', 'LOWER', 'LENGTH', 'TRIM', 'REVERSE', 'ABS',
                     'ROUND', 'CEIL', 'FLOOR', 'SQRT', 'SIGN', 'LN', 'EXP',
                     'SIN', 'COS', 'TAN']
        for fn in null_fns:
            result = _builtin_apply(fn, [None])
            assert result is None, f"{fn}(NULL) should be NULL, got {result}"

    def test_unknown_function(self):
        """Unknown functions should return sentinel."""
        from builtin_functions import _UNKNOWN_FUNC
        result = _builtin_apply('NONEXISTENT', [1])
        assert result is _UNKNOWN_FUNC

    def test_substring_bounds(self):
        """SUBSTRING with out-of-bounds start."""
        assert _builtin_apply('SUBSTRING', ['abc', 5]) == ''
        assert _builtin_apply('SUBSTRING', ['abc', 1, 100]) == 'abc'

    def test_left_right_bounds(self):
        assert _builtin_apply('LEFT', ['hi', 100]) == 'hi'
        assert _builtin_apply('RIGHT', ['hi', 100]) == 'hi'
        assert _builtin_apply('RIGHT', ['hi', 0]) == ''

    def test_repeat_zero(self):
        assert _builtin_apply('REPEAT', ['x', 0]) == ''

    def test_empty_string_functions(self):
        assert _builtin_apply('LENGTH', ['']) == 0
        assert _builtin_apply('UPPER', ['']) == ''
        assert _builtin_apply('REVERSE', ['']) == ''

    def test_greatest_all_null(self):
        assert _builtin_apply('GREATEST', [None, None]) is None

    def test_least_all_null(self):
        assert _builtin_apply('LEAST', [None, None]) is None


class TestWindowFunctionsStillWork:
    """Ensure window functions from C264 still work through BuiltinDB."""

    def test_row_number(self, db):
        r = db.execute("SELECT name, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t WHERE name IS NOT NULL")
        assert len(r.rows) == 4
        # Row numbers should be 1-4
        rns = [row[1] for row in r.rows]
        assert sorted(rns) == [1, 2, 3, 4]

    def test_rank(self, db):
        r = db.execute("SELECT name, RANK() OVER (ORDER BY val) AS rnk FROM t WHERE val IS NOT NULL")
        assert len(r.rows) == 4


class TestExistingFeatures:
    """Ensure existing SQL features (joins, aggregates, etc.) still work."""

    def test_join(self):
        db = BuiltinDB()
        db.execute("CREATE TABLE a (id INT, x TEXT)")
        db.execute("CREATE TABLE b (id INT, y TEXT)")
        db.execute("INSERT INTO a VALUES (1, 'foo')")
        db.execute("INSERT INTO b VALUES (1, 'bar')")
        r = db.execute("SELECT a.x, b.y FROM a JOIN b ON a.id = b.id")
        assert r.rows[0] == ['foo', 'bar']

    def test_aggregate(self, db):
        r = db.execute("SELECT COUNT(*) AS cnt FROM t")
        assert r.rows[0][0] == 5

    def test_group_by(self):
        db = BuiltinDB()
        db.execute("CREATE TABLE g (cat TEXT, val INT)")
        db.execute("INSERT INTO g VALUES ('a', 1)")
        db.execute("INSERT INTO g VALUES ('a', 2)")
        db.execute("INSERT INTO g VALUES ('b', 3)")
        r = db.execute("SELECT cat, SUM(val) AS s FROM g GROUP BY cat ORDER BY cat")
        assert r.rows[0] == ['a', 3]
        assert r.rows[1] == ['b', 3]

    def test_insert_update_delete(self):
        db = BuiltinDB()
        db.execute("CREATE TABLE crud (id INT, name TEXT)")
        db.execute("INSERT INTO crud VALUES (1, 'x')")
        r = db.execute("SELECT name FROM crud WHERE id = 1")
        assert r.rows[0][0] == 'x'
        db.execute("UPDATE crud SET name = 'y' WHERE id = 1")
        r = db.execute("SELECT name FROM crud WHERE id = 1")
        assert r.rows[0][0] == 'y'
        db.execute("DELETE FROM crud WHERE id = 1")
        r = db.execute("SELECT COUNT(*) AS c FROM crud")
        assert r.rows[0][0] == 0

    def test_create_table_as_select(self):
        db = BuiltinDB()
        db.execute("CREATE TABLE src (id INT, name TEXT)")
        db.execute("INSERT INTO src VALUES (1, 'a')")
        db.execute("INSERT INTO src VALUES (2, 'b')")
        db.execute("CREATE TABLE dst AS SELECT * FROM src")
        r = db.execute("SELECT COUNT(*) AS c FROM dst")
        assert r.rows[0][0] == 2


# =============================================================================
# Direct unit tests for _builtin_apply
# =============================================================================

class TestBuiltinApplyDirect:
    """Test the raw function dispatch without SQL parsing."""

    def test_all_string_fns_exist(self):
        fns = ['UPPER', 'LOWER', 'LENGTH', 'TRIM', 'LTRIM', 'RTRIM',
               'SUBSTRING', 'REPLACE', 'CONCAT', 'CONCAT_WS', 'REVERSE',
               'REPEAT', 'LPAD', 'RPAD', 'POSITION', 'INSTR', 'LEFT',
               'RIGHT', 'STARTS_WITH', 'ENDS_WITH', 'ASCII', 'CHAR',
               'INITCAP', 'TRANSLATE']
        from builtin_functions import _UNKNOWN_FUNC
        for fn in fns:
            # With appropriate args, should not return sentinel
            if fn in ('REPLACE', 'CONCAT_WS', 'STARTS_WITH', 'ENDS_WITH', 'TRANSLATE'):
                result = _builtin_apply(fn, ['abc', 'a', 'x'])
            elif fn in ('SUBSTRING', 'REPEAT', 'LPAD', 'RPAD', 'LEFT', 'RIGHT'):
                result = _builtin_apply(fn, ['abc', 2])
            elif fn in ('POSITION', 'INSTR'):
                result = _builtin_apply(fn, ['abc', 'b'])
            elif fn in ('CONCAT',):
                result = _builtin_apply(fn, ['a', 'b'])
            elif fn in ('ASCII', 'CHAR'):
                result = _builtin_apply(fn, [65])
            else:
                result = _builtin_apply(fn, ['hello'])
            assert result is not _UNKNOWN_FUNC, f"{fn} not recognized"

    def test_all_math_fns_exist(self):
        fns = ['ABS', 'ROUND', 'CEIL', 'CEILING', 'FLOOR', 'POWER', 'POW',
               'SQRT', 'MOD', 'SIGN', 'LOG', 'LOG2', 'LOG10', 'LN', 'EXP',
               'PI', 'GREATEST', 'LEAST', 'TRUNCATE', 'TRUNC',
               'DEGREES', 'RADIANS', 'SIN', 'COS', 'TAN', 'ASIN', 'ACOS',
               'ATAN', 'ATAN2']
        from builtin_functions import _UNKNOWN_FUNC
        for fn in fns:
            if fn in ('POWER', 'POW', 'MOD', 'ATAN2'):
                result = _builtin_apply(fn, [1, 1])
            elif fn in ('GREATEST', 'LEAST'):
                result = _builtin_apply(fn, [1, 2])
            elif fn == 'PI':
                result = _builtin_apply(fn, [])
            else:
                result = _builtin_apply(fn, [1])
            assert result is not _UNKNOWN_FUNC, f"{fn} not recognized"

    def test_all_null_fns_exist(self):
        fns = ['COALESCE', 'NULLIF', 'IFNULL', 'NVL', 'IIF']
        from builtin_functions import _UNKNOWN_FUNC
        for fn in fns:
            if fn == 'IIF':
                result = _builtin_apply(fn, [1, 'a', 'b'])
            elif fn == 'COALESCE':
                result = _builtin_apply(fn, [1])
            else:
                result = _builtin_apply(fn, [1, 2])
            assert result is not _UNKNOWN_FUNC, f"{fn} not recognized"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
