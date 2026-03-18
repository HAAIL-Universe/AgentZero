"""
Tests for C268: SQL Set Operations (UNION, INTERSECT, EXCEPT)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C267_common_table_expressions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C266_subqueries'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C265_builtin_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C264_window_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C263_ctas'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C262_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C261_foreign_keys'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from set_operations import SetOpDB, SetOpParser, SetOperation, SetOpWithClauses, parse_set_op_sql
from mini_database import ParseError, CompileError, ResultSet


@pytest.fixture
def db():
    """Create a database with test data."""
    d = SetOpDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary INT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 85000)")
    d.execute("INSERT INTO employees VALUES (3, 'Carol', 'Sales', 75000)")
    d.execute("INSERT INTO employees VALUES (4, 'Dave', 'Sales', 80000)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 'Marketing', 70000)")

    d.execute("CREATE TABLE contractors (id INT, name TEXT, dept TEXT, rate INT)")
    d.execute("INSERT INTO contractors VALUES (10, 'Frank', 'Engineering', 100)")
    d.execute("INSERT INTO contractors VALUES (11, 'Grace', 'Sales', 90)")
    d.execute("INSERT INTO contractors VALUES (12, 'Alice', 'Engineering', 110)")  # same name as employee

    d.execute("CREATE TABLE products (id INT, name TEXT, category TEXT, price INT)")
    d.execute("INSERT INTO products VALUES (1, 'Widget', 'A', 10)")
    d.execute("INSERT INTO products VALUES (2, 'Gadget', 'A', 20)")
    d.execute("INSERT INTO products VALUES (3, 'Doohickey', 'B', 15)")
    d.execute("INSERT INTO products VALUES (4, 'Thingamajig', 'B', 25)")
    d.execute("INSERT INTO products VALUES (5, 'Widget', 'A', 10)")  # duplicate

    return d


# =====================================================================
# UNION Tests (top-level, extending C267's CTE-internal UNION)
# =====================================================================

class TestUnion:
    """Top-level UNION operations."""

    def test_basic_union(self, db):
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Engineering'
            UNION
            SELECT name FROM contractors WHERE dept = 'Engineering'
        """)
        # Engineering employees = Alice, Bob; Engineering contractors = Frank, Alice
        # UNION deduplicates: Alice, Bob, Frank
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Bob', 'Frank']

    def test_union_all(self, db):
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Engineering'
            UNION ALL
            SELECT name FROM contractors WHERE dept = 'Engineering'
        """)
        names = sorted(r[0] for r in r.rows)
        # Alice (emp), Bob (emp), Frank (cont), Alice (cont) = 4 rows
        assert len(names) == 4
        assert names.count('Alice') == 2

    def test_union_removes_duplicates(self, db):
        r = db.execute("""
            SELECT name, category, price FROM products WHERE name = 'Widget'
            UNION
            SELECT name, category, price FROM products WHERE category = 'A'
        """)
        # Widget appears twice in products but UNION deduplicates
        # Left: (Widget, A, 10), (Widget, A, 10) -> deduplicated to (Widget, A, 10)
        # Right: (Widget, A, 10), (Gadget, A, 20), (Widget, A, 10) -> Widget, Gadget
        # UNION of all: (Widget, A, 10), (Gadget, A, 20)
        assert len(r.rows) == 2

    def test_union_all_preserves_duplicates(self, db):
        r = db.execute("""
            SELECT name FROM products WHERE category = 'A'
            UNION ALL
            SELECT name FROM products WHERE category = 'A'
        """)
        # Category A: Widget, Gadget, Widget (3 rows) x 2 = 6 rows
        assert len(r.rows) == 6

    def test_union_different_column_names(self, db):
        """UNION uses left query's column names."""
        r = db.execute("""
            SELECT name AS employee_name FROM employees WHERE id = 1
            UNION
            SELECT name FROM contractors WHERE id = 10
        """)
        assert r.columns[0] == 'employee_name'
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Frank']

    def test_union_with_literals(self, db):
        r = db.execute("""
            SELECT 1, 'hello'
            UNION
            SELECT 2, 'world'
        """)
        assert len(r.rows) == 2

    def test_union_single_column(self, db):
        r = db.execute("""
            SELECT dept FROM employees
            UNION
            SELECT dept FROM contractors
        """)
        depts = sorted(r[0] for r in r.rows)
        assert depts == ['Engineering', 'Marketing', 'Sales']


# =====================================================================
# INTERSECT Tests
# =====================================================================

class TestIntersect:
    """INTERSECT operations."""

    def test_basic_intersect(self, db):
        """Names that appear in both employees and contractors."""
        r = db.execute("""
            SELECT name FROM employees
            INTERSECT
            SELECT name FROM contractors
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Alice']  # Only Alice is in both

    def test_intersect_no_overlap(self, db):
        """INTERSECT with no common rows returns empty."""
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Marketing'
            INTERSECT
            SELECT name FROM contractors
        """)
        assert len(r.rows) == 0

    def test_intersect_full_overlap(self, db):
        """INTERSECT where all rows match."""
        r = db.execute("""
            SELECT dept FROM employees WHERE dept = 'Engineering'
            INTERSECT
            SELECT dept FROM contractors WHERE dept = 'Engineering'
        """)
        depts = [r[0] for r in r.rows]
        assert depts == ['Engineering']

    def test_intersect_removes_duplicates(self, db):
        """INTERSECT implicitly deduplicates."""
        r = db.execute("""
            SELECT dept FROM employees
            INTERSECT
            SELECT dept FROM contractors
        """)
        # employees: Engineering, Engineering, Sales, Sales, Marketing
        # contractors: Engineering, Sales, Engineering
        # INTERSECT: Engineering, Sales (deduplicated)
        depts = sorted(r[0] for r in r.rows)
        assert depts == ['Engineering', 'Sales']

    def test_intersect_all(self, db):
        """INTERSECT ALL preserves multiplicity."""
        r = db.execute("""
            SELECT dept FROM employees
            INTERSECT ALL
            SELECT dept FROM contractors
        """)
        # employees dept counts: Engineering=2, Sales=2, Marketing=1
        # contractors dept counts: Engineering=2, Sales=1
        # INTERSECT ALL: Engineering=min(2,2)=2, Sales=min(2,1)=1
        depts = sorted(r[0] for r in r.rows)
        assert depts == ['Engineering', 'Engineering', 'Sales']

    def test_intersect_multiple_columns(self, db):
        """INTERSECT on multi-column rows."""
        r = db.execute("""
            SELECT name, dept FROM employees
            INTERSECT
            SELECT name, dept FROM contractors
        """)
        # Only (Alice, Engineering) appears in both
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 'Engineering'

    def test_intersect_with_where(self, db):
        r = db.execute("""
            SELECT name FROM employees WHERE salary > 80000
            INTERSECT
            SELECT name FROM contractors WHERE rate > 100
        """)
        # employees salary>80000: Alice(90k), Bob(85k)
        # contractors rate>100: Alice(110)
        # INTERSECT: Alice
        names = [r[0] for r in r.rows]
        assert names == ['Alice']

    def test_intersect_empty_left(self, db):
        r = db.execute("""
            SELECT name FROM employees WHERE salary > 200000
            INTERSECT
            SELECT name FROM contractors
        """)
        assert len(r.rows) == 0

    def test_intersect_empty_right(self, db):
        r = db.execute("""
            SELECT name FROM employees
            INTERSECT
            SELECT name FROM contractors WHERE rate > 999
        """)
        assert len(r.rows) == 0

    def test_intersect_preserves_left_order(self, db):
        """INTERSECT preserves order from left operand."""
        r = db.execute("""
            SELECT name FROM employees
            INTERSECT
            SELECT name FROM contractors
        """)
        # Alice is first in employees, and is the only intersection
        assert r.rows[0][0] == 'Alice'


# =====================================================================
# EXCEPT Tests
# =====================================================================

class TestExcept:
    """EXCEPT operations."""

    def test_basic_except(self, db):
        """Employee names not in contractors."""
        r = db.execute("""
            SELECT name FROM employees
            EXCEPT
            SELECT name FROM contractors
        """)
        names = sorted(r[0] for r in r.rows)
        # employees: Alice, Bob, Carol, Dave, Eve
        # contractors: Frank, Grace, Alice
        # EXCEPT: Bob, Carol, Dave, Eve
        assert names == ['Bob', 'Carol', 'Dave', 'Eve']

    def test_except_all_removed(self, db):
        """EXCEPT where all left rows are in right."""
        r = db.execute("""
            SELECT dept FROM employees WHERE dept = 'Engineering'
            EXCEPT
            SELECT dept FROM contractors WHERE dept = 'Engineering'
        """)
        # Both have 'Engineering', EXCEPT deduplicates first
        assert len(r.rows) == 0

    def test_except_no_overlap(self, db):
        """EXCEPT with no overlap returns all left rows (deduplicated)."""
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Marketing'
            EXCEPT
            SELECT name FROM contractors WHERE dept = 'Engineering'
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Eve']

    def test_except_removes_duplicates(self, db):
        """EXCEPT implicitly deduplicates result."""
        r = db.execute("""
            SELECT dept FROM employees
            EXCEPT
            SELECT dept FROM contractors WHERE dept = 'Marketing'
        """)
        # employees depts: Eng, Eng, Sales, Sales, Marketing
        # right: (none, no marketing contractors)
        # EXCEPT dedup: Engineering, Sales, Marketing
        depts = sorted(r[0] for r in r.rows)
        assert depts == ['Engineering', 'Marketing', 'Sales']

    def test_except_all(self, db):
        """EXCEPT ALL subtracts by count."""
        r = db.execute("""
            SELECT dept FROM employees
            EXCEPT ALL
            SELECT dept FROM contractors
        """)
        # employees: Eng(2), Sales(2), Marketing(1)
        # contractors: Eng(2), Sales(1)
        # EXCEPT ALL: Eng=2-2=0, Sales=2-1=1, Marketing=1-0=1
        depts = sorted(r[0] for r in r.rows)
        assert depts == ['Marketing', 'Sales']

    def test_except_all_preserves_order(self, db):
        """EXCEPT ALL processes left rows in order."""
        r = db.execute("""
            SELECT name FROM employees
            EXCEPT ALL
            SELECT name FROM contractors
        """)
        # employees: Alice, Bob, Carol, Dave, Eve
        # contractors: Frank, Grace, Alice
        # EXCEPT ALL: remove one Alice -> Bob, Carol, Dave, Eve
        names = [r[0] for r in r.rows]
        assert 'Alice' not in names
        assert len(names) == 4

    def test_except_empty_right(self, db):
        """EXCEPT with empty right returns all left rows (deduplicated)."""
        r = db.execute("""
            SELECT name FROM employees
            EXCEPT
            SELECT name FROM contractors WHERE rate > 999
        """)
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']

    def test_except_empty_left(self, db):
        r = db.execute("""
            SELECT name FROM employees WHERE salary > 200000
            EXCEPT
            SELECT name FROM contractors
        """)
        assert len(r.rows) == 0

    def test_except_multiple_columns(self, db):
        """EXCEPT on multi-column rows."""
        r = db.execute("""
            SELECT name, dept FROM employees
            EXCEPT
            SELECT name, dept FROM contractors
        """)
        # (Alice, Engineering) removed; rest stay
        rows = sorted(r.rows, key=lambda x: x[0])
        assert len(rows) == 4
        names = [r[0] for r in rows]
        assert 'Alice' not in names

    def test_except_reverse(self, db):
        """Contractor names not in employees."""
        r = db.execute("""
            SELECT name FROM contractors
            EXCEPT
            SELECT name FROM employees
        """)
        names = sorted(r[0] for r in r.rows)
        assert names == ['Frank', 'Grace']


# =====================================================================
# Chained Set Operations
# =====================================================================

class TestChainedOps:
    """Multiple set operations chained together."""

    def test_union_then_intersect(self, db):
        """(A UNION B) INTERSECT C -- left-associative."""
        r = db.execute("""
            SELECT name FROM employees
            UNION
            SELECT name FROM contractors
            INTERSECT
            SELECT name FROM employees WHERE dept = 'Engineering'
        """)
        # Left-associative: (employees UNION contractors) INTERSECT eng_employees
        # UNION: Alice, Bob, Carol, Dave, Eve, Frank, Grace
        # eng_employees: Alice, Bob
        # INTERSECT: Alice, Bob
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Bob']

    def test_except_then_union(self, db):
        """(A EXCEPT B) UNION C."""
        r = db.execute("""
            SELECT name FROM employees
            EXCEPT
            SELECT name FROM contractors
            UNION
            SELECT name FROM contractors WHERE dept = 'Sales'
        """)
        # EXCEPT: Bob, Carol, Dave, Eve (employees minus Alice)
        # UNION with Sales contractors: Grace
        # Result: Bob, Carol, Dave, Eve, Grace
        names = sorted(r[0] for r in r.rows)
        assert names == ['Bob', 'Carol', 'Dave', 'Eve', 'Grace']

    def test_three_way_union(self, db):
        db.execute("CREATE TABLE interns (id INT, name TEXT)")
        db.execute("INSERT INTO interns VALUES (20, 'Ivy')")
        db.execute("INSERT INTO interns VALUES (21, 'Jack')")

        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Engineering'
            UNION
            SELECT name FROM contractors WHERE dept = 'Engineering'
            UNION
            SELECT name FROM interns
        """)
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Bob', 'Frank', 'Ivy', 'Jack']

    def test_intersect_then_except(self, db):
        """(A INTERSECT B) EXCEPT C."""
        r = db.execute("""
            SELECT dept FROM employees
            INTERSECT
            SELECT dept FROM contractors
            EXCEPT
            SELECT dept FROM employees WHERE name = 'Carol'
        """)
        # INTERSECT: Engineering, Sales
        # EXCEPT Sales (Carol's dept): Engineering
        depts = [r[0] for r in r.rows]
        assert depts == ['Engineering']


# =====================================================================
# ORDER BY / LIMIT / OFFSET on Set Operations
# =====================================================================

class TestSetOpClauses:
    """ORDER BY, LIMIT, OFFSET applied to set operation results."""

    def test_union_order_by(self, db):
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Engineering'
            UNION
            SELECT name FROM contractors WHERE dept = 'Engineering'
            ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Bob', 'Frank']

    def test_union_order_by_desc(self, db):
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Engineering'
            UNION
            SELECT name FROM contractors WHERE dept = 'Engineering'
            ORDER BY name DESC
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Frank', 'Bob', 'Alice']

    def test_intersect_order_by(self, db):
        r = db.execute("""
            SELECT dept FROM employees
            INTERSECT
            SELECT dept FROM contractors
            ORDER BY dept
        """)
        depts = [r[0] for r in r.rows]
        assert depts == ['Engineering', 'Sales']

    def test_except_order_by(self, db):
        r = db.execute("""
            SELECT name FROM employees
            EXCEPT
            SELECT name FROM contractors
            ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Bob', 'Carol', 'Dave', 'Eve']

    def test_union_limit(self, db):
        r = db.execute("""
            SELECT name FROM employees
            UNION
            SELECT name FROM contractors
            ORDER BY name
            LIMIT 3
        """)
        assert len(r.rows) == 3
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Bob', 'Carol']

    def test_union_offset(self, db):
        r = db.execute("""
            SELECT name FROM employees
            UNION
            SELECT name FROM contractors
            ORDER BY name
            LIMIT 2
            OFFSET 2
        """)
        assert len(r.rows) == 2
        names = [r[0] for r in r.rows]
        assert names == ['Carol', 'Dave']

    def test_intersect_limit(self, db):
        r = db.execute("""
            SELECT dept FROM employees
            INTERSECT
            SELECT dept FROM contractors
            ORDER BY dept
            LIMIT 1
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Engineering'

    def test_except_limit_offset(self, db):
        r = db.execute("""
            SELECT name FROM employees
            EXCEPT
            SELECT name FROM contractors
            ORDER BY name
            LIMIT 2
            OFFSET 1
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Carol', 'Dave']


# =====================================================================
# Set Operations with CTEs
# =====================================================================

class TestSetOpsWithCTEs:
    """Set operations composed with CTEs."""

    def test_cte_with_intersect_body(self, db):
        r = db.execute("""
            WITH common_names AS (
                SELECT name FROM employees
                INTERSECT
                SELECT name FROM contractors
            )
            SELECT name FROM common_names
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_cte_with_except_body(self, db):
        r = db.execute("""
            WITH unique_employees AS (
                SELECT name FROM employees
                EXCEPT
                SELECT name FROM contractors
            )
            SELECT name FROM unique_employees ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Bob', 'Carol', 'Dave', 'Eve']

    def test_cte_main_query_set_op(self, db):
        """Main query after CTE is a set operation."""
        r = db.execute("""
            WITH eng AS (
                SELECT name FROM employees WHERE dept = 'Engineering'
            )
            SELECT name FROM eng
            UNION
            SELECT name FROM contractors WHERE dept = 'Sales'
        """)
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Bob', 'Grace']

    def test_cte_main_intersect(self, db):
        r = db.execute("""
            WITH all_people AS (
                SELECT name FROM employees
                UNION ALL
                SELECT name FROM contractors
            )
            SELECT name FROM all_people
            INTERSECT
            SELECT name FROM employees WHERE dept = 'Engineering'
        """)
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Bob']

    def test_multiple_ctes_with_set_ops(self, db):
        r = db.execute("""
            WITH
                emp_names AS (SELECT name FROM employees),
                cont_names AS (SELECT name FROM contractors),
                common AS (
                    SELECT name FROM emp_names
                    INTERSECT
                    SELECT name FROM cont_names
                )
            SELECT name FROM common
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_recursive_cte_still_works(self, db):
        """Recursive CTEs still work with the new parser."""
        r = db.execute("""
            WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM nums WHERE n < 5
            )
            SELECT n FROM nums
        """)
        values = [r[0] for r in r.rows]
        assert values == [1, 2, 3, 4, 5]


# =====================================================================
# Column Count Validation
# =====================================================================

class TestColumnValidation:
    """Column count mismatch errors."""

    def test_union_column_mismatch(self, db):
        with pytest.raises(CompileError, match="column count"):
            db.execute("""
                SELECT name, dept FROM employees
                UNION
                SELECT name FROM contractors
            """)

    def test_intersect_column_mismatch(self, db):
        with pytest.raises(CompileError, match="column count"):
            db.execute("""
                SELECT name FROM employees
                INTERSECT
                SELECT name, dept FROM contractors
            """)

    def test_except_column_mismatch(self, db):
        with pytest.raises(CompileError, match="column count"):
            db.execute("""
                SELECT name, dept, salary FROM employees
                EXCEPT
                SELECT name, dept FROM contractors
            """)


# =====================================================================
# Edge Cases
# =====================================================================

class TestEdgeCases:
    """Edge cases and corner scenarios."""

    def test_intersect_with_nulls(self, db):
        """NULL handling in set operations."""
        db.execute("CREATE TABLE t1 (x INT)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (NULL)")
        db.execute("CREATE TABLE t2 (x INT)")
        db.execute("INSERT INTO t2 VALUES (1)")
        db.execute("INSERT INTO t2 VALUES (NULL)")

        r = db.execute("SELECT x FROM t1 INTERSECT SELECT x FROM t2")
        values = sorted(r.rows, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
        assert len(values) == 2  # 1 and NULL both match

    def test_except_with_nulls(self, db):
        db.execute("CREATE TABLE t1 (x INT)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (2)")
        db.execute("INSERT INTO t1 VALUES (NULL)")
        db.execute("CREATE TABLE t2 (x INT)")
        db.execute("INSERT INTO t2 VALUES (1)")
        db.execute("INSERT INTO t2 VALUES (NULL)")

        r = db.execute("SELECT x FROM t1 EXCEPT SELECT x FROM t2")
        values = [r[0] for r in r.rows]
        assert values == [2]

    def test_union_empty_tables(self, db):
        db.execute("CREATE TABLE empty1 (x INT)")
        db.execute("CREATE TABLE empty2 (x INT)")

        r = db.execute("SELECT x FROM empty1 UNION SELECT x FROM empty2")
        assert len(r.rows) == 0

    def test_intersect_empty_tables(self, db):
        db.execute("CREATE TABLE empty1 (x INT)")
        db.execute("CREATE TABLE empty2 (x INT)")

        r = db.execute("SELECT x FROM empty1 INTERSECT SELECT x FROM empty2")
        assert len(r.rows) == 0

    def test_except_empty_tables(self, db):
        db.execute("CREATE TABLE empty1 (x INT)")
        db.execute("CREATE TABLE empty2 (x INT)")

        r = db.execute("SELECT x FROM empty1 EXCEPT SELECT x FROM empty2")
        assert len(r.rows) == 0

    def test_set_op_single_row(self, db):
        r = db.execute("SELECT 1 UNION SELECT 1")
        assert len(r.rows) == 1

    def test_set_op_single_row_all(self, db):
        r = db.execute("SELECT 1 UNION ALL SELECT 1")
        assert len(r.rows) == 2

    def test_intersect_single_row_match(self, db):
        r = db.execute("SELECT 1 INTERSECT SELECT 1")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_intersect_single_row_no_match(self, db):
        r = db.execute("SELECT 1 INTERSECT SELECT 2")
        assert len(r.rows) == 0

    def test_except_single_rows(self, db):
        r = db.execute("SELECT 1 EXCEPT SELECT 2")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_except_single_rows_match(self, db):
        r = db.execute("SELECT 1 EXCEPT SELECT 1")
        assert len(r.rows) == 0

    def test_intersect_all_multiplicity(self, db):
        """INTERSECT ALL: min(left_count, right_count)."""
        db.execute("CREATE TABLE t1 (x INT)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (2)")

        db.execute("CREATE TABLE t2 (x INT)")
        db.execute("INSERT INTO t2 VALUES (1)")
        db.execute("INSERT INTO t2 VALUES (1)")
        db.execute("INSERT INTO t2 VALUES (3)")

        r = db.execute("SELECT x FROM t1 INTERSECT ALL SELECT x FROM t2")
        values = sorted(r[0] for r in r.rows)
        # 1 appears min(3,2)=2 times; 2 has 0 in right; 3 has 0 in left
        assert values == [1, 1]

    def test_except_all_multiplicity(self, db):
        """EXCEPT ALL: left_count - right_count."""
        db.execute("CREATE TABLE t1 (x INT)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t1 VALUES (2)")
        db.execute("INSERT INTO t1 VALUES (2)")

        db.execute("CREATE TABLE t2 (x INT)")
        db.execute("INSERT INTO t2 VALUES (1)")
        db.execute("INSERT INTO t2 VALUES (2)")
        db.execute("INSERT INTO t2 VALUES (2)")
        db.execute("INSERT INTO t2 VALUES (2)")

        r = db.execute("SELECT x FROM t1 EXCEPT ALL SELECT x FROM t2")
        values = sorted(r[0] for r in r.rows)
        # 1: 3-1=2, 2: 2-3=0 (can't go negative)
        assert values == [1, 1]

    def test_string_values(self, db):
        r = db.execute("""
            SELECT name FROM employees
            INTERSECT
            SELECT name FROM contractors
        """)
        assert r.rows[0][0] == 'Alice'

    def test_mixed_types_in_set_op(self, db):
        """Numbers and strings in same column."""
        db.execute("CREATE TABLE t1 (x TEXT)")
        db.execute("INSERT INTO t1 VALUES ('a')")
        db.execute("INSERT INTO t1 VALUES ('b')")
        db.execute("CREATE TABLE t2 (x TEXT)")
        db.execute("INSERT INTO t2 VALUES ('b')")
        db.execute("INSERT INTO t2 VALUES ('c')")

        r = db.execute("SELECT x FROM t1 INTERSECT SELECT x FROM t2")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'b'


# =====================================================================
# Set Operations with Aggregation
# =====================================================================

class TestSetOpsWithAggregation:
    """Set operations combined with GROUP BY and aggregation."""

    def test_union_of_aggregates(self, db):
        r = db.execute("""
            SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept
            UNION
            SELECT dept, COUNT(*) AS cnt FROM contractors GROUP BY dept
            ORDER BY dept
        """)
        # employees: (Engineering,2), (Marketing,1), (Sales,2)
        # contractors: (Engineering,2), (Sales,1)
        # UNION dedup: (Engineering,2) matches -> 4 distinct rows
        assert len(r.rows) == 4

    def test_intersect_of_aggregates(self, db):
        r = db.execute("""
            SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept
            INTERSECT
            SELECT dept, COUNT(*) AS cnt FROM contractors GROUP BY dept
        """)
        # employees: (Engineering,2), (Marketing,1), (Sales,2)
        # contractors: (Engineering,2), (Sales,1)
        # INTERSECT: (Engineering,2) matches
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Engineering'
        assert r.rows[0][1] == 2

    def test_except_of_aggregates(self, db):
        r = db.execute("""
            SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept
            EXCEPT
            SELECT dept, COUNT(*) AS cnt FROM contractors GROUP BY dept
            ORDER BY dept
        """)
        # employees: (Engineering,2), (Marketing,1), (Sales,2)
        # contractors: (Engineering,2), (Sales,1)
        # EXCEPT: (Marketing,1), (Sales,2) remain
        assert len(r.rows) == 2


# =====================================================================
# Set Operations with WHERE, DISTINCT, expressions
# =====================================================================

class TestSetOpsWithExpressions:
    """Set operations with complex expressions in SELECT."""

    def test_union_with_expressions(self, db):
        r = db.execute("""
            SELECT name, salary * 12 AS annual FROM employees WHERE dept = 'Engineering'
            UNION
            SELECT name, rate * 2000 AS annual FROM contractors WHERE dept = 'Engineering'
            ORDER BY annual DESC
        """)
        # Alice: 90000*12=1080000, Bob: 85000*12=1020000
        # Frank: 100*2000=200000, Alice: 110*2000=220000
        # UNION dedup by (name, annual): all 4 distinct
        assert len(r.rows) == 4

    def test_intersect_with_distinct_select(self, db):
        r = db.execute("""
            SELECT DISTINCT dept FROM employees
            INTERSECT
            SELECT DISTINCT dept FROM contractors
        """)
        depts = sorted(r[0] for r in r.rows)
        assert depts == ['Engineering', 'Sales']

    def test_except_with_case(self, db):
        r = db.execute("""
            SELECT name, CASE WHEN salary > 80000 THEN 'high' ELSE 'low' END AS level
            FROM employees
            EXCEPT
            SELECT name, 'high' FROM contractors
        """)
        # employees: (Alice,high), (Bob,high), (Carol,low), (Dave,low), (Eve,low)
        # contractors as high: (Frank,high), (Grace,high), (Alice,high)
        # EXCEPT: (Alice,high) removed, rest stay
        assert len(r.rows) == 4


# =====================================================================
# Backward Compatibility
# =====================================================================

class TestBackwardCompatibility:
    """Existing SQL features still work through SetOpDB."""

    def test_basic_select(self, db):
        r = db.execute("SELECT name FROM employees WHERE dept = 'Engineering' ORDER BY name")
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Bob']

    def test_insert_select(self, db):
        r = db.execute("INSERT INTO employees VALUES (6, 'Zara', 'HR', 60000)")
        assert r.rows_affected == 1

    def test_update(self, db):
        db.execute("UPDATE employees SET salary = 95000 WHERE name = 'Alice'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert r.rows[0][0] == 95000

    def test_delete(self, db):
        db.execute("DELETE FROM employees WHERE name = 'Eve'")
        r = db.execute("SELECT COUNT(*) FROM employees")
        assert r.rows[0][0] == 4

    def test_join(self, db):
        r = db.execute("""
            SELECT employees.name, contractors.dept
            FROM employees
            JOIN contractors ON employees.dept = contractors.dept
            WHERE employees.name = 'Alice'
        """)
        assert len(r.rows) == 2  # Alice joins with Frank and Alice(contractor)

    def test_group_by(self, db):
        r = db.execute("SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept ORDER BY dept")
        assert len(r.rows) == 3

    def test_subquery(self, db):
        r = db.execute("""
            SELECT name FROM employees
            WHERE salary > (SELECT AVG(salary) FROM employees)
            ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert 'Alice' in names

    def test_cte(self, db):
        r = db.execute("""
            WITH eng AS (SELECT name FROM employees WHERE dept = 'Engineering')
            SELECT name FROM eng ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Bob']

    def test_cte_union_inside(self, db):
        """C267 UNION inside CTE still works."""
        r = db.execute("""
            WITH combined AS (
                SELECT name FROM employees WHERE dept = 'Engineering'
                UNION ALL
                SELECT name FROM contractors WHERE dept = 'Engineering'
            )
            SELECT name FROM combined ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert len(names) == 4  # Alice, Alice, Bob, Frank

    def test_between(self, db):
        r = db.execute("SELECT name FROM employees WHERE salary BETWEEN 75000 AND 85000 ORDER BY name")
        names = [r[0] for r in r.rows]
        assert names == ['Bob', 'Carol', 'Dave']

    def test_like(self, db):
        r = db.execute("SELECT name FROM employees WHERE name LIKE 'A%'")
        assert r.rows[0][0] == 'Alice'

    def test_in_list(self, db):
        r = db.execute("SELECT name FROM employees WHERE dept IN ('Engineering', 'Marketing') ORDER BY name")
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']

    def test_is_null(self, db):
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (NULL)")
        db.execute("INSERT INTO t VALUES (1)")
        r = db.execute("SELECT x FROM t WHERE x IS NULL")
        assert len(r.rows) == 1
        assert r.rows[0][0] is None


# =====================================================================
# Multiple statements
# =====================================================================

class TestMultipleStatements:
    """Multiple statement execution with set operations."""

    def test_multi_with_set_ops(self, db):
        results = db.execute_many("""
            SELECT name FROM employees WHERE dept = 'Engineering'
            UNION
            SELECT name FROM contractors WHERE dept = 'Engineering';

            SELECT dept FROM employees
            INTERSECT
            SELECT dept FROM contractors;
        """)
        assert len(results) == 2
        # First: union of engineering names
        names = sorted(r[0] for r in results[0].rows)
        assert names == ['Alice', 'Bob', 'Frank']
        # Second: intersect of departments
        depts = sorted(r[0] for r in results[1].rows)
        assert depts == ['Engineering', 'Sales']


# =====================================================================
# Parsing Tests
# =====================================================================

class TestParsing:
    """Parser correctly produces SetOperation AST nodes."""

    def test_parse_intersect(self):
        ast = parse_set_op_sql("SELECT 1 INTERSECT SELECT 2")
        assert isinstance(ast, SetOperation)
        assert ast.op == 'intersect'
        assert ast.all_ is False

    def test_parse_intersect_all(self):
        ast = parse_set_op_sql("SELECT 1 INTERSECT ALL SELECT 2")
        assert isinstance(ast, SetOperation)
        assert ast.op == 'intersect'
        assert ast.all_ is True

    def test_parse_except(self):
        ast = parse_set_op_sql("SELECT 1 EXCEPT SELECT 2")
        assert isinstance(ast, SetOperation)
        assert ast.op == 'except'
        assert ast.all_ is False

    def test_parse_except_all(self):
        ast = parse_set_op_sql("SELECT 1 EXCEPT ALL SELECT 2")
        assert isinstance(ast, SetOperation)
        assert ast.op == 'except'
        assert ast.all_ is True

    def test_parse_union_still_works(self):
        ast = parse_set_op_sql("SELECT 1 UNION SELECT 2")
        assert isinstance(ast, SetOperation)
        assert ast.op == 'union'

    def test_parse_chained(self):
        ast = parse_set_op_sql("SELECT 1 UNION SELECT 2 INTERSECT SELECT 3")
        # Left-associative: (1 UNION 2) INTERSECT 3
        assert isinstance(ast, SetOperation)
        assert ast.op == 'intersect'
        assert isinstance(ast.left, SetOperation)
        assert ast.left.op == 'union'

    def test_parse_with_order_by(self):
        ast = parse_set_op_sql("SELECT 1 UNION SELECT 2 ORDER BY 1")
        assert isinstance(ast, SetOpWithClauses)
        assert ast.order_by is not None

    def test_parse_with_limit(self):
        ast = parse_set_op_sql("SELECT 1 UNION SELECT 2 LIMIT 1")
        assert isinstance(ast, SetOpWithClauses)
        assert ast.limit == 1

    def test_parse_with_offset(self):
        ast = parse_set_op_sql("SELECT 1 UNION SELECT 2 LIMIT 1 OFFSET 1")
        assert isinstance(ast, SetOpWithClauses)
        assert ast.limit == 1
        assert ast.offset == 1

    def test_plain_select_no_set_op(self):
        """Plain SELECT still parses correctly."""
        from mini_database import SelectStmt
        ast = parse_set_op_sql("SELECT 1")
        assert isinstance(ast, SelectStmt)
