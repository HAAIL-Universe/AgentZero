"""
Tests for C254: UNION / INTERSECT / EXCEPT Set Operations
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from set_operations import SetOpDB, SetOpStmt, DatabaseError


class TestBasicUnion(unittest.TestCase):
    """Test basic UNION operations."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (id INT, name TEXT)")
        self.db.execute("CREATE TABLE t2 (id INT, name TEXT)")
        self.db.execute("INSERT INTO t1 VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO t1 VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO t1 VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO t2 VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO t2 VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO t2 VALUES (4, 'Diana')")

    def test_union_removes_duplicates(self):
        r = self.db.execute("SELECT id, name FROM t1 UNION SELECT id, name FROM t2")
        self.assertEqual(len(r.rows), 4)  # 1,2,3,4
        ids = sorted(row[0] for row in r.rows)
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_union_all_keeps_duplicates(self):
        r = self.db.execute("SELECT id, name FROM t1 UNION ALL SELECT id, name FROM t2")
        self.assertEqual(len(r.rows), 6)  # 3 + 3

    def test_union_uses_left_column_names(self):
        r = self.db.execute("SELECT id AS x, name AS y FROM t1 UNION SELECT id, name FROM t2")
        self.assertEqual(r.columns, ['x', 'y'])

    def test_union_single_column(self):
        r = self.db.execute("SELECT id FROM t1 UNION SELECT id FROM t2")
        ids = sorted(row[0] for row in r.rows)
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_union_all_preserves_order(self):
        r = self.db.execute("SELECT id FROM t1 UNION ALL SELECT id FROM t2")
        ids = [row[0] for row in r.rows]
        self.assertEqual(ids, [1, 2, 3, 2, 3, 4])

    def test_union_with_literals(self):
        r = self.db.execute("SELECT 1, 'hello' UNION SELECT 2, 'world'")
        self.assertEqual(len(r.rows), 2)

    def test_union_with_where(self):
        r = self.db.execute(
            "SELECT id, name FROM t1 WHERE id > 1 "
            "UNION SELECT id, name FROM t2 WHERE id < 4"
        )
        ids = sorted(row[0] for row in r.rows)
        self.assertEqual(ids, [2, 3])


class TestBasicIntersect(unittest.TestCase):
    """Test basic INTERSECT operations."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (id INT, name TEXT)")
        self.db.execute("CREATE TABLE t2 (id INT, name TEXT)")
        self.db.execute("INSERT INTO t1 VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO t1 VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO t1 VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO t2 VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO t2 VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO t2 VALUES (4, 'Diana')")

    def test_intersect_common_rows(self):
        r = self.db.execute("SELECT id, name FROM t1 INTERSECT SELECT id, name FROM t2")
        self.assertEqual(len(r.rows), 2)
        ids = sorted(row[0] for row in r.rows)
        self.assertEqual(ids, [2, 3])

    def test_intersect_no_common_rows(self):
        r = self.db.execute(
            "SELECT id, name FROM t1 WHERE id = 1 "
            "INTERSECT SELECT id, name FROM t2 WHERE id = 4"
        )
        self.assertEqual(len(r.rows), 0)

    def test_intersect_single_column(self):
        r = self.db.execute("SELECT id FROM t1 INTERSECT SELECT id FROM t2")
        ids = sorted(row[0] for row in r.rows)
        self.assertEqual(ids, [2, 3])

    def test_intersect_removes_duplicates(self):
        """INTERSECT without ALL deduplicates."""
        self.db.execute("INSERT INTO t1 VALUES (2, 'Bob')")  # duplicate
        self.db.execute("INSERT INTO t2 VALUES (2, 'Bob')")  # duplicate
        r = self.db.execute("SELECT id, name FROM t1 INTERSECT SELECT id, name FROM t2")
        ids = sorted(row[0] for row in r.rows)
        self.assertEqual(ids, [2, 3])  # still just 2 unique rows

    def test_intersect_all_preserves_duplicates(self):
        """INTERSECT ALL preserves min duplicate count."""
        self.db.execute("INSERT INTO t1 VALUES (2, 'Bob')")  # now 2 copies in t1
        self.db.execute("INSERT INTO t2 VALUES (2, 'Bob')")  # now 2 copies in t2
        self.db.execute("INSERT INTO t2 VALUES (2, 'Bob')")  # now 3 copies in t2
        r = self.db.execute("SELECT id, name FROM t1 INTERSECT ALL SELECT id, name FROM t2")
        bobs = [row for row in r.rows if row[0] == 2]
        self.assertEqual(len(bobs), 2)  # min(2, 3) = 2


class TestBasicExcept(unittest.TestCase):
    """Test basic EXCEPT operations."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (id INT, name TEXT)")
        self.db.execute("CREATE TABLE t2 (id INT, name TEXT)")
        self.db.execute("INSERT INTO t1 VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO t1 VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO t1 VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO t2 VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO t2 VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO t2 VALUES (4, 'Diana')")

    def test_except_basic(self):
        r = self.db.execute("SELECT id, name FROM t1 EXCEPT SELECT id, name FROM t2")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 1)
        self.assertEqual(r.rows[0][1], 'Alice')

    def test_except_reverse(self):
        r = self.db.execute("SELECT id, name FROM t2 EXCEPT SELECT id, name FROM t1")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 4)
        self.assertEqual(r.rows[0][1], 'Diana')

    def test_except_identical_sets(self):
        r = self.db.execute("SELECT id, name FROM t1 EXCEPT SELECT id, name FROM t1")
        self.assertEqual(len(r.rows), 0)

    def test_except_no_overlap(self):
        r = self.db.execute(
            "SELECT id, name FROM t1 WHERE id = 1 "
            "EXCEPT SELECT id, name FROM t2 WHERE id = 4"
        )
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 1)

    def test_except_all(self):
        """EXCEPT ALL subtracts duplicate counts."""
        self.db.execute("INSERT INTO t1 VALUES (2, 'Bob')")  # 2 Bobs in t1
        self.db.execute("INSERT INTO t1 VALUES (2, 'Bob')")  # 3 Bobs in t1
        # t2 has 1 Bob
        r = self.db.execute("SELECT id, name FROM t1 EXCEPT ALL SELECT id, name FROM t2")
        bobs = [row for row in r.rows if row[0] == 2]
        self.assertEqual(len(bobs), 2)  # 3 - 1 = 2

    def test_except_all_removes_all(self):
        """EXCEPT ALL can remove all copies."""
        self.db.execute("INSERT INTO t2 VALUES (1, 'Alice')")  # now both have Alice
        r = self.db.execute("SELECT id, name FROM t1 EXCEPT ALL SELECT id, name FROM t2")
        self.assertEqual(len(r.rows), 0)  # all removed


class TestSetOpWithOrderBy(unittest.TestCase):
    """Test set operations with ORDER BY."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (id INT, name TEXT)")
        self.db.execute("CREATE TABLE t2 (id INT, name TEXT)")
        self.db.execute("INSERT INTO t1 VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO t1 VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO t2 VALUES (4, 'Diana')")
        self.db.execute("INSERT INTO t2 VALUES (2, 'Bob')")

    def test_union_order_by_column(self):
        r = self.db.execute(
            "SELECT id, name FROM t1 UNION SELECT id, name FROM t2 ORDER BY id"
        )
        ids = [row[0] for row in r.rows]
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_union_order_by_desc(self):
        r = self.db.execute(
            "SELECT id, name FROM t1 UNION SELECT id, name FROM t2 ORDER BY id DESC"
        )
        ids = [row[0] for row in r.rows]
        self.assertEqual(ids, [4, 3, 2, 1])

    def test_union_order_by_name(self):
        r = self.db.execute(
            "SELECT id, name FROM t1 UNION SELECT id, name FROM t2 ORDER BY name"
        )
        names = [row[1] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob', 'Charlie', 'Diana'])

    def test_intersect_order_by(self):
        self.db.execute("INSERT INTO t2 VALUES (1, 'Alice')")
        r = self.db.execute(
            "SELECT id, name FROM t1 INTERSECT SELECT id, name FROM t2 ORDER BY id DESC"
        )
        ids = [row[0] for row in r.rows]
        self.assertEqual(ids, [1])

    def test_except_order_by(self):
        r = self.db.execute(
            "SELECT id, name FROM t1 EXCEPT SELECT id, name FROM t2 ORDER BY id"
        )
        ids = [row[0] for row in r.rows]
        self.assertEqual(ids, [1, 3])


class TestSetOpWithLimit(unittest.TestCase):
    """Test set operations with LIMIT and OFFSET."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE nums (n INT)")
        for i in range(1, 6):
            self.db.execute(f"INSERT INTO nums VALUES ({i})")

    def test_union_with_limit(self):
        r = self.db.execute(
            "SELECT n FROM nums WHERE n <= 3 "
            "UNION SELECT n FROM nums WHERE n >= 3 "
            "ORDER BY n LIMIT 3"
        )
        self.assertEqual(len(r.rows), 3)
        self.assertEqual([row[0] for row in r.rows], [1, 2, 3])

    def test_union_with_offset(self):
        r = self.db.execute(
            "SELECT n FROM nums WHERE n <= 3 "
            "UNION SELECT n FROM nums WHERE n >= 3 "
            "ORDER BY n LIMIT 2 OFFSET 2"
        )
        self.assertEqual([row[0] for row in r.rows], [3, 4])

    def test_union_all_with_limit(self):
        r = self.db.execute(
            "SELECT n FROM nums UNION ALL SELECT n FROM nums ORDER BY n LIMIT 4"
        )
        self.assertEqual(len(r.rows), 4)
        self.assertEqual([row[0] for row in r.rows], [1, 1, 2, 2])


class TestChainedSetOps(unittest.TestCase):
    """Test chaining multiple set operations."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE a (x INT)")
        self.db.execute("CREATE TABLE b (x INT)")
        self.db.execute("CREATE TABLE c (x INT)")
        for v in [1, 2, 3]:
            self.db.execute(f"INSERT INTO a VALUES ({v})")
        for v in [2, 3, 4]:
            self.db.execute(f"INSERT INTO b VALUES ({v})")
        for v in [3, 4, 5]:
            self.db.execute(f"INSERT INTO c VALUES ({v})")

    def test_union_chain(self):
        r = self.db.execute(
            "SELECT x FROM a UNION SELECT x FROM b UNION SELECT x FROM c"
        )
        vals = sorted(row[0] for row in r.rows)
        self.assertEqual(vals, [1, 2, 3, 4, 5])

    def test_union_all_chain(self):
        r = self.db.execute(
            "SELECT x FROM a UNION ALL SELECT x FROM b UNION ALL SELECT x FROM c"
        )
        self.assertEqual(len(r.rows), 9)  # 3 + 3 + 3

    def test_except_chain(self):
        """a EXCEPT b EXCEPT c = (a - b) - c"""
        r = self.db.execute(
            "SELECT x FROM a EXCEPT SELECT x FROM b EXCEPT SELECT x FROM c"
        )
        # a - b = {1}, then {1} - c = {1}
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 1)

    def test_intersect_chain(self):
        r = self.db.execute(
            "SELECT x FROM a INTERSECT SELECT x FROM b INTERSECT SELECT x FROM c"
        )
        # a & b = {2,3}, then {2,3} & c = {3}
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 3)

    def test_mixed_union_except(self):
        """(a UNION b) EXCEPT c"""
        r = self.db.execute(
            "SELECT x FROM a UNION SELECT x FROM b EXCEPT SELECT x FROM c ORDER BY x"
        )
        # UNION and EXCEPT at same precedence, left-to-right
        # (a UNION b) = {1,2,3,4}, then - c = {1,2}
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2])

    def test_intersect_higher_precedence(self):
        """INTERSECT binds tighter than UNION.
        a UNION b INTERSECT c = a UNION (b INTERSECT c)"""
        r = self.db.execute(
            "SELECT x FROM a UNION SELECT x FROM b INTERSECT SELECT x FROM c ORDER BY x"
        )
        # b INTERSECT c = {3,4}
        # a UNION {3,4} = {1,2,3,4}
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2, 3, 4])

    def test_except_then_union(self):
        r = self.db.execute(
            "SELECT x FROM a EXCEPT SELECT x FROM b UNION SELECT x FROM c ORDER BY x"
        )
        # (a EXCEPT b) = {1}, then UNION c = {1, 3, 4, 5}
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 3, 4, 5])


class TestSetOpWithAggregates(unittest.TestCase):
    """Test set operations with aggregate functions."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE sales (region TEXT, amount INT)")
        self.db.execute("INSERT INTO sales VALUES ('East', 100)")
        self.db.execute("INSERT INTO sales VALUES ('East', 200)")
        self.db.execute("INSERT INTO sales VALUES ('West', 300)")
        self.db.execute("INSERT INTO sales VALUES ('West', 400)")

    def test_union_aggregates(self):
        r = self.db.execute(
            "SELECT region, SUM(amount) AS total FROM sales GROUP BY region "
            "UNION "
            "SELECT 'Total', SUM(amount) FROM sales"
        )
        self.assertEqual(len(r.rows), 3)  # East, West, Total

    def test_intersect_with_having(self):
        self.db.execute("CREATE TABLE sales2 (region TEXT, amount INT)")
        self.db.execute("INSERT INTO sales2 VALUES ('East', 150)")
        self.db.execute("INSERT INTO sales2 VALUES ('West', 350)")
        r = self.db.execute(
            "SELECT region FROM sales GROUP BY region "
            "INTERSECT "
            "SELECT region FROM sales2 GROUP BY region"
        )
        regions = sorted(row[0] for row in r.rows)
        self.assertEqual(regions, ['East', 'West'])


class TestSetOpWithCTEs(unittest.TestCase):
    """Test set operations combined with CTEs."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT)")
        self.db.execute("INSERT INTO employees VALUES (1, 'Alice', 'Eng')")
        self.db.execute("INSERT INTO employees VALUES (2, 'Bob', 'Eng')")
        self.db.execute("INSERT INTO employees VALUES (3, 'Charlie', 'Sales')")
        self.db.execute("INSERT INTO employees VALUES (4, 'Diana', 'Sales')")
        self.db.execute("INSERT INTO employees VALUES (5, 'Eve', 'HR')")

    def test_cte_then_union(self):
        r = self.db.execute(
            "WITH eng AS (SELECT id, name FROM employees WHERE dept = 'Eng') "
            "SELECT id, name FROM eng "
            "UNION "
            "SELECT id, name FROM employees WHERE dept = 'HR'"
        )
        self.assertEqual(len(r.rows), 3)  # Alice, Bob, Eve

    def test_cte_used_in_both_sides(self):
        r = self.db.execute(
            "WITH all_emp AS (SELECT id, name, dept FROM employees) "
            "SELECT id, name FROM all_emp WHERE dept = 'Eng' "
            "INTERSECT "
            "SELECT id, name FROM all_emp WHERE id <= 3"
        )
        # Eng: 1,2. id<=3: 1,2,3. Intersect: 1,2
        self.assertEqual(len(r.rows), 2)

    def test_union_inside_cte(self):
        r = self.db.execute(
            "WITH combined AS ("
            "  SELECT name FROM employees WHERE dept = 'Eng' "
            "  UNION "
            "  SELECT name FROM employees WHERE dept = 'HR'"
            ") "
            "SELECT name FROM combined ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob', 'Eve'])


class TestSetOpColumnValidation(unittest.TestCase):
    """Test column count mismatch errors."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (a INT, b INT)")
        self.db.execute("CREATE TABLE t2 (x INT)")

    def test_column_count_mismatch(self):
        with self.assertRaises(DatabaseError) as ctx:
            self.db.execute("SELECT a, b FROM t1 UNION SELECT x FROM t2")
        self.assertIn("column count", str(ctx.exception).lower())

    def test_column_count_mismatch_intersect(self):
        with self.assertRaises(DatabaseError) as ctx:
            self.db.execute("SELECT a FROM t1 INTERSECT SELECT x, x FROM t2")
        self.assertIn("column count", str(ctx.exception).lower())


class TestSetOpNulls(unittest.TestCase):
    """Test NULL handling in set operations."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (id INT, val TEXT)")
        self.db.execute("CREATE TABLE t2 (id INT, val TEXT)")

    def test_union_with_nulls(self):
        self.db.execute("INSERT INTO t1 VALUES (1, NULL)")
        self.db.execute("INSERT INTO t2 VALUES (1, NULL)")
        r = self.db.execute("SELECT id, val FROM t1 UNION SELECT id, val FROM t2")
        self.assertEqual(len(r.rows), 1)  # (1, NULL) deduped

    def test_intersect_with_nulls(self):
        self.db.execute("INSERT INTO t1 VALUES (1, NULL)")
        self.db.execute("INSERT INTO t2 VALUES (1, NULL)")
        r = self.db.execute("SELECT id, val FROM t1 INTERSECT SELECT id, val FROM t2")
        self.assertEqual(len(r.rows), 1)
        self.assertIsNone(r.rows[0][1])

    def test_except_with_nulls(self):
        self.db.execute("INSERT INTO t1 VALUES (1, NULL)")
        self.db.execute("INSERT INTO t1 VALUES (2, 'hello')")
        self.db.execute("INSERT INTO t2 VALUES (1, NULL)")
        r = self.db.execute("SELECT id, val FROM t1 EXCEPT SELECT id, val FROM t2")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 2)


class TestSetOpEmpty(unittest.TestCase):
    """Test set operations with empty results."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (id INT)")
        self.db.execute("CREATE TABLE t2 (id INT)")
        self.db.execute("INSERT INTO t2 VALUES (1)")

    def test_union_with_empty_left(self):
        r = self.db.execute("SELECT id FROM t1 UNION SELECT id FROM t2")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 1)

    def test_intersect_with_empty(self):
        r = self.db.execute("SELECT id FROM t1 INTERSECT SELECT id FROM t2")
        self.assertEqual(len(r.rows), 0)

    def test_except_empty_minus_nonempty(self):
        r = self.db.execute("SELECT id FROM t1 EXCEPT SELECT id FROM t2")
        self.assertEqual(len(r.rows), 0)

    def test_except_nonempty_minus_empty(self):
        r = self.db.execute("SELECT id FROM t2 EXCEPT SELECT id FROM t1")
        self.assertEqual(len(r.rows), 1)


class TestSetOpWithJoins(unittest.TestCase):
    """Test set operations with JOINed queries."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE dept (id INT, name TEXT)")
        self.db.execute("CREATE TABLE emp (id INT, name TEXT, dept_id INT)")
        self.db.execute("INSERT INTO dept VALUES (1, 'Engineering')")
        self.db.execute("INSERT INTO dept VALUES (2, 'Sales')")
        self.db.execute("INSERT INTO emp VALUES (1, 'Alice', 1)")
        self.db.execute("INSERT INTO emp VALUES (2, 'Bob', 1)")
        self.db.execute("INSERT INTO emp VALUES (3, 'Charlie', 2)")

    def test_union_of_joins(self):
        r = self.db.execute(
            "SELECT emp.name, dept.name FROM emp "
            "JOIN dept ON emp.dept_id = dept.id WHERE dept.id = 1 "
            "UNION "
            "SELECT emp.name, dept.name FROM emp "
            "JOIN dept ON emp.dept_id = dept.id WHERE dept.id = 2"
        )
        self.assertEqual(len(r.rows), 3)


class TestSetOpDistinct(unittest.TestCase):
    """Test UNION vs UNION ALL dedup behavior."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t (x INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")

    def test_union_deduplicates_within_and_across(self):
        r = self.db.execute("SELECT x FROM t UNION SELECT x FROM t")
        vals = sorted(row[0] for row in r.rows)
        self.assertEqual(vals, [1, 2])

    def test_union_all_keeps_all(self):
        r = self.db.execute("SELECT x FROM t UNION ALL SELECT x FROM t")
        self.assertEqual(len(r.rows), 6)  # 3 + 3


class TestSetOpWithExpressions(unittest.TestCase):
    """Test set operations with computed expressions."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE nums (n INT)")
        for i in range(1, 6):
            self.db.execute(f"INSERT INTO nums VALUES ({i})")

    def test_union_with_arithmetic(self):
        r = self.db.execute(
            "SELECT n * 2 AS val FROM nums WHERE n <= 2 "
            "UNION "
            "SELECT n + 10 AS val FROM nums WHERE n <= 2 "
            "ORDER BY val"
        )
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [2, 4, 11, 12])

    def test_intersect_computed_values(self):
        r = self.db.execute(
            "SELECT n FROM nums WHERE n > 2 "
            "INTERSECT "
            "SELECT n FROM nums WHERE n < 5"
        )
        vals = sorted(row[0] for row in r.rows)
        self.assertEqual(vals, [3, 4])


class TestSetOpWithDistinctSelect(unittest.TestCase):
    """Test DISTINCT in individual SELECTs within set operations."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t (x INT, y TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'a')")
        self.db.execute("INSERT INTO t VALUES (1, 'a')")
        self.db.execute("INSERT INTO t VALUES (2, 'b')")

    def test_distinct_in_union_operand(self):
        r = self.db.execute(
            "SELECT DISTINCT x, y FROM t "
            "UNION ALL "
            "SELECT DISTINCT x, y FROM t"
        )
        self.assertEqual(len(r.rows), 4)  # 2 distinct + 2 distinct


class TestSetOpWithSubqueries(unittest.TestCase):
    """Test set operations with various subquery patterns."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (x INT)")
        self.db.execute("CREATE TABLE t2 (x INT)")
        self.db.execute("CREATE TABLE t3 (x INT)")
        for v in [1, 2, 3]:
            self.db.execute(f"INSERT INTO t1 VALUES ({v})")
        for v in [2, 3, 4]:
            self.db.execute(f"INSERT INTO t2 VALUES ({v})")
        for v in [3, 4, 5]:
            self.db.execute(f"INSERT INTO t3 VALUES ({v})")

    def test_parenthesized_union_then_intersect(self):
        """(t1 UNION t2) INTERSECT t3 -- parens override precedence"""
        r = self.db.execute(
            "(SELECT x FROM t1 UNION SELECT x FROM t2) "
            "INTERSECT SELECT x FROM t3 ORDER BY x"
        )
        # t1 UNION t2 = {1,2,3,4}. INTERSECT t3 = {3,4}
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [3, 4])


class TestSetOpWithWindowFunctions(unittest.TestCase):
    """Test set operations with window functions from C252."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE scores (name TEXT, score INT)")
        self.db.execute("INSERT INTO scores VALUES ('Alice', 90)")
        self.db.execute("INSERT INTO scores VALUES ('Bob', 85)")
        self.db.execute("INSERT INTO scores VALUES ('Charlie', 95)")

    def test_union_with_aggregate_result(self):
        r = self.db.execute(
            "SELECT name, score FROM scores WHERE score >= 90 "
            "UNION "
            "SELECT 'Average', AVG(score) FROM scores"
        )
        self.assertTrue(len(r.rows) >= 3)  # Alice, Charlie, Average


class TestSetOpRecursiveCTE(unittest.TestCase):
    """Test that recursive CTEs still work through SetOpDB."""

    def setUp(self):
        self.db = SetOpDB()

    def test_recursive_cte_still_works(self):
        r = self.db.execute(
            "WITH RECURSIVE cnt(x) AS ("
            "  SELECT 1 "
            "  UNION ALL "
            "  SELECT x + 1 FROM cnt WHERE x < 5"
            ") SELECT x FROM cnt ORDER BY x"
        )
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2, 3, 4, 5])

    def test_recursive_cte_with_set_op_main(self):
        self.db.execute("CREATE TABLE extra (x INT)")
        self.db.execute("INSERT INTO extra VALUES (10)")
        r = self.db.execute(
            "WITH RECURSIVE cnt(x) AS ("
            "  SELECT 1 "
            "  UNION ALL "
            "  SELECT x + 1 FROM cnt WHERE x < 3"
            ") "
            "SELECT x FROM cnt "
            "UNION "
            "SELECT x FROM extra "
            "ORDER BY x"
        )
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2, 3, 10])


class TestSetOpNonCTE(unittest.TestCase):
    """Test that non-CTE/non-set-op statements still work."""

    def setUp(self):
        self.db = SetOpDB()

    def test_basic_create_insert_select(self):
        self.db.execute("CREATE TABLE t (id INT, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'hello')")
        r = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(r.rows), 1)

    def test_update(self):
        self.db.execute("CREATE TABLE t (id INT, val INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        self.db.execute("UPDATE t SET val = 20 WHERE id = 1")
        r = self.db.execute("SELECT val FROM t WHERE id = 1")
        self.assertEqual(r.rows[0][0], 20)

    def test_delete(self):
        self.db.execute("CREATE TABLE t (id INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")
        self.db.execute("DELETE FROM t WHERE id = 1")
        r = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(r.rows), 1)


class TestSetOpLiterals(unittest.TestCase):
    """Test set operations with literal-only queries."""

    def setUp(self):
        self.db = SetOpDB()

    def test_union_literals(self):
        r = self.db.execute("SELECT 1 UNION SELECT 2 UNION SELECT 3")
        vals = sorted(row[0] for row in r.rows)
        self.assertEqual(vals, [1, 2, 3])

    def test_union_all_literals(self):
        r = self.db.execute("SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 2")
        self.assertEqual(len(r.rows), 3)

    def test_intersect_literals(self):
        r = self.db.execute("SELECT 1 INTERSECT SELECT 1")
        self.assertEqual(len(r.rows), 1)

    def test_except_literals(self):
        r = self.db.execute("SELECT 1 EXCEPT SELECT 1")
        self.assertEqual(len(r.rows), 0)

    def test_union_string_literals(self):
        r = self.db.execute("SELECT 'hello' UNION SELECT 'world' ORDER BY 1")
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, ['hello', 'world'])


class TestSetOpOrderByOrdinal(unittest.TestCase):
    """Test ORDER BY with ordinal position (ORDER BY 1, 2)."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t (a INT, b TEXT)")
        self.db.execute("INSERT INTO t VALUES (2, 'x')")
        self.db.execute("INSERT INTO t VALUES (1, 'y')")
        self.db.execute("INSERT INTO t VALUES (3, 'z')")

    def test_order_by_ordinal(self):
        r = self.db.execute(
            "SELECT a, b FROM t WHERE a <= 2 "
            "UNION "
            "SELECT a, b FROM t WHERE a >= 2 "
            "ORDER BY 1"
        )
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2, 3])

    def test_order_by_second_column(self):
        r = self.db.execute(
            "SELECT a, b FROM t WHERE a <= 2 "
            "UNION "
            "SELECT a, b FROM t WHERE a >= 2 "
            "ORDER BY 2"
        )
        vals = [row[1] for row in r.rows]
        self.assertEqual(vals, ['x', 'y', 'z'])


class TestSetOpMultipleTables(unittest.TestCase):
    """Test set operations across many tables."""

    def setUp(self):
        self.db = SetOpDB()
        for tname in ['t1', 't2', 't3', 't4']:
            self.db.execute(f"CREATE TABLE {tname} (val INT)")

    def test_four_way_union(self):
        self.db.execute("INSERT INTO t1 VALUES (1)")
        self.db.execute("INSERT INTO t2 VALUES (2)")
        self.db.execute("INSERT INTO t3 VALUES (3)")
        self.db.execute("INSERT INTO t4 VALUES (4)")
        r = self.db.execute(
            "SELECT val FROM t1 UNION SELECT val FROM t2 "
            "UNION SELECT val FROM t3 UNION SELECT val FROM t4 "
            "ORDER BY val"
        )
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2, 3, 4])


class TestSetOpMixedTypes(unittest.TestCase):
    """Test set operations with mixed value types."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (val INT)")
        self.db.execute("CREATE TABLE t2 (val INT)")
        self.db.execute("INSERT INTO t1 VALUES (1)")
        self.db.execute("INSERT INTO t1 VALUES (2)")
        self.db.execute("INSERT INTO t2 VALUES (2)")
        self.db.execute("INSERT INTO t2 VALUES (3)")

    def test_union_preserves_types(self):
        r = self.db.execute("SELECT val FROM t1 UNION SELECT val FROM t2 ORDER BY val")
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2, 3])
        self.assertIsInstance(vals[0], int)


class TestSetOpCTEBody(unittest.TestCase):
    """Test set operations inside CTE body definitions."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t1 (x INT)")
        self.db.execute("CREATE TABLE t2 (x INT)")
        for v in [1, 2, 3]:
            self.db.execute(f"INSERT INTO t1 VALUES ({v})")
        for v in [3, 4, 5]:
            self.db.execute(f"INSERT INTO t2 VALUES ({v})")

    def test_intersect_in_cte_body(self):
        r = self.db.execute(
            "WITH common AS ("
            "  SELECT x FROM t1 "
            "  INTERSECT "
            "  SELECT x FROM t2"
            ") "
            "SELECT x FROM common"
        )
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 3)

    def test_except_in_cte_body(self):
        r = self.db.execute(
            "WITH only_t1 AS ("
            "  SELECT x FROM t1 "
            "  EXCEPT "
            "  SELECT x FROM t2"
            ") "
            "SELECT x FROM only_t1 ORDER BY x"
        )
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 2])

    def test_chained_in_cte_body(self):
        self.db.execute("CREATE TABLE t3 (x INT)")
        self.db.execute("INSERT INTO t3 VALUES (1)")
        r = self.db.execute(
            "WITH result AS ("
            "  SELECT x FROM t1 "
            "  EXCEPT "
            "  SELECT x FROM t2 "
            "  EXCEPT "
            "  SELECT x FROM t3"
            ") "
            "SELECT x FROM result ORDER BY x"
        )
        # t1 - t2 = {1,2}, then - t3 = {2}
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [2])


class TestSetOpEdgeCases(unittest.TestCase):
    """Various edge cases."""

    def setUp(self):
        self.db = SetOpDB()

    def test_union_same_table_different_filters(self):
        self.db.execute("CREATE TABLE t (id INT, status TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'active')")
        self.db.execute("INSERT INTO t VALUES (2, 'inactive')")
        self.db.execute("INSERT INTO t VALUES (3, 'active')")
        r = self.db.execute(
            "SELECT id FROM t WHERE status = 'active' "
            "UNION "
            "SELECT id FROM t WHERE id = 2"
        )
        ids = sorted(row[0] for row in r.rows)
        self.assertEqual(ids, [1, 2, 3])

    def test_except_then_count(self):
        """Verify we can use set op result further."""
        self.db.execute("CREATE TABLE a (x INT)")
        self.db.execute("CREATE TABLE b (x INT)")
        self.db.execute("INSERT INTO a VALUES (1)")
        self.db.execute("INSERT INTO a VALUES (2)")
        self.db.execute("INSERT INTO a VALUES (3)")
        self.db.execute("INSERT INTO b VALUES (2)")
        # EXCEPT gives {1,3}
        r = self.db.execute("SELECT x FROM a EXCEPT SELECT x FROM b ORDER BY x")
        vals = [row[0] for row in r.rows]
        self.assertEqual(vals, [1, 3])

    def test_multiple_columns_intersection(self):
        self.db.execute("CREATE TABLE p1 (a INT, b TEXT)")
        self.db.execute("CREATE TABLE p2 (a INT, b TEXT)")
        self.db.execute("INSERT INTO p1 VALUES (1, 'x')")
        self.db.execute("INSERT INTO p1 VALUES (2, 'y')")
        self.db.execute("INSERT INTO p2 VALUES (1, 'x')")
        self.db.execute("INSERT INTO p2 VALUES (2, 'z')")  # different b
        r = self.db.execute("SELECT a, b FROM p1 INTERSECT SELECT a, b FROM p2")
        self.assertEqual(len(r.rows), 1)  # only (1,'x') matches both
        self.assertEqual(r.rows[0], [1, 'x'])

    def test_union_with_case_expression(self):
        self.db.execute("CREATE TABLE t (val INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")
        r = self.db.execute(
            "SELECT CASE WHEN val = 1 THEN 'one' ELSE 'other' END AS label FROM t "
            "UNION "
            "SELECT 'extra'"
        )
        labels = sorted(row[0] for row in r.rows)
        self.assertIn('one', labels)
        self.assertIn('other', labels)
        self.assertIn('extra', labels)


class TestSetOpPractical(unittest.TestCase):
    """Practical use cases for set operations."""

    def setUp(self):
        self.db = SetOpDB()
        # Two membership lists
        self.db.execute("CREATE TABLE club_a (member TEXT)")
        self.db.execute("CREATE TABLE club_b (member TEXT)")
        self.db.execute("INSERT INTO club_a VALUES ('Alice')")
        self.db.execute("INSERT INTO club_a VALUES ('Bob')")
        self.db.execute("INSERT INTO club_a VALUES ('Charlie')")
        self.db.execute("INSERT INTO club_b VALUES ('Bob')")
        self.db.execute("INSERT INTO club_b VALUES ('Charlie')")
        self.db.execute("INSERT INTO club_b VALUES ('Diana')")

    def test_all_members(self):
        """All unique members across both clubs."""
        r = self.db.execute(
            "SELECT member FROM club_a UNION SELECT member FROM club_b ORDER BY member"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob', 'Charlie', 'Diana'])

    def test_members_in_both(self):
        """Members in both clubs."""
        r = self.db.execute(
            "SELECT member FROM club_a INTERSECT SELECT member FROM club_b ORDER BY member"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Bob', 'Charlie'])

    def test_exclusive_to_club_a(self):
        """Members only in club A."""
        r = self.db.execute(
            "SELECT member FROM club_a EXCEPT SELECT member FROM club_b"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice'])

    def test_exclusive_to_club_b(self):
        """Members only in club B."""
        r = self.db.execute(
            "SELECT member FROM club_b EXCEPT SELECT member FROM club_a"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Diana'])

    def test_symmetric_difference(self):
        """Members in exactly one club (symmetric difference).
        (A EXCEPT B) UNION (B EXCEPT A) -- needs parens for correct grouping"""
        r = self.db.execute(
            "(SELECT member FROM club_a EXCEPT SELECT member FROM club_b) "
            "UNION "
            "(SELECT member FROM club_b EXCEPT SELECT member FROM club_a) "
            "ORDER BY member"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Diana'])


class TestSetOpCaseInsensitive(unittest.TestCase):
    """Test case insensitivity of keywords."""

    def setUp(self):
        self.db = SetOpDB()
        self.db.execute("CREATE TABLE t (x INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")

    def test_union_lowercase(self):
        r = self.db.execute("select x from t union select x from t")
        vals = sorted(row[0] for row in r.rows)
        self.assertEqual(vals, [1, 2])


if __name__ == '__main__':
    unittest.main()
