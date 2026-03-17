"""
Tests for C257: Table Aliases Fix
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from table_aliases import TableAliasDB


def make_db():
    """Create a test database with sample data."""
    db = TableAliasDB()
    db.execute("CREATE TABLE employees (id INT, name TEXT, dept_id INT, salary INT)")
    db.execute("INSERT INTO employees VALUES (1, 'Alice', 10, 70000)")
    db.execute("INSERT INTO employees VALUES (2, 'Bob', 20, 60000)")
    db.execute("INSERT INTO employees VALUES (3, 'Carol', 10, 80000)")
    db.execute("INSERT INTO employees VALUES (4, 'Dave', 30, 55000)")
    db.execute("INSERT INTO employees VALUES (5, 'Eve', 20, 75000)")

    db.execute("CREATE TABLE departments (id INT, name TEXT, budget INT)")
    db.execute("INSERT INTO departments VALUES (10, 'Engineering', 500000)")
    db.execute("INSERT INTO departments VALUES (20, 'Marketing', 300000)")
    db.execute("INSERT INTO departments VALUES (30, 'Sales', 200000)")

    db.execute("CREATE TABLE projects (id INT, name TEXT, dept_id INT, lead_id INT)")
    db.execute("INSERT INTO projects VALUES (100, 'Alpha', 10, 1)")
    db.execute("INSERT INTO projects VALUES (101, 'Beta', 20, 2)")
    db.execute("INSERT INTO projects VALUES (102, 'Gamma', 10, 3)")
    db.execute("INSERT INTO projects VALUES (103, 'Delta', 30, 4)")
    return db


# =============================================================================
# Core: Basic table alias resolution (the original bug)
# =============================================================================

class TestBasicAliases(unittest.TestCase):
    """Test the core bug fix: FROM t1 x JOIN t2 y ON x.col = y.col"""

    def setUp(self):
        self.db = make_db()

    def test_join_with_aliases(self):
        """The original bug: aliases in JOIN ON clause."""
        r = self.db.execute(
            "SELECT e.name, d.name FROM employees e "
            "JOIN departments d ON e.dept_id = d.id"
        )
        names = sorted([(row[0], row[1]) for row in r.rows])
        self.assertIn(('Alice', 'Engineering'), names)
        self.assertIn(('Bob', 'Marketing'), names)
        self.assertIn(('Dave', 'Sales'), names)
        self.assertEqual(len(names), 5)

    def test_join_with_as_keyword(self):
        """Aliases using explicit AS keyword."""
        r = self.db.execute(
            "SELECT e.name, d.name FROM employees AS e "
            "JOIN departments AS d ON e.dept_id = d.id"
        )
        names = sorted([(row[0], row[1]) for row in r.rows])
        self.assertIn(('Alice', 'Engineering'), names)
        self.assertEqual(len(names), 5)

    def test_join_without_aliases_still_works(self):
        """Non-aliased JOINs should still work (regression check)."""
        r = self.db.execute(
            "SELECT employees.name, departments.name FROM employees "
            "JOIN departments ON employees.dept_id = departments.id"
        )
        self.assertEqual(len(r.rows), 5)

    def test_alias_in_select_list(self):
        """Alias-qualified columns in SELECT."""
        r = self.db.execute(
            "SELECT e.id, e.name FROM employees e WHERE e.id = 1"
        )
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 1)
        self.assertEqual(r.rows[0][1], 'Alice')

    def test_alias_in_where(self):
        """Alias-qualified columns in WHERE clause."""
        r = self.db.execute(
            "SELECT e.name FROM employees e WHERE e.salary > 65000"
        )
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Alice', 'Carol', 'Eve'])

    def test_single_table_alias(self):
        """Single table with alias (no JOIN)."""
        r = self.db.execute("SELECT e.name FROM employees e ORDER BY e.name")
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'])


# =============================================================================
# Multi-table JOINs with aliases
# =============================================================================

class TestMultiTableAliases(unittest.TestCase):
    """Test aliases across multiple JOINs."""

    def setUp(self):
        self.db = make_db()

    def test_three_table_join(self):
        """Three-table JOIN with aliases."""
        r = self.db.execute(
            "SELECT e.name, d.name, p.name "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "JOIN projects p ON p.lead_id = e.id"
        )
        rows = sorted(r.rows)
        self.assertTrue(len(rows) >= 4)
        # Alice leads Alpha in Engineering
        alice_rows = [row for row in rows if row[0] == 'Alice']
        self.assertTrue(len(alice_rows) >= 1)
        self.assertEqual(alice_rows[0][1], 'Engineering')
        self.assertEqual(alice_rows[0][2], 'Alpha')

    def test_mixed_alias_and_real_name(self):
        """Mix of aliased and non-aliased table references."""
        r = self.db.execute(
            "SELECT e.name, departments.name FROM employees e "
            "JOIN departments ON e.dept_id = departments.id"
        )
        self.assertEqual(len(r.rows), 5)

    def test_left_join_with_aliases(self):
        """LEFT JOIN with aliases."""
        self.db.execute("INSERT INTO employees VALUES (6, 'Frank', 99, 50000)")
        r = self.db.execute(
            "SELECT e.name, d.name FROM employees e "
            "LEFT JOIN departments d ON e.dept_id = d.id "
            "ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertIn('Frank', names)
        frank_row = [row for row in r.rows if row[0] == 'Frank'][0]
        self.assertIsNone(frank_row[1])  # No matching department

    def test_cross_join_with_aliases(self):
        """CROSS JOIN with aliases."""
        r = self.db.execute(
            "SELECT e.name, d.name FROM employees e "
            "CROSS JOIN departments d"
        )
        # 5 employees x 3 departments = 15
        self.assertEqual(len(r.rows), 15)


# =============================================================================
# Self-joins (same table, different aliases)
# =============================================================================

class TestSelfJoins(unittest.TestCase):
    """Test self-joins where the same table appears twice with different aliases."""

    def setUp(self):
        self.db = make_db()

    def test_self_join_basic(self):
        """Self-join: find employees in the same department."""
        r = self.db.execute(
            "SELECT e1.name, e2.name "
            "FROM employees e1 "
            "JOIN employees e2 ON e1.dept_id = e2.dept_id "
            "WHERE e1.id < e2.id"
        )
        # Dept 10: (Alice, Carol), Dept 20: (Bob, Eve)
        pairs = sorted([(row[0], row[1]) for row in r.rows])
        self.assertIn(('Alice', 'Carol'), pairs)
        self.assertIn(('Bob', 'Eve'), pairs)
        self.assertEqual(len(pairs), 2)

    def test_self_join_inequality(self):
        """Self-join with inequality: employees earning more than others."""
        r = self.db.execute(
            "SELECT e1.name, e2.name "
            "FROM employees e1 "
            "JOIN employees e2 ON e1.dept_id = e2.dept_id "
            "WHERE e1.salary > e2.salary"
        )
        # In dept 10: Carol(80k) > Alice(70k)
        # In dept 20: Eve(75k) > Bob(60k)
        pairs = sorted([(row[0], row[1]) for row in r.rows])
        self.assertIn(('Carol', 'Alice'), pairs)
        self.assertIn(('Eve', 'Bob'), pairs)

    def test_self_join_with_aggregation(self):
        """Self-join with GROUP BY."""
        r = self.db.execute(
            "SELECT e1.dept_id, COUNT(*) as pair_count "
            "FROM employees e1 "
            "JOIN employees e2 ON e1.dept_id = e2.dept_id "
            "WHERE e1.id < e2.id "
            "GROUP BY e1.dept_id"
        )
        rows = sorted(r.rows)
        # Dept 10: 1 pair, Dept 20: 1 pair
        self.assertEqual(len(rows), 2)


# =============================================================================
# Aliases with aggregation
# =============================================================================

class TestAliasesWithAggregation(unittest.TestCase):
    """Test aliases combined with GROUP BY, HAVING, and aggregate functions."""

    def setUp(self):
        self.db = make_db()

    def test_group_by_with_alias(self):
        """GROUP BY using alias-qualified column."""
        r = self.db.execute(
            "SELECT d.name, COUNT(*) as cnt "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "GROUP BY d.name "
            "ORDER BY d.name"
        )
        rows = r.rows
        self.assertEqual(len(rows), 3)
        # Engineering: 2, Marketing: 2, Sales: 1
        eng = [row for row in rows if row[0] == 'Engineering'][0]
        self.assertEqual(eng[1], 2)

    def test_having_with_alias(self):
        """HAVING clause with alias-qualified reference."""
        r = self.db.execute(
            "SELECT d.name, COUNT(*) as cnt "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "GROUP BY d.name "
            "HAVING cnt >= 2"
        )
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Engineering', 'Marketing'])

    def test_sum_with_alias(self):
        """SUM with alias-qualified column."""
        r = self.db.execute(
            "SELECT d.name, SUM(e.salary) as total_salary "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "GROUP BY d.name "
            "ORDER BY d.name"
        )
        eng = [row for row in r.rows if row[0] == 'Engineering'][0]
        self.assertEqual(eng[1], 150000)  # Alice 70k + Carol 80k

    def test_avg_with_alias(self):
        """AVG with alias-qualified column."""
        r = self.db.execute(
            "SELECT d.name, AVG(e.salary) as avg_salary "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "GROUP BY d.name "
            "ORDER BY d.name"
        )
        eng = [row for row in r.rows if row[0] == 'Engineering'][0]
        self.assertEqual(eng[1], 75000)  # (70k + 80k) / 2

    def test_min_max_with_alias(self):
        """MIN/MAX with alias-qualified column."""
        r = self.db.execute(
            "SELECT d.name, MIN(e.salary) as min_sal, MAX(e.salary) as max_sal "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "GROUP BY d.name "
            "ORDER BY d.name"
        )
        eng = [row for row in r.rows if row[0] == 'Engineering'][0]
        self.assertEqual(eng[1], 70000)
        self.assertEqual(eng[2], 80000)


# =============================================================================
# Aliases with ORDER BY
# =============================================================================

class TestAliasesWithOrderBy(unittest.TestCase):
    """Test aliases in ORDER BY clauses."""

    def setUp(self):
        self.db = make_db()

    def test_order_by_alias_column(self):
        """ORDER BY using alias-qualified column."""
        r = self.db.execute(
            "SELECT e.name, e.salary FROM employees e ORDER BY e.salary"
        )
        salaries = [row[1] for row in r.rows]
        self.assertEqual(salaries, sorted(salaries))

    def test_order_by_alias_desc(self):
        """ORDER BY DESC using alias."""
        r = self.db.execute(
            "SELECT e.name, e.salary FROM employees e ORDER BY e.salary DESC"
        )
        salaries = [row[1] for row in r.rows]
        self.assertEqual(salaries, sorted(salaries, reverse=True))

    def test_order_by_joined_alias(self):
        """ORDER BY on joined table alias."""
        r = self.db.execute(
            "SELECT e.name, d.name "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "ORDER BY d.name, e.name"
        )
        dept_names = [row[1] for row in r.rows]
        self.assertEqual(dept_names, sorted(dept_names))


# =============================================================================
# Aliases with DISTINCT
# =============================================================================

class TestAliasesWithDistinct(unittest.TestCase):
    """Test aliases with DISTINCT."""

    def setUp(self):
        self.db = make_db()

    def test_distinct_with_alias(self):
        """DISTINCT on alias-qualified query."""
        r = self.db.execute(
            "SELECT DISTINCT e.dept_id FROM employees e ORDER BY e.dept_id"
        )
        dept_ids = [row[0] for row in r.rows]
        self.assertEqual(dept_ids, [10, 20, 30])


# =============================================================================
# Aliases with LIMIT/OFFSET
# =============================================================================

class TestAliasesWithLimit(unittest.TestCase):
    """Test aliases with LIMIT and OFFSET."""

    def setUp(self):
        self.db = make_db()

    def test_limit_with_alias(self):
        """LIMIT with alias."""
        r = self.db.execute(
            "SELECT e.name FROM employees e ORDER BY e.name LIMIT 3"
        )
        self.assertEqual(len(r.rows), 3)
        self.assertEqual(r.rows[0][0], 'Alice')

    def test_offset_with_alias(self):
        """LIMIT/OFFSET with alias."""
        r = self.db.execute(
            "SELECT e.name FROM employees e ORDER BY e.name LIMIT 2 OFFSET 2"
        )
        self.assertEqual(len(r.rows), 2)
        self.assertEqual(r.rows[0][0], 'Carol')
        self.assertEqual(r.rows[1][0], 'Dave')


# =============================================================================
# Aliases with subqueries
# =============================================================================

class TestAliasesWithSubqueries(unittest.TestCase):
    """Test aliases combined with subqueries."""

    def setUp(self):
        self.db = make_db()

    def test_alias_with_in_subquery(self):
        """Alias in outer query with IN subquery."""
        r = self.db.execute(
            "SELECT e.name FROM employees e "
            "WHERE e.dept_id IN (SELECT id FROM departments WHERE budget > 250000) "
            "ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob', 'Carol', 'Eve'])

    def test_alias_with_exists_subquery(self):
        """Alias in outer query with EXISTS subquery."""
        r = self.db.execute(
            "SELECT e.name FROM employees e "
            "WHERE EXISTS (SELECT 1 FROM projects p WHERE p.lead_id = e.id) "
            "ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob', 'Carol', 'Dave'])

    def test_alias_with_scalar_subquery(self):
        """Alias with scalar subquery in SELECT."""
        r = self.db.execute(
            "SELECT e.name, "
            "(SELECT d.name FROM departments d WHERE d.id = e.dept_id) as dept "
            "FROM employees e WHERE e.id = 1"
        )
        self.assertEqual(r.rows[0][0], 'Alice')
        self.assertEqual(r.rows[0][1], 'Engineering')


# =============================================================================
# Aliases with derived tables
# =============================================================================

class TestAliasesWithDerivedTables(unittest.TestCase):
    """Test aliases combined with derived tables."""

    def setUp(self):
        self.db = make_db()

    def test_derived_table_join_with_alias(self):
        """JOIN derived table with aliased real table."""
        r = self.db.execute(
            "SELECT e.name, sub.dept_name "
            "FROM employees e "
            "JOIN (SELECT id, name as dept_name FROM departments) AS sub "
            "ON e.dept_id = sub.id "
            "ORDER BY e.name"
        )
        self.assertEqual(len(r.rows), 5)
        alice = [row for row in r.rows if row[0] == 'Alice'][0]
        self.assertEqual(alice[1], 'Engineering')

    def test_alias_in_derived_table_source(self):
        """Alias used inside a derived table definition."""
        r = self.db.execute(
            "SELECT sub.name, sub.salary "
            "FROM (SELECT name, salary FROM employees WHERE salary > 65000) AS sub "
            "ORDER BY sub.salary"
        )
        names = [row[0] for row in r.rows]
        self.assertIn('Alice', names)
        self.assertIn('Carol', names)
        self.assertIn('Eve', names)


# =============================================================================
# Aliases with CTEs
# =============================================================================

class TestAliasesWithCTEs(unittest.TestCase):
    """Test aliases combined with CTEs."""

    def setUp(self):
        self.db = make_db()

    def test_cte_with_aliased_join(self):
        """CTE result joined with aliased table."""
        r = self.db.execute(
            "WITH high_earners AS (SELECT id, name, dept_id FROM employees WHERE salary > 65000) "
            "SELECT h.name, d.name "
            "FROM high_earners h "
            "JOIN departments d ON h.dept_id = d.id "
            "ORDER BY h.name"
        )
        names = [row[0] for row in r.rows]
        self.assertIn('Alice', names)
        self.assertIn('Carol', names)
        self.assertIn('Eve', names)


# =============================================================================
# Aliases with WHERE conditions (various operators)
# =============================================================================

class TestAliasesInWhereConditions(unittest.TestCase):
    """Test alias-qualified columns in various WHERE expressions."""

    def setUp(self):
        self.db = make_db()

    def test_between_with_alias(self):
        """BETWEEN using alias-qualified column."""
        r = self.db.execute(
            "SELECT e.name FROM employees e WHERE e.salary BETWEEN 60000 AND 70000 ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob'])

    def test_in_list_with_alias(self):
        """IN list using alias-qualified column."""
        r = self.db.execute(
            "SELECT e.name FROM employees e WHERE e.dept_id IN (10, 30) ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Carol', 'Dave'])

    def test_is_null_with_alias(self):
        """IS NULL using alias-qualified column."""
        self.db.execute("CREATE TABLE nullable_t (id INT, val TEXT)")
        self.db.execute("INSERT INTO nullable_t VALUES (1, 'hello')")
        self.db.execute("INSERT INTO nullable_t VALUES (2, NULL)")
        r = self.db.execute("SELECT n.id FROM nullable_t n WHERE n.val IS NULL")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 2)

    def test_is_not_null_with_alias(self):
        """IS NOT NULL using alias-qualified column."""
        self.db.execute("CREATE TABLE nullable_t (id INT, val TEXT)")
        self.db.execute("INSERT INTO nullable_t VALUES (1, 'hello')")
        self.db.execute("INSERT INTO nullable_t VALUES (2, NULL)")
        r = self.db.execute("SELECT n.id FROM nullable_t n WHERE n.val IS NOT NULL")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 1)

    def test_like_with_alias(self):
        """LIKE using alias-qualified column."""
        r = self.db.execute(
            "SELECT e.name FROM employees e WHERE e.name LIKE 'A%'"
        )
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 'Alice')

    def test_and_or_with_aliases(self):
        """Complex AND/OR with aliases."""
        r = self.db.execute(
            "SELECT e.name FROM employees e "
            "WHERE (e.dept_id = 10 AND e.salary > 75000) OR e.dept_id = 30 "
            "ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Carol', 'Dave'])

    def test_not_with_alias(self):
        """NOT with alias-qualified column."""
        r = self.db.execute(
            "SELECT e.name FROM employees e WHERE NOT e.dept_id = 10 ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Bob', 'Dave', 'Eve'])


# =============================================================================
# Aliases with arithmetic expressions
# =============================================================================

class TestAliasesWithArithmetic(unittest.TestCase):
    """Test alias-qualified columns in arithmetic expressions."""

    def setUp(self):
        self.db = make_db()

    def test_arithmetic_in_select(self):
        """Arithmetic with alias-qualified column in SELECT."""
        r = self.db.execute(
            "SELECT e.name, e.salary * 2 as double_sal FROM employees e WHERE e.id = 1"
        )
        self.assertEqual(r.rows[0][0], 'Alice')
        self.assertEqual(r.rows[0][1], 140000)

    def test_arithmetic_in_where(self):
        """Arithmetic with alias in WHERE."""
        r = self.db.execute(
            "SELECT e.name FROM employees e WHERE e.salary + 10000 > 80000 ORDER BY e.name"
        )
        names = [row[0] for row in r.rows]
        self.assertIn('Carol', names)
        self.assertIn('Eve', names)


# =============================================================================
# SELECT * with aliases
# =============================================================================

class TestSelectStarWithAliases(unittest.TestCase):
    """Test SELECT * when tables have aliases."""

    def setUp(self):
        self.db = make_db()

    def test_select_star_single_aliased_table(self):
        """SELECT * from an aliased table."""
        r = self.db.execute("SELECT * FROM employees e WHERE e.id = 1")
        self.assertEqual(len(r.rows), 1)
        # Should have all employee columns
        self.assertIn(1, r.rows[0])
        self.assertIn('Alice', r.rows[0])

    def test_select_star_joined_aliased_tables(self):
        """SELECT * from aliased JOINed tables."""
        r = self.db.execute(
            "SELECT * FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "WHERE e.id = 1"
        )
        self.assertEqual(len(r.rows), 1)
        # Should have columns from both tables
        row = r.rows[0]
        self.assertIn('Alice', row)
        self.assertIn('Engineering', row)


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases for table alias handling."""

    def setUp(self):
        self.db = make_db()

    def test_alias_same_as_other_table_name(self):
        """Alias that matches another table's actual name."""
        # Use 'departments' as alias for employees -- should not cause confusion
        r = self.db.execute(
            "SELECT departments.name FROM employees departments WHERE departments.id = 1"
        )
        self.assertEqual(r.rows[0][0], 'Alice')

    def test_single_char_alias(self):
        """Single character aliases."""
        r = self.db.execute(
            "SELECT a.name, b.name FROM employees a "
            "JOIN departments b ON a.dept_id = b.id "
            "WHERE a.id = 1"
        )
        self.assertEqual(r.rows[0][0], 'Alice')
        self.assertEqual(r.rows[0][1], 'Engineering')

    def test_long_alias(self):
        """Long alias names."""
        r = self.db.execute(
            "SELECT emp_table.name FROM employees emp_table WHERE emp_table.id = 1"
        )
        self.assertEqual(r.rows[0][0], 'Alice')

    def test_alias_with_unqualified_columns(self):
        """Mix of alias-qualified and unqualified columns."""
        r = self.db.execute(
            "SELECT name, e.salary FROM employees e WHERE e.id = 1"
        )
        self.assertEqual(r.rows[0][0], 'Alice')
        self.assertEqual(r.rows[0][1], 70000)

    def test_multiple_conditions_with_aliases(self):
        """Multiple join conditions using aliases."""
        r = self.db.execute(
            "SELECT e.name, p.name "
            "FROM employees e "
            "JOIN projects p ON p.lead_id = e.id AND p.dept_id = e.dept_id "
            "ORDER BY e.name"
        )
        self.assertTrue(len(r.rows) >= 1)

    def test_alias_preserves_null_handling(self):
        """NULL values should work correctly with aliases."""
        self.db.execute("CREATE TABLE t1 (id INT, val INT)")
        self.db.execute("CREATE TABLE t2 (id INT, ref_id INT)")
        self.db.execute("INSERT INTO t1 VALUES (1, NULL)")
        self.db.execute("INSERT INTO t2 VALUES (10, 1)")
        r = self.db.execute(
            "SELECT a.val FROM t1 a JOIN t2 b ON a.id = b.ref_id"
        )
        self.assertEqual(len(r.rows), 1)
        self.assertIsNone(r.rows[0][0])


# =============================================================================
# Aliases with CASE expressions
# =============================================================================

class TestAliasesWithCase(unittest.TestCase):
    """Test aliases in CASE expressions."""

    def setUp(self):
        self.db = make_db()

    def test_case_with_alias(self):
        """CASE WHEN using alias-qualified column."""
        r = self.db.execute(
            "SELECT e.name, "
            "CASE WHEN e.salary > 70000 THEN 'high' "
            "     WHEN e.salary > 55000 THEN 'mid' "
            "     ELSE 'low' END as level "
            "FROM employees e WHERE e.id = 3"
        )
        self.assertEqual(r.rows[0][0], 'Carol')
        self.assertEqual(r.rows[0][1], 'high')


# =============================================================================
# Aliases with functions
# =============================================================================

class TestAliasesWithFunctions(unittest.TestCase):
    """Test aliases with built-in functions."""

    def setUp(self):
        self.db = make_db()

    def test_upper_with_alias(self):
        """UPPER() with alias-qualified column."""
        r = self.db.execute(
            "SELECT UPPER(e.name) FROM employees e WHERE e.id = 1"
        )
        self.assertEqual(r.rows[0][0], 'ALICE')

    def test_coalesce_with_alias(self):
        """COALESCE with alias-qualified column."""
        self.db.execute("CREATE TABLE nullable_t (id INT, val TEXT)")
        self.db.execute("INSERT INTO nullable_t VALUES (1, NULL)")
        r = self.db.execute(
            "SELECT COALESCE(n.val, 'default') FROM nullable_t n"
        )
        self.assertEqual(r.rows[0][0], 'default')


# =============================================================================
# Backward compatibility: C256 derived table tests
# =============================================================================

class TestBackwardCompat(unittest.TestCase):
    """Ensure existing functionality still works."""

    def setUp(self):
        self.db = make_db()

    def test_basic_select(self):
        """Basic SELECT without aliases."""
        r = self.db.execute("SELECT name FROM employees WHERE id = 1")
        self.assertEqual(r.rows[0][0], 'Alice')

    def test_basic_join(self):
        """Basic JOIN without aliases."""
        r = self.db.execute(
            "SELECT employees.name FROM employees "
            "JOIN departments ON employees.dept_id = departments.id "
            "WHERE departments.name = 'Engineering'"
        )
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Alice', 'Carol'])

    def test_insert_update_delete(self):
        """DML operations still work."""
        self.db.execute("INSERT INTO employees VALUES (6, 'Frank', 10, 50000)")
        r = self.db.execute("SELECT name FROM employees WHERE id = 6")
        self.assertEqual(r.rows[0][0], 'Frank')

        self.db.execute("UPDATE employees SET salary = 55000 WHERE id = 6")
        r = self.db.execute("SELECT salary FROM employees WHERE id = 6")
        self.assertEqual(r.rows[0][0], 55000)

        self.db.execute("DELETE FROM employees WHERE id = 6")
        r = self.db.execute("SELECT * FROM employees WHERE id = 6")
        self.assertEqual(len(r.rows), 0)

    def test_create_drop_table(self):
        """DDL operations still work."""
        self.db.execute("CREATE TABLE temp_t (id INT, name TEXT)")
        self.db.execute("INSERT INTO temp_t VALUES (1, 'test')")
        r = self.db.execute("SELECT * FROM temp_t")
        self.assertEqual(len(r.rows), 1)
        self.db.execute("DROP TABLE temp_t")

    def test_derived_table(self):
        """Derived tables still work."""
        r = self.db.execute(
            "SELECT sub.name FROM "
            "(SELECT name, salary FROM employees WHERE salary > 70000) AS sub "
            "ORDER BY sub.name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Carol', 'Eve'])

    def test_cte(self):
        """CTEs still work."""
        r = self.db.execute(
            "WITH eng AS (SELECT name FROM employees WHERE dept_id = 10) "
            "SELECT name FROM eng ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Carol'])

    def test_set_operations(self):
        """UNION still works."""
        r = self.db.execute(
            "SELECT name FROM employees WHERE dept_id = 10 "
            "UNION "
            "SELECT name FROM employees WHERE salary > 70000"
        )
        names = sorted([row[0] for row in r.rows])
        self.assertIn('Alice', names)
        self.assertIn('Carol', names)
        self.assertIn('Eve', names)

    def test_group_by_having(self):
        """GROUP BY/HAVING still works."""
        r = self.db.execute(
            "SELECT dept_id, COUNT(*) as cnt FROM employees "
            "GROUP BY dept_id HAVING cnt >= 2 ORDER BY dept_id"
        )
        self.assertEqual(len(r.rows), 2)

    def test_subquery_in_where(self):
        """Subqueries in WHERE still work."""
        r = self.db.execute(
            "SELECT name FROM employees "
            "WHERE dept_id IN (SELECT id FROM departments WHERE budget > 400000)"
        )
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Alice', 'Carol'])

    def test_order_by_limit(self):
        """ORDER BY with LIMIT still works."""
        r = self.db.execute(
            "SELECT name FROM employees ORDER BY salary DESC LIMIT 3"
        )
        self.assertEqual(len(r.rows), 3)
        self.assertEqual(r.rows[0][0], 'Carol')


# =============================================================================
# Complex combined scenarios
# =============================================================================

class TestComplexScenarios(unittest.TestCase):
    """Complex queries combining multiple features with aliases."""

    def setUp(self):
        self.db = make_db()

    def test_alias_join_group_having_order_limit(self):
        """Full query pipeline with aliases."""
        r = self.db.execute(
            "SELECT d.name, COUNT(*) as cnt, AVG(e.salary) as avg_sal "
            "FROM employees e "
            "JOIN departments d ON e.dept_id = d.id "
            "GROUP BY d.name "
            "HAVING cnt >= 2 "
            "ORDER BY avg_sal DESC "
            "LIMIT 2"
        )
        self.assertEqual(len(r.rows), 2)

    def test_alias_self_join_with_subquery(self):
        """Self-join with subquery filter."""
        r = self.db.execute(
            "SELECT e1.name, e2.name "
            "FROM employees e1 "
            "JOIN employees e2 ON e1.dept_id = e2.dept_id "
            "WHERE e1.id < e2.id "
            "AND e1.dept_id IN (SELECT id FROM departments WHERE budget > 250000) "
            "ORDER BY e1.name"
        )
        pairs = [(row[0], row[1]) for row in r.rows]
        self.assertTrue(len(pairs) >= 1)

    def test_three_aliases_with_aggregation(self):
        """Three-table join with aliases and aggregation."""
        r = self.db.execute(
            "SELECT d.name, COUNT(DISTINCT p.id) as project_count "
            "FROM departments d "
            "JOIN employees e ON e.dept_id = d.id "
            "JOIN projects p ON p.lead_id = e.id "
            "GROUP BY d.name "
            "ORDER BY d.name"
        )
        self.assertTrue(len(r.rows) >= 1)


if __name__ == '__main__':
    unittest.main()
