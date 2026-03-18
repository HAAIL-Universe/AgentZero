"""
Tests for C267: Common Table Expressions (CTEs)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cte import (
    CTEDB, CTEParser, parse_cte_sql, parse_cte_sql_multi,
    CTEStatement, CTEDefinition, UnionQuery,
    MAX_RECURSION_DEPTH,
)
from mini_database import ParseError, CompileError


@pytest.fixture
def db():
    d = CTEDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary FLOAT, manager_id INT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000, NULL)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 80000, 1)")
    d.execute("INSERT INTO employees VALUES (3, 'Carol', 'Sales', 70000, 1)")
    d.execute("INSERT INTO employees VALUES (4, 'Dave', 'Sales', 60000, 3)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 95000, 2)")
    d.execute("INSERT INTO employees VALUES (6, 'Frank', 'Marketing', 65000, 1)")
    return d


@pytest.fixture
def db_with_products():
    d = CTEDB()
    d.execute("CREATE TABLE products (id INT, name TEXT, category TEXT, price FLOAT)")
    d.execute("INSERT INTO products VALUES (1, 'Laptop', 'Electronics', 999.99)")
    d.execute("INSERT INTO products VALUES (2, 'Phone', 'Electronics', 699.99)")
    d.execute("INSERT INTO products VALUES (3, 'Tablet', 'Electronics', 499.99)")
    d.execute("INSERT INTO products VALUES (4, 'Desk', 'Furniture', 299.99)")
    d.execute("INSERT INTO products VALUES (5, 'Chair', 'Furniture', 199.99)")
    d.execute("INSERT INTO products VALUES (6, 'Lamp', 'Furniture', 49.99)")
    return d


# =============================================================================
# Parsing Tests
# =============================================================================

class TestCTEParsing:
    """Tests for CTE parsing."""

    def test_parse_basic_cte(self):
        stmt = parse_cte_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
        assert isinstance(stmt, CTEStatement)
        assert len(stmt.ctes) == 1
        assert stmt.ctes[0].name == 'x'
        assert not stmt.recursive

    def test_parse_multiple_ctes(self):
        sql = "WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT x, y FROM a, b"
        stmt = parse_cte_sql(sql)
        assert isinstance(stmt, CTEStatement)
        assert len(stmt.ctes) == 2
        assert stmt.ctes[0].name == 'a'
        assert stmt.ctes[1].name == 'b'

    def test_parse_cte_with_column_list(self):
        sql = "WITH nums(val) AS (SELECT 42) SELECT val FROM nums"
        stmt = parse_cte_sql(sql)
        assert stmt.ctes[0].column_list == ['val']

    def test_parse_cte_with_multiple_columns(self):
        sql = "WITH data(x, y, z) AS (SELECT 1, 2, 3) SELECT x, y, z FROM data"
        stmt = parse_cte_sql(sql)
        assert stmt.ctes[0].column_list == ['x', 'y', 'z']

    def test_parse_recursive_cte(self):
        sql = """WITH RECURSIVE cnt AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM cnt WHERE n < 5
        ) SELECT n FROM cnt"""
        stmt = parse_cte_sql(sql)
        assert stmt.recursive
        assert len(stmt.ctes) == 1
        assert stmt.ctes[0].name == 'cnt'

    def test_parse_union_in_cte(self):
        sql = "WITH u AS (SELECT 1 AS v UNION ALL SELECT 2) SELECT v FROM u"
        stmt = parse_cte_sql(sql)
        assert isinstance(stmt.ctes[0].query, UnionQuery)

    def test_parse_multi_statement(self):
        sql = "WITH x AS (SELECT 1 AS a) SELECT a FROM x; SELECT 42"
        stmts = parse_cte_sql_multi(sql)
        assert len(stmts) == 2
        assert isinstance(stmts[0], CTEStatement)


# =============================================================================
# Basic CTE Tests
# =============================================================================

class TestBasicCTE:
    """Tests for basic CTE functionality."""

    def test_simple_cte(self, db):
        r = db.execute("WITH eng AS (SELECT * FROM employees WHERE dept = 'Engineering') SELECT name FROM eng")
        names = sorted(r[0] for r in r.rows)
        assert names == ['Alice', 'Bob', 'Eve']

    def test_cte_with_aggregation(self, db):
        r = db.execute("""
            WITH dept_stats AS (
                SELECT dept, AVG(salary) AS avg_sal
                FROM employees
                GROUP BY dept
            )
            SELECT dept, avg_sal FROM dept_stats ORDER BY avg_sal DESC
        """)
        assert r.rows[0][0] == 'Engineering'

    def test_cte_with_where(self, db):
        r = db.execute("""
            WITH high_earners AS (
                SELECT name, salary FROM employees WHERE salary > 75000
            )
            SELECT name FROM high_earners ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']

    def test_cte_with_order_by(self, db):
        r = db.execute("""
            WITH sorted AS (
                SELECT name, salary FROM employees
            )
            SELECT name FROM sorted ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == sorted(names)

    def test_cte_with_limit(self, db):
        r = db.execute("""
            WITH all_emp AS (
                SELECT name, salary FROM employees ORDER BY salary DESC
            )
            SELECT name FROM all_emp LIMIT 3
        """)
        assert len(r.rows) == 3

    def test_cte_select_star(self, db):
        r = db.execute("""
            WITH eng AS (
                SELECT id, name FROM employees WHERE dept = 'Engineering'
            )
            SELECT * FROM eng ORDER BY id
        """)
        assert len(r.columns) == 2
        assert r.rows[0][1] == 'Alice'

    def test_cte_alias(self, db):
        r = db.execute("""
            WITH e AS (SELECT name, salary FROM employees)
            SELECT e.name FROM e WHERE e.salary > 80000 ORDER BY e.name
        """)
        names = [r[0] for r in r.rows]
        assert 'Alice' in names
        assert 'Eve' in names

    def test_cte_from_literal(self):
        d = CTEDB()
        r = d.execute("WITH nums AS (SELECT 42 AS val) SELECT val FROM nums")
        assert r.rows == [[42]]

    def test_cte_distinct(self, db):
        r = db.execute("""
            WITH depts AS (
                SELECT DISTINCT dept FROM employees
            )
            SELECT dept FROM depts ORDER BY dept
        """)
        depts = [r[0] for r in r.rows]
        assert depts == ['Engineering', 'Marketing', 'Sales']


# =============================================================================
# Multiple CTEs
# =============================================================================

class TestMultipleCTEs:
    """Tests for multiple CTE definitions."""

    def test_two_independent_ctes(self, db):
        r = db.execute("""
            WITH
                eng AS (SELECT COUNT(*) AS cnt FROM employees WHERE dept = 'Engineering'),
                sales AS (SELECT COUNT(*) AS cnt FROM employees WHERE dept = 'Sales')
            SELECT eng.cnt AS eng_count, sales.cnt AS sales_count
            FROM eng, sales
        """)
        # Cross join of single-row results
        assert len(r.rows) == 1
        assert r.rows[0][0] == 3  # Engineering count
        assert r.rows[0][1] == 2  # Sales count

    def test_chained_ctes(self, db):
        """Second CTE references first CTE."""
        r = db.execute("""
            WITH
                eng AS (
                    SELECT name, salary FROM employees WHERE dept = 'Engineering'
                ),
                top_eng AS (
                    SELECT name FROM eng WHERE salary > 85000
                )
            SELECT name FROM top_eng ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Eve']

    def test_three_chained_ctes(self, db):
        r = db.execute("""
            WITH
                all_emp AS (SELECT name, dept, salary FROM employees),
                eng AS (SELECT name, salary FROM all_emp WHERE dept = 'Engineering'),
                top_eng AS (SELECT name FROM eng WHERE salary > 85000)
            SELECT name FROM top_eng ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Eve']

    def test_cte_join_with_real_table(self, db):
        r = db.execute("""
            WITH dept_avg AS (
                SELECT dept, AVG(salary) AS avg_sal
                FROM employees
                GROUP BY dept
            )
            SELECT e.name, e.salary, d.avg_sal
            FROM employees AS e
            JOIN dept_avg AS d ON e.dept = d.dept
            WHERE e.salary > d.avg_sal
            ORDER BY e.name
        """)
        # Employees above their department average
        names = [r[0] for r in r.rows]
        assert len(names) > 0

    def test_cte_cross_join(self, db):
        r = db.execute("""
            WITH
                x AS (SELECT 1 AS a),
                y AS (SELECT 2 AS b)
            SELECT a, b FROM x, y
        """)
        assert r.rows == [[1, 2]]


# =============================================================================
# CTE Column Lists
# =============================================================================

class TestCTEColumnLists:
    """Tests for CTEs with explicit column lists."""

    def test_column_rename(self, db):
        r = db.execute("""
            WITH emp_info(employee_name, employee_salary) AS (
                SELECT name, salary FROM employees WHERE dept = 'Engineering'
            )
            SELECT employee_name FROM emp_info ORDER BY employee_name
        """)
        names = [r[0] for r in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']

    def test_column_rename_in_where(self, db):
        r = db.execute("""
            WITH emp_info(employee_name, employee_salary) AS (
                SELECT name, salary FROM employees
            )
            SELECT employee_name FROM emp_info WHERE employee_salary > 80000 ORDER BY employee_name
        """)
        names = [r[0] for r in r.rows]
        assert 'Alice' in names

    def test_column_count_mismatch(self, db):
        with pytest.raises(CompileError, match="column names"):
            db.execute("""
                WITH bad(a, b, c) AS (SELECT name FROM employees)
                SELECT * FROM bad
            """)

    def test_column_list_with_aggregation(self, db):
        r = db.execute("""
            WITH stats(department, avg_salary, emp_count) AS (
                SELECT dept, AVG(salary), COUNT(*)
                FROM employees
                GROUP BY dept
            )
            SELECT department, emp_count FROM stats ORDER BY department
        """)
        assert r.columns[0] == 'department'
        assert r.columns[1] == 'emp_count'


# =============================================================================
# Recursive CTEs
# =============================================================================

class TestRecursiveCTEs:
    """Tests for recursive CTEs."""

    def test_counting_sequence(self, db):
        r = db.execute("""
            WITH RECURSIVE cnt(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM cnt WHERE n < 5
            )
            SELECT n FROM cnt
        """)
        values = [r[0] for r in r.rows]
        assert values == [1, 2, 3, 4, 5]

    def test_fibonacci(self, db):
        r = db.execute("""
            WITH RECURSIVE fib(a, b) AS (
                SELECT 0, 1
                UNION ALL
                SELECT b, a + b FROM fib WHERE b < 50
            )
            SELECT a FROM fib
        """)
        values = [r[0] for r in r.rows]
        assert values[:7] == [0, 1, 1, 2, 3, 5, 8]

    def test_powers_of_two(self, db):
        r = db.execute("""
            WITH RECURSIVE pow2(n) AS (
                SELECT 1
                UNION ALL
                SELECT n * 2 FROM pow2 WHERE n < 64
            )
            SELECT n FROM pow2
        """)
        values = [r[0] for r in r.rows]
        assert values == [1, 2, 4, 8, 16, 32, 64]

    def test_org_hierarchy(self, db):
        """Walk the org tree from Alice (id=1) down."""
        r = db.execute("""
            WITH RECURSIVE org(id, name, lvl) AS (
                SELECT id, name, 0 FROM employees WHERE id = 1
                UNION ALL
                SELECT e.id, e.name, org.lvl + 1
                FROM employees AS e
                JOIN org ON e.manager_id = org.id
            )
            SELECT name, lvl FROM org ORDER BY lvl, name
        """)
        names = [r[0] for r in r.rows]
        assert names[0] == 'Alice'  # Level 0
        levels = [r[1] for r in r.rows]
        assert 0 in levels
        assert 1 in levels

    def test_recursive_with_depth_limit(self, db):
        """Ensure recursive CTE with limit terminates."""
        r = db.execute("""
            WITH RECURSIVE seq(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM seq WHERE n < 10
            )
            SELECT COUNT(*) AS cnt FROM seq
        """)
        assert r.rows[0][0] == 10

    def test_recursive_union_dedup(self, db):
        """UNION (not ALL) deduplicates."""
        r = db.execute("""
            WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION
                SELECT CASE WHEN n < 3 THEN n + 1 ELSE 1 END FROM nums WHERE n < 5
            )
            SELECT n FROM nums ORDER BY n
        """)
        values = [r[0] for r in r.rows]
        # UNION deduplicates: once we generate 1 again, it stops
        assert len(values) == len(set(values))

    def test_recursive_max_depth_exceeded(self, db):
        """Infinite recursion hits the safety limit."""
        with pytest.raises(CompileError, match="exceeded maximum depth"):
            db.execute("""
                WITH RECURSIVE inf(n) AS (
                    SELECT 1
                    UNION ALL
                    SELECT n + 1 FROM inf
                )
                SELECT n FROM inf
            """)

    def test_recursive_empty_base(self, db):
        r = db.execute("""
            WITH RECURSIVE empty(n) AS (
                SELECT id FROM employees WHERE id = 999
                UNION ALL
                SELECT n + 1 FROM empty WHERE n < 5
            )
            SELECT n FROM empty
        """)
        assert r.rows == []

    def test_recursive_depth_tracking(self, db):
        r = db.execute("""
            WITH RECURSIVE depths(n, depth) AS (
                SELECT 1, 0
                UNION ALL
                SELECT n + 1, depth + 1 FROM depths WHERE depth < 3
            )
            SELECT n, depth FROM depths ORDER BY depth
        """)
        assert r.rows[0] == [1, 0]
        assert r.rows[3] == [4, 3]
        assert len(r.rows) == 4


# =============================================================================
# CTE with Subqueries
# =============================================================================

class TestCTEWithSubqueries:
    """Tests for CTEs combined with subquery features."""

    def test_cte_join_for_scalar_filter(self, db):
        """Use CTE via JOIN instead of scalar subquery for filtering."""
        r = db.execute("""
            WITH avg_sal AS (
                SELECT AVG(salary) AS avg_s FROM employees
            )
            SELECT e.name FROM employees AS e
            JOIN avg_sal ON e.salary > avg_sal.avg_s
            ORDER BY e.name
        """)
        names = [r[0] for r in r.rows]
        assert 'Alice' in names
        assert 'Eve' in names

    def test_cte_join_for_existence(self, db):
        """Use CTE via JOIN to check existence."""
        r = db.execute("""
            WITH eng AS (
                SELECT id FROM employees WHERE dept = 'Engineering'
            )
            SELECT e.name FROM employees AS e
            JOIN eng ON e.id = eng.id
            ORDER BY e.name
        """)
        names = [r[0] for r in r.rows]
        assert sorted(names) == ['Alice', 'Bob', 'Eve']

    def test_cte_with_regular_subquery(self, db):
        """Subquery against real table still works inside CTE main query."""
        r = db.execute("""
            WITH eng AS (
                SELECT name, salary FROM employees WHERE dept = 'Engineering'
            )
            SELECT name FROM eng
            WHERE salary > (SELECT AVG(salary) FROM employees)
            ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert 'Alice' in names


# =============================================================================
# CTE with Joins
# =============================================================================

class TestCTEWithJoins:
    """Tests for CTEs used in JOINs."""

    def test_cte_inner_join(self, db):
        r = db.execute("""
            WITH dept_stats AS (
                SELECT dept, AVG(salary) AS avg_sal
                FROM employees
                GROUP BY dept
            )
            SELECT e.name, e.salary, d.avg_sal
            FROM employees AS e
            JOIN dept_stats AS d ON e.dept = d.dept
            ORDER BY e.name
        """)
        assert len(r.rows) == 6

    def test_cte_left_join(self, db):
        r = db.execute("""
            WITH managers AS (
                SELECT DISTINCT manager_id AS mid FROM employees WHERE manager_id IS NOT NULL
            )
            SELECT e.name, m.mid
            FROM employees AS e
            LEFT JOIN managers AS m ON e.id = m.mid
            ORDER BY e.name
        """)
        assert len(r.rows) == 6

    def test_two_cte_join(self, db):
        r = db.execute("""
            WITH
                eng AS (SELECT name, salary FROM employees WHERE dept = 'Engineering'),
                sales AS (SELECT name, salary FROM employees WHERE dept = 'Sales')
            SELECT eng.name AS eng_name, sales.name AS sales_name
            FROM eng
            CROSS JOIN sales
            ORDER BY eng_name, sales_name
        """)
        assert len(r.rows) == 6  # 3 eng * 2 sales


# =============================================================================
# CTE with Products (second fixture)
# =============================================================================

class TestCTEWithProducts:
    """Tests using the products fixture."""

    def test_category_stats(self, db_with_products):
        r = db_with_products.execute("""
            WITH cat_stats AS (
                SELECT category, COUNT(*) AS cnt, AVG(price) AS avg_price
                FROM products
                GROUP BY category
            )
            SELECT category, cnt, avg_price FROM cat_stats ORDER BY category
        """)
        assert len(r.rows) == 2
        assert r.rows[0][0] == 'Electronics'
        assert r.rows[0][1] == 3

    def test_above_average_in_category(self, db_with_products):
        r = db_with_products.execute("""
            WITH cat_avg AS (
                SELECT category, AVG(price) AS avg_price
                FROM products
                GROUP BY category
            )
            SELECT p.name, p.price, c.avg_price
            FROM products AS p
            JOIN cat_avg AS c ON p.category = c.category
            WHERE p.price > c.avg_price
            ORDER BY p.name
        """)
        names = [r[0] for r in r.rows]
        assert 'Laptop' in names

    def test_price_ranking(self, db_with_products):
        r = db_with_products.execute("""
            WITH ranked AS (
                SELECT name, price, category FROM products
            )
            SELECT name, price FROM ranked ORDER BY price DESC LIMIT 3
        """)
        assert r.rows[0][0] == 'Laptop'
        assert len(r.rows) == 3


# =============================================================================
# CTE Edge Cases
# =============================================================================

class TestCTEEdgeCases:
    """Tests for edge cases."""

    def test_cte_empty_result(self, db):
        r = db.execute("""
            WITH empty AS (
                SELECT name FROM employees WHERE salary > 1000000
            )
            SELECT COUNT(*) AS cnt FROM empty
        """)
        assert r.rows[0][0] == 0

    def test_cte_single_value(self):
        d = CTEDB()
        r = d.execute("WITH x AS (SELECT 42 AS val) SELECT val FROM x")
        assert r.rows == [[42]]

    def test_cte_multiple_rows(self):
        d = CTEDB()
        d.execute("CREATE TABLE nums (n INT)")
        d.execute("INSERT INTO nums VALUES (1)")
        d.execute("INSERT INTO nums VALUES (2)")
        d.execute("INSERT INTO nums VALUES (3)")
        r = d.execute("""
            WITH doubled AS (SELECT n, n * 2 AS d FROM nums)
            SELECT n, d FROM doubled ORDER BY n
        """)
        assert r.rows == [[1, 2], [2, 4], [3, 6]]

    def test_cte_reuses_name_from_table(self, db):
        """CTE can shadow a real table name."""
        r = db.execute("""
            WITH employees AS (
                SELECT 'Virtual' AS name, 0 AS salary
            )
            SELECT name FROM employees
        """)
        assert r.rows == [['Virtual']]

    def test_cte_with_null_values(self, db):
        r = db.execute("""
            WITH mgrs AS (
                SELECT name, manager_id FROM employees
            )
            SELECT name FROM mgrs WHERE manager_id IS NULL
        """)
        assert r.rows == [['Alice']]

    def test_cte_preserves_types(self):
        d = CTEDB()
        r = d.execute("""
            WITH data AS (SELECT 1 AS i, 2.5 AS f, 'hello' AS s)
            SELECT i, f, s FROM data
        """)
        assert r.rows[0] == [1, 2.5, 'hello']

    def test_non_cte_still_works(self, db):
        """Non-CTE queries should still work through CTEDB."""
        r = db.execute("SELECT name FROM employees WHERE dept = 'Sales' ORDER BY name")
        names = [r[0] for r in r.rows]
        assert names == ['Carol', 'Dave']

    def test_insert_still_works(self, db):
        db.execute("INSERT INTO employees VALUES (7, 'Grace', 'Engineering', 88000, 1)")
        r = db.execute("SELECT name FROM employees WHERE id = 7")
        assert r.rows == [['Grace']]

    def test_update_still_works(self, db):
        db.execute("UPDATE employees SET salary = 100000 WHERE name = 'Alice'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert r.rows == [[100000]]

    def test_delete_still_works(self, db):
        db.execute("DELETE FROM employees WHERE name = 'Frank'")
        r = db.execute("SELECT COUNT(*) FROM employees")
        assert r.rows[0][0] == 5


# =============================================================================
# CTE with Aggregation
# =============================================================================

class TestCTEAggregation:
    """Tests for CTEs with various aggregation patterns."""

    def test_cte_count(self, db):
        r = db.execute("""
            WITH dept_counts AS (
                SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept
            )
            SELECT dept, cnt FROM dept_counts ORDER BY cnt DESC
        """)
        assert r.rows[0][0] == 'Engineering'
        assert r.rows[0][1] == 3

    def test_cte_sum(self, db):
        r = db.execute("""
            WITH dept_sums AS (
                SELECT dept, SUM(salary) AS total FROM employees GROUP BY dept
            )
            SELECT dept, total FROM dept_sums ORDER BY total DESC
        """)
        assert r.rows[0][0] == 'Engineering'

    def test_cte_having(self, db):
        """Filter CTE results instead of using HAVING (known C247 HAVING limitation)."""
        r = db.execute("""
            WITH dept_counts AS (
                SELECT dept, COUNT(*) AS cnt
                FROM employees
                GROUP BY dept
            )
            SELECT dept FROM dept_counts WHERE cnt > 1 ORDER BY dept
        """)
        depts = [r[0] for r in r.rows]
        assert 'Engineering' in depts
        assert 'Sales' in depts
        assert 'Marketing' not in depts

    def test_aggregation_over_cte(self, db):
        """Aggregate over CTE results."""
        r = db.execute("""
            WITH eng AS (
                SELECT salary FROM employees WHERE dept = 'Engineering'
            )
            SELECT AVG(salary) AS avg_sal FROM eng
        """)
        # (90000 + 80000 + 95000) / 3 = 88333.33...
        assert abs(r.rows[0][0] - 88333.33) < 1

    def test_group_by_over_cte(self, db):
        r = db.execute("""
            WITH all_emp AS (
                SELECT name, dept, salary FROM employees
            )
            SELECT dept, MAX(salary) AS max_sal
            FROM all_emp
            GROUP BY dept
            ORDER BY max_sal DESC
        """)
        assert r.rows[0][0] == 'Engineering'
        assert r.rows[0][1] == 95000


# =============================================================================
# UNION in CTEs
# =============================================================================

class TestCTEUnion:
    """Tests for UNION within CTEs."""

    def test_union_all(self, db):
        r = db.execute("""
            WITH combined AS (
                SELECT name FROM employees WHERE dept = 'Engineering'
                UNION ALL
                SELECT name FROM employees WHERE dept = 'Sales'
            )
            SELECT name FROM combined ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert len(names) == 5

    def test_union_distinct(self, db):
        r = db.execute("""
            WITH depts AS (
                SELECT dept FROM employees WHERE salary > 60000
                UNION
                SELECT dept FROM employees WHERE salary < 70000
            )
            SELECT dept FROM depts ORDER BY dept
        """)
        depts = [r[0] for r in r.rows]
        assert len(depts) == len(set(depts))

    def test_union_different_sources(self):
        d = CTEDB()
        d.execute("CREATE TABLE t1 (x INT)")
        d.execute("CREATE TABLE t2 (y INT)")
        d.execute("INSERT INTO t1 VALUES (1)")
        d.execute("INSERT INTO t1 VALUES (2)")
        d.execute("INSERT INTO t2 VALUES (3)")
        d.execute("INSERT INTO t2 VALUES (4)")
        r = d.execute("""
            WITH combined AS (
                SELECT x AS val FROM t1
                UNION ALL
                SELECT y AS val FROM t2
            )
            SELECT val FROM combined ORDER BY val
        """)
        values = [r[0] for r in r.rows]
        assert values == [1, 2, 3, 4]


# =============================================================================
# Recursive CTE Advanced
# =============================================================================

class TestRecursiveCTEAdvanced:
    """Advanced recursive CTE tests."""

    def test_factorial(self, db):
        r = db.execute("""
            WITH RECURSIVE fact(n, f) AS (
                SELECT 1, 1
                UNION ALL
                SELECT n + 1, f * (n + 1) FROM fact WHERE n < 7
            )
            SELECT n, f FROM fact ORDER BY n
        """)
        assert r.rows[0] == [1, 1]
        assert r.rows[4] == [5, 120]
        assert r.rows[6] == [7, 5040]

    def test_triangular_numbers(self, db):
        r = db.execute("""
            WITH RECURSIVE tri(n, t) AS (
                SELECT 1, 1
                UNION ALL
                SELECT n + 1, t + n + 1 FROM tri WHERE n < 5
            )
            SELECT n, t FROM tri ORDER BY n
        """)
        # T(1)=1, T(2)=3, T(3)=6, T(4)=10, T(5)=15
        vals = [r[1] for r in r.rows]
        assert vals == [1, 3, 6, 10, 15]

    def test_recursive_cte_with_filter(self, db):
        """Recursive CTE with filter on final result."""
        r = db.execute("""
            WITH RECURSIVE seq(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM seq WHERE n < 20
            )
            SELECT n FROM seq WHERE n > 15 ORDER BY n
        """)
        values = [r[0] for r in r.rows]
        assert values == [16, 17, 18, 19, 20]

    def test_recursive_graph_paths(self, db):
        """Traverse manager hierarchy from Bob."""
        r = db.execute("""
            WITH RECURSIVE chain(id, name, depth) AS (
                SELECT id, name, 0 FROM employees WHERE name = 'Bob'
                UNION ALL
                SELECT e.id, e.name, chain.depth + 1
                FROM employees AS e
                JOIN chain ON e.id = (SELECT manager_id FROM employees WHERE id = chain.id)
                WHERE chain.depth < 10
            )
            SELECT name, depth FROM chain ORDER BY depth
        """)
        names = [r[0] for r in r.rows]
        assert names[0] == 'Bob'
        assert 'Alice' in names

    def test_recursive_sum(self, db):
        r = db.execute("""
            WITH RECURSIVE cumsum(n, total) AS (
                SELECT 1, 1
                UNION ALL
                SELECT n + 1, total + n + 1 FROM cumsum WHERE n < 10
            )
            SELECT total FROM cumsum WHERE n = 10
        """)
        assert r.rows[0][0] == 55  # sum(1..10)

    def test_recursive_with_column_list(self, db):
        r = db.execute("""
            WITH RECURSIVE counter(val) AS (
                SELECT 1
                UNION ALL
                SELECT val + 1 FROM counter WHERE val < 3
            )
            SELECT val FROM counter ORDER BY val
        """)
        assert [r[0] for r in r.rows] == [1, 2, 3]


# =============================================================================
# CTE with Multiple Features
# =============================================================================

class TestCTEComplex:
    """Tests combining CTEs with multiple SQL features."""

    def test_cte_with_distinct(self, db):
        r = db.execute("""
            WITH all_depts AS (
                SELECT DISTINCT dept FROM employees
            )
            SELECT dept FROM all_depts ORDER BY dept
        """)
        depts = [r[0] for r in r.rows]
        assert depts == ['Engineering', 'Marketing', 'Sales']

    def test_cte_with_offset(self, db):
        r = db.execute("""
            WITH ranked AS (
                SELECT name, salary FROM employees ORDER BY salary DESC
            )
            SELECT name FROM ranked LIMIT 2 OFFSET 1
        """)
        assert len(r.rows) == 2

    def test_cte_with_case(self, db):
        r = db.execute("""
            WITH categorized AS (
                SELECT name,
                    CASE
                        WHEN salary > 85000 THEN 'high'
                        WHEN salary > 65000 THEN 'mid'
                        ELSE 'low'
                    END AS tier
                FROM employees
            )
            SELECT tier, COUNT(*) AS cnt
            FROM categorized
            GROUP BY tier
            ORDER BY tier
        """)
        tiers = {r[0]: r[1] for r in r.rows}
        assert 'high' in tiers
        assert 'mid' in tiers

    def test_cte_with_min_max(self, db):
        r = db.execute("""
            WITH salary_range AS (
                SELECT dept, MIN(salary) AS min_sal, MAX(salary) AS max_sal
                FROM employees
                GROUP BY dept
            )
            SELECT dept, max_sal - min_sal AS range
            FROM salary_range
            ORDER BY range DESC
        """)
        assert len(r.rows) == 3

    def test_nested_cte_aggregation(self, db):
        """CTE over CTE with aggregation."""
        r = db.execute("""
            WITH
                dept_stats AS (
                    SELECT dept, AVG(salary) AS avg_sal
                    FROM employees
                    GROUP BY dept
                ),
                overall AS (
                    SELECT AVG(avg_sal) AS grand_avg FROM dept_stats
                )
            SELECT grand_avg FROM overall
        """)
        # avg of (88333, 65000, 65000) ~ 72777
        assert r.rows[0][0] is not None

    def test_cte_join_two_ctes(self, db):
        r = db.execute("""
            WITH
                dept_count AS (
                    SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept
                ),
                dept_sal AS (
                    SELECT dept, SUM(salary) AS total FROM employees GROUP BY dept
                )
            SELECT c.dept, c.cnt, s.total
            FROM dept_count AS c
            JOIN dept_sal AS s ON c.dept = s.dept
            ORDER BY c.dept
        """)
        assert len(r.rows) == 3
        # Engineering: 3 people, 265000 total
        eng = [r for r in r.rows if r[0] == 'Engineering'][0]
        assert eng[1] == 3
        assert eng[2] == 265000


# =============================================================================
# Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Ensure CTEDB works with all previous SQL features."""

    def test_subquery_in_where(self, db):
        r = db.execute("""
            SELECT name FROM employees
            WHERE salary > (SELECT AVG(salary) FROM employees)
            ORDER BY name
        """)
        assert len(r.rows) > 0

    def test_create_table(self):
        d = CTEDB()
        d.execute("CREATE TABLE test (id INT, val TEXT)")
        d.execute("INSERT INTO test VALUES (1, 'hello')")
        r = d.execute("SELECT val FROM test")
        assert r.rows == [['hello']]

    def test_update(self, db):
        db.execute("UPDATE employees SET salary = 100000 WHERE name = 'Eve'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Eve'")
        assert r.rows == [[100000]]

    def test_delete(self, db):
        db.execute("DELETE FROM employees WHERE dept = 'Marketing'")
        r = db.execute("SELECT COUNT(*) FROM employees")
        assert r.rows[0][0] == 5

    def test_group_by(self, db):
        """Test GROUP BY without HAVING (HAVING with raw COUNT is a known C247 limitation)."""
        r = db.execute("""
            SELECT dept, COUNT(*) AS cnt
            FROM employees
            GROUP BY dept
            ORDER BY dept
        """)
        depts = {r[0]: r[1] for r in r.rows}
        assert depts['Engineering'] == 3
        assert depts['Marketing'] == 1

    def test_order_by_limit(self, db):
        r = db.execute("SELECT name FROM employees ORDER BY salary DESC LIMIT 3")
        assert len(r.rows) == 3
        assert r.rows[0][0] == 'Eve'

    def test_join(self, db):
        d = CTEDB()
        d.execute("CREATE TABLE a (id INT, val TEXT)")
        d.execute("CREATE TABLE b (id INT, ref INT, label TEXT)")
        d.execute("INSERT INTO a VALUES (1, 'x')")
        d.execute("INSERT INTO b VALUES (10, 1, 'B1')")
        r = d.execute("SELECT a.val, b.label FROM a JOIN b ON a.id = b.ref")
        assert r.rows == [['x', 'B1']]

    def test_derived_table(self, db):
        r = db.execute("""
            SELECT d.name FROM (
                SELECT name, salary FROM employees WHERE salary > 80000
            ) AS d
            ORDER BY d.name
        """)
        names = [r[0] for r in r.rows]
        assert 'Alice' in names


# =============================================================================
# Misc / Robustness
# =============================================================================

class TestCTERobustness:
    """Robustness and stress tests."""

    def test_many_ctes(self):
        d = CTEDB()
        d.execute("CREATE TABLE base (n INT)")
        d.execute("INSERT INTO base VALUES (1)")
        sql = "WITH c0 AS (SELECT n FROM base)"
        for i in range(1, 10):
            sql += f", c{i} AS (SELECT n FROM c{i-1})"
        sql += f" SELECT n FROM c9"
        r = d.execute(sql)
        assert r.rows == [[1]]

    def test_recursive_cte_moderate_depth(self, db):
        r = db.execute("""
            WITH RECURSIVE seq(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM seq WHERE n < 100
            )
            SELECT COUNT(*) AS cnt FROM seq
        """)
        assert r.rows[0][0] == 100

    def test_execute_many_with_cte(self, db):
        results = db.execute_many("""
            WITH eng AS (SELECT name FROM employees WHERE dept = 'Engineering')
            SELECT COUNT(*) FROM eng;
            SELECT COUNT(*) FROM employees
        """)
        assert len(results) == 2
        assert results[0].rows[0][0] == 3
        assert results[1].rows[0][0] == 6

    def test_cte_with_like(self, db):
        r = db.execute("""
            WITH filtered AS (
                SELECT name FROM employees WHERE name LIKE 'A%'
            )
            SELECT name FROM filtered
        """)
        assert r.rows == [['Alice']]

    def test_cte_with_between(self, db):
        r = db.execute("""
            WITH mid_salary AS (
                SELECT name, salary FROM employees
                WHERE salary BETWEEN 65000 AND 85000
            )
            SELECT name FROM mid_salary ORDER BY name
        """)
        names = [r[0] for r in r.rows]
        assert 'Bob' in names
        assert 'Carol' in names

    def test_cte_with_is_null(self, db):
        r = db.execute("""
            WITH top_level AS (
                SELECT name FROM employees WHERE manager_id IS NULL
            )
            SELECT name FROM top_level
        """)
        assert r.rows == [['Alice']]

    def test_cte_with_not_null(self, db):
        r = db.execute("""
            WITH managed AS (
                SELECT name FROM employees WHERE manager_id IS NOT NULL
            )
            SELECT COUNT(*) AS cnt FROM managed
        """)
        assert r.rows[0][0] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
