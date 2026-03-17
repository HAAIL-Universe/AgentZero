"""
Tests for C253: Common Table Expressions (CTEs)

Tests:
1. Basic non-recursive CTEs
2. Multiple CTEs in one WITH clause
3. CTE with column aliases
4. Recursive CTEs (fibonacci, hierarchies, graph traversal)
5. CTE with WHERE, ORDER BY, LIMIT
6. CTE with aggregation and GROUP BY
7. CTE with JOINs
8. CTE referencing earlier CTEs
9. CTE with DISTINCT
10. Recursive CTE depth limiting
11. CTE with expressions and functions
12. Integration with existing DB features (window functions, views, triggers)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from common_table_expressions import (
    CTEDB, CTESelectStmt, CTEDef, UnionStmt,
    CTEParser, CTELexer,
    ResultSet, DatabaseError, ParseError,
    MAX_RECURSIVE_DEPTH,
)
from transaction_manager import IsolationLevel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def db():
    d = CTEDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, manager_id INT, salary INT, dept TEXT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', NULL, 100, 'Engineering')")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 1, 80, 'Engineering')")
    d.execute("INSERT INTO employees VALUES (3, 'Charlie', 1, 90, 'Sales')")
    d.execute("INSERT INTO employees VALUES (4, 'Diana', 2, 70, 'Engineering')")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 2, 75, 'Sales')")
    d.execute("INSERT INTO employees VALUES (6, 'Frank', 3, 65, 'Sales')")
    return d


@pytest.fixture
def tree_db():
    """Hierarchical tree data for recursive CTE tests."""
    d = CTEDB()
    d.execute("CREATE TABLE categories (id INT, name TEXT, parent_id INT)")
    d.execute("INSERT INTO categories VALUES (1, 'Root', NULL)")
    d.execute("INSERT INTO categories VALUES (2, 'Electronics', 1)")
    d.execute("INSERT INTO categories VALUES (3, 'Clothing', 1)")
    d.execute("INSERT INTO categories VALUES (4, 'Phones', 2)")
    d.execute("INSERT INTO categories VALUES (5, 'Laptops', 2)")
    d.execute("INSERT INTO categories VALUES (6, 'Shirts', 3)")
    d.execute("INSERT INTO categories VALUES (7, 'iPhones', 4)")
    d.execute("INSERT INTO categories VALUES (8, 'Android', 4)")
    return d


# =============================================================================
# 1. Basic Non-Recursive CTEs
# =============================================================================

class TestBasicCTE:
    def test_simple_cte(self, db):
        r = db.execute("""
            WITH eng AS (
                SELECT id, name, salary FROM employees WHERE dept = 'Engineering'
            )
            SELECT * FROM eng
        """)
        assert len(r.rows) == 3
        names = {row[1] for row in r.rows}
        assert names == {'Alice', 'Bob', 'Diana'}

    def test_cte_with_where(self, db):
        r = db.execute("""
            WITH high_salary AS (
                SELECT name, salary FROM employees WHERE salary > 75
            )
            SELECT name FROM high_salary WHERE salary >= 90
        """)
        names = {row[0] for row in r.rows}
        assert names == {'Alice', 'Charlie'}

    def test_cte_with_order_by(self, db):
        r = db.execute("""
            WITH ranked AS (
                SELECT name, salary FROM employees
            )
            SELECT name, salary FROM ranked ORDER BY salary DESC
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 100

    def test_cte_with_limit(self, db):
        r = db.execute("""
            WITH all_emps AS (
                SELECT name, salary FROM employees
            )
            SELECT name FROM all_emps ORDER BY salary DESC LIMIT 3
        """)
        assert len(r.rows) == 3

    def test_cte_with_distinct(self, db):
        r = db.execute("""
            WITH depts AS (
                SELECT dept FROM employees
            )
            SELECT DISTINCT dept FROM depts ORDER BY dept
        """)
        assert len(r.rows) == 2
        assert r.rows[0][0] == 'Engineering'
        assert r.rows[1][0] == 'Sales'

    def test_cte_select_star(self, db):
        r = db.execute("""
            WITH all_data AS (
                SELECT * FROM employees
            )
            SELECT * FROM all_data WHERE id = 1
        """)
        assert len(r.rows) == 1
        assert r.rows[0][1] == 'Alice'


# =============================================================================
# 2. Multiple CTEs
# =============================================================================

class TestMultipleCTEs:
    def test_two_ctes(self, db):
        r = db.execute("""
            WITH
                eng AS (SELECT name, salary FROM employees WHERE dept = 'Engineering'),
                sales AS (SELECT name, salary FROM employees WHERE dept = 'Sales')
            SELECT name, salary FROM eng ORDER BY salary DESC
        """)
        names = {row[0] for row in r.rows}
        assert names == {'Alice', 'Bob', 'Diana'}

    def test_cte_referencing_earlier_cte(self, db):
        r = db.execute("""
            WITH
                high_sal AS (SELECT id, name, salary FROM employees WHERE salary > 70),
                very_high AS (SELECT name, salary FROM high_sal WHERE salary > 85)
            SELECT name FROM very_high ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Charlie']

    def test_three_ctes(self, db):
        r = db.execute("""
            WITH
                all_emps AS (SELECT id, name, salary, dept FROM employees),
                eng AS (SELECT name, salary FROM all_emps WHERE dept = 'Engineering'),
                top_eng AS (SELECT name FROM eng WHERE salary >= 80)
            SELECT name FROM top_eng ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob']


# =============================================================================
# 3. CTE with Column Aliases
# =============================================================================

class TestCTEColumnAliases:
    def test_column_aliases(self, db):
        r = db.execute("""
            WITH emp_info(employee_name, employee_salary) AS (
                SELECT name, salary FROM employees WHERE dept = 'Engineering'
            )
            SELECT employee_name, employee_salary FROM emp_info
            ORDER BY employee_salary DESC
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 100
        assert 'employee_name' in r.columns
        assert 'employee_salary' in r.columns

    def test_column_alias_count_mismatch(self, db):
        with pytest.raises(DatabaseError, match="column aliases"):
            db.execute("""
                WITH bad(a, b, c) AS (
                    SELECT name, salary FROM employees
                )
                SELECT * FROM bad
            """)

    def test_column_aliases_with_star(self, db):
        r = db.execute("""
            WITH renamed(eid, ename, mgr, pay, department) AS (
                SELECT * FROM employees
            )
            SELECT ename, pay FROM renamed WHERE department = 'Sales'
            ORDER BY pay DESC
        """)
        assert r.rows[0][0] == 'Charlie'
        assert r.rows[0][1] == 90


# =============================================================================
# 4. Recursive CTEs
# =============================================================================

class TestRecursiveCTEs:
    def test_fibonacci(self):
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        d.execute("INSERT INTO dummy VALUES (1)")
        r = d.execute("""
            WITH RECURSIVE fib(n, a, b) AS (
                SELECT 1, 0, 1
                UNION ALL
                SELECT n + 1, b, a + b FROM fib WHERE n < 10
            )
            SELECT n, a FROM fib ORDER BY n
        """)
        # Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        assert len(r.rows) == 10
        fibs = [row[1] for row in r.rows]
        assert fibs == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_counting(self):
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        r = d.execute("""
            WITH RECURSIVE cnt(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM cnt WHERE n < 5
            )
            SELECT n FROM cnt ORDER BY n
        """)
        assert [row[0] for row in r.rows] == [1, 2, 3, 4, 5]

    def test_powers_of_two(self):
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        r = d.execute("""
            WITH RECURSIVE powers(n, val) AS (
                SELECT 0, 1
                UNION ALL
                SELECT n + 1, val * 2 FROM powers WHERE n < 8
            )
            SELECT n, val FROM powers ORDER BY n
        """)
        assert len(r.rows) == 9
        for row in r.rows:
            assert row[1] == 2 ** row[0]

    def test_hierarchy_traversal(self, tree_db):
        """Find all descendants of Electronics (id=2)."""
        r = tree_db.execute("""
            WITH RECURSIVE descendants(id, name, parent_id, depth) AS (
                SELECT id, name, parent_id, 0
                FROM categories WHERE id = 2
                UNION ALL
                SELECT c.id, c.name, c.parent_id, d.depth + 1
                FROM categories c
                JOIN descendants d ON c.parent_id = d.id
            )
            SELECT name, depth FROM descendants ORDER BY depth, name
        """)
        assert len(r.rows) == 5  # Electronics, Phones, Laptops, iPhones, Android
        assert r.rows[0][0] == 'Electronics'
        assert r.rows[0][1] == 0

    def test_ancestors(self, tree_db):
        """Find all ancestors of iPhones (id=7)."""
        r = tree_db.execute("""
            WITH RECURSIVE ancestors(id, name, parent_id) AS (
                SELECT id, name, parent_id
                FROM categories WHERE id = 7
                UNION ALL
                SELECT c.id, c.name, c.parent_id
                FROM categories c
                JOIN ancestors a ON c.id = a.parent_id
            )
            SELECT name FROM ancestors ORDER BY id
        """)
        names = [row[0] for row in r.rows]
        assert 'iPhones' in names
        assert 'Phones' in names
        assert 'Electronics' in names
        assert 'Root' in names

    def test_recursive_depth_limit(self):
        """Recursive CTE that never terminates should hit depth limit."""
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        d.execute("INSERT INTO dummy VALUES (1)")
        with pytest.raises(DatabaseError, match="exceeded maximum depth"):
            d.execute("""
                WITH RECURSIVE infinite(n) AS (
                    SELECT 1
                    UNION ALL
                    SELECT n + 1 FROM infinite
                )
                SELECT n FROM infinite
            """)

    def test_recursive_union_distinct(self):
        """UNION (without ALL) deduplicates."""
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        r = d.execute("""
            WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION
                SELECT n + 1 FROM nums WHERE n < 5
            )
            SELECT n FROM nums ORDER BY n
        """)
        assert [row[0] for row in r.rows] == [1, 2, 3, 4, 5]


# =============================================================================
# 5. CTE with Aggregation
# =============================================================================

class TestCTEAggregation:
    def test_aggregate_in_cte(self, db):
        r = db.execute("""
            WITH dept_stats AS (
                SELECT dept, COUNT(*) AS cnt, SUM(salary) AS total
                FROM employees
                GROUP BY dept
            )
            SELECT dept, cnt, total FROM dept_stats ORDER BY dept
        """)
        assert len(r.rows) == 2
        # Engineering: 3 employees, 250 total
        assert r.rows[0][0] == 'Engineering'
        assert r.rows[0][1] == 3
        assert r.rows[0][2] == 250
        # Sales: 3 employees, 230 total
        assert r.rows[1][0] == 'Sales'
        assert r.rows[1][1] == 3
        assert r.rows[1][2] == 230

    def test_aggregate_on_cte_result(self, db):
        r = db.execute("""
            WITH eng AS (
                SELECT salary FROM employees WHERE dept = 'Engineering'
            )
            SELECT SUM(salary) AS total, AVG(salary) AS avg_sal FROM eng
        """)
        assert r.rows[0][0] == 250  # 100 + 80 + 70
        # AVG: 250/3 ~= 83.33
        assert abs(r.rows[0][1] - 83.33) < 0.01

    def test_count_on_cte(self, db):
        r = db.execute("""
            WITH sales AS (
                SELECT * FROM employees WHERE dept = 'Sales'
            )
            SELECT COUNT(*) AS cnt FROM sales
        """)
        assert r.rows[0][0] == 3

    def test_group_by_on_cte(self, db):
        r = db.execute("""
            WITH all_emps AS (
                SELECT * FROM employees
            )
            SELECT dept, MAX(salary) AS max_sal FROM all_emps GROUP BY dept ORDER BY dept
        """)
        assert r.rows[0][0] == 'Engineering'
        assert r.rows[0][1] == 100
        assert r.rows[1][0] == 'Sales'
        assert r.rows[1][1] == 90

    def test_having_on_cte(self, db):
        r = db.execute("""
            WITH all_emps AS (
                SELECT * FROM employees
            )
            SELECT dept, COUNT(*) AS cnt FROM all_emps
            GROUP BY dept
            HAVING COUNT(*) >= 3
            ORDER BY dept
        """)
        assert len(r.rows) == 2


# =============================================================================
# 6. CTE with JOINs
# =============================================================================

class TestCTEJoins:
    def test_cte_join_real_table(self, db):
        """CTE joined with a real table."""
        r = db.execute("""
            WITH managers AS (
                SELECT id, name FROM employees WHERE manager_id IS NULL
            )
            SELECT e.name, m.name AS manager_name
            FROM employees e
            JOIN managers m ON e.manager_id = m.id
            ORDER BY e.name
        """)
        assert len(r.rows) == 2  # Bob and Charlie report to Alice
        names = [row[0] for row in r.rows]
        assert 'Bob' in names
        assert 'Charlie' in names

    def test_two_ctes_joined(self, db):
        """Two CTEs joined together."""
        r = db.execute("""
            WITH
                eng AS (SELECT id, name, salary FROM employees WHERE dept = 'Engineering'),
                sales AS (SELECT id, name, salary FROM employees WHERE dept = 'Sales')
            SELECT eng.name AS eng_name, sales.name AS sales_name
            FROM eng
            JOIN sales ON eng.salary = sales.salary
        """)
        # No exact salary matches between eng and sales in our data
        # Engineering: 100, 80, 70. Sales: 90, 75, 65.
        assert len(r.rows) == 0

    def test_cte_self_reference_via_join(self, db):
        """CTE result joined with original table."""
        r = db.execute("""
            WITH top_earners AS (
                SELECT id, name FROM employees WHERE salary >= 80
            )
            SELECT e.name, e.salary
            FROM employees e
            JOIN top_earners t ON e.id = t.id
            ORDER BY e.salary DESC
        """)
        assert len(r.rows) == 3  # Alice (100), Charlie (90), Bob (80)
        assert r.rows[0][0] == 'Alice'


# =============================================================================
# 7. CTE with Expressions and Functions
# =============================================================================

class TestCTEExpressions:
    def test_arithmetic_in_cte(self, db):
        r = db.execute("""
            WITH bonuses AS (
                SELECT name, salary, salary * 2 AS double_salary
                FROM employees
            )
            SELECT name, double_salary FROM bonuses
            WHERE double_salary > 150
            ORDER BY double_salary DESC
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 200

    def test_case_expression_in_cte(self, db):
        r = db.execute("""
            WITH levels AS (
                SELECT name, salary,
                       CASE WHEN salary >= 90 THEN 'Senior'
                            WHEN salary >= 70 THEN 'Mid'
                            ELSE 'Junior' END AS level
                FROM employees
            )
            SELECT name, level FROM levels WHERE level = 'Senior' ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Charlie' in names

    def test_coalesce_in_cte(self, db):
        r = db.execute("""
            WITH managers AS (
                SELECT name, COALESCE(manager_id, 0) AS mgr
                FROM employees
            )
            SELECT name, mgr FROM managers WHERE mgr = 0
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_is_null_in_cte(self, db):
        r = db.execute("""
            WITH top_level AS (
                SELECT name FROM employees WHERE manager_id IS NULL
            )
            SELECT * FROM top_level
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_between_in_cte(self, db):
        r = db.execute("""
            WITH mid_range AS (
                SELECT name, salary FROM employees WHERE salary BETWEEN 70 AND 85
            )
            SELECT name FROM mid_range ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert 'Bob' in names
        assert 'Diana' in names
        assert 'Eve' in names

    def test_in_list_in_cte(self, db):
        r = db.execute("""
            WITH named AS (
                SELECT * FROM employees WHERE name IN ('Alice', 'Bob', 'Charlie')
            )
            SELECT name FROM named ORDER BY name
        """)
        assert [row[0] for row in r.rows] == ['Alice', 'Bob', 'Charlie']


# =============================================================================
# 8. Recursive CTE Practical Examples
# =============================================================================

class TestRecursivePractical:
    def test_org_chart_depth(self, db):
        """Build org chart with depth levels."""
        r = db.execute("""
            WITH RECURSIVE org(id, name, depth) AS (
                SELECT id, name, 0 FROM employees WHERE manager_id IS NULL
                UNION ALL
                SELECT e.id, e.name, o.depth + 1
                FROM employees e
                JOIN org o ON e.manager_id = o.id
            )
            SELECT name, depth FROM org ORDER BY depth, name
        """)
        # Alice(0), Bob(1), Charlie(1), Diana(2), Eve(2), Frank(2)
        assert r.rows[0] == ['Alice', 0]
        depth_1 = [row[0] for row in r.rows if row[1] == 1]
        assert sorted(depth_1) == ['Bob', 'Charlie']
        depth_2 = [row[0] for row in r.rows if row[1] == 2]
        assert sorted(depth_2) == ['Diana', 'Eve', 'Frank']

    def test_path_building(self, tree_db):
        """Build paths from root to leaves."""
        r = tree_db.execute("""
            WITH RECURSIVE paths(id, name, path) AS (
                SELECT id, name, name FROM categories WHERE parent_id IS NULL
                UNION ALL
                SELECT c.id, c.name, CONCAT(p.path, '/', c.name)
                FROM categories c
                JOIN paths p ON c.parent_id = p.id
            )
            SELECT name, path FROM paths ORDER BY path
        """)
        # Verify paths exist
        paths = {row[0]: row[1] for row in r.rows}
        assert paths.get('Root') == 'Root'
        assert paths.get('Electronics') == 'Root/Electronics'
        assert paths.get('Phones') == 'Root/Electronics/Phones'

    def test_running_total(self):
        """Compute running total using recursive CTE."""
        d = CTEDB()
        d.execute("CREATE TABLE sales (month INT, amount INT)")
        d.execute("INSERT INTO sales VALUES (1, 100)")
        d.execute("INSERT INTO sales VALUES (2, 150)")
        d.execute("INSERT INTO sales VALUES (3, 200)")
        d.execute("INSERT INTO sales VALUES (4, 120)")
        r = d.execute("""
            WITH RECURSIVE running(month, amount, total) AS (
                SELECT month, amount, amount FROM sales WHERE month = 1
                UNION ALL
                SELECT s.month, s.amount, r.total + s.amount
                FROM sales s
                JOIN running r ON s.month = r.month + 1
            )
            SELECT month, amount, total FROM running ORDER BY month
        """)
        assert len(r.rows) == 4
        assert r.rows[0] == [1, 100, 100]
        assert r.rows[1] == [2, 150, 250]
        assert r.rows[2] == [3, 200, 450]
        assert r.rows[3] == [4, 120, 570]

    def test_category_full_tree(self, tree_db):
        """Full tree traversal from root."""
        r = tree_db.execute("""
            WITH RECURSIVE tree(id, name, depth) AS (
                SELECT id, name, 0 FROM categories WHERE parent_id IS NULL
                UNION ALL
                SELECT c.id, c.name, t.depth + 1
                FROM categories c
                JOIN tree t ON c.parent_id = t.id
            )
            SELECT name, depth FROM tree ORDER BY depth, name
        """)
        assert len(r.rows) == 8  # all 8 categories
        assert r.rows[0] == ['Root', 0]
        depth_1 = [row[0] for row in r.rows if row[1] == 1]
        assert sorted(depth_1) == ['Clothing', 'Electronics']


# =============================================================================
# 9. Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_empty_cte(self):
        d = CTEDB()
        d.execute("CREATE TABLE t (x INT)")
        r = d.execute("""
            WITH empty AS (
                SELECT x FROM t
            )
            SELECT * FROM empty
        """)
        assert len(r.rows) == 0

    def test_single_row_cte(self):
        d = CTEDB()
        d.execute("CREATE TABLE t (x INT)")
        d.execute("INSERT INTO t VALUES (42)")
        r = d.execute("""
            WITH one AS (
                SELECT x FROM t
            )
            SELECT x FROM one
        """)
        assert r.rows == [[42]]

    def test_cte_name_case_insensitive(self, db):
        """CTE names should be case-insensitive."""
        r = db.execute("""
            WITH MyData AS (
                SELECT name FROM employees WHERE id = 1
            )
            SELECT * FROM mydata
        """)
        assert r.rows[0][0] == 'Alice'

    def test_cte_shadows_real_table(self):
        """CTE with same name as real table should shadow it."""
        d = CTEDB()
        d.execute("CREATE TABLE t (x INT)")
        d.execute("INSERT INTO t VALUES (1)")
        d.execute("INSERT INTO t VALUES (2)")
        d.execute("INSERT INTO t VALUES (3)")
        r = d.execute("""
            WITH t AS (
                SELECT 99 AS x
            )
            SELECT * FROM t
        """)
        # Should get CTE result, not the real table
        assert len(r.rows) == 1
        assert r.rows[0][0] == 99

    def test_recursive_single_iteration(self):
        """Recursive CTE that terminates after base case."""
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        r = d.execute("""
            WITH RECURSIVE single(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM single WHERE n < 1
            )
            SELECT n FROM single
        """)
        assert r.rows == [[1]]

    def test_null_handling_in_cte(self, db):
        r = db.execute("""
            WITH nullable AS (
                SELECT name, manager_id FROM employees
            )
            SELECT name FROM nullable WHERE manager_id IS NULL
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_cte_with_offset(self, db):
        r = db.execute("""
            WITH all_emps AS (
                SELECT name, salary FROM employees
            )
            SELECT name FROM all_emps ORDER BY salary DESC LIMIT 2 OFFSET 1
        """)
        assert len(r.rows) == 2

    def test_multiple_columns_recursive(self):
        """Recursive CTE with multiple computed columns."""
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        r = d.execute("""
            WITH RECURSIVE seq(n, sq, cube) AS (
                SELECT 1, 1, 1
                UNION ALL
                SELECT n + 1, (n + 1) * (n + 1), (n + 1) * (n + 1) * (n + 1)
                FROM seq WHERE n < 5
            )
            SELECT n, sq, cube FROM seq ORDER BY n
        """)
        assert len(r.rows) == 5
        for row in r.rows:
            assert row[1] == row[0] ** 2
            assert row[2] == row[0] ** 3


# =============================================================================
# 10. CTE with non-SELECT main statements
# =============================================================================

class TestCTEWithDML:
    def test_cte_in_context(self, db):
        """CTE followed by SELECT is the primary use case."""
        r = db.execute("""
            WITH eng AS (
                SELECT id FROM employees WHERE dept = 'Engineering'
            )
            SELECT COUNT(*) AS cnt FROM eng
        """)
        assert r.rows[0][0] == 3


# =============================================================================
# 11. Non-CTE statements still work
# =============================================================================

class TestNonCTEStatements:
    def test_regular_select(self, db):
        r = db.execute("SELECT name FROM employees WHERE id = 1")
        assert r.rows[0][0] == 'Alice'

    def test_regular_insert(self, db):
        db.execute("INSERT INTO employees VALUES (7, 'Grace', 3, 85, 'Sales')")
        r = db.execute("SELECT name FROM employees WHERE id = 7")
        assert r.rows[0][0] == 'Grace'

    def test_regular_update(self, db):
        db.execute("UPDATE employees SET salary = 110 WHERE id = 1")
        r = db.execute("SELECT salary FROM employees WHERE id = 1")
        assert r.rows[0][0] == 110

    def test_regular_delete(self, db):
        db.execute("DELETE FROM employees WHERE id = 6")
        r = db.execute("SELECT COUNT(*) AS cnt FROM employees")
        assert r.rows[0][0] == 5

    def test_create_and_drop(self):
        d = CTEDB()
        d.execute("CREATE TABLE test (a INT)")
        d.execute("DROP TABLE test")
        r = d.execute("SHOW TABLES")
        assert len(r.rows) == 0


# =============================================================================
# 12. Complex Recursive Patterns
# =============================================================================

class TestComplexRecursive:
    def test_factorial(self):
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        r = d.execute("""
            WITH RECURSIVE fact(n, val) AS (
                SELECT 0, 1
                UNION ALL
                SELECT n + 1, val * (n + 1) FROM fact WHERE n < 10
            )
            SELECT n, val FROM fact ORDER BY n
        """)
        import math
        for row in r.rows:
            assert row[1] == math.factorial(row[0])

    def test_geometric_series(self):
        d = CTEDB()
        d.execute("CREATE TABLE dummy (x INT)")
        r = d.execute("""
            WITH RECURSIVE geo(n, val, total) AS (
                SELECT 0, 1, 1
                UNION ALL
                SELECT n + 1, val * 2, total + val * 2 FROM geo WHERE n < 7
            )
            SELECT n, val, total FROM geo ORDER BY n
        """)
        assert len(r.rows) == 8
        # Check: 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
        assert r.rows[-1][2] == 255

    def test_subtree_sizes(self, tree_db):
        """Count nodes under each category."""
        r = tree_db.execute("""
            WITH RECURSIVE subtree(id, name, root_id) AS (
                SELECT id, name, id FROM categories
                UNION ALL
                SELECT c.id, c.name, s.root_id
                FROM categories c
                JOIN subtree s ON c.parent_id = s.id
                WHERE c.id != s.root_id
            )
            SELECT root_id, COUNT(*) AS size FROM subtree
            GROUP BY root_id
            ORDER BY size DESC
        """)
        # Root (1) has all 8 nodes under it
        assert r.rows[0][0] == 1
        assert r.rows[0][1] == 8


# =============================================================================
# 13. String operations in CTEs
# =============================================================================

class TestCTEStringOps:
    def test_concat_in_recursive(self, tree_db):
        """String concatenation in recursive CTE (path building)."""
        r = tree_db.execute("""
            WITH RECURSIVE paths(id, path) AS (
                SELECT id, name FROM categories WHERE parent_id IS NULL
                UNION ALL
                SELECT c.id, CONCAT(p.path, ' > ', c.name)
                FROM categories c
                JOIN paths p ON c.parent_id = p.id
            )
            SELECT path FROM paths WHERE id = 7
        """)
        assert 'Root' in r.rows[0][0]
        assert 'iPhones' in r.rows[0][0]

    def test_upper_in_cte(self, db):
        r = db.execute("""
            WITH upper_names AS (
                SELECT UPPER(name) AS uname FROM employees
            )
            SELECT uname FROM upper_names ORDER BY uname LIMIT 3
        """)
        assert all(row[0] == row[0].upper() for row in r.rows)


# =============================================================================
# 14. CTE with complex WHERE
# =============================================================================

class TestCTEComplexWhere:
    def test_and_or_conditions(self, db):
        r = db.execute("""
            WITH filtered AS (
                SELECT name, salary, dept FROM employees
            )
            SELECT name FROM filtered
            WHERE (dept = 'Engineering' AND salary > 75)
               OR (dept = 'Sales' AND salary > 80)
            ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Charlie' in names

    def test_not_in_cte_where(self, db):
        r = db.execute("""
            WITH non_managers AS (
                SELECT name FROM employees WHERE manager_id IS NOT NULL
            )
            SELECT name FROM non_managers ORDER BY name
        """)
        assert len(r.rows) == 5
        assert 'Alice' not in [row[0] for row in r.rows]


# =============================================================================
# 15. Parser edge cases
# =============================================================================

class TestParserEdgeCases:
    def test_with_keyword_not_cte(self, db):
        """Regular query (no WITH) should still work."""
        r = db.execute("SELECT name FROM employees WHERE id = 1")
        assert r.rows[0][0] == 'Alice'

    def test_cte_with_semicolon(self, db):
        """CTE followed by semicolon."""
        r = db.execute("""
            WITH eng AS (
                SELECT name FROM employees WHERE dept = 'Engineering'
            )
            SELECT COUNT(*) AS cnt FROM eng;
        """)
        assert r.rows[0][0] == 3

    def test_multiple_statements_mixed(self, db):
        """Multiple statements, some with CTEs, some without."""
        results = db.execute_many("""
            SELECT COUNT(*) AS cnt FROM employees;
            WITH eng AS (SELECT name FROM employees WHERE dept = 'Engineering')
            SELECT COUNT(*) AS cnt FROM eng;
        """)
        assert results[0].rows[0][0] == 6
        assert results[1].rows[0][0] == 3


# =============================================================================
# 16. CTE with MIN/MAX/AVG
# =============================================================================

class TestCTEMinMaxAvg:
    def test_min_salary(self, db):
        r = db.execute("""
            WITH salaries AS (
                SELECT salary FROM employees
            )
            SELECT MIN(salary) AS min_sal FROM salaries
        """)
        assert r.rows[0][0] == 65

    def test_max_salary(self, db):
        r = db.execute("""
            WITH salaries AS (
                SELECT salary FROM employees
            )
            SELECT MAX(salary) AS max_sal FROM salaries
        """)
        assert r.rows[0][0] == 100

    def test_avg_salary(self, db):
        r = db.execute("""
            WITH salaries AS (
                SELECT salary FROM employees
            )
            SELECT AVG(salary) AS avg_sal FROM salaries
        """)
        expected = (100 + 80 + 90 + 70 + 75 + 65) / 6
        assert abs(r.rows[0][0] - expected) < 0.01

    def test_distinct_count(self, db):
        r = db.execute("""
            WITH depts AS (
                SELECT dept FROM employees
            )
            SELECT COUNT(DISTINCT dept) AS cnt FROM depts
        """)
        assert r.rows[0][0] == 2


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
