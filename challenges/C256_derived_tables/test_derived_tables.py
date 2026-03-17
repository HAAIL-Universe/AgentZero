"""
Tests for C256: Derived Tables (Subqueries in FROM clause)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from derived_tables import DerivedTableDB, DerivedTable


@pytest.fixture
def db():
    d = DerivedTableDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary INT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 'eng', 90000)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 'eng', 80000)")
    d.execute("INSERT INTO employees VALUES (3, 'Carol', 'sales', 70000)")
    d.execute("INSERT INTO employees VALUES (4, 'Dave', 'sales', 60000)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 'hr', 75000)")
    return d


@pytest.fixture
def db_orders():
    d = DerivedTableDB()
    d.execute("CREATE TABLE customers (id INT, name TEXT, city TEXT)")
    d.execute("INSERT INTO customers VALUES (1, 'Alice', 'NYC')")
    d.execute("INSERT INTO customers VALUES (2, 'Bob', 'LA')")
    d.execute("INSERT INTO customers VALUES (3, 'Carol', 'NYC')")

    d.execute("CREATE TABLE orders (id INT, customer_id INT, amount INT, status TEXT)")
    d.execute("INSERT INTO orders VALUES (1, 1, 100, 'complete')")
    d.execute("INSERT INTO orders VALUES (2, 1, 200, 'complete')")
    d.execute("INSERT INTO orders VALUES (3, 2, 150, 'pending')")
    d.execute("INSERT INTO orders VALUES (4, 3, 300, 'complete')")
    d.execute("INSERT INTO orders VALUES (5, 3, 50, 'pending')")
    return d


# =====================================================================
# Basic Derived Tables in FROM
# =====================================================================

class TestBasicDerivedTables:
    def test_simple_derived_table(self, db):
        """FROM (SELECT ...) AS alias"""
        r = db.execute("SELECT d.name FROM (SELECT name FROM employees) AS d")
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']

    def test_derived_table_with_where(self, db):
        """Derived table with WHERE in inner query"""
        r = db.execute("""
            SELECT d.name, d.salary
            FROM (SELECT name, salary FROM employees WHERE dept = 'eng') AS d
        """)
        assert len(r.rows) == 2
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Bob']

    def test_derived_table_with_outer_where(self, db):
        """WHERE on columns from derived table"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name, salary FROM employees) AS d
            WHERE d.salary > 75000
        """)
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Bob']

    def test_derived_table_all_columns(self, db):
        """SELECT * from derived table"""
        r = db.execute("""
            SELECT * FROM (SELECT name, dept FROM employees WHERE dept = 'hr') AS d
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Eve'
        assert r.rows[0][1] == 'hr'

    def test_derived_table_column_subset(self, db):
        """Select specific columns from derived table"""
        r = db.execute("""
            SELECT d.dept
            FROM (SELECT name, dept, salary FROM employees) AS d
            WHERE d.salary >= 80000
        """)
        depts = sorted([row[0] for row in r.rows])
        assert depts == ['eng', 'eng']

    def test_derived_table_with_alias_no_as(self, db):
        """Derived table with alias without AS keyword"""
        r = db.execute("""
            SELECT d.name FROM (SELECT name FROM employees WHERE dept = 'hr') d
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Eve'


# =====================================================================
# Derived Tables with Aggregation
# =====================================================================

class TestDerivedTablesAggregation:
    def test_aggregate_in_derived_table(self, db):
        """Derived table that computes aggregates"""
        r = db.execute("""
            SELECT d.dept, d.avg_sal
            FROM (SELECT dept, AVG(salary) AS avg_sal FROM employees GROUP BY dept) AS d
            ORDER BY d.dept
        """)
        assert len(r.rows) == 3
        assert r.rows[0][0] == 'eng'
        assert r.rows[0][1] == 85000  # (90000+80000)/2
        assert r.rows[1][0] == 'hr'
        assert r.rows[1][1] == 75000
        assert r.rows[2][0] == 'sales'
        assert r.rows[2][1] == 65000  # (70000+60000)/2

    def test_aggregate_over_derived_table(self, db):
        """Aggregate computed over derived table rows"""
        r = db.execute("""
            SELECT COUNT(*) FROM (SELECT name FROM employees WHERE dept = 'eng') AS d
        """)
        assert r.rows[0][0] == 2

    def test_sum_over_derived_table(self, db):
        """SUM over derived table"""
        r = db.execute("""
            SELECT SUM(d.salary)
            FROM (SELECT salary FROM employees WHERE dept = 'sales') AS d
        """)
        assert r.rows[0][0] == 130000  # 70000+60000

    def test_group_by_on_derived_table(self, db):
        """GROUP BY on derived table output"""
        r = db.execute("""
            SELECT d.dept, COUNT(*)
            FROM (SELECT name, dept FROM employees) AS d
            GROUP BY d.dept
            ORDER BY d.dept
        """)
        assert len(r.rows) == 3
        assert r.rows[0][:2] == ['eng', 2]
        assert r.rows[1][:2] == ['hr', 1]
        assert r.rows[2][:2] == ['sales', 2]

    def test_having_on_derived_table(self, db):
        """HAVING on derived table with aggregation"""
        r = db.execute("""
            SELECT d.dept, COUNT(*) AS cnt
            FROM (SELECT name, dept FROM employees) AS d
            GROUP BY d.dept
            HAVING COUNT(*) > 1
            ORDER BY d.dept
        """)
        assert len(r.rows) == 2
        assert r.rows[0][0] == 'eng'
        assert r.rows[1][0] == 'sales'

    def test_max_min_on_derived(self, db):
        """MAX/MIN on derived table"""
        r = db.execute("""
            SELECT MAX(d.salary), MIN(d.salary)
            FROM (SELECT salary FROM employees) AS d
        """)
        assert r.rows[0][0] == 90000
        assert r.rows[0][1] == 60000


# =====================================================================
# Derived Tables in JOINs
# =====================================================================

class TestDerivedTablesJoin:
    def test_join_with_derived_table(self, db_orders):
        """JOIN regular table with derived table"""
        r = db_orders.execute("""
            SELECT c.name, d.total
            FROM customers AS c
            JOIN (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) AS d
                ON c.id = d.customer_id
            ORDER BY c.name
        """)
        assert len(r.rows) == 3
        assert r.rows[0][:2] == ['Alice', 300]
        assert r.rows[1][:2] == ['Bob', 150]
        assert r.rows[2][:2] == ['Carol', 350]

    def test_derived_table_in_from_join_regular(self, db_orders):
        """Derived table in FROM, JOIN with regular table"""
        r = db_orders.execute("""
            SELECT d.name, o.amount
            FROM (SELECT id, name FROM customers WHERE city = 'NYC') AS d
            JOIN orders AS o ON d.id = o.customer_id
            ORDER BY d.name, o.amount
        """)
        assert len(r.rows) == 4  # Alice(100,200) + Carol(300,50)
        assert r.rows[0][:2] == ['Alice', 100]
        assert r.rows[1][:2] == ['Alice', 200]
        assert r.rows[2][:2] == ['Carol', 50]
        assert r.rows[3][:2] == ['Carol', 300]

    def test_two_derived_tables_join(self, db_orders):
        """JOIN between two derived tables"""
        r = db_orders.execute("""
            SELECT a.name, b.total
            FROM (SELECT id, name FROM customers) AS a
            JOIN (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) AS b
                ON a.id = b.customer_id
            ORDER BY a.name
        """)
        assert len(r.rows) == 3
        assert r.rows[0][:2] == ['Alice', 300]
        assert r.rows[1][:2] == ['Bob', 150]
        assert r.rows[2][:2] == ['Carol', 350]

    def test_left_join_derived_table(self, db_orders):
        """LEFT JOIN with derived table"""
        # Add a customer with no orders
        db_orders.execute("INSERT INTO customers VALUES (4, 'Dan', 'SF')")
        r = db_orders.execute("""
            SELECT c.name, d.total
            FROM customers AS c
            LEFT JOIN (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) AS d
                ON c.id = d.customer_id
            ORDER BY c.name
        """)
        assert len(r.rows) == 4
        assert r.rows[0][:2] == ['Alice', 300]
        assert r.rows[1][:2] == ['Bob', 150]
        assert r.rows[2][:2] == ['Carol', 350]
        assert r.rows[3][0] == 'Dan'
        assert r.rows[3][1] is None  # No matching orders

    def test_cross_join_derived_tables(self, db):
        """CROSS JOIN between derived tables (implicit)"""
        r = db.execute("""
            SELECT a.dept, b.dept
            FROM (SELECT DISTINCT dept FROM employees WHERE dept = 'eng') AS a
            CROSS JOIN (SELECT DISTINCT dept FROM employees WHERE dept = 'sales') AS b
        """)
        assert len(r.rows) == 1
        assert r.rows[0][:2] == ['eng', 'sales']


# =====================================================================
# Nested Derived Tables
# =====================================================================

class TestNestedDerivedTables:
    def test_nested_derived_table(self, db):
        """Derived table inside a derived table"""
        r = db.execute("""
            SELECT outer_d.name
            FROM (
                SELECT inner_d.name, inner_d.salary
                FROM (SELECT name, salary FROM employees WHERE dept = 'eng') AS inner_d
                WHERE inner_d.salary > 85000
            ) AS outer_d
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_double_nested_derived(self, db):
        """Three levels of derived tables"""
        r = db.execute("""
            SELECT d3.name
            FROM (
                SELECT d2.name
                FROM (
                    SELECT d1.name, d1.salary
                    FROM (SELECT name, salary FROM employees) AS d1
                    WHERE d1.salary > 70000
                ) AS d2
                WHERE d2.salary < 90000
            ) AS d3
            ORDER BY d3.name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Eve']

    def test_nested_with_aggregation(self, db):
        """Nested derived table with aggregation at inner level"""
        r = db.execute("""
            SELECT d.dept
            FROM (
                SELECT dept, AVG(salary) AS avg_sal
                FROM employees
                GROUP BY dept
            ) AS d
            WHERE d.avg_sal > 70000
            ORDER BY d.dept
        """)
        depts = [row[0] for row in r.rows]
        assert depts == ['eng', 'hr']


# =====================================================================
# Derived Tables with Subqueries
# =====================================================================

class TestDerivedTablesWithSubqueries:
    def test_subquery_in_derived_table_where(self, db):
        """Derived table with subquery in its WHERE"""
        r = db.execute("""
            SELECT d.name
            FROM (
                SELECT name, salary
                FROM employees
                WHERE salary > (SELECT AVG(salary) FROM employees)
            ) AS d
            ORDER BY d.name
        """)
        # AVG salary = 75000, > 75000: Alice(90k), Bob(80k)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob']

    def test_subquery_in_outer_where(self, db):
        """Subquery in WHERE of outer query referencing derived table"""
        r = db.execute("""
            SELECT d.name, d.salary
            FROM (SELECT name, salary, dept FROM employees) AS d
            WHERE d.salary > (SELECT AVG(salary) FROM employees WHERE dept = d.dept)
            ORDER BY d.name
        """)
        # eng avg=85000: Alice(90k) passes; sales avg=65000: Carol(70k) passes
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Carol' in names

    def test_exists_with_derived_table(self, db_orders):
        """EXISTS subquery with derived table"""
        r = db_orders.execute("""
            SELECT c.name
            FROM customers AS c
            WHERE EXISTS (
                SELECT 1 FROM (SELECT customer_id FROM orders WHERE status = 'complete') AS d
                WHERE d.customer_id = c.id
            )
            ORDER BY c.name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Carol']

    def test_in_subquery_over_derived_table(self, db_orders):
        """IN subquery using derived table"""
        r = db_orders.execute("""
            SELECT name
            FROM customers
            WHERE id IN (
                SELECT d.customer_id
                FROM (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) AS d
                WHERE d.total > 200
            )
            ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Carol']


# =====================================================================
# Derived Tables with ORDER BY / LIMIT / OFFSET
# =====================================================================

class TestDerivedTablesOrderLimit:
    def test_order_by_on_derived(self, db):
        """ORDER BY on derived table columns"""
        r = db.execute("""
            SELECT d.name, d.salary
            FROM (SELECT name, salary FROM employees) AS d
            ORDER BY d.salary DESC
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[-1][0] == 'Dave'

    def test_limit_on_derived(self, db):
        """LIMIT on derived table result"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name, salary FROM employees ORDER BY salary DESC) AS d
            LIMIT 2
        """)
        assert len(r.rows) == 2

    def test_offset_on_derived(self, db):
        """OFFSET on derived table result"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name FROM employees ORDER BY name) AS d
            LIMIT 2 OFFSET 1
        """)
        assert len(r.rows) == 2

    def test_limit_inside_derived(self, db):
        """LIMIT inside the derived table query"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name FROM employees ORDER BY salary DESC LIMIT 3) AS d
            ORDER BY d.name
        """)
        assert len(r.rows) == 3

    def test_distinct_on_derived(self, db):
        """DISTINCT on derived table output"""
        r = db.execute("""
            SELECT DISTINCT d.dept
            FROM (SELECT dept FROM employees) AS d
            ORDER BY d.dept
        """)
        assert len(r.rows) == 3
        assert r.rows[0][0] == 'eng'
        assert r.rows[1][0] == 'hr'
        assert r.rows[2][0] == 'sales'


# =====================================================================
# Derived Tables with CTEs
# =====================================================================

class TestDerivedTablesWithCTEs:
    def test_cte_with_derived_table(self, db):
        """CTE and derived table together"""
        r = db.execute("""
            WITH eng AS (SELECT name, salary FROM employees WHERE dept = 'eng')
            SELECT d.name
            FROM (SELECT name, salary FROM eng) AS d
            WHERE d.salary > 85000
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_derived_table_references_cte(self, db):
        """Derived table in JOIN references CTE"""
        r = db.execute("""
            WITH dept_avg AS (
                SELECT dept, AVG(salary) AS avg_sal
                FROM employees
                GROUP BY dept
            )
            SELECT e.name, d.avg_sal
            FROM employees AS e
            JOIN (SELECT dept, avg_sal FROM dept_avg) AS d ON e.dept = d.dept
            WHERE e.salary > d.avg_sal
            ORDER BY e.name
        """)
        # eng avg=85000: Alice(90k); sales avg=65000: Carol(70k)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Carol' in names


# =====================================================================
# Derived Tables with Set Operations
# =====================================================================

class TestDerivedTablesSetOps:
    def test_union_in_derived_table(self, db):
        """UNION inside derived table"""
        r = db.execute("""
            SELECT d.name
            FROM (
                SELECT name FROM employees WHERE dept = 'eng'
                UNION
                SELECT name FROM employees WHERE dept = 'hr'
            ) AS d
            ORDER BY d.name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']


# =====================================================================
# Column Aliasing in Derived Tables
# =====================================================================

class TestDerivedTableColumnAliases:
    def test_aliased_columns(self, db):
        """Columns with aliases in derived table"""
        r = db.execute("""
            SELECT d.emp_name, d.emp_salary
            FROM (SELECT name AS emp_name, salary AS emp_salary FROM employees) AS d
            WHERE d.emp_salary > 80000
            ORDER BY d.emp_name
        """)
        assert len(r.rows) == 1
        assert r.rows[0][:2] == ['Alice', 90000]

    def test_computed_column_alias(self, db):
        """Computed column with alias in derived table"""
        r = db.execute("""
            SELECT d.bonus
            FROM (SELECT salary * 2 AS bonus FROM employees WHERE name = 'Alice') AS d
        """)
        assert r.rows[0][0] == 180000

    def test_count_alias_in_derived(self, db):
        """COUNT with alias accessible from outer query"""
        r = db.execute("""
            SELECT d.cnt
            FROM (SELECT COUNT(*) AS cnt FROM employees) AS d
        """)
        assert r.rows[0][0] == 5


# =====================================================================
# Derived Tables Preserving Existing Functionality
# =====================================================================

class TestBackwardCompatibility:
    def test_regular_select(self, db):
        """Regular SELECT still works"""
        r = db.execute("SELECT name FROM employees WHERE dept = 'hr'")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Eve'

    def test_regular_join(self, db_orders):
        """Regular JOIN still works (using table names, not aliases -- known alias limitation)"""
        r = db_orders.execute("""
            SELECT customers.name, orders.amount
            FROM customers
            JOIN orders ON customers.id = orders.customer_id
            WHERE orders.status = 'complete'
            ORDER BY customers.name, orders.amount
        """)
        assert len(r.rows) == 3

    def test_regular_subquery(self, db):
        """Regular subqueries still work"""
        r = db.execute("""
            SELECT name FROM employees
            WHERE salary > (SELECT AVG(salary) FROM employees)
            ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob']

    def test_regular_cte(self, db):
        """CTEs still work"""
        r = db.execute("""
            WITH eng AS (SELECT name FROM employees WHERE dept = 'eng')
            SELECT name FROM eng ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob']

    def test_regular_set_operations(self, db):
        """Set operations still work"""
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'eng'
            UNION
            SELECT name FROM employees WHERE dept = 'hr'
        """)
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Bob', 'Eve']

    def test_insert_update_delete(self, db):
        """DML still works"""
        db.execute("INSERT INTO employees VALUES (6, 'Frank', 'eng', 95000)")
        r = db.execute("SELECT name FROM employees WHERE id = 6")
        assert r.rows[0][0] == 'Frank'

        db.execute("UPDATE employees SET salary = 96000 WHERE id = 6")
        r = db.execute("SELECT salary FROM employees WHERE id = 6")
        assert r.rows[0][0] == 96000

        db.execute("DELETE FROM employees WHERE id = 6")
        r = db.execute("SELECT COUNT(*) FROM employees")
        assert r.rows[0][0] == 5

    def test_create_drop_table(self):
        """DDL still works"""
        db = DerivedTableDB()
        db.execute("CREATE TABLE test (id INT, val TEXT)")
        db.execute("INSERT INTO test VALUES (1, 'hello')")
        r = db.execute("SELECT val FROM test")
        assert r.rows[0][0] == 'hello'
        db.execute("DROP TABLE test")


# =====================================================================
# Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_empty_derived_table(self, db):
        """Derived table that returns no rows"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name FROM employees WHERE dept = 'nonexistent') AS d
        """)
        assert len(r.rows) == 0

    def test_single_row_derived_table(self, db):
        """Derived table returning exactly one row"""
        r = db.execute("""
            SELECT d.cnt
            FROM (SELECT COUNT(*) AS cnt FROM employees) AS d
        """)
        assert r.rows[0][0] == 5

    def test_derived_table_with_null_values(self):
        """Derived table with NULL values"""
        db = DerivedTableDB()
        db.execute("CREATE TABLE t (id INT, val TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'a')")
        db.execute("INSERT INTO t VALUES (2, NULL)")
        r = db.execute("""
            SELECT d.val FROM (SELECT val FROM t ORDER BY id) AS d
        """)
        assert r.rows[0][0] == 'a'
        assert r.rows[1][0] is None

    def test_multiple_derived_tables_in_query(self, db):
        """Multiple derived tables used together"""
        r = db.execute("""
            SELECT a.dept, a.cnt, b.avg_sal
            FROM (SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept) AS a
            JOIN (SELECT dept, AVG(salary) AS avg_sal FROM employees GROUP BY dept) AS b
                ON a.dept = b.dept
            ORDER BY a.dept
        """)
        assert len(r.rows) == 3
        assert r.rows[0][:3] == ['eng', 2, 85000]
        assert r.rows[1][:3] == ['hr', 1, 75000]
        assert r.rows[2][:3] == ['sales', 2, 65000]

    def test_derived_table_column_not_qualified(self, db):
        """Access derived table column without qualifier"""
        r = db.execute("""
            SELECT name FROM (SELECT name FROM employees WHERE dept = 'hr') AS d
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Eve'

    def test_derived_table_with_between(self, db):
        """BETWEEN on derived table column"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name, salary FROM employees) AS d
            WHERE d.salary BETWEEN 70000 AND 85000
            ORDER BY d.name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Carol', 'Eve']

    def test_derived_table_with_like(self, db):
        """LIKE on derived table column"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name FROM employees) AS d
            WHERE d.name LIKE 'A%'
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_derived_table_with_is_null(self):
        """IS NULL on derived table column"""
        db = DerivedTableDB()
        db.execute("CREATE TABLE t (id INT, val TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'a')")
        db.execute("INSERT INTO t VALUES (2, NULL)")
        r = db.execute("""
            SELECT d.id FROM (SELECT id, val FROM t) AS d WHERE d.val IS NULL
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 2

    def test_derived_table_with_in_list(self, db):
        """IN list on derived table column"""
        r = db.execute("""
            SELECT d.name
            FROM (SELECT name, dept FROM employees) AS d
            WHERE d.dept IN ('eng', 'hr')
            ORDER BY d.name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']

    def test_derived_table_with_case(self, db):
        """CASE expression on derived table column"""
        r = db.execute("""
            SELECT d.name,
                   CASE WHEN d.salary > 80000 THEN 'high' ELSE 'normal' END AS level
            FROM (SELECT name, salary FROM employees) AS d
            ORDER BY d.name
        """)
        assert r.rows[0][:2] == ['Alice', 'high']  # 90k
        assert r.rows[1][:2] == ['Bob', 'normal']   # 80k -- not > 80k
        assert r.rows[4][:2] == ['Eve', 'normal']


# =====================================================================
# Complex Compositions
# =====================================================================

class TestComplexCompositions:
    def test_derived_table_with_multiple_joins(self, db_orders):
        """Derived table joined with multiple regular tables"""
        db_orders.execute("CREATE TABLE products (id INT, name TEXT)")
        db_orders.execute("INSERT INTO products VALUES (1, 'Widget')")
        db_orders.execute("INSERT INTO products VALUES (2, 'Gadget')")

        r = db_orders.execute("""
            SELECT d.total_orders
            FROM (SELECT COUNT(*) AS total_orders FROM orders) AS d
        """)
        assert r.rows[0][0] == 5

    def test_top_n_per_group_pattern(self, db):
        """Top-N per group using derived table"""
        r = db.execute("""
            SELECT d.name, d.salary, d.dept
            FROM (SELECT name, salary, dept FROM employees ORDER BY salary DESC) AS d
            WHERE d.salary = (SELECT MAX(salary) FROM employees WHERE dept = d.dept)
            ORDER BY d.dept
        """)
        # Top earner per dept: Alice(eng,90k), Eve(hr,75k), Carol(sales,70k)
        assert len(r.rows) == 3
        assert r.rows[0][:3] == ['Alice', 90000, 'eng']
        assert r.rows[1][:3] == ['Eve', 75000, 'hr']
        assert r.rows[2][:3] == ['Carol', 70000, 'sales']

    def test_running_statistics(self, db):
        """Derived table for computing statistics"""
        r = db.execute("""
            SELECT d.dept, d.total_salary, d.headcount
            FROM (
                SELECT dept,
                       SUM(salary) AS total_salary,
                       COUNT(*) AS headcount
                FROM employees
                GROUP BY dept
            ) AS d
            WHERE d.headcount > 1
            ORDER BY d.total_salary DESC
        """)
        assert len(r.rows) == 2
        assert r.rows[0][0] == 'eng'
        assert r.rows[0][1] == 170000
        assert r.rows[0][2] == 2
        assert r.rows[1][0] == 'sales'
        assert r.rows[1][1] == 130000
        assert r.rows[1][2] == 2

    def test_derived_table_with_not_in(self, db_orders):
        """NOT IN subquery on derived table"""
        r = db_orders.execute("""
            SELECT name FROM customers
            WHERE id NOT IN (
                SELECT d.customer_id
                FROM (SELECT customer_id FROM orders WHERE status = 'complete') AS d
            )
            ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Bob']

    def test_derived_table_in_select_list_subquery(self, db):
        """Scalar subquery in SELECT that uses a derived table"""
        r = db.execute("""
            SELECT name, (
                SELECT d.avg_sal
                FROM (SELECT AVG(salary) AS avg_sal FROM employees) AS d
            ) AS overall_avg
            FROM employees
            WHERE name = 'Alice'
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 75000  # avg of all: (90k+80k+70k+60k+75k)/5

    def test_derived_table_with_functions(self, db):
        """Functions on derived table columns"""
        r = db.execute("""
            SELECT UPPER(d.name), LENGTH(d.name)
            FROM (SELECT name FROM employees WHERE dept = 'hr') AS d
        """)
        assert r.rows[0][0] == 'EVE'
        assert r.rows[0][1] == 3

    def test_arithmetic_on_derived_columns(self, db):
        """Arithmetic expressions on derived table columns"""
        r = db.execute("""
            SELECT d.name, d.salary + 10000 AS raised
            FROM (SELECT name, salary FROM employees WHERE name = 'Alice') AS d
        """)
        assert r.rows[0][:2] == ['Alice', 100000]

    def test_coalesce_on_derived(self):
        """COALESCE on derived table with NULLs"""
        db = DerivedTableDB()
        db.execute("CREATE TABLE t (id INT, val TEXT)")
        db.execute("INSERT INTO t VALUES (1, NULL)")
        db.execute("INSERT INTO t VALUES (2, 'hello')")
        r = db.execute("""
            SELECT COALESCE(d.val, 'default')
            FROM (SELECT val FROM t ORDER BY id) AS d
        """)
        assert r.rows[0][0] == 'default'
        assert r.rows[1][0] == 'hello'


# =====================================================================
# Derived Table Error Handling
# =====================================================================

class TestDerivedTableErrors:
    def test_derived_table_requires_alias(self, db):
        """Derived table without alias should error"""
        with pytest.raises(Exception):
            db.execute("SELECT * FROM (SELECT name FROM employees)")


# =====================================================================
# Multi-statement and execute_many
# =====================================================================

class TestMultiStatement:
    def test_execute_many_with_derived(self, db):
        """execute_many works with derived tables"""
        results = db.execute_many("""
            SELECT COUNT(*) FROM employees;
            SELECT d.name FROM (SELECT name FROM employees WHERE dept = 'hr') AS d
        """)
        assert len(results) == 2
        assert results[0].rows[0][0] == 5
        assert results[1].rows[0][0] == 'Eve'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
