"""
Tests for C266: SQL Subqueries
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from subqueries import (
    SubqueryDB, SubqueryParser, parse_subquery_sql, parse_subquery_sql_multi,
    subquery_eval_expr, SqlSubquery, SqlExists, SqlInSubquery,
    SqlQuantifiedComparison, SubqueryTableRef,
)


@pytest.fixture
def db():
    """Create a SubqueryDB with test data."""
    d = SubqueryDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, dept_id INT, salary INT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 1, 80000)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 1, 90000)")
    d.execute("INSERT INTO employees VALUES (3, 'Charlie', 2, 70000)")
    d.execute("INSERT INTO employees VALUES (4, 'Diana', 2, 95000)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 3, 85000)")

    d.execute("CREATE TABLE departments (id INT, name TEXT, budget INT)")
    d.execute("INSERT INTO departments VALUES (1, 'Engineering', 500000)")
    d.execute("INSERT INTO departments VALUES (2, 'Marketing', 300000)")
    d.execute("INSERT INTO departments VALUES (3, 'Sales', 400000)")
    d.execute("INSERT INTO departments VALUES (4, 'HR', 200000)")

    d.execute("CREATE TABLE orders (id INT, emp_id INT, amount INT)")
    d.execute("INSERT INTO orders VALUES (1, 1, 500)")
    d.execute("INSERT INTO orders VALUES (2, 1, 300)")
    d.execute("INSERT INTO orders VALUES (3, 2, 700)")
    d.execute("INSERT INTO orders VALUES (4, 3, 200)")
    d.execute("INSERT INTO orders VALUES (5, 5, 600)")
    return d


# =============================================================================
# Parser Tests
# =============================================================================

class TestSubqueryParser:
    """Test parsing of subquery syntax."""

    def test_parse_scalar_subquery(self):
        stmt = parse_subquery_sql("SELECT (SELECT 1)")
        col_expr = stmt.columns[0].expr
        assert isinstance(col_expr, SqlSubquery)

    def test_parse_scalar_subquery_with_alias(self):
        stmt = parse_subquery_sql("SELECT (SELECT 1) AS val")
        assert stmt.columns[0].alias == 'val'
        assert isinstance(stmt.columns[0].expr, SqlSubquery)

    def test_parse_exists(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE EXISTS (SELECT 1 FROM departments)")
        assert isinstance(stmt.where, SqlExists)
        assert not stmt.where.negated

    def test_parse_not_exists(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE NOT EXISTS (SELECT 1 FROM departments WHERE id = 99)")
        assert isinstance(stmt.where, SqlExists)
        assert stmt.where.negated

    def test_parse_in_subquery(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE dept_id IN (SELECT id FROM departments)")
        assert isinstance(stmt.where, SqlInSubquery)
        assert not stmt.where.negated

    def test_parse_not_in_subquery(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE dept_id NOT IN (SELECT id FROM departments WHERE budget > 400000)")
        assert isinstance(stmt.where, SqlInSubquery)
        assert stmt.where.negated

    def test_parse_regular_in_still_works(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE id IN (1, 2, 3)")
        from mini_database import SqlInList
        assert isinstance(stmt.where, SqlInList)

    def test_parse_quantified_all(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE salary > ALL (SELECT budget FROM departments)")
        assert isinstance(stmt.where, SqlQuantifiedComparison)
        assert stmt.where.quantifier == 'ALL'
        assert stmt.where.op == '>'

    def test_parse_quantified_any(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE salary = ANY (SELECT salary FROM employees)")
        assert isinstance(stmt.where, SqlQuantifiedComparison)
        assert stmt.where.quantifier == 'ANY'

    def test_parse_quantified_some(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE salary >= SOME (SELECT salary FROM employees)")
        assert isinstance(stmt.where, SqlQuantifiedComparison)
        assert stmt.where.quantifier == 'SOME'

    def test_parse_derived_table(self):
        stmt = parse_subquery_sql("SELECT * FROM (SELECT 1 AS x) AS sub")
        assert isinstance(stmt.from_table, SubqueryTableRef)
        assert stmt.from_table.alias == 'sub'

    def test_parse_derived_table_no_alias_fails(self):
        with pytest.raises(Exception):
            parse_subquery_sql("SELECT * FROM (SELECT 1 AS x)")

    def test_parse_subquery_in_select_list(self):
        stmt = parse_subquery_sql("SELECT id, (SELECT MAX(salary) FROM employees) AS max_sal FROM departments")
        assert isinstance(stmt.columns[1].expr, SqlSubquery)
        assert stmt.columns[1].alias == 'max_sal'

    def test_parse_nested_subqueries(self):
        stmt = parse_subquery_sql(
            "SELECT * FROM employees WHERE dept_id IN "
            "(SELECT id FROM departments WHERE budget > (SELECT 250000))"
        )
        assert isinstance(stmt.where, SqlInSubquery)

    def test_parse_regular_parenthesized_expr(self):
        """Ensure regular (expr) still works."""
        stmt = parse_subquery_sql("SELECT (1 + 2) AS val")
        from mini_database import SqlBinOp
        assert isinstance(stmt.columns[0].expr, SqlBinOp)

    def test_parse_comparison_still_works(self):
        """Ensure regular comparisons still work."""
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE salary > 80000")
        from mini_database import SqlComparison
        assert isinstance(stmt.where, SqlComparison)

    def test_parse_is_null_still_works(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE name IS NOT NULL")
        from mini_database import SqlIsNull
        assert isinstance(stmt.where, SqlIsNull)

    def test_parse_between_still_works(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE salary BETWEEN 70000 AND 90000")
        from mini_database import SqlBetween
        assert isinstance(stmt.where, SqlBetween)

    def test_parse_like_still_works(self):
        stmt = parse_subquery_sql("SELECT * FROM employees WHERE name LIKE 'A%'")
        from mini_database import SqlComparison
        assert isinstance(stmt.where, SqlComparison)
        assert stmt.where.op == 'like'


# =============================================================================
# Scalar Subquery Tests
# =============================================================================

class TestScalarSubqueries:
    """Test scalar subqueries."""

    def test_scalar_subquery_in_select(self, db):
        r = db.execute("SELECT (SELECT 42) AS val")
        assert r.rows[0][0] == 42

    def test_scalar_subquery_max(self, db):
        r = db.execute("SELECT (SELECT MAX(salary) FROM employees) AS max_sal")
        assert r.rows[0][0] == 95000

    def test_scalar_subquery_count(self, db):
        r = db.execute("SELECT (SELECT COUNT(*) FROM departments) AS dept_count")
        assert r.rows[0][0] == 4

    def test_scalar_subquery_in_select_with_columns(self, db):
        r = db.execute(
            "SELECT name, (SELECT MAX(salary) FROM employees) AS max_sal "
            "FROM departments ORDER BY name"
        )
        assert len(r.rows) == 4
        for row in r.rows:
            assert row[1] == 95000

    def test_scalar_subquery_min(self, db):
        r = db.execute("SELECT (SELECT MIN(budget) FROM departments) AS min_budget")
        assert r.rows[0][0] == 200000

    def test_scalar_subquery_avg(self, db):
        r = db.execute("SELECT (SELECT AVG(salary) FROM employees) AS avg_sal")
        assert r.rows[0][0] == 84000.0

    def test_scalar_subquery_returns_null_when_empty(self, db):
        r = db.execute("SELECT (SELECT salary FROM employees WHERE id = 999) AS val")
        assert r.rows[0][0] is None

    def test_scalar_subquery_arithmetic(self, db):
        r = db.execute("SELECT (SELECT MAX(salary) FROM employees) - (SELECT MIN(salary) FROM employees) AS range")
        assert r.rows[0][0] == 25000

    def test_scalar_subquery_in_where(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        # avg = 84000; above: Bob(90k), Diana(95k), Eve(85k)
        assert names == ['Bob', 'Diana', 'Eve']

    def test_scalar_subquery_in_where_eq(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary = (SELECT MAX(salary) FROM employees)"
        )
        assert r.rows[0][0] == 'Diana'

    def test_multiple_scalar_subqueries(self, db):
        r = db.execute(
            "SELECT (SELECT MIN(salary) FROM employees) AS min_sal, "
            "(SELECT MAX(salary) FROM employees) AS max_sal"
        )
        assert r.rows[0][0] == 70000
        assert r.rows[0][1] == 95000


# =============================================================================
# IN Subquery Tests
# =============================================================================

class TestInSubqueries:
    """Test IN (subquery) and NOT IN (subquery)."""

    def test_in_subquery_basic(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE budget >= 400000) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        # Engineering(500k) and Sales(400k) -> dept 1,3
        assert names == ['Alice', 'Bob', 'Eve']

    def test_not_in_subquery(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE dept_id NOT IN (SELECT id FROM departments WHERE budget >= 400000) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        # Marketing(300k) -> dept 2
        assert names == ['Charlie', 'Diana']

    def test_in_subquery_with_single_value(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Sales')"
        )
        assert r.rows[0][0] == 'Eve'

    def test_in_subquery_empty_result(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE budget > 999999)"
        )
        assert r.rows == []

    def test_not_in_subquery_all_match(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE dept_id NOT IN (SELECT id FROM departments WHERE budget > 999999) ORDER BY name"
        )
        assert len(r.rows) == 5

    def test_in_subquery_different_columns(self, db):
        """IN subquery where outer and inner columns differ."""
        r = db.execute(
            "SELECT name FROM employees WHERE id IN (SELECT emp_id FROM orders) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        # orders: emp_ids 1,1,2,3,5
        assert names == ['Alice', 'Bob', 'Charlie', 'Eve']

    def test_in_subquery_with_where(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE id IN (SELECT emp_id FROM orders WHERE amount > 400) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        # orders > 400: emp 1(500), 2(700), 5(600)
        assert names == ['Alice', 'Bob', 'Eve']


# =============================================================================
# EXISTS Tests
# =============================================================================

class TestExistsSubqueries:
    """Test EXISTS and NOT EXISTS."""

    def test_exists_true(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM orders WHERE orders.emp_id = employees.id) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob', 'Charlie', 'Eve']

    def test_exists_false(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM departments WHERE id = 999)"
        )
        assert r.rows == []

    def test_not_exists(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.emp_id = employees.id) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        # Diana has no orders
        assert names == ['Diana']

    def test_exists_always_true(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM departments) ORDER BY name"
        )
        assert len(r.rows) == 5

    def test_exists_with_condition(self, db):
        r = db.execute(
            "SELECT name FROM departments WHERE EXISTS (SELECT 1 FROM employees WHERE employees.dept_id = departments.id) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        # HR (dept 4) has no employees
        assert 'HR' not in names
        assert len(names) == 3


# =============================================================================
# Quantified Comparison Tests (ALL/ANY/SOME)
# =============================================================================

class TestQuantifiedComparisons:
    """Test ALL, ANY, SOME quantified comparisons."""

    def test_greater_than_all(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary > ALL (SELECT budget FROM departments WHERE budget < 300000)"
        )
        # budget < 300k: HR(200k). All employees have salary > 200k? No, let's check
        # Actually all salaries (70k-95k) < 200k... Wait, budget 200000 vs salary 70000-95000
        # salary > 200000 is false for all
        assert r.rows == []

    def test_greater_than_all_subquery(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary > ALL (SELECT salary FROM employees WHERE dept_id = 2) ORDER BY name"
        )
        # dept 2: Charlie(70k), Diana(95k). Need salary > 95k. Nobody qualifies.
        assert r.rows == []

    def test_less_than_all(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary < ALL (SELECT salary FROM employees WHERE dept_id = 1) ORDER BY name"
        )
        # dept 1: Alice(80k), Bob(90k). Need salary < 80k. Charlie(70k) qualifies.
        assert r.rows[0][0] == 'Charlie'

    def test_equal_any(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary = ANY (SELECT salary FROM employees WHERE dept_id = 1) ORDER BY name"
        )
        # dept 1 salaries: 80k, 90k
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Bob' in names

    def test_greater_than_any(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary > ANY (SELECT salary FROM employees WHERE dept_id = 2) ORDER BY name"
        )
        # dept 2: 70k, 95k. Need salary > 70k (at least one).
        names = [row[0] for row in r.rows]
        # Alice(80k), Bob(90k), Diana(95k), Eve(85k) > 70k
        assert 'Alice' in names
        assert 'Charlie' not in names  # 70k not > 70k

    def test_some_equals_any(self, db):
        """SOME is a synonym for ANY."""
        r = db.execute(
            "SELECT name FROM employees WHERE salary > SOME (SELECT salary FROM employees WHERE dept_id = 2) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert 'Alice' in names

    def test_all_with_empty_set(self, db):
        """ALL with empty set should return all rows (vacuously true)."""
        r = db.execute(
            "SELECT name FROM employees WHERE salary > ALL (SELECT salary FROM employees WHERE id = 999) ORDER BY name"
        )
        assert len(r.rows) == 5

    def test_any_with_empty_set(self, db):
        """ANY with empty set should return no rows."""
        r = db.execute(
            "SELECT name FROM employees WHERE salary > ANY (SELECT salary FROM employees WHERE id = 999)"
        )
        assert r.rows == []


# =============================================================================
# Derived Table (FROM subquery) Tests
# =============================================================================

class TestDerivedTables:
    """Test subqueries in FROM clause."""

    def test_simple_derived_table(self, db):
        r = db.execute(
            "SELECT * FROM (SELECT id, name FROM employees WHERE salary > 80000) AS high_earners ORDER BY name"
        )
        names = [row[1] for row in r.rows]
        assert names == ['Bob', 'Diana', 'Eve']

    def test_derived_table_with_alias_columns(self, db):
        r = db.execute(
            "SELECT high_earners.name FROM (SELECT id, name FROM employees WHERE salary > 80000) AS high_earners ORDER BY high_earners.name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Diana', 'Eve']

    def test_derived_table_with_where(self, db):
        r = db.execute(
            "SELECT name FROM (SELECT id, name, salary FROM employees) AS e WHERE salary > 85000 ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Diana']

    def test_derived_table_aggregation(self, db):
        r = db.execute(
            "SELECT * FROM (SELECT dept_id, COUNT(*) AS cnt FROM employees GROUP BY dept_id) AS dept_counts ORDER BY dept_id"
        )
        assert r.rows[0][1] == 2  # dept 1: 2 employees
        assert r.rows[1][1] == 2  # dept 2: 2 employees
        assert r.rows[2][1] == 1  # dept 3: 1 employee

    def test_derived_table_with_limit(self, db):
        r = db.execute(
            "SELECT name FROM (SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3) AS top3 ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert len(names) == 3

    def test_derived_table_expression_only(self, db):
        r = db.execute(
            "SELECT x FROM (SELECT 42 AS x) AS sub"
        )
        assert r.rows[0][0] == 42


# =============================================================================
# Correlated Subquery Tests
# =============================================================================

class TestCorrelatedSubqueries:
    """Test correlated subqueries (referencing outer query)."""

    def test_correlated_exists(self, db):
        """Employees who have at least one order."""
        r = db.execute(
            "SELECT name FROM employees WHERE EXISTS "
            "(SELECT 1 FROM orders WHERE orders.emp_id = employees.id) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob', 'Charlie', 'Eve']

    def test_correlated_not_exists(self, db):
        """Employees with no orders."""
        r = db.execute(
            "SELECT name FROM employees WHERE NOT EXISTS "
            "(SELECT 1 FROM orders WHERE orders.emp_id = employees.id) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['Diana']

    def test_correlated_scalar(self, db):
        """Correlated scalar subquery: count of orders per employee."""
        r = db.execute(
            "SELECT name, (SELECT COUNT(*) FROM orders WHERE orders.emp_id = employees.id) AS order_count "
            "FROM employees ORDER BY name"
        )
        data = {row[0]: row[1] for row in r.rows}
        assert data['Alice'] == 2
        assert data['Bob'] == 1
        assert data['Diana'] == 0

    def test_correlated_scalar_sum(self, db):
        """Correlated scalar subquery: sum of order amounts."""
        r = db.execute(
            "SELECT name, (SELECT SUM(amount) FROM orders WHERE orders.emp_id = employees.id) AS total "
            "FROM employees ORDER BY name"
        )
        data = {row[0]: row[1] for row in r.rows}
        assert data['Alice'] == 800  # 500 + 300
        assert data['Bob'] == 700

    def test_correlated_where_scalar(self, db):
        """Correlated subquery in WHERE clause."""
        r = db.execute(
            "SELECT name FROM employees WHERE salary > "
            "(SELECT AVG(salary) FROM employees AS e2 WHERE e2.dept_id = employees.dept_id) ORDER BY name"
        )
        # dept 1 avg: (80k+90k)/2=85k. Bob(90k) > 85k. Alice(80k) < 85k.
        # dept 2 avg: (70k+95k)/2=82.5k. Diana(95k) > 82.5k.
        # dept 3: only Eve, avg=85k. Eve not > 85k.
        names = [row[0] for row in r.rows]
        assert 'Bob' in names
        assert 'Diana' in names

    def test_correlated_exists_with_departments(self, db):
        """Departments that have employees."""
        r = db.execute(
            "SELECT name FROM departments WHERE EXISTS "
            "(SELECT 1 FROM employees WHERE employees.dept_id = departments.id) ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert 'HR' not in names
        assert len(names) == 3


# =============================================================================
# Nested Subquery Tests
# =============================================================================

class TestNestedSubqueries:
    """Test nested (multi-level) subqueries."""

    def test_subquery_in_subquery(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE dept_id IN "
            "(SELECT id FROM departments WHERE budget > (SELECT AVG(budget) FROM departments)) ORDER BY name"
        )
        # AVG budget = (500k+300k+400k+200k)/4 = 350k
        # Departments > 350k: Engineering(500k), Sales(400k) -> dept 1,3
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']

    def test_double_nested_scalar(self, db):
        r = db.execute(
            "SELECT (SELECT (SELECT 42)) AS val"
        )
        assert r.rows[0][0] == 42

    def test_nested_in_and_scalar(self, db):
        """IN subquery combined with scalar subquery."""
        r = db.execute(
            "SELECT name FROM employees "
            "WHERE dept_id IN (SELECT id FROM departments WHERE budget > 300000) "
            "AND salary > (SELECT AVG(salary) FROM employees) ORDER BY name"
        )
        # dept > 300k budget: Engineering(500k), Sales(400k) -> dept 1,3
        # avg salary = 84000
        # Alice(80k, dept1): no (< avg). Bob(90k, dept1): yes. Eve(85k, dept3): yes.
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Eve']


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_subquery_returns_null(self, db):
        r = db.execute("SELECT (SELECT salary FROM employees WHERE id = 999) AS val")
        assert r.rows[0][0] is None

    def test_in_subquery_with_nulls(self, db):
        db.execute("CREATE TABLE nullable (id INT, val INT)")
        db.execute("INSERT INTO nullable VALUES (1, NULL)")
        db.execute("INSERT INTO nullable VALUES (2, 10)")
        r = db.execute("SELECT id FROM nullable WHERE val IN (SELECT salary FROM employees WHERE id = 999)")
        assert r.rows == []

    def test_empty_table_subquery(self, db):
        db.execute("CREATE TABLE empty_table (id INT, val INT)")
        r = db.execute("SELECT (SELECT COUNT(*) FROM empty_table) AS cnt")
        assert r.rows[0][0] == 0

    def test_derived_table_empty(self, db):
        r = db.execute(
            "SELECT * FROM (SELECT id FROM employees WHERE id = 999) AS empty_sub"
        )
        assert r.rows == []

    def test_subquery_with_distinct(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE dept_id IN "
            "(SELECT DISTINCT dept_id FROM employees WHERE salary > 75000) ORDER BY name"
        )
        # DISTINCT dept_ids where salary > 75k: 1 (Alice,Bob), 2 (Diana), 3 (Eve)
        assert len(r.rows) == 5

    def test_subquery_with_limit(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary = "
            "(SELECT salary FROM employees ORDER BY salary DESC LIMIT 1)"
        )
        assert r.rows[0][0] == 'Diana'

    def test_subquery_with_offset(self, db):
        r = db.execute(
            "SELECT name FROM employees WHERE salary = "
            "(SELECT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1)"
        )
        assert r.rows[0][0] == 'Bob'  # 2nd highest salary

    def test_multiple_in_subqueries(self, db):
        """Multiple subqueries in WHERE with AND."""
        r = db.execute(
            "SELECT name FROM employees "
            "WHERE dept_id IN (SELECT id FROM departments WHERE budget >= 400000) "
            "AND id IN (SELECT emp_id FROM orders) ORDER BY name"
        )
        # dept >= 400k: 1,3. Has orders: 1,1,2,3,5.
        # Intersection: Alice(1,dept1), Bob(2,dept1), Eve(5,dept3)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']

    def test_regular_query_still_works(self, db):
        """Ensure non-subquery queries still work."""
        r = db.execute("SELECT name FROM employees WHERE salary > 80000 ORDER BY name")
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Diana', 'Eve']

    def test_regular_in_list_still_works(self, db):
        r = db.execute("SELECT name FROM employees WHERE id IN (1, 3, 5) ORDER BY name")
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Charlie', 'Eve']

    def test_regular_not_in_list_still_works(self, db):
        r = db.execute("SELECT name FROM employees WHERE id NOT IN (1, 3, 5) ORDER BY name")
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Diana']

    def test_builtin_functions_still_work(self, db):
        r = db.execute("SELECT UPPER(name) FROM employees WHERE id = 1")
        assert r.rows[0][0] == 'ALICE'

    def test_window_functions_still_work(self, db):
        r = db.execute(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn FROM employees ORDER BY rn"
        )
        assert r.rows[0][0] == 'Diana'  # highest salary

    def test_case_expression_still_works(self, db):
        r = db.execute(
            "SELECT name, CASE WHEN salary > 85000 THEN 'high' ELSE 'low' END AS level "
            "FROM employees ORDER BY name"
        )
        data = {row[0]: row[1] for row in r.rows}
        assert data['Diana'] == 'high'
        assert data['Charlie'] == 'low'


# =============================================================================
# Complex Integration Tests
# =============================================================================

class TestComplexQueries:
    """Test complex queries combining subqueries with other features."""

    def test_subquery_with_order_by(self, db):
        r = db.execute(
            "SELECT name FROM employees "
            "WHERE dept_id IN (SELECT id FROM departments WHERE budget > 300000) "
            "ORDER BY salary DESC"
        )
        names = [row[0] for row in r.rows]
        assert names[0] == 'Diana'  # 95k
        assert names[-1] == 'Alice'  # 80k (wait, Alice is dept 1 which is > 300k)
        # dept > 300k: Engineering(500k), Sales(400k) -> dept 1,3
        # Alice(80k,1), Bob(90k,1), Eve(85k,3)
        # Wait Diana is dept 2 (Marketing, 300k) -- not > 300k
        # So Diana not included
        assert 'Diana' not in names

    def test_subquery_with_distinct_outer(self, db):
        r = db.execute(
            "SELECT DISTINCT dept_id FROM employees "
            "WHERE id IN (SELECT emp_id FROM orders) ORDER BY dept_id"
        )
        dept_ids = [row[0] for row in r.rows]
        # orders: emp 1,1,2,3,5. Depts: 1,1,1,2,3. Distinct: 1,2,3
        assert dept_ids == [1, 2, 3]

    def test_subquery_with_limit_outer(self, db):
        r = db.execute(
            "SELECT name FROM employees "
            "WHERE dept_id IN (SELECT id FROM departments WHERE budget > 300000) "
            "ORDER BY salary DESC LIMIT 2"
        )
        assert len(r.rows) == 2
        assert r.rows[0][0] == 'Bob'  # 90k

    def test_correlated_with_aggregate(self, db):
        """Correlated subquery with aggregate -- classic 'above department average' pattern."""
        r = db.execute(
            "SELECT name, salary FROM employees "
            "WHERE salary > (SELECT AVG(salary) FROM employees AS e2 WHERE e2.dept_id = employees.dept_id) "
            "ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert 'Bob' in names
        assert 'Diana' in names

    def test_exists_as_semi_join(self, db):
        """EXISTS used as semi-join pattern."""
        r = db.execute(
            "SELECT d.name FROM departments AS d "
            "WHERE EXISTS (SELECT 1 FROM employees AS e WHERE e.dept_id = d.id AND e.salary > 85000) "
            "ORDER BY d.name"
        )
        names = [row[0] for row in r.rows]
        # Bob(90k, dept1), Diana(95k, dept2) -> Engineering, Marketing
        assert 'Engineering' in names
        assert 'Marketing' in names

    def test_not_exists_as_anti_join(self, db):
        """NOT EXISTS used as anti-join pattern."""
        r = db.execute(
            "SELECT d.name FROM departments AS d "
            "WHERE NOT EXISTS (SELECT 1 FROM employees AS e WHERE e.dept_id = d.id) "
            "ORDER BY d.name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['HR']

    def test_subquery_in_case(self, db):
        """Subquery inside CASE expression."""
        r = db.execute(
            "SELECT name, "
            "CASE WHEN salary > (SELECT AVG(salary) FROM employees) THEN 'above' ELSE 'below' END AS position "
            "FROM employees ORDER BY name"
        )
        data = {row[0]: row[1] for row in r.rows}
        # avg = 84000
        assert data['Alice'] == 'below'  # 80k
        assert data['Bob'] == 'above'  # 90k

    def test_multiple_correlated_in_select(self, db):
        """Multiple correlated subqueries in SELECT list."""
        r = db.execute(
            "SELECT name, "
            "(SELECT COUNT(*) FROM orders WHERE orders.emp_id = employees.id) AS orders, "
            "(SELECT SUM(amount) FROM orders WHERE orders.emp_id = employees.id) AS total "
            "FROM employees WHERE id = 1"
        )
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 2
        assert r.rows[0][2] == 800


# =============================================================================
# Derived Table Advanced Tests
# =============================================================================

class TestDerivedTableAdvanced:
    """Advanced derived table tests."""

    def test_derived_table_with_join(self, db):
        """Join a regular table with a derived table."""
        r = db.execute(
            "SELECT e.name, totals.total_amount "
            "FROM employees AS e "
            "JOIN (SELECT emp_id, SUM(amount) AS total_amount FROM orders GROUP BY emp_id) AS totals "
            "ON e.id = totals.emp_id "
            "ORDER BY totals.total_amount DESC"
        )
        assert len(r.rows) >= 1
        # Alice: 800, Bob: 700, Eve: 600, Charlie: 200
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 800

    def test_derived_table_with_where_on_derived(self, db):
        r = db.execute(
            "SELECT name FROM (SELECT name, salary FROM employees) AS sub WHERE salary >= 90000 ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Diana']

    def test_nested_derived_tables(self, db):
        """Derived table inside another derived table."""
        r = db.execute(
            "SELECT name FROM "
            "(SELECT name, salary FROM "
            "(SELECT id, name, salary FROM employees WHERE salary > 70000) AS inner_sub) AS outer_sub "
            "WHERE salary > 85000 ORDER BY name"
        )
        names = [row[0] for row in r.rows]
        assert names == ['Bob', 'Diana']


# =============================================================================
# Performance / Stress Tests
# =============================================================================

class TestStress:
    """Test with larger data sets."""

    def test_large_in_subquery(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE big (id INT, val INT)")
        db.execute("CREATE TABLE small_t (id INT)")
        for i in range(100):
            db.execute(f"INSERT INTO big VALUES ({i}, {i * 10})")
        for i in range(0, 100, 5):
            db.execute(f"INSERT INTO small_t VALUES ({i})")
        r = db.execute("SELECT COUNT(*) FROM big WHERE id IN (SELECT id FROM small_t)")
        assert r.rows[0][0] == 20  # 0,5,10,...,95

    def test_correlated_on_larger_data(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE items (id INT, category INT, price INT)")
        for i in range(50):
            cat = i % 5
            price = 100 + i * 7
            db.execute(f"INSERT INTO items VALUES ({i}, {cat}, {price})")
        r = db.execute(
            "SELECT id FROM items WHERE price > "
            "(SELECT AVG(price) FROM items AS i2 WHERE i2.category = items.category) "
            "ORDER BY id"
        )
        assert len(r.rows) > 0

    def test_exists_on_larger_data(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE parents (id INT, name TEXT)")
        db.execute("CREATE TABLE children (id INT, parent_id INT)")
        for i in range(30):
            db.execute(f"INSERT INTO parents VALUES ({i}, 'parent_{i}')")
        for i in range(0, 30, 3):
            db.execute(f"INSERT INTO children VALUES ({i * 10}, {i})")
        r = db.execute(
            "SELECT name FROM parents WHERE EXISTS (SELECT 1 FROM children WHERE children.parent_id = parents.id) ORDER BY name"
        )
        assert len(r.rows) == 10  # every 3rd parent has a child


# =============================================================================
# Regression Tests -- ensure base features still work
# =============================================================================

class TestRegression:
    """Ensure all previous functionality still works."""

    def test_create_insert_select(self, db):
        r = db.execute("SELECT COUNT(*) FROM employees")
        assert r.rows[0][0] == 5

    def test_update(self, db):
        db.execute("UPDATE employees SET salary = 100000 WHERE name = 'Alice'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert r.rows[0][0] == 100000

    def test_delete(self, db):
        db.execute("DELETE FROM employees WHERE name = 'Eve'")
        r = db.execute("SELECT COUNT(*) FROM employees")
        assert r.rows[0][0] == 4

    def test_join(self, db):
        r = db.execute(
            "SELECT e.name, d.name AS dept FROM employees AS e "
            "JOIN departments AS d ON e.dept_id = d.id ORDER BY e.name"
        )
        assert len(r.rows) == 5

    def test_group_by(self, db):
        r = db.execute(
            "SELECT dept_id, COUNT(*) AS cnt FROM employees GROUP BY dept_id ORDER BY dept_id"
        )
        assert r.rows[0][1] == 2  # dept 1

    def test_having(self, db):
        r = db.execute(
            "SELECT dept_id, COUNT(*) AS cnt FROM employees GROUP BY dept_id HAVING cnt >= 2 ORDER BY dept_id"
        )
        assert len(r.rows) == 2  # dept 1 and 2

    def test_order_by_desc(self, db):
        r = db.execute("SELECT name FROM employees ORDER BY salary DESC")
        assert r.rows[0][0] == 'Diana'

    def test_limit_offset(self, db):
        r = db.execute("SELECT name FROM employees ORDER BY id LIMIT 2 OFFSET 1")
        assert len(r.rows) == 2
        assert r.rows[0][0] == 'Bob'

    def test_between(self, db):
        r = db.execute("SELECT name FROM employees WHERE salary BETWEEN 80000 AND 90000 ORDER BY name")
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Eve' in names

    def test_is_null(self, db):
        db.execute("CREATE TABLE test_null (id INT, val INT)")
        db.execute("INSERT INTO test_null VALUES (1, NULL)")
        db.execute("INSERT INTO test_null VALUES (2, 10)")
        r = db.execute("SELECT id FROM test_null WHERE val IS NULL")
        assert r.rows[0][0] == 1

    def test_like(self, db):
        r = db.execute("SELECT name FROM employees WHERE name LIKE 'A%'")
        assert r.rows[0][0] == 'Alice'

    def test_math_functions(self, db):
        r = db.execute("SELECT ABS(-5), ROUND(3.7)")
        assert r.rows[0][0] == 5
        assert r.rows[0][1] == 4

    def test_string_functions(self, db):
        r = db.execute("SELECT UPPER('hello'), LENGTH('world')")
        assert r.rows[0][0] == 'HELLO'
        assert r.rows[0][1] == 5

    def test_coalesce(self, db):
        r = db.execute("SELECT COALESCE(NULL, NULL, 42)")
        assert r.rows[0][0] == 42

    def test_cast(self, db):
        r = db.execute("SELECT CAST(3.7 AS INTEGER)")
        assert r.rows[0][0] == 3

    def test_case_when(self, db):
        r = db.execute("SELECT CASE WHEN 1 = 1 THEN 'yes' ELSE 'no' END")
        assert r.rows[0][0] == 'yes'

    def test_execute_many(self, db):
        results = db.execute_many("SELECT 1; SELECT 2")
        assert results[0].rows[0][0] == 1
        assert results[1].rows[0][0] == 2
