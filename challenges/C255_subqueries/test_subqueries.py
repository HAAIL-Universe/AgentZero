"""
Tests for C255: SQL Subqueries
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from subqueries import SubqueryDB, DatabaseError, ParseError


@pytest.fixture
def db():
    d = SubqueryDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary INT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 85000)")
    d.execute("INSERT INTO employees VALUES (3, 'Carol', 'Marketing', 75000)")
    d.execute("INSERT INTO employees VALUES (4, 'Dave', 'Marketing', 70000)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 'Sales', 80000)")
    d.execute("INSERT INTO employees VALUES (6, 'Frank', 'Sales', 65000)")

    d.execute("CREATE TABLE departments (name TEXT, budget INT, manager_id INT)")
    d.execute("INSERT INTO departments VALUES ('Engineering', 500000, 1)")
    d.execute("INSERT INTO departments VALUES ('Marketing', 300000, 3)")
    d.execute("INSERT INTO departments VALUES ('Sales', 250000, 5)")

    d.execute("CREATE TABLE projects (id INT, name TEXT, dept TEXT, cost INT)")
    d.execute("INSERT INTO projects VALUES (1, 'Alpha', 'Engineering', 100000)")
    d.execute("INSERT INTO projects VALUES (2, 'Beta', 'Engineering', 200000)")
    d.execute("INSERT INTO projects VALUES (3, 'Gamma', 'Marketing', 50000)")
    d.execute("INSERT INTO projects VALUES (4, 'Delta', 'Sales', 75000)")
    return d


# =========================================================================
# Scalar Subqueries
# =========================================================================

class TestScalarSubqueries:
    def test_scalar_in_select(self, db):
        """Scalar subquery in SELECT list."""
        r = db.execute("SELECT name, (SELECT MAX(salary) FROM employees) AS max_sal FROM employees WHERE id = 1")
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 90000

    def test_scalar_in_where(self, db):
        """Scalar subquery in WHERE clause."""
        r = db.execute("SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)")
        names = [row[0] for row in r.rows]
        # avg = (90+85+75+70+80+65)/6 = 77500
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Eve' in names
        assert 'Carol' not in names

    def test_scalar_in_where_eq(self, db):
        """Scalar subquery with = comparison."""
        r = db.execute("SELECT name FROM employees WHERE salary = (SELECT MAX(salary) FROM employees)")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_scalar_returns_null(self, db):
        """Scalar subquery returning no rows yields NULL."""
        r = db.execute("SELECT (SELECT name FROM employees WHERE id = 999) AS missing")
        assert r.rows[0][0] is None

    def test_scalar_too_many_rows(self, db):
        """Scalar subquery returning multiple rows raises error."""
        with pytest.raises(DatabaseError, match="more than one row"):
            db.execute("SELECT (SELECT name FROM employees) AS all_names")

    def test_scalar_with_aggregates(self, db):
        """Scalar subquery with aggregate function."""
        r = db.execute("SELECT name, salary - (SELECT MIN(salary) FROM employees) AS above_min FROM employees WHERE id = 1")
        assert r.rows[0][1] == 90000 - 65000

    def test_scalar_min_salary(self, db):
        """Compare to MIN subquery."""
        r = db.execute("SELECT name FROM employees WHERE salary = (SELECT MIN(salary) FROM employees)")
        assert r.rows[0][0] == 'Frank'

    def test_scalar_count(self, db):
        """Scalar subquery with COUNT."""
        r = db.execute("SELECT (SELECT COUNT(*) FROM employees) AS total")
        assert r.rows[0][0] == 6


# =========================================================================
# IN Subqueries
# =========================================================================

class TestInSubqueries:
    def test_in_subquery_basic(self, db):
        """Basic IN subquery."""
        r = db.execute("SELECT name FROM employees WHERE dept IN (SELECT name FROM departments WHERE budget > 400000)")
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Carol' not in names

    def test_in_subquery_all_match(self, db):
        """IN subquery matching all departments."""
        r = db.execute("SELECT name FROM employees WHERE dept IN (SELECT name FROM departments)")
        assert len(r.rows) == 6

    def test_in_subquery_no_match(self, db):
        """IN subquery with no matching values."""
        r = db.execute("SELECT name FROM employees WHERE dept IN (SELECT name FROM departments WHERE budget > 1000000)")
        assert len(r.rows) == 0

    def test_not_in_subquery(self, db):
        """NOT IN subquery."""
        r = db.execute("SELECT name FROM employees WHERE dept NOT IN (SELECT name FROM departments WHERE budget < 300000)")
        names = [row[0] for row in r.rows]
        # Sales budget is 250000, so exclude Sales
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Carol' in names
        assert 'Eve' not in names

    def test_in_subquery_with_where(self, db):
        """IN subquery with WHERE in both outer and inner."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE salary > 70000
            AND dept IN (SELECT dept FROM projects WHERE cost > 100000)
        """)
        names = [row[0] for row in r.rows]
        # Projects with cost > 100000: Beta (Engineering)
        # Employees in Engineering with salary > 70000: Alice, Bob
        assert set(names) == {'Alice', 'Bob'}

    def test_in_subquery_single_column_validation(self, db):
        """IN subquery must return exactly one column."""
        with pytest.raises(DatabaseError, match="one column"):
            db.execute("SELECT name FROM employees WHERE id IN (SELECT id, name FROM employees)")

    def test_in_subquery_with_aggregate(self, db):
        """IN subquery using aggregate in inner query."""
        r = db.execute("""
            SELECT name FROM departments
            WHERE manager_id IN (SELECT id FROM employees WHERE salary > 80000)
        """)
        names = [row[0] for row in r.rows]
        # Employees with salary > 80000: Alice (90k, id=1), Bob (85k, id=2)
        # Department managers: Engineering(1), Marketing(3), Sales(5)
        assert names == ['Engineering']


# =========================================================================
# EXISTS Subqueries
# =========================================================================

class TestExistsSubqueries:
    def test_exists_basic(self, db):
        """Basic EXISTS subquery."""
        r = db.execute("""
            SELECT name FROM departments d
            WHERE EXISTS (SELECT 1 FROM projects WHERE dept = d.name)
        """)
        names = sorted([row[0] for row in r.rows])
        assert names == ['Engineering', 'Marketing', 'Sales']

    def test_exists_no_match(self, db):
        """EXISTS with no matching rows."""
        r = db.execute("""
            SELECT name FROM departments
            WHERE EXISTS (SELECT 1 FROM employees WHERE salary > 200000)
        """)
        assert len(r.rows) == 0

    def test_not_exists(self, db):
        """NOT EXISTS subquery."""
        # Add a department with no projects
        db.execute("INSERT INTO departments VALUES ('HR', 100000, 7)")
        r = db.execute("""
            SELECT name FROM departments d
            WHERE NOT EXISTS (SELECT 1 FROM projects WHERE dept = d.name)
        """)
        names = [row[0] for row in r.rows]
        assert 'HR' in names

    def test_exists_correlated(self, db):
        """Correlated EXISTS subquery -- inner query references outer."""
        r = db.execute("""
            SELECT name FROM employees e
            WHERE EXISTS (
                SELECT 1 FROM departments d WHERE d.manager_id = e.id
            )
        """)
        names = sorted([row[0] for row in r.rows])
        # Managers: Alice(1), Carol(3), Eve(5)
        assert names == ['Alice', 'Carol', 'Eve']

    def test_exists_with_count(self, db):
        """EXISTS always true if inner query uses aggregate (always produces row)."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE EXISTS (SELECT COUNT(*) FROM departments)
        """)
        # COUNT always returns a row, so EXISTS is true for all
        assert len(r.rows) == 6


# =========================================================================
# Correlated Subqueries
# =========================================================================

class TestCorrelatedSubqueries:
    def test_correlated_scalar(self, db):
        """Correlated scalar subquery in SELECT list."""
        r = db.execute("""
            SELECT e.name, (SELECT COUNT(*) FROM projects p WHERE p.dept = e.dept) AS proj_count
            FROM employees e WHERE e.id <= 3
        """)
        # Alice (Engineering, 2 projects), Bob (Engineering, 2), Carol (Marketing, 1)
        for row in r.rows:
            if row[0] == 'Alice':
                assert row[1] == 2
            elif row[0] == 'Bob':
                assert row[1] == 2
            elif row[0] == 'Carol':
                assert row[1] == 1

    def test_correlated_where(self, db):
        """Correlated subquery in WHERE clause."""
        r = db.execute("""
            SELECT name, salary FROM employees e
            WHERE salary > (
                SELECT AVG(salary) FROM employees e2 WHERE e2.dept = e.dept
            )
        """)
        names = [row[0] for row in r.rows]
        # Engineering avg = 87500: Alice 90000 > avg
        # Marketing avg = 72500: Carol 75000 > avg
        # Sales avg = 72500: Eve 80000 > avg
        assert 'Alice' in names
        assert 'Carol' in names
        assert 'Eve' in names

    def test_correlated_in_select(self, db):
        """Correlated scalar subquery computing department budget."""
        r = db.execute("""
            SELECT e.name, (SELECT d.budget FROM departments d WHERE d.name = e.dept) AS dept_budget
            FROM employees e WHERE e.id = 1
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 500000

    def test_correlated_exists_filter(self, db):
        """Correlated EXISTS to find employees who manage a department."""
        r = db.execute("""
            SELECT name FROM employees e
            WHERE EXISTS (SELECT 1 FROM departments WHERE manager_id = e.id)
        """)
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Carol', 'Eve']


# =========================================================================
# Subqueries with Comparison Operators
# =========================================================================

class TestComparisonSubqueries:
    def test_greater_than_subquery(self, db):
        """x > (SELECT ...)."""
        r = db.execute("SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)")
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Frank' not in names

    def test_less_than_subquery(self, db):
        """x < (SELECT ...)."""
        r = db.execute("SELECT name FROM employees WHERE salary < (SELECT AVG(salary) FROM employees)")
        names = [row[0] for row in r.rows]
        assert 'Carol' in names
        assert 'Dave' in names
        assert 'Frank' in names

    def test_ne_subquery(self, db):
        """x != (SELECT ...)."""
        r = db.execute("SELECT name FROM employees WHERE salary != (SELECT MAX(salary) FROM employees)")
        names = [row[0] for row in r.rows]
        assert 'Alice' not in names
        assert len(names) == 5

    def test_le_subquery(self, db):
        """x <= (SELECT ...)."""
        r = db.execute("SELECT name FROM employees WHERE salary <= (SELECT MIN(salary) FROM employees)")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Frank'

    def test_ge_subquery(self, db):
        """x >= (SELECT ...)."""
        r = db.execute("SELECT name FROM employees WHERE salary >= (SELECT MAX(salary) FROM employees)")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'


# =========================================================================
# Nested Subqueries
# =========================================================================

class TestNestedSubqueries:
    def test_subquery_in_subquery(self, db):
        """Subquery inside another subquery."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE dept IN (
                SELECT name FROM departments
                WHERE budget > (SELECT AVG(budget) FROM departments)
            )
        """)
        names = [row[0] for row in r.rows]
        # avg budget = (500+300+250)/3 = 350000
        # Engineering (500k) > avg
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Carol' not in names

    def test_double_scalar_subquery(self, db):
        """Two scalar subqueries in same SELECT."""
        r = db.execute("""
            SELECT
                (SELECT MAX(salary) FROM employees) AS max_sal,
                (SELECT MIN(salary) FROM employees) AS min_sal
        """)
        assert r.rows[0][0] == 90000
        assert r.rows[0][1] == 65000

    def test_subquery_in_having(self, db):
        """Subquery in HAVING clause."""
        r = db.execute("""
            SELECT dept, AVG(salary) AS avg_sal FROM employees
            GROUP BY dept
            HAVING AVG(salary) > (SELECT AVG(salary) FROM employees)
        """)
        depts = [row[0] for row in r.rows]
        # Overall avg = 77500
        # Engineering avg = 87500 > 77500 -> yes
        # Marketing avg = 72500 < 77500 -> no
        # Sales avg = 72500 < 77500 -> no
        assert depts == ['Engineering']


# =========================================================================
# Subqueries with JOINs
# =========================================================================

class TestSubqueriesWithJoins:
    def test_subquery_with_join(self, db):
        """Subquery referencing joined tables."""
        r = db.execute("""
            SELECT employees.name FROM employees
            JOIN departments ON employees.dept = departments.name
            WHERE departments.budget > (SELECT AVG(budget) FROM departments)
        """)
        names = [row[0] for row in r.rows]
        assert set(names) == {'Alice', 'Bob'}

    def test_in_subquery_join_result(self, db):
        """IN subquery returning join results."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE dept IN (
                SELECT departments.name FROM departments
                JOIN projects ON departments.name = projects.dept
                WHERE projects.cost > 100000
            )
        """)
        names = [row[0] for row in r.rows]
        # Projects > 100k: Beta (Engineering) -- so only Engineering employees
        assert set(names) == {'Alice', 'Bob'}


# =========================================================================
# Subqueries with Aggregates
# =========================================================================

class TestSubqueriesWithAggregates:
    def test_scalar_with_count(self, db):
        """Scalar subquery with COUNT in SELECT."""
        r = db.execute("""
            SELECT dept, COUNT(*) AS cnt,
                   (SELECT COUNT(*) FROM employees) AS total
            FROM employees GROUP BY dept
        """)
        for row in r.rows:
            assert row[2] == 6  # total always 6

    def test_scalar_sum(self, db):
        """Scalar subquery with SUM."""
        r = db.execute("SELECT (SELECT SUM(salary) FROM employees) AS total_salary")
        assert r.rows[0][0] == 90000 + 85000 + 75000 + 70000 + 80000 + 65000

    def test_subquery_in_having_with_count(self, db):
        """HAVING uses subquery to compare group count."""
        r = db.execute("""
            SELECT dept, COUNT(*) AS cnt FROM employees
            GROUP BY dept
            HAVING cnt >= (SELECT COUNT(*) FROM departments)
        """)
        # Each dept has 2 employees, 3 departments
        # So no dept has count >= 3
        assert len(r.rows) == 0


# =========================================================================
# Subqueries in UPDATE/DELETE
# =========================================================================

class TestSubqueriesInDML:
    def test_update_with_in_subquery(self, db):
        """UPDATE using IN subquery."""
        db.execute("""
            UPDATE employees SET salary = salary + 5000
            WHERE dept IN (SELECT name FROM departments WHERE budget > 400000)
        """)
        r = db.execute("SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY name")
        # Alice: 90000+5000, Bob: 85000+5000
        assert r.rows[0][1] == 95000
        assert r.rows[1][1] == 90000

    def test_delete_with_exists_subquery(self, db):
        """DELETE using EXISTS subquery."""
        db.execute("""
            DELETE FROM projects
            WHERE EXISTS (SELECT 1 FROM departments WHERE departments.name = projects.dept AND budget < 260000)
        """)
        r = db.execute("SELECT name FROM projects ORDER BY name")
        names = [row[0] for row in r.rows]
        # Sales budget 250000 < 260000, so delete Delta
        assert 'Delta' not in names
        assert 'Alpha' in names

    def test_update_with_scalar_subquery(self, db):
        """UPDATE SET using subquery comparison in WHERE."""
        db.execute("""
            UPDATE employees SET salary = salary + 10000
            WHERE salary = (SELECT MAX(salary) FROM employees)
        """)
        r = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert r.rows[0][0] == 100000

    def test_delete_with_in_subquery(self, db):
        """DELETE using IN subquery."""
        db.execute("""
            DELETE FROM employees
            WHERE id IN (SELECT manager_id FROM departments)
        """)
        r = db.execute("SELECT COUNT(*) FROM employees")
        # Deleted Alice(1), Carol(3), Eve(5) -- 3 managers
        assert r.rows[0][0] == 3


# =========================================================================
# Edge Cases
# =========================================================================

class TestEdgeCases:
    def test_subquery_no_from(self, db):
        """Subquery without FROM (just SELECT)."""
        r = db.execute("SELECT (SELECT 42) AS answer")
        assert r.rows[0][0] == 42

    def test_subquery_empty_table(self, db):
        """Subquery against empty table."""
        db.execute("CREATE TABLE empty_table (id INT)")
        r = db.execute("SELECT (SELECT COUNT(*) FROM empty_table) AS cnt")
        assert r.rows[0][0] == 0

    def test_in_subquery_empty_result(self, db):
        """IN subquery returning empty result."""
        r = db.execute("SELECT name FROM employees WHERE id IN (SELECT id FROM employees WHERE salary > 200000)")
        assert len(r.rows) == 0

    def test_exists_empty_table(self, db):
        """EXISTS on empty table."""
        db.execute("CREATE TABLE empty_table (id INT)")
        r = db.execute("SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM empty_table)")
        assert len(r.rows) == 0

    def test_not_exists_empty_table(self, db):
        """NOT EXISTS on empty table returns all rows."""
        db.execute("CREATE TABLE empty_table (id INT)")
        r = db.execute("SELECT name FROM employees WHERE NOT EXISTS (SELECT 1 FROM empty_table)")
        assert len(r.rows) == 6

    def test_scalar_subquery_null_in_agg(self, db):
        """Scalar subquery with NULL values in aggregate."""
        db.execute("CREATE TABLE nullable (val INT)")
        db.execute("INSERT INTO nullable VALUES (10)")
        db.execute("INSERT INTO nullable VALUES (NULL)")
        db.execute("INSERT INTO nullable VALUES (20)")
        r = db.execute("SELECT (SELECT AVG(val) FROM nullable) AS avg_val")
        assert r.rows[0][0] == 15.0

    def test_multiple_subqueries_in_where(self, db):
        """Multiple subqueries combined in WHERE."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE salary > (SELECT AVG(salary) FROM employees)
            AND dept IN (SELECT name FROM departments WHERE budget > 200000)
        """)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names

    def test_subquery_with_distinct(self, db):
        """Subquery with DISTINCT."""
        r = db.execute("SELECT name FROM employees WHERE dept IN (SELECT DISTINCT dept FROM projects)")
        assert len(r.rows) == 6  # all departments have projects

    def test_subquery_with_order_limit(self, db):
        """Scalar subquery with ORDER BY and LIMIT."""
        r = db.execute("""
            SELECT (SELECT salary FROM employees ORDER BY salary DESC LIMIT 1) AS top_salary
        """)
        assert r.rows[0][0] == 90000


# =========================================================================
# Non-subquery Regression Tests
# =========================================================================

class TestRegression:
    def test_basic_select(self, db):
        """Basic SELECT still works."""
        r = db.execute("SELECT name FROM employees WHERE id = 1")
        assert r.rows[0][0] == 'Alice'

    def test_regular_in_list(self, db):
        """Regular IN (val1, val2) still works."""
        r = db.execute("SELECT name FROM employees WHERE id IN (1, 2, 3)")
        assert len(r.rows) == 3

    def test_regular_not_in(self, db):
        """Regular NOT IN still works."""
        r = db.execute("SELECT name FROM employees WHERE id NOT IN (1, 2)")
        assert len(r.rows) == 4

    def test_join(self, db):
        """JOIN still works."""
        r = db.execute("""
            SELECT employees.name, departments.budget FROM employees
            JOIN departments ON employees.dept = departments.name
            WHERE employees.id = 1
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 500000

    def test_group_by_having(self, db):
        """GROUP BY with HAVING still works (uses alias -- known C247 limitation)."""
        r = db.execute("SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept HAVING cnt > 1")
        assert len(r.rows) == 3  # all depts have 2 employees

    def test_create_insert_select(self, db):
        """Basic DDL + DML still works."""
        db.execute("CREATE TABLE test (x INT)")
        db.execute("INSERT INTO test VALUES (1)")
        r = db.execute("SELECT x FROM test")
        assert r.rows[0][0] == 1

    def test_update_delete(self, db):
        """UPDATE and DELETE still work."""
        db.execute("UPDATE employees SET salary = 95000 WHERE name = 'Alice'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert r.rows[0][0] == 95000
        db.execute("DELETE FROM employees WHERE name = 'Frank'")
        r = db.execute("SELECT COUNT(*) FROM employees")
        assert r.rows[0][0] == 5

    def test_set_operations(self, db):
        """UNION/INTERSECT/EXCEPT still work."""
        r = db.execute("""
            SELECT name FROM employees WHERE dept = 'Engineering'
            UNION
            SELECT name FROM employees WHERE dept = 'Sales'
        """)
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Bob', 'Eve', 'Frank']


# =========================================================================
# Subqueries with CTEs
# =========================================================================

class TestSubqueriesWithCTEs:
    def test_cte_with_in_subquery(self, db):
        """CTE combined with IN subquery."""
        r = db.execute("""
            WITH big_depts AS (
                SELECT name FROM departments WHERE budget > 200000
            )
            SELECT name FROM employees
            WHERE dept IN (SELECT name FROM big_depts)
        """)
        assert len(r.rows) == 6  # all depts > 200k

    def test_subquery_inside_cte_body(self, db):
        """Subquery used inside a CTE body."""
        r = db.execute("""
            WITH high_salary AS (
                SELECT name, salary FROM employees
                WHERE salary > (SELECT AVG(salary) FROM employees)
            )
            SELECT name FROM high_salary ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice', 'Bob', 'Eve']


# =========================================================================
# Subqueries with Set Operations
# =========================================================================

class TestSubqueriesWithSetOps:
    def test_in_subquery_with_union(self, db):
        """IN subquery using UNION."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE id IN (
                SELECT manager_id FROM departments
                UNION
                SELECT id FROM employees WHERE salary > 85000
            )
        """)
        names = sorted([row[0] for row in r.rows])
        # Managers: 1(Alice), 3(Carol), 5(Eve)
        # salary > 85k: 1(Alice)
        # Union: 1, 3, 5
        assert names == ['Alice', 'Carol', 'Eve']


# =========================================================================
# Practical Queries
# =========================================================================

class TestPracticalQueries:
    def test_employees_above_dept_avg(self, db):
        """Find employees earning above their department average."""
        r = db.execute("""
            SELECT e.name, e.dept, e.salary
            FROM employees e
            WHERE e.salary > (
                SELECT AVG(salary) FROM employees e2 WHERE e2.dept = e.dept
            )
            ORDER BY e.name
        """)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Carol' in names
        assert 'Eve' in names

    def test_dept_with_highest_avg_salary(self, db):
        """Find department with highest average salary using CTE."""
        r = db.execute("""
            WITH dept_avgs AS (
                SELECT dept, AVG(salary) AS avg_sal FROM employees GROUP BY dept
            )
            SELECT dept, avg_sal FROM dept_avgs
            WHERE avg_sal = (SELECT MAX(avg_sal) FROM dept_avgs)
        """)
        # Engineering avg = 87500 is highest
        assert r.rows[0][0] == 'Engineering'

    def test_departments_with_projects(self, db):
        """Find departments that have at least one project."""
        r = db.execute("""
            SELECT name FROM departments d
            WHERE EXISTS (SELECT 1 FROM projects p WHERE p.dept = d.name)
            ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Engineering', 'Marketing', 'Sales']

    def test_managers_and_their_salary(self, db):
        """Find department managers with their salary."""
        r = db.execute("""
            SELECT d.name AS dept,
                   (SELECT e.name FROM employees e WHERE e.id = d.manager_id) AS manager_name,
                   (SELECT e.salary FROM employees e WHERE e.id = d.manager_id) AS manager_salary
            FROM departments d ORDER BY d.name
        """)
        for row in r.rows:
            if row[0] == 'Engineering':
                assert row[1] == 'Alice'
                assert row[2] == 90000
            elif row[0] == 'Marketing':
                assert row[1] == 'Carol'
                assert row[2] == 75000
            elif row[0] == 'Sales':
                assert row[1] == 'Eve'
                assert row[2] == 80000

    def test_employees_in_expensive_depts(self, db):
        """Find employees in departments with above-average budget."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE dept IN (
                SELECT name FROM departments
                WHERE budget > (SELECT AVG(budget) FROM departments)
            )
            ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        # avg budget = 350000, only Engineering (500k) > avg
        assert names == ['Alice', 'Bob']

    def test_project_cost_vs_dept_budget(self, db):
        """Projects that cost more than 20% of department budget."""
        r = db.execute("""
            SELECT p.name FROM projects p
            WHERE p.cost > (
                SELECT d.budget * 0.2 FROM departments d WHERE d.name = p.dept
            )
            ORDER BY p.name
        """)
        # Engineering budget*0.2 = 100000. Alpha=100000 (not >), Beta=200000 (yes)
        # Marketing budget*0.2 = 60000. Gamma=50000 (no)
        # Sales budget*0.2 = 50000. Delta=75000 (yes)
        names = [row[0] for row in r.rows]
        assert 'Beta' in names
        assert 'Delta' in names
        assert 'Alpha' not in names


# =========================================================================
# Arithmetic and Expressions in Subqueries
# =========================================================================

class TestSubqueryExpressions:
    def test_scalar_arithmetic(self, db):
        """Scalar subquery in arithmetic expression."""
        r = db.execute("""
            SELECT name, salary - (SELECT AVG(salary) FROM employees) AS diff
            FROM employees WHERE id = 1
        """)
        assert r.rows[0][0] == 'Alice'
        # 90000 - 77500 = 12500
        assert abs(r.rows[0][1] - 12500) < 1

    def test_subquery_with_literal_comparison(self, db):
        """Subquery compared with literal."""
        r = db.execute("SELECT name FROM employees WHERE (SELECT COUNT(*) FROM projects WHERE dept = employees.dept) > 1")
        names = sorted([row[0] for row in r.rows])
        # Engineering has 2 projects
        assert names == ['Alice', 'Bob']


# =========================================================================
# Case Insensitivity
# =========================================================================

class TestCaseInsensitivity:
    def test_keywords_case(self, db):
        """Keywords work in various cases."""
        r = db.execute("select name from employees where salary in (select max(salary) from employees)")
        assert r.rows[0][0] == 'Alice'

    def test_exists_case(self, db):
        """EXISTS in lowercase."""
        r = db.execute("select name from employees e where exists (select 1 from departments where manager_id = e.id)")
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Carol', 'Eve']


# =========================================================================
# Additional Coverage Tests
# =========================================================================

class TestAdditionalCoverage:
    def test_not_in_with_nulls(self, db):
        """NOT IN subquery with NULL in outer value."""
        db.execute("CREATE TABLE t (val INT)")
        db.execute("INSERT INTO t VALUES (1)")
        db.execute("INSERT INTO t VALUES (NULL)")
        r = db.execute("SELECT val FROM t WHERE val NOT IN (SELECT salary FROM employees WHERE id = 1)")
        # val=1 is not 90000, so included; val=NULL -> false for IN
        vals = [row[0] for row in r.rows]
        assert 1 in vals

    def test_exists_select_star(self, db):
        """EXISTS with SELECT *."""
        r = db.execute("SELECT name FROM employees WHERE EXISTS (SELECT * FROM departments WHERE budget > 400000)")
        assert len(r.rows) == 6  # Engineering has budget > 400k

    def test_scalar_subquery_with_where(self, db):
        """Scalar subquery with filtered WHERE."""
        r = db.execute("SELECT (SELECT salary FROM employees WHERE name = 'Bob') AS bob_salary")
        assert r.rows[0][0] == 85000

    def test_in_subquery_distinct(self, db):
        """IN subquery with DISTINCT."""
        r = db.execute("SELECT name FROM employees WHERE dept IN (SELECT DISTINCT dept FROM projects)")
        assert len(r.rows) == 6

    def test_nested_exists_in(self, db):
        """Combination of EXISTS and IN."""
        r = db.execute("""
            SELECT name FROM employees e
            WHERE EXISTS (SELECT 1 FROM departments WHERE manager_id = e.id)
            AND dept IN (SELECT name FROM departments WHERE budget > 200000)
        """)
        names = sorted([row[0] for row in r.rows])
        assert names == ['Alice', 'Carol', 'Eve']

    def test_subquery_with_between(self, db):
        """WHERE with BETWEEN still works alongside subqueries."""
        r = db.execute("SELECT name FROM employees WHERE salary BETWEEN 70000 AND (SELECT AVG(salary) FROM employees)")
        names = sorted([row[0] for row in r.rows])
        # avg = 77500, between 70000 and 77500: Carol(75000), Dave(70000)
        assert 'Carol' in names
        assert 'Dave' in names

    def test_correlated_not_exists(self, db):
        """Correlated NOT EXISTS."""
        db.execute("CREATE TABLE orders (id INT, emp_id INT, amount INT)")
        db.execute("INSERT INTO orders VALUES (1, 1, 500)")
        db.execute("INSERT INTO orders VALUES (2, 3, 300)")
        r = db.execute("""
            SELECT name FROM employees e
            WHERE NOT EXISTS (SELECT 1 FROM orders WHERE emp_id = e.id)
        """)
        names = sorted([row[0] for row in r.rows])
        # Employees without orders: Bob(2), Dave(4), Eve(5), Frank(6)
        assert 'Alice' not in names
        assert 'Carol' not in names
        assert len(names) == 4

    def test_scalar_subquery_string(self, db):
        """Scalar subquery returning string value."""
        r = db.execute("SELECT (SELECT name FROM employees WHERE id = 1) AS top_emp")
        assert r.rows[0][0] == 'Alice'

    def test_multiple_in_subqueries(self, db):
        """Multiple IN subqueries in same WHERE."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE dept IN (SELECT name FROM departments WHERE budget > 400000)
            AND id IN (SELECT manager_id FROM departments)
        """)
        # Engineering budget > 400k, managers in Engineering: Alice(1)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'

    def test_subquery_is_null(self, db):
        """IS NULL with subquery result."""
        r = db.execute("""
            SELECT name FROM employees
            WHERE (SELECT name FROM departments WHERE budget > 9999999) IS NULL
        """)
        # Subquery returns NULL (no rows), IS NULL is true for all
        assert len(r.rows) == 6

    def test_correlated_max(self, db):
        """Correlated subquery finding max salary per dept."""
        r = db.execute("""
            SELECT name, salary FROM employees e
            WHERE salary = (SELECT MAX(salary) FROM employees e2 WHERE e2.dept = e.dept)
            ORDER BY name
        """)
        names = [row[0] for row in r.rows]
        # Max per dept: Engineering=Alice(90k), Marketing=Carol(75k), Sales=Eve(80k)
        assert names == ['Alice', 'Carol', 'Eve']

    def test_subquery_in_case(self, db):
        """Subquery inside CASE expression."""
        r = db.execute("""
            SELECT name,
                   CASE WHEN salary > (SELECT AVG(salary) FROM employees) THEN 'above' ELSE 'below' END AS level
            FROM employees WHERE id = 1
        """)
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 'above'

    def test_exists_unconditional_true(self, db):
        """EXISTS with unconditional match."""
        r = db.execute("SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM departments)")
        assert len(r.rows) == 6

    def test_subquery_arithmetic_both_sides(self, db):
        """Arithmetic using two scalar subqueries."""
        r = db.execute("""
            SELECT (SELECT MAX(salary) FROM employees) - (SELECT MIN(salary) FROM employees) AS salary_range
        """)
        assert r.rows[0][0] == 90000 - 65000

    def test_in_subquery_with_limit(self, db):
        """IN subquery with LIMIT."""
        r = db.execute("SELECT name FROM employees WHERE id IN (SELECT id FROM employees ORDER BY salary DESC LIMIT 2)")
        names = sorted([row[0] for row in r.rows])
        # Top 2 salaries: Alice(90k, id=1), Bob(85k, id=2)
        assert names == ['Alice', 'Bob']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
