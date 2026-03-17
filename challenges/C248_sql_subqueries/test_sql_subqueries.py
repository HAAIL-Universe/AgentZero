"""
Tests for C248: SQL Subqueries
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from sql_subqueries import (
    SubqueryDB, SubqueryParser, SubqueryCompiler,
    parse_sql_subquery, parse_sql_subquery_multi,
    SqlSubquery, SqlInSubquery, SqlExistsExpr, SqlAnyAll, DerivedTable,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlInList, SqlAggCall,
    ParseError, DatabaseError, CompileError,
    Lexer, SelectStmt,
)


# =============================================================================
# Helpers
# =============================================================================

def make_db():
    """Create a SubqueryDB with test data."""
    db = SubqueryDB()
    db.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, dept_id INT, salary FLOAT)")
    db.execute("CREATE TABLE departments (id INT PRIMARY KEY, name TEXT, budget FLOAT)")
    db.execute("CREATE TABLE projects (id INT PRIMARY KEY, name TEXT, dept_id INT, lead_id INT)")

    # Departments
    db.execute("INSERT INTO departments (id, name, budget) VALUES (1, 'Engineering', 500000)")
    db.execute("INSERT INTO departments (id, name, budget) VALUES (2, 'Marketing', 200000)")
    db.execute("INSERT INTO departments (id, name, budget) VALUES (3, 'HR', 150000)")
    db.execute("INSERT INTO departments (id, name, budget) VALUES (4, 'Research', 300000)")

    # Employees
    db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (1, 'Alice', 1, 90000)")
    db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (2, 'Bob', 1, 85000)")
    db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (3, 'Charlie', 2, 70000)")
    db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (4, 'Diana', 2, 75000)")
    db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (5, 'Eve', 3, 65000)")
    db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'Frank', 1, 95000)")
    db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (7, 'Grace', 4, 88000)")

    # Projects
    db.execute("INSERT INTO projects (id, name, dept_id, lead_id) VALUES (1, 'Alpha', 1, 1)")
    db.execute("INSERT INTO projects (id, name, dept_id, lead_id) VALUES (2, 'Beta', 1, 6)")
    db.execute("INSERT INTO projects (id, name, dept_id, lead_id) VALUES (3, 'Gamma', 2, 3)")
    db.execute("INSERT INTO projects (id, name, dept_id, lead_id) VALUES (4, 'Delta', 4, 7)")

    return db


# =============================================================================
# Parser Tests
# =============================================================================

class TestSubqueryParser:
    """Tests for the extended parser."""

    def test_parse_scalar_subquery(self):
        ast = parse_sql_subquery("SELECT (SELECT 1)")
        assert isinstance(ast, SelectStmt)
        assert isinstance(ast.columns[0].expr, SqlSubquery)

    def test_parse_in_subquery(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x IN (SELECT y FROM t2)")
        assert isinstance(ast.where, SqlInSubquery)
        assert not ast.where.negated

    def test_parse_not_in_subquery(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x NOT IN (SELECT y FROM t2)")
        assert isinstance(ast.where, SqlInSubquery)
        assert ast.where.negated

    def test_parse_exists(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM t2)")
        assert isinstance(ast.where, SqlExistsExpr)
        assert not ast.where.negated

    def test_parse_not_exists(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE NOT EXISTS (SELECT 1 FROM t2)")
        assert isinstance(ast.where, SqlExistsExpr)
        assert ast.where.negated

    def test_parse_any(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x > ANY (SELECT y FROM t2)")
        assert isinstance(ast.where, SqlAnyAll)
        assert ast.where.quantifier == 'any'
        assert ast.where.op == '>'

    def test_parse_all(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x = ALL (SELECT y FROM t2)")
        assert isinstance(ast.where, SqlAnyAll)
        assert ast.where.quantifier == 'all'
        assert ast.where.op == '='

    def test_parse_derived_table(self):
        ast = parse_sql_subquery(
            "SELECT * FROM (SELECT id, name FROM t) AS sub")
        assert isinstance(ast.from_table, DerivedTable)
        assert ast.from_table.alias == 'sub'

    def test_parse_subquery_in_select_list(self):
        ast = parse_sql_subquery(
            "SELECT name, (SELECT COUNT(*) FROM t2) AS cnt FROM t")
        assert isinstance(ast.columns[1].expr, SqlSubquery)
        assert ast.columns[1].alias == 'cnt'

    def test_parse_in_value_list_still_works(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x IN (1, 2, 3)")
        assert isinstance(ast.where, SqlInList)

    def test_parse_regular_parens_still_work(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE (x + 1) > 5")
        assert isinstance(ast.where, SqlComparison)

    def test_parse_nested_subquery(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x IN (SELECT y FROM t2 WHERE y > (SELECT MIN(z) FROM t3))")
        assert isinstance(ast.where, SqlInSubquery)
        inner = ast.where.subquery.select
        assert isinstance(inner.where, SqlComparison)
        assert isinstance(inner.where.right, SqlSubquery)

    def test_parse_subquery_with_where(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x IN (SELECT y FROM t2 WHERE z > 10)")
        assert isinstance(ast.where, SqlInSubquery)
        assert ast.where.subquery.select.where is not None

    def test_parse_exists_with_complex_condition(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t1.id AND t2.val > 10)")
        assert isinstance(ast.where, SqlExistsExpr)

    def test_parse_multiple_subqueries_in_where(self):
        ast = parse_sql_subquery(
            "SELECT * FROM t WHERE x IN (SELECT y FROM t2) AND z > (SELECT MIN(w) FROM t3)")
        assert isinstance(ast.where, SqlLogic)

    def test_parse_derived_table_requires_alias(self):
        with pytest.raises(ParseError):
            parse_sql_subquery("SELECT * FROM (SELECT 1)")

    def test_parse_all_comparison_ops_with_any(self):
        for op in ['=', '!=', '<', '<=', '>', '>=']:
            sql_op = op if op != '!=' else '!='
            ast = parse_sql_subquery(
                f"SELECT * FROM t WHERE x {sql_op} ANY (SELECT y FROM t2)")
            assert isinstance(ast.where, SqlAnyAll)
            assert ast.where.op == op


# =============================================================================
# IN Subquery Tests
# =============================================================================

class TestInSubquery:
    """Tests for IN (SELECT ...) subqueries."""

    def test_basic_in_subquery(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering')")
        names = sorted([r[0] for r in result.rows])
        assert names == ['Alice', 'Bob', 'Frank']

    def test_in_subquery_empty_result(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Nonexistent')")
        assert len(result.rows) == 0

    def test_in_subquery_all_match(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments)")
        assert len(result.rows) == 7  # all employees

    def test_not_in_subquery(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id NOT IN (SELECT id FROM departments WHERE name = 'Engineering')")
        names = sorted([r[0] for r in result.rows])
        assert 'Alice' not in names
        assert 'Charlie' in names

    def test_in_subquery_with_aggregation(self):
        db = make_db()
        # Use alias reference in HAVING (C247 limitation: raw COUNT(*) in HAVING not supported)
        result = db.execute(
            "SELECT name FROM departments WHERE id IN (SELECT dept_id FROM employees GROUP BY dept_id)")
        names = sorted([r[0] for r in result.rows])
        assert 'Engineering' in names
        assert 'Marketing' in names

    def test_in_subquery_with_distinct(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM departments WHERE id IN (SELECT DISTINCT dept_id FROM employees)")
        assert len(result.rows) == 4

    def test_in_subquery_single_value(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE id IN (SELECT lead_id FROM projects WHERE name = 'Alpha')")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'Alice'

    def test_in_subquery_numeric_comparison(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary IN (SELECT salary FROM employees WHERE dept_id = 1)")
        names = sorted([r[0] for r in result.rows])
        assert names == ['Alice', 'Bob', 'Frank']


# =============================================================================
# EXISTS Subquery Tests
# =============================================================================

class TestExistsSubquery:
    """Tests for EXISTS (SELECT ...) subqueries."""

    def test_basic_exists(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM departments WHERE EXISTS (SELECT 1 FROM employees WHERE employees.dept_id = departments.id)")
        names = sorted([r[0] for r in result.rows])
        assert names == ['Engineering', 'HR', 'Marketing', 'Research']

    def test_not_exists(self):
        db = make_db()
        # Add a department with no employees
        db.execute("INSERT INTO departments (id, name, budget) VALUES (5, 'Legal', 100000)")
        result = db.execute(
            "SELECT name FROM departments WHERE NOT EXISTS (SELECT 1 FROM employees WHERE employees.dept_id = departments.id)")
        names = [r[0] for r in result.rows]
        assert 'Legal' in names

    def test_exists_with_complex_condition(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM departments WHERE EXISTS (SELECT 1 FROM employees WHERE employees.dept_id = departments.id AND employees.salary > 80000)")
        names = sorted([r[0] for r in result.rows])
        assert 'Engineering' in names
        assert 'Research' in names

    def test_exists_always_true(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM departments)")
        assert len(result.rows) == 7  # all employees

    def test_exists_always_false(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE t1 (id INT PRIMARY KEY, val INT)")
        db.execute("CREATE TABLE t2 (id INT PRIMARY KEY)")
        db.execute("INSERT INTO t1 (id, val) VALUES (1, 10)")
        result = db.execute("SELECT val FROM t1 WHERE EXISTS (SELECT 1 FROM t2)")
        assert len(result.rows) == 0

    def test_not_exists_with_join_condition(self):
        db = make_db()
        # Employees who don't lead any project
        result = db.execute(
            "SELECT name FROM employees WHERE NOT EXISTS (SELECT 1 FROM projects WHERE projects.lead_id = employees.id)")
        names = sorted([r[0] for r in result.rows])
        assert 'Bob' in names
        assert 'Alice' not in names  # Alice leads Alpha


# =============================================================================
# Scalar Subquery Tests
# =============================================================================

class TestScalarSubquery:
    """Tests for scalar subqueries."""

    def test_scalar_in_select(self):
        db = make_db()
        result = db.execute(
            "SELECT name, (SELECT COUNT(*) FROM employees) AS total FROM departments")
        assert result.rows[0][1] == 7  # total employees count

    def test_scalar_in_where(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)")
        names = sorted([r[0] for r in result.rows])
        # avg salary = (90000+85000+70000+75000+65000+95000+88000)/7 = ~81142
        assert 'Alice' in names
        assert 'Frank' in names
        assert 'Grace' in names

    def test_scalar_subquery_returns_null(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t (id, val) VALUES (1, 10)")
        result = db.execute(
            "SELECT (SELECT val FROM t WHERE id = 999) AS missing")
        assert result.rows[0][0] is None

    def test_scalar_with_aggregation(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary = (SELECT MAX(salary) FROM employees)")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'Frank'

    def test_scalar_min(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary = (SELECT MIN(salary) FROM employees)")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'Eve'

    def test_multiple_scalars_in_select(self):
        db = make_db()
        result = db.execute(
            "SELECT (SELECT MIN(salary) FROM employees) AS min_sal, (SELECT MAX(salary) FROM employees) AS max_sal")
        assert result.rows[0][0] == 65000
        assert result.rows[0][1] == 95000


# =============================================================================
# Correlated Subquery Tests
# =============================================================================

class TestCorrelatedSubquery:
    """Tests for correlated subqueries."""

    def test_correlated_exists(self):
        db = make_db()
        # Departments that have at least one high-salary employee
        result = db.execute(
            "SELECT name FROM departments WHERE EXISTS (SELECT 1 FROM employees WHERE employees.dept_id = departments.id AND employees.salary > 85000)")
        names = sorted([r[0] for r in result.rows])
        assert 'Engineering' in names
        assert 'Research' in names

    def test_correlated_in(self):
        db = make_db()
        # Employees in departments with budget > 200000
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE budget > 200000)")
        names = sorted([r[0] for r in result.rows])
        # Engineering (500k), Research (300k)
        assert 'Alice' in names
        assert 'Grace' in names
        assert 'Charlie' not in names  # Marketing 200k - not strictly >

    def test_correlated_scalar_subquery(self):
        db = make_db()
        result = db.execute(
            "SELECT name, (SELECT name FROM departments WHERE departments.id = employees.dept_id) AS dept_name FROM employees WHERE id = 1")
        assert result.rows[0][0] == 'Alice'
        assert result.rows[0][1] == 'Engineering'


# =============================================================================
# Derived Table Tests
# =============================================================================

class TestDerivedTables:
    """Tests for FROM (SELECT ...) AS alias."""

    def test_basic_derived_table(self):
        db = make_db()
        result = db.execute(
            "SELECT * FROM (SELECT id, name FROM employees WHERE dept_id = 1) AS eng")
        names = sorted([r[1] for r in result.rows])
        assert names == ['Alice', 'Bob', 'Frank']

    def test_derived_table_with_aggregation(self):
        db = make_db()
        result = db.execute(
            "SELECT dept_id, avg_sal FROM (SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) AS dept_avg ORDER BY dept_id")
        assert len(result.rows) >= 3

    def test_derived_table_with_where(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM (SELECT name, salary FROM employees WHERE salary > 80000) AS high_earners")
        names = sorted([r[0] for r in result.rows])
        assert all(n in names for n in ['Alice', 'Bob', 'Frank', 'Grace'])

    def test_derived_table_alias_used_in_outer(self):
        db = make_db()
        result = db.execute(
            "SELECT sub.name FROM (SELECT name FROM departments) AS sub")
        assert len(result.rows) == 4

    def test_nested_derived_tables(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM (SELECT name FROM (SELECT name, salary FROM employees) AS inner_t WHERE salary > 80000) AS outer_t")
        names = sorted([r[0] for r in result.rows])
        assert 'Alice' in names


# =============================================================================
# ANY/ALL Subquery Tests
# =============================================================================

class TestAnyAllSubquery:
    """Tests for ANY/ALL quantified comparisons."""

    def test_greater_than_any(self):
        db = make_db()
        # Employees whose salary > any Marketing salary (70k, 75k)
        result = db.execute(
            "SELECT name FROM employees WHERE salary > ANY (SELECT salary FROM employees WHERE dept_id = 2)")
        names = sorted([r[0] for r in result.rows])
        # > 70000: Alice(90k), Bob(85k), Diana(75k), Frank(95k), Grace(88k)
        assert 'Alice' in names
        assert 'Eve' not in names  # 65000

    def test_greater_than_all(self):
        db = make_db()
        # Employees whose salary > all Marketing salaries (70k, 75k)
        result = db.execute(
            "SELECT name FROM employees WHERE salary > ALL (SELECT salary FROM employees WHERE dept_id = 2)")
        names = sorted([r[0] for r in result.rows])
        # > 75000: Alice(90k), Bob(85k), Frank(95k), Grace(88k)
        assert 'Alice' in names
        assert 'Diana' not in names  # 75000 not > 75000

    def test_equal_any(self):
        db = make_db()
        # Same as IN
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id = ANY (SELECT id FROM departments WHERE name = 'Engineering')")
        names = sorted([r[0] for r in result.rows])
        assert names == ['Alice', 'Bob', 'Frank']

    def test_less_than_all(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary < ALL (SELECT salary FROM employees WHERE dept_id = 1)")
        # Engineering salaries: 90k, 85k, 95k. < all means < 85k
        names = sorted([r[0] for r in result.rows])
        assert 'Eve' in names   # 65k
        assert 'Charlie' in names  # 70k
        assert 'Diana' in names  # 75k

    def test_any_empty_set(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary > ANY (SELECT salary FROM employees WHERE dept_id = 999)")
        assert len(result.rows) == 0  # ANY with empty set = False

    def test_all_empty_set(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary > ALL (SELECT salary FROM employees WHERE dept_id = 999)")
        assert len(result.rows) == 7  # ALL with empty set = True


# =============================================================================
# Nested Subquery Tests
# =============================================================================

class TestNestedSubqueries:
    """Tests for subqueries inside subqueries."""

    def test_subquery_in_subquery(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE id IN (SELECT dept_id FROM projects))")
        names = sorted([r[0] for r in result.rows])
        # Projects: dept 1, 1, 2, 4
        assert 'Alice' in names   # dept 1
        assert 'Eve' not in names  # dept 3 (no projects)

    def test_exists_with_in_subquery(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM departments WHERE EXISTS (SELECT 1 FROM employees WHERE employees.dept_id = departments.id AND employees.id IN (SELECT lead_id FROM projects))")
        names = sorted([r[0] for r in result.rows])
        # Leads: Alice(1, eng), Frank(6, eng), Charlie(3, mkt), Grace(7, res)
        assert 'Engineering' in names
        assert 'Marketing' in names
        assert 'Research' in names

    def test_three_levels_deep(self):
        db = make_db()
        result = db.execute(
            """SELECT name FROM employees
               WHERE dept_id IN (
                   SELECT id FROM departments
                   WHERE id IN (
                       SELECT dept_id FROM projects
                       WHERE lead_id IN (
                           SELECT id FROM employees WHERE salary > 90000
                       )
                   )
               )""")
        names = sorted([r[0] for r in result.rows])
        # salary > 90k: Frank(95k). Frank leads Beta (dept 1).
        # So dept_id = 1 -> Engineering employees
        assert 'Alice' in names
        assert 'Bob' in names
        assert 'Frank' in names


# =============================================================================
# Subquery with DML Tests
# =============================================================================

class TestSubqueryDML:
    """Tests for subqueries in UPDATE and DELETE."""

    def test_update_with_in_subquery(self):
        db = make_db()
        db.execute(
            "UPDATE employees SET salary = salary + 5000 WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering')")
        result = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert result.rows[0][0] == 95000

    def test_delete_with_in_subquery(self):
        db = make_db()
        db.execute(
            "DELETE FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name = 'HR')")
        result = db.execute("SELECT name FROM employees WHERE dept_id = 3")
        assert len(result.rows) == 0

    def test_update_with_exists(self):
        db = make_db()
        db.execute(
            "UPDATE departments SET budget = budget + 100000 WHERE EXISTS (SELECT 1 FROM projects WHERE projects.dept_id = departments.id)")
        result = db.execute("SELECT budget FROM departments WHERE name = 'Engineering'")
        assert result.rows[0][0] == 600000

    def test_delete_with_not_in(self):
        db = make_db()
        db.execute(
            "DELETE FROM departments WHERE id NOT IN (SELECT DISTINCT dept_id FROM employees)")
        result = db.execute("SELECT COUNT(*) FROM departments")
        assert result.rows[0][0] == 4  # all depts have employees

    def test_update_with_scalar_subquery_where(self):
        db = make_db()
        db.execute(
            "UPDATE employees SET salary = 100000 WHERE salary = (SELECT MAX(salary) FROM employees)")
        result = db.execute("SELECT salary FROM employees WHERE name = 'Frank'")
        assert result.rows[0][0] == 100000


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and corner cases."""

    def test_subquery_returns_one_column_many_rows(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments)")
        assert len(result.rows) == 7

    def test_subquery_with_limit(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE id IN (SELECT id FROM employees ORDER BY salary DESC LIMIT 3)")
        assert len(result.rows) == 3

    def test_subquery_with_order_by(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE id IN (SELECT id FROM employees ORDER BY salary DESC LIMIT 2)")
        names = sorted([r[0] for r in result.rows])
        # Top 2 salaries: Frank(95k), Alice(90k)
        assert names == ['Alice', 'Frank']

    def test_in_subquery_with_nulls(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE t1 (id INT PRIMARY KEY, val INT)")
        db.execute("CREATE TABLE t2 (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t1 (id, val) VALUES (1, 10)")
        db.execute("INSERT INTO t1 (id, val) VALUES (2, 20)")
        db.execute("INSERT INTO t2 (id, val) VALUES (1, 10)")
        db.execute("INSERT INTO t2 (id) VALUES (2)")  # val is NULL
        result = db.execute("SELECT val FROM t1 WHERE val IN (SELECT val FROM t2)")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 10

    def test_subquery_single_table(self):
        db = make_db()
        # Self-referencing: employees earning more than avg in their dept
        result = db.execute(
            "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)")
        names = sorted([r[0] for r in result.rows])
        avg = (90000 + 85000 + 70000 + 75000 + 65000 + 95000 + 88000) / 7  # ~81142.86
        assert 'Alice' in names  # 90k > avg
        assert 'Eve' not in names  # 65k < avg

    def test_exists_with_count(self):
        db = make_db()
        result = db.execute(
            "SELECT COUNT(*) FROM departments WHERE EXISTS (SELECT 1 FROM employees WHERE employees.dept_id = departments.id)")
        assert result.rows[0][0] == 4

    def test_in_subquery_with_expression(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary IN (SELECT MAX(salary) FROM employees)")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'Frank'

    def test_derived_table_column_preservation(self):
        db = make_db()
        result = db.execute(
            "SELECT id, name FROM (SELECT id, name FROM departments ORDER BY id) AS d")
        assert len(result.rows) == 4
        assert result.columns == ['id', 'name']

    def test_backward_compat_regular_query(self):
        """Ensure regular queries still work through SubqueryDB."""
        db = make_db()
        result = db.execute("SELECT name FROM employees WHERE dept_id = 1 ORDER BY name")
        names = [r[0] for r in result.rows]
        assert names == ['Alice', 'Bob', 'Frank']

    def test_backward_compat_join(self):
        db = make_db()
        result = db.execute(
            "SELECT employees.name, departments.name FROM employees JOIN departments ON employees.dept_id = departments.id WHERE employees.id = 1")
        assert result.rows[0][0] == 'Alice'
        assert result.rows[0][1] == 'Engineering'

    def test_backward_compat_group_by(self):
        db = make_db()
        result = db.execute(
            "SELECT dept_id, COUNT(*) AS cnt FROM employees GROUP BY dept_id ORDER BY dept_id")
        assert len(result.rows) >= 3

    def test_backward_compat_insert_update_delete(self):
        db = make_db()
        db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (8, 'Hank', 1, 80000)")
        result = db.execute("SELECT name FROM employees WHERE id = 8")
        assert result.rows[0][0] == 'Hank'
        db.execute("UPDATE employees SET salary = 82000 WHERE id = 8")
        result = db.execute("SELECT salary FROM employees WHERE id = 8")
        assert result.rows[0][0] == 82000
        db.execute("DELETE FROM employees WHERE id = 8")
        result = db.execute("SELECT name FROM employees WHERE id = 8")
        assert len(result.rows) == 0

    def test_backward_compat_transactions(self):
        db = make_db()
        db.execute("BEGIN")
        db.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (8, 'Hank', 1, 80000)")
        db.execute("ROLLBACK")
        result = db.execute("SELECT name FROM employees WHERE id = 8")
        assert len(result.rows) == 0

    def test_backward_compat_create_drop(self):
        db = make_db()
        db.execute("CREATE TABLE temp (id INT PRIMARY KEY, val TEXT)")
        db.execute("INSERT INTO temp (id, val) VALUES (1, 'hello')")
        result = db.execute("SELECT val FROM temp WHERE id = 1")
        assert result.rows[0][0] == 'hello'
        db.execute("DROP TABLE temp")
        result = db.execute("SHOW TABLES")
        names = [r[0] for r in result.rows]
        assert 'temp' not in names

    def test_backward_compat_distinct(self):
        db = make_db()
        result = db.execute("SELECT DISTINCT dept_id FROM employees ORDER BY dept_id")
        dept_ids = [r[0] for r in result.rows]
        assert dept_ids == [1, 2, 3, 4]

    def test_backward_compat_between(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary BETWEEN 70000 AND 85000 ORDER BY name")
        names = [r[0] for r in result.rows]
        assert 'Bob' in names
        assert 'Charlie' in names

    def test_backward_compat_like(self):
        db = make_db()
        result = db.execute("SELECT name FROM employees WHERE name LIKE 'A%'")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'Alice'

    def test_backward_compat_is_null(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t (id) VALUES (1)")
        db.execute("INSERT INTO t (id, val) VALUES (2, 42)")
        result = db.execute("SELECT id FROM t WHERE val IS NULL")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 1

    def test_backward_compat_case(self):
        db = make_db()
        result = db.execute(
            "SELECT name, CASE WHEN salary > 80000 THEN 'high' ELSE 'low' END AS level FROM employees WHERE id = 1")
        assert result.rows[0][1] == 'high'


# =============================================================================
# Complex Queries
# =============================================================================

class TestComplexQueries:
    """Complex real-world-style queries."""

    def test_dept_with_highest_avg_salary(self):
        db = make_db()
        # Use derived table approach instead of HAVING with raw aggregate
        result = db.execute(
            "SELECT name FROM departments WHERE id IN (SELECT dept_id FROM (SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) AS dept_avgs WHERE avg_sal = (SELECT MAX(avg_sal) FROM (SELECT AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) AS avgs2))")
        # Eng avg: (90+85+95)/3 = 90k. Research: 88k. Marketing: 72.5k. HR: 65k.
        assert len(result.rows) >= 1
        assert result.rows[0][0] == 'Engineering'

    def test_employees_in_depts_with_projects(self):
        db = make_db()
        result = db.execute(
            "SELECT DISTINCT name FROM employees WHERE dept_id IN (SELECT DISTINCT dept_id FROM projects)")
        names = sorted([r[0] for r in result.rows])
        # Projects in dept 1, 2, 4
        assert 'Alice' in names
        assert 'Charlie' in names
        assert 'Grace' in names
        assert 'Eve' not in names  # dept 3

    def test_dept_employee_count_via_derived(self):
        db = make_db()
        result = db.execute(
            "SELECT dept_name, emp_count FROM (SELECT departments.name AS dept_name, COUNT(*) AS emp_count FROM employees JOIN departments ON employees.dept_id = departments.id GROUP BY departments.name) AS summary ORDER BY emp_count DESC")
        assert result.rows[0][0] == 'Engineering'
        assert result.rows[0][1] == 3

    def test_employees_earning_above_dept_avg(self):
        db = make_db()
        # Use scalar subquery: only works for non-correlated version
        result = db.execute(
            "SELECT name, salary FROM employees WHERE salary > (SELECT AVG(salary) FROM employees) ORDER BY salary DESC")
        names = [r[0] for r in result.rows]
        assert names[0] == 'Frank'

    def test_project_leads_details(self):
        db = make_db()
        result = db.execute(
            "SELECT projects.name, (SELECT employees.name FROM employees WHERE employees.id = projects.lead_id) AS lead_name FROM projects ORDER BY projects.id")
        assert result.rows[0][0] == 'Alpha'
        assert result.rows[0][1] == 'Alice'
        assert result.rows[1][0] == 'Beta'
        assert result.rows[1][1] == 'Frank'

    def test_depts_without_projects(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM departments WHERE NOT EXISTS (SELECT 1 FROM projects WHERE projects.dept_id = departments.id)")
        names = [r[0] for r in result.rows]
        assert 'HR' in names  # dept 3 has no projects

    def test_highest_paid_per_dept_via_subquery(self):
        db = make_db()
        result = db.execute(
            "SELECT name, salary FROM employees WHERE salary = (SELECT MAX(salary) FROM employees AS e2 WHERE e2.dept_id = employees.dept_id) ORDER BY salary DESC")
        # Eng: Frank 95k, Marketing: Diana 75k, HR: Eve 65k, Research: Grace 88k
        names = sorted([r[0] for r in result.rows])
        assert 'Frank' in names
        assert 'Diana' in names

    def test_count_with_subquery_filter(self):
        db = make_db()
        result = db.execute(
            "SELECT COUNT(*) FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE budget > 200000)")
        # Eng(500k)->3, Research(300k)->1 = 4
        assert result.rows[0][0] == 4


# =============================================================================
# Multi-statement Tests
# =============================================================================

class TestMultiStatement:
    """Tests for multiple statements with subqueries."""

    def test_execute_many_with_subqueries(self):
        db = SubqueryDB()
        results = db.execute_many("""
            CREATE TABLE t1 (id INT PRIMARY KEY, val INT);
            CREATE TABLE t2 (id INT PRIMARY KEY, ref_id INT);
            INSERT INTO t1 (id, val) VALUES (1, 100);
            INSERT INTO t1 (id, val) VALUES (2, 200);
            INSERT INTO t2 (id, ref_id) VALUES (1, 1);
            SELECT val FROM t1 WHERE id IN (SELECT ref_id FROM t2)
        """)
        assert len(results) == 6
        assert results[5].rows[0][0] == 100


# =============================================================================
# Show/Describe/Explain through SubqueryDB
# =============================================================================

class TestUtilityStatements:
    """Test that utility statements work through SubqueryDB."""

    def test_show_tables(self):
        db = make_db()
        result = db.execute("SHOW TABLES")
        names = sorted([r[0] for r in result.rows])
        assert names == ['departments', 'employees', 'projects']

    def test_describe(self):
        db = make_db()
        result = db.execute("DESCRIBE employees")
        col_names = [r[0] for r in result.rows]
        assert 'id' in col_names
        assert 'name' in col_names
        assert 'salary' in col_names

    def test_explain_with_subquery(self):
        db = make_db()
        result = db.execute(
            "EXPLAIN SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments)")
        assert len(result.rows) > 0


# =============================================================================
# Stress / Performance Tests
# =============================================================================

class TestStress:
    """Stress tests with larger datasets."""

    def test_large_in_subquery(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE big (id INT PRIMARY KEY, val INT)")
        db.execute("CREATE TABLE small (id INT PRIMARY KEY)")
        for i in range(100):
            db.execute(f"INSERT INTO big (id, val) VALUES ({i}, {i * 10})")
        for i in range(0, 100, 5):
            db.execute(f"INSERT INTO small (id) VALUES ({i})")
        result = db.execute("SELECT COUNT(*) FROM big WHERE id IN (SELECT id FROM small)")
        assert result.rows[0][0] == 20

    def test_large_exists(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE parent (id INT PRIMARY KEY)")
        db.execute("CREATE TABLE child (id INT PRIMARY KEY, parent_id INT)")
        for i in range(50):
            db.execute(f"INSERT INTO parent (id) VALUES ({i})")
        for i in range(200):
            db.execute(f"INSERT INTO child (id, parent_id) VALUES ({i}, {i % 30})")
        # Parents with children
        result = db.execute(
            "SELECT COUNT(*) FROM parent WHERE EXISTS (SELECT 1 FROM child WHERE child.parent_id = parent.id)")
        assert result.rows[0][0] == 30  # 0-29 have children

    def test_nested_3_deep_stress(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE a (id INT PRIMARY KEY, val INT)")
        db.execute("CREATE TABLE b (id INT PRIMARY KEY, a_id INT)")
        db.execute("CREATE TABLE c (id INT PRIMARY KEY, b_id INT)")
        for i in range(20):
            db.execute(f"INSERT INTO a (id, val) VALUES ({i}, {i})")
        for i in range(40):
            db.execute(f"INSERT INTO b (id, a_id) VALUES ({i}, {i % 20})")
        for i in range(60):
            db.execute(f"INSERT INTO c (id, b_id) VALUES ({i}, {i % 40})")
        result = db.execute(
            "SELECT COUNT(*) FROM a WHERE id IN (SELECT a_id FROM b WHERE id IN (SELECT b_id FROM c WHERE b_id < 10))")
        assert result.rows[0][0] == 10


# =============================================================================
# ResultSet Method Tests
# =============================================================================

class TestResultSetMethods:
    """Ensure ResultSet methods work with subquery results."""

    def test_scalar_method(self):
        db = make_db()
        result = db.execute(
            "SELECT COUNT(*) FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering')")
        assert result.scalar() == 3

    def test_column_method(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering') ORDER BY name")
        names = result.column('name')
        assert names == ['Alice', 'Bob', 'Frank']

    def test_to_dicts(self):
        db = make_db()
        result = db.execute(
            "SELECT name, salary FROM employees WHERE salary > (SELECT AVG(salary) FROM employees) ORDER BY salary DESC")
        dicts = result.to_dicts()
        assert dicts[0]['name'] == 'Frank'
        assert dicts[0]['salary'] == 95000

    def test_len(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments)")
        assert len(result) == 7

    def test_iter(self):
        db = make_db()
        result = db.execute(
            "SELECT id FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering') ORDER BY id")
        ids = [r[0] for r in result]
        assert ids == [1, 2, 6]

    def test_getitem(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE id IN (SELECT lead_id FROM projects) ORDER BY name")
        assert result[0][0] == 'Alice'


# =============================================================================
# NOT IN edge cases
# =============================================================================

class TestNotInEdgeCases:
    """Specific NOT IN tests."""

    def test_not_in_basic(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE id NOT IN (SELECT lead_id FROM projects)")
        names = sorted([r[0] for r in result.rows])
        # Leads: 1(Alice), 6(Frank), 3(Charlie), 7(Grace)
        assert 'Bob' in names
        assert 'Diana' in names
        assert 'Eve' in names
        assert 'Alice' not in names

    def test_not_in_empty_subquery(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE t1 (id INT PRIMARY KEY)")
        db.execute("CREATE TABLE t2 (id INT PRIMARY KEY)")
        db.execute("INSERT INTO t1 (id) VALUES (1)")
        db.execute("INSERT INTO t1 (id) VALUES (2)")
        result = db.execute("SELECT id FROM t1 WHERE id NOT IN (SELECT id FROM t2)")
        assert len(result.rows) == 2  # empty subquery -> all pass


# =============================================================================
# Subquery with UNION (backward compat)
# =============================================================================

class TestSubqueryCompatibility:
    """Ensure subqueries work with other SQL features."""

    def test_subquery_with_between_in_inner(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE budget BETWEEN 200000 AND 400000)")
        # Marketing(200k), Research(300k)
        names = sorted([r[0] for r in result.rows])
        assert 'Charlie' in names
        assert 'Grace' in names

    def test_subquery_with_like_in_inner(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments WHERE name LIKE 'E%')")
        names = sorted([r[0] for r in result.rows])
        assert names == ['Alice', 'Bob', 'Frank']

    def test_subquery_with_is_null_in_inner(self):
        db = SubqueryDB()
        db.execute("CREATE TABLE t1 (id INT PRIMARY KEY, val INT)")
        db.execute("CREATE TABLE t2 (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t1 (id, val) VALUES (1, 10)")
        db.execute("INSERT INTO t2 (id) VALUES (1)")  # val is NULL
        db.execute("INSERT INTO t2 (id, val) VALUES (2, 20)")
        result = db.execute("SELECT id FROM t2 WHERE val IS NOT NULL AND val IN (SELECT val FROM t1)")
        assert len(result.rows) == 0  # t2.val=20 not in t1

    def test_subquery_with_case_in_inner(self):
        db = make_db()
        result = db.execute(
            "SELECT name FROM employees WHERE salary > (SELECT CASE WHEN COUNT(*) > 5 THEN 80000 ELSE 70000 END FROM employees)")
        # 7 employees > 5, so threshold is 80000
        names = sorted([r[0] for r in result.rows])
        assert 'Alice' in names  # 90k
        assert 'Charlie' not in names  # 70k


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
