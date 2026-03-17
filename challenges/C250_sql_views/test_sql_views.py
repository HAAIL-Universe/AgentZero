"""
Tests for C250: SQL Views
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C249_stored_procedures'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from sql_views import (
    ViewDB, ViewCatalog, ViewDefinition, ViewParser, ViewLexer,
    CreateViewStmt, DropViewStmt, ShowViewsStmt,
    CheckOption, DatabaseError, ParseError,
)


@pytest.fixture
def db():
    d = ViewDB(pool_size=32)
    d.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, dept TEXT, salary INT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 'eng', 100)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 'eng', 120)")
    d.execute("INSERT INTO employees VALUES (3, 'Carol', 'sales', 90)")
    d.execute("INSERT INTO employees VALUES (4, 'Dave', 'sales', 80)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 'hr', 95)")
    return d


@pytest.fixture
def db_multi(db):
    """DB with multiple tables for join tests."""
    db.execute("CREATE TABLE departments (id INT PRIMARY KEY, name TEXT, budget INT)")
    db.execute("INSERT INTO departments VALUES (1, 'eng', 500)")
    db.execute("INSERT INTO departments VALUES (2, 'sales', 300)")
    db.execute("INSERT INTO departments VALUES (3, 'hr', 200)")
    return db


# =============================================================================
# 1. CREATE VIEW Basics
# =============================================================================

class TestCreateView:
    def test_create_simple_view(self, db):
        r = db.execute("CREATE VIEW eng_team AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        assert 'CREATE VIEW' in r.message

    def test_create_view_select_star(self, db):
        db.execute("CREATE VIEW all_employees AS SELECT * FROM employees")
        r = db.execute("SELECT * FROM all_employees")
        assert len(r.rows) == 5

    def test_create_view_with_column_aliases(self, db):
        db.execute("CREATE VIEW emp_names(employee_name, department) AS SELECT name, dept FROM employees")
        r = db.execute("SELECT employee_name FROM emp_names")
        assert len(r.rows) == 5
        names = [row[0] for row in r.rows]
        assert 'Alice' in names

    def test_create_view_already_exists(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        with pytest.raises(DatabaseError, match="already exists"):
            db.execute("CREATE VIEW v1 AS SELECT name FROM employees")

    def test_create_or_replace_view(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        db.execute("CREATE OR REPLACE VIEW v1 AS SELECT name, salary FROM employees")
        r = db.execute("SELECT * FROM v1")
        assert len(r.columns) == 2

    def test_create_view_wrong_column_count(self, db):
        with pytest.raises(DatabaseError, match="column names"):
            db.execute("CREATE VIEW v1(a, b, c) AS SELECT name FROM employees")

    def test_create_view_nonexistent_table(self, db):
        with pytest.raises(DatabaseError, match="does not exist"):
            db.execute("CREATE VIEW v1 AS SELECT * FROM nonexistent")

    def test_create_view_with_expressions(self, db):
        db.execute("CREATE VIEW salary_doubled AS SELECT name, salary * 2 AS doubled FROM employees")
        r = db.execute("SELECT * FROM salary_doubled")
        assert len(r.rows) == 5

    def test_create_view_with_where(self, db):
        db.execute("CREATE VIEW high_salary AS SELECT * FROM employees WHERE salary > 95")
        r = db.execute("SELECT * FROM high_salary")
        assert all(row[3] > 95 for row in r.rows)

    def test_create_view_with_order(self, db):
        db.execute("CREATE VIEW ordered_emp AS SELECT name, salary FROM employees ORDER BY salary DESC")
        r = db.execute("SELECT * FROM ordered_emp")
        salaries = [row[1] for row in r.rows]
        assert salaries == sorted(salaries, reverse=True)


# =============================================================================
# 2. DROP VIEW
# =============================================================================

class TestDropView:
    def test_drop_view(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        r = db.execute("DROP VIEW v1")
        assert 'DROP VIEW' in r.message

    def test_drop_nonexistent_view(self, db):
        with pytest.raises(DatabaseError, match="does not exist"):
            db.execute("DROP VIEW nonexistent")

    def test_drop_view_if_exists(self, db):
        r = db.execute("DROP VIEW IF EXISTS nonexistent")
        assert 'DROP VIEW' in r.message

    def test_drop_view_then_create(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        db.execute("DROP VIEW v1")
        db.execute("CREATE VIEW v1 AS SELECT salary FROM employees")
        r = db.execute("SELECT * FROM v1")
        assert r.columns == ['salary']


# =============================================================================
# 3. SHOW VIEWS
# =============================================================================

class TestShowViews:
    def test_show_views_empty(self, db):
        r = db.execute("SHOW VIEWS")
        assert r.columns == ['view_name']
        assert len(r.rows) == 0

    def test_show_views_with_views(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        db.execute("CREATE VIEW v2 AS SELECT salary FROM employees")
        r = db.execute("SHOW VIEWS")
        names = [row[0] for row in r.rows]
        assert 'v1' in names
        assert 'v2' in names

    def test_show_views_after_drop(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        db.execute("CREATE VIEW v2 AS SELECT salary FROM employees")
        db.execute("DROP VIEW v1")
        r = db.execute("SHOW VIEWS")
        names = [row[0] for row in r.rows]
        assert 'v1' not in names
        assert 'v2' in names


# =============================================================================
# 4. SELECT Through Views
# =============================================================================

class TestSelectThroughView:
    def test_select_all_from_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        r = db.execute("SELECT * FROM eng")
        assert len(r.rows) == 2
        names = {row[0] for row in r.rows}
        assert names == {'Alice', 'Bob'}

    def test_select_specific_columns(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        r = db.execute("SELECT name FROM eng")
        assert len(r.rows) == 2

    def test_select_with_additional_where(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        r = db.execute("SELECT name FROM eng WHERE salary > 100")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Bob'

    def test_select_with_order_by(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        r = db.execute("SELECT name FROM eng ORDER BY salary DESC")
        assert r.rows[0][0] == 'Bob'

    def test_select_with_limit(self, db):
        db.execute("CREATE VIEW all_emp AS SELECT name FROM employees")
        r = db.execute("SELECT * FROM all_emp LIMIT 3")
        assert len(r.rows) == 3

    def test_select_count_from_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        r = db.execute("SELECT COUNT(*) AS cnt FROM eng")
        assert r.rows[0][0] == 2

    def test_select_aggregate_from_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        r = db.execute("SELECT SUM(salary) AS total FROM eng")
        assert r.rows[0][0] == 220

    def test_view_with_column_aliases_select(self, db):
        db.execute("CREATE VIEW emp_v(emp_name, emp_sal) AS SELECT name, salary FROM employees")
        r = db.execute("SELECT emp_name FROM emp_v WHERE emp_sal > 100")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Bob'

    def test_select_star_preserves_column_names(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name, dept FROM employees")
        r = db.execute("SELECT * FROM v1")
        assert 'name' in r.columns
        assert 'dept' in r.columns

    def test_view_distinct(self, db):
        db.execute("CREATE VIEW depts AS SELECT DISTINCT dept FROM employees")
        r = db.execute("SELECT * FROM depts")
        assert len(r.rows) == 3

    def test_view_with_offset(self, db):
        db.execute("CREATE VIEW all_emp AS SELECT name FROM employees ORDER BY name ASC")
        r = db.execute("SELECT * FROM all_emp LIMIT 2 OFFSET 1")
        assert len(r.rows) == 2


# =============================================================================
# 5. Nested Views
# =============================================================================

class TestNestedViews:
    def test_view_over_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        db.execute("CREATE VIEW high_eng AS SELECT name FROM eng WHERE salary > 100")
        r = db.execute("SELECT * FROM high_eng")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Bob'

    def test_three_level_nesting(self, db):
        db.execute("CREATE VIEW v1 AS SELECT * FROM employees WHERE dept = 'eng'")
        db.execute("CREATE VIEW v2 AS SELECT name, salary FROM v1 WHERE salary > 100")
        db.execute("CREATE VIEW v3 AS SELECT name FROM v2")
        r = db.execute("SELECT * FROM v3")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Bob'

    def test_circular_view_detection(self, db):
        # We can't create truly circular views since validation checks refs exist,
        # but we can test the expansion detection
        db.execute("CREATE VIEW v1 AS SELECT * FROM employees")
        # Replace v1 to reference itself would need to bypass validation
        # Instead test that the catalog detects it
        catalog = db.view_catalog
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlStar, ViewDefinition
        # Manually create a circular reference
        q = SelectStmt(columns=[SelectExpr(expr=SqlStar(table=None))],
                       from_table=TableRef(table_name='v_self'))
        vdef = ViewDefinition(name='v_self', columns=None, query=q)
        catalog.views['v_self'] = vdef
        with pytest.raises(DatabaseError, match="Circular"):
            db.execute("SELECT * FROM v_self")


# =============================================================================
# 6. Updatable Views -- INSERT
# =============================================================================

class TestInsertThroughView:
    def test_insert_through_simple_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'eng'")
        db.execute("INSERT INTO eng VALUES (6, 'Frank', 'eng', 110)")
        r = db.execute("SELECT * FROM employees WHERE id = 6")
        assert len(r.rows) == 1
        assert r.rows[0][1] == 'Frank'

    def test_insert_through_view_with_columns(self, db):
        db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'eng'")
        db.execute("INSERT INTO eng (id, name, dept, salary) VALUES (7, 'Grace', 'eng', 130)")
        r = db.execute("SELECT name FROM employees WHERE id = 7")
        assert r.rows[0][0] == 'Grace'

    def test_insert_through_view_with_aliases(self, db):
        db.execute("CREATE VIEW emp_v(eid, ename, edept, esal) AS SELECT id, name, dept, salary FROM employees")
        db.execute("INSERT INTO emp_v (eid, ename, edept, esal) VALUES (8, 'Heidi', 'hr', 105)")
        r = db.execute("SELECT name FROM employees WHERE id = 8")
        assert r.rows[0][0] == 'Heidi'

    def test_insert_non_updatable_view_fails(self, db):
        db.execute("CREATE VIEW dept_counts AS SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept")
        with pytest.raises(DatabaseError, match="not updatable"):
            db.execute("INSERT INTO dept_counts VALUES ('marketing', 1)")


# =============================================================================
# 7. Updatable Views -- UPDATE
# =============================================================================

class TestUpdateThroughView:
    def test_update_through_simple_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'eng'")
        db.execute("UPDATE eng SET salary = 150 WHERE name = 'Alice'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert r.rows[0][0] == 150

    def test_update_only_affects_view_rows(self, db):
        db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'eng'")
        db.execute("UPDATE eng SET salary = 200")
        # Only eng employees should be updated
        r = db.execute("SELECT salary FROM employees WHERE dept = 'eng'")
        assert all(row[0] == 200 for row in r.rows)
        r2 = db.execute("SELECT salary FROM employees WHERE dept = 'sales'")
        assert all(row[0] != 200 for row in r2.rows)

    def test_update_through_aliased_view(self, db):
        db.execute("CREATE VIEW emp_v(eid, ename, edept, esal) AS SELECT id, name, dept, salary FROM employees")
        db.execute("UPDATE emp_v SET esal = 999 WHERE ename = 'Carol'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Carol'")
        assert r.rows[0][0] == 999


# =============================================================================
# 8. Updatable Views -- DELETE
# =============================================================================

class TestDeleteThroughView:
    def test_delete_through_simple_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'eng'")
        db.execute("DELETE FROM eng WHERE name = 'Alice'")
        r = db.execute("SELECT * FROM employees WHERE name = 'Alice'")
        assert len(r.rows) == 0

    def test_delete_all_from_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'eng'")
        db.execute("DELETE FROM eng")
        r = db.execute("SELECT * FROM employees WHERE dept = 'eng'")
        assert len(r.rows) == 0
        # Sales/HR still exist
        r2 = db.execute("SELECT * FROM employees WHERE dept = 'sales'")
        assert len(r2.rows) == 2

    def test_delete_non_updatable_view_fails(self, db):
        db.execute("CREATE VIEW agg AS SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept")
        with pytest.raises(DatabaseError, match="not updatable"):
            db.execute("DELETE FROM agg WHERE dept = 'eng'")


# =============================================================================
# 9. View Dependency Tracking
# =============================================================================

class TestViewDependencies:
    def test_drop_table_with_view_dependency(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        with pytest.raises(DatabaseError, match="referenced by"):
            db.execute("DROP TABLE employees")

    def test_drop_table_after_view_dropped(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        db.execute("DROP VIEW v1")
        # Now can drop table
        db.execute("DROP TABLE employees")

    def test_drop_view_with_dependent_view(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name, salary FROM employees")
        db.execute("CREATE VIEW v2 AS SELECT name FROM v1")
        with pytest.raises(DatabaseError, match="referenced by"):
            db.execute("DROP VIEW v1")

    def test_drop_dependent_chain(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name, salary FROM employees")
        db.execute("CREATE VIEW v2 AS SELECT name FROM v1")
        db.execute("DROP VIEW v2")
        db.execute("DROP VIEW v1")  # Now OK


# =============================================================================
# 10. ViewCatalog Unit Tests
# =============================================================================

class TestViewCatalog:
    def test_create_and_get(self):
        cat = ViewCatalog()
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'))],
                       from_table=TableRef(table_name='t1'))
        vdef = ViewDefinition(name='v1', columns=None, query=q)
        cat.create_view(vdef)
        assert cat.has_view('v1')
        assert cat.get_view('v1') is vdef

    def test_case_insensitive(self):
        cat = ViewCatalog()
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'))],
                       from_table=TableRef(table_name='t1'))
        cat.create_view(ViewDefinition(name='MyView', columns=None, query=q))
        assert cat.has_view('myview')
        assert cat.has_view('MYVIEW')

    def test_list_views(self):
        cat = ViewCatalog()
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='x'))],
                       from_table=TableRef(table_name='t'))
        cat.create_view(ViewDefinition(name='b_view', columns=None, query=q))
        cat.create_view(ViewDefinition(name='a_view', columns=None, query=q))
        assert cat.list_views() == ['a_view', 'b_view']

    def test_get_dependents(self):
        cat = ViewCatalog()
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q1 = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='x'))],
                        from_table=TableRef(table_name='base_table'))
        q2 = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='x'))],
                        from_table=TableRef(table_name='v1'))
        cat.create_view(ViewDefinition(name='v1', columns=None, query=q1))
        cat.create_view(ViewDefinition(name='v2', columns=None, query=q2))
        assert cat.get_dependents('base_table') == ['v1']
        assert cat.get_dependents('v1') == ['v2']

    def test_drop_nonexistent_raises(self):
        cat = ViewCatalog()
        with pytest.raises(DatabaseError):
            cat.drop_view('nope')

    def test_drop_if_exists_silent(self):
        cat = ViewCatalog()
        cat.drop_view('nope', if_exists=True)  # No error


# =============================================================================
# 11. ViewDefinition Properties
# =============================================================================

class TestViewDefinition:
    def test_is_updatable_simple(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'))],
                       from_table=TableRef(table_name='t1'))
        vdef = ViewDefinition(name='v', columns=None, query=q)
        assert vdef.is_updatable()

    def test_not_updatable_with_group_by(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='dept'))],
                       from_table=TableRef(table_name='t1'),
                       group_by=[SqlColumnRef(table=None, column='dept')])
        vdef = ViewDefinition(name='v', columns=None, query=q)
        assert not vdef.is_updatable()

    def test_not_updatable_with_distinct(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'))],
                       from_table=TableRef(table_name='t1'),
                       distinct=True)
        vdef = ViewDefinition(name='v', columns=None, query=q)
        assert not vdef.is_updatable()

    def test_not_updatable_with_aggregate(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlAggCall
        q = SelectStmt(columns=[SelectExpr(expr=SqlAggCall(func='COUNT', arg=None, distinct=False))],
                       from_table=TableRef(table_name='t1'))
        vdef = ViewDefinition(name='v', columns=None, query=q)
        assert not vdef.is_updatable()

    def test_not_updatable_with_limit(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'))],
                       from_table=TableRef(table_name='t1'),
                       limit=10)
        vdef = ViewDefinition(name='v', columns=None, query=q)
        assert not vdef.is_updatable()

    def test_not_updatable_with_join(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, JoinClause, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'))],
                       from_table=TableRef(table_name='t1'),
                       joins=[JoinClause(join_type='inner', table=TableRef(table_name='t2'))])
        vdef = ViewDefinition(name='v', columns=None, query=q)
        assert not vdef.is_updatable()

    def test_get_base_table(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'))],
                       from_table=TableRef(table_name='users'))
        vdef = ViewDefinition(name='v', columns=None, query=q)
        assert vdef.get_base_table() == 'users'

    def test_get_column_mapping(self):
        from sql_views import SelectStmt, SelectExpr, TableRef, SqlColumnRef
        q = SelectStmt(columns=[
            SelectExpr(expr=SqlColumnRef(table=None, column='name')),
            SelectExpr(expr=SqlColumnRef(table=None, column='age'))
        ], from_table=TableRef(table_name='t1'))
        vdef = ViewDefinition(name='v', columns=['n', 'a'], query=q)
        m = vdef.get_column_mapping()
        assert m == {'n': 'name', 'a': 'age'}


# =============================================================================
# 12. Parser Tests
# =============================================================================

class TestParser:
    def test_parse_create_view(self):
        lexer = ViewLexer("CREATE VIEW v1 AS SELECT name FROM t1")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, CreateViewStmt)
        assert stmt.name == 'v1'
        assert not stmt.replace

    def test_parse_create_or_replace_view(self):
        lexer = ViewLexer("CREATE OR REPLACE VIEW v1 AS SELECT name FROM t1")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, CreateViewStmt)
        assert stmt.replace

    def test_parse_create_view_with_columns(self):
        lexer = ViewLexer("CREATE VIEW v1(a, b) AS SELECT name, age FROM t1")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, CreateViewStmt)
        assert stmt.columns == ['a', 'b']

    def test_parse_drop_view(self):
        lexer = ViewLexer("DROP VIEW v1")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, DropViewStmt)
        assert stmt.name == 'v1'
        assert not stmt.if_exists

    def test_parse_drop_view_if_exists(self):
        lexer = ViewLexer("DROP VIEW IF EXISTS v1")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, DropViewStmt)
        assert stmt.if_exists

    def test_parse_show_views(self):
        lexer = ViewLexer("SHOW VIEWS")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, ShowViewsStmt)

    def test_parse_create_table_still_works(self):
        lexer = ViewLexer("CREATE TABLE t1 (id INT PRIMARY KEY)")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        from sql_views import CreateTableStmt
        assert isinstance(stmt, CreateTableStmt)

    def test_parse_drop_table_still_works(self):
        lexer = ViewLexer("DROP TABLE t1")
        parser = ViewParser(lexer.tokens)
        stmt = parser._parse_statement()
        from sql_views import DropTableStmt
        assert isinstance(stmt, DropTableStmt)


# =============================================================================
# 13. Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_view_with_no_rows(self, db):
        db.execute("CREATE VIEW empty AS SELECT * FROM employees WHERE salary > 9999")
        r = db.execute("SELECT * FROM empty")
        assert len(r.rows) == 0

    def test_view_reflects_data_changes(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        r1 = db.execute("SELECT COUNT(*) AS cnt FROM eng")
        assert r1.rows[0][0] == 2

        db.execute("INSERT INTO employees VALUES (6, 'Frank', 'eng', 130)")
        r2 = db.execute("SELECT COUNT(*) AS cnt FROM eng")
        assert r2.rows[0][0] == 3

    def test_view_after_delete(self, db):
        db.execute("CREATE VIEW eng AS SELECT name FROM employees WHERE dept = 'eng'")
        db.execute("DELETE FROM employees WHERE name = 'Alice'")
        r = db.execute("SELECT * FROM eng")
        assert len(r.rows) == 1

    def test_view_after_update(self, db):
        db.execute("CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'")
        db.execute("UPDATE employees SET dept = 'sales' WHERE name = 'Alice'")
        r = db.execute("SELECT * FROM eng")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Bob'

    def test_multiple_views_same_table(self, db):
        db.execute("CREATE VIEW v_eng AS SELECT * FROM employees WHERE dept = 'eng'")
        db.execute("CREATE VIEW v_sales AS SELECT * FROM employees WHERE dept = 'sales'")
        r1 = db.execute("SELECT COUNT(*) AS c FROM v_eng")
        r2 = db.execute("SELECT COUNT(*) AS c FROM v_sales")
        assert r1.rows[0][0] == 2
        assert r2.rows[0][0] == 2

    def test_view_with_between(self, db):
        db.execute("CREATE VIEW mid_sal AS SELECT name FROM employees WHERE salary BETWEEN 90 AND 100")
        r = db.execute("SELECT * FROM mid_sal")
        names = {row[0] for row in r.rows}
        assert 'Alice' in names  # salary 100
        assert 'Carol' in names  # salary 90
        assert 'Eve' in names    # salary 95

    def test_view_with_in_list(self, db):
        db.execute("CREATE VIEW some AS SELECT name FROM employees WHERE dept IN ('eng', 'hr')")
        r = db.execute("SELECT * FROM some")
        assert len(r.rows) == 3

    def test_view_with_is_null(self, db):
        db.execute("CREATE TABLE nullable (id INT PRIMARY KEY, val TEXT)")
        db.execute("INSERT INTO nullable VALUES (1, 'x')")
        db.execute("INSERT INTO nullable VALUES (2, NULL)")
        db.execute("CREATE VIEW non_null AS SELECT * FROM nullable WHERE val IS NOT NULL")
        r = db.execute("SELECT * FROM non_null")
        assert len(r.rows) == 1

    def test_view_with_like(self, db):
        db.execute("CREATE VIEW a_names AS SELECT name FROM employees WHERE name LIKE 'A%'")
        r = db.execute("SELECT * FROM a_names")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice'


# =============================================================================
# 14. Views with Aggregation
# =============================================================================

class TestViewsWithAggregation:
    def test_view_with_group_by(self, db):
        db.execute("CREATE VIEW dept_stats AS SELECT dept, COUNT(*) AS cnt, SUM(salary) AS total FROM employees GROUP BY dept")
        r = db.execute("SELECT * FROM dept_stats ORDER BY dept ASC")
        assert len(r.rows) == 3

    def test_view_with_having(self, db):
        db.execute("CREATE VIEW big_depts AS SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept HAVING cnt > 1")
        r = db.execute("SELECT * FROM big_depts")
        depts = {row[0] for row in r.rows}
        assert 'eng' in depts
        assert 'sales' in depts
        assert 'hr' not in depts  # only 1 employee

    def test_select_from_aggregate_view(self, db):
        db.execute("CREATE VIEW dept_counts AS SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept")
        r = db.execute("SELECT dept FROM dept_counts WHERE cnt = 1")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'hr'


# =============================================================================
# 15. Views with Stored Procedures Integration
# =============================================================================

class TestViewsWithProcs:
    def test_view_and_procedure_coexist(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        db.execute("""
            CREATE PROCEDURE add_emp(IN p_id INT, IN p_name TEXT, IN p_dept TEXT, IN p_sal INT)
            BEGIN
                INSERT INTO employees VALUES (p_id, p_name, p_dept, p_sal);
            END
        """)
        db.execute("CALL add_emp(10, 'Zara', 'eng', 140)")
        r = db.execute("SELECT * FROM v1")
        names = [row[0] for row in r.rows]
        assert 'Zara' in names

    def test_view_with_udf(self, db):
        db.execute("""
            CREATE FUNCTION get_bonus() RETURNS INT
            BEGIN
                RETURN 50;
            END
        """)
        # UDF with constant args in view definition
        db.execute("CREATE VIEW with_bonus AS SELECT name, get_bonus() AS bonus FROM employees")
        r = db.execute("SELECT * FROM with_bonus WHERE name = 'Alice'")
        assert r.rows[0][0] == 'Alice'
        assert r.rows[0][1] == 50

    def test_show_views_not_in_show_tables(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        r_tables = db.execute("SHOW TABLES")
        table_names = [row[0] for row in r_tables.rows]
        assert 'v1' not in table_names

    def test_show_functions_still_works(self, db):
        db.execute("""
            CREATE FUNCTION my_func(IN x INT) RETURNS INT
            BEGIN
                RETURN x;
            END
        """)
        r = db.execute("SHOW FUNCTIONS")
        assert len(r.rows) == 1


# =============================================================================
# 16. DESCRIBE VIEW
# =============================================================================

class TestDescribeView:
    def test_describe_simple_view(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name, salary FROM employees")
        r = db.execute("DESCRIBE v1")
        assert len(r.rows) == 2
        col_names = [row[0] for row in r.rows]
        assert 'name' in col_names
        assert 'salary' in col_names

    def test_describe_view_with_aliases(self, db):
        db.execute("CREATE VIEW v1(a, b) AS SELECT name, salary FROM employees")
        r = db.execute("DESCRIBE v1")
        col_names = [row[0] for row in r.rows]
        assert 'a' in col_names
        assert 'b' in col_names

    def test_describe_view_star(self, db):
        db.execute("CREATE VIEW v1 AS SELECT * FROM employees")
        r = db.execute("DESCRIBE v1")
        assert len(r.rows) == 4  # id, name, dept, salary

    def test_describe_table_still_works(self, db):
        r = db.execute("DESCRIBE employees")
        assert len(r.rows) == 4


# =============================================================================
# 17. Stress Tests
# =============================================================================

class TestStress:
    def test_many_views(self, db):
        for i in range(20):
            db.execute(f"CREATE VIEW v{i} AS SELECT name FROM employees WHERE salary > {80 + i}")
        r = db.execute("SHOW VIEWS")
        assert len(r.rows) == 20

    def test_view_over_large_dataset(self, db):
        db.execute("CREATE TABLE big (id INT PRIMARY KEY, val INT)")
        for i in range(100):
            db.execute(f"INSERT INTO big VALUES ({i}, {i * 10})")
        db.execute("CREATE VIEW big_v AS SELECT * FROM big WHERE val > 500")
        r = db.execute("SELECT COUNT(*) AS cnt FROM big_v")
        assert r.rows[0][0] == 49  # 51-99

    def test_rapid_create_drop(self, db):
        for i in range(10):
            db.execute(f"CREATE VIEW temp_v AS SELECT name FROM employees")
            db.execute("DROP VIEW temp_v")


# =============================================================================
# 18. Updatable Nested Views
# =============================================================================

class TestUpdatableNestedViews:
    def test_insert_through_nested_view(self, db):
        db.execute("CREATE VIEW v1 AS SELECT id, name, dept, salary FROM employees")
        db.execute("CREATE VIEW v2 AS SELECT id, name, dept, salary FROM v1 WHERE dept = 'eng'")
        db.execute("INSERT INTO v2 VALUES (10, 'Test', 'eng', 200)")
        r = db.execute("SELECT name FROM employees WHERE id = 10")
        assert r.rows[0][0] == 'Test'

    def test_update_through_nested_view(self, db):
        db.execute("CREATE VIEW v1 AS SELECT id, name, dept, salary FROM employees")
        db.execute("CREATE VIEW v2 AS SELECT id, name, dept, salary FROM v1 WHERE dept = 'eng'")
        db.execute("UPDATE v2 SET salary = 999 WHERE name = 'Alice'")
        r = db.execute("SELECT salary FROM employees WHERE name = 'Alice'")
        assert r.rows[0][0] == 999

    def test_delete_through_nested_view(self, db):
        db.execute("CREATE VIEW v1 AS SELECT id, name, dept, salary FROM employees")
        db.execute("CREATE VIEW v2 AS SELECT id, name, dept, salary FROM v1 WHERE dept = 'eng'")
        db.execute("DELETE FROM v2 WHERE name = 'Alice'")
        r = db.execute("SELECT * FROM employees WHERE name = 'Alice'")
        assert len(r.rows) == 0


# =============================================================================
# 19. View with CASE expressions
# =============================================================================

class TestViewWithCase:
    def test_view_with_case(self, db):
        db.execute("""
            CREATE VIEW salary_tier AS
            SELECT name,
                   CASE WHEN salary > 100 THEN 'high'
                        WHEN salary > 90 THEN 'medium'
                        ELSE 'low' END AS tier
            FROM employees
        """)
        r = db.execute("SELECT * FROM salary_tier WHERE name = 'Bob'")
        assert r.rows[0][1] == 'high'

    def test_view_with_arithmetic(self, db):
        db.execute("CREATE VIEW bonus AS SELECT name, salary + 10 AS with_bonus FROM employees")
        r = db.execute("SELECT with_bonus FROM bonus WHERE name = 'Alice'")
        assert r.rows[0][0] == 110


# =============================================================================
# 20. Replace existing view
# =============================================================================

class TestReplaceView:
    def test_replace_changes_definition(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        r1 = db.execute("SELECT * FROM v1")
        assert len(r1.columns) == 1

        db.execute("CREATE OR REPLACE VIEW v1 AS SELECT name, salary FROM employees")
        r2 = db.execute("SELECT * FROM v1")
        assert len(r2.columns) == 2

    def test_replace_preserves_data_access(self, db):
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees WHERE dept = 'eng'")
        db.execute("CREATE OR REPLACE VIEW v1 AS SELECT name FROM employees WHERE dept = 'sales'")
        r = db.execute("SELECT * FROM v1")
        names = {row[0] for row in r.rows}
        assert names == {'Carol', 'Dave'}


# =============================================================================
# 21. Mixed operations
# =============================================================================

class TestMixedOperations:
    def test_execute_many_with_views(self, db):
        results = db.execute_many("""
            CREATE VIEW v1 AS SELECT name FROM employees;
            SELECT * FROM v1;
        """)
        assert len(results) == 2
        assert 'CREATE VIEW' in results[0].message
        assert len(results[1].rows) == 5

    def test_view_in_transaction(self, db):
        db.execute("BEGIN")
        db.execute("CREATE VIEW v1 AS SELECT name FROM employees")
        r = db.execute("SELECT * FROM v1")
        assert len(r.rows) == 5
        db.execute("COMMIT")
        r2 = db.execute("SELECT * FROM v1")
        assert len(r2.rows) == 5

    def test_insert_then_query_view(self, db):
        db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'eng'")
        db.execute("INSERT INTO employees VALUES (6, 'Frank', 'eng', 130)")
        r = db.execute("SELECT * FROM eng")
        names = {row[1] for row in r.rows}
        assert 'Frank' in names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
