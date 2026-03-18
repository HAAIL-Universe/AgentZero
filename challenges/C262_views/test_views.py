"""
Tests for C262: SQL Views
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from views import (
    ViewDB, ViewDef, ViewRegistry, ViewParser,
    parse_view_sql, parse_view_sql_multi,
    CreateViewStmt, DropViewStmt,
)
from mini_database import (
    CatalogError, ParseError, ResultSet,
    SelectStmt, CreateTableStmt, DropTableStmt,
)


# =============================================================================
# ViewDef dataclass
# =============================================================================

class TestViewDef:

    def test_basic_view_def(self):
        from mini_database import SqlColumnRef, SelectExpr, TableRef
        stmt = SelectStmt(
            columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'), alias=None)],
            from_table=TableRef(table_name='users', alias=None),
            joins=[], where=None, group_by=None, having=None,
            order_by=None, limit=None, offset=None, distinct=False,
        )
        vd = ViewDef(name='v', select_sql='SELECT name FROM users',
                     select_stmt=stmt, column_aliases=[])
        assert vd.name == 'v'
        assert vd.is_updatable is True
        assert vd.base_table == 'users'

    def test_non_updatable_distinct(self):
        from mini_database import SqlColumnRef, SelectExpr, TableRef
        stmt = SelectStmt(
            columns=[SelectExpr(expr=SqlColumnRef(table=None, column='name'), alias=None)],
            from_table=TableRef(table_name='users', alias=None),
            joins=[], where=None, group_by=None, having=None,
            order_by=None, limit=None, offset=None, distinct=True,
        )
        vd = ViewDef(name='v', select_sql='', select_stmt=stmt, column_aliases=[])
        assert vd.is_updatable is False
        assert vd.base_table is None

    def test_non_updatable_group_by(self):
        from mini_database import SqlColumnRef, SelectExpr, TableRef
        stmt = SelectStmt(
            columns=[SelectExpr(expr=SqlColumnRef(table=None, column='dept'), alias=None)],
            from_table=TableRef(table_name='users', alias=None),
            joins=[], where=None, group_by=['dept'], having=None,
            order_by=None, limit=None, offset=None, distinct=False,
        )
        vd = ViewDef(name='v', select_sql='', select_stmt=stmt, column_aliases=[])
        assert vd.is_updatable is False

    def test_column_names_from_select(self):
        from mini_database import SqlColumnRef, SelectExpr, TableRef
        stmt = SelectStmt(
            columns=[
                SelectExpr(expr=SqlColumnRef(table=None, column='id'), alias=None),
                SelectExpr(expr=SqlColumnRef(table=None, column='name'), alias='full_name'),
            ],
            from_table=TableRef(table_name='users', alias=None),
            joins=[], where=None, group_by=None, having=None,
            order_by=None, limit=None, offset=None, distinct=False,
        )
        vd = ViewDef(name='v', select_sql='', select_stmt=stmt, column_aliases=[])
        assert vd.get_column_names() == ['id', 'full_name']

    def test_column_names_with_aliases(self):
        from mini_database import SqlColumnRef, SelectExpr, TableRef
        stmt = SelectStmt(
            columns=[
                SelectExpr(expr=SqlColumnRef(table=None, column='id'), alias=None),
                SelectExpr(expr=SqlColumnRef(table=None, column='name'), alias=None),
            ],
            from_table=TableRef(table_name='users', alias=None),
            joins=[], where=None, group_by=None, having=None,
            order_by=None, limit=None, offset=None, distinct=False,
        )
        vd = ViewDef(name='v', select_sql='', select_stmt=stmt,
                     column_aliases=['a', 'b'])
        assert vd.get_column_names() == ['a', 'b']


# =============================================================================
# ViewRegistry
# =============================================================================

class TestViewRegistry:

    def test_create_and_get(self):
        reg = ViewRegistry()
        reg.create('v1', 'SELECT 1', None, [])
        assert reg.exists('v1')
        assert reg.get('v1').name == 'v1'

    def test_create_duplicate_fails(self):
        reg = ViewRegistry()
        reg.create('v1', 'SELECT 1', None, [])
        with pytest.raises(CatalogError, match="already exists"):
            reg.create('v1', 'SELECT 2', None, [])

    def test_create_or_replace(self):
        reg = ViewRegistry()
        reg.create('v1', 'SELECT 1', None, [])
        reg.create('v1', 'SELECT 2', None, [], or_replace=True)
        assert reg.get('v1').select_sql == 'SELECT 2'

    def test_drop(self):
        reg = ViewRegistry()
        reg.create('v1', 'SELECT 1', None, [])
        reg.drop('v1')
        assert not reg.exists('v1')

    def test_drop_nonexistent_fails(self):
        reg = ViewRegistry()
        with pytest.raises(CatalogError, match="does not exist"):
            reg.drop('v1')

    def test_drop_if_exists(self):
        reg = ViewRegistry()
        reg.drop('v1', if_exists=True)  # no error

    def test_list_views(self):
        reg = ViewRegistry()
        reg.create('b_view', 'SELECT 1', None, [])
        reg.create('a_view', 'SELECT 2', None, [])
        assert reg.list_views() == ['a_view', 'b_view']


# =============================================================================
# Parser -- CREATE VIEW
# =============================================================================

class TestParserCreateView:

    def test_basic_create_view(self):
        stmt = parse_view_sql("CREATE VIEW active_users AS SELECT id, name FROM users WHERE active = 1")
        assert isinstance(stmt, CreateViewStmt)
        assert stmt.view_name == 'active_users'
        assert stmt.or_replace is False
        assert stmt.column_aliases == []
        assert isinstance(stmt.select_stmt, SelectStmt)

    def test_create_or_replace_view(self):
        stmt = parse_view_sql("CREATE OR REPLACE VIEW v AS SELECT id FROM t")
        assert isinstance(stmt, CreateViewStmt)
        assert stmt.view_name == 'v'
        assert stmt.or_replace is True

    def test_create_view_with_column_aliases(self):
        stmt = parse_view_sql("CREATE VIEW v (a, b) AS SELECT id, name FROM t")
        assert isinstance(stmt, CreateViewStmt)
        assert stmt.column_aliases == ['a', 'b']

    def test_create_view_select_has_where(self):
        stmt = parse_view_sql("CREATE VIEW v AS SELECT id FROM t WHERE id > 5")
        assert stmt.select_stmt.where is not None

    def test_create_table_still_works(self):
        stmt = parse_view_sql("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        assert isinstance(stmt, CreateTableStmt)


# =============================================================================
# Parser -- DROP VIEW
# =============================================================================

class TestParserDropView:

    def test_drop_view(self):
        stmt = parse_view_sql("DROP VIEW v")
        assert isinstance(stmt, DropViewStmt)
        assert stmt.view_name == 'v'
        assert stmt.if_exists is False

    def test_drop_view_if_exists(self):
        stmt = parse_view_sql("DROP VIEW IF EXISTS v")
        assert isinstance(stmt, DropViewStmt)
        assert stmt.if_exists is True

    def test_drop_table_still_works(self):
        stmt = parse_view_sql("DROP TABLE t")
        assert isinstance(stmt, DropTableStmt)


# =============================================================================
# CREATE VIEW execution
# =============================================================================

class TestCreateView:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
        self.db.execute("INSERT INTO users VALUES (3, 'Carol', 35)")

    def test_create_view(self):
        result = self.db.execute("CREATE VIEW young AS SELECT id, name FROM users WHERE age < 30")
        assert result.rows[0][0] == 'View created'

    def test_create_view_duplicate_fails(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM users")
        with pytest.raises(CatalogError, match="already exists"):
            self.db.execute("CREATE VIEW v AS SELECT name FROM users")

    def test_create_or_replace_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM users")
        self.db.execute("CREATE OR REPLACE VIEW v AS SELECT name FROM users")
        result = self.db.execute("SELECT * FROM v")
        assert result.columns == ['name']

    def test_create_view_references_nonexistent_table(self):
        with pytest.raises(CatalogError, match="does not exist"):
            self.db.execute("CREATE VIEW v AS SELECT id FROM nonexistent")

    def test_create_view_conflicts_with_table(self):
        with pytest.raises(CatalogError, match="table with that name"):
            self.db.execute("CREATE VIEW users AS SELECT id FROM users")

    def test_create_view_with_column_aliases(self):
        self.db.execute("CREATE VIEW v (uid, uname) AS SELECT id, name FROM users")
        result = self.db.execute("SELECT * FROM v")
        assert result.columns == ['uid', 'uname']

    def test_create_view_column_count_mismatch(self):
        with pytest.raises(CatalogError, match="column list"):
            self.db.execute("CREATE VIEW v (a, b, c) AS SELECT id, name FROM users")


# =============================================================================
# SELECT from views
# =============================================================================

class TestSelectFromView:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price FLOAT, active INT)")
        self.db.execute("INSERT INTO products VALUES (1, 'Widget', 9.99, 1)")
        self.db.execute("INSERT INTO products VALUES (2, 'Gadget', 24.99, 1)")
        self.db.execute("INSERT INTO products VALUES (3, 'Doodad', 4.99, 0)")
        self.db.execute("INSERT INTO products VALUES (4, 'Thingamajig', 49.99, 1)")

    def test_select_all_from_view(self):
        self.db.execute("CREATE VIEW active_products AS SELECT id, name, price FROM products WHERE active = 1")
        result = self.db.execute("SELECT * FROM active_products")
        assert len(result.rows) == 3
        names = [r[1] for r in result.rows]
        assert 'Widget' in names
        assert 'Gadget' in names
        assert 'Thingamajig' in names
        assert 'Doodad' not in names

    def test_select_columns_from_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, price FROM products WHERE active = 1")
        result = self.db.execute("SELECT name, price FROM v")
        assert result.columns == ['name', 'price']
        assert len(result.rows) == 3

    def test_select_with_where_from_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, price FROM products WHERE active = 1")
        result = self.db.execute("SELECT name FROM v WHERE price > 20")
        assert len(result.rows) == 2
        names = [r[0] for r in result.rows]
        assert 'Gadget' in names
        assert 'Thingamajig' in names

    def test_select_with_order_by(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, price FROM products WHERE active = 1")
        result = self.db.execute("SELECT name FROM v ORDER BY price")
        assert result.rows[0][0] == 'Widget'
        assert result.rows[-1][0] == 'Thingamajig'

    def test_select_with_order_by_desc(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, price FROM products WHERE active = 1")
        result = self.db.execute("SELECT name FROM v ORDER BY price DESC")
        assert result.rows[0][0] == 'Thingamajig'
        assert result.rows[-1][0] == 'Widget'

    def test_select_with_limit(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, price FROM products WHERE active = 1")
        result = self.db.execute("SELECT name FROM v ORDER BY price LIMIT 2")
        assert len(result.rows) == 2

    def test_select_with_offset(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, price FROM products WHERE active = 1")
        result = self.db.execute("SELECT name FROM v ORDER BY price LIMIT 1 OFFSET 1")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'Gadget'

    def test_select_distinct_from_view(self):
        self.db.execute("INSERT INTO products VALUES (5, 'Widget', 9.99, 1)")
        self.db.execute("CREATE VIEW v AS SELECT name FROM products WHERE active = 1")
        result = self.db.execute("SELECT DISTINCT name FROM v")
        names = [r[0] for r in result.rows]
        assert names.count('Widget') == 1

    def test_view_reflects_data_changes(self):
        """View is not materialized -- reflects current data."""
        self.db.execute("CREATE VIEW v AS SELECT id, name FROM products WHERE active = 1")
        result1 = self.db.execute("SELECT * FROM v")
        assert len(result1.rows) == 3

        self.db.execute("INSERT INTO products VALUES (5, 'NewItem', 1.99, 1)")
        result2 = self.db.execute("SELECT * FROM v")
        assert len(result2.rows) == 4

    def test_view_with_column_aliases(self):
        self.db.execute("CREATE VIEW v (pid, pname) AS SELECT id, name FROM products")
        result = self.db.execute("SELECT pid, pname FROM v")
        assert result.columns == ['pid', 'pname']
        assert len(result.rows) == 4


# =============================================================================
# Nested views
# =============================================================================

class TestNestedViews:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE items (id INT PRIMARY KEY, name TEXT, category TEXT, price FLOAT)")
        self.db.execute("INSERT INTO items VALUES (1, 'A', 'electronics', 100)")
        self.db.execute("INSERT INTO items VALUES (2, 'B', 'electronics', 200)")
        self.db.execute("INSERT INTO items VALUES (3, 'C', 'books', 20)")
        self.db.execute("INSERT INTO items VALUES (4, 'D', 'books', 30)")

    def test_view_of_view(self):
        self.db.execute("CREATE VIEW electronics AS SELECT id, name, price FROM items WHERE category = 'electronics'")
        self.db.execute("CREATE VIEW cheap_electronics AS SELECT id, name FROM electronics WHERE price < 150")
        result = self.db.execute("SELECT * FROM cheap_electronics")
        assert len(result.rows) == 1
        assert result.rows[0][1] == 'A'

    def test_three_level_nesting(self):
        self.db.execute("CREATE VIEW all_items AS SELECT id, name, category, price FROM items")
        self.db.execute("CREATE VIEW electronics AS SELECT id, name, price FROM all_items WHERE category = 'electronics'")
        self.db.execute("CREATE VIEW cheap_electronics AS SELECT name FROM electronics WHERE price < 150")
        result = self.db.execute("SELECT * FROM cheap_electronics")
        assert len(result.rows) == 1


# =============================================================================
# DROP VIEW
# =============================================================================

class TestDropView:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")

    def test_drop_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t")
        result = self.db.execute("DROP VIEW v")
        assert result.rows[0][0] == 'View dropped'
        assert not self.db.is_view('v')

    def test_drop_nonexistent_view_fails(self):
        with pytest.raises(CatalogError, match="does not exist"):
            self.db.execute("DROP VIEW nonexistent")

    def test_drop_view_if_exists(self):
        result = self.db.execute("DROP VIEW IF EXISTS nonexistent")
        assert result.rows[0][0] == 'View dropped'

    def test_drop_view_with_dependent_fails(self):
        self.db.execute("CREATE VIEW v1 AS SELECT id FROM t")
        self.db.execute("CREATE VIEW v2 AS SELECT id FROM v1")
        with pytest.raises(CatalogError, match="depends on it"):
            self.db.execute("DROP VIEW v1")

    def test_drop_view_after_dropping_dependent(self):
        self.db.execute("CREATE VIEW v1 AS SELECT id FROM t")
        self.db.execute("CREATE VIEW v2 AS SELECT id FROM v1")
        self.db.execute("DROP VIEW v2")
        self.db.execute("DROP VIEW v1")  # should work now

    def test_drop_table_with_view_dependency_fails(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t")
        with pytest.raises(CatalogError, match="depend on it"):
            self.db.execute("DROP TABLE t")


# =============================================================================
# Updatable views -- INSERT
# =============================================================================

class TestInsertIntoView:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, dept TEXT, salary INT)")
        self.db.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 100)")
        self.db.execute("INSERT INTO employees VALUES (2, 'Bob', 'Sales', 80)")

    def test_insert_into_simple_view(self):
        self.db.execute("CREATE VIEW all_emp AS SELECT id, name, dept, salary FROM employees")
        self.db.execute("INSERT INTO all_emp VALUES (3, 'Carol', 'Engineering', 90)")
        result = self.db.execute("SELECT * FROM employees WHERE id = 3")
        assert len(result.rows) == 1
        assert result.rows[0][1] == 'Carol'

    def test_insert_into_filtered_view(self):
        """Insert into view with WHERE -- row goes into base table even if it doesn't match view filter."""
        self.db.execute("CREATE VIEW eng AS SELECT id, name, dept, salary FROM employees WHERE dept = 'Engineering'")
        self.db.execute("INSERT INTO eng VALUES (3, 'Dave', 'Sales', 70)")
        # Dave is in the base table
        result = self.db.execute("SELECT * FROM employees WHERE id = 3")
        assert len(result.rows) == 1
        # But not visible through the view
        result2 = self.db.execute("SELECT * FROM eng")
        names = [r[1] for r in result2.rows]
        assert 'Dave' not in names

    def test_insert_non_updatable_view_fails(self):
        self.db.execute("CREATE VIEW v AS SELECT DISTINCT name FROM employees")
        with pytest.raises(CatalogError, match="not updatable"):
            self.db.execute("INSERT INTO v VALUES ('Zoe')")

    def test_insert_into_view_with_aliases(self):
        self.db.execute("CREATE VIEW v (eid, ename, edept, esal) AS SELECT id, name, dept, salary FROM employees")
        self.db.execute("INSERT INTO v (eid, ename, edept, esal) VALUES (4, 'Eve', 'HR', 85)")
        result = self.db.execute("SELECT * FROM employees WHERE id = 4")
        assert len(result.rows) == 1
        assert result.rows[0][1] == 'Eve'


# =============================================================================
# Updatable views -- UPDATE
# =============================================================================

class TestUpdateView:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE staff (id INT PRIMARY KEY, name TEXT, salary INT)")
        self.db.execute("INSERT INTO staff VALUES (1, 'Alice', 100)")
        self.db.execute("INSERT INTO staff VALUES (2, 'Bob', 80)")
        self.db.execute("INSERT INTO staff VALUES (3, 'Carol', 120)")

    def test_update_through_view(self):
        self.db.execute("CREATE VIEW all_staff AS SELECT id, name, salary FROM staff")
        self.db.execute("UPDATE all_staff SET salary = 110 WHERE name = 'Bob'")
        result = self.db.execute("SELECT salary FROM staff WHERE name = 'Bob'")
        assert result.rows[0][0] == 110

    def test_update_filtered_view(self):
        """UPDATE through view with WHERE -- view's WHERE merged with UPDATE's WHERE."""
        self.db.execute("CREATE VIEW high_salary AS SELECT id, name, salary FROM staff WHERE salary > 90")
        self.db.execute("UPDATE high_salary SET salary = 200 WHERE name = 'Carol'")
        result = self.db.execute("SELECT salary FROM staff WHERE name = 'Carol'")
        assert result.rows[0][0] == 200
        # Alice (salary=100, matches view) should be unaffected
        result2 = self.db.execute("SELECT salary FROM staff WHERE name = 'Alice'")
        assert result2.rows[0][0] == 100

    def test_update_non_updatable_fails(self):
        self.db.execute("CREATE VIEW v AS SELECT DISTINCT name FROM staff")
        with pytest.raises(CatalogError, match="not updatable"):
            self.db.execute("UPDATE v SET name = 'X' WHERE name = 'Alice'")

    def test_update_all_rows_through_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, salary FROM staff")
        self.db.execute("UPDATE v SET salary = 999")
        result = self.db.execute("SELECT salary FROM staff")
        assert all(r[0] == 999 for r in result.rows)


# =============================================================================
# Updatable views -- DELETE
# =============================================================================

class TestDeleteFromView:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE items (id INT PRIMARY KEY, name TEXT, active INT)")
        self.db.execute("INSERT INTO items VALUES (1, 'A', 1)")
        self.db.execute("INSERT INTO items VALUES (2, 'B', 0)")
        self.db.execute("INSERT INTO items VALUES (3, 'C', 1)")

    def test_delete_from_simple_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name, active FROM items")
        self.db.execute("DELETE FROM v WHERE name = 'B'")
        result = self.db.execute("SELECT * FROM items")
        assert len(result.rows) == 2

    def test_delete_from_filtered_view(self):
        """DELETE through filtered view -- merged WHERE only deletes matching rows."""
        self.db.execute("CREATE VIEW active_items AS SELECT id, name, active FROM items WHERE active = 1")
        self.db.execute("DELETE FROM active_items WHERE name = 'A'")
        result = self.db.execute("SELECT * FROM items")
        # A deleted, B and C remain
        assert len(result.rows) == 2
        names = [r[1] for r in result.rows]
        assert 'A' not in names

    def test_delete_non_updatable_fails(self):
        self.db.execute("CREATE VIEW v AS SELECT DISTINCT name FROM items")
        with pytest.raises(CatalogError, match="not updatable"):
            self.db.execute("DELETE FROM v WHERE name = 'A'")


# =============================================================================
# SHOW TABLES with views
# =============================================================================

class TestShowTables:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE t1 (id INT PRIMARY KEY)")
        self.db.execute("CREATE TABLE t2 (id INT PRIMARY KEY)")

    def test_show_tables_no_views(self):
        result = self.db.execute("SHOW TABLES")
        assert len(result.rows) == 2
        assert result.columns == ['name', 'type']
        types = [r[1] for r in result.rows]
        assert all(t == 'TABLE' for t in types)

    def test_show_tables_with_views(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t1")
        result = self.db.execute("SHOW TABLES")
        assert len(result.rows) == 3
        names = [r[0] for r in result.rows]
        types = {r[0]: r[1] for r in result.rows}
        assert 'v' in names
        assert types['v'] == 'VIEW'
        assert types['t1'] == 'TABLE'


# =============================================================================
# DESCRIBE view
# =============================================================================

class TestDescribeView:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT, value FLOAT)")
        self.db.execute("INSERT INTO t VALUES (1, 'test', 3.14)")

    def test_describe_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id, name FROM t")
        result = self.db.execute("DESCRIBE v")
        assert len(result.rows) == 2
        col_names = [r[0] for r in result.rows]
        assert 'id' in col_names
        assert 'name' in col_names

    def test_describe_view_with_aliases(self):
        self.db.execute("CREATE VIEW v (a, b) AS SELECT id, name FROM t")
        result = self.db.execute("DESCRIBE v")
        col_names = [r[0] for r in result.rows]
        assert col_names == ['a', 'b']

    def test_describe_table_still_works(self):
        result = self.db.execute("DESCRIBE t")
        col_names = [r[0] for r in result.rows]
        assert 'id' in col_names
        assert 'name' in col_names
        assert 'value' in col_names


# =============================================================================
# Introspection API
# =============================================================================

class TestIntrospectionAPI:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")

    def test_get_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t")
        vd = self.db.get_view('v')
        assert vd is not None
        assert vd.name == 'v'

    def test_get_view_nonexistent(self):
        assert self.db.get_view('nope') is None

    def test_list_views(self):
        self.db.execute("CREATE VIEW v1 AS SELECT id FROM t")
        self.db.execute("CREATE VIEW v2 AS SELECT name FROM t")
        assert self.db.list_views() == ['v1', 'v2']

    def test_is_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t")
        assert self.db.is_view('v') is True
        assert self.db.is_view('t') is False
        assert self.db.is_view('nonexistent') is False


# =============================================================================
# Passthrough (non-view operations)
# =============================================================================

class TestPassthrough:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'Alice')")

    def test_insert_into_table(self):
        self.db.execute("INSERT INTO t VALUES (2, 'Bob')")
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 2

    def test_update_table(self):
        self.db.execute("UPDATE t SET name = 'ALICE' WHERE id = 1")
        result = self.db.execute("SELECT name FROM t WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'

    def test_delete_from_table(self):
        self.db.execute("DELETE FROM t WHERE id = 1")
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 0

    def test_select_from_table(self):
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 1

    def test_create_index_still_works(self):
        self.db.execute("CREATE INDEX idx_name ON t (name)")
        # just verify no error


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:

    def setup_method(self):
        self.db = ViewDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        self.db.execute("INSERT INTO t VALUES (2, 20)")
        self.db.execute("INSERT INTO t VALUES (3, 30)")

    def test_view_with_star(self):
        self.db.execute("CREATE VIEW v AS SELECT * FROM t")
        result = self.db.execute("SELECT * FROM v")
        assert len(result.rows) == 3
        assert len(result.columns) >= 2

    def test_view_with_computed_column(self):
        self.db.execute("CREATE VIEW v AS SELECT id, val * 2 AS doubled FROM t")
        result = self.db.execute("SELECT * FROM v")
        assert len(result.rows) == 3
        # Find the doubled column
        doubled_idx = result.columns.index('doubled')
        assert result.rows[0][doubled_idx] == 20

    def test_view_with_where_and_order(self):
        self.db.execute("CREATE VIEW v AS SELECT id, val FROM t WHERE val > 10 ORDER BY val DESC")
        # Note: ORDER BY in view definition is not guaranteed to persist in outer queries
        # but the WHERE should filter
        result = self.db.execute("SELECT * FROM v")
        assert len(result.rows) == 2

    def test_empty_view_result(self):
        self.db.execute("CREATE VIEW v AS SELECT id, val FROM t WHERE val > 100")
        result = self.db.execute("SELECT * FROM v")
        assert len(result.rows) == 0

    def test_execute_many(self):
        results = self.db.execute_many(
            "CREATE VIEW v AS SELECT id FROM t;"
            "SELECT * FROM v;"
        )
        assert len(results) == 2
        assert results[0].rows[0][0] == 'View created'
        assert len(results[1].rows) == 3

    def test_view_updated_data(self):
        """Views are not materialized -- always show current data."""
        self.db.execute("CREATE VIEW v AS SELECT id, val FROM t")
        self.db.execute("UPDATE t SET val = 99 WHERE id = 1")
        result = self.db.execute("SELECT val FROM v WHERE id = 1")
        assert result.rows[0][0] == 99

    def test_view_after_delete(self):
        self.db.execute("CREATE VIEW v AS SELECT id, val FROM t")
        self.db.execute("DELETE FROM t WHERE id = 2")
        result = self.db.execute("SELECT * FROM v")
        assert len(result.rows) == 2

    def test_or_replace_changes_query(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t WHERE val > 10")
        r1 = self.db.execute("SELECT * FROM v")
        assert len(r1.rows) == 2

        self.db.execute("CREATE OR REPLACE VIEW v AS SELECT id FROM t WHERE val > 20")
        r2 = self.db.execute("SELECT * FROM v")
        assert len(r2.rows) == 1

    def test_view_with_alias_select(self):
        self.db.execute("CREATE VIEW v AS SELECT id AS item_id, val AS amount FROM t")
        result = self.db.execute("SELECT item_id, amount FROM v")
        assert result.columns == ['item_id', 'amount']
        assert len(result.rows) == 3
