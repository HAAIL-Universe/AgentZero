"""
Tests for C263: CREATE TABLE AS SELECT (CTAS)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C262_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from ctas import (
    CTASDB, CTASParser, CreateTableAsSelectStmt,
    parse_ctas_sql, parse_ctas_sql_multi,
    _infer_type, _infer_column_type,
)
from mini_database import (
    CatalogError, ParseError, ResultSet,
    SelectStmt, CreateTableStmt, DropTableStmt,
)


# =============================================================================
# Type inference
# =============================================================================

class TestTypeInference:

    def test_infer_int(self):
        assert _infer_type(42) == 'INT'

    def test_infer_float(self):
        assert _infer_type(3.14) == 'FLOAT'

    def test_infer_str(self):
        assert _infer_type('hello') == 'TEXT'

    def test_infer_bool(self):
        assert _infer_type(True) == 'BOOL'

    def test_infer_none(self):
        assert _infer_type(None) == 'TEXT'

    def test_column_type_all_int(self):
        assert _infer_column_type([1, 2, 3]) == 'INT'

    def test_column_type_mixed_int_float(self):
        assert _infer_column_type([1, 2.5, 3]) == 'FLOAT'

    def test_column_type_all_none(self):
        assert _infer_column_type([None, None]) == 'TEXT'

    def test_column_type_with_nulls(self):
        assert _infer_column_type([None, 42, None]) == 'INT'

    def test_column_type_text(self):
        assert _infer_column_type(['a', 'b', 'c']) == 'TEXT'

    def test_column_type_empty(self):
        assert _infer_column_type([]) == 'TEXT'


# =============================================================================
# Parser -- CTAS
# =============================================================================

class TestParserCTAS:

    def test_basic_ctas(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 AS SELECT id, name FROM t1")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.table_name == 't2'
        assert stmt.column_names is None
        assert stmt.if_not_exists is False
        assert isinstance(stmt.select_stmt, SelectStmt)

    def test_ctas_if_not_exists(self):
        stmt = parse_ctas_sql("CREATE TABLE IF NOT EXISTS t2 AS SELECT id FROM t1")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.if_not_exists is True

    def test_ctas_with_column_names(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 (a, b) AS SELECT id, name FROM t1")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.column_names == ['a', 'b']

    def test_ctas_with_where(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 AS SELECT id FROM t1 WHERE id > 5")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.select_stmt.where is not None

    def test_regular_create_table_still_works(self):
        stmt = parse_ctas_sql("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        assert isinstance(stmt, CreateTableStmt)

    def test_create_view_still_works(self):
        from views import CreateViewStmt
        stmt = parse_ctas_sql("CREATE VIEW v AS SELECT id FROM t")
        assert isinstance(stmt, CreateViewStmt)

    def test_ctas_with_star(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 AS SELECT * FROM t1")
        assert isinstance(stmt, CreateTableAsSelectStmt)

    def test_ctas_with_order_by(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 AS SELECT id FROM t1 ORDER BY id")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.select_stmt.order_by is not None

    def test_ctas_with_limit(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 AS SELECT id FROM t1 LIMIT 10")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.select_stmt.limit == 10


# =============================================================================
# Basic CTAS execution
# =============================================================================

class TestBasicCTAS:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, salary INT, dept TEXT)")
        self.db.execute("INSERT INTO employees VALUES (1, 'Alice', 100, 'Engineering')")
        self.db.execute("INSERT INTO employees VALUES (2, 'Bob', 80, 'Sales')")
        self.db.execute("INSERT INTO employees VALUES (3, 'Carol', 120, 'Engineering')")
        self.db.execute("INSERT INTO employees VALUES (4, 'Dave', 90, 'Sales')")

    def test_ctas_basic(self):
        result = self.db.execute("CREATE TABLE emp_copy AS SELECT id, name, salary FROM employees")
        assert 'created' in result.rows[0][0].lower()
        assert '4' in result.rows[0][0]

        data = self.db.execute("SELECT * FROM emp_copy")
        assert len(data.rows) == 4

    def test_ctas_with_where(self):
        self.db.execute("CREATE TABLE engineers AS SELECT id, name, salary FROM employees WHERE dept = 'Engineering'")
        result = self.db.execute("SELECT * FROM engineers")
        assert len(result.rows) == 2
        names = [r[1] for r in result.rows]
        assert 'Alice' in names
        assert 'Carol' in names

    def test_ctas_preserves_data(self):
        self.db.execute("CREATE TABLE backup AS SELECT id, name FROM employees")
        result = self.db.execute("SELECT name FROM backup ORDER BY id")
        assert result.rows[0][0] == 'Alice'
        assert result.rows[1][0] == 'Bob'
        assert result.rows[2][0] == 'Carol'
        assert result.rows[3][0] == 'Dave'

    def test_ctas_select_star(self):
        self.db.execute("CREATE TABLE emp_all AS SELECT * FROM employees")
        result = self.db.execute("SELECT * FROM emp_all")
        assert len(result.rows) == 4
        assert len(result.columns) == 4

    def test_ctas_with_expressions(self):
        self.db.execute("CREATE TABLE bonuses AS SELECT name, salary * 2 AS double_salary FROM employees")
        result = self.db.execute("SELECT * FROM bonuses")
        assert len(result.rows) == 4
        assert 'double_salary' in result.columns

    def test_ctas_with_order_by(self):
        self.db.execute("CREATE TABLE sorted_emp AS SELECT name, salary FROM employees ORDER BY salary DESC")
        result = self.db.execute("SELECT * FROM sorted_emp")
        assert result.rows[0][0] == 'Carol'

    def test_ctas_with_limit(self):
        self.db.execute("CREATE TABLE top2 AS SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 2")
        result = self.db.execute("SELECT * FROM top2")
        assert len(result.rows) == 2

    def test_ctas_empty_result(self):
        self.db.execute("CREATE TABLE nobody AS SELECT name FROM employees WHERE salary > 1000")
        result = self.db.execute("SELECT * FROM nobody")
        assert len(result.rows) == 0

    def test_ctas_new_table_is_independent(self):
        """CTAS creates a snapshot -- changes to source don't affect new table."""
        self.db.execute("CREATE TABLE snapshot AS SELECT id, name FROM employees")
        self.db.execute("INSERT INTO employees VALUES (5, 'Eve', 110, 'HR')")
        result = self.db.execute("SELECT * FROM snapshot")
        assert len(result.rows) == 4  # Eve not in snapshot


# =============================================================================
# CTAS with explicit column names
# =============================================================================

class TestCTASWithColumnNames:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, val TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'hello')")
        self.db.execute("INSERT INTO t VALUES (2, 'world')")

    def test_ctas_with_column_names(self):
        self.db.execute("CREATE TABLE t2 (item_id, item_val) AS SELECT id, val FROM t")
        result = self.db.execute("SELECT * FROM t2")
        assert result.columns == ['item_id', 'item_val']
        assert len(result.rows) == 2

    def test_ctas_column_count_mismatch(self):
        with pytest.raises(CatalogError, match="column list"):
            self.db.execute("CREATE TABLE t2 (a, b, c) AS SELECT id, val FROM t")

    def test_ctas_column_names_queryable(self):
        self.db.execute("CREATE TABLE t2 (x, y) AS SELECT id, val FROM t")
        result = self.db.execute("SELECT x FROM t2 WHERE x = 1")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 1


# =============================================================================
# CTAS with views
# =============================================================================

class TestCTASWithViews:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price FLOAT, active INT)")
        self.db.execute("INSERT INTO products VALUES (1, 'Widget', 9.99, 1)")
        self.db.execute("INSERT INTO products VALUES (2, 'Gadget', 24.99, 1)")
        self.db.execute("INSERT INTO products VALUES (3, 'Doodad', 4.99, 0)")

    def test_ctas_from_view(self):
        self.db.execute("CREATE VIEW active AS SELECT id, name, price FROM products WHERE active = 1")
        self.db.execute("CREATE TABLE active_snapshot AS SELECT * FROM active")
        result = self.db.execute("SELECT * FROM active_snapshot")
        assert len(result.rows) == 2

    def test_ctas_from_view_with_where(self):
        self.db.execute("CREATE VIEW active AS SELECT id, name, price FROM products WHERE active = 1")
        self.db.execute("CREATE TABLE expensive AS SELECT name, price FROM active WHERE price > 20")
        result = self.db.execute("SELECT * FROM expensive")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'Gadget'


# =============================================================================
# CTAS error handling
# =============================================================================

class TestCTASErrors:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'test')")

    def test_ctas_table_already_exists(self):
        with pytest.raises(CatalogError, match="already exists"):
            self.db.execute("CREATE TABLE t AS SELECT id FROM t")

    def test_ctas_if_not_exists_on_existing(self):
        result = self.db.execute("CREATE TABLE IF NOT EXISTS t AS SELECT id FROM t")
        assert 'already exists' in result.rows[0][0].lower()

    def test_ctas_conflicts_with_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t")
        with pytest.raises(CatalogError, match="already exists"):
            self.db.execute("CREATE TABLE v AS SELECT id FROM t")

    def test_ctas_source_table_not_found(self):
        with pytest.raises(CatalogError, match="does not exist"):
            self.db.execute("CREATE TABLE t2 AS SELECT id FROM nonexistent")


# =============================================================================
# Type inference in CTAS
# =============================================================================

class TestCTASTypeInference:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE mixed (id INT PRIMARY KEY, name TEXT, score FLOAT, active BOOL)")
        self.db.execute("INSERT INTO mixed VALUES (1, 'Alice', 95.5, true)")
        self.db.execute("INSERT INTO mixed VALUES (2, 'Bob', 88.0, false)")

    def test_int_column_preserved(self):
        self.db.execute("CREATE TABLE t2 AS SELECT id FROM mixed")
        # Can use it as int
        result = self.db.execute("SELECT id FROM t2 WHERE id > 1")
        assert len(result.rows) == 1

    def test_text_column_preserved(self):
        self.db.execute("CREATE TABLE t2 AS SELECT name FROM mixed")
        result = self.db.execute("SELECT name FROM t2 WHERE name = 'Alice'")
        assert len(result.rows) == 1

    def test_float_column_preserved(self):
        self.db.execute("CREATE TABLE t2 AS SELECT score FROM mixed")
        result = self.db.execute("SELECT score FROM t2 WHERE score > 90")
        assert len(result.rows) == 1


# =============================================================================
# CTAS with DISTINCT
# =============================================================================

class TestCTASDistinct:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, category TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'A')")
        self.db.execute("INSERT INTO t VALUES (2, 'B')")
        self.db.execute("INSERT INTO t VALUES (3, 'A')")
        self.db.execute("INSERT INTO t VALUES (4, 'B')")

    def test_ctas_with_distinct(self):
        self.db.execute("CREATE TABLE categories AS SELECT DISTINCT category FROM t")
        result = self.db.execute("SELECT * FROM categories")
        assert len(result.rows) == 2
        cats = sorted([r[0] for r in result.rows])
        assert cats == ['A', 'B']


# =============================================================================
# Passthrough (existing operations still work)
# =============================================================================

class TestPassthrough:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'Alice')")

    def test_select(self):
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 1

    def test_insert(self):
        self.db.execute("INSERT INTO t VALUES (2, 'Bob')")
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 2

    def test_update(self):
        self.db.execute("UPDATE t SET name = 'ALICE' WHERE id = 1")
        result = self.db.execute("SELECT name FROM t WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'

    def test_delete(self):
        self.db.execute("DELETE FROM t WHERE id = 1")
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 0

    def test_create_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t")
        result = self.db.execute("SELECT * FROM v")
        assert len(result.rows) == 1

    def test_drop_view(self):
        self.db.execute("CREATE VIEW v AS SELECT id FROM t")
        self.db.execute("DROP VIEW v")
        assert not self.db.is_view('v')

    def test_show_tables(self):
        result = self.db.execute("SHOW TABLES")
        assert len(result.rows) >= 1

    def test_describe(self):
        result = self.db.execute("DESCRIBE t")
        assert len(result.rows) >= 2

    def test_create_index(self):
        self.db.execute("CREATE INDEX idx_name ON t (name)")
        # no error

    def test_drop_table(self):
        self.db.execute("CREATE TABLE t2 (id INT PRIMARY KEY)")
        self.db.execute("DROP TABLE t2")

    def test_execute_many(self):
        results = self.db.execute_many(
            "INSERT INTO t VALUES (2, 'Bob');"
            "SELECT * FROM t;"
        )
        assert len(results) == 2
        assert len(results[1].rows) == 2


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:

    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE src (id INT PRIMARY KEY, val INT)")
        self.db.execute("INSERT INTO src VALUES (1, 10)")
        self.db.execute("INSERT INTO src VALUES (2, 20)")
        self.db.execute("INSERT INTO src VALUES (3, 30)")

    def test_ctas_then_query_new_table(self):
        self.db.execute("CREATE TABLE dst AS SELECT id, val FROM src WHERE val > 10")
        result = self.db.execute("SELECT * FROM dst ORDER BY id")
        assert len(result.rows) == 2
        assert result.rows[0][1] == 20

    def test_ctas_from_ctas(self):
        """Create a table from another CTAS-created table."""
        self.db.execute("CREATE TABLE mid AS SELECT id, val FROM src WHERE val > 10")
        self.db.execute("CREATE TABLE dst AS SELECT * FROM mid")
        result = self.db.execute("SELECT * FROM dst")
        assert len(result.rows) == 2

    def test_ctas_with_alias_in_select(self):
        self.db.execute("CREATE TABLE dst AS SELECT id, val * 2 AS doubled FROM src")
        result = self.db.execute("SELECT doubled FROM dst WHERE id = 2")
        assert result.rows[0][0] == 40

    def test_multiple_ctas_different_tables(self):
        self.db.execute("CREATE TABLE a AS SELECT id FROM src WHERE val = 10")
        self.db.execute("CREATE TABLE b AS SELECT id FROM src WHERE val = 20")
        self.db.execute("CREATE TABLE c AS SELECT id FROM src WHERE val = 30")
        assert len(self.db.execute("SELECT * FROM a").rows) == 1
        assert len(self.db.execute("SELECT * FROM b").rows) == 1
        assert len(self.db.execute("SELECT * FROM c").rows) == 1

    def test_ctas_with_null_values(self):
        self.db.execute("CREATE TABLE with_nulls (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO with_nulls VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO with_nulls (id) VALUES (2)")
        self.db.execute("CREATE TABLE copy AS SELECT * FROM with_nulls")
        result = self.db.execute("SELECT * FROM copy ORDER BY id")
        assert result.rows[0][1] == 'Alice'
        assert result.rows[1][1] is None

    def test_ctas_table_usable_for_insert(self):
        """Table created via CTAS can receive new inserts."""
        self.db.execute("CREATE TABLE dst AS SELECT id, val FROM src")
        self.db.execute("INSERT INTO dst VALUES (99, 999)")
        result = self.db.execute("SELECT * FROM dst WHERE id = 99")
        assert len(result.rows) == 1
        assert result.rows[0][1] == 999

    def test_ctas_from_view_of_view(self):
        """CTAS from a nested view."""
        self.db.execute("CREATE VIEW v1 AS SELECT id, val FROM src WHERE val > 10")
        self.db.execute("CREATE VIEW v2 AS SELECT id FROM v1")
        self.db.execute("CREATE TABLE from_nested AS SELECT * FROM v2")
        result = self.db.execute("SELECT * FROM from_nested")
        assert len(result.rows) == 2
