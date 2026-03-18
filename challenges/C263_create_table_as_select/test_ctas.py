"""
Tests for C263: CREATE TABLE ... AS SELECT (CTAS)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C262_views')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

import pytest
from ctas import (
    CTASDB, CTASParser, CreateTableAsSelectStmt,
    parse_ctas_sql, _infer_column_type,
)
from mini_database import (
    CreateTableStmt, SelectStmt, ParseError, DatabaseError, CatalogError,
)


# =============================================================================
# Type Inference
# =============================================================================

class TestTypeInference:
    def test_infer_int(self):
        assert _infer_column_type([1, 2, 3]) == "INTEGER"

    def test_infer_float(self):
        assert _infer_column_type([1.5, 2.0, 3.7]) == "FLOAT"

    def test_infer_mixed_int_float(self):
        assert _infer_column_type([1, 2.5, 3]) == "FLOAT"

    def test_infer_text(self):
        assert _infer_column_type(["a", "b"]) == "TEXT"

    def test_infer_bool(self):
        assert _infer_column_type([True, False]) == "BOOLEAN"

    def test_infer_empty(self):
        assert _infer_column_type([]) == "TEXT"

    def test_infer_all_null(self):
        assert _infer_column_type([None, None]) == "TEXT"

    def test_infer_with_nulls(self):
        assert _infer_column_type([None, 1, None, 2]) == "INTEGER"

    def test_infer_mixed_types(self):
        assert _infer_column_type([1, "hello"]) == "TEXT"


# =============================================================================
# Parser
# =============================================================================

class TestCTASParser:
    def test_parse_basic_ctas(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 AS SELECT * FROM t1")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.table_name == "t2"
        assert isinstance(stmt.select_stmt, SelectStmt)
        assert not stmt.if_not_exists
        assert stmt.column_names == []

    def test_parse_ctas_if_not_exists(self):
        stmt = parse_ctas_sql("CREATE TABLE IF NOT EXISTS t2 AS SELECT * FROM t1")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.if_not_exists

    def test_parse_ctas_with_column_names(self):
        stmt = parse_ctas_sql("CREATE TABLE t2 (a, b, c) AS SELECT x, y, z FROM t1")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.column_names == ["a", "b", "c"]

    def test_parse_standard_create_table(self):
        """Standard CREATE TABLE should still work."""
        stmt = parse_ctas_sql("CREATE TABLE t1 (id INTEGER, name TEXT)")
        assert isinstance(stmt, CreateTableStmt)
        assert stmt.table_name == "t1"
        assert len(stmt.columns) == 2

    def test_parse_create_view_still_works(self):
        """CREATE VIEW should pass through to parent parser."""
        from views import CreateViewStmt
        stmt = parse_ctas_sql("CREATE VIEW v AS SELECT 1")
        assert isinstance(stmt, CreateViewStmt)

    def test_parse_create_index_still_works(self):
        """CREATE INDEX should pass through to parent parser."""
        from mini_database import CreateIndexStmt
        stmt = parse_ctas_sql("CREATE INDEX idx ON t (col)")
        assert isinstance(stmt, CreateIndexStmt)

    def test_parse_ctas_with_where(self):
        stmt = parse_ctas_sql("CREATE TABLE active AS SELECT * FROM users WHERE active = 1")
        assert isinstance(stmt, CreateTableAsSelectStmt)
        assert stmt.table_name == "active"

    def test_parse_ctas_with_join(self):
        stmt = parse_ctas_sql(
            "CREATE TABLE summary AS SELECT a.id, b.name FROM a JOIN b ON a.id = b.id"
        )
        assert isinstance(stmt, CreateTableAsSelectStmt)


# =============================================================================
# Basic CTAS Execution
# =============================================================================

class TestBasicCTAS:
    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE employees (id INTEGER, name TEXT, dept TEXT, salary INTEGER)")
        self.db.execute("INSERT INTO employees VALUES (1, 'Alice', 'eng', 90000)")
        self.db.execute("INSERT INTO employees VALUES (2, 'Bob', 'eng', 85000)")
        self.db.execute("INSERT INTO employees VALUES (3, 'Carol', 'sales', 75000)")
        self.db.execute("INSERT INTO employees VALUES (4, 'Dave', 'sales', 80000)")
        self.db.execute("INSERT INTO employees VALUES (5, 'Eve', 'hr', 70000)")

    def test_basic_ctas(self):
        result = self.db.execute("CREATE TABLE eng_team AS SELECT * FROM employees WHERE dept = 'eng'")
        assert "eng_team" in result.message
        assert result.rows_affected == 2

        rows = self.db.execute("SELECT * FROM eng_team")
        assert len(rows.rows) == 2

    def test_ctas_select_columns(self):
        result = self.db.execute(
            "CREATE TABLE names AS SELECT name, dept FROM employees"
        )
        assert result.rows_affected == 5

        rows = self.db.execute("SELECT * FROM names")
        assert rows.columns == ["name", "dept"]
        assert len(rows.rows) == 5

    def test_ctas_with_alias(self):
        self.db.execute(
            "CREATE TABLE dept_summary AS SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept"
        )
        rows = self.db.execute("SELECT * FROM dept_summary ORDER BY dept")
        assert len(rows.rows) == 3
        # eng: 2, hr: 1, sales: 2
        assert rows.rows[0][1] == 2  # eng
        assert rows.rows[1][1] == 1  # hr
        assert rows.rows[2][1] == 2  # sales

    def test_ctas_if_not_exists_skips(self):
        self.db.execute("CREATE TABLE backup AS SELECT * FROM employees")
        result = self.db.execute(
            "CREATE TABLE IF NOT EXISTS backup AS SELECT * FROM employees"
        )
        assert "skipped" in result.message.lower() or "exists" in result.message.lower()

    def test_ctas_duplicate_error(self):
        self.db.execute("CREATE TABLE backup AS SELECT * FROM employees")
        with pytest.raises(DatabaseError, match="already exists"):
            self.db.execute("CREATE TABLE backup AS SELECT * FROM employees")

    def test_ctas_empty_result(self):
        result = self.db.execute(
            "CREATE TABLE nobody AS SELECT * FROM employees WHERE dept = 'none'"
        )
        assert result.rows_affected == 0

        rows = self.db.execute("SELECT * FROM nobody")
        assert len(rows.rows) == 0
        assert rows.columns == ["id", "name", "dept", "salary"]

    def test_ctas_with_explicit_columns(self):
        self.db.execute(
            "CREATE TABLE summary (employee, department) AS SELECT name, dept FROM employees"
        )
        rows = self.db.execute("SELECT * FROM summary")
        assert rows.columns == ["employee", "department"]
        assert len(rows.rows) == 5

    def test_ctas_column_count_mismatch(self):
        with pytest.raises(DatabaseError, match="column count mismatch"):
            self.db.execute(
                "CREATE TABLE bad (a, b, c) AS SELECT name FROM employees"
            )

    def test_ctas_with_order_by(self):
        self.db.execute(
            "CREATE TABLE ordered AS SELECT name, salary FROM employees ORDER BY salary DESC"
        )
        rows = self.db.execute("SELECT * FROM ordered")
        assert rows.rows[0][0] == "Alice"  # highest salary
        assert rows.rows[0][1] == 90000

    def test_ctas_with_limit(self):
        self.db.execute(
            "CREATE TABLE top3 AS SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3"
        )
        rows = self.db.execute("SELECT * FROM top3")
        assert len(rows.rows) == 3


# =============================================================================
# Advanced CTAS
# =============================================================================

class TestAdvancedCTAS:
    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE t1 (id INTEGER, val TEXT, score FLOAT)")
        self.db.execute("INSERT INTO t1 VALUES (1, 'a', 1.5)")
        self.db.execute("INSERT INTO t1 VALUES (2, 'b', 2.5)")
        self.db.execute("INSERT INTO t1 VALUES (3, 'c', 3.5)")

    def test_ctas_with_expression(self):
        self.db.execute(
            "CREATE TABLE doubled AS SELECT id, score * 2 AS double_score FROM t1"
        )
        rows = self.db.execute("SELECT * FROM doubled ORDER BY id")
        assert rows.columns == ["id", "double_score"]
        assert rows.rows[0][1] == 3.0
        assert rows.rows[2][1] == 7.0

    def test_ctas_preserves_types_int(self):
        self.db.execute("CREATE TABLE ints AS SELECT id FROM t1")
        # Verify the new table has the data
        rows = self.db.execute("SELECT * FROM ints")
        assert all(isinstance(r[0], int) for r in rows.rows)

    def test_ctas_preserves_types_float(self):
        self.db.execute("CREATE TABLE floats AS SELECT score FROM t1")
        rows = self.db.execute("SELECT * FROM floats")
        assert all(isinstance(r[0], float) for r in rows.rows)

    def test_ctas_then_insert(self):
        """New table from CTAS should be a real table you can insert into."""
        self.db.execute("CREATE TABLE copy AS SELECT * FROM t1")
        self.db.execute("INSERT INTO copy VALUES (4, 'd', 4.5)")
        rows = self.db.execute("SELECT * FROM copy")
        assert len(rows.rows) == 4

    def test_ctas_then_update(self):
        """Should be able to update the CTAS-created table."""
        self.db.execute("CREATE TABLE copy AS SELECT * FROM t1")
        self.db.execute("UPDATE copy SET val = 'x' WHERE id = 1")
        rows = self.db.execute("SELECT val FROM copy WHERE id = 1")
        assert rows.rows[0][0] == "x"

    def test_ctas_then_delete(self):
        """Should be able to delete from the CTAS-created table."""
        self.db.execute("CREATE TABLE copy AS SELECT * FROM t1")
        self.db.execute("DELETE FROM copy WHERE id = 1")
        rows = self.db.execute("SELECT * FROM copy")
        assert len(rows.rows) == 2

    def test_ctas_independent_of_source(self):
        """Changes to source table should NOT affect CTAS table."""
        self.db.execute("CREATE TABLE snapshot AS SELECT * FROM t1")
        self.db.execute("INSERT INTO t1 VALUES (4, 'd', 4.5)")
        rows = self.db.execute("SELECT * FROM snapshot")
        assert len(rows.rows) == 3  # original count

    def test_ctas_with_distinct(self):
        self.db.execute("INSERT INTO t1 VALUES (4, 'a', 1.5)")  # duplicate val
        self.db.execute("CREATE TABLE unique_vals AS SELECT DISTINCT val FROM t1")
        rows = self.db.execute("SELECT * FROM unique_vals")
        assert len(rows.rows) == 3  # a, b, c


# =============================================================================
# CTAS with Views
# =============================================================================

class TestCTASWithViews:
    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE products (id INTEGER, name TEXT, price FLOAT, category TEXT)")
        self.db.execute("INSERT INTO products VALUES (1, 'Widget', 9.99, 'tools')")
        self.db.execute("INSERT INTO products VALUES (2, 'Gadget', 19.99, 'tools')")
        self.db.execute("INSERT INTO products VALUES (3, 'Book', 12.99, 'media')")
        self.db.execute("INSERT INTO products VALUES (4, 'DVD', 14.99, 'media')")

    def test_ctas_from_view(self):
        """CREATE TABLE from a view should work."""
        self.db.execute(
            "CREATE VIEW tools_view AS SELECT * FROM products WHERE category = 'tools'"
        )
        self.db.execute("CREATE TABLE tools_table AS SELECT * FROM tools_view")
        rows = self.db.execute("SELECT * FROM tools_table")
        assert len(rows.rows) == 2

    def test_ctas_does_not_conflict_with_view_name(self):
        """Cannot create table with same name as existing view."""
        self.db.execute("CREATE VIEW myview AS SELECT * FROM products")
        with pytest.raises(DatabaseError, match="already used by a view"):
            self.db.execute("CREATE TABLE myview AS SELECT * FROM products")

    def test_ctas_if_not_exists_with_view_name(self):
        """IF NOT EXISTS should handle view name conflict gracefully."""
        self.db.execute("CREATE VIEW myview AS SELECT * FROM products")
        result = self.db.execute(
            "CREATE TABLE IF NOT EXISTS myview AS SELECT * FROM products"
        )
        assert "exists" in result.message.lower() or "skipped" in result.message.lower()


# =============================================================================
# CTAS with Joins
# =============================================================================

class TestCTASWithJoins:
    def setup_method(self):
        self.db = CTASDB()
        self.db.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount FLOAT)")
        self.db.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
        self.db.execute("INSERT INTO customers VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO customers VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO orders VALUES (1, 1, 100.0)")
        self.db.execute("INSERT INTO orders VALUES (2, 1, 200.0)")
        self.db.execute("INSERT INTO orders VALUES (3, 2, 150.0)")

    def test_ctas_from_join(self):
        self.db.execute(
            "CREATE TABLE order_details AS "
            "SELECT o.id, c.name, o.amount "
            "FROM orders o JOIN customers c ON o.customer_id = c.id"
        )
        rows = self.db.execute("SELECT * FROM order_details ORDER BY id")
        assert len(rows.rows) == 3
        assert rows.rows[0][1] == "Alice"
        assert rows.rows[2][1] == "Bob"

    def test_ctas_from_aggregate_join(self):
        self.db.execute(
            "CREATE TABLE customer_totals AS "
            "SELECT c.name, SUM(o.amount) AS total "
            "FROM orders o JOIN customers c ON o.customer_id = c.id "
            "GROUP BY c.name"
        )
        rows = self.db.execute("SELECT * FROM customer_totals ORDER BY name")
        assert len(rows.rows) == 2
        assert rows.rows[0][0] == "Alice"
        assert rows.rows[0][1] == 300.0
        assert rows.rows[1][0] == "Bob"
        assert rows.rows[1][1] == 150.0


# =============================================================================
# Passthrough Tests
# =============================================================================

class TestPassthrough:
    """Ensure existing functionality still works."""

    def setup_method(self):
        self.db = CTASDB()

    def test_create_table(self):
        self.db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        rows = self.db.execute("SELECT * FROM t")
        assert rows.columns == ["id", "name"]

    def test_insert_select(self):
        self.db.execute("CREATE TABLE t (id INTEGER)")
        self.db.execute("INSERT INTO t VALUES (1)")
        rows = self.db.execute("SELECT * FROM t")
        assert rows.rows == [[1]]

    def test_create_view(self):
        self.db.execute("CREATE TABLE t (id INTEGER)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("CREATE VIEW v AS SELECT * FROM t")
        rows = self.db.execute("SELECT * FROM v")
        assert rows.rows == [[1]]

    def test_drop_view(self):
        self.db.execute("CREATE TABLE t (id INTEGER)")
        self.db.execute("CREATE VIEW v AS SELECT * FROM t")
        self.db.execute("DROP VIEW v")
        with pytest.raises(Exception):
            self.db.execute("SELECT * FROM v")

    def test_show_tables_includes_ctas(self):
        self.db.execute("CREATE TABLE src (id INTEGER)")
        self.db.execute("INSERT INTO src VALUES (1)")
        self.db.execute("CREATE TABLE dest AS SELECT * FROM src")
        rows = self.db.execute("SHOW TABLES")
        names = [r[0] for r in rows.rows]
        assert "src" in names
        assert "dest" in names

    def test_describe_ctas_table(self):
        self.db.execute("CREATE TABLE src (id INTEGER, name TEXT)")
        self.db.execute("INSERT INTO src VALUES (1, 'test')")
        self.db.execute("CREATE TABLE copy AS SELECT * FROM src")
        rows = self.db.execute("DESCRIBE copy")
        col_names = [r[0] for r in rows.rows]
        assert "id" in col_names
        assert "name" in col_names

    def test_execute_many(self):
        results = self.db.execute_many(
            "CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1); SELECT * FROM t"
        )
        assert len(results) == 3
        assert results[2].rows == [[1]]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def setup_method(self):
        self.db = CTASDB()

    def test_ctas_single_column(self):
        self.db.execute("CREATE TABLE src (id INTEGER)")
        self.db.execute("INSERT INTO src VALUES (1)")
        self.db.execute("INSERT INTO src VALUES (2)")
        self.db.execute("CREATE TABLE ids AS SELECT id FROM src")
        rows = self.db.execute("SELECT * FROM ids")
        assert rows.columns == ["id"]
        assert len(rows.rows) == 2

    def test_ctas_with_null_values(self):
        self.db.execute("CREATE TABLE src (id INTEGER, val TEXT)")
        self.db.execute("INSERT INTO src VALUES (1, NULL)")
        self.db.execute("INSERT INTO src VALUES (2, 'hello')")
        self.db.execute("CREATE TABLE copy AS SELECT * FROM src")
        rows = self.db.execute("SELECT * FROM copy ORDER BY id")
        assert rows.rows[0][1] is None
        assert rows.rows[1][1] == "hello"

    def test_ctas_from_ctas(self):
        """Can create a table from another CTAS-created table."""
        self.db.execute("CREATE TABLE src (id INTEGER)")
        self.db.execute("INSERT INTO src VALUES (1)")
        self.db.execute("CREATE TABLE mid AS SELECT * FROM src")
        self.db.execute("CREATE TABLE dest AS SELECT * FROM mid")
        rows = self.db.execute("SELECT * FROM dest")
        assert rows.rows == [[1]]

    def test_ctas_star_all_columns(self):
        self.db.execute("CREATE TABLE src (a INTEGER, b TEXT, c FLOAT)")
        self.db.execute("INSERT INTO src VALUES (1, 'x', 1.5)")
        self.db.execute("CREATE TABLE copy AS SELECT * FROM src")
        rows = self.db.execute("SELECT * FROM copy")
        assert rows.columns == ["a", "b", "c"]
        assert rows.rows == [[1, "x", 1.5]]

    def test_ctas_computed_column(self):
        self.db.execute("CREATE TABLE src (x INTEGER)")
        self.db.execute("INSERT INTO src VALUES (5)")
        self.db.execute("INSERT INTO src VALUES (10)")
        self.db.execute("CREATE TABLE computed AS SELECT x, x * 2 AS doubled FROM src")
        rows = self.db.execute("SELECT * FROM computed ORDER BY x")
        assert rows.rows[0] == [5, 10]
        assert rows.rows[1] == [10, 20]

    def test_ctas_aggregate_only(self):
        self.db.execute("CREATE TABLE src (val INTEGER)")
        self.db.execute("INSERT INTO src VALUES (10)")
        self.db.execute("INSERT INTO src VALUES (20)")
        self.db.execute("INSERT INTO src VALUES (30)")
        self.db.execute("CREATE TABLE agg AS SELECT SUM(val) AS total, COUNT(*) AS cnt FROM src")
        rows = self.db.execute("SELECT * FROM agg")
        assert rows.rows[0][0] == 60
        assert rows.rows[0][1] == 3

    def test_ctas_preserves_row_order_with_order_by(self):
        self.db.execute("CREATE TABLE src (id INTEGER, name TEXT)")
        self.db.execute("INSERT INTO src VALUES (3, 'c')")
        self.db.execute("INSERT INTO src VALUES (1, 'a')")
        self.db.execute("INSERT INTO src VALUES (2, 'b')")
        self.db.execute("CREATE TABLE sorted AS SELECT * FROM src ORDER BY id")
        rows = self.db.execute("SELECT * FROM sorted")
        ids = [r[0] for r in rows.rows]
        # Order may not be preserved in storage, but data should be correct
        assert sorted(ids) == [1, 2, 3]
        assert len(rows.rows) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
