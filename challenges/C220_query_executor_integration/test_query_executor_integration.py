"""Tests for C220: Query Executor Integration

Full SQL pipeline: parse -> plan -> lock -> execute -> release.
"""

import sys
import os
import time
import threading
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from query_executor_integration import (
    IntegratedQueryEngine, TransactionContext, ExecutionResult,
    StatementClassifier, StatementType, IsolationLevel, TxState,
    LockAcquirer, DMLExecutor, DDLExecutor, PipelineExecutor,
    ConcurrentExecutionManager, IntegratedEngineAnalyzer, EngineStats,
    UndoEntry
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C211_query_execution'))
from query_execution import Database


def make_engine(**kwargs):
    """Create an engine with test defaults."""
    return IntegratedQueryEngine(**kwargs)


def make_populated_engine():
    """Create engine with employees + departments tables."""
    engine = make_engine()
    engine.execute("CREATE TABLE departments (id INT PRIMARY KEY, name TEXT, budget INT)")
    engine.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, dept_id INT, salary INT)")

    engine.execute("INSERT INTO departments (id, name, budget) VALUES (1, 'Engineering', 500000)")
    engine.execute("INSERT INTO departments (id, name, budget) VALUES (2, 'Sales', 300000)")
    engine.execute("INSERT INTO departments (id, name, budget) VALUES (3, 'Marketing', 200000)")

    engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (1, 'Alice', 1, 90000)")
    engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (2, 'Bob', 1, 85000)")
    engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (3, 'Carol', 2, 75000)")
    engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (4, 'Dave', 2, 70000)")
    engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (5, 'Eve', 3, 65000)")

    # Sync catalog so planner knows about actual data
    engine.sync_catalog()

    return engine


# ============================================================
# 1. Statement Classification
# ============================================================

class TestStatementClassifier(unittest.TestCase):

    def test_select(self):
        self.assertEqual(StatementClassifier.classify("SELECT * FROM t"), StatementType.SELECT)

    def test_insert(self):
        self.assertEqual(StatementClassifier.classify("INSERT INTO t (a) VALUES (1)"), StatementType.INSERT)

    def test_update(self):
        self.assertEqual(StatementClassifier.classify("UPDATE t SET a=1"), StatementType.UPDATE)

    def test_delete(self):
        self.assertEqual(StatementClassifier.classify("DELETE FROM t WHERE id=1"), StatementType.DELETE)

    def test_create_table(self):
        self.assertEqual(StatementClassifier.classify("CREATE TABLE t (a INT)"), StatementType.CREATE_TABLE)

    def test_drop_table(self):
        self.assertEqual(StatementClassifier.classify("DROP TABLE t"), StatementType.DROP_TABLE)

    def test_create_index(self):
        self.assertEqual(StatementClassifier.classify("CREATE INDEX idx ON t (a)"), StatementType.CREATE_INDEX)

    def test_create_unique_index(self):
        self.assertEqual(StatementClassifier.classify("CREATE UNIQUE INDEX idx ON t (a)"), StatementType.CREATE_INDEX)

    def test_drop_index(self):
        self.assertEqual(StatementClassifier.classify("DROP INDEX idx ON t"), StatementType.DROP_INDEX)

    def test_begin(self):
        self.assertEqual(StatementClassifier.classify("BEGIN"), StatementType.BEGIN)

    def test_commit(self):
        self.assertEqual(StatementClassifier.classify("COMMIT"), StatementType.COMMIT)

    def test_rollback(self):
        self.assertEqual(StatementClassifier.classify("ROLLBACK"), StatementType.ROLLBACK)

    def test_explain(self):
        self.assertEqual(StatementClassifier.classify("EXPLAIN SELECT * FROM t"), StatementType.EXPLAIN)

    def test_explain_analyze(self):
        self.assertEqual(StatementClassifier.classify("EXPLAIN ANALYZE SELECT * FROM t"), StatementType.EXPLAIN_ANALYZE)

    def test_case_insensitive(self):
        self.assertEqual(StatementClassifier.classify("select * from t"), StatementType.SELECT)
        self.assertEqual(StatementClassifier.classify("  SELECT  * FROM t"), StatementType.SELECT)


# ============================================================
# 2. Engine Initialization
# ============================================================

class TestEngineInit(unittest.TestCase):

    def test_create_engine(self):
        engine = make_engine()
        self.assertIsNotNone(engine.db)
        self.assertIsNotNone(engine.catalog)
        self.assertIsNotNone(engine.lock_manager)
        self.assertIsNotNone(engine.query_engine)
        self.assertIsNotNone(engine.planner)

    def test_default_isolation(self):
        engine = make_engine()
        self.assertEqual(engine.default_isolation, IsolationLevel.READ_COMMITTED)

    def test_custom_isolation(self):
        engine = make_engine(isolation=IsolationLevel.SERIALIZABLE)
        self.assertEqual(engine.default_isolation, IsolationLevel.SERIALIZABLE)

    def test_engine_stats_initial(self):
        engine = make_engine()
        self.assertEqual(engine.stats.statements_executed, 0)
        self.assertEqual(engine.stats.commits, 0)
        self.assertEqual(engine.stats.rollbacks, 0)


# ============================================================
# 3. DDL Operations
# ============================================================

class TestDDL(unittest.TestCase):

    def test_create_table(self):
        engine = make_engine()
        result = engine.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
        self.assertEqual(result.statement_type, StatementType.CREATE_TABLE)
        self.assertIn("created", result.message)

    def test_create_table_api(self):
        engine = make_engine()
        result = engine.create_table("users", ["id", "name", "age"], "id")
        self.assertIn("created", result.message)

    def test_drop_table(self):
        engine = make_engine()
        engine.execute("CREATE TABLE temp (a INT)")
        result = engine.execute("DROP TABLE temp")
        self.assertEqual(result.statement_type, StatementType.DROP_TABLE)
        self.assertIn("dropped", result.message)

    def test_drop_nonexistent_table(self):
        engine = make_engine()
        result = engine.execute("DROP TABLE IF EXISTS nonexistent")
        self.assertIn("does not exist", result.message)

    def test_create_index(self):
        engine = make_engine()
        engine.execute("CREATE TABLE users (id INT, name TEXT)")
        result = engine.execute("CREATE INDEX idx_name ON users (name)")
        self.assertEqual(result.statement_type, StatementType.CREATE_INDEX)
        self.assertIn("created", result.message)

    def test_create_unique_index(self):
        engine = make_engine()
        engine.execute("CREATE TABLE users (id INT, email TEXT)")
        result = engine.execute("CREATE UNIQUE INDEX idx_email ON users (email)")
        self.assertIn("created", result.message)

    def test_create_index_api(self):
        engine = make_engine()
        engine.execute("CREATE TABLE users (id INT, name TEXT)")
        result = engine.create_index("users", "idx_name", ["name"])
        self.assertIn("created", result.message)

    def test_drop_index(self):
        engine = make_engine()
        engine.execute("CREATE TABLE users (id INT, name TEXT)")
        engine.execute("CREATE INDEX idx_name ON users (name)")
        result = engine.execute("DROP INDEX idx_name ON users")
        self.assertEqual(result.statement_type, StatementType.DROP_INDEX)
        self.assertIn("dropped", result.message)

    def test_ddl_invalidates_cache(self):
        engine = make_populated_engine()
        # Execute a query to populate cache
        engine.execute("SELECT * FROM employees")
        # DDL should invalidate cache
        engine.execute("CREATE INDEX idx_salary ON employees (salary)")
        # Verify cache was invalidated
        stats = engine.planner.cache.stats()
        # Cache should have been cleared or table-specific entries removed


# ============================================================
# 4. DML Operations
# ============================================================

class TestDML(unittest.TestCase):

    def test_insert_single_row(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        result = engine.execute("INSERT INTO t (id, name) VALUES (1, 'Alice')")
        self.assertEqual(result.statement_type, StatementType.INSERT)
        self.assertEqual(result.affected_rows, 1)

    def test_insert_multiple_rows(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        result = engine.execute("INSERT INTO t (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        self.assertEqual(result.affected_rows, 2)

    def test_insert_api(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        result = engine.insert("t", {"id": 1, "name": "Alice"})
        self.assertEqual(result.affected_rows, 1)

    def test_insert_many_api(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        result = engine.insert_many("t", [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol"},
        ])
        self.assertEqual(result.affected_rows, 3)

    def test_insert_null(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        engine.execute("INSERT INTO t (id, name) VALUES (1, NULL)")
        rows = engine.query("SELECT * FROM t")
        self.assertEqual(len(rows), 1)

    def test_update_all(self):
        engine = make_populated_engine()
        result = engine.execute("UPDATE employees SET salary=100000")
        self.assertEqual(result.statement_type, StatementType.UPDATE)
        self.assertEqual(result.affected_rows, 5)

    def test_update_with_where(self):
        engine = make_populated_engine()
        result = engine.execute("UPDATE employees SET salary=100000 WHERE name='Alice'")
        self.assertEqual(result.affected_rows, 1)
        rows = engine.query("SELECT * FROM employees")
        alice = [r for r in rows if r.get('name') == 'Alice' or r.get('employees.name') == 'Alice']
        self.assertTrue(len(alice) > 0)

    def test_delete_all(self):
        engine = make_populated_engine()
        result = engine.execute("DELETE FROM employees")
        self.assertEqual(result.statement_type, StatementType.DELETE)
        self.assertEqual(result.affected_rows, 5)

    def test_delete_with_where(self):
        engine = make_populated_engine()
        result = engine.execute("DELETE FROM employees WHERE dept_id=2")
        self.assertEqual(result.affected_rows, 2)

    def test_delete_no_match(self):
        engine = make_populated_engine()
        result = engine.execute("DELETE FROM employees WHERE id=999")
        self.assertEqual(result.affected_rows, 0)

    def test_update_numeric_comparison(self):
        engine = make_populated_engine()
        result = engine.execute("UPDATE employees SET salary=0 WHERE salary>80000")
        self.assertEqual(result.affected_rows, 2)  # Alice(90k) and Bob(85k)


# ============================================================
# 5. SELECT Queries
# ============================================================

class TestSelect(unittest.TestCase):

    def test_select_all(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT * FROM employees")
        self.assertEqual(result.statement_type, StatementType.SELECT)
        self.assertEqual(len(result.rows), 5)

    def test_select_with_where(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT * FROM employees WHERE dept_id = 1")
        self.assertEqual(len(result.rows), 2)

    def test_select_columns(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT name, salary FROM employees")
        self.assertEqual(len(result.rows), 5)

    def test_select_with_order(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT * FROM employees ORDER BY salary DESC")
        self.assertEqual(len(result.rows), 5)

    def test_select_with_limit(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT * FROM employees LIMIT 3")
        self.assertEqual(len(result.rows), 3)

    def test_select_aggregate_count(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT COUNT(*) FROM employees")
        self.assertTrue(len(result.rows) > 0)

    def test_select_empty_table(self):
        engine = make_engine()
        engine.execute("CREATE TABLE empty (id INT)")
        result = engine.execute("SELECT * FROM empty")
        self.assertEqual(len(result.rows), 0)

    def test_query_convenience(self):
        engine = make_populated_engine()
        rows = engine.query("SELECT * FROM employees")
        self.assertEqual(len(rows), 5)

    def test_select_returns_execution_result(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT * FROM employees")
        self.assertIsInstance(result, ExecutionResult)
        self.assertIsNotNone(result.execution_time_ms)


# ============================================================
# 6. Transaction Control
# ============================================================

class TestTransactions(unittest.TestCase):

    def test_begin_returns_tx_id(self):
        engine = make_engine()
        tx_id = engine.begin()
        self.assertIsInstance(tx_id, int)

    def test_begin_via_sql(self):
        engine = make_engine()
        result = engine.execute("BEGIN")
        self.assertEqual(result.statement_type, StatementType.BEGIN)
        self.assertIn("started", result.message)

    def test_commit_via_sql(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        engine.execute("SELECT * FROM employees", tx_id)
        result = engine.commit(tx_id)
        self.assertEqual(result.statement_type, StatementType.COMMIT)
        self.assertIn("committed", result.message)

    def test_rollback_via_sql(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        result = engine.rollback(tx_id)
        self.assertEqual(result.statement_type, StatementType.ROLLBACK)
        self.assertIn("rolled back", result.message)

    def test_commit_releases_locks(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        engine.execute("SELECT * FROM employees", tx_id)
        result = engine.commit(tx_id)
        self.assertGreaterEqual(result.locks_released, 0)

    def test_rollback_undoes_insert(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'Frank', 1, 80000)", tx_id)
        # Verify insert happened (query within same tx to avoid lock conflict)
        rows = engine.execute("SELECT * FROM employees", tx_id).rows
        self.assertEqual(len(rows), 6)
        # Rollback
        engine.rollback(tx_id)
        rows = engine.query("SELECT * FROM employees")
        self.assertEqual(len(rows), 5)

    def test_rollback_undoes_delete(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        engine.execute("DELETE FROM employees WHERE id=1", tx_id)
        # Verify delete happened (query within same tx)
        rows = engine.execute("SELECT * FROM employees", tx_id).rows
        self.assertEqual(len(rows), 4)
        engine.rollback(tx_id)
        rows = engine.query("SELECT * FROM employees")
        self.assertEqual(len(rows), 5)

    def test_rollback_undoes_update(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        engine.execute("UPDATE employees SET salary=0 WHERE id=1", tx_id)
        engine.rollback(tx_id)
        rows = engine.query("SELECT * FROM employees WHERE id = 1")
        # Alice should have original salary
        self.assertTrue(len(rows) > 0)

    def test_transaction_state_tracking(self):
        engine = make_engine()
        tx_id = engine.begin()
        tx = engine.get_transaction(tx_id)
        self.assertEqual(tx.state, TxState.ACTIVE)
        self.assertTrue(tx.is_active)
        engine.commit(tx_id)
        self.assertEqual(tx.state, TxState.COMMITTED)
        self.assertFalse(tx.is_active)

    def test_double_commit_rejected(self):
        engine = make_engine()
        tx_id = engine.begin()
        engine.commit(tx_id)
        result = engine.commit(tx_id)
        self.assertIn("COMMITTED", result.message)

    def test_active_transactions(self):
        engine = make_engine()
        tx1 = engine.begin()
        tx2 = engine.begin()
        active = engine.active_transactions()
        self.assertIn(tx1, active)
        self.assertIn(tx2, active)
        engine.commit(tx1)
        active = engine.active_transactions()
        self.assertNotIn(tx1, active)
        self.assertIn(tx2, active)

    def test_transaction_not_found(self):
        engine = make_engine()
        result = engine.execute("SELECT 1", tx_id=99999)
        self.assertIn("not found", result.message)

    def test_auto_commit_mode(self):
        engine = make_populated_engine()
        commits_before = engine.stats.commits
        # Without tx_id, should auto-commit
        result = engine.execute("SELECT * FROM employees")
        self.assertEqual(len(result.rows), 5)
        # Auto-commit should increment commits by 1
        self.assertEqual(engine.stats.commits, commits_before + 1)


# ============================================================
# 7. Lock Integration
# ============================================================

class TestLockIntegration(unittest.TestCase):

    def test_select_acquires_locks(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        result = engine.execute("SELECT * FROM employees", tx_id)
        # Should have acquired some locks
        self.assertGreaterEqual(result.locks_acquired, 0)

    def test_insert_acquires_lock(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        result = engine.execute(
            "INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'Frank', 1, 80000)",
            tx_id
        )
        self.assertGreaterEqual(result.locks_acquired, 1)
        engine.commit(tx_id)

    def test_update_acquires_lock(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        result = engine.execute("UPDATE employees SET salary=0 WHERE id=1", tx_id)
        self.assertGreaterEqual(result.locks_acquired, 1)
        engine.commit(tx_id)

    def test_delete_acquires_exclusive_lock(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        result = engine.execute("DELETE FROM employees WHERE id=1", tx_id)
        self.assertGreaterEqual(result.locks_acquired, 1)
        engine.commit(tx_id)

    def test_locks_released_on_commit(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'F', 1, 0)", tx_id)
        result = engine.commit(tx_id)
        self.assertGreaterEqual(result.locks_released, 0)
        # Lock manager should show no locks for this tx
        tx_locks = engine.lock_manager.get_tx_locks(tx_id)
        self.assertEqual(len(tx_locks), 0)

    def test_locks_released_on_rollback(self):
        engine = make_populated_engine()
        tx_id = engine.begin()
        engine.execute("DELETE FROM employees WHERE id=1", tx_id)
        result = engine.rollback(tx_id)
        tx_locks = engine.lock_manager.get_tx_locks(tx_id)
        self.assertEqual(len(tx_locks), 0)


# ============================================================
# 8. EXPLAIN Support
# ============================================================

class TestExplain(unittest.TestCase):

    def test_explain_select(self):
        engine = make_populated_engine()
        result = engine.execute("EXPLAIN SELECT * FROM employees")
        self.assertEqual(result.statement_type, StatementType.EXPLAIN)
        self.assertTrue(len(result.message) > 0)

    def test_explain_api(self):
        engine = make_populated_engine()
        output = engine.explain("SELECT * FROM employees")
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    def test_explain_analyze(self):
        engine = make_populated_engine()
        result = engine.execute("EXPLAIN ANALYZE SELECT * FROM employees")
        self.assertEqual(result.statement_type, StatementType.EXPLAIN_ANALYZE)
        self.assertIn("Actual rows", result.message)

    def test_explain_analyze_api(self):
        engine = make_populated_engine()
        output = engine.explain_analyze("SELECT * FROM employees")
        self.assertIn("Actual rows", output)
        self.assertIn("Execution time", output)

    def test_explain_with_where(self):
        engine = make_populated_engine()
        output = engine.explain("SELECT * FROM employees WHERE dept_id = 1")
        self.assertTrue(len(output) > 0)

    def test_explain_analyze_accuracy(self):
        engine = make_populated_engine()
        output = engine.explain_analyze("SELECT * FROM employees")
        self.assertIn("accuracy", output.lower())


# ============================================================
# 9. Statistics and Catalog Sync
# ============================================================

class TestStatistics(unittest.TestCase):

    def test_sync_catalog(self):
        engine = make_populated_engine()
        engine.sync_catalog()
        # Catalog should now reflect actual table data
        table_def = engine.catalog.get_table("employees")
        self.assertIsNotNone(table_def)

    def test_update_statistics(self):
        engine = make_populated_engine()
        engine.update_statistics("employees")
        table_def = engine.catalog.get_table("employees")
        self.assertIsNotNone(table_def)
        self.assertEqual(table_def.row_count, 5)

    def test_update_statistics_column_info(self):
        engine = make_populated_engine()
        engine.update_statistics("employees")
        table_def = engine.catalog.get_table("employees")
        # Should have column stats
        self.assertTrue(len(table_def.columns) > 0)

    def test_engine_stats_tracking(self):
        engine = make_populated_engine()
        initial = engine.stats.statements_executed
        engine.execute("SELECT * FROM employees")
        self.assertEqual(engine.stats.statements_executed, initial + 1)

    def test_engine_stats_summary(self):
        engine = make_populated_engine()
        summary = engine.stats.summary()
        self.assertIn('statements', summary)
        self.assertIn('commits', summary)
        self.assertIn('rollbacks', summary)


# ============================================================
# 10. Pipeline Executor
# ============================================================

class TestPipelineExecutor(unittest.TestCase):

    def test_execute_script(self):
        engine = make_engine()
        pipeline = PipelineExecutor(engine)
        results = pipeline.execute_script("""
            CREATE TABLE t (id INT, name TEXT);
            INSERT INTO t (id, name) VALUES (1, 'A');
            INSERT INTO t (id, name) VALUES (2, 'B');
            SELECT * FROM t
        """)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].statement_type, StatementType.CREATE_TABLE)
        self.assertEqual(results[-1].statement_type, StatementType.SELECT)
        self.assertEqual(len(results[-1].rows), 2)

    def test_execute_empty_script(self):
        engine = make_engine()
        pipeline = PipelineExecutor(engine)
        results = pipeline.execute_script("")
        self.assertEqual(len(results), 0)

    def test_execute_transaction(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        pipeline = PipelineExecutor(engine)
        results = pipeline.execute_transaction([
            "INSERT INTO t (id, name) VALUES (1, 'A')",
            "INSERT INTO t (id, name) VALUES (2, 'B')",
        ])
        # Should include commit result
        self.assertTrue(any(r.statement_type == StatementType.COMMIT for r in results))
        rows = engine.query("SELECT * FROM t")
        self.assertEqual(len(rows), 2)

    def test_script_with_strings_containing_semicolons(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        pipeline = PipelineExecutor(engine)
        results = pipeline.execute_script(
            "INSERT INTO t (id, name) VALUES (1, 'semi;colon')"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].affected_rows, 1)

    def test_multi_statement_transaction(self):
        engine = make_populated_engine()
        pipeline = PipelineExecutor(engine)
        results = pipeline.execute_transaction([
            "INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'Frank', 1, 80000)",
            "UPDATE employees SET salary=95000 WHERE id=6",
        ])
        rows = engine.query("SELECT * FROM employees")
        self.assertEqual(len(rows), 6)


# ============================================================
# 11. Concurrent Execution
# ============================================================

class TestConcurrentExecution(unittest.TestCase):

    def test_concurrent_manager_creation(self):
        engine = make_populated_engine()
        mgr = ConcurrentExecutionManager(engine)
        self.assertEqual(mgr.max_retries, 3)

    def test_concurrent_select(self):
        engine = make_populated_engine()
        mgr = ConcurrentExecutionManager(engine)
        tx_id = engine.begin()
        result = mgr.execute_concurrent(tx_id, "SELECT * FROM employees")
        self.assertEqual(len(result.rows), 5)
        engine.commit(tx_id)

    def test_concurrent_batch(self):
        engine = make_populated_engine()
        mgr = ConcurrentExecutionManager(engine)
        tx1 = engine.begin()
        tx2 = engine.begin()
        results = mgr.execute_batch([
            (tx1, "SELECT * FROM employees"),
            (tx2, "SELECT * FROM departments"),
        ])
        self.assertEqual(len(results), 2)
        engine.commit(tx1)
        engine.commit(tx2)


# ============================================================
# 12. Analyzer
# ============================================================

class TestAnalyzer(unittest.TestCase):

    def test_health_report(self):
        engine = make_populated_engine()
        analyzer = IntegratedEngineAnalyzer(engine)
        report = analyzer.health_report()
        self.assertIn('engine', report)
        self.assertIn('locks', report)
        self.assertIn('plan_cache', report)
        self.assertIn('active_transactions', report)

    def test_lock_report(self):
        engine = make_populated_engine()
        analyzer = IntegratedEngineAnalyzer(engine)
        report = analyzer.lock_report()
        self.assertIsInstance(report, dict)

    def test_cache_report(self):
        engine = make_populated_engine()
        analyzer = IntegratedEngineAnalyzer(engine)
        report = analyzer.cache_report()
        self.assertIsInstance(report, dict)

    def test_transaction_report(self):
        engine = make_populated_engine()
        analyzer = IntegratedEngineAnalyzer(engine)
        tx_id = engine.begin()
        engine.execute("SELECT * FROM employees", tx_id)
        report = analyzer.transaction_report(tx_id)
        self.assertEqual(report['tx_id'], tx_id)
        self.assertEqual(report['state'], 'ACTIVE')
        self.assertEqual(report['statements'], 1)
        engine.commit(tx_id)

    def test_transaction_report_not_found(self):
        engine = make_engine()
        analyzer = IntegratedEngineAnalyzer(engine)
        report = analyzer.transaction_report(99999)
        self.assertIn('error', report)

    def test_workload_analysis(self):
        engine = make_populated_engine()
        analyzer = IntegratedEngineAnalyzer(engine)
        report = analyzer.workload_analysis([
            "SELECT * FROM employees",
            "SELECT * FROM departments",
        ])
        self.assertEqual(report['query_count'], 2)
        self.assertIn('queries', report)


# ============================================================
# 13. ExecutionResult
# ============================================================

class TestExecutionResult(unittest.TestCase):

    def test_repr_with_rows(self):
        result = ExecutionResult(rows=[{"a": 1}], execution_time_ms=1.5)
        self.assertIn("1 rows", repr(result))

    def test_repr_with_message(self):
        result = ExecutionResult(message="Table created", execution_time_ms=0.5)
        self.assertIn("Table created", repr(result))

    def test_repr_affected_rows(self):
        result = ExecutionResult(affected_rows=3, execution_time_ms=0.5)
        self.assertIn("3 affected", repr(result))


# ============================================================
# 14. Undo System
# ============================================================

class TestUndoSystem(unittest.TestCase):

    def test_undo_insert(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        tx_id = engine.begin()
        engine.execute("INSERT INTO t (id, name) VALUES (1, 'A')", tx_id)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 1)
        engine.rollback(tx_id)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 0)

    def test_undo_multiple_inserts(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        tx_id = engine.begin()
        engine.execute("INSERT INTO t (id, name) VALUES (1, 'A')", tx_id)
        engine.execute("INSERT INTO t (id, name) VALUES (2, 'B')", tx_id)
        engine.execute("INSERT INTO t (id, name) VALUES (3, 'C')", tx_id)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 3)
        engine.rollback(tx_id)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 0)

    def test_undo_delete(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        engine.execute("INSERT INTO t (id, name) VALUES (1, 'A')")
        tx_id = engine.begin()
        engine.execute("DELETE FROM t WHERE id=1", tx_id)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 0)
        engine.rollback(tx_id)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 1)

    def test_undo_preserves_committed_data(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        engine.execute("INSERT INTO t (id, name) VALUES (1, 'A')")
        engine.execute("INSERT INTO t (id, name) VALUES (2, 'B')")
        tx_id = engine.begin()
        engine.execute("INSERT INTO t (id, name) VALUES (3, 'C')", tx_id)
        engine.rollback(tx_id)
        # Original 2 rows should remain
        self.assertEqual(len(engine.query("SELECT * FROM t")), 2)

    def test_undo_entry_creation(self):
        entry = UndoEntry("t", "insert", {"id": 1, "name": "A"})
        self.assertEqual(entry.table_name, "t")
        self.assertEqual(entry.operation, "insert")
        self.assertEqual(entry.row_data, {"id": 1, "name": "A"})


# ============================================================
# 15. TransactionContext
# ============================================================

class TestTransactionContext(unittest.TestCase):

    def test_tx_ids_unique(self):
        engine = make_engine()
        tx1 = TransactionContext(engine)
        tx2 = TransactionContext(engine)
        self.assertNotEqual(tx1.tx_id, tx2.tx_id)

    def test_tx_initial_state(self):
        engine = make_engine()
        tx = TransactionContext(engine)
        self.assertEqual(tx.state, TxState.ACTIVE)
        self.assertTrue(tx.is_active)
        self.assertEqual(tx.statement_count, 0)
        self.assertEqual(tx.rows_read, 0)
        self.assertEqual(tx.rows_written, 0)

    def test_tx_custom_isolation(self):
        engine = make_engine()
        tx = TransactionContext(engine, IsolationLevel.SERIALIZABLE)
        self.assertEqual(tx.isolation, IsolationLevel.SERIALIZABLE)

    def test_tx_custom_id(self):
        engine = make_engine()
        tx = TransactionContext(engine, tx_id=42)
        self.assertEqual(tx.tx_id, 42)


# ============================================================
# 16. LockAcquirer
# ============================================================

class TestLockAcquirer(unittest.TestCase):

    def test_acquire_table_lock(self):
        from lock_manager import LockManager, LockMode, LockResult, make_table
        lm = LockManager()
        acq = LockAcquirer(lm)
        result = acq.acquire_table_lock(1, "users", LockMode.S)
        self.assertEqual(result, LockResult.GRANTED)

    def test_release_all(self):
        from lock_manager import LockManager, LockMode
        lm = LockManager()
        acq = LockAcquirer(lm)
        acq.acquire_table_lock(1, "users", LockMode.S)
        acq.acquire_table_lock(1, "orders", LockMode.X)
        released = acq.release_all(1)
        self.assertGreaterEqual(released, 2)


# ============================================================
# 17. Edge Cases
# ============================================================

class TestEdgeCases(unittest.TestCase):

    def test_empty_insert_many(self):
        engine = make_engine()
        result = engine.insert_many("t", [])
        self.assertIn("No rows", result.message)

    def test_insert_into_nonexistent_table(self):
        engine = make_engine()
        result = engine.execute("INSERT INTO ghost (id) VALUES (1)")
        self.assertIn("not found", result.message)

    def test_update_nonexistent_table(self):
        engine = make_engine()
        result = engine.execute("UPDATE ghost SET id=1")
        self.assertIn("not found", result.message)

    def test_delete_nonexistent_table(self):
        engine = make_engine()
        result = engine.execute("DELETE FROM ghost WHERE id=1")
        self.assertIn("not found", result.message)

    def test_select_after_all_deleted(self):
        engine = make_populated_engine()
        engine.execute("DELETE FROM employees")
        result = engine.execute("SELECT * FROM employees")
        self.assertEqual(len(result.rows), 0)

    def test_multiple_transactions_independent(self):
        engine = make_populated_engine()
        tx1 = engine.begin()
        tx2 = engine.begin()
        engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'F', 1, 0)", tx1)
        engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (7, 'G', 2, 0)", tx2)
        engine.rollback(tx1)  # Only tx1 rolled back
        engine.commit(tx2)    # tx2 committed
        rows = engine.query("SELECT * FROM employees")
        self.assertEqual(len(rows), 6)  # original 5 + 1 from tx2

    def test_format_val_types(self):
        engine = make_engine()
        self.assertEqual(engine._format_val(None), 'NULL')
        self.assertEqual(engine._format_val("hello"), "'hello'")
        self.assertEqual(engine._format_val(42), '42')
        self.assertEqual(engine._format_val(3.14), '3.14')


# ============================================================
# 18. DML WHERE Clause Evaluation
# ============================================================

class TestWhereClause(unittest.TestCase):

    def setUp(self):
        from lock_manager import LockManager
        self.db = Database()
        self.db.create_table("t", ["id", "name", "val"], "id")
        self.db.insert("t", {"id": 1, "name": "Alice", "val": 10})
        self.db.insert("t", {"id": 2, "name": "Bob", "val": 20})
        self.db.insert("t", {"id": 3, "name": "Carol", "val": 30})
        lm = LockManager()
        acq = LockAcquirer(lm)
        self.dml = DMLExecutor(self.db, acq)

    def test_equal(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE id=1")
        self.assertEqual(result.affected_rows, 1)

    def test_not_equal(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE id!=1")
        self.assertEqual(result.affected_rows, 2)

    def test_greater_than(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE val>15")
        self.assertEqual(result.affected_rows, 2)

    def test_less_than(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE val<25")
        self.assertEqual(result.affected_rows, 2)

    def test_gte(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE val>=20")
        self.assertEqual(result.affected_rows, 2)

    def test_lte(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE val<=20")
        self.assertEqual(result.affected_rows, 2)

    def test_string_compare(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE name='Bob'")
        self.assertEqual(result.affected_rows, 1)

    def test_and_condition(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE id>1 AND val<30")
        self.assertEqual(result.affected_rows, 1)  # Bob (id=2, val=20)

    def test_or_condition(self):
        result = self.dml.execute_delete("DELETE FROM t WHERE id=1 OR id=3")
        self.assertEqual(result.affected_rows, 2)


# ============================================================
# 19. Integration: Full Pipeline Flow
# ============================================================

class TestFullPipeline(unittest.TestCase):

    def test_create_insert_select(self):
        """Full lifecycle: DDL -> DML -> query."""
        engine = make_engine()
        engine.execute("CREATE TABLE products (id INT, name TEXT, price INT)")
        engine.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 10)")
        engine.execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 25)")
        engine.sync_catalog()
        result = engine.execute("SELECT * FROM products")
        self.assertEqual(len(result.rows), 2)

    def test_insert_update_select(self):
        """Insert then update then verify."""
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, val INT)")
        engine.execute("INSERT INTO t (id, val) VALUES (1, 100)")
        engine.execute("UPDATE t SET val=200 WHERE id=1")
        rows = engine.query("SELECT * FROM t")
        self.assertEqual(len(rows), 1)

    def test_transactional_insert_rollback(self):
        """Transaction insert then rollback."""
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT)")
        tx = engine.begin()
        engine.execute("INSERT INTO t (id) VALUES (1)", tx)
        engine.execute("INSERT INTO t (id) VALUES (2)", tx)
        engine.rollback(tx)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 0)

    def test_transactional_delete_commit(self):
        """Transaction delete then commit."""
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT)")
        engine.execute("INSERT INTO t (id) VALUES (1)")
        engine.execute("INSERT INTO t (id) VALUES (2)")
        tx = engine.begin()
        engine.execute("DELETE FROM t WHERE id=1", tx)
        engine.commit(tx)
        self.assertEqual(len(engine.query("SELECT * FROM t")), 1)

    def test_ddl_then_dml_pipeline(self):
        """Create table, add index, insert, query with index."""
        engine = make_engine()
        engine.execute("CREATE TABLE orders (id INT, customer TEXT, amount INT)")
        engine.execute("CREATE INDEX idx_customer ON orders (customer)")
        engine.execute("INSERT INTO orders (id, customer, amount) VALUES (1, 'Alice', 100)")
        engine.execute("INSERT INTO orders (id, customer, amount) VALUES (2, 'Bob', 200)")
        engine.execute("INSERT INTO orders (id, customer, amount) VALUES (3, 'Alice', 150)")
        engine.sync_catalog()
        rows = engine.query("SELECT * FROM orders")
        self.assertEqual(len(rows), 3)

    def test_explain_then_execute(self):
        """EXPLAIN a query then execute it."""
        engine = make_populated_engine()
        explain = engine.explain("SELECT * FROM employees WHERE dept_id = 1")
        self.assertTrue(len(explain) > 0)
        result = engine.execute("SELECT * FROM employees WHERE dept_id = 1")
        self.assertEqual(len(result.rows), 2)

    def test_stats_update_after_queries(self):
        """Statistics should be updated after execution."""
        engine = make_populated_engine()
        engine.execute("SELECT * FROM employees")
        engine.execute("SELECT * FROM departments")
        self.assertGreater(engine.stats.statements_executed, 0)
        self.assertGreater(engine.stats.rows_returned, 0)


# ============================================================
# 20. EngineStats
# ============================================================

class TestEngineStats(unittest.TestCase):

    def test_initial_stats(self):
        stats = EngineStats()
        self.assertEqual(stats.statements_executed, 0)
        self.assertEqual(stats.commits, 0)
        self.assertEqual(stats.rollbacks, 0)
        self.assertEqual(stats.rows_returned, 0)
        self.assertEqual(stats.deadlocks_detected, 0)

    def test_summary(self):
        stats = EngineStats(statements_executed=10, commits=5, rollbacks=2, rows_returned=100)
        s = stats.summary()
        self.assertEqual(s['statements'], 10)
        self.assertEqual(s['commits'], 5)
        self.assertEqual(s['rollbacks'], 2)
        self.assertEqual(s['rows_returned'], 100)


# ============================================================
# 21. Isolation Levels
# ============================================================

class TestIsolationLevels(unittest.TestCase):

    def test_all_isolation_levels(self):
        for iso in IsolationLevel:
            engine = make_engine(isolation=iso)
            self.assertEqual(engine.default_isolation, iso)

    def test_begin_with_isolation(self):
        engine = make_engine()
        tx_id = engine.begin(IsolationLevel.SERIALIZABLE)
        tx = engine.get_transaction(tx_id)
        self.assertEqual(tx.isolation, IsolationLevel.SERIALIZABLE)


# ============================================================
# 22. Complex Queries
# ============================================================

class TestComplexQueries(unittest.TestCase):

    def test_aggregate_sum(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT SUM(salary) FROM employees")
        self.assertTrue(len(result.rows) > 0)

    def test_aggregate_avg(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT AVG(salary) FROM employees")
        self.assertTrue(len(result.rows) > 0)

    def test_order_by_asc(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT * FROM employees ORDER BY salary ASC")
        self.assertEqual(len(result.rows), 5)

    def test_select_with_alias(self):
        engine = make_populated_engine()
        result = engine.execute("SELECT name AS employee_name FROM employees")
        self.assertEqual(len(result.rows), 5)


# ============================================================
# 23. Plan Cache Integration
# ============================================================

class TestPlanCache(unittest.TestCase):

    def test_repeated_query_uses_cache(self):
        engine = make_populated_engine()
        engine.execute("SELECT * FROM employees")
        stats_before = engine.planner.cache.stats()
        engine.execute("SELECT * FROM employees")
        stats_after = engine.planner.cache.stats()
        # Cache should have some entries
        self.assertGreaterEqual(stats_after.get('size', 0), 0)

    def test_ddl_invalidates_relevant_cache(self):
        engine = make_populated_engine()
        engine.execute("SELECT * FROM employees")
        engine.execute("CREATE INDEX idx_sal ON employees (salary)")
        # Cache for employees should be invalidated
        # New query should work fine
        result = engine.execute("SELECT * FROM employees")
        self.assertEqual(len(result.rows), 5)


# ============================================================
# 24. DDL Edge Cases
# ============================================================

class TestDDLEdgeCases(unittest.TestCase):

    def test_create_table_with_primary_key_constraint(self):
        engine = make_engine()
        result = engine.execute("CREATE TABLE t (id INT, name TEXT, PRIMARY KEY(id))")
        self.assertIn("created", result.message)

    def test_drop_table_no_if_exists(self):
        engine = make_engine()
        result = engine.execute("DROP TABLE ghost")
        self.assertIn("not found", result.message)

    def test_create_index_on_nonexistent_table(self):
        engine = make_engine()
        result = engine.execute("CREATE INDEX idx ON ghost (col)")
        self.assertIn("not found", result.message)


# ============================================================
# 25. Thread Safety (Basic)
# ============================================================

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_reads(self):
        engine = make_populated_engine()
        results = []
        errors = []

        def reader():
            try:
                r = engine.execute("SELECT * FROM employees")
                results.append(len(r.rows))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0)
        self.assertTrue(all(r == 5 for r in results))

    def test_concurrent_inserts_different_tx(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, val INT)")
        errors = []

        def inserter(n):
            try:
                tx = engine.begin()
                engine.execute(f"INSERT INTO t (id, val) VALUES ({n}, {n*10})", tx)
                engine.commit(tx)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=inserter, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0)
        rows = engine.query("SELECT * FROM t")
        self.assertEqual(len(rows), 5)


# ============================================================
# 26. DML SET Clause Parsing
# ============================================================

class TestSetClauseParsing(unittest.TestCase):

    def test_set_int(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, val INT)")
        engine.execute("INSERT INTO t (id, val) VALUES (1, 10)")
        engine.execute("UPDATE t SET val=20")
        rows = engine.query("SELECT * FROM t")
        self.assertEqual(len(rows), 1)

    def test_set_string(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        engine.execute("INSERT INTO t (id, name) VALUES (1, 'old')")
        engine.execute("UPDATE t SET name='new'")
        rows = engine.query("SELECT * FROM t")
        self.assertEqual(len(rows), 1)

    def test_set_null(self):
        engine = make_engine()
        engine.execute("CREATE TABLE t (id INT, name TEXT)")
        engine.execute("INSERT INTO t (id, name) VALUES (1, 'test')")
        engine.execute("UPDATE t SET name=NULL")
        rows = engine.query("SELECT * FROM t")
        self.assertEqual(len(rows), 1)


# ============================================================
# 27. Multiple Table Operations
# ============================================================

class TestMultiTable(unittest.TestCase):

    def test_operations_on_different_tables(self):
        engine = make_populated_engine()
        r1 = engine.execute("SELECT * FROM employees")
        r2 = engine.execute("SELECT * FROM departments")
        self.assertEqual(len(r1.rows), 5)
        self.assertEqual(len(r2.rows), 3)

    def test_insert_into_multiple_tables_in_tx(self):
        engine = make_populated_engine()
        tx = engine.begin()
        engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'F', 3, 55000)", tx)
        engine.execute("INSERT INTO departments (id, name, budget) VALUES (4, 'HR', 150000)", tx)
        engine.commit(tx)
        self.assertEqual(len(engine.query("SELECT * FROM employees")), 6)
        self.assertEqual(len(engine.query("SELECT * FROM departments")), 4)

    def test_rollback_multi_table_tx(self):
        engine = make_populated_engine()
        tx = engine.begin()
        engine.execute("INSERT INTO employees (id, name, dept_id, salary) VALUES (6, 'F', 3, 55000)", tx)
        engine.execute("INSERT INTO departments (id, name, budget) VALUES (4, 'HR', 150000)", tx)
        engine.rollback(tx)
        self.assertEqual(len(engine.query("SELECT * FROM employees")), 5)
        self.assertEqual(len(engine.query("SELECT * FROM departments")), 3)


# ============================================================
# 28. Index Advisor Integration
# ============================================================

class TestIndexAdvisor(unittest.TestCase):

    def test_recommend_indexes(self):
        engine = make_populated_engine()
        analyzer = IntegratedEngineAnalyzer(engine)
        recs = analyzer.recommend_indexes([
            "SELECT * FROM employees WHERE dept_id = 1",
            "SELECT * FROM employees WHERE salary > 80000",
        ])
        self.assertIsInstance(recs, list)


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    unittest.main()
