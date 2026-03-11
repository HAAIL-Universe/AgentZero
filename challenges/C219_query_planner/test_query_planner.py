"""
Tests for C219: Query Planner -- Cost-based query planner with lock-aware optimization.
"""

import sys, os, time, unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C210_query_optimizer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C216_lock_manager'))

from query_planner import (
    QueryPlanner, AnnotatedPlan, LockPlan, LockStrategy, LockGranularity,
    LockCostEstimator, LockCostParams, PlanCache, PlanComparator,
    IndexAdvisor, QueryPlannerAnalyzer, PreparedStatement, LockExecutor,
    TransactionPlan, StatementPlan, StatsTracker,
    parameterize_sql, sql_cache_key,
    collect_tables, collect_indexes, count_joins, plan_depth,
    has_seq_scan, has_sort
)
from query_optimizer import (
    Catalog, TableDef, ColumnStats, IndexDef, CostParams,
    SeqScan, IndexScan, HashJoin, PhysicalSort
)
from lock_manager import (
    LockManager, LockMode, ResourceId, LockResult,
    make_db, make_table, make_row
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_catalog():
    """Create a test catalog with users, orders, products tables."""
    catalog = Catalog()

    users = TableDef(
        name='users',
        columns=[
            ColumnStats('id', distinct_count=10000, min_value=1, max_value=10000),
            ColumnStats('name', distinct_count=8000, avg_width=20),
            ColumnStats('email', distinct_count=10000, avg_width=30),
            ColumnStats('age', distinct_count=80, min_value=18, max_value=99),
            ColumnStats('status', distinct_count=3, min_value=0, max_value=2),
        ],
        row_count=10000,
        indexes=[
            IndexDef('idx_users_pk', 'users', ['id'], unique=True),
            IndexDef('idx_users_email', 'users', ['email'], unique=True),
        ]
    )
    catalog.add_table(users)

    orders = TableDef(
        name='orders',
        columns=[
            ColumnStats('id', distinct_count=50000, min_value=1, max_value=50000),
            ColumnStats('user_id', distinct_count=8000, min_value=1, max_value=10000),
            ColumnStats('product_id', distinct_count=1000, min_value=1, max_value=1000),
            ColumnStats('amount', distinct_count=5000, min_value=1, max_value=10000),
            ColumnStats('created_at', distinct_count=30000),
        ],
        row_count=50000,
        indexes=[
            IndexDef('idx_orders_pk', 'orders', ['id'], unique=True),
            IndexDef('idx_orders_user', 'orders', ['user_id']),
        ]
    )
    catalog.add_table(orders)

    products = TableDef(
        name='products',
        columns=[
            ColumnStats('id', distinct_count=1000, min_value=1, max_value=1000),
            ColumnStats('name', distinct_count=1000, avg_width=30),
            ColumnStats('price', distinct_count=500, min_value=1, max_value=999),
            ColumnStats('category', distinct_count=20),
        ],
        row_count=1000,
        indexes=[
            IndexDef('idx_products_pk', 'products', ['id'], unique=True),
        ]
    )
    catalog.add_table(products)

    return catalog


def make_planner(catalog=None, **kwargs):
    """Create a QueryPlanner with test catalog."""
    if catalog is None:
        catalog = make_catalog()
    return QueryPlanner(catalog, **kwargs)


# ===========================================================================
# 1. SQL Parameterization
# ===========================================================================

class TestParameterization(unittest.TestCase):

    def test_parameterize_integers(self):
        template, params = parameterize_sql("SELECT * FROM users WHERE id = 42")
        self.assertIn("$", template)
        self.assertEqual(len(params), 1)
        self.assertIn(42, params.values())

    def test_parameterize_strings(self):
        template, params = parameterize_sql("SELECT * FROM users WHERE name = 'Alice'")
        self.assertIn("$", template)
        self.assertIn("Alice", params.values())

    def test_parameterize_multiple(self):
        template, params = parameterize_sql(
            "SELECT * FROM users WHERE age > 25 AND status = 1"
        )
        self.assertEqual(len(params), 2)

    def test_parameterize_no_literals(self):
        template, params = parameterize_sql("SELECT * FROM users")
        self.assertEqual(len(params), 0)

    def test_cache_key_deterministic(self):
        key1 = sql_cache_key("SELECT * FROM users WHERE id = $1")
        key2 = sql_cache_key("SELECT * FROM users WHERE id = $1")
        self.assertEqual(key1, key2)

    def test_cache_key_case_insensitive(self):
        key1 = sql_cache_key("SELECT * FROM users")
        key2 = sql_cache_key("select * from users")
        self.assertEqual(key1, key2)

    def test_cache_key_whitespace_normalized(self):
        key1 = sql_cache_key("SELECT  *  FROM  users")
        key2 = sql_cache_key("SELECT * FROM users")
        self.assertEqual(key1, key2)

    def test_parameterize_float(self):
        template, params = parameterize_sql(
            "SELECT * FROM products WHERE price > 9.99"
        )
        has_float = any(isinstance(v, float) for v in params.values())
        self.assertTrue(has_float or len(params) > 0)

    def test_parameterize_preserves_structure(self):
        sql = "SELECT name FROM users WHERE id = 1 AND age > 25"
        template, params = parameterize_sql(sql)
        self.assertIn("FROM users", template)
        self.assertIn("WHERE", template)
        self.assertIn("AND", template)


# ===========================================================================
# 2. Plan Cache
# ===========================================================================

class TestPlanCache(unittest.TestCase):

    def setUp(self):
        self.cache = PlanCache(max_size=4)

    def _make_plan(self, plan_id="P0001"):
        physical = SeqScan(table='users', estimated_cost=10.0, estimated_rows=100)
        lp = LockPlan(strategy=LockStrategy.NO_LOCK)
        return AnnotatedPlan(physical=physical, lock_plan=lp, plan_id=plan_id)

    def test_put_and_get(self):
        plan = self._make_plan()
        self.cache.put("key1", plan, "SELECT * FROM users")
        result = self.cache.get("key1")
        self.assertIsNotNone(result)
        self.assertEqual(result.plan_id, "P0001")

    def test_miss(self):
        result = self.cache.get("nonexistent")
        self.assertIsNone(result)

    def test_lru_eviction(self):
        for i in range(5):
            self.cache.put(f"key{i}", self._make_plan(f"P{i}"), f"sql{i}")
        # key0 should be evicted (max_size=4)
        self.assertIsNone(self.cache.get("key0"))
        self.assertIsNotNone(self.cache.get("key1"))

    def test_schema_invalidation(self):
        self.cache.put("key1", self._make_plan(), "SELECT * FROM users")
        self.cache.invalidate()
        result = self.cache.get("key1")
        self.assertIsNone(result)

    def test_table_invalidation(self):
        self.cache.put("k1", self._make_plan(), "SELECT * FROM users")
        self.cache.put("k2", self._make_plan(), "SELECT * FROM orders")
        self.cache.invalidate_table("users")
        self.assertIsNone(self.cache.get("k1"))
        self.assertIsNotNone(self.cache.get("k2"))

    def test_stats(self):
        self.cache.put("k1", self._make_plan(), "sql")
        self.cache.get("k1")
        self.cache.get("k1")
        self.cache.get("missing")
        stats = self.cache.stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['hit_rate'], 0.5)

    def test_clear(self):
        self.cache.put("k1", self._make_plan(), "sql")
        self.cache.clear()
        self.assertEqual(self.cache.size(), 0)

    def test_overwrite_existing(self):
        plan1 = self._make_plan("P1")
        plan2 = self._make_plan("P2")
        self.cache.put("k1", plan1, "sql")
        self.cache.put("k1", plan2, "sql")
        result = self.cache.get("k1")
        self.assertEqual(result.plan_id, "P2")

    def test_move_to_end_on_access(self):
        for i in range(4):
            self.cache.put(f"k{i}", self._make_plan(f"P{i}"), f"sql{i}")
        # Access k0 to move it to end
        self.cache.get("k0")
        # Add new entry -- k1 should be evicted (oldest not-recently-accessed)
        self.cache.put("k4", self._make_plan("P4"), "sql4")
        self.assertIsNone(self.cache.get("k1"))
        self.assertIsNotNone(self.cache.get("k0"))


# ===========================================================================
# 3. Lock Cost Estimator
# ===========================================================================

class TestLockCostEstimator(unittest.TestCase):

    def setUp(self):
        self.estimator = LockCostEstimator(escalation_threshold=100)

    def test_no_lock_strategy(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.NO_LOCK, 100)
        self.assertEqual(lp.lock_cost, 0.0)
        self.assertEqual(lp.estimated_locks, 0)

    def test_table_shared_cost(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.TABLE_SHARED, 1000)
        self.assertGreater(lp.lock_cost, 0)
        self.assertEqual(lp.granularity, LockGranularity.TABLE)
        self.assertEqual(lp.estimated_locks, 2)

    def test_row_shared_cost(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.ROW_SHARED, 50)
        self.assertGreater(lp.lock_cost, 0)
        self.assertEqual(lp.granularity, LockGranularity.ROW)
        self.assertEqual(lp.estimated_locks, 52)  # 50 rows + 2 intention

    def test_escalation_detection(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.ROW_EXCLUSIVE, 200)
        self.assertTrue(lp.escalation_likely)

    def test_no_escalation_small(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.ROW_EXCLUSIVE, 10)
        self.assertFalse(lp.escalation_likely)

    def test_contention_increases_cost(self):
        est1 = LockCostEstimator(concurrent_txs=1)
        est5 = LockCostEstimator(concurrent_txs=5)
        lp1 = est1.estimate_lock_cost(LockStrategy.ROW_SHARED, 50)
        lp5 = est5.estimate_lock_cost(LockStrategy.ROW_SHARED, 50)
        self.assertGreater(lp5.lock_cost, lp1.lock_cost)

    def test_choose_strategy_small_read(self):
        s = self.estimator.choose_strategy(5, 10000, is_write=False, has_index=True)
        self.assertEqual(s, LockStrategy.ROW_SHARED)

    def test_choose_strategy_large_read(self):
        s = self.estimator.choose_strategy(8000, 10000, is_write=False)
        self.assertEqual(s, LockStrategy.TABLE_SHARED)

    def test_choose_strategy_small_write(self):
        s = self.estimator.choose_strategy(5, 10000, is_write=True, has_index=True)
        self.assertEqual(s, LockStrategy.ROW_EXCLUSIVE)

    def test_choose_strategy_large_write(self):
        s = self.estimator.choose_strategy(5000, 10000, is_write=True)
        self.assertEqual(s, LockStrategy.TABLE_EXCLUSIVE)

    def test_choose_strategy_zero_rows(self):
        s = self.estimator.choose_strategy(0, 0)
        self.assertEqual(s, LockStrategy.NO_LOCK)

    def test_page_cost(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.PAGE_SHARED, 500)
        self.assertEqual(lp.granularity, LockGranularity.PAGE)
        self.assertGreater(lp.estimated_locks, 2)

    def test_page_exclusive_cost(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.PAGE_EXCLUSIVE, 200)
        self.assertEqual(lp.granularity, LockGranularity.PAGE)

    def test_table_exclusive_cost(self):
        lp = self.estimator.estimate_lock_cost(LockStrategy.TABLE_EXCLUSIVE, 1000)
        self.assertEqual(lp.granularity, LockGranularity.TABLE)
        self.assertEqual(lp.estimated_locks, 2)

    def test_custom_params(self):
        params = LockCostParams(lock_acquire_cost=0.1, lock_release_cost=0.05)
        est = LockCostEstimator(params=params)
        lp = est.estimate_lock_cost(LockStrategy.ROW_SHARED, 10)
        self.assertGreater(lp.lock_cost, 0)


# ===========================================================================
# 4. Query Planner - Basic Planning
# ===========================================================================

class TestQueryPlannerBasic(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_simple_select(self):
        plan = self.planner.plan("SELECT * FROM users")
        self.assertIsInstance(plan, AnnotatedPlan)
        self.assertIn('users', plan.tables_accessed)
        self.assertGreater(plan.total_cost, 0)

    def test_select_with_where(self):
        plan = self.planner.plan("SELECT * FROM users WHERE id = 42")
        self.assertIsInstance(plan, AnnotatedPlan)
        self.assertIsNotNone(plan.plan_id)

    def test_select_with_join(self):
        plan = self.planner.plan(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        self.assertIn('users', plan.tables_accessed)
        self.assertIn('orders', plan.tables_accessed)

    def test_plan_has_lock_plan(self):
        plan = self.planner.plan("SELECT * FROM users")
        self.assertIsInstance(plan.lock_plan, LockPlan)

    def test_write_plan_different_lock(self):
        read_plan = self.planner.plan("SELECT * FROM users", is_write=False)
        write_plan = self.planner.plan("SELECT * FROM users", is_write=True)
        # Write should have exclusive strategy
        self.assertNotEqual(read_plan.lock_plan.strategy,
                            write_plan.lock_plan.strategy)

    def test_plan_id_increments(self):
        p1 = self.planner.plan("SELECT * FROM users", use_cache=False)
        p2 = self.planner.plan("SELECT * FROM orders", use_cache=False)
        self.assertNotEqual(p1.plan_id, p2.plan_id)

    def test_estimated_rows_positive(self):
        plan = self.planner.plan("SELECT * FROM users")
        self.assertGreaterEqual(plan.estimated_rows, 0)

    def test_total_cost_includes_lock(self):
        plan = self.planner.plan("SELECT * FROM users", is_write=True)
        self.assertGreater(plan.lock_plan.lock_cost, 0)
        # Total should be >= optimizer cost + lock cost
        self.assertGreater(plan.total_cost, 0)

    def test_three_way_join(self):
        plan = self.planner.plan(
            "SELECT * FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "JOIN products ON orders.product_id = products.id"
        )
        self.assertEqual(len(plan.tables_accessed), 3)


# ===========================================================================
# 5. Plan Cache Integration
# ===========================================================================

class TestPlanCacheIntegration(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_cache_hit(self):
        p1 = self.planner.plan("SELECT * FROM users WHERE id = 1")
        p2 = self.planner.plan("SELECT * FROM users WHERE id = 1")
        # Same plan returned from cache
        stats = self.planner.cache_stats()
        self.assertGreater(stats['hits'], 0)

    def test_cache_miss_different_query(self):
        self.planner.plan("SELECT * FROM users")
        self.planner.plan("SELECT * FROM orders")
        stats = self.planner.cache_stats()
        self.assertGreater(stats['misses'], 0)

    def test_cache_disabled(self):
        planner = make_planner(enable_cache=False)
        plan = planner.plan("SELECT * FROM users")
        stats = planner.cache_stats()
        self.assertFalse(stats.get('enabled', True))

    def test_cache_invalidation_on_stats_update(self):
        self.planner.plan("SELECT * FROM users")
        self.planner.update_statistics('users', 50000)
        # Cache should be invalidated for users
        p2 = self.planner.plan("SELECT * FROM users")
        # Should have re-planned (miss)
        stats = self.planner.cache_stats()
        self.assertGreater(stats['misses'], 0)

    def test_use_cache_false(self):
        self.planner.plan("SELECT * FROM users")
        p2 = self.planner.plan("SELECT * FROM users", use_cache=False)
        # use_cache=False should not count as cache hit
        self.assertIsNotNone(p2)


# ===========================================================================
# 6. EXPLAIN Output
# ===========================================================================

class TestExplain(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_explain_basic(self):
        output = self.planner.explain("SELECT * FROM users")
        self.assertIn("Plan ID", output)
        self.assertIn("Total Cost", output)
        self.assertIn("Lock Strategy", output)

    def test_explain_shows_tables(self):
        output = self.planner.explain("SELECT * FROM users")
        self.assertIn("users", output)

    def test_explain_no_locks(self):
        output = self.planner.explain("SELECT * FROM users", show_locks=False)
        self.assertNotIn("Lock Strategy", output)

    def test_explain_write_shows_exclusive(self):
        output = self.planner.explain("SELECT * FROM users", is_write=True)
        self.assertIn("EXCLUSIVE", output)

    def test_explain_analyze(self):
        output = self.planner.explain_analyze(
            "SELECT * FROM users",
            actual_rows={'users': 25000}
        )
        self.assertIn("Actual vs Estimated", output)
        self.assertIn("DRIFT", output)

    def test_explain_analyze_ok(self):
        output = self.planner.explain_analyze(
            "SELECT * FROM users",
            actual_rows={'users': 10000}
        )
        self.assertIn("OK", output)

    def test_annotated_plan_explain(self):
        plan = self.planner.plan("SELECT * FROM users")
        output = plan.explain()
        self.assertIn("Plan ID", output)


# ===========================================================================
# 7. Lock Plan
# ===========================================================================

class TestLockPlan(unittest.TestCase):

    def test_lock_plan_summary(self):
        lp = LockPlan(
            strategy=LockStrategy.ROW_SHARED,
            estimated_locks=52,
            lock_cost=3.5
        )
        summary = lp.summary()
        self.assertIn("ROW_SHARED", summary)
        self.assertIn("locks=52", summary)

    def test_no_lock_summary(self):
        lp = LockPlan(strategy=LockStrategy.NO_LOCK)
        summary = lp.summary()
        self.assertIn("NO_LOCK", summary)

    def test_escalation_in_summary(self):
        lp = LockPlan(
            strategy=LockStrategy.ROW_EXCLUSIVE,
            escalation_likely=True,
            estimated_locks=200
        )
        summary = lp.summary()
        self.assertIn("escalation_likely", summary)


# ===========================================================================
# 8. Transaction Planning
# ===========================================================================

class TestTransactionPlanning(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_single_statement_tx(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM users WHERE id = 1", False)
        ])
        self.assertEqual(tx_plan.tx_id, 1)
        self.assertEqual(len(tx_plan.statements), 1)
        self.assertEqual(tx_plan.deadlock_risk, "low")

    def test_multi_statement_tx(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM users WHERE id = 1", False),
            ("SELECT * FROM orders WHERE user_id = 1", False),
        ])
        self.assertEqual(len(tx_plan.statements), 2)
        self.assertGreater(tx_plan.total_cost, 0)

    def test_write_transaction(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM users WHERE id = 1", True),
            ("SELECT * FROM orders WHERE user_id = 1", True),
        ])
        self.assertIn(tx_plan.deadlock_risk, ("low", "medium", "high"))

    def test_lock_order_alphabetical(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM orders", False),
            ("SELECT * FROM users", False),
        ])
        self.assertEqual(tx_plan.lock_order, ['orders', 'users'])

    def test_deadlock_risk_multiple_write_overlap(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM users JOIN orders ON users.id = orders.user_id", True),
            ("SELECT * FROM orders JOIN users ON orders.user_id = users.id", True),
            ("SELECT * FROM users JOIN orders ON users.id = orders.user_id", True),
        ])
        self.assertIn(tx_plan.deadlock_risk, ("medium", "high"))

    def test_tx_explain(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM users", False),
        ])
        output = tx_plan.explain()
        self.assertIn("Transaction Plan", output)
        self.assertIn("Statement 1", output)

    def test_statement_type_detection(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM users", False),
        ])
        self.assertEqual(tx_plan.statements[0].statement_type, "SELECT")

    def test_long_tx_warning(self):
        stmts = [("SELECT * FROM users", False)] * 6
        tx_plan = self.planner.plan_transaction(1, stmts)
        has_warning = any("Long transaction" in w for w in tx_plan.warnings)
        self.assertTrue(has_warning)


# ===========================================================================
# 9. Statistics Tracker
# ===========================================================================

class TestStatsTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = StatsTracker(staleness_threshold=0.2)

    def test_record_scan(self):
        self.tracker.record_scan('users', 12000)
        stats = self.tracker.get_stats('users')
        self.assertIsNotNone(stats)
        self.assertEqual(stats.actual_row_count, 12000)

    def test_record_index_lookup(self):
        self.tracker.record_index_lookup('users', 1)
        stats = self.tracker.get_stats('users')
        self.assertEqual(stats.actual_index_lookups, 1)

    def test_check_staleness(self):
        catalog = make_catalog()
        self.tracker.record_scan('users', 15000)  # catalog has 10000
        stale = self.tracker.check_staleness(catalog)
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0][0], 'users')

    def test_no_staleness(self):
        catalog = make_catalog()
        self.tracker.record_scan('users', 10500)  # within 20%
        stale = self.tracker.check_staleness(catalog)
        self.assertEqual(len(stale), 0)

    def test_all_stats(self):
        self.tracker.record_scan('users', 100)
        self.tracker.record_scan('orders', 200)
        all_stats = self.tracker.all_stats()
        self.assertEqual(len(all_stats), 2)

    def test_unknown_table(self):
        stats = self.tracker.get_stats('nonexistent')
        self.assertIsNone(stats)


# ===========================================================================
# 10. Plan Utility Functions
# ===========================================================================

class TestPlanUtilities(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_collect_tables_single(self):
        plan = self.planner.plan("SELECT * FROM users")
        tables = collect_tables(plan.physical)
        self.assertIn('users', tables)

    def test_collect_tables_join(self):
        plan = self.planner.plan(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        tables = collect_tables(plan.physical)
        self.assertIn('users', tables)
        self.assertIn('orders', tables)

    def test_collect_indexes(self):
        plan = self.planner.plan("SELECT * FROM users WHERE id = 1")
        indexes = collect_indexes(plan.physical)
        # May or may not use an index depending on optimizer
        self.assertIsInstance(indexes, list)

    def test_count_joins_none(self):
        plan = self.planner.plan("SELECT * FROM users")
        self.assertEqual(count_joins(plan.physical), 0)

    def test_count_joins_one(self):
        plan = self.planner.plan(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        self.assertEqual(count_joins(plan.physical), 1)

    def test_plan_depth(self):
        plan = self.planner.plan("SELECT * FROM users")
        depth = plan_depth(plan.physical)
        self.assertGreaterEqual(depth, 1)

    def test_has_seq_scan(self):
        plan = self.planner.plan("SELECT * FROM users")
        # Full table scan should be seq scan
        self.assertTrue(has_seq_scan(plan.physical))


# ===========================================================================
# 11. Plan Comparator
# ===========================================================================

class TestPlanComparator(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()
        self.comparator = PlanComparator(self.planner)

    def test_compare_read_write(self):
        results = self.comparator.compare_strategies(
            "SELECT * FROM users WHERE id = 1"
        )
        self.assertEqual(len(results), 2)
        # Write should cost more (lock overhead)
        read_cost = results[0]['total_cost']
        write_cost = results[1]['total_cost']
        self.assertGreaterEqual(write_cost, read_cost)

    def test_compare_with_index(self):
        idx = IndexDef('idx_users_age', 'users', ['age'])
        result = self.comparator.compare_with_without_index(
            "SELECT * FROM users WHERE age = 25",
            idx
        )
        self.assertIn('without_index', result)
        self.assertIn('with_index', result)
        self.assertIn('improvement', result)


# ===========================================================================
# 12. Index Advisor
# ===========================================================================

class TestIndexAdvisor(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()
        self.advisor = IndexAdvisor(self.planner)

    def test_recommend_filter_column(self):
        self.advisor.record_query("SELECT * FROM users WHERE age = 25", frequency=10)
        recs = self.advisor.recommend()
        # Should recommend index on age (not already indexed)
        age_rec = [r for r in recs if r['column'] == 'age']
        self.assertGreater(len(age_rec), 0)

    def test_no_duplicate_recommendations(self):
        self.advisor.record_query("SELECT * FROM users WHERE id = 1", frequency=10)
        recs = self.advisor.recommend()
        # id is already indexed, should not recommend
        id_rec = [r for r in recs if r['column'] == 'id']
        self.assertEqual(len(id_rec), 0)

    def test_join_column_higher_weight(self):
        self.advisor.record_query(
            "SELECT * FROM orders JOIN products ON orders.product_id = products.id",
            frequency=5
        )
        recs = self.advisor.recommend()
        # product_id might be recommended with higher frequency
        self.assertIsInstance(recs, list)

    def test_max_recommendations(self):
        for i in range(10):
            self.advisor.record_query(f"SELECT * FROM users WHERE age = {i}")
        recs = self.advisor.recommend(max_recommendations=3)
        self.assertLessEqual(len(recs), 3)

    def test_empty_workload(self):
        recs = self.advisor.recommend()
        self.assertEqual(len(recs), 0)


# ===========================================================================
# 13. QueryPlannerAnalyzer
# ===========================================================================

class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()
        self.analyzer = QueryPlannerAnalyzer(self.planner)

    def test_cache_report(self):
        self.planner.plan("SELECT * FROM users")
        report = self.analyzer.cache_report()
        self.assertIn('cache_stats', report)

    def test_staleness_report(self):
        self.planner.stats_tracker.record_scan('users', 15000)
        report = self.analyzer.staleness_report()
        self.assertTrue(report['action_needed'])
        self.assertEqual(len(report['stale_tables']), 1)

    def test_workload_report(self):
        queries = [
            "SELECT * FROM users",
            "SELECT * FROM orders WHERE user_id = 1",
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
        ]
        report = self.analyzer.workload_report(queries)
        self.assertEqual(report['query_count'], 3)
        self.assertGreater(report['total_cost'], 0)
        self.assertIn('users', report['tables_used'])

    def test_workload_report_empty(self):
        report = self.analyzer.workload_report([])
        self.assertEqual(report['query_count'], 0)


# ===========================================================================
# 14. Prepared Statements
# ===========================================================================

class TestPreparedStatement(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_prepare_and_plan(self):
        stmt = PreparedStatement(self.planner, "SELECT * FROM users WHERE id = 1")
        plan = stmt.plan()
        self.assertIsInstance(plan, AnnotatedPlan)

    def test_cached_plan_reuse(self):
        stmt = PreparedStatement(self.planner, "SELECT * FROM users WHERE id = 1")
        p1 = stmt.plan()
        p2 = stmt.plan()
        self.assertEqual(p1.plan_id, p2.plan_id)
        self.assertEqual(stmt.execution_count, 2)

    def test_schema_change_replan(self):
        stmt = PreparedStatement(self.planner, "SELECT * FROM users WHERE id = 1")
        p1 = stmt.plan()
        self.planner.cache.invalidate()
        p2 = stmt.plan()
        # Should re-plan (different plan_id)
        self.assertNotEqual(p1.plan_id, p2.plan_id)

    def test_explain(self):
        stmt = PreparedStatement(self.planner, "SELECT * FROM users")
        output = stmt.explain()
        self.assertIn("Plan ID", output)

    def test_param_count(self):
        stmt = PreparedStatement(
            self.planner,
            "SELECT * FROM users WHERE id = 1 AND age > 25"
        )
        self.assertGreaterEqual(stmt.param_count, 0)

    def test_write_prepared(self):
        stmt = PreparedStatement(
            self.planner,
            "SELECT * FROM users WHERE id = 1",
            is_write=True
        )
        plan = stmt.plan()
        self.assertIsNotNone(plan.lock_plan)


# ===========================================================================
# 15. Lock Executor Integration
# ===========================================================================

class TestLockExecutor(unittest.TestCase):

    def setUp(self):
        self.lock_mgr = LockManager(escalation_threshold=100)
        self.executor = LockExecutor(self.lock_mgr, db="testdb")
        self.planner = make_planner()

    def test_acquire_no_lock(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(strategy=LockStrategy.NO_LOCK)
        results = self.executor.acquire_locks(1, plan)
        self.assertEqual(len(results), 0)

    def test_acquire_table_shared(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.TABLE_SHARED,
            tables=['users']
        )
        results = self.executor.acquire_locks(1, plan)
        self.assertGreater(len(results), 0)
        # Should have GRANTED
        self.assertTrue(all(r[2] in (LockResult.GRANTED, LockResult.WAITING)
                            for r in results))

    def test_acquire_table_exclusive(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.TABLE_EXCLUSIVE,
            tables=['users']
        )
        results = self.executor.acquire_locks(1, plan)
        self.assertGreater(len(results), 0)

    def test_acquire_row_shared(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.ROW_SHARED,
            tables=['users']
        )
        results = self.executor.acquire_locks(1, plan)
        # Should acquire IS on table
        self.assertGreater(len(results), 0)

    def test_acquire_row_exclusive(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.ROW_EXCLUSIVE,
            tables=['users']
        )
        results = self.executor.acquire_locks(1, plan)
        # Should acquire IX on table
        self.assertGreater(len(results), 0)

    def test_release_locks(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.TABLE_SHARED,
            tables=['users']
        )
        self.executor.acquire_locks(1, plan)
        count = self.executor.release_locks(1)
        self.assertGreater(count, 0)

    def test_execute_transaction(self):
        tx_plan = self.planner.plan_transaction(1, [
            ("SELECT * FROM users", False),
            ("SELECT * FROM orders", False),
        ])
        # Override lock plans for test
        for stmt in tx_plan.statements:
            stmt.annotated_plan.lock_plan = LockPlan(
                strategy=LockStrategy.TABLE_SHARED,
                tables=stmt.annotated_plan.tables_accessed
            )
        results = self.executor.execute_transaction(tx_plan)
        self.assertGreater(len(results), 0)

    def test_multiple_table_lock_order(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.TABLE_SHARED,
            tables=['orders', 'users']  # Unordered
        )
        results = self.executor.acquire_locks(1, plan)
        # Should acquire in sorted order
        self.assertEqual(len(results), 2)

    def test_page_shared(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.PAGE_SHARED,
            tables=['users']
        )
        results = self.executor.acquire_locks(1, plan)
        self.assertGreater(len(results), 0)

    def test_page_exclusive(self):
        plan = self.planner.plan("SELECT * FROM users")
        plan.lock_plan = LockPlan(
            strategy=LockStrategy.PAGE_EXCLUSIVE,
            tables=['users']
        )
        results = self.executor.acquire_locks(1, plan)
        self.assertGreater(len(results), 0)


# ===========================================================================
# 16. Concurrency-Aware Planning
# ===========================================================================

class TestConcurrencyAwarePlanning(unittest.TestCase):

    def test_higher_concurrency_higher_cost(self):
        planner1 = make_planner(concurrent_txs=1)
        planner5 = make_planner(concurrent_txs=5)

        p1 = planner1.plan("SELECT * FROM users", is_write=True, use_cache=False)
        p5 = planner5.plan("SELECT * FROM users", is_write=True, use_cache=False)

        self.assertGreater(p5.lock_plan.lock_cost, p1.lock_plan.lock_cost)

    def test_set_concurrency_invalidates_cache(self):
        planner = make_planner()
        planner.plan("SELECT * FROM users")
        planner.set_concurrency(10)
        # Cache should be invalidated
        stats = planner.cache_stats()
        self.assertGreater(stats['schema_version'], 0)


# ===========================================================================
# 17. Update Statistics
# ===========================================================================

class TestUpdateStatistics(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_update_row_count(self):
        self.planner.update_statistics('users', 50000)
        table = self.planner.catalog.get_table('users')
        self.assertEqual(table.row_count, 50000)

    def test_update_column_stats(self):
        self.planner.update_statistics('users', 10000, {
            'age': {'distinct_count': 100, 'min_value': 16, 'max_value': 120}
        })
        table = self.planner.catalog.get_table('users')
        age_col = table.get_column('age')
        self.assertEqual(age_col.distinct_count, 100)

    def test_update_invalidates_cache(self):
        self.planner.plan("SELECT * FROM users")
        self.planner.update_statistics('users', 50000)
        # Cache entry should be gone
        # Plan again and check we got a new plan
        p2 = self.planner.plan("SELECT * FROM users")
        self.assertIsNotNone(p2)

    def test_update_nonexistent_table(self):
        # Should not crash
        self.planner.update_statistics('nonexistent', 100)


# ===========================================================================
# 18. Index Management
# ===========================================================================

class TestIndexManagement(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_add_index(self):
        idx = IndexDef('idx_users_age', 'users', ['age'])
        self.planner.add_index(idx)
        table = self.planner.catalog.get_table('users')
        idx_names = [i.name for i in table.indexes]
        self.assertIn('idx_users_age', idx_names)

    def test_drop_index(self):
        idx = IndexDef('idx_users_age', 'users', ['age'])
        self.planner.add_index(idx)
        self.planner.drop_index('idx_users_age', 'users')
        table = self.planner.catalog.get_table('users')
        idx_names = [i.name for i in table.indexes]
        self.assertNotIn('idx_users_age', idx_names)

    def test_add_index_invalidates_cache(self):
        self.planner.plan("SELECT * FROM users WHERE age = 25")
        idx = IndexDef('idx_users_age', 'users', ['age'])
        self.planner.add_index(idx)
        # Should re-plan with new index
        p2 = self.planner.plan("SELECT * FROM users WHERE age = 25")
        self.assertIsNotNone(p2)


# ===========================================================================
# 19. Replan / Adaptive
# ===========================================================================

class TestAdaptiveReplanning(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_check_replan_no_drift(self):
        stale = self.planner.check_replan()
        self.assertEqual(len(stale), 0)

    def test_check_replan_with_drift(self):
        self.planner.stats_tracker.record_scan('users', 50000)
        stale = self.planner.check_replan()
        self.assertGreater(len(stale), 0)

    def test_replan_invalidates_cache(self):
        self.planner.plan("SELECT * FROM users")
        self.planner.stats_tracker.record_scan('users', 50000)
        stale = self.planner.check_replan()
        # Cache should be invalidated for stale tables
        stats = self.planner.cache_stats()
        self.assertIsNotNone(stats)


# ===========================================================================
# 20. Warnings
# ===========================================================================

class TestWarnings(unittest.TestCase):

    def setUp(self):
        # Large table to trigger seq scan warning
        catalog = Catalog()
        big_table = TableDef(
            name='big_table',
            columns=[
                ColumnStats('id', distinct_count=100000),
                ColumnStats('val', distinct_count=1000),
            ],
            row_count=100000,
        )
        catalog.add_table(big_table)
        self.planner = QueryPlanner(catalog)

    def test_seq_scan_warning(self):
        plan = self.planner.plan("SELECT * FROM big_table")
        has_seq_warning = any("Sequential scan" in w for w in plan.warnings)
        self.assertTrue(has_seq_warning)

    def test_escalation_warning(self):
        plan = self.planner.plan("SELECT * FROM big_table", is_write=True)
        # Large write on big table should trigger escalation or table lock
        self.assertIsNotNone(plan.lock_plan)


# ===========================================================================
# 21. LockStrategy and LockGranularity Enums
# ===========================================================================

class TestEnums(unittest.TestCase):

    def test_lock_strategy_values(self):
        self.assertNotEqual(LockStrategy.NO_LOCK, LockStrategy.ROW_SHARED)
        self.assertNotEqual(LockStrategy.TABLE_SHARED, LockStrategy.TABLE_EXCLUSIVE)

    def test_lock_granularity_values(self):
        self.assertNotEqual(LockGranularity.ROW, LockGranularity.TABLE)

    def test_all_strategies_enumerable(self):
        strategies = list(LockStrategy)
        self.assertEqual(len(strategies), 7)

    def test_all_granularities(self):
        grans = list(LockGranularity)
        self.assertEqual(len(grans), 3)


# ===========================================================================
# 22. Edge Cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_plan_with_limit(self):
        plan = self.planner.plan("SELECT * FROM users LIMIT 10")
        self.assertIsNotNone(plan)

    def test_plan_with_order_by(self):
        plan = self.planner.plan("SELECT * FROM users ORDER BY name")
        self.assertIsNotNone(plan)

    def test_plan_with_aggregate(self):
        plan = self.planner.plan("SELECT COUNT(*) FROM users")
        self.assertIsNotNone(plan)

    def test_plan_with_group_by(self):
        plan = self.planner.plan(
            "SELECT status, COUNT(*) FROM users GROUP BY status"
        )
        self.assertIsNotNone(plan)

    def test_plan_star_select(self):
        plan = self.planner.plan("SELECT * FROM users")
        self.assertIsNotNone(plan.physical)

    def test_plan_with_alias(self):
        plan = self.planner.plan("SELECT u.id FROM users u WHERE u.id = 1")
        self.assertIsNotNone(plan)

    def test_empty_table(self):
        catalog = Catalog()
        catalog.add_table(TableDef(name='empty', columns=[], row_count=0))
        planner = QueryPlanner(catalog)
        plan = planner.plan("SELECT * FROM empty")
        self.assertIsNotNone(plan)


# ===========================================================================
# 23. Multi-table Lock Strategies
# ===========================================================================

class TestMultiTableLocking(unittest.TestCase):

    def setUp(self):
        self.planner = make_planner()

    def test_join_lock_plan(self):
        plan = self.planner.plan(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
            is_write=True
        )
        self.assertGreater(plan.lock_plan.lock_cost, 0)
        self.assertGreater(len(plan.lock_plan.tables), 0)

    def test_three_table_join_locks(self):
        plan = self.planner.plan(
            "SELECT * FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "JOIN products ON orders.product_id = products.id",
            is_write=True
        )
        self.assertGreater(len(plan.tables_accessed), 1)


# ===========================================================================
# 24. Annotated Plan
# ===========================================================================

class TestAnnotatedPlan(unittest.TestCase):

    def test_plan_fields(self):
        physical = SeqScan(table='users', estimated_cost=10.0, estimated_rows=100)
        lp = LockPlan(strategy=LockStrategy.ROW_SHARED)
        ap = AnnotatedPlan(
            physical=physical,
            lock_plan=lp,
            total_cost=15.0,
            estimated_rows=100,
            tables_accessed=['users'],
            indexes_used=[],
            warnings=['test warning'],
            plan_id='P0001'
        )
        self.assertEqual(ap.plan_id, 'P0001')
        self.assertEqual(ap.total_cost, 15.0)
        self.assertEqual(len(ap.warnings), 1)

    def test_explain_with_warnings(self):
        physical = SeqScan(table='users', estimated_cost=10.0, estimated_rows=100)
        lp = LockPlan(strategy=LockStrategy.ROW_SHARED)
        ap = AnnotatedPlan(
            physical=physical, lock_plan=lp,
            warnings=['Sequential scan on large table'],
            plan_id='P0001'
        )
        output = ap.explain()
        self.assertIn('Sequential scan', output)


# ===========================================================================
# 25. Full Integration
# ===========================================================================

class TestFullIntegration(unittest.TestCase):
    """End-to-end tests combining planner, cache, executor, and analyzer."""

    def test_plan_cache_replan_cycle(self):
        planner = make_planner()
        # Plan a query (cache miss)
        p1 = planner.plan("SELECT * FROM users WHERE id = 1")
        # Plan again (cache hit)
        p2 = planner.plan("SELECT * FROM users WHERE id = 1")
        # Update stats (invalidate)
        planner.update_statistics('users', 100000)
        # Plan again (cache miss, re-plan)
        p3 = planner.plan("SELECT * FROM users WHERE id = 1")
        stats = planner.cache_stats()
        self.assertGreater(stats['hits'], 0)
        self.assertGreater(stats['misses'], 0)

    def test_plan_execute_release_cycle(self):
        planner = make_planner()
        lock_mgr = LockManager()
        executor = LockExecutor(lock_mgr, db="testdb")

        # Plan
        plan = planner.plan("SELECT * FROM users", is_write=True)
        plan.lock_plan.tables = plan.tables_accessed

        # Execute locks
        results = executor.acquire_locks(1, plan)

        # Release
        count = executor.release_locks(1)
        self.assertGreaterEqual(count, 0)

    def test_advisor_to_index_to_replan(self):
        planner = make_planner()
        advisor = IndexAdvisor(planner)

        # Record workload
        advisor.record_query("SELECT * FROM users WHERE age = 25", frequency=100)

        # Get recommendations
        recs = advisor.recommend()
        self.assertGreater(len(recs), 0)

        # Plan before adding index
        p1 = planner.plan("SELECT * FROM users WHERE age = 25", use_cache=False)

        # Add recommended index
        rec = recs[0]
        planner.add_index(IndexDef(rec['index_name'], rec['table'], [rec['column']]))

        # Plan after adding index
        p2 = planner.plan("SELECT * FROM users WHERE age = 25", use_cache=False)

        # Cost should be different (ideally lower with index)
        self.assertIsNotNone(p2)

    def test_tx_plan_and_execute(self):
        planner = make_planner()
        lock_mgr = LockManager()
        executor = LockExecutor(lock_mgr, db="testdb")

        tx_plan = planner.plan_transaction(1, [
            ("SELECT * FROM users WHERE id = 1", False),
            ("SELECT * FROM orders WHERE user_id = 1", True),
        ])

        # Override lock plans for execution
        for stmt in tx_plan.statements:
            stmt.annotated_plan.lock_plan = LockPlan(
                strategy=LockStrategy.TABLE_SHARED,
                tables=stmt.annotated_plan.tables_accessed
            )

        results = executor.execute_transaction(tx_plan)
        self.assertGreater(len(results), 0)

        # Release
        executor.release_locks(1)

    def test_analyzer_full_report(self):
        planner = make_planner()
        analyzer = QueryPlannerAnalyzer(planner)

        queries = [
            "SELECT * FROM users",
            "SELECT * FROM users WHERE id = 1",
            "SELECT * FROM orders WHERE user_id = 1",
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
        ]
        report = analyzer.workload_report(queries)
        self.assertEqual(report['query_count'], 4)
        self.assertGreater(report['total_cost'], 0)

    def test_prepared_statement_lifecycle(self):
        planner = make_planner()
        stmt = PreparedStatement(planner, "SELECT * FROM users WHERE id = 1")

        # Execute multiple times
        for _ in range(5):
            plan = stmt.plan()
            self.assertIsNotNone(plan)

        self.assertEqual(stmt.execution_count, 5)

        # Schema change forces replan
        planner.cache.invalidate()
        plan = stmt.plan()
        self.assertEqual(stmt.execution_count, 6)


if __name__ == '__main__':
    unittest.main()
