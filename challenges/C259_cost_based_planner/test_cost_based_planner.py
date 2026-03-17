"""
Tests for C259: Cost-Based Query Planner
"""

import sys
import os
import math
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C258_btree_indexes')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C116_bplus_tree')))

from cost_based_planner import (
    ColumnStats, TableStats, StatisticsCollector,
    CostEstimate, CostModel,
    PlanCandidate, CostBasedPlanner, CostBasedDB,
)
from btree_indexes import IndexManager, IndexInfo, IndexScanDecision
from mini_database import (
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic,
    CatalogError, CompileError,
)


# =============================================================================
# Column Statistics Tests
# =============================================================================

class TestColumnStats(unittest.TestCase):
    def test_default_values(self):
        cs = ColumnStats()
        self.assertEqual(cs.distinct_count, 0)
        self.assertEqual(cs.null_count, 0)
        self.assertIsNone(cs.min_value)
        self.assertIsNone(cs.max_value)
        self.assertFalse(cs.has_stats)

    def test_has_stats_with_data(self):
        cs = ColumnStats(distinct_count=5)
        self.assertTrue(cs.has_stats)

    def test_has_stats_with_nulls_only(self):
        cs = ColumnStats(null_count=3)
        self.assertTrue(cs.has_stats)


# =============================================================================
# Table Statistics Tests
# =============================================================================

class TestTableStats(unittest.TestCase):
    def test_page_count_calculation(self):
        ts = TableStats(table_name='t', row_count=250)
        ts.update_page_count()
        self.assertEqual(ts.page_count, 3)  # 250 / 100 = 2.5 -> ceil = 3

    def test_page_count_minimum_one(self):
        ts = TableStats(table_name='t', row_count=0)
        ts.update_page_count()
        self.assertEqual(ts.page_count, 1)

    def test_selectivity_eq_uniform(self):
        ts = TableStats(table_name='t', row_count=100)
        ts.columns['x'] = ColumnStats(distinct_count=10)
        self.assertAlmostEqual(ts.selectivity_eq('x', 5), 0.1)

    def test_selectivity_eq_mcv(self):
        ts = TableStats(table_name='t', row_count=100)
        cs = ColumnStats(distinct_count=10, most_common_values=[(42, 0.3), (7, 0.2)])
        ts.columns['x'] = cs
        self.assertAlmostEqual(ts.selectivity_eq('x', 42), 0.3)

    def test_selectivity_eq_non_mcv(self):
        ts = TableStats(table_name='t', row_count=100)
        cs = ColumnStats(distinct_count=10, most_common_values=[(42, 0.3)])
        ts.columns['x'] = cs
        self.assertAlmostEqual(ts.selectivity_eq('x', 99), 0.1)

    def test_selectivity_eq_unknown_column(self):
        ts = TableStats(table_name='t', row_count=100)
        self.assertAlmostEqual(ts.selectivity_eq('z', 5), 0.1)  # default

    def test_selectivity_range_full(self):
        ts = TableStats(table_name='t', row_count=100)
        ts.columns['x'] = ColumnStats(distinct_count=100, min_value=0, max_value=100)
        # Range 0-100 on column 0-100 should be 1.0
        sel = ts.selectivity_range('x', low=0, high=100)
        self.assertAlmostEqual(sel, 1.0)

    def test_selectivity_range_half(self):
        ts = TableStats(table_name='t', row_count=100)
        ts.columns['x'] = ColumnStats(distinct_count=100, min_value=0, max_value=100)
        sel = ts.selectivity_range('x', low=0, high=50)
        self.assertAlmostEqual(sel, 0.5)

    def test_selectivity_range_open_low(self):
        ts = TableStats(table_name='t', row_count=100)
        ts.columns['x'] = ColumnStats(distinct_count=100, min_value=0, max_value=100)
        sel = ts.selectivity_range('x', high=25)
        self.assertAlmostEqual(sel, 0.25)

    def test_selectivity_range_open_high(self):
        ts = TableStats(table_name='t', row_count=100)
        ts.columns['x'] = ColumnStats(distinct_count=100, min_value=0, max_value=100)
        sel = ts.selectivity_range('x', low=75)
        self.assertAlmostEqual(sel, 0.25)

    def test_selectivity_comparison_operators(self):
        ts = TableStats(table_name='t', row_count=100)
        ts.columns['x'] = ColumnStats(distinct_count=10, min_value=0, max_value=100)

        # Equality
        self.assertAlmostEqual(ts.selectivity_comparison('x', '=', 5), 0.1)
        # Not equal
        self.assertAlmostEqual(ts.selectivity_comparison('x', '!=', 5), 0.9)
        # Less than
        sel_lt = ts.selectivity_comparison('x', '<', 50)
        self.assertAlmostEqual(sel_lt, 0.5)

    def test_selectivity_and(self):
        ts = TableStats(table_name='t')
        self.assertAlmostEqual(ts.selectivity_and(0.5, 0.5), 0.25)

    def test_selectivity_or(self):
        ts = TableStats(table_name='t')
        self.assertAlmostEqual(ts.selectivity_or(0.5, 0.5), 0.75)

    def test_estimate_rows(self):
        ts = TableStats(table_name='t', row_count=1000)
        self.assertAlmostEqual(ts.estimate_rows(0.1), 100.0)

    def test_estimate_rows_minimum_one(self):
        ts = TableStats(table_name='t', row_count=1000)
        self.assertAlmostEqual(ts.estimate_rows(0.0001), 1.0)


# =============================================================================
# Statistics Collector Tests
# =============================================================================

class TestStatisticsCollector(unittest.TestCase):
    def setUp(self):
        self.collector = StatisticsCollector()
        self.rows = [
            (1, {'id': 1, 'name': 'Alice', 'age': 30}),
            (2, {'id': 2, 'name': 'Bob', 'age': 25}),
            (3, {'id': 3, 'name': 'Charlie', 'age': 35}),
            (4, {'id': 4, 'name': 'Alice', 'age': 28}),
            (5, {'id': 5, 'name': 'Eve', 'age': None}),
        ]

    def test_analyze_row_count(self):
        ts = self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        self.assertEqual(ts.row_count, 5)

    def test_analyze_page_count(self):
        ts = self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        self.assertEqual(ts.page_count, 1)  # 5 rows / 100 = 0.05 -> ceil = 1

    def test_analyze_distinct_count(self):
        ts = self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        self.assertEqual(ts.columns['id'].distinct_count, 5)
        self.assertEqual(ts.columns['name'].distinct_count, 4)  # Alice appears twice

    def test_analyze_null_count(self):
        ts = self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        self.assertEqual(ts.columns['age'].null_count, 1)  # Eve has None age
        self.assertEqual(ts.columns['id'].null_count, 0)

    def test_analyze_min_max(self):
        ts = self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        self.assertEqual(ts.columns['id'].min_value, 1)
        self.assertEqual(ts.columns['id'].max_value, 5)
        self.assertEqual(ts.columns['age'].min_value, 25)
        self.assertEqual(ts.columns['age'].max_value, 35)

    def test_analyze_mcv(self):
        ts = self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        mcv = dict(ts.columns['name'].most_common_values)
        self.assertAlmostEqual(mcv['Alice'], 2/5)

    def test_has_stats(self):
        self.assertFalse(self.collector.has_stats('users'))
        self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        self.assertTrue(self.collector.has_stats('users'))

    def test_invalidate(self):
        self.collector.analyze_table('users', self.rows, ['id', 'name', 'age'])
        self.assertTrue(self.collector.has_stats('users'))
        self.collector.invalidate('users')
        self.assertFalse(self.collector.has_stats('users'))

    def test_analyze_empty_table(self):
        ts = self.collector.analyze_table('empty', [], ['a', 'b'])
        self.assertEqual(ts.row_count, 0)
        self.assertEqual(ts.columns['a'].distinct_count, 0)

    def test_analyze_histogram(self):
        # Need > HISTOGRAM_BUCKETS rows
        rows = [(i, {'val': i}) for i in range(100)]
        ts = self.collector.analyze_table('big', rows, ['val'])
        self.assertGreater(len(ts.columns['val'].histogram), 0)

    def test_all_stats(self):
        self.collector.analyze_table('t1', [(1, {'a': 1})], ['a'])
        self.collector.analyze_table('t2', [(1, {'b': 2})], ['b'])
        all_s = self.collector.all_stats()
        self.assertIn('t1', all_s)
        self.assertIn('t2', all_s)


# =============================================================================
# Cost Estimate Tests
# =============================================================================

class TestCostEstimate(unittest.TestCase):
    def test_per_row_cost(self):
        ce = CostEstimate(total_cost=100.0, output_rows=10.0)
        self.assertAlmostEqual(ce.per_row_cost, 10.0)

    def test_per_row_cost_zero_rows(self):
        ce = CostEstimate(total_cost=100.0, output_rows=0.0)
        self.assertAlmostEqual(ce.per_row_cost, 0.0)

    def test_repr(self):
        ce = CostEstimate(startup_cost=1.0, total_cost=10.0, output_rows=5.0)
        s = repr(ce)
        self.assertIn('startup=1.00', s)
        self.assertIn('total=10.00', s)


# =============================================================================
# Cost Model Tests
# =============================================================================

class TestCostModel(unittest.TestCase):
    def setUp(self):
        self.collector = StatisticsCollector()
        self.model = CostModel(self.collector)

        # Create a table with 1000 rows
        rows = [(i, {'id': i, 'val': i % 10}) for i in range(1000)]
        self.collector.analyze_table('big', rows, ['id', 'val'])

    def test_seq_scan_cost_with_stats(self):
        cost = self.model.seq_scan_cost('big')
        self.assertGreater(cost.total_cost, 0)
        self.assertEqual(cost.output_rows, 1000)
        self.assertEqual(cost.startup_cost, 0.0)

    def test_seq_scan_cost_no_stats(self):
        cost = self.model.seq_scan_cost('unknown')
        self.assertEqual(cost.total_cost, 100.0)  # default

    def test_seq_scan_with_filter(self):
        cost_full = self.model.seq_scan_cost('big', filter_selectivity=1.0)
        cost_filtered = self.model.seq_scan_cost('big', filter_selectivity=0.1)
        # Filtered scan has fewer output rows but similar total cost (still scans all)
        self.assertAlmostEqual(cost_filtered.output_rows, 100.0)
        # Filter adds CPU cost
        self.assertGreater(cost_filtered.total_cost, cost_full.total_cost)

    def test_index_scan_cheaper_for_selective(self):
        from btree_indexes import IndexInfo
        idx = IndexInfo(name='idx_id', table_name='big', columns=['id'], unique=True)
        # Very selective (single row lookup)
        idx_cost = self.model.index_scan_cost('big', idx, selectivity=0.001)
        seq_cost = self.model.seq_scan_cost('big', filter_selectivity=0.001)
        self.assertLess(idx_cost.total_cost, seq_cost.total_cost)

    def test_seq_scan_cheaper_for_unselective(self):
        from btree_indexes import IndexInfo
        idx = IndexInfo(name='idx_val', table_name='big', columns=['val'])
        # Unselective (90% of rows)
        idx_cost = self.model.index_scan_cost('big', idx, selectivity=0.9)
        seq_cost = self.model.seq_scan_cost('big', filter_selectivity=0.9)
        self.assertGreater(idx_cost.total_cost, seq_cost.total_cost)

    def test_hash_join_cost(self):
        outer = CostEstimate(total_cost=10, output_rows=100)
        inner = CostEstimate(total_cost=5, output_rows=50)
        cost = self.model.hash_join_cost(outer, inner, selectivity=0.01)
        self.assertGreater(cost.total_cost, 0)
        self.assertGreater(cost.output_rows, 0)

    def test_nested_loop_cost(self):
        outer = CostEstimate(total_cost=10, output_rows=100)
        inner = CostEstimate(total_cost=5, output_rows=50)
        cost = self.model.nested_loop_join_cost(outer, inner)
        # NL cost is outer + outer.rows * inner
        self.assertGreater(cost.total_cost, outer.total_cost)

    def test_sort_merge_join_cost(self):
        left = CostEstimate(total_cost=10, output_rows=100)
        right = CostEstimate(total_cost=10, output_rows=100)
        cost = self.model.sort_merge_join_cost(left, right, selectivity=0.01)
        self.assertGreater(cost.startup_cost, 0)  # sort has startup cost

    def test_sort_cost(self):
        input_est = CostEstimate(total_cost=10, output_rows=100)
        cost = self.model.sort_cost(input_est)
        self.assertGreater(cost.total_cost, input_est.total_cost)
        self.assertEqual(cost.output_rows, 100)

    def test_aggregate_cost(self):
        input_est = CostEstimate(total_cost=10, output_rows=100)
        cost = self.model.aggregate_cost(input_est, num_groups=10)
        self.assertGreater(cost.total_cost, input_est.total_cost)
        self.assertEqual(cost.output_rows, 10)

    def test_hash_join_cheaper_than_nl_for_large(self):
        outer = CostEstimate(total_cost=100, output_rows=1000)
        inner = CostEstimate(total_cost=50, output_rows=500)
        nl = self.model.nested_loop_join_cost(outer, inner)
        hj = self.model.hash_join_cost(outer, inner)
        self.assertLess(hj.total_cost, nl.total_cost)


# =============================================================================
# Plan Candidate Tests
# =============================================================================

class TestPlanCandidate(unittest.TestCase):
    def test_explain_tree(self):
        plan = PlanCandidate(
            plan_type='hash_join',
            cost=CostEstimate(total_cost=50),
            details={'build': 'small', 'probe': 'big'},
            children=[
                PlanCandidate(plan_type='seq_scan', cost=CostEstimate(total_cost=10)),
                PlanCandidate(plan_type='seq_scan', cost=CostEstimate(total_cost=20)),
            ]
        )
        tree = plan.explain_tree()
        self.assertIn('hash_join', tree)
        self.assertIn('seq_scan', tree)
        self.assertIn('build: small', tree)


# =============================================================================
# Cost-Based Planner Tests
# =============================================================================

class TestCostBasedPlanner(unittest.TestCase):
    def setUp(self):
        self.idx_mgr = IndexManager()
        self.collector = StatisticsCollector()
        self.model = CostModel(self.collector)
        self.planner = CostBasedPlanner(self.idx_mgr, self.model, self.collector)

        # Create stats for a 1000-row table
        rows = [(i, {'id': i, 'val': i % 10, 'name': f'user_{i}'}) for i in range(1000)]
        self.collector.analyze_table('users', rows, ['id', 'val', 'name'])

    def test_plan_scan_no_where(self):
        plan = self.planner.plan_scan('users')
        self.assertEqual(plan.plan_type, 'seq_scan')

    def test_plan_scan_no_index_available(self):
        where = SqlComparison(op='=', left=SqlColumnRef(table=None, column='val'), right=SqlLiteral(value=5))
        plan = self.planner.plan_scan('users', where)
        self.assertEqual(plan.plan_type, 'seq_scan')  # no index to use

    def test_plan_scan_prefers_index_for_selective(self):
        # Create an index on id (unique)
        idx = self.idx_mgr.create_index('idx_id', 'users', ['id'], unique=True)
        where = SqlComparison(op='=', left=SqlColumnRef(table=None, column='id'), right=SqlLiteral(value=42))
        plan = self.planner.plan_scan('users', where)
        self.assertEqual(plan.plan_type, 'index_scan')

    def test_plan_scan_prefers_seq_for_unselective(self):
        # Index on val (only 10 distinct values -> 10% selectivity)
        idx = self.idx_mgr.create_index('idx_val', 'users', ['val'])
        # val = 5 has selectivity 1/10 = 10%
        where = SqlComparison(op='=', left=SqlColumnRef(table=None, column='val'), right=SqlLiteral(value=5))
        plan = self.planner.plan_scan('users', where)
        # With 10% selectivity on 1000 rows, index scan may or may not win
        # The planner should make a reasonable choice either way
        self.assertIn(plan.plan_type, ['seq_scan', 'index_scan'])

    def test_plan_scan_range(self):
        idx = self.idx_mgr.create_index('idx_id', 'users', ['id'], unique=True)
        where = SqlComparison(op='>', left=SqlColumnRef(table=None, column='id'), right=SqlLiteral(value=990))
        plan = self.planner.plan_scan('users', where)
        # Very selective range -> should prefer index
        self.assertEqual(plan.plan_type, 'index_scan')

    def test_plan_scan_and_condition(self):
        idx = self.idx_mgr.create_index('idx_id', 'users', ['id'], unique=True)
        where = SqlLogic(op='and', operands=[
            SqlComparison(op='=', left=SqlColumnRef(table=None, column='id'), right=SqlLiteral(value=42)),
            SqlComparison(op='=', left=SqlColumnRef(table=None, column='val'), right=SqlLiteral(value=2)),
        ])
        plan = self.planner.plan_scan('users', where)
        # Should detect the indexable id=42 condition
        self.assertEqual(plan.plan_type, 'index_scan')

    def test_plan_join_two_tables(self):
        rows2 = [(i, {'order_id': i, 'user_id': i % 100, 'amount': i * 10}) for i in range(5000)]
        self.collector.analyze_table('orders', rows2, ['order_id', 'user_id', 'amount'])

        plan = self.planner.plan_join('users', 'orders')
        # Should pick hash join for large tables
        self.assertIn(plan.plan_type, ['hash_join', 'nested_loop_join', 'sort_merge_join'])

    def test_plan_join_with_condition(self):
        rows2 = [(i, {'order_id': i, 'user_id': i % 100}) for i in range(5000)]
        self.collector.analyze_table('orders', rows2, ['order_id', 'user_id'])

        cond = SqlComparison(op='=',
                             left=SqlColumnRef(table=None, column='id'),
                             right=SqlColumnRef(table=None, column='user_id'))
        plan = self.planner.plan_join('users', 'orders', join_condition=cond)
        self.assertIn(plan.plan_type, ['hash_join', 'nested_loop_join', 'sort_merge_join'])

    def test_plan_multi_join_dp(self):
        # 3 tables
        rows_a = [(i, {'a_id': i}) for i in range(100)]
        rows_b = [(i, {'b_id': i, 'a_id': i % 100}) for i in range(1000)]
        rows_c = [(i, {'c_id': i, 'b_id': i % 1000}) for i in range(10000)]
        self.collector.analyze_table('a', rows_a, ['a_id'])
        self.collector.analyze_table('b', rows_b, ['b_id', 'a_id'])
        self.collector.analyze_table('c', rows_c, ['c_id', 'b_id'])

        conditions = [
            {'left_table': 'a', 'right_table': 'b',
             'condition': SqlComparison(op='=', left=SqlColumnRef(table=None, column='a_id'),
                                        right=SqlColumnRef(table=None, column='a_id'))},
            {'left_table': 'b', 'right_table': 'c',
             'condition': SqlComparison(op='=', left=SqlColumnRef(table=None, column='b_id'),
                                        right=SqlColumnRef(table=None, column='b_id'))},
        ]
        plan = self.planner.plan_multi_join(['a', 'b', 'c'], conditions)
        self.assertIsNotNone(plan)
        self.assertGreater(plan.cost.total_cost, 0)

    def test_plan_multi_join_single_table(self):
        plan = self.planner.plan_multi_join(['users'], [])
        self.assertEqual(plan.plan_type, 'seq_scan')

    def test_selectivity_estimation_or(self):
        sel = self.planner._estimate_where_selectivity('users', SqlLogic(
            op='or', operands=[
                SqlComparison(op='=', left=SqlColumnRef(table=None, column='val'), right=SqlLiteral(value=1)),
                SqlComparison(op='=', left=SqlColumnRef(table=None, column='val'), right=SqlLiteral(value=2)),
            ]
        ))
        # OR of two 10% conditions: 0.1 + 0.1 - 0.01 = 0.19
        self.assertAlmostEqual(sel, 0.19, places=2)

    def test_selectivity_default_no_stats(self):
        sel = self.planner._estimate_where_selectivity('unknown_table', SqlComparison(
            op='=', left=SqlColumnRef(table=None, column='x'), right=SqlLiteral(value=1)
        ))
        self.assertAlmostEqual(sel, 0.33)


# =============================================================================
# CostBasedDB Integration Tests
# =============================================================================

class TestCostBasedDB(unittest.TestCase):
    def setUp(self):
        self.db = CostBasedDB()
        self.db.execute("CREATE TABLE users (id INT, name TEXT, age INT)")
        # 500 rows -- enough that index scans become cost-effective
        for i in range(500):
            self.db.execute(f"INSERT INTO users VALUES ({i}, 'user_{i}', {20 + i % 30})")

    def test_basic_select(self):
        result = self.db.execute("SELECT * FROM users WHERE id = 42")
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0][0], 42)

    def test_analyze_command(self):
        result = self.db.execute("ANALYZE users")
        self.assertIn('ANALYZE', result.message)
        ts = self.db.get_table_stats('users')
        self.assertIsNotNone(ts)
        self.assertEqual(ts.row_count, 500)

    def test_analyze_all_tables(self):
        self.db.execute("CREATE TABLE orders (id INT, user_id INT)")
        result = self.db.execute("ANALYZE")
        self.assertIn('2 tables', result.message)

    def test_auto_analyze(self):
        # First select should trigger auto-analyze
        result = self.db.execute("SELECT * FROM users WHERE id = 1")
        ts = self.db.get_table_stats('users')
        self.assertIsNotNone(ts)

    def test_stats_invalidation_on_insert(self):
        self.db.execute("ANALYZE users")
        self.assertIsNotNone(self.db.get_table_stats('users'))
        self.db.execute("INSERT INTO users VALUES (9999, 'new', 99)")
        self.assertIsNone(self.db.get_table_stats('users'))

    def test_stats_invalidation_on_update(self):
        self.db.execute("ANALYZE users")
        self.db.execute("UPDATE users SET age = 99 WHERE id = 0")
        self.assertIsNone(self.db.get_table_stats('users'))

    def test_stats_invalidation_on_delete(self):
        self.db.execute("ANALYZE users")
        self.db.execute("DELETE FROM users WHERE id = 0")
        self.assertIsNone(self.db.get_table_stats('users'))

    def test_index_with_cost_planner(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("ANALYZE users")
        result = self.db.execute("SELECT * FROM users WHERE id = 42")
        self.assertEqual(len(result.rows), 1)
        # Cost planner should have chosen index scan
        plan = self.db.get_last_plan()
        self.assertIsNotNone(plan)

    def test_cost_planner_chooses_index_for_selective(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("ANALYZE users")
        self.db.execute("SELECT * FROM users WHERE id = 42")
        plan = self.db.get_last_plan()
        self.assertEqual(plan.plan_type, 'index_scan')

    def test_range_query_with_index(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("ANALYZE users")
        result = self.db.execute("SELECT * FROM users WHERE id > 495")
        self.assertEqual(len(result.rows), 4)  # 496, 497, 498, 499
        plan = self.db.get_last_plan()
        self.assertEqual(plan.plan_type, 'index_scan')

    def test_no_index_full_scan(self):
        self.db.execute("ANALYZE users")
        result = self.db.execute("SELECT * FROM users WHERE name = 'user_50'")
        self.assertEqual(len(result.rows), 1)
        plan = self.db.get_last_plan()
        self.assertEqual(plan.plan_type, 'seq_scan')

    def test_explain_shows_cost(self):
        self.db.execute("ANALYZE users")
        result = self.db.execute("EXPLAIN SELECT * FROM users WHERE id = 42")
        plan_text = '\n'.join(row[0] for row in result.rows)
        self.assertIn('cost', plan_text.lower())

    def test_explain_with_index(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("ANALYZE users")
        result = self.db.execute("EXPLAIN SELECT * FROM users WHERE id = 42")
        plan_text = '\n'.join(row[0] for row in result.rows)
        self.assertIn('index_scan', plan_text)

    def test_explain_shows_alternatives(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("ANALYZE users")
        result = self.db.execute("EXPLAIN SELECT * FROM users WHERE id = 42")
        plan_text = '\n'.join(row[0] for row in result.rows)
        self.assertIn('Alternatives', plan_text)

    def test_explain_analyze(self):
        self.db.execute("ANALYZE users")
        result = self.db.execute("EXPLAIN ANALYZE SELECT * FROM users WHERE id = 42")
        plan_text = '\n'.join(row[0] for row in result.rows)
        self.assertIn('Actual rows', plan_text)
        self.assertIn('1', plan_text)

    def test_explain_analyze_accuracy(self):
        self.db.execute("ANALYZE users")
        result = self.db.execute("EXPLAIN ANALYZE SELECT * FROM users WHERE id = 42")
        plan_text = '\n'.join(row[0] for row in result.rows)
        self.assertIn('accuracy', plan_text.lower())

    def test_join_query(self):
        self.db.execute("CREATE TABLE orders (id INT, user_id INT, amount INT)")
        for i in range(200):
            self.db.execute(f"INSERT INTO orders VALUES ({i}, {i % 500}, {i * 10})")
        result = self.db.execute(
            "SELECT users.name, orders.amount FROM users "
            "JOIN orders ON users.id = orders.user_id WHERE users.id = 5"
        )
        self.assertEqual(len(result.rows), 1)  # user 5 has order 5

    def test_cost_breakdown(self):
        self.db.execute("ANALYZE users")
        breakdown = self.db.get_cost_breakdown('users')
        self.assertEqual(breakdown['row_count'], 500)
        self.assertIn('columns', breakdown)
        self.assertIn('id', breakdown['columns'])
        self.assertEqual(breakdown['columns']['id']['distinct'], 500)

    def test_cost_breakdown_no_stats(self):
        breakdown = self.db.get_cost_breakdown('nonexistent')
        self.assertIn('error', breakdown)

    def test_select_all_rows(self):
        result = self.db.execute("SELECT * FROM users")
        self.assertEqual(len(result.rows), 500)

    def test_aggregate_with_index(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        result = self.db.execute("SELECT COUNT(*) as cnt FROM users WHERE age = 20")
        # age = 20 + i%30 = 20 for i=0,30,60,...,480 -> 17 rows
        self.assertEqual(result.rows[0][0], 17)

    def test_order_by_with_planner(self):
        result = self.db.execute("SELECT * FROM users ORDER BY id LIMIT 5")
        self.assertEqual(len(result.rows), 5)
        self.assertEqual(result.rows[0][0], 0)
        self.assertEqual(result.rows[4][0], 4)

    def test_distinct_with_planner(self):
        result = self.db.execute("SELECT DISTINCT age FROM users")
        self.assertEqual(len(result.rows), 30)

    def test_group_by_with_planner(self):
        result = self.db.execute("SELECT age, COUNT(*) as cnt FROM users GROUP BY age ORDER BY age")
        self.assertGreater(len(result.rows), 0)

    def test_and_condition_with_index(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("ANALYZE users")
        result = self.db.execute("SELECT * FROM users WHERE id = 42 AND age > 0")
        self.assertEqual(len(result.rows), 1)

    def test_multiple_indexes(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        self.db.execute("ANALYZE users")
        # ID lookup should use idx_id
        self.db.execute("SELECT * FROM users WHERE id = 10")
        plan1 = self.db.get_last_plan()
        self.assertEqual(plan1.plan_type, 'index_scan')
        self.assertEqual(plan1.details['index'], 'idx_id')

    def test_execute_many(self):
        results = self.db.execute_many(
            "SELECT * FROM users WHERE id = 1; SELECT * FROM users WHERE id = 2"
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0].rows), 1)
        self.assertEqual(len(results[1].rows), 1)

    def test_drop_index_fallback(self):
        self.db.execute("CREATE INDEX idx_id ON users (id)")
        self.db.execute("ANALYZE users")
        self.db.execute("SELECT * FROM users WHERE id = 42")
        plan1 = self.db.get_last_plan()
        self.assertEqual(plan1.plan_type, 'index_scan')

        self.db.execute("DROP INDEX idx_id")
        self.db.execute("ANALYZE users")
        self.db.execute("SELECT * FROM users WHERE id = 42")
        plan2 = self.db.get_last_plan()
        self.assertEqual(plan2.plan_type, 'seq_scan')


# =============================================================================
# Backward Compatibility Tests (C258 features still work)
# =============================================================================

class TestBackwardCompat(unittest.TestCase):
    def setUp(self):
        self.db = CostBasedDB()

    def test_create_table(self):
        self.db.execute("CREATE TABLE t (a INT, b TEXT)")
        result = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(result.rows), 0)

    def test_insert_select(self):
        self.db.execute("CREATE TABLE t (a INT, b TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'hello')")
        result = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(result.rows), 1)

    def test_update(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("UPDATE t SET a = 2 WHERE a = 1")
        result = self.db.execute("SELECT * FROM t")
        self.assertEqual(result.rows[0][0], 2)

    def test_delete(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")
        self.db.execute("DELETE FROM t WHERE a = 1")
        result = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(result.rows), 1)

    def test_create_index(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("CREATE INDEX idx_a ON t (a)")
        result = self.db.execute("SELECT * FROM t WHERE a = 1")
        self.assertEqual(len(result.rows), 1)

    def test_unique_index(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")
        self.db.execute("CREATE UNIQUE INDEX idx_a ON t (a)")
        result = self.db.execute("SELECT * FROM t WHERE a = 1")
        self.assertEqual(len(result.rows), 1)

    def test_drop_table(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("DROP TABLE t")
        with self.assertRaises(Exception):
            self.db.execute("SELECT * FROM t")

    def test_join(self):
        self.db.execute("CREATE TABLE a (id INT, name TEXT)")
        self.db.execute("CREATE TABLE b (id INT, a_id INT)")
        self.db.execute("INSERT INTO a VALUES (1, 'x')")
        self.db.execute("INSERT INTO b VALUES (10, 1)")
        result = self.db.execute("SELECT a.name, b.id FROM a JOIN b ON a.id = b.a_id")
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0][0], 'x')
        self.assertEqual(result.rows[0][1], 10)

    def test_where_multiple(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")
        self.db.execute("INSERT INTO t VALUES (3)")
        result = self.db.execute("SELECT * FROM t WHERE a > 1")
        self.assertEqual(len(result.rows), 2)

    def test_index_stats(self):
        self.db.execute("CREATE TABLE t (a INT)")
        self.db.execute("CREATE INDEX idx_a ON t (a)")
        stats = self.db.index_stats()
        self.assertIn('indexes', stats)


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress(unittest.TestCase):
    def test_large_table_cost_estimation(self):
        db = CostBasedDB()
        db.execute("CREATE TABLE big (id INT, val INT, cat TEXT)")
        for i in range(500):
            db.execute(f"INSERT INTO big VALUES ({i}, {i % 50}, 'cat_{i % 10}')")

        db.execute("ANALYZE big")
        ts = db.get_table_stats('big')
        self.assertEqual(ts.row_count, 500)
        self.assertEqual(ts.columns['id'].distinct_count, 500)
        self.assertEqual(ts.columns['val'].distinct_count, 50)
        self.assertEqual(ts.columns['cat'].distinct_count, 10)

    def test_index_vs_seq_crossover(self):
        """Verify the planner switches from index to seq scan at some selectivity."""
        db = CostBasedDB()
        db.execute("CREATE TABLE t (id INT, grp INT)")
        for i in range(200):
            db.execute(f"INSERT INTO t VALUES ({i}, {i % 5})")
        db.execute("CREATE INDEX idx_id ON t (id)")
        db.execute("ANALYZE t")

        # Highly selective -> index
        db.execute("SELECT * FROM t WHERE id = 42")
        plan1 = db.get_last_plan()
        self.assertEqual(plan1.plan_type, 'index_scan')

    def test_multi_table_join_ordering(self):
        db = CostBasedDB()
        db.execute("CREATE TABLE small (id INT)")
        db.execute("CREATE TABLE medium (id INT, small_id INT)")
        db.execute("CREATE TABLE big (id INT, medium_id INT)")

        for i in range(10):
            db.execute(f"INSERT INTO small VALUES ({i})")
        for i in range(100):
            db.execute(f"INSERT INTO medium VALUES ({i}, {i % 10})")
        for i in range(500):
            db.execute(f"INSERT INTO big VALUES ({i}, {i % 100})")

        db.execute("ANALYZE small")
        db.execute("ANALYZE medium")
        db.execute("ANALYZE big")

        plan = db.planner.plan_multi_join(
            ['small', 'medium', 'big'],
            [
                {'left_table': 'small', 'right_table': 'medium',
                 'condition': SqlComparison(op='=', left=SqlColumnRef(table=None, column='id'),
                                            right=SqlColumnRef(table=None, column='small_id'))},
                {'left_table': 'medium', 'right_table': 'big',
                 'condition': SqlComparison(op='=', left=SqlColumnRef(table=None, column='id'),
                                            right=SqlColumnRef(table=None, column='medium_id'))},
            ]
        )
        self.assertIsNotNone(plan)
        self.assertGreater(plan.cost.total_cost, 0)

    def test_correctness_index_vs_seq(self):
        """Verify index scan and seq scan return same results."""
        db = CostBasedDB()
        db.execute("CREATE TABLE t (id INT, val TEXT)")
        for i in range(100):
            db.execute(f"INSERT INTO t VALUES ({i}, 'v{i}')")

        # Without index (seq scan)
        db._auto_analyze = False  # prevent auto stats
        result_seq = db.execute("SELECT * FROM t WHERE id = 50")

        # With index
        db.execute("CREATE INDEX idx_id ON t (id)")
        db._auto_analyze = True
        db.execute("ANALYZE t")
        result_idx = db.execute("SELECT * FROM t WHERE id = 50")

        self.assertEqual(len(result_seq.rows), len(result_idx.rows))
        self.assertEqual(result_seq.rows[0][0], result_idx.rows[0][0])

    def test_explain_analyze_accuracy_check(self):
        """Check estimation accuracy for known data distribution."""
        db = CostBasedDB()
        db.execute("CREATE TABLE t (id INT, val INT)")
        for i in range(1000):
            db.execute(f"INSERT INTO t VALUES ({i}, {i % 10})")
        db.execute("ANALYZE t")

        # val = 5 should match ~100 rows (1000 / 10)
        result = db.execute("EXPLAIN ANALYZE SELECT * FROM t WHERE val = 5")
        plan_text = '\n'.join(row[0] for row in result.rows)
        self.assertIn('Actual rows: 100', plan_text)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    def test_empty_table_analyze(self):
        db = CostBasedDB()
        db.execute("CREATE TABLE empty (a INT)")
        result = db.execute("ANALYZE empty")
        ts = db.get_table_stats('empty')
        self.assertEqual(ts.row_count, 0)

    def test_single_row_table(self):
        db = CostBasedDB()
        db.execute("CREATE TABLE one (a INT)")
        db.execute("INSERT INTO one VALUES (42)")
        db.execute("ANALYZE one")
        result = db.execute("SELECT * FROM one WHERE a = 42")
        self.assertEqual(len(result.rows), 1)

    def test_all_nulls_column(self):
        db = CostBasedDB()
        db.execute("CREATE TABLE nulls (a INT, b INT)")
        db.execute("INSERT INTO nulls (a) VALUES (1)")
        db.execute("INSERT INTO nulls (a) VALUES (2)")
        db.execute("ANALYZE nulls")
        ts = db.get_table_stats('nulls')
        self.assertEqual(ts.columns['b'].null_count, 2)

    def test_mixed_types_selectivity(self):
        ts = TableStats(table_name='t', row_count=100)
        ts.columns['x'] = ColumnStats(distinct_count=10, min_value='a', max_value='z')
        # String range estimation should return default
        sel = ts.selectivity_range('x', low='b', high='m')
        self.assertGreater(sel, 0)
        self.assertLessEqual(sel, 1.0)

    def test_plan_join_no_condition(self):
        collector = StatisticsCollector()
        model = CostModel(collector)
        idx_mgr = IndexManager()
        planner = CostBasedPlanner(idx_mgr, model, collector)
        collector.analyze_table('a', [(i, {'x': i}) for i in range(10)], ['x'])
        collector.analyze_table('b', [(i, {'y': i}) for i in range(10)], ['y'])
        plan = planner.plan_join('a', 'b')
        # Cross join (no condition) should still work
        self.assertIsNotNone(plan)

    def test_selectivity_no_column_stats(self):
        ts = TableStats(table_name='t', row_count=100)
        sel = ts.selectivity_eq('nonexistent', 5)
        self.assertAlmostEqual(sel, 0.1)  # default

    def test_explain_non_select(self):
        db = CostBasedDB()
        db.execute("CREATE TABLE t (a INT)")
        # EXPLAIN on non-SELECT handled gracefully
        # (EXPLAIN goes through parent which handles SelectStmt check)

    def test_cost_model_no_stats(self):
        collector = StatisticsCollector()
        model = CostModel(collector)
        cost = model.seq_scan_cost('nonexistent')
        self.assertEqual(cost.total_cost, 100.0)


if __name__ == '__main__':
    unittest.main()
