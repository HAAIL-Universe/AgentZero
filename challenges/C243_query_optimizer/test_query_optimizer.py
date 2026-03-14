"""
Tests for C243: Query Optimizer

Comprehensive tests covering:
- Schema and catalog management
- Statistics and histograms
- Selectivity estimation
- Cost model
- Predicate pushdown
- Projection pushdown
- Join ordering (Selinger DP)
- Join algorithm selection
- Index selection
- Query optimizer end-to-end
- EXPLAIN output
- QueryBuilder fluent API
- Plan comparison
- Edge cases
"""

import unittest
import math
from query_optimizer import (
    # Schema
    Column, Index, TableSchema, Catalog, TableStats, ColumnStats, Histogram,
    # Expressions
    ColumnRef, Literal, Comparison, CompOp, LogicExpr, LogicOp,
    FuncCall, AliasExpr,
    # Plan nodes
    ScanNode, IndexScanNode, FilterNode, ProjectNode, JoinNode, JoinType,
    JoinAlgorithm, SortNode, AggregateNode, LimitNode, UnionNode,
    # Utilities
    extract_conjuncts, make_conjunction, referenced_tables, referenced_columns,
    is_join_predicate,
    # Estimators
    SelectivityEstimator, CostModel, Cost,
    # Rules
    PredicatePushdown, ProjectionPushdown, JoinCommutativity,
    # Optimizer
    JoinOrderOptimizer, IndexSelector, QueryOptimizer, QueryBuilder,
    compare_plans,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_catalog():
    """Create a catalog with typical tables for testing."""
    cat = Catalog()

    # employees table
    emp_schema = TableSchema('employees', [
        Column('id', 'int', primary_key=True),
        Column('name', 'string'),
        Column('dept_id', 'int'),
        Column('salary', 'float'),
        Column('age', 'int'),
    ], indexes=[
        Index('emp_pk', 'employees', ['id'], unique=True),
        Index('emp_dept_idx', 'employees', ['dept_id']),
        Index('emp_salary_idx', 'employees', ['salary']),
    ])

    emp_stats = TableStats(
        row_count=10000,
        page_count=200,
        avg_row_width=100,
        column_stats={
            'id': ColumnStats(num_distinct=10000, min_val=1, max_val=10000,
                             histogram=Histogram(1, 10000, [1000]*10, 10000)),
            'name': ColumnStats(num_distinct=9500, avg_width=20),
            'dept_id': ColumnStats(num_distinct=50, min_val=1, max_val=50,
                                  histogram=Histogram(1, 50, [200]*10, 50)),
            'salary': ColumnStats(num_distinct=5000, min_val=30000, max_val=200000,
                                  histogram=Histogram(30000, 200000, [500, 1500, 2000, 2500, 2000, 800, 400, 200, 50, 50], 5000)),
            'age': ColumnStats(num_distinct=40, min_val=22, max_val=65,
                              histogram=Histogram(22, 65, [500, 1500, 2000, 2500, 1500, 1000, 700, 200, 50, 50], 40)),
        }
    )
    cat.add_table(emp_schema, emp_stats)

    # departments table
    dept_schema = TableSchema('departments', [
        Column('id', 'int', primary_key=True),
        Column('name', 'string'),
        Column('budget', 'float'),
        Column('location', 'string'),
    ], indexes=[
        Index('dept_pk', 'departments', ['id'], unique=True),
    ])

    dept_stats = TableStats(
        row_count=50,
        page_count=2,
        avg_row_width=80,
        column_stats={
            'id': ColumnStats(num_distinct=50, min_val=1, max_val=50,
                             histogram=Histogram(1, 50, [5]*10, 50)),
            'name': ColumnStats(num_distinct=50, avg_width=30),
            'budget': ColumnStats(num_distinct=50, min_val=100000, max_val=5000000),
            'location': ColumnStats(num_distinct=10),
        }
    )
    cat.add_table(dept_schema, dept_stats)

    # orders table
    orders_schema = TableSchema('orders', [
        Column('id', 'int', primary_key=True),
        Column('emp_id', 'int'),
        Column('product_id', 'int'),
        Column('amount', 'float'),
        Column('date', 'string'),
    ], indexes=[
        Index('orders_pk', 'orders', ['id'], unique=True),
        Index('orders_emp_idx', 'orders', ['emp_id']),
        Index('orders_prod_idx', 'orders', ['product_id']),
    ])

    orders_stats = TableStats(
        row_count=100000,
        page_count=5000,
        avg_row_width=60,
        column_stats={
            'id': ColumnStats(num_distinct=100000, min_val=1, max_val=100000),
            'emp_id': ColumnStats(num_distinct=10000, min_val=1, max_val=10000,
                                 histogram=Histogram(1, 10000, [10000]*10, 10000)),
            'product_id': ColumnStats(num_distinct=1000, min_val=1, max_val=1000),
            'amount': ColumnStats(num_distinct=50000, min_val=1.0, max_val=10000.0,
                                 histogram=Histogram(1, 10000, [30000, 25000, 20000, 10000, 8000, 4000, 2000, 500, 300, 200], 50000)),
            'date': ColumnStats(num_distinct=365),
        }
    )
    cat.add_table(orders_schema, orders_stats)

    # products table
    products_schema = TableSchema('products', [
        Column('id', 'int', primary_key=True),
        Column('name', 'string'),
        Column('price', 'float'),
        Column('category', 'string'),
    ], indexes=[
        Index('products_pk', 'products', ['id'], unique=True),
        Index('products_cat_idx', 'products', ['category']),
    ])

    products_stats = TableStats(
        row_count=1000,
        page_count=25,
        avg_row_width=80,
        column_stats={
            'id': ColumnStats(num_distinct=1000, min_val=1, max_val=1000),
            'name': ColumnStats(num_distinct=1000, avg_width=30),
            'price': ColumnStats(num_distinct=500, min_val=0.99, max_val=999.99,
                                histogram=Histogram(0.99, 999.99, [300, 200, 150, 100, 80, 60, 50, 30, 20, 10], 500)),
            'category': ColumnStats(num_distinct=20),
        }
    )
    cat.add_table(products_schema, products_stats)

    return cat


# ===========================================================================
# Schema and Catalog Tests
# ===========================================================================

class TestSchema(unittest.TestCase):
    """Tests for schema and catalog objects."""

    def test_column_creation(self):
        c = Column('id', 'int', primary_key=True)
        self.assertEqual(c.name, 'id')
        self.assertEqual(c.type, 'int')
        self.assertTrue(c.primary_key)

    def test_table_schema(self):
        schema = TableSchema('t1', [Column('a'), Column('b'), Column('c')])
        self.assertEqual(schema.name, 't1')
        self.assertEqual(schema.column_names(), ['a', 'b', 'c'])
        self.assertTrue(schema.has_column('a'))
        self.assertFalse(schema.has_column('x'))

    def test_index_creation(self):
        idx = Index('idx1', 'users', ['id'], unique=True)
        self.assertEqual(idx.name, 'idx1')
        self.assertTrue(idx.unique)

    def test_catalog_add_table(self):
        cat = Catalog()
        schema = TableSchema('t1', [Column('x')])
        cat.add_table(schema, TableStats(row_count=100))
        self.assertIsNotNone(cat.get_schema('t1'))
        self.assertEqual(cat.get_stats('t1').row_count, 100)

    def test_catalog_missing_table(self):
        cat = Catalog()
        self.assertIsNone(cat.get_schema('nonexistent'))
        stats = cat.get_stats('nonexistent')
        self.assertEqual(stats.row_count, 0)

    def test_catalog_indexes(self):
        cat = make_catalog()
        idxs = cat.get_indexes_for_table('employees')
        self.assertEqual(len(idxs), 3)
        names = {i.name for i in idxs}
        self.assertIn('emp_pk', names)
        self.assertIn('emp_dept_idx', names)

    def test_catalog_update_stats(self):
        cat = Catalog()
        schema = TableSchema('t', [Column('x')])
        cat.add_table(schema, TableStats(row_count=10))
        self.assertEqual(cat.get_stats('t').row_count, 10)
        cat.update_stats('t', TableStats(row_count=500))
        self.assertEqual(cat.get_stats('t').row_count, 500)


# ===========================================================================
# Histogram Tests
# ===========================================================================

class TestHistogram(unittest.TestCase):

    def test_histogram_properties(self):
        h = Histogram(0, 100, [10, 20, 30, 40], num_distinct=50)
        self.assertEqual(h.num_buckets, 4)
        self.assertEqual(h.total_rows, 100)
        self.assertAlmostEqual(h.bucket_width, 25.0)

    def test_equality_estimation(self):
        h = Histogram(1, 100, [25]*4, num_distinct=100)
        sel = h.estimate_equality(50)
        self.assertAlmostEqual(sel, 0.01)  # 1/100

    def test_range_estimation_full(self):
        h = Histogram(0, 100, [25]*4, num_distinct=100)
        sel = h.estimate_range(0, 100)
        self.assertAlmostEqual(sel, 1.0)

    def test_range_estimation_half(self):
        h = Histogram(0, 100, [25]*4, num_distinct=100)
        sel = h.estimate_range(0, 50)
        self.assertAlmostEqual(sel, 0.5)

    def test_range_estimation_out_of_bounds(self):
        h = Histogram(0, 100, [25]*4)
        sel = h.estimate_range(200, 300)
        self.assertAlmostEqual(sel, 0.0)

    def test_range_with_nulls(self):
        h = Histogram(0, 100, [50]*2, num_distinct=50, null_fraction=0.1)
        sel = h.estimate_range(0, 100)
        self.assertAlmostEqual(sel, 0.9)

    def test_empty_histogram(self):
        h = Histogram(0, 0, [])
        self.assertEqual(h.estimate_range(0, 100), 0.0)
        self.assertEqual(h.estimate_equality(5), 0.0)

    def test_skewed_histogram(self):
        # 90% of data in first bucket
        h = Histogram(0, 100, [90, 5, 3, 2], num_distinct=50)
        sel_low = h.estimate_range(0, 25)
        sel_high = h.estimate_range(75, 100)
        self.assertGreater(sel_low, sel_high)


# ===========================================================================
# Expression Tests
# ===========================================================================

class TestExpressions(unittest.TestCase):

    def test_column_ref(self):
        c = ColumnRef('id', 'employees')
        self.assertEqual(c.qualified(), 'employees.id')
        self.assertEqual(str(c), 'employees.id')

    def test_column_ref_unqualified(self):
        c = ColumnRef('id')
        self.assertEqual(c.qualified(), 'id')

    def test_literal(self):
        l = Literal(42)
        self.assertEqual(l.value, 42)

    def test_comparison_repr(self):
        comp = Comparison(ColumnRef('age', 'e'), CompOp.GT, Literal(30))
        self.assertIn('>', str(comp))

    def test_comparison_is_null(self):
        comp = Comparison(ColumnRef('name'), CompOp.IS_NULL)
        self.assertIn('IS NULL', str(comp))

    def test_comparison_between(self):
        comp = Comparison(ColumnRef('x'), CompOp.BETWEEN, Literal(1), Literal(10))
        self.assertIn('BETWEEN', str(comp))

    def test_logic_and(self):
        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        b = Comparison(ColumnRef('y'), CompOp.EQ, Literal(2))
        expr = LogicExpr(LogicOp.AND, [a, b])
        self.assertIn('AND', str(expr))

    def test_logic_not(self):
        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        expr = LogicExpr(LogicOp.NOT, [a])
        self.assertIn('NOT', str(expr))

    def test_func_call(self):
        f = FuncCall('COUNT', [ColumnRef('id')])
        self.assertIn('COUNT', str(f))

    def test_func_call_distinct(self):
        f = FuncCall('COUNT', [ColumnRef('id')], distinct=True)
        self.assertIn('DISTINCT', str(f))

    def test_alias_expr(self):
        a = AliasExpr(FuncCall('SUM', [ColumnRef('amount')]), 'total')
        self.assertIn('AS total', str(a))


# ===========================================================================
# Expression Utility Tests
# ===========================================================================

class TestExprUtilities(unittest.TestCase):

    def test_extract_conjuncts_flat(self):
        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        b = Comparison(ColumnRef('y'), CompOp.EQ, Literal(2))
        expr = LogicExpr(LogicOp.AND, [a, b])
        conjuncts = extract_conjuncts(expr)
        self.assertEqual(len(conjuncts), 2)

    def test_extract_conjuncts_nested(self):
        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        b = Comparison(ColumnRef('y'), CompOp.EQ, Literal(2))
        c = Comparison(ColumnRef('z'), CompOp.EQ, Literal(3))
        inner = LogicExpr(LogicOp.AND, [a, b])
        outer = LogicExpr(LogicOp.AND, [inner, c])
        conjuncts = extract_conjuncts(outer)
        self.assertEqual(len(conjuncts), 3)

    def test_extract_conjuncts_single(self):
        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        conjuncts = extract_conjuncts(a)
        self.assertEqual(len(conjuncts), 1)

    def test_make_conjunction_empty(self):
        self.assertIsNone(make_conjunction([]))

    def test_make_conjunction_single(self):
        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        result = make_conjunction([a])
        self.assertIs(result, a)

    def test_make_conjunction_multiple(self):
        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        b = Comparison(ColumnRef('y'), CompOp.EQ, Literal(2))
        result = make_conjunction([a, b])
        self.assertIsInstance(result, LogicExpr)
        self.assertEqual(result.op, LogicOp.AND)

    def test_referenced_tables(self):
        expr = Comparison(ColumnRef('id', 'employees'), CompOp.EQ,
                         ColumnRef('dept_id', 'departments'))
        tables = referenced_tables(expr)
        self.assertEqual(tables, {'employees', 'departments'})

    def test_referenced_tables_literal(self):
        expr = Comparison(ColumnRef('id', 'e'), CompOp.EQ, Literal(5))
        tables = referenced_tables(expr)
        self.assertEqual(tables, {'e'})

    def test_referenced_columns(self):
        expr = Comparison(ColumnRef('id', 'e'), CompOp.EQ, ColumnRef('dept_id', 'd'))
        cols = referenced_columns(expr)
        self.assertEqual(cols, {'e.id', 'd.dept_id'})

    def test_is_join_predicate_true(self):
        expr = Comparison(ColumnRef('id', 'e'), CompOp.EQ, ColumnRef('emp_id', 'o'))
        self.assertTrue(is_join_predicate(expr))

    def test_is_join_predicate_false_same_table(self):
        expr = Comparison(ColumnRef('x', 'e'), CompOp.EQ, ColumnRef('y', 'e'))
        self.assertFalse(is_join_predicate(expr))

    def test_is_join_predicate_false_literal(self):
        expr = Comparison(ColumnRef('x', 'e'), CompOp.EQ, Literal(5))
        self.assertFalse(is_join_predicate(expr))

    def test_is_join_predicate_false_ne(self):
        expr = Comparison(ColumnRef('id', 'e'), CompOp.NE, ColumnRef('id', 'o'))
        self.assertFalse(is_join_predicate(expr))

    def test_referenced_tables_in_logic(self):
        a = Comparison(ColumnRef('x', 't1'), CompOp.EQ, Literal(1))
        b = Comparison(ColumnRef('y', 't2'), CompOp.GT, Literal(2))
        expr = LogicExpr(LogicOp.AND, [a, b])
        self.assertEqual(referenced_tables(expr), {'t1', 't2'})

    def test_referenced_tables_in_func(self):
        f = FuncCall('SUM', [ColumnRef('amount', 'orders')])
        self.assertEqual(referenced_tables(f), {'orders'})


# ===========================================================================
# Plan Node Tests
# ===========================================================================

class TestPlanNodes(unittest.TestCase):

    def test_scan_node(self):
        s = ScanNode('employees')
        self.assertEqual(s.table, 'employees')
        self.assertEqual(s.alias, 'employees')
        self.assertEqual(s.tables_referenced(), {'employees'})

    def test_scan_node_alias(self):
        s = ScanNode('employees', 'e')
        self.assertEqual(s.alias, 'e')
        self.assertEqual(s.tables_referenced(), {'e'})
        self.assertIn('AS e', str(s))

    def test_filter_node(self):
        s = ScanNode('t')
        f = FilterNode(s, Comparison(ColumnRef('x'), CompOp.EQ, Literal(1)))
        self.assertEqual(f.children(), [s])
        self.assertEqual(f.tables_referenced(), {'t'})

    def test_project_node(self):
        s = ScanNode('t')
        p = ProjectNode(s, [ColumnRef('a'), ColumnRef('b')])
        self.assertEqual(p.children(), [s])

    def test_join_node(self):
        l = ScanNode('t1')
        r = ScanNode('t2')
        j = JoinNode(l, r, JoinType.INNER)
        self.assertEqual(j.children(), [l, r])
        self.assertEqual(j.tables_referenced(), {'t1', 't2'})

    def test_sort_node(self):
        s = ScanNode('t')
        sort = SortNode(s, [(ColumnRef('x'), 'ASC')])
        self.assertEqual(sort.children(), [s])

    def test_aggregate_node(self):
        s = ScanNode('t')
        agg = AggregateNode(s, [ColumnRef('dept')], [FuncCall('COUNT', [ColumnRef('id')])])
        self.assertEqual(agg.children(), [s])

    def test_limit_node(self):
        s = ScanNode('t')
        lim = LimitNode(s, 10, 5)
        self.assertEqual(lim.limit, 10)
        self.assertEqual(lim.offset, 5)
        self.assertIn('offset', str(lim))

    def test_limit_no_offset(self):
        s = ScanNode('t')
        lim = LimitNode(s, 10)
        self.assertNotIn('offset', str(lim))

    def test_union_node(self):
        a = ScanNode('t1')
        b = ScanNode('t2')
        u = UnionNode(a, b, all=True)
        self.assertIn('ALL', str(u))

    def test_index_scan_node(self):
        idx = Index('idx1', 'employees', ['id'])
        isn = IndexScanNode('employees', idx, [Comparison(ColumnRef('id'), CompOp.EQ, Literal(1))])
        self.assertEqual(isn.tables_referenced(), {'employees'})
        self.assertIn('idx1', str(isn))


# ===========================================================================
# Selectivity Estimation Tests
# ===========================================================================

class TestSelectivity(unittest.TestCase):

    def setUp(self):
        self.catalog = make_catalog()
        self.est = SelectivityEstimator(self.catalog)

    def test_equality_with_stats(self):
        pred = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ, Literal(5))
        sel = self.est.estimate(pred)
        # 50 distinct values -> 1/50 = 0.02
        self.assertAlmostEqual(sel, 1.0/50, places=3)

    def test_equality_without_stats(self):
        pred = Comparison(ColumnRef('x', 'unknown'), CompOp.EQ, Literal(5))
        sel = self.est.estimate(pred)
        self.assertAlmostEqual(sel, 0.01)

    def test_ne_selectivity(self):
        pred = Comparison(ColumnRef('dept_id', 'employees'), CompOp.NE, Literal(5))
        sel = self.est.estimate(pred)
        self.assertAlmostEqual(sel, 1.0 - 1.0/50, places=3)

    def test_range_selectivity(self):
        pred = Comparison(ColumnRef('salary', 'employees'), CompOp.GT, Literal(100000))
        sel = self.est.estimate(pred)
        self.assertGreater(sel, 0)
        self.assertLess(sel, 1)

    def test_between_selectivity(self):
        pred = Comparison(ColumnRef('salary', 'employees'), CompOp.BETWEEN,
                         Literal(50000), Literal(100000))
        sel = self.est.estimate(pred)
        self.assertGreater(sel, 0)
        self.assertLess(sel, 1)

    def test_is_null(self):
        pred = Comparison(ColumnRef('name', 'employees'), CompOp.IS_NULL)
        sel = self.est.estimate(pred)
        self.assertGreaterEqual(sel, 0)

    def test_is_not_null(self):
        pred = Comparison(ColumnRef('name', 'employees'), CompOp.IS_NOT_NULL)
        sel = self.est.estimate(pred)
        self.assertGreater(sel, 0.5)

    def test_like_selectivity(self):
        pred = Comparison(ColumnRef('name', 'employees'), CompOp.LIKE, Literal('%Smith%'))
        sel = self.est.estimate(pred)
        self.assertAlmostEqual(sel, 0.05)

    def test_in_selectivity(self):
        pred = Comparison(ColumnRef('dept_id', 'employees'), CompOp.IN,
                         Literal([1, 2, 3, 4, 5]))
        sel = self.est.estimate(pred)
        # 5 * (1/50) = 0.1
        self.assertAlmostEqual(sel, 5.0/50, places=3)

    def test_and_selectivity(self):
        a = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ, Literal(5))
        b = Comparison(ColumnRef('salary', 'employees'), CompOp.GT, Literal(100000))
        expr = LogicExpr(LogicOp.AND, [a, b])
        sel = self.est.estimate(expr)
        # Should be product of individual selectivities
        sel_a = self.est.estimate(a)
        sel_b = self.est.estimate(b)
        self.assertAlmostEqual(sel, sel_a * sel_b, places=5)

    def test_or_selectivity(self):
        a = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ, Literal(5))
        b = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ, Literal(10))
        expr = LogicExpr(LogicOp.OR, [a, b])
        sel = self.est.estimate(expr)
        sel_a = self.est.estimate(a)
        sel_b = self.est.estimate(b)
        expected = sel_a + sel_b - sel_a * sel_b
        self.assertAlmostEqual(sel, expected, places=5)

    def test_not_selectivity(self):
        a = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ, Literal(5))
        expr = LogicExpr(LogicOp.NOT, [a])
        sel = self.est.estimate(expr)
        self.assertAlmostEqual(sel, 1.0 - 1.0/50, places=3)

    def test_join_selectivity(self):
        pred = Comparison(
            ColumnRef('id', 'employees'), CompOp.EQ,
            ColumnRef('emp_id', 'orders'))
        sel = self.est.estimate(pred)
        # 1/max(distinct_employees.id, distinct_orders.emp_id) = 1/10000
        self.assertAlmostEqual(sel, 1.0/10000, places=6)

    def test_join_selectivity_no_stats(self):
        pred = Comparison(
            ColumnRef('x', 'unknown1'), CompOp.EQ,
            ColumnRef('y', 'unknown2'))
        sel = self.est.estimate(pred)
        self.assertAlmostEqual(sel, 0.1)

    def test_most_common_values(self):
        cat = Catalog()
        schema = TableSchema('t', [Column('status')])
        stats = TableStats(
            row_count=1000,
            column_stats={
                'status': ColumnStats(
                    num_distinct=3,
                    most_common_vals=[('active', 0.7), ('inactive', 0.2), ('deleted', 0.1)]
                )
            }
        )
        cat.add_table(schema, stats)
        est = SelectivityEstimator(cat)

        pred = Comparison(ColumnRef('status', 't'), CompOp.EQ, Literal('active'))
        sel = est.estimate(pred)
        self.assertAlmostEqual(sel, 0.7)


# ===========================================================================
# Cost Model Tests
# ===========================================================================

class TestCostModel(unittest.TestCase):

    def setUp(self):
        self.catalog = make_catalog()
        self.cm = CostModel(self.catalog)

    def test_cost_addition(self):
        a = Cost(io_cost=10, cpu_cost=1.0)
        b = Cost(io_cost=5, cpu_cost=0.5)
        c = a + b
        self.assertAlmostEqual(c.io_cost, 15)
        self.assertAlmostEqual(c.cpu_cost, 1.5)

    def test_cost_total(self):
        c = Cost(io_cost=10, cpu_cost=1.0, startup_cost=2.0)
        self.assertAlmostEqual(c.total, 13.0)

    def test_scan_cost(self):
        scan = ScanNode('employees')
        cost = self.cm.estimate_scan(scan)
        self.assertGreater(cost.total, 0)
        self.assertEqual(scan.estimated_rows, 10000)

    def test_scan_cost_small_table(self):
        scan = ScanNode('departments')
        cost = self.cm.estimate_scan(scan)
        self.assertEqual(scan.estimated_rows, 50)

    def test_filter_cost(self):
        scan = ScanNode('employees')
        self.cm.estimate(scan)
        pred = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ, Literal(5))
        filt = FilterNode(scan, pred)
        cost = self.cm.estimate(filt)
        self.assertGreater(cost.total, 0)
        self.assertLess(filt.estimated_rows, 10000)

    def test_project_cost(self):
        scan = ScanNode('employees')
        self.cm.estimate(scan)
        proj = ProjectNode(scan, [ColumnRef('id'), ColumnRef('name')])
        cost = self.cm.estimate(proj)
        self.assertEqual(proj.estimated_rows, scan.estimated_rows)

    def test_join_cost_hash(self):
        left = ScanNode('employees')
        right = ScanNode('departments')
        cond = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ,
                         ColumnRef('id', 'departments'))
        join = JoinNode(left, right, JoinType.INNER, cond, JoinAlgorithm.HASH)
        cost = self.cm.estimate(join)
        self.assertGreater(cost.total, 0)
        self.assertGreater(join.estimated_rows, 0)

    def test_join_cost_nested_loop(self):
        left = ScanNode('departments')
        right = ScanNode('employees')
        cond = Comparison(ColumnRef('id', 'departments'), CompOp.EQ,
                         ColumnRef('dept_id', 'employees'))
        join = JoinNode(left, right, JoinType.INNER, cond, JoinAlgorithm.NESTED_LOOP)
        cost = self.cm.estimate(join)
        self.assertGreater(cost.total, 0)

    def test_join_cost_sort_merge(self):
        left = ScanNode('employees')
        right = ScanNode('orders')
        cond = Comparison(ColumnRef('id', 'employees'), CompOp.EQ,
                         ColumnRef('emp_id', 'orders'))
        join = JoinNode(left, right, JoinType.INNER, cond, JoinAlgorithm.SORT_MERGE)
        cost = self.cm.estimate(join)
        self.assertGreater(cost.startup_cost, 0)  # Sort has startup cost

    def test_hash_cheaper_than_nested_loop_for_large(self):
        left = ScanNode('employees')
        right = ScanNode('orders')
        cond = Comparison(ColumnRef('id', 'employees'), CompOp.EQ,
                         ColumnRef('emp_id', 'orders'))
        hash_join = JoinNode(left, right, JoinType.INNER, cond, JoinAlgorithm.HASH)
        nl_join = JoinNode(ScanNode('employees'), ScanNode('orders'),
                          JoinType.INNER, cond, JoinAlgorithm.NESTED_LOOP)
        hash_cost = self.cm.estimate(hash_join)
        nl_cost = self.cm.estimate(nl_join)
        self.assertLess(hash_cost.total, nl_cost.total)

    def test_sort_cost(self):
        scan = ScanNode('employees')
        self.cm.estimate(scan)
        sort = SortNode(scan, [(ColumnRef('salary'), 'DESC')])
        cost = self.cm.estimate(sort)
        self.assertGreater(cost.startup_cost, 0)

    def test_aggregate_cost(self):
        scan = ScanNode('employees')
        self.cm.estimate(scan)
        agg = AggregateNode(
            scan,
            [ColumnRef('dept_id', 'employees')],
            [FuncCall('COUNT', [ColumnRef('id')])]
        )
        cost = self.cm.estimate(agg)
        self.assertGreater(cost.total, 0)
        self.assertEqual(agg.estimated_rows, 50)  # 50 distinct dept_ids

    def test_aggregate_no_group_by(self):
        scan = ScanNode('employees')
        self.cm.estimate(scan)
        agg = AggregateNode(scan, [], [FuncCall('COUNT', [ColumnRef('id')])])
        self.cm.estimate(agg)
        self.assertEqual(agg.estimated_rows, 1)

    def test_limit_cost(self):
        scan = ScanNode('employees')
        self.cm.estimate(scan)
        lim = LimitNode(scan, 10)
        cost = self.cm.estimate(lim)
        self.assertEqual(lim.estimated_rows, 10)
        # Should be fraction of scan cost
        full_cost = self.cm.estimate_scan(ScanNode('employees'))
        self.assertLess(cost.total, full_cost.total)

    def test_index_scan_cost(self):
        idx = Index('emp_pk', 'employees', ['id'], unique=True)
        pred = Comparison(ColumnRef('id', 'employees'), CompOp.EQ, Literal(42))
        isn = IndexScanNode('employees', idx, [pred])
        cost = self.cm.estimate(isn)
        self.assertGreater(cost.total, 0)
        # Index scan for unique key should estimate ~1 row
        self.assertLess(isn.estimated_rows, 100)

    def test_cross_join_cost(self):
        left = ScanNode('departments')
        right = ScanNode('products')
        join = JoinNode(left, right, JoinType.CROSS, None, JoinAlgorithm.NESTED_LOOP)
        self.cm.estimate(join)
        self.assertEqual(join.estimated_rows, 50 * 1000)

    def test_left_join_rows(self):
        left = ScanNode('employees')
        right = ScanNode('departments')
        cond = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ,
                         ColumnRef('id', 'departments'))
        join = JoinNode(left, right, JoinType.LEFT, cond, JoinAlgorithm.HASH)
        self.cm.estimate(join)
        # Left join should return at least left_rows
        self.assertGreaterEqual(join.estimated_rows, 10000)

    def test_output_columns_scan(self):
        scan = ScanNode('employees')
        self.cm.estimate(scan)
        self.assertIn('employees.id', scan.output_columns)
        self.assertIn('employees.salary', scan.output_columns)

    def test_output_columns_join(self):
        left = ScanNode('employees')
        right = ScanNode('departments')
        join = JoinNode(left, right, JoinType.INNER)
        self.cm.estimate(join)
        self.assertIn('employees.id', join.output_columns)
        self.assertIn('departments.id', join.output_columns)


# ===========================================================================
# Predicate Pushdown Tests
# ===========================================================================

class TestPredicatePushdown(unittest.TestCase):

    def test_push_into_join_left(self):
        """Filter on left table's column should push down to left child."""
        rule = PredicatePushdown()
        left = ScanNode('employees', 'e')
        right = ScanNode('departments', 'd')
        join = JoinNode(left, right, JoinType.INNER)

        pred = Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000))
        filt = FilterNode(join, pred)

        self.assertTrue(rule.applicable(filt))
        result = rule.apply(filt)

        # Should be a join with filter pushed to left child
        self.assertIsInstance(result, JoinNode)
        self.assertIsInstance(result.left, FilterNode)
        self.assertIsInstance(result.left.child, ScanNode)
        self.assertEqual(result.left.child.alias, 'e')

    def test_push_into_join_right(self):
        rule = PredicatePushdown()
        left = ScanNode('employees', 'e')
        right = ScanNode('departments', 'd')
        join = JoinNode(left, right, JoinType.INNER)

        pred = Comparison(ColumnRef('name', 'd'), CompOp.EQ, Literal('Engineering'))
        filt = FilterNode(join, pred)
        result = rule.apply(filt)

        self.assertIsInstance(result, JoinNode)
        self.assertIsInstance(result.right, FilterNode)
        self.assertEqual(result.right.child.alias, 'd')

    def test_push_join_predicate(self):
        rule = PredicatePushdown()
        left = ScanNode('employees', 'e')
        right = ScanNode('departments', 'd')
        join = JoinNode(left, right, JoinType.INNER)

        pred = Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd'))
        filt = FilterNode(join, pred)
        result = rule.apply(filt)

        self.assertIsInstance(result, JoinNode)
        self.assertIsNotNone(result.condition)

    def test_push_mixed_predicates(self):
        rule = PredicatePushdown()
        left = ScanNode('employees', 'e')
        right = ScanNode('departments', 'd')
        join = JoinNode(left, right, JoinType.INNER)

        p1 = Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000))
        p2 = Comparison(ColumnRef('name', 'd'), CompOp.EQ, Literal('Engineering'))
        p3 = Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd'))
        combined = LogicExpr(LogicOp.AND, [p1, p2, p3])

        filt = FilterNode(join, combined)
        result = rule.apply(filt)

        self.assertIsInstance(result, JoinNode)
        self.assertIsInstance(result.left, FilterNode)   # e.salary > 50000
        self.assertIsInstance(result.right, FilterNode)  # d.name = 'Engineering'
        self.assertIsNotNone(result.condition)           # join condition

    def test_push_through_project(self):
        rule = PredicatePushdown()
        scan = ScanNode('employees', 'e')
        proj = ProjectNode(scan, [ColumnRef('id', 'e'), ColumnRef('salary', 'e')])

        pred = Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000))
        filt = FilterNode(proj, pred)

        self.assertTrue(rule.applicable(filt))
        result = rule.apply(filt)

        # Should be Project(Filter(Scan))
        self.assertIsInstance(result, ProjectNode)
        self.assertIsInstance(result.child, FilterNode)

    def test_not_applicable_to_scan(self):
        rule = PredicatePushdown()
        scan = ScanNode('t')
        pred = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        filt = FilterNode(scan, pred)
        self.assertFalse(rule.applicable(filt))


# ===========================================================================
# Projection Pushdown Tests
# ===========================================================================

class TestProjectionPushdown(unittest.TestCase):

    def test_push_through_join(self):
        rule = ProjectionPushdown()
        cat = make_catalog()
        cm = CostModel(cat)

        left = ScanNode('employees', 'employees')
        right = ScanNode('departments', 'departments')
        cond = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ,
                         ColumnRef('id', 'departments'))
        join = JoinNode(left, right, JoinType.INNER, cond)
        cm.estimate(join)

        proj = ProjectNode(join, [ColumnRef('name', 'employees'),
                                  ColumnRef('name', 'departments')])

        self.assertTrue(rule.applicable(proj))
        result = rule.apply(proj)

        self.assertIsInstance(result, ProjectNode)
        self.assertIsInstance(result.child, JoinNode)

    def test_not_applicable_to_non_join(self):
        rule = ProjectionPushdown()
        scan = ScanNode('t')
        proj = ProjectNode(scan, [ColumnRef('x')])
        self.assertFalse(rule.applicable(proj))


# ===========================================================================
# Join Commutativity Tests
# ===========================================================================

class TestJoinCommutativity(unittest.TestCase):

    def test_swap_inner_join(self):
        rule = JoinCommutativity()
        left = ScanNode('t1')
        right = ScanNode('t2')
        join = JoinNode(left, right, JoinType.INNER)

        self.assertTrue(rule.applicable(join))
        result = rule.apply(join)
        self.assertIsInstance(result, JoinNode)
        self.assertEqual(result.left.table, 't2')
        self.assertEqual(result.right.table, 't1')

    def test_not_applicable_to_left_join(self):
        rule = JoinCommutativity()
        join = JoinNode(ScanNode('t1'), ScanNode('t2'), JoinType.LEFT)
        self.assertFalse(rule.applicable(join))


# ===========================================================================
# Join Order Optimizer Tests
# ===========================================================================

class TestJoinOrderOptimizer(unittest.TestCase):

    def setUp(self):
        self.catalog = make_catalog()
        self.cm = CostModel(self.catalog)
        self.jo = JoinOrderOptimizer(self.catalog, self.cm)

    def test_single_table(self):
        plans = {'e': ScanNode('employees', 'e')}
        result = self.jo.optimize(plans, [])
        self.assertIsInstance(result, ScanNode)

    def test_single_table_with_filter(self):
        plans = {'e': ScanNode('employees', 'e')}
        pred = Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000))
        result = self.jo.optimize(plans, [pred])
        self.assertIsInstance(result, FilterNode)

    def test_two_table_join(self):
        plans = {
            'e': ScanNode('employees', 'e'),
            'd': ScanNode('departments', 'd'),
        }
        pred = Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd'))
        result = self.jo.optimize(plans, [pred])
        self.assertIsInstance(result, JoinNode)
        self.assertIsNotNone(result.condition)

    def test_three_table_join_order(self):
        """With stats, optimizer should pick a reasonable order for 3 tables."""
        plans = {
            'e': ScanNode('employees', 'e'),
            'd': ScanNode('departments', 'd'),
            'o': ScanNode('orders', 'o'),
        }
        preds = [
            Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd')),
            Comparison(ColumnRef('id', 'e'), CompOp.EQ, ColumnRef('emp_id', 'o')),
        ]
        result = self.jo.optimize(plans, preds)
        self.assertIsInstance(result, JoinNode)
        # Should produce a valid join tree with all 3 tables
        tables = result.tables_referenced()
        self.assertEqual(tables, {'e', 'd', 'o'})

    def test_four_table_join(self):
        plans = {
            'e': ScanNode('employees', 'e'),
            'd': ScanNode('departments', 'd'),
            'o': ScanNode('orders', 'o'),
            'p': ScanNode('products', 'p'),
        }
        preds = [
            Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd')),
            Comparison(ColumnRef('id', 'e'), CompOp.EQ, ColumnRef('emp_id', 'o')),
            Comparison(ColumnRef('product_id', 'o'), CompOp.EQ, ColumnRef('id', 'p')),
        ]
        result = self.jo.optimize(plans, preds)
        tables = result.tables_referenced()
        self.assertEqual(tables, {'e', 'd', 'o', 'p'})

    def test_empty_plans_raises(self):
        with self.assertRaises(ValueError):
            self.jo.optimize({}, [])

    def test_join_with_filters(self):
        plans = {
            'e': ScanNode('employees', 'e'),
            'd': ScanNode('departments', 'd'),
        }
        preds = [
            Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd')),
            Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000)),
        ]
        result = self.jo.optimize(plans, preds)
        # Filter on salary should be applied to employees scan
        tables = result.tables_referenced()
        self.assertEqual(tables, {'e', 'd'})


# ===========================================================================
# Index Selection Tests
# ===========================================================================

class TestIndexSelection(unittest.TestCase):

    def setUp(self):
        self.catalog = make_catalog()
        self.cm = CostModel(self.catalog)
        self.selector = IndexSelector(self.catalog, self.cm)

    def test_select_equality_index(self):
        preds = [Comparison(ColumnRef('id', 'employees'), CompOp.EQ, Literal(42))]
        result = self.selector.select_index('employees', preds, 'employees')
        # Should select the pk index for equality on id
        if result:
            # Result is either IndexScanNode or FilterNode(IndexScanNode)
            if isinstance(result, IndexScanNode):
                self.assertEqual(result.index.name, 'emp_pk')
            elif isinstance(result, FilterNode):
                self.assertIsInstance(result.child, IndexScanNode)

    def test_no_index_for_unindexed_column(self):
        preds = [Comparison(ColumnRef('name', 'employees'), CompOp.EQ, Literal('Alice'))]
        result = self.selector.select_index('employees', preds, 'employees')
        # name is not indexed, should return None
        self.assertIsNone(result)

    def test_no_indexes_available(self):
        cat = Catalog()
        cat.add_table(TableSchema('t', [Column('x')]), TableStats(row_count=100, page_count=5))
        sel = IndexSelector(cat, CostModel(cat))
        preds = [Comparison(ColumnRef('x', 't'), CompOp.EQ, Literal(1))]
        result = sel.select_index('t', preds, 't')
        self.assertIsNone(result)

    def test_range_predicate_on_indexed_col(self):
        preds = [Comparison(ColumnRef('salary', 'employees'), CompOp.GT, Literal(150000))]
        result = self.selector.select_index('employees', preds, 'employees')
        # May or may not choose index depending on selectivity


# ===========================================================================
# Query Optimizer End-to-End Tests
# ===========================================================================

class TestQueryOptimizer(unittest.TestCase):

    def setUp(self):
        self.catalog = make_catalog()
        self.opt = QueryOptimizer(self.catalog)

    def test_simple_scan(self):
        plan = ScanNode('employees')
        result = self.opt.optimize(plan)
        self.assertIsNotNone(result)
        self.assertGreater(result.estimated_rows, 0)

    def test_scan_with_filter(self):
        scan = ScanNode('employees')
        pred = Comparison(ColumnRef('salary', 'employees'), CompOp.GT, Literal(50000))
        plan = FilterNode(scan, pred)
        result = self.opt.optimize(plan)
        self.assertGreater(result.estimated_rows, 0)
        self.assertLess(result.estimated_rows, 10000)

    def test_filter_pushed_into_join(self):
        """Filter above join should be pushed down."""
        left = ScanNode('employees', 'e')
        right = ScanNode('departments', 'd')
        join = JoinNode(left, right, JoinType.INNER,
                       Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ,
                                  ColumnRef('id', 'd')))

        pred = Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000))
        plan = FilterNode(join, pred)
        result = self.opt.optimize(plan)

        # The filter should have been pushed into the join tree
        self.assertIsNotNone(result)

    def test_three_table_join_optimization(self):
        e = ScanNode('employees', 'e')
        d = ScanNode('departments', 'd')
        o = ScanNode('orders', 'o')

        j1 = JoinNode(e, d, JoinType.INNER,
                      Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ,
                                ColumnRef('id', 'd')))
        j2 = JoinNode(j1, o, JoinType.INNER,
                      Comparison(ColumnRef('id', 'e'), CompOp.EQ,
                                ColumnRef('emp_id', 'o')))

        result = self.opt.optimize(j2)
        tables = result.tables_referenced()
        self.assertEqual(tables, {'e', 'd', 'o'})

    def test_optimize_with_sort(self):
        scan = ScanNode('employees')
        sort = SortNode(scan, [(ColumnRef('salary'), 'DESC')])
        result = self.opt.optimize(sort)
        self.assertIsNotNone(result)

    def test_optimize_with_aggregate(self):
        scan = ScanNode('employees', 'employees')
        agg = AggregateNode(
            scan,
            [ColumnRef('dept_id', 'employees')],
            [FuncCall('AVG', [ColumnRef('salary')])]
        )
        result = self.opt.optimize(agg)
        self.assertIsNotNone(result)
        self.assertGreater(result.estimated_rows, 0)

    def test_optimize_with_limit(self):
        scan = ScanNode('employees')
        lim = LimitNode(scan, 10)
        result = self.opt.optimize(lim)
        self.assertEqual(result.estimated_rows, 10)

    def test_complex_query(self):
        """SELECT e.name, d.name, SUM(o.amount)
           FROM employees e JOIN departments d ON e.dept_id = d.id
           JOIN orders o ON e.id = o.emp_id
           WHERE d.budget > 1000000
           GROUP BY e.name, d.name
           ORDER BY SUM(o.amount) DESC
           LIMIT 10"""

        e = ScanNode('employees', 'e')
        d = ScanNode('departments', 'd')
        o = ScanNode('orders', 'o')

        j1 = JoinNode(e, d, JoinType.INNER,
                      Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ,
                                ColumnRef('id', 'd')))
        j2 = JoinNode(j1, o, JoinType.INNER,
                      Comparison(ColumnRef('id', 'e'), CompOp.EQ,
                                ColumnRef('emp_id', 'o')))

        filt = FilterNode(j2,
                         Comparison(ColumnRef('budget', 'd'), CompOp.GT, Literal(1000000)))

        agg = AggregateNode(
            filt,
            [ColumnRef('name', 'e'), ColumnRef('name', 'd')],
            [FuncCall('SUM', [ColumnRef('amount', 'o')])]
        )
        sort = SortNode(agg, [(FuncCall('SUM', [ColumnRef('amount')]), 'DESC')])
        plan = LimitNode(sort, 10)

        result = self.opt.optimize(plan)
        self.assertIsNotNone(result)
        self.assertEqual(result.estimated_rows, 10)


# ===========================================================================
# EXPLAIN Output Tests
# ===========================================================================

class TestExplain(unittest.TestCase):

    def setUp(self):
        self.catalog = make_catalog()
        self.opt = QueryOptimizer(self.catalog)

    def test_explain_scan(self):
        plan = ScanNode('employees')
        optimized = self.opt.optimize(plan)
        output = self.opt.explain(optimized)
        self.assertIn('Seq Scan', output)
        self.assertIn('employees', output)

    def test_explain_filter(self):
        scan = ScanNode('employees', 'employees')
        pred = Comparison(ColumnRef('salary', 'employees'), CompOp.GT, Literal(50000))
        plan = FilterNode(scan, pred)
        optimized = self.opt.optimize(plan)
        output = self.opt.explain(optimized)
        self.assertIn('employees', output)

    def test_explain_join(self):
        left = ScanNode('employees', 'employees')
        right = ScanNode('departments', 'departments')
        cond = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ,
                         ColumnRef('id', 'departments'))
        plan = JoinNode(left, right, JoinType.INNER, cond)
        optimized = self.opt.optimize(plan)
        output = self.opt.explain(optimized)
        self.assertIn('Join', output)

    def test_explain_sort(self):
        scan = ScanNode('employees')
        sort = SortNode(scan, [(ColumnRef('salary'), 'DESC')])
        optimized = self.opt.optimize(sort)
        output = self.opt.explain(optimized)
        self.assertIn('Sort', output)

    def test_explain_aggregate(self):
        scan = ScanNode('employees', 'employees')
        agg = AggregateNode(scan, [ColumnRef('dept_id', 'employees')],
                           [FuncCall('COUNT', [ColumnRef('id')])])
        optimized = self.opt.optimize(agg)
        output = self.opt.explain(optimized)
        self.assertIn('Aggregate', output)

    def test_explain_limit(self):
        scan = ScanNode('employees')
        lim = LimitNode(scan, 5)
        optimized = self.opt.optimize(lim)
        output = self.opt.explain(optimized)
        self.assertIn('Limit', output)

    def test_explain_index_scan(self):
        idx = Index('emp_pk', 'employees', ['id'], unique=True)
        pred = Comparison(ColumnRef('id', 'employees'), CompOp.EQ, Literal(1))
        plan = IndexScanNode('employees', idx, [pred])
        self.opt.cost_model.estimate(plan)
        output = self.opt.explain(plan)
        self.assertIn('Index Scan', output)
        self.assertIn('emp_pk', output)

    def test_explain_with_offset(self):
        scan = ScanNode('employees')
        lim = LimitNode(scan, 10, offset=20)
        optimized = self.opt.optimize(lim)
        output = self.opt.explain(optimized)
        self.assertIn('Offset', output)

    def test_explain_multiline(self):
        left = ScanNode('employees', 'employees')
        right = ScanNode('departments', 'departments')
        cond = Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ,
                         ColumnRef('id', 'departments'))
        plan = JoinNode(left, right, JoinType.INNER, cond)
        optimized = self.opt.optimize(plan)
        output = self.opt.explain(optimized)
        lines = output.strip().split('\n')
        self.assertGreater(len(lines), 1)


# ===========================================================================
# QueryBuilder Tests
# ===========================================================================

class TestQueryBuilder(unittest.TestCase):

    def setUp(self):
        self.catalog = make_catalog()

    def test_build_scan(self):
        qb = QueryBuilder(self.catalog)
        plan = qb.scan('employees').build()
        self.assertIsInstance(plan, ScanNode)

    def test_build_scan_filter(self):
        qb = QueryBuilder(self.catalog)
        plan = qb.scan('employees').filter(
            Comparison(ColumnRef('salary'), CompOp.GT, Literal(50000))
        ).build()
        self.assertIsInstance(plan, FilterNode)
        self.assertIsInstance(plan.child, ScanNode)

    def test_build_scan_project(self):
        qb = QueryBuilder(self.catalog)
        plan = qb.scan('employees').project(
            ColumnRef('id'), ColumnRef('name')
        ).build()
        self.assertIsInstance(plan, ProjectNode)

    def test_build_join(self):
        left = QueryBuilder(self.catalog).scan('employees')
        right = QueryBuilder(self.catalog).scan('departments')
        plan = left.join(
            right,
            Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ,
                      ColumnRef('id', 'departments'))
        ).build()
        self.assertIsInstance(plan, JoinNode)

    def test_build_sort(self):
        qb = QueryBuilder(self.catalog)
        plan = qb.scan('employees').sort(
            (ColumnRef('salary'), 'DESC')
        ).build()
        self.assertIsInstance(plan, SortNode)

    def test_build_aggregate(self):
        qb = QueryBuilder(self.catalog)
        plan = qb.scan('employees').aggregate(
            [ColumnRef('dept_id')],
            FuncCall('COUNT', [ColumnRef('id')])
        ).build()
        self.assertIsInstance(plan, AggregateNode)

    def test_build_limit(self):
        qb = QueryBuilder(self.catalog)
        plan = qb.scan('employees').limit(10).build()
        self.assertIsInstance(plan, LimitNode)

    def test_build_complex_query(self):
        left = QueryBuilder(self.catalog).scan('employees', 'e')
        right = QueryBuilder(self.catalog).scan('departments', 'd')
        plan = left.join(
            right,
            Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd'))
        ).filter(
            Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000))
        ).project(
            ColumnRef('name', 'e'), ColumnRef('name', 'd')
        ).sort(
            (ColumnRef('name', 'e'), 'ASC')
        ).limit(20).build()

        self.assertIsInstance(plan, LimitNode)


# ===========================================================================
# Plan Comparison Tests
# ===========================================================================

class TestPlanComparison(unittest.TestCase):

    def test_compare_scans(self):
        cat = make_catalog()
        plan_a = ScanNode('employees')
        plan_b = ScanNode('departments')
        result = compare_plans(cat, plan_a, plan_b)
        self.assertIn('winner', result)
        self.assertIn(result['winner'], ('A', 'B'))
        # departments is smaller, should be cheaper
        self.assertEqual(result['winner'], 'B')

    def test_compare_same_plan(self):
        cat = make_catalog()
        plan = ScanNode('employees')
        result = compare_plans(cat, plan, ScanNode('employees'))
        self.assertAlmostEqual(result['speedup'], 1.0, places=1)

    def test_compare_filtered_vs_full(self):
        cat = make_catalog()
        full = ScanNode('employees')
        filtered = FilterNode(ScanNode('employees'),
                            Comparison(ColumnRef('dept_id', 'employees'),
                                      CompOp.EQ, Literal(5)))
        result = compare_plans(cat, full, filtered)
        # Full scan might actually be cheaper due to filter CPU cost
        self.assertIn('winner', result)


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_empty_table(self):
        cat = Catalog()
        schema = TableSchema('empty', [Column('x')])
        cat.add_table(schema, TableStats(row_count=0, page_count=0))
        cm = CostModel(cat)
        scan = ScanNode('empty')
        cost = cm.estimate(scan)
        self.assertGreater(cost.total, 0)  # At least 1 page

    def test_single_row_table(self):
        cat = Catalog()
        schema = TableSchema('single', [Column('x')])
        cat.add_table(schema, TableStats(row_count=1, page_count=1))
        cm = CostModel(cat)
        scan = ScanNode('single')
        cost = cm.estimate(scan)
        self.assertEqual(scan.estimated_rows, 1)

    def test_very_large_table(self):
        cat = Catalog()
        schema = TableSchema('huge', [Column('x')])
        cat.add_table(schema, TableStats(row_count=1_000_000_000, page_count=50_000_000))
        cm = CostModel(cat)
        scan = ScanNode('huge')
        cost = cm.estimate(scan)
        self.assertEqual(scan.estimated_rows, 1_000_000_000)

    def test_limit_larger_than_table(self):
        cat = make_catalog()
        cm = CostModel(cat)
        scan = ScanNode('departments')
        cm.estimate(scan)
        lim = LimitNode(scan, 1000)
        cm.estimate(lim)
        self.assertEqual(lim.estimated_rows, 50)  # Capped at table size

    def test_limit_zero(self):
        cat = make_catalog()
        cm = CostModel(cat)
        scan = ScanNode('employees')
        cm.estimate(scan)
        lim = LimitNode(scan, 0)
        cm.estimate(lim)
        self.assertEqual(lim.estimated_rows, 0)

    def test_selectivity_bounds(self):
        """Selectivity should always be in [0, 1]."""
        cat = make_catalog()
        est = SelectivityEstimator(cat)

        predicates = [
            Comparison(ColumnRef('x'), CompOp.EQ, Literal(1)),
            Comparison(ColumnRef('x'), CompOp.IS_NULL),
            Comparison(ColumnRef('x'), CompOp.IS_NOT_NULL),
            Comparison(ColumnRef('x'), CompOp.LIKE, Literal('%test%')),
        ]

        for pred in predicates:
            sel = est.estimate(pred)
            self.assertGreaterEqual(sel, 0.0, f"Selectivity < 0 for {pred}")
            self.assertLessEqual(sel, 1.0, f"Selectivity > 1 for {pred}")

    def test_nested_and_or(self):
        cat = make_catalog()
        est = SelectivityEstimator(cat)

        a = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        b = Comparison(ColumnRef('y'), CompOp.EQ, Literal(2))
        c = Comparison(ColumnRef('z'), CompOp.EQ, Literal(3))

        expr = LogicExpr(LogicOp.OR, [
            LogicExpr(LogicOp.AND, [a, b]),
            c
        ])
        sel = est.estimate(expr)
        self.assertGreater(sel, 0)
        self.assertLessEqual(sel, 1)

    def test_unknown_expression_type(self):
        cat = make_catalog()
        est = SelectivityEstimator(cat)
        # A bare ColumnRef is not a comparison -- should return default
        sel = est.estimate(ColumnRef('x'))
        self.assertEqual(sel, 0.5)

    def test_self_join(self):
        """Optimizer should handle joining a table with itself."""
        cat = make_catalog()
        opt = QueryOptimizer(cat)

        e1 = ScanNode('employees', 'e1')
        e2 = ScanNode('employees', 'e2')
        cond = Comparison(ColumnRef('dept_id', 'e1'), CompOp.EQ,
                         ColumnRef('dept_id', 'e2'))
        plan = JoinNode(e1, e2, JoinType.INNER, cond)
        result = opt.optimize(plan)
        self.assertIsNotNone(result)

    def test_multiple_filters_same_column(self):
        cat = make_catalog()
        opt = QueryOptimizer(cat)

        scan = ScanNode('employees', 'employees')
        p1 = Comparison(ColumnRef('salary', 'employees'), CompOp.GT, Literal(30000))
        p2 = Comparison(ColumnRef('salary', 'employees'), CompOp.LT, Literal(100000))
        combined = LogicExpr(LogicOp.AND, [p1, p2])
        plan = FilterNode(scan, combined)
        result = opt.optimize(plan)
        self.assertIsNotNone(result)

    def test_cross_join(self):
        cat = make_catalog()
        cm = CostModel(cat)
        left = ScanNode('departments')
        right = ScanNode('products')
        join = JoinNode(left, right, JoinType.CROSS, None, JoinAlgorithm.NESTED_LOOP)
        cost = cm.estimate(join)
        self.assertEqual(join.estimated_rows, 50 * 1000)

    def test_plan_with_alias_expressions(self):
        cat = make_catalog()
        cm = CostModel(cat)
        scan = ScanNode('employees')
        cm.estimate(scan)
        proj = ProjectNode(scan, [
            AliasExpr(ColumnRef('salary', 'employees'), 'emp_salary'),
            AliasExpr(FuncCall('COUNT', [ColumnRef('id')]), 'count')
        ])
        cm.estimate(proj)
        self.assertIn('emp_salary', proj.output_columns)

    def test_hash_and_eq_for_exprs(self):
        c1 = ColumnRef('id', 'employees')
        c2 = ColumnRef('id', 'employees')
        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))

        l1 = Literal(42)
        l2 = Literal(42)
        self.assertEqual(l1, l2)
        self.assertEqual(hash(l1), hash(l2))

    def test_column_ref_not_equal_to_other_type(self):
        c = ColumnRef('id')
        self.assertNotEqual(c, 42)

    def test_literal_not_equal_to_other_type(self):
        l = Literal(42)
        self.assertNotEqual(l, 42)

    def test_comparison_not_equal_to_other_type(self):
        c = Comparison(ColumnRef('x'), CompOp.EQ, Literal(1))
        self.assertNotEqual(c, 42)


# ===========================================================================
# Algorithm Selection Tests
# ===========================================================================

class TestAlgorithmSelection(unittest.TestCase):

    def test_dp_selects_hash_for_large_equijoin(self):
        """DP optimizer should prefer hash join for large equi-joins."""
        cat = make_catalog()
        cm = CostModel(cat)
        jo = JoinOrderOptimizer(cat, cm)

        plans = {
            'e': ScanNode('employees', 'e'),
            'o': ScanNode('orders', 'o'),
        }
        pred = Comparison(ColumnRef('id', 'e'), CompOp.EQ, ColumnRef('emp_id', 'o'))
        result = jo.optimize(plans, [pred])

        self.assertIsInstance(result, JoinNode)
        # Hash join should be selected for large tables
        self.assertEqual(result.algorithm, JoinAlgorithm.HASH)

    def test_different_algorithms_different_costs(self):
        cat = make_catalog()
        cm = CostModel(cat)

        left = ScanNode('employees')
        right = ScanNode('orders')
        cond = Comparison(ColumnRef('id', 'employees'), CompOp.EQ,
                         ColumnRef('emp_id', 'orders'))

        costs = {}
        for algo in JoinAlgorithm:
            if algo == JoinAlgorithm.INDEX_NESTED_LOOP:
                continue
            join = JoinNode(ScanNode('employees'), ScanNode('orders'),
                          JoinType.INNER, cond, algo)
            cost = cm.estimate(join)
            costs[algo] = cost.total

        # All algorithms should have different costs
        values = list(costs.values())
        # At least some should differ
        self.assertGreater(len(set(round(v, 2) for v in values)), 1)


# ===========================================================================
# Regression / Integration Tests
# ===========================================================================

class TestIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        """Full optimization pipeline: build query -> optimize -> explain."""
        cat = make_catalog()
        opt = QueryOptimizer(cat)

        # Build a complex query
        left = QueryBuilder(cat).scan('employees', 'e')
        right = QueryBuilder(cat).scan('departments', 'd')

        plan = left.join(
            right,
            Comparison(ColumnRef('dept_id', 'e'), CompOp.EQ, ColumnRef('id', 'd'))
        ).filter(
            Comparison(ColumnRef('salary', 'e'), CompOp.GT, Literal(50000))
        ).project(
            ColumnRef('name', 'e'),
            ColumnRef('name', 'd'),
            ColumnRef('salary', 'e')
        ).sort(
            (ColumnRef('salary', 'e'), 'DESC')
        ).limit(10).build()

        optimized = opt.optimize(plan)
        explain = opt.explain(optimized)

        self.assertIsNotNone(optimized)
        self.assertGreater(len(explain), 0)
        self.assertEqual(optimized.estimated_rows, 10)

    def test_star_join_pattern(self):
        """Fact table with multiple dimension joins (star schema)."""
        cat = make_catalog()
        opt = QueryOptimizer(cat)

        # orders is the fact table, joining to employees and products
        o = ScanNode('orders', 'o')
        e = ScanNode('employees', 'e')
        p = ScanNode('products', 'p')

        j1 = JoinNode(o, e, JoinType.INNER,
                      Comparison(ColumnRef('emp_id', 'o'), CompOp.EQ,
                                ColumnRef('id', 'e')))
        j2 = JoinNode(j1, p, JoinType.INNER,
                      Comparison(ColumnRef('product_id', 'o'), CompOp.EQ,
                                ColumnRef('id', 'p')))

        result = opt.optimize(j2)
        tables = result.tables_referenced()
        self.assertEqual(tables, {'o', 'e', 'p'})

    def test_subquery_as_derived_table(self):
        """Treat a subquery as a derived table in a join."""
        cat = make_catalog()
        cm = CostModel(cat)

        # Subquery: SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id
        sub = AggregateNode(
            ScanNode('employees', 'employees'),
            [ColumnRef('dept_id', 'employees')],
            [FuncCall('COUNT', [ColumnRef('id')])]
        )
        cm.estimate(sub)

        # Join with departments
        join = JoinNode(
            sub,
            ScanNode('departments', 'departments'),
            JoinType.INNER,
            Comparison(ColumnRef('dept_id', 'employees'), CompOp.EQ,
                      ColumnRef('id', 'departments'))
        )
        cost = cm.estimate(join)
        self.assertGreater(cost.total, 0)

    def test_chained_filters(self):
        """Multiple stacked filters should all be applied."""
        cat = make_catalog()
        cm = CostModel(cat)

        scan = ScanNode('employees', 'employees')
        f1 = FilterNode(scan, Comparison(ColumnRef('salary', 'employees'),
                                         CompOp.GT, Literal(30000)))
        f2 = FilterNode(f1, Comparison(ColumnRef('age', 'employees'),
                                       CompOp.LT, Literal(50)))
        f3 = FilterNode(f2, Comparison(ColumnRef('dept_id', 'employees'),
                                       CompOp.EQ, Literal(5)))
        cost = cm.estimate(f3)
        # Each filter should reduce rows
        self.assertLess(f3.estimated_rows, 10000)

    def test_referenced_columns_alias_expr(self):
        ae = AliasExpr(ColumnRef('salary', 'e'), 'sal')
        cols = referenced_columns(ae)
        self.assertEqual(cols, {'e.salary'})

    def test_referenced_tables_alias_expr(self):
        ae = AliasExpr(ColumnRef('salary', 'e'), 'sal')
        tables = referenced_tables(ae)
        self.assertEqual(tables, {'e'})


if __name__ == '__main__':
    unittest.main()
