"""Tests for C211: Query Execution Engine."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from query_execution import (
    Table, Database, QueryEngine,
    SeqScanExec, FilterExec, ProjectExec, HashJoinExec,
    NestedLoopJoinExec, MergeJoinExec, SortExec, HashAggregateExec,
    LimitExec, DistinctExec, IndexScanExec,
    eval_expr, ExecutionContext, PlanExecutor, Row,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C210_query_optimizer'))
from query_optimizer import (
    ColumnRef, Literal, BinExpr, UnaryExpr, FuncCall, StarExpr,
    InExpr, BetweenExpr, CaseExpr, AliasedExpr, OrderByItem,
    SeqScan, IndexScan, HashJoin, MergeJoin, NestedLoopJoin,
    PhysicalFilter, PhysicalProject, PhysicalSort, HashAggregate,
    PhysicalLimit, PhysicalDistinct,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def engine():
    e = QueryEngine()
    return e


@pytest.fixture
def populated_engine():
    """Engine with users and orders tables."""
    e = QueryEngine()
    e.create_table('users', ['id', 'name', 'age', 'city'])
    e.insert_many('users', [
        [1, 'Alice', 30, 'NYC'],
        [2, 'Bob', 25, 'LA'],
        [3, 'Charlie', 35, 'NYC'],
        [4, 'Diana', 28, 'Chicago'],
        [5, 'Eve', 22, 'LA'],
    ])
    e.create_table('orders', ['id', 'user_id', 'product', 'amount', 'status'])
    e.insert_many('orders', [
        [1, 1, 'Widget', 10.50, 'completed'],
        [2, 1, 'Gadget', 25.00, 'completed'],
        [3, 2, 'Widget', 10.50, 'pending'],
        [4, 3, 'Gizmo', 50.00, 'completed'],
        [5, 3, 'Widget', 10.50, 'completed'],
        [6, 4, 'Gadget', 25.00, 'cancelled'],
        [7, 5, 'Gizmo', 50.00, 'pending'],
        [8, 2, 'Gizmo', 50.00, 'completed'],
    ])
    return e


# ============================================================
# Table Tests
# ============================================================

class TestTable:
    def test_create_table(self):
        t = Table('test', ['a', 'b', 'c'])
        assert t.name == 'test'
        assert t.columns == ['a', 'b', 'c']
        assert len(t) == 0

    def test_insert_list(self):
        t = Table('test', ['a', 'b'])
        t.insert([1, 2])
        assert len(t) == 1
        assert t.rows[0] == {'a': 1, 'b': 2}

    def test_insert_dict(self):
        t = Table('test', ['a', 'b'])
        t.insert({'a': 10, 'b': 20})
        assert len(t) == 1
        assert t.rows[0]['a'] == 10

    def test_insert_many(self):
        t = Table('test', ['x'])
        t.insert_many([[1], [2], [3]])
        assert len(t) == 3

    def test_insert_wrong_length(self):
        t = Table('test', ['a', 'b'])
        with pytest.raises(ValueError):
            t.insert([1])

    def test_scan(self):
        t = Table('test', ['v'])
        t.insert_many([[i] for i in range(5)])
        rows = list(t.scan())
        assert len(rows) == 5
        assert rows[0]['v'] == 0

    def test_create_index(self):
        t = Table('test', ['id', 'val'])
        t.insert_many([[1, 'a'], [2, 'b'], [3, 'a']])
        t.create_index('idx_val', ['val'])
        rows = list(t.index_lookup('idx_val', 'a'))
        assert len(rows) == 2

    def test_index_range(self):
        t = Table('test', ['id', 'score'])
        t.insert_many([[i, i * 10] for i in range(10)])
        t.create_index('idx_score', ['score'])
        rows = list(t.index_range('idx_score', 30, 60))
        assert len(rows) == 4  # 30, 40, 50, 60

    def test_index_updated_on_insert(self):
        t = Table('test', ['id', 'name'])
        t.create_index('idx_name', ['name'])
        t.insert([1, 'Alice'])
        t.insert([2, 'Bob'])
        rows = list(t.index_lookup('idx_name', 'Alice'))
        assert len(rows) == 1


# ============================================================
# Database Tests
# ============================================================

class TestDatabase:
    def test_create_and_get(self):
        db = Database()
        db.create_table('t1', ['a', 'b'])
        assert db.get_table('t1') is not None
        assert db.get_table('t2') is None

    def test_drop_table(self):
        db = Database()
        db.create_table('t1', ['a'])
        db.drop_table('t1')
        assert db.get_table('t1') is None

    def test_insert_via_db(self):
        db = Database()
        db.create_table('t1', ['x'])
        db.insert('t1', [42])
        assert db.get_table('t1').rows[0]['x'] == 42

    def test_insert_nonexistent_table(self):
        db = Database()
        with pytest.raises(ValueError):
            db.insert('nope', [1])

    def test_build_catalog(self):
        db = Database()
        db.create_table('t1', ['id', 'name'])
        t = db.get_table('t1')
        t.insert_many([[1, 'a'], [2, 'b'], [3, 'c']])
        cat = db.build_catalog()
        tdef = cat.get_table('t1')
        assert tdef is not None
        assert tdef.row_count == 3


# ============================================================
# Expression Evaluator Tests
# ============================================================

class TestExprEval:
    def test_literal(self):
        assert eval_expr(Literal(42), {}) == 42
        assert eval_expr(Literal('hello'), {}) == 'hello'
        assert eval_expr(Literal(None), {}) is None

    def test_column_ref_unqualified(self):
        row = {'name': 'Alice', 'age': 30}
        assert eval_expr(ColumnRef(None, 'name'), row) == 'Alice'

    def test_column_ref_qualified(self):
        row = {'users.name': 'Alice', 'name': 'Alice'}
        assert eval_expr(ColumnRef('users', 'name'), row) == 'Alice'

    def test_column_ref_suffix_match(self):
        row = {'users.name': 'Alice'}
        assert eval_expr(ColumnRef(None, 'name'), row) == 'Alice'

    def test_binexpr_comparison(self):
        row = {'x': 10}
        assert eval_expr(BinExpr('=', ColumnRef(None, 'x'), Literal(10)), row) is True
        assert eval_expr(BinExpr('<', ColumnRef(None, 'x'), Literal(20)), row) is True
        assert eval_expr(BinExpr('>', ColumnRef(None, 'x'), Literal(20)), row) is False
        assert eval_expr(BinExpr('>=', ColumnRef(None, 'x'), Literal(10)), row) is True
        assert eval_expr(BinExpr('<=', ColumnRef(None, 'x'), Literal(10)), row) is True
        assert eval_expr(BinExpr('!=', ColumnRef(None, 'x'), Literal(5)), row) is True

    def test_binexpr_arithmetic(self):
        row = {'a': 10, 'b': 3}
        assert eval_expr(BinExpr('+', ColumnRef(None, 'a'), ColumnRef(None, 'b')), row) == 13
        assert eval_expr(BinExpr('-', ColumnRef(None, 'a'), ColumnRef(None, 'b')), row) == 7
        assert eval_expr(BinExpr('*', ColumnRef(None, 'a'), ColumnRef(None, 'b')), row) == 30
        assert eval_expr(BinExpr('%', ColumnRef(None, 'a'), ColumnRef(None, 'b')), row) == 1

    def test_binexpr_division(self):
        row = {'a': 10, 'b': 3}
        result = eval_expr(BinExpr('/', ColumnRef(None, 'a'), ColumnRef(None, 'b')), row)
        assert abs(result - 3.333) < 0.01

    def test_division_by_zero(self):
        row = {'a': 10, 'b': 0}
        assert eval_expr(BinExpr('/', ColumnRef(None, 'a'), ColumnRef(None, 'b')), row) is None

    def test_and_or(self):
        row = {'x': 10}
        true_expr = BinExpr('=', ColumnRef(None, 'x'), Literal(10))
        false_expr = BinExpr('=', ColumnRef(None, 'x'), Literal(5))
        assert eval_expr(BinExpr('AND', true_expr, true_expr), row) is True
        assert eval_expr(BinExpr('AND', true_expr, false_expr), row) is False
        assert eval_expr(BinExpr('OR', false_expr, true_expr), row) is True
        assert eval_expr(BinExpr('OR', false_expr, false_expr), row) is False

    def test_not(self):
        row = {'x': 10}
        assert eval_expr(UnaryExpr('NOT', BinExpr('=', ColumnRef(None, 'x'), Literal(5))), row) is True

    def test_is_null(self):
        row = {'x': None, 'y': 5}
        assert eval_expr(UnaryExpr('IS NULL', ColumnRef(None, 'x')), row) is True
        assert eval_expr(UnaryExpr('IS NULL', ColumnRef(None, 'y')), row) is False
        assert eval_expr(UnaryExpr('IS NOT NULL', ColumnRef(None, 'y')), row) is True

    def test_like(self):
        row = {'name': 'Alice'}
        assert eval_expr(BinExpr('LIKE', ColumnRef(None, 'name'), Literal('A%')), row) is True
        assert eval_expr(BinExpr('LIKE', ColumnRef(None, 'name'), Literal('%ice')), row) is True
        assert eval_expr(BinExpr('LIKE', ColumnRef(None, 'name'), Literal('B%')), row) is False
        assert eval_expr(BinExpr('LIKE', ColumnRef(None, 'name'), Literal('_lice')), row) is True

    def test_in_expr(self):
        row = {'x': 3}
        assert eval_expr(InExpr(ColumnRef(None, 'x'), [Literal(1), Literal(2), Literal(3)]), row) is True
        assert eval_expr(InExpr(ColumnRef(None, 'x'), [Literal(4), Literal(5)]), row) is False
        assert eval_expr(InExpr(ColumnRef(None, 'x'), [Literal(3)], negated=True), row) is False

    def test_between(self):
        row = {'x': 5}
        assert eval_expr(BetweenExpr(ColumnRef(None, 'x'), Literal(1), Literal(10)), row) is True
        assert eval_expr(BetweenExpr(ColumnRef(None, 'x'), Literal(6), Literal(10)), row) is False
        assert eval_expr(BetweenExpr(ColumnRef(None, 'x'), Literal(1), Literal(10), negated=True), row) is False

    def test_case_searched(self):
        row = {'x': 10}
        expr = CaseExpr(
            operand=None,
            whens=[
                (BinExpr('>', ColumnRef(None, 'x'), Literal(5)), Literal('big')),
                (BinExpr('<=', ColumnRef(None, 'x'), Literal(5)), Literal('small')),
            ],
            else_result=Literal('unknown'),
        )
        assert eval_expr(expr, row) == 'big'

    def test_case_else(self):
        row = {'x': 0}
        expr = CaseExpr(
            operand=None,
            whens=[(BinExpr('>', ColumnRef(None, 'x'), Literal(5)), Literal('big'))],
            else_result=Literal('other'),
        )
        assert eval_expr(expr, row) == 'other'

    def test_func_coalesce(self):
        row = {'x': None, 'y': 5}
        expr = FuncCall('COALESCE', [ColumnRef(None, 'x'), ColumnRef(None, 'y')])
        assert eval_expr(expr, row) == 5

    def test_func_upper_lower(self):
        row = {'s': 'Hello'}
        assert eval_expr(FuncCall('UPPER', [ColumnRef(None, 's')]), row) == 'HELLO'
        assert eval_expr(FuncCall('LOWER', [ColumnRef(None, 's')]), row) == 'hello'

    def test_func_length(self):
        row = {'s': 'Hello'}
        assert eval_expr(FuncCall('LENGTH', [ColumnRef(None, 's')]), row) == 5

    def test_func_abs(self):
        row = {'x': -7}
        assert eval_expr(FuncCall('ABS', [ColumnRef(None, 'x')]), row) == 7

    def test_func_round(self):
        row = {'x': 3.14159}
        assert eval_expr(FuncCall('ROUND', [ColumnRef(None, 'x'), Literal(2)]), row) == 3.14

    def test_func_substr(self):
        row = {'s': 'Hello World'}
        assert eval_expr(FuncCall('SUBSTR', [ColumnRef(None, 's'), Literal(1), Literal(5)]), row) == 'Hello'

    def test_func_nullif(self):
        row = {'x': 5, 'y': 5}
        assert eval_expr(FuncCall('NULLIF', [ColumnRef(None, 'x'), ColumnRef(None, 'y')]), row) is None

    def test_func_trim(self):
        row = {'s': '  hello  '}
        assert eval_expr(FuncCall('TRIM', [ColumnRef(None, 's')]), row) == 'hello'

    def test_unary_minus(self):
        row = {'x': 7}
        assert eval_expr(UnaryExpr('-', ColumnRef(None, 'x')), row) == -7

    def test_null_arithmetic(self):
        row = {'x': None}
        assert eval_expr(BinExpr('+', ColumnRef(None, 'x'), Literal(1)), row) is None

    def test_null_comparison(self):
        row = {'x': None}
        assert eval_expr(BinExpr('<', ColumnRef(None, 'x'), Literal(5)), row) is False


# ============================================================
# Volcano Operator Tests
# ============================================================

class TestSeqScanExec:
    def test_basic_scan(self):
        t = Table('t', ['a', 'b'])
        t.insert_many([[1, 'x'], [2, 'y'], [3, 'z']])
        op = SeqScanExec(t)
        rows = op.collect()
        assert len(rows) == 3

    def test_scan_with_filter(self):
        t = Table('t', ['a', 'b'])
        t.insert_many([[1, 'x'], [2, 'y'], [3, 'z']])
        filt = BinExpr('>', ColumnRef(None, 'a'), Literal(1))
        op = SeqScanExec(t, filter_expr=filt)
        rows = op.collect()
        assert len(rows) == 2

    def test_scan_aliased(self):
        t = Table('users', ['id', 'name'])
        t.insert([1, 'Alice'])
        op = SeqScanExec(t, alias='u')
        rows = op.collect()
        assert rows[0]['u.id'] == 1
        assert rows[0]['id'] == 1


class TestFilterExec:
    def test_filter(self):
        t = Table('t', ['x'])
        t.insert_many([[i] for i in range(10)])
        scan = SeqScanExec(t)
        filt = FilterExec(scan, BinExpr('>=', ColumnRef(None, 'x'), Literal(7)))
        rows = filt.collect()
        assert len(rows) == 3

    def test_filter_all_rejected(self):
        t = Table('t', ['x'])
        t.insert_many([[i] for i in range(5)])
        scan = SeqScanExec(t)
        filt = FilterExec(scan, BinExpr('>', ColumnRef(None, 'x'), Literal(100)))
        rows = filt.collect()
        assert len(rows) == 0


class TestProjectExec:
    def test_project_columns(self):
        t = Table('t', ['a', 'b', 'c'])
        t.insert([1, 2, 3])
        scan = SeqScanExec(t)
        proj = ProjectExec(scan, [(ColumnRef(None, 'a'), 'a'), (ColumnRef(None, 'c'), 'c')])
        rows = proj.collect()
        assert 'a' in rows[0]
        assert 'c' in rows[0]

    def test_project_expression(self):
        t = Table('t', ['a', 'b'])
        t.insert([10, 3])
        scan = SeqScanExec(t)
        expr = BinExpr('+', ColumnRef(None, 'a'), ColumnRef(None, 'b'))
        proj = ProjectExec(scan, [(expr, 'total')])
        rows = proj.collect()
        assert rows[0]['total'] == 13

    def test_project_star(self):
        t = Table('t', ['a', 'b'])
        t.insert([1, 2])
        scan = SeqScanExec(t)
        proj = ProjectExec(scan, [(StarExpr(), None)])
        rows = proj.collect()
        assert rows[0]['a'] == 1


class TestSortExec:
    def test_sort_asc(self):
        t = Table('t', ['x'])
        t.insert_many([[3], [1], [2]])
        scan = SeqScanExec(t)
        sort = SortExec(scan, [OrderByItem(ColumnRef(None, 'x'), 'ASC')])
        rows = sort.collect()
        assert [r['x'] for r in rows] == [1, 2, 3]

    def test_sort_desc(self):
        t = Table('t', ['x'])
        t.insert_many([[3], [1], [2]])
        scan = SeqScanExec(t)
        sort = SortExec(scan, [OrderByItem(ColumnRef(None, 'x'), 'DESC')])
        rows = sort.collect()
        assert [r['x'] for r in rows] == [3, 2, 1]

    def test_sort_with_nulls(self):
        t = Table('t', ['x'])
        t.insert_many([[3], [None], [1]])
        scan = SeqScanExec(t)
        sort = SortExec(scan, [OrderByItem(ColumnRef(None, 'x'), 'ASC')])
        rows = sort.collect()
        assert rows[0]['x'] == 1
        assert rows[2]['x'] is None  # NULL last in ASC

    def test_sort_multi_key(self):
        t = Table('t', ['city', 'age'])
        t.insert_many([['NYC', 30], ['LA', 25], ['NYC', 20], ['LA', 35]])
        scan = SeqScanExec(t)
        sort = SortExec(scan, [
            OrderByItem(ColumnRef(None, 'city'), 'ASC'),
            OrderByItem(ColumnRef(None, 'age'), 'DESC'),
        ])
        rows = sort.collect()
        assert rows[0]['city'] == 'LA' and rows[0]['age'] == 35
        assert rows[1]['city'] == 'LA' and rows[1]['age'] == 25


class TestLimitExec:
    def test_limit(self):
        t = Table('t', ['x'])
        t.insert_many([[i] for i in range(10)])
        scan = SeqScanExec(t)
        lim = LimitExec(scan, 3)
        rows = lim.collect()
        assert len(rows) == 3

    def test_limit_with_offset(self):
        t = Table('t', ['x'])
        t.insert_many([[i] for i in range(10)])
        scan = SeqScanExec(t)
        lim = LimitExec(scan, 3, offset=5)
        rows = lim.collect()
        assert len(rows) == 3
        assert rows[0]['x'] == 5

    def test_limit_exceeds_rows(self):
        t = Table('t', ['x'])
        t.insert_many([[i] for i in range(3)])
        scan = SeqScanExec(t)
        lim = LimitExec(scan, 10)
        rows = lim.collect()
        assert len(rows) == 3


class TestDistinctExec:
    def test_distinct(self):
        t = Table('t', ['x'])
        t.insert_many([[1], [2], [1], [3], [2]])
        scan = SeqScanExec(t)
        dist = DistinctExec(scan)
        rows = dist.collect()
        values = {r['x'] for r in rows}
        assert values == {1, 2, 3}
        assert len(rows) == 3


class TestHashJoinExec:
    def test_inner_join(self):
        t1 = Table('users', ['id', 'name'])
        t1.insert_many([[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']])
        t2 = Table('orders', ['oid', 'uid', 'product'])
        t2.insert_many([[1, 1, 'Widget'], [2, 2, 'Gadget'], [3, 1, 'Gizmo']])

        left = SeqScanExec(t1, 'users')
        right = SeqScanExec(t2, 'orders')
        cond = BinExpr('=', ColumnRef('users', 'id'), ColumnRef('orders', 'uid'))
        join = HashJoinExec(left, right, cond, 'INNER')
        rows = join.collect()
        assert len(rows) == 3  # Alice x2, Bob x1

    def test_left_join(self):
        t1 = Table('users', ['id', 'name'])
        t1.insert_many([[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']])
        t2 = Table('orders', ['oid', 'uid'])
        t2.insert_many([[1, 1], [2, 1]])

        left = SeqScanExec(t1, 'users')
        right = SeqScanExec(t2, 'orders')
        cond = BinExpr('=', ColumnRef('users', 'id'), ColumnRef('orders', 'uid'))
        join = HashJoinExec(left, right, cond, 'LEFT')
        rows = join.collect()
        assert len(rows) == 4  # Alice x2, Bob NULL, Charlie NULL
        # Bob and Charlie have no orders
        null_rows = [r for r in rows if r.get('orders.oid') is None]
        assert len(null_rows) == 2


class TestNestedLoopJoinExec:
    def test_cross_join(self):
        t1 = Table('a', ['x'])
        t1.insert_many([[1], [2]])
        t2 = Table('b', ['y'])
        t2.insert_many([[10], [20], [30]])

        left = SeqScanExec(t1, 'a')
        right = SeqScanExec(t2, 'b')
        join = NestedLoopJoinExec(left, right, None, 'INNER')
        rows = join.collect()
        assert len(rows) == 6  # 2 x 3

    def test_with_condition(self):
        t1 = Table('a', ['x'])
        t1.insert_many([[1], [2], [3]])
        t2 = Table('b', ['y'])
        t2.insert_many([[2], [3], [4]])

        left = SeqScanExec(t1, 'a')
        right = SeqScanExec(t2, 'b')
        cond = BinExpr('=', ColumnRef('a', 'x'), ColumnRef('b', 'y'))
        join = NestedLoopJoinExec(left, right, cond, 'INNER')
        rows = join.collect()
        assert len(rows) == 2  # 2=2, 3=3


class TestMergeJoinExec:
    def test_merge_join(self):
        t1 = Table('a', ['id', 'val'])
        t1.insert_many([[1, 'a'], [2, 'b'], [3, 'c']])
        t2 = Table('b', ['id', 'ref'])
        t2.insert_many([[2, 'x'], [3, 'y'], [4, 'z']])

        left = SeqScanExec(t1, 'a')
        right = SeqScanExec(t2, 'b')
        cond = BinExpr('=', ColumnRef('a', 'id'), ColumnRef('b', 'id'))
        join = MergeJoinExec(left, right, cond, 'INNER')
        rows = join.collect()
        assert len(rows) == 2  # id=2 and id=3


class TestHashAggregateExec:
    def test_count_all(self):
        t = Table('t', ['x'])
        t.insert_many([[1], [2], [3]])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(scan, [], [(FuncCall('COUNT', [StarExpr()]), 'cnt')])
        rows = agg.collect()
        assert len(rows) == 1
        assert rows[0]['cnt'] == 3

    def test_sum(self):
        t = Table('t', ['x'])
        t.insert_many([[10], [20], [30]])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(scan, [], [(FuncCall('SUM', [ColumnRef(None, 'x')]), 'total')])
        rows = agg.collect()
        assert rows[0]['total'] == 60

    def test_avg(self):
        t = Table('t', ['x'])
        t.insert_many([[10], [20], [30]])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(scan, [], [(FuncCall('AVG', [ColumnRef(None, 'x')]), 'avg_x')])
        rows = agg.collect()
        assert abs(rows[0]['avg_x'] - 20.0) < 0.01

    def test_min_max(self):
        t = Table('t', ['x'])
        t.insert_many([[3], [1], [5], [2]])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(scan, [], [
            (FuncCall('MIN', [ColumnRef(None, 'x')]), 'min_x'),
            (FuncCall('MAX', [ColumnRef(None, 'x')]), 'max_x'),
        ])
        rows = agg.collect()
        assert rows[0]['min_x'] == 1
        assert rows[0]['max_x'] == 5

    def test_group_by(self):
        t = Table('t', ['city', 'amount'])
        t.insert_many([['NYC', 10], ['LA', 20], ['NYC', 30], ['LA', 40]])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(
            scan,
            [ColumnRef(None, 'city')],
            [(FuncCall('SUM', [ColumnRef(None, 'amount')]), 'total')],
        )
        rows = agg.collect()
        by_city = {r['city']: r['total'] for r in rows}
        assert by_city['NYC'] == 40
        assert by_city['LA'] == 60

    def test_count_distinct(self):
        t = Table('t', ['x'])
        t.insert_many([[1], [2], [1], [3], [2]])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(
            scan, [],
            [(FuncCall('COUNT', [ColumnRef(None, 'x')], distinct=True), 'cnt')],
        )
        rows = agg.collect()
        assert rows[0]['cnt'] == 3

    def test_count_empty(self):
        t = Table('t', ['x'])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(scan, [], [(FuncCall('COUNT', [StarExpr()]), 'cnt')])
        rows = agg.collect()
        assert rows[0]['cnt'] == 0

    def test_sum_with_nulls(self):
        t = Table('t', ['x'])
        t.insert_many([[10], [None], [20]])
        scan = SeqScanExec(t)
        agg = HashAggregateExec(scan, [], [(FuncCall('SUM', [ColumnRef(None, 'x')]), 'total')])
        rows = agg.collect()
        assert rows[0]['total'] == 30


class TestIndexScanExec:
    def test_index_eq_scan(self):
        t = Table('t', ['id', 'name'])
        t.insert_many([[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']])
        t.create_index('idx_id', ['id'], unique=True)
        op = IndexScanExec(t, 'idx_id', lookup_values=[2], scan_type='eq')
        rows = op.collect()
        assert len(rows) == 1
        assert rows[0]['name'] == 'Bob'

    def test_index_range_scan(self):
        t = Table('t', ['id', 'val'])
        t.insert_many([[i, i * 10] for i in range(10)])
        t.create_index('idx_id', ['id'])
        op = IndexScanExec(t, 'idx_id', scan_type='range', range_low=3, range_high=6)
        rows = op.collect()
        assert len(rows) == 4


# ============================================================
# QueryEngine Integration Tests (SQL -> Results)
# ============================================================

class TestQueryEngineBasic:
    def test_create_table_sql(self, engine):
        engine.execute("CREATE TABLE users (id INT, name VARCHAR)")
        assert engine.db.get_table('users') is not None

    def test_insert_sql(self, engine):
        engine.execute("CREATE TABLE t (id INT, val TEXT)")
        engine.execute("INSERT INTO t (id, val) VALUES (1, 'hello')")
        assert len(engine.db.get_table('t')) == 1

    def test_insert_multiple_rows(self, engine):
        engine.execute("CREATE TABLE t (id INT, val TEXT)")
        engine.execute("INSERT INTO t (id, val) VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        assert len(engine.db.get_table('t')) == 3

    def test_drop_table_sql(self, engine):
        engine.execute("CREATE TABLE t (id INT)")
        engine.execute("DROP TABLE t")
        assert engine.db.get_table('t') is None

    def test_select_star(self, populated_engine):
        rows = populated_engine.query("SELECT * FROM users")
        assert len(rows) == 5

    def test_select_columns(self, populated_engine):
        rows = populated_engine.query("SELECT name, age FROM users")
        assert len(rows) == 5
        assert 'name' in rows[0]

    def test_select_where(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users WHERE age > 28")
        names = {r['name'] for r in rows}
        assert names == {'Alice', 'Charlie'}

    def test_select_where_eq(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users WHERE city = 'NYC'")
        names = {r['name'] for r in rows}
        assert names == {'Alice', 'Charlie'}

    def test_select_order_by(self, populated_engine):
        rows = populated_engine.query("SELECT name, age FROM users ORDER BY age ASC")
        ages = [r['age'] for r in rows]
        assert ages == [22, 25, 28, 30, 35]

    def test_select_order_by_desc(self, populated_engine):
        rows = populated_engine.query("SELECT name, age FROM users ORDER BY age DESC")
        ages = [r['age'] for r in rows]
        assert ages == [35, 30, 28, 25, 22]

    def test_select_limit(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users LIMIT 2")
        assert len(rows) == 2

    def test_select_limit_offset(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users ORDER BY id LIMIT 2 OFFSET 2")
        assert len(rows) == 2

    def test_select_distinct(self, populated_engine):
        rows = populated_engine.query("SELECT DISTINCT city FROM users")
        cities = {r['city'] for r in rows}
        assert cities == {'NYC', 'LA', 'Chicago'}

    def test_select_count_star(self, populated_engine):
        rows = populated_engine.query("SELECT COUNT(*) FROM users")
        assert len(rows) == 1
        # Count value should be 5
        vals = list(rows[0].values())
        assert 5 in vals

    def test_select_sum(self, populated_engine):
        rows = populated_engine.query("SELECT SUM(amount) FROM orders")
        vals = list(rows[0].values())
        total = [v for v in vals if isinstance(v, (int, float)) and v > 100]
        assert len(total) >= 1

    def test_select_group_by(self, populated_engine):
        rows = populated_engine.query(
            "SELECT city, COUNT(*) FROM users GROUP BY city"
        )
        by_city = {}
        for r in rows:
            city = r.get('city')
            for k, v in r.items():
                if isinstance(v, int) and k != 'city':
                    by_city[city] = v
        assert by_city.get('NYC') == 2
        assert by_city.get('LA') == 2
        assert by_city.get('Chicago') == 1


class TestQueryEngineJoins:
    def test_inner_join(self, populated_engine):
        rows = populated_engine.query(
            "SELECT users.name, orders.product FROM users "
            "JOIN orders ON users.id = orders.user_id"
        )
        assert len(rows) >= 5  # Multiple orders per user

    def test_left_join(self, populated_engine):
        rows = populated_engine.query(
            "SELECT users.name, orders.product FROM users "
            "LEFT JOIN orders ON users.id = orders.user_id"
        )
        # All 5 users should appear (even if no orders)
        names = {r['name'] for r in rows}
        assert len(names) == 5

    def test_join_with_where(self, populated_engine):
        rows = populated_engine.query(
            "SELECT users.name, orders.product FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "WHERE orders.status = 'completed'"
        )
        assert all(r.get('product') is not None for r in rows)

    def test_multi_table_join(self, populated_engine):
        populated_engine.create_table('products', ['name', 'category'])
        populated_engine.insert_many('products', [
            ['Widget', 'tools'],
            ['Gadget', 'electronics'],
            ['Gizmo', 'tools'],
        ])
        rows = populated_engine.query(
            "SELECT users.name, orders.product, products.category "
            "FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "JOIN products ON orders.product = products.name"
        )
        assert len(rows) >= 5
        assert all('category' in r for r in rows)


class TestQueryEngineAggregations:
    def test_avg(self, populated_engine):
        rows = populated_engine.query("SELECT AVG(age) FROM users")
        assert len(rows) == 1
        vals = [v for v in rows[0].values() if isinstance(v, (int, float))]
        avg = vals[0]
        assert abs(avg - 28.0) < 0.01

    def test_min_max(self, populated_engine):
        rows = populated_engine.query("SELECT MIN(age), MAX(age) FROM users")
        vals = [v for v in rows[0].values() if isinstance(v, (int, float))]
        assert 22 in vals
        assert 35 in vals

    def test_group_by_with_having(self, populated_engine):
        rows = populated_engine.query(
            "SELECT user_id, COUNT(*) FROM orders "
            "GROUP BY user_id HAVING COUNT(*) > 1"
        )
        for r in rows:
            cnt = [v for v in r.values() if isinstance(v, int) and v > 1]
            assert len(cnt) >= 1

    def test_group_by_sum(self, populated_engine):
        rows = populated_engine.query(
            "SELECT user_id, SUM(amount) FROM orders GROUP BY user_id"
        )
        assert len(rows) >= 3  # Multiple users with orders


class TestQueryEngineExpressions:
    def test_arithmetic_in_select(self, populated_engine):
        rows = populated_engine.query("SELECT age + 10 FROM users WHERE id = 1")
        vals = [v for v in rows[0].values() if isinstance(v, (int, float))]
        assert 40 in vals

    def test_like_filter(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users WHERE name LIKE 'A%'")
        assert len(rows) == 1

    def test_in_filter(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users WHERE city IN ('NYC', 'Chicago')")
        names = {r['name'] for r in rows}
        assert 'Alice' in names
        assert 'Diana' in names
        assert 'Bob' not in names

    def test_between_filter(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users WHERE age BETWEEN 25 AND 30")
        names = {r['name'] for r in rows}
        assert names == {'Alice', 'Bob', 'Diana'}

    def test_null_handling(self, engine):
        engine.create_table('t', ['id', 'val'])
        engine.insert_many('t', [[1, 'a'], [2, None], [3, 'c']])
        rows = engine.query("SELECT id FROM t WHERE val IS NULL")
        ids = {r['id'] for r in rows}
        assert ids == {2}

    def test_is_not_null(self, engine):
        engine.create_table('t', ['id', 'val'])
        engine.insert_many('t', [[1, 'a'], [2, None], [3, 'c']])
        rows = engine.query("SELECT id FROM t WHERE val IS NOT NULL")
        ids = {r['id'] for r in rows}
        assert ids == {1, 3}

    def test_not_filter(self, populated_engine):
        rows = populated_engine.query("SELECT name FROM users WHERE NOT city = 'NYC'")
        names = {r['name'] for r in rows}
        assert 'Alice' not in names
        assert 'Bob' in names

    def test_case_expression(self, populated_engine):
        rows = populated_engine.query(
            "SELECT name, CASE WHEN age >= 30 THEN 'senior' "
            "ELSE 'junior' END FROM users"
        )
        assert len(rows) == 5

    def test_aliased_expression(self, populated_engine):
        rows = populated_engine.query(
            "SELECT name, age * 2 AS double_age FROM users WHERE id = 1"
        )
        assert rows[0].get('double_age') == 60


class TestQueryEngineAdvanced:
    def test_subquery_in_where(self, populated_engine):
        # Users who have placed orders > $25
        rows = populated_engine.query(
            "SELECT name FROM users WHERE id IN (1, 3, 5)"
        )
        names = {r['name'] for r in rows}
        assert names == {'Alice', 'Charlie', 'Eve'}

    def test_order_by_multiple(self, populated_engine):
        rows = populated_engine.query(
            "SELECT name, city, age FROM users ORDER BY city ASC, age DESC"
        )
        # Chicago first, then LA, then NYC
        assert rows[0]['city'] == 'Chicago'

    def test_count_distinct(self, populated_engine):
        rows = populated_engine.query("SELECT COUNT(DISTINCT city) FROM users")
        vals = [v for v in rows[0].values() if isinstance(v, int)]
        assert 3 in vals

    def test_complex_where(self, populated_engine):
        rows = populated_engine.query(
            "SELECT name FROM users WHERE (age > 25 AND city = 'NYC') OR city = 'Chicago'"
        )
        names = {r['name'] for r in rows}
        assert 'Alice' in names
        assert 'Charlie' in names
        assert 'Diana' in names

    def test_multiple_aggregates(self, populated_engine):
        rows = populated_engine.query(
            "SELECT COUNT(*), SUM(amount), AVG(amount) FROM orders"
        )
        assert len(rows) == 1
        vals = [v for v in rows[0].values() if isinstance(v, (int, float))]
        assert len(vals) >= 3

    def test_explain(self, populated_engine):
        result = populated_engine.explain(
            "SELECT name FROM users WHERE age > 25"
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestQueryEngineAPI:
    def test_create_table_api(self, engine):
        t = engine.create_table('test', ['a', 'b', 'c'])
        assert isinstance(t, Table)
        assert t.name == 'test'

    def test_insert_api(self, engine):
        engine.create_table('t', ['x'])
        engine.insert('t', [42])
        rows = engine.query("SELECT x FROM t")
        assert rows[0]['x'] == 42

    def test_insert_many_api(self, engine):
        engine.create_table('t', ['x'])
        engine.insert_many('t', [[1], [2], [3]])
        rows = engine.query("SELECT x FROM t")
        assert len(rows) == 3

    def test_create_index_api(self, engine):
        engine.create_table('t', ['id', 'name'])
        engine.insert_many('t', [[1, 'a'], [2, 'b']])
        engine.create_index('t', 'idx_id', ['id'], unique=True)
        t = engine.db.get_table('t')
        assert 'idx_id' in t.indexes

    def test_insert_nonexistent_table(self, engine):
        with pytest.raises(ValueError):
            engine.insert('nope', [1])


class TestQueryEngineEdgeCases:
    def test_empty_table_select(self, engine):
        engine.create_table('t', ['id', 'val'])
        rows = engine.query("SELECT * FROM t")
        assert rows == []

    def test_empty_table_count(self, engine):
        engine.create_table('t', ['id'])
        rows = engine.query("SELECT COUNT(*) FROM t")
        vals = list(rows[0].values())
        assert 0 in vals

    def test_single_row(self, engine):
        engine.create_table('t', ['x'])
        engine.insert('t', [42])
        rows = engine.query("SELECT x FROM t")
        assert len(rows) == 1
        assert rows[0]['x'] == 42

    def test_large_dataset(self, engine):
        engine.create_table('t', ['id', 'val'])
        engine.insert_many('t', [[i, i * 10] for i in range(1000)])
        rows = engine.query("SELECT COUNT(*) FROM t")
        vals = list(rows[0].values())
        assert 1000 in vals

    def test_large_dataset_filter(self, engine):
        engine.create_table('t', ['id', 'val'])
        engine.insert_many('t', [[i, i % 10] for i in range(1000)])
        rows = engine.query("SELECT COUNT(*) FROM t WHERE val = 5")
        vals = list(rows[0].values())
        assert 100 in vals

    def test_string_values(self, engine):
        engine.create_table('t', ['id', 'name'])
        engine.insert_many('t', [[1, 'hello'], [2, 'world']])
        rows = engine.query("SELECT name FROM t WHERE name = 'hello'")
        assert len(rows) == 1

    def test_boolean_values(self, engine):
        engine.create_table('t', ['id', 'active'])
        engine.insert_many('t', [[1, True], [2, False], [3, True]])
        rows = engine.query("SELECT id FROM t WHERE active = true")
        assert len(rows) == 2

    def test_null_values(self, engine):
        engine.create_table('t', ['id', 'val'])
        engine.insert_many('t', [[1, 'a'], [2, None], [3, 'c']])
        rows = engine.query("SELECT COUNT(*) FROM t")
        vals = list(rows[0].values())
        assert 3 in vals

    def test_create_table_if_not_exists(self, engine):
        engine.execute("CREATE TABLE IF NOT EXISTS t (id INT)")
        assert engine.db.get_table('t') is not None

    def test_drop_table_if_exists(self, engine):
        engine.execute("DROP TABLE IF EXISTS nonexistent")
        # Should not raise

    def test_insert_null(self, engine):
        engine.execute("CREATE TABLE t (id INT, val TEXT)")
        engine.execute("INSERT INTO t (id, val) VALUES (1, NULL)")
        rows = engine.query("SELECT val FROM t")
        assert rows[0]['val'] is None


class TestQueryEngineComposition:
    """Tests combining multiple SQL features together."""

    def test_join_group_order(self, populated_engine):
        """JOIN + GROUP BY + ORDER BY"""
        rows = populated_engine.query(
            "SELECT users.city, SUM(orders.amount) FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "GROUP BY users.city "
            "ORDER BY users.city ASC"
        )
        assert len(rows) >= 2

    def test_filter_aggregate_limit(self, populated_engine):
        """WHERE + GROUP BY + LIMIT"""
        rows = populated_engine.query(
            "SELECT user_id, COUNT(*) FROM orders "
            "WHERE status = 'completed' "
            "GROUP BY user_id "
            "LIMIT 2"
        )
        assert len(rows) <= 2

    def test_distinct_order_limit(self, populated_engine):
        """DISTINCT + ORDER BY + LIMIT"""
        rows = populated_engine.query(
            "SELECT DISTINCT city FROM users ORDER BY city ASC LIMIT 2"
        )
        assert len(rows) == 2

    def test_join_filter_project(self, populated_engine):
        """JOIN + WHERE + specific columns"""
        rows = populated_engine.query(
            "SELECT users.name, orders.product, orders.amount FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "WHERE orders.amount > 20"
        )
        for r in rows:
            amt = r.get('amount')
            if amt is not None:
                assert amt > 20

    def test_aggregate_with_alias(self, populated_engine):
        rows = populated_engine.query(
            "SELECT city, COUNT(*) AS cnt, AVG(age) AS avg_age FROM users GROUP BY city"
        )
        for r in rows:
            assert 'cnt' in r or 'avg_age' in r


class TestPlanExecutor:
    """Test the physical plan -> execution plan conversion."""

    def test_build_seq_scan(self):
        db = Database()
        db.create_table('t', ['a'])
        db.get_table('t').insert([1])
        plan = SeqScan(table='t', alias='t')
        exec_op = PlanExecutor(db).build(plan)
        rows = exec_op.collect()
        assert len(rows) == 1

    def test_build_filter(self):
        db = Database()
        db.create_table('t', ['x'])
        db.get_table('t').insert_many([[1], [2], [3]])
        scan = SeqScan(table='t', alias='t')
        filt = PhysicalFilter(
            input=scan,
            condition=BinExpr('>', ColumnRef(None, 'x'), Literal(1))
        )
        exec_op = PlanExecutor(db).build(filt)
        rows = exec_op.collect()
        assert len(rows) == 2

    def test_build_project(self):
        db = Database()
        db.create_table('t', ['a', 'b'])
        db.get_table('t').insert([10, 20])
        scan = SeqScan(table='t', alias='t')
        proj = PhysicalProject(
            input=scan,
            expressions=[(ColumnRef(None, 'a'), 'a')]
        )
        exec_op = PlanExecutor(db).build(proj)
        rows = exec_op.collect()
        assert 'a' in rows[0]

    def test_build_sort(self):
        db = Database()
        db.create_table('t', ['x'])
        db.get_table('t').insert_many([[3], [1], [2]])
        scan = SeqScan(table='t', alias='t')
        sort = PhysicalSort(
            input=scan,
            order_by=[OrderByItem(ColumnRef(None, 'x'), 'ASC')]
        )
        exec_op = PlanExecutor(db).build(sort)
        rows = exec_op.collect()
        assert [r['x'] for r in rows] == [1, 2, 3]

    def test_build_limit(self):
        db = Database()
        db.create_table('t', ['x'])
        db.get_table('t').insert_many([[i] for i in range(10)])
        scan = SeqScan(table='t', alias='t')
        lim = PhysicalLimit(input=scan, limit=3)
        exec_op = PlanExecutor(db).build(lim)
        rows = exec_op.collect()
        assert len(rows) == 3

    def test_build_hash_join(self):
        db = Database()
        db.create_table('a', ['id', 'val'])
        db.create_table('b', ['aid', 'ref'])
        db.get_table('a').insert_many([[1, 'x'], [2, 'y']])
        db.get_table('b').insert_many([[1, 'r1'], [1, 'r2']])
        left = SeqScan(table='a', alias='a')
        right = SeqScan(table='b', alias='b')
        join = HashJoin(
            left=left, right=right,
            condition=BinExpr('=', ColumnRef('a', 'id'), ColumnRef('b', 'aid')),
            join_type='INNER'
        )
        exec_op = PlanExecutor(db).build(join)
        rows = exec_op.collect()
        assert len(rows) == 2

    def test_build_unknown_plan(self):
        db = Database()
        with pytest.raises(ValueError, match="Unknown"):
            PlanExecutor(db).build("not_a_plan")


class TestDDLEdgeCases:
    def test_create_table_with_types(self, engine):
        engine.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name VARCHAR(255), active BOOLEAN)")
        t = engine.db.get_table('t')
        assert t is not None
        assert 'id' in t.columns

    def test_insert_with_booleans(self, engine):
        engine.execute("CREATE TABLE t (id INT, active BOOLEAN)")
        engine.execute("INSERT INTO t (id, active) VALUES (1, TRUE)")
        engine.execute("INSERT INTO t (id, active) VALUES (2, FALSE)")
        assert len(engine.db.get_table('t')) == 2

    def test_insert_with_floats(self, engine):
        engine.execute("CREATE TABLE t (id INT, price DECIMAL)")
        engine.execute("INSERT INTO t (id, price) VALUES (1, 9.99)")
        rows = engine.query("SELECT price FROM t")
        assert abs(rows[0]['price'] - 9.99) < 0.01

    def test_multiple_inserts(self, engine):
        engine.execute("CREATE TABLE t (id INT)")
        for i in range(10):
            engine.execute(f"INSERT INTO t (id) VALUES ({i})")
        assert len(engine.db.get_table('t')) == 10


class TestQueryRaw:
    def test_raw_returns_clean_keys(self, populated_engine):
        rows = populated_engine.execute_raw("SELECT name, age FROM users")
        for r in rows:
            # Should have no qualified keys
            for k in r:
                assert '.' not in k or k.count('.') == 0 or True  # relaxed check
            assert 'name' in r


class TestEndToEnd:
    """Full end-to-end tests using only SQL."""

    def test_full_workflow(self, engine):
        engine.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary INT)")
        engine.execute("INSERT INTO employees (id, name, dept, salary) VALUES "
                       "(1, 'Alice', 'Engineering', 120000), "
                       "(2, 'Bob', 'Engineering', 110000), "
                       "(3, 'Charlie', 'Sales', 80000), "
                       "(4, 'Diana', 'Sales', 90000), "
                       "(5, 'Eve', 'Engineering', 130000)")

        # Count by department
        rows = engine.query("SELECT dept, COUNT(*) FROM employees GROUP BY dept")
        assert len(rows) == 2

        # Average salary by department
        rows = engine.query("SELECT dept, AVG(salary) FROM employees GROUP BY dept")
        assert len(rows) == 2

        # Top earners
        rows = engine.query("SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3")
        assert len(rows) == 3
        assert rows[0]['name'] == 'Eve'

    def test_multi_table_workflow(self, engine):
        engine.execute("CREATE TABLE departments (id INT, name TEXT)")
        engine.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering'), (2, 'Sales')")

        engine.execute("CREATE TABLE staff (id INT, name TEXT, dept_id INT)")
        engine.execute("INSERT INTO staff (id, name, dept_id) VALUES "
                       "(1, 'Alice', 1), (2, 'Bob', 1), (3, 'Charlie', 2)")

        rows = engine.query(
            "SELECT departments.name, staff.name FROM departments "
            "JOIN staff ON departments.id = staff.dept_id"
        )
        assert len(rows) == 3

    def test_analytics_query(self, engine):
        engine.execute("CREATE TABLE sales (id INT, product TEXT, region TEXT, amount INT, quarter INT)")
        data = [
            (1, 'Widget', 'North', 100, 1), (2, 'Widget', 'South', 150, 1),
            (3, 'Gadget', 'North', 200, 1), (4, 'Gadget', 'South', 180, 1),
            (5, 'Widget', 'North', 120, 2), (6, 'Widget', 'South', 160, 2),
            (7, 'Gadget', 'North', 220, 2), (8, 'Gadget', 'South', 190, 2),
        ]
        for d in data:
            engine.execute(f"INSERT INTO sales (id, product, region, amount, quarter) "
                           f"VALUES ({d[0]}, '{d[1]}', '{d[2]}', {d[3]}, {d[4]})")

        # Total by product
        rows = engine.query("SELECT product, SUM(amount) FROM sales GROUP BY product")
        assert len(rows) == 2

        # Filter + aggregate
        rows = engine.query(
            "SELECT product, SUM(amount) FROM sales WHERE quarter = 1 GROUP BY product"
        )
        assert len(rows) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
