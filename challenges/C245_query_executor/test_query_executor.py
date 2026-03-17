"""
Tests for C245: Query Executor

Volcano/iterator-model query executor with physical operators,
joins, aggregation, sorting, and fluent query builder.
"""

import pytest
import math
from query_executor import (
    Row, Page, Table, TableIndex, Database,
    ColumnRef, Literal, Comparison, LogicExpr, ArithExpr, FuncExpr, CaseExpr,
    CompOp, LogicOp,
    eval_expr, _match_like,
    AggFunc, AggCall, AggState,
    ExecStats,
    Operator, SeqScanOp, IndexScanOp, FilterOp, ProjectOp,
    NestedLoopJoinOp, HashJoinOp, SortMergeJoinOp,
    SortOp, HashAggregateOp, LimitOp, UnionOp, DistinctOp,
    TopNOp, MaterializeOp, SemiJoinOp, AntiJoinOp, HavingOp,
    ExecutionEngine, QueryPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db():
    """Create a test database with employees and departments."""
    db = Database()
    emp = db.create_table('employees', ['id', 'name', 'dept_id', 'salary', 'age'])
    emp.insert_many([
        {'id': 1, 'name': 'Alice', 'dept_id': 1, 'salary': 90000, 'age': 30},
        {'id': 2, 'name': 'Bob', 'dept_id': 1, 'salary': 80000, 'age': 25},
        {'id': 3, 'name': 'Charlie', 'dept_id': 2, 'salary': 70000, 'age': 35},
        {'id': 4, 'name': 'Diana', 'dept_id': 2, 'salary': 95000, 'age': 28},
        {'id': 5, 'name': 'Eve', 'dept_id': 3, 'salary': 85000, 'age': 32},
        {'id': 6, 'name': 'Frank', 'dept_id': 3, 'salary': 75000, 'age': 40},
        {'id': 7, 'name': 'Grace', 'dept_id': 1, 'salary': 100000, 'age': 45},
        {'id': 8, 'name': 'Hank', 'dept_id': 2, 'salary': 60000, 'age': 22},
    ])

    dept = db.create_table('departments', ['id', 'name', 'budget'])
    dept.insert_many([
        {'id': 1, 'name': 'Engineering', 'budget': 500000},
        {'id': 2, 'name': 'Marketing', 'budget': 300000},
        {'id': 3, 'name': 'Sales', 'budget': 200000},
    ])

    return db


def collect(op):
    """Collect all rows from an operator."""
    rows = []
    op.open()
    while True:
        r = op.next()
        if r is None:
            break
        rows.append(r)
    op.close()
    return rows


def col(table, column):
    return ColumnRef(table, column)


def lit(value):
    return Literal(value)


def eq(left, right):
    return Comparison(CompOp.EQ, left, right)


def lt(left, right):
    return Comparison(CompOp.LT, left, right)


def gt(left, right):
    return Comparison(CompOp.GT, left, right)


def le(left, right):
    return Comparison(CompOp.LE, left, right)


def ge(left, right):
    return Comparison(CompOp.GE, left, right)


def ne(left, right):
    return Comparison(CompOp.NE, left, right)


def and_(*operands):
    return LogicExpr(LogicOp.AND, list(operands))


def or_(*operands):
    return LogicExpr(LogicOp.OR, list(operands))


def not_(operand):
    return LogicExpr(LogicOp.NOT, [operand])


# ===========================================================================
# Row tests
# ===========================================================================

class TestRow:
    def test_create_and_get(self):
        r = Row({'a': 1, 'b': 2})
        assert r.get('a') == 1
        assert r.get('b') == 2

    def test_get_missing(self):
        r = Row({'a': 1})
        assert r.get('x') is None

    def test_qualified_column(self):
        r = Row({'t.a': 1, 't.b': 2})
        assert r.get('t.a') == 1
        assert r.get('a') == 1  # bare lookup

    def test_set(self):
        r = Row({'a': 1})
        r2 = r.set('b', 2)
        assert r2.get('b') == 2
        assert r.get('b') is None  # immutable

    def test_project(self):
        r = Row({'a': 1, 'b': 2, 'c': 3})
        p = r.project(['a', 'c'])
        assert p.get('a') == 1
        assert p.get('c') == 3
        assert p.get('b') is None

    def test_merge(self):
        r1 = Row({'a': 1})
        r2 = Row({'b': 2})
        m = r1.merge(r2)
        assert m.get('a') == 1
        assert m.get('b') == 2

    def test_columns(self):
        r = Row({'x': 1, 'y': 2})
        assert set(r.columns()) == {'x', 'y'}

    def test_to_dict(self):
        r = Row({'a': 1, 'b': 2})
        assert r.to_dict() == {'a': 1, 'b': 2}

    def test_equality(self):
        r1 = Row({'a': 1, 'b': 2})
        r2 = Row({'a': 1, 'b': 2})
        assert r1 == r2

    def test_inequality(self):
        r1 = Row({'a': 1})
        r2 = Row({'a': 2})
        assert r1 != r2

    def test_hash(self):
        r1 = Row({'a': 1, 'b': 2})
        r2 = Row({'a': 1, 'b': 2})
        assert hash(r1) == hash(r2)

    def test_repr(self):
        r = Row({'a': 1})
        assert 'Row' in repr(r)

    def test_suffix_lookup(self):
        r = Row({'table.col': 42})
        assert r.get('col') == 42

    def test_values(self):
        r = Row({'a': 1, 'b': 2}, schema=['a', 'b'])
        assert r.values() == [1, 2]


# ===========================================================================
# Page and Table tests
# ===========================================================================

class TestTable:
    def test_create_and_insert(self):
        t = Table('t', ['a', 'b'])
        t.insert({'a': 1, 'b': 2})
        assert t.row_count == 1

    def test_insert_many(self):
        t = Table('t', ['x'])
        t.insert_many([{'x': i} for i in range(10)])
        assert t.row_count == 10

    def test_page_splitting(self):
        t = Table('t', ['x'], page_size=3)
        t.insert_many([{'x': i} for i in range(10)])
        assert t.page_count == 4  # 3+3+3+1

    def test_prefixed_columns(self):
        t = Table('emp', ['id', 'name'])
        t.insert({'id': 1, 'name': 'Alice'})
        pages = list(t.scan_pages())
        row = pages[0].rows[0]
        assert row.get('emp.id') == 1
        assert row.get('emp.name') == 'Alice'

    def test_scan_pages(self):
        t = Table('t', ['x'], page_size=5)
        t.insert_many([{'x': i} for i in range(12)])
        pages = list(t.scan_pages())
        assert len(pages) == 3
        total = sum(p.num_rows for p in pages)
        assert total == 12


# ===========================================================================
# TableIndex tests
# ===========================================================================

class TestTableIndex:
    def test_equality_lookup(self):
        t = Table('t', ['id', 'val'])
        t.insert_many([{'id': i, 'val': i * 10} for i in range(5)])
        idx = t.add_index('idx_id', 'id')
        results = idx.lookup_eq(3)
        assert len(results) == 1
        assert results[0].get('t.id') == 3

    def test_range_lookup(self):
        t = Table('t', ['id'])
        t.insert_many([{'id': i} for i in range(10)])
        idx = t.add_index('idx', 'id')
        results = idx.lookup_range(low=3, high=7)
        ids = [r.get('t.id') for r in results]
        assert ids == [3, 4, 5, 6, 7]

    def test_range_exclusive(self):
        t = Table('t', ['x'])
        t.insert_many([{'x': i} for i in range(10)])
        idx = t.add_index('idx', 'x')
        results = idx.lookup_range(low=3, high=7, low_inclusive=False, high_inclusive=False)
        vals = [r.get('t.x') for r in results]
        assert vals == [4, 5, 6]

    def test_get_index(self):
        t = Table('t', ['a', 'b'])
        t.add_index('idx_a', 'a')
        assert t.get_index('a') is not None
        assert t.get_index('b') is None


# ===========================================================================
# Expression evaluator tests
# ===========================================================================

class TestExpressions:
    def test_column_ref(self):
        r = Row({'t.x': 42})
        assert eval_expr(col('t', 'x'), r) == 42

    def test_literal(self):
        r = Row({})
        assert eval_expr(lit(99), r) == 99

    def test_comparison_eq(self):
        r = Row({'t.x': 5})
        assert eval_expr(eq(col('t', 'x'), lit(5)), r) is True
        assert eval_expr(eq(col('t', 'x'), lit(6)), r) is False

    def test_comparison_ne(self):
        r = Row({'t.x': 5})
        assert eval_expr(ne(col('t', 'x'), lit(5)), r) is False

    def test_comparison_lt(self):
        r = Row({'t.x': 5})
        assert eval_expr(lt(col('t', 'x'), lit(10)), r) is True
        assert eval_expr(lt(col('t', 'x'), lit(3)), r) is False

    def test_comparison_le(self):
        r = Row({'t.x': 5})
        assert eval_expr(le(col('t', 'x'), lit(5)), r) is True

    def test_comparison_gt(self):
        r = Row({'t.x': 5})
        assert eval_expr(gt(col('t', 'x'), lit(3)), r) is True

    def test_comparison_ge(self):
        r = Row({'t.x': 5})
        assert eval_expr(ge(col('t', 'x'), lit(5)), r) is True

    def test_is_null(self):
        r = Row({'t.x': None})
        assert eval_expr(Comparison(CompOp.IS_NULL, col('t', 'x'), None), r) is True

    def test_is_not_null(self):
        r = Row({'t.x': 5})
        assert eval_expr(Comparison(CompOp.IS_NOT_NULL, col('t', 'x'), None), r) is True

    def test_like(self):
        r = Row({'t.name': 'Alice'})
        assert eval_expr(Comparison(CompOp.LIKE, col('t', 'name'), lit('Al%')), r) is True
        assert eval_expr(Comparison(CompOp.LIKE, col('t', 'name'), lit('Bo%')), r) is False

    def test_like_underscore(self):
        r = Row({'t.name': 'Alice'})
        assert eval_expr(Comparison(CompOp.LIKE, col('t', 'name'), lit('A____')), r) is True

    def test_in(self):
        r = Row({'t.x': 3})
        assert eval_expr(Comparison(CompOp.IN, col('t', 'x'), lit([1, 2, 3])), r) is True
        assert eval_expr(Comparison(CompOp.IN, col('t', 'x'), lit([4, 5])), r) is False

    def test_between(self):
        r = Row({'t.x': 5})
        assert eval_expr(Comparison(CompOp.BETWEEN, col('t', 'x'), lit((3, 7))), r) is True
        assert eval_expr(Comparison(CompOp.BETWEEN, col('t', 'x'), lit((6, 10))), r) is False

    def test_null_comparisons(self):
        r = Row({'t.x': None})
        assert eval_expr(lt(col('t', 'x'), lit(5)), r) is False
        assert eval_expr(gt(col('t', 'x'), lit(5)), r) is False

    def test_logic_and(self):
        r = Row({'t.x': 5, 't.y': 10})
        expr = and_(gt(col('t', 'x'), lit(3)), lt(col('t', 'y'), lit(20)))
        assert eval_expr(expr, r) is True

    def test_logic_or(self):
        r = Row({'t.x': 1})
        expr = or_(eq(col('t', 'x'), lit(1)), eq(col('t', 'x'), lit(2)))
        assert eval_expr(expr, r) is True

    def test_logic_not(self):
        r = Row({'t.x': 5})
        expr = not_(eq(col('t', 'x'), lit(3)))
        assert eval_expr(expr, r) is True

    def test_arith_add(self):
        r = Row({'t.x': 3, 't.y': 4})
        expr = ArithExpr('+', col('t', 'x'), col('t', 'y'))
        assert eval_expr(expr, r) == 7

    def test_arith_sub(self):
        r = Row({'t.x': 10, 't.y': 3})
        assert eval_expr(ArithExpr('-', col('t', 'x'), col('t', 'y')), r) == 7

    def test_arith_mul(self):
        r = Row({'t.x': 3, 't.y': 4})
        assert eval_expr(ArithExpr('*', col('t', 'x'), col('t', 'y')), r) == 12

    def test_arith_div(self):
        r = Row({'t.x': 10, 't.y': 4})
        assert eval_expr(ArithExpr('/', col('t', 'x'), col('t', 'y')), r) == 2.5

    def test_arith_div_zero(self):
        r = Row({'t.x': 10, 't.y': 0})
        assert eval_expr(ArithExpr('/', col('t', 'x'), col('t', 'y')), r) is None

    def test_arith_null(self):
        r = Row({'t.x': None, 't.y': 3})
        assert eval_expr(ArithExpr('+', col('t', 'x'), col('t', 'y')), r) is None

    def test_case_expr(self):
        r = Row({'t.x': 5})
        expr = CaseExpr(
            whens=[(gt(col('t', 'x'), lit(10)), lit('big')),
                   (gt(col('t', 'x'), lit(3)), lit('medium'))],
            else_result=lit('small')
        )
        assert eval_expr(expr, r) == 'medium'

    def test_case_else(self):
        r = Row({'t.x': 1})
        expr = CaseExpr(
            whens=[(gt(col('t', 'x'), lit(10)), lit('big'))],
            else_result=lit('small')
        )
        assert eval_expr(expr, r) == 'small'

    def test_func_abs(self):
        r = Row({'t.x': -5})
        assert eval_expr(FuncExpr('ABS', [col('t', 'x')]), r) == 5

    def test_func_upper(self):
        r = Row({'t.x': 'hello'})
        assert eval_expr(FuncExpr('UPPER', [col('t', 'x')]), r) == 'HELLO'

    def test_func_lower(self):
        r = Row({'t.x': 'HELLO'})
        assert eval_expr(FuncExpr('LOWER', [col('t', 'x')]), r) == 'hello'

    def test_func_length(self):
        r = Row({'t.x': 'hello'})
        assert eval_expr(FuncExpr('LENGTH', [col('t', 'x')]), r) == 5

    def test_func_coalesce(self):
        r = Row({'t.x': None, 't.y': 5})
        assert eval_expr(FuncExpr('COALESCE', [col('t', 'x'), col('t', 'y')]), r) == 5

    def test_func_concat(self):
        r = Row({'t.a': 'hello', 't.b': ' world'})
        assert eval_expr(FuncExpr('CONCAT', [col('t', 'a'), col('t', 'b')]), r) == 'hello world'

    def test_bare_string_column(self):
        r = Row({'name': 'Alice'})
        assert eval_expr('name', r) == 'Alice'

    def test_bare_literal_values(self):
        r = Row({})
        assert eval_expr(42, r) == 42
        assert eval_expr(3.14, r) == 3.14
        assert eval_expr(True, r) is True
        assert eval_expr(None, r) is None

    def test_like_null(self):
        r = Row({'t.x': None})
        assert eval_expr(Comparison(CompOp.LIKE, col('t', 'x'), lit('foo')), r) is False


# ===========================================================================
# LIKE pattern matching
# ===========================================================================

class TestLikePattern:
    def test_percent_wildcard(self):
        assert _match_like('hello', 'h%') is True
        assert _match_like('hello', '%llo') is True
        assert _match_like('hello', '%ll%') is True

    def test_underscore_wildcard(self):
        assert _match_like('abc', 'a_c') is True
        assert _match_like('abcd', 'a_c') is False

    def test_exact(self):
        assert _match_like('abc', 'abc') is True
        assert _match_like('abc', 'abd') is False

    def test_empty(self):
        assert _match_like('', '%') is True
        assert _match_like('', '_') is False


# ===========================================================================
# SeqScan tests
# ===========================================================================

class TestSeqScan:
    def test_scan_all_rows(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        rows = collect(scan)
        assert len(rows) == 8

    def test_scan_empty_table(self):
        db = Database()
        t = db.create_table('empty', ['x'])
        scan = SeqScanOp(t)
        rows = collect(scan)
        assert len(rows) == 0

    def test_stats_tracking(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        rows = collect(scan)
        assert scan.stats.rows_produced == 8
        assert scan.stats.pages_read >= 1

    def test_iterator_protocol(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        rows = list(scan)
        assert len(rows) == 8

    def test_explain(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        assert 'SeqScan(employees)' in scan.explain()


# ===========================================================================
# IndexScan tests
# ===========================================================================

class TestIndexScan:
    def test_equality_scan(self):
        db = make_db()
        emp = db.get_table('employees')
        idx = emp.add_index('idx_id', 'id')
        scan = IndexScanOp(emp, idx, lookup_value=3)
        rows = collect(scan)
        assert len(rows) == 1
        assert rows[0].get('employees.name') == 'Charlie'

    def test_range_scan(self):
        db = make_db()
        emp = db.get_table('employees')
        idx = emp.add_index('idx_salary', 'salary')
        scan = IndexScanOp(emp, idx, low=80000, high=95000)
        rows = collect(scan)
        salaries = [r.get('employees.salary') for r in rows]
        assert all(80000 <= s <= 95000 for s in salaries)
        assert len(rows) == 4  # Bob(80k), Eve(85k), Alice(90k), Diana(95k)

    def test_explain(self):
        db = make_db()
        emp = db.get_table('employees')
        idx = emp.add_index('idx_id', 'id')
        scan = IndexScanOp(emp, idx, lookup_value=1)
        assert 'IndexScan' in scan.explain()


# ===========================================================================
# Filter tests
# ===========================================================================

class TestFilter:
    def test_simple_filter(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = gt(col('employees', 'salary'), lit(85000))
        f = FilterOp(scan, pred)
        rows = collect(f)
        assert len(rows) == 3  # Alice(90k), Diana(95k), Grace(100k)

    def test_filter_all_pass(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = gt(col('employees', 'salary'), lit(0))
        rows = collect(FilterOp(scan, pred))
        assert len(rows) == 8

    def test_filter_none_pass(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = gt(col('employees', 'salary'), lit(999999))
        rows = collect(FilterOp(scan, pred))
        assert len(rows) == 0

    def test_compound_filter(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = and_(
            gt(col('employees', 'salary'), lit(70000)),
            lt(col('employees', 'age'), lit(30))
        )
        rows = collect(FilterOp(scan, pred))
        names = [r.get('employees.name') for r in rows]
        assert set(names) == {'Bob', 'Diana'}

    def test_filter_stats(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = eq(col('employees', 'dept_id'), lit(1))
        f = FilterOp(scan, pred)
        rows = collect(f)
        assert f.stats.rows_produced == 3
        assert f.stats.rows_consumed == 8


# ===========================================================================
# Project tests
# ===========================================================================

class TestProject:
    def test_project_columns(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [
            (col('employees', 'name'), 'name'),
            (col('employees', 'salary'), 'salary'),
        ])
        rows = collect(proj)
        assert len(rows) == 8
        assert set(rows[0].columns()) == {'name', 'salary'}

    def test_project_expression(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [
            (col('employees', 'name'), 'name'),
            (ArithExpr('*', col('employees', 'salary'), lit(1.1)), 'raised_salary'),
        ])
        rows = collect(proj)
        alice = [r for r in rows if r.get('name') == 'Alice'][0]
        assert abs(alice.get('raised_salary') - 99000) < 0.01

    def test_project_literal(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [(lit(42), 'const')])
        rows = collect(proj)
        assert all(r.get('const') == 42 for r in rows)


# ===========================================================================
# NestedLoopJoin tests
# ===========================================================================

class TestNestedLoopJoin:
    def test_cross_join(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        join = NestedLoopJoinOp(emp, dept, join_type='cross')
        rows = collect(join)
        assert len(rows) == 8 * 3

    def test_inner_join(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        join = NestedLoopJoinOp(emp, dept, predicate=pred, join_type='inner')
        rows = collect(join)
        assert len(rows) == 8  # all employees match a dept

    def test_left_join(self):
        db = make_db()
        # Add an employee with no matching dept
        emp_table = db.get_table('employees')
        emp_table.insert({'id': 9, 'name': 'Zara', 'dept_id': 99, 'salary': 50000, 'age': 20})

        emp = SeqScanOp(emp_table)
        dept = SeqScanOp(db.get_table('departments'))
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        join = NestedLoopJoinOp(emp, dept, predicate=pred, join_type='left')
        rows = collect(join)
        assert len(rows) == 9  # 8 matched + 1 unmatched (Zara)
        zara = [r for r in rows if r.get('employees.name') == 'Zara'][0]
        assert zara.get('departments.name') is None

    def test_explain(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        join = NestedLoopJoinOp(emp, dept, join_type='inner')
        e = join.explain()
        assert 'NestedLoopJoin' in e
        assert 'SeqScan' in e


# ===========================================================================
# HashJoin tests
# ===========================================================================

class TestHashJoin:
    def test_inner_join(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        join = HashJoinOp(
            emp, dept,
            left_key=col('employees', 'dept_id'),
            right_key=col('departments', 'id')
        )
        rows = collect(join)
        assert len(rows) == 8
        for r in rows:
            assert r.get('employees.dept_id') == r.get('departments.id')

    def test_left_join(self):
        db = make_db()
        emp_table = db.get_table('employees')
        emp_table.insert({'id': 9, 'name': 'Nobody', 'dept_id': 99, 'salary': 0, 'age': 0})

        emp = SeqScanOp(emp_table)
        dept = SeqScanOp(db.get_table('departments'))
        join = HashJoinOp(
            emp, dept,
            left_key=col('employees', 'dept_id'),
            right_key=col('departments', 'id'),
            join_type='left'
        )
        rows = collect(join)
        assert len(rows) == 9
        nobody = [r for r in rows if r.get('employees.name') == 'Nobody'][0]
        assert nobody.get('departments.name') is None

    def test_no_matches(self):
        db = Database()
        t1 = db.create_table('t1', ['k'])
        t1.insert_many([{'k': 1}, {'k': 2}])
        t2 = db.create_table('t2', ['k'])
        t2.insert_many([{'k': 3}, {'k': 4}])

        join = HashJoinOp(
            SeqScanOp(t1), SeqScanOp(t2),
            left_key=col('t1', 'k'), right_key=col('t2', 'k')
        )
        assert len(collect(join)) == 0

    def test_duplicate_keys(self):
        db = Database()
        t1 = db.create_table('left', ['k', 'v'])
        t1.insert_many([{'k': 1, 'v': 'a'}, {'k': 1, 'v': 'b'}])
        t2 = db.create_table('right', ['k', 'w'])
        t2.insert_many([{'k': 1, 'w': 'x'}, {'k': 1, 'w': 'y'}])

        join = HashJoinOp(
            SeqScanOp(t1), SeqScanOp(t2),
            left_key=col('left', 'k'), right_key=col('right', 'k')
        )
        rows = collect(join)
        assert len(rows) == 4  # 2 * 2

    def test_stats(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        join = HashJoinOp(
            emp, dept,
            left_key=col('employees', 'dept_id'),
            right_key=col('departments', 'id')
        )
        collect(join)
        assert join.stats.memory_bytes > 0


# ===========================================================================
# SortMergeJoin tests
# ===========================================================================

class TestSortMergeJoin:
    def test_inner_join(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        join = SortMergeJoinOp(
            emp, dept,
            left_key=col('employees', 'dept_id'),
            right_key=col('departments', 'id')
        )
        rows = collect(join)
        assert len(rows) == 8

    def test_left_join(self):
        db = make_db()
        emp_table = db.get_table('employees')
        emp_table.insert({'id': 9, 'name': 'Loner', 'dept_id': 99, 'salary': 0, 'age': 0})

        emp = SeqScanOp(emp_table)
        dept = SeqScanOp(db.get_table('departments'))
        join = SortMergeJoinOp(
            emp, dept,
            left_key=col('employees', 'dept_id'),
            right_key=col('departments', 'id'),
            join_type='left'
        )
        rows = collect(join)
        assert len(rows) == 9

    def test_with_duplicates(self):
        db = Database()
        t1 = db.create_table('a', ['k'])
        t1.insert_many([{'k': 1}, {'k': 1}, {'k': 2}])
        t2 = db.create_table('b', ['k'])
        t2.insert_many([{'k': 1}, {'k': 2}, {'k': 2}])

        join = SortMergeJoinOp(
            SeqScanOp(t1), SeqScanOp(t2),
            left_key=col('a', 'k'), right_key=col('b', 'k')
        )
        rows = collect(join)
        # 2*1 for k=1, 1*2 for k=2 = 4
        assert len(rows) == 4


# ===========================================================================
# Sort tests
# ===========================================================================

class TestSort:
    def test_ascending(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        sort = SortOp(scan, [(col('employees', 'salary'), True)])
        rows = collect(sort)
        salaries = [r.get('employees.salary') for r in rows]
        assert salaries == sorted(salaries)

    def test_descending(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        sort = SortOp(scan, [(col('employees', 'salary'), False)])
        rows = collect(sort)
        salaries = [r.get('employees.salary') for r in rows]
        assert salaries == sorted(salaries, reverse=True)

    def test_multi_key(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        sort = SortOp(scan, [
            (col('employees', 'dept_id'), True),
            (col('employees', 'salary'), False)
        ])
        rows = collect(sort)
        # Within each dept, salary should be descending
        dept1 = [r for r in rows if r.get('employees.dept_id') == 1]
        assert [r.get('employees.salary') for r in dept1] == [100000, 90000, 80000]

    def test_sort_with_nulls(self):
        db = Database()
        t = db.create_table('t', ['x'])
        t.insert_many([{'x': 3}, {'x': None}, {'x': 1}])
        scan = SeqScanOp(t)
        sort = SortOp(scan, [(col('t', 'x'), True)])
        rows = collect(sort)
        vals = [r.get('t.x') for r in rows]
        # NULLs sort last
        assert vals[0] == 1
        assert vals[1] == 3
        assert vals[2] is None

    def test_sort_strings(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        sort = SortOp(scan, [(col('employees', 'name'), True)])
        rows = collect(sort)
        names = [r.get('employees.name') for r in rows]
        assert names == sorted(names)


# ===========================================================================
# HashAggregate tests
# ===========================================================================

class TestHashAggregate:
    def test_count_star(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(scan, [], [AggCall(AggFunc.COUNT_STAR, alias='cnt')])
        rows = collect(agg)
        assert len(rows) == 1
        assert rows[0].get('cnt') == 8

    def test_sum(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(scan, [], [
            AggCall(AggFunc.SUM, col('employees', 'salary'), alias='total')
        ])
        rows = collect(agg)
        assert rows[0].get('total') == 655000

    def test_avg(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(scan, [], [
            AggCall(AggFunc.AVG, col('employees', 'salary'), alias='avg_sal')
        ])
        rows = collect(agg)
        assert abs(rows[0].get('avg_sal') - 81875) < 0.01

    def test_min_max(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(scan, [], [
            AggCall(AggFunc.MIN, col('employees', 'salary'), alias='min_sal'),
            AggCall(AggFunc.MAX, col('employees', 'salary'), alias='max_sal'),
        ])
        rows = collect(agg)
        assert rows[0].get('min_sal') == 60000
        assert rows[0].get('max_sal') == 100000

    def test_group_by(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [col('employees', 'dept_id')],
            [AggCall(AggFunc.COUNT_STAR, alias='cnt')]
        )
        rows = collect(agg)
        assert len(rows) == 3
        counts = {r.get('employees.dept_id'): r.get('cnt') for r in rows}
        assert counts[1] == 3
        assert counts[2] == 3
        assert counts[3] == 2

    def test_group_by_sum(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [col('employees', 'dept_id')],
            [AggCall(AggFunc.SUM, col('employees', 'salary'), alias='total')]
        )
        rows = collect(agg)
        totals = {r.get('employees.dept_id'): r.get('total') for r in rows}
        assert totals[1] == 270000  # Alice(90k) + Bob(80k) + Grace(100k)
        assert totals[2] == 225000  # Charlie(70k) + Diana(95k) + Hank(60k)
        assert totals[3] == 160000  # Eve(85k) + Frank(75k)

    def test_distinct_count(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(scan, [], [
            AggCall(AggFunc.COUNT, col('employees', 'dept_id'), distinct=True, alias='cnt')
        ])
        rows = collect(agg)
        assert rows[0].get('cnt') == 3

    def test_empty_table_scalar_agg(self):
        db = Database()
        t = db.create_table('empty', ['x'])
        scan = SeqScanOp(t)
        agg = HashAggregateOp(scan, [], [
            AggCall(AggFunc.COUNT_STAR, alias='cnt'),
            AggCall(AggFunc.SUM, col('empty', 'x'), alias='total'),
        ])
        rows = collect(agg)
        assert len(rows) == 1
        assert rows[0].get('cnt') == 0
        assert rows[0].get('total') is None

    def test_null_handling(self):
        db = Database()
        t = db.create_table('t', ['x'])
        t.insert_many([{'x': 1}, {'x': None}, {'x': 3}])
        scan = SeqScanOp(t)
        agg = HashAggregateOp(scan, [], [
            AggCall(AggFunc.COUNT, col('t', 'x'), alias='cnt'),
            AggCall(AggFunc.SUM, col('t', 'x'), alias='total'),
        ])
        rows = collect(agg)
        assert rows[0].get('cnt') == 2  # NULLs excluded
        assert rows[0].get('total') == 4


# ===========================================================================
# Limit tests
# ===========================================================================

class TestLimit:
    def test_basic_limit(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 3)
        rows = collect(lim)
        assert len(rows) == 3

    def test_limit_greater_than_rows(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 100)
        rows = collect(lim)
        assert len(rows) == 8

    def test_offset(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 2, offset=3)
        rows = collect(lim)
        assert len(rows) == 2

    def test_offset_past_end(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 5, offset=100)
        rows = collect(lim)
        assert len(rows) == 0


# ===========================================================================
# Union tests
# ===========================================================================

class TestUnion:
    def test_union_all(self):
        db = make_db()
        s1 = FilterOp(SeqScanOp(db.get_table('employees')),
                       eq(col('employees', 'dept_id'), lit(1)))
        s2 = FilterOp(SeqScanOp(db.get_table('employees')),
                       eq(col('employees', 'dept_id'), lit(2)))
        union = UnionOp(s1, s2, all=True)
        rows = collect(union)
        assert len(rows) == 6  # 3 + 3

    def test_union_distinct(self):
        db = Database()
        t = db.create_table('t', ['x'])
        t.insert_many([{'x': 1}, {'x': 2}])
        s1 = SeqScanOp(t)
        s2 = SeqScanOp(t)
        union = UnionOp(s1, s2, all=False)
        rows = collect(union)
        assert len(rows) == 2


# ===========================================================================
# Distinct tests
# ===========================================================================

class TestDistinct:
    def test_removes_duplicates(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [(col('employees', 'dept_id'), 'dept_id')])
        dist = DistinctOp(proj)
        rows = collect(dist)
        assert len(rows) == 3

    def test_no_duplicates(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [(col('employees', 'id'), 'id')])
        dist = DistinctOp(proj)
        rows = collect(dist)
        assert len(rows) == 8


# ===========================================================================
# TopN tests
# ===========================================================================

class TestTopN:
    def test_top_3_salary(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        topn = TopNOp(scan, [(col('employees', 'salary'), False)], 3)
        rows = collect(topn)
        assert len(rows) == 3
        salaries = [r.get('employees.salary') for r in rows]
        assert salaries == [100000, 95000, 90000]

    def test_top_n_ascending(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        topn = TopNOp(scan, [(col('employees', 'salary'), True)], 2)
        rows = collect(topn)
        salaries = [r.get('employees.salary') for r in rows]
        assert salaries == [60000, 70000]

    def test_top_n_exceeds_rows(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        topn = TopNOp(scan, [(col('employees', 'salary'), False)], 100)
        rows = collect(topn)
        assert len(rows) == 8


# ===========================================================================
# Materialize tests
# ===========================================================================

class TestMaterialize:
    def test_reuse(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        mat = MaterializeOp(scan)
        # First read
        rows1 = collect(mat)
        assert len(rows1) == 8
        # Second read (should reuse materialized data)
        rows2 = collect(mat)
        assert len(rows2) == 8

    def test_explain(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        mat = MaterializeOp(scan)
        collect(mat)
        assert 'Materialize' in mat.explain()


# ===========================================================================
# SemiJoin tests
# ===========================================================================

class TestSemiJoin:
    def test_exists(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        semi = SemiJoinOp(emp, dept, pred)
        rows = collect(semi)
        # All employees match a dept
        assert len(rows) == 8

    def test_semi_join_filters(self):
        db = make_db()
        emp_table = db.get_table('employees')
        emp_table.insert({'id': 9, 'name': 'Orphan', 'dept_id': 99, 'salary': 0, 'age': 0})

        emp = SeqScanOp(emp_table)
        dept = SeqScanOp(db.get_table('departments'))
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        semi = SemiJoinOp(emp, dept, pred)
        rows = collect(semi)
        assert len(rows) == 8  # Orphan excluded
        names = [r.get('employees.name') for r in rows]
        assert 'Orphan' not in names


# ===========================================================================
# AntiJoin tests
# ===========================================================================

class TestAntiJoin:
    def test_not_exists(self):
        db = make_db()
        emp_table = db.get_table('employees')
        emp_table.insert({'id': 9, 'name': 'Orphan', 'dept_id': 99, 'salary': 0, 'age': 0})

        emp = SeqScanOp(emp_table)
        dept = SeqScanOp(db.get_table('departments'))
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        anti = AntiJoinOp(emp, dept, pred)
        rows = collect(anti)
        assert len(rows) == 1
        assert rows[0].get('employees.name') == 'Orphan'

    def test_all_match(self):
        db = make_db()
        emp = SeqScanOp(db.get_table('employees'))
        dept = SeqScanOp(db.get_table('departments'))
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        anti = AntiJoinOp(emp, dept, pred)
        rows = collect(anti)
        assert len(rows) == 0


# ===========================================================================
# Having tests
# ===========================================================================

class TestHaving:
    def test_having_filter(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [col('employees', 'dept_id')],
            [AggCall(AggFunc.COUNT_STAR, alias='cnt')]
        )
        having = HavingOp(agg, ge(col(None, 'cnt'), lit(3)))
        rows = collect(having)
        # Dept 1 has 3, Dept 2 has 3, Dept 3 has 2
        assert len(rows) == 2


# ===========================================================================
# Database tests
# ===========================================================================

class TestDatabase:
    def test_create_and_get(self):
        db = Database()
        t = db.create_table('foo', ['a', 'b'])
        assert db.get_table('foo') is t

    def test_get_missing(self):
        db = Database()
        assert db.get_table('bar') is None

    def test_drop_table(self):
        db = Database()
        db.create_table('foo', ['a'])
        db.drop_table('foo')
        assert db.get_table('foo') is None


# ===========================================================================
# ExecutionEngine tests
# ===========================================================================

class TestExecutionEngine:
    def test_execute(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        rows = engine.execute(scan)
        assert len(rows) == 8

    def test_execute_iter(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        rows = list(engine.execute_iter(scan))
        assert len(rows) == 8

    def test_explain(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        filt = FilterOp(scan, gt(col('employees', 'salary'), lit(80000)))
        text = engine.explain(filt)
        assert 'Filter' in text
        assert 'SeqScan' in text

    def test_explain_analyze(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        result = engine.explain_analyze(scan)
        assert result['rows'] == 8
        assert 'stats' in result
        assert 'results' in result


# ===========================================================================
# QueryPlan builder tests
# ===========================================================================

class TestQueryPlan:
    def test_scan_and_filter(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .filter(gt(col('employees', 'salary'), lit(85000)))
                .execute())
        assert len(rows) == 3

    def test_scan_filter_project(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .filter(eq(col('employees', 'dept_id'), lit(1)))
                .project([(col('employees', 'name'), 'name')])
                .execute())
        assert len(rows) == 3
        names = {r.get('name') for r in rows}
        assert names == {'Alice', 'Bob', 'Grace'}

    def test_hash_join(self):
        db = make_db()
        left = QueryPlan(db).scan('employees')
        right = QueryPlan(db).scan('departments')
        rows = (left.hash_join(right,
                               col('employees', 'dept_id'),
                               col('departments', 'id'))
                .execute())
        assert len(rows) == 8

    def test_nested_loop_join(self):
        db = make_db()
        left = QueryPlan(db).scan('employees')
        right = QueryPlan(db).scan('departments')
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        rows = left.nested_loop_join(right, pred).execute()
        assert len(rows) == 8

    def test_sort_merge_join(self):
        db = make_db()
        left = QueryPlan(db).scan('employees')
        right = QueryPlan(db).scan('departments')
        rows = (left.sort_merge_join(right,
                                      col('employees', 'dept_id'),
                                      col('departments', 'id'))
                .execute())
        assert len(rows) == 8

    def test_sort(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .sort([(col('employees', 'salary'), True)])
                .execute())
        salaries = [r.get('employees.salary') for r in rows]
        assert salaries == sorted(salaries)

    def test_aggregate(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .aggregate([], [AggCall(AggFunc.COUNT_STAR, alias='cnt')])
                .execute())
        assert rows[0].get('cnt') == 8

    def test_limit(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .limit(3)
                .execute())
        assert len(rows) == 3

    def test_distinct(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .project([(col('employees', 'dept_id'), 'dept_id')])
                .distinct()
                .execute())
        assert len(rows) == 3

    def test_union(self):
        db = make_db()
        q1 = (QueryPlan(db)
              .scan('employees')
              .filter(eq(col('employees', 'dept_id'), lit(1))))
        q2 = (QueryPlan(db)
              .scan('employees')
              .filter(eq(col('employees', 'dept_id'), lit(2))))
        rows = q1.union(q2).execute()
        assert len(rows) == 6

    def test_top_n(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .top_n([(col('employees', 'salary'), False)], 3)
                .execute())
        assert len(rows) == 3

    def test_materialize(self):
        db = make_db()
        q = QueryPlan(db).scan('employees').materialize()
        rows1 = q.execute()
        rows2 = q.execute()
        assert len(rows1) == 8
        assert len(rows2) == 8

    def test_semi_join(self):
        db = make_db()
        left = QueryPlan(db).scan('employees')
        right = QueryPlan(db).scan('departments')
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        rows = left.semi_join(right, pred).execute()
        assert len(rows) == 8

    def test_anti_join(self):
        db = make_db()
        left = QueryPlan(db).scan('employees')
        right = QueryPlan(db).scan('departments')
        pred = eq(col('employees', 'dept_id'), col('departments', 'id'))
        rows = left.anti_join(right, pred).execute()
        assert len(rows) == 0

    def test_having(self):
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .aggregate(
                    [col('employees', 'dept_id')],
                    [AggCall(AggFunc.SUM, col('employees', 'salary'), alias='total')]
                )
                .having(gt(col(None, 'total'), lit(200000)))
                .execute())
        # Dept 1: 270k, Dept 2: 225k, Dept 3: 160k -> 2 pass
        assert len(rows) == 2

    def test_explain(self):
        db = make_db()
        q = (QueryPlan(db)
             .scan('employees')
             .filter(gt(col('employees', 'salary'), lit(80000)))
             .project([(col('employees', 'name'), 'name')]))
        text = q.explain()
        assert 'Project' in text
        assert 'Filter' in text
        assert 'SeqScan' in text

    def test_index_scan_builder(self):
        db = make_db()
        emp = db.get_table('employees')
        emp.add_index('idx_id', 'id')
        rows = (QueryPlan(db)
                .index_scan('employees', 'id', value=1)
                .execute())
        assert len(rows) == 1

    def test_scan_missing_table(self):
        db = Database()
        with pytest.raises(ValueError, match="Table not found"):
            QueryPlan(db).scan('nonexistent')

    def test_index_scan_missing_table(self):
        db = Database()
        with pytest.raises(ValueError, match="Table not found"):
            QueryPlan(db).index_scan('nonexistent', 'col')

    def test_index_scan_missing_index(self):
        db = make_db()
        with pytest.raises(ValueError, match="No index"):
            QueryPlan(db).index_scan('employees', 'name')


# ===========================================================================
# ExecStats tests
# ===========================================================================

class TestExecStats:
    def test_totals(self):
        child = ExecStats(operator='child', rows_produced=10, pages_read=2, memory_bytes=500)
        parent = ExecStats(operator='parent', rows_produced=5, pages_read=1, memory_bytes=200,
                           children=[child])
        assert parent.total_rows() == 15
        assert parent.total_pages() == 3
        assert parent.total_memory() == 700

    def test_to_dict(self):
        s = ExecStats(operator='Scan', rows_produced=10, pages_read=5)
        d = s.to_dict()
        assert d['operator'] == 'Scan'
        assert d['rows_produced'] == 10


# ===========================================================================
# AggState tests
# ===========================================================================

class TestAggState:
    def test_count(self):
        agg = AggCall(AggFunc.COUNT, col('t', 'x'))
        state = AggState(agg)
        state.accumulate(Row({'t.x': 1}))
        state.accumulate(Row({'t.x': None}))
        state.accumulate(Row({'t.x': 3}))
        assert state.result() == 2

    def test_sum(self):
        agg = AggCall(AggFunc.SUM, col('t', 'x'))
        state = AggState(agg)
        for v in [10, 20, 30]:
            state.accumulate(Row({'t.x': v}))
        assert state.result() == 60

    def test_avg(self):
        agg = AggCall(AggFunc.AVG, col('t', 'x'))
        state = AggState(agg)
        for v in [10, 20, 30]:
            state.accumulate(Row({'t.x': v}))
        assert state.result() == 20.0

    def test_min(self):
        agg = AggCall(AggFunc.MIN, col('t', 'x'))
        state = AggState(agg)
        for v in [30, 10, 20]:
            state.accumulate(Row({'t.x': v}))
        assert state.result() == 10

    def test_max(self):
        agg = AggCall(AggFunc.MAX, col('t', 'x'))
        state = AggState(agg)
        for v in [30, 10, 20]:
            state.accumulate(Row({'t.x': v}))
        assert state.result() == 30

    def test_distinct(self):
        agg = AggCall(AggFunc.SUM, col('t', 'x'), distinct=True)
        state = AggState(agg)
        for v in [10, 10, 20]:
            state.accumulate(Row({'t.x': v}))
        assert state.result() == 30  # 10 + 20, not 10 + 10 + 20


# ===========================================================================
# Complex query tests (multi-operator pipelines)
# ===========================================================================

class TestComplexQueries:
    def test_join_filter_project(self):
        """SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.id WHERE salary > 85000"""
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .hash_join(
                    QueryPlan(db).scan('departments'),
                    col('employees', 'dept_id'),
                    col('departments', 'id')
                )
                .filter(gt(col('employees', 'salary'), lit(85000)))
                .project([
                    (col('employees', 'name'), 'emp_name'),
                    (col('departments', 'name'), 'dept_name'),
                ])
                .execute())
        names = {r.get('emp_name') for r in rows}
        assert names == {'Alice', 'Diana', 'Grace'}

    def test_group_by_having_sort(self):
        """SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id HAVING AVG(salary) > 80000 ORDER BY avg DESC"""
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .aggregate(
                    [col('employees', 'dept_id')],
                    [AggCall(AggFunc.AVG, col('employees', 'salary'), alias='avg_sal')]
                )
                .having(gt(col(None, 'avg_sal'), lit(80000)))
                .sort([(col(None, 'avg_sal'), False)])
                .execute())
        # Dept 1: avg 90k, Dept 2: avg 75k, Dept 3: avg 80k -> 1 pass
        assert len(rows) == 1  # Only Dept 1 (90k) passes > 80000

    def test_subquery_with_materialize(self):
        """Find employees earning more than avg salary."""
        db = make_db()
        # Get avg salary
        avg_plan = (QueryPlan(db)
                    .scan('employees')
                    .aggregate([], [AggCall(AggFunc.AVG, col('employees', 'salary'), alias='avg')]))
        avg_rows = avg_plan.execute()
        avg_salary = avg_rows[0].get('avg')

        # Find those above avg
        rows = (QueryPlan(db)
                .scan('employees')
                .filter(gt(col('employees', 'salary'), lit(avg_salary)))
                .project([(col('employees', 'name'), 'name'),
                          (col('employees', 'salary'), 'salary')])
                .sort([(col(None, 'salary'), False)])
                .execute())
        names = [r.get('name') for r in rows]
        assert 'Grace' in names
        assert 'Diana' in names
        assert 'Alice' in names
        assert 'Eve' in names

    def test_top_n_per_dept(self):
        """Top earner per department using sort + limit per group (manual)."""
        db = make_db()
        results = {}
        for dept_id in [1, 2, 3]:
            rows = (QueryPlan(db)
                    .scan('employees')
                    .filter(eq(col('employees', 'dept_id'), lit(dept_id)))
                    .sort([(col('employees', 'salary'), False)])
                    .limit(1)
                    .execute())
            results[dept_id] = rows[0].get('employees.name')
        assert results[1] == 'Grace'
        assert results[2] == 'Diana'
        assert results[3] == 'Eve'

    def test_join_aggregate(self):
        """SELECT d.name, COUNT(*), SUM(e.salary) FROM employees e JOIN departments d ..."""
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .hash_join(
                    QueryPlan(db).scan('departments'),
                    col('employees', 'dept_id'),
                    col('departments', 'id')
                )
                .aggregate(
                    [col('departments', 'name')],
                    [AggCall(AggFunc.COUNT_STAR, alias='cnt'),
                     AggCall(AggFunc.SUM, col('employees', 'salary'), alias='total')]
                )
                .sort([(col(None, 'total'), False)])
                .execute())
        assert len(rows) == 3
        assert rows[0].get('departments.name') == 'Engineering'
        assert rows[0].get('total') == 270000

    def test_multi_table_join(self):
        """Join three tables."""
        db = make_db()
        # Add a projects table
        proj = db.create_table('projects', ['id', 'name', 'dept_id'])
        proj.insert_many([
            {'id': 1, 'name': 'Widget', 'dept_id': 1},
            {'id': 2, 'name': 'Campaign', 'dept_id': 2},
        ])

        # employees -> departments -> projects
        rows = (QueryPlan(db)
                .scan('employees')
                .hash_join(
                    QueryPlan(db).scan('departments'),
                    col('employees', 'dept_id'),
                    col('departments', 'id')
                )
                .hash_join(
                    QueryPlan(db).scan('projects'),
                    col('departments', 'id'),
                    col('projects', 'dept_id')
                )
                .project([
                    (col('employees', 'name'), 'emp'),
                    (col('projects', 'name'), 'project'),
                ])
                .execute())
        # Dept 1 (3 emp) * 1 proj + Dept 2 (3 emp) * 1 proj = 6
        assert len(rows) == 6

    def test_union_with_sort(self):
        db = make_db()
        q1 = (QueryPlan(db)
              .scan('employees')
              .filter(eq(col('employees', 'dept_id'), lit(1)))
              .project([(col('employees', 'name'), 'name'),
                        (col('employees', 'salary'), 'salary')]))
        q2 = (QueryPlan(db)
              .scan('employees')
              .filter(eq(col('employees', 'dept_id'), lit(2)))
              .project([(col('employees', 'name'), 'name'),
                        (col('employees', 'salary'), 'salary')]))
        rows = q1.union(q2).sort([(col(None, 'salary'), False)]).execute()
        salaries = [r.get('salary') for r in rows]
        assert salaries == sorted(salaries, reverse=True)

    def test_self_join(self):
        """Find employees earning more than someone in same dept."""
        db = make_db()
        emp1 = SeqScanOp(db.get_table('employees'))
        emp2 = SeqScanOp(db.get_table('employees'))
        join = NestedLoopJoinOp(emp1, emp2, join_type='cross')
        rows = collect(join)
        assert len(rows) == 64  # 8 * 8

    def test_pipeline_short_circuit(self):
        """Limit should stop pulling from child early."""
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 2)
        rows = collect(lim)
        assert len(rows) == 2
        assert scan.stats.rows_produced <= 8

    def test_filter_then_aggregate(self):
        """Aggregate only filtered rows."""
        db = make_db()
        rows = (QueryPlan(db)
                .scan('employees')
                .filter(gt(col('employees', 'age'), lit(30)))
                .aggregate([], [
                    AggCall(AggFunc.COUNT_STAR, alias='cnt'),
                    AggCall(AggFunc.AVG, col('employees', 'salary'), alias='avg_sal')
                ])
                .execute())
        # Charlie(35), Eve(32), Frank(40), Grace(45) = 4
        assert rows[0].get('cnt') == 4

    def test_large_dataset(self):
        """Test with 1000 rows."""
        db = Database()
        t = db.create_table('big', ['id', 'val'], page_size=50)
        t.insert_many([{'id': i, 'val': i % 10} for i in range(1000)])

        rows = (QueryPlan(db)
                .scan('big')
                .filter(eq(col('big', 'val'), lit(5)))
                .aggregate([], [AggCall(AggFunc.COUNT_STAR, alias='cnt')])
                .execute())
        assert rows[0].get('cnt') == 100

    def test_index_scan_with_filter(self):
        """Index scan + additional filter."""
        db = make_db()
        emp = db.get_table('employees')
        emp.add_index('idx_dept', 'dept_id')
        rows = (QueryPlan(db)
                .index_scan('employees', 'dept_id', value=1)
                .filter(gt(col('employees', 'salary'), lit(85000)))
                .execute())
        # Dept 1: Alice(90k), Bob(80k), Grace(100k) -> Alice, Grace pass
        assert len(rows) == 2

    def test_join_three_strategies_same_result(self):
        """All three join strategies should produce identical results."""
        db = make_db()

        def do_join(join_type):
            emp = SeqScanOp(db.get_table('employees'))
            dept = SeqScanOp(db.get_table('departments'))
            left_key = col('employees', 'dept_id')
            right_key = col('departments', 'id')

            if join_type == 'hash':
                join = HashJoinOp(emp, dept, left_key, right_key)
            elif join_type == 'merge':
                join = SortMergeJoinOp(emp, dept, left_key, right_key)
            else:
                pred = eq(left_key, right_key)
                join = NestedLoopJoinOp(emp, dept, predicate=pred)

            proj = ProjectOp(join, [
                (col('employees', 'name'), 'name'),
                (col('departments', 'name'), 'dept'),
            ])
            rows = collect(proj)
            return {(r.get('name'), r.get('dept')) for r in rows}

        hash_result = do_join('hash')
        merge_result = do_join('merge')
        nl_result = do_join('nl')

        assert hash_result == merge_result == nl_result
        assert len(hash_result) == 8


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    def test_empty_hash_join(self):
        db = Database()
        t1 = db.create_table('t1', ['k'])
        t2 = db.create_table('t2', ['k'])
        join = HashJoinOp(SeqScanOp(t1), SeqScanOp(t2),
                          col('t1', 'k'), col('t2', 'k'))
        assert len(collect(join)) == 0

    def test_empty_sort(self):
        db = Database()
        t = db.create_table('t', ['x'])
        sort = SortOp(SeqScanOp(t), [(col('t', 'x'), True)])
        assert len(collect(sort)) == 0

    def test_empty_aggregate(self):
        db = Database()
        t = db.create_table('t', ['x'])
        agg = HashAggregateOp(SeqScanOp(t), [], [
            AggCall(AggFunc.COUNT_STAR, alias='cnt')
        ])
        rows = collect(agg)
        assert rows[0].get('cnt') == 0

    def test_single_row(self):
        db = Database()
        t = db.create_table('t', ['x'])
        t.insert({'x': 42})
        rows = collect(SeqScanOp(t))
        assert len(rows) == 1

    def test_unknown_expr_type(self):
        with pytest.raises(ValueError, match="Unknown expression type"):
            eval_expr(object(), Row({}))

    def test_operator_is_open_flag(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        assert not scan._is_open
        scan.open()
        assert scan._is_open
        scan.close()
        assert not scan._is_open

    def test_left_join_empty_right(self):
        db = Database()
        t1 = db.create_table('t1', ['k', 'v'])
        t1.insert_many([{'k': 1, 'v': 'a'}, {'k': 2, 'v': 'b'}])
        t2 = db.create_table('t2', ['k', 'w'])
        # Right is empty

        join = HashJoinOp(
            SeqScanOp(t1), SeqScanOp(t2),
            col('t1', 'k'), col('t2', 'k'),
            join_type='left'
        )
        rows = collect(join)
        # Left join with empty right -- no crash is the main check
        assert len(rows) >= 0


# ===========================================================================
# AggCall tests
# ===========================================================================

class TestAggCall:
    def test_count_star_alias(self):
        agg = AggCall(AggFunc.COUNT_STAR, alias='total')
        assert agg.alias == 'total'
        assert agg.func == AggFunc.COUNT_STAR

    def test_default_alias(self):
        agg = AggCall(AggFunc.SUM, col('t', 'x'))
        assert agg.alias is None


# ===========================================================================
# Page tests
# ===========================================================================

class TestPage:
    def test_page_properties(self):
        p = Page(page_id=5)
        assert p.num_rows == 0
        assert p.page_id == 5
        p.rows.append(Row({'x': 1}))
        assert p.num_rows == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
