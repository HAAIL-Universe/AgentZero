"""
Tests for C245: Query Executor

Covers:
- Row operations (get, set, project, merge)
- Table and page storage
- Table indexes (equality, range lookup)
- Expression evaluator (column refs, literals, comparisons, logic, arithmetic, case, functions)
- Aggregate functions (COUNT, SUM, AVG, MIN, MAX, COUNT(*), DISTINCT)
- Physical operators: SeqScan, IndexScan, Filter, Project, HashJoin,
  NestedLoopJoin, SortMergeJoin, Sort, HashAggregate, Limit, Union,
  Distinct, TopN, Materialize, SemiJoin, AntiJoin, Having
- Execution engine (execute, execute_iter, explain, explain_analyze)
- QueryPlan fluent builder
- Edge cases (empty tables, NULLs, large datasets)
- Integration (multi-operator pipelines)
"""

import unittest
from query_executor import (
    Row, Page, Table, TableIndex,
    CompOp, LogicOp, ColumnRef, Literal, Comparison, LogicExpr,
    ArithExpr, FuncExpr, CaseExpr, AggFunc, AggCall, AggState,
    eval_expr, ExecStats,
    Operator, SeqScanOp, IndexScanOp, FilterOp, ProjectOp,
    NestedLoopJoinOp, HashJoinOp, SortMergeJoinOp,
    SortOp, HashAggregateOp, LimitOp, UnionOp, DistinctOp,
    TopNOp, MaterializeOp, SemiJoinOp, AntiJoinOp, HavingOp,
    Database, ExecutionEngine, QueryPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_db():
    """Create a test database with employees and departments."""
    db = Database()

    emp = db.create_table('employees', ['id', 'name', 'dept_id', 'salary'])
    emp.insert_many([
        {'id': 1, 'name': 'Alice', 'dept_id': 1, 'salary': 90000},
        {'id': 2, 'name': 'Bob', 'dept_id': 1, 'salary': 80000},
        {'id': 3, 'name': 'Carol', 'dept_id': 2, 'salary': 95000},
        {'id': 4, 'name': 'Dave', 'dept_id': 2, 'salary': 70000},
        {'id': 5, 'name': 'Eve', 'dept_id': 3, 'salary': 110000},
        {'id': 6, 'name': 'Frank', 'dept_id': 3, 'salary': 65000},
        {'id': 7, 'name': 'Grace', 'dept_id': 1, 'salary': 85000},
        {'id': 8, 'name': 'Heidi', 'dept_id': 2, 'salary': 92000},
    ])

    dept = db.create_table('departments', ['id', 'name', 'budget'])
    dept.insert_many([
        {'id': 1, 'name': 'Engineering', 'budget': 500000},
        {'id': 2, 'name': 'Marketing', 'budget': 300000},
        {'id': 3, 'name': 'Sales', 'budget': 200000},
    ])

    return db


def collect(op):
    """Execute an operator and collect all rows."""
    rows = []
    op.open()
    while True:
        r = op.next()
        if r is None:
            break
        rows.append(r)
    op.close()
    return rows


# ===========================================================================
# 1. Row
# ===========================================================================

class TestRow(unittest.TestCase):
    def test_get_basic(self):
        r = Row({'a': 1, 'b': 2})
        self.assertEqual(r.get('a'), 1)
        self.assertEqual(r.get('b'), 2)

    def test_get_qualified(self):
        r = Row({'t.a': 1, 't.b': 2})
        self.assertEqual(r.get('t.a'), 1)

    def test_get_bare_from_qualified(self):
        r = Row({'t.a': 1})
        self.assertEqual(r.get('a'), 1)

    def test_get_suffix_match(self):
        r = Row({'employees.name': 'Alice'})
        self.assertEqual(r.get('name'), 'Alice')

    def test_get_missing(self):
        r = Row({'a': 1})
        self.assertIsNone(r.get('z'))

    def test_set(self):
        r = Row({'a': 1})
        r2 = r.set('b', 2)
        self.assertEqual(r2.get('b'), 2)
        self.assertIsNone(r.get('b'))  # original unchanged

    def test_project(self):
        r = Row({'a': 1, 'b': 2, 'c': 3})
        p = r.project(['a', 'c'])
        self.assertEqual(p.get('a'), 1)
        self.assertEqual(p.get('c'), 3)
        self.assertIsNone(p.get('b'))

    def test_merge(self):
        r1 = Row({'a': 1})
        r2 = Row({'b': 2})
        m = r1.merge(r2)
        self.assertEqual(m.get('a'), 1)
        self.assertEqual(m.get('b'), 2)

    def test_columns(self):
        r = Row({'x': 1, 'y': 2})
        self.assertEqual(set(r.columns()), {'x', 'y'})

    def test_to_dict(self):
        r = Row({'a': 1, 'b': 2})
        self.assertEqual(r.to_dict(), {'a': 1, 'b': 2})

    def test_equality(self):
        r1 = Row({'a': 1, 'b': 2})
        r2 = Row({'a': 1, 'b': 2})
        self.assertEqual(r1, r2)

    def test_hash(self):
        r1 = Row({'a': 1, 'b': 2})
        r2 = Row({'a': 1, 'b': 2})
        self.assertEqual(hash(r1), hash(r2))

    def test_repr(self):
        r = Row({'a': 1})
        self.assertIn('a', repr(r))


# ===========================================================================
# 2. Table and Page
# ===========================================================================

class TestTable(unittest.TestCase):
    def test_insert_and_count(self):
        t = Table('t', ['id', 'val'])
        t.insert({'id': 1, 'val': 'a'})
        t.insert({'id': 2, 'val': 'b'})
        self.assertEqual(t.row_count, 2)

    def test_pages(self):
        t = Table('t', ['id'], page_size=3)
        for i in range(10):
            t.insert({'id': i})
        self.assertEqual(t.page_count, 4)  # ceil(10/3)

    def test_insert_many(self):
        t = Table('t', ['id'])
        t.insert_many([{'id': i} for i in range(5)])
        self.assertEqual(t.row_count, 5)

    def test_scan_pages(self):
        t = Table('t', ['id'], page_size=5)
        t.insert_many([{'id': i} for i in range(12)])
        pages = list(t.scan_pages())
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0].num_rows, 5)

    def test_prefixed_columns(self):
        t = Table('emp', ['id', 'name'])
        t.insert({'id': 1, 'name': 'Alice'})
        page = list(t.scan_pages())[0]
        row = page.rows[0]
        self.assertEqual(row.get('emp.id'), 1)

    def test_add_index(self):
        t = Table('t', ['id', 'val'])
        t.insert_many([{'id': i, 'val': i * 10} for i in range(5)])
        idx = t.add_index('idx_id', 'id')
        self.assertEqual(len(idx.lookup_eq(3)), 1)

    def test_get_index(self):
        t = Table('t', ['id'])
        t.add_index('idx_id', 'id')
        self.assertIsNotNone(t.get_index('id'))
        self.assertIsNone(t.get_index('nonexistent'))

    def test_index_updated_on_insert(self):
        t = Table('t', ['id'])
        t.add_index('idx_id', 'id')
        t.insert({'id': 42})
        idx = t.get_index('id')
        results = idx.lookup_eq(42)
        self.assertEqual(len(results), 1)


# ===========================================================================
# 3. TableIndex
# ===========================================================================

class TestTableIndex(unittest.TestCase):
    def setUp(self):
        self.idx = TableIndex('idx', 'emp', 'id')
        for i in range(10):
            self.idx.insert(Row({f'emp.id': i, 'emp.name': f'n{i}'}))

    def test_lookup_eq(self):
        results = self.idx.lookup_eq(5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get('emp.id'), 5)

    def test_lookup_eq_miss(self):
        results = self.idx.lookup_eq(99)
        self.assertEqual(len(results), 0)

    def test_lookup_range(self):
        results = self.idx.lookup_range(low=3, high=7)
        ids = [r.get('emp.id') for r in results]
        self.assertEqual(sorted(ids), [3, 4, 5, 6, 7])

    def test_lookup_range_exclusive(self):
        results = self.idx.lookup_range(low=3, high=7,
                                        low_inclusive=False, high_inclusive=False)
        ids = [r.get('emp.id') for r in results]
        self.assertEqual(sorted(ids), [4, 5, 6])

    def test_lookup_range_unbounded(self):
        results = self.idx.lookup_range(high=2)
        self.assertEqual(len(results), 3)  # 0, 1, 2

    def test_lookup_range_null_handling(self):
        idx = TableIndex('idx', 't', 'x')
        idx.insert(Row({'t.x': None}))
        idx.insert(Row({'t.x': 1}))
        results = idx.lookup_range(low=0)
        self.assertEqual(len(results), 1)


# ===========================================================================
# 4. Expression Evaluator
# ===========================================================================

class TestExprEval(unittest.TestCase):
    def setUp(self):
        self.row = Row({'t.x': 10, 't.y': 20, 't.name': 'Alice', 't.z': None})

    def test_column_ref(self):
        self.assertEqual(eval_expr(ColumnRef('t', 'x'), self.row), 10)

    def test_literal(self):
        self.assertEqual(eval_expr(Literal(42), self.row), 42)

    def test_bare_string_column(self):
        self.assertEqual(eval_expr('t.x', self.row), 10)

    def test_int_literal(self):
        self.assertEqual(eval_expr(99, self.row), 99)

    def test_none_literal(self):
        self.assertIsNone(eval_expr(None, self.row))

    def test_comparison_eq(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.EQ, ColumnRef('t', 'x'), Literal(10)), self.row))

    def test_comparison_ne(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.NE, ColumnRef('t', 'x'), Literal(5)), self.row))

    def test_comparison_lt(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.LT, ColumnRef('t', 'x'), Literal(20)), self.row))

    def test_comparison_le(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.LE, ColumnRef('t', 'x'), Literal(10)), self.row))

    def test_comparison_gt(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.GT, ColumnRef('t', 'y'), Literal(10)), self.row))

    def test_comparison_ge(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.GE, ColumnRef('t', 'y'), Literal(20)), self.row))

    def test_comparison_is_null(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.IS_NULL, ColumnRef('t', 'z'), None), self.row))

    def test_comparison_is_not_null(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.IS_NOT_NULL, ColumnRef('t', 'x'), None), self.row))

    def test_comparison_like(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.LIKE, ColumnRef('t', 'name'), Literal('A%')), self.row))
        self.assertFalse(eval_expr(
            Comparison(CompOp.LIKE, ColumnRef('t', 'name'), Literal('B%')), self.row))

    def test_comparison_like_underscore(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.LIKE, ColumnRef('t', 'name'), Literal('A____')), self.row))

    def test_comparison_in(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.IN, ColumnRef('t', 'x'), Literal([5, 10, 15])), self.row))

    def test_comparison_between(self):
        self.assertTrue(eval_expr(
            Comparison(CompOp.BETWEEN, ColumnRef('t', 'x'), Literal((5, 15))), self.row))

    def test_comparison_null_lt(self):
        self.assertFalse(eval_expr(
            Comparison(CompOp.LT, ColumnRef('t', 'z'), Literal(10)), self.row))

    def test_logic_and(self):
        expr = LogicExpr(LogicOp.AND, [
            Comparison(CompOp.GT, ColumnRef('t', 'x'), Literal(5)),
            Comparison(CompOp.LT, ColumnRef('t', 'y'), Literal(30)),
        ])
        self.assertTrue(eval_expr(expr, self.row))

    def test_logic_or(self):
        expr = LogicExpr(LogicOp.OR, [
            Comparison(CompOp.EQ, ColumnRef('t', 'x'), Literal(99)),
            Comparison(CompOp.EQ, ColumnRef('t', 'y'), Literal(20)),
        ])
        self.assertTrue(eval_expr(expr, self.row))

    def test_logic_not(self):
        expr = LogicExpr(LogicOp.NOT, [
            Comparison(CompOp.EQ, ColumnRef('t', 'x'), Literal(99)),
        ])
        self.assertTrue(eval_expr(expr, self.row))

    def test_arith_add(self):
        self.assertEqual(eval_expr(ArithExpr('+', ColumnRef('t', 'x'), Literal(5)), self.row), 15)

    def test_arith_sub(self):
        self.assertEqual(eval_expr(ArithExpr('-', ColumnRef('t', 'y'), ColumnRef('t', 'x')), self.row), 10)

    def test_arith_mul(self):
        self.assertEqual(eval_expr(ArithExpr('*', ColumnRef('t', 'x'), Literal(2)), self.row), 20)

    def test_arith_div(self):
        self.assertEqual(eval_expr(ArithExpr('/', ColumnRef('t', 'y'), ColumnRef('t', 'x')), self.row), 2.0)

    def test_arith_div_zero(self):
        self.assertIsNone(eval_expr(ArithExpr('/', ColumnRef('t', 'x'), Literal(0)), self.row))

    def test_arith_null(self):
        self.assertIsNone(eval_expr(ArithExpr('+', ColumnRef('t', 'z'), Literal(1)), self.row))

    def test_case_expr(self):
        case = CaseExpr(
            whens=[
                (Comparison(CompOp.GT, ColumnRef('t', 'x'), Literal(50)), Literal('high')),
                (Comparison(CompOp.GT, ColumnRef('t', 'x'), Literal(5)), Literal('medium')),
            ],
            else_result=Literal('low')
        )
        self.assertEqual(eval_expr(case, self.row), 'medium')

    def test_case_else(self):
        case = CaseExpr(
            whens=[(Comparison(CompOp.EQ, ColumnRef('t', 'x'), Literal(99)), Literal('match'))],
            else_result=Literal('no_match')
        )
        self.assertEqual(eval_expr(case, self.row), 'no_match')

    def test_case_no_else(self):
        case = CaseExpr(
            whens=[(Comparison(CompOp.EQ, ColumnRef('t', 'x'), Literal(99)), Literal('match'))]
        )
        self.assertIsNone(eval_expr(case, self.row))

    def test_func_abs(self):
        self.assertEqual(eval_expr(FuncExpr('ABS', [Literal(-5)]), self.row), 5)

    def test_func_upper(self):
        self.assertEqual(eval_expr(FuncExpr('UPPER', [ColumnRef('t', 'name')]), self.row), 'ALICE')

    def test_func_lower(self):
        self.assertEqual(eval_expr(FuncExpr('LOWER', [ColumnRef('t', 'name')]), self.row), 'alice')

    def test_func_length(self):
        self.assertEqual(eval_expr(FuncExpr('LENGTH', [ColumnRef('t', 'name')]), self.row), 5)

    def test_func_coalesce(self):
        self.assertEqual(eval_expr(FuncExpr('COALESCE', [ColumnRef('t', 'z'), Literal(42)]), self.row), 42)

    def test_func_concat(self):
        self.assertEqual(eval_expr(FuncExpr('CONCAT', [Literal('a'), Literal('b')]), self.row), 'ab')

    def test_func_null(self):
        self.assertIsNone(eval_expr(FuncExpr('ABS', [Literal(None)]), self.row))

    def test_unknown_expr_raises(self):
        with self.assertRaises(ValueError):
            eval_expr(object(), self.row)


# ===========================================================================
# 5. Aggregate State
# ===========================================================================

class TestAggState(unittest.TestCase):
    def _rows(self):
        return [
            Row({'val': 10}), Row({'val': 20}), Row({'val': 30}),
            Row({'val': None}), Row({'val': 20}),
        ]

    def test_count(self):
        state = AggState(AggCall(AggFunc.COUNT, ColumnRef(None, 'val')))
        for r in self._rows():
            state.accumulate(r)
        self.assertEqual(state.result(), 4)  # None skipped

    def test_count_star(self):
        state = AggState(AggCall(AggFunc.COUNT_STAR))
        for r in self._rows():
            state.accumulate(r)
        self.assertEqual(state.result(), 5)

    def test_sum(self):
        state = AggState(AggCall(AggFunc.SUM, ColumnRef(None, 'val')))
        for r in self._rows():
            state.accumulate(r)
        self.assertEqual(state.result(), 80)

    def test_avg(self):
        state = AggState(AggCall(AggFunc.AVG, ColumnRef(None, 'val')))
        for r in self._rows():
            state.accumulate(r)
        self.assertEqual(state.result(), 20.0)

    def test_min(self):
        state = AggState(AggCall(AggFunc.MIN, ColumnRef(None, 'val')))
        for r in self._rows():
            state.accumulate(r)
        self.assertEqual(state.result(), 10)

    def test_max(self):
        state = AggState(AggCall(AggFunc.MAX, ColumnRef(None, 'val')))
        for r in self._rows():
            state.accumulate(r)
        self.assertEqual(state.result(), 30)

    def test_count_distinct(self):
        state = AggState(AggCall(AggFunc.COUNT, ColumnRef(None, 'val'), distinct=True))
        for r in self._rows():
            state.accumulate(r)
        self.assertEqual(state.result(), 3)  # 10, 20, 30

    def test_sum_empty(self):
        state = AggState(AggCall(AggFunc.SUM, ColumnRef(None, 'val')))
        self.assertIsNone(state.result())

    def test_avg_empty(self):
        state = AggState(AggCall(AggFunc.AVG, ColumnRef(None, 'val')))
        self.assertIsNone(state.result())


# ===========================================================================
# 6. SeqScan
# ===========================================================================

class TestSeqScan(unittest.TestCase):
    def test_scan_all(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        rows = collect(scan)
        self.assertEqual(len(rows), 8)

    def test_scan_empty_table(self):
        db = Database()
        db.create_table('empty', ['id'])
        rows = collect(SeqScanOp(db.get_table('empty')))
        self.assertEqual(len(rows), 0)

    def test_scan_stats(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        collect(scan)
        self.assertEqual(scan.stats.rows_produced, 8)
        self.assertGreater(scan.stats.pages_read, 0)

    def test_explain(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        self.assertIn('employees', scan.explain())


# ===========================================================================
# 7. IndexScan
# ===========================================================================

class TestIndexScan(unittest.TestCase):
    def setUp(self):
        self.db = make_db()
        self.db.get_table('employees').add_index('idx_dept', 'dept_id')

    def test_eq_lookup(self):
        t = self.db.get_table('employees')
        idx = t.get_index('dept_id')
        scan = IndexScanOp(t, idx, lookup_value=1)
        rows = collect(scan)
        self.assertEqual(len(rows), 3)  # Alice, Bob, Grace

    def test_range_lookup(self):
        t = self.db.get_table('employees')
        idx = t.get_index('dept_id')
        scan = IndexScanOp(t, idx, low=2, high=3)
        rows = collect(scan)
        self.assertEqual(len(rows), 5)  # dept 2 (3) + dept 3 (2)

    def test_explain(self):
        t = self.db.get_table('employees')
        idx = t.get_index('dept_id')
        scan = IndexScanOp(t, idx, lookup_value=1)
        self.assertIn('idx_dept', scan.explain())


# ===========================================================================
# 8. Filter
# ===========================================================================

class TestFilter(unittest.TestCase):
    def test_filter_eq(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = Comparison(CompOp.EQ, ColumnRef('employees', 'dept_id'), Literal(1))
        filt = FilterOp(scan, pred)
        rows = collect(filt)
        self.assertEqual(len(rows), 3)

    def test_filter_gt(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = Comparison(CompOp.GT, ColumnRef('employees', 'salary'), Literal(90000))
        filt = FilterOp(scan, pred)
        rows = collect(filt)
        self.assertEqual(len(rows), 3)  # Carol 95k, Eve 110k, Heidi 92k

    def test_filter_and(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = LogicExpr(LogicOp.AND, [
            Comparison(CompOp.EQ, ColumnRef('employees', 'dept_id'), Literal(1)),
            Comparison(CompOp.GT, ColumnRef('employees', 'salary'), Literal(82000)),
        ])
        filt = FilterOp(scan, pred)
        rows = collect(filt)
        self.assertEqual(len(rows), 2)  # Alice 90k, Grace 85k

    def test_filter_stats(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = Comparison(CompOp.EQ, ColumnRef('employees', 'dept_id'), Literal(1))
        filt = FilterOp(scan, pred)
        collect(filt)
        self.assertEqual(filt.stats.rows_produced, 3)
        self.assertEqual(filt.stats.rows_consumed, 8)

    def test_explain(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        pred = Comparison(CompOp.EQ, ColumnRef('employees', 'dept_id'), Literal(1))
        filt = FilterOp(scan, pred)
        self.assertIn('Filter', filt.explain())


# ===========================================================================
# 9. Project
# ===========================================================================

class TestProject(unittest.TestCase):
    def test_project_columns(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [
            (ColumnRef('employees', 'name'), 'name'),
            (ColumnRef('employees', 'salary'), 'salary'),
        ])
        rows = collect(proj)
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows[0].columns()), {'name', 'salary'})

    def test_project_computed(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [
            (ColumnRef('employees', 'name'), 'name'),
            (ArithExpr('*', ColumnRef('employees', 'salary'), Literal(1.1)), 'new_salary'),
        ])
        rows = collect(proj)
        alice = rows[0]
        self.assertAlmostEqual(alice.get('new_salary'), 99000.0)

    def test_project_explain(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [(ColumnRef('employees', 'name'), 'name')])
        self.assertIn('Project', proj.explain())
        self.assertIn('name', proj.explain())


# ===========================================================================
# 10. Hash Join
# ===========================================================================

class TestHashJoin(unittest.TestCase):
    def test_inner_join(self):
        db = make_db()
        left = SeqScanOp(db.get_table('employees'))
        right = SeqScanOp(db.get_table('departments'))
        join = HashJoinOp(
            left, right,
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id'),
        )
        rows = collect(join)
        self.assertEqual(len(rows), 8)  # All employees match a dept

    def test_join_values(self):
        db = make_db()
        left = SeqScanOp(db.get_table('employees'))
        right = SeqScanOp(db.get_table('departments'))
        join = HashJoinOp(
            left, right,
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id'),
        )
        rows = collect(join)
        # Alice is in dept 1 = Engineering
        alice = [r for r in rows if r.get('employees.name') == 'Alice'][0]
        self.assertEqual(alice.get('departments.name'), 'Engineering')

    def test_left_join(self):
        db = make_db()
        # Add an employee with no dept
        db.get_table('employees').insert({'id': 9, 'name': 'Zoe', 'dept_id': 99, 'salary': 50000})
        left = SeqScanOp(db.get_table('employees'))
        right = SeqScanOp(db.get_table('departments'))
        join = HashJoinOp(
            left, right,
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id'),
            join_type='left'
        )
        rows = collect(join)
        self.assertEqual(len(rows), 9)  # 8 matched + 1 unmatched
        zoe = [r for r in rows if r.get('employees.name') == 'Zoe'][0]
        self.assertIsNone(zoe.get('departments.name'))

    def test_join_no_matches(self):
        db = Database()
        t1 = db.create_table('a', ['id'])
        t1.insert({'id': 1})
        t2 = db.create_table('b', ['id'])
        t2.insert({'id': 2})
        join = HashJoinOp(
            SeqScanOp(t1), SeqScanOp(t2),
            ColumnRef('a', 'id'), ColumnRef('b', 'id')
        )
        rows = collect(join)
        self.assertEqual(len(rows), 0)

    def test_explain(self):
        db = make_db()
        join = HashJoinOp(
            SeqScanOp(db.get_table('employees')),
            SeqScanOp(db.get_table('departments')),
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id'),
        )
        self.assertIn('HashJoin', join.explain())


# ===========================================================================
# 11. Nested Loop Join
# ===========================================================================

class TestNestedLoopJoin(unittest.TestCase):
    def test_cross_join(self):
        db = make_db()
        left = SeqScanOp(db.get_table('departments'))
        right = SeqScanOp(db.get_table('departments'))
        join = NestedLoopJoinOp(left, right)
        rows = collect(join)
        self.assertEqual(len(rows), 9)  # 3 * 3

    def test_inner_with_predicate(self):
        db = make_db()
        left = SeqScanOp(db.get_table('employees'))
        right = SeqScanOp(db.get_table('departments'))
        pred = Comparison(CompOp.EQ,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        join = NestedLoopJoinOp(left, right, predicate=pred, join_type='inner')
        rows = collect(join)
        self.assertEqual(len(rows), 8)

    def test_left_join_unmatched(self):
        db = Database()
        t1 = db.create_table('a', ['id', 'val'])
        t1.insert_many([{'id': 1, 'val': 'x'}, {'id': 2, 'val': 'y'}])
        t2 = db.create_table('b', ['id', 'val'])
        t2.insert({'id': 1, 'val': 'z'})
        pred = Comparison(CompOp.EQ, ColumnRef('a', 'id'), ColumnRef('b', 'id'))
        join = NestedLoopJoinOp(
            SeqScanOp(t1), SeqScanOp(t2), predicate=pred, join_type='left')
        rows = collect(join)
        self.assertEqual(len(rows), 2)
        row2 = [r for r in rows if r.get('a.id') == 2][0]
        self.assertIsNone(row2.get('b.val'))


# ===========================================================================
# 12. Sort-Merge Join
# ===========================================================================

class TestSortMergeJoin(unittest.TestCase):
    def test_equi_join(self):
        db = make_db()
        left = SeqScanOp(db.get_table('employees'))
        right = SeqScanOp(db.get_table('departments'))
        join = SortMergeJoinOp(
            left, right,
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id'),
        )
        rows = collect(join)
        self.assertEqual(len(rows), 8)

    def test_left_join(self):
        db = make_db()
        db.get_table('employees').insert({'id': 9, 'name': 'Zoe', 'dept_id': 99, 'salary': 50000})
        left = SeqScanOp(db.get_table('employees'))
        right = SeqScanOp(db.get_table('departments'))
        join = SortMergeJoinOp(
            left, right,
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id'),
            join_type='left'
        )
        rows = collect(join)
        self.assertEqual(len(rows), 9)

    def test_explain(self):
        db = make_db()
        join = SortMergeJoinOp(
            SeqScanOp(db.get_table('employees')),
            SeqScanOp(db.get_table('departments')),
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id'),
        )
        self.assertIn('SortMergeJoin', join.explain())


# ===========================================================================
# 13. Sort
# ===========================================================================

class TestSort(unittest.TestCase):
    def test_sort_ascending(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        sort = SortOp(scan, [(ColumnRef('employees', 'salary'), True)])
        rows = collect(sort)
        salaries = [r.get('employees.salary') for r in rows]
        self.assertEqual(salaries, sorted(salaries))

    def test_sort_descending(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        sort = SortOp(scan, [(ColumnRef('employees', 'salary'), False)])
        rows = collect(sort)
        salaries = [r.get('employees.salary') for r in rows]
        self.assertEqual(salaries, sorted(salaries, reverse=True))

    def test_sort_string(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        sort = SortOp(scan, [(ColumnRef('employees', 'name'), True)])
        rows = collect(sort)
        names = [r.get('employees.name') for r in rows]
        self.assertEqual(names, sorted(names))

    def test_explain(self):
        db = make_db()
        sort = SortOp(SeqScanOp(db.get_table('employees')),
                       [(ColumnRef('employees', 'salary'), False)])
        self.assertIn('Sort', sort.explain())
        self.assertIn('DESC', sort.explain())


# ===========================================================================
# 14. Hash Aggregate
# ===========================================================================

class TestHashAggregate(unittest.TestCase):
    def test_count_by_group(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [ColumnRef('employees', 'dept_id')],
            [AggCall(AggFunc.COUNT_STAR, alias='cnt')]
        )
        rows = collect(agg)
        self.assertEqual(len(rows), 3)
        counts = {r.get('employees.dept_id'): r.get('cnt') for r in rows}
        self.assertEqual(counts[1], 3)
        self.assertEqual(counts[2], 3)
        self.assertEqual(counts[3], 2)

    def test_avg_salary(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [ColumnRef('employees', 'dept_id')],
            [AggCall(AggFunc.AVG, ColumnRef('employees', 'salary'), alias='avg_sal')]
        )
        rows = collect(agg)
        dept1 = [r for r in rows if r.get('employees.dept_id') == 1][0]
        self.assertAlmostEqual(dept1.get('avg_sal'), (90000 + 80000 + 85000) / 3)

    def test_scalar_aggregate(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan, [],
            [AggCall(AggFunc.COUNT_STAR, alias='total'),
             AggCall(AggFunc.MAX, ColumnRef('employees', 'salary'), alias='max_sal')]
        )
        rows = collect(agg)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get('total'), 8)
        self.assertEqual(rows[0].get('max_sal'), 110000)

    def test_empty_table_scalar(self):
        db = Database()
        db.create_table('empty', ['val'])
        scan = SeqScanOp(db.get_table('empty'))
        agg = HashAggregateOp(scan, [],
                              [AggCall(AggFunc.COUNT_STAR, alias='cnt')])
        rows = collect(agg)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get('cnt'), 0)

    def test_multiple_aggs(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [ColumnRef('employees', 'dept_id')],
            [
                AggCall(AggFunc.COUNT_STAR, alias='cnt'),
                AggCall(AggFunc.SUM, ColumnRef('employees', 'salary'), alias='total_sal'),
                AggCall(AggFunc.MIN, ColumnRef('employees', 'salary'), alias='min_sal'),
                AggCall(AggFunc.MAX, ColumnRef('employees', 'salary'), alias='max_sal'),
            ]
        )
        rows = collect(agg)
        self.assertEqual(len(rows), 3)


# ===========================================================================
# 15. Limit
# ===========================================================================

class TestLimit(unittest.TestCase):
    def test_basic_limit(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 3)
        rows = collect(lim)
        self.assertEqual(len(rows), 3)

    def test_limit_with_offset(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 2, offset=3)
        rows = collect(lim)
        self.assertEqual(len(rows), 2)

    def test_limit_exceeds_rows(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 100)
        rows = collect(lim)
        self.assertEqual(len(rows), 8)

    def test_limit_zero(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        lim = LimitOp(scan, 0)
        rows = collect(lim)
        self.assertEqual(len(rows), 0)

    def test_explain(self):
        db = make_db()
        lim = LimitOp(SeqScanOp(db.get_table('employees')), 5, offset=2)
        self.assertIn('Limit', lim.explain())
        self.assertIn('offset', lim.explain())


# ===========================================================================
# 16. Union
# ===========================================================================

class TestUnion(unittest.TestCase):
    def test_union_all(self):
        db = make_db()
        left = SeqScanOp(db.get_table('departments'))
        right = SeqScanOp(db.get_table('departments'))
        union = UnionOp(left, right, all=True)
        rows = collect(union)
        self.assertEqual(len(rows), 6)

    def test_union_distinct(self):
        db = make_db()
        left = SeqScanOp(db.get_table('departments'))
        right = SeqScanOp(db.get_table('departments'))
        union = UnionOp(left, right, all=False)
        rows = collect(union)
        self.assertEqual(len(rows), 3)

    def test_explain(self):
        db = make_db()
        union = UnionOp(SeqScanOp(db.get_table('departments')),
                        SeqScanOp(db.get_table('departments')), all=True)
        self.assertIn('UnionAll', union.explain())


# ===========================================================================
# 17. Distinct
# ===========================================================================

class TestDistinct(unittest.TestCase):
    def test_distinct_removes_dupes(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        proj = ProjectOp(scan, [(ColumnRef('employees', 'dept_id'), 'dept_id')])
        dist = DistinctOp(proj)
        rows = collect(dist)
        self.assertEqual(len(rows), 3)


# ===========================================================================
# 18. TopN
# ===========================================================================

class TestTopN(unittest.TestCase):
    def test_top_3_salary(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        topn = TopNOp(scan, [(ColumnRef('employees', 'salary'), False)], 3)
        rows = collect(topn)
        self.assertEqual(len(rows), 3)
        salaries = [r.get('employees.salary') for r in rows]
        self.assertEqual(salaries, sorted(salaries, reverse=True))
        self.assertEqual(salaries[0], 110000)

    def test_top_n_exceeds(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        topn = TopNOp(scan, [(ColumnRef('employees', 'salary'), True)], 100)
        rows = collect(topn)
        self.assertEqual(len(rows), 8)


# ===========================================================================
# 19. Materialize
# ===========================================================================

class TestMaterialize(unittest.TestCase):
    def test_materialize_reuse(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('departments'))
        mat = MaterializeOp(scan)

        # First pass
        rows1 = collect(mat)
        self.assertEqual(len(rows1), 3)

        # Second pass (reuses materialized data)
        rows2 = collect(mat)
        self.assertEqual(len(rows2), 3)

    def test_explain(self):
        db = make_db()
        mat = MaterializeOp(SeqScanOp(db.get_table('departments')))
        collect(mat)
        self.assertIn('Materialize', mat.explain())


# ===========================================================================
# 20. Semi Join
# ===========================================================================

class TestSemiJoin(unittest.TestCase):
    def test_semi_join(self):
        db = make_db()
        # Employees in departments with budget > 250000
        left = SeqScanOp(db.get_table('employees'))
        right = FilterOp(
            SeqScanOp(db.get_table('departments')),
            Comparison(CompOp.GT, ColumnRef('departments', 'budget'), Literal(250000))
        )
        pred = Comparison(CompOp.EQ,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        semi = SemiJoinOp(left, right, pred)
        rows = collect(semi)
        # Depts with budget > 250k: Engineering (500k), Marketing (300k) -> depts 1, 2
        dept_ids = set(r.get('employees.dept_id') for r in rows)
        self.assertEqual(dept_ids, {1, 2})
        self.assertEqual(len(rows), 6)  # 3 from dept 1 + 3 from dept 2


# ===========================================================================
# 21. Anti Join
# ===========================================================================

class TestAntiJoin(unittest.TestCase):
    def test_anti_join(self):
        db = make_db()
        # Employees NOT in departments with budget > 250000
        left = SeqScanOp(db.get_table('employees'))
        right = FilterOp(
            SeqScanOp(db.get_table('departments')),
            Comparison(CompOp.GT, ColumnRef('departments', 'budget'), Literal(250000))
        )
        pred = Comparison(CompOp.EQ,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        anti = AntiJoinOp(left, right, pred)
        rows = collect(anti)
        # Only dept 3 (Sales, 200k) employees
        dept_ids = set(r.get('employees.dept_id') for r in rows)
        self.assertEqual(dept_ids, {3})
        self.assertEqual(len(rows), 2)


# ===========================================================================
# 22. Having
# ===========================================================================

class TestHaving(unittest.TestCase):
    def test_having_filter(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [ColumnRef('employees', 'dept_id')],
            [AggCall(AggFunc.COUNT_STAR, alias='cnt')]
        )
        having = HavingOp(agg, Comparison(CompOp.GE, ColumnRef(None, 'cnt'), Literal(3)))
        rows = collect(having)
        self.assertEqual(len(rows), 2)  # dept 1 (3) and dept 2 (3)


# ===========================================================================
# 23. Iterator Protocol
# ===========================================================================

class TestIterator(unittest.TestCase):
    def test_for_loop(self):
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        count = 0
        for row in scan:
            count += 1
        self.assertEqual(count, 8)


# ===========================================================================
# 24. ExecStats
# ===========================================================================

class TestExecStats(unittest.TestCase):
    def test_totals(self):
        parent = ExecStats(operator='Join', rows_produced=10, pages_read=5, memory_bytes=1000)
        child1 = ExecStats(operator='Scan', rows_produced=100, pages_read=20)
        child2 = ExecStats(operator='Scan', rows_produced=50, pages_read=10)
        parent.children = [child1, child2]
        self.assertEqual(parent.total_rows(), 160)
        self.assertEqual(parent.total_pages(), 35)

    def test_to_dict(self):
        s = ExecStats(operator='Scan', rows_produced=5)
        d = s.to_dict()
        self.assertEqual(d['operator'], 'Scan')
        self.assertEqual(d['rows_produced'], 5)


# ===========================================================================
# 25. Database and ExecutionEngine
# ===========================================================================

class TestDatabase(unittest.TestCase):
    def test_create_and_get(self):
        db = Database()
        t = db.create_table('t', ['id'])
        self.assertIs(db.get_table('t'), t)

    def test_drop_table(self):
        db = Database()
        db.create_table('t', ['id'])
        db.drop_table('t')
        self.assertIsNone(db.get_table('t'))

    def test_get_missing(self):
        db = Database()
        self.assertIsNone(db.get_table('missing'))


class TestExecutionEngine(unittest.TestCase):
    def test_execute(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        rows = engine.execute(scan)
        self.assertEqual(len(rows), 8)

    def test_execute_iter(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        count = sum(1 for _ in engine.execute_iter(scan))
        self.assertEqual(count, 8)

    def test_explain(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        self.assertIn('SeqScan', engine.explain(scan))

    def test_explain_analyze(self):
        db = make_db()
        engine = ExecutionEngine(db)
        scan = SeqScanOp(db.get_table('employees'))
        result = engine.explain_analyze(scan)
        self.assertEqual(result['rows'], 8)
        self.assertIn('stats', result)


# ===========================================================================
# 26. QueryPlan Builder
# ===========================================================================

class TestQueryPlan(unittest.TestCase):
    def setUp(self):
        self.db = make_db()

    def test_scan(self):
        rows = QueryPlan(self.db).scan('employees').execute()
        self.assertEqual(len(rows), 8)

    def test_scan_missing(self):
        with self.assertRaises(ValueError):
            QueryPlan(self.db).scan('missing')

    def test_filter(self):
        rows = (QueryPlan(self.db)
                .scan('employees')
                .filter(Comparison(CompOp.GT, ColumnRef('employees', 'salary'), Literal(90000)))
                .execute())
        self.assertEqual(len(rows), 3)

    def test_project(self):
        rows = (QueryPlan(self.db)
                .scan('employees')
                .project([(ColumnRef('employees', 'name'), 'name')])
                .execute())
        self.assertEqual(set(rows[0].columns()), {'name'})

    def test_hash_join(self):
        left = QueryPlan(self.db).scan('employees')
        right = QueryPlan(self.db).scan('departments')
        rows = left.hash_join(
            right,
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id')
        ).execute()
        self.assertEqual(len(rows), 8)

    def test_nested_loop_join(self):
        left = QueryPlan(self.db).scan('employees')
        right = QueryPlan(self.db).scan('departments')
        pred = Comparison(CompOp.EQ,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        rows = left.nested_loop_join(right, pred).execute()
        self.assertEqual(len(rows), 8)

    def test_sort_merge_join(self):
        left = QueryPlan(self.db).scan('employees')
        right = QueryPlan(self.db).scan('departments')
        rows = left.sort_merge_join(
            right,
            ColumnRef('employees', 'dept_id'),
            ColumnRef('departments', 'id')
        ).execute()
        self.assertEqual(len(rows), 8)

    def test_sort(self):
        rows = (QueryPlan(self.db)
                .scan('employees')
                .sort([(ColumnRef('employees', 'salary'), False)])
                .execute())
        salaries = [r.get('employees.salary') for r in rows]
        self.assertEqual(salaries, sorted(salaries, reverse=True))

    def test_aggregate(self):
        rows = (QueryPlan(self.db)
                .scan('employees')
                .aggregate(
                    [ColumnRef('employees', 'dept_id')],
                    [AggCall(AggFunc.COUNT_STAR, alias='cnt')]
                )
                .execute())
        self.assertEqual(len(rows), 3)

    def test_having(self):
        rows = (QueryPlan(self.db)
                .scan('employees')
                .aggregate(
                    [ColumnRef('employees', 'dept_id')],
                    [AggCall(AggFunc.COUNT_STAR, alias='cnt')]
                )
                .having(Comparison(CompOp.GE, ColumnRef(None, 'cnt'), Literal(3)))
                .execute())
        self.assertEqual(len(rows), 2)

    def test_limit(self):
        rows = QueryPlan(self.db).scan('employees').limit(3).execute()
        self.assertEqual(len(rows), 3)

    def test_distinct(self):
        rows = (QueryPlan(self.db)
                .scan('employees')
                .project([(ColumnRef('employees', 'dept_id'), 'dept_id')])
                .distinct()
                .execute())
        self.assertEqual(len(rows), 3)

    def test_union(self):
        left = QueryPlan(self.db).scan('departments')
        right = QueryPlan(self.db).scan('departments')
        rows = left.union(right, all=True).execute()
        self.assertEqual(len(rows), 6)

    def test_top_n(self):
        rows = (QueryPlan(self.db)
                .scan('employees')
                .top_n([(ColumnRef('employees', 'salary'), False)], 3)
                .execute())
        self.assertEqual(len(rows), 3)

    def test_materialize(self):
        plan = QueryPlan(self.db).scan('departments').materialize()
        rows1 = plan.execute()
        rows2 = plan.execute()
        self.assertEqual(len(rows1), 3)
        self.assertEqual(len(rows2), 3)

    def test_semi_join(self):
        left = QueryPlan(self.db).scan('employees')
        right = QueryPlan(self.db).scan('departments').filter(
            Comparison(CompOp.GT, ColumnRef('departments', 'budget'), Literal(400000)))
        pred = Comparison(CompOp.EQ,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        rows = left.semi_join(right, pred).execute()
        self.assertEqual(len(rows), 3)  # Only Engineering (500k)

    def test_anti_join(self):
        left = QueryPlan(self.db).scan('employees')
        right = QueryPlan(self.db).scan('departments').filter(
            Comparison(CompOp.GT, ColumnRef('departments', 'budget'), Literal(250000)))
        pred = Comparison(CompOp.EQ,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        rows = left.anti_join(right, pred).execute()
        self.assertEqual(len(rows), 2)  # Sales dept only

    def test_index_scan(self):
        self.db.get_table('employees').add_index('idx_dept', 'dept_id')
        rows = QueryPlan(self.db).index_scan('employees', 'dept_id', value=2).execute()
        self.assertEqual(len(rows), 3)

    def test_index_scan_no_index(self):
        with self.assertRaises(ValueError):
            QueryPlan(self.db).index_scan('employees', 'nonexistent')

    def test_explain(self):
        plan = QueryPlan(self.db).scan('employees').filter(
            Comparison(CompOp.GT, ColumnRef('employees', 'salary'), Literal(90000)))
        self.assertIn('Filter', plan.explain())

    def test_complex_pipeline(self):
        """SELECT d.name, COUNT(*), AVG(salary) FROM employees e
           JOIN departments d ON e.dept_id = d.id
           WHERE salary > 70000
           GROUP BY d.name
           HAVING COUNT(*) >= 2
           ORDER BY d.name"""
        left = QueryPlan(self.db).scan('employees')
        right = QueryPlan(self.db).scan('departments')
        rows = (left
                .hash_join(right,
                           ColumnRef('employees', 'dept_id'),
                           ColumnRef('departments', 'id'))
                .filter(Comparison(CompOp.GT, ColumnRef('employees', 'salary'), Literal(70000)))
                .aggregate(
                    [ColumnRef('departments', 'name')],
                    [AggCall(AggFunc.COUNT_STAR, alias='cnt'),
                     AggCall(AggFunc.AVG, ColumnRef('employees', 'salary'), alias='avg_sal')]
                )
                .having(Comparison(CompOp.GE, ColumnRef(None, 'cnt'), Literal(2)))
                .sort([(ColumnRef('departments', 'name'), True)])
                .execute())
        # Engineering: Alice 90k, Bob 80k, Grace 85k -> 3, avg 85k
        # Marketing: Carol 95k, Heidi 92k (Dave 70k excluded) -> 2, avg 93.5k
        # Sales: Eve 110k (Frank 65k excluded) -> 1, filtered by HAVING
        self.assertEqual(len(rows), 2)
        names = [r.get('departments.name') for r in rows]
        self.assertEqual(names, ['Engineering', 'Marketing'])


# ===========================================================================
# 27. Integration: Complex Queries
# ===========================================================================

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.db = make_db()

    def test_top_earner_per_dept(self):
        """Find highest salary per department."""
        scan = SeqScanOp(self.db.get_table('employees'))
        agg = HashAggregateOp(
            scan,
            [ColumnRef('employees', 'dept_id')],
            [AggCall(AggFunc.MAX, ColumnRef('employees', 'salary'), alias='max_sal')]
        )
        sort = SortOp(agg, [(ColumnRef(None, 'max_sal'), False)])
        rows = collect(sort)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].get('max_sal'), 110000)

    def test_filtered_join_with_limit(self):
        left = SeqScanOp(self.db.get_table('employees'))
        right = SeqScanOp(self.db.get_table('departments'))
        join = HashJoinOp(left, right,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        filt = FilterOp(join, Comparison(CompOp.GT,
                                          ColumnRef('employees', 'salary'), Literal(80000)))
        sort = SortOp(filt, [(ColumnRef('employees', 'salary'), False)])
        lim = LimitOp(sort, 3)
        rows = collect(lim)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].get('employees.salary'), 110000)

    def test_self_join(self):
        """Find employees in same dept with higher salary."""
        t = self.db.get_table('employees')
        left = SeqScanOp(t)
        right = SeqScanOp(t)
        # Cross join with predicate
        pred = LogicExpr(LogicOp.AND, [
            Comparison(CompOp.EQ, ColumnRef('employees', 'dept_id'),
                       ColumnRef('employees', 'dept_id')),
        ])
        # Note: self-join with same-named columns merges -- would need aliases
        # Just verify it runs
        join = NestedLoopJoinOp(left, right)
        rows = collect(join)
        self.assertEqual(len(rows), 64)  # 8 * 8

    def test_correlated_subquery_pattern(self):
        """Semi-join simulating correlated subquery."""
        left = SeqScanOp(self.db.get_table('employees'))
        right = FilterOp(
            SeqScanOp(self.db.get_table('departments')),
            Comparison(CompOp.EQ, ColumnRef('departments', 'name'), Literal('Engineering'))
        )
        pred = Comparison(CompOp.EQ,
                          ColumnRef('employees', 'dept_id'),
                          ColumnRef('departments', 'id'))
        semi = SemiJoinOp(left, right, pred)
        rows = collect(semi)
        self.assertEqual(len(rows), 3)

    def test_not_exists_pattern(self):
        """Anti-join simulating NOT EXISTS."""
        left = SeqScanOp(self.db.get_table('departments'))
        right = FilterOp(
            SeqScanOp(self.db.get_table('employees')),
            Comparison(CompOp.GT, ColumnRef('employees', 'salary'), Literal(200000))
        )
        pred = Comparison(CompOp.EQ,
                          ColumnRef('departments', 'id'),
                          ColumnRef('employees', 'dept_id'))
        anti = AntiJoinOp(left, right, pred)
        rows = collect(anti)
        self.assertEqual(len(rows), 3)  # No one earns > 200k

    def test_union_with_filter(self):
        """UNION of two filtered scans."""
        plan1 = FilterOp(
            SeqScanOp(self.db.get_table('employees')),
            Comparison(CompOp.EQ, ColumnRef('employees', 'dept_id'), Literal(1)))
        plan2 = FilterOp(
            SeqScanOp(self.db.get_table('employees')),
            Comparison(CompOp.EQ, ColumnRef('employees', 'dept_id'), Literal(2)))
        union = UnionOp(plan1, plan2, all=True)
        rows = collect(union)
        self.assertEqual(len(rows), 6)

    def test_large_dataset(self):
        """Performance: scan and filter 10k rows."""
        db = Database()
        t = db.create_table('big', ['id', 'val'], page_size=1000)
        t.insert_many([{'id': i, 'val': i % 100} for i in range(10000)])
        scan = SeqScanOp(t)
        filt = FilterOp(scan, Comparison(CompOp.EQ, ColumnRef('big', 'val'), Literal(42)))
        rows = collect(filt)
        self.assertEqual(len(rows), 100)

    def test_three_way_hash_join(self):
        """3-way join: employees -> departments -> employees (self via dept)."""
        e1 = SeqScanOp(self.db.get_table('employees'))
        d = SeqScanOp(self.db.get_table('departments'))
        join1 = HashJoinOp(e1, d,
                           ColumnRef('employees', 'dept_id'),
                           ColumnRef('departments', 'id'))
        # Join result has employees.* and departments.*
        # Now join with filtered employees for same dept
        e2 = FilterOp(
            SeqScanOp(self.db.get_table('employees')),
            Comparison(CompOp.GT, ColumnRef('employees', 'salary'), Literal(90000))
        )
        # NL join since columns would collide -- just testing execution
        join2 = NestedLoopJoinOp(join1, e2)
        lim = LimitOp(join2, 5)
        rows = collect(lim)
        self.assertEqual(len(rows), 5)


# ===========================================================================
# 28. Edge Cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    def test_empty_join(self):
        db = Database()
        t1 = db.create_table('a', ['id'])
        t2 = db.create_table('b', ['id'])
        join = HashJoinOp(SeqScanOp(t1), SeqScanOp(t2),
                          ColumnRef('a', 'id'), ColumnRef('b', 'id'))
        rows = collect(join)
        self.assertEqual(len(rows), 0)

    def test_all_nulls_aggregate(self):
        db = Database()
        t = db.create_table('t', ['val'])
        t.insert_many([{'val': None}, {'val': None}])
        scan = SeqScanOp(t)
        agg = HashAggregateOp(
            scan, [],
            [AggCall(AggFunc.SUM, ColumnRef('t', 'val'), alias='s'),
             AggCall(AggFunc.COUNT, ColumnRef('t', 'val'), alias='c'),
             AggCall(AggFunc.COUNT_STAR, alias='cs')]
        )
        rows = collect(agg)
        self.assertIsNone(rows[0].get('s'))
        self.assertEqual(rows[0].get('c'), 0)
        self.assertEqual(rows[0].get('cs'), 2)

    def test_sort_with_nulls(self):
        db = Database()
        t = db.create_table('t', ['val'])
        t.insert_many([{'val': 3}, {'val': None}, {'val': 1}, {'val': None}, {'val': 2}])
        scan = SeqScanOp(t)
        sort = SortOp(scan, [(ColumnRef('t', 'val'), True)])
        rows = collect(sort)
        vals = [r.get('t.val') for r in rows]
        # Non-null values should be sorted, nulls at end
        non_nulls = [v for v in vals if v is not None]
        self.assertEqual(non_nulls, [1, 2, 3])

    def test_deeply_nested_operators(self):
        db = make_db()
        op = SeqScanOp(db.get_table('employees'))
        for i in range(10):
            op = FilterOp(op, Comparison(CompOp.IS_NOT_NULL,
                                          ColumnRef('employees', 'name'), None))
        rows = collect(op)
        self.assertEqual(len(rows), 8)

    def test_operator_reuse(self):
        """Operators can be opened/closed multiple times."""
        db = make_db()
        scan = SeqScanOp(db.get_table('employees'))
        rows1 = collect(scan)
        rows2 = collect(scan)
        self.assertEqual(len(rows1), len(rows2))

    def test_like_special_chars(self):
        row = Row({'t.name': 'hello.world'})
        self.assertTrue(eval_expr(
            Comparison(CompOp.LIKE, ColumnRef('t', 'name'), Literal('hello.%')), row))

    def test_between_boundary(self):
        row = Row({'t.x': 10})
        self.assertTrue(eval_expr(
            Comparison(CompOp.BETWEEN, ColumnRef('t', 'x'), Literal((10, 20))), row))
        self.assertTrue(eval_expr(
            Comparison(CompOp.BETWEEN, ColumnRef('t', 'x'), Literal((5, 10))), row))

    def test_in_single_value(self):
        row = Row({'t.x': 10})
        self.assertTrue(eval_expr(
            Comparison(CompOp.IN, ColumnRef('t', 'x'), Literal(10)), row))


if __name__ == '__main__':
    unittest.main()
