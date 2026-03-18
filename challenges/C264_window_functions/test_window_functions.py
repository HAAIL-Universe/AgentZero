"""
Tests for C264: Window Functions
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from window_functions import (
    WindowDB, WindowParser, WindowSpec, FrameBound, SqlWindowFunc,
    parse_window_sql, parse_window_sql_multi,
    _frame_indices, _default_frame, _compute_rank_info,
    RANKING_FUNCS, NAVIGATION_FUNCS, AGGREGATE_FUNCS,
)


# =============================================================================
# Helpers
# =============================================================================

@pytest.fixture
def db():
    """Create a WindowDB with test data."""
    d = WindowDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary INT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 'eng', 100)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 'eng', 120)")
    d.execute("INSERT INTO employees VALUES (3, 'Carol', 'eng', 90)")
    d.execute("INSERT INTO employees VALUES (4, 'Dave', 'sales', 80)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 'sales', 110)")
    d.execute("INSERT INTO employees VALUES (6, 'Frank', 'sales', 110)")
    d.execute("INSERT INTO employees VALUES (7, 'Grace', 'hr', 95)")
    d.execute("INSERT INTO employees VALUES (8, 'Hank', 'hr', 85)")
    return d


@pytest.fixture
def scores_db():
    """DB with score data for ranking tests."""
    d = WindowDB()
    d.execute("CREATE TABLE scores (student TEXT, subject TEXT, score INT)")
    d.execute("INSERT INTO scores VALUES ('Alice', 'math', 90)")
    d.execute("INSERT INTO scores VALUES ('Bob', 'math', 85)")
    d.execute("INSERT INTO scores VALUES ('Carol', 'math', 90)")
    d.execute("INSERT INTO scores VALUES ('Dave', 'math', 80)")
    d.execute("INSERT INTO scores VALUES ('Alice', 'eng', 95)")
    d.execute("INSERT INTO scores VALUES ('Bob', 'eng', 88)")
    d.execute("INSERT INTO scores VALUES ('Carol', 'eng', 92)")
    d.execute("INSERT INTO scores VALUES ('Dave', 'eng', 88)")
    return d


def result_to_dicts(result):
    """Convert ResultSet to list of dicts."""
    return [dict(zip(result.columns, row)) for row in result.rows]


# =============================================================================
# Parser tests
# =============================================================================

class TestParser:
    def test_parse_row_number_over_empty(self):
        stmt = parse_window_sql("SELECT ROW_NUMBER() OVER () FROM t")
        assert isinstance(stmt.columns[0].expr, SqlWindowFunc)
        assert stmt.columns[0].expr.func_name.lower() == 'row_number'

    def test_parse_rank_with_order_by(self):
        stmt = parse_window_sql("SELECT RANK() OVER (ORDER BY salary DESC) FROM t")
        wf = stmt.columns[0].expr
        assert isinstance(wf, SqlWindowFunc)
        assert wf.window.order_by is not None
        assert len(wf.window.order_by) == 1
        assert wf.window.order_by[0][1] == False  # DESC

    def test_parse_partition_by(self):
        stmt = parse_window_sql(
            "SELECT SUM(salary) OVER (PARTITION BY dept) FROM t"
        )
        wf = stmt.columns[0].expr
        assert wf.window.partition_by is not None
        assert len(wf.window.partition_by) == 1

    def test_parse_partition_and_order(self):
        stmt = parse_window_sql(
            "SELECT RANK() OVER (PARTITION BY dept ORDER BY salary DESC) FROM t"
        )
        wf = stmt.columns[0].expr
        assert wf.window.partition_by is not None
        assert wf.window.order_by is not None

    def test_parse_frame_rows_between(self):
        stmt = parse_window_sql(
            "SELECT SUM(salary) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t"
        )
        wf = stmt.columns[0].expr
        assert wf.window.frame_start.bound_type == 'preceding'
        assert wf.window.frame_start.offset == 1
        assert wf.window.frame_end.bound_type == 'following'
        assert wf.window.frame_end.offset == 1

    def test_parse_frame_unbounded(self):
        stmt = parse_window_sql(
            "SELECT SUM(salary) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t"
        )
        wf = stmt.columns[0].expr
        assert wf.window.frame_start.bound_type == 'unbounded_preceding'
        assert wf.window.frame_end.bound_type == 'current_row'

    def test_parse_with_alias(self):
        stmt = parse_window_sql(
            "SELECT ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t"
        )
        assert stmt.columns[0].alias == 'rn'

    def test_parse_lag_with_offset(self):
        stmt = parse_window_sql(
            "SELECT LAG(salary, 2) OVER (ORDER BY id) FROM t"
        )
        wf = stmt.columns[0].expr
        assert wf.func_name.lower() == 'lag'
        assert len(wf.args) == 2

    def test_parse_multiple_window_funcs(self):
        stmt = parse_window_sql(
            "SELECT ROW_NUMBER() OVER (ORDER BY id) AS rn, "
            "RANK() OVER (ORDER BY salary DESC) AS rnk FROM t"
        )
        assert len(stmt.columns) == 2
        assert isinstance(stmt.columns[0].expr, SqlWindowFunc)
        assert isinstance(stmt.columns[1].expr, SqlWindowFunc)

    def test_parse_mixed_columns_and_window(self):
        stmt = parse_window_sql(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t"
        )
        assert len(stmt.columns) == 2
        assert not isinstance(stmt.columns[0].expr, SqlWindowFunc)
        assert isinstance(stmt.columns[1].expr, SqlWindowFunc)

    def test_parse_ntile(self):
        stmt = parse_window_sql(
            "SELECT NTILE(4) OVER (ORDER BY salary) FROM t"
        )
        wf = stmt.columns[0].expr
        assert wf.func_name.lower() == 'ntile'
        assert len(wf.args) == 1

    def test_parse_count_star_over(self):
        stmt = parse_window_sql(
            "SELECT COUNT(*) OVER (PARTITION BY dept) AS cnt FROM t"
        )
        wf = stmt.columns[0].expr
        assert wf.func_name.lower() == 'count'
        assert len(wf.args) == 0


# =============================================================================
# Frame index tests
# =============================================================================

class TestFrameIndices:
    def test_unbounded_preceding_to_current(self):
        s, e = _frame_indices(
            FrameBound('unbounded_preceding'),
            FrameBound('current_row'),
            3, 10
        )
        assert s == 0
        assert e == 3

    def test_current_to_unbounded_following(self):
        s, e = _frame_indices(
            FrameBound('current_row'),
            FrameBound('unbounded_following'),
            3, 10
        )
        assert s == 3
        assert e == 9

    def test_n_preceding_to_n_following(self):
        s, e = _frame_indices(
            FrameBound('preceding', 2),
            FrameBound('following', 2),
            5, 10
        )
        assert s == 3
        assert e == 7

    def test_preceding_at_start(self):
        s, e = _frame_indices(
            FrameBound('preceding', 3),
            FrameBound('current_row'),
            1, 10
        )
        assert s == 0  # clamped
        assert e == 1

    def test_following_at_end(self):
        s, e = _frame_indices(
            FrameBound('current_row'),
            FrameBound('following', 5),
            8, 10
        )
        assert s == 8
        assert e == 9  # clamped

    def test_full_partition(self):
        s, e = _frame_indices(
            FrameBound('unbounded_preceding'),
            FrameBound('unbounded_following'),
            5, 10
        )
        assert s == 0
        assert e == 9


# =============================================================================
# ROW_NUMBER tests
# =============================================================================

class TestRowNumber:
    def test_row_number_basic(self, db):
        r = db.execute(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY salary) AS rn FROM employees"
        )
        dicts = result_to_dicts(r)
        # Should assign 1..8 ordered by salary
        rns = [d['rn'] for d in dicts]
        assert sorted(rns) == list(range(1, 9))

    def test_row_number_partitioned(self, db):
        r = db.execute(
            "SELECT name, dept, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) AS rn "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # Within each dept, rn should start at 1
        for dept in ['eng', 'sales', 'hr']:
            dept_rows = [d for d in dicts if d['dept'] == dept]
            rns = sorted([d['rn'] for d in dept_rows])
            assert rns == list(range(1, len(dept_rows) + 1))

    def test_row_number_desc(self, db):
        r = db.execute(
            "SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn FROM employees"
        )
        dicts = result_to_dicts(r)
        # Highest salary gets rn=1
        top = [d for d in dicts if d['rn'] == 1][0]
        assert top['name'] == 'Bob'  # salary=120


# =============================================================================
# RANK / DENSE_RANK tests
# =============================================================================

class TestRank:
    def test_rank_with_ties(self, scores_db):
        r = scores_db.execute(
            "SELECT student, subject, score, "
            "RANK() OVER (PARTITION BY subject ORDER BY score DESC) AS rnk "
            "FROM scores"
        )
        dicts = result_to_dicts(r)
        math_rows = [d for d in dicts if d.get('subject') == 'math']
        # Scores: 90, 90, 85, 80 -> ranks: 1, 1, 3, 4
        math_rows.sort(key=lambda x: -x.get('score', 0) if x.get('score') is not None else 0)
        ranks = [d['rnk'] for d in math_rows]
        assert ranks == [1, 1, 3, 4]

    def test_dense_rank_with_ties(self, scores_db):
        r = scores_db.execute(
            "SELECT student, subject, score, "
            "DENSE_RANK() OVER (PARTITION BY subject ORDER BY score DESC) AS drnk "
            "FROM scores"
        )
        dicts = result_to_dicts(r)
        math_rows = [d for d in dicts if d.get('subject') == 'math']
        math_rows.sort(key=lambda x: -x.get('score', 0) if x.get('score') is not None else 0)
        dranks = [d['drnk'] for d in math_rows]
        assert dranks == [1, 1, 2, 3]

    def test_rank_no_partition(self, db):
        r = db.execute(
            "SELECT name, salary, RANK() OVER (ORDER BY salary DESC) AS rnk FROM employees"
        )
        dicts = result_to_dicts(r)
        # Eve and Frank both have salary=110
        ties = [d for d in dicts if d['salary'] == 110]
        assert all(d['rnk'] == ties[0]['rnk'] for d in ties)

    def test_dense_rank_no_partition(self, db):
        r = db.execute(
            "SELECT name, salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS drnk FROM employees"
        )
        dicts = result_to_dicts(r)
        # salary order: 120, 110, 110, 100, 95, 90, 85, 80
        # dense_rank:     1,   2,   2,   3,  4,  5,  6,  7
        bob = [d for d in dicts if d['name'] == 'Bob'][0]
        assert bob['drnk'] == 1
        alice = [d for d in dicts if d['name'] == 'Alice'][0]
        assert alice['drnk'] == 3  # after 120 and 110


# =============================================================================
# NTILE tests
# =============================================================================

class TestNtile:
    def test_ntile_even_distribution(self, db):
        r = db.execute(
            "SELECT name, NTILE(4) OVER (ORDER BY salary) AS q FROM employees"
        )
        dicts = result_to_dicts(r)
        quartiles = [d['q'] for d in dicts]
        assert set(quartiles) == {1, 2, 3, 4}
        # 8 rows / 4 buckets = 2 per bucket
        for q in [1, 2, 3, 4]:
            assert quartiles.count(q) == 2

    def test_ntile_uneven(self, db):
        r = db.execute(
            "SELECT name, NTILE(3) OVER (ORDER BY salary) AS tercile FROM employees"
        )
        dicts = result_to_dicts(r)
        # 8 rows / 3 = first buckets get extra
        terciles = [d['tercile'] for d in dicts]
        assert set(terciles) == {1, 2, 3}


# =============================================================================
# LAG / LEAD tests
# =============================================================================

class TestLagLead:
    def test_lag_basic(self, db):
        r = db.execute(
            "SELECT name, salary, LAG(salary) OVER (ORDER BY salary) AS prev_sal "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # First row should have NULL lag
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[0]['prev_sal'] is None
        # Second row should have first row's salary
        assert dicts[1]['prev_sal'] == dicts[0]['salary']

    def test_lag_with_offset(self, db):
        r = db.execute(
            "SELECT name, salary, LAG(salary, 2) OVER (ORDER BY salary) AS prev2 "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[0]['prev2'] is None
        assert dicts[1]['prev2'] is None
        assert dicts[2]['prev2'] == dicts[0]['salary']

    def test_lag_with_default(self, db):
        r = db.execute(
            "SELECT name, salary, LAG(salary, 1, 0) OVER (ORDER BY salary) AS prev_sal "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[0]['prev_sal'] == 0  # default value

    def test_lead_basic(self, db):
        r = db.execute(
            "SELECT name, salary, LEAD(salary) OVER (ORDER BY salary) AS next_sal "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[-1]['next_sal'] is None
        assert dicts[0]['next_sal'] == dicts[1]['salary']

    def test_lead_with_offset(self, db):
        r = db.execute(
            "SELECT name, salary, LEAD(salary, 2) OVER (ORDER BY salary) AS next2 "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[-1]['next2'] is None
        assert dicts[-2]['next2'] is None
        assert dicts[0]['next2'] == dicts[2]['salary']

    def test_lag_partitioned(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "LAG(salary) OVER (PARTITION BY dept ORDER BY salary) AS prev_sal "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # First employee in each dept should have NULL prev_sal
        for dept in ['eng', 'sales', 'hr']:
            dept_rows = sorted(
                [d for d in dicts if d['dept'] == dept],
                key=lambda x: x['salary']
            )
            assert dept_rows[0]['prev_sal'] is None


# =============================================================================
# FIRST_VALUE / LAST_VALUE / NTH_VALUE tests
# =============================================================================

class TestValueFunctions:
    def test_first_value(self, db):
        r = db.execute(
            "SELECT name, salary, FIRST_VALUE(name) OVER (ORDER BY salary DESC) AS top_name "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # With default frame (UNBOUNDED PRECEDING to CURRENT ROW), first_value is the first in partition
        assert all(d['top_name'] == 'Bob' for d in dicts)

    def test_last_value_full_frame(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "LAST_VALUE(name) OVER (ORDER BY salary ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS bottom_name "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # Last value in the whole partition (ordered by salary asc) is the highest salary
        assert all(d['bottom_name'] == 'Bob' for d in dicts)

    def test_first_value_partitioned(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "FIRST_VALUE(name) OVER (PARTITION BY dept ORDER BY salary DESC) AS top_in_dept "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        eng_top = [d for d in dicts if d['dept'] == 'eng']
        assert all(d['top_in_dept'] == 'Bob' for d in eng_top)

    def test_nth_value(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "NTH_VALUE(name, 2) OVER (ORDER BY salary DESC "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS second_name "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # Second highest salary: either Eve or Frank (both 110)
        second_names = set(d['second_name'] for d in dicts)
        assert second_names.issubset({'Eve', 'Frank'})


# =============================================================================
# Aggregate window function tests
# =============================================================================

class TestAggregateWindows:
    def test_sum_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "SUM(salary) OVER (PARTITION BY dept) AS dept_total "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        eng = [d for d in dicts if d['dept'] == 'eng']
        # eng salaries: 100 + 120 + 90 = 310
        assert all(d['dept_total'] == 310 for d in eng)
        sales = [d for d in dicts if d['dept'] == 'sales']
        assert all(d['dept_total'] == 300 for d in sales)

    def test_avg_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "AVG(salary) OVER (PARTITION BY dept) AS dept_avg "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        hr = [d for d in dicts if d['dept'] == 'hr']
        # hr salaries: 95, 85 -> avg = 90
        assert all(d['dept_avg'] == 90.0 for d in hr)

    def test_count_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, "
            "COUNT(*) OVER (PARTITION BY dept) AS dept_size "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        eng = [d for d in dicts if d['dept'] == 'eng']
        assert all(d['dept_size'] == 3 for d in eng)
        hr = [d for d in dicts if d['dept'] == 'hr']
        assert all(d['dept_size'] == 2 for d in hr)

    def test_min_max_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "MIN(salary) OVER (PARTITION BY dept) AS dept_min, "
            "MAX(salary) OVER (PARTITION BY dept) AS dept_max "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        eng = [d for d in dicts if d['dept'] == 'eng']
        assert all(d['dept_min'] == 90 for d in eng)
        assert all(d['dept_max'] == 120 for d in eng)

    def test_running_sum(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary) AS running_total "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        # Running sum should accumulate
        running = 0
        for d in dicts:
            running += d['salary']
            assert d['running_total'] == running

    def test_running_count(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "COUNT(*) OVER (ORDER BY salary) AS running_count "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        expected = 1
        for i, d in enumerate(dicts):
            # With ties, running count includes all ties
            assert d['running_count'] >= expected
            expected = d['running_count'] + 1

    def test_sum_no_partition_no_order(self, db):
        r = db.execute(
            "SELECT name, salary, SUM(salary) OVER () AS total FROM employees"
        )
        dicts = result_to_dicts(r)
        total = 100 + 120 + 90 + 80 + 110 + 110 + 95 + 85
        assert all(d['total'] == total for d in dicts)


# =============================================================================
# Frame specification tests
# =============================================================================

class TestFrameSpec:
    def test_moving_average_3(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "AVG(salary) OVER (ORDER BY salary "
            "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving_avg "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        # First row: avg of rows[0:2]
        # Middle rows: avg of 3 rows
        # Last row: avg of rows[-2:]
        salaries = [d['salary'] for d in dicts]
        assert dicts[0]['moving_avg'] == pytest.approx(
            sum(salaries[0:2]) / 2
        )

    def test_preceding_only(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary "
            "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS last3_sum "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        salaries = [d['salary'] for d in dicts]
        # First: just itself
        assert dicts[0]['last3_sum'] == salaries[0]
        # Second: sum of 2
        assert dicts[1]['last3_sum'] == sum(salaries[0:2])
        # Third+: sum of 3
        assert dicts[2]['last3_sum'] == sum(salaries[0:3])

    def test_following_only(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary "
            "ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) AS next2_sum "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        salaries = [d['salary'] for d in dicts]
        # Last row: just itself
        assert dicts[-1]['next2_sum'] == salaries[-1]
        # Others: sum of current + next
        assert dicts[0]['next2_sum'] == salaries[0] + salaries[1]


# =============================================================================
# Multiple window functions in one query
# =============================================================================

class TestMultipleWindowFuncs:
    def test_two_different_funcs(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn, "
            "SUM(salary) OVER () AS total "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        total = sum(d['salary'] for d in dicts)
        assert all(d['total'] == total for d in dicts)
        rns = sorted(d['rn'] for d in dicts)
        assert rns == list(range(1, 9))

    def test_different_partitions(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) AS dept_rn, "
            "ROW_NUMBER() OVER (ORDER BY salary) AS global_rn "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # dept_rn resets per dept
        for dept in ['eng', 'sales', 'hr']:
            dept_rows = [d for d in dicts if d['dept'] == dept]
            assert sorted(d['dept_rn'] for d in dept_rows) == list(range(1, len(dept_rows) + 1))
        # global_rn is 1..8
        assert sorted(d['global_rn'] for d in dicts) == list(range(1, 9))

    def test_three_funcs_same_query(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "RANK() OVER (ORDER BY salary DESC) AS rnk, "
            "DENSE_RANK() OVER (ORDER BY salary DESC) AS drnk, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees"
        )
        assert len(r.columns) == 5
        assert len(r.rows) == 8


# =============================================================================
# Mixed regular columns and window functions
# =============================================================================

class TestMixedColumns:
    def test_name_and_window(self, db):
        r = db.execute(
            "SELECT name, salary, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM employees"
        )
        dicts = result_to_dicts(r)
        assert all('name' in d and 'salary' in d and 'rn' in d for d in dicts)

    def test_expression_and_window(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (PARTITION BY dept) AS dept_total "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        for d in dicts:
            assert d['salary'] is not None
            assert d['dept_total'] is not None


# =============================================================================
# PERCENT_RANK / CUME_DIST tests
# =============================================================================

class TestPercentRankCumeDist:
    def test_percent_rank_basic(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "PERCENT_RANK() OVER (ORDER BY salary) AS prank "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        pranks = [d['prank'] for d in dicts]
        # Lowest salary should have percent_rank 0
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[0]['prank'] == pytest.approx(0.0)
        # Highest should have percent_rank 1.0
        assert dicts[-1]['prank'] == pytest.approx(1.0)

    def test_cume_dist_basic(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "CUME_DIST() OVER (ORDER BY salary) AS cdist "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # All cume_dist values should be between 0 and 1
        for d in dicts:
            assert 0 < d['cdist'] <= 1.0
        # Highest salary should have cume_dist 1.0
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[-1]['cdist'] == pytest.approx(1.0)


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_empty_table(self):
        d = WindowDB()
        d.execute("CREATE TABLE empty (id INT, val INT)")
        r = d.execute(
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM empty"
        )
        assert len(r.rows) == 0

    def test_single_row(self):
        d = WindowDB()
        d.execute("CREATE TABLE one (id INT, val INT)")
        d.execute("INSERT INTO one VALUES (1, 42)")
        r = d.execute(
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn, "
            "RANK() OVER (ORDER BY id) AS rnk, "
            "SUM(val) OVER () AS total "
            "FROM one"
        )
        dicts = result_to_dicts(r)
        assert dicts[0]['rn'] == 1
        assert dicts[0]['rnk'] == 1
        assert dicts[0]['total'] == 42

    def test_all_same_values(self):
        d = WindowDB()
        d.execute("CREATE TABLE same (id INT, val INT)")
        d.execute("INSERT INTO same VALUES (1, 10)")
        d.execute("INSERT INTO same VALUES (2, 10)")
        d.execute("INSERT INTO same VALUES (3, 10)")
        r = d.execute(
            "SELECT id, RANK() OVER (ORDER BY val) AS rnk, "
            "DENSE_RANK() OVER (ORDER BY val) AS drnk "
            "FROM same"
        )
        dicts = result_to_dicts(r)
        # All same value -> all rank 1
        assert all(d['rnk'] == 1 for d in dicts)
        assert all(d['drnk'] == 1 for d in dicts)

    def test_null_values(self):
        d = WindowDB()
        d.execute("CREATE TABLE nulls (id INT, val INT)")
        d.execute("INSERT INTO nulls VALUES (1, 10)")
        d.execute("INSERT INTO nulls VALUES (2, NULL)")
        d.execute("INSERT INTO nulls VALUES (3, 30)")
        r = d.execute(
            "SELECT id, val, SUM(val) OVER () AS total FROM nulls"
        )
        dicts = result_to_dicts(r)
        # SUM should skip NULLs: 10 + 30 = 40
        assert all(d['total'] == 40 for d in dicts)

    def test_window_preserves_row_order(self, db):
        r = db.execute(
            "SELECT name, salary, SUM(salary) OVER () AS total FROM employees"
        )
        # Should return all 8 rows
        assert len(r.rows) == 8

    def test_window_with_where(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees WHERE dept = 'eng'"
        )
        dicts = result_to_dicts(r)
        assert len(dicts) == 3
        assert sorted(d['rn'] for d in dicts) == [1, 2, 3]

    def test_count_star_window(self, db):
        r = db.execute(
            "SELECT name, dept, "
            "COUNT(*) OVER (PARTITION BY dept) AS dept_size "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        eng = [d for d in dicts if d['dept'] == 'eng']
        assert all(d['dept_size'] == 3 for d in eng)

    def test_multiple_partition_by_columns(self, scores_db):
        # Partition by two columns
        r = scores_db.execute(
            "SELECT student, subject, score, "
            "ROW_NUMBER() OVER (PARTITION BY subject ORDER BY score DESC) AS rnk "
            "FROM scores"
        )
        dicts = result_to_dicts(r)
        # Each subject should have ranks 1-4
        for subj in ['math', 'eng']:
            subj_rows = [d for d in dicts if d.get('subject') == subj]
            assert sorted(d['rnk'] for d in subj_rows) == [1, 2, 3, 4]


# =============================================================================
# Non-window operations still work
# =============================================================================

class TestNonWindowOps:
    def test_regular_select(self, db):
        r = db.execute("SELECT name, salary FROM employees WHERE dept = 'hr'")
        assert len(r.rows) == 2

    def test_insert_update_delete(self, db):
        db.execute("INSERT INTO employees VALUES (9, 'Iris', 'hr', 92)")
        r = db.execute("SELECT name FROM employees WHERE id = 9")
        assert r.rows[0][0] == 'Iris'
        db.execute("UPDATE employees SET salary = 93 WHERE id = 9")
        r = db.execute("SELECT salary FROM employees WHERE id = 9")
        assert r.rows[0][0] == 93
        db.execute("DELETE FROM employees WHERE id = 9")
        r = db.execute("SELECT name FROM employees WHERE id = 9")
        assert len(r.rows) == 0

    def test_aggregate_without_window(self, db):
        r = db.execute("SELECT dept, SUM(salary) FROM employees GROUP BY dept")
        assert len(r.rows) == 3

    def test_create_table(self, db):
        db.execute("CREATE TABLE test_tbl (a INT, b TEXT)")
        db.execute("INSERT INTO test_tbl VALUES (1, 'x')")
        r = db.execute("SELECT a, b FROM test_tbl")
        assert r.rows[0] == [1, 'x']

    def test_ctas_still_works(self, db):
        db.execute(
            "CREATE TABLE eng_salaries AS SELECT name, salary FROM employees WHERE dept = 'eng'"
        )
        r = db.execute("SELECT name, salary FROM eng_salaries")
        assert len(r.rows) == 3


# =============================================================================
# Integration / complex queries
# =============================================================================

class TestIntegration:
    def test_salary_percentile(self, db):
        """Salary with running percentage."""
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary) AS cumulative, "
            "SUM(salary) OVER () AS total "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        for d in dicts:
            assert d['cumulative'] <= d['total']

    def test_dept_rank_and_global_rank(self, db):
        """Compare department rank vs global rank."""
        r = db.execute(
            "SELECT name, dept, salary, "
            "RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS dept_rnk, "
            "RANK() OVER (ORDER BY salary DESC) AS global_rnk "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # Bob: dept_rnk=1 in eng, global_rnk=1
        bob = [d for d in dicts if d['name'] == 'Bob'][0]
        assert bob['dept_rnk'] == 1
        assert bob['global_rnk'] == 1

    def test_moving_sum_with_partition(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "SUM(salary) OVER (PARTITION BY dept ORDER BY salary "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        # Within each dept, running sum should accumulate
        for dept in ['eng', 'sales', 'hr']:
            dept_rows = sorted(
                [d for d in dicts if d['dept'] == dept],
                key=lambda x: x['salary']
            )
            running = 0
            for d in dept_rows:
                running += d['salary']
                assert d['running'] == running

    def test_lag_lead_same_query(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "LAG(salary) OVER (ORDER BY salary) AS prev, "
            "LEAD(salary) OVER (ORDER BY salary) AS next "
            "FROM employees"
        )
        dicts = result_to_dicts(r)
        dicts.sort(key=lambda d: d['salary'])
        assert dicts[0]['prev'] is None
        assert dicts[-1]['next'] is None

    def test_window_with_where_filter(self):
        """Window functions applied after WHERE filter."""
        d = WindowDB()
        d.execute("CREATE TABLE sales (id INT, region TEXT, amount INT)")
        d.execute("INSERT INTO sales VALUES (1, 'east', 100)")
        d.execute("INSERT INTO sales VALUES (2, 'east', 200)")
        d.execute("INSERT INTO sales VALUES (3, 'west', 150)")
        d.execute("INSERT INTO sales VALUES (4, 'west', 250)")
        d.execute("INSERT INTO sales VALUES (5, 'east', 300)")
        r = d.execute(
            "SELECT region, amount, "
            "SUM(amount) OVER (PARTITION BY region) AS region_total "
            "FROM sales WHERE amount >= 150"
        )
        dicts = result_to_dicts(r)
        # east rows with amount >= 150: 200, 300 -> total 500
        east_rows = [dd for dd in dicts if dd['region'] == 'east']
        assert all(dd['region_total'] == 500 for dd in east_rows)
        # west rows: 150, 250 -> total 400
        west_rows = [dd for dd in dicts if dd['region'] == 'west']
        assert all(dd['region_total'] == 400 for dd in west_rows)


# =============================================================================
# Default frame behavior
# =============================================================================

class TestDefaultFrames:
    def test_default_frame_ranking(self):
        s, e = _default_frame('row_number', True)
        assert s.bound_type == 'unbounded_preceding'
        assert e.bound_type == 'unbounded_following'

    def test_default_frame_agg_with_order(self):
        s, e = _default_frame('sum', True)
        assert s.bound_type == 'unbounded_preceding'
        assert e.bound_type == 'current_row'

    def test_default_frame_agg_no_order(self):
        s, e = _default_frame('sum', False)
        assert s.bound_type == 'unbounded_preceding'
        assert e.bound_type == 'unbounded_following'


# =============================================================================
# Rank info computation
# =============================================================================

class TestRankInfo:
    def test_compute_rank_no_order(self):
        from query_executor import Row
        rows = [Row({'a': 1}), Row({'a': 2})]
        infos = _compute_rank_info(rows, None)
        assert all(i['rank'] == 1 for i in infos)

    def test_compute_rank_with_ties(self):
        from query_executor import Row
        from mini_database import SqlColumnRef
        rows = [Row({'a': 10}), Row({'a': 10}), Row({'a': 20})]
        infos = _compute_rank_info(rows, [(SqlColumnRef(None, 'a'), True)])
        assert infos[0]['rank'] == 1
        assert infos[1]['rank'] == 1
        assert infos[2]['rank'] == 3
        assert infos[0]['dense_rank'] == 1
        assert infos[1]['dense_rank'] == 1
        assert infos[2]['dense_rank'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
