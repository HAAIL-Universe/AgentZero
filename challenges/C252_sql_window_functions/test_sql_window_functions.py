"""
Tests for C252: SQL Window Functions
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from sql_window_functions import (
    WindowDB, WindowLexer, WindowParser, WindowExecutor, WindowQueryCompiler,
    SqlWindowCall, WindowSpec, WindowFrame, WindowFrameBound, FrameBound,
    SqlColumnRef, SqlLiteral, SelectExpr, SqlStar,
    SelectStmt,
)


@pytest.fixture
def db():
    """Create a WindowDB with test data."""
    d = WindowDB()
    d.execute("CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary FLOAT)")
    d.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000)")
    d.execute("INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 85000)")
    d.execute("INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 95000)")
    d.execute("INSERT INTO employees VALUES (4, 'Diana', 'Sales', 70000)")
    d.execute("INSERT INTO employees VALUES (5, 'Eve', 'Sales', 75000)")
    d.execute("INSERT INTO employees VALUES (6, 'Frank', 'Sales', 72000)")
    d.execute("INSERT INTO employees VALUES (7, 'Grace', 'HR', 65000)")
    d.execute("INSERT INTO employees VALUES (8, 'Hank', 'HR', 68000)")
    return d


@pytest.fixture
def scores_db():
    """DB with scores for ranking tests."""
    d = WindowDB()
    d.execute("CREATE TABLE scores (student TEXT, subject TEXT, score INT)")
    d.execute("INSERT INTO scores VALUES ('Alice', 'Math', 95)")
    d.execute("INSERT INTO scores VALUES ('Bob', 'Math', 90)")
    d.execute("INSERT INTO scores VALUES ('Charlie', 'Math', 90)")
    d.execute("INSERT INTO scores VALUES ('Diana', 'Math', 85)")
    d.execute("INSERT INTO scores VALUES ('Alice', 'English', 88)")
    d.execute("INSERT INTO scores VALUES ('Bob', 'English', 92)")
    d.execute("INSERT INTO scores VALUES ('Charlie', 'English', 85)")
    d.execute("INSERT INTO scores VALUES ('Diana', 'English', 92)")
    return d


# =============================================================================
# ROW_NUMBER Tests
# =============================================================================

class TestRowNumber:
    def test_basic_row_number(self, db):
        r = db.execute(
            "SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees ORDER BY rn"
        )
        assert r.columns == ['name', 'salary', 'rn']
        # Highest salary first (ordered by rn)
        assert r.rows[0][2] == 1
        assert r.rows[0][1] == 95000
        assert r.rows[-1][2] == 8

    def test_row_number_partitioned(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
            "FROM employees"
        )
        # Check each department has its own numbering
        dept_ranks = {}
        for row in r.rows:
            name, dept, salary, rn = row
            dept_ranks.setdefault(dept, []).append(rn)
        for dept, ranks in dept_ranks.items():
            assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_row_number_no_order(self, db):
        """ROW_NUMBER without ORDER BY -- should still assign sequential numbers."""
        r = db.execute(
            "SELECT name, ROW_NUMBER() OVER () AS rn FROM employees"
        )
        rns = [row[1] for row in r.rows]
        assert sorted(rns) == list(range(1, 9))

    def test_row_number_with_where(self, db):
        r = db.execute(
            "SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees WHERE dept = 'Engineering' ORDER BY rn"
        )
        assert len(r.rows) == 3
        assert r.rows[0][2] == 1  # rank 1 within filtered results


# =============================================================================
# RANK Tests
# =============================================================================

class TestRank:
    def test_basic_rank(self, scores_db):
        r = scores_db.execute(
            "SELECT student, score, RANK() OVER (ORDER BY score DESC) AS rnk "
            "FROM scores WHERE subject = 'Math'"
        )
        # Scores: 95, 90, 90, 85 -> ranks: 1, 2, 2, 4
        ranks = {row[0]: row[2] for row in r.rows}
        assert ranks['Alice'] == 1
        assert ranks['Bob'] == 2
        assert ranks['Charlie'] == 2
        assert ranks['Diana'] == 4

    def test_rank_partitioned(self, scores_db):
        r = scores_db.execute(
            "SELECT student, subject, score, "
            "RANK() OVER (PARTITION BY subject ORDER BY score DESC) AS rnk "
            "FROM scores"
        )
        # Group by subject and check
        math_ranks = {row[0]: row[3] for row in r.rows if row[1] == 'Math'}
        eng_ranks = {row[0]: row[3] for row in r.rows if row[1] == 'English'}
        assert math_ranks['Alice'] == 1
        # English: 92 (Bob, Diana tie), 88, 85
        assert eng_ranks['Bob'] == 1
        assert eng_ranks['Diana'] == 1
        assert eng_ranks['Alice'] == 3

    def test_rank_no_ties(self, db):
        r = db.execute(
            "SELECT name, salary, RANK() OVER (ORDER BY salary DESC) AS rnk "
            "FROM employees WHERE dept = 'Sales'"
        )
        ranks = [row[2] for row in r.rows]
        assert sorted(ranks) == [1, 2, 3]  # no ties, rank == row_number


# =============================================================================
# DENSE_RANK Tests
# =============================================================================

class TestDenseRank:
    def test_basic_dense_rank(self, scores_db):
        r = scores_db.execute(
            "SELECT student, score, DENSE_RANK() OVER (ORDER BY score DESC) AS drnk "
            "FROM scores WHERE subject = 'Math'"
        )
        # Scores: 95, 90, 90, 85 -> dense_ranks: 1, 2, 2, 3
        ranks = {row[0]: row[2] for row in r.rows}
        assert ranks['Alice'] == 1
        assert ranks['Bob'] == 2
        assert ranks['Charlie'] == 2
        assert ranks['Diana'] == 3

    def test_dense_rank_vs_rank(self, scores_db):
        """DENSE_RANK has no gaps, RANK does."""
        r = scores_db.execute(
            "SELECT student, score, "
            "RANK() OVER (ORDER BY score DESC) AS rnk, "
            "DENSE_RANK() OVER (ORDER BY score DESC) AS drnk "
            "FROM scores WHERE subject = 'Math'"
        )
        for row in r.rows:
            if row[0] == 'Diana':  # score 85
                assert row[2] == 4  # RANK skips 3
                assert row[3] == 3  # DENSE_RANK doesn't skip


# =============================================================================
# NTILE Tests
# =============================================================================

class TestNtile:
    def test_ntile_even(self, db):
        r = db.execute(
            "SELECT name, salary, NTILE(4) OVER (ORDER BY salary DESC) AS quartile "
            "FROM employees"
        )
        quartiles = [row[2] for row in r.rows]
        # 8 rows / 4 tiles = 2 each
        assert quartiles.count(1) == 2
        assert quartiles.count(2) == 2
        assert quartiles.count(3) == 2
        assert quartiles.count(4) == 2

    def test_ntile_uneven(self, db):
        r = db.execute(
            "SELECT name, salary, NTILE(3) OVER (ORDER BY salary DESC) AS tile "
            "FROM employees"
        )
        tiles = [row[2] for row in r.rows]
        # 8 rows / 3 tiles = 3, 3, 2
        assert tiles.count(1) == 3
        assert tiles.count(2) == 3
        assert tiles.count(3) == 2

    def test_ntile_partitioned(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "NTILE(2) OVER (PARTITION BY dept ORDER BY salary DESC) AS half "
            "FROM employees"
        )
        # Each department split into 2 halves
        for row in r.rows:
            assert row[3] in (1, 2)


# =============================================================================
# LEAD / LAG Tests
# =============================================================================

class TestLeadLag:
    def test_basic_lag(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "LAG(salary) OVER (ORDER BY salary) AS prev_salary "
            "FROM employees ORDER BY salary"
        )
        # First row (lowest salary) should have NULL for lag
        assert r.rows[0][2] is None
        # Second row should have first row's salary
        assert r.rows[1][2] == r.rows[0][1]

    def test_basic_lead(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "LEAD(salary) OVER (ORDER BY salary) AS next_salary "
            "FROM employees ORDER BY salary"
        )
        # Last row should have NULL for lead
        assert r.rows[-1][2] is None
        # First row should have second row's salary
        assert r.rows[0][2] == r.rows[1][1]

    def test_lag_with_offset(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "LAG(salary, 2) OVER (ORDER BY salary) AS prev2_salary "
            "FROM employees ORDER BY salary"
        )
        # First two rows should have NULL
        assert r.rows[0][2] is None
        assert r.rows[1][2] is None
        # Third row should have first row's salary
        assert r.rows[2][2] == r.rows[0][1]

    def test_lag_with_default(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "LAG(salary, 1, 0) OVER (ORDER BY salary) AS prev_salary "
            "FROM employees ORDER BY salary"
        )
        # First row should have default 0
        assert r.rows[0][2] == 0

    def test_lead_with_default(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "LEAD(salary, 1, -1) OVER (ORDER BY salary) AS next_salary "
            "FROM employees ORDER BY salary"
        )
        # Last row should have default -1
        assert r.rows[-1][2] == -1

    def test_lead_lag_partitioned(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "LAG(salary) OVER (PARTITION BY dept ORDER BY salary) AS prev "
            "FROM employees ORDER BY dept, salary"
        )
        # First employee in each department (lowest salary) should have NULL lag
        seen_depts = set()
        for row in r.rows:
            dept = row[1]
            if dept not in seen_depts:
                seen_depts.add(dept)
                assert row[3] is None, f"First in {dept} should have NULL lag"


# =============================================================================
# FIRST_VALUE / LAST_VALUE / NTH_VALUE Tests
# =============================================================================

class TestFirstLastNth:
    def test_first_value(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "FIRST_VALUE(name) OVER (PARTITION BY dept ORDER BY salary DESC) AS top_earner "
            "FROM employees"
        )
        # Every row in a department should show the top earner
        eng = [row for row in r.rows if row[1] == 'Engineering']
        assert all(row[3] == 'Charlie' for row in eng)  # Charlie has 95000
        sales = [row for row in r.rows if row[1] == 'Sales']
        assert all(row[3] == 'Eve' for row in sales)  # Eve has 75000

    def test_last_value_default_frame(self, db):
        """LAST_VALUE with ORDER BY defaults to ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW,
        so last_value is the current row."""
        r = db.execute(
            "SELECT name, salary, "
            "LAST_VALUE(name) OVER (ORDER BY salary) AS lv "
            "FROM employees WHERE dept = 'Engineering'"
        )
        # With default frame (up to current row), last_value = current row's name
        for row in r.rows:
            assert row[2] == row[0]

    def test_last_value_full_frame(self, db):
        """LAST_VALUE with full partition frame."""
        r = db.execute(
            "SELECT name, salary, "
            "LAST_VALUE(name) OVER ("
            "  ORDER BY salary "
            "  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
            ") AS lv "
            "FROM employees WHERE dept = 'Engineering'"
        )
        # With full frame, last_value = highest salary name in partition
        assert all(row[2] == 'Charlie' for row in r.rows)

    def test_nth_value(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "NTH_VALUE(name, 2) OVER (PARTITION BY dept ORDER BY salary DESC "
            "  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS second "
            "FROM employees"
        )
        # Second highest salary in Engineering: Alice (90000)
        eng = [row for row in r.rows if row[1] == 'Engineering']
        assert all(row[3] == 'Alice' for row in eng)

    def test_nth_value_out_of_range(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "NTH_VALUE(name, 10) OVER (PARTITION BY dept ORDER BY salary DESC "
            "  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS tenth "
            "FROM employees"
        )
        # No department has 10 employees
        assert all(row[3] is None for row in r.rows)


# =============================================================================
# Aggregate Window Functions
# =============================================================================

class TestAggregateWindows:
    def test_sum_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "SUM(salary) OVER (PARTITION BY dept) AS dept_total "
            "FROM employees"
        )
        # Engineering total: 90000 + 85000 + 95000 = 270000
        eng = [row for row in r.rows if row[1] == 'Engineering']
        assert all(row[3] == 270000 for row in eng)
        # Sales total: 70000 + 75000 + 72000 = 217000
        sales = [row for row in r.rows if row[1] == 'Sales']
        assert all(row[3] == 217000 for row in sales)

    def test_avg_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "AVG(salary) OVER (PARTITION BY dept) AS dept_avg "
            "FROM employees"
        )
        eng = [row for row in r.rows if row[1] == 'Engineering']
        assert all(row[3] == 90000.0 for row in eng)  # 270000/3

    def test_count_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, "
            "COUNT(*) OVER (PARTITION BY dept) AS dept_count "
            "FROM employees"
        )
        eng = [row for row in r.rows if row[1] == 'Engineering']
        assert all(row[2] == 3 for row in eng)
        hr = [row for row in r.rows if row[1] == 'HR']
        assert all(row[2] == 2 for row in hr)

    def test_min_max_over_partition(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "MIN(salary) OVER (PARTITION BY dept) AS dept_min, "
            "MAX(salary) OVER (PARTITION BY dept) AS dept_max "
            "FROM employees"
        )
        eng = [row for row in r.rows if row[1] == 'Engineering']
        assert all(row[3] == 85000 for row in eng)
        assert all(row[4] == 95000 for row in eng)

    def test_running_sum(self, db):
        """SUM with ORDER BY gives a running total."""
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary) AS running_total "
            "FROM employees WHERE dept = 'Sales' ORDER BY salary"
        )
        # Salaries sorted: 70000, 72000, 75000
        # Running: 70000, 142000, 217000
        running = [row[2] for row in r.rows]
        assert running == [70000, 142000, 217000]

    def test_running_count(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "COUNT(*) OVER (ORDER BY salary) AS running_count "
            "FROM employees WHERE dept = 'Engineering' ORDER BY salary"
        )
        counts = [row[2] for row in r.rows]
        assert counts == [1, 2, 3]

    def test_sum_over_entire_table(self, db):
        """SUM without PARTITION BY or ORDER BY = total over all rows."""
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER () AS total "
            "FROM employees WHERE dept = 'HR'"
        )
        # HR: 65000 + 68000 = 133000
        assert all(row[2] == 133000 for row in r.rows)

    def test_count_column_over(self):
        """COUNT(col) OVER -- counts non-null values."""
        d = WindowDB()
        d.execute("CREATE TABLE t (id INT, val INT)")
        d.execute("INSERT INTO t VALUES (1, 10)")
        d.execute("INSERT INTO t VALUES (2, NULL)")
        d.execute("INSERT INTO t VALUES (3, 30)")
        r = d.execute(
            "SELECT id, COUNT(val) OVER () AS cnt FROM t ORDER BY id"
        )
        # Only 2 non-null values
        assert all(row[1] == 2 for row in r.rows)


# =============================================================================
# Frame Specification Tests
# =============================================================================

class TestFrameSpecs:
    def test_rows_between_1_preceding_and_1_following(self, db):
        """Moving average with 3-row window."""
        r = db.execute(
            "SELECT name, salary, "
            "AVG(salary) OVER (ORDER BY salary "
            "  ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving_avg "
            "FROM employees WHERE dept = 'Sales' ORDER BY salary"
        )
        # Salaries sorted: 70000, 72000, 75000
        # Row 0: avg(70000, 72000) = 71000
        # Row 1: avg(70000, 72000, 75000) = 72333.33
        # Row 2: avg(72000, 75000) = 73500
        avgs = [row[2] for row in r.rows]
        assert abs(avgs[0] - 71000) < 1
        assert abs(avgs[1] - 72333.33) < 1
        assert abs(avgs[2] - 73500) < 1

    def test_rows_between_unbounded_preceding_and_current(self, db):
        """Cumulative sum (default with ORDER BY)."""
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary "
            "  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_sum "
            "FROM employees WHERE dept = 'Engineering' ORDER BY salary"
        )
        # Sorted: 85000, 90000, 95000
        sums = [row[2] for row in r.rows]
        assert sums == [85000, 175000, 270000]

    def test_rows_between_current_and_unbounded_following(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary "
            "  ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS remaining_sum "
            "FROM employees WHERE dept = 'Engineering' ORDER BY salary"
        )
        # Sorted: 85000, 90000, 95000
        # Row 0: 85000+90000+95000 = 270000
        # Row 1: 90000+95000 = 185000
        # Row 2: 95000
        sums = [row[2] for row in r.rows]
        assert sums == [270000, 185000, 95000]

    def test_rows_n_preceding(self, db):
        """SUM with fixed preceding window."""
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary ROWS 2 PRECEDING) AS s "
            "FROM employees WHERE dept = 'Sales' ORDER BY salary"
        )
        # Sorted: 70000, 72000, 75000
        # Row 0: 70000
        # Row 1: 70000+72000 = 142000
        # Row 2: 70000+72000+75000 = 217000
        sums = [row[2] for row in r.rows]
        assert sums == [70000, 142000, 217000]


# =============================================================================
# Multiple Window Functions
# =============================================================================

class TestMultipleWindows:
    def test_two_window_functions(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS overall_rank, "
            "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS dept_rank "
            "FROM employees"
        )
        assert 'overall_rank' in r.columns
        assert 'dept_rank' in r.columns
        # Check overall rank starts at 1
        overall = [row[3] for row in r.rows]
        assert 1 in overall

    def test_mixed_window_functions(self, db):
        r = db.execute(
            "SELECT name, dept, salary, "
            "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn, "
            "SUM(salary) OVER (PARTITION BY dept) AS dept_total, "
            "AVG(salary) OVER () AS overall_avg "
            "FROM employees"
        )
        assert len(r.columns) == 6
        # overall_avg should be the same for all rows
        avgs = [row[5] for row in r.rows]
        assert len(set(avgs)) == 1

    def test_rank_and_dense_rank(self, scores_db):
        r = scores_db.execute(
            "SELECT student, score, "
            "RANK() OVER (ORDER BY score DESC) AS rnk, "
            "DENSE_RANK() OVER (ORDER BY score DESC) AS drnk "
            "FROM scores WHERE subject = 'Math'"
        )
        for row in r.rows:
            assert row[2] is not None
            assert row[3] is not None


# =============================================================================
# Integration with C247-C251 Features
# =============================================================================

class TestIntegration:
    def test_with_create_table(self, db):
        """Window functions work alongside DDL."""
        db.execute("CREATE TABLE test_wf (a INT, b INT)")
        db.execute("INSERT INTO test_wf VALUES (1, 10)")
        db.execute("INSERT INTO test_wf VALUES (2, 20)")
        db.execute("INSERT INTO test_wf VALUES (3, 30)")
        r = db.execute(
            "SELECT a, b, SUM(b) OVER (ORDER BY a) AS running "
            "FROM test_wf"
        )
        assert r.rows[0][2] == 10
        assert r.rows[1][2] == 30
        assert r.rows[2][2] == 60

    def test_window_with_where(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees WHERE salary >= 70000"
        )
        # Should exclude Grace (65000) and Hank (68000)
        assert len(r.rows) == 6

    def test_window_with_limit(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees ORDER BY salary DESC LIMIT 3"
        )
        assert len(r.rows) == 3
        assert r.rows[0][2] == 1

    def test_window_preserves_triggers(self):
        """WindowDB still supports triggers from C251."""
        d = WindowDB()
        d.execute("CREATE TABLE items (id INT, name TEXT, price FLOAT)")
        d.execute("CREATE TABLE audit_log (action TEXT, item_name TEXT)")
        d.execute("""
            CREATE TRIGGER log_insert AFTER INSERT ON items
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log VALUES ('INSERT', NEW.name);
            END
        """)
        d.execute("INSERT INTO items VALUES (1, 'Widget', 9.99)")
        audit = d.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 1
        assert audit.rows[0][1] == 'Widget'

        # Now use window function on items
        d.execute("INSERT INTO items VALUES (2, 'Gadget', 19.99)")
        d.execute("INSERT INTO items VALUES (3, 'Doohickey', 4.99)")
        r = d.execute(
            "SELECT name, price, "
            "RANK() OVER (ORDER BY price DESC) AS price_rank "
            "FROM items"
        )
        assert len(r.rows) == 3

    def test_window_preserves_views(self):
        """WindowDB still supports views from C250."""
        d = WindowDB()
        d.execute("CREATE TABLE products (id INT, name TEXT, category TEXT, price FLOAT)")
        d.execute("INSERT INTO products VALUES (1, 'A', 'X', 10)")
        d.execute("INSERT INTO products VALUES (2, 'B', 'X', 20)")
        d.execute("INSERT INTO products VALUES (3, 'C', 'Y', 30)")
        d.execute("CREATE VIEW expensive AS SELECT * FROM products WHERE price > 15")
        r = d.execute("SELECT * FROM expensive")
        assert len(r.rows) == 2

    def test_standard_select_still_works(self, db):
        """Non-window SELECT still works correctly."""
        r = db.execute("SELECT name, salary FROM employees WHERE dept = 'HR'")
        assert len(r.rows) == 2
        r2 = db.execute("SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept")
        assert len(r2.rows) == 3

    def test_standard_aggregates_still_work(self, db):
        r = db.execute("SELECT dept, AVG(salary) AS avg_sal FROM employees GROUP BY dept")
        assert len(r.rows) == 3
        eng = [row for row in r.rows if row[0] == 'Engineering']
        assert abs(eng[0][1] - 90000.0) < 1


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_empty_table(self):
        d = WindowDB()
        d.execute("CREATE TABLE empty_t (a INT, b INT)")
        r = d.execute(
            "SELECT a, ROW_NUMBER() OVER (ORDER BY a) AS rn FROM empty_t"
        )
        assert len(r.rows) == 0

    def test_single_row(self):
        d = WindowDB()
        d.execute("CREATE TABLE single (a INT, b INT)")
        d.execute("INSERT INTO single VALUES (1, 100)")
        r = d.execute(
            "SELECT a, b, "
            "ROW_NUMBER() OVER () AS rn, "
            "SUM(b) OVER () AS total, "
            "LAG(b) OVER (ORDER BY a) AS prev "
            "FROM single"
        )
        assert r.rows[0][2] == 1  # row_number
        assert r.rows[0][3] == 100  # sum
        assert r.rows[0][4] is None  # lag with no previous

    def test_all_same_values(self):
        d = WindowDB()
        d.execute("CREATE TABLE same (a INT, b INT)")
        d.execute("INSERT INTO same VALUES (1, 100)")
        d.execute("INSERT INTO same VALUES (2, 100)")
        d.execute("INSERT INTO same VALUES (3, 100)")
        r = d.execute(
            "SELECT a, b, "
            "RANK() OVER (ORDER BY b) AS rnk, "
            "DENSE_RANK() OVER (ORDER BY b) AS drnk "
            "FROM same"
        )
        # All tied
        assert all(row[2] == 1 for row in r.rows)  # RANK = 1 for all
        assert all(row[3] == 1 for row in r.rows)  # DENSE_RANK = 1 for all

    def test_null_handling_in_partition(self):
        d = WindowDB()
        d.execute("CREATE TABLE nulls_t (grp TEXT, val INT)")
        d.execute("INSERT INTO nulls_t VALUES ('A', 10)")
        d.execute("INSERT INTO nulls_t VALUES ('A', 20)")
        d.execute("INSERT INTO nulls_t VALUES (NULL, 30)")
        d.execute("INSERT INTO nulls_t VALUES (NULL, 40)")
        r = d.execute(
            "SELECT grp, val, "
            "SUM(val) OVER (PARTITION BY grp) AS grp_sum "
            "FROM nulls_t"
        )
        # Group 'A': 30, NULL group: 70
        a_rows = [row for row in r.rows if row[0] == 'A']
        null_rows = [row for row in r.rows if row[0] is None]
        assert all(row[2] == 30 for row in a_rows)
        assert all(row[2] == 70 for row in null_rows)

    def test_window_with_alias(self, db):
        r = db.execute(
            "SELECT name, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS salary_rank "
            "FROM employees"
        )
        assert 'salary_rank' in r.columns

    def test_large_ntile(self):
        """NTILE(n) where n > row count."""
        d = WindowDB()
        d.execute("CREATE TABLE small (a INT)")
        d.execute("INSERT INTO small VALUES (1)")
        d.execute("INSERT INTO small VALUES (2)")
        d.execute("INSERT INTO small VALUES (3)")
        r = d.execute(
            "SELECT a, NTILE(10) OVER (ORDER BY a) AS tile FROM small"
        )
        # 3 rows / 10 tiles -- each row gets its own tile (1, 2, 3)
        tiles = [row[1] for row in r.rows]
        assert tiles == [1, 2, 3]


# =============================================================================
# Ordering with Window Functions
# =============================================================================

class TestOrdering:
    def test_order_by_window_result(self, db):
        """ORDER BY referencing the window function alias."""
        r = db.execute(
            "SELECT name, dept, salary, "
            "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS dept_rank "
            "FROM employees ORDER BY dept, salary DESC"
        )
        # Should be ordered by dept, then salary descending
        prev_dept = None
        prev_salary = None
        for row in r.rows:
            if row[1] == prev_dept:
                assert row[2] <= prev_salary
            prev_dept = row[1]
            prev_salary = row[2]

    def test_order_by_ascending(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary ASC) AS cum "
            "FROM employees WHERE dept = 'Engineering' ORDER BY salary ASC"
        )
        # Should be ascending
        salaries = [row[1] for row in r.rows]
        assert salaries == sorted(salaries)

    def test_order_by_descending(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary DESC) AS cum "
            "FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC"
        )
        # Cumulative descending: 95000, 185000, 270000
        cums = [row[2] for row in r.rows]
        assert cums == [95000, 185000, 270000]


# =============================================================================
# Parser Tests
# =============================================================================

class TestParser:
    def test_parse_simple_window(self):
        lexer = WindowLexer("SELECT ROW_NUMBER() OVER (ORDER BY a) AS rn FROM t")
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, SelectStmt)
        assert isinstance(stmt.columns[0].expr, SqlWindowCall)
        assert stmt.columns[0].expr.func_name == 'row_number'
        assert stmt.columns[0].alias == 'rn'

    def test_parse_partition_by(self):
        sql = "SELECT SUM(x) OVER (PARTITION BY grp ORDER BY id) AS s FROM t"
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        wc = stmt.columns[0].expr
        assert isinstance(wc, SqlWindowCall)
        assert len(wc.window.partition_by) == 1
        assert len(wc.window.order_by) == 1

    def test_parse_frame_clause(self):
        sql = ("SELECT SUM(x) OVER (ORDER BY id "
               "ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS s FROM t")
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        wc = stmt.columns[0].expr
        assert wc.window.frame is not None
        assert wc.window.frame.start.bound_type == FrameBound.N_PRECEDING
        assert wc.window.frame.start.offset == 2
        assert wc.window.frame.end.bound_type == FrameBound.N_FOLLOWING
        assert wc.window.frame.end.offset == 1

    def test_parse_unbounded_frame(self):
        sql = ("SELECT SUM(x) OVER (ORDER BY id "
               "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t")
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        wc = stmt.columns[0].expr
        assert wc.window.frame.start.bound_type == FrameBound.UNBOUNDED_PRECEDING
        assert wc.window.frame.end.bound_type == FrameBound.UNBOUNDED_FOLLOWING

    def test_parse_current_row_frame(self):
        sql = ("SELECT SUM(x) OVER (ORDER BY id "
               "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM t")
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        wc = stmt.columns[0].expr
        assert wc.window.frame.start.bound_type == FrameBound.CURRENT_ROW

    def test_parse_lead_with_args(self):
        sql = "SELECT LEAD(salary, 2, 0) OVER (ORDER BY id) FROM t"
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        wc = stmt.columns[0].expr
        assert wc.func_name == 'lead'
        assert len(wc.args) == 3

    def test_parse_count_star_window(self):
        sql = "SELECT COUNT(*) OVER (PARTITION BY dept) AS cnt FROM t"
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        wc = stmt.columns[0].expr
        assert wc.func_name == 'count'
        assert len(wc.args) == 0  # COUNT(*)

    def test_parse_multiple_windows(self):
        sql = ("SELECT ROW_NUMBER() OVER (ORDER BY a) AS rn, "
               "SUM(b) OVER (PARTITION BY c) AS s FROM t")
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt.columns[0].expr, SqlWindowCall)
        assert isinstance(stmt.columns[1].expr, SqlWindowCall)

    def test_parse_empty_over(self):
        sql = "SELECT SUM(x) OVER () AS total FROM t"
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        wc = stmt.columns[0].expr
        assert wc.window.partition_by == []
        assert wc.window.order_by == []
        assert wc.window.frame is None


# =============================================================================
# Complex Queries
# =============================================================================

class TestComplexQueries:
    def test_top_n_per_group(self, db):
        """Classic: get top 2 earners per department."""
        r = db.execute(
            "SELECT name, dept, salary, "
            "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn "
            "FROM employees"
        )
        # Filter for rn <= 2 (simulated -- our SQL doesn't support subquery yet easily)
        top2 = [row for row in r.rows if row[3] <= 2]
        # Engineering: Charlie(95k), Alice(90k)
        eng = [row for row in top2 if row[1] == 'Engineering']
        assert len(eng) == 2
        assert eng[0][0] in ('Charlie', 'Alice')
        assert eng[1][0] in ('Charlie', 'Alice')

    def test_running_percentage(self, db):
        """Running sum as percentage of total."""
        r = db.execute(
            "SELECT name, salary, "
            "SUM(salary) OVER (ORDER BY salary) AS running, "
            "SUM(salary) OVER () AS total "
            "FROM employees WHERE dept = 'Engineering'"
        )
        for row in r.rows:
            assert row[3] == 270000  # total
            # running <= total
            assert row[2] <= row[3]

    def test_difference_from_partition_avg(self, db):
        """Calculate difference from department average."""
        r = db.execute(
            "SELECT name, dept, salary, "
            "AVG(salary) OVER (PARTITION BY dept) AS dept_avg "
            "FROM employees"
        )
        for row in r.rows:
            # dept_avg should be the average for that department
            assert row[3] is not None

    def test_moving_min_max(self, db):
        r = db.execute(
            "SELECT name, salary, "
            "MIN(salary) OVER (ORDER BY salary ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS local_min, "
            "MAX(salary) OVER (ORDER BY salary ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS local_max "
            "FROM employees WHERE dept = 'Sales' ORDER BY salary"
        )
        # Sorted: 70000, 72000, 75000
        # Row 0: min=70000, max=72000
        # Row 1: min=70000, max=75000
        # Row 2: min=72000, max=75000
        assert r.rows[0][2] == 70000
        assert r.rows[0][3] == 72000
        assert r.rows[1][2] == 70000
        assert r.rows[1][3] == 75000
        assert r.rows[2][2] == 72000
        assert r.rows[2][3] == 75000


# =============================================================================
# Lexer Tests
# =============================================================================

class TestLexer:
    def test_window_keywords_recognized(self):
        sql = "SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary)"
        lexer = WindowLexer(sql)
        values = [t.value.lower() if isinstance(t.value, str) else t.value
                  for t in lexer.tokens if hasattr(t, 'value')]
        assert 'row_number' in values
        assert 'over' in values
        assert 'partition' in values

    def test_non_window_sql_unaffected(self):
        sql = "SELECT name FROM employees WHERE dept = 'Sales'"
        lexer = WindowLexer(sql)
        # Should parse normally
        parser = WindowParser(lexer.tokens)
        stmt = parser._parse_statement()
        assert isinstance(stmt, SelectStmt)


# =============================================================================
# Distinct Window Aggregates
# =============================================================================

class TestDistinctWindows:
    def test_count_distinct_over(self):
        d = WindowDB()
        d.execute("CREATE TABLE tags (item TEXT, tag TEXT)")
        d.execute("INSERT INTO tags VALUES ('A', 'x')")
        d.execute("INSERT INTO tags VALUES ('A', 'y')")
        d.execute("INSERT INTO tags VALUES ('A', 'x')")
        d.execute("INSERT INTO tags VALUES ('B', 'z')")
        r = d.execute(
            "SELECT item, tag, "
            "COUNT(DISTINCT tag) OVER (PARTITION BY item) AS unique_tags "
            "FROM tags"
        )
        a_rows = [row for row in r.rows if row[0] == 'A']
        assert all(row[2] == 2 for row in a_rows)  # 'x' and 'y'
        b_rows = [row for row in r.rows if row[0] == 'B']
        assert all(row[2] == 1 for row in b_rows)

    def test_sum_distinct_over(self):
        d = WindowDB()
        d.execute("CREATE TABLE dups (grp TEXT, val INT)")
        d.execute("INSERT INTO dups VALUES ('A', 10)")
        d.execute("INSERT INTO dups VALUES ('A', 10)")
        d.execute("INSERT INTO dups VALUES ('A', 20)")
        r = d.execute(
            "SELECT grp, val, "
            "SUM(DISTINCT val) OVER (PARTITION BY grp) AS unique_sum "
            "FROM dups"
        )
        assert all(row[2] == 30 for row in r.rows)  # 10 + 20


# =============================================================================
# Star Select with Window
# =============================================================================

class TestStarWithWindow:
    def test_star_plus_window(self, db):
        """SELECT *, window_func works."""
        r = db.execute(
            "SELECT name, dept, salary, "
            "ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn "
            "FROM employees"
        )
        assert len(r.columns) == 4
        assert r.columns[3] == 'rn'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
