"""
Tests for C269: SQL String Concatenation (|| operator)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C268_set_operations'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C267_common_table_expressions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C266_subqueries'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C265_builtin_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C264_window_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C263_ctas'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C262_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C261_foreign_keys'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from string_concat import StringConcatDB
from mini_database import ParseError, Lexer, TokenType


@pytest.fixture
def db():
    """Create a database with test data."""
    d = StringConcatDB()
    d.execute("CREATE TABLE users (id INT, first_name TEXT, last_name TEXT, age INT, city TEXT)")
    d.execute("INSERT INTO users VALUES (1, 'Alice', 'Smith', 30, 'NYC')")
    d.execute("INSERT INTO users VALUES (2, 'Bob', 'Jones', 25, 'LA')")
    d.execute("INSERT INTO users VALUES (3, 'Carol', 'Davis', 35, 'NYC')")
    d.execute("INSERT INTO users VALUES (4, 'Dave', 'Wilson', 28, 'Chicago')")
    d.execute("INSERT INTO users VALUES (5, 'Eve', 'Brown', 32, 'LA')")

    d.execute("CREATE TABLE products (id INT, name TEXT, category TEXT, price INT)")
    d.execute("INSERT INTO products VALUES (1, 'Widget', 'Tools', 10)")
    d.execute("INSERT INTO products VALUES (2, 'Gadget', 'Electronics', 25)")
    d.execute("INSERT INTO products VALUES (3, 'Doohickey', 'Tools', 15)")
    d.execute("INSERT INTO products VALUES (4, 'Thingamajig', 'Electronics', 30)")

    d.execute("CREATE TABLE orders (id INT, user_id INT, product_id INT, qty INT)")
    d.execute("INSERT INTO orders VALUES (1, 1, 1, 3)")
    d.execute("INSERT INTO orders VALUES (2, 1, 2, 1)")
    d.execute("INSERT INTO orders VALUES (3, 2, 3, 2)")
    d.execute("INSERT INTO orders VALUES (4, 3, 1, 5)")
    d.execute("INSERT INTO orders VALUES (5, 4, 4, 1)")

    return d


@pytest.fixture
def nulldb():
    """Database with NULL values for NULL propagation tests."""
    d = StringConcatDB()
    d.execute("CREATE TABLE nullable (id INT, a TEXT, b TEXT)")
    d.execute("INSERT INTO nullable VALUES (1, 'hello', 'world')")
    d.execute("INSERT INTO nullable VALUES (2, NULL, 'world')")
    d.execute("INSERT INTO nullable VALUES (3, 'hello', NULL)")
    d.execute("INSERT INTO nullable VALUES (4, NULL, NULL)")
    d.execute("INSERT INTO nullable VALUES (5, '', 'empty')")
    return d


# =====================================================================
# Lexer Tests
# =====================================================================

class TestLexer:
    """Test that || is properly tokenized."""

    def test_concat_token(self):
        lexer = Lexer("SELECT 'a' || 'b'")
        tokens = lexer.tokens
        types = [t.type for t in tokens]
        assert TokenType.CONCAT in types

    def test_concat_token_value(self):
        lexer = Lexer("SELECT 'a' || 'b'")
        tokens = lexer.tokens
        concat_tok = [t for t in tokens if t.type == TokenType.CONCAT][0]
        assert concat_tok.value == '||'

    def test_concat_multiple(self):
        lexer = Lexer("SELECT 'a' || 'b' || 'c'")
        tokens = lexer.tokens
        concat_count = sum(1 for t in tokens if t.type == TokenType.CONCAT)
        assert concat_count == 2

    def test_concat_no_spaces(self):
        lexer = Lexer("SELECT 'a'||'b'")
        tokens = lexer.tokens
        types = [t.type for t in tokens]
        assert TokenType.CONCAT in types

    def test_concat_with_columns(self):
        lexer = Lexer("SELECT first_name || last_name FROM t")
        tokens = lexer.tokens
        types = [t.type for t in tokens]
        assert TokenType.CONCAT in types


# =====================================================================
# Basic Concatenation
# =====================================================================

class TestBasicConcat:
    """Basic string concatenation with ||."""

    def test_two_string_literals(self, db):
        r = db.execute("SELECT 'hello' || ' world' AS greeting")
        assert r.rows[0][0] == 'hello world'

    def test_column_concat(self, db):
        r = db.execute("SELECT first_name || ' ' || last_name AS full_name FROM users WHERE id = 1")
        assert r.rows[0][0] == 'Alice Smith'

    def test_all_rows(self, db):
        r = db.execute("SELECT first_name || ' ' || last_name AS full_name FROM users ORDER BY id")
        names = [row[0] for row in r.rows]
        assert names == ['Alice Smith', 'Bob Jones', 'Carol Davis', 'Dave Wilson', 'Eve Brown']

    def test_number_to_string(self, db):
        r = db.execute("SELECT 'Age: ' || age AS info FROM users WHERE id = 1")
        assert r.rows[0][0] == 'Age: 30'

    def test_number_concat_number(self, db):
        r = db.execute("SELECT 1 || 2 AS result")
        assert r.rows[0][0] == '12'

    def test_empty_string_concat(self, db):
        r = db.execute("SELECT '' || 'hello' AS result")
        assert r.rows[0][0] == 'hello'

    def test_empty_both_sides(self, db):
        r = db.execute("SELECT '' || '' AS result")
        assert r.rows[0][0] == ''

    def test_three_way_concat(self, db):
        r = db.execute("SELECT first_name || ' from ' || city AS info FROM users WHERE id = 1")
        assert r.rows[0][0] == 'Alice from NYC'

    def test_four_way_concat(self, db):
        r = db.execute("SELECT first_name || ' ' || last_name || ' (' || city || ')' AS info FROM users WHERE id = 1")
        assert r.rows[0][0] == 'Alice Smith (NYC)'

    def test_concat_with_alias(self, db):
        r = db.execute("SELECT first_name || last_name AS combined FROM users WHERE id = 1")
        assert r.columns == ['combined']
        assert r.rows[0][0] == 'AliceSmith'


# =====================================================================
# NULL Propagation
# =====================================================================

class TestNullPropagation:
    """NULL || anything = NULL (SQL standard)."""

    def test_null_left(self, nulldb):
        r = nulldb.execute("SELECT a || b AS result FROM nullable WHERE id = 2")
        assert r.rows[0][0] is None

    def test_null_right(self, nulldb):
        r = nulldb.execute("SELECT a || b AS result FROM nullable WHERE id = 3")
        assert r.rows[0][0] is None

    def test_both_null(self, nulldb):
        r = nulldb.execute("SELECT a || b AS result FROM nullable WHERE id = 4")
        assert r.rows[0][0] is None

    def test_no_null(self, nulldb):
        r = nulldb.execute("SELECT a || b AS result FROM nullable WHERE id = 1")
        assert r.rows[0][0] == 'helloworld'

    def test_empty_not_null(self, nulldb):
        r = nulldb.execute("SELECT a || b AS result FROM nullable WHERE id = 5")
        assert r.rows[0][0] == 'empty'

    def test_null_literal_left(self, db):
        r = db.execute("SELECT NULL || 'hello' AS result")
        assert r.rows[0][0] is None

    def test_null_literal_right(self, db):
        r = db.execute("SELECT 'hello' || NULL AS result")
        assert r.rows[0][0] is None

    def test_null_chain(self, db):
        r = db.execute("SELECT 'a' || NULL || 'b' AS result")
        assert r.rows[0][0] is None


# =====================================================================
# WHERE Clause
# =====================================================================

class TestWhereConcat:
    """Using || in WHERE conditions."""

    def test_where_concat_equals(self, db):
        r = db.execute("SELECT id FROM users WHERE first_name || ' ' || last_name = 'Alice Smith'")
        assert r.rows[0][0] == 1

    def test_where_concat_like(self, db):
        r = db.execute("SELECT id FROM users WHERE first_name || last_name LIKE '%lice%'")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_where_concat_comparison(self, db):
        r = db.execute("SELECT id FROM users WHERE first_name || last_name > 'D' ORDER BY id")
        ids = [row[0] for row in r.rows]
        assert 4 in ids  # DaveWilson > D
        assert 5 in ids  # EveBrown > D

    def test_where_concat_in(self, db):
        r = db.execute("SELECT id FROM users WHERE first_name || ' ' || last_name IN ('Alice Smith', 'Bob Jones') ORDER BY id")
        ids = [row[0] for row in r.rows]
        assert ids == [1, 2]


# =====================================================================
# Operator Precedence
# =====================================================================

class TestPrecedence:
    """|| has lower precedence than arithmetic, higher than comparison."""

    def test_concat_lower_than_addition(self, db):
        # 'Value: ' || (1 + 2) should be 'Value: 3'
        r = db.execute("SELECT 'Value: ' || 1 + 2 AS result")
        assert r.rows[0][0] == 'Value: 3'

    def test_concat_lower_than_multiplication(self, db):
        # 'Product: ' || (3 * 4) should be 'Product: 12'
        r = db.execute("SELECT 'Product: ' || 3 * 4 AS result")
        assert r.rows[0][0] == 'Product: 12'

    def test_concat_higher_than_comparison(self, db):
        # first_name || last_name = 'AliceSmith' should parse as (first_name || last_name) = 'AliceSmith'
        r = db.execute("SELECT id FROM users WHERE first_name || last_name = 'AliceSmith'")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_concat_higher_than_and(self, db):
        r = db.execute("SELECT id FROM users WHERE first_name || ' ' || last_name = 'Alice Smith' AND age = 30")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_left_associativity(self, db):
        # 'a' || 'b' || 'c' should be ('a' || 'b') || 'c' = 'abc'
        r = db.execute("SELECT 'a' || 'b' || 'c' AS result")
        assert r.rows[0][0] == 'abc'

    def test_parentheses_override(self, db):
        # ('hello' || ' ') || 'world' vs 'hello' || (' ' || 'world')
        # Both should give same result since concat is associative
        r1 = db.execute("SELECT ('hello' || ' ') || 'world' AS result")
        r2 = db.execute("SELECT 'hello' || (' ' || 'world') AS result")
        assert r1.rows[0][0] == 'hello world'
        assert r2.rows[0][0] == 'hello world'

    def test_concat_with_arithmetic_in_column(self, db):
        # name || ': $' || (price * 2)
        r = db.execute("SELECT name || ': $' || price * 2 AS info FROM products WHERE id = 1")
        assert r.rows[0][0] == 'Widget: $20'

    def test_mixed_arithmetic_and_concat(self, db):
        r = db.execute("SELECT 'Age+5: ' || age + 5 AS result FROM users WHERE id = 1")
        assert r.rows[0][0] == 'Age+5: 35'


# =====================================================================
# ORDER BY with Concatenation
# =====================================================================

class TestOrderBy:
    """Sorting by concatenated expressions."""

    def test_order_by_concat(self, db):
        r = db.execute("SELECT first_name || ' ' || last_name AS name FROM users ORDER BY first_name || ' ' || last_name")
        names = [row[0] for row in r.rows]
        assert names == sorted(names)

    def test_order_by_concat_desc(self, db):
        r = db.execute("SELECT first_name || ' ' || last_name AS name FROM users ORDER BY first_name || ' ' || last_name DESC")
        names = [row[0] for row in r.rows]
        assert names == sorted(names, reverse=True)

    def test_order_by_alias(self, db):
        r = db.execute("SELECT first_name || ' ' || last_name AS full_name FROM users ORDER BY full_name")
        names = [row[0] for row in r.rows]
        assert names == sorted(names)


# =====================================================================
# GROUP BY with Concatenation
# =====================================================================

class TestGroupBy:
    """Grouping by concatenated expressions."""

    def test_group_by_concat(self, db):
        r = db.execute("SELECT city || '-group' AS grp, COUNT(*) AS cnt FROM users GROUP BY city || '-group' ORDER BY grp")
        rows = [(row[0], row[1]) for row in r.rows]
        assert ('Chicago-group', 1) in rows
        assert ('LA-group', 2) in rows
        assert ('NYC-group', 2) in rows

    def test_having_with_concat(self, db):
        r = db.execute("""
            SELECT city, 'Count: ' || COUNT(*) AS info
            FROM users
            GROUP BY city
            HAVING COUNT(*) > 1
            ORDER BY city
        """)
        assert len(r.rows) == 2  # LA and NYC have count > 1
        for row in r.rows:
            assert row[1].startswith('Count: ')


# =====================================================================
# CASE Expression
# =====================================================================

class TestCaseConcat:
    """Concatenation inside CASE expressions."""

    def test_concat_in_case_result(self, db):
        r = db.execute("""
            SELECT CASE WHEN age > 30 THEN 'Senior: ' || first_name
                        ELSE 'Junior: ' || first_name END AS label
            FROM users WHERE id = 1
        """)
        assert r.rows[0][0] == 'Junior: Alice'

    def test_concat_in_case_result_senior(self, db):
        r = db.execute("""
            SELECT CASE WHEN age > 30 THEN 'Senior: ' || first_name
                        ELSE 'Junior: ' || first_name END AS label
            FROM users WHERE id = 3
        """)
        assert r.rows[0][0] == 'Senior: Carol'

    def test_case_with_concat_condition(self, db):
        r = db.execute("""
            SELECT id FROM users
            WHERE CASE WHEN first_name || last_name = 'AliceSmith' THEN 1 ELSE 0 END = 1
        """)
        assert r.rows[0][0] == 1


# =====================================================================
# JOIN with Concatenation
# =====================================================================

class TestJoinConcat:
    """Concatenation in JOIN queries."""

    def test_concat_across_join(self, db):
        r = db.execute("""
            SELECT u.first_name || ' ordered ' || p.name AS info
            FROM users u
            JOIN orders o ON u.id = o.user_id
            JOIN products p ON p.id = o.product_id
            WHERE u.id = 1
            ORDER BY p.name
        """)
        infos = [row[0] for row in r.rows]
        assert 'Alice ordered Gadget' in infos
        assert 'Alice ordered Widget' in infos

    def test_concat_in_join_condition_equivalent(self, db):
        # Using concat in WHERE across a join
        r = db.execute("""
            SELECT u.first_name || ': ' || p.name AS info
            FROM users u
            JOIN orders o ON u.id = o.user_id
            JOIN products p ON p.id = o.product_id
            WHERE u.first_name || ': ' || p.name = 'Alice: Widget'
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 'Alice: Widget'


# =====================================================================
# Subqueries with Concatenation
# =====================================================================

class TestSubqueryConcat:
    """Concatenation in subqueries."""

    def test_concat_in_subquery_select(self, db):
        r = db.execute("""
            SELECT full_name FROM (
                SELECT first_name || ' ' || last_name AS full_name FROM users
            ) sub
            ORDER BY full_name
        """)
        names = [row[0] for row in r.rows]
        assert names[0] == 'Alice Smith'

    def test_concat_in_where_subquery(self, db):
        r = db.execute("""
            SELECT first_name FROM users
            WHERE first_name || ' ' || last_name IN (
                SELECT first_name || ' ' || last_name FROM users WHERE city = 'NYC'
            )
            ORDER BY first_name
        """)
        names = [row[0] for row in r.rows]
        assert 'Alice' in names
        assert 'Carol' in names

    def test_scalar_subquery_concat(self, db):
        r = db.execute("""
            SELECT 'Best: ' || (SELECT first_name FROM users WHERE id = 1) AS label
        """)
        assert r.rows[0][0] == 'Best: Alice'


# =====================================================================
# CTE with Concatenation
# =====================================================================

class TestCTEConcat:
    """Concatenation in Common Table Expressions."""

    def test_concat_in_cte(self, db):
        r = db.execute("""
            WITH names AS (
                SELECT first_name || ' ' || last_name AS full_name, city
                FROM users
            )
            SELECT full_name FROM names WHERE city = 'NYC' ORDER BY full_name
        """)
        names = [row[0] for row in r.rows]
        assert names == ['Alice Smith', 'Carol Davis']

    def test_concat_in_cte_and_main(self, db):
        r = db.execute("""
            WITH info AS (
                SELECT id, first_name || ' ' || last_name AS name FROM users
            )
            SELECT name || ' (ID: ' || id || ')' AS label FROM info WHERE id <= 2 ORDER BY id
        """)
        assert r.rows[0][0] == 'Alice Smith (ID: 1)'
        assert r.rows[1][0] == 'Bob Jones (ID: 2)'


# =====================================================================
# UNION with Concatenation
# =====================================================================

class TestUnionConcat:
    """Concatenation with set operations."""

    def test_concat_in_union(self, db):
        r = db.execute("""
            SELECT 'User: ' || first_name AS label FROM users WHERE id = 1
            UNION ALL
            SELECT 'Product: ' || name AS label FROM products WHERE id = 1
        """)
        labels = sorted([row[0] for row in r.rows])
        assert labels == ['Product: Widget', 'User: Alice']

    def test_concat_in_intersect(self, db):
        # Users in NYC with concat labels
        r = db.execute("""
            SELECT first_name || '-' || city AS tag FROM users WHERE city = 'NYC'
            INTERSECT
            SELECT first_name || '-' || city AS tag FROM users WHERE age > 28
        """)
        tags = [row[0] for row in r.rows]
        # Alice (30) and Carol (35) are in NYC and > 28
        assert 'Alice-NYC' in tags
        assert 'Carol-NYC' in tags


# =====================================================================
# Built-in Functions with Concatenation
# =====================================================================

class TestBuiltinFuncConcat:
    """Concatenation combined with built-in functions."""

    def test_concat_with_upper(self, db):
        r = db.execute("SELECT UPPER(first_name) || ' ' || UPPER(last_name) AS name FROM users WHERE id = 1")
        assert r.rows[0][0] == 'ALICE SMITH'

    def test_concat_with_lower(self, db):
        r = db.execute("SELECT LOWER(first_name || last_name) AS name FROM users WHERE id = 1")
        assert r.rows[0][0] == 'alicesmith'

    def test_concat_with_length(self, db):
        r = db.execute("SELECT first_name || ' (' || LENGTH(first_name) || ' chars)' AS info FROM users WHERE id = 1")
        assert r.rows[0][0] == 'Alice (5 chars)'

    def test_concat_with_coalesce(self, nulldb):
        r = nulldb.execute("SELECT COALESCE(a, 'N/A') || ' ' || COALESCE(b, 'N/A') AS result FROM nullable ORDER BY id")
        results = [row[0] for row in r.rows]
        assert results[0] == 'hello world'
        assert results[1] == 'N/A world'
        assert results[2] == 'hello N/A'
        assert results[3] == 'N/A N/A'
        assert results[4] == ' empty'

    def test_concat_vs_concat_function(self, db):
        # || operator and CONCAT() function should produce the same result
        r1 = db.execute("SELECT first_name || ' ' || last_name AS name FROM users WHERE id = 1")
        r2 = db.execute("SELECT CONCAT(first_name, ' ', last_name) AS name FROM users WHERE id = 1")
        assert r1.rows[0][0] == r2.rows[0][0]


# =====================================================================
# Type Coercion
# =====================================================================

class TestTypeCoercion:
    """Non-string operands are converted to strings."""

    def test_int_concat(self, db):
        r = db.execute("SELECT 42 || ' items' AS result")
        assert r.rows[0][0] == '42 items'

    def test_float_concat(self, db):
        r = db.execute("SELECT 3.14 || ' pi' AS result")
        assert '3.14' in r.rows[0][0]

    def test_bool_true_concat(self, db):
        r = db.execute("SELECT TRUE || ' value' AS result")
        # True converts to string
        assert r.rows[0][0] is not None

    def test_bool_false_concat(self, db):
        r = db.execute("SELECT FALSE || ' value' AS result")
        assert r.rows[0][0] is not None

    def test_column_int_concat(self, db):
        r = db.execute("SELECT name || ': $' || price AS info FROM products WHERE id = 1")
        assert r.rows[0][0] == 'Widget: $10'

    def test_arithmetic_result_concat(self, db):
        r = db.execute("SELECT name || ' x' || qty || ' = $' || price * qty AS info FROM products p JOIN orders o ON p.id = o.product_id WHERE o.id = 1")
        assert r.rows[0][0] == 'Widget x3 = $30'


# =====================================================================
# Edge Cases
# =====================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_char_concat(self, db):
        r = db.execute("SELECT 'a' || 'b' AS result")
        assert r.rows[0][0] == 'ab'

    def test_long_chain(self, db):
        r = db.execute("SELECT 'a' || 'b' || 'c' || 'd' || 'e' || 'f' AS result")
        assert r.rows[0][0] == 'abcdef'

    def test_concat_in_insert_select(self, db):
        db.execute("CREATE TABLE labels (label TEXT)")
        db.execute("INSERT INTO labels SELECT first_name || ' ' || last_name FROM users WHERE id = 1")
        r = db.execute("SELECT label FROM labels")
        assert r.rows[0][0] == 'Alice Smith'

    def test_concat_with_like_rhs(self, db):
        # Concat on the right side of LIKE
        r = db.execute("SELECT id FROM users WHERE first_name LIKE 'A' || '%'")
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_concat_with_not_like_rhs(self, db):
        r = db.execute("SELECT id FROM users WHERE first_name NOT LIKE 'A' || '%' ORDER BY id")
        ids = [row[0] for row in r.rows]
        assert 1 not in ids
        assert len(ids) == 4

    def test_concat_both_sides_of_comparison(self, db):
        r = db.execute("""
            SELECT id FROM users
            WHERE first_name || last_name = 'Alice' || 'Smith'
        """)
        assert len(r.rows) == 1
        assert r.rows[0][0] == 1

    def test_concat_in_distinct(self, db):
        db.execute("INSERT INTO users VALUES (6, 'Alice', 'Smith', 40, 'Boston')")
        r = db.execute("SELECT DISTINCT first_name || ' ' || last_name AS name FROM users ORDER BY name")
        names = [row[0] for row in r.rows]
        assert names.count('Alice Smith') == 1

    def test_concat_in_limit(self, db):
        r = db.execute("SELECT first_name || ' ' || last_name AS name FROM users ORDER BY name LIMIT 2")
        assert len(r.rows) == 2

    def test_concat_in_update_where(self, db):
        db.execute("UPDATE users SET city = 'SF' WHERE first_name || last_name = 'AliceSmith'")
        r = db.execute("SELECT city FROM users WHERE id = 1")
        assert r.rows[0][0] == 'SF'

    def test_concat_in_delete_where(self, db):
        db.execute("DELETE FROM users WHERE first_name || ' ' || last_name = 'Eve Brown'")
        r = db.execute("SELECT COUNT(*) AS cnt FROM users")
        assert r.rows[0][0] == 4


# =====================================================================
# Aggregate with Concatenation
# =====================================================================

class TestAggregateConcat:
    """Concatenation with aggregate functions."""

    def test_concat_with_count(self, db):
        r = db.execute("SELECT city || ': ' || COUNT(*) AS info FROM users GROUP BY city ORDER BY city")
        infos = [row[0] for row in r.rows]
        assert 'Chicago: 1' in infos
        assert 'LA: 2' in infos
        assert 'NYC: 2' in infos

    def test_concat_with_sum(self, db):
        r = db.execute("SELECT city || ' total age: ' || SUM(age) AS info FROM users GROUP BY city ORDER BY city")
        infos = [row[0] for row in r.rows]
        assert 'Chicago total age: 28' in infos

    def test_concat_with_min_max(self, db):
        r = db.execute("SELECT city || ': ' || MIN(age) || '-' || MAX(age) AS range FROM users GROUP BY city ORDER BY city")
        ranges = [row[0] for row in r.rows]
        assert 'NYC: 30-35' in ranges
        assert 'LA: 25-32' in ranges


# =====================================================================
# Complex Queries
# =====================================================================

class TestComplexQueries:
    """Complex queries combining multiple features."""

    def test_cte_union_concat(self, db):
        r = db.execute("""
            WITH all_people AS (
                SELECT first_name || ' ' || last_name AS name, 'employee' AS type FROM users
                UNION ALL
                SELECT name || ' (product)' AS name, 'product' AS type FROM products
            )
            SELECT name FROM all_people WHERE type = 'employee' ORDER BY name LIMIT 3
        """)
        names = [row[0] for row in r.rows]
        assert len(names) == 3
        assert names[0] == 'Alice Smith'

    def test_nested_subquery_concat(self, db):
        r = db.execute("""
            SELECT label FROM (
                SELECT first_name || ': ' || city AS label
                FROM users
                WHERE city IN (SELECT DISTINCT city FROM users WHERE age > 30)
            ) sub
            ORDER BY label
        """)
        labels = [row[0] for row in r.rows]
        assert 'Alice: NYC' in labels
        assert 'Carol: NYC' in labels

    def test_concat_in_having_filter(self, db):
        r = db.execute("""
            SELECT city, MIN(first_name) || ' to ' || MAX(first_name) AS name_range
            FROM users
            GROUP BY city
            HAVING COUNT(*) > 1
            ORDER BY city
        """)
        assert len(r.rows) == 2  # LA and NYC
        for row in r.rows:
            assert ' to ' in row[1]

    def test_multiple_concat_expressions(self, db):
        r = db.execute("""
            SELECT
                first_name || ' ' || last_name AS full_name,
                city || ', ' || 'USA' AS location,
                'Age: ' || age AS age_info
            FROM users
            WHERE id = 1
        """)
        assert r.rows[0][0] == 'Alice Smith'
        assert r.rows[0][1] == 'NYC, USA'
        assert r.rows[0][2] == 'Age: 30'
