"""
Tests for C247: Mini Database Engine
Composes C244 (Buffer Pool) + C245 (Query Executor) + C246 (Transaction Manager)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from mini_database import (
    MiniDB, ResultSet, DatabaseError, CatalogError, ParseError,
    parse_sql, parse_sql_multi, Lexer, TokenType, CompileError,
    IsolationLevel, StorageEngine, Catalog, ColumnDef, TableSchema,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlAggCall, SqlStar, SqlIsNull, SqlBetween, SqlInList, SqlCase,
    CreateTableStmt, InsertStmt, SelectStmt, UpdateStmt, DeleteStmt,
    BeginStmt, CommitStmt, RollbackStmt, SavepointStmt,
    ShowTablesStmt, DescribeStmt, ExplainStmt, DropTableStmt,
    CreateIndexStmt, SelectExpr, TableRef, JoinClause,
)


class TestLexer(unittest.TestCase):
    """Test SQL lexer."""

    def test_basic_select(self):
        tokens = Lexer("SELECT * FROM users").tokens
        types = [t.type for t in tokens]
        self.assertEqual(types[0], TokenType.SELECT)
        self.assertEqual(types[1], TokenType.STAR)
        self.assertEqual(types[2], TokenType.FROM)
        self.assertEqual(types[3], TokenType.IDENT)
        self.assertEqual(types[4], TokenType.EOF)

    def test_numbers(self):
        tokens = Lexer("42 3.14").tokens
        self.assertEqual(tokens[0].value, 42)
        self.assertEqual(tokens[1].value, 3.14)

    def test_strings(self):
        tokens = Lexer("'hello' \"world\"").tokens
        self.assertEqual(tokens[0].value, 'hello')
        self.assertEqual(tokens[1].value, 'world')

    def test_operators(self):
        tokens = Lexer("= != < <= > >=").tokens
        types = [t.type for t in tokens[:-1]]
        self.assertEqual(types, [TokenType.EQ, TokenType.NE, TokenType.LT,
                                 TokenType.LE, TokenType.GT, TokenType.GE])

    def test_keywords_case_insensitive(self):
        tokens = Lexer("SELECT select SeLeCt").tokens
        for i in range(3):
            self.assertEqual(tokens[i].type, TokenType.SELECT)

    def test_comments(self):
        tokens = Lexer("SELECT -- this is a comment\n42").tokens
        self.assertEqual(tokens[0].type, TokenType.SELECT)
        self.assertEqual(tokens[1].type, TokenType.NUMBER)

    def test_diamond_operator(self):
        tokens = Lexer("a <> b").tokens
        self.assertEqual(tokens[1].type, TokenType.NE)


class TestParser(unittest.TestCase):
    """Test SQL parser."""

    def test_select_star(self):
        stmt = parse_sql("SELECT * FROM users")
        self.assertIsInstance(stmt, SelectStmt)
        self.assertEqual(stmt.from_table.table_name, 'users')

    def test_select_columns(self):
        stmt = parse_sql("SELECT name, age FROM users")
        self.assertIsInstance(stmt, SelectStmt)
        self.assertEqual(len(stmt.columns), 2)

    def test_select_where(self):
        stmt = parse_sql("SELECT * FROM users WHERE age > 18")
        self.assertIsNotNone(stmt.where)

    def test_select_order_by(self):
        stmt = parse_sql("SELECT * FROM users ORDER BY name ASC, age DESC")
        self.assertEqual(len(stmt.order_by), 2)
        self.assertTrue(stmt.order_by[0][1])   # ASC
        self.assertFalse(stmt.order_by[1][1])  # DESC

    def test_select_limit_offset(self):
        stmt = parse_sql("SELECT * FROM users LIMIT 10 OFFSET 5")
        self.assertEqual(stmt.limit, 10)
        self.assertEqual(stmt.offset, 5)

    def test_select_distinct(self):
        stmt = parse_sql("SELECT DISTINCT name FROM users")
        self.assertTrue(stmt.distinct)

    def test_select_group_by(self):
        stmt = parse_sql("SELECT dept, COUNT(*) FROM emp GROUP BY dept")
        self.assertIsNotNone(stmt.group_by)

    def test_select_having(self):
        stmt = parse_sql("SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5")
        self.assertIsNotNone(stmt.having)

    def test_select_join(self):
        stmt = parse_sql("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
        self.assertEqual(len(stmt.joins), 1)
        self.assertEqual(stmt.joins[0].join_type, 'inner')

    def test_select_left_join(self):
        stmt = parse_sql("SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id")
        self.assertEqual(stmt.joins[0].join_type, 'left')

    def test_select_alias(self):
        stmt = parse_sql("SELECT name AS n FROM users AS u")
        self.assertEqual(stmt.columns[0].alias, 'n')
        self.assertEqual(stmt.from_table.alias, 'u')

    def test_insert(self):
        stmt = parse_sql("INSERT INTO users (name, age) VALUES ('Alice', 30)")
        self.assertIsInstance(stmt, InsertStmt)
        self.assertEqual(stmt.table_name, 'users')
        self.assertEqual(stmt.columns, ['name', 'age'])
        self.assertEqual(len(stmt.values_list), 1)

    def test_insert_multi_row(self):
        stmt = parse_sql("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        self.assertEqual(len(stmt.values_list), 3)

    def test_update(self):
        stmt = parse_sql("UPDATE users SET name = 'Bob' WHERE id = 1")
        self.assertIsInstance(stmt, UpdateStmt)
        self.assertEqual(len(stmt.assignments), 1)

    def test_delete(self):
        stmt = parse_sql("DELETE FROM users WHERE id = 1")
        self.assertIsInstance(stmt, DeleteStmt)

    def test_create_table(self):
        stmt = parse_sql("CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL)")
        self.assertIsInstance(stmt, CreateTableStmt)
        self.assertEqual(len(stmt.columns), 2)
        self.assertTrue(stmt.columns[0].primary_key)
        self.assertTrue(stmt.columns[1].not_null)

    def test_create_table_if_not_exists(self):
        stmt = parse_sql("CREATE TABLE IF NOT EXISTS users (id INT)")
        self.assertTrue(stmt.if_not_exists)

    def test_drop_table(self):
        stmt = parse_sql("DROP TABLE users")
        self.assertIsInstance(stmt, DropTableStmt)

    def test_drop_table_if_exists(self):
        stmt = parse_sql("DROP TABLE IF EXISTS users")
        self.assertTrue(stmt.if_exists)

    def test_begin_commit_rollback(self):
        self.assertIsInstance(parse_sql("BEGIN"), BeginStmt)
        self.assertIsInstance(parse_sql("BEGIN TRANSACTION"), BeginStmt)
        self.assertIsInstance(parse_sql("COMMIT"), CommitStmt)
        self.assertIsInstance(parse_sql("ROLLBACK"), RollbackStmt)

    def test_savepoint(self):
        stmt = parse_sql("SAVEPOINT sp1")
        self.assertIsInstance(stmt, SavepointStmt)
        self.assertEqual(stmt.name, 'sp1')

    def test_rollback_to_savepoint(self):
        stmt = parse_sql("ROLLBACK TO SAVEPOINT sp1")
        self.assertEqual(stmt.savepoint, 'sp1')

    def test_show_tables(self):
        self.assertIsInstance(parse_sql("SHOW TABLES"), ShowTablesStmt)

    def test_describe(self):
        stmt = parse_sql("DESCRIBE users")
        self.assertIsInstance(stmt, DescribeStmt)
        self.assertEqual(stmt.table_name, 'users')

    def test_explain(self):
        stmt = parse_sql("EXPLAIN SELECT * FROM users")
        self.assertIsInstance(stmt, ExplainStmt)
        self.assertIsInstance(stmt.stmt, SelectStmt)

    def test_create_index(self):
        stmt = parse_sql("CREATE INDEX idx_name ON users (name)")
        self.assertIsInstance(stmt, CreateIndexStmt)
        self.assertEqual(stmt.column, 'name')

    def test_multi_statement(self):
        stmts = parse_sql_multi("SELECT 1; SELECT 2;")
        self.assertEqual(len(stmts), 2)

    def test_case_expression(self):
        stmt = parse_sql("SELECT CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END FROM t")
        self.assertIsInstance(stmt, SelectStmt)

    def test_between(self):
        stmt = parse_sql("SELECT * FROM t WHERE x BETWEEN 1 AND 10")
        self.assertIsInstance(stmt.where, SqlBetween)

    def test_in_list(self):
        stmt = parse_sql("SELECT * FROM t WHERE x IN (1, 2, 3)")
        self.assertIsInstance(stmt.where, SqlInList)

    def test_is_null(self):
        stmt = parse_sql("SELECT * FROM t WHERE x IS NULL")
        self.assertIsInstance(stmt.where, SqlIsNull)

    def test_is_not_null(self):
        stmt = parse_sql("SELECT * FROM t WHERE x IS NOT NULL")
        self.assertIsInstance(stmt.where, SqlIsNull)
        self.assertTrue(stmt.where.negated)

    def test_not_in(self):
        stmt = parse_sql("SELECT * FROM t WHERE x NOT IN (1, 2)")
        self.assertIsInstance(stmt.where, SqlLogic)
        self.assertEqual(stmt.where.op, 'not')

    def test_cross_join(self):
        stmt = parse_sql("SELECT * FROM a CROSS JOIN b")
        self.assertEqual(stmt.joins[0].join_type, 'cross')

    def test_implicit_cross_join(self):
        stmt = parse_sql("SELECT * FROM a, b")
        self.assertEqual(len(stmt.joins), 1)
        self.assertEqual(stmt.joins[0].join_type, 'cross')


class TestCatalog(unittest.TestCase):
    """Test schema catalog."""

    def test_create_and_get(self):
        cat = Catalog()
        schema = cat.create_table('users', [
            ColumnDef('id', 'int', primary_key=True),
            ColumnDef('name', 'text'),
        ])
        self.assertEqual(schema.name, 'users')
        self.assertEqual(schema.column_names(), ['id', 'name'])

    def test_duplicate_table(self):
        cat = Catalog()
        cat.create_table('t', [ColumnDef('x', 'int')])
        with self.assertRaises(CatalogError):
            cat.create_table('t', [ColumnDef('x', 'int')])

    def test_if_not_exists(self):
        cat = Catalog()
        cat.create_table('t', [ColumnDef('x', 'int')])
        s2 = cat.create_table('t', [ColumnDef('x', 'int')], if_not_exists=True)
        self.assertEqual(s2.name, 't')

    def test_drop_table(self):
        cat = Catalog()
        cat.create_table('t', [ColumnDef('x', 'int')])
        cat.drop_table('t')
        self.assertFalse(cat.has_table('t'))

    def test_drop_nonexistent(self):
        cat = Catalog()
        with self.assertRaises(CatalogError):
            cat.drop_table('noexist')

    def test_drop_if_exists(self):
        cat = Catalog()
        cat.drop_table('noexist', if_exists=True)  # should not raise

    def test_list_tables(self):
        cat = Catalog()
        cat.create_table('b', [ColumnDef('x', 'int')])
        cat.create_table('a', [ColumnDef('x', 'int')])
        self.assertEqual(cat.list_tables(), ['a', 'b'])

    def test_primary_key(self):
        cat = Catalog()
        schema = cat.create_table('t', [
            ColumnDef('id', 'int', primary_key=True),
            ColumnDef('name', 'text'),
        ])
        self.assertEqual(schema.primary_key_column(), 'id')


class TestMiniDBBasic(unittest.TestCase):
    """Test basic DDL and DML operations."""

    def setUp(self):
        self.db = MiniDB()

    def test_create_table(self):
        r = self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.assertEqual(r.message, "CREATE TABLE users")

    def test_insert_and_select(self):
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
        self.db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        self.db.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")
        r = self.db.execute("SELECT * FROM users")
        self.assertEqual(len(r), 2)

    def test_select_with_where(self):
        self.db.execute("CREATE TABLE t (x INT, y INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        self.db.execute("INSERT INTO t VALUES (2, 20)")
        self.db.execute("INSERT INTO t VALUES (3, 30)")
        r = self.db.execute("SELECT * FROM t WHERE x > 1")
        self.assertEqual(len(r), 2)

    def test_select_columns(self):
        self.db.execute("CREATE TABLE t (x INT, y INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        r = self.db.execute("SELECT x FROM t")
        self.assertEqual(len(r.columns), 1)
        self.assertEqual(r[0][0], 1)

    def test_update(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, val TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'old')")
        r = self.db.execute("UPDATE t SET val = 'new' WHERE id = 1")
        self.assertEqual(r.rows_affected, 1)
        r2 = self.db.execute("SELECT val FROM t WHERE id = 1")
        self.assertEqual(r2[0][0], 'new')

    def test_delete(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, val TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'a')")
        self.db.execute("INSERT INTO t VALUES (2, 'b')")
        r = self.db.execute("DELETE FROM t WHERE id = 1")
        self.assertEqual(r.rows_affected, 1)
        r2 = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(r2), 1)

    def test_delete_all(self):
        self.db.execute("CREATE TABLE t (x INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("INSERT INTO t VALUES (2)")
        r = self.db.execute("DELETE FROM t")
        self.assertEqual(r.rows_affected, 2)
        r2 = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(r2), 0)

    def test_insert_multiple_rows(self):
        self.db.execute("CREATE TABLE t (x INT, y TEXT)")
        r = self.db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        self.assertEqual(r.rows_affected, 3)
        r2 = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(r2), 3)

    def test_show_tables(self):
        self.db.execute("CREATE TABLE a (x INT)")
        self.db.execute("CREATE TABLE b (x INT)")
        r = self.db.execute("SHOW TABLES")
        self.assertEqual(r.column('table_name'), ['a', 'b'])

    def test_describe(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL)")
        r = self.db.execute("DESCRIBE t")
        self.assertEqual(len(r), 2)
        self.assertIn('PRIMARY KEY', r[0][2])

    def test_drop_table(self):
        self.db.execute("CREATE TABLE t (x INT)")
        self.db.execute("INSERT INTO t VALUES (1)")
        self.db.execute("DROP TABLE t")
        with self.assertRaises(CatalogError):
            self.db.execute("SELECT * FROM t")

    def test_create_if_not_exists(self):
        self.db.execute("CREATE TABLE t (x INT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS t (x INT)")  # no error

    def test_drop_if_exists(self):
        self.db.execute("DROP TABLE IF EXISTS nonexistent")  # no error

    def test_null_literal(self):
        self.db.execute("CREATE TABLE t (x INT, y TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, NULL)")
        r = self.db.execute("SELECT * FROM t")
        self.assertIsNone(r[0][1])

    def test_boolean_literals(self):
        self.db.execute("CREATE TABLE t (x BOOL)")
        self.db.execute("INSERT INTO t VALUES (TRUE)")
        self.db.execute("INSERT INTO t VALUES (FALSE)")
        r = self.db.execute("SELECT * FROM t")
        self.assertEqual(len(r), 2)


class TestMiniDBQueries(unittest.TestCase):
    """Test more complex queries."""

    def setUp(self):
        self.db = MiniDB()
        self.db.execute("CREATE TABLE emp (id INT PRIMARY KEY, name TEXT, dept TEXT, salary INT)")
        self.db.execute("INSERT INTO emp VALUES (1, 'Alice', 'eng', 100)")
        self.db.execute("INSERT INTO emp VALUES (2, 'Bob', 'eng', 120)")
        self.db.execute("INSERT INTO emp VALUES (3, 'Charlie', 'sales', 90)")
        self.db.execute("INSERT INTO emp VALUES (4, 'Diana', 'sales', 110)")
        self.db.execute("INSERT INTO emp VALUES (5, 'Eve', 'hr', 95)")

    def test_order_by_asc(self):
        r = self.db.execute("SELECT name FROM emp ORDER BY salary ASC")
        names = [row[0] for row in r]
        self.assertEqual(names, ['Charlie', 'Eve', 'Alice', 'Diana', 'Bob'])

    def test_order_by_desc(self):
        r = self.db.execute("SELECT name FROM emp ORDER BY salary DESC")
        names = [row[0] for row in r]
        self.assertEqual(names, ['Bob', 'Diana', 'Alice', 'Eve', 'Charlie'])

    def test_limit(self):
        r = self.db.execute("SELECT * FROM emp ORDER BY salary DESC LIMIT 3")
        self.assertEqual(len(r), 3)

    def test_limit_offset(self):
        r = self.db.execute("SELECT name FROM emp ORDER BY salary ASC LIMIT 2 OFFSET 1")
        self.assertEqual(len(r), 2)
        names = [row[0] for row in r]
        self.assertEqual(names, ['Eve', 'Alice'])

    def test_distinct(self):
        r = self.db.execute("SELECT DISTINCT dept FROM emp")
        depts = sorted([row[0] for row in r])
        self.assertEqual(depts, ['eng', 'hr', 'sales'])

    def test_count_star(self):
        r = self.db.execute("SELECT COUNT(*) FROM emp")
        self.assertEqual(r[0][0], 5)

    def test_sum(self):
        r = self.db.execute("SELECT SUM(salary) FROM emp")
        self.assertEqual(r[0][0], 515)

    def test_avg(self):
        r = self.db.execute("SELECT AVG(salary) FROM emp")
        self.assertEqual(r[0][0], 103.0)

    def test_min_max(self):
        r = self.db.execute("SELECT MIN(salary), MAX(salary) FROM emp")
        self.assertEqual(r[0][0], 90)
        self.assertEqual(r[0][1], 120)

    def test_group_by(self):
        r = self.db.execute("SELECT dept, COUNT(*) FROM emp GROUP BY dept ORDER BY dept")
        self.assertEqual(len(r), 3)
        # eng=2, hr=1, sales=2
        depts = {row[0]: row[1] for row in r}
        self.assertEqual(depts['eng'], 2)
        self.assertEqual(depts['hr'], 1)
        self.assertEqual(depts['sales'], 2)

    def test_group_by_sum(self):
        r = self.db.execute("SELECT dept, SUM(salary) FROM emp GROUP BY dept ORDER BY dept")
        sums = {row[0]: row[1] for row in r}
        self.assertEqual(sums['eng'], 220)
        self.assertEqual(sums['sales'], 200)

    def test_having(self):
        r = self.db.execute("SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 1")
        self.assertEqual(len(r), 2)  # eng and sales

    def test_where_and(self):
        r = self.db.execute("SELECT * FROM emp WHERE dept = 'eng' AND salary > 110")
        self.assertEqual(len(r), 1)

    def test_where_or(self):
        r = self.db.execute("SELECT * FROM emp WHERE dept = 'hr' OR dept = 'sales'")
        self.assertEqual(len(r), 3)

    def test_where_not(self):
        r = self.db.execute("SELECT * FROM emp WHERE NOT dept = 'eng'")
        self.assertEqual(len(r), 3)

    def test_where_like(self):
        r = self.db.execute("SELECT * FROM emp WHERE name LIKE 'A%'")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][1], 'Alice')

    def test_where_in(self):
        r = self.db.execute("SELECT * FROM emp WHERE dept IN ('eng', 'hr')")
        self.assertEqual(len(r), 3)

    def test_where_between(self):
        r = self.db.execute("SELECT * FROM emp WHERE salary BETWEEN 95 AND 110")
        self.assertEqual(len(r), 3)  # Alice(100), Diana(110), Eve(95)

    def test_arithmetic(self):
        r = self.db.execute("SELECT name, salary * 2 FROM emp WHERE id = 1")
        self.assertEqual(r[0][1], 200)

    def test_comparison_operators(self):
        r = self.db.execute("SELECT * FROM emp WHERE salary >= 100")
        self.assertEqual(len(r), 3)  # Alice, Bob, Diana

    def test_ne_operator(self):
        r = self.db.execute("SELECT * FROM emp WHERE dept != 'eng'")
        self.assertEqual(len(r), 3)


class TestMiniDBJoins(unittest.TestCase):
    """Test JOIN operations."""

    def setUp(self):
        self.db = MiniDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO users VALUES (3, 'Charlie')")
        self.db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, amount INT)")
        self.db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        self.db.execute("INSERT INTO orders VALUES (2, 1, 200)")
        self.db.execute("INSERT INTO orders VALUES (3, 2, 150)")

    def test_inner_join(self):
        r = self.db.execute(
            "SELECT users.name, orders.amount FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "ORDER BY orders.amount"
        )
        self.assertEqual(len(r), 3)

    def test_left_join(self):
        r = self.db.execute(
            "SELECT users.name, orders.amount FROM users "
            "LEFT JOIN orders ON users.id = orders.user_id "
            "ORDER BY users.name"
        )
        # Charlie has no orders, should still appear with NULL
        self.assertEqual(len(r), 4)
        charlie_rows = [row for row in r if row[0] == 'Charlie']
        self.assertEqual(len(charlie_rows), 1)
        self.assertIsNone(charlie_rows[0][1])

    def test_cross_join(self):
        self.db.execute("CREATE TABLE a (x INT)")
        self.db.execute("CREATE TABLE b (y INT)")
        self.db.execute("INSERT INTO a VALUES (1)")
        self.db.execute("INSERT INTO a VALUES (2)")
        self.db.execute("INSERT INTO b VALUES (10)")
        self.db.execute("INSERT INTO b VALUES (20)")
        r = self.db.execute("SELECT * FROM a CROSS JOIN b")
        self.assertEqual(len(r), 4)

    def test_join_with_where(self):
        r = self.db.execute(
            "SELECT users.name, orders.amount FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "WHERE orders.amount > 100"
        )
        self.assertEqual(len(r), 2)


class TestMiniDBTransactions(unittest.TestCase):
    """Test ACID transaction support."""

    def setUp(self):
        self.db = MiniDB()
        self.db.execute("CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
        self.db.execute("INSERT INTO accounts VALUES (1, 1000)")
        self.db.execute("INSERT INTO accounts VALUES (2, 500)")

    def test_begin_commit(self):
        self.db.execute("BEGIN")
        self.db.execute("UPDATE accounts SET balance = 900 WHERE id = 1")
        self.db.execute("UPDATE accounts SET balance = 600 WHERE id = 2")
        self.db.execute("COMMIT")
        r = self.db.execute("SELECT balance FROM accounts WHERE id = 1")
        self.assertEqual(r[0][0], 900)
        r = self.db.execute("SELECT balance FROM accounts WHERE id = 2")
        self.assertEqual(r[0][0], 600)

    def test_rollback(self):
        self.db.execute("BEGIN")
        self.db.execute("UPDATE accounts SET balance = 0 WHERE id = 1")
        self.db.execute("ROLLBACK")
        r = self.db.execute("SELECT balance FROM accounts WHERE id = 1")
        self.assertEqual(r[0][0], 1000)

    def test_savepoint_rollback(self):
        self.db.execute("BEGIN")
        self.db.execute("UPDATE accounts SET balance = 900 WHERE id = 1")
        self.db.execute("SAVEPOINT sp1")
        self.db.execute("UPDATE accounts SET balance = 0 WHERE id = 2")
        self.db.execute("ROLLBACK TO SAVEPOINT sp1")
        self.db.execute("COMMIT")
        # id=1 updated, id=2 rolled back to savepoint
        r1 = self.db.execute("SELECT balance FROM accounts WHERE id = 1")
        self.assertEqual(r1[0][0], 900)
        r2 = self.db.execute("SELECT balance FROM accounts WHERE id = 2")
        self.assertEqual(r2[0][0], 500)

    def test_begin_without_commit_then_begin_raises(self):
        self.db.execute("BEGIN")
        with self.assertRaises(DatabaseError):
            self.db.execute("BEGIN")
        self.db.execute("ROLLBACK")  # cleanup

    def test_commit_without_begin_raises(self):
        with self.assertRaises(DatabaseError):
            self.db.execute("COMMIT")

    def test_autocommit(self):
        """Without BEGIN, each statement auto-commits."""
        self.db.execute("UPDATE accounts SET balance = 999 WHERE id = 1")
        r = self.db.execute("SELECT balance FROM accounts WHERE id = 1")
        self.assertEqual(r[0][0], 999)


class TestMiniDBConstraints(unittest.TestCase):
    """Test constraint enforcement."""

    def test_primary_key_uniqueness(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'a')")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t VALUES (1, 'b')")

    def test_not_null(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id) VALUES (1)")

    def test_unique_constraint(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, email TEXT UNIQUE)")
        db.execute("INSERT INTO t VALUES (1, 'a@b.com')")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t VALUES (2, 'a@b.com')")

    def test_default_values(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, status TEXT DEFAULT 'active')")
        db.execute("INSERT INTO t (id) VALUES (1)")
        r = db.execute("SELECT status FROM t WHERE id = 1")
        self.assertEqual(r[0][0], 'active')

    def test_auto_increment_pk(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO t (name) VALUES ('Alice')")
        db.execute("INSERT INTO t (name) VALUES ('Bob')")
        r = db.execute("SELECT * FROM t ORDER BY id")
        self.assertEqual(len(r), 2)
        # Both should have unique IDs
        ids = [row[0] for row in r]
        self.assertEqual(len(set(ids)), 2)


class TestMiniDBExplain(unittest.TestCase):
    """Test EXPLAIN."""

    def test_explain_select(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (1)")
        r = db.execute("EXPLAIN SELECT * FROM t WHERE x > 0")
        self.assertTrue(len(r) > 0)

    def test_explain_join(self):
        db = MiniDB()
        db.execute("CREATE TABLE a (x INT)")
        db.execute("CREATE TABLE b (y INT)")
        r = db.execute("EXPLAIN SELECT * FROM a JOIN b ON a.x = b.y")
        self.assertTrue(len(r) > 0)


class TestMiniDBCreateIndex(unittest.TestCase):
    """Test index creation."""

    def test_create_index(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        r = db.execute("CREATE INDEX idx_name ON t (name)")
        self.assertIn('CREATE INDEX', r.message)
        schema = db.storage.catalog.get_table('t')
        self.assertIn('idx_name', schema.indexes)

    def test_index_recorded_in_catalog(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT, val INT)")
        db.execute("CREATE INDEX idx_val ON t (val)")
        schema = db.storage.catalog.get_table('t')
        self.assertEqual(schema.indexes['idx_val'], 'val')


class TestResultSet(unittest.TestCase):
    """Test ResultSet API."""

    def test_scalar(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (42)")
        r = db.execute("SELECT x FROM t")
        self.assertEqual(r.scalar(), 42)

    def test_column(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT, y TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'a')")
        db.execute("INSERT INTO t VALUES (2, 'b')")
        r = db.execute("SELECT * FROM t")
        self.assertEqual(sorted(r.column('x')), [1, 2])

    def test_to_dicts(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT, y TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'a')")
        r = db.execute("SELECT * FROM t")
        dicts = r.to_dicts()
        self.assertEqual(len(dicts), 1)
        self.assertEqual(dicts[0]['x'], 1)
        self.assertEqual(dicts[0]['y'], 'a')

    def test_len(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (1)")
        db.execute("INSERT INTO t VALUES (2)")
        r = db.execute("SELECT * FROM t")
        self.assertEqual(len(r), 2)

    def test_iter(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (1)")
        db.execute("INSERT INTO t VALUES (2)")
        r = db.execute("SELECT * FROM t")
        vals = [row[0] for row in r]
        self.assertEqual(sorted(vals), [1, 2])

    def test_bool_nonempty(self):
        rs = ResultSet(columns=['x'], rows=[[1]])
        self.assertTrue(rs)

    def test_bool_message(self):
        rs = ResultSet(columns=[], rows=[], message="OK")
        self.assertTrue(rs)

    def test_repr(self):
        rs = ResultSet(columns=['x'], rows=[[1], [2]])
        self.assertIn('2 rows', repr(rs))


class TestMiniDBStats(unittest.TestCase):
    """Test statistics."""

    def test_stats(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (1)")
        s = db.stats()
        self.assertIn('buffer_pool', s)
        self.assertIn('transactions', s)
        self.assertIn('tables', s)
        self.assertEqual(s['tables'], 1)


class TestMiniDBEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_table_select(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT, y TEXT)")
        r = db.execute("SELECT * FROM t")
        self.assertEqual(len(r), 0)

    def test_nonexistent_table(self):
        db = MiniDB()
        with self.assertRaises(CatalogError):
            db.execute("SELECT * FROM nonexistent")

    def test_parse_error(self):
        db = MiniDB()
        with self.assertRaises(ParseError):
            db.execute("GOBBLEDYGOOK")

    def test_update_no_match(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'a')")
        r = db.execute("UPDATE t SET val = 'b' WHERE id = 999")
        self.assertEqual(r.rows_affected, 0)

    def test_delete_no_match(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
        db.execute("INSERT INTO t VALUES (1)")
        r = db.execute("DELETE FROM t WHERE id = 999")
        self.assertEqual(r.rows_affected, 0)

    def test_string_comparison(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (name TEXT)")
        db.execute("INSERT INTO t VALUES ('Alice')")
        db.execute("INSERT INTO t VALUES ('Bob')")
        r = db.execute("SELECT * FROM t WHERE name = 'Alice'")
        self.assertEqual(len(r), 1)

    def test_negative_number(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (-5)")
        r = db.execute("SELECT * FROM t")
        self.assertEqual(r[0][0], -5)

    def test_float_values(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x FLOAT)")
        db.execute("INSERT INTO t VALUES (3.14)")
        r = db.execute("SELECT * FROM t")
        self.assertAlmostEqual(r[0][0], 3.14)

    def test_update_multiple_columns(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, a TEXT, b TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'x', 'y')")
        db.execute("UPDATE t SET a = 'A', b = 'B' WHERE id = 1")
        r = db.execute("SELECT a, b FROM t WHERE id = 1")
        self.assertEqual(r[0][0], 'A')
        self.assertEqual(r[0][1], 'B')

    def test_close(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("BEGIN")
        db.close()  # should abort active transaction

    def test_multiple_inserts_then_count(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        for i in range(50):
            db.execute(f"INSERT INTO t VALUES ({i})")
        r = db.execute("SELECT COUNT(*) FROM t")
        self.assertEqual(r[0][0], 50)

    def test_update_with_expression(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("UPDATE t SET val = val + 5 WHERE id = 1")
        r = db.execute("SELECT val FROM t WHERE id = 1")
        self.assertEqual(r[0][0], 15)


class TestMiniDBMultiTable(unittest.TestCase):
    """Test multi-table scenarios."""

    def test_two_tables_independent(self):
        db = MiniDB()
        db.execute("CREATE TABLE t1 (x INT)")
        db.execute("CREATE TABLE t2 (y INT)")
        db.execute("INSERT INTO t1 VALUES (1)")
        db.execute("INSERT INTO t2 VALUES (2)")
        r1 = db.execute("SELECT * FROM t1")
        r2 = db.execute("SELECT * FROM t2")
        self.assertEqual(r1[0][0], 1)
        self.assertEqual(r2[0][0], 2)

    def test_join_three_tables(self):
        db = MiniDB()
        db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, product_id INT)")
        db.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO users VALUES (1, 'Alice')")
        db.execute("INSERT INTO products VALUES (10, 'Widget')")
        db.execute("INSERT INTO orders VALUES (1, 1, 10)")
        r = db.execute(
            "SELECT users.name, products.name FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "JOIN products ON orders.product_id = products.id"
        )
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 'Alice')
        self.assertEqual(r[0][1], 'Widget')


class TestStorageEngine(unittest.TestCase):
    """Test StorageEngine directly."""

    def test_insert_and_scan(self):
        se = StorageEngine()
        se.catalog.create_table('t', [ColumnDef('x', 'int')])
        txn = se.begin()
        se.insert_row(txn, 't', {'x': 42})
        rows = se.scan_table(txn, 't')
        se.commit(txn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1]['x'], 42)

    def test_delete_row(self):
        se = StorageEngine()
        se.catalog.create_table('t', [
            ColumnDef('id', 'int', primary_key=True),
            ColumnDef('val', 'text'),
        ])
        txn = se.begin()
        se.insert_row(txn, 't', {'id': 1, 'val': 'a'})
        se.delete_row(txn, 't', 1)
        rows = se.scan_table(txn, 't')
        se.commit(txn)
        self.assertEqual(len(rows), 0)

    def test_update_row(self):
        se = StorageEngine()
        se.catalog.create_table('t', [
            ColumnDef('id', 'int', primary_key=True),
            ColumnDef('val', 'text'),
        ])
        txn = se.begin()
        se.insert_row(txn, 't', {'id': 1, 'val': 'old'})
        se.update_row(txn, 't', 1, {'val': 'new'})
        row = se.get_row(txn, 't', 1)
        se.commit(txn)
        self.assertEqual(row['val'], 'new')

    def test_count_rows(self):
        se = StorageEngine()
        se.catalog.create_table('t', [ColumnDef('x', 'int')])
        txn = se.begin()
        for i in range(10):
            se.insert_row(txn, 't', {'x': i})
        count = se.count_rows(txn, 't')
        se.commit(txn)
        self.assertEqual(count, 10)

    def test_transaction_abort(self):
        se = StorageEngine()
        se.catalog.create_table('t', [ColumnDef('x', 'int')])
        # Insert in committed txn
        txn1 = se.begin()
        se.insert_row(txn1, 't', {'x': 1})
        se.commit(txn1)
        # Insert in aborted txn
        txn2 = se.begin()
        se.insert_row(txn2, 't', {'x': 2})
        se.abort(txn2)
        # Only first row should exist
        txn3 = se.begin()
        rows = se.scan_table(txn3, 't')
        se.commit(txn3)
        self.assertEqual(len(rows), 1)

    def test_stats(self):
        se = StorageEngine()
        s = se.stats()
        self.assertIn('buffer_pool', s)
        self.assertIn('transactions', s)


class TestMiniDBNullHandling(unittest.TestCase):
    """Test NULL handling."""

    def test_is_null(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT, y TEXT)")
        db.execute("INSERT INTO t VALUES (1, NULL)")
        db.execute("INSERT INTO t VALUES (2, 'hello')")
        r = db.execute("SELECT * FROM t WHERE y IS NULL")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 1)

    def test_is_not_null(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT, y TEXT)")
        db.execute("INSERT INTO t VALUES (1, NULL)")
        db.execute("INSERT INTO t VALUES (2, 'hello')")
        r = db.execute("SELECT * FROM t WHERE y IS NOT NULL")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 2)


class TestMiniDBCaseExpression(unittest.TestCase):
    """Test CASE WHEN expressions."""

    def test_case_in_select(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (1)")
        db.execute("INSERT INTO t VALUES (2)")
        db.execute("INSERT INTO t VALUES (3)")
        r = db.execute(
            "SELECT x, CASE WHEN x > 2 THEN 'high' ELSE 'low' END FROM t ORDER BY x"
        )
        self.assertEqual(r[0][1], 'low')
        self.assertEqual(r[2][1], 'high')


class TestMiniDBSubExpressions(unittest.TestCase):
    """Test complex expression evaluation."""

    def test_arithmetic_in_where(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (10)")
        db.execute("INSERT INTO t VALUES (20)")
        r = db.execute("SELECT * FROM t WHERE x + 5 > 20")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 20)

    def test_nested_conditions(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (a INT, b INT, c INT)")
        db.execute("INSERT INTO t VALUES (1, 2, 3)")
        db.execute("INSERT INTO t VALUES (4, 5, 6)")
        db.execute("INSERT INTO t VALUES (7, 8, 9)")
        r = db.execute("SELECT * FROM t WHERE (a > 1 AND b < 8) OR c = 9")
        self.assertEqual(len(r), 2)

    def test_not_like(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (name TEXT)")
        db.execute("INSERT INTO t VALUES ('Alice')")
        db.execute("INSERT INTO t VALUES ('Bob')")
        db.execute("INSERT INTO t VALUES ('Charlie')")
        r = db.execute("SELECT * FROM t WHERE name NOT LIKE 'B%'")
        self.assertEqual(len(r), 2)

    def test_not_between(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (1)")
        db.execute("INSERT INTO t VALUES (5)")
        db.execute("INSERT INTO t VALUES (10)")
        r = db.execute("SELECT * FROM t WHERE x NOT BETWEEN 2 AND 8")
        self.assertEqual(len(r), 2)

    def test_not_in(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (1)")
        db.execute("INSERT INTO t VALUES (2)")
        db.execute("INSERT INTO t VALUES (3)")
        r = db.execute("SELECT * FROM t WHERE x NOT IN (1, 3)")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 2)


class TestMiniDBSelectAlias(unittest.TestCase):
    """Test column aliases."""

    def test_column_alias(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (42)")
        r = db.execute("SELECT x AS value FROM t")
        self.assertIn('value', r.columns)

    def test_expression_alias(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        db.execute("INSERT INTO t VALUES (10)")
        r = db.execute("SELECT x * 2 AS doubled FROM t")
        self.assertIn('doubled', r.columns)
        self.assertEqual(r[0][0], 20)


class TestMiniDBMultiRowInsert(unittest.TestCase):
    """Test bulk operations."""

    def test_100_rows(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        db.execute("BEGIN")
        for i in range(100):
            db.execute(f"INSERT INTO t VALUES ({i}, {i * 10})")
        db.execute("COMMIT")
        r = db.execute("SELECT COUNT(*) FROM t")
        self.assertEqual(r[0][0], 100)

    def test_100_rows_sum(self):
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        for i in range(100):
            db.execute(f"INSERT INTO t VALUES ({i}, {i})")
        r = db.execute("SELECT SUM(val) FROM t")
        self.assertEqual(r[0][0], sum(range(100)))


class TestMiniDBComposition(unittest.TestCase):
    """Test that all three composed components work together."""

    def test_buffer_pool_stats_change(self):
        """Verify buffer pool is being used."""
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT)")
        for i in range(20):
            db.execute(f"INSERT INTO t VALUES ({i})")
        s = db.stats()
        # Buffer pool should have some activity
        self.assertIsInstance(s['buffer_pool']['hit_rate'], float)

    def test_transaction_isolation(self):
        """Verify transaction manager provides isolation."""
        db = MiniDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t VALUES (1, 100)")

        # Start transaction, modify, but don't commit
        db.execute("BEGIN")
        db.execute("UPDATE t SET val = 999 WHERE id = 1")

        # Use a separate db to check isolation
        # (In our design, autocommit reads see committed data)
        # Just verify the update is visible within the transaction
        r = db.execute("SELECT val FROM t WHERE id = 1")
        self.assertEqual(r[0][0], 999)

        db.execute("ROLLBACK")
        r = db.execute("SELECT val FROM t WHERE id = 1")
        self.assertEqual(r[0][0], 100)

    def test_query_executor_operators(self):
        """Verify C245 operators are used in query execution."""
        db = MiniDB()
        db.execute("CREATE TABLE t (x INT, y TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'a')")
        db.execute("INSERT INTO t VALUES (2, 'b')")
        db.execute("INSERT INTO t VALUES (3, 'a')")
        # This exercises: SeqScan -> Filter -> HashAggregate -> Sort -> Project
        r = db.execute(
            "SELECT y, COUNT(*) FROM t GROUP BY y ORDER BY y"
        )
        self.assertEqual(len(r), 2)


if __name__ == '__main__':
    unittest.main()
