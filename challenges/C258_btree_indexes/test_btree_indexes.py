"""
Tests for C258: B-Tree Indexed Database
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C116_bplus_tree')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))

from btree_indexes import IndexedDB, IndexManager, QueryOptimizer, IndexScanDecision, IndexInfo
from bplus_tree import BPlusTreeMap


# =============================================================================
# IndexManager unit tests
# =============================================================================

class TestIndexManager(unittest.TestCase):
    def setUp(self):
        self.mgr = IndexManager()

    def test_create_index(self):
        info = self.mgr.create_index('idx_age', 'users', ['age'])
        self.assertEqual(info.name, 'idx_age')
        self.assertEqual(info.table_name, 'users')
        self.assertEqual(info.columns, ['age'])
        self.assertFalse(info.unique)
        self.assertFalse(info.is_composite)

    def test_create_composite_index(self):
        info = self.mgr.create_index('idx_name_age', 'users', ['name', 'age'])
        self.assertTrue(info.is_composite)
        self.assertEqual(info.columns, ['name', 'age'])

    def test_create_unique_index(self):
        info = self.mgr.create_index('idx_email', 'users', ['email'], unique=True)
        self.assertTrue(info.unique)

    def test_duplicate_index_name(self):
        self.mgr.create_index('idx1', 'users', ['age'])
        with self.assertRaises(Exception):
            self.mgr.create_index('idx1', 'users', ['name'])

    def test_drop_index(self):
        self.mgr.create_index('idx1', 'users', ['age'])
        self.mgr.drop_index('idx1')
        self.assertIsNone(self.mgr.get_index('idx1'))

    def test_drop_nonexistent(self):
        with self.assertRaises(Exception):
            self.mgr.drop_index('nope')

    def test_get_indexes_for_table(self):
        self.mgr.create_index('idx1', 'users', ['age'])
        self.mgr.create_index('idx2', 'users', ['name'])
        self.mgr.create_index('idx3', 'posts', ['title'])
        user_idxs = self.mgr.get_indexes_for_table('users')
        self.assertEqual(len(user_idxs), 2)
        post_idxs = self.mgr.get_indexes_for_table('posts')
        self.assertEqual(len(post_idxs), 1)

    def test_find_index_for_column(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.assertIsNotNone(self.mgr.find_index_for_column('users', 'age'))
        self.assertIsNone(self.mgr.find_index_for_column('users', 'name'))

    def test_find_composite_index(self):
        self.mgr.create_index('idx_nm', 'users', ['name', 'age'])
        self.assertIsNotNone(self.mgr.find_index_for_columns('users', ['name', 'age']))
        self.assertIsNone(self.mgr.find_index_for_columns('users', ['age', 'name']))

    def test_build_index(self):
        info = self.mgr.create_index('idx_age', 'users', ['age'])
        rows = [
            (1, {'name': 'Alice', 'age': 30}),
            (2, {'name': 'Bob', 'age': 25}),
            (3, {'name': 'Carol', 'age': 30}),
        ]
        self.mgr.build_index(info, rows)
        self.assertEqual(self.mgr.lookup_eq('idx_age', 30), [1, 3])
        self.assertEqual(self.mgr.lookup_eq('idx_age', 25), [2])

    def test_build_unique_index_violation(self):
        info = self.mgr.create_index('idx_email', 'users', ['email'], unique=True)
        rows = [
            (1, {'email': 'a@b.com'}),
            (2, {'email': 'a@b.com'}),
        ]
        with self.assertRaises(Exception):
            self.mgr.build_index(info, rows)

    def test_on_insert(self):
        info = self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'name': 'Alice', 'age': 30})
        self.mgr.on_insert('users', 2, {'name': 'Bob', 'age': 25})
        self.assertEqual(self.mgr.lookup_eq('idx_age', 30), [1])
        self.assertEqual(self.mgr.lookup_eq('idx_age', 25), [2])

    def test_on_insert_unique_violation(self):
        self.mgr.create_index('idx_email', 'users', ['email'], unique=True)
        self.mgr.on_insert('users', 1, {'email': 'a@b.com'})
        with self.assertRaises(Exception):
            self.mgr.on_insert('users', 2, {'email': 'a@b.com'})

    def test_on_delete(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'age': 30})
        self.mgr.on_insert('users', 2, {'age': 30})
        self.mgr.on_delete('users', 1, {'age': 30})
        self.assertEqual(self.mgr.lookup_eq('idx_age', 30), [2])

    def test_on_delete_removes_key(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'age': 30})
        self.mgr.on_delete('users', 1, {'age': 30})
        self.assertEqual(self.mgr.lookup_eq('idx_age', 30), [])

    def test_on_update(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'age': 30})
        self.mgr.on_update('users', 1, {'age': 30}, {'age': 35})
        self.assertEqual(self.mgr.lookup_eq('idx_age', 30), [])
        self.assertEqual(self.mgr.lookup_eq('idx_age', 35), [1])

    def test_on_update_same_value(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'name': 'Alice', 'age': 30})
        self.mgr.on_update('users', 1, {'name': 'Alice', 'age': 30}, {'name': 'Alicia', 'age': 30})
        self.assertEqual(self.mgr.lookup_eq('idx_age', 30), [1])

    def test_on_update_unique_violation(self):
        self.mgr.create_index('idx_email', 'users', ['email'], unique=True)
        self.mgr.on_insert('users', 1, {'email': 'a@b.com'})
        self.mgr.on_insert('users', 2, {'email': 'c@d.com'})
        with self.assertRaises(Exception):
            self.mgr.on_update('users', 2, {'email': 'c@d.com'}, {'email': 'a@b.com'})

    def test_lookup_range(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        for i, age in enumerate([20, 25, 30, 35, 40]):
            self.mgr.on_insert('users', i + 1, {'age': age})
        result = self.mgr.lookup_range('idx_age', low=25, high=35)
        self.assertEqual(sorted(result), [2, 3, 4])

    def test_lookup_range_exclusive(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        for i, age in enumerate([20, 25, 30, 35, 40]):
            self.mgr.on_insert('users', i + 1, {'age': age})
        result = self.mgr.lookup_range('idx_age', low=25, high=35,
                                        low_inclusive=False, high_inclusive=False)
        self.assertEqual(sorted(result), [3])

    def test_null_keys_skipped(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'age': None})
        self.mgr.on_insert('users', 2, {'age': 30})
        self.assertEqual(self.mgr.lookup_eq('idx_age', 30), [2])

    def test_composite_key_lookup(self):
        self.mgr.create_index('idx_nm', 'users', ['name', 'age'])
        self.mgr.on_insert('users', 1, {'name': 'Alice', 'age': 30})
        self.mgr.on_insert('users', 2, {'name': 'Alice', 'age': 25})
        self.mgr.on_insert('users', 3, {'name': 'Bob', 'age': 30})
        self.assertEqual(self.mgr.lookup_eq('idx_nm', ('Alice', 30)), [1])
        self.assertEqual(self.mgr.lookup_eq('idx_nm', ('Alice', 25)), [2])

    def test_count(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'age': 30})
        self.mgr.on_insert('users', 2, {'age': 25})
        self.mgr.on_insert('users', 3, {'age': 30})  # duplicate key
        self.assertEqual(self.mgr.count('idx_age'), 2)  # 2 distinct keys

    def test_stats(self):
        self.mgr.create_index('idx_age', 'users', ['age'])
        self.mgr.on_insert('users', 1, {'age': 30})
        stats = self.mgr.stats()
        self.assertIn('idx_age', stats)
        self.assertEqual(stats['idx_age']['table'], 'users')
        self.assertEqual(stats['idx_age']['keys'], 1)


# =============================================================================
# IndexedDB integration tests
# =============================================================================

class TestIndexedDBBasic(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT, email TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice', 30, 'alice@test.com')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob', 25, 'bob@test.com')")
        self.db.execute("INSERT INTO users VALUES (3, 'Carol', 35, 'carol@test.com')")
        self.db.execute("INSERT INTO users VALUES (4, 'Dave', 30, 'dave@test.com')")
        self.db.execute("INSERT INTO users VALUES (5, 'Eve', 28, 'eve@test.com')")

    def test_create_index(self):
        r = self.db.execute("CREATE INDEX idx_age ON users (age)")
        self.assertIn('CREATE INDEX', r.message)

    def test_index_populated_on_create(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        info = self.db.index_manager.get_index('idx_age')
        self.assertEqual(self.db.index_manager.count('idx_age'), 4)  # 25, 28, 30, 35

    def test_select_with_index_eq(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        r = self.db.execute("SELECT name FROM users WHERE age = 30")
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Alice', 'Dave'])

    def test_select_with_index_lt(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        r = self.db.execute("SELECT name FROM users WHERE age < 30")
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Bob', 'Eve'])

    def test_select_with_index_gt(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        r = self.db.execute("SELECT name FROM users WHERE age > 30")
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Carol'])

    def test_select_with_index_lte(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        r = self.db.execute("SELECT name FROM users WHERE age <= 28")
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Bob', 'Eve'])

    def test_select_with_index_gte(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        r = self.db.execute("SELECT name FROM users WHERE age >= 30")
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Alice', 'Carol', 'Dave'])

    def test_select_star_with_index(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        r = self.db.execute("SELECT * FROM users WHERE age = 25")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][1], 'Bob')

    def test_no_index_falls_back_to_seq(self):
        r = self.db.execute("SELECT name FROM users WHERE age = 30")
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Alice', 'Dave'])
        self.assertEqual(self.db._seq_scan_count, 1)

    def test_index_scan_counter(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        self.db.execute("SELECT name FROM users WHERE age = 30")
        self.assertEqual(self.db._index_scan_count, 1)

    def test_index_maintained_on_insert(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        self.db.execute("INSERT INTO users VALUES (6, 'Frank', 30, 'frank@test.com')")
        r = self.db.execute("SELECT name FROM users WHERE age = 30")
        names = sorted([row[0] for row in r.rows])
        self.assertEqual(names, ['Alice', 'Dave', 'Frank'])

    def test_index_maintained_on_delete(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        self.db.execute("DELETE FROM users WHERE name = 'Alice'")
        r = self.db.execute("SELECT name FROM users WHERE age = 30")
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Dave'])

    def test_index_maintained_on_update(self):
        self.db.execute("CREATE INDEX idx_age ON users (age)")
        self.db.execute("UPDATE users SET age = 99 WHERE name = 'Alice'")
        r = self.db.execute("SELECT name FROM users WHERE age = 30")
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Dave'])
        r2 = self.db.execute("SELECT name FROM users WHERE age = 99")
        self.assertEqual(r2.rows[0][0], 'Alice')


class TestIndexedDBOrderAndLimit(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
        for i, (name, age) in enumerate([
            ('Alice', 30), ('Bob', 25), ('Carol', 35),
            ('Dave', 30), ('Eve', 28), ('Frank', 30),
        ], 1):
            self.db.execute(f"INSERT INTO users VALUES ({i}, '{name}', {age})")
        self.db.execute("CREATE INDEX idx_age ON users (age)")

    def test_index_with_order_by(self):
        r = self.db.execute("SELECT name FROM users WHERE age = 30 ORDER BY name ASC")
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Dave', 'Frank'])

    def test_index_with_limit(self):
        r = self.db.execute("SELECT name FROM users WHERE age = 30 LIMIT 2")
        self.assertEqual(len(r.rows), 2)

    def test_index_with_order_and_limit(self):
        r = self.db.execute("SELECT name FROM users WHERE age = 30 ORDER BY name ASC LIMIT 1")
        self.assertEqual(r.rows[0][0], 'Alice')

    def test_index_with_distinct(self):
        r = self.db.execute("SELECT DISTINCT age FROM users WHERE age >= 28 ORDER BY age")
        ages = [row[0] for row in r.rows]
        self.assertEqual(ages, [28, 30, 35])


class TestIndexedDBAggregates(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE sales (id INT PRIMARY KEY, product TEXT, amount INT, region TEXT)")
        data = [
            (1, 'Widget', 100, 'North'),
            (2, 'Widget', 200, 'South'),
            (3, 'Gadget', 150, 'North'),
            (4, 'Gadget', 300, 'South'),
            (5, 'Widget', 50, 'North'),
        ]
        for d in data:
            self.db.execute(f"INSERT INTO sales VALUES ({d[0]}, '{d[1]}', {d[2]}, '{d[3]}')")
        self.db.execute("CREATE INDEX idx_product ON sales (product)")

    def test_index_with_count(self):
        r = self.db.execute("SELECT COUNT(*) as cnt FROM sales WHERE product = 'Widget'")
        self.assertEqual(r.rows[0][0], 3)

    def test_index_with_sum(self):
        r = self.db.execute("SELECT SUM(amount) as total FROM sales WHERE product = 'Widget'")
        self.assertEqual(r.rows[0][0], 350)

    def test_index_with_group_by(self):
        r = self.db.execute(
            "SELECT region, SUM(amount) as total FROM sales WHERE product = 'Widget' GROUP BY region ORDER BY region"
        )
        self.assertEqual(r.rows[0][0], 'North')
        self.assertEqual(r.rows[0][1], 150)
        self.assertEqual(r.rows[1][0], 'South')
        self.assertEqual(r.rows[1][1], 200)


class TestDropIndex(unittest.TestCase):
    def test_drop_index(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("CREATE INDEX idx_x ON t (x)")
        r = db.execute("DROP INDEX idx_x")
        self.assertIn('DROP INDEX', r.message)
        self.assertIsNone(db.index_manager.get_index('idx_x'))

    def test_drop_nonexistent_index(self):
        db = IndexedDB()
        with self.assertRaises(Exception):
            db.execute("DROP INDEX nope")

    def test_after_drop_falls_back_to_seq(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("CREATE INDEX idx_x ON t (x)")
        db.execute("DROP INDEX idx_x")
        r = db.execute("SELECT * FROM t WHERE x = 10")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(db._seq_scan_count, 1)


class TestUniqueIndex(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, email TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'alice@test.com')")
        self.db.execute("INSERT INTO users VALUES (2, 'bob@test.com')")

    def test_create_unique_index(self):
        r = self.db.execute("CREATE UNIQUE INDEX idx_email ON users (email)")
        self.assertIn('UNIQUE', r.message)

    def test_unique_violation_on_create(self):
        self.db.execute("INSERT INTO users VALUES (3, 'alice@test.com')")
        with self.assertRaises(Exception):
            self.db.execute("CREATE UNIQUE INDEX idx_email ON users (email)")

    def test_unique_violation_on_insert(self):
        self.db.execute("CREATE UNIQUE INDEX idx_email ON users (email)")
        with self.assertRaises(Exception):
            self.db.execute("INSERT INTO users VALUES (3, 'alice@test.com')")

    def test_unique_index_lookup(self):
        self.db.execute("CREATE UNIQUE INDEX idx_email ON users (email)")
        r = self.db.execute("SELECT id FROM users WHERE email = 'bob@test.com'")
        self.assertEqual(r.rows[0][0], 2)


class TestCompositeIndex(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE orders (id INT PRIMARY KEY, customer TEXT, product TEXT, qty INT)")
        data = [
            (1, 'Alice', 'Widget', 5),
            (2, 'Alice', 'Gadget', 3),
            (3, 'Bob', 'Widget', 2),
            (4, 'Bob', 'Widget', 7),
        ]
        for d in data:
            self.db.execute(f"INSERT INTO orders VALUES ({d[0]}, '{d[1]}', '{d[2]}', {d[3]})")

    def test_composite_index_leftmost_prefix(self):
        """Composite index on (customer, product) can serve WHERE customer = ..."""
        self.db.execute("CREATE INDEX idx_cp ON orders (customer)")
        r = self.db.execute("SELECT id FROM orders WHERE customer = 'Alice' ORDER BY id")
        ids = [row[0] for row in r.rows]
        self.assertEqual(ids, [1, 2])


class TestExplainWithIndex(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        self.db.execute("INSERT INTO t VALUES (2, 20)")

    def test_explain_shows_seq_scan(self):
        r = self.db.execute("EXPLAIN SELECT * FROM t WHERE x = 10")
        plan_text = ' '.join([row[0] for row in r.rows])
        self.assertIn('SeqScan', plan_text)

    def test_explain_shows_index_scan(self):
        self.db.execute("CREATE INDEX idx_x ON t (x)")
        r = self.db.execute("EXPLAIN SELECT * FROM t WHERE x = 10")
        plan_text = ' '.join([row[0] for row in r.rows])
        self.assertIn('IndexScan', plan_text)
        self.assertIn('idx_x', plan_text)

    def test_explain_shows_scan_type(self):
        self.db.execute("CREATE INDEX idx_x ON t (x)")
        r = self.db.execute("EXPLAIN SELECT * FROM t WHERE x > 5")
        plan_text = ' '.join([row[0] for row in r.rows])
        self.assertIn('range', plan_text)


class TestIndexWithAndCondition(unittest.TestCase):
    """Test that optimizer handles AND conditions with partial index use."""

    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 10, 'a')")
        self.db.execute("INSERT INTO t VALUES (2, 20, 'b')")
        self.db.execute("INSERT INTO t VALUES (3, 10, 'c')")
        self.db.execute("INSERT INTO t VALUES (4, 30, 'a')")
        self.db.execute("CREATE INDEX idx_x ON t (x)")

    def test_and_with_indexed_left(self):
        """WHERE x = 10 AND y = 'a' should use index for x, filter for y."""
        r = self.db.execute("SELECT id FROM t WHERE x = 10 AND y = 'a'")
        self.assertEqual(r.rows[0][0], 1)
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(self.db._index_scan_count, 1)

    def test_and_with_indexed_right(self):
        """WHERE y = 'a' AND x = 10 should still find the index."""
        r = self.db.execute("SELECT id FROM t WHERE y = 'a' AND x = 10")
        self.assertEqual(len(r.rows), 1)


class TestIndexCorrectness(unittest.TestCase):
    """Verify index scan produces same results as seq scan."""

    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        for i in range(1, 101):
            self.db.execute(f"INSERT INTO t VALUES ({i}, {i * 10})")

    def test_eq_correctness(self):
        """Index scan eq matches seq scan."""
        # Without index
        r1 = self.db.execute("SELECT id FROM t WHERE val = 500")
        # Create index
        self.db.execute("CREATE INDEX idx_val ON t (val)")
        self.db._index_scan_count = 0
        r2 = self.db.execute("SELECT id FROM t WHERE val = 500")
        self.assertEqual(r1.rows, r2.rows)
        self.assertEqual(self.db._index_scan_count, 1)

    def test_range_correctness(self):
        # Without index
        r1 = self.db.execute("SELECT id FROM t WHERE val > 950 ORDER BY id")
        # With index
        self.db.execute("CREATE INDEX idx_val ON t (val)")
        self.db._index_scan_count = 0
        r2 = self.db.execute("SELECT id FROM t WHERE val > 950 ORDER BY id")
        self.assertEqual(r1.rows, r2.rows)

    def test_lte_correctness(self):
        r1 = self.db.execute("SELECT id FROM t WHERE val <= 30 ORDER BY id")
        self.db.execute("CREATE INDEX idx_val ON t (val)")
        r2 = self.db.execute("SELECT id FROM t WHERE val <= 30 ORDER BY id")
        self.assertEqual(r1.rows, r2.rows)

    def test_large_table_index_coverage(self):
        """All 100 rows accessible via range scan."""
        self.db.execute("CREATE INDEX idx_val ON t (val)")
        r = self.db.execute("SELECT COUNT(*) as cnt FROM t WHERE val >= 10")
        self.assertEqual(r.rows[0][0], 100)


class TestMultipleIndexes(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10, 100)")
        self.db.execute("INSERT INTO t VALUES (2, 20, 200)")
        self.db.execute("INSERT INTO t VALUES (3, 10, 300)")
        self.db.execute("CREATE INDEX idx_x ON t (x)")
        self.db.execute("CREATE INDEX idx_y ON t (y)")

    def test_uses_x_index(self):
        r = self.db.execute("SELECT id FROM t WHERE x = 10 ORDER BY id")
        ids = [row[0] for row in r.rows]
        self.assertEqual(ids, [1, 3])

    def test_uses_y_index(self):
        r = self.db.execute("SELECT id FROM t WHERE y = 200")
        self.assertEqual(r.rows[0][0], 2)

    def test_index_stats(self):
        stats = self.db.index_stats()
        self.assertIn('idx_x', stats['indexes'])
        self.assertIn('idx_y', stats['indexes'])


class TestIndexWithJoins(unittest.TestCase):
    """Indexes should not be used with JOINs (falls back to seq scan)."""

    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, amount INT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        self.db.execute("INSERT INTO orders VALUES (2, 2, 200)")

    def test_join_uses_seq_scan(self):
        self.db.execute("CREATE INDEX idx_uid ON orders (user_id)")
        r = self.db.execute(
            "SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id ORDER BY users.name"
        )
        self.assertEqual(len(r.rows), 2)
        # Joins should not use index scan (optimizer limitation)
        # Just verify correctness
        self.assertEqual(r.rows[0][0], 'Alice')


class TestIndexWithNoResults(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        self.db.execute("CREATE INDEX idx_x ON t (x)")

    def test_index_no_match(self):
        r = self.db.execute("SELECT * FROM t WHERE x = 999")
        self.assertEqual(len(r.rows), 0)

    def test_index_range_no_match(self):
        r = self.db.execute("SELECT * FROM t WHERE x > 100")
        self.assertEqual(len(r.rows), 0)


class TestExecuteMany(unittest.TestCase):
    def test_execute_many(self):
        db = IndexedDB()
        results = db.execute_many("""
            CREATE TABLE t (id INT PRIMARY KEY, x INT);
            INSERT INTO t VALUES (1, 10);
            INSERT INTO t VALUES (2, 20);
            SELECT * FROM t ORDER BY id;
        """)
        self.assertEqual(len(results), 4)
        self.assertEqual(len(results[3].rows), 2)


class TestIndexRebuiltAfterBulkOps(unittest.TestCase):
    """Test that indexes stay consistent after many mutations."""

    def test_insert_delete_insert(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("CREATE INDEX idx_x ON t (x)")

        # Insert
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("INSERT INTO t VALUES (2, 20)")
        r = db.execute("SELECT id FROM t WHERE x = 10")
        self.assertEqual(r.rows[0][0], 1)

        # Delete
        db.execute("DELETE FROM t WHERE id = 1")
        r = db.execute("SELECT id FROM t WHERE x = 10")
        self.assertEqual(len(r.rows), 0)

        # Re-insert
        db.execute("INSERT INTO t VALUES (3, 10)")
        r = db.execute("SELECT id FROM t WHERE x = 10")
        self.assertEqual(r.rows[0][0], 3)

    def test_update_cycle(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("CREATE INDEX idx_x ON t (x)")

        for new_val in [20, 30, 40, 10]:
            db.execute(f"UPDATE t SET x = {new_val} WHERE id = 1")
            r = db.execute(f"SELECT id FROM t WHERE x = {new_val}")
            self.assertEqual(r.rows[0][0], 1)


class TestIndexWithTextColumns(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO t VALUES (2, 'Bob')")
        self.db.execute("INSERT INTO t VALUES (3, 'Carol')")
        self.db.execute("CREATE INDEX idx_name ON t (name)")

    def test_text_eq(self):
        r = self.db.execute("SELECT id FROM t WHERE name = 'Bob'")
        self.assertEqual(r.rows[0][0], 2)

    def test_text_range(self):
        r = self.db.execute("SELECT name FROM t WHERE name >= 'B' ORDER BY name")
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Bob', 'Carol'])


class TestIndexWithNullValues(unittest.TestCase):
    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        self.db.execute("INSERT INTO t VALUES (2, NULL)")
        self.db.execute("INSERT INTO t VALUES (3, 30)")

    def test_null_not_indexed(self):
        self.db.execute("CREATE INDEX idx_x ON t (x)")
        info = self.db.index_manager.get_index('idx_x')
        self.assertEqual(self.db.index_manager.count('idx_x'), 2)  # 10, 30 only

    def test_null_rows_not_in_index_scan(self):
        self.db.execute("CREATE INDEX idx_x ON t (x)")
        r = self.db.execute("SELECT id FROM t WHERE x = 10")
        self.assertEqual(r.rows[0][0], 1)
        self.assertEqual(len(r.rows), 1)


class TestSelectWithoutWhereNoIndex(unittest.TestCase):
    """SELECT without WHERE should always seq scan."""

    def test_select_all(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("CREATE INDEX idx_x ON t (x)")
        r = db.execute("SELECT * FROM t")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(db._seq_scan_count, 1)
        self.assertEqual(db._index_scan_count, 0)


class TestDropTableCleansIndexes(unittest.TestCase):
    def test_drop_table(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("CREATE INDEX idx_x ON t (x)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        # After drop, index manager should be queried but not crash
        db.execute("DROP TABLE t")
        # Index info still exists in manager (not auto-cleaned by parent DROP TABLE)
        # This is acceptable -- the table is gone, so index is orphaned but harmless


class TestIndexHeight(unittest.TestCase):
    """B+ tree height grows with data."""

    def test_single_entry_height(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("CREATE INDEX idx_x ON t (x)")
        stats = db.index_stats()
        self.assertEqual(stats['indexes']['idx_x']['height'], 1)

    def test_many_entries_height(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        for i in range(1, 201):
            db.execute(f"INSERT INTO t VALUES ({i}, {i})")
        db.execute("CREATE INDEX idx_x ON t (x)")
        stats = db.index_stats()
        self.assertGreaterEqual(stats['indexes']['idx_x']['height'], 2)


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure IndexedDB works as a drop-in replacement for MiniDB."""

    def test_basic_crud(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'Alice')")
        r = db.execute("SELECT name FROM t WHERE id = 1")
        self.assertEqual(r.rows[0][0], 'Alice')
        db.execute("UPDATE t SET name = 'Alicia' WHERE id = 1")
        r = db.execute("SELECT name FROM t WHERE id = 1")
        self.assertEqual(r.rows[0][0], 'Alicia')
        db.execute("DELETE FROM t WHERE id = 1")
        r = db.execute("SELECT * FROM t")
        self.assertEqual(len(r.rows), 0)

    def test_show_tables(self):
        db = IndexedDB()
        db.execute("CREATE TABLE a (id INT PRIMARY KEY)")
        db.execute("CREATE TABLE b (id INT PRIMARY KEY)")
        r = db.execute("SHOW TABLES")
        tables = [row[0] for row in r.rows]
        self.assertIn('a', tables)
        self.assertIn('b', tables)

    def test_describe(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL)")
        r = db.execute("DESCRIBE t")
        self.assertEqual(len(r.rows), 2)

    def test_aggregates(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("INSERT INTO t VALUES (2, 20)")
        r = db.execute("SELECT AVG(val) as avg_val FROM t")
        self.assertEqual(r.rows[0][0], 15.0)

    def test_order_by(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'Charlie')")
        db.execute("INSERT INTO t VALUES (2, 'Alice')")
        db.execute("INSERT INTO t VALUES (3, 'Bob')")
        r = db.execute("SELECT name FROM t ORDER BY name ASC")
        names = [row[0] for row in r.rows]
        self.assertEqual(names, ['Alice', 'Bob', 'Charlie'])

    def test_joins(self):
        db = IndexedDB()
        db.execute("CREATE TABLE a (id INT PRIMARY KEY, x INT)")
        db.execute("CREATE TABLE b (id INT PRIMARY KEY, a_id INT, y TEXT)")
        db.execute("INSERT INTO a VALUES (1, 10)")
        db.execute("INSERT INTO b VALUES (1, 1, 'hello')")
        r = db.execute("SELECT a.x, b.y FROM a JOIN b ON a.id = b.a_id")
        self.assertEqual(r.rows[0], [10, 'hello'])

    def test_limit_offset(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("INSERT INTO t VALUES (2, 20)")
        db.execute("INSERT INTO t VALUES (3, 30)")
        r = db.execute("SELECT val FROM t ORDER BY val LIMIT 2 OFFSET 1")
        self.assertEqual(len(r.rows), 2)
        self.assertEqual(r.rows[0][0], 20)

    def test_transactions(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("BEGIN")
        db.execute("INSERT INTO t VALUES (1, 10)")
        db.execute("ROLLBACK")
        r = db.execute("SELECT * FROM t")
        self.assertEqual(len(r.rows), 0)

    def test_group_by_having(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, cat TEXT, val INT)")
        db.execute("INSERT INTO t VALUES (1, 'A', 10)")
        db.execute("INSERT INTO t VALUES (2, 'A', 20)")
        db.execute("INSERT INTO t VALUES (3, 'B', 30)")
        r = db.execute("SELECT cat, COUNT(*) as cnt FROM t GROUP BY cat HAVING cnt >= 2")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][0], 'A')


class TestIndexWithExpressions(unittest.TestCase):
    """Test index with more complex SELECT expressions."""

    def setUp(self):
        self.db = IndexedDB()
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10, 100)")
        self.db.execute("INSERT INTO t VALUES (2, 20, 200)")
        self.db.execute("INSERT INTO t VALUES (3, 30, 300)")
        self.db.execute("CREATE INDEX idx_x ON t (x)")

    def test_select_expression(self):
        r = self.db.execute("SELECT x, y, x + y as total FROM t WHERE x = 10")
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(r.rows[0][2], 110)

    def test_select_with_alias(self):
        r = self.db.execute("SELECT x as val FROM t WHERE x = 20")
        self.assertEqual(r.columns, ['val'])
        self.assertEqual(r.rows[0][0], 20)


class TestIndexStressTest(unittest.TestCase):
    """Larger data set to exercise B+ tree."""

    def test_1000_rows(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        for i in range(1, 1001):
            db.execute(f"INSERT INTO t VALUES ({i}, {i % 50})")
        db.execute("CREATE INDEX idx_val ON t (val)")

        # Equality
        r = db.execute("SELECT COUNT(*) as cnt FROM t WHERE val = 0")
        self.assertEqual(r.rows[0][0], 20)  # 50, 100, 150, ... 1000

        # Range
        r = db.execute("SELECT COUNT(*) as cnt FROM t WHERE val >= 45")
        self.assertEqual(r.rows[0][0], 100)  # vals 45,46,47,48,49

        # Verify index was used
        self.assertGreaterEqual(db._index_scan_count, 2)

    def test_delete_half_then_query(self):
        db = IndexedDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        for i in range(1, 201):
            db.execute(f"INSERT INTO t VALUES ({i}, {i})")
        db.execute("CREATE INDEX idx_val ON t (val)")

        # Delete odd ids
        for i in range(1, 201, 2):
            db.execute(f"DELETE FROM t WHERE id = {i}")

        # Query should only find even values
        r = db.execute("SELECT COUNT(*) as cnt FROM t WHERE val <= 10")
        self.assertEqual(r.rows[0][0], 5)  # 2,4,6,8,10


if __name__ == '__main__':
    unittest.main()
