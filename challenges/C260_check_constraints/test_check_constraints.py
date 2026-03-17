"""
Tests for C260: CHECK Constraints
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from check_constraints import (
    CheckConstraint, CheckTableSchema, CheckEvaluator, CheckDB,
    AlterAddCheckStmt, AlterDropConstraintStmt,
    _extract_columns, _expr_to_sql, parse_check_sql,
)
from mini_database import (
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlIsNull, SqlBetween, SqlInList,
    CatalogError, CompileError, CreateTableStmt,
)


# =============================================================================
# CheckConstraint Dataclass Tests
# =============================================================================

class TestCheckConstraint(unittest.TestCase):
    def test_basic_creation(self):
        cc = CheckConstraint(name='positive', expr=None, columns=['x'], source='x > 0')
        self.assertEqual(cc.name, 'positive')
        self.assertEqual(cc.columns, ['x'])
        self.assertEqual(cc.source, 'x > 0')

    def test_unnamed(self):
        cc = CheckConstraint(name=None, expr=None, columns=['x'])
        self.assertIsNone(cc.name)

    def test_repr(self):
        cc = CheckConstraint(name='pos', expr=None, columns=['x'], source='x > 0')
        self.assertIn('pos', repr(cc))
        self.assertIn('x > 0', repr(cc))

    def test_repr_unnamed(self):
        cc = CheckConstraint(name=None, expr=None, columns=[], source='a = b')
        self.assertIn('unnamed', repr(cc))


# =============================================================================
# Column Extraction Tests
# =============================================================================

class TestExtractColumns(unittest.TestCase):
    def test_column_ref(self):
        expr = SqlColumnRef(None, 'x')
        self.assertEqual(_extract_columns(expr), ['x'])

    def test_comparison(self):
        expr = SqlComparison('>', SqlColumnRef(None, 'age'), SqlLiteral(18))
        self.assertEqual(_extract_columns(expr), ['age'])

    def test_logic_and(self):
        expr = SqlLogic('and', [
            SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0)),
            SqlComparison('<', SqlColumnRef(None, 'y'), SqlLiteral(100)),
        ])
        self.assertEqual(_extract_columns(expr), ['x', 'y'])

    def test_binop(self):
        expr = SqlBinOp('+', SqlColumnRef(None, 'a'), SqlColumnRef(None, 'b'))
        self.assertEqual(_extract_columns(expr), ['a', 'b'])

    def test_unique_columns(self):
        expr = SqlLogic('and', [
            SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0)),
            SqlComparison('<', SqlColumnRef(None, 'x'), SqlLiteral(100)),
        ])
        self.assertEqual(_extract_columns(expr), ['x'])

    def test_is_null(self):
        expr = SqlIsNull(SqlColumnRef(None, 'name'), negated=True)
        self.assertEqual(_extract_columns(expr), ['name'])

    def test_between(self):
        expr = SqlBetween(SqlColumnRef(None, 'age'), SqlLiteral(0), SqlLiteral(120))
        self.assertEqual(_extract_columns(expr), ['age'])

    def test_in_list(self):
        expr = SqlInList(SqlColumnRef(None, 'status'), [SqlLiteral('A'), SqlLiteral('B')])
        self.assertEqual(_extract_columns(expr), ['status'])

    def test_literal_only(self):
        expr = SqlLiteral(42)
        self.assertEqual(_extract_columns(expr), [])


# =============================================================================
# Expression to SQL Tests
# =============================================================================

class TestExprToSql(unittest.TestCase):
    def test_column_ref(self):
        self.assertEqual(_expr_to_sql(SqlColumnRef(None, 'x')), 'x')

    def test_qualified_column(self):
        self.assertEqual(_expr_to_sql(SqlColumnRef('t', 'x')), 't.x')

    def test_literal_int(self):
        self.assertEqual(_expr_to_sql(SqlLiteral(42)), '42')

    def test_literal_string(self):
        self.assertEqual(_expr_to_sql(SqlLiteral('hello')), "'hello'")

    def test_literal_null(self):
        self.assertEqual(_expr_to_sql(SqlLiteral(None)), 'NULL')

    def test_literal_bool(self):
        self.assertEqual(_expr_to_sql(SqlLiteral(True)), 'TRUE')
        self.assertEqual(_expr_to_sql(SqlLiteral(False)), 'FALSE')

    def test_comparison(self):
        expr = SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0))
        self.assertEqual(_expr_to_sql(expr), 'x > 0')

    def test_logic_and(self):
        expr = SqlLogic('and', [
            SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0)),
            SqlComparison('<', SqlColumnRef(None, 'x'), SqlLiteral(100)),
        ])
        self.assertEqual(_expr_to_sql(expr), '(x > 0 AND x < 100)')

    def test_logic_not(self):
        expr = SqlLogic('not', [SqlComparison('=', SqlColumnRef(None, 'x'), SqlLiteral(0))])
        self.assertEqual(_expr_to_sql(expr), 'NOT x = 0')

    def test_is_null(self):
        expr = SqlIsNull(SqlColumnRef(None, 'x'), negated=False)
        self.assertEqual(_expr_to_sql(expr), 'x IS NULL')

    def test_is_not_null(self):
        expr = SqlIsNull(SqlColumnRef(None, 'x'), negated=True)
        self.assertEqual(_expr_to_sql(expr), 'x IS NOT NULL')

    def test_between(self):
        expr = SqlBetween(SqlColumnRef(None, 'age'), SqlLiteral(18), SqlLiteral(65))
        self.assertEqual(_expr_to_sql(expr), 'age BETWEEN 18 AND 65')

    def test_in_list(self):
        expr = SqlInList(SqlColumnRef(None, 'x'), [SqlLiteral(1), SqlLiteral(2), SqlLiteral(3)])
        self.assertEqual(_expr_to_sql(expr), 'x IN (1, 2, 3)')

    def test_binop(self):
        expr = SqlBinOp('+', SqlColumnRef(None, 'a'), SqlColumnRef(None, 'b'))
        self.assertEqual(_expr_to_sql(expr), '(a + b)')


# =============================================================================
# CheckTableSchema Tests
# =============================================================================

class TestCheckTableSchema(unittest.TestCase):
    def _make_schema(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(
            name='test',
            columns=[ColumnDef('id', 'int', primary_key=True),
                     ColumnDef('value', 'int')],
        )
        return CheckTableSchema(base_schema=base)

    def test_basic_properties(self):
        cs = self._make_schema()
        self.assertEqual(cs.name, 'test')
        self.assertEqual(len(cs.columns), 2)
        self.assertEqual(cs.column_names(), ['id', 'value'])
        self.assertEqual(cs.primary_key_column(), 'id')

    def test_add_check(self):
        cs = self._make_schema()
        cc = CheckConstraint(name='pos', expr=None, columns=['value'], source='value > 0')
        cs.add_check(cc)
        self.assertEqual(len(cs.check_constraints), 1)
        self.assertEqual(cs.check_constraints[0].name, 'pos')

    def test_add_check_auto_name(self):
        cs = self._make_schema()
        cc = CheckConstraint(name=None, expr=None, columns=['value'])
        cs.add_check(cc)
        self.assertIsNotNone(cc.name)
        self.assertIn('test_check_', cc.name)

    def test_add_duplicate_name_raises(self):
        cs = self._make_schema()
        cc1 = CheckConstraint(name='pos', expr=None, columns=['value'])
        cc2 = CheckConstraint(name='pos', expr=None, columns=['value'])
        cs.add_check(cc1)
        with self.assertRaises(CatalogError):
            cs.add_check(cc2)

    def test_drop_check(self):
        cs = self._make_schema()
        cc = CheckConstraint(name='pos', expr=None, columns=['value'])
        cs.add_check(cc)
        cs.drop_check('pos')
        self.assertEqual(len(cs.check_constraints), 0)

    def test_drop_nonexistent_raises(self):
        cs = self._make_schema()
        with self.assertRaises(CatalogError):
            cs.drop_check('nope')

    def test_get_check(self):
        cs = self._make_schema()
        cc = CheckConstraint(name='pos', expr=None, columns=['value'])
        cs.add_check(cc)
        self.assertIsNotNone(cs.get_check('pos'))
        self.assertIsNone(cs.get_check('nope'))

    def test_next_rowid_passthrough(self):
        cs = self._make_schema()
        cs.next_rowid = 42
        self.assertEqual(cs.next_rowid, 42)
        self.assertEqual(cs.base_schema.next_rowid, 42)

    def test_get_column(self):
        cs = self._make_schema()
        self.assertIsNotNone(cs.get_column('id'))
        self.assertIsNone(cs.get_column('nope'))

    def test_indexes(self):
        cs = self._make_schema()
        self.assertIsInstance(cs.indexes, dict)


# =============================================================================
# CheckEvaluator Tests
# =============================================================================

class TestCheckEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = CheckEvaluator()

    def test_simple_gt(self):
        expr = SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0))
        cc = CheckConstraint(name='pos', expr=expr, columns=['x'])
        # Should pass
        self.evaluator.validate_row('t', {'x': 5}, [cc])
        # Should fail
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': -1}, [cc])

    def test_equality(self):
        expr = SqlComparison('=', SqlColumnRef(None, 'status'), SqlLiteral('active'))
        cc = CheckConstraint(name='active_only', expr=expr, columns=['status'])
        self.evaluator.validate_row('t', {'status': 'active'}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'status': 'inactive'}, [cc])

    def test_and_logic(self):
        expr = SqlLogic('and', [
            SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0)),
            SqlComparison('<', SqlColumnRef(None, 'x'), SqlLiteral(100)),
        ])
        cc = CheckConstraint(name='range', expr=expr, columns=['x'])
        self.evaluator.validate_row('t', {'x': 50}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': 150}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': -5}, [cc])

    def test_or_logic(self):
        expr = SqlLogic('or', [
            SqlComparison('=', SqlColumnRef(None, 'x'), SqlLiteral(1)),
            SqlComparison('=', SqlColumnRef(None, 'x'), SqlLiteral(2)),
        ])
        cc = CheckConstraint(name='one_or_two', expr=expr, columns=['x'])
        self.evaluator.validate_row('t', {'x': 1}, [cc])
        self.evaluator.validate_row('t', {'x': 2}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': 3}, [cc])

    def test_not_logic(self):
        expr = SqlLogic('not', [
            SqlComparison('=', SqlColumnRef(None, 'x'), SqlLiteral(0)),
        ])
        cc = CheckConstraint(name='nonzero', expr=expr, columns=['x'])
        self.evaluator.validate_row('t', {'x': 5}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': 0}, [cc])

    def test_null_satisfies_check(self):
        """Per SQL standard, NULL satisfies CHECK (not FALSE)."""
        expr = SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0))
        cc = CheckConstraint(name='pos', expr=expr, columns=['x'])
        # NULL comparison returns NULL, which satisfies CHECK
        self.evaluator.validate_row('t', {'x': None}, [cc])

    def test_multi_column(self):
        expr = SqlComparison('<', SqlColumnRef(None, 'start_date'),
                            SqlColumnRef(None, 'end_date'))
        cc = CheckConstraint(name='date_order', expr=expr, columns=['start_date', 'end_date'])
        self.evaluator.validate_row('t', {'start_date': 1, 'end_date': 10}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'start_date': 10, 'end_date': 1}, [cc])

    def test_between(self):
        expr = SqlBetween(SqlColumnRef(None, 'age'), SqlLiteral(0), SqlLiteral(150))
        cc = CheckConstraint(name='valid_age', expr=expr, columns=['age'])
        self.evaluator.validate_row('t', {'age': 25}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'age': 200}, [cc])

    def test_in_list(self):
        expr = SqlInList(SqlColumnRef(None, 'grade'), [SqlLiteral('A'), SqlLiteral('B'), SqlLiteral('C')])
        cc = CheckConstraint(name='valid_grade', expr=expr, columns=['grade'])
        self.evaluator.validate_row('t', {'grade': 'A'}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'grade': 'F'}, [cc])

    def test_is_not_null(self):
        expr = SqlIsNull(SqlColumnRef(None, 'name'), negated=True)
        cc = CheckConstraint(name='name_required', expr=expr, columns=['name'])
        self.evaluator.validate_row('t', {'name': 'Alice'}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'name': None}, [cc])

    def test_arithmetic_expr(self):
        # CHECK (x + y > 10)
        expr = SqlComparison('>',
            SqlBinOp('+', SqlColumnRef(None, 'x'), SqlColumnRef(None, 'y')),
            SqlLiteral(10))
        cc = CheckConstraint(name='sum_check', expr=expr, columns=['x', 'y'])
        self.evaluator.validate_row('t', {'x': 6, 'y': 6}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': 3, 'y': 3}, [cc])

    def test_multiple_constraints(self):
        c1 = CheckConstraint(name='pos', expr=SqlComparison('>', SqlColumnRef(None, 'x'), SqlLiteral(0)), columns=['x'])
        c2 = CheckConstraint(name='lt100', expr=SqlComparison('<', SqlColumnRef(None, 'x'), SqlLiteral(100)), columns=['x'])
        self.evaluator.validate_row('t', {'x': 50}, [c1, c2])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': -1}, [c1, c2])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': 150}, [c1, c2])

    def test_gte_and_lte(self):
        expr = SqlLogic('and', [
            SqlComparison('>=', SqlColumnRef(None, 'x'), SqlLiteral(1)),
            SqlComparison('<=', SqlColumnRef(None, 'x'), SqlLiteral(5)),
        ])
        cc = CheckConstraint(name='range', expr=expr, columns=['x'])
        self.evaluator.validate_row('t', {'x': 1}, [cc])
        self.evaluator.validate_row('t', {'x': 5}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': 0}, [cc])

    def test_ne_operator(self):
        expr = SqlComparison('!=', SqlColumnRef(None, 'x'), SqlLiteral(0))
        cc = CheckConstraint(name='nonzero', expr=expr, columns=['x'])
        self.evaluator.validate_row('t', {'x': 5}, [cc])
        with self.assertRaises(CatalogError):
            self.evaluator.validate_row('t', {'x': 0}, [cc])


# =============================================================================
# Parser Tests
# =============================================================================

class TestCheckParser(unittest.TestCase):
    def test_column_level_check(self):
        stmt = parse_check_sql("CREATE TABLE t (x INT CHECK (x > 0))")
        self.assertIsInstance(stmt, CreateTableStmt)
        self.assertTrue(hasattr(stmt, '_col_checks'))
        self.assertEqual(len(stmt._col_checks), 1)
        self.assertIsNotNone(stmt._col_checks[0])

    def test_table_level_check(self):
        stmt = parse_check_sql("CREATE TABLE t (x INT, y INT, CHECK (x < y))")
        self.assertTrue(hasattr(stmt, '_table_checks'))
        self.assertEqual(len(stmt._table_checks), 1)

    def test_named_constraint(self):
        stmt = parse_check_sql(
            "CREATE TABLE t (x INT, CONSTRAINT positive_x CHECK (x > 0))"
        )
        self.assertEqual(len(stmt._table_checks), 1)
        self.assertEqual(stmt._table_checks[0].name, 'positive_x')

    def test_multiple_checks(self):
        stmt = parse_check_sql("""
            CREATE TABLE t (
                x INT CHECK (x > 0),
                y INT CHECK (y > 0),
                CHECK (x < y)
            )
        """)
        self.assertEqual(len(stmt._col_checks), 2)
        self.assertEqual(len(stmt._table_checks), 1)

    def test_check_with_and(self):
        stmt = parse_check_sql("CREATE TABLE t (x INT CHECK (x > 0 AND x < 100))")
        self.assertEqual(len(stmt._col_checks), 1)
        check = stmt._col_checks[0]
        self.assertIsInstance(check.expr, SqlLogic)

    def test_check_with_between(self):
        stmt = parse_check_sql("CREATE TABLE t (age INT CHECK (age BETWEEN 0 AND 150))")
        check = stmt._col_checks[0]
        self.assertIsInstance(check.expr, SqlBetween)

    def test_check_with_in(self):
        stmt = parse_check_sql("CREATE TABLE t (grade TEXT CHECK (grade IN ('A', 'B', 'C')))")
        check = stmt._col_checks[0]
        self.assertIsInstance(check.expr, SqlInList)

    def test_alter_add_check(self):
        stmt = parse_check_sql("ALTER TABLE t ADD CHECK (x > 0)")
        self.assertIsInstance(stmt, AlterAddCheckStmt)
        self.assertEqual(stmt.table_name, 't')

    def test_alter_add_named_constraint(self):
        stmt = parse_check_sql("ALTER TABLE t ADD CONSTRAINT pos CHECK (x > 0)")
        self.assertIsInstance(stmt, AlterAddCheckStmt)
        self.assertEqual(stmt.constraint.name, 'pos')

    def test_alter_drop_constraint(self):
        stmt = parse_check_sql("ALTER TABLE t DROP CONSTRAINT pos")
        self.assertIsInstance(stmt, AlterDropConstraintStmt)
        self.assertEqual(stmt.constraint_name, 'pos')

    def test_if_not_exists_with_check(self):
        stmt = parse_check_sql(
            "CREATE TABLE IF NOT EXISTS t (x INT CHECK (x > 0))"
        )
        self.assertTrue(stmt.if_not_exists)
        self.assertEqual(len(stmt._col_checks), 1)

    def test_primary_key_with_check(self):
        stmt = parse_check_sql(
            "CREATE TABLE t (id INT PRIMARY KEY, val INT CHECK (val >= 0))"
        )
        self.assertTrue(stmt.columns[0].primary_key)
        self.assertEqual(len(stmt._col_checks), 1)

    def test_check_with_not(self):
        stmt = parse_check_sql("CREATE TABLE t (x INT CHECK (NOT x = 0))")
        check = stmt._col_checks[0]
        self.assertIsInstance(check.expr, SqlLogic)
        self.assertEqual(check.expr.op, 'not')

    def test_check_with_is_null(self):
        stmt = parse_check_sql("CREATE TABLE t (x INT CHECK (x IS NOT NULL))")
        check = stmt._col_checks[0]
        self.assertIsInstance(check.expr, SqlIsNull)

    def test_check_with_arithmetic(self):
        stmt = parse_check_sql("CREATE TABLE t (x INT, y INT, CHECK (x + y > 10))")
        self.assertEqual(len(stmt._table_checks), 1)


# =============================================================================
# Integration: CheckDB INSERT Tests
# =============================================================================

class TestCheckDBInsert(unittest.TestCase):
    def test_insert_satisfies_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE items (id INT PRIMARY KEY, price INT CHECK (price > 0))")
        db.execute("INSERT INTO items (id, price) VALUES (1, 10)")
        result = db.execute("SELECT * FROM items")
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0][1], 10)

    def test_insert_violates_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE items (id INT PRIMARY KEY, price INT CHECK (price > 0))")
        with self.assertRaises(CatalogError) as ctx:
            db.execute("INSERT INTO items (id, price) VALUES (1, -5)")
        self.assertIn('CHECK', str(ctx.exception))

    def test_insert_boundary_value(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x >= 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 0)")
        result = db.execute("SELECT x FROM t")
        self.assertEqual(result.rows[0][0], 0)

    def test_insert_multi_column_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE ranges (id INT PRIMARY KEY, lo INT, hi INT, CHECK (lo < hi))")
        db.execute("INSERT INTO ranges (id, lo, hi) VALUES (1, 1, 10)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO ranges (id, lo, hi) VALUES (2, 10, 5)")

    def test_insert_multiple_checks(self):
        db = CheckDB()
        db.execute("""
            CREATE TABLE products (
                id INT PRIMARY KEY,
                price INT CHECK (price > 0),
                quantity INT CHECK (quantity >= 0)
            )
        """)
        db.execute("INSERT INTO products (id, price, quantity) VALUES (1, 10, 5)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO products (id, price, quantity) VALUES (2, -1, 5)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO products (id, price, quantity) VALUES (3, 10, -1)")

    def test_insert_named_constraint(self):
        db = CheckDB()
        db.execute("""
            CREATE TABLE t (
                id INT PRIMARY KEY,
                x INT,
                CONSTRAINT positive_x CHECK (x > 0)
            )
        """)
        db.execute("INSERT INTO t (id, x) VALUES (1, 5)")
        with self.assertRaises(CatalogError) as ctx:
            db.execute("INSERT INTO t (id, x) VALUES (2, -1)")
        self.assertIn('positive_x', str(ctx.exception))

    def test_insert_check_with_between(self):
        db = CheckDB()
        db.execute("CREATE TABLE ages (id INT PRIMARY KEY, age INT CHECK (age BETWEEN 0 AND 150))")
        db.execute("INSERT INTO ages (id, age) VALUES (1, 25)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO ages (id, age) VALUES (2, 200)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO ages (id, age) VALUES (3, -1)")

    def test_insert_check_with_in(self):
        db = CheckDB()
        db.execute("CREATE TABLE grades (id INT PRIMARY KEY, grade TEXT CHECK (grade IN ('A', 'B', 'C')))")
        db.execute("INSERT INTO grades (id, grade) VALUES (1, 'A')")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO grades (id, grade) VALUES (2, 'F')")

    def test_insert_null_satisfies_check(self):
        """SQL standard: NULL satisfies CHECK."""
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id) VALUES (1)")
        result = db.execute("SELECT x FROM t")
        self.assertIsNone(result.rows[0][0])

    def test_insert_multiple_rows(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("INSERT INTO t (id, x) VALUES (2, 20)")
        result = db.execute("SELECT COUNT(*) FROM t")
        self.assertEqual(result.rows[0][0], 2)

    def test_insert_rollback_on_violation(self):
        """Row should not be inserted if CHECK fails."""
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        try:
            db.execute("INSERT INTO t (id, x) VALUES (1, -5)")
        except CatalogError:
            pass
        result = db.execute("SELECT COUNT(*) FROM t")
        self.assertEqual(result.rows[0][0], 0)

    def test_insert_check_with_arithmetic(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y INT, CHECK (x + y > 10))")
        db.execute("INSERT INTO t (id, x, y) VALUES (1, 6, 6)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, x, y) VALUES (2, 3, 3)")


# =============================================================================
# Integration: CheckDB UPDATE Tests
# =============================================================================

class TestCheckDBUpdate(unittest.TestCase):
    def test_update_satisfies_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("UPDATE t SET x = 20 WHERE id = 1")
        result = db.execute("SELECT x FROM t WHERE id = 1")
        self.assertEqual(result.rows[0][0], 20)

    def test_update_violates_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        with self.assertRaises(CatalogError):
            db.execute("UPDATE t SET x = -5 WHERE id = 1")

    def test_update_preserves_original_on_violation(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        try:
            db.execute("UPDATE t SET x = -5 WHERE id = 1")
        except CatalogError:
            pass
        result = db.execute("SELECT x FROM t WHERE id = 1")
        self.assertEqual(result.rows[0][0], 10)

    def test_update_multi_column_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, lo INT, hi INT, CHECK (lo < hi))")
        db.execute("INSERT INTO t (id, lo, hi) VALUES (1, 1, 10)")
        db.execute("UPDATE t SET lo = 5 WHERE id = 1")
        with self.assertRaises(CatalogError):
            db.execute("UPDATE t SET lo = 20 WHERE id = 1")

    def test_update_with_expression(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("UPDATE t SET x = x + 5 WHERE id = 1")
        result = db.execute("SELECT x FROM t WHERE id = 1")
        self.assertEqual(result.rows[0][0], 15)

    def test_update_no_matching_rows(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("UPDATE t SET x = 20 WHERE id = 999")
        result = db.execute("SELECT x FROM t WHERE id = 1")
        self.assertEqual(result.rows[0][0], 10)


# =============================================================================
# Integration: ALTER TABLE Tests
# =============================================================================

class TestCheckDBAlter(unittest.TestCase):
    def test_alter_add_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("ALTER TABLE t ADD CHECK (x > 0)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, x) VALUES (2, -5)")

    def test_alter_add_check_validates_existing(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t (id, x) VALUES (1, -5)")
        with self.assertRaises(CatalogError):
            db.execute("ALTER TABLE t ADD CHECK (x > 0)")

    def test_alter_add_named_constraint(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("ALTER TABLE t ADD CONSTRAINT positive CHECK (x > 0)")
        constraints = db.get_constraints('t')
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].name, 'positive')

    def test_alter_drop_constraint(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("ALTER TABLE t ADD CONSTRAINT pos CHECK (x > 0)")
        db.execute("ALTER TABLE t DROP CONSTRAINT pos")
        constraints = db.get_constraints('t')
        self.assertEqual(len(constraints), 0)
        # Should now allow negative values
        db.execute("INSERT INTO t (id, x) VALUES (1, -5)")

    def test_alter_drop_nonexistent(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        with self.assertRaises(CatalogError):
            db.execute("ALTER TABLE t DROP CONSTRAINT nope")

    def test_alter_add_multiple_checks(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y INT)")
        db.execute("ALTER TABLE t ADD CONSTRAINT pos_x CHECK (x > 0)")
        db.execute("ALTER TABLE t ADD CONSTRAINT pos_y CHECK (y > 0)")
        constraints = db.get_constraints('t')
        self.assertEqual(len(constraints), 2)

    def test_alter_add_duplicate_name(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("ALTER TABLE t ADD CONSTRAINT pos CHECK (x > 0)")
        with self.assertRaises(CatalogError):
            db.execute("ALTER TABLE t ADD CONSTRAINT pos CHECK (x > 0)")


# =============================================================================
# Integration: Constraint Introspection Tests
# =============================================================================

class TestCheckDBIntrospection(unittest.TestCase):
    def test_get_constraints_empty(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
        self.assertEqual(db.get_constraints('t'), [])

    def test_get_constraints_with_checks(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        constraints = db.get_constraints('t')
        self.assertEqual(len(constraints), 1)

    def test_describe_constraints(self):
        db = CheckDB()
        db.execute("""
            CREATE TABLE t (
                id INT PRIMARY KEY,
                x INT,
                CONSTRAINT pos CHECK (x > 0)
            )
        """)
        result = db.describe_constraints('t')
        self.assertEqual(result.columns, ['constraint_name', 'expression', 'columns'])
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0][0], 'pos')
        self.assertIn('x', result.rows[0][2])

    def test_describe_empty(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
        result = db.describe_constraints('t')
        self.assertEqual(len(result.rows), 0)


# =============================================================================
# Integration: Complex Expression Tests
# =============================================================================

class TestComplexExpressions(unittest.TestCase):
    def test_nested_and_or(self):
        db = CheckDB()
        db.execute("""
            CREATE TABLE t (
                id INT PRIMARY KEY,
                x INT,
                CHECK (x > 0 AND (x < 50 OR x > 100))
            )
        """)
        db.execute("INSERT INTO t (id, x) VALUES (1, 25)")
        db.execute("INSERT INTO t (id, x) VALUES (2, 150)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, x) VALUES (3, 75)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, x) VALUES (4, -1)")

    def test_check_with_subtraction(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y INT, CHECK (x - y >= 0))")
        db.execute("INSERT INTO t (id, x, y) VALUES (1, 10, 5)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, x, y) VALUES (2, 3, 10)")

    def test_check_with_multiplication(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y INT, CHECK (x * y > 0))")
        db.execute("INSERT INTO t (id, x, y) VALUES (1, 3, 4)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, x, y) VALUES (2, 3, -4)")

    def test_check_string_comparison(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT CHECK (name != ''))")
        db.execute("INSERT INTO t (id, name) VALUES (1, 'Alice')")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, name) VALUES (2, '')")

    def test_check_equality_constraint(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT, y INT, CHECK (x = y))")
        db.execute("INSERT INTO t (id, x, y) VALUES (1, 5, 5)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, x, y) VALUES (2, 5, 6)")


# =============================================================================
# Integration: Table Without Checks (passthrough)
# =============================================================================

class TestNoChecksPassthrough(unittest.TestCase):
    def test_table_without_checks_insert(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t (id, x) VALUES (1, -100)")
        result = db.execute("SELECT x FROM t")
        self.assertEqual(result.rows[0][0], -100)

    def test_table_without_checks_update(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT)")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("UPDATE t SET x = -100 WHERE id = 1")
        result = db.execute("SELECT x FROM t WHERE id = 1")
        self.assertEqual(result.rows[0][0], -100)

    def test_regular_constraints_still_work(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT NOT NULL UNIQUE)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id) VALUES (1)")

    def test_select_still_works(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("INSERT INTO t (id, x) VALUES (2, 20)")
        result = db.execute("SELECT x FROM t WHERE x > 15 ORDER BY x")
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0][0], 20)

    def test_delete_still_works(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("DELETE FROM t WHERE id = 1")
        result = db.execute("SELECT COUNT(*) FROM t")
        self.assertEqual(result.rows[0][0], 0)

    def test_aggregation_still_works(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 10)")
        db.execute("INSERT INTO t (id, x) VALUES (2, 20)")
        result = db.execute("SELECT SUM(x) FROM t")
        self.assertEqual(result.rows[0][0], 30)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    def test_check_on_default_value(self):
        """Default value should satisfy CHECK."""
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT DEFAULT 10 CHECK (x > 0))")
        db.execute("INSERT INTO t (id) VALUES (1)")
        result = db.execute("SELECT x FROM t")
        self.assertEqual(result.rows[0][0], 10)

    def test_check_default_violates(self):
        """Default that violates CHECK should still fail on insert."""
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT DEFAULT 0 CHECK (x > 0))")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id) VALUES (1)")

    def test_check_with_not_null(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT NOT NULL CHECK (x > 0))")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id) VALUES (1)")

    def test_many_constraints_on_one_table(self):
        db = CheckDB()
        db.execute("""
            CREATE TABLE t (
                id INT PRIMARY KEY,
                a INT CHECK (a > 0),
                b INT CHECK (b > 0),
                c INT CHECK (c > 0),
                CHECK (a + b > c),
                CHECK (a < 1000)
            )
        """)
        db.execute("INSERT INTO t (id, a, b, c) VALUES (1, 5, 5, 3)")
        with self.assertRaises(CatalogError):
            db.execute("INSERT INTO t (id, a, b, c) VALUES (2, 1, 1, 10)")

    def test_constraint_error_message_includes_name(self):
        db = CheckDB()
        db.execute("""
            CREATE TABLE t (
                id INT PRIMARY KEY,
                x INT,
                CONSTRAINT my_check CHECK (x > 0)
            )
        """)
        with self.assertRaises(CatalogError) as ctx:
            db.execute("INSERT INTO t (id, x) VALUES (1, -1)")
        self.assertIn('my_check', str(ctx.exception))

    def test_create_if_not_exists_idempotent(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("CREATE TABLE IF NOT EXISTS t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (id, x) VALUES (1, 5)")

    def test_auto_increment_with_check(self):
        db = CheckDB()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, x INT CHECK (x > 0))")
        db.execute("INSERT INTO t (x) VALUES (10)")
        db.execute("INSERT INTO t (x) VALUES (20)")
        result = db.execute("SELECT COUNT(*) FROM t")
        self.assertEqual(result.rows[0][0], 2)


if __name__ == '__main__':
    unittest.main()
