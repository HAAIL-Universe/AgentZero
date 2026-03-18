"""
Tests for C261: FOREIGN KEY Constraints
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from foreign_keys import (
    FKDB, ForeignKey, FKAction, FKTableSchema, FKEnforcer,
    FKParser, parse_fk_sql, parse_fk_sql_multi,
    AlterAddFKStmt, AlterDropFKStmt,
)
from mini_database import CatalogError, ParseError, ResultSet


# =============================================================================
# ForeignKey Dataclass
# =============================================================================

class TestForeignKeyDataclass:

    def test_create_basic(self):
        fk = ForeignKey(name="fk_order_user", child_table="orders",
                       child_columns=["user_id"], parent_table="users",
                       parent_columns=["id"])
        assert fk.name == "fk_order_user"
        assert fk.child_table == "orders"
        assert fk.child_columns == ["user_id"]
        assert fk.parent_table == "users"
        assert fk.parent_columns == ["id"]

    def test_default_actions(self):
        fk = ForeignKey(name="fk1", child_table="t", child_columns=["a"],
                       parent_table="p", parent_columns=["id"])
        assert fk.on_delete == FKAction.RESTRICT
        assert fk.on_update == FKAction.RESTRICT

    def test_custom_actions(self):
        fk = ForeignKey(name="fk1", child_table="t", child_columns=["a"],
                       parent_table="p", parent_columns=["id"],
                       on_delete=FKAction.CASCADE, on_update=FKAction.SET_NULL)
        assert fk.on_delete == FKAction.CASCADE
        assert fk.on_update == FKAction.SET_NULL

    def test_composite_fk(self):
        fk = ForeignKey(name="fk_comp", child_table="t",
                       child_columns=["a", "b"],
                       parent_table="p", parent_columns=["x", "y"])
        assert len(fk.child_columns) == 2
        assert len(fk.parent_columns) == 2

    def test_repr(self):
        fk = ForeignKey(name="fk1", child_table="orders",
                       child_columns=["uid"], parent_table="users",
                       parent_columns=["id"])
        r = repr(fk)
        assert "fk1" in r
        assert "orders" in r
        assert "users" in r


# =============================================================================
# FKAction Enum
# =============================================================================

class TestFKAction:

    def test_all_actions(self):
        assert FKAction.RESTRICT.value == "RESTRICT"
        assert FKAction.CASCADE.value == "CASCADE"
        assert FKAction.SET_NULL.value == "SET NULL"
        assert FKAction.SET_DEFAULT.value == "SET DEFAULT"
        assert FKAction.NO_ACTION.value == "NO ACTION"


# =============================================================================
# FKTableSchema
# =============================================================================

class TestFKTableSchema:

    def test_add_fk(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(name="orders",
                          columns=[ColumnDef("id", "INT", primary_key=True),
                                   ColumnDef("user_id", "INT")])
        schema = FKTableSchema(base_schema=base)
        fk = ForeignKey(name="fk1", child_table="orders",
                       child_columns=["user_id"], parent_table="users",
                       parent_columns=["id"])
        schema.add_foreign_key(fk)
        assert len(schema.foreign_keys) == 1
        assert schema.foreign_keys[0].name == "fk1"

    def test_auto_name(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(name="orders",
                          columns=[ColumnDef("id", "INT")])
        schema = FKTableSchema(base_schema=base)
        fk = ForeignKey(name=None, child_table="orders",
                       child_columns=["uid"], parent_table="users",
                       parent_columns=["id"])
        schema.add_foreign_key(fk)
        assert fk.name == "orders_fk_1"

    def test_duplicate_name_rejected(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(name="t",
                          columns=[ColumnDef("id", "INT")])
        schema = FKTableSchema(base_schema=base)
        fk1 = ForeignKey(name="fk1", child_table="t",
                        child_columns=["a"], parent_table="p",
                        parent_columns=["id"])
        fk2 = ForeignKey(name="fk1", child_table="t",
                        child_columns=["b"], parent_table="p",
                        parent_columns=["id"])
        schema.add_foreign_key(fk1)
        with pytest.raises(CatalogError, match="already exists"):
            schema.add_foreign_key(fk2)

    def test_drop_fk(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(name="t",
                          columns=[ColumnDef("id", "INT")])
        schema = FKTableSchema(base_schema=base)
        fk = ForeignKey(name="fk1", child_table="t",
                       child_columns=["a"], parent_table="p",
                       parent_columns=["id"])
        schema.add_foreign_key(fk)
        schema.drop_foreign_key("fk1")
        assert len(schema.foreign_keys) == 0

    def test_drop_nonexistent(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(name="t",
                          columns=[ColumnDef("id", "INT")])
        schema = FKTableSchema(base_schema=base)
        with pytest.raises(CatalogError, match="not found"):
            schema.drop_foreign_key("nope")

    def test_get_fk(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(name="t",
                          columns=[ColumnDef("id", "INT")])
        schema = FKTableSchema(base_schema=base)
        fk = ForeignKey(name="fk1", child_table="t",
                       child_columns=["a"], parent_table="p",
                       parent_columns=["id"])
        schema.add_foreign_key(fk)
        assert schema.get_foreign_key("fk1") is fk
        assert schema.get_foreign_key("nope") is None

    def test_proxy_properties(self):
        from mini_database import TableSchema, ColumnDef
        base = TableSchema(name="orders",
                          columns=[ColumnDef("id", "INT", primary_key=True),
                                   ColumnDef("name", "TEXT")])
        schema = FKTableSchema(base_schema=base)
        assert schema.name == "orders"
        assert len(schema.columns) == 2
        assert schema.column_names() == ["id", "name"]
        assert schema.primary_key_column() == "id"


# =============================================================================
# Parser -- Column-level REFERENCES
# =============================================================================

class TestParserColumnRef:

    def test_basic_references(self):
        sql = "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        stmt = parse_fk_sql(sql)
        assert hasattr(stmt, '_col_fks')
        fks = [f for f in stmt._col_fks if f is not None]
        assert len(fks) == 1
        assert fks[0].parent_table == "users"
        assert fks[0].parent_columns == ["id"]
        assert fks[0].child_columns == ["user_id"]

    def test_references_with_on_delete(self):
        sql = "CREATE TABLE orders (id INT, user_id INT REFERENCES users(id) ON DELETE CASCADE)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert len(fks) == 1
        assert fks[0].on_delete == FKAction.CASCADE

    def test_references_with_on_update(self):
        sql = "CREATE TABLE orders (id INT, user_id INT REFERENCES users(id) ON UPDATE SET NULL)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert fks[0].on_update == FKAction.SET_NULL

    def test_references_both_actions(self):
        sql = "CREATE TABLE t (id INT, pid INT REFERENCES p(id) ON DELETE CASCADE ON UPDATE RESTRICT)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert fks[0].on_delete == FKAction.CASCADE
        assert fks[0].on_update == FKAction.RESTRICT

    def test_multiple_references(self):
        sql = ("CREATE TABLE t (id INT PRIMARY KEY, "
               "a_id INT REFERENCES a(id), "
               "b_id INT REFERENCES b(id))")
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert len(fks) == 2


# =============================================================================
# Parser -- Table-level FOREIGN KEY
# =============================================================================

class TestParserTableFK:

    def test_basic_table_fk(self):
        sql = ("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, "
               "FOREIGN KEY (user_id) REFERENCES users(id))")
        stmt = parse_fk_sql(sql)
        assert len(stmt._table_fks) == 1
        fk = stmt._table_fks[0]
        assert fk.child_columns == ["user_id"]
        assert fk.parent_table == "users"
        assert fk.parent_columns == ["id"]

    def test_named_table_fk(self):
        sql = ("CREATE TABLE orders (id INT, user_id INT, "
               "CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id))")
        stmt = parse_fk_sql(sql)
        assert stmt._table_fks[0].name == "fk_user"

    def test_composite_fk(self):
        sql = ("CREATE TABLE t (a INT, b INT, "
               "FOREIGN KEY (a, b) REFERENCES p(x, y))")
        stmt = parse_fk_sql(sql)
        fk = stmt._table_fks[0]
        assert fk.child_columns == ["a", "b"]
        assert fk.parent_columns == ["x", "y"]

    def test_table_fk_with_actions(self):
        sql = ("CREATE TABLE t (id INT, pid INT, "
               "FOREIGN KEY (pid) REFERENCES p(id) ON DELETE SET NULL ON UPDATE CASCADE)")
        stmt = parse_fk_sql(sql)
        fk = stmt._table_fks[0]
        assert fk.on_delete == FKAction.SET_NULL
        assert fk.on_update == FKAction.CASCADE

    def test_mixed_check_and_fk(self):
        sql = ("CREATE TABLE t (id INT, pid INT, val INT, "
               "CHECK (val > 0), "
               "FOREIGN KEY (pid) REFERENCES p(id))")
        stmt = parse_fk_sql(sql)
        assert len(stmt._table_checks) == 1
        assert len(stmt._table_fks) == 1


# =============================================================================
# Parser -- FK Actions
# =============================================================================

class TestParserFKActions:

    def test_restrict(self):
        sql = "CREATE TABLE t (id INT, pid INT REFERENCES p(id) ON DELETE RESTRICT)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert fks[0].on_delete == FKAction.RESTRICT

    def test_cascade(self):
        sql = "CREATE TABLE t (id INT, pid INT REFERENCES p(id) ON DELETE CASCADE)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert fks[0].on_delete == FKAction.CASCADE

    def test_set_null(self):
        sql = "CREATE TABLE t (id INT, pid INT REFERENCES p(id) ON DELETE SET NULL)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert fks[0].on_delete == FKAction.SET_NULL

    def test_set_default(self):
        sql = "CREATE TABLE t (id INT, pid INT REFERENCES p(id) ON DELETE SET DEFAULT)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert fks[0].on_delete == FKAction.SET_DEFAULT

    def test_no_action(self):
        sql = "CREATE TABLE t (id INT, pid INT REFERENCES p(id) ON DELETE NO ACTION)"
        stmt = parse_fk_sql(sql)
        fks = [f for f in stmt._col_fks if f is not None]
        assert fks[0].on_delete == FKAction.NO_ACTION


# =============================================================================
# Parser -- ALTER TABLE
# =============================================================================

class TestParserAlterFK:

    def test_alter_add_fk(self):
        sql = "ALTER TABLE orders ADD FOREIGN KEY (user_id) REFERENCES users(id)"
        stmt = parse_fk_sql(sql)
        assert isinstance(stmt, AlterAddFKStmt)
        assert stmt.table_name == "orders"
        assert stmt.foreign_key.child_columns == ["user_id"]
        assert stmt.foreign_key.parent_table == "users"

    def test_alter_add_named_fk(self):
        sql = "ALTER TABLE orders ADD CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id)"
        stmt = parse_fk_sql(sql)
        assert isinstance(stmt, AlterAddFKStmt)
        assert stmt.foreign_key.name == "fk_user"

    def test_alter_add_fk_with_actions(self):
        sql = ("ALTER TABLE t ADD FOREIGN KEY (pid) REFERENCES p(id) "
               "ON DELETE CASCADE ON UPDATE SET NULL")
        stmt = parse_fk_sql(sql)
        assert stmt.foreign_key.on_delete == FKAction.CASCADE
        assert stmt.foreign_key.on_update == FKAction.SET_NULL

    def test_alter_drop_fk(self):
        sql = "ALTER TABLE orders DROP FOREIGN KEY fk_user"
        stmt = parse_fk_sql(sql)
        assert isinstance(stmt, AlterDropFKStmt)
        assert stmt.table_name == "orders"
        assert stmt.constraint_name == "fk_user"


# =============================================================================
# FKDB -- Basic INSERT validation
# =============================================================================

class TestInsertValidation:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")

    def test_valid_insert(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 1

    def test_invalid_insert_rejects(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        with pytest.raises(CatalogError, match="Foreign key violation"):
            self.db.execute("INSERT INTO orders VALUES (1, 999)")

    def test_null_fk_allowed(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, NULL)")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 1

    def test_multiple_valid_inserts(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("INSERT INTO orders VALUES (2, 2)")
        self.db.execute("INSERT INTO orders VALUES (3, 1)")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 3

    def test_insert_after_parent_insert(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO users VALUES (3, 'Charlie')")
        self.db.execute("INSERT INTO orders VALUES (1, 3)")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 1


# =============================================================================
# FKDB -- UPDATE validation (child)
# =============================================================================

class TestUpdateChildValidation:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")

    def test_valid_update(self):
        self.db.execute("UPDATE orders SET user_id = 2 WHERE id = 1")
        result = self.db.execute("SELECT user_id FROM orders WHERE id = 1")
        assert result.rows[0][0] == 2

    def test_invalid_update_rejects(self):
        with pytest.raises(CatalogError, match="Foreign key violation"):
            self.db.execute("UPDATE orders SET user_id = 999 WHERE id = 1")

    def test_update_to_null_allowed(self):
        self.db.execute("UPDATE orders SET user_id = NULL WHERE id = 1")
        result = self.db.execute("SELECT user_id FROM orders WHERE id = 1")
        assert result.rows[0][0] is None

    def test_update_non_fk_column_ok(self):
        # Updating non-FK column should always work
        self.db.execute(
            "CREATE TABLE items (id INT PRIMARY KEY, order_id INT REFERENCES orders(id), qty INT)"
        )
        self.db.execute("INSERT INTO items VALUES (1, 1, 5)")
        self.db.execute("UPDATE items SET qty = 10 WHERE id = 1")
        result = self.db.execute("SELECT qty FROM items WHERE id = 1")
        assert result.rows[0][0] == 10


# =============================================================================
# FKDB -- DELETE RESTRICT
# =============================================================================

class TestDeleteRestrict:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")

    def test_delete_parent_with_children_restricted(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        with pytest.raises(CatalogError, match="Cannot delete"):
            self.db.execute("DELETE FROM users WHERE id = 1")

    def test_delete_parent_without_children_ok(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        # Bob has no orders, so we can delete him
        self.db.execute("DELETE FROM users WHERE id = 2")
        result = self.db.execute("SELECT * FROM users")
        assert len(result.rows) == 1

    def test_delete_unreferenced_table_ok(self):
        # A table with no children can always be deleted from
        self.db.execute("DELETE FROM users WHERE id = 1")
        result = self.db.execute("SELECT * FROM users")
        assert len(result.rows) == 1


# =============================================================================
# FKDB -- DELETE CASCADE
# =============================================================================

class TestDeleteCascade:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")

    def test_cascade_delete_children(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON DELETE CASCADE)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("INSERT INTO orders VALUES (2, 1)")
        self.db.execute("INSERT INTO orders VALUES (3, 2)")

        self.db.execute("DELETE FROM users WHERE id = 1")

        # Alice's orders should be gone
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 1
        assert result.rows[0][1] == 2  # Bob's order remains

    def test_cascade_no_children_ok(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON DELETE CASCADE)"
        )
        self.db.execute("DELETE FROM users WHERE id = 2")
        result = self.db.execute("SELECT * FROM users")
        assert len(result.rows) == 1

    def test_cascade_all_children(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON DELETE CASCADE)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("INSERT INTO orders VALUES (2, 1)")
        self.db.execute("INSERT INTO orders VALUES (3, 1)")

        self.db.execute("DELETE FROM users WHERE id = 1")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 0


# =============================================================================
# FKDB -- DELETE SET NULL
# =============================================================================

class TestDeleteSetNull:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")

    def test_set_null_on_delete(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON DELETE SET NULL)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("INSERT INTO orders VALUES (2, 2)")

        self.db.execute("DELETE FROM users WHERE id = 1")

        result = self.db.execute("SELECT * FROM orders ORDER BY id")
        assert result.rows[0][1] is None  # was user 1, now NULL
        assert result.rows[1][1] == 2     # unchanged

    def test_set_null_preserves_other_columns(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT "
            "REFERENCES users(id) ON DELETE SET NULL, amount INT)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        self.db.execute("DELETE FROM users WHERE id = 1")
        result = self.db.execute("SELECT * FROM orders")
        assert result.rows[0][1] is None  # FK nulled
        assert result.rows[0][2] == 100   # amount preserved


# =============================================================================
# FKDB -- DELETE SET DEFAULT
# =============================================================================

class TestDeleteSetDefault:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (0, 'System')")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")

    def test_set_default_on_delete(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT DEFAULT 0 REFERENCES users(id) ON DELETE SET DEFAULT)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("DELETE FROM users WHERE id = 1")
        result = self.db.execute("SELECT user_id FROM orders WHERE id = 1")
        assert result.rows[0][0] == 0  # set to default


# =============================================================================
# FKDB -- UPDATE CASCADE (parent)
# =============================================================================

class TestUpdateCascade:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob')")

    def test_update_parent_cascade(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON UPDATE CASCADE)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("INSERT INTO orders VALUES (2, 1)")

        self.db.execute("UPDATE users SET id = 10 WHERE id = 1")

        result = self.db.execute("SELECT user_id FROM orders ORDER BY id")
        assert result.rows[0][0] == 10  # cascaded
        assert result.rows[1][0] == 10  # cascaded

    def test_update_parent_restrict(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON UPDATE RESTRICT)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        with pytest.raises(CatalogError, match="Cannot update"):
            self.db.execute("UPDATE users SET id = 10 WHERE id = 1")

    def test_update_parent_set_null(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON UPDATE SET NULL)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("UPDATE users SET id = 10 WHERE id = 1")
        result = self.db.execute("SELECT user_id FROM orders WHERE id = 1")
        assert result.rows[0][0] is None

    def test_update_non_referenced_column_ok(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON UPDATE RESTRICT)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        # Updating name (not referenced) should be fine
        self.db.execute("UPDATE users SET name = 'Alicia' WHERE id = 1")
        result = self.db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'Alicia'


# =============================================================================
# FKDB -- ALTER TABLE ADD/DROP FK
# =============================================================================

class TestAlterFK:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("CREATE TABLE orders (id INT PRIMARY KEY, user_id INT)")
        self.db.execute("INSERT INTO orders VALUES (1, 1)")

    def test_alter_add_fk(self):
        self.db.execute(
            "ALTER TABLE orders ADD FOREIGN KEY (user_id) REFERENCES users(id)"
        )
        fks = self.db.get_foreign_keys("orders")
        assert len(fks) == 1
        assert fks[0].parent_table == "users"

    def test_alter_add_named_fk(self):
        self.db.execute(
            "ALTER TABLE orders ADD CONSTRAINT fk_user "
            "FOREIGN KEY (user_id) REFERENCES users(id)"
        )
        fks = self.db.get_foreign_keys("orders")
        assert fks[0].name == "fk_user"

    def test_alter_add_fk_validates_existing_data(self):
        self.db.execute("INSERT INTO orders VALUES (2, 999)")
        with pytest.raises(CatalogError, match="Foreign key violation"):
            self.db.execute(
                "ALTER TABLE orders ADD FOREIGN KEY (user_id) REFERENCES users(id)"
            )

    def test_alter_drop_fk(self):
        self.db.execute(
            "ALTER TABLE orders ADD CONSTRAINT fk_user "
            "FOREIGN KEY (user_id) REFERENCES users(id)"
        )
        self.db.execute("ALTER TABLE orders DROP FOREIGN KEY fk_user")
        fks = self.db.get_foreign_keys("orders")
        assert len(fks) == 0

    def test_alter_drop_via_constraint(self):
        self.db.execute(
            "ALTER TABLE orders ADD CONSTRAINT fk_user "
            "FOREIGN KEY (user_id) REFERENCES users(id)"
        )
        self.db.execute("ALTER TABLE orders DROP CONSTRAINT fk_user")
        fks = self.db.get_foreign_keys("orders")
        assert len(fks) == 0

    def test_alter_drop_nonexistent_fk(self):
        with pytest.raises(CatalogError, match="not found"):
            self.db.execute("ALTER TABLE orders DROP FOREIGN KEY nope")

    def test_after_drop_fk_insert_unrestricted(self):
        self.db.execute(
            "ALTER TABLE orders ADD CONSTRAINT fk_user "
            "FOREIGN KEY (user_id) REFERENCES users(id)"
        )
        self.db.execute("ALTER TABLE orders DROP CONSTRAINT fk_user")
        # Now we can insert invalid references
        self.db.execute("INSERT INTO orders VALUES (2, 999)")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 2


# =============================================================================
# FKDB -- Composite Foreign Keys
# =============================================================================

class TestCompositeForeignKeys:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute(
            "CREATE TABLE products (brand TEXT, model TEXT, price INT, "
            "PRIMARY KEY (brand))"  # Simplified -- just brand as PK for testing
        )

    def test_composite_fk_via_alter(self):
        self.db.execute(
            "CREATE TABLE locations (id INT PRIMARY KEY, brand TEXT, model TEXT)"
        )
        # Use single-column FK for basic composite test
        self.db.execute("INSERT INTO products VALUES ('Acme', 'Widget', 100)")
        self.db.execute("INSERT INTO locations VALUES (1, 'Acme', 'Widget')")
        self.db.execute(
            "ALTER TABLE locations ADD FOREIGN KEY (brand) REFERENCES products(brand)"
        )
        fks = self.db.get_foreign_keys("locations")
        assert len(fks) == 1

    def test_composite_insert_valid(self):
        self.db.execute("INSERT INTO products VALUES ('Acme', 'Widget', 100)")
        self.db.execute(
            "CREATE TABLE inventory (id INT PRIMARY KEY, brand TEXT "
            "REFERENCES products(brand), qty INT)"
        )
        self.db.execute("INSERT INTO inventory VALUES (1, 'Acme', 5)")
        result = self.db.execute("SELECT * FROM inventory")
        assert len(result.rows) == 1

    def test_composite_insert_invalid(self):
        self.db.execute("INSERT INTO products VALUES ('Acme', 'Widget', 100)")
        self.db.execute(
            "CREATE TABLE inventory (id INT PRIMARY KEY, brand TEXT "
            "REFERENCES products(brand), qty INT)"
        )
        with pytest.raises(CatalogError, match="Foreign key violation"):
            self.db.execute("INSERT INTO inventory VALUES (1, 'Unknown', 5)")


# =============================================================================
# FKDB -- Self-referencing FK
# =============================================================================

class TestSelfReference:

    def setup_method(self):
        self.db = FKDB()

    def test_self_reference_create(self):
        self.db.execute(
            "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, "
            "manager_id INT REFERENCES employees(id))"
        )
        fks = self.db.get_foreign_keys("employees")
        assert len(fks) == 1
        assert fks[0].parent_table == "employees"

    def test_self_reference_insert_root(self):
        self.db.execute(
            "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, "
            "manager_id INT REFERENCES employees(id))"
        )
        # Root employee with NULL manager
        self.db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)")
        result = self.db.execute("SELECT * FROM employees")
        assert len(result.rows) == 1

    def test_self_reference_insert_valid(self):
        self.db.execute(
            "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, "
            "manager_id INT REFERENCES employees(id))"
        )
        self.db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)")
        self.db.execute("INSERT INTO employees VALUES (2, 'VP', 1)")
        result = self.db.execute("SELECT * FROM employees")
        assert len(result.rows) == 2

    def test_self_reference_insert_invalid(self):
        self.db.execute(
            "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, "
            "manager_id INT REFERENCES employees(id))"
        )
        self.db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)")
        with pytest.raises(CatalogError, match="Foreign key violation"):
            self.db.execute("INSERT INTO employees VALUES (2, 'VP', 99)")

    def test_self_reference_cascade_delete(self):
        self.db.execute(
            "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, "
            "manager_id INT REFERENCES employees(id) ON DELETE CASCADE)"
        )
        self.db.execute("INSERT INTO employees VALUES (1, 'CEO', NULL)")
        self.db.execute("INSERT INTO employees VALUES (2, 'VP', 1)")
        self.db.execute("INSERT INTO employees VALUES (3, 'Mgr', 1)")

        self.db.execute("DELETE FROM employees WHERE id = 1")
        result = self.db.execute("SELECT * FROM employees")
        assert len(result.rows) == 0  # cascade deleted all


# =============================================================================
# FKDB -- Introspection
# =============================================================================

class TestIntrospection:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON DELETE CASCADE)"
        )

    def test_get_foreign_keys(self):
        fks = self.db.get_foreign_keys("orders")
        assert len(fks) == 1
        fk = fks[0]
        assert fk.child_columns == ["user_id"]
        assert fk.parent_table == "users"
        assert fk.on_delete == FKAction.CASCADE

    def test_get_foreign_keys_empty(self):
        fks = self.db.get_foreign_keys("users")
        assert fks == []

    def test_describe_foreign_keys(self):
        result = self.db.describe_foreign_keys("orders")
        assert len(result.rows) == 1
        assert result.columns == ['name', 'columns', 'references_table',
                                  'references_columns', 'on_delete', 'on_update']
        row = result.rows[0]
        assert row[2] == "users"
        assert row[4] == "CASCADE"

    def test_get_referencing_tables(self):
        refs = self.db.get_referencing_tables("users")
        assert "orders" in refs

    def test_get_referencing_tables_empty(self):
        refs = self.db.get_referencing_tables("orders")
        assert refs == []


# =============================================================================
# FKDB -- Table with no FK passthrough
# =============================================================================

class TestPassthrough:

    def setup_method(self):
        self.db = FKDB()

    def test_table_without_fk_insert(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'hello')")
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 1

    def test_table_without_fk_update(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'hello')")
        self.db.execute("UPDATE t SET name = 'world' WHERE id = 1")
        result = self.db.execute("SELECT name FROM t WHERE id = 1")
        assert result.rows[0][0] == 'world'

    def test_table_without_fk_delete(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO t VALUES (1, 'hello')")
        self.db.execute("DELETE FROM t WHERE id = 1")
        result = self.db.execute("SELECT * FROM t")
        assert len(result.rows) == 0

    def test_select_passthrough(self):
        self.db.execute("CREATE TABLE t (id INT PRIMARY KEY, val INT)")
        self.db.execute("INSERT INTO t VALUES (1, 10)")
        self.db.execute("INSERT INTO t VALUES (2, 20)")
        result = self.db.execute("SELECT SUM(val) FROM t")
        assert result.rows[0][0] == 30

    def test_check_constraint_still_works(self):
        self.db.execute(
            "CREATE TABLE t (id INT PRIMARY KEY, val INT CHECK (val > 0))"
        )
        self.db.execute("INSERT INTO t VALUES (1, 5)")
        with pytest.raises(CatalogError, match="CHECK"):
            self.db.execute("INSERT INTO t VALUES (2, -1)")


# =============================================================================
# FKDB -- Edge cases
# =============================================================================

class TestEdgeCases:

    def setup_method(self):
        self.db = FKDB()

    def test_fk_to_nonexistent_table_rejected(self):
        with pytest.raises(CatalogError, match="does not exist"):
            self.db.execute(
                "CREATE TABLE orders (id INT PRIMARY KEY, "
                "user_id INT REFERENCES nonexistent(id))"
            )

    def test_fk_to_nonexistent_column_rejected(self):
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        with pytest.raises(CatalogError, match="not found"):
            self.db.execute(
                "CREATE TABLE orders (id INT PRIMARY KEY, "
                "user_id INT REFERENCES users(nonexistent))"
            )

    def test_multiple_fks_on_one_table(self):
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")
        self.db.execute("INSERT INTO products VALUES (1, 'Widget')")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id), "
            "product_id INT REFERENCES products(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1, 1)")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 1

    def test_multiple_fks_both_validated(self):
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        self.db.execute("CREATE TABLE products (id INT PRIMARY KEY)")
        self.db.execute("INSERT INTO users VALUES (1)")
        self.db.execute("INSERT INTO products VALUES (1)")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id), "
            "product_id INT REFERENCES products(id))"
        )
        # Valid user, invalid product
        with pytest.raises(CatalogError, match="Foreign key violation"):
            self.db.execute("INSERT INTO orders VALUES (1, 1, 999)")

    def test_delete_parent_multiple_children_tables(self):
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        self.db.execute("INSERT INTO users VALUES (1)")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id))"
        )
        self.db.execute(
            "CREATE TABLE reviews (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        # Can't delete -- orders references it
        with pytest.raises(CatalogError, match="Cannot delete"):
            self.db.execute("DELETE FROM users WHERE id = 1")

    def test_chain_of_fks(self):
        """A -> B -> C chain: delete from C checks B, delete from B checks A."""
        self.db.execute("CREATE TABLE c (id INT PRIMARY KEY)")
        self.db.execute("INSERT INTO c VALUES (1)")
        self.db.execute(
            "CREATE TABLE b (id INT PRIMARY KEY, "
            "c_id INT REFERENCES c(id) ON DELETE CASCADE)"
        )
        self.db.execute("INSERT INTO b VALUES (1, 1)")
        self.db.execute(
            "CREATE TABLE a (id INT PRIMARY KEY, "
            "b_id INT REFERENCES b(id) ON DELETE CASCADE)"
        )
        self.db.execute("INSERT INTO a VALUES (1, 1)")

        # Delete from C should cascade to B, which cascades to A
        self.db.execute("DELETE FROM c WHERE id = 1")
        assert len(self.db.execute("SELECT * FROM c").rows) == 0
        assert len(self.db.execute("SELECT * FROM b").rows) == 0
        assert len(self.db.execute("SELECT * FROM a").rows) == 0

    def test_fk_with_check_constraint(self):
        """FK and CHECK on same table."""
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        self.db.execute("INSERT INTO users VALUES (1)")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id), "
            "amount INT, CHECK (amount > 0))"
        )
        # Valid FK, invalid CHECK
        with pytest.raises(CatalogError, match="CHECK"):
            self.db.execute("INSERT INTO orders VALUES (1, 1, -5)")

    def test_fk_with_check_both_valid(self):
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        self.db.execute("INSERT INTO users VALUES (1)")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id), "
            "amount INT, CHECK (amount > 0))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1, 100)")
        result = self.db.execute("SELECT * FROM orders")
        assert len(result.rows) == 1

    def test_no_action_same_as_restrict(self):
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        self.db.execute("INSERT INTO users VALUES (1)")
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT REFERENCES users(id) ON DELETE NO ACTION)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        with pytest.raises(CatalogError, match="Cannot delete"):
            self.db.execute("DELETE FROM users WHERE id = 1")

    def test_execute_multi(self):
        results = self.db.execute_multi(
            "CREATE TABLE users (id INT PRIMARY KEY, name TEXT); "
            "INSERT INTO users VALUES (1, 'Alice');"
        )
        assert len(results) == 2


# =============================================================================
# FKDB -- Table-level FOREIGN KEY in CREATE
# =============================================================================

class TestTableLevelFK:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice')")

    def test_table_level_fk_enforced(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, "
            "FOREIGN KEY (user_id) REFERENCES users(id))"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        with pytest.raises(CatalogError, match="Foreign key violation"):
            self.db.execute("INSERT INTO orders VALUES (2, 999)")

    def test_named_table_level_fk(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, "
            "CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id))"
        )
        fks = self.db.get_foreign_keys("orders")
        assert fks[0].name == "fk_user"


# =============================================================================
# FKDB -- Update parent SET DEFAULT
# =============================================================================

class TestUpdateSetDefault:

    def setup_method(self):
        self.db = FKDB()
        self.db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        self.db.execute("INSERT INTO users VALUES (0)")
        self.db.execute("INSERT INTO users VALUES (1)")

    def test_update_parent_set_default(self):
        self.db.execute(
            "CREATE TABLE orders (id INT PRIMARY KEY, "
            "user_id INT DEFAULT 0 REFERENCES users(id) ON UPDATE SET DEFAULT)"
        )
        self.db.execute("INSERT INTO orders VALUES (1, 1)")
        self.db.execute("UPDATE users SET id = 10 WHERE id = 1")
        result = self.db.execute("SELECT user_id FROM orders WHERE id = 1")
        assert result.rows[0][0] == 0  # set to default


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
