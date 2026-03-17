"""
Tests for C251: SQL Triggers
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from sql_triggers import (
    TriggerDB, TriggerCatalog, TriggerDefinition,
    TriggerTiming, TriggerEvent, TriggerSignal,
    TriggerLexer, TriggerParser,
    CreateTriggerStmt, DropTriggerStmt, ShowTriggersStmt, AlterTriggerStmt,
    SignalStmt, TriggerSetStmt, TriggerIfStmt,
    DatabaseError,
)


# =============================================================================
# Helpers
# =============================================================================

def make_db():
    db = TriggerDB()
    db.execute("CREATE TABLE users (id INT, name VARCHAR, email VARCHAR)")
    db.execute("CREATE TABLE audit_log (action VARCHAR, user_id INT, details VARCHAR)")
    return db


def rows(result):
    """Extract rows from ResultSet."""
    return result.rows


def col_rows(result):
    """Return list of dicts from ResultSet."""
    return [dict(zip(result.columns, r)) for r in result.rows]


# =============================================================================
# 1. Trigger Catalog
# =============================================================================

class TestTriggerCatalog:
    def test_create_trigger(self):
        cat = TriggerCatalog()
        tdef = TriggerDefinition(
            name="t1", timing=TriggerTiming.BEFORE,
            event=TriggerEvent.INSERT, table_name="users", body=[]
        )
        cat.create_trigger(tdef)
        assert cat.has_trigger("t1")

    def test_create_duplicate_raises(self):
        cat = TriggerCatalog()
        tdef = TriggerDefinition(
            name="t1", timing=TriggerTiming.BEFORE,
            event=TriggerEvent.INSERT, table_name="users", body=[]
        )
        cat.create_trigger(tdef)
        with pytest.raises(DatabaseError, match="already exists"):
            cat.create_trigger(tdef)

    def test_create_replace(self):
        cat = TriggerCatalog()
        tdef = TriggerDefinition(
            name="t1", timing=TriggerTiming.BEFORE,
            event=TriggerEvent.INSERT, table_name="users", body=[]
        )
        cat.create_trigger(tdef)
        tdef2 = TriggerDefinition(
            name="t1", timing=TriggerTiming.AFTER,
            event=TriggerEvent.INSERT, table_name="users", body=[]
        )
        cat.create_trigger(tdef2, replace=True)
        assert cat.get_trigger("t1").timing == TriggerTiming.AFTER

    def test_drop_trigger(self):
        cat = TriggerCatalog()
        tdef = TriggerDefinition(
            name="t1", timing=TriggerTiming.BEFORE,
            event=TriggerEvent.INSERT, table_name="users", body=[]
        )
        cat.create_trigger(tdef)
        cat.drop_trigger("t1")
        assert not cat.has_trigger("t1")

    def test_drop_nonexistent_raises(self):
        cat = TriggerCatalog()
        with pytest.raises(DatabaseError, match="does not exist"):
            cat.drop_trigger("nope")

    def test_drop_if_exists(self):
        cat = TriggerCatalog()
        cat.drop_trigger("nope", if_exists=True)  # no error

    def test_get_triggers_for(self):
        cat = TriggerCatalog()
        t1 = TriggerDefinition(name="t1", timing=TriggerTiming.BEFORE,
                               event=TriggerEvent.INSERT, table_name="users", body=[])
        t2 = TriggerDefinition(name="t2", timing=TriggerTiming.AFTER,
                               event=TriggerEvent.INSERT, table_name="users", body=[])
        t3 = TriggerDefinition(name="t3", timing=TriggerTiming.BEFORE,
                               event=TriggerEvent.UPDATE, table_name="users", body=[])
        cat.create_trigger(t1)
        cat.create_trigger(t2)
        cat.create_trigger(t3)
        result = cat.get_triggers_for("users", TriggerTiming.BEFORE, TriggerEvent.INSERT)
        assert len(result) == 1
        assert result[0].name == "t1"

    def test_creation_order(self):
        cat = TriggerCatalog()
        t1 = TriggerDefinition(name="a", timing=TriggerTiming.BEFORE,
                               event=TriggerEvent.INSERT, table_name="t", body=[])
        t2 = TriggerDefinition(name="b", timing=TriggerTiming.BEFORE,
                               event=TriggerEvent.INSERT, table_name="t", body=[])
        cat.create_trigger(t1)
        cat.create_trigger(t2)
        result = cat.get_triggers_for("t", TriggerTiming.BEFORE, TriggerEvent.INSERT)
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_enable_disable(self):
        cat = TriggerCatalog()
        tdef = TriggerDefinition(name="t1", timing=TriggerTiming.BEFORE,
                                event=TriggerEvent.INSERT, table_name="users", body=[])
        cat.create_trigger(tdef)
        cat.enable_trigger("t1", False)
        assert not cat.get_trigger("t1").enabled
        result = cat.get_triggers_for("users", TriggerTiming.BEFORE, TriggerEvent.INSERT)
        assert len(result) == 0  # disabled triggers not returned

    def test_list_triggers(self):
        cat = TriggerCatalog()
        t1 = TriggerDefinition(name="t1", timing=TriggerTiming.BEFORE,
                               event=TriggerEvent.INSERT, table_name="users", body=[])
        t2 = TriggerDefinition(name="t2", timing=TriggerTiming.AFTER,
                               event=TriggerEvent.DELETE, table_name="orders", body=[])
        cat.create_trigger(t1)
        cat.create_trigger(t2)
        assert len(cat.list_triggers()) == 2
        assert len(cat.list_triggers("users")) == 1
        assert len(cat.list_triggers("orders")) == 1

    def test_case_insensitive(self):
        cat = TriggerCatalog()
        tdef = TriggerDefinition(name="MyTrigger", timing=TriggerTiming.BEFORE,
                                event=TriggerEvent.INSERT, table_name="Users", body=[])
        cat.create_trigger(tdef)
        assert cat.has_trigger("mytrigger")
        assert cat.has_trigger("MYTRIGGER")
        result = cat.get_triggers_for("users", TriggerTiming.BEFORE, TriggerEvent.INSERT)
        assert len(result) == 1


# =============================================================================
# 2. CREATE TRIGGER parsing
# =============================================================================

class TestCreateTriggerParsing:
    def test_basic_before_insert(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER audit_insert
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('INSERT', NEW.id, NEW.name);
            END
        """)
        result = db.execute("SHOW TRIGGERS")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 'audit_insert'
        assert result.rows[0][1] == 'BEFORE'
        assert result.rows[0][2] == 'INSERT'

    def test_after_update(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER track_update
            AFTER UPDATE ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('UPDATE', OLD.id, NEW.name);
            END
        """)
        result = db.execute("SHOW TRIGGERS")
        assert result.rows[0][1] == 'AFTER'
        assert result.rows[0][2] == 'UPDATE'

    def test_after_delete(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER track_delete
            AFTER DELETE ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('DELETE', OLD.id, OLD.name);
            END
        """)
        result = db.execute("SHOW TRIGGERS")
        assert result.rows[0][2] == 'DELETE'

    def test_create_or_replace(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('v1');
            END
        """)
        db.execute("""
            CREATE OR REPLACE TRIGGER t1 AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('v2');
            END
        """)
        result = db.execute("SHOW TRIGGERS")
        assert len(result.rows) == 1
        assert result.rows[0][1] == 'AFTER'

    def test_with_when_condition(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER check_name
            BEFORE INSERT ON users
            FOR EACH ROW
            WHEN (NEW.name IS NOT NULL)
            BEGIN
                INSERT INTO audit_log (action) VALUES ('has_name');
            END
        """)
        result = db.execute("SHOW TRIGGERS")
        assert len(result.rows) == 1

    def test_update_of_columns(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER track_email_change
            AFTER UPDATE OF email ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('EMAIL_CHANGE', OLD.id, NEW.email);
            END
        """)
        result = db.execute("SHOW TRIGGERS")
        assert result.rows[0][2] == 'UPDATE'

    def test_nonexistent_table_raises(self):
        db = make_db()
        with pytest.raises(DatabaseError, match="does not exist"):
            db.execute("""
                CREATE TRIGGER bad BEFORE INSERT ON nosuchtable
                FOR EACH ROW BEGIN
                    INSERT INTO audit_log (action) VALUES ('x');
                END
            """)


# =============================================================================
# 3. DROP TRIGGER
# =============================================================================

class TestDropTrigger:
    def test_drop_trigger(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('x');
            END
        """)
        db.execute("DROP TRIGGER t1")
        result = db.execute("SHOW TRIGGERS")
        assert len(result.rows) == 0

    def test_drop_if_exists(self):
        db = make_db()
        db.execute("DROP TRIGGER IF EXISTS nonexistent")  # no error

    def test_drop_nonexistent_raises(self):
        db = make_db()
        with pytest.raises(DatabaseError, match="does not exist"):
            db.execute("DROP TRIGGER nonexistent")


# =============================================================================
# 4. SHOW TRIGGERS
# =============================================================================

class TestShowTriggers:
    def test_show_all(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('x');
            END
        """)
        db.execute("""
            CREATE TRIGGER t2 AFTER DELETE ON audit_log
            FOR EACH ROW BEGIN
                INSERT INTO users (id, name) VALUES (0, 'cleanup');
            END
        """)
        result = db.execute("SHOW TRIGGERS")
        assert len(result.rows) == 2

    def test_show_on_table(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('x');
            END
        """)
        db.execute("""
            CREATE TRIGGER t2 AFTER DELETE ON audit_log
            FOR EACH ROW BEGIN
                INSERT INTO users (id, name) VALUES (0, 'cleanup');
            END
        """)
        result = db.execute("SHOW TRIGGERS ON users")
        assert len(result.rows) == 1
        assert result.rows[0][0] == 't1'


# =============================================================================
# 5. ALTER TRIGGER (enable/disable)
# =============================================================================

class TestAlterTrigger:
    def test_disable_trigger(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('tracked');
            END
        """)
        db.execute("ALTER TRIGGER t1 DISABLE")
        result = db.execute("SHOW TRIGGERS")
        assert result.rows[0][4] == 'DISABLED'

        # Insert should not trigger audit
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        audit = db.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 0

    def test_enable_trigger(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('tracked');
            END
        """)
        db.execute("ALTER TRIGGER t1 DISABLE")
        db.execute("ALTER TRIGGER t1 ENABLE")
        result = db.execute("SHOW TRIGGERS")
        assert result.rows[0][4] == 'ENABLED'


# =============================================================================
# 6. BEFORE INSERT triggers
# =============================================================================

class TestBeforeInsert:
    def test_audit_logging(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER audit_insert
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('INSERT', NEW.id, NEW.name);
            END
        """)
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'a@b.com')")
        audit = db.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 1
        assert audit.rows[0][0] == 'INSERT'
        assert audit.rows[0][1] == 1
        assert audit.rows[0][2] == 'Alice'

    def test_modify_new_values(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER uppercase_name
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                SET NEW.name = UPPER(NEW.name);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
        result = db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'

    def test_set_default_values(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER default_email
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                SET NEW.email = CONCAT(LOWER(NEW.name), '@default.com');
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        result = db.execute("SELECT email FROM users WHERE id = 1")
        assert result.rows[0][0] == 'alice@default.com'

    def test_signal_cancels_insert(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER validate_id
            BEFORE INSERT ON users
            FOR EACH ROW
            WHEN (NEW.id < 0)
            BEGIN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'ID must be positive';
            END
        """)
        with pytest.raises(DatabaseError, match="ID must be positive"):
            db.execute("INSERT INTO users (id, name) VALUES (-1, 'Bad')")

        # Valid insert should work
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Good')")
        result = db.execute("SELECT COUNT(*) AS cnt FROM users")
        assert result.rows[0][0] == 1

    def test_multiple_rows_insert(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER count_insert
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id) VALUES ('INSERT', NEW.id);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A'), (2, 'B'), (3, 'C')")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 3


# =============================================================================
# 7. AFTER INSERT triggers
# =============================================================================

class TestAfterInsert:
    def test_after_insert_audit(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER after_insert_audit
            AFTER INSERT ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('AFTER_INSERT', NEW.id, NEW.name);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        audit = db.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 1
        assert audit.rows[0][0] == 'AFTER_INSERT'

    def test_after_insert_sees_committed_data(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER after_count
            AFTER INSERT ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id) VALUES ('COUNT', NEW.id);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'B')")
        audit = db.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 2


# =============================================================================
# 8. BEFORE UPDATE triggers
# =============================================================================

class TestBeforeUpdate:
    def test_update_audit(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'old@a.com')")
        db.execute("""
            CREATE TRIGGER update_audit
            BEFORE UPDATE ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('UPDATE', OLD.id, NEW.email);
            END
        """)
        db.execute("UPDATE users SET email = 'new@a.com' WHERE id = 1")
        audit = db.execute("SELECT * FROM audit_log")
        assert audit.rows[0][0] == 'UPDATE'
        assert audit.rows[0][1] == 1
        assert audit.rows[0][2] == 'new@a.com'

    def test_modify_new_values_on_update(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("""
            CREATE TRIGGER force_upper
            BEFORE UPDATE ON users
            FOR EACH ROW
            BEGIN
                SET NEW.name = UPPER(NEW.name);
            END
        """)
        db.execute("UPDATE users SET name = 'bob' WHERE id = 1")
        result = db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'BOB'

    def test_signal_cancels_update(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("""
            CREATE TRIGGER no_rename
            BEFORE UPDATE ON users
            FOR EACH ROW
            WHEN (OLD.name != NEW.name)
            BEGIN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Cannot rename users';
            END
        """)
        with pytest.raises(DatabaseError, match="Cannot rename"):
            db.execute("UPDATE users SET name = 'Bob' WHERE id = 1")

        # Name should be unchanged
        result = db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'Alice'

    def test_old_new_values(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'a@b.com')")
        db.execute("""
            CREATE TRIGGER log_change
            BEFORE UPDATE ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('CHANGE', OLD.id, CONCAT(OLD.name, '->', NEW.name));
            END
        """)
        db.execute("UPDATE users SET name = 'Bob' WHERE id = 1")
        audit = db.execute("SELECT details FROM audit_log")
        assert audit.rows[0][0] == 'Alice->Bob'

    def test_update_of_specific_column(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'a@b.com')")
        db.execute("""
            CREATE TRIGGER email_change_only
            AFTER UPDATE OF email ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, details) VALUES ('EMAIL_CHANGE', NEW.email);
            END
        """)
        # Change name only -- should NOT fire trigger
        db.execute("UPDATE users SET name = 'Bob' WHERE id = 1")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 0

        # Change email -- should fire trigger
        db.execute("UPDATE users SET email = 'new@b.com' WHERE id = 1")
        audit = db.execute("SELECT action, details FROM audit_log")
        assert len(audit.rows) == 1
        assert audit.rows[0][0] == 'EMAIL_CHANGE'
        assert audit.rows[0][1] == 'new@b.com'


# =============================================================================
# 9. AFTER UPDATE triggers
# =============================================================================

class TestAfterUpdate:
    def test_after_update_audit(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("""
            CREATE TRIGGER after_update
            AFTER UPDATE ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('AFTER_UPDATE', NEW.id, NEW.name);
            END
        """)
        db.execute("UPDATE users SET name = 'Bob' WHERE id = 1")
        audit = db.execute("SELECT * FROM audit_log")
        assert audit.rows[0][0] == 'AFTER_UPDATE'
        assert audit.rows[0][2] == 'Bob'


# =============================================================================
# 10. BEFORE DELETE triggers
# =============================================================================

class TestBeforeDelete:
    def test_delete_audit(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("""
            CREATE TRIGGER before_delete
            BEFORE DELETE ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('DELETE', OLD.id, OLD.name);
            END
        """)
        db.execute("DELETE FROM users WHERE id = 1")
        audit = db.execute("SELECT * FROM audit_log")
        assert audit.rows[0][0] == 'DELETE'
        assert audit.rows[0][1] == 1
        assert audit.rows[0][2] == 'Alice'

    def test_signal_cancels_delete(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Admin')")
        db.execute("""
            CREATE TRIGGER protect_admin
            BEFORE DELETE ON users
            FOR EACH ROW
            WHEN (OLD.name = 'Admin')
            BEGIN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Cannot delete admin';
            END
        """)
        with pytest.raises(DatabaseError, match="Cannot delete admin"):
            db.execute("DELETE FROM users WHERE id = 1")

        # User should still exist
        result = db.execute("SELECT COUNT(*) AS cnt FROM users")
        assert result.rows[0][0] == 1


# =============================================================================
# 11. AFTER DELETE triggers
# =============================================================================

class TestAfterDelete:
    def test_after_delete_audit(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("""
            CREATE TRIGGER after_delete
            AFTER DELETE ON users
            FOR EACH ROW
            BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('AFTER_DELETE', OLD.id, OLD.name);
            END
        """)
        db.execute("DELETE FROM users WHERE id = 1")
        audit = db.execute("SELECT * FROM audit_log")
        assert audit.rows[0][0] == 'AFTER_DELETE'

    def test_cascade_cleanup(self):
        """AFTER DELETE trigger cleans up related data."""
        db = TriggerDB()
        db.execute("CREATE TABLE orders (id INT, user_id INT, product VARCHAR)")
        db.execute("CREATE TABLE users (id INT, name VARCHAR)")
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO orders (id, user_id, product) VALUES (1, 1, 'Book')")
        db.execute("INSERT INTO orders (id, user_id, product) VALUES (2, 1, 'Pen')")
        db.execute("""
            CREATE TRIGGER cascade_delete
            AFTER DELETE ON users
            FOR EACH ROW
            BEGIN
                DELETE FROM orders WHERE user_id = OLD.id;
            END
        """)
        db.execute("DELETE FROM users WHERE id = 1")
        orders = db.execute("SELECT COUNT(*) AS cnt FROM orders")
        assert orders.rows[0][0] == 0


# =============================================================================
# 12. WHEN condition
# =============================================================================

class TestWhenCondition:
    def test_when_fires_conditionally(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER only_alice
            BEFORE INSERT ON users
            FOR EACH ROW
            WHEN (NEW.name = 'Alice')
            BEGIN
                INSERT INTO audit_log (action) VALUES ('ALICE_INSERT');
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 1

    def test_when_with_comparison(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("""
            CREATE TRIGGER salary_check
            BEFORE UPDATE ON users
            FOR EACH ROW
            WHEN (NEW.id > 0)
            BEGIN
                INSERT INTO audit_log (action) VALUES ('VALID_UPDATE');
            END
        """)
        db.execute("UPDATE users SET name = 'Bob' WHERE id = 1")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 1

    def test_when_with_is_null(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER null_check
            BEFORE INSERT ON users
            FOR EACH ROW
            WHEN (NEW.email IS NULL)
            BEGIN
                SET NEW.email = 'default@example.com';
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        result = db.execute("SELECT email FROM users WHERE id = 1")
        assert result.rows[0][0] == 'default@example.com'

        # Insert with email -- trigger should not fire
        db.execute("INSERT INTO users (id, name, email) VALUES (2, 'Bob', 'bob@b.com')")
        result = db.execute("SELECT email FROM users WHERE id = 2")
        assert result.rows[0][0] == 'bob@b.com'


# =============================================================================
# 13. Trigger body: IF/ELSEIF/ELSE
# =============================================================================

class TestTriggerIfElse:
    def test_if_then(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER classify_user
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                IF NEW.id > 100 THEN
                    SET NEW.email = 'vip';
                END IF;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (200, 'VIP')")
        db.execute("INSERT INTO users (id, name) VALUES (50, 'Regular')")
        r1 = db.execute("SELECT email FROM users WHERE id = 200")
        assert r1.rows[0][0] == 'vip'
        r2 = db.execute("SELECT email FROM users WHERE id = 50")
        assert r2.rows[0][0] is None

    def test_if_else(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER tag_user
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                IF NEW.id >= 100 THEN
                    SET NEW.email = 'premium';
                ELSE
                    SET NEW.email = 'basic';
                END IF;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (150, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (50, 'B')")
        r1 = db.execute("SELECT email FROM users WHERE id = 150")
        assert r1.rows[0][0] == 'premium'
        r2 = db.execute("SELECT email FROM users WHERE id = 50")
        assert r2.rows[0][0] == 'basic'

    def test_if_elseif_else(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER tier_user
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                IF NEW.id >= 1000 THEN
                    SET NEW.email = 'gold';
                ELSEIF NEW.id >= 100 THEN
                    SET NEW.email = 'silver';
                ELSE
                    SET NEW.email = 'bronze';
                END IF;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1500, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (500, 'B')")
        db.execute("INSERT INTO users (id, name) VALUES (10, 'C')")
        r1 = db.execute("SELECT email FROM users WHERE id = 1500")
        assert r1.rows[0][0] == 'gold'
        r2 = db.execute("SELECT email FROM users WHERE id = 500")
        assert r2.rows[0][0] == 'silver'
        r3 = db.execute("SELECT email FROM users WHERE id = 10")
        assert r3.rows[0][0] == 'bronze'


# =============================================================================
# 14. SIGNAL statement
# =============================================================================

class TestSignal:
    def test_signal_basic(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER reject_all
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'No inserts allowed';
            END
        """)
        with pytest.raises(DatabaseError, match="No inserts allowed"):
            db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")

    def test_conditional_signal(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER validate_name
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                IF NEW.name IS NULL THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Name required';
                END IF;
            END
        """)
        with pytest.raises(DatabaseError, match="Name required"):
            db.execute("INSERT INTO users (id) VALUES (1)")

        # Valid insert should work
        db.execute("INSERT INTO users (id, name) VALUES (2, 'Alice')")
        result = db.execute("SELECT COUNT(*) AS cnt FROM users")
        assert result.rows[0][0] == 1


# =============================================================================
# 15. Multiple triggers per table
# =============================================================================

class TestMultipleTriggers:
    def test_multiple_before_insert(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                SET NEW.name = UPPER(NEW.name);
            END
        """)
        db.execute("""
            CREATE TRIGGER t2 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                SET NEW.email = CONCAT(NEW.name, '@auto.com');
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
        result = db.execute("SELECT name, email FROM users WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'
        assert result.rows[0][1] == 'ALICE@auto.com'

    def test_before_and_after(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER before_t BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                SET NEW.name = UPPER(NEW.name);
            END
        """)
        db.execute("""
            CREATE TRIGGER after_t AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, details) VALUES ('AFTER', NEW.name);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
        result = db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'
        audit = db.execute("SELECT details FROM audit_log")
        assert audit.rows[0][0] == 'ALICE'


# =============================================================================
# 16. Trigger ordering (creation order)
# =============================================================================

class TestTriggerOrdering:
    def test_execution_order(self):
        db = make_db()
        # Create two BEFORE INSERT triggers that both modify name
        db.execute("""
            CREATE TRIGGER first_trigger BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                SET NEW.email = 'first';
            END
        """)
        db.execute("""
            CREATE TRIGGER second_trigger BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                SET NEW.email = CONCAT(NEW.email, '_second');
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")
        result = db.execute("SELECT email FROM users WHERE id = 1")
        # First trigger sets email='first', second trigger appends '_second'
        assert result.rows[0][0] == 'first_second'


# =============================================================================
# 17. Nested triggers (trigger firing another trigger)
# =============================================================================

class TestNestedTriggers:
    def test_cascading_triggers(self):
        """Insert into users triggers insert into audit_log, which has its own trigger."""
        db = TriggerDB()
        db.execute("CREATE TABLE users (id INT, name VARCHAR)")
        db.execute("CREATE TABLE audit_log (action VARCHAR, target VARCHAR)")
        db.execute("CREATE TABLE meta_log (event VARCHAR)")

        db.execute("""
            CREATE TRIGGER user_audit AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, target) VALUES ('USER_INSERT', NEW.name);
            END
        """)
        db.execute("""
            CREATE TRIGGER audit_meta AFTER INSERT ON audit_log
            FOR EACH ROW BEGIN
                INSERT INTO meta_log (event) VALUES ('AUDIT_LOGGED');
            END
        """)

        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")

        audit = db.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 1
        meta = db.execute("SELECT * FROM meta_log")
        assert len(meta.rows) == 1

    def test_trigger_depth_limit(self):
        """Self-referencing trigger should hit depth limit."""
        db = TriggerDB()
        db.execute("CREATE TABLE counter (n INT)")
        db.execute("INSERT INTO counter (n) VALUES (0)")
        db.execute("""
            CREATE TRIGGER infinite_loop AFTER UPDATE ON counter
            FOR EACH ROW
            WHEN (NEW.n < 100)
            BEGIN
                UPDATE counter SET n = NEW.n + 1 WHERE n = NEW.n;
            END
        """)
        with pytest.raises(DatabaseError, match="nesting depth"):
            db.execute("UPDATE counter SET n = 1 WHERE n = 0")


# =============================================================================
# 18. Trigger expressions: functions, arithmetic
# =============================================================================

class TestTriggerExpressions:
    def test_arithmetic_in_set(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER double_id
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                SET NEW.id = NEW.id * 2;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (5, 'Alice')")
        result = db.execute("SELECT id FROM users")
        assert result.rows[0][0] == 10

    def test_string_functions(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER format_name
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                SET NEW.name = UPPER(NEW.name);
                SET NEW.email = LOWER(NEW.email);
            END
        """)
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'alice', 'ALICE@A.COM')")
        result = db.execute("SELECT name, email FROM users WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'
        assert result.rows[0][1] == 'alice@a.com'

    def test_coalesce_in_trigger(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER default_name
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                SET NEW.name = COALESCE(NEW.name, 'Unknown');
            END
        """)
        db.execute("INSERT INTO users (id) VALUES (1)")
        result = db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'Unknown'

    def test_case_expression_in_when(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER classify
            BEFORE INSERT ON users
            FOR EACH ROW
            WHEN (NEW.id > 0)
            BEGIN
                SET NEW.email = CASE
                    WHEN NEW.id >= 100 THEN 'premium'
                    WHEN NEW.id >= 10 THEN 'standard'
                    ELSE 'basic'
                END;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (200, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (50, 'B')")
        db.execute("INSERT INTO users (id, name) VALUES (5, 'C')")
        r1 = db.execute("SELECT email FROM users WHERE id = 200")
        assert r1.rows[0][0] == 'premium'
        r2 = db.execute("SELECT email FROM users WHERE id = 50")
        assert r2.rows[0][0] == 'standard'
        r3 = db.execute("SELECT email FROM users WHERE id = 5")
        assert r3.rows[0][0] == 'basic'


# =============================================================================
# 19. Trigger with DECLARE variables
# =============================================================================

class TestTriggerDeclare:
    def test_declare_and_set(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER compute_email
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                DECLARE prefix VARCHAR DEFAULT 'user';
                SET NEW.email = CONCAT(prefix, '_', NEW.name);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
        result = db.execute("SELECT email FROM users WHERE id = 1")
        assert result.rows[0][0] == 'user_alice'


# =============================================================================
# 20. Trigger on DROP TABLE cascades
# =============================================================================

class TestDropTableCascade:
    def test_drop_table_drops_triggers(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('x');
            END
        """)
        db.execute("DROP TABLE users")
        result = db.execute("SHOW TRIGGERS")
        assert len(result.rows) == 0

    def test_drop_table_with_view_dep_fails(self):
        db = make_db()
        db.execute("CREATE VIEW user_view AS SELECT id, name FROM users")
        with pytest.raises(DatabaseError, match="referenced by view"):
            db.execute("DROP TABLE users")


# =============================================================================
# 21. INSTEAD OF triggers on views
# =============================================================================

class TestInsteadOfTriggers:
    def test_instead_of_insert(self):
        db = make_db()
        db.execute("CREATE VIEW user_names AS SELECT id, name FROM users")
        db.execute("""
            CREATE TRIGGER io_insert INSTEAD OF INSERT ON user_names
            FOR EACH ROW BEGIN
                INSERT INTO users (id, name, email)
                VALUES (NEW.id, NEW.name, 'via_view@test.com');
            END
        """)
        db.execute("INSERT INTO user_names (id, name) VALUES (1, 'Alice')")
        result = db.execute("SELECT * FROM users")
        assert len(result.rows) == 1
        assert result.rows[0][2] == 'via_view@test.com'

    def test_instead_of_delete(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'a@b.com')")
        db.execute("CREATE VIEW active_users AS SELECT id, name FROM users")
        db.execute("""
            CREATE TRIGGER io_delete INSTEAD OF DELETE ON active_users
            FOR EACH ROW BEGIN
                UPDATE users SET email = 'deactivated' WHERE id = OLD.id;
            END
        """)
        db.execute("DELETE FROM active_users WHERE id = 1")
        # User should still exist but be deactivated
        result = db.execute("SELECT email FROM users WHERE id = 1")
        assert result.rows[0][0] == 'deactivated'

    def test_instead_of_on_table_fails(self):
        db = make_db()
        with pytest.raises(DatabaseError, match="INSTEAD OF.*only.*views"):
            db.execute("""
                CREATE TRIGGER bad INSTEAD OF INSERT ON users
                FOR EACH ROW BEGIN
                    INSERT INTO audit_log (action) VALUES ('x');
                END
            """)

    def test_before_after_on_view_fails(self):
        db = make_db()
        db.execute("CREATE VIEW v1 AS SELECT * FROM users")
        with pytest.raises(DatabaseError, match="BEFORE/AFTER.*cannot.*views"):
            db.execute("""
                CREATE TRIGGER bad BEFORE INSERT ON v1
                FOR EACH ROW BEGIN
                    INSERT INTO audit_log (action) VALUES ('x');
                END
            """)


# =============================================================================
# 22. Trigger interop with views
# =============================================================================

class TestTriggerViewInterop:
    def test_trigger_fires_on_view_base_table(self):
        """When inserting through an updatable view, BEFORE trigger on base table fires."""
        db = make_db()
        db.execute("CREATE VIEW simple_users AS SELECT id, name FROM users")
        db.execute("""
            CREATE TRIGGER upper_name BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                SET NEW.name = UPPER(NEW.name);
            END
        """)
        # Insert through updatable view goes to base table, trigger should fire
        db.execute("INSERT INTO simple_users (id, name) VALUES (1, 'alice')")
        result = db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'


# =============================================================================
# 23. Trigger interop with stored procedures
# =============================================================================

class TestTriggerProcInterop:
    def test_trigger_fires_on_proc_insert(self):
        """When a stored procedure inserts, triggers should fire."""
        db = make_db()
        db.execute("""
            CREATE TRIGGER audit_insert AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id) VALUES ('FROM_PROC', NEW.id);
            END
        """)
        db.execute("""
            CREATE PROCEDURE add_user(IN p_id INT, IN p_name VARCHAR)
            BEGIN
                INSERT INTO users (id, name) VALUES (p_id, p_name);
            END
        """)
        db.execute("CALL add_user(1, 'Alice')")
        audit = db.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 1
        assert audit.rows[0][0] == 'FROM_PROC'


# =============================================================================
# 24. Trigger with SQL DML in body
# =============================================================================

class TestTriggerDMLBody:
    def test_insert_in_body(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER replicate
            AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('REPLICATE', NEW.id, NEW.name);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        audit = db.execute("SELECT * FROM audit_log")
        assert len(audit.rows) == 1

    def test_update_in_body(self):
        db = make_db()
        db.execute("CREATE TABLE counters (name VARCHAR, val INT)")
        db.execute("INSERT INTO counters (name, val) VALUES ('inserts', 0)")
        db.execute("""
            CREATE TRIGGER count_inserts
            AFTER INSERT ON users
            FOR EACH ROW BEGIN
                UPDATE counters SET val = val + 1 WHERE name = 'inserts';
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'B')")
        result = db.execute("SELECT val FROM counters WHERE name = 'inserts'")
        assert result.rows[0][0] == 2

    def test_delete_in_body(self):
        db = make_db()
        db.execute("INSERT INTO audit_log (action, user_id) VALUES ('old_entry', 99)")
        db.execute("""
            CREATE TRIGGER cleanup_audit
            BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                DELETE FROM audit_log WHERE user_id = 99;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log WHERE user_id = 99")
        assert audit.rows[0][0] == 0


# =============================================================================
# 25. Full inherited functionality (C247-C250)
# =============================================================================

class TestInheritedFunctionality:
    def test_basic_crud(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'a@b.com')")
        result = db.execute("SELECT * FROM users")
        assert len(result.rows) == 1

    def test_views_still_work(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'a@b.com')")
        db.execute("CREATE VIEW user_names AS SELECT id, name FROM users")
        result = db.execute("SELECT * FROM user_names")
        assert len(result.rows) == 1

    def test_procs_still_work(self):
        db = make_db()
        db.execute("""
            CREATE FUNCTION add_one(IN x INT) RETURNS INT
            BEGIN
                RETURN x + 1;
            END
        """)
        result = db.execute("SELECT add_one(5) AS result")
        assert result.rows[0][0] == 6

    def test_transactions(self):
        db = make_db()
        db.execute("BEGIN")
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("ROLLBACK")
        result = db.execute("SELECT COUNT(*) AS cnt FROM users")
        assert result.rows[0][0] == 0

    def test_subqueries(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")
        result = db.execute("SELECT * FROM users WHERE id IN (1, 2)")
        assert len(result.rows) == 2

    def test_joins(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO audit_log (action, user_id, details) VALUES ('test', 1, 'detail')")
        result = db.execute("""
            SELECT users.name, audit_log.action
            FROM users
            JOIN audit_log ON users.id = audit_log.user_id
        """)
        assert len(result.rows) == 1

    def test_aggregations(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'B')")
        db.execute("INSERT INTO users (id, name) VALUES (3, 'C')")
        result = db.execute("SELECT COUNT(*) AS cnt, MAX(id) AS max_id FROM users")
        assert result.rows[0][0] == 3
        assert result.rows[0][1] == 3


# =============================================================================
# 26. Edge cases
# =============================================================================

class TestEdgeCases:
    def test_trigger_on_empty_table(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 AFTER DELETE ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('deleted');
            END
        """)
        # Delete from empty table -- no rows match, no trigger fires
        db.execute("DELETE FROM users WHERE id = 1")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 0

    def test_update_no_matching_rows(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE UPDATE ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('updated');
            END
        """)
        db.execute("UPDATE users SET name = 'x' WHERE id = 999")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 0

    def test_disabled_trigger_not_fired(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action) VALUES ('fired');
            END
        """)
        db.execute("ALTER TRIGGER t1 DISABLE")
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 0

    def test_null_handling_in_new(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER null_check
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                IF NEW.email IS NULL THEN
                    SET NEW.email = 'none';
                END IF;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        result = db.execute("SELECT email FROM users WHERE id = 1")
        assert result.rows[0][0] == 'none'

    def test_trigger_preserves_data_types(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER preserve_types
            BEFORE INSERT ON users
            FOR EACH ROW
            BEGIN
                SET NEW.id = NEW.id + 0;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (42, 'Alice')")
        result = db.execute("SELECT id FROM users WHERE name = 'Alice'")
        assert result.rows[0][0] == 42
        assert isinstance(result.rows[0][0], int)


# =============================================================================
# 27. Complex scenario: full audit trail
# =============================================================================

class TestComplexScenarios:
    def test_full_audit_trail(self):
        """Complete audit trail for all CRUD operations."""
        db = make_db()

        db.execute("""
            CREATE TRIGGER audit_insert AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('INSERT', NEW.id, NEW.name);
            END
        """)
        db.execute("""
            CREATE TRIGGER audit_update AFTER UPDATE ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('UPDATE', NEW.id, CONCAT(OLD.name, '->', NEW.name));
            END
        """)
        db.execute("""
            CREATE TRIGGER audit_delete AFTER DELETE ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('DELETE', OLD.id, OLD.name);
            END
        """)

        # INSERT
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        # UPDATE
        db.execute("UPDATE users SET name = 'Bob' WHERE id = 1")
        # DELETE
        db.execute("DELETE FROM users WHERE id = 1")

        audit = db.execute("SELECT action, user_id, details FROM audit_log ORDER BY user_id")
        assert len(audit.rows) == 3
        assert audit.rows[0][0] == 'INSERT'
        assert audit.rows[1][0] == 'UPDATE'
        assert audit.rows[1][2] == 'Alice->Bob'
        assert audit.rows[2][0] == 'DELETE'
        assert audit.rows[2][2] == 'Bob'

    def test_data_validation_pipeline(self):
        """Multiple triggers forming a validation pipeline."""
        db = make_db()

        # Trigger 1: Validate
        db.execute("""
            CREATE TRIGGER validate BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                IF NEW.id <= 0 THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Invalid ID';
                END IF;
            END
        """)

        # Trigger 2: Normalize
        db.execute("""
            CREATE TRIGGER normalize BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                SET NEW.name = UPPER(NEW.name);
            END
        """)

        # Trigger 3: Audit
        db.execute("""
            CREATE TRIGGER audit AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id, details)
                VALUES ('VALIDATED_INSERT', NEW.id, NEW.name);
            END
        """)

        # Invalid insert blocked
        with pytest.raises(DatabaseError, match="Invalid ID"):
            db.execute("INSERT INTO users (id, name) VALUES (-1, 'bad')")

        # Valid insert goes through pipeline
        db.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
        result = db.execute("SELECT name FROM users WHERE id = 1")
        assert result.rows[0][0] == 'ALICE'
        audit = db.execute("SELECT details FROM audit_log")
        assert audit.rows[0][0] == 'ALICE'

    def test_soft_delete_pattern(self):
        """INSTEAD OF DELETE on view implements soft delete."""
        db = TriggerDB()
        db.execute("CREATE TABLE users (id INT, name VARCHAR, deleted INT)")
        db.execute("INSERT INTO users (id, name, deleted) VALUES (1, 'Alice', 0)")
        db.execute("INSERT INTO users (id, name, deleted) VALUES (2, 'Bob', 0)")

        db.execute("CREATE VIEW active_users AS SELECT id, name FROM users WHERE deleted = 0")

        db.execute("""
            CREATE TRIGGER soft_delete INSTEAD OF DELETE ON active_users
            FOR EACH ROW BEGIN
                UPDATE users SET deleted = 1 WHERE id = OLD.id;
            END
        """)

        db.execute("DELETE FROM active_users WHERE id = 1")

        # User still exists in base table
        result = db.execute("SELECT COUNT(*) AS cnt FROM users")
        assert result.rows[0][0] == 2

        # But not visible through view
        result = db.execute("SELECT COUNT(*) AS cnt FROM active_users")
        assert result.rows[0][0] == 1

    def test_auto_increment_pattern(self):
        """BEFORE INSERT trigger simulates auto-increment."""
        db = TriggerDB()
        db.execute("CREATE TABLE items (id INT, name VARCHAR)")
        db.execute("CREATE TABLE sequences (name VARCHAR, next_val INT)")
        db.execute("INSERT INTO sequences (name, next_val) VALUES ('items_id', 1)")

        db.execute("""
            CREATE TRIGGER auto_id BEFORE INSERT ON items
            FOR EACH ROW BEGIN
                IF NEW.id IS NULL THEN
                    SET NEW.id = 1;
                END IF;
            END
        """)

        db.execute("INSERT INTO items (name) VALUES ('Widget')")
        result = db.execute("SELECT id, name FROM items")
        assert result.rows[0][0] == 1

    def test_cross_table_integrity(self):
        """Trigger enforces referential integrity."""
        db = TriggerDB()
        db.execute("CREATE TABLE departments (id INT, name VARCHAR)")
        db.execute("CREATE TABLE employees (id INT, dept_id INT, name VARCHAR)")

        db.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
        db.execute("INSERT INTO departments (id, name) VALUES (2, 'Sales')")

        db.execute("""
            CREATE TRIGGER check_dept BEFORE INSERT ON employees
            FOR EACH ROW BEGIN
                IF NEW.dept_id IS NOT NULL THEN
                    DECLARE dept_exists INT DEFAULT 0;
                    SET dept_exists = NEW.dept_id;
                END IF;
            END
        """)

        db.execute("INSERT INTO employees (id, dept_id, name) VALUES (1, 1, 'Alice')")
        result = db.execute("SELECT COUNT(*) AS cnt FROM employees")
        assert result.rows[0][0] == 1


# =============================================================================
# 28. Trigger with local variable operations
# =============================================================================

class TestTriggerLocalVars:
    def test_declare_default(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER var_test BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                DECLARE counter INT DEFAULT 42;
                SET NEW.id = counter;
            END
        """)
        db.execute("INSERT INTO users (name) VALUES ('Alice')")
        result = db.execute("SELECT id FROM users")
        assert result.rows[0][0] == 42

    def test_variable_arithmetic(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER calc BEFORE INSERT ON users
            FOR EACH ROW BEGIN
                DECLARE base INT DEFAULT 100;
                SET NEW.id = base + NEW.id;
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (5, 'Alice')")
        result = db.execute("SELECT id FROM users")
        assert result.rows[0][0] == 105


# =============================================================================
# 29. Trigger interop with existing SQL features
# =============================================================================

class TestTriggerSQLInterop:
    def test_trigger_with_order_by(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (3, 'C')")
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'B')")

        db.execute("""
            CREATE TRIGGER t1 AFTER DELETE ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id) VALUES ('DEL', OLD.id);
            END
        """)

        db.execute("DELETE FROM users WHERE id = 2")
        result = db.execute("SELECT * FROM users ORDER BY id")
        assert len(result.rows) == 2
        audit = db.execute("SELECT user_id FROM audit_log")
        assert audit.rows[0][0] == 2

    def test_trigger_with_aggregate_query(self):
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id) VALUES ('INSERT', NEW.id);
            END
        """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'A')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'B')")

        result = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert result.rows[0][0] == 2

    def test_describe_table_still_works(self):
        db = make_db()
        result = db.execute("DESCRIBE users")
        assert len(result.rows) >= 3


# =============================================================================
# 30. Stress/boundary tests
# =============================================================================

class TestStressBoundary:
    def test_many_triggers(self):
        """10 triggers on same table."""
        db = make_db()
        for i in range(10):
            db.execute(f"""
                CREATE TRIGGER t{i} AFTER INSERT ON users
                FOR EACH ROW BEGIN
                    INSERT INTO audit_log (action) VALUES ('trigger_{i}');
                END
            """)
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 10

    def test_many_rows_with_trigger(self):
        """Insert 50 rows, each fires a trigger."""
        db = make_db()
        db.execute("""
            CREATE TRIGGER t1 AFTER INSERT ON users
            FOR EACH ROW BEGIN
                INSERT INTO audit_log (action, user_id) VALUES ('INSERT', NEW.id);
            END
        """)
        for i in range(50):
            db.execute(f"INSERT INTO users (id, name) VALUES ({i}, 'User{i}')")
        audit = db.execute("SELECT COUNT(*) AS cnt FROM audit_log")
        assert audit.rows[0][0] == 50

    def test_trigger_with_complex_where(self):
        db = make_db()
        db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')")
        db.execute("INSERT INTO users (id, name) VALUES (3, 'Charlie')")

        db.execute("""
            CREATE TRIGGER t1 BEFORE DELETE ON users
            FOR EACH ROW
            WHEN (OLD.id > 1 AND OLD.name != 'Charlie')
            BEGIN
                INSERT INTO audit_log (action, user_id) VALUES ('CONDITIONAL_DEL', OLD.id);
            END
        """)

        db.execute("DELETE FROM users WHERE id >= 1")
        audit = db.execute("SELECT * FROM audit_log")
        # Only id=2 (Bob) should match WHEN: id>1 AND name!='Charlie'
        assert len(audit.rows) == 1
        assert audit.rows[0][1] == 2


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
