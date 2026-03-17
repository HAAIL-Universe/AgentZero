"""
Tests for C249: Stored Procedures & User-Defined Functions
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from stored_procedures import (
    ProcDB, ProcLexer, ProcParser, proc_parse,
    CreateFunctionStmt, CreateProcedureStmt, CallStmt,
    DropFunctionStmt, DropProcedureStmt,
    DeclareVarStmt, SetVarStmt, ReturnStmt, IfStmt, WhileStmt,
    SelectIntoStmt, ShowFunctionsStmt, ShowProceduresStmt,
    ParamDef, ParamMode, StoredFunction, StoredProcedure,
    RoutineCatalog, ProcExecutor, DatabaseError, ResultSet,
)


# =============================================================================
# Helpers
# =============================================================================

def make_db():
    """Create a ProcDB with some test tables."""
    db = ProcDB()
    db.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, salary FLOAT, dept TEXT)")
    db.execute("INSERT INTO employees VALUES (1, 'Alice', 50000, 'engineering')")
    db.execute("INSERT INTO employees VALUES (2, 'Bob', 60000, 'engineering')")
    db.execute("INSERT INTO employees VALUES (3, 'Charlie', 45000, 'sales')")
    db.execute("INSERT INTO employees VALUES (4, 'Diana', 70000, 'engineering')")
    db.execute("INSERT INTO employees VALUES (5, 'Eve', 55000, 'sales')")
    return db


# =============================================================================
# 1. Parsing Tests
# =============================================================================

class TestParsing:
    """Test the extended parser for procedure/function syntax."""

    def test_parse_create_function_simple(self):
        stmts = proc_parse("""
            CREATE FUNCTION add_one(x INT) RETURNS INT
            BEGIN
                RETURN x + 1;
            END
        """)
        assert len(stmts) == 1
        assert isinstance(stmts[0], CreateFunctionStmt)
        assert stmts[0].name == 'add_one'
        assert len(stmts[0].params) == 1
        assert stmts[0].params[0].name == 'x'
        assert stmts[0].return_type == 'int'

    def test_parse_create_function_multi_param(self):
        stmts = proc_parse("""
            CREATE FUNCTION calc(a INT, b FLOAT) RETURNS FLOAT
            BEGIN
                RETURN a + b;
            END
        """)
        assert len(stmts[0].params) == 2
        assert stmts[0].params[0].param_type == 'int'
        assert stmts[0].params[1].param_type == 'float'

    def test_parse_create_function_deterministic(self):
        stmts = proc_parse("""
            CREATE FUNCTION square(x INT) RETURNS INT DETERMINISTIC
            BEGIN
                RETURN x * x;
            END
        """)
        assert stmts[0].deterministic is True

    def test_parse_create_or_replace_function(self):
        stmts = proc_parse("""
            CREATE OR REPLACE FUNCTION greet(name TEXT) RETURNS TEXT
            BEGIN
                RETURN name;
            END
        """)
        assert stmts[0].replace is True

    def test_parse_create_procedure(self):
        stmts = proc_parse("""
            CREATE PROCEDURE update_salary(IN emp_id INT, IN amount FLOAT)
            BEGIN
                UPDATE employees SET salary = amount WHERE id = emp_id;
            END
        """)
        assert isinstance(stmts[0], CreateProcedureStmt)
        assert stmts[0].name == 'update_salary'
        assert len(stmts[0].params) == 2
        assert stmts[0].params[0].mode == ParamMode.IN
        assert stmts[0].params[1].mode == ParamMode.IN

    def test_parse_procedure_out_param(self):
        stmts = proc_parse("""
            CREATE PROCEDURE get_count(OUT total INT)
            BEGIN
                SET total = 42;
            END
        """)
        assert stmts[0].params[0].mode == ParamMode.OUT

    def test_parse_procedure_inout_param(self):
        stmts = proc_parse("""
            CREATE PROCEDURE double_it(INOUT val INT)
            BEGIN
                SET val = val * 2;
            END
        """)
        assert stmts[0].params[0].mode == ParamMode.INOUT

    def test_parse_call(self):
        stmts = proc_parse("CALL my_proc(1, 'hello')")
        assert isinstance(stmts[0], CallStmt)
        assert stmts[0].name == 'my_proc'
        assert len(stmts[0].args) == 2

    def test_parse_drop_function(self):
        stmts = proc_parse("DROP FUNCTION my_func")
        assert isinstance(stmts[0], DropFunctionStmt)
        assert stmts[0].name == 'my_func'
        assert stmts[0].if_exists is False

    def test_parse_drop_function_if_exists(self):
        stmts = proc_parse("DROP FUNCTION IF EXISTS my_func")
        assert stmts[0].if_exists is True

    def test_parse_drop_procedure(self):
        stmts = proc_parse("DROP PROCEDURE my_proc")
        assert isinstance(stmts[0], DropProcedureStmt)

    def test_parse_declare_var(self):
        stmts = proc_parse("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE x INT DEFAULT 0;
                RETURN x;
            END
        """)
        body = stmts[0].body
        assert isinstance(body[0], DeclareVarStmt)
        assert body[0].name == 'x'
        assert body[0].var_type == 'int'

    def test_parse_if_then_else(self):
        stmts = proc_parse("""
            CREATE FUNCTION sign(x INT) RETURNS INT
            BEGIN
                IF x > 0 THEN
                    RETURN 1;
                ELSEIF x < 0 THEN
                    RETURN -1;
                ELSE
                    RETURN 0;
                END IF;
            END
        """)
        body = stmts[0].body
        assert isinstance(body[0], IfStmt)
        assert len(body[0].elseif_clauses) == 1
        assert body[0].else_body is not None

    def test_parse_while_loop(self):
        stmts = proc_parse("""
            CREATE FUNCTION sum_to(n INT) RETURNS INT
            BEGIN
                DECLARE result INT DEFAULT 0;
                DECLARE i INT DEFAULT 1;
                WHILE i <= n DO
                    SET result = result + i;
                    SET i = i + 1;
                END WHILE;
                RETURN result;
            END
        """)
        body = stmts[0].body
        assert isinstance(body[2], WhileStmt)

    def test_parse_select_into(self):
        stmts = proc_parse("""
            CREATE PROCEDURE get_salary(IN emp_id INT, OUT sal FLOAT)
            BEGIN
                SELECT salary INTO sal FROM employees WHERE id = emp_id;
            END
        """)
        body = stmts[0].body
        assert isinstance(body[0], SelectIntoStmt)
        assert body[0].variables == ['sal']

    def test_parse_show_functions(self):
        stmts = proc_parse("SHOW FUNCTIONS")
        assert isinstance(stmts[0], ShowFunctionsStmt)

    def test_parse_show_procedures(self):
        stmts = proc_parse("SHOW PROCEDURES")
        assert isinstance(stmts[0], ShowProceduresStmt)

    def test_parse_no_params(self):
        stmts = proc_parse("""
            CREATE FUNCTION get_pi() RETURNS FLOAT
            BEGIN
                RETURN 3.14159;
            END
        """)
        assert len(stmts[0].params) == 0

    def test_parse_labeled_while(self):
        stmts = proc_parse("""
            CREATE PROCEDURE p()
            BEGIN
                DECLARE i INT DEFAULT 0;
                loop1: WHILE i < 10 DO
                    SET i = i + 1;
                    IF i = 5 THEN
                        LEAVE loop1;
                    END IF;
                END WHILE;
            END
        """)
        body = stmts[0].body
        assert isinstance(body[1], WhileStmt)
        assert body[1].label == 'loop1'

    def test_parse_standard_sql_through_proc_parser(self):
        """Standard SQL should still work through the extended parser."""
        stmts = proc_parse("SELECT 1 + 2")
        assert len(stmts) == 1

    def test_parse_create_table_through_proc_parser(self):
        stmts = proc_parse("CREATE TABLE test (id INT PRIMARY KEY, name TEXT)")
        assert len(stmts) == 1

    def test_parse_multiple_statements(self):
        stmts = proc_parse("""
            CREATE FUNCTION f(x INT) RETURNS INT
            BEGIN RETURN x; END;
            CALL f(1);
        """)
        assert len(stmts) == 2
        assert isinstance(stmts[0], CreateFunctionStmt)
        assert isinstance(stmts[1], CallStmt)


# =============================================================================
# 2. Routine Catalog Tests
# =============================================================================

class TestRoutineCatalog:

    def test_create_function(self):
        cat = RoutineCatalog()
        func = StoredFunction(name='f', params=[], return_type='int', body=[])
        cat.create_function(func)
        assert cat.get_function('f') is func

    def test_create_duplicate_function_error(self):
        cat = RoutineCatalog()
        func = StoredFunction(name='f', params=[], return_type='int', body=[])
        cat.create_function(func)
        with pytest.raises(DatabaseError, match="already exists"):
            cat.create_function(func)

    def test_create_or_replace_function(self):
        cat = RoutineCatalog()
        func1 = StoredFunction(name='f', params=[], return_type='int', body=[])
        func2 = StoredFunction(name='f', params=[], return_type='float', body=[])
        cat.create_function(func1)
        cat.create_function(func2, replace=True)
        assert cat.get_function('f').return_type == 'float'

    def test_drop_function(self):
        cat = RoutineCatalog()
        func = StoredFunction(name='f', params=[], return_type='int', body=[])
        cat.create_function(func)
        cat.drop_function('f')
        with pytest.raises(DatabaseError):
            cat.get_function('f')

    def test_drop_function_if_exists(self):
        cat = RoutineCatalog()
        cat.drop_function('nonexistent', if_exists=True)  # no error

    def test_drop_function_not_found(self):
        cat = RoutineCatalog()
        with pytest.raises(DatabaseError, match="not found"):
            cat.drop_function('nonexistent')

    def test_create_procedure(self):
        cat = RoutineCatalog()
        proc = StoredProcedure(name='p', params=[], body=[])
        cat.create_procedure(proc)
        assert cat.get_procedure('p') is proc

    def test_list_functions(self):
        cat = RoutineCatalog()
        cat.create_function(StoredFunction('b', [], 'int', []))
        cat.create_function(StoredFunction('a', [], 'int', []))
        assert cat.list_functions() == ['a', 'b']

    def test_list_procedures(self):
        cat = RoutineCatalog()
        cat.create_procedure(StoredProcedure('y', [], []))
        cat.create_procedure(StoredProcedure('x', [], []))
        assert cat.list_procedures() == ['x', 'y']


# =============================================================================
# 3. Simple UDF Tests
# =============================================================================

class TestSimpleUDFs:

    def test_constant_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION get_pi() RETURNS FLOAT
            BEGIN
                RETURN 3.14159;
            END
        """)
        result = db.execute("CALL get_pi()")
        assert abs(result.rows[0][0] - 3.14159) < 0.001

    def test_identity_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION identity(x INT) RETURNS INT
            BEGIN
                RETURN x;
            END
        """)
        result = db.execute("CALL identity(42)")
        assert result.rows[0][0] == 42

    def test_add_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION add(a INT, b INT) RETURNS INT
            BEGIN
                RETURN a + b;
            END
        """)
        result = db.execute("CALL add(3, 4)")
        assert result.rows[0][0] == 7

    def test_multiply_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION multiply(a INT, b INT) RETURNS INT
            BEGIN
                RETURN a * b;
            END
        """)
        result = db.execute("CALL multiply(6, 7)")
        assert result.rows[0][0] == 42

    def test_string_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION greet(name TEXT) RETURNS TEXT
            BEGIN
                RETURN concat('Hello, ', name);
            END
        """)
        result = db.execute("CALL greet('World')")
        assert result.rows[0][0] == 'Hello, World'

    def test_boolean_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION is_positive(x INT) RETURNS BOOL
            BEGIN
                RETURN x > 0;
            END
        """)
        result = db.execute("CALL is_positive(5)")
        assert result.rows[0][0] is True
        result = db.execute("CALL is_positive(-1)")
        assert result.rows[0][0] is False

    def test_float_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION avg_of_two(a FLOAT, b FLOAT) RETURNS FLOAT
            BEGIN
                RETURN (a + b) / 2;
            END
        """)
        result = db.execute("CALL avg_of_two(3.0, 5.0)")
        assert abs(result.rows[0][0] - 4.0) < 0.001

    def test_null_handling(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION safe_val(x INT) RETURNS INT
            BEGIN
                IF x IS NULL THEN
                    RETURN 0;
                END IF;
                RETURN x;
            END
        """)
        result = db.execute("CALL safe_val(NULL)")
        assert result.rows[0][0] == 0
        result = db.execute("CALL safe_val(5)")
        assert result.rows[0][0] == 5


# =============================================================================
# 4. Control Flow Tests
# =============================================================================

class TestControlFlow:

    def test_if_then(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION abs_val(x INT) RETURNS INT
            BEGIN
                IF x < 0 THEN
                    RETURN x * -1;
                END IF;
                RETURN x;
            END
        """)
        assert db.execute("CALL abs_val(-5)").rows[0][0] == 5
        assert db.execute("CALL abs_val(3)").rows[0][0] == 3

    def test_if_then_else(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION max_of(a INT, b INT) RETURNS INT
            BEGIN
                IF a > b THEN
                    RETURN a;
                ELSE
                    RETURN b;
                END IF;
            END
        """)
        assert db.execute("CALL max_of(3, 5)").rows[0][0] == 5
        assert db.execute("CALL max_of(7, 2)").rows[0][0] == 7

    def test_if_elseif_else(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION classify(x INT) RETURNS TEXT
            BEGIN
                IF x > 0 THEN
                    RETURN 'positive';
                ELSEIF x < 0 THEN
                    RETURN 'negative';
                ELSE
                    RETURN 'zero';
                END IF;
            END
        """)
        assert db.execute("CALL classify(5)").rows[0][0] == 'positive'
        assert db.execute("CALL classify(-3)").rows[0][0] == 'negative'
        assert db.execute("CALL classify(0)").rows[0][0] == 'zero'

    def test_multiple_elseif(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION grade(score INT) RETURNS TEXT
            BEGIN
                IF score >= 90 THEN
                    RETURN 'A';
                ELSEIF score >= 80 THEN
                    RETURN 'B';
                ELSEIF score >= 70 THEN
                    RETURN 'C';
                ELSEIF score >= 60 THEN
                    RETURN 'D';
                ELSE
                    RETURN 'F';
                END IF;
            END
        """)
        assert db.execute("CALL grade(95)").rows[0][0] == 'A'
        assert db.execute("CALL grade(85)").rows[0][0] == 'B'
        assert db.execute("CALL grade(75)").rows[0][0] == 'C'
        assert db.execute("CALL grade(65)").rows[0][0] == 'D'
        assert db.execute("CALL grade(50)").rows[0][0] == 'F'

    def test_while_loop_sum(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION sum_to(n INT) RETURNS INT
            BEGIN
                DECLARE result INT DEFAULT 0;
                DECLARE i INT DEFAULT 1;
                WHILE i <= n DO
                    SET result = result + i;
                    SET i = i + 1;
                END WHILE;
                RETURN result;
            END
        """)
        assert db.execute("CALL sum_to(10)").rows[0][0] == 55
        assert db.execute("CALL sum_to(1)").rows[0][0] == 1
        assert db.execute("CALL sum_to(0)").rows[0][0] == 0

    def test_while_loop_factorial(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION factorial(n INT) RETURNS INT
            BEGIN
                DECLARE result INT DEFAULT 1;
                DECLARE i INT DEFAULT 2;
                WHILE i <= n DO
                    SET result = result * i;
                    SET i = i + 1;
                END WHILE;
                RETURN result;
            END
        """)
        assert db.execute("CALL factorial(5)").rows[0][0] == 120
        assert db.execute("CALL factorial(0)").rows[0][0] == 1
        assert db.execute("CALL factorial(1)").rows[0][0] == 1

    def test_nested_if(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION nested_check(a INT, b INT) RETURNS TEXT
            BEGIN
                IF a > 0 THEN
                    IF b > 0 THEN
                        RETURN 'both positive';
                    ELSE
                        RETURN 'a positive, b not';
                    END IF;
                ELSE
                    RETURN 'a not positive';
                END IF;
            END
        """)
        assert db.execute("CALL nested_check(1, 1)").rows[0][0] == 'both positive'
        assert db.execute("CALL nested_check(1, -1)").rows[0][0] == 'a positive, b not'
        assert db.execute("CALL nested_check(-1, 1)").rows[0][0] == 'a not positive'

    def test_leave_loop(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION find_first_gt(threshold INT) RETURNS INT
            BEGIN
                DECLARE i INT DEFAULT 0;
                myloop: WHILE i < 100 DO
                    SET i = i + 1;
                    IF i > threshold THEN
                        LEAVE myloop;
                    END IF;
                END WHILE;
                RETURN i;
            END
        """)
        assert db.execute("CALL find_first_gt(5)").rows[0][0] == 6

    def test_iterate_loop(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION sum_odd(n INT) RETURNS INT
            BEGIN
                DECLARE result INT DEFAULT 0;
                DECLARE i INT DEFAULT 0;
                myloop: WHILE i < n DO
                    SET i = i + 1;
                    IF i % 2 = 0 THEN
                        ITERATE myloop;
                    END IF;
                    SET result = result + i;
                END WHILE;
                RETURN result;
            END
        """)
        # Sum of odd numbers from 1 to 10: 1+3+5+7+9 = 25
        assert db.execute("CALL sum_odd(10)").rows[0][0] == 25


# =============================================================================
# 5. Variable & Assignment Tests
# =============================================================================

class TestVariables:

    def test_declare_with_default(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE x INT DEFAULT 42;
                RETURN x;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 42

    def test_declare_without_default(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE x INT;
                RETURN x;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 0  # default for int

    def test_set_variable(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE x INT DEFAULT 0;
                SET x = 10;
                RETURN x;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 10

    def test_multiple_variables(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE a INT DEFAULT 1;
                DECLARE b INT DEFAULT 2;
                DECLARE c INT DEFAULT 0;
                SET c = a + b;
                RETURN c;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 3

    def test_variable_reassignment(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE x INT DEFAULT 1;
                SET x = x + 1;
                SET x = x * 3;
                RETURN x;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 6

    def test_string_variable(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS TEXT
            BEGIN
                DECLARE msg TEXT DEFAULT 'hello';
                RETURN msg;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 'hello'

    def test_float_default(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS FLOAT
            BEGIN
                DECLARE x FLOAT;
                RETURN x;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 0.0


# =============================================================================
# 6. Procedure Tests (with DML)
# =============================================================================

class TestProcedures:

    def test_simple_procedure(self):
        db = make_db()
        db.execute("""
            CREATE PROCEDURE give_raise(IN emp_id INT, IN amount FLOAT)
            BEGIN
                UPDATE employees SET salary = salary + amount WHERE id = emp_id;
            END
        """)
        db.execute("CALL give_raise(1, 10000)")
        result = db.execute("SELECT salary FROM employees WHERE id = 1")
        assert result.rows[0][0] == 60000.0

    def test_procedure_out_param(self):
        db = make_db()
        db.execute("""
            CREATE PROCEDURE get_employee_salary(IN emp_id INT, OUT sal FLOAT)
            BEGIN
                SELECT salary INTO sal FROM employees WHERE id = emp_id;
            END
        """)
        result = db.execute("CALL get_employee_salary(2, 0)")
        assert result.rows[0][0] == 60000.0

    def test_procedure_inout_param(self):
        db = ProcDB()
        db.execute("""
            CREATE PROCEDURE double_it(INOUT val INT)
            BEGIN
                SET val = val * 2;
            END
        """)
        result = db.execute("CALL double_it(21)")
        assert result.rows[0][0] == 42

    def test_procedure_insert(self):
        db = make_db()
        db.execute("""
            CREATE PROCEDURE add_employee(IN emp_id INT, IN emp_name TEXT,
                                         IN emp_salary FLOAT, IN emp_dept TEXT)
            BEGIN
                INSERT INTO employees VALUES (emp_id, emp_name, emp_salary, emp_dept);
            END
        """)
        db.execute("CALL add_employee(6, 'Frank', 65000, 'marketing')")
        result = db.execute("SELECT name FROM employees WHERE id = 6")
        assert result.rows[0][0] == 'Frank'

    def test_procedure_delete(self):
        db = make_db()
        db.execute("""
            CREATE PROCEDURE remove_employee(IN emp_id INT)
            BEGIN
                DELETE FROM employees WHERE id = emp_id;
            END
        """)
        db.execute("CALL remove_employee(3)")
        result = db.execute("SELECT COUNT(*) AS cnt FROM employees")
        assert result.rows[0][0] == 4

    def test_procedure_multiple_statements(self):
        db = make_db()
        db.execute("""
            CREATE PROCEDURE transfer_dept(IN emp_id INT, IN new_dept TEXT)
            BEGIN
                DECLARE old_salary FLOAT DEFAULT 0;
                SELECT salary INTO old_salary FROM employees WHERE id = emp_id;
                UPDATE employees SET dept = new_dept WHERE id = emp_id;
                UPDATE employees SET salary = old_salary + 5000 WHERE id = emp_id;
            END
        """)
        db.execute("CALL transfer_dept(3, 'engineering')")
        result = db.execute("SELECT dept, salary FROM employees WHERE id = 3")
        assert result.rows[0][0] == 'engineering'
        assert result.rows[0][1] == 50000.0

    def test_procedure_with_loop(self):
        db = ProcDB()
        db.execute("CREATE TABLE numbers (val INT)")
        db.execute("""
            CREATE PROCEDURE fill_numbers(IN n INT)
            BEGIN
                DECLARE i INT DEFAULT 1;
                WHILE i <= n DO
                    INSERT INTO numbers VALUES (i);
                    SET i = i + 1;
                END WHILE;
            END
        """)
        db.execute("CALL fill_numbers(5)")
        result = db.execute("SELECT COUNT(*) AS cnt FROM numbers")
        assert result.rows[0][0] == 5

    def test_procedure_with_conditional_insert(self):
        db = make_db()
        db.execute("""
            CREATE PROCEDURE conditional_raise(IN threshold FLOAT, IN bonus FLOAT)
            BEGIN
                UPDATE employees SET salary = salary + bonus WHERE salary < threshold;
            END
        """)
        db.execute("CALL conditional_raise(55000, 5000)")
        # Alice (50000) and Charlie (45000) should get raises
        result = db.execute("SELECT salary FROM employees WHERE id = 1")
        assert result.rows[0][0] == 55000.0
        result = db.execute("SELECT salary FROM employees WHERE id = 3")
        assert result.rows[0][0] == 50000.0
        # Bob (60000) should not
        result = db.execute("SELECT salary FROM employees WHERE id = 2")
        assert result.rows[0][0] == 60000.0

    def test_procedure_no_params(self):
        db = ProcDB()
        db.execute("CREATE TABLE log (msg TEXT)")
        db.execute("""
            CREATE PROCEDURE log_hello()
            BEGIN
                INSERT INTO log VALUES ('hello');
            END
        """)
        db.execute("CALL log_hello()")
        result = db.execute("SELECT msg FROM log")
        assert result.rows[0][0] == 'hello'


# =============================================================================
# 7. Recursion Tests
# =============================================================================

class TestRecursion:

    def test_recursive_factorial(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION rfact(n INT) RETURNS INT
            BEGIN
                IF n <= 1 THEN
                    RETURN 1;
                END IF;
                RETURN n * rfact(n - 1);
            END
        """)
        assert db.execute("CALL rfact(5)").rows[0][0] == 120
        assert db.execute("CALL rfact(0)").rows[0][0] == 1

    def test_recursive_fibonacci(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION fib(n INT) RETURNS INT
            BEGIN
                IF n <= 0 THEN RETURN 0; END IF;
                IF n = 1 THEN RETURN 1; END IF;
                RETURN fib(n - 1) + fib(n - 2);
            END
        """)
        assert db.execute("CALL fib(0)").rows[0][0] == 0
        assert db.execute("CALL fib(1)").rows[0][0] == 1
        assert db.execute("CALL fib(6)").rows[0][0] == 8
        assert db.execute("CALL fib(10)").rows[0][0] == 55

    def test_recursion_depth_limit(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION infinite(n INT) RETURNS INT
            BEGIN
                RETURN infinite(n + 1);
            END
        """)
        with pytest.raises(DatabaseError, match="recursion depth"):
            db.execute("CALL infinite(0)")

    def test_recursive_gcd(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION gcd(a INT, b INT) RETURNS INT
            BEGIN
                IF b = 0 THEN
                    RETURN a;
                END IF;
                RETURN gcd(b, a % b);
            END
        """)
        assert db.execute("CALL gcd(12, 8)").rows[0][0] == 4
        assert db.execute("CALL gcd(100, 75)").rows[0][0] == 25
        assert db.execute("CALL gcd(7, 13)").rows[0][0] == 1

    def test_recursive_power(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION rpower(base INT, exp INT) RETURNS INT
            BEGIN
                IF exp = 0 THEN RETURN 1; END IF;
                RETURN base * rpower(base, exp - 1);
            END
        """)
        assert db.execute("CALL rpower(2, 10)").rows[0][0] == 1024
        assert db.execute("CALL rpower(3, 4)").rows[0][0] == 81


# =============================================================================
# 8. Function Calls in Queries
# =============================================================================

class TestUDFsInQueries:

    def test_udf_in_select_constant(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION double(x INT) RETURNS INT
            BEGIN
                RETURN x * 2;
            END
        """)
        result = db.execute("SELECT double(5) AS val")
        assert result.rows[0][0] == 10

    def test_udf_in_select_with_arithmetic(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION square(x INT) RETURNS INT
            BEGIN
                RETURN x * x;
            END
        """)
        result = db.execute("SELECT square(3) + square(4) AS val")
        assert result.rows[0][0] == 25

    def test_udf_nested_calls(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION inc(x INT) RETURNS INT
            BEGIN RETURN x + 1; END
        """)
        db.execute("""
            CREATE FUNCTION double(x INT) RETURNS INT
            BEGIN RETURN x * 2; END
        """)
        result = db.execute("SELECT double(inc(5)) AS val")
        assert result.rows[0][0] == 12

    def test_udf_in_where_constant(self):
        db = make_db()
        db.execute("""
            CREATE FUNCTION threshold() RETURNS FLOAT
            BEGIN
                RETURN 55000;
            END
        """)
        result = db.execute("SELECT name FROM employees WHERE salary > threshold()")
        names = [r[0] for r in result.rows]
        assert 'Bob' in names
        assert 'Diana' in names
        assert 'Alice' not in names


# =============================================================================
# 9. Exception Handling Tests
# =============================================================================

class TestExceptionHandling:

    def test_handler_continue(self):
        db = ProcDB()
        db.execute("CREATE TABLE log (msg TEXT)")
        db.execute("""
            CREATE PROCEDURE test_handler()
            BEGIN
                DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
                BEGIN
                    INSERT INTO log VALUES ('error caught');
                END;
                INSERT INTO nonexistent VALUES (1);
                INSERT INTO log VALUES ('after error');
            END
        """)
        db.execute("CALL test_handler()")
        result = db.execute("SELECT msg FROM log")
        msgs = [r[0] for r in result.rows]
        assert 'error caught' in msgs
        assert 'after error' in msgs

    def test_handler_exit(self):
        db = ProcDB()
        db.execute("CREATE TABLE log (msg TEXT)")
        db.execute("""
            CREATE PROCEDURE test_exit_handler()
            BEGIN
                DECLARE EXIT HANDLER FOR SQLEXCEPTION
                BEGIN
                    INSERT INTO log VALUES ('error caught');
                END;
                INSERT INTO nonexistent VALUES (1);
                INSERT INTO log VALUES ('should not reach');
            END
        """)
        db.execute("CALL test_exit_handler()")
        result = db.execute("SELECT msg FROM log")
        msgs = [r[0] for r in result.rows]
        assert 'error caught' in msgs
        assert 'should not reach' not in msgs

    def test_no_handler_propagates(self):
        db = ProcDB()
        db.execute("""
            CREATE PROCEDURE test_no_handler()
            BEGIN
                INSERT INTO nonexistent VALUES (1);
            END
        """)
        with pytest.raises(Exception):
            db.execute("CALL test_no_handler()")


# =============================================================================
# 10. DROP Tests
# =============================================================================

class TestDropRoutines:

    def test_drop_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN RETURN 1; END
        """)
        db.execute("DROP FUNCTION f")
        with pytest.raises(DatabaseError):
            db.execute("CALL f()")

    def test_drop_function_if_exists(self):
        db = ProcDB()
        db.execute("DROP FUNCTION IF EXISTS nonexistent")  # no error

    def test_drop_procedure(self):
        db = ProcDB()
        db.execute("""
            CREATE PROCEDURE p()
            BEGIN
                DECLARE x INT DEFAULT 1;
            END
        """)
        db.execute("DROP PROCEDURE p")
        with pytest.raises(DatabaseError):
            db.execute("CALL p()")

    def test_drop_procedure_if_exists(self):
        db = ProcDB()
        db.execute("DROP PROCEDURE IF EXISTS nonexistent")

    def test_drop_nonexistent_function_error(self):
        db = ProcDB()
        with pytest.raises(DatabaseError, match="not found"):
            db.execute("DROP FUNCTION nonexistent")

    def test_drop_nonexistent_procedure_error(self):
        db = ProcDB()
        with pytest.raises(DatabaseError, match="not found"):
            db.execute("DROP PROCEDURE nonexistent")


# =============================================================================
# 11. CREATE OR REPLACE Tests
# =============================================================================

class TestCreateOrReplace:

    def test_replace_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN RETURN 1; END
        """)
        assert db.execute("CALL f()").rows[0][0] == 1

        db.execute("""
            CREATE OR REPLACE FUNCTION f() RETURNS INT
            BEGIN RETURN 2; END
        """)
        assert db.execute("CALL f()").rows[0][0] == 2

    def test_replace_procedure(self):
        db = ProcDB()
        db.execute("CREATE TABLE log (msg TEXT)")
        db.execute("""
            CREATE PROCEDURE p()
            BEGIN
                INSERT INTO log VALUES ('v1');
            END
        """)
        db.execute("CALL p()")

        db.execute("""
            CREATE OR REPLACE PROCEDURE p()
            BEGIN
                INSERT INTO log VALUES ('v2');
            END
        """)
        db.execute("CALL p()")

        result = db.execute("SELECT msg FROM log")
        msgs = [r[0] for r in result.rows]
        assert 'v1' in msgs
        assert 'v2' in msgs


# =============================================================================
# 12. SHOW Tests
# =============================================================================

class TestShowRoutines:

    def test_show_functions_empty(self):
        db = ProcDB()
        result = db.execute("SHOW FUNCTIONS")
        assert len(result.rows) == 0

    def test_show_functions(self):
        db = ProcDB()
        db.execute("CREATE FUNCTION b() RETURNS INT BEGIN RETURN 1; END")
        db.execute("CREATE FUNCTION a() RETURNS INT BEGIN RETURN 2; END")
        result = db.execute("SHOW FUNCTIONS")
        assert result.rows[0][0] == 'a'
        assert result.rows[1][0] == 'b'

    def test_show_procedures(self):
        db = ProcDB()
        db.execute("CREATE PROCEDURE q() BEGIN DECLARE x INT; END")
        db.execute("CREATE PROCEDURE p() BEGIN DECLARE x INT; END")
        result = db.execute("SHOW PROCEDURES")
        assert result.rows[0][0] == 'p'
        assert result.rows[1][0] == 'q'


# =============================================================================
# 13. SELECT INTO Tests
# =============================================================================

class TestSelectInto:

    def test_select_into_single_var(self):
        db = make_db()
        db.execute("""
            CREATE PROCEDURE get_salary(IN emp_id INT, OUT sal FLOAT)
            BEGIN
                SELECT salary INTO sal FROM employees WHERE id = emp_id;
            END
        """)
        result = db.execute("CALL get_salary(1, 0)")
        assert result.rows[0][0] == 50000.0

    def test_select_into_used_in_computation(self):
        db = make_db()
        db.execute("""
            CREATE FUNCTION salary_bonus(emp_id INT) RETURNS FLOAT
            BEGIN
                DECLARE sal FLOAT DEFAULT 0;
                SELECT salary INTO sal FROM employees WHERE id = emp_id;
                RETURN sal * 0.1;
            END
        """)
        result = db.execute("CALL salary_bonus(2)")
        assert abs(result.rows[0][0] - 6000.0) < 0.01

    def test_select_into_count(self):
        db = make_db()
        db.execute("""
            CREATE FUNCTION count_dept(dept_name TEXT) RETURNS INT
            BEGIN
                DECLARE cnt INT DEFAULT 0;
                SELECT COUNT(*) AS cnt_val INTO cnt FROM employees WHERE dept = dept_name;
                RETURN cnt;
            END
        """)
        result = db.execute("CALL count_dept('engineering')")
        assert result.rows[0][0] == 3

    def test_select_into_no_rows(self):
        db = make_db()
        db.execute("""
            CREATE FUNCTION get_name(emp_id INT) RETURNS TEXT
            BEGIN
                DECLARE result TEXT DEFAULT 'not found';
                SELECT name INTO result FROM employees WHERE id = emp_id;
                RETURN result;
            END
        """)
        result = db.execute("CALL get_name(999)")
        assert result.rows[0][0] == 'not found'


# =============================================================================
# 14. Built-in Function Tests
# =============================================================================

class TestBuiltinFunctions:

    def test_abs(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(x INT) RETURNS INT
            BEGIN RETURN abs(x); END
        """)
        assert db.execute("CALL f(-5)").rows[0][0] == 5

    def test_upper_lower(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(s TEXT) RETURNS TEXT
            BEGIN RETURN upper(s); END
        """)
        assert db.execute("CALL f('hello')").rows[0][0] == 'HELLO'

    def test_length(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(s TEXT) RETURNS INT
            BEGIN RETURN length(s); END
        """)
        assert db.execute("CALL f('hello')").rows[0][0] == 5

    def test_concat(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(a TEXT, b TEXT) RETURNS TEXT
            BEGIN RETURN concat(a, ' ', b); END
        """)
        assert db.execute("CALL f('hello', 'world')").rows[0][0] == 'hello world'

    def test_coalesce(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(x INT) RETURNS INT
            BEGIN RETURN coalesce(x, 0); END
        """)
        assert db.execute("CALL f(NULL)").rows[0][0] == 0
        assert db.execute("CALL f(5)").rows[0][0] == 5

    def test_substr(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(s TEXT, start INT, len INT) RETURNS TEXT
            BEGIN RETURN substr(s, start, len); END
        """)
        assert db.execute("CALL f('hello world', 1, 5)").rows[0][0] == 'hello'

    def test_replace(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(s TEXT) RETURNS TEXT
            BEGIN RETURN replace(s, 'world', 'there'); END
        """)
        assert db.execute("CALL f('hello world')").rows[0][0] == 'hello there'

    def test_trim(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(s TEXT) RETURNS TEXT
            BEGIN RETURN trim(s); END
        """)
        assert db.execute("CALL f('  hello  ')").rows[0][0] == 'hello'

    def test_round(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(x FLOAT) RETURNS INT
            BEGIN RETURN round(x); END
        """)
        assert db.execute("CALL f(3.7)").rows[0][0] == 4

    def test_greatest_least(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION fmax(a INT, b INT, c INT) RETURNS INT
            BEGIN RETURN greatest(a, b, c); END
        """)
        assert db.execute("CALL fmax(3, 7, 5)").rows[0][0] == 7

    def test_power(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(b INT, e INT) RETURNS INT
            BEGIN RETURN power(b, e); END
        """)
        assert db.execute("CALL f(2, 10)").rows[0][0] == 1024

    def test_mod(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(a INT, b INT) RETURNS INT
            BEGIN RETURN mod(a, b); END
        """)
        assert db.execute("CALL f(10, 3)").rows[0][0] == 1


# =============================================================================
# 15. Complex Procedure Tests
# =============================================================================

class TestComplexProcedures:

    def test_fibonacci_table(self):
        db = ProcDB()
        db.execute("CREATE TABLE fib_table (n INT, val INT)")
        db.execute("""
            CREATE PROCEDURE gen_fib(IN count INT)
            BEGIN
                DECLARE a INT DEFAULT 0;
                DECLARE b INT DEFAULT 1;
                DECLARE temp INT DEFAULT 0;
                DECLARE i INT DEFAULT 0;
                WHILE i < count DO
                    INSERT INTO fib_table VALUES (i, a);
                    SET temp = b;
                    SET b = a + b;
                    SET a = temp;
                    SET i = i + 1;
                END WHILE;
            END
        """)
        db.execute("CALL gen_fib(8)")
        result = db.execute("SELECT val FROM fib_table ORDER BY n ASC")
        vals = [r[0] for r in result.rows]
        assert vals == [0, 1, 1, 2, 3, 5, 8, 13]

    def test_procedure_calling_function(self):
        db = make_db()
        db.execute("""
            CREATE FUNCTION calc_bonus(salary FLOAT) RETURNS FLOAT
            BEGIN
                IF salary > 60000 THEN
                    RETURN salary * 0.15;
                ELSEIF salary > 50000 THEN
                    RETURN salary * 0.10;
                ELSE
                    RETURN salary * 0.05;
                END IF;
            END
        """)
        db.execute("""
            CREATE PROCEDURE apply_bonuses()
            BEGIN
                DECLARE emp_count INT DEFAULT 0;
                DECLARE i INT DEFAULT 1;
                SELECT COUNT(*) AS cnt INTO emp_count FROM employees;
                WHILE i <= emp_count DO
                    DECLARE sal FLOAT DEFAULT 0;
                    SELECT salary INTO sal FROM employees WHERE id = i;
                    DECLARE bonus FLOAT DEFAULT 0;
                    SET bonus = calc_bonus(sal);
                    UPDATE employees SET salary = salary + bonus WHERE id = i;
                    SET i = i + 1;
                END WHILE;
            END
        """)
        db.execute("CALL apply_bonuses()")
        # Alice: 50000 * 0.05 = 2500 -> 52500
        result = db.execute("SELECT salary FROM employees WHERE id = 1")
        assert abs(result.rows[0][0] - 52500.0) < 0.01

    def test_function_calling_function(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION add(a INT, b INT) RETURNS INT
            BEGIN RETURN a + b; END
        """)
        db.execute("""
            CREATE FUNCTION add3(a INT, b INT, c INT) RETURNS INT
            BEGIN RETURN add(add(a, b), c); END
        """)
        assert db.execute("CALL add3(1, 2, 3)").rows[0][0] == 6

    def test_procedure_calling_procedure(self):
        db = ProcDB()
        db.execute("CREATE TABLE log (msg TEXT)")
        db.execute("""
            CREATE PROCEDURE inner_proc(IN msg TEXT)
            BEGIN
                INSERT INTO log VALUES (msg);
            END
        """)
        db.execute("""
            CREATE PROCEDURE outer_proc()
            BEGIN
                CALL inner_proc('first');
                CALL inner_proc('second');
            END
        """)
        db.execute("CALL outer_proc()")
        result = db.execute("SELECT msg FROM log")
        msgs = [r[0] for r in result.rows]
        assert 'first' in msgs
        assert 'second' in msgs

    def test_prime_checker(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION is_prime(n INT) RETURNS BOOL
            BEGIN
                IF n <= 1 THEN RETURN false; END IF;
                IF n <= 3 THEN RETURN true; END IF;
                IF n % 2 = 0 THEN RETURN false; END IF;
                DECLARE i INT DEFAULT 3;
                WHILE i * i <= n DO
                    IF n % i = 0 THEN RETURN false; END IF;
                    SET i = i + 2;
                END WHILE;
                RETURN true;
            END
        """)
        assert db.execute("CALL is_prime(2)").rows[0][0] is True
        assert db.execute("CALL is_prime(7)").rows[0][0] is True
        assert db.execute("CALL is_prime(4)").rows[0][0] is False
        assert db.execute("CALL is_prime(1)").rows[0][0] is False
        assert db.execute("CALL is_prime(97)").rows[0][0] is True

    def test_collatz_steps(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION collatz(n INT) RETURNS INT
            BEGIN
                DECLARE steps INT DEFAULT 0;
                WHILE n != 1 DO
                    IF n % 2 = 0 THEN
                        SET n = n / 2;
                    ELSE
                        SET n = 3 * n + 1;
                    END IF;
                    SET steps = steps + 1;
                END WHILE;
                RETURN steps;
            END
        """)
        assert db.execute("CALL collatz(1)").rows[0][0] == 0
        assert db.execute("CALL collatz(2)").rows[0][0] == 1
        assert db.execute("CALL collatz(6)").rows[0][0] == 8
        assert db.execute("CALL collatz(27)").rows[0][0] == 111


# =============================================================================
# 16. Type Coercion Tests
# =============================================================================

class TestTypeCoercion:

    def test_int_to_float(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(x INT) RETURNS FLOAT
            BEGIN RETURN x; END
        """)
        result = db.execute("CALL f(42)")
        assert isinstance(result.rows[0][0], float)
        assert result.rows[0][0] == 42.0

    def test_float_to_int(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(x FLOAT) RETURNS INT
            BEGIN RETURN x; END
        """)
        result = db.execute("CALL f(3.7)")
        assert isinstance(result.rows[0][0], int)
        assert result.rows[0][0] == 3

    def test_int_to_text(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(x INT) RETURNS TEXT
            BEGIN RETURN x; END
        """)
        result = db.execute("CALL f(42)")
        assert result.rows[0][0] == '42'

    def test_null_coercion(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN RETURN NULL; END
        """)
        result = db.execute("CALL f()")
        assert result.rows[0][0] is None


# =============================================================================
# 17. Edge Cases
# =============================================================================

class TestEdgeCases:

    def test_empty_function_body(self):
        """Function with no RETURN returns default."""
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE x INT DEFAULT 42;
            END
        """)
        result = db.execute("CALL f()")
        assert result.rows[0][0] == 0  # default for INT

    def test_function_with_many_params(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION sum5(a INT, b INT, c INT, d INT, e INT) RETURNS INT
            BEGIN RETURN a + b + c + d + e; END
        """)
        assert db.execute("CALL sum5(1, 2, 3, 4, 5)").rows[0][0] == 15

    def test_procedure_no_params_no_body(self):
        db = ProcDB()
        db.execute("""
            CREATE PROCEDURE noop()
            BEGIN
                DECLARE x INT;
            END
        """)
        result = db.execute("CALL noop()")
        assert result.message == 'CALL noop'

    def test_call_nonexistent(self):
        db = ProcDB()
        with pytest.raises(DatabaseError, match="not found"):
            db.execute("CALL nonexistent()")

    def test_while_zero_iterations(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN
                DECLARE x INT DEFAULT 0;
                WHILE x > 100 DO
                    SET x = x + 1;
                END WHILE;
                RETURN x;
            END
        """)
        assert db.execute("CALL f()").rows[0][0] == 0

    def test_division_by_zero(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f() RETURNS INT
            BEGIN RETURN 1 / 0; END
        """)
        with pytest.raises(DatabaseError, match="Division by zero"):
            db.execute("CALL f()")

    def test_modulo_operator(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION f(a INT, b INT) RETURNS INT
            BEGIN RETURN a % b; END
        """)
        assert db.execute("CALL f(17, 5)").rows[0][0] == 2

    def test_standard_sql_still_works(self):
        """Ensure standard SQL operations work through ProcDB."""
        db = ProcDB()
        db.execute("CREATE TABLE test (id INT PRIMARY KEY, val TEXT)")
        db.execute("INSERT INTO test VALUES (1, 'hello')")
        result = db.execute("SELECT val FROM test WHERE id = 1")
        assert result.rows[0][0] == 'hello'
        db.execute("UPDATE test SET val = 'world' WHERE id = 1")
        result = db.execute("SELECT val FROM test WHERE id = 1")
        assert result.rows[0][0] == 'world'
        db.execute("DELETE FROM test WHERE id = 1")
        result = db.execute("SELECT COUNT(*) AS cnt FROM test")
        assert result.rows[0][0] == 0


# =============================================================================
# 18. Multiple OUT Parameter Tests
# =============================================================================

class TestMultipleOutParams:

    def test_two_out_params(self):
        db = ProcDB()
        db.execute("""
            CREATE PROCEDURE min_max(IN a INT, IN b INT, OUT mn INT, OUT mx INT)
            BEGIN
                IF a < b THEN
                    SET mn = a;
                    SET mx = b;
                ELSE
                    SET mn = b;
                    SET mx = a;
                END IF;
            END
        """)
        result = db.execute("CALL min_max(7, 3, 0, 0)")
        assert result.rows[0][0] == 3   # mn
        assert result.rows[0][1] == 7   # mx

    def test_swap_inout(self):
        db = ProcDB()
        db.execute("""
            CREATE PROCEDURE swap(INOUT a INT, INOUT b INT)
            BEGIN
                DECLARE temp INT DEFAULT 0;
                SET temp = a;
                SET a = b;
                SET b = temp;
            END
        """)
        result = db.execute("CALL swap(1, 2)")
        assert result.rows[0][0] == 2   # a
        assert result.rows[0][1] == 1   # b


# =============================================================================
# 19. Mathematical Algorithm Tests
# =============================================================================

class TestMathAlgorithms:

    def test_integer_sqrt(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION isqrt(n INT) RETURNS INT
            BEGIN
                IF n < 0 THEN RETURN -1; END IF;
                IF n = 0 THEN RETURN 0; END IF;
                DECLARE x INT DEFAULT 0;
                SET x = n;
                DECLARE y INT DEFAULT 0;
                SET y = (x + 1) / 2;
                WHILE y < x DO
                    SET x = y;
                    SET y = (x + n / x) / 2;
                END WHILE;
                RETURN x;
            END
        """)
        assert db.execute("CALL isqrt(0)").rows[0][0] == 0
        assert db.execute("CALL isqrt(4)").rows[0][0] == 2
        assert db.execute("CALL isqrt(16)").rows[0][0] == 4
        assert db.execute("CALL isqrt(100)").rows[0][0] == 10

    def test_sum_of_digits(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION digit_sum(n INT) RETURNS INT
            BEGIN
                DECLARE total INT DEFAULT 0;
                IF n < 0 THEN SET n = n * -1; END IF;
                WHILE n > 0 DO
                    SET total = total + n % 10;
                    SET n = n / 10;
                END WHILE;
                RETURN total;
            END
        """)
        assert db.execute("CALL digit_sum(123)").rows[0][0] == 6
        assert db.execute("CALL digit_sum(9999)").rows[0][0] == 36
        assert db.execute("CALL digit_sum(0)").rows[0][0] == 0

    def test_reverse_number(self):
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION reverse_num(n INT) RETURNS INT
            BEGIN
                DECLARE result INT DEFAULT 0;
                WHILE n > 0 DO
                    SET result = result * 10 + n % 10;
                    SET n = n / 10;
                END WHILE;
                RETURN result;
            END
        """)
        assert db.execute("CALL reverse_num(123)").rows[0][0] == 321
        assert db.execute("CALL reverse_num(100)").rows[0][0] == 1


# =============================================================================
# 20. Integration Tests (Procedures + Full SQL)
# =============================================================================

class TestIntegration:

    def test_create_table_from_procedure(self):
        """Procedures should work alongside table operations."""
        db = ProcDB()
        db.execute("CREATE TABLE accounts (id INT, balance FLOAT)")
        db.execute("INSERT INTO accounts VALUES (1, 1000)")
        db.execute("INSERT INTO accounts VALUES (2, 500)")

        db.execute("""
            CREATE PROCEDURE transfer(IN from_id INT, IN to_id INT, IN amount FLOAT)
            BEGIN
                UPDATE accounts SET balance = balance - amount WHERE id = from_id;
                UPDATE accounts SET balance = balance + amount WHERE id = to_id;
            END
        """)

        db.execute("CALL transfer(1, 2, 200)")

        r1 = db.execute("SELECT balance FROM accounts WHERE id = 1")
        r2 = db.execute("SELECT balance FROM accounts WHERE id = 2")
        assert r1.rows[0][0] == 800.0
        assert r2.rows[0][0] == 700.0

    def test_batch_operations(self):
        db = ProcDB()
        db.execute("CREATE TABLE items (id INT, name TEXT, price FLOAT)")

        db.execute("""
            CREATE PROCEDURE add_item(IN item_id INT, IN item_name TEXT, IN item_price FLOAT)
            BEGIN
                INSERT INTO items VALUES (item_id, item_name, item_price);
            END
        """)

        db.execute("CALL add_item(1, 'apple', 1.50)")
        db.execute("CALL add_item(2, 'banana', 0.75)")
        db.execute("CALL add_item(3, 'cherry', 2.00)")

        result = db.execute("SELECT COUNT(*) AS cnt FROM items")
        assert result.rows[0][0] == 3

        result = db.execute("SELECT name FROM items WHERE price > 1.0 ORDER BY name ASC")
        names = [r[0] for r in result.rows]
        assert names == ['apple', 'cherry']

    def test_function_in_insert_values(self):
        """Use CALL to get a value, then use it."""
        db = ProcDB()
        db.execute("""
            CREATE FUNCTION next_id() RETURNS INT
            BEGIN
                RETURN 42;
            END
        """)
        result = db.execute("CALL next_id()")
        assert result.rows[0][0] == 42

    def test_show_tables_still_works(self):
        db = ProcDB()
        db.execute("CREATE TABLE t1 (id INT)")
        db.execute("CREATE TABLE t2 (id INT)")
        result = db.execute("SHOW TABLES")
        names = [r[0] for r in result.rows]
        assert 't1' in names
        assert 't2' in names

    def test_describe_still_works(self):
        db = ProcDB()
        db.execute("CREATE TABLE test (id INT PRIMARY KEY, name TEXT NOT NULL)")
        result = db.execute("DESCRIBE test")
        assert len(result.rows) == 2

    def test_drop_table_still_works(self):
        db = ProcDB()
        db.execute("CREATE TABLE temp (id INT)")
        db.execute("DROP TABLE temp")
        with pytest.raises(Exception):
            db.execute("SELECT * FROM temp")

    def test_join_still_works(self):
        db = ProcDB()
        db.execute("CREATE TABLE a (id INT, val TEXT)")
        db.execute("CREATE TABLE b (id INT, ref_id INT, info TEXT)")
        db.execute("INSERT INTO a VALUES (1, 'x')")
        db.execute("INSERT INTO b VALUES (1, 1, 'y')")
        result = db.execute("SELECT a.val, b.info FROM a JOIN b ON a.id = b.ref_id")
        assert result.rows[0][0] == 'x'
        assert result.rows[0][1] == 'y'


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
