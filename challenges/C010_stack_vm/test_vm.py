"""Tests for C010 Stack VM."""

import pytest
from stack_vm import (
    lex, Parser, Compiler, VM, execute, compile_source, disassemble,
    LexError, ParseError, CompileError, VMError, Op, FnObject
)


# ============================================================
# Lexer Tests
# ============================================================

class TestLexer:
    def test_integers(self):
        tokens = lex("42 0 100")
        assert tokens[0].value == 42
        assert tokens[1].value == 0
        assert tokens[2].value == 100

    def test_floats(self):
        tokens = lex("3.14 0.5")
        assert tokens[0].value == 3.14
        assert tokens[1].value == 0.5

    def test_strings(self):
        tokens = lex('"hello" "world"')
        assert tokens[0].value == "hello"
        assert tokens[1].value == "world"

    def test_unterminated_string(self):
        with pytest.raises(LexError):
            lex('"oops')

    def test_keywords(self):
        tokens = lex("let if else while fn return print true false and or not")
        from stack_vm import TokenType
        expected = [TokenType.LET, TokenType.IF, TokenType.ELSE, TokenType.WHILE,
                    TokenType.FN, TokenType.RETURN, TokenType.PRINT,
                    TokenType.TRUE, TokenType.FALSE, TokenType.AND, TokenType.OR,
                    TokenType.NOT, TokenType.EOF]
        assert [t.type for t in tokens] == expected

    def test_operators(self):
        tokens = lex("+ - * / % == != < > <= >= =")
        from stack_vm import TokenType
        expected = [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
                    TokenType.PERCENT, TokenType.EQ, TokenType.NE, TokenType.LT,
                    TokenType.GT, TokenType.LE, TokenType.GE, TokenType.ASSIGN,
                    TokenType.EOF]
        assert [t.type for t in tokens] == expected

    def test_delimiters(self):
        tokens = lex("( ) { } , ;")
        from stack_vm import TokenType
        expected = [TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE,
                    TokenType.RBRACE, TokenType.COMMA, TokenType.SEMICOLON,
                    TokenType.EOF]
        assert [t.type for t in tokens] == expected

    def test_comments(self):
        tokens = lex("42 // this is a comment\n43")
        assert tokens[0].value == 42
        assert tokens[1].value == 43

    def test_line_tracking(self):
        tokens = lex("a\nb\nc")
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3

    def test_unexpected_char(self):
        with pytest.raises(LexError):
            lex("@")


# ============================================================
# Parser Tests
# ============================================================

class TestParser:
    def parse(self, src):
        return Parser(lex(src)).parse()

    def test_int_literal(self):
        from stack_vm import IntLit
        prog = self.parse("42;")
        assert isinstance(prog.stmts[0], IntLit)

    def test_binary_op(self):
        from stack_vm import BinOp
        prog = self.parse("1 + 2;")
        assert isinstance(prog.stmts[0], BinOp)
        assert prog.stmts[0].op == '+'

    def test_precedence(self):
        from stack_vm import BinOp
        prog = self.parse("1 + 2 * 3;")
        # Should be 1 + (2 * 3)
        expr = prog.stmts[0]
        assert expr.op == '+'
        assert expr.right.op == '*'

    def test_let_declaration(self):
        from stack_vm import LetDecl
        prog = self.parse("let x = 5;")
        assert isinstance(prog.stmts[0], LetDecl)
        assert prog.stmts[0].name == 'x'

    def test_if_else(self):
        from stack_vm import IfStmt
        prog = self.parse("if (true) { 1; } else { 2; }")
        assert isinstance(prog.stmts[0], IfStmt)
        assert prog.stmts[0].else_body is not None

    def test_while_loop(self):
        from stack_vm import WhileStmt
        prog = self.parse("while (true) { 1; }")
        assert isinstance(prog.stmts[0], WhileStmt)

    def test_function_decl(self):
        from stack_vm import FnDecl
        prog = self.parse("fn add(a, b) { return a + b; }")
        assert isinstance(prog.stmts[0], FnDecl)
        assert prog.stmts[0].params == ['a', 'b']

    def test_function_call(self):
        from stack_vm import CallExpr
        prog = self.parse("foo(1, 2);")
        assert isinstance(prog.stmts[0], CallExpr)
        assert prog.stmts[0].callee == 'foo'
        assert len(prog.stmts[0].args) == 2

    def test_nested_parens(self):
        from stack_vm import BinOp
        prog = self.parse("(1 + 2) * 3;")
        expr = prog.stmts[0]
        assert expr.op == '*'
        assert expr.left.op == '+'

    def test_unary_minus(self):
        from stack_vm import UnaryOp
        prog = self.parse("-5;")
        assert isinstance(prog.stmts[0], UnaryOp)

    def test_assignment(self):
        from stack_vm import Assign
        prog = self.parse("let x = 0; x = 5;")
        assert isinstance(prog.stmts[1], Assign)

    def test_parse_error(self):
        with pytest.raises(ParseError):
            self.parse("let = ;")


# ============================================================
# Compiler Tests
# ============================================================

class TestCompiler:
    def test_compile_int(self):
        chunk, _ = compile_source("42;")
        assert Op.CONST in chunk.code

    def test_compile_add(self):
        chunk, _ = compile_source("1 + 2;")
        assert Op.ADD in chunk.code

    def test_compile_variable(self):
        chunk, _ = compile_source("let x = 5;")
        assert Op.STORE in chunk.code

    def test_compile_function(self):
        chunk, compiler = compile_source("fn foo() { return 1; }")
        assert 'foo' in compiler.functions

    def test_disassemble(self):
        chunk, _ = compile_source("let x = 5;")
        text = disassemble(chunk)
        assert 'CONST' in text
        assert 'STORE' in text


# ============================================================
# VM Execution Tests -- Arithmetic
# ============================================================

class TestArithmetic:
    def test_add(self):
        r = execute("let x = 3 + 4; print(x);")
        assert r['output'] == ['7']

    def test_subtract(self):
        r = execute("let x = 10 - 3; print(x);")
        assert r['output'] == ['7']

    def test_multiply(self):
        r = execute("let x = 6 * 7; print(x);")
        assert r['output'] == ['42']

    def test_divide_int(self):
        r = execute("let x = 10 / 3; print(x);")
        assert r['output'] == ['3']  # integer division

    def test_divide_float(self):
        r = execute("let x = 10.0 / 3.0; print(x);")
        val = float(r['output'][0])
        assert abs(val - 3.333333) < 0.001

    def test_modulo(self):
        r = execute("let x = 10 % 3; print(x);")
        assert r['output'] == ['1']

    def test_negate(self):
        r = execute("let x = -5; print(x);")
        assert r['output'] == ['-5']

    def test_complex_expr(self):
        r = execute("let x = (2 + 3) * 4 - 1; print(x);")
        assert r['output'] == ['19']

    def test_division_by_zero(self):
        with pytest.raises(VMError, match="Division by zero"):
            execute("let x = 1 / 0;")

    def test_modulo_by_zero(self):
        with pytest.raises(VMError, match="Modulo by zero"):
            execute("let x = 1 % 0;")


# ============================================================
# VM Execution Tests -- Comparison & Logic
# ============================================================

class TestComparison:
    def test_equal(self):
        r = execute("print(5 == 5);")
        assert r['output'] == ['true']

    def test_not_equal(self):
        r = execute("print(5 != 3);")
        assert r['output'] == ['true']

    def test_less_than(self):
        r = execute("print(3 < 5);")
        assert r['output'] == ['true']

    def test_greater_than(self):
        r = execute("print(5 > 3);")
        assert r['output'] == ['true']

    def test_less_equal(self):
        r = execute("print(5 <= 5);")
        assert r['output'] == ['true']

    def test_greater_equal(self):
        r = execute("print(5 >= 6);")
        assert r['output'] == ['false']

    def test_and_true(self):
        r = execute("print(true and true);")
        assert r['output'] == ['true']

    def test_and_false(self):
        r = execute("print(true and false);")
        assert r['output'] == ['false']

    def test_or_true(self):
        r = execute("print(false or true);")
        assert r['output'] == ['true']

    def test_or_false(self):
        r = execute("print(false or false);")
        assert r['output'] == ['false']

    def test_not(self):
        r = execute("print(not true);")
        assert r['output'] == ['false']

    def test_short_circuit_and(self):
        # If short-circuit works, x stays 0
        r = execute("let x = 0; let y = false and 5; print(y);")
        assert r['output'] == ['false']

    def test_short_circuit_or(self):
        r = execute("let y = true or 5; print(y);")
        assert r['output'] == ['true']


# ============================================================
# VM Execution Tests -- Variables
# ============================================================

class TestVariables:
    def test_let_and_use(self):
        r = execute("let x = 42; print(x);")
        assert r['output'] == ['42']

    def test_reassign(self):
        r = execute("let x = 1; x = 2; print(x);")
        assert r['output'] == ['2']

    def test_multiple_vars(self):
        r = execute("let a = 1; let b = 2; print(a + b);")
        assert r['output'] == ['3']

    def test_var_in_expression(self):
        r = execute("let x = 10; let y = x * 2 + 1; print(y);")
        assert r['output'] == ['21']

    def test_undefined_var(self):
        with pytest.raises(VMError, match="Undefined variable"):
            execute("print(x);")


# ============================================================
# VM Execution Tests -- Control Flow
# ============================================================

class TestControlFlow:
    def test_if_true(self):
        r = execute("if (true) { print(1); }")
        assert r['output'] == ['1']

    def test_if_false(self):
        r = execute("if (false) { print(1); }")
        assert r['output'] == []

    def test_if_else(self):
        r = execute("if (false) { print(1); } else { print(2); }")
        assert r['output'] == ['2']

    def test_if_else_chain(self):
        r = execute("""
            let x = 2;
            if (x == 1) { print(1); }
            else if (x == 2) { print(2); }
            else { print(3); }
        """)
        assert r['output'] == ['2']

    def test_while_loop(self):
        r = execute("""
            let i = 0;
            let sum = 0;
            while (i < 5) {
                sum = sum + i;
                i = i + 1;
            }
            print(sum);
        """)
        assert r['output'] == ['10']

    def test_while_no_execute(self):
        r = execute("while (false) { print(1); }")
        assert r['output'] == []

    def test_nested_loops(self):
        r = execute("""
            let count = 0;
            let i = 0;
            while (i < 3) {
                let j = 0;
                while (j < 3) {
                    count = count + 1;
                    j = j + 1;
                }
                i = i + 1;
            }
            print(count);
        """)
        assert r['output'] == ['9']

    def test_if_in_while(self):
        r = execute("""
            let i = 0;
            let evens = 0;
            while (i < 10) {
                if (i % 2 == 0) {
                    evens = evens + 1;
                }
                i = i + 1;
            }
            print(evens);
        """)
        assert r['output'] == ['5']


# ============================================================
# VM Execution Tests -- Functions
# ============================================================

class TestFunctions:
    def test_simple_function(self):
        r = execute("""
            fn add(a, b) {
                return a + b;
            }
            print(add(3, 4));
        """)
        assert r['output'] == ['7']

    def test_function_no_args(self):
        r = execute("""
            fn greet() {
                return 42;
            }
            print(greet());
        """)
        assert r['output'] == ['42']

    def test_function_no_return(self):
        r = execute("""
            fn side_effect() {
                print(99);
            }
            side_effect();
        """)
        assert r['output'] == ['99']

    def test_recursive_factorial(self):
        r = execute("""
            fn fact(n) {
                if (n <= 1) {
                    return 1;
                }
                return n * fact(n - 1);
            }
            print(fact(5));
        """)
        assert r['output'] == ['120']

    def test_recursive_fibonacci(self):
        r = execute("""
            fn fib(n) {
                if (n <= 1) {
                    return n;
                }
                return fib(n - 1) + fib(n - 2);
            }
            print(fib(10));
        """)
        assert r['output'] == ['55']

    def test_function_multiple_calls(self):
        r = execute("""
            fn double(x) {
                return x * 2;
            }
            print(double(3));
            print(double(5));
            print(double(10));
        """)
        assert r['output'] == ['6', '10', '20']

    def test_function_wrong_arity(self):
        with pytest.raises(VMError, match="expects"):
            execute("""
                fn add(a, b) { return a + b; }
                add(1);
            """)

    def test_call_non_function(self):
        with pytest.raises(VMError, match="Cannot call"):
            execute("""
                let x = 5;
                x(1);
            """)

    def test_function_with_locals(self):
        r = execute("""
            fn compute(x) {
                let doubled = x * 2;
                let tripled = x * 3;
                return doubled + tripled;
            }
            print(compute(4));
        """)
        assert r['output'] == ['20']

    def test_closure_like_behavior(self):
        """Functions can access globals defined before the call."""
        r = execute("""
            let multiplier = 10;
            fn scale(x) {
                return x * multiplier;
            }
            print(scale(5));
        """)
        assert r['output'] == ['50']


# ============================================================
# VM Execution Tests -- Strings
# ============================================================

class TestStrings:
    def test_string_literal(self):
        r = execute('print("hello");')
        assert r['output'] == ['hello']

    def test_string_concat(self):
        r = execute('print("hello" + " " + "world");')
        assert r['output'] == ['hello world']

    def test_string_equality(self):
        r = execute('print("abc" == "abc");')
        assert r['output'] == ['true']

    def test_string_inequality(self):
        r = execute('print("abc" != "def");')
        assert r['output'] == ['true']


# ============================================================
# VM Execution Tests -- Type Interactions
# ============================================================

class TestTypes:
    def test_bool_print(self):
        r = execute("print(true); print(false);")
        assert r['output'] == ['true', 'false']

    def test_none_print(self):
        r = execute("""
            fn nothing() { }
            print(nothing());
        """)
        assert r['output'] == ['None']

    def test_string_repeat(self):
        r = execute('print("ab" * 3);')
        assert r['output'] == ['ababab']

    def test_float_arithmetic(self):
        r = execute("print(1.5 + 2.5);")
        assert r['output'] == ['4.0']


# ============================================================
# VM Execution Tests -- Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        r = execute("")
        assert r['output'] == []

    def test_many_variables(self):
        src = ""
        for i in range(50):
            src += f"let v{i} = {i};\n"
        src += "print(v0 + v49);\n"
        r = execute(src)
        assert r['output'] == ['49']

    def test_deeply_nested_if(self):
        r = execute("""
            let x = 1;
            if (x == 1) {
                if (x == 1) {
                    if (x == 1) {
                        print(42);
                    }
                }
            }
        """)
        assert r['output'] == ['42']

    def test_execution_limit(self):
        with pytest.raises(VMError, match="Execution limit"):
            execute("while (true) { let x = 1; }")

    def test_multiple_prints(self):
        r = execute("print(1); print(2); print(3);")
        assert r['output'] == ['1', '2', '3']

    def test_complex_program(self):
        """Compute sum of squares from 1 to 10."""
        r = execute("""
            fn square(x) {
                return x * x;
            }
            let sum = 0;
            let i = 1;
            while (i <= 10) {
                sum = sum + square(i);
                i = i + 1;
            }
            print(sum);
        """)
        assert r['output'] == ['385']

    def test_gcd(self):
        r = execute("""
            fn gcd(a, b) {
                while (b != 0) {
                    let temp = b;
                    b = a % b;
                    a = temp;
                }
                return a;
            }
            print(gcd(48, 18));
        """)
        assert r['output'] == ['6']

    def test_power_function(self):
        r = execute("""
            fn power(base, exp) {
                let result = 1;
                let i = 0;
                while (i < exp) {
                    result = result * base;
                    i = i + 1;
                }
                return result;
            }
            print(power(2, 10));
        """)
        assert r['output'] == ['1024']

    def test_is_prime(self):
        r = execute("""
            fn is_prime(n) {
                if (n < 2) { return false; }
                let i = 2;
                while (i * i <= n) {
                    if (n % i == 0) { return false; }
                    i = i + 1;
                }
                return true;
            }
            // Print primes up to 20
            let n = 2;
            while (n <= 20) {
                if (is_prime(n)) {
                    print(n);
                }
                n = n + 1;
            }
        """)
        assert r['output'] == ['2', '3', '5', '7', '11', '13', '17', '19']

    def test_env_after_execution(self):
        r = execute("let x = 42; let y = x + 8;")
        assert r['env']['x'] == 42
        assert r['env']['y'] == 50

    def test_steps_counted(self):
        r = execute("let x = 1;")
        assert r['steps'] > 0


# ============================================================
# Disassembler Tests
# ============================================================

class TestDisassemble:
    def test_basic(self):
        chunk, _ = compile_source("let x = 5;")
        text = disassemble(chunk)
        assert 'CONST' in text
        assert 'STORE' in text
        assert 'HALT' in text

    def test_function_reference(self):
        chunk, _ = compile_source("fn foo() { return 1; }")
        text = disassemble(chunk)
        assert 'fn:foo' in text
