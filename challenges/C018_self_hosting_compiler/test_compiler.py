"""
Tests for C018: Self-Hosting Compiler
Tests the extended VM, host compiler, self-compiler, and bootstrap verification.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from compiler import (
    Op, Builtin, Chunk, TokenType, Token, FnObject,
    lex, Parser, Compiler, VM, CompileError, ParseError, VMError, LexError,
    compile_source, execute, disassemble,
    compile_subset_with_host, run_self_compiler, bootstrap_verify,
    SELF_COMPILER_SOURCE,
)


# ============================================================
# Extended VM: Array Operations
# ============================================================

class TestArrayOperations:
    def test_empty_array_literal(self):
        r = execute("let a = []; print(len(a));")
        assert r['output'] == ['0']

    def test_array_literal_with_elements(self):
        r = execute("let a = [10, 20, 30]; print(len(a));")
        assert r['output'] == ['3']

    def test_array_index_get(self):
        r = execute("let a = [10, 20, 30]; print(a[0]); print(a[1]); print(a[2]);")
        assert r['output'] == ['10', '20', '30']

    def test_array_index_set(self):
        r = execute("let a = [10, 20, 30]; a[1] = 99; print(a[1]);")
        assert r['output'] == ['99']

    def test_array_push(self):
        r = execute("let a = []; push(a, 1); push(a, 2); print(len(a)); print(a[0]); print(a[1]);")
        assert r['output'] == ['2', '1', '2']

    def test_array_nested(self):
        r = execute("let a = [[1, 2], [3, 4]]; print(a[0][0]); print(a[1][1]);")
        assert r['output'] == ['1', '4']

    def test_array_in_loop(self):
        r = execute("""
            let a = [];
            let i = 0;
            while (i < 5) {
                push(a, i * i);
                i = i + 1;
            }
            print(a[0]); print(a[2]); print(a[4]);
        """)
        assert r['output'] == ['0', '4', '16']

    def test_array_passed_to_function(self):
        r = execute("""
            fn sum(arr, n) {
                let total = 0;
                let i = 0;
                while (i < n) {
                    total = total + arr[i];
                    i = i + 1;
                }
                return total;
            }
            let a = [10, 20, 30];
            print(sum(a, 3));
        """)
        assert r['output'] == ['60']

    def test_array_index_out_of_bounds(self):
        with pytest.raises(VMError, match="out of bounds"):
            execute("let a = [1, 2]; print(a[5]);")

    def test_index_non_array(self):
        with pytest.raises(VMError, match="non-array"):
            execute("let x = 42; print(x[0]);")


# ============================================================
# Extended VM: String Builtins
# ============================================================

class TestStringBuiltins:
    def test_len_string(self):
        r = execute('let s = "hello"; print(len(s));')
        assert r['output'] == ['5']

    def test_len_empty_string(self):
        r = execute('let s = ""; print(len(s));')
        assert r['output'] == ['0']

    def test_char_at(self):
        r = execute('let s = "abc"; print(char_at(s, 0)); print(char_at(s, 2));')
        assert r['output'] == ['a', 'c']

    def test_char_code(self):
        r = execute('print(char_code("A")); print(char_code("0"));')
        assert r['output'] == ['65', '48']

    def test_from_code(self):
        r = execute('print(from_code(65)); print(from_code(48));')
        assert r['output'] == ['A', '0']

    def test_substr(self):
        r = execute('let s = "hello world"; print(substr(s, 0, 5)); print(substr(s, 6, 5));')
        assert r['output'] == ['hello', 'world']

    def test_to_str(self):
        r = execute('print(to_str(42)); print(to_str(0));')
        assert r['output'] == ['42', '0']

    def test_to_int(self):
        r = execute('let n = to_int("42"); print(n + 8);')
        assert r['output'] == ['50']

    def test_type_of(self):
        r = execute("""
            print(type_of(42));
            print(type_of("hi"));
            print(type_of(true));
            print(type_of([]));
        """)
        assert r['output'] == ['int', 'string', 'bool', 'array']

    def test_string_concat(self):
        r = execute('let a = "hello"; let b = " world"; print(a + b);')
        assert r['output'] == ['hello world']

    def test_char_at_out_of_bounds(self):
        with pytest.raises(VMError, match="out of bounds"):
            execute('print(char_at("hi", 5));')

    def test_escape_sequences(self):
        r = execute(r'let s = "a\nb"; print(len(s));')
        assert r['output'] == ['3']


# ============================================================
# Extended VM: Comprehensive
# ============================================================

class TestExtendedVM:
    def test_build_string_from_codes(self):
        r = execute("""
            let result = "";
            let i = 0;
            while (i < 3) {
                result = result + from_code(65 + i);
                i = i + 1;
            }
            print(result);
        """)
        assert r['output'] == ['ABC']

    def test_string_to_char_array(self):
        r = execute("""
            let s = "abc";
            let arr = [];
            let i = 0;
            while (i < len(s)) {
                push(arr, char_at(s, i));
                i = i + 1;
            }
            print(len(arr));
            print(arr[0]);
            print(arr[2]);
        """)
        assert r['output'] == ['3', 'a', 'c']

    def test_array_of_arrays_as_struct(self):
        r = execute("""
            fn make_point(x, y) {
                return [x, y];
            }
            fn point_x(p) { return p[0]; }
            fn point_y(p) { return p[1]; }
            let p = make_point(3, 4);
            print(point_x(p));
            print(point_y(p));
        """)
        assert r['output'] == ['3', '4']


# ============================================================
# Host Compiler: Existing Features Still Work
# ============================================================

class TestHostCompilerBasics:
    def test_arithmetic(self):
        r = execute("print(2 + 3 * 4);")
        assert r['output'] == ['14']

    def test_variables(self):
        r = execute("let x = 10; let y = 20; print(x + y);")
        assert r['output'] == ['30']

    def test_if_else(self):
        r = execute("let x = 5; if (x > 3) { print(1); } else { print(0); }")
        assert r['output'] == ['1']

    def test_while_loop(self):
        r = execute("""
            let i = 0;
            let sum = 0;
            while (i < 10) {
                sum = sum + i;
                i = i + 1;
            }
            print(sum);
        """)
        assert r['output'] == ['45']

    def test_functions(self):
        r = execute("""
            fn add(a, b) { return a + b; }
            print(add(3, 4));
        """)
        assert r['output'] == ['7']

    def test_recursion(self):
        r = execute("""
            fn fib(n) {
                if (n <= 1) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            print(fib(10));
        """)
        assert r['output'] == ['55']

    def test_nested_if(self):
        r = execute("""
            let x = 15;
            if (x > 20) {
                print(3);
            } else if (x > 10) {
                print(2);
            } else {
                print(1);
            }
        """)
        assert r['output'] == ['2']

    def test_boolean_logic(self):
        r = execute("print(true and false); print(true or false);")
        assert r['output'] == ['false', 'true']

    def test_string_equality(self):
        r = execute('let a = "hi"; let b = "hi"; print(a == b);')
        assert r['output'] == ['true']

    def test_modulo(self):
        r = execute("print(17 % 5);")
        assert r['output'] == ['2']

    def test_negation(self):
        r = execute("print(-42);")
        assert r['output'] == ['-42']

    def test_comparison_operators(self):
        r = execute("print(3 <= 3); print(3 >= 4); print(3 != 4);")
        assert r['output'] == ['true', 'false', 'true']


# ============================================================
# Lexer Edge Cases
# ============================================================

class TestLexer:
    def test_brackets(self):
        tokens = lex("[]")
        assert tokens[0].type == TokenType.LBRACKET
        assert tokens[1].type == TokenType.RBRACKET

    def test_comment_skipping(self):
        tokens = lex("42 // comment\n43")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.INT, TokenType.INT]

    def test_all_operators(self):
        tokens = lex("+ - * / % == != < > <= >= = ( ) { } [ ] , ;")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.EQ, TokenType.NE, TokenType.LT,
            TokenType.GT, TokenType.LE, TokenType.GE, TokenType.ASSIGN,
            TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE, TokenType.RBRACE,
            TokenType.LBRACKET, TokenType.RBRACKET, TokenType.COMMA, TokenType.SEMICOLON,
        ]
        assert types == expected

    def test_keywords(self):
        tokens = lex("let if else while fn return print true false and or not")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        expected = [
            TokenType.LET, TokenType.IF, TokenType.ELSE, TokenType.WHILE,
            TokenType.FN, TokenType.RETURN, TokenType.PRINT,
            TokenType.TRUE, TokenType.FALSE,
            TokenType.AND, TokenType.OR, TokenType.NOT,
        ]
        assert types == expected

    def test_string_escape(self):
        tokens = lex(r'"hello\nworld"')
        assert tokens[0].value == "hello\nworld"

    def test_unterminated_string(self):
        with pytest.raises(LexError, match="Unterminated"):
            lex('"hello')

    def test_unexpected_char(self):
        with pytest.raises(LexError, match="Unexpected"):
            lex("@")


# ============================================================
# Parser Edge Cases
# ============================================================

class TestParser:
    def test_array_literal_empty(self):
        from compiler import ArrayLit, ExprStmt
        tokens = lex("[];")
        p = Parser(tokens)
        result = p.parse()
        stmt = result.stmts[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, ArrayLit)
        assert stmt.expr.elements == []

    def test_array_literal_elements(self):
        from compiler import ArrayLit, IntLit, ExprStmt
        tokens = lex("[1, 2, 3];")
        p = Parser(tokens)
        result = p.parse()
        stmt = result.stmts[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, ArrayLit)
        assert len(stmt.expr.elements) == 3

    def test_index_expr(self):
        from compiler import IndexExpr, Var, ExprStmt
        tokens = lex("a[0];")
        p = Parser(tokens)
        result = p.parse()
        stmt = result.stmts[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, IndexExpr)
        assert isinstance(stmt.expr.obj, Var)

    def test_index_assign(self):
        from compiler import IndexAssign
        tokens = lex("a[0] = 5;")
        p = Parser(tokens)
        result = p.parse()
        stmt = result.stmts[0]
        assert isinstance(stmt, IndexAssign)

    def test_chained_index(self):
        from compiler import IndexExpr, ExprStmt
        tokens = lex("a[0][1];")
        p = Parser(tokens)
        result = p.parse()
        stmt = result.stmts[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, IndexExpr)
        assert isinstance(stmt.expr.obj, IndexExpr)


# ============================================================
# Disassembly
# ============================================================

class TestDisassembly:
    def test_disassemble_basic(self):
        chunk, _ = compile_source("print(42);")
        text = disassemble(chunk)
        assert "CONST" in text
        assert "PRINT" in text
        assert "HALT" in text

    def test_disassemble_builtin(self):
        chunk, _ = compile_source("print(len([]));")
        text = disassemble(chunk)
        assert "BUILTIN" in text
        assert "LEN" in text

    def test_disassemble_function(self):
        chunk, _ = compile_source("fn f() { return 1; } print(f());")
        text = disassemble(chunk)
        assert "fn:f" in text


# ============================================================
# Self-Compiler: Can it compile the simplest programs?
# ============================================================

class TestSelfCompilerSimple:
    """Test the self-compiler on simple programs."""

    def test_self_compiler_parses(self):
        """The self-compiler source itself compiles without error."""
        chunk, compiler = compile_source(SELF_COMPILER_SOURCE)
        assert chunk is not None

    def test_self_compiler_runs(self):
        """The self-compiler runs on a trivial input."""
        code, consts, names = run_self_compiler("print(42);")
        assert len(code) > 0
        assert Op.HALT in code

    def test_compile_single_print(self):
        """Self-compiler produces correct bytecode for print(42)."""
        src = "print(42);"
        self_code, self_consts, self_names = run_self_compiler(src)
        host_code, host_consts, host_names = compile_subset_with_host(src)
        assert self_code == host_code
        assert self_consts == host_consts
        assert self_names == host_names

    def test_compile_arithmetic(self):
        src = "print(2 + 3);"
        result = bootstrap_verify(src)
        assert result['match'], f"Mismatch:\nhost={result['host']}\nself={result['self']}"

    def test_compile_complex_arithmetic(self):
        src = "print(2 + 3 * 4);"
        result = bootstrap_verify(src)
        assert result['match'], f"Mismatch:\nhost={result['host']}\nself={result['self']}"

    def test_compile_subtraction(self):
        src = "print(10 - 3);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_division(self):
        src = "print(10 / 2);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_modulo(self):
        src = "print(17 % 5);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_negation(self):
        src = "print(-42);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_comparison_eq(self):
        src = "print(1 == 1);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_comparison_ne(self):
        src = "print(1 != 2);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_comparison_lt(self):
        src = "print(1 < 2);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_comparison_gt(self):
        src = "print(2 > 1);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_comparison_le(self):
        src = "print(1 <= 1);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_comparison_ge(self):
        src = "print(2 >= 1);"
        result = bootstrap_verify(src)
        assert result['match']


# ============================================================
# Self-Compiler: Variables
# ============================================================

class TestSelfCompilerVariables:
    def test_compile_let(self):
        src = "let x = 42; print(x);"
        result = bootstrap_verify(src)
        assert result['match'], f"Mismatch:\nhost={result['host']}\nself={result['self']}"

    def test_compile_multiple_vars(self):
        src = "let x = 10; let y = 20; print(x + y);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_assignment(self):
        src = "let x = 1; x = 2; print(x);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_var_arithmetic(self):
        src = "let a = 3; let b = 4; print(a * b + 1);"
        result = bootstrap_verify(src)
        assert result['match']


# ============================================================
# Self-Compiler: Control Flow
# ============================================================

class TestSelfCompilerControlFlow:
    def test_compile_if_simple(self):
        src = "let x = 5; if (x > 3) { print(1); }"
        result = bootstrap_verify(src)
        assert result['match'], f"Mismatch:\nhost={result['host']}\nself={result['self']}"

    def test_compile_if_else(self):
        src = "let x = 2; if (x > 3) { print(1); } else { print(0); }"
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_while(self):
        src = "let i = 0; while (i < 5) { print(i); i = i + 1; }"
        result = bootstrap_verify(src)
        assert result['match'], f"Mismatch:\nhost={result['host']}\nself={result['self']}"

    def test_compile_nested_if(self):
        src = """
            let x = 15;
            if (x > 20) {
                print(3);
            } else if (x > 10) {
                print(2);
            } else {
                print(1);
            }
        """
        result = bootstrap_verify(src)
        assert result['match']

    def test_compile_nested_while(self):
        src = """
            let i = 0;
            while (i < 3) {
                let j = 0;
                while (j < 2) {
                    print(i + j);
                    j = j + 1;
                }
                i = i + 1;
            }
        """
        result = bootstrap_verify(src)
        assert result['match']


# ============================================================
# Bootstrap: Full Programs
# ============================================================

class TestBootstrap:
    def test_fibonacci_iterative(self):
        src = """
            let a = 0;
            let b = 1;
            let i = 0;
            while (i < 10) {
                print(a);
                let temp = a + b;
                a = b;
                b = temp;
                i = i + 1;
            }
        """
        result = bootstrap_verify(src)
        assert result['match'], f"Mismatch:\nhost={result['host']}\nself={result['self']}"

    def test_factorial_iterative(self):
        src = """
            let n = 10;
            let result = 1;
            while (n > 0) {
                result = result * n;
                n = n - 1;
            }
            print(result);
        """
        result = bootstrap_verify(src)
        assert result['match']

    def test_sum_of_squares(self):
        src = """
            let i = 1;
            let sum = 0;
            while (i <= 10) {
                sum = sum + i * i;
                i = i + 1;
            }
            print(sum);
        """
        result = bootstrap_verify(src)
        assert result['match']

    def test_collatz_steps(self):
        src = """
            let n = 27;
            let steps = 0;
            while (n != 1) {
                if (n % 2 == 0) {
                    n = n / 2;
                } else {
                    n = n * 3 + 1;
                }
                steps = steps + 1;
            }
            print(steps);
        """
        result = bootstrap_verify(src)
        assert result['match']

    def test_gcd(self):
        src = """
            let a = 48;
            let b = 18;
            while (b != 0) {
                let temp = b;
                b = a % b;
                a = temp;
            }
            print(a);
        """
        result = bootstrap_verify(src)
        assert result['match']

    def test_prime_sieve_small(self):
        src = """
            let n = 2;
            while (n < 20) {
                let is_prime = 1;
                let d = 2;
                while (d * d <= n) {
                    if (n % d == 0) {
                        is_prime = 0;
                    }
                    d = d + 1;
                }
                if (is_prime == 1) {
                    print(n);
                }
                n = n + 1;
            }
        """
        result = bootstrap_verify(src)
        assert result['match']

    def test_empty_program(self):
        src = ""
        # Both should just produce HALT
        result = bootstrap_verify(src)
        assert result['match']

    def test_just_comments(self):
        src = "// nothing here\n"
        result = bootstrap_verify(src)
        assert result['match']

    def test_multiple_prints(self):
        src = "print(1); print(2); print(3); print(4); print(5);"
        result = bootstrap_verify(src)
        assert result['match']


# ============================================================
# Bootstrap: Verify execution matches
# ============================================================

class TestBootstrapExecution:
    """Not just bytecode equality -- verify the bytecode actually runs correctly."""

    def _run_self_compiled(self, src):
        """Compile src with self-compiler, then run the produced bytecode."""
        bc, consts, names = run_self_compiler(src)
        chunk = Chunk()
        chunk.code = bc
        chunk.constants = consts
        chunk.names = names
        vm = VM(chunk)
        vm.run()
        return vm.output

    def test_exec_print_42(self):
        output = self._run_self_compiled("print(42);")
        assert output == ['42']

    def test_exec_arithmetic(self):
        output = self._run_self_compiled("print(2 + 3 * 4);")
        assert output == ['14']

    def test_exec_fibonacci(self):
        src = """
            let a = 0;
            let b = 1;
            let i = 0;
            while (i < 8) {
                print(a);
                let temp = a + b;
                a = b;
                b = temp;
                i = i + 1;
            }
        """
        output = self._run_self_compiled(src)
        assert output == ['0', '1', '1', '2', '3', '5', '8', '13']

    def test_exec_factorial(self):
        src = """
            let n = 6;
            let result = 1;
            while (n > 0) {
                result = result * n;
                n = n - 1;
            }
            print(result);
        """
        output = self._run_self_compiled(src)
        assert output == ['720']

    def test_exec_if_else(self):
        src = """
            let x = 5;
            if (x > 10) {
                print(1);
            } else {
                print(0);
            }
        """
        output = self._run_self_compiled(src)
        assert output == ['0']

    def test_exec_nested_loops(self):
        src = """
            let i = 1;
            while (i <= 3) {
                let j = 1;
                while (j <= 3) {
                    if (i == j) {
                        print(i);
                    }
                    j = j + 1;
                }
                i = i + 1;
            }
        """
        output = self._run_self_compiled(src)
        assert output == ['1', '2', '3']

    def test_exec_gcd(self):
        src = """
            let a = 48;
            let b = 18;
            while (b != 0) {
                let temp = b;
                b = a % b;
                a = temp;
            }
            print(a);
        """
        output = self._run_self_compiled(src)
        assert output == ['6']

    def test_exec_collatz(self):
        src = """
            let n = 6;
            let steps = 0;
            while (n != 1) {
                if (n % 2 == 0) {
                    n = n / 2;
                } else {
                    n = n * 3 + 1;
                }
                steps = steps + 1;
            }
            print(steps);
        """
        output = self._run_self_compiled(src)
        assert output == ['8']


# ============================================================
# Meta-test: Self-compiler compiles a mini-compiler
# ============================================================

class TestMetaCompilation:
    """The ultimate test: can the self-compiler compile a compiler-like program?
    This tests the self-compiler on a program that LOOKS like a compiler
    (building bytecode arrays, doing string matching, etc.) but only uses
    the integer subset."""

    def test_bytecode_builder(self):
        """A program that builds bytecode-like output -- tests variables,
        loops, conditionals, and arithmetic together in a compiler-like pattern."""
        src = """
            // Simulate building bytecode for: print(1 + 2)
            // CONST 0 (value 1), CONST 1 (value 2), ADD, PRINT, HALT
            let code_0 = 1;   // OP_CONST
            let code_1 = 0;   // constant index 0
            let code_2 = 1;   // OP_CONST
            let code_3 = 1;   // constant index 1
            let code_4 = 4;   // OP_ADD
            let code_5 = 27;  // OP_PRINT
            let code_6 = 28;  // OP_HALT

            // Verify: should have 7 instructions
            print(7);

            // Simulate patching a jump target
            let jump_target = 0;
            let instr_count = 0;
            while (instr_count < 5) {
                instr_count = instr_count + 1;
            }
            jump_target = instr_count;
            print(jump_target);
        """
        result = bootstrap_verify(src)
        assert result['match']

        # Also verify execution
        output = execute(src)['output']
        assert output == ['7', '5']

    def test_constant_pool_simulation(self):
        """Simulate a constant pool with uniqueness check."""
        src = """
            // Simulate constant pool: add values, skip duplicates
            // Using individual variables since subset has no arrays
            let pool_0 = 0;
            let pool_1 = 0;
            let pool_2 = 0;
            let pool_count = 0;

            // Add constant 42
            pool_0 = 42;
            pool_count = 1;

            // Add constant 99
            pool_1 = 99;
            pool_count = 2;

            // Add constant 42 again -- should not increase count
            let found = 0;
            if (pool_0 == 42) { found = 1; }
            if (found == 0) {
                if (pool_1 == 42) { found = 1; }
            }
            if (found == 0) {
                pool_2 = 42;
                pool_count = pool_count + 1;
            }

            print(pool_count); // should be 2, not 3
        """
        result = bootstrap_verify(src)
        assert result['match']

        output = execute(src)['output']
        assert output == ['2']


# ============================================================
# Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    def test_deeply_nested_arithmetic(self):
        src = "print(((((1 + 2) * 3) - 4) / 5) % 3);"
        result = bootstrap_verify(src)
        assert result['match']
        output = execute(src)['output']
        assert output == ['1']

    def test_many_variables(self):
        src = """
            let a = 1; let b = 2; let c = 3; let d = 4; let e = 5;
            let f = 6; let g = 7; let h = 8; let i = 9; let j = 10;
            print(a + b + c + d + e + f + g + h + i + j);
        """
        result = bootstrap_verify(src)
        assert result['match']
        output = execute(src)['output']
        assert output == ['55']

    def test_zero_iterations(self):
        src = """
            let i = 10;
            while (i < 5) {
                print(i);
                i = i + 1;
            }
            print(0);
        """
        result = bootstrap_verify(src)
        assert result['match']
        output = execute(src)['output']
        assert output == ['0']

    def test_negative_numbers(self):
        src = "print(-1 + -2);"
        result = bootstrap_verify(src)
        assert result['match']
        output = execute(src)['output']
        assert output == ['-3']

    def test_large_numbers(self):
        src = "print(999999 * 999999);"
        result = bootstrap_verify(src)
        assert result['match']

    def test_chained_assignment(self):
        src = "let x = 0; let y = 0; x = 5; y = x; print(y);"
        result = bootstrap_verify(src)
        assert result['match']
        output = execute(src)['output']
        assert output == ['5']

    def test_if_no_else(self):
        src = """
            let x = 1;
            if (x == 1) {
                print(1);
            }
            print(2);
        """
        result = bootstrap_verify(src)
        assert result['match']

    def test_parenthesized_expr(self):
        src = "print((1 + 2) * (3 + 4));"
        result = bootstrap_verify(src)
        assert result['match']
        output = execute(src)['output']
        assert output == ['21']


# ============================================================
# Counting
# ============================================================

def test_total_test_count():
    """Meta-test: verify we have enough tests."""
    import inspect
    count = 0
    for name, obj in globals().items():
        if inspect.isclass(obj) and name.startswith('Test'):
            for method_name in dir(obj):
                if method_name.startswith('test_'):
                    count += 1
    # Also count standalone test functions
    for name, obj in globals().items():
        if name.startswith('test_') and inspect.isfunction(obj):
            count += 1
    assert count >= 90, f"Only {count} tests, need at least 90"
