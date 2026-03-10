"""
Tests for C033: Effect Handlers Runtime
Composes C032 (effect system) + C010 (stack VM) with algebraic effects at runtime.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from effect_runtime import (
    lex, parse, compile_program, run, run_checked,
    EffectVM, EffectError, ResumeError, CompileError,
    Parser, Compiler, Op, Chunk,
    EffectDecl, PerformExpr, HandleWith, HandlerClause, ResumeExpr,
    IntLit, FloatLit, StringLit, BoolLit, Var, Assign, UnaryOp, BinOp,
    LetDecl, FnDecl, Block, IfStmt, WhileStmt, ReturnStmt, PrintStmt,
    CallExpr, Program, ExtTokenType, TokenType,
)


# =========================================================================
# Section 1: Lexer tests
# =========================================================================
class TestLexer:
    def test_basic_tokens(self):
        tokens = lex("let x = 42;")
        types = [t.type for t in tokens]
        assert TokenType.LET in types
        assert TokenType.IDENT in types
        assert TokenType.ASSIGN in types
        assert TokenType.INT in types
        assert TokenType.SEMICOLON in types

    def test_effect_keywords(self):
        tokens = lex("effect perform handle with resume")
        types = [t.type for t in tokens]
        assert ExtTokenType.EFFECT in types
        assert ExtTokenType.PERFORM in types
        assert ExtTokenType.HANDLE in types
        assert ExtTokenType.WITH in types
        assert ExtTokenType.RESUME in types

    def test_dot_token(self):
        tokens = lex("IO.print")
        assert any(t.type == ExtTokenType.DOT for t in tokens)

    def test_arrow_token(self):
        tokens = lex("-> {}")
        assert any(t.type == ExtTokenType.ARROW for t in tokens)

    def test_all_operators(self):
        tokens = lex("+ - * / % == != < > <= >=")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        expected = [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
                   TokenType.PERCENT, TokenType.EQ, TokenType.NE, TokenType.LT,
                   TokenType.GT, TokenType.LE, TokenType.GE]
        assert types == expected

    def test_string_literal(self):
        tokens = lex('"hello world"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"

    def test_float_literal(self):
        tokens = lex("3.14")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == 3.14

    def test_line_tracking(self):
        tokens = lex("x\ny\nz")
        lines = [t.line for t in tokens if t.type == TokenType.IDENT]
        assert lines == [1, 2, 3]

    def test_comment_skip(self):
        tokens = lex("x // this is a comment\ny")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        assert len(idents) == 2

    def test_boolean_literals(self):
        tokens = lex("true false")
        assert tokens[0].type == TokenType.TRUE
        assert tokens[1].type == TokenType.FALSE

    def test_unexpected_char(self):
        with pytest.raises(SyntaxError, match="Unexpected character"):
            lex("@")


# =========================================================================
# Section 2: Parser tests
# =========================================================================
class TestParser:
    def test_let_decl(self):
        ast = parse("let x = 5;")
        assert isinstance(ast.stmts[0], LetDecl)
        assert ast.stmts[0].name == "x"

    def test_fn_decl(self):
        ast = parse("fn add(a, b) { return a + b; }")
        fn = ast.stmts[0]
        assert isinstance(fn, FnDecl)
        assert fn.name == "add"
        assert fn.params == ["a", "b"]

    def test_if_else(self):
        ast = parse("if (x) { let a = 1; } else { let b = 2; }")
        stmt = ast.stmts[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_body is not None

    def test_while_loop(self):
        ast = parse("while (x > 0) { x = x - 1; }")
        stmt = ast.stmts[0]
        assert isinstance(stmt, WhileStmt)

    def test_effect_decl(self):
        ast = parse("effect IO { print(msg); read(); }")
        decl = ast.stmts[0]
        assert isinstance(decl, EffectDecl)
        assert decl.name == "IO"
        assert len(decl.operations) == 2
        assert decl.operations[0] == ("print", ["msg"])
        assert decl.operations[1] == ("read", [])

    def test_perform_expr(self):
        ast = parse("perform IO.print(42);")
        stmt = ast.stmts[0]
        assert isinstance(stmt, PerformExpr)
        assert stmt.effect == "IO"
        assert stmt.operation == "print"
        assert len(stmt.args) == 1

    def test_perform_no_args(self):
        ast = parse("perform IO.read();")
        stmt = ast.stmts[0]
        assert isinstance(stmt, PerformExpr)
        assert stmt.args == []

    def test_perform_multiple_args(self):
        ast = parse("perform State.set(key, val);")
        stmt = ast.stmts[0]
        assert len(stmt.args) == 2

    def test_handle_with(self):
        src = """
        handle {
            perform IO.print(42);
        } with {
            IO.print(msg) -> {
                let x = msg;
            }
        }
        """
        ast = parse(src)
        hw = ast.stmts[0]
        assert isinstance(hw, HandleWith)
        assert len(hw.handlers) == 1
        assert hw.handlers[0].effect == "IO"
        assert hw.handlers[0].operation == "print"
        assert hw.handlers[0].params == ["msg"]

    def test_handle_multiple_handlers(self):
        src = """
        handle {
            let x = 1;
        } with {
            IO.print(msg) -> { let a = 1; }
            IO.read() -> { let b = 2; }
        }
        """
        ast = parse(src)
        hw = ast.stmts[0]
        assert len(hw.handlers) == 2

    def test_resume_expr(self):
        src = """
        handle {
            let x = perform IO.read();
        } with {
            IO.read() -> {
                resume(42);
            }
        }
        """
        ast = parse(src)
        hw = ast.stmts[0]
        # Handler body should contain resume
        handler_body = hw.handlers[0].body
        assert any(isinstance(s, ResumeExpr) for s in handler_body.stmts)

    def test_nested_expressions(self):
        ast = parse("let x = (1 + 2) * 3;")
        assert isinstance(ast.stmts[0], LetDecl)

    def test_unary_ops(self):
        ast = parse("let x = -5; let y = not true;")
        assert isinstance(ast.stmts[0].value, UnaryOp)
        assert isinstance(ast.stmts[1].value, UnaryOp)

    def test_call_expr(self):
        ast = parse("foo(1, 2, 3);")
        assert isinstance(ast.stmts[0], CallExpr)
        assert ast.stmts[0].callee == "foo"
        assert len(ast.stmts[0].args) == 3

    def test_assignment(self):
        ast = parse("x = 10;")
        assert isinstance(ast.stmts[0], Assign)

    def test_return_value(self):
        ast = parse("fn f() { return 42; }")
        fn = ast.stmts[0]
        ret = fn.body.stmts[0]
        assert isinstance(ret, ReturnStmt)

    def test_print_stmt(self):
        ast = parse("print(42);")
        assert isinstance(ast.stmts[0], PrintStmt)

    def test_else_if(self):
        ast = parse("if (a) { let x = 1; } else if (b) { let x = 2; } else { let x = 3; }")
        stmt = ast.stmts[0]
        assert isinstance(stmt.else_body, IfStmt)

    def test_comparison_ops(self):
        ast = parse("let x = a < b;")
        assert isinstance(ast.stmts[0].value, BinOp)
        assert ast.stmts[0].value.op == '<'

    def test_logical_ops(self):
        ast = parse("let x = a and b or c;")
        assert isinstance(ast.stmts[0].value, BinOp)


# =========================================================================
# Section 3: Compiler tests
# =========================================================================
class TestCompiler:
    def test_compile_basic(self):
        ast = parse("let x = 42;")
        chunk = compile_program(ast)
        assert Op.CONST in chunk.code
        assert Op.STORE in chunk.code
        assert Op.HALT in chunk.code

    def test_compile_arithmetic(self):
        ast = parse("let x = 1 + 2;")
        chunk = compile_program(ast)
        assert Op.ADD in chunk.code

    def test_compile_function(self):
        ast = parse("fn f() { return 1; }")
        chunk = compile_program(ast)
        assert Op.STORE in chunk.code

    def test_compile_if(self):
        ast = parse("if (true) { let x = 1; }")
        chunk = compile_program(ast)
        assert Op.JUMP_IF_FALSE in chunk.code

    def test_compile_while(self):
        ast = parse("let x = 5; while (x > 0) { x = x - 1; }")
        chunk = compile_program(ast)
        assert Op.JUMP in chunk.code
        assert Op.JUMP_IF_FALSE in chunk.code

    def test_compile_perform(self):
        ast = parse("perform IO.print(42);")
        chunk = compile_program(ast)
        assert Op.PERFORM in chunk.code

    def test_compile_handle(self):
        src = """
        handle {
            let x = 1;
        } with {
            IO.print(msg) -> { let y = msg; }
        }
        """
        ast = parse(src)
        chunk = compile_program(ast)
        assert Op.INSTALL_HANDLER in chunk.code
        assert Op.REMOVE_HANDLER in chunk.code

    def test_compile_resume(self):
        src = """
        handle {
            let x = perform IO.read();
        } with {
            IO.read() -> { resume(42); }
        }
        """
        ast = parse(src)
        chunk = compile_program(ast)
        # RESUME is in the handler clause's chunk, not the main chunk
        # Verify handler object exists in constants with RESUME in its clause chunk
        from effect_runtime import HandlerObject
        handler_objs = [c for c in chunk.constants if isinstance(c, HandlerObject)]
        assert len(handler_objs) == 1
        clause = list(handler_objs[0].clauses.values())[0]
        assert Op.RESUME in clause.chunk.code

    def test_compile_effect_decl(self):
        ast = parse("effect IO { print(msg); }")
        chunk = compile_program(ast)
        # Effect decl produces no bytecode, just registers effect
        # Check it doesn't crash
        assert Op.HALT in chunk.code

    def test_compile_print(self):
        ast = parse("print(42);")
        chunk = compile_program(ast)
        assert Op.PRINT in chunk.code


# =========================================================================
# Section 4: Basic VM execution (no effects)
# =========================================================================
class TestBasicExecution:
    def test_integer_arithmetic(self):
        result, output = run("let x = 2 + 3; print(x);")
        assert output == ["5"]

    def test_subtraction(self):
        result, output = run("print(10 - 3);")
        assert output == ["7"]

    def test_multiplication(self):
        result, output = run("print(4 * 5);")
        assert output == ["20"]

    def test_division(self):
        result, output = run("print(10 / 3);")
        assert output == ["3"]

    def test_modulo(self):
        result, output = run("print(10 % 3);")
        assert output == ["1"]

    def test_float_arithmetic(self):
        result, output = run("print(3.14 + 1.0);")
        assert output == ["4.140000000000001"] or output == ["4.14"]

    def test_negation(self):
        result, output = run("print(-5);")
        assert output == ["-5"]

    def test_boolean_not(self):
        result, output = run("print(not true);")
        assert output == ["False"]

    def test_comparison(self):
        result, output = run("print(3 < 5); print(5 < 3);")
        assert output == ["True", "False"]

    def test_equality(self):
        result, output = run("print(3 == 3); print(3 != 4);")
        assert output == ["True", "True"]

    def test_string(self):
        result, output = run('print("hello");')
        assert output == ["hello"]

    def test_variables(self):
        result, output = run("let x = 10; let y = 20; print(x + y);")
        assert output == ["30"]

    def test_assignment(self):
        result, output = run("let x = 1; x = 2; print(x);")
        assert output == ["2"]

    def test_if_true(self):
        result, output = run("if (true) { print(1); }")
        assert output == ["1"]

    def test_if_false(self):
        result, output = run("if (false) { print(1); }")
        assert output == []

    def test_if_else(self):
        result, output = run("if (false) { print(1); } else { print(2); }")
        assert output == ["2"]

    def test_while_loop(self):
        result, output = run("""
            let x = 0;
            while (x < 3) {
                print(x);
                x = x + 1;
            }
        """)
        assert output == ["0", "1", "2"]

    def test_function_call(self):
        result, output = run("""
            fn double(n) { return n * 2; }
            print(double(5));
        """)
        assert output == ["10"]

    def test_function_two_params(self):
        result, output = run("""
            fn add(a, b) { return a + b; }
            print(add(3, 4));
        """)
        assert output == ["7"]

    def test_recursive_function(self):
        result, output = run("""
            fn fact(n) {
                if (n <= 1) { return 1; }
                return n * fact(n - 1);
            }
            print(fact(5));
        """)
        assert output == ["120"]

    def test_nested_calls(self):
        result, output = run("""
            fn f(x) { return x + 1; }
            fn g(x) { return f(x) * 2; }
            print(g(3));
        """)
        assert output == ["8"]

    def test_short_circuit_and(self):
        result, output = run("print(false and true);")
        assert output == ["False"]

    def test_short_circuit_or(self):
        result, output = run("print(true or false);")
        assert output == ["True"]

    def test_complex_expression(self):
        result, output = run("print((2 + 3) * (4 - 1));")
        assert output == ["15"]

    def test_multiple_functions(self):
        result, output = run("""
            fn square(x) { return x * x; }
            fn cube(x) { return x * square(x); }
            print(cube(3));
        """)
        assert output == ["27"]

    def test_undefined_variable(self):
        with pytest.raises(RuntimeError, match="Undefined variable"):
            run("print(x);")

    def test_execution_limit(self):
        with pytest.raises(RuntimeError, match="Execution limit"):
            run("while (true) { let x = 1; }")

    def test_bool_true(self):
        result, output = run("print(true);")
        assert output == ["True"]

    def test_string_concat(self):
        result, output = run('print("hello" + " " + "world");')
        assert output == ["hello world"]

    def test_ge_le(self):
        result, output = run("print(3 >= 3); print(3 <= 4);")
        assert output == ["True", "True"]

    def test_ne(self):
        result, output = run("print(1 != 2);")
        assert output == ["True"]


# =========================================================================
# Section 5: Effect declarations
# =========================================================================
class TestEffectDeclarations:
    def test_effect_decl_parses(self):
        result, output = run("effect IO { print(msg); read(); }")
        # No crash

    def test_effect_decl_multiple(self):
        result, output = run("""
            effect IO { print(msg); read(); }
            effect State { get(); set(val); }
        """)
        # No crash

    def test_effect_decl_no_ops(self):
        result, output = run("effect Empty { }")

    def test_effect_with_multiple_params(self):
        result, output = run("effect DB { query(table, filter); }")


# =========================================================================
# Section 6: Perform and handle (core effect runtime)
# =========================================================================
class TestPerformAndHandle:
    def test_simple_handle(self):
        """Handler intercepts perform and executes handler body."""
        result, output = run("""
            handle {
                perform IO.print(42);
            } with {
                IO.print(msg) -> {
                    print(msg);
                }
            }
        """)
        assert output == ["42"]

    def test_handle_no_effect(self):
        """Body doesn't perform -- handler is installed but never triggered."""
        result, output = run("""
            handle {
                print(1);
            } with {
                IO.print(msg) -> {
                    print(msg);
                }
            }
        """)
        assert output == ["1"]

    def test_unhandled_effect(self):
        """Performing without handler raises EffectError."""
        with pytest.raises(EffectError, match="Unhandled effect"):
            run("perform IO.print(42);")

    def test_handle_multiple_performs(self):
        """Handler intercepts multiple performs."""
        result, output = run("""
            handle {
                perform Log.msg("hello");
                perform Log.msg("world");
            } with {
                Log.msg(text) -> {
                    print(text);
                    resume(0);
                }
            }
        """)
        assert output == ["hello", "world"]

    def test_handle_with_resume(self):
        """Resume returns a value to the perform site."""
        result, output = run("""
            handle {
                let x = perform Ask.number();
                print(x);
            } with {
                Ask.number() -> {
                    resume(42);
                }
            }
        """)
        assert output == ["42"]

    def test_resume_with_computation(self):
        """Handler does computation before resuming."""
        result, output = run("""
            handle {
                let x = perform Math.double(5);
                print(x);
            } with {
                Math.double(n) -> {
                    resume(n * 2);
                }
            }
        """)
        assert output == ["10"]

    def test_handle_different_effects(self):
        """Multiple handlers for different effects."""
        result, output = run("""
            handle {
                handle {
                    let x = perform Ask.val();
                    perform Log.msg(x);
                } with {
                    Ask.val() -> {
                        resume(99);
                    }
                }
            } with {
                Log.msg(v) -> {
                    print(v);
                    resume(0);
                }
            }
        """)
        assert output == ["99"]

    def test_handler_no_resume(self):
        """Handler that doesn't resume -- aborts continuation."""
        result, output = run("""
            handle {
                perform Abort.stop();
                print(999);
            } with {
                Abort.stop() -> {
                    print(0);
                }
            }
        """)
        # print(999) should NOT execute because handler doesn't resume
        assert output == ["0"]

    def test_handler_override_value(self):
        """Handler transforms the value."""
        result, output = run("""
            handle {
                let x = perform Transform.inc(10);
                print(x);
            } with {
                Transform.inc(n) -> {
                    resume(n + 1);
                }
            }
        """)
        assert output == ["11"]

    def test_perform_in_function(self):
        """Effect performed inside a function, handled outside."""
        result, output = run("""
            fn greet() {
                perform IO.say("hello");
            }
            handle {
                greet();
            } with {
                IO.say(msg) -> {
                    print(msg);
                    resume(0);
                }
            }
        """)
        assert output == ["hello"]

    def test_multiple_perform_with_resume(self):
        """Multiple performs with resume, tracking state."""
        result, output = run("""
            handle {
                let a = perform Counter.next();
                let b = perform Counter.next();
                let c = perform Counter.next();
                print(a);
                print(b);
                print(c);
            } with {
                Counter.next() -> {
                    resume(1);
                }
            }
        """)
        # Each resume returns 1 (stateless handler)
        assert output == ["1", "1", "1"]

    def test_perform_with_two_args(self):
        """Perform with multiple arguments."""
        result, output = run("""
            handle {
                let x = perform Math.add(3, 4);
                print(x);
            } with {
                Math.add(a, b) -> {
                    resume(a + b);
                }
            }
        """)
        assert output == ["7"]

    def test_effect_in_loop(self):
        """Perform inside a loop, handler resumes each time."""
        result, output = run("""
            handle {
                let i = 0;
                while (i < 3) {
                    let x = perform Gen.val(i);
                    print(x);
                    i = i + 1;
                }
            } with {
                Gen.val(n) -> {
                    resume(n * 10);
                }
            }
        """)
        assert output == ["0", "10", "20"]

    def test_effect_in_conditional(self):
        """Perform inside conditional."""
        result, output = run("""
            handle {
                if (true) {
                    let x = perform Val.get();
                    print(x);
                }
            } with {
                Val.get() -> {
                    resume(77);
                }
            }
        """)
        assert output == ["77"]


# =========================================================================
# Section 7: Nested handlers
# =========================================================================
class TestNestedHandlers:
    def test_inner_handler_takes_priority(self):
        """Inner handler catches effect before outer handler."""
        result, output = run("""
            handle {
                handle {
                    let x = perform V.get();
                    print(x);
                } with {
                    V.get() -> {
                        resume(1);
                    }
                }
            } with {
                V.get() -> {
                    resume(2);
                }
            }
        """)
        assert output == ["1"]

    def test_outer_handler_catches_unhandled(self):
        """Outer handler catches what inner doesn't handle."""
        result, output = run("""
            handle {
                handle {
                    perform A.do();
                    perform B.do();
                } with {
                    A.do() -> {
                        print("A handled");
                        resume(0);
                    }
                }
            } with {
                B.do() -> {
                    print("B handled");
                    resume(0);
                }
            }
        """)
        assert output == ["A handled", "B handled"]

    def test_handler_scope_removal(self):
        """After handle block, handler is removed."""
        result, output = run("""
            handle {
                handle {
                    let x = 1;
                } with {
                    V.get() -> { resume(0); }
                }
                // V.get handler is now removed
                let y = perform V.get();
                print(y);
            } with {
                V.get() -> {
                    resume(99);
                }
            }
        """)
        assert output == ["99"]

    def test_deeply_nested(self):
        """Three levels of handler nesting."""
        result, output = run("""
            handle {
                handle {
                    handle {
                        let x = perform C.val();
                        print(x);
                    } with {
                        C.val() -> { resume(3); }
                    }
                } with {
                    B.val() -> { resume(2); }
                }
            } with {
                A.val() -> { resume(1); }
            }
        """)
        assert output == ["3"]


# =========================================================================
# Section 8: Effect patterns (real-world use cases)
# =========================================================================
class TestEffectPatterns:
    def test_exception_pattern(self):
        """Effects as exception handling -- abort without resume."""
        result, output = run("""
            handle {
                print("start");
                perform Error.throw("oops");
                print("unreachable");
            } with {
                Error.throw(msg) -> {
                    print(msg);
                }
            }
        """)
        assert output == ["start", "oops"]

    def test_logging_pattern(self):
        """Effects as a logging sink."""
        result, output = run("""
            fn work() {
                perform Log.info("step 1");
                perform Log.info("step 2");
                perform Log.info("done");
            }
            handle {
                work();
            } with {
                Log.info(msg) -> {
                    print(msg);
                    resume(0);
                }
            }
        """)
        assert output == ["step 1", "step 2", "done"]

    def test_reader_pattern(self):
        """Effects as environment/config reader."""
        result, output = run("""
            fn get_config() {
                return perform Config.read();
            }
            handle {
                let val = get_config();
                print(val);
            } with {
                Config.read() -> {
                    resume(42);
                }
            }
        """)
        assert output == ["42"]

    def test_early_return_pattern(self):
        """Effects for early return from nested computation."""
        result, output = run("""
            fn search(items) {
                let i = 0;
                while (i < items) {
                    if (i == 3) {
                        perform Found.at(i);
                    }
                    i = i + 1;
                }
                return -1;
            }
            handle {
                let result = search(5);
                print(result);
            } with {
                Found.at(idx) -> {
                    print(idx);
                }
            }
        """)
        # Handler doesn't resume, so search is aborted and print(result) not reached
        assert output == ["3"]

    def test_generator_pattern(self):
        """Effects as generators -- yield values."""
        result, output = run("""
            fn generate() {
                perform Yield.val(1);
                perform Yield.val(2);
                perform Yield.val(3);
            }
            handle {
                generate();
            } with {
                Yield.val(v) -> {
                    print(v);
                    resume(0);
                }
            }
        """)
        assert output == ["1", "2", "3"]

    def test_accumulator_pattern(self):
        """Handler accumulates values via side effects (print)."""
        result, output = run("""
            handle {
                perform Emit.val(10);
                perform Emit.val(20);
                perform Emit.val(30);
            } with {
                Emit.val(v) -> {
                    print(v);
                    resume(0);
                }
            }
        """)
        assert output == ["10", "20", "30"]

    def test_transform_pipeline(self):
        """Chained effects for transformation."""
        result, output = run("""
            fn process(x) {
                let a = perform Step.transform(x);
                return a;
            }
            handle {
                let result = process(5);
                print(result);
            } with {
                Step.transform(v) -> {
                    resume(v * 3);
                }
            }
        """)
        assert output == ["15"]

    def test_choice_pattern(self):
        """Effect for non-deterministic choice."""
        result, output = run("""
            handle {
                let x = perform Choice.pick();
                print(x);
            } with {
                Choice.pick() -> {
                    resume(42);
                }
            }
        """)
        assert output == ["42"]


# =========================================================================
# Section 9: Resume semantics
# =========================================================================
class TestResumeSemantics:
    def test_resume_returns_value(self):
        result, output = run("""
            handle {
                let x = perform V.get();
                print(x);
            } with {
                V.get() -> { resume(100); }
            }
        """)
        assert output == ["100"]

    def test_resume_string(self):
        result, output = run("""
            handle {
                let x = perform V.get();
                print(x);
            } with {
                V.get() -> { resume("hello"); }
            }
        """)
        assert output == ["hello"]

    def test_resume_bool(self):
        result, output = run("""
            handle {
                let x = perform V.get();
                print(x);
            } with {
                V.get() -> { resume(true); }
            }
        """)
        assert output == ["True"]

    def test_resume_computation_result(self):
        result, output = run("""
            handle {
                let x = perform V.compute(5);
                print(x);
            } with {
                V.compute(n) -> {
                    let result = n * n + 1;
                    resume(result);
                }
            }
        """)
        assert output == ["26"]

    def test_resume_outside_handler_errors(self):
        """resume outside handler context should error."""
        with pytest.raises(ResumeError):
            run("resume(42);")

    def test_no_resume_aborts(self):
        """Not resuming means continuation is abandoned."""
        result, output = run("""
            handle {
                perform X.stop();
                print("not reached");
            } with {
                X.stop() -> {
                    print("stopped");
                }
            }
        """)
        assert output == ["stopped"]
        assert "not reached" not in output

    def test_resume_preserves_locals(self):
        """After resume, local variables at perform site are available."""
        result, output = run("""
            handle {
                let before = 10;
                let x = perform V.get();
                print(before);
                print(x);
            } with {
                V.get() -> { resume(20); }
            }
        """)
        assert output == ["10", "20"]


# =========================================================================
# Section 10: Functions and effects interaction
# =========================================================================
class TestFunctionsAndEffects:
    def test_effect_in_nested_function(self):
        result, output = run("""
            fn inner() {
                return perform V.get();
            }
            fn outer() {
                return inner();
            }
            handle {
                let x = outer();
                print(x);
            } with {
                V.get() -> { resume(42); }
            }
        """)
        assert output == ["42"]

    def test_function_after_handle(self):
        result, output = run("""
            fn f() { return 5; }
            handle {
                perform X.noop();
            } with {
                X.noop() -> { resume(0); }
            }
            print(f());
        """)
        assert output == ["5"]

    def test_effect_across_function_boundary(self):
        """Effect performed deep in call stack, handled at top."""
        result, output = run("""
            fn level3() { return perform Deep.val(); }
            fn level2() { return level3(); }
            fn level1() { return level2(); }
            handle {
                let x = level1();
                print(x);
            } with {
                Deep.val() -> { resume(777); }
            }
        """)
        assert output == ["777"]

    def test_function_with_multiple_effects(self):
        result, output = run("""
            fn work() {
                perform Log.msg("start");
                let x = perform Data.fetch();
                perform Log.msg("got data");
                return x;
            }
            handle {
                handle {
                    let result = work();
                    print(result);
                } with {
                    Data.fetch() -> { resume(42); }
                }
            } with {
                Log.msg(m) -> {
                    print(m);
                    resume(0);
                }
            }
        """)
        assert output == ["start", "got data", "42"]

    def test_recursive_with_effect(self):
        """Recursive function performing effects."""
        result, output = run("""
            fn countdown(n) {
                if (n <= 0) { return 0; }
                perform Tick.val(n);
                return countdown(n - 1);
            }
            handle {
                countdown(3);
            } with {
                Tick.val(v) -> {
                    print(v);
                    resume(0);
                }
            }
        """)
        assert output == ["3", "2", "1"]


# =========================================================================
# Section 11: Edge cases and error handling
# =========================================================================
class TestEdgeCases:
    def test_empty_program(self):
        result, output = run("")
        assert output == []

    def test_empty_handle_body(self):
        result, output = run("""
            handle {
            } with {
                X.y() -> { resume(0); }
            }
        """)
        assert output == []

    def test_handle_with_print_after(self):
        result, output = run("""
            handle {
                let x = perform V.get();
            } with {
                V.get() -> { resume(5); }
            }
            print(42);
        """)
        assert output == ["42"]

    def test_zero_arg_perform(self):
        result, output = run("""
            handle {
                let x = perform V.get();
                print(x);
            } with {
                V.get() -> { resume(0); }
            }
        """)
        assert output == ["0"]

    def test_division_by_zero(self):
        result, output = run("print(10 / 0);")
        assert output == ["0"]

    def test_deeply_nested_arithmetic(self):
        result, output = run("print(((1 + 2) * 3 - 4) / 5);")
        assert output == ["1"]

    def test_many_variables(self):
        result, output = run("""
            let a = 1; let b = 2; let c = 3; let d = 4; let e = 5;
            print(a + b + c + d + e);
        """)
        assert output == ["15"]

    def test_function_returning_none(self):
        result, output = run("""
            fn f() { let x = 1; }
            let result = f();
        """)
        # Should not crash

    def test_handler_with_conditional(self):
        result, output = run("""
            handle {
                let x = perform V.get(5);
                print(x);
            } with {
                V.get(n) -> {
                    if (n > 3) {
                        resume(n * 2);
                    } else {
                        resume(n);
                    }
                }
            }
        """)
        assert output == ["10"]

    def test_handler_with_loop(self):
        result, output = run("""
            handle {
                let x = perform V.sum(3);
                print(x);
            } with {
                V.sum(n) -> {
                    let total = 0;
                    let i = 0;
                    while (i <= n) {
                        total = total + i;
                        i = i + 1;
                    }
                    resume(total);
                }
            }
        """)
        assert output == ["6"]  # 0+1+2+3

    def test_return_in_handle_body(self):
        """Returning from a function inside handle body."""
        result, output = run("""
            fn work() {
                handle {
                    let x = perform V.get();
                    return x;
                } with {
                    V.get() -> { resume(42); }
                }
            }
            print(work());
        """)
        assert output == ["42"]


# =========================================================================
# Section 12: Composition correctness tests
# =========================================================================
class TestCompositionCorrectness:
    def test_effect_value_types(self):
        """Different value types through effects."""
        result, output = run("""
            handle {
                let a = perform V.int();
                let b = perform V.str();
                let c = perform V.bool();
                print(a);
                print(b);
                print(c);
            } with {
                V.int() -> { resume(42); }
                V.str() -> { resume("hello"); }
                V.bool() -> { resume(true); }
            }
        """)
        assert output == ["42", "hello", "True"]

    def test_effect_with_arithmetic_resume(self):
        """Resume value used in arithmetic."""
        result, output = run("""
            handle {
                let x = perform V.get();
                let y = x * 2 + 1;
                print(y);
            } with {
                V.get() -> { resume(5); }
            }
        """)
        assert output == ["11"]

    def test_sequential_handles(self):
        """Two handle blocks in sequence."""
        result, output = run("""
            handle {
                let a = perform A.val();
                print(a);
            } with {
                A.val() -> { resume(1); }
            }
            handle {
                let b = perform B.val();
                print(b);
            } with {
                B.val() -> { resume(2); }
            }
        """)
        assert output == ["1", "2"]

    def test_effect_preserves_call_stack(self):
        """After resume, call stack is correctly restored."""
        result, output = run("""
            fn helper() {
                let x = perform V.get();
                return x + 1;
            }
            fn caller() {
                let y = helper();
                return y + 10;
            }
            handle {
                let result = caller();
                print(result);
            } with {
                V.get() -> { resume(5); }
            }
        """)
        assert output == ["16"]  # 5 + 1 + 10

    def test_effect_in_loop_with_accumulation(self):
        """Effects in a loop with external accumulation."""
        result, output = run("""
            handle {
                let sum = 0;
                let i = 0;
                while (i < 5) {
                    let x = perform Gen.next(i);
                    sum = sum + x;
                    i = i + 1;
                }
                print(sum);
            } with {
                Gen.next(n) -> { resume(n * 2); }
            }
        """)
        # sum = 0+2+4+6+8 = 20
        assert output == ["20"]

    def test_handler_sees_handler_params(self):
        """Handler clause can use its parameters in computation."""
        result, output = run("""
            handle {
                let x = perform Math.square(7);
                print(x);
            } with {
                Math.square(n) -> {
                    resume(n * n);
                }
            }
        """)
        assert output == ["49"]


# =========================================================================
# Section 13: Stress and boundary tests
# =========================================================================
class TestStressBoundary:
    def test_many_performs(self):
        """Many sequential performs."""
        result, output = run("""
            handle {
                let i = 0;
                while (i < 20) {
                    perform Count.tick();
                    i = i + 1;
                }
                print(i);
            } with {
                Count.tick() -> { resume(0); }
            }
        """)
        assert output == ["20"]

    def test_long_chain_of_calls(self):
        """Effect at the bottom of a deep call chain."""
        result, output = run("""
            fn f5() { return perform V.get(); }
            fn f4() { return f5(); }
            fn f3() { return f4(); }
            fn f2() { return f3(); }
            fn f1() { return f2(); }
            handle {
                let x = f1();
                print(x);
            } with {
                V.get() -> { resume(42); }
            }
        """)
        assert output == ["42"]

    def test_fibonacci_with_effect(self):
        """Fibonacci using effects for base cases."""
        result, output = run("""
            fn fib(n) {
                if (n <= 1) {
                    return perform Base.val(n);
                }
                return fib(n - 1) + fib(n - 2);
            }
            handle {
                let x = fib(7);
                print(x);
            } with {
                Base.val(n) -> { resume(n); }
            }
        """)
        assert output == ["13"]

    def test_nested_loops_with_effects(self):
        result, output = run("""
            handle {
                let i = 0;
                while (i < 3) {
                    let j = 0;
                    while (j < 2) {
                        perform Log.val(i * 10 + j);
                        j = j + 1;
                    }
                    i = i + 1;
                }
            } with {
                Log.val(v) -> {
                    print(v);
                    resume(0);
                }
            }
        """)
        assert output == ["0", "1", "10", "11", "20", "21"]

    def test_handler_chain_abort(self):
        """Inner handler aborts, outer handler catches different effect."""
        result, output = run("""
            handle {
                handle {
                    perform Log.msg("before");
                    perform Abort.now();
                    perform Log.msg("after");
                } with {
                    Abort.now() -> {
                        print("aborted");
                    }
                }
            } with {
                Log.msg(m) -> {
                    print(m);
                    resume(0);
                }
            }
        """)
        assert output == ["before", "aborted"]


# =========================================================================
# Section 14: Integration / smoke tests
# =========================================================================
class TestIntegration:
    def test_full_program_with_effects(self):
        """A realistic program using effects for I/O abstraction."""
        result, output = run("""
            effect Console { write(msg); prompt(msg); }

            fn greet(name) {
                perform Console.write("Hello, ");
                perform Console.write(name);
                perform Console.write("!");
            }

            handle {
                greet("World");
            } with {
                Console.write(msg) -> {
                    print(msg);
                    resume(0);
                }
                Console.prompt(msg) -> {
                    print(msg);
                    resume("default");
                }
            }
        """)
        assert output == ["Hello, ", "World", "!"]

    def test_state_effect_simulation(self):
        """Simulate stateful computation via handler-local state."""
        result, output = run("""
            fn increment() {
                let current = perform State.get();
                perform State.set(current + 1);
            }

            handle {
                perform State.set(0);
                increment();
                increment();
                increment();
                let final = perform State.get();
                print(final);
            } with {
                State.get() -> {
                    resume(0);
                }
                State.set(v) -> {
                    resume(0);
                }
            }
        """)
        # Note: without true mutable handler state, each get returns 0
        # This tests the structural pattern, not stateful handlers
        assert output == ["0"]

    def test_program_with_functions_loops_effects(self):
        """Complex program combining all features."""
        result, output = run("""
            fn range_sum(start, end) {
                let sum = 0;
                let i = start;
                while (i < end) {
                    let x = perform Filter.check(i);
                    if (x) {
                        sum = sum + i;
                    }
                    i = i + 1;
                }
                return sum;
            }

            handle {
                let result = range_sum(0, 10);
                print(result);
            } with {
                Filter.check(n) -> {
                    // Only keep even numbers
                    if (n % 2 == 0) {
                        resume(true);
                    } else {
                        resume(false);
                    }
                }
            }
        """)
        # Sum of even numbers 0-9: 0+2+4+6+8 = 20
        assert output == ["20"]

    def test_cooperative_effects(self):
        """Two different effect types cooperating."""
        result, output = run("""
            fn process() {
                let data = perform Input.read();
                let transformed = data * 2;
                perform Output.write(transformed);
                return transformed;
            }

            handle {
                handle {
                    let x = process();
                    print(x);
                } with {
                    Input.read() -> { resume(21); }
                }
            } with {
                Output.write(v) -> {
                    print(v);
                    resume(0);
                }
            }
        """)
        assert output == ["42", "42"]

    def test_effect_as_strategy_pattern(self):
        """Effects implementing strategy pattern."""
        result, output = run("""
            fn compute(a, b) {
                return perform Op.apply(a, b);
            }

            // Addition strategy
            handle {
                let x = compute(3, 4);
                print(x);
            } with {
                Op.apply(a, b) -> { resume(a + b); }
            }

            // Multiplication strategy
            handle {
                let y = compute(3, 4);
                print(y);
            } with {
                Op.apply(a, b) -> { resume(a * b); }
            }
        """)
        assert output == ["7", "12"]


# =========================================================================
# Run with: pytest test_effect_runtime.py -v
# =========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
