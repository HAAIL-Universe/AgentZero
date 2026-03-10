"""
Tests for C032: Effect System Type Checker
Challenge C032 -- AgentZero Session 033

Tests cover:
  1. Lexer basics
  2. Parser -- all node types
  3. Basic type checking (inherited from C013 patterns)
  4. Effect inference -- print, perform, throw
  5. Effect annotations -- declared vs inferred
  6. Effect handlers -- handle/with
  7. Effect propagation through call chains
  8. Try/catch and Error effect
  9. Custom effect declarations
  10. Effect polymorphism and row variables
  11. Resume in handlers
  12. Nested handlers
  13. Complex compositions
  14. Edge cases and error messages
"""

import pytest
from effect_system import (
    # Lexer
    lex, Token, TokenType, LexError,
    # Parser
    Parser, ParseError, parse_source, Program,
    # AST
    IntLit, FloatLit, StringLit, BoolLit, Var, UnaryOp, BinOp,
    Assign, LetDecl, Block, IfStmt, WhileStmt, FnDecl, CallExpr,
    ReturnStmt, PrintStmt, Perform, HandleWith, EffectHandler,
    Resume, ThrowExpr, TryCatch, EffectDecl,
    # Types
    INT, FLOAT, STRING, BOOL, VOID, TInt, TFloat, TString, TBool, TVoid,
    TFunc, TVar,
    # Effects
    Effect, EffectRow, EffectVar, PURE, IO, STATE, ERROR, ASYNC,
    # Checker
    EffectChecker, EffectError, check_source, check_program, format_errors,
    # Utilities
    resolve, unify, UnificationError, effect_subset, unify_effects,
    EffectRegistry,
)


# ============================================================
# Section 1: Lexer Tests
# ============================================================

class TestLexer:
    def test_integers(self):
        tokens = lex("42 0 999")
        assert tokens[0].type == TokenType.INT
        assert tokens[0].value == 42
        assert tokens[1].value == 0
        assert tokens[2].value == 999

    def test_floats(self):
        tokens = lex("3.14 0.5")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == 3.14

    def test_strings(self):
        tokens = lex('"hello" "world"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_string_escapes(self):
        tokens = lex(r'"line\none"')
        assert tokens[0].value == "line\none"

    def test_keywords(self):
        tokens = lex("let fn if else while return true false and or not print")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert TokenType.LET in types
        assert TokenType.FN in types
        assert TokenType.IF in types
        assert TokenType.ELSE in types
        assert TokenType.WHILE in types
        assert TokenType.RETURN in types
        assert TokenType.TRUE in types
        assert TokenType.FALSE in types

    def test_effect_keywords(self):
        tokens = lex("perform handle with resume throw try catch effect")
        types = [t.type for t in tokens[:-1]]
        assert TokenType.PERFORM in types
        assert TokenType.HANDLE in types
        assert TokenType.WITH in types
        assert TokenType.RESUME in types
        assert TokenType.THROW in types
        assert TokenType.TRY in types
        assert TokenType.CATCH in types
        assert TokenType.EFFECT in types

    def test_operators(self):
        tokens = lex("+ - * / % = == != < > <= >= ! -> . , ; : ( ) { }")
        types = [t.type for t in tokens[:-1]]
        assert TokenType.PLUS in types
        assert TokenType.ARROW in types
        assert TokenType.DOT in types
        assert TokenType.COLON in types
        assert TokenType.BANG in types

    def test_identifiers(self):
        tokens = lex("foo bar_baz _x abc123")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "foo"

    def test_line_tracking(self):
        tokens = lex("a\nb\nc")
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3

    def test_comments(self):
        tokens = lex("a // comment\nb")
        assert len([t for t in tokens if t.type != TokenType.EOF]) == 2

    def test_unterminated_string(self):
        with pytest.raises(LexError):
            lex('"hello')

    def test_unexpected_char(self):
        with pytest.raises(LexError):
            lex("@")


# ============================================================
# Section 2: Parser Tests
# ============================================================

class TestParser:
    def test_let_declaration(self):
        prog = parse_source("let x = 42;")
        assert len(prog.stmts) == 1
        assert isinstance(prog.stmts[0], LetDecl)
        assert prog.stmts[0].name == "x"

    def test_let_with_type_annotation(self):
        prog = parse_source("let x: int = 42;")
        stmt = prog.stmts[0]
        assert isinstance(stmt, LetDecl)
        assert stmt.type_ann == INT

    def test_function_declaration(self):
        prog = parse_source("fn add(a, b) { return a + b; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, FnDecl)
        assert stmt.name == "add"
        assert len(stmt.params) == 2

    def test_function_with_types(self):
        prog = parse_source("fn add(a: int, b: int) -> int { return a + b; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, FnDecl)
        assert stmt.ret_ann == INT
        assert stmt.params[0] == ("a", INT)
        assert stmt.params[1] == ("b", INT)

    def test_function_with_effects(self):
        prog = parse_source("fn greet(name: string) -> void ! IO { print(name); }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, FnDecl)
        assert stmt.effect_ann is not None
        assert IO in stmt.effect_ann.effects

    def test_function_with_multiple_effects(self):
        prog = parse_source("fn work() -> int ! IO, State { return 1; }")
        stmt = prog.stmts[0]
        assert IO in stmt.effect_ann.effects
        assert STATE in stmt.effect_ann.effects

    def test_if_stmt(self):
        prog = parse_source("if (true) { let x = 1; }")
        assert isinstance(prog.stmts[0], IfStmt)

    def test_if_else(self):
        prog = parse_source("if (true) { let x = 1; } else { let x = 2; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_body is not None

    def test_while_stmt(self):
        prog = parse_source("while (true) { let x = 1; }")
        assert isinstance(prog.stmts[0], WhileStmt)

    def test_perform(self):
        prog = parse_source('perform IO.print("hello");')
        stmt = prog.stmts[0]
        assert isinstance(stmt, Perform)
        assert stmt.effect == "IO"
        assert stmt.operation == "print"
        assert len(stmt.args) == 1

    def test_handle_with(self):
        src = """
        handle {
            perform IO.print("hello");
        } with {
            IO.print(msg) -> {
                resume(0);
            }
        }
        """
        prog = parse_source(src)
        stmt = prog.stmts[0]
        assert isinstance(stmt, HandleWith)
        assert len(stmt.handlers) == 1
        assert stmt.handlers[0].effect == "IO"
        assert stmt.handlers[0].operation == "print"

    def test_throw(self):
        prog = parse_source('throw("error!");')
        stmt = prog.stmts[0]
        assert isinstance(stmt, ThrowExpr)

    def test_try_catch(self):
        src = """
        try {
            throw("oops");
        } catch(e) {
            print(e);
        }
        """
        prog = parse_source(src)
        stmt = prog.stmts[0]
        assert isinstance(stmt, TryCatch)
        assert stmt.catch_name == "e"

    def test_effect_declaration(self):
        src = """
        effect Logger {
            log(string) -> void;
            level() -> int;
        }
        """
        prog = parse_source(src)
        stmt = prog.stmts[0]
        assert isinstance(stmt, EffectDecl)
        assert stmt.name == "Logger"
        assert len(stmt.operations) == 2
        assert stmt.operations[0][0] == "log"
        assert stmt.operations[1][0] == "level"

    def test_resume(self):
        # Resume in expression position
        src = """
        handle {
            perform IO.print("hi");
        } with {
            IO.print(msg) -> {
                resume(0);
            }
        }
        """
        prog = parse_source(src)
        handler = prog.stmts[0].handlers[0]
        # The resume is inside the handler body
        assert isinstance(handler.body.stmts[0], Resume)

    def test_binary_ops(self):
        prog = parse_source("let x = 1 + 2 * 3;")
        stmt = prog.stmts[0]
        # Should be 1 + (2 * 3) due to precedence
        assert isinstance(stmt.value, BinOp)
        assert stmt.value.op == "+"

    def test_comparison(self):
        prog = parse_source("let x = 1 < 2;")
        assert isinstance(prog.stmts[0].value, BinOp)
        assert prog.stmts[0].value.op == "<"

    def test_logical(self):
        prog = parse_source("let x = true and false;")
        assert isinstance(prog.stmts[0].value, BinOp)
        assert prog.stmts[0].value.op == "and"

    def test_unary_minus(self):
        prog = parse_source("let x = -5;")
        assert isinstance(prog.stmts[0].value, UnaryOp)
        assert prog.stmts[0].value.op == "-"

    def test_unary_not(self):
        prog = parse_source("let x = not true;")
        assert isinstance(prog.stmts[0].value, UnaryOp)
        assert prog.stmts[0].value.op == "not"

    def test_call_expr(self):
        prog = parse_source("let x = foo(1, 2);")
        assert isinstance(prog.stmts[0].value, CallExpr)
        assert prog.stmts[0].value.callee == "foo"

    def test_assignment(self):
        prog = parse_source("let x = 1; x = 2;")
        assert isinstance(prog.stmts[1], Assign)

    def test_nested_blocks(self):
        prog = parse_source("{ { let x = 1; } }")
        assert isinstance(prog.stmts[0], Block)
        assert isinstance(prog.stmts[0].stmts[0], Block)

    def test_parse_error(self):
        with pytest.raises(ParseError):
            parse_source("let = ;")

    def test_empty_function(self):
        prog = parse_source("fn noop() { }")
        assert isinstance(prog.stmts[0], FnDecl)
        assert len(prog.stmts[0].body.stmts) == 0

    def test_multiple_statements(self):
        prog = parse_source("let a = 1; let b = 2; let c = a + b;")
        assert len(prog.stmts) == 3


# ============================================================
# Section 3: Basic Type Checking
# ============================================================

class TestBasicTypes:
    def test_int_literal(self):
        errors, checker = check_source("let x = 42;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TInt)

    def test_float_literal(self):
        errors, checker = check_source("let x = 3.14;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TFloat)

    def test_string_literal(self):
        errors, checker = check_source('let x = "hello";')
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TString)

    def test_bool_literal(self):
        errors, checker = check_source("let x = true;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TBool)

    def test_int_arithmetic(self):
        errors, _ = check_source("let x = 1 + 2;")
        assert len(errors) == 0

    def test_float_promotion(self):
        errors, checker = check_source("let x = 1 + 2.0;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TFloat)

    def test_string_concat(self):
        errors, checker = check_source('let x = "a" + "b";')
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TString)

    def test_type_error_arithmetic(self):
        errors, _ = check_source('let x = 1 + "hello";')
        assert len(errors) == 1
        assert "Cannot apply" in errors[0].message

    def test_comparison(self):
        errors, checker = check_source("let x = 1 < 2;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TBool)

    def test_equality(self):
        errors, _ = check_source("let x = 1 == 1;")
        assert len(errors) == 0

    def test_logical_operators(self):
        errors, _ = check_source("let x = true and false;")
        assert len(errors) == 0

    def test_unary_negation(self):
        errors, checker = check_source("let x = -5;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TInt)

    def test_unary_not(self):
        errors, checker = check_source("let x = not true;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TBool)

    def test_undefined_variable(self):
        errors, _ = check_source("let x = y;")
        assert len(errors) == 1
        assert "Undefined" in errors[0].message

    def test_type_annotation_match(self):
        errors, _ = check_source("let x: int = 42;")
        assert len(errors) == 0

    def test_type_annotation_mismatch(self):
        errors, _ = check_source('let x: int = "hello";')
        assert len(errors) == 1
        assert "Type mismatch" in errors[0].message

    def test_assignment_type_check(self):
        errors, _ = check_source('let x = 1; x = "hello";')
        assert len(errors) == 1
        assert "Cannot assign" in errors[0].message

    def test_assignment_to_undefined(self):
        errors, _ = check_source("x = 1;")
        assert len(errors) == 1
        assert "undefined" in errors[0].message.lower()

    def test_if_condition_bool(self):
        errors, _ = check_source("if (true) { let x = 1; }")
        assert len(errors) == 0

    def test_if_condition_non_bool(self):
        errors, _ = check_source("if (42) { let x = 1; }")
        assert len(errors) == 1
        assert "Condition must be bool" in errors[0].message

    def test_while_condition(self):
        errors, _ = check_source("while (true) { let x = 1; }")
        assert len(errors) == 0

    def test_while_condition_non_bool(self):
        errors, _ = check_source('while ("yes") { let x = 1; }')
        assert len(errors) == 1

    def test_block_scoping(self):
        errors, checker = check_source("let x = 1; { let y = 2; }")
        assert len(errors) == 0
        assert checker.env.lookup("x") is not None
        # y should not be in outer scope
        assert checker.env.lookup("y") is None


# ============================================================
# Section 4: Function Type Checking
# ============================================================

class TestFunctions:
    def test_simple_function(self):
        errors, _ = check_source("fn add(a, b) { return a + b; }")
        assert len(errors) == 0

    def test_function_call(self):
        src = """
        fn add(a, b) { return a + b; }
        let x = add(1, 2);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_function_arity_mismatch(self):
        src = """
        fn add(a, b) { return a + b; }
        let x = add(1);
        """
        errors, _ = check_source(src)
        assert len(errors) == 1
        assert "expects 2 args" in errors[0].message

    def test_typed_params(self):
        src = """
        fn add(a: int, b: int) -> int {
            return a + b;
        }
        let x = add(1, 2);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_typed_param_mismatch(self):
        src = """
        fn add(a: int, b: int) -> int {
            return a + b;
        }
        let x = add(1, "hello");
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1

    def test_return_type_check(self):
        src = """
        fn foo() -> int {
            return "hello";
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 1
        assert "Return type mismatch" in errors[0].message

    def test_return_outside_function(self):
        errors, _ = check_source("return 42;")
        assert len(errors) == 1
        assert "outside of function" in errors[0].message

    def test_void_return(self):
        src = """
        fn noop() -> void {
            return;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_calling_non_function(self):
        errors, _ = check_source("let x = 5; let y = x(1);")
        assert len(errors) == 1
        assert "not a function" in errors[0].message

    def test_recursive_function(self):
        src = """
        fn fact(n) {
            if (n < 2) {
                return 1;
            }
            return n * fact(n - 1);
        }
        let x = fact(5);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_nested_function_calls(self):
        src = """
        fn double(x) { return x + x; }
        fn quadruple(x) { return double(double(x)); }
        let x = quadruple(3);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0


# ============================================================
# Section 5: Effect Inference
# ============================================================

class TestEffectInference:
    def test_pure_function(self):
        src = """
        fn add(a, b) {
            return a + b;
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("add")
        assert effects.is_pure

    def test_print_implies_io(self):
        src = """
        fn greet(name) {
            print(name);
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("greet")
        assert IO in effects.effects

    def test_perform_implies_effect(self):
        src = """
        fn read_config() {
            perform State.get("config");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("read_config")
        assert STATE in effects.effects

    def test_throw_implies_error(self):
        src = """
        fn validate(x) {
            if (x < 0) {
                throw("negative");
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("validate")
        assert ERROR in effects.effects

    def test_multiple_effects_inferred(self):
        src = """
        fn process() {
            print("start");
            perform State.get("data");
            throw("oops");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("process")
        assert IO in effects.effects
        assert STATE in effects.effects
        assert ERROR in effects.effects

    def test_no_effect_in_pure_function(self):
        src = """
        fn square(x) {
            return x * x;
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("square")
        assert effects.is_pure

    def test_conditional_effect(self):
        """Effect inferred even if only in one branch."""
        src = """
        fn maybe_print(flag) {
            if (flag) {
                print("yes");
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("maybe_print")
        assert IO in effects.effects

    def test_loop_effect(self):
        src = """
        fn count() {
            let i = 0;
            while (i < 10) {
                print(i);
                i = i + 1;
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("count")
        assert IO in effects.effects


# ============================================================
# Section 6: Effect Annotations
# ============================================================

class TestEffectAnnotations:
    def test_matching_annotation(self):
        src = """
        fn greet(name: string) -> void ! IO {
            print(name);
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_missing_effect_in_annotation(self):
        """Function performs IO but doesn't declare it."""
        src = """
        fn greet(name: string) -> void {
            print(name);
        }
        """
        # No annotation means no check -- only explicit annotations are checked
        errors, checker = check_source(src)
        # No error because no effect annotation means "effects not declared"
        assert len(errors) == 0
        # But effects are still inferred
        assert IO in checker.get_function_effects("greet").effects

    def test_undeclared_effect_error(self):
        """Function performs IO but declares only State."""
        src = """
        fn greet(name: string) -> void ! State {
            print(name);
        }
        """
        errors, _ = check_source(src)
        assert any(e.kind == "effect" for e in errors)
        assert any("undeclared effect" in e.message.lower() for e in errors)

    def test_extra_declared_effect_is_ok(self):
        """Declaring more effects than used is fine (conservative)."""
        src = """
        fn greet(name: string) -> void ! IO, State {
            print(name);
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_pure_with_io_annotation(self):
        """Pure function with IO annotation is ok (conservative)."""
        src = """
        fn add(a: int, b: int) -> int ! IO {
            return a + b;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_multiple_effect_annotation(self):
        src = """
        fn work() -> void ! IO, State, Error {
            print("working");
            perform State.set("status", 1);
            throw("done");
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0


# ============================================================
# Section 7: Effect Propagation
# ============================================================

class TestEffectPropagation:
    def test_call_propagates_declared_effects(self):
        src = """
        fn greet() -> void ! IO {
            print("hello");
        }
        fn main() {
            greet();
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        # main calls greet which has IO effect
        effects = checker.get_function_effects("main")
        assert IO in effects.effects

    def test_call_propagates_inferred_effects(self):
        src = """
        fn greet() {
            print("hello");
        }
        fn main() {
            greet();
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("main")
        assert IO in effects.effects

    def test_transitive_propagation(self):
        src = """
        fn a() { print("a"); }
        fn b() { a(); }
        fn c() { b(); }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        # c -> b -> a -> print (IO)
        assert IO in checker.get_function_effects("c").effects

    def test_undeclared_propagated_effect(self):
        """Caller declared pure but calls effectful function."""
        src = """
        fn greet() -> void ! IO {
            print("hi");
        }
        fn main() -> void {
            greet();
        }
        """
        errors, checker = check_source(src)
        # main has no effect annotation, so no error
        assert len(errors) == 0
        # But effects are tracked
        assert IO in checker.get_function_effects("main").effects

    def test_multiple_calls_combine_effects(self):
        src = """
        fn log_msg() { print("log"); }
        fn save_state() { perform State.set("x", 1); }
        fn both() {
            log_msg();
            save_state();
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("both")
        assert IO in effects.effects
        assert STATE in effects.effects


# ============================================================
# Section 8: Effect Handlers
# ============================================================

class TestEffectHandlers:
    def test_handle_discharges_effect(self):
        src = """
        fn work() {
            handle {
                perform IO.print("hello");
            } with {
                IO.print(msg) -> {
                    resume(0);
                }
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        # IO is handled, so should not be in effects
        assert IO not in effects.effects

    def test_unhandled_effect_remains(self):
        src = """
        fn work() {
            handle {
                perform IO.print("hello");
                perform State.get("x");
            } with {
                IO.print(msg) -> {
                    resume(0);
                }
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        # IO is handled, State is not
        assert IO not in effects.effects
        assert STATE in effects.effects

    def test_handler_body_type_checked(self):
        src = """
        fn work() {
            handle {
                perform IO.print("hello");
            } with {
                IO.print(msg) -> {
                    let x = msg + 1;
                }
            }
        }
        """
        errors, _ = check_source(src)
        # msg is string, msg + 1 should error
        assert any("Cannot apply" in e.message for e in errors)

    def test_handler_params_typed(self):
        """Handler params get types from the effect operation signature."""
        src = """
        fn work() {
            handle {
                perform State.set("key", 42);
            } with {
                State.set(k, v) -> {
                    let s = k + " done";
                    resume(0);
                }
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_unknown_effect_in_handler(self):
        src = """
        fn work() {
            handle {
                let x = 1;
            } with {
                Unknown.op(x) -> {
                    resume(0);
                }
            }
        }
        """
        errors, _ = check_source(src)
        assert any("Unknown effect" in e.message for e in errors)

    def test_unknown_operation_in_handler(self):
        src = """
        fn work() {
            handle {
                let x = 1;
            } with {
                IO.nonexistent(x) -> {
                    resume(0);
                }
            }
        }
        """
        errors, _ = check_source(src)
        assert any("Unknown operation" in e.message for e in errors)

    def test_nested_handlers(self):
        src = """
        fn work() {
            handle {
                handle {
                    perform IO.print("hello");
                    perform State.get("x");
                } with {
                    IO.print(msg) -> {
                        resume(0);
                    }
                }
            } with {
                State.get(key) -> {
                    resume(0);
                }
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert effects.is_pure  # Both effects handled


# ============================================================
# Section 9: Try/Catch and Error Effect
# ============================================================

class TestTryCatch:
    def test_try_catch_basic(self):
        src = """
        fn work() {
            try {
                throw("oops");
            } catch(e) {
                print(e);
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        # Error is handled by try/catch, but print adds IO
        assert ERROR not in effects.effects
        assert IO in effects.effects

    def test_unhandled_throw(self):
        src = """
        fn work() {
            throw("oops");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert ERROR in effects.effects

    def test_catch_binds_error(self):
        src = """
        fn work() {
            try {
                throw("oops");
            } catch(e) {
                let msg = e + " handled";
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_try_without_throw_is_ok(self):
        src = """
        fn work() {
            try {
                let x = 1;
            } catch(e) {
                let y = 2;
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_nested_try_catch(self):
        src = """
        fn work() {
            try {
                try {
                    throw("inner");
                } catch(e1) {
                    throw("rethrow");
                }
            } catch(e2) {
                print(e2);
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert ERROR not in effects.effects
        assert IO in effects.effects


# ============================================================
# Section 10: Custom Effects
# ============================================================

class TestCustomEffects:
    def test_declare_custom_effect(self):
        src = """
        effect Logger {
            log(string) -> void;
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert checker.registry.has_effect("Logger")

    def test_perform_custom_effect(self):
        src = """
        effect Logger {
            log(string) -> void;
        }
        fn work() {
            perform Logger.log("hello");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert Effect("Logger") in effects.effects

    def test_handle_custom_effect(self):
        src = """
        effect Logger {
            log(string) -> void;
        }
        fn work() {
            handle {
                perform Logger.log("hello");
            } with {
                Logger.log(msg) -> {
                    resume(0);
                }
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert Effect("Logger") not in effects.effects

    def test_custom_effect_type_checking(self):
        """Argument types checked against effect operation signature."""
        src = """
        effect Logger {
            log(string) -> void;
        }
        fn work() {
            perform Logger.log(42);
        }
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1
        assert any("expected string" in e.message.lower() for e in errors)

    def test_custom_effect_arity(self):
        src = """
        effect Logger {
            log(string) -> void;
        }
        fn work() {
            perform Logger.log("a", "b");
        }
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1
        assert any("expects 1 args" in e.message for e in errors)

    def test_custom_effect_with_return(self):
        src = """
        effect Config {
            get(string) -> int;
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        op = checker.registry.lookup_operation("Config", "get")
        assert op is not None
        assert op[1] == INT

    def test_multiple_operations(self):
        src = """
        effect Database {
            read(string) -> string;
            write(string, string) -> void;
            delete(string) -> void;
        }
        fn work() {
            perform Database.read("key");
            perform Database.write("key", "value");
            perform Database.delete("key");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert Effect("Database") in effects.effects


# ============================================================
# Section 11: Resume
# ============================================================

class TestResume:
    def test_resume_in_handler(self):
        src = """
        fn work() {
            handle {
                perform IO.print("hello");
            } with {
                IO.print(msg) -> {
                    resume(0);
                }
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_resume_outside_handler(self):
        src = """
        fn work() {
            resume(42);
        }
        """
        errors, _ = check_source(src)
        assert any("outside of effect handler" in e.message for e in errors)


# ============================================================
# Section 12: Effect Rows and Subtyping
# ============================================================

class TestEffectRows:
    def test_pure_is_subset_of_everything(self):
        assert effect_subset(PURE, EffectRow(frozenset({IO})))
        assert effect_subset(PURE, EffectRow(frozenset({IO, STATE, ERROR})))

    def test_same_row_is_subset(self):
        row = EffectRow(frozenset({IO, STATE}))
        assert effect_subset(row, row)

    def test_subset_check(self):
        sub = EffectRow(frozenset({IO}))
        sup = EffectRow(frozenset({IO, STATE}))
        assert effect_subset(sub, sup)

    def test_not_subset(self):
        sub = EffectRow(frozenset({IO, ERROR}))
        sup = EffectRow(frozenset({IO, STATE}))
        assert not effect_subset(sub, sup)

    def test_effect_row_union(self):
        r1 = EffectRow(frozenset({IO}))
        r2 = EffectRow(frozenset({STATE}))
        combined = r1.union(r2)
        assert IO in combined.effects
        assert STATE in combined.effects

    def test_effect_row_without(self):
        r = EffectRow(frozenset({IO, STATE, ERROR}))
        r2 = r.without(STATE)
        assert STATE not in r2.effects
        assert IO in r2.effects
        assert ERROR in r2.effects

    def test_effect_row_add(self):
        r = PURE.add(IO)
        assert IO in r.effects
        assert not r.is_pure

    def test_pure_is_pure(self):
        assert PURE.is_pure

    def test_non_empty_not_pure(self):
        assert not EffectRow(frozenset({IO})).is_pure

    def test_open_row_absorbs_effects(self):
        """An open row (with row variable) accepts any effect."""
        evar = EffectVar(1)
        open_row = EffectRow(frozenset({IO}), evar)
        sub = EffectRow(frozenset({IO, STATE}))
        assert effect_subset(sub, open_row)  # STATE absorbed by row var

    def test_effect_row_repr_pure(self):
        assert repr(PURE) == "Pure"

    def test_effect_row_repr_single(self):
        r = EffectRow(frozenset({IO}))
        assert "IO" in repr(r)

    def test_effect_row_repr_multiple(self):
        r = EffectRow(frozenset({IO, STATE}))
        s = repr(r)
        assert "IO" in s
        assert "State" in s


# ============================================================
# Section 13: Effect Registry
# ============================================================

class TestEffectRegistry:
    def test_builtin_effects(self):
        reg = EffectRegistry()
        assert reg.has_effect("IO")
        assert reg.has_effect("State")
        assert reg.has_effect("Error")
        assert reg.has_effect("Async")

    def test_lookup_operation(self):
        reg = EffectRegistry()
        op = reg.lookup_operation("IO", "print")
        assert op is not None
        assert op[0] == [STRING]  # param types
        assert op[1] == VOID      # return type

    def test_lookup_unknown_effect(self):
        reg = EffectRegistry()
        assert reg.lookup_operation("Unknown", "op") is None

    def test_lookup_unknown_operation(self):
        reg = EffectRegistry()
        assert reg.lookup_operation("IO", "nonexistent") is None

    def test_register_custom(self):
        reg = EffectRegistry()
        reg.register("Logger", [("log", [STRING], VOID)])
        assert reg.has_effect("Logger")
        op = reg.lookup_operation("Logger", "log")
        assert op is not None


# ============================================================
# Section 14: Type Unification
# ============================================================

class TestUnification:
    def test_same_type(self):
        assert unify(INT, INT) == INT

    def test_tvar_binds(self):
        tv = TVar(1)
        result = unify(tv, INT)
        assert result == INT
        assert tv.bound == INT

    def test_tvar_binds_reverse(self):
        tv = TVar(1)
        result = unify(INT, tv)
        assert result == INT

    def test_int_float_promotion(self):
        assert unify(INT, FLOAT) == FLOAT

    def test_incompatible_types(self):
        with pytest.raises(UnificationError):
            unify(INT, STRING)

    def test_function_types(self):
        f1 = TFunc((INT,), INT)
        f2 = TFunc((INT,), INT)
        result = unify(f1, f2)
        assert isinstance(result, TFunc)

    def test_function_arity_mismatch(self):
        f1 = TFunc((INT,), INT)
        f2 = TFunc((INT, INT), INT)
        with pytest.raises(UnificationError):
            unify(f1, f2)

    def test_resolve_chain(self):
        t1 = TVar(1)
        t2 = TVar(2)
        t1.bound = t2
        t2.bound = INT
        assert resolve(t1) == INT

    def test_occurs_check(self):
        tv = TVar(1)
        with pytest.raises(UnificationError):
            unify(tv, TFunc((tv,), INT))


# ============================================================
# Section 15: Effect Unification
# ============================================================

class TestEffectUnification:
    def test_unify_pure_with_pure(self):
        result = unify_effects(PURE, PURE)
        assert result.is_pure

    def test_unify_with_effects(self):
        r1 = EffectRow(frozenset({IO}))
        r2 = EffectRow(frozenset({STATE}))
        result = unify_effects(r1, r2)
        assert IO in result.effects
        assert STATE in result.effects

    def test_unify_none_handling(self):
        result = unify_effects(None, EffectRow(frozenset({IO})))
        assert IO in result.effects

    def test_unify_both_none(self):
        result = unify_effects(None, None)
        assert result.is_pure

    def test_effect_var_resolution(self):
        ev = EffectVar(1)
        ev.bound = EffectRow(frozenset({IO}))
        row = EffectRow(frozenset(), ev)
        result = unify_effects(row, PURE)
        assert IO in result.effects


# ============================================================
# Section 16: Complex Programs
# ============================================================

class TestComplexPrograms:
    def test_effectful_pipeline(self):
        src = """
        fn read_input() -> string ! IO {
            perform IO.read();
            return "data";
        }
        fn process(data: string) -> int {
            return 42;
        }
        fn write_output(result: int) -> void ! IO {
            print(result);
        }
        fn main() {
            let data = read_input();
            let result = process(data);
            write_output(result);
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert checker.get_function_effects("process").is_pure
        assert IO in checker.get_function_effects("main").effects

    def test_state_machine_with_effects(self):
        src = """
        fn init_state() {
            perform State.set("count", 0);
        }
        fn increment() {
            perform State.get("count");
            perform State.set("count", 1);
        }
        fn run() {
            init_state();
            increment();
            increment();
            print("done");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("run")
        assert IO in effects.effects
        assert STATE in effects.effects

    def test_error_handling_chain(self):
        src = """
        fn validate(x) {
            if (x < 0) {
                throw("negative");
            }
            return x;
        }
        fn safe_validate(x) {
            let result = 0;
            try {
                result = validate(x);
            } catch(e) {
                result = 0;
            }
            return result;
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert ERROR in checker.get_function_effects("validate").effects
        # safe_validate catches the error
        assert ERROR not in checker.get_function_effects("safe_validate").effects

    def test_handler_as_dependency_injection(self):
        """Use handlers to inject behavior."""
        src = """
        effect Logger {
            log(string) -> void;
        }
        fn business_logic() {
            perform Logger.log("starting");
            let result = 1 + 2;
            perform Logger.log("done");
            return result;
        }
        fn main() {
            handle {
                business_logic();
            } with {
                Logger.log(msg) -> {
                    print(msg);
                    resume(0);
                }
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        # business_logic has Logger effect
        assert Effect("Logger") in checker.get_function_effects("business_logic").effects
        # main handles Logger but handler uses print (IO)
        main_effects = checker.get_function_effects("main")
        assert Effect("Logger") not in main_effects.effects

    def test_multiple_custom_effects(self):
        src = """
        effect Cache {
            get(string) -> int;
            put(string, int) -> void;
        }
        effect Metrics {
            count(string) -> void;
            timer(string) -> int;
        }
        fn tracked_lookup(key: string) {
            perform Metrics.count("lookup");
            perform Cache.get(key);
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("tracked_lookup")
        assert Effect("Cache") in effects.effects
        assert Effect("Metrics") in effects.effects

    def test_many_functions(self):
        src = """
        fn a() { return 1; }
        fn b() { return a() + 1; }
        fn c() { return b() + 1; }
        fn d() { return c() + 1; }
        fn e() { return d() + 1; }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        for name in "abcde":
            assert checker.get_function_effects(name).is_pure

    def test_mixed_pure_and_effectful(self):
        src = """
        fn pure_add(a, b) { return a + b; }
        fn effectful_add(a, b) {
            print(a);
            return a + b;
        }
        fn main() {
            let x = pure_add(1, 2);
            let y = effectful_add(3, 4);
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert checker.get_function_effects("pure_add").is_pure
        assert IO in checker.get_function_effects("effectful_add").effects
        assert IO in checker.get_function_effects("main").effects


# ============================================================
# Section 17: Error Messages
# ============================================================

class TestErrorMessages:
    def test_format_no_errors(self):
        assert format_errors([]) == "No errors."

    def test_format_one_error(self):
        errs = [EffectError("bad", 5, "type")]
        s = format_errors(errs)
        assert "1 error" in s
        assert "Line 5" in s
        assert "bad" in s

    def test_format_multiple_errors(self):
        errs = [
            EffectError("type mismatch", 1, "type"),
            EffectError("undeclared effect", 3, "effect"),
        ]
        s = format_errors(errs)
        assert "2 error" in s

    def test_error_repr(self):
        e = EffectError("test error", 10, "effect")
        assert "EffectError" in repr(e)
        assert "line 10" in repr(e).lower()

    def test_effect_error_kind(self):
        src = """
        fn work() {
            perform Unknown.op();
        }
        """
        errors, _ = check_source(src)
        assert any(e.kind == "effect" for e in errors)

    def test_type_error_kind(self):
        src = 'let x = 1 + "hello";'
        errors, _ = check_source(src)
        assert any(e.kind == "type" for e in errors)


# ============================================================
# Section 18: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        errors, _ = check_source("")
        assert len(errors) == 0

    def test_single_expression(self):
        errors, _ = check_source("42;")
        assert len(errors) == 0

    def test_nested_if_else(self):
        src = """
        fn choose(a, b, c) {
            if (a) {
                if (b) {
                    return 1;
                } else {
                    return 2;
                }
            } else {
                if (c) {
                    return 3;
                } else {
                    return 4;
                }
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_deeply_nested_blocks(self):
        src = "{ { { { let x = 1; } } } }"
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_multiple_errors_reported(self):
        src = """
        let x = 1 + "a";
        let y = true + 5;
        let z = w;
        """
        errors, _ = check_source(src)
        assert len(errors) >= 3

    def test_perform_with_multiple_args(self):
        src = """
        fn work() {
            perform State.set("key", 42);
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_if_else_chain(self):
        src = """
        fn classify(x) {
            if (x < 0) {
                return -1;
            } else if (x == 0) {
                return 0;
            } else {
                return 1;
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_while_with_effects(self):
        src = """
        fn log_loop() {
            let i = 0;
            while (i < 5) {
                print(i);
                i = i + 1;
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert IO in checker.get_function_effects("log_loop").effects

    def test_function_defined_after_use(self):
        """Forward reference -- function used before defined."""
        src = """
        fn main() {
            let x = helper();
        }
        fn helper() {
            return 42;
        }
        """
        errors, _ = check_source(src)
        # This will error because helper isn't defined yet at call site
        assert any("Undefined" in e.message for e in errors)

    def test_modulo_operator(self):
        errors, checker = check_source("let x = 10 % 3;")
        assert len(errors) == 0
        assert isinstance(resolve(checker.env.lookup("x")), TInt)

    def test_division(self):
        errors, checker = check_source("let x = 10 / 3;")
        assert len(errors) == 0

    def test_string_comparison_error(self):
        errors, _ = check_source('let x = "a" < 5;')
        assert len(errors) >= 1

    def test_boolean_arithmetic_error(self):
        errors, _ = check_source("let x = true + false;")
        assert len(errors) >= 1

    def test_negate_string_error(self):
        errors, _ = check_source('let x = -"hello";')
        assert len(errors) >= 1

    def test_not_int_error(self):
        errors, _ = check_source("let x = not 5;")
        assert len(errors) >= 1

    def test_logical_with_non_bool(self):
        errors, _ = check_source("let x = 1 and 2;")
        assert len(errors) >= 1

    def test_effect_annotation_with_param_types(self):
        src = """
        fn fetch(url: string) -> string ! IO, Error {
            perform IO.read();
            return "data";
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0


# ============================================================
# Section 19: Type Representations
# ============================================================

class TestTypeRepresentations:
    def test_int_repr(self):
        assert repr(INT) == "int"

    def test_float_repr(self):
        assert repr(FLOAT) == "float"

    def test_string_repr(self):
        assert repr(STRING) == "string"

    def test_bool_repr(self):
        assert repr(BOOL) == "bool"

    def test_void_repr(self):
        assert repr(VOID) == "void"

    def test_func_repr(self):
        f = TFunc((INT, INT), INT)
        assert "fn(int, int) -> int" in repr(f)

    def test_func_with_effects_repr(self):
        f = TFunc((INT,), VOID, EffectRow(frozenset({IO})))
        s = repr(f)
        assert "IO" in s

    def test_tvar_repr_unbound(self):
        tv = TVar(42)
        assert "?T42" in repr(tv)

    def test_tvar_repr_bound(self):
        tv = TVar(1)
        tv.bound = INT
        assert repr(tv) == "int"

    def test_effect_repr(self):
        assert repr(IO) == "IO"
        assert repr(STATE) == "State"

    def test_effect_ordering(self):
        assert IO < STATE  # "IO" < "State" alphabetically

    def test_effect_var_repr_unbound(self):
        ev = EffectVar(1)
        assert "?e1" in repr(ev)

    def test_effect_var_repr_bound(self):
        ev = EffectVar(1)
        ev.bound = EffectRow(frozenset({IO}))
        assert "IO" in repr(ev)


# ============================================================
# Section 20: Full Integration
# ============================================================

class TestIntegration:
    def test_complete_program(self):
        """A complete program using multiple features."""
        src = """
        effect Logger {
            log(string) -> void;
            level() -> int;
        }

        fn pure_compute(x: int, y: int) -> int {
            return x * y + x - y;
        }

        fn validate(x: int) -> int ! Error {
            if (x < 0) {
                throw("negative input");
            }
            return x;
        }

        fn logged_compute(x: int, y: int) -> int ! Logger {
            perform Logger.log("computing");
            let result = pure_compute(x, y);
            perform Logger.log("done");
            return result;
        }

        fn safe_compute(x: int, y: int) -> int {
            let result = 0;
            try {
                let vx = validate(x);
                let vy = validate(y);
                result = pure_compute(vx, vy);
            } catch(e) {
                result = -1;
            }
            return result;
        }

        fn main() {
            handle {
                let r = logged_compute(3, 4);
                print(r);
            } with {
                Logger.log(msg) -> {
                    print(msg);
                    resume(0);
                }
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0

        # Verify effect inference
        assert checker.get_function_effects("pure_compute").is_pure
        assert ERROR in checker.get_function_effects("validate").effects
        assert Effect("Logger") in checker.get_function_effects("logged_compute").effects
        assert ERROR not in checker.get_function_effects("safe_compute").effects

    def test_effect_handler_chain(self):
        """Chain of handlers progressively discharging effects."""
        src = """
        effect A {
            do_a(int) -> void;
        }
        effect B {
            do_b(int) -> void;
        }
        effect C {
            do_c(int) -> void;
        }

        fn work() {
            perform A.do_a(1);
            perform B.do_b(2);
            perform C.do_c(3);
        }

        fn handle_a() {
            handle {
                work();
            } with {
                A.do_a(x) -> {
                    resume(0);
                }
            }
        }

        fn handle_ab() {
            handle {
                handle_a();
            } with {
                B.do_b(x) -> {
                    resume(0);
                }
            }
        }

        fn handle_all() {
            handle {
                handle_ab();
            } with {
                C.do_c(x) -> {
                    resume(0);
                }
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        work_effs = checker.get_function_effects("work")
        assert Effect("A") in work_effs.effects
        assert Effect("B") in work_effs.effects
        assert Effect("C") in work_effs.effects

    def test_effect_annotation_mismatch_precise(self):
        src = """
        fn work() -> void ! IO {
            perform State.set("x", 1);
            print("hi");
        }
        """
        errors, _ = check_source(src)
        effect_errors = [e for e in errors if e.kind == "effect"]
        assert len(effect_errors) == 1
        assert "State" in effect_errors[0].message

    def test_handler_wrong_param_count(self):
        src = """
        fn work() {
            handle {
                perform State.set("key", 42);
            } with {
                State.set(k) -> {
                    resume(0);
                }
            }
        }
        """
        errors, _ = check_source(src)
        assert any("expects 2 params" in e.message for e in errors)

    def test_empty_handle_with(self):
        src = """
        fn work() {
            handle {
                let x = 1;
            } with {
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_perform_in_conditional(self):
        src = """
        fn work(flag) {
            if (flag) {
                perform IO.print("yes");
            } else {
                perform State.get("no");
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert IO in effects.effects
        assert STATE in effects.effects

    def test_zero_arg_function(self):
        src = """
        fn get_value() -> int {
            return 42;
        }
        let x = get_value();
        """
        errors, checker = check_source(src)
        assert len(errors) == 0

    def test_effect_declaration_multiple_params(self):
        src = """
        effect Http {
            request(string, string) -> string;
        }
        fn fetch() {
            perform Http.request("GET", "/api");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert Effect("Http") in checker.get_function_effects("fetch").effects

    def test_handle_then_perform_again(self):
        """After handling, new perform of same effect is unhandled."""
        src = """
        fn work() {
            handle {
                perform IO.print("handled");
            } with {
                IO.print(msg) -> {
                    resume(0);
                }
            }
            perform IO.print("unhandled");
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        effects = checker.get_function_effects("work")
        assert IO in effects.effects

    def test_deeply_nested_effects(self):
        src = """
        fn a() { print("a"); }
        fn b() { a(); perform State.get("x"); }
        fn c() { b(); throw("err"); }
        fn d() {
            try {
                c();
            } catch(e) {
                print(e);
            }
        }
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        # d catches Error, but IO and State propagate
        d_effects = checker.get_function_effects("d")
        assert IO in d_effects.effects
        assert STATE in d_effects.effects
        assert ERROR not in d_effects.effects


# ============================================================
# Section 21: API Tests
# ============================================================

class TestAPI:
    def test_check_source_returns_tuple(self):
        result = check_source("let x = 1;")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_program(self):
        prog = parse_source("let x = 42;")
        errors, checker = check_program(prog)
        assert len(errors) == 0

    def test_parse_source(self):
        prog = parse_source("let x = 1;")
        assert isinstance(prog, Program)

    def test_get_function_type(self):
        _, checker = check_source("fn add(a, b) { return a + b; }")
        ft = checker.get_function_type("add")
        assert isinstance(ft, TFunc)

    def test_get_function_type_undefined(self):
        _, checker = check_source("let x = 1;")
        assert checker.get_function_type("nonexistent") is None

    def test_get_function_effects_undefined(self):
        _, checker = check_source("let x = 1;")
        effects = checker.get_function_effects("nonexistent")
        assert effects.is_pure

    def test_checker_fresh_tvar(self):
        checker = EffectChecker()
        t1 = checker.fresh_tvar()
        t2 = checker.fresh_tvar()
        assert t1.id != t2.id

    def test_checker_fresh_evar(self):
        checker = EffectChecker()
        e1 = checker.fresh_evar()
        e2 = checker.fresh_evar()
        assert e1.id != e2.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
