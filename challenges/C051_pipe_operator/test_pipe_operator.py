"""Tests for C051: Pipe Operator"""
import pytest
from pipe_operator import run, execute, lex, parse, TokenType, CallExpr, Var

# ============================================================
# Section 1: Basic Pipe
# ============================================================

class TestBasicPipe:
    def test_pipe_to_function(self):
        r, o = run('fn double(x) { return x * 2; } let r = 5 |> double; print r;')
        assert o == ['10']

    def test_pipe_to_builtin(self):
        r, o = run('let a = [1, 2, 3]; let r = a |> len; print r;')
        assert o == ['3']

    def test_pipe_to_lambda(self):
        r, o = run('let r = 10 |> fn(x) { return x + 1; }; print r;')
        assert o == ['11']

    def test_pipe_preserves_value(self):
        r, o = run('fn identity(x) { return x; } let r = 42 |> identity; print r;')
        assert o == ['42']

    def test_pipe_with_string(self):
        r, o = run('let r = "hello" |> len; print r;')
        assert o == ['5']

    def test_pipe_with_zero(self):
        r, o = run('fn inc(x) { return x + 1; } let r = 0 |> inc; print r;')
        assert o == ['1']

    def test_pipe_with_bool(self):
        r, o = run('fn negate(x) { return not x; } let r = true |> negate; print r;')
        assert o == ['false']

    def test_pipe_with_none(self):
        r, o = run('''
            fn is_nil(x) { return x == false; }
            let r = false |> is_nil;
            print r;
        ''')
        assert o == ['true']


# ============================================================
# Section 2: Pipe Chaining
# ============================================================

class TestPipeChaining:
    def test_two_pipes(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            fn inc(x) { return x + 1; }
            let r = 5 |> double |> inc;
            print r;
        ''')
        assert o == ['11']  # double(5) = 10, inc(10) = 11

    def test_three_pipes(self):
        r, o = run('''
            fn a(x) { return x + 1; }
            fn b(x) { return x * 2; }
            fn c(x) { return x - 3; }
            let r = 10 |> a |> b |> c;
            print r;
        ''')
        assert o == ['19']  # a(10)=11, b(11)=22, c(22)=19

    def test_four_pipes(self):
        r, o = run('''
            fn a(x) { return x + 1; }
            fn b(x) { return x * 2; }
            fn c(x) { return x + 3; }
            fn d(x) { return x * 10; }
            let r = 1 |> a |> b |> c |> d;
            print r;
        ''')
        assert o == ['70']  # a(1)=2, b(2)=4, c(4)=7, d(7)=70

    def test_chain_with_print(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            fn triple(x) { return x * 3; }
            print 5 |> double |> triple;
        ''')
        assert o == ['30']  # double(5)=10, triple(10)=30


# ============================================================
# Section 3: Pipe with Arguments (Prepend)
# ============================================================

class TestPipeWithArgs:
    def test_pipe_with_one_extra_arg(self):
        r, o = run('''
            fn add(a, b) { return a + b; }
            let r = 5 |> add(3);
            print r;
        ''')
        assert o == ['8']  # add(5, 3)

    def test_pipe_with_two_extra_args(self):
        r, o = run('''
            fn clamp(val, lo, hi) {
                if (val < lo) { return lo; }
                if (val > hi) { return hi; }
                return val;
            }
            print 15 |> clamp(0, 10);
        ''')
        assert o == ['10']  # clamp(15, 0, 10) = 10

    def test_pipe_chain_with_args(self):
        r, o = run('''
            fn add(a, b) { return a + b; }
            fn mul(a, b) { return a * b; }
            let r = 5 |> add(3) |> mul(2);
            print r;
        ''')
        assert o == ['16']  # add(5,3)=8, mul(8,2)=16

    def test_pipe_mix_with_and_without_args(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            fn add(a, b) { return a + b; }
            fn negate(x) { return 0 - x; }
            let r = 3 |> double |> add(10) |> negate;
            print r;
        ''')
        assert o == ['-16']  # double(3)=6, add(6,10)=16, negate(16)=-16

    def test_pipe_with_string_arg(self):
        r, o = run('''
            fn greet(name, greeting) { return f"${greeting} ${name}"; }
            let r = "world" |> greet("hello");
            print r;
        ''')
        assert o == ['hello world']


# ============================================================
# Section 4: Pipe with Expressions
# ============================================================

class TestPipeWithExpressions:
    def test_pipe_from_expression(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            let r = (3 + 4) |> double;
            print r;
        ''')
        assert o == ['14']  # double(7) = 14

    def test_pipe_from_arithmetic(self):
        r, o = run('''
            fn abs_val(x) {
                if (x < 0) { return 0 - x; }
                return x;
            }
            let r = 3 - 10 |> abs_val;
            print r;
        ''')
        assert o == ['7']  # (3-10)=-7... wait, pipe binds lower than subtraction
        # Actually: 3 - 10 = -7, then |> abs_val = abs_val(-7) = 7

    def test_pipe_from_function_result(self):
        r, o = run('''
            fn get_val() { return 42; }
            fn double(x) { return x * 2; }
            let r = get_val() |> double;
            print r;
        ''')
        assert o == ['84']

    def test_pipe_from_array_element(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            let a = [10, 20, 30];
            let r = a[1] |> double;
            print r;
        ''')
        assert o == ['40']

    def test_pipe_from_hash_field(self):
        r, o = run('''
            fn inc(x) { return x + 1; }
            let h = {age: 25};
            let r = h.age |> inc;
            print r;
        ''')
        assert o == ['26']

    def test_pipe_from_conditional(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            let x = 5;
            let val = if (x > 3) x else 0;
            let r = val |> double;
            print r;
        ''')
        assert o == ['10']


# ============================================================
# Section 5: Pipe with Closures and Lambdas
# ============================================================

class TestPipeWithClosures:
    def test_pipe_to_closure(self):
        r, o = run('''
            fn make_adder(n) {
                return fn(x) { return x + n; };
            }
            let add5 = make_adder(5);
            let r = 10 |> add5;
            print r;
        ''')
        assert o == ['15']

    def test_pipe_to_inline_lambda(self):
        r, o = run('''
            let r = 7 |> fn(x) { return x * x; };
            print r;
        ''')
        assert o == ['49']

    def test_pipe_chain_lambdas(self):
        r, o = run('''
            let r = 5
                |> fn(x) { return x * 2; }
                |> fn(x) { return x + 1; };
            print r;
        ''')
        assert o == ['11']  # 5*2=10, 10+1=11

    def test_pipe_to_method_like_closure(self):
        r, o = run('''
            let obj = {
                multiplier: 3,
                apply: fn(x) { return x * 3; }
            };
            let r = 4 |> obj.apply;
            print r;
        ''')
        assert o == ['12']


# ============================================================
# Section 6: Pipe with Builtins
# ============================================================

class TestPipeWithBuiltins:
    def test_pipe_to_len(self):
        r, o = run('let r = [1, 2, 3, 4] |> len; print r;')
        assert o == ['4']

    def test_pipe_to_type(self):
        r, o = run('let r = 42 |> type; print r;')
        assert o == ['int']

    def test_pipe_to_string(self):
        r, o = run('let r = 42 |> string; print r;')
        assert o == ['42']

    def test_pipe_to_push(self):
        r, o = run('''
            let a = [1, 2];
            a |> push(3);
            print a;
        ''')
        assert o == ['[1, 2, 3]']  # push(a, 3)

    def test_pipe_chain_builtins(self):
        r, o = run('''
            let r = [1, 2, 3] |> len |> string;
            print r;
        ''')
        assert o == ['3']

    def test_pipe_to_keys(self):
        r, o = run('''
            let h = {a: 1, b: 2, c: 3};
            let r = h |> keys |> len;
            print r;
        ''')
        assert o == ['3']


# ============================================================
# Section 7: Pipe with Arrays and Hashes
# ============================================================

class TestPipeWithCollections:
    def test_pipe_array_literal(self):
        r, o = run('''
            fn first(arr) { return arr[0]; }
            let r = [10, 20, 30] |> first;
            print r;
        ''')
        assert o == ['10']

    def test_pipe_hash_literal(self):
        r, o = run('''
            fn get_name(obj) { return obj.name; }
            let r = {name: "alice"} |> get_name;
            print r;
        ''')
        assert o == ['alice']

    def test_pipe_transform_array(self):
        r, o = run('''
            fn double_all(arr) {
                let result = [];
                for (x in arr) { push(result, x * 2); }
                return result;
            }
            let doubled = [1, 2, 3] |> double_all;
            print doubled;
        ''')
        assert o == ['[2, 4, 6]']

    def test_pipe_with_spread(self):
        r, o = run('''
            fn sum(arr) {
                let total = 0;
                for (x in arr) { total = total + x; }
                return total;
            }
            let a = [1, 2];
            let b = [3, 4];
            let r = [...a, ...b] |> sum;
            print r;
        ''')
        assert o == ['10']


# ============================================================
# Section 8: Pipe Precedence
# ============================================================

class TestPipePrecedence:
    def test_pipe_lower_than_arithmetic(self):
        """Pipe binds lower than +, so 3 + 4 |> fn means (3+4) |> fn"""
        r, o = run('''
            fn double(x) { return x * 2; }
            let r = 3 + 4 |> double;
            print r;
        ''')
        assert o == ['14']

    def test_pipe_lower_than_comparison(self):
        """3 > 2 |> fn means (3>2) |> fn"""
        r, o = run('''
            fn to_num(x) { if (x) { return 1; } return 0; }
            let r = 3 > 2 |> to_num;
            print r;
        ''')
        assert o == ['1']

    def test_pipe_lower_than_logic(self):
        """true and false |> fn means (true and false) |> fn"""
        r, o = run('''
            fn to_str(x) { if (x) { return "yes"; } return "no"; }
            let r = true and false |> to_str;
            print r;
        ''')
        assert o == ['no']

    def test_pipe_in_assignment(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            let x = 5 |> double;
            print x;
        ''')
        assert o == ['10']

    def test_pipe_in_return(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            fn process(x) { return x |> double; }
            let r = process(7);
            print r;
        ''')
        assert o == ['14']

    def test_pipe_in_condition(self):
        r, o = run('''
            fn is_even(x) { return x % 2 == 0; }
            let x = 4;
            if (x |> is_even) { print "even"; } else { print "odd"; }
        ''')
        assert o == ['even']


# ============================================================
# Section 9: Pipe with Generators
# ============================================================

class TestPipeWithGenerators:
    def test_pipe_to_next(self):
        r, o = run('''
            fn* counter() {
                yield 1;
                yield 2;
                yield 3;
            }
            let g = counter();
            let r = g |> next;
            print r;
        ''')
        assert o == ['1']

    def test_pipe_generator_result(self):
        r, o = run('''
            fn* counter() {
                yield 10;
                yield 20;
                yield 30;
            }
            let g = counter();
            let a = g |> next;
            let b = g |> next;
            print a;
            print b;
        ''')
        assert o == ['10', '20']


# ============================================================
# Section 10: Pipe with Error Handling
# ============================================================

class TestPipeWithErrors:
    def test_pipe_in_try_catch(self):
        r, o = run('''
            fn fail(x) { throw "error"; }
            try {
                let r = 5 |> fail;
            } catch (e) {
                print e;
            }
        ''')
        assert o == ['error']

    def test_pipe_error_propagation(self):
        r, o = run('''
            fn step1(x) { return x + 1; }
            fn step2(x) { throw f"bad: ${x}"; }
            fn step3(x) { return x * 2; }
            try {
                let r = 5 |> step1 |> step2 |> step3;
            } catch (e) {
                print e;
            }
        ''')
        assert o == ['bad: 6']  # step1(5)=6, step2(6) throws


# ============================================================
# Section 11: Pipe with Destructuring
# ============================================================

class TestPipeWithDestructuring:
    def test_pipe_result_destructured(self):
        r, o = run('''
            fn get_pair() { return [10, 20]; }
            fn sum_pair(arr) { return arr[0] + arr[1]; }
            let r = get_pair() |> sum_pair;
            print r;
        ''')
        assert o == ['30']

    def test_pipe_into_destructure_let(self):
        r, o = run('''
            fn swap(arr) { return [arr[1], arr[0]]; }
            let result = [1, 2] |> swap;
            let [a, b] = result;
            print a;
            print b;
        ''')
        assert o == ['2', '1']


# ============================================================
# Section 12: Pipe with Modules
# ============================================================

class TestPipeWithModules:
    def test_pipe_with_imported_function(self):
        from pipe_operator import ModuleRegistry
        reg = ModuleRegistry()
        reg.register("math", '''
            export fn double(x) { return x * 2; }
            export fn square(x) { return x * x; }
        ''')
        r, o = run('''
            import { double, square } from "math";
            let r = 3 |> double |> square;
            print r;
        ''', registry=reg)
        assert o == ['36']  # double(3)=6, square(6)=36


# ============================================================
# Section 13: Complex Pipe Patterns
# ============================================================

class TestComplexPipe:
    def test_pipeline_data_processing(self):
        """Realistic data processing pipeline using simple functions"""
        r, o = run('''
            fn double(x) { return x * 2; }
            fn add10(x) { return x + 10; }
            fn negate(x) { return 0 - x; }
            let r = 5 |> double |> add10 |> negate;
            print r;
        ''')
        assert o == ['-20']  # double(5)=10, add10(10)=20, negate(20)=-20

    def test_pipe_builder_pattern(self):
        """Builder-like pattern with pipe using intermediate vars"""
        r, o = run('''
            fn with_name(config, name) {
                config.name = name;
                return config;
            }
            fn with_age(config, age) {
                config.age = age;
                return config;
            }
            fn build(config) {
                return f"${config.name} is ${config.age}";
            }
            let c = {} |> with_name("alice");
            let c2 = c |> with_age(30);
            let r = c2 |> build;
            print r;
        ''')
        assert o == ['alice is 30']

    def test_nested_pipes(self):
        """Pipe inside pipe argument"""
        r, o = run('''
            fn double(x) { return x * 2; }
            fn add(a, b) { return a + b; }
            let r = 3 |> add(4 |> double);
            print r;
        ''')
        assert o == ['11']  # add(3, double(4)) = add(3, 8) = 11

    def test_pipe_with_spread_args(self):
        r, o = run('''
            fn sum3(a, b, c) { return a + b + c; }
            let args = [20, 30];
            let r = 10 |> sum3(...args);
            print r;
        ''')
        assert o == ['60']  # sum3(10, 20, 30) = 60

    def test_pipe_multiline(self):
        """Pipe across multiple lines"""
        r, o = run('''
            fn a(x) { return x + 1; }
            fn b(x) { return x * 2; }
            fn c(x) { return x - 3; }
            let r = 10
                |> a
                |> b
                |> c;
            print r;
        ''')
        assert o == ['19']


# ============================================================
# Section 14: Lexer Tests
# ============================================================

class TestLexer:
    def test_pipe_token(self):
        tokens = lex("a |> b")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.PIPE in types

    def test_pipe_not_confused_with_or(self):
        tokens = lex("a || b |> c")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.IDENT, TokenType.OR, TokenType.IDENT, TokenType.PIPE, TokenType.IDENT]

    def test_multiple_pipes_lex(self):
        tokens = lex("a |> b |> c")
        pipe_count = sum(1 for t in tokens if t.type == TokenType.PIPE)
        assert pipe_count == 2

    def test_pipe_with_parens(self):
        tokens = lex("a |> b(x)")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.IDENT, TokenType.PIPE, TokenType.IDENT,
                         TokenType.LPAREN, TokenType.IDENT, TokenType.RPAREN]


# ============================================================
# Section 15: Parser Desugaring Tests
# ============================================================

class TestParserDesugaring:
    def test_simple_pipe_desugars_to_call(self):
        prog = parse("a |> b;")
        stmt = prog.stmts[0]
        assert isinstance(stmt, CallExpr)
        assert isinstance(stmt.callee, Var) and stmt.callee.name == 'b'
        assert len(stmt.args) == 1
        assert isinstance(stmt.args[0], Var) and stmt.args[0].name == 'a'

    def test_pipe_with_args_prepends(self):
        prog = parse("a |> b(x);")
        stmt = prog.stmts[0]
        assert isinstance(stmt, CallExpr)
        assert isinstance(stmt.callee, Var) and stmt.callee.name == 'b'
        assert len(stmt.args) == 2
        assert isinstance(stmt.args[0], Var) and stmt.args[0].name == 'a'
        assert isinstance(stmt.args[1], Var) and stmt.args[1].name == 'x'

    def test_chain_desugars_to_nested_calls(self):
        prog = parse("a |> b |> c;")
        stmt = prog.stmts[0]
        # c(b(a))
        assert isinstance(stmt, CallExpr)
        assert isinstance(stmt.callee, Var) and stmt.callee.name == 'c'
        assert len(stmt.args) == 1
        inner = stmt.args[0]
        assert isinstance(inner, CallExpr)
        assert isinstance(inner.callee, Var) and inner.callee.name == 'b'
        assert len(inner.args) == 1
        assert isinstance(inner.args[0], Var) and inner.args[0].name == 'a'


# ============================================================
# Section 16: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_pipe_with_negative_number(self):
        r, o = run('''
            fn abs_val(x) { if (x < 0) { return 0 - x; } return x; }
            let r = -5 |> abs_val;
            print r;
        ''')
        assert o == ['5']

    def test_pipe_result_as_condition(self):
        r, o = run('''
            fn is_positive(x) { return x > 0; }
            let check = -3 |> is_positive;
            if (check) { print "pos"; } else { print "neg"; }
        ''')
        assert o == ['neg']

    def test_pipe_with_recursive_function(self):
        r, o = run('''
            fn factorial(n) {
                if (n <= 1) { return 1; }
                return n * factorial(n - 1);
            }
            let r = 5 |> factorial;
            print r;
        ''')
        assert o == ['120']

    def test_pipe_in_loop(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            for (i in [1, 2, 3]) {
                print i |> double;
            }
        ''')
        assert o == ['2', '4', '6']

    def test_pipe_stored_in_variable(self):
        r, o = run('''
            fn inc(x) { return x + 1; }
            let results = [];
            for (i in [1, 2, 3]) {
                push(results, i |> inc);
            }
            print results;
        ''')
        assert o == ['[2, 3, 4]']

    def test_pipe_with_float(self):
        r, o = run('''
            fn round_down(x) { return x - (x % 1); }
            let r = 3.7 |> round_down;
            print r;
        ''')
        assert o == ['3.0']

    def test_pipe_many_args(self):
        r, o = run('''
            fn combine(a, b, c, d) { return a + b + c + d; }
            let r = 1 |> combine(2, 3, 4);
            print r;
        ''')
        assert o == ['10']  # combine(1, 2, 3, 4) = 10


# ============================================================
# Section 17: Pipe with All C050 Features
# ============================================================

class TestPipeWithC050Features:
    def test_pipe_with_spread_in_array(self):
        r, o = run('''
            fn sum(arr) {
                let total = 0;
                for (x in arr) { total = total + x; }
                return total;
            }
            let a = [1, 2];
            let b = [3, 4];
            let r = [...a, ...b, 5] |> sum;
            print r;
        ''')
        assert o == ['15']

    def test_pipe_with_fstring(self):
        r, o = run('''
            fn shout(s) { return f"${s}!!!"; }
            let r = "hello" |> shout;
            print r;
        ''')
        assert o == ['hello!!!']

    def test_pipe_with_destructuring_result(self):
        r, o = run('''
            fn get_record() { return {x: 10, y: 20}; }
            fn sum_xy(obj) { return obj.x + obj.y; }
            let r = get_record() |> sum_xy;
            print r;
        ''')
        assert o == ['30']

    def test_pipe_with_hash_access(self):
        r, o = run('''
            fn get_x(h) { return h.x; }
            fn double(n) { return n * 2; }
            let r = {x: 5, y: 10} |> get_x |> double;
            print r;
        ''')
        assert o == ['10']
