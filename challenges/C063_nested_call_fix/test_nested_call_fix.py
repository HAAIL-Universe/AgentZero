"""
Tests for C063: Nested Call Stack Fix
Challenge C063 -- AgentZero Session 064

Tests the fix for expression-statement stack corruption.
Root cause: Assign, IndexAssign, DotAssign, and CallExpr left values
on the VM stack when used as statements, corrupting nested calls.
"""

import pytest
from nested_call_fix import run, execute, make_default_registry


def r(src, **kw):
    reg = make_default_registry(**kw)
    return run(src, registry=reg)


def val(src, **kw):
    return r(src, **kw)[0]


def out(src, **kw):
    return r(src, **kw)[1]


# ============================================================
# Section 1: Core bug reproduction -- nested method calls as args
# ============================================================

class TestCoreBugFix:
    """The original bug: method calls with property mutation as arguments."""

    def test_method_with_property_set_as_arg(self):
        assert val('''
class Foo {
  init() { this.x = 0; }
  inc() { this.x = this.x + 1; return this.x; }
}
fn identity(a) { return a; }
let f = Foo();
identity(f.inc());
''') == 1

    def test_method_with_two_property_sets_as_arg(self):
        assert val('''
class Foo {
  init() { this.x = 0; this.y = 0; }
  inc() { this.x = this.x + 1; this.y = this.y + 2; return this.x + this.y; }
}
fn identity(a) { return a; }
let f = Foo();
identity(f.inc());
''') == 3

    def test_assertEqual_with_method_call(self):
        """The exact pattern from session 063."""
        assert out('''
class Stack {
  init() { this.items = []; this.size = 0; }
  push(val) { this.items.push(val); this.size = this.size + 1; }
  pop() { this.size = this.size - 1; return this.items.pop(); }
}
fn assertEqual(a, b) { if (a != b) { throw "FAIL"; } }
let s = Stack();
s.push(1); s.push(2); s.push(3);
assertEqual(s.pop(), 3);
assertEqual(s.pop(), 2);
assertEqual(s.pop(), 1);
print "ok";
''') == ['ok']

    def test_stdlib_stack_nested_call(self):
        """Test with actual stdlib Stack."""
        assert out('''
import "collections";
fn assertEqual(a, b) { if (a != b) { throw "FAIL: " + string(a) + " != " + string(b); } }
let s = Stack();
s.push(10); s.push(20); s.push(30);
assertEqual(s.pop(), 30);
assertEqual(s.pop(), 20);
assertEqual(s.pop(), 10);
print "ok";
''') == ['ok']

    def test_multiple_method_calls_as_args(self):
        assert val('''
class Counter {
  init() { this.n = 0; }
  next() { this.n = this.n + 1; return this.n; }
}
fn add(a, b) { return a + b; }
let c = Counter();
add(c.next(), c.next());
''') == 3


# ============================================================
# Section 2: Simple assignment as expression-statement
# ============================================================

class TestAssignmentStatementFix:
    """Assignment statements that leaked values."""

    def test_assign_in_function_body(self):
        assert val('''
fn foo() { let x = 0; x = x + 1; return x; }
fn identity(a) { return a; }
identity(foo());
''') == 1

    def test_multiple_assigns_in_function(self):
        assert val('''
fn foo() { let x = 0; let y = 0; x = 1; y = 2; return x + y; }
fn identity(a) { return a; }
identity(foo());
''') == 3

    def test_assign_does_not_corrupt_two_arg_call(self):
        assert val('''
fn bar() { let x = 0; x = x + 5; return x; }
fn add(a, b) { return a + b; }
add(bar(), 10);
''') == 15

    def test_chained_assigns_in_function(self):
        assert val('''
fn foo() { let x = 0; x = 1; x = 2; x = 3; return x; }
fn identity(a) { return a; }
identity(foo());
''') == 3


# ============================================================
# Section 3: CallExpr as statement (bare function calls)
# ============================================================

class TestCallExprStatementFix:
    """Bare function calls used as statements that leaked return values."""

    def test_bare_call_in_function(self):
        assert val('''
fn sideEffect() { return 999; }
fn foo() { sideEffect(); return 42; }
fn identity(a) { return a; }
identity(foo());
''') == 42

    def test_multiple_bare_calls_in_function(self):
        assert val('''
fn a() { return 1; }
fn b() { return 2; }
fn foo() { a(); b(); return 3; }
fn identity(a) { return a; }
identity(foo());
''') == 3

    def test_method_call_as_statement(self):
        """s.push(val) returns this -- must not leak."""
        assert val('''
class Stack {
  init() { this.items = []; }
  push(val) { this.items.push(val); return this; }
  peek() { return this.items[this.items.length - 1]; }
}
fn identity(a) { return a; }
let s = Stack();
s.push(1);
s.push(2);
identity(s.peek());
''') == 2

    def test_array_method_as_statement(self):
        assert val('''
fn foo() { let a = []; a.push(1); a.push(2); return a.length; }
fn identity(a) { return a; }
identity(foo());
''') == 2


# ============================================================
# Section 4: DotAssign (this.x = val) as statement
# ============================================================

class TestDotAssignStatementFix:
    """DotAssign (property assignment) that leaked values."""

    def test_dot_assign_in_init(self):
        assert val('''
class Foo {
  init() { this.a = 1; this.b = 2; this.c = 3; }
  sum() { return this.a + this.b + this.c; }
}
fn identity(a) { return a; }
let f = Foo();
identity(f.sum());
''') == 6

    def test_dot_assign_in_method(self):
        assert val('''
class Foo {
  init() { this.x = 0; }
  set(v) { this.x = v; }
  get() { return this.x; }
}
fn identity(a) { return a; }
let f = Foo();
f.set(42);
identity(f.get());
''') == 42

    def test_multiple_dot_assigns_in_method(self):
        assert val('''
class Point {
  init(x, y) { this.x = x; this.y = y; }
  translate(dx, dy) { this.x = this.x + dx; this.y = this.y + dy; return this; }
  sum() { return this.x + this.y; }
}
fn identity(a) { return a; }
let p = Point(1, 2);
p.translate(10, 20);
identity(p.sum());
''') == 33


# ============================================================
# Section 5: IndexAssign (arr[i] = val) as statement
# ============================================================

class TestIndexAssignStatementFix:
    """IndexAssign that leaked values."""

    def test_array_index_assign_in_function(self):
        assert val('''
fn foo() { let a = [0, 0, 0]; a[0] = 1; a[1] = 2; a[2] = 3; return a; }
fn identity(a) { return a; }
identity(foo());
''') == [1, 2, 3]

    def test_hash_index_assign_in_function(self):
        assert val('''
fn foo() { let h = {}; h["x"] = 10; h["y"] = 20; return h; }
fn identity(a) { return a; }
identity(foo());
''') == {'x': 10, 'y': 20}


# ============================================================
# Section 6: Nested calls (multiple levels)
# ============================================================

class TestNestedCalls:
    """Multiple levels of nesting with expression-statements."""

    def test_double_nested_call(self):
        assert val('''
fn a(x) { return x + 1; }
fn b() { let x = 0; x = 5; return x; }
fn c(x) { return a(x); }
c(b());
''') == 6

    def test_triple_nested_method_call(self):
        assert out('''
class Counter {
  init() { this.n = 0; }
  inc() { this.n = this.n + 1; return this.n; }
}
fn identity(x) { return x; }
fn check(a, b) { if (a != b) { throw "fail"; } print "pass"; }
let c = Counter();
check(identity(c.inc()), 1);
check(identity(c.inc()), 2);
''') == ['pass', 'pass']

    def test_fn_call_results_as_both_args(self):
        assert val('''
fn inc(x) { let y = x; y = y + 1; return y; }
fn add(a, b) { return a + b; }
add(inc(1), inc(2));
''') == 5

    def test_method_results_as_both_args(self):
        assert val('''
class Box {
  init(v) { this.v = v; }
  get() { return this.v; }
  set(v) { this.v = v; return this; }
}
fn add(a, b) { return a + b; }
let a = Box(10);
let b = Box(20);
add(a.get(), b.get());
''') == 30


# ============================================================
# Section 7: Interaction with control flow
# ============================================================

class TestControlFlowInteraction:
    """Expression-statements inside if/while/for bodies."""

    def test_assign_in_if_body(self):
        assert val('''
fn foo(flag) {
  let x = 0;
  if (flag) { x = 10; }
  else { x = 20; }
  return x;
}
fn identity(a) { return a; }
identity(foo(true));
''') == 10

    def test_assign_in_while_body(self):
        assert val('''
fn sum_to(n) {
  let total = 0;
  let i = 1;
  while (i <= n) {
    total = total + i;
    i = i + 1;
  }
  return total;
}
fn identity(a) { return a; }
identity(sum_to(5));
''') == 15

    def test_assign_in_for_body(self):
        assert val('''
fn sum_arr(arr) {
  let total = 0;
  for (x in arr) {
    total = total + x;
  }
  return total;
}
fn identity(a) { return a; }
identity(sum_arr([1, 2, 3, 4, 5]));
''') == 15

    def test_method_call_in_loop(self):
        assert val('''
class Counter {
  init() { this.n = 0; }
  inc() { this.n = this.n + 1; }
  get() { return this.n; }
}
fn identity(a) { return a; }
let c = Counter();
let i = 0;
while (i < 5) { c.inc(); i = i + 1; }
identity(c.get());
''') == 5


# ============================================================
# Section 8: Closures and higher-order functions
# ============================================================

class TestClosuresAndHOF:
    """Expression-statements in closures."""

    def test_closure_with_assign(self):
        assert val('''
fn makeCounter() {
  let state = {n: 0};
  return fn() { state.n = state.n + 1; return state.n; };
}
fn identity(a) { return a; }
let counter = makeCounter();
counter();
identity(counter());
''') == 2

    def test_callback_with_assign(self):
        assert val('''
fn apply(f, x) { return f(x); }
fn double(x) { let result = x; result = result * 2; return result; }
fn identity(a) { return a; }
identity(apply(double, 5));
''') == 10


# ============================================================
# Section 9: Try/catch with expression-statements
# ============================================================

class TestTryCatch:
    """Expression-statements inside try/catch."""

    def test_assign_in_try(self):
        assert val('''
fn foo() {
  let x = 0;
  try {
    x = 42;
  } catch (e) {
    x = -1;
  }
  return x;
}
fn identity(a) { return a; }
identity(foo());
''') == 42

    def test_method_call_in_try(self):
        assert out('''
class Foo {
  init() { this.x = 0; }
  set(v) { this.x = v; }
  get() { return this.x; }
}
fn assertEqual(a, b) { if (a != b) { throw "FAIL"; } }
let f = Foo();
try { f.set(99); } catch (e) {}
assertEqual(f.get(), 99);
print "ok";
''') == ['ok']


# ============================================================
# Section 10: Generators with expression-statements
# ============================================================

class TestGenerators:
    """Expression-statements in generators."""

    def test_generator_with_assign(self):
        assert val('''
fn* counter() {
  let n = 0;
  while (true) {
    n = n + 1;
    yield n;
  }
}
fn identity(a) { return a; }
let g = counter();
next(g);
next(g);
identity(next(g));
''') == 3


# ============================================================
# Section 11: Assignment as expression (not statement)
# ============================================================

class TestAssignAsExpression:
    """Assignment used as expression should still work."""

    def test_let_with_assign_value(self):
        """let y = (x = 5) should work."""
        assert val('''
let x = 0;
let y = x = 5;
y;
''') == 5

    def test_assign_in_return(self):
        """return (x = 5) should work."""
        assert val('''
let x = 0;
fn foo() { return x = 42; }
foo();
''') == 42

    def test_assign_in_condition(self):
        """Assignment value used in condition."""
        assert out('''
let x = 0;
if (x = 5) { print x; }
''') == ['5']


# ============================================================
# Section 12: Stdlib interaction tests
# ============================================================

class TestStdlibInteraction:
    """Test with stdlib modules that previously triggered the bug."""

    def test_stdlib_queue_nested(self):
        assert out('''
import "collections";
fn assertEqual(a, b) { if (a != b) { throw "FAIL: " + string(a) + " != " + string(b); } }
let q = Queue();
q.enqueue(1); q.enqueue(2); q.enqueue(3);
assertEqual(q.dequeue(), 1);
assertEqual(q.dequeue(), 2);
assertEqual(q.dequeue(), 3);
print "ok";
''') == ['ok']

    def test_stdlib_iter_as_arg(self):
        """identity(toArray(range(3))) was a reported bug pattern."""
        assert val('''
import "iter";
fn identity(a) { return a; }
identity(toArray(range(3)));
''') == [0, 1, 2]

    def test_stdlib_functional_compose(self):
        assert val('''
import "functional";
fn double(x) { return x * 2; }
fn inc(x) { return x + 1; }
fn identity(a) { return a; }
let f = compose(double, inc);
identity(f(5));
''') == 12

    def test_stdlib_set_nested(self):
        assert out('''
import "collections";
fn assertEqual(a, b) { if (a != b) { throw "FAIL: " + string(a) + " != " + string(b); } }
let s = Set();
s.add(1); s.add(2); s.add(3); s.add(2);
assertEqual(s.size, 3);
assertEqual(s.has(2), true);
print "ok";
''') == ['ok']

    def test_stdlib_testing_module(self):
        assert out('''
import "testing";
import "collections";
let s = Stack();
s.push(1); s.push(2);
assertEqual(s.pop(), 2);
assertEqual(s.pop(), 1);
print "ok";
''') == ['ok']


# ============================================================
# Section 13: Complex composition patterns
# ============================================================

class TestComplexPatterns:
    """Complex real-world patterns that would have failed before the fix."""

    def test_builder_pattern(self):
        assert val('''
class Builder {
  init() { this.parts = []; }
  add(part) { this.parts.push(part); return this; }
  build() { return this.parts; }
}
fn identity(a) { return a; }
let b = Builder();
b.add("a");
b.add("b");
b.add("c");
identity(b.build());
''') == ['a', 'b', 'c']

    def test_state_machine(self):
        assert val('''
class StateMachine {
  init() { this.state = "idle"; this.transitions = 0; }
  transition(to) {
    this.state = to;
    this.transitions = this.transitions + 1;
    return this.state;
  }
  info() { return [this.state, this.transitions]; }
}
fn identity(a) { return a; }
let sm = StateMachine();
sm.transition("running");
sm.transition("paused");
sm.transition("running");
identity(sm.info());
''') == ['running', 3]

    def test_observer_pattern(self):
        assert out('''
class EventEmitter {
  init() { this.listeners = []; }
  on(callback) { this.listeners.push(callback); }
  emit(val) {
    for (listener in this.listeners) {
      listener(val);
    }
  }
}
fn identity(a) { return a; }
let results = [];
let ee = EventEmitter();
ee.on(fn(v) { results.push(v * 2); });
ee.on(fn(v) { results.push(v * 3); });
ee.emit(5);
print identity(results);
''') == ['[10, 15]']

    def test_recursive_with_assign(self):
        assert val('''
fn fibonacci(n) {
  if (n <= 1) { return n; }
  let a = 0;
  let b = 1;
  let i = 2;
  while (i <= n) {
    let temp = b;
    b = a + b;
    a = temp;
    i = i + 1;
  }
  return b;
}
fn identity(a) { return a; }
identity(fibonacci(10));
''') == 55

    def test_linked_list_operations(self):
        assert out('''
import "collections";
fn assertEqual(a, b) { if (a != b) { throw "FAIL: " + string(a) + " != " + string(b); } }
let ll = LinkedList();
ll.append(1); ll.append(2); ll.append(3);
assertEqual(ll.size, 3);
assertEqual(ll.first(), 1);
assertEqual(ll.last(), 3);
assertEqual(ll.contains(2), true);
print "ok";
''') == ['ok']


# ============================================================
# Section 14: Edge cases
# ============================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_function_body(self):
        """Function with no statements should work."""
        assert val('''
fn noop() {}
fn identity(a) { return a; }
identity(noop());
''') is None

    def test_function_with_only_let(self):
        """Function body with only let (no expression-statements)."""
        assert val('''
fn foo() { let x = 5; return x; }
fn identity(a) { return a; }
identity(foo());
''') == 5

    def test_deeply_nested_assigns(self):
        """Many assignments in a function body."""
        assert val('''
fn foo() {
  let a = 0; let b = 0; let c = 0; let d = 0; let e = 0;
  a = 1; b = 2; c = 3; d = 4; e = 5;
  a = a + 1; b = b + 1; c = c + 1; d = d + 1; e = e + 1;
  return a + b + c + d + e;
}
fn identity(a) { return a; }
identity(foo());
''') == 20

    def test_single_expression_program(self):
        """A program with just one expression should still return its value."""
        assert val('5 + 3;') == 8

    def test_last_stmt_is_expression(self):
        """Last statement value should be program result."""
        assert val('''
let x = 10;
x + 5;
''') == 15

    def test_last_stmt_is_call(self):
        """Last statement is a call -- value should be result."""
        assert val('''
fn double(x) { return x * 2; }
double(5);
''') == 10

    def test_many_assigns_before_return(self):
        """Function with many assignments before return."""
        assert val('''
fn compute() {
  let x = 0;
  x = x + 10;
  x = x + 20;
  x = x + 30;
  return x;
}
fn identity(a) { return a; }
identity(compute());
''') == 60

    def test_spread_call_with_assigns(self):
        """Spread call where inner function has assignments."""
        assert val('''
fn makeArgs() {
  let a = [1, 2, 3];
  a[0] = 10;
  return a;
}
fn sum(...args) {
  let total = 0;
  for (x in args) { total = total + x; }
  return total;
}
fn identity(a) { return a; }
identity(sum(...makeArgs()));
''') == 15

    def test_finally_with_assigns(self):
        """Try/finally with assignments in both blocks."""
        assert val('''
fn foo() {
  let x = 0;
  try {
    x = 10;
  } finally {
    x = x + 5;
  }
  return x;
}
fn identity(a) { return a; }
identity(foo());
''') == 15


# ============================================================
# Section 15: Backward compatibility
# ============================================================

class TestBackwardCompat:
    """Ensure existing patterns still work correctly."""

    def test_simple_class(self):
        assert val('''
class Point {
  init(x, y) { this.x = x; this.y = y; }
  distSq() { return this.x * this.x + this.y * this.y; }
}
let p = Point(3, 4);
p.distSq();
''') == 25

    def test_inheritance(self):
        assert val('''
class Animal {
  init(name) { this.name = name; }
  speak() { return this.name + " speaks"; }
}
class Dog < Animal {
  init(name) { super.init(name); }
  speak() { return this.name + " barks"; }
}
let d = Dog("Rex");
d.speak();
''') == 'Rex barks'

    def test_closures(self):
        assert val('''
fn makeAdder(n) {
  return fn(x) { return x + n; };
}
let add5 = makeAdder(5);
add5(10);
''') == 15

    def test_for_in_basic(self):
        assert val('''
let total = 0;
for (x in [1, 2, 3, 4, 5]) {
  total = total + x;
}
total;
''') == 15

    def test_destructuring(self):
        assert val('''
let [a, b, c] = [10, 20, 30];
a + b + c;
''') == 60

    def test_pipe_operator(self):
        assert val('''
fn double(x) { return x * 2; }
fn inc(x) { return x + 1; }
5 |> double |> inc;
''') == 11

    def test_optional_chaining(self):
        assert val('''
let obj = {a: {b: {c: 42}}};
obj?.a?.b?.c;
''') == 42

    def test_null_coalescing(self):
        assert val('''
let x = null;
x ?? 42;
''') == 42

    def test_enum_basic(self):
        assert val('''
enum Color { Red, Green, Blue }
Color.Green.ordinal;
''') == 1

    def test_spread_array(self):
        assert val('''
let a = [1, 2, 3];
let b = [0, ...a, 4];
b;
''') == [0, 1, 2, 3, 4]

    def test_string_interpolation(self):
        assert val('''
let name = "world";
f"hello ${name}";
''') == 'hello world'

    def test_rest_params(self):
        assert val('''
fn sum(...args) {
  let total = 0;
  for (x in args) { total = total + x; }
  return total;
}
sum(1, 2, 3, 4, 5);
''') == 15


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
