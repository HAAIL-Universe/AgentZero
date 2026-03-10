"""
Tests for C058: Static Methods, Getters, and Setters
"""
import pytest
from static_getters_setters import run, execute


# ---- Helper ----

def run_code(source):
    """Run source, return (result, output_lines)."""
    return run(source)


def run_output(source):
    """Run source, return joined output."""
    _, output = run(source)
    return "\n".join(output)


# ==============================================================
# STATIC METHODS
# ==============================================================

class TestStaticMethods:

    def test_basic_static_method(self):
        r, out = run_code("""
        class MathUtils {
            static add(a, b) { return a + b; }
        }
        print MathUtils.add(3, 4);
        """)
        assert out == ["7"]

    def test_static_method_no_this(self):
        """Static methods don't have 'this' bound."""
        r, out = run_code("""
        class Foo {
            static create() { return 42; }
        }
        print Foo.create();
        """)
        assert out == ["42"]

    def test_static_and_instance_methods(self):
        r, out = run_code("""
        class Counter {
            init(n) { this.n = n; }
            increment() { this.n = this.n + 1; }
            get() { return this.n; }
            static zero() { return Counter(0); }
        }
        let c = Counter.zero();
        c.increment();
        c.increment();
        print c.get();
        """)
        assert out == ["2"]

    def test_static_method_with_args(self):
        r, out = run_code("""
        class Calc {
            static multiply(a, b) { return a * b; }
            static square(x) { return Calc.multiply(x, x); }
        }
        print Calc.square(5);
        """)
        assert out == ["25"]

    def test_static_method_inheritance(self):
        r, out = run_code("""
        class Base {
            static greet() { return "hello"; }
        }
        class Child < Base {
        }
        print Child.greet();
        """)
        assert out == ["hello"]

    def test_static_method_override(self):
        r, out = run_code("""
        class Base {
            static name() { return "Base"; }
        }
        class Child < Base {
            static name() { return "Child"; }
        }
        print Base.name();
        print Child.name();
        """)
        assert out == ["Base", "Child"]

    def test_static_method_returns_instance(self):
        r, out = run_code("""
        class Point {
            init(x, y) { this.x = x; this.y = y; }
            static origin() { return Point(0, 0); }
            toString() { return string(this.x) + "," + string(this.y); }
        }
        let p = Point.origin();
        print p.toString();
        """)
        assert out == ["0,0"]

    def test_static_no_instance_access(self):
        """Static methods are not accessible on instances."""
        r, out = run_code("""
        class Foo {
            static bar() { return 1; }
        }
        let f = Foo();
        try {
            f.bar();
        } catch(e) {
            print "caught";
        }
        """)
        assert out == ["caught"]

    def test_static_called_without_parens_is_fn(self):
        r, out = run_code("""
        class Foo {
            static bar() { return 99; }
        }
        let f = Foo.bar;
        print f();
        """)
        assert out == ["99"]

    def test_multiple_static_methods(self):
        r, out = run_code("""
        class Math {
            static max(a, b) { if (a > b) { return a; } else { return b; } }
            static min(a, b) { if (a < b) { return a; } else { return b; } }
            static clamp(x, lo, hi) { return Math.max(lo, Math.min(x, hi)); }
        }
        print Math.max(3, 7);
        print Math.min(3, 7);
        print Math.clamp(10, 0, 5);
        """)
        assert out == ["7", "3", "5"]

    def test_static_with_closures(self):
        r, out = run_code("""
        class Factory {
            static make(n) {
                return fn() { return n * 2; };
            }
        }
        let f = Factory.make(5);
        print f();
        """)
        assert out == ["10"]

    def test_static_error_unknown_method(self):
        r, out = run_code("""
        class Foo {
            static bar() { return 1; }
        }
        try {
            Foo.baz();
        } catch(e) {
            print "caught";
        }
        """)
        assert out == ["caught"]


# ==============================================================
# GETTERS
# ==============================================================

class TestGetters:

    def test_basic_getter(self):
        r, out = run_code("""
        class Circle {
            init(r) { this.r = r; }
            get area() { return 3.14159 * this.r * this.r; }
        }
        let c = Circle(5);
        print c.area;
        """)
        assert out == ["78.53975"]

    def test_getter_no_parens(self):
        """Getters are accessed as properties, not called with ()."""
        r, out = run_code("""
        class Foo {
            get value() { return 42; }
        }
        let f = Foo();
        print f.value;
        """)
        assert out == ["42"]

    def test_getter_with_internal_state(self):
        r, out = run_code("""
        class Person {
            init(first, last) {
                this.first = first;
                this.last = last;
            }
            get fullName() { return this.first + " " + this.last; }
        }
        let p = Person("John", "Doe");
        print p.fullName;
        """)
        assert out == ["John Doe"]

    def test_getter_computed_from_other_props(self):
        r, out = run_code("""
        class Rect {
            init(w, h) { this.w = w; this.h = h; }
            get area() { return this.w * this.h; }
            get perimeter() { return 2 * (this.w + this.h); }
        }
        let r = Rect(3, 4);
        print r.area;
        print r.perimeter;
        """)
        assert out == ["12", "14"]

    def test_getter_overrides_property(self):
        """Getter takes precedence over same-name instance property."""
        r, out = run_code("""
        class Foo {
            init() { this.x = 10; }
            get x() { return 99; }
        }
        let f = Foo();
        print f.x;
        """)
        assert out == ["99"]

    def test_getter_inheritance(self):
        r, out = run_code("""
        class Base {
            init() { this.val = 5; }
            get doubled() { return this.val * 2; }
        }
        class Child < Base {
            init() { super.init(); }
        }
        let c = Child();
        print c.doubled;
        """)
        assert out == ["10"]

    def test_getter_override_in_subclass(self):
        r, out = run_code("""
        class Base {
            get name() { return "Base"; }
        }
        class Child < Base {
            get name() { return "Child"; }
        }
        let b = Base();
        let c = Child();
        print b.name;
        print c.name;
        """)
        assert out == ["Base", "Child"]

    def test_getter_called_multiple_times(self):
        r, out = run_code("""
        class Counter {
            init() { this.count = 0; }
            get next() {
                this.count = this.count + 1;
                return this.count;
            }
        }
        let c = Counter();
        print c.next;
        print c.next;
        print c.next;
        """)
        assert out == ["1", "2", "3"]

    def test_getter_returning_array(self):
        r, out = run_code("""
        class Range {
            init(start, end) { this.start = start; this.end = end; }
            get values() {
                let result = [];
                let i = this.start;
                while (i <= this.end) {
                    result.push(i);
                    i = i + 1;
                }
                return result;
            }
        }
        let r = Range(1, 3);
        print r.values;
        """)
        assert out == ["[1, 2, 3]"]


# ==============================================================
# SETTERS
# ==============================================================

class TestSetters:

    def test_basic_setter(self):
        r, out = run_code("""
        class Temp {
            init() { this._celsius = 0; }
            get celsius() { return this._celsius; }
            set celsius(c) { this._celsius = c; }
        }
        let t = Temp();
        t.celsius = 100;
        print t.celsius;
        """)
        assert out == ["100"]

    def test_setter_with_validation(self):
        r, out = run_code("""
        class Age {
            init() { this._age = 0; }
            get age() { return this._age; }
            set age(v) {
                if (v < 0) { this._age = 0; }
                else { this._age = v; }
            }
        }
        let a = Age();
        a.age = 25;
        print a.age;
        a.age = -5;
        print a.age;
        """)
        assert out == ["25", "0"]

    def test_setter_with_transform(self):
        r, out = run_code("""
        class Lower {
            init() { this._text = ""; }
            get text() { return this._text; }
            set text(s) { this._text = s.toLowerCase(); }
        }
        let l = Lower();
        l.text = "HELLO";
        print l.text;
        """)
        assert out == ["hello"]

    def test_setter_inheritance(self):
        r, out = run_code("""
        class Base {
            init() { this._val = 0; }
            set val(v) { this._val = v * 2; }
            get val() { return this._val; }
        }
        class Child < Base {
            init() { super.init(); }
        }
        let c = Child();
        c.val = 5;
        print c.val;
        """)
        assert out == ["10"]

    def test_setter_override(self):
        r, out = run_code("""
        class Base {
            init() { this._x = 0; }
            set x(v) { this._x = v; }
            get x() { return this._x; }
        }
        class Child < Base {
            init() { super.init(); }
            set x(v) { this._x = v + 100; }
        }
        let b = Base();
        let c = Child();
        b.x = 5;
        c.x = 5;
        print b.x;
        print c.x;
        """)
        assert out == ["5", "105"]

    def test_getter_setter_pair(self):
        r, out = run_code("""
        class Temperature {
            init(c) { this._c = c; }
            get fahrenheit() { return this._c * 9 / 5 + 32; }
            set fahrenheit(f) { this._c = (f - 32) * 5 / 9; }
            get celsius() { return this._c; }
        }
        let t = Temperature(0);
        print t.fahrenheit;
        t.fahrenheit = 212;
        print t.celsius;
        """)
        assert out[0] in ["32", "32.0"]
        assert out[1] in ["100", "100.0"]

    def test_setter_no_getter(self):
        """Can have setter without getter -- property accessed normally."""
        r, out = run_code("""
        class Foo {
            init() { this._log = []; }
            set action(v) {
                this._log.push(v);
            }
        }
        let f = Foo();
        f.action = "start";
        f.action = "stop";
        print f._log;
        """)
        assert out == ["[start, stop]"]


# ==============================================================
# COMBINED FEATURES
# ==============================================================

class TestCombined:

    def test_static_getter_setter_together(self):
        r, out = run_code("""
        class Config {
            init(name) { this._name = name; }
            get name() { return this._name; }
            set name(v) { this._name = v; }
            static default() { return Config("default"); }
        }
        let c = Config.default();
        print c.name;
        c.name = "custom";
        print c.name;
        """)
        assert out == ["default", "custom"]

    def test_static_with_getter_setter_and_methods(self):
        r, out = run_code("""
        class Stack {
            init() { this._items = []; }
            push(item) { this._items.push(item); }
            pop() { return this._items.pop(); }
            get size() { return this._items.length; }
            get isEmpty() { return this._items.length == 0; }
            static create() { return Stack(); }
        }
        let s = Stack.create();
        print s.isEmpty;
        s.push(1);
        s.push(2);
        print s.size;
        print s.isEmpty;
        print s.pop();
        print s.size;
        """)
        assert out == ["true", "2", "false", "2", "1"]

    def test_class_hierarchy_with_all_features(self):
        r, out = run_code("""
        class Shape {
            init(type) { this._type = type; }
            get type() { return this._type; }
            static unit() { return Shape("unit"); }
        }
        class Circle < Shape {
            init(r) {
                super.init("circle");
                this._r = r;
            }
            get radius() { return this._r; }
            set radius(r) { this._r = r; }
            static unit() { return Circle(1); }
        }
        let s = Shape.unit();
        print s.type;
        let c = Circle.unit();
        print c.type;
        print c.radius;
        c.radius = 10;
        print c.radius;
        """)
        assert out == ["unit", "circle", "1", "10"]

    def test_method_named_get_or_set_without_modifier(self):
        """'get' and 'set' as regular method names still work."""
        r, out = run_code("""
        class Map {
            init() { this.data = {}; }
            get(key) { return this.data[key]; }
            set(key, val) { this.data[key] = val; }
        }
        let m = Map();
        m.set("a", 1);
        print m.get("a");
        """)
        assert out == ["1"]

    def test_static_as_method_name(self):
        """'static' as regular method name still works."""
        r, out = run_code("""
        class Foo {
            static() { return "I am a method named static"; }
        }
        let f = Foo();
        print f.static();
        """)
        assert out == ["I am a method named static"]


# ==============================================================
# EDGE CASES
# ==============================================================

class TestEdgeCases:

    def test_getter_with_try_catch(self):
        r, out = run_code("""
        class Risky {
            init() { this._data = null; }
            get data() {
                if (this._data == null) {
                    throw "no data";
                }
                return this._data;
            }
            set data(v) { this._data = v; }
        }
        let r = Risky();
        try {
            print r.data;
        } catch(e) {
            print e;
        }
        r.data = "hello";
        print r.data;
        """)
        assert out == ["no data", "hello"]

    def test_getter_in_expression(self):
        r, out = run_code("""
        class Vec {
            init(x, y) { this.x = x; this.y = y; }
            get magnitude() { return this.x * this.x + this.y * this.y; }
        }
        let v = Vec(3, 4);
        print v.magnitude + 1;
        """)
        assert out == ["26"]

    def test_setter_returns_value(self):
        """Assignment with setter still evaluates to the assigned value."""
        r, out = run_code("""
        class Foo {
            init() { this._x = 0; }
            set x(v) { this._x = v * 2; }
            get x() { return this._x; }
        }
        let f = Foo();
        let result = f.x = 5;
        print result;
        print f.x;
        """)
        assert out == ["5", "10"]

    def test_multiple_getters_setters(self):
        r, out = run_code("""
        class Point {
            init(x, y) { this._x = x; this._y = y; }
            get x() { return this._x; }
            get y() { return this._y; }
            set x(v) { this._x = v; }
            set y(v) { this._y = v; }
        }
        let p = Point(1, 2);
        print p.x;
        print p.y;
        p.x = 10;
        p.y = 20;
        print p.x;
        print p.y;
        """)
        assert out == ["1", "2", "10", "20"]

    def test_getter_with_string_interpolation(self):
        r, out = run_code("""
        class User {
            init(name) { this._name = name; }
            get greeting() { return f"Hello, ${this._name}!"; }
        }
        let u = User("Alice");
        print u.greeting;
        """)
        assert out == ["Hello, Alice!"]

    def test_static_with_array_operations(self):
        r, out = run_code("""
        class ListUtils {
            static sum(arr) {
                let total = 0;
                for (x in arr) {
                    total = total + x;
                }
                return total;
            }
            static avg(arr) {
                return ListUtils.sum(arr) / arr.length;
            }
        }
        print ListUtils.sum([1, 2, 3, 4, 5]);
        print ListUtils.avg([10, 20, 30]);
        """)
        assert out == ["15", "20"]

    def test_inherited_getter_uses_child_state(self):
        r, out = run_code("""
        class Base {
            get info() { return this.name + ":" + string(this.val); }
        }
        class Child < Base {
            init(name, val) {
                this.name = name;
                this.val = val;
            }
        }
        let c = Child("test", 42);
        print c.info;
        """)
        assert out == ["test:42"]

    def test_setter_called_from_method(self):
        r, out = run_code("""
        class Bounded {
            init(lo, hi) {
                this.lo = lo;
                this.hi = hi;
                this._val = lo;
            }
            get val() { return this._val; }
            set val(v) {
                if (v < this.lo) { this._val = this.lo; }
                else if (v > this.hi) { this._val = this.hi; }
                else { this._val = v; }
            }
            reset() { this.val = this.lo; }
        }
        let b = Bounded(0, 10);
        b.val = 5;
        print b.val;
        b.val = 15;
        print b.val;
        b.reset();
        print b.val;
        """)
        assert out == ["5", "10", "0"]

    def test_static_method_as_callback(self):
        r, out = run_code("""
        class Transform {
            static double(x) { return x * 2; }
        }
        let arr = [1, 2, 3];
        let result = arr.map(Transform.double);
        print result;
        """)
        assert out == ["[2, 4, 6]"]

    def test_getter_on_optional_chain(self):
        r, out = run_code("""
        class Box {
            init(v) { this._v = v; }
            get value() { return this._v; }
        }
        let b = Box(42);
        print b?.value;
        let n = null;
        print n?.value;
        """)
        assert out == ["42", "null"]


# ==============================================================
# PARSER VALIDATION
# ==============================================================

class TestParserValidation:

    def test_getter_with_params_is_error(self):
        with pytest.raises(Exception):
            run_code("""
            class Foo {
                get bar(x) { return x; }
            }
            """)

    def test_setter_with_no_params_is_error(self):
        with pytest.raises(Exception):
            run_code("""
            class Foo {
                set bar() { return 1; }
            }
            """)

    def test_setter_with_two_params_is_error(self):
        with pytest.raises(Exception):
            run_code("""
            class Foo {
                set bar(a, b) { return 1; }
            }
            """)


# ==============================================================
# EXISTING FUNCTIONALITY REGRESSION
# ==============================================================

class TestRegression:

    def test_basic_class_still_works(self):
        r, out = run_code("""
        class Dog {
            init(name) { this.name = name; }
            speak() { return this.name + " says woof"; }
        }
        let d = Dog("Rex");
        print d.speak();
        """)
        assert out == ["Rex says woof"]

    def test_inheritance_still_works(self):
        r, out = run_code("""
        class Animal {
            init(name) { this.name = name; }
            speak() { return this.name; }
        }
        class Cat < Animal {
            init(name) { super.init(name); }
            speak() { return super.speak() + " meows"; }
        }
        let c = Cat("Whiskers");
        print c.speak();
        """)
        assert out == ["Whiskers meows"]

    def test_instanceof_still_works(self):
        r, out = run_code("""
        class A {}
        class B < A {}
        let b = B();
        print instanceof(b, B);
        print instanceof(b, A);
        """)
        assert out == ["true", "true"]

    def test_hash_maps_still_work(self):
        r, out = run_code("""
        let h = {a: 1, b: 2};
        print h.a;
        print h.size;
        """)
        assert out == ["1", "2"]

    def test_string_methods_still_work(self):
        r, out = run_code("""
        let s = "hello world";
        print s.toUpperCase();
        print s.split(" ");
        """)
        assert out == ["HELLO WORLD", "[hello, world]"]

    def test_array_methods_still_work(self):
        r, out = run_code("""
        let a = [3, 1, 2];
        print a.sort();
        print a.length;
        """)
        assert out == ["[1, 2, 3]", "3"]

    def test_closures_still_work(self):
        r, out = run_code("""
        fn make(n) { return fn() { return n; }; }
        let f = make(42);
        print f();
        """)
        assert out == ["42"]

    def test_async_still_works(self):
        r, out = run_code("""
        async fn greet() { return "hi"; }
        let p = greet();
        print await p;
        """)
        assert out == ["hi"]

    def test_for_in_still_works(self):
        r, out = run_code("""
        let sum = 0;
        for (x in [1, 2, 3]) { sum = sum + x; }
        print sum;
        """)
        assert out == ["6"]

    def test_destructuring_still_works(self):
        r, out = run_code("""
        let [a, b, c] = [1, 2, 3];
        print a + b + c;
        """)
        assert out == ["6"]

    def test_spread_still_works(self):
        r, out = run_code("""
        let a = [1, 2];
        let b = [...a, 3, 4];
        print b;
        """)
        assert out == ["[1, 2, 3, 4]"]

    def test_pipe_still_works(self):
        r, out = run_code("""
        fn double(x) { return x * 2; }
        fn add1(x) { return x + 1; }
        print 5 |> double |> add1;
        """)
        assert out == ["11"]

    def test_optional_chaining_still_works(self):
        r, out = run_code("""
        let h = {a: {b: 42}};
        print h?.a?.b;
        let n = null;
        print n?.a;
        """)
        assert out == ["42", "null"]

    def test_null_coalescing_still_works(self):
        r, out = run_code("""
        let x = null ?? "default";
        print x;
        """)
        assert out == ["default"]

    def test_try_finally_still_works(self):
        r, out = run_code("""
        let x = 0;
        try {
            x = 1;
        } finally {
            x = x + 10;
        }
        print x;
        """)
        assert out == ["11"]

    def test_generators_still_work(self):
        r, out = run_code("""
        fn* count(n) {
            let i = 0;
            while (i < n) {
                yield i;
                i = i + 1;
            }
        }
        let g = count(3);
        print next(g);
        print next(g);
        print next(g);
        """)
        assert out == ["0", "1", "2"]

    def test_modules_still_work(self):
        r, out = run_code("""
        let x = 42;
        print x;
        """)
        assert out == ["42"]

    def test_error_handling_still_works(self):
        r, out = run_code("""
        try {
            throw "oops";
        } catch(e) {
            print e;
        }
        """)
        assert out == ["oops"]

    def test_if_else_still_works(self):
        r, out = run_code("""
        let x = 3;
        if (x == 1) { print "one"; }
        else if (x == 2) { print "two"; }
        else { print "other"; }
        """)
        assert out == ["other"]


# ==============================================================
# ADVANCED PATTERNS
# ==============================================================

class TestAdvanced:

    def test_builder_pattern(self):
        r, out = run_code("""
        class QueryBuilder {
            init() {
                this._table = "";
                this._conditions = [];
                this._limit = null;
            }
            set table(t) { this._table = t; }
            get table() { return this._table; }

            addCondition(c) {
                this._conditions.push(c);
                return this;
            }

            setLimit(n) {
                this._limit = n;
                return this;
            }

            get query() {
                let q = "SELECT * FROM " + this._table;
                if (this._conditions.length > 0) {
                    q = q + " WHERE " + this._conditions.join(" AND ");
                }
                if (this._limit != null) {
                    q = q + " LIMIT " + string(this._limit);
                }
                return q;
            }

            static create(table) {
                let qb = QueryBuilder();
                qb.table = table;
                return qb;
            }
        }
        let q = QueryBuilder.create("users")
            .addCondition("age > 18")
            .addCondition("active = 1")
            .setLimit(10);
        print q.query;
        """)
        assert out == ["SELECT * FROM users WHERE age > 18 AND active = 1 LIMIT 10"]

    def test_observable_pattern(self):
        r, out = run_code("""
        class Observable {
            init(val) {
                this._val = val;
                this._listeners = [];
            }
            get value() { return this._val; }
            set value(v) {
                this._val = v;
                for (listener in this._listeners) {
                    listener(v);
                }
            }
            subscribe(callback) {
                this._listeners.push(callback);
            }
        }
        let log = [];
        let obs = Observable(0);
        obs.subscribe(fn(v) { log.push(v); });
        obs.value = 1;
        obs.value = 2;
        obs.value = 3;
        print log;
        """)
        assert out == ["[1, 2, 3]"]

    def test_singleton_pattern(self):
        r, out = run_code("""
        class Singleton {
            init() { this.data = "instance"; }
            static instance() {
                return Singleton();
            }
        }
        let a = Singleton.instance();
        print a.data;
        """)
        assert out == ["instance"]

    def test_computed_property_via_getter(self):
        r, out = run_code("""
        class Matrix2x2 {
            init(a, b, c, d) {
                this.a = a;
                this.b = b;
                this.c = c;
                this.d = d;
            }
            get determinant() { return this.a * this.d - this.b * this.c; }
            get trace() { return this.a + this.d; }
            get isIdentity() {
                return this.a == 1 && this.b == 0 && this.c == 0 && this.d == 1;
            }
            static identity() { return Matrix2x2(1, 0, 0, 1); }
        }
        let m = Matrix2x2(2, 3, 1, 4);
        print m.determinant;
        print m.trace;
        print m.isIdentity;
        let id = Matrix2x2.identity();
        print id.isIdentity;
        """)
        assert out == ["5", "6", "false", "true"]

    def test_deep_inheritance_chain(self):
        r, out = run_code("""
        class A {
            get label() { return "A"; }
            static type() { return "base"; }
        }
        class B < A {
        }
        class C < B {
            get label() { return "C"; }
        }
        let a = A();
        let b = B();
        let c = C();
        print a.label;
        print b.label;
        print c.label;
        print C.type();
        """)
        assert out == ["A", "A", "C", "base"]

    def test_event_emitter(self):
        r, out = run_code("""
        class EventEmitter {
            init() { this._handlers = {}; }
            on(event, handler) {
                if (!this._handlers.has(event)) {
                    this._handlers[event] = [];
                }
                this._handlers[event].push(handler);
            }
            emit(event, data) {
                if (this._handlers.has(event)) {
                    for (h in this._handlers[event]) {
                        h(data);
                    }
                }
            }
            get eventCount() { return this._handlers.keys().length; }
            static create() { return EventEmitter(); }
        }
        let ee = EventEmitter.create();
        let results = [];
        ee.on("data", fn(v) { results.push(v); });
        ee.on("data", fn(v) { results.push(v * 2); });
        ee.emit("data", 5);
        print results;
        print ee.eventCount;
        """)
        assert out == ["[5, 10]", "1"]


# ==============================================================
# STRESS TESTS
# ==============================================================

class TestStress:

    def test_many_static_methods(self):
        r, out = run_code("""
        class Utils {
            static a() { return 1; }
            static b() { return 2; }
            static c() { return 3; }
            static d() { return 4; }
            static e() { return 5; }
        }
        print Utils.a() + Utils.b() + Utils.c() + Utils.d() + Utils.e();
        """)
        assert out == ["15"]

    def test_many_getters(self):
        r, out = run_code("""
        class MultiGet {
            init() { this._a = 1; this._b = 2; this._c = 3; }
            get a() { return this._a; }
            get b() { return this._b; }
            get c() { return this._c; }
            get sum() { return this._a + this._b + this._c; }
        }
        let m = MultiGet();
        print m.a;
        print m.b;
        print m.c;
        print m.sum;
        """)
        assert out == ["1", "2", "3", "6"]

    def test_getter_setter_loop(self):
        r, out = run_code("""
        class Acc {
            init() { this._n = 0; }
            get n() { return this._n; }
            set n(v) { this._n = v; }
        }
        let a = Acc();
        let i = 0;
        while (i < 100) {
            a.n = a.n + 1;
            i = i + 1;
        }
        print a.n;
        """)
        assert out == ["100"]

    def test_static_recursive(self):
        r, out = run_code("""
        class Fib {
            static calc(n) {
                if (n <= 1) { return n; }
                return Fib.calc(n - 1) + Fib.calc(n - 2);
            }
        }
        print Fib.calc(10);
        """)
        assert out == ["55"]

    def test_chained_getter_access(self):
        r, out = run_code("""
        class Node {
            init(val, next) { this._val = val; this._next = next; }
            get val() { return this._val; }
            get next() { return this._next; }
        }
        let list = Node(1, Node(2, Node(3, null)));
        print list.val;
        print list.next.val;
        print list.next.next.val;
        """)
        assert out == ["1", "2", "3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
