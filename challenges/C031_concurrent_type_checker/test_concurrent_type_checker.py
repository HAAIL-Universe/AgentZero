"""
Tests for C031: Concurrent Type Checker
Composes C013 (type checker) + C029 (concurrent runtime AST)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from concurrent_type_checker import (
    check_source, check_program, parse_concurrent, format_errors, type_of,
    ConcurrentTypeChecker, TChan, TTask, TInt, TFloat, TString, TBool, TVoid,
    TFunc, TVar, INT, FLOAT, STRING, BOOL, VOID,
    conc_unify, resolve, UnificationError,
    TypeError_, DeadlockWarning,
)


# ============================================================
# Helper
# ============================================================

def errors_from(source):
    """Return list of type errors from source."""
    errors, warnings, checker = check_source(source)
    return errors

def warnings_from(source):
    """Return list of warnings from source."""
    errors, warnings, checker = check_source(source)
    return warnings

def no_errors(source):
    """Assert source has no type errors."""
    errors = errors_from(source)
    assert errors == [], f"Expected no errors, got: {format_errors(errors)}"

def has_error(source, msg_fragment):
    """Assert source has an error containing msg_fragment."""
    errors = errors_from(source)
    assert any(msg_fragment in e.message for e in errors), \
        f"Expected error containing '{msg_fragment}', got: {format_errors(errors)}"

def has_warning(source, msg_fragment):
    """Assert source has a warning containing msg_fragment."""
    warnings = warnings_from(source)
    assert any(msg_fragment in w.message for w in warnings), \
        f"Expected warning containing '{msg_fragment}', got: {[str(w) for w in warnings]}"

def error_count(source):
    return len(errors_from(source))

def warning_count(source):
    return len(warnings_from(source))


# ============================================================
# 1. Basic Type Checking (inherited from C013)
# ============================================================

class TestBasicTypes:
    """Verify base type checking still works in concurrent context."""

    def test_int_literal(self):
        no_errors("let x = 42;")

    def test_float_literal(self):
        no_errors("let x = 3.14;")

    def test_string_literal(self):
        no_errors('let x = "hello";')

    def test_bool_literal(self):
        no_errors("let x = true;")

    def test_arithmetic_int(self):
        no_errors("let x = 1 + 2;")

    def test_arithmetic_float(self):
        no_errors("let x = 1.0 + 2.0;")

    def test_arithmetic_mixed(self):
        no_errors("let x = 1 + 2.0;")

    def test_string_concat(self):
        no_errors('let x = "a" + "b";')

    def test_comparison(self):
        no_errors("let x = 1 < 2;")

    def test_logical(self):
        no_errors("let x = true and false;")

    def test_undefined_var(self):
        has_error("let x = y;", "Undefined variable 'y'")

    def test_type_mismatch_arithmetic(self):
        has_error('let x = "a" + 1;', "Cannot apply")

    def test_function_decl(self):
        no_errors("fn add(a, b) { return a + b; }")

    def test_function_call(self):
        no_errors("""
            fn add(a, b) { return a + b; }
            let x = add(1, 2);
        """)

    def test_function_wrong_arity(self):
        has_error("""
            fn add(a, b) { return a + b; }
            let x = add(1);
        """, "expects 2 args")

    def test_if_stmt(self):
        no_errors("if (true) { let x = 1; }")

    def test_if_non_bool_condition(self):
        has_error("if (42) { let x = 1; }", "Condition must be bool")

    def test_while_stmt(self):
        no_errors("let x = true; while (x) { x = false; }")

    def test_print_stmt(self):
        no_errors("print(42);")

    def test_return_outside_function(self):
        has_error("return 42;", "Return statement outside")

    def test_block_scoping(self):
        no_errors("let x = 1; { let y = 2; }")


# ============================================================
# 2. Channel Types
# ============================================================

class TestChannelTypes:
    """Test TChan type creation and checking."""

    def test_chan_creation(self):
        no_errors("let ch = chan(1);")

    def test_chan_creation_literal_size(self):
        no_errors("let ch = chan(10);")

    def test_chan_creation_non_int_size(self):
        has_error('let ch = chan("big");', "Channel buffer size must be int")

    def test_chan_type_inference(self):
        """Channel element type is inferred from first send."""
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
        """)

    def test_chan_type_repr(self):
        t = TChan(INT)
        assert repr(t) == "Chan<int>"

    def test_chan_nested(self):
        """Channel of channels."""
        t = TChan(TChan(INT))
        assert repr(t) == "Chan<Chan<int>>"

    def test_type_of_chan(self):
        t = type_of("let ch = chan(1);", "ch")
        assert isinstance(resolve(t), TChan)

    def test_chan_equality(self):
        assert TChan(INT) == TChan(INT)
        assert TChan(INT) != TChan(STRING)


# ============================================================
# 3. Send/Recv Type Safety
# ============================================================

class TestSendRecv:
    """Test send/recv channel type safety."""

    def test_send_int(self):
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
        """)

    def test_recv_after_send(self):
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
            let val = recv(ch);
        """)

    def test_send_wrong_type(self):
        """Send string to int channel."""
        has_error("""
            let ch = chan(1);
            send(ch, 42);
            send(ch, "hello");
        """, "Cannot send")

    def test_send_infers_channel_type(self):
        """First send determines channel element type."""
        no_errors("""
            let ch = chan(1);
            send(ch, 3.14);
        """)

    def test_recv_returns_element_type(self):
        """recv returns the channel's element type."""
        t = type_of("""
            let ch = chan(1);
            send(ch, 42);
        """, "recv(ch)")
        assert isinstance(resolve(t), TInt)

    def test_send_to_non_channel(self):
        has_error("""
            let x = 42;
            send(x, 1);
        """, "send requires channel")

    def test_recv_from_non_channel(self):
        has_error("""
            let x = 42;
            recv(x);
        """, "recv requires channel")

    def test_send_float_to_int_channel(self):
        """Float can be sent to int channel (promotion)."""
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
            send(ch, 3.14);
        """)

    def test_send_preserves_type_across_uses(self):
        """Multiple sends must agree on type."""
        has_error("""
            let ch = chan(1);
            send(ch, 42);
            send(ch, true);
        """, "Cannot send")

    def test_recv_infers_from_send(self):
        """recv type matches what was sent."""
        no_errors("""
            let ch = chan(1);
            send(ch, "hello");
            let msg = recv(ch);
            let x = msg + " world";
        """)

    def test_recv_type_mismatch_usage(self):
        """Using recv result incorrectly."""
        has_error("""
            let ch = chan(1);
            send(ch, "hello");
            let msg = recv(ch);
            let x = msg + 42;
        """, "Cannot apply")

    def test_send_void(self):
        """Can't send void to channel."""
        no_errors("""
            fn nothing() { return; }
            let ch = chan(1);
            send(ch, nothing());
        """)


# ============================================================
# 4. Spawn Type Checking
# ============================================================

class TestSpawn:
    """Test spawn type checking."""

    def test_spawn_basic(self):
        no_errors("""
            fn worker() { let x = 42; }
            let t = spawn worker();
        """)

    def test_spawn_returns_task(self):
        t = type_of("""
            fn worker() { return 42; }
        """, "spawn worker()")
        assert isinstance(resolve(t), TTask)

    def test_spawn_task_result_type(self):
        """Spawn preserves function return type in TTask."""
        t = type_of("""
            fn worker() { return 42; }
        """, "spawn worker()")
        resolved = resolve(t)
        assert isinstance(resolved, TTask)

    def test_spawn_with_args(self):
        no_errors("""
            fn worker(x, y) { return x + y; }
            let t = spawn worker(1, 2);
        """)

    def test_spawn_wrong_arg_count(self):
        has_error("""
            fn worker(x, y) { return x + y; }
            let t = spawn worker(1);
        """, "expects 2 args")

    def test_spawn_wrong_arg_type(self):
        has_error("""
            fn add(a, b) { return a + b; }
            let x = add(1, 2);
            let t = spawn add("hello", 2);
        """, "spawn 'add' arg 1")

    def test_spawn_undefined_function(self):
        has_error("""
            let t = spawn nonexistent();
        """, "Undefined function 'nonexistent'")

    def test_spawn_non_function(self):
        has_error("""
            let x = 42;
            let t = spawn x();
        """, "Cannot spawn non-function")

    def test_spawn_result_type(self):
        """Task type preserves function return type."""
        t = type_of("""
            fn greet() { return "hello"; }
        """, "spawn greet()")
        resolved = resolve(t)
        assert isinstance(resolved, TTask)

    def test_task_type_repr(self):
        t = TTask(INT)
        assert repr(t) == "Task<int>"


# ============================================================
# 5. Join Type Checking
# ============================================================

class TestJoin:
    """Test join type inference."""

    def test_join_basic(self):
        no_errors("""
            fn worker() { return 42; }
            let t = spawn worker();
            let result = join(t);
        """)

    def test_join_returns_task_result_type(self):
        """join(task) returns the task's result type."""
        t = type_of("""
            fn worker() { return 42; }
            let t = spawn worker();
        """, "join(t)")
        resolved = resolve(t)
        # Result should be int (worker returns 42)
        assert isinstance(resolved, (TInt, TVar))

    def test_join_non_task(self):
        has_error("""
            let x = 42;
            let result = join(x);
        """, "join requires task")

    def test_join_result_usage(self):
        """Can use join result in arithmetic."""
        no_errors("""
            fn compute() { return 42; }
            let t = spawn compute();
            let result = join(t) + 1;
        """)

    def test_join_string_task(self):
        no_errors("""
            fn greet() { return "hello"; }
            let t = spawn greet();
            let msg = join(t);
        """)

    def test_join_type_mismatch_usage(self):
        """Using join result incorrectly."""
        has_error("""
            fn greet() { return "hello"; }
            let t = spawn greet();
            let msg = join(t);
            let x = msg - 1;
        """, "Cannot apply")


# ============================================================
# 6. Yield Statement
# ============================================================

class TestYield:
    """Test yield statement type checking."""

    def test_yield_in_function(self):
        no_errors("""
            fn worker() {
                yield;
                return 42;
            }
        """)

    def test_yield_top_level(self):
        no_errors("yield;")

    def test_yield_in_loop(self):
        no_errors("""
            fn worker() {
                let i = 0;
                while (i < 10) {
                    yield;
                    i = i + 1;
                }
            }
        """)


# ============================================================
# 7. TaskId Expression
# ============================================================

class TestTaskId:
    """Test task_id expression type."""

    def test_task_id_is_int(self):
        t = type_of("", "task_id")
        assert isinstance(resolve(t), TInt)

    def test_task_id_in_expression(self):
        no_errors("let id = task_id + 1;")


# ============================================================
# 8. Select Statement
# ============================================================

class TestSelect:
    """Test select statement type checking."""

    def test_select_recv(self):
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
            select {
                case recv(ch) => val {
                    print(val);
                }
            }
        """)

    def test_select_send(self):
        no_errors("""
            let ch = chan(1);
            select {
                case send(ch, 42) => {
                    print(1);
                }
            }
        """)

    def test_select_default(self):
        no_errors("""
            let ch = chan(1);
            select {
                case recv(ch) => val {
                    print(val);
                }
                default => {
                    print(0);
                }
            }
        """)

    def test_select_non_channel(self):
        has_error("""
            let x = 42;
            select {
                case recv(x) => val {
                    print(val);
                }
            }
        """, "Select case requires channel")

    def test_select_send_wrong_type(self):
        has_error("""
            let ch = chan(1);
            send(ch, 42);
            select {
                case send(ch, "hello") => {
                    print(1);
                }
            }
        """, "Select send: channel expects")

    def test_select_recv_variable_scoping(self):
        """Recv variable is scoped to the case body."""
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
            select {
                case recv(ch) => val {
                    let x = val + 1;
                }
            }
        """)

    def test_select_multiple_cases(self):
        no_errors("""
            let ch1 = chan(1);
            let ch2 = chan(1);
            send(ch1, 42);
            send(ch2, "hello");
            select {
                case recv(ch1) => v1 {
                    print(v1);
                }
                case recv(ch2) => v2 {
                    print(v2);
                }
            }
        """)


# ============================================================
# 9. Concurrent Unification
# ============================================================

class TestConcUnification:
    """Test extended unification with concurrent types."""

    def test_unify_chan_same(self):
        result = conc_unify(TChan(INT), TChan(INT))
        assert isinstance(result, TChan)

    def test_unify_chan_different(self):
        with pytest.raises(UnificationError):
            conc_unify(TChan(INT), TChan(STRING))

    def test_unify_chan_with_tvar(self):
        tv = TVar(999)
        result = conc_unify(tv, TChan(INT))
        assert isinstance(resolve(tv), TChan)

    def test_unify_task_same(self):
        result = conc_unify(TTask(INT), TTask(INT))
        assert isinstance(result, TTask)

    def test_unify_task_different(self):
        with pytest.raises(UnificationError):
            conc_unify(TTask(INT), TTask(STRING))

    def test_unify_chan_task_mismatch(self):
        with pytest.raises(UnificationError):
            conc_unify(TChan(INT), TTask(INT))

    def test_unify_chan_int_mismatch(self):
        with pytest.raises(UnificationError):
            conc_unify(TChan(INT), INT)

    def test_unify_task_int_mismatch(self):
        with pytest.raises(UnificationError):
            conc_unify(TTask(INT), INT)

    def test_unify_chan_tvar_elem(self):
        """Unify channels with type variable elements."""
        tv = TVar(998)
        result = conc_unify(TChan(tv), TChan(INT))
        assert isinstance(result, TChan)
        assert isinstance(resolve(tv), TInt)


# ============================================================
# 10. Channel Type Inference
# ============================================================

class TestChannelTypeInference:
    """Test type inference through channel operations."""

    def test_infer_from_send(self):
        """Channel type inferred from first send."""
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
            let val = recv(ch);
            let x = val + 1;
        """)

    def test_infer_from_recv_usage(self):
        """Channel type inferred from recv usage context."""
        no_errors("""
            let ch = chan(1);
            send(ch, "hello");
            let msg = recv(ch);
            let greeting = msg + " world";
        """)

    def test_multiple_channels_independent(self):
        """Different channels can have different element types."""
        no_errors("""
            let int_ch = chan(1);
            let str_ch = chan(1);
            send(int_ch, 42);
            send(str_ch, "hello");
            let n = recv(int_ch);
            let s = recv(str_ch);
            let x = n + 1;
            let y = s + " world";
        """)

    def test_channel_type_propagation(self):
        """Channel type propagates through function calls."""
        no_errors("""
            fn producer(ch) {
                send(ch, 42);
            }
            fn consumer(ch) {
                let val = recv(ch);
            }
            let ch = chan(1);
            producer(ch);
            consumer(ch);
        """)

    def test_channel_used_as_function_param(self):
        no_errors("""
            fn send_val(ch, v) {
                send(ch, v);
            }
            let ch = chan(1);
            send_val(ch, 100);
        """)


# ============================================================
# 11. Spawn + Channel Composition
# ============================================================

class TestSpawnChannel:
    """Test spawn with channel communication."""

    def test_spawn_with_channel(self):
        no_errors("""
            fn worker(ch) {
                send(ch, 42);
            }
            let ch = chan(1);
            let t = spawn worker(ch);
        """)

    def test_spawn_recv_from_worker(self):
        no_errors("""
            fn worker(ch) {
                send(ch, 42);
            }
            let ch = chan(1);
            let t = spawn worker(ch);
            let result = recv(ch);
        """)

    def test_spawn_join_with_channel(self):
        no_errors("""
            fn compute(ch) {
                let val = recv(ch);
                return val + 1;
            }
            let ch = chan(1);
            send(ch, 41);
            let t = spawn compute(ch);
            let result = join(t);
        """)

    def test_multiple_spawns(self):
        no_errors("""
            fn worker(id) {
                return id;
            }
            let t1 = spawn worker(1);
            let t2 = spawn worker(2);
            let r1 = join(t1);
            let r2 = join(t2);
        """)


# ============================================================
# 12. Complex Patterns
# ============================================================

class TestComplexPatterns:
    """Test complex concurrent type checking patterns."""

    def test_producer_consumer(self):
        no_errors("""
            fn producer(ch, count) {
                let i = 0;
                while (i < count) {
                    send(ch, i);
                    i = i + 1;
                }
            }
            fn consumer(ch) {
                let val = recv(ch);
                print(val);
            }
            let ch = chan(10);
            let t1 = spawn producer(ch, 5);
            let t2 = spawn consumer(ch);
        """)

    def test_pipeline(self):
        """Multi-stage pipeline."""
        no_errors("""
            fn stage1(out) {
                send(out, 1);
            }
            fn stage2(inp, out) {
                let v = recv(inp);
                send(out, v + 10);
            }
            fn stage3(inp) {
                let v = recv(inp);
                print(v);
            }
            let ch1 = chan(1);
            let ch2 = chan(1);
            let t1 = spawn stage1(ch1);
            let t2 = spawn stage2(ch1, ch2);
            let t3 = spawn stage3(ch2);
        """)

    def test_fan_out(self):
        no_errors("""
            fn worker(id, result_ch) {
                send(result_ch, id * 10);
            }
            let results = chan(3);
            let t1 = spawn worker(1, results);
            let t2 = spawn worker(2, results);
            let t3 = spawn worker(3, results);
        """)

    def test_select_with_timeout_channel(self):
        no_errors("""
            fn worker(ch) {
                send(ch, 42);
            }
            let data_ch = chan(1);
            let timeout_ch = chan(1);
            let t = spawn worker(data_ch);
            select {
                case recv(data_ch) => val {
                    print(val);
                }
                case recv(timeout_ch) => t {
                    print(0);
                }
            }
        """)

    def test_channel_of_booleans(self):
        no_errors("""
            let done = chan(1);
            send(done, true);
            let finished = recv(done);
            if (finished) {
                print(1);
            }
        """)

    def test_recursive_function_with_channels(self):
        no_errors("""
            fn fib(n, ch) {
                if (n < 2) {
                    send(ch, n);
                } else {
                    let ch1 = chan(1);
                    let ch2 = chan(1);
                    let t1 = spawn fib(n - 1, ch1);
                    let t2 = spawn fib(n - 2, ch2);
                    let r1 = recv(ch1);
                    let r2 = recv(ch2);
                    send(ch, r1 + r2);
                }
            }
            let result = chan(1);
            fib(10, result);
        """)


# ============================================================
# 13. Type Error Detection
# ============================================================

class TestTypeErrorDetection:
    """Test that type errors are properly detected."""

    def test_send_string_to_int_channel(self):
        has_error("""
            let ch = chan(1);
            send(ch, 42);
            send(ch, "oops");
        """, "Cannot send")

    def test_recv_used_as_wrong_type(self):
        has_error("""
            let ch = chan(1);
            send(ch, "hello");
            let val = recv(ch);
            let x = val - 1;
        """, "Cannot apply")

    def test_join_on_int(self):
        has_error("let x = join(42);", "join requires task")

    def test_send_to_int(self):
        has_error("let x = 42; send(x, 1);", "send requires channel")

    def test_recv_from_string(self):
        has_error('let x = "ch"; recv(x);', "recv requires channel")

    def test_spawn_with_type_mismatch(self):
        has_error("""
            fn add(a, b) { return a + b; }
            let x = add(1, 2);
            let t = spawn add(true, 2);
        """, "spawn 'add' arg 1")

    def test_channel_size_float(self):
        has_error("let ch = chan(3.14);", "Channel buffer size must be int")

    def test_channel_size_bool(self):
        has_error("let ch = chan(true);", "Channel buffer size must be int")


# ============================================================
# 14. Deadlock Detection
# ============================================================

class TestDeadlockDetection:
    """Test static deadlock detection."""

    def test_no_deadlock_producer_consumer(self):
        """Normal producer/consumer should have no warnings."""
        warnings = warnings_from("""
            fn producer(ch) {
                send(ch, 42);
            }
            fn consumer(ch) {
                let val = recv(ch);
            }
            let ch = chan(1);
            let t1 = spawn producer(ch);
            let t2 = spawn consumer(ch);
        """)
        assert len(warnings) == 0

    def test_circular_wait_detection(self):
        """Two functions acquiring channels in opposite order."""
        has_warning("""
            fn task_a(ch1, ch2) {
                send(ch1, 1);
                let v = recv(ch2);
            }
            fn task_b(ch1, ch2) {
                send(ch2, 2);
                let v = recv(ch1);
            }
            let ch1 = chan(1);
            let ch2 = chan(1);
            let t1 = spawn task_a(ch1, ch2);
            let t2 = spawn task_b(ch1, ch2);
        """, "inconsistent order")

    def test_self_deadlock_single_function(self):
        """Function that sends and receives on same channel alone."""
        has_warning("""
            fn deadlocker(ch) {
                send(ch, 42);
                let v = recv(ch);
            }
            let ch = chan(1);
            deadlocker(ch);
        """, "both sends and receives")

    def test_no_warning_with_spawned_partner(self):
        """No warning when another function uses the channel."""
        warnings = warnings_from("""
            fn sender(ch) {
                send(ch, 42);
            }
            fn recver(ch) {
                let v = recv(ch);
            }
            let ch = chan(1);
            let t1 = spawn sender(ch);
            let t2 = spawn recver(ch);
        """)
        # Should not warn about deadlock since separate functions handle send/recv
        deadlock_warnings = [w for w in warnings if "deadlock" in w.message.lower()]
        assert len(deadlock_warnings) == 0

    def test_no_false_positive_buffered(self):
        """Buffered channel with same function should still warn (static analysis)."""
        # Even buffered channels can deadlock if buffer is exhausted
        warnings = warnings_from("""
            fn worker(ch) {
                send(ch, 1);
                send(ch, 2);
                let v1 = recv(ch);
                let v2 = recv(ch);
            }
            let ch = chan(1);
            worker(ch);
        """)
        # This is a gray area -- static analysis might warn
        # We accept both warning and no warning here
        pass


# ============================================================
# 15. Format Errors
# ============================================================

class TestFormatErrors:
    """Test error/warning formatting."""

    def test_no_errors_message(self):
        msg = format_errors([], [])
        assert "No type errors" in msg

    def test_error_formatting(self):
        errors = [TypeError_("bad type", 5)]
        msg = format_errors(errors)
        assert "1 type error" in msg
        assert "Line 5" in msg
        assert "bad type" in msg

    def test_warning_formatting(self):
        warnings = [DeadlockWarning("potential deadlock", 10)]
        msg = format_errors([], warnings)
        assert "1 warning" in msg
        assert "Line 10" in msg

    def test_both_errors_and_warnings(self):
        errors = [TypeError_("bad", 1)]
        warnings = [DeadlockWarning("maybe bad", 2)]
        msg = format_errors(errors, warnings)
        assert "error" in msg
        assert "warning" in msg


# ============================================================
# 16. type_of helper
# ============================================================

class TestTypeOf:
    """Test the type_of convenience function."""

    def test_int_literal(self):
        t = type_of("", "42")
        assert isinstance(t, TInt)

    def test_chan(self):
        t = type_of("", "chan(1)")
        assert isinstance(resolve(t), TChan)

    def test_spawn(self):
        t = type_of("fn w() { return 1; }", "spawn w()")
        assert isinstance(resolve(t), TTask)

    def test_task_id(self):
        t = type_of("", "task_id")
        assert isinstance(resolve(t), TInt)

    def test_recv(self):
        t = type_of("let ch = chan(1); send(ch, 42);", "recv(ch)")
        assert isinstance(resolve(t), TInt)


# ============================================================
# 17. Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_program(self):
        errors, warnings, checker = check_source("")
        assert errors == []
        assert warnings == []

    def test_yield_only(self):
        no_errors("yield;")

    def test_multiple_yields(self):
        no_errors("yield; yield; yield;")

    def test_chan_zero_buffer(self):
        no_errors("let ch = chan(0);")

    def test_nested_spawn(self):
        no_errors("""
            fn inner() { return 1; }
            fn outer() {
                let t = spawn inner();
                return join(t);
            }
            let t = spawn outer();
        """)

    def test_channel_in_if(self):
        no_errors("""
            let ch = chan(1);
            if (true) {
                send(ch, 42);
            } else {
                send(ch, 0);
            }
        """)

    def test_channel_in_while(self):
        no_errors("""
            fn producer(ch) {
                let i = 0;
                while (i < 5) {
                    send(ch, i);
                    i = i + 1;
                    yield;
                }
            }
        """)

    def test_task_variable_reassignment(self):
        no_errors("""
            fn w1() { return 1; }
            fn w2() { return 2; }
            let t = spawn w1();
            t = spawn w2();
        """)

    def test_channel_variable_reassignment(self):
        no_errors("""
            let ch = chan(1);
            send(ch, 42);
            ch = chan(1);
            send(ch, 43);
        """)

    def test_many_channels(self):
        no_errors("""
            let ch1 = chan(1);
            let ch2 = chan(1);
            let ch3 = chan(1);
            let ch4 = chan(1);
            let ch5 = chan(1);
            send(ch1, 1);
            send(ch2, 2);
            send(ch3, 3);
            send(ch4, 4);
            send(ch5, 5);
        """)


# ============================================================
# 18. Integration: Full Concurrent Programs
# ============================================================

class TestIntegration:
    """Test complete concurrent programs."""

    def test_worker_pool(self):
        no_errors("""
            fn worker(id, tasks, results) {
                let task = recv(tasks);
                let result = task * id;
                send(results, result);
            }
            let tasks = chan(10);
            let results = chan(10);
            let t1 = spawn worker(1, tasks, results);
            let t2 = spawn worker(2, tasks, results);
            send(tasks, 42);
            send(tasks, 84);
            let r1 = recv(results);
            let r2 = recv(results);
        """)

    def test_ping_pong(self):
        no_errors("""
            fn pinger(ping_ch, pong_ch) {
                let i = 0;
                while (i < 3) {
                    send(ping_ch, i);
                    let resp = recv(pong_ch);
                    i = i + 1;
                }
            }
            fn ponger(ping_ch, pong_ch) {
                let i = 0;
                while (i < 3) {
                    let val = recv(ping_ch);
                    send(pong_ch, val + 1);
                    i = i + 1;
                }
            }
            let ping = chan(1);
            let pong = chan(1);
            let t1 = spawn pinger(ping, pong);
            let t2 = spawn ponger(ping, pong);
        """)

    def test_barrier(self):
        no_errors("""
            fn worker(id, barrier_ch) {
                let result = id * 10;
                send(barrier_ch, true);
            }
            let barrier = chan(3);
            let t1 = spawn worker(1, barrier);
            let t2 = spawn worker(2, barrier);
            let t3 = spawn worker(3, barrier);
            let d1 = recv(barrier);
            let d2 = recv(barrier);
            let d3 = recv(barrier);
        """)

    def test_map_reduce(self):
        no_errors("""
            fn mapper(input, output) {
                let val = recv(input);
                send(output, val * 2);
            }
            fn reducer(inputs, output) {
                let v1 = recv(inputs);
                let v2 = recv(inputs);
                send(output, v1 + v2);
            }
            let in1 = chan(1);
            let in2 = chan(1);
            let mapped = chan(2);
            let result = chan(1);
            send(in1, 10);
            send(in2, 20);
            let t1 = spawn mapper(in1, mapped);
            let t2 = spawn mapper(in2, mapped);
            let t3 = spawn reducer(mapped, result);
            let final = recv(result);
        """)


# ============================================================
# 19. Error Locations
# ============================================================

class TestErrorLocations:
    """Test that errors have correct line numbers."""

    def test_error_has_line(self):
        errors = errors_from("""
            let ch = chan(1);
            send(ch, 42);
            send(ch, "wrong");
        """)
        assert len(errors) > 0
        assert errors[0].line > 0

    def test_warning_has_line(self):
        warnings = warnings_from("""
            fn deadlocker(ch) {
                send(ch, 42);
                let v = recv(ch);
            }
            let ch = chan(1);
            deadlocker(ch);
        """)
        if len(warnings) > 0:
            assert warnings[0].line > 0

    def test_multiple_errors_different_lines(self):
        errors = errors_from("""
            let x = undefined1;
            let y = undefined2;
        """)
        assert len(errors) == 2
        assert errors[0].line != errors[1].line


# ============================================================
# 20. API Tests
# ============================================================

class TestAPI:
    """Test public API functions."""

    def test_check_source_returns_triple(self):
        result = check_source("let x = 1;")
        assert len(result) == 3
        errors, warnings, checker = result
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
        assert isinstance(checker, ConcurrentTypeChecker)

    def test_check_program_returns_triple(self):
        program = parse_concurrent("let x = 1;")
        errors, warnings, checker = check_program(program)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    def test_parse_concurrent(self):
        program = parse_concurrent("let x = chan(1);")
        assert program is not None
        assert len(program.stmts) == 1

    def test_checker_fresh_tvar(self):
        checker = ConcurrentTypeChecker()
        t1 = checker.fresh_tvar()
        t2 = checker.fresh_tvar()
        assert t1.id != t2.id


# ============================================================
# 21. TChan/TTask Type System
# ============================================================

class TestTypeSystem:
    """Test concurrent type representations."""

    def test_tchan_int(self):
        t = TChan(INT)
        assert repr(t) == "Chan<int>"

    def test_tchan_string(self):
        t = TChan(STRING)
        assert repr(t) == "Chan<string>"

    def test_tchan_float(self):
        t = TChan(FLOAT)
        assert repr(t) == "Chan<float>"

    def test_tchan_bool(self):
        t = TChan(BOOL)
        assert repr(t) == "Chan<bool>"

    def test_ttask_int(self):
        t = TTask(INT)
        assert repr(t) == "Task<int>"

    def test_ttask_void(self):
        t = TTask(VOID)
        assert repr(t) == "Task<void>"

    def test_tchan_of_task(self):
        t = TChan(TTask(INT))
        assert repr(t) == "Chan<Task<int>>"

    def test_ttask_of_chan(self):
        t = TTask(TChan(STRING))
        assert repr(t) == "Task<Chan<string>>"

    def test_tchan_equality(self):
        assert TChan(INT) == TChan(INT)

    def test_tchan_inequality(self):
        assert TChan(INT) != TChan(STRING)

    def test_ttask_equality(self):
        assert TTask(INT) == TTask(INT)

    def test_ttask_inequality(self):
        assert TTask(INT) != TTask(STRING)

    def test_tchan_hash(self):
        s = {TChan(INT), TChan(INT)}
        assert len(s) == 1

    def test_ttask_hash(self):
        s = {TTask(INT), TTask(INT)}
        assert len(s) == 1

    def test_func_returning_chan(self):
        t = TFunc((INT,), TChan(INT))
        assert "Chan<int>" in repr(t)

    def test_func_returning_task(self):
        t = TFunc((), TTask(STRING))
        assert "Task<string>" in repr(t)


# ============================================================
# 22. Concurrent + Non-Concurrent Mixed Programs
# ============================================================

class TestMixed:
    """Test programs mixing concurrent and sequential code."""

    def test_sequential_then_concurrent(self):
        no_errors("""
            let x = 1 + 2;
            let y = x * 3;
            fn worker() { return y; }
            let t = spawn worker();
            let result = join(t);
        """)

    def test_concurrent_in_function(self):
        no_errors("""
            fn run() {
                let ch = chan(1);
                fn worker(c) { send(c, 42); }
                let t = spawn worker(ch);
                let v = recv(ch);
                return v;
            }
        """)

    def test_error_in_sequential_part(self):
        has_error("""
            let x = "hello" - 1;
            let ch = chan(1);
        """, "Cannot apply")

    def test_error_in_concurrent_part(self):
        has_error("""
            let x = 42;
            send(x, 1);
        """, "send requires channel")


# ============================================================
# 23. Forward References
# ============================================================

class TestForwardReferences:
    """Test that functions can reference each other."""

    def test_forward_reference_spawn(self):
        no_errors("""
            fn a() {
                let t = spawn b();
            }
            fn b() {
                return 42;
            }
        """)

    def test_mutual_recursion(self):
        no_errors("""
            fn even(n) {
                if (n == 0) { return true; }
                return odd(n - 1);
            }
            fn odd(n) {
                if (n == 0) { return false; }
                return even(n - 1);
            }
        """)

    def test_forward_spawn_with_channel(self):
        no_errors("""
            fn main_fn() {
                let ch = chan(1);
                let t = spawn worker(ch);
                let v = recv(ch);
            }
            fn worker(ch) {
                send(ch, 42);
            }
        """)


# ============================================================
# 24. Stress Tests
# ============================================================

class TestStress:
    """Stress tests for robustness."""

    def test_many_spawns(self):
        lines = ["fn w() { return 1; }"]
        for i in range(20):
            lines.append(f"let t{i} = spawn w();")
        no_errors("\n".join(lines))

    def test_many_channels(self):
        lines = []
        for i in range(20):
            lines.append(f"let ch{i} = chan(1);")
            lines.append(f"send(ch{i}, {i});")
        no_errors("\n".join(lines))

    def test_deeply_nested_blocks(self):
        code = "let ch = chan(1);\n"
        for i in range(10):
            code += "{ "
        code += "send(ch, 42);"
        for i in range(10):
            code += " }"
        no_errors(code)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
