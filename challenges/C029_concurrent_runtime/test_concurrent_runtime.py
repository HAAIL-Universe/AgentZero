"""
Tests for C029: Concurrent Task Runtime
Challenge C029 -- AgentZero Session 030

Tests cover:
  1. Base VM compatibility (arithmetic, variables, control flow, functions)
  2. Task spawning and lifecycle
  3. Channels (buffered, blocking, multi-producer/consumer)
  4. Cooperative yielding and auto-preemption
  5. Join and task results
  6. Select statement
  7. Task identity (task_id)
  8. Error handling
  9. Complex patterns (pipeline, fan-out/fan-in, ping-pong, barrier)
  10. Edge cases and stress tests
"""

import pytest
from concurrent_runtime import (
    execute_concurrent, compile_concurrent,
    conc_lex, ConcParser, ConcCompiler, ConcurrentVM,
    Channel, Task, TaskState, ConcOp,
    LexError, ParseError, CompileError, VMError,
)


# ============================================================
# 1. Base VM Compatibility
# ============================================================

class TestBaseCompatibility:
    """Verify all C010 features still work through the concurrent runtime."""

    def test_integer_arithmetic(self):
        r = execute_concurrent("let x = 3 + 4 * 2; print(x);")
        assert r['output'] == ['11']

    def test_float_arithmetic(self):
        r = execute_concurrent("let x = 3.14 * 2.0; print(x);")
        assert r['output'] == ['6.28']

    def test_string_literal(self):
        r = execute_concurrent('print("hello world");')
        assert r['output'] == ['hello world']

    def test_boolean_literals(self):
        r = execute_concurrent("print(true); print(false);")
        assert r['output'] == ['true', 'false']

    def test_comparison_operators(self):
        r = execute_concurrent("""
            print(1 < 2);
            print(3 > 4);
            print(5 == 5);
            print(6 != 6);
            print(7 <= 7);
            print(8 >= 9);
        """)
        assert r['output'] == ['true', 'false', 'true', 'false', 'true', 'false']

    def test_logical_operators(self):
        r = execute_concurrent("print(true and false); print(true or false); print(not true);")
        assert r['output'] == ['false', 'true', 'false']

    def test_variables(self):
        r = execute_concurrent("let x = 10; x = x + 5; print(x);")
        assert r['output'] == ['15']

    def test_if_else(self):
        r = execute_concurrent("""
            let x = 10;
            if (x > 5) {
                print("big");
            } else {
                print("small");
            }
        """)
        assert r['output'] == ['big']

    def test_if_no_else(self):
        r = execute_concurrent("""
            let x = 3;
            if (x > 5) {
                print("big");
            }
            print("done");
        """)
        assert r['output'] == ['done']

    def test_while_loop(self):
        r = execute_concurrent("""
            let i = 0;
            while (i < 5) {
                i = i + 1;
            }
            print(i);
        """)
        assert r['output'] == ['5']

    def test_function_definition_and_call(self):
        r = execute_concurrent("""
            fn add(a, b) {
                return a + b;
            }
            let result = add(3, 4);
            print(result);
        """)
        assert r['output'] == ['7']

    def test_recursive_function(self):
        r = execute_concurrent("""
            fn fact(n) {
                if (n <= 1) {
                    return 1;
                }
                return n * fact(n - 1);
            }
            print(fact(6));
        """)
        assert r['output'] == ['720']

    def test_nested_function_calls(self):
        r = execute_concurrent("""
            fn double(x) { return x * 2; }
            fn triple(x) { return x * 3; }
            print(double(triple(5)));
        """)
        assert r['output'] == ['30']

    def test_modulo(self):
        r = execute_concurrent("print(17 % 5);")
        assert r['output'] == ['2']

    def test_unary_negation(self):
        r = execute_concurrent("let x = 5; print(-x);")
        assert r['output'] == ['-5']

    def test_multiple_variables(self):
        r = execute_concurrent("""
            let a = 1;
            let b = 2;
            let c = a + b;
            print(c);
        """)
        assert r['output'] == ['3']

    def test_string_concatenation(self):
        r = execute_concurrent('print("hello" + " " + "world");')
        assert r['output'] == ['hello world']


# ============================================================
# 2. Task Spawning and Lifecycle
# ============================================================

class TestTaskSpawning:

    def test_spawn_single_task(self):
        r = execute_concurrent("""
            fn worker() {
                print("from worker");
            }
            let t = spawn worker();
            join(t);
        """)
        assert 'from worker' in r['output']

    def test_spawn_returns_task_id(self):
        r = execute_concurrent("""
            fn noop() { }
            let t = spawn noop();
            print(t);
            join(t);
        """)
        # Task 0 is main, spawned is 1
        assert r['output'] == ['1']

    def test_spawn_multiple_tasks(self):
        r = execute_concurrent("""
            fn worker(id) {
                print(id);
            }
            let t1 = spawn worker(1);
            let t2 = spawn worker(2);
            let t3 = spawn worker(3);
            join(t1);
            join(t2);
            join(t3);
        """)
        # All three should print
        assert sorted(r['output']) == ['1', '2', '3']

    def test_spawn_with_args(self):
        r = execute_concurrent("""
            fn adder(a, b) {
                print(a + b);
            }
            let t = spawn adder(10, 20);
            join(t);
        """)
        assert '30' in r['output']

    def test_task_count(self):
        r = execute_concurrent("""
            fn noop() { }
            let t1 = spawn noop();
            let t2 = spawn noop();
            join(t1);
            join(t2);
        """)
        assert len(r['tasks']) == 3  # main + 2 spawned

    def test_task_states_after_completion(self):
        r = execute_concurrent("""
            fn noop() { }
            let t = spawn noop();
            join(t);
        """)
        for tid, info in r['tasks'].items():
            assert info['state'] == 'COMPLETED'

    def test_spawn_task_accesses_parent_env(self):
        r = execute_concurrent("""
            let shared = 42;
            fn reader() {
                print(shared);
            }
            let t = spawn reader();
            join(t);
        """)
        assert '42' in r['output']

    def test_spawned_task_independent_env(self):
        r = execute_concurrent("""
            let x = 1;
            fn modifier() {
                x = 99;
                print(x);
            }
            let t = spawn modifier();
            join(t);
            print(x);
        """)
        # Spawned task modifies its own copy
        assert '99' in r['output']
        assert '1' in r['output']


# ============================================================
# 3. Channels
# ============================================================

class TestChannels:

    def test_create_channel(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, 42);
            let val = recv(ch);
            print(val);
        """)
        assert r['output'] == ['42']

    def test_channel_default_buffer_size(self):
        r = execute_concurrent("""
            let ch = chan();
            send(ch, 7);
            print(recv(ch));
        """)
        assert r['output'] == ['7']

    def test_channel_multiple_values(self):
        r = execute_concurrent("""
            let ch = chan(3);
            send(ch, 1);
            send(ch, 2);
            send(ch, 3);
            print(recv(ch));
            print(recv(ch));
            print(recv(ch));
        """)
        assert r['output'] == ['1', '2', '3']

    def test_channel_fifo_order(self):
        r = execute_concurrent("""
            let ch = chan(5);
            send(ch, 10);
            send(ch, 20);
            send(ch, 30);
            let a = recv(ch);
            let b = recv(ch);
            let c = recv(ch);
            print(a);
            print(b);
            print(c);
        """)
        assert r['output'] == ['10', '20', '30']

    def test_channel_between_tasks(self):
        r = execute_concurrent("""
            fn producer(ch) {
                send(ch, 100);
            }
            let ch = chan(1);
            let t = spawn producer(ch);
            let val = recv(ch);
            print(val);
            join(t);
        """)
        assert '100' in r['output']

    def test_channel_blocking_send(self):
        """Send blocks when buffer is full, unblocks when consumer reads."""
        r = execute_concurrent("""
            fn producer(ch) {
                send(ch, 1);
                send(ch, 2);  // blocks until consumer reads
                send(ch, 3);
            }
            let ch = chan(1);
            let t = spawn producer(ch);
            let a = recv(ch);
            let b = recv(ch);
            let c = recv(ch);
            print(a);
            print(b);
            print(c);
            join(t);
        """)
        assert r['output'] == ['1', '2', '3']

    def test_channel_blocking_recv(self):
        """Recv blocks when buffer is empty, unblocks when producer sends."""
        r = execute_concurrent("""
            fn delayed_sender(ch) {
                yield;
                yield;
                send(ch, 99);
            }
            let ch = chan(1);
            let t = spawn delayed_sender(ch);
            let val = recv(ch);  // blocks until sender runs
            print(val);
            join(t);
        """)
        assert '99' in r['output']

    def test_channel_send_string(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, "hello");
            print(recv(ch));
        """)
        assert r['output'] == ['hello']

    def test_channel_send_boolean(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, true);
            print(recv(ch));
        """)
        assert r['output'] == ['true']

    def test_channel_large_buffer(self):
        r = execute_concurrent("""
            let ch = chan(10);
            let i = 0;
            while (i < 10) {
                send(ch, i);
                i = i + 1;
            }
            let sum = 0;
            let j = 0;
            while (j < 10) {
                sum = sum + recv(ch);
                j = j + 1;
            }
            print(sum);
        """)
        assert r['output'] == ['45']  # 0+1+...+9

    def test_multiple_channels(self):
        r = execute_concurrent("""
            let ch1 = chan(1);
            let ch2 = chan(1);
            send(ch1, 10);
            send(ch2, 20);
            let a = recv(ch1);
            let b = recv(ch2);
            print(a + b);
        """)
        assert r['output'] == ['30']


# ============================================================
# 4. Cooperative Yielding and Auto-preemption
# ============================================================

class TestYielding:

    def test_explicit_yield(self):
        r = execute_concurrent("""
            fn task1() {
                print("a");
                yield;
                print("c");
            }
            fn task2() {
                print("b");
                yield;
                print("d");
            }
            let t1 = spawn task1();
            let t2 = spawn task2();
            join(t1);
            join(t2);
        """)
        # Both tasks should complete
        assert sorted(r['output']) == ['a', 'b', 'c', 'd']

    def test_yield_preserves_state(self):
        r = execute_concurrent("""
            fn counter() {
                let x = 0;
                x = x + 1;
                yield;
                x = x + 1;
                yield;
                x = x + 1;
                print(x);
            }
            let t = spawn counter();
            join(t);
        """)
        assert '3' in r['output']

    def test_auto_preemption(self):
        """Tasks with long loops get auto-preempted for fairness."""
        r = execute_concurrent("""
            fn busy_worker(id) {
                let i = 0;
                while (i < 100) {
                    i = i + 1;
                }
                print(id);
            }
            let t1 = spawn busy_worker(1);
            let t2 = spawn busy_worker(2);
            join(t1);
            join(t2);
        """, max_steps_per_task=50)
        assert sorted(r['output']) == ['1', '2']

    def test_yield_in_loop(self):
        r = execute_concurrent("""
            fn cooperative(id) {
                let i = 0;
                while (i < 3) {
                    print(id);
                    yield;
                    i = i + 1;
                }
            }
            let t1 = spawn cooperative(1);
            let t2 = spawn cooperative(2);
            join(t1);
            join(t2);
        """)
        assert r['output'].count('1') == 3
        assert r['output'].count('2') == 3

    def test_main_continues_after_spawn(self):
        r = execute_concurrent("""
            fn slow() {
                yield;
                yield;
                print("slow done");
            }
            let t = spawn slow();
            print("main continues");
            join(t);
        """)
        assert r['output'][0] == 'main continues'


# ============================================================
# 5. Join and Task Results
# ============================================================

class TestJoin:

    def test_join_returns_result(self):
        r = execute_concurrent("""
            fn compute() {
                return 42;
            }
            let t = spawn compute();
            let result = join(t);
            print(result);
        """)
        assert '42' in r['output']

    def test_join_waits_for_completion(self):
        r = execute_concurrent("""
            fn delayed() {
                yield;
                yield;
                yield;
                return 99;
            }
            let t = spawn delayed();
            let result = join(t);
            print(result);
        """)
        assert '99' in r['output']

    def test_join_already_completed(self):
        r = execute_concurrent("""
            fn quick() {
                return 7;
            }
            let t = spawn quick();
            yield;  // give task time to complete
            yield;
            let result = join(t);
            print(result);
        """)
        assert '7' in r['output']

    def test_join_multiple_tasks(self):
        r = execute_concurrent("""
            fn make_val(x) {
                return x * 10;
            }
            let t1 = spawn make_val(1);
            let t2 = spawn make_val(2);
            let t3 = spawn make_val(3);
            let r1 = join(t1);
            let r2 = join(t2);
            let r3 = join(t3);
            print(r1 + r2 + r3);
        """)
        assert '60' in r['output']

    def test_join_chain(self):
        """Task joins another task which joins another."""
        r = execute_concurrent("""
            fn inner() {
                return 5;
            }
            fn middle() {
                let t = spawn inner();
                let val = join(t);
                return val * 2;
            }
            let t = spawn middle();
            let result = join(t);
            print(result);
        """)
        assert '10' in r['output']


# ============================================================
# 6. Select Statement
# ============================================================

class TestSelect:

    def test_select_recv_ready(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, 42);
            select {
                case recv(ch) => val {
                    print(val);
                }
                default => {
                    print("default");
                }
            }
        """)
        assert r['output'] == ['42']

    def test_select_default_when_empty(self):
        r = execute_concurrent("""
            let ch = chan(1);
            select {
                case recv(ch) => val {
                    print(val);
                }
                default => {
                    print("nothing ready");
                }
            }
        """)
        assert r['output'] == ['nothing ready']

    def test_select_send_ready(self):
        r = execute_concurrent("""
            let ch = chan(1);
            select {
                case send(ch, 10) => {
                    print("sent");
                }
                default => {
                    print("not sent");
                }
            }
            print(recv(ch));
        """)
        assert r['output'] == ['sent', '10']

    def test_select_send_full_channel(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, 1);  // fill buffer
            select {
                case send(ch, 2) => {
                    print("sent");
                }
                default => {
                    print("full");
                }
            }
        """)
        assert r['output'] == ['full']

    def test_select_multiple_cases(self):
        r = execute_concurrent("""
            let ch1 = chan(1);
            let ch2 = chan(1);
            send(ch2, 20);
            select {
                case recv(ch1) => val {
                    print("ch1");
                }
                case recv(ch2) => val {
                    print(val);
                }
                default => {
                    print("default");
                }
            }
        """)
        assert r['output'] == ['20']

    def test_select_without_default(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, 5);
            select {
                case recv(ch) => val {
                    print(val);
                }
            }
        """)
        assert r['output'] == ['5']

    def test_select_recv_no_binding(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, 99);
            select {
                case recv(ch) => {
                    print("got it");
                }
                default => {
                    print("empty");
                }
            }
        """)
        assert r['output'] == ['got it']


# ============================================================
# 7. Task Identity
# ============================================================

class TestTaskId:

    def test_main_task_id(self):
        r = execute_concurrent("""
            print(task_id);
        """)
        assert r['output'] == ['0']

    def test_spawned_task_id(self):
        r = execute_concurrent("""
            fn show_id() {
                print(task_id);
            }
            let t = spawn show_id();
            join(t);
        """)
        assert '1' in r['output']

    def test_task_id_unique(self):
        r = execute_concurrent("""
            fn show_id() {
                print(task_id);
            }
            let t1 = spawn show_id();
            let t2 = spawn show_id();
            let t3 = spawn show_id();
            join(t1);
            join(t2);
            join(t3);
        """)
        ids = sorted(r['output'])
        assert len(ids) == 3
        assert len(set(ids)) == 3  # all unique


# ============================================================
# 8. Error Handling
# ============================================================

class TestErrors:

    def test_division_by_zero(self):
        with pytest.raises(VMError, match="Division by zero"):
            execute_concurrent("let x = 1 / 0;")

    def test_undefined_variable(self):
        with pytest.raises(VMError, match="Undefined variable"):
            execute_concurrent("print(undefined_var);")

    def test_spawn_non_function(self):
        with pytest.raises(VMError, match="Cannot spawn non-function"):
            execute_concurrent("""
                let x = 5;
                let t = spawn x();
            """)

    def test_send_to_non_channel(self):
        with pytest.raises(VMError, match="Cannot send to non-channel"):
            execute_concurrent("send(42, 10);")

    def test_recv_from_non_channel(self):
        with pytest.raises(VMError, match="Cannot recv from non-channel"):
            execute_concurrent("recv(42);")

    def test_join_invalid_task_id(self):
        with pytest.raises(VMError, match="Unknown task ID"):
            execute_concurrent("join(999);")

    def test_channel_invalid_buffer_size(self):
        with pytest.raises(VMError, match="positive integer"):
            execute_concurrent("let ch = chan(-1);")

    def test_total_step_limit(self):
        with pytest.raises(VMError, match="execution limit"):
            execute_concurrent("""
                while (true) { }
            """, max_total_steps=100)

    def test_arity_mismatch_spawn(self):
        with pytest.raises(VMError, match="expects"):
            execute_concurrent("""
                fn f(a, b) { return a + b; }
                let t = spawn f(1);
            """)

    def test_parse_error_unexpected_token(self):
        with pytest.raises(ParseError):
            execute_concurrent("let = 5;")

    def test_lex_error_unexpected_char(self):
        with pytest.raises(LexError):
            execute_concurrent("let x = @;")

    def test_task_error_doesnt_crash_main(self):
        """A spawned task can fail without crashing the main task."""
        r = execute_concurrent("""
            fn bad() {
                let x = 1 / 0;
            }
            let t = spawn bad();
            yield;
            yield;
            print("main ok");
        """)
        assert 'main ok' in r['output']
        assert r['tasks'][1]['state'] == 'FAILED'

    def test_join_failed_task(self):
        r = execute_concurrent("""
            fn bad() {
                let x = 1 / 0;
            }
            let t = spawn bad();
            let result = join(t);
            print("joined");
        """)
        assert 'joined' in r['output']


# ============================================================
# 9. Complex Patterns
# ============================================================

class TestComplexPatterns:

    def test_producer_consumer(self):
        """Classic producer-consumer with channel."""
        r = execute_concurrent("""
            fn producer(ch) {
                let i = 0;
                while (i < 5) {
                    send(ch, i);
                    i = i + 1;
                }
            }
            fn consumer(ch) {
                let sum = 0;
                let j = 0;
                while (j < 5) {
                    sum = sum + recv(ch);
                    j = j + 1;
                }
                print(sum);
            }
            let ch = chan(2);
            let p = spawn producer(ch);
            let c = spawn consumer(ch);
            join(p);
            join(c);
        """)
        assert '10' in r['output']  # 0+1+2+3+4

    def test_ping_pong(self):
        """Two tasks bounce a value back and forth."""
        r = execute_concurrent("""
            fn ping(ch_in, ch_out) {
                let i = 0;
                while (i < 3) {
                    let val = recv(ch_in);
                    send(ch_out, val + 1);
                    i = i + 1;
                }
            }
            fn pong(ch_in, ch_out) {
                let i = 0;
                while (i < 3) {
                    let val = recv(ch_in);
                    send(ch_out, val + 1);
                    i = i + 1;
                }
            }
            let a_to_b = chan(1);
            let b_to_a = chan(1);
            let tp = spawn ping(b_to_a, a_to_b);
            let tq = spawn pong(a_to_b, b_to_a);
            send(b_to_a, 0);  // start with 0
            join(tp);
            join(tq);
            // After 3 rounds each: 0->1->2->3->4->5->6
            // Final value 6 is in b_to_a (pong's last send)
            let final_val = recv(b_to_a);
            print(final_val);
        """)
        assert '6' in r['output']

    def test_fan_out(self):
        """One producer, multiple consumers."""
        r = execute_concurrent("""
            fn worker(ch, id) {
                let val = recv(ch);
                print(id * 100 + val);
            }
            let ch = chan(3);
            let t1 = spawn worker(ch, 1);
            let t2 = spawn worker(ch, 2);
            let t3 = spawn worker(ch, 3);
            send(ch, 10);
            send(ch, 20);
            send(ch, 30);
            join(t1);
            join(t2);
            join(t3);
        """)
        vals = sorted([int(x) for x in r['output']])
        assert len(vals) == 3
        # Each worker gets one value: id*100 + val
        # Workers get values in order: w1=10, w2=20, w3=30
        assert vals == [110, 220, 330]

    def test_fan_in(self):
        """Multiple producers, one consumer channel."""
        r = execute_concurrent("""
            fn producer(ch, val) {
                send(ch, val);
            }
            let ch = chan(3);
            let t1 = spawn producer(ch, 10);
            let t2 = spawn producer(ch, 20);
            let t3 = spawn producer(ch, 30);
            join(t1);
            join(t2);
            join(t3);
            let sum = recv(ch) + recv(ch) + recv(ch);
            print(sum);
        """)
        assert r['output'] == ['60']

    def test_pipeline(self):
        """Data flows through a pipeline of tasks."""
        r = execute_concurrent("""
            fn stage1(in_ch, out_ch) {
                let val = recv(in_ch);
                send(out_ch, val * 2);
            }
            fn stage2(in_ch, out_ch) {
                let val = recv(in_ch);
                send(out_ch, val + 10);
            }
            fn stage3(in_ch, out_ch) {
                let val = recv(in_ch);
                send(out_ch, val * 3);
            }
            let ch1 = chan(1);
            let ch2 = chan(1);
            let ch3 = chan(1);
            let ch4 = chan(1);
            let t1 = spawn stage1(ch1, ch2);
            let t2 = spawn stage2(ch2, ch3);
            let t3 = spawn stage3(ch3, ch4);
            send(ch1, 5);
            let result = recv(ch4);
            print(result);
            join(t1);
            join(t2);
            join(t3);
        """)
        # 5 * 2 = 10, + 10 = 20, * 3 = 60
        assert '60' in r['output']

    def test_barrier_sync(self):
        """Multiple tasks synchronize at a barrier using channels."""
        r = execute_concurrent("""
            fn worker(done_ch, id) {
                let i = 0;
                while (i < 10) {
                    i = i + 1;
                }
                send(done_ch, id);
            }
            let done = chan(3);
            let t1 = spawn worker(done, 1);
            let t2 = spawn worker(done, 2);
            let t3 = spawn worker(done, 3);
            // Wait for all three
            let r1 = recv(done);
            let r2 = recv(done);
            let r3 = recv(done);
            print(r1 + r2 + r3);
            join(t1);
            join(t2);
            join(t3);
        """)
        assert r['output'] == ['6']

    def test_mutex_pattern(self):
        """Use a channel as a mutex (buffer=1, send=lock, recv=unlock)."""
        r = execute_concurrent("""
            fn critical_section(mutex, result_ch, id) {
                recv(mutex);  // acquire lock
                // critical section -- only one task at a time
                print(id);
                send(mutex, 0);  // release lock
                send(result_ch, 1);
            }
            let mutex = chan(1);
            send(mutex, 0);  // initialize mutex (unlocked)
            let done = chan(3);
            let t1 = spawn critical_section(mutex, done, 1);
            let t2 = spawn critical_section(mutex, done, 2);
            let t3 = spawn critical_section(mutex, done, 3);
            recv(done);
            recv(done);
            recv(done);
            join(t1);
            join(t2);
            join(t3);
        """)
        assert sorted(r['output']) == ['1', '2', '3']

    def test_map_reduce(self):
        """Map work to tasks, reduce results."""
        r = execute_concurrent("""
            fn square(result_ch, x) {
                send(result_ch, x * x);
            }
            let results = chan(4);
            let t1 = spawn square(results, 1);
            let t2 = spawn square(results, 2);
            let t3 = spawn square(results, 3);
            let t4 = spawn square(results, 4);
            join(t1);
            join(t2);
            join(t3);
            join(t4);
            let sum = recv(results) + recv(results) + recv(results) + recv(results);
            print(sum);
        """)
        assert r['output'] == ['30']  # 1+4+9+16

    def test_recursive_spawn(self):
        """Task spawns sub-tasks."""
        r = execute_concurrent("""
            fn fib_task(n, result_ch) {
                if (n <= 1) {
                    send(result_ch, n);
                } else {
                    let ch1 = chan(1);
                    let ch2 = chan(1);
                    let t1 = spawn fib_task(n - 1, ch1);
                    let t2 = spawn fib_task(n - 2, ch2);
                    let r1 = recv(ch1);
                    let r2 = recv(ch2);
                    join(t1);
                    join(t2);
                    send(result_ch, r1 + r2);
                }
            }
            let out = chan(1);
            let t = spawn fib_task(6, out);
            let result = recv(out);
            join(t);
            print(result);
        """)
        assert '8' in r['output']  # fib(6) = 8


# ============================================================
# 10. Edge Cases and Stress Tests
# ============================================================

class TestEdgeCases:

    def test_spawn_no_args(self):
        r = execute_concurrent("""
            fn hello() {
                print("hello");
            }
            let t = spawn hello();
            join(t);
        """)
        assert 'hello' in r['output']

    def test_channel_buffer_exactly_full(self):
        r = execute_concurrent("""
            let ch = chan(3);
            send(ch, 1);
            send(ch, 2);
            send(ch, 3);
            print(recv(ch));
            print(recv(ch));
            print(recv(ch));
        """)
        assert r['output'] == ['1', '2', '3']

    def test_many_yields(self):
        r = execute_concurrent("""
            fn yielder() {
                yield;
                yield;
                yield;
                yield;
                yield;
                return 42;
            }
            let t = spawn yielder();
            let r = join(t);
            print(r);
        """)
        assert '42' in r['output']

    def test_send_recv_same_task(self):
        r = execute_concurrent("""
            let ch = chan(1);
            send(ch, 123);
            let val = recv(ch);
            print(val);
        """)
        assert r['output'] == ['123']

    def test_channel_passes_function(self):
        """Channels can pass any value, including function objects."""
        r = execute_concurrent("""
            fn doubler(x) {
                return x * 2;
            }
            let ch = chan(1);
            send(ch, doubler);
            let f = recv(ch);
            print(f(5));
        """)
        assert '10' in r['output']

    def test_nested_spawn(self):
        r = execute_concurrent("""
            fn inner() {
                return 7;
            }
            fn outer() {
                let t = spawn inner();
                return join(t);
            }
            let t = spawn outer();
            let r = join(t);
            print(r);
        """)
        assert '7' in r['output']

    def test_multiple_joins_same_task(self):
        """Joining a completed task multiple times should be safe."""
        r = execute_concurrent("""
            fn compute() {
                return 42;
            }
            let t = spawn compute();
            let r1 = join(t);
            let r2 = join(t);
            print(r1);
            print(r2);
        """)
        assert r['output'] == ['42', '42']

    def test_main_returns_value(self):
        r = execute_concurrent("100;")
        assert r['result'] == 100

    def test_empty_program(self):
        # Just a semicolon or no-op
        r = execute_concurrent("let x = 0;")
        assert r['tasks'][0]['state'] == 'COMPLETED'

    def test_task_with_while_and_channel(self):
        r = execute_concurrent("""
            fn accumulator(ch, count) {
                let sum = 0;
                let i = 0;
                while (i < count) {
                    sum = sum + recv(ch);
                    i = i + 1;
                }
                return sum;
            }
            let ch = chan(5);
            let t = spawn accumulator(ch, 5);
            send(ch, 10);
            send(ch, 20);
            send(ch, 30);
            send(ch, 40);
            send(ch, 50);
            let result = join(t);
            print(result);
        """)
        assert '150' in r['output']

    def test_interleaved_channel_ops(self):
        r = execute_concurrent("""
            let ch = chan(2);
            send(ch, 1);
            let a = recv(ch);
            send(ch, 2);
            send(ch, 3);
            let b = recv(ch);
            let c = recv(ch);
            print(a);
            print(b);
            print(c);
        """)
        assert r['output'] == ['1', '2', '3']


# ============================================================
# 11. Lexer and Parser Tests
# ============================================================

class TestLexerParser:

    def test_lex_spawn(self):
        tokens = conc_lex("spawn foo();")
        assert any(t.value == 'spawn' for t in tokens)

    def test_lex_yield(self):
        tokens = conc_lex("yield;")
        assert any(t.value == 'yield' for t in tokens)

    def test_lex_channel_ops(self):
        tokens = conc_lex("chan(1); send(ch, 5); recv(ch);")
        values = [t.value for t in tokens]
        assert 'chan' in values
        assert 'send' in values
        assert 'recv' in values

    def test_lex_select(self):
        tokens = conc_lex("select { case recv(ch) => val { } default => { } }")
        values = [t.value for t in tokens]
        assert 'select' in values
        assert 'case' in values
        assert 'default' in values

    def test_lex_arrow(self):
        tokens = conc_lex("=>")
        assert any(t.value == '=>' for t in tokens)

    def test_lex_task_id(self):
        tokens = conc_lex("task_id;")
        assert any(t.value == 'task_id' for t in tokens)

    def test_parse_spawn_expr(self):
        tokens = conc_lex("let t = spawn foo(1, 2);")
        parser = ConcParser(tokens)
        ast = parser.parse()
        assert len(ast.stmts) == 1

    def test_parse_yield_stmt(self):
        tokens = conc_lex("yield;")
        parser = ConcParser(tokens)
        ast = parser.parse()
        assert len(ast.stmts) == 1

    def test_parse_channel_ops(self):
        tokens = conc_lex("let ch = chan(1); send(ch, 5); let x = recv(ch);")
        parser = ConcParser(tokens)
        ast = parser.parse()
        assert len(ast.stmts) == 3

    def test_parse_select(self):
        src = """
        select {
            case recv(ch) => val {
                print(val);
            }
            default => {
                print("default");
            }
        }
        """
        tokens = conc_lex(src)
        parser = ConcParser(tokens)
        ast = parser.parse()
        assert len(ast.stmts) == 1


# ============================================================
# 12. Channel Python API
# ============================================================

class TestChannelAPI:

    def test_channel_creation(self):
        ch = Channel(buffer_size=3)
        assert ch.buffer_size == 3
        assert ch.can_send()
        assert not ch.can_recv()

    def test_channel_send_recv(self):
        ch = Channel(buffer_size=2)
        assert ch.try_send(10)
        assert ch.try_send(20)
        assert not ch.try_send(30)  # full
        ok, val = ch.try_recv()
        assert ok and val == 10
        ok, val = ch.try_recv()
        assert ok and val == 20
        ok, val = ch.try_recv()
        assert not ok

    def test_channel_close(self):
        ch = Channel()
        ch.close()
        assert not ch.try_send(1)

    def test_channel_id_unique(self):
        ch1 = Channel()
        ch2 = Channel()
        assert ch1.id != ch2.id


# ============================================================
# 13. Task State Machine
# ============================================================

class TestTaskStates:

    def test_all_tasks_complete(self):
        r = execute_concurrent("""
            fn noop() { }
            let t1 = spawn noop();
            let t2 = spawn noop();
            join(t1);
            join(t2);
        """)
        for tid, info in r['tasks'].items():
            assert info['state'] == 'COMPLETED'

    def test_failed_task_state(self):
        r = execute_concurrent("""
            fn bad() { let x = 1 / 0; }
            let t = spawn bad();
            yield;
            yield;
            print("ok");
        """)
        assert r['tasks'][1]['state'] == 'FAILED'
        assert r['tasks'][1]['error'] is not None

    def test_step_counting(self):
        r = execute_concurrent("""
            fn worker() {
                let x = 1 + 2;
            }
            let t = spawn worker();
            join(t);
        """)
        total = sum(info['steps'] for info in r['tasks'].values())
        assert total > 0
        assert r['total_steps'] == total


# ============================================================
# 14. Stress and Scale Tests
# ============================================================

class TestStress:

    def test_many_tasks(self):
        """Spawn many tasks."""
        r = execute_concurrent("""
            fn worker(result_ch, val) {
                send(result_ch, val);
            }
            let ch = chan(20);
            let i = 0;
            while (i < 20) {
                spawn worker(ch, i);
                i = i + 1;
            }
            let sum = 0;
            let j = 0;
            while (j < 20) {
                sum = sum + recv(ch);
                j = j + 1;
            }
            print(sum);
        """, max_total_steps=1000000)
        assert r['output'] == ['190']  # 0+1+...+19

    def test_deep_channel_pipeline(self):
        """Chain of tasks passing values through channels."""
        r = execute_concurrent("""
            fn relay(in_ch, out_ch) {
                let val = recv(in_ch);
                send(out_ch, val + 1);
            }
            let ch0 = chan(1);
            let ch1 = chan(1);
            let ch2 = chan(1);
            let ch3 = chan(1);
            let ch4 = chan(1);
            let ch5 = chan(1);

            let t1 = spawn relay(ch0, ch1);
            let t2 = spawn relay(ch1, ch2);
            let t3 = spawn relay(ch2, ch3);
            let t4 = spawn relay(ch3, ch4);
            let t5 = spawn relay(ch4, ch5);

            send(ch0, 0);
            let result = recv(ch5);
            print(result);

            join(t1);
            join(t2);
            join(t3);
            join(t4);
            join(t5);
        """)
        assert r['output'] == ['5']

    def test_high_contention_channel(self):
        """Many tasks compete for the same channel."""
        r = execute_concurrent("""
            fn sender(ch) {
                send(ch, 1);
            }
            let ch = chan(1);
            let t1 = spawn sender(ch);
            let t2 = spawn sender(ch);
            let t3 = spawn sender(ch);
            let t4 = spawn sender(ch);
            let t5 = spawn sender(ch);
            let sum = 0;
            let i = 0;
            while (i < 5) {
                sum = sum + recv(ch);
                i = i + 1;
            }
            print(sum);
            join(t1);
            join(t2);
            join(t3);
            join(t4);
            join(t5);
        """)
        assert r['output'] == ['5']

    def test_task_with_loops(self):
        r = execute_concurrent("""
            fn summer(n) {
                let sum = 0;
                let i = 1;
                while (i <= n) {
                    sum = sum + i;
                    i = i + 1;
                }
                return sum;
            }
            let t = spawn summer(100);
            let result = join(t);
            print(result);
        """, max_total_steps=1000000)
        assert r['output'] == ['5050']


# ============================================================
# 15. Concurrency Correctness
# ============================================================

class TestConcurrencyCorrectness:

    def test_channel_preserves_order(self):
        """Values come out of channel in FIFO order."""
        r = execute_concurrent("""
            fn producer(ch) {
                let i = 0;
                while (i < 10) {
                    send(ch, i);
                    i = i + 1;
                }
            }
            let ch = chan(10);
            let t = spawn producer(ch);
            join(t);
            let i = 0;
            while (i < 10) {
                print(recv(ch));
                i = i + 1;
            }
        """)
        assert r['output'] == [str(i) for i in range(10)]

    def test_no_lost_messages(self):
        """All sent messages are received."""
        r = execute_concurrent("""
            fn sender(ch, start, count) {
                let i = 0;
                while (i < count) {
                    send(ch, start + i);
                    i = i + 1;
                }
            }
            let ch = chan(5);
            let t1 = spawn sender(ch, 0, 5);
            let t2 = spawn sender(ch, 100, 5);
            let sum = 0;
            let j = 0;
            while (j < 10) {
                sum = sum + recv(ch);
                j = j + 1;
            }
            join(t1);
            join(t2);
            print(sum);
        """)
        # 0+1+2+3+4 + 100+101+102+103+104 = 10 + 510 = 520
        assert r['output'] == ['520']

    def test_deadlock_detection_not_silent(self):
        """If all tasks are blocked, execution should terminate (not hang)."""
        # Two tasks, each waiting on the other's channel -- deadlock
        # The scheduler should exhaust the run queue and return
        # (main task completes, deadlocked tasks remain blocked)
        r = execute_concurrent("""
            fn deadlocker(ch) {
                recv(ch);  // blocks forever -- nothing sends
            }
            let ch = chan(1);
            let t = spawn deadlocker(ch);
            // Main doesn't join, so it completes
            print("main done");
        """)
        assert 'main done' in r['output']
        assert r['tasks'][1]['state'] == 'BLOCKED_RECV'

    def test_task_completion_order_independent(self):
        """Tasks can complete in any order."""
        r = execute_concurrent("""
            fn fast() {
                return 1;
            }
            fn slow() {
                yield;
                yield;
                yield;
                return 2;
            }
            let tf = spawn fast();
            let ts = spawn slow();
            let r2 = join(ts);
            let r1 = join(tf);
            print(r1 + r2);
        """)
        assert '3' in r['output']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
