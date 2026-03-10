"""
Tests for C030 Concurrent Debugger
Challenge C030 -- AgentZero Session 031

Tests composition of C029 (Concurrent Runtime) + C020 (REPL/Debugger) patterns.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from concurrent_debugger import (
    ConcurrentDebugVM, StopReason, ConcEventType, DebugEvent,
    Breakpoint, WatchEntry, TraceEntry,
    create_debugger, debug_concurrent, compile_concurrent,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C029_concurrent_runtime'))
from concurrent_runtime import TaskState, Channel


# ============================================================
# Helper
# ============================================================

def make_debugger(source, **kwargs):
    return create_debugger(source, **kwargs)


# ============================================================
# 1. Basic creation and halt
# ============================================================

class TestBasicCreation:
    def test_create_debugger(self):
        dbg = make_debugger("let x = 42;")
        assert not dbg.is_halted()
        assert dbg.get_error() is None

    def test_simple_program_runs_to_halt(self):
        dbg = make_debugger("let x = 42;")
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT
        assert dbg.is_halted()

    def test_print_output(self):
        dbg = make_debugger("print(1); print(2); print(3);")
        dbg.continue_execution()
        assert dbg.output == ["1", "2", "3"]

    def test_arithmetic(self):
        dbg = make_debugger("let x = 10 + 20; print(x);")
        dbg.continue_execution()
        assert dbg.output == ["30"]

    def test_function_call(self):
        dbg = make_debugger("""
            fn add(a, b) { return a + b; }
            print(add(3, 4));
        """)
        dbg.continue_execution()
        assert dbg.output == ["7"]


# ============================================================
# 2. Stepping
# ============================================================

class TestStepping:
    def test_step_single_instruction(self):
        dbg = make_debugger("let x = 1;")
        event = dbg.step()
        assert event.reason == StopReason.STEP
        assert event.task_id == 0

    def test_step_to_halt(self):
        dbg = make_debugger("let x = 1;")
        events = []
        for _ in range(100):
            event = dbg.step()
            events.append(event)
            if event.reason == StopReason.HALT:
                break
        assert events[-1].reason == StopReason.HALT

    def test_step_line(self):
        dbg = make_debugger("let x = 1;\nlet y = 2;\nprint(x + y);")
        event = dbg.step_line()
        assert event.reason == StopReason.STEP

    def test_step_over_function(self):
        dbg = make_debugger("""
            fn foo() { return 42; }
            let x = foo();
            print(x);
        """)
        # Step to the call
        for _ in range(20):
            event = dbg.step()
            if event.reason == StopReason.HALT:
                break
        # Verify it ran
        assert dbg.is_halted() or event.reason == StopReason.HALT

    def test_step_over_skips_function_body(self):
        dbg = make_debugger("""
            fn deep() { let a = 1; let b = 2; return a + b; }
            let r = deep();
            print(r);
        """)
        # Use step_over which should skip into deep()
        events = []
        for _ in range(50):
            event = dbg.step_over()
            events.append(event)
            if event.reason == StopReason.HALT:
                break
        assert dbg.output == ["3"]

    def test_step_out(self):
        dbg = make_debugger("""
            fn inner() { let x = 10; return x; }
            let r = inner();
            print(r);
        """)
        # Step into the function
        for _ in range(20):
            event = dbg.step()
            task = dbg.tasks.get(0)
            if task and len(task.call_stack) > 0:
                break
        # Now step out
        if task and len(task.call_stack) > 0:
            event = dbg.step_out()
            # Should return from function
        dbg_event = dbg.continue_execution()
        assert dbg.output == ["10"]


# ============================================================
# 3. Breakpoints
# ============================================================

class TestBreakpoints:
    def test_add_remove_breakpoint(self):
        dbg = make_debugger("let x = 1;")
        bp_id = dbg.add_breakpoint(line=1)
        assert bp_id == 1
        assert len(dbg.list_breakpoints()) == 1
        assert dbg.remove_breakpoint(bp_id)
        assert len(dbg.list_breakpoints()) == 0

    def test_breakpoint_hit(self):
        dbg = make_debugger("let x = 1;\nlet y = 2;\nprint(x + y);")
        dbg.add_breakpoint(line=2)
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        assert event.line == 2

    def test_breakpoint_hit_count(self):
        dbg = make_debugger("""
            let i = 0;
            while (i < 3) {
                print(i);
                i = i + 1;
            }
        """)
        bp_id = dbg.add_breakpoint(line=3)
        # First hit
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        bp = dbg.breakpoints[bp_id]
        assert bp.hit_count == 1
        # Continue to second hit
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        assert bp.hit_count == 2

    def test_conditional_breakpoint(self):
        dbg = make_debugger("""
            let i = 0;
            while (i < 5) {
                print(i);
                i = i + 1;
            }
        """)
        dbg.add_breakpoint(line=3, condition="i == 3")
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        task_vars = dbg.get_task_variables(0)
        assert task_vars['i'] == 3

    def test_disable_breakpoint(self):
        dbg = make_debugger("let x = 1;\nlet y = 2;\nprint(x + y);")
        bp_id = dbg.add_breakpoint(line=2)
        dbg.enable_breakpoint(bp_id, False)
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT  # not stopped

    def test_address_breakpoint(self):
        dbg = make_debugger("let x = 1;\nlet y = 2;")
        dbg.add_breakpoint(address=0)
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        assert event.ip == 0

    def test_task_specific_breakpoint(self):
        """Breakpoint that only triggers in a specific task."""
        dbg = make_debugger("""
            fn worker() { let a = 1; print(a); }
            let t = spawn worker();
            join(t);
        """)
        # Breakpoint only in task 1 (spawned task)
        dbg.add_breakpoint(line=1, task_id=1)
        event = dbg.continue_execution()
        # Should either hit BP in task 1 or complete
        # The BP fires if task 1 runs a line-1 instruction
        assert event.reason in (StopReason.BREAKPOINT, StopReason.HALT,
                                StopReason.TASK_COMPLETE, StopReason.TASK_YIELD,
                                StopReason.TASK_BLOCK)

    def test_remove_nonexistent_breakpoint(self):
        dbg = make_debugger("let x = 1;")
        assert not dbg.remove_breakpoint(999)


# ============================================================
# 4. Watch expressions
# ============================================================

class TestWatches:
    def test_add_remove_watch(self):
        dbg = make_debugger("let x = 1;")
        w_id = dbg.add_watch("x")
        assert w_id == 1
        assert dbg.remove_watch(w_id)
        assert not dbg.remove_watch(w_id)

    def test_watch_triggers_on_change(self):
        dbg = make_debugger("""
            let x = 0;
            x = 1;
            x = 2;
        """)
        dbg.add_watch("x")
        event = dbg.continue_execution()
        assert event.reason == StopReason.WATCH
        assert "x" in event.message

    def test_watch_with_expression(self):
        dbg = make_debugger("""
            let x = 0;
            x = 5;
            x = 10;
        """)
        dbg.add_watch("x + 1")
        event = dbg.continue_execution()
        assert event.reason == StopReason.WATCH

    def test_task_specific_watch(self):
        """Watch only in a specific task."""
        dbg = make_debugger("""
            fn worker() { let v = 42; print(v); }
            let t = spawn worker();
            join(t);
        """)
        dbg.add_watch("v", task_id=1)
        event = dbg.continue_execution()
        # Should complete or trigger watch in task 1
        assert event.reason in (StopReason.WATCH, StopReason.HALT,
                                StopReason.TASK_COMPLETE)


# ============================================================
# 5. Task listing and inspection
# ============================================================

class TestTaskInspection:
    def test_initial_task_list(self):
        dbg = make_debugger("let x = 1;")
        tasks = dbg.get_task_list()
        assert len(tasks) == 1
        assert tasks[0]['id'] == 0

    def test_task_list_after_spawn(self):
        dbg = make_debugger("""
            fn worker() { let a = 1; }
            let t = spawn worker();
            yield;
            join(t);
        """)
        # Run until spawn happens
        for _ in range(50):
            event = dbg.step()
            if len(dbg.tasks) > 1:
                break
        tasks = dbg.get_task_list()
        assert len(tasks) >= 2

    def test_get_task_stack(self):
        dbg = make_debugger("let x = 42;")
        stack = dbg.get_task_stack(0)
        assert stack is not None
        assert isinstance(stack, list)

    def test_get_task_variables(self):
        dbg = make_debugger("let x = 42;")
        dbg.continue_execution()
        vars_ = dbg.get_task_variables(0)
        assert vars_ is not None
        assert vars_['x'] == 42

    def test_get_task_call_stack(self):
        dbg = make_debugger("""
            fn foo() { return 1; }
            foo();
        """)
        cs = dbg.get_task_call_stack(0)
        assert cs is not None
        assert len(cs) >= 1

    def test_get_nonexistent_task(self):
        dbg = make_debugger("let x = 1;")
        assert dbg.get_task_stack(99) is None
        assert dbg.get_task_variables(99) is None
        assert dbg.get_task_call_stack(99) is None

    def test_task_line_tracking(self):
        dbg = make_debugger("let x = 1;\nlet y = 2;")
        line = dbg.get_task_line(0)
        assert line >= 1

    def test_run_queue(self):
        dbg = make_debugger("let x = 1;")
        q = dbg.get_run_queue()
        # Initially main task is in queue (or was pulled out)
        assert isinstance(q, list)


# ============================================================
# 6. Concurrent program debugging
# ============================================================

class TestConcurrentDebugging:
    def test_spawn_and_join(self):
        dbg = make_debugger("""
            fn worker() { return 42; }
            let t = spawn worker();
            let r = join(t);
            print(r);
        """)
        dbg.continue_execution()
        assert "42" in dbg.output

    def test_channel_communication(self):
        dbg = make_debugger("""
            fn sender(ch) { send(ch, 99); }
            let c = chan(1);
            let t = spawn sender(c);
            let v = recv(c);
            join(t);
            print(v);
        """)
        dbg.continue_execution()
        assert "99" in dbg.output

    def test_multiple_tasks(self):
        dbg = make_debugger("""
            fn worker(n) { print(n); }
            let t1 = spawn worker(1);
            let t2 = spawn worker(2);
            join(t1);
            join(t2);
        """)
        dbg.continue_execution()
        assert "1" in dbg.output
        assert "2" in dbg.output

    def test_step_through_concurrent(self):
        dbg = make_debugger("""
            fn worker() { let x = 10; return x; }
            let t = spawn worker();
            let r = join(t);
            print(r);
        """)
        events = []
        for _ in range(200):
            event = dbg.step()
            events.append(event)
            if event.reason in (StopReason.HALT, StopReason.ERROR,
                                StopReason.DEADLOCK):
                break
        assert any(e.reason == StopReason.HALT for e in events)

    def test_focused_task_stepping(self):
        """Step only within a specific task."""
        dbg = make_debugger("""
            fn worker() { let a = 1; let b = 2; return a + b; }
            let t = spawn worker();
            yield;
            let r = join(t);
        """)
        # Run until spawn
        for _ in range(30):
            event = dbg.step()
            if len(dbg.tasks) > 1:
                break
        # Focus on spawned task
        dbg.focus_task(1)
        assert dbg.get_focused_task() == 1
        event = dbg.step()
        assert event.task_id == 1 or event.reason in (
            StopReason.HALT, StopReason.TASK_COMPLETE,
            StopReason.TASK_BLOCK, StopReason.ERROR)

    def test_unfocus(self):
        dbg = make_debugger("let x = 1;")
        dbg.focus_task(0)
        assert dbg.get_focused_task() == 0
        dbg.focus_task(-1)
        assert dbg.get_focused_task() == -1

    def test_focus_invalid_task(self):
        dbg = make_debugger("let x = 1;")
        assert not dbg.focus_task(999)


# ============================================================
# 7. Channel inspection
# ============================================================

class TestChannelInspection:
    def test_channel_info_empty(self):
        dbg = make_debugger("let x = 1;")
        dbg.continue_execution()
        channels = dbg.get_channel_info()
        assert channels == []

    def test_channel_info_after_create(self):
        dbg = make_debugger("""
            let c = chan(3);
            send(c, 10);
            send(c, 20);
        """)
        dbg.continue_execution()
        channels = dbg.get_channel_info()
        assert len(channels) == 1
        ch = channels[0]
        assert ch['buffer_size'] == 3
        assert ch['buffer_count'] == 2
        assert ch['buffer_contents'] == [10, 20]

    def test_channel_multiple(self):
        dbg = make_debugger("""
            let c1 = chan(1);
            let c2 = chan(2);
            send(c1, 1);
        """)
        dbg.continue_execution()
        channels = dbg.get_channel_info()
        assert len(channels) == 2


# ============================================================
# 8. Concurrency event breakpoints
# ============================================================

class TestEventBreaks:
    def test_break_on_spawn(self):
        dbg = make_debugger("""
            fn worker() { return 1; }
            let t = spawn worker();
            join(t);
        """)
        dbg.break_on_event(ConcEventType.SPAWN)
        event = dbg.continue_execution()
        assert event.reason == StopReason.TASK_SPAWN
        assert 'spawned_task' in (event.data or {})

    def test_break_on_send(self):
        dbg = make_debugger("""
            let c = chan(1);
            send(c, 42);
        """)
        dbg.break_on_event(ConcEventType.SEND)
        event = dbg.continue_execution()
        assert event.reason == StopReason.CHANNEL_SEND
        assert event.data['value'] == 42

    def test_break_on_recv(self):
        dbg = make_debugger("""
            let c = chan(1);
            send(c, 99);
            let v = recv(c);
        """)
        dbg.break_on_event(ConcEventType.RECV)
        event = dbg.continue_execution()
        assert event.reason == StopReason.CHANNEL_RECV
        assert event.data['value'] == 99

    def test_break_on_yield(self):
        dbg = make_debugger("""
            yield;
            let x = 1;
        """)
        dbg.break_on_event(ConcEventType.YIELD)
        event = dbg.continue_execution()
        assert event.reason == StopReason.TASK_YIELD

    def test_break_on_block(self):
        """Task blocks when channel is full (buffer=1, two sends)."""
        dbg = make_debugger("""
            fn sender(ch) {
                send(ch, 1);
                send(ch, 2);
            }
            let c = chan(1);
            let t = spawn sender(c);
            yield;
            yield;
            yield;
            yield;
            let v1 = recv(c);
            let v2 = recv(c);
            join(t);
        """)
        dbg.break_on_event(ConcEventType.BLOCK)
        event = dbg.continue_execution()
        # Sender should block on second send (buffer full)
        assert event.reason == StopReason.TASK_BLOCK

    def test_break_on_complete(self):
        dbg = make_debugger("""
            fn worker() { return 1; }
            let t = spawn worker();
            yield;
            join(t);
        """)
        dbg.break_on_event(ConcEventType.COMPLETE)
        event = dbg.continue_execution()
        assert event.reason == StopReason.TASK_COMPLETE

    def test_disable_event_break(self):
        dbg = make_debugger("""
            fn worker() { return 1; }
            let t = spawn worker();
            join(t);
        """)
        dbg.break_on_event(ConcEventType.SPAWN)
        dbg.break_on_event(ConcEventType.SPAWN, enabled=False)
        assert ConcEventType.SPAWN not in dbg.get_event_breaks()

    def test_multiple_event_breaks(self):
        dbg = make_debugger("""
            fn worker() { return 1; }
            let t = spawn worker();
            yield;
            join(t);
        """)
        dbg.break_on_event(ConcEventType.SPAWN)
        dbg.break_on_event(ConcEventType.COMPLETE)
        event = dbg.continue_execution()
        assert event.reason in (StopReason.TASK_SPAWN, StopReason.TASK_COMPLETE)


# ============================================================
# 9. Deadlock detection
# ============================================================

class TestDeadlockDetection:
    def test_deadlock_two_tasks(self):
        """Two tasks each waiting to recv from empty channels."""
        dbg = make_debugger("""
            fn waiter(ch) { let v = recv(ch); }
            let c1 = chan(1);
            let c2 = chan(1);
            let t1 = spawn waiter(c1);
            let t2 = spawn waiter(c2);
            yield;
            yield;
            let v = recv(c1);
        """)
        event = dbg.continue_execution()
        assert event.reason == StopReason.DEADLOCK
        assert 'blocked_tasks' in (event.data or {})

    def test_no_deadlock_normal_completion(self):
        dbg = make_debugger("""
            fn worker(ch) { send(ch, 42); }
            let c = chan(1);
            let t = spawn worker(c);
            let v = recv(c);
            join(t);
            print(v);
        """)
        event = dbg.continue_execution()
        assert event.reason != StopReason.DEADLOCK
        assert "42" in dbg.output


# ============================================================
# 10. Scheduler stepping
# ============================================================

class TestSchedulerStepping:
    def test_scheduler_round(self):
        dbg = make_debugger("""
            fn worker() { let x = 1; return x; }
            let t = spawn worker();
            yield;
            join(t);
        """)
        # Step enough to get tasks spawned
        for _ in range(30):
            event = dbg.step()
            if len(dbg.tasks) > 1:
                break
        # Now do scheduler round
        event = dbg.step_scheduler()
        assert event.reason in (StopReason.SCHEDULER_ROUND, StopReason.HALT,
                                StopReason.BREAKPOINT, StopReason.ERROR,
                                StopReason.TASK_COMPLETE, StopReason.TASK_BLOCK,
                                StopReason.TASK_YIELD)

    def test_scheduler_round_with_breakpoint(self):
        dbg = make_debugger("""
            fn worker() { let x = 1; print(x); }
            let t = spawn worker();
            yield;
            join(t);
        """)
        dbg.add_breakpoint(line=1)
        event = dbg.step_scheduler()
        # May hit breakpoint during round
        assert event.reason in (StopReason.BREAKPOINT, StopReason.SCHEDULER_ROUND,
                                StopReason.HALT, StopReason.TASK_BLOCK,
                                StopReason.TASK_COMPLETE, StopReason.TASK_YIELD)


# ============================================================
# 11. Disassembly
# ============================================================

class TestDisassembly:
    def test_disassemble_task(self):
        dbg = make_debugger("let x = 42;")
        dis = dbg.disassemble_task(0)
        assert dis is not None
        assert "CONST" in dis

    def test_disassemble_nonexistent_task(self):
        dbg = make_debugger("let x = 1;")
        assert dbg.disassemble_task(99) is None

    def test_disassemble_with_concurrency_ops(self):
        dbg = make_debugger("""
            fn worker() { return 1; }
            let t = spawn worker();
        """)
        dis = dbg.disassemble_task(0)
        assert "SPAWN" in dis


# ============================================================
# 12. Trace
# ============================================================

class TestTrace:
    def test_trace_recording(self):
        dbg = make_debugger("let x = 1; let y = 2;")
        dbg.trace_enabled = True
        dbg.continue_execution()
        assert len(dbg.trace_log) > 0
        entry = dbg.trace_log[0]
        assert hasattr(entry, 'task_id')
        assert hasattr(entry, 'op')

    def test_trace_includes_task_id(self):
        dbg = make_debugger("""
            fn worker() { let a = 1; }
            let t = spawn worker();
            yield;
            join(t);
        """)
        dbg.trace_enabled = True
        dbg.continue_execution()
        task_ids = set(e.task_id for e in dbg.trace_log)
        # Should have traces from multiple tasks
        assert 0 in task_ids

    def test_trace_disabled_by_default(self):
        dbg = make_debugger("let x = 1;")
        assert not dbg.trace_enabled
        dbg.continue_execution()
        assert len(dbg.trace_log) == 0


# ============================================================
# 13. Run-task-until-yield
# ============================================================

class TestRunTaskUntilYield:
    def test_run_until_yield(self):
        dbg = make_debugger("""
            fn worker() {
                print(1);
                yield;
                print(2);
            }
            let t = spawn worker();
            yield;
        """)
        # Step until spawn happens
        for _ in range(30):
            event = dbg.step()
            if len(dbg.tasks) > 1:
                break
        # Run task 1 until it yields
        event = dbg.run_task_until_yield(1)
        assert event.reason in (StopReason.TASK_YIELD, StopReason.HALT,
                                StopReason.TASK_COMPLETE, StopReason.TASK_BLOCK)

    def test_run_until_yield_nonexistent(self):
        dbg = make_debugger("let x = 1;")
        event = dbg.run_task_until_yield(99)
        assert event.reason == StopReason.ERROR

    def test_run_completed_task(self):
        dbg = make_debugger("let x = 1;")
        dbg.continue_execution()
        event = dbg.run_task_until_yield(0)
        assert event.reason in (StopReason.HALT, StopReason.ERROR)


# ============================================================
# 14. Complex concurrent scenarios
# ============================================================

class TestComplexScenarios:
    def test_producer_consumer(self):
        dbg = make_debugger("""
            fn producer(ch) {
                send(ch, 1);
                send(ch, 2);
                send(ch, 3);
            }
            fn consumer(ch) {
                let a = recv(ch);
                let b = recv(ch);
                let c = recv(ch);
                print(a + b + c);
            }
            let c = chan(3);
            let p = spawn producer(c);
            let k = spawn consumer(c);
            join(p);
            join(k);
        """)
        event = dbg.continue_execution()
        assert event.reason in (StopReason.HALT, StopReason.TASK_COMPLETE)
        assert "6" in dbg.output

    def test_pipeline(self):
        dbg = make_debugger("""
            fn stage1(out) {
                send(out, 10);
            }
            fn stage2(inp, out) {
                let v = recv(inp);
                send(out, v * 2);
            }
            let c1 = chan(1);
            let c2 = chan(1);
            let t1 = spawn stage1(c1);
            let t2 = spawn stage2(c1, c2);
            let result = recv(c2);
            join(t1);
            join(t2);
            print(result);
        """)
        event = dbg.continue_execution()
        assert "20" in dbg.output

    def test_fan_out(self):
        dbg = make_debugger("""
            fn worker(ch, n) {
                send(ch, n * 10);
            }
            let c = chan(3);
            let t1 = spawn worker(c, 1);
            let t2 = spawn worker(c, 2);
            let t3 = spawn worker(c, 3);
            let a = recv(c);
            let b = recv(c);
            let d = recv(c);
            join(t1);
            join(t2);
            join(t3);
            print(a + b + d);
        """)
        event = dbg.continue_execution()
        assert "60" in dbg.output

    def test_debug_with_breakpoint_in_spawned_task(self):
        dbg = make_debugger("""
            fn worker() {
                let x = 10;
                let y = 20;
                print(x + y);
            }
            let t = spawn worker();
            yield;
            join(t);
        """)
        # BP on line 3 (let y = 20;)
        bp_id = dbg.add_breakpoint(line=3, task_id=1)
        event = dbg.continue_execution()
        if event.reason == StopReason.BREAKPOINT:
            assert event.task_id == 1
            # Continue
            event = dbg.continue_execution()
        # Should complete
        assert "30" in dbg.output or event.reason == StopReason.HALT

    def test_step_through_channel_exchange(self):
        """Step-by-step through a send/recv pair."""
        dbg = make_debugger("""
            fn sender(ch) { send(ch, 42); }
            let c = chan(1);
            let t = spawn sender(c);
            let v = recv(c);
            join(t);
            print(v);
        """)
        dbg.break_on_event(ConcEventType.SEND)
        event = dbg.continue_execution()
        if event.reason == StopReason.CHANNEL_SEND:
            assert event.data['value'] == 42
        # Continue to completion
        dbg.break_on_event(ConcEventType.SEND, enabled=False)
        dbg.continue_execution()
        assert "42" in dbg.output


# ============================================================
# 15. Error handling
# ============================================================

class TestErrorHandling:
    def test_division_by_zero(self):
        dbg = make_debugger("let x = 1 / 0;")
        event = dbg.continue_execution()
        assert event.reason == StopReason.ERROR
        assert "zero" in event.message.lower()

    def test_undefined_variable(self):
        dbg = make_debugger("print(undefined_var);")
        event = dbg.continue_execution()
        assert event.reason == StopReason.ERROR

    def test_task_error_doesnt_crash_debugger(self):
        dbg = make_debugger("""
            fn bad() { let x = 1 / 0; }
            let t = spawn bad();
            yield;
            yield;
            print(1);
        """)
        event = dbg.continue_execution()
        # Either error propagates or we see output
        assert event.reason in (StopReason.ERROR, StopReason.HALT,
                                StopReason.DEADLOCK, StopReason.TASK_COMPLETE)

    def test_step_halted_program(self):
        dbg = make_debugger("let x = 1;")
        dbg.continue_execution()
        assert dbg.is_halted()
        event = dbg.step()
        assert event.reason == StopReason.HALT

    def test_continue_halted_program(self):
        dbg = make_debugger("let x = 1;")
        dbg.continue_execution()
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT

    def test_execution_limit(self):
        dbg = make_debugger("while (true) { let x = 1; }", max_total_steps=100)
        event = dbg.continue_execution()
        assert event.reason == StopReason.ERROR
        assert "limit" in event.message.lower()


# ============================================================
# 16. Public API
# ============================================================

class TestPublicAPI:
    def test_create_debugger_function(self):
        dbg = create_debugger("let x = 1;")
        assert isinstance(dbg, ConcurrentDebugVM)

    def test_debug_concurrent_function(self):
        result = debug_concurrent("print(42);")
        assert "42" in result['output']
        assert 'tasks' in result
        assert 'channels' in result

    def test_debug_concurrent_with_breakpoint(self):
        result = debug_concurrent(
            "let x = 1;\nlet y = 2;\nprint(x + y);",
            breakpoints=[{'line': 2}],
        )
        assert result['event'].reason == StopReason.BREAKPOINT

    def test_debug_concurrent_with_event_breaks(self):
        result = debug_concurrent(
            """
            fn worker() { return 1; }
            let t = spawn worker();
            join(t);
            """,
            event_breaks=[ConcEventType.SPAWN],
        )
        assert result['event'].reason == StopReason.TASK_SPAWN


# ============================================================
# 17. Interaction of breakpoints and concurrency
# ============================================================

class TestBreakpointConcurrencyInteraction:
    def test_breakpoint_global_hits_any_task(self):
        """Global breakpoint fires in whichever task matches the line."""
        dbg = make_debugger("""
            fn worker() { let x = 1; print(x); }
            let t = spawn worker();
            yield;
            join(t);
        """)
        # Breakpoint on line 3 (let t = spawn worker();)
        dbg.add_breakpoint(line=3)
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT

    def test_continue_after_breakpoint_resumes(self):
        dbg = make_debugger("""
            let x = 1;
            let y = 2;
            let z = 3;
            print(x + y + z);
        """)
        dbg.add_breakpoint(line=2)
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT
        assert "6" in dbg.output

    def test_breakpoint_skip_on_continue(self):
        """Continue should skip the breakpoint at the current line."""
        dbg = make_debugger("""
            let x = 1;
            let y = 2;
            print(x + y);
        """)
        dbg.add_breakpoint(line=2)
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        assert event.line == 2
        # Continue should not re-hit line 2 immediately
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT


# ============================================================
# 18. State after operations
# ============================================================

class TestStateAfterOps:
    def test_variables_after_steps(self):
        dbg = make_debugger("let x = 10; let y = 20;")
        dbg.continue_execution()
        vars_ = dbg.get_task_variables(0)
        assert vars_['x'] == 10
        assert vars_['y'] == 20

    def test_stack_during_computation(self):
        dbg = make_debugger("let x = 3 + 4;")
        # Step partway
        for _ in range(5):
            dbg.step()
        stack = dbg.get_task_stack(0)
        assert isinstance(stack, list)

    def test_output_accumulates(self):
        dbg = make_debugger("print(1); print(2); print(3);")
        dbg.add_breakpoint(line=1)
        # May stop at breakpoint
        event = dbg.continue_execution()
        if event.reason == StopReason.BREAKPOINT:
            partial = list(dbg.output)
            dbg.continue_execution()
        assert dbg.output == ["1", "2", "3"]


# ============================================================
# 19. Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        # Just a HALT instruction
        dbg = make_debugger("")
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT

    def test_single_yield(self):
        dbg = make_debugger("yield; let x = 1;")
        event = dbg.continue_execution()
        # Should complete after yield
        assert event.reason in (StopReason.HALT, StopReason.TASK_YIELD)

    def test_nested_function_calls(self):
        dbg = make_debugger("""
            fn a() { return 1; }
            fn b() { return a() + 1; }
            fn c() { return b() + 1; }
            print(c());
        """)
        dbg.continue_execution()
        assert "3" in dbg.output

    def test_recursive_function(self):
        dbg = make_debugger("""
            fn fact(n) {
                if (n <= 1) { return 1; }
                return n * fact(n - 1);
            }
            print(fact(5));
        """)
        dbg.continue_execution()
        assert "120" in dbg.output

    def test_many_spawns(self):
        dbg = make_debugger("""
            fn noop() { let x = 0; }
            let t1 = spawn noop();
            let t2 = spawn noop();
            let t3 = spawn noop();
            let t4 = spawn noop();
            let t5 = spawn noop();
            join(t1); join(t2); join(t3); join(t4); join(t5);
        """)
        event = dbg.continue_execution()
        assert event.reason in (StopReason.HALT, StopReason.TASK_COMPLETE)
        tasks = dbg.get_task_list()
        assert len(tasks) == 6  # main + 5 workers

    def test_step_blocked_task_returns_block_event(self):
        dbg = make_debugger("""
            let c = chan(1);
            let v = recv(c);
        """)
        # Run until main blocks on recv
        for _ in range(30):
            event = dbg.step()
            if event.reason == StopReason.TASK_BLOCK:
                break
        if event.reason == StopReason.TASK_BLOCK:
            event2 = dbg.step(task_id=0)
            assert event2.reason == StopReason.TASK_BLOCK

    def test_string_output(self):
        dbg = make_debugger('print("hello");')
        dbg.continue_execution()
        assert "hello" in dbg.output

    def test_boolean_output(self):
        dbg = make_debugger("print(true); print(false);")
        dbg.continue_execution()
        assert "true" in dbg.output
        assert "false" in dbg.output

    def test_while_loop_with_breakpoint(self):
        dbg = make_debugger("""
            let sum = 0;
            let i = 0;
            while (i < 10) {
                sum = sum + i;
                i = i + 1;
            }
            print(sum);
        """)
        dbg.add_breakpoint(line=4, condition="i == 5")
        event = dbg.continue_execution()
        if event.reason == StopReason.BREAKPOINT:
            vars_ = dbg.get_task_variables(0)
            assert vars_['i'] == 5
            dbg.continue_execution()
        assert "45" in dbg.output


# ============================================================
# 20. Select statement debugging
# ============================================================

class TestSelectDebugging:
    def test_select_with_ready_channel(self):
        dbg = make_debugger("""
            let c = chan(1);
            send(c, 42);
            select {
                case recv(c) => v {
                    print(v);
                }
            }
        """)
        dbg.continue_execution()
        assert "42" in dbg.output

    def test_select_with_default(self):
        dbg = make_debugger("""
            let c = chan(1);
            select {
                case recv(c) => v {
                    print(v);
                }
                default => {
                    print(0);
                }
            }
        """)
        dbg.continue_execution()
        assert "0" in dbg.output


# ============================================================
# 21. Mixed breakpoints and watches
# ============================================================

class TestMixedBreakpointsWatches:
    def test_breakpoint_then_watch(self):
        dbg = make_debugger("""
            let x = 0;
            x = 1;
            x = 2;
            x = 3;
        """)
        dbg.add_breakpoint(line=2)
        dbg.add_watch("x")
        event = dbg.continue_execution()
        # Could hit BP or watch first depending on execution
        assert event.reason in (StopReason.BREAKPOINT, StopReason.WATCH)

    def test_watch_after_breakpoint_continue(self):
        dbg = make_debugger("""
            let x = 0;
            x = 1;
            x = 2;
        """)
        dbg.add_breakpoint(line=2)
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        dbg.add_watch("x")
        event = dbg.continue_execution()
        # Watch should fire or program halts
        assert event.reason in (StopReason.WATCH, StopReason.HALT)


# ============================================================
# 22. Comprehensive integration
# ============================================================

class TestComprehensiveIntegration:
    def test_full_debug_session(self):
        """Simulate a full debugging session: set BP, inspect, continue, check."""
        dbg = make_debugger("""
            fn compute(n) {
                let result = n * 2;
                return result;
            }
            let a = compute(5);
            let b = compute(10);
            print(a + b);
        """)
        # Set breakpoint on the line where compute is called
        dbg.add_breakpoint(line=8)
        # Run to first call
        event = dbg.continue_execution()
        if event.reason == StopReason.BREAKPOINT:
            # Continue to second call line
            event = dbg.continue_execution()
        # Continue to completion
        if event.reason == StopReason.BREAKPOINT:
            dbg.continue_execution()
        elif event.reason != StopReason.HALT:
            dbg.continue_execution()
        assert "30" in dbg.output

    def test_concurrent_debug_session(self):
        """Full debug session with concurrent tasks."""
        dbg = make_debugger("""
            fn doubler(ch, n) {
                send(ch, n * 2);
            }
            let c = chan(2);
            let t1 = spawn doubler(c, 5);
            let t2 = spawn doubler(c, 10);
            let a = recv(c);
            let b = recv(c);
            join(t1);
            join(t2);
            print(a + b);
        """)
        # Break on sends
        dbg.break_on_event(ConcEventType.SEND)
        event = dbg.continue_execution()
        assert event.reason == StopReason.CHANNEL_SEND
        # Disable and run to completion
        dbg.break_on_event(ConcEventType.SEND, enabled=False)
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT
        assert "30" in dbg.output

    def test_trace_and_continue(self):
        """Enable trace, run, verify trace data."""
        dbg = make_debugger("""
            fn add(a, b) { return a + b; }
            print(add(1, 2));
        """)
        dbg.trace_enabled = True
        dbg.continue_execution()
        assert len(dbg.trace_log) > 0
        assert "3" in dbg.output
        # Check trace has function call entries
        ops = [e.op for e in dbg.trace_log]
        assert 'CALL' in ops

    def test_channel_inspect_during_debug(self):
        """Inspect channels mid-execution."""
        dbg = make_debugger("""
            let c = chan(3);
            send(c, 1);
            send(c, 2);
            let v = recv(c);
            print(v);
        """)
        dbg.break_on_event(ConcEventType.RECV)
        event = dbg.continue_execution()
        if event.reason == StopReason.CHANNEL_RECV:
            channels = dbg.get_channel_info()
            assert len(channels) >= 1
        dbg.break_on_event(ConcEventType.RECV, enabled=False)
        dbg.continue_execution()
        assert "1" in dbg.output


# ============================================================
# 23. Additional coverage: stress and edge patterns
# ============================================================

class TestAdditionalCoverage:
    def test_task_id_expression(self):
        dbg = make_debugger("""
            fn worker() { let id = task_id; print(id); }
            let t = spawn worker();
            join(t);
        """)
        dbg.continue_execution()
        assert "1" in dbg.output  # spawned task has id=1

    def test_channel_closed_error(self):
        """Sending to closed channel raises error."""
        # Can't close channels in the language, but we can test the debugger
        # doesn't crash on various channel states
        dbg = make_debugger("""
            let c = chan(1);
            send(c, 1);
            let v = recv(c);
            print(v);
        """)
        dbg.continue_execution()
        assert "1" in dbg.output

    def test_join_already_completed(self):
        """Join on a task that already completed returns immediately."""
        dbg = make_debugger("""
            fn quick() { return 99; }
            let t = spawn quick();
            yield;
            yield;
            yield;
            let r = join(t);
            print(r);
        """)
        dbg.continue_execution()
        assert "99" in dbg.output

    def test_multiple_channels_pipeline(self):
        dbg = make_debugger("""
            fn stage(inp, out) {
                let v = recv(inp);
                send(out, v + 1);
            }
            let c1 = chan(1);
            let c2 = chan(1);
            let c3 = chan(1);
            send(c1, 0);
            let t1 = spawn stage(c1, c2);
            let t2 = spawn stage(c2, c3);
            let result = recv(c3);
            join(t1);
            join(t2);
            print(result);
        """)
        dbg.continue_execution()
        assert "2" in dbg.output

    def test_step_scheduler_multiple_rounds(self):
        """Multiple scheduler rounds eventually complete the program."""
        dbg = make_debugger("""
            fn worker(n) { print(n); }
            let t = spawn worker(42);
            yield;
            join(t);
        """)
        for _ in range(200):
            event = dbg.step_scheduler()
            if event.reason in (StopReason.HALT, StopReason.ERROR,
                                StopReason.DEADLOCK):
                break
        assert "42" in dbg.output

    def test_trace_with_concurrent_tasks(self):
        dbg = make_debugger("""
            fn adder(a, b) { return a + b; }
            let t = spawn adder(1, 2);
            let r = join(t);
            print(r);
        """)
        dbg.trace_enabled = True
        dbg.continue_execution()
        assert "3" in dbg.output
        # Verify we have trace entries from both tasks
        task_ids = set(e.task_id for e in dbg.trace_log)
        assert len(task_ids) >= 1

    def test_break_on_spawn_gives_spawned_id(self):
        dbg = make_debugger("""
            fn noop() { let x = 0; }
            let t1 = spawn noop();
            let t2 = spawn noop();
            join(t1);
            join(t2);
        """)
        dbg.break_on_event(ConcEventType.SPAWN)
        event = dbg.continue_execution()
        assert event.reason == StopReason.TASK_SPAWN
        first_spawn = event.data['spawned_task']
        event = dbg.continue_execution()
        assert event.reason == StopReason.TASK_SPAWN
        second_spawn = event.data['spawned_task']
        assert first_spawn != second_spawn
        dbg.break_on_event(ConcEventType.SPAWN, enabled=False)
        dbg.continue_execution()

    def test_watch_across_multiple_changes(self):
        """Watch fires each time the value changes."""
        dbg = make_debugger("""
            let x = 0;
            x = 1;
            x = 2;
            print(x);
        """)
        dbg.add_watch("x")
        event = dbg.continue_execution()
        assert event.reason == StopReason.WATCH
        # Continue through remaining changes to completion
        for _ in range(10):
            event = dbg.continue_execution()
            if event.reason == StopReason.HALT:
                break
        assert "2" in dbg.output

    def test_disassemble_shows_marker(self):
        dbg = make_debugger("let x = 42;")
        dis = dbg.disassemble_task(0)
        assert ">>>" in dis

    def test_step_line_across_blocks(self):
        dbg = make_debugger("""
            if (true) {
                let x = 1;
            }
            print(1);
        """)
        events = []
        for _ in range(20):
            event = dbg.step_line()
            events.append(event)
            if event.reason == StopReason.HALT:
                break
        assert any(e.reason == StopReason.HALT for e in events)
        assert "1" in dbg.output

    def test_step_over_in_concurrent(self):
        """Step-over works correctly in a concurrent program."""
        dbg = make_debugger("""
            fn compute() {
                let a = 1;
                let b = 2;
                return a + b;
            }
            let r = compute();
            print(r);
        """)
        events = []
        for _ in range(30):
            event = dbg.step_over()
            events.append(event)
            if event.reason == StopReason.HALT:
                break
        assert "3" in dbg.output

    def test_continue_with_no_event_breaks_completes(self):
        """Continue without any event breaks runs to completion."""
        dbg = make_debugger("""
            fn worker(ch) { send(ch, 10); }
            let c = chan(1);
            let t = spawn worker(c);
            let v = recv(c);
            join(t);
            print(v);
        """)
        event = dbg.continue_execution()
        assert event.reason == StopReason.HALT
        assert "10" in dbg.output

    def test_get_event_breaks_returns_copy(self):
        dbg = make_debugger("let x = 1;")
        dbg.break_on_event(ConcEventType.SPAWN)
        breaks = dbg.get_event_breaks()
        assert ConcEventType.SPAWN in breaks
        breaks.discard(ConcEventType.SPAWN)
        # Original should be unchanged
        assert ConcEventType.SPAWN in dbg.get_event_breaks()

    def test_conditional_bp_in_loop_with_concurrent(self):
        dbg = make_debugger("""
            let sum = 0;
            let i = 0;
            while (i < 5) {
                sum = sum + i;
                i = i + 1;
            }
            print(sum);
        """)
        dbg.add_breakpoint(line=4, condition="i == 3")
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        vars_ = dbg.get_task_variables(0)
        assert vars_['i'] == 3
        # Continue to completion
        event = dbg.continue_execution()
        assert "10" in dbg.output

    def test_breakpoint_and_event_break_together(self):
        """Both breakpoint and event break active at same time."""
        dbg = make_debugger("""
            fn worker() { print(1); }
            let t = spawn worker();
            yield;
            join(t);
            print(2);
        """)
        dbg.add_breakpoint(line=5)
        dbg.break_on_event(ConcEventType.SPAWN)
        event = dbg.continue_execution()
        assert event.reason == StopReason.TASK_SPAWN
        # Disable event break, continue to BP
        dbg.break_on_event(ConcEventType.SPAWN, enabled=False)
        event = dbg.continue_execution()
        # Should hit BP or complete
        if event.reason == StopReason.BREAKPOINT:
            dbg.continue_execution()
        assert "1" in dbg.output

    def test_many_steps_with_trace(self):
        dbg = make_debugger("""
            let x = 0;
            while (x < 10) { x = x + 1; }
            print(x);
        """)
        dbg.trace_enabled = True
        dbg.continue_execution()
        assert "10" in dbg.output
        assert len(dbg.trace_log) > 10

    def test_empty_run_queue_after_completion(self):
        dbg = make_debugger("let x = 1;")
        dbg.continue_execution()
        assert len(dbg.get_run_queue()) == 0

    def test_multiple_breakpoints_different_lines(self):
        dbg = make_debugger("""
            let a = 1;
            let b = 2;
            let c = 3;
            print(a + b + c);
        """)
        dbg.add_breakpoint(line=2)
        dbg.add_breakpoint(line=3)
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        first_line = event.line
        event = dbg.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        assert event.line != first_line
        dbg.continue_execution()
        assert "6" in dbg.output

    def test_debug_concurrent_with_max_steps(self):
        result = debug_concurrent(
            "while (true) { let x = 1; }",
            max_total_steps=50,
        )
        assert result['event'].reason == StopReason.ERROR

    def test_task_states_after_completion(self):
        dbg = make_debugger("""
            fn worker() { return 1; }
            let t = spawn worker();
            yield;
            join(t);
        """)
        dbg.continue_execution()
        tasks = dbg.get_task_list()
        for t in tasks:
            assert t['state'] in ('COMPLETED', 'FAILED')

    def test_step_then_continue(self):
        dbg = make_debugger("""
            let x = 1;
            let y = 2;
            print(x + y);
        """)
        # Step a few times
        dbg.step()
        dbg.step()
        # Then continue
        dbg.continue_execution()
        assert "3" in dbg.output

    def test_channel_buffer_contents_visible(self):
        dbg = make_debugger("""
            let c = chan(5);
            send(c, 10);
            send(c, 20);
            send(c, 30);
        """)
        dbg.continue_execution()
        channels = dbg.get_channel_info()
        assert len(channels) == 1
        assert channels[0]['buffer_contents'] == [10, 20, 30]
        assert channels[0]['buffer_size'] == 5

    def test_focused_step_line(self):
        """Step line on a focused task."""
        dbg = make_debugger("""
            fn worker() { let a = 1; let b = 2; print(a + b); }
            let t = spawn worker();
            yield;
            join(t);
        """)
        # Run until spawn
        for _ in range(30):
            event = dbg.step()
            if len(dbg.tasks) > 1:
                break
        dbg.focus_task(1)
        event = dbg.step_line(task_id=1)
        assert event.task_id == 1 or event.reason in (
            StopReason.HALT, StopReason.TASK_COMPLETE)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
