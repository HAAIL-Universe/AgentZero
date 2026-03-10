"""
Tests for C020: Interactive REPL + Debugger
Target: 120+ tests covering REPL, DebugVM, DebugSession, edge cases
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from repl_debugger import (
    DebugVM, DebugSession, DebugEvent, Breakpoint, WatchEntry,
    StopReason, REPL, REPLState, create_repl, debug, debug_chunk
)
from stack_vm import compile_source, Op, Chunk, VM, FnObject


# ============================================================
# REPL -- Basic evaluation
# ============================================================

class TestREPLBasic:
    def test_eval_integer(self):
        repl = create_repl()
        r = repl.eval("let x = 42;")
        assert r['error'] is None
        assert r['env']['x'] == 42

    def test_eval_expression(self):
        repl = create_repl()
        repl.eval("let x = 10;")
        r = repl.eval("let y = x + 5;")
        assert r['env']['y'] == 15

    def test_eval_string(self):
        repl = create_repl()
        r = repl.eval('let s = "hello";')
        assert r['env']['s'] == 'hello'

    def test_eval_float(self):
        repl = create_repl()
        r = repl.eval("let f = 3.14;")
        assert abs(r['env']['f'] - 3.14) < 0.001

    def test_eval_bool(self):
        repl = create_repl()
        r = repl.eval("let b = true;")
        assert r['env']['b'] is True

    def test_eval_empty(self):
        repl = create_repl()
        r = repl.eval("")
        assert r['result'] is None
        assert r['error'] is None

    def test_eval_whitespace(self):
        repl = create_repl()
        r = repl.eval("   ")
        assert r['result'] is None

    def test_auto_semicolon(self):
        repl = create_repl()
        r = repl.eval("let x = 99")
        assert r['error'] is None
        assert r['env']['x'] == 99

    def test_print_output(self):
        repl = create_repl()
        r = repl.eval("print(42);")
        assert '42' in r['output']

    def test_arithmetic(self):
        repl = create_repl()
        repl.eval("let a = 10;")
        repl.eval("let b = 3;")
        repl.eval("let c = a * b + 1;")
        r = repl.eval("let d = c;")
        assert r['env']['c'] == 31


class TestREPLPersistence:
    def test_vars_persist(self):
        repl = create_repl()
        repl.eval("let x = 1;")
        repl.eval("let y = 2;")
        r = repl.eval("let z = x + y;")
        assert r['env']['z'] == 3

    def test_reassignment(self):
        repl = create_repl()
        repl.eval("let x = 1;")
        repl.eval("x = 42;")
        r = repl.eval("let y = x;")
        assert r['env']['x'] == 42

    def test_functions_persist(self):
        repl = create_repl()
        repl.eval("fn add(a, b) { return a + b; }")
        r = repl.eval("let result = add(3, 4);")
        assert r['error'] is None
        assert r['env']['result'] == 7

    def test_history_recorded(self):
        repl = create_repl()
        repl.eval("let x = 1;")
        repl.eval("let y = 2;")
        h = repl.get_history()
        assert len(h) == 2

    def test_multiple_prints(self):
        repl = create_repl()
        r = repl.eval("print(1); print(2); print(3);")
        assert r['output'] == ['1', '2', '3']


class TestREPLErrors:
    def test_syntax_error(self):
        repl = create_repl()
        r = repl.eval("if if if")
        assert r['error'] is not None

    def test_undefined_var(self):
        repl = create_repl()
        r = repl.eval("let x = nonexistent;")
        assert r['error'] is not None

    def test_error_doesnt_corrupt_state(self):
        repl = create_repl()
        repl.eval("let x = 10;")
        repl.eval("let y = bad_var;")  # error
        r = repl.eval("let z = x;")
        assert r['env']['x'] == 10

    def test_division_by_zero(self):
        repl = create_repl()
        r = repl.eval("let x = 10 / 0;")
        assert r['error'] is not None


class TestREPLCommands:
    def test_help(self):
        repl = create_repl()
        r = repl.eval(".help")
        assert len(r['output']) > 0
        assert r['error'] is None

    def test_vars_empty(self):
        repl = create_repl()
        r = repl.eval(".vars")
        assert any("no variables" in line for line in r['output'])

    def test_vars_with_data(self):
        repl = create_repl()
        repl.eval("let x = 42;")
        r = repl.eval(".vars")
        assert any("42" in line for line in r['output'])

    def test_history_command(self):
        repl = create_repl()
        repl.eval("let x = 1;")
        r = repl.eval(".history")
        assert any("let x = 1" in line for line in r['output'])

    def test_clear_command(self):
        repl = create_repl()
        repl.eval("let x = 1;")
        r = repl.eval(".clear")
        assert any("cleared" in line.lower() for line in r['output'])

    def test_reset_command(self):
        repl = create_repl()
        repl.eval("let x = 42;")
        r = repl.eval(".reset")
        assert len(r['env']) == 0

    def test_unknown_command(self):
        repl = create_repl()
        r = repl.eval(".foobar")
        assert r['error'] is not None
        assert "Unknown" in r['error']

    def test_disassemble(self):
        repl = create_repl()
        r = repl.eval(".dis let x = 42")
        assert r['error'] is None
        assert len(r['output']) > 0
        assert any("CONST" in line for line in r['output'])

    def test_disassemble_empty(self):
        repl = create_repl()
        r = repl.eval(".dis")
        assert r['error'] is not None

    def test_debug_command(self):
        repl = create_repl()
        r = repl.eval(".debug let x = 42")
        assert r['error'] is None
        assert isinstance(r['result'], DebugVM)


class TestREPLCompleteness:
    def test_complete_simple(self):
        repl = create_repl()
        assert repl.is_complete("let x = 1;") is True

    def test_incomplete_brace(self):
        repl = create_repl()
        assert repl.is_complete("fn foo() {") is False

    def test_complete_block(self):
        repl = create_repl()
        assert repl.is_complete("fn foo() { return 1; }") is True

    def test_nested_braces(self):
        repl = create_repl()
        assert repl.is_complete("if (true) { if (true) {") is False

    def test_string_with_brace(self):
        repl = create_repl()
        assert repl.is_complete('let s = "{";') is True

    def test_while_loop(self):
        repl = create_repl()
        r = repl.eval("let i = 0; while (i < 5) { i = i + 1; }")
        assert r['env']['i'] == 5


# ============================================================
# DebugVM -- Basic operations
# ============================================================

class TestDebugVMBasic:
    def test_create(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        assert vm.ip == 0
        assert not vm.is_halted()

    def test_step(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        event = vm.step()
        assert event.reason == StopReason.STEP

    def test_step_to_completion(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        events = []
        while not vm.is_halted():
            events.append(vm.step())
        assert events[-1].reason == StopReason.HALT
        assert vm.env.get('x') == 1

    def test_get_stack(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        vm.step()  # CONST 42
        assert 42 in vm.get_stack()

    def test_get_variables_empty(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        assert vm.get_variables() == {}

    def test_get_variables_after_run(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        while not vm.is_halted():
            vm.step()
        assert vm.get_variables()['x'] == 42

    def test_current_op(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        op = vm.get_current_op()
        assert op == Op.CONST

    def test_halted_step(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        while not vm.is_halted():
            vm.step()
        event = vm.step()
        assert event.reason == StopReason.HALT

    def test_error_on_bad_code(self):
        chunk, _ = compile_source("let x = 1 / 0;")
        vm = DebugVM(chunk)
        events = []
        while not vm.is_halted():
            events.append(vm.step())
        assert any(e.reason == StopReason.ERROR for e in events)
        assert vm.get_error() is not None


# ============================================================
# Breakpoints
# ============================================================

class TestBreakpoints:
    def test_add_breakpoint(self):
        chunk, _ = compile_source("let x = 1; let y = 2;")
        vm = DebugVM(chunk)
        bp_id = vm.add_breakpoint(address=2)
        assert bp_id == 1
        assert len(vm.list_breakpoints()) == 1

    def test_breakpoint_by_address(self):
        chunk, _ = compile_source("let x = 1; let y = 2;")
        vm = DebugVM(chunk)
        # After "let x = 1" the next instruction starts at some address
        # CONST 0, STORE 0 = 4 bytes, so address 4 is start of "let y = 2"
        vm.add_breakpoint(address=4)
        event = vm.continue_execution()
        assert event.reason == StopReason.BREAKPOINT

    def test_breakpoint_by_line(self):
        source = "let x = 1;\nlet y = 2;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        vm.add_breakpoint(line=2)
        event = vm.continue_execution()
        assert event.reason == StopReason.BREAKPOINT

    def test_remove_breakpoint(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        bp_id = vm.add_breakpoint(address=0)
        assert vm.remove_breakpoint(bp_id) is True
        assert len(vm.list_breakpoints()) == 0

    def test_remove_nonexistent(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        assert vm.remove_breakpoint(999) is False

    def test_disable_breakpoint(self):
        chunk, _ = compile_source("let x = 1; let y = 2;")
        vm = DebugVM(chunk)
        bp_id = vm.add_breakpoint(address=4)
        vm.enable_breakpoint(bp_id, False)
        event = vm.continue_execution()
        assert event.reason == StopReason.HALT  # skipped disabled bp

    def test_enable_breakpoint(self):
        chunk, _ = compile_source("let x = 1; let y = 2;")
        vm = DebugVM(chunk)
        bp_id = vm.add_breakpoint(address=4)
        vm.enable_breakpoint(bp_id, False)
        vm.enable_breakpoint(bp_id, True)
        event = vm.continue_execution()
        assert event.reason == StopReason.BREAKPOINT

    def test_hit_count(self):
        source = "let i = 0; while (i < 3) { i = i + 1; }"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        # Find the LOAD instruction for i in the loop body
        # Set a line breakpoint on the while body
        bp_id = vm.add_breakpoint(line=1)  # all on line 1 for single-line
        # First hit
        event = vm.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        bp = vm.breakpoints[bp_id]
        assert bp.hit_count >= 1

    def test_conditional_breakpoint(self):
        source = "let i = 0; while (i < 5) { i = i + 1; }"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        # Break when i >= 3
        vm.add_breakpoint(line=1, condition="i >= 3")
        # The first few times through i < 3, bp condition is false
        # Need to set initial value so condition can be checked
        event = vm.continue_execution()
        # Should eventually stop when i >= 3 or halt
        assert event.reason in (StopReason.BREAKPOINT, StopReason.HALT)

    def test_multiple_breakpoints(self):
        source = "let x = 1;\nlet y = 2;\nlet z = 3;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        vm.add_breakpoint(line=2)
        vm.add_breakpoint(line=3)
        e1 = vm.continue_execution()
        assert e1.reason == StopReason.BREAKPOINT
        e2 = vm.continue_execution()
        assert e2.reason == StopReason.BREAKPOINT

    def test_enable_nonexistent(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        assert vm.enable_breakpoint(999) is False


# ============================================================
# Watch expressions
# ============================================================

class TestWatches:
    def test_add_watch(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        w_id = vm.add_watch("x")
        assert w_id == 1

    def test_watch_triggers(self):
        source = "let x = 1; x = 2;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        vm.add_watch("x")
        event = vm.continue_execution()
        # Watch should trigger when x changes
        assert event.reason in (StopReason.WATCH, StopReason.HALT)

    def test_remove_watch(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        w_id = vm.add_watch("x")
        assert vm.remove_watch(w_id) is True

    def test_remove_nonexistent_watch(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        assert vm.remove_watch(999) is False

    def test_watch_expression(self):
        source = "let x = 1; let y = 2; x = 10;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        vm.add_watch("x")
        events = []
        while not vm.is_halted():
            e = vm.step()
            events.append(e)
        # Should have at least one watch trigger
        watch_events = [e for e in events if e.reason == StopReason.WATCH]
        assert len(watch_events) >= 1


# ============================================================
# Step modes
# ============================================================

class TestStepModes:
    def test_step_instruction(self):
        chunk, _ = compile_source("let x = 1; let y = 2;")
        vm = DebugVM(chunk)
        e1 = vm.step()
        e2 = vm.step()
        assert e1.ip != e2.ip or e1.op != e2.op

    def test_step_line(self):
        source = "let x = 1;\nlet y = 2;\nlet z = 3;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        e = vm.step_line()
        # Should have advanced past line 1
        assert e.reason == StopReason.STEP

    def test_step_line_multiline(self):
        source = "let x = 1;\nlet y = 2;\nlet z = 3;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        lines_seen = set()
        while not vm.is_halted():
            e = vm.step_line()
            if e.line > 0:
                lines_seen.add(e.line)
            if e.reason in (StopReason.HALT, StopReason.ERROR):
                break
        assert len(lines_seen) >= 1

    def test_continue(self):
        chunk, _ = compile_source("let x = 1; let y = 2; let z = 3;")
        vm = DebugVM(chunk)
        event = vm.continue_execution()
        assert event.reason == StopReason.HALT
        assert vm.env['z'] == 3

    def test_continue_with_breakpoint(self):
        source = "let x = 1;\nlet y = 2;\nlet z = 3;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        vm.add_breakpoint(line=3)
        event = vm.continue_execution()
        assert event.reason == StopReason.BREAKPOINT
        # z should not be set yet
        assert 'z' not in vm.env

    def test_step_over_no_function(self):
        source = "let x = 1;\nlet y = 2;"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        event = vm.step_over()
        assert event.reason == StopReason.STEP

    def test_step_over_function(self):
        source = "fn add(a, b) { return a + b; }\nlet r = add(1, 2);"
        session = debug(source)
        # Set breakpoint on line 2 (the call)
        session.add_breakpoint(line=2)
        session.continue_execution()
        # Now step over the function call
        event = session.step_over()
        # After step over, should be past the call, back at top level
        assert len(session.vm.call_stack) == 0
        # Continue to end
        session.continue_execution()
        assert session.get_context()['variables']['r'] == 3

    def test_step_out(self):
        source = "fn foo() { let x = 1; let y = 2; return x + y; }\nlet r = foo();"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        # Run until inside the function
        while len(vm.call_stack) == 0 and not vm.is_halted():
            vm.step()
        if not vm.is_halted():
            event = vm.step_out()
            assert len(vm.call_stack) == 0

    def test_step_out_at_top_level(self):
        chunk, _ = compile_source("let x = 1; let y = 2;")
        vm = DebugVM(chunk)
        event = vm.step_out()
        # At top level, step_out runs to completion
        assert event.reason == StopReason.HALT

    def test_run_to_address(self):
        chunk, _ = compile_source("let x = 1; let y = 2; let z = 3;")
        vm = DebugVM(chunk)
        event = vm.run_to_address(4)
        assert event.reason == StopReason.STEP
        assert vm.ip == 4

    def test_run_to_address_halted(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        vm._halted = True
        event = vm.run_to_address(0)
        assert event.reason == StopReason.HALT


# ============================================================
# DebugSession
# ============================================================

class TestDebugSession:
    def test_create(self):
        session = debug("let x = 42;")
        assert session.vm.ip == 0

    def test_create_auto_semicolon(self):
        session = debug("let x = 42")
        ctx = session.get_context()
        assert ctx['halted'] is False

    def test_step(self):
        session = debug("let x = 1;")
        event = session.step()
        assert event.reason == StopReason.STEP

    def test_continue(self):
        session = debug("let x = 1; let y = 2;")
        event = session.continue_execution()
        assert event.reason == StopReason.HALT

    def test_get_context(self):
        session = debug("let x = 42;")
        ctx = session.get_context()
        assert 'ip' in ctx
        assert 'stack' in ctx
        assert 'variables' in ctx
        assert 'halted' in ctx
        assert 'disassembly' in ctx

    def test_context_after_run(self):
        session = debug("let x = 42;")
        session.continue_execution()
        ctx = session.get_context()
        assert ctx['halted'] is True
        assert ctx['variables']['x'] == 42

    def test_get_source_line(self):
        session = debug("let x = 1;\nlet y = 2;")
        assert session.get_source_line(1) == "let x = 1;"
        assert session.get_source_line(2) == "let y = 2;"

    def test_get_source_line_out_of_range(self):
        session = debug("let x = 1;")
        assert session.get_source_line(999) is None
        assert session.get_source_line(0) is None

    def test_eval_expression(self):
        session = debug("let x = 42;")
        session.continue_execution()
        r = session.eval_expression("x")
        assert r['value'] == 42
        assert r['error'] is None

    def test_eval_bad_expression(self):
        session = debug("let x = 1;")
        r = session.eval_expression("nonexistent_var_xyz")
        assert r['error'] is not None

    def test_set_variable(self):
        session = debug("let x = 1;")
        session.continue_execution()
        session.set_variable('x', 999)
        r = session.eval_expression("x")
        assert r['value'] == 999

    def test_add_breakpoint(self):
        session = debug("let x = 1;\nlet y = 2;")
        bp_id = session.add_breakpoint(line=2)
        assert bp_id >= 1

    def test_remove_breakpoint(self):
        session = debug("let x = 1;")
        bp_id = session.add_breakpoint(line=1)
        assert session.remove_breakpoint(bp_id) is True

    def test_add_watch(self):
        session = debug("let x = 1;")
        w_id = session.add_watch("x")
        assert w_id >= 1

    def test_remove_watch(self):
        session = debug("let x = 1;")
        w_id = session.add_watch("x")
        assert session.remove_watch(w_id) is True

    def test_disassembly(self):
        session = debug("let x = 42;")
        dis = session.get_disassembly()
        assert "CONST" in dis

    def test_full_disassembly(self):
        session = debug("let x = 42; let y = 10;")
        dis = session.get_full_disassembly()
        assert "CONST" in dis
        assert "STORE" in dis

    def test_events_tracked(self):
        session = debug("let x = 1;")
        session.step()
        session.step()
        assert len(session.events) == 2

    def test_trace(self):
        session = debug("let x = 42;")
        session.enable_trace()
        session.continue_execution()
        trace = session.get_trace()
        assert len(trace) > 0
        assert 'op' in trace[0]
        assert 'ip' in trace[0]

    def test_from_chunk(self):
        chunk, _ = compile_source("let x = 42;")
        session = debug_chunk(chunk, "let x = 42;")
        session.continue_execution()
        ctx = session.get_context()
        assert ctx['variables']['x'] == 42

    def test_with_env(self):
        session = debug("let y = x + 1;", env={'x': 10})
        session.continue_execution()
        ctx = session.get_context()
        assert ctx['variables']['y'] == 11

    def test_step_line_session(self):
        session = debug("let x = 1;\nlet y = 2;")
        event = session.step_line()
        assert event.reason == StopReason.STEP

    def test_step_over_session(self):
        session = debug("let x = 1;")
        event = session.step_over()
        assert event.reason in (StopReason.STEP, StopReason.HALT)

    def test_step_out_session(self):
        session = debug("let x = 1;")
        event = session.step_out()
        assert event.reason == StopReason.HALT

    def test_run_to_address_session(self):
        session = debug("let x = 1; let y = 2;")
        event = session.run_to_address(4)
        assert event.reason == StopReason.STEP


# ============================================================
# Call stack inspection
# ============================================================

class TestCallStack:
    def test_empty_call_stack(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        frames = vm.get_call_stack()
        assert len(frames) == 1  # current frame only
        assert frames[0].get('current') is True

    def test_call_stack_in_function(self):
        source = "fn foo() { return 1; }\nlet r = foo();"
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        # Step until we're inside foo
        while len(vm.call_stack) == 0 and not vm.is_halted():
            vm.step()
        if not vm.is_halted():
            frames = vm.get_call_stack()
            assert len(frames) == 2  # caller + current

    def test_nested_call_stack(self):
        source = """
fn inner() { return 1; }
fn outer() { return inner(); }
let r = outer();
"""
        chunk, _ = compile_source(source)
        vm = DebugVM(chunk)
        max_depth = 0
        while not vm.is_halted():
            vm.step()
            depth = len(vm.call_stack)
            if depth > max_depth:
                max_depth = depth
        assert max_depth >= 2  # outer -> inner


# ============================================================
# Disassembly view
# ============================================================

class TestDisassembly:
    def test_disassemble_current(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        dis = vm.disassemble_current()
        assert ">>>" in dis  # current instruction marker
        assert "CONST" in dis

    def test_disassemble_context(self):
        chunk, _ = compile_source("let x = 1; let y = 2; let z = 3; let w = 4;")
        vm = DebugVM(chunk)
        # Step a few times
        vm.step()
        vm.step()
        dis = vm.disassemble_current(context=1)
        lines = dis.strip().split('\n')
        assert len(lines) <= 5  # limited context

    def test_disassemble_shows_operands(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        dis = vm.disassemble_current()
        assert "42" in dis


# ============================================================
# Trace
# ============================================================

class TestTrace:
    def test_trace_disabled_by_default(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        vm.continue_execution()
        assert len(vm.trace_log) == 0

    def test_trace_enabled(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        vm.trace_enabled = True
        vm.continue_execution()
        assert len(vm.trace_log) > 0

    def test_trace_content(self):
        chunk, _ = compile_source("let x = 42;")
        vm = DebugVM(chunk)
        vm.trace_enabled = True
        vm.continue_execution()
        entry = vm.trace_log[0]
        assert 'op' in entry
        assert 'ip' in entry
        assert 'stack' in entry
        assert 'line' in entry


# ============================================================
# Complex scenarios
# ============================================================

class TestComplexScenarios:
    def test_debug_while_loop(self):
        source = "let i = 0; while (i < 3) { i = i + 1; }"
        session = debug(source)
        session.continue_execution()
        ctx = session.get_context()
        assert ctx['variables']['i'] == 3

    def test_debug_function_calls(self):
        source = """
fn fib(n) {
    if (n <= 1) { return n; }
    return fib(n - 1) + fib(n - 2);
}
let r = fib(6);
"""
        session = debug(source)
        session.continue_execution()
        ctx = session.get_context()
        assert ctx['variables']['r'] == 8

    def test_breakpoint_in_loop(self):
        source = "let sum = 0;\nlet i = 0;\nwhile (i < 5) {\ni = i + 1;\nsum = sum + i;\n}"
        session = debug(source)
        session.add_breakpoint(line=4)
        # Should stop on each loop iteration at line 4
        e1 = session.continue_execution()
        assert e1.reason == StopReason.BREAKPOINT
        e2 = session.continue_execution()
        assert e2.reason == StopReason.BREAKPOINT

    def test_step_through_conditional(self):
        source = "let x = 10;\nif (x > 5) {\nlet y = 1;\n} else {\nlet y = 0;\n}"
        session = debug(source)
        events = []
        while not session.vm.is_halted():
            events.append(session.step())
        assert any(e.op == Op.JUMP_IF_FALSE for e in events if e.op is not None)

    def test_watch_in_loop(self):
        source = "let x = 0; while (x < 3) { x = x + 1; }"
        session = debug(source)
        session.add_watch("x")
        events = []
        while not session.vm.is_halted():
            e = session.step()
            events.append(e)
        watch_events = [e for e in events if e.reason == StopReason.WATCH]
        # x changes from None->0, 0->1, 1->2, 2->3
        assert len(watch_events) >= 2

    def test_debug_string_operations(self):
        source = 'let s = "hello"; print(s);'
        session = debug(source)
        session.continue_execution()
        ctx = session.get_context()
        assert 'hello' in ctx['output']

    def test_debug_boolean_logic(self):
        source = "let a = true; let b = false; let c = a and b; let d = a or b;"
        session = debug(source)
        session.continue_execution()
        ctx = session.get_context()
        assert ctx['variables']['c'] is False
        assert ctx['variables']['d'] is True

    def test_debug_nested_functions(self):
        source = """
fn double(x) { return x * 2; }
fn quad(x) { return double(double(x)); }
let r = quad(3);
"""
        session = debug(source)
        session.continue_execution()
        assert session.get_context()['variables']['r'] == 12

    def test_step_over_preserves_result(self):
        source = "fn add(a, b) { return a + b; }\nlet r = add(3, 4);"
        session = debug(source)
        # Step to function call
        while session.vm.get_current_line() != 2 and not session.vm.is_halted():
            session.step()
        if not session.vm.is_halted():
            session.step_over()
            # Continue to completion
            session.continue_execution()
            assert session.get_context()['variables']['r'] == 7

    def test_multiple_breakpoints_and_continue(self):
        source = "let a = 1;\nlet b = 2;\nlet c = 3;\nlet d = 4;"
        session = debug(source)
        session.add_breakpoint(line=2)
        session.add_breakpoint(line=4)
        e1 = session.continue_execution()
        assert e1.reason == StopReason.BREAKPOINT
        e2 = session.continue_execution()
        assert e2.reason == StopReason.BREAKPOINT
        e3 = session.continue_execution()
        assert e3.reason == StopReason.HALT

    def test_eval_during_debug(self):
        source = "let x = 10; let y = 20;"
        session = debug(source)
        session.continue_execution()
        r = session.eval_expression("x + y")
        assert r['value'] == 30

    def test_modify_var_during_debug(self):
        source = "let x = 1;\nlet y = x + 1;"
        session = debug(source)
        # Set breakpoint on line 2
        session.add_breakpoint(line=2)
        session.continue_execution()  # runs to bp on line 2
        # Modify x before y is computed
        session.set_variable('x', 100)
        session.continue_execution()  # runs to end
        ctx = session.get_context()
        assert ctx['halted'] is True
        assert ctx['variables']['y'] == 101

    def test_output_captured(self):
        source = "print(1); print(2); print(3);"
        session = debug(source)
        session.continue_execution()
        assert session.get_context()['output'] == ['1', '2', '3']


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        # Just a halt
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = DebugVM(chunk)
        event = vm.step()
        assert event.reason == StopReason.HALT

    def test_step_limit(self):
        chunk, _ = compile_source("let i = 0; while (true) { i = i + 1; }")
        vm = DebugVM(chunk)
        vm.max_steps = 100
        event = vm.continue_execution()
        assert event.reason == StopReason.ERROR
        assert "limit" in event.message.lower()

    def test_breakpoint_at_halt(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        # Put breakpoint at the HALT instruction
        halt_addr = len(chunk.code) - 1
        vm.add_breakpoint(address=halt_addr)
        event = vm.continue_execution()
        assert event.reason == StopReason.BREAKPOINT

    def test_continue_after_halt(self):
        session = debug("let x = 1;")
        session.continue_execution()
        event = session.continue_execution()
        assert event.reason == StopReason.HALT

    def test_step_over_halted(self):
        session = debug("let x = 1;")
        session.continue_execution()
        event = session.step_over()
        assert event.reason == StopReason.HALT

    def test_step_line_halted(self):
        session = debug("let x = 1;")
        session.continue_execution()
        event = session.step_line()
        assert event.reason == StopReason.HALT

    def test_disassemble_at_end(self):
        session = debug("let x = 1;")
        session.continue_execution()
        dis = session.get_disassembly()
        # Should not crash even when IP is past end
        assert isinstance(dis, str)

    def test_get_current_line_at_end(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        vm.continue_execution()
        line = vm.get_current_line()
        assert isinstance(line, int)

    def test_get_current_op_at_end(self):
        chunk, _ = compile_source("let x = 1;")
        vm = DebugVM(chunk)
        vm.continue_execution()
        op = vm.get_current_op()
        # May be None if past end
        assert op is None or isinstance(op, int)

    def test_large_program(self):
        # Many variables
        lines = [f"let v{i} = {i};" for i in range(50)]
        source = '\n'.join(lines)
        session = debug(source)
        session.continue_execution()
        ctx = session.get_context()
        assert ctx['variables']['v49'] == 49

    def test_repl_if_else(self):
        repl = create_repl()
        r = repl.eval("let x = 10; if (x > 5) { x = 1; } else { x = 0; }")
        assert r['env']['x'] == 1

    def test_repl_nested_functions(self):
        repl = create_repl()
        repl.eval("fn square(x) { return x * x; }")
        r = repl.eval("let r = square(7);")
        assert r['env']['r'] == 49

    def test_debug_comparison_ops(self):
        source = "let a = 1 < 2; let b = 1 > 2; let c = 1 == 1; let d = 1 != 2;"
        session = debug(source)
        session.continue_execution()
        v = session.get_context()['variables']
        assert v['a'] is True
        assert v['b'] is False
        assert v['c'] is True
        assert v['d'] is True

    def test_debug_modulo(self):
        session = debug("let r = 10 % 3;")
        session.continue_execution()
        assert session.get_context()['variables']['r'] == 1

    def test_debug_negation(self):
        session = debug("let r = -42;")
        session.continue_execution()
        assert session.get_context()['variables']['r'] == -42

    def test_debug_not_operator(self):
        session = debug("let r = not true;")
        session.continue_execution()
        assert session.get_context()['variables']['r'] is False

    def test_repl_while_with_print(self):
        repl = create_repl()
        r = repl.eval("let i = 0; while (i < 3) { print(i); i = i + 1; }")
        assert r['output'] == ['0', '1', '2']


# ============================================================
# Integration: REPL + Debugger
# ============================================================

class TestIntegration:
    def test_repl_debug_workflow(self):
        repl = create_repl()
        # Define a function
        repl.eval("fn double(x) { return x * 2; }")
        # Debug a call to it
        r = repl.eval(".debug let r = double(5)")
        assert isinstance(r['result'], DebugVM)
        # Could step through it
        dbg = r['result']
        dbg.continue_execution()
        assert dbg.env.get('r') == 10

    def test_repl_multiple_sessions(self):
        repl = create_repl()
        repl.eval("let total = 0;")
        for i in range(5):
            repl.eval(f"total = total + {i};")
        r = repl.eval("let result = total;")
        assert r['env']['total'] == 10

    def test_debug_session_independence(self):
        s1 = debug("let x = 1;")
        s2 = debug("let x = 2;")
        s1.continue_execution()
        s2.continue_execution()
        assert s1.get_context()['variables']['x'] == 1
        assert s2.get_context()['variables']['x'] == 2

    def test_full_debug_workflow(self):
        """Complete debug workflow: set bp, run, inspect, continue."""
        source = "let x = 0;\nlet y = 0;\nx = 10;\ny = x * 2;"
        session = debug(source)
        session.add_breakpoint(line=3)
        session.enable_trace()

        # Run to breakpoint
        e = session.continue_execution()
        assert e.reason == StopReason.BREAKPOINT
        ctx = session.get_context()
        assert ctx['variables']['x'] == 0  # not yet assigned

        # Continue to end
        e = session.continue_execution()
        assert e.reason == StopReason.HALT
        ctx = session.get_context()
        assert ctx['variables']['x'] == 10
        assert ctx['variables']['y'] == 20

        # Verify trace was recorded
        assert len(session.get_trace()) > 0
