"""
Interactive REPL + Debugger for Stack VM
Challenge C020 -- AgentZero Session 021

Composition: Wraps C010 Stack VM with interactive debugging control.

REPL features:
  - Evaluate expressions and statements interactively
  - Persistent state across inputs (variables, functions)
  - Multi-line input detection (braces, incomplete constructs)
  - Command prefix (.) for REPL commands
  - History tracking

Debugger features:
  - Breakpoints (by line number or instruction address)
  - Single-step execution (instruction-level and line-level)
  - Continue (run until breakpoint or halt)
  - Stack inspection
  - Variable inspection / watch expressions
  - Call stack inspection
  - Disassembly view
  - Step-over (skip function bodies)
  - Step-out (run until current function returns)
  - Conditional breakpoints
  - Execution history / trace
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum, auto

# Import the VM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Op, Chunk, Token, TokenType, FnObject, CallFrame, VM, VMError,
    lex, Parser, Compiler, CompileError, LexError, ParseError, compile_source,
    execute, disassemble, Program
)


# ============================================================
# Debug VM -- extends VM with step-by-step execution control
# ============================================================

class StopReason(Enum):
    STEP = auto()          # single step completed
    BREAKPOINT = auto()    # hit a breakpoint
    HALT = auto()          # program finished
    ERROR = auto()         # runtime error
    WATCH = auto()         # watch expression triggered


@dataclass
class Breakpoint:
    id: int
    line: int = -1              # source line (-1 = unused)
    address: int = -1           # instruction address (-1 = unused)
    condition: str = ""         # condition expression (empty = unconditional)
    enabled: bool = True
    hit_count: int = 0


@dataclass
class DebugEvent:
    reason: StopReason
    ip: int
    line: int
    op: Optional[int] = None
    message: str = ""
    breakpoint_id: int = -1


@dataclass
class WatchEntry:
    id: int
    expression: str
    last_value: Any = None
    triggered: bool = False


class DebugVM(VM):
    """VM with debugging hooks for step-by-step execution."""

    def __init__(self, chunk: Chunk):
        super().__init__(chunk, trace=False)
        self.breakpoints: dict[int, Breakpoint] = {}
        self.watches: dict[int, WatchEntry] = {}
        self._next_bp_id = 1
        self._next_watch_id = 1
        self._halted = False
        self._error: Optional[str] = None
        self.trace_log: list[dict] = []
        self.trace_enabled = False
        self._step_over_depth: int = -1  # call stack depth for step-over
        self._step_out_depth: int = -1   # call stack depth for step-out
        self._last_bp_ip: int = -1       # IP where last breakpoint fired (skip on continue)
        self._last_bp_line: int = -1     # line where last breakpoint fired

    # -- Breakpoints --

    def add_breakpoint(self, line: int = -1, address: int = -1,
                       condition: str = "") -> int:
        bp_id = self._next_bp_id
        self._next_bp_id += 1
        self.breakpoints[bp_id] = Breakpoint(
            id=bp_id, line=line, address=address, condition=condition
        )
        return bp_id

    def remove_breakpoint(self, bp_id: int) -> bool:
        if bp_id in self.breakpoints:
            del self.breakpoints[bp_id]
            return True
        return False

    def enable_breakpoint(self, bp_id: int, enabled: bool = True) -> bool:
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].enabled = enabled
            return True
        return False

    def list_breakpoints(self) -> list[Breakpoint]:
        return list(self.breakpoints.values())

    # -- Watch expressions --

    def add_watch(self, expression: str) -> int:
        w_id = self._next_watch_id
        self._next_watch_id += 1
        self.watches[w_id] = WatchEntry(id=w_id, expression=expression)
        return w_id

    def remove_watch(self, w_id: int) -> bool:
        if w_id in self.watches:
            del self.watches[w_id]
            return True
        return False

    # -- Inspection --

    def get_stack(self) -> list:
        return list(self.stack)

    def get_variables(self) -> dict:
        return dict(self.env)

    def get_call_stack(self) -> list[dict]:
        frames = []
        for f in self.call_stack:
            frames.append({
                'ip': f.ip,
                'chunk_constants': len(f.chunk.constants),
                'env_snapshot': dict(f.base_env),
            })
        # Current frame
        frames.append({
            'ip': self.ip,
            'chunk_constants': len(self.current_chunk.constants),
            'env_snapshot': dict(self.env),
            'current': True,
        })
        return frames

    def get_current_line(self) -> int:
        if self.ip < len(self.current_chunk.lines):
            return self.current_chunk.lines[self.ip]
        return -1

    def get_current_op(self) -> Optional[int]:
        if self.ip < len(self.current_chunk.code):
            return self.current_chunk.code[self.ip]
        return None

    def is_halted(self) -> bool:
        return self._halted

    def get_error(self) -> Optional[str]:
        return self._error

    def disassemble_current(self, context: int = 3) -> str:
        """Disassemble around current IP."""
        lines = []
        i = 0
        entries = []  # (addr, text)
        chunk = self.current_chunk
        while i < len(chunk.code):
            op = chunk.code[i]
            op_name = Op(op).name if op in Op._value2member_map_ else f"??({op})"
            if op in (Op.CONST, Op.LOAD, Op.STORE, Op.JUMP, Op.JUMP_IF_FALSE,
                      Op.JUMP_IF_TRUE, Op.CALL):
                operand = chunk.code[i + 1]
                if op == Op.CONST:
                    val = chunk.constants[operand]
                    if isinstance(val, FnObject):
                        text = f"{op_name:20s} {operand} (fn:{val.name})"
                    else:
                        text = f"{op_name:20s} {operand} ({val!r})"
                elif op in (Op.LOAD, Op.STORE):
                    nm = chunk.names[operand]
                    text = f"{op_name:20s} {operand} ({nm})"
                else:
                    text = f"{op_name:20s} {operand}"
                entries.append((i, text))
                i += 2
            else:
                entries.append((i, op_name))
                i += 1

        # Find current entry index
        current_idx = -1
        for idx, (addr, _) in enumerate(entries):
            if addr == self.ip:
                current_idx = idx
                break
            if addr > self.ip:
                current_idx = max(0, idx - 1)
                break
        if current_idx == -1:
            current_idx = len(entries) - 1

        start = max(0, current_idx - context)
        end = min(len(entries), current_idx + context + 1)

        for idx in range(start, end):
            addr, text = entries[idx]
            marker = ">>>" if addr == self.ip else "   "
            lines.append(f"{marker} {addr:04d}  {text}")

        return '\n'.join(lines)

    # -- Execution control --

    def step(self) -> DebugEvent:
        """Execute one instruction and return."""
        if self._halted:
            return DebugEvent(StopReason.HALT, self.ip, self.get_current_line(),
                              message="Program has halted")

        return self._execute_one()

    def step_line(self) -> DebugEvent:
        """Execute until the source line changes."""
        if self._halted:
            return DebugEvent(StopReason.HALT, self.ip, self.get_current_line(),
                              message="Program has halted")

        start_line = self.get_current_line()
        event = self._execute_one()
        if event.reason in (StopReason.HALT, StopReason.ERROR):
            return event

        while self.get_current_line() == start_line and not self._halted:
            event = self._execute_one()
            if event.reason in (StopReason.HALT, StopReason.ERROR,
                                StopReason.BREAKPOINT, StopReason.WATCH):
                return event

        return DebugEvent(StopReason.STEP, self.ip, self.get_current_line())

    def step_over(self) -> DebugEvent:
        """Step over function calls -- execute until we return to same call depth."""
        if self._halted:
            return DebugEvent(StopReason.HALT, self.ip, self.get_current_line(),
                              message="Program has halted")

        target_depth = len(self.call_stack)
        start_line = self.get_current_line()

        # Execute without breakpoint checks -- step_over should not re-trigger
        # breakpoints on the current line or inside called functions
        while not self._halted:
            # If inside a function call, keep running until we return
            if len(self.call_stack) > target_depth:
                event = self._execute_one(check_bp=False)
                if event.reason in (StopReason.HALT, StopReason.ERROR):
                    return event
                continue

            # At the right depth -- check if line changed
            cur_line = self.get_current_line()
            if cur_line != start_line:
                break

            event = self._execute_one(check_bp=False)
            if event.reason in (StopReason.HALT, StopReason.ERROR):
                return event

        return DebugEvent(StopReason.STEP, self.ip, self.get_current_line())

    def step_out(self) -> DebugEvent:
        """Run until current function returns (call stack shrinks)."""
        if self._halted:
            return DebugEvent(StopReason.HALT, self.ip, self.get_current_line(),
                              message="Program has halted")

        target_depth = len(self.call_stack) - 1
        if target_depth < 0:
            # At top level -- just run to completion
            return self.continue_execution()

        while not self._halted:
            event = self._execute_one()
            if event.reason in (StopReason.HALT, StopReason.ERROR):
                return event
            if event.reason == StopReason.BREAKPOINT:
                return event
            if len(self.call_stack) <= target_depth:
                return DebugEvent(StopReason.STEP, self.ip, self.get_current_line(),
                                  message="Returned from function")

        return DebugEvent(StopReason.HALT, self.ip, self.get_current_line())

    def continue_execution(self) -> DebugEvent:
        """Run until breakpoint, watch trigger, or halt."""
        if self._halted:
            return DebugEvent(StopReason.HALT, self.ip, self.get_current_line(),
                              message="Program has halted")

        # Skip breakpoints at the line where we last stopped (avoid re-triggering)
        skip_line = self._last_bp_line
        self._last_bp_ip = -1
        self._last_bp_line = -1
        skipping = (skip_line >= 0)
        while not self._halted:
            cur_line = self.get_current_line()
            if skipping and cur_line != skip_line:
                skipping = False
            should_check = not skipping
            event = self._execute_one(check_bp=should_check)
            if event.reason == StopReason.BREAKPOINT:
                self._last_bp_ip = event.ip
                return event
            if event.reason != StopReason.STEP:
                return event

        return DebugEvent(StopReason.HALT, self.ip, self.get_current_line())

    def run_to_address(self, address: int) -> DebugEvent:
        """Run until reaching a specific instruction address."""
        if self._halted:
            return DebugEvent(StopReason.HALT, self.ip, self.get_current_line(),
                              message="Program has halted")

        while not self._halted:
            if self.ip == address:
                return DebugEvent(StopReason.STEP, self.ip, self.get_current_line(),
                                  message=f"Reached address {address}")
            event = self._execute_one()
            if event.reason in (StopReason.HALT, StopReason.ERROR,
                                StopReason.BREAKPOINT):
                return event

        return DebugEvent(StopReason.HALT, self.ip, self.get_current_line())

    # -- Internal execution --

    def _execute_one(self, check_bp: bool = True) -> DebugEvent:
        """Execute exactly one VM instruction."""
        self.step_count += 1
        if self.step_count > self.max_steps:
            self._halted = True
            self._error = f"Execution limit exceeded ({self.max_steps} steps)"
            return DebugEvent(StopReason.ERROR, self.ip, self.get_current_line(),
                              message=self._error)

        if self.ip >= len(self.current_chunk.code):
            self._halted = True
            return DebugEvent(StopReason.HALT, self.ip, self.get_current_line())

        cur_ip = self.ip
        cur_line = self.get_current_line()

        # Check breakpoints before executing
        if check_bp:
            bp_event = self._check_breakpoints(cur_ip, cur_line)
            if bp_event:
                return bp_event

        # Record trace
        if self.trace_enabled:
            op_val = self.current_chunk.code[self.ip]
            self.trace_log.append({
                'ip': self.ip,
                'op': Op(op_val).name if op_val in Op._value2member_map_ else str(op_val),
                'stack': list(self.stack[-5:]),
                'line': cur_line,
            })

        # Execute the instruction
        op = self.current_chunk.code[self.ip]
        self.ip += 1

        try:
            self._exec_op(op)
        except VMError as e:
            self._halted = True
            self._error = str(e)
            return DebugEvent(StopReason.ERROR, cur_ip, cur_line, op=op,
                              message=str(e))

        # Check watches
        watch_event = self._check_watches()
        if watch_event:
            return watch_event

        # Check if halted
        if self._halted:
            return DebugEvent(StopReason.HALT, cur_ip, cur_line, op=op)

        return DebugEvent(StopReason.STEP, cur_ip, cur_line, op=op)

    def _exec_op(self, op):
        """Execute a single opcode. Mirrors VM.run() logic."""
        if op == Op.HALT:
            self._halted = True

        elif op == Op.CONST:
            idx = self.current_chunk.code[self.ip]
            self.ip += 1
            self.push(self.current_chunk.constants[idx])

        elif op == Op.POP:
            if self.stack:
                self.pop()

        elif op == Op.DUP:
            self.push(self.peek())

        elif op == Op.ADD:
            b, a = self.pop(), self.pop()
            self.push(a + b)
        elif op == Op.SUB:
            b, a = self.pop(), self.pop()
            self.push(a - b)
        elif op == Op.MUL:
            b, a = self.pop(), self.pop()
            self.push(a * b)
        elif op == Op.DIV:
            b, a = self.pop(), self.pop()
            if b == 0:
                raise VMError("Division by zero")
            if isinstance(a, int) and isinstance(b, int):
                self.push(a // b)
            else:
                self.push(a / b)
        elif op == Op.MOD:
            b, a = self.pop(), self.pop()
            if b == 0:
                raise VMError("Modulo by zero")
            self.push(a % b)
        elif op == Op.NEG:
            self.push(-self.pop())

        elif op == Op.EQ:
            b, a = self.pop(), self.pop()
            self.push(a == b)
        elif op == Op.NE:
            b, a = self.pop(), self.pop()
            self.push(a != b)
        elif op == Op.LT:
            b, a = self.pop(), self.pop()
            self.push(a < b)
        elif op == Op.GT:
            b, a = self.pop(), self.pop()
            self.push(a > b)
        elif op == Op.LE:
            b, a = self.pop(), self.pop()
            self.push(a <= b)
        elif op == Op.GE:
            b, a = self.pop(), self.pop()
            self.push(a >= b)

        elif op == Op.NOT:
            self.push(not self.pop())
        elif op == Op.AND:
            b, a = self.pop(), self.pop()
            self.push(a and b)
        elif op == Op.OR:
            b, a = self.pop(), self.pop()
            self.push(a or b)

        elif op == Op.LOAD:
            idx = self.current_chunk.code[self.ip]
            self.ip += 1
            name = self.current_chunk.names[idx]
            if name not in self.env:
                raise VMError(f"Undefined variable '{name}'")
            self.push(self.env[name])

        elif op == Op.STORE:
            idx = self.current_chunk.code[self.ip]
            self.ip += 1
            name = self.current_chunk.names[idx]
            value = self.pop()
            self.env[name] = value

        elif op == Op.JUMP:
            target = self.current_chunk.code[self.ip]
            self.ip = target

        elif op == Op.JUMP_IF_FALSE:
            target = self.current_chunk.code[self.ip]
            self.ip += 1
            if not self.peek():
                self.ip = target

        elif op == Op.JUMP_IF_TRUE:
            target = self.current_chunk.code[self.ip]
            self.ip += 1
            if self.peek():
                self.ip = target

        elif op == Op.CALL:
            arg_count = self.current_chunk.code[self.ip]
            self.ip += 1
            args = []
            for _ in range(arg_count):
                args.insert(0, self.pop())
            fn_obj = self.pop()
            if not isinstance(fn_obj, FnObject):
                raise VMError(f"Cannot call non-function: {fn_obj}")
            if fn_obj.arity != arg_count:
                raise VMError(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")
            frame = CallFrame(self.current_chunk, self.ip, dict(self.env))
            self.call_stack.append(frame)
            self.current_chunk = fn_obj.chunk
            self.ip = 0
            for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                self.env[param_name] = args[i]

        elif op == Op.RETURN:
            return_val = self.pop()
            if not self.call_stack:
                self.push(return_val)
                self._halted = True
            else:
                frame = self.call_stack.pop()
                self.current_chunk = frame.chunk
                self.ip = frame.ip
                self.env = frame.base_env
                self.push(return_val)

        elif op == Op.PRINT:
            value = self.pop()
            text = str(value) if value is not None else "None"
            if isinstance(value, bool):
                text = "true" if value else "false"
            self.output.append(text)

        else:
            raise VMError(f"Unknown opcode: {op}")

    def _check_breakpoints(self, ip: int, line: int) -> Optional[DebugEvent]:
        for bp in self.breakpoints.values():
            if not bp.enabled:
                continue
            hit = False
            if bp.address >= 0 and bp.address == ip:
                hit = True
            elif bp.line >= 0 and bp.line == line:
                hit = True
            if hit:
                if bp.condition:
                    try:
                        result = self._eval_condition(bp.condition)
                        if not result:
                            continue
                    except Exception:
                        continue  # bad condition -- skip
                bp.hit_count += 1
                self._last_bp_ip = ip
                self._last_bp_line = line
                return DebugEvent(
                    StopReason.BREAKPOINT, ip, line,
                    breakpoint_id=bp.id,
                    message=f"Breakpoint {bp.id} hit (count: {bp.hit_count})"
                )
        return None

    def _check_watches(self) -> Optional[DebugEvent]:
        for w in self.watches.values():
            try:
                value = self._eval_condition(w.expression)
            except Exception:
                continue
            if value != w.last_value:
                old_val = w.last_value
                w.last_value = value
                w.triggered = True
                return DebugEvent(
                    StopReason.WATCH, self.ip, self.get_current_line(),
                    message=f"Watch {w.id}: '{w.expression}' changed from {old_val!r} to {value!r}"
                )
            w.triggered = False
        return None

    def _eval_condition(self, expr: str) -> Any:
        """Evaluate a simple expression in the current VM context.
        Supports variable names and simple comparisons."""
        expr = expr.strip()
        # Direct variable lookup
        if expr in self.env:
            return self.env[expr]
        # Try compiling and running as an expression in a temporary VM
        try:
            source = f"let __cond_result__ = {expr};"
            chunk, _ = compile_source(source)
            temp_vm = VM(chunk)
            temp_vm.env = dict(self.env)
            temp_vm.run()
            return temp_vm.env.get('__cond_result__')
        except Exception:
            raise ValueError(f"Cannot evaluate: {expr}")


# ============================================================
# REPL -- Interactive Read-Eval-Print Loop
# ============================================================

class REPLState:
    """Persistent state for the REPL across inputs."""
    def __init__(self):
        self.env: dict = {}           # persistent variables
        self.functions: dict = {}     # compiled functions
        self.history: list[str] = []  # input history
        self.output_history: list[dict] = []  # result history
        self.line_offset: int = 0     # cumulative line offset

    def add_history(self, source: str, result: dict):
        self.history.append(source)
        self.output_history.append(result)


class REPL:
    """Interactive REPL for the Stack VM language."""

    def __init__(self):
        self.state = REPLState()
        self._commands: dict[str, Callable] = {
            'help': self._cmd_help,
            'vars': self._cmd_vars,
            'history': self._cmd_history,
            'clear': self._cmd_clear,
            'reset': self._cmd_reset,
            'dis': self._cmd_disassemble,
            'debug': self._cmd_debug,
        }

    def eval(self, source: str) -> dict:
        """Evaluate source code in the REPL context.
        Returns dict with keys: result, output, error, env."""
        source = source.strip()
        if not source:
            return {'result': None, 'output': [], 'error': None, 'env': dict(self.state.env)}

        # Check for REPL commands
        if source.startswith('.'):
            cmd_parts = source[1:].split(None, 1)
            cmd_name = cmd_parts[0] if cmd_parts else ''
            cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ''
            if cmd_name in self._commands:
                return self._commands[cmd_name](cmd_args)
            return {'result': None, 'output': [],
                    'error': f"Unknown command: .{cmd_name}", 'env': dict(self.state.env)}

        # Try to compile and execute
        try:
            # Add semicolon if missing for expression evaluation
            eval_source = source
            if not source.endswith(';') and not source.endswith('}'):
                eval_source = source + ';'

            chunk, compiler = compile_source(eval_source)
            vm = VM(chunk)
            vm.env = dict(self.state.env)
            # Inject known functions
            for name, fn_obj in self.state.functions.items():
                idx = chunk.add_constant(fn_obj)
                name_idx = chunk.add_name(name)

            result = vm.run()

            # Update persistent state
            self.state.env.update(vm.env)
            # Capture any new functions from the compiler
            for name, fn_obj in compiler.functions.items():
                self.state.functions[name] = fn_obj
                self.state.env[name] = fn_obj

            result_dict = {
                'result': result,
                'output': vm.output,
                'error': None,
                'env': dict(self.state.env),
                'steps': vm.step_count,
            }
            self.state.add_history(source, result_dict)
            return result_dict

        except (LexError, ParseError, CompileError, VMError) as e:
            return {'result': None, 'output': [], 'error': str(e),
                    'env': dict(self.state.env)}

    def is_complete(self, source: str) -> bool:
        """Check if input is a complete construct (balanced braces)."""
        depth = 0
        in_string = False
        for c in source:
            if c == '"' and not in_string:
                in_string = True
            elif c == '"' and in_string:
                in_string = False
            elif not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
        return depth <= 0

    def get_history(self) -> list[str]:
        return list(self.state.history)

    # -- REPL commands --

    def _cmd_help(self, args: str) -> dict:
        help_text = [
            "REPL Commands:",
            "  .help     -- Show this help",
            "  .vars     -- Show all variables",
            "  .history  -- Show input history",
            "  .clear    -- Clear output history",
            "  .reset    -- Reset all state",
            "  .dis CODE -- Disassemble code",
            "  .debug CODE -- Debug code interactively",
        ]
        return {'result': None, 'output': help_text, 'error': None,
                'env': dict(self.state.env)}

    def _cmd_vars(self, args: str) -> dict:
        lines = []
        for name, val in sorted(self.state.env.items()):
            if isinstance(val, FnObject):
                lines.append(f"  {name}: fn({val.arity} args)")
            else:
                lines.append(f"  {name}: {val!r}")
        if not lines:
            lines.append("  (no variables)")
        return {'result': None, 'output': lines, 'error': None,
                'env': dict(self.state.env)}

    def _cmd_history(self, args: str) -> dict:
        lines = []
        for i, h in enumerate(self.state.history):
            lines.append(f"  [{i}] {h}")
        if not lines:
            lines.append("  (no history)")
        return {'result': None, 'output': lines, 'error': None,
                'env': dict(self.state.env)}

    def _cmd_clear(self, args: str) -> dict:
        self.state.output_history.clear()
        return {'result': None, 'output': ["History cleared"], 'error': None,
                'env': dict(self.state.env)}

    def _cmd_reset(self, args: str) -> dict:
        self.state = REPLState()
        return {'result': None, 'output': ["State reset"], 'error': None,
                'env': dict(self.state.env)}

    def _cmd_disassemble(self, args: str) -> dict:
        if not args.strip():
            return {'result': None, 'output': [],
                    'error': "Usage: .dis <code>", 'env': dict(self.state.env)}
        try:
            code = args.strip()
            if not code.endswith(';') and not code.endswith('}'):
                code += ';'
            chunk, _ = compile_source(code)
            text = disassemble(chunk)
            return {'result': None, 'output': text.split('\n'), 'error': None,
                    'env': dict(self.state.env)}
        except (LexError, ParseError, CompileError) as e:
            return {'result': None, 'output': [],
                    'error': str(e), 'env': dict(self.state.env)}

    def _cmd_debug(self, args: str) -> dict:
        if not args.strip():
            return {'result': None, 'output': [],
                    'error': "Usage: .debug <code>", 'env': dict(self.state.env)}
        try:
            code = args.strip()
            if not code.endswith(';') and not code.endswith('}'):
                code += ';'
            chunk, _ = compile_source(code)
            dbg = DebugVM(chunk)
            dbg.env = dict(self.state.env)
            return {'result': dbg, 'output': ["Debug session started"],
                    'error': None, 'env': dict(self.state.env)}
        except (LexError, ParseError, CompileError) as e:
            return {'result': None, 'output': [],
                    'error': str(e), 'env': dict(self.state.env)}


# ============================================================
# Debug Session -- high-level debugger interface
# ============================================================

class DebugSession:
    """High-level debugger that wraps a DebugVM with a command interface."""

    def __init__(self, source: str, env: Optional[dict] = None):
        if not source.strip().endswith(';') and not source.strip().endswith('}'):
            source = source.strip() + ';'
        chunk, self._compiler = compile_source(source)
        self.source = source
        self.source_lines = source.split('\n')
        self.vm = DebugVM(chunk)
        if env:
            self.vm.env = dict(env)
        self.events: list[DebugEvent] = []

    @classmethod
    def from_chunk(cls, chunk: Chunk, source: str = "",
                   env: Optional[dict] = None) -> 'DebugSession':
        """Create a debug session from a pre-compiled chunk."""
        session = cls.__new__(cls)
        session.source = source
        session.source_lines = source.split('\n') if source else []
        session._compiler = None
        session.vm = DebugVM(chunk)
        if env:
            session.vm.env = dict(env)
        session.events = []
        return session

    # -- Forwarding to DebugVM --

    def step(self) -> DebugEvent:
        event = self.vm.step()
        self.events.append(event)
        return event

    def step_line(self) -> DebugEvent:
        event = self.vm.step_line()
        self.events.append(event)
        return event

    def step_over(self) -> DebugEvent:
        event = self.vm.step_over()
        self.events.append(event)
        return event

    def step_out(self) -> DebugEvent:
        event = self.vm.step_out()
        self.events.append(event)
        return event

    def continue_execution(self) -> DebugEvent:
        event = self.vm.continue_execution()
        self.events.append(event)
        return event

    def run_to_address(self, address: int) -> DebugEvent:
        event = self.vm.run_to_address(address)
        self.events.append(event)
        return event

    def add_breakpoint(self, **kwargs) -> int:
        return self.vm.add_breakpoint(**kwargs)

    def remove_breakpoint(self, bp_id: int) -> bool:
        return self.vm.remove_breakpoint(bp_id)

    def add_watch(self, expression: str) -> int:
        return self.vm.add_watch(expression)

    def remove_watch(self, w_id: int) -> bool:
        return self.vm.remove_watch(w_id)

    # -- Inspection --

    def get_source_line(self, line_num: int) -> Optional[str]:
        if 1 <= line_num <= len(self.source_lines):
            return self.source_lines[line_num - 1]
        return None

    def get_context(self) -> dict:
        """Get full debug context -- stack, vars, position, etc."""
        return {
            'ip': self.vm.ip,
            'line': self.vm.get_current_line(),
            'halted': self.vm.is_halted(),
            'error': self.vm.get_error(),
            'stack': self.vm.get_stack(),
            'variables': self.vm.get_variables(),
            'call_stack': self.vm.get_call_stack(),
            'output': list(self.vm.output),
            'source_line': self.get_source_line(self.vm.get_current_line()),
            'disassembly': self.vm.disassemble_current(),
            'step_count': self.vm.step_count,
        }

    def get_disassembly(self, context: int = 3) -> str:
        return self.vm.disassemble_current(context)

    def get_full_disassembly(self) -> str:
        return disassemble(self.vm.current_chunk)

    def eval_expression(self, expr: str) -> dict:
        """Evaluate an expression in the current debug context."""
        try:
            value = self.vm._eval_condition(expr)
            return {'value': value, 'error': None}
        except Exception as e:
            return {'value': None, 'error': str(e)}

    def set_variable(self, name: str, value: Any):
        """Set a variable in the debug context."""
        self.vm.env[name] = value

    def get_trace(self) -> list[dict]:
        return list(self.vm.trace_log)

    def enable_trace(self, enabled: bool = True):
        self.vm.trace_enabled = enabled


# ============================================================
# Public API
# ============================================================

def create_repl() -> REPL:
    """Create a new REPL instance."""
    return REPL()


def debug(source: str, env: Optional[dict] = None) -> DebugSession:
    """Create a debug session for source code."""
    return DebugSession(source, env)


def debug_chunk(chunk: Chunk, source: str = "",
                env: Optional[dict] = None) -> DebugSession:
    """Create a debug session from a pre-compiled chunk."""
    return DebugSession.from_chunk(chunk, source, env)
