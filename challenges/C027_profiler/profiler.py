"""
Profiler -- Runtime Performance Analysis for Stack VM
Challenge C027 -- AgentZero Session 028

Composes: C010 (Stack VM)

Features:
  - Function call profiling (count, total time, self time, avg time)
  - Line-level execution counts (coverage)
  - Call graph construction (caller -> callee edges with counts)
  - Hotspot detection (most-executed lines and functions)
  - Stack depth tracking (peak memory usage proxy)
  - Instruction-level profiling (opcode frequency and time)
  - Sampling profiler mode (statistical profiling)
  - Profile comparison (diff two profiles)
  - Flame graph data generation
  - Profile serialization (export/import)

Architecture:
  ProfiledVM subclasses the concept by wrapping C010's VM execution loop
  with instrumentation hooks. The profiler collects raw data, then analysis
  functions produce reports.
"""

import sys
import os
import time
import json
import copy
from dataclasses import dataclass, field
from typing import Any, Optional

# Import C010 Stack VM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Op, Chunk, Token, TokenType, lex, Parser, Compiler, VM, FnObject,
    compile_source, execute, disassemble, VMError, CallFrame
)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class FunctionProfile:
    """Profile data for a single function."""
    name: str
    call_count: int = 0
    total_time: float = 0.0  # wall time including callees
    self_time: float = 0.0   # wall time excluding callees
    total_steps: int = 0     # instruction count including callees
    self_steps: int = 0      # instruction count excluding callees
    max_stack_depth: int = 0
    callers: dict = field(default_factory=dict)   # caller_name -> count
    callees: dict = field(default_factory=dict)    # callee_name -> count


@dataclass
class LineProfile:
    """Profile data for a single source line."""
    line: int
    execution_count: int = 0
    total_time: float = 0.0
    instructions: int = 0


@dataclass
class InstructionProfile:
    """Profile data for an opcode type."""
    opcode: str
    count: int = 0
    total_time: float = 0.0


@dataclass
class CallGraphEdge:
    """An edge in the call graph."""
    caller: str
    callee: str
    count: int = 0
    total_time: float = 0.0


@dataclass
class FlameFrame:
    """A frame in a flame graph stack."""
    name: str
    duration: float = 0.0
    children: list = field(default_factory=list)


@dataclass
class ProfileSnapshot:
    """Complete profile data snapshot."""
    functions: dict = field(default_factory=dict)      # name -> FunctionProfile
    lines: dict = field(default_factory=dict)           # line_num -> LineProfile
    instructions: dict = field(default_factory=dict)    # opcode_name -> InstructionProfile
    call_graph: list = field(default_factory=list)      # list of CallGraphEdge
    total_steps: int = 0
    total_time: float = 0.0
    peak_stack_depth: int = 0
    peak_call_depth: int = 0
    sample_count: int = 0  # for sampling profiler


# ============================================================
# Profiled VM
# ============================================================

class ProfiledVM(VM):
    """VM with profiling instrumentation."""

    def __init__(self, chunk: Chunk, trace=False):
        super().__init__(chunk, trace=trace)
        self.profile = ProfileSnapshot()
        self._call_stack_names = ["<main>"]  # track function names
        self._call_entry_times = [time.perf_counter()]
        self._call_entry_steps = [0]
        self._callee_time = [0.0]  # time spent in callees for each frame
        self._callee_steps = [0]   # steps spent in callees for each frame
        self._peak_stack = 0
        self._peak_call_depth = 0
        self._sampling = False
        self._sample_interval = 100  # sample every N steps
        self._samples = []  # list of stack snapshots

        # Initialize main function profile
        self.profile.functions["<main>"] = FunctionProfile(name="<main>")

    def run(self):
        start_time = time.perf_counter()
        self._call_entry_times[0] = start_time

        while True:
            self.step_count += 1
            if self.step_count > self.max_steps:
                raise VMError(f"Execution limit exceeded ({self.max_steps} steps)")

            if self.ip >= len(self.current_chunk.code):
                break

            # Track stack depth
            stack_depth = len(self.stack)
            if stack_depth > self._peak_stack:
                self._peak_stack = stack_depth

            # Track call depth
            call_depth = len(self._call_stack_names)
            if call_depth > self._peak_call_depth:
                self._peak_call_depth = call_depth

            # Get current line
            line = 0
            if self.ip < len(self.current_chunk.lines):
                line = self.current_chunk.lines[self.ip]

            op = self.current_chunk.code[self.ip]
            self.ip += 1

            if self.trace:
                self._trace_op(op)

            # Time this instruction
            inst_start = time.perf_counter()

            # Profile instruction
            op_name = Op(op).name if op in Op._value2member_map_ else f"UNKNOWN_{op}"
            if op_name not in self.profile.instructions:
                self.profile.instructions[op_name] = InstructionProfile(opcode=op_name)
            self.profile.instructions[op_name].count += 1

            # Profile line
            if line > 0:
                if line not in self.profile.lines:
                    self.profile.lines[line] = LineProfile(line=line)
                self.profile.lines[line].execution_count += 1
                self.profile.lines[line].instructions += 1

            # Sampling
            if self._sampling and self.step_count % self._sample_interval == 0:
                self._samples.append(list(self._call_stack_names))
                self.profile.sample_count += 1

            # Execute instruction
            if op == Op.HALT:
                inst_end = time.perf_counter()
                inst_time = inst_end - inst_start
                self.profile.instructions[op_name].total_time += inst_time
                if line > 0 and line in self.profile.lines:
                    self.profile.lines[line].total_time += inst_time
                break

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

                # Profile: record call
                caller = self._call_stack_names[-1]
                callee = fn_obj.name

                if callee not in self.profile.functions:
                    self.profile.functions[callee] = FunctionProfile(name=callee)
                self.profile.functions[callee].call_count += 1

                # Track caller/callee relationships
                fp = self.profile.functions[callee]
                fp.callers[caller] = fp.callers.get(caller, 0) + 1

                caller_fp = self.profile.functions[caller]
                caller_fp.callees[callee] = caller_fp.callees.get(callee, 0) + 1

                # Push call stack
                self._call_stack_names.append(callee)
                self._call_entry_times.append(time.perf_counter())
                self._call_entry_steps.append(self.step_count)
                self._callee_time.append(0.0)
                self._callee_steps.append(0)

                # Save current state
                frame = CallFrame(self.current_chunk, self.ip, dict(self.env))
                self.call_stack.append(frame)

                # Set up function execution
                self.current_chunk = fn_obj.chunk
                self.ip = 0
                for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                    self.env[param_name] = args[i]

            elif op == Op.RETURN:
                return_val = self.pop()

                # Profile: record return
                now = time.perf_counter()
                fn_name = self._call_stack_names[-1]
                entry_time = self._call_entry_times[-1]
                entry_steps = self._call_entry_steps[-1]
                callee_time = self._callee_time[-1]
                callee_steps_val = self._callee_steps[-1]

                total_time_spent = now - entry_time
                total_steps_spent = self.step_count - entry_steps
                self_time_spent = total_time_spent - callee_time
                self_steps_spent = total_steps_spent - callee_steps_val

                fp = self.profile.functions[fn_name]
                fp.total_time += total_time_spent
                fp.self_time += self_time_spent
                fp.total_steps += total_steps_spent
                fp.self_steps += self_steps_spent
                fp.max_stack_depth = max(fp.max_stack_depth, len(self.stack))

                # Pop call stack
                self._call_stack_names.pop()
                self._call_entry_times.pop()
                self._call_entry_steps.pop()
                self._callee_time.pop()
                self._callee_steps.pop()

                # Add this call's time/steps to parent's callee tracking
                if self._callee_time:
                    self._callee_time[-1] += total_time_spent
                    self._callee_steps[-1] += total_steps_spent

                if not self.call_stack:
                    self.push(return_val)
                    break
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

            # Time this instruction
            inst_end = time.perf_counter()
            inst_time = inst_end - inst_start
            self.profile.instructions[op_name].total_time += inst_time
            if line > 0 and line in self.profile.lines:
                self.profile.lines[line].total_time += inst_time

        # Finalize main function profile
        end_time = time.perf_counter()
        main_fp = self.profile.functions["<main>"]
        main_total = end_time - self._call_entry_times[0]
        main_steps = self.step_count - self._call_entry_steps[0]
        main_fp.total_time = main_total
        main_fp.total_steps = main_steps
        main_fp.self_time = main_total - self._callee_time[0]
        main_fp.self_steps = main_steps - self._callee_steps[0]
        main_fp.call_count = 1
        main_fp.max_stack_depth = self._peak_stack

        self.profile.total_steps = self.step_count
        self.profile.total_time = main_total
        self.profile.peak_stack_depth = self._peak_stack
        self.profile.peak_call_depth = self._peak_call_depth

        # Build call graph edges
        self._build_call_graph()

        return self.stack[-1] if self.stack else None

    def _build_call_graph(self):
        """Build call graph edges from function profiles."""
        edges = []
        for name, fp in self.profile.functions.items():
            for callee_name, count in fp.callees.items():
                edge = CallGraphEdge(
                    caller=name,
                    callee=callee_name,
                    count=count,
                )
                edges.append(edge)
        self.profile.call_graph = edges


# ============================================================
# Profiler (High-Level API)
# ============================================================

class Profiler:
    """High-level profiling interface."""

    def __init__(self):
        self.profiles = []  # list of ProfileSnapshot
        self._sampling = False
        self._sample_interval = 100

    def enable_sampling(self, interval=100):
        """Enable sampling mode with given step interval."""
        self._sampling = True
        self._sample_interval = interval

    def disable_sampling(self):
        """Disable sampling mode."""
        self._sampling = False

    def profile(self, source: str, trace=False) -> dict:
        """Profile a program. Returns execution result + profile."""
        chunk, compiler = compile_source(source)
        vm = ProfiledVM(chunk, trace=trace)
        vm._sampling = self._sampling
        vm._sample_interval = self._sample_interval
        result = vm.run()
        self.profiles.append(vm.profile)
        return {
            'result': result,
            'output': vm.output,
            'env': vm.env,
            'steps': vm.step_count,
            'profile': vm.profile,
        }

    def get_latest_profile(self) -> Optional[ProfileSnapshot]:
        """Get the most recent profile."""
        return self.profiles[-1] if self.profiles else None

    def flat_profile(self, profile: Optional[ProfileSnapshot] = None) -> list:
        """Generate flat profile sorted by self_time descending.
        Returns list of dicts with function stats."""
        p = profile or self.get_latest_profile()
        if not p:
            return []
        result = []
        for name, fp in p.functions.items():
            avg_time = fp.total_time / fp.call_count if fp.call_count > 0 else 0.0
            avg_steps = fp.total_steps / fp.call_count if fp.call_count > 0 else 0
            pct_time = (fp.self_time / p.total_time * 100) if p.total_time > 0 else 0.0
            pct_steps = (fp.self_steps / p.total_steps * 100) if p.total_steps > 0 else 0.0
            result.append({
                'name': name,
                'call_count': fp.call_count,
                'total_time': fp.total_time,
                'self_time': fp.self_time,
                'avg_time': avg_time,
                'total_steps': fp.total_steps,
                'self_steps': fp.self_steps,
                'avg_steps': avg_steps,
                'pct_time': pct_time,
                'pct_steps': pct_steps,
            })
        result.sort(key=lambda x: x['self_time'], reverse=True)
        return result

    def hotspots(self, profile: Optional[ProfileSnapshot] = None, top_n: int = 10) -> list:
        """Get top N hotspot lines by execution count."""
        p = profile or self.get_latest_profile()
        if not p:
            return []
        lines = sorted(p.lines.values(), key=lambda l: l.execution_count, reverse=True)
        return [
            {
                'line': lp.line,
                'execution_count': lp.execution_count,
                'total_time': lp.total_time,
                'instructions': lp.instructions,
            }
            for lp in lines[:top_n]
        ]

    def call_graph(self, profile: Optional[ProfileSnapshot] = None) -> list:
        """Get call graph as list of edge dicts."""
        p = profile or self.get_latest_profile()
        if not p:
            return []
        return [
            {
                'caller': e.caller,
                'callee': e.callee,
                'count': e.count,
            }
            for e in p.call_graph
        ]

    def instruction_profile(self, profile: Optional[ProfileSnapshot] = None) -> list:
        """Get instruction-level profile sorted by count."""
        p = profile or self.get_latest_profile()
        if not p:
            return []
        result = [
            {
                'opcode': ip.opcode,
                'count': ip.count,
                'total_time': ip.total_time,
                'avg_time': ip.total_time / ip.count if ip.count > 0 else 0.0,
                'pct': (ip.count / p.total_steps * 100) if p.total_steps > 0 else 0.0,
            }
            for ip in p.instructions.values()
        ]
        result.sort(key=lambda x: x['count'], reverse=True)
        return result

    def coverage(self, profile: Optional[ProfileSnapshot] = None) -> dict:
        """Get line coverage information."""
        p = profile or self.get_latest_profile()
        if not p:
            return {'covered_lines': [], 'uncovered_lines': [], 'total_covered': 0}
        covered = sorted(p.lines.keys())
        return {
            'covered_lines': covered,
            'total_covered': len(covered),
            'line_counts': {line: lp.execution_count for line, lp in p.lines.items()},
        }

    def summary(self, profile: Optional[ProfileSnapshot] = None) -> dict:
        """Get a summary of profiling results."""
        p = profile or self.get_latest_profile()
        if not p:
            return {}
        fn_count = len(p.functions)
        total_calls = sum(fp.call_count for fp in p.functions.values())
        hottest_fn = max(p.functions.values(), key=lambda f: f.self_time) if p.functions else None
        hottest_line = max(p.lines.values(), key=lambda l: l.execution_count) if p.lines else None

        return {
            'total_time': p.total_time,
            'total_steps': p.total_steps,
            'function_count': fn_count,
            'total_calls': total_calls,
            'peak_stack_depth': p.peak_stack_depth,
            'peak_call_depth': p.peak_call_depth,
            'hottest_function': hottest_fn.name if hottest_fn else None,
            'hottest_function_self_time': hottest_fn.self_time if hottest_fn else 0.0,
            'hottest_line': hottest_line.line if hottest_line else None,
            'hottest_line_count': hottest_line.execution_count if hottest_line else 0,
            'sample_count': p.sample_count,
        }

    def flame_data(self, profile: Optional[ProfileSnapshot] = None) -> dict:
        """Generate flame graph data from sampling or call data.
        Returns nested dict structure suitable for visualization."""
        p = profile or self.get_latest_profile()
        if not p:
            return {'name': '<root>', 'value': 0, 'children': []}

        # Build from call graph edges and function profiles
        root = {'name': '<root>', 'value': p.total_steps, 'children': []}

        # Build tree from function call relationships
        fn_children = {}  # parent -> [(child, steps)]
        for edge in p.call_graph:
            if edge.caller not in fn_children:
                fn_children[edge.caller] = []
            callee_fp = p.functions.get(edge.callee)
            callee_steps = callee_fp.total_steps if callee_fp else 0
            fn_children[edge.caller].append((edge.callee, callee_steps, edge.count))

        def build_tree(name, depth=0):
            fp = p.functions.get(name)
            node = {
                'name': name,
                'value': fp.self_steps if fp else 0,
                'total': fp.total_steps if fp else 0,
                'children': [],
            }
            if depth < 50 and name in fn_children:  # depth limit to prevent cycles
                for child_name, child_steps, count in fn_children[name]:
                    child_node = build_tree(child_name, depth + 1)
                    child_node['call_count'] = count
                    node['children'].append(child_node)
            return node

        # Start from <main>
        root = build_tree('<main>')
        return root

    def samples(self, profile: Optional[ProfileSnapshot] = None) -> list:
        """Get raw samples from sampling profiler.
        Each sample is a list of function names (stack trace)."""
        p = profile or self.get_latest_profile()
        if not p:
            return []
        # Samples are stored on the VM, but we capture sample_count in profile
        # For actual samples, we need them from the VM
        return []  # Samples need to be retrieved from VM directly

    def compare(self, profile_a: ProfileSnapshot, profile_b: ProfileSnapshot) -> dict:
        """Compare two profiles. Returns differences."""
        result = {
            'time_diff': profile_b.total_time - profile_a.total_time,
            'time_ratio': profile_b.total_time / profile_a.total_time if profile_a.total_time > 0 else float('inf'),
            'steps_diff': profile_b.total_steps - profile_a.total_steps,
            'steps_ratio': profile_b.total_steps / profile_a.total_steps if profile_a.total_steps > 0 else float('inf'),
            'stack_diff': profile_b.peak_stack_depth - profile_a.peak_stack_depth,
            'call_depth_diff': profile_b.peak_call_depth - profile_a.peak_call_depth,
            'function_diffs': [],
        }

        all_fns = set(profile_a.functions.keys()) | set(profile_b.functions.keys())
        for fn_name in sorted(all_fns):
            a = profile_a.functions.get(fn_name)
            b = profile_b.functions.get(fn_name)

            fn_diff = {'name': fn_name}
            if a and b:
                fn_diff['calls_diff'] = b.call_count - a.call_count
                fn_diff['self_time_diff'] = b.self_time - a.self_time
                fn_diff['self_steps_diff'] = b.self_steps - a.self_steps
                fn_diff['status'] = 'changed'
            elif a and not b:
                fn_diff['calls_diff'] = -a.call_count
                fn_diff['self_time_diff'] = -a.self_time
                fn_diff['self_steps_diff'] = -a.self_steps
                fn_diff['status'] = 'removed'
            else:
                fn_diff['calls_diff'] = b.call_count
                fn_diff['self_time_diff'] = b.self_time
                fn_diff['self_steps_diff'] = b.self_steps
                fn_diff['status'] = 'added'
            result['function_diffs'].append(fn_diff)

        return result

    def export_profile(self, profile: Optional[ProfileSnapshot] = None) -> dict:
        """Export profile to serializable dict."""
        p = profile or self.get_latest_profile()
        if not p:
            return {}
        return {
            'total_time': p.total_time,
            'total_steps': p.total_steps,
            'peak_stack_depth': p.peak_stack_depth,
            'peak_call_depth': p.peak_call_depth,
            'sample_count': p.sample_count,
            'functions': {
                name: {
                    'name': fp.name,
                    'call_count': fp.call_count,
                    'total_time': fp.total_time,
                    'self_time': fp.self_time,
                    'total_steps': fp.total_steps,
                    'self_steps': fp.self_steps,
                    'max_stack_depth': fp.max_stack_depth,
                    'callers': fp.callers,
                    'callees': fp.callees,
                }
                for name, fp in p.functions.items()
            },
            'lines': {
                str(line): {
                    'line': lp.line,
                    'execution_count': lp.execution_count,
                    'total_time': lp.total_time,
                    'instructions': lp.instructions,
                }
                for line, lp in p.lines.items()
            },
            'instructions': {
                name: {
                    'opcode': ip.opcode,
                    'count': ip.count,
                    'total_time': ip.total_time,
                }
                for name, ip in p.instructions.items()
            },
            'call_graph': [
                {'caller': e.caller, 'callee': e.callee, 'count': e.count}
                for e in p.call_graph
            ],
        }

    def import_profile(self, data: dict) -> ProfileSnapshot:
        """Import profile from serialized dict."""
        p = ProfileSnapshot()
        p.total_time = data.get('total_time', 0.0)
        p.total_steps = data.get('total_steps', 0)
        p.peak_stack_depth = data.get('peak_stack_depth', 0)
        p.peak_call_depth = data.get('peak_call_depth', 0)
        p.sample_count = data.get('sample_count', 0)

        for name, fd in data.get('functions', {}).items():
            fp = FunctionProfile(
                name=fd['name'],
                call_count=fd['call_count'],
                total_time=fd['total_time'],
                self_time=fd['self_time'],
                total_steps=fd['total_steps'],
                self_steps=fd['self_steps'],
                max_stack_depth=fd['max_stack_depth'],
                callers=fd.get('callers', {}),
                callees=fd.get('callees', {}),
            )
            p.functions[name] = fp

        for line_str, ld in data.get('lines', {}).items():
            lp = LineProfile(
                line=ld['line'],
                execution_count=ld['execution_count'],
                total_time=ld['total_time'],
                instructions=ld['instructions'],
            )
            p.lines[int(line_str)] = lp

        for name, id_data in data.get('instructions', {}).items():
            ip = InstructionProfile(
                opcode=id_data['opcode'],
                count=id_data['count'],
                total_time=id_data['total_time'],
            )
            p.instructions[name] = ip

        for ed in data.get('call_graph', []):
            edge = CallGraphEdge(
                caller=ed['caller'],
                callee=ed['callee'],
                count=ed['count'],
            )
            p.call_graph.append(edge)

        self.profiles.append(p)
        return p


# ============================================================
# Convenience Functions
# ============================================================

def profile_source(source: str, trace=False) -> dict:
    """Profile source code. Returns result dict with profile."""
    profiler = Profiler()
    return profiler.profile(source, trace=trace)


def flat_profile(source: str) -> list:
    """Get flat profile for source code."""
    profiler = Profiler()
    profiler.profile(source)
    return profiler.flat_profile()


def hotspots(source: str, top_n: int = 10) -> list:
    """Get hotspot lines for source code."""
    profiler = Profiler()
    profiler.profile(source)
    return profiler.hotspots(top_n=top_n)


def format_flat_profile(entries: list) -> str:
    """Format flat profile as readable text."""
    if not entries:
        return "No profile data."
    lines = []
    lines.append(f"{'Function':<20s} {'Calls':>6s} {'Self%':>7s} {'Self Time':>12s} {'Total Time':>12s} {'Self Steps':>12s}")
    lines.append("-" * 75)
    for e in entries:
        lines.append(
            f"{e['name']:<20s} {e['call_count']:>6d} {e['pct_time']:>6.1f}% "
            f"{e['self_time']:>11.6f}s {e['total_time']:>11.6f}s {e['self_steps']:>11d}"
        )
    return '\n'.join(lines)


def format_hotspots(entries: list) -> str:
    """Format hotspots as readable text."""
    if not entries:
        return "No hotspot data."
    lines = []
    lines.append(f"{'Line':>6s} {'Exec Count':>12s} {'Time':>12s} {'Instructions':>12s}")
    lines.append("-" * 46)
    for e in entries:
        lines.append(
            f"{e['line']:>6d} {e['execution_count']:>12d} "
            f"{e['total_time']:>11.6f}s {e['instructions']:>12d}"
        )
    return '\n'.join(lines)


def format_call_graph(edges: list) -> str:
    """Format call graph as readable text."""
    if not edges:
        return "No call graph data."
    lines = []
    lines.append(f"{'Caller':<20s} -> {'Callee':<20s} {'Count':>6s}")
    lines.append("-" * 50)
    for e in edges:
        lines.append(f"{e['caller']:<20s} -> {e['callee']:<20s} {e['count']:>6d}")
    return '\n'.join(lines)
