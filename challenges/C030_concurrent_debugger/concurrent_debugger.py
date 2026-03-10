"""
Concurrent Debugger -- Debug concurrent programs with task-aware stepping
Challenge C030 -- AgentZero Session 031

Composition: C029 (Concurrent Runtime) + C020 (REPL/Debugger)

Features:
  - Task-aware breakpoints (global or per-task)
  - Step within a focused task or across all tasks
  - Scheduler-level stepping (one full scheduling round)
  - Channel inspection (buffer contents, waiting tasks)
  - Task listing with state, IP, stack, call depth
  - Concurrency event breakpoints (spawn, send, recv, yield, join, block)
  - Deadlock detection (all tasks blocked)
  - Execution trace per-task
  - Watch expressions (per-task or global)
  - Step-over and step-out within a task
"""

import sys
import os
from enum import Enum, IntEnum, auto
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
import copy

# Import C029 concurrent runtime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C029_concurrent_runtime'))
from concurrent_runtime import (
    ConcOp, ConcTokenType, CONC_KEYWORDS,
    conc_lex, ConcParser, ConcCompiler,
    Channel, Task, TaskState, ConcurrentVM,
    compile_concurrent, execute_concurrent,
    SpawnExpr, YieldStmt, ChanExpr, SendExpr, RecvExpr,
    JoinExpr, SelectStmt, TaskIdExpr,
)

# Import C010 base
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Op, Chunk, Token, TokenType, FnObject, CallFrame, VM, VMError,
    lex, compile_source, Program,
)


# ============================================================
# Stop Reasons (extended for concurrency)
# ============================================================

class StopReason(Enum):
    STEP = auto()               # single instruction step
    BREAKPOINT = auto()         # hit a breakpoint
    HALT = auto()               # program finished
    ERROR = auto()              # runtime error
    WATCH = auto()              # watch expression triggered
    TASK_SPAWN = auto()         # new task spawned
    TASK_YIELD = auto()         # task yielded
    TASK_BLOCK = auto()         # task blocked on channel/join
    TASK_COMPLETE = auto()      # task completed
    TASK_FAILED = auto()        # task failed
    CHANNEL_SEND = auto()       # value sent to channel
    CHANNEL_RECV = auto()       # value received from channel
    DEADLOCK = auto()           # all tasks blocked -- deadlock
    SCHEDULER_ROUND = auto()    # one full scheduler round completed
    PREEMPT = auto()            # task preempted (time slice exhausted)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Breakpoint:
    id: int
    line: int = -1
    address: int = -1
    condition: str = ""
    enabled: bool = True
    hit_count: int = 0
    task_id: int = -1           # -1 = all tasks


@dataclass
class DebugEvent:
    reason: StopReason
    task_id: int
    ip: int
    line: int
    op: Optional[int] = None
    message: str = ""
    breakpoint_id: int = -1
    data: Any = None            # extra data (channel info, task info, etc.)


@dataclass
class WatchEntry:
    id: int
    expression: str
    last_value: Any = None
    triggered: bool = False
    task_id: int = -1           # -1 = all tasks


@dataclass
class TraceEntry:
    task_id: int
    ip: int
    op: str
    line: int
    stack_top: list = field(default_factory=list)


class ConcEventType(Enum):
    """Concurrency events that can be breakpointed."""
    SPAWN = auto()
    YIELD = auto()
    SEND = auto()
    RECV = auto()
    JOIN = auto()
    BLOCK = auto()
    COMPLETE = auto()
    PREEMPT = auto()


# ============================================================
# Concurrent Debug VM
# ============================================================

class ConcurrentDebugVM:
    """
    Debugger for concurrent programs. Wraps ConcurrentVM execution model
    with step-by-step control, breakpoints, watches, and task awareness.
    """

    def __init__(self, chunk: Chunk, functions=None,
                 max_steps_per_task=1000, max_total_steps=500000):
        self.functions = functions or {}
        self.max_steps_per_task = max_steps_per_task
        self.max_total_steps = max_total_steps
        self.output = []
        self.total_steps = 0

        # Task management (mirrors ConcurrentVM)
        self.next_task_id = 0
        self.tasks = {}
        self.run_queue = deque()

        # Create main task
        main_task = self._create_task(chunk)
        for name, fn_obj in self.functions.items():
            main_task.env[name] = fn_obj
        self.main_task_id = main_task.id

        # Debug state
        self.breakpoints = {}
        self.watches = {}
        self._next_bp_id = 1
        self._next_watch_id = 1
        self._halted = False
        self._error = None
        self.trace_log = []
        self.trace_enabled = False
        self._focused_task = -1         # -1 = no focus (step all)
        self._event_breaks = set()      # ConcEventType values to break on
        self._last_bp_line = -1
        self._last_bp_task = -1

    # -- Task management --

    def _create_task(self, chunk, env=None):
        task_id = self.next_task_id
        self.next_task_id += 1
        task = Task(id=task_id, chunk=chunk, env=env or {})
        self.tasks[task_id] = task
        self.run_queue.append(task_id)
        return task

    # -- Focus --

    def focus_task(self, task_id: int) -> bool:
        """Focus debugging on a specific task. -1 for all tasks."""
        if task_id == -1 or task_id in self.tasks:
            self._focused_task = task_id
            return True
        return False

    def get_focused_task(self) -> int:
        return self._focused_task

    # -- Event breaks --

    def break_on_event(self, event_type: ConcEventType, enabled: bool = True):
        """Enable/disable breaking on concurrency events."""
        if enabled:
            self._event_breaks.add(event_type)
        else:
            self._event_breaks.discard(event_type)

    def get_event_breaks(self) -> set:
        return set(self._event_breaks)

    # -- Breakpoints --

    def add_breakpoint(self, line: int = -1, address: int = -1,
                       condition: str = "", task_id: int = -1) -> int:
        bp_id = self._next_bp_id
        self._next_bp_id += 1
        self.breakpoints[bp_id] = Breakpoint(
            id=bp_id, line=line, address=address,
            condition=condition, task_id=task_id,
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

    def list_breakpoints(self) -> list:
        return list(self.breakpoints.values())

    # -- Watch expressions --

    def add_watch(self, expression: str, task_id: int = -1) -> int:
        w_id = self._next_watch_id
        self._next_watch_id += 1
        self.watches[w_id] = WatchEntry(
            id=w_id, expression=expression, task_id=task_id,
        )
        return w_id

    def remove_watch(self, w_id: int) -> bool:
        if w_id in self.watches:
            del self.watches[w_id]
            return True
        return False

    # -- Inspection --

    def is_halted(self) -> bool:
        return self._halted

    def get_error(self) -> Optional[str]:
        return self._error

    def get_task_list(self) -> list:
        """Get info about all tasks."""
        result = []
        for tid, task in sorted(self.tasks.items()):
            chunk = task.current_chunk or task.chunk
            line = chunk.lines[task.ip] if task.ip < len(chunk.lines) else -1
            result.append({
                'id': tid,
                'state': task.state.name,
                'ip': task.ip,
                'line': line,
                'stack_depth': len(task.stack),
                'call_depth': len(task.call_stack),
                'steps': task.step_count,
                'result': task.result,
                'error': task.error,
            })
        return result

    def get_task_stack(self, task_id: int) -> Optional[list]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        return list(task.stack)

    def get_task_variables(self, task_id: int) -> Optional[dict]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        return dict(task.env)

    def get_task_call_stack(self, task_id: int) -> Optional[list]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        frames = []
        for f in task.call_stack:
            frames.append({
                'ip': f.ip,
                'env_snapshot': dict(f.base_env),
            })
        chunk = task.current_chunk or task.chunk
        frames.append({
            'ip': task.ip,
            'env_snapshot': dict(task.env),
            'current': True,
        })
        return frames

    def get_task_line(self, task_id: int) -> int:
        task = self.tasks.get(task_id)
        if task is None:
            return -1
        chunk = task.current_chunk or task.chunk
        if task.ip < len(chunk.lines):
            return chunk.lines[task.ip]
        return -1

    def get_channel_info(self) -> list:
        """Find all channels reachable from task environments."""
        channels = {}
        for task in self.tasks.values():
            for name, val in task.env.items():
                if isinstance(val, Channel) and val.id not in channels:
                    # Find which tasks reference this channel
                    channels[val.id] = {
                        'id': val.id,
                        'buffer_size': val.buffer_size,
                        'buffer_count': len(val.buffer),
                        'buffer_contents': list(val.buffer),
                        'closed': val.closed,
                        'send_waiters': len(val.send_waiters),
                        'recv_waiters': len(val.recv_waiters),
                    }
        return sorted(channels.values(), key=lambda c: c['id'])

    def get_run_queue(self) -> list:
        """Get the current run queue (task IDs)."""
        return list(self.run_queue)

    def disassemble_task(self, task_id: int, context: int = 3) -> Optional[str]:
        """Disassemble around a task's current IP."""
        task = self.tasks.get(task_id)
        if task is None:
            return None
        chunk = task.current_chunk or task.chunk
        return self._disassemble_chunk(chunk, task.ip, context)

    def _disassemble_chunk(self, chunk, current_ip, context=3):
        lines = []
        i = 0
        entries = []
        while i < len(chunk.code):
            op = chunk.code[i]
            op_name = self._op_name(op)
            if op in (Op.CONST, Op.LOAD, Op.STORE, Op.JUMP, Op.JUMP_IF_FALSE,
                      Op.JUMP_IF_TRUE, Op.CALL) or op in (ConcOp.SPAWN,):
                operand = chunk.code[i + 1]
                if op == Op.CONST:
                    val = chunk.constants[operand]
                    if isinstance(val, FnObject):
                        text = f"{op_name:20s} {operand} (fn:{val.name})"
                    else:
                        text = f"{op_name:20s} {operand} ({val!r})"
                elif op in (Op.LOAD, Op.STORE):
                    nm = chunk.names[operand] if operand < len(chunk.names) else '?'
                    text = f"{op_name:20s} {operand} ({nm})"
                else:
                    text = f"{op_name:20s} {operand}"
                entries.append((i, text))
                i += 2
            else:
                entries.append((i, op_name))
                i += 1

        current_idx = -1
        for idx, (addr, _) in enumerate(entries):
            if addr == current_ip:
                current_idx = idx
                break
            if addr > current_ip:
                current_idx = max(0, idx - 1)
                break
        if current_idx == -1:
            current_idx = len(entries) - 1 if entries else 0

        start = max(0, current_idx - context)
        end = min(len(entries), current_idx + context + 1)

        for idx in range(start, end):
            addr, text = entries[idx]
            marker = ">>>" if addr == current_ip else "   "
            lines.append(f"{marker} {addr:04d}  {text}")

        return '\n'.join(lines)

    def _op_name(self, op):
        if op in Op._value2member_map_:
            return Op(op).name
        if op in ConcOp._value2member_map_:
            return ConcOp(op).name
        return f"??({op})"

    # -- Execution Control --

    def step(self, task_id: int = -1) -> DebugEvent:
        """Execute one instruction in a specific task (or next in queue).
        task_id=-1 uses focused task, or next from run queue."""
        if self._halted:
            return DebugEvent(StopReason.HALT, -1, -1, -1,
                              message="Program has halted")

        target_id = task_id if task_id >= 0 else self._focused_task
        if target_id >= 0:
            task = self.tasks.get(target_id)
            if task is None:
                return DebugEvent(StopReason.ERROR, target_id, -1, -1,
                                  message=f"Unknown task {target_id}")
            if task.state in (TaskState.COMPLETED, TaskState.FAILED):
                return DebugEvent(StopReason.HALT, target_id, task.ip,
                                  self.get_task_line(target_id),
                                  message=f"Task {target_id} already finished")
            if task.state in (TaskState.BLOCKED_SEND, TaskState.BLOCKED_RECV,
                              TaskState.BLOCKED_JOIN):
                return DebugEvent(StopReason.TASK_BLOCK, target_id, task.ip,
                                  self.get_task_line(target_id),
                                  message=f"Task {target_id} is blocked")
        else:
            # Pick next from run queue
            target_id = self._next_runnable()
            if target_id is None:
                return self._check_deadlock_or_halt()

        return self._execute_one_instruction(target_id, check_bp=True)

    def step_line(self, task_id: int = -1) -> DebugEvent:
        """Step until the line changes in the specified task."""
        if self._halted:
            return DebugEvent(StopReason.HALT, -1, -1, -1,
                              message="Program has halted")

        target_id = task_id if task_id >= 0 else self._focused_task
        if target_id < 0:
            target_id = self._next_runnable()
            if target_id is None:
                return self._check_deadlock_or_halt()

        task = self.tasks.get(target_id)
        if task is None or task.state in (TaskState.COMPLETED, TaskState.FAILED):
            return DebugEvent(StopReason.HALT, target_id or -1, -1, -1,
                              message="Task not available")

        start_line = self.get_task_line(target_id)
        event = self._execute_one_instruction(target_id, check_bp=False)
        if event.reason in (StopReason.HALT, StopReason.ERROR,
                            StopReason.TASK_SPAWN, StopReason.TASK_YIELD,
                            StopReason.TASK_BLOCK, StopReason.TASK_COMPLETE,
                            StopReason.PREEMPT):
            return event

        while self.get_task_line(target_id) == start_line:
            task = self.tasks.get(target_id)
            if task is None or task.state != TaskState.RUNNING:
                break
            event = self._execute_one_instruction(target_id, check_bp=True)
            if event.reason in (StopReason.HALT, StopReason.ERROR,
                                StopReason.BREAKPOINT, StopReason.WATCH,
                                StopReason.TASK_SPAWN, StopReason.TASK_YIELD,
                                StopReason.TASK_BLOCK, StopReason.TASK_COMPLETE,
                                StopReason.PREEMPT):
                return event

        return DebugEvent(StopReason.STEP, target_id,
                          task.ip if task else -1,
                          self.get_task_line(target_id))

    def step_over(self, task_id: int = -1) -> DebugEvent:
        """Step over function calls in the specified task."""
        if self._halted:
            return DebugEvent(StopReason.HALT, -1, -1, -1,
                              message="Program has halted")

        target_id = task_id if task_id >= 0 else self._focused_task
        if target_id < 0:
            target_id = self._next_runnable()
            if target_id is None:
                return self._check_deadlock_or_halt()

        task = self.tasks.get(target_id)
        if task is None or task.state in (TaskState.COMPLETED, TaskState.FAILED):
            return DebugEvent(StopReason.HALT, target_id or -1, -1, -1,
                              message="Task not available")

        target_depth = len(task.call_stack)
        start_line = self.get_task_line(target_id)

        while True:
            task = self.tasks.get(target_id)
            if task is None or task.state not in (TaskState.READY, TaskState.RUNNING):
                break

            if len(task.call_stack) > target_depth:
                event = self._execute_one_instruction(target_id, check_bp=False)
                if event.reason in (StopReason.HALT, StopReason.ERROR,
                                    StopReason.TASK_BLOCK, StopReason.PREEMPT):
                    return event
                continue

            cur_line = self.get_task_line(target_id)
            if cur_line != start_line:
                break

            event = self._execute_one_instruction(target_id, check_bp=False)
            if event.reason in (StopReason.HALT, StopReason.ERROR,
                                StopReason.TASK_BLOCK, StopReason.PREEMPT):
                return event

        return DebugEvent(StopReason.STEP, target_id,
                          task.ip if task else -1,
                          self.get_task_line(target_id))

    def step_out(self, task_id: int = -1) -> DebugEvent:
        """Run until current function returns in the specified task."""
        if self._halted:
            return DebugEvent(StopReason.HALT, -1, -1, -1,
                              message="Program has halted")

        target_id = task_id if task_id >= 0 else self._focused_task
        if target_id < 0:
            target_id = self._next_runnable()
            if target_id is None:
                return self._check_deadlock_or_halt()

        task = self.tasks.get(target_id)
        if task is None or task.state in (TaskState.COMPLETED, TaskState.FAILED):
            return DebugEvent(StopReason.HALT, target_id or -1, -1, -1,
                              message="Task not available")

        target_depth = len(task.call_stack) - 1
        if target_depth < 0:
            return self.continue_execution()

        while True:
            task = self.tasks.get(target_id)
            if task is None or task.state not in (TaskState.READY, TaskState.RUNNING):
                break
            event = self._execute_one_instruction(target_id, check_bp=True)
            if event.reason in (StopReason.HALT, StopReason.ERROR,
                                StopReason.BREAKPOINT, StopReason.TASK_BLOCK):
                return event
            if len(task.call_stack) <= target_depth:
                return DebugEvent(StopReason.STEP, target_id,
                                  task.ip, self.get_task_line(target_id),
                                  message="Returned from function")

        return DebugEvent(StopReason.HALT, target_id,
                          task.ip if task else -1,
                          self.get_task_line(target_id) if task else -1)

    def step_scheduler(self) -> DebugEvent:
        """Execute one full scheduling round: run each ready task for one instruction."""
        if self._halted:
            return DebugEvent(StopReason.HALT, -1, -1, -1,
                              message="Program has halted")

        if not self.run_queue:
            return self._check_deadlock_or_halt()

        tasks_run = []
        # Snapshot current queue
        queue_snapshot = list(self.run_queue)

        for tid in queue_snapshot:
            task = self.tasks.get(tid)
            if task is None or task.state in (TaskState.COMPLETED, TaskState.FAILED):
                if tid in self.run_queue:
                    self.run_queue.remove(tid)
                continue
            if task.state in (TaskState.BLOCKED_SEND, TaskState.BLOCKED_RECV,
                              TaskState.BLOCKED_JOIN):
                if tid in self.run_queue:
                    self.run_queue.remove(tid)
                continue

            event = self._execute_one_instruction(tid, check_bp=True)
            tasks_run.append((tid, event))

            if event.reason in (StopReason.BREAKPOINT, StopReason.WATCH,
                                StopReason.ERROR):
                return event

        self._process_blocked_tasks()

        if not tasks_run:
            return self._check_deadlock_or_halt()

        return DebugEvent(
            StopReason.SCHEDULER_ROUND, -1, -1, -1,
            message=f"Scheduler round: {len(tasks_run)} tasks stepped",
            data={'tasks_run': [t[0] for t in tasks_run]},
        )

    def continue_execution(self) -> DebugEvent:
        """Run until breakpoint, watch, concurrency event break, or halt."""
        if self._halted:
            return DebugEvent(StopReason.HALT, -1, -1, -1,
                              message="Program has halted")

        # Skip breakpoint at the line where we last stopped
        skip_line = self._last_bp_line
        skip_task = self._last_bp_task
        self._last_bp_line = -1
        self._last_bp_task = -1
        skipping = (skip_line >= 0)

        while True:
            if self.total_steps > self.max_total_steps:
                self._halted = True
                self._error = f"Execution limit exceeded ({self.max_total_steps} steps)"
                return DebugEvent(StopReason.ERROR, -1, -1, -1,
                                  message=self._error)

            tid = self._next_runnable()
            if tid is None:
                self._process_blocked_tasks()
                tid = self._next_runnable()
                if tid is None:
                    return self._check_deadlock_or_halt()

            task = self.tasks[tid]

            # Handle skip logic
            should_check = True
            if skipping and tid == skip_task:
                cur_line = self.get_task_line(tid)
                if cur_line == skip_line:
                    should_check = False
                else:
                    skipping = False

            event = self._execute_one_instruction(tid, check_bp=should_check)

            if event.reason == StopReason.BREAKPOINT:
                self._last_bp_line = event.line
                self._last_bp_task = event.task_id
                return event
            if event.reason in (StopReason.WATCH, StopReason.ERROR,
                                StopReason.DEADLOCK):
                return event

            # Check concurrency event breaks -- only stop if requested
            _reason_to_event = {
                StopReason.TASK_SPAWN: ConcEventType.SPAWN,
                StopReason.TASK_YIELD: ConcEventType.YIELD,
                StopReason.TASK_BLOCK: ConcEventType.BLOCK,
                StopReason.CHANNEL_SEND: ConcEventType.SEND,
                StopReason.CHANNEL_RECV: ConcEventType.RECV,
                StopReason.PREEMPT: ConcEventType.PREEMPT,
                StopReason.TASK_COMPLETE: ConcEventType.COMPLETE,
                StopReason.TASK_FAILED: ConcEventType.COMPLETE,
            }
            evt_type = _reason_to_event.get(event.reason)
            if evt_type and evt_type in self._event_breaks:
                return event

            # Check if ALL tasks are done
            if event.reason in (StopReason.HALT, StopReason.TASK_COMPLETE,
                                StopReason.TASK_FAILED):
                self._process_blocked_tasks()
                all_done = all(
                    t.state in (TaskState.COMPLETED, TaskState.FAILED)
                    for t in self.tasks.values()
                )
                if all_done:
                    self._halted = True
                    main_task = self.tasks.get(self.main_task_id)
                    return DebugEvent(
                        StopReason.HALT, self.main_task_id,
                        main_task.ip if main_task else -1,
                        self.get_task_line(self.main_task_id) if main_task else -1,
                        message="All tasks completed",
                    )

    def run_task_until_yield(self, task_id: int) -> DebugEvent:
        """Run a specific task until it yields, blocks, or completes."""
        if self._halted:
            return DebugEvent(StopReason.HALT, -1, -1, -1,
                              message="Program has halted")

        task = self.tasks.get(task_id)
        if task is None:
            return DebugEvent(StopReason.ERROR, task_id, -1, -1,
                              message=f"Unknown task {task_id}")
        if task.state in (TaskState.COMPLETED, TaskState.FAILED):
            return DebugEvent(StopReason.HALT, task_id, task.ip,
                              self.get_task_line(task_id),
                              message=f"Task {task_id} already finished")

        while True:
            task = self.tasks.get(task_id)
            if task is None or task.state not in (TaskState.READY, TaskState.RUNNING):
                break

            event = self._execute_one_instruction(task_id, check_bp=True)
            if event.reason != StopReason.STEP:
                return event

        task = self.tasks.get(task_id)
        return DebugEvent(StopReason.HALT, task_id,
                          task.ip if task else -1,
                          self.get_task_line(task_id) if task else -1)

    # -- Internal execution --

    def _next_runnable(self) -> Optional[int]:
        """Get next runnable task from queue."""
        while self.run_queue:
            tid = self.run_queue.popleft()
            task = self.tasks.get(tid)
            if task and task.state in (TaskState.READY, TaskState.RUNNING):
                return tid
        return None

    def _check_deadlock_or_halt(self) -> DebugEvent:
        """Check if we're in deadlock or all tasks finished."""
        # Re-queue any runnable tasks that fell out of the queue
        for task in self.tasks.values():
            if task.state in (TaskState.READY, TaskState.RUNNING):
                if task.id not in self.run_queue:
                    self.run_queue.append(task.id)

        has_blocked = False
        has_runnable = bool(self.run_queue)
        for task in self.tasks.values():
            if task.state in (TaskState.BLOCKED_SEND, TaskState.BLOCKED_RECV,
                              TaskState.BLOCKED_JOIN):
                has_blocked = True

        if has_runnable:
            # Not deadlock -- there are still runnable tasks
            # This shouldn't normally be called when tasks are runnable
            tid = self.run_queue[0]
            task = self.tasks[tid]
            return DebugEvent(StopReason.STEP, tid, task.ip,
                              self.get_task_line(tid))

        if has_blocked:
            self._halted = True
            self._error = "Deadlock: all tasks blocked"
            blocked_ids = [
                t.id for t in self.tasks.values()
                if t.state in (TaskState.BLOCKED_SEND, TaskState.BLOCKED_RECV,
                               TaskState.BLOCKED_JOIN)
            ]
            return DebugEvent(StopReason.DEADLOCK, -1, -1, -1,
                              message="Deadlock detected: all tasks blocked",
                              data={'blocked_tasks': blocked_ids})

        self._halted = True
        main_task = self.tasks.get(self.main_task_id)
        return DebugEvent(StopReason.HALT, self.main_task_id,
                          main_task.ip if main_task else -1,
                          self.get_task_line(self.main_task_id) if main_task else -1,
                          message="All tasks completed")

    def _execute_one_instruction(self, task_id: int,
                                  check_bp: bool = True) -> DebugEvent:
        """Execute exactly one instruction in the given task."""
        self.total_steps += 1
        if self.total_steps > self.max_total_steps:
            self._halted = True
            self._error = f"Execution limit exceeded ({self.max_total_steps} steps)"
            return DebugEvent(StopReason.ERROR, task_id, -1, -1,
                              message=self._error)

        task = self.tasks[task_id]
        task.state = TaskState.RUNNING
        chunk = task.current_chunk or task.chunk

        if task.ip >= len(chunk.code):
            task.result = task.stack[-1] if task.stack else None
            task.state = TaskState.COMPLETED
            self._wake_joiners(task)
            self._process_blocked_tasks()
            evt = self._maybe_event_break(ConcEventType.COMPLETE, task_id,
                                          task.ip, self.get_task_line(task_id),
                                          f"Task {task_id} completed")
            if evt:
                return evt
            return DebugEvent(StopReason.TASK_COMPLETE, task_id,
                              task.ip, self.get_task_line(task_id),
                              message=f"Task {task_id} completed")

        cur_ip = task.ip
        cur_line = self.get_task_line(task_id)

        # Check breakpoints
        if check_bp:
            bp_event = self._check_breakpoints(task_id, cur_ip, cur_line)
            if bp_event:
                # Re-queue task so it's available when we continue
                task.state = TaskState.READY
                self.run_queue.append(task_id)
                return bp_event

        # Trace
        if self.trace_enabled:
            op_val = chunk.code[task.ip]
            self.trace_log.append(TraceEntry(
                task_id=task_id, ip=task.ip,
                op=self._op_name(op_val), line=cur_line,
                stack_top=list(task.stack[-3:]),
            ))

        op = chunk.code[task.ip]
        task.ip += 1
        task.step_count += 1

        try:
            result = self._exec_op(task, chunk, op)
        except VMError as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            self._wake_joiners(task)
            self._process_blocked_tasks()
            self._halted = (task_id == self.main_task_id)
            self._error = str(e) if self._halted else self._error
            evt = self._maybe_event_break(ConcEventType.COMPLETE, task_id,
                                          cur_ip, cur_line, f"Task {task_id} failed: {e}")
            if evt:
                evt.reason = StopReason.TASK_FAILED
                return evt
            return DebugEvent(StopReason.ERROR, task_id, cur_ip, cur_line,
                              op=op, message=str(e))

        # Handle the execution result
        if result is not None:
            # Re-queue task if it's still in RUNNING state (event break fired
            # mid-instruction without a state transition like yield/block/complete)
            if task.state == TaskState.RUNNING:
                task.state = TaskState.READY
                self.run_queue.append(task_id)
            return result

        # Check watches
        watch_event = self._check_watches(task_id)
        if watch_event:
            return watch_event

        # Check if task completed
        if task.state == TaskState.COMPLETED:
            self._wake_joiners(task)
            self._process_blocked_tasks()
            return DebugEvent(StopReason.TASK_COMPLETE, task_id,
                              cur_ip, cur_line,
                              message=f"Task {task_id} completed")

        # Re-queue if still running
        if task.state == TaskState.RUNNING:
            task.state = TaskState.READY
            self.run_queue.append(task_id)

        return DebugEvent(StopReason.STEP, task_id, cur_ip, cur_line, op=op)

    def _exec_op(self, task, chunk, op):
        """Execute a single opcode for a task. Returns DebugEvent or None."""
        if op == Op.HALT:
            task.result = task.stack[-1] if task.stack else None
            task.state = TaskState.COMPLETED
            self._wake_joiners(task)
            self._process_blocked_tasks()
            evt = self._maybe_event_break(ConcEventType.COMPLETE, task.id,
                                          task.ip, self.get_task_line(task.id),
                                          f"Task {task.id} completed (HALT)")
            if evt:
                return evt
            all_done = all(
                t.state in (TaskState.COMPLETED, TaskState.FAILED)
                for t in self.tasks.values()
            )
            if all_done:
                self._halted = True
            return DebugEvent(StopReason.HALT, task.id, task.ip - 1,
                              self.get_task_line(task.id))

        elif op == Op.CONST:
            idx = chunk.code[task.ip]
            task.ip += 1
            task.stack.append(chunk.constants[idx])

        elif op == Op.POP:
            if task.stack:
                task.stack.pop()

        elif op == Op.DUP:
            if not task.stack:
                raise VMError("Stack underflow on DUP")
            task.stack.append(task.stack[-1])

        elif op == Op.ADD:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a + b)
        elif op == Op.SUB:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a - b)
        elif op == Op.MUL:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a * b)
        elif op == Op.DIV:
            b, a = task.stack.pop(), task.stack.pop()
            if b == 0:
                raise VMError("Division by zero")
            if isinstance(a, int) and isinstance(b, int):
                task.stack.append(a // b)
            else:
                task.stack.append(a / b)
        elif op == Op.MOD:
            b, a = task.stack.pop(), task.stack.pop()
            if b == 0:
                raise VMError("Modulo by zero")
            task.stack.append(a % b)
        elif op == Op.NEG:
            task.stack.append(-task.stack.pop())

        elif op == Op.EQ:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a == b)
        elif op == Op.NE:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a != b)
        elif op == Op.LT:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a < b)
        elif op == Op.GT:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a > b)
        elif op == Op.LE:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a <= b)
        elif op == Op.GE:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a >= b)

        elif op == Op.NOT:
            task.stack.append(not task.stack.pop())
        elif op == Op.AND:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a and b)
        elif op == Op.OR:
            b, a = task.stack.pop(), task.stack.pop()
            task.stack.append(a or b)

        elif op == Op.LOAD:
            idx = chunk.code[task.ip]
            task.ip += 1
            name = chunk.names[idx]
            if name not in task.env:
                raise VMError(f"Undefined variable '{name}'")
            task.stack.append(task.env[name])

        elif op == Op.STORE:
            idx = chunk.code[task.ip]
            task.ip += 1
            name = chunk.names[idx]
            value = task.stack.pop()
            task.env[name] = value

        elif op == Op.JUMP:
            target = chunk.code[task.ip]
            task.ip = target

        elif op == Op.JUMP_IF_FALSE:
            target = chunk.code[task.ip]
            task.ip += 1
            if not task.stack[-1]:
                task.ip = target

        elif op == Op.JUMP_IF_TRUE:
            target = chunk.code[task.ip]
            task.ip += 1
            if task.stack[-1]:
                task.ip = target

        elif op == Op.CALL:
            arg_count = chunk.code[task.ip]
            task.ip += 1
            args = []
            for _ in range(arg_count):
                args.insert(0, task.stack.pop())
            fn_obj = task.stack.pop()
            if not isinstance(fn_obj, FnObject):
                raise VMError(f"Cannot call non-function: {fn_obj}")
            if fn_obj.arity != arg_count:
                raise VMError(
                    f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")
            frame = CallFrame(chunk, task.ip, dict(task.env))
            task.call_stack.append(frame)
            new_chunk = fn_obj.chunk
            task.current_chunk = new_chunk
            task.ip = 0
            for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                task.env[param_name] = args[i]

        elif op == Op.RETURN:
            return_val = task.stack.pop()
            if not task.call_stack:
                task.stack.append(return_val)
                task.result = return_val
                task.state = TaskState.COMPLETED
            else:
                frame = task.call_stack.pop()
                task.current_chunk = frame.chunk
                task.ip = frame.ip
                task.env = frame.base_env
                task.stack.append(return_val)

        elif op == Op.PRINT:
            value = task.stack.pop()
            text = str(value) if value is not None else "None"
            if isinstance(value, bool):
                text = "true" if value else "false"
            self.output.append(text)

        # === Concurrency opcodes ===
        elif op == ConcOp.SPAWN:
            arg_count = chunk.code[task.ip]
            task.ip += 1
            args = []
            for _ in range(arg_count):
                args.insert(0, task.stack.pop())
            fn_obj = task.stack.pop()
            if not isinstance(fn_obj, FnObject):
                raise VMError(f"Cannot spawn non-function: {fn_obj}")
            if fn_obj.arity != arg_count:
                raise VMError(
                    f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")
            new_task = self._create_task(fn_obj.chunk)
            new_task.env = dict(task.env)
            for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                new_task.env[param_name] = args[i]
            task.stack.append(new_task.id)
            evt = self._maybe_event_break(ConcEventType.SPAWN, task.id,
                                          task.ip, self.get_task_line(task.id),
                                          f"Task {new_task.id} spawned from task {task.id}",
                                          data={'spawned_task': new_task.id})
            if evt:
                return evt

        elif op == ConcOp.YIELD:
            task.current_chunk = chunk
            task.state = TaskState.READY
            self.run_queue.append(task.id)
            evt = self._maybe_event_break(ConcEventType.YIELD, task.id,
                                          task.ip, self.get_task_line(task.id),
                                          f"Task {task.id} yielded")
            if evt:
                return evt
            return DebugEvent(StopReason.TASK_YIELD, task.id,
                              task.ip, self.get_task_line(task.id),
                              message=f"Task {task.id} yielded")

        elif op == ConcOp.CHAN_NEW:
            size = task.stack.pop()
            if not isinstance(size, int) or size < 1:
                raise VMError(f"Channel buffer size must be positive integer, got {size}")
            ch = Channel(buffer_size=size)
            task.stack.append(ch)

        elif op == ConcOp.CHAN_SEND:
            value = task.stack.pop()
            ch = task.stack.pop()
            if not isinstance(ch, Channel):
                raise VMError(f"Cannot send to non-channel: {ch}")
            if ch.closed:
                raise VMError("Cannot send to closed channel")
            if ch.try_send(value):
                task.stack.append(True)
                evt = self._maybe_event_break(ConcEventType.SEND, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} sent {value!r} to chan:{ch.id}",
                                              data={'channel': ch.id, 'value': value})
                if evt:
                    return evt
            else:
                task.state = TaskState.BLOCKED_SEND
                task.blocked_channel = ch
                task.blocked_value = value
                task.current_chunk = chunk
                evt = self._maybe_event_break(ConcEventType.BLOCK, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} blocked on send to chan:{ch.id}",
                                              data={'channel': ch.id, 'blocked_on': 'send'})
                if evt:
                    return evt
                return DebugEvent(StopReason.TASK_BLOCK, task.id,
                                  task.ip, self.get_task_line(task.id),
                                  message=f"Task {task.id} blocked on send")

        elif op == ConcOp.CHAN_RECV:
            ch = task.stack.pop()
            if not isinstance(ch, Channel):
                raise VMError(f"Cannot recv from non-channel: {ch}")
            success, value = ch.try_recv()
            if success:
                task.stack.append(value)
                evt = self._maybe_event_break(ConcEventType.RECV, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} received {value!r} from chan:{ch.id}",
                                              data={'channel': ch.id, 'value': value})
                if evt:
                    return evt
            else:
                task.state = TaskState.BLOCKED_RECV
                task.blocked_channel = ch
                task.current_chunk = chunk
                evt = self._maybe_event_break(ConcEventType.BLOCK, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} blocked on recv from chan:{ch.id}",
                                              data={'channel': ch.id, 'blocked_on': 'recv'})
                if evt:
                    return evt
                return DebugEvent(StopReason.TASK_BLOCK, task.id,
                                  task.ip, self.get_task_line(task.id),
                                  message=f"Task {task.id} blocked on recv")

        elif op == ConcOp.TASK_JOIN:
            target_id = task.stack.pop()
            if not isinstance(target_id, int):
                raise VMError(f"Cannot join non-task-id: {target_id}")
            target = self.tasks.get(target_id)
            if target is None:
                raise VMError(f"Unknown task ID: {target_id}")
            if target.state in (TaskState.COMPLETED, TaskState.FAILED):
                task.stack.append(target.result)
                evt = self._maybe_event_break(ConcEventType.JOIN, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} joined task {target_id} (already done)",
                                              data={'joined_task': target_id})
                if evt:
                    return evt
            else:
                task.state = TaskState.BLOCKED_JOIN
                task.blocked_on_task = target_id
                task.current_chunk = chunk
                evt = self._maybe_event_break(ConcEventType.BLOCK, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} blocked joining task {target_id}",
                                              data={'blocked_on': 'join', 'joined_task': target_id})
                if evt:
                    return evt
                return DebugEvent(StopReason.TASK_BLOCK, task.id,
                                  task.ip, self.get_task_line(task.id),
                                  message=f"Task {task.id} blocked on join({target_id})")

        elif op == ConcOp.TASK_ID:
            task.stack.append(task.id)

        elif op == ConcOp.CHAN_TRY_SEND:
            value = task.stack.pop()
            ch = task.stack.pop()
            if not isinstance(ch, Channel):
                raise VMError(f"Cannot send to non-channel: {ch}")
            success = ch.try_send(value)
            task.stack.append(success)
            if success:
                evt = self._maybe_event_break(ConcEventType.SEND, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} try-sent {value!r} to chan:{ch.id}",
                                              data={'channel': ch.id, 'value': value})
                if evt:
                    return evt

        elif op == ConcOp.CHAN_TRY_RECV:
            ch = task.stack.pop()
            if not isinstance(ch, Channel):
                raise VMError(f"Cannot recv from non-channel: {ch}")
            success, value = ch.try_recv()
            task.stack.append(value)
            task.stack.append(success)
            if success:
                evt = self._maybe_event_break(ConcEventType.RECV, task.id,
                                              task.ip, self.get_task_line(task.id),
                                              f"Task {task.id} try-received {value!r} from chan:{ch.id}",
                                              data={'channel': ch.id, 'value': value})
                if evt:
                    return evt

        elif op == ConcOp.SELECT:
            pass  # handled at compile level

        else:
            raise VMError(f"Unknown opcode: {op}")

        return None

    def _maybe_event_break(self, event_type, task_id, ip, line, message,
                           data=None) -> Optional[DebugEvent]:
        """Check if we should break on this concurrency event."""
        if event_type in self._event_breaks:
            reason_map = {
                ConcEventType.SPAWN: StopReason.TASK_SPAWN,
                ConcEventType.YIELD: StopReason.TASK_YIELD,
                ConcEventType.SEND: StopReason.CHANNEL_SEND,
                ConcEventType.RECV: StopReason.CHANNEL_RECV,
                ConcEventType.JOIN: StopReason.TASK_BLOCK,
                ConcEventType.BLOCK: StopReason.TASK_BLOCK,
                ConcEventType.COMPLETE: StopReason.TASK_COMPLETE,
                ConcEventType.PREEMPT: StopReason.PREEMPT,
            }
            return DebugEvent(
                reason_map.get(event_type, StopReason.STEP),
                task_id, ip, line,
                message=message, data=data,
            )
        return None

    def _check_breakpoints(self, task_id, ip, line) -> Optional[DebugEvent]:
        for bp in self.breakpoints.values():
            if not bp.enabled:
                continue
            if bp.task_id >= 0 and bp.task_id != task_id:
                continue
            hit = False
            if bp.address >= 0 and bp.address == ip:
                hit = True
            elif bp.line >= 0 and bp.line == line:
                hit = True
            if hit:
                if bp.condition:
                    try:
                        result = self._eval_condition(bp.condition, task_id)
                        if not result:
                            continue
                    except Exception:
                        continue
                bp.hit_count += 1
                self._last_bp_line = line
                self._last_bp_task = task_id
                return DebugEvent(
                    StopReason.BREAKPOINT, task_id, ip, line,
                    breakpoint_id=bp.id,
                    message=f"Breakpoint {bp.id} hit in task {task_id} (count: {bp.hit_count})",
                )
        return None

    def _check_watches(self, task_id) -> Optional[DebugEvent]:
        for w in self.watches.values():
            if w.task_id >= 0 and w.task_id != task_id:
                continue
            try:
                value = self._eval_condition(w.expression, task_id)
            except Exception:
                continue
            if value != w.last_value:
                old_val = w.last_value
                w.last_value = value
                w.triggered = True
                return DebugEvent(
                    StopReason.WATCH, task_id, -1,
                    self.get_task_line(task_id),
                    message=f"Watch {w.id}: '{w.expression}' changed from {old_val!r} to {value!r} in task {task_id}",
                )
            w.triggered = False
        return None

    def _eval_condition(self, expr: str, task_id: int) -> Any:
        """Evaluate an expression in a task's context."""
        expr = expr.strip()
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"No task {task_id}")
        if expr in task.env:
            return task.env[expr]
        try:
            source = f"let __cond_result__ = {expr};"
            chunk, _ = compile_source(source)
            temp_vm = VM(chunk)
            temp_vm.env = dict(task.env)
            temp_vm.run()
            return temp_vm.env.get('__cond_result__')
        except Exception:
            raise ValueError(f"Cannot evaluate: {expr}")

    def _wake_joiners(self, completed_task):
        """Wake tasks joining on the completed task."""
        for tid, t in self.tasks.items():
            if t.state == TaskState.BLOCKED_JOIN and t.blocked_on_task == completed_task.id:
                t.state = TaskState.READY
                t.blocked_on_task = None
                result = completed_task.result
                if completed_task.state == TaskState.FAILED:
                    result = None
                t.stack.append(result)
                if tid not in self.run_queue:
                    self.run_queue.append(tid)

    def _process_blocked_tasks(self):
        """Try to unblock tasks waiting on channels."""
        changed = True
        while changed:
            changed = False
            for tid, task in list(self.tasks.items()):
                if task.state == TaskState.BLOCKED_SEND:
                    ch = task.blocked_channel
                    if ch.can_send():
                        ch.buffer.append(task.blocked_value)
                        task.blocked_channel = None
                        task.blocked_value = None
                        task.state = TaskState.READY
                        task.stack.append(True)
                        if tid not in self.run_queue:
                            self.run_queue.append(tid)
                        changed = True
                elif task.state == TaskState.BLOCKED_RECV:
                    ch = task.blocked_channel
                    success, value = ch.try_recv()
                    if success:
                        task.blocked_channel = None
                        task.state = TaskState.READY
                        task.stack.append(value)
                        if tid not in self.run_queue:
                            self.run_queue.append(tid)
                        changed = True


# ============================================================
# Public API
# ============================================================

def create_debugger(source: str, **kwargs) -> ConcurrentDebugVM:
    """Compile source and create a concurrent debugger."""
    chunk, compiler = compile_concurrent(source)
    return ConcurrentDebugVM(chunk, functions=compiler.functions, **kwargs)


def debug_concurrent(source: str, breakpoints=None, event_breaks=None,
                     max_total_steps=500000) -> dict:
    """Convenience: compile, set breakpoints, run to completion or first break."""
    dbg = create_debugger(source, max_total_steps=max_total_steps)
    if breakpoints:
        for bp in breakpoints:
            dbg.add_breakpoint(**bp)
    if event_breaks:
        for eb in event_breaks:
            dbg.break_on_event(eb)
    event = dbg.continue_execution()
    return {
        'event': event,
        'output': dbg.output,
        'tasks': dbg.get_task_list(),
        'channels': dbg.get_channel_info(),
    }
