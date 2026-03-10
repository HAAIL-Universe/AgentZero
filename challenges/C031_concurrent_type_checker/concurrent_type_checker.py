"""
Concurrent Type Checker
Challenge C031 -- AgentZero Session 032

Extends C013's type system with concurrent types from C029:
  - TChan(elem): typed channels -- send/recv must match element type
  - TTask(ret): task handles -- spawn returns typed task, join extracts result
  - Type checking for: spawn, chan, send, recv, join, select, yield, task_id
  - Channel type safety: prevents sending wrong types through channels
  - Spawn type inference: spawn f(args) checks f's signature, returns TTask(ret)
  - Join type inference: join(task) returns the task's result type
  - Select case consistency: all recv cases on same channel must agree on type
  - Static deadlock detection: identifies potential deadlocks through resource ordering
    analysis -- channels acquired in inconsistent orders across tasks

Architecture:
  Source -> ConcLex -> ConcParse -> AST -> ConcurrentTypeChecker -> (typed result or errors)

Composes: C013 (type checker) + C029 (concurrent AST nodes + parser)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Import C013 type system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C013_type_checker'))
from type_checker import (
    TInt, TFloat, TString, TBool, TVoid, TFunc, TVar,
    INT, FLOAT, STRING, BOOL, VOID, TYPE_NAMES,
    TypeError_, TypeEnv, resolve, occurs_in, unify, UnificationError,
    types_compatible, TypeChecker,
)

# Import C029 parser and AST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C029_concurrent_runtime'))
from concurrent_runtime import (
    conc_lex, ConcParser, ConcTokenType,
    SpawnExpr, YieldStmt, ChanExpr, SendExpr, RecvExpr,
    JoinExpr, SelectStmt, TaskIdExpr,
)

# Import C010 AST nodes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Program, IntLit, FloatLit, StringLit, BoolLit,
    Var, BinOp, UnaryOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, ReturnStmt, PrintStmt,
    FnDecl, CallExpr,
)


# ============================================================
# Concurrent Type Representations
# ============================================================

@dataclass(frozen=True)
class TChan:
    """Channel type parameterized by element type."""
    elem: Any  # the type of values sent/received

    def __repr__(self):
        return f"Chan<{self.elem!r}>"

    def __eq__(self, other):
        if isinstance(other, TChan):
            e1 = resolve(self.elem) if isinstance(self.elem, TVar) else self.elem
            e2 = resolve(other.elem) if isinstance(other.elem, TVar) else other.elem
            return e1 == e2
        return NotImplemented

    def __hash__(self):
        e = resolve(self.elem) if isinstance(self.elem, TVar) else self.elem
        return hash(('TChan', type(e).__name__))


@dataclass(frozen=True)
class TTask:
    """Task type parameterized by result type."""
    result: Any  # the type returned by the task

    def __repr__(self):
        return f"Task<{self.result!r}>"

    def __eq__(self, other):
        if isinstance(other, TTask):
            r1 = resolve(self.result) if isinstance(self.result, TVar) else self.result
            r2 = resolve(other.result) if isinstance(other.result, TVar) else other.result
            return r1 == r2
        return NotImplemented

    def __hash__(self):
        r = resolve(self.result) if isinstance(self.result, TVar) else self.result
        return hash(('TTask', type(r).__name__))


# ============================================================
# Extended Unification for Concurrent Types
# ============================================================

def conc_unify(t1, t2):
    """Unify two types, supporting concurrent types."""
    t1 = resolve(t1)
    t2 = resolve(t2)

    # Same type
    if t1 == t2:
        return t1

    # Type variables bind to anything
    if isinstance(t1, TVar):
        if occurs_in(t1, t2):
            raise UnificationError(f"Infinite type: {t1} ~ {t2}")
        t1.bound = t2
        return t2

    if isinstance(t2, TVar):
        if occurs_in(t2, t1):
            raise UnificationError(f"Infinite type: {t2} ~ {t1}")
        t2.bound = t1
        return t1

    # Int promotes to float
    if isinstance(t1, TInt) and isinstance(t2, TFloat):
        return FLOAT
    if isinstance(t1, TFloat) and isinstance(t2, TInt):
        return FLOAT

    # Channel types -- unify element types
    if isinstance(t1, TChan) and isinstance(t2, TChan):
        elem = conc_unify(t1.elem, t2.elem)
        return TChan(elem)

    # Task types -- unify result types
    if isinstance(t1, TTask) and isinstance(t2, TTask):
        result = conc_unify(t1.result, t2.result)
        return TTask(result)

    # Function types
    if isinstance(t1, TFunc) and isinstance(t2, TFunc):
        if len(t1.params) != len(t2.params):
            raise UnificationError(
                f"Function arity mismatch: {t1} vs {t2}")
        unified_params = tuple(conc_unify(p1, p2) for p1, p2 in zip(t1.params, t2.params))
        unified_ret = conc_unify(t1.ret, t2.ret)
        return TFunc(unified_params, unified_ret)

    raise UnificationError(f"Cannot unify {t1!r} with {t2!r}")


def conc_occurs_in(tvar, t):
    """Extended occurs check supporting concurrent types."""
    t = resolve(t)
    if isinstance(t, TVar):
        return t.id == tvar.id
    if isinstance(t, TFunc):
        return any(conc_occurs_in(tvar, p) for p in t.params) or conc_occurs_in(tvar, t.ret)
    if isinstance(t, TChan):
        return conc_occurs_in(tvar, t.elem)
    if isinstance(t, TTask):
        return conc_occurs_in(tvar, t.result)
    return False


# ============================================================
# Channel Usage Tracking (for deadlock detection)
# ============================================================

@dataclass
class ChannelUsage:
    """Tracks how a channel is used within a function/task."""
    name: str           # variable name
    line: int           # where declared/first used
    sends: list = field(default_factory=list)   # lines where send occurs
    recvs: list = field(default_factory=list)   # lines where recv occurs


@dataclass
class TaskInfo:
    """Information about a spawned task for deadlock analysis."""
    name: str                   # function name
    line: int                   # spawn line
    channels_acquired: list = field(default_factory=list)  # (channel_name, op, line) in order


@dataclass
class DeadlockWarning:
    """A potential deadlock detected through static analysis."""
    message: str
    line: int
    severity: str = "warning"  # "warning" or "error"

    def __repr__(self):
        return f"DeadlockWarning at line {self.line}: {self.message}"


# ============================================================
# Concurrent Type Checker
# ============================================================

class ConcurrentTypeChecker:
    """
    Type checker extended with concurrent type analysis.

    Checks:
    1. Channel type safety (send/recv match element types)
    2. Spawn type correctness (function exists, args match)
    3. Join type inference (returns task result type)
    4. Select case consistency
    5. Static deadlock detection via resource ordering
    """

    def __init__(self):
        self.errors = []
        self.warnings = []  # DeadlockWarnings
        self.env = TypeEnv()
        self._tvar_counter = 0
        self._return_type_stack = []
        self._channel_usages = {}     # name -> ChannelUsage
        self._task_infos = []         # TaskInfo list
        self._current_task = None     # current function being analyzed
        self._function_bodies = {}    # name -> (params, body, line) for spawn analysis
        self._channel_order = {}      # maps function -> list of (channel, op, line)
        self._in_function = None      # name of current function being checked

    def fresh_tvar(self):
        self._tvar_counter += 1
        return TVar(self._tvar_counter)

    def error(self, msg, line):
        self.errors.append(TypeError_(msg, line))

    def warning(self, msg, line, severity="warning"):
        self.warnings.append(DeadlockWarning(msg, line, severity))

    # ---- Main entry ----

    def check(self, program):
        """Type-check a program. Returns list of TypeError_."""
        self.errors = []
        self.warnings = []

        # First pass: register all function declarations
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                self._register_function(stmt)

        # Second pass: check all statements
        for stmt in program.stmts:
            self.check_stmt(stmt)

        # Third pass: deadlock detection
        self._detect_deadlocks()

        return self.errors

    def _register_function(self, node):
        """Pre-register function type for forward references and spawn analysis."""
        param_types = []
        for _ in node.params:
            param_types.append(self.fresh_tvar())
        ret_type = self.fresh_tvar()
        fn_type = TFunc(tuple(param_types), ret_type)
        self.env.define(node.name, fn_type)
        self._function_bodies[node.name] = (node.params, node.body, node.line)

    # ---- Statement checking ----

    def check_stmt(self, node):
        method = f'check_{type(node).__name__}'
        handler = getattr(self, method, None)
        if handler:
            handler(node)
        else:
            self.infer(node)

    def check_LetDecl(self, node):
        value_type = self.infer(node.value)
        if value_type is not None:
            self.env.define(node.name, value_type)
            # Track channel variables
            resolved = resolve(value_type)
            if isinstance(resolved, TChan):
                self._channel_usages[node.name] = ChannelUsage(
                    name=node.name, line=node.line)
        else:
            self.env.define(node.name, self.fresh_tvar())

    def check_Assign(self, node):
        value_type = self.infer(node.value)
        existing = self.env.lookup(node.name)
        if existing is None:
            self.error(f"Assignment to undefined variable '{node.name}'", node.line)
            self.env.define(node.name, value_type if value_type else self.fresh_tvar())
        else:
            if value_type is not None:
                try:
                    conc_unify(existing, value_type)
                except UnificationError:
                    self.error(
                        f"Cannot assign {value_type!r} to variable '{node.name}' of type {existing!r}",
                        node.line
                    )

    def check_Block(self, node):
        child_env = self.env.child()
        old_env = self.env
        self.env = child_env
        for stmt in node.stmts:
            self.check_stmt(stmt)
        self.env = old_env

    def check_IfStmt(self, node):
        cond_type = self.infer(node.cond)
        if cond_type is not None:
            resolved = resolve(cond_type)
            if not isinstance(resolved, (TBool, TVar)):
                self.error(f"Condition must be bool, got {resolved!r}", node.line)
        self.check_stmt(node.then_body)
        if node.else_body:
            self.check_stmt(node.else_body)

    def check_WhileStmt(self, node):
        cond_type = self.infer(node.cond)
        if cond_type is not None:
            resolved = resolve(cond_type)
            if not isinstance(resolved, (TBool, TVar)):
                self.error(f"While condition must be bool, got {resolved!r}", node.line)
        self.check_stmt(node.body)

    def check_FnDecl(self, node):
        fn_type = self.env.lookup(node.name)
        if fn_type is None:
            # Not pre-registered (shouldn't happen)
            self._register_function(node)
            fn_type = self.env.lookup(node.name)

        fn_type = resolve(fn_type)
        fn_env = self.env.child()

        if isinstance(fn_type, TFunc):
            for param_name, pt in zip(node.params, fn_type.params):
                fn_env.define(param_name, pt)
            ret_type = fn_type.ret
        else:
            for param_name in node.params:
                fn_env.define(param_name, self.fresh_tvar())
            ret_type = self.fresh_tvar()

        old_env = self.env
        old_fn = self._in_function
        self.env = fn_env
        self._in_function = node.name
        self._return_type_stack.append(ret_type)
        self._channel_order[node.name] = []

        self.check_stmt(node.body)

        self._return_type_stack.pop()
        self._in_function = old_fn
        self.env = old_env

    def check_ReturnStmt(self, node):
        if node.value is not None:
            ret_type = self.infer(node.value)
        else:
            ret_type = VOID

        if self._return_type_stack:
            expected = self._return_type_stack[-1]
            if ret_type is not None:
                try:
                    conc_unify(expected, ret_type)
                except UnificationError:
                    self.error(
                        f"Return type mismatch: expected {resolve(expected)!r}, got {ret_type!r}",
                        node.line
                    )
        else:
            self.error("Return statement outside of function", node.line)

    def check_PrintStmt(self, node):
        self.infer(node.value)

    def check_YieldStmt(self, node):
        # yield is always valid in concurrent context, no type constraints
        pass

    def check_SelectStmt(self, node):
        """Type-check a select statement with case consistency."""
        for case in node.cases:
            op, ch_expr, val_expr, var_name, body = case
            ch_type = self.infer(ch_expr)

            if ch_type is not None:
                resolved_ch = resolve(ch_type)
                if not isinstance(resolved_ch, (TChan, TVar)):
                    self.error(
                        f"Select case requires channel, got {resolved_ch!r}",
                        node.line
                    )
                    continue

                if op == 'send' and val_expr is not None:
                    val_type = self.infer(val_expr)
                    if isinstance(resolved_ch, TChan) and val_type is not None:
                        try:
                            conc_unify(resolved_ch.elem, val_type)
                        except UnificationError:
                            self.error(
                                f"Select send: channel expects {resolved_ch.elem!r}, got {val_type!r}",
                                node.line
                            )
                    # Track channel usage
                    self._track_channel_op(ch_expr, 'send', node.line)

                elif op == 'recv':
                    if isinstance(resolved_ch, TChan) and var_name is not None:
                        # Define the received variable in the case body scope
                        child_env = self.env.child()
                        child_env.define(var_name, resolved_ch.elem)
                        old_env = self.env
                        self.env = child_env
                        for s in body.stmts:
                            self.check_stmt(s)
                        self.env = old_env
                        self._track_channel_op(ch_expr, 'recv', node.line)
                        continue  # body already checked
                    self._track_channel_op(ch_expr, 'recv', node.line)

            # Check body
            self.check_stmt(body)

        if node.default_body:
            self.check_stmt(node.default_body)

    # ---- Expression inference ----

    def infer(self, node):
        method = f'infer_{type(node).__name__}'
        handler = getattr(self, method, None)
        if handler:
            return handler(node)

        # Try as statement
        stmt_method = f'check_{type(node).__name__}'
        stmt_handler = getattr(self, stmt_method, None)
        if stmt_handler:
            stmt_handler(node)
            return VOID

        line = getattr(node, 'line', 0)
        self.error(f"Cannot type-check {type(node).__name__}", line)
        return None

    def infer_IntLit(self, node):
        return INT

    def infer_FloatLit(self, node):
        return FLOAT

    def infer_StringLit(self, node):
        return STRING

    def infer_BoolLit(self, node):
        return BOOL

    def infer_Var(self, node):
        t = self.env.lookup(node.name)
        if t is None:
            self.error(f"Undefined variable '{node.name}'", node.line)
            return None
        return resolve(t)

    def infer_UnaryOp(self, node):
        operand_type = self.infer(node.operand)
        if operand_type is None:
            return None
        operand_type = resolve(operand_type)

        if node.op == '-':
            if isinstance(operand_type, TInt):
                return INT
            if isinstance(operand_type, TFloat):
                return FLOAT
            if isinstance(operand_type, TVar):
                return operand_type
            self.error(f"Cannot negate type {operand_type!r}", node.line)
            return None

        if node.op == 'not':
            if isinstance(operand_type, TBool):
                return BOOL
            if isinstance(operand_type, TVar):
                try:
                    conc_unify(operand_type, BOOL)
                except UnificationError:
                    pass
                return BOOL
            self.error(f"'not' requires bool, got {operand_type!r}", node.line)
            return None

        return None

    def infer_BinOp(self, node):
        left_type = self.infer(node.left)
        right_type = self.infer(node.right)

        if left_type is None or right_type is None:
            return None

        left_type = resolve(left_type)
        right_type = resolve(right_type)

        if node.op in ('+', '-', '*', '/', '%'):
            return self._check_arithmetic(node.op, left_type, right_type, node.line)
        if node.op in ('<', '>', '<=', '>='):
            return self._check_comparison(left_type, right_type, node.line)
        if node.op in ('==', '!='):
            return self._check_equality(left_type, right_type, node.line)
        if node.op in ('and', 'or'):
            return self._check_logical(left_type, right_type, node.line)

        self.error(f"Unknown operator '{node.op}'", node.line)
        return None

    def _check_arithmetic(self, op, lt, rt, line):
        if op == '+' and isinstance(lt, TString) and isinstance(rt, TString):
            return STRING
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            if isinstance(lt, TFloat) or isinstance(rt, TFloat):
                return FLOAT
            return INT
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            try:
                result = conc_unify(lt, rt)
                return resolve(result)
            except UnificationError:
                pass
            return self.fresh_tvar()
        self.error(f"Cannot apply '{op}' to {lt!r} and {rt!r}", line)
        return None

    def _check_comparison(self, lt, rt, line):
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            return BOOL
        if isinstance(lt, TString) and isinstance(rt, TString):
            return BOOL
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            return BOOL
        self.error(f"Cannot compare {lt!r} and {rt!r}", line)
        return None

    def _check_equality(self, lt, rt, line):
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            return BOOL
        if type(lt) == type(rt):
            return BOOL
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            return BOOL
        self.error(f"Cannot compare {lt!r} and {rt!r} for equality", line)
        return None

    def _check_logical(self, lt, rt, line):
        if isinstance(lt, TVar):
            try:
                conc_unify(lt, BOOL)
            except UnificationError:
                pass
        elif not isinstance(lt, TBool):
            self.error(f"Logical operator requires bool, got {lt!r}", line)
            return None
        if isinstance(rt, TVar):
            try:
                conc_unify(rt, BOOL)
            except UnificationError:
                pass
        elif not isinstance(rt, TBool):
            self.error(f"Logical operator requires bool, got {rt!r}", line)
            return None
        return BOOL

    def infer_CallExpr(self, node):
        fn_type = self.env.lookup(node.callee)
        if fn_type is None:
            self.error(f"Undefined function '{node.callee}'", node.line)
            return None

        fn_type = resolve(fn_type)

        if isinstance(fn_type, TVar):
            arg_types = []
            for arg in node.args:
                at = self.infer(arg)
                arg_types.append(at if at else self.fresh_tvar())
            ret = self.fresh_tvar()
            fn_t = TFunc(tuple(arg_types), ret)
            try:
                conc_unify(fn_type, fn_t)
            except UnificationError as e:
                self.error(str(e), node.line)
            return ret

        if not isinstance(fn_type, TFunc):
            self.error(f"'{node.callee}' is not a function (type: {fn_type!r})", node.line)
            return None

        if len(node.args) != len(fn_type.params):
            self.error(
                f"Function '{node.callee}' expects {len(fn_type.params)} args, got {len(node.args)}",
                node.line
            )
            return resolve(fn_type.ret)

        for i, (arg, param_type) in enumerate(zip(node.args, fn_type.params)):
            arg_type = self.infer(arg)
            if arg_type is not None:
                try:
                    conc_unify(arg_type, param_type)
                except UnificationError:
                    self.error(
                        f"Argument {i+1} of '{node.callee}': expected {resolve(param_type)!r}, got {arg_type!r}",
                        node.line
                    )

        return resolve(fn_type.ret)

    def infer_Assign(self, node):
        self.check_Assign(node)
        t = self.env.lookup(node.name)
        return resolve(t) if t else None

    # ---- Concurrent expression inference ----

    def infer_SpawnExpr(self, node):
        """spawn func(args) -> TTask(return_type)"""
        fn_type = self.env.lookup(node.callee)
        if fn_type is None:
            self.error(f"Undefined function '{node.callee}' in spawn", node.line)
            return TTask(self.fresh_tvar())

        fn_type = resolve(fn_type)

        if isinstance(fn_type, TVar):
            # Unknown function type -- create one
            arg_types = []
            for arg in node.args:
                at = self.infer(arg)
                arg_types.append(at if at else self.fresh_tvar())
            ret = self.fresh_tvar()
            fn_t = TFunc(tuple(arg_types), ret)
            try:
                conc_unify(fn_type, fn_t)
            except UnificationError as e:
                self.error(str(e), node.line)
            self._task_infos.append(TaskInfo(name=node.callee, line=node.line))
            return TTask(ret)

        if not isinstance(fn_type, TFunc):
            self.error(f"Cannot spawn non-function '{node.callee}' (type: {fn_type!r})", node.line)
            return TTask(self.fresh_tvar())

        # Check argument count
        if len(node.args) != len(fn_type.params):
            self.error(
                f"spawn '{node.callee}' expects {len(fn_type.params)} args, got {len(node.args)}",
                node.line
            )
        else:
            # Check argument types
            for i, (arg, param_type) in enumerate(zip(node.args, fn_type.params)):
                arg_type = self.infer(arg)
                if arg_type is not None:
                    try:
                        conc_unify(arg_type, param_type)
                    except UnificationError:
                        self.error(
                            f"spawn '{node.callee}' arg {i+1}: expected {resolve(param_type)!r}, got {arg_type!r}",
                            node.line
                        )

        self._task_infos.append(TaskInfo(name=node.callee, line=node.line))
        return TTask(resolve(fn_type.ret))

    def infer_ChanExpr(self, node):
        """chan(size) -> TChan(?T) -- element type inferred from usage"""
        size_type = self.infer(node.size)
        if size_type is not None:
            resolved = resolve(size_type)
            if not isinstance(resolved, (TInt, TVar)):
                self.error(f"Channel buffer size must be int, got {resolved!r}", node.line)
        elem_type = self.fresh_tvar()
        return TChan(elem_type)

    def infer_SendExpr(self, node):
        """send(channel, value) -> void"""
        ch_type = self.infer(node.channel)
        val_type = self.infer(node.value)

        if ch_type is not None:
            resolved_ch = resolve(ch_type)
            if isinstance(resolved_ch, TChan):
                if val_type is not None:
                    try:
                        conc_unify(resolved_ch.elem, val_type)
                    except UnificationError:
                        self.error(
                            f"Cannot send {val_type!r} to channel of type {resolved_ch!r}",
                            node.line
                        )
                self._track_channel_op(node.channel, 'send', node.line)
            elif isinstance(resolved_ch, TVar):
                # Infer channel type from sent value
                if val_type is not None:
                    chan_t = TChan(val_type)
                    try:
                        conc_unify(resolved_ch, chan_t)
                    except UnificationError:
                        self.error(f"Type mismatch in send", node.line)
                self._track_channel_op(node.channel, 'send', node.line)
            else:
                self.error(f"send requires channel, got {resolved_ch!r}", node.line)

        return VOID

    def infer_RecvExpr(self, node):
        """recv(channel) -> element_type"""
        ch_type = self.infer(node.channel)

        if ch_type is not None:
            resolved_ch = resolve(ch_type)
            if isinstance(resolved_ch, TChan):
                self._track_channel_op(node.channel, 'recv', node.line)
                return resolve(resolved_ch.elem)
            elif isinstance(resolved_ch, TVar):
                elem = self.fresh_tvar()
                chan_t = TChan(elem)
                try:
                    conc_unify(resolved_ch, chan_t)
                except UnificationError:
                    pass
                self._track_channel_op(node.channel, 'recv', node.line)
                return elem
            else:
                self.error(f"recv requires channel, got {resolved_ch!r}", node.line)

        return self.fresh_tvar()

    def infer_JoinExpr(self, node):
        """join(task) -> result_type"""
        task_type = self.infer(node.task)

        if task_type is not None:
            resolved = resolve(task_type)
            if isinstance(resolved, TTask):
                return resolve(resolved.result)
            elif isinstance(resolved, TVar):
                ret = self.fresh_tvar()
                task_t = TTask(ret)
                try:
                    conc_unify(resolved, task_t)
                except UnificationError:
                    pass
                return ret
            else:
                self.error(f"join requires task, got {resolved!r}", node.line)

        return self.fresh_tvar()

    def infer_TaskIdExpr(self, node):
        """task_id -> int"""
        return INT

    # ---- Channel usage tracking ----

    def _track_channel_op(self, ch_expr, op, line):
        """Track channel operations for deadlock detection."""
        if isinstance(ch_expr, Var):
            name = ch_expr.name
            if name in self._channel_usages:
                usage = self._channel_usages[name]
                if op == 'send':
                    usage.sends.append(line)
                else:
                    usage.recvs.append(line)

            # Track ordering within current function
            if self._in_function is not None:
                if self._in_function not in self._channel_order:
                    self._channel_order[self._in_function] = []
                self._channel_order[self._in_function].append((name, op, line))

    # ---- Deadlock Detection ----

    def _detect_deadlocks(self):
        """
        Static deadlock detection through resource ordering analysis.

        Detects:
        1. Single-channel deadlocks: same task sends and recvs on same channel
           without other tasks participating
        2. Circular wait: tasks acquire channels in inconsistent orders
        """
        # Collect all channel names used across all functions
        all_ch_names = set()
        for fn_name, ops in self._channel_order.items():
            for name, op, line in ops:
                all_ch_names.add(name)

        # Check each channel for same-function deadlock
        for ch_name in all_ch_names:
            self._check_same_function_deadlock(ch_name)

        # Check for circular wait patterns across functions
        self._check_circular_wait()

    def _check_same_function_deadlock(self, ch_name):
        """Detect potential deadlock when same function sends and receives."""
        send_fns = set()
        recv_fns = set()

        for fn_name, ops in self._channel_order.items():
            for name, op, line in ops:
                if name == ch_name:
                    if op == 'send':
                        send_fns.add(fn_name)
                    else:
                        recv_fns.add(fn_name)

        # If a function both sends and receives on the same channel,
        # and no other function participates, it's a potential deadlock
        both = send_fns & recv_fns

        for fn in both:
            other_users = (send_fns | recv_fns) - {fn}
            if not other_users:
                for name, op, line in self._channel_order.get(fn, []):
                    if name == ch_name and op == 'send':
                        self.warning(
                            f"Potential deadlock: function '{fn}' both sends and receives on channel '{ch_name}' with no other participants",
                            line
                        )
                        break

    def _check_circular_wait(self):
        """
        Detect circular wait by checking channel acquisition order.

        If function A acquires ch1 then ch2, and function B acquires ch2 then ch1,
        this is a potential deadlock (circular wait / resource ordering violation).
        """
        # Build acquisition orders per function
        orders = {}  # fn -> [(ch_name, line)]
        for fn_name, ops in self._channel_order.items():
            if len(ops) >= 2:
                # Get unique channels in order of first appearance
                seen = set()
                order = []
                for name, op, line in ops:
                    if name not in seen:
                        seen.add(name)
                        order.append((name, line))
                if len(order) >= 2:
                    orders[fn_name] = order

        # Check for inconsistent orderings
        fn_names = list(orders.keys())
        for i in range(len(fn_names)):
            for j in range(i + 1, len(fn_names)):
                fn_a = fn_names[i]
                fn_b = fn_names[j]
                order_a = [ch for ch, _ in orders[fn_a]]
                order_b = [ch for ch, _ in orders[fn_b]]

                # Find common channels
                common = set(order_a) & set(order_b)
                if len(common) >= 2:
                    # Check if ordering is inconsistent
                    common_order_a = [ch for ch in order_a if ch in common]
                    common_order_b = [ch for ch in order_b if ch in common]

                    if common_order_a != common_order_b:
                        line_a = orders[fn_a][0][1]
                        self.warning(
                            f"Potential circular deadlock: '{fn_a}' and '{fn_b}' acquire channels in inconsistent order ({', '.join(common_order_a)} vs {', '.join(common_order_b)})",
                            line_a
                        )


# ============================================================
# Public API
# ============================================================

def parse_concurrent(source):
    """Parse concurrent source code into AST."""
    tokens = conc_lex(source)
    parser = ConcParser(tokens)
    return parser.parse()


def check_source(source):
    """
    Type-check concurrent source code.
    Returns (errors, warnings, checker).
    """
    program = parse_concurrent(source)
    checker = ConcurrentTypeChecker()
    errors = checker.check(program)
    return errors, checker.warnings, checker


def check_program(program):
    """
    Type-check a pre-parsed concurrent Program AST.
    Returns (errors, warnings, checker).
    """
    checker = ConcurrentTypeChecker()
    errors = checker.check(program)
    return errors, checker.warnings, checker


def format_errors(errors, warnings=None):
    """Format errors and warnings as a human-readable string."""
    lines = []
    if errors:
        lines.append(f"Found {len(errors)} type error(s):")
        for e in errors:
            lines.append(f"  Line {e.line}: {e.message}")
    if warnings:
        lines.append(f"Found {len(warnings)} warning(s):")
        for w in warnings:
            lines.append(f"  Line {w.line}: [{w.severity}] {w.message}")
    if not lines:
        return "No type errors or warnings."
    return "\n".join(lines)


def type_of(source, expr_source=None):
    """
    Get the inferred type of the last expression in the source.
    If expr_source is provided, type-check source then infer expr_source.
    """
    if expr_source is not None:
        full = source + "\n" + expr_source + ";"
    else:
        full = source

    program = parse_concurrent(full)
    checker = ConcurrentTypeChecker()

    # Check all but last
    for stmt in program.stmts[:-1]:
        checker.check_stmt(stmt)

    # Infer last
    if program.stmts:
        last = program.stmts[-1]
        return checker.infer(last)

    return VOID
