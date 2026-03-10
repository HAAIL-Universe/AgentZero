"""V022: Trace Partitioning for Abstract Interpretation.

Instead of merging abstract states at control flow joins (losing precision),
maintain separate abstract states for different execution histories (traces).
This is strictly more precise than standard abstract interpretation.

Composes: V020 (Abstract Domain Functor) + C039 (Abstract Interpreter) + C010 (Parser)

Key concepts:
- Partition token: identifies which execution path led to a state
- Partitioned environment: maps partition tokens -> abstract environments
- Selective merging: merge partitions when budget exceeded
- Partition directives: control where partitioning happens

Example precision gain:
    if (x > 0) { y = 1; } else { y = -1; }
    z = x * y;

    Standard: x in TOP, y in {-1, 1} -> z in TOP
    Trace partitioned:
      Partition [then]: x in POS, y = 1 -> z = x (POS)
      Partition [else]: x in NON_POS, y = -1 -> z = -x (NON_NEG)
    Result: z is NON_NEG in BOTH partitions (more precise!)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Callable, Any
from enum import Enum
from copy import deepcopy

# --- Path setup ---
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(_work, 'V020_abstract_domain_functor'))

from stack_vm import lex, Parser, IntLit, Var, BinOp, UnaryOp, LetDecl, Assign
from stack_vm import IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt, Block
from domain_functor import (
    AbstractDomain, SignDomain, IntervalDomain, ConstDomain, ParityDomain,
    ProductDomain, ReducedProductDomain, FunctorInterpreter, DomainEnv,
    make_sign_interval, make_full_product, create_custom_domain,
    standard_reducers, analyze_with_domain
)


# ============================================================
# Partition Tokens
# ============================================================

class PartitionKind(Enum):
    """What kind of control flow created this partition."""
    BRANCH_THEN = "then"
    BRANCH_ELSE = "else"
    LOOP_ENTRY = "loop_entry"
    LOOP_ITER = "loop_iter"
    LOOP_EXIT = "loop_exit"
    CALL_SITE = "call"
    INITIAL = "init"


@dataclass(frozen=True)
class PartitionToken:
    """Identifies a trace partition.

    A sequence of (kind, label) pairs recording the execution history.
    E.g., [("then", "line5"), ("else", "line12")] = took then at line 5,
    else at line 12.
    """
    path: tuple  # tuple of (PartitionKind, str) pairs

    @staticmethod
    def initial():
        return PartitionToken(path=((PartitionKind.INITIAL, "start"),))

    def extend(self, kind: PartitionKind, label: str) -> 'PartitionToken':
        return PartitionToken(path=self.path + ((kind, label),))

    def depth(self) -> int:
        return len(self.path)

    def __repr__(self):
        parts = []
        for kind, label in self.path:
            if kind == PartitionKind.INITIAL:
                continue
            parts.append(f"{kind.value}:{label}")
        return "[" + "/".join(parts) + "]" if parts else "[init]"


# ============================================================
# Partitioned Environment
# ============================================================

class PartitionedEnv:
    """An abstract environment that maintains separate states per partition.

    Instead of one DomainEnv, maintains a dict of PartitionToken -> DomainEnv.
    Each partition tracks a different execution history.
    """

    def __init__(self, factory: Callable, partitions: Optional[Dict[PartitionToken, DomainEnv]] = None):
        self._factory = factory
        if partitions is not None:
            self._partitions = dict(partitions)
        else:
            token = PartitionToken.initial()
            self._partitions = {token: DomainEnv(factory)}

    @property
    def partitions(self) -> Dict[PartitionToken, DomainEnv]:
        return self._partitions

    @property
    def num_partitions(self) -> int:
        return len(self._partitions)

    def copy(self) -> 'PartitionedEnv':
        new_parts = {t: e.copy() for t, e in self._partitions.items()}
        return PartitionedEnv(self._factory, new_parts)

    def get_merged(self) -> DomainEnv:
        """Merge all partitions into a single environment (for final results)."""
        envs = list(self._partitions.values())
        if not envs:
            return DomainEnv(self._factory)
        result = envs[0].copy()
        for env in envs[1:]:
            result = result.join(env)
        return result

    def get_variable_per_partition(self, name: str) -> Dict[PartitionToken, Any]:
        """Get the abstract value of a variable in each partition."""
        result = {}
        for token, env in self._partitions.items():
            result[token] = env.get(name)
        return result

    def join(self, other: 'PartitionedEnv') -> 'PartitionedEnv':
        """Join two partitioned environments.

        Partitions with matching tokens are joined. Non-matching are kept as-is.
        """
        new_parts = {}
        all_tokens = set(self._partitions.keys()) | set(other._partitions.keys())
        for token in all_tokens:
            if token in self._partitions and token in other._partitions:
                new_parts[token] = self._partitions[token].join(other._partitions[token])
            elif token in self._partitions:
                new_parts[token] = self._partitions[token].copy()
            else:
                new_parts[token] = other._partitions[token].copy()
        return PartitionedEnv(self._factory, new_parts)

    def widen(self, other: 'PartitionedEnv') -> 'PartitionedEnv':
        """Widen partitioned environments (for loop convergence)."""
        new_parts = {}
        all_tokens = set(self._partitions.keys()) | set(other._partitions.keys())
        for token in all_tokens:
            if token in self._partitions and token in other._partitions:
                new_parts[token] = self._partitions[token].widen(other._partitions[token])
            elif token in self._partitions:
                new_parts[token] = self._partitions[token].copy()
            else:
                new_parts[token] = other._partitions[token].copy()
        return PartitionedEnv(self._factory, new_parts)

    def equals(self, other: 'PartitionedEnv') -> bool:
        """Check if two partitioned environments are equal (fixpoint check)."""
        if set(self._partitions.keys()) != set(other._partitions.keys()):
            return False
        for token in self._partitions:
            if not self._partitions[token].equals(other._partitions[token]):
                return False
        return True

    def merge_partitions(self, tokens_to_merge: List[PartitionToken],
                         new_token: PartitionToken) -> 'PartitionedEnv':
        """Merge specific partitions into one (budget control)."""
        envs = [self._partitions[t] for t in tokens_to_merge if t in self._partitions]
        if not envs:
            return self.copy()
        merged = envs[0].copy()
        for env in envs[1:]:
            merged = merged.join(env)
        new_parts = {t: e.copy() for t, e in self._partitions.items()
                     if t not in tokens_to_merge}
        new_parts[new_token] = merged
        return PartitionedEnv(self._factory, new_parts)

    def __repr__(self):
        parts = []
        for token, env in sorted(self._partitions.items(), key=lambda x: str(x[0])):
            vars_str = ", ".join(f"{n}={env.get(n)}" for n in sorted(env.names))
            parts.append(f"  {token}: {{{vars_str}}}")
        return "PartitionedEnv(\n" + "\n".join(parts) + "\n)"


# ============================================================
# Partition Policy
# ============================================================

class PartitionPolicy:
    """Controls where and how to partition."""

    def __init__(self, max_partitions: int = 16,
                 partition_branches: bool = True,
                 partition_loops: bool = False,
                 max_loop_unroll: int = 3,
                 partition_depth: int = 8):
        self.max_partitions = max_partitions
        self.partition_branches = partition_branches
        self.partition_loops = partition_loops
        self.max_loop_unroll = max_loop_unroll
        self.partition_depth = partition_depth

    def should_partition_branch(self, current_partitions: int,
                                 token_depth: int) -> bool:
        """Whether to partition at a branch point."""
        if not self.partition_branches:
            return False
        if current_partitions * 2 > self.max_partitions:
            return False
        if token_depth >= self.partition_depth:
            return False
        return True

    def should_partition_loop(self, current_partitions: int,
                               iteration: int) -> bool:
        """Whether to partition at a loop iteration."""
        if not self.partition_loops:
            return False
        if iteration >= self.max_loop_unroll:
            return False
        if current_partitions * 2 > self.max_partitions:
            return False
        return True


# ============================================================
# Trace Partitioning Interpreter
# ============================================================

class TracePartitionInterpreter:
    """Abstract interpreter with trace partitioning.

    Instead of maintaining a single abstract environment, maintains
    a PartitionedEnv with separate states per execution trace.
    At branch points, creates new partitions instead of joining.
    """

    def __init__(self, domain_factory: Callable,
                 policy: Optional[PartitionPolicy] = None,
                 max_iterations: int = 50):
        self._factory = domain_factory
        self._policy = policy or PartitionPolicy()
        self._max_iterations = max_iterations
        self._functions = {}
        self._warnings = []
        self._partition_stats = {
            'max_partitions_reached': 0,
            'partitions_merged': 0,
            'branches_partitioned': 0,
            'branches_merged': 0,
            'loop_iterations_partitioned': 0,
        }
        # Underlying single-env interpreter for expression evaluation
        self._base = FunctorInterpreter(domain_factory, max_iterations)

    def analyze(self, source: str) -> dict:
        """Analyze source code with trace partitioning."""
        tokens = lex(source)
        program = Parser(tokens).parse()

        # Collect function declarations
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                self._functions[stmt.name] = stmt

        # Initial partitioned environment
        penv = PartitionedEnv(self._factory)

        # Interpret statements
        penv = self._interpret_stmts(program.stmts, penv)

        # Track max partitions
        self._partition_stats['max_partitions_reached'] = max(
            self._partition_stats['max_partitions_reached'],
            penv.num_partitions
        )

        return {
            'partitioned_env': penv,
            'merged_env': penv.get_merged(),
            'num_partitions': penv.num_partitions,
            'warnings': list(self._warnings),
            'functions': list(self._functions.keys()),
            'stats': dict(self._partition_stats),
        }

    def _interpret_stmts(self, stmts: list, penv: PartitionedEnv) -> PartitionedEnv:
        for stmt in stmts:
            penv = self._interpret_stmt(stmt, penv)
        return penv

    def _interpret_stmt(self, stmt, penv: PartitionedEnv) -> PartitionedEnv:
        if isinstance(stmt, FnDecl):
            self._functions[stmt.name] = stmt
            return penv
        elif isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, penv)
        elif isinstance(stmt, WhileStmt):
            return self._interpret_while(stmt, penv)
        elif isinstance(stmt, (LetDecl, Assign)):
            return self._interpret_assign_like(stmt, penv)
        elif isinstance(stmt, PrintStmt):
            return penv
        elif isinstance(stmt, ReturnStmt):
            return penv
        else:
            return penv

    def _interpret_assign_like(self, stmt, penv: PartitionedEnv) -> PartitionedEnv:
        """Handle LetDecl and Assign: evaluate in each partition independently."""
        new_parts = {}
        for token, env in penv.partitions.items():
            new_env = env.copy()
            val = self._eval_expr(stmt.value, new_env)
            name = stmt.name if isinstance(stmt, (LetDecl, Assign)) else stmt.name
            new_env.set(name, val)
            new_parts[token] = new_env
        return PartitionedEnv(self._factory, new_parts)

    def _interpret_if(self, stmt: IfStmt, penv: PartitionedEnv) -> PartitionedEnv:
        """Handle if-statement with optional trace partitioning.

        If policy allows, create separate partitions for then/else branches.
        Otherwise, join as standard interpreter does.
        """
        should_partition = self._policy.should_partition_branch(
            penv.num_partitions,
            max((t.depth() for t in penv.partitions), default=0)
        )

        if should_partition:
            return self._interpret_if_partitioned(stmt, penv)
        else:
            return self._interpret_if_merged(stmt, penv)

    def _interpret_if_partitioned(self, stmt: IfStmt, penv: PartitionedEnv) -> PartitionedEnv:
        """Partition: keep then/else branches as separate partitions."""
        self._partition_stats['branches_partitioned'] += 1
        label = self._stmt_label(stmt)
        new_parts = {}

        for token, env in penv.partitions.items():
            then_env, else_env = self._refine_condition(stmt.cond, env)

            # Check if branches are feasible
            then_feasible = not self._env_is_bot(then_env)
            else_feasible = not self._env_is_bot(else_env)

            if then_feasible:
                then_token = token.extend(PartitionKind.BRANCH_THEN, label)
                stmts = stmt.then_body if isinstance(stmt.then_body, list) else (
                    stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
                )
                then_result = self._interpret_stmts_single(stmts, then_env)
                new_parts[then_token] = then_result

            if else_feasible:
                else_token = token.extend(PartitionKind.BRANCH_ELSE, label)
                if stmt.else_body is not None:
                    else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else (
                        stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                    )
                    else_result = self._interpret_stmts_single(else_stmts, else_env)
                else:
                    else_result = else_env
                new_parts[else_token] = else_result

        result = PartitionedEnv(self._factory, new_parts)

        # Budget enforcement
        if result.num_partitions > self._policy.max_partitions:
            result = self._enforce_budget(result)

        return result

    def _interpret_if_merged(self, stmt: IfStmt, penv: PartitionedEnv) -> PartitionedEnv:
        """Standard join: merge then/else results within each partition."""
        self._partition_stats['branches_merged'] += 1
        new_parts = {}
        for token, env in penv.partitions.items():
            then_env, else_env = self._refine_condition(stmt.cond, env)

            stmts = stmt.then_body if isinstance(stmt.then_body, list) else (
                stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
            )
            then_result = self._interpret_stmts_single(stmts, then_env)

            if stmt.else_body is not None:
                else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else (
                    stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                )
                else_result = self._interpret_stmts_single(else_stmts, else_env)
            else:
                else_result = else_env

            new_parts[token] = then_result.join(else_result)

        return PartitionedEnv(self._factory, new_parts)

    def _interpret_while(self, stmt: WhileStmt, penv: PartitionedEnv) -> PartitionedEnv:
        """Handle while loop with optional iteration partitioning.

        If partition_loops is enabled, keeps separate partitions for different
        iteration counts (up to max_loop_unroll). After that, merges and widens.
        """
        if self._policy.partition_loops:
            return self._interpret_while_partitioned(stmt, penv)
        else:
            return self._interpret_while_standard(stmt, penv)

    def _interpret_while_standard(self, stmt: WhileStmt, penv: PartitionedEnv) -> PartitionedEnv:
        """Standard widening fixpoint, applied per partition."""
        new_parts = {}
        for token, env in penv.partitions.items():
            result = self._while_fixpoint(stmt, env)
            new_parts[token] = result
        return PartitionedEnv(self._factory, new_parts)

    def _interpret_while_partitioned(self, stmt: WhileStmt, penv: PartitionedEnv) -> PartitionedEnv:
        """Unroll first few iterations as separate partitions, then widen."""
        label = self._stmt_label(stmt)
        new_parts = {}

        for token, env in penv.partitions.items():
            # Unroll first max_loop_unroll iterations
            current = env.copy()
            unrolled_exits = []  # environments that exit the loop

            for i in range(self._policy.max_loop_unroll):
                if not self._policy.should_partition_loop(
                    len(new_parts) + len(unrolled_exits) + 1, i
                ):
                    break

                # Refine for condition
                body_env, exit_env = self._refine_condition(stmt.cond, current)

                # Collect exit partition
                if not self._env_is_bot(exit_env):
                    exit_token = token.extend(PartitionKind.LOOP_EXIT, f"{label}_i{i}")
                    unrolled_exits.append((exit_token, exit_env))

                if self._env_is_bot(body_env):
                    break  # Loop body unreachable

                # Execute body
                body_stmts = stmt.body if isinstance(stmt.body, list) else (
                    stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                )
                current = self._interpret_stmts_single(body_stmts, body_env)

                self._partition_stats['loop_iterations_partitioned'] += 1

            # After unrolling, do standard fixpoint for remaining iterations
            remaining = self._while_fixpoint(stmt, current)
            remain_token = token.extend(PartitionKind.LOOP_ITER, f"{label}_rest")
            new_parts[remain_token] = remaining

            # Add exit partitions
            for exit_token, exit_env in unrolled_exits:
                new_parts[exit_token] = exit_env

        result = PartitionedEnv(self._factory, new_parts)
        if result.num_partitions > self._policy.max_partitions:
            result = self._enforce_budget(result)
        return result

    def _while_fixpoint(self, stmt: WhileStmt, env: DomainEnv) -> DomainEnv:
        """Standard widening fixpoint for a single partition."""
        body_stmts = stmt.body if isinstance(stmt.body, list) else (
            stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
        )

        current = env.copy()
        for _ in range(self._max_iterations):
            body_env, _ = self._refine_condition(stmt.cond, current)

            if self._env_is_bot(body_env):
                break

            after_body = self._interpret_stmts_single(body_stmts, body_env)
            next_env = current.widen(after_body)

            if current.equals(next_env):
                break
            current = next_env

        # Exit: condition is false
        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env

    def _interpret_stmts_single(self, stmts: list, env: DomainEnv) -> DomainEnv:
        """Interpret statements in a single (non-partitioned) environment."""
        for stmt in stmts:
            env = self._interpret_stmt_single(stmt, env)
        return env

    def _interpret_stmt_single(self, stmt, env: DomainEnv) -> DomainEnv:
        """Single-env statement interpreter (for within a partition)."""
        if isinstance(stmt, FnDecl):
            self._functions[stmt.name] = stmt
            return env
        elif isinstance(stmt, (LetDecl, Assign)):
            new_env = env.copy()
            val = self._eval_expr(stmt.value, new_env)
            new_env.set(stmt.name, val)
            return new_env
        elif isinstance(stmt, IfStmt):
            return self._if_single(stmt, env)
        elif isinstance(stmt, WhileStmt):
            return self._while_fixpoint(stmt, env)
        elif isinstance(stmt, PrintStmt):
            return env
        elif isinstance(stmt, ReturnStmt):
            return env
        else:
            return env

    def _if_single(self, stmt: IfStmt, env: DomainEnv) -> DomainEnv:
        """Standard if handling in a single environment (join)."""
        then_env, else_env = self._refine_condition(stmt.cond, env)

        stmts = stmt.then_body if isinstance(stmt.then_body, list) else (
            stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
        )
        then_result = self._interpret_stmts_single(stmts, then_env)

        if stmt.else_body is not None:
            else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else (
                stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
            )
            else_result = self._interpret_stmts_single(else_stmts, else_env)
        else:
            else_result = else_env

        return then_result.join(else_result)

    # --- Expression evaluation ---

    def _eval_expr(self, expr, env: DomainEnv):
        """Evaluate expression in a single environment."""
        if isinstance(expr, IntLit):
            return self._factory(expr.value)
        elif isinstance(expr, Var):
            return env.get(expr.name)
        elif isinstance(expr, BinOp):
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            op = expr.op
            if op == '+':
                return left.add(right)
            elif op == '-':
                return left.sub(right)
            elif op == '*':
                return left.mul(right)
            elif op in ('<', '<=', '>', '>=', '==', '!='):
                return self._factory(None)  # TOP for comparisons
            else:
                return self._factory(None)
        elif isinstance(expr, UnaryOp):
            operand = self._eval_expr(expr.operand, env)
            if expr.op == '-':
                return operand.neg()
            return operand
        elif isinstance(expr, CallExpr):
            return self._eval_call(expr, env)
        else:
            return self._factory(None)

    def _eval_call(self, expr: CallExpr, env: DomainEnv):
        """Evaluate function call."""
        fn_name = expr.callee if isinstance(expr.callee, str) else expr.callee.name
        if fn_name not in self._functions:
            return self._factory(None)

        fn = self._functions[fn_name]
        call_env = env.copy()
        for param, arg in zip(fn.params, expr.args):
            param_name = param if isinstance(param, str) else param.name
            val = self._eval_expr(arg, env)
            call_env.set(param_name, val)

        body_stmts = fn.body if isinstance(fn.body, list) else (
            fn.body.stmts if isinstance(fn.body, Block) else [fn.body]
        )
        for s in body_stmts:
            if isinstance(s, ReturnStmt):
                return self._eval_expr(s.value, call_env)
            call_env = self._interpret_stmt_single(s, call_env)

        return self._factory(None)

    # --- Condition refinement ---

    def _refine_condition(self, cond, env: DomainEnv) -> Tuple[DomainEnv, DomainEnv]:
        """Refine environment for condition being true/false.

        Returns (then_env, else_env).
        """
        if isinstance(cond, BinOp) and cond.op in ('<', '<=', '>', '>=', '==', '!='):
            return self._refine_comparison(cond, env)
        elif isinstance(cond, Var):
            # Nonzero = true, zero = false
            return env.copy(), env.copy()
        else:
            return env.copy(), env.copy()

    def _refine_comparison(self, cond: BinOp, env: DomainEnv) -> Tuple[DomainEnv, DomainEnv]:
        """Refine environment based on a comparison."""
        left_val = self._eval_expr(cond.left, env)
        right_val = self._eval_expr(cond.right, env)

        op = cond.op

        # Compute refined values for then branch
        if op == '<':
            then_left, then_right = left_val.refine_lt(right_val)
            else_right, else_left = right_val.refine_le(left_val)  # >= is !(< )
        elif op == '<=':
            then_left, then_right = left_val.refine_le(right_val)
            else_right, else_left = right_val.refine_lt(left_val)  # > is !(<=)
        elif op == '>':
            then_right, then_left = right_val.refine_lt(left_val)
            else_left, else_right = left_val.refine_le(right_val)
        elif op == '>=':
            then_right, then_left = right_val.refine_le(left_val)
            else_left, else_right = left_val.refine_lt(right_val)
        elif op == '==':
            then_left, then_right = left_val.refine_eq(right_val)
            else_left, else_right = left_val.refine_ne(right_val)
        elif op == '!=':
            then_left, then_right = left_val.refine_ne(right_val)
            else_left, else_right = left_val.refine_eq(right_val)
        else:
            return env.copy(), env.copy()

        then_env = env.copy()
        else_env = env.copy()

        # Apply refinements to variables
        if isinstance(cond.left, Var):
            then_env.set(cond.left.name, then_left)
            else_env.set(cond.left.name, else_left)
        if isinstance(cond.right, Var):
            then_env.set(cond.right.name, then_right)
            else_env.set(cond.right.name, else_right)

        return then_env, else_env

    # --- Utility ---

    def _env_is_bot(self, env: DomainEnv) -> bool:
        """Check if environment is bottom (unreachable)."""
        for name in env.names:
            val = env.get(name)
            if val.is_bot():
                return True
        return False

    def _stmt_label(self, stmt) -> str:
        """Create a label for a statement (for partition tokens)."""
        if isinstance(stmt, IfStmt):
            return f"if_{id(stmt) % 10000}"
        elif isinstance(stmt, WhileStmt):
            return f"while_{id(stmt) % 10000}"
        else:
            return f"stmt_{id(stmt) % 10000}"

    def _enforce_budget(self, penv: PartitionedEnv) -> PartitionedEnv:
        """Merge oldest/shallowest partitions to stay within budget."""
        self._partition_stats['partitions_merged'] += 1
        while penv.num_partitions > self._policy.max_partitions:
            tokens = list(penv.partitions.keys())
            # Merge the two partitions with deepest paths (least general)
            tokens.sort(key=lambda t: t.depth(), reverse=True)
            to_merge = tokens[:2]
            # Create a merged token from the common prefix
            merged_token = self._common_prefix_token(to_merge[0], to_merge[1])
            penv = penv.merge_partitions(to_merge, merged_token)
        return penv

    def _common_prefix_token(self, t1: PartitionToken, t2: PartitionToken) -> PartitionToken:
        """Find common prefix of two tokens."""
        prefix = []
        for a, b in zip(t1.path, t2.path):
            if a == b:
                prefix.append(a)
            else:
                break
        if not prefix:
            prefix = [(PartitionKind.INITIAL, "merged")]
        return PartitionToken(path=tuple(prefix))


# ============================================================
# High-level APIs
# ============================================================

def trace_partition_analyze(source: str,
                            domain_factory: Optional[Callable] = None,
                            policy: Optional[PartitionPolicy] = None,
                            max_iterations: int = 50) -> dict:
    """Analyze source code with trace partitioning.

    Args:
        source: C10 source code
        domain_factory: Domain factory (default: sign x interval)
        policy: Partition policy (default: branch partitioning, budget 16)
        max_iterations: Max fixpoint iterations

    Returns:
        dict with partitioned_env, merged_env, num_partitions, warnings, stats
    """
    if domain_factory is None:
        domain_factory = make_sign_interval()
    interp = TracePartitionInterpreter(domain_factory, policy, max_iterations)
    return interp.analyze(source)


def trace_partition_full(source: str,
                         policy: Optional[PartitionPolicy] = None) -> dict:
    """Analyze with full product domain (sign x interval x const x parity)."""
    return trace_partition_analyze(source, make_full_product(), policy)


def get_variable_partitions(source: str, var_name: str,
                             domain_factory: Optional[Callable] = None,
                             policy: Optional[PartitionPolicy] = None) -> dict:
    """Get the abstract value of a variable in each partition.

    Returns:
        dict mapping PartitionToken -> abstract value
    """
    result = trace_partition_analyze(source, domain_factory, policy)
    penv = result['partitioned_env']
    return penv.get_variable_per_partition(var_name)


def compare_precision(source: str,
                      domain_factory: Optional[Callable] = None,
                      policy: Optional[PartitionPolicy] = None) -> dict:
    """Compare standard analysis vs trace partitioning.

    Returns:
        dict with 'standard', 'partitioned', 'precision_gains' keys
    """
    if domain_factory is None:
        domain_factory = make_sign_interval()

    # Standard analysis
    std_result = analyze_with_domain(source, domain_factory)
    std_env = std_result['env']

    # Trace partitioning
    tp_result = trace_partition_analyze(source, domain_factory, policy)
    tp_merged = tp_result['merged_env']
    tp_penv = tp_result['partitioned_env']

    # Find precision gains
    gains = []
    all_vars = std_env.names | tp_merged.names
    for var in sorted(all_vars):
        std_val = std_env.get(var)
        tp_val = tp_merged.get(var)
        # Check if tp is more precise (tp <= std but not std <= tp)
        if tp_val.leq(std_val) and not std_val.leq(tp_val):
            gains.append({
                'variable': var,
                'standard': str(std_val),
                'partitioned': str(tp_val),
                'per_partition': {
                    str(token): str(val)
                    for token, val in tp_penv.get_variable_per_partition(var).items()
                }
            })

    return {
        'standard': {var: str(std_env.get(var)) for var in sorted(all_vars)},
        'partitioned_merged': {var: str(tp_merged.get(var)) for var in sorted(all_vars)},
        'num_partitions': tp_result['num_partitions'],
        'precision_gains': gains,
        'stats': tp_result['stats'],
    }


def analyze_with_loop_partitioning(source: str,
                                    domain_factory: Optional[Callable] = None,
                                    max_unroll: int = 3) -> dict:
    """Analyze with loop iteration partitioning enabled.

    Keeps separate partitions for the first max_unroll loop iterations.
    """
    policy = PartitionPolicy(
        partition_branches=True,
        partition_loops=True,
        max_loop_unroll=max_unroll,
        max_partitions=32,
    )
    return trace_partition_analyze(source, domain_factory, policy)


def analyze_branches_only(source: str,
                           domain_factory: Optional[Callable] = None,
                           max_partitions: int = 16) -> dict:
    """Analyze with branch partitioning only (no loop partitioning)."""
    policy = PartitionPolicy(
        partition_branches=True,
        partition_loops=False,
        max_partitions=max_partitions,
    )
    return trace_partition_analyze(source, domain_factory, policy)


def analyze_no_partition(source: str,
                          domain_factory: Optional[Callable] = None) -> dict:
    """Analyze with no partitioning (equivalent to standard, for comparison)."""
    policy = PartitionPolicy(
        partition_branches=False,
        partition_loops=False,
    )
    return trace_partition_analyze(source, domain_factory, policy)
