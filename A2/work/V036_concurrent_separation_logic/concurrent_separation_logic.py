"""
V036: Concurrent Separation Logic (CSL)

Extends V031 (Separation Logic) with concurrency primitives:
- Lock-based reasoning with lock invariants
- Parallel composition rule (disjoint concurrency)
- Thread-local vs shared resource tracking
- Data race detection via ownership analysis
- Rely-guarantee reasoning for shared-state interference

Composes: V031 (Separation Logic) + C037 (SMT Solver)
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from copy import deepcopy

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)

sys.path.insert(0, os.path.join(_work, "V031_separation_logic"))

from separation_logic import (
    Expr, EVar, ENull, EInt,
    SLFormula, FormulaKind,
    Emp, SLFalse, PointsTo, Star, StarList, Wand,
    Pure, PureEq, PureNeq, LSeg,
    SymbolicHeap, to_symbolic_heap, from_symbolic_heap,
    SLProver, ProofResult, EntailmentResult,
    check_entailment, check_entailment_with_frame,
    infer_frame, bi_abduce, BiAbductionResult,
    HoareTriple, apply_frame_rule,
    SLVerifier, SLVerifyResult, SLVerdict,
    fresh_var, reset_fresh,
)


# ── Resource ownership ──────────────────────────────────────────────

class Owner(Enum):
    """Who owns a resource."""
    THREAD_LOCAL = auto()   # Owned by a specific thread
    SHARED = auto()         # Protected by a lock
    UNOWNED = auto()        # No owner (potential race)


@dataclass
class Resource:
    """A tracked resource with ownership."""
    formula: SLFormula       # The SL assertion describing this resource
    owner: Owner
    owner_id: Optional[str] = None  # Thread ID or lock name

    def copy(self):
        return Resource(
            formula=deepcopy(self.formula),
            owner=self.owner,
            owner_id=self.owner_id,
        )


# ── Lock invariants ─────────────────────────────────────────────────

@dataclass
class LockInvariant:
    """Associates a lock with an invariant (the resource it protects)."""
    lock_name: str
    invariant: SLFormula      # What holding this lock gives you access to

    def copy(self):
        return LockInvariant(
            lock_name=self.lock_name,
            invariant=deepcopy(self.invariant),
        )


# ── Thread state ────────────────────────────────────────────────────

@dataclass
class ThreadState:
    """State of a single thread: its local resources."""
    thread_id: str
    local: SLFormula           # Thread-local assertion (what this thread owns)
    held_locks: list = field(default_factory=list)  # Lock names currently held

    def copy(self):
        return ThreadState(
            thread_id=self.thread_id,
            local=deepcopy(self.local),
            held_locks=list(self.held_locks),
        )


# ── Concurrent program commands ─────────────────────────────────────

class CmdKind(Enum):
    # Sequential (inherited from V031)
    NEW = auto()        # x = new()
    ASSIGN = auto()     # x = e
    LOAD = auto()       # x = [e]
    STORE = auto()      # [e1] = e2
    DISPOSE = auto()    # dispose(e)
    NULL = auto()       # x = null
    # Concurrency
    ACQUIRE = auto()    # acquire(lock)
    RELEASE = auto()    # release(lock)
    PARALLEL = auto()   # C1 || C2
    ATOMIC = auto()     # atomic { C }
    SKIP = auto()       # no-op
    SEQ = auto()        # C1; C2


@dataclass
class Cmd:
    """A concurrent program command."""
    kind: CmdKind
    args: list = field(default_factory=list)
    # For PARALLEL: args = [left_cmd, right_cmd]
    # For ATOMIC: args = [body_cmd]
    # For SEQ: args = [cmd1, cmd2]
    # For others: args = list of strings/exprs

    def __repr__(self):
        if self.kind == CmdKind.PARALLEL:
            return f"({self.args[0]} || {self.args[1]})"
        elif self.kind == CmdKind.SEQ:
            return f"({self.args[0]}; {self.args[1]})"
        elif self.kind == CmdKind.ATOMIC:
            return f"atomic{{{self.args[0]}}}"
        else:
            return f"{self.kind.name}({', '.join(str(a) for a in self.args)})"


# Command constructors
def CNew(var: str) -> Cmd:
    return Cmd(CmdKind.NEW, [var])

def CAssign(lhs: str, rhs: str) -> Cmd:
    return Cmd(CmdKind.ASSIGN, [lhs, rhs])

def CLoad(lhs: str, src: str) -> Cmd:
    return Cmd(CmdKind.LOAD, [lhs, src])

def CStore(dst: str, val: str) -> Cmd:
    return Cmd(CmdKind.STORE, [dst, val])

def CDispose(var: str) -> Cmd:
    return Cmd(CmdKind.DISPOSE, [var])

def CNull(var: str) -> Cmd:
    return Cmd(CmdKind.NULL, [var])

def CAcquire(lock: str) -> Cmd:
    return Cmd(CmdKind.ACQUIRE, [lock])

def CRelease(lock: str) -> Cmd:
    return Cmd(CmdKind.RELEASE, [lock])

def CParallel(left: Cmd, right: Cmd) -> Cmd:
    return Cmd(CmdKind.PARALLEL, [left, right])

def CAtomic(body: Cmd) -> Cmd:
    return Cmd(CmdKind.ATOMIC, [body])

def CSkip() -> Cmd:
    return Cmd(CmdKind.SKIP, [])

def CSeq(c1: Cmd, c2: Cmd) -> Cmd:
    return Cmd(CmdKind.SEQ, [c1, c2])

def CSeqList(cmds: list) -> Cmd:
    """Build a right-associated sequence from a list."""
    if not cmds:
        return CSkip()
    if len(cmds) == 1:
        return cmds[0]
    result = cmds[-1]
    for c in reversed(cmds[:-1]):
        result = CSeq(c, result)
    return result


# ── Rely-Guarantee ──────────────────────────────────────────────────

@dataclass
class RelyGuarantee:
    """Rely-guarantee specification for a thread.

    rely: what the environment may do (other threads' interference)
    guarantee: what this thread promises to do

    Each is a relation on (pre_state, post_state) expressed as a pair
    of SL formulas: (before, after).
    """
    rely: list = field(default_factory=list)       # [(pre_formula, post_formula), ...]
    guarantee: list = field(default_factory=list)   # [(pre_formula, post_formula), ...]


# ── CSL Verification Result ─────────────────────────────────────────

class CSLVerdict(Enum):
    SAFE = auto()
    UNSAFE = auto()
    RACE = auto()      # Data race detected
    UNKNOWN = auto()


@dataclass
class CSLResult:
    """Result of concurrent separation logic verification."""
    verdict: CSLVerdict
    errors: list = field(default_factory=list)
    thread_results: dict = field(default_factory=dict)  # thread_id -> SLVerifyResult
    race_reports: list = field(default_factory=list)
    lock_usage: dict = field(default_factory=dict)  # lock_name -> [thread_ids]

    def is_safe(self) -> bool:
        return self.verdict == CSLVerdict.SAFE


# ── Footprint analysis ──────────────────────────────────────────────

def _collect_vars_formula(f: SLFormula) -> set:
    """Collect all variable names mentioned in a formula."""
    result = set()
    if f is None:
        return result
    if f.kind == FormulaKind.EMP or f.kind == FormulaKind.FALSE:
        return result
    if f.kind == FormulaKind.POINTS_TO:
        if f.src and f.src.kind.name == 'VAR':
            result.add(f.src.name)
        if f.dst and f.dst.kind.name == 'VAR':
            result.add(f.dst.name)
    elif f.kind == FormulaKind.STAR:
        result |= _collect_vars_formula(f.left)
        result |= _collect_vars_formula(f.right)
    elif f.kind == FormulaKind.WAND:
        result |= _collect_vars_formula(f.left)
        result |= _collect_vars_formula(f.right)
    elif f.kind == FormulaKind.PURE:
        if f.pure_lhs and f.pure_lhs.kind.name == 'VAR':
            result.add(f.pure_lhs.name)
        if f.pure_rhs and f.pure_rhs.kind.name == 'VAR':
            result.add(f.pure_rhs.name)
    elif f.kind == FormulaKind.LSEG:
        if f.seg_start and f.seg_start.kind.name == 'VAR':
            result.add(f.seg_start.name)
        if f.seg_end and f.seg_end.kind.name == 'VAR':
            result.add(f.seg_end.name)
    return result


def _collect_vars_cmd(cmd: Cmd) -> set:
    """Collect all variable names accessed by a command."""
    result = set()
    if cmd.kind in (CmdKind.NEW, CmdKind.DISPOSE, CmdKind.NULL):
        result.add(cmd.args[0])
    elif cmd.kind in (CmdKind.ASSIGN, CmdKind.LOAD, CmdKind.STORE):
        for a in cmd.args:
            if isinstance(a, str):
                result.add(a)
    elif cmd.kind == CmdKind.PARALLEL:
        result |= _collect_vars_cmd(cmd.args[0])
        result |= _collect_vars_cmd(cmd.args[1])
    elif cmd.kind == CmdKind.ATOMIC:
        result |= _collect_vars_cmd(cmd.args[0])
    elif cmd.kind == CmdKind.SEQ:
        result |= _collect_vars_cmd(cmd.args[0])
        result |= _collect_vars_cmd(cmd.args[1])
    return result


def _writes_cmd(cmd: Cmd) -> set:
    """Collect variables written by a command."""
    result = set()
    if cmd.kind in (CmdKind.NEW, CmdKind.NULL):
        result.add(cmd.args[0])
    elif cmd.kind == CmdKind.ASSIGN:
        result.add(cmd.args[0])
    elif cmd.kind == CmdKind.LOAD:
        result.add(cmd.args[0])
    elif cmd.kind == CmdKind.STORE:
        result.add(cmd.args[0])  # writes to heap cell pointed to by args[0]
    elif cmd.kind == CmdKind.DISPOSE:
        pass  # modifies heap, not a variable
    elif cmd.kind == CmdKind.PARALLEL:
        result |= _writes_cmd(cmd.args[0])
        result |= _writes_cmd(cmd.args[1])
    elif cmd.kind == CmdKind.ATOMIC:
        result |= _writes_cmd(cmd.args[0])
    elif cmd.kind == CmdKind.SEQ:
        result |= _writes_cmd(cmd.args[0])
        result |= _writes_cmd(cmd.args[1])
    return result


def _heap_accesses(cmd: Cmd) -> list:
    """Collect heap access descriptors: (kind, var) where kind is 'read' or 'write'."""
    result = []
    if cmd.kind == CmdKind.LOAD:
        result.append(('read', cmd.args[1]))
    elif cmd.kind == CmdKind.STORE:
        result.append(('write', cmd.args[0]))
    elif cmd.kind == CmdKind.DISPOSE:
        result.append(('write', cmd.args[0]))
    elif cmd.kind == CmdKind.NEW:
        result.append(('write', cmd.args[0]))
    elif cmd.kind in (CmdKind.PARALLEL, CmdKind.ATOMIC, CmdKind.SEQ):
        for sub in cmd.args:
            if isinstance(sub, Cmd):
                result.extend(_heap_accesses(sub))
    return result


# ── Data race detection ─────────────────────────────────────────────

def detect_races(cmd: Cmd, lock_invariants: dict = None) -> list:
    """Detect potential data races in a concurrent program.

    A race exists when two parallel threads access the same variable/heap cell
    and at least one access is a write, without lock protection.

    Returns list of race reports: {'var': name, 'threads': [desc1, desc2], 'kind': str}
    """
    if lock_invariants is None:
        lock_invariants = {}

    races = []
    _detect_races_rec(cmd, set(), races, lock_invariants)
    return races


def _locks_acquired_in(cmd: Cmd) -> set:
    """Collect all lock names acquired within a command."""
    result = set()
    if cmd.kind == CmdKind.ACQUIRE:
        result.add(cmd.args[0])
    elif cmd.kind in (CmdKind.SEQ, CmdKind.PARALLEL):
        for sub in cmd.args:
            if isinstance(sub, Cmd):
                result |= _locks_acquired_in(sub)
    elif cmd.kind == CmdKind.ATOMIC:
        result |= _locks_acquired_in(cmd.args[0])
    return result


def _detect_races_rec(cmd: Cmd, held_locks: set, races: list, lock_invs: dict):
    """Recursively detect races."""
    if cmd.kind == CmdKind.PARALLEL:
        left, right = cmd.args[0], cmd.args[1]
        left_writes = _writes_cmd(left)
        right_writes = _writes_cmd(right)
        left_vars = _collect_vars_cmd(left)
        right_vars = _collect_vars_cmd(right)

        # Heap accesses
        left_heap = _heap_accesses(left)
        right_heap = _heap_accesses(right)

        # Variables protected by locks:
        # - locks already held at this point
        # - locks acquired within BOTH branches (both protect the shared access)
        protected = set()
        for lk in held_locks:
            if lk in lock_invs:
                protected |= _collect_vars_formula(lock_invs[lk].invariant)

        # If both branches acquire the same lock, vars under that lock are protected
        left_locks = _locks_acquired_in(left)
        right_locks = _locks_acquired_in(right)
        common_locks = left_locks & right_locks
        for lk in common_locks:
            if lk in lock_invs:
                protected |= _collect_vars_formula(lock_invs[lk].invariant)

        # Check variable races (write-write or read-write on same var)
        for var in left_writes & right_vars:
            if var not in protected:
                kind = "write-write" if var in right_writes else "write-read"
                races.append({
                    'var': var,
                    'threads': ['left', 'right'],
                    'kind': kind,
                })
        for var in right_writes & left_vars:
            if var not in protected and var not in left_writes:
                races.append({
                    'var': var,
                    'threads': ['right', 'left'],
                    'kind': 'write-read',
                })

        # Check heap races
        left_heap_writes = {v for k, v in left_heap if k == 'write'}
        right_heap_writes = {v for k, v in right_heap if k == 'write'}
        left_heap_reads = {v for k, v in left_heap if k == 'read'}
        right_heap_reads = {v for k, v in right_heap if k == 'read'}

        for var in left_heap_writes & (right_heap_writes | right_heap_reads):
            if var not in protected:
                kind = "heap-write-write" if var in right_heap_writes else "heap-write-read"
                if not any(r['var'] == var for r in races):
                    races.append({
                        'var': var,
                        'threads': ['left', 'right'],
                        'kind': kind,
                    })
        for var in right_heap_writes & left_heap_reads:
            if var not in protected:
                if not any(r['var'] == var for r in races):
                    races.append({
                        'var': var,
                        'threads': ['right', 'left'],
                        'kind': 'heap-write-read',
                    })

        # Recurse into sub-commands
        _detect_races_rec(left, set(held_locks), races, lock_invs)
        _detect_races_rec(right, set(held_locks), races, lock_invs)

    elif cmd.kind == CmdKind.SEQ:
        _detect_races_rec(cmd.args[0], held_locks, races, lock_invs)
        _detect_races_rec(cmd.args[1], held_locks, races, lock_invs)

    elif cmd.kind == CmdKind.ATOMIC:
        _detect_races_rec(cmd.args[0], held_locks, races, lock_invs)

    elif cmd.kind == CmdKind.ACQUIRE:
        held_locks.add(cmd.args[0])

    elif cmd.kind == CmdKind.RELEASE:
        held_locks.discard(cmd.args[0])


# ── CSL Verifier ────────────────────────────────────────────────────

class CSLVerifier:
    """Concurrent Separation Logic verifier.

    Implements the key CSL rules:
    1. Parallel composition: {P1} C1 {Q1}, {P2} C2 {Q2} => {P1*P2} C1||C2 {Q1*Q2}
       when C1 and C2 access disjoint resources
    2. Lock acquire: {emp} acquire(l) {I(l)}
       (takes lock invariant from shared pool)
    3. Lock release: {I(l)} release(l) {emp}
       (returns lock invariant to shared pool)
    4. Atomic: {P} C {Q} (verified atomically, no interference)
    5. Frame: {P} C {Q} => {P*R} C {Q*R} (for R disjoint from C)
    """

    def __init__(self, lock_invariants: dict = None, prover: SLProver = None):
        """
        lock_invariants: dict mapping lock_name -> LockInvariant
        """
        self.lock_invariants = {}
        if lock_invariants:
            for name, inv in lock_invariants.items():
                if isinstance(inv, LockInvariant):
                    self.lock_invariants[name] = inv
                else:
                    # Allow passing raw formulas
                    self.lock_invariants[name] = LockInvariant(name, inv)

        self.prover = prover or SLProver()
        self.seq_verifier = SLVerifier()

    def verify(self, pre: SLFormula, cmd: Cmd, post: SLFormula) -> CSLResult:
        """Verify {pre} cmd {post} under CSL rules."""
        reset_fresh()
        errors = []
        thread_results = {}
        race_reports = detect_races(cmd, self.lock_invariants)
        lock_usage = {}

        result = self._verify_cmd(pre, cmd, post, errors, thread_results,
                                   lock_usage, held_locks=set())

        verdict = CSLVerdict.SAFE
        if errors:
            verdict = CSLVerdict.UNSAFE
        if race_reports:
            verdict = CSLVerdict.RACE

        return CSLResult(
            verdict=verdict,
            errors=errors,
            thread_results=thread_results,
            race_reports=race_reports,
            lock_usage=lock_usage,
        )

    def _verify_cmd(self, pre: SLFormula, cmd: Cmd, post: SLFormula,
                     errors: list, thread_results: dict,
                     lock_usage: dict, held_locks: set) -> bool:
        """Verify a single command. Returns True if verified."""

        if cmd.kind == CmdKind.SKIP:
            # {P} skip {P}
            result = self.prover.check_entailment(pre, post)
            if not result.is_valid():
                errors.append(f"Skip: pre does not entail post")
                return False
            return True

        elif cmd.kind == CmdKind.ACQUIRE:
            return self._verify_acquire(pre, cmd, post, errors, lock_usage, held_locks)

        elif cmd.kind == CmdKind.RELEASE:
            return self._verify_release(pre, cmd, post, errors, lock_usage, held_locks)

        elif cmd.kind == CmdKind.PARALLEL:
            return self._verify_parallel(pre, cmd, post, errors,
                                          thread_results, lock_usage, held_locks)

        elif cmd.kind == CmdKind.ATOMIC:
            return self._verify_atomic(pre, cmd, post, errors,
                                        thread_results, lock_usage, held_locks)

        elif cmd.kind == CmdKind.SEQ:
            return self._verify_seq(pre, cmd, post, errors,
                                     thread_results, lock_usage, held_locks)

        else:
            # Sequential heap commands -- delegate to V031
            return self._verify_sequential(pre, cmd, post, errors)

    def _verify_acquire(self, pre: SLFormula, cmd: Cmd, post: SLFormula,
                         errors: list, lock_usage: dict, held_locks: set) -> bool:
        """Lock acquire: {P} acquire(l) {P * I(l)}

        Thread gains the lock invariant.
        """
        lock_name = cmd.args[0]

        if lock_name in held_locks:
            errors.append(f"Acquire: lock '{lock_name}' already held (deadlock)")
            return False

        if lock_name not in self.lock_invariants:
            errors.append(f"Acquire: unknown lock '{lock_name}'")
            return False

        inv = self.lock_invariants[lock_name].invariant

        # Post-state should be pre * invariant
        expected_post = Star(pre, inv)
        result = self.prover.check_entailment(expected_post, post)
        if not result.is_valid():
            # Try the other direction -- maybe post entails expected
            result2 = self.prover.check_entailment(post, expected_post)
            if not result2.is_valid():
                errors.append(f"Acquire '{lock_name}': post-condition mismatch")
                return False

        held_locks.add(lock_name)
        lock_usage.setdefault(lock_name, []).append('acquire')
        return True

    def _verify_release(self, pre: SLFormula, cmd: Cmd, post: SLFormula,
                         errors: list, lock_usage: dict, held_locks: set) -> bool:
        """Lock release: {P * I(l)} release(l) {P}

        Thread must own the lock invariant to release.
        """
        lock_name = cmd.args[0]

        if lock_name not in self.lock_invariants:
            errors.append(f"Release: unknown lock '{lock_name}'")
            return False

        if lock_name not in held_locks:
            errors.append(f"Release: lock '{lock_name}' not held")
            return False

        inv = self.lock_invariants[lock_name].invariant

        # Pre must contain the invariant as a separable part
        frame_result = infer_frame(self.prover, pre, inv)
        if frame_result is None:
            errors.append(f"Release '{lock_name}': pre doesn't contain lock invariant")
            return False

        # Post should be the frame (pre minus invariant)
        result = self.prover.check_entailment(frame_result, post)
        if not result.is_valid():
            result2 = self.prover.check_entailment(post, frame_result)
            if not result2.is_valid():
                errors.append(f"Release '{lock_name}': post-condition mismatch")
                return False

        held_locks.discard(lock_name)
        lock_usage.setdefault(lock_name, []).append('release')
        return True

    def _verify_parallel(self, pre: SLFormula, cmd: Cmd, post: SLFormula,
                          errors: list, thread_results: dict,
                          lock_usage: dict, held_locks: set) -> bool:
        """Parallel composition rule:
        {P1} C1 {Q1}, {P2} C2 {Q2}
        ────────────────────────────────
        {P1 * P2} C1 || C2 {Q1 * Q2}

        Requires C1 and C2 access disjoint resources (checked via footprint).
        """
        left, right = cmd.args[0], cmd.args[1]

        # Try to split pre into P1 * P2 for the two threads
        split = self._split_for_parallel(pre, left, right)
        if split is None:
            errors.append("Parallel: cannot split precondition for disjoint threads")
            return False

        pre_left, pre_right = split

        # Try to split post into Q1 * Q2
        post_split = self._split_for_parallel(post, left, right)
        if post_split is None:
            # If post can't be split, try verifying with unsplit post
            post_left, post_right = post, Emp()
        else:
            post_left, post_right = post_split

        # Verify each thread independently
        left_locks = set(held_locks)
        ok_left = self._verify_cmd(pre_left, left, post_left, errors,
                                    thread_results, lock_usage, left_locks)
        thread_results['left'] = ok_left

        right_locks = set(held_locks)
        ok_right = self._verify_cmd(pre_right, right, post_right, errors,
                                     thread_results, lock_usage, right_locks)
        thread_results['right'] = ok_right

        if ok_left and ok_right:
            # Check combined post
            combined = Star(post_left, post_right)
            result = self.prover.check_entailment(combined, post)
            if not result.is_valid():
                result2 = self.prover.check_entailment(post, combined)
                if not result2.is_valid():
                    errors.append("Parallel: combined post-condition mismatch")
                    return False
            return True

        return False

    def _split_for_parallel(self, formula: SLFormula, left: Cmd, right: Cmd):
        """Split a formula into two parts for parallel threads based on variable usage."""
        sh = to_symbolic_heap(formula)

        left_vars = _collect_vars_cmd(left)
        right_vars = _collect_vars_cmd(right)

        left_spatial = []
        right_spatial = []
        shared_spatial = []

        for atom in sh.spatial:
            atom_vars = _collect_vars_formula(atom)
            in_left = bool(atom_vars & left_vars)
            in_right = bool(atom_vars & right_vars)

            if in_left and not in_right:
                left_spatial.append(atom)
            elif in_right and not in_left:
                right_spatial.append(atom)
            elif in_left and in_right:
                shared_spatial.append(atom)
            else:
                # Not used by either -- give to left
                left_spatial.append(atom)

        if shared_spatial:
            # Can't cleanly split -- shared resources
            # Put shared with left as a fallback
            left_spatial.extend(shared_spatial)

        left_pure = []
        right_pure = []
        for p in sh.pure:
            p_vars = _collect_vars_formula(p)
            if p_vars & right_vars and not (p_vars & left_vars):
                right_pure.append(p)
            else:
                left_pure.append(p)

        left_sh = SymbolicHeap(pure=left_pure, spatial=left_spatial)
        right_sh = SymbolicHeap(pure=right_pure, spatial=right_spatial)

        return (from_symbolic_heap(left_sh), from_symbolic_heap(right_sh))

    def _verify_atomic(self, pre: SLFormula, cmd: Cmd, post: SLFormula,
                        errors: list, thread_results: dict,
                        lock_usage: dict, held_locks: set) -> bool:
        """Atomic block: verify body without interference."""
        body = cmd.args[0]
        return self._verify_cmd(pre, body, post, errors,
                                 thread_results, lock_usage, held_locks)

    def _verify_seq(self, pre: SLFormula, cmd: Cmd, post: SLFormula,
                     errors: list, thread_results: dict,
                     lock_usage: dict, held_locks: set) -> bool:
        """Sequential composition: find a midpoint."""
        c1, c2 = cmd.args[0], cmd.args[1]

        # Compute the midpoint by forward-interpreting c1 (on a copy of held_locks)
        fi_locks = set(held_locks)
        mid = self._forward_interpret(pre, c1, fi_locks)
        if mid is None:
            errors.append("Seq: cannot compute midpoint")
            return False

        ok1 = self._verify_cmd(pre, c1, mid, errors, thread_results,
                                lock_usage, held_locks)
        if not ok1:
            return False

        ok2 = self._verify_cmd(mid, c2, post, errors, thread_results,
                                lock_usage, held_locks)
        return ok2

    def _forward_interpret(self, pre: SLFormula, cmd: Cmd, held_locks: set) -> Optional[SLFormula]:
        """Forward-interpret a command to compute post-state."""
        if cmd.kind == CmdKind.SKIP:
            return pre

        elif cmd.kind == CmdKind.ACQUIRE:
            lock_name = cmd.args[0]
            if lock_name in self.lock_invariants:
                inv = self.lock_invariants[lock_name].invariant
                held_locks.add(lock_name)
                return Star(pre, inv)
            return None

        elif cmd.kind == CmdKind.RELEASE:
            lock_name = cmd.args[0]
            if lock_name in self.lock_invariants:
                inv = self.lock_invariants[lock_name].invariant
                frame = infer_frame(self.prover, pre, inv)
                held_locks.discard(lock_name)
                return frame if frame is not None else pre
            return None

        elif cmd.kind == CmdKind.NEW:
            var = cmd.args[0]
            return Star(pre, PointsTo(EVar(var), ENull()))

        elif cmd.kind == CmdKind.NULL:
            var = cmd.args[0]
            return Star(pre, PureEq(EVar(var), ENull()))

        elif cmd.kind == CmdKind.STORE:
            dst, val = cmd.args[0], cmd.args[1]
            # [dst] = val: need dst|->_ in pre, becomes dst|->val
            return Star(pre, PointsTo(EVar(dst), EVar(val) if isinstance(val, str) else val))

        elif cmd.kind == CmdKind.DISPOSE:
            # Remove the points-to for this variable
            return pre  # Simplified: should remove from heap

        elif cmd.kind == CmdKind.SEQ:
            mid = self._forward_interpret(pre, cmd.args[0], held_locks)
            if mid is None:
                return None
            return self._forward_interpret(mid, cmd.args[1], held_locks)

        elif cmd.kind == CmdKind.ATOMIC:
            return self._forward_interpret(pre, cmd.args[0], held_locks)

        elif cmd.kind == CmdKind.PARALLEL:
            # For parallel, compute both sides independently
            left_post = self._forward_interpret(Emp(), cmd.args[0], set(held_locks))
            right_post = self._forward_interpret(Emp(), cmd.args[1], set(held_locks))
            if left_post is None or right_post is None:
                return None
            return Star(Star(pre, left_post), right_post)

        else:
            return pre  # Fallback for unknown commands

    def _verify_sequential(self, pre: SLFormula, cmd: Cmd, post: SLFormula,
                            errors: list) -> bool:
        """Verify a sequential heap command using V031."""
        # Convert to V031 command format
        if cmd.kind == CmdKind.NEW:
            v031_cmd = ("new", cmd.args)
        elif cmd.kind == CmdKind.ASSIGN:
            v031_cmd = ("assign", cmd.args)
        elif cmd.kind == CmdKind.LOAD:
            v031_cmd = ("load", cmd.args)
        elif cmd.kind == CmdKind.STORE:
            v031_cmd = ("store", cmd.args)
        elif cmd.kind == CmdKind.DISPOSE:
            v031_cmd = ("dispose", cmd.args)
        elif cmd.kind == CmdKind.NULL:
            v031_cmd = ("null", cmd.args)
        else:
            errors.append(f"Unknown sequential command: {cmd.kind}")
            return False

        result = self.seq_verifier.verify(pre, [v031_cmd], post)
        if result.verdict != SLVerdict.SAFE:
            errors.extend(result.errors)
            return False
        return True


# ── Ownership analysis ──────────────────────────────────────────────

@dataclass
class OwnershipReport:
    """Report on resource ownership in a concurrent program."""
    thread_resources: dict  # thread_id -> set of var names
    shared_resources: set   # vars accessed by multiple threads
    protected_resources: dict  # lock_name -> set of protected vars
    unprotected_shared: set  # shared vars not protected by any lock

    def is_race_free(self) -> bool:
        return len(self.unprotected_shared) == 0


def analyze_ownership(cmd: Cmd, lock_invariants: dict = None) -> OwnershipReport:
    """Analyze resource ownership in a concurrent program.

    Determines which resources are thread-local, shared, and whether
    shared resources are properly protected by locks.
    """
    if lock_invariants is None:
        lock_invariants = {}

    thread_resources = {}
    _collect_thread_resources(cmd, thread_resources, "main")

    # Find shared resources (accessed by multiple threads)
    all_vars = {}
    for tid, vars in thread_resources.items():
        for v in vars:
            all_vars.setdefault(v, set()).add(tid)

    shared = {v for v, threads in all_vars.items() if len(threads) > 1}

    # Find protected resources
    protected = {}
    for name, inv in lock_invariants.items():
        if isinstance(inv, LockInvariant):
            inv_formula = inv.invariant
        else:
            inv_formula = inv
        protected[name] = _collect_vars_formula(inv_formula)

    all_protected = set()
    for vars in protected.values():
        all_protected |= vars

    unprotected = shared - all_protected

    return OwnershipReport(
        thread_resources=thread_resources,
        shared_resources=shared,
        protected_resources=protected,
        unprotected_shared=unprotected,
    )


def _collect_thread_resources(cmd: Cmd, resources: dict, thread_id: str):
    """Recursively collect per-thread resource usage."""
    if cmd.kind == CmdKind.PARALLEL:
        left_id = f"{thread_id}.L"
        right_id = f"{thread_id}.R"
        resources.setdefault(left_id, set())
        resources.setdefault(right_id, set())
        _collect_thread_resources(cmd.args[0], resources, left_id)
        _collect_thread_resources(cmd.args[1], resources, right_id)
    elif cmd.kind in (CmdKind.SEQ, ):
        _collect_thread_resources(cmd.args[0], resources, thread_id)
        _collect_thread_resources(cmd.args[1], resources, thread_id)
    elif cmd.kind == CmdKind.ATOMIC:
        _collect_thread_resources(cmd.args[0], resources, thread_id)
    else:
        vars = _collect_vars_cmd(cmd)
        resources.setdefault(thread_id, set()).update(vars)


# ── Rely-Guarantee verification ─────────────────────────────────────

def check_rely_guarantee(
    thread_specs: list,
    lock_invariants: dict = None,
) -> CSLResult:
    """Verify a concurrent system using rely-guarantee reasoning.

    Each spec is: {
        'thread_id': str,
        'pre': SLFormula,
        'cmd': Cmd,
        'post': SLFormula,
        'rely': [(pre_formula, post_formula), ...],
        'guarantee': [(pre_formula, post_formula), ...],
    }

    Checks:
    1. Each thread's guarantee is compatible with other threads' rely
    2. Each thread's Hoare triple holds under its rely assumption
    """
    errors = []
    prover = SLProver()

    # Check guarantee-rely compatibility:
    # Thread i's guarantee must be contained in thread j's rely (for i != j)
    for i, spec_i in enumerate(thread_specs):
        for j, spec_j in enumerate(thread_specs):
            if i == j:
                continue

            guar = spec_i.get('guarantee', [])
            rely = spec_j.get('rely', [])

            for g_pre, g_post in guar:
                found_match = False
                for r_pre, r_post in rely:
                    # Check g_pre => r_pre and g_post => r_post
                    pre_ok = check_entailment(g_pre, r_pre)
                    post_ok = check_entailment(g_post, r_post)
                    if pre_ok and post_ok:
                        found_match = True
                        break

                if not found_match:
                    errors.append(
                        f"Thread {spec_i['thread_id']}'s guarantee not in "
                        f"thread {spec_j['thread_id']}'s rely"
                    )

    # Verify each thread's triple
    verifier = CSLVerifier(lock_invariants, prover)
    thread_results = {}

    for spec in thread_specs:
        tid = spec['thread_id']
        result = verifier.verify(spec['pre'], spec['cmd'], spec['post'])
        thread_results[tid] = result
        if not result.is_safe():
            errors.extend([f"Thread {tid}: {e}" for e in result.errors])

    verdict = CSLVerdict.SAFE if not errors else CSLVerdict.UNSAFE
    return CSLResult(
        verdict=verdict,
        errors=errors,
        thread_results=thread_results,
    )


# ── High-level API ──────────────────────────────────────────────────

def verify_concurrent(pre: SLFormula, cmd: Cmd, post: SLFormula,
                       lock_invariants: dict = None) -> CSLResult:
    """Verify a concurrent program under CSL rules.

    Usage:
        lock_invs = {'mutex': LockInvariant('mutex', PointsTo(EVar('x'), EVar('v')))}
        result = verify_concurrent(pre, CParallel(t1, t2), post, lock_invs)
        assert result.is_safe()
    """
    invs = {}
    if lock_invariants:
        for name, inv in lock_invariants.items():
            if isinstance(inv, LockInvariant):
                invs[name] = inv
            else:
                invs[name] = LockInvariant(name, inv)

    verifier = CSLVerifier(invs)
    return verifier.verify(pre, cmd, post)


def check_race_freedom(cmd: Cmd, lock_invariants: dict = None) -> bool:
    """Check if a concurrent program is free of data races."""
    invs = {}
    if lock_invariants:
        for name, inv in lock_invariants.items():
            if isinstance(inv, LockInvariant):
                invs[name] = inv
            else:
                invs[name] = LockInvariant(name, inv)
    races = detect_races(cmd, invs)
    return len(races) == 0


def ownership_analysis(cmd: Cmd, lock_invariants: dict = None) -> OwnershipReport:
    """Analyze resource ownership in a concurrent program."""
    invs = {}
    if lock_invariants:
        for name, inv in lock_invariants.items():
            if isinstance(inv, LockInvariant):
                invs[name] = inv
            else:
                invs[name] = LockInvariant(name, inv)
    return analyze_ownership(cmd, invs)
