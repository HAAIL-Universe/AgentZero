"""V104: Relational Abstract Domains (Octagon + Zone)

Implements relational abstract domains that track constraints between pairs
of variables, capturing relationships that interval analysis loses.

Two domains:
1. Zone (DBM): constraints of the form x - y <= c
2. Octagon: constraints of the form +/-x +/-y <= c (via variable doubling)

Composes with:
- V020 (AbstractDomain protocol) for domain functor integration
- C039 (abstract interpreter) for comparison
- C010 (parser) for C10 source analysis
"""

import sys
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Any
from enum import Enum

sys.path.insert(0, 'Z:/AgentZero/challenges/C010_stack_vm')
sys.path.insert(0, 'Z:/AgentZero/challenges/C039_abstract_interpreter')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V020_abstract_domain_functor')

from stack_vm import lex, Parser
from abstract_interpreter import (
    analyze as c039_analyze, Interval, Sign,
    AbstractInterpreter, AbstractEnv
)
from domain_functor import AbstractDomain, FunctorInterpreter

INF = float('inf')


# ---------------------------------------------------------------------------
# Zone Domain (Difference Bound Matrix)
# ---------------------------------------------------------------------------

class ZoneDomain:
    """Zone abstract domain: tracks x - y <= c constraints via DBM.

    Variables are indexed 0..n-1 plus a special variable 0 representing
    the constant zero. DBM[i][j] = c means var_i - var_j <= c.

    DBM[0][j] = c means -var_j <= c (i.e., var_j >= -c)
    DBM[i][0] = c means var_i <= c (upper bound)
    """

    def __init__(self, var_names: List[str]):
        self.var_names = list(var_names)
        self._var_index = {name: i + 1 for i, name in enumerate(var_names)}
        n = len(var_names) + 1  # +1 for the zero variable
        self.n = n
        # Initialize to TOP (all INF)
        self.dbm = [[INF] * n for _ in range(n)]
        # x - x <= 0 (diagonal)
        for i in range(n):
            self.dbm[i][i] = 0

    def copy(self) -> 'ZoneDomain':
        z = ZoneDomain(self.var_names)
        z.dbm = [row[:] for row in self.dbm]
        return z

    def _idx(self, name: str) -> int:
        return self._var_index[name]

    def is_bot(self) -> bool:
        """Check for negative cycle (inconsistency)."""
        for i in range(self.n):
            if self.dbm[i][i] < 0:
                return True
        return False

    @staticmethod
    def bot(var_names: List[str]) -> 'ZoneDomain':
        z = ZoneDomain(var_names)
        z.dbm[0][0] = -1  # negative cycle -> bot
        return z

    def set_upper(self, var: str, c: float):
        """var <= c"""
        idx = self._idx(var)
        self.dbm[idx][0] = min(self.dbm[idx][0], c)

    def set_lower(self, var: str, c: float):
        """var >= c (i.e., -var <= -c, which is 0 - var <= -c)"""
        idx = self._idx(var)
        self.dbm[0][idx] = min(self.dbm[0][idx], -c)

    def set_diff(self, x: str, y: str, c: float):
        """x - y <= c"""
        xi, yi = self._idx(x), self._idx(y)
        self.dbm[xi][yi] = min(self.dbm[xi][yi], c)

    def get_upper(self, var: str) -> float:
        """Returns c such that var <= c."""
        return self.dbm[self._idx(var)][0]

    def get_lower(self, var: str) -> float:
        """Returns c such that var >= c."""
        return -self.dbm[0][self._idx(var)]

    def get_interval(self, var: str) -> Tuple[float, float]:
        lo = self.get_lower(var)
        hi = self.get_upper(var)
        return (lo, hi)

    def get_diff_bound(self, x: str, y: str) -> float:
        """Returns c such that x - y <= c."""
        return self.dbm[self._idx(x)][self._idx(y)]

    def close(self):
        """Floyd-Warshall shortest path closure for tightening."""
        n = self.n
        for k in range(n):
            for i in range(n):
                if self.dbm[i][k] == INF:
                    continue
                for j in range(n):
                    if self.dbm[k][j] == INF:
                        continue
                    d = self.dbm[i][k] + self.dbm[k][j]
                    if d < self.dbm[i][j]:
                        self.dbm[i][j] = d

    def assign_const(self, var: str, c: float):
        """var := c (strong update)."""
        idx = self._idx(var)
        n = self.n
        # Forget var
        for j in range(n):
            if j != idx:
                self.dbm[idx][j] = INF
                self.dbm[j][idx] = INF
        # Set var = c => var <= c and var >= c
        self.dbm[idx][0] = c    # var - 0 <= c => var <= c
        self.dbm[0][idx] = -c   # 0 - var <= -c => var >= c
        # Re-derive difference constraints via closure rows
        for j in range(n):
            if j != idx and j != 0:
                # var - other <= c - lower(other)
                if self.dbm[0][j] != INF:
                    self.dbm[idx][j] = min(self.dbm[idx][j],
                                           c + self.dbm[0][j])
                # other - var <= upper(other) - c
                if self.dbm[j][0] != INF:
                    self.dbm[j][idx] = min(self.dbm[j][idx],
                                           self.dbm[j][0] - c)

    def assign_var(self, target: str, source: str):
        """target := source (copy)."""
        ti, si = self._idx(target), self._idx(source)
        n = self.n
        # Forget target
        for j in range(n):
            if j != ti:
                self.dbm[ti][j] = INF
                self.dbm[j][ti] = INF
        # target = source => target - source = 0
        self.dbm[ti][si] = 0
        self.dbm[si][ti] = 0
        # Propagate: target - j <= source - j, j - target <= j - source
        for j in range(n):
            if j != ti and j != si:
                self.dbm[ti][j] = min(self.dbm[ti][j], self.dbm[si][j])
                self.dbm[j][ti] = min(self.dbm[j][ti], self.dbm[j][si])

    def assign_add(self, target: str, x: str, c: float):
        """target := x + c."""
        ti, xi = self._idx(target), self._idx(x)
        n = self.n
        # Forget target
        for j in range(n):
            if j != ti:
                self.dbm[ti][j] = INF
                self.dbm[j][ti] = INF
        # target = x + c => target - x = c
        self.dbm[ti][xi] = c
        self.dbm[xi][ti] = -c
        # Propagate
        for j in range(n):
            if j != ti and j != xi:
                if self.dbm[xi][j] != INF:
                    self.dbm[ti][j] = min(self.dbm[ti][j],
                                          c + self.dbm[xi][j])
                if self.dbm[j][xi] != INF:
                    self.dbm[j][ti] = min(self.dbm[j][ti],
                                          self.dbm[j][xi] - c)

    def assign_sub_vars(self, target: str, x: str, y: str):
        """target := x - y."""
        ti, xi, yi = self._idx(target), self._idx(x), self._idx(y)
        n = self.n
        # Forget target
        for j in range(n):
            if j != ti:
                self.dbm[ti][j] = INF
                self.dbm[j][ti] = INF
        # target = x - y
        # target - x <= -lower(y) = dbm[0][yi]
        if self.dbm[0][yi] != INF:
            self.dbm[ti][xi] = self.dbm[0][yi]
        # x - target <= lower(y)
        if self.dbm[yi][0] != INF:
            # y <= dbm[yi][0], x - target = y, so x - target <= dbm[yi][0]
            pass
        # target <= upper(x) - lower(y)
        if self.dbm[xi][0] != INF and self.dbm[0][yi] != INF:
            self.dbm[ti][0] = min(self.dbm[ti][0],
                                  self.dbm[xi][0] + self.dbm[0][yi])
        # -target <= -lower(x) + upper(y)
        if self.dbm[0][xi] != INF and self.dbm[yi][0] != INF:
            self.dbm[0][ti] = min(self.dbm[0][ti],
                                  self.dbm[0][xi] + self.dbm[yi][0])
        # target - other: use x-y bounds + other bounds
        # This is approximate -- full closure will tighten
        self.close()

    def assign_add_vars(self, target: str, x: str, y: str):
        """target := x + y."""
        ti, xi, yi = self._idx(target), self._idx(x), self._idx(y)
        n = self.n
        for j in range(n):
            if j != ti:
                self.dbm[ti][j] = INF
                self.dbm[j][ti] = INF
        # target <= upper(x) + upper(y)
        if self.dbm[xi][0] != INF and self.dbm[yi][0] != INF:
            self.dbm[ti][0] = min(self.dbm[ti][0],
                                  self.dbm[xi][0] + self.dbm[yi][0])
        # -target <= -lower(x) + -lower(y)
        if self.dbm[0][xi] != INF and self.dbm[0][yi] != INF:
            self.dbm[0][ti] = min(self.dbm[0][ti],
                                  self.dbm[0][xi] + self.dbm[0][yi])
        self.close()

    def forget(self, var: str):
        """Remove all constraints involving var."""
        idx = self._idx(var)
        n = self.n
        for j in range(n):
            if j != idx:
                self.dbm[idx][j] = INF
                self.dbm[j][idx] = INF

    def add_var(self, name: str):
        """Add a new variable to the domain."""
        if name in self._var_index:
            return
        self.var_names.append(name)
        self._var_index[name] = self.n
        # Extend DBM
        for row in self.dbm:
            row.append(INF)
        self.dbm.append([INF] * (self.n + 1))
        self.dbm[self.n][self.n] = 0
        self.n += 1

    def join(self, other: 'ZoneDomain') -> 'ZoneDomain':
        """Least upper bound: pointwise max."""
        # Ensure same variables
        all_vars = list(dict.fromkeys(self.var_names + other.var_names))
        result = ZoneDomain(all_vars)
        if self.is_bot():
            r = other.copy()
            for v in all_vars:
                if v not in other._var_index:
                    r.add_var(v)
            return r
        if other.is_bot():
            r = self.copy()
            for v in all_vars:
                if v not in self._var_index:
                    r.add_var(v)
            return r
        for i in range(result.n):
            for j in range(result.n):
                vi = self._result_to_self(i, result)
                vj = self._result_to_self(j, result)
                oi = self._result_to_other(i, result, other)
                oj = self._result_to_other(j, result, other)
                s_val = self.dbm[vi][vj] if vi is not None and vj is not None else INF
                o_val = other.dbm[oi][oj] if oi is not None and oj is not None else INF
                result.dbm[i][j] = max(s_val, o_val)
        return result

    def _result_to_self(self, idx: int, result: 'ZoneDomain') -> Optional[int]:
        if idx == 0:
            return 0
        name = result.var_names[idx - 1]
        return self._var_index.get(name)

    def _result_to_other(self, idx: int, result: 'ZoneDomain',
                         other: 'ZoneDomain') -> Optional[int]:
        if idx == 0:
            return 0
        name = result.var_names[idx - 1]
        return other._var_index.get(name)

    def widen(self, other: 'ZoneDomain') -> 'ZoneDomain':
        """Widening: if new bound exceeds old, go to INF."""
        all_vars = list(dict.fromkeys(self.var_names + other.var_names))
        result = ZoneDomain(all_vars)
        if self.is_bot():
            r = other.copy()
            for v in all_vars:
                if v not in other._var_index:
                    r.add_var(v)
            return r
        for i in range(result.n):
            for j in range(result.n):
                vi = self._result_to_self(i, result)
                vj = self._result_to_self(j, result)
                oi = self._result_to_other(i, result, other)
                oj = self._result_to_other(j, result, other)
                s_val = self.dbm[vi][vj] if vi is not None and vj is not None else INF
                o_val = other.dbm[oi][oj] if oi is not None and oj is not None else INF
                if o_val <= s_val:
                    result.dbm[i][j] = s_val
                else:
                    result.dbm[i][j] = INF  # widen to infinity
        return result

    def meet(self, other: 'ZoneDomain') -> 'ZoneDomain':
        """Greatest lower bound: pointwise min."""
        all_vars = list(dict.fromkeys(self.var_names + other.var_names))
        result = ZoneDomain(all_vars)
        for i in range(result.n):
            for j in range(result.n):
                vi = self._result_to_self(i, result)
                vj = self._result_to_self(j, result)
                oi = self._result_to_other(i, result, other)
                oj = self._result_to_other(j, result, other)
                s_val = self.dbm[vi][vj] if vi is not None and vj is not None else INF
                o_val = other.dbm[oi][oj] if oi is not None and oj is not None else INF
                result.dbm[i][j] = min(s_val, o_val)
        return result

    def leq(self, other: 'ZoneDomain') -> bool:
        """self <= other iff every constraint in other is weaker."""
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        all_vars = list(dict.fromkeys(self.var_names + other.var_names))
        for var in all_vars:
            if var in self._var_index and var in other._var_index:
                si, oi = self._idx(var), other._idx(var)
                # Check bounds
                if self.dbm[si][0] > other.dbm[oi][0]:
                    return False
                if self.dbm[0][si] > other.dbm[0][oi]:
                    return False
        # Check pairwise
        for v1 in all_vars:
            for v2 in all_vars:
                if v1 == v2:
                    continue
                if v1 in self._var_index and v2 in self._var_index and \
                   v1 in other._var_index and v2 in other._var_index:
                    s1, s2 = self._idx(v1), self._idx(v2)
                    o1, o2 = other._idx(v1), other._idx(v2)
                    if self.dbm[s1][s2] > other.dbm[o1][o2]:
                        return False
        return True

    def equals(self, other: 'ZoneDomain') -> bool:
        return self.leq(other) and other.leq(self)

    def get_constraints(self) -> List[str]:
        """Human-readable list of constraints."""
        constraints = []
        for v in self.var_names:
            lo, hi = self.get_interval(v)
            if lo != -INF and hi != INF:
                constraints.append(f"{lo} <= {v} <= {hi}")
            elif lo != -INF:
                constraints.append(f"{v} >= {lo}")
            elif hi != INF:
                constraints.append(f"{v} <= {hi}")
        for i, v1 in enumerate(self.var_names):
            for j, v2 in enumerate(self.var_names):
                if i != j:
                    c = self.get_diff_bound(v1, v2)
                    if c != INF:
                        constraints.append(f"{v1} - {v2} <= {c}")
        return constraints

    def __repr__(self):
        if self.is_bot():
            return "Zone(BOT)"
        return f"Zone({', '.join(self.get_constraints())})"


# ---------------------------------------------------------------------------
# Octagon Domain (variable doubling on top of DBM)
# ---------------------------------------------------------------------------

class OctagonDomain:
    """Octagon abstract domain: tracks +/-x +/-y <= c constraints.

    Uses variable doubling: each variable x becomes x+ (index 2i) and x- (2i+1)
    where x+ represents +x and x- represents -x. Then x+ - y- <= c encodes x + y <= c.

    The DBM is of size 2n x 2n.
    """

    def __init__(self, var_names: List[str]):
        self.var_names = list(var_names)
        self._var_index = {name: i for i, name in enumerate(var_names)}
        n = len(var_names)
        size = 2 * n
        self.size = size
        # Initialize to TOP
        self.dbm = [[INF] * size for _ in range(size)]
        for i in range(size):
            self.dbm[i][i] = 0

    def copy(self) -> 'OctagonDomain':
        o = OctagonDomain(self.var_names)
        o.dbm = [row[:] for row in self.dbm]
        return o

    def _pos(self, name: str) -> int:
        """Index for +x."""
        return 2 * self._var_index[name]

    def _neg(self, name: str) -> int:
        """Index for -x."""
        return 2 * self._var_index[name] + 1

    def is_bot(self) -> bool:
        for i in range(self.size):
            if self.dbm[i][i] < 0:
                return True
        return False

    @staticmethod
    def bot(var_names: List[str]) -> 'OctagonDomain':
        o = OctagonDomain(var_names)
        if o.size > 0:
            o.dbm[0][0] = -1
        return o

    def set_upper(self, var: str, c: float):
        """x <= c => x+ - x- <= 2c."""
        p, n = self._pos(var), self._neg(var)
        self.dbm[p][n] = min(self.dbm[p][n], 2 * c)

    def set_lower(self, var: str, c: float):
        """x >= c => x- - x+ <= -2c."""
        p, n = self._pos(var), self._neg(var)
        self.dbm[n][p] = min(self.dbm[n][p], -2 * c)

    def set_sum_upper(self, x: str, y: str, c: float):
        """x + y <= c => x+ - y- <= c."""
        self.dbm[self._pos(x)][self._neg(y)] = min(
            self.dbm[self._pos(x)][self._neg(y)], c)
        # Also y+ - x- <= c (symmetry)
        self.dbm[self._pos(y)][self._neg(x)] = min(
            self.dbm[self._pos(y)][self._neg(x)], c)

    def set_sum_lower(self, x: str, y: str, c: float):
        """x + y >= c => x- - y+ <= -c (and y- - x+ <= -c)."""
        self.dbm[self._neg(x)][self._pos(y)] = min(
            self.dbm[self._neg(x)][self._pos(y)], -c)
        self.dbm[self._neg(y)][self._pos(x)] = min(
            self.dbm[self._neg(y)][self._pos(x)], -c)

    def set_diff_upper(self, x: str, y: str, c: float):
        """x - y <= c => x+ - y+ <= c."""
        self.dbm[self._pos(x)][self._pos(y)] = min(
            self.dbm[self._pos(x)][self._pos(y)], c)
        # Also y- - x- <= c
        self.dbm[self._neg(y)][self._neg(x)] = min(
            self.dbm[self._neg(y)][self._neg(x)], c)

    def set_diff_lower(self, x: str, y: str, c: float):
        """x - y >= c => y+ - x+ <= -c (and x- - y- <= -c)."""
        self.dbm[self._pos(y)][self._pos(x)] = min(
            self.dbm[self._pos(y)][self._pos(x)], -c)
        self.dbm[self._neg(x)][self._neg(y)] = min(
            self.dbm[self._neg(x)][self._neg(y)], -c)

    def get_upper(self, var: str) -> float:
        """x <= c where c = dbm[x+][x-] / 2."""
        val = self.dbm[self._pos(var)][self._neg(var)]
        if val == INF:
            return INF
        return val / 2.0

    def get_lower(self, var: str) -> float:
        """x >= c where c = -dbm[x-][x+] / 2."""
        val = self.dbm[self._neg(var)][self._pos(var)]
        if val == INF:
            return -INF
        return -val / 2.0

    def get_interval(self, var: str) -> Tuple[float, float]:
        return (self.get_lower(var), self.get_upper(var))

    def get_sum_bound(self, x: str, y: str) -> float:
        """x + y <= c."""
        return self.dbm[self._pos(x)][self._neg(y)]

    def get_diff_bound(self, x: str, y: str) -> float:
        """x - y <= c."""
        return self.dbm[self._pos(x)][self._pos(y)]

    def close(self):
        """Strong closure for octagon: Floyd-Warshall + tightening."""
        n = self.size
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                if self.dbm[i][k] == INF:
                    continue
                for j in range(n):
                    if self.dbm[k][j] == INF:
                        continue
                    d = self.dbm[i][k] + self.dbm[k][j]
                    if d < self.dbm[i][j]:
                        self.dbm[i][j] = d
        # Strengthening: use unary bounds to tighten binary
        for i in range(n):
            bar_i = i ^ 1  # partner (pos <-> neg)
            for j in range(n):
                if i == j:
                    continue
                bar_j = j ^ 1
                # dbm[i][j] <= (dbm[i][bar_i] + dbm[bar_j][j]) / 2
                if self.dbm[i][bar_i] != INF and self.dbm[bar_j][j] != INF:
                    candidate = (self.dbm[i][bar_i] + self.dbm[bar_j][j]) / 2.0
                    if candidate < self.dbm[i][j]:
                        self.dbm[i][j] = candidate

    def assign_const(self, var: str, c: float):
        """var := c."""
        p, n = self._pos(var), self._neg(var)
        # Forget var
        for j in range(self.size):
            if j != p and j != n:
                self.dbm[p][j] = INF
                self.dbm[j][p] = INF
                self.dbm[n][j] = INF
                self.dbm[j][n] = INF
        # Set var = c
        self.dbm[p][n] = 2 * c    # var+ - var- <= 2c => var <= c
        self.dbm[n][p] = -2 * c   # var- - var+ <= -2c => var >= c
        # Propagate constraints with other variables.
        # other_upper = dbm[op][on] / 2, other_lower = -dbm[on][op] / 2
        for other in self.var_names:
            if other == var:
                continue
            op, on = self._pos(other), self._neg(other)
            o_hi_2 = self.dbm[op][on]   # 2 * upper(other)
            o_lo_2 = self.dbm[on][op]   # -2 * lower(other)
            # var - other <= c - lower(other)
            # DBM: dbm[p][op] = var+ - other+ = var - other
            if o_lo_2 != INF:
                other_lo = -o_lo_2 / 2.0
                self.dbm[p][op] = min(self.dbm[p][op], c - other_lo)
            # other - var <= upper(other) - c
            # DBM: dbm[op][p] = other+ - var+ = other - var
            if o_hi_2 != INF:
                other_hi = o_hi_2 / 2.0
                self.dbm[op][p] = min(self.dbm[op][p], other_hi - c)
            # var + other <= c + upper(other)
            # DBM: dbm[p][on] = var+ - other- = var + other
            if o_hi_2 != INF:
                other_hi = o_hi_2 / 2.0
                self.dbm[p][on] = min(self.dbm[p][on], c + other_hi)
            # symmetric: other + var <= upper(other) + c
            # DBM: dbm[op][n] = other+ - var- = other + var
            if o_hi_2 != INF:
                other_hi = o_hi_2 / 2.0
                self.dbm[op][n] = min(self.dbm[op][n], other_hi + c)
            # -(var + other) <= -(c + lower(other))
            # var- - other+ = -var - other, bound = -(c + lower(other))
            if o_lo_2 != INF:
                other_lo = -o_lo_2 / 2.0
                self.dbm[n][op] = min(self.dbm[n][op], -(c + other_lo))
            # other- - var+ = -other - var, bound = -(lower(other) + c)
            if o_lo_2 != INF:
                other_lo = -o_lo_2 / 2.0
                self.dbm[on][p] = min(self.dbm[on][p], -(other_lo + c))
            # -var + other = other - var (same as dbm[op][p], already set above)
            # var - other (already set as dbm[p][op] above)
            # -var - (-other) = other - var (dbm[on][n] = -other- - var- = -(-other) - (-var) = other - var)
            # Wait: dbm[n][on] = var- - other- = (-var) - (-other) = other - var
            if o_lo_2 != INF:
                other_lo = -o_lo_2 / 2.0
                self.dbm[n][on] = min(self.dbm[n][on], other_lo - c)  # wait...
            # Actually: other - var <= upper(other) - c
            # dbm[n][on] = var- - other- = (-var) - (-other) = other - var
            if o_hi_2 != INF:
                other_hi = o_hi_2 / 2.0
                self.dbm[n][on] = min(self.dbm[n][on], other_hi - c)
            # dbm[on][n] = other- - var- = (-other) - (-var) = var - other
            if o_lo_2 != INF:
                other_lo = -o_lo_2 / 2.0
                self.dbm[on][n] = min(self.dbm[on][n], c - other_lo)

    def assign_var(self, target: str, source: str):
        """target := source."""
        if target == source:
            return
        tp, tn = self._pos(target), self._neg(target)
        sp, sn = self._pos(source), self._neg(source)
        # Forget target
        for j in range(self.size):
            if j != tp and j != tn:
                self.dbm[tp][j] = INF
                self.dbm[j][tp] = INF
                self.dbm[tn][j] = INF
                self.dbm[j][tn] = INF
        self.dbm[tp][tn] = INF
        self.dbm[tn][tp] = INF
        # target = source => target+ - source+ = 0, source- - target- = 0
        self.dbm[tp][sp] = 0
        self.dbm[sp][tp] = 0
        self.dbm[tn][sn] = 0
        self.dbm[sn][tn] = 0
        # Unary bounds: target+ - target- = source+ - source-
        self.dbm[tp][tn] = self.dbm[sp][sn]
        self.dbm[tn][tp] = self.dbm[sn][sp]
        # Cross: target+ - source- = source+ - source- (via t+ - s+ = 0)
        self.dbm[tp][sn] = self.dbm[sp][sn]
        self.dbm[sn][tp] = self.dbm[sn][sp]
        # source+ - target- = source+ - source- (via s- - t- = 0)
        self.dbm[sp][tn] = self.dbm[sp][sn]
        self.dbm[tn][sp] = self.dbm[sn][sp]
        # Propagate through other variables
        for other in self.var_names:
            if other == target or other == source:
                continue
            op, on = self._pos(other), self._neg(other)
            # target+ - other+ <= source+ - other+
            self.dbm[tp][op] = min(self.dbm[tp][op], self.dbm[sp][op])
            self.dbm[op][tp] = min(self.dbm[op][tp], self.dbm[op][sp])
            # target- - other- <= source- - other-
            self.dbm[tn][on] = min(self.dbm[tn][on], self.dbm[sn][on])
            self.dbm[on][tn] = min(self.dbm[on][tn], self.dbm[on][sn])
            # target+ - other- <= source+ - other-
            self.dbm[tp][on] = min(self.dbm[tp][on], self.dbm[sp][on])
            self.dbm[on][tp] = min(self.dbm[on][tp], self.dbm[on][sp])
            # target- - other+ <= source- - other+
            self.dbm[tn][op] = min(self.dbm[tn][op], self.dbm[sn][op])
            self.dbm[op][tn] = min(self.dbm[op][tn], self.dbm[op][sn])

    def assign_add_const(self, target: str, source: str, c: float):
        """target := source + c."""
        if target == source:
            # In-place: shift all bounds by c
            p, n = self._pos(target), self._neg(target)
            for j in range(self.size):
                if j != p and j != n:
                    if self.dbm[p][j] != INF:
                        self.dbm[p][j] += c
                    if self.dbm[j][p] != INF:
                        self.dbm[j][p] -= c
                    if self.dbm[n][j] != INF:
                        self.dbm[n][j] -= c
                    if self.dbm[j][n] != INF:
                        self.dbm[j][n] += c
            self.dbm[p][n] += 2 * c
            self.dbm[n][p] -= 2 * c
            return
        # General case: copy then shift
        self.assign_var(target, source)
        tp, tn = self._pos(target), self._neg(target)
        for j in range(self.size):
            if j != tp and j != tn:
                if self.dbm[tp][j] != INF:
                    self.dbm[tp][j] += c
                if self.dbm[j][tp] != INF:
                    self.dbm[j][tp] -= c
                if self.dbm[tn][j] != INF:
                    self.dbm[tn][j] -= c
                if self.dbm[j][tn] != INF:
                    self.dbm[j][tn] += c
        self.dbm[tp][tn] += 2 * c
        self.dbm[tn][tp] -= 2 * c

    def forget(self, var: str):
        """Remove all constraints involving var."""
        p, n = self._pos(var), self._neg(var)
        for j in range(self.size):
            if j != p:
                self.dbm[p][j] = INF
                self.dbm[j][p] = INF
            if j != n:
                self.dbm[n][j] = INF
                self.dbm[j][n] = INF
        self.dbm[p][n] = INF
        self.dbm[n][p] = INF
        self.dbm[p][p] = 0
        self.dbm[n][n] = 0

    def add_var(self, name: str):
        """Add a new variable."""
        if name in self._var_index:
            return
        self._var_index[name] = len(self.var_names)
        self.var_names.append(name)
        # Add two rows/cols
        for row in self.dbm:
            row.append(INF)
            row.append(INF)
        self.dbm.append([INF] * (self.size + 2))
        self.dbm.append([INF] * (self.size + 2))
        self.size += 2
        self.dbm[self.size - 2][self.size - 2] = 0
        self.dbm[self.size - 1][self.size - 1] = 0

    def join(self, other: 'OctagonDomain') -> 'OctagonDomain':
        """Pointwise max."""
        all_vars = list(dict.fromkeys(self.var_names + other.var_names))
        result = OctagonDomain(all_vars)
        if self.is_bot():
            r = other.copy()
            for v in all_vars:
                if v not in other._var_index:
                    r.add_var(v)
            return r
        if other.is_bot():
            r = self.copy()
            for v in all_vars:
                if v not in self._var_index:
                    r.add_var(v)
            return r
        for i in range(result.size):
            for j in range(result.size):
                si = self._map_idx(i, result)
                sj = self._map_idx(j, result)
                oi = other._map_idx(i, result)
                oj = other._map_idx(j, result)
                s_val = self.dbm[si][sj] if si is not None and sj is not None else INF
                o_val = other.dbm[oi][oj] if oi is not None and oj is not None else INF
                result.dbm[i][j] = max(s_val, o_val)
        return result

    def _map_idx(self, result_idx: int, result: 'OctagonDomain') -> Optional[int]:
        """Map index from result's variable numbering to self's."""
        var_num = result_idx // 2
        is_neg = result_idx % 2
        if var_num >= len(result.var_names):
            return None
        name = result.var_names[var_num]
        if name not in self._var_index:
            return None
        return 2 * self._var_index[name] + is_neg

    def widen(self, other: 'OctagonDomain') -> 'OctagonDomain':
        """Standard widening: if bound increases, go to INF."""
        all_vars = list(dict.fromkeys(self.var_names + other.var_names))
        result = OctagonDomain(all_vars)
        if self.is_bot():
            r = other.copy()
            for v in all_vars:
                if v not in other._var_index:
                    r.add_var(v)
            return r
        for i in range(result.size):
            for j in range(result.size):
                si = self._map_idx(i, result)
                sj = self._map_idx(j, result)
                oi = other._map_idx(i, result)
                oj = other._map_idx(j, result)
                s_val = self.dbm[si][sj] if si is not None and sj is not None else INF
                o_val = other.dbm[oi][oj] if oi is not None and oj is not None else INF
                if o_val <= s_val:
                    result.dbm[i][j] = s_val
                else:
                    result.dbm[i][j] = INF
        return result

    def meet(self, other: 'OctagonDomain') -> 'OctagonDomain':
        """Pointwise min."""
        all_vars = list(dict.fromkeys(self.var_names + other.var_names))
        result = OctagonDomain(all_vars)
        for i in range(result.size):
            for j in range(result.size):
                si = self._map_idx(i, result)
                sj = self._map_idx(j, result)
                oi = other._map_idx(i, result)
                oj = other._map_idx(j, result)
                s_val = self.dbm[si][sj] if si is not None and sj is not None else INF
                o_val = other.dbm[oi][oj] if oi is not None and oj is not None else INF
                result.dbm[i][j] = min(s_val, o_val)
        return result

    def leq(self, other: 'OctagonDomain') -> bool:
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        # Every constraint in self must be at least as tight as in other
        for v in self.var_names:
            if v not in other._var_index:
                continue
            sp, sn = self._pos(v), self._neg(v)
            op, on = other._pos(v), other._neg(v)
            if self.dbm[sp][sn] > other.dbm[op][on]:
                return False
            if self.dbm[sn][sp] > other.dbm[on][op]:
                return False
        for i, v1 in enumerate(self.var_names):
            for j, v2 in enumerate(self.var_names):
                if v1 == v2:
                    continue
                if v1 not in other._var_index or v2 not in other._var_index:
                    continue
                for a in [self._pos(v1), self._neg(v1)]:
                    for b in [self._pos(v2), self._neg(v2)]:
                        oa = other._map_idx(a, self)
                        # Recompute properly
                        pass
        # Simpler approach: check all entries for shared variables
        shared = [v for v in self.var_names if v in other._var_index]
        for v1 in shared:
            for v2 in shared:
                for di in [0, 1]:
                    for dj in [0, 1]:
                        si = 2 * self._var_index[v1] + di
                        sj = 2 * self._var_index[v2] + dj
                        oi = 2 * other._var_index[v1] + di
                        oj = 2 * other._var_index[v2] + dj
                        if self.dbm[si][sj] > other.dbm[oi][oj]:
                            return False
        return True

    def equals(self, other: 'OctagonDomain') -> bool:
        return self.leq(other) and other.leq(self)

    def get_constraints(self) -> List[str]:
        """Human-readable constraints."""
        constraints = []
        for v in self.var_names:
            lo, hi = self.get_interval(v)
            if lo != -INF and hi != INF:
                if lo == hi:
                    constraints.append(f"{v} = {_fmt(lo)}")
                else:
                    constraints.append(f"{_fmt(lo)} <= {v} <= {_fmt(hi)}")
            elif lo != -INF:
                constraints.append(f"{v} >= {_fmt(lo)}")
            elif hi != INF:
                constraints.append(f"{v} <= {_fmt(hi)}")
        for i, v1 in enumerate(self.var_names):
            for j, v2 in enumerate(self.var_names):
                if i >= j:
                    continue
                # x - y bound
                diff = self.get_diff_bound(v1, v2)
                if diff != INF:
                    constraints.append(f"{v1} - {v2} <= {_fmt(diff)}")
                diff2 = self.get_diff_bound(v2, v1)
                if diff2 != INF:
                    constraints.append(f"{v2} - {v1} <= {_fmt(diff2)}")
                # x + y bound
                sbound = self.get_sum_bound(v1, v2)
                if sbound != INF:
                    constraints.append(f"{v1} + {v2} <= {_fmt(sbound)}")
                # -(x+y) bound via neg indices
                neg_sbound = self.dbm[self._neg(v1)][self._pos(v2)]
                if neg_sbound != INF:
                    constraints.append(f"{v1} + {v2} >= {_fmt(-neg_sbound)}")
        return constraints

    def __repr__(self):
        if self.is_bot():
            return "Octagon(BOT)"
        cs = self.get_constraints()
        if not cs:
            return "Octagon(TOP)"
        return f"Octagon({', '.join(cs[:10])}{'...' if len(cs) > 10 else ''})"


def _fmt(v: float) -> str:
    if v == INF:
        return 'inf'
    if v == -INF:
        return '-inf'
    if v == int(v):
        return str(int(v))
    return str(v)


# ---------------------------------------------------------------------------
# C10 Interpreter with Octagon Domain
# ---------------------------------------------------------------------------

class OctagonInterpreter:
    """Abstract interpreter for C10 source using octagon domain."""

    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations
        self.warnings = []

    def analyze(self, source: str) -> dict:
        tokens = lex(source)
        ast = Parser(tokens).parse()
        oct_env = OctagonDomain([])
        functions = {}
        for stmt in ast.stmts:
            cls = stmt.__class__.__name__
            if cls == 'FnDecl':
                functions[stmt.name] = stmt
            else:
                oct_env = self._interpret_stmt(stmt, oct_env, functions)
        return {
            'env': oct_env,
            'warnings': self.warnings,
            'functions': list(functions.keys()),
        }

    def _interpret_stmt(self, stmt, env: OctagonDomain,
                        functions: dict) -> OctagonDomain:
        if env.is_bot():
            return env
        cls = stmt.__class__.__name__
        if cls == 'LetDecl':
            return self._interpret_let(stmt, env, functions)
        elif cls == 'Assign':
            return self._interpret_assign(stmt, env, functions)
        elif cls == 'IfStmt':
            return self._interpret_if(stmt, env, functions)
        elif cls == 'WhileStmt':
            return self._interpret_while(stmt, env, functions)
        elif cls == 'Block':
            return self._interpret_block(stmt, env, functions)
        elif cls == 'ReturnStmt':
            return env
        elif cls == 'PrintStmt':
            return env
        return env

    def _interpret_block(self, block, env, functions):
        stmts = block.stmts if hasattr(block, 'stmts') else block
        if isinstance(stmts, list):
            for s in stmts:
                env = self._interpret_stmt(s, env, functions)
        return env

    def _interpret_let(self, stmt, env, functions):
        name = stmt.name
        if name not in env._var_index:
            env.add_var(name)
        val = self._eval_expr(stmt.value, env)
        self._apply_assignment(name, stmt.value, val, env)
        return env

    def _interpret_assign(self, stmt, env, functions):
        name = stmt.name
        if name not in env._var_index:
            env.add_var(name)
        val = self._eval_expr(stmt.value, env)
        self._apply_assignment(name, stmt.value, val, env)
        return env

    def _apply_assignment(self, target: str, expr, val, env: OctagonDomain):
        """Apply assignment target := expr to the octagon."""
        cls = expr.__class__.__name__
        if cls == 'IntLit':
            env.assign_const(target, float(expr.value))
        elif cls == 'Var':
            src = expr.name
            if src in env._var_index:
                env.assign_var(target, src)
            else:
                env.forget(target)
        elif cls == 'BinOp':
            op, left, right = expr.op, expr.left, expr.right
            left_cls = left.__class__.__name__
            right_cls = right.__class__.__name__
            if op == '+':
                if left_cls == 'Var' and right_cls == 'IntLit':
                    if left.name in env._var_index:
                        env.assign_add_const(target, left.name, float(right.value))
                        return
                elif left_cls == 'IntLit' and right_cls == 'Var':
                    if right.name in env._var_index:
                        env.assign_add_const(target, right.name, float(left.value))
                        return
                elif left_cls == 'Var' and right_cls == 'Var':
                    if left.name in env._var_index and right.name in env._var_index:
                        # target = x + y -- fall back to interval bounds
                        env.forget(target)
                        lo_x, hi_x = env.get_interval(left.name)
                        lo_y, hi_y = env.get_interval(right.name)
                        if lo_x != -INF and lo_y != -INF:
                            env.set_lower(target, lo_x + lo_y)
                        if hi_x != INF and hi_y != INF:
                            env.set_upper(target, hi_x + hi_y)
                        # Also set relational: target - x <= upper(y), target - y <= upper(x)
                        if hi_y != INF:
                            env.set_diff_upper(target, left.name, hi_y)
                        if lo_y != -INF:
                            env.set_diff_lower(target, left.name, lo_y)
                        if hi_x != INF:
                            env.set_diff_upper(target, right.name, hi_x)
                        if lo_x != -INF:
                            env.set_diff_lower(target, right.name, lo_x)
                        return
            elif op == '-':
                if left_cls == 'Var' and right_cls == 'IntLit':
                    if left.name in env._var_index:
                        env.assign_add_const(target, left.name, -float(right.value))
                        return
                elif left_cls == 'Var' and right_cls == 'Var':
                    if left.name in env._var_index and right.name in env._var_index:
                        # target = x - y
                        lo_x, hi_x = env.get_interval(left.name)
                        lo_y, hi_y = env.get_interval(right.name)
                        env.forget(target)
                        # Unary bounds
                        if lo_x != -INF and hi_y != INF:
                            env.set_lower(target, lo_x - hi_y)
                        if hi_x != INF and lo_y != -INF:
                            env.set_upper(target, hi_x - lo_y)
                        # Relational: target = x - y => target + y = x
                        # Sum constraint: target + y <= upper(x), target + y >= lower(x)
                        if hi_x != INF:
                            env.set_sum_upper(target, right.name, hi_x)
                        if lo_x != -INF:
                            env.set_sum_lower(target, right.name, lo_x)
                        # Diff constraint: target - x = -y
                        # target - x <= -lower(y), x - target <= upper(y)
                        if lo_y != -INF:
                            env.set_diff_upper(target, left.name, -lo_y)
                        if hi_y != INF:
                            env.set_diff_lower(target, left.name, -hi_y)
                        return
            elif op == '*':
                if left_cls == 'Var' and right_cls == 'IntLit':
                    c = right.value
                    if left.name in env._var_index and c != 0:
                        lo, hi = env.get_interval(left.name)
                        env.forget(target)
                        if c > 0:
                            if lo != -INF:
                                env.set_lower(target, lo * c)
                            if hi != INF:
                                env.set_upper(target, hi * c)
                        else:
                            if hi != INF:
                                env.set_lower(target, hi * c)
                            if lo != -INF:
                                env.set_upper(target, lo * c)
                        return
                elif left_cls == 'IntLit' and right_cls == 'Var':
                    c = left.value
                    if right.name in env._var_index and c != 0:
                        lo, hi = env.get_interval(right.name)
                        env.forget(target)
                        if c > 0:
                            if lo != -INF:
                                env.set_lower(target, lo * c)
                            if hi != INF:
                                env.set_upper(target, hi * c)
                        else:
                            if hi != INF:
                                env.set_lower(target, hi * c)
                            if lo != -INF:
                                env.set_upper(target, lo * c)
                        return
            # Fall through: non-relational
            env.forget(target)
            if val is not None:
                lo, hi = val
                if lo != -INF:
                    env.set_lower(target, lo)
                if hi != INF:
                    env.set_upper(target, hi)
        elif cls == 'UnaryOp':
            if expr.op == '-' and expr.operand.__class__.__name__ == 'Var':
                src = expr.operand.name
                if src in env._var_index:
                    # target = -src: flip sign
                    lo, hi = env.get_interval(src)
                    env.forget(target)
                    if hi != INF:
                        env.set_lower(target, -hi)
                    if lo != -INF:
                        env.set_upper(target, -lo)
                    # Relational: target + src = 0
                    env.set_sum_upper(target, src, 0)
                    env.set_sum_lower(target, src, 0)
                    return
            env.forget(target)
            if val is not None:
                lo, hi = val
                if lo != -INF:
                    env.set_lower(target, lo)
                if hi != INF:
                    env.set_upper(target, hi)
        else:
            env.forget(target)
            if val is not None:
                lo, hi = val
                if lo != -INF:
                    env.set_lower(target, lo)
                if hi != INF:
                    env.set_upper(target, hi)

    def _eval_expr(self, expr, env: OctagonDomain) -> Optional[Tuple[float, float]]:
        """Evaluate expression to interval bounds (for non-relational fallback)."""
        cls = expr.__class__.__name__
        if cls == 'IntLit':
            v = float(expr.value)
            return (v, v)
        elif cls == 'Var':
            if expr.name in env._var_index:
                return env.get_interval(expr.name)
            return (-INF, INF)
        elif cls == 'BinOp':
            l = self._eval_expr(expr.left, env)
            r = self._eval_expr(expr.right, env)
            if l is None or r is None:
                return None
            lo_l, hi_l = l
            lo_r, hi_r = r
            if expr.op == '+':
                lo = lo_l + lo_r if lo_l != -INF and lo_r != -INF else -INF
                hi = hi_l + hi_r if hi_l != INF and hi_r != INF else INF
                return (lo, hi)
            elif expr.op == '-':
                lo = lo_l - hi_r if lo_l != -INF and hi_r != INF else -INF
                hi = hi_l - lo_r if hi_l != INF and lo_r != -INF else INF
                return (lo, hi)
            elif expr.op == '*':
                if lo_l == -INF or hi_l == INF or lo_r == -INF or hi_r == INF:
                    return (-INF, INF)
                products = [lo_l * lo_r, lo_l * hi_r, hi_l * lo_r, hi_l * hi_r]
                return (min(products), max(products))
            elif expr.op == '/':
                # Check div by zero
                if lo_r <= 0 <= hi_r:
                    self.warnings.append(f"Possible division by zero")
                    return (-INF, INF)
                if lo_l == -INF or hi_l == INF:
                    return (-INF, INF)
                vals = []
                for a in [lo_r, hi_r]:
                    if a != 0:
                        vals.append(lo_l / a)
                        vals.append(hi_l / a)
                if not vals:
                    return (-INF, INF)
                return (min(vals), max(vals))
        elif cls == 'UnaryOp':
            if expr.op == '-':
                v = self._eval_expr(expr.operand, env)
                if v is None:
                    return None
                return (-v[1], -v[0])
        elif cls == 'CallExpr':
            return (-INF, INF)
        return (-INF, INF)

    def _refine_condition(self, cond, env: OctagonDomain
                          ) -> Tuple[OctagonDomain, OctagonDomain]:
        """Refine env for then-branch and else-branch."""
        then_env = env.copy()
        else_env = env.copy()
        cls = cond.__class__.__name__
        if cls == 'BinOp':
            op = cond.op
            left, right = cond.left, cond.right
            left_cls = left.__class__.__name__
            right_cls = right.__class__.__name__

            if left_cls == 'Var' and right_cls == 'IntLit':
                v, c = left.name, float(right.value)
                if v in env._var_index:
                    if op == '<':
                        then_env.set_upper(v, c - 1)
                        else_env.set_lower(v, c)
                    elif op == '<=':
                        then_env.set_upper(v, c)
                        else_env.set_lower(v, c + 1)
                    elif op == '>':
                        then_env.set_lower(v, c + 1)
                        else_env.set_upper(v, c)
                    elif op == '>=':
                        then_env.set_lower(v, c)
                        else_env.set_upper(v, c - 1)
                    elif op == '==':
                        then_env.set_lower(v, c)
                        then_env.set_upper(v, c)
                    elif op == '!=':
                        pass  # Can't refine precisely

            elif left_cls == 'IntLit' and right_cls == 'Var':
                c, v = float(left.value), right.name
                if v in env._var_index:
                    if op == '<':
                        then_env.set_lower(v, c + 1)
                        else_env.set_upper(v, c)
                    elif op == '<=':
                        then_env.set_lower(v, c)
                        else_env.set_upper(v, c - 1)
                    elif op == '>':
                        then_env.set_upper(v, c - 1)
                        else_env.set_lower(v, c)
                    elif op == '>=':
                        then_env.set_upper(v, c)
                        else_env.set_lower(v, c + 1)
                    elif op == '==':
                        then_env.set_lower(v, c)
                        then_env.set_upper(v, c)

            elif left_cls == 'Var' and right_cls == 'Var':
                x, y = left.name, right.name
                if x in env._var_index and y in env._var_index:
                    if op == '<':
                        # x < y => x - y <= -1
                        then_env.set_diff_upper(x, y, -1)
                        # else: x >= y => y - x <= 0
                        else_env.set_diff_upper(y, x, 0)
                    elif op == '<=':
                        then_env.set_diff_upper(x, y, 0)
                        else_env.set_diff_upper(y, x, -1)
                    elif op == '>':
                        then_env.set_diff_upper(y, x, -1)
                        else_env.set_diff_upper(x, y, 0)
                    elif op == '>=':
                        then_env.set_diff_upper(y, x, 0)
                        else_env.set_diff_upper(x, y, -1)
                    elif op == '==':
                        then_env.set_diff_upper(x, y, 0)
                        then_env.set_diff_upper(y, x, 0)

            # Handle x - y < c pattern
            if left_cls == 'BinOp' and left.op == '-':
                if (left.left.__class__.__name__ == 'Var' and
                    left.right.__class__.__name__ == 'Var' and
                    right_cls == 'IntLit'):
                    x = left.left.name
                    y = left.right.name
                    c = float(right.value)
                    if x in env._var_index and y in env._var_index:
                        if op == '<':
                            then_env.set_diff_upper(x, y, c - 1)
                            else_env.set_diff_lower(x, y, c)
                        elif op == '<=':
                            then_env.set_diff_upper(x, y, c)
                            else_env.set_diff_lower(x, y, c + 1)
                        elif op == '>':
                            then_env.set_diff_lower(x, y, c + 1)
                            else_env.set_diff_upper(x, y, c)
                        elif op == '>=':
                            then_env.set_diff_lower(x, y, c)
                            else_env.set_diff_upper(x, y, c - 1)

        return then_env, else_env

    def _interpret_if(self, stmt, env, functions):
        then_env, else_env = self._refine_condition(stmt.cond, env)
        then_env = self._interpret_block(stmt.then_body, then_env, functions)
        if stmt.else_body:
            else_env = self._interpret_block(stmt.else_body, else_env, functions)
        if then_env.is_bot():
            return else_env
        if else_env.is_bot():
            return then_env
        return then_env.join(else_env)

    def _interpret_while(self, stmt, env, functions):
        # Fixpoint with widening
        current = env.copy()
        for iteration in range(self.max_iterations):
            then_env, _ = self._refine_condition(stmt.cond, current)
            body_env = self._interpret_block(stmt.body, then_env, functions)
            next_env = current.widen(body_env.join(current))
            if next_env.equals(current):
                break
            current = next_env
        # Exit condition
        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def octagon_analyze(source: str, max_iterations: int = 50) -> dict:
    """Analyze C10 source with octagon domain.

    Returns dict with 'env' (OctagonDomain), 'warnings', 'functions'.
    """
    interp = OctagonInterpreter(max_iterations=max_iterations)
    return interp.analyze(source)


def zone_analyze(source: str, max_iterations: int = 50) -> dict:
    """Analyze C10 source with zone domain.

    Returns dict with 'env' (ZoneDomain), 'warnings', 'functions'.
    """
    interp = ZoneInterpreter(max_iterations=max_iterations)
    return interp.analyze(source)


def get_variable_range(source: str, var_name: str,
                       domain: str = 'octagon') -> Tuple[float, float]:
    """Get variable interval from relational analysis."""
    if domain == 'octagon':
        result = octagon_analyze(source)
    else:
        result = zone_analyze(source)
    env = result['env']
    if var_name in env._var_index:
        return env.get_interval(var_name)
    return (-INF, INF)


def get_relational_constraints(source: str,
                                domain: str = 'octagon') -> List[str]:
    """Get all relational constraints from analysis."""
    if domain == 'octagon':
        result = octagon_analyze(source)
    else:
        result = zone_analyze(source)
    return result['env'].get_constraints()


def compare_analyses(source: str) -> dict:
    """Compare octagon vs interval (C039) analysis results."""
    # Octagon
    oct_result = octagon_analyze(source)
    oct_env = oct_result['env']

    # C039 interval
    c039_result = c039_analyze(source)
    c039_env = c039_result['env']

    comparison = {
        'octagon_constraints': oct_env.get_constraints(),
        'interval_results': {},
        'precision_gains': [],
    }

    for var in oct_env.var_names:
        oct_lo, oct_hi = oct_env.get_interval(var)
        c039_interval = c039_env.get_interval(var)
        c039_lo = c039_interval.lo if c039_interval else -INF
        c039_hi = c039_interval.hi if c039_interval else INF

        comparison['interval_results'][var] = {
            'octagon': (oct_lo, oct_hi),
            'c039': (c039_lo, c039_hi),
        }

        # Check if octagon is more precise
        oct_width = oct_hi - oct_lo if oct_lo != -INF and oct_hi != INF else INF
        c039_width = c039_hi - c039_lo if c039_lo != -INF and c039_hi != INF else INF
        if oct_width < c039_width:
            comparison['precision_gains'].append({
                'variable': var,
                'octagon_range': (oct_lo, oct_hi),
                'interval_range': (c039_lo, c039_hi),
                'improvement': c039_width - oct_width if c039_width != INF else 'infinite',
            })

    # Relational constraints (octagon only)
    comparison['relational_only'] = []
    for c in oct_env.get_constraints():
        if '-' in c and '<=' in c and c.count('<=') == 1:
            parts = c.split('<=')
            if '-' in parts[0]:
                comparison['relational_only'].append(c)
        elif '+' in c:
            comparison['relational_only'].append(c)

    return comparison


def verify_relational_property(source: str, property_str: str) -> dict:
    """Verify a relational property about the program.

    property_str examples:
    - "x - y <= 5"
    - "x + y <= 10"
    - "x <= 100"
    """
    result = octagon_analyze(source)
    env = result['env']

    # Parse simple properties
    prop = property_str.strip()
    verdict = 'UNKNOWN'
    details = ''

    if ' - ' in prop and '<=' in prop:
        parts = prop.split('<=')
        lhs = parts[0].strip()
        rhs = float(parts[1].strip())
        vars_parts = lhs.split(' - ')
        x, y = vars_parts[0].strip(), vars_parts[1].strip()
        if x in env._var_index and y in env._var_index:
            bound = env.get_diff_bound(x, y)
            if bound <= rhs:
                verdict = 'VERIFIED'
                details = f"Proven: {x} - {y} <= {_fmt(bound)} <= {_fmt(rhs)}"
            else:
                verdict = 'UNKNOWN'
                details = f"Best bound: {x} - {y} <= {_fmt(bound)}, need <= {_fmt(rhs)}"
    elif ' + ' in prop and '<=' in prop:
        parts = prop.split('<=')
        lhs = parts[0].strip()
        rhs = float(parts[1].strip())
        vars_parts = lhs.split(' + ')
        x, y = vars_parts[0].strip(), vars_parts[1].strip()
        if x in env._var_index and y in env._var_index:
            bound = env.get_sum_bound(x, y)
            if bound <= rhs:
                verdict = 'VERIFIED'
                details = f"Proven: {x} + {y} <= {_fmt(bound)} <= {_fmt(rhs)}"
            else:
                verdict = 'UNKNOWN'
                details = f"Best bound: {x} + {y} <= {_fmt(bound)}, need <= {_fmt(rhs)}"
    elif '<=' in prop:
        parts = prop.split('<=')
        var = parts[0].strip()
        bound_val = float(parts[1].strip())
        if var in env._var_index:
            hi = env.get_upper(var)
            if hi <= bound_val:
                verdict = 'VERIFIED'
                details = f"Proven: {var} <= {_fmt(hi)} <= {_fmt(bound_val)}"
            else:
                verdict = 'UNKNOWN'
                details = f"Best bound: {var} <= {_fmt(hi)}, need <= {_fmt(bound_val)}"
    elif '>=' in prop:
        parts = prop.split('>=')
        var = parts[0].strip()
        bound_val = float(parts[1].strip())
        if var in env._var_index:
            lo = env.get_lower(var)
            if lo >= bound_val:
                verdict = 'VERIFIED'
                details = f"Proven: {var} >= {_fmt(lo)} >= {_fmt(bound_val)}"
            else:
                verdict = 'UNKNOWN'
                details = f"Best bound: {var} >= {_fmt(lo)}, need >= {_fmt(bound_val)}"

    return {
        'property': property_str,
        'verdict': verdict,
        'details': details,
        'all_constraints': env.get_constraints(),
    }


# ---------------------------------------------------------------------------
# Zone Interpreter (reuses OctagonInterpreter structure with ZoneDomain)
# ---------------------------------------------------------------------------

class ZoneInterpreter:
    """Abstract interpreter for C10 using zone domain."""

    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations
        self.warnings = []

    def analyze(self, source: str) -> dict:
        tokens = lex(source)
        ast = Parser(tokens).parse()
        zone = ZoneDomain([])
        functions = {}
        for stmt in ast.stmts:
            cls = stmt.__class__.__name__
            if cls == 'FnDecl':
                functions[stmt.name] = stmt
            else:
                zone = self._interpret_stmt(stmt, zone, functions)
        return {
            'env': zone,
            'warnings': self.warnings,
            'functions': list(functions.keys()),
        }

    def _interpret_stmt(self, stmt, env: ZoneDomain, functions) -> ZoneDomain:
        if env.is_bot():
            return env
        cls = stmt.__class__.__name__
        if cls == 'LetDecl':
            return self._interpret_let(stmt, env, functions)
        elif cls == 'Assign':
            return self._interpret_assign(stmt, env, functions)
        elif cls == 'IfStmt':
            return self._interpret_if(stmt, env, functions)
        elif cls == 'WhileStmt':
            return self._interpret_while(stmt, env, functions)
        elif cls == 'Block':
            return self._interpret_block(stmt, env, functions)
        return env

    def _interpret_block(self, block, env, functions):
        stmts = block.stmts if hasattr(block, 'stmts') else block
        if isinstance(stmts, list):
            for s in stmts:
                env = self._interpret_stmt(s, env, functions)
        return env

    def _interpret_let(self, stmt, env, functions):
        name = stmt.name
        if name not in env._var_index:
            env.add_var(name)
        self._apply_zone_assignment(name, stmt.value, env)
        return env

    def _interpret_assign(self, stmt, env, functions):
        name = stmt.name
        if name not in env._var_index:
            env.add_var(name)
        self._apply_zone_assignment(name, stmt.value, env)
        return env

    def _apply_zone_assignment(self, target, expr, env: ZoneDomain):
        cls = expr.__class__.__name__
        if cls == 'IntLit':
            env.assign_const(target, float(expr.value))
        elif cls == 'Var':
            if expr.name in env._var_index:
                env.assign_var(target, expr.name)
            else:
                env.forget(target)
        elif cls == 'BinOp':
            op = expr.op
            left_cls = expr.left.__class__.__name__
            right_cls = expr.right.__class__.__name__
            if op == '+':
                if left_cls == 'Var' and right_cls == 'IntLit':
                    if expr.left.name in env._var_index:
                        env.assign_add(target, expr.left.name, float(expr.right.value))
                        return
                elif left_cls == 'IntLit' and right_cls == 'Var':
                    if expr.right.name in env._var_index:
                        env.assign_add(target, expr.right.name, float(expr.left.value))
                        return
                elif left_cls == 'Var' and right_cls == 'Var':
                    if expr.left.name in env._var_index and expr.right.name in env._var_index:
                        env.assign_add_vars(target, expr.left.name, expr.right.name)
                        return
            elif op == '-':
                if left_cls == 'Var' and right_cls == 'IntLit':
                    if expr.left.name in env._var_index:
                        env.assign_add(target, expr.left.name, -float(expr.right.value))
                        return
                elif left_cls == 'Var' and right_cls == 'Var':
                    if expr.left.name in env._var_index and expr.right.name in env._var_index:
                        env.assign_sub_vars(target, expr.left.name, expr.right.name)
                        return
            env.forget(target)
        else:
            env.forget(target)

    def _refine_condition(self, cond, env: ZoneDomain):
        then_env = env.copy()
        else_env = env.copy()
        cls = cond.__class__.__name__
        if cls == 'BinOp':
            op = cond.op
            left_cls = cond.left.__class__.__name__
            right_cls = cond.right.__class__.__name__

            if left_cls == 'Var' and right_cls == 'IntLit':
                v, c = cond.left.name, float(cond.right.value)
                if v in env._var_index:
                    if op == '<':
                        then_env.set_upper(v, c - 1)
                        else_env.set_lower(v, c)
                    elif op == '<=':
                        then_env.set_upper(v, c)
                        else_env.set_lower(v, c + 1)
                    elif op == '>':
                        then_env.set_lower(v, c + 1)
                        else_env.set_upper(v, c)
                    elif op == '>=':
                        then_env.set_lower(v, c)
                        else_env.set_upper(v, c - 1)
                    elif op == '==':
                        then_env.set_lower(v, c)
                        then_env.set_upper(v, c)

            elif left_cls == 'Var' and right_cls == 'Var':
                x, y = cond.left.name, cond.right.name
                if x in env._var_index and y in env._var_index:
                    if op == '<':
                        then_env.set_diff(x, y, -1)
                        else_env.set_diff(y, x, 0)
                    elif op == '<=':
                        then_env.set_diff(x, y, 0)
                        else_env.set_diff(y, x, -1)
                    elif op == '>':
                        then_env.set_diff(y, x, -1)
                        else_env.set_diff(x, y, 0)
                    elif op == '>=':
                        then_env.set_diff(y, x, 0)
                        else_env.set_diff(x, y, -1)
                    elif op == '==':
                        then_env.set_diff(x, y, 0)
                        then_env.set_diff(y, x, 0)

        return then_env, else_env

    def _interpret_if(self, stmt, env, functions):
        then_env, else_env = self._refine_condition(stmt.cond, env)
        then_env = self._interpret_block(stmt.then_body, then_env, functions)
        if stmt.else_body:
            else_env = self._interpret_block(stmt.else_body, else_env, functions)
        if then_env.is_bot():
            return else_env
        if else_env.is_bot():
            return then_env
        return then_env.join(else_env)

    def _interpret_while(self, stmt, env, functions):
        current = env.copy()
        for _ in range(self.max_iterations):
            then_env, _ = self._refine_condition(stmt.cond, current)
            body_env = self._interpret_block(stmt.body, then_env, functions)
            next_env = current.widen(body_env.join(current))
            if next_env.equals(current):
                break
            current = next_env
        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env
