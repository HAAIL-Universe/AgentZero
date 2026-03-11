"""V087: Abstract Interpretation over Strings

Composes V081 (Symbolic Automata) + V086 (String Constraints) to provide
abstract domains for string-valued program analysis.

Domains:
  - LengthDomain: interval [lo, hi] over string length
  - PrefixDomain: known constant prefix
  - SuffixDomain: known constant suffix
  - CharSetDomain: per-position character set + overall alphabet constraint
  - SFADomain: full symbolic automaton tracking (most precise)
  - StringProduct: reduced product of all domains with cross-domain tightening

The interpreter analyzes simple string programs (assignments, concat, slice,
conditionals, loops) and computes abstract string states at each point.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V086_string_constraints'))

from symbolic_automata import (
    SFA, CharAlgebra, sfa_from_string, sfa_from_char_class, sfa_from_range,
    sfa_any_char, sfa_epsilon, sfa_empty, sfa_concat, sfa_star, sfa_plus,
    sfa_optional, sfa_intersection, sfa_union, sfa_complement, sfa_difference,
    sfa_is_equivalent, sfa_is_subset,
)

from string_constraints import (
    StringConstraintSolver, regex_constraint, equals_const,
    length_eq, length_range, length_le, length_ge,
    contains, prefix, suffix, StringConstraint,
)

from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List, Tuple, Any
from enum import Enum
from copy import deepcopy
import re

# ============================================================
# Abstract Domain Protocol
# ============================================================

INF = float('inf')

class DomainKind(Enum):
    LENGTH = "length"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CHARSET = "charset"
    SFA = "sfa"
    PRODUCT = "product"


# ============================================================
# Length Domain: [lo, hi] interval over string length
# ============================================================

@dataclass(frozen=True)
class LengthDomain:
    """Interval abstraction of string length: lo <= len(s) <= hi."""
    lo: int = 0
    hi: float = INF  # use INF for unbounded

    kind = DomainKind.LENGTH

    @staticmethod
    def top():
        return LengthDomain(0, INF)

    @staticmethod
    def bot():
        return LengthDomain(1, 0)  # lo > hi => empty

    @staticmethod
    def exact(n: int):
        return LengthDomain(n, n)

    @staticmethod
    def at_least(n: int):
        return LengthDomain(n, INF)

    @staticmethod
    def at_most(n: int):
        return LengthDomain(0, n)

    @staticmethod
    def from_string(s: str):
        n = len(s)
        return LengthDomain(n, n)

    def is_bot(self) -> bool:
        return self.lo > self.hi

    def is_top(self) -> bool:
        return self.lo == 0 and self.hi == INF

    def contains(self, n: int) -> bool:
        return self.lo <= n <= self.hi

    def join(self, other: 'LengthDomain') -> 'LengthDomain':
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        return LengthDomain(min(self.lo, other.lo), max(self.hi, other.hi))

    def meet(self, other: 'LengthDomain') -> 'LengthDomain':
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        return LengthDomain(lo, hi)

    def widen(self, other: 'LengthDomain') -> 'LengthDomain':
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        lo = self.lo if other.lo >= self.lo else 0
        hi = self.hi if other.hi <= self.hi else INF
        return LengthDomain(lo, hi)

    def concat(self, other: 'LengthDomain') -> 'LengthDomain':
        """Length of concatenation."""
        if self.is_bot() or other.is_bot():
            return LengthDomain.bot()
        lo = self.lo + other.lo
        hi = self.hi + other.hi if self.hi != INF and other.hi != INF else INF
        return LengthDomain(lo, hi)

    def slice(self, start: Optional[int], end: Optional[int]) -> 'LengthDomain':
        """Length after slicing s[start:end]."""
        if self.is_bot():
            return LengthDomain.bot()
        s = start if start is not None else 0
        if end is not None:
            max_slice = max(0, end - s)
            # If we know the exact source length, compute exact slice length
            if self.lo == self.hi and self.hi != INF:
                actual_end = min(end, int(self.hi))
                actual_start = min(s, int(self.hi))
                exact = max(0, actual_end - actual_start)
                return LengthDomain(exact, exact)
            return LengthDomain(0, min(max_slice, self.hi if self.hi != INF else max_slice))
        else:
            # s[start:] -- result length depends on source length
            if self.lo == self.hi and self.hi != INF:
                exact = max(0, int(self.hi) - s)
                return LengthDomain(exact, exact)
            lo_result = max(0, int(self.lo) - s) if self.lo >= s else 0
            hi_result = self.hi - s if self.hi != INF else INF
            hi_result = max(0, hi_result) if hi_result != INF else INF
            return LengthDomain(lo_result, hi_result)

    def __le__(self, other: 'LengthDomain') -> bool:
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        return other.lo <= self.lo and self.hi <= other.hi

    def __repr__(self):
        if self.is_bot():
            return "Len(BOT)"
        if self.is_top():
            return "Len(TOP)"
        hi_s = str(self.hi) if self.hi != INF else "inf"
        if self.lo == self.hi:
            return f"Len({self.lo})"
        return f"Len([{self.lo},{hi_s}])"


# ============================================================
# Prefix Domain: known constant prefix
# ============================================================

@dataclass(frozen=True)
class PrefixDomain:
    """Tracks a known constant prefix of a string.
    prefix=None means BOT, prefix="" means TOP (no info)."""
    prefix: Optional[str] = ""

    kind = DomainKind.PREFIX

    @staticmethod
    def top():
        return PrefixDomain("")

    @staticmethod
    def bot():
        return PrefixDomain(None)

    @staticmethod
    def from_string(s: str):
        return PrefixDomain(s)

    def is_bot(self) -> bool:
        return self.prefix is None

    def is_top(self) -> bool:
        return self.prefix == ""

    def contains(self, s: str) -> bool:
        if self.is_bot():
            return False
        return s.startswith(self.prefix)

    def join(self, other: 'PrefixDomain') -> 'PrefixDomain':
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        # Longest common prefix
        lcp = _longest_common_prefix(self.prefix, other.prefix)
        return PrefixDomain(lcp)

    def meet(self, other: 'PrefixDomain') -> 'PrefixDomain':
        if self.is_bot() or other.is_bot():
            return PrefixDomain.bot()
        # One must be a prefix of the other, else BOT
        if self.prefix.startswith(other.prefix):
            return self
        if other.prefix.startswith(self.prefix):
            return other
        return PrefixDomain.bot()

    def widen(self, other: 'PrefixDomain') -> 'PrefixDomain':
        # Same as join for prefix domain (finite height via string shortening)
        return self.join(other)

    def concat(self, other: 'PrefixDomain', other_len: 'LengthDomain' = None) -> 'PrefixDomain':
        """Prefix of concatenation. If we know full string (exact length), extend prefix."""
        if self.is_bot() or other.is_bot():
            return PrefixDomain.bot()
        # If self has exact known length matching prefix, extend with other's prefix
        if other_len and not other_len.is_bot():
            # We only know the prefix of self, so result prefix is just self.prefix
            # unless self is fully known (length domain exact = len(prefix))
            pass
        return PrefixDomain(self.prefix)

    def __le__(self, other: 'PrefixDomain') -> bool:
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        return self.prefix.startswith(other.prefix)

    def __repr__(self):
        if self.is_bot():
            return "Prefix(BOT)"
        if self.is_top():
            return "Prefix(TOP)"
        return f"Prefix({self.prefix!r})"


def _longest_common_prefix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


# ============================================================
# Suffix Domain: known constant suffix
# ============================================================

@dataclass(frozen=True)
class SuffixDomain:
    """Tracks a known constant suffix of a string.
    suffix=None means BOT, suffix="" means TOP."""
    suffix: Optional[str] = ""

    kind = DomainKind.SUFFIX

    @staticmethod
    def top():
        return SuffixDomain("")

    @staticmethod
    def bot():
        return SuffixDomain(None)

    @staticmethod
    def from_string(s: str):
        return SuffixDomain(s)

    def is_bot(self) -> bool:
        return self.suffix is None

    def is_top(self) -> bool:
        return self.suffix == ""

    def contains(self, s: str) -> bool:
        if self.is_bot():
            return False
        return s.endswith(self.suffix)

    def join(self, other: 'SuffixDomain') -> 'SuffixDomain':
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        lcs = _longest_common_suffix(self.suffix, other.suffix)
        return SuffixDomain(lcs)

    def meet(self, other: 'SuffixDomain') -> 'SuffixDomain':
        if self.is_bot() or other.is_bot():
            return SuffixDomain.bot()
        if self.suffix.endswith(other.suffix):
            return self
        if other.suffix.endswith(self.suffix):
            return other
        return SuffixDomain.bot()

    def widen(self, other: 'SuffixDomain') -> 'SuffixDomain':
        return self.join(other)

    def concat(self, other: 'SuffixDomain') -> 'SuffixDomain':
        """Suffix of concatenation is suffix of the second operand."""
        if self.is_bot() or other.is_bot():
            return SuffixDomain.bot()
        return other

    def __le__(self, other: 'SuffixDomain') -> bool:
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        return self.suffix.endswith(other.suffix)

    def __repr__(self):
        if self.is_bot():
            return "Suffix(BOT)"
        if self.is_top():
            return "Suffix(TOP)"
        return f"Suffix({self.suffix!r})"


def _longest_common_suffix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[-(i+1)] == b[-(i+1)]:
        i += 1
    return a[len(a)-i:] if i > 0 else ""


# ============================================================
# Character Set Domain: per-position + overall char sets
# ============================================================

@dataclass
class CharSetDomain:
    """Per-position character sets plus overall alphabet constraint.
    chars[i] = set of possible chars at position i.
    alphabet = set of all possible chars anywhere in the string.
    If chars is None -> BOT. If alphabet is None -> TOP (all chars)."""
    chars: Optional[List[Set[str]]] = None  # per-position; None = BOT
    alphabet: Optional[Set[str]] = None  # overall; None = all chars
    _is_bot: bool = False

    kind = DomainKind.CHARSET

    @staticmethod
    def top():
        d = CharSetDomain()
        d.chars = []
        d.alphabet = None
        return d

    @staticmethod
    def bot():
        d = CharSetDomain()
        d._is_bot = True
        return d

    @staticmethod
    def from_string(s: str):
        d = CharSetDomain()
        d.chars = [{c} for c in s]
        d.alphabet = set(s) if s else None
        return d

    @staticmethod
    def from_alphabet(alphabet: Set[str]):
        d = CharSetDomain()
        d.chars = []
        d.alphabet = set(alphabet)
        return d

    def is_bot(self) -> bool:
        return self._is_bot

    def is_top(self) -> bool:
        return not self._is_bot and len(self.chars or []) == 0 and self.alphabet is None

    def contains(self, s: str) -> bool:
        if self._is_bot:
            return False
        if self.chars:
            if len(s) != len(self.chars):
                return False  # only if we have exact position info
            for i, c in enumerate(s):
                if c not in self.chars[i]:
                    return False
        if self.alphabet:
            for c in s:
                if c not in self.alphabet:
                    return False
        return True

    def join(self, other: 'CharSetDomain') -> 'CharSetDomain':
        if self.is_bot():
            return deepcopy(other)
        if other.is_bot():
            return deepcopy(self)
        result = CharSetDomain()
        # Per-position: union where both have info, drop where lengths differ
        min_len = min(len(self.chars), len(other.chars))
        if min_len > 0 and len(self.chars) == len(other.chars):
            result.chars = [self.chars[i] | other.chars[i] for i in range(min_len)]
        else:
            result.chars = []  # different lengths -> lose position info
        # Alphabet: union
        if self.alphabet is None or other.alphabet is None:
            result.alphabet = None
        else:
            result.alphabet = self.alphabet | other.alphabet
        return result

    def meet(self, other: 'CharSetDomain') -> 'CharSetDomain':
        if self.is_bot() or other.is_bot():
            return CharSetDomain.bot()
        result = CharSetDomain()
        if self.chars and other.chars and len(self.chars) == len(other.chars):
            result.chars = [self.chars[i] & other.chars[i] for i in range(len(self.chars))]
            if any(len(s) == 0 for s in result.chars):
                return CharSetDomain.bot()
        elif self.chars:
            result.chars = list(self.chars)
        elif other.chars:
            result.chars = list(other.chars)
        else:
            result.chars = []
        if self.alphabet is not None and other.alphabet is not None:
            result.alphabet = self.alphabet & other.alphabet
            if len(result.alphabet) == 0:
                return CharSetDomain.bot()
        elif self.alphabet is not None:
            result.alphabet = set(self.alphabet)
        elif other.alphabet is not None:
            result.alphabet = set(other.alphabet)
        else:
            result.alphabet = None
        return result

    def widen(self, other: 'CharSetDomain') -> 'CharSetDomain':
        return self.join(other)

    def concat(self, other: 'CharSetDomain') -> 'CharSetDomain':
        if self.is_bot() or other.is_bot():
            return CharSetDomain.bot()
        result = CharSetDomain()
        # Concatenate per-position info only if both have non-empty position info
        # (empty chars=[] means no position info, i.e. TOP or unknown length)
        if (self.chars is not None and other.chars is not None and
            len(self.chars) > 0 and len(other.chars) > 0):
            result.chars = list(self.chars) + list(other.chars)
        else:
            result.chars = []
        # Alphabet union
        if self.alphabet is None or other.alphabet is None:
            result.alphabet = None
        else:
            result.alphabet = self.alphabet | other.alphabet
        return result

    def __repr__(self):
        if self._is_bot:
            return "CharSet(BOT)"
        if self.is_top():
            return "CharSet(TOP)"
        parts = []
        if self.chars:
            pos_strs = []
            for i, cs in enumerate(self.chars):
                if len(cs) == 1:
                    pos_strs.append(f"{next(iter(cs))}")
                else:
                    pos_strs.append(f"{''.join(sorted(cs))}")
            parts.append(f"pos=[{','.join(pos_strs)}]")
        if self.alphabet:
            parts.append(f"alpha={''.join(sorted(self.alphabet))}")
        return f"CharSet({'; '.join(parts)})"


# ============================================================
# SFA Domain: symbolic automaton tracking
# ============================================================

@dataclass
class SFADomain:
    """Full symbolic automaton abstraction -- most precise string domain.
    Uses V081 SFA operations for join (union), meet (intersection), etc."""
    sfa: Optional[SFA] = None  # None means BOT
    _is_top: bool = False

    kind = DomainKind.SFA

    @staticmethod
    def top():
        d = SFADomain()
        d._is_top = True
        # Sigma* -- accept all strings
        algebra = CharAlgebra()
        d.sfa = sfa_star(sfa_any_char(algebra))
        return d

    @staticmethod
    def bot():
        return SFADomain(sfa=None)

    @staticmethod
    def from_string(s: str):
        algebra = CharAlgebra()
        d = SFADomain()
        d.sfa = sfa_from_string(s, algebra)
        return d

    @staticmethod
    def from_sfa(sfa: SFA):
        d = SFADomain()
        d.sfa = sfa
        return d

    @staticmethod
    def from_regex(pattern: str, alphabet: CharAlgebra = None):
        """Build SFA domain from regex pattern using V084 if available."""
        if alphabet is None:
            alphabet = CharAlgebra()
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V084_symbolic_regex'))
            from symbolic_regex import compile_regex
            d = SFADomain()
            d.sfa = compile_regex(pattern, alphabet)
            return d
        except ImportError:
            # Fallback: treat as exact string
            return SFADomain.from_string(pattern)

    def is_bot(self) -> bool:
        return self.sfa is None

    def is_top(self) -> bool:
        return self._is_top

    def contains(self, s: str) -> bool:
        if self.sfa is None:
            return False
        return self.sfa.accepts(list(s))

    def join(self, other: 'SFADomain') -> 'SFADomain':
        if self.is_bot():
            return deepcopy(other)
        if other.is_bot():
            return deepcopy(self)
        if self._is_top or other._is_top:
            return SFADomain.top()
        result = SFADomain()
        result.sfa = sfa_union(self.sfa, other.sfa)
        return result

    def meet(self, other: 'SFADomain') -> 'SFADomain':
        if self.is_bot() or other.is_bot():
            return SFADomain.bot()
        if self._is_top:
            return deepcopy(other)
        if other._is_top:
            return deepcopy(self)
        result = SFADomain()
        result.sfa = sfa_intersection(self.sfa, other.sfa)
        if result.sfa.is_empty():
            return SFADomain.bot()
        return result

    def widen(self, other: 'SFADomain') -> 'SFADomain':
        # SFA widening: just go to TOP if state count exceeds threshold
        joined = self.join(other)
        if joined.sfa and joined.sfa.count_states() > 100:
            return SFADomain.top()
        return joined

    def concat(self, other: 'SFADomain') -> 'SFADomain':
        if self.is_bot() or other.is_bot():
            return SFADomain.bot()
        result = SFADomain()
        result.sfa = sfa_concat(self.sfa, other.sfa)
        return result

    def accepted_word(self) -> Optional[str]:
        if self.sfa is None:
            return None
        w = self.sfa.accepted_word()
        if w is None:
            return None
        return ''.join(w)

    def language_size(self, max_length: int = 10) -> int:
        """Count accepted strings up to max_length (for finite approximation)."""
        if self.sfa is None:
            return 0
        count = 0
        # BFS enumeration up to max_length
        from collections import deque
        algebra = self.sfa.algebra
        det = self.sfa.determinize()
        queue = deque()
        queue.append((det.initial, []))
        while queue:
            state, word = queue.popleft()
            if len(word) <= max_length and state in det.accepting:
                count += 1
            if len(word) >= max_length:
                continue
            for (src, pred, tgt) in det.transitions:
                if src == state:
                    wit = algebra.witness(pred)
                    if wit is not None:
                        queue.append((tgt, word + [wit]))
        return count

    def __repr__(self):
        if self.is_bot():
            return "SFA(BOT)"
        if self._is_top:
            return "SFA(TOP)"
        states = self.sfa.count_states() if self.sfa else 0
        return f"SFA({states} states)"


# ============================================================
# String Product Domain: reduced product of all string domains
# ============================================================

@dataclass
class StringProduct:
    """Reduced product of Length x Prefix x Suffix x CharSet x SFA domains.
    Cross-domain reduction tightens each domain using info from the others."""
    length: LengthDomain = None
    prefix: PrefixDomain = None
    suffix: SuffixDomain = None
    charset: CharSetDomain = None
    sfa_dom: SFADomain = None

    kind = DomainKind.PRODUCT

    def __post_init__(self):
        if self.length is None:
            self.length = LengthDomain.top()
        if self.prefix is None:
            self.prefix = PrefixDomain.top()
        if self.suffix is None:
            self.suffix = SuffixDomain.top()
        if self.charset is None:
            self.charset = CharSetDomain.top()
        if self.sfa_dom is None:
            self.sfa_dom = None  # SFA is optional (expensive)

    @staticmethod
    def top(use_sfa: bool = False):
        p = StringProduct(
            length=LengthDomain.top(),
            prefix=PrefixDomain.top(),
            suffix=SuffixDomain.top(),
            charset=CharSetDomain.top(),
        )
        if use_sfa:
            p.sfa_dom = SFADomain.top()
        return p

    @staticmethod
    def bot():
        return StringProduct(
            length=LengthDomain.bot(),
            prefix=PrefixDomain.bot(),
            suffix=SuffixDomain.bot(),
            charset=CharSetDomain.bot(),
            sfa_dom=SFADomain.bot(),
        )

    @staticmethod
    def from_string(s: str, use_sfa: bool = False):
        p = StringProduct(
            length=LengthDomain.from_string(s),
            prefix=PrefixDomain.from_string(s),
            suffix=SuffixDomain.from_string(s),
            charset=CharSetDomain.from_string(s),
        )
        if use_sfa:
            p.sfa_dom = SFADomain.from_string(s)
        p._reduce()
        return p

    def is_bot(self) -> bool:
        return (self.length.is_bot() or self.prefix.is_bot() or
                self.suffix.is_bot() or self.charset.is_bot() or
                (self.sfa_dom is not None and self.sfa_dom.is_bot()))

    def is_top(self) -> bool:
        return (self.length.is_top() and self.prefix.is_top() and
                self.suffix.is_top() and self.charset.is_top() and
                (self.sfa_dom is None or self.sfa_dom.is_top()))

    def join(self, other: 'StringProduct') -> 'StringProduct':
        result = StringProduct(
            length=self.length.join(other.length),
            prefix=self.prefix.join(other.prefix),
            suffix=self.suffix.join(other.suffix),
            charset=self.charset.join(other.charset),
        )
        if self.sfa_dom is not None and other.sfa_dom is not None:
            result.sfa_dom = self.sfa_dom.join(other.sfa_dom)
        result._reduce()
        return result

    def meet(self, other: 'StringProduct') -> 'StringProduct':
        result = StringProduct(
            length=self.length.meet(other.length),
            prefix=self.prefix.meet(other.prefix),
            suffix=self.suffix.meet(other.suffix),
            charset=self.charset.meet(other.charset),
        )
        if self.sfa_dom is not None and other.sfa_dom is not None:
            result.sfa_dom = self.sfa_dom.meet(other.sfa_dom)
        result._reduce()
        return result

    def widen(self, other: 'StringProduct') -> 'StringProduct':
        result = StringProduct(
            length=self.length.widen(other.length),
            prefix=self.prefix.widen(other.prefix),
            suffix=self.suffix.widen(other.suffix),
            charset=self.charset.widen(other.charset),
        )
        if self.sfa_dom is not None and other.sfa_dom is not None:
            result.sfa_dom = self.sfa_dom.widen(other.sfa_dom)
        return result

    def concat(self, other: 'StringProduct') -> 'StringProduct':
        result = StringProduct(
            length=self.length.concat(other.length),
            prefix=self.prefix.concat(other.prefix, other.length),
            suffix=self.suffix.concat(other.suffix),
            charset=self.charset.concat(other.charset),
        )
        if self.sfa_dom is not None and other.sfa_dom is not None:
            result.sfa_dom = self.sfa_dom.concat(other.sfa_dom)
        # If self is exact (prefix==suffix==known, length exact), extend prefix
        if (not self.prefix.is_bot() and not self.prefix.is_top() and
            self.length.lo == self.length.hi and
            self.length.lo == len(self.prefix.prefix) and
            not other.prefix.is_bot()):
            new_prefix = self.prefix.prefix + other.prefix.prefix
            result.prefix = PrefixDomain(new_prefix)
        result._reduce()
        return result

    def _reduce(self):
        """Cross-domain reduction: tighten each domain using info from others."""
        if self.is_bot():
            return

        # 1. Prefix/suffix length tightens length lower bound
        min_len = 0
        if not self.prefix.is_bot() and not self.prefix.is_top():
            min_len = max(min_len, len(self.prefix.prefix))
        if not self.suffix.is_bot() and not self.suffix.is_top():
            min_len = max(min_len, len(self.suffix.suffix))
        if min_len > 0:
            self.length = self.length.meet(LengthDomain.at_least(min_len))

        # 2. CharSet position count tightens length
        if (not self.charset.is_bot() and self.charset.chars and
            len(self.charset.chars) > 0):
            n = len(self.charset.chars)
            self.length = self.length.meet(LengthDomain.exact(n))

        # 3. Length upper bound can trim prefix/suffix knowledge
        # (prefix can't be longer than max possible string)
        if (not self.length.is_bot() and self.length.hi != INF and
            not self.prefix.is_bot() and not self.prefix.is_top()):
            max_len = int(self.length.hi)
            if len(self.prefix.prefix) > max_len:
                self.prefix = PrefixDomain.bot()  # inconsistent -> BOT

        # 4. Exact length + full prefix/suffix -> check consistency
        if (self.length.lo == self.length.hi and self.length.lo != INF and
            not self.prefix.is_bot() and not self.suffix.is_bot() and
            not self.prefix.is_top() and not self.suffix.is_top()):
            n = int(self.length.lo)
            if len(self.prefix.prefix) + len(self.suffix.suffix) > n:
                # Check overlap consistency
                overlap = len(self.prefix.prefix) + len(self.suffix.suffix) - n
                if overlap > 0:
                    prefix_tail = self.prefix.prefix[-overlap:]
                    suffix_head = self.suffix.suffix[:overlap]
                    if prefix_tail != suffix_head:
                        # Inconsistent -> BOT
                        self.length = LengthDomain.bot()

        # 5. Prefix chars tighten charset positions
        if (not self.prefix.is_bot() and not self.prefix.is_top() and
            not self.charset.is_bot() and self.charset.chars):
            for i, c in enumerate(self.prefix.prefix):
                if i < len(self.charset.chars):
                    self.charset.chars[i] = self.charset.chars[i] & {c}
                    if len(self.charset.chars[i]) == 0:
                        self.charset = CharSetDomain.bot()
                        return

    def contains(self, s: str) -> bool:
        """Check if concrete string is in this abstract element."""
        return (self.length.contains(len(s)) and
                self.prefix.contains(s) and
                self.suffix.contains(s) and
                self.charset.contains(s) and
                (self.sfa_dom is None or self.sfa_dom.contains(s)))

    def __repr__(self):
        if self.is_bot():
            return "StringProduct(BOT)"
        parts = []
        if not self.length.is_top():
            parts.append(str(self.length))
        if not self.prefix.is_top():
            parts.append(str(self.prefix))
        if not self.suffix.is_top():
            parts.append(str(self.suffix))
        if not self.charset.is_top():
            parts.append(str(self.charset))
        if self.sfa_dom and not self.sfa_dom.is_top():
            parts.append(str(self.sfa_dom))
        if not parts:
            return "StringProduct(TOP)"
        return f"StringProduct({', '.join(parts)})"


# ============================================================
# String Abstract Environment
# ============================================================

class StringEnv:
    """Maps variable names to StringProduct abstract values."""

    def __init__(self, use_sfa: bool = False):
        self.bindings: Dict[str, StringProduct] = {}
        self.use_sfa = use_sfa

    def get(self, name: str) -> StringProduct:
        return self.bindings.get(name, StringProduct.top(self.use_sfa))

    def set(self, name: str, val: StringProduct):
        self.bindings[name] = val

    def copy(self) -> 'StringEnv':
        env = StringEnv(self.use_sfa)
        env.bindings = {k: deepcopy(v) for k, v in self.bindings.items()}
        return env

    def join(self, other: 'StringEnv') -> 'StringEnv':
        result = StringEnv(self.use_sfa)
        all_vars = set(self.bindings) | set(other.bindings)
        for v in all_vars:
            a = self.get(v)
            b = other.get(v)
            result.bindings[v] = a.join(b)
        return result

    def widen(self, other: 'StringEnv') -> 'StringEnv':
        result = StringEnv(self.use_sfa)
        all_vars = set(self.bindings) | set(other.bindings)
        for v in all_vars:
            a = self.get(v)
            b = other.get(v)
            result.bindings[v] = a.widen(b)
        return result

    def __eq__(self, other):
        if not isinstance(other, StringEnv):
            return False
        all_vars = set(self.bindings) | set(other.bindings)
        for v in all_vars:
            a = self.get(v)
            b = other.get(v)
            # Approximate equality check
            if repr(a) != repr(b):
                return False
        return True

    def __repr__(self):
        items = ', '.join(f'{k}: {v}' for k, v in sorted(self.bindings.items()))
        return f"StringEnv({{{items}}})"


# ============================================================
# String Program AST (simple imperative string language)
# ============================================================

class SStmt:
    pass

@dataclass
class SAssign(SStmt):
    """x = expr"""
    var: str
    expr: 'SExpr'

@dataclass
class SConcat(SStmt):
    """x = y + z (string concat)"""
    var: str
    left: str
    right: str

@dataclass
class SSlice(SStmt):
    """x = y[start:end]"""
    var: str
    source: str
    start: Optional[int] = None
    end: Optional[int] = None

@dataclass
class SIf(SStmt):
    """if cond then_body else_body"""
    cond: 'SCond'
    then_body: List[SStmt]
    else_body: List[SStmt] = field(default_factory=list)

@dataclass
class SWhile(SStmt):
    """while cond body"""
    cond: 'SCond'
    body: List[SStmt]

@dataclass
class SAssert(SStmt):
    """assert cond"""
    cond: 'SCond'

class SExpr:
    pass

@dataclass
class SConst(SExpr):
    """String constant"""
    value: str

@dataclass
class SVar(SExpr):
    """Variable reference"""
    name: str

@dataclass
class SConcatExpr(SExpr):
    """Concatenation expression"""
    left: SExpr
    right: SExpr

class SCond:
    pass

@dataclass
class SLenEq(SCond):
    """len(x) == n"""
    var: str
    n: int

@dataclass
class SLenLt(SCond):
    """len(x) < n"""
    var: str
    n: int

@dataclass
class SLenGt(SCond):
    """len(x) > n"""
    var: str
    n: int

@dataclass
class SStartsWith(SCond):
    """x.startswith(prefix)"""
    var: str
    prefix: str

@dataclass
class SEndsWith(SCond):
    """x.endswith(suffix)"""
    var: str
    suffix: str

@dataclass
class SEquals(SCond):
    """x == s"""
    var: str
    value: str

@dataclass
class SNotEquals(SCond):
    """x != s"""
    var: str
    value: str

@dataclass
class SContains(SCond):
    """substr in x"""
    var: str
    substr: str

@dataclass
class SIsEmpty(SCond):
    """x == "" """
    var: str

@dataclass
class SNot(SCond):
    """not cond"""
    inner: SCond


# ============================================================
# String Abstract Interpreter
# ============================================================

class StringInterpreter:
    """Abstract interpreter for string programs using StringProduct domain."""

    def __init__(self, use_sfa: bool = False, max_iter: int = 20):
        self.use_sfa = use_sfa
        self.max_iter = max_iter
        self.warnings: List[str] = []
        self.assertions: List[Tuple[str, bool]] = []  # (desc, holds?)

    def analyze(self, stmts: List[SStmt], init_env: StringEnv = None) -> StringEnv:
        """Analyze a list of statements starting from init_env."""
        if init_env is None:
            init_env = StringEnv(self.use_sfa)
        self.warnings = []
        self.assertions = []
        return self._analyze_block(stmts, init_env)

    def _analyze_block(self, stmts: List[SStmt], env: StringEnv) -> StringEnv:
        for stmt in stmts:
            env = self._analyze_stmt(stmt, env)
        return env

    def _analyze_stmt(self, stmt: SStmt, env: StringEnv) -> StringEnv:
        if isinstance(stmt, SAssign):
            val = self._eval_expr(stmt.expr, env)
            env = env.copy()
            env.set(stmt.var, val)
            return env

        elif isinstance(stmt, SConcat):
            left = env.get(stmt.left)
            right = env.get(stmt.right)
            result = left.concat(right)
            env = env.copy()
            env.set(stmt.var, result)
            return env

        elif isinstance(stmt, SSlice):
            source = env.get(stmt.source)
            result = self._abstract_slice(source, stmt.start, stmt.end)
            env = env.copy()
            env.set(stmt.var, result)
            return env

        elif isinstance(stmt, SIf):
            then_env = self._refine_env(env, stmt.cond, True)
            else_env = self._refine_env(env, stmt.cond, False)
            then_result = self._analyze_block(stmt.then_body, then_env)
            else_result = self._analyze_block(stmt.else_body, else_env) if stmt.else_body else else_env
            return then_result.join(else_result)

        elif isinstance(stmt, SWhile):
            # Fixed-point iteration with widening
            current = env
            for i in range(self.max_iter):
                body_env = self._refine_env(current, stmt.cond, True)
                after_body = self._analyze_block(stmt.body, body_env)
                next_env = current.widen(after_body)
                if next_env == current:
                    break
                current = next_env
            # Exit condition
            return self._refine_env(current, stmt.cond, False)

        elif isinstance(stmt, SAssert):
            holds = self._check_condition(stmt.cond, env)
            desc = repr(stmt.cond)
            self.assertions.append((desc, holds))
            if not holds:
                self.warnings.append(f"Assertion may fail: {desc}")
            return env

        return env

    def _eval_expr(self, expr: SExpr, env: StringEnv) -> StringProduct:
        if isinstance(expr, SConst):
            return StringProduct.from_string(expr.value, self.use_sfa)
        elif isinstance(expr, SVar):
            return env.get(expr.name)
        elif isinstance(expr, SConcatExpr):
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            return left.concat(right)
        return StringProduct.top(self.use_sfa)

    def _abstract_slice(self, source: StringProduct, start: Optional[int],
                        end: Optional[int]) -> StringProduct:
        """Compute abstract value of source[start:end]."""
        result = StringProduct()
        result.length = source.length.slice(start, end)

        # Prefix: if slicing from 0, prefix is preserved up to slice length
        if start is None or start == 0:
            if not source.prefix.is_bot() and not source.prefix.is_top():
                if end is not None:
                    result.prefix = PrefixDomain(source.prefix.prefix[:end])
                else:
                    result.prefix = source.prefix
            else:
                result.prefix = source.prefix
        else:
            result.prefix = PrefixDomain.top()

        # Suffix: if slicing to end, suffix is preserved
        if end is None:
            result.suffix = source.suffix
        else:
            result.suffix = SuffixDomain.top()

        result.charset = CharSetDomain.top()  # Conservative for slicing
        return result

    def _refine_env(self, env: StringEnv, cond: SCond, branch: bool) -> StringEnv:
        """Refine environment based on condition truth value."""
        env = env.copy()

        if isinstance(cond, SNot):
            return self._refine_env(env, cond.inner, not branch)

        if isinstance(cond, SLenEq):
            val = env.get(cond.var)
            if branch:
                constraint = LengthDomain.exact(cond.n)
            else:
                # len != n: could be < n or > n -> TOP (no useful refinement)
                return env
            refined = StringProduct(
                length=val.length.meet(constraint),
                prefix=val.prefix,
                suffix=val.suffix,
                charset=val.charset,
                sfa_dom=val.sfa_dom,
            )
            refined._reduce()
            env.set(cond.var, refined)
            return env

        elif isinstance(cond, SLenLt):
            val = env.get(cond.var)
            if branch:
                constraint = LengthDomain(0, cond.n - 1)
            else:
                constraint = LengthDomain.at_least(cond.n)
            refined = StringProduct(
                length=val.length.meet(constraint),
                prefix=val.prefix,
                suffix=val.suffix,
                charset=val.charset,
                sfa_dom=val.sfa_dom,
            )
            refined._reduce()
            env.set(cond.var, refined)
            return env

        elif isinstance(cond, SLenGt):
            val = env.get(cond.var)
            if branch:
                constraint = LengthDomain.at_least(cond.n + 1)
            else:
                constraint = LengthDomain(0, cond.n)
            refined = StringProduct(
                length=val.length.meet(constraint),
                prefix=val.prefix,
                suffix=val.suffix,
                charset=val.charset,
                sfa_dom=val.sfa_dom,
            )
            refined._reduce()
            env.set(cond.var, refined)
            return env

        elif isinstance(cond, SStartsWith):
            val = env.get(cond.var)
            if branch:
                new_prefix = val.prefix.meet(PrefixDomain(cond.prefix))
                new_len = val.length.meet(LengthDomain.at_least(len(cond.prefix)))
                refined = StringProduct(
                    length=new_len,
                    prefix=new_prefix,
                    suffix=val.suffix,
                    charset=val.charset,
                    sfa_dom=val.sfa_dom,
                )
                refined._reduce()
                env.set(cond.var, refined)
            return env

        elif isinstance(cond, SEndsWith):
            val = env.get(cond.var)
            if branch:
                new_suffix = val.suffix.meet(SuffixDomain(cond.suffix))
                new_len = val.length.meet(LengthDomain.at_least(len(cond.suffix)))
                refined = StringProduct(
                    length=new_len,
                    prefix=val.prefix,
                    suffix=new_suffix,
                    charset=val.charset,
                    sfa_dom=val.sfa_dom,
                )
                refined._reduce()
                env.set(cond.var, refined)
            return env

        elif isinstance(cond, SEquals):
            val = env.get(cond.var)
            if branch:
                exact = StringProduct.from_string(cond.value, self.use_sfa)
                refined = val.meet(exact)
                env.set(cond.var, refined)
            return env

        elif isinstance(cond, SNotEquals):
            # Not very useful for refinement in general
            return env

        elif isinstance(cond, SContains):
            val = env.get(cond.var)
            if branch:
                new_len = val.length.meet(LengthDomain.at_least(len(cond.substr)))
                refined = StringProduct(
                    length=new_len,
                    prefix=val.prefix,
                    suffix=val.suffix,
                    charset=val.charset,
                    sfa_dom=val.sfa_dom,
                )
                refined._reduce()
                env.set(cond.var, refined)
            return env

        elif isinstance(cond, SIsEmpty):
            val = env.get(cond.var)
            if branch:
                exact = StringProduct.from_string("", self.use_sfa)
                refined = val.meet(exact)
                env.set(cond.var, refined)
            else:
                new_len = val.length.meet(LengthDomain.at_least(1))
                refined = StringProduct(
                    length=new_len,
                    prefix=val.prefix,
                    suffix=val.suffix,
                    charset=val.charset,
                    sfa_dom=val.sfa_dom,
                )
                refined._reduce()
                env.set(cond.var, refined)
            return env

        return env

    def _check_condition(self, cond: SCond, env: StringEnv) -> bool:
        """Check if condition DEFINITELY holds (True) or MAY fail (False)."""
        if isinstance(cond, SNot):
            return not self._check_condition_may_hold(cond.inner, env)

        if isinstance(cond, SLenEq):
            val = env.get(cond.var)
            return val.length.lo == cond.n and val.length.hi == cond.n

        elif isinstance(cond, SLenLt):
            val = env.get(cond.var)
            return val.length.hi < cond.n

        elif isinstance(cond, SLenGt):
            val = env.get(cond.var)
            return val.length.lo > cond.n

        elif isinstance(cond, SStartsWith):
            val = env.get(cond.var)
            if val.prefix.is_bot():
                return False
            if val.prefix.is_top():
                return False
            return val.prefix.prefix.startswith(cond.prefix)

        elif isinstance(cond, SEndsWith):
            val = env.get(cond.var)
            if val.suffix.is_bot():
                return False
            if val.suffix.is_top():
                return False
            return val.suffix.suffix.endswith(cond.suffix)

        elif isinstance(cond, SEquals):
            val = env.get(cond.var)
            return (val.length.lo == val.length.hi == len(cond.value) and
                    val.prefix.prefix == cond.value and
                    val.suffix.suffix == cond.value)

        elif isinstance(cond, SIsEmpty):
            val = env.get(cond.var)
            return val.length.lo == 0 and val.length.hi == 0

        elif isinstance(cond, SContains):
            # Hard to prove definitely
            return False

        return False

    def _check_condition_may_hold(self, cond: SCond, env: StringEnv) -> bool:
        """Check if condition MAY hold (for NOT checking)."""
        if isinstance(cond, SLenEq):
            val = env.get(cond.var)
            return val.length.contains(cond.n)
        elif isinstance(cond, SIsEmpty):
            val = env.get(cond.var)
            return val.length.contains(0)
        return True  # conservatively may hold


# ============================================================
# High-Level Analysis APIs
# ============================================================

def analyze_string_program(stmts: List[SStmt], init_vars: Dict[str, str] = None,
                           use_sfa: bool = False) -> Dict[str, Any]:
    """Analyze a string program and return abstract state + warnings.

    Args:
        stmts: List of string program statements
        init_vars: Initial variable assignments (name -> concrete string)
        use_sfa: Whether to track full SFA domain (precise but expensive)

    Returns:
        dict with 'env' (final StringEnv), 'warnings', 'assertions'
    """
    interp = StringInterpreter(use_sfa=use_sfa)
    env = StringEnv(use_sfa)
    if init_vars:
        for name, value in init_vars.items():
            env.set(name, StringProduct.from_string(value, use_sfa))
    final_env = interp.analyze(stmts, env)
    return {
        'env': final_env,
        'warnings': interp.warnings,
        'assertions': interp.assertions,
    }


def get_variable_info(env: StringEnv, var: str) -> Dict[str, Any]:
    """Get detailed info about a variable's abstract string value."""
    val = env.get(var)
    info = {
        'is_bot': val.is_bot(),
        'is_top': val.is_top(),
        'length_lo': val.length.lo,
        'length_hi': val.length.hi,
        'known_prefix': val.prefix.prefix if not val.prefix.is_bot() else None,
        'known_suffix': val.suffix.suffix if not val.suffix.is_bot() else None,
    }
    if val.charset and not val.charset.is_bot() and val.charset.chars:
        info['char_positions'] = len(val.charset.chars)
        info['char_sets'] = [sorted(cs) for cs in val.charset.chars]
    if val.sfa_dom and not val.sfa_dom.is_bot():
        info['sfa_states'] = val.sfa_dom.sfa.count_states() if val.sfa_dom.sfa else 0
        w = val.sfa_dom.accepted_word()
        if w is not None:
            info['example_word'] = w
    return info


def compare_domains(stmts: List[SStmt], init_vars: Dict[str, str] = None) -> Dict[str, Any]:
    """Compare precision of analysis with and without SFA domain."""
    result_no_sfa = analyze_string_program(stmts, init_vars, use_sfa=False)
    result_sfa = analyze_string_program(stmts, init_vars, use_sfa=True)
    comparison = {
        'without_sfa': {},
        'with_sfa': {},
    }
    all_vars = set(result_no_sfa['env'].bindings) | set(result_sfa['env'].bindings)
    for var in sorted(all_vars):
        comparison['without_sfa'][var] = get_variable_info(result_no_sfa['env'], var)
        comparison['with_sfa'][var] = get_variable_info(result_sfa['env'], var)
    comparison['warnings_no_sfa'] = result_no_sfa['warnings']
    comparison['warnings_sfa'] = result_sfa['warnings']
    return comparison


def string_domain_from_constraints(var: str, constraints: list) -> StringProduct:
    """Build a StringProduct from V086 StringConstraint objects.
    Bridges the constraint solver with the abstract domain."""
    from string_constraints import ConstraintKind

    solver = StringConstraintSolver()
    for c in constraints:
        solver.add(c)
    result = solver.check()
    if result.result.name == 'UNSAT':
        return StringProduct.bot()

    product = StringProduct.top()

    # Extract info from constraints
    for c in constraints:
        kind = c.kind
        if kind == ConstraintKind.REGEX:
            try:
                sfa_d = SFADomain.from_regex(c.pattern)
                if product.sfa_dom is None:
                    product.sfa_dom = sfa_d
                else:
                    product.sfa_dom = product.sfa_dom.meet(sfa_d)
            except Exception:
                pass
        elif kind == ConstraintKind.EQUALS_CONST:
            return StringProduct.from_string(c.pattern)
        elif kind == ConstraintKind.LENGTH_EQ:
            product.length = product.length.meet(LengthDomain.exact(c.value))
        elif kind == ConstraintKind.LENGTH_RANGE:
            product.length = product.length.meet(LengthDomain(c.value, c.value2))
        elif kind == ConstraintKind.LENGTH_LE:
            product.length = product.length.meet(LengthDomain.at_most(c.value))
        elif kind == ConstraintKind.LENGTH_GE:
            product.length = product.length.meet(LengthDomain.at_least(c.value))
        elif kind == ConstraintKind.PREFIX:
            product.prefix = product.prefix.meet(PrefixDomain(c.pattern))
        elif kind == ConstraintKind.SUFFIX:
            product.suffix = product.suffix.meet(SuffixDomain(c.pattern))

    product._reduce()
    return product


def analyze_string_flow(sources: Dict[str, StringProduct],
                        operations: List[Tuple[str, str, List[str]]]) -> Dict[str, StringProduct]:
    """Analyze a sequence of string operations (data flow graph style).

    Args:
        sources: Initial string abstract values
        operations: List of (result_var, operation, [operand_vars])
            Operations: 'concat', 'slice_prefix', 'slice_suffix', 'upper', 'lower', 'reverse'

    Returns:
        Dict mapping all variables to their abstract values
    """
    env = dict(sources)
    for result_var, op, operands in operations:
        if op == 'concat' and len(operands) == 2:
            a = env.get(operands[0], StringProduct.top())
            b = env.get(operands[1], StringProduct.top())
            env[result_var] = a.concat(b)
        elif op == 'assign' and len(operands) == 1:
            env[result_var] = env.get(operands[0], StringProduct.top())
        elif op == 'const':
            env[result_var] = StringProduct.from_string(operands[0])
        elif op == 'slice_prefix' and len(operands) == 2:
            # Take first N chars
            src = env.get(operands[0], StringProduct.top())
            n = int(operands[1])
            sliced = StringProduct(
                length=LengthDomain(0, min(n, src.length.hi if src.length.hi != INF else n)),
                prefix=src.prefix,
                suffix=SuffixDomain.top(),
                charset=CharSetDomain.top(),
            )
            sliced._reduce()
            env[result_var] = sliced
        elif op == 'slice_suffix' and len(operands) == 2:
            # Take last N chars
            src = env.get(operands[0], StringProduct.top())
            n = int(operands[1])
            sliced = StringProduct(
                length=LengthDomain(0, min(n, src.length.hi if src.length.hi != INF else n)),
                prefix=PrefixDomain.top(),
                suffix=src.suffix,
                charset=CharSetDomain.top(),
            )
            sliced._reduce()
            env[result_var] = sliced
        else:
            # Unknown op -> TOP
            env[result_var] = StringProduct.top()
    return env


def check_string_property(env: StringEnv, var: str, prop: SCond) -> str:
    """Check if a string property holds, may hold, or is impossible.

    Returns: 'TRUE' (definitely holds), 'FALSE' (impossible), 'UNKNOWN' (may or may not)
    """
    interp = StringInterpreter()
    holds = interp._check_condition(prop, env)
    if holds:
        return 'TRUE'

    # Check if it's impossible
    refined = interp._refine_env(env, prop, True)
    val = refined.get(var)
    if val.is_bot():
        return 'FALSE'

    return 'UNKNOWN'
