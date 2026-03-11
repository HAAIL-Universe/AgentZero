"""
V084: Symbolic Regex
====================

Regex to SFA compilation and regex equivalence checking via SFA operations.

Composes V081 (Symbolic Automata) to provide:
- Regex parser: full regex syntax (concat, alternation, Kleene star, plus,
  optional, char classes, ranges, dot, anchors, escape sequences)
- Regex to NFA compilation (Thompson construction adapted for symbolic transitions)
- Regex to DFA compilation (via NFA + determinization)
- Regex equivalence checking via SFA difference emptiness
- Regex inclusion checking (L(r1) subset L(r2))
- Regex intersection checking (L(r1) & L(r2) non-empty)
- Witness generation (find string in difference/intersection)
- Regex simplification / minimization (via DFA minimization)
- Derivative-based matching (Brzozowski derivatives on regex AST)

This enables formal reasoning about regex patterns: are two regexes equivalent?
Does one subsume another? What strings match one but not the other?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))

from symbolic_automata import (
    SFA, SFATransition, CharAlgebra, IntAlgebra,
    Pred, PredKind, PTrue, PFalse, PChar, PRange, PAnd, POr, PNot,
    sfa_intersection, sfa_union, sfa_complement, sfa_difference,
    sfa_is_equivalent, sfa_is_subset, sfa_concat, sfa_star, sfa_plus,
    sfa_epsilon, sfa_empty, sfa_any_char, sfa_from_string,
    sfa_from_char_class, sfa_from_range, sfa_stats, compare_sfas,
)
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
from enum import Enum


# ============================================================================
# Regex AST
# ============================================================================

class RegexKind(Enum):
    LITERAL = "literal"       # single character
    DOT = "dot"               # any character
    CHAR_CLASS = "class"      # character class [abc] or [a-z]
    NEG_CLASS = "neg_class"   # negated class [^abc]
    CONCAT = "concat"         # sequence
    ALT = "alt"               # alternation (|)
    STAR = "star"             # Kleene star (*)
    PLUS = "plus"             # one or more (+)
    OPTIONAL = "optional"     # zero or one (?)
    EPSILON = "epsilon"       # empty string
    EMPTY = "empty"           # empty language (matches nothing)


@dataclass(frozen=True)
class Regex:
    kind: RegexKind
    char: Optional[str] = None            # for LITERAL
    children: Tuple['Regex', ...] = ()    # for CONCAT, ALT
    child: Optional['Regex'] = None       # for STAR, PLUS, OPTIONAL
    ranges: Tuple[Tuple[str, str], ...] = ()  # for CHAR_CLASS, NEG_CLASS

    def __repr__(self):
        if self.kind == RegexKind.LITERAL:
            return repr(self.char)
        if self.kind == RegexKind.DOT:
            return "."
        if self.kind == RegexKind.CHAR_CLASS:
            parts = []
            for lo, hi in self.ranges:
                if lo == hi:
                    parts.append(lo)
                else:
                    parts.append(f"{lo}-{hi}")
            return f"[{''.join(parts)}]"
        if self.kind == RegexKind.NEG_CLASS:
            parts = []
            for lo, hi in self.ranges:
                if lo == hi:
                    parts.append(lo)
                else:
                    parts.append(f"{lo}-{hi}")
            return f"[^{''.join(parts)}]"
        if self.kind == RegexKind.CONCAT:
            return "".join(repr(c) for c in self.children)
        if self.kind == RegexKind.ALT:
            return "(" + "|".join(repr(c) for c in self.children) + ")"
        if self.kind == RegexKind.STAR:
            return f"({self.child})*"
        if self.kind == RegexKind.PLUS:
            return f"({self.child})+"
        if self.kind == RegexKind.OPTIONAL:
            return f"({self.child})?"
        if self.kind == RegexKind.EPSILON:
            return "eps"
        if self.kind == RegexKind.EMPTY:
            return "empty"
        return f"Regex({self.kind})"


# Regex constructors
def RLit(c):
    return Regex(RegexKind.LITERAL, char=c)

def RDot():
    return Regex(RegexKind.DOT)

def RClass(ranges):
    """ranges: list of (lo, hi) tuples."""
    return Regex(RegexKind.CHAR_CLASS, ranges=tuple(ranges))

def RNegClass(ranges):
    return Regex(RegexKind.NEG_CLASS, ranges=tuple(ranges))

def RConcat(*children):
    if len(children) == 0:
        return REpsilon()
    if len(children) == 1:
        return children[0]
    # Flatten nested concats
    flat = []
    for c in children:
        if c.kind == RegexKind.CONCAT:
            flat.extend(c.children)
        else:
            flat.append(c)
    return Regex(RegexKind.CONCAT, children=tuple(flat))

def RAlt(*children):
    if len(children) == 0:
        return REmpty()
    if len(children) == 1:
        return children[0]
    # Flatten nested alts
    flat = []
    for c in children:
        if c.kind == RegexKind.ALT:
            flat.extend(c.children)
        else:
            flat.append(c)
    return Regex(RegexKind.ALT, children=tuple(flat))

def RStar(child):
    if child.kind == RegexKind.EPSILON or child.kind == RegexKind.EMPTY:
        return REpsilon()
    if child.kind == RegexKind.STAR:
        return child  # a** = a*
    return Regex(RegexKind.STAR, child=child)

def RPlus(child):
    if child.kind == RegexKind.EPSILON:
        return REpsilon()
    if child.kind == RegexKind.EMPTY:
        return REmpty()
    return Regex(RegexKind.PLUS, child=child)

def ROptional(child):
    if child.kind == RegexKind.EPSILON:
        return REpsilon()
    return Regex(RegexKind.OPTIONAL, child=child)

def REpsilon():
    return Regex(RegexKind.EPSILON)

def REmpty():
    return Regex(RegexKind.EMPTY)


# ============================================================================
# Regex Parser
# ============================================================================

class RegexParseError(Exception):
    pass


class RegexParser:
    """
    Parse regex string into Regex AST.

    Supported syntax:
    - Literals: a, b, c, ...
    - Escape sequences: \\n, \\t, \\., \\*, \\+, \\?, \\(, \\), \\[, \\], \\|, \\\\
    - Shorthand classes: \\d (digit), \\w (word), \\s (space), \\D, \\W, \\S (negated)
    - Dot: . (any character except newline by default)
    - Character classes: [abc], [a-z], [^abc], [a-zA-Z0-9]
    - Alternation: a|b
    - Concatenation: ab
    - Kleene star: a*
    - Plus: a+
    - Optional: a?
    - Grouping: (ab)
    - Empty string: (empty regex or explicit epsilon)
    """

    ESCAPE_MAP = {
        'n': '\n', 't': '\t', 'r': '\r',
        '.': '.', '*': '*', '+': '+', '?': '?',
        '(': '(', ')': ')', '[': '[', ']': ']',
        '|': '|', '\\': '\\', '{': '{', '}': '}',
        '^': '^', '$': '$',
    }

    def __init__(self, pattern):
        self.pattern = pattern
        self.pos = 0

    def parse(self):
        if not self.pattern:
            return REpsilon()
        result = self._parse_alt()
        if self.pos < len(self.pattern):
            raise RegexParseError(
                f"Unexpected character at position {self.pos}: {self.pattern[self.pos]!r}")
        return result

    def _peek(self):
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def _advance(self):
        c = self.pattern[self.pos]
        self.pos += 1
        return c

    def _parse_alt(self):
        branches = [self._parse_concat()]
        while self._peek() == '|':
            self._advance()  # consume '|'
            branches.append(self._parse_concat())
        if len(branches) == 1:
            return branches[0]
        return RAlt(*branches)

    def _parse_concat(self):
        parts = []
        while self._peek() is not None and self._peek() not in (')', '|'):
            parts.append(self._parse_quantifier())
        if not parts:
            return REpsilon()
        if len(parts) == 1:
            return parts[0]
        return RConcat(*parts)

    def _parse_quantifier(self):
        base = self._parse_atom()
        while self._peek() in ('*', '+', '?'):
            q = self._advance()
            if q == '*':
                base = RStar(base)
            elif q == '+':
                base = RPlus(base)
            elif q == '?':
                base = ROptional(base)
        return base

    def _parse_atom(self):
        c = self._peek()
        if c == '(':
            self._advance()  # consume '('
            if self._peek() == ')':
                self._advance()
                return REpsilon()
            inner = self._parse_alt()
            if self._peek() != ')':
                raise RegexParseError(f"Expected ')' at position {self.pos}")
            self._advance()  # consume ')'
            return inner
        elif c == '[':
            return self._parse_char_class()
        elif c == '.':
            self._advance()
            return RDot()
        elif c == '\\':
            return self._parse_escape()
        elif c in ('*', '+', '?', ')', '|'):
            raise RegexParseError(f"Unexpected {c!r} at position {self.pos}")
        else:
            self._advance()
            return RLit(c)

    def _parse_escape(self):
        self._advance()  # consume '\\'
        if self.pos >= len(self.pattern):
            raise RegexParseError("Unexpected end of pattern after '\\'")
        c = self._advance()
        if c == 'd':
            return RClass([('0', '9')])
        if c == 'D':
            return RNegClass([('0', '9')])
        if c == 'w':
            return RClass([('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')])
        if c == 'W':
            return RNegClass([('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')])
        if c == 's':
            return RClass([(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')])
        if c == 'S':
            return RNegClass([(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')])
        if c in self.ESCAPE_MAP:
            return RLit(self.ESCAPE_MAP[c])
        # Unknown escape -- treat as literal
        return RLit(c)

    def _parse_char_class(self):
        self._advance()  # consume '['
        negated = False
        if self._peek() == '^':
            negated = True
            self._advance()
        ranges = []
        # Handle ] as first char in class
        if self._peek() == ']':
            ranges.append((']', ']'))
            self._advance()
        while self._peek() is not None and self._peek() != ']':
            lo = self._parse_class_char()
            if self._peek() == '-' and self.pos + 1 < len(self.pattern) and self.pattern[self.pos + 1] != ']':
                self._advance()  # consume '-'
                hi = self._parse_class_char()
                ranges.append((lo, hi))
            else:
                ranges.append((lo, lo))
        if self._peek() != ']':
            raise RegexParseError(f"Expected ']' at position {self.pos}")
        self._advance()  # consume ']'
        if negated:
            return RNegClass(ranges)
        return RClass(ranges)

    def _parse_class_char(self):
        if self._peek() == '\\':
            self._advance()
            if self.pos >= len(self.pattern):
                raise RegexParseError("Unexpected end in character class escape")
            c = self._advance()
            return self.ESCAPE_MAP.get(c, c)
        c = self._advance()
        return c


def parse_regex(pattern):
    """Parse a regex pattern string into a Regex AST."""
    return RegexParser(pattern).parse()


# ============================================================================
# Regex to SFA Compilation (Thompson-style)
# ============================================================================

class RegexCompiler:
    """Compile Regex AST to SFA using V081's SFA combinators."""

    def __init__(self, algebra=None):
        self.algebra = algebra or CharAlgebra()

    def compile(self, regex):
        """Compile regex to SFA (possibly nondeterministic)."""
        return self._compile(regex)

    def _compile(self, regex):
        """Returns an SFA for the given regex."""
        kind = regex.kind

        if kind == RegexKind.EPSILON:
            return sfa_epsilon(self.algebra)

        if kind == RegexKind.EMPTY:
            return sfa_empty(self.algebra)

        if kind == RegexKind.LITERAL:
            return sfa_from_string(regex.char, self.algebra)

        if kind == RegexKind.DOT:
            return sfa_any_char(self.algebra)

        if kind == RegexKind.CHAR_CLASS:
            pred = self._ranges_to_pred(regex.ranges)
            return SFA(
                states={0, 1},
                initial=0,
                accepting={1},
                transitions=[SFATransition(0, pred, 1)],
                algebra=self.algebra
            )

        if kind == RegexKind.NEG_CLASS:
            pred = PNot(self._ranges_to_pred(regex.ranges))
            return SFA(
                states={0, 1},
                initial=0,
                accepting={1},
                transitions=[SFATransition(0, pred, 1)],
                algebra=self.algebra
            )

        if kind == RegexKind.CONCAT:
            if not regex.children:
                return sfa_epsilon(self.algebra)
            result = self._compile(regex.children[0])
            for child in regex.children[1:]:
                result = sfa_concat(result, self._compile(child))
            return result

        if kind == RegexKind.ALT:
            if not regex.children:
                return sfa_empty(self.algebra)
            result = self._compile(regex.children[0])
            for child in regex.children[1:]:
                result = sfa_union(result, self._compile(child))
            return result

        if kind == RegexKind.STAR:
            return sfa_star(self._compile(regex.child))

        if kind == RegexKind.PLUS:
            return sfa_plus(self._compile(regex.child))

        if kind == RegexKind.OPTIONAL:
            inner = self._compile(regex.child)
            return sfa_union(sfa_epsilon(self.algebra), inner)

        raise ValueError(f"Unknown regex kind: {kind}")

    def _ranges_to_pred(self, ranges):
        """Convert list of (lo, hi) ranges to a predicate."""
        pred = PFalse()
        for lo, hi in ranges:
            if lo == hi:
                pred = POr(pred, PChar(lo))
            else:
                pred = POr(pred, PRange(lo, hi))
        return pred


def compile_regex(pattern, algebra=None):
    """Parse and compile a regex pattern to an SFA (NFA)."""
    if algebra is None:
        algebra = CharAlgebra()
    ast = parse_regex(pattern)
    compiler = RegexCompiler(algebra)
    return compiler.compile(ast)


def compile_regex_dfa(pattern, algebra=None):
    """Parse and compile a regex pattern to a deterministic SFA."""
    nfa = compile_regex(pattern, algebra)
    return nfa.determinize()


def compile_regex_min(pattern, algebra=None):
    """Parse and compile a regex pattern to a minimal DFA."""
    nfa = compile_regex(pattern, algebra)
    return nfa.determinize().minimize()


# ============================================================================
# Regex Equivalence and Comparison
# ============================================================================

def regex_equivalent(pattern1, pattern2, algebra=None):
    """Check if two regex patterns are equivalent (accept same language)."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    return sfa_is_equivalent(sfa1, sfa2)


def regex_subset(pattern1, pattern2, algebra=None):
    """Check if L(pattern1) is a subset of L(pattern2)."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    return sfa_is_subset(sfa1, sfa2)


def regex_intersects(pattern1, pattern2, algebra=None):
    """Check if L(pattern1) and L(pattern2) have non-empty intersection."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    inter = sfa_intersection(sfa1, sfa2)
    return not inter.is_empty()


def regex_difference_witness(pattern1, pattern2, algebra=None):
    """Find a string in L(pattern1) but not in L(pattern2), or None."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    diff = sfa_difference(sfa1, sfa2)
    return diff.accepted_word()


def regex_intersection_witness(pattern1, pattern2, algebra=None):
    """Find a string in both L(pattern1) and L(pattern2), or None."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    inter = sfa_intersection(sfa1, sfa2)
    return inter.accepted_word()


def regex_is_empty(pattern, algebra=None):
    """Check if a regex matches no strings."""
    sfa = compile_regex(pattern, algebra)
    return sfa.is_empty()


def regex_accepts_epsilon(pattern, algebra=None):
    """Check if a regex matches the empty string."""
    sfa = compile_regex(pattern, algebra)
    # Check if initial state is reachable to an accepting state with 0 transitions
    return sfa.initial in sfa.accepting


def regex_sample(pattern, algebra=None):
    """Find a shortest string matching the regex, or None if empty."""
    sfa = compile_regex(pattern, algebra)
    return sfa.accepted_word()


def regex_compare(pattern1, pattern2, algebra=None):
    """
    Full comparison of two regexes. Returns dict with:
    - equivalent: bool
    - subset_1_2: L(p1) subset L(p2)?
    - subset_2_1: L(p2) subset L(p1)?
    - witness_in_1_not_2: string in L(p1) - L(p2) if any
    - witness_in_2_not_1: string in L(p2) - L(p1) if any
    - witness_intersection: string in L(p1) & L(p2) if any
    - sfa1_stats, sfa2_stats: size info
    """
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()

    diff12 = sfa_difference(sfa1, sfa2)
    diff21 = sfa_difference(sfa2, sfa1)
    inter = sfa_intersection(sfa1, sfa2)

    w12 = diff12.accepted_word()
    w21 = diff21.accepted_word()
    wi = inter.accepted_word()

    return {
        'equivalent': diff12.is_empty() and diff21.is_empty(),
        'subset_1_2': diff12.is_empty(),
        'subset_2_1': diff21.is_empty(),
        'witness_in_1_not_2': w12,
        'witness_in_2_not_1': w21,
        'witness_intersection': wi,
        'sfa1_stats': sfa_stats(sfa1),
        'sfa2_stats': sfa_stats(sfa2),
    }


# ============================================================================
# Brzozowski Derivatives (direct regex matching without SFA)
# ============================================================================

def nullable(regex):
    """Check if a regex matches the empty string (structural, no SFA needed)."""
    kind = regex.kind
    if kind == RegexKind.EPSILON:
        return True
    if kind == RegexKind.EMPTY:
        return False
    if kind == RegexKind.LITERAL:
        return False
    if kind == RegexKind.DOT:
        return False
    if kind == RegexKind.CHAR_CLASS:
        return False
    if kind == RegexKind.NEG_CLASS:
        return False
    if kind == RegexKind.CONCAT:
        return all(nullable(c) for c in regex.children)
    if kind == RegexKind.ALT:
        return any(nullable(c) for c in regex.children)
    if kind == RegexKind.STAR:
        return True
    if kind == RegexKind.PLUS:
        return nullable(regex.child)
    if kind == RegexKind.OPTIONAL:
        return True
    return False


def derivative(regex, char, algebra=None):
    """
    Compute Brzozowski derivative of regex with respect to a character.
    D_c(r) = { w | c.w in L(r) }
    """
    if algebra is None:
        algebra = CharAlgebra()
    kind = regex.kind

    if kind == RegexKind.EPSILON:
        return REmpty()
    if kind == RegexKind.EMPTY:
        return REmpty()
    if kind == RegexKind.LITERAL:
        return REpsilon() if regex.char == char else REmpty()
    if kind == RegexKind.DOT:
        return REpsilon()  # dot matches any single char
    if kind == RegexKind.CHAR_CLASS:
        pred = _ranges_to_pred(regex.ranges)
        return REpsilon() if algebra.evaluate(pred, char) else REmpty()
    if kind == RegexKind.NEG_CLASS:
        pred = PNot(_ranges_to_pred(regex.ranges))
        return REpsilon() if algebra.evaluate(pred, char) else REmpty()
    if kind == RegexKind.ALT:
        derivs = [derivative(c, char, algebra) for c in regex.children]
        return _simplify_alt(derivs)
    if kind == RegexKind.CONCAT:
        # D_c(r1.r2...rn) = D_c(r1).r2...rn | (if nullable(r1)) D_c(r2...rn)
        children = regex.children
        if not children:
            return REmpty()
        first_deriv = derivative(children[0], char, algebra)
        rest = RConcat(*children[1:]) if len(children) > 1 else REpsilon()
        result = RConcat(first_deriv, rest) if rest.kind != RegexKind.EPSILON else first_deriv
        if nullable(children[0]):
            rest_deriv = derivative(RConcat(*children[1:]) if len(children) > 1 else REpsilon(),
                                     char, algebra)
            result = RAlt(result, rest_deriv)
        return result
    if kind == RegexKind.STAR:
        # D_c(r*) = D_c(r) . r*
        inner_deriv = derivative(regex.child, char, algebra)
        return RConcat(inner_deriv, regex)
    if kind == RegexKind.PLUS:
        # D_c(r+) = D_c(r) . r*
        inner_deriv = derivative(regex.child, char, algebra)
        return RConcat(inner_deriv, RStar(regex.child))
    if kind == RegexKind.OPTIONAL:
        # D_c(r?) = D_c(r)
        return derivative(regex.child, char, algebra)

    return REmpty()


def _ranges_to_pred(ranges):
    pred = PFalse()
    for lo, hi in ranges:
        if lo == hi:
            pred = POr(pred, PChar(lo))
        else:
            pred = POr(pred, PRange(lo, hi))
    return pred


def _simplify_alt(derivs):
    """Simplify alternation of derivatives."""
    non_empty = [d for d in derivs if d.kind != RegexKind.EMPTY]
    if not non_empty:
        return REmpty()
    if len(non_empty) == 1:
        return non_empty[0]
    return RAlt(*non_empty)


def derivative_match(regex, word, algebra=None):
    """Match a word against a regex using Brzozowski derivatives."""
    if algebra is None:
        algebra = CharAlgebra()
    if isinstance(regex, str):
        regex = parse_regex(regex)
    current = regex
    for c in word:
        current = derivative(current, c, algebra)
    return nullable(current)


# ============================================================================
# Regex Operations via SFA
# ============================================================================

def regex_union(pattern1, pattern2, algebra=None):
    """Compute union of two regexes, return resulting SFA."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    return sfa_union(sfa1, sfa2)


def regex_intersection(pattern1, pattern2, algebra=None):
    """Compute intersection of two regexes, return resulting SFA."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    return sfa_intersection(sfa1, sfa2)


def regex_complement(pattern, algebra=None):
    """Compute complement of a regex, return resulting SFA."""
    sfa = compile_regex(pattern, algebra).determinize()
    return sfa_complement(sfa)


def regex_difference(pattern1, pattern2, algebra=None):
    """Compute difference L(p1) - L(p2), return resulting SFA."""
    sfa1 = compile_regex(pattern1, algebra).determinize()
    sfa2 = compile_regex(pattern2, algebra).determinize()
    return sfa_difference(sfa1, sfa2)


# ============================================================================
# Regex AST Utilities
# ============================================================================

def regex_to_string(regex):
    """Convert regex AST back to a pattern string (best-effort)."""
    kind = regex.kind
    if kind == RegexKind.LITERAL:
        c = regex.char
        if c in r'\.+*?()[]|{}^$':
            return '\\' + c
        if c == '\n':
            return '\\n'
        if c == '\t':
            return '\\t'
        if c == '\r':
            return '\\r'
        return c
    if kind == RegexKind.DOT:
        return '.'
    if kind == RegexKind.EPSILON:
        return ''
    if kind == RegexKind.EMPTY:
        return '(?!.)'  # never-match pattern
    if kind == RegexKind.CHAR_CLASS:
        return _ranges_to_class_str(regex.ranges, negated=False)
    if kind == RegexKind.NEG_CLASS:
        return _ranges_to_class_str(regex.ranges, negated=True)
    if kind == RegexKind.CONCAT:
        return ''.join(_wrap_if_needed(c, 'concat') for c in regex.children)
    if kind == RegexKind.ALT:
        return '|'.join(_wrap_if_needed(c, 'alt') for c in regex.children)
    if kind == RegexKind.STAR:
        return _wrap_if_needed(regex.child, 'quant') + '*'
    if kind == RegexKind.PLUS:
        return _wrap_if_needed(regex.child, 'quant') + '+'
    if kind == RegexKind.OPTIONAL:
        return _wrap_if_needed(regex.child, 'quant') + '?'
    return str(regex)


def _ranges_to_class_str(ranges, negated=False):
    parts = []
    for lo, hi in ranges:
        if lo == hi:
            parts.append(lo)
        else:
            parts.append(f"{lo}-{hi}")
    prefix = '^' if negated else ''
    return f"[{prefix}{''.join(parts)}]"


def _wrap_if_needed(regex, context):
    """Wrap regex in parens if needed for precedence."""
    if context == 'quant' and regex.kind in (RegexKind.CONCAT, RegexKind.ALT):
        return '(' + regex_to_string(regex) + ')'
    if context == 'concat' and regex.kind == RegexKind.ALT:
        return '(' + regex_to_string(regex) + ')'
    return regex_to_string(regex)


def regex_size(regex):
    """Count the number of AST nodes in a regex."""
    if regex.kind in (RegexKind.LITERAL, RegexKind.DOT, RegexKind.EPSILON,
                       RegexKind.EMPTY, RegexKind.CHAR_CLASS, RegexKind.NEG_CLASS):
        return 1
    if regex.kind in (RegexKind.STAR, RegexKind.PLUS, RegexKind.OPTIONAL):
        return 1 + regex_size(regex.child)
    if regex.kind in (RegexKind.CONCAT, RegexKind.ALT):
        return 1 + sum(regex_size(c) for c in regex.children)
    return 1
