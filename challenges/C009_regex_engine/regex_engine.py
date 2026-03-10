"""
Regex Engine -- Thompson's NFA Construction
============================================
Challenge C009 (difficulty 3)

Supports:
  - Concatenation (ab)
  - Alternation (a|b)
  - Kleene star (a*)
  - Plus (a+)
  - Optional (a?)
  - Grouping ((ab))
  - Character classes ([abc], [a-z], [^abc])
  - Dot (.) -- matches any character except newline
  - Anchors (^ start, $ end)
  - Escaping (\\., \\*, etc.)

Architecture:
  1. Lexer: regex string -> tokens
  2. Parser: tokens -> AST (recursive descent)
  3. Compiler: AST -> NFA (Thompson's construction)
  4. Matcher: NFA simulation (epsilon closure + step)
"""

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# AST Nodes
# ============================================================

class ASTNode:
    pass

@dataclass
class Literal(ASTNode):
    char: str

@dataclass
class Dot(ASTNode):
    pass

@dataclass
class CharClass(ASTNode):
    chars: set
    negated: bool = False

@dataclass
class Concat(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass
class Alternation(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass
class Star(ASTNode):
    child: ASTNode

@dataclass
class Plus(ASTNode):
    child: ASTNode

@dataclass
class Optional_(ASTNode):
    child: ASTNode

@dataclass
class AnchorStart(ASTNode):
    pass

@dataclass
class AnchorEnd(ASTNode):
    pass

@dataclass
class Empty(ASTNode):
    pass


# ============================================================
# Lexer
# ============================================================

TOKEN_LITERAL = 'LITERAL'
TOKEN_DOT = 'DOT'
TOKEN_STAR = 'STAR'
TOKEN_PLUS = 'PLUS'
TOKEN_QUESTION = 'QUESTION'
TOKEN_PIPE = 'PIPE'
TOKEN_LPAREN = 'LPAREN'
TOKEN_RPAREN = 'RPAREN'
TOKEN_LBRACKET = 'LBRACKET'
TOKEN_CARET = 'CARET'
TOKEN_DOLLAR = 'DOLLAR'

SPECIAL_CHARS = set('.*+?|()[]^$\\')

@dataclass
class Token:
    type: str
    value: str = ''


def lex(pattern: str) -> list:
    """Tokenize a regex pattern."""
    tokens = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == '\\':
            if i + 1 >= len(pattern):
                raise ValueError("Trailing backslash")
            i += 1
            next_c = pattern[i]
            # Shorthand classes
            if next_c == 'd':
                tokens.append(Token('CHARCLASS', '0123456789'))
            elif next_c == 'w':
                chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
                tokens.append(Token('CHARCLASS', chars))
            elif next_c == 's':
                tokens.append(Token('CHARCLASS', ' \t\n\r\f\v'))
            elif next_c == 'n':
                tokens.append(Token(TOKEN_LITERAL, '\n'))
            elif next_c == 't':
                tokens.append(Token(TOKEN_LITERAL, '\t'))
            else:
                tokens.append(Token(TOKEN_LITERAL, next_c))
        elif c == '.':
            tokens.append(Token(TOKEN_DOT))
        elif c == '*':
            tokens.append(Token(TOKEN_STAR))
        elif c == '+':
            tokens.append(Token(TOKEN_PLUS))
        elif c == '?':
            tokens.append(Token(TOKEN_QUESTION))
        elif c == '|':
            tokens.append(Token(TOKEN_PIPE))
        elif c == '(':
            tokens.append(Token(TOKEN_LPAREN))
        elif c == ')':
            tokens.append(Token(TOKEN_RPAREN))
        elif c == '[':
            # Parse character class inline
            i += 1
            negated = False
            if i < len(pattern) and pattern[i] == '^':
                negated = True
                i += 1
            chars = set()
            # Allow ] as first char in class
            if i < len(pattern) and pattern[i] == ']':
                chars.add(']')
                i += 1
            while i < len(pattern) and pattern[i] != ']':
                if pattern[i] == '\\' and i + 1 < len(pattern):
                    i += 1
                    chars.add(pattern[i])
                elif (i + 2 < len(pattern) and pattern[i + 1] == '-'
                      and pattern[i + 2] != ']'):
                    start = ord(pattern[i])
                    end = ord(pattern[i + 2])
                    for code in range(start, end + 1):
                        chars.add(chr(code))
                    i += 2
                else:
                    chars.add(pattern[i])
                i += 1
            if i >= len(pattern):
                raise ValueError("Unterminated character class")
            # Store as special token
            if negated:
                tokens.append(Token('NEGCHARCLASS', ''.join(sorted(chars))))
            else:
                tokens.append(Token('CHARCLASS', ''.join(sorted(chars))))
        elif c == '^':
            tokens.append(Token(TOKEN_CARET))
        elif c == '$':
            tokens.append(Token(TOKEN_DOLLAR))
        else:
            tokens.append(Token(TOKEN_LITERAL, c))
        i += 1
    return tokens


# ============================================================
# Parser (recursive descent)
# ============================================================

class Parser:
    """
    Grammar:
      regex     -> anchor_start? alternation anchor_end?
      alternation -> concat ('|' concat)*
      concat    -> quantified+
      quantified -> atom ('*' | '+' | '?')?
      atom      -> literal | dot | charclass | '(' alternation ')'
    """

    def __init__(self, tokens: list):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> ASTNode:
        node = self.regex()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self.tokens[self.pos]}")
        return node

    def regex(self) -> ASTNode:
        parts = []

        # Check for ^ anchor
        if self.peek() and self.peek().type == TOKEN_CARET:
            self.consume()
            parts.append(AnchorStart())

        # Main body
        body = self.alternation()
        if body is not None:
            parts.append(body)

        # Check for $ anchor
        if self.peek() and self.peek().type == TOKEN_DOLLAR:
            self.consume()
            parts.append(AnchorEnd())

        if not parts:
            return Empty()
        result = parts[0]
        for p in parts[1:]:
            result = Concat(result, p)
        return result

    def alternation(self) -> Optional[ASTNode]:
        left = self.concat()
        while self.peek() and self.peek().type == TOKEN_PIPE:
            self.consume()
            right = self.concat()
            if left is None:
                left = Empty()
            if right is None:
                right = Empty()
            left = Alternation(left, right)
        return left

    def concat(self) -> Optional[ASTNode]:
        nodes = []
        while self.peek() and self.peek().type not in (TOKEN_PIPE, TOKEN_RPAREN, TOKEN_DOLLAR):
            node = self.quantified()
            if node is not None:
                nodes.append(node)
        if not nodes:
            return None
        result = nodes[0]
        for n in nodes[1:]:
            result = Concat(result, n)
        return result

    def quantified(self) -> Optional[ASTNode]:
        node = self.atom()
        if node is None:
            return None
        tok = self.peek()
        if tok:
            if tok.type == TOKEN_STAR:
                self.consume()
                return Star(node)
            elif tok.type == TOKEN_PLUS:
                self.consume()
                return Plus(node)
            elif tok.type == TOKEN_QUESTION:
                self.consume()
                return Optional_(node)
        return node

    def atom(self) -> Optional[ASTNode]:
        tok = self.peek()
        if tok is None:
            return None

        if tok.type == TOKEN_LITERAL:
            self.consume()
            return Literal(tok.value)
        elif tok.type == TOKEN_DOT:
            self.consume()
            return Dot()
        elif tok.type == 'CHARCLASS':
            self.consume()
            return CharClass(set(tok.value))
        elif tok.type == 'NEGCHARCLASS':
            self.consume()
            return CharClass(set(tok.value), negated=True)
        elif tok.type == TOKEN_LPAREN:
            self.consume()
            inner = self.alternation()
            if inner is None:
                inner = Empty()
            if not self.peek() or self.peek().type != TOKEN_RPAREN:
                raise ValueError("Unmatched parenthesis")
            self.consume()
            return inner
        elif tok.type == TOKEN_CARET:
            # Caret inside expression (not at start) -- treat as literal
            self.consume()
            return Literal('^')
        else:
            return None


# ============================================================
# NFA
# ============================================================

@dataclass
class NFAState:
    id: int
    transitions: dict = field(default_factory=dict)  # char -> [NFAState]
    epsilon: list = field(default_factory=list)       # [NFAState]
    is_accept: bool = False
    anchor_start: bool = False
    anchor_end: bool = False

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, NFAState) and self.id == other.id


class NFAFragment:
    """A fragment of NFA with one start state and one dangling end state."""
    def __init__(self, start: NFAState, end: NFAState):
        self.start = start
        self.end = end


class Compiler:
    """Compile AST to NFA using Thompson's construction."""

    def __init__(self):
        self._id = 0

    def new_state(self) -> NFAState:
        s = NFAState(id=self._id)
        self._id += 1
        return s

    def compile(self, ast: ASTNode) -> NFAFragment:
        return self._compile(ast)

    def _compile(self, node: ASTNode) -> NFAFragment:
        if isinstance(node, Literal):
            return self._literal(node.char)
        elif isinstance(node, Dot):
            return self._dot()
        elif isinstance(node, CharClass):
            return self._charclass(node.chars, node.negated)
        elif isinstance(node, Concat):
            return self._concat(node.left, node.right)
        elif isinstance(node, Alternation):
            return self._alternation(node.left, node.right)
        elif isinstance(node, Star):
            return self._star(node.child)
        elif isinstance(node, Plus):
            return self._plus(node.child)
        elif isinstance(node, Optional_):
            return self._optional(node.child)
        elif isinstance(node, AnchorStart):
            return self._anchor_start()
        elif isinstance(node, AnchorEnd):
            return self._anchor_end()
        elif isinstance(node, Empty):
            return self._empty()
        else:
            raise ValueError(f"Unknown AST node: {type(node)}")

    def _literal(self, char: str) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        start.transitions[char] = [end]
        return NFAFragment(start, end)

    def _dot(self) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        start.transitions[('dot',)] = [end]  # Unique sentinel for wildcard dot
        return NFAFragment(start, end)

    def _charclass(self, chars: set, negated: bool) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        if negated:
            start.transitions[('neg', frozenset(chars))] = [end]
        else:
            start.transitions[('pos', frozenset(chars))] = [end]
        return NFAFragment(start, end)

    def _concat(self, left: ASTNode, right: ASTNode) -> NFAFragment:
        frag_l = self._compile(left)
        frag_r = self._compile(right)
        frag_l.end.epsilon.append(frag_r.start)
        return NFAFragment(frag_l.start, frag_r.end)

    def _alternation(self, left: ASTNode, right: ASTNode) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        frag_l = self._compile(left)
        frag_r = self._compile(right)
        start.epsilon.append(frag_l.start)
        start.epsilon.append(frag_r.start)
        frag_l.end.epsilon.append(end)
        frag_r.end.epsilon.append(end)
        return NFAFragment(start, end)

    def _star(self, child: ASTNode) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        frag = self._compile(child)
        start.epsilon.append(frag.start)
        start.epsilon.append(end)
        frag.end.epsilon.append(frag.start)
        frag.end.epsilon.append(end)
        return NFAFragment(start, end)

    def _plus(self, child: ASTNode) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        frag = self._compile(child)
        start.epsilon.append(frag.start)
        frag.end.epsilon.append(frag.start)
        frag.end.epsilon.append(end)
        return NFAFragment(start, end)

    def _optional(self, child: ASTNode) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        frag = self._compile(child)
        start.epsilon.append(frag.start)
        start.epsilon.append(end)
        frag.end.epsilon.append(end)
        return NFAFragment(start, end)

    def _anchor_start(self) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        start.anchor_start = True
        start.epsilon.append(end)
        return NFAFragment(start, end)

    def _anchor_end(self) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        start.anchor_end = True
        start.epsilon.append(end)
        return NFAFragment(start, end)

    def _empty(self) -> NFAFragment:
        start = self.new_state()
        end = self.new_state()
        start.epsilon.append(end)
        return NFAFragment(start, end)


# ============================================================
# Matcher (NFA simulation)
# ============================================================

def _epsilon_closure(states: set, input_char: str = None, pos: int = 0,
                     text_len: int = 0) -> set:
    """Compute epsilon closure, respecting anchors."""
    stack = list(states)
    closure = set()
    while stack:
        s = stack.pop()
        if s.id in closure:
            continue
        # Check anchor constraints
        if s.anchor_start and pos != 0:
            continue
        if s.anchor_end and pos != text_len:
            continue
        closure.add(s.id)
        for next_s in s.epsilon:
            if next_s.id not in closure:
                stack.append(next_s)
    return closure


def _get_state_map(fragment: NFAFragment) -> dict:
    """Build id -> state mapping."""
    mapping = {}
    stack = [fragment.start]
    while stack:
        s = stack.pop()
        if s.id in mapping:
            continue
        mapping[s.id] = s
        for targets in s.transitions.values():
            for t in targets:
                if t.id not in mapping:
                    stack.append(t)
        for e in s.epsilon:
            if e.id not in mapping:
                stack.append(e)
    return mapping


def _step(state_map: dict, current_ids: set, char: str) -> set:
    """Advance NFA by one character."""
    next_states = set()
    for sid in current_ids:
        state = state_map[sid]
        for key, targets in state.transitions.items():
            match = False
            if key == ('dot',):
                # Dot matches any char except newline
                if char != '\n':
                    match = True
            elif isinstance(key, tuple):
                kind, chars = key
                if kind == 'pos' and char in chars:
                    match = True
                elif kind == 'neg' and char not in chars and char != '\n':
                    match = True
            elif key == char:
                match = True
            if match:
                for t in targets:
                    next_states.add(t.id)
    return next_states


class Regex:
    """Compiled regular expression."""

    def __init__(self, pattern: str):
        self.pattern = pattern
        tokens = lex(pattern)
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        self._fragment = compiler.compile(ast)
        self._fragment.end.is_accept = True
        self._state_map = _get_state_map(self._fragment)
        self._accept_id = self._fragment.end.id

        # Detect if pattern is anchored
        self._anchored_start = isinstance(ast, Concat) and isinstance(ast.left, AnchorStart)
        if isinstance(ast, AnchorStart):
            self._anchored_start = True
        # Walk concat chain to find anchor start
        node = ast
        while isinstance(node, Concat):
            if isinstance(node.left, AnchorStart):
                self._anchored_start = True
                break
            node = node.left

    def _epsilon_closure_at(self, states: set, pos: int, text_len: int) -> set:
        """Get epsilon closure respecting position for anchors."""
        stack = list(states)
        closure = set()
        while stack:
            s = stack.pop()
            if s.id in closure:
                continue
            if s.anchor_start and pos != 0:
                continue
            if s.anchor_end and pos != text_len:
                continue
            closure.add(s.id)
            for next_s in s.epsilon:
                if next_s.id not in closure:
                    stack.append(next_s)
        return closure

    def fullmatch(self, text: str) -> bool:
        """Check if the entire text matches the pattern."""
        current = self._epsilon_closure_at({self._fragment.start}, 0, len(text))
        for i, ch in enumerate(text):
            next_raw = _step(self._state_map, current, ch)
            # Epsilon closure for position after consuming char
            next_states = set()
            for sid in next_raw:
                s = self._state_map[sid]
                next_states.update(
                    self._epsilon_closure_at({s}, i + 1, len(text))
                )
            current = next_states
            if not current:
                return False
        return self._accept_id in current

    def search(self, text: str) -> bool:
        """Check if the pattern matches anywhere in the text."""
        # Try starting at each position
        start_range = 1 if self._anchored_start else len(text) + 1
        for start_pos in range(start_range):
            current = self._epsilon_closure_at(
                {self._fragment.start}, start_pos, len(text)
            )
            if self._accept_id in current:
                return True
            for i in range(start_pos, len(text)):
                ch = text[i]
                next_raw = _step(self._state_map, current, ch)
                next_states = set()
                for sid in next_raw:
                    s = self._state_map[sid]
                    next_states.update(
                        self._epsilon_closure_at({s}, i + 1, len(text))
                    )
                current = next_states
                if not current:
                    break
                if self._accept_id in current:
                    return True
        return False

    def findall(self, text: str) -> list:
        """Find all non-overlapping matches. Returns list of matched strings."""
        results = []
        pos = 0
        while pos <= len(text):
            best_end = -1
            current = self._epsilon_closure_at(
                {self._fragment.start}, pos, len(text)
            )
            if self._accept_id in current:
                best_end = pos  # Empty match
            i = pos
            while i < len(text):
                ch = text[i]
                next_raw = _step(self._state_map, current, ch)
                next_states = set()
                for sid in next_raw:
                    s = self._state_map[sid]
                    next_states.update(
                        self._epsilon_closure_at({s}, i + 1, len(text))
                    )
                current = next_states
                if not current:
                    break
                i += 1
                if self._accept_id in current:
                    best_end = i
            if best_end >= pos:
                results.append(text[pos:best_end])
                pos = best_end + (1 if best_end == pos else 0)
            else:
                pos += 1
            if self._anchored_start:
                break
        return results


# ============================================================
# Convenience functions
# ============================================================

def compile(pattern: str) -> Regex:
    """Compile a regex pattern."""
    return Regex(pattern)

def fullmatch(pattern: str, text: str) -> bool:
    """Check if the entire text matches the pattern."""
    return Regex(pattern).fullmatch(text)

def search(pattern: str, text: str) -> bool:
    """Check if the pattern matches anywhere in the text."""
    return Regex(pattern).search(text)

def findall(pattern: str, text: str) -> list:
    """Find all non-overlapping matches."""
    return Regex(pattern).findall(text)
