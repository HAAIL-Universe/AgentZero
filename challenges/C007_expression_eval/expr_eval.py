"""
Expression Evaluator

Recursive descent parser for mathematical expressions.
Supports: +, -, *, /, parentheses, unary minus, variables, assignment.

Grammar:
  statement  -> IDENT '=' expr | expr
  expr       -> term (('+' | '-') term)*
  term       -> unary (('*' | '/') unary)*
  unary      -> '-' unary | primary
  primary    -> NUMBER | IDENT | '(' expr ')'
"""

from enum import Enum, auto
from typing import Optional


class TokenType(Enum):
    NUMBER = auto()
    IDENT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    LPAREN = auto()
    RPAREN = auto()
    EQUALS = auto()
    EOF = auto()


class Token:
    __slots__ = ("type", "value", "pos")

    def __init__(self, type: TokenType, value: str, pos: int):
        self.type = type
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, pos={self.pos})"


class EvalError(Exception):
    """Raised for syntax or evaluation errors."""
    def __init__(self, message: str, pos: Optional[int] = None):
        self.pos = pos
        super().__init__(message)


# --- Lexer ---

def tokenize(source: str) -> list[Token]:
    tokens = []
    i = 0
    while i < len(source):
        ch = source[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit() or (ch == '.' and i + 1 < len(source) and source[i + 1].isdigit()):
            start = i
            has_dot = (ch == '.')
            i += 1
            while i < len(source) and (source[i].isdigit() or (source[i] == '.' and not has_dot)):
                if source[i] == '.':
                    has_dot = True
                i += 1
            tokens.append(Token(TokenType.NUMBER, source[start:i], start))
            continue

        if ch.isalpha() or ch == '_':
            start = i
            i += 1
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            tokens.append(Token(TokenType.IDENT, source[start:i], start))
            continue

        simple = {
            '+': TokenType.PLUS, '-': TokenType.MINUS,
            '*': TokenType.STAR, '/': TokenType.SLASH,
            '(': TokenType.LPAREN, ')': TokenType.RPAREN,
            '=': TokenType.EQUALS,
        }
        if ch in simple:
            tokens.append(Token(simple[ch], ch, i))
            i += 1
            continue

        raise EvalError(f"Unexpected character '{ch}'", pos=i)

    tokens.append(Token(TokenType.EOF, "", i))
    return tokens


# --- Parser / Evaluator ---

class Evaluator:
    def __init__(self, variables: Optional[dict[str, float]] = None):
        self.variables: dict[str, float] = variables or {}
        self._tokens: list[Token] = []
        self._pos = 0

    def evaluate(self, source: str) -> float:
        """Parse and evaluate a single expression or assignment."""
        self._tokens = tokenize(source)
        self._pos = 0
        result = self._statement()
        if self._current().type != TokenType.EOF:
            raise EvalError(
                f"Unexpected token '{self._current().value}'",
                pos=self._current().pos
            )
        return result

    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, type: TokenType) -> Token:
        tok = self._current()
        if tok.type != type:
            raise EvalError(
                f"Expected {type.name}, got '{tok.value}'",
                pos=tok.pos
            )
        return self._advance()

    def _statement(self) -> float:
        """statement -> IDENT '=' expr | expr"""
        # Look ahead: if IDENT followed by '=', it's assignment
        if (self._current().type == TokenType.IDENT
                and self._pos + 1 < len(self._tokens)
                and self._tokens[self._pos + 1].type == TokenType.EQUALS):
            name = self._advance().value
            self._advance()  # consume '='
            value = self._expr()
            self.variables[name] = value
            return value
        return self._expr()

    def _expr(self) -> float:
        """expr -> term (('+' | '-') term)*"""
        left = self._term()
        while self._current().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._advance()
            right = self._term()
            if op.type == TokenType.PLUS:
                left += right
            else:
                left -= right
        return left

    def _term(self) -> float:
        """term -> unary (('*' | '/') unary)*"""
        left = self._unary()
        while self._current().type in (TokenType.STAR, TokenType.SLASH):
            op = self._advance()
            right = self._unary()
            if op.type == TokenType.STAR:
                left *= right
            else:
                if right == 0:
                    raise EvalError("Division by zero", pos=op.pos)
                left /= right
        return left

    def _unary(self) -> float:
        """unary -> '-' unary | primary"""
        if self._current().type == TokenType.MINUS:
            self._advance()
            return -self._unary()
        return self._primary()

    def _primary(self) -> float:
        """primary -> NUMBER | IDENT | '(' expr ')'"""
        tok = self._current()

        if tok.type == TokenType.NUMBER:
            self._advance()
            return float(tok.value)

        if tok.type == TokenType.IDENT:
            self._advance()
            if tok.value not in self.variables:
                raise EvalError(f"Undefined variable '{tok.value}'", pos=tok.pos)
            return self.variables[tok.value]

        if tok.type == TokenType.LPAREN:
            self._advance()
            value = self._expr()
            self._expect(TokenType.RPAREN)
            return value

        raise EvalError(f"Unexpected token '{tok.value}'", pos=tok.pos)


def calc(expr: str, variables: dict[str, float] = None) -> float:
    """Convenience function: evaluate a single expression."""
    ev = Evaluator(variables)
    return ev.evaluate(expr)


if __name__ == "__main__":
    ev = Evaluator()
    print("Expression evaluator. Type 'quit' to exit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip().lower() == "quit":
            break
        try:
            result = ev.evaluate(line)
            print(f"  = {result}")
        except EvalError as e:
            print(f"  Error: {e}")
