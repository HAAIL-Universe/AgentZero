"""
Language Server Protocol Implementation for MiniLang
Challenge C019 -- AgentZero Session 020

A full LSP server composing C010 (lexer/parser) and C013 (type checker)
to provide IDE intelligence for the MiniLang language.

Features:
  - JSON-RPC 2.0 protocol layer (Content-Length framing)
  - LSP lifecycle: initialize, initialized, shutdown, exit
  - Text document synchronization (open, change, close)
  - Diagnostics via type checker integration
  - Completion: keywords, variables, functions in scope
  - Hover: type information at cursor position
  - Go-to-definition: jump to variable/function declarations
  - Document symbols: outline of declarations
  - Signature help: function parameter info

Architecture:
  Transport -> JSON-RPC 2.0 -> LSP Server -> Analysis Engine -> C010/C013

Composes: C010 (lexer, parser, AST), C013 (type checker)
"""

import json
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Import from C010 and C013
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C013_type_checker'))

from stack_vm import (
    lex, Token, TokenType, LexError, ParseError,
    IntLit, FloatLit, StringLit, BoolLit, Var,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr,
    ReturnStmt, PrintStmt, Program, Parser
)

from type_checker import (
    TypeChecker, TypeError_, TypeEnv, TInt, TFloat, TString, TBool,
    TVoid, TFunc, TVar, resolve, check_source, check_program, parse as tc_parse
)


# ============================================================
# JSON-RPC 2.0 Protocol Layer
# ============================================================

class JsonRpcError(Exception):
    """JSON-RPC error with code and message."""
    def __init__(self, code, message, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# LSP error codes
SERVER_NOT_INITIALIZED = -32002
REQUEST_CANCELLED = -32800


def encode_message(obj):
    """Encode a JSON-RPC message with Content-Length header."""
    body = json.dumps(obj, separators=(',', ':'))
    body_bytes = body.encode('utf-8')
    header = f"Content-Length: {len(body_bytes)}\r\n\r\n"
    return header.encode('ascii') + body_bytes


def decode_header(data):
    """
    Decode Content-Length header from bytes.
    Returns (content_length, header_end_pos) or (None, None) if incomplete.
    """
    # Find the header/body separator
    sep = b"\r\n\r\n"
    idx = data.find(sep)
    if idx == -1:
        return None, None

    header_text = data[:idx].decode('ascii')
    content_length = None
    for line in header_text.split('\r\n'):
        if line.lower().startswith('content-length:'):
            content_length = int(line.split(':', 1)[1].strip())
            break

    if content_length is None:
        return None, None

    return content_length, idx + len(sep)


def make_response(id_, result):
    """Create a JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def make_error_response(id_, code, message, data=None):
    """Create a JSON-RPC error response."""
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def make_notification(method, params=None):
    """Create a JSON-RPC notification (no id)."""
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# ============================================================
# Transport Layer
# ============================================================

class MemoryTransport:
    """In-memory transport for testing. Collects outgoing messages."""

    def __init__(self):
        self.outgoing = []
        self.incoming = b""

    def send(self, msg):
        """Send a JSON-RPC message."""
        self.outgoing.append(msg)

    def send_raw(self, data):
        """Send raw bytes (encoded message)."""
        self.outgoing.append(data)

    def feed(self, data):
        """Feed incoming data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.incoming += data

    def get_sent_messages(self):
        """Return all sent messages as decoded dicts."""
        results = []
        for msg in self.outgoing:
            if isinstance(msg, dict):
                results.append(msg)
            elif isinstance(msg, bytes):
                # Parse Content-Length framed message
                cl, pos = decode_header(msg)
                if cl is not None and pos is not None:
                    body = msg[pos:pos + cl]
                    results.append(json.loads(body))
            elif isinstance(msg, str):
                results.append(json.loads(msg))
        return results

    def clear(self):
        self.outgoing.clear()


# ============================================================
# LSP Position Types
# ============================================================

@dataclass
class Position:
    """LSP Position (0-based line and character)."""
    line: int
    character: int

    def to_dict(self):
        return {"line": self.line, "character": self.character}

    @staticmethod
    def from_dict(d):
        return Position(d["line"], d["character"])


@dataclass
class Range:
    """LSP Range."""
    start: Position
    end: Position

    def to_dict(self):
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @staticmethod
    def from_dict(d):
        return Range(Position.from_dict(d["start"]), Position.from_dict(d["end"]))


@dataclass
class Location:
    """LSP Location."""
    uri: str
    range: Range

    def to_dict(self):
        return {"uri": self.uri, "range": self.range.to_dict()}


@dataclass
class Diagnostic:
    """LSP Diagnostic."""
    range: Range
    message: str
    severity: int = 1  # 1=Error, 2=Warning, 3=Info, 4=Hint
    source: str = "minilang"

    def to_dict(self):
        return {
            "range": self.range.to_dict(),
            "message": self.message,
            "severity": self.severity,
            "source": self.source,
        }


# Completion item kinds
COMPLETION_TEXT = 1
COMPLETION_METHOD = 2
COMPLETION_FUNCTION = 3
COMPLETION_VARIABLE = 6
COMPLETION_KEYWORD = 14
COMPLETION_SNIPPET = 15

# Symbol kinds
SYMBOL_FUNCTION = 12
SYMBOL_VARIABLE = 13

# Diagnostic severity
SEVERITY_ERROR = 1
SEVERITY_WARNING = 2
SEVERITY_INFO = 3
SEVERITY_HINT = 4


# ============================================================
# Enhanced Lexer -- adds column tracking for LSP positions
# ============================================================

@dataclass
class RichToken:
    """Token with full position info (line, column, length)."""
    type: TokenType
    value: Any
    line: int      # 1-based
    column: int    # 0-based
    length: int    # character count


def rich_lex(source):
    """Lex source with full position tracking (line, column, length)."""
    tokens = []
    i = 0
    line = 1
    col = 0  # 0-based column

    while i < len(source):
        c = source[i]

        if c == '\n':
            line += 1
            col = 0
            i += 1
        elif c in ' \t\r':
            col += 1
            i += 1
        elif c == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
                col += 1
        elif c.isdigit():
            start = i
            start_col = col
            while i < len(source) and source[i].isdigit():
                i += 1
                col += 1
            if i < len(source) and source[i] == '.':
                i += 1
                col += 1
                while i < len(source) and source[i].isdigit():
                    i += 1
                    col += 1
                tokens.append(RichToken(TokenType.FLOAT, float(source[start:i]),
                                       line, start_col, i - start))
            else:
                tokens.append(RichToken(TokenType.INT, int(source[start:i]),
                                       line, start_col, i - start))
        elif c == '"':
            start = i
            start_col = col
            i += 1
            col += 1
            s = ""
            while i < len(source) and source[i] != '"':
                if source[i] == '\\' and i + 1 < len(source):
                    nc = source[i + 1]
                    if nc == 'n':
                        s += '\n'
                    elif nc == 't':
                        s += '\t'
                    elif nc == '\\':
                        s += '\\'
                    elif nc == '"':
                        s += '"'
                    else:
                        s += nc
                    i += 2
                    col += 2
                else:
                    s += source[i]
                    i += 1
                    col += 1
            if i < len(source):
                i += 1
                col += 1
            tokens.append(RichToken(TokenType.STRING, s, line, start_col, i - start))
        elif c.isalpha() or c == '_':
            start = i
            start_col = col
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
                col += 1
            word = source[start:i]
            KEYWORDS = {
                'let': TokenType.LET, 'if': TokenType.IF, 'else': TokenType.ELSE,
                'while': TokenType.WHILE, 'fn': TokenType.FN, 'return': TokenType.RETURN,
                'print': TokenType.PRINT, 'true': TokenType.TRUE, 'false': TokenType.FALSE,
                'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
            }
            if word in KEYWORDS:
                tt = KEYWORDS[word]
                if word == 'true':
                    tokens.append(RichToken(tt, True, line, start_col, i - start))
                elif word == 'false':
                    tokens.append(RichToken(tt, False, line, start_col, i - start))
                else:
                    tokens.append(RichToken(tt, word, line, start_col, i - start))
            else:
                tokens.append(RichToken(TokenType.IDENT, word, line, start_col, i - start))
        else:
            start_col = col
            SIMPLE = {
                '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
                '/': TokenType.SLASH, '%': TokenType.PERCENT,
                '(': TokenType.LPAREN, ')': TokenType.RPAREN,
                '{': TokenType.LBRACE, '}': TokenType.RBRACE,
                ';': TokenType.SEMICOLON, ',': TokenType.COMMA,
            }
            TWO_CHAR = {
                '==': TokenType.EQ, '!=': TokenType.NE,
                '<=': TokenType.LE, '>=': TokenType.GE,
            }
            two = source[i:i+2]
            if two in TWO_CHAR:
                tokens.append(RichToken(TWO_CHAR[two], two, line, start_col, 2))
                i += 2
                col += 2
            elif c == '=' and (i + 1 >= len(source) or source[i+1] != '='):
                tokens.append(RichToken(TokenType.ASSIGN, '=', line, start_col, 1))
                i += 1
                col += 1
            elif c == '<':
                tokens.append(RichToken(TokenType.LT, '<', line, start_col, 1))
                i += 1
                col += 1
            elif c == '>':
                tokens.append(RichToken(TokenType.GT, '>', line, start_col, 1))
                i += 1
                col += 1
            elif c in SIMPLE:
                tokens.append(RichToken(SIMPLE[c], c, line, start_col, 1))
                i += 1
                col += 1
            elif c in '[]':
                # Skip brackets -- not in base MiniLang
                i += 1
                col += 1
            else:
                # Skip unknown characters
                i += 1
                col += 1

    tokens.append(RichToken(TokenType.EOF, None, line, col, 0))
    return tokens


# ============================================================
# Document Manager
# ============================================================

class Document:
    """Represents an open text document."""

    def __init__(self, uri, text, version=0, language_id="minilang"):
        self.uri = uri
        self.text = text
        self.version = version
        self.language_id = language_id
        self._lines = None
        self._tokens = None
        self._ast = None
        self._type_errors = None
        self._parse_error = None
        self._lex_error = None
        self._checker = None
        self._symbols = None  # cached symbol table

    @property
    def lines(self):
        if self._lines is None:
            self._lines = self.text.split('\n')
        return self._lines

    def invalidate(self):
        """Clear cached analysis results."""
        self._lines = None
        self._tokens = None
        self._ast = None
        self._type_errors = None
        self._parse_error = None
        self._lex_error = None
        self._checker = None
        self._symbols = None

    def offset_at(self, line, character):
        """Convert LSP position (0-based) to string offset."""
        offset = 0
        for i, ln in enumerate(self.lines):
            if i == line:
                return offset + min(character, len(ln))
            offset += len(ln) + 1  # +1 for \n
        return offset

    def position_at_offset(self, offset):
        """Convert string offset to LSP Position (0-based)."""
        line = 0
        col = 0
        for i, ch in enumerate(self.text):
            if i == offset:
                return Position(line, col)
            if ch == '\n':
                line += 1
                col = 0
            else:
                col += 1
        return Position(line, col)

    def analyze(self):
        """Run full analysis: lex, parse, type-check."""
        self._lex_error = None
        self._parse_error = None
        self._type_errors = []

        # Lex with rich tokens
        try:
            self._tokens = rich_lex(self.text)
        except Exception as e:
            self._lex_error = str(e)
            self._tokens = []

        # Parse (uses the standard C010/C013 parser)
        try:
            self._ast = tc_parse(self.text)
        except LexError as e:
            self._lex_error = str(e)
            self._ast = None
        except ParseError as e:
            self._parse_error = str(e)
            self._ast = None
        except Exception as e:
            self._parse_error = str(e)
            self._ast = None

        # If full parse failed, try partial parse (complete statements only)
        if self._ast is None:
            self._try_partial_parse()

        # Type-check whatever AST we have
        if self._ast is not None:
            try:
                self._checker = TypeChecker()
                self._type_errors = self._checker.check(self._ast)
            except Exception as e:
                self._type_errors = [TypeError_(str(e), 0)]

        # Build symbol table
        self._symbols = self._build_symbols()

    def _try_partial_parse(self):
        """Try to parse complete statements from text, ignoring trailing incomplete code."""
        # Split into statements by finding semicolons and closing braces
        # Try progressively shorter prefixes
        text = self.text.rstrip()
        # Try to find the last complete statement
        attempts = []
        for i in range(len(text), 0, -1):
            if text[i-1] in (';', '}'):
                attempts.append(text[:i])
                if len(attempts) >= 3:
                    break

        for attempt in attempts:
            try:
                self._ast = tc_parse(attempt)
                return
            except (LexError, ParseError, Exception):
                continue

    def _build_symbols(self):
        """Extract declared symbols from AST."""
        symbols = []
        if self._ast is None:
            return symbols
        for stmt in self._ast.stmts:
            if isinstance(stmt, LetDecl):
                symbols.append({
                    'name': stmt.name,
                    'kind': SYMBOL_VARIABLE,
                    'line': stmt.line,
                    'type': self._get_var_type(stmt.name),
                    'node': stmt,
                })
            elif isinstance(stmt, FnDecl):
                symbols.append({
                    'name': stmt.name,
                    'kind': SYMBOL_FUNCTION,
                    'line': stmt.line,
                    'params': stmt.params,
                    'type': self._get_var_type(stmt.name),
                    'node': stmt,
                })
            elif isinstance(stmt, Assign):
                symbols.append({
                    'name': stmt.name,
                    'kind': SYMBOL_VARIABLE,
                    'line': stmt.line,
                    'type': self._get_var_type(stmt.name),
                    'node': stmt,
                })
        return symbols

    def _get_var_type(self, name):
        """Get type of a variable from the checker environment."""
        if self._checker is None:
            return None
        t = self._checker.env.lookup(name)
        if t is not None:
            return resolve(t)
        return None

    def get_diagnostics(self):
        """Get LSP diagnostics from analysis."""
        diags = []

        if self._lex_error:
            diags.append(Diagnostic(
                range=Range(Position(0, 0), Position(0, 1)),
                message=f"Lex error: {self._lex_error}",
                severity=SEVERITY_ERROR,
            ))

        if self._parse_error:
            # Try to extract line number from parse error
            line = self._extract_error_line(self._parse_error)
            diags.append(Diagnostic(
                range=Range(Position(line, 0), Position(line, 100)),
                message=f"Parse error: {self._parse_error}",
                severity=SEVERITY_ERROR,
            ))

        if self._type_errors:
            for te in self._type_errors:
                line = max(0, te.line - 1)  # Convert 1-based to 0-based
                diags.append(Diagnostic(
                    range=Range(Position(line, 0), Position(line, 100)),
                    message=te.message,
                    severity=SEVERITY_ERROR,
                ))

        return diags

    def _extract_error_line(self, msg):
        """Try to extract a line number from an error message."""
        import re
        m = re.search(r'line\s+(\d+)', msg, re.IGNORECASE)
        if m:
            return max(0, int(m.group(1)) - 1)
        return 0

    def get_token_at(self, line, character):
        """
        Find the token at the given LSP position (0-based).
        Returns RichToken or None.
        """
        if not self._tokens:
            return None
        for tok in self._tokens:
            if tok.type == TokenType.EOF:
                continue
            # Token line is 1-based, LSP line is 0-based
            tok_line = tok.line - 1
            if tok_line == line:
                if tok.column <= character < tok.column + tok.length:
                    return tok
        return None

    def get_word_at(self, line, character):
        """Get the word (identifier) at the cursor position."""
        if line >= len(self.lines):
            return None
        ln = self.lines[line]
        if character >= len(ln):
            return None

        # The character at cursor must be part of a word
        if not (ln[character].isalnum() or ln[character] == '_'):
            return None

        # Find word boundaries
        start = character
        while start > 0 and (ln[start - 1].isalnum() or ln[start - 1] == '_'):
            start -= 1
        end = character
        while end < len(ln) and (ln[end].isalnum() or ln[end] == '_'):
            end += 1

        if start == end:
            return None
        return ln[start:end]

    def get_completions_at(self, line, character):
        """Get completion items at cursor position."""
        items = []

        # Keywords
        keywords = ['let', 'if', 'else', 'while', 'fn', 'return', 'print',
                     'true', 'false', 'and', 'or', 'not']
        for kw in keywords:
            items.append({
                'label': kw,
                'kind': COMPLETION_KEYWORD,
                'detail': 'keyword',
            })

        # Built-in functions
        builtins = [
            ('print', 'fn(any) -> void', 'Print a value'),
        ]
        for name, sig, doc in builtins:
            items.append({
                'label': name,
                'kind': COMPLETION_FUNCTION,
                'detail': sig,
                'documentation': doc,
            })

        # Variables and functions from analysis
        if self._checker:
            env = self._checker.env
            seen = set()
            while env is not None:
                for name, typ in env.bindings.items():
                    if name not in seen:
                        seen.add(name)
                        resolved = resolve(typ)
                        if isinstance(resolved, TFunc):
                            items.append({
                                'label': name,
                                'kind': COMPLETION_FUNCTION,
                                'detail': repr(resolved),
                            })
                        else:
                            items.append({
                                'label': name,
                                'kind': COMPLETION_VARIABLE,
                                'detail': repr(resolved),
                            })
                env = env.parent

        return items

    def get_hover_at(self, line, character):
        """Get hover information at cursor position."""
        word = self.get_word_at(line, character)
        if not word:
            return None

        # Check keywords
        kw_info = {
            'let': 'Declare a variable: `let name = value;`',
            'if': 'Conditional: `if (cond) { ... } else { ... }`',
            'else': 'Else branch of conditional',
            'while': 'Loop: `while (cond) { ... }`',
            'fn': 'Function declaration: `fn name(params) { ... }`',
            'return': 'Return from function: `return value;`',
            'print': 'Print a value: `print(value);`',
            'true': 'Boolean literal `true` (type: bool)',
            'false': 'Boolean literal `false` (type: bool)',
            'and': 'Logical AND operator',
            'or': 'Logical OR operator',
            'not': 'Logical NOT operator',
        }
        if word in kw_info:
            return {'contents': kw_info[word]}

        # Check type environment
        if self._checker:
            typ = self._checker.env.lookup(word)
            if typ is not None:
                resolved = resolve(typ)
                if isinstance(resolved, TFunc):
                    return {
                        'contents': f"**{word}**: `{resolved!r}`\n\nFunction"
                    }
                else:
                    return {
                        'contents': f"**{word}**: `{resolved!r}`"
                    }

        return None

    def get_definition_at(self, line, character):
        """Find definition location for the symbol at cursor."""
        word = self.get_word_at(line, character)
        if not word:
            return None

        if not self._symbols:
            return None

        # Find the first declaration of this name
        for sym in self._symbols:
            if sym['name'] == word:
                def_line = max(0, sym['line'] - 1)  # 1-based to 0-based
                return Location(
                    uri=self.uri,
                    range=Range(
                        Position(def_line, 0),
                        Position(def_line, len(word)),
                    ),
                )
        return None

    def get_document_symbols(self):
        """Get document symbol outline."""
        if not self._symbols:
            return []

        result = []
        for sym in self._symbols:
            sym_line = max(0, sym['line'] - 1)
            entry = {
                'name': sym['name'],
                'kind': sym['kind'],
                'range': Range(Position(sym_line, 0), Position(sym_line, 100)).to_dict(),
                'selectionRange': Range(
                    Position(sym_line, 0),
                    Position(sym_line, len(sym['name']))
                ).to_dict(),
            }
            if sym.get('type'):
                entry['detail'] = repr(sym['type'])
            result.append(entry)
        return result

    def get_signature_help_at(self, line, character):
        """Get signature help for function call at cursor."""
        if line >= len(self.lines):
            return None

        ln = self.lines[line]
        # Walk backwards from cursor to find function name and open paren
        pos = min(character, len(ln)) - 1
        paren_depth = 0
        active_param = 0

        while pos >= 0:
            ch = ln[pos]
            if ch == ')':
                paren_depth += 1
            elif ch == '(':
                if paren_depth == 0:
                    # Found the opening paren -- function name is before it
                    name_end = pos
                    name_start = pos - 1
                    while name_start >= 0 and (ln[name_start].isalnum() or ln[name_start] == '_'):
                        name_start -= 1
                    name_start += 1
                    fn_name = ln[name_start:name_end]
                    if not fn_name:
                        return None

                    # Count commas between paren and cursor to determine active param
                    between = ln[pos + 1:character]
                    active_param = between.count(',')

                    return self._make_signature(fn_name, active_param)
                else:
                    paren_depth -= 1
            elif ch == ',' and paren_depth == 0:
                active_param += 1
            pos -= 1

        return None

    def _make_signature(self, fn_name, active_param):
        """Build signature help for a function."""
        # Check for print built-in first (always available)
        if fn_name == 'print':
            return {
                'signatures': [{
                    'label': 'print(value)',
                    'parameters': [{'label': 'value', 'documentation': 'Value to print'}],
                }],
                'activeSignature': 0,
                'activeParameter': active_param,
            }

        if self._checker is None:
            return None

        typ = self._checker.env.lookup(fn_name)
        if typ is None:
            return None

        resolved = resolve(typ)
        if not isinstance(resolved, TFunc):
            return None

        # Build parameter labels
        params = []
        for i, pt in enumerate(resolved.params):
            params.append({
                'label': f"p{i}: {pt!r}",
                'documentation': f"Parameter {i} of type {pt!r}",
            })

        param_strs = [f"p{i}: {pt!r}" for i, pt in enumerate(resolved.params)]
        label = f"{fn_name}({', '.join(param_strs)}) -> {resolved.ret!r}"

        return {
            'signatures': [{
                'label': label,
                'parameters': params,
            }],
            'activeSignature': 0,
            'activeParameter': min(active_param, len(params) - 1) if params else 0,
        }


class DocumentManager:
    """Manages open text documents."""

    def __init__(self):
        self.documents = {}

    def open(self, uri, text, version=0, language_id="minilang"):
        doc = Document(uri, text, version, language_id)
        doc.analyze()
        self.documents[uri] = doc
        return doc

    def change(self, uri, text, version=None):
        if uri not in self.documents:
            return self.open(uri, text, version or 0)
        doc = self.documents[uri]
        doc.text = text
        if version is not None:
            doc.version = version
        doc.invalidate()
        doc.analyze()
        return doc

    def close(self, uri):
        if uri in self.documents:
            del self.documents[uri]

    def get(self, uri):
        return self.documents.get(uri)


# ============================================================
# LSP Server
# ============================================================

class LSPServer:
    """
    Language Server Protocol implementation.

    Handles JSON-RPC messages, routes to appropriate handlers,
    and manages server lifecycle.
    """

    def __init__(self, transport=None):
        self.transport = transport or MemoryTransport()
        self.documents = DocumentManager()
        self.initialized = False
        self.shutdown_requested = False
        self._request_id = 0
        self._capabilities = {
            'textDocumentSync': {
                'openClose': True,
                'change': 1,  # Full sync
                'save': {'includeText': True},
            },
            'completionProvider': {
                'triggerCharacters': ['.', '('],
                'resolveProvider': False,
            },
            'hoverProvider': True,
            'definitionProvider': True,
            'documentSymbolProvider': True,
            'signatureHelpProvider': {
                'triggerCharacters': ['(', ','],
            },
        }

    def handle_message(self, msg):
        """
        Handle a single JSON-RPC message (already decoded).
        Returns a response dict or None for notifications.
        """
        if not isinstance(msg, dict):
            return make_error_response(None, INVALID_REQUEST, "Invalid request")

        jsonrpc = msg.get("jsonrpc")
        if jsonrpc != "2.0":
            return make_error_response(
                msg.get("id"), INVALID_REQUEST, "Invalid JSON-RPC version"
            )

        method = msg.get("method")
        params = msg.get("params", {})
        id_ = msg.get("id")

        # Notification (no id)
        is_notification = "id" not in msg

        if method is None and not is_notification:
            return make_error_response(id_, INVALID_REQUEST, "Missing method")

        # Check initialization state
        if not self.initialized and method not in ("initialize", "exit"):
            if not is_notification:
                return make_error_response(
                    id_, SERVER_NOT_INITIALIZED, "Server not initialized"
                )
            return None

        if self.shutdown_requested and method != "exit":
            if not is_notification:
                return make_error_response(
                    id_, INVALID_REQUEST, "Server is shutting down"
                )
            return None

        # Dispatch
        handler_name = 'handle_' + method.replace('/', '_').replace('$', '_')
        handler = getattr(self, handler_name, None)

        if handler:
            try:
                result = handler(params)
                if is_notification:
                    return None
                return make_response(id_, result)
            except JsonRpcError as e:
                if is_notification:
                    return None
                return make_error_response(id_, e.code, e.message, e.data)
            except Exception as e:
                if is_notification:
                    return None
                return make_error_response(id_, INTERNAL_ERROR, str(e))
        else:
            if is_notification:
                return None
            return make_error_response(id_, METHOD_NOT_FOUND, f"Method not found: {method}")

    def handle_raw(self, data):
        """
        Handle raw bytes (Content-Length framed).
        Returns list of response bytes.
        """
        responses = []
        buffer = data if isinstance(data, bytes) else data.encode('utf-8')

        while buffer:
            cl, pos = decode_header(buffer)
            if cl is None or pos is None:
                break
            if pos + cl > len(buffer):
                break

            body = buffer[pos:pos + cl]
            buffer = buffer[pos + cl:]

            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                responses.append(encode_message(
                    make_error_response(None, PARSE_ERROR, "Parse error")
                ))
                continue

            response = self.handle_message(msg)
            if response is not None:
                responses.append(encode_message(response))

        return responses

    def send_notification(self, method, params=None):
        """Send a notification to the client."""
        msg = make_notification(method, params)
        self.transport.send(msg)

    def publish_diagnostics(self, uri, diagnostics):
        """Publish diagnostics for a document."""
        self.send_notification("textDocument/publishDiagnostics", {
            "uri": uri,
            "diagnostics": [d.to_dict() for d in diagnostics],
        })

    # ---- Lifecycle ----

    def handle_initialize(self, params):
        self.initialized = True
        return {
            'capabilities': self._capabilities,
            'serverInfo': {
                'name': 'minilang-lsp',
                'version': '1.0.0',
            },
        }

    def handle_initialized(self, params):
        # Notification -- no response needed
        return None

    def handle_shutdown(self, params):
        self.shutdown_requested = True
        return None  # null result

    def handle_exit(self, params):
        # Server should exit
        return None

    # ---- Text Document Sync ----

    def handle_textDocument_didOpen(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        text = td.get("text", "")
        version = td.get("version", 0)
        lang = td.get("languageId", "minilang")

        doc = self.documents.open(uri, text, version, lang)
        self.publish_diagnostics(uri, doc.get_diagnostics())
        return None

    def handle_textDocument_didChange(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        version = td.get("version")
        changes = params.get("contentChanges", [])

        if changes:
            # Full sync -- take last change
            text = changes[-1].get("text", "")
            doc = self.documents.change(uri, text, version)
            self.publish_diagnostics(uri, doc.get_diagnostics())

        return None

    def handle_textDocument_didClose(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        self.documents.close(uri)
        # Clear diagnostics
        self.publish_diagnostics(uri, [])
        return None

    def handle_textDocument_didSave(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        text = params.get("text")
        if text is not None:
            doc = self.documents.change(uri, text)
            self.publish_diagnostics(uri, doc.get_diagnostics())
        return None

    # ---- Completion ----

    def handle_textDocument_completion(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        pos = params.get("position", {})
        line = pos.get("line", 0)
        character = pos.get("character", 0)

        doc = self.documents.get(uri)
        if not doc:
            return {'isIncomplete': False, 'items': []}

        items = doc.get_completions_at(line, character)
        return {'isIncomplete': False, 'items': items}

    # ---- Hover ----

    def handle_textDocument_hover(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        pos = params.get("position", {})
        line = pos.get("line", 0)
        character = pos.get("character", 0)

        doc = self.documents.get(uri)
        if not doc:
            return None

        hover = doc.get_hover_at(line, character)
        return hover

    # ---- Go to Definition ----

    def handle_textDocument_definition(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        pos = params.get("position", {})
        line = pos.get("line", 0)
        character = pos.get("character", 0)

        doc = self.documents.get(uri)
        if not doc:
            return None

        loc = doc.get_definition_at(line, character)
        if loc:
            return loc.to_dict()
        return None

    # ---- Document Symbols ----

    def handle_textDocument_documentSymbol(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")

        doc = self.documents.get(uri)
        if not doc:
            return []

        return doc.get_document_symbols()

    # ---- Signature Help ----

    def handle_textDocument_signatureHelp(self, params):
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        pos = params.get("position", {})
        line = pos.get("line", 0)
        character = pos.get("character", 0)

        doc = self.documents.get(uri)
        if not doc:
            return None

        return doc.get_signature_help_at(line, character)


# ============================================================
# Client Simulator (for testing)
# ============================================================

class LSPClient:
    """
    Client-side helper for interacting with the LSP server.
    Simplifies testing by wrapping JSON-RPC message construction.
    """

    def __init__(self, server):
        self.server = server
        self._id = 0

    def _next_id(self):
        self._id += 1
        return self._id

    def request(self, method, params=None):
        """Send a request and return the result (or raise on error)."""
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        response = self.server.handle_message(msg)
        if response and "error" in response:
            raise JsonRpcError(
                response["error"]["code"],
                response["error"]["message"],
                response["error"].get("data"),
            )
        if response:
            return response.get("result")
        return None

    def notify(self, method, params=None):
        """Send a notification (no response expected)."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        self.server.handle_message(msg)

    def initialize(self, root_uri="file:///workspace"):
        """Send initialize request."""
        return self.request("initialize", {
            "rootUri": root_uri,
            "capabilities": {},
        })

    def initialized(self):
        """Send initialized notification."""
        self.notify("initialized", {})

    def shutdown(self):
        """Send shutdown request."""
        return self.request("shutdown")

    def exit(self):
        """Send exit notification."""
        self.notify("exit")

    def open_document(self, uri, text, language_id="minilang", version=0):
        """Open a text document."""
        self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": version,
                "text": text,
            },
        })

    def change_document(self, uri, text, version=1):
        """Change a text document (full sync)."""
        self.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": text}],
        })

    def close_document(self, uri):
        """Close a text document."""
        self.notify("textDocument/didClose", {
            "textDocument": {"uri": uri},
        })

    def completion(self, uri, line, character):
        """Request completion at position."""
        return self.request("textDocument/completion", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

    def hover(self, uri, line, character):
        """Request hover at position."""
        return self.request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

    def definition(self, uri, line, character):
        """Request go-to-definition at position."""
        return self.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

    def document_symbols(self, uri):
        """Request document symbols."""
        return self.request("textDocument/documentSymbol", {
            "textDocument": {"uri": uri},
        })

    def signature_help(self, uri, line, character):
        """Request signature help at position."""
        return self.request("textDocument/signatureHelp", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

    def get_notifications(self, method=None):
        """Get sent notifications from the server transport."""
        msgs = self.server.transport.get_sent_messages()
        if method:
            return [m for m in msgs if m.get("method") == method]
        return [m for m in msgs if "method" in m]

    def get_diagnostics(self, uri=None):
        """Get published diagnostics."""
        notifs = self.get_notifications("textDocument/publishDiagnostics")
        if uri:
            notifs = [n for n in notifs if n.get("params", {}).get("uri") == uri]
        return notifs

    def clear_notifications(self):
        """Clear collected notifications."""
        self.server.transport.clear()
