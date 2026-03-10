"""
Tests for C019 LSP Server
Target: 100+ tests covering all LSP features
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from lsp_server import (
    # JSON-RPC
    encode_message, decode_header, make_response, make_error_response,
    make_notification, JsonRpcError,
    PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INTERNAL_ERROR,
    SERVER_NOT_INITIALIZED,
    # Transport
    MemoryTransport,
    # LSP types
    Position, Range, Location, Diagnostic,
    SEVERITY_ERROR, SEVERITY_WARNING,
    COMPLETION_KEYWORD, COMPLETION_FUNCTION, COMPLETION_VARIABLE,
    SYMBOL_FUNCTION, SYMBOL_VARIABLE,
    # Lexer
    RichToken, rich_lex,
    # Documents
    Document, DocumentManager,
    # Server
    LSPServer, LSPClient,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import TokenType


# ============================================================
# JSON-RPC 2.0 Protocol Tests
# ============================================================

class TestJsonRpcEncode:
    def test_encode_basic(self):
        msg = {"jsonrpc": "2.0", "id": 1, "result": None}
        raw = encode_message(msg)
        assert b"Content-Length:" in raw
        assert b"\r\n\r\n" in raw

    def test_encode_decode_roundtrip(self):
        msg = {"jsonrpc": "2.0", "id": 1, "result": {"x": 42}}
        raw = encode_message(msg)
        cl, pos = decode_header(raw)
        body = raw[pos:pos + cl]
        decoded = json.loads(body)
        assert decoded == msg

    def test_content_length_correct(self):
        msg = {"jsonrpc": "2.0", "method": "test"}
        raw = encode_message(msg)
        cl, pos = decode_header(raw)
        body = raw[pos:]
        assert len(body) == cl

    def test_decode_header_incomplete(self):
        cl, pos = decode_header(b"Content-Len")
        assert cl is None
        assert pos is None

    def test_decode_header_no_content_length(self):
        cl, pos = decode_header(b"X-Custom: foo\r\n\r\n{}")
        assert cl is None

    def test_encode_unicode(self):
        msg = {"jsonrpc": "2.0", "id": 1, "result": "hello"}
        raw = encode_message(msg)
        cl, pos = decode_header(raw)
        assert cl is not None


class TestJsonRpcHelpers:
    def test_make_response(self):
        r = make_response(1, {"x": 1})
        assert r["jsonrpc"] == "2.0"
        assert r["id"] == 1
        assert r["result"] == {"x": 1}

    def test_make_error_response(self):
        r = make_error_response(2, -32600, "Bad request")
        assert r["error"]["code"] == -32600
        assert r["error"]["message"] == "Bad request"
        assert r["id"] == 2

    def test_make_error_response_with_data(self):
        r = make_error_response(3, -32603, "Internal", {"detail": "oops"})
        assert r["error"]["data"]["detail"] == "oops"

    def test_make_notification(self):
        n = make_notification("test/notify", {"key": "val"})
        assert n["method"] == "test/notify"
        assert "id" not in n

    def test_make_notification_no_params(self):
        n = make_notification("test/ping")
        assert "params" not in n


# ============================================================
# Transport Tests
# ============================================================

class TestMemoryTransport:
    def test_send_dict(self):
        t = MemoryTransport()
        t.send({"jsonrpc": "2.0", "method": "test"})
        msgs = t.get_sent_messages()
        assert len(msgs) == 1
        assert msgs[0]["method"] == "test"

    def test_send_bytes(self):
        t = MemoryTransport()
        msg = {"jsonrpc": "2.0", "id": 1, "result": 42}
        t.send_raw(encode_message(msg))
        msgs = t.get_sent_messages()
        assert len(msgs) == 1
        assert msgs[0]["result"] == 42

    def test_clear(self):
        t = MemoryTransport()
        t.send({"test": True})
        t.clear()
        assert len(t.get_sent_messages()) == 0


# ============================================================
# Position / Range / Location Tests
# ============================================================

class TestPositionTypes:
    def test_position_to_dict(self):
        p = Position(3, 7)
        assert p.to_dict() == {"line": 3, "character": 7}

    def test_position_from_dict(self):
        p = Position.from_dict({"line": 1, "character": 5})
        assert p.line == 1
        assert p.character == 5

    def test_range_to_dict(self):
        r = Range(Position(0, 0), Position(0, 10))
        d = r.to_dict()
        assert d["start"]["line"] == 0
        assert d["end"]["character"] == 10

    def test_range_from_dict(self):
        r = Range.from_dict({
            "start": {"line": 1, "character": 2},
            "end": {"line": 3, "character": 4},
        })
        assert r.start.line == 1
        assert r.end.character == 4

    def test_location_to_dict(self):
        loc = Location("file:///test.ml", Range(Position(0, 0), Position(0, 5)))
        d = loc.to_dict()
        assert d["uri"] == "file:///test.ml"
        assert d["range"]["start"]["line"] == 0

    def test_diagnostic_to_dict(self):
        diag = Diagnostic(
            range=Range(Position(2, 0), Position(2, 10)),
            message="Type error",
            severity=SEVERITY_ERROR,
        )
        d = diag.to_dict()
        assert d["message"] == "Type error"
        assert d["severity"] == 1
        assert d["source"] == "minilang"


# ============================================================
# Rich Lexer Tests
# ============================================================

class TestRichLex:
    def test_empty(self):
        tokens = rich_lex("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_integer(self):
        tokens = rich_lex("42")
        assert tokens[0].type == TokenType.INT
        assert tokens[0].value == 42
        assert tokens[0].line == 1
        assert tokens[0].column == 0
        assert tokens[0].length == 2

    def test_float(self):
        tokens = rich_lex("3.14")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == 3.14
        assert tokens[0].length == 4

    def test_string(self):
        tokens = rich_lex('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
        assert tokens[0].length == 7

    def test_string_escape(self):
        tokens = rich_lex('"he\\nllo"')
        assert tokens[0].value == "he\nllo"

    def test_identifier(self):
        tokens = rich_lex("foo")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "foo"
        assert tokens[0].column == 0
        assert tokens[0].length == 3

    def test_keyword_let(self):
        tokens = rich_lex("let")
        assert tokens[0].type == TokenType.LET

    def test_keyword_fn(self):
        tokens = rich_lex("fn")
        assert tokens[0].type == TokenType.FN

    def test_keyword_true_false(self):
        tokens = rich_lex("true false")
        assert tokens[0].type == TokenType.TRUE
        assert tokens[0].value == True
        assert tokens[1].type == TokenType.FALSE
        assert tokens[1].value == False

    def test_operators(self):
        tokens = rich_lex("+ - * / %")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.PLUS in types
        assert TokenType.MINUS in types
        assert TokenType.STAR in types

    def test_two_char_operators(self):
        tokens = rich_lex("== != <= >=")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.EQ in types
        assert TokenType.NE in types
        assert TokenType.LE in types
        assert TokenType.GE in types

    def test_assign_vs_eq(self):
        tokens = rich_lex("= ==")
        assert tokens[0].type == TokenType.ASSIGN
        assert tokens[1].type == TokenType.EQ

    def test_multiline_positions(self):
        tokens = rich_lex("let x = 1;\nlet y = 2;")
        # Find 'y' token
        y_tok = [t for t in tokens if t.type == TokenType.IDENT and t.value == 'y']
        assert len(y_tok) == 1
        assert y_tok[0].line == 2
        assert y_tok[0].column == 4

    def test_comment_skipped(self):
        tokens = rich_lex("// comment\n42")
        non_eof = [t for t in tokens if t.type != TokenType.EOF]
        assert len(non_eof) == 1
        assert non_eof[0].value == 42

    def test_column_after_spaces(self):
        tokens = rich_lex("  x")
        assert tokens[0].column == 2

    def test_parens(self):
        tokens = rich_lex("()")
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.RPAREN

    def test_braces(self):
        tokens = rich_lex("{}")
        assert tokens[0].type == TokenType.LBRACE
        assert tokens[1].type == TokenType.RBRACE

    def test_semicolon(self):
        tokens = rich_lex(";")
        assert tokens[0].type == TokenType.SEMICOLON

    def test_lt_gt(self):
        tokens = rich_lex("< >")
        assert tokens[0].type == TokenType.LT
        assert tokens[1].type == TokenType.GT


# ============================================================
# Document Tests
# ============================================================

class TestDocument:
    def test_create(self):
        doc = Document("file:///test.ml", "let x = 1;")
        assert doc.uri == "file:///test.ml"
        assert doc.text == "let x = 1;"

    def test_lines(self):
        doc = Document("", "line1\nline2\nline3")
        assert doc.lines == ["line1", "line2", "line3"]

    def test_offset_at(self):
        doc = Document("", "abc\ndef\nghi")
        assert doc.offset_at(0, 0) == 0
        assert doc.offset_at(0, 2) == 2
        assert doc.offset_at(1, 0) == 4
        assert doc.offset_at(1, 2) == 6
        assert doc.offset_at(2, 1) == 9

    def test_offset_at_clamps(self):
        doc = Document("", "ab\ncd")
        # Character beyond line length
        assert doc.offset_at(0, 10) == 2  # clamped to line end

    def test_position_at_offset(self):
        doc = Document("", "abc\ndef")
        p = doc.position_at_offset(0)
        assert p.line == 0 and p.character == 0
        p = doc.position_at_offset(4)
        assert p.line == 1 and p.character == 0
        p = doc.position_at_offset(5)
        assert p.line == 1 and p.character == 1

    def test_invalidate(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        assert doc._tokens is not None
        doc.invalidate()
        assert doc._tokens is None

    def test_analyze_valid(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        assert doc._ast is not None
        assert doc._type_errors == []
        assert doc._parse_error is None

    def test_analyze_parse_error(self):
        doc = Document("", "let = ;")
        doc.analyze()
        assert doc._parse_error is not None

    def test_analyze_type_error(self):
        # Assignment to undefined variable
        doc = Document("", 'x = 1;')
        doc.analyze()
        assert len(doc._type_errors) > 0

    def test_get_diagnostics_clean(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        diags = doc.get_diagnostics()
        assert len(diags) == 0

    def test_get_diagnostics_type_error(self):
        doc = Document("", "x = 1;")
        doc.analyze()
        diags = doc.get_diagnostics()
        assert len(diags) > 0
        assert diags[0].severity == SEVERITY_ERROR

    def test_get_diagnostics_parse_error(self):
        doc = Document("", "let = ;")
        doc.analyze()
        diags = doc.get_diagnostics()
        assert len(diags) > 0
        assert "Parse error" in diags[0].message or "parse" in diags[0].message.lower()

    def test_get_word_at(self):
        doc = Document("", "let foo = 1;")
        doc.analyze()
        assert doc.get_word_at(0, 4) == "foo"
        assert doc.get_word_at(0, 5) == "foo"
        assert doc.get_word_at(0, 6) == "foo"

    def test_get_word_at_boundary(self):
        doc = Document("", "let x = 1;")
        assert doc.get_word_at(0, 0) == "let"
        assert doc.get_word_at(0, 3) is None  # space
        assert doc.get_word_at(0, 4) == "x"

    def test_get_word_at_out_of_range(self):
        doc = Document("", "x")
        assert doc.get_word_at(5, 0) is None

    def test_get_token_at(self):
        doc = Document("", "let x = 42;")
        doc.analyze()
        tok = doc.get_token_at(0, 0)
        assert tok is not None
        assert tok.type == TokenType.LET
        tok = doc.get_token_at(0, 8)
        assert tok is not None
        assert tok.value == 42

    def test_get_token_at_none(self):
        doc = Document("", "x")
        doc.analyze()
        tok = doc.get_token_at(5, 0)  # out of range
        assert tok is None


# ============================================================
# Document Completions Tests
# ============================================================

class TestDocumentCompletions:
    def test_keywords_present(self):
        doc = Document("", "")
        doc.analyze()
        items = doc.get_completions_at(0, 0)
        labels = [i['label'] for i in items]
        assert 'let' in labels
        assert 'fn' in labels
        assert 'if' in labels
        assert 'while' in labels
        assert 'return' in labels

    def test_variables_from_analysis(self):
        doc = Document("", "let myVar = 42;\n")
        doc.analyze()
        items = doc.get_completions_at(1, 0)
        labels = [i['label'] for i in items]
        assert 'myVar' in labels

    def test_functions_from_analysis(self):
        doc = Document("", "fn add(a, b) { return a + b; }\n")
        doc.analyze()
        items = doc.get_completions_at(1, 0)
        labels = [i['label'] for i in items]
        assert 'add' in labels
        # Should be marked as function
        add_item = [i for i in items if i['label'] == 'add'][0]
        assert add_item['kind'] == COMPLETION_FUNCTION

    def test_builtin_print(self):
        doc = Document("", "")
        doc.analyze()
        items = doc.get_completions_at(0, 0)
        labels = [i['label'] for i in items]
        assert 'print' in labels


# ============================================================
# Document Hover Tests
# ============================================================

class TestDocumentHover:
    def test_hover_keyword(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        hover = doc.get_hover_at(0, 0)  # "let"
        assert hover is not None
        assert 'let' in hover['contents'].lower() or 'variable' in hover['contents'].lower()

    def test_hover_variable(self):
        doc = Document("", "let x = 42;")
        doc.analyze()
        hover = doc.get_hover_at(0, 4)  # "x"
        assert hover is not None
        assert 'int' in hover['contents']

    def test_hover_function(self):
        doc = Document("", "fn add(a, b) { return a + b; }")
        doc.analyze()
        hover = doc.get_hover_at(0, 3)  # "add"
        assert hover is not None
        assert 'fn' in hover['contents'].lower() or 'Function' in hover['contents']

    def test_hover_no_info(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        hover = doc.get_hover_at(0, 3)  # space
        assert hover is None

    def test_hover_bool_keyword(self):
        doc = Document("", "true")
        doc.analyze()
        hover = doc.get_hover_at(0, 0)
        assert hover is not None
        assert 'bool' in hover['contents']


# ============================================================
# Document Definition Tests
# ============================================================

class TestDocumentDefinition:
    def test_go_to_variable_def(self):
        doc = Document("file:///test.ml", "let x = 1;\nprint(x);")
        doc.analyze()
        loc = doc.get_definition_at(1, 6)  # "x" in print(x)
        assert loc is not None
        assert loc.range.start.line == 0  # defined on line 0

    def test_go_to_function_def(self):
        doc = Document("file:///test.ml", "fn foo() { return 1; }\nfoo();")
        doc.analyze()
        loc = doc.get_definition_at(1, 0)  # "foo" in foo()
        assert loc is not None
        assert loc.range.start.line == 0

    def test_definition_not_found(self):
        doc = Document("file:///test.ml", "let x = 1;")
        doc.analyze()
        loc = doc.get_definition_at(0, 8)  # number literal
        assert loc is None

    def test_definition_uri_preserved(self):
        doc = Document("file:///my/file.ml", "let x = 1;\nprint(x);")
        doc.analyze()
        loc = doc.get_definition_at(1, 6)
        assert loc is not None
        assert loc.uri == "file:///my/file.ml"


# ============================================================
# Document Symbols Tests
# ============================================================

class TestDocumentSymbols:
    def test_variable_symbol(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        syms = doc.get_document_symbols()
        assert len(syms) >= 1
        assert syms[0]['name'] == 'x'
        assert syms[0]['kind'] == SYMBOL_VARIABLE

    def test_function_symbol(self):
        doc = Document("", "fn foo() { return 1; }")
        doc.analyze()
        syms = doc.get_document_symbols()
        assert any(s['name'] == 'foo' and s['kind'] == SYMBOL_FUNCTION for s in syms)

    def test_multiple_symbols(self):
        doc = Document("", "let a = 1;\nlet b = 2;\nfn c() { return 3; }")
        doc.analyze()
        syms = doc.get_document_symbols()
        names = [s['name'] for s in syms]
        assert 'a' in names
        assert 'b' in names
        assert 'c' in names

    def test_symbol_has_range(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        syms = doc.get_document_symbols()
        assert 'range' in syms[0]
        assert 'selectionRange' in syms[0]

    def test_empty_document_no_symbols(self):
        doc = Document("", "")
        doc.analyze()
        syms = doc.get_document_symbols()
        assert len(syms) == 0


# ============================================================
# Signature Help Tests
# ============================================================

class TestSignatureHelp:
    def test_print_signature(self):
        doc = Document("", "print(")
        doc.analyze()
        sig = doc.get_signature_help_at(0, 6)
        assert sig is not None
        assert len(sig['signatures']) > 0
        assert 'print' in sig['signatures'][0]['label']

    def test_user_function_signature(self):
        doc = Document("", "fn add(a, b) { return a + b; }\nadd(")
        doc.analyze()
        sig = doc.get_signature_help_at(1, 4)
        assert sig is not None
        assert 'add' in sig['signatures'][0]['label']

    def test_active_parameter(self):
        doc = Document("", "fn foo(a, b, c) { return a; }\nfoo(1, 2,")
        doc.analyze()
        sig = doc.get_signature_help_at(1, 9)
        assert sig is not None
        assert sig['activeParameter'] == 2

    def test_no_signature_outside_call(self):
        doc = Document("", "let x = 1;")
        doc.analyze()
        sig = doc.get_signature_help_at(0, 5)
        assert sig is None


# ============================================================
# Document Manager Tests
# ============================================================

class TestDocumentManager:
    def test_open_and_get(self):
        dm = DocumentManager()
        doc = dm.open("file:///a.ml", "let x = 1;")
        assert dm.get("file:///a.ml") is doc

    def test_change(self):
        dm = DocumentManager()
        dm.open("file:///a.ml", "let x = 1;")
        doc = dm.change("file:///a.ml", "let y = 2;", version=1)
        assert doc.text == "let y = 2;"
        assert doc.version == 1

    def test_close(self):
        dm = DocumentManager()
        dm.open("file:///a.ml", "let x = 1;")
        dm.close("file:///a.ml")
        assert dm.get("file:///a.ml") is None

    def test_change_creates_if_missing(self):
        dm = DocumentManager()
        doc = dm.change("file:///new.ml", "let z = 3;")
        assert doc.text == "let z = 3;"

    def test_multiple_documents(self):
        dm = DocumentManager()
        dm.open("file:///a.ml", "let a = 1;")
        dm.open("file:///b.ml", "let b = 2;")
        assert dm.get("file:///a.ml").text == "let a = 1;"
        assert dm.get("file:///b.ml").text == "let b = 2;"


# ============================================================
# LSP Server Lifecycle Tests
# ============================================================

class TestLSPLifecycle:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)

    def test_initialize(self):
        result = self.client.initialize()
        assert 'capabilities' in result
        assert 'serverInfo' in result
        assert result['serverInfo']['name'] == 'minilang-lsp'

    def test_capabilities_returned(self):
        result = self.client.initialize()
        caps = result['capabilities']
        assert caps['textDocumentSync']['openClose'] is True
        assert caps['completionProvider'] is not None
        assert caps['hoverProvider'] is True
        assert caps['definitionProvider'] is True
        assert caps['documentSymbolProvider'] is True
        assert caps['signatureHelpProvider'] is not None

    def test_initialized_notification(self):
        self.client.initialize()
        # Should not raise
        self.client.initialized()

    def test_shutdown(self):
        self.client.initialize()
        result = self.client.shutdown()
        assert result is None  # null result on success

    def test_exit_after_shutdown(self):
        self.client.initialize()
        self.client.shutdown()
        # Exit should not raise
        self.client.exit()

    def test_request_before_initialize(self):
        with pytest.raises(JsonRpcError) as exc_info:
            self.client.request("textDocument/completion", {})
        assert exc_info.value.code == SERVER_NOT_INITIALIZED

    def test_request_after_shutdown(self):
        self.client.initialize()
        self.client.shutdown()
        with pytest.raises(JsonRpcError):
            self.client.request("textDocument/completion", {})

    def test_method_not_found(self):
        self.client.initialize()
        with pytest.raises(JsonRpcError) as exc_info:
            self.client.request("nonexistent/method", {})
        assert exc_info.value.code == METHOD_NOT_FOUND


# ============================================================
# LSP Server Message Handling Tests
# ============================================================

class TestLSPMessageHandling:
    def setup_method(self):
        self.server = LSPServer()

    def test_invalid_jsonrpc_version(self):
        resp = self.server.handle_message({"jsonrpc": "1.0", "id": 1, "method": "test"})
        assert "error" in resp

    def test_non_dict_message(self):
        resp = self.server.handle_message("not a dict")
        assert "error" in resp

    def test_notification_no_response(self):
        # Initialize first
        self.server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        # Notification should return None
        resp = self.server.handle_message({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        assert resp is None

    def test_handle_raw_basic(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        raw = encode_message(msg)
        responses = self.server.handle_raw(raw)
        assert len(responses) == 1
        # Decode response
        cl, pos = decode_header(responses[0])
        body = json.loads(responses[0][pos:pos + cl])
        assert "result" in body

    def test_handle_raw_parse_error(self):
        bad = b"Content-Length: 5\r\n\r\n{bad}"
        responses = self.server.handle_raw(bad)
        assert len(responses) == 1

    def test_handle_raw_multiple_messages(self):
        msg1 = encode_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        # After init, send shutdown
        msg2 = encode_message({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
        responses = self.server.handle_raw(msg1 + msg2)
        assert len(responses) == 2


# ============================================================
# LSP Server Document Sync Tests
# ============================================================

class TestLSPDocumentSync:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_open_document(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        doc = self.server.documents.get("file:///test.ml")
        assert doc is not None
        assert doc.text == "let x = 1;"

    def test_open_publishes_diagnostics(self):
        self.client.open_document("file:///test.ml", "x = 1;")
        notifs = self.client.get_diagnostics("file:///test.ml")
        assert len(notifs) > 0
        diags = notifs[-1]["params"]["diagnostics"]
        assert len(diags) > 0

    def test_open_clean_no_diagnostics(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        notifs = self.client.get_diagnostics("file:///test.ml")
        assert len(notifs) > 0
        diags = notifs[-1]["params"]["diagnostics"]
        assert len(diags) == 0

    def test_change_document(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        self.client.change_document("file:///test.ml", "let y = 2;", version=1)
        doc = self.server.documents.get("file:///test.ml")
        assert doc.text == "let y = 2;"

    def test_change_publishes_diagnostics(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        self.client.clear_notifications()
        self.client.change_document("file:///test.ml", "y = 1;", version=1)
        notifs = self.client.get_diagnostics("file:///test.ml")
        assert len(notifs) > 0

    def test_close_document(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        self.client.close_document("file:///test.ml")
        assert self.server.documents.get("file:///test.ml") is None

    def test_close_clears_diagnostics(self):
        self.client.open_document("file:///test.ml", "x = 1;")
        self.client.clear_notifications()
        self.client.close_document("file:///test.ml")
        notifs = self.client.get_diagnostics("file:///test.ml")
        assert len(notifs) > 0
        assert notifs[-1]["params"]["diagnostics"] == []


# ============================================================
# LSP Server Completion Tests
# ============================================================

class TestLSPCompletion:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_completion_returns_items(self):
        self.client.open_document("file:///test.ml", "let x = 1;\n")
        result = self.client.completion("file:///test.ml", 1, 0)
        assert 'items' in result
        assert len(result['items']) > 0

    def test_completion_includes_keywords(self):
        self.client.open_document("file:///test.ml", "")
        result = self.client.completion("file:///test.ml", 0, 0)
        labels = [i['label'] for i in result['items']]
        assert 'let' in labels
        assert 'fn' in labels

    def test_completion_includes_variables(self):
        self.client.open_document("file:///test.ml", "let myVar = 42;\n")
        result = self.client.completion("file:///test.ml", 1, 0)
        labels = [i['label'] for i in result['items']]
        assert 'myVar' in labels

    def test_completion_no_document(self):
        result = self.client.completion("file:///missing.ml", 0, 0)
        assert result['items'] == []


# ============================================================
# LSP Server Hover Tests
# ============================================================

class TestLSPHover:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_hover_variable(self):
        self.client.open_document("file:///test.ml", "let x = 42;")
        result = self.client.hover("file:///test.ml", 0, 4)
        assert result is not None
        assert 'int' in result['contents']

    def test_hover_keyword(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        result = self.client.hover("file:///test.ml", 0, 0)
        assert result is not None

    def test_hover_no_info(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        result = self.client.hover("file:///test.ml", 0, 3)  # space
        assert result is None

    def test_hover_no_document(self):
        result = self.client.hover("file:///missing.ml", 0, 0)
        assert result is None

    def test_hover_string_variable(self):
        self.client.open_document("file:///test.ml", 'let s = "hello";')
        result = self.client.hover("file:///test.ml", 0, 4)
        assert result is not None
        assert 'string' in result['contents']

    def test_hover_bool_variable(self):
        self.client.open_document("file:///test.ml", "let b = true;")
        result = self.client.hover("file:///test.ml", 0, 4)
        assert result is not None
        assert 'bool' in result['contents']


# ============================================================
# LSP Server Definition Tests
# ============================================================

class TestLSPDefinition:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_definition_variable(self):
        self.client.open_document("file:///test.ml", "let x = 1;\nprint(x);")
        result = self.client.definition("file:///test.ml", 1, 6)
        assert result is not None
        assert result['range']['start']['line'] == 0

    def test_definition_function(self):
        self.client.open_document("file:///test.ml", "fn foo() { return 1; }\nfoo();")
        result = self.client.definition("file:///test.ml", 1, 0)
        assert result is not None
        assert result['range']['start']['line'] == 0

    def test_definition_not_found(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        result = self.client.definition("file:///test.ml", 0, 8)
        assert result is None

    def test_definition_no_document(self):
        result = self.client.definition("file:///missing.ml", 0, 0)
        assert result is None


# ============================================================
# LSP Server Document Symbols Tests
# ============================================================

class TestLSPDocumentSymbols:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_symbols_returned(self):
        self.client.open_document("file:///test.ml", "let x = 1;\nfn foo() { return 1; }")
        result = self.client.document_symbols("file:///test.ml")
        assert len(result) >= 2

    def test_symbol_kinds(self):
        self.client.open_document("file:///test.ml", "let x = 1;\nfn foo() { return 1; }")
        result = self.client.document_symbols("file:///test.ml")
        kinds = {s['name']: s['kind'] for s in result}
        assert kinds.get('x') == SYMBOL_VARIABLE
        assert kinds.get('foo') == SYMBOL_FUNCTION

    def test_empty_document(self):
        self.client.open_document("file:///test.ml", "")
        result = self.client.document_symbols("file:///test.ml")
        assert result == []

    def test_no_document(self):
        result = self.client.document_symbols("file:///missing.ml")
        assert result == []


# ============================================================
# LSP Server Signature Help Tests
# ============================================================

class TestLSPSignatureHelp:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_signature_help_print(self):
        self.client.open_document("file:///test.ml", "print(")
        result = self.client.signature_help("file:///test.ml", 0, 6)
        assert result is not None
        assert len(result['signatures']) > 0

    def test_signature_help_user_fn(self):
        self.client.open_document("file:///test.ml", "fn add(a, b) { return a + b; }\nadd(")
        result = self.client.signature_help("file:///test.ml", 1, 4)
        assert result is not None

    def test_signature_help_no_context(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        result = self.client.signature_help("file:///test.ml", 0, 5)
        assert result is None

    def test_signature_help_no_document(self):
        result = self.client.signature_help("file:///missing.ml", 0, 0)
        assert result is None


# ============================================================
# Integration / End-to-End Tests
# ============================================================

class TestIntegration:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_full_workflow(self):
        """Open, edit, get diagnostics, get completions, close."""
        uri = "file:///workflow.ml"

        # Open with valid code
        self.client.open_document(uri, "let x = 1;\nlet y = 2;\n")
        diags = self.client.get_diagnostics(uri)
        assert diags[-1]["params"]["diagnostics"] == []

        # Edit to introduce error
        self.client.clear_notifications()
        self.client.change_document(uri, "x = 1;\nlet y = 2;\n", version=1)
        diags = self.client.get_diagnostics(uri)
        assert len(diags[-1]["params"]["diagnostics"]) > 0

        # Get completions
        result = self.client.completion(uri, 2, 0)
        labels = [i['label'] for i in result['items']]
        assert 'y' in labels

        # Fix the error
        self.client.clear_notifications()
        self.client.change_document(uri, "let x = 1;\nlet y = 2;\n", version=2)
        diags = self.client.get_diagnostics(uri)
        assert diags[-1]["params"]["diagnostics"] == []

        # Close
        self.client.close_document(uri)
        assert self.server.documents.get(uri) is None

    def test_multiple_documents(self):
        """Test with multiple open documents."""
        self.client.open_document("file:///a.ml", "let a = 1;")
        self.client.open_document("file:///b.ml", "let b = 2;")

        ha = self.client.hover("file:///a.ml", 0, 4)
        hb = self.client.hover("file:///b.ml", 0, 4)
        assert ha is not None
        assert hb is not None

    def test_function_hover_and_definition(self):
        """Hover on function call and navigate to definition."""
        uri = "file:///funcs.ml"
        self.client.open_document(uri, "fn greet() { print(1); }\ngreet();")

        # Hover on "greet" at call site
        hover = self.client.hover(uri, 1, 0)
        assert hover is not None

        # Definition should point to line 0
        defn = self.client.definition(uri, 1, 0)
        assert defn is not None
        assert defn['range']['start']['line'] == 0

    def test_diagnostics_multiline(self):
        """Type errors reported on correct lines."""
        uri = "file:///multi.ml"
        code = "let x = 1;\nlet y = true;\nif (x) { print(1); }"
        self.client.open_document(uri, code)
        notifs = self.client.get_diagnostics(uri)
        diags = notifs[-1]["params"]["diagnostics"]
        # "if (x)" should report condition must be bool on line 2 (0-based)
        assert any(d['range']['start']['line'] == 2 for d in diags)

    def test_completion_after_function_def(self):
        """Variables defined after function should appear in completions."""
        uri = "file:///scope.ml"
        code = "fn foo() { return 1; }\nlet bar = foo();\n"
        self.client.open_document(uri, code)
        result = self.client.completion(uri, 2, 0)
        labels = [i['label'] for i in result['items']]
        assert 'foo' in labels
        assert 'bar' in labels

    def test_raw_protocol_roundtrip(self):
        """Test raw Content-Length protocol handling."""
        # Send initialize via raw
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        raw = encode_message(msg)
        responses = self.server.handle_raw(raw)
        assert len(responses) == 1

        # Decode response
        cl, pos = decode_header(responses[0])
        body = json.loads(responses[0][pos:pos + cl])
        assert body["id"] == 1
        assert "result" in body
        assert "capabilities" in body["result"]

    def test_type_inference_in_hover(self):
        """Hover shows inferred types correctly."""
        uri = "file:///infer.ml"
        self.client.open_document(uri, 'let s = "hello";\nlet n = 42;\nlet b = true;')
        hs = self.client.hover(uri, 0, 4)
        assert 'string' in hs['contents']
        hn = self.client.hover(uri, 1, 4)
        assert 'int' in hn['contents']
        hb = self.client.hover(uri, 2, 4)
        assert 'bool' in hb['contents']

    def test_float_type_hover(self):
        """Float type is reported correctly."""
        uri = "file:///float.ml"
        self.client.open_document(uri, "let f = 3.14;")
        h = self.client.hover(uri, 0, 4)
        assert h is not None
        assert 'float' in h['contents']

    def test_symbol_detail_has_type(self):
        """Document symbols include type detail."""
        uri = "file:///sym.ml"
        self.client.open_document(uri, "let x = 42;")
        syms = self.client.document_symbols(uri)
        x_sym = [s for s in syms if s['name'] == 'x']
        assert len(x_sym) == 1
        assert 'detail' in x_sym[0]
        assert 'int' in x_sym[0]['detail']

    def test_lex_error_diagnostic(self):
        """Lex errors produce diagnostics."""
        uri = "file:///lex.ml"
        # Unterminated string may cause a lex error
        self.client.open_document(uri, '"unterminated')
        notifs = self.client.get_diagnostics(uri)
        # Should have some diagnostic (lex or parse error)
        assert len(notifs) > 0

    def test_save_triggers_analysis(self):
        """didSave with text re-analyzes."""
        uri = "file:///save.ml"
        self.client.open_document(uri, "let x = 1;")
        self.client.clear_notifications()
        # Simulate save with text
        self.client.notify("textDocument/didSave", {
            "textDocument": {"uri": uri},
            "text": "y = 1;",
        })
        notifs = self.client.get_diagnostics(uri)
        assert len(notifs) > 0
        assert len(notifs[-1]["params"]["diagnostics"]) > 0


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    def setup_method(self):
        self.server = LSPServer()
        self.client = LSPClient(self.server)
        self.client.initialize()
        self.client.initialized()

    def test_empty_document(self):
        self.client.open_document("file:///empty.ml", "")
        result = self.client.completion("file:///empty.ml", 0, 0)
        assert 'items' in result

    def test_whitespace_only(self):
        self.client.open_document("file:///ws.ml", "   \n\n  ")
        diags = self.client.get_diagnostics("file:///ws.ml")
        # Should not crash
        assert len(diags) > 0

    def test_very_long_line(self):
        long_code = "let x = " + "1 + " * 100 + "1;"
        self.client.open_document("file:///long.ml", long_code)
        # Should not crash
        result = self.client.hover("file:///long.ml", 0, 4)
        assert result is not None

    def test_nested_functions(self):
        code = "fn outer() { fn inner() { return 1; } return inner(); }"
        self.client.open_document("file:///nest.ml", code)
        syms = self.client.document_symbols("file:///nest.ml")
        names = [s['name'] for s in syms]
        assert 'outer' in names

    def test_unicode_in_strings(self):
        code = 'let s = "hello world";'
        self.client.open_document("file:///uni.ml", code)
        h = self.client.hover("file:///uni.ml", 0, 4)
        assert h is not None

    def test_change_empty_contentchanges(self):
        """Change with no content changes should not crash."""
        self.client.open_document("file:///test.ml", "let x = 1;")
        self.client.notify("textDocument/didChange", {
            "textDocument": {"uri": "file:///test.ml", "version": 1},
            "contentChanges": [],
        })
        # Should still have original doc
        doc = self.server.documents.get("file:///test.ml")
        assert doc.text == "let x = 1;"

    def test_hover_on_last_char_of_line(self):
        self.client.open_document("file:///test.ml", "let x = 1;")
        result = self.client.hover("file:///test.ml", 0, 10)  # semicolon
        # May or may not have hover, but should not crash
        assert True

    def test_completion_on_second_line(self):
        code = "let first = 1;\nlet second = 2;\n"
        self.client.open_document("file:///test.ml", code)
        result = self.client.completion("file:///test.ml", 2, 0)
        labels = [i['label'] for i in result['items']]
        assert 'first' in labels
        assert 'second' in labels


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
