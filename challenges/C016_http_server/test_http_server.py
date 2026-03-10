"""Tests for C016: HTTP Server."""

import json
import socket
import threading
import time
import pytest
from http_server import (
    Request, Response, CaseInsensitiveDict, Route, Router, HTTPServer,
    parse_request, ParseError, guess_content_type, STATUS_PHRASES,
)


# ===== CaseInsensitiveDict =====

class TestCaseInsensitiveDict:
    def test_basic_set_get(self):
        d = CaseInsensitiveDict()
        d["Content-Type"] = "text/html"
        assert d["content-type"] == "text/html"
        assert d["CONTENT-TYPE"] == "text/html"

    def test_contains(self):
        d = CaseInsensitiveDict({"Host": "example.com"})
        assert "host" in d
        assert "HOST" in d
        assert "Host" in d
        assert "missing" not in d

    def test_get_default(self):
        d = CaseInsensitiveDict()
        assert d.get("missing") is None
        assert d.get("missing", "default") == "default"

    def test_overwrite(self):
        d = CaseInsensitiveDict()
        d["Content-Type"] = "text/html"
        d["content-type"] = "application/json"
        assert d["Content-Type"] == "application/json"
        assert len(d) == 1

    def test_init_from_list(self):
        d = CaseInsensitiveDict([("Host", "example.com"), ("Accept", "text/html")])
        assert d["host"] == "example.com"
        assert d["accept"] == "text/html"

    def test_items_keys_values(self):
        d = CaseInsensitiveDict({"A": "1", "B": "2"})
        assert len(d.items()) == 2
        assert len(d.keys()) == 2
        assert len(d.values()) == 2

    def test_copy_from_another(self):
        d1 = CaseInsensitiveDict({"Host": "example.com"})
        d2 = CaseInsensitiveDict(d1)
        assert d2["host"] == "example.com"


# ===== HTTP Parsing =====

class TestParseRequest:
    def test_simple_get(self):
        raw = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req, rest = parse_request(raw)
        assert req is not None
        assert req.method == "GET"
        assert req.path == "/"
        assert req.version == "HTTP/1.1"
        assert req.headers["Host"] == "localhost"
        assert rest == b""

    def test_post_with_body(self):
        body = b'{"key":"value"}'
        raw = (
            b"POST /api/data HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"\r\n" + body
        )
        req, rest = parse_request(raw)
        assert req.method == "POST"
        assert req.path == "/api/data"
        assert req.body == body
        assert req.json() == {"key": "value"}

    def test_query_string_single(self):
        raw = b"GET /search?q=hello HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.query["q"] == "hello"
        assert req.path == "/search"

    def test_query_string_multiple(self):
        raw = b"GET /search?q=hello&page=2 HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.query["q"] == "hello"
        assert req.query["page"] == "2"

    def test_query_string_multi_value(self):
        raw = b"GET /filter?tag=a&tag=b HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.query["tag"] == ["a", "b"]

    def test_url_encoding(self):
        raw = b"GET /path%20with%20spaces HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.path == "/path with spaces"

    def test_incomplete_headers(self):
        raw = b"GET / HTTP/1.1\r\nHost: local"
        req, rest = parse_request(raw)
        assert req is None
        assert rest == raw

    def test_incomplete_body(self):
        raw = b"POST /data HTTP/1.1\r\nContent-Length: 100\r\n\r\nshort"
        req, rest = parse_request(raw)
        assert req is None

    def test_malformed_request_line(self):
        raw = b"BADLINE\r\n\r\n"
        with pytest.raises(ParseError):
            parse_request(raw)

    def test_invalid_method(self):
        raw = b"BREW /coffee HTTP/1.1\r\n\r\n"
        with pytest.raises(ParseError):
            parse_request(raw)

    def test_invalid_version(self):
        raw = b"GET / XTTP/1.1\r\n\r\n"
        with pytest.raises(ParseError):
            parse_request(raw)

    def test_headers_too_large(self):
        raw = b"GET / HTTP/1.1\r\n" + b"X-Big: " + b"A" * 70000 + b"\r\n"
        with pytest.raises(ParseError, match="Headers too large"):
            parse_request(raw)

    def test_multiple_requests_in_buffer(self):
        req1 = b"GET /first HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req2 = b"GET /second HTTP/1.1\r\nHost: localhost\r\n\r\n"
        r, rest = parse_request(req1 + req2)
        assert r.path == "/first"
        r2, rest2 = parse_request(rest)
        assert r2.path == "/second"

    def test_chunked_request_body(self):
        raw = (
            b"POST /upload HTTP/1.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"5\r\nHello\r\n"
            b"6\r\n World\r\n"
            b"0\r\n\r\n"
        )
        req, rest = parse_request(raw)
        assert req is not None
        assert req.body == b"Hello World"

    def test_chunked_incomplete(self):
        raw = (
            b"POST /upload HTTP/1.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"5\r\nHel"
        )
        req, _ = parse_request(raw)
        assert req is None

    def test_content_type_property(self):
        raw = b"POST /api HTTP/1.1\r\nContent-Type: application/json\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.content_type == "application/json"

    def test_content_length_property(self):
        raw = b"POST /api HTTP/1.1\r\nContent-Length: 42\r\n\r\n" + b"x" * 42
        req, _ = parse_request(raw)
        assert req.content_length == 42

    def test_no_content_length(self):
        raw = b"GET / HTTP/1.1\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.content_length is None

    def test_delete_method(self):
        raw = b"DELETE /users/5 HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.method == "DELETE"

    def test_put_method(self):
        body = b"updated"
        raw = b"PUT /item/3 HTTP/1.1\r\nContent-Length: 7\r\n\r\n" + body
        req, _ = parse_request(raw)
        assert req.method == "PUT"
        assert req.body == b"updated"

    def test_options_method(self):
        raw = b"OPTIONS /api HTTP/1.1\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.method == "OPTIONS"

    def test_patch_method(self):
        raw = b"PATCH /item/1 HTTP/1.1\r\nContent-Length: 3\r\n\r\nfoo"
        req, _ = parse_request(raw)
        assert req.method == "PATCH"
        assert req.body == b"foo"


# ===== Response =====

class TestResponse:
    def test_text_response(self):
        r = Response.text("Hello")
        assert r.status == 200
        assert r.body == b"Hello"
        assert r.headers["Content-Type"] == "text/plain; charset=utf-8"

    def test_html_response(self):
        r = Response.html("<h1>Hi</h1>")
        assert r.headers["Content-Type"] == "text/html; charset=utf-8"

    def test_json_response(self):
        r = Response.json_response({"status": "ok"})
        assert r.status == 200
        data = json.loads(r.body)
        assert data["status"] == "ok"
        assert r.headers["Content-Type"] == "application/json"

    def test_redirect(self):
        r = Response.redirect("/new-location")
        assert r.status == 302
        assert r.headers["Location"] == "/new-location"

    def test_redirect_301(self):
        r = Response.redirect("/new", 301)
        assert r.status == 301

    def test_custom_status(self):
        r = Response(201, b"Created")
        assert r.status == 201

    def test_serialize_basic(self):
        r = Response.text("OK")
        data = r.serialize()
        assert data.startswith(b"HTTP/1.1 200 OK\r\n")
        assert b"Content-Length: 2\r\n" in data
        assert data.endswith(b"\r\n\r\nOK")

    def test_serialize_includes_date(self):
        r = Response.text("test")
        data = r.serialize()
        assert b"Date:" in data

    def test_serialize_keep_alive(self):
        r = Response.text("test")
        data = r.serialize(keep_alive=True)
        assert b"Connection: keep-alive" in data

    def test_serialize_close(self):
        r = Response.text("test")
        data = r.serialize(keep_alive=False)
        assert b"Connection: close" in data

    def test_chunked_response(self):
        r = Response(200)
        r.headers["Content-Type"] = "text/plain"
        r.add_chunk("Hello")
        r.add_chunk(" World")
        data = r.serialize()
        assert b"Transfer-Encoding: chunked" in data
        assert b"5\r\nHello\r\n" in data
        assert b"6\r\n World\r\n" in data
        assert b"0\r\n\r\n" in data

    def test_204_no_content_length(self):
        r = Response(204)
        data = r.serialize()
        # 204 should not have Content-Length
        assert b"Content-Length" not in data

    def test_string_body_auto_encode(self):
        r = Response(200, "hello")
        assert r.body == b"hello"

    def test_status_phrases(self):
        for code, phrase in STATUS_PHRASES.items():
            r = Response(code)
            data = r.serialize()
            assert f"{code} {phrase}".encode() in data


# ===== Route Matching =====

class TestRoute:
    def test_exact_match(self):
        r = Route("GET", "/users", lambda req: None)
        assert r.match("GET", "/users") == {}
        assert r.match("POST", "/users") is None
        assert r.match("GET", "/other") is None

    def test_param_match(self):
        r = Route("GET", "/users/:id", lambda req: None)
        result = r.match("GET", "/users/42")
        assert result == {"id": "42"}

    def test_multi_param(self):
        r = Route("GET", "/users/:uid/posts/:pid", lambda req: None)
        result = r.match("GET", "/users/1/posts/5")
        assert result == {"uid": "1", "pid": "5"}

    def test_wildcard(self):
        r = Route("GET", "/files/*", lambda req: None)
        result = r.match("GET", "/files/a/b/c")
        assert result == {"_wildcard": "a/b/c"}

    def test_wildcard_single(self):
        r = Route("GET", "/files/*", lambda req: None)
        result = r.match("GET", "/files/test.txt")
        assert result == {"_wildcard": "test.txt"}

    def test_no_match_too_short(self):
        r = Route("GET", "/a/b/c", lambda req: None)
        assert r.match("GET", "/a/b") is None

    def test_no_match_too_long(self):
        r = Route("GET", "/a", lambda req: None)
        assert r.match("GET", "/a/b") is None

    def test_wildcard_method(self):
        r = Route("*", "/health", lambda req: None)
        assert r.match("GET", "/health") == {}
        assert r.match("POST", "/health") == {}

    def test_root_path(self):
        r = Route("GET", "/", lambda req: None)
        assert r.match("GET", "/") == {}

    def test_trailing_slash_normalize(self):
        r = Route("GET", "/users", lambda req: None)
        # trailing slash on route pattern
        r2 = Route("GET", "/users/", lambda req: None)
        assert r.match("GET", "/users") == {}
        assert r2.match("GET", "/users") == {}


# ===== Router =====

class TestRouter:
    def test_basic_routing(self):
        router = Router()
        router.get("/hello", lambda req: Response.text("Hello!"))
        req = Request("GET", "/hello", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.status == 200
        assert resp.body == b"Hello!"

    def test_404_not_found(self):
        router = Router()
        req = Request("GET", "/missing", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.status == 404

    def test_405_method_not_allowed(self):
        router = Router()
        router.get("/data", lambda req: Response.text("ok"))
        req = Request("POST", "/data", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.status == 405
        assert "GET" in resp.headers["Allow"]

    def test_path_params(self):
        router = Router()
        router.get("/users/:id", lambda req: Response.text(f"User {req.params['id']}"))
        req = Request("GET", "/users/42", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.body == b"User 42"

    def test_multiple_routes(self):
        router = Router()
        router.get("/a", lambda req: Response.text("A"))
        router.get("/b", lambda req: Response.text("B"))
        router.post("/a", lambda req: Response.text("Posted A"))

        r1 = router.handle(Request("GET", "/a", "HTTP/1.1", [], b""))
        assert r1.body == b"A"
        r2 = router.handle(Request("GET", "/b", "HTTP/1.1", [], b""))
        assert r2.body == b"B"
        r3 = router.handle(Request("POST", "/a", "HTTP/1.1", [], b""))
        assert r3.body == b"Posted A"

    def test_middleware_passthrough(self):
        router = Router()
        calls = []
        def mw(req, next_fn):
            calls.append("before")
            result = next_fn()
            calls.append("after")
            return result
        router.use(mw)
        router.get("/test", lambda req: Response.text("ok"))
        resp = router.handle(Request("GET", "/test", "HTTP/1.1", [], b""))
        assert resp.status == 200
        assert calls == ["before", "after"]

    def test_middleware_short_circuit(self):
        router = Router()
        def auth_mw(req, next_fn):
            if req.headers.get("Authorization") is None:
                return Response.text("Unauthorized", 401)
            return next_fn()
        router.use(auth_mw)
        router.get("/secret", lambda req: Response.text("classified"))

        # Without auth
        req1 = Request("GET", "/secret", "HTTP/1.1", [], b"")
        assert router.handle(req1).status == 401

        # With auth
        req2 = Request("GET", "/secret", "HTTP/1.1", [("Authorization", "Bearer x")], b"")
        assert router.handle(req2).status == 200

    def test_error_handler(self):
        router = Router()
        def bad_handler(req):
            raise ValueError("oops")
        def err_handler(req, err):
            return Response.json_response({"error": str(err)}, 500)
        router.get("/crash", bad_handler)
        router.on_error(err_handler)
        resp = router.handle(Request("GET", "/crash", "HTTP/1.1", [], b""))
        assert resp.status == 500
        assert b"oops" in resp.body

    def test_error_without_handler(self):
        router = Router()
        router.get("/crash", lambda req: 1/0)
        resp = router.handle(Request("GET", "/crash", "HTTP/1.1", [], b""))
        assert resp.status == 500

    def test_static_files(self):
        router = Router()
        router.static("/static", {
            "/style.css": (b"body { color: red; }", "text/css"),
            "/app.js": (b"console.log('hi');", "application/javascript"),
        })
        req = Request("GET", "/static/style.css", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.status == 200
        assert resp.body == b"body { color: red; }"
        assert resp.headers["Content-Type"] == "text/css"

    def test_static_file_not_found(self):
        router = Router()
        router.static("/static", {"/index.html": (b"<h1>Hi</h1>", "text/html")})
        req = Request("GET", "/static/missing.html", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.status == 404

    def test_post_json(self):
        router = Router()
        def create_user(req):
            data = req.json()
            return Response.json_response({"id": 1, "name": data["name"]}, 201)
        router.post("/users", create_user)

        body = json.dumps({"name": "Alice"}).encode()
        req = Request("POST", "/users", "HTTP/1.1",
                      [("Content-Type", "application/json")], body)
        resp = router.handle(req)
        assert resp.status == 201
        data = json.loads(resp.body)
        assert data["name"] == "Alice"

    def test_put_update(self):
        router = Router()
        router.put("/items/:id", lambda req: Response.json_response(
            {"id": req.params["id"], "updated": True}
        ))
        req = Request("PUT", "/items/7", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        data = json.loads(resp.body)
        assert data["id"] == "7"
        assert data["updated"] is True

    def test_delete_route(self):
        router = Router()
        router.delete("/items/:id", lambda req: Response(204))
        req = Request("DELETE", "/items/3", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.status == 204

    def test_wildcard_route(self):
        router = Router()
        router.get("/files/*", lambda req: Response.text(req.params["_wildcard"]))
        req = Request("GET", "/files/docs/readme.md", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.body == b"docs/readme.md"

    def test_route_priority_first_match(self):
        router = Router()
        router.get("/items/:id", lambda req: Response.text("param"))
        router.get("/items/special", lambda req: Response.text("exact"))
        # First match wins
        req = Request("GET", "/items/special", "HTTP/1.1", [], b"")
        resp = router.handle(req)
        assert resp.body == b"param"  # param route registered first

    def test_middleware_chain(self):
        router = Router()
        order = []
        def mw1(req, next_fn):
            order.append("mw1-before")
            r = next_fn()
            order.append("mw1-after")
            return r
        def mw2(req, next_fn):
            order.append("mw2-before")
            r = next_fn()
            order.append("mw2-after")
            return r
        router.use(mw1)
        router.use(mw2)
        router.get("/test", lambda req: Response.text("ok"))
        router.handle(Request("GET", "/test", "HTTP/1.1", [], b""))
        assert order == ["mw1-before", "mw2-before", "mw2-after", "mw1-after"]


# ===== Content Type Detection =====

class TestContentType:
    def test_html(self):
        assert guess_content_type("index.html") == "text/html"

    def test_css(self):
        assert guess_content_type("style.css") == "text/css"

    def test_js(self):
        assert guess_content_type("app.js") == "application/javascript"

    def test_json(self):
        assert guess_content_type("data.json") == "application/json"

    def test_png(self):
        assert guess_content_type("image.png") == "image/png"

    def test_unknown(self):
        assert guess_content_type("file.xyz") == "application/octet-stream"

    def test_svg(self):
        assert guess_content_type("icon.svg") == "image/svg+xml"

    def test_pdf(self):
        assert guess_content_type("doc.pdf") == "application/pdf"


# ===== Integration: Live Server Tests =====

class TestHTTPServerIntegration:
    """Tests that start a real server and make socket-level requests."""

    def _make_server(self):
        router = Router()
        router.get("/", lambda req: Response.text("Home"))
        router.get("/echo/:msg", lambda req: Response.text(req.params["msg"]))
        router.post("/json", lambda req: Response.json_response(req.json()))
        router.get("/chunked", self._chunked_handler)
        router.get("/headers", lambda req: Response.json_response(
            {"host": req.headers.get("Host", "none")}
        ))
        router.get("/query", lambda req: Response.json_response(req.query))
        router.get("/error", lambda req: (_ for _ in ()).throw(ValueError("boom")))
        router.on_error(lambda req, e: Response.text(str(e), 500))
        router.static("/static", {
            "/test.txt": (b"static content", "text/plain"),
        })
        server = HTTPServer(router, port=0)
        server.start()
        return server

    def _chunked_handler(self, req):
        r = Response(200)
        r.headers["Content-Type"] = "text/plain"
        r.add_chunk("chunk1")
        r.add_chunk("chunk2")
        return r

    def _raw_request(self, port, request_bytes, recv_size=4096):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect(("127.0.0.1", port))
        s.sendall(request_bytes)
        response = b""
        while True:
            try:
                chunk = s.recv(recv_size)
                if not chunk:
                    break
                response += chunk
            except socket.timeout:
                break
        s.close()
        return response

    def _parse_response(self, raw):
        """Parse raw response bytes into (status, headers_dict, body)."""
        head_end = raw.find(b"\r\n\r\n")
        if head_end == -1:
            return None, {}, raw
        head = raw[:head_end].decode("utf-8")
        body = raw[head_end + 4:]
        lines = head.split("\r\n")
        status_line = lines[0]
        status = int(status_line.split(" ")[1])
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
        return status, headers, body

    def test_simple_get(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status, headers, body = self._parse_response(raw)
            assert status == 200
            assert body == b"Home"
        finally:
            server.stop()

    def test_path_params(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET /echo/hello HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status, _, body = self._parse_response(raw)
            assert status == 200
            assert body == b"hello"
        finally:
            server.stop()

    def test_post_json(self):
        server = self._make_server()
        try:
            payload = b'{"x":1}'
            req = (
                b"POST /json HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
                b"Connection: close\r\n"
                b"\r\n" + payload
            )
            raw = self._raw_request(server.port, req)
            status, _, body = self._parse_response(raw)
            assert status == 200
            assert json.loads(body) == {"x": 1}
        finally:
            server.stop()

    def test_404(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET /nope HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status, _, _ = self._parse_response(raw)
            assert status == 404
        finally:
            server.stop()

    def test_static_file(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET /static/test.txt HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status, headers, body = self._parse_response(raw)
            assert status == 200
            assert body == b"static content"
            assert headers["Content-Type"] == "text/plain"
        finally:
            server.stop()

    def test_query_string(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET /query?name=alice&age=30 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status, _, body = self._parse_response(raw)
            assert status == 200
            data = json.loads(body)
            assert data["name"] == "alice"
            assert data["age"] == "30"
        finally:
            server.stop()

    def test_custom_headers(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET /headers HTTP/1.1\r\nHost: myhost.com\r\nConnection: close\r\n\r\n")
            status, _, body = self._parse_response(raw)
            assert status == 200
            data = json.loads(body)
            assert data["host"] == "myhost.com"
        finally:
            server.stop()

    def test_error_handling(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET /error HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status, _, body = self._parse_response(raw)
            assert status == 500
        finally:
            server.stop()

    def test_head_method(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"HEAD / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            status, headers, body = self._parse_response(raw)
            assert status == 200
            # HEAD should not have a body
            assert body == b""
        finally:
            server.stop()

    def test_keep_alive_multiple_requests(self):
        """Test multiple requests on same connection."""
        server = self._make_server()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect(("127.0.0.1", server.port))

            # First request
            s.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            time.sleep(0.1)
            # Second request on same connection
            s.sendall(b"GET /echo/world HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")

            response = b""
            while True:
                try:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                except socket.timeout:
                    break
            s.close()

            # Should contain both responses
            assert b"Home" in response
            assert b"world" in response
        finally:
            server.stop()

    def test_chunked_response(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"GET /chunked HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            assert b"Transfer-Encoding: chunked" in raw
            # Should contain our chunk data
            assert b"chunk1" in raw
            assert b"chunk2" in raw
        finally:
            server.stop()

    def test_malformed_request(self):
        server = self._make_server()
        try:
            raw = self._raw_request(server.port, b"BADREQUEST\r\n\r\n")
            status, _, _ = self._parse_response(raw)
            assert status == 400
        finally:
            server.stop()

    def test_concurrent_requests(self):
        server = self._make_server()
        results = []
        errors = []

        def make_request(msg):
            try:
                raw = self._raw_request(
                    server.port,
                    f"GET /echo/{msg} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
                )
                status, _, body = self._parse_response(raw)
                results.append((msg, status, body))
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=make_request, args=(f"msg{i}",)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert len(errors) == 0
            assert len(results) == 10
            for msg, status, body in results:
                assert status == 200
                assert body == msg.encode()
        finally:
            server.stop()

    def test_request_count(self):
        server = self._make_server()
        try:
            for i in range(3):
                self._raw_request(server.port, b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            time.sleep(0.2)
            assert server.request_count >= 3
        finally:
            server.stop()


# ===== Edge Cases =====

class TestEdgeCases:
    def test_empty_body_json(self):
        req = Request("GET", "/", "HTTP/1.1", [], b"")
        assert req.json() is None

    def test_request_json_cached(self):
        body = b'{"a":1}'
        req = Request("POST", "/", "HTTP/1.1", [("Content-Type", "application/json")], body)
        j1 = req.json()
        j2 = req.json()
        assert j1 is j2  # same object (cached)

    def test_response_body_string(self):
        r = Response(200, "hello")
        assert isinstance(r.body, bytes)

    def test_route_no_leading_slash(self):
        """Routes should still work without leading slash in path."""
        r = Route("GET", "users", lambda req: None)
        assert r.match("GET", "/users") == {}

    def test_deep_path_params(self):
        r = Route("GET", "/a/:b/c/:d/e/:f", lambda req: None)
        result = r.match("GET", "/a/1/c/2/e/3")
        assert result == {"b": "1", "d": "2", "f": "3"}

    def test_unicode_response_body(self):
        r = Response.text("Hello \u00e9\u00e8\u00ea")
        assert "utf-8" in r.headers["Content-Type"]
        assert r.body == "Hello \u00e9\u00e8\u00ea".encode("utf-8")

    def test_empty_query_value(self):
        raw = b"GET /search?q= HTTP/1.1\r\nHost: localhost\r\n\r\n"
        req, _ = parse_request(raw)
        assert req.query["q"] == ""

    def test_multiple_middleware_short_circuit(self):
        router = Router()
        order = []
        def mw1(req, next_fn):
            order.append("mw1")
            return Response.text("blocked", 403)  # short circuit
        def mw2(req, next_fn):
            order.append("mw2")  # should never reach
            return next_fn()
        router.use(mw1)
        router.use(mw2)
        router.get("/test", lambda req: Response.text("ok"))
        resp = router.handle(Request("GET", "/test", "HTTP/1.1", [], b""))
        assert resp.status == 403
        assert order == ["mw1"]  # mw2 never called

    def test_response_custom_header(self):
        r = Response.text("ok")
        r.headers["X-Custom"] = "value"
        data = r.serialize()
        assert b"X-Custom: value" in data


# ===== Full API Application Test =====

class TestRESTfulAPI:
    """Tests a complete REST API pattern."""

    def _build_api(self):
        router = Router()
        store = {}
        next_id = [1]

        def list_items(req):
            return Response.json_response(list(store.values()))

        def get_item(req):
            item = store.get(req.params["id"])
            if item is None:
                return Response.text("Not Found", 404)
            return Response.json_response(item)

        def create_item(req):
            data = req.json()
            item_id = str(next_id[0])
            next_id[0] += 1
            item = {"id": item_id, **data}
            store[item_id] = item
            return Response.json_response(item, 201)

        def update_item(req):
            item_id = req.params["id"]
            if item_id not in store:
                return Response.text("Not Found", 404)
            data = req.json()
            store[item_id].update(data)
            return Response.json_response(store[item_id])

        def delete_item(req):
            item_id = req.params["id"]
            if item_id not in store:
                return Response.text("Not Found", 404)
            del store[item_id]
            return Response(204)

        router.get("/items", list_items)
        router.get("/items/:id", get_item)
        router.post("/items", create_item)
        router.put("/items/:id", update_item)
        router.delete("/items/:id", delete_item)

        return router, store

    def test_crud_lifecycle(self):
        router, store = self._build_api()

        # Create
        body = json.dumps({"name": "Widget"}).encode()
        req = Request("POST", "/items", "HTTP/1.1",
                      [("Content-Type", "application/json")], body)
        resp = router.handle(req)
        assert resp.status == 201
        item = json.loads(resp.body)
        assert item["name"] == "Widget"
        assert item["id"] == "1"

        # Read
        resp = router.handle(Request("GET", "/items/1", "HTTP/1.1", [], b""))
        assert resp.status == 200
        assert json.loads(resp.body)["name"] == "Widget"

        # List
        resp = router.handle(Request("GET", "/items", "HTTP/1.1", [], b""))
        assert resp.status == 200
        items = json.loads(resp.body)
        assert len(items) == 1

        # Update
        body = json.dumps({"name": "Updated Widget"}).encode()
        req = Request("PUT", "/items/1", "HTTP/1.1",
                      [("Content-Type", "application/json")], body)
        resp = router.handle(req)
        assert resp.status == 200
        assert json.loads(resp.body)["name"] == "Updated Widget"

        # Delete
        resp = router.handle(Request("DELETE", "/items/1", "HTTP/1.1", [], b""))
        assert resp.status == 204

        # Verify deleted
        resp = router.handle(Request("GET", "/items/1", "HTTP/1.1", [], b""))
        assert resp.status == 404

    def test_create_multiple(self):
        router, store = self._build_api()
        for i in range(5):
            body = json.dumps({"name": f"Item {i}"}).encode()
            req = Request("POST", "/items", "HTTP/1.1",
                          [("Content-Type", "application/json")], body)
            resp = router.handle(req)
            assert resp.status == 201

        resp = router.handle(Request("GET", "/items", "HTTP/1.1", [], b""))
        assert len(json.loads(resp.body)) == 5

    def test_update_nonexistent(self):
        router, _ = self._build_api()
        body = json.dumps({"name": "ghost"}).encode()
        req = Request("PUT", "/items/999", "HTTP/1.1", [], body)
        resp = router.handle(req)
        assert resp.status == 404

    def test_delete_nonexistent(self):
        router, _ = self._build_api()
        resp = router.handle(Request("DELETE", "/items/999", "HTTP/1.1", [], b""))
        assert resp.status == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
