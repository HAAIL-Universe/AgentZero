"""
C016: HTTP Server -- Built from raw sockets.

Features:
- HTTP/1.1 request parsing (method, path, headers, body)
- Response building with proper status codes and headers
- Router with exact, parameterized (/users/:id), and wildcard routes
- Middleware pipeline (before/after hooks)
- Content-Type negotiation and auto-detection
- Chunked transfer encoding (responses)
- Keep-alive connection handling
- Query string parsing
- JSON request/response helpers
- Static file serving (from a virtual filesystem for testability)

Difficulty: 4
Domain: Networking / Protocol / IO
"""

import json
import socket
import threading
import io
from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# HTTP Request
# ---------------------------------------------------------------------------

class Request:
    """Parsed HTTP request."""

    def __init__(self, method, path, version, headers, body, raw_path=None):
        self.method = method.upper()
        self.raw_path = raw_path or path
        parsed = urlparse(path)
        self.path = unquote(parsed.path)
        self.query_string = parsed.query
        self.query = {}
        if parsed.query:
            for k, v in parse_qs(parsed.query, keep_blank_values=True).items():
                self.query[k] = v if len(v) > 1 else v[0]
        self.version = version
        self.headers = CaseInsensitiveDict(headers)
        self.body = body
        self.params = {}  # filled by router for path params
        self._json_cache = None

    def json(self):
        """Parse body as JSON."""
        if self._json_cache is None:
            self._json_cache = json.loads(self.body) if self.body else None
        return self._json_cache

    @property
    def content_type(self):
        return self.headers.get("Content-Type", "")

    @property
    def content_length(self):
        val = self.headers.get("Content-Length")
        return int(val) if val is not None else None


# ---------------------------------------------------------------------------
# HTTP Response
# ---------------------------------------------------------------------------

STATUS_PHRASES = {
    200: "OK", 201: "Created", 204: "No Content",
    301: "Moved Permanently", 302: "Found", 304: "Not Modified",
    400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
    404: "Not Found", 405: "Method Not Allowed",
    500: "Internal Server Error", 502: "Bad Gateway",
}


class Response:
    """HTTP response builder."""

    def __init__(self, status=200, body=b"", headers=None):
        self.status = status
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.body = body
        self.headers = CaseInsensitiveDict(headers or {})
        self.chunked = False
        self._chunks = []

    @staticmethod
    def text(content, status=200):
        r = Response(status, content.encode("utf-8") if isinstance(content, str) else content)
        r.headers["Content-Type"] = "text/plain; charset=utf-8"
        return r

    @staticmethod
    def html(content, status=200):
        r = Response(status, content.encode("utf-8") if isinstance(content, str) else content)
        r.headers["Content-Type"] = "text/html; charset=utf-8"
        return r

    @staticmethod
    def json_response(data, status=200):
        body = json.dumps(data, separators=(",", ":"))
        r = Response(status, body.encode("utf-8"))
        r.headers["Content-Type"] = "application/json"
        return r

    @staticmethod
    def redirect(location, status=302):
        r = Response(status)
        r.headers["Location"] = location
        return r

    def add_chunk(self, data):
        """Add a chunk for chunked transfer encoding."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._chunks.append(data)
        self.chunked = True

    def serialize(self, keep_alive=False):
        """Serialize to bytes for sending over socket."""
        phrase = STATUS_PHRASES.get(self.status, "Unknown")
        lines = [f"HTTP/1.1 {self.status} {phrase}"]

        if "Date" not in self.headers:
            self.headers["Date"] = datetime.now(timezone.utc).strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )

        if self.chunked and self._chunks:
            self.headers["Transfer-Encoding"] = "chunked"
            if keep_alive:
                self.headers["Connection"] = "keep-alive"
            else:
                self.headers["Connection"] = "close"

            for k, v in self.headers.items():
                lines.append(f"{k}: {v}")
            lines.append("")
            lines.append("")
            head = "\r\n".join(lines).encode("utf-8")

            parts = [head]
            for chunk in self._chunks:
                size_line = f"{len(chunk):x}\r\n".encode("utf-8")
                parts.append(size_line)
                parts.append(chunk)
                parts.append(b"\r\n")
            parts.append(b"0\r\n\r\n")
            return b"".join(parts)
        else:
            body = self.body or b""
            if self.status != 204:
                self.headers["Content-Length"] = str(len(body))
            if keep_alive:
                self.headers["Connection"] = "keep-alive"
            else:
                self.headers["Connection"] = "close"

            for k, v in self.headers.items():
                lines.append(f"{k}: {v}")
            lines.append("")
            lines.append("")
            head = "\r\n".join(lines).encode("utf-8")
            return head + body


# ---------------------------------------------------------------------------
# Case-insensitive dict for headers
# ---------------------------------------------------------------------------

class CaseInsensitiveDict:
    """Dict with case-insensitive string keys."""

    def __init__(self, data=None):
        self._store = {}  # lowered_key -> (original_key, value)
        if data:
            if isinstance(data, CaseInsensitiveDict):
                self._store = dict(data._store)
            elif isinstance(data, dict):
                for k, v in data.items():
                    self[k] = v
            else:
                for k, v in data:
                    self[k] = v

    def __setitem__(self, key, value):
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __contains__(self, key):
        return key.lower() in self._store

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        return [(orig_key, v) for _, (orig_key, v) in self._store.items()]

    def keys(self):
        return [orig_key for _, (orig_key, v) in self._store.items()]

    def values(self):
        return [v for _, (k, v) in self._store.items()]

    def __repr__(self):
        return repr(dict(self.items()))

    def __len__(self):
        return len(self._store)


# ---------------------------------------------------------------------------
# HTTP Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when HTTP parsing fails."""
    pass


def parse_request(data):
    """Parse raw bytes into a Request object.

    Returns (Request, remaining_bytes) or raises ParseError.
    Returns (None, data) if not enough data yet.
    """
    # Find end of headers
    header_end = data.find(b"\r\n\r\n")
    if header_end == -1:
        # Check if we have too much header data (protect against abuse)
        if len(data) > 65536:
            raise ParseError("Headers too large")
        return None, data

    header_bytes = data[:header_end]
    rest = data[header_end + 4:]

    # Decode headers
    try:
        header_text = header_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ParseError("Invalid header encoding")

    lines = header_text.split("\r\n")
    if not lines:
        raise ParseError("Empty request")

    # Parse request line
    request_line = lines[0]
    parts = request_line.split(" ")
    if len(parts) < 3:
        raise ParseError(f"Malformed request line: {request_line}")

    method = parts[0]
    path = parts[1]
    version = parts[2]

    valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
    if method.upper() not in valid_methods:
        raise ParseError(f"Unknown method: {method}")

    if not version.startswith("HTTP/"):
        raise ParseError(f"Invalid HTTP version: {version}")

    # Parse headers
    headers = []
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers.append((key.strip(), value.strip()))

    header_dict = CaseInsensitiveDict(headers)

    # Determine body length
    body = b""
    transfer_encoding = header_dict.get("Transfer-Encoding", "").lower()
    content_length = header_dict.get("Content-Length")

    if transfer_encoding == "chunked":
        # Parse chunked body
        body_parts = []
        remaining = rest
        while True:
            line_end = remaining.find(b"\r\n")
            if line_end == -1:
                return None, data  # need more data
            size_str = remaining[:line_end].decode("utf-8").strip()
            if ";" in size_str:
                size_str = size_str.split(";")[0]
            try:
                chunk_size = int(size_str, 16)
            except ValueError:
                raise ParseError(f"Invalid chunk size: {size_str}")

            if chunk_size == 0:
                # End of chunks -- skip trailing \r\n
                trailing = remaining[line_end + 2:]
                if trailing.startswith(b"\r\n"):
                    trailing = trailing[2:]
                body = b"".join(body_parts)
                req = Request(method, path, version, headers, body)
                return req, trailing

            chunk_start = line_end + 2
            chunk_end = chunk_start + chunk_size
            if len(remaining) < chunk_end + 2:
                return None, data  # need more data
            body_parts.append(remaining[chunk_start:chunk_end])
            remaining = remaining[chunk_end + 2:]

    elif content_length is not None:
        try:
            length = int(content_length)
        except ValueError:
            raise ParseError(f"Invalid Content-Length: {content_length}")

        if len(rest) < length:
            return None, data  # need more data
        body = rest[:length]
        rest = rest[length:]
    # else: no body

    req = Request(method, path, version, headers, body)
    return req, rest


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Route:
    """Single route definition."""

    def __init__(self, method, pattern, handler):
        self.method = method.upper()
        self.pattern = pattern
        self.handler = handler
        self.segments = self._parse(pattern)

    def _parse(self, pattern):
        """Parse pattern into segment descriptors."""
        segments = []
        for part in pattern.strip("/").split("/"):
            if not part:
                continue
            if part.startswith(":"):
                segments.append(("param", part[1:]))
            elif part == "*":
                segments.append(("wildcard", None))
            else:
                segments.append(("literal", part))
        return segments

    def match(self, method, path):
        """Try to match a request. Returns params dict or None."""
        if self.method != "*" and self.method != method:
            return None

        path_parts = [p for p in path.strip("/").split("/") if p]

        # Wildcard at end matches anything
        has_wildcard = self.segments and self.segments[-1][0] == "wildcard"

        if has_wildcard:
            check_segs = self.segments[:-1]
            if len(path_parts) < len(check_segs):
                return None
        else:
            if len(path_parts) != len(self.segments):
                return None
            check_segs = self.segments

        params = {}
        for seg, part in zip(check_segs, path_parts):
            kind, name = seg
            if kind == "literal":
                if part != name:
                    return None
            elif kind == "param":
                params[name] = part

        if has_wildcard:
            # Capture remaining path
            remaining = path_parts[len(check_segs):]
            params["_wildcard"] = "/".join(remaining)

        return params


class Router:
    """HTTP request router with middleware support."""

    def __init__(self):
        self.routes = []
        self.middleware = []
        self.error_handler = None
        self._virtual_fs = {}  # path -> (content_bytes, content_type)
        self._static_prefix = None

    def use(self, middleware_fn):
        """Add middleware. Signature: fn(req, res, next) -> Response or None."""
        self.middleware.append(middleware_fn)

    def route(self, method, pattern, handler):
        """Register a route handler. Signature: fn(req) -> Response."""
        self.routes.append(Route(method, pattern, handler))

    def get(self, pattern, handler):
        self.route("GET", pattern, handler)

    def post(self, pattern, handler):
        self.route("POST", pattern, handler)

    def put(self, pattern, handler):
        self.route("PUT", pattern, handler)

    def delete(self, pattern, handler):
        self.route("DELETE", pattern, handler)

    def patch(self, pattern, handler):
        self.route("PATCH", pattern, handler)

    def on_error(self, handler):
        """Set error handler. Signature: fn(req, error) -> Response."""
        self.error_handler = handler

    def static(self, prefix, virtual_files):
        """Serve static files from a virtual filesystem dict.

        virtual_files: {path: (bytes, content_type)}
        """
        self._static_prefix = prefix.rstrip("/")
        self._virtual_fs = virtual_files

    def _find_route(self, method, path):
        """Find matching route. Returns (route, params) or (None, None)."""
        for route in self.routes:
            params = route.match(method, path)
            if params is not None:
                return route, params
        return None, None

    def _check_method_allowed(self, path):
        """Check if any route matches this path with any method."""
        allowed = set()
        for route in self.routes:
            # Try matching with wildcard method
            path_parts = [p for p in path.strip("/").split("/") if p]
            has_wildcard = route.segments and route.segments[-1][0] == "wildcard"
            if has_wildcard:
                check_segs = route.segments[:-1]
                if len(path_parts) < len(check_segs):
                    continue
            else:
                if len(path_parts) != len(route.segments):
                    continue
                check_segs = route.segments

            matched = True
            for seg, part in zip(check_segs, path_parts):
                kind, name = seg
                if kind == "literal" and part != name:
                    matched = False
                    break
            if matched:
                allowed.add(route.method)
        return allowed

    def handle(self, request):
        """Route a request through middleware and handlers."""
        try:
            # Run middleware chain
            idx = 0

            def next_middleware():
                nonlocal idx
                if idx < len(self.middleware):
                    mw = self.middleware[idx]
                    idx += 1
                    result = mw(request, next_middleware)
                    return result
                return None

            mw_result = next_middleware()
            if mw_result is not None:
                return mw_result

            # Check static files
            if self._static_prefix and request.path.startswith(self._static_prefix + "/"):
                file_path = request.path[len(self._static_prefix):]
                if file_path in self._virtual_fs:
                    content, ctype = self._virtual_fs[file_path]
                    r = Response(200, content)
                    r.headers["Content-Type"] = ctype
                    return r

            # Find matching route
            route, params = self._find_route(request.method, request.path)
            if route is None:
                # Check if the path exists with a different method
                allowed = self._check_method_allowed(request.path)
                if allowed:
                    r = Response(405, b"Method Not Allowed")
                    r.headers["Content-Type"] = "text/plain"
                    r.headers["Allow"] = ", ".join(sorted(allowed))
                    return r
                return Response.text("Not Found", 404)

            request.params = params
            return route.handler(request)

        except Exception as e:
            if self.error_handler:
                try:
                    return self.error_handler(request, e)
                except Exception:
                    pass
            return Response.text(f"Internal Server Error", 500)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class HTTPServer:
    """HTTP/1.1 server with keep-alive support."""

    def __init__(self, router, host="127.0.0.1", port=0):
        self.router = router
        self.host = host
        self.port = port
        self._socket = None
        self._running = False
        self._threads = []
        self.request_count = 0
        self._lock = threading.Lock()

    def start(self):
        """Start the server in a background thread."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self.port = self._socket.getsockname()[1]  # get actual port if 0
        self._socket.listen(16)
        self._socket.settimeout(1.0)
        self._running = True

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2)

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._socket.accept()
                t = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
                t.start()
                self._threads.append(t)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_connection(self, conn):
        """Handle a single connection (may serve multiple requests via keep-alive)."""
        conn.settimeout(5.0)
        buffer = b""
        keep_alive = True

        try:
            while keep_alive and self._running:
                # Read data
                try:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                except socket.timeout:
                    break

                # Try to parse a request
                try:
                    request, remaining = parse_request(buffer)
                except ParseError as e:
                    resp = Response.text(str(e), 400)
                    conn.sendall(resp.serialize())
                    break

                if request is None:
                    continue  # need more data

                buffer = remaining

                # Determine keep-alive
                conn_header = request.headers.get("Connection", "").lower()
                if request.version == "HTTP/1.1":
                    keep_alive = conn_header != "close"
                else:
                    keep_alive = conn_header == "keep-alive"

                # Handle HEAD method
                is_head = request.method == "HEAD"
                if is_head:
                    request.method = "GET"

                # Route the request
                response = self.router.handle(request)

                with self._lock:
                    self.request_count += 1

                # Serialize and send
                resp_bytes = response.serialize(keep_alive=keep_alive)
                if is_head:
                    # For HEAD, send headers only
                    head_end = resp_bytes.find(b"\r\n\r\n")
                    if head_end != -1:
                        resp_bytes = resp_bytes[:head_end + 4]

                conn.sendall(resp_bytes)

        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------

MIME_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".pdf": "application/pdf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def guess_content_type(path):
    """Guess MIME type from file extension."""
    for ext, mime in MIME_TYPES.items():
        if path.endswith(ext):
            return mime
    return "application/octet-stream"
