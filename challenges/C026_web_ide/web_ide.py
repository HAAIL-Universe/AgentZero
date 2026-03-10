"""
C026: Web IDE -- Browser-accessible IDE
Composes C016 (HTTP Server) + C024 (IDE)

Wraps the IDE in an HTTP server with REST API and static frontend.
All IDE operations available via JSON API endpoints.
Event polling for live notifications.
"""

import sys, os, json, urllib.parse, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C016_http_server'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C024_ide'))

from http_server import HTTPServer, Router, Request, Response
from ide import IDE, IDEEvent


# ---------------------------------------------------------------------------
# HTML Frontend
# ---------------------------------------------------------------------------

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentZero Web IDE</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Consolas', 'Monaco', monospace; background: #1e1e2e; color: #cdd6f4; display: flex; flex-direction: column; height: 100vh; }
header { background: #181825; padding: 8px 16px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #313244; }
header h1 { font-size: 14px; color: #89b4fa; }
.toolbar { display: flex; gap: 8px; }
.toolbar button { background: #313244; color: #cdd6f4; border: 1px solid #45475a; padding: 4px 12px; cursor: pointer; font-family: inherit; font-size: 12px; border-radius: 4px; }
.toolbar button:hover { background: #45475a; }
.main { display: flex; flex: 1; overflow: hidden; }
.sidebar { width: 220px; background: #181825; border-right: 1px solid #313244; overflow-y: auto; padding: 8px; }
.sidebar h3 { font-size: 11px; color: #a6adc8; text-transform: uppercase; margin-bottom: 8px; }
.file-list { list-style: none; }
.file-list li { padding: 4px 8px; cursor: pointer; font-size: 13px; border-radius: 4px; }
.file-list li:hover { background: #313244; }
.file-list li.active { background: #45475a; color: #89b4fa; }
.content { flex: 1; display: flex; flex-direction: column; }
.editor-area { flex: 1; position: relative; }
.editor-area textarea { width: 100%; height: 100%; background: #1e1e2e; color: #cdd6f4; border: none; padding: 16px; font-family: inherit; font-size: 14px; resize: none; outline: none; tab-size: 2; }
.panels { height: 200px; border-top: 1px solid #313244; display: flex; flex-direction: column; }
.panel-tabs { display: flex; background: #181825; border-bottom: 1px solid #313244; }
.panel-tabs button { background: none; color: #a6adc8; border: none; padding: 6px 16px; cursor: pointer; font-family: inherit; font-size: 12px; border-bottom: 2px solid transparent; }
.panel-tabs button.active { color: #89b4fa; border-bottom-color: #89b4fa; }
.panel-content { flex: 1; overflow-y: auto; padding: 8px 16px; font-size: 13px; white-space: pre-wrap; }
.status-bar { background: #181825; padding: 4px 16px; font-size: 11px; color: #a6adc8; display: flex; justify-content: space-between; border-top: 1px solid #313244; }
.diagnostic { color: #f38ba8; }
.notification { color: #a6e3a1; }
#repl-input { width: 100%; background: #313244; color: #cdd6f4; border: 1px solid #45475a; padding: 4px 8px; font-family: inherit; font-size: 13px; margin-top: 4px; }
</style>
</head>
<body>
<header>
  <h1>AgentZero Web IDE</h1>
  <div class="toolbar">
    <button onclick="createFile()">New File</button>
    <button onclick="saveFile()">Save</button>
    <button onclick="runFile()">Run</button>
    <button onclick="checkFile()">Check</button>
    <button onclick="debugFile()">Debug</button>
  </div>
</header>
<div class="main">
  <div class="sidebar">
    <h3>Files</h3>
    <ul class="file-list" id="file-list"></ul>
    <h3 style="margin-top:16px">Packages</h3>
    <ul class="file-list" id="pkg-list"></ul>
  </div>
  <div class="content">
    <div class="editor-area">
      <textarea id="editor" placeholder="Select or create a file..." spellcheck="false"></textarea>
    </div>
    <div class="panels">
      <div class="panel-tabs">
        <button class="active" onclick="showPanel('output')">Output</button>
        <button onclick="showPanel('terminal')">Terminal</button>
        <button onclick="showPanel('diagnostics')">Diagnostics</button>
        <button onclick="showPanel('debug')">Debug</button>
      </div>
      <div class="panel-content" id="panel-output"></div>
      <div class="panel-content" id="panel-terminal" style="display:none"></div>
      <div class="panel-content" id="panel-diagnostics" style="display:none"></div>
      <div class="panel-content" id="panel-debug" style="display:none"></div>
      <input id="repl-input" placeholder="REPL> type expression and press Enter" onkeydown="if(event.key==='Enter')replEval()">
    </div>
  </div>
</div>
<div class="status-bar">
  <span id="status-left">Ready</span>
  <span id="status-right"></span>
</div>
<script>
const API = '';
let currentFile = null;
let files = [];

async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  return r.json();
}

async function refreshFiles() {
  const data = await api('GET', '/api/files');
  files = data.files || [];
  const ul = document.getElementById('file-list');
  ul.innerHTML = files.map(f =>
    `<li class="${f===currentFile?'active':''}" onclick="openFile('${f}')">${f}</li>`
  ).join('');
}

async function openFile(path) {
  currentFile = path;
  const data = await api('POST', '/api/editor/open', { path });
  document.getElementById('editor').value = data.content || '';
  document.getElementById('status-left').textContent = path;
  refreshFiles();
  if (data.diagnostics) showDiagnostics(data.diagnostics);
}

async function createFile() {
  const name = prompt('File name:', 'new.ml');
  if (!name) return;
  await api('POST', '/api/files', { path: name, content: '' });
  await refreshFiles();
  openFile(name);
}

async function saveFile() {
  if (!currentFile) return;
  const content = document.getElementById('editor').value;
  const data = await api('PUT', '/api/files/' + encodeURIComponent(currentFile), { content });
  if (data.diagnostics) showDiagnostics(data.diagnostics);
  document.getElementById('status-left').textContent = currentFile + ' (saved)';
}

async function runFile() {
  if (!currentFile) return;
  await saveFile();
  const data = await api('POST', '/api/build/run', { path: currentFile });
  const out = document.getElementById('panel-output');
  out.textContent = data.success ? (data.output || 'OK') : ('Error: ' + (data.error || 'unknown'));
  showPanel('output');
}

async function checkFile() {
  if (!currentFile) return;
  await saveFile();
  const data = await api('POST', '/api/build/check', { path: currentFile });
  const out = document.getElementById('panel-output');
  out.textContent = data.success ? 'No errors found.' : ('Error: ' + (data.error || 'unknown'));
  showPanel('output');
}

async function debugFile() {
  if (!currentFile) return;
  await saveFile();
  const data = await api('POST', '/api/debug/start', { path: currentFile });
  const dbg = document.getElementById('panel-debug');
  dbg.textContent = data.success ? JSON.stringify(data.context, null, 2) : ('Error: ' + (data.error || 'unknown'));
  showPanel('debug');
}

async function replEval() {
  const input = document.getElementById('repl-input');
  const source = input.value;
  input.value = '';
  const data = await api('POST', '/api/terminal/eval', { source });
  const term = document.getElementById('panel-terminal');
  term.textContent += '> ' + source + '\n';
  if (data.output) term.textContent += data.output + '\n';
  if (data.result !== undefined && data.result !== null) term.textContent += '= ' + data.result + '\n';
  if (data.error) term.textContent += 'Error: ' + data.error + '\n';
  showPanel('terminal');
}

function showDiagnostics(diags) {
  const el = document.getElementById('panel-diagnostics');
  el.innerHTML = diags.map(d =>
    `<div class="diagnostic">Line ${d.line||'?'}: ${d.message||d}</div>`
  ).join('');
}

function showPanel(name) {
  ['output','terminal','diagnostics','debug'].forEach(p => {
    document.getElementById('panel-' + p).style.display = p === name ? '' : 'none';
  });
  document.querySelectorAll('.panel-tabs button').forEach(b => {
    b.classList.toggle('active', b.textContent.toLowerCase() === name);
  });
}

async function pollNotifications() {
  try {
    const data = await api('GET', '/api/notifications');
    if (data.notifications && data.notifications.length > 0) {
      const right = document.getElementById('status-right');
      right.textContent = data.notifications.map(n => n.event).join(', ');
      setTimeout(() => { right.textContent = ''; }, 3000);
    }
  } catch(e) {}
  setTimeout(pollNotifications, 2000);
}

refreshFiles();
pollNotifications();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Web IDE Server
# ---------------------------------------------------------------------------

class WebIDE:
    """Browser-accessible IDE wrapping C024 IDE with C016 HTTP Server."""

    def __init__(self, project_root="/project", host="127.0.0.1", port=0):
        self.ide = IDE(project_root)
        self.ide.initialize()
        self.router = Router()
        self.host = host
        self.port = port
        self.server = None
        self._lock = threading.Lock()
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Add CORS and JSON error handling middleware."""
        def cors_middleware(request, next_handler):
            if request.method == "OPTIONS":
                resp = Response(status=204)
                resp.headers["Access-Control-Allow-Origin"] = "*"
                resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
                resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
                return resp
            response = next_handler()
            if response:
                response.headers["Access-Control-Allow-Origin"] = "*"
            return response

        self.router.use(cors_middleware)

        def error_handler(request, error):
            return Response.json_response(
                {"error": str(error), "type": type(error).__name__},
                status=500
            )
        self.router.on_error(error_handler)

    def _setup_routes(self):
        """Register all API routes."""
        r = self.router

        # -- Frontend --
        r.get("/", self._serve_frontend)

        # -- Files --
        r.get("/api/files", self._list_files)
        r.post("/api/files", self._create_file)
        r.route("GET", "/api/files/*", self._read_file)
        r.route("PUT", "/api/files/*", self._write_file)
        r.route("DELETE", "/api/files/*", self._delete_file)

        # -- Editor --
        r.post("/api/editor/open", self._editor_open)
        r.post("/api/editor/close", self._editor_close)
        r.post("/api/editor/complete", self._editor_complete)
        r.post("/api/editor/hover", self._editor_hover)
        r.post("/api/editor/definition", self._editor_definition)
        r.post("/api/editor/symbols", self._editor_symbols)
        r.post("/api/editor/signature", self._editor_signature)
        r.route("GET", "/api/editor/diagnostics/*", self._editor_diagnostics)
        r.post("/api/editor/refresh", self._editor_refresh)

        # -- Terminal --
        r.post("/api/terminal/eval", self._terminal_eval)
        r.post("/api/terminal/run", self._terminal_run)
        r.post("/api/terminal/load", self._terminal_load)
        r.get("/api/terminal/variables", self._terminal_variables)
        r.post("/api/terminal/reset", self._terminal_reset)
        r.get("/api/terminal/history", self._terminal_history)

        # -- Debug --
        r.post("/api/debug/start", self._debug_start)
        r.post("/api/debug/stop", self._debug_stop)
        r.post("/api/debug/step", self._debug_step)
        r.post("/api/debug/step-line", self._debug_step_line)
        r.post("/api/debug/step-over", self._debug_step_over)
        r.post("/api/debug/step-out", self._debug_step_out)
        r.post("/api/debug/continue", self._debug_continue)
        r.post("/api/debug/breakpoint", self._debug_add_breakpoint)
        r.delete("/api/debug/breakpoint/:id", self._debug_remove_breakpoint)
        r.post("/api/debug/watch", self._debug_add_watch)
        r.delete("/api/debug/watch/:id", self._debug_remove_watch)
        r.get("/api/debug/context", self._debug_context)
        r.post("/api/debug/eval", self._debug_eval)
        r.get("/api/debug/sessions", self._debug_sessions)
        r.get("/api/debug/disassembly", self._debug_disassembly)
        r.get("/api/debug/trace", self._debug_trace)

        # -- Packages --
        r.get("/api/packages", self._packages_list)
        r.post("/api/packages/publish", self._packages_publish)
        r.post("/api/packages/add", self._packages_add)
        r.delete("/api/packages/:name", self._packages_remove)
        r.post("/api/packages/install", self._packages_install)
        r.get("/api/packages/tree", self._packages_tree)
        r.get("/api/packages/audit", self._packages_audit)

        # -- Build --
        r.post("/api/build/run", self._build_run)
        r.post("/api/build/check", self._build_check)
        r.post("/api/build/compile", self._build_compile)
        r.route("GET", "/api/build/disassemble/*", self._build_disassemble)
        r.get("/api/build/log", self._build_log)

        # -- Project --
        r.get("/api/project/summary", self._project_summary)
        r.get("/api/notifications", self._get_notifications)
        r.post("/api/project/save", self._project_save)
        r.post("/api/project/restore", self._project_restore)
        r.post("/api/project/check-all", self._project_check_all)
        r.post("/api/project/run-all", self._project_run_all)
        r.post("/api/project/refresh-diagnostics", self._project_refresh_diagnostics)

    # -- Helpers --

    def _body_json(self, request):
        """Parse request body as JSON, return dict."""
        data = request.json()
        return data if data else {}

    def _ok(self, data=None):
        """Return success JSON response."""
        if data is None:
            data = {"ok": True}
        resp = Response.json_response(data)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    def _err(self, msg, status=400):
        """Return error JSON response."""
        resp = Response.json_response({"error": msg}, status=status)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    def _decode_path_param(self, request):
        """Decode URL-encoded path parameter from wildcard or named param."""
        raw = request.params.get("_wildcard", request.params.get("path", ""))
        return urllib.parse.unquote(raw)

    # -- Frontend handler --

    def _serve_frontend(self, request):
        resp = Response.html(FRONTEND_HTML)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    # -- File handlers --

    def _strip_root(self, path):
        """Strip project root prefix from path for clean API responses."""
        root = self.ide.workspace.root
        if path.startswith(root + "/"):
            return path[len(root) + 1:]
        if path.startswith(root):
            return path[len(root):]
        return path

    def _list_files(self, request):
        pattern = request.query.get("pattern")
        with self._lock:
            files = self.ide.workspace.list_files(pattern)
        # Strip project root prefix for clean API
        files = [self._strip_root(f) for f in files]
        return self._ok({"files": files})

    def _create_file(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        content = data.get("content", "")
        if not path:
            return self._err("path is required")
        with self._lock:
            result = self.ide.create_file(path, content)
        return self._ok(result)

    def _read_file(self, request):
        path = self._decode_path_param(request)
        with self._lock:
            content = self.ide.workspace.read_file(path)
        if content is None:
            return self._err("File not found: " + path, 404)
        return self._ok({"path": path, "content": content})

    def _write_file(self, request):
        path = self._decode_path_param(request)
        data = self._body_json(request)
        content = data.get("content", "")
        with self._lock:
            result = self.ide.edit_file(path, content)
        return self._ok(result)

    def _delete_file(self, request):
        path = self._decode_path_param(request)
        with self._lock:
            ok = self.ide.delete_file(path)
        if not ok:
            return self._err("File not found: " + path, 404)
        return self._ok({"deleted": path})

    # -- Editor handlers --

    def _editor_open(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        with self._lock:
            result = self.ide.open_file(path)
            content = self.ide.workspace.read_file(path)
        result["content"] = content
        return self._ok(result)

    def _editor_close(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        with self._lock:
            self.ide.close_file(path)
        return self._ok({"closed": path})

    def _editor_complete(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        line = data.get("line", 0)
        character = data.get("character", 0)
        with self._lock:
            items = self.ide.complete(path, line, character)
        return self._ok({"completions": items})

    def _editor_hover(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        line = data.get("line", 0)
        character = data.get("character", 0)
        with self._lock:
            result = self.ide.hover(path, line, character)
        return self._ok({"hover": result})

    def _editor_definition(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        line = data.get("line", 0)
        character = data.get("character", 0)
        with self._lock:
            result = self.ide.definition(path, line, character)
        return self._ok({"definition": result})

    def _editor_symbols(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        with self._lock:
            result = self.ide.symbols(path)
        return self._ok({"symbols": result})

    def _editor_signature(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        line = data.get("line", 0)
        character = data.get("character", 0)
        with self._lock:
            result = self.ide.signature_help(path, line, character)
        return self._ok({"signature": result})

    def _editor_diagnostics(self, request):
        path = self._decode_path_param(request)
        with self._lock:
            result = self.ide.diagnostics(path)
        return self._ok({"path": path, "diagnostics": result})

    def _editor_refresh(self, request):
        with self._lock:
            result = self.ide.refresh_diagnostics()
        return self._ok(result)

    # -- Terminal handlers --

    def _terminal_eval(self, request):
        data = self._body_json(request)
        source = data.get("source", "")
        with self._lock:
            result = self.ide.eval(source)
        return self._ok(result)

    def _terminal_run(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        with self._lock:
            result = self.ide.run_file_in_repl(path)
        return self._ok(result)

    def _terminal_load(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        with self._lock:
            result = self.ide.terminal.load_file(path)
        return self._ok(result)

    def _terminal_variables(self, request):
        with self._lock:
            variables = self.ide.terminal.get_variables()
        # Convert to JSON-safe format
        safe_vars = {}
        for k, v in variables.items():
            try:
                json.dumps(v)
                safe_vars[k] = v
            except (TypeError, ValueError):
                safe_vars[k] = str(v)
        return self._ok({"variables": safe_vars})

    def _terminal_reset(self, request):
        with self._lock:
            result = self.ide.terminal.reset()
        return self._ok(result)

    def _terminal_history(self, request):
        with self._lock:
            history = self.ide.terminal.get_history()
        return self._ok({"history": history})

    # -- Debug handlers --

    def _debug_start(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        env = data.get("env")
        source = data.get("source")
        with self._lock:
            if source:
                result = self.ide.debugger.start_source(source, env=env)
            elif path:
                result = self.ide.debug_file(path, env=env)
            else:
                return self._err("path or source is required")
        return self._ok(result)

    def _debug_stop(self, request):
        data = self._body_json(request)
        session_id = data.get("session_id")
        with self._lock:
            ok = self.ide.debug_stop(session_id)
        return self._ok({"stopped": ok})

    def _debug_step(self, request):
        data = self._body_json(request)
        session_id = data.get("session_id")
        with self._lock:
            result = self.ide.debug_step(session_id)
        return self._ok({"step": result})

    def _debug_step_line(self, request):
        data = self._body_json(request)
        session_id = data.get("session_id")
        with self._lock:
            result = self.ide.debug_step_line(session_id)
        return self._ok({"step": result})

    def _debug_step_over(self, request):
        data = self._body_json(request)
        session_id = data.get("session_id")
        with self._lock:
            result = self.ide.debug_step_over(session_id)
        return self._ok({"step": result})

    def _debug_step_out(self, request):
        data = self._body_json(request)
        session_id = data.get("session_id")
        with self._lock:
            result = self.ide.debug_step_out(session_id)
        return self._ok({"step": result})

    def _debug_continue(self, request):
        data = self._body_json(request)
        session_id = data.get("session_id")
        with self._lock:
            result = self.ide.debug_continue(session_id)
        return self._ok({"step": result})

    def _debug_add_breakpoint(self, request):
        data = self._body_json(request)
        line = data.get("line", -1)
        address = data.get("address", -1)
        condition = data.get("condition", "")
        session_id = data.get("session_id")
        with self._lock:
            bp_id = self.ide.debug_add_breakpoint(line, address, condition, session_id)
        if bp_id is None:
            return self._err("No active debug session", 400)
        return self._ok({"breakpoint_id": bp_id})

    def _debug_remove_breakpoint(self, request):
        bp_id = int(request.params.get("id", "0"))
        session_id = self._body_json(request).get("session_id") if request.body else None
        with self._lock:
            ok = self.ide.debug_remove_breakpoint(bp_id, session_id)
        return self._ok({"removed": ok})

    def _debug_add_watch(self, request):
        data = self._body_json(request)
        expression = data.get("expression", "")
        session_id = data.get("session_id")
        with self._lock:
            watch_id = self.ide.debug_add_watch(expression, session_id)
        if watch_id is None:
            return self._err("No active debug session", 400)
        return self._ok({"watch_id": watch_id})

    def _debug_remove_watch(self, request):
        watch_id = int(request.params.get("id", "0"))
        session_id = self._body_json(request).get("session_id") if request.body else None
        with self._lock:
            ok = self.ide.debugger.remove_watch(watch_id, session_id)
        return self._ok({"removed": ok})

    def _debug_context(self, request):
        session_id = request.query.get("session_id")
        with self._lock:
            result = self.ide.debug_context(session_id)
        return self._ok({"context": result})

    def _debug_eval(self, request):
        data = self._body_json(request)
        expr = data.get("expression", "")
        session_id = data.get("session_id")
        with self._lock:
            result = self.ide.debug_eval(expr, session_id)
        return self._ok({"result": result})

    def _debug_sessions(self, request):
        with self._lock:
            sessions = self.ide.debugger.list_sessions()
            active = self.ide.debugger.get_active_session()
        return self._ok({"sessions": sessions, "active": active})

    def _debug_disassembly(self, request):
        session_id = request.query.get("session_id")
        context = int(request.query.get("context", "3"))
        with self._lock:
            result = self.ide.debugger.get_disassembly(session_id, context)
        return self._ok({"disassembly": result})

    def _debug_trace(self, request):
        session_id = request.query.get("session_id")
        with self._lock:
            result = self.ide.debugger.get_trace(session_id)
        return self._ok({"trace": result})

    # -- Package handlers --

    def _packages_list(self, request):
        with self._lock:
            packages = self.ide.list_packages()
        return self._ok({"packages": packages})

    def _packages_publish(self, request):
        data = self._body_json(request)
        name = data.get("name", "")
        version = data.get("version", "1.0.0")
        files = data.get("files")
        deps = data.get("dependencies")
        description = data.get("description", "")
        with self._lock:
            result = self.ide.publish_package(name, version, files, deps, description)
        return self._ok(result)

    def _packages_add(self, request):
        data = self._body_json(request)
        name = data.get("name", "")
        constraint = data.get("constraint", ">=0.0.0")
        dev = data.get("dev", False)
        with self._lock:
            result = self.ide.add_dependency(name, constraint, dev)
        # Convert SemVer objects to strings
        safe_result = {}
        for k, v in result.items():
            safe_result[k] = str(v)
        return self._ok({"installed": safe_result})

    def _packages_remove(self, request):
        name = request.params.get("name", "")
        with self._lock:
            removed = self.ide.remove_dependency(name)
        return self._ok({"removed": removed})

    def _packages_install(self, request):
        data = self._body_json(request)
        include_dev = data.get("include_dev", False)
        with self._lock:
            result = self.ide.install_packages(include_dev)
        # Convert SemVer objects to strings
        safe_result = {}
        for k, v in result.items():
            safe_result[k] = str(v)
        return self._ok({"installed": safe_result})

    def _packages_tree(self, request):
        with self._lock:
            tree = self.ide.dependency_tree()
        return self._ok({"tree": tree})

    def _packages_audit(self, request):
        with self._lock:
            result = self.ide.package_audit()
        return self._ok(result)

    # -- Build handlers --

    def _build_run(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        source = data.get("source")
        env = data.get("env")
        with self._lock:
            if source:
                result = self.ide.build.run_source(source, env)
            else:
                result = self.ide.run_file(path)
        # result is a dict from ide.run_file or BuildResult from build.run_source
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        return self._ok(result)

    def _build_check(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        with self._lock:
            result = self.ide.build.check(path)
        return self._ok(result)

    def _build_compile(self, request):
        data = self._body_json(request)
        path = data.get("path", "")
        with self._lock:
            result = self.ide.build.compile(path)
        # chunk is not JSON-serializable, convert to summary
        safe = {
            "success": result.get("success", False),
            "error": result.get("error"),
        }
        if result.get("chunk"):
            safe["has_chunk"] = True
            safe["instruction_count"] = len(result["chunk"].code) if hasattr(result["chunk"], 'code') else 0
        return self._ok(safe)

    def _build_disassemble(self, request):
        path = self._decode_path_param(request)
        with self._lock:
            result = self.ide.build.disassemble(path)
        return self._ok({"path": path, "disassembly": result})

    def _build_log(self, request):
        with self._lock:
            log = self.ide.build.get_build_log()
        return self._ok({"log": log})

    # -- Project handlers --

    def _project_summary(self, request):
        with self._lock:
            summary = self.ide.get_project_summary()
        return self._ok(summary)

    def _get_notifications(self, request):
        clear = request.query.get("clear", "true").lower() != "false"
        with self._lock:
            notifications = self.ide.get_notifications(clear=clear)
        return self._ok({"notifications": notifications})

    def _project_save(self, request):
        with self._lock:
            state = self.ide.save_state()
        return self._ok({"state": state})

    def _project_restore(self, request):
        data = self._body_json(request)
        state = data.get("state", {})
        with self._lock:
            self.ide.restore_state(state)
        return self._ok({"restored": True})

    def _project_check_all(self, request):
        with self._lock:
            result = self.ide.check_project()
        return self._ok(result)

    def _project_run_all(self, request):
        with self._lock:
            result = self.ide.run_all()
        return self._ok(result)

    def _project_refresh_diagnostics(self, request):
        with self._lock:
            result = self.ide.refresh_diagnostics()
        return self._ok(result)

    # -- Server lifecycle --

    def start(self):
        """Start the HTTP server."""
        self.server = HTTPServer(self.router, self.host, self.port)
        self.server.start()
        self.port = self.server.port
        return self

    def stop(self):
        """Stop the HTTP server and shut down IDE."""
        if self.server:
            self.server.stop()
            self.server = None
        with self._lock:
            self.ide.shutdown()

    @property
    def url(self):
        """Return the base URL of the running server."""
        return f"http://{self.host}:{self.port}"

    @property
    def request_count(self):
        """Return total requests served."""
        return self.server.request_count if self.server else 0
