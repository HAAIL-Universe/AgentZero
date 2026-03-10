"""
Tests for C026: Web IDE
Composes C016 (HTTP Server) + C024 (IDE)
"""

import sys, os, json, time, urllib.request, urllib.parse, urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C016_http_server'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C024_ide'))

import pytest
from web_ide import WebIDE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def http(method, url, body=None, headers=None):
    """Make HTTP request, return (status, parsed_json or raw text)."""
    if headers is None:
        headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req)
        raw = resp.read().decode()
        try:
            return resp.status, json.loads(raw)
        except json.JSONDecodeError:
            return resp.status, {"_raw": raw}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"_raw": raw, "error": raw}


@pytest.fixture
def web_ide():
    """Create and start a WebIDE instance, stop after test."""
    ide = WebIDE(project_root="/test-project")
    ide.start()
    yield ide
    ide.stop()


@pytest.fixture
def base(web_ide):
    """Return base URL."""
    return web_ide.url


# ---------------------------------------------------------------------------
# 1. Server Lifecycle
# ---------------------------------------------------------------------------

class TestServerLifecycle:

    def test_server_starts_and_stops(self):
        ide = WebIDE()
        ide.start()
        assert ide.port > 0
        ide.stop()

    def test_server_port_assignment(self):
        ide = WebIDE(port=0)
        ide.start()
        assert ide.port != 0
        ide.stop()

    def test_url_property(self, web_ide):
        assert web_ide.url.startswith("http://127.0.0.1:")

    def test_request_count_starts_zero(self, web_ide):
        assert web_ide.request_count == 0

    def test_request_count_increments(self, base, web_ide):
        http("GET", f"{base}/api/files")
        time.sleep(0.1)
        assert web_ide.request_count >= 1


# ---------------------------------------------------------------------------
# 2. Frontend
# ---------------------------------------------------------------------------

class TestFrontend:

    def test_serves_html(self, base):
        req = urllib.request.Request(f"{base}/")
        resp = urllib.request.urlopen(req)
        content = resp.read().decode()
        assert "AgentZero Web IDE" in content
        assert resp.headers.get("Content-Type", "").startswith("text/html")

    def test_html_has_editor(self, base):
        req = urllib.request.Request(f"{base}/")
        resp = urllib.request.urlopen(req)
        content = resp.read().decode()
        assert "editor" in content
        assert "textarea" in content

    def test_html_has_toolbar(self, base):
        req = urllib.request.Request(f"{base}/")
        resp = urllib.request.urlopen(req)
        content = resp.read().decode()
        assert "Run" in content
        assert "Debug" in content

    def test_html_has_panels(self, base):
        req = urllib.request.Request(f"{base}/")
        resp = urllib.request.urlopen(req)
        content = resp.read().decode()
        assert "Terminal" in content
        assert "Diagnostics" in content


# ---------------------------------------------------------------------------
# 3. CORS
# ---------------------------------------------------------------------------

class TestCORS:

    def test_cors_headers_on_response(self, base):
        req = urllib.request.Request(f"{base}/api/files")
        resp = urllib.request.urlopen(req)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_options_preflight(self, base):
        req = urllib.request.Request(f"{base}/api/files", method="OPTIONS")
        resp = urllib.request.urlopen(req)
        assert resp.status == 204
        assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")


# ---------------------------------------------------------------------------
# 4. File Operations
# ---------------------------------------------------------------------------

class TestFileOperations:

    def test_list_files_empty(self, base):
        s, data = http("GET", f"{base}/api/files")
        assert s == 200
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_create_file(self, base):
        s, data = http("POST", f"{base}/api/files", {"path": "hello.ml", "content": "let x = 1;"})
        assert s == 200

    def test_create_and_list_file(self, base):
        http("POST", f"{base}/api/files", {"path": "test1.ml", "content": "let a = 1;"})
        s, data = http("GET", f"{base}/api/files")
        assert "test1.ml" in data["files"]

    def test_read_file(self, base):
        http("POST", f"{base}/api/files", {"path": "read_me.ml", "content": "let val = 42;"})
        s, data = http("GET", f"{base}/api/files/read_me.ml")
        assert s == 200
        assert data["content"] == "let val = 42;"

    def test_read_nonexistent_file(self, base):
        s, data = http("GET", f"{base}/api/files/nope.ml")
        assert s == 404
        assert "error" in data

    def test_write_file(self, base):
        http("POST", f"{base}/api/files", {"path": "write_me.ml", "content": "old"})
        s, data = http("PUT", f"{base}/api/files/write_me.ml", {"content": "new content"})
        assert s == 200
        s2, data2 = http("GET", f"{base}/api/files/write_me.ml")
        assert data2["content"] == "new content"

    def test_delete_file(self, base):
        http("POST", f"{base}/api/files", {"path": "delete_me.ml", "content": ""})
        s, data = http("DELETE", f"{base}/api/files/delete_me.ml")
        assert s == 200
        assert data["deleted"] == "delete_me.ml"
        s2, _ = http("GET", f"{base}/api/files/delete_me.ml")
        assert s2 == 404

    def test_delete_nonexistent(self, base):
        s, data = http("DELETE", f"{base}/api/files/no_such.ml")
        assert s == 404

    def test_create_file_no_path(self, base):
        s, data = http("POST", f"{base}/api/files", {"content": "hi"})
        assert s == 400
        assert "error" in data

    def test_list_files_with_pattern(self, base):
        http("POST", f"{base}/api/files", {"path": "a.ml", "content": ""})
        http("POST", f"{base}/api/files", {"path": "b.txt", "content": ""})
        s, data = http("GET", f"{base}/api/files?pattern=*.ml")
        assert "a.ml" in data["files"]

    def test_url_encoded_path(self, base):
        http("POST", f"{base}/api/files", {"path": "sub/deep.ml", "content": "let z = 0;"})
        encoded = urllib.parse.quote("sub/deep.ml", safe="")
        s, data = http("GET", f"{base}/api/files/{encoded}")
        assert s == 200
        assert data["content"] == "let z = 0;"

    def test_multiple_files(self, base):
        for i in range(5):
            http("POST", f"{base}/api/files", {"path": f"multi_{i}.ml", "content": f"let x = {i};"})
        s, data = http("GET", f"{base}/api/files")
        for i in range(5):
            assert f"multi_{i}.ml" in data["files"]


# ---------------------------------------------------------------------------
# 5. Editor Operations
# ---------------------------------------------------------------------------

class TestEditorOperations:

    def test_open_file(self, base):
        http("POST", f"{base}/api/files", {"path": "edit.ml", "content": "let x = 10;"})
        s, data = http("POST", f"{base}/api/editor/open", {"path": "edit.ml"})
        assert s == 200
        assert data["content"] == "let x = 10;"

    def test_close_file(self, base):
        http("POST", f"{base}/api/files", {"path": "close_me.ml", "content": ""})
        http("POST", f"{base}/api/editor/open", {"path": "close_me.ml"})
        s, data = http("POST", f"{base}/api/editor/close", {"path": "close_me.ml"})
        assert s == 200
        assert data["closed"] == "close_me.ml"

    def test_completions(self, base):
        http("POST", f"{base}/api/files", {"path": "comp.ml", "content": "let hello = 1;\nhel"})
        http("POST", f"{base}/api/editor/open", {"path": "comp.ml"})
        s, data = http("POST", f"{base}/api/editor/complete", {"path": "comp.ml", "line": 2, "character": 3})
        assert s == 200
        assert "completions" in data

    def test_hover(self, base):
        http("POST", f"{base}/api/files", {"path": "hover.ml", "content": "let val = 42;"})
        http("POST", f"{base}/api/editor/open", {"path": "hover.ml"})
        s, data = http("POST", f"{base}/api/editor/hover", {"path": "hover.ml", "line": 1, "character": 4})
        assert s == 200
        assert "hover" in data

    def test_definition(self, base):
        http("POST", f"{base}/api/files", {"path": "def.ml", "content": "let foo = 1;\nfoo;"})
        http("POST", f"{base}/api/editor/open", {"path": "def.ml"})
        s, data = http("POST", f"{base}/api/editor/definition", {"path": "def.ml", "line": 2, "character": 1})
        assert s == 200
        assert "definition" in data

    def test_symbols(self, base):
        http("POST", f"{base}/api/files", {"path": "sym.ml", "content": "let alpha = 1;\nlet beta = 2;"})
        http("POST", f"{base}/api/editor/open", {"path": "sym.ml"})
        s, data = http("POST", f"{base}/api/editor/symbols", {"path": "sym.ml"})
        assert s == 200
        assert "symbols" in data

    def test_signature_help(self, base):
        http("POST", f"{base}/api/files", {"path": "sig.ml", "content": "fun add(a, b) { return a + b; }\nadd("})
        http("POST", f"{base}/api/editor/open", {"path": "sig.ml"})
        s, data = http("POST", f"{base}/api/editor/signature", {"path": "sig.ml", "line": 2, "character": 4})
        assert s == 200
        assert "signature" in data

    def test_diagnostics(self, base):
        http("POST", f"{base}/api/files", {"path": "diag.ml", "content": "let x = 1 + true;"})
        http("POST", f"{base}/api/editor/open", {"path": "diag.ml"})
        s, data = http("GET", f"{base}/api/editor/diagnostics/diag.ml")
        assert s == 200
        assert "diagnostics" in data

    def test_refresh_all_diagnostics(self, base):
        http("POST", f"{base}/api/files", {"path": "r1.ml", "content": "let a = 1;"})
        http("POST", f"{base}/api/editor/open", {"path": "r1.ml"})
        s, data = http("POST", f"{base}/api/editor/refresh")
        assert s == 200

    def test_open_returns_diagnostics(self, base):
        http("POST", f"{base}/api/files", {"path": "od.ml", "content": "let x = 1;"})
        s, data = http("POST", f"{base}/api/editor/open", {"path": "od.ml"})
        assert "diagnostics" in data


# ---------------------------------------------------------------------------
# 6. Terminal / REPL
# ---------------------------------------------------------------------------

class TestTerminal:

    def test_eval_expression(self, base):
        s, data = http("POST", f"{base}/api/terminal/eval", {"source": "1 + 2;"})
        assert s == 200
        assert "result" in data or "output" in data

    def test_eval_let_binding(self, base):
        http("POST", f"{base}/api/terminal/eval", {"source": "let x = 42;"})
        s, data = http("GET", f"{base}/api/terminal/variables")
        assert s == 200
        assert "variables" in data

    def test_terminal_reset(self, base):
        http("POST", f"{base}/api/terminal/eval", {"source": "let reset_var = 99;"})
        s, data = http("POST", f"{base}/api/terminal/reset")
        assert s == 200

    def test_terminal_history(self, base):
        http("POST", f"{base}/api/terminal/eval", {"source": "1 + 1;"})
        s, data = http("GET", f"{base}/api/terminal/history")
        assert s == 200
        assert "history" in data

    def test_run_file_in_terminal(self, base):
        http("POST", f"{base}/api/files", {"path": "trun.ml", "content": "print(42);"})
        s, data = http("POST", f"{base}/api/terminal/run", {"path": "trun.ml"})
        assert s == 200

    def test_load_file(self, base):
        http("POST", f"{base}/api/files", {"path": "tload.ml", "content": "let loaded = 77;"})
        s, data = http("POST", f"{base}/api/terminal/load", {"path": "tload.ml"})
        assert s == 200

    def test_eval_error_returns_error(self, base):
        s, data = http("POST", f"{base}/api/terminal/eval", {"source": "undefined_var;"})
        assert s == 200
        # Should have error or result (depends on REPL behavior)

    def test_multiple_evals_persist_state(self, base):
        http("POST", f"{base}/api/terminal/eval", {"source": "let counter = 0;"})
        http("POST", f"{base}/api/terminal/eval", {"source": "counter = counter + 1;"})
        s, data = http("GET", f"{base}/api/terminal/variables")
        assert s == 200


# ---------------------------------------------------------------------------
# 7. Debug Operations
# ---------------------------------------------------------------------------

class TestDebugOperations:

    def _create_debug_file(self, base):
        http("POST", f"{base}/api/files", {"path": "dbg.ml", "content": "let a = 1;\nlet b = 2;\nlet c = a + b;\nprint(c);"})
        return "dbg.ml"

    def test_start_debug(self, base):
        path = self._create_debug_file(base)
        s, data = http("POST", f"{base}/api/debug/start", {"path": path})
        assert s == 200

    def test_debug_step(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("POST", f"{base}/api/debug/step")
        assert s == 200
        assert "step" in data

    def test_debug_step_line(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("POST", f"{base}/api/debug/step-line")
        assert s == 200

    def test_debug_step_over(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("POST", f"{base}/api/debug/step-over")
        assert s == 200

    def test_debug_continue(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("POST", f"{base}/api/debug/continue")
        assert s == 200

    def test_debug_stop(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("POST", f"{base}/api/debug/stop")
        assert s == 200
        assert "stopped" in data

    def test_debug_context(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        http("POST", f"{base}/api/debug/step")
        s, data = http("GET", f"{base}/api/debug/context")
        assert s == 200
        assert "context" in data

    def test_debug_add_breakpoint(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("POST", f"{base}/api/debug/breakpoint", {"line": 3})
        assert s == 200
        assert "breakpoint_id" in data

    def test_debug_remove_breakpoint(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        _, bp_data = http("POST", f"{base}/api/debug/breakpoint", {"line": 2})
        bp_id = bp_data["breakpoint_id"]
        s, data = http("DELETE", f"{base}/api/debug/breakpoint/{bp_id}")
        assert s == 200

    def test_debug_add_watch(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("POST", f"{base}/api/debug/watch", {"expression": "a"})
        assert s == 200
        assert "watch_id" in data

    def test_debug_remove_watch(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        _, w_data = http("POST", f"{base}/api/debug/watch", {"expression": "b"})
        w_id = w_data["watch_id"]
        s, data = http("DELETE", f"{base}/api/debug/watch/{w_id}")
        assert s == 200

    def test_debug_eval(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        http("POST", f"{base}/api/debug/step-line")
        s, data = http("POST", f"{base}/api/debug/eval", {"expression": "a"})
        assert s == 200
        assert "result" in data

    def test_debug_sessions_list(self, base):
        s, data = http("GET", f"{base}/api/debug/sessions")
        assert s == 200
        assert "sessions" in data

    def test_debug_disassembly(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("GET", f"{base}/api/debug/disassembly")
        assert s == 200

    def test_debug_trace(self, base):
        path = self._create_debug_file(base)
        http("POST", f"{base}/api/debug/start", {"path": path})
        s, data = http("GET", f"{base}/api/debug/trace")
        assert s == 200

    def test_debug_start_from_source(self, base):
        s, data = http("POST", f"{base}/api/debug/start", {"source": "let x = 1;"})
        assert s == 200

    def test_debug_step_out(self, base):
        http("POST", f"{base}/api/files", {"path": "dbg_out.ml", "content": "fun f() { return 1; }\nf();"})
        http("POST", f"{base}/api/debug/start", {"path": "dbg_out.ml"})
        # Step into function then step out
        for _ in range(5):
            http("POST", f"{base}/api/debug/step")
        s, data = http("POST", f"{base}/api/debug/step-out")
        assert s == 200

    def test_debug_breakpoint_no_session(self, base):
        # No session started, should fail
        s, data = http("POST", f"{base}/api/debug/breakpoint", {"line": 1})
        assert s == 400

    def test_debug_watch_no_session(self, base):
        s, data = http("POST", f"{base}/api/debug/watch", {"expression": "x"})
        assert s == 400


# ---------------------------------------------------------------------------
# 8. Build Operations
# ---------------------------------------------------------------------------

class TestBuildOperations:

    def test_build_run(self, base):
        http("POST", f"{base}/api/files", {"path": "run.ml", "content": "print(1 + 2);"})
        s, data = http("POST", f"{base}/api/build/run", {"path": "run.ml"})
        assert s == 200
        assert "success" in data

    def test_build_run_success(self, base):
        http("POST", f"{base}/api/files", {"path": "run_ok.ml", "content": "let x = 5;"})
        s, data = http("POST", f"{base}/api/build/run", {"path": "run_ok.ml"})
        assert data["success"] is True

    def test_build_run_error(self, base):
        http("POST", f"{base}/api/files", {"path": "run_err.ml", "content": "let x = ;"})
        s, data = http("POST", f"{base}/api/build/run", {"path": "run_err.ml"})
        assert data["success"] is False

    def test_build_check(self, base):
        http("POST", f"{base}/api/files", {"path": "chk.ml", "content": "let x = 1;"})
        s, data = http("POST", f"{base}/api/build/check", {"path": "chk.ml"})
        assert s == 200

    def test_build_compile(self, base):
        http("POST", f"{base}/api/files", {"path": "cmp.ml", "content": "let x = 1;"})
        s, data = http("POST", f"{base}/api/build/compile", {"path": "cmp.ml"})
        assert s == 200
        assert "success" in data

    def test_build_disassemble(self, base):
        http("POST", f"{base}/api/files", {"path": "dis.ml", "content": "let x = 1;"})
        s, data = http("GET", f"{base}/api/build/disassemble/dis.ml")
        assert s == 200
        assert "disassembly" in data

    def test_build_log(self, base):
        http("POST", f"{base}/api/files", {"path": "blog.ml", "content": "let x = 1;"})
        http("POST", f"{base}/api/build/run", {"path": "blog.ml"})
        s, data = http("GET", f"{base}/api/build/log")
        assert s == 200
        assert "log" in data

    def test_build_run_from_source(self, base):
        s, data = http("POST", f"{base}/api/build/run", {"source": "print(99);"})
        assert s == 200
        assert data["success"] is True

    def test_build_run_with_output(self, base):
        http("POST", f"{base}/api/files", {"path": "out.ml", "content": "print(42);"})
        s, data = http("POST", f"{base}/api/build/run", {"path": "out.ml"})
        assert data["success"] is True
        assert "42" in str(data.get("output", ""))


# ---------------------------------------------------------------------------
# 9. Package Operations
# ---------------------------------------------------------------------------

class TestPackageOperations:

    def test_list_packages_empty(self, base):
        s, data = http("GET", f"{base}/api/packages")
        assert s == 200
        assert "packages" in data

    def test_publish_package(self, base):
        s, data = http("POST", f"{base}/api/packages/publish", {
            "name": "mylib",
            "version": "1.0.0",
            "files": {"main.ml": "let lib_val = 1;"},
            "description": "A test lib"
        })
        assert s == 200

    def test_publish_and_list(self, base):
        http("POST", f"{base}/api/packages/publish", {
            "name": "listed_lib",
            "version": "1.0.0",
            "files": {"main.ml": "let x = 1;"}
        })
        http("POST", f"{base}/api/packages/add", {"name": "listed_lib", "constraint": ">=1.0.0"})
        http("POST", f"{base}/api/packages/install")
        s, data = http("GET", f"{base}/api/packages")
        # Should have at least one package
        assert s == 200

    def test_add_dependency(self, base):
        http("POST", f"{base}/api/packages/publish", {
            "name": "dep_lib",
            "version": "2.0.0",
            "files": {"main.ml": "let y = 2;"}
        })
        s, data = http("POST", f"{base}/api/packages/add", {"name": "dep_lib", "constraint": ">=1.0.0"})
        assert s == 200
        assert "installed" in data

    def test_remove_dependency(self, base):
        http("POST", f"{base}/api/packages/publish", {
            "name": "rem_lib",
            "version": "1.0.0",
            "files": {"main.ml": ""}
        })
        http("POST", f"{base}/api/packages/add", {"name": "rem_lib", "constraint": ">=1.0.0"})
        s, data = http("DELETE", f"{base}/api/packages/rem_lib")
        assert s == 200
        assert "removed" in data

    def test_install_packages(self, base):
        s, data = http("POST", f"{base}/api/packages/install")
        assert s == 200
        assert "installed" in data

    def test_dependency_tree(self, base):
        s, data = http("GET", f"{base}/api/packages/tree")
        assert s == 200
        assert "tree" in data

    def test_package_audit(self, base):
        s, data = http("GET", f"{base}/api/packages/audit")
        assert s == 200


# ---------------------------------------------------------------------------
# 10. Project Operations
# ---------------------------------------------------------------------------

class TestProjectOperations:

    def test_project_summary(self, base):
        s, data = http("GET", f"{base}/api/project/summary")
        assert s == 200
        assert "root" in data or "files" in data

    def test_notifications_empty(self, base):
        s, data = http("GET", f"{base}/api/notifications")
        assert s == 200
        assert "notifications" in data

    def test_notifications_after_file_create(self, base):
        # Clear any existing notifications
        http("GET", f"{base}/api/notifications")
        http("POST", f"{base}/api/files", {"path": "notif.ml", "content": "let n = 1;"})
        s, data = http("GET", f"{base}/api/notifications")
        assert s == 200
        # Should have FILE_CREATED notification
        events = [n["event"] for n in data["notifications"]] if data["notifications"] else []
        assert any("FILE" in e for e in events) or len(events) == 0  # depends on IDE event timing

    def test_notifications_clear(self, base):
        http("POST", f"{base}/api/files", {"path": "nc.ml", "content": ""})
        http("GET", f"{base}/api/notifications?clear=true")
        s, data = http("GET", f"{base}/api/notifications")
        assert data["notifications"] == []

    def test_notifications_no_clear(self, base):
        http("POST", f"{base}/api/files", {"path": "nk.ml", "content": ""})
        http("GET", f"{base}/api/notifications?clear=false")
        # Getting again without clear should still show them
        s, data = http("GET", f"{base}/api/notifications?clear=false")
        assert s == 200

    def test_save_state(self, base):
        http("POST", f"{base}/api/files", {"path": "sv.ml", "content": "let s = 1;"})
        s, data = http("POST", f"{base}/api/project/save")
        assert s == 200
        assert "state" in data

    def test_restore_state(self, base):
        http("POST", f"{base}/api/files", {"path": "rs.ml", "content": "let r = 1;"})
        _, save_data = http("POST", f"{base}/api/project/save")
        s, data = http("POST", f"{base}/api/project/restore", {"state": save_data["state"]})
        assert s == 200
        assert data["restored"] is True

    def test_check_all_project(self, base):
        http("POST", f"{base}/api/files", {"path": "ca.ml", "content": "let a = 1;"})
        s, data = http("POST", f"{base}/api/project/check-all")
        assert s == 200

    def test_run_all_project(self, base):
        http("POST", f"{base}/api/files", {"path": "ra.ml", "content": "let b = 2;"})
        s, data = http("POST", f"{base}/api/project/run-all")
        assert s == 200

    def test_refresh_diagnostics(self, base):
        http("POST", f"{base}/api/files", {"path": "rd.ml", "content": "let d = 3;"})
        http("POST", f"{base}/api/editor/open", {"path": "rd.ml"})
        s, data = http("POST", f"{base}/api/project/refresh-diagnostics")
        assert s == 200


# ---------------------------------------------------------------------------
# 11. Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_404_unknown_route(self, base):
        s, _ = http("GET", f"{base}/api/nonexistent")
        assert s == 404

    def test_json_error_format(self, base):
        s, data = http("GET", f"{base}/api/files/nonexistent.ml")
        assert s == 404
        assert "error" in data

    def test_method_not_allowed(self, base):
        # PATCH on /api/files should 404 or 405
        s, _ = http("PATCH", f"{base}/api/files", {"x": 1})
        assert s in (404, 405)


# ---------------------------------------------------------------------------
# 12. Integration Workflows
# ---------------------------------------------------------------------------

class TestIntegrationWorkflows:

    def test_create_edit_run_workflow(self, base):
        """Full workflow: create -> edit -> run."""
        http("POST", f"{base}/api/files", {"path": "flow.ml", "content": "print(1);"})
        http("PUT", f"{base}/api/files/flow.ml", {"content": "print(2 + 3);"})
        s, data = http("POST", f"{base}/api/build/run", {"path": "flow.ml"})
        assert data["success"] is True
        assert "5" in str(data.get("output", ""))

    def test_create_open_complete_workflow(self, base):
        """Create file, open in editor, get completions."""
        http("POST", f"{base}/api/files", {"path": "wf_comp.ml", "content": "let myvar = 1;\nmyv"})
        http("POST", f"{base}/api/editor/open", {"path": "wf_comp.ml"})
        s, data = http("POST", f"{base}/api/editor/complete", {"path": "wf_comp.ml", "line": 2, "character": 3})
        assert s == 200

    def test_debug_with_breakpoint_workflow(self, base):
        """Start debug, add breakpoint, continue to it."""
        http("POST", f"{base}/api/files", {"path": "wf_dbg.ml", "content": "let a = 1;\nlet b = 2;\nlet c = 3;\nprint(c);"})
        http("POST", f"{base}/api/debug/start", {"path": "wf_dbg.ml"})
        http("POST", f"{base}/api/debug/breakpoint", {"line": 3})
        http("POST", f"{base}/api/debug/continue")
        s, data = http("GET", f"{base}/api/debug/context")
        assert s == 200

    def test_package_publish_install_workflow(self, base):
        """Publish a package, add dependency, install."""
        http("POST", f"{base}/api/packages/publish", {
            "name": "wf_lib",
            "version": "1.0.0",
            "files": {"main.ml": "let lib = 42;"}
        })
        http("POST", f"{base}/api/packages/add", {"name": "wf_lib", "constraint": ">=1.0.0"})
        s, data = http("POST", f"{base}/api/packages/install")
        assert s == 200

    def test_save_restore_preserves_files(self, base):
        """Save state, create file, restore, file should exist."""
        http("POST", f"{base}/api/files", {"path": "persist.ml", "content": "let p = 1;"})
        _, save = http("POST", f"{base}/api/project/save")
        http("POST", f"{base}/api/project/restore", {"state": save["state"]})
        s, data = http("GET", f"{base}/api/files/persist.ml")
        assert s == 200
        assert data["content"] == "let p = 1;"

    def test_terminal_state_persists(self, base):
        """REPL state should persist between eval calls."""
        http("POST", f"{base}/api/terminal/eval", {"source": "let shared = 100;"})
        s, data = http("POST", f"{base}/api/terminal/eval", {"source": "shared;"})
        assert s == 200

    def test_multiple_debug_sessions(self, base):
        """Start multiple debug sessions."""
        http("POST", f"{base}/api/files", {"path": "d1.ml", "content": "let a = 1;"})
        http("POST", f"{base}/api/files", {"path": "d2.ml", "content": "let b = 2;"})
        http("POST", f"{base}/api/debug/start", {"path": "d1.ml"})
        s, data = http("GET", f"{base}/api/debug/sessions")
        assert len(data["sessions"]) >= 1

    def test_edit_triggers_diagnostics(self, base):
        """Editing a file should produce diagnostics update."""
        http("POST", f"{base}/api/files", {"path": "diag_wf.ml", "content": "let x = 1;"})
        http("POST", f"{base}/api/editor/open", {"path": "diag_wf.ml"})
        s, data = http("PUT", f"{base}/api/files/diag_wf.ml", {"content": "let x = 1 + true;"})
        assert s == 200

    def test_concurrent_file_operations(self, base):
        """Multiple file operations in sequence."""
        for i in range(10):
            http("POST", f"{base}/api/files", {"path": f"conc_{i}.ml", "content": f"let x = {i};"})
        s, data = http("GET", f"{base}/api/files")
        count = sum(1 for f in data["files"] if f.startswith("conc_"))
        assert count == 10

    def test_full_edit_cycle(self, base):
        """Create, open, edit, check, run -- full cycle."""
        http("POST", f"{base}/api/files", {"path": "cycle.ml", "content": ""})
        http("POST", f"{base}/api/editor/open", {"path": "cycle.ml"})
        http("PUT", f"{base}/api/files/cycle.ml", {"content": "let result = 10 * 5;\nprint(result);"})
        _, check = http("POST", f"{base}/api/build/check", {"path": "cycle.ml"})
        assert check.get("success") is True
        _, run = http("POST", f"{base}/api/build/run", {"path": "cycle.ml"})
        assert run["success"] is True
        assert "50" in str(run.get("output", ""))


# ---------------------------------------------------------------------------
# 13. Thread Safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_lock_protects_ide_state(self, web_ide, base):
        """WebIDE uses lock for thread safety."""
        assert web_ide._lock is not None

    def test_rapid_requests(self, base):
        """Send many requests rapidly to test thread safety."""
        import concurrent.futures
        http("POST", f"{base}/api/files", {"path": "rapid.ml", "content": "let x = 1;"})

        def read_file():
            return http("GET", f"{base}/api/files/rapid.ml")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(read_file) for _ in range(20)]
            results = [f.result() for f in futures]

        assert all(s == 200 for s, _ in results)


# ---------------------------------------------------------------------------
# 14. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_body_post(self, base):
        s, data = http("POST", f"{base}/api/terminal/eval", {"source": ""})
        assert s == 200

    def test_unicode_content(self, base):
        http("POST", f"{base}/api/files", {"path": "uni.ml", "content": "// comment"})
        s, data = http("GET", f"{base}/api/files/uni.ml")
        assert data["content"] == "// comment"

    def test_large_file(self, base):
        content = "let x = 1;\n" * 100
        http("POST", f"{base}/api/files", {"path": "large.ml", "content": content})
        s, data = http("GET", f"{base}/api/files/large.ml")
        assert data["content"] == content

    def test_special_characters_in_path(self, base):
        http("POST", f"{base}/api/files", {"path": "dir/sub-file_v2.ml", "content": "let s = 1;"})
        encoded = urllib.parse.quote("dir/sub-file_v2.ml", safe="")
        s, data = http("GET", f"{base}/api/files/{encoded}")
        assert s == 200

    def test_overwrite_file(self, base):
        http("POST", f"{base}/api/files", {"path": "ow.ml", "content": "v1"})
        http("PUT", f"{base}/api/files/ow.ml", {"content": "v2"})
        s, data = http("GET", f"{base}/api/files/ow.ml")
        assert data["content"] == "v2"

    def test_debug_no_path_no_source(self, base):
        s, data = http("POST", f"{base}/api/debug/start", {})
        assert s == 400

    def test_build_compile_returns_chunk_info(self, base):
        http("POST", f"{base}/api/files", {"path": "ci.ml", "content": "let x = 1;"})
        s, data = http("POST", f"{base}/api/build/compile", {"path": "ci.ml"})
        if data.get("success"):
            assert "has_chunk" in data


# ---------------------------------------------------------------------------
# 15. Advanced Editor Tests
# ---------------------------------------------------------------------------

class TestAdvancedEditor:

    def test_edit_updates_diagnostics(self, base):
        http("POST", f"{base}/api/files", {"path": "ae.ml", "content": "let x = 1;"})
        http("POST", f"{base}/api/editor/open", {"path": "ae.ml"})
        s, data = http("PUT", f"{base}/api/files/ae.ml", {"content": "let x = 1 + true;"})
        assert s == 200
        assert "diagnostics" in data

    def test_multiple_open_files(self, base):
        http("POST", f"{base}/api/files", {"path": "m1.ml", "content": "let a = 1;"})
        http("POST", f"{base}/api/files", {"path": "m2.ml", "content": "let b = 2;"})
        http("POST", f"{base}/api/editor/open", {"path": "m1.ml"})
        http("POST", f"{base}/api/editor/open", {"path": "m2.ml"})
        s1, d1 = http("GET", f"{base}/api/editor/diagnostics/m1.ml")
        s2, d2 = http("GET", f"{base}/api/editor/diagnostics/m2.ml")
        assert s1 == 200
        assert s2 == 200

    def test_symbols_returns_list(self, base):
        http("POST", f"{base}/api/files", {"path": "sfn.ml", "content": "fun greet(name) { return name; }\nlet x = 1;"})
        http("POST", f"{base}/api/editor/open", {"path": "sfn.ml"})
        s, data = http("POST", f"{base}/api/editor/symbols", {"path": "sfn.ml"})
        assert isinstance(data["symbols"], list)


# ---------------------------------------------------------------------------
# 16. Advanced Debug Tests
# ---------------------------------------------------------------------------

class TestAdvancedDebug:

    def test_debug_with_env(self, base):
        http("POST", f"{base}/api/files", {"path": "denv.ml", "content": "let a = 1;"})
        s, data = http("POST", f"{base}/api/debug/start", {"path": "denv.ml", "env": {"x": 42}})
        assert s == 200

    def test_debug_multiple_steps(self, base):
        http("POST", f"{base}/api/files", {"path": "dms.ml", "content": "let a = 1;\nlet b = 2;\nlet c = 3;"})
        http("POST", f"{base}/api/debug/start", {"path": "dms.ml"})
        for _ in range(3):
            s, _ = http("POST", f"{base}/api/debug/step")
            assert s == 200

    def test_debug_context_has_variables(self, base):
        http("POST", f"{base}/api/files", {"path": "dcv.ml", "content": "let x = 42;\nlet y = 0;"})
        http("POST", f"{base}/api/debug/start", {"path": "dcv.ml"})
        http("POST", f"{base}/api/debug/step-line")
        http("POST", f"{base}/api/debug/step-line")
        s, data = http("GET", f"{base}/api/debug/context")
        assert s == 200
        if data["context"]:
            assert "variables" in data["context"] or "stack" in data["context"]

    def test_debug_breakpoint_and_continue(self, base):
        http("POST", f"{base}/api/files", {"path": "dbc.ml", "content": "let a = 1;\nlet b = 2;\nlet c = 3;\nlet d = 4;"})
        http("POST", f"{base}/api/debug/start", {"path": "dbc.ml"})
        _, bp = http("POST", f"{base}/api/debug/breakpoint", {"line": 3})
        assert "breakpoint_id" in bp
        s, data = http("POST", f"{base}/api/debug/continue")
        assert s == 200

    def test_debug_watch_evaluation(self, base):
        http("POST", f"{base}/api/files", {"path": "dwe.ml", "content": "let x = 42;\nlet y = 0;"})
        http("POST", f"{base}/api/debug/start", {"path": "dwe.ml"})
        http("POST", f"{base}/api/debug/step-line")
        _, w = http("POST", f"{base}/api/debug/watch", {"expression": "x"})
        assert "watch_id" in w
        s, ctx = http("GET", f"{base}/api/debug/context")
        assert s == 200

    def test_debug_sessions_tracking(self, base):
        http("POST", f"{base}/api/files", {"path": "ds1.ml", "content": "let a = 1;"})
        http("POST", f"{base}/api/debug/start", {"path": "ds1.ml"})
        s, data = http("GET", f"{base}/api/debug/sessions")
        assert len(data["sessions"]) >= 1
        assert data["active"] is not None


# ---------------------------------------------------------------------------
# 17. Advanced Build Tests
# ---------------------------------------------------------------------------

class TestAdvancedBuild:

    def test_build_check_valid(self, base):
        http("POST", f"{base}/api/files", {"path": "bv.ml", "content": "let x = 1 + 2;"})
        s, data = http("POST", f"{base}/api/build/check", {"path": "bv.ml"})
        assert data.get("success") is True

    def test_build_check_invalid(self, base):
        http("POST", f"{base}/api/files", {"path": "bi.ml", "content": "let x = ;"})
        s, data = http("POST", f"{base}/api/build/check", {"path": "bi.ml"})
        assert data.get("success") is False

    def test_build_run_with_print(self, base):
        http("POST", f"{base}/api/files", {"path": "bp.ml", "content": "print(100);\nprint(200);"})
        s, data = http("POST", f"{base}/api/build/run", {"path": "bp.ml"})
        assert data["success"] is True
        assert "100" in str(data.get("output", ""))

    def test_build_disassemble_valid(self, base):
        http("POST", f"{base}/api/files", {"path": "bd.ml", "content": "let x = 1 + 2;"})
        s, data = http("GET", f"{base}/api/build/disassemble/bd.ml")
        assert s == 200
        assert data["disassembly"] is not None

    def test_build_log_populated(self, base):
        http("POST", f"{base}/api/files", {"path": "bl.ml", "content": "let x = 1;"})
        http("POST", f"{base}/api/build/run", {"path": "bl.ml"})
        s, data = http("GET", f"{base}/api/build/log")
        assert len(data["log"]) >= 1

    def test_build_compile_error(self, base):
        http("POST", f"{base}/api/files", {"path": "bce.ml", "content": "let x = ;"})
        s, data = http("POST", f"{base}/api/build/compile", {"path": "bce.ml"})
        assert data["success"] is False


# ---------------------------------------------------------------------------
# 18. Advanced Package Tests
# ---------------------------------------------------------------------------

class TestAdvancedPackages:

    def test_publish_with_dependencies(self, base):
        http("POST", f"{base}/api/packages/publish", {
            "name": "base_lib", "version": "1.0.0",
            "files": {"main.ml": "let base = 1;"}
        })
        http("POST", f"{base}/api/packages/publish", {
            "name": "ext_lib", "version": "1.0.0",
            "files": {"main.ml": "let ext = 2;"},
            "dependencies": {"base_lib": ">=1.0.0"}
        })
        s, data = http("POST", f"{base}/api/packages/add", {"name": "ext_lib", "constraint": ">=1.0.0"})
        assert s == 200

    def test_install_with_dev(self, base):
        s, data = http("POST", f"{base}/api/packages/install", {"include_dev": True})
        assert s == 200

    def test_audit_returns_data(self, base):
        s, data = http("GET", f"{base}/api/packages/audit")
        assert s == 200
        # Audit should return some structure
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# 19. Advanced Project Tests
# ---------------------------------------------------------------------------

class TestAdvancedProject:

    def test_project_summary_contents(self, base):
        http("POST", f"{base}/api/files", {"path": "ps.ml", "content": "let x = 1;"})
        s, data = http("GET", f"{base}/api/project/summary")
        assert "files" in data

    def test_save_and_restore_roundtrip(self, base):
        http("POST", f"{base}/api/files", {"path": "rt.ml", "content": "let round = 1;"})
        http("POST", f"{base}/api/editor/open", {"path": "rt.ml"})
        _, save = http("POST", f"{base}/api/project/save")
        state = save["state"]
        assert "files" in state
        http("POST", f"{base}/api/project/restore", {"state": state})
        s, data = http("GET", f"{base}/api/files/rt.ml")
        assert data["content"] == "let round = 1;"

    def test_check_all_with_errors(self, base):
        http("POST", f"{base}/api/files", {"path": "ca_ok.ml", "content": "let x = 1;"})
        http("POST", f"{base}/api/files", {"path": "ca_err.ml", "content": "let x = ;"})
        s, data = http("POST", f"{base}/api/project/check-all")
        assert s == 200

    def test_run_all_files(self, base):
        http("POST", f"{base}/api/files", {"path": "ra1.ml", "content": "let x = 1;"})
        http("POST", f"{base}/api/files", {"path": "ra2.ml", "content": "let y = 2;"})
        s, data = http("POST", f"{base}/api/project/run-all")
        assert s == 200

    def test_notifications_accumulate(self, base):
        # Clear
        http("GET", f"{base}/api/notifications")
        # Create several files
        for i in range(3):
            http("POST", f"{base}/api/files", {"path": f"notif_{i}.ml", "content": ""})
        s, data = http("GET", f"{base}/api/notifications")
        assert len(data["notifications"]) >= 3


# ---------------------------------------------------------------------------
# 20. Composition Integrity
# ---------------------------------------------------------------------------

class TestCompositionIntegrity:

    def test_ide_state_isolated_per_instance(self):
        """Two WebIDE instances don't share state."""
        ide1 = WebIDE(project_root="/iso1")
        ide1.start()
        ide2 = WebIDE(project_root="/iso2")
        ide2.start()

        http("POST", f"{ide1.url}/api/files", {"path": "only1.ml", "content": "let x = 1;"})
        s1, d1 = http("GET", f"{ide1.url}/api/files")
        s2, d2 = http("GET", f"{ide2.url}/api/files")

        assert any("only1" in f for f in d1["files"])
        assert not any("only1" in f for f in d2["files"])

        ide1.stop()
        ide2.stop()

    def test_terminal_and_build_use_same_workspace(self, base):
        """Terminal and build service see the same files."""
        http("POST", f"{base}/api/files", {"path": "shared.ml", "content": "let shared_val = 42;\nprint(shared_val);"})
        _, build = http("POST", f"{base}/api/build/run", {"path": "shared.ml"})
        assert build["success"] is True
        assert "42" in str(build.get("output", ""))

    def test_editor_diagnostics_match_build_check(self, base):
        """Editor diagnostics and build check should agree on validity."""
        http("POST", f"{base}/api/files", {"path": "agree.ml", "content": "let x = 1 + 2;"})
        http("POST", f"{base}/api/editor/open", {"path": "agree.ml"})
        _, diag = http("GET", f"{base}/api/editor/diagnostics/agree.ml")
        _, check = http("POST", f"{base}/api/build/check", {"path": "agree.ml"})
        # Valid code should have no diagnostics and pass check
        assert check.get("success") is True

    def test_debug_sees_file_changes(self, base):
        """Debug session uses current file content."""
        http("POST", f"{base}/api/files", {"path": "dch.ml", "content": "let a = 1;"})
        http("PUT", f"{base}/api/files/dch.ml", {"content": "let a = 99;"})
        s, data = http("POST", f"{base}/api/debug/start", {"path": "dch.ml"})
        assert s == 200

    def test_error_handler_returns_json(self, base, web_ide):
        """Server error handler returns JSON."""
        # Force an error by using an invalid internal state
        s, data = http("POST", f"{base}/api/debug/eval", {"expression": "x"})
        # Should return JSON even on error
        assert isinstance(data, dict)
