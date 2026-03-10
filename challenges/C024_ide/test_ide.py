"""
Tests for C024 Integrated Development Environment
Target: 150+ tests covering all subsystems and cross-system integration.
"""

import pytest
import sys
import os
import json
import copy

sys.path.insert(0, os.path.dirname(__file__))
from ide import (
    IDE, Workspace, WorkspaceFile,
    EditorService, TerminalService, DebugService, PackageService,
    BuildService, BuildResult, IDEEvent, IDENotification
)


# ============================================================
# Workspace Tests
# ============================================================

class TestWorkspace:
    def test_create_file(self):
        ws = Workspace("/proj")
        f = ws.create_file("main.ml", "let x = 1;")
        assert f.path == "/proj/main.ml"
        assert f.content == "let x = 1;"
        assert f.version == 0
        assert not f.dirty

    def test_create_file_absolute_path(self):
        ws = Workspace("/proj")
        f = ws.create_file("/proj/main.ml", "let x = 1;")
        assert f.path == "/proj/main.ml"

    def test_write_file_creates(self):
        ws = Workspace("/proj")
        f = ws.write_file("new.ml", "let y = 2;")
        assert f.content == "let y = 2;"
        assert ws.exists("new.ml")

    def test_write_file_updates(self):
        ws = Workspace("/proj")
        ws.create_file("main.ml", "let x = 1;")
        f = ws.write_file("main.ml", "let x = 2;")
        assert f.content == "let x = 2;"
        assert f.version == 1
        assert f.dirty

    def test_read_file(self):
        ws = Workspace("/proj")
        ws.create_file("main.ml", "let x = 1;")
        assert ws.read_file("main.ml") == "let x = 1;"

    def test_read_nonexistent(self):
        ws = Workspace("/proj")
        assert ws.read_file("nope.ml") is None

    def test_delete_file(self):
        ws = Workspace("/proj")
        ws.create_file("main.ml", "let x = 1;")
        assert ws.delete_file("main.ml")
        assert not ws.exists("main.ml")

    def test_delete_nonexistent(self):
        ws = Workspace("/proj")
        assert not ws.delete_file("nope.ml")

    def test_list_files(self):
        ws = Workspace("/proj")
        ws.create_file("a.ml", "")
        ws.create_file("b.ml", "")
        ws.create_file("c.txt", "")
        assert ws.list_files() == ["/proj/a.ml", "/proj/b.ml", "/proj/c.txt"]

    def test_list_files_filter(self):
        ws = Workspace("/proj")
        ws.create_file("a.ml", "")
        ws.create_file("b.ml", "")
        ws.create_file("c.txt", "")
        assert ws.list_files("*.ml") == ["/proj/a.ml", "/proj/b.ml"]

    def test_exists(self):
        ws = Workspace("/proj")
        ws.create_file("main.ml", "")
        assert ws.exists("main.ml")
        assert not ws.exists("other.ml")

    def test_get_file(self):
        ws = Workspace("/proj")
        ws.create_file("main.ml", "content")
        f = ws.get_file("main.ml")
        assert f is not None
        assert f.content == "content"

    def test_uri_conversion(self):
        ws = Workspace("/proj")
        assert ws.to_uri("main.ml") == "file:///proj/main.ml"
        assert ws.from_uri("file:///proj/main.ml") == "/proj/main.ml"

    def test_change_watcher(self):
        ws = Workspace("/proj")
        events = []
        ws.on_change(lambda evt, path: events.append((evt, path)))
        ws.create_file("a.ml", "")
        ws.write_file("a.ml", "updated")
        ws.delete_file("a.ml")
        assert len(events) == 3
        assert events[0][0] == "created"
        assert events[1][0] == "changed"
        assert events[2][0] == "deleted"

    def test_multiple_writes_increment_version(self):
        ws = Workspace("/proj")
        ws.create_file("main.ml", "v0")
        ws.write_file("main.ml", "v1")
        ws.write_file("main.ml", "v2")
        f = ws.get_file("main.ml")
        assert f.version == 2


# ============================================================
# Editor Service Tests (LSP)
# ============================================================

class TestEditorService:
    def setup_method(self):
        self.ws = Workspace("/proj")
        self.editor = EditorService(self.ws)
        self.editor.initialize()

    def teardown_method(self):
        self.editor.shutdown()

    def test_initialize(self):
        assert self.editor._initialized

    def test_open_file_no_errors(self):
        self.ws.create_file("main.ml", "let x = 1;")
        diags = self.editor.open_file("main.ml")
        assert isinstance(diags, list)
        # Valid code should have no errors
        error_diags = [d for d in diags if d.get('severity', 0) == 1]
        assert len(error_diags) == 0

    def test_open_file_with_errors(self):
        self.ws.create_file("bad.ml", "let x = ;")
        diags = self.editor.open_file("bad.ml")
        assert len(diags) > 0

    def test_update_file(self):
        self.ws.create_file("main.ml", "let x = 1;")
        self.editor.open_file("main.ml")
        diags = self.editor.update_file("main.ml", "let x = ;")
        assert len(diags) > 0

    def test_close_file(self):
        self.ws.create_file("main.ml", "let x = 1;")
        self.editor.open_file("main.ml")
        self.editor.close_file("main.ml")
        assert not self.editor.is_open("main.ml")

    def test_complete(self):
        self.ws.create_file("main.ml", "let count = 1;\ncou")
        self.editor.open_file("main.ml")
        items = self.editor.complete("main.ml", 1, 3)
        assert isinstance(items, list)

    def test_hover(self):
        self.ws.create_file("main.ml", "let x = 42;\nx;")
        self.editor.open_file("main.ml")
        result = self.editor.hover("main.ml", 1, 0)
        # May or may not have hover info depending on analysis

    def test_symbols(self):
        self.ws.create_file("main.ml", "let x = 1;\nfn add(a, b) { a + b; }")
        self.editor.open_file("main.ml")
        syms = self.editor.symbols("main.ml")
        assert isinstance(syms, list)
        assert len(syms) >= 1  # At least the function

    def test_is_open(self):
        self.ws.create_file("main.ml", "let x = 1;")
        assert not self.editor.is_open("main.ml")
        self.editor.open_file("main.ml")
        assert self.editor.is_open("main.ml")

    def test_get_open_files(self):
        self.ws.create_file("a.ml", "let a = 1;")
        self.ws.create_file("b.ml", "let b = 2;")
        self.editor.open_file("a.ml")
        self.editor.open_file("b.ml")
        assert len(self.editor.get_open_files()) == 2

    def test_add_package_completions(self):
        self.ws.create_file("main.ml", "let x = 1;\n")
        self.editor.open_file("main.ml")
        self.editor.add_package_completions(["math", "io"])
        items = self.editor.complete("main.ml", 1, 0)
        labels = [i['label'] for i in items]
        assert "math" in labels
        assert "io" in labels

    def test_refresh_all(self):
        self.ws.create_file("a.ml", "let x = 1;")
        self.ws.create_file("b.ml", "let y = 2;")
        self.editor.open_file("a.ml")
        self.editor.open_file("b.ml")
        results = self.editor.refresh_all()
        assert "a.ml" in results
        assert "b.ml" in results

    def test_open_nonexistent_file(self):
        diags = self.editor.open_file("nope.ml")
        assert diags == []

    def test_definition(self):
        self.ws.create_file("main.ml", "let x = 1;\nprint(x);")
        self.editor.open_file("main.ml")
        result = self.editor.definition("main.ml", 1, 6)
        # Definition may or may not resolve depending on LSP analysis


# ============================================================
# Terminal Service Tests (REPL)
# ============================================================

class TestTerminalService:
    def setup_method(self):
        self.ws = Workspace("/proj")
        self.term = TerminalService(self.ws)

    def test_eval_expression(self):
        result = self.term.eval("1 + 2;")
        assert result['error'] is None
        assert result['result'] == 3

    def test_eval_variable(self):
        self.term.eval("let x = 10;")
        result = self.term.eval("x;")
        assert result['result'] == 10

    def test_eval_error(self):
        result = self.term.eval("let = ;")
        assert result['error'] is not None

    def test_eval_empty(self):
        result = self.term.eval("")
        assert result['error'] is None
        assert result['result'] is None

    def test_persistent_state(self):
        self.term.eval("let a = 5;")
        self.term.eval("let b = 10;")
        result = self.term.eval("a + b;")
        assert result['result'] == 15

    def test_run_file(self):
        self.ws.create_file("main.ml", "let x = 42;\nprint(x);")
        result = self.term.run_file("main.ml")
        assert result['error'] is None
        assert "42" in result['output']

    def test_run_nonexistent_file(self):
        result = self.term.run_file("nope.ml")
        assert result['error'] is not None

    def test_get_variables(self):
        self.term.eval("let x = 1;")
        self.term.eval("let y = 2;")
        vars = self.term.get_variables()
        assert 'x' in vars
        assert 'y' in vars

    def test_get_history(self):
        self.term.eval("1 + 1;")
        self.term.eval("2 + 2;")
        hist = self.term.get_history()
        assert len(hist) == 2

    def test_output_log(self):
        self.term.eval("1 + 1;")
        log = self.term.get_output_log()
        assert len(log) == 1
        assert log[0]['input'] == "1 + 1;"

    def test_reset(self):
        self.term.eval("let x = 1;")
        self.term.reset()
        result = self.term.eval("x;")
        assert result['error'] is not None

    def test_is_complete(self):
        assert self.term.is_complete("let x = 1;")
        assert not self.term.is_complete("if (true) {")

    def test_inject_variables(self):
        self.term.inject_variables({'x': 42})
        result = self.term.eval("x;")
        assert result['result'] == 42

    def test_print_output(self):
        result = self.term.eval("print(42);")
        assert "42" in result['output']

    def test_function_definition(self):
        result = self.term.eval("fn double(x) { return x * 2; }\ndouble(5);")
        assert result['result'] == 10

    def test_repl_command(self):
        result = self.term.eval(".help")
        assert result['output'] is not None
        assert len(result['output']) > 0


# ============================================================
# Debug Service Tests
# ============================================================

class TestDebugService:
    def setup_method(self):
        self.ws = Workspace("/proj")
        self.debug = DebugService(self.ws)

    def test_start_debug(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = 2;\nprint(x + y);")
        result = self.debug.start("main.ml")
        assert result['error'] is None
        assert result['session_id'] is not None

    def test_start_nonexistent(self):
        result = self.debug.start("nope.ml")
        assert result['error'] is not None

    def test_start_invalid_code(self):
        self.ws.create_file("bad.ml", "let = ;")
        result = self.debug.start("bad.ml")
        assert result['error'] is not None

    def test_step(self):
        self.ws.create_file("main.ml", "let x = 1;")
        self.debug.start("main.ml")
        event = self.debug.step()
        assert event is not None
        assert 'reason' in event

    def test_step_line(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = 2;")
        self.debug.start("main.ml")
        event = self.debug.step_line()
        assert event is not None

    def test_continue(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = 2;")
        self.debug.start("main.ml")
        event = self.debug.continue_execution()
        assert event is not None
        assert event['reason'] == 'HALT'

    def test_breakpoint(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = 2;\nlet z = 3;")
        self.debug.start("main.ml")
        bp_id = self.debug.add_breakpoint(line=2)
        assert bp_id is not None
        event = self.debug.continue_execution()
        assert event['reason'] == 'BREAKPOINT'

    def test_remove_breakpoint(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = 2;")
        self.debug.start("main.ml")
        bp_id = self.debug.add_breakpoint(line=2)
        assert self.debug.remove_breakpoint(bp_id)

    def test_watch(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = x + 1;")
        self.debug.start("main.ml")
        w_id = self.debug.add_watch("x")
        assert w_id is not None

    def test_get_context(self):
        self.ws.create_file("main.ml", "let x = 1;")
        self.debug.start("main.ml")
        self.debug.step_line()
        ctx = self.debug.get_context()
        assert ctx is not None
        assert 'variables' in ctx

    def test_disassembly(self):
        self.ws.create_file("main.ml", "let x = 1;")
        self.debug.start("main.ml")
        dis = self.debug.get_disassembly()
        assert dis is not None

    def test_eval_expression(self):
        self.ws.create_file("main.ml", "let x = 42;")
        self.debug.start("main.ml")
        self.debug.continue_execution()
        result = self.debug.eval_expression("x")
        assert result is not None

    def test_multiple_sessions(self):
        self.ws.create_file("a.ml", "let a = 1;")
        self.ws.create_file("b.ml", "let b = 2;")
        self.debug.start("/proj/a.ml")
        self.debug.start("/proj/b.ml")
        assert len(self.debug.list_sessions()) == 2

    def test_stop_session(self):
        self.ws.create_file("main.ml", "let x = 1;")
        self.debug.start("main.ml")
        assert self.debug.stop()
        assert len(self.debug.list_sessions()) == 0

    def test_active_session(self):
        self.ws.create_file("a.ml", "let a = 1;")
        self.ws.create_file("b.ml", "let b = 2;")
        self.debug.start("/proj/a.ml")
        self.debug.start("/proj/b.ml")
        assert self.debug.get_active_session() == "/proj/b.ml"
        self.debug.set_active_session("/proj/a.ml")
        assert self.debug.get_active_session() == "/proj/a.ml"

    def test_start_source(self):
        result = self.debug.start_source("let x = 1;", session_id="test")
        assert result['error'] is None
        assert "test" in self.debug.list_sessions()

    def test_step_no_session(self):
        assert self.debug.step() is None

    def test_enable_trace(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = 2;")
        self.debug.start("main.ml")
        self.debug.enable_trace(True)
        self.debug.continue_execution()
        trace = self.debug.get_trace()
        assert trace is not None
        assert len(trace) > 0

    def test_get_source_line(self):
        self.ws.create_file("main.ml", "let x = 1;\nlet y = 2;")
        self.debug.start("main.ml")
        line = self.debug.get_source_line(1)
        assert line is not None

    def test_step_over(self):
        self.ws.create_file("main.ml", "fn foo() { 1; }\nfoo();")
        self.debug.start("main.ml")
        event = self.debug.step_over()
        assert event is not None

    def test_step_out(self):
        self.ws.create_file("main.ml", "fn foo() { let x = 1; x; }\nfoo();")
        self.debug.start("main.ml")
        # Step into the function first
        self.debug.step_line()
        event = self.debug.step_out()
        assert event is not None

    def test_debug_env(self):
        self.ws.create_file("main.ml", "print(x);")
        result = self.debug.start("main.ml", env={'x': 99})
        assert result['error'] is None


# ============================================================
# Package Service Tests
# ============================================================

class TestPackageService:
    def setup_method(self):
        self.ws = Workspace("/proj")
        self.pkg = PackageService(self.ws)

    def test_publish(self):
        spec = self.pkg.publish("math", "1.0.0", files={"main.ml": "let pi = 3;"})
        assert spec.name == "math"

    def test_add_dependency(self):
        self.pkg.publish("math", "1.0.0")
        resolved = self.pkg.add_dependency("math", "^1.0.0")
        assert "math" in resolved

    def test_remove_dependency(self):
        self.pkg.publish("math", "1.0.0")
        self.pkg.add_dependency("math", "^1.0.0")
        removed = self.pkg.remove_dependency("math")
        assert "math" in removed

    def test_install(self):
        self.pkg.publish("a", "1.0.0")
        self.pkg.publish("b", "1.0.0")
        self.pkg._project_requirements = {"a": "^1.0.0", "b": "^1.0.0"}
        resolved = self.pkg.install()
        assert "a" in resolved
        assert "b" in resolved

    def test_list_installed(self):
        self.pkg.publish("math", "1.0.0")
        self.pkg.add_dependency("math", "^1.0.0")
        installed = self.pkg.list_installed()
        assert len(installed) >= 1
        names = [p['name'] for p in installed]
        assert "math" in names

    def test_get_requirements(self):
        self.pkg.publish("math", "1.0.0")
        self.pkg.add_dependency("math", "^1.0.0")
        reqs = self.pkg.get_requirements()
        assert "math" in reqs['dependencies']

    def test_dev_dependency(self):
        self.pkg.publish("test-lib", "1.0.0")
        self.pkg.add_dependency("test-lib", "^1.0.0", dev=True)
        reqs = self.pkg.get_requirements()
        assert "test-lib" in reqs['devDependencies']

    def test_installed_package_names(self):
        self.pkg.publish("a", "1.0.0")
        self.pkg.publish("b", "1.0.0")
        self.pkg.add_dependency("a", "^1.0.0")
        self.pkg.add_dependency("b", "^1.0.0")
        names = self.pkg.installed_package_names()
        assert "a" in names
        assert "b" in names

    def test_get_lockfile(self):
        self.pkg.publish("math", "1.0.0")
        self.pkg.add_dependency("math", "^1.0.0")
        lockfile = self.pkg.get_lockfile()
        assert isinstance(lockfile, dict)

    def test_dependency_tree(self):
        self.pkg.publish("a", "1.0.0", deps={"b": "^1.0.0"})
        self.pkg.publish("b", "1.0.0")
        self.pkg.add_dependency("a", "^1.0.0")
        tree = self.pkg.dependency_tree()
        assert isinstance(tree, dict)

    def test_audit(self):
        self.pkg.publish("math", "1.0.0")
        self.pkg.add_dependency("math", "^1.0.0")
        result = self.pkg.audit()
        assert isinstance(result, dict)

    def test_get_install_log(self):
        self.pkg.publish("math", "1.0.0")
        self.pkg.add_dependency("math", "^1.0.0")
        log = self.pkg.get_install_log()
        assert isinstance(log, list)

    def test_transitive_deps(self):
        self.pkg.publish("core", "1.0.0")
        self.pkg.publish("utils", "1.0.0", deps={"core": "^1.0.0"})
        resolved = self.pkg.add_dependency("utils", "^1.0.0")
        assert "core" in resolved
        assert "utils" in resolved


# ============================================================
# Build Service Tests
# ============================================================

class TestBuildService:
    def setup_method(self):
        self.ws = Workspace("/proj")
        self.build = BuildService(self.ws)

    def test_check_valid(self):
        self.ws.create_file("main.ml", "let x = 1;")
        result = self.build.check("main.ml")
        assert result['success']

    def test_check_invalid(self):
        self.ws.create_file("bad.ml", "let = ;")
        result = self.build.check("bad.ml")
        assert not result['success']

    def test_check_nonexistent(self):
        result = self.build.check("nope.ml")
        assert not result['success']

    def test_compile(self):
        self.ws.create_file("main.ml", "let x = 1;")
        result = self.build.compile("main.ml")
        assert result['success']
        assert result['chunk'] is not None

    def test_compile_error(self):
        self.ws.create_file("bad.ml", "let = ;")
        result = self.build.compile("bad.ml")
        assert not result['success']

    def test_run(self):
        self.ws.create_file("main.ml", "1 + 2;")
        result = self.build.run("main.ml")
        assert result.success
        assert result.result == 3

    def test_run_with_output(self):
        self.ws.create_file("main.ml", "print(42);")
        result = self.build.run("main.ml")
        assert result.success
        assert "42" in result.output

    def test_run_error(self):
        self.ws.create_file("bad.ml", "let = ;")
        result = self.build.run("bad.ml")
        assert not result.success
        assert result.error is not None

    def test_run_nonexistent(self):
        result = self.build.run("nope.ml")
        assert not result.success

    def test_run_source(self):
        result = self.build.run_source("1 + 2;")
        assert result.success
        assert result.result == 3

    def test_disassemble(self):
        self.ws.create_file("main.ml", "let x = 1;")
        text = self.build.disassemble("main.ml")
        assert text is not None
        assert len(text) > 0

    def test_disassemble_nonexistent(self):
        assert self.build.disassemble("nope.ml") is None

    def test_build_log(self):
        self.ws.create_file("main.ml", "let x = 1;")
        self.build.run("main.ml")
        log = self.build.get_build_log()
        assert len(log) == 1
        assert log[0]['success']

    def test_build_result_to_dict(self):
        result = BuildResult(success=True, output=[42], result=42, steps=5)
        d = result.to_dict()
        assert d['success']
        assert d['result'] == 42
        assert d['steps'] == 5

    def test_run_with_env(self):
        self.ws.create_file("main.ml", "print(x);")
        result = self.build.run("main.ml", env={'x': 99})
        assert result.success
        assert "99" in result.output

    def test_compile_nonexistent(self):
        result = self.build.compile("nope.ml")
        assert not result['success']

    def test_disassemble_invalid(self):
        self.ws.create_file("bad.ml", "let = ;")
        assert self.build.disassemble("bad.ml") is None


# ============================================================
# IDE Integration Tests
# ============================================================

class TestIDEBasic:
    def setup_method(self):
        self.ide = IDE("/proj")
        self.ide.initialize()

    def teardown_method(self):
        self.ide.shutdown()

    def test_initialize(self):
        assert self.ide._initialized

    def test_create_file(self):
        result = self.ide.create_file("main.ml", "let x = 1;")
        assert 'path' in result
        assert 'diagnostics' in result

    def test_open_file(self):
        self.ide.workspace.create_file("main.ml", "let x = 1;")
        result = self.ide.open_file("main.ml")
        assert 'path' in result

    def test_open_nonexistent(self):
        result = self.ide.open_file("nope.ml")
        assert 'error' in result

    def test_edit_file(self):
        self.ide.create_file("main.ml", "let x = 1;")
        result = self.ide.edit_file("main.ml", "let x = 2;")
        assert 'diagnostics' in result

    def test_close_file(self):
        self.ide.create_file("main.ml", "let x = 1;")
        self.ide.close_file("main.ml")
        assert not self.ide.editor.is_open("main.ml")

    def test_delete_file(self):
        self.ide.create_file("main.ml", "let x = 1;")
        assert self.ide.delete_file("main.ml")
        assert not self.ide.workspace.exists("main.ml")

    def test_delete_closes_editor(self):
        self.ide.create_file("main.ml", "let x = 1;")
        assert self.ide.editor.is_open("main.ml")
        self.ide.delete_file("main.ml")
        assert not self.ide.editor.is_open("main.ml")


class TestIDEEditor:
    def setup_method(self):
        self.ide = IDE("/proj")
        self.ide.initialize()

    def teardown_method(self):
        self.ide.shutdown()

    def test_complete(self):
        self.ide.create_file("main.ml", "let count = 1;\ncou")
        items = self.ide.complete("main.ml", 1, 3)
        assert isinstance(items, list)

    def test_hover(self):
        self.ide.create_file("main.ml", "let x = 42;\nx;")
        self.ide.hover("main.ml", 1, 0)  # Just verify it doesn't crash

    def test_definition(self):
        self.ide.create_file("main.ml", "let x = 1;\nprint(x);")
        self.ide.definition("main.ml", 1, 6)

    def test_symbols(self):
        self.ide.create_file("main.ml", "fn foo(a) { a; }\nfn bar(b) { b; }")
        syms = self.ide.symbols("main.ml")
        assert len(syms) >= 2

    def test_diagnostics_error(self):
        self.ide.create_file("bad.ml", "let = ;")
        diags = self.ide.diagnostics("bad.ml")
        assert len(diags) > 0

    def test_diagnostics_clean(self):
        self.ide.create_file("good.ml", "let x = 1;")
        diags = self.ide.diagnostics("good.ml")
        error_diags = [d for d in diags if d.get('severity', 0) == 1]
        assert len(error_diags) == 0


class TestIDETerminal:
    def setup_method(self):
        self.ide = IDE("/proj")
        self.ide.initialize()

    def teardown_method(self):
        self.ide.shutdown()

    def test_eval(self):
        result = self.ide.eval("1 + 2;")
        assert result['result'] == 3

    def test_run_file(self):
        self.ide.create_file("main.ml", "print(42);")
        result = self.ide.run_file("main.ml")
        assert result['success']
        assert "42" in result['output']

    def test_run_file_in_repl(self):
        self.ide.create_file("main.ml", "let x = 42;")
        self.ide.run_file_in_repl("main.ml")
        result = self.ide.eval("x;")
        assert result['result'] == 42


class TestIDEDebugger:
    def setup_method(self):
        self.ide = IDE("/proj")
        self.ide.initialize()

    def teardown_method(self):
        self.ide.shutdown()

    def test_debug_file(self):
        self.ide.create_file("main.ml", "let x = 1;\nlet y = 2;")
        result = self.ide.debug_file("main.ml")
        assert result['error'] is None

    def test_debug_step(self):
        self.ide.create_file("main.ml", "let x = 1;\nlet y = 2;")
        self.ide.debug_file("main.ml")
        event = self.ide.debug_step()
        assert event is not None

    def test_debug_step_line(self):
        self.ide.create_file("main.ml", "let x = 1;\nlet y = 2;")
        self.ide.debug_file("main.ml")
        event = self.ide.debug_step_line()
        assert event is not None

    def test_debug_continue(self):
        self.ide.create_file("main.ml", "let x = 1;")
        self.ide.debug_file("main.ml")
        event = self.ide.debug_continue()
        assert event['reason'] == 'HALT'

    def test_debug_breakpoint(self):
        self.ide.create_file("main.ml", "let x = 1;\nlet y = 2;\nlet z = 3;")
        self.ide.debug_file("main.ml")
        bp = self.ide.debug_add_breakpoint(line=2)
        assert bp is not None
        event = self.ide.debug_continue()
        assert event['reason'] == 'BREAKPOINT'

    def test_debug_watch(self):
        self.ide.create_file("main.ml", "let x = 1;")
        self.ide.debug_file("main.ml")
        w_id = self.ide.debug_add_watch("x")
        assert w_id is not None

    def test_debug_context(self):
        self.ide.create_file("main.ml", "let x = 42;")
        self.ide.debug_file("main.ml")
        self.ide.debug_continue()
        ctx = self.ide.debug_context()
        assert ctx is not None

    def test_debug_eval(self):
        self.ide.create_file("main.ml", "let x = 42;")
        self.ide.debug_file("main.ml")
        self.ide.debug_continue()
        result = self.ide.debug_eval("x")
        assert result is not None

    def test_debug_stop(self):
        self.ide.create_file("main.ml", "let x = 1;")
        self.ide.debug_file("main.ml")
        assert self.ide.debug_stop()

    def test_debug_step_over(self):
        self.ide.create_file("main.ml", "fn foo() { 1; }\nfoo();")
        self.ide.debug_file("main.ml")
        event = self.ide.debug_step_over()
        assert event is not None

    def test_debug_with_env(self):
        self.ide.create_file("main.ml", "print(x);")
        result = self.ide.debug_file("main.ml", env={'x': 99})
        assert result['error'] is None


class TestIDEPackages:
    def setup_method(self):
        self.ide = IDE("/proj")
        self.ide.initialize()

    def teardown_method(self):
        self.ide.shutdown()

    def test_publish_and_install(self):
        self.ide.publish_package("math", "1.0.0")
        resolved = self.ide.add_dependency("math", "^1.0.0")
        assert "math" in resolved

    def test_remove_dependency(self):
        self.ide.publish_package("math", "1.0.0")
        self.ide.add_dependency("math", "^1.0.0")
        removed = self.ide.remove_dependency("math")
        assert "math" in removed

    def test_list_packages(self):
        self.ide.publish_package("a", "1.0.0")
        self.ide.publish_package("b", "1.0.0")
        self.ide.add_dependency("a", "^1.0.0")
        self.ide.add_dependency("b", "^1.0.0")
        pkgs = self.ide.list_packages()
        names = [p['name'] for p in pkgs]
        assert "a" in names
        assert "b" in names

    def test_dependency_tree(self):
        self.ide.publish_package("core", "1.0.0")
        self.ide.publish_package("utils", "1.0.0", deps={"core": "^1.0.0"})
        self.ide.add_dependency("utils", "^1.0.0")
        tree = self.ide.dependency_tree()
        assert isinstance(tree, dict)

    def test_package_audit(self):
        self.ide.publish_package("math", "1.0.0")
        self.ide.add_dependency("math", "^1.0.0")
        result = self.ide.package_audit()
        assert isinstance(result, dict)

    def test_install_packages(self):
        self.ide.publish_package("a", "1.0.0")
        self.ide.packages._project_requirements = {"a": "^1.0.0"}
        resolved = self.ide.install_packages()
        assert "a" in resolved


# ============================================================
# Cross-System Integration Tests
# ============================================================

class TestCrossSystem:
    def setup_method(self):
        self.ide = IDE("/proj")
        self.ide.initialize()

    def teardown_method(self):
        self.ide.shutdown()

    def test_package_completions_in_editor(self):
        """Installing packages should add them to editor completions."""
        self.ide.publish_package("math", "1.0.0")
        self.ide.publish_package("io", "1.0.0")
        self.ide.add_dependency("math", "^1.0.0")
        self.ide.add_dependency("io", "^1.0.0")
        self.ide.create_file("main.ml", "let x = 1;\n")
        items = self.ide.complete("main.ml", 1, 0)
        labels = [i['label'] for i in items]
        assert "math" in labels
        assert "io" in labels

    def test_repl_uses_file_context(self):
        """Running a file in REPL makes its vars available."""
        self.ide.create_file("lib.ml", "let answer = 42;")
        self.ide.run_file_in_repl("lib.ml")
        result = self.ide.eval("answer;")
        assert result['result'] == 42

    def test_edit_then_run(self):
        """Edit a file, check diagnostics, then run it."""
        self.ide.create_file("main.ml", "let x = 10;")
        diags = self.ide.edit_file("main.ml", "let x = 20;\nprint(x);")
        error_diags = [d for d in diags['diagnostics'] if d.get('severity', 0) == 1]
        assert len(error_diags) == 0
        result = self.ide.run_file("main.ml")
        assert result['success']
        assert "20" in result['output']

    def test_debug_then_inspect(self):
        """Debug a file and inspect variables at breakpoint."""
        self.ide.create_file("main.ml", "let x = 10;\nlet y = 20;\nlet z = x + y;")
        self.ide.debug_file("main.ml")
        self.ide.debug_add_breakpoint(line=3)
        event = self.ide.debug_continue()
        assert event['reason'] == 'BREAKPOINT'
        ctx = self.ide.debug_context()
        assert 'variables' in ctx

    def test_multiple_files_workflow(self):
        """Create multiple files, open them, check all diagnostics."""
        self.ide.create_file("a.ml", "let a = 1;")
        self.ide.create_file("b.ml", "let b = 2;")
        self.ide.create_file("c.ml", "let c = 3;")
        results = self.ide.check_project()
        for path, check in results.items():
            assert check['success'], f"File {path} has errors"

    def test_run_all_files(self):
        """Run all files in workspace."""
        self.ide.create_file("a.ml", "let a = 1;")
        self.ide.create_file("b.ml", "let b = 2;")
        results = self.ide.run_all()
        assert len(results) == 2
        for path, r in results.items():
            assert r['success']

    def test_notification_flow(self):
        """IDE emits notifications for key events."""
        self.ide.create_file("main.ml", "let x = 1;")
        self.ide.edit_file("main.ml", "let x = 2;")
        self.ide.delete_file("main.ml")
        notifs = self.ide.get_notifications()
        events = [n['event'] for n in notifs]
        assert 'FILE_CREATED' in events
        assert 'FILE_CHANGED' in events
        assert 'FILE_DELETED' in events

    def test_event_handler(self):
        """Register event handlers and verify they fire."""
        events_fired = []
        self.ide.on(IDEEvent.FILE_CREATED, lambda e, d: events_fired.append(e))
        self.ide.create_file("main.ml", "let x = 1;")
        assert len(events_fired) == 1

    def test_project_summary(self):
        """Get a comprehensive project summary."""
        self.ide.create_file("main.ml", "let x = 1;")
        self.ide.publish_package("math", "1.0.0")
        self.ide.add_dependency("math", "^1.0.0")
        self.ide.eval("let y = 2;")
        summary = self.ide.get_project_summary()
        assert len(summary['files']) >= 1
        assert len(summary['open_files']) >= 1
        assert len(summary['packages_installed']) >= 1
        assert 'y' in summary['repl_variables']

    def test_refresh_diagnostics(self):
        """Refresh all open file diagnostics."""
        self.ide.create_file("a.ml", "let a = 1;")
        self.ide.create_file("b.ml", "let b = 2;")
        results = self.ide.refresh_diagnostics()
        assert len(results) == 2

    def test_save_and_restore_state(self):
        """Save IDE state and restore it."""
        self.ide.create_file("main.ml", "let x = 42;")
        self.ide.eval("let y = 10;")
        state = self.ide.save_state()

        # Create new IDE and restore
        ide2 = IDE("/proj")
        ide2.initialize()
        ide2.restore_state(state)
        assert ide2.workspace.exists("main.ml")
        content = ide2.workspace.read_file("main.ml")
        assert content == "let x = 42;"
        ide2.shutdown()

    def test_edit_updates_diagnostics(self):
        """Editing a file from valid to invalid updates diagnostics."""
        self.ide.create_file("main.ml", "let x = 1;")
        diags1 = self.ide.diagnostics("main.ml")
        error1 = [d for d in diags1 if d.get('severity', 0) == 1]
        assert len(error1) == 0

        result = self.ide.edit_file("main.ml", "let = ;")
        assert len(result['diagnostics']) > 0

    def test_build_then_debug(self):
        """Run a file, then debug the same file."""
        self.ide.create_file("main.ml", "let x = 5;\nlet y = x * 2;\nprint(y);")
        run = self.ide.run_file("main.ml")
        assert run['success']
        assert "10" in run['output']

        dbg = self.ide.debug_file("main.ml")
        assert dbg['error'] is None
        event = self.ide.debug_continue()
        assert event['reason'] == 'HALT'

    def test_debug_file_with_repl_env(self):
        """Use REPL to define vars, then debug a file that uses them."""
        self.ide.eval("let base = 100;")
        env = self.ide.terminal.get_variables()
        self.ide.create_file("main.ml", "let result = base + 1;\nprint(result);")
        result = self.ide.debug_file("main.ml", env=env)
        assert result['error'] is None
        event = self.ide.debug_continue()
        # Should run with base=100 from REPL env

    def test_package_install_updates_completions(self):
        """Adding a dependency updates editor completions."""
        self.ide.publish_package("http", "1.0.0")
        self.ide.create_file("main.ml", "let x = 1;\n")
        items_before = self.ide.complete("main.ml", 1, 0)
        labels_before = [i['label'] for i in items_before]
        assert "http" not in labels_before

        self.ide.add_dependency("http", "^1.0.0")
        items_after = self.ide.complete("main.ml", 1, 0)
        labels_after = [i['label'] for i in items_after]
        assert "http" in labels_after

    def test_remove_package_updates_completions(self):
        """Removing a dependency removes it from completions."""
        self.ide.publish_package("http", "1.0.0")
        self.ide.add_dependency("http", "^1.0.0")
        self.ide.create_file("main.ml", "let x = 1;\n")
        self.ide.remove_dependency("http")
        items = self.ide.complete("main.ml", 1, 0)
        labels = [i['label'] for i in items]
        assert "http" not in labels

    def test_clear_notifications(self):
        self.ide.create_file("main.ml", "let x = 1;")
        notifs = self.ide.get_notifications(clear=True)
        assert len(notifs) > 0
        assert len(self.ide.get_notifications()) == 0

    def test_full_workflow(self):
        """End-to-end: create project, add deps, write code, debug, run."""
        # 1. Publish packages
        self.ide.publish_package("math", "1.0.0")
        self.ide.publish_package("io", "2.0.0")

        # 2. Add dependencies
        self.ide.add_dependency("math", "^1.0.0")
        self.ide.add_dependency("io", "^2.0.0")

        # 3. Create files
        self.ide.create_file("main.ml",
            "let x = 10;\nlet y = 20;\nlet sum = x + y;\nprint(sum);")
        self.ide.create_file("helper.ml",
            "fn double(n) { return n * 2; }")

        # 4. Check diagnostics
        diags = self.ide.diagnostics("main.ml")
        error_diags = [d for d in diags if d.get('severity', 0) == 1]
        assert len(error_diags) == 0

        # 5. Get completions
        items = self.ide.complete("main.ml", 0, 0)
        assert isinstance(items, list)

        # 6. Run in REPL
        self.ide.run_file_in_repl("helper.ml")
        result = self.ide.eval("double(5);")
        assert result['result'] == 10

        # 7. Build and run
        run_result = self.ide.run_file("main.ml")
        assert run_result['success']
        assert "30" in run_result['output']

        # 8. Debug
        self.ide.debug_file("main.ml")
        self.ide.debug_add_breakpoint(line=3)
        event = self.ide.debug_continue()
        assert event['reason'] == 'BREAKPOINT'
        self.ide.debug_stop()

        # 9. Summary
        summary = self.ide.get_project_summary()
        assert len(summary['files']) >= 2
        assert len(summary['packages_installed']) >= 2


class TestIDEEdgeCases:
    def setup_method(self):
        self.ide = IDE("/proj")
        self.ide.initialize()

    def teardown_method(self):
        self.ide.shutdown()

    def test_shutdown_stops_debug(self):
        self.ide.create_file("main.ml", "let x = 1;")
        self.ide.debug_file("main.ml")
        self.ide.shutdown()
        assert len(self.ide.debugger.list_sessions()) == 0

    def test_empty_workspace(self):
        summary = self.ide.get_project_summary()
        assert summary['files'] == []

    def test_check_empty_project(self):
        results = self.ide.check_project()
        assert results == {}

    def test_run_all_empty(self):
        results = self.ide.run_all()
        assert results == {}

    def test_signature_help(self):
        self.ide.create_file("main.ml", "fn add(a, b) { a + b; }\nadd(")
        result = self.ide.signature_help("main.ml", 1, 4)
        # May or may not have signature info

    def test_multiple_debug_sessions(self):
        self.ide.create_file("a.ml", "let a = 1;")
        self.ide.create_file("b.ml", "let b = 2;")
        self.ide.debug_file("/proj/a.ml")
        self.ide.debug_file("/proj/b.ml")
        assert len(self.ide.debugger.list_sessions()) == 2

    def test_save_state_with_packages(self):
        self.ide.publish_package("math", "1.0.0")
        self.ide.add_dependency("math", "^1.0.0")
        state = self.ide.save_state()
        assert "math" in state['requirements']['dependencies']

    def test_restore_files_only(self):
        state = {
            'root': '/proj',
            'files': {'/proj/main.ml': {'content': 'let x = 1;', 'language': 'minilang', 'version': 0}},
            'open_files': [],
            'requirements': {'dependencies': {}, 'devDependencies': {}},
            'repl_env': {},
        }
        ide2 = IDE("/proj")
        ide2.initialize()
        ide2.restore_state(state)
        assert ide2.workspace.exists("/proj/main.ml")
        ide2.shutdown()

    def test_eval_multiline(self):
        code = """
let a = 1;
let b = 2;
let c = a + b;
c;
"""
        result = self.ide.eval(code)
        assert result['result'] == 3

    def test_run_file_nonexistent(self):
        result = self.ide.run_file("nope.ml")
        assert not result['success']

    def test_debug_nonexistent(self):
        result = self.ide.debug_file("nope.ml")
        assert result['error'] is not None

    def test_workspace_root(self):
        assert self.ide.workspace.root == "/proj"

    def test_build_then_repl(self):
        """Build result doesn't affect REPL; REPL is separate."""
        self.ide.create_file("main.ml", "let x = 100;")
        self.ide.run_file("main.ml")
        # REPL should be independent
        result = self.ide.eval("1 + 1;")
        assert result['result'] == 2
