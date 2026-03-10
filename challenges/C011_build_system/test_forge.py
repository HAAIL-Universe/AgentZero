"""
Tests for Forge Build System (C011)
Tests composition of dependency resolver (C008) and stack VM (C010).
"""

import os
import sys
import time
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from forge import (
    Forge, Target, BuildResult, BuildReport, BuildError,
    StalenessChecker, GuardEvaluator, GuardError,
    BuildfileParser, BuildfileError,
    CycleError, MissingDependencyError,
)


# ============================================================
# Target definition tests
# ============================================================

class TestTarget:
    def test_basic_target(self):
        t = Target(name="app")
        assert t.name == "app"
        assert t.depends == []
        assert t.sources == []
        assert t.outputs == []
        assert t.command is None
        assert t.guard is None
        assert not t.phony

    def test_target_with_all_fields(self):
        t = Target(
            name="lib",
            depends=["utils"],
            sources=["src/lib.py"],
            outputs=["build/lib.pyc"],
            command="compile lib",
            guard="version > 2",
            phony=False,
            description="Build the library",
        )
        assert t.depends == ["utils"]
        assert t.guard == "version > 2"

    def test_target_repr(self):
        t = Target(name="test")
        assert "test" in repr(t)


# ============================================================
# BuildResult tests
# ============================================================

class TestBuildResult:
    def test_ok_statuses(self):
        for status in ("built", "skipped", "up-to-date", "guard-false"):
            r = BuildResult("t", status)
            assert r.ok

    def test_failed_status(self):
        r = BuildResult("t", "failed", error="bad")
        assert not r.ok

    def test_duration(self):
        r = BuildResult("t", "built", duration=1.5)
        assert r.duration == 1.5


# ============================================================
# BuildReport tests
# ============================================================

class TestBuildReport:
    def test_empty_report(self):
        r = BuildReport()
        assert r.ok
        assert r.built_count == 0
        assert r.skipped_count == 0
        assert r.failed_count == 0

    def test_mixed_results(self):
        r = BuildReport(results=[
            BuildResult("a", "built"),
            BuildResult("b", "up-to-date"),
            BuildResult("c", "guard-false"),
            BuildResult("d", "failed", error="x"),
        ])
        assert not r.ok
        assert r.built_count == 1
        assert r.skipped_count == 2
        assert r.failed_count == 1

    def test_summary(self):
        r = BuildReport(results=[
            BuildResult("a", "built"),
            BuildResult("b", "skipped"),
        ])
        s = r.summary()
        assert "1 built" in s
        assert "1 skipped" in s
        assert "0 failed" in s


# ============================================================
# StalenessChecker tests
# ============================================================

class TestStaleness:
    def test_phony_always_stale(self):
        checker = StalenessChecker()
        t = Target(name="clean", phony=True)
        assert checker.is_stale(t)

    def test_no_outputs_always_stale(self):
        checker = StalenessChecker()
        t = Target(name="run", sources=["src/x.py"])
        assert checker.is_stale(t)

    def test_missing_output_is_stale(self):
        checker = StalenessChecker()
        t = Target(name="build", outputs=["nonexistent_file_xyz.bin"])
        assert checker.is_stale(t)

    def test_up_to_date(self):
        with tempfile.TemporaryDirectory() as d:
            # Create source and output where output is newer
            src = os.path.join(d, "src.txt")
            out = os.path.join(d, "out.txt")
            with open(src, 'w') as f:
                f.write("source")
            time.sleep(0.05)
            with open(out, 'w') as f:
                f.write("output")

            checker = StalenessChecker(d)
            t = Target(name="t", sources=["src.txt"], outputs=["out.txt"])
            assert not checker.is_stale(t)

    def test_stale_when_source_newer(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.txt")
            with open(out, 'w') as f:
                f.write("output")
            time.sleep(0.05)
            src = os.path.join(d, "src.txt")
            with open(src, 'w') as f:
                f.write("source updated")

            checker = StalenessChecker(d)
            t = Target(name="t", sources=["src.txt"], outputs=["out.txt"])
            assert checker.is_stale(t)

    def test_outputs_exist_no_sources(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.txt")
            with open(out, 'w') as f:
                f.write("data")
            checker = StalenessChecker(d)
            t = Target(name="t", outputs=["out.txt"])
            assert not checker.is_stale(t)

    def test_multiple_sources_one_stale(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.txt")
            src1 = os.path.join(d, "a.txt")
            src2 = os.path.join(d, "b.txt")
            with open(src1, 'w') as f:
                f.write("a")
            with open(out, 'w') as f:
                f.write("out")
            time.sleep(0.05)
            with open(src2, 'w') as f:
                f.write("b newer")

            checker = StalenessChecker(d)
            t = Target(name="t", sources=["a.txt", "b.txt"], outputs=["out.txt"])
            assert checker.is_stale(t)

    def test_multiple_outputs_one_missing(self):
        with tempfile.TemporaryDirectory() as d:
            out1 = os.path.join(d, "out1.txt")
            with open(out1, 'w') as f:
                f.write("out1")
            checker = StalenessChecker(d)
            t = Target(name="t", outputs=["out1.txt", "out2.txt"])
            assert checker.is_stale(t)


# ============================================================
# GuardEvaluator tests (Stack VM integration)
# ============================================================

class TestGuardEvaluator:
    def test_true_guard(self):
        g = GuardEvaluator({"version": 3})
        assert g.evaluate("version >= 3") is True

    def test_false_guard(self):
        g = GuardEvaluator({"version": 2})
        assert g.evaluate("version >= 3") is False

    def test_boolean_variable(self):
        g = GuardEvaluator({"debug": True})
        assert g.evaluate("debug == true") is True

    def test_boolean_false(self):
        g = GuardEvaluator({"debug": False})
        assert g.evaluate("debug == true") is False

    def test_string_comparison(self):
        g = GuardEvaluator({"mode": "release"})
        assert g.evaluate('mode == "release"') is True

    def test_compound_expression(self):
        g = GuardEvaluator({"version": 3, "debug": True})
        assert g.evaluate("version > 2 and debug == true") is True

    def test_arithmetic_in_guard(self):
        g = GuardEvaluator({"x": 10})
        assert g.evaluate("x * 2 > 15") is True

    def test_no_variables(self):
        g = GuardEvaluator()
        assert g.evaluate("1 + 1 == 2") is True

    def test_invalid_expression_raises(self):
        g = GuardEvaluator()
        with pytest.raises(GuardError):
            g.evaluate("@@@invalid")

    def test_numeric_truthy(self):
        g = GuardEvaluator({"x": 42})
        assert g.evaluate("x") is True

    def test_numeric_falsy(self):
        g = GuardEvaluator({"x": 0})
        assert g.evaluate("x") is False


# ============================================================
# BuildfileParser tests
# ============================================================

class TestBuildfileParser:
    def test_empty(self):
        targets, vars = BuildfileParser("").parse()
        assert targets == []
        assert vars == {}

    def test_comments_only(self):
        targets, vars = BuildfileParser("# just a comment\n# another").parse()
        assert targets == []

    def test_var_int(self):
        _, vars = BuildfileParser('var version = 3').parse()
        assert vars["version"] == 3

    def test_var_float(self):
        _, vars = BuildfileParser('var pi = 3.14').parse()
        assert abs(vars["pi"] - 3.14) < 0.001

    def test_var_string(self):
        _, vars = BuildfileParser('var mode = "release"').parse()
        assert vars["mode"] == "release"

    def test_var_bool(self):
        _, vars = BuildfileParser('var debug = true\nvar opt = false').parse()
        assert vars["debug"] is True
        assert vars["opt"] is False

    def test_simple_target(self):
        text = '''
        target "app" {
            command "build app"
        }
        '''
        targets, _ = BuildfileParser(text).parse()
        assert len(targets) == 1
        assert targets[0].name == "app"
        assert targets[0].command == "build app"

    def test_target_with_depends(self):
        text = '''
        target "app" {
            depends ["lib", "utils"]
            command "build"
        }
        '''
        targets, _ = BuildfileParser(text).parse()
        assert targets[0].depends == ["lib", "utils"]

    def test_target_with_sources_outputs(self):
        text = '''
        target "lib" {
            sources ["src/a.py", "src/b.py"]
            outputs ["build/lib.pyc"]
        }
        '''
        targets, _ = BuildfileParser(text).parse()
        assert targets[0].sources == ["src/a.py", "src/b.py"]
        assert targets[0].outputs == ["build/lib.pyc"]

    def test_target_phony(self):
        text = 'target "clean" { phony true }'
        targets, _ = BuildfileParser(text).parse()
        assert targets[0].phony is True

    def test_target_guard(self):
        text = 'target "app" { guard "version >= 3" }'
        targets, _ = BuildfileParser(text).parse()
        assert targets[0].guard == "version >= 3"

    def test_target_description(self):
        text = 'target "app" { description "Build the app" }'
        targets, _ = BuildfileParser(text).parse()
        assert targets[0].description == "Build the app"

    def test_multiple_targets(self):
        text = '''
        target "a" { command "build a" }
        target "b" { depends ["a"] command "build b" }
        '''
        targets, _ = BuildfileParser(text).parse()
        assert len(targets) == 2
        assert targets[0].name == "a"
        assert targets[1].name == "b"
        assert targets[1].depends == ["a"]

    def test_full_buildfile(self):
        text = '''
        var debug = true
        var version = 3

        # The library
        target "utils" {
            sources ["src/utils.py"]
            outputs ["build/utils.pyc"]
            command "compile utils"
        }

        target "app" {
            depends ["utils"]
            sources ["src/app.py"]
            outputs ["build/app.pyc"]
            command "compile app"
            guard "version >= 3"
            description "Main application"
        }

        target "clean" {
            phony true
            command "rm -rf build/"
        }
        '''
        targets, vars = BuildfileParser(text).parse()
        assert vars == {"debug": True, "version": 3}
        assert len(targets) == 3
        assert targets[1].guard == "version >= 3"

    def test_syntax_error_unknown_key(self):
        text = 'target "x" { badkey "val" }'
        with pytest.raises(BuildfileError):
            BuildfileParser(text).parse()

    def test_syntax_error_unexpected_token(self):
        text = 'foobar'
        with pytest.raises(BuildfileError):
            BuildfileParser(text).parse()

    def test_unterminated_string(self):
        text = 'var x = "hello'
        with pytest.raises(BuildfileError):
            BuildfileParser(text).parse()

    def test_unterminated_target(self):
        text = 'target "x" {'
        with pytest.raises(BuildfileError):
            BuildfileParser(text).parse()

    def test_empty_list(self):
        text = 'target "x" { depends [] }'
        targets, _ = BuildfileParser(text).parse()
        assert targets[0].depends == []

    def test_vars_and_targets_mixed(self):
        text = '''
        var a = 1
        target "x" { command "go" }
        var b = 2
        target "y" { depends ["x"] }
        '''
        targets, vars = BuildfileParser(text).parse()
        assert vars == {"a": 1, "b": 2}
        assert len(targets) == 2


# ============================================================
# Forge engine -- dependency resolution tests
# ============================================================

class TestForgeDependencies:
    def test_single_target(self):
        f = Forge()
        f.target("app")
        order, groups = f.plan()
        assert order == ["app"]
        assert groups == [["app"]]

    def test_linear_chain(self):
        f = Forge()
        f.target("c", depends=["b"])
        f.target("b", depends=["a"])
        f.target("a")
        order, groups = f.plan()
        assert order == ["a", "b", "c"]

    def test_diamond_dependency(self):
        f = Forge()
        f.target("app", depends=["lib", "utils"])
        f.target("lib", depends=["base"])
        f.target("utils", depends=["base"])
        f.target("base")
        order, groups = f.plan()
        assert order.index("base") < order.index("lib")
        assert order.index("base") < order.index("utils")
        assert order.index("lib") < order.index("app")
        assert order.index("utils") < order.index("app")

    def test_parallel_groups(self):
        f = Forge()
        f.target("app", depends=["lib", "utils"])
        f.target("lib")
        f.target("utils")
        _, groups = f.plan()
        assert groups[0] == ["lib", "utils"]  # parallel
        assert groups[1] == ["app"]

    def test_cycle_detection(self):
        f = Forge()
        f.target("a", depends=["b"])
        f.target("b", depends=["a"])
        with pytest.raises(CycleError):
            f.plan()

    def test_unknown_target(self):
        f = Forge()
        f.target("a")
        with pytest.raises(BuildError):
            f.plan(["nonexistent"])

    def test_selective_build(self):
        f = Forge()
        f.target("a")
        f.target("b", depends=["a"])
        f.target("c", depends=["a"])
        # Only build b -- should include a as dependency
        order, _ = f.plan(["b"])
        assert "a" in order
        assert "b" in order
        assert "c" not in order

    def test_selective_build_transitive(self):
        f = Forge()
        f.target("a")
        f.target("b", depends=["a"])
        f.target("c", depends=["b"])
        f.target("d", depends=["a"])
        order, _ = f.plan(["c"])
        assert set(order) == {"a", "b", "c"}


# ============================================================
# Forge engine -- build execution tests
# ============================================================

class TestForgeBuild:
    def test_dry_run(self):
        f = Forge()
        f.target("a", phony=True)
        report = f.build(dry_run=True)
        assert report.ok
        assert report.results[0].status == "skipped"

    def test_build_with_action(self):
        results = []
        f = Forge()
        f.target("a", phony=True, action=lambda: results.append("a"))
        f.target("b", depends=["a"], phony=True, action=lambda: results.append("b"))
        report = f.build()
        assert report.ok
        assert results == ["a", "b"]

    def test_build_order_respected(self):
        order = []
        f = Forge()
        f.target("base", phony=True, action=lambda: order.append("base"))
        f.target("mid", depends=["base"], phony=True, action=lambda: order.append("mid"))
        f.target("top", depends=["mid"], phony=True, action=lambda: order.append("top"))
        report = f.build()
        assert report.ok
        assert order == ["base", "mid", "top"]

    def test_action_failure_stops_build(self):
        def fail():
            raise RuntimeError("boom")

        f = Forge()
        f.target("a", phony=True, action=fail)
        f.target("b", depends=["a"], phony=True, action=lambda: None)
        report = f.build()
        assert not report.ok
        assert report.failed_count == 1
        assert len(report.results) == 1  # stopped after failure
        assert "boom" in report.results[0].error

    def test_no_op_target(self):
        f = Forge()
        f.target("group", depends=["a", "b"])
        f.target("a", phony=True)
        f.target("b", phony=True)
        report = f.build()
        assert report.ok
        assert report.built_count == 3

    def test_force_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.txt")
            out = os.path.join(d, "out.txt")
            with open(src, 'w') as f_:
                f_.write("src")
            time.sleep(0.05)
            with open(out, 'w') as f_:
                f_.write("out")

            built = []
            f = Forge(base_dir=d)
            f.target("t", sources=["src.txt"], outputs=["out.txt"],
                     action=lambda: built.append(1))

            # Without force -- up to date
            report = f.build()
            assert report.results[0].status == "up-to-date"
            assert built == []

            # With force -- rebuild
            report = f.build(force=True)
            assert report.results[0].status == "built"
            assert built == [1]


# ============================================================
# Forge engine -- guard integration tests (VM)
# ============================================================

class TestForgeGuards:
    def test_guard_true(self):
        built = []
        f = Forge()
        f.var("version", 3)
        f.target("app", phony=True, guard="version >= 3",
                 action=lambda: built.append(1))
        report = f.build()
        assert report.ok
        assert report.results[0].status == "built"
        assert built == [1]

    def test_guard_false_skips(self):
        built = []
        f = Forge()
        f.var("version", 2)
        f.target("app", phony=True, guard="version >= 3",
                 action=lambda: built.append(1))
        report = f.build()
        assert report.ok
        assert report.results[0].status == "guard-false"
        assert built == []

    def test_guard_with_boolean(self):
        built = []
        f = Forge()
        f.var("debug", True)
        f.target("debug_info", phony=True, guard="debug == true",
                 action=lambda: built.append(1))
        report = f.build()
        assert built == [1]

    def test_guard_false_doesnt_block_dependents(self):
        """If a guard is false, the target is skipped but dependents still build."""
        order = []
        f = Forge()
        f.var("skip", False)
        f.target("optional", phony=True, guard="skip == true",
                 action=lambda: order.append("optional"))
        f.target("main", depends=["optional"], phony=True,
                 action=lambda: order.append("main"))
        report = f.build()
        assert report.ok
        assert "optional" not in order
        assert "main" in order

    def test_guard_compound_expression(self):
        f = Forge()
        f.var("version", 3)
        f.var("debug", True)
        f.target("t", phony=True, guard="version > 2 and debug == true")
        report = f.build()
        assert report.results[0].status == "built"

    def test_guard_arithmetic(self):
        f = Forge()
        f.var("cores", 4)
        f.target("parallel", phony=True, guard="cores * 2 >= 8")
        report = f.build()
        assert report.results[0].status == "built"

    def test_guard_string_variable(self):
        f = Forge()
        f.var("mode", "release")
        f.target("optimize", phony=True, guard='mode == "release"')
        report = f.build()
        assert report.results[0].status == "built"


# ============================================================
# Forge -- buildfile loading tests
# ============================================================

class TestForgeBuildfile:
    def test_load_from_string(self):
        f = Forge()
        f.load_buildfile_str('''
        var version = 5
        target "app" {
            depends ["lib"]
            command "build"
            guard "version > 3"
        }
        target "lib" {
            command "build lib"
        }
        ''')
        assert f.variables["version"] == 5
        assert "app" in f.targets
        assert "lib" in f.targets
        assert f.targets["app"].depends == ["lib"]
        assert f.targets["app"].guard == "version > 3"

    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            bf = os.path.join(d, "Forgefile")
            with open(bf, 'w') as fh:
                fh.write('target "hello" { phony true command "echo hi" }')

            f = Forge(base_dir=d)
            f.load_buildfile(bf)
            assert "hello" in f.targets

    def test_buildfile_then_plan(self):
        f = Forge()
        f.load_buildfile_str('''
        target "a" { command "build a" }
        target "b" { depends ["a"] command "build b" }
        target "c" { depends ["a", "b"] command "build c" }
        ''')
        order, groups = f.plan()
        assert order == ["a", "b", "c"]

    def test_buildfile_vars_as_guards(self):
        built = []
        f = Forge()
        f.load_buildfile_str('''
        var level = 5
        target "advanced" {
            phony true
            guard "level >= 5"
        }
        ''')
        f.targets["advanced"].action = lambda: built.append(1)
        report = f.build()
        assert report.results[0].status == "built"

    def test_buildfile_guard_blocks(self):
        f = Forge()
        f.load_buildfile_str('''
        var level = 2
        target "advanced" {
            phony true
            guard "level >= 5"
        }
        ''')
        report = f.build()
        assert report.results[0].status == "guard-false"


# ============================================================
# Forge -- staleness integration tests
# ============================================================

class TestForgeStaleness:
    def test_rebuild_when_source_updated(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.txt")
            out = os.path.join(d, "out.txt")

            # First: create output
            with open(src, 'w') as fh:
                fh.write("v1")
            time.sleep(0.05)
            with open(out, 'w') as fh:
                fh.write("compiled v1")

            built = []
            f = Forge(base_dir=d)
            f.target("t", sources=["src.txt"], outputs=["out.txt"],
                     action=lambda: built.append(1))

            # Should be up to date
            report = f.build()
            assert report.results[0].status == "up-to-date"

            # Update source
            time.sleep(0.05)
            with open(src, 'w') as fh:
                fh.write("v2")

            report = f.build()
            assert report.results[0].status == "built"
            assert built == [1]

    def test_missing_output_triggers_build(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.txt")
            with open(src, 'w') as fh:
                fh.write("data")

            built = []
            f = Forge(base_dir=d)
            f.target("t", sources=["src.txt"], outputs=["out.txt"],
                     action=lambda: built.append(1))
            report = f.build()
            assert report.results[0].status == "built"


# ============================================================
# Forge -- add_target method
# ============================================================

class TestForgeAddTarget:
    def test_add_target_object(self):
        f = Forge()
        t = Target(name="x", phony=True)
        f.add_target(t)
        assert "x" in f.targets

    def test_add_multiple(self):
        f = Forge()
        f.add_target(Target(name="a"))
        f.add_target(Target(name="b", depends=["a"]))
        order, _ = f.plan()
        assert order == ["a", "b"]


# ============================================================
# Forge -- log tests
# ============================================================

class TestForgeLog:
    def test_log_records_actions(self):
        f = Forge()
        f.target("a", phony=True, action=lambda: None)
        f.build()
        assert len(f.log) > 0
        assert any("BUILT" in l for l in f.log)

    def test_log_records_up_to_date(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.txt")
            with open(out, 'w') as fh:
                fh.write("data")
            f = Forge(base_dir=d)
            f.target("t", outputs=["out.txt"])
            f.build()
            assert any("UP-TO-DATE" in l for l in f.log)

    def test_log_records_guard_false(self):
        f = Forge()
        f.var("x", 0)
        f.target("t", phony=True, guard="x > 5")
        f.build()
        assert any("GUARD-FALSE" in l for l in f.log)

    def test_log_records_dry_run(self):
        f = Forge()
        f.target("t", phony=True)
        f.build(dry_run=True)
        assert any("WOULD-BUILD" in l for l in f.log)


# ============================================================
# Forge -- variables
# ============================================================

class TestForgeVariables:
    def test_var_chaining(self):
        f = Forge()
        result = f.var("a", 1).var("b", 2)
        assert result is f  # chaining works
        assert f.variables == {"a": 1, "b": 2}

    def test_vars_in_report(self):
        f = Forge()
        f.var("mode", "debug")
        f.target("t", phony=True)
        report = f.build()
        assert report.variables == {"mode": "debug"}


# ============================================================
# Edge cases and integration
# ============================================================

class TestEdgeCases:
    def test_empty_build(self):
        f = Forge()
        report = f.build()
        assert report.ok
        assert report.results == []

    def test_target_no_action_no_command(self):
        f = Forge()
        f.target("group", depends=["a"])
        f.target("a", phony=True)
        report = f.build()
        assert report.ok

    def test_large_dependency_tree(self):
        f = Forge()
        for i in range(50):
            deps = [f"t{i-1}"] if i > 0 else []
            f.target(f"t{i}", depends=deps, phony=True)
        order, _ = f.plan()
        assert len(order) == 50
        assert order[0] == "t0"
        assert order[-1] == "t49"

    def test_wide_parallel_tree(self):
        f = Forge()
        for i in range(20):
            f.target(f"leaf{i}", phony=True)
        f.target("root", depends=[f"leaf{i}" for i in range(20)], phony=True)
        _, groups = f.plan()
        assert len(groups[0]) == 20  # all leaves parallel
        assert groups[1] == ["root"]

    def test_full_pipeline_buildfile_to_execution(self):
        """End-to-end: parse buildfile, plan, execute with guards."""
        built = []
        f = Forge()
        f.load_buildfile_str('''
        var version = 3
        var debug = false

        target "base" {
            phony true
        }
        target "lib" {
            depends ["base"]
            phony true
            guard "version >= 2"
        }
        target "app" {
            depends ["lib"]
            phony true
            guard "version >= 3"
        }
        target "debug_extras" {
            depends ["app"]
            phony true
            guard "debug == true"
        }
        ''')
        # Attach actions
        for name in f.targets:
            n = name  # capture
            f.targets[n].action = lambda n=n: built.append(n)

        report = f.build()
        assert report.ok
        assert "base" in built
        assert "lib" in built
        assert "app" in built
        assert "debug_extras" not in built  # guard blocked

    def test_buildfile_negative_number(self):
        _, vars = BuildfileParser('var x = -5').parse()
        assert vars["x"] == -5

    def test_selective_build_with_guard(self):
        f = Forge()
        f.var("enabled", False)
        f.target("a", phony=True, guard="enabled == true")
        f.target("b", depends=["a"], phony=True)
        report = f.build(["b"])
        assert report.ok
        # a was guard-blocked, b still built
        statuses = {r.target: r.status for r in report.results}
        assert statuses["a"] == "guard-false"
        assert statuses["b"] == "built"


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
