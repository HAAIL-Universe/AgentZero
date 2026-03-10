"""Tests for V033: Python Code Analyzer."""

import pytest
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from python_analyzer import (
    analyze_file, analyze_directory, analyze_a1_challenges,
    complexity_report, findings_report,
    ComplexityVisitor, DefUseVisitor, DataFlowVisitor,
    taint_analysis, analyze_function, analyze_exception_safety,
    analyze_imports,
    Finding, FindingKind, Severity, FunctionMetrics, FileAnalysis, AnalysisResult,
)


# ============================================================
# ComplexityVisitor tests
# ============================================================

class TestComplexityVisitor:
    def _analyze(self, code):
        tree = ast.parse(code)
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        cv = ComplexityVisitor()
        cv.visit(fn)
        return cv

    def test_linear_function(self):
        cv = self._analyze("def f(x):\n  return x + 1")
        assert cv.cyclomatic == 1
        assert cv.cognitive == 0
        assert cv.max_nesting == 0

    def test_single_if(self):
        cv = self._analyze("def f(x):\n  if x > 0:\n    return 1\n  return 0")
        assert cv.cyclomatic == 2
        assert cv.num_branches == 1
        assert cv.num_returns == 2

    def test_nested_if(self):
        cv = self._analyze(
            "def f(x):\n"
            "  if x > 0:\n"
            "    if x < 10:\n"
            "      return 1\n"
            "  return 0"
        )
        assert cv.cyclomatic == 3
        assert cv.max_nesting == 2
        # Cognitive: if(+1+0) + nested if(+1+1) = 3
        assert cv.cognitive == 3

    def test_while_loop(self):
        cv = self._analyze(
            "def f(n):\n"
            "  i = 0\n"
            "  while i < n:\n"
            "    i += 1\n"
            "  return i"
        )
        assert cv.cyclomatic == 2
        assert cv.has_loop is True

    def test_for_loop(self):
        cv = self._analyze(
            "def f(xs):\n"
            "  s = 0\n"
            "  for x in xs:\n"
            "    s += x\n"
            "  return s"
        )
        assert cv.cyclomatic == 2
        assert cv.has_loop is True

    def test_boolean_ops(self):
        cv = self._analyze(
            "def f(a, b, c):\n"
            "  if a and b and c:\n"
            "    return 1\n"
            "  return 0"
        )
        # if (+1) + and-chain with 3 values (+2)
        assert cv.cyclomatic == 4

    def test_try_except(self):
        cv = self._analyze(
            "def f():\n"
            "  try:\n"
            "    return 1\n"
            "  except ValueError:\n"
            "    return 0\n"
        )
        assert cv.cyclomatic == 2

    def test_ternary(self):
        cv = self._analyze("def f(x):\n  return x if x > 0 else -x")
        assert cv.cyclomatic == 2

    def test_list_comprehension(self):
        cv = self._analyze("def f(xs):\n  return [x for x in xs if x > 0]")
        # for (+1) + if (+1)
        assert cv.cyclomatic == 3

    def test_call_tracking(self):
        cv = self._analyze("def f(x):\n  return abs(x) + max(x, 0)")
        assert 'abs' in cv.calls
        assert 'max' in cv.calls

    def test_deep_nesting(self):
        cv = self._analyze(
            "def f(x):\n"
            "  if x > 0:\n"
            "    if x > 1:\n"
            "      if x > 2:\n"
            "        if x > 3:\n"
            "          if x > 4:\n"
            "            return x"
        )
        assert cv.max_nesting == 5


# ============================================================
# DefUseVisitor tests
# ============================================================

class TestDefUseVisitor:
    def _analyze(self, code):
        tree = ast.parse(code)
        dv = DefUseVisitor()
        dv.visit(tree)
        return dv

    def test_simple_assignment(self):
        dv = self._analyze("x = 1\nprint(x)")
        assert 'x' in dv.defs
        assert 'x' in dv.uses

    def test_import(self):
        dv = self._analyze("import os\nos.path.exists('.')")
        assert 'os' in dv.defs
        assert 'os' in dv.imports
        assert 'os' in dv.uses

    def test_from_import(self):
        dv = self._analyze("from os import path\npath.exists('.')")
        assert 'path' in dv.defs
        assert 'path' in dv.imports
        assert 'path' in dv.uses

    def test_function_params(self):
        dv = self._analyze("def f(a, b):\n  return a + b")
        assert 'a' in dv.defs
        assert 'b' in dv.defs
        assert 'a' in dv.uses
        assert 'b' in dv.uses


# ============================================================
# DataFlowVisitor tests
# ============================================================

class TestDataFlowVisitor:
    def _analyze(self, code):
        tree = ast.parse(code)
        df = DataFlowVisitor()
        df.visit(tree)
        return df

    def test_simple_flow(self):
        df = self._analyze("x = 1\ny = x + 2")
        assert any(f[0] == 'x' and f[1] == 'y' for f in df.flows)

    def test_augmented_assign(self):
        df = self._analyze("x = 1\nx += 2")
        assert any(f[0] == 'x' and f[1] == 'x' and f[3] == 'aug_assign' for f in df.flows)

    def test_tuple_unpack(self):
        df = self._analyze("a = 1\nb = 2\nx, y = a, b")
        assert any(f[1] == 'x' for f in df.flows)
        assert any(f[1] == 'y' for f in df.flows)


# ============================================================
# Taint Analysis tests
# ============================================================

class TestTaintAnalysis:
    def test_direct_taint(self):
        code = "x = user_input\ny = x + 1"
        tree = ast.parse(code)
        findings = taint_analysis(tree, "test.py", {"user_input"})
        assert any(f.kind == FindingKind.TAINTED_FLOW for f in findings)

    def test_no_taint(self):
        code = "x = 42\ny = x + 1"
        tree = ast.parse(code)
        findings = taint_analysis(tree, "test.py", {"user_input"})
        assert len(findings) == 0

    def test_transitive_taint(self):
        code = "x = user_input\ny = x + 1\nz = y * 2"
        tree = ast.parse(code)
        findings = taint_analysis(tree, "test.py", {"user_input"})
        taint_targets = {f.message for f in findings if f.kind == FindingKind.TAINTED_FLOW}
        # y gets tainted from x, z gets tainted from y
        assert len(findings) >= 2

    def test_sensitive_sink(self):
        code = "x = user_input\neval(x)"
        tree = ast.parse(code)
        findings = taint_analysis(tree, "test.py", {"user_input"})
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert "eval" in errors[0].message


# ============================================================
# Exception Safety tests
# ============================================================

class TestExceptionSafety:
    def test_bare_except(self):
        code = "try:\n  x = 1\nexcept:\n  pass"
        tree = ast.parse(code)
        findings = analyze_exception_safety(tree, "test.py")
        assert any(f.kind == FindingKind.BARE_EXCEPT for f in findings)

    def test_broad_except(self):
        code = "try:\n  x = 1\nexcept Exception:\n  pass"
        tree = ast.parse(code)
        findings = analyze_exception_safety(tree, "test.py")
        assert any(f.kind == FindingKind.BROAD_EXCEPT for f in findings)

    def test_specific_except_ok(self):
        code = "try:\n  x = 1\nexcept ValueError:\n  pass"
        tree = ast.parse(code)
        findings = analyze_exception_safety(tree, "test.py")
        assert len(findings) == 0


# ============================================================
# Import Analysis tests
# ============================================================

class TestImportAnalysis:
    def test_unused_import(self):
        code = "import os\nx = 1"
        tree = ast.parse(code)
        imports, findings = analyze_imports(tree, "test.py")
        assert any(f.kind == FindingKind.UNUSED_IMPORT for f in findings)

    def test_used_import(self):
        code = "import os\nos.path.exists('.')"
        tree = ast.parse(code)
        imports, findings = analyze_imports(tree, "test.py")
        unused = [f for f in findings if f.kind == FindingKind.UNUSED_IMPORT]
        assert len(unused) == 0


# ============================================================
# Function Analysis tests
# ============================================================

class TestFunctionAnalysis:
    def _analyze_fn(self, code):
        tree = ast.parse(code)
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        return analyze_function(fn, "test.py")

    def test_mutable_default_list(self):
        metrics, findings = self._analyze_fn("def f(x=[]):\n  return x")
        assert any(f.kind == FindingKind.MUTABLE_DEFAULT for f in findings)

    def test_mutable_default_dict(self):
        metrics, findings = self._analyze_fn("def f(x={}):\n  return x")
        assert any(f.kind == FindingKind.MUTABLE_DEFAULT for f in findings)

    def test_safe_default(self):
        metrics, findings = self._analyze_fn("def f(x=None):\n  return x")
        assert not any(f.kind == FindingKind.MUTABLE_DEFAULT for f in findings)

    def test_inconsistent_return(self):
        code = "def f(x):\n  if x > 0:\n    return x\n  return"
        metrics, findings = self._analyze_fn(code)
        assert any(f.kind == FindingKind.INCONSISTENT_RETURN for f in findings)

    def test_consistent_return(self):
        code = "def f(x):\n  if x > 0:\n    return x\n  return 0"
        metrics, findings = self._analyze_fn(code)
        assert not any(f.kind == FindingKind.INCONSISTENT_RETURN for f in findings)

    def test_long_param_list(self):
        code = "def f(a, b, c, d, e, f_):\n  return a"
        metrics, findings = self._analyze_fn(code)
        assert any(f.kind == FindingKind.LONG_PARAMETER_LIST for f in findings)

    def test_metrics_basic(self):
        code = "def f(x):\n  if x > 0:\n    return x\n  return -x"
        metrics, _ = self._analyze_fn(code)
        assert metrics.name == 'f'
        assert metrics.num_params == 1
        assert metrics.cyclomatic == 2
        assert metrics.num_returns == 2


# ============================================================
# File Analysis tests
# ============================================================

class TestFileAnalysis:
    def test_analyze_self(self):
        """Analyze this test file itself."""
        this_file = os.path.abspath(__file__)
        result = analyze_file(this_file)
        assert result.num_lines > 0
        assert result.num_functions > 0

    def test_analyze_analyzer(self):
        """Analyze the analyzer itself (meta!)."""
        analyzer_path = os.path.join(os.path.dirname(__file__), 'python_analyzer.py')
        result = analyze_file(analyzer_path)
        assert result.num_lines > 0
        assert result.num_functions > 10
        assert result.num_classes > 0


# ============================================================
# A1 Codebase Analysis tests
# ============================================================

class TestA1Analysis:
    @pytest.fixture
    def challenge_dir(self):
        return r'Z:\AgentZero\challenges'

    def test_analyze_c010(self, challenge_dir):
        """C010 stack_vm should be analyzable."""
        path = os.path.join(challenge_dir, 'C010_stack_vm', 'stack_vm.py')
        if not os.path.exists(path):
            pytest.skip("C010 not found")
        result = analyze_file(path)
        assert result.num_lines > 1000
        assert result.num_functions > 20
        # Known: lex() and run() are complex
        complex_fns = [m for m in result.function_metrics if m.cyclomatic > 20]
        assert len(complex_fns) >= 2, "lex() and run() should be flagged as complex"

    def test_analyze_c035(self, challenge_dir):
        """C035 SAT solver should be analyzable."""
        path = os.path.join(challenge_dir, 'C035_sat_solver', 'sat_solver.py')
        if not os.path.exists(path):
            pytest.skip("C035 not found")
        result = analyze_file(path)
        assert result.num_lines > 700
        # _analyze (CDCL conflict analysis) should be flagged
        cdcl = [m for m in result.function_metrics if m.name == '_analyze']
        assert len(cdcl) >= 1
        assert cdcl[0].cyclomatic > 15

    def test_analyze_c037(self, challenge_dir):
        """C037 SMT solver should be analyzable."""
        path = os.path.join(challenge_dir, 'C037_smt_solver', 'smt_solver.py')
        if not os.path.exists(path):
            pytest.skip("C037 not found")
        result = analyze_file(path)
        assert result.num_lines > 1600
        # _check_theory is the most complex function
        theory_fns = [m for m in result.function_metrics if m.name == '_check_theory']
        assert len(theory_fns) >= 1
        assert theory_fns[0].cyclomatic > 40

    def test_analyze_c039(self, challenge_dir):
        """C039 abstract interpreter should be analyzable."""
        path = os.path.join(challenge_dir, 'C039_abstract_interpreter', 'abstract_interpreter.py')
        if not os.path.exists(path):
            pytest.skip("C039 not found")
        result = analyze_file(path)
        assert result.num_lines > 1000
        # sign_mul has high CC due to case analysis
        sign_fns = [m for m in result.function_metrics if m.name == 'sign_mul']
        assert len(sign_fns) >= 1
        assert sign_fns[0].cyclomatic > 20

    def test_full_verification_stack(self, challenge_dir):
        """Analyze the complete verification stack (C010, C035-C039)."""
        result = analyze_a1_challenges(challenge_dir)
        assert len(result.files) >= 6
        assert result.total_findings > 50
        # Should find high-complexity functions
        assert result.findings_by_kind.get('high_cyclomatic', 0) > 10

    def test_unused_imports_detected(self, challenge_dir):
        """Several A1 files have unused imports."""
        result = analyze_a1_challenges(challenge_dir)
        assert result.findings_by_kind.get('unused_import', 0) > 5


# ============================================================
# Report Generation tests
# ============================================================

class TestReports:
    def test_complexity_report(self):
        fa = FileAnalysis(
            file="test.py", num_lines=100, num_functions=2, num_classes=0,
            function_metrics=[
                FunctionMetrics("f1", "test.py", 1, 10, 10, 2, 15, 20, 3, 1, 3, True),
                FunctionMetrics("f2", "test.py", 11, 20, 10, 1, 3, 2, 1, 1, 0, False),
            ]
        )
        result = AnalysisResult(files=[fa])
        report = complexity_report(result)
        assert "f1" in report
        assert "f2" in report
        assert "Cyclomatic" in report

    def test_findings_report(self):
        fa = FileAnalysis(
            file="test.py", num_lines=100, num_functions=1, num_classes=0,
            findings=[Finding(
                kind=FindingKind.HIGH_CYCLOMATIC,
                severity=Severity.WARNING,
                file="test.py", line=1, col=0,
                message="Test finding",
            )]
        )
        result = AnalysisResult(files=[fa], total_findings=1)
        report = findings_report(result)
        assert "Test finding" in report

    def test_summary(self):
        result = AnalysisResult(
            total_findings=5,
            findings_by_severity={"warning": 3, "error": 2},
            findings_by_kind={"high_cyclomatic": 5},
        )
        summary = result.summary()
        assert "5" in summary
        assert "warning" in summary


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_function(self):
        code = "def f():\n  pass"
        tree = ast.parse(code)
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        metrics, findings = analyze_function(fn, "test.py")
        assert metrics.cyclomatic == 1
        assert metrics.cognitive == 0
        assert len(findings) == 0

    def test_async_function(self):
        code = "async def f(x):\n  if x:\n    return 1\n  return 0"
        tree = ast.parse(code)
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)][0]
        metrics, findings = analyze_function(fn, "test.py")
        assert metrics.cyclomatic == 2

    def test_class_method(self):
        code = "class C:\n  def m(self, x):\n    if x:\n      return 1\n    return 0"
        tree = ast.parse(code)
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        metrics, findings = analyze_function(fn, "test.py")
        assert metrics.name == 'm'
        assert metrics.cyclomatic == 2

    def test_generator_expression(self):
        cv = ComplexityVisitor()
        tree = ast.parse("def f(xs):\n  return sum(x for x in xs if x > 0)")
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        cv.visit(fn)
        assert cv.cyclomatic >= 3  # base + for + if

    def test_dict_comprehension(self):
        cv = ComplexityVisitor()
        tree = ast.parse("def f(xs):\n  return {x: x*2 for x in xs if x > 0}")
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
        cv.visit(fn)
        assert cv.cyclomatic >= 3
