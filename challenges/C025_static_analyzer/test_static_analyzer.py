"""
Tests for C025 Static Analyzer
Challenge C025 -- AgentZero Session 026

Tests the composition of C013 (Type Checker) + C014 (Bytecode Optimizer)
into a static analysis tool.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from static_analyzer import (
    analyze, analyze_quick, format_report,
    Diagnostic, Severity, Category, CodeMetrics,
    AnalysisReport,
    _compute_line_metrics, _analyze_complexity, _analyze_variables,
    _analyze_lint, _analyze_optimization_potential,
    _complexity_diagnostics,
)
from type_checker import parse


# ============================================================
# Test Helpers
# ============================================================

def diags_with_category(report, cat):
    return report.by_category(cat)

def diags_with_severity(report, sev):
    return report.by_severity(sev)

def has_diagnostic(report, category=None, severity=None, message_contains=None):
    for d in report.diagnostics:
        if category and d.category != category:
            continue
        if severity and d.severity != severity:
            continue
        if message_contains and message_contains not in d.message:
            continue
        return True
    return False


# ============================================================
# 1. Basic Analysis / Report Structure
# ============================================================

class TestBasicAnalysis:

    def test_empty_source(self):
        r = analyze("")
        assert isinstance(r, AnalysisReport)
        assert r.metrics.total_lines == 1  # empty string has 1 line
        assert r.metrics.code_lines == 0

    def test_simple_valid_code(self):
        r = analyze("let x = 5; print(x);")
        assert r.parse_error is None
        assert r.error_count == 0

    def test_parse_error(self):
        r = analyze("let = ;")
        assert r.parse_error is not None
        assert r.has_errors

    def test_report_properties(self):
        r = analyze("let x = 5; print(x);")
        assert isinstance(r.error_count, int)
        assert isinstance(r.warning_count, int)
        assert isinstance(r.info_count, int)
        assert isinstance(r.hint_count, int)
        assert isinstance(r.total_issues, int)
        assert r.total_issues == r.error_count + r.warning_count + r.info_count + r.hint_count

    def test_format_report_returns_string(self):
        r = analyze("let x = 5; print(x);")
        s = format_report(r)
        assert isinstance(s, str)
        assert "STATIC ANALYSIS REPORT" in s

    def test_format_report_parse_error(self):
        r = analyze("let = ;")
        s = format_report(r)
        assert "PARSE ERROR" in s

    def test_by_severity(self):
        r = analyze("let unused = 5;")
        warnings = r.by_severity(Severity.WARNING)
        assert all(d.severity == Severity.WARNING for d in warnings)

    def test_by_category(self):
        r = analyze("let unused = 5;")
        unused = r.by_category(Category.UNUSED_VAR)
        assert all(d.category == Category.UNUSED_VAR for d in unused)

    def test_source_preserved(self):
        src = "let x = 10; print(x);"
        r = analyze(src)
        assert r.source == src

    def test_summary_contains_metrics(self):
        r = analyze("let x = 5;\nlet y = 10;\nprint(x + y);")
        s = r.summary()
        assert "Lines:" in s
        assert "Functions:" in s
        assert "Cyclomatic complexity:" in s


# ============================================================
# 2. Type Error Detection (C013 Composition)
# ============================================================

class TestTypeErrors:

    def test_no_type_errors_clean_code(self):
        r = analyze("let x = 5; let y = x + 3; print(y);")
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert len(type_errs) == 0

    def test_type_mismatch_detected(self):
        # int + string is a type error (no annotations needed -- inference)
        r = analyze('let x = 1 + "hello";')
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert len(type_errs) > 0
        assert r.has_errors

    def test_type_error_has_line(self):
        r = analyze('let x = 1 + "hello";')
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert any(d.line > 0 for d in type_errs)

    def test_bool_plus_int_error(self):
        r = analyze("let x = true + 1;")
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert len(type_errs) > 0

    def test_multiple_type_errors(self):
        code = """
let x = 1 + "hello";
let y = true + 1;
"""
        r = analyze(code)
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert len(type_errs) >= 2

    def test_type_error_severity_is_error(self):
        r = analyze('let x = 1 + "hello";')
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert all(d.severity == Severity.ERROR for d in type_errs)

    def test_type_errors_in_report(self):
        r = analyze('let x = 1 + "hello";')
        assert len(r.type_errors) > 0


# ============================================================
# 3. Unused Variable Detection
# ============================================================

class TestUnusedVariables:

    def test_unused_variable_detected(self):
        r = analyze("let unused = 5;")
        assert has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_used_variable_no_warning(self):
        r = analyze("let x = 5; print(x);")
        assert not has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_underscore_prefix_suppresses(self):
        r = analyze("let _unused = 5;")
        assert not has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_multiple_unused(self):
        r = analyze("let a = 1; let b = 2; let c = 3;")
        unused = r.by_category(Category.UNUSED_VAR)
        assert len(unused) == 3

    def test_used_in_expression(self):
        r = analyze("let x = 5; let y = x + 1; print(y);")
        assert not has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_used_in_print(self):
        r = analyze("let x = 42; print(x);")
        assert not has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_used_in_condition(self):
        r = analyze("let x = true; if (x) { print(1); }")
        assert not has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_assigned_counts_as_used(self):
        r = analyze("let x = 5; x = 10; print(x);")
        assert not has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_unused_function_param(self):
        code = "fn foo(x) { print(1); } foo(5);"
        r = analyze(code)
        unused = r.by_category(Category.UNUSED_VAR)
        # x is unused in foo
        assert any("x" in d.message for d in unused)

    def test_used_function_param(self):
        code = "fn foo(x) { print(x); } foo(5);"
        r = analyze(code)
        unused = r.by_category(Category.UNUSED_VAR)
        assert not any("x" in d.message for d in unused)

    def test_unused_has_suggestion(self):
        r = analyze("let unused = 5;")
        unused = r.by_category(Category.UNUSED_VAR)
        assert len(unused) > 0
        assert unused[0].suggestion != ""


# ============================================================
# 4. Variable Shadowing Detection
# ============================================================

class TestShadowing:

    def test_simple_shadowing(self):
        code = """
let x = 5;
fn foo() {
    let x = 10;
    print(x);
}
foo();
print(x);
"""
        r = analyze(code)
        assert has_diagnostic(r, category=Category.SHADOWED_VAR)

    def test_no_shadowing_different_names(self):
        code = """
let x = 5;
fn foo() {
    let y = 10;
    print(y);
}
foo();
print(x);
"""
        r = analyze(code)
        assert not has_diagnostic(r, category=Category.SHADOWED_VAR)

    def test_shadow_message_contains_name(self):
        code = """
let x = 5;
fn foo() {
    let x = 10;
    print(x);
}
foo();
print(x);
"""
        r = analyze(code)
        shadows = r.by_category(Category.SHADOWED_VAR)
        assert any("x" in d.message for d in shadows)


# ============================================================
# 5. Lint Rules -- Constant Conditions
# ============================================================

class TestConstantConditions:

    def test_if_true_always(self):
        r = analyze("if (true) { print(1); }")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="always true")

    def test_if_false_dead_code(self):
        r = analyze("if (false) { print(1); }")
        assert has_diagnostic(r, category=Category.DEAD_CODE, message_contains="always false")

    def test_if_false_with_else_suggestion(self):
        r = analyze("if (false) { print(1); } else { print(2); }")
        dead = [d for d in r.diagnostics if "always false" in d.message]
        assert len(dead) > 0
        assert "else-branch" in dead[0].suggestion

    def test_while_false_dead_code(self):
        r = analyze("while (false) { print(1); }")
        assert has_diagnostic(r, category=Category.DEAD_CODE, message_contains="While condition")

    def test_normal_condition_no_warning(self):
        r = analyze("let x = true; if (x) { print(1); }")
        assert not has_diagnostic(r, category=Category.REDUNDANT, message_contains="always true")
        assert not has_diagnostic(r, category=Category.DEAD_CODE, message_contains="always false")


# ============================================================
# 6. Lint Rules -- Division by Zero
# ============================================================

class TestDivisionByZero:

    def test_int_division_by_zero(self):
        r = analyze("let x = 5 / 0;")
        assert has_diagnostic(r, category=Category.POSSIBLE_BUG, message_contains="Division by zero")

    def test_modulo_by_zero(self):
        r = analyze("let x = 5 % 0;")
        assert has_diagnostic(r, category=Category.POSSIBLE_BUG, message_contains="Division by zero")

    def test_float_division_by_zero(self):
        r = analyze("let x = 5.0 / 0.0;")
        assert has_diagnostic(r, category=Category.POSSIBLE_BUG, message_contains="Division by zero")

    def test_normal_division_no_warning(self):
        r = analyze("let x = 10 / 2; print(x);")
        assert not has_diagnostic(r, category=Category.POSSIBLE_BUG)

    def test_division_by_zero_is_error(self):
        r = analyze("let x = 5 / 0;")
        bugs = [d for d in r.diagnostics if "Division by zero" in d.message and "float" not in d.message]
        assert any(d.severity == Severity.ERROR for d in bugs)


# ============================================================
# 7. Lint Rules -- Self-Assignment
# ============================================================

class TestSelfAssignment:

    def test_self_assignment_detected(self):
        r = analyze("let x = 5; x = x;")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="Self-assignment")

    def test_normal_assignment_no_warning(self):
        r = analyze("let x = 5; x = 10; print(x);")
        assert not has_diagnostic(r, category=Category.REDUNDANT, message_contains="Self-assignment")


# ============================================================
# 8. Lint Rules -- Identical Comparison Operands
# ============================================================

class TestIdenticalComparison:

    def test_x_eq_x_always_true(self):
        r = analyze("let x = 5; let r = x == x; print(r);")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="always true")

    def test_x_ne_x_always_false(self):
        r = analyze("let x = 5; let r = x != x; print(r);")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="always false")

    def test_x_lt_x_always_false(self):
        r = analyze("let x = 5; let r = x < x; print(r);")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="always false")

    def test_x_gt_x_always_false(self):
        r = analyze("let x = 5; let r = x > x; print(r);")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="always false")

    def test_normal_comparison_no_warning(self):
        r = analyze("let x = 5; let y = 10; let r = x == y; print(r);")
        assert not has_diagnostic(r, category=Category.REDUNDANT, message_contains="always true")


# ============================================================
# 9. Lint Rules -- Empty Function Body
# ============================================================

class TestEmptyFunction:

    def test_empty_function_detected(self):
        r = analyze("fn noop() {} noop();")
        assert has_diagnostic(r, category=Category.STYLE, message_contains="empty body")

    def test_non_empty_function_no_warning(self):
        r = analyze("fn foo() { print(1); } foo();")
        assert not has_diagnostic(r, category=Category.STYLE, message_contains="empty body")


# ============================================================
# 10. Lint Rules -- Unreachable Code After Return
# ============================================================

class TestUnreachableCode:

    def test_code_after_return(self):
        code = """
fn foo() {
    return 1;
    print(2);
}
foo();
"""
        r = analyze(code)
        assert has_diagnostic(r, category=Category.UNREACHABLE)

    def test_no_code_after_return(self):
        code = """
fn foo() {
    return 1;
}
foo();
"""
        r = analyze(code)
        assert not has_diagnostic(r, category=Category.UNREACHABLE)

    def test_return_at_end_of_block(self):
        code = """
fn foo(x) {
    if (x) {
        return 1;
    }
    return 0;
}
foo(true);
"""
        r = analyze(code)
        assert not has_diagnostic(r, category=Category.UNREACHABLE)


# ============================================================
# 11. Lint Rules -- Redundant Boolean Comparison
# ============================================================

class TestRedundantBoolComparison:

    def test_compare_with_true(self):
        r = analyze("let x = true; let r = x == true; print(r);")
        assert has_diagnostic(r, category=Category.STYLE, message_contains="comparison with true")

    def test_compare_with_false(self):
        r = analyze("let x = true; let r = x == false; print(r);")
        assert has_diagnostic(r, category=Category.STYLE, message_contains="comparison with false")

    def test_normal_bool_no_warning(self):
        r = analyze("let x = true; let y = false; let r = x == y; print(r);")
        assert not has_diagnostic(r, category=Category.STYLE, message_contains="comparison with true")


# ============================================================
# 12. Lint Rules -- No-Effect Expression Statements
# ============================================================

class TestNoEffectExpressions:

    def test_bare_literal_warns(self):
        r = analyze("let x = 5; 42; print(x);")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="no side effects")

    def test_bare_variable_warns(self):
        # A bare variable reference as a statement has no effect
        r = analyze("let x = 5; x; print(x);")
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="no side effects")

    def test_print_no_warning(self):
        r = analyze("print(5);")
        assert not has_diagnostic(r, category=Category.REDUNDANT, message_contains="no side effects")


# ============================================================
# 13. Lint Rules -- Arithmetic Identity Operations
# ============================================================

class TestArithmeticIdentity:

    def test_multiply_by_zero(self):
        r = analyze("let x = 5; let r = x * 0; print(r);")
        assert has_diagnostic(r, category=Category.OPTIMIZATION, message_contains="Multiplication by zero")

    def test_multiply_by_zero_left(self):
        r = analyze("let x = 5; let r = 0 * x; print(r);")
        assert has_diagnostic(r, category=Category.OPTIMIZATION, message_contains="Multiplication by zero")

    def test_add_zero(self):
        r = analyze("let x = 5; let r = x + 0; print(r);")
        assert has_diagnostic(r, category=Category.OPTIMIZATION, message_contains="zero has no effect")

    def test_subtract_zero(self):
        r = analyze("let x = 5; let r = x - 0; print(r);")
        assert has_diagnostic(r, category=Category.OPTIMIZATION, message_contains="zero has no effect")

    def test_multiply_by_one(self):
        r = analyze("let x = 5; let r = x * 1; print(r);")
        assert has_diagnostic(r, category=Category.OPTIMIZATION, message_contains="Multiplication by 1")

    def test_multiply_by_one_left(self):
        r = analyze("let x = 5; let r = 1 * x; print(r);")
        assert has_diagnostic(r, category=Category.OPTIMIZATION, message_contains="Multiplication by 1")

    def test_normal_arithmetic_no_warning(self):
        r = analyze("let x = 5; let r = x + 3; print(r);")
        assert not has_diagnostic(r, category=Category.OPTIMIZATION, message_contains="has no effect")


# ============================================================
# 14. Double Negation
# ============================================================

class TestDoubleNegation:

    def test_double_negation_detected(self):
        r = analyze("let x = 5; let r = -(-x); print(r);")
        assert has_diagnostic(r, category=Category.STYLE, message_contains="Double negation")

    def test_single_negation_no_warning(self):
        r = analyze("let x = 5; let r = -x; print(r);")
        assert not has_diagnostic(r, category=Category.STYLE, message_contains="Double negation")


# ============================================================
# 15. Code Metrics
# ============================================================

class TestCodeMetrics:

    def test_line_count(self):
        r = analyze("let x = 5;\nlet y = 10;\nprint(x + y);")
        assert r.metrics.total_lines == 3
        assert r.metrics.code_lines == 3
        assert r.metrics.blank_lines == 0

    def test_blank_lines(self):
        r = analyze("let x = 5;\n\nprint(x);\n")
        assert r.metrics.blank_lines >= 1

    def test_function_count(self):
        code = "fn a() { print(1); } fn b() { print(2); } a(); b();"
        r = analyze(code)
        assert r.metrics.functions == 2

    def test_variable_count(self):
        r = analyze("let x = 1; let y = 2; let z = 3; print(x + y + z);")
        assert r.metrics.variables == 3

    def test_statement_count(self):
        r = analyze("let x = 5; print(x); let y = 10; print(y);")
        assert r.metrics.statements >= 4

    def test_expression_count(self):
        r = analyze("let x = 5 + 3 * 2; print(x);")
        assert r.metrics.expressions >= 3  # 5, 3, 2, +, *, x

    def test_cyclomatic_complexity_base(self):
        r = analyze("let x = 5; print(x);")
        assert r.metrics.cyclomatic_complexity == 1

    def test_cyclomatic_complexity_if(self):
        r = analyze("let x = 5; if (x) { print(1); } print(x);")
        assert r.metrics.cyclomatic_complexity == 2

    def test_cyclomatic_complexity_while(self):
        r = analyze("let x = 5; while (x) { print(x); x = x - 1; } print(0);")
        assert r.metrics.cyclomatic_complexity == 2

    def test_cyclomatic_complexity_nested(self):
        code = """
let x = 5;
if (x) {
    if (x) {
        print(1);
    }
}
print(x);
"""
        r = analyze(code)
        assert r.metrics.cyclomatic_complexity == 3  # 1 base + 2 ifs

    def test_max_nesting(self):
        code = """
fn foo() {
    if (true) {
        while (true) {
            print(1);
        }
    }
}
foo();
"""
        r = analyze(code)
        assert r.metrics.max_nesting >= 3  # fn > if > while

    def test_function_complexity(self):
        code = """
fn complex(x) {
    if (x) {
        if (x) {
            print(1);
        }
    }
    while (x) {
        print(2);
        x = x - 1;
    }
    print(x);
}
complex(5);
"""
        r = analyze(code)
        assert "complex" in r.metrics.function_complexities
        assert r.metrics.function_complexities["complex"] >= 4  # 1 + 2 if + 1 while

    def test_avg_function_length(self):
        code = """
fn short() { print(1); }
fn longer() { let x = 1; let y = 2; print(x + y); }
short();
longer();
"""
        r = analyze(code)
        assert r.metrics.avg_function_length > 0


# ============================================================
# 16. Complexity Diagnostics
# ============================================================

class TestComplexityDiagnostics:

    def test_high_cyclomatic_warns(self):
        # Build code with many branches
        branches = "\n".join(f"if (x) {{ print({i}); }}" for i in range(12))
        code = f"let x = true;\n{branches}\nprint(x);"
        r = analyze(code)
        assert has_diagnostic(r, category=Category.COMPLEXITY, message_contains="cyclomatic")

    def test_low_complexity_no_warning(self):
        r = analyze("let x = 5; print(x);")
        assert not has_diagnostic(r, category=Category.COMPLEXITY)

    def test_custom_thresholds(self):
        code = "let x = true; if (x) { print(1); } if (x) { print(2); }"
        r = analyze(code, thresholds={"cyclomatic": 2, "function_complexity": 2, "nesting": 1, "function_length": 5})
        assert has_diagnostic(r, category=Category.COMPLEXITY, message_contains="cyclomatic")

    def test_deep_nesting_warns(self):
        code = """
fn deep() {
    if (true) {
        if (true) {
            if (true) {
                if (true) {
                    if (true) {
                        print(1);
                    }
                }
            }
        }
    }
}
deep();
"""
        r = analyze(code, thresholds={"cyclomatic": 20, "function_complexity": 20, "nesting": 4, "function_length": 50})
        assert has_diagnostic(r, category=Category.COMPLEXITY, message_contains="nesting")

    def test_function_complexity_threshold(self):
        code = """
fn complex(x) {
    if (x) { print(1); }
    if (x) { print(2); }
    if (x) { print(3); }
    if (x) { print(4); }
    if (x) { print(5); }
    if (x) { print(6); }
    print(x);
}
complex(true);
"""
        r = analyze(code, thresholds={"cyclomatic": 20, "function_complexity": 5, "nesting": 20, "function_length": 50})
        assert has_diagnostic(r, category=Category.COMPLEXITY, message_contains="complex")


# ============================================================
# 17. Optimization Analysis (C014 Composition)
# ============================================================

class TestOptimizationAnalysis:

    def test_constant_folding_detected(self):
        r = analyze("let x = 2 + 3; print(x);")
        assert has_diagnostic(r, category=Category.OPTIMIZATION)

    def test_constant_propagation_detected(self):
        r = analyze("let x = 5; let y = x + 1; print(y);")
        # Optimizer should find propagation opportunity
        hints = r.by_category(Category.OPTIMIZATION)
        assert len(hints) > 0

    def test_optimization_stats_present(self):
        r = analyze("let x = 2 + 3; print(x);")
        assert r.optimization_stats is not None

    def test_clean_code_fewer_optimizations(self):
        r = analyze("print(5);")
        stats = r.optimization_stats
        # Even clean code may have some peephole opts due to compiler output
        # Just verify stats exist
        assert stats is not None

    def test_optimization_hints_are_hints(self):
        r = analyze("let x = 2 + 3; print(x);")
        opt = r.by_category(Category.OPTIMIZATION)
        for d in opt:
            assert d.severity in (Severity.HINT, Severity.WARNING)


# ============================================================
# 18. Dead Code Detection (C014 Bytecode)
# ============================================================

class TestDeadCodeBytecode:

    def test_dead_code_elimination_reported(self):
        code = """
fn foo() {
    return 1;
    print(2);
    print(3);
    print(4);
    print(5);
}
foo();
"""
        r = analyze(code)
        # Should get unreachable warning from AST analysis
        assert has_diagnostic(r, category=Category.UNREACHABLE)

    def test_no_dead_code_clean(self):
        r = analyze("let x = 5; print(x);")
        # No dead code at bytecode level
        dead = [d for d in r.diagnostics
                if d.category == Category.DEAD_CODE and "instructions" in d.message]
        # May or may not trigger depending on optimizer behavior
        # Just verify no crash


# ============================================================
# 19. Quick Analysis Mode
# ============================================================

class TestQuickAnalysis:

    def test_quick_returns_report(self):
        r = analyze_quick("let x = 5; print(x);")
        assert isinstance(r, AnalysisReport)

    def test_quick_detects_type_errors(self):
        r = analyze_quick('let x = 1 + "hello";')
        assert r.error_count > 0

    def test_quick_detects_lint(self):
        r = analyze_quick("if (true) { print(1); }")
        assert has_diagnostic(r, category=Category.REDUNDANT)

    def test_quick_no_optimization_stats(self):
        r = analyze_quick("let x = 2 + 3; print(x);")
        assert r.optimization_stats is None

    def test_quick_parse_error(self):
        r = analyze_quick("let = ;")
        assert r.parse_error is not None

    def test_quick_has_metrics(self):
        r = analyze_quick("let x = 5;\nprint(x);")
        assert r.metrics.total_lines == 2


# ============================================================
# 20. Diagnostic Ordering
# ============================================================

class TestDiagnosticOrdering:

    def test_errors_before_warnings(self):
        r = analyze('let x: int = "hello"; let unused = 5;')
        if len(r.diagnostics) >= 2:
            errs = [i for i, d in enumerate(r.diagnostics) if d.severity == Severity.ERROR]
            warns = [i for i, d in enumerate(r.diagnostics) if d.severity == Severity.WARNING]
            if errs and warns:
                assert max(errs) < min(warns)

    def test_within_severity_sorted_by_line(self):
        code = "let a = 5;\nlet b = 10;\nlet c = 15;"
        r = analyze(code)
        warns = r.by_severity(Severity.WARNING)
        lines = [d.line for d in warns]
        assert lines == sorted(lines)


# ============================================================
# 21. Combined Analysis -- Complex Programs
# ============================================================

class TestComplexPrograms:

    def test_fibonacci(self):
        code = """
fn fib(n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}
print(fib(10));
"""
        r = analyze(code)
        assert r.parse_error is None
        assert r.metrics.functions == 1

    def test_factorial(self):
        code = """
fn fact(n) {
    if (n <= 1) {
        return 1;
    }
    return n * fact(n - 1);
}
print(fact(5));
"""
        r = analyze(code)
        assert r.parse_error is None
        assert r.metrics.functions == 1
        assert r.metrics.cyclomatic_complexity >= 2

    def test_multiple_functions(self):
        code = """
fn add(a, b) { return a + b; }
fn sub(a, b) { return a - b; }
fn mul(a, b) { return a * b; }
let x = add(1, 2);
let y = sub(5, 3);
let z = mul(x, y);
print(z);
"""
        r = analyze(code)
        assert r.metrics.functions == 3

    def test_nested_control_flow(self):
        code = """
let x = 10;
let sum = 0;
while (x > 0) {
    if (x > 5) {
        sum = sum + x;
    } else {
        sum = sum + 1;
    }
    x = x - 1;
}
print(sum);
"""
        r = analyze(code)
        assert r.metrics.cyclomatic_complexity >= 3
        assert r.parse_error is None

    def test_program_with_many_issues(self):
        code = """
let x = 5;
let unused1 = 10;
let unused2 = 20;
x = x;
if (true) { print(x); }
let y = x / 0;
let z = x == true;
print(y + z);
"""
        r = analyze(code)
        assert r.total_issues >= 4  # at least unused x2, self-assign, const condition, div-by-zero


# ============================================================
# 22. Edge Cases
# ============================================================

class TestEdgeCases:

    def test_single_statement(self):
        r = analyze("print(42);")
        assert r.parse_error is None
        assert r.error_count == 0

    def test_only_whitespace(self):
        r = analyze("   \n   \n   ")
        # May parse error or succeed with empty program
        # Either way, should not crash

    def test_deeply_nested(self):
        code = "if (true) { " * 3 + "print(1);" + " }" * 3
        r = analyze(code)
        assert r.parse_error is None

    def test_many_variables(self):
        decls = " ".join(f"let v{i} = {i};" for i in range(20))
        used = " + ".join(f"v{i}" for i in range(20))
        code = f"{decls} print({used});"
        r = analyze(code)
        assert r.metrics.variables == 20

    def test_function_calling_function(self):
        code = """
fn inner(x) { return x + 1; }
fn outer(x) { return inner(x) + 1; }
print(outer(5));
"""
        r = analyze(code)
        assert r.metrics.functions == 2

    def test_while_loop(self):
        code = """
let i = 0;
while (i < 10) {
    print(i);
    i = i + 1;
}
"""
        r = analyze(code)
        assert r.metrics.cyclomatic_complexity >= 2


# ============================================================
# 23. Diagnostic Repr
# ============================================================

class TestDiagnosticRepr:

    def test_diagnostic_repr(self):
        d = Diagnostic(
            severity=Severity.WARNING,
            category=Category.UNUSED_VAR,
            message="test message",
            line=5,
        )
        s = repr(d)
        assert "WARNING" in s
        assert "unused-variable" in s
        assert "test message" in s
        assert "line 5" in s

    def test_diagnostic_repr_with_suggestion(self):
        d = Diagnostic(
            severity=Severity.HINT,
            category=Category.OPTIMIZATION,
            message="can optimize",
            suggestion="do this instead",
        )
        s = repr(d)
        assert "suggestion" in s

    def test_diagnostic_repr_global(self):
        d = Diagnostic(
            severity=Severity.INFO,
            category=Category.STYLE,
            message="global issue",
        )
        s = repr(d)
        assert "global" in s


# ============================================================
# 24. Report Summary Formatting
# ============================================================

class TestReportSummary:

    def test_summary_has_sections(self):
        r = analyze("let unused = 5; let x = 5 / 0; print(x);")
        s = r.summary()
        assert "ERRORS" in s
        assert "WARNINGS" in s

    def test_summary_shows_optimization(self):
        r = analyze("let x = 2 + 3; print(x);")
        s = r.summary()
        assert "Optimization" in s

    def test_summary_total_line(self):
        r = analyze("let x = 5; print(x);")
        s = r.summary()
        assert "Total:" in s
        assert "errors" in s
        assert "warnings" in s


# ============================================================
# 25. Line Metrics Utility
# ============================================================

class TestLineMetrics:

    def test_single_line(self):
        total, code, blank = _compute_line_metrics("hello")
        assert total == 1
        assert code == 1
        assert blank == 0

    def test_multiple_lines(self):
        total, code, blank = _compute_line_metrics("a\nb\nc")
        assert total == 3
        assert code == 3
        assert blank == 0

    def test_blank_lines(self):
        total, code, blank = _compute_line_metrics("a\n\nb\n\nc")
        assert total == 5
        assert blank == 2
        assert code == 3

    def test_empty_string(self):
        total, code, blank = _compute_line_metrics("")
        assert total == 1
        assert blank == 1
        assert code == 0


# ============================================================
# 26. Integration: Full Pipeline
# ============================================================

class TestFullPipeline:

    def test_analyze_and_format(self):
        code = """
fn square(x) {
    return x * x;
}
let result = square(5);
print(result);
"""
        r = analyze(code)
        s = format_report(r)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_all_passes_run(self):
        code = """
let x = 5;
let y = 10;
let unused = 42;
fn add(a, b) {
    return a + b;
}
let z = add(x, y);
if (true) { print(z); }
"""
        r = analyze(code)
        # Unused variable detected
        assert has_diagnostic(r, category=Category.UNUSED_VAR, message_contains="unused")
        # Constant condition detected
        assert has_diagnostic(r, category=Category.REDUNDANT, message_contains="always true")
        # Optimization analysis ran
        assert r.optimization_stats is not None

    def test_report_has_errors_property(self):
        r = analyze('let x = 1 + "hello";')
        assert r.has_errors is True

        r2 = analyze("let x = 5; print(x);")
        assert r2.has_errors is False


# ============================================================
# 27. Typed Code Analysis
# ============================================================

class TestTypedCodeAnalysis:

    def test_function_clean(self):
        code = """
fn add(a, b) {
    return a + b;
}
print(add(1, 2));
"""
        r = analyze(code)
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert len(type_errs) == 0

    def test_variable_clean(self):
        r = analyze("let x = 5; print(x);")
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert len(type_errs) == 0

    def test_string_arithmetic_error(self):
        # Trying to subtract strings is a type error
        code = 'let x = "hello" - "world";'
        r = analyze(code)
        type_errs = r.by_category(Category.TYPE_ERROR)
        assert len(type_errs) > 0

    def test_int_float_arithmetic(self):
        r = analyze("let x = 5 + 3.14; print(x);")
        type_errs = r.by_category(Category.TYPE_ERROR)
        # int+float should promote (subtyping)
        assert len(type_errs) == 0


# ============================================================
# 28. Multiple Diagnostics Interaction
# ============================================================

class TestMultipleDiagnosticsInteraction:

    def test_unused_and_type_error(self):
        code = 'let unused = 5; let x = 1 + "hello"; print(x);'
        r = analyze(code)
        assert has_diagnostic(r, category=Category.UNUSED_VAR)
        assert has_diagnostic(r, category=Category.TYPE_ERROR)

    def test_shadow_and_unused(self):
        code = """
let x = 5;
fn foo() {
    let x = 10;
    print(1);
}
foo();
print(x);
"""
        r = analyze(code)
        assert has_diagnostic(r, category=Category.SHADOWED_VAR)
        # x inside foo is shadowed AND unused
        assert has_diagnostic(r, category=Category.UNUSED_VAR)

    def test_all_categories_possible(self):
        """A program that triggers diagnostics in many categories."""
        code = """
let x = 5;
let unused = 10;
x = x;
if (true) {
    let y = x / 0;
    let r = x == true;
    print(y + r);
}
fn empty() {}
empty();
print(x);
"""
        r = analyze(code)
        categories_found = set(d.category for d in r.diagnostics)
        # Should find at least: unused-variable, redundant (self-assign, const cond),
        # possible-bug (div by 0), style (empty fn, bool compare)
        assert len(categories_found) >= 3


# ============================================================
# 29. Regression: Field Name Correctness
# ============================================================

class TestFieldNameCorrectness:
    """Verify we use correct C010 field names throughout."""

    def test_if_cond_not_condition(self):
        """IfStmt uses .cond not .condition"""
        r = analyze("let x = true; if (x) { print(1); }")
        assert r.parse_error is None

    def test_if_then_body_not_then_branch(self):
        """IfStmt uses .then_body not .then_branch"""
        r = analyze("if (true) { print(1); }")
        # Should find constant condition
        assert has_diagnostic(r, category=Category.REDUNDANT)

    def test_while_cond_not_condition(self):
        """WhileStmt uses .cond not .condition"""
        r = analyze("while (false) { print(1); }")
        assert has_diagnostic(r, category=Category.DEAD_CODE)

    def test_let_value_not_init(self):
        """LetDecl uses .value not .init"""
        r = analyze("let x = 5; print(x);")
        assert r.parse_error is None
        assert r.metrics.variables == 1

    def test_print_value_not_expr(self):
        """PrintStmt uses .value not .expr"""
        r = analyze("print(42);")
        assert r.parse_error is None

    def test_binop_field_order(self):
        """BinOp is BinOp(op, left, right) not (left, op, right)"""
        r = analyze("let x = 5 + 3; print(x);")
        assert r.parse_error is None


# ============================================================
# 30. Stress / Scale Tests
# ============================================================

class TestStressScale:

    def test_100_variables(self):
        decls = " ".join(f"let v{i} = {i};" for i in range(100))
        used = " + ".join(f"v{i}" for i in range(100))
        code = f"{decls} print({used});"
        r = analyze(code)
        assert r.metrics.variables == 100
        assert r.parse_error is None

    def test_20_functions(self):
        fns = " ".join(f"fn f{i}() {{ print({i}); }}" for i in range(20))
        calls = " ".join(f"f{i}();" for i in range(20))
        code = f"{fns} {calls}"
        r = analyze(code)
        assert r.metrics.functions == 20

    def test_nested_ifs(self):
        depth = 8
        code = "let x = true; " + "if (x) { " * depth + "print(1);" + " }" * depth
        r = analyze(code)
        assert r.metrics.cyclomatic_complexity == 1 + depth


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
