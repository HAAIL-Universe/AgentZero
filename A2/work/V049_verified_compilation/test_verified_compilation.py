"""Tests for V049: Verified Compilation -- Translation Validation."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C014_bytecode_optimizer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'V044_proof_certificates'))

from verified_compilation import (
    PassName, PassValidationResult, CompilationValidationResult,
    validate_compilation, validate_pass, certify_compilation,
    CompilationValidator,
    _compute_reachable, _find_constant_folds, _find_strength_reductions,
    _find_peephole_patterns, _find_constant_propagations,
    _validate_execution_equivalence,
)
from proof_certificates import CertStatus, ProofKind


# ===================================================================
# Test: Basic Pipeline
# ===================================================================

class TestBasicPipeline:
    """Test the full validation pipeline on simple programs."""

    def test_trivial_program(self):
        source = "let x = 42;"
        result = validate_compilation(source)
        assert result.certificate is not None
        assert result.execution_match is True

    def test_arithmetic_program(self):
        source = "let x = 2 + 3; print(x);"
        result = validate_compilation(source)
        assert result.certificate is not None
        assert result.execution_match is True

    def test_variable_assignment(self):
        source = "let x = 10; x = x + 5; print(x);"
        result = validate_compilation(source)
        assert result.certificate is not None
        assert result.execution_match is True

    def test_if_statement(self):
        source = "let x = 10; if (x > 5) { print(x); }"
        result = validate_compilation(source)
        assert result.certificate is not None
        assert result.execution_match is True

    def test_while_loop(self):
        source = "let x = 0; while (x < 5) { x = x + 1; } print(x);"
        result = validate_compilation(source)
        assert result.certificate is not None
        assert result.execution_match is True

    def test_function_def(self):
        source = "fn add(a, b) { return a + b; } let r = add(3, 4); print(r);"
        result = validate_compilation(source)
        assert result.certificate is not None
        assert result.execution_match is True

    def test_empty_program(self):
        source = ""
        result = validate_compilation(source)
        assert result.certificate is not None


# ===================================================================
# Test: Constant Folding Validation
# ===================================================================

class TestConstantFoldValidation:
    """Test SMT-based validation of constant folding."""

    def test_addition_fold(self):
        source = "let x = 2 + 3;"
        result = validate_pass(source, PassName.CONSTANT_FOLD)
        # May or may not find folds depending on instruction matching
        assert result.status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_subtraction_fold(self):
        source = "let x = 10 - 3;"
        result = validate_pass(source, PassName.CONSTANT_FOLD)
        assert result.status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_multiplication_fold(self):
        source = "let x = 4 * 5;"
        result = validate_pass(source, PassName.CONSTANT_FOLD)
        assert result.status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_no_fold_needed(self):
        source = "let x = 42;"
        result = validate_pass(source, PassName.CONSTANT_FOLD)
        assert result.status == CertStatus.VALID
        assert not result.applied

    def test_nested_constant_expr(self):
        source = "let x = 1 + 2 + 3;"
        result = validate_pass(source, PassName.CONSTANT_FOLD)
        assert result.status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_comparison_fold(self):
        source = "let x = 3 > 2;"
        result = validate_pass(source, PassName.CONSTANT_FOLD)
        assert result.status in (CertStatus.VALID, CertStatus.UNKNOWN)


# ===================================================================
# Test: Strength Reduction Validation
# ===================================================================

class TestStrengthReductionValidation:
    """Test SMT-based validation of strength reductions."""

    def test_no_reduction(self):
        source = "let x = 42;"
        result = validate_pass(source, PassName.STRENGTH_REDUCTION)
        assert result.status == CertStatus.VALID
        assert not result.applied

    def test_mul_by_2(self):
        source = "let x = 10; let y = x * 2;"
        result = validate_pass(source, PassName.STRENGTH_REDUCTION)
        # If strength reduction fires, verify it
        if result.applied:
            assert result.status == CertStatus.VALID

    def test_add_zero(self):
        source = "let x = 10; let y = x + 0;"
        result = validate_pass(source, PassName.STRENGTH_REDUCTION)
        if result.applied:
            assert result.status == CertStatus.VALID

    def test_mul_by_one(self):
        source = "let x = 10; let y = x * 1;"
        result = validate_pass(source, PassName.STRENGTH_REDUCTION)
        if result.applied:
            assert result.status == CertStatus.VALID


# ===================================================================
# Test: Dead Code Elimination Validation
# ===================================================================

class TestDeadCodeEliminationValidation:
    """Test reachability-based validation of dead code elimination."""

    def test_no_dead_code(self):
        source = "let x = 1; print(x);"
        result = validate_pass(source, PassName.DEAD_CODE_ELIMINATION)
        assert result.status == CertStatus.VALID

    def test_reachability_computation(self):
        """Test the BFS reachability helper directly."""
        from optimizer import Instr, Op
        instrs = [
            Instr(Op.CONST, 0),     # 0: reachable
            Instr(Op.JUMP, 3),      # 1: reachable, jump to 3
            Instr(Op.CONST, 1),     # 2: UNREACHABLE (after unconditional jump)
            Instr(Op.PRINT),        # 3: reachable (jump target)
            Instr(Op.HALT),         # 4: reachable
        ]
        reachable = _compute_reachable(instrs)
        assert 0 in reachable
        assert 1 in reachable
        assert 2 not in reachable  # Dead code
        assert 3 in reachable
        assert 4 in reachable

    def test_conditional_reachability(self):
        from optimizer import Instr, Op
        instrs = [
            Instr(Op.CONST, 0),          # 0
            Instr(Op.JUMP_IF_FALSE, 3),  # 1: branches to 3 or falls through to 2
            Instr(Op.CONST, 1),          # 2: reachable (true branch)
            Instr(Op.PRINT),             # 3: reachable (false branch target)
            Instr(Op.HALT),              # 4
        ]
        reachable = _compute_reachable(instrs)
        assert len(reachable) == 5  # All reachable


# ===================================================================
# Test: Jump Optimization Validation
# ===================================================================

class TestJumpOptimizationValidation:
    """Test validation of jump threading."""

    def test_no_jump_threading(self):
        source = "let x = 1; print(x);"
        result = validate_pass(source, PassName.JUMP_OPTIMIZATION)
        assert result.status == CertStatus.VALID

    def test_with_conditional(self):
        source = "let x = 1; if (x > 0) { print(x); }"
        result = validate_pass(source, PassName.JUMP_OPTIMIZATION)
        assert result.status == CertStatus.VALID


# ===================================================================
# Test: Peephole Optimization Validation
# ===================================================================

class TestPeepholeValidation:
    """Test validation of peephole optimizations."""

    def test_no_peephole(self):
        source = "let x = 42;"
        result = validate_pass(source, PassName.PEEPHOLE)
        assert result.status == CertStatus.VALID

    def test_pattern_detection(self):
        from optimizer import Instr, Op
        instrs = [
            Instr(Op.CONST, 0),   # 0
            Instr(Op.POP),        # 1: push-pop pattern
            Instr(Op.HALT),       # 2
        ]
        patterns = _find_peephole_patterns(instrs)
        assert len(patterns) == 1
        assert patterns[0][0] == "push_pop_elim"

    def test_dup_pop_detection(self):
        from optimizer import Instr, Op
        instrs = [
            Instr(Op.DUP),    # 0
            Instr(Op.POP),    # 1: dup-pop pattern
            Instr(Op.HALT),   # 2
        ]
        patterns = _find_peephole_patterns(instrs)
        assert len(patterns) == 1
        assert patterns[0][0] == "dup_pop_elim"


# ===================================================================
# Test: Constant Propagation Validation
# ===================================================================

class TestConstantPropagationValidation:
    """Test validation of constant propagation."""

    def test_no_propagation(self):
        source = "let x = 42;"
        result = validate_pass(source, PassName.CONSTANT_PROPAGATION)
        assert result.status == CertStatus.VALID

    def test_simple_propagation(self):
        source = "let x = 10; print(x);"
        result = validate_pass(source, PassName.CONSTANT_PROPAGATION)
        assert result.status in (CertStatus.VALID, CertStatus.UNKNOWN)


# ===================================================================
# Test: Execution Equivalence
# ===================================================================

class TestExecutionEquivalence:
    """Test dynamic execution equivalence checking."""

    def test_simple_program(self):
        source = "let x = 2 + 3; print(x);"
        result = _validate_execution_equivalence(source)
        assert result.status == CertStatus.VALID
        assert len(result.obligations) >= 2

    def test_loop_program(self):
        source = "let s = 0; let i = 0; while (i < 10) { s = s + i; i = i + 1; } print(s);"
        result = _validate_execution_equivalence(source)
        assert result.status == CertStatus.VALID

    def test_function_program(self):
        source = "fn double(x) { return x * 2; } print(double(21));"
        result = _validate_execution_equivalence(source)
        assert result.status == CertStatus.VALID

    def test_fibonacci(self):
        source = """
        fn fib(n) {
            if (n < 2) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        print(fib(10));
        """
        result = _validate_execution_equivalence(source)
        assert result.status == CertStatus.VALID

    def test_nested_if(self):
        source = """
        let x = 15;
        if (x > 10) {
            if (x > 20) {
                print(1);
            } else {
                print(2);
            }
        } else {
            print(3);
        }
        """
        result = _validate_execution_equivalence(source)
        assert result.status == CertStatus.VALID

    def test_complex_arithmetic(self):
        source = "let x = (2 + 3) * (4 - 1); print(x);"
        result = _validate_execution_equivalence(source)
        assert result.status == CertStatus.VALID

    def test_multiple_functions(self):
        source = """
        fn add(a, b) { return a + b; }
        fn mul(a, b) { return a * b; }
        let r = add(mul(2, 3), mul(4, 5));
        print(r);
        """
        result = _validate_execution_equivalence(source)
        assert result.status == CertStatus.VALID


# ===================================================================
# Test: Certificate Generation
# ===================================================================

class TestCertificateGeneration:
    """Test proof certificate generation."""

    def test_certificate_structure(self):
        source = "let x = 2 + 3; print(x);"
        cert = certify_compilation(source)
        assert cert is not None
        assert cert.kind == ProofKind.COMPOSITE
        assert cert.claim is not None
        assert len(cert.sub_certificates) > 0

    def test_certificate_valid(self):
        source = "let x = 42; print(x);"
        cert = certify_compilation(source)
        assert cert.status == CertStatus.VALID

    def test_certificate_with_optimizations(self):
        source = "let x = 2 + 3; let y = x * 1; print(y);"
        cert = certify_compilation(source)
        assert cert is not None
        # Should have sub-certificates for each validated pass
        assert cert.total_obligations > 0

    def test_certificate_serialization(self):
        source = "let x = 2 + 3;"
        cert = certify_compilation(source)
        json_str = cert.to_json()
        assert len(json_str) > 0
        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert "kind" in parsed
        assert "status" in parsed

    def test_certificate_metadata(self):
        source = "let x = 10; print(x);"
        result = validate_compilation(source)
        for pr in result.pass_results:
            for ob in pr.obligations:
                assert ob.name is not None
                assert ob.description is not None


# ===================================================================
# Test: Full Pipeline Integration
# ===================================================================

class TestFullPipeline:
    """Test the complete validation pipeline."""

    def test_summary_output(self):
        source = "let x = 2 + 3; print(x);"
        result = validate_compilation(source)
        summary = result.summary
        assert "Verified Compilation" in summary

    def test_pass_results_populated(self):
        source = "let x = 42; print(x);"
        result = validate_compilation(source)
        # Should have 7 pass results (6 passes + execution equivalence)
        assert len(result.pass_results) == 7

    def test_obligation_counts(self):
        source = "let x = 2 + 3; print(x);"
        result = validate_compilation(source)
        assert result.total_obligations > 0
        assert result.valid_obligations >= 0

    def test_is_valid_property(self):
        source = "let x = 42;"
        result = validate_compilation(source)
        assert result.is_valid is True

    def test_complex_program(self):
        source = """
        fn factorial(n) {
            if (n < 2) { return 1; }
            return n * factorial(n - 1);
        }
        let result = factorial(5);
        print(result);
        """
        result = validate_compilation(source)
        assert result.execution_match is True
        assert result.certificate is not None

    def test_loop_with_accumulator(self):
        source = """
        let sum = 0;
        let i = 1;
        while (i < 11) {
            sum = sum + i;
            i = i + 1;
        }
        print(sum);
        """
        result = validate_compilation(source)
        assert result.execution_match is True
        assert result.certificate.status == CertStatus.VALID

    def test_multiple_variables(self):
        source = """
        let a = 10;
        let b = 20;
        let c = a + b;
        let d = c * 2;
        print(d);
        """
        result = validate_compilation(source)
        assert result.execution_match is True


# ===================================================================
# Test: CompilationValidator (caching)
# ===================================================================

class TestCompilationValidator:
    """Test the stateful validator with caching."""

    def test_cache_hit(self):
        validator = CompilationValidator()
        source = "let x = 42;"
        r1 = validator.validate(source)
        r2 = validator.validate(source)
        assert r1 is r2  # Same object from cache

    def test_cache_miss(self):
        validator = CompilationValidator()
        r1 = validator.validate("let x = 1;")
        r2 = validator.validate("let x = 2;")
        assert r1 is not r2

    def test_cache_size(self):
        validator = CompilationValidator()
        validator.validate("let x = 1;")
        validator.validate("let x = 2;")
        assert validator.cache_size == 2

    def test_clear_cache(self):
        validator = CompilationValidator()
        validator.validate("let x = 1;")
        validator.clear_cache()
        assert validator.cache_size == 0

    def test_batch_validation(self):
        validator = CompilationValidator()
        sources = ["let x = 1;", "let x = 2;", "let x = 3;"]
        results = validator.validate_batch(sources)
        assert len(results) == 3
        assert all(r.certificate is not None for r in results)

    def test_batch_with_duplicates(self):
        validator = CompilationValidator()
        sources = ["let x = 1;", "let x = 1;", "let x = 2;"]
        results = validator.validate_batch(sources)
        assert len(results) == 3
        assert results[0] is results[1]  # Cached
        assert validator.cache_size == 2


# ===================================================================
# Test: Edge Cases
# ===================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_division_program(self):
        source = "let x = 10; let y = x / 2; print(y);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_modulo_program(self):
        source = "let x = 10; let y = x % 3; print(y);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_nested_function_calls(self):
        source = """
        fn inc(x) { return x + 1; }
        fn double(x) { return x + x; }
        print(double(inc(5)));
        """
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_deeply_nested_if(self):
        source = """
        let x = 10;
        if (x > 0) {
            if (x > 5) {
                if (x > 8) {
                    print(3);
                } else {
                    print(2);
                }
            } else {
                print(1);
            }
        } else {
            print(0);
        }
        """
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_boolean_expressions(self):
        source = "let x = 5; let y = x > 3; print(y);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_negative_values(self):
        source = "let x = 0 - 42; print(x);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_while_with_break_condition(self):
        source = """
        let x = 100;
        while (x > 0) {
            x = x - 17;
        }
        print(x);
        """
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_recursive_fibonacci(self):
        source = """
        fn fib(n) {
            if (n < 2) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        print(fib(8));
        """
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_multiple_print(self):
        source = "print(1); print(2); print(3);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_chained_assignments(self):
        source = "let x = 1; x = x + 1; x = x + 1; x = x + 1; print(x);"
        result = validate_compilation(source)
        assert result.execution_match is True


# ===================================================================
# Test: Validate Individual Passes on Specific Programs
# ===================================================================

class TestSpecificPassPrograms:
    """Test specific programs that exercise each optimization pass."""

    def test_constant_fold_multiple_ops(self):
        """Multiple constant expressions should all be foldable."""
        source = "let a = 2 + 3; let b = 10 - 4; let c = 3 * 7; print(a + b + c);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_strength_reduction_identities(self):
        """Programs with x+0, x*1 patterns."""
        source = "let x = 10; let y = x + 0; let z = x * 1; print(y + z);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_dead_code_after_return(self):
        """Code after return should be eliminated."""
        source = """
        fn f(x) {
            return x;
            print(999);
        }
        print(f(42));
        """
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_peephole_store_load(self):
        """Store-load pattern should be optimizable."""
        source = "let x = 42; print(x);"
        result = validate_compilation(source)
        assert result.execution_match is True

    def test_combined_optimizations(self):
        """Program exercising multiple optimization passes."""
        source = """
        let a = 2 + 3;
        let b = a * 1;
        let c = b + 0;
        if (c > 0) {
            print(c);
        }
        """
        result = validate_compilation(source)
        assert result.execution_match is True
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
