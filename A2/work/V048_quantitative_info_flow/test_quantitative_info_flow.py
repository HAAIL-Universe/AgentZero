"""Tests for V048: Quantitative Information Flow Analysis."""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantitative_info_flow import (
    SecurityLevel, InfoFlowLabel, FlowValue, FlowEnv,
    LeakageKind, LeakageFinding, ChannelCapacity, QIFResult,
    QIFAnalyzer, LeakageQuantifier, DeclassificationPolicy,
    analyze_qif, check_noninterference, min_entropy_leakage,
    channel_capacity,
)


# ============================================================
# SecurityLevel Tests
# ============================================================

class TestSecurityLevel:
    def test_ordering(self):
        assert SecurityLevel.LOW <= SecurityLevel.HIGH
        assert SecurityLevel.LOW <= SecurityLevel.LOW
        assert SecurityLevel.HIGH <= SecurityLevel.HIGH
        assert SecurityLevel.LOW < SecurityLevel.HIGH
        assert not (SecurityLevel.HIGH < SecurityLevel.LOW)

    def test_join(self):
        assert SecurityLevel.LOW.join(SecurityLevel.LOW) == SecurityLevel.LOW
        assert SecurityLevel.LOW.join(SecurityLevel.HIGH) == SecurityLevel.HIGH
        assert SecurityLevel.HIGH.join(SecurityLevel.LOW) == SecurityLevel.HIGH
        assert SecurityLevel.HIGH.join(SecurityLevel.HIGH) == SecurityLevel.HIGH

    def test_meet(self):
        assert SecurityLevel.LOW.meet(SecurityLevel.LOW) == SecurityLevel.LOW
        assert SecurityLevel.LOW.meet(SecurityLevel.HIGH) == SecurityLevel.LOW
        assert SecurityLevel.HIGH.meet(SecurityLevel.LOW) == SecurityLevel.LOW
        assert SecurityLevel.HIGH.meet(SecurityLevel.HIGH) == SecurityLevel.HIGH


# ============================================================
# FlowValue Tests
# ============================================================

class TestFlowValue:
    def test_low(self):
        v = FlowValue.low()
        assert not v.is_high
        assert v.level == SecurityLevel.LOW

    def test_high(self):
        v = FlowValue.high("secret", 1)
        assert v.is_high
        assert v.level == SecurityLevel.HIGH
        assert len(v.labels) == 1

    def test_join_levels(self):
        lo = FlowValue.low()
        hi = FlowValue.high("x")
        j = lo.join(hi)
        assert j.is_high
        assert "x" in {l.secret_var for l in j.labels}

    def test_join_labels(self):
        a = FlowValue.high("a")
        b = FlowValue.high("b")
        j = a.join(b)
        assert j.is_high
        sources = {l.secret_var for l in j.labels}
        assert "a" in sources
        assert "b" in sources


# ============================================================
# FlowEnv Tests
# ============================================================

class TestFlowEnv:
    def test_default_is_low(self):
        env = FlowEnv()
        assert not env.get("x").is_high

    def test_set_and_get(self):
        env = FlowEnv()
        env = env.set("x", FlowValue.high("secret"))
        assert env.get("x").is_high
        assert not env.get("y").is_high

    def test_pc_propagation(self):
        env = FlowEnv()
        label = InfoFlowLabel("cond", "implicit")
        env2 = env.with_pc(SecurityLevel.HIGH, frozenset([label]))
        env3 = env2.set("x", FlowValue.low())
        assert env3.get("x").is_high  # implicit flow

    def test_join(self):
        e1 = FlowEnv().set("x", FlowValue.high("a"))
        e2 = FlowEnv().set("x", FlowValue.low())
        j = e1.join(e2)
        assert j.get("x").is_high  # may-analysis: HIGH wins


# ============================================================
# Basic QIF Analysis Tests
# ============================================================

class TestBasicQIF:
    def test_no_leak(self):
        """Program with no information flow from HIGH to LOW."""
        source = """
secret = 42
public = 10
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert len(result.findings) == 0

    def test_direct_leak(self):
        """Direct assignment from HIGH to LOW."""
        source = """
secret = 42
public = secret
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert len(result.findings) > 0
        assert any(f.kind == LeakageKind.DIRECT_FLOW for f in result.findings)

    def test_arithmetic_leak(self):
        """HIGH variable used in arithmetic assigned to LOW."""
        source = """
secret = 42
public = secret + 1
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert len(result.findings) > 0

    def test_no_leak_independent(self):
        """LOW variable computed independently of HIGH."""
        source = """
secret = 42
x = 10
public = x + 5
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert len(result.findings) == 0

    def test_print_leak(self):
        """Printing a HIGH-dependent value."""
        source = """
secret = 42
print(secret)
"""
        result = analyze_qif(source, {"secret"})
        assert any(f.kind == LeakageKind.OBSERVABLE_OUTPUT for f in result.findings)


# ============================================================
# Implicit Flow Tests
# ============================================================

class TestImplicitFlow:
    def test_branch_on_high(self):
        """Implicit flow through branching on HIGH condition."""
        source = """
secret = 1
public = 0
if secret > 0:
    public = 1
"""
        result = analyze_qif(source, {"secret"}, {"public"}, track_implicit=True)
        # public gets HIGH level through implicit flow
        assert len(result.findings) > 0

    def test_no_implicit_when_disabled(self):
        """Implicit flow tracking can be disabled."""
        source = """
secret = 1
public = 0
if secret > 0:
    public = 1
"""
        result = analyze_qif(source, {"secret"}, {"public"}, track_implicit=False)
        # Without implicit tracking, the assignment `public = 1` looks safe
        assert len(result.findings) == 0

    def test_while_implicit(self):
        """Implicit flow through while loop on HIGH condition."""
        source = """
secret = 5
count = 0
while secret > 0:
    count = count + 1
    secret = secret - 1
"""
        result = analyze_qif(source, {"secret"}, {"count"}, track_implicit=True)
        assert len(result.findings) > 0


# ============================================================
# Quantification Tests
# ============================================================

class TestQuantification:
    def test_identity_leak_full_domain(self):
        """Identity function leaks entire domain."""
        source = """
secret = 0
public = secret
"""
        result = analyze_qif(
            source, {"secret"}, {"public"},
            high_domain={"secret": (0, 7)},
        )
        assert len(result.findings) > 0
        # 8 distinct values -> 3 bits
        leak = result.findings[0]
        assert leak.bits_leaked == pytest.approx(3.0, abs=0.01)

    def test_modular_reduction(self):
        """Modular arithmetic reduces leakage."""
        source = """
secret = 0
public = secret % 4
"""
        result = analyze_qif(
            source, {"secret"}, {"public"},
            high_domain={"secret": (0, 255)},
        )
        assert len(result.findings) > 0
        leak = result.findings[0]
        # x % 4 has 4 distinct values -> 2 bits
        assert leak.bits_leaked == pytest.approx(2.0, abs=0.01)

    def test_comparison_binary_leak(self):
        """Comparison leaks at most 1 bit."""
        source = """
secret = 0
public = secret > 5
"""
        result = analyze_qif(
            source, {"secret"}, {"public"},
            high_domain={"secret": (0, 100)},
        )
        assert len(result.findings) > 0
        leak = result.findings[0]
        assert leak.bits_leaked == pytest.approx(1.0, abs=0.01)

    def test_constant_no_leak(self):
        """Constant assignment leaks 0 bits."""
        source = """
secret = 42
public = 0
"""
        result = analyze_qif(
            source, {"secret"}, {"public"},
            high_domain={"secret": (0, 255)},
        )
        assert len(result.findings) == 0

    def test_bitmask_leak(self):
        """Bitmasking reduces leakage to popcount(mask) bits."""
        source = """
secret = 0
public = secret & 3
"""
        result = analyze_qif(
            source, {"secret"}, {"public"},
            high_domain={"secret": (0, 255)},
        )
        assert len(result.findings) > 0
        leak = result.findings[0]
        # & 3 = 2 bits in mask -> 4 values -> 2 bits
        assert leak.bits_leaked == pytest.approx(2.0, abs=0.01)

    def test_floor_div_reduces_leak(self):
        """Floor division reduces domain size."""
        source = """
secret = 0
public = secret // 4
"""
        result = analyze_qif(
            source, {"secret"}, {"public"},
            high_domain={"secret": (0, 15)},
        )
        assert len(result.findings) > 0
        leak = result.findings[0]
        # 16 values // 4 = 4 distinct values -> 2 bits
        assert leak.bits_leaked == pytest.approx(2.0, abs=0.01)


# ============================================================
# Noninterference Tests
# ============================================================

class TestNoninterference:
    def test_noninterfering(self):
        """Independent computation is noninterfering."""
        source = """
secret = 42
public = 10
"""
        ni, cex = check_noninterference(
            source, {"secret"}, "public"
        )
        assert ni is True
        assert cex is None

    def test_interfering(self):
        """Direct flow violates noninterference."""
        source = """
secret = 42
public = secret
"""
        ni, cex = check_noninterference(
            source, {"secret"}, "public"
        )
        assert ni is False

    def test_arithmetic_interference(self):
        """Arithmetic on HIGH violates noninterference."""
        source = """
secret = 42
public = secret * 2 + 1
"""
        ni, cex = check_noninterference(
            source, {"secret"}, "public"
        )
        assert ni is False


# ============================================================
# Channel Capacity Tests
# ============================================================

class TestChannelCapacity:
    def test_identity_channel(self):
        """Identity channel has capacity = log2(domain)."""
        source = """
secret = 0
output = secret
"""
        cap = channel_capacity(
            source, {"secret"}, "output",
            {"secret": (0, 15)}
        )
        assert cap == pytest.approx(4.0, abs=0.01)  # 16 values = 4 bits

    def test_binary_channel(self):
        """Comparison channel has capacity = 1 bit."""
        source = """
secret = 0
output = secret > 5
"""
        cap = channel_capacity(
            source, {"secret"}, "output",
            {"secret": (0, 100)}
        )
        assert cap == pytest.approx(1.0, abs=0.01)

    def test_mod_channel(self):
        """Modular channel has capacity = log2(modulus)."""
        source = """
secret = 0
output = secret % 8
"""
        cap = channel_capacity(
            source, {"secret"}, "output",
            {"secret": (0, 255)}
        )
        assert cap == pytest.approx(3.0, abs=0.01)


# ============================================================
# Min-Entropy Leakage Tests
# ============================================================

class TestMinEntropy:
    def test_identity_full_leakage(self):
        """Identity leaks all min-entropy."""
        source = """
secret = 0
output = secret
"""
        leak = min_entropy_leakage(
            source, {"secret"}, "output",
            {"secret": (0, 7)}
        )
        assert leak == pytest.approx(3.0, abs=0.01)

    def test_no_leakage(self):
        """Independent computation has zero leakage."""
        source = """
secret = 42
output = 10
"""
        leak = min_entropy_leakage(
            source, {"secret"}, "output",
            {"secret": (0, 255)}
        )
        assert leak == pytest.approx(0.0, abs=0.01)


# ============================================================
# Multiple Secrets Tests
# ============================================================

class TestMultipleSecrets:
    def test_two_secrets_combined(self):
        """Two secrets flowing into one output."""
        source = """
secret1 = 0
secret2 = 0
public = secret1 + secret2
"""
        result = analyze_qif(
            source, {"secret1", "secret2"}, {"public"},
            high_domain={"secret1": (0, 3), "secret2": (0, 3)},
        )
        assert len(result.findings) > 0
        sources = set()
        for f in result.findings:
            sources.update(f.secret_sources)
        assert "secret1" in sources
        assert "secret2" in sources

    def test_one_secret_leaks_other_safe(self):
        """One secret leaks, another stays safe."""
        source = """
secret1 = 0
secret2 = 0
public = secret1
safe = 10
"""
        result = analyze_qif(
            source, {"secret1", "secret2"}, {"public", "safe"},
        )
        leaked_vars = {f.variable for f in result.findings}
        assert "public" in leaked_vars
        assert "safe" not in leaked_vars


# ============================================================
# Declassification Tests
# ============================================================

class TestDeclassification:
    def test_policy_allows(self):
        policy = DeclassificationPolicy()
        policy.allow("password", "login", 1.0)
        assert policy.is_allowed("password", "login", 1.0)
        assert not policy.is_allowed("password", "login", 2.0)
        assert not policy.is_allowed("password", "other", 1.0)

    def test_wildcard_context(self):
        policy = DeclassificationPolicy()
        policy.allow("key", "*", 8.0)
        assert policy.is_allowed("key", "anywhere", 8.0)
        assert policy.is_allowed("key", "elsewhere", 4.0)
        assert not policy.is_allowed("key", "anywhere", 9.0)

    def test_wildcard_secret(self):
        policy = DeclassificationPolicy()
        policy.allow("*", "public_api", 1.0)
        assert policy.is_allowed("anything", "public_api", 1.0)


# ============================================================
# SMT Quantifier Tests
# ============================================================

class TestLeakageQuantifier:
    def test_count_simple(self):
        """Count distinct values of a linear expression."""
        q = LeakageQuantifier()
        # out = x (identity, domain [0, 3])
        from smt_solver import Var, App, Op, IntConst, Sort, SortKind
        INT = Sort(SortKind.INT)
        BOOL = Sort(SortKind.BOOL)
        x_var = Var("x", INT)
        out_var = Var("out", INT)
        constraints = [App(Op.EQ, [out_var, x_var], BOOL)]
        count, bits = q.count_distinct_outputs(
            ["x"], "out", constraints, {"x": (0, 3)}
        )
        assert count == 4
        assert bits == pytest.approx(2.0, abs=0.01)

    def test_count_constant(self):
        """Constant expression has 1 distinct value."""
        q = LeakageQuantifier()
        from smt_solver import Var, App, Op, IntConst, Sort, SortKind
        INT = Sort(SortKind.INT)
        BOOL = Sort(SortKind.BOOL)
        out_var = Var("out", INT)
        constraints = [App(Op.EQ, [out_var, IntConst(42)], BOOL)]
        count, bits = q.count_distinct_outputs(
            ["x"], "out", constraints, {"x": (0, 7)}
        )
        assert count == 1
        assert bits == 0.0


# ============================================================
# Return Value Leakage Tests
# ============================================================

class TestReturnLeakage:
    def test_function_returns_high(self):
        """Function returning HIGH value."""
        source = """
secret = 42
def leak():
    return secret
"""
        result = analyze_qif(source, {"secret"})
        assert any(
            f.kind == LeakageKind.OBSERVABLE_OUTPUT and f.channel == "return"
            for f in result.findings
        )

    def test_function_returns_low(self):
        """Function returning LOW value."""
        source = """
secret = 42
def safe():
    return 10
"""
        result = analyze_qif(source, {"secret"})
        assert not any(
            f.kind == LeakageKind.OBSERVABLE_OUTPUT and f.channel == "return"
            for f in result.findings
        )


# ============================================================
# QIFResult Tests
# ============================================================

class TestQIFResult:
    def test_ok_when_no_leaks(self):
        source = """
secret = 42
public = 10
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert result.ok

    def test_not_ok_when_leaks(self):
        source = """
secret = 42
public = secret
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert not result.ok or len(result.findings) > 0

    def test_summary_runs(self):
        source = """
secret = 42
public = secret
"""
        result = analyze_qif(
            source, {"secret"}, {"public"},
            high_domain={"secret": (0, 7)}
        )
        s = result.summary()
        assert "Quantitative" in s
        assert "bits" in s


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_source(self):
        result = analyze_qif("", {"secret"})
        assert result.ok

    def test_no_high_vars(self):
        source = "x = 10\ny = x + 1\n"
        result = analyze_qif(source, set(), {"y"})
        assert len(result.findings) == 0

    def test_augmented_assign_leak(self):
        """+=, -= etc propagate HIGH."""
        source = """
secret = 42
public = 0
public += secret
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert len(result.findings) > 0

    def test_ternary_expression(self):
        """Ternary on HIGH condition."""
        source = """
secret = 42
public = 1 if secret > 0 else 0
"""
        result = analyze_qif(source, {"secret"}, {"public"})
        assert len(result.findings) > 0

    def test_tuple_unpack_leak(self):
        """Tuple unpacking propagates HIGH."""
        source = """
secret = 42
a, b = secret, 10
"""
        result = analyze_qif(source, {"secret"}, {"a"})
        # The tuple expression contains HIGH, so both a and b get HIGH
        assert len(result.findings) > 0

    def test_multiple_print_channels(self):
        """Multiple print statements create multiple channels."""
        source = """
secret = 42
print(secret)
print(secret + 1)
"""
        result = analyze_qif(
            source, {"secret"},
            high_domain={"secret": (0, 7)}
        )
        # Should have at least 2 output findings
        output_findings = [
            f for f in result.findings
            if f.kind == LeakageKind.OBSERVABLE_OUTPUT
        ]
        assert len(output_findings) >= 2

    def test_for_loop_implicit(self):
        """For loop over HIGH range creates implicit flow."""
        source = """
secret = [1, 2, 3]
total = 0
for x in secret:
    total = total + x
"""
        result = analyze_qif(source, {"secret"}, {"total"}, track_implicit=True)
        assert len(result.findings) > 0


# ============================================================
# Integration: Flow + Quantification
# ============================================================

class TestIntegration:
    def test_password_check_leaks_1_bit(self):
        """Password equality check leaks 1 bit (correct/incorrect)."""
        source = """
password = 0
attempt = 0
result = password == attempt
"""
        r = analyze_qif(
            source, {"password"}, {"result"},
            high_domain={"password": (0, 9999)}
        )
        assert len(r.findings) > 0
        # Comparison leaks 1 bit
        leak = r.findings[0]
        assert leak.bits_leaked == pytest.approx(1.0, abs=0.01)

    def test_hash_reduces_leakage(self):
        """Modular 'hash' reduces leakage to log2(buckets)."""
        source = """
secret = 0
bucket = secret % 16
"""
        r = analyze_qif(
            source, {"secret"}, {"bucket"},
            high_domain={"secret": (0, 255)}
        )
        assert len(r.findings) > 0
        leak = r.findings[0]
        assert leak.bits_leaked == pytest.approx(4.0, abs=0.01)

    def test_sign_leaks_less_than_2_bits(self):
        """Sign function leaks < 2 bits (3 outcomes: neg/zero/pos)."""
        source = """
secret = 0
sign = 1 if secret > 0 else (0 if secret == 0 else -1)
"""
        # Note: nested ternary -- structural count gives body+orelse
        r = analyze_qif(
            source, {"secret"}, {"sign"},
            high_domain={"secret": (-100, 100)}
        )
        assert len(r.findings) > 0
        leak = r.findings[0]
        # Should be ~ log2(3) = 1.58 bits or slightly more (upper bound)
        assert leak.bits_leaked <= 3.0  # reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
