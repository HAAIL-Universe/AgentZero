"""
Tests for Package Manager (C021)
Target: 100+ tests covering semver, constraints, registry, resolver, lockfile, install
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from package_manager import (
    SemVer, Constraint, PackageSpec, Registry, RegistryError,
    Resolver, ResolutionError, Lockfile, LockEntry,
    PackageManager, InstalledPackage, create_test_registry,
)


# ============================================================
# SemVer Parsing
# ============================================================

class TestSemVerParsing:
    def test_parse_full(self):
        v = SemVer.parse("1.2.3")
        assert v.major == 1 and v.minor == 2 and v.patch == 3

    def test_parse_major_only(self):
        v = SemVer.parse("5")
        assert v == SemVer(5, 0, 0)

    def test_parse_major_minor(self):
        v = SemVer.parse("3.7")
        assert v == SemVer(3, 7, 0)

    def test_parse_with_v_prefix(self):
        v = SemVer.parse("v2.1.0")
        assert v == SemVer(2, 1, 0)

    def test_parse_prerelease(self):
        v = SemVer.parse("1.0.0-alpha")
        assert v.prerelease == ('alpha',)

    def test_parse_prerelease_numeric(self):
        v = SemVer.parse("1.0.0-alpha.1")
        assert v.prerelease == ('alpha', 1)

    def test_parse_build_metadata(self):
        v = SemVer.parse("1.0.0+build.123")
        assert v.build == ('build', '123')

    def test_parse_prerelease_and_build(self):
        v = SemVer.parse("1.0.0-beta.2+build.456")
        assert v.prerelease == ('beta', 2)
        assert v.build == ('build', '456')

    def test_str_roundtrip(self):
        cases = ["1.2.3", "0.0.0", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0+build"]
        for s in cases:
            assert str(SemVer.parse(s)) == s

    def test_parse_whitespace(self):
        v = SemVer.parse("  1.2.3  ")
        assert v == SemVer(1, 2, 3)


# ============================================================
# SemVer Comparison
# ============================================================

class TestSemVerComparison:
    def test_equal(self):
        assert SemVer(1, 2, 3) == SemVer(1, 2, 3)

    def test_not_equal(self):
        assert SemVer(1, 2, 3) != SemVer(1, 2, 4)

    def test_less_than_patch(self):
        assert SemVer(1, 2, 3) < SemVer(1, 2, 4)

    def test_less_than_minor(self):
        assert SemVer(1, 2, 9) < SemVer(1, 3, 0)

    def test_less_than_major(self):
        assert SemVer(1, 9, 9) < SemVer(2, 0, 0)

    def test_prerelease_less_than_release(self):
        assert SemVer.parse("1.0.0-alpha") < SemVer.parse("1.0.0")

    def test_prerelease_ordering(self):
        assert SemVer.parse("1.0.0-alpha") < SemVer.parse("1.0.0-beta")

    def test_prerelease_numeric_ordering(self):
        assert SemVer.parse("1.0.0-alpha.1") < SemVer.parse("1.0.0-alpha.2")

    def test_build_metadata_ignored_in_comparison(self):
        assert SemVer.parse("1.0.0+build1") == SemVer.parse("1.0.0+build2")

    def test_sorting(self):
        versions = [
            SemVer.parse("2.0.0"),
            SemVer.parse("1.0.0"),
            SemVer.parse("1.1.0"),
            SemVer.parse("1.0.0-alpha"),
        ]
        result = sorted(versions)
        assert result == [
            SemVer.parse("1.0.0-alpha"),
            SemVer.parse("1.0.0"),
            SemVer.parse("1.1.0"),
            SemVer.parse("2.0.0"),
        ]

    def test_bump_major(self):
        assert SemVer(1, 2, 3).bump_major() == SemVer(2, 0, 0)

    def test_bump_minor(self):
        assert SemVer(1, 2, 3).bump_minor() == SemVer(1, 3, 0)

    def test_bump_patch(self):
        assert SemVer(1, 2, 3).bump_patch() == SemVer(1, 2, 4)

    def test_hash_equality(self):
        s = {SemVer(1, 0, 0), SemVer(1, 0, 0)}
        assert len(s) == 1


# ============================================================
# Constraints
# ============================================================

class TestConstraints:
    def test_exact(self):
        c = Constraint("1.2.3")
        assert c.satisfies(SemVer(1, 2, 3))
        assert not c.satisfies(SemVer(1, 2, 4))

    def test_exact_with_equals(self):
        c = Constraint("=1.2.3")
        assert c.satisfies(SemVer(1, 2, 3))
        assert not c.satisfies(SemVer(1, 2, 4))

    def test_gte(self):
        c = Constraint(">=1.0.0")
        assert c.satisfies(SemVer(1, 0, 0))
        assert c.satisfies(SemVer(2, 0, 0))
        assert not c.satisfies(SemVer(0, 9, 0))

    def test_lte(self):
        c = Constraint("<=1.5.0")
        assert c.satisfies(SemVer(1, 5, 0))
        assert c.satisfies(SemVer(1, 0, 0))
        assert not c.satisfies(SemVer(1, 5, 1))

    def test_gt(self):
        c = Constraint(">1.0.0")
        assert not c.satisfies(SemVer(1, 0, 0))
        assert c.satisfies(SemVer(1, 0, 1))

    def test_lt(self):
        c = Constraint("<2.0.0")
        assert c.satisfies(SemVer(1, 9, 9))
        assert not c.satisfies(SemVer(2, 0, 0))

    def test_neq(self):
        c = Constraint("!=1.0.0")
        assert not c.satisfies(SemVer(1, 0, 0))
        assert c.satisfies(SemVer(1, 0, 1))

    def test_caret_major(self):
        c = Constraint("^1.2.3")
        assert c.satisfies(SemVer(1, 2, 3))
        assert c.satisfies(SemVer(1, 9, 9))
        assert not c.satisfies(SemVer(2, 0, 0))
        assert not c.satisfies(SemVer(1, 2, 2))

    def test_caret_zero_major(self):
        c = Constraint("^0.2.3")
        assert c.satisfies(SemVer(0, 2, 3))
        assert c.satisfies(SemVer(0, 2, 9))
        assert not c.satisfies(SemVer(0, 3, 0))

    def test_caret_zero_zero(self):
        c = Constraint("^0.0.3")
        assert c.satisfies(SemVer(0, 0, 3))
        assert not c.satisfies(SemVer(0, 0, 4))

    def test_tilde(self):
        c = Constraint("~1.2.3")
        assert c.satisfies(SemVer(1, 2, 3))
        assert c.satisfies(SemVer(1, 2, 9))
        assert not c.satisfies(SemVer(1, 3, 0))

    def test_tilde_minor(self):
        c = Constraint("~1.2")
        assert c.satisfies(SemVer(1, 2, 0))
        assert c.satisfies(SemVer(1, 2, 5))
        assert not c.satisfies(SemVer(1, 3, 0))

    def test_wildcard_star(self):
        c = Constraint("*")
        assert c.satisfies(SemVer(0, 0, 0))
        assert c.satisfies(SemVer(99, 99, 99))

    def test_wildcard_major(self):
        c = Constraint("1.*")
        assert c.satisfies(SemVer(1, 0, 0))
        assert c.satisfies(SemVer(1, 9, 9))
        assert not c.satisfies(SemVer(2, 0, 0))

    def test_wildcard_minor(self):
        c = Constraint("1.2.*")
        assert c.satisfies(SemVer(1, 2, 0))
        assert c.satisfies(SemVer(1, 2, 99))
        assert not c.satisfies(SemVer(1, 3, 0))

    def test_range_and(self):
        c = Constraint(">=1.0.0 <2.0.0")
        assert c.satisfies(SemVer(1, 0, 0))
        assert c.satisfies(SemVer(1, 9, 9))
        assert not c.satisfies(SemVer(2, 0, 0))
        assert not c.satisfies(SemVer(0, 9, 9))

    def test_or(self):
        c = Constraint("^1.0.0 || ^2.0.0")
        assert c.satisfies(SemVer(1, 5, 0))
        assert c.satisfies(SemVer(2, 5, 0))
        assert not c.satisfies(SemVer(3, 0, 0))

    def test_filter(self):
        versions = [SemVer.parse(v) for v in ["0.9.0", "1.0.0", "1.5.0", "2.0.0"]]
        c = Constraint("^1.0.0")
        result = c.filter(versions)
        assert result == [SemVer(1, 5, 0), SemVer(1, 0, 0)]

    def test_best_match(self):
        versions = [SemVer.parse(v) for v in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]]
        c = Constraint("^1.0.0")
        assert c.best_match(versions) == SemVer(1, 2, 0)

    def test_best_match_none(self):
        versions = [SemVer.parse("1.0.0")]
        c = Constraint("^2.0.0")
        assert c.best_match(versions) is None


# ============================================================
# Registry
# ============================================================

class TestRegistry:
    def test_publish_and_get(self):
        reg = Registry()
        spec = PackageSpec(name="foo", version=SemVer(1, 0, 0))
        reg.publish(spec)
        assert reg.get("foo", SemVer(1, 0, 0)) == spec

    def test_publish_duplicate_raises(self):
        reg = Registry()
        spec = PackageSpec(name="foo", version=SemVer(1, 0, 0))
        reg.publish(spec)
        with pytest.raises(RegistryError):
            reg.publish(spec)

    def test_versions(self):
        reg = Registry()
        for v in ["1.0.0", "2.0.0", "1.5.0"]:
            reg.publish(PackageSpec(name="foo", version=SemVer.parse(v)))
        versions = reg.versions("foo")
        assert versions == [SemVer(2, 0, 0), SemVer(1, 5, 0), SemVer(1, 0, 0)]

    def test_versions_empty(self):
        reg = Registry()
        assert reg.versions("nonexistent") == []

    def test_yank(self):
        reg = Registry()
        reg.publish(PackageSpec(name="foo", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="foo", version=SemVer(2, 0, 0)))
        reg.yank("foo", SemVer(1, 0, 0))
        assert SemVer(1, 0, 0) not in reg.versions("foo")
        assert SemVer(1, 0, 0) in reg.versions("foo", include_yanked=True)

    def test_yank_nonexistent_raises(self):
        reg = Registry()
        with pytest.raises(RegistryError):
            reg.yank("foo", SemVer(1, 0, 0))

    def test_get_nonexistent_raises(self):
        reg = Registry()
        with pytest.raises(RegistryError):
            reg.get("foo", SemVer(1, 0, 0))

    def test_search(self):
        reg = Registry()
        reg.publish(PackageSpec(name="json-parser", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="json-utils", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="xml-parser", version=SemVer(1, 0, 0)))
        assert reg.search("json") == ["json-parser", "json-utils"]

    def test_latest(self):
        reg = Registry()
        reg.publish(PackageSpec(name="foo", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="foo", version=SemVer(2, 0, 0)))
        assert reg.latest("foo").version == SemVer(2, 0, 0)

    def test_latest_skips_yanked(self):
        reg = Registry()
        reg.publish(PackageSpec(name="foo", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="foo", version=SemVer(2, 0, 0)))
        reg.yank("foo", SemVer(2, 0, 0))
        assert reg.latest("foo").version == SemVer(1, 0, 0)

    def test_contains(self):
        reg = Registry()
        reg.publish(PackageSpec(name="foo", version=SemVer(1, 0, 0)))
        assert "foo" in reg
        assert "bar" not in reg

    def test_all_packages(self):
        reg = Registry()
        reg.publish(PackageSpec(name="b", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0)))
        assert reg.all_packages() == ["a", "b"]

    def test_republish_after_yank(self):
        reg = Registry()
        spec = PackageSpec(name="foo", version=SemVer(1, 0, 0))
        reg.publish(spec)
        reg.yank("foo", SemVer(1, 0, 0))
        # Can republish a yanked version
        new_spec = PackageSpec(name="foo", version=SemVer(1, 0, 0), description="fixed")
        reg.publish(new_spec)
        assert reg.get("foo", SemVer(1, 0, 0)).description == "fixed"


# ============================================================
# PackageSpec
# ============================================================

class TestPackageSpec:
    def test_to_dict_from_dict_roundtrip(self):
        spec = PackageSpec(
            name="foo", version=SemVer(1, 2, 3),
            dependencies={"bar": "^1.0.0"},
            description="A package"
        )
        d = spec.to_dict()
        spec2 = PackageSpec.from_dict(d)
        assert spec2.name == spec.name
        assert spec2.version == spec.version
        assert spec2.dependencies == spec.dependencies

    def test_content_hash_deterministic(self):
        spec = PackageSpec(name="foo", version=SemVer(1, 0, 0), files={"a.py": "code"})
        h1 = spec.content_hash()
        h2 = spec.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_differs(self):
        s1 = PackageSpec(name="foo", version=SemVer(1, 0, 0))
        s2 = PackageSpec(name="foo", version=SemVer(1, 0, 1))
        assert s1.content_hash() != s2.content_hash()


# ============================================================
# Resolver -- Basic
# ============================================================

class TestResolverBasic:
    def _make_registry(self):
        reg = Registry()
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="a", version=SemVer(2, 0, 0)))
        reg.publish(PackageSpec(name="b", version=SemVer(1, 0, 0),
                                dependencies={"a": "^1.0.0"}))
        reg.publish(PackageSpec(name="c", version=SemVer(1, 0, 0),
                                dependencies={"a": "^2.0.0"}))
        return reg

    def test_simple_resolve(self):
        reg = self._make_registry()
        r = Resolver(reg)
        result = r.resolve({"a": "^1.0.0"})
        assert result["a"] == SemVer(1, 0, 0)

    def test_resolve_picks_latest(self):
        reg = Registry()
        reg.publish(PackageSpec(name="x", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="x", version=SemVer(1, 1, 0)))
        reg.publish(PackageSpec(name="x", version=SemVer(1, 2, 0)))
        r = Resolver(reg)
        result = r.resolve({"x": "^1.0.0"})
        assert result["x"] == SemVer(1, 2, 0)

    def test_resolve_with_transitive_deps(self):
        reg = self._make_registry()
        r = Resolver(reg)
        result = r.resolve({"b": "^1.0.0"})
        assert "a" in result
        assert "b" in result
        assert result["a"] == SemVer(1, 0, 0)

    def test_resolve_conflict(self):
        """b needs a^1, c needs a^2 -- conflict if both required."""
        reg = self._make_registry()
        r = Resolver(reg)
        with pytest.raises(ResolutionError):
            r.resolve({"b": "^1.0.0", "c": "^1.0.0"})

    def test_resolve_not_found(self):
        reg = Registry()
        r = Resolver(reg)
        with pytest.raises(ResolutionError, match="not found"):
            r.resolve({"nonexistent": "^1.0.0"})

    def test_resolve_no_matching_version(self):
        reg = Registry()
        reg.publish(PackageSpec(name="x", version=SemVer(1, 0, 0)))
        r = Resolver(reg)
        with pytest.raises(ResolutionError):
            r.resolve({"x": "^5.0.0"})


# ============================================================
# Resolver -- Advanced
# ============================================================

class TestResolverAdvanced:
    def test_diamond_dependency(self):
        """
        app -> lib-a ^1.0 -> utils ^1.0
        app -> lib-b ^1.0 -> utils ^1.0
        Should resolve utils to a single compatible version.
        """
        reg = Registry()
        reg.publish(PackageSpec(name="utils", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="utils", version=SemVer(1, 1, 0)))
        reg.publish(PackageSpec(name="lib-a", version=SemVer(1, 0, 0),
                                dependencies={"utils": "^1.0.0"}))
        reg.publish(PackageSpec(name="lib-b", version=SemVer(1, 0, 0),
                                dependencies={"utils": ">=1.1.0"}))
        r = Resolver(reg)
        result = r.resolve({"lib-a": "^1.0.0", "lib-b": "^1.0.0"})
        assert result["utils"] == SemVer(1, 1, 0)

    def test_backtracking(self):
        """
        x@2.0 depends on y@^2.0 (no y@2.x exists)
        x@1.0 depends on y@^1.0
        Should backtrack from x@2.0 to x@1.0.
        """
        reg = Registry()
        reg.publish(PackageSpec(name="y", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="x", version=SemVer(2, 0, 0),
                                dependencies={"y": "^2.0.0"}))
        reg.publish(PackageSpec(name="x", version=SemVer(1, 0, 0),
                                dependencies={"y": "^1.0.0"}))
        r = Resolver(reg)
        result = r.resolve({"x": ">=1.0.0"})
        assert result["x"] == SemVer(1, 0, 0)
        assert result["y"] == SemVer(1, 0, 0)

    def test_locked_version_preferred(self):
        reg = Registry()
        reg.publish(PackageSpec(name="x", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="x", version=SemVer(1, 1, 0)))
        r = Resolver(reg)
        result = r.resolve({"x": "^1.0.0"}, locked={"x": SemVer(1, 0, 0)})
        assert result["x"] == SemVer(1, 0, 0)

    def test_dev_dependencies(self):
        reg = Registry()
        reg.publish(PackageSpec(name="test-lib", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="app", version=SemVer(1, 0, 0),
                                dev_dependencies={"test-lib": "^1.0.0"}))
        r = Resolver(reg)
        # Without dev deps
        result = r.resolve({"app": "^1.0.0"}, include_dev=False)
        assert "test-lib" not in result
        # With dev deps
        result = r.resolve({"app": "^1.0.0"}, include_dev=True)
        assert "test-lib" in result

    def test_deep_chain(self):
        """a -> b -> c -> d, all at ^1.0.0"""
        reg = Registry()
        reg.publish(PackageSpec(name="d", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="c", version=SemVer(1, 0, 0),
                                dependencies={"d": "^1.0.0"}))
        reg.publish(PackageSpec(name="b", version=SemVer(1, 0, 0),
                                dependencies={"c": "^1.0.0"}))
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0),
                                dependencies={"b": "^1.0.0"}))
        r = Resolver(reg)
        result = r.resolve({"a": "^1.0.0"})
        assert len(result) == 4
        assert all(v == SemVer(1, 0, 0) for v in result.values())

    def test_multiple_versions_available(self):
        """With many versions, resolver picks the best compatible set."""
        reg = Registry()
        for v in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]:
            reg.publish(PackageSpec(name="lib", version=SemVer.parse(v)))
        reg.publish(PackageSpec(name="app-a", version=SemVer(1, 0, 0),
                                dependencies={"lib": ">=1.1.0 <2.0.0"}))
        reg.publish(PackageSpec(name="app-b", version=SemVer(1, 0, 0),
                                dependencies={"lib": "~1.1"}))
        r = Resolver(reg)
        result = r.resolve({"app-a": "^1.0.0", "app-b": "^1.0.0"})
        # lib must be >=1.1.0 <2.0.0 AND >=1.1.0 <1.2.0 => 1.1.x
        assert result["lib"] == SemVer(1, 1, 0)

    def test_cycle_detection(self):
        """a -> b -> a creates a cycle."""
        reg = Registry()
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0),
                                dependencies={"b": "^1.0.0"}))
        reg.publish(PackageSpec(name="b", version=SemVer(1, 0, 0),
                                dependencies={"a": "^1.0.0"}))
        r = Resolver(reg)
        with pytest.raises(ResolutionError, match="[Cc]ircular"):
            r.resolve({"a": "^1.0.0"})


# ============================================================
# Lockfile
# ============================================================

class TestLockfile:
    def _make_locked(self):
        reg = Registry()
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="b", version=SemVer(2, 0, 0),
                                dependencies={"a": "^1.0.0"}))
        lf = Lockfile()
        lf.lock({"a": SemVer(1, 0, 0), "b": SemVer(2, 0, 0)}, reg)
        return lf, reg

    def test_lock_creates_entries(self):
        lf, _ = self._make_locked()
        assert "a" in lf.entries
        assert "b" in lf.entries

    def test_locked_versions(self):
        lf, _ = self._make_locked()
        versions = lf.locked_versions()
        assert versions["a"] == SemVer(1, 0, 0)
        assert versions["b"] == SemVer(2, 0, 0)

    def test_is_satisfied(self):
        lf, _ = self._make_locked()
        assert lf.is_satisfied({"a": "^1.0.0", "b": "^2.0.0"})
        assert not lf.is_satisfied({"a": "^2.0.0"})

    def test_is_satisfied_missing_package(self):
        lf, _ = self._make_locked()
        assert not lf.is_satisfied({"c": "^1.0.0"})

    def test_json_roundtrip(self):
        lf, _ = self._make_locked()
        json_str = lf.to_json()
        lf2 = Lockfile.from_json(json_str)
        assert lf2.entries["a"].version == SemVer(1, 0, 0)
        assert lf2.entries["b"].version == SemVer(2, 0, 0)

    def test_integrity_check(self):
        lf, reg = self._make_locked()
        assert lf.verify_integrity(reg) == []

    def test_integrity_mismatch(self):
        lf, reg = self._make_locked()
        lf.entries["a"].integrity = "wrong_hash"
        mismatches = lf.verify_integrity(reg)
        assert "a" in mismatches

    def test_diff(self):
        lf1, reg = self._make_locked()
        lf2 = Lockfile()
        reg.publish(PackageSpec(name="a", version=SemVer(1, 1, 0)))
        reg.publish(PackageSpec(name="c", version=SemVer(1, 0, 0)))
        lf2.lock({"a": SemVer(1, 1, 0), "c": SemVer(1, 0, 0)}, reg)

        diff = lf1.diff(lf2)
        assert "a" in diff["changed"]
        assert "b" in diff["removed"]
        assert "c" in diff["added"]

    def test_diff_no_changes(self):
        lf, reg = self._make_locked()
        diff = lf.diff(lf)
        assert diff == {"added": {}, "removed": {}, "changed": {}}

    def test_deterministic_json(self):
        """Lockfile JSON should be deterministic (sorted keys)."""
        lf, _ = self._make_locked()
        j1 = lf.to_json()
        j2 = lf.to_json()
        assert j1 == j2


# ============================================================
# PackageManager -- Install
# ============================================================

class TestPackageManagerInstall:
    def _make_pm(self):
        reg = create_test_registry()
        return PackageManager(reg), reg

    def test_basic_install(self):
        pm, _ = self._make_pm()
        result = pm.install({"utils": "^1.0.0"})
        assert "utils" in result
        assert result["utils"] >= SemVer(1, 0, 0)

    def test_install_with_deps(self):
        pm, _ = self._make_pm()
        result = pm.install({"http": "^1.0.0"})
        assert "http" in result
        assert "utils" in result
        assert "logger" in result

    def test_install_creates_lockfile(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"})
        assert "utils" in pm.lockfile.entries

    def test_install_log(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"})
        log = pm.get_install_log()
        assert any("installed utils" in entry for entry in log)

    def test_install_idempotent(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"})
        # Second install should not re-install
        pm.install({"utils": "^1.0.0"})
        log = pm.get_install_log()
        assert len(log) == 0  # nothing to do

    def test_install_update_when_version_changes(self):
        pm, reg = self._make_pm()
        pm.install({"utils": "1.0.0"})
        assert pm.installed["utils"].version == SemVer(1, 0, 0)
        # Now install with different constraint
        pm.install({"utils": "1.1.0"})
        assert pm.installed["utils"].version == SemVer(1, 1, 0)
        log = pm.get_install_log()
        assert any("updated utils" in entry for entry in log)

    def test_install_removes_unneeded(self):
        pm, _ = self._make_pm()
        pm.install({"http": "^1.0.0"})
        assert "logger" in pm.installed
        # Now install only utils
        pm.install({"utils": "^1.0.0"})
        assert "logger" not in pm.installed

    def test_install_flat_strategy(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"}, strategy="flat")
        assert pm.installed["utils"].path == "packages/utils"

    def test_install_nested_strategy(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"}, strategy="nested")
        assert "node_modules" in pm.installed["utils"].path

    def test_install_with_lockfile(self):
        pm, _ = self._make_pm()
        # First install
        pm.install({"utils": "^1.0.0"})
        v1 = pm.installed["utils"].version
        # Second install should use lockfile
        pm.install({"utils": "^1.0.0"}, use_lockfile=True)
        assert pm.installed["utils"].version == v1

    def test_install_webapp_transitive(self):
        pm, _ = self._make_pm()
        result = pm.install({"webapp": "^1.0.0"})
        assert "webapp" in result
        assert "http" in result
        assert "json-parser" in result
        assert "utils" in result
        assert "logger" in result


# ============================================================
# PackageManager -- Uninstall
# ============================================================

class TestPackageManagerUninstall:
    def test_uninstall(self):
        pm, _ = TestPackageManagerInstall._make_pm(None)
        pm.install({"utils": "^1.0.0"})
        removed = pm.uninstall(["utils"])
        assert "utils" in removed
        assert "utils" not in pm.installed

    def test_uninstall_nonexistent(self):
        pm, _ = TestPackageManagerInstall._make_pm(None)
        removed = pm.uninstall(["nonexistent"])
        assert removed == []

    def test_uninstall_with_orphan_removal(self):
        pm, _ = TestPackageManagerInstall._make_pm(None)
        pm.install({"http": "^1.0.0"})
        assert "utils" in pm.installed
        assert "logger" in pm.installed
        # Uninstall http -- utils and logger should become orphans
        removed = pm.uninstall(["http"], requirements={})
        assert "http" in removed


# ============================================================
# PackageManager -- Update
# ============================================================

class TestPackageManagerUpdate:
    def test_update_all(self):
        pm, reg = TestPackageManagerInstall._make_pm(None)
        pm.install({"utils": "1.0.0"})
        assert pm.installed["utils"].version == SemVer(1, 0, 0)
        changes = pm.update({"utils": "^1.0.0"})
        # Should update to latest ^1.x
        assert pm.installed["utils"].version > SemVer(1, 0, 0)

    def test_update_specific_package(self):
        pm, reg = TestPackageManagerInstall._make_pm(None)
        pm.install({"http": "^1.0.0"})
        initial_utils = pm.installed["utils"].version
        # Update only logger
        changes = pm.update({"http": "^1.0.0"}, packages=["logger"])
        # utils should not have changed (it was locked)


# ============================================================
# PackageManager -- Misc
# ============================================================

class TestPackageManagerMisc:
    def _make_pm(self):
        return TestPackageManagerInstall._make_pm(None)

    def test_list_installed(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"})
        listed = pm.list_installed()
        assert len(listed) == 1
        assert listed[0].name == "utils"

    def test_install_order(self):
        pm, _ = self._make_pm()
        pm.install({"http": "^1.0.0"})
        order = pm.install_order()
        # utils and logger should come before http
        assert order.index("utils") < order.index("http")
        assert order.index("logger") < order.index("http")

    def test_dependency_tree(self):
        pm, _ = self._make_pm()
        pm.install({"http": "^1.0.0"})
        tree = pm.dependency_tree({"http": "^1.0.0"})
        assert "http" in tree
        assert "utils" in tree["http"]["deps"]
        assert "logger" in tree["http"]["deps"]

    def test_dependency_tree_empty(self):
        pm, _ = self._make_pm()
        assert pm.dependency_tree() == {}

    def test_audit_clean(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"})
        issues = pm.audit()
        assert issues["yanked"] == []
        assert issues["integrity_mismatch"] == []
        assert issues["missing_deps"] == []

    def test_audit_yanked(self):
        pm, reg = self._make_pm()
        pm.install({"utils": "^1.0.0"})
        v = pm.installed["utils"].version
        reg.yank("utils", v)
        issues = pm.audit()
        assert len(issues["yanked"]) == 1

    def test_audit_integrity_mismatch(self):
        pm, _ = self._make_pm()
        pm.install({"utils": "^1.0.0"})
        pm.installed["utils"].integrity = "tampered"
        issues = pm.audit()
        assert "utils" in issues["integrity_mismatch"]

    def test_audit_missing_deps(self):
        pm, _ = self._make_pm()
        pm.install({"http": "^1.0.0"})
        # Manually remove a dep
        del pm.installed["utils"]
        issues = pm.audit()
        assert any("utils" in issue for issue in issues["missing_deps"])


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """Publish, install, lock, update, uninstall."""
        reg = Registry()
        reg.publish(PackageSpec(name="core", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="core", version=SemVer(1, 1, 0)))
        reg.publish(PackageSpec(name="app", version=SemVer(1, 0, 0),
                                dependencies={"core": "^1.0.0"}))

        pm = PackageManager(reg)

        # Install
        result = pm.install({"app": "^1.0.0"})
        assert result["core"] == SemVer(1, 1, 0)
        assert result["app"] == SemVer(1, 0, 0)

        # Lock
        lf_json = pm.lockfile.to_json()
        lf2 = Lockfile.from_json(lf_json)
        assert lf2.entries["core"].version == SemVer(1, 1, 0)

        # Publish new version
        reg.publish(PackageSpec(name="core", version=SemVer(1, 2, 0)))

        # Lockfile should prevent upgrade
        pm2 = PackageManager(reg)
        pm2.lockfile = lf2
        result2 = pm2.install({"app": "^1.0.0"})
        assert result2["core"] == SemVer(1, 1, 0)  # locked

        # Update should get latest
        pm2.update({"app": "^1.0.0"})
        assert pm2.installed["core"].version == SemVer(1, 2, 0)

        # Uninstall
        pm2.uninstall(["app", "core"])
        assert len(pm2.installed) == 0

    def test_complex_dependency_graph(self):
        """Multiple packages with overlapping dependencies."""
        reg = Registry()
        reg.publish(PackageSpec(name="base", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="base", version=SemVer(1, 1, 0)))
        reg.publish(PackageSpec(name="base", version=SemVer(2, 0, 0)))

        reg.publish(PackageSpec(name="mid-a", version=SemVer(1, 0, 0),
                                dependencies={"base": "^1.0.0"}))
        reg.publish(PackageSpec(name="mid-b", version=SemVer(1, 0, 0),
                                dependencies={"base": ">=1.1.0 <2.0.0"}))
        reg.publish(PackageSpec(name="top", version=SemVer(1, 0, 0),
                                dependencies={"mid-a": "^1.0.0", "mid-b": "^1.0.0"}))

        pm = PackageManager(reg)
        result = pm.install({"top": "^1.0.0"})

        # base must satisfy both ^1.0.0 AND >=1.1.0 <2.0.0 => 1.1.0
        assert result["base"] == SemVer(1, 1, 0)
        assert len(result) == 4

    def test_version_conflict_error_message(self):
        """Verify conflict errors are informative."""
        reg = Registry()
        reg.publish(PackageSpec(name="lib", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="lib", version=SemVer(2, 0, 0)))
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0),
                                dependencies={"lib": "^1.0.0"}))
        reg.publish(PackageSpec(name="b", version=SemVer(1, 0, 0),
                                dependencies={"lib": "^2.0.0"}))

        pm = PackageManager(reg)
        with pytest.raises(ResolutionError) as exc_info:
            pm.install({"a": "^1.0.0", "b": "^1.0.0"})
        assert "lib" in str(exc_info.value)

    def test_lockfile_diff_after_update(self):
        reg = Registry()
        reg.publish(PackageSpec(name="x", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="x", version=SemVer(1, 1, 0)))

        pm = PackageManager(reg)
        pm.install({"x": "1.0.0"})
        old_lf = Lockfile.from_json(pm.lockfile.to_json())

        pm.install({"x": "1.1.0"})
        diff = old_lf.diff(pm.lockfile)
        assert "x" in diff["changed"]
        assert diff["changed"]["x"] == (SemVer(1, 0, 0), SemVer(1, 1, 0))

    def test_install_with_files(self):
        """Packages with file content are installed correctly."""
        reg = Registry()
        reg.publish(PackageSpec(
            name="my-lib", version=SemVer(1, 0, 0),
            files={"index.py": "print('hello')", "README.md": "# my-lib"}
        ))
        pm = PackageManager(reg)
        pm.install({"my-lib": "^1.0.0"})
        assert pm.installed["my-lib"].files["index.py"] == "print('hello')"

    def test_wildcard_constraint_install(self):
        reg = Registry()
        reg.publish(PackageSpec(name="any-ver", version=SemVer(3, 5, 2)))
        pm = PackageManager(reg)
        result = pm.install({"any-ver": "*"})
        assert result["any-ver"] == SemVer(3, 5, 2)

    def test_or_constraint_install(self):
        reg = Registry()
        reg.publish(PackageSpec(name="lib", version=SemVer(1, 5, 0)))
        reg.publish(PackageSpec(name="lib", version=SemVer(3, 0, 0)))
        pm = PackageManager(reg)
        result = pm.install({"lib": "^1.0.0 || ^3.0.0"})
        assert result["lib"] == SemVer(3, 0, 0)  # picks latest matching


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_self_dependency_cycle(self):
        reg = Registry()
        reg.publish(PackageSpec(name="self-ref", version=SemVer(1, 0, 0),
                                dependencies={"self-ref": "^1.0.0"}))
        pm = PackageManager(reg)
        with pytest.raises(ResolutionError):
            pm.install({"self-ref": "^1.0.0"})

    def test_empty_requirements(self):
        reg = Registry()
        pm = PackageManager(reg)
        result = pm.install({})
        assert result == {}

    def test_prerelease_version_in_registry(self):
        reg = Registry()
        reg.publish(PackageSpec(name="beta-lib", version=SemVer.parse("1.0.0-beta.1")))
        reg.publish(PackageSpec(name="beta-lib", version=SemVer(1, 0, 0)))
        pm = PackageManager(reg)
        result = pm.install({"beta-lib": ">=1.0.0-beta.1"})
        assert result["beta-lib"] == SemVer(1, 0, 0)  # prefers release

    def test_many_versions_resolver_efficiency(self):
        """Resolver should handle packages with many versions."""
        reg = Registry()
        for i in range(50):
            reg.publish(PackageSpec(name="big", version=SemVer(1, i, 0)))
        pm = PackageManager(reg)
        result = pm.install({"big": "^1.0.0"})
        assert result["big"] == SemVer(1, 49, 0)

    def test_lockfile_empty(self):
        lf = Lockfile()
        assert lf.locked_versions() == {}
        assert lf.is_satisfied({}) is True
        j = lf.to_json()
        lf2 = Lockfile.from_json(j)
        assert lf2.locked_versions() == {}

    def test_constraint_prerelease(self):
        c = Constraint(">=1.0.0-alpha")
        assert c.satisfies(SemVer.parse("1.0.0-alpha"))
        assert c.satisfies(SemVer.parse("1.0.0-beta"))
        assert c.satisfies(SemVer.parse("1.0.0"))

    def test_semver_repr(self):
        v = SemVer(1, 2, 3)
        assert repr(v) == "SemVer(1.2.3)"

    def test_registry_len(self):
        reg = Registry()
        assert len(reg) == 0
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0)))
        assert len(reg) == 1

    def test_package_manager_no_lockfile(self):
        """Install without lockfile should always resolve fresh."""
        reg = Registry()
        reg.publish(PackageSpec(name="x", version=SemVer(1, 0, 0)))
        reg.publish(PackageSpec(name="x", version=SemVer(1, 1, 0)))
        pm = PackageManager(reg)
        pm.install({"x": "1.0.0"})
        # Force fresh resolve
        pm.install({"x": "^1.0.0"}, use_lockfile=False)
        assert pm.installed["x"].version == SemVer(1, 1, 0)

    def test_three_way_diamond(self):
        """
        root -> a, b, c
        a -> shared ^1.0
        b -> shared >=1.2.0
        c -> shared ~1.2
        shared must be 1.2.x
        """
        reg = Registry()
        for v in ["1.0.0", "1.1.0", "1.2.0", "1.2.1", "1.3.0"]:
            reg.publish(PackageSpec(name="shared", version=SemVer.parse(v)))
        reg.publish(PackageSpec(name="a", version=SemVer(1, 0, 0),
                                dependencies={"shared": "^1.0.0"}))
        reg.publish(PackageSpec(name="b", version=SemVer(1, 0, 0),
                                dependencies={"shared": ">=1.2.0"}))
        reg.publish(PackageSpec(name="c", version=SemVer(1, 0, 0),
                                dependencies={"shared": "~1.2"}))
        pm = PackageManager(reg)
        result = pm.install({"a": "^1.0.0", "b": "^1.0.0", "c": "^1.0.0"})
        assert result["shared"] == SemVer(1, 2, 1)  # highest that satisfies all


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
