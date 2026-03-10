"""
Package Manager
Challenge C021 -- AgentZero Session 022

A package manager that composes:
  - Dependency resolver (C008) for topological sort and cycle detection
  - Semver version parsing, comparison, and constraint matching

Features:
  - Semantic versioning: parse, compare, sort versions
  - Version constraints: ^, ~, >=, <=, >, <, =, *, ranges, AND/OR
  - Package registry: publish, query, yank versions
  - Version resolver: backtracking SAT-style resolution with conflict reporting
  - Lockfile: deterministic, reproducible installs with integrity hashes
  - Install: fetch from registry, flat or nested install strategies
  - Uninstall: remove packages and check for orphans
  - Dependency tree: visualize full resolved dependency graph
"""

import os
import sys
import json
import hashlib
import copy
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# Import dependency resolver from C008
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C008_dependency_resolver'))
from resolver import DependencyGraph, CycleError


# ============================================================
# Semantic Versioning
# ============================================================

@dataclass(frozen=True, order=False)
class SemVer:
    """Semantic version: major.minor.patch[-prerelease][+build]"""
    major: int
    minor: int
    patch: int
    prerelease: tuple = ()
    build: tuple = ()

    @staticmethod
    def parse(s: str) -> 'SemVer':
        """Parse a semver string like '1.2.3', '1.2.3-alpha.1', '1.2.3+build'."""
        s = s.strip()
        if s.startswith('v'):
            s = s[1:]

        # Split off build metadata
        build = ()
        if '+' in s:
            s, build_str = s.split('+', 1)
            build = tuple(build_str.split('.'))

        # Split off prerelease
        prerelease = ()
        if '-' in s:
            s, pre_str = s.split('-', 1)
            parts = []
            for p in pre_str.split('.'):
                if p.isdigit():
                    parts.append(int(p))
                else:
                    parts.append(p)
            prerelease = tuple(parts)

        # Parse major.minor.patch
        parts = s.split('.')
        if len(parts) == 1:
            return SemVer(int(parts[0]), 0, 0, prerelease, build)
        if len(parts) == 2:
            return SemVer(int(parts[0]), int(parts[1]), 0, prerelease, build)
        if len(parts) >= 3:
            return SemVer(int(parts[0]), int(parts[1]), int(parts[2]), prerelease, build)

    def _cmp_key(self):
        """Comparison key per semver spec: build metadata is ignored."""
        # Prerelease has lower precedence than release
        # No prerelease = higher than any prerelease
        if self.prerelease:
            pre_key = []
            for p in self.prerelease:
                if isinstance(p, int):
                    pre_key.append((0, p, ''))
                else:
                    pre_key.append((1, 0, p))
            return (self.major, self.minor, self.patch, 0, tuple(pre_key))
        else:
            return (self.major, self.minor, self.patch, 1, ())

    def __lt__(self, other):
        if not isinstance(other, SemVer):
            return NotImplemented
        return self._cmp_key() < other._cmp_key()

    def __le__(self, other):
        if not isinstance(other, SemVer):
            return NotImplemented
        return self._cmp_key() <= other._cmp_key()

    def __gt__(self, other):
        if not isinstance(other, SemVer):
            return NotImplemented
        return self._cmp_key() > other._cmp_key()

    def __ge__(self, other):
        if not isinstance(other, SemVer):
            return NotImplemented
        return self._cmp_key() >= other._cmp_key()

    def __eq__(self, other):
        if not isinstance(other, SemVer):
            return NotImplemented
        return self._cmp_key() == other._cmp_key()

    def __hash__(self):
        return hash(self._cmp_key())

    def __str__(self):
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += '-' + '.'.join(str(p) for p in self.prerelease)
        if self.build:
            s += '+' + '.'.join(self.build)
        return s

    def __repr__(self):
        return f"SemVer({self!s})"

    def bump_major(self) -> 'SemVer':
        return SemVer(self.major + 1, 0, 0)

    def bump_minor(self) -> 'SemVer':
        return SemVer(self.major, self.minor + 1, 0)

    def bump_patch(self) -> 'SemVer':
        return SemVer(self.major, self.minor, self.patch + 1)


# ============================================================
# Version Constraints
# ============================================================

class Constraint:
    """A version constraint that can test whether a version satisfies it."""

    def __init__(self, spec: str):
        self.spec = spec.strip()
        self._matchers = self._parse(self.spec)

    def _parse(self, spec: str) -> list:
        """Parse a constraint spec into a list of matchers.

        Supports:
          - Exact: "1.2.3" or "=1.2.3"
          - Comparisons: ">=1.0.0", "<2.0.0", ">1.0", "<=1.5.0"
          - Caret: "^1.2.3" (>=1.2.3, <2.0.0), "^0.2.3" (>=0.2.3, <0.3.0)
          - Tilde: "~1.2.3" (>=1.2.3, <1.3.0), "~1.2" (>=1.2.0, <1.3.0)
          - Wildcard: "*", "1.*", "1.2.*"
          - Range: ">=1.0.0 <2.0.0" (AND -- space separated)
          - OR: ">=1.0 || >=2.0" (pipe separated)
        """
        # OR groups
        if '||' in spec:
            groups = [s.strip() for s in spec.split('||')]
            return [('or', [self._parse(g) for g in groups])]

        # AND groups (space-separated comparisons)
        parts = self._split_and(spec)
        if len(parts) > 1:
            return [('and', [self._parse_single(p) for p in parts])]

        return [self._parse_single(spec)]

    def _split_and(self, spec: str) -> list[str]:
        """Split space-separated constraints, keeping operators with their versions."""
        parts = []
        current = ''
        tokens = spec.split()
        for t in tokens:
            if t and t[0] in '>=<^~!' and current:
                parts.append(current.strip())
                current = t
            elif current:
                current += ' ' + t
            else:
                current = t
        if current:
            parts.append(current.strip())
        return parts

    def _parse_single(self, spec: str) -> tuple:
        spec = spec.strip()

        if spec == '*':
            return ('any',)

        if spec.endswith('.*'):
            # Wildcard like "1.*" or "1.2.*"
            prefix = spec[:-2]
            parts = prefix.split('.')
            if len(parts) == 1:
                v = int(parts[0])
                return ('range', SemVer(v, 0, 0), SemVer(v + 1, 0, 0))
            elif len(parts) == 2:
                ma, mi = int(parts[0]), int(parts[1])
                return ('range', SemVer(ma, mi, 0), SemVer(ma, mi + 1, 0))

        if spec.startswith('^'):
            v = SemVer.parse(spec[1:])
            if v.major != 0:
                upper = SemVer(v.major + 1, 0, 0)
            elif v.minor != 0:
                upper = SemVer(0, v.minor + 1, 0)
            else:
                upper = SemVer(0, 0, v.patch + 1)
            return ('range', v, upper)

        if spec.startswith('~'):
            v = SemVer.parse(spec[1:])
            upper = SemVer(v.major, v.minor + 1, 0)
            return ('range', v, upper)

        if spec.startswith('>='):
            return ('gte', SemVer.parse(spec[2:]))
        if spec.startswith('<='):
            return ('lte', SemVer.parse(spec[2:]))
        if spec.startswith('!='):
            return ('neq', SemVer.parse(spec[2:]))
        if spec.startswith('>'):
            return ('gt', SemVer.parse(spec[1:]))
        if spec.startswith('<'):
            return ('lt', SemVer.parse(spec[1:]))
        if spec.startswith('='):
            return ('eq', SemVer.parse(spec[1:]))

        # Bare version = exact match
        return ('eq', SemVer.parse(spec))

    def satisfies(self, version: SemVer) -> bool:
        """Check if a version satisfies this constraint."""
        return all(self._check(m, version) for m in self._matchers)

    def _check(self, matcher: tuple, version: SemVer) -> bool:
        op = matcher[0]

        if op == 'any':
            return True
        elif op == 'eq':
            return version == matcher[1]
        elif op == 'neq':
            return version != matcher[1]
        elif op == 'gt':
            return version > matcher[1]
        elif op == 'gte':
            return version >= matcher[1]
        elif op == 'lt':
            return version < matcher[1]
        elif op == 'lte':
            return version <= matcher[1]
        elif op == 'range':
            return version >= matcher[1] and version < matcher[2]
        elif op == 'and':
            return all(self._check(m, version) for m in matcher[1])
        elif op == 'or':
            return any(
                all(self._check(m, version) for m in group)
                for group in matcher[1]
            )
        return False

    def filter(self, versions: list[SemVer]) -> list[SemVer]:
        """Return versions that satisfy this constraint, sorted descending."""
        return sorted([v for v in versions if self.satisfies(v)], reverse=True)

    def best_match(self, versions: list[SemVer]) -> Optional[SemVer]:
        """Return the highest version satisfying this constraint, or None."""
        matches = self.filter(versions)
        return matches[0] if matches else None

    def __repr__(self):
        return f"Constraint({self.spec!r})"

    def __str__(self):
        return self.spec


# ============================================================
# Package Metadata
# ============================================================

@dataclass
class PackageSpec:
    """A package definition with name, version, dependencies, and metadata."""
    name: str
    version: SemVer
    dependencies: dict[str, str] = field(default_factory=dict)  # name -> constraint string
    dev_dependencies: dict[str, str] = field(default_factory=dict)
    description: str = ""
    author: str = ""
    license: str = ""
    files: dict[str, str] = field(default_factory=dict)  # path -> content (for testing)
    yanked: bool = False

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'version': str(self.version),
            'dependencies': dict(self.dependencies),
            'dev_dependencies': dict(self.dev_dependencies),
            'description': self.description,
            'author': self.author,
            'license': self.license,
        }

    @staticmethod
    def from_dict(d: dict) -> 'PackageSpec':
        return PackageSpec(
            name=d['name'],
            version=SemVer.parse(d['version']),
            dependencies=d.get('dependencies', {}),
            dev_dependencies=d.get('dev_dependencies', {}),
            description=d.get('description', ''),
            author=d.get('author', ''),
            license=d.get('license', ''),
        )

    def content_hash(self) -> str:
        """SHA-256 hash of package content for integrity checking."""
        data = json.dumps(self.to_dict(), sort_keys=True) + json.dumps(self.files, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def __repr__(self):
        return f"PackageSpec({self.name}@{self.version})"


# ============================================================
# Registry
# ============================================================

class RegistryError(Exception):
    pass


class Registry:
    """Package registry -- stores published packages and their metadata."""

    def __init__(self):
        self._packages: dict[str, dict[SemVer, PackageSpec]] = defaultdict(dict)

    def publish(self, spec: PackageSpec) -> None:
        """Publish a package version. Raises if version already exists."""
        if spec.version in self._packages[spec.name]:
            existing = self._packages[spec.name][spec.version]
            if not existing.yanked:
                raise RegistryError(
                    f"Version {spec.version} of '{spec.name}' already published"
                )
        self._packages[spec.name][spec.version] = spec

    def yank(self, name: str, version: SemVer) -> None:
        """Yank a version (mark as not recommended, still resolvable if pinned)."""
        if name not in self._packages or version not in self._packages[name]:
            raise RegistryError(f"Package '{name}@{version}' not found")
        self._packages[name][version].yanked = True

    def get(self, name: str, version: SemVer) -> PackageSpec:
        """Get a specific package version."""
        if name not in self._packages or version not in self._packages[name]:
            raise RegistryError(f"Package '{name}@{version}' not found")
        return self._packages[name][version]

    def versions(self, name: str, include_yanked: bool = False) -> list[SemVer]:
        """Get all available versions of a package, sorted descending."""
        if name not in self._packages:
            return []
        versions = []
        for v, spec in self._packages[name].items():
            if include_yanked or not spec.yanked:
                versions.append(v)
        return sorted(versions, reverse=True)

    def search(self, query: str) -> list[str]:
        """Search for packages by name substring."""
        return sorted(name for name in self._packages if query.lower() in name.lower())

    def all_packages(self) -> list[str]:
        """List all package names."""
        return sorted(self._packages.keys())

    def latest(self, name: str) -> Optional[PackageSpec]:
        """Get the latest non-yanked version of a package."""
        versions = self.versions(name)
        if not versions:
            return None
        return self._packages[name][versions[0]]

    def __contains__(self, name: str) -> bool:
        return name in self._packages and len(self.versions(name)) > 0

    def __len__(self) -> int:
        return len(self._packages)


# ============================================================
# Resolution Errors
# ============================================================

class ResolutionError(Exception):
    """Raised when dependency resolution fails."""
    def __init__(self, message: str, conflicts: list = None):
        self.conflicts = conflicts or []
        super().__init__(message)


# ============================================================
# Version Resolver
# ============================================================

class Resolver:
    """Backtracking dependency resolver with version constraint solving.

    Given a set of root requirements, finds a compatible set of package versions
    from the registry. Uses backtracking when conflicts arise.
    """

    def __init__(self, registry: Registry):
        self.registry = registry
        self._resolution_path: list[str] = []  # for conflict reporting

    def resolve(self, requirements: dict[str, str],
                locked: dict[str, SemVer] = None,
                include_dev: bool = False) -> dict[str, SemVer]:
        """Resolve a set of requirements to concrete versions.

        Args:
            requirements: {package_name: constraint_string}
            locked: Previously locked versions to prefer
            include_dev: Whether to include dev_dependencies

        Returns:
            {package_name: resolved_version}

        Raises:
            ResolutionError if no valid resolution exists
        """
        locked = locked or {}
        resolved = {}
        constraints = {}  # name -> list of (constraint, required_by)

        # Seed with root requirements
        for name, spec in requirements.items():
            constraints[name] = [(Constraint(spec), 'root')]

        return self._solve(constraints, resolved, locked, include_dev)

    def _solve(self, constraints: dict, resolved: dict,
               locked: dict, include_dev: bool) -> dict[str, SemVer]:
        """Recursive backtracking solver."""
        # Find the next unresolved package
        unresolved = [name for name in constraints if name not in resolved]
        if not unresolved:
            # All resolved -- verify with topological sort for cycle detection
            self._check_cycles(resolved, include_dev)
            return dict(resolved)

        # Pick the most constrained package first (MCV heuristic)
        name = self._pick_next(unresolved, constraints)

        # Get candidate versions
        candidates = self._get_candidates(name, constraints[name], locked)

        if not candidates:
            # Build conflict report
            constraint_strs = [f"{c} (required by {by})" for c, by in constraints[name]]
            available = self.registry.versions(name)
            if not available:
                raise ResolutionError(
                    f"Package '{name}' not found in registry",
                    conflicts=[(name, constraint_strs)]
                )
            raise ResolutionError(
                f"No version of '{name}' satisfies all constraints: "
                + ", ".join(constraint_strs)
                + f". Available: {', '.join(str(v) for v in available)}",
                conflicts=[(name, constraint_strs)]
            )

        # Try each candidate
        errors = []
        for version in candidates:
            # Tentatively resolve
            resolved[name] = version

            # Get this version's dependencies
            spec = self.registry.get(name, version)
            deps = dict(spec.dependencies)
            if include_dev:
                deps.update(spec.dev_dependencies)

            # Add new constraints from this package's deps
            new_constraints = copy.deepcopy(constraints)
            conflict = False
            for dep_name, dep_spec in deps.items():
                new_constraint = (Constraint(dep_spec), f"{name}@{version}")
                if dep_name not in new_constraints:
                    new_constraints[dep_name] = [new_constraint]
                else:
                    new_constraints[dep_name] = list(new_constraints[dep_name]) + [new_constraint]

                # Check if already resolved and compatible
                if dep_name in resolved:
                    if not Constraint(dep_spec).satisfies(resolved[dep_name]):
                        conflict = True
                        errors.append(
                            f"{name}@{version} requires {dep_name} {dep_spec}, "
                            f"but {resolved[dep_name]} is already resolved"
                        )
                        break

            if conflict:
                del resolved[name]
                continue

            try:
                return self._solve(new_constraints, resolved, locked, include_dev)
            except ResolutionError as e:
                errors.append(str(e))
                del resolved[name]
                # Remove any deps this candidate added
                for dep_name in deps:
                    if dep_name in resolved and dep_name not in constraints:
                        del resolved[dep_name]
                continue

        raise ResolutionError(
            f"Could not resolve '{name}': exhausted all candidates. "
            + "; ".join(errors),
            conflicts=[(name, errors)]
        )

    def _pick_next(self, unresolved: list[str],
                   constraints: dict) -> str:
        """Pick the most constrained unresolved package (MCV heuristic)."""
        def score(name):
            versions = self.registry.versions(name)
            matching = 0
            for v in versions:
                if all(c.satisfies(v) for c, _ in constraints[name]):
                    matching += 1
            return matching

        return min(unresolved, key=score)

    def _get_candidates(self, name: str, constraint_pairs: list,
                        locked: dict) -> list[SemVer]:
        """Get candidate versions in preference order."""
        versions = self.registry.versions(name)

        # Filter by all constraints
        valid = []
        for v in versions:
            if all(c.satisfies(v) for c, _ in constraint_pairs):
                valid.append(v)

        # Prefer locked version if compatible
        if name in locked and locked[name] in valid:
            valid.remove(locked[name])
            valid.insert(0, locked[name])

        return valid

    def _check_cycles(self, resolved: dict, include_dev: bool) -> None:
        """Verify no circular dependencies in resolved set."""
        graph = DependencyGraph(strict=False)
        for name, version in resolved.items():
            spec = self.registry.get(name, version)
            deps = list(spec.dependencies.keys())
            if include_dev:
                deps.extend(spec.dev_dependencies.keys())
            # Only include deps that are in resolved set
            deps = [d for d in deps if d in resolved]
            graph.add(name, deps if deps else None)
        try:
            graph.resolve()
        except CycleError as e:
            raise ResolutionError(
                f"Circular dependency detected: {' -> '.join(e.cycle)}",
                conflicts=[('cycle', e.cycle)]
            )


# ============================================================
# Lockfile
# ============================================================

@dataclass
class LockEntry:
    """A single locked package entry."""
    name: str
    version: SemVer
    integrity: str  # content hash
    dependencies: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'version': str(self.version),
            'integrity': self.integrity,
            'dependencies': dict(self.dependencies),
        }

    @staticmethod
    def from_dict(d: dict) -> 'LockEntry':
        return LockEntry(
            name=d['name'],
            version=SemVer.parse(d['version']),
            integrity=d['integrity'],
            dependencies=d.get('dependencies', {}),
        )


class Lockfile:
    """Deterministic lockfile for reproducible installs."""

    def __init__(self):
        self.entries: dict[str, LockEntry] = {}
        self.lockfile_version: int = 1

    def lock(self, resolved: dict[str, SemVer], registry: Registry) -> None:
        """Create lock entries from resolved versions."""
        self.entries.clear()
        for name, version in sorted(resolved.items()):
            spec = registry.get(name, version)
            entry = LockEntry(
                name=name,
                version=version,
                integrity=spec.content_hash(),
                dependencies=dict(spec.dependencies),
            )
            self.entries[name] = entry

    def locked_versions(self) -> dict[str, SemVer]:
        """Get the locked versions as a dict."""
        return {name: entry.version for name, entry in self.entries.items()}

    def is_satisfied(self, requirements: dict[str, str]) -> bool:
        """Check if current lockfile satisfies the given requirements."""
        for name, spec in requirements.items():
            if name not in self.entries:
                return False
            c = Constraint(spec)
            if not c.satisfies(self.entries[name].version):
                return False
        return True

    def verify_integrity(self, registry: Registry) -> list[str]:
        """Check integrity of all locked packages. Returns list of mismatches."""
        mismatches = []
        for name, entry in self.entries.items():
            try:
                spec = registry.get(name, entry.version)
                if spec.content_hash() != entry.integrity:
                    mismatches.append(name)
            except RegistryError:
                mismatches.append(name)
        return mismatches

    def to_dict(self) -> dict:
        return {
            'lockfile_version': self.lockfile_version,
            'packages': {
                name: entry.to_dict()
                for name, entry in sorted(self.entries.items())
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def from_json(data: str) -> 'Lockfile':
        d = json.loads(data)
        lf = Lockfile()
        lf.lockfile_version = d.get('lockfile_version', 1)
        for name, entry_dict in d.get('packages', {}).items():
            lf.entries[name] = LockEntry.from_dict(entry_dict)
        return lf

    def diff(self, other: 'Lockfile') -> dict:
        """Compare two lockfiles. Returns added, removed, changed packages."""
        added = {}
        removed = {}
        changed = {}

        all_names = set(self.entries.keys()) | set(other.entries.keys())
        for name in sorted(all_names):
            in_self = name in self.entries
            in_other = name in other.entries
            if in_self and not in_other:
                removed[name] = self.entries[name].version
            elif not in_self and in_other:
                added[name] = other.entries[name].version
            elif self.entries[name].version != other.entries[name].version:
                changed[name] = (self.entries[name].version, other.entries[name].version)

        return {'added': added, 'removed': removed, 'changed': changed}


# ============================================================
# Install Target
# ============================================================

@dataclass
class InstalledPackage:
    """A package installed in the local project."""
    name: str
    version: SemVer
    integrity: str
    path: str  # install location
    files: dict[str, str] = field(default_factory=dict)


class InstallError(Exception):
    pass


# ============================================================
# Package Manager
# ============================================================

class PackageManager:
    """Full package manager: resolve, install, uninstall, update."""

    def __init__(self, registry: Registry, install_dir: str = "packages"):
        self.registry = registry
        self.install_dir = install_dir
        self.resolver = Resolver(registry)
        self.lockfile = Lockfile()
        self.installed: dict[str, InstalledPackage] = {}
        self._install_log: list[str] = []

    def install(self, requirements: dict[str, str],
                include_dev: bool = False,
                use_lockfile: bool = True,
                strategy: str = "flat") -> dict[str, SemVer]:
        """Install packages from requirements.

        Args:
            requirements: {name: constraint_string}
            include_dev: Include dev dependencies
            use_lockfile: Honor existing lockfile if compatible
            strategy: "flat" (all in install_dir/) or "nested" (node_modules style)

        Returns:
            Dict of installed package names to versions
        """
        self._install_log.clear()

        # Check if lockfile satisfies current requirements
        locked = {}
        if use_lockfile and self.lockfile.entries:
            if self.lockfile.is_satisfied(requirements):
                locked = self.lockfile.locked_versions()

        # Resolve
        resolved = self.resolver.resolve(
            requirements, locked=locked, include_dev=include_dev
        )

        # Determine what needs installing/updating/removing
        to_install = {}
        to_update = {}
        to_remove = {}

        for name, version in resolved.items():
            if name not in self.installed:
                to_install[name] = version
            elif self.installed[name].version != version:
                to_update[name] = (self.installed[name].version, version)
            # else: already installed at correct version

        for name in list(self.installed.keys()):
            if name not in resolved:
                to_remove[name] = self.installed[name].version

        # Execute installs
        for name, version in to_install.items():
            self._install_package(name, version, strategy)

        # Execute updates
        for name, (old_ver, new_ver) in to_update.items():
            self._uninstall_package(name)
            self._install_package(name, new_ver, strategy)
            self._install_log.append(f"updated {name}: {old_ver} -> {new_ver}")

        # Execute removes
        for name, version in to_remove.items():
            self._uninstall_package(name)
            self._install_log.append(f"removed {name}@{version}")

        # Update lockfile
        self.lockfile.lock(resolved, self.registry)

        return resolved

    def uninstall(self, names: list[str],
                  requirements: dict[str, str] = None) -> list[str]:
        """Uninstall packages. Returns list of actually removed packages.

        Also removes orphaned dependencies if requirements are provided.
        """
        removed = []
        for name in names:
            if name in self.installed:
                self._uninstall_package(name)
                removed.append(name)

        # Find and remove orphans if we know the requirements
        if requirements is not None:
            orphans = self._find_orphans(requirements)
            for name in orphans:
                if name in self.installed:
                    self._uninstall_package(name)
                    removed.append(name)

        return removed

    def update(self, requirements: dict[str, str],
               packages: list[str] = None,
               include_dev: bool = False) -> dict[str, tuple]:
        """Update packages to latest compatible versions.

        Args:
            requirements: Current requirements
            packages: Specific packages to update (None = all)
            include_dev: Include dev dependencies

        Returns:
            Dict of {name: (old_version, new_version)} for changed packages
        """
        # Resolve without lockfile to get latest versions
        old_versions = {name: pkg.version for name, pkg in self.installed.items()}

        # If updating specific packages, keep others locked
        locked = {}
        if packages:
            for name, pkg in self.installed.items():
                if name not in packages:
                    locked[name] = pkg.version

        resolved = self.resolver.resolve(
            requirements, locked=locked, include_dev=include_dev
        )

        # Install with new resolution (no lockfile)
        self.install(requirements, include_dev=include_dev, use_lockfile=False)

        # Report changes
        changes = {}
        for name, version in resolved.items():
            old = old_versions.get(name)
            if old and old != version:
                changes[name] = (old, version)

        return changes

    def list_installed(self) -> list[InstalledPackage]:
        """List all installed packages, sorted by name."""
        return sorted(self.installed.values(), key=lambda p: p.name)

    def dependency_tree(self, root_requirements: dict[str, str] = None) -> dict:
        """Build a dependency tree from installed packages.

        Returns nested dict: {name: {version, deps: {name: {version, deps: ...}}}}
        """
        if not self.installed:
            return {}

        def build_tree(name: str, seen: set = None) -> dict:
            if seen is None:
                seen = set()
            if name not in self.installed:
                return {'version': '?', 'deps': {}}
            if name in seen:
                return {'version': str(self.installed[name].version), 'deps': {'...': 'circular'}}

            seen = seen | {name}
            pkg = self.installed[name]
            spec = self.registry.get(name, pkg.version)

            deps = {}
            for dep_name in sorted(spec.dependencies.keys()):
                if dep_name in self.installed:
                    deps[dep_name] = build_tree(dep_name, seen)

            return {
                'version': str(pkg.version),
                'deps': deps,
            }

        if root_requirements:
            tree = {}
            for name in sorted(root_requirements.keys()):
                if name in self.installed:
                    tree[name] = build_tree(name)
            return tree

        # Show all top-level packages
        tree = {}
        for name in sorted(self.installed.keys()):
            tree[name] = build_tree(name)
        return tree

    def install_order(self) -> list[str]:
        """Get topological install order for currently installed packages."""
        graph = DependencyGraph(strict=False)
        for name, pkg in self.installed.items():
            spec = self.registry.get(name, pkg.version)
            deps = [d for d in spec.dependencies if d in self.installed]
            graph.add(name, deps if deps else None)
        return graph.resolve()

    def audit(self) -> dict:
        """Audit installed packages for issues."""
        issues = {
            'yanked': [],
            'integrity_mismatch': [],
            'missing_deps': [],
        }

        for name, pkg in self.installed.items():
            # Check if yanked
            spec = self.registry.get(name, pkg.version)
            if spec.yanked:
                issues['yanked'].append(f"{name}@{pkg.version}")

            # Check integrity
            if spec.content_hash() != pkg.integrity:
                issues['integrity_mismatch'].append(name)

            # Check deps are installed
            for dep_name in spec.dependencies:
                if dep_name not in self.installed:
                    issues['missing_deps'].append(f"{name} requires {dep_name}")

        return issues

    def get_install_log(self) -> list[str]:
        """Get the log of install operations from the last install/uninstall."""
        return list(self._install_log)

    def _install_package(self, name: str, version: SemVer, strategy: str) -> None:
        """Install a single package."""
        spec = self.registry.get(name, version)

        if strategy == "flat":
            path = f"{self.install_dir}/{name}"
        else:  # nested
            path = f"{self.install_dir}/{name}/node_modules"

        self.installed[name] = InstalledPackage(
            name=name,
            version=version,
            integrity=spec.content_hash(),
            path=path,
            files=dict(spec.files),
        )
        self._install_log.append(f"installed {name}@{version}")

    def _uninstall_package(self, name: str) -> None:
        """Uninstall a single package."""
        if name in self.installed:
            del self.installed[name]

    def _find_orphans(self, requirements: dict[str, str]) -> list[str]:
        """Find packages that are no longer needed by any requirement."""
        # Resolve current requirements to see what's still needed
        try:
            needed = self.resolver.resolve(requirements)
        except ResolutionError:
            return []

        orphans = []
        for name in list(self.installed.keys()):
            if name not in needed and name not in requirements:
                orphans.append(name)
        return orphans


# ============================================================
# Convenience: create a populated registry for testing
# ============================================================

def create_test_registry() -> Registry:
    """Create a registry with some test packages for demonstration."""
    reg = Registry()

    # Utility library with multiple versions
    for v in ["0.1.0", "0.2.0", "0.2.1", "1.0.0", "1.1.0", "1.2.0", "2.0.0"]:
        reg.publish(PackageSpec(name="utils", version=SemVer.parse(v),
                                description="Utility functions"))

    # Logging library
    for v in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]:
        deps = {"utils": "^1.0.0"} if SemVer.parse(v) >= SemVer.parse("1.1.0") else {}
        reg.publish(PackageSpec(name="logger", version=SemVer.parse(v),
                                dependencies=deps, description="Logging"))

    # HTTP client
    for v in ["1.0.0", "1.1.0", "2.0.0"]:
        reg.publish(PackageSpec(name="http", version=SemVer.parse(v),
                                dependencies={"utils": "^1.0.0", "logger": "^1.0.0"},
                                description="HTTP client"))

    # JSON parser
    for v in ["1.0.0", "2.0.0", "3.0.0"]:
        reg.publish(PackageSpec(name="json-parser", version=SemVer.parse(v),
                                description="JSON parser"))

    # Web framework
    reg.publish(PackageSpec(name="webapp", version=SemVer.parse("1.0.0"),
                            dependencies={"http": "^1.0.0", "json-parser": "^2.0.0"},
                            description="Web framework"))

    return reg


if __name__ == "__main__":
    reg = create_test_registry()
    pm = PackageManager(reg)

    print("Installing webapp...")
    result = pm.install({"webapp": "^1.0.0"})
    print("Resolved:", {k: str(v) for k, v in result.items()})
    print("Install order:", pm.install_order())
    print("Log:", pm.get_install_log())
