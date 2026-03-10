"""
Dependency Resolver

Resolves dependency graphs using topological sort (Kahn's algorithm).
Detects cycles, missing dependencies, and provides install order.

Features:
- Add packages with dependencies
- Resolve full install order (topological sort)
- Detect circular dependencies (reports the cycle)
- Detect missing dependencies
- Query: what depends on X? What does X need?
- Parallel groups: which packages can install simultaneously?
"""

from collections import defaultdict, deque
from typing import Optional


class CycleError(Exception):
    """Raised when a circular dependency is detected."""
    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        path = " -> ".join(cycle)
        super().__init__(f"Circular dependency: {path}")


class MissingDependencyError(Exception):
    """Raised when a package depends on something not in the graph."""
    def __init__(self, package: str, missing: str):
        self.package = package
        self.missing = missing
        super().__init__(f"Package '{package}' depends on '{missing}' which is not defined")


class DependencyGraph:
    def __init__(self, strict: bool = True):
        """Create a dependency graph.

        Args:
            strict: If True, raise MissingDependencyError for undefined deps.
                    If False, auto-add missing deps as packages with no deps.
        """
        self._deps: dict[str, set[str]] = {}
        self._strict = strict

    def add(self, package: str, depends_on: Optional[list[str]] = None) -> None:
        """Add a package with its dependencies."""
        if package not in self._deps:
            self._deps[package] = set()
        if depends_on:
            for dep in depends_on:
                self._deps[package].add(dep)
                if dep not in self._deps:
                    if self._strict:
                        raise MissingDependencyError(package, dep)
                    self._deps[dep] = set()

    def remove(self, package: str) -> None:
        """Remove a package. Raises KeyError if not found."""
        if package not in self._deps:
            raise KeyError(f"Package '{package}' not found")
        del self._deps[package]
        # Remove from others' dependency lists
        for deps in self._deps.values():
            deps.discard(package)

    @property
    def packages(self) -> list[str]:
        """All packages in alphabetical order."""
        return sorted(self._deps.keys())

    def dependencies_of(self, package: str) -> set[str]:
        """Direct dependencies of a package."""
        if package not in self._deps:
            raise KeyError(f"Package '{package}' not found")
        return set(self._deps[package])

    def dependents_of(self, package: str) -> set[str]:
        """Packages that directly depend on this package."""
        if package not in self._deps:
            raise KeyError(f"Package '{package}' not found")
        return {pkg for pkg, deps in self._deps.items() if package in deps}

    def all_dependencies_of(self, package: str) -> set[str]:
        """All transitive dependencies of a package (not including itself)."""
        if package not in self._deps:
            raise KeyError(f"Package '{package}' not found")
        result = set()
        stack = list(self._deps[package])
        while stack:
            dep = stack.pop()
            if dep not in result:
                result.add(dep)
                if dep in self._deps:
                    stack.extend(self._deps[dep] - result)
        return result

    def resolve(self) -> list[str]:
        """Return install order using Kahn's algorithm (topological sort).

        Raises CycleError if circular dependencies exist.
        Returns packages in an order where each package comes after all its deps.
        """
        # Build in-degree map
        in_degree = {pkg: 0 for pkg in self._deps}
        for pkg, deps in self._deps.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[pkg] = in_degree.get(pkg, 0)
                    # dep needs to be installed first; pkg has an incoming edge from dep
                    pass

        # Recompute properly
        in_degree = {pkg: 0 for pkg in self._deps}
        for pkg, deps in self._deps.items():
            for dep in deps:
                # pkg depends on dep means there's an edge dep -> pkg
                # so pkg's in-degree should count... no wait.
                # In Kahn's: edge from dep -> pkg means pkg can't install until dep is done
                # in_degree counts incoming edges TO a node
                # The edge is dep -> pkg, so in_degree[pkg] += 1
                pass

        # Clean version
        in_degree = defaultdict(int)
        for pkg in self._deps:
            if pkg not in in_degree:
                in_degree[pkg] = 0
            for dep in self._deps[pkg]:
                # Edge: dep -> pkg (dep must come before pkg)
                # This means pkg has an incoming edge
                in_degree[pkg] += 1

        # Start with nodes that have no incoming edges (no dependencies)
        queue = deque(sorted(pkg for pkg, deg in in_degree.items() if deg == 0))
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            # For each package that depends on this node, reduce in-degree
            for pkg, deps in self._deps.items():
                if node in deps:
                    in_degree[pkg] -= 1
                    if in_degree[pkg] == 0:
                        queue.append(pkg)
            # Keep queue sorted for deterministic output
            queue = deque(sorted(queue))

        if len(result) != len(self._deps):
            # Cycle exists -- find it
            cycle = self._find_cycle()
            raise CycleError(cycle)

        return result

    def resolve_parallel(self) -> list[list[str]]:
        """Return groups of packages that can be installed in parallel.

        Each group contains packages whose dependencies are all in earlier groups.
        Raises CycleError if circular dependencies exist.
        """
        remaining = {pkg: set(deps) for pkg, deps in self._deps.items()}
        groups = []

        while remaining:
            # Find all packages with no remaining deps
            ready = sorted(pkg for pkg, deps in remaining.items() if len(deps) == 0)
            if not ready:
                cycle = self._find_cycle()
                raise CycleError(cycle)
            groups.append(ready)
            # Remove installed packages from remaining
            for pkg in ready:
                del remaining[pkg]
            for deps in remaining.values():
                deps -= set(ready)

        return groups

    def _find_cycle(self) -> list[str]:
        """Find and return a cycle in the graph using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {pkg: WHITE for pkg in self._deps}
        parent = {}

        def dfs(node):
            color[node] = GRAY
            for dep in sorted(self._deps.get(node, set())):
                if color.get(dep) == GRAY:
                    # Found cycle -- reconstruct
                    if dep == node:
                        return [dep, dep]  # self-loop
                    cycle = [dep, node]
                    current = node
                    while parent.get(current) != dep:
                        current = parent[current]
                        cycle.append(current)
                    cycle.reverse()
                    cycle.append(cycle[0])  # close the cycle
                    return cycle
                if color.get(dep) == WHITE:
                    parent[dep] = node
                    result = dfs(dep)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for pkg in sorted(self._deps):
            if color[pkg] == WHITE:
                result = dfs(pkg)
                if result:
                    return result
        return []

    def __len__(self):
        return len(self._deps)

    def __contains__(self, package: str):
        return package in self._deps

    def __repr__(self):
        return f"DependencyGraph({len(self._deps)} packages)"


if __name__ == "__main__":
    g = DependencyGraph(strict=False)
    g.add("app", ["framework", "database"])
    g.add("framework", ["utils", "logging"])
    g.add("database", ["utils"])
    g.add("utils")
    g.add("logging")

    print("Install order:", g.resolve())
    print("Parallel groups:", g.resolve_parallel())
    print("app needs:", g.all_dependencies_of("app"))
    print("utils used by:", g.dependents_of("utils"))
