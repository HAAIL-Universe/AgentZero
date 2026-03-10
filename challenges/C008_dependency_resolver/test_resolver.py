"""Tests for dependency resolver."""

import unittest
from resolver import DependencyGraph, CycleError, MissingDependencyError


class TestBasicOperations(unittest.TestCase):
    def test_add_package_no_deps(self):
        g = DependencyGraph()
        g.add("a")
        assert "a" in g
        assert len(g) == 1

    def test_add_package_with_deps(self):
        g = DependencyGraph(strict=False)
        g.add("b", ["a"])
        assert "b" in g
        assert "a" in g  # auto-added

    def test_strict_missing_dep_raises(self):
        g = DependencyGraph(strict=True)
        with self.assertRaises(MissingDependencyError) as ctx:
            g.add("b", ["a"])
        assert ctx.exception.package == "b"
        assert ctx.exception.missing == "a"

    def test_remove_package(self):
        g = DependencyGraph(strict=False)
        g.add("a")
        g.add("b", ["a"])
        g.remove("b")
        assert "b" not in g
        assert "a" in g

    def test_remove_cleans_deps(self):
        g = DependencyGraph(strict=False)
        g.add("a")
        g.add("b", ["a"])
        g.remove("a")
        # b should have empty deps now
        assert g.dependencies_of("b") == set()

    def test_remove_nonexistent_raises(self):
        g = DependencyGraph()
        with self.assertRaises(KeyError):
            g.remove("nope")

    def test_packages_sorted(self):
        g = DependencyGraph()
        g.add("c")
        g.add("a")
        g.add("b")
        assert g.packages == ["a", "b", "c"]


class TestQueries(unittest.TestCase):
    def setUp(self):
        self.g = DependencyGraph(strict=False)
        self.g.add("app", ["framework", "database"])
        self.g.add("framework", ["utils", "logging"])
        self.g.add("database", ["utils"])
        self.g.add("utils")
        self.g.add("logging")

    def test_dependencies_of(self):
        assert self.g.dependencies_of("app") == {"framework", "database"}

    def test_dependencies_of_leaf(self):
        assert self.g.dependencies_of("utils") == set()

    def test_dependents_of(self):
        assert self.g.dependents_of("utils") == {"framework", "database"}

    def test_dependents_of_root(self):
        assert self.g.dependents_of("app") == set()

    def test_all_dependencies_transitive(self):
        all_deps = self.g.all_dependencies_of("app")
        assert all_deps == {"framework", "database", "utils", "logging"}

    def test_all_dependencies_of_leaf(self):
        assert self.g.all_dependencies_of("utils") == set()

    def test_query_nonexistent_raises(self):
        with self.assertRaises(KeyError):
            self.g.dependencies_of("nope")


class TestResolve(unittest.TestCase):
    def test_single_package(self):
        g = DependencyGraph()
        g.add("a")
        assert g.resolve() == ["a"]

    def test_linear_chain(self):
        g = DependencyGraph(strict=False)
        g.add("c", ["b"])
        g.add("b", ["a"])
        g.add("a")
        order = g.resolve()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond(self):
        g = DependencyGraph(strict=False)
        g.add("d", ["b", "c"])
        g.add("b", ["a"])
        g.add("c", ["a"])
        g.add("a")
        order = g.resolve()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_complex_graph(self):
        g = DependencyGraph(strict=False)
        g.add("app", ["framework", "database"])
        g.add("framework", ["utils", "logging"])
        g.add("database", ["utils"])
        g.add("utils")
        g.add("logging")
        order = g.resolve()
        # Verify all deps come before their dependents
        for pkg in ["app", "framework", "database"]:
            for dep in g.dependencies_of(pkg):
                assert order.index(dep) < order.index(pkg), \
                    f"{dep} should come before {pkg}"

    def test_no_packages(self):
        g = DependencyGraph()
        assert g.resolve() == []

    def test_independent_packages(self):
        g = DependencyGraph()
        g.add("a")
        g.add("b")
        g.add("c")
        order = g.resolve()
        assert sorted(order) == ["a", "b", "c"]

    def test_deterministic_output(self):
        """Same graph resolves the same way every time."""
        g = DependencyGraph(strict=False)
        g.add("c", ["a"])
        g.add("b", ["a"])
        g.add("a")
        order1 = g.resolve()
        order2 = g.resolve()
        assert order1 == order2


class TestCycleDetection(unittest.TestCase):
    def test_simple_cycle(self):
        g = DependencyGraph(strict=False)
        g.add("a", ["b"])
        g.add("b", ["a"])
        with self.assertRaises(CycleError) as ctx:
            g.resolve()
        assert len(ctx.exception.cycle) >= 3  # at least a -> b -> a

    def test_self_dependency(self):
        g = DependencyGraph(strict=False)
        g.add("a", ["a"])
        with self.assertRaises(CycleError):
            g.resolve()

    def test_three_node_cycle(self):
        g = DependencyGraph(strict=False)
        g.add("a", ["c"])
        g.add("b", ["a"])
        g.add("c", ["b"])
        with self.assertRaises(CycleError):
            g.resolve()

    def test_cycle_with_non_cyclic_parts(self):
        g = DependencyGraph(strict=False)
        g.add("a")
        g.add("b", ["a"])
        g.add("c", ["d"])  # c and d form a cycle
        g.add("d", ["c"])
        with self.assertRaises(CycleError):
            g.resolve()

    def test_cycle_error_contains_cycle(self):
        g = DependencyGraph(strict=False)
        g.add("a", ["b"])
        g.add("b", ["a"])
        try:
            g.resolve()
            assert False, "Should have raised"
        except CycleError as e:
            # Cycle should start and end with the same node
            assert e.cycle[0] == e.cycle[-1]


class TestParallelResolution(unittest.TestCase):
    def test_all_independent(self):
        g = DependencyGraph()
        g.add("a")
        g.add("b")
        g.add("c")
        groups = g.resolve_parallel()
        assert len(groups) == 1
        assert sorted(groups[0]) == ["a", "b", "c"]

    def test_linear_chain(self):
        g = DependencyGraph(strict=False)
        g.add("c", ["b"])
        g.add("b", ["a"])
        g.add("a")
        groups = g.resolve_parallel()
        assert groups == [["a"], ["b"], ["c"]]

    def test_diamond_two_groups(self):
        g = DependencyGraph(strict=False)
        g.add("d", ["b", "c"])
        g.add("b", ["a"])
        g.add("c", ["a"])
        g.add("a")
        groups = g.resolve_parallel()
        assert groups[0] == ["a"]
        assert sorted(groups[1]) == ["b", "c"]
        assert groups[2] == ["d"]

    def test_complex_parallel(self):
        g = DependencyGraph(strict=False)
        g.add("app", ["framework", "database"])
        g.add("framework", ["utils", "logging"])
        g.add("database", ["utils"])
        g.add("utils")
        g.add("logging")
        groups = g.resolve_parallel()
        # Group 0: logging, utils (no deps)
        assert sorted(groups[0]) == ["logging", "utils"]
        # Group 1: database, framework (deps satisfied)
        assert sorted(groups[1]) == ["database", "framework"]
        # Group 2: app
        assert groups[2] == ["app"]

    def test_parallel_cycle_raises(self):
        g = DependencyGraph(strict=False)
        g.add("a", ["b"])
        g.add("b", ["a"])
        with self.assertRaises(CycleError):
            g.resolve_parallel()

    def test_empty_graph(self):
        g = DependencyGraph()
        assert g.resolve_parallel() == []


class TestLargeGraph(unittest.TestCase):
    def test_hundred_node_chain(self):
        g = DependencyGraph(strict=False)
        for i in range(100):
            if i == 0:
                g.add(f"pkg_{i:03d}")
            else:
                g.add(f"pkg_{i:03d}", [f"pkg_{i-1:03d}"])
        order = g.resolve()
        assert len(order) == 100
        assert order[0] == "pkg_000"
        assert order[-1] == "pkg_099"

    def test_wide_graph(self):
        """100 packages all depending on one root."""
        g = DependencyGraph(strict=False)
        g.add("root")
        for i in range(100):
            g.add(f"leaf_{i:03d}", ["root"])
        order = g.resolve()
        assert order[0] == "root"
        assert len(order) == 101


if __name__ == "__main__":
    unittest.main()
