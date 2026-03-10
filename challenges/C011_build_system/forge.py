"""
Forge Build System
Challenge C011 -- AgentZero Session 012

A build system that composes:
  - Dependency resolver (C008) for build order via topological sort
  - Stack VM (C010) for evaluating build conditions/guards

Features:
  - Define targets with dependencies, sources, outputs, commands
  - Conditional builds: guard expressions evaluated by the VM
  - File staleness: only rebuild when sources are newer than outputs
  - Parallel build groups: which targets can build simultaneously
  - Dry-run mode: show what would build without executing
  - Build variables: pass configuration into guard expressions
  - Buildfile parser: load target definitions from a declarative format
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

# Import from sibling challenges
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C008_dependency_resolver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from resolver import DependencyGraph, CycleError, MissingDependencyError
from stack_vm import lex, Parser, Compiler, VM


# ============================================================
# Build Target
# ============================================================

@dataclass
class Target:
    """A build target with dependencies, files, and optional guard."""
    name: str
    depends: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    command: Optional[str] = None
    action: Optional[Callable] = None  # Python callable alternative
    guard: Optional[str] = None  # VM expression -- skip if evaluates to false
    phony: bool = False  # Always rebuild (like .PHONY in make)
    description: str = ""

    def __repr__(self):
        return f"Target({self.name!r})"


# ============================================================
# Build Result
# ============================================================

@dataclass
class BuildResult:
    """Result of building a single target."""
    target: str
    status: str  # "built", "skipped", "up-to-date", "failed", "guard-false"
    duration: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in ("built", "skipped", "up-to-date", "guard-false")


@dataclass
class BuildReport:
    """Full report from a build run."""
    results: list[BuildResult] = field(default_factory=list)
    build_order: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    variables: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def built_count(self) -> int:
        return sum(1 for r in self.results if r.status == "built")

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.status in ("skipped", "up-to-date", "guard-false"))

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def summary(self) -> str:
        total = len(self.results)
        return (f"Build: {self.built_count} built, {self.skipped_count} skipped, "
                f"{self.failed_count} failed, {total} total")


# ============================================================
# Staleness Checker
# ============================================================

class StalenessChecker:
    """Determines if a target needs rebuilding based on file modification times."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.getcwd()

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def _mtime(self, path: str) -> Optional[float]:
        resolved = self._resolve(path)
        try:
            return os.path.getmtime(resolved)
        except OSError:
            return None

    def is_stale(self, target: Target) -> bool:
        """Check if a target needs rebuilding.

        A target is stale if:
        - It's phony (always rebuild)
        - Any output file is missing
        - Any source file is newer than any output file
        - It has no outputs (always rebuild)
        - It has no sources and no outputs (always rebuild)
        """
        if target.phony:
            return True

        if not target.outputs:
            return True

        # Check all outputs exist
        output_times = []
        for out in target.outputs:
            mt = self._mtime(out)
            if mt is None:
                return True  # Output missing
            output_times.append(mt)

        if not target.sources:
            # Has outputs but no sources -- up to date if outputs exist
            return False

        oldest_output = min(output_times)

        # Check if any source is newer than oldest output
        for src in target.sources:
            mt = self._mtime(src)
            if mt is None:
                continue  # Missing source -- not our problem to detect here
            if mt > oldest_output:
                return True

        return False


# ============================================================
# Guard Evaluator (uses Stack VM)
# ============================================================

class GuardEvaluator:
    """Evaluates guard expressions using the Stack VM."""

    def __init__(self, variables: Optional[dict] = None):
        self.variables = variables or {}

    def evaluate(self, expression: str) -> bool:
        """Evaluate a guard expression. Returns True if the target should build.

        The expression is wrapped in a program that assigns variables and
        then evaluates the expression, capturing the result via the VM.
        """
        # Build a program that sets variables and evaluates the expression
        lines = []
        for name, value in self.variables.items():
            if isinstance(value, str):
                lines.append(f'let {name} = "{value}";')
            elif isinstance(value, bool):
                lines.append(f'let {name} = {"true" if value else "false"};')
            elif isinstance(value, (int, float)):
                lines.append(f'let {name} = {value};')

        # The expression becomes the last statement; we capture via print
        # and check the VM's output
        lines.append(f'let __guard_result = {expression};')
        lines.append('print(__guard_result);')

        program = '\n'.join(lines)

        try:
            tokens = lex(program)
            parser = Parser(tokens)
            ast = parser.parse()
            compiler = Compiler()
            bytecode = compiler.compile(ast)
            vm = VM(bytecode)
            vm.run()

            # Check what was printed
            if vm.output:
                result = vm.output[-1].strip()
                # The VM prints "True" or "False" for booleans
                if result.lower() == 'true':
                    return True
                elif result.lower() == 'false':
                    return False
                # Numeric: 0 is false, anything else is true
                try:
                    return float(result) != 0
                except ValueError:
                    return bool(result)
            return True  # No output means we can't determine -- default build
        except Exception as e:
            raise GuardError(f"Guard evaluation failed for '{expression}': {e}")


class GuardError(Exception):
    """Raised when a guard expression fails to evaluate."""
    pass


# ============================================================
# Buildfile Parser
# ============================================================

class BuildfileParser:
    """Parses a simple declarative build file format.

    Format:
        var debug = true
        var version = 3

        target "lib" {
            sources ["src/lib.py"]
            outputs ["build/lib.pyc"]
            command "python -m compileall src/lib.py"
            description "Compile the library"
        }

        target "app" {
            depends ["lib", "utils"]
            sources ["src/app.py"]
            outputs ["build/app.pyc"]
            command "python -m compileall src/app.py"
            guard "version >= 3"
        }

        target "clean" {
            phony true
            command "rm -rf build/"
        }
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.targets: list[Target] = []
        self.variables: dict = {}

    def parse(self) -> tuple[list[Target], dict]:
        """Parse the buildfile and return (targets, variables)."""
        while self.pos < len(self.text):
            self._skip_ws_and_comments()
            if self.pos >= len(self.text):
                break
            word = self._peek_word()
            if word == 'var':
                self._parse_var()
            elif word == 'target':
                self._parse_target()
            elif word == '':
                break
            else:
                raise BuildfileError(f"Unexpected token: {word!r} at position {self.pos}")
        return self.targets, self.variables

    def _skip_ws_and_comments(self):
        while self.pos < len(self.text):
            if self.text[self.pos] in ' \t\r\n':
                self.pos += 1
            elif self.text[self.pos] == '#':
                # Skip to end of line
                while self.pos < len(self.text) and self.text[self.pos] != '\n':
                    self.pos += 1
            else:
                break

    def _peek_word(self) -> str:
        i = self.pos
        while i < len(self.text) and self.text[i].isalnum() or (i < len(self.text) and self.text[i] == '_'):
            i += 1
        return self.text[self.pos:i]

    def _read_word(self) -> str:
        word = self._peek_word()
        self.pos += len(word)
        return word

    def _expect(self, char: str):
        self._skip_ws_and_comments()
        if self.pos >= len(self.text) or self.text[self.pos] != char:
            found = self.text[self.pos] if self.pos < len(self.text) else 'EOF'
            raise BuildfileError(f"Expected '{char}', found '{found}' at position {self.pos}")
        self.pos += 1

    def _read_string(self) -> str:
        self._skip_ws_and_comments()
        if self.pos >= len(self.text) or self.text[self.pos] != '"':
            raise BuildfileError(f"Expected string at position {self.pos}")
        self.pos += 1  # skip opening quote
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != '"':
            if self.text[self.pos] == '\\':
                self.pos += 1  # skip escaped char
            self.pos += 1
        if self.pos >= len(self.text):
            raise BuildfileError("Unterminated string")
        result = self.text[start:self.pos]
        self.pos += 1  # skip closing quote
        return result

    def _read_string_list(self) -> list[str]:
        self._skip_ws_and_comments()
        self._expect('[')
        items = []
        self._skip_ws_and_comments()
        while self.pos < len(self.text) and self.text[self.pos] != ']':
            items.append(self._read_string())
            self._skip_ws_and_comments()
            if self.pos < len(self.text) and self.text[self.pos] == ',':
                self.pos += 1
            self._skip_ws_and_comments()
        self._expect(']')
        return items

    def _read_value(self):
        """Read a value: string, number, boolean."""
        self._skip_ws_and_comments()
        if self.pos >= len(self.text):
            raise BuildfileError("Unexpected EOF")
        ch = self.text[self.pos]
        if ch == '"':
            return self._read_string()
        # Check for boolean or number
        word = self._peek_word()
        if word == 'true':
            self.pos += 4
            return True
        elif word == 'false':
            self.pos += 5
            return False
        # Try number
        start = self.pos
        if ch == '-':
            self.pos += 1
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        num_str = self.text[start:self.pos]
        if not num_str or num_str == '-':
            raise BuildfileError(f"Cannot parse value at position {start}")
        if '.' in num_str:
            return float(num_str)
        return int(num_str)

    def _parse_var(self):
        self._read_word()  # consume 'var'
        self._skip_ws_and_comments()
        name = self._read_word()
        self._skip_ws_and_comments()
        self._expect('=')
        value = self._read_value()
        self.variables[name] = value

    def _parse_target(self):
        self._read_word()  # consume 'target'
        self._skip_ws_and_comments()
        name = self._read_string()
        self._skip_ws_and_comments()
        self._expect('{')

        target = Target(name=name)

        while True:
            self._skip_ws_and_comments()
            if self.pos >= len(self.text):
                raise BuildfileError("Unterminated target block")
            if self.text[self.pos] == '}':
                self.pos += 1
                break
            key = self._read_word()
            self._skip_ws_and_comments()
            if key == 'depends':
                target.depends = self._read_string_list()
            elif key == 'sources':
                target.sources = self._read_string_list()
            elif key == 'outputs':
                target.outputs = self._read_string_list()
            elif key == 'command':
                target.command = self._read_string()
            elif key == 'guard':
                target.guard = self._read_string()
            elif key == 'phony':
                target.phony = self._read_value()
            elif key == 'description':
                target.description = self._read_string()
            else:
                raise BuildfileError(f"Unknown target property: {key!r}")

        self.targets.append(target)


class BuildfileError(Exception):
    """Raised when a buildfile has syntax errors."""
    pass


# ============================================================
# Forge Build Engine
# ============================================================

class Forge:
    """The build engine. Composes dependency resolution, guard evaluation,
    staleness checking, and command execution into a build pipeline."""

    def __init__(self, base_dir: Optional[str] = None):
        self.targets: dict[str, Target] = {}
        self.variables: dict = {}
        self.base_dir = base_dir or os.getcwd()
        self._staleness = StalenessChecker(self.base_dir)
        self._log: list[str] = []

    def var(self, name: str, value) -> 'Forge':
        """Set a build variable."""
        self.variables[name] = value
        return self

    def target(self, name: str, **kwargs) -> Target:
        """Add a target programmatically."""
        t = Target(name=name, **kwargs)
        self.targets[name] = t
        return t

    def add_target(self, t: Target) -> None:
        """Add a pre-built Target object."""
        self.targets[t.name] = t

    def load_buildfile(self, path: str) -> None:
        """Load targets and variables from a buildfile."""
        resolved = path
        if not os.path.isabs(path):
            resolved = os.path.join(self.base_dir, path)
        with open(resolved, 'r') as f:
            text = f.read()
        self.load_buildfile_str(text)

    def load_buildfile_str(self, text: str) -> None:
        """Load targets and variables from a buildfile string."""
        parser = BuildfileParser(text)
        targets, variables = parser.parse()
        self.variables.update(variables)
        for t in targets:
            self.targets[t.name] = t

    def _build_graph(self, target_names: Optional[list[str]] = None) -> DependencyGraph:
        """Build dependency graph for requested targets (and their transitive deps)."""
        graph = DependencyGraph(strict=False)

        if target_names is None:
            # Build all targets
            for t in self.targets.values():
                graph.add(t.name, t.depends if t.depends else None)
        else:
            # Build only requested targets and their dependencies
            needed = set()
            stack = list(target_names)
            while stack:
                name = stack.pop()
                if name in needed:
                    continue
                if name not in self.targets:
                    raise BuildError(f"Unknown target: {name!r}")
                needed.add(name)
                stack.extend(self.targets[name].depends)

            for name in needed:
                t = self.targets[name]
                # Only include deps that are also targets
                deps = [d for d in t.depends if d in self.targets]
                graph.add(name, deps if deps else None)

        return graph

    def plan(self, target_names: Optional[list[str]] = None) -> tuple[list[str], list[list[str]]]:
        """Return (build_order, parallel_groups) without building.

        Raises CycleError or BuildError on problems.
        """
        graph = self._build_graph(target_names)
        order = graph.resolve()
        groups = graph.resolve_parallel()
        return order, groups

    def build(self, target_names: Optional[list[str]] = None,
              dry_run: bool = False,
              force: bool = False) -> BuildReport:
        """Execute a build.

        Args:
            target_names: Specific targets to build (None = all)
            dry_run: If True, report what would build without executing
            force: If True, ignore staleness and rebuild everything
        """
        self._log = []
        report = BuildReport(variables=dict(self.variables))

        # Resolve build order
        graph = self._build_graph(target_names)
        report.build_order = graph.resolve()
        report.parallel_groups = graph.resolve_parallel()

        # Guard evaluator with build variables
        guard_eval = GuardEvaluator(self.variables)

        # Build each target in order
        for target_name in report.build_order:
            target = self.targets[target_name]
            result = self._build_target(target, guard_eval, dry_run, force)
            report.results.append(result)

            # Stop on failure
            if not result.ok:
                self._log.append(f"FAILED: {target_name} -- {result.error}")
                break

        return report

    def _build_target(self, target: Target, guard_eval: GuardEvaluator,
                      dry_run: bool, force: bool) -> BuildResult:
        """Build a single target."""
        start = time.time()

        # Check guard condition
        if target.guard:
            try:
                should_build = guard_eval.evaluate(target.guard)
                if not should_build:
                    self._log.append(f"GUARD-FALSE: {target.name} (guard: {target.guard})")
                    return BuildResult(target.name, "guard-false",
                                       duration=time.time() - start)
            except GuardError as e:
                return BuildResult(target.name, "failed",
                                   duration=time.time() - start,
                                   error=str(e))

        # Check staleness
        if not force and not self._staleness.is_stale(target):
            self._log.append(f"UP-TO-DATE: {target.name}")
            return BuildResult(target.name, "up-to-date",
                               duration=time.time() - start)

        # Dry run -- just report
        if dry_run:
            self._log.append(f"WOULD-BUILD: {target.name}")
            return BuildResult(target.name, "skipped",
                               duration=time.time() - start)

        # Execute the build action
        try:
            if target.action:
                target.action()
                self._log.append(f"BUILT: {target.name} (action)")
            elif target.command:
                # Execute shell command
                ret = os.system(target.command)
                if ret != 0:
                    return BuildResult(target.name, "failed",
                                       duration=time.time() - start,
                                       error=f"Command exited with code {ret}")
                self._log.append(f"BUILT: {target.name} (command)")
            else:
                # No action -- it's a grouping target
                self._log.append(f"BUILT: {target.name} (no-op)")
        except Exception as e:
            return BuildResult(target.name, "failed",
                               duration=time.time() - start,
                               error=str(e))

        return BuildResult(target.name, "built", duration=time.time() - start)

    @property
    def log(self) -> list[str]:
        return list(self._log)


class BuildError(Exception):
    """Raised for build configuration errors."""
    pass


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    # Demo: a small project with dependencies and guards
    forge = Forge()
    forge.var("debug", True)
    forge.var("version", 3)

    forge.target("utils", sources=["src/utils.py"], outputs=["build/utils.pyc"],
                 command='echo "Building utils"')
    forge.target("lib", depends=["utils"], sources=["src/lib.py"],
                 outputs=["build/lib.pyc"], command='echo "Building lib"')
    forge.target("app", depends=["lib", "utils"], sources=["src/app.py"],
                 outputs=["build/app.pyc"], command='echo "Building app"',
                 guard="version >= 3")
    forge.target("tests", depends=["app"], phony=True,
                 command='echo "Running tests"',
                 guard="debug == true")
    forge.target("clean", phony=True, command='echo "Cleaning"')

    order, groups = forge.plan()
    print("Build order:", order)
    print("Parallel groups:", groups)

    report = forge.build(dry_run=True)
    print("\nDry run:")
    for r in report.results:
        print(f"  {r.target}: {r.status}")
    print(report.summary())
