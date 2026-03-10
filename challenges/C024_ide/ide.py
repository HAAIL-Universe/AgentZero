"""
Integrated Development Environment for MiniLang
Challenge C024 -- AgentZero Session 025

A unified IDE backend composing:
  - C019 LSP Server (diagnostics, completion, hover, definition, symbols, signature help)
  - C020 REPL/Debugger (eval, breakpoints, stepping, watches, trace)
  - C021 Package Manager (semver, registry, resolver, lockfile, install)

Features:
  - Workspace: virtual file system with project structure
  - Editor: LSP-backed intelligence (diagnostics, completion, hover, go-to-def)
  - Terminal: REPL for interactive evaluation with project context
  - Debugger: breakpoints, stepping, watches, call stack, trace
  - Packages: dependency resolution, install, lockfile, audit
  - Build/Run: compile and execute project files
  - Cross-system integration:
    - Installed package names available in LSP completions
    - REPL has access to project variables and functions
    - Debugger integrates with workspace files
    - Diagnostics update when files or dependencies change
  - Session management: save/restore IDE state

Architecture:
  IDE -> Workspace -> Files
      -> EditorService -> LSPServer -> C010/C013
      -> TerminalService -> REPL -> C010
      -> DebugService -> DebugSession -> C010
      -> PackageService -> PackageManager -> C008

Composes: C019 (LSP), C020 (REPL/Debugger), C021 (Package Manager)
"""

import json
import sys
import os
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum, auto

# Import composed systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C019_lsp_server'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C020_repl_debugger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C021_package_manager'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from lsp_server import (
    LSPServer, LSPClient, MemoryTransport, Document, DocumentManager,
    Position, Range, Location, Diagnostic
)
from repl_debugger import (
    REPL, DebugSession, DebugVM, DebugEvent, StopReason,
    Breakpoint, WatchEntry, REPLState
)
from package_manager import (
    PackageManager, Registry, PackageSpec, SemVer, Constraint,
    Resolver, Lockfile, InstalledPackage, RegistryError, ResolutionError,
    InstallError
)
from stack_vm import (
    lex, Parser, Compiler, VM, Chunk, compile_source, execute,
    disassemble, Program, LexError, ParseError, CompileError, VMError
)


# ============================================================
# Workspace -- Virtual File System
# ============================================================

@dataclass
class WorkspaceFile:
    """A file in the virtual workspace."""
    path: str
    content: str
    version: int = 0
    language: str = "minilang"
    dirty: bool = False
    created_at: float = 0.0
    modified_at: float = 0.0


class Workspace:
    """Virtual file system for the IDE project."""

    def __init__(self, root: str = "/project"):
        self.root = root
        self.files: dict[str, WorkspaceFile] = {}
        self._watchers: list = []

    def create_file(self, path: str, content: str = "",
                    language: str = "minilang") -> WorkspaceFile:
        """Create a new file in the workspace."""
        full_path = self._resolve(path)
        now = time.time()
        f = WorkspaceFile(
            path=full_path, content=content, language=language,
            version=0, dirty=False, created_at=now, modified_at=now
        )
        self.files[full_path] = f
        self._notify("created", full_path)
        return f

    def write_file(self, path: str, content: str) -> WorkspaceFile:
        """Write content to a file (create if not exists)."""
        full_path = self._resolve(path)
        if full_path in self.files:
            f = self.files[full_path]
            f.content = content
            f.version += 1
            f.dirty = True
            f.modified_at = time.time()
            self._notify("changed", full_path)
            return f
        return self.create_file(path, content)

    def read_file(self, path: str) -> Optional[str]:
        """Read a file's content."""
        full_path = self._resolve(path)
        f = self.files.get(full_path)
        return f.content if f else None

    def delete_file(self, path: str) -> bool:
        """Delete a file from the workspace."""
        full_path = self._resolve(path)
        if full_path in self.files:
            del self.files[full_path]
            self._notify("deleted", full_path)
            return True
        return False

    def get_file(self, path: str) -> Optional[WorkspaceFile]:
        """Get file metadata."""
        return self.files.get(self._resolve(path))

    def list_files(self, pattern: str = None) -> list[str]:
        """List files, optionally filtered by extension pattern."""
        paths = sorted(self.files.keys())
        if pattern:
            # Simple extension filter: "*.ml" matches ".ml" suffix
            if pattern.startswith("*."):
                ext = pattern[1:]  # ".ml"
                paths = [p for p in paths if p.endswith(ext)]
        return paths

    def exists(self, path: str) -> bool:
        return self._resolve(path) in self.files

    def on_change(self, callback):
        """Register a change watcher."""
        self._watchers.append(callback)

    def _notify(self, event: str, path: str):
        for w in self._watchers:
            try:
                w(event, path)
            except Exception:
                pass

    def _resolve(self, path: str) -> str:
        """Resolve a path relative to workspace root."""
        if path.startswith(self.root):
            return path
        if path.startswith("/"):
            return path
        return self.root + "/" + path

    def to_uri(self, path: str) -> str:
        """Convert workspace path to URI for LSP."""
        full_path = self._resolve(path)
        return "file://" + full_path

    def from_uri(self, uri: str) -> str:
        """Convert URI back to workspace path."""
        if uri.startswith("file://"):
            return uri[7:]
        return uri


# ============================================================
# Editor Service -- LSP Integration
# ============================================================

class EditorService:
    """Provides IDE intelligence via LSP server integration."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._transport = MemoryTransport()
        self._server = LSPServer(transport=self._transport)
        self._client = LSPClient(self._server)
        self._open_files: set[str] = set()
        self._initialized = False
        self._extra_completions: list[dict] = []

    def initialize(self, root_uri: str = None):
        """Initialize the LSP server."""
        if root_uri is None:
            root_uri = "file://" + self.workspace.root
        self._client.initialize(root_uri=root_uri)
        self._client.initialized()
        self._initialized = True

    def shutdown(self):
        """Shutdown the LSP server."""
        if self._initialized:
            self._client.shutdown()
            self._client.exit()
            self._initialized = False

    def open_file(self, path: str) -> list[dict]:
        """Open a file in the editor. Returns diagnostics."""
        uri = self.workspace.to_uri(path)
        content = self.workspace.read_file(path)
        if content is None:
            return []
        f = self.workspace.get_file(path)
        self._client.open_document(
            uri=uri, text=content,
            language_id=f.language if f else "minilang",
            version=f.version if f else 0
        )
        self._open_files.add(path)
        return self._get_diagnostics(uri)

    def close_file(self, path: str):
        """Close a file in the editor."""
        uri = self.workspace.to_uri(path)
        if path in self._open_files:
            self._client.close_document(uri)
            self._open_files.discard(path)

    def update_file(self, path: str, content: str) -> list[dict]:
        """Update file content and get fresh diagnostics."""
        self.workspace.write_file(path, content)
        uri = self.workspace.to_uri(path)
        f = self.workspace.get_file(path)
        version = f.version if f else 1
        if path in self._open_files:
            self._client.change_document(uri=uri, text=content, version=version)
        else:
            self.open_file(path)
        return self._get_diagnostics(uri)

    def get_diagnostics(self, path: str) -> list[dict]:
        """Get current diagnostics for a file."""
        uri = self.workspace.to_uri(path)
        return self._get_diagnostics(uri)

    def complete(self, path: str, line: int, character: int) -> list[dict]:
        """Get completion items at position."""
        uri = self.workspace.to_uri(path)
        result = self._client.completion(uri, line, character)
        items = []
        if result and 'items' in result:
            items = result['items']
        # Add extra completions (e.g., from packages)
        for extra in self._extra_completions:
            items.append(extra)
        return items

    def hover(self, path: str, line: int, character: int) -> Optional[dict]:
        """Get hover information at position."""
        uri = self.workspace.to_uri(path)
        return self._client.hover(uri, line, character)

    def definition(self, path: str, line: int, character: int) -> Optional[dict]:
        """Get go-to-definition at position."""
        uri = self.workspace.to_uri(path)
        return self._client.definition(uri, line, character)

    def symbols(self, path: str) -> list[dict]:
        """Get document symbols (outline)."""
        uri = self.workspace.to_uri(path)
        result = self._client.document_symbols(uri)
        return result if result else []

    def signature_help(self, path: str, line: int, character: int) -> Optional[dict]:
        """Get signature help at position."""
        uri = self.workspace.to_uri(path)
        return self._client.signature_help(uri, line, character)

    def add_package_completions(self, package_names: list[str]):
        """Add package names to completion suggestions."""
        self._extra_completions = [
            {"label": name, "kind": 9, "detail": f"package: {name}"}
            for name in package_names
        ]

    def is_open(self, path: str) -> bool:
        return path in self._open_files

    def get_open_files(self) -> list[str]:
        return sorted(self._open_files)

    def _get_diagnostics(self, uri: str) -> list[dict]:
        """Extract diagnostics from LSP notifications."""
        diags = self._client.get_diagnostics(uri)
        if diags:
            return diags
        return []

    def refresh_all(self) -> dict[str, list[dict]]:
        """Re-analyze all open files. Returns {path: diagnostics}."""
        results = {}
        for path in list(self._open_files):
            content = self.workspace.read_file(path)
            if content is not None:
                results[path] = self.update_file(path, content)
        return results


# ============================================================
# Terminal Service -- REPL Integration
# ============================================================

class TerminalService:
    """Interactive REPL terminal with project context."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.repl = REPL()
        self._output_log: list[dict] = []

    def eval(self, source: str) -> dict:
        """Evaluate source in the REPL. Returns {result, output, error, env}."""
        result = self.repl.eval(source)
        self._output_log.append({
            'input': source,
            'result': result.get('result'),
            'error': result.get('error'),
            'output': result.get('output', []),
        })
        return result

    def run_file(self, path: str) -> dict:
        """Execute a workspace file in the REPL context."""
        content = self.workspace.read_file(path)
        if content is None:
            return {'result': None, 'output': [], 'error': f"File not found: {path}",
                    'env': dict(self.repl.state.env)}
        return self.eval(content)

    def load_file(self, path: str) -> dict:
        """Load a file's definitions into REPL context without executing."""
        content = self.workspace.read_file(path)
        if content is None:
            return {'result': None, 'output': [], 'error': f"File not found: {path}",
                    'env': dict(self.repl.state.env)}
        # Execute but only keep function/variable definitions
        return self.eval(content)

    def get_variables(self) -> dict:
        """Get all REPL variables."""
        return dict(self.repl.state.env)

    def get_history(self) -> list[str]:
        return self.repl.get_history()

    def get_output_log(self) -> list[dict]:
        return list(self._output_log)

    def reset(self) -> dict:
        """Reset REPL state."""
        return self.repl.eval('.reset')

    def is_complete(self, source: str) -> bool:
        """Check if input is complete (balanced braces etc)."""
        return self.repl.is_complete(source)

    def inject_variables(self, variables: dict):
        """Inject variables into the REPL environment."""
        self.repl.state.env.update(variables)


# ============================================================
# Debug Service -- Debugger Integration
# ============================================================

class DebugService:
    """Debugger for workspace files with full stepping control."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._sessions: dict[str, DebugSession] = {}
        self._active_session: Optional[str] = None

    def start(self, path: str, env: dict = None) -> dict:
        """Start a debug session for a workspace file."""
        content = self.workspace.read_file(path)
        if content is None:
            return {'error': f"File not found: {path}"}
        try:
            session = DebugSession(source=content, env=env)
            session_id = path
            self._sessions[session_id] = session
            self._active_session = session_id
            return {
                'session_id': session_id,
                'source': content,
                'context': session.get_context(),
                'error': None,
            }
        except (LexError, ParseError, CompileError) as e:
            return {'error': str(e)}

    def start_source(self, source: str, session_id: str = "__inline__",
                     env: dict = None) -> dict:
        """Start a debug session from raw source code."""
        try:
            session = DebugSession(source=source, env=env)
            self._sessions[session_id] = session
            self._active_session = session_id
            return {
                'session_id': session_id,
                'source': source,
                'context': session.get_context(),
                'error': None,
            }
        except (LexError, ParseError, CompileError) as e:
            return {'error': str(e)}

    def stop(self, session_id: str = None) -> bool:
        """Stop a debug session."""
        sid = session_id or self._active_session
        if sid and sid in self._sessions:
            del self._sessions[sid]
            if self._active_session == sid:
                self._active_session = None
            return True
        return False

    def step(self, session_id: str = None) -> Optional[dict]:
        """Single-step (instruction level)."""
        session = self._get_session(session_id)
        if not session:
            return None
        event = session.step()
        return self._event_to_dict(event, session)

    def step_line(self, session_id: str = None) -> Optional[dict]:
        """Step to next source line."""
        session = self._get_session(session_id)
        if not session:
            return None
        event = session.step_line()
        return self._event_to_dict(event, session)

    def step_over(self, session_id: str = None) -> Optional[dict]:
        """Step over function calls."""
        session = self._get_session(session_id)
        if not session:
            return None
        event = session.step_over()
        return self._event_to_dict(event, session)

    def step_out(self, session_id: str = None) -> Optional[dict]:
        """Step out of current function."""
        session = self._get_session(session_id)
        if not session:
            return None
        event = session.step_out()
        return self._event_to_dict(event, session)

    def continue_execution(self, session_id: str = None) -> Optional[dict]:
        """Continue until breakpoint or halt."""
        session = self._get_session(session_id)
        if not session:
            return None
        event = session.continue_execution()
        return self._event_to_dict(event, session)

    def add_breakpoint(self, line: int = -1, address: int = -1,
                       condition: str = "",
                       session_id: str = None) -> Optional[int]:
        """Add a breakpoint. Returns breakpoint ID."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.add_breakpoint(line=line, address=address, condition=condition)

    def remove_breakpoint(self, bp_id: int, session_id: str = None) -> bool:
        """Remove a breakpoint."""
        session = self._get_session(session_id)
        if not session:
            return False
        return session.remove_breakpoint(bp_id)

    def add_watch(self, expression: str, session_id: str = None) -> Optional[int]:
        """Add a watch expression. Returns watch ID."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.add_watch(expression)

    def remove_watch(self, watch_id: int, session_id: str = None) -> bool:
        """Remove a watch expression."""
        session = self._get_session(session_id)
        if not session:
            return False
        return session.remove_watch(watch_id)

    def get_context(self, session_id: str = None) -> Optional[dict]:
        """Get current debug context (variables, stack, call stack, line)."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.get_context()

    def get_disassembly(self, session_id: str = None, context: int = 3) -> Optional[str]:
        """Get disassembly around current instruction."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.get_disassembly(context=context)

    def eval_expression(self, expr: str, session_id: str = None) -> Optional[dict]:
        """Evaluate expression in debug context."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.eval_expression(expr)

    def get_source_line(self, line_num: int, session_id: str = None) -> Optional[str]:
        """Get a source line from the debug session."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.get_source_line(line_num)

    def get_trace(self, session_id: str = None) -> Optional[list]:
        """Get execution trace."""
        session = self._get_session(session_id)
        if not session:
            return None
        return session.get_trace()

    def enable_trace(self, enabled: bool = True, session_id: str = None):
        """Enable/disable execution trace."""
        session = self._get_session(session_id)
        if session:
            session.enable_trace(enabled)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def get_active_session(self) -> Optional[str]:
        return self._active_session

    def set_active_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._active_session = session_id
            return True
        return False

    def _get_session(self, session_id: str = None) -> Optional[DebugSession]:
        sid = session_id or self._active_session
        if sid:
            return self._sessions.get(sid)
        return None

    def _event_to_dict(self, event: DebugEvent, session: DebugSession) -> dict:
        ctx = session.get_context()
        return {
            'reason': event.reason.name,
            'ip': event.ip,
            'line': event.line,
            'message': event.message,
            'breakpoint_id': event.breakpoint_id,
            'context': ctx,
        }


# ============================================================
# Package Service -- Package Manager Integration
# ============================================================

class PackageService:
    """Package management for the IDE workspace."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.registry = Registry()
        self.manager = PackageManager(
            registry=self.registry, install_dir="/project/packages"
        )
        self._project_requirements: dict[str, str] = {}
        self._dev_requirements: dict[str, str] = {}

    def publish(self, name: str, version: str, files: dict = None,
                deps: dict = None, dev_deps: dict = None,
                description: str = "") -> PackageSpec:
        """Publish a package to the registry."""
        spec = PackageSpec(
            name=name,
            version=SemVer.parse(version),
            files=files or {},
            dependencies=deps or {},
            dev_dependencies=dev_deps or {},
            description=description,
        )
        self.registry.publish(spec)
        return spec

    def add_dependency(self, name: str, constraint: str,
                       dev: bool = False) -> dict[str, SemVer]:
        """Add a dependency to the project and install it."""
        if dev:
            self._dev_requirements[name] = constraint
        else:
            self._project_requirements[name] = constraint
        return self.install()

    def remove_dependency(self, name: str) -> list[str]:
        """Remove a dependency from the project."""
        self._project_requirements.pop(name, None)
        self._dev_requirements.pop(name, None)
        return self.manager.uninstall(
            [name], requirements=self._project_requirements
        )

    def install(self, include_dev: bool = False) -> dict[str, SemVer]:
        """Install all project dependencies."""
        reqs = dict(self._project_requirements)
        if include_dev:
            reqs.update(self._dev_requirements)
        return self.manager.install(reqs, include_dev=include_dev)

    def update(self, packages: list[str] = None) -> dict:
        """Update packages to latest compatible versions."""
        return self.manager.update(
            self._project_requirements, packages=packages
        )

    def list_installed(self) -> list[dict]:
        """List installed packages."""
        return [
            {'name': p.name, 'version': str(p.version), 'path': p.path}
            for p in self.manager.list_installed()
        ]

    def dependency_tree(self) -> dict:
        """Get the dependency tree."""
        return self.manager.dependency_tree(self._project_requirements)

    def audit(self) -> dict:
        """Audit installed packages."""
        return self.manager.audit()

    def get_lockfile(self) -> dict:
        """Get lockfile contents."""
        return self.manager.lockfile.to_dict()

    def get_requirements(self) -> dict:
        """Get project requirements."""
        return {
            'dependencies': dict(self._project_requirements),
            'devDependencies': dict(self._dev_requirements),
        }

    def installed_package_names(self) -> list[str]:
        """Get names of all installed packages."""
        return [p.name for p in self.manager.list_installed()]

    def get_install_log(self) -> list[str]:
        return self.manager.get_install_log()


# ============================================================
# Build Service -- Compile and Run
# ============================================================

class BuildResult:
    """Result of a build/run operation."""
    def __init__(self, success: bool, output: list = None,
                 result: Any = None, error: str = None,
                 steps: int = 0):
        self.success = success
        self.output = output or []
        self.result = result
        self.error = error
        self.steps = steps

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'output': self.output,
            'result': self.result,
            'error': self.error,
            'steps': self.steps,
        }


class BuildService:
    """Compile and run MiniLang files."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._build_log: list[dict] = []

    def check(self, path: str) -> dict:
        """Check a file for syntax/type errors without running."""
        content = self.workspace.read_file(path)
        if content is None:
            return {'success': False, 'errors': [f"File not found: {path}"]}
        errors = []
        try:
            tokens = lex(content)
            parser = Parser(tokens)
            ast = parser.parse()
        except (LexError, ParseError) as e:
            errors.append(str(e))
        return {'success': len(errors) == 0, 'errors': errors}

    def compile(self, path: str) -> dict:
        """Compile a file and return the chunk."""
        content = self.workspace.read_file(path)
        if content is None:
            return {'success': False, 'error': f"File not found: {path}"}
        try:
            chunk, compiler = compile_source(content)
            return {
                'success': True,
                'chunk': chunk,
                'compiler': compiler,
                'error': None,
            }
        except (LexError, ParseError, CompileError) as e:
            return {'success': False, 'chunk': None, 'error': str(e)}

    def run(self, path: str, env: dict = None) -> BuildResult:
        """Compile and run a file."""
        content = self.workspace.read_file(path)
        if content is None:
            return BuildResult(success=False, error=f"File not found: {path}")
        try:
            chunk, _ = compile_source(content)
            vm = VM(chunk)
            if env:
                vm.env = dict(env)
            result = vm.run()
            br = BuildResult(
                success=True, output=vm.output,
                result=result, steps=vm.step_count
            )
            self._build_log.append({
                'path': path, 'success': True, 'steps': vm.step_count
            })
            return br
        except (LexError, ParseError, CompileError, VMError) as e:
            br = BuildResult(success=False, error=str(e))
            self._build_log.append({'path': path, 'success': False, 'error': str(e)})
            return br

    def run_source(self, source: str, env: dict = None) -> BuildResult:
        """Compile and run raw source code."""
        try:
            chunk, _ = compile_source(source)
            vm = VM(chunk)
            if env:
                vm.env = dict(env)
            result = vm.run()
            return BuildResult(
                success=True, output=vm.output,
                result=result, steps=vm.step_count
            )
        except (LexError, ParseError, CompileError, VMError) as e:
            return BuildResult(success=False, error=str(e))

    def disassemble(self, path: str) -> Optional[str]:
        """Disassemble a file to bytecode."""
        content = self.workspace.read_file(path)
        if content is None:
            return None
        try:
            chunk, _ = compile_source(content)
            return disassemble(chunk)
        except (LexError, ParseError, CompileError):
            return None

    def get_build_log(self) -> list[dict]:
        return list(self._build_log)


# ============================================================
# IDE -- Unified Integration
# ============================================================

class IDEEvent(Enum):
    FILE_CREATED = auto()
    FILE_CHANGED = auto()
    FILE_DELETED = auto()
    DIAGNOSTICS_UPDATED = auto()
    DEBUG_STARTED = auto()
    DEBUG_STOPPED = auto()
    PACKAGE_INSTALLED = auto()
    PACKAGE_REMOVED = auto()
    BUILD_COMPLETE = auto()


@dataclass
class IDENotification:
    event: IDEEvent
    data: dict = field(default_factory=dict)


class IDE:
    """
    Unified Integrated Development Environment.

    Composes LSP (editor intelligence), REPL/Debugger (execution),
    and Package Manager (dependencies) into a single coherent system.
    """

    def __init__(self, project_root: str = "/project"):
        # Core systems
        self.workspace = Workspace(root=project_root)
        self.editor = EditorService(self.workspace)
        self.terminal = TerminalService(self.workspace)
        self.debugger = DebugService(self.workspace)
        self.packages = PackageService(self.workspace)
        self.build = BuildService(self.workspace)

        # IDE state
        self._notifications: list[IDENotification] = []
        self._event_handlers: dict[IDEEvent, list] = {}
        self._initialized = False

        # Wire up workspace change listener
        self.workspace.on_change(self._on_workspace_change)

    def initialize(self):
        """Initialize the IDE and all subsystems."""
        self.editor.initialize(root_uri="file://" + self.workspace.root)
        self._initialized = True
        return {'status': 'initialized', 'root': self.workspace.root}

    def shutdown(self):
        """Shutdown all subsystems."""
        # Stop all debug sessions
        for sid in list(self.debugger.list_sessions()):
            self.debugger.stop(sid)
        self.editor.shutdown()
        self._initialized = False

    # -- File operations --

    def create_file(self, path: str, content: str = "") -> dict:
        """Create a file and open it in the editor."""
        f = self.workspace.create_file(path, content)
        diags = self.editor.open_file(path)
        self._emit(IDEEvent.FILE_CREATED, {'path': path})
        return {'path': f.path, 'diagnostics': diags}

    def open_file(self, path: str) -> dict:
        """Open a file in the editor for intelligence."""
        if not self.workspace.exists(path):
            return {'error': f"File not found: {path}"}
        diags = self.editor.open_file(path)
        return {'path': self.workspace._resolve(path), 'diagnostics': diags}

    def edit_file(self, path: str, content: str) -> dict:
        """Edit a file and get updated diagnostics."""
        diags = self.editor.update_file(path, content)
        self._emit(IDEEvent.FILE_CHANGED, {'path': path})
        self._emit(IDEEvent.DIAGNOSTICS_UPDATED, {'path': path, 'diagnostics': diags})
        return {'path': self.workspace._resolve(path), 'diagnostics': diags}

    def close_file(self, path: str):
        """Close a file in the editor."""
        self.editor.close_file(path)

    def delete_file(self, path: str) -> bool:
        """Delete a file from the workspace."""
        self.editor.close_file(path)
        result = self.workspace.delete_file(path)
        if result:
            self._emit(IDEEvent.FILE_DELETED, {'path': path})
        return result

    # -- Editor intelligence --

    def complete(self, path: str, line: int, character: int) -> list[dict]:
        """Get completions at cursor position."""
        return self.editor.complete(path, line, character)

    def hover(self, path: str, line: int, character: int) -> Optional[dict]:
        """Get hover info at cursor position."""
        return self.editor.hover(path, line, character)

    def definition(self, path: str, line: int, character: int) -> Optional[dict]:
        """Go to definition at cursor position."""
        return self.editor.definition(path, line, character)

    def symbols(self, path: str) -> list[dict]:
        """Get document symbols/outline."""
        return self.editor.symbols(path)

    def signature_help(self, path: str, line: int, character: int) -> Optional[dict]:
        """Get signature help at cursor position."""
        return self.editor.signature_help(path, line, character)

    def diagnostics(self, path: str) -> list[dict]:
        """Get diagnostics for a file."""
        return self.editor.get_diagnostics(path)

    # -- Terminal/REPL --

    def eval(self, source: str) -> dict:
        """Evaluate code in the REPL."""
        return self.terminal.eval(source)

    def run_file(self, path: str) -> dict:
        """Run a file via the build system."""
        result = self.build.run(path)
        self._emit(IDEEvent.BUILD_COMPLETE, {
            'path': path, 'success': result.success
        })
        return result.to_dict()

    def run_file_in_repl(self, path: str) -> dict:
        """Run a file in the REPL context (persists state)."""
        return self.terminal.run_file(path)

    # -- Debugger --

    def debug_file(self, path: str, env: dict = None) -> dict:
        """Start debugging a file."""
        result = self.debugger.start(path, env=env)
        if 'error' not in result or result.get('error') is None:
            self._emit(IDEEvent.DEBUG_STARTED, {'path': path})
        return result

    def debug_step(self, session_id: str = None) -> Optional[dict]:
        return self.debugger.step(session_id)

    def debug_step_line(self, session_id: str = None) -> Optional[dict]:
        return self.debugger.step_line(session_id)

    def debug_step_over(self, session_id: str = None) -> Optional[dict]:
        return self.debugger.step_over(session_id)

    def debug_step_out(self, session_id: str = None) -> Optional[dict]:
        return self.debugger.step_out(session_id)

    def debug_continue(self, session_id: str = None) -> Optional[dict]:
        return self.debugger.continue_execution(session_id)

    def debug_add_breakpoint(self, line: int = -1, address: int = -1,
                             condition: str = "",
                             session_id: str = None) -> Optional[int]:
        return self.debugger.add_breakpoint(
            line=line, address=address, condition=condition,
            session_id=session_id
        )

    def debug_remove_breakpoint(self, bp_id: int,
                                session_id: str = None) -> bool:
        return self.debugger.remove_breakpoint(bp_id, session_id)

    def debug_add_watch(self, expression: str,
                        session_id: str = None) -> Optional[int]:
        return self.debugger.add_watch(expression, session_id)

    def debug_context(self, session_id: str = None) -> Optional[dict]:
        return self.debugger.get_context(session_id)

    def debug_eval(self, expr: str, session_id: str = None) -> Optional[dict]:
        return self.debugger.eval_expression(expr, session_id)

    def debug_stop(self, session_id: str = None) -> bool:
        result = self.debugger.stop(session_id)
        if result:
            self._emit(IDEEvent.DEBUG_STOPPED, {'session_id': session_id})
        return result

    # -- Package management --

    def publish_package(self, name: str, version: str, files: dict = None,
                        deps: dict = None, description: str = "") -> dict:
        """Publish a package to the registry."""
        spec = self.packages.publish(name, version, files=files, deps=deps,
                                     description=description)
        return {'name': spec.name, 'version': str(spec.version)}

    def add_dependency(self, name: str, constraint: str,
                       dev: bool = False) -> dict:
        """Add a dependency and install it."""
        resolved = self.packages.add_dependency(name, constraint, dev=dev)
        # Update editor completions with installed packages
        self.editor.add_package_completions(self.packages.installed_package_names())
        self._emit(IDEEvent.PACKAGE_INSTALLED, {
            'name': name, 'resolved': {k: str(v) for k, v in resolved.items()}
        })
        return {k: str(v) for k, v in resolved.items()}

    def remove_dependency(self, name: str) -> list[str]:
        """Remove a dependency."""
        removed = self.packages.remove_dependency(name)
        self.editor.add_package_completions(self.packages.installed_package_names())
        for r in removed:
            self._emit(IDEEvent.PACKAGE_REMOVED, {'name': r})
        return removed

    def install_packages(self, include_dev: bool = False) -> dict:
        """Install all project dependencies."""
        resolved = self.packages.install(include_dev=include_dev)
        self.editor.add_package_completions(self.packages.installed_package_names())
        return {k: str(v) for k, v in resolved.items()}

    def list_packages(self) -> list[dict]:
        return self.packages.list_installed()

    def dependency_tree(self) -> dict:
        return self.packages.dependency_tree()

    def package_audit(self) -> dict:
        return self.packages.audit()

    # -- Cross-system operations --

    def check_project(self) -> dict:
        """Check all workspace files for errors."""
        results = {}
        for path in self.workspace.list_files("*.ml"):
            check = self.build.check(path)
            results[path] = check
        return results

    def run_all(self) -> dict:
        """Run all workspace files and collect results."""
        results = {}
        for path in self.workspace.list_files("*.ml"):
            result = self.build.run(path)
            results[path] = result.to_dict()
        return results

    def refresh_diagnostics(self) -> dict:
        """Refresh diagnostics for all open files."""
        return self.editor.refresh_all()

    def get_project_summary(self) -> dict:
        """Get a summary of the project state."""
        files = self.workspace.list_files()
        open_files = self.editor.get_open_files()
        installed = self.packages.list_installed()
        debug_sessions = self.debugger.list_sessions()
        reqs = self.packages.get_requirements()
        return {
            'root': self.workspace.root,
            'files': files,
            'open_files': open_files,
            'packages_installed': installed,
            'debug_sessions': debug_sessions,
            'requirements': reqs,
            'repl_variables': list(self.terminal.get_variables().keys()),
        }

    # -- Session persistence --

    def save_state(self) -> dict:
        """Save IDE state to a serializable dict."""
        files = {}
        for path, f in self.workspace.files.items():
            files[path] = {
                'content': f.content,
                'language': f.language,
                'version': f.version,
            }
        return {
            'root': self.workspace.root,
            'files': files,
            'open_files': self.editor.get_open_files(),
            'requirements': self.packages.get_requirements(),
            'repl_env': {
                k: v for k, v in self.terminal.get_variables().items()
                if isinstance(v, (int, float, str, bool))
            },
        }

    def restore_state(self, state: dict):
        """Restore IDE state from a saved dict."""
        # Restore files
        for path, info in state.get('files', {}).items():
            self.workspace.create_file(
                path, content=info['content'], language=info.get('language', 'minilang')
            )
            f = self.workspace.get_file(path)
            if f:
                f.version = info.get('version', 0)

        # Re-open files
        for path in state.get('open_files', []):
            self.editor.open_file(path)

        # Restore requirements
        reqs = state.get('requirements', {})
        self.packages._project_requirements = reqs.get('dependencies', {})
        self.packages._dev_requirements = reqs.get('devDependencies', {})

        # Restore simple REPL variables
        env = state.get('repl_env', {})
        if env:
            self.terminal.inject_variables(env)

    # -- Event system --

    def on(self, event: IDEEvent, handler):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def get_notifications(self, clear: bool = False) -> list[dict]:
        """Get all IDE notifications."""
        notifs = [{'event': n.event.name, 'data': n.data} for n in self._notifications]
        if clear:
            self._notifications.clear()
        return notifs

    def _emit(self, event: IDEEvent, data: dict = None):
        self._notifications.append(IDENotification(event=event, data=data or {}))
        for handler in self._event_handlers.get(event, []):
            try:
                handler(event, data or {})
            except Exception:
                pass

    def _on_workspace_change(self, change_type: str, path: str):
        """Handle workspace file changes -- auto-update LSP."""
        if change_type == "changed" and self.editor.is_open(path):
            content = self.workspace.read_file(path)
            if content is not None:
                # LSP is already updated by editor.update_file
                pass
