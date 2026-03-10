"""
Profiler-Guided Optimizer (PGO)
Challenge C028 -- AgentZero Session 029

Composes: C027 (Profiler) + C014 (Bytecode Optimizer)

Concept: Instead of blindly optimizing all code, use profiling data to
identify hotspots and apply targeted optimization where it matters most.
This mirrors how real compilers (GCC -fprofile-use, LLVM PGO) work.

Features:
  - Profile-first optimization: measure before optimizing
  - Hot function detection with configurable thresholds
  - Targeted optimization: only optimize hot functions
  - Optimization suggestions based on profiling patterns
  - Before/after comparison with speedup metrics
  - Iterative PGO: profile -> optimize -> re-profile -> verify
  - Optimization budget: skip functions not worth optimizing
  - Inlining suggestions based on call frequency and body size
  - Report generation with actionable recommendations

Architecture:
  Source -> Compile -> Profile (C027) -> Analyze Hotspots ->
  Optimize Hot Paths (C014) -> Re-profile -> Compare -> Report
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Optional

# Import C027 Profiler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C027_profiler'))
from profiler import (
    Profiler, ProfiledVM, ProfileSnapshot, FunctionProfile,
    LineProfile, InstructionProfile, CallGraphEdge,
    profile_source, flat_profile as get_flat_profile,
)

# Import C014 Optimizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C014_bytecode_optimizer'))
from optimizer import (
    optimize_chunk, optimize_all, optimize_source, OptimizationStats,
    decode_chunk, addrs_to_indices, indices_to_addrs, encode_instructions,
    constant_fold, constant_propagation, strength_reduce, peephole,
    optimize_jumps, eliminate_dead_code, Instr,
)

# Import C010 Stack VM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Op, Chunk, Token, TokenType, lex, Parser, Compiler, VM, FnObject,
    compile_source, execute, disassemble, VMError,
)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class HotFunction:
    """A function identified as a hotspot."""
    name: str
    call_count: int = 0
    self_steps: int = 0
    self_time: float = 0.0
    pct_steps: float = 0.0      # percentage of total steps
    pct_time: float = 0.0       # percentage of total time
    heat_score: float = 0.0     # combined hotness metric
    chunk_size: int = 0         # bytecode size


@dataclass
class InlineSuggestion:
    """Suggestion to inline a function call."""
    callee: str
    caller: str
    call_count: int = 0
    callee_size: int = 0        # bytecode instructions
    estimated_savings: int = 0  # estimated steps saved
    reason: str = ""


@dataclass
class OptimizationSuggestion:
    """A suggested optimization action."""
    target: str                 # function name or "<main>"
    category: str               # "constant_fold", "strength_reduce", "inline", etc.
    description: str = ""
    estimated_impact: float = 0.0  # 0.0-1.0 estimated improvement
    priority: int = 0           # higher = more important


@dataclass
class PGOResult:
    """Result of a profiler-guided optimization pass."""
    original_profile: Optional[ProfileSnapshot] = None
    optimized_profile: Optional[ProfileSnapshot] = None
    hot_functions: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    inline_suggestions: list = field(default_factory=list)
    optimization_stats: Optional[OptimizationStats] = None
    original_steps: int = 0
    optimized_steps: int = 0
    steps_saved: int = 0
    speedup: float = 1.0       # ratio of old/new steps
    iterations: int = 0
    original_output: list = field(default_factory=list)
    optimized_output: list = field(default_factory=list)
    same_output: bool = True


@dataclass
class PGOConfig:
    """Configuration for PGO optimization."""
    hot_threshold: float = 0.1       # function is hot if >= 10% of total steps
    min_call_count: int = 2          # minimum calls to consider for optimization
    inline_max_size: int = 20        # max bytecode size for inline candidate
    inline_min_calls: int = 3        # min calls for inline suggestion
    max_iterations: int = 3          # max PGO iterations
    convergence_threshold: float = 0.01  # stop if improvement < 1%
    optimize_cold: bool = False      # whether to optimize cold functions
    max_opt_rounds: int = 10         # max optimization rounds per function


# ============================================================
# Hot Function Detection
# ============================================================

def detect_hot_functions(profile: ProfileSnapshot, config: PGOConfig) -> list:
    """Identify hot functions from profiling data.
    Returns list of HotFunction sorted by heat_score descending."""
    hot = []
    total_steps = profile.total_steps
    total_time = profile.total_time

    for name, fp in profile.functions.items():
        pct_steps = (fp.self_steps / total_steps * 100) if total_steps > 0 else 0.0
        pct_time = (fp.self_time / total_time * 100) if total_time > 0 else 0.0

        # Heat score: weighted combination of step% and call frequency
        heat_score = (pct_steps / 100.0) * 0.7 + min(fp.call_count / 100.0, 1.0) * 0.3

        hf = HotFunction(
            name=name,
            call_count=fp.call_count,
            self_steps=fp.self_steps,
            self_time=fp.self_time,
            pct_steps=pct_steps,
            pct_time=pct_time,
            heat_score=heat_score,
        )
        hot.append(hf)

    hot.sort(key=lambda h: h.heat_score, reverse=True)
    return hot


def filter_hot(hot_functions: list, config: PGOConfig) -> list:
    """Filter to only genuinely hot functions based on config thresholds."""
    return [
        hf for hf in hot_functions
        if (hf.pct_steps / 100.0 >= config.hot_threshold
            or hf.call_count >= config.min_call_count)
    ]


# ============================================================
# Optimization Suggestions
# ============================================================

def generate_suggestions(
    profile: ProfileSnapshot,
    hot_functions: list,
    compiler: Optional[Compiler] = None,
    config: PGOConfig = None,
) -> list:
    """Generate optimization suggestions based on profiling data."""
    if config is None:
        config = PGOConfig()
    suggestions = []

    for hf in hot_functions:
        # High self-step functions: likely computational hotspots
        if hf.pct_steps > 30:
            suggestions.append(OptimizationSuggestion(
                target=hf.name,
                category="hotspot",
                description=f"Function '{hf.name}' uses {hf.pct_steps:.1f}% of total steps -- primary optimization target",
                estimated_impact=min(hf.pct_steps / 100.0, 0.5),
                priority=10,
            ))

        # Frequently called functions: call overhead matters
        if hf.call_count >= 10:
            suggestions.append(OptimizationSuggestion(
                target=hf.name,
                category="call_overhead",
                description=f"Function '{hf.name}' called {hf.call_count} times -- call overhead is significant",
                estimated_impact=min(hf.call_count / 1000.0, 0.3),
                priority=7,
            ))

        # Functions with high self-steps per call: optimize body
        if hf.call_count > 0:
            steps_per_call = hf.self_steps / hf.call_count
            if steps_per_call > 50:
                suggestions.append(OptimizationSuggestion(
                    target=hf.name,
                    category="body_optimization",
                    description=f"Function '{hf.name}' averages {steps_per_call:.0f} steps/call -- body can be optimized",
                    estimated_impact=min(steps_per_call / 500.0, 0.4),
                    priority=8,
                ))

    # Instruction-level suggestions
    for op_name, ip in profile.instructions.items():
        pct = (ip.count / profile.total_steps * 100) if profile.total_steps > 0 else 0
        if op_name in ('CONST', 'LOAD', 'STORE') and pct > 30:
            suggestions.append(OptimizationSuggestion(
                target="<global>",
                category="instruction_pattern",
                description=f"High {op_name} frequency ({pct:.1f}%) -- constant folding/propagation may help",
                estimated_impact=0.1,
                priority=5,
            ))

    suggestions.sort(key=lambda s: s.priority, reverse=True)
    return suggestions


def generate_inline_suggestions(
    profile: ProfileSnapshot,
    compiler: Optional[Compiler] = None,
    config: PGOConfig = None,
) -> list:
    """Generate inlining suggestions based on call frequency and function size."""
    if config is None:
        config = PGOConfig()
    suggestions = []

    if not compiler:
        return suggestions

    for name, fp in profile.functions.items():
        if name == "<main>":
            continue

        # Check if function exists in compiler
        fn_obj = compiler.functions.get(name)
        if not fn_obj:
            continue

        body_size = len(fn_obj.chunk.code)

        # Small frequently-called function: inline candidate
        if (fp.call_count >= config.inline_min_calls
                and body_size <= config.inline_max_size):

            # Estimate savings: avoid CALL/RETURN overhead per invocation
            # CALL setup is ~5 instructions overhead, RETURN is ~3
            overhead_per_call = 8
            estimated_savings = fp.call_count * overhead_per_call

            for caller_name, count in fp.callers.items():
                suggestions.append(InlineSuggestion(
                    callee=name,
                    caller=caller_name,
                    call_count=count,
                    callee_size=body_size,
                    estimated_savings=estimated_savings,
                    reason=f"Small function ({body_size} bytes) called {count} times from {caller_name}",
                ))

    suggestions.sort(key=lambda s: s.estimated_savings, reverse=True)
    return suggestions


# ============================================================
# Targeted Optimization
# ============================================================

def optimize_targeted(
    chunk: Chunk,
    compiler: Optional[Compiler],
    hot_names: set,
    config: PGOConfig,
) -> tuple:
    """Optimize only hot functions, leaving cold code untouched.
    Returns (optimized_chunk, stats)."""

    # Always optimize main chunk (it contains the top-level code)
    opt_chunk, main_stats = optimize_chunk(chunk, max_rounds=config.max_opt_rounds)

    total = OptimizationStats()
    total.rounds = main_stats.rounds
    total.original_size = main_stats.original_size
    total.optimized_size = main_stats.optimized_size
    total.constant_folds = main_stats.constant_folds
    total.strength_reductions = main_stats.strength_reductions
    total.peephole_opts = main_stats.peephole_opts
    total.dead_code_eliminations = main_stats.dead_code_eliminations
    total.jump_optimizations = main_stats.jump_optimizations
    total.constant_propagations = main_stats.constant_propagations

    if compiler:
        for name, fn_obj in compiler.functions.items():
            is_hot = name in hot_names or config.optimize_cold

            if is_hot:
                opt_fn_chunk, fn_stats = optimize_chunk(
                    fn_obj.chunk, max_rounds=config.max_opt_rounds
                )
                opt_fn = FnObject(fn_obj.name, fn_obj.arity, opt_fn_chunk)

                total.original_size += fn_stats.original_size
                total.optimized_size += fn_stats.optimized_size
                total.constant_folds += fn_stats.constant_folds
                total.strength_reductions += fn_stats.strength_reductions
                total.peephole_opts += fn_stats.peephole_opts
                total.dead_code_eliminations += fn_stats.dead_code_eliminations
                total.jump_optimizations += fn_stats.jump_optimizations
                total.constant_propagations += fn_stats.constant_propagations
                total.rounds = max(total.rounds, fn_stats.rounds)
            else:
                # Keep cold function unchanged
                opt_fn = fn_obj
                total.original_size += len(fn_obj.chunk.code)
                total.optimized_size += len(fn_obj.chunk.code)

            # Update function reference in chunk constants
            for i, c in enumerate(opt_chunk.constants):
                if isinstance(c, FnObject) and c.name == name:
                    opt_chunk.constants[i] = opt_fn

    return opt_chunk, total


# ============================================================
# Profile-Guided Optimizer
# ============================================================

class ProfileGuidedOptimizer:
    """Main PGO engine composing Profiler and Optimizer."""

    def __init__(self, config: Optional[PGOConfig] = None):
        self.config = config or PGOConfig()
        self.profiler = Profiler()
        self.history = []  # list of PGOResult

    def optimize(self, source: str) -> PGOResult:
        """Run one iteration of profile-guided optimization.

        Steps:
        1. Compile source
        2. Profile execution
        3. Detect hot functions
        4. Generate suggestions
        5. Apply targeted optimization
        6. Re-profile optimized code
        7. Compare and report
        """
        result = PGOResult()

        # Step 1: Compile
        chunk, compiler = compile_source(source)

        # Step 2: Profile original execution
        profiler = Profiler()
        prof_result = profiler.profile(source)
        result.original_profile = profiler.get_latest_profile()
        result.original_steps = prof_result['steps']
        result.original_output = list(prof_result['output'])

        # Step 3: Detect hot functions
        all_hot = detect_hot_functions(result.original_profile, self.config)
        result.hot_functions = filter_hot(all_hot, self.config)

        # Step 4: Generate suggestions
        result.suggestions = generate_suggestions(
            result.original_profile, result.hot_functions, compiler, self.config
        )
        result.inline_suggestions = generate_inline_suggestions(
            result.original_profile, compiler, self.config
        )

        # Step 5: Apply targeted optimization
        hot_names = {hf.name for hf in result.hot_functions}
        opt_chunk, stats = optimize_targeted(chunk, compiler, hot_names, self.config)
        result.optimization_stats = stats

        # Step 6: Re-profile optimized code
        opt_vm = ProfiledVM(opt_chunk)
        opt_vm.run()
        result.optimized_profile = opt_vm.profile
        result.optimized_steps = opt_vm.step_count
        result.optimized_output = list(opt_vm.output)

        # Step 7: Compare
        result.steps_saved = result.original_steps - result.optimized_steps
        result.speedup = (
            result.original_steps / result.optimized_steps
            if result.optimized_steps > 0
            else float('inf')
        )
        result.same_output = result.original_output == result.optimized_output
        result.iterations = 1

        self.history.append(result)
        return result

    def iterative_optimize(self, source: str) -> PGOResult:
        """Run multiple PGO iterations until convergence.

        Each iteration:
        1. Profile current code
        2. Optimize based on profile
        3. Check if improvement exceeds threshold
        4. If not, stop (diminishing returns)
        """
        # First iteration works on source
        chunk, compiler = compile_source(source)

        # Profile original
        profiler = Profiler()
        prof_result = profiler.profile(source)
        original_profile = profiler.get_latest_profile()
        original_steps = prof_result['steps']
        original_output = list(prof_result['output'])

        all_hot = detect_hot_functions(original_profile, self.config)
        hot_functions = filter_hot(all_hot, self.config)
        suggestions = generate_suggestions(
            original_profile, hot_functions, compiler, self.config
        )
        inline_suggestions = generate_inline_suggestions(
            original_profile, compiler, self.config
        )

        # Apply optimization
        hot_names = {hf.name for hf in hot_functions}
        current_chunk, total_stats = optimize_targeted(
            chunk, compiler, hot_names, self.config
        )

        prev_steps = original_steps
        iterations = 1

        # Iterative refinement
        for iteration in range(1, self.config.max_iterations):
            # Profile the optimized code
            opt_vm = ProfiledVM(current_chunk)
            opt_vm.run()
            current_steps = opt_vm.step_count

            # Check convergence
            if prev_steps > 0:
                improvement = (prev_steps - current_steps) / prev_steps
            else:
                improvement = 0.0

            if improvement < self.config.convergence_threshold:
                break

            # Re-detect hotspots in optimized code
            new_hot = detect_hot_functions(opt_vm.profile, self.config)
            new_hot_filtered = filter_hot(new_hot, self.config)
            new_hot_names = {hf.name for hf in new_hot_filtered}

            # Re-optimize (note: we optimize the already-optimized chunk)
            current_chunk, iter_stats = optimize_targeted(
                current_chunk, None, new_hot_names, self.config
            )

            total_stats.rounds += iter_stats.rounds
            total_stats.constant_folds += iter_stats.constant_folds
            total_stats.strength_reductions += iter_stats.strength_reductions
            total_stats.peephole_opts += iter_stats.peephole_opts
            total_stats.dead_code_eliminations += iter_stats.dead_code_eliminations
            total_stats.jump_optimizations += iter_stats.jump_optimizations
            total_stats.constant_propagations += iter_stats.constant_propagations
            total_stats.optimized_size = iter_stats.optimized_size

            prev_steps = current_steps
            iterations += 1

        # Final profile
        final_vm = ProfiledVM(current_chunk)
        final_vm.run()
        optimized_profile = final_vm.profile
        optimized_steps = final_vm.step_count
        optimized_output = list(final_vm.output)

        result = PGOResult(
            original_profile=original_profile,
            optimized_profile=optimized_profile,
            hot_functions=hot_functions,
            suggestions=suggestions,
            inline_suggestions=inline_suggestions,
            optimization_stats=total_stats,
            original_steps=original_steps,
            optimized_steps=optimized_steps,
            steps_saved=original_steps - optimized_steps,
            speedup=(
                original_steps / optimized_steps
                if optimized_steps > 0
                else float('inf')
            ),
            iterations=iterations,
            original_output=original_output,
            optimized_output=optimized_output,
            same_output=original_output == optimized_output,
        )

        self.history.append(result)
        return result

    def analyze_only(self, source: str) -> PGOResult:
        """Profile and analyze without applying optimization.
        Useful for getting suggestions without modifying code."""
        result = PGOResult()

        chunk, compiler = compile_source(source)

        profiler = Profiler()
        prof_result = profiler.profile(source)
        result.original_profile = profiler.get_latest_profile()
        result.original_steps = prof_result['steps']
        result.original_output = list(prof_result['output'])

        all_hot = detect_hot_functions(result.original_profile, self.config)
        result.hot_functions = filter_hot(all_hot, self.config)

        result.suggestions = generate_suggestions(
            result.original_profile, result.hot_functions, compiler, self.config
        )
        result.inline_suggestions = generate_inline_suggestions(
            result.original_profile, compiler, self.config
        )

        # No optimization applied
        result.optimized_steps = result.original_steps
        result.steps_saved = 0
        result.speedup = 1.0
        result.iterations = 0
        result.same_output = True

        self.history.append(result)
        return result

    def compare_strategies(self, source: str) -> dict:
        """Compare different optimization strategies:
        1. No optimization
        2. Blind optimization (optimize everything)
        3. PGO (targeted optimization)
        """
        chunk, compiler = compile_source(source)

        # 1. No optimization: profile only
        profiler = Profiler()
        prof_result = profiler.profile(source)
        no_opt_steps = prof_result['steps']
        no_opt_output = list(prof_result['output'])

        # 2. Blind optimization: optimize everything
        opt_chunk_blind, _, blind_stats = optimize_all(chunk, compiler)
        blind_vm = ProfiledVM(opt_chunk_blind)
        blind_vm.run()
        blind_steps = blind_vm.step_count
        blind_output = list(blind_vm.output)

        # 3. PGO: targeted optimization
        # Re-compile (optimize_all consumed the chunk)
        chunk2, compiler2 = compile_source(source)
        profile = profiler.get_latest_profile()
        all_hot = detect_hot_functions(profile, self.config)
        hot_filtered = filter_hot(all_hot, self.config)
        hot_names = {hf.name for hf in hot_filtered}
        pgo_chunk, pgo_stats = optimize_targeted(chunk2, compiler2, hot_names, self.config)
        pgo_vm = ProfiledVM(pgo_chunk)
        pgo_vm.run()
        pgo_steps = pgo_vm.step_count
        pgo_output = list(pgo_vm.output)

        return {
            'no_opt': {
                'steps': no_opt_steps,
                'output': no_opt_output,
            },
            'blind_opt': {
                'steps': blind_steps,
                'output': blind_output,
                'stats': blind_stats,
                'speedup': no_opt_steps / blind_steps if blind_steps > 0 else float('inf'),
                'same_output': no_opt_output == blind_output,
            },
            'pgo': {
                'steps': pgo_steps,
                'output': pgo_output,
                'stats': pgo_stats,
                'speedup': no_opt_steps / pgo_steps if pgo_steps > 0 else float('inf'),
                'hot_functions': [hf.name for hf in hot_filtered],
                'same_output': no_opt_output == pgo_output,
            },
        }

    def get_history(self) -> list:
        """Get history of PGO runs."""
        return self.history

    def get_trend(self) -> list:
        """Get optimization trend across runs."""
        return [
            {
                'iteration': i + 1,
                'original_steps': r.original_steps,
                'optimized_steps': r.optimized_steps,
                'speedup': r.speedup,
                'steps_saved': r.steps_saved,
            }
            for i, r in enumerate(self.history)
        ]


# ============================================================
# Report Generation
# ============================================================

def format_pgo_report(result: PGOResult) -> str:
    """Format a PGO result as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PROFILE-GUIDED OPTIMIZATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Original steps:   {result.original_steps:>10d}")
    lines.append(f"  Optimized steps:  {result.optimized_steps:>10d}")
    lines.append(f"  Steps saved:      {result.steps_saved:>10d}")
    lines.append(f"  Speedup:          {result.speedup:>10.2f}x")
    lines.append(f"  Iterations:       {result.iterations:>10d}")
    lines.append(f"  Output preserved: {'Yes' if result.same_output else 'NO -- MISMATCH'}")
    lines.append("")

    # Hot functions
    if result.hot_functions:
        lines.append("HOT FUNCTIONS")
        lines.append("-" * 40)
        lines.append(f"  {'Function':<20s} {'Calls':>6s} {'Steps%':>7s} {'Heat':>6s}")
        for hf in result.hot_functions[:10]:
            lines.append(
                f"  {hf.name:<20s} {hf.call_count:>6d} "
                f"{hf.pct_steps:>6.1f}% {hf.heat_score:>5.3f}"
            )
        lines.append("")

    # Optimization stats
    if result.optimization_stats:
        stats = result.optimization_stats
        lines.append("OPTIMIZATION PASSES APPLIED")
        lines.append("-" * 40)
        lines.append(f"  Rounds:               {stats.rounds}")
        lines.append(f"  Constant folds:       {stats.constant_folds}")
        lines.append(f"  Constant propagations:{stats.constant_propagations}")
        lines.append(f"  Strength reductions:  {stats.strength_reductions}")
        lines.append(f"  Peephole opts:        {stats.peephole_opts}")
        lines.append(f"  Jump optimizations:   {stats.jump_optimizations}")
        lines.append(f"  Dead code eliminated: {stats.dead_code_eliminations}")
        lines.append(f"  Size reduction:       {stats.size_reduction:.1%}")
        lines.append("")

    # Suggestions
    if result.suggestions:
        lines.append("OPTIMIZATION SUGGESTIONS")
        lines.append("-" * 40)
        for s in result.suggestions[:5]:
            lines.append(f"  [{s.category}] {s.description}")
        lines.append("")

    # Inline suggestions
    if result.inline_suggestions:
        lines.append("INLINE CANDIDATES")
        lines.append("-" * 40)
        for s in result.inline_suggestions[:5]:
            lines.append(f"  {s.reason}")
        lines.append("")

    lines.append("=" * 60)
    return '\n'.join(lines)


def format_comparison_report(comparison: dict) -> str:
    """Format strategy comparison as readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  OPTIMIZATION STRATEGY COMPARISON")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  {'Strategy':<20s} {'Steps':>10s} {'Speedup':>10s} {'Output OK':>10s}")
    lines.append("  " + "-" * 52)

    no_opt = comparison['no_opt']
    blind = comparison['blind_opt']
    pgo = comparison['pgo']

    lines.append(f"  {'No optimization':<20s} {no_opt['steps']:>10d} {'1.00x':>10s} {'Yes':>10s}")
    lines.append(
        f"  {'Blind optimization':<20s} {blind['steps']:>10d} "
        f"{blind['speedup']:>9.2f}x {'Yes' if blind['same_output'] else 'NO':>10s}"
    )
    lines.append(
        f"  {'PGO (targeted)':<20s} {pgo['steps']:>10d} "
        f"{pgo['speedup']:>9.2f}x {'Yes' if pgo['same_output'] else 'NO':>10s}"
    )

    if pgo.get('hot_functions'):
        lines.append("")
        lines.append(f"  PGO hot targets: {', '.join(pgo['hot_functions'])}")

    lines.append("")
    lines.append("=" * 60)
    return '\n'.join(lines)


# ============================================================
# Convenience Functions
# ============================================================

def pgo_optimize(source: str, config: Optional[PGOConfig] = None) -> PGOResult:
    """One-shot PGO optimization."""
    optimizer = ProfileGuidedOptimizer(config)
    return optimizer.optimize(source)


def pgo_iterative(source: str, config: Optional[PGOConfig] = None) -> PGOResult:
    """Iterative PGO optimization."""
    optimizer = ProfileGuidedOptimizer(config)
    return optimizer.iterative_optimize(source)


def pgo_analyze(source: str, config: Optional[PGOConfig] = None) -> PGOResult:
    """Analyze without optimizing."""
    optimizer = ProfileGuidedOptimizer(config)
    return optimizer.analyze_only(source)


def pgo_compare(source: str, config: Optional[PGOConfig] = None) -> dict:
    """Compare optimization strategies."""
    optimizer = ProfileGuidedOptimizer(config)
    return optimizer.compare_strategies(source)
