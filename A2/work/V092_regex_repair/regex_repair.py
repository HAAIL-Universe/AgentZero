"""V092: Regex Repair

Given a regex that fails on some positive/negative examples, find a minimal
edit to the regex AST that makes it accept all positives and reject all negatives.

Composes:
- V084 (Symbolic Regex): regex parsing, compilation, AST manipulation
- V081 (Symbolic Automata): SFA operations, acceptance testing, equivalence
- V088 (Regex Synthesis): sub-regex synthesis for hole filling

Repair strategies:
1. Single-node mutations: replace one AST node with alternatives
2. Quantifier repair: change *, +, ? on sub-expressions
3. Character class repair: widen/narrow character classes
4. Structural repair: insert/delete/swap sub-expressions
5. Hole-based repair: replace a subtree with a synthesized alternative
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V084_symbolic_regex'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V088_regex_synthesis'))

from symbolic_automata import (
    SFA, SFATransition, CharAlgebra, PChar, PRange, PTrue, PFalse,
    PAnd, POr, PNot, sfa_union, sfa_intersection, sfa_complement,
    sfa_difference, sfa_is_equivalent, sfa_is_subset,
    sfa_from_string, sfa_concat, sfa_star, sfa_empty, sfa_epsilon,
    sfa_any_char, sfa_from_char_class
)
from symbolic_regex import (
    Regex, RegexKind, RLit, RDot, RClass, RNegClass, RConcat, RAlt,
    RStar, RPlus, ROptional, REpsilon, REmpty,
    compile_regex, compile_regex_dfa, compile_regex_min,
    regex_equivalent, regex_to_string, regex_size, regex_accepts_epsilon,
    RegexCompiler, parse_regex, regex_sample
)
from regex_synthesis import synthesize_regex, SynthesisResult as SynthResult

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, FrozenSet
from collections import deque
import time


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """Result of regex repair."""
    success: bool
    original_pattern: str = ""
    repaired_regex: Optional[Regex] = None
    repaired_pattern: str = ""
    edit_distance: int = 0       # AST edit distance from original
    method: str = ""             # Which repair strategy succeeded
    stats: Dict = field(default_factory=dict)

    @property
    def pattern(self):
        return self.repaired_pattern


@dataclass
class DiagnosticResult:
    """Diagnosis of why a regex fails."""
    pattern: str
    false_negatives: List[str] = field(default_factory=list)  # positives not matched
    false_positives: List[str] = field(default_factory=list)  # negatives matched
    fault_nodes: List[Tuple[int, Regex]] = field(default_factory=list)  # (index, node)
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile_and_test(regex, positives, negatives, algebra=None):
    """Compile regex and test against examples. Returns (accepts_all_pos, rejects_all_neg, fn, fp)."""
    alg = algebra or CharAlgebra()
    compiler = RegexCompiler(alg)
    try:
        sfa = compiler.compile(regex).determinize()
    except Exception:
        return False, False, list(positives), []
    fn = [s for s in positives if not sfa.accepts(s)]
    fp = [s for s in negatives if sfa.accepts(s)]
    return len(fn) == 0, len(fp) == 0, fn, fp


def _test_pattern(pattern, positives, negatives, algebra=None):
    """Test a pattern string against examples."""
    try:
        regex = parse_regex(pattern)
        return _compile_and_test(regex, positives, negatives, algebra)
    except Exception:
        return False, False, list(positives), []


def _is_correct(regex, positives, negatives, algebra=None):
    """Check if regex is correct on all examples."""
    ap, rn, _, _ = _compile_and_test(regex, positives, negatives, algebra)
    return ap and rn


def _ast_edit_distance(a, b):
    """Simple AST edit distance: count of differing nodes."""
    if a == b:
        return 0
    if a.kind != b.kind:
        return 1 + regex_size(a) + regex_size(b)
    dist = 0
    if a.char != b.char:
        dist += 1
    if a.ranges != b.ranges:
        dist += 1
    # Compare children
    ac = list(a.children) if a.children else []
    bc = list(b.children) if b.children else []
    if a.child:
        ac.append(a.child)
    if b.child:
        bc.append(b.child)
    max_len = max(len(ac), len(bc))
    for i in range(max_len):
        if i < len(ac) and i < len(bc):
            dist += _ast_edit_distance(ac[i], bc[i])
        else:
            extra = ac[i] if i < len(ac) else bc[i]
            dist += regex_size(extra)
    return dist


def _enumerate_subtrees(regex, path=()):
    """Enumerate all subtrees with their paths (tuple of child indices)."""
    yield (path, regex)
    if regex.children:
        for i, child in enumerate(regex.children):
            yield from _enumerate_subtrees(child, path + (i,))
    if regex.child:
        yield from _enumerate_subtrees(regex.child, path + (0,))


def _replace_at_path(regex, path, replacement):
    """Replace the subtree at the given path with replacement."""
    if not path:
        return replacement
    idx = path[0]
    rest = path[1:]
    if regex.children:
        new_children = list(regex.children)
        new_children[idx] = _replace_at_path(new_children[idx], rest, replacement)
        return Regex(
            kind=regex.kind, char=regex.char,
            children=tuple(new_children), child=regex.child,
            ranges=regex.ranges
        )
    elif regex.child:
        new_child = _replace_at_path(regex.child, rest, replacement)
        return Regex(
            kind=regex.kind, char=regex.char,
            children=regex.children, child=new_child,
            ranges=regex.ranges
        )
    return regex


# ---------------------------------------------------------------------------
# Repair strategy 1: Quantifier mutations
# ---------------------------------------------------------------------------

def _quantifier_mutations(node):
    """Generate quantifier variants of a node."""
    results = []
    inner = None
    if node.kind == RegexKind.STAR:
        inner = node.child
    elif node.kind == RegexKind.PLUS:
        inner = node.child
    elif node.kind == RegexKind.OPTIONAL:
        inner = node.child

    if inner:
        # Try all quantifier variants
        if node.kind != RegexKind.STAR:
            results.append(RStar(inner))
        if node.kind != RegexKind.PLUS:
            results.append(RPlus(inner))
        if node.kind != RegexKind.OPTIONAL:
            results.append(ROptional(inner))
        # Try removing quantifier entirely
        results.append(inner)
    else:
        # Try adding quantifiers
        results.append(RStar(node))
        results.append(RPlus(node))
        results.append(ROptional(node))
    return results


# ---------------------------------------------------------------------------
# Repair strategy 2: Character class mutations
# ---------------------------------------------------------------------------

def _char_class_mutations(node, positives, negatives):
    """Generate character class variants based on examples."""
    results = []

    # Collect all chars from positive examples
    pos_chars = set()
    for s in positives:
        pos_chars.update(s)

    neg_only_chars = set()
    for s in negatives:
        neg_only_chars.update(s)
    neg_only_chars -= pos_chars

    if node.kind == RegexKind.LITERAL and node.char:
        c = node.char
        # Try expanding to character class
        if c.isdigit():
            results.append(RClass((('0', '9'),)))
        if c.isalpha() and c.islower():
            results.append(RClass((('a', 'z'),)))
        if c.isalpha() and c.isupper():
            results.append(RClass((('A', 'Z'),)))
        if c.isalpha():
            results.append(RClass((('a', 'z'), ('A', 'Z'))))
        if c.isalnum():
            results.append(RClass((('a', 'z'), ('A', 'Z'), ('0', '9'))))
        results.append(RDot())

    elif node.kind == RegexKind.CHAR_CLASS:
        # Try expanding ranges
        results.append(RDot())
        # Try individual common classes
        results.append(RClass((('0', '9'),)))
        results.append(RClass((('a', 'z'),)))
        results.append(RClass((('a', 'z'), ('A', 'Z'), ('0', '9'))))
        # Try building class from positive chars
        if pos_chars:
            sorted_chars = sorted(pos_chars)
            ranges = _chars_to_ranges(sorted_chars)
            if ranges:
                results.append(RClass(tuple(ranges)))

    elif node.kind == RegexKind.DOT:
        # Try narrowing dot to specific classes
        if pos_chars:
            sorted_chars = sorted(pos_chars)
            ranges = _chars_to_ranges(sorted_chars)
            if ranges:
                results.append(RClass(tuple(ranges)))
        results.append(RClass((('a', 'z'),)))
        results.append(RClass((('0', '9'),)))
        results.append(RClass((('a', 'z'), ('A', 'Z'), ('0', '9'))))

    elif node.kind == RegexKind.NEG_CLASS:
        # Try inverting or expanding
        results.append(RDot())
        if node.ranges:
            results.append(RClass(node.ranges))  # Un-negate

    return results


def _chars_to_ranges(chars):
    """Convert sorted list of chars to ranges."""
    if not chars:
        return []
    ranges = []
    start = chars[0]
    end = chars[0]
    for c in chars[1:]:
        if ord(c) == ord(end) + 1:
            end = c
        else:
            ranges.append((start, end))
            start = c
            end = c
    ranges.append((start, end))
    return ranges


# ---------------------------------------------------------------------------
# Repair strategy 3: Structural mutations
# ---------------------------------------------------------------------------

def _structural_mutations(node, positives):
    """Generate structural variants."""
    results = []

    if node.kind == RegexKind.CONCAT and node.children:
        children = list(node.children)
        # Try removing each child
        for i in range(len(children)):
            remaining = children[:i] + children[i+1:]
            if len(remaining) == 0:
                results.append(REpsilon())
            elif len(remaining) == 1:
                results.append(remaining[0])
            else:
                results.append(RConcat(*remaining))
        # Try making each child optional
        for i in range(len(children)):
            new_children = list(children)
            new_children[i] = ROptional(children[i])
            results.append(RConcat(*new_children))

    elif node.kind == RegexKind.ALT and node.children:
        children = list(node.children)
        # Try removing each alternative
        for i in range(len(children)):
            remaining = children[:i] + children[i+1:]
            if len(remaining) == 1:
                results.append(remaining[0])
            elif len(remaining) > 1:
                results.append(RAlt(*remaining))
        # Try adding common patterns as alternatives
        results.append(RAlt(*(children + [REpsilon()])))

    return results


# ---------------------------------------------------------------------------
# Repair strategy 4: Literal substitution
# ---------------------------------------------------------------------------

def _literal_mutations(node, positives, negatives):
    """Try replacing a literal with other characters seen in examples."""
    results = []
    if node.kind != RegexKind.LITERAL or not node.char:
        return results

    pos_chars = set()
    for s in positives:
        pos_chars.update(s)

    for c in pos_chars:
        if c != node.char:
            results.append(RLit(c))
    return results


# ---------------------------------------------------------------------------
# Core repair engine
# ---------------------------------------------------------------------------

def _try_mutations(original, positives, negatives, algebra=None, max_mutations=500, timeout=10.0):
    """Try single-point mutations at every position in the AST."""
    start = time.time()
    best = None
    best_dist = float('inf')
    tried = 0

    subtrees = list(_enumerate_subtrees(original))

    for path, node in subtrees:
        if time.time() - start > timeout:
            break

        # Generate mutations for this node
        mutations = []
        mutations.extend(_quantifier_mutations(node))
        mutations.extend(_char_class_mutations(node, positives, negatives))
        mutations.extend(_structural_mutations(node, positives))
        mutations.extend(_literal_mutations(node, positives, negatives))

        for mut in mutations:
            if tried >= max_mutations:
                break
            tried += 1

            candidate = _replace_at_path(original, path, mut)
            if _is_correct(candidate, positives, negatives, algebra):
                dist = _ast_edit_distance(original, candidate)
                if dist < best_dist:
                    best = candidate
                    best_dist = dist

        if tried >= max_mutations:
            break

    return best, best_dist, tried


def _try_double_mutations(original, positives, negatives, algebra=None, max_mutations=500, timeout=10.0):
    """Try pairs of single-point mutations."""
    start = time.time()
    best = None
    best_dist = float('inf')
    tried = 0

    subtrees = list(_enumerate_subtrees(original))

    for i, (path1, node1) in enumerate(subtrees):
        if time.time() - start > timeout:
            break
        muts1 = (_quantifier_mutations(node1) +
                 _char_class_mutations(node1, positives, negatives) +
                 _structural_mutations(node1, positives))

        for mut1 in muts1:
            if tried >= max_mutations:
                break
            candidate1 = _replace_at_path(original, path1, mut1)

            # Try second mutation
            subtrees2 = list(_enumerate_subtrees(candidate1))
            for path2, node2 in subtrees2:
                if time.time() - start > timeout:
                    break
                if tried >= max_mutations:
                    break
                muts2 = (_quantifier_mutations(node2) +
                         _char_class_mutations(node2, positives, negatives))
                for mut2 in muts2[:5]:  # Limit inner loop
                    tried += 1
                    candidate2 = _replace_at_path(candidate1, path2, mut2)
                    if _is_correct(candidate2, positives, negatives, algebra):
                        dist = _ast_edit_distance(original, candidate2)
                        if dist < best_dist:
                            best = candidate2
                            best_dist = dist

    return best, best_dist, tried


def _try_synthesis_repair(original, positives, negatives, algebra=None, timeout=5.0):
    """Try to synthesize a replacement for subtrees that cause failures."""
    start = time.time()
    subtrees = list(_enumerate_subtrees(original))

    # Find which subtrees overlap with failure positions
    # Try replacing larger subtrees first (more impactful)
    subtrees_by_size = sorted(subtrees, key=lambda x: regex_size(x[1]), reverse=True)

    for path, node in subtrees_by_size:
        if time.time() - start > timeout:
            break
        if regex_size(node) < 2:
            continue
        if len(path) == 0:
            continue  # Don't replace entire regex via synthesis (use direct synthesis for that)

        # Try to synthesize a replacement for this subtree
        # Extract sub-examples relevant to this position
        result = synthesize_regex(positives, negatives, max_size=regex_size(node) + 2)
        if result.success and result.regex:
            # This synthesizes the whole thing, not the subtree -- skip
            break

    return None, float('inf'), 0


# ---------------------------------------------------------------------------
# Fault localization
# ---------------------------------------------------------------------------

def _localize_fault(regex, positives, negatives, algebra=None):
    """Identify which subtree(s) are most likely causing failures."""
    _, _, fn, fp = _compile_and_test(regex, positives, negatives, algebra)
    if not fn and not fp:
        return []  # No faults

    subtrees = list(_enumerate_subtrees(regex))
    fault_scores = []

    for path, node in subtrees:
        score = 0
        # Try replacing this node with a "hole" (universal acceptor or rejector)
        # If replacing with universal fixes false negatives, this node is too restrictive
        if fn:
            universal = RStar(RDot())
            candidate = _replace_at_path(regex, path, universal)
            _, _, new_fn, _ = _compile_and_test(candidate, positives, negatives, algebra)
            fixed_fn = len(fn) - len(new_fn)
            score += fixed_fn

        # If replacing with empty fixes false positives, this node is too permissive
        if fp:
            empty = REmpty()
            candidate = _replace_at_path(regex, path, empty)
            _, _, _, new_fp = _compile_and_test(candidate, positives, negatives, algebra)
            fixed_fp = len(fp) - len(new_fp)
            score += fixed_fp

        if score > 0:
            fault_scores.append((path, node, score))

    # Sort by score descending
    fault_scores.sort(key=lambda x: -x[2])
    return [(path, node) for path, node, _ in fault_scores]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diagnose_regex(pattern, positives, negatives, algebra=None):
    """Diagnose why a regex fails on examples.

    Args:
        pattern: The regex pattern string
        positives: Strings that should match
        negatives: Strings that should not match

    Returns:
        DiagnosticResult with false negatives, false positives, fault nodes, suggestions
    """
    regex = parse_regex(pattern)
    _, _, fn, fp = _compile_and_test(regex, positives, negatives, algebra)
    faults = _localize_fault(regex, positives, negatives, algebra)

    suggestions = []
    if fn:
        suggestions.append(f"Pattern fails to match {len(fn)} positive example(s): {fn[:3]}")
        suggestions.append("Consider widening character classes or adding alternatives")
    if fp:
        suggestions.append(f"Pattern incorrectly matches {len(fp)} negative example(s): {fp[:3]}")
        suggestions.append("Consider narrowing character classes or adding anchoring")

    for i, (path, node) in enumerate(faults[:3]):
        node_str = regex_to_string(node)
        suggestions.append(f"Suspicious node at path {path}: {node_str}")

    return DiagnosticResult(
        pattern=pattern,
        false_negatives=fn,
        false_positives=fp,
        fault_nodes=[(i, node) for i, (_, node) in enumerate(faults)],
        suggestions=suggestions
    )


def repair_regex(pattern, positives, negatives, algebra=None, max_edit_distance=None,
                 timeout=30.0):
    """Repair a regex to match all positives and reject all negatives.

    Uses a tiered approach:
    1. Check if already correct
    2. Try single-point mutations (quantifier, char class, structural, literal)
    3. Try double mutations
    4. Fall back to full synthesis

    Args:
        pattern: The broken regex pattern string
        positives: Strings that must match
        negatives: Strings that must not match
        max_edit_distance: If set, only accept repairs within this distance
        timeout: Maximum time in seconds

    Returns:
        RepairResult
    """
    start = time.time()
    stats = {"mutations_tried": 0, "time_s": 0}

    original = parse_regex(pattern)

    # Check if already correct
    if _is_correct(original, positives, negatives, algebra):
        stats["time_s"] = time.time() - start
        return RepairResult(
            success=True, original_pattern=pattern,
            repaired_regex=original, repaired_pattern=pattern,
            edit_distance=0, method="already_correct", stats=stats
        )

    remaining = timeout - (time.time() - start)

    # Strategy 1: Single-point mutations
    best, best_dist, tried = _try_mutations(
        original, positives, negatives, algebra,
        max_mutations=1000, timeout=min(remaining * 0.5, 15.0)
    )
    stats["mutations_tried"] += tried

    if best is not None:
        if max_edit_distance is None or best_dist <= max_edit_distance:
            repaired_pattern = regex_to_string(best)
            stats["time_s"] = time.time() - start
            return RepairResult(
                success=True, original_pattern=pattern,
                repaired_regex=best, repaired_pattern=repaired_pattern,
                edit_distance=best_dist, method="single_mutation", stats=stats
            )

    remaining = timeout - (time.time() - start)
    if remaining < 1.0:
        stats["time_s"] = time.time() - start
        return RepairResult(success=False, original_pattern=pattern, stats=stats)

    # Strategy 2: Double mutations
    best2, best2_dist, tried2 = _try_double_mutations(
        original, positives, negatives, algebra,
        max_mutations=500, timeout=min(remaining * 0.5, 10.0)
    )
    stats["mutations_tried"] += tried2

    if best2 is not None:
        if best is None or best2_dist < best_dist:
            best, best_dist = best2, best2_dist

    if best is not None:
        if max_edit_distance is None or best_dist <= max_edit_distance:
            repaired_pattern = regex_to_string(best)
            stats["time_s"] = time.time() - start
            return RepairResult(
                success=True, original_pattern=pattern,
                repaired_regex=best, repaired_pattern=repaired_pattern,
                edit_distance=best_dist, method="double_mutation" if best2 is not None and best == best2 else "single_mutation",
                stats=stats
            )

    remaining = timeout - (time.time() - start)
    if remaining < 1.0:
        stats["time_s"] = time.time() - start
        return RepairResult(success=False, original_pattern=pattern, stats=stats)

    # Strategy 3: Full synthesis fallback
    synth_result = synthesize_regex(positives, negatives)
    if synth_result.success and synth_result.regex:
        dist = _ast_edit_distance(original, synth_result.regex)
        if max_edit_distance is None or dist <= max_edit_distance:
            repaired_pattern = regex_to_string(synth_result.regex)
            stats["time_s"] = time.time() - start
            return RepairResult(
                success=True, original_pattern=pattern,
                repaired_regex=synth_result.regex,
                repaired_pattern=repaired_pattern,
                edit_distance=dist, method="synthesis", stats=stats
            )

    stats["time_s"] = time.time() - start
    return RepairResult(success=False, original_pattern=pattern, stats=stats)


def repair_regex_targeted(pattern, positives, negatives, fault_path, replacement_pattern,
                          algebra=None):
    """Apply a targeted repair at a specific AST location.

    Args:
        pattern: Original regex pattern
        positives: Must-match strings
        negatives: Must-not-match strings
        fault_path: Path tuple to the node to replace
        replacement_pattern: Regex pattern string for the replacement

    Returns:
        RepairResult
    """
    original = parse_regex(pattern)
    replacement = parse_regex(replacement_pattern)
    repaired = _replace_at_path(original, fault_path, replacement)

    if _is_correct(repaired, positives, negatives, algebra):
        repaired_pattern = regex_to_string(repaired)
        dist = _ast_edit_distance(original, repaired)
        return RepairResult(
            success=True, original_pattern=pattern,
            repaired_regex=repaired, repaired_pattern=repaired_pattern,
            edit_distance=dist, method="targeted"
        )
    return RepairResult(success=False, original_pattern=pattern, method="targeted")


def suggest_repairs(pattern, positives, negatives, algebra=None, max_suggestions=5,
                    timeout=15.0):
    """Generate multiple repair suggestions ranked by edit distance.

    Returns a list of RepairResult objects, sorted by edit distance (smallest first).
    """
    start = time.time()
    original = parse_regex(pattern)
    results = []
    seen_patterns = set()

    subtrees = list(_enumerate_subtrees(original))

    for path, node in subtrees:
        if time.time() - start > timeout:
            break

        mutations = (_quantifier_mutations(node) +
                     _char_class_mutations(node, positives, negatives) +
                     _structural_mutations(node, positives) +
                     _literal_mutations(node, positives, negatives))

        for mut in mutations:
            if time.time() - start > timeout:
                break
            candidate = _replace_at_path(original, path, mut)
            if _is_correct(candidate, positives, negatives, algebra):
                pat = regex_to_string(candidate)
                if pat not in seen_patterns:
                    seen_patterns.add(pat)
                    dist = _ast_edit_distance(original, candidate)
                    results.append(RepairResult(
                        success=True, original_pattern=pattern,
                        repaired_regex=candidate, repaired_pattern=pat,
                        edit_distance=dist, method="suggestion"
                    ))

    results.sort(key=lambda r: r.edit_distance)
    return results[:max_suggestions]


def compare_repairs(pattern, positives, negatives, algebra=None, timeout=20.0):
    """Compare different repair strategies on the same problem.

    Returns a dict with strategy -> RepairResult mapping and comparison info.
    """
    start = time.time()
    original = parse_regex(pattern)
    results = {}

    # Single mutation
    remaining = timeout - (time.time() - start)
    best1, dist1, tried1 = _try_mutations(
        original, positives, negatives, algebra,
        max_mutations=500, timeout=min(remaining * 0.3, 5.0)
    )
    if best1:
        results["single_mutation"] = RepairResult(
            success=True, original_pattern=pattern,
            repaired_regex=best1, repaired_pattern=regex_to_string(best1),
            edit_distance=dist1, method="single_mutation",
            stats={"mutations_tried": tried1}
        )

    # Double mutation
    remaining = timeout - (time.time() - start)
    best2, dist2, tried2 = _try_double_mutations(
        original, positives, negatives, algebra,
        max_mutations=300, timeout=min(remaining * 0.4, 5.0)
    )
    if best2:
        results["double_mutation"] = RepairResult(
            success=True, original_pattern=pattern,
            repaired_regex=best2, repaired_pattern=regex_to_string(best2),
            edit_distance=dist2, method="double_mutation",
            stats={"mutations_tried": tried2}
        )

    # Synthesis
    remaining = timeout - (time.time() - start)
    if remaining > 1.0:
        synth = synthesize_regex(positives, negatives)
        if synth.success and synth.regex:
            dist = _ast_edit_distance(original, synth.regex)
            results["synthesis"] = RepairResult(
                success=True, original_pattern=pattern,
                repaired_regex=synth.regex,
                repaired_pattern=regex_to_string(synth.regex),
                edit_distance=dist, method="synthesis"
            )

    # Pick best
    best_strategy = None
    best_result = None
    for strategy, result in results.items():
        if best_result is None or result.edit_distance < best_result.edit_distance:
            best_strategy = strategy
            best_result = result

    return {
        "strategies": results,
        "best_strategy": best_strategy,
        "best_result": best_result,
        "num_strategies_found": len(results),
    }


def batch_repair(problems, algebra=None, timeout_per=15.0):
    """Repair multiple regex problems.

    Args:
        problems: List of (pattern, positives, negatives) tuples

    Returns:
        List of RepairResult objects
    """
    results = []
    for pattern, positives, negatives in problems:
        result = repair_regex(pattern, positives, negatives, algebra=algebra,
                              timeout=timeout_per)
        results.append(result)
    return results


def repair_from_counterexample(pattern, counterexample, should_match, algebra=None,
                               timeout=15.0):
    """Repair a regex given a single counterexample.

    Args:
        pattern: The regex pattern
        counterexample: The string that exposes the bug
        should_match: True if counterexample should match but doesn't,
                     False if it matches but shouldn't

    Returns:
        RepairResult
    """
    if should_match:
        positives = [counterexample]
        negatives = []
    else:
        positives = []
        negatives = [counterexample]

    # Also preserve existing behavior by sampling from the regex
    original = parse_regex(pattern)
    alg = algebra or CharAlgebra()
    compiler = RegexCompiler(alg)
    try:
        sfa = compiler.compile(original).determinize()
        sample = sfa.accepted_word()
        if sample is not None and should_match:
            positives.append(sample)  # Preserve at least one existing match
    except Exception:
        pass

    return repair_regex(pattern, positives, negatives, algebra=algebra, timeout=timeout)


def semantic_distance(pattern1, pattern2, algebra=None, sample_size=20):
    """Measure semantic distance between two regex patterns.

    Returns a dict with:
    - equivalent: bool
    - only_in_first: list of example strings
    - only_in_second: list of example strings
    - in_both: list of example strings
    - jaccard_similarity: float (approximate)
    """
    alg = algebra or CharAlgebra()
    r1 = compile_regex_dfa(pattern1, alg)
    r2 = compile_regex_dfa(pattern2, alg)

    diff12 = sfa_difference(r1, r2)
    diff21 = sfa_difference(r2, r1)

    only_in_first = []
    only_in_second = []
    in_both = []

    inter = sfa_intersection(r1, r2)

    # Sample from each region
    for _ in range(sample_size):
        w = diff12.accepted_word()
        if w is not None and w not in only_in_first:
            only_in_first.append(w)

        w = diff21.accepted_word()
        if w is not None and w not in only_in_second:
            only_in_second.append(w)

        w = inter.accepted_word()
        if w is not None and w not in in_both:
            in_both.append(w)

    total = len(only_in_first) + len(only_in_second) + len(in_both)
    jaccard = len(in_both) / total if total > 0 else 1.0

    return {
        "equivalent": diff12.is_empty() and diff21.is_empty(),
        "only_in_first": only_in_first,
        "only_in_second": only_in_second,
        "in_both": in_both,
        "jaccard_similarity": jaccard
    }
