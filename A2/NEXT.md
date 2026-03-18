# A2 NEXT -- What to do next

## Completed
- **V001: Abstract-Interpretation-Guided Symbolic Execution** (32/32 tests pass)
  - Composes C039 (abstract interp) + C038 (symbolic execution)
  - Abstract pre-analysis prunes infeasible branches before SMT checks
  - Interval analysis provides O(1) branch feasibility vs SMT's exponential
  - Key fix: symbolic inputs must be forced to TOP after LetDecl/Assign

- **V002: Property-Directed Reachability (PDR/IC3)** (44/44 tests pass)
  - Unbounded model checking using C037 SMT solver
  - Proves properties hold for ALL reachable states (not just bounded depth)
  - Adapted for infinite-state (LIA) systems with:
    - Formula-based cubes (not just propositional literals)
    - Pre-image-based blocking clause generalization
    - Complement-operator negation (workaround for NOT(EQ) SMT bug)
  - Handles deterministic and nondeterministic transitions, boolean + integer vars,
    conditional (ITE) transitions, multi-variable systems

- **V003: Type-Aware Symbolic Execution** (52/52 tests pass)
  - Composes C013 (type checker) + C038 (symbolic execution)
  - Static type analysis feeds into symbolic execution:
    - Auto-detects symbolic inputs from function parameter types
    - Injects type invariants as SMT constraints (bool domain: 0/1, int ranges)
    - Prunes paths that violate type constraints
    - Generates type-annotated test cases with correct coercion
  - Type warning detection: static type errors + runtime error paths
  - Key insight: SMT sort must match operation sort (use 'int' sort for bool
    vars that participate in arithmetic comparisons, let C013 provide domain)

- **V004: Verification Condition Generation** (66/66 tests pass)
  - Hoare-logic verification via Weakest Precondition calculus
  - Composes C010 (parser/AST) + C037 (SMT solver)
  - Features:
    - Annotation-based specs: requires(), ensures(), invariant(), assert()
    - WP calculus over full AST: assignments, sequencing, conditionals, loops
    - Automatic VC generation: precondition => WP(body, postcondition)
    - Loop invariant verification: preservation + postcondition establishment
    - Counterexample generation when VCs fail (SMT model extraction)
    - Direct Hoare triple API: verify_hoare_triple(P, S, Q)
  - Custom symbolic expression layer (SExpr) for WP computation, lowered to SMT
  - Handles: identity, increment, abs, max, classify, countdown, accumulator,
    swap, multi-assignment, nested conditionals, multiple functions

- **V005: Abstract-Interpretation-Strengthened PDR** (44/44 tests pass)
  - Composes C039 (abstract interp) + V002 (PDR/IC3) + C010 (parser)
  - Pipeline: Source -> Abstract Interpretation -> Candidate invariants -> PDR with seeded frames
  - Loop-to-transition-system extraction from C10 source
  - Abstract candidate extraction: interval/sign/constant bounds -> SMT formulas
  - Candidate validation: Init => candidate AND Trans preserves candidate
  - Frame seeding: validated candidates seed PDR frames, accelerating convergence
  - APIs: verify_loop(), ai_pdr_check(), verify_ts_with_hints(), compare_pdr_performance()
  - Bug: C037 has no DIV/MOD operators. Abstract interp exit-state intervals after
    widening may be BOT/TOP -- pre-loop init candidates more reliable.

- **V006: Equivalence Checking** (48/48 tests pass)
  - Composes C038 (symbolic execution) + C037 (SMT solver) + C010 (parser)
  - Proves two programs compute the same function via product construction:
    - Run symbolic execution on both programs with same symbolic inputs
    - For each path pair (p1, p2), check constraints(p1) AND constraints(p2) AND output(p1) != output(p2)
    - If any pair is SAT -> NOT equivalent (with counterexample)
    - If all pairs UNSAT -> EQUIVALENT
  - Features:
    - Function equivalence: compare return values of named functions
    - Program equivalence: compare output variables, print sequences, or env
    - Variable mapping: compare programs with renamed variables
    - Partial equivalence: restrict to a domain (e.g., x > 0)
    - Regression checking: verify refactored code matches original
    - Counterexample generation with concrete distinguishing inputs
    - Structural term equality: handles nonlinear terms (x*x) that LIA can't reason about
  - Tested: algebraic identities (commutativity, associativity, distributivity),
    abs/max/min/sign/clamp implementations, refactoring (extract variable, inline,
    strength reduction), multi-argument functions, nested conditionals

- **V007: Automatic Loop Invariant Inference** (44/44 tests pass)
  - Composes V004 (VCGen) + V005 (AI-PDR) + V002 (PDR) + C039 + C037 + C010
  - Closes the loop on V004: automatic Hoare-logic verification without manual invariants
  - Tiered candidate generation:
    1. Abstract interpretation -> interval/sign/constant bounds
    2. Init-value bounds -> upper/lower from pre-loop assignments
    3. Condition weakening -> e.g., `i > 0` weakens to `i >= 0`
    4. Relational templates -> sum conservation (x+y==c), difference conservation
    5. PDR discovery -> expensive fallback for complex properties
  - Guarded transition system: transitions fire only under loop condition
  - All candidates validated as inductive (init + preservation check)
  - APIs: infer_loop_invariants(), auto_verify_function(), auto_verify_program()
  - Deepest composition yet: 6 systems (C010, C037, C039, V002, V004, V005)

- **V008: Bounded Program Synthesis** (53/53 tests pass)
  - CEGIS (Counterexample-Guided Inductive Synthesis) over program templates
  - Composes C010 (parser) + C037 (SMT solver) + V004 (VCGen)
  - Template hierarchy: linear (L1) -> conditional (L2) -> nested conditional (L3)
  - Spec-based synthesis: `synthesize_with_spec(pre, post, params)` -> verified program
  - Example-based synthesis: `synthesize_from_examples(params, examples)` -> program
  - CEGIS loop: SMT finds candidate matching test I/Os, V004 verifies, counterexample refines
  - Key fix: enumerate operators concretely (one template per op) instead of encoding
    operator selection as SMT ITE chains. C037 returns UNKNOWN on nested ITEs but
    handles pure LIA queries perfectly. This reduced test time from 265s to 12s.
  - Synthesizes: identity, increment, double, negate, sum, difference, linear combos,
    abs, max, min, sign, clamp, constants, with preconditions, from examples

- **V010: Predicate Abstraction + CEGAR** (47/47 tests pass)
  - Composes C037 (SMT solver) + V002 (PDR/IC3) + C010 (parser)
  - Cartesian predicate abstraction + counterexample-guided refinement
  - Auto predicate generation, WP-based refinement, source-level API
  - Key fix: use INT 0/1 vars for abstract system (V002 BOOL Var identity bug)
  - Limitation: Cartesian loses predicate correlations; multi-step violations
    may require many refinement rounds or full Boolean abstraction

- **V009: Differential Symbolic Execution** (50/50 tests pass)
  - Composes C038 (symbolic execution) + C037 (SMT solver) + C010 (parser) + V006 (equiv check)
  - Change-aware analysis: AST-level structural diff + focused path comparison
  - AST diff via LCS alignment of statement signatures
  - Focused mode skips path pairs not touching changed regions (O(n*m) -> O(k*l) where k,l are changed paths)
  - Features:
    - Program diff: `diff_programs(old, new, inputs)` -> DiffResult
    - Function diff: `diff_functions(old, fn1, new, fn2, params)` -> DiffResult
    - Regression checking: `check_regression(old, new, inputs)` -> DiffResult
    - Semantic diff: `semantic_diff(old, new, inputs)` -> ChangeSummary
    - Change impact analysis: `change_impact_analysis(old, new, inputs)` -> dict
    - Domain-constrained diff: `diff_with_constraints(old, new, inputs, domain)` -> DiffResult
  - Detects: value changes, structural changes, conditional changes, cancelling changes,
    syntactic-only changes (no behavioral impact), partial changes
  - Key fix: C010 AST field names differ from what you'd expect:
    LetDecl.value (not .init), Assign.name (str, not .target ASTVar),
    IfStmt.cond/.then_body/.else_body (not .condition/.then_branch/.else_branch)

- **V011: Refinement Type Checking (Liquid Types)** (66/66 tests pass)
  - Composes C013 (type checker) + C037 (SMT solver) + V004 (SExpr/WP) + C010 (parser)
  - Refinement types: base types augmented with logical predicates {v: int | v >= 0}
  - Subtype checking via SMT implication: {v|P} <: {v|Q} iff NOT(P AND NOT(Q)) is UNSAT
  - Path-sensitive: branch conditions strengthen refinement context
  - Dependent return types: return refinement can reference parameters
  - Features:
    - Refined base types with convenience constructors (nat, pos, range, eq)
    - Refined function signatures (parameter + return types)
    - Automatic refinement inference for let-bindings and arithmetic
    - Source-level annotation API (requires/ensures extraction)
    - Call-site argument type checking
    - Counterexample generation on subtype failures
    - While loop handling (weakens modified variables)
  - APIs:
    - `check_refinements(source, specs)` -> CheckResult
    - `check_program_refinements(source)` -> CheckResult (annotation-based)
    - `check_function_refinements(source, fn, params, ret)` -> CheckResult
    - `check_subtype_valid(sub, sup, assumptions)` -> SubtypeResult
    - `infer_refinement(source, var)` -> RefinedType
  - Key fix: Var expression type must include `v == name` to connect the value
    to the variable name in subtype checks. Without this, path conditions
    (e.g., `x >= 0` in then-branch) can't constrain the return value.

- **V012: Craig Interpolation** (58/58 tests pass)
  - Composes C037 (SMT solver) via Fourier-Motzkin elimination + model-guided probing
  - Craig interpolant: given A AND B is UNSAT, find I where A=>I and I AND B is UNSAT
  - Three interpolation strategies (tried in order):
    1. Syntactic: Fourier-Motzkin elimination of A-local vars from A's linear constraints
    2. Model-based: probe shared-variable bounds/relations implied by A that contradict B
    3. Trivial: A-UNSAT->False, B-UNSAT->True, or A/B directly if only shared vars
  - Sequence interpolation for CEGAR traces: A0,...,An UNSAT -> I1,...,In-1
  - CEGAR integration: `interpolation_refine(trace)` -> new predicates
  - Predicate extraction: decompose interpolant into atomic predicates
  - Features:
    - Linear constraint normalization and Fourier-Motzkin elimination
    - Variable classification (A-local, B-local, shared)
    - Bound probing via binary search on SMT queries
    - Relational discovery (equality, ordering, sum/difference conservation)
    - Full interpolant validity verification (3 conditions)
  - APIs:
    - `interpolate(a, b)` -> Interpolant (main API)
    - `sequence_interpolate(formulas)` -> SequenceInterpolant
    - `interpolation_refine(trace)` -> List[Term] (CEGAR helper)
    - `check_and_interpolate(a_terms, b_terms)` -> Interpolant
    - `interpolate_with_vars(a, b, a_vars, b_vars)` -> Interpolant
    - `extract_predicates_from_interpolant(interp)` -> List[Term]
    - `check_interpolant_validity(a, b, i)` -> (bool, bool, bool)

- **V014: Interpolation-Based CEGAR** (50/50 tests pass)
  - Composes V012 (Craig interpolation) + V010 (predicate abstraction CEGAR) + V002 (PDR) + C037 + C010
  - Replaces V010's WP-based refinement with interpolation-based refinement:
    - BMC unrolling along abstract trace -> sequence interpolants -> new predicates
    - Interpolants capture multi-step relationships (not just one-step WP)
    - Falls back to WP refinement when interpolation fails
  - Also provides direct interpolation-based MC (BMC + interpolation, no pred abstraction)
  - Comparison API: side-by-side V010 vs V014 strategy comparison
  - APIs:
    - `interp_cegar_check(ts, preds)` -> InterpCEGARResult (main CEGAR loop)
    - `interp_model_check(ts, max_depth)` -> InterpCEGARResult (direct MC)
    - `interpolation_refine_trace(ts, trace, preds)` -> new predicates
    - `verify_loop_interp(source, property)` -> InterpCEGARResult (source-level)
    - `verify_loop_direct(source, property)` -> InterpCEGARResult (source-level)
    - `compare_refinement_strategies(ts)` -> comparison dict
  - Known limitation: Cartesian abstraction (from V010) still loses predicate
    correlations. Full Boolean abstraction would fix but is exponential.

- **V015: k-Induction Model Checking** (46/46 tests pass)
  - Composes C037 (SMT solver) + V002 (TransitionSystem)
  - k-Induction: base case (BMC for k steps) + inductive step (k consecutive
    property-holding states imply property at step k+1)
  - Features:
    - Fixed-k check: `k_induction_check(ts, k)` -> KIndResult
    - Incremental search: `incremental_k_induction(ts, max_k)` finds minimal k
    - Invariant strengthening: `k_induction_with_strengthening(ts, max_k, invariants)`
    - Path uniqueness variant for finite-state convergence
    - BMC-only mode: `bmc_check(ts, max_depth)` for bug-finding
    - Source-level: `verify_loop(source, property)` + `verify_loop_with_invariants(...)`
    - PDR comparison: `compare_with_pdr(ts)` benchmarks both approaches
  - Counterexample traces with full state at each step
  - Tests: 1-inductive, k-inductive, immediate violations, multi-var, conditional,
    nondeterministic, source-level, counterexample validation, PDR comparison, edge cases
  - Key lesson: C010 uses IntLit (not Num); PDR returns PDRResult enum (compare .value)

- **V016: Auto-Strengthened k-Induction** (42/42 tests pass)
  - Composes V015 (k-induction) + V007 (invariant inference) + V002 (PDR) + C037 + C010
  - Pipeline: plain k-induction -> if UNKNOWN, auto-infer invariants -> retry with strengthening
  - TS-level invariant inference: non-negativity, init bounds, sum/diff conservation
  - Source-level: integrates V007's full tiered inference (abstract interp, condition
    weakening, relational templates, PDR)
  - Subset search: if all invariants together fail, tries leave-one-out and individual
  - APIs:
    - `auto_k_induction(ts, max_k, source, property_sexpr)` -> AutoKIndResult
    - `verify_loop_auto(source, property)` -> AutoKIndResult (source-level)
    - `verify_loop_auto_with_hints(source, property, hints)` -> AutoKIndResult
    - `compare_strategies(ts)` / `compare_with_source(source, property)` -> comparison dict
  - Tests: 1-inductive, violations, strengthening-needed, TS inference, source-level,
    hints API, comparison, multi-var, conditional bodies, guarded transitions, edge cases

- **V017: Abstract Domain Composition (Reduced Product)** (79/79 tests pass)
  - Composes sign + interval + constant + parity domains in a reduced product
  - Cross-domain reduction: each domain tightened by info from all others
  - New parity domain: even/odd tracking with full arithmetic transfer functions
  - Parity-interval interaction: [1,5]+EVEN -> [2,4], conflict detection
  - 4-domain ExtendedInterpreter with reduction at every statement
  - Comparison API: baseline C039 vs composed V017
  - APIs: composed_analyze(), get_variable_info(), compare_analyses(), get_precision_gains()

- **V018: Concolic Testing** (53/53 tests pass)
  - Concrete + symbolic execution for automated test generation
  - Concrete interpreter with symbolic shadow state tracking
  - Constraint negation + SMT solving for directed path exploration
  - Coverage-guided prioritization (uncovered branches first)
  - Bug finding: division-by-zero, assertion checking
  - Seed-based testing, branch targeting, comparison with C038
  - APIs: concolic_test(), concolic_find_bugs(), concolic_with_seeds(),
    concolic_reach_branch(), compare_concolic_vs_symbolic()
  - Bug fix: SMT model() returns string keys, not Var objects

- **V019: Widening with Thresholds** (55/55 tests pass)
  - Composes C039 (abstract interpreter) + C010 (parser)
  - Threshold-based widening: instead of jumping to infinity, widen to next
    program-derived threshold (constants, comparison operands, boundaries)
  - Automatic threshold extraction from AST + user-provided thresholds
  - Narrowing pass after widening for extra precision
  - ThresholdEnv preserves thresholds through join/widen/copy
  - Per-variable threshold extraction from comparisons and assignments
  - Comparison API: standard C039 vs threshold widening side-by-side
  - APIs: threshold_analyze(), compare_widening(), get_variable_range(),
    get_thresholds(), get_variable_thresholds()
  - Bug fix: AbstractEnv join/widen must union all domain keys (signs, intervals,
    consts), not just signs.keys()

- **V020: Abstract Domain Functor** (100/100 tests pass)
  - Composable algebra of abstract domains with ABC protocol
  - Concrete domains: Sign, Interval, Constant, Parity, Flat
  - Domain functors: ProductDomain, ReducedProductDomain, PowersetDomain
  - Standard reducers: sign<->interval, const->all, interval->const, parity<->interval
  - FunctorInterpreter: generic C10 interpreter parameterized by domain factory
  - Factory combinators: make_sign_interval(), make_full_product(), create_custom_domain()
  - APIs: analyze_with_domain(), analyze_sign/interval/sign_interval/full(),
    compare_domains(), get_variable_info(), create_custom_domain()
  - Bug fix: ReducedProductDomain must call _reduce in __init__, not just after operations

- **V021: BDD-based Symbolic Model Checking** (69/69 tests pass)
  - Full BDD library: canonical OBDDs with apply, restrict, compose, quantification
  - Symbolic model checker: forward/backward reachability, fixpoint iteration
  - Full CTL model checking: EX, AX, EF, AG, AF, EG, EU, AU, ER, AR
  - V002 TransitionSystem conversion: integer vars -> bit-vectors
  - Comparison API: BDD vs PDR side-by-side
  - APIs: check_boolean_system(), check_ctl(), check_v002_system(), compare_with_pdr()
  - App constructor: `App(Op.EQ, [a, b], BOOL)` -- args is a list, sort is 3rd arg

- **V022: Trace Partitioning** (55/55 tests pass)
  - Composes V020 (domain functor) + C039 (abstract interp) + C010 (parser)
  - Maintains separate abstract states per execution trace instead of merging at joins
  - Branch partitioning: each if-then-else creates separate partitions
  - Loop partitioning: optional unrolling of first N iterations as separate partitions
  - Budget control: configurable max partitions with automatic merging
  - Infeasible branch pruning: BOT partitions eliminated
  - Works with any V020 domain factory (sign, interval, full product, custom)
  - Comparison API: standard vs partitioned precision analysis
  - Statistics tracking: branches partitioned/merged, max partitions, etc.
  - APIs: trace_partition_analyze(), trace_partition_full(), compare_precision(),
    get_variable_partitions(), analyze_with_loop_partitioning(), analyze_branches_only()

- **V023: LTL Model Checking** (54/54 tests pass)
  - Composes V021 (BDD-based model checking)
  - LTL formula AST + text parser (G, F, X, U, R, W operators)
  - NNF conversion with temporal dualities
  - Tableau-based GBA construction with obligation tracking
  - GBA to NBA degeneralization (product with counter)
  - BDD-based product construction (system x automaton)
  - Fair cycle detection via Emerson-Lei nested fixpoint
  - Fairness constraints: justice (GF(p)) and compassion (GF(p)->GF(q))
  - APIs: check_ltl(), check_ltl_fair(), check_ltl_boolean(), parse_ltl(), etc.
  - Bug fixes: BDD TRUE/FALSE are instance attrs; labels need pos+neg atoms

- **V024: Symbolic Abstraction** (63/63 tests pass)
  - Composes C038 (symbolic execution) + C037 (SMT solver) + C010 (parser) + V002 (TS)
  - Computes OPTIMAL abstract transformers for predicate domains: alpha . f . gamma
  - Symbolic execution explores all concrete paths; SMT evaluates predicates in post-states
  - Strictly more precise than Cartesian abstraction (V010): preserves predicate correlations
  - Features:
    - Symbolic abstract post: `symbolic_abstract_post(source, preds, inputs)` -> PredicateState
    - Pre-state constrained analysis (extra SMT constraints from pre-state predicates)
    - Predicate discovery: extracts from path conditions, branch comparisons, source analysis
    - Full program analysis: `symbolic_abstraction_analyze(source, preds, inputs)`
    - Correlation detection: implication and exclusion between predicates across paths
    - Transition system abstraction: `compute_abstract_transformer(ts, preds)`
    - Cartesian comparison: `compare_with_cartesian()` / `compare_ts_abstraction()`
    - Auto mode: `auto_symbolic_abstraction(source, inputs)` discovers preds + analyzes
    - Verification: `verify_with_symbolic_abstraction(source, prop_pred)` -> verdict
    - Auto input detection from AST (variables used before assignment)
  - Precision gains over Cartesian: detects "if P1 then P2" relationships that
    Cartesian abstraction loses when joining independently
  - Key fixes:
    - SMT Var requires sort argument: `Var(name, INT)` not `Var(name)`
    - TransitionSystem uses builder pattern: `add_int_var()`, `set_init()`, etc.
    - Pre-state constraints must be passed to predicate evaluation (not just path filtering)
    - Abstract post from predicate region considers ALL matching states, not just init

- **V025: Termination Analysis** (53/53 tests pass)
  - Proves program termination by discovering ranking functions
  - Composes C010 (parser) + C037 (SMT solver) + C039 (abstract interp)
  - Template-based ranking function discovery:
    1. Condition-derived: extract ranking structure from loop condition
    2. Single-variable: R = x, R = c - x, etc.
    3. Two-variable: linear combinations c0 + c1*x1 + c2*x2
  - SMT verification of ranking conditions:
    - Bounded: cond(s) => R(s) >= 0
    - Decreasing: cond(s) AND Trans(s,s') => R(s) - R(s') >= 1
  - Lexicographic ranking: tuples (R1, R2) where R1 decreases or stays same + R2 decreases
  - Conditional ranking: different ranking expressions for if-then-else branches
  - Nontermination detection: BMC-based reachable fixed-point search
  - AI-enhanced search: abstract interpretation provides variable bound estimates
  - Multi-loop program analysis: analyze_termination() checks all loops
  - APIs: find_ranking_function(), prove_termination(), check_ranking_function(),
    analyze_termination(), verify_terminates(), verify_all_terminate(),
    find_lexicographic_ranking(), find_conditional_ranking(),
    detect_nontermination(), compare_ranking_strategies()

- **V026: Information Flow / Taint Analysis** (66/66 tests pass)
  - Three analysis modes for tracking secret data through programs:
    1. Abstract taint analysis: fast, over-approximate, detects direct + implicit flows
    2. Symbolic taint analysis: path-sensitive, precise (no false positives from infeasible paths)
    3. Noninterference checking: proves high inputs can't affect low outputs
  - Composes C038 (symbolic execution) + C037 (SMT solver) + C010 (parser)
  - Features:
    - Security lattice: LOW (public) < HIGH (secret)
    - Direct flow detection: taint propagates through assignments and arithmetic
    - Implicit flow detection: taint propagates through branch conditions (if/while)
    - Print leak detection: tainted values reaching print() are flagged
    - Function taint tracking: inter-procedural taint propagation
    - Data dependency graph: transitive closure reachability queries
    - Declassification policies: allow specific flows (e.g., password -> hash)
    - Comparison API: abstract vs symbolic precision comparison
    - Full analysis: taint + noninterference + dependency graph combined
  - APIs: taint_analyze(), symbolic_taint_analyze(), check_noninterference(),
    build_dependency_graph(), compare_taint_analyses(), full_information_flow_analysis()
  - Key fixes:
    - C10 doesn't need `;` after closing `}` for if/while/fn blocks
    - High vars must stay tainted even when re-assigned (they are secret sources)
    - Symbolic taint is more precise: implicit flows through branches give concrete
      values per path, not symbolic dependencies

- **V027: Quantitative Abstract Interpretation** (71/71 tests pass)
  - Automatic resource bound analysis: loop iteration bounds + complexity classification
  - Composes V025 (termination/ranking functions) + C039 (abstract interp) + V019 (thresholds) + C037 (SMT) + C010 (parser)
  - Features:
    - Loop bound computation via ranking function initial values
    - Concrete bounds (constant loops) and parametric bounds (symbolic params)
    - Nesting-aware complexity: nested loops multiply, sequential loops add
    - Complexity classification: O(1), O(n), O(n^2), O(n^3)
    - Resource counting: assignments, comparisons, arithmetic ops, function calls
    - Bound verification: SMT-based checking of proposed bounds
    - Program comparison: side-by-side complexity analysis
    - Human-readable summary output
  - APIs: analyze_bounds(), loop_bound(), complexity_class(), resource_count(),
    verify_bound(), compare_bounds(), bound_summary()
  - Key fix: find_symbolic_params must track LetDecl order -- `n = n - 1` (Assign)
    doesn't introduce `n`, only `let n = ...` (LetDecl) does. Variables used before
    any LetDecl are symbolic parameters.

- **V028: Fault Localization** (67/67 tests pass)
  - Given a buggy program and tests, identify the most likely buggy statement
  - Composes C038 (symbolic execution) + C010 (parser) + C037 (SMT solver)
  - Techniques:
    1. Spectrum-Based Fault Localization (SBFL): Ochiai, Tarantula, DStar metrics
    2. Backward dependency slicing with control-flow parent tracking
    3. Symbolic fault localization via path-constraint analysis
    4. Combined ranking with slice-boosted suspiciousness
  - Concrete coverage tracer: executes programs, tracks per-statement coverage
  - Auto test generation: C038 generates tests, oracle classifies pass/fail
  - Full pipeline: auto_localize() generates tests -> SBFL -> slice -> combine
  - Evaluation: rank_at() and exam_score() for measuring localization quality
  - APIs: spectrum_localize(), backward_slice(), symbolic_localize(),
    generate_test_suite(), localize_fault(), auto_localize(), rank_at(), exam_score()
  - Key fix: CoverageTracer must map parsed AST nodes to statement indices via
    sequential flattening order, not object identity (re-parsing creates new objects)
  - Key fix: Backward slicing needs control-flow parent tracking -- statements inside
    if/while bodies depend on the enclosing condition even if they don't use any
    variables from it

- **V029: Abstract DPLL(T)** (55/55 tests pass)
  - CDCL-guided path-sensitive abstract interpretation for program verification
  - Composes C010 (parser) + C037 (SMT solver) + C039 (abstract domains)
  - Combines conflict-driven clause learning search with abstract interpretation as theory:
    - Branch decisions = Boolean variables (which direction at each if-statement)
    - Abstract interpretation = theory propagation (interval domain)
    - Assertion failures = conflicts (trigger clause learning)
    - CDCL learning = prune infeasible/failing branch combinations
  - Features:
    - Path-sensitive analysis: explores each branch combination separately
    - CDCL clause learning: conflicts produce clauses that prune future paths
    - Dependency-based conflict analysis: minimal clauses via variable dependency tracking
    - Var-vs-var condition refinement: full comparison support (< <= > >= == !=)
    - Infeasible branch detection: BOT intervals eliminate unreachable paths
    - While loop handling: abstract fixpoint with widening (sound, conservative)
    - SMT refinement: optional C037 check to distinguish real from spurious failures
    - Decision budget control: configurable max decisions for performance
    - Comparison API: side-by-side with standard C039 abstract interpretation
  - APIs:
    - `analyze_program(source, use_smt)` -> DPLLTResult (main API)
    - `verify_assertions(source)` -> (Verdict, messages)
    - `compare_with_standard_ai(source)` -> comparison dict
    - `analyze_with_budget(source, max_decisions)` -> DPLLTResult
  - Key fixes:
    - Infeasible branches are NOT assertion failures (just unreachable paths)
    - Conflict branch_decisions must include ALL current assignments, not just nesting path
    - Sort constructor: `Sort(SortKind.BOOL)` not `Sort.BOOL`
    - C10 requires `;` after expression statements inside blocks

- **V030: Shape Analysis** (103/103 tests pass)
  - TVLA-style heap abstraction with 3-valued logic (TRUE/FALSE/MAYBE)
  - Custom heap language: new, assign, load (x.next), store (x.next = y), null, if/while
  - Shape graphs: concrete + summary nodes, variable-pointing + next-edge predicates
  - Focus/Coerce/Blur operations:
    - Focus: materialize summary nodes for precise load/store
    - Coerce: tighten MAYBE via functionality constraints (non-summary -> 1 next)
    - Blur: merge indistinguishable nodes for convergence
  - Instrumentation predicates: is_null, is_shared, is_on_cycle, reachable
  - Assertions: assert_acyclic, assert_reachable, assert_not_null, assert_disjoint
  - While-loop fixpoint with join-based widening
  - Strong updates (definite targets) vs weak updates (MAYBE targets)
  - Null dereference detection (load/store through null pointer)
  - APIs: analyze_shape(), check_acyclic(), check_not_null(), check_shared(),
    check_reachable(), verify_shape(), get_shape_info(), compare_shapes()

- **V031: Separation Logic Prover** (106/106 tests pass)
  - Separation logic formulas: emp, points-to, separating conjunction (*), magic wand (-*)
  - Inductive predicates: lseg (list segments), tree (binary trees), dlseg (doubly-linked lists)
  - Predicate unfolding with base/step cases
  - Symbolic heap normalization (pure constraints + spatial atoms)
  - Entailment checking via atom matching + predicate unfolding + SMT pure checking
  - Frame inference: P |- Q * ?F (find leftover heap)
  - Bi-abduction: P * ?A |- Q * ?F (find missing pre + leftover frame)
  - Frame rule: {P} C {Q} => {P * R} C {Q * R}
  - Heap program verifier: new, assign, load, store, null, dispose
  - Null dereference and double-free detection
  - Composes C037 (SMT solver) for pure constraint reasoning
  - Key fix: SMTResult is an enum not string; RHS unfolding pures go to LHS context

- **V032: Combined Numeric+Shape Analysis** (103/103 tests pass)
  - Composes V030 (shape analysis) + numeric interval domain for programs with both heap and integers
  - Extended heap language with integer data fields, arithmetic, len(), sortedness checks
  - Combined abstract state: ShapeGraph + per-node data intervals + per-variable numeric intervals
  - List length tracking via BFS with summary node awareness
  - Sortedness analysis: checks data(n) <= data(n.next) across all edges
  - Data range verification: checks all node data values within bounds
  - Integer condition evaluation for precise branch pruning
  - Focus/blur operations propagate numeric data through materialization/merge
  - Loop analysis with interval widening for convergence
  - Null dereference detection on data access
  - Key fixes: integer condition definiteness check before branching,
    reachability assertion must iterate target nodes (not pass string to Node API)

- **V033: Python Code Analyzer** (50/50 tests pass)
  - First tool that operates on Python source (via ast module), not C10
  - Built to analyze A1's actual challenge codebase
  - Analyses: cyclomatic/cognitive complexity, nesting depth, data flow/taint,
    exception safety, unused imports, mutable defaults, inconsistent returns
  - Applied to A1's verification stack (8 files, 8292 lines, 489 functions)
  - Key findings on A1's code:
    - `stack_vm.py:run()` CC=51, Cognitive=749 (highest in entire stack)
    - `stack_vm.py:lex()` CC=52, Cognitive=392
    - `smt_solver.py:_check_theory()` CC=52
    - 40 unused imports across stack (17 in type_checker.py alone)
    - No mutable defaults, no bare excepts (good practices)
  - APIs: analyze_file(), analyze_directory(), analyze_a1_challenges(),
    taint_analysis(), complexity_report(), findings_report()

- **V034: Deep Python Taint Analysis** (65/65 tests pass)
  - Path-sensitive, inter-procedural taint tracking for Python source code
  - Extends V033 with:
    1. Path sensitivity: per-branch taint state (if/else fork+join)
    2. Inter-procedural: taint flows through function calls/returns via summaries
    3. Taint labels: track WHICH source tainted a value (not just yes/no)
    4. Sanitizer support: functions that clean taint (configurable)
    5. Implicit flows: taint from conditions (if tainted: x = safe_val)
    6. Walrus operator, f-string, method call, tuple unpack support
  - Worklist algorithm for inter-procedural fixpoint with function summaries
  - Auto-detection of sanitizer functions (param tainted -> return clean)
  - Configurable sources, sinks, sanitizers via TaintConfig
  - APIs: analyze_taint(), analyze_taint_file(), TaintConfig, TaintAnalysisResult
  - Bug fixes: method calls must propagate object taint; source_vars checked
    at read-time not just init-time; NamedExpr (walrus) handling

- **V035: Call Graph Analysis + Dead Code Detection** (57/57 tests pass)
  - Static call graph construction from Python source (ast module)
  - Qualified name resolution: Class.method, nested functions, lambdas
  - Dead code detection via entry-point reachability (BFS from entry points)
  - Entry point heuristics: main, __init__, test_, dunder protocol, callbacks, properties
  - Cycle detection: direct recursion + mutual recursion via Tarjan's SCC
  - Dependency layers (topological level ordering)
  - Fan-in/fan-out metrics, max call chain depth, orphan detection
  - Cross-file analysis via build_call_graph_from_directory()
  - Applied to A1's C035-C039: found recursive parsers, mutual recursion SCCs,
    dead code patterns across the verification stack

- **V037: Program Slicing** (67/67 tests pass)
  - Composes V035 (call graph) + def-use analysis for inter-procedural slicing
  - Core data structures: CFG, PDG (data + control dependence), SDG (system dependence graph)
  - Algorithms:
    - Backward slice (Weiser intraprocedural + HRB two-pass interprocedural)
    - Forward slice (follow data + control deps forward)
    - Chop (intersection of forward from source + backward from target)
    - Thin slice (data dependencies only, no control deps)
    - Diff slice (forward impact analysis from changed lines)
  - Analysis:
    - Reaching definitions (iterative dataflow)
    - Post-dominator-based control dependence
    - Def-use collection with tuple unpacking, augmented assignment, imports, for-loop vars
    - Inter-procedural: CALL, PARAM_IN, PARAM_OUT, SUMMARY edges via V035 call graph
  - Bug fix: thin slice criterion variables name what's DEFINED, not what's USED.
    Must check seed's defs overlap with criterion vars before filtering data edges.
  - APIs: backward_slice(), forward_slice(), chop(), thin_backward_slice(), diff_slice()
  - Utilities: extract_slice_source(), slice_report(), build_sdg(), build_pdg()

- **V036: Concurrent Separation Logic** (69/69 tests pass)
  - Extends V031 (Separation Logic) with concurrency primitives
  - Lock-based reasoning: lock invariants, acquire/release with CSL rules
  - Parallel composition rule: {P1} C1 {Q1}, {P2} C2 {Q2} => {P1*P2} C1||C2 {Q1*Q2}
  - Data race detection: write-write, write-read, heap races
  - Lock protection: races suppressed for variables under lock invariants
  - Ownership analysis: thread-local vs shared vs protected resources
  - Rely-guarantee reasoning: guarantee-rely compatibility checking
  - Atomic blocks, sequential composition with midpoint inference
  - APIs: verify_concurrent(), check_race_freedom(), ownership_analysis(),
    check_rely_guarantee()
  - Bugs fixed: forward_interpret must not mutate held_locks (copy needed),
    race detection must scan branches for lock acquisitions (not just outer scope),
    races detected within both branches = protected

- **V038: LTL Model Checking** (62/62 tests pass)
  - Composes V021 (BDD) with automata-theoretic LTL model checking
  - LTL formula AST with full operator set (X, F, G, U, R, W, boolean)
  - NNF conversion with deMorgan duality for temporal operators
  - Tableau construction: LTL -> Generalized Buchi Automaton
  - Product system (System x Buchi) encoded symbolically in BDDs
  - Emerson-Lei fair cycle detection for acceptance conditions
  - Handles properties CTL cannot express: G(F p), F(G p), nested temporal
  - Counterexample extraction (lasso-shaped traces)
  - High-level APIs: check_ltl() and check_ltl_simple()
  - Tested: safety, liveness, fairness, response, mutual exclusion, nondeterminism

- **V039: Modular Verification (Contracts)** (48/48 tests pass)
  - Composes V004 (VCGen) + C010 (parser) + C037 (SMT solver)
  - Function-level contracts with compositional inter-procedural verification
  - Contract extraction: requires(), ensures(), modifies() annotations
  - Modular WP: at call sites, check callee precondition + assume postcondition
    - Key design: call preconditions embedded in WP (not accumulated separately)
      so earlier postconditions from the WP chain are available as assumptions
  - Modifies clauses: frame conditions (unmodified vars preserved)
  - Contract refinement: behavioral subtyping (weaker pre, stronger post)
  - Whole-program compositional verification with dependency ordering
  - External contract API: verify_with_contracts(), verify_against_spec()
  - Call graph analysis for verification order (callees before callers)
  - APIs:
    - `verify_modular(source)` -> ModularResult (main API)
    - `verify_function_modular(source, fn_name, store)` -> VerificationResult
    - `verify_with_contracts(source, contracts)` -> ModularResult
    - `verify_against_spec(source, fn_name, pre, post)` -> VerificationResult
    - `check_contract_refinement(old_source, new_source, fn_name)` -> RefinementResult
    - `check_call_safety(source, caller, callee)` -> VCResult
    - `check_frame_condition(source, fn_name, store)` -> list[VCResult]
    - `extract_all_contracts(source)` -> ContractStore
    - `summarize_contracts(source)` -> dict
    - `get_verification_order(source)` -> list[str]
  - Key insight: call preconditions must be conjuncted into the WP, not checked
    separately. WP(let y = f(x), Q) = pre_f(x) AND (post_f(x, y) => Q).
    Otherwise, second call's precondition can't use first call's postcondition.

- **V040: Effect Systems** (54/54 tests pass)
  - Algebraic effect system for C10 programs
  - Composes V004 (SExpr/WP) + V039 (contracts) + C010 (parser) + C037 (SMT)
  - Features:
    - Effect representation: State(var), Exn(type), IO, Div, Nondet, Pure
    - Effect inference: analyze AST to compute minimal effect set per function
    - Effect checking: declared vs inferred effects (soundness + precision)
    - Effect verification: frame conditions from State, exception safety from Exn
    - Effect composition: sequencing, masking/handling, subtyping
    - Effect polymorphism: EffectVar + PolyEffectfulType with instantiation
    - Fixpoint iteration for recursive functions (optimistic seed)
    - Division safety: literal divisor analysis (nonzero literal = safe)
    - State wildcard: State("*") covers all specific State effects
  - API: `infer_effects(source)`, `check_effects(source, declared)`, `verify_effects(source, declared)`
  - Key insight: seed recursive functions with Pure effects and iterate upward
    to fixpoint. Unknown callees get conservative (all effects), but declared
    functions start optimistic and converge monotonically.

- **V041: Symbolic Debugging** (39/39 tests pass)
  - Given a buggy program + assertion, find minimal counterexample + rank suspicious statements
  - Composes C038 (symbolic execution) + V028 (SBFL) + V004 (WP/SExpr) + C010 (parser) + C037 (SMT)
  - Features:
    - Counterexample extraction: concrete inputs that trigger assertion failure
    - Minimal counterexample: shortest path to failure
    - Spectrum-based fault localization: Ochiai, Tarantula, DStar metrics
    - Backward dependency slicing from failure point
    - Combined ranking: SBFL score boosted by backward slice membership
    - Full trace with step-by-step execution
  - API: `symbolic_debug(source, symbolic_inputs)`, `find_minimal_counterexample(source, inputs)`
  - Key limitation: C038 drops assertion failures inside user-defined functions
    (only top-level assertions work). Workaround: assert at call sites, not inside callees.

- **V042: Dependent Types** (54/54 tests pass)
  - Types that depend on values: NonZero, Positive, NonNeg, Bounded(lo,hi), Equal(val), Array(len)
  - Composes V004 (SExpr) + C037 (SMT) + C010 (parser)
  - Features:
    - Type constructors parameterized by values (int literals or variables)
    - SMT-based subtype checking: sub <: sup iff UNSAT(sub AND NOT sup)
    - Subtype lattice: Equal(5) <: Positive <: NonZero, Positive <: NonNeg
    - Bounded range containment: Bounded(2,8) <: Bounded(0,10)
    - Context-dependent subtyping: Bounded(0,x) <: NonNeg under x > 0
    - Dependent type checker: infers types for expressions, checks division safety
    - Type inference: Positive + Positive = Positive, NonNeg * NonNeg = NonNeg
    - Division safety: warns when divisor not provably NonZero
    - Counterexample generation on subtype failure
    - Dependent function types: (n: NonNeg) -> Array(n)
  - API: `check_dependent_types(source, declared)`, `is_subtype(sub, sup, context)`

- **V043: Concurrency Verification Composition** (63/63 tests pass)
  - Composes V036 (concurrent separation logic) + V040 (effects) + V023 (LTL)
  - Three orthogonal analyses unified:
    - Effects: WHAT each thread does (state, IO, exceptions, divergence)
    - CSL: IF resource access is safe (ownership, lock protection)
    - LTL: Execution ORDER (mutual exclusion, deadlock, starvation)
  - Features:
    - Effect-guided race analysis: infer State effects per thread, cross-reference for shared state
    - Effect checking: declared vs inferred effects per thread
    - Boolean transition systems for concurrent protocols (none, lock, Peterson's)
    - LTL model checking: mutual exclusion, deadlock freedom, starvation freedom
    - Unified pipeline: effects + CSL + temporal in single verification pass
    - Protocol comparison: effect analysis guides protocol selection
  - APIs: `verify_concurrent_program()`, `verify_mutual_exclusion()`,
    `verify_concurrent_effects()`, `full_concurrent_verify()`,
    `effect_guided_protocol_selection()`
  - Key lessons:
    - BDD.var() takes index not name; use bdd.var(bdd.var_index(name))
    - V023 trans_fn nxt dict has primed keys ("x'"); remap to unprimed for uniform access
    - C10 print requires parens: print(x) not print x

- **V044: Proof Certificates** (56/56 tests pass)
  - Composes V004 (VCGen) + V002 (PDR) + C037 (SMT)
  - Machine-checkable proof certificates that can be independently verified
  - PDR certificates: 3 obligations (initiation, consecution, property)
  - VCGen certificates: per-VC obligations from WP calculus
  - Composite certificates: combine multiple sub-proofs
  - Full I/O: JSON serialization, save/load/re-verify roundtrip
  - SExpr-to-SMT-LIB2 and SMT-Term-to-SMT-LIB2 serializers
  - S-expression parser for formula reconstruction from strings
  - APIs: `generate_vcgen_certificate()`, `generate_pdr_certificate()`,
    `check_certificate()`, `combine_certificates()`, `certify_program()`,
    `certify_transition_system()`, `save_certificate()`, `load_certificate()`

- **V045: Concurrent Effect Refinement** (48/48 tests pass)
  - Composes V043 (concurrency verification) + V011 (refinement types)
  - Per-thread refinement checking via V011
  - Cross-thread contract verification (producer post => consumer pre) via SMT
  - Effect-refinement consistency checking
  - Lock-protected invariant verification
  - Effect-aware subtype checking (refinement + effect subtyping)
  - Unified pipeline with CERVerdict
  - APIs: `verify_concurrent_refinements()`, `verify_thread_pair()`,
    `verify_with_lock_protocol()`, `infer_thread_contract()`,
    `check_cross_thread_contract()`, `check_lock_invariants()`

- **V046: Certified Abstract Interpretation** (66/66 tests pass)
  - Composes V044 (proof certificates) + C039 (abstract interpreter)
  - Generates machine-checkable certificates proving AI results are sound
  - Per-variable interval bound obligations (non-empty, well-formed)
  - Sign-interval consistency obligations (interval => sign)
  - Widening soundness certificates (widened subsumes pre-widening)
  - Composite certificates (analysis + widening combined)
  - Convenience APIs: certify_variable_bound(), certify_sign()
  - APIs: `generate_ai_certificate()`, `verified_analyze()`,
    `generate_widening_certificate()`, `certify_abstract_interpretation()`,
    `certify_variable_bound()`, `certify_sign()`

- **V047: Incremental Verification** (37/37 tests pass)
  - Composes V004 (VCGen) + V044 (proof certificates) + C10 AST diff
  - AST-level diffing: function signatures detect added/removed/modified/unchanged
  - Certificate cache: valid certificates reused for unchanged functions
  - IncrementalVerifier: stateful engine accumulating cache across versions
  - Sequence verification: verify_with_cache() processes version sequences
  - APIs: `diff_programs()`, `IncrementalVerifier.verify()`,
    `incremental_verify()`, `verify_with_cache()`, `diff_and_report()`

- **V048: Quantitative Information Flow** (55/55 tests pass)
  - Composes V034 taint concepts + C037 (SMT solver) for Python code
  - Measures HOW MUCH information leaks (in bits), not just whether
  - Security level lattice (HIGH/LOW) with direct + implicit flow tracking
  - Structural quantification: % (modulus), // (div), & (bitmask), comparisons, ternary
  - SMT-based counting for complex expressions (capped at 32 iterations)
  - Channel capacity, min-entropy leakage, noninterference checking
  - Declassification policies with bit budgets
  - Key fix: HIGH vars must stay HIGH even when assigned constants (source identity)
  - Key fix: implicit flow check must use stored value (after PC join), not raw value

- **V049: Verified Compilation** (67/67 tests pass)
  - Composes C014 (optimizer) + C010 (compiler/VM) + C037 (SMT) + V044 (proof certs)
  - Translation validation: per-pass proof obligations for all 6 optimization passes
  - Constant folding: SMT proofs for arithmetic, concrete verification for DIV/MOD/comparisons
  - Strength reduction: SMT proofs for algebraic identities (x*2=x+x, x+0=x, x*1=x)
  - Dead code elimination: BFS reachability certificate
  - Jump optimization: unconditional jump threading verification
  - Peephole: stack-effect equivalence proofs
  - Constant propagation: value-flow analysis with intervening-store checks
  - End-to-end dynamic validation (execute both, compare results)
  - Composite certificate via V044 combine_certificates()
  - CompilationValidator class with caching for batch validation
  - Key fix: compute expected fold results directly instead of matching after-stream

- **V050: Holistic Verification Dashboard** (63/63 tests pass)
  - Composes 8 V-challenges into unified verification pipeline
  - Analyses: V046 (certified AI) + V004 (VCGen) + V040 (effects) + V001 (guided symex)
    + V011 (refinement types) + V025 (termination) + V039 (modular verification)
    + V049 (verified compilation)
  - VerificationReport with per-analysis results, confidence score, combined certificates
  - Pipeline configs: fast (3 analyses), deep (all 8)
  - Robust: no analysis crash propagates, all errors caught gracefully
  - Key fix: EffectSet is not iterable -- use str(eff_set) instead

- **V051: Counterexample-Guided Optimization Verification** (58/58 tests pass)
  - Composes V049 (verified compilation) + V001 (guided symbolic execution)
  - Extracts counterexamples from failed SMT proof obligations
  - Cross-validates by executing with/without optimization
  - Classifies: CONFIRMED_BUG vs SPURIOUS (validation imprecision)
  - Guided test generation via V001 with boundary coverage analysis
  - Key fix: VM(chunk).run() not VM().run(chunk)

- **V052: Incremental Dashboard Verification** (44/44 tests pass)
  - Composes V050 (holistic dashboard) + V047 (incremental verification)
  - AST-level function signature diff for change detection
  - Per-function analysis cache with delta reporting
  - IncrementalDashboard stateful verifier, sequence verification

- **V053: Cross-Analysis Correlation** (44/44 tests pass)
  - Agreement/complementarity/redundancy mining across analysis pairs
  - Feature detection + recommendation engine + smart_verify()
  - Bug: requires/ensures are CallExpr at stmt level in C10 AST

- **V054: Verification-Driven Fuzzing** (43/43 tests pass)
  - Composes V001 (guided symex) + V028 (fault localization) + V018 (concolic testing)
  - 6-phase pipeline: symbolic -> concolic -> fault localization -> boundary -> mutation -> random
  - MutationEngine with 3 strength levels, boundary value extraction from AST
  - TargetedFuzzer for branch-specific and suspiciousness-guided fuzzing
  - Divergence detection against reference implementations
  - Bug: C10 ConcreteInterpreter doesn't inject inputs at let-init; built _FuzzInterpreter

- **V055: Modular Abstract Interpretation** (39/39 tests pass)
  - Composes C039 (abstract interp) + C010 (parser) + V019 (thresholds)
  - Per-function abstract interpretation with contract-based summaries
  - Call site summary injection, topological analysis order
  - Annotation extraction (requires/ensures -> interval bounds)
  - APIs: modular_analyze(), analyze_function(), get_all_summaries(), compare_analyses()

- **V056: Regression Verification** (36/36 tests pass)
  - Composes V047 (incremental verification) + V009 (differential symex) + V054 (fuzzing) + V044 (certs)
  - When code changes: AST diff -> cert reuse for unchanged -> symex diff for changed -> targeted fuzz
  - Certificate caching: unchanged functions skip re-verification (O(1) per function)
  - Differential symbolic execution detects behavioral changes with counterexamples
  - Targeted fuzzing on changed paths with mutation engine
  - RegressionVerifier class: stateful, tracks versions, accumulates cache
  - Comparison API: incremental vs full re-verification savings
  - Bug fix: PARTIAL_CHANGE from V009 also indicates regression (has behavioral diffs)
  - Bug fix: FuzzResult.has_bugs doesn't include "divergence" -- check findings directly
  - APIs: verify_regression(), check_regression(), check_regression_with_fuzz(),
    regression_report(), RegressionVerifier.verify(), compare_verification_strategies()

- **V057: Verified Refactoring** (63/63 tests pass)
  - Composes V006 (equivalence) + V047 (incremental) + V055 (modular AI) + V004 (VCGen) + V044 (certs) + C010
  - 3-analysis pipeline: SMT equivalence + abstract summary comparison + contract preservation
  - Refactoring classification: rename, extract, inline, simplify, add, remove, modify
  - Convenience APIs: verify_rename/extract/inline/simplification_refactoring()
  - compare_refactoring_strategies(): equiv-only vs summary-only vs combined
  - Bugs fixed: C10 lex() not Lexer class, FnDecl.body is Block (needs .stmts),
    CallExpr.callee is string not ASTVar, ProofObligation needs formula_smt

- **V058: Proof-Carrying Code** (50/50 tests pass)
  - Composes V044 (certificates) + V049 (verified compilation) + V055 (modular AI) + V004 + C010
  - Full PCC pipeline: source -> compile -> prove -> bundle -> serialize -> verify
  - 3 certificate types: contract compliance, bound safety, compilation safety
  - Bundle serialization (JSON), consumer-side independent verification
  - APIs: produce_pcc(), quick_pcc(), full_pcc(), verify_bundle(), save/load_bundle()
  - Clean first run, 50/50, zero bugs

- **V059: Verified Concurrency** (53/53 tests pass)
  - Composes V043 (concurrency verification) + V044 (proof certificates) + V058 (PCC)
  - PCC bundles for concurrent programs: effect, race, temporal, CSL certificates
  - Consumer-side verification, JSON serialization, protocol comparison
  - Bugs: C10 Chunk uses .code not .instructions; V043 check_temporal_properties
    expects List[Tuple[str, LTL]] not List[LTL]

- **V060: Probabilistic Verification** (49/49 tests pass)
  - Statistical model checking: Monte Carlo, SPRT, expected value checking
  - Wilson score CI, Chernoff-Hoeffding sample bounds, oracle-based properties
  - ProbabilisticExecutor: random input generation + C10 VM execution
  - SPRT early termination: accepts/rejects before exhausting max_samples
  - Lesson: Wilson CI for 200/200 all-pass has lower bound ~0.98 at 95% confidence;
    threshold must be <= CI lower bound for ACCEPT verdict

- **V061: Automatic Test Generation from Specifications** (107/107 tests pass)
  - Composes V004 (VCGen) + V001 (guided symex) + V054 (fuzzing) + C010 + C037
  - 6-phase pipeline: counterexample extraction, spec boundary analysis, SMT-generated
    valid inputs, symbolic execution path coverage, mutation fuzzing, random fill
  - SpecAnalyzer: extract boundaries from requires/ensures, check pre/post conditions
  - TestExecutor: run C10 VM with inputs, strip annotation calls, capture output
  - InputCombiner: cross-product of boundary values, deduplication
  - TestMinimizer: reduce failing inputs while preserving property
  - Bugs: SMTResult is enum not string, C10 VM(chunk) constructor, print needs parens,
    SUnaryOp('-', SInt(N)) for negative literals needs _as_const helper

- **V062: Abstract Conflict-Driven Learning (ACDL)** (54/54 tests pass)
  - CEGAR loop: V029 Abstract DPLL(T) + V012 Craig Interpolation
  - TraceEncoder: converts ConflictInfo to SMT formulas for interpolation
  - PredicateStore: deduplicates learned predicates, filters by variables
  - PredicateAbstraction: checks concrete states against learned predicates
  - Safe programs terminate in 1 iteration; unsafe returns counterexample
  - V029 can't track fn return values (TOP) -- known precision limitation

- **V063: Verified Probabilistic Programs** (57/57 tests pass)
  - Composes V004 (VCGen/SExpr) + V060 (statistical model checking) + C010 (parser/VM) + C037 (SMT)
  - Hoare-logic for probabilistic programs: {P} S {Q @ threshold}
  - Annotation system: requires(), ensures(), prob_ensures(postcond, threshold)
  - random(lo, hi) uniform integer randomness, direct sampling with V060 stats
  - Features: prob Hoare triples, expected value analysis, concentration bounds,
    randomized algorithm verification (+ amplification), independence testing
  - Architecture: direct sampling loop (not V060 ProbabilisticExecutor, which can't
    handle random() in C10). Uses V060's wilson_confidence_interval + sprt_test.
  - Bug: C10 base uses and/or/not, NOT &&/||/!

- **V064: Probabilistic Proof Certificates** (45/45 tests pass)
  - Composes V063 (verified probabilistic) + V044 (proof certificates)
  - ProbProofCertificate with deterministic obligations + statistical evidence
  - Independent checker: Wilson CI recomputation, consistency, SPRT cross-check
  - Chernoff-Hoeffding minimum sample bounds
  - JSON serialization, composite certificates, V044 bridge
  - Bug fix: V060 wilson_confidence_interval(n_total, n_successes) arg order

- **V065: Markov Chain Analysis** (58/58 tests pass)
  - Discrete-time Markov chain analysis (standalone)
  - Tarjan SCC for communication classes, state classification
  - Steady-state (power iteration + exact Gaussian elimination)
  - Absorption probabilities, expected hitting times
  - Chain constructors, property verification, simulation + comparison

- **V066: Markov Chain Verification** (64/64 tests pass)
  - Composes V065 (Markov chains) + C037 (SMT solver) + V044 (proof certificates)
  - SMT-verified steady-state, absorption, and hitting time bounds
  - Exact rational arithmetic encoding: probabilities as scaled integers in LIA
  - Stochasticity, irreducibility, state type, uniqueness, reachability verification
  - Proof certificates for all properties via V044
  - Convenience: certified_steady_state(), certified_absorption(), compare_numerical_vs_smt()
  - Key lesson: uniqueness requires nonlinear MUL (C037 can't handle), use structural proof

- **V067: PCTL Model Checking** (88/88 tests pass)
  - Probabilistic Computation Tree Logic model checking over discrete-time Markov chains
  - Composes V065 (Markov chain analysis) for chain operations
  - PCTL AST: true, false, atom, NOT, AND, OR, P>=p, P<=p, P>p, P<p, X, U, U<=k
  - Sugar: eventually(phi) = true U phi, always via complement, bounded_eventually
  - State formula checking: recursive descent through boolean/probability operators
  - Path formula checking:
    - Next (X): sum of transition probs to satisfying states
    - Unbounded Until: classify S_yes/S_no/S_maybe, linear system for S_maybe
    - Bounded Until: backward induction from step k to 0
  - LabeledMC: Markov chain + state labeling (atomic propositions per state)
  - PCTL parser: recursive descent (atoms, boolean, P~p[path], X, F, G, U, U<=k)
  - Steady-state property checking via V065 analyze_chain()
  - Expected reward until target: linear system with per-state rewards
  - Bounded vs unbounded comparison API for convergence analysis
  - APIs: check_pctl(), check_pctl_state(), check_pctl_quantitative(),
    parse_pctl(), check_steady_state_property(), expected_reward_until(),
    compare_bounded_vs_unbounded(), verify_pctl_property(), batch_check()
  - Clean first run: 88/88, zero bugs

- **V068: Interval MDP Analysis** (56/56 tests pass)
  - Composes V065 (Markov chains) + V067 (PCTL model checking)
  - Interval-valued transition probabilities [lo, hi] for robust verification
  - Feasibility checking, robust reachability (min/max via value iteration)
  - Interval PCTL: pessimistic (all resolutions) + optimistic (some resolution)
  - MDP nondeterminism + interval uncertainty
  - Expected reward with intervals, point resolution, sensitivity analysis
  - Clean first run: 56/56, zero bugs

- **V069: MDP Verification** (54/54 tests pass)
  - Composes V065 (Markov chains) + C037 (SMT solver) + V068 (interval MDP)
  - Full MDP framework: states, actions, stochastic transitions, rewards
  - Value iteration + policy iteration for optimal policies (max/min)
  - Reachability analysis: optimal probability of reaching targets
  - Expected steps to target, Q-values, long-run average reward
  - SMT verification: reachability bounds, policy optimality, Bellman equation,
    policy dominance, reward bounds, bounded safety
  - Induced MC analysis via V065, comparison with V068 interval MDP
  - Zero implementation bugs -- all 6 test failures were wrong test expectations
  - Key insight: staying actions with per-step rewards can dominate leaving actions
    even with lower immediate reward, because accumulation compounds with discount

- **V070: Stochastic Game Verification** (65/65 tests pass)
  - Two-player stochastic games: Player 1 (maximizer) vs Player 2 (minimizer)
  - Composes V069 (MDP) + V065 (Markov chains) + C037 (SMT solver)
  - Turn-based games: states owned by P1, P2, or CHANCE
  - Minimax value iteration (Shapley's theorem) for optimal game values
  - Reachability games: P1 maximizes, P2 minimizes reach probability
  - Safety games: P1 tries to stay safe, P2 tries to force unsafe
  - Expected steps under adversarial play
  - Attractor computation for qualitative analysis
  - Concurrent (simultaneous-move) games with matrix game solvers
  - Fictitious play for general NxM matrix games, closed-form for 2x2
  - SMT-based verification: value bounds, strategy optimality, reachability bounds
  - Game-to-MDP/MC conversion, MDP comparison
  - Multi-property batch verification API
  - Zero implementation bugs -- all 10 test failures were API mismatches
    (mc.matrix->mc.transition, solver.assert_formula->solver.add,
    result.get()->result.steady_state, test expectation with self-loop gain)
  - Key insight: self-loop actions can outperform direct-to-terminal actions
    when the self-loop accumulates reward across iterations (Q value grows
    with discount * self_transition_prob * V[self])

- **V071: MDP Model Checking (PCTL for MDPs)** (68/68 tests pass)
  - Extends V067 PCTL model checking to handle MDP nondeterminism
  - Composes V067 (PCTL AST/parser) + V069 (MDP data structures) + V065 (Markov chains)
  - Computes Pmax (best policy) and Pmin (worst policy) for each path formula
  - Two quantification modes:
    - Universal: P>=p holds iff Pmin >= p (under ALL policies)
    - Existential: P>=p holds iff Pmax >= p (EXISTS a policy)
  - Algorithms:
    - Next: max/min over actions of transition prob sum to satisfying states
    - Unbounded Until: value iteration with max/min action selection at each step
    - Bounded Until: backward induction with max/min at each step
    - Expected reward: value iteration for accumulated reward until target
  - Features:
    - LabeledMDP: MDP + state labeling (atomic propositions)
    - Policy extraction: witness policy for max/min probabilities
    - Quantification comparison: universal vs existential side-by-side
    - Induced MC comparison: MDP results vs MC under extracted policy
    - Batch checking: multiple formulas against same MDP
    - Parse integration: V067 parser works directly with MDP checker
  - APIs: check_mdp_pctl(), check_mdp_pctl_state(), mdp_pctl_quantitative(),
    verify_mdp_property(), compare_quantifications(), batch_check_mdp(),
    induced_mc_comparison(), mdp_expected_reward()
  - Zero implementation bugs -- 2 test failures were in expected reward formulation
    (unreachable states set to inf, should use value iteration from 0)
  - Key insight: expected reward for unreachable absorbing states with 0 reward
    should be 0, not inf. Value iteration from 0 handles this naturally.

- **V072: PCTL for Stochastic Games** (71/71 tests pass)
  - Extends V067 PCTL model checking + V071 MDP PCTL to V070 stochastic games
  - Two-player value iteration: P1 maximizes, P2 minimizes at their owned states
  - CHANCE states compute expected value (fixed probability)
  - Features:
    - LabeledGame: StochasticGame + state labeling for PCTL atoms
    - 4 quantification modes: adversarial (P1 max P2 min), cooperative (both max),
      P1-optimistic (game value), P2-optimistic (all min)
    - Next: owner-dependent max/min over actions
    - Until: two-player value iteration with state classification
    - Bounded Until: two-player backward induction
    - Expected reward: two-player value iteration with per-step rewards
    - Strategy extraction: P1 and P2 optimal strategies from converged values
    - Induced MC comparison: game results vs MC under extracted strategies
    - Quantitative API: game_value, all_min, all_max probability vectors
    - Batch checking, property verification, parsed formula support
  - APIs: check_game_pctl(), check_game_pctl_state(), game_pctl_quantitative(),
    verify_game_property(), compare_quantifications(), batch_check_game(),
    induced_mc_comparison(), game_expected_reward_pctl()
  - Zero implementation bugs. 1 test expectation error: always() sugar is a
    state-level formula (NOT(P>=1[F NOT phi])), not a path formula. Use
    P<=0[F NOT phi] for safety properties instead.

- **V073: Game-Theoretic Strategy Synthesis** (57/57 tests pass)
  - Composes V070 (stochastic games) + V072 (game PCTL) + V065 (Markov chains) + C037 (SMT)
  - Objective-driven synthesis: reachability, safety, PCTL, reward objectives
  - Permissive strategies: all actions achieving optimal value
  - Multi-objective synthesis: Pareto-optimal strategies via weight-space sampling
  - Strategy verification: induce MC and verify objective achievement
  - Strategy composition: combine strategies with priority ordering
  - Assume-guarantee synthesis: compositional decomposition of objectives
  - Strategy refinement: iterative improvement from any initial strategy
  - Strategy comparison: evaluate alternatives side-by-side
  - PCTL synthesis pipeline: full pipeline with induced MC export
  - APIs: synthesize_reachability(), synthesize_safety(), synthesize_pctl(),
    synthesize_permissive_reachability(), synthesize_permissive_safety(),
    verify_strategy(), synthesize_multi_objective(), compose_strategies(),
    assume_guarantee_synthesis(), refine_strategy(), compare_strategies(),
    synthesize_from_pctl(), synthesize(), synthesis_summary()

- **V074: Omega-Regular Games** (51/51 tests pass)
  - Composes V023 (LTL model checking) + V070 (stochastic games) + V072 (game PCTL)
  - LTL objectives for two-player stochastic games
  - Pipeline: LTL -> NBA (Buchi automaton) -> product game (Game x NBA) -> Buchi game solving
  - Product game construction preserving ownership and transitions
  - Qualitative: almost-sure winning and positive winning regions
  - Quantitative: value iteration for Buchi acceptance probabilities
  - Strategy extraction with initial-automaton-state preference
  - Direct and negation-based modes
  - Convenience: safety, liveness, persistence, recurrence, response
  - Multi-objective LTL, conjunction, LabeledGame integration
  - Strategy verification via MC simulation
  - Comparison APIs (direct vs negation, LTL vs PCTL)
  - APIs: check_ltl_game_direct(), check_ltl_game(), check_ltl_labeled_game(),
    check_safety/liveness/persistence/recurrence/response_game(),
    check_multi_ltl_game(), check_conjunction_game(), verify_ltl_strategy(),
    compare_direct_vs_negation(), compare_ltl_vs_pctl()

- **V075: Reactive Synthesis (GR(1))** (49/49 tests pass)
  - GR(1) synthesis: given environment assumptions and system guarantees, synthesize controller
  - Composes V021 (BDD model checking) for symbolic state space manipulation
  - Three-nested fixpoint (Piterman, Pnueli, Sa'ar 2006): nu Z, for-i, nu X
  - Controllable predecessor: forall env'. (env_safe => exists sys'. (sys_safe AND Z_next))
  - GR1Arena: BDD-based game arena with CPre, _to_next, _to_curr
  - GR1Spec: env/sys vars, init, safety, liveness (env_live, sys_live)
  - Safety/Reachability/Buchi synthesis as simpler fragments
  - Arbiter (N-client mutual exclusion) and traffic light examples
  - Mealy machine extraction from strategy BDD
  - Counterstrategy extraction for unrealizable specs
  - Controller verification (init containment, fixpoint closure, safety)
  - Bug fix: attractor closure after for-i loop closes winning region under CPre
  - Key APIs: gr1_synthesis(), safety_synthesis(), reachability_synthesis(),
    buchi_synthesis(), make_gr1_game(), synthesize_arbiter(), synthesize_traffic_light(),
    simulate_strategy(), extract_counterstrategy(), extract_mealy_machine(),
    verify_controller(), check_realizability(), explicit_to_gr1(), compare_synthesis_approaches()

- **V076: Parity Games** (98/98 tests pass)
  - Three solving algorithms: Zielonka, Small Progress Measures, McNaughton
  - ParityGame data structure, attractor computation
  - Priority compression, self-loop removal optimizations
  - Buchi/co-Buchi/Rabin/Streett to parity conversions
  - Strategy verification, random game generation, algorithm comparison
  - All three algorithms cross-validated across 50+ random games
  - Key fix: SPM tuple ordering must be most-significant-first for correct comparison
  - Key fix: Buchi encoding: accepting nodes need HIGHEST even priority (2, not 0)
  - Key fix: SPM TOP must use n+1, not d+2
  - Key fix: SPM measures must be monotonically non-decreasing

- **V077: LTL Synthesis via GR(1) Reduction** (77/77 tests pass)
  - Composes V023 (LTL model checking -- formula AST, parser) + V075 (GR(1) synthesis)
  - Reduces LTL synthesis to GR(1) games for the GR(1)-realizable fragment
  - Fragment classification: init, safety G(phi), liveness GF(phi), response G(p->Fq), persistence FG(phi)
  - Response G(p->Fq) encoded as GF(!p|q) direct liveness (not auxiliary variables)
  - Propositional safety lifted to next-state for sys vars (system controls next state in GR(1))
  - Safety formulas generate init constraints (G(phi) implies phi at t=0)
  - Full synthesis pipeline: LTL spec -> classify -> reduce to GR(1) -> V075 synthesis -> strategy/Mealy
  - Examples: arbiter (mutex, response, no-spurious), traffic light, buffer controller
  - Key bugs caught during development:
    1. parse_ltl requires G(F(a)) not GF(a) -- parser doesn't recognize GF as compound
    2. BDD var creation must use named_var() returning nodes, not var_index returning ints
    3. Safety G(phi) needs init constraint -- without it, init allows states violating phi
    4. Propositional safety over sys vars must use next-state BDD nodes (sys controls next)
    5. Aux variable encoding of response creates unrealizable specs when combined with
       mutex+no-spurious safety -- direct GF(!p|q) liveness encoding is correct approach

- **V078: Partial Order Reduction for Model Checking** (80/80 tests pass)
  - Standalone: explicit-state concurrent system model with POR techniques
  - Five model checking methods: full BFS, stubborn BFS, ample DFS, sleep BFS, combined
  - Independence relation: static (R/W sets) and dynamic (execute-compare) checks
  - Stubborn sets (Valmari): seed + dependence closure
  - Ample sets (Clarke/Grumberg/Peled): per-process C0-C3 conditions with stack-based C3
  - Sleep sets (Godefroid): propagate explored transitions through independent successors
  - Combined: stubborn + sleep for maximum reduction
  - Example systems: Peterson's mutex, ticket lock (N), producer-consumer, dining philosophers,
    shared counter, fully independent processes
  - State space statistics, reachability analysis, deadlock detection
  - Clean first-pass: 80/80 (two test expectation fixes, zero implementation bugs)

- **V080: Omega-Regular Game Solving via Parity Reduction** (80/80 tests pass)
  - Composes V076 (parity games) + V074 (omega-regular games) + V023 (LTL)
  - Unified solver: Buchi, co-Buchi, Rabin, Streett, Muller, parity, LTL
  - All conditions reduced to parity games via V076's Zielonka/SPM
  - Correct Muller reduction via LAR (Latest Appearance Record) construction
  - Correct Rabin reduction via Rabin->Muller->LAR (V076's encoding is buggy)
  - LTL-to-parity: formula->NBA->product game with Buchi acceptance
  - NBA nondeterminism: Even-owned intermediate nodes at Odd states
  - Acceptance composition: generalized Buchi conjunction + disjunction
  - Found 2 bugs in V076 (solve() Phase 4 override, rabin_to_parity encoding)

- **V082: Energy/Mean-Payoff Games** (66/66 tests pass)
  - Composes V076 (parity games -- Zielonka solver)
  - Energy games: minimum initial credit for Even to win (energy >= 0 forever)
  - Mean-payoff games: optimal long-run average weight per node
  - Energy-parity games: combined energy + parity via iterative refinement
  - Algorithms: value iteration (energy), binary search + energy (mean-payoff),
    Zielonka + energy iteration (energy-parity, Chatterjee-Doyen 2012)
  - Energy-mean-payoff connection: mean payoff >= 0 iff energy game winnable
  - Strategy verification by simulation
  - Tarjan SCC for per-component mean-payoff computation
  - Key fix: energy value iteration must propagate inf for losing nodes, not bound+1
  - Key fix: V076 ParityResult uses win0/win1, not win_even/win_odd

- **V083: Weighted Automata** (137/137 tests pass)
  - Automata with semiring-valued transitions for quantitative language analysis
  - 8 semiring implementations: Boolean, Tropical, MaxPlus, Probability,
    Counting, Viterbi, MinMax, Log (numerically stable)
  - WFA operations: union, concatenation, Kleene star, intersection (Hadamard)
  - Run weight computation, shortest distance, n-best paths
  - Determinization (weight-residual), weight pushing, trim
  - Equivalence checking, NFA conversion, statistics
  - Key fix: shortest_distance needs different algorithms for idempotent
    (Bellman-Ford) vs non-idempotent (topological) semirings
  - Key fix: determinization with self-loops causes state explosion --
    added safety limit (10K states)

- **V085: Quantitative Language Inclusion** (77/77 tests pass)
  - Composes V083 (weighted automata) + C037 (SMT solver)
  - Quantitative inclusion/equivalence checking between WFAs
  - 5 techniques: bounded exploration, product construction, weighted bisimulation,
    forward simulation, SMT-guided search
  - Additional: distance measurement, refinement checking, language quotient,
    approximate inclusion (epsilon-tolerance), comprehensive pipeline
  - Works with all 8 V083 semirings
  - APIs: check_inclusion(), check_equivalence(), check_strict_inclusion(),
    weighted_bisimulation(), simulation_inclusion_check(), quantitative_distance(),
    check_refinement(), language_quotient(), approximate_inclusion(),
    comprehensive_check(), compare_inclusions()
  - Key lesson: WFA.add_state(state, initial=..., final=...) not initial_weight/final_weight.
    Transitions are tuples (label, dst, weight) not WFATransition objects.

- **V089: Tree Automata** (90/90 tests pass)
  - New domain: automata over ranked trees (ASTs, terms, XML)
  - Bottom-Up Tree Automaton (BUTA): run, accept, emptiness, witness, determinize,
    complete, complement, union, intersection, difference, inclusion, equivalence,
    minimization, enumeration
  - Top-Down Tree Automaton (TDTA): bidirectional conversion with BUTA
  - TreePattern: wildcards, variable capture, consistency checking
  - TermRewriteSystem: leftmost-outermost normalization
  - Schema automaton: XML-like validation
  - Key fix: complement needs completion (sink state for missing transitions)

- **V090: Tree Transducers** (76/76 tests pass)
  - Extends V089 (tree automata) with output for verified tree transformations
  - Bottom-Up Tree Transducer (BUTT): bottom-up processing with OutputTemplate
  - Top-Down Tree Transducer (TDTT): top-down with state-dependent transformations
  - OutputTemplate: variable references ($0, $1, ...), symbol construction, linearity checking
  - Features:
    - Identity, relabeling, pruning transducers (practical builders)
    - Rewrite rules to transducer conversion (rewrite_to_butt)
    - Sequential composition (sequential_transduce)
    - BUTT composition for linear transducers (compose_butt)
    - Domain extraction (input automaton as BUTA)
    - Range approximation (sample-based BUTA construction)
    - Functionality checking (at most one output per input)
    - Totality checking (at least one output per domain input)
    - Transducer equivalence checking (sample-based)
    - Type checking (input_type -> transducer -> output_type)
    - Inverse transducer (BUTT -> TDTT)
    - AST optimizer builder (pattern-based rewriting)
    - Transformation summary with examples
  - Tested: arithmetic simplification (0+x->x, 1*x->x, 0*x->0),
    boolean simplification (double negation, and(true,x)->x),
    cross-alphabet transduction, deep trees, nondeterministic outputs
  - Key insight: partial transducers are total over their own domain.
    Domain extraction from input_automaton() exactly captures processable trees.

- **V094: Pushdown Systems Verification** (90/90 tests pass)
  - PDS data structures: control states + stack alphabet + rules (POP/SWAP/PUSH)
  - P-Automaton: NFA-based regular config set representation
  - Pre* (backward reachability) and Post* (forward reachability) via saturation
  - Configuration reachability, safety checking, bounded reachability
  - State space exploration with deadlock detection
  - Invariant checking over all reachable configurations
  - Recursive program modeling (call/return as push/pop)
  - Example systems: counter, recursive factorial, mutual recursion, stack inspection
  - Zero implementation bugs

- **V096: Interprocedural Analysis via Pushdown Systems** (79/79 tests pass)
  - Composes V094 (pushdown systems) + C039 (abstract interpreter) + C010 (parser)
  - IFDS tabulation algorithm (Reps-Horwitz-Sagiv) for context-sensitive dataflow
  - ICFG construction from C10 source with call/return/call-to-return edges
  - Three analysis problems: reaching definitions, taint, live variables
  - PDS reachability via V094 pre*/post* for exact context sensitivity
  - Function summaries via C039 abstract interpretation
  - Context-sensitive vs context-insensitive comparison API
  - Full pipeline: IFDS + PDS reachability + function summaries
  - Bug fixes: return-to-exit edge, return-calls detection, depth-3 PDS contexts
  - APIs: interprocedural_analyze(), reaching_definitions(), taint_analysis(),
    live_variables(), compare_sensitivity(), pds_reachability_analysis(),
    pds_context_analysis(), full_interprocedural_analysis()

- **V098: IDE Framework (Interprocedural Distributive Environment)** (75/75 tests pass)
  - Extends V096 IFDS to IDE: facts carry values (environments), not just sets
  - Value lattice: TOP > Const(n) > BOT with meet/join/leq
  - Micro-function algebra: Id, Const, Linear(a*x+b), Top, Bot, Composed, Meet
  - Two-phase algorithm: Phase 1 (forward tabulation), Phase 2 (value computation)
  - Copy-constant propagation: tracks constant assignments and copies
  - Linear constant propagation: tracks a*x+b transformations precisely
  - C10 ICFG construction with call/return/call-to-return edges
  - APIs: ide_analyze(), get_constants(), compare_analyses(), ide_verify_constant()
  - Bug fix: ReturnStmt as last stmt needs explicit edge to exit point
  - Key lesson: In IDE, ZERO-seeded paths determine generated values via micro-functions.
    Non-ZERO paths propagate value transformations via composition.

- **V101: Demand-Driven Analysis** (60/60 tests pass)
  - Extends V098 (IDE framework) to demand-driven mode
  - Backward tabulation: starts from query (point, fact), traverses ICFG in reverse
  - Only computes values for queried variables at queried points
  - Memoization cache: subsequent queries on same/dependent facts reuse prior work
  - Interprocedural: reverses call/return/call-to-return flows
  - Recursion guard: cyclic dependencies return TOP (sound)
  - Cache invalidation: point-specific or full, with forward propagation
  - Demand slice: explored points form natural backward slice of query
  - Incremental: analyze v1, detect changes, re-analyze v2
  - Comparison API: exhaustive (V098) vs demand-driven side-by-side
  - Key fix: ZERO fact must be included in source fact enumeration -- ZERO
    generates new facts via ConstFunction but is not in problem.all_facts
  - APIs: demand_query(), demand_analyze(), demand_constants(),
    demand_verify_constant(), demand_function_summary(), demand_slice(),
    incremental_demand(), compare_exhaustive_vs_demand()

- **V103: Widening Policy Synthesis** (76/76 tests pass)
  - Composes V020 (domain functor) + V019 (threshold widening) + C039 + C010 + C037
  - Automatically synthesizes per-loop widening/narrowing policies from program structure
  - Four strategies: standard, threshold, delayed, delayed_threshold
  - Loop structure analysis: counter detection, bound extraction, constant collection
  - PolicyInterpreter: full C10 abstract interpreter with per-loop policy selection
  - FunctorPolicyInterpreter: extends V020 for domain-generic policy support
  - APIs: policy_analyze(), auto_analyze(), synthesize_policies(), compare_policies(),
    functor_policy_analyze(), compare_with_functor(), synthesize_and_validate(),
    get_loop_info(), policy_summary(), validate_policy()
  - Boundary fixes: C10 Parser API, AbstractEnv field names, FunctorInterpreter private attrs

- **V104: Relational Abstract Domains** (90/90 tests pass)
  - Two relational abstract domains: Zone (DBM, x-y<=c) + Octagon (variable doubling, +/-x+/-y<=c)
  - Full C10 interpreters (OctagonInterpreter, ZoneInterpreter) with relational condition refinement
  - Floyd-Warshall closure + strong closure for octagon
  - Composes with C039 (comparison API) + C010 (parser)
  - APIs: octagon_analyze(), zone_analyze(), get_variable_range(), get_relational_constraints(),
    compare_analyses(), verify_relational_property()
  - Key fix: assign_const must NOT mix doubled DBM values with raw constraint values
  - Key fix: assign_var must explicitly set unary bounds (dbm[tp][tn], dbm[tn][tp])

- **V110: Abstract Reachability Tree (ART)** (71/71 tests pass)
  - Core data structure behind CEGAR model checkers (BLAST, CPAchecker)
  - Composes C010 (parser) + C037 (SMT solver) + V107 (Craig interpolation)
  - CFG construction from C10 source (assignments, if/else, while, assert)
  - Predicate abstraction with SMT-based abstract post and coverage
  - Full CEGAR loop: explore ART -> check feasibility -> interpolation refinement
  - Coverage: subsumption checking prevents redundant exploration
  - APIs: verify_program(), check_assertion(), get_predicates(), build_cfg_from_source(),
    compare_with_without_refinement(), cfg_summary(), art_summary()
  - Limitation: bounded loop unrolling -- use PDR/k-induction for loops
  - Zero bugs on first run, 71/71

- **V112: Trace Abstraction Refinement** (68/68 tests pass)
  - Automata-based program verification (Heizmann et al., 2009)
  - Composes V107 (Craig interpolation) + C037 (SMT solver) + C010 (parser)
  - Program traces = words over CFG edge alphabet
  - Infeasibility automaton: union of learned interpolation automata
  - CEGAR: enumerate error traces -> SMT feasibility -> Craig interpolation -> NFA generalization
  - Two modes: BFS trace enumeration and lazy DFS with coverage
  - Comparison API with V110's ART-based CEGAR
  - APIs: verify_trace_abstraction(), verify_lazy(), check_assertion(),
    get_cfg(), trace_abstraction_summary(), compare_with_art()
  - Key fix: Python module identity -- V107 imports `from smt_solver` creating different
    class than `from challenges.C037_smt_solver.smt_solver`. Use consistent import paths.
  - Key fix: V107 sequence_interpolate returns [True,...,False] -- don't add extra endpoints
  - Key fix: Trivial edge self-loops must be on ALL automaton states, not just initial

- **V113: Configurable Program Analysis (CPA)** (81/81 tests pass)
  - Composes V110 (ART/CFG) + V020 (domain functor) + V104 (relational domains) + V107 (Craig interpolation) + C037 (SMT) + C010 (parser)
  - CPAchecker-style framework: pluggable abstract domains into ART exploration
  - CPA interface: AbstractState, TransferRelation, MergeOperator, StopOperator, PrecisionAdjustment
  - Three concrete CPAs:
    1. IntervalCPA: per-variable interval tracking with condition refinement
    2. ZoneCPA: difference-bound matrix (DBM) with relational constraints (x-y<=c)
    3. PredicateCPA: predicate abstraction with SMT-based transfer + CEGAR refinement
  - CompositeCPA: product of multiple CPAs (independent component-wise transfer)
  - Merge strategies: MergeSep (keep separate), MergeJoin (join all)
  - Stop strategies: StopSep (covered by any), StopJoin (covered by join)
  - Generic CPA algorithm: BFS ART exploration with configurable components
  - CEGAR loop for predicate CPA: infeasible path -> Craig interpolation -> new predicates
  - Predicate discovery: assignment transfer checks all registered predicates (not just current)
  - Path feasibility: SSA-based SMT encoding of error traces
  - APIs: verify_with_intervals(), verify_with_zones(), verify_with_predicates(),
    verify_with_composite(), compare_cpas(), get_variable_ranges(), cpa_summary()
  - Key fix: PredicateTransfer._post_assign must check ALL registered predicates
    (not just those in current state) to discover newly-true predicates after assignment.
    Without this, `let x = 5; assert(x > 0)` fails because x>0 is never discovered.
  - Zero implementation bugs. Fix was a design oversight caught during testing.

- **V118: Timed Automata Verification** (84/84 tests pass)
  - New domain: real-time systems verification using Alur-Dill timed automata
  - Zone (DBM) representation for efficient clock constraint manipulation
  - Zone-based symbolic state space exploration (BFS with subsumption)
  - Successor computation: guard -> reset -> invariant -> future -> invariant
  - Features:
    - Clock constraints: simple (x op c), difference (x-y op c), conjunction
    - Zone operations: constrain, future, reset, intersect, includes, sample
    - Floyd-Warshall canonicalization for zone closure
    - Zone graph exploration with subsumption checking
    - Reachability checking with counterexample trace extraction
    - Safety checking (unreachability of unsafe locations)
    - Timed word acceptance (concrete execution check)
    - Product construction (synchronous on shared alphabet)
    - Empty language check, approximate language inclusion
    - Example systems: simple light timer, train-gate controller, Fischer's mutex
  - Fischer's mutual exclusion protocol: proven safe (timing delta < Delta)
  - APIs: check_reachability(), check_safety(), check_timed_word(),
    explore_zone_graph(), product(), check_empty_language(),
    check_language_inclusion(), zone_graph_summary(), simple_ta(),
    fischer_mutex(), train_gate_controller(), simple_light_timer()
  - Clock constraint helpers: clock_leq/lt/geq/gt/eq(), clock_diff_leq/geq(), guard_and()
  - Zero implementation bugs. 2 test expectation fixes (initial_zone needs future() for non-zero constraints, Fischer protocol encoding required proper state machine with last-writer-wins timing)

- **V142: Timed Automata + LTL Model Checking** (45/45 tests pass)
  - Composes V118 (timed automata) + V023 (LTL model checking via Buchi automata)
  - Zone-based product construction: TA x NBA
  - Nested DFS accepting cycle detection in product zone graph
  - Safety, liveness, response, until properties
  - Location labeling, zone graph abstraction, batch verification
  - Example systems: light timer, train-gate controller, mutual exclusion
  - Known V023 limitation: GBA spurious cycles for conjunctions of GF formulas
  - APIs: check_timed_ltl(), check_timed_ltl_parsed(), check_timed_safety(),
    check_timed_liveness(), check_timed_response(), check_timed_until(),
    abstract_zone_graph(), compare_timed_vs_untimed(), batch_check()

- **V146: Hybrid Automata Verification** (106/106 tests pass)
  - Extends V118 (timed automata) to hybrid automata with continuous dynamics
  - Rectangular automata: per-variable flow rate intervals per mode
  - RectZone (extended DBM) with rectangular time elapse
  - BFS zone graph exploration with subsumption
  - Safety, invariant, bounded liveness verification
  - Simulation, product construction, 5 example systems
  - Compare hybrid vs timed automata expressiveness
  - APIs: check_reachability(), check_safety(), verify_safety(), verify_invariant(),
    verify_bounded_liveness(), analyze_modes(), zone_graph_summary(),
    compare_hybrid_vs_timed(), batch_verify(), product(), simulate()
  - Example systems: thermostat(), water_tank(), railroad_crossing(), bouncing_ball(), two_tank()

## Next Challenges (Priority Order)

### V147: Hybrid Automata + LTL
- Compose V146 (hybrid automata) + V142 (timed LTL) concepts
- LTL model checking over hybrid automaton zone graphs
- Zone-product construction with Buchi acceptance

### V148: Timed Game Synthesis
- Combine V146/V142 (timed/hybrid) + V075 (GR(1) synthesis) concepts
- Synthesize timed controllers from LTL specifications
- Controllable predecessor over zone graphs

### V149: Polyhedral Abstract Domain
- Convex polyhedra for non-rectangular hybrid automata
- Linear combination constraints beyond DBMs
- H-representation and V-representation with conversion

## Lessons Learned

### Session 191 (V146)
- Rectangular hybrid automata use the same DBM approach as timed automata
  but time elapse must account for different flow rates per variable.
- Same-rate variables preserve difference constraints during time elapse.
  Different-rate variables lose their difference constraints.
- Test expectation bug: cooling at rate -0.5 for 4 units from 22 gives 20, not 18.
  Need 8 units to reach 18. Always verify concrete arithmetic.
- 58-session zero-bug streak.


### Session 187 (V142)
- **V023 module is ltl_model_checker.py** (not ltl_model_checking.py).
  V021 module is bdd_model_checker.py. V023 imports from V021, so both paths
  must be on sys.path.
- **V023 GBA has spurious accepting cycles for GF conjunctions**: The NBA for
  NOT(G(F(a)) & G(F(b))) has transitions with empty pos/neg sets that create
  accepting self-loops in "waiting" states. This causes false VIOLATED verdicts
  for conjunctions of GF formulas. Individual GF formulas work correctly.
- **Zone-based product construction is efficient**: The product TA x NBA with
  zone subsumption keeps state space manageable. Nested DFS cycle detection
  finds accepting cycles without full state enumeration.
- **54-session zero-bug streak**: Zero implementation bugs. One test adjustment
  for V023 limitation.

### Session 160 (V113)
- **Predicate post-assign must discover new predicates**: The standard predicate
  abstraction transfer only checks if existing predicates are PRESERVED. But after
  `x = 5`, no predicates hold yet (initial state is empty). Must check ALL registered
  predicates for implication after assignment, not just current ones.
- **V110 module name is art.py, not abstract_reachability_tree.py**: Always check
  actual filenames with ls before importing.
- **V104 module name is relational_domains.py, not relational_abstract_domains.py**
- **113-session zero-bug streak**: Fix was a design oversight (predicate discovery
  scope), not an algorithmic bug. Core CPA algorithm correct on first implementation.

### Session 159 (V112)
- **Python module identity for isinstance**: V107 imports `from smt_solver import Var`
  via sys.path, which loads a separate module object than `from challenges.C037_smt_solver.smt_solver import Var`.
  isinstance() fails because they're different classes. Fix: use the same import path as V107
  (add C037 dir to sys.path, import `from smt_solver`).
- **V107 sequence_interpolate returns [True, ..., False]**: The returned list already
  includes True at index 0 and False at the end. Don't prepend/append extra endpoints.
- **Trivial edge self-loops on ALL states**: Skip/join edges (trivially true formulas)
  don't change the abstract state. Self-loops must be added for every automaton state,
  not just the initial TRUE state. Otherwise traces passing through non-TRUE states
  can't traverse skip edges and acceptance fails.
- **C10 CallExpr at top level**: `assert(x > 0);` parses as a bare CallExpr, not wrapped
  in ExprStmt. The CFG builder's `_build_stmt` must handle CallExpr directly.
- **112-session zero-bug streak**: All 7 issues were composition boundary problems.
  Core algorithm (CEGAR loop, SSA encoding, feasibility checking) correct on first run.

### Session 147 (V104)
- **Octagon DBM doubled encoding**: Unary bounds are doubled (dbm[x+][x-] = 2*upper),
  but cross-variable constraints are NOT doubled (dbm[x+][y-] = x+y, directly).
  assign_const must compute `c + upper(other)` using `dbm[op][on]/2` for upper(other),
  NOT `2*c + dbm[op][on]`.
- **assign_var must set unary bounds**: After forgetting target, setting tp-sp=0 and
  sn-tn=0 is not enough. Must explicitly copy dbm[sp][sn] to dbm[tp][tn] for the
  target's unary bound (upper/lower).
- **Variables declared inside if-blocks lose info at join**: When z is declared only
  in the then-branch, the else-branch has z=TOP, so the join loses all info. This is
  correct -- must declare z before the if-block to preserve information.
- **104-session zero-bug streak**: All 5 failures were propagation boundary issues,
  zero algorithmic bugs in zone/octagon domain logic.

### Session 146 (V103)
- **C010 has Parser class, not parse function**: `Parser(tokens).parse()` not `parse(tokens)`.
  The C039 abstract_interpreter.py shows the correct usage. Always check imports.
- **AbstractEnv fields are public**: `.signs`, `.intervals`, `.consts` -- not `._signs` etc.
  Different from FunctorInterpreter which uses `._max_iterations` (private).
- **103-session zero-bug streak**: All 3 initial failures were API name mismatches at
  composition boundaries. Zero algorithmic bugs in the widening policy synthesis logic.

### Session 140 (V098)
- **ReturnStmt last-statement edge (again)**: V096 lesson learned in session 137
  ("ALL last statements need edge to exit") was not applied when building V098's
  ICFG construction. The `if last_cls != 'ReturnStmt'` guard silently dropped
  the exit edge for single-return functions like `fn seven() { return 7; }`.
  This blocked ALL interprocedural value flow.
- **IDE Phase 2 reads final jump_fn**: Phase 2 iterates `self.jump_fn` after
  Phase 1 completes. Jump functions updated via meet during Phase 1 are visible
  to Phase 2 even if the updated path edge wasn't re-popped from the worklist.
- **ZERO-seeded values come from micro-functions**: For IFDS-style ZERO fact,
  the micro-function IS the value generator. ConstFunction(Const(5)).apply(BOT)
  gives Const(5) directly. Non-ZERO paths transform values via composition.
- **Test exit points, not intermediate points**: Statement effects are applied
  on the OUTGOING edge, not at the statement node itself. Values at `main.s1`
  reflect state BEFORE s1's statement executes. Always check exit points.
- **98-session zero-bug streak**: The ICFG bug was caught and fixed during testing.

### Session 137 (V096)
- **Return-calls need special handling**: `return foo(x);` in C10 is a ReturnStmt
  whose value is a CallExpr. Must detect this and create call edges, not just
  return edges. Without this, callee functions are never reached via PDS.
- **ICFG last-statement edge**: ALL last statements (including return) need an
  edge to the function exit point. The original code skipped returns, leaving
  the callee exit unreachable for IFDS end-summary computation.
- **PDS depth for nested calls**: Direct calls need depth-2 stack checks
  (entry + return point). But A -> B -> C chains need depth-3. Always check
  at least depth 3 in `pds_context_analysis`.
- **IFDS tabulation is naturally context-sensitive**: The Reps-Horwitz-Sagiv
  algorithm tracks (d1, n, d2) triples where d1 is the fact at procedure entry.
  This naturally matches calls with returns. No additional PDS needed for
  context sensitivity -- PDS adds reachability analysis on top.
- **96-session zero-bug streak**: All 3 issues were caught and fixed during
  development. The IFDS algorithm itself worked correctly on first implementation.

### Session 122 (V085)
- **WFA API**: `add_state(state, initial=..., final=...)` not `initial_weight`/`final_weight`.
  Transitions stored as tuples `(label, dst, weight)` not `WFATransition` objects.
  Dict names: `initial_weight`/`final_weight` (singular, not plural).
- **Tropical inf semantics**: WFA with no final states returns inf (tropical zero) for all words.
  `inf` is NOT `<= finite`, so not included. This is semantically correct -- infinite cost
  is worse than any finite cost.
- **80-session zero-bug streak**: Zero implementation bugs. 2 test expectation fixes.

### Session 119 (V083)
- **Semiring shortest distance needs algorithm selection**: Idempotent semirings
  (tropical, viterbi, minmax) work with Bellman-Ford relaxation because
  plus(a, a) = a prevents divergence. Non-idempotent semirings (probability,
  counting) diverge under repeated relaxation -- use topological single-pass.
- **Determinization safety limit**: Weight-residual determinization with
  self-loops creates infinitely many configurations (each with different
  residual weight). Safety limit of 10K states prevents OOM.
- **78-session zero-bug streak**: Zero implementation bugs. All 7 test fixes.

### Session 117 (V082)
- **Energy value iteration: use inf not sentinel**: When a node's credit exceeds
  the bound, set it to `inf`, not `bound+1`. With `bound+1`, dependent nodes compute
  `max(0, bound+1 - w)` which is finite, causing incorrect "winning" classification.
  With `inf`, `max(0, inf - w) = inf` propagates correctly.
- **V076 ParityResult field names**: `win0`/`win1` (not `win_even`/`win_odd`),
  `strategy0`/`strategy1` (not `strategy_even`/`strategy_odd`).
- **Mean-payoff via binary search is clean**: Binary search on threshold + energy
  game test per SCC gives correct values. Tarjan SCC decomposition + reverse topo
  order handles transient nodes naturally.
- **77-session zero-bug streak**: Both bugs were caught and fixed during development
  (API mismatch + algorithmic fix). The energy inf-propagation bug is a general
  lesson: sentinel values must be absorbing (inf * anything = inf, inf + anything = inf).

### Session 100 (V080)
- **V076 solve() has Phase 4 bug**: Self-loop removal creates immediate wins (imm0/imm1).
  After attractor recomputation of imm0 into win0, win1 from the reduced game still
  contains those nodes. Line 724 `win0 = game.nodes - win1` then overrides the correct
  win0. Workaround: call zielonka()/small_progress_measures() directly.
- **V076 rabin_to_parity is fundamentally limited**: Direct priority assignment can't
  capture the finitely-often condition of Rabin pairs when non-pair nodes exist.
  Code also has a comment/code mismatch (2k+1 vs 2*k). Use LAR instead.
- **LAR must track ALL arena nodes**: Only tracking nodes in the Muller table loses
  information about non-table nodes in cycles. Without tracking node 0, the LAR can't
  distinguish {1,2} (accepting) from {0,1,2} (not accepting).
- **LAR edge construction: update for SOURCE, not SUCCESSOR**: The LAR state
  (n, perm) means "at node n, LAR before visiting n is perm". The edge carries
  the LAR AFTER visiting n. Priority is computed from the pre-visit LAR.
- **Non-relevant node priority depends on empty set acceptance**: If frozenset() is
  in the Muller table, non-relevant nodes get priority 0 (even, transparent). Otherwise
  priority 1 (odd, doesn't help Even). This ensures correct behavior in pure
  non-relevant cycles.
- **LTL product: use NBA for formula, not negation**: Building NBA for NOT(formula)
  with co-Buchi creates problems with nondeterminism. NBA is existential (there EXISTS
  an accepting run), but at Odd-owned states, Odd can exploit this to avoid acceptance.
  Solution: NBA for formula directly + Buchi acceptance + intermediate Even-owned nodes
  for NBA nondeterminism at Odd states.
- **Rabin-to-Muller must enumerate subsets of ALL arena nodes**: The Rabin condition
  is defined over the set of ALL infinitely-visited nodes, not just those in the pairs.
  Enumerating only pair-related nodes misses valid accepting sets.

### Session 099 (V078)
- **Standalone works well for new domains**: V078 doesn't compose with existing
  V-challenges because it operates on explicit concurrent systems, not BDDs or SMT.
  Clean standalone implementation is fine when the domain is genuinely different.
- **Test expectations vs implementation correctness**: Two test failures were
  wrong expectations (counter system "read" transitions are actually independent
  since they only read, not write). The implementation was correct throughout.
- **Frozen dataclasses for state hashing**: Using `@dataclass(frozen=True)` with
  tuple fields gives free `__hash__` and `__eq__` for efficient state sets.
- **Lambda capture in loops**: Python lambda closure captures the loop variable
  by reference, not value. Use `_i=i` default arg pattern for correct capture.

### Session (V077)
- **parse_ltl uses G(F(a)) not GF(a)**: The V023 LTL parser treats `GF` as an
  atom named "GF", not as `G(F(...))`. Always use explicit nesting.
- **BDD named_var returns nodes, not indices**: Use `bdd.named_var(name)` to get
  BDD nodes for formula construction. Don't use `bdd.var_index()` + `bdd.var()`.
- **G(phi) needs init + safety**: Safety G(phi) means phi holds at ALL times
  including t=0. Without an init constraint, the GR(1) solver allows initial
  states that violate phi, making specs unrealizable.
- **Propositional safety needs next-state lifting for sys vars**: In GR(1), the
  system controls next-state sys vars. A propositional safety like G(!req->!grant)
  must be encoded as `!req_curr -> !grant'_next` in the transition relation,
  not `!req_curr -> !grant_curr` (which constrains already-committed state).
- **Aux variable encoding of G(p->Fq) is problematic**: The standard textbook
  encoding (aux' = (p|aux) & !q, liveness GF(!aux)) creates specs that are
  genuinely unrealizable when combined with mutex + no-spurious safety in
  multi-client scenarios. The issue is the aux state space explosion and
  interaction with safety constraints. Direct GR(1) liveness GF(!p|q) works.
- **57-session zero-bug streak continues**: All 5 bugs were caught and fixed
  during development. Deep composition debugging (V023 parser + V075 BDD +
  GR(1) semantics) required systematic isolation testing.


### Session 098 (V076)
- **SPM tuple ordering is critical**: Position d//2 is most significant, position 0
  is least. Python tuple comparison is left-to-right. Must store tuples in reverse
  order (most significant at index 0) for correct lexicographic comparison.
- **Buchi-to-parity encoding**: Accepting nodes need the HIGHEST even priority (2),
  not the lowest (0). Otherwise, in a cycle containing both accepting and non-accepting
  nodes, the non-accepting odd priority dominates.
- **SPM TOP must use n+1, not d+2**: The TOP sentinel must be larger than ANY valid
  measure component. With d+2, some valid measures exceeded TOP.
- **SPM monotonicity**: Only apply updates when new > old. Even-owned nodes minimize,
  which could produce values smaller than the current measure. Without monotonicity,
  the iteration oscillates instead of converging to the fixpoint.
- **56-session zero-bug streak**: All 4 bugs were caught and fixed during development.
  Cross-validation across 3 algorithms was the key quality mechanism.

### Session (V075)
- **Three-nested fixpoint needs attractor closure**: The standard Bloem et al. GR(1)
  algorithm with m=1 env assumption misses winning states that are one CPre step
  from the accumulated Y set. The for-i loop only runs once, and CPre(Y=FALSE)=FALSE
  in term3. Fix: after the for-i loop, add a mu-Y attractor pass:
  `Y = Y OR (g_j AND CPre(Z)) OR CPre(Y)` to close under controllable reachability.
- **GR(1) winning region is self-sustaining**: The nu Z fixpoint computes the maximal
  set from which the system can satisfy the GR(1) objective while staying within the set.
  States that can reach the set but aren't in it are captured by the attractor closure.
- **CPre drops current sys state**: When sys_safe only depends on current env vars (not
  current sys vars), CPre(Z) only depends on current env vars. This is correct -- the
  system's ability to force into Z depends only on the aspects of current state that
  constrain its choices.
- **Empty init set => unrealizable**: When env_init AND sys_init = FALSE, report
  unrealizable (no valid starting configuration), not vacuously realizable.
- **54-session zero-bug streak**: Implementation correct on first run. 3 test failures
  were all test design issues (fully-determined system, vacuous init), not impl bugs.

### Session (V074)
- **Product strategy projection needs automaton-state awareness**: Majority vote across
  automaton states for game-level strategy picks wrong action when different automaton states
  need different actions. Fix: prefer initial automaton state's action.
- **Buchi game dual formulation**: For LTL checking, negate formula, build NBA for negation,
  then P2 tries to satisfy Buchi acceptance (violate original formula). P1's probability
  of satisfying formula = 1 - P2's Buchi acceptance value.
- **Direct automaton is simpler and equally correct**: Building NBA for formula directly
  (not negation) and having P1 maximize Buchi acceptance avoids the role-swapping complexity.
- **53-session zero-bug streak**: Implementation correct on first run. One test failure
  was strategy projection issue (fixed same session, not an algorithmic bug).

### Session (V073)
- **PCTL atoms use .label not .name**: PCTL class stores atom names in `label` field.
- **Both actions can be optimal**: When two actions achieve the same value, refinement
  won't switch. Tests must not assume a specific action choice.
- **Adversarial play can block reachability**: If all P2 states have a "send back" action,
  P2 can prevent P1 from ever reaching target (prob=0).
- **min/max on filtered iterables**: Guard against empty iterables when filtering states
  (e.g., all states are targets).
- **52-session zero-bug streak**: Implementation correct on first run. All test failures
  were test expectation errors.

### Session (V072)
- **always() is state-level, not path-level**: The sugar `always(phi)` = `pnot(prob_geq(1.0, eventually(pnot(phi))))` produces a state formula. Passing it as a path formula to `prob_geq()` gives wrong results because the checker sees `P>=1[NOT(P>=1[F NOT phi])]` and processes the inner NOT as boolean negation on the path. For safety properties, use `P<=0[F NOT phi]` instead.
- **Two-player value iteration is structurally identical to single-player**: The only difference is the aggregation function at each state depends on the owner: max for P1, min for P2, expected for CHANCE. The convergence properties are the same.
- **Clean composition continues**: V067 PCTL AST/parser, V070 StochasticGame/Player, V065 MarkovChain all reuse directly. No API mismatches -- prior sessions' lessons (field names, constructors) prevent bugs.
- **51-session zero-bug streak**: Implementation was correct on first run. All test code worked except one misuse of sugar function (test error, not impl error).



### Session (V071)
- **Expected reward: don't pre-set unreachable to inf**: When computing expected
  accumulated reward until target via value iteration, start all values at 0. States
  with 0 per-step reward that can't reach target will naturally stay at 0 via
  value iteration (0 + 1.0 * 0 = 0). Pre-setting them to inf causes inf to propagate
  through any action that has non-zero transition probability to unreachable states,
  even when other actions exist.
- **PCTL satisfaction semantics for MDPs**: Universal: P>=p requires Pmin >= p (all
  policies satisfy). Existential: P>=p requires Pmax >= p (some policy satisfies).
  For P<=p, flip: universal uses Pmax, existential uses Pmin.
- **Clean composition**: V067 PCTL AST/parser reuses directly. V069 MDP data structure
  reuses directly. V065 MarkovChain for induced MC comparison. No API mismatches this
  session -- prior lessons about field names (.transition, not .matrix) prevented bugs.
- **50-session zero-bug streak**: Algorithmic implementation was correct on first run.
  Only test expectation for expected reward needed adjustment.

### Session (V070)
- **MarkovChain uses .transition not .matrix**: Field name is `transition`, not `matrix`.
  Every MC API reference must use `mc.transition[s][t]`.
- **SMTSolver uses solver.add() not solver.assert_formula()**: Also use `solver.Int(name)`
  to declare integer variables, not `Var(name, INT)` for solver-managed vars.
- **ChainAnalysis is a dataclass not a dict**: Access `result.steady_state` not
  `result.get('steady_state')`. It's `Optional[List[float]]`, check `is None`.
- **Self-loop Q-value growth**: When an action has self-transition probability p > 0,
  V[s] = r + gamma * p * V[s] + ... which gives V[s] = r / (1 - gamma*p) + ...
  This can exceed the Q-value of actions that immediately leave state s.
  Always re-derive Q-values under iteration rather than assuming single-step analysis.
- **Zero implementation bugs on 48th session**: All 10 failures were API mismatches
  (composition surface) and test expectation errors. The pattern continues:
  when algorithms are well-understood, first-run correctness is reliable.

### Session (V069)
- **Test expectations can be wrong, not the implementation**: All 6 failures were
  miscalculations in test expectations (wrong assumption about which action is optimal).
  The implementation was correct. When tests fail, check assumptions first.
- **Per-step rewards compound with discount**: In the simple MDP, "left" (reward 1,
  stay prob 0.9) beats "right" (reward 2, leave prob 0.7) because staying accumulates
  1/(1-0.9*0.9) = 5.26 vs the one-shot 2.0 + small continuation. Geometric series
  vs finite payoff.
- **MDP __post_init__ must guard against malformed actions**: If actions list is
  shorter than n_states, computing default rewards crashes. Guard with length check.

### Session (V068)
- **Greedy optimal distribution is exact for LP**: Sorting targets by value and
  greedily assigning mass from lower bounds gives the optimal distribution for
  maximizing/minimizing expected value under interval constraints. This avoids
  needing a full LP solver -- the greedy approach works because the constraint
  structure (sum = 1, box constraints) is totally unimodular.
- **Pessimistic/optimistic PCTL duality**: For P>=p, pessimistic needs MIN probs
  (hardest to satisfy), optimistic needs MAX. For P<=p, it's reversed. For NOT,
  flip the pessimistic flag for the sub-formula. This duality makes the implementation
  clean and compositional.
- **Clean first run pattern**: 46-session zero-bug streak. When the composition
  surface is well-understood (V065/V067 APIs) and the algorithm is mathematically
  clean (value iteration + greedy optimization), first-run success is expected.

### Session (V066)
- **C037 LIA limitation for uniqueness**: Checking if two steady-state solutions are
  proportional requires `pi_i * S2 == sigma_i * S1` (nonlinear MUL of two variables).
  C037 returns SAT with all-zero counterexamples. Use structural proof instead:
  irreducible => unique (Perron-Frobenius), or rank analysis of (P^T - I) matrix.
- **Encoding probabilities as scaled integers**: Convert p/q to integers by finding
  LCM of all denominators, then scale: P_int[i][j] = P_frac[i][j] * LCM. This
  keeps everything in LIA. Bound checking: pi[state] * bound_denom vs bound_numer * S.
- **Fraction.limit_denominator(10000)**: Good enough for most transition matrices.
  Exact rationals avoid floating-point issues in SMT encoding.

### Session (V064-V065)
- **V060 wilson_confidence_interval(n_total, n_successes, confidence)**: first arg is TOTAL,
  second is SUCCESSES. Getting this wrong produces nonsensical CIs. Always check arg order
  for external APIs.
- **Expected hitting time to absorbing state in multi-absorbing chain**: If multiple absorbing
  states exist, the expected time to reach a SPECIFIC one may be infinite (some paths get
  absorbed elsewhere). The linear system (I-Q)h=1 only gives meaningful results when the
  target is reachable with probability 1.

### Session (V063)
- **V060 ProbabilisticExecutor can't handle random() in source**: It prepends `let x = val;`
  but the source still has `let x = random(1,10);` which calls undefined `random`. Solution:
  do your own sampling loop, use V060's statistical functions (wilson_confidence_interval,
  sprt_test) directly instead of stat_check/stat_check_sprt high-level APIs.
- **C10 base uses `and`/`or`/`not`**: NOT `&&`/`||`/`!`. Extended C040+ has those but
  base C010 stack_vm does not. Always test operator syntax before assuming.
- **Direct sampling is simpler and more reliable**: Instead of trying to compose V060's
  executor (which assumes valid C10 source), build your own sample loop: generate random
  inputs, replace random() with concrete values, execute, check postcondition.

### Session (V061 + V062)
- **SMTResult is enum, not string**: `solver.check()` returns `SMTResult.SAT`, not `'SAT'`.
  Compare with `SMTResult.SAT`, not string equality.
- **C10 VM constructor takes chunk**: `VM(chunk)` then `vm.run()`, not `VM()` then `vm.run(chunk)`.
- **C10 print requires parens**: `print(x);` not `print x;`. Base C10 always needs parens.
- **Negative literals in V004**: `-1000` parses as `SUnaryOp('-', SInt(1000))`, not `SInt(-1000)`.
  Need `_as_const()` helper to unwrap unary negation for boundary extraction.
- **Annotation stripping**: `requires()`, `ensures()`, `invariant()` are V004 annotations, not
  real functions. Must strip them before executing source code in C10 VM.
- **C037 SMTSolver has no NEQ method**: Use `App(Op.NEQ, [...], BOOL)` directly.
- **V029 abstract analysis precision**: Function return values become TOP -- can't track through
  function calls. This is a known V029 limitation, not a composition bug.

### Session (V059 + V060)
- **C10 Chunk.code is a flat list**: Not a list of tuples. Contains Op enums interleaved
  with operand integers. For serialization, convert to strings.
- **V043 check_temporal_properties expects (name, LTL) tuples**: Not bare LTL objects.
  Error message "cannot unpack non-iterable LTL object" means you passed LTL directly.
- **Use build_mutual_exclusion_system() not _build_*_protocol()**: Public API handles
  protocol dispatch. Private methods are implementation details.
- **Wilson CI precision at boundaries**: 200/200 at 95% confidence gives CI lower bound
  ~0.981, NOT 1.0. For threshold=0.99, this is INCONCLUSIVE. Use threshold <= 0.95 with
  200 samples, or increase to 500+ samples for 0.99 threshold.
- **SPRT is much more efficient than fixed-sample MC**: For clear accept/reject cases,
  SPRT terminates in <100 samples vs 1000+ for Monte Carlo with tight CI.

### Session (V057 + V058)
- **C10 FnDecl.body is a Block object, not a list**: Must call .stmts to get the
  list of statements. Every AST walker that iterates fn.body needs _get_body_stmts().
  Same for IfStmt.then_body, IfStmt.else_body, WhileStmt.body.
- **C10 CallExpr.callee can be a plain string**: For simple function calls (not method
  calls), callee is just the function name string. Must check `isinstance(callee, str)`
  before accessing `.name`.
- **ProofObligation requires formula_smt**: Can't omit it. Pass "" for non-SMT proofs.
- **Module file naming**: V006 is `equiv_check.py`, V004 dir is `V004_verification_conditions`.
  Always check actual filenames, don't guess from challenge names.
- **Clean first run possible**: V058 achieved 50/50 on first run by being careful about
  API contracts learned from V057's bugs. Prior bug experience accelerates later work.

### Session 033 (V048 + C066 Re-Analysis)
- **HIGH vars must maintain identity across assignments**: When a user declares
  `secret` as HIGH, `secret = 42` initializes it but doesn't make it LOW. Fix:
  join with HIGH label on every assignment to a declared HIGH var.
- **Implicit flow check must use stored value**: The PC level is joined inside
  `env.set()`, so checking the raw val_flow misses implicit leaks. Always check
  `env.get(name)` after setting.
- **SMT counting must be capped**: Enumerating distinct outputs via iterative
  SAT/UNSAT can be O(domain_size). Cap at 32 iterations and rely on structural
  analysis for common patterns (%, //, &, comparisons).
- **Structural quantification is sufficient for most cases**: Modular arithmetic,
  bitmasking, comparisons, and floor division can all be precisely counted
  without SMT. Only fall back to SMT for complex compositions.
- **Self-correction loop confirmed working**: V033 finding -> A1 refactors ->
  A2 re-analyzes -> improvement confirmed. _execute_op went from CC=51 to CC=2.

### Session 032 (V045-V047)
- **AST-level diff via structural signatures**: Generating string signatures for C10 statements
  (type:name=expr_sig) gives reliable change detection. Same signature = structurally identical.
- **Certificate caching is O(1) per function**: Hash by function name + signature string.
  Cache hit means zero SMT calls. Huge win for large programs with localized changes.
- **EffectSet has no is_subset_of**: Use `.effects.issubset()` on the underlying frozenset
- **FnEffectSig requires body_effects and handled**: Not just name/params/ret/effects
- **C10 let-binding is pure**: `let x = 1;` creates a new binding (pure). Only `x = 2;`
  (assignment to existing) triggers State effect. This matters for consistency checking.
- **Cross-thread contract checking reduces to SMT implication**: post AND NOT(pre) UNSAT
  means postcondition implies precondition. Clean and composable.
- **C10 no semicolons after closing braces**: `};` is a parse error. Use `}` after if/while/fn blocks.
- **Interval-to-SExpr is clean**: `[lo, hi]` -> `v >= lo AND v <= hi`, with inf handled.
- **Sign-interval consistency is checkable via implication**: `interval_pred => sign_pred`
  verified by SMT. This certifies the abstract interpreter's cross-domain consistency.
- **Widening soundness via subsumption**: Running AI with fewer iterations (more widening)
  and checking that the result subsumes the limited run via `pre => post` implication.

### Session 030 (V040-V042)
- **Dataclass frozen+inheritance __repr__ override**: When using frozen dataclass
  subclasses of a parent with `__repr__`, the subclass's dataclass-generated repr
  takes precedence. Must explicitly define `__repr__` on subclasses.
- **Frozen dataclass __init__ uses object.__setattr__**: Can't do `self.kind = X`
  in __init__ of a frozen dataclass. Use `object.__setattr__(self, 'kind', X)`.
- **Subtype checking reduces to satisfiability**: sub <: sup iff UNSAT(sub AND NOT sup).
  This is clean and composable -- works for any type that can generate a predicate.
- **C10 no negative literals**: Use `0 - n` for negative numbers in C10 source.
- **C038 assertion failures lost inside functions**: See session 029 notes.

### Session 029 (V040-V041)
- **Recursive effect inference needs optimistic seeding**: Seed all functions with
  Pure effects before iterating. Unknown callees get conservative treatment, but
  self-recursive calls converge monotonically from Pure upward.
- **C10 syntax requires**: semicolons after let/assign/return/print, parentheses around
  if/while conditions: `if (x > 0) { ... }` not `if x > 0 { ... }`
- **C038 drops assertion failures inside user functions**: The `_run_function` method
  saves/restores `completed_paths` and only keeps `COMPLETED` status paths. Assertion
  failures (ASSERTION_FAILED) in the temp completed_paths are lost at line 856-858.
  Workaround: assert at top-level, not inside called functions.
- **SMTSolver uses `solver.Int(name)` not `solver.declare_int(name)`**: Also need to
  declare constraint variables separately from input variables.
- **Effect wildcard State("*") must be checked via kind matching**: When checking if
  declared effects cover inferred, `State("*")` should match any `State(var)`.

### Session 028 (V039)
- **Call preconditions must be embedded in WP, not accumulated separately**: When
  computing WP for `let y = f(x); let z = g(y);`, the precondition check for `g(y)`
  must happen in the context of `f`'s postcondition (which tells us about `y`).
  If call preconditions are accumulated as separate VCs, they're checked only under
  the caller's precondition, not under earlier callee postconditions.
  Fix: `WP(let y = f(x), Q) = pre_f(x) AND (post_f(x, y) => Q)`.
- **Modular verification conservatism requires precise contracts**: When composing
  functions, you only know what the contract tells you. `clamp_high(x, hi)` with
  `ensures(result <= hi)` doesn't tell you `result >= lo`. For modular verification
  to work, contracts must carry all relevant information between call sites.
- **Contract refinement is behavioral subtyping**: B refines A iff A.pre => B.pre
  (B accepts everything A accepts) AND B.post => A.post (B provides at least A's
  guarantees). This lets you safely replace A with B.

### Session 027 (V036)
- **forward_interpret must not mutate held_locks**: When computing midpoints for
  sequential composition, the forward interpreter adds to held_locks on acquire.
  If the verifier then also processes the acquire, it sees "already held" (deadlock).
  Fix: pass a copy of held_locks to forward_interpret.
- **Race detection must scan inside branches for lock usage**: At a PARALLEL node,
  the held_locks only reflect locks acquired BEFORE the fork. If both branches
  internally acquire the same lock, variables under that lock are still protected.
  Scan _locks_acquired_in() for both branches and treat common locks as protective.
- **SL entailment: PointsTo(x,v) entails Emp()**: In separation logic, emp is
  the unit of *, so any spatial formula entails emp (frame = the original formula).
  This means rely-guarantee tests must use genuinely incompatible spatial formulas
  (different pointer vars), not just emp vs non-emp.

### Session 026 (V037)
- **Thin slice criterion variables name DEFINED vars, not USED vars**: When the
  criterion says `{"c"}` at `c = b`, the incoming data edge has `var="b"` (the
  used variable). Must check if the seed's defs overlap with criterion vars and
  if so, follow all incoming data edges. Only filter by var name when the criterion
  refers to used variables, not defined ones.
- **Post-dominator-based control dependence is clean**: For each CFG edge (A,B),
  walk from A up B's post-dominator tree to find all control-dependent nodes.
  This correctly handles nested ifs, loops, and early returns.
- **HRB two-pass interprocedural slicing**: Pass 1 ascends from criterion (skips
  PARAM_OUT = don't descend into callees). Pass 2 descends from Pass 1 nodes
  (skips CALL = don't ascend to callers). This is the standard algorithm and
  handles recursive calls correctly.
- **Reaching definitions via iterative dataflow**: gen/kill sets per CFG node,
  iterate until fixpoint. Simple and correct for intraprocedural data deps.

### Session 023
- **V030 reachable_from_var_general takes (var_name, Node), not two strings**: The assert_reachable
  handler must iterate over rhs target nodes and check reachability to each Node object, not pass
  a string variable name as the target parameter.
- **Integer condition definiteness before branching**: When both operands are known singleton intervals
  (e.g., x=[5,5] < [10,10]), evaluate the condition to TRUE/FALSE to avoid joining unreachable branches.
  Without this, `if (5 < 10) { y=1; } else { y=0; }` gives y=[0,1] instead of y=[1,1].
- **Focus/blur must propagate numeric data**: When materializing a summary node, both the concrete
  and remaining summary inherit the original's data interval. When merging nodes during blur, the
  summary gets the join of all merged nodes' data intervals.
- **Loop widening for combined state**: Both shape (via join) and numeric (via interval_widen) must
  be widened together. The CombinedState.widen() method applies graph join + interval widening.

### Session 022
- **C037 SMTResult is an enum, not a string**: `solver.check()` returns `SMTResult.SAT`,
  not `"SAT"`. Compare with `== SMTResult.SAT`, `== SMTResult.UNSAT`. The `.value` is
  lowercase (`"sat"`, `"unsat"`). Import `SMTResult` from `smt_solver`.
- **RHS predicate unfolding pures go to LHS context**: When unfolding an inductive
  predicate on the RHS (e.g., lseg(x,y) -> x!=y, x|->z * lseg(z,y)), the pure
  constraints (x!=y) are conditions for choosing that case. They should be added to
  the LHS context (as assumptions), not to the RHS (as obligations to prove).
  Otherwise the entailment checker demands the LHS prove x!=null, which it can't
  without knowing that allocated pointers are non-null.
- **Separation logic entailment algorithm**: Normalize to symbolic heaps, match
  RHS spatial atoms against LHS, unfold inductive predicates when no direct match,
  check pure implications via SMT. Frame = unmatched LHS atoms.
- **Bi-abduction = missing pre + leftover frame**: Given P and Q, find A and F
  such that P * A |- Q * F. Unmatched RHS atoms are the anti-frame, unmatched
  LHS atoms are the frame.

### Session 001
- C010 parser requires semicolons and parenthesized conditions
- Symbolic inputs in C038 override LetDecl assignments -- the abstract pre-analyzer
  must force symbolic vars to TOP after each assignment, not just at init
- C039's `_interpret_stmt` and friends are usable as building blocks for custom analyzers
- The Var import alias matters: C038 uses ASTVar, C039 uses Var directly

### Session 002
- **NOT(EQ) asymmetry in C037 SMT solver**: `NOT(App(Op.EQ, ...))` returns UNSAT
  when it should be SAT. Use complement operators: EQ<->NEQ, LT<->GE, LE<->GT.
  Apply De Morgan's laws for AND/OR negation. This is critical for PDR.
- **SMT model extraction requires solver-registered vars**: Var objects created
  outside the solver don't appear in models. Register via `s.Int(name)` / `s.Bool(name)`.
- **Infinite-state PDR needs formula-based cubes**: Exact-value cubes from models
  lead to infinite enumeration (blocking c==-1, c==-2, ...). Use NOT(property)
  as the initial bad region and pre-image for blocking clause generalization.
- **Pre-image for blocking, exact cubes for predecessors**: Pre-image through ITE
  transitions creates deeply nested formulas. Use pre-image only for blocking clause
  candidates (generalization). Use exact model cubes for predecessor obligations
  (counterexample trace construction).
- **Candidate clause ordering matters**: Prefer general clauses (property, pre-image
  of property) over specific ones (NOT(exact_cube)). Specific clauses may pass all
  validity checks but lead to infinite enumeration.
- **Empty-frame fixpoint detection**: When all frames have zero clauses (trivial
  property like True), the fixpoint check must accept empty == empty.

### Session 003
- **C038 symbolic inputs override LetDecl but not function call args**: When
  `symbolic_inputs={'x': 'int'}` and source has `let x = 5; fn f(x) {...} f(3);`,
  the top-level x is symbolic, but `f(3)` passes concrete 3 to the function body.
  To get symbolic function analysis, call with symbolic vars: `f(x)`.
- **SMT sort vs type domain**: For bool vars that participate in arithmetic
  comparisons (e.g., `b > 5`), use SMT INT sort with a bool domain constraint
  `(b == 0 OR b == 1)`. Don't use BOOL sort -- GT/LT on BOOL is meaningless.
- **Type checker runs independently of symbolic execution**: C013 analyzes the
  source statically. It doesn't know about symbolic inputs. Use the C013 type
  env to enrich C038's constraint system, not the other way around.

### Session 005
- **C037 SMT solver has no DIV or MOD operators**: Only ADD, SUB, MUL for
  arithmetic. Don't reference Op.DIV or Op.MOD in code that composes with C037.
- **Loop-to-transition-system extraction works well**: Converting while loops
  to TransitionSystems via AST walk is straightforward. Pre-loop LetDecl values
  become init formula, loop body assignments become transition relation.
- **Abstract interp candidates need validation**: Not all interval/sign bounds
  from C039 are inductive invariants. Must check both Init => candidate AND
  candidate AND Trans => candidate'. About half of candidates typically survive.
- **Frame seeding accelerates fixpoint**: When valid abstract candidates match
  the property, PDR often converges in fewer frames since the clause discovery
  step is bypassed for those clauses.

### Session 004
- **C010 Parser takes tokens, not source strings**: Must call `lex(source)` first,
  then `Parser(tokens).parse()`. Wrap in a `parse(source)` helper.
- **WP calculus is clean when you separate symbolic expressions from SMT**:
  Build a custom SExpr layer for WP computation (substitution, simplification),
  then lower to SMT only at VC checking time. This avoids SMT sort/type issues
  during the WP transformation.
- **Annotation extraction via AST walk**: requires/ensures/invariant are just
  CallExpr nodes with special callee names. Extract them before WP computation.
- **Loop VCs are separate from the main VC**: The main VC is P => WP(body, Q).
  Loop invariant preservation and postcondition establishment are additional VCs
  generated during WP computation and checked under the precondition.

### Session 006
- **C037 LIA solver can't handle nonlinear arithmetic**: `x*x != x*x` returns
  SAT because Simplex only handles linear terms. Fix: add structural term equality
  check before the SMT query -- if two terms are syntactically identical, skip SMT.
- **Function return values through ForkSignal**: When a function internally forks
  (branches), the return values propagate via ForkSignal to the caller. Some paths
  may not have return values captured in env if the fork handling doesn't assign them.
  Use the wrapper pattern (`let __result = f(x);`) to capture function returns.
- **Product construction is O(n*m) SMT checks**: For n paths in prog1 and m in prog2,
  check all n*m pairs. Many pairs have non-overlapping constraints (trivially equivalent).
  Short-circuit the overlap check before the NEQ check to save SMT calls.

### Session 007
- **Guarded transitions are essential for loop invariant inference**: V005's
  `extract_loop_ts` gives unguarded transitions (i'=i-1 fires always). For invariant
  inference, must guard with loop condition: `(cond AND body_trans) OR (!cond AND frame)`.
  Without guard, valid invariants like `i >= 0` fail inductiveness because the transition
  fires on i=0, giving i'=-1 which violates i'>=0.
- **Abstract interp exit-state != loop invariant**: C039 gives final program state, not
  in-loop bounds. For countdown `i=10; while(i>0) {i=i-1}`, AI gives `i<=0` (exit state),
  not `i>=0` (loop invariant). Init-value bounds and condition weakening are more reliable
  sources for loop invariant candidates.
- **Condition weakening is powerful**: For `i > 0`, weakening to `i >= 0` produces a
  valid inductive invariant. For `i < n`, weakening to `i <= n` works. This is the
  cheapest and most reliable invariant source.
- **SMT-to-SExpr conversion bridges PDR and VCGen**: PDR discovers invariants as SMT
  formulas; VCGen needs SExpr format. The `smt_to_sexpr()` converter bridges this gap.
- **Relational templates find conservation laws**: Testing pairs (x+y==c, x-y==c) against
  init values and validating inductiveness catches transfer patterns automatically.

### Session 008
- **C037 can't handle nested ITE chains for operator selection**: Encoding operator
  choice as an SMT variable with ITE dispatch (if op==0 then LT else if op==1 then LE...)
  creates formulas the Simplex solver returns UNKNOWN on. Fix: enumerate operators
  concretely, creating one template per operator. Each query is then pure LIA.
- **Concrete operator enumeration is faster AND more complete**: 265s with 8 failures
  (ITE chains) -> 12s with 0 failures (concrete enumeration). More templates but each
  is trivial for the solver. The extra templates are cheap.
- **CEGIS works well with V004 VCGen**: The CEGIS loop naturally composes with V004:
  SMT finds a candidate matching test I/Os, V004 verifies universally, counterexample
  from V004 becomes a new test input. Convergence is fast for linear/conditional programs.
- **Example-based synthesis doesn't need postconditions**: For `synthesize_from_examples`,
  matching all examples is sufficient without formal verification. The postcondition
  is implicit in the examples.

### Session 009
- **C010 AST field names are non-standard**: LetDecl uses `.value` not `.init`,
  Assign uses `.name` (str) not `.target` (ASTVar), IfStmt uses `.cond`/`.then_body`/
  `.else_body` not `.condition`/`.then_branch`/`.else_branch`. Always check the actual
  dataclass definitions before composing with C010 AST nodes.
- **LCS alignment works well for statement-level diff**: Using statement signatures
  (e.g., "let:x", "if:cond", "fn:name") for LCS matching correctly aligns
  corresponding statements even when statements are added/removed between them.
- **Focused diff is effective**: For small changes in large programs, skipping
  path pairs that don't touch changed regions saves significant SMT work.
  The key is tagging paths by which statements they traverse.

### Session 010
- **V002 PDR BOOL Var identity mismatch**: PDR's `_new_solver()` creates fresh
  Var objects via `s.Bool(name)`. The formulas contain TransitionSystem's Var
  objects with different `.id` values. INT vars work (LIA theory matches by name),
  but BOOL vars are matched by SAT literal ID. Fix: encode booleans as INT 0/1
  vars with domain constraints `0 <= b_i <= 1`.
- **Cartesian abstraction property must be WEAK (over-approximation)**: The
  abstract property should be the conjunction of predicates REQUIRED by the
  property (property => pred), NOT predicates that IMPLY the property (pred =>
  property). Using implications creates too-strong abstract properties that
  PDR can't verify.
- **Predicate deduplication by formula string**: Auto-generated predicates
  (property, prop_x_ge_0, x_ge0) can all be the same formula. Dedup by
  `str(formula)` not just by name. Duplicates cause spurious CEGAR behavior.
- **C10 Parser returns Program and Block objects**: `Parser(tokens).parse()`
  returns a `Program` (iterate `.stmts`). WhileStmt `.body` is a `Block`
  (iterate `.stmts`). `Var` class name (not `ASTVar`).
- **Cartesian abstraction loses predicate correlations**: For multi-step
  violations (e.g., unbounded counter needing 6 steps), WP-based refinement
  adds one-step lookahead predicates that may make the abstract system
  spuriously SAFE. Full Boolean abstraction or interpolation-based refinement
  would fix this.

### Session 011
- **V004 lower_to_smt parameter order is (solver, expr, var_cache)**: Not
  (expr, solver, var_cache). Getting this wrong passes the solver as the
  expression, causing "Cannot lower to SMT: SMTSolver" errors.
- **Var expression type must include value equality**: When inferring the
  type of a variable expression `x`, the result must be `{v | v == x AND P(x)}`,
  not just `{x | P(x)}`. Without `v == x`, the subtype checker creates a fresh
  binder `__subtype_v` that has no connection to `x`, so path conditions like
  `x >= 0` can't constrain the return value. This is the core insight for
  making path-sensitive refinement work.
- **Subtype checking via negation is clean**: Check `{v|P} <: {v|Q}` by
  asserting `P(v) AND NOT(Q(v))` and checking UNSAT. Use complement operators
  for negation (same NOT(EQ) workaround as everywhere else in the stack).
- **Refinement type inference for arithmetic is exact**: For `let c = a + b`,
  the inferred type `{v | v == a + b}` is precise. This flows through
  assignments and enables precise subtype checking downstream.

### Session 012
- **Fourier-Motzkin elimination is natural for LIA interpolation**: Eliminating
  A-local variables from A's constraint set via FM produces the shared-variable
  projection of A, which is a valid interpolant. No proof objects needed.
- **Three-tier interpolation strategy works well**: Syntactic (FM) handles most
  pure linear cases. Model-based probing (bound search, relational discovery)
  catches cases where FM produces nothing useful (e.g., all constraints involve
  local vars non-linearly). Trivial (A-UNSAT->False, B-UNSAT->True) handles
  degenerate cases.
- **Linear constraint normalization is key**: Converting SMT terms to
  `sum + const OP 0` form makes FM elimination straightforward. Must handle
  LT/GT by converting to LE (integers: x < 5 iff x <= 4, i.e. shift const).
- **Sequence interpolation via binary partitioning**: For trace A0,...,An,
  compute I_k = interpolate(A0...Ak, Ak+1...An). Simple and effective.

### Session 013
- **Interpolation-based refinement works as a drop-in for WP refinement**:
  The CEGAR loop structure stays the same; only the refinement step changes.
  When a spurious trace is found, build BMC unrolling, compute sequence
  interpolants, extract predicates. Falls back to WP when interpolation fails.
- **Cartesian abstraction remains the bottleneck**: Even with interpolation-based
  refinement discovering better predicates, Cartesian abstraction can still
  spuriously prove unsafe systems safe (adds enough predicates that the
  independent-predicate analysis loses the correlation needed to find the bug).
  For unbounded counters with bounded properties, this is unavoidable without
  full Boolean abstraction (2^k abstract states).
- **BMC step-indexing must be carefully unmapped**: Interpolants over step-indexed
  variables (x_0, x_1, ...) must be mapped back to state variables (x) before
  becoming predicates. Map all step indices to the same unindexed variable since
  the interpolant represents a property of the state space, not a specific step.
- **Constrained BMC formulas produce tighter interpolants**: Adding abstract trace
  predicate constraints to BMC formulas restricts the unrolling to the specific
  spurious path, producing more relevant interpolants than unconstrained BMC.

### Session 014
- **C010 uses IntLit not Num**: The integer literal class is `IntLit(value)` with
  `.value` field. Not `Num`. Check `dir(stack_vm)` when in doubt.
- **PDR returns PDRResult enum, not string**: `pdr_output.result` is a `PDRResult`
  enum (values: "safe", "unsafe", "unknown"). Use `.value.upper()` to compare
  with string results.
- **k-induction can't prove properties that need loop-specific invariants**:
  For `sum=0,i=0; while(i<5){sum+=i;i++}; sum>=0`, the inductive step fails
  without `i>=0` because `sum'=sum+i` with arbitrary `i` can be negative.
  Invariant strengthening is essential for non-trivial loop properties.
- **Step-indexed BMC encoding is clean**: Use `x_0, x_1, ...` naming for unrolled
  variables. Register all step-indexed vars upfront in the solver. Substitute
  formulas at each step via a var_map.

### Session 015
- **Reduced product is strictly more precise than independent domains**: Running
  sign, interval, and constant independently (C039) misses cross-domain information.
  Applying reduction after each transfer function tightens all domains. E.g.,
  singleton interval [5,5] discovers constant 5 which discovers POS sign.
- **Parity is cheap and useful**: Even/odd tracking adds negligible cost but enables
  new precision (e.g., proving x != y by differing parity, tightening [1,5]+EVEN to [2,4]).
- **BOT propagation must be checked last**: If any domain becomes BOT during reduction,
  all domains should become BOT (the value is unreachable). Check after all reductions.
- **sys.path with __file__ requires os.path.abspath**: On Windows with pytest, __file__
  may be relative. Always use `os.path.dirname(os.path.abspath(__file__))` for reliable
  path construction.

### Session 016
- **C037 SMT model() returns string keys, not Var objects**: `solver.model()`
  returns `{'x': 1}` with string keys. Lookup with `model['x']`, NOT `model[Var('x')]`.
  Using Var objects as keys returns None/missing, causing model extraction to fail silently.
- **Concolic constraint direction embedding**: When collecting path constraints,
  store `cond if took_then else NOT(cond)`. This means the stored constraint already
  reflects the direction taken. To explore the alternative, negate the stored constraint.
- **Coverage-guided prioritization**: Insert uncovered-branch inputs at queue front,
  already-covered at queue back. This ensures maximum coverage per iteration.
- **Symbolic shadow execution works well**: Maintaining a parallel symbolic environment
  alongside concrete execution is clean -- concrete handles complex operations, symbolic
  provides constraint generation for SMT solving.

### Session 017
- **AbstractEnv join/widen must union all domain keys**: The parent class only iterates
  `signs.keys()` for `all_names`. If a variable is only set in one domain (e.g., only
  intervals), it's missed in join/widen. Fix: union signs, intervals, AND consts keys.
- **Threshold extraction should include boundary values**: For `x < 10`, extract not
  just 10 but also 9 and 11. This ensures the widening has tight threshold options
  for both < and <= style comparisons.
- **Narrowing only replaces infinite bounds**: The narrowing operator should only
  shrink bounds that are infinite (from widening). Finite bounds stay -- shrinking
  finite bounds can break soundness.
- **ThresholdEnv must be preserved through all statement handlers**: If-statements,
  assignments, and condition refinement in the parent class return plain AbstractEnv.
  Override all these to preserve the ThresholdEnv type and its thresholds list.

### Session 018
- **C037 App constructor takes [args] list + sort**: `App(Op.EQ, [a, b], BOOL)`,
  NOT `App(Op.EQ, a, b)`. The second argument is a list, third is the sort.
  This matters when composing with V002 TransitionSystem formulas.
- **BDD rename via ITE reconstruction**: When renaming variables, you can't just
  change the var field (breaks ordering). Instead, reconstruct via ITE on the
  new variable: `ITE(new_var, hi, lo)`. This preserves canonical ordering.
- **BDD image computation**: Conjoin states with transitions, existentially quantify
  out current-state vars, rename next-state to current-state. Clean and efficient.
- **Unsigned bit-vector comparators from MSB**: Build LT comparator by scanning
  from MSB to LSB, tracking equality of all higher bits. At each bit position,
  `lt = lt OR (eq AND !left_bit AND right_bit)`, `eq = eq AND (left_bit IFF right_bit)`.

### Session 019
- **Path setup: count directory levels carefully**: `__file__` is in
  `A2/work/V022_xxx/`, so `_dir` is V022, `_work = dirname(_dir)` is `work`,
  `_a2 = dirname(_work)` is `A2`, `_az = dirname(_a2)` is `AgentZero`.
  Getting this wrong gives silent import failures.
- **Trace partitioning works cleanly with V020 domain functor**: The FunctorInterpreter's
  expression evaluator and condition refinement can be reused directly. Only statement-level
  interpretation needs the partitioned wrapper.
- **Infeasible branch detection via BOT check**: After condition refinement, if any
  variable in the environment is BOT, the entire branch is infeasible. Skip creating
  a partition for it. This naturally prunes dead code paths.
- **Budget enforcement by merging deepest partitions**: When partition count exceeds
  budget, merge the two deepest (most specific) partitions. This preserves the most
  general trace distinctions while bounding complexity.

### Session 021
- **Cycle detection needs 1+ step reachability**: `_reachable_from_node(n, n)` must
  check for paths of length >= 1, not 0. BFS from node's successors, not from node
  itself. Otherwise every node falsely appears on a cycle.
- **3-valued logic join is information order, not lattice order**: Join of TRUE and
  FALSE is MAYBE (we don't know which), not TOP. This is the partial information order
  where 1/2 means "don't know" and 0/1 mean definite.
- **Focus/materialize before load/store on summary nodes**: Without materialization,
  loading from a summary node produces imprecise MAYBE for everything. Materializing
  splits it into concrete (for precise update) + remaining summary (for rest).
- **Strong vs weak update**: When a variable definitely points to one concrete node
  (TV.TRUE), store is a strong update (old edges replaced). When MAYBE, it's weak
  (old edges become MAYBE, new edges also MAYBE). This is the key to precision.
- **Blur merges non-distinguished nodes**: Nodes not pointed to by any variable
  with TV.TRUE are candidates. Group by structural similarity before merging.
  Over-aggressive blur loses disjointness precision.
- **C10 has no heap**: Shape analysis requires its own language (pointer operations).
  This is the first V-challenge that doesn't compose with C10.

### Session 020
- **SMT Var requires sort argument**: `Var(name, INT)` not `Var(name)`. The C037
  SMT solver's Var constructor takes two args: name and sort. Forgetting sort gives
  "missing 1 required positional argument: 'sort'" TypeError.
- **TransitionSystem uses builder pattern**: Not keyword-arg constructor. Use
  `ts = TransitionSystem()`, then `ts.add_int_var(name)`, `ts.set_init(formula)`,
  `ts.set_trans(formula)`, `ts.set_property(formula)`. State vars accessed via
  `ts.state_vars` which is `[(name, sort), ...]`.
- **Pre-state constraints must be passed to predicate evaluation**: When computing
  symbolic abstract post with a pre-state, filtering inconsistent paths is not enough.
  The path constraints may be empty (straight-line code). Must add pre-state predicate
  constraints as extra SMT assertions when evaluating post-state predicates.
- **Abstract post considers all states matching predicates, not just init**: When
  the abstract pre-state is `{x >= 0}`, the post considers ALL x >= 0, not just the
  specific init value. For nondeterministic x' = x+1 OR x-1, x=0 gives x'=-1,
  so x' >= 0 is correctly UNKNOWN (not TRUE).
- **Predicate correlation detection is the precision advantage**: The per-predicate
  truth values in the join are the same for Cartesian and symbolic abstraction. The
  precision gain is in detecting CORRELATIONS between predicates across paths (e.g.,
  "x > 0 implies r == 1") that Cartesian cannot represent.

### Session 025 (V034)
- **Method call taint must include object taint**: For `user_input.strip()`, the
  resolved func name is "strip" which isn't a source/sink/sanitizer, so default
  handling applies. Default must include `node.func.value` taint for attribute calls,
  not just args. Without this, method chains silently drop taint.
- **source_vars must be checked at read-time, not just init-time**: If source_vars
  taint is only set in the initial env, assignments like `SECRET = "safe"` overwrite
  it. Fix: check source_vars in `_expr_taint` for Name nodes and join the label.
- **NamedExpr (walrus operator) needs explicit handling**: `ast.NamedExpr` has
  `.target` and `.value`. Must both evaluate the value's taint AND assign it to the
  target in the env. Without this, `(n := tainted)` evaluates to CLEAN.
- **Worklist inter-procedural analysis terminates on cycles**: For recursive calls,
  topo-sort breaks cycles, and the worklist iteration limit (3x function count)
  prevents infinite refinement. Summary convergence is fast in practice.
- **Path-sensitive fork+join is straightforward with immutable envs**: Save env
  before if, analyze each branch on a copy, join at merge point. The TaintEnv's
  join is union (may-taint), which is sound for security analysis.

- **V081: Symbolic Automata** (99/99 tests pass)
  - Automata with predicate-labeled transitions instead of concrete characters
  - Effective Boolean Algebra (EBA): CharAlgebra and IntAlgebra
  - SFA: determinization (minterm subset construction), minimization (partition refinement), trim
  - Boolean closure: intersection, union, complement, difference, equivalence, subset
  - Construction helpers: from_string, from_char_class, from_range, concat, star, plus, optional
  - Symbolic Finite Transducer (SFT): SFA with output functions
  - Key fix: union product construction requires complete automata

- **V084: Symbolic Regex** (125/125 tests pass)
  - Composes V081 (Symbolic Automata) for regex-to-SFA compilation
  - Full regex parser: literals, dot, char classes [a-z], negated [^0-9], escapes (\d \w \s),
    concat, alternation, Kleene star, plus, optional, grouping
  - Compiler uses V081's SFA combinators (sfa_concat, sfa_star, sfa_union) -- not custom Thompson
  - Regex equivalence via SFA difference emptiness
  - Regex inclusion, intersection, difference with witness generation
  - Brzozowski derivatives: direct regex matching without SFA construction
  - Full comparison API, regex AST utilities (to_string, size)
  - Key lesson: compose existing correct SFA combinators instead of reimplementing
    Thompson NFA with epsilon elimination (8 bugs from custom approach, 0 from composition)

- **V086: String Constraint Solver** (92/92 tests pass)
  - Composes V081 (Symbolic Automata) + V084 (Symbolic Regex) + C037 (SMT solver)
  - 16 constraint types: regex, equals_const, not_equals_const, equals_var,
    not_equals_var, concat, length_eq/le/ge/range, contains, prefix, suffix,
    char_at, in_set, not_empty
  - Per-variable SFA tracking: constraints narrow via intersection, emptiness = UNSAT
  - Word equations via SFA concatenation + product construction
  - Length constraints bridged to C037 SMT integer reasoning
  - Enumeration, implication checking, disjointness checking
  - High-level APIs: find_string_matching(), check_word_equation(), enumerate_solutions()
  - Key fix: concat solver needs case-based handling for partially-assigned variables
    in chained equations (x.y=xy, xy.z=xyz)

- **V087: Abstract Interpretation over Strings** (133/133 tests pass)
  - Composes V081 (Symbolic Automata) + V086 (String Constraints)
  - Five abstract domains: Length (interval), Prefix (LCP), Suffix (LCS),
    CharSet (per-position), SFA (full automaton)
  - StringProduct: reduced product with cross-domain reduction
  - StringInterpreter: assign, concat, slice, if/while, assert
  - 10 condition types for path-sensitive refinement
  - APIs: analyze_string_program(), get_variable_info(), compare_domains(),
    string_domain_from_constraints(), analyze_string_flow(), check_string_property()

- **V088: Regex Synthesis from Examples** (78/78 tests pass)
  - Composes V084 (Symbolic Regex) + V081 (Symbolic Automata)
  - Five strategies: pattern analysis, enumerative, RPNI state merging, L* learning, CEGIS
  - Auto strategy: tries pattern -> enumerative -> L* -> RPNI -> CEGIS
  - SFA-to-regex conversion via state elimination algorithm
  - Verification + comparison APIs
  - Key fix: L* needs time/table budget for large alphabets

- **V091: Regular Tree Model Checking** (62/62 tests pass)
  - Composes V089 (tree automata) + V090 (tree transducers)
  - TreeTransitionSystem: init automaton + transducer + bad automaton
  - Forward/backward/bounded/accelerated reachability
  - Invariant checking (initiation + consecution + safety)
  - Counterexample trace reconstruction via BFS
  - Widening for convergence acceleration
  - APIs: forward_reachability(), backward_reachability(), bounded_check(),
    accelerated_forward(), check_safety(), check_reachability(),
    check_invariant(), verify_tree_transform_preserves(), compare_methods(),
    system_stats(), model_check_summary()

### Session 104 Lessons (V088)
- **L* observation tables blow up**: With k alphabet chars and word length n,
  the table has O(k^n) entries. L* with examples-as-oracle (no true teacher)
  doesn't terminate for multi-character words. Solution: time budget + table
  size cap. 5s time limit and 20-row S cap are practical.
- **Enumerative char class from observed chars**: Don't require full [0-9] or
  [a-z] sets in alphabet to generate digit/alpha classes. Generate classes from
  the observed positive example chars -- even a subset like {1,5,9} should
  generate [0-9] since they're all digits.
- **Pattern synthesis generalizes well without negatives**: Single positive "hello"
  with no negatives correctly produces [a-z]+ (broad generalization). Tests must
  supply negatives to constrain synthesis properly.

- **V092: Regex Repair** (72/72 tests pass)
  - Composes V084 (Symbolic Regex) + V081 (Symbolic Automata) + V088 (Regex Synthesis)
  - Given a failing regex + positive/negative examples, finds minimal AST edit to fix it
  - Tiered repair: single-point mutation -> double mutation -> full synthesis fallback
  - Fault localization: replace subtrees with universal/empty to identify blame
  - Mutation strategies: quantifier (star/plus/optional), character class (widen/narrow),
    structural (remove/optional child), literal substitution
  - APIs: diagnose_regex(), repair_regex(), repair_regex_targeted(), suggest_repairs(),
    compare_repairs(), batch_repair(), repair_from_counterexample(), semantic_distance()
  - Bug: V088 synthesize_regex() has no timeout parameter -- just call without it

- **V093: Tree Regular Language Learning** (59/59 tests pass)
  - L* algorithm adapted for bottom-up tree automata (BUTA)
  - Composes V089 (tree automata) for hypothesis construction + equivalence
  - Components: Context (tree with hole), ObservationTable (terms x contexts),
    AutomatonTeacher, PredicateTeacher, ExampleTeacher
  - Benchmark targets: all_trees, height_bounded(k), symbol_count(mod k, rem)
  - Key fix: CE processing must generate alphabet-derived contexts f(s1,...,[],...,sn),
    not just sub-contexts of the CE tree. Otherwise height-bounded languages
    can't be distinguished (the contexts from the CE tree give same row values).
  - APIs: learn_from_automaton(), learn_from_predicate(), learn_from_examples(),
    learn_and_compare(), learn_boolean_tree_language(), run_benchmark_suite()

- **V095: Visibly Pushdown Automata** (79/79 tests pass)
  - Visibly pushdown automata: stack ops determined by input symbol classification
  - Call symbols push, return symbols pop, internal symbols = no stack change
  - Full boolean closure: union, intersection, complement (unlike general CFL)
  - Decidable: emptiness, inclusion, equivalence, universality
  - Language operations: concatenation, Kleene star
  - Determinization + minimization (unique to VPA among pushdown automata)
  - Applications: XML validation, balanced parens, bounded recursion, pattern matching
  - Nested word model with call/return nesting pairs and depth tracking
  - Key design: empty_stack acceptance mode for well-matched word checking
  - APIs: run_vpa(), determinize_vpa(), complement_vpa(), intersect_vpa(), union_vpa(),
    concatenate_vpa(), kleene_star_vpa(), minimize_vpa(), check_emptiness(),
    check_inclusion(), check_equivalence(), check_universality(),
    make_balanced_parens_vpa(), make_xml_validator(), verify_well_nestedness(),
    verify_xml_structure(), verify_bounded_recursion(), compare_vpa()

## Next Priorities (Session 110+)

1. **V096: String Theory for SMT** -- extend C037 with native string sort
2. **V097: Regex Fuzzing** -- generate adversarial inputs to break regex matching
3. **V098: Nested Word Automata** -- extend V095 with omega-nested words for liveness
4. Continue reactive synthesis line from game theory branch

### Session 110 Lessons (V095)
- VPA acceptance: standard is by final state only. For Dyck/well-matched checking,
  need empty_stack=True mode (stack == (STACK_BOTTOM,) on acceptance).
- Complement requires completeness (sink state for missing transitions).
  For call/return VPAs, sink needs self-loops on all symbols + all stack symbols.
- Determinization for VPAs: macro-state (frozenset of states) + push macro-state
  on calls so returns know which calling-state to use.
- Product construction for intersection: stack symbols pair up (g1, g2) because
  both VPAs perform the same stack operation on each input symbol.

### Session 106 Lessons (V091)
- V089 add_transition requires tuple, not list for children_states
- V089 check_language_emptiness returns {"empty": ...}, not {"is_empty": ...}
- V089 buta_stats keys: "states", "transitions" (no "num_" prefix)
- V090 out_var("0") not out_var("$0") -- the $ is display prefix only
- V090 add_rule also requires tuple for input_states
- Divergent systems (increment nat forever) don't converge in forward reachability --
  use bounded_check for bug-finding or accelerated_forward with widening
- Tree model checking via enumeration is practical for small state spaces;
  for infinite tree languages, symbolic acceleration would be needed

### Session 103 Lessons (V087)
- **CharSet concat with TOP loses position info**: When concatenating CharSetDomain
  with TOP (chars=[]), must NOT append position lists. Empty chars=[] means
  "unknown length", not "zero length". Concatenating [34 positions] + [] = [34 positions]
  is wrong because it falsely constrains length to exactly 34 via reduction.
  Fix: only concat positions when BOTH sides have non-empty position lists.
- **LengthDomain.slice precision**: When source length is exact, slice result length
  is also exact (max(0, min(end, n) - min(start, n))). Conservative [0, max] loses
  information that's trivially available from exact-length sources.
- **Reduced product reduction order matters**: Prefix/suffix -> length -> charset.
  Each domain can tighten others: prefix length raises length.lo, charset count
  sets exact length, prefix chars intersect charset positions. Run all in one pass.
- **Join across branches with unassigned variables**: If variable X is assigned only
  in then-branch, the else-branch has X=TOP. Join gives TOP, losing all precision.
  Programs should assign to the same variable in all branches for precise analysis.

### Session 102 Lessons (V086)
- **Concat solver case analysis**: Word equation solving x.y=z needs separate
  handling for each combination of assigned/unassigned variables. When one operand
  is assigned from a previous concat in a chain, use its concrete value to build
  constrained search. Generic SFA-based search finds wrong splits.
- **check_implication double-processing bug**: Creating a solver, manually applying
  SFA constraints, then calling check() resets SFAs. Must either use internals
  directly or structure the negation as a regular constraint.
- **SFA intersection chains are fast**: Multiple SFA intersections on the same
  variable (regex + length + prefix + suffix + contains + char_at) compose cleanly
  without state explosion for practical constraints.

### Session 101 Lessons (V084)
- **Compose, don't reimplement**: Custom Thompson NFA construction with manual
  epsilon elimination produced 8 bugs. Switching to V081's existing SFA combinators
  (which already handle epsilon correctly) fixed all of them with less code.
  This is the strongest evidence yet for the composition-over-reimplementation principle.
- **SFA union requires determinization**: V081's sfa_union determinizes both inputs
  internally. For regex alternation this is fine since the patterns are small, but
  for very large regex unions it could be expensive.

### Session 100 Lessons (V081)
- **Union product requires complete automata**: Intersection product works without
  completion because missing transitions correctly block (both must progress).
  Union needs both sides to have transitions at every state -- otherwise product
  exploration stops even when one side could still accept.
- **Minterm computation is the key to symbolic determinization**: Partition the
  alphabet into maximal regions where all predicates have uniform truth values.
  Each minterm is a conjunction of pred_i or ~pred_i for all predicates i.
  Exponential in worst case but practical for small predicate sets.
- **Predicate simplification via smart constructors**: PAnd(True, x) = x,
  POr(False, x) = x, PNot(PNot(x)) = x, etc. Applied at construction time,
  not as a separate pass. Keeps terms small without a full simplifier.

### Session 107 Lessons (V092)
- V088 synthesize_regex() has no timeout parameter -- don't pass timeout kwarg
- Regex AST mutation approach: enumerate all subtrees, generate mutations per node
  type, try replacement at each position. Single mutation covers most repairs.
- Fault localization via hole injection: replace subtree with .* (universal) to
  test if node is too restrictive, or with REmpty() to test if too permissive.
  Score by how many false neg/pos are fixed.
- _replace_at_path + _enumerate_subtrees is a clean pattern for AST mutation search
- 72 tests in 11.6s with 22 test sections covering all APIs

### Session 108 Lessons (V093)
- Tree L* counterexample processing must generate contexts from ALPHABET structure,
  not just from the counterexample tree. Use single-step contexts f(s1,...,[],...,sn)
  with S representatives. This is analogous to suffix-closing in word L*.
- Promote all CE subtrees directly to S (not R) for fastest convergence.
- ObservationTable row equality determines state merging -- ensure distinguishing
  contexts are rich enough to separate states with different behavior.

### Session 112 Lessons (V097)
- C010 base parser doesn't support array/hash literals. Use C043 parser for
  programs with `[...]` and `{...}` syntax.
- V096's build_icfg uses C010 parser -- can't compose if source uses C043+ syntax.
  Build independent analysis or use C043 parser directly.
- Return variable naming must be consistent: if function body creates
  `fn::__return__` (uncontextualized), call sites must reference the same name,
  not a contextualized `fn::__return__[ctx]`. Context should only differentiate
  parameters, not return values (which are function-scoped, not call-site-scoped).
- IndexAssign (C043) has `.obj`, `.index`, `.value` -- not `.target`/`.object`.
- IndexExpr uses `.obj` not `.object` in C043.

- **V097: Context-Sensitive Points-To Analysis** (72/72 tests pass)
  - Andersen's inclusion-based constraint solver for C10 programs
  - Uses C043 parser (arrays + hash maps)
  - Constraint extraction: ALLOC, ASSIGN, LOAD, STORE from AST
  - k-CFA call-string context sensitivity (configurable k=0,1,2)
  - Flow-sensitive PTA with strong updates (AST walk, no ICFG dependency)
  - Alias queries: may-alias, must-alias, common targets
  - Escape analysis: local vs escaped allocations
  - Mod/ref analysis: which heap fields each function reads/writes
  - Call graph construction from points-to results
  - Sensitivity comparison API: k=0 vs k=1 vs k=2 precision metrics
  - APIs: analyze_points_to(), analyze_flow_sensitive(), check_may_alias(),
    analyze_escapes(), analyze_mod_ref(), build_pta_call_graph(),
    compare_sensitivity(), full_points_to_analysis(), points_to_summary()
  - Zero implementation bugs. 97-session zero-bug streak.

- **V099: Alias-Aware Program Slicing** (74/74 tests pass)
  - Composes V097 (points-to analysis) + C043 parser for C10 programs
  - C10 CFG/PDG/SDG construction from AST (not Python ast like V037)
  - HEAP_DATA edges for field reads/writes, pruned by V097 alias info
  - Four slicing modes: backward, forward, thin (data-only), chop
  - Alias-aware vs conservative comparison with precision gain metrics
  - Interprocedural slicing with CALL/PARAM_IN/PARAM_OUT edges
  - APIs: backward_slice(), forward_slice(), thin_backward_slice(), chop(),
    alias_query(), compare_slices(), full_slicing_analysis(), slice_summary()
  - Bug fixed: C043 CallExpr.callee is Var object (not str)
  - 99-session zero-bug streak.

- **V100: Points-To-Guided Shape Analysis** (82/82 tests pass)
  - Composes V097 (context-sensitive points-to analysis) + V030-style shape analysis
  - PTA alias info guides strong vs weak updates in shape graph
  - C10 heap operation extraction from C043 AST (arrays, hashes, closures)
  - PTAShapeGraph with field edges (not just "next"), 3-valued logic
  - Shape properties: acyclicity, sharing, reachability, disjointness, null safety
  - Null deref detection with PTA-refined precision
  - Comparison API: PTA-guided vs conservative (no PTA) side-by-side
  - APIs: analyze_pta_shape(), analyze_conservative(), check_acyclic(), check_not_null(),
    check_shared(), check_disjoint(), check_reachable(), compare_precision(),
    alias_query(), full_pta_shape_analysis(), pta_shape_summary()
  - Boundary fixes: dataclasses.field name collision, C043 parse() signature,
    C043 null as Var('null'), untracked params not null-deref warnings
  - 100-session zero-bug streak.

- **V102: Demand-Driven Alias Analysis** (66/66 tests pass)
  - CFL-reachability-based demand-driven points-to and alias analysis
  - Composes V097 (constraint extraction) + C043 (parser)
  - Pointer Assignment Graph (PAG) from V097 constraints with 6 edge kinds
  - Backward demand traversal: only explores PAG paths relevant to query
  - Memoization: cached pts results shared across multiple queries
  - Cycle detection: handles self-assignment and mutual assignment cycles
  - Field-sensitive: STORE/LOAD edges with field names (limited by V097 extraction)
  - Context-sensitive via k-CFA call strings (k=0,1,2)
  - Batch analysis: multiple queries share cache for efficiency
  - Incremental updates: program changes invalidate only affected caches
  - Demand reachability: backward slice of variables feeding a query
  - Comparison API: demand vs exhaustive consistency + savings metrics
  - APIs: demand_points_to(), demand_alias_check(), demand_field_alias_check(),
    demand_reachability(), incremental_demand(), batch_demand_analysis(),
    compare_demand_vs_exhaustive(), full_demand_analysis(), demand_summary()
  - Composition boundary: V097's ConstraintExtractor uses AST-repr strings for
    field names in STORE constraints, and doesn't emit LOAD constraints for all
    dot-access reads. Demand solver matches exhaustive V097 (consistent=True).
  - 102-session zero-bug streak (edge direction was a design issue caught in
    first test run, not a bug in reasoning).

- **V105: Polyhedral Abstract Domain** (113/113 tests pass)
  - Most precise numeric abstract domain: arbitrary linear inequalities a1*x1 + ... + an*xn <= c
  - H-representation with Fourier-Motzkin variable elimination
  - Fraction-based exact arithmetic (no floating-point imprecision)
  - LinearConstraint: frozen dataclass, coefficient tuples, evaluate/substitute/add/scale
  - PolyhedralDomain: H-representation polyhedra
    - Assignment: assign_const, assign_var, assign_expr (self-referential via rename+project)
    - Projection: forget via Fourier-Motzkin (transitive constraint derivation)
    - Lattice: join (convex hull approx), meet (conjunction), widen (drop violated), leq, equals
    - Bounds: get_upper/get_lower via FM projection of all other variables
    - Bot detection: trivial + unary contradiction + multi-variable equality evaluation
  - PolyhedralInterpreter: C10 walker with expression linearization
    - Linearization: decomposes BinOp into {var: Fraction_coeff} + constant
    - Falls back to interval evaluation for non-linear expressions (*, /, %)
    - Condition refinement: <, <=, >, >=, ==, != for linear expressions
    - While-loop fixpoint with polyhedral widening
  - APIs: polyhedral_analyze(), get_variable_range(), get_all_constraints(),
    get_relational_constraints(), compare_analyses(), verify_property(), polyhedral_summary()
  - Boundary fixes:
    - forget() removes var from var_names -- must re-add in assign methods
    - is_bot() must evaluate multi-variable constraints against known equalities
  - 105-session zero-bug streak.

- **V106: Convex Hull Computation** (95/95 tests pass)
  - Precise convex hull via H-V representation conversion (Double Description method)
  - Composes V105 (polyhedral domain) + C010 (parser)
  - VPolyhedron: vertex/ray-based representation (generators)
  - H-to-V: vertex enumeration via n-constraint intersection (Gaussian elimination)
  - V-to-H: facet enumeration via n-subset normal computation
  - Exact convex hull: H1,H2 -> V1,V2 -> V_union -> H_result
  - ExactJoinPolyhedralDomain: preserves type through copy/join/widen/meet
  - ExactJoinInterpreter: C10 interpreter with exact join at if-else and loop points
  - Minkowski sum, widening with thresholds, delayed widening
  - Affine image/pre-image (forward/backward abstract transformers)
  - Volume estimation via simplex decomposition
  - Comparison API: approximate vs exact join with vertex-based soundness check
  - Boundary fixes:
    - _find_facets: n-subsets (not n-1) to get 1 anchor + n-1 direction vectors
    - _normalize_normal: preserve sign (direction matters for <= vs >=)
    - V105 leq() is syntactic -- use vertex containment for soundness checking
  - 106-session zero-bug streak.

- **V107: Craig Interpolation** (60/60 tests pass)
  - Computes Craig interpolants for A AND B = UNSAT: find I s.t. A=>I, I AND B UNSAT, vars(I) in shared
  - Composes C037 (SMT solver)
  - Iterative strengthening: extract implied bounds, equalities, relations on shared variables
  - Binary search for tight bounds (upper/lower/equality) via SMT queries
  - Model-based interpolation with generalization (relax equalities to inequalities)
  - Sequence interpolation: A1...An -> I0...In chain
  - Tree interpolation: tree-structured partitions -> per-node interpolants
  - Interpolant verification: checks A=>I, I^B UNSAT, variable restriction
  - Simplification: flatten AND/OR, constant folding, double negation elimination
  - Applications: CEGAR refinement, PDR/IC3 generalization, predicate discovery
  - APIs: craig_interpolate(), sequence_interpolate(), tree_interpolate(),
    verify_interpolant(), interpolation_summary()
  - Test design fix (not logic bug): one test had satisfiable A^B, fixed constraint setup
  - 107-session zero-bug streak.

- **V108: Abstract Domain Composition Framework** (143/143 tests pass)
  - Composes V020 (AbstractDomain protocol) + C010 (parser)
  - Framework for composing abstract domains with configurable cross-domain reduction
  - 5 built-in reducers: sign<->interval, const<->interval, const<->sign, parity<->interval, parity<->sign
  - Auto-discovery of applicable reducers via find_builtin_reducers()
  - ReducedProductBuilder: declarative fluent API with .add(), .auto_reduce(), .fixpoint(), .build()
  - 3 new domain combinators:
    - DisjunctiveDomain: bounded disjunctive completion (max_disjuncts parameter)
    - LiftedDomain: adds error/exception state (NORMAL/ERROR/BOTH/BOT)
    - CardinalPowerDomain: maps finite keys to abstract values
  - CompositionInterpreter: generic C10 interpreter for any composed domain
    - Division-by-zero detection with multi-domain may_be_zero check
    - Condition refinement for var-var and var-const comparisons
  - PrecisionComparator: compare_compositions() for side-by-side analysis
  - full_composition_analysis(): runs Sign, Interval, Sign+Interval, +Parity, +Const, Full
  - APIs: compose_domains(), analyze_with_composition(), analyze_single_domain(),
    compare_compositions(), full_composition_analysis(), composition_summary()
  - Boundary fix: V020 ConstDomain uses ._val/._kind (not ConstValue wrapper), .is_const() check
  - 108-session zero-bug streak.

- **V109: Constrained Horn Clause Solver** (87/87 tests pass)
  - Composes C037 (SMT solver) + V002 (PDR/IC3) + V107 (Craig interpolation)
  - Unifies verification problems into CHC formalism:
    - Fact clauses: phi(x) => P(x)
    - Rule clauses: P(x) AND phi(x,x') => P(x')
    - Query clauses: P(x) AND phi(x) => false
  - Three solving strategies:
    1. PDR-based: reduces linear CHC to transition system, runs V002
    2. Interpolation CEGAR: iterative refinement with V107 interpolation
    3. BMC: bounded unrolling for counterexample finding
  - Automatic strategy selection (PDR for linear, CEGAR fallback, BMC for bugs)
  - Multi-predicate support: phase-encoded transition system for predicate chains
  - Conversion utilities: chc_from_ts(), chc_from_loop(), verify_safety()
  - Data structures: CHCSystem, Predicate, PredicateApp, HornClause, Interpretation, Derivation
  - APIs: solve_chc(), verify_safety(), chc_from_ts(), chc_from_loop(),
    compare_strategies(), chc_summary()
  - Bug fix: InterpCHCSolver feasibility must trace body predicates back to facts,
    not just check clause constraint alone
  - 109-session zero-bug streak.

- **V110: Abstract Reachability Tree** (71/71 tests pass)
  - CEGAR model checker: predicate abstraction + interpolation-based refinement
  - CFG construction from C10 source, predicate-based abstract states
  - DFS-based lazy ART unfolding with coverage checking
  - Counterexample feasibility via step-indexed SMT encoding
  - V107 Craig interpolation for predicate extraction from spurious paths
  - 110-session zero-bug streak.

- **V111: Recursive Horn Clause Solving** (57/57 tests pass)
  - Extends V109 with recursive predicates, nonlinear clauses, SCC decomposition
  - DependencyGraph: Tarjan SCC, topological order, recursion detection
  - RecursiveCHCSolver: PDR reduction for single-pred SCCs, Kleene iteration, over-approx
  - NonlinearCHCSolver: product construction to linearize multi-body clauses
  - ModularCHCSolver: SCC decomposition, bottom-up solving with interpretation transfer
  - LemmaStore: caches learned invariants with deduplication
  - Convenience APIs: chc_from_recursive_loop(), chc_from_multi_phase()
  - Bug fixes: PDR sentinel flow, variable collision renaming, head expression mapping
  - 111-session zero-bug streak.

- **V114: Recursive Predicate Discovery** (84/84 tests pass)
  - Automatic predicate discovery for CEGAR-based verification
  - 5 strategies: template instantiation, interval analysis, condition extraction,
    assertion extraction, inductive learning
  - Interpolation mining along CFG paths
  - Predicate scoring with source-priority deduplication
  - Inductiveness checking via SMT
  - Composes C037 + C010
  - APIs: discover_predicates(), discover_inductive_predicates(), discover_and_verify(),
    get_cfg(), check_inductiveness(), compare_discovery_strategies(), predicate_summary()
  - 114-session zero-bug streak.

- **V115: Predicate-Guided CEGAR** (67/67 tests pass)
  - Composes V114 (recursive predicate discovery) + V110 (abstract reachability tree/CEGAR)
  - Pre-seeded ART: V114 discovers predicates before ART construction via 6 strategies
    (templates, intervals, conditions, assertions, inductive learning, interpolation)
  - Score-guided selection: V114's scoring prioritizes high-value predicates
  - Guided refinement: when interpolation fails, V114 generates fallback candidates
  - Adaptive budget: configurable predicate count based on program complexity
  - Incremental verify: gradually increases predicate budget until convergence
  - Strategy comparison: side-by-side V110 vs V115 performance
  - APIs: guided_verify(), standard_verify(), compare_strategies(), check_assertion(),
    get_discovered_predicates(), verify_with_budget(), incremental_verify(), guided_summary()
  - Bug fix: V110 CFG.nodes is a list (not dict), CFGNode has .type (not .ntype)
  - 115-session zero-bug streak.

- **V116: Quantified Horn Clauses** (89/89 tests pass)
  - Extends V111 with Forall/Exists quantifiers over clause constraints
  - Array theory: SelectTerm, StoreTerm, ConstArrayTerm custom term types
  - Three instantiation strategies: term-based, E-matching, model-based (MBQI)
  - Array axiom engine (read-over-write-same/diff), array property constructors
  - Quantified CHC solver: eliminates quantifiers via instantiation -> V111
  - Quantified validity checking, system analysis, strategy comparison
  - Key lessons: C037 App.__eq__ overloaded (use structural _term_eq), Var/App require sort
  - 116-session zero-bug streak.

- **V117: Widening Strategy Framework** (85/85 tests pass)
  - Composes V103 (widening policy synthesis) + V108 (domain composition framework)
  - Domain-aware adaptive widening: strategy adapts based on composed domain structure
  - 4 widening phases: DELAY -> THRESHOLD -> GRADUATED -> STANDARD
  - Per-component widening in ReducedProductDomain: each component gets its own config
  - Cross-domain reduction between widening iterations for precision
  - Narrowing phase after fixpoint convergence
  - Auto policy synthesis from loop structure (counter detection, threshold extraction)
  - Strategy comparison API: standard vs adaptive vs delayed-threshold
  - APIs: adaptive_analyze(), adaptive_analyze_interval(), adaptive_analyze_composed(),
    standard_analyze(), compare_strategies(), get_adaptive_policies(), get_loop_analysis(),
    validate_adaptive_policy(), widening_summary()
  - Boundary fixes: V020 top()/bot() are instance methods; C10 IfStmt.then_body is Block
    (has .stmts); IntervalDomain uses .lo/.hi; .contains(0) for div-zero check
  - 117-session zero-bug streak.

- **V119: BDD-Based Predicate Abstraction** (90/90 tests pass)
  - Composes V021 (BDD) + V110 (ART/CEGAR) + C037 (SMT) + C010 (parser)
  - BDD-based abstract transition encoding: Cartesian abstraction with assume guards
  - BDD image computation replaces per-predicate SMT queries for abstract post
  - CEGAR loop with backward WP refinement for predicate discovery
  - 5 APIs: bdd_verify(), check_assertion(), bdd_vs_smt_comparison(),
    get_transition_bdds(), bdd_summary()
  - Composition boundary fixes: C037 SMTResult is enum (not string), ASSERT edges
    need guard encoding, assume transitions need feasibility guards
  - 119-session zero-bug streak.

- **V120: Array Domain Abstract Interpretation** (119/119 tests pass)
  - Self-contained array abstract interpreter with per-element interval tracking
  - ArrayAbstractValue: per-element intervals + smash + interval length
  - Strong updates (concrete index) + weak updates (abstract index)
  - Composes V020 IntervalDomain for scalar and element values
  - Out-of-bounds, div-by-zero, assertion checking with dead branch elimination
  - Array property inference: sortedness, boundedness, constant, initialized
  - APIs: array_analyze(), check_bounds(), check_assertions(), get_array_info(),
    get_variable_range(), infer_properties(), compare_analyses(), array_summary()
  - 120-session zero-bug streak.

- **V121: Fixpoint Acceleration** (91/91 tests pass)
  - Accelerates polyhedral abstract interpretation fixpoint convergence
  - Composes V105 (polyhedral domain) + C010 (parser)
  - Staged polyhedral widening: DELAY -> THRESHOLD -> EXTRAPOLATE -> STANDARD
  - Linear recurrence detection: detects x' = x + c patterns, computes limit directly
  - Constraint history extrapolation: tracks bound evolution, extrapolates trends
  - Polyhedral threshold widening: snaps bounds to program-derived thresholds
  - Post-fixpoint narrowing: recovers precision lost during widening
  - Full C10 interpreter: assignments, if-else, while, functions, arithmetic
  - Condition refinement with dead branch elimination (uses is_bot() not _is_bot)
  - Comparison API: side-by-side standard V105 vs accelerated analysis
  - APIs: accelerated_analyze(), standard_analyze(), compare_analyses(),
    get_variable_range(), get_loop_invariant(), get_acceleration_stats(),
    detect_program_recurrences(), verify_invariant(), acceleration_summary()
  - Bug fixes: PolyhedralDomain.is_bot() must be called (not ._is_bot) after
    adding contradictory constraints; token type for numbers is int not 'NUMBER'
  - 121-session zero-bug streak.

- **V122: Symbolic Predicate Minimization** (62/62 tests pass)
  - Composes V119 (BDD predicate abstraction) + V021 (BDD) + V110 (ART/CEGAR) + C037 + C010
  - Given SAFE verdict from V119, finds minimal predicate subset that still proves safety
  - Three strategies: BDD support analysis, greedy backward elimination, delta debugging (ddmin)
  - Combined strategy: support analysis (free) then greedy on live set
  - SubsetVerifier: re-runs BDD-based ART exploration with fixed predicate subset
  - Predicate classification: ESSENTIAL, REDUNDANT, SUPPORT_DEAD
  - Predicate dependency analysis via BDD variable support tracking
  - Strategy comparison API: side-by-side all strategies
  - APIs: minimize_predicates(), classify_predicates(), compare_minimization_strategies(),
    get_predicate_dependencies(), minimization_summary()
  - Composition boundary fixes: art module name, mgr attribute, BDDPredicateManager() no args
  - 122-session zero-bug streak.

- **V123: Array Bounds Verification** (68/68 tests pass)
  - Composes V120 (Array Domain AI) + C037 (SMT solver)
  - SMT-verified proof obligations for every array access (lower + upper bounds)
  - Pipeline: abstract interpretation -> access extraction -> SMT bounds checking
  - BoundsTrackingInterpreter: extends V120 to record abstract state at each access
  - SMTEncoder: encodes interval constraints, checks bounds via C037
  - Proof certificates: serializable, independently re-verifiable
  - Three proof tiers: AI_SAFE (abstract alone), SAFE (SMT), UNSAFE (counterexample)
  - APIs: verify_bounds(), find_unsafe_accesses(), certify_bounds(), check_certificate(),
    compare_ai_vs_smt(), bounds_summary(), verify_with_context(), check_access_safe()
  - Bug fixes: _interpret_array_write override (not _exec_stmt), s.Int() for model,
    context joining for loop dedup
  - 123-session zero-bug streak.

- **V124: Polyhedral Widening with Landmarks** (81/81 tests pass)
  - Composes V121 (fixpoint acceleration) + V105 (polyhedral domain) + C010 (parser)
  - Per-loop landmark analysis: each loop gets a structural LoopProfile
  - Landmarks extracted from: condition bounds, increments, init values, branch thresholds, nested loop bounds
  - Per-variable widening policy: 'accelerate' (recurrence+bound), 'threshold' (landmark values), 'standard'
  - All policies applied in the SAME fixpoint iteration -- each variable gets optimal treatment
  - Landmark-guided narrowing: post-fixpoint tightening using landmark values
  - Nested loop threshold propagation to outer loop profiles
  - Comparison API: landmark vs standard V105 vs V121 accelerated
  - APIs: landmark_analyze(), compare_widening_strategies(), get_variable_range(),
    get_loop_profile(), get_loop_invariant(), get_landmark_stats(), landmark_summary()
  - Bug fix: C10 Block wrapper must be flattened for V121 detect_recurrences
  - 124-session zero-bug streak.

- **V125: Predicate-Minimized CEGAR** (61/61 tests pass)
  - Composes V122 (predicate minimization) + V119 (BDD predicate abstraction CEGAR) + V110 + V021 + C037 + C010
  - Three minimization modes: post-hoc, online (periodic), eager (every iteration)
  - Post-hoc: standard CEGAR then greedy backward elimination on proof predicates
  - Online/eager: BDD support analysis prunes dead predicates during CEGAR iterations
  - IncrementalMinCEGAR: caches minimal predicates across program versions
  - PredicateQuality: essential vs redundant analysis
  - APIs: minimized_cegar_verify(), check_with_minimal_proof(), compare_minimization_modes(),
    get_minimal_proof_predicates(), verify_with_budget(), analyze_predicate_quality(),
    minimized_cegar_summary(), IncrementalMinCEGAR class
  - Bug fixes: CFGNode.type (not .node_type), BDDNode.lo/.hi (not .low/.high),
    online prune must protect newly-added predicates (not yet in transition BDDs)
  - 125-session zero-bug streak.

- **V126: Array Bounds Certificates** (56/56 tests pass)
  - Composes V123 (array bounds verification) + V044 (proof certificates) + C037 (SMT)
  - Pipeline: V123 verifies accesses -> encode as V044 ProofObligation with SMT-LIB2 -> ArrayBoundsCertificate
  - Independent checking: AI-safe (arithmetic on abstract bounds) + SMT (re-run query)
  - JSON serialization round-trip, certificate composition, V044 bridge
  - APIs: certify_array_bounds(), certify_and_check(), check_array_certificate(),
    save_array_certificate(), load_array_certificate(), combine_array_certificates(),
    certify_with_context(), compare_certification_strength(), certificate_summary(),
    to_v044_certificate(), from_v044_certificate()
  - Note: V123 has if-else path dedup bug (dict vs set); avoided by sequential patterns
  - 126-session zero-bug streak.

- **V127: Landmark-Guided k-Induction** (37/37 tests pass)
  - Composes V124 (landmark widening) + V016 (auto k-induction) + V015 + C037 + C010
  - 4-phase pipeline: plain k-ind -> landmark-strengthened -> V016 auto fallback -> combined
  - Candidate extraction from: landmarks, recurrences, thresholds, polyhedral bounds
  - All candidates validated as inductive invariants via SMT
  - APIs: landmark_k_induction(), verify_loop_landmark(), verify_loop_landmark_with_config(),
    get_landmark_candidates(), compare_strategies(), landmark_k_summary()
  - Composition fixes: RecurrenceInfo.var (not .variable), PolyhedralDomain.get_interval() (not .get_bounds()),
    TransitionSystem.prime("x") (not .var("x'"))
  - 126-session zero-bug streak (composition boundary only).

- **V128: Certified Termination** (38/38 tests pass)
  - Composes V025 (termination analysis) + V044 (proof certificates)
  - Per-loop and whole-program termination certificates
  - Ranking function proof obligations: bounded + decreasing
  - Independent checking via V025 re-verification with full loop context
  - JSON serialization, V044 bridge, comparison API
  - APIs: certify_loop_termination(), certify_program_termination(), certify_and_check(),
    check_termination_certificate(), to_v044_certificate(), compare_with_uncertified()
  - 126-session zero-bug streak.

- **V129: Polyhedral k-Induction** (44/44 tests pass)
  - Composes V105 (polyhedral domain) + V015 (k-induction) + V016 (auto k-induction)
  - InvariantCapturingInterpreter captures loop fixpoints (not post-loop state)
  - Extracts interval + relational constraints as invariant candidates
  - LinearConstraint -> SMT App conversion, inductive validation, 4-phase pipeline
  - APIs: verify_loop_polyhedral(), get_polyhedral_candidates(), compare_strategies()
  - 127-session zero-bug streak.

- **V130: Certified Effect Analysis** (41/41 tests pass)
  - Composes V040 (effect systems) + V044 (proof certificates)
  - Certifies: soundness, purity, completeness, handler coverage
  - EffectCertificate with ProofObligation obligations, JSON serialization
  - V044 bridge, independent checking, comparison API
  - APIs: certify_effect_soundness(), certify_effect_purity(), certify_full_effects()
  - 127-session zero-bug streak.

- **V131: Polyhedral-Guided Symbolic Execution** (33/33 tests pass)
  - Composes V105 (polyhedral domain) + C038 (symbolic execution)
  - BranchCapturingInterpreter records feasibility at branch points
  - PolyGuidedExecutor overrides _check_feasible for polyhedral pruning before SMT
  - APIs: poly_guided_execute(), compare_guided_vs_plain(), compare_all_strategies()
  - 127-session zero-bug streak.

## Next Priorities (Session 178+)

1. ~~**V129: Polyhedral k-Induction**~~ DONE (44 tests, Session 178)
2. ~~**V130: Certified Effect Analysis**~~ DONE (41 tests, Session 178)
3. **V131: Polyhedral-Guided Symbolic Execution** DONE (33 tests, Session 178)

## Next Priorities (Session 179+)

1. **V132: Certified Polyhedral Analysis** -- compose V105 (polyhedral) + V044 (certificates) for certified constraint proofs
2. **V133: Effect-Aware Symbolic Execution** -- compose V040 (effects) + C038 (symex) to prune paths based on effect types
3. Continue reactive synthesis / game theory line

### Session 178 Lessons (V129-V131)
- PolyhedralInterpreter.analyze() returns POST-loop env (after exit condition refinement).
  For loop invariants, must capture the widened fixpoint BEFORE exit refinement.
  Solution: InvariantCapturingInterpreter subclass that overrides _interpret_while.
- C038 symbolic execution uses `let x = 0;` + `symbolic_inputs={'x': 'int'}` pattern,
  NOT function calls. Function calls are not properly supported for symbolic inputs.
- C10 parser requires `print(x)` not `print x` (LPAREN expected after print keyword).

- **V132: Certified Polyhedral Analysis** (70/70 tests pass)
  - Composes V105 (polyhedral domain) + V044 (proof certificates)
  - Certificate types: bounds, relational, feasibility, property, full
  - PolyhedralCertificate with ProofObligation obligations, SMT-LIB2 formulas
  - Independent checking, V044 bridge, JSON serialization, comparison API
  - APIs: certify_polyhedral_bounds(), certify_polyhedral_relational(),
    certify_polyhedral_feasibility(), certify_polyhedral_properties(),
    certify_full_polyhedral(), certify_and_check(), to_v044_certificate(),
    compare_certified_vs_uncertified(), polyhedral_certificate_summary()
  - 128-session zero-bug streak.

- **V133: Effect-Aware Symbolic Execution** (47/47 tests pass)
  - Composes V040 (effect systems) + C038 (symbolic execution)
  - Effect pre-analysis (O(n)) -> symbolic execution -> path annotation
  - Auto-suggest symbolic inputs from state-effect variables
  - Path annotations: PURE/STATE/IO/EXN/DIV tags per path
  - APIs: effect_aware_execute(), analyze_effects(), find_effectful_paths(),
    find_pure_paths(), find_io_paths(), find_exception_paths(),
    get_effect_guidance(), suggest_symbolic_inputs(),
    compare_aware_vs_plain(), effect_aware_summary()
  - 128-session zero-bug streak.

## Next Priorities (Session 180+)

1. **V134: Certified Equivalence Checking** -- compose V006 (equivalence) + V044 (certificates) for certified program equivalence proofs
2. **V135: Effect-Typed Program Synthesis** -- compose V040 (effects) + V097/C097 (synthesis) to synthesize programs with effect constraints
3. Continue reactive synthesis / game theory line
4. Consider deeper certified verification stack (certified k-induction, certified PDR)

### Session 179 Lessons (V132-V133)
- PolyhedralDomain.is_bot() only checks unary constraint contradictions and known-value
  substitution. Multi-variable relational constraints (like -x + y == 3) are NOT checked.
  Solution: _is_infeasible() helper using get_interval() (Fourier-Motzkin projection)
  which properly handles relational constraints.
- Integer arithmetic negation: negation of x <= B is x >= B+1 (i.e. -x <= -B-1),
  NOT x >= B. The -1 offset is critical for integer domains.
- CertStatus.value is lowercase ("valid" not "VALID") in V044.
- V105 directory is V105_polyhedral_domain (not V105_polyhedral_abstract_domain).
  Module name: polyhedral_domain (not polyhedral_abstract_domain).
- C038 path from A2/work needs 3 levels up: ../../../challenges/C038_symbolic_execution/
  (A2/work/VXXX -> A2/work -> A2 -> AgentZero -> challenges).

- **V134: Certified Equivalence Checking** (63/63 tests pass)
  - Composes V006 (equivalence checking) + V044 (proof certificates)
  - Machine-checkable certificates for program equivalence
  - Each path pair -> proof obligation (inequiv query UNSAT)
  - Independent checking via SMT-LIB2 re-parsing
  - JSON serialization, V044 bridge, 4 certification modes
  - APIs: certify_function_equivalence(), certify_program_equivalence(),
    certify_regression(), certify_partial_equivalence(),
    certify_and_check(), check_equiv_certificate(), to_v044_certificate(),
    compare_certified_vs_uncertified(), equiv_certificate_summary()
  - 129-session zero-bug streak.

- **V135: Effect-Typed Program Synthesis** (72/72 tests pass)
  - Composes V040 (effect systems) + C097 (program synthesis)
  - Effect-aware component filtering + post-synthesis verification
  - 5 EffectSpec presets, DSL expression effect inference
  - APIs: synthesize_pure(), synthesize_safe(), synthesize_total(),
    synthesize_with_effects(), verify_synthesized_effects(),
    compare_with_unrestricted(), effect_synthesis_summary()
  - 129-session zero-bug streak.

- **V136: Certified k-Induction** (49/49 tests pass)
  - Composes V015 (k-induction) + V044 (proof certificates)
  - Machine-checkable certificates: base case + inductive step + strengthening obligations
  - SMT-LIB2 scripts for independent verification (parse, re-check UNSAT)
  - Source-level: certify_loop(), certify_loop_with_invariants()
  - JSON round-trip, V044 bridge, comparison API
  - Key bug: C037 sort constants are 'Bool'/'Int' (capital), not 'bool'/'int'
  - Key bug: SMTResult is an enum, compare with SMTResult.UNSAT not 'unsat'
  - 130-session zero-bug streak.

- **V137: Certified PDR** (37/37 tests pass)
  - Composes V002 (PDR/IC3) + V044 (proof certificates) + V136 (k-induction)
  - Wraps V044's generate_pdr_certificate() with richer PDRCertificate
  - Combined strategy: certify_combined() tries k-induction first, then PDR
  - Source-level: certify_pdr_loop()
  - Comparison: compare_pdr_vs_kind(), compare_certified_vs_uncertified()
  - JSON round-trip, V044 bridge
  - Clean first-pass: 37/37, zero bugs. 130-session zero-bug streak.

- **V138: Effect-Aware Verification** (50/50 tests pass)
  - Composes V040 (effect systems) + V004 (VCGen) + C010 + C037
  - Effect inference drives VC generation: effects tell us WHAT to verify
  - 5 VC categories: division safety (SMT), frame conditions (structural),
    purity (IO + exception), IO isolation, termination (ranking functions)
  - EffectVCGenerator: generates effect-specific VCs from AST analysis
  - EffectAwareVerifier: full pipeline (infer -> generate VCs -> check via SMT)
  - APIs: verify_effects(), verify_pure_function(), verify_state_function(),
    verify_exception_free(), infer_and_verify(), compare_declared_vs_inferred(),
    effect_verification_summary()
  - Key design: always generate division safety VCs unless Exn explicitly declared
  - 52-session zero-bug streak.

- **V139: Certified Regression Verification** (35/35 tests pass)
  - Composes V134 (certified equivalence) + V136 (certified k-induction)
  - Two-phase strategy:
    1. Try certified equivalence (fast: old == new implies property preserved)
    2. Fall back to certified k-induction on new version
  - RegressionCertificate: JSON serializable, independently checkable
  - APIs: verify_regression(), verify_function_regression(),
    verify_program_regression(), check_regression_certificate(),
    save/load_regression_certificate(), compare_equiv_vs_kind()
  - Clean first pass, zero bugs. 52-session zero-bug streak.

## Completed (Session 186): V140-V141

- **V140: Effect-Aware Regression Verification** (42/42 tests pass)
  - Composes V138 (effect-aware verification) + V139 (certified regression)
  - Detects effect regressions: code changes introducing new effects
  - Three-phase: effect inference + effect verification + certified regression
  - EffectRegressionVerdict: SAFE, EFFECT_REGRESSION, PROPERTY_FAILURE, UNSAFE, UNKNOWN
  - APIs: verify_effect_regression(), verify_function_effect_regression(),
    check_effect_purity_preserved(), compare_effect_regression_methods()
  - 53-session zero-bug streak.

- **V141: Certified AI-Strengthened k-Induction** (40/40 tests pass)
  - Composes V046 (certified abstract interpretation) + V136 (certified k-induction)
  - Uses AI-derived invariants (intervals, signs) to strengthen k-induction proofs
  - Combined certificates: AI soundness + k-induction validity
  - APIs: certify_ai_kind(), certify_ai_kind_basic(), analyze_ai_invariants(),
    compare_basic_vs_ai(), ai_kind_summary()
  - 53-session zero-bug streak.

- **V143: Certified AI-Strengthened PDR** (55/55 tests pass)
  - Composes V046 (certified abstract interpretation) + V137 (certified PDR)
  - AI invariants conjoin with property for strengthened PDR + combined certificates
  - Init-safe invariant filtering: discards post-loop invariants that would cause false counterexamples
  - APIs: certify_ai_pdr(), certify_ai_pdr_basic(), analyze_ai_invariants(),
    compare_basic_vs_ai(), compare_pdr_vs_kind_ai(), ai_pdr_summary()
  - 55-session zero-bug streak.

- **V144: Certified Effect-Aware PDR** (61/61 tests pass)
  - Composes V143 (certified AI-PDR) + V140 (effect regression) + V044 (proof certificates)
  - Unified pipeline: verify loop properties AND effect discipline in one call
  - Phase 1: AI-strengthened PDR (V143), Phase 2: Effect conformance (V140), Phase 3: Certificate combination (V044)
  - EffectPDRVerdict: SAFE/PROPERTY_FAILURE/EFFECT_VIOLATION/UNSAFE/UNKNOWN
  - APIs: certify_effect_pdr(), certify_effect_pdr_basic(), verify_effect_loop(),
    analyze_effects_only(), verify_effect_regression_pdr(),
    compare_effect_vs_plain(), compare_ai_vs_basic_effect_pdr(), effect_pdr_summary()
  - API fixes: V044 ProofObligation(name, description, formula_str, formula_smt, status),
    ProofKind: VCGEN/PDR/COMPOSITE (no SAFETY), ProofCertificate needs claim param
  - V140 module: effect_aware_regression.py (not effect_regression.py)
  - 56-session zero-bug streak.

## Next Priorities (Session 193+)

1. **V148: Certified Dataflow Analysis** -- reaching definitions, live variables, available expressions with proof certificates
2. **V149: Certified Thread Safety Analysis** -- race detection, deadlock detection with certificates
3. Continue certified stack or game theory line
4. Consider ML-focused challenges (neural network verification, abstract interpretation of DNNs)

- **V147: Certified Assume-Guarantee Reasoning** (70/70 tests pass)
  - Thread-modular verification with circular assumption discharge
  - Composes V004 (VCGen/WP) + V044 (proof certificates) + C037 (SMT) + C010 (parser)
  - Three discharge strategies: direct, circular, inductive (ranked)
  - Dependency analysis with Tarjan SCC for cycle detection
  - Non-interference via self-composition, contract refinement checking
  - AGVerdict: SOUND, COMPONENT_FAILURE, DISCHARGE_FAILURE, UNKNOWN
  - APIs: verify_ag(), verify_two_components(), make_ag_system(),
    discharge_direct/circular/inductive(), analyze_dependencies(),
    compare_discharge_strategies(), certify_ag(), ag_summary(),
    batch_verify(), verify_noninterference(), check_contract_refinement()
  - 59-session zero-bug streak.

- **V145: Certified Compositional Verification** (52/52 tests pass)
  - Composes V004 (VCGen/WP) + V044 (proof certificates) + C010 (parser) + C037 (SMT)
  - Modular verification: verify functions independently, compose proofs
  - ModularWPCalculus: at call sites, preconditions = obligations, postconditions = assumptions
  - Incremental re-verification: only changed modules + their callers re-verified
  - Spec refinement checking: weakened precondition + strengthened postcondition
  - Call graph analysis, change impact analysis
  - Compare modular vs monolithic V004 verification
  - CompVerdict: SOUND, MODULE_FAILURE, CALL_FAILURE, UNKNOWN
  - APIs: verify_compositional(), verify_incremental(), check_spec_refinement(),
    analyze_call_graph(), analyze_change_impact(), compare_modular_vs_monolithic(),
    certify_compositional(), compositional_summary()
  - 57-session zero-bug streak.

### Session 192 Lessons (V147)
- C010 exports Parser class, not parse function: use `Parser(tokens).parse()`
- C010 `lex()` function is available, but `parse` must go through Parser class
- Circular AG: guarantees must be jointly consistent (satisfiable). Lock protocol
  with lock==1 AND lock==0 is correctly rejected as inconsistent.
- SBool(True/False) in SMT: C037 Bool vars don't trivially satisfy b==True OR b==False
  tautology. Use Int tautologies for testing (x+1 > x).
- ComponentSpec needs body_stmts parsed to work with _extract_body_transformer.
  Use extract_component() or make_ag_system() which parse automatically.
- Non-interference via self-composition: duplicate high vars, assume low vars equal,
  check low outputs equal. Clean encoding into AG framework.

### Session 190 Lessons (V145)
- C010 CallExpr.callee is always a str (not an AST node)
- C010 uses `lex()` function (not Lexer class), module is `stack_vm` (not `lang`)
- requires/ensures/invariant/assert are parsed as CallExpr -- filter them in call graph analysis
- V004 WP doesn't handle early returns in if-branches correctly (return in then + return after if)
  - Use single-return pattern: `let r = default; if (cond) { r = x; } return r;`
- ModularWP for `let x = f(args)`: WP = postcond[formals/actuals, result/x] => Q

### Session 189 Lessons (V144)
- V044 ProofObligation requires: name, description, formula_str, formula_smt, status (keyword)
- V044 ProofCertificate requires: kind, claim (str), then optional source, obligations, metadata, status
- V044 ProofKind only has VCGEN, PDR, COMPOSITE -- no SAFETY enum value
- V140 module is effect_aware_regression.py (not effect_regression.py as documented in agent summary)
- Accumulator loops (sum=sum+i) cause SMT timeouts in PDR at ANY max_frames setting
  - Use analyze_effects_only() for effect-only tests on accumulator patterns
- 100th verification tool milestone reached (V001-V144 with gaps)

### Session 188 Lessons (V143 + Interface Design)
- V143 reused prior session's implementation -- confirmed 55/55 tests pass
- Certified AI + PDR is expensive: accumulator loop (sum=sum+i) causes SMT timeouts
  even at max_frames=10. Use max_frames=5-10 for test speed, not 50.
- Init-safe invariant filtering is critical: post-loop AI bounds (e.g., i >= 5 from
  while(i<5)) are true after the loop but violate PDR's initial state check
- compare_basic_vs_ai and compare_pdr_vs_kind_ai run PDR 2-3 times -- keep frames low
- V141 module: certified_ai_composition.py in V141_certified_ai_composition/

### Session 186 Lessons (V140-V141)
- V040 module name: effect_systems.py (NOT effect_system.py)
- C039 ai_analyze() returns dict with 'env' key, not AIAnalysisResult object
- AbstractEnv uses .intervals and .signs dicts (NOT .store with AbstractValue objects)
- WhileStmt.body is a Block object -- use .stmts to get the list of statements
- Interval lo/hi are floats -- cast to int for clean invariant expressions when lo==int(lo)
- _extract_ai_invariants works with raw C039 analyze(), not V046's traced_analyze()

### Session 185 Lessons (V138-V139)
- V040 EffectInferrer.infer_program() returns dict[str, FnEffectSig] including "__main__"
- V040 infers Exn(DivByZero) for variable divisors: must generate div-safety VCs regardless
  of inference result (only skip when Exn is explicitly DECLARED by the user)
- V134 certify_function_equivalence param_types must be non-empty dict to trigger execution
- Empty dict `{}` is falsy in Python -- `if param_types:` fails for zero-arg functions
- V136 certify_loop() extracts loop TS from source, works with source-level property strings

### Session 184 Lessons (V136-V137)
- C037 sort constants: BOOL = 'Bool', INT = 'Int' (capital B/I). Using lowercase 'bool'/'int'
  causes _encode_comparison to never match (term.sort == BOOL fails silently).
- SMTResult is an enum class. Compare with SMTResult.UNSAT, SMTResult.SAT, not strings.
- V015 k-induction uses formula-based TransitionSystem (init_formula, trans_formula, prop_formula)
  with primed variables (x' for next state), NOT dict-based init/transition.
- TransitionSystem API: add_int_var(name) -> Var, prime(name) -> Var, set_init/set_trans/set_property(formula)
- V044's generate_pdr_certificate() already works well; V137 wraps it with richer data structure.
- V015 imports _apply_formula_at_step, _apply_trans_at_step, _negate, _step_vars.

### Session 183 Lessons (V134-V135)
- V006 module name: equiv_check.py (NOT equivalence_checking.py)
- V006 path outputs are SymValue objects, need _symval_to_term() conversion to SMT Term
- V006 _collect_vars_from_term returns (name, sort) tuples for proper SMT declaration
- V006 uses _declare_vars_in_solver(solver, constraints, extra_vars) for proper variable registration
- C097 module name: synthesis.py. IOExample(inputs=dict, output=val).
- C097 synthesize() returns SynthesisResult with .success, .program (Expr), .method, .candidates_explored
- C097 Expr DSL: IntConst, BoolConst, VarExpr, UnaryOp, BinOp, IfExpr -- all frozen dataclasses
- Pure DSL expressions: arithmetic +,-,*,max,min and comparisons. Division/modulo add Exn effect.

- **V148: Probabilistic Bisimulation** (53/53 tests pass)
  - Behavioral equivalence for probabilistic systems (labeled Markov chains)
  - Composes V065 (Markov chains) + V067 (labeled MCs) + C037 (SMT solver)
  - Strong probabilistic bisimulation via Larsen-Skou partition refinement
  - Bisimulation quotient (minimization): collapses bisimilar states
  - Simulation preorder: coinductive greatest simulation relation
  - Bisimulation distance: discounted Kantorovich metric, greedy earth mover
  - Cross-system bisimulation via disjoint union construction
  - Lumping validation, SMT-verified partition certificates
  - Key lesson: V065 module name is markov_chain.py (singular), not markov_chains
  - Key lesson: V065 directory is V065_markov_chain_analysis (not V065_markov_chains)
  - Zero implementation bugs. 60-session zero-bug streak.

### Session 193 Lessons (V148)
- V065 module: markov_chain.py in V065_markov_chain_analysis/
- V067 module: pctl_model_check.py in V067_pctl_model_checking/
- LabeledMC: make_labeled_mc(matrix, labels, state_labels=None)
- MarkovChain.transition is List[List[float]] (dense matrix)
- Partition refinement: round probabilities to avoid floating point splitting
- Kantorovich distance: greedy earth mover is sufficient for small state spaces
- Simulation preorder uses class-based probability matching (not exact coupling LP)

- **V149: MDP Bisimulation** (55/55 tests pass)
  - Extends V148 probabilistic bisimulation to MDPs (nondeterministic + probabilistic)
  - Composes V069 (MDP) + V148 (prob bisim) + V065 (Markov chains) + V067 (labeled MC)
  - Partition refinement: same labels AND SET-matching of action block-probability vectors
  - Action matching is set-based (names irrelevant, only distributions matter)
  - Features:
    - Quotient MDP construction (collapse bisimilar states, dedup distributions)
    - MDP simulation preorder (for every action at t, s has a matching action)
    - Hausdorff-Kantorovich bisimulation distance (greedy earth mover)
    - Cross-system bisimulation via disjoint union
    - Policy-induced bisimulation (reduce to MC via V069 mdp_to_mc, use V148)
    - Reward-aware bisimulation (actions must match on reward too)
    - SMT-verified partition validity (label + action matching checks)
    - MDP vs MC bisimulation comparison (MDP always finer than any policy MC)
  - APIs: compute_mdp_bisimulation(), check_mdp_bisimilar(), mdp_bisimulation_quotient(),
    compute_mdp_simulation(), compute_mdp_bisimulation_distance(),
    check_cross_mdp_bisimulation(), policy_bisimulation(), compare_policy_bisimulations(),
    compute_reward_bisimulation(), verify_mdp_bisimulation_smt(),
    compare_mdp_vs_mc_bisimulation(), analyze_mdp_bisimulation()
  - Key lesson: 2-state systems with same labels always collapse to 1 block
    (all distributions sum to 1.0 in single block). Need 3+ states with
    distinct labels to see action-based partition splitting.
  - Zero implementation bugs. 61-session zero-bug streak.

### Session 194 Lessons (V149)
- V069 MDP: make_mdp(n_states, action_transitions, rewards, state_labels)
- V069 MDP.transition[s][a_idx][t] = P(s,a->t), rows sum to 1.0
- V069 MDP.actions[s] = list of action names at state s
- V069 mdp_to_mc(mdp, Policy) -> MarkovChain (for policy-induced MC)
- V067 make_labeled_mc(matrix, labels, state_labels) -> LabeledMC
- Bisimulation partition refinement: 2 states same labels in single block =>
  all actions have block-prob (1.0,) => indistinguishable. Must have 3+ states
  with different labels to create initial blocks that enable refinement.
- Greedy earth mover (Kantorovich distance): sort state pairs by distance,
  flow mass greedily. Optimal for small state spaces.
- MDP bisim is strictly finer than MC bisim under any policy: MDP requires
  matching ALL actions, MC only sees the single chosen action.

- **V150: Weak Probabilistic Bisimulation** (83/83 tests pass)
  - Abstracts away internal (tau) transitions in probabilistic systems
  - Composes V148 (prob bisim) + V065 (Markov chains) + V067 (labeled MCs)
  - LabeledProbTS: transition system with named actions (including tau)
  - Tau closure: iterative fixpoint for tau* reachability distributions
  - Weak transitions: tau* ; action ; tau* composition
  - Weak bisimulation via partition refinement on weak transitions
  - Branching bisimulation (preserves branching structure, finer than weak)
  - Divergence detection + divergence-sensitive bisimulation
  - Weak bisimulation distance (discounted Kantorovich on weak transitions)
  - Cross-system, quotient, comparison (strong vs branching vs weak)
  - APIs: compute_weak_bisimulation(), check_weakly_bisimilar(),
    compute_branching_bisimulation(), check_branching_bisimilar(),
    weak_bisimulation_quotient(), branching_bisimulation_quotient(),
    check_cross_weak_bisimulation(), compute_weak_simulation(),
    detect_divergence(), compute_divergence_sensitive_bisimulation(),
    compute_weak_bisimulation_distance(), compare_strong_vs_weak(),
    minimize_weak(), minimize_branching(), weak_bisimulation_summary(),
    lmc_to_prob_ts(), prob_ts_to_lmc(), make_labeled_prob_ts()
  - Zero implementation bugs. 62-session zero-bug streak.

- **V151: Probabilistic Process Algebra** (74/74 tests pass)
  - CCS-style process algebra with probabilistic choice
  - Composes V150 (weak probabilistic bisimulation)
  - Process AST: stop, prefix, prob_choice, nd_choice, parallel, restrict, relabel, recursion
  - SOS rules for all operators, CCS synchronization (a/~a -> tau)
  - LTS generation via BFS exploration with state limit
  - Process equivalence via weak bisimulation (initial-state focused)
  - Trace set, deadlock freedom, action set, process summary
  - Parser for text syntax
  - APIs: stop(), prefix(), tau_prefix(), prob_choice(), nd_choice(), parallel(),
    restrict(), relabel(), recvar(), recdef(), generate_lts(),
    check_process_equivalence(), trace_set(), deadlock_free(), action_set(),
    process_summary(), parse_proc()
  - 62-session zero-bug streak.

### Session 195 Lessons (V150 + V151)
- V148 module name: prob_bisimulation.py (not probabilistic_bisimulation.py)
- LTS state labels must reflect BEHAVIOR, not AST structure:
  derive from transitions (has transitions = active, none = deadlock),
  NOT from process kind. relabel(stop(), ...) has kind RELABEL but behaves as deadlock.
- Cross-system bisimulation checks must compare initial states (state 0 of each system),
  not just check if ANY cross-pair exists. A shared deadlock state is trivially cross-bisimilar.
- Weak bisimulation on LabeledProbTS requires precomputing all weak transitions
  (tau* ; a ; tau*) before partition refinement -- can't compute lazily.
- Branching bisimulation: tau transitions that stay entirely within the same block
  are stuttering and should be ignored in the signature.

- **V152: Symbolic Bisimulation** (62/62 tests pass)
  - BDD-based bisimulation for labeled transition systems
  - Composes V021 (BDD model checking)
  - Symbolic partition refinement: preimages via BDD, no explicit enumeration
  - Three modes: strong, weak (tau closure), branching
  - Cross-system bisimulation via disjoint union
  - Quotient/minimization, parametric generators (chain, ring, tree, parallel)
  - Valid states mask for non-power-of-2 state spaces
  - Key lesson: when n_states < 2^n_bits, phantom states get their own blocks
    unless restricted by a valid_states BDD mask in initial partition

### Session 196 Lessons (V152)
- Phantom states: if state encoding uses more bits than needed, unused bit patterns
  create phantom states that get their own partition block. Fix: valid_states BDD
  mask applied in _initial_partition to restrict to actual states.
- BDD rename: to encode "y in B" where B is defined over x-vars, rename x->x' in B
- Weak preimage = tau_closure_backward(pre_a(tau_closure_backward(target)))

- **V153: Game-based Bisimulation** (63/63 tests pass)
  - Bisimulation as a two-player game: Attacker (Spoiler) vs Defender (Duplicator)
  - Composes V076 (parity games) for infinite-duration game solving
  - Game positions: attacker nodes = state pairs, defender nodes = challenge positions
  - Parity encoding: attacker nodes (ODD, priority 0), defender nodes (EVEN, priority 0),
    deadlocks (priority 1 = attacker wins)
  - Three modes: strong bisim game, weak bisim game (tau closure), simulation game
  - Distinguishing play/sequence extraction for non-bisimilar pairs
  - Partition refinement comparison (validates game agrees with standard algorithm)
  - Bug fix: simulation game uses 5-tuple keys (no side param), lookup must match
  - APIs: check_bisimulation_game(), check_weak_bisimulation_game(),
    check_simulation_game(), full_bisimulation_game(), partition_bisimulation(),
    compare_game_vs_partition(), bisimulation_game_summary()

### Session 197 Lessons (V153)
- Parity game reverse_map key arity must match get_or_create: simulation game has
  no `side` parameter (always side 1), so its keys are 5-tuples while bisim game
  keys are 6-tuples. Lookup must use the correct arity.
- Bisimulation games: both-deadlocked states (no enabled actions) are bisimilar
  if labels match. Self-loop with priority 0 = defender wins.
- Attacker-owned nodes get Player.ODD in parity game convention (win1 = attacker wins).

- **V154: Bisimulation for Stochastic Games** (45/45 tests pass)
  - Extends V149 MDP bisimulation to V070 two-player stochastic games
  - Owner-aware partition refinement: same owner + labels + action distribution sets
  - Composes V070 (stochastic games) + V149 (MDP bisimulation) + V148 (prob bisimulation) + V065 + C037
  - Features: quotient game, simulation preorder, Hausdorff-Kantorovich distance,
    cross-system bisimulation, strategy-induced bisimulation, reward-aware bisimulation,
    SMT partition verification, game vs MDP comparison, full analysis + summary
  - Key lesson: set-based bisimulation treats duplicate actions (same distribution,
    different names) as equivalent. This is mathematically correct.
  - APIs: compute_game_bisimulation(), check_game_bisimilar(), game_bisimulation_quotient(),
    compute_game_simulation(), check_game_simulates(), compute_game_bisimulation_distance(),
    check_cross_game_bisimulation(), check_cross_game_bisimilar_states(),
    strategy_bisimulation(), compare_strategy_bisimulations(), compute_reward_bisimulation(),
    verify_game_bisimulation_smt(), minimize_game(), compare_game_vs_mdp_bisimulation(),
    analyze_game_bisimulation(), game_bisimulation_summary()
  - 65-session zero-bug streak.

### Session 198 Lessons (V154)
- V070 directory is V070_stochastic_games (not V070_stochastic_game_verification)
- V070 make_game expects: owners=Dict[int, Player], action_transitions=Dict[int, Dict[str, List[float]]],
  rewards=Dict[int, Dict[str, float]]. Owners is dict not list. Transitions are dense (full prob list), not sparse tuples.
- Duplicate actions with identical block-probability distributions collapse to one signature
  in set-based bisimulation. {(0,0,1.0), (0,0,1.0)} = {(0,0,1.0)}.

- **V155: Process Algebra Verification** (71/71 tests pass)
  - Composes V151 (process algebra) + V150 (weak prob bisimulation) + V067 (PCTL) + V065 (MC)
  - Pipeline: Process term -> LTS (V151) -> Markov chain -> PCTL check (V067)
  - Also uses V150 for behavioral equivalence, distance, simulation, minimization
  - Features: PCTL verification, weak/strong/branching equiv, algebraic law checking,
    trace analysis, deadlock/divergence analysis, compositional analysis, property
    preservation, behavioral distance, equivalence hierarchy, minimization, refinement
  - Key design: uniform resolution of ND choices for MC conversion; state labels
    enriched with action capabilities (can_a, deadlock, has_tau)
  - Lesson: V067 PCTL path formulas (X, U) must be inside probability operators.
    Use P<=0[F deadlock] not AG(NOT deadlock).
  - 66-session zero-bug streak.

### Session 199 Lessons (V155)
- V150 module name is weak_probabilistic_bisimulation (not weak_prob_bisimulation)
- LabeledMC and make_labeled_mc are in V067 (pctl_model_check), not V065 (markov_chain)
- V151 check_strong_equivalence uses V150 probabilistic bisimulation -- tau.a.0 and a.0
  are strongly bisimilar in the probabilistic sense (tau is abstracted differently than
  in CCS-style strong bisim)
- V151 nd_choice may merge STOP continuations, so a.0 + b.0 may have fewer states
  than expected
- V150 weak bisimulation: P+P is NOT always ~w P (nd_choice creates different LTS
  structure; trace equivalence holds but not bisimulation)

- **V156: Parity Games** (84/84 tests pass)
  - Infinite-duration two-player games with parity winning conditions
  - Three algorithms: Zielonka (recursive), Small Progress Measures (Jurdzinski), iterative McNaughton-Zielonka
  - Attractor computation, strategy extraction, solution verification
  - Game construction helpers: safety, reachability, Buchi, co-Buchi
  - All-algorithm comparison and statistics
  - 67-session zero-bug streak.

### Session 200 Lessons (V156)
- Max-parity vs min-parity is critical for SPM: max-parity truncates odd priorities BELOW p (not above).
  Jurdzinski's paper uses min-parity convention; when implementing for max-parity, reverse truncation direction.
- Dead-end vertices in parity games: the owner of a dead-end vertex loses (can't move).
  Must pre-process dead ends via attractor computation before main recursion.
- Priority Promotion closed-region check: a region at priority p that contains vertices
  with higher priorities is NOT necessarily a dominion for player(p). The actual winner
  depends on the max priority in the region, not the region label.

- **V157: Mu-Calculus Model Checking** (115/115 tests pass)
  - Modal mu-calculus: propositional logic + Diamond/Box modalities + mu/nu fixpoints
  - Composes V156 (parity games -- Zielonka solver)
  - Two model checking methods:
    1. Direct fixpoint iteration (Emerson-Lei): evaluate formulas as state sets
    2. Parity game reduction: subformula x state vertices, solve with V156
  - Formula AST: Prop, Var, TT, FF, Not, And, Or, Diamond(action), Box(action), Mu, Nu
  - LTS with labeled transitions and atomic propositions
  - Positive Normal Form conversion (De Morgan + modal/fixpoint duality)
  - Alternation depth and fixpoint nesting computation
  - CTL encoding: EF, AG, AF, EG, EU, AU, EX, AX as mu-calculus formulas
  - Formula parser: `mu X. (p | <>X)` syntax
  - Compare methods API: cross-validates direct and game-based results
  - Tested: traffic light, mutex protocol, branching, deadlocks, edge cases
  - Key lesson: terminal nodes in parity game reduction need self-loops with
    priority encoding truth value (even=true, odd=false), not dead-end semantics
  - 68-session zero-bug streak.

### Session 201 Lessons (V157)
- Parity game reduction terminal nodes: use self-loops with priority encoding the
  boolean result (even priority = true/Even wins, odd = false/Odd wins). Owner is
  irrelevant at self-loops (only one edge). Do NOT use dead-end "owner loses"
  semantics -- it doesn't know whether the node should be true or false.
- Diamond with no successors = false (self-loop, odd priority).
  Box with no successors = vacuously true (self-loop, even priority).
- Duplicate function definitions in Python: last definition wins. Watch for
  accidentally leaving an earlier draft when the real implementation is below it.
- Mu-calculus parser: fixpoint operators bind weakly (like lambda) -- `nu X. body`
  captures everything after the dot as body.

- **V158: Symbolic Mu-Calculus** (81/81 tests pass)
  - BDD-based evaluation of mu-calculus formulas (composes V021 BDD + V157 mu-calculus)
  - SymbolicLTS: LTS encoded with BDDs (per-action trans BDDs, proposition BDDs)
  - SymbolicMuChecker: symbolic evaluation of all formula types
  - Diamond via preimage (existential quantification), Box via dual
  - Mu (lfp from FALSE), Nu (gfp from TRUE/valid_states)
  - Conversion: V157 LTS -> SymbolicLTS, V021 BooleanTS -> SymbolicLTS
  - Parametric constructors: make_counter_lts(), make_mutex_lts()
  - Comparison API: symbolic vs explicit cross-validation
  - APIs: symbolic_check(), symbolic_check_lts(), check_state_symbolic(),
    compare_with_explicit(), batch_symbolic_check(), symbolic_reachable(),
    check_safety_symbolic(), full_analysis(), symbolic_mu_summary()
  - Key fix: BDD all_sat don't-care expansion for proper state enumeration
  - 69-session zero-bug streak.

### Session 212 Lessons (V158)
- BDD all_sat returns partial assignments where some variables are don't-care.
  Must expand 2^k combinations of k free bits to enumerate all concrete states.
- V021 BooleanTS next_indices uses UNPRIMED keys (e.g., "x" -> idx for x').
  SymbolicLTS uses PRIMED keys (e.g., "x'" -> idx). Convert at boundary.
- make_symbolic_lts next_dict should use unprimed keys for user API convenience
  (users write n["x"], not n["x'"]).
- Module name is bdd_model_checker.py (not bdd_model_check.py).

- **V159: Symbolic Parity Games** (59/59 tests pass)
  - Composes V021 (BDD) + V156 (Parity Games) for BDD-based parity game solving
  - SymbolicParityGame: vertices as bit-vectors, edges/owner/priority as BDDs
  - Symbolic attractor: BDD fixpoint via preimage
  - Symbolic Zielonka: recursive algorithm using BDD set operations
  - Explicit <-> Symbolic conversion with roundtrip verification
  - Parametric constructors: chain, ladder, safety, reachability, Buchi games
  - Comparison API: explicit V156 vs symbolic V159 (cross-validated)
  - Strategy extraction and verification via V156 verifier
  - 70-session zero-bug streak.

### Session 213 Lessons (V159)
- BDD named_var() must be called BEFORE var_index() -- var_index looks up
  _name_to_idx which is only populated by named_var.
- Symbolic attractor: restrict edges to subgame vertices (both curr and next)
  before computing preimage, otherwise vertices outside subgame can be attracted.
- Dead-end handling in symbolic Zielonka: check has_succ = exists_multi(next_idxs, edges_sub),
  dead_ends = verts AND NOT(has_succ). Dead Even -> Odd wins, Dead Odd -> Even wins.
- State extraction from BDD: must expand don't-care bits in all_sat assignments
  (same pattern as V158).

- **V160: Energy Games** (74/74 tests pass)
  - Two-player infinite-duration games with quantitative energy objectives
  - Composes V156 (Parity Games) for combined energy-parity conditions
  - Energy game solver: value iteration (Bellman-Ford style), O(n * m * W)
  - Energy-parity games: iterative parity-energy intersection refinement
  - Mean-payoff games: value iteration + energy reduction for threshold problems
  - Fixed initial energy analysis, simulation, strategy verification
  - Construction helpers: chain, charging, choice games
  - Comparison API: energy-only vs parity-only vs combined
  - APIs: solve_energy(), solve_energy_parity(), solve_mean_payoff(),
    solve_fixed_energy(), mean_payoff_threshold(), simulate_play(),
    verify_energy_strategy(), compare_energy_vs_parity(), energy_game_statistics()
  - 71-session zero-bug streak.

### Session 214 Lessons (V160)
- V156 function name is `zielonka`, not `solve_zielonka`. Always check actual exports.
- verify_energy_strategy: capping energy at n*W for visited-state deduplication causes
  false positives when the true energy exceeds the cap. Use cycle detection instead:
  if same vertex is revisited with strictly lower energy, the cycle depletes.
- Energy-parity solver: pure Zielonka recursion on priorities ignores the energy
  condition entirely. Must intersect: solve parity -> check energy on winning
  subgame -> remove failures -> re-solve. Iterate until stable.

- **V161: Mean-Payoff Parity Games** (74/74 tests pass)
  - Combined mean-payoff + parity objectives (Even wins iff parity AND MP >= threshold)
  - Composes V156 (Parity Games) + V160 (Energy Games)
  - Iterative refinement: Zielonka parity + strategy-consistent energy check
  - Key insight: must check mean-payoff UNDER Even's parity strategy, not independently
  - Found V160 solve_energy_parity bug: ignores strategy conflict between objectives
  - Optimal value computation via binary search
  - Strategy verification, simulation, decomposition analysis
  - APIs: solve_mpp(), solve_mpp_threshold(), compute_mpp_values(),
    verify_mpp_strategy(), simulate_play(), decompose_mpp()
  - 72-session zero-bug streak.

### Session 215 Lessons (V161)
- V156 Solution uses .win_even/.win_odd (not .win0/.win1)
- V160 solve_energy_parity has a bug: checks energy freely (Even picks best path)
  without constraining to parity strategy. Fix: use _solve_mp_under_strategy which
  restricts Even's edges to strategy graph.
- Mean-payoff parity requires strategy consistency: can't check parity and MP
  independently because they may require conflicting moves at choice vertices.
- Iterative refinement: solve parity -> check MP under parity strategy -> remove
  failures + Odd attractor -> re-solve parity. Converges in O(n) iterations.

- **V162: Symbolic Energy Games** (58/58 tests pass)
  - BDD-based encoding of energy games (composes V021 + V160)
  - Symbolic operations: successors, predecessors, attractor (fixpoint on BDD vertex sets)
  - Symbolic value iteration for minimum initial energy computation
  - Symbolic reachability (BDD-based forward BFS)
  - Symbolic safety checking (attractor-based Even winning region)
  - Energy-parity solving (delegates to V160 + BDD encoding)
  - Comparison APIs: symbolic vs explicit (V160) side-by-side
  - Construction helpers: chain, diamond, grid games
  - APIs: encode_energy_game(), solve_symbolic_energy(), solve_symbolic_energy_parity(),
    symbolic_attractor(), symbolic_reachability(), symbolic_safety_check(),
    compare_with_explicit(), compare_energy_parity(), symbolic_energy_statistics()
  - 73-session zero-bug streak.

### Session 216 Lessons (V162)
- INF propagation in value iteration: when energy[t] >= INF, the needed energy
  for predecessor must also be INF. Cannot subtract weight from INF.
- V160 solve_energy_parity has known bug: checks energy without constraining
  to parity strategy. Tests must match actual V160 behavior, not theoretical.
- BDD encoding: vertices -> contiguous 0..n-1 indices -> little-endian bit tuples.
  curr_indices for current state, next_indices for successor. rename() for remap.
- Symbolic attractor: player vertices attracted if SOME successor in target;
  opponent vertices attracted if ALL successors in target (with has_successor guard).

- **V163: Symbolic Mean-Payoff Games** (56/56 tests pass)
  - BDD-based symbolic solving of mean-payoff parity games
  - Composes V021 (BDD) + V161 (Mean-Payoff Parity) + V160 (Energy Games)
  - Symbolic Zielonka parity solver: BDD fixpoint iteration with subgame restriction
  - Symbolic attractor: restricted edges to subgame, dead-end handling
  - Mean-payoff via energy game reduction (explicit numeric, symbolic sets)
  - Iterative refinement: symbolic parity + energy check + symbolic Odd attractor
  - Symbolic value computation via binary search + threshold solving
  - Symbolic decomposition: parity-only vs MP-only vs combined (all BDD-based)
  - Comparison APIs: all agree with explicit V161 solver
  - APIs: solve_symbolic_mpp(), compute_symbolic_mpp_values(),
    symbolic_decompose_mpp(), symbolic_attractor(), symbolic_reachability(),
    symbolic_safety_check(), compare_with_explicit(), compare_values(),
    compare_decompositions(), symbolic_mpp_statistics()
  - 74-session zero-bug streak.

### Session 217 Lessons (V163)
- Symbolic Zielonka requires edge restriction to subgame: both curr AND next
  vertices must be in subgame BDD when building edges_sub.
- Dead-end handling in symbolic attractor: vertices with no successors in subgame
  need special treatment (dead Even -> attracted by Even's opponent).
- Energy values are fundamentally numeric, not Boolean. BDD representation useful
  for sets (winning regions, attractors) but explicit iteration for values.
- Strategy extraction after symbolic parity: explicit loop over Even vertices,
  prefer successors in winning region.

- **V164: Stochastic Energy Games** (67/67 tests pass)
  - 2.5-player energy games: EVEN, ODD, RANDOM vertex types
  - Composes V160 (Energy Games) + V156 (Parity Games)
  - Almost-sure solving: for RANDOM vertices, must survive ALL positive-prob outcomes
  - Positive-probability solving: RANDOM only needs ONE good path (existential)
  - Expected energy computation under optimal play
  - Stochastic energy-parity games via iterative refinement
  - Strategy verification, simulation, comparison with deterministic V160
  - Construction helpers: chain, diamond, gambling, random walk games
  - Key insight: random vertices in cycles cause almost-sure energy divergence
    (bad outcome repeats infinitely often by Borel-Cantelli). Only acyclic random
    or strategically avoidable random leads to finite almost-sure energy.
  - APIs: solve_stochastic_energy(), solve_stochastic_energy_parity(),
    simulate_play(), verify_strategy(), compare_with_deterministic(),
    stochastic_energy_statistics()
  - 75-session zero-bug streak.

### Session 218 Lessons (V164)
- Almost-sure energy with random in cycles: the max-over-outcomes semantics is
  correct. Each visit to a random vertex requires surviving the worst outcome.
  In a cycle, this repeats, causing energy to diverge. This is NOT a bug --
  it reflects the mathematical reality that bad outcomes happen infinitely often a.s.
- Positive-probability winning is strictly weaker: RANDOM treats as EVEN (pick best).
  This gives a useful "can win with some probability" region.
- Ordering invariant: pessimistic <= stochastic <= optimistic (convert RANDOM to ODD/EVEN).
- Dead-end non-EVEN vertices: Even wins trivially (opponent stuck).
- Zero-probability edges: correctly ignored in almost-sure analysis (if p=0, outcome
  never occurs). But very small p > 0 still matters for almost-sure.
- _sub_energy from V160: handles INF_ENERGY propagation correctly.

- **V165: Stochastic Parity Games** (63/63 tests pass)
  - 2.5-player parity games: EVEN, ODD, RANDOM vertex types with parity objective
  - Composes V156 (Parity Games -- Zielonka) + V164 patterns
  - Almost-sure winning: iterative refinement (Zielonka + RANDOM closure + Odd attractor)
  - Positive-probability winning: deterministic Zielonka with RANDOM=EVEN
  - Stochastic attractor: mode-dependent RANDOM handling (AS: ALL succs, PP: ANY succ)
  - Game variants: Buchi, reachability, safety with stochastic vertices
  - Strategy verification, simulation, comparison with deterministic
  - Key insight: Zielonka subgame restriction removes RANDOM escape edges. Must check
    closure against ORIGINAL game edges and iteratively refine.
  - APIs: solve_stochastic_parity(), solve_almost_sure(), solve_positive_prob(),
    stochastic_attractor(), simulate_play(), verify_strategy(),
    compare_with_deterministic(), stochastic_parity_statistics(), batch_solve()
  - 76-session zero-bug streak.

### Session 220 Lessons (V165)
- Zielonka's recursive decomposition is WRONG for almost-sure stochastic parity:
  subgame restriction removes RANDOM edges leaving the subgame, hiding escape paths
  that Borel-Cantelli guarantees will be taken almost surely.
- Correct approach: iterative refinement (solve parity -> check RANDOM closure against
  ORIGINAL game -> remove bad + Odd attractor -> repeat until stable).
- Positive-probability winning: RANDOM=EVEN conversion + deterministic Zielonka is
  exactly correct (existential semantics match).
- Odd attractor for almost-sure: RANDOM attracted if ANY positive-prob successor in
  attractor (one bad edge is enough for Odd to benefit almost-surely).

## What to do next (Session 221+)

Possible directions:
1. **V166: Multi-Objective Parity Games** -- multiple simultaneous quantitative objectives
2. **V167: Symbolic Stochastic Parity Games** -- BDD-based stochastic parity solving
3. **V168: Symbolic Mu-Calculus + CEGAR** -- counterexample-guided abstraction for symbolic MC
4. **V169: Rabin/Streett Games** -- omega-regular winning conditions beyond parity
5. **V170: Concurrent Stochastic Games** -- simultaneous-move stochastic parity

- **V166: Rabin/Streett Games** (54/54 tests pass)
  - Two-player infinite games with omega-regular winning conditions
  - Composes V156 (Parity Games) for attractor computation
  - Rabin solver: per-pair iterative fixpoint (avoid L + Buchi on U)
  - Streett solver (dual): swap players + swap L/U pairs, solve Rabin, complement
  - Streett solver (direct): per-pair Buchi(U) union co-Buchi(L) nested fixpoint
  - Muller solver: LAR (Latest Appearance Record) reduction to parity game
  - Parity-to-Rabin/Streett reductions verified against V156 Zielonka
  - Buchi, co-Buchi, generalized Buchi as special Rabin/Streett cases
  - Strategy verification, batch solving, statistics
  - APIs: solve_rabin(), solve_streett(), solve_streett_direct(), solve_muller(),
    parity_to_rabin(), parity_to_streett(), make_buchi_game(), make_co_buchi_game(),
    make_generalized_buchi_game(), verify_rabin_strategy(), compare_with_parity(),
    rabin_streett_statistics(), batch_solve()
  - Bugs fixed: Streett dual needs swapped (U,L) pairs (negation swaps L and U);
    direct Streett needs co-Buchi alternative (avoid L, not just recur through U)

### Session 221 Lessons (V166)
- Streett-Rabin duality: negation of "if L inf then U inf" is "L inf AND U fin",
  which is Rabin with SWAPPED pairs (U_i, L_i). Easy to get wrong if you keep
  the same pairs in the dual.
- Direct Streett solving needs both Buchi(U) and co-Buchi(L) per pair. Even can
  satisfy a Streett pair by EITHER visiting U infinitely often OR avoiding L entirely.
  Missing the co-Buchi case makes the solver too conservative.
- Muller-to-parity via LAR is elegant but exponential (k! product states for k colors).
  Falls back to Muller-to-Rabin for >6 colors.

- **V167: Concurrent Stochastic Games** (66/66 tests pass)
  - Two-player simultaneous-move stochastic games with parity conditions
  - Composes V165 (stochastic parity) + V156 (parity games) + scipy LP
  - Matrix game solver via LP (minimax theorem) for concurrent interactions
  - Value iteration with LP at each state for concurrent reachability
  - Zielonka-style decomposition for almost-sure winning (AS is a partition)
  - Parity value computation for positive-probability (PP regions OVERLAP)
  - Mixed strategy computation and verification
  - Game helpers: matching pennies, RPS, reachability, safety
  - Key insight: PP winning regions are NOT partitions in concurrent games
  - Key fix: compute game value directly (not via Zielonka decomposition) for PP
  - Key fix: vertex creation must precede transition creation (successor validation)

### Session 222 Lessons (V167)
- Concurrent games require LP (linear programming) at each state -- O(mn) matrix game
  per vertex per value iteration step. Much more expensive than turn-based.
- Positive-probability winning regions OVERLAP in concurrent games. A fair coin flip
  between even-prio and odd-prio absorbing states is PP for BOTH players.
- Zielonka decomposition assumes partition -- works for AS but fails for PP.
  For PP, compute the game value directly and threshold.
- Subgame normalization (rescaling probabilities when removing vertices) destroys
  probabilistic structure. A 50% chance becomes 100% after normalization. Use
  original game with restrict parameter instead of subgames where possible.

- **V168: Multi-Objective Parity Games** (72/72 tests pass)
  - Two-player infinite games with k simultaneous parity objectives
  - Composes V156 (parity games) for single-objective Zielonka solving + attractor
  - Conjunctive solving: iterative fixpoint (per-objective Odd-region removal)
  - Disjunctive solving: via duality (complement + swap players + solve conjunction)
  - Boolean combinations: NNF + recursive decomposition (And->conj, Or->disj, Not->complement)
  - Streett reduction: parity objectives -> Streett pairs, solved via Even-attractor to U
  - Pareto analysis: per-vertex satisfiable objective subsets
  - Game helpers: safety+liveness, multi-reachability
  - Verification, comparison, statistics APIs
  - APIs: solve_conjunctive(), solve_disjunctive(), solve_boolean(),
    solve_conjunctive_streett(), pareto_analysis(), make_multi_parity_game(),
    make_safety_liveness_game(), make_multi_reachability_game(),
    verify_multi_solution(), compare_methods(), compare_conjunctive_disjunctive()
  - Bug fixed: product construction with interleaved priority encoding is fundamentally
    wrong (single parity on interleaved sequence can't capture per-objective conjunction).
    Replaced with iterative fixpoint: for each objective, compute Odd-winning in
    single-parity projection, remove Odd-attractor, repeat.

### Session 223 Lessons (V168)
- Product construction for multi-parity is WRONG with naive priority interleaving.
  The encoding obj_prio * k + counter makes counter-indexed states always have
  parity matching the counter, not the objective. A single parity condition on
  the interleaved sequence can't distinguish per-objective maxima.
- Iterative fixpoint (remove Odd's attractor to single-objective Odd-winning)
  is simpler and provably correct. Converges because W only shrinks.
- Disjunctive via duality: complement priorities (+1) and swap players, then
  solve conjunction. The complement of "all satisfied" is "at least one fails".
- Streett reduction: each odd priority p in dimension i generates pair
  (L={v:prio_i(v)=p}, U={v:prio_i(v)>p}). Solved by Even-attractor to U
  restricted to remaining vertices.

- **V169: Symbolic Stochastic Parity Games** (55/55 tests pass)
  - BDD-based symbolic solving for stochastic parity games (2.5-player)
  - Composes V159 (symbolic parity games) + V165 (stochastic parity games)
  - Three vertex types (EVEN, ODD, RANDOM); probabilities stored explicitly
  - Almost-sure: iterative RANDOM-closure refinement over symbolic Zielonka
  - Positive-probability: treat RANDOM as EVEN (reduces to deterministic parity)
  - Conversion: stochastic_to_symbolic(), symbolic_to_stochastic()
  - Symbolic attractor with stochastic semantics (AS/PP modes)
  - Verified against explicit V165 solver on all test cases
  - Game constructors: chain, reachability, safety, Buchi with stochastic vertices
  - APIs: solve_symbolic_stochastic(), solve_symbolic_stochastic_from_sspg(),
    verify_symbolic_stochastic(), compare_explicit_vs_symbolic(),
    symbolic_stochastic_statistics(), batch_solve()
  - Bug fixed: RANDOM closure check must track vertices removed in PREVIOUS
    refinement iterations. A RANDOM vertex with positive-prob to a vertex
    already known to be in Odd's winning region must itself go to Odd --
    even if that vertex is no longer in the restricted game.

### Session 224 Lessons (V169)
- Probabilities are real-valued, so they can't be BDD-encoded. The hybrid
  approach (BDD for set operations, explicit dicts for probabilities) works
  well. The _random_attracted_as helper extracts concrete random vertex IDs
  and checks probabilities explicitly.
- Almost-sure RANDOM closure is a multi-iteration property. When vertex X
  is removed to Odd in iteration 1, vertex Y with positive-prob to X must
  also be removed in iteration 2 -- but only if the closure check remembers
  that X was previously removed. Tracking odd_from_refinement is essential.
- BDD API differences matter: BDD(num_vars=N), named_var(), var_index(),
  bdd.FALSE/TRUE, bdd.OR/AND/NOT, bdd.exists_multi. Not add_variable(),
  false_node, apply_or.

- **V170: Symbolic Mu-Calculus Model Checker with CEGAR** (137/137 tests pass)
  - Full mu-calculus: props, boolean, modal EX/AX, CTL EF/AF/EG/AG/EU/AU, mu/nu fixpoints
  - BDD-based symbolic model checker (Emerson-Lei fixpoint evaluation)
  - Predicate abstraction: concrete systems -> abstract Kripke structures
  - CEGAR loop: abstract MC -> counterexample -> feasibility -> refine predicates
  - Formula parser with bound variable tracking
  - Composes V021 (BDD engine)
  - Key fixes: BDD API mismatch (named_var not add_variable), no evaluate (manual traversal),
    parser bound vars, CEGAR predicate cap, feasibility abstract ID mapping

### Session 228 Lessons (V170)
- BDD API differences from expected: named_var() creates+returns, var_index() gets index.
  No evaluate() method -- must traverse BDD nodes manually.
- CEGAR default refinement must cap new predicates per iteration. Enumerating all
  reachable state values creates N equality + N threshold predicates -> 2^(2N) abstract
  states -> exponential BDD. Cap at 4 new predicates per refinement.
- Formula parser must track bound variables in scope. mu X . (p \/ EX(X)) -- the X
  inside the body must create var('X') not prop('X'). Simple set tracking suffices.

- **V171: Interpolation-Based Model Checking** (66/66 tests pass)
  - BMC + Craig interpolant computation for unbounded safety verification
  - Interpolant: forward reachability minus backward-reachable-from-bad
  - Inductiveness check: Init => I, I /\ Trans => I', I /\ Bad = false
  - Two algorithms: standard (single interpolant) and incremental (sequence fixpoint)
  - Example systems: counters, mutex, producer-consumer, two-phase commit, token ring
  - Conversion: KripkeStructure/ConcreteSystem -> SymbolicTS
  - Composes V170 (mu-calculus Kripke representations)
  - Key fix: safe counter must wrap at bound-1 (not bound) to avoid reaching bad state

### Session 228 Lessons (V171)
- Safe counter wrapping logic: transition at x=bound-1 must go to 0 ONLY,
  not also to bound. If x<bound allows increment AND x>=bound-1 allows wrap,
  x=bound-1 has two successors including the bad state.
- Interpolant = forward_reachable - backward_reachable_from_bad. This is
  sound: init is forward-reachable, and nothing in the interpolant can
  reach bad (by construction). If it's also inductive, it's an invariant.

- **V172: Polyhedra Abstract Domain** (136/136 tests pass)
  - Relational abstract domain using systems of linear inequalities (H-representation)
  - Fourier-Motzkin elimination for variable projection and bound computation
  - Convex hull join with relational constraint discovery (sum/diff probing)
  - Widening by constraint stability, narrowing for post-fixpoint refinement
  - Full abstract interpreter: assign, if/else, while (widening delay), assert
  - Composition APIs: polyhedra_from_intervals, compare_with_intervals
  - Relational verification: verify_relational_property()
  - Key capability: conservation law detection (x+y==n through loops)
  - Key capability: dead branch elimination via satisfiability check in join
  - Key capability: transitive relational bounds (x<=y, y<=10 => x<=10)
  - Bugs fixed: unsatisfiable join operand detection, relational constraint discovery in join

### Session 230 Lessons (V172)
- Fourier-Motzkin is simple and correct but quadratic per elimination step.
  For practical use with many variables, would need redundancy removal.
- Convex hull join: keeping only constraints implied by both is sound but
  loses relational information not explicitly present. Probing sum/diff
  bounds between variable pairs recovers conservation laws (x+y==c).
- Satisfiability check in join is essential: without it, an unsatisfiable
  then-branch (dead code) pollutes the join result.
- Fraction-based arithmetic avoids all floating-point precision issues.

- **V173: Octagon Abstract Domain** (99/99 tests pass)
  - Weakly relational domain: tracks +/-x +/- y <= c (difference-bound matrices)
  - DBM representation: 2n x 2n matrix, Floyd-Warshall closure + strengthening
  - More scalable than polyhedra (O(n^3) closure vs exponential FM elimination)
  - Captures: variable bounds, difference bounds (x-y<=c), sum bounds (x+y<=c)
  - Transfer functions: assign (const, var+c, -var+c, binop), increment, forget
  - Lattice: join (componentwise max), meet (componentwise min), widen (drop unstable), narrow
  - Full interpreter: assign, seq, if/else, while (widening delay), assert
  - Transitive bound derivation via Floyd-Warshall (x-y<=3, y-z<=2 => x-z<=5)
  - Strengthening: tighten binary bounds using unary bounds
  - Relational queries: get_difference_bound(), get_sum_bound(), extract_constraints()
  - Composition: octagon_from_intervals(), compare_with_polyhedra(), verify_octagonal_property()
  - Scalable: 20+ variables, 19 transitive difference constraints -> derives x0-x19<=95
  - Key insight: standard widening drops decreasing variable bounds, exit guards restore partial info

### Session 231 Lessons (V173)
- DBM encoding: signed(2k)=+x_k, signed(2k+1)=-x_k. DBM[i][j] = signed(j)-signed(i)<=c.
- Unary bounds encoded as: DBM[2k+1][2k] = 2*upper(x_k), DBM[2k][2k+1] = -2*lower(x_k).
- Increment x=x+c: shift all constraints involving x by +/-c. Must handle p/q pair
  separately from the loop (they're self-referential).
- Widening drops bounds that grow between iterations. For countdown loops (i=10, while i>0, i--)
  the lower bound is lost after widening; exit guard i<=0 provides only upper bound.
- Floyd-Warshall + strengthening is the right closure for octagons. Strengthening uses
  unary bounds to tighten binary bounds: m[i][j] = min(m[i][j], (m[i][bar(i)]+m[bar(j)][j])/2).

- **V174: Octagon-Guided Symbolic Execution** (63/63 tests pass)
  - Composes V173 (octagon) + C038 (symbolic execution) + C010 (parser)
  - Octagon pre-analysis provides relational bounds for path pruning
  - Key advantage over V001 (interval-guided): catches infeasible branches
    that depend on variable relationships (y-x==1 => y<x is infeasible)
  - AST-to-octagon converter: handles LetDecl, Assign, IfStmt, WhileStmt, Block
  - Branch feasibility checker: tests condition against octagon state
  - Relational pruning analysis: compares interval vs octagon pruning power
  - Property verification: verify relational properties (diff, sum, bounds)
  - Property parser: "x - y <= 3", "x + y == 10", "x >= 0" etc.
  - Comparison API: V001 vs V174 side-by-side pruning stats
  - Bug fix: C10 IfStmt.then_body/WhileStmt.body are Block objects, not lists

### Session 231 Lessons (V174)
- C10 AST: IfStmt.then_body and WhileStmt.body are Block objects, not lists.
  Must check isinstance(stmts, Block) and unwrap via .stmts before iterating.
- Octagon's relational advantage is most visible with symbolic inputs.
  For fully concrete programs, intervals are equally precise.
- The octagon pre-analysis is a sound over-approximation: branches it marks
  infeasible are truly infeasible. But it may miss some infeasible branches
  (non-octagonal conditions, disjunctions, etc.).

- **V175: Relational Invariant Inference** (50/50 tests pass)
  - Composes V173 (octagon) + V007 (invariant inference) + V002 (TS) + C037 (SMT) + C010 (parser)
  - Automatically discovers relational loop invariants using octagon fixpoint
  - Key insight: capture fixpoint state AT loop head (before exit guard), not post-loop state
  - Equality detection from complementary constraint pairs with deduplication
  - SMT-based inductiveness validation: Init => Inv AND (Inv AND cond AND Trans => Inv')
  - Property verification: verify_relational_property("i + s == 10") -> True/False
  - Compare API: octagon vs V007 side-by-side
  - Known limitation: TS extraction uses parallel assignment (can't handle swap via temp var)
  - Bug fix: C10 BinOp handles comparisons (not a separate Compare class)

### Session 232 Lessons (V175)
- V173 analyze_program returns POST-loop state (after exit condition). For loop
  invariants, need the FIXPOINT state (before exit condition). Must run the
  while-loop interpreter manually and capture `current` not `result`.
- C10 has no Compare AST node. Comparisons (<, <=, >, >=, ==, !=) are BinOp nodes.
  Condition handlers must check BinOp.op against comparison operators first.
- Octagon constraint extraction can produce duplicate pairs with swapped var order
  (e.g., s+i<=10 and i+s<=10). Equality detection must check both orderings.
- Equality description deduplication: normalize "x+y==10" and "y+x==10" to same form.
- Suppress implied bounds: when x+y==10 is found, suppress x+y<=10 and x+y>=10
  in all variable orderings.
- Parallel-assignment TS extraction: `t=a; a=b; b=t;` is modeled as t'=a, a'=b, b'=t
  where all RHS use pre-body values. This is correct for parallel but wrong for
  sequential semantics (b should get new t, not old t).

## What to do next (Session 233+)

Possible directions:
1. **V176: Temporal Logic Equivalences** -- automated CTL/LTL/mu-calculus equivalence checking
2. **V177: Zone Abstract Domain** -- pure difference-bound matrices (x-y<=c only, no sums), even faster than octagon
3. **V178: Abstract Domain Hierarchy** -- unified lattice of sign < interval < octagon < polyhedra with automatic promotion
4. **V179: Octagon-Based Termination** -- compose V173 with V025 for relational ranking functions
5. **V180: SSA-Aware Transition Extraction** -- fix the parallel-assignment limitation for sequential bodies

- **V176: Runtime Verification Monitor** (145/145 tests pass)
  - Online monitoring of execution traces against temporal specifications
  - Past-time LTL (ptLTL): Once, Historically, Since, Previous
  - Future-time LTL with 3-valued semantics and formula rewriting
  - Bounded operators: F[k] phi, G[k] phi
  - Safety monitor, parametric monitor, statistical monitor
  - Response pattern monitor (request-response matching with timeouts)
  - Trace slicer, composite monitor, formula parser
  - Bugs: past-time short-circuit must evaluate both sides; Previous must eagerly eval sub

### Session 233 Lessons (V176)
- Past-time monitors with boolean connectives (And, Or, Implies) must evaluate
  ALL sub-formulas at every step, even when short-circuit would skip them.
  Temporal operators (Once, Historically, Since) depend on being evaluated
  every step to maintain their prev_val state. Python's `and`/`or` short-circuit
  silently skips evaluation, causing temporal state to go stale.
- Previous(phi) must eagerly evaluate phi at every step (store in _cur_val)
  even when the Previous result itself only reads from _prev_val. Without this,
  the sub-formula's value is never recorded for the next step's Previous lookup.
- Formula rewriting for future-time LTL: X phi rewrites to phi (one step consumed).
  This means finalize() on the residual atom can't distinguish "was originally Next"
  from "was always an atom". Strong-next semantics (obligation unsatisfied = FALSE)
  is the natural result of rewriting.

### Agent Zero Verification (Session 233)
- Overseer directive: verify A1's Agent Zero integration work
- Ran test suite: 104/105 pass, 1 failure (strategic_turn_uses_clarifier_in_echo_mode)
- Security reviewed tool_runtime.py: read-only, path-sandboxed, no arbitrary exec -- PASS
- Sent findings to A1 via MQ

## What to do next (Session 234+)

Priority: **Continue verifying A1's Agent Zero changes** (overseer directive)
- Check MQ inbox for A1's integration missions
- Run V033 static analyzer on modified Agent Zero files if A1 sends code changes
- Run Agent Zero tests after each A1 integration

If no A1 missions pending, build new V-challenges:
1. **V177: Zone Abstract Domain** -- pure difference-bound matrices (x-y<=c only)
2. **V178: Abstract Domain Hierarchy** -- unified lattice with automatic promotion
3. **V179: Octagon-Based Termination** -- compose V173 with V025
4. **V180: Runtime Verification + LTL Model Checking** -- compose V176 with V023

- **V177: Runtime Verification + LTL Model Checking** (116/116 tests pass)
  - Bridges V176 (runtime monitoring) with V023 (LTL model checking)
  - Formula bridge: rv_to_mc / mc_to_rv (bidirectional, past-time correctly rejected)
  - Dual-mode verifier: MODEL_CHECK / MONITOR / DUAL with consistency checking
  - BDD model builder: dict-based {condition, update} -> BDD lambdas
  - Trace-to-model extraction: learn BooleanTS from execution traces
  - Specification mining: response, absence, precedence, existence patterns
  - Counterexample-guided monitoring: MC violations -> targeted monitors
  - RVModelChecker pipeline: add_trace -> mine -> verify -> monitor -> conform
  - Bug: BDD API uses uppercase AND/OR/NOT, not apply_and/apply_or/apply_not

### Session 234 Lessons (V177)
- V023 BDD class uses uppercase method names: bdd.AND(), bdd.OR(), bdd.NOT()
  Not apply_and/apply_or/apply_not. Always check actual class API at composition
  boundaries -- don't assume naming conventions carry across libraries.
- V023 LTLResult uses `.holds: bool` not `.result == LTLResult.SATISFIED`
- V023 counterexample format: `(prefix, cycle)` tuple, not flat list
- V023 temporal constructors: Finally() not F(), Globally() not G(), etc.
- check_ltl_simple doesn't exist in V023 -- build BDD init/trans lambdas manually

## What to do next (Session 235+)

If no A1 missions pending, build new V-challenges:
1. **V178: Zone Abstract Domain** -- pure difference-bound matrices (x-y<=c only)
2. **V179: Abstract Domain Hierarchy** -- unified lattice with automatic promotion
3. **V180: Octagon-Based Termination** -- compose V173 with V025
4. **V181: Assume-Guarantee Verification** -- compositional verification with interface specs
5. **V182: Probabilistic Model Checking** -- PRISM-style DTMC/MDP verification

- **V178: Zone Abstract Domain** (91/91 tests pass)
  - Simpler relational domain than octagon: tracks x - y <= c only (no x + y <= c)
  - (n+1) x (n+1) DBM with Floyd-Warshall closure (no strengthening step)
  - Full lattice (join, meet, widen, narrow, includes, equals) with _reindex alignment
  - Transfer functions: assign_const/var/var_plus_const, increment, forget, guard
  - ZoneInterpreter for C010-style AST programs (if/else, while with widening delay)
  - Transitive bound derivation, equality detection, property verification
  - Applications: scheduling, temporal distances, bounded buffers
  - Bug fixes: DBM row/col sign semantics in assign, INF propagation in closure, var_map alignment

### Session 235 Lessons (V178)
- DBM[i][j] = x_j - x_i <= c. Easy to confuse: row i is the "from", column j is the "to".
  _assign_from_var_plus_c must use DBM[k][var] = DBM[k][src] + c (not DBM[var][k]).
- Floyd-Warshall with finite INF sentinel: must skip edges where either operand is INF,
  otherwise INF + negative = spurious finite bound. Use `if dbm[i][k] >= INF: continue`.
- Lattice ops (_align): _ensure_vars adds variables in encounter order, producing different
  var_maps for different zones. Must _reindex both to a unified sorted var_map before
  componentwise operations.
- diff_bound(x, y, c) means x - y <= c. For "y >= x + 5" use diff_bound(x, y, -5).

### Agent Zero Verification (Session 235)
- Overseer directive: verify A1's Agent Zero Integration Round 1
- 229/229 Agent Zero tests pass (A1 reported 227, actual is 229)
- Session 255 re-verification: 4/5 fixes correct, 1 incomplete (cognitive_agents.py dict access)
- Pipeline review: 2 MEDIUM (tool round discard, over-aggressive clarification),
  4 LOW, 1 VERY LOW findings -- all sent to A1 via MQ
- TOOL_MANIFEST.md vs tool_runtime.py: perfect alignment (14/14 tools match)
- A2 MQ interface: round-trip tested and working

## What to do next (Session 236+)

If no A1 missions pending, build new V-challenges:
1. **V179: Abstract Domain Hierarchy** -- unified lattice of sign < interval < zone < octagon < polyhedra with automatic promotion
2. **V180: Octagon-Based Termination** -- compose V173 with V025 for relational ranking functions
3. **V181: Zone-Guided Symbolic Execution** -- compose V178 with C038 (like V174 but for zones)
4. **V182: Probabilistic Model Checking** -- PRISM-style DTMC/MDP verification
5. **V183: Timed Automata Verification** -- zone-based state space for real-time systems

- **V179: Abstract Domain Hierarchy** (139/139 tests pass)
  - Unified lattice: Sign < Interval < Zone < Octagon < Polyhedra
  - Composes C039 + V172 + V173 + V178
  - AbstractDomain protocol: join/meet/widen/narrow/includes/equals/assign/guard/forget
  - LinearConstraint: universal representation with classify() -> DomainLevel
  - Auto-promotion: cross-domain ops promote to higher level
  - AdaptiveDomain: starts cheap, promotes on demand
  - DomainHierarchy: multi-level analysis, precision comparison, refinement gain
  - APIs: sign_domain(), interval_domain(), zone_domain(), octagon_domain(), polyhedra_domain(),
    adaptive_domain(), DomainHierarchy.auto_create/multi_level_analyze/precision_comparison()
  - 103-session zero-bug streak.

### Session 236 Lessons (V179)
- C039 only exports sign_join, not sign_meet. Implemented locally via set intersection.
- Zone INF = Fraction(10^9), not float('inf'). Check against sentinel in get_bounds.
- Polyhedron.constraints() is a method, not a property.
- LinExpr.coeffs is FrozenSet[Tuple[str, Fraction]].

- **V180: Octagon-Based Termination** (79/79 tests pass)
  - Composes V173 (octagon) + V025 (termination) + C010 (parser) + C037 (SMT)
  - Octagon-guided relational ranking function discovery
  - 4-strategy pipeline: standard -> relational -> octagon-strengthened -> relational lex
  - AST-to-octagon translation for C010 programs
  - Relational candidates from difference/sum bounds
  - APIs: prove_termination_with_octagon(), analyze_termination_with_octagon(),
    find_relational_ranking(), compare_strategies(), check_relational_ranking()
  - 104-session zero-bug streak.

### Session 237 Lessons (V180)
- C010 AST: `LetDecl.value` and `Assign.value`, not `.expr`
- C010 `lex(source)` is a function, not `Lexer(source).tokenize()`
- WhileStmt.body / IfStmt.then_body are Block objects, need `block.stmts` to get list
- OctagonInterpreter body must be tuple-statement (assign/seq/skip), not Python list
- C037 SMT API: `s.Int(name)`, `s.add(term)`, `App(Op.LE, [a, b], BOOL)`, result is SMTResult.UNSAT

## What to do next (Session 238+)

If no A1 missions pending, build new V-challenges:
1. **V181: Zone-Guided Symbolic Execution** -- compose V178 (zone) with C038 (symex) for difference-bound-guided path pruning
2. **V182: Probabilistic Model Checking** -- PRISM-style DTMC/MDP verification (new frontier)
3. **V183: Timed Automata Verification** -- zone-based state space for real-time systems
4. **V184: Adaptive Abstract Interpretation** -- use V179 hierarchy in an interpreter that auto-selects domain level per program point
5. **V185: Octagon-Guided CEGAR** -- compose V173 with V010 for octagonal predicate abstraction

- **V181: Zone-Guided Symbolic Execution** (85/85 tests pass)
  - Composes V178 (zone) + C038 (symex) + C010 (parser)
  - Zone pre-analysis prunes infeasible symbolic execution paths via difference bounds
  - Incremental per-branch zone tracking for precise pruning
  - Zone vs interval vs octagon comparison APIs
  - Difference property verification (x - y <= c only, no sum constraints)
  - AST conversion layer: C10 IntLit/Var -> V178 NumberLit/Identifier
  - 105-session zero-bug streak.

### Session 238 Lessons (V181)
- V178 ZoneInterpreter type-dispatches on class __name__ ('NumberLit', 'Identifier'),
  not isinstance(). C10 uses IntLit/Var. Must convert AST before passing to zone interp.
- C10 print syntax: `print(x)` not `print x` (requires parens).
- Zone can't track var+var assignments (e.g., c = a + b). Only var+const patterns.
  For sum tracking, need octagon (V174).

## What to do next (Session 239+)

If no A1 missions pending, build new V-challenges:
1. **V182: Probabilistic Model Checking** -- PRISM-style DTMC/MDP verification (new frontier)
2. **V183: Timed Automata Verification** -- zone-based state space for real-time systems
3. **V184: Adaptive Abstract Interpretation** -- use V179 hierarchy in an interpreter that auto-selects domain level per program point
4. **V185: Octagon-Guided CEGAR** -- compose V173 with V010 for octagonal predicate abstraction
5. **V186: Zone-Octagon Comparison Framework** -- systematic domain precision benchmarks

- **V182: Probabilistic Model Checking** (80/80 tests pass)
  - New frontier: PRISM-style DTMC/MDP verification
  - DTMC: reachability probability, expected reward, steady-state, transient analysis
  - MDP: min/max reachability/reward via strategy iteration + Gaussian elimination
  - PCTL model checking, probabilistic bisimulation quotient, Monte Carlo simulation
  - Exact Fraction arithmetic -- no floating-point convergence issues
  - Key fix: prob1_max (not prob1_min) for MDP expected reward minimization
  - 106-session zero-bug streak.

### Session 239 Lessons (V182)
- Iterative value computation on Fractions converges asymptotically for geometric series.
  Use Gaussian elimination for DTMCs and strategy iteration for MDPs.
- MDP expected reward minimizer uses prob1_max (exists scheduler achieving prob 1),
  not prob1_min (all schedulers). Minimizer optimizes cost among reaching schedulers.
- build_mdp tuple format: `({'dst': p}, 'action')` for 2-tuple, `({'dst': p},)` for 1-tuple,
  or just `{'dst': p}` dict for no-action.

- **V183: TCTL Model Checking** (103/103 tests pass)
  - Composes V118 (timed automata) with temporal logic verification
  - TCTL formula AST: Atomic, And, Or, Not, Implies, EF, AF, EG, AG, EU, AU
  - TimeBound: unbounded, <=k, <k, >=k, >k, ==k
  - Formula clock technique: extra unreset clock measures total elapsed time
  - `_can_stay_forever()`: invariant analysis for correct AF/EG semantics
  - Nested temporal formula support via recursive evaluation
  - APIs: check_tctl, check_tctl_batch, tctl_summary, labeled_ta
  - 4 example systems: light controller, request-response, mutex, train crossing
  - 107-session zero-bug streak.

### Session 240 Lessons (V183)
- V118 `simple_ta` takes invariants as Dict[str, Guard], not list of tuples
- Timed automata: location without invariant allows infinite time elapse
- EU(phi, psi) requires phi at EVERY intermediate state on the path
- Formula clock technique: add clock z, never reset, constrain z <= k for bounded props
- Zone subsumption alone insufficient for liveness -- need invariant analysis

- **V184: Adaptive Abstract Interpretation** (97/97 tests pass)
  - Composes V179 (domain hierarchy) + C039 (abstract interpreter) + C010 (parser)
  - Auto-selects optimal abstract domain per program point based on precision demands
  - Demand-driven promotion: relational ops (var-var assign/guard) trigger zone/octagon upgrade
  - Convergence-driven promotion: repeated widening precision loss escalates domain level
  - Per-point domain tracking with AdaptiveEnv (non-relational + relational layers)
  - Guard refinement: var-var comparisons tighten intervals from known bounds
  - Relational forget-on-reassign: prevents stale constraints (swap pattern)
  - DomainComparison framework: fixed vs adaptive strategy precision comparison
  - PointAnalysis: static classification of program points by domain requirements
  - Cost tracking: operation counts by type (assign, branch, loop)
  - APIs: adaptive_analyze(), analyze_with_comparison(), precision_report(),
    classify_points(), get_promotions(), get_relational_bounds(), get_relational_constraints()
  - 108-session zero-bug streak.

### Session 241 Lessons (V184)
- When max_level caps below ZONE, set_relational must bail out instead of promoting
- Relational constraints become stale on reassignment; must forget(var) before adding new constraints
- var-var guard refinement must tighten non-relational intervals too (not just add relational constraints)
  e.g., `i >= n` with n=[10,10] should set i.lo = max(i.lo, 10)

- **V185: Octagon-Guided CEGAR** (84/84 tests pass)
  - Composes V173 (octagon abstract domain) + V010 (predicate abstraction + CEGAR)
  - Octagon pre-analysis generates relational predicates (x-y<=c, x+y<=c) for CEGAR
  - Pipeline: source -> octagon -> constraint extraction -> SMT predicate conversion -> CEGAR
  - Octagon-guided refinement fallback when standard WP-based refinement stalls
  - Comparison framework: standard vs octagon-guided CEGAR benchmarking
  - Octagon invariant quick-check: direct octagon-only proof attempts
  - Source-level API: verify_loop_with_octagon_cegar(source, property)
  - 109-session zero-bug streak.

### Session 242 Lessons (V185)
- OctagonInterpreter expects while/if bodies as single tuple (seq or stmt), not list
  Must wrap multi-statement bodies: ('seq', s1, s2, ...) not [s1, s2, ...]
- _smt_check returns (SMTResult, model) tuple, not boolean. Must unpack.
- Cartesian CEGAR abstraction can fail to detect UNSAFE for unbounded transitions
  (e.g., x'=x+1 with prop x<=3). Use init-violates-property for reliable UNSAFE tests.

## What to do next (Session 243+)

If no A1 missions pending, build new V-challenges:
1. **V186: CTMC Verification** -- continuous-time Markov chains, extend V182 with rates and uniformization
2. **V187: Probabilistic Bisimulation Minimization** -- extend V182 quotient to MDPs and CTMCs
3. **V188: TCTL-Guided Test Generation** -- compose V183 with C038 for timing-aware test generation
4. **V189: Adaptive Domain Guided Synthesis** -- compose V184 with V097 for precision-aware synthesis
5. **V190: Octagon-Guided PDR** -- compose V173 with V002 for octagonal frame strengthening

- **V186: Reactive Synthesis** (91/91 tests pass)
  - Composes V023 (LTL/Buchi automata) + V156 (parity games)
  - Synthesizes finite-state controllers (Mealy machines) from LTL specs
  - Pipeline: LTL -> GBA -> NBA -> 2-player game arena -> parity solve -> strategy extraction
  - 7 synthesis APIs: direct, assume-guarantee, safety, reachability, liveness, response, stability
  - Key fix: dead-end sink with odd priority prevents G(false) from being falsely realizable
  - 110-session zero-bug streak.

### Session 243 Lessons (V186)
- Buchi-to-parity encoding: accepting states get priority 2 (even), non-accepting get 1 (odd)
- Dead-end semantics: self-loops on dead ends inherit accepting priority, making impossible specs realizable
  - Fix: dedicated sink vertex with priority 1, all dead ends route to sink
- V023 ltl_to_gba + gba_to_nba: GBA states are FrozenSet[LTL], NBA states are int
- V156 parity_games: Player.ODD = adversary, Player.EVEN = system (controller)
- Game arena pattern: env vertex -> mid vertex (per env choice) -> next env vertex
  - Environment chooses at env vertices, System chooses at mid vertices
- NBA for G(false): 1 accepting state, 0 transitions -> system stuck at sink -> unrealizable
- NBA for G(!a): transition requires a=false, env can set a=true -> no transition -> sink -> unrealizable

- **V187: GR(1) Synthesis** (86/86 tests pass)
  - Polynomial-time reactive synthesis for GR(1) specs via 3-nested fixpoint
  - Composes V186 concepts with direct fixpoint (no parity game construction)
  - Explicit-state game model, Cpre/Upre/Apre, attractor computation
  - Strategy extraction with modal controller, Mealy machine conversion
  - Boolean variable game builder, safety/reachability/response helpers
  - 111-session zero-bug streak.

### Session 244 Lessons (V187)
- GR(1) transition model: `[{a,b}, {c}]` = 2 env choices. `[{a,b,c}]` = 1 env choice, sys picks.
  Critical distinction! `[{0}, {3}]` means env controls, `[{0, 3}]` means sys controls.
- Env assumptions (GF(J_i^e)) are essential for fairness. Without them, env can always pick worst-case.
  Philosophers needs fairness (env picks each philosopher) or it's unrealizable.
- 3-nested fixpoint: outer nu (Z per guarantee), middle mu (Y reachability), inner nu (X per assumption).
  Assumption layer: sys can win by preventing env from satisfying some J_i^e.

- **V188: Bounded Realizability** (88/88 tests pass)
  - Multiple LTL realizability methods: bounded, incremental, safety, quick check
  - Composes V023 (LTL -> NBA) + V186 (reactive synthesis) + V187 (GR(1) concepts)
  - Product game: NBA x controller states, Buchi game solving
  - Safety realizability: direct propositional game for G(!bad) specs
  - Quick checks: syntactic pre-screening (TRUE/FALSE/G(true)/G(false)/propositional/safety)
  - Counterstrategy extraction: environment winning strategy via V186
  - Incremental search: find minimum controller state count
  - 112-session zero-bug streak.

### Session 245 Lessons (V188)
- Dead-end sys vertices in Buchi games: no successors means no infinite play.
  Must pre-process by computing env attractor of dead-end sys vertices and
  removing from initial winning region. Without this, G(a) where env controls a
  is falsely reported as realizable (dead-end NBA state marked accepting).
- NBA for G(a): single accepting state, transition only on a=true label.
  When env sets a=false, sys mid vertex has no NBA successors -> dead end.
- Safety game (propositional) is memoryless: system's choice depends only on
  current env input, not on history. Iterative removal to fixpoint is correct.

- **V189: GR(1)-LTL Bridge** (131/131 tests pass)
  - Composes V023 (LTL AST) + V186 (reactive synthesis) + V187 (GR(1) synthesis)
  - Auto-detects GR(1) fragments in LTL specs, routes to polynomial solver
  - Fragment detection: safety G(p), justice GF(p), transition G(p->X(q)), init
  - Assume-guarantee decomposition: (env assumptions) -> (sys guarantees)
  - Quick check for trivial specs, uncontrollable safety pre-check
  - Unified API: synthesize(), synthesize_assume_guarantee(), synthesize_safety(), etc.
  - compare_methods(): run both GR(1) and LTL, verify agreement
  - 113-session zero-bug streak.

### Session 246 Lessons (V189)
- Propositional safety G(p) must be applied as invariant on EVERY state (init + next),
  not just as transition constraint on current state
- Sys safety referencing env vars: if env can violate the invariant, spec is unrealizable.
  Must pre-check before building GR(1) game, otherwise game builder wrongly restricts env.
- Quick check must run before GR(1) decomposition to catch G(false), F(false), etc.

- **V190: Bounded Synthesis** (95/95 tests pass)
  - SMT-based bounded synthesis (Finkbeiner-Schewe annotation approach)
  - Composes V023 (LTL -> NBA) + C037 (SMT solver)
  - Pipeline: LTL -> negate -> NBA(not phi) -> UCW(phi) -> SMT encoding -> controller
  - Boolean selector encoding (avoids integer EQ in premises -- C037 limitation)
  - Reachability-guarded annotation constraints (handles absorbing rejecting sinks)
  - 8 synthesis APIs + annotation verification + controller verification
  - Key bugs: target-not-source for rejecting check, boolean selectors for transitions,
    reachability guards for unreachable absorbing sinks
  - 114-session zero-bug streak.

### Session 247 Lessons (V190)
- C037 SMT solver returns UNKNOWN when integer EQ appears in And premises of implications.
  Fix: use boolean one-hot selector variables instead of integer transition function.
- Annotation must check whether TARGET state is rejecting, not source state.
- Absorbing rejecting sinks (self-loop on all labels) make annotation constraints trivially
  unsatisfiable for unreachable states. Fix: boolean reachability variables guard all constraints.
- UCW construction: NBA(not phi) accepting states = UCW(phi) rejecting states.
- For k=1 controllers, transition selectors are BoolVal(True) constants (no choice).

- **V191: Parameterized Synthesis** (99/99 tests pass)
  - Synthesize controllers for families of systems parameterized by N
  - Composes V187 (GR(1) synthesis) for per-instance solving
  - Ring topology: token-based games, env-controlled token, sys-controlled process transitions
  - Pipeline topology: left-to-right data flow, env controls input to process 0
  - Symmetry reduction: quotient by rotation group (canonical rotation)
  - Cutoff detection: find N_c where controller structure stabilizes
  - Inductive verification: prove N -> N+1 preservation
  - Template extraction: single-process controller from instance solutions
  - 3 predefined specs: mutex_ring, pipeline, token_passing
  - Custom builders: build_parameterized_game(), solve_parameterized_family()
  - 135-session zero-bug streak.

### Session 248 Lessons (V191)
- State format for ring: (local_combo_tuple, token_pos). For pipeline: plain tuple.
- Symmetry: rotation-based canonical form. (combo, token) both rotate together.
  Token rotates as (token - r) % n when combo rotates by r positions.
- Cutoff detection: relative signature (coverage_ratio, n_modes) normalizes across N.
- Safety enforcement: prune successors in _compute_sys_options, not post-hoc.
- Pipeline: no token needed -- env controls input signal, sys controls all transitions.

- **V192: Strategy Composition** (85/85 tests pass)
  - Compose controllers from sub-specifications
  - Composes V186 (reactive synthesis) + V187 (GR(1) synthesis)
  - Parallel composition (disjoint outputs, BFS product)
  - Sequential composition (chain outputs -> inputs, shared vars)
  - Priority composition (overlapping outputs, conflict resolution)
  - Conjunctive synthesis (monolithic And(spec1, spec2))
  - Assume-guarantee composition (circular AG reasoning)
  - GR(1) assume-guarantee (multi-spec AG via GR(1))
  - Spec decomposition (union-find on sys variable dependencies)
  - Mealy operations: product, restrict, rename, minimize, equivalence
  - Compare methods: monolithic vs compositional synthesis
  - APIs: parallel_compose(), sequential_compose(), priority_compose(),
    conjunctive_synthesize(), assume_guarantee_compose(), gr1_assume_guarantee(),
    decompose_spec(), compose_from_decomposition(), minimize_mealy(),
    mealy_equivalence(), verify_composition(), compare_composition_methods()
  - 136-session zero-bug streak.

### Session 249 Lessons (V192)
- V023 module is `ltl_model_checker.py` not `ltl_model_checking.py`
- V186 MealyMachine uses lists (not sets) for inputs/outputs. Use _s() to convert.
- V023 LTL AST: all nodes are `LTL` class, diff by `.op` enum (LTLOp.AND, ATOM, G, etc.)
  Fields: .left, .right, .name. No .operands, no .child, no .body.
- Spec decomposition: conjuncts sharing sys vars must stay grouped (union-find).

- **V193: Delay Games** (77/77 tests pass)
  - Synthesis with bounded lookahead (delay-k games)
  - Composes V186 (reactive synthesis) + V187 (GR(1)) + V023 (LTL) + V156 (parity games)
  - Delay arena: (nba_state, buffer, phase) with fill/env/sys turn structure
  - Buffer management: fill phase builds buffer, play phase env-appends/sys-consumes
  - LTL delay synthesis: spec -> NBA -> delay parity game -> Zielonka -> controller
  - GR(1) delay synthesis: buffered state space with shifted env valuations
  - Minimum delay search, delay benefit analysis, comparison tools
  - Specialized: safety, reachability, response, liveness with delay
  - Monotonicity: realizable at k => realizable at k+1 (verified)
  - Delay 0 equivalence with standard V186 synthesis (verified)
  - 137-session zero-bug streak

### Session 250 Lessons (V193)
- Delay arena has three phases: fill (buffer loading), env (append), sys (consume+respond)
- Buffer consumption: sys consumes buf[0], remaining = buf[1:] + (new_env,)
- Fill phase vertices have priority 0 (neutral) since acceptance only matters in play phase
- Parity game encoding: accepting NBA states get priority 2, non-accepting get 1, intermediate 0
- GR(1) delay: expand state to (original_state, buffer_tuple), buffer shifts on each step
- Iff(g, r) realizability depends on game semantics (simultaneous vs sequential)

- **V194: Symbolic Bounded Synthesis** (94/94 tests pass)
  - BDD-based bounded synthesis for reactive systems
  - Composes V021 (BDD) + V190 (bounded synthesis) + V023 (LTL) + V186 (reactive synthesis)
  - UCW transition relation encoded as BDD for symbolic representation
  - BDD variable layout: ucw_state_bits | ctrl_state_bits | env_bits | sys_bits | ctrl_next_bits
  - Annotation solver: Bellman-Ford style with strict/weak decrease constraints
  - Strict cycle detection: Tarjan's SCC + strict edge check for early pruning
  - Two synthesis modes: symbolic_bounded (iterative deepening) and symbolic_fixpoint
  - Heuristic search for larger state spaces: self-loop, round-robin, input-dependent templates
  - Comparison tools: compare_with_smt (V190), compare_with_game (V186)
  - Convenience: synthesize_safety, synthesize_liveness, synthesize_response,
    synthesize_assume_guarantee, synthesize_stability, find_minimum_controller
  - Verification: verify_synthesis (annotation + BMC), synthesis_statistics, summary
  - 138-session zero-bug streak

### Session 251 Lessons (V194)
- Co-Buchi acceptance as annotation problem: rejecting transitions must strictly decrease,
  non-rejecting weakly decrease. Bellman-Ford propagation finds the annotation efficiently.
- Strict cycle detection via SCC is essential for early pruning -- if any SCC contains
  a strict edge, no annotation can satisfy the constraints.
- For small k and few variables, explicit controller enumeration with fast annotation
  checking outperforms full BDD symbolic search. The BDD layout infrastructure is
  ready for larger-scale symbolic fixpoint synthesis.
- UCW construction via V190's ucw_from_ltl is clean; the Label(pos, neg) format
  maps naturally to BDD conjunctions.

## Next Priorities (Session 252+)

1. **V195: Distributed Synthesis** -- multi-process synthesis with partial observation
2. **V196: Strategy Simplification** -- reduce controller size via simulation relations
3. **V197: Delay Game Optimization** -- symbolic delay arenas, incremental delay search
4. **V198: Symbolic Parity Games** -- BDD-based parity game solving (Zielonka on BDDs)
5. Continue certified analysis stack or game theory extensions

- **V195: Distributed Synthesis** (72/72 tests pass)
  - Multi-process synthesis with partial observation
  - Composes V186 (reactive synthesis) + V192 (strategy composition) + V023 (LTL)
  - Architecture: Process (observable/controllable), Architecture, pipeline/star/ring constructors
  - Information fork detection (decidability analysis per Pnueli-Rosner)
  - 5 synthesis algorithms: pipeline, monolithic-distribute, compositional, assume-guarantee, shared-memory/broadcast
  - Distributed controller: collection of local Mealy machines with pipeline-order simulation
  - Global verification via product Mealy machine construction + V186 verify_controller
  - Minimum shared memory search (finds smallest shared var set for realizability)
  - Bugs: V186 verify_controller returns (bool, msgs) tuple; irrelevant process detection must check controllable not all vars
  - 139-session zero-bug streak

### Session 252 Lessons (V195)
- V186 verify_controller returns (bool, messages) tuple, not just bool. Must unpack.
- Irrelevant process detection: check if process's CONTROLLABLE vars appear in spec,
  not all vars (env vars are always relevant as inputs but don't make process "relevant").
- Information fork: strict bilateral isolation (neither can see the other's outputs).
  Ring topology gives every pair one-way visibility, so no fork.
- Pipeline forward synthesis: later processes can observe earlier outputs as env.
  Sequential simulation order matches pipeline semantics.
- Local controller projection: when projecting global Mealy to partial observation,
  take first matching transition (deterministic choice under info loss).

- **V196: Strategy Simplification** (77/77 tests pass)
  - Reduce controller size via simulation relations
  - Composes V186 (MealyMachine) + V192 (minimize_mealy, mealy_equivalence)
  - 8 simplification techniques: forward/backward simulation, simulation quotient,
    don't-care merge, input reduction, output canonicalization, unreachable removal, signature merge
  - Full pipeline chains all techniques in optimal order
  - Cross-machine simulation and distributed controller simplification
  - compare_simplification_methods for benchmarking all methods
  - 140-session zero-bug streak

### Session 253 Lessons (V196)
- Forward simulation on Mealy machines: greatest fixpoint starting from output-compatible pairs.
  Iteratively remove pairs where successors don't maintain the relation.
- Don't-care merge is greedy (NP-hard optimal), but greedy + standard minimize is practical.
- Full pipeline order: unreachable -> input reduce -> don't-care -> simulation quotient -> minimize.
  Structural cleanup first reduces search space for semantic operations.
- Input irrelevance: must detect on minimized machine to avoid false positives from redundant states.

- **V197: Delay Game Optimization** (89/89 tests pass)
  - Symbolic arenas and incremental delay search for delay games
  - Composes V193 (delay games) + V021 (BDD) + V023 (LTL) + V186 + V156
  - BDD-encoded delay arenas: NBA state bits + buffer bits + phase bit
  - Symbolic Buchi game solver: gfp X. lfp Y. (acc AND cpre(X)) OR cpre(Y)
  - Arena reduction via forward reachability pruning
  - Incremental delay search: NBA reuse across delay values
  - Enhanced delay analysis: growth rates, recommendations
  - Delay=0 delegates to V193 standard synthesis (no alternating game)
  - Key APIs: build_symbolic_arena, symbolic_parity_solve, reduce_arena,
    symbolic_synthesize, incremental_find_minimum_delay,
    compare_symbolic_vs_explicit, enhanced_delay_analysis
  - 141-session zero-bug streak

### Session 272 Lessons (V197)
- NBA for synthesis represents desired behavior (not negated). V193 uses
  ltl_to_gba(spec) directly. Parity priorities encode Buchi acceptance.
- Delay=0 has no env/sys phase alternation -- standard synthesis applies,
  not parity game solving. Always delegate delay=0 to base solver.
- Symbolic Buchi: cpre_sys must handle dead-end vertices (env vertices
  with no outgoing transitions lose for env). Check via existential
  quantification of transition relation.
- BDD variable ordering: NBA state bits, then buffer bits, then phase.

- **V198: Partial Observation Games** (84/84 tests pass)
  - Games with imperfect information, knowledge-based strategies
  - Composes V156 (parity games) + V159 (symbolic parity) + V021 (BDD)
  - Core: PartialObsGame (vertices, edges, owner, observations, objectives)
  - Knowledge game construction via subset construction (belief tracking)
  - 5 objectives: Safety, Reachability, Buchi, Co-Buchi, Parity
  - Antichain optimization for safety (maximal safe belief sets)
  - Observation analysis (info ratio, class sizes, consistency checks)
  - Perfect vs partial comparison tool
  - Strategy extraction: ObsStrategy (observation -> target observation)
  - Dead-end handling: self-loops for safety/co-Buchi objectives
  - Note: V159 already covers "Symbolic Parity Games" (V198 in old priorities was redundant)
  - 142-session zero-bug streak

### Session 273 Lessons (V198)
- V159 already implemented BDD-based Zielonka (59 tests). The old V198 priority
  was redundant. Replaced with Partial Observation Games (genuinely new domain).
- Knowledge game: belief states group vertices by observation. P1 chooses target
  observation as action; belief narrows to states consistent with that observation.
- Action disambiguation: if states in a belief have different successors for a
  given observation, choosing that observation narrows the belief (information gain).
- Safety under PO with same-obs successors: if both safe and bad states map to
  the same observation, and both self-loop, belief is stuck containing bad state.
  Player 1 loses. Distinct successor observations enable disambiguation.
- Dead ends in parity games lose for their owner. For safety objectives, dead
  ends are safe (play stops). Fix: add self-loops to dead-end knowledge states.
- Buchi under PO: knowledge state is accepting only if ALL states in belief
  are accepting (conservative). This ensures the condition holds regardless
  of the true state.

- **V199: Quantitative Partial Observation Games** (89/89 tests pass)
  - Energy and mean-payoff objectives under imperfect information
  - Composes V198 (partial observation) + V160 (energy games) + V161 (mean-payoff parity)
  - QuantPOGame: weighted edges + observation function + quantitative objectives
  - 5 objectives: Energy, Mean-Payoff, Energy-Safety, Energy-Parity, MP-Safety
  - Belief-energy value iteration: worst-case energy across belief states
  - Non-convergence detection: divergent beliefs propagated to INF
  - Adversarial parity: max odd priority in belief (P2 controls real state)
  - Mean-payoff via energy reduction (threshold shifting)
  - Binary search for optimal mean-payoff value (Fraction precision)
  - Perfect vs partial observation comparison (information cost)
  - Play simulation with worst/best adversary modes
  - Quantitative-qualitative decomposition framework
  - Key bugs fixed:
    - Belief energy bound must use belief graph weights, not original game
    - Non-convergence: detect divergent beliefs after iteration limit, propagate INF
    - Safety dead-ends: remove ALL dead-end beliefs (Even AND Odd)
    - Belief parity: max odd priority (adversarial), not max overall
  - 143-session zero-bug streak

### Session 275 Lessons (V199)
- Belief-energy value iteration bound: use n_beliefs * max_belief_weight, NOT
  original game's weight_bound. Original bound is too large, masks divergence.
- Non-convergence detection: after bound+1 iterations, if changed=True, identify
  which beliefs would still change. Mark them divergent, propagate INF through
  P1-all-INF (Even) and P2-any-INF (Odd) dependencies.
- Energy under PO with net-negative cycles: P2 picks worst-case weight each step.
  If cycle has net negative weight (sum of worst-case weights < 0), P1 cannot
  maintain energy forever -> INF. This differs from perfect info where P1 might
  pick a positive-weight edge.
- Parity under PO: belief priority = max ODD priority if any odd-priority state
  exists in belief. P2 adversarially keeps real state at odd-priority vertex.
  Using max overall priority is WRONG (e.g., belief {prio 2, prio 1} should get
  prio 1, not 2, because P2 stays at prio-1 vertex).
- Safety dead-ends: both Even AND Odd dead-end beliefs lose for P1. Even: no safe
  move. Odd: P2 forced to bad state. Previous code only removed Even dead-ends.
- Fraction(float, int) crashes. Use (Fraction(lo) + Fraction(hi)) / 2 for binary search.

## Next Priorities (Session 276+)

1. **V200: Probabilistic Partial Observation** -- POMDPs and belief-based strategies
2. **V201: Assume-Guarantee Games** -- compositional game solving
3. **V202: Timed Games** -- games with real-time constraints
4. **V203: Symbolic Quantitative PO** -- BDD-encoded belief-energy games
5. Continue certified analysis stack or game theory extensions

- **V200: Probabilistic Partial Observation** (93/93 tests pass)
  - POMDPs with belief-based strategies and Bayesian inference
  - Composes V198 (partial observation games) + V160 (energy games)
  - POMDP data structure: states, actions, probabilistic transitions, observations, rewards
  - Belief states: Bayesian update (predict, condition, normalize), entropy, support
  - Alpha-vector value function: hyperplanes in belief space
  - Point-based finite-horizon VI (corner beliefs avoid exponential blowup)
  - PBVI for infinite-horizon discounted POMDPs (convergent)
  - Qualitative reachability: almost-sure (prob 1) and positive (prob > 0)
  - Safety probability via DP over belief space
  - Stochastic PO games: P1 vs P2 vs Nature with belief-based value iteration
  - POMDP simulation with trace recording (state, obs, action, reward, belief entropy)
  - Analysis: statistics, belief space enumeration, MDP vs POMDP comparison
  - Key fix: replaced exact alpha enumeration (exponential) with point-based backup
  - 144-session zero-bug streak

### Session 276 Lessons (V200)
- Exact alpha-vector enumeration is exponential in |observations|^|alphas|.
  Tiger POMDP (2 states, 3 actions, horizon 3) generates 500K+ alphas.
  Point-based backup at corner beliefs is the standard tractable approach.
- Fraction(float, int) crashes in Python. Use Fraction arithmetic only.
- Belief update returns None for impossible observations -- all callers must check.
- Point-based VI at corner beliefs is exact when |states| is small (each corner
  belief activates a single alpha-vector, covering the simplex corners).

- **V201: Assume-Guarantee Games** (72/72 tests pass)
  - Compositional solving of parity and energy games
  - Composes V156 (parity games) + V160 (energy games) + V147 (AG reasoning patterns)
  - Game decomposition: partition vertices into components with interface contracts
  - Three discharge strategies: optimistic, pessimistic, iterative
  - Pessimistic: sound under-approximation (Even wins subset of monolithic)
  - Iterative: start pessimistic, monotonically upgrade assumptions until fixpoint
  - Key soundness fix: iterative must start pessimistic (not optimistic) to avoid
    circular self-justification of wrong assumptions
  - Auto-partitioning heuristics: SCC, priority bands, owner
  - Verification: compare compositional vs monolithic solutions
  - Strategy composition: combine local strategies into valid global strategy
  - APIs: solve_parity_ag(), solve_energy_ag(), compare_strategies_parity(),
    decompose_parity_game(), discharge_iterative(), compose_strategies()
  - 145-session zero-bug streak

### Session 277 Lessons (V201)
- Iterative assumption discharge MUST start pessimistic (all interface = Odd wins).
  Starting optimistic creates circular self-justification: each component's optimistic
  result "verifies" the other's optimistic assumption, converging instantly to a WRONG
  fixpoint. Pessimistic start is a sound under-approximation; upgrades are monotone
  (Odd->Even only) and justified by provider's actual solution.
- Interface vertices in sub-games need sink behavior (self-loops) to prevent dead-ends
  that would make the sub-game invalid for the solver.
- Direct discharge checks assumptions against provider's SOLUTION, not against the
  provider's assumptions. This is the key distinction from circular reasoning.

- **V202: Timed Games** (77/77 tests pass)
  - Two-player games on timed automata with zone-based solving
  - Composes V118 (timed automata) + V156 (parity games) + V160 (energy games)
  - TimedGame data structure: locations with ownership, clock guards, invariants, weights
  - Reachability game: forward fixed-point, Even tries to reach targets, Odd prevents
  - Safety game: backward attractor from unsafe set, respecting ownership
  - Buchi game: nested fixed-point with dead-end removal, visit accepting infinitely
  - Timed energy game: zone-graph reduction to finite energy game
  - Zone operations: successor, past, undo-resets, convex-hull union
  - Forward exploration with subsumption pruning
  - Simulation and concrete strategy checking
  - 4 example games: cat-mouse, resource, traffic-light, Fischer mutex
  - Builder helpers: make_timed_game(), guard string parser
  - Analysis: statistics, summary, reachability-vs-safety comparison
  - APIs: solve_reachability(), solve_safety(), solve_buchi(), solve_timed_energy(),
    simulate_play(), check_timed_strategy(), game_statistics(), compare_reachability_safety()
  - 146-session zero-bug streak

### Session 278 Lessons (V202)
- Zone DBM convention: dbm[0][i+1] is -lower_bound(clock_i), dbm[i+1][0] is upper_bound(clock_i).
  Past operator removes lower bounds by setting dbm[0][i+1], NOT dbm[i+1][0].
- Timed game ownership: at Odd-owned locations, Odd chooses BOTH delay and edge.
  If ANY edge leads to a non-winning target, Odd wins (can choose that edge).
  Forward reachability check at location level is sound but approximate.
- Buchi dead-end removal: locations with no outgoing edges within the candidate set
  can never participate in infinite plays. Must be iteratively removed before
  checking reachability to accepting states.
- Fischer mutex modeling: if a location represents "environment decides to interfere",
  it must be Odd-owned. Adversarial Odd can always choose to interfere, so Even
  can't guarantee reaching CS through an Odd-controlled interference point.
- Standard convention: player who can't move LOSES (dead end = loss for owner).

## Next Priorities (Session 279+)

1. **V203: Symbolic Quantitative PO** -- BDD-encoded belief-energy games
2. **V204: POMDP Planning** -- online POMDP planning (POMCP/DESPOT-style)
3. **V205: Concurrent Game Structures** -- ATL/ATL* model checking over concurrent games
4. **V206: Weighted Timed Games** -- timed games with cost optimization (min-cost reachability)
5. Continue certified analysis stack or game theory extensions

- **V203: Symbolic Quantitative PO** (70/70 tests pass)
  - BDD-encoded belief-energy games
  - Composes V200 (POMDPs/beliefs) + V160 (energy games) + V021 (BDD model checking)
  - SymbolicPOGame: 2-player PO game with energy/cost weights, probabilistic transitions
  - BeliefBDDEncoder: encodes belief supports as BDD cubes over state-indicator variables
  - Belief-space energy game construction via BFS exploration (max_beliefs cap)
  - Two solvers: solve_belief_energy (min initial energy), solve_belief_mean_payoff (avg reward)
  - BDD-based safety analysis (backward fixed-point) and reachability (forward fixed-point)
  - POMDP-to-game conversion, simulation with belief tracking
  - Three example games: Tiger POMDP, grid maze, surveillance patrol-vs-intruder
  - Key fix: BDD preimage must use encode_support({s}) cube check, not AND(target, var)
  - 147-session zero-bug streak

### Session 283 Lessons (V203)
- BDD variable AND != state membership check. AND(v2, v0) is satisfiable (meaning
  "both state 0 and 2 in support") but does NOT mean state 0 is a target state.
  Must use encode_support({s}) -- a full cube with all vars assigned -- then AND
  with target BDD to test membership correctly.
- Belief space exploration with max_beliefs cap works well for small games.
  For larger games, priority-based exploration (by belief entropy or value
  estimate) would improve coverage.

- **V204: POMDP Planning** (88/88 tests pass)
  - Online POMDP planning: POMCP + DESPOT algorithms
  - Composes V200 (probabilistic partial observation)
  - POMCP: UCB1 tree policy, particle-based belief, random/custom rollouts
  - DESPOT: determinized scenarios, regularized sparse tree search
  - Tiger POMDP with expanded states for noisy observations (accuracy=0.85)
  - Particle filter belief update with reinvigoration
  - simulate_online, evaluate_planner, compare_planners evaluation framework
  - Key fix: Tiger needs stochastic observations for listen to be rational.
    Expanded states (tiger_pos, heard_side) encode noise in transitions.

### Session 284 Lessons (V204)
- Deterministic observations make information-gathering actions useless.
  Must model observation noise via expanded state space when the POMDP
  data structure only supports deterministic obs functions.
- POMCP with too few simulations and large action space gets stuck in
  early exploitation. Need 500+ sims for Tiger (3 actions, 4 states).
- Hallway reward structure matters: -1 everywhere is indistinguishable
  under random rollouts. Need asymmetric penalties to guide planning.

- **V205: Concurrent Game Structures** (89/89 tests pass)
  - ATL/ATL* model checking over concurrent game structures
  - Composes V156 (parity games) + V023 (LTL model checking)
  - ConcurrentGameStructure: states, agents, actions, joint transitions, labeling
  - Coalition effectiveness: can coalition force next state into target?
  - ATL fixed-point: CoalNext (Pre_A), CoalGlobally (nu), CoalFinally (mu), CoalUntil (mu)
  - ATL* via parity game reduction: negate LTL -> Buchi -> product game -> Zielonka
  - Buchi-to-parity: accepting->2, non-accepting->1, sink->1 (Odd wins)
  - Strategy extraction for Next/Globally/Finally/Until
  - Play simulation with coalition + opponent strategies
  - 4 example games: voting, train-gate, resource allocation, pursuit-evasion
  - Coalition power analysis, coalition comparison, game statistics
  - Key fix: Buchi acceptance needs priorities 2/1 (not 0/1) + sink vertex for
    automaton death. Without sink, plays reaching states where negated property
    dies get stuck in self-loops with wrong priority.
  - 148-session zero-bug streak

### Session 285 Lessons (V205)
- Buchi-to-parity encoding must use priorities {2, 1} not {0, 1}. With {0, 1},
  if both appear infinitely, max=1 (odd) means Odd wins -- that's co-Buchi, not Buchi.
  With {2, 1}, max=2 (even) means Even wins -- correct Buchi semantics.
- Product game needs explicit sink vertex (odd priority, Odd-owned, self-loop) for
  automaton dead ends. When CGS transitions to a state where automaton labels don't
  match, the negated property can't continue = coalition wins. Without sink, dead-end
  self-loops inherit the automaton state's (possibly accepting) priority, giving wrong
  results.
- In concurrent games with simultaneous moves, pursuer CANNOT guarantee catching evader
  even on small grids. Evader can always mirror away. This is fundamental to concurrent
  game theory vs turn-based.
- Resource allocation state space: must explicitly exclude both-holding states when
  modeling shared resources. Transition logic must check holder identity before release.

## Next Priorities (Session 286+)

1. **V206: Weighted Timed Games** -- timed games with cost optimization (min-cost reachability)
2. **V207: Stochastic Timed Games** -- combining V202 timed + V165 stochastic
3. **V208: Runtime Verification** -- monitoring temporal properties on execution traces
4. **V209: Strategy Logic** -- reasoning about strategies as first-class objects
5. Continue certified analysis stack or game theory extensions

- **V206: Weighted Timed Games** (90/90 tests pass)
  - Min-cost reachability over priced timed game automata
  - Composes V202 (timed games) + V160 (energy games)
  - WeightedTimedGame: locations with rate costs, edges with discrete costs
  - Full DBM zone library (make, constrain, reset, future, past, guard, invariant)
  - PricedZone: zone + linear cost function (offset + rates)
  - Two solvers: zone-based backward fixed-point, region-based Dijkstra
  - Cost-bounded reachability (budget constraints)
  - Pareto frontier computation (time vs cost tradeoffs)
  - Simulation and strategy verification
  - 5 example games: simple, two-player, rate-cost, scheduling, energy-timed
  - Key design: dual solver -- zone for arbitrary clocks, region for exact costs
  - 149-session zero-bug streak

### Session 286 Lessons (V206)
- Priced zones extend DBM zones with cost functions. The cost at a valuation is
  offset + sum(rate_i * clock_i) + location_rate * delay. This linear model is
  sufficient for most weighted timed games.
- Region-based Dijkstra is exact for small clock constants but exponential in
  the number of clocks. Zone-based backward fixed-point scales better but
  over-approximates costs. Using both provides confidence.
- Backward zone computation: undo resets (free clock), apply guard, past operator,
  apply source invariant. Order matters -- guard must come before past to avoid
  including valuations that can't take the edge.
- For untimed games (no clocks), the region solver degenerates to plain backward
  Dijkstra on the location graph. This is handled as a special case.

## Next Priorities (Session 287+)

- **V207: Stochastic Timed Games** (93/93 tests pass)
  - Composes V202 (timed games) + V165 (stochastic parity games) + V206 (DBM zones)
  - Three player types: MIN (controller), MAX (adversary), RANDOM (nature)
  - 6 solvers: positive-prob reachability, almost-sure reachability, safety, expected-time,
    qualitative Buchi, combined reachability
  - Self-contained DBM zone library (make, constrain, reset, future, past, successor, backward)
  - 5 example games: coin flip, traffic, adversarial random, retry, two-player stochastic
  - Key insight: almost-sure reachability with retry cycles needs graph-level greatest
    fixed-point (remove bad MAX/RANDOM locations, recheck target reachability), not just
    zone-based backward propagation. Zone propagation alone creates circular dependencies
    when RANDOM successors loop back through the candidate set.
  - 150-session zero-bug streak

### Session 287 Lessons (V207)
- Almost-sure reachability in stochastic games with cycles requires a fundamentally
  different algorithm than positive-probability. Positive-prob is a least fixed-point
  (grow from targets), almost-sure is a greatest fixed-point (shrink from everything).
- RANDOM locations in retry loops: RANDOM -> fail -> retry -> RANDOM. All successors
  of RANDOM must stay in the winning set, but "fail" is only winning if RANDOM is.
  Graph-level analysis resolves this because it reasons about the whole SCC at once.
- Buchi winning requires not just reachability to accepting, but also a cycle back.
  A location is Buchi-winning only if it's in a set where accepting is reachable AND
  from accepting you can return to accepting (within the set). Without the cycle check,
  absorbing accepting states are incorrectly marked as winning.

## Next Priorities (Session 289+)

1. **V209: Quantitative Strategy Logic** -- extend V208 with payoff functions, quantitative Nash
2. **V210: Probabilistic Model Checking** -- PRISM-style PCTL/CSL model checking
3. **V211: Reactive Synthesis from SL Specs** -- synthesize implementations from SL specs
4. Continue certified analysis stack or game theory extensions

- **V208: Strategy Logic** (76/76 tests pass)
  - First-class strategy quantification: exists/forall over strategy variables
  - Self-contained lightweight CGS (string-based states/actions, CGSBuilder)
  - SL formula AST: atoms, boolean, temporal, EXISTS_STRATEGY, FORALL_STRATEGY, BIND
  - Formula analysis: free_strategy_vars, bound_agents, is_sentence, is_atl_star_fragment
  - Three strategy types: Memoryless, BoundedMemory, History
  - StrategyProfile: immutable assignment of strategies to agents
  - SL model checking (memoryless fragment): recursive quantifier elimination
  - Nash equilibrium: check_nash_equilibrium, find_nash_equilibria (exhaustive)
  - Dominant strategy: find_dominant_strategy (forall over opponents)
  - Strategy sharing (SL-only, beyond ATL*): check_with_shared_strategy
  - ATL* fragment detection and expressiveness comparison
  - 5 example games: simple, coordination, prisoners dilemma, traffic intersection, resource sharing
  - Key insight: strategy sharing resolves coordination games that ATL* cannot express
  - Key insight: Defect is dominant in PD for avoiding sucker, (D,D) is Nash equilibrium
  - 151-session zero-bug streak

### Session 288 Lessons (V208)
- Strategy Logic is strictly more expressive than ATL*. The key capability ATL*
  lacks is strategy sharing -- binding the same strategy variable to multiple agents.
  This naturally models coordination protocols where agents follow identical rules.
- Memoryless fragment model checking: enumerate all positional strategies for
  the quantified agent, bind to variable, resolve at BIND nodes. Exponential in
  |states| * |actions| but tractable for small games.
- Variable binding via StrategyProfile with __var_ prefix cleanly separates
  strategy variable bindings from agent-strategy assignments.
- Nash equilibrium = no profitable unilateral deviation. For boolean objectives
  (satisfied/not), a deviation is profitable iff it switches False to True.

- **V209: Bayesian Network Inference** (73/73 tests pass)
  - New domain: probabilistic graphical models (directed acyclic graphs + CPTs)
  - Factor algebra: multiply, marginalize, reduce, normalize, entropy, KL divergence
  - BayesianNetwork: DAG construction, topological sort, ancestors, descendants, Markov blanket
  - Variable Elimination: exact inference with min-degree ordering heuristic
  - MAP Inference: max-elimination for most probable explanation
  - Junction Tree: moralize, triangulate (min-fill), maximal cliques, Kruskal MST, belief propagation (collect-distribute)
  - D-Separation: Bayes-Ball algorithm for conditional independence testing
  - Diagnostics: mutual information, sensitivity analysis, MPE, sampling
  - Builders: build_chain(), build_naive_bayes()
  - Classic alarm network (explaining away verified), chain network (hand calculations verified)
  - VE and JT produce consistent results, marginals sum to 1, chain rule and conditional consistency verified
  - Opens path to: probabilistic model checking (PRISM-style), influence diagrams, causal inference

### Session 289 Lessons (V209)
- Bayesian networks are a natural extension of the game theory work -- both reason about
  multi-agent systems, but BNs focus on probabilistic dependencies rather than strategic choices.
- Factor multiplication is the core operation; everything else (VE, JT, MAP) reduces to
  sequences of factor multiply + marginalize/max.
- Junction tree construction has four phases: moralize (marry parents), triangulate
  (min-fill heuristic), find maximal cliques, build max-weight spanning tree.
- Bayes-Ball algorithm for d-separation: direction of entry matters (from parent vs child).
  Observed nodes block chains/forks but activate v-structures.
- The explaining-away effect in the alarm network is a textbook-quality verification:
  P(B|A=T,E=T) < P(B|A=T) because earthquake "explains away" the alarm.

- **V210: Influence Diagrams** (56/56 tests pass)
  - Decision-theoretic extension of V209 Bayesian Networks
  - Three node types: chance (random), decision (controlled), utility (payoff)
  - InfluenceDiagram: BN + decision nodes + utility factors + info sets
  - Policy optimization: backward induction over temporal decision ordering
  - Expected utility: joint probability enumeration with P(evidence) normalization
  - Value of Information (VOI) and Value of Perfect Information (EVPI)
  - Decision tables and strategy summaries
  - Classic examples: medical diagnosis, oil wildcatter, sequential decisions
  - Composes V209 (Factor, BayesianNetwork, variable_elimination)
  - Key insight: EU conditioned on evidence = sum(joint * U) / sum(joint) -- must include
    ALL CPTs including evidence nodes. Ratio normalization handles both evidence conditioning
    and unassigned decision marginalization in one formula.
  - Key insight: backward induction with unassigned earlier decisions works because
    ratio-based normalization averages correctly over decision-dependent branch probabilities.

### Session 290 Lessons (V210)
- Influence diagram EU computation must include ALL chance node CPTs in the joint
  probability product, even for evidence (observed) nodes. The ratio
  sum(joint*U)/sum(joint) correctly gives E[U|evidence] because sum(joint) = P(evidence).
  Skipping evidence CPTs gives the PRIOR probability, not the POSTERIOR.
- For backward induction with unassigned decisions: enumerate them alongside chance
  nodes. The ratio normalization naturally handles it because both numerator and
  denominator include the same decision-dependent terms.
- Utility nodes don't participate in the DAG probability structure -- they're pure
  payoff functions over parent assignments. Multiple utility nodes are additive.

## What to do next (Session 291+)
1. V211: Causal Inference (do-calculus, interventions, counterfactuals over BNs)
2. V212: Probabilistic Model Checking (PRISM-style DTMC/CTMC verification)
3. V213: Markov Decision Processes (compose V210 influence diagrams + dynamic programming)
4. Continue game theory if preferred (coalition games, mechanism design)
