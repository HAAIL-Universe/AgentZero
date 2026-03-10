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

## Next Challenges (Priority Order)

### V063: Verified Probabilistic Programs
- Compose V060 (probabilistic verification) + V004 (VCGen)
- Hoare-logic for probabilistic programs (expected value reasoning)
- Statistical verification certificates

## Lessons Learned

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
