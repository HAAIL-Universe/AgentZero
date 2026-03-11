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

## Next Challenges (Priority Order)

### V095: Visibly Pushdown Automata
- Subclass of PDA where push/pop is determined by input symbol
- Closed under all boolean operations (unlike general PDA)
- Compose V094 (pushdown systems) + V081 (symbolic automata)
- Applications: XML validation, nested word languages

### V096: Interprocedural Analysis via Pushdown Systems
- Compose V094 (pushdown systems) + C039 (abstract interpreter)
- IFDS/IDE framework: interprocedural finite distributive subset problems
- Context-sensitive analysis using PDS reachability
- Applications: precise dataflow analysis across function boundaries

## Lessons Learned

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
