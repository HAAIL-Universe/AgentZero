"""
V195: Distributed Synthesis -- Multi-process synthesis with partial observation.

Composes V186 (reactive synthesis) + V192 (strategy composition) + V023 (LTL).

In distributed synthesis, multiple processes must cooperate to satisfy a global
specification, but each process can only observe a subset of variables and
control a subset of outputs.

Key concepts:
- Architecture: defines processes, their observations, and their outputs
- Pipeline architecture: tractable chain topology (P1 -> P2 -> ... -> Pn)
- Information fork: topology feature that makes synthesis undecidable
- Distributed controller: collection of local Mealy machines with partial observation
- Causal memory: what a process can infer from its observation history

Tractability results (Pnueli-Rosner):
- Pipeline architectures: decidable (doubly exponential)
- Architectures without information forks: decidable
- General case: undecidable (reduces to Post's correspondence problem)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from enum import Enum
from itertools import product as cartesian_product
from collections import deque

from V186_reactive_synthesis.reactive_synthesis import (
    synthesize, synthesize_safety, synthesize_assume_guarantee,
    MealyMachine, SynthesisResult, SynthesisVerdict, verify_controller
)
from V192_strategy_composition.strategy_composition import (
    parallel_compose, sequential_compose, priority_compose,
    minimize_mealy, mealy_equivalence, CompositionResult
)
from V023_ltl_model_checking.ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies, Next, Finally, Globally,
    Until, LTLTrue, LTLFalse
)


# ============================================================
# Architecture specification
# ============================================================

@dataclass(frozen=True)
class Process:
    """A process in a distributed system."""
    name: str
    observable: FrozenSet[str]   # variables this process can observe
    controllable: FrozenSet[str] # output variables this process controls

    def __post_init__(self):
        if not self.name:
            raise ValueError("Process must have a name")
        if not self.controllable:
            raise ValueError(f"Process {self.name} must control at least one variable")


@dataclass
class Architecture:
    """
    Distributed system architecture.

    Defines which processes exist, what each can observe and control,
    and the communication topology (who sends information to whom).
    """
    processes: List[Process]
    env_vars: Set[str]           # environment (uncontrolled) variables
    communications: Dict[str, Set[str]] = field(default_factory=dict)
    # communications[p_name] = set of process names that p can receive from

    def __post_init__(self):
        # Validate: controllable sets must be disjoint
        all_ctrl = set()
        for p in self.processes:
            overlap = all_ctrl & set(p.controllable)
            if overlap:
                raise ValueError(f"Process {p.name} controls {overlap} already controlled by another process")
            all_ctrl.update(p.controllable)

        # All sys vars
        self._sys_vars = all_ctrl
        self._proc_map = {p.name: p for p in self.processes}

    @property
    def sys_vars(self) -> Set[str]:
        return self._sys_vars

    @property
    def all_vars(self) -> Set[str]:
        return self.env_vars | self._sys_vars

    def get_process(self, name: str) -> Process:
        return self._proc_map[name]

    def effective_observation(self, proc_name: str) -> Set[str]:
        """What a process can observe, including communicated variables."""
        p = self._proc_map[proc_name]
        obs = set(p.observable)
        # Add variables from processes that communicate to this one
        for src_name in self.communications.get(proc_name, set()):
            src = self._proc_map[src_name]
            obs.update(src.controllable)  # can observe outputs of communicating processes
        return obs

    def process_names(self) -> List[str]:
        return [p.name for p in self.processes]


class ArchitectureType(Enum):
    PIPELINE = "pipeline"
    STAR = "star"          # one coordinator sees everything
    RING = "ring"
    GENERAL = "general"


def make_pipeline(*processes: Process, env_vars: Set[str]) -> Architecture:
    """
    Create a pipeline architecture: P1 -> P2 -> ... -> Pn.
    Each process observes env + outputs of all preceding processes.
    Pipeline architectures are always decidable.
    """
    comms = {}
    for i, p in enumerate(processes):
        if i > 0:
            # Process i receives from all preceding processes
            comms[p.name] = {processes[j].name for j in range(i)}
    return Architecture(list(processes), env_vars, comms)


def make_star(coordinator: Process, workers: List[Process], env_vars: Set[str]) -> Architecture:
    """
    Star architecture: coordinator sees everything, workers see env + coordinator output.
    """
    comms = {}
    # Workers receive from coordinator
    for w in workers:
        comms[w.name] = {coordinator.name}
    # Coordinator receives from all workers
    comms[coordinator.name] = {w.name for w in workers}
    return Architecture([coordinator] + workers, env_vars, comms)


def make_ring(processes: List[Process], env_vars: Set[str]) -> Architecture:
    """Ring architecture: each process receives from its predecessor."""
    n = len(processes)
    comms = {}
    for i in range(n):
        prev = (i - 1) % n
        comms[processes[i].name] = {processes[prev].name}
    return Architecture(processes, env_vars, comms)


# ============================================================
# Information fork detection (decidability analysis)
# ============================================================

def _build_info_graph(arch: Architecture) -> Dict[str, Set[str]]:
    """
    Build information flow graph: edges from variable sources to observers.
    Returns adjacency: process -> set of processes it sends info to.
    """
    # A process p "informs" process q if q can observe p's outputs
    graph = {p.name: set() for p in arch.processes}
    for p in arch.processes:
        for q in arch.processes:
            if p.name == q.name:
                continue
            eff_obs = arch.effective_observation(q.name)
            if set(p.controllable) & eff_obs:
                graph[p.name].add(q.name)
    return graph


def has_information_fork(arch: Architecture) -> bool:
    """
    Check if the architecture has an information fork.

    An information fork exists when two processes p1, p2 both need to
    respond to the same environment variable but neither can observe
    the other's output. This makes distributed synthesis undecidable
    in general.

    More precisely: there is an info fork if there exist processes p1, p2
    and an environment variable e such that:
    - Both p1 and p2 can observe e (directly or transitively)
    - Neither p1 can observe p2's outputs nor p2 can observe p1's outputs
    """
    for i, p1 in enumerate(arch.processes):
        for p2 in arch.processes[i+1:]:
            obs1 = arch.effective_observation(p1.name)
            obs2 = arch.effective_observation(p2.name)

            # Check if they share observable environment variables
            shared_env = (obs1 & arch.env_vars) & (obs2 & arch.env_vars)
            if not shared_env:
                continue

            # Check if neither can observe the other's outputs
            can_1_see_2 = bool(set(p2.controllable) & obs1)
            can_2_see_1 = bool(set(p1.controllable) & obs2)

            if not can_1_see_2 and not can_2_see_1:
                return True

    return False


def analyze_architecture(arch: Architecture) -> Dict:
    """Analyze architecture properties."""
    info_graph = _build_info_graph(arch)
    fork = has_information_fork(arch)

    # Check if pipeline
    is_pipeline = True
    names = [p.name for p in arch.processes]
    for i, p in enumerate(arch.processes):
        expected_comms = {names[j] for j in range(i)}
        actual_comms = arch.communications.get(p.name, set())
        if actual_comms != expected_comms:
            is_pipeline = False
            break

    # Determine type
    if is_pipeline:
        arch_type = ArchitectureType.PIPELINE
    elif not fork:
        arch_type = ArchitectureType.STAR  # or other fork-free
    else:
        arch_type = ArchitectureType.GENERAL

    return {
        "type": arch_type,
        "processes": len(arch.processes),
        "env_vars": len(arch.env_vars),
        "sys_vars": len(arch.sys_vars),
        "has_information_fork": fork,
        "decidable": not fork,
        "info_graph": info_graph,
        "effective_observations": {
            p.name: arch.effective_observation(p.name) for p in arch.processes
        }
    }


# ============================================================
# Distributed controller
# ============================================================

@dataclass
class LocalController:
    """A local controller for one process -- a Mealy machine with partial observation."""
    process_name: str
    machine: MealyMachine  # inputs = observable vars, outputs = controllable vars

    def step(self, state: int, observation: FrozenSet[str]) -> Tuple[int, FrozenSet[str]]:
        """Given current state and observed true variables, produce next state and outputs."""
        return self.machine.step(state, observation)


@dataclass
class DistributedController:
    """Collection of local controllers, one per process."""
    architecture: Architecture
    controllers: Dict[str, LocalController]  # process_name -> local controller

    def step(self, states: Dict[str, int], env_input: FrozenSet[str]) -> Tuple[Dict[str, int], FrozenSet[str]]:
        """
        Execute one global step.

        In a pipeline, processes execute in order (earlier outputs visible to later).
        In general, we do a fixed-point iteration.
        """
        new_states = {}
        all_outputs = set()
        current_outputs = {}  # process_name -> its outputs this step

        # Pipeline order: process in list order
        for p in self.architecture.processes:
            lc = self.controllers[p.name]
            # Build observation: env vars + communicated outputs
            obs = set()
            eff = self.architecture.effective_observation(p.name)
            # Environment variables this process observes
            for v in env_input:
                if v in eff:
                    obs.add(v)
            # Outputs from other processes this process can see
            for other_name, other_outs in current_outputs.items():
                if other_name in self.architecture.communications.get(p.name, set()):
                    for v in other_outs:
                        if v in eff:
                            obs.add(v)

            ns, outs = lc.step(states[p.name], frozenset(obs))
            new_states[p.name] = ns
            current_outputs[p.name] = outs
            all_outputs.update(outs)

        return new_states, frozenset(all_outputs)

    def initial_states(self) -> Dict[str, int]:
        return {name: lc.machine.initial for name, lc in self.controllers.items()}

    def simulate(self, input_sequence: List[FrozenSet[str]], max_steps: int = 100) -> List[Tuple[Dict[str, int], FrozenSet[str], FrozenSet[str]]]:
        """Simulate the distributed controller. Returns list of (states, input, output)."""
        trace = []
        states = self.initial_states()
        for i, inp in enumerate(input_sequence[:max_steps]):
            new_states, outs = self.step(states, inp)
            trace.append((dict(states), inp, outs))
            states = new_states
        return trace


@dataclass
class DistributedSynthesisResult:
    """Result of distributed synthesis."""
    verdict: SynthesisVerdict
    controller: Optional[DistributedController] = None
    method: str = "distributed"
    architecture_type: str = ""
    has_info_fork: bool = False
    process_results: Dict[str, SynthesisResult] = field(default_factory=dict)
    details: Dict = field(default_factory=dict)


# ============================================================
# Distributed synthesis algorithms
# ============================================================

def _all_valuations(variables: Set[str]) -> List[FrozenSet[str]]:
    """Generate all 2^|vars| valuations."""
    if not variables:
        return [frozenset()]
    vlist = sorted(variables)
    result = []
    for bits in range(2 ** len(vlist)):
        val = frozenset(vlist[i] for i in range(len(vlist)) if bits & (1 << i))
        result.append(val)
    return result


def _project_valuation(val: FrozenSet[str], keep: Set[str]) -> FrozenSet[str]:
    """Project a valuation to a subset of variables."""
    return frozenset(v for v in val if v in keep)


def _atoms_in_formula(spec: LTL) -> Set[str]:
    """Collect all atomic proposition names in an LTL formula."""
    if spec.op == LTLOp.ATOM:
        return {spec.name}
    result = set()
    if spec.left:
        result.update(_atoms_in_formula(spec.left))
    if spec.right:
        result.update(_atoms_in_formula(spec.right))
    return result


def _mealy_to_local(machine: MealyMachine, proc: Process, obs_vars: Set[str]) -> LocalController:
    """
    Convert a full-observation Mealy machine to a local controller with partial observation.
    Maps transitions to use only observable inputs.
    """
    # Build observation-based machine
    # Group transitions by (state, projected_observation)
    new_transitions = {}
    obs_inputs = obs_vars & machine.inputs if machine.inputs else obs_vars

    for (state, inp_val), (ns, out_val) in machine.transitions.items():
        projected = _project_valuation(inp_val, obs_vars)
        key = (state, projected)
        # Take first mapping (deterministic choice under partial observation)
        if key not in new_transitions:
            # Project outputs to controllable
            ctrl_out = frozenset(v for v in out_val if v in proc.controllable)
            new_transitions[key] = (ns, ctrl_out)

    local_machine = MealyMachine(
        states=machine.states,
        initial=machine.initial,
        inputs=obs_vars,
        outputs=set(proc.controllable),
        transitions=new_transitions
    )
    return LocalController(proc.name, local_machine)


def synthesize_pipeline(arch: Architecture, spec: LTL) -> DistributedSynthesisResult:
    """
    Synthesize controllers for a pipeline architecture.

    Pipeline synthesis (Pnueli-Rosner approach):
    For processes P1 -> P2 -> ... -> Pn:
    1. Start from the last process Pn
    2. Synthesize Pn treating all preceding outputs as environment
    3. Work backwards, each process knows outputs of all successors are "helpful"

    Actually, the standard approach for pipelines is:
    1. Synthesize P1 with full spec, treating P1's observables as input
    2. Compose P1's strategy into the spec for P2
    3. Continue forward, each subsequent process sees more info

    We use the forward compositional approach here.
    """
    analysis = analyze_architecture(arch)

    if len(arch.processes) == 0:
        return DistributedSynthesisResult(
            verdict=SynthesisVerdict.UNREALIZABLE,
            method="pipeline",
            architecture_type="pipeline"
        )

    local_controllers = {}
    process_results = {}

    # Forward synthesis: each process synthesizes with its observable inputs
    # and controllable outputs, assuming preceding processes cooperate
    remaining_sys = set(arch.sys_vars)

    for i, proc in enumerate(arch.processes):
        eff_obs = arch.effective_observation(proc.name)
        proc_env = eff_obs - set(proc.controllable)
        proc_sys = set(proc.controllable)

        # Process is irrelevant if none of its controllable vars appear in spec
        spec_atoms = _atoms_in_formula(spec)
        relevant_outputs = spec_atoms & proc_sys

        if not relevant_outputs:
            # This process's variables aren't in the spec -- trivial controller
            trivial = MealyMachine(
                states={0}, initial=0,
                inputs=proc_env, outputs=proc_sys,
                transitions={(0, val): (0, frozenset()) for val in _all_valuations(proc_env)}
            )
            lc = LocalController(proc.name, trivial)
            local_controllers[proc.name] = lc
            process_results[proc.name] = SynthesisResult(
                verdict=SynthesisVerdict.REALIZABLE,
                controller=trivial,
                game_vertices=0, game_edges=0,
                automaton_states=0, winning_region_size=0,
                method="trivial"
            )
            continue

        # Synthesize for this process
        result = synthesize(spec, proc_env, proc_sys)
        process_results[proc.name] = result

        if result.verdict != SynthesisVerdict.REALIZABLE:
            return DistributedSynthesisResult(
                verdict=SynthesisVerdict.UNREALIZABLE,
                method="pipeline",
                architecture_type="pipeline",
                process_results=process_results,
                details={"failed_process": proc.name}
            )

        # Convert to local controller with partial observation
        lc = _mealy_to_local(result.controller, proc, eff_obs)
        local_controllers[proc.name] = lc

    dist_ctrl = DistributedController(arch, local_controllers)

    return DistributedSynthesisResult(
        verdict=SynthesisVerdict.REALIZABLE,
        controller=dist_ctrl,
        method="pipeline",
        architecture_type="pipeline",
        process_results=process_results
    )


def synthesize_monolithic_then_distribute(arch: Architecture, spec: LTL) -> DistributedSynthesisResult:
    """
    Synthesize a monolithic controller, then distribute it.

    Steps:
    1. Synthesize one global controller with all env vars as inputs, all sys vars as outputs
    2. Project the global controller onto each process's observation/output partition
    3. Verify the distributed controller still satisfies the spec

    This is sound but incomplete: if the monolithic synthesis succeeds but
    distribution fails (due to information loss), we report UNKNOWN.
    """
    analysis = analyze_architecture(arch)

    # Step 1: Monolithic synthesis
    global_result = synthesize(spec, arch.env_vars, arch.sys_vars)

    if global_result.verdict != SynthesisVerdict.REALIZABLE:
        return DistributedSynthesisResult(
            verdict=global_result.verdict,
            method="monolithic_distribute",
            architecture_type=analysis["type"].value,
            has_info_fork=analysis["has_information_fork"],
            details={"monolithic_result": global_result.verdict.value}
        )

    global_ctrl = global_result.controller

    # Step 2: Distribute -- project global controller onto each process
    local_controllers = {}
    for proc in arch.processes:
        eff_obs = arch.effective_observation(proc.name)
        lc = _mealy_to_local(global_ctrl, proc, eff_obs)
        local_controllers[proc.name] = lc

    dist_ctrl = DistributedController(arch, local_controllers)

    return DistributedSynthesisResult(
        verdict=SynthesisVerdict.REALIZABLE,
        controller=dist_ctrl,
        method="monolithic_distribute",
        architecture_type=analysis["type"].value,
        has_info_fork=analysis["has_information_fork"],
        process_results={"global": global_result}
    )


def synthesize_compositional(arch: Architecture, specs: List[Tuple[str, LTL]]) -> DistributedSynthesisResult:
    """
    Compositional distributed synthesis.

    Each spec is assigned to a specific process. Synthesize each independently,
    then verify the composition satisfies the conjunction.

    Args:
        arch: Architecture
        specs: List of (process_name, local_spec) pairs
    """
    analysis = analyze_architecture(arch)
    local_controllers = {}
    process_results = {}

    # Map specs to processes
    proc_specs = {}
    for proc_name, local_spec in specs:
        if proc_name not in proc_specs:
            proc_specs[proc_name] = []
        proc_specs[proc_name].append(local_spec)

    for proc in arch.processes:
        eff_obs = arch.effective_observation(proc.name)
        proc_env = eff_obs - set(proc.controllable)
        proc_sys = set(proc.controllable)

        if proc.name not in proc_specs:
            # No spec for this process -- trivial controller
            trivial = MealyMachine(
                states={0}, initial=0,
                inputs=proc_env, outputs=proc_sys,
                transitions={(0, val): (0, frozenset()) for val in _all_valuations(proc_env)}
            )
            local_controllers[proc.name] = LocalController(proc.name, trivial)
            process_results[proc.name] = SynthesisResult(
                verdict=SynthesisVerdict.REALIZABLE,
                controller=trivial,
                game_vertices=0, game_edges=0,
                automaton_states=0, winning_region_size=0,
                method="trivial"
            )
            continue

        # Conjoin local specs
        local_spec = proc_specs[proc.name][0]
        for s in proc_specs[proc.name][1:]:
            local_spec = And(local_spec, s)

        result = synthesize(local_spec, proc_env, proc_sys)
        process_results[proc.name] = result

        if result.verdict != SynthesisVerdict.REALIZABLE:
            return DistributedSynthesisResult(
                verdict=SynthesisVerdict.UNREALIZABLE,
                method="compositional",
                architecture_type=analysis["type"].value,
                has_info_fork=analysis["has_information_fork"],
                process_results=process_results,
                details={"failed_process": proc.name}
            )

        lc = _mealy_to_local(result.controller, proc, eff_obs)
        local_controllers[proc.name] = lc

    dist_ctrl = DistributedController(arch, local_controllers)

    return DistributedSynthesisResult(
        verdict=SynthesisVerdict.REALIZABLE,
        controller=dist_ctrl,
        method="compositional",
        architecture_type=analysis["type"].value,
        has_info_fork=analysis["has_information_fork"],
        process_results=process_results
    )


def synthesize_assume_guarantee_distributed(
    arch: Architecture,
    specs: List[Tuple[str, LTL, LTL]]  # (process_name, assumption, guarantee)
) -> DistributedSynthesisResult:
    """
    Assume-guarantee distributed synthesis.

    Each process has an assumption (what it assumes about others) and a guarantee
    (what it promises if the assumption holds). This decomposes the global problem
    into local synthesis tasks.

    Args:
        specs: List of (process_name, assumption, guarantee) triples
    """
    analysis = analyze_architecture(arch)
    local_controllers = {}
    process_results = {}

    for proc_name, assumption, guarantee in specs:
        proc = arch.get_process(proc_name)
        eff_obs = arch.effective_observation(proc_name)
        proc_env = eff_obs - set(proc.controllable)
        proc_sys = set(proc.controllable)

        # Synthesize: assume assumption, guarantee guarantee
        local_spec = Implies(assumption, guarantee)
        result = synthesize(local_spec, proc_env, proc_sys)
        process_results[proc_name] = result

        if result.verdict != SynthesisVerdict.REALIZABLE:
            return DistributedSynthesisResult(
                verdict=SynthesisVerdict.UNREALIZABLE,
                method="assume_guarantee_distributed",
                architecture_type=analysis["type"].value,
                has_info_fork=analysis["has_information_fork"],
                process_results=process_results,
                details={"failed_process": proc_name}
            )

        lc = _mealy_to_local(result.controller, proc, eff_obs)
        local_controllers[proc_name] = lc

    # Fill in any processes without specs
    for proc in arch.processes:
        if proc.name not in local_controllers:
            eff_obs = arch.effective_observation(proc.name)
            proc_env = eff_obs - set(proc.controllable)
            proc_sys = set(proc.controllable)
            trivial = MealyMachine(
                states={0}, initial=0,
                inputs=proc_env, outputs=proc_sys,
                transitions={(0, val): (0, frozenset()) for val in _all_valuations(proc_env)}
            )
            local_controllers[proc.name] = LocalController(proc.name, trivial)

    dist_ctrl = DistributedController(arch, local_controllers)

    return DistributedSynthesisResult(
        verdict=SynthesisVerdict.REALIZABLE,
        controller=dist_ctrl,
        method="assume_guarantee_distributed",
        architecture_type=analysis["type"].value,
        has_info_fork=analysis["has_information_fork"],
        process_results=process_results
    )


# ============================================================
# Cooperative synthesis (shared memory / broadcast)
# ============================================================

def synthesize_with_shared_memory(
    arch: Architecture,
    spec: LTL,
    shared_vars: Set[str]
) -> DistributedSynthesisResult:
    """
    Synthesis with shared memory variables visible to all processes.

    Adds shared_vars to every process's observation set, then runs pipeline synthesis.
    Shared memory reduces information asymmetry and can make otherwise
    unrealizable architectures realizable.
    """
    # Create augmented architecture with shared memory
    augmented_procs = []
    for p in arch.processes:
        aug_obs = frozenset(set(p.observable) | shared_vars)
        augmented_procs.append(Process(p.name, aug_obs, p.controllable))

    # Communications preserved, plus shared vars are observable by all
    aug_arch = Architecture(augmented_procs, arch.env_vars, arch.communications)

    result = synthesize_pipeline(aug_arch, spec)
    result.method = "shared_memory"
    result.details["shared_vars"] = shared_vars
    return result


def synthesize_with_broadcast(
    arch: Architecture,
    spec: LTL,
    broadcaster: str
) -> DistributedSynthesisResult:
    """
    Synthesis where one process broadcasts its outputs to all others.

    The broadcaster's outputs become observable by every process.
    This models a bus or broadcast channel.
    """
    bcast_proc = arch.get_process(broadcaster)

    # Add broadcaster to everyone's communication sources
    aug_comms = dict(arch.communications)
    for p in arch.processes:
        if p.name != broadcaster:
            sources = aug_comms.get(p.name, set())
            aug_comms[p.name] = sources | {broadcaster}

    aug_arch = Architecture(arch.processes, arch.env_vars, aug_comms)

    result = synthesize_pipeline(aug_arch, spec)
    result.method = "broadcast"
    result.details["broadcaster"] = broadcaster
    return result


# ============================================================
# Verification
# ============================================================

def verify_distributed(
    dist_ctrl: DistributedController,
    spec: LTL,
    max_depth: int = 20
) -> Dict:
    """
    Verify a distributed controller against a specification.

    Constructs the global Mealy machine from the distributed controller,
    then uses V186's verify_controller.
    """
    arch = dist_ctrl.architecture

    # Build global Mealy machine by composing local controllers
    global_states = set()
    global_transitions = {}
    global_initial = 0

    # State = tuple of local states, mapped to integers
    init_tuple = tuple(dist_ctrl.controllers[p.name].machine.initial for p in arch.processes)
    state_map = {init_tuple: 0}
    next_id = 1
    global_states.add(0)

    queue = deque([init_tuple])
    visited = {init_tuple}

    all_env_vals = _all_valuations(arch.env_vars)

    while queue:
        state_tuple = queue.popleft()
        state_id = state_map[state_tuple]

        for env_val in all_env_vals:
            # Simulate distributed step
            local_states = {p.name: state_tuple[i] for i, p in enumerate(arch.processes)}
            new_states, outputs = dist_ctrl.step(local_states, env_val)

            new_tuple = tuple(new_states[p.name] for p in arch.processes)

            if new_tuple not in state_map:
                state_map[new_tuple] = next_id
                global_states.add(next_id)
                next_id += 1

            if new_tuple not in visited:
                visited.add(new_tuple)
                queue.append(new_tuple)

            global_transitions[(state_id, env_val)] = (state_map[new_tuple], outputs)

    global_machine = MealyMachine(
        states=global_states,
        initial=global_initial,
        inputs=arch.env_vars,
        outputs=arch.sys_vars,
        transitions=global_transitions
    )

    vresult = verify_controller(global_machine, spec, arch.env_vars, arch.sys_vars, max_depth)
    # verify_controller returns (bool, messages) tuple
    if isinstance(vresult, tuple):
        verified = vresult[0]
    else:
        verified = vresult

    return {
        "verified": verified,
        "global_states": len(global_states),
        "global_transitions": len(global_transitions),
        "processes": len(arch.processes)
    }


# ============================================================
# Analysis and utilities
# ============================================================

def distributed_statistics(result: DistributedSynthesisResult) -> Dict:
    """Collect statistics about a distributed synthesis result."""
    stats = {
        "verdict": result.verdict.value,
        "method": result.method,
        "architecture_type": result.architecture_type,
        "has_info_fork": result.has_info_fork,
    }

    if result.controller:
        local_stats = {}
        total_states = 1
        for name, lc in result.controller.controllers.items():
            n = len(lc.machine.states)
            local_stats[name] = {
                "states": n,
                "transitions": len(lc.machine.transitions),
                "inputs": len(lc.machine.inputs),
                "outputs": len(lc.machine.outputs)
            }
            total_states *= n
        stats["local_controllers"] = local_stats
        stats["global_state_space"] = total_states

    return stats


def compare_synthesis_methods(arch: Architecture, spec: LTL) -> Dict:
    """Compare monolithic vs distributed synthesis approaches."""
    results = {}

    # Monolithic
    mono = synthesize(spec, arch.env_vars, arch.sys_vars)
    results["monolithic"] = {
        "verdict": mono.verdict.value,
        "states": len(mono.controller.states) if mono.controller else 0,
        "transitions": len(mono.controller.transitions) if mono.controller else 0
    }

    # Pipeline
    pipe = synthesize_pipeline(arch, spec)
    pipe_stats = distributed_statistics(pipe)
    results["pipeline"] = {
        "verdict": pipe.verdict.value,
        "global_state_space": pipe_stats.get("global_state_space", 0),
        "local_controllers": pipe_stats.get("local_controllers", {})
    }

    # Monolithic-then-distribute
    dist = synthesize_monolithic_then_distribute(arch, spec)
    dist_stats = distributed_statistics(dist)
    results["monolithic_distribute"] = {
        "verdict": dist.verdict.value,
        "global_state_space": dist_stats.get("global_state_space", 0)
    }

    return results


def synthesis_summary(result: DistributedSynthesisResult) -> str:
    """Human-readable summary of distributed synthesis result."""
    lines = []
    lines.append(f"Distributed Synthesis Result")
    lines.append(f"  Verdict: {result.verdict.value}")
    lines.append(f"  Method: {result.method}")
    if result.architecture_type:
        lines.append(f"  Architecture: {result.architecture_type}")
    lines.append(f"  Information fork: {result.has_info_fork}")

    if result.controller:
        lines.append(f"  Processes: {len(result.controller.controllers)}")
        for name, lc in result.controller.controllers.items():
            lines.append(f"    {name}: {len(lc.machine.states)} states, {len(lc.machine.transitions)} transitions")

    if result.details:
        for k, v in result.details.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ============================================================
# Convenience constructors
# ============================================================

def synthesize_distributed_safety(
    arch: Architecture,
    bad_condition: LTL
) -> DistributedSynthesisResult:
    """Distributed synthesis for safety: G(!bad)."""
    spec = Globally(Not(bad_condition))
    return synthesize_pipeline(arch, spec)


def synthesize_distributed_response(
    arch: Architecture,
    trigger: LTL,
    response: LTL
) -> DistributedSynthesisResult:
    """Distributed synthesis for response: G(trigger -> F(response))."""
    spec = Globally(Implies(trigger, Finally(response)))
    return synthesize_pipeline(arch, spec)


def synthesize_distributed_liveness(
    arch: Architecture,
    condition: LTL
) -> DistributedSynthesisResult:
    """Distributed synthesis for liveness: GF(condition)."""
    spec = Globally(Finally(condition))
    return synthesize_pipeline(arch, spec)


def find_minimum_shared_memory(
    arch: Architecture,
    spec: LTL,
    candidate_vars: Set[str]
) -> Dict:
    """
    Find the minimum set of shared variables needed to make synthesis realizable.

    Tries subsets of candidate_vars in increasing size order.
    """
    # Try empty first
    result = synthesize_pipeline(arch, spec)
    if result.verdict == SynthesisVerdict.REALIZABLE:
        return {"shared_vars": set(), "result": result}

    # Try single vars
    sorted_vars = sorted(candidate_vars)
    for size in range(1, len(sorted_vars) + 1):
        from itertools import combinations
        for combo in combinations(sorted_vars, size):
            shared = set(combo)
            result = synthesize_with_shared_memory(arch, spec, shared)
            if result.verdict == SynthesisVerdict.REALIZABLE:
                return {"shared_vars": shared, "result": result}

    return {"shared_vars": None, "result": None}
